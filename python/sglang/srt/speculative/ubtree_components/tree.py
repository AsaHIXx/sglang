# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""GPU-resident batched UBTree construction and SGLang tree metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


@dataclass
class UBTreeBatch:
    tokens: torch.Tensor
    parent_ids: torch.Tensor
    depths: torch.Tensor
    path_scores: torch.Tensor
    visibility: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    # Optional graph-owned pre-verify outputs.  They are populated only when
    # proposal construction is captured as the tail of the full draft graph.
    verify_positions: torch.Tensor | None = None
    preverify_staged: bool = False

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.tokens.shape[1])


def compact_tree_values_to_front(
    values: torch.Tensor,
    accept_index: torch.Tensor,
    *,
    batch_size: int,
    node_count: int,
) -> torch.Tensor:
    """Move each accepted path to the front without cloning contiguous input."""

    width = accept_index.shape[1]
    safe = accept_index.to(torch.int64).clamp_min(0).reshape(-1)
    gathered = values[safe]
    # Advanced indexing materializes ``gathered`` before the write, so source
    # rows cannot be overwritten before they are read. Both runtime callers
    # discard the original tree ordering after compaction.
    output = values if values.is_contiguous() else values.contiguous()
    output.view(batch_size, node_count, *values.shape[1:])[:, :width].copy_(
        gathered.view(batch_size, width, *values.shape[1:])
    )
    return output


@triton.jit
def _build_tree_metadata_kernel(
    parent_ids,
    visibility,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    parent_stride_batch: tl.constexpr,
    visibility_stride_batch: tl.constexpr,
    visibility_stride_query: tl.constexpr,
    num_nodes: tl.constexpr,
    max_depth: tl.constexpr,
    block_size: tl.constexpr,
):
    program = tl.program_id(0)
    batch_index = program // num_nodes
    node = program % num_nodes
    keys = tl.arange(0, block_size)
    valid = keys < num_nodes
    parent_row = parent_ids + batch_index * parent_stride_batch

    # One program owns one tree row. Traverse the selected node's ancestor
    # chain for every candidate key, while the same lanes also find its first
    # child and next sibling. This replaces the per-node gather/copy/scatter
    # sequence with one graph node for all visibility and retrieval metadata.
    current = tl.full((block_size,), node, tl.int64)
    visible = valid & (keys == node)
    for _ in range(max_depth):
        active = current > 0
        safe_current = tl.where(active, current, 0)
        current = tl.load(parent_row + safe_current, mask=active, other=-1)
        visible |= active & (keys == current)

    output_row = (
        visibility
        + batch_index * visibility_stride_batch
        + node * visibility_stride_query
    )
    tl.store(output_row + keys, visible, mask=valid)

    candidate_parents = tl.load(parent_row + keys, mask=valid, other=-2)
    first_child = tl.min(
        tl.where(valid & (candidate_parents == node), keys, num_nodes), axis=0
    )
    node_parent = tl.load(parent_row + node)
    next_sibling = tl.min(
        tl.where(
            valid & (keys > node) & (candidate_parents == node_parent),
            keys,
            num_nodes,
        ),
        axis=0,
    )
    flat_index = batch_index * num_nodes + node
    tl.store(retrieve_index + flat_index, flat_index)
    tl.store(
        retrieve_next_token + flat_index,
        tl.where(first_child < num_nodes, first_child, -1),
    )
    tl.store(
        retrieve_next_sibling + flat_index,
        tl.where((node > 0) & (next_sibling < num_nodes), next_sibling, -1),
    )


def _tree_metadata(
    parent_ids: torch.Tensor, *, max_depth: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build visibility and linked-tree metadata in one CUDA kernel."""

    batch, nodes = parent_ids.shape
    if parent_ids.is_cuda:
        visibility = torch.empty(
            (batch, nodes, nodes), dtype=torch.bool, device=parent_ids.device
        )
        retrieve_index = torch.empty_like(parent_ids)
        next_token = torch.empty_like(parent_ids)
        next_sibling = torch.empty_like(parent_ids)
        block_size = triton.next_power_of_2(nodes)
        _build_tree_metadata_kernel[(batch * nodes,)](
            parent_ids,
            visibility,
            retrieve_index,
            next_token,
            next_sibling,
            parent_stride_batch=parent_ids.stride(0),
            visibility_stride_batch=visibility.stride(0),
            visibility_stride_query=visibility.stride(1),
            num_nodes=nodes,
            max_depth=max_depth,
            block_size=block_size,
        )
        return visibility, retrieve_index, next_token, next_sibling

    visibility = _visibility(parent_ids)
    retrieve_index, next_token, next_sibling = _retrieval_links(parent_ids)
    return visibility, retrieve_index, next_token, next_sibling


def _visibility(parent_ids: torch.Tensor) -> torch.Tensor:
    batch, nodes = parent_ids.shape
    result = torch.zeros(
        (batch, nodes, nodes), dtype=torch.bool, device=parent_ids.device
    )
    result[:, 0, 0] = True
    for node in range(1, nodes):
        parent = parent_ids[:, node]
        parent_rows = result.gather(
            1, parent[:, None, None].expand(batch, 1, nodes)
        ).squeeze(1)
        result[:, node].copy_(parent_rows)
        result[:, node, node] = True
    return result


def _retrieval_links(parent_ids: torch.Tensor):
    batch, nodes = parent_ids.shape
    device = parent_ids.device
    next_token = torch.full((batch, nodes), -1, dtype=torch.long, device=device)
    next_sibling = torch.full_like(next_token, -1)
    batch_index = torch.arange(batch, device=device)
    retrieve_index = torch.arange(batch * nodes, dtype=torch.long, device=device).view(
        batch, nodes
    )
    local_node_ids = torch.arange(nodes, dtype=torch.long, device=device)
    # Reverse insertion produces the same ascending sibling order as SGLang's
    # EAGLE tree builder, while never copying parents or node ids from the host.
    # ``scatter_`` is also CUDA Graph capture-safe; advanced-index assignment
    # with the Python ``node`` integer is interpreted as a CPU-to-CUDA copy.
    for node in range(nodes - 1, 0, -1):
        parent = parent_ids[:, node]
        old_first = next_token[batch_index, parent]
        next_sibling[:, node].copy_(old_first)
        next_token.scatter_(
            1,
            parent[:, None],
            local_node_ids[node : node + 1].expand(batch, 1),
        )
    return retrieve_index, next_token, next_sibling


@triton.jit
def _score_frontier_kernel(
    edge_log_probs,
    candidate_indices,
    scores,
    expanded_scores,
    depth: tl.constexpr,
    depth_count: tl.constexpr,
    candidate_k: tl.constexpr,
    width: tl.constexpr,
    max_nodes: tl.constexpr,
    expansion_count: tl.constexpr,
    expansion_block: tl.constexpr,
):
    """Build the flattened frontier scores consumed by exact torch.topk."""

    batch_index = tl.program_id(0)
    offsets = tl.arange(0, expansion_block)
    valid = offsets < expansion_count
    parent_slots = offsets // candidate_k
    successors = offsets % candidate_k

    if depth == 0:
        frontier_candidates = tl.zeros((expansion_block,), tl.int64)
        frontier_scores = tl.where(parent_slots == 0, 0.0, -3.402823466e38)
    else:
        previous_start: tl.constexpr = 1 + (depth - 1) * width
        safe_parent_slots = tl.minimum(parent_slots, width - 1)
        previous_offsets = batch_index * max_nodes + previous_start + safe_parent_slots
        frontier_candidates = tl.load(
            candidate_indices + previous_offsets, mask=valid, other=0
        )
        frontier_scores = tl.load(
            scores + previous_offsets, mask=valid, other=-3.402823466e38
        )

    edge_offsets = (
        (batch_index * depth_count + depth) * candidate_k + frontier_candidates
    ) * candidate_k + successors
    edge_scores = tl.load(
        edge_log_probs + edge_offsets, mask=valid, other=-3.402823466e38
    )
    tl.store(
        expanded_scores + batch_index * expansion_count + offsets,
        frontier_scores + edge_scores,
        mask=valid,
    )


@triton.jit
def _materialize_frontier_kernel(
    candidate_ids,
    chosen_flat,
    chosen_scores,
    tokens,
    parents,
    depths,
    candidate_indices,
    scores,
    depth: tl.constexpr,
    depth_count: tl.constexpr,
    candidate_k: tl.constexpr,
    width: tl.constexpr,
    max_nodes: tl.constexpr,
    rank_block: tl.constexpr,
):
    """Materialize one exact torch.topk result into the supertree buffers."""

    batch_index = tl.program_id(0)
    ranks = tl.arange(0, rank_block)
    valid = ranks < width
    topk_offsets = batch_index * width + ranks
    selected_flat = tl.load(chosen_flat + topk_offsets, mask=valid, other=0)
    selected_scores = tl.load(
        chosen_scores + topk_offsets, mask=valid, other=-3.402823466e38
    )
    parent_slots = selected_flat // candidate_k
    successors = selected_flat % candidate_k
    output_start: tl.constexpr = 1 + depth * width
    output_offsets = batch_index * max_nodes + output_start + ranks
    token_offsets = (batch_index * depth_count + depth) * candidate_k + successors
    if depth == 0:
        parent_nodes = tl.zeros((rank_block,), tl.int64)
    else:
        previous_start: tl.constexpr = 1 + (depth - 1) * width
        parent_nodes = previous_start + parent_slots

    tl.store(
        tokens + output_offsets,
        tl.load(candidate_ids + token_offsets, mask=valid, other=0),
        mask=valid,
    )
    tl.store(parents + output_offsets, parent_nodes, mask=valid)
    tl.store(depths + output_offsets, depth + 1, mask=valid)
    tl.store(candidate_indices + output_offsets, successors, mask=valid)
    tl.store(scores + output_offsets, selected_scores, mask=valid)


@triton.jit
def _select_tree_nodes_kernel(
    super_depths,
    super_scores,
    selected_old_ids,
    depth_bonus: tl.constexpr,
    budget: tl.constexpr,
    max_nodes: tl.constexpr,
    selected_nodes: tl.constexpr,
    selection_block: tl.constexpr,
    chosen_block: tl.constexpr,
):
    """Fuse descending stable global-budget selection and old-id sorting."""

    batch_index = tl.program_id(0)
    offsets = tl.arange(0, selection_block)
    old_nodes = offsets + 1
    valid = old_nodes < max_nodes
    source_offsets = batch_index * max_nodes + old_nodes
    values = tl.load(super_scores + source_offsets, mask=valid, other=-3.402823466e38)
    if depth_bonus != 0.0:
        old_depths = tl.load(super_depths + source_offsets, mask=valid, other=0)
        values += depth_bonus * old_depths.to(tl.float32)

    chosen_slots = tl.arange(0, chosen_block)
    chosen_old = tl.full((chosen_block,), max_nodes, tl.int64)
    lane_ids = tl.arange(0, selection_block)
    # Descending stable argsort[:budget]: equal scores are consumed from the
    # lower old-node id first.
    for rank in tl.static_range(0, budget):
        chosen_lane = tl.argmax(values, axis=0, tie_break_left=True)
        chosen_old = tl.where(chosen_slots == rank, chosen_lane + 1, chosen_old)
        values = tl.where(lane_ids == chosen_lane, -3.402823466e38, values)
    chosen_old = tl.sort(chosen_old, dim=0, descending=False)

    tl.store(selected_old_ids + batch_index * selected_nodes, 0)
    tl.store(
        selected_old_ids + batch_index * selected_nodes + chosen_slots + 1,
        chosen_old,
        mask=chosen_slots < budget,
    )


@triton.jit
def _materialize_selected_tree_kernel(
    anchor_tokens,
    super_tokens,
    super_parents,
    super_depths,
    super_scores,
    selected_old_ids,
    selected_tokens,
    selected_parents,
    selected_depths,
    selected_scores,
    anchor_stride: tl.constexpr,
    selected_nodes: tl.constexpr,
    super_stride: tl.constexpr,
    output_block: tl.constexpr,
    search_steps: tl.constexpr,
):
    """Gather selected rows and remap old parent ids to dense selected ids."""

    batch_index = tl.program_id(0)
    lanes = tl.arange(0, output_block)
    valid = lanes < selected_nodes
    output_offsets = batch_index * selected_nodes + lanes
    selected_old = tl.load(selected_old_ids + output_offsets, mask=valid, other=0)
    # max_nodes is not needed here: every masked selected_old is valid.
    super_offsets = batch_index * super_stride + selected_old
    anchor = tl.load(anchor_tokens + batch_index * anchor_stride)
    token = tl.where(
        lanes == 0,
        anchor,
        tl.load(super_tokens + super_offsets, mask=valid, other=0),
    )
    depth = tl.where(
        lanes == 0,
        0,
        tl.load(super_depths + super_offsets, mask=valid, other=0),
    )
    score = tl.where(
        lanes == 0,
        0.0,
        tl.load(super_scores + super_offsets, mask=valid, other=-3.402823466e38),
    )
    parent_old = tl.where(
        lanes == 0,
        -1,
        tl.load(super_parents + super_offsets, mask=valid, other=-1),
    )

    low = tl.zeros((output_block,), tl.int32)
    high = tl.full((output_block,), selected_nodes, tl.int32)
    for _ in tl.static_range(0, search_steps):
        middle = (low + high) // 2
        safe_middle = tl.minimum(middle, selected_nodes - 1)
        middle_old = tl.load(
            selected_old_ids + batch_index * selected_nodes + safe_middle,
            mask=valid,
            other=selected_nodes,
        )
        move_right = (low < high) & (middle_old < parent_old)
        low = tl.where(move_right, middle + 1, low)
        high = tl.where(move_right, high, middle)
    parent_new = tl.where(lanes == 0, -1, low)

    tl.store(selected_tokens + output_offsets, token, mask=valid)
    tl.store(selected_parents + output_offsets, parent_new, mask=valid)
    tl.store(selected_depths + output_offsets, depth, mask=valid)
    tl.store(selected_scores + output_offsets, score, mask=valid)


def _build_ubtree_batch_fused(
    *,
    anchor_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    edge_log_probs: torch.Tensor,
    budget: int,
    supertree_width: int,
    depth_bonus: float = 0.0,
) -> UBTreeBatch:
    """CUDA implementation fusing score/gather work around exact torch.topk."""

    batch, depth_count, candidate_k = candidate_ids.shape
    width = int(supertree_width)
    max_nodes = 1 + depth_count * width
    selected_nodes = int(budget) + 1
    device = candidate_ids.device

    super_tokens = torch.empty((batch, max_nodes), dtype=torch.long, device=device)
    super_parents = torch.empty_like(super_tokens)
    super_depths = torch.empty_like(super_tokens)
    candidate_indices = torch.empty_like(super_tokens)
    super_scores = torch.empty((batch, max_nodes), dtype=torch.float32, device=device)
    expansion_count = width * candidate_k
    expanded_scores = torch.empty(
        (batch, expansion_count), dtype=torch.float32, device=device
    )
    expansion_block = triton.next_power_of_2(expansion_count)
    rank_block = triton.next_power_of_2(width)
    for depth in range(depth_count):
        _score_frontier_kernel[(batch,)](
            edge_log_probs,
            candidate_indices,
            super_scores,
            expanded_scores,
            depth=depth,
            depth_count=depth_count,
            candidate_k=candidate_k,
            width=width,
            max_nodes=max_nodes,
            expansion_count=expansion_count,
            expansion_block=expansion_block,
            num_warps=8 if expansion_block >= 512 else 4,
        )
        # Keep the framework's exact CUDA top-k implementation and tie order.
        # Real GSM8K traces contain exact ties; replacing this with repeated
        # Triton argmax changes selected token order and can change generation.
        chosen_scores, chosen_flat = torch.topk(
            expanded_scores, width, dim=-1, sorted=True
        )
        _materialize_frontier_kernel[(batch,)](
            candidate_ids,
            chosen_flat,
            chosen_scores,
            super_tokens,
            super_parents,
            super_depths,
            candidate_indices,
            super_scores,
            depth=depth,
            depth_count=depth_count,
            candidate_k=candidate_k,
            width=width,
            max_nodes=max_nodes,
            rank_block=rank_block,
            num_warps=1,
        )

    selected_tokens = torch.empty(
        (batch, selected_nodes), dtype=torch.long, device=device
    )
    selected_parents = torch.empty_like(selected_tokens)
    selected_depths = torch.empty_like(selected_tokens)
    selected_scores = torch.empty(
        (batch, selected_nodes), dtype=torch.float32, device=device
    )
    selected_old_ids = torch.empty_like(selected_tokens)
    _select_tree_nodes_kernel[(batch,)](
        super_depths,
        super_scores,
        selected_old_ids,
        depth_bonus=float(depth_bonus),
        budget=int(budget),
        max_nodes=max_nodes,
        selected_nodes=selected_nodes,
        selection_block=triton.next_power_of_2(max_nodes - 1),
        chosen_block=triton.next_power_of_2(selected_nodes),
        num_warps=4,
    )
    output_block = triton.next_power_of_2(selected_nodes)
    _materialize_selected_tree_kernel[(batch,)](
        anchor_tokens,
        super_tokens,
        super_parents,
        super_depths,
        super_scores,
        selected_old_ids,
        selected_tokens,
        selected_parents,
        selected_depths,
        selected_scores,
        anchor_stride=anchor_tokens.stride(0),
        selected_nodes=selected_nodes,
        super_stride=max_nodes,
        output_block=output_block,
        search_steps=int(selected_nodes).bit_length(),
        num_warps=4,
    )
    visibility, retrieve_index, next_token, next_sibling = _tree_metadata(
        selected_parents, max_depth=depth_count + 1
    )
    return UBTreeBatch(
        tokens=selected_tokens,
        parent_ids=selected_parents,
        depths=selected_depths,
        path_scores=selected_scores,
        visibility=visibility,
        retrieve_index=retrieve_index,
        retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling,
    )


@torch.inference_mode()
def _build_ubtree_batch_torch(
    *,
    anchor_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    edge_log_probs: torch.Tensor,
    budget: int,
    supertree_width: int,
    depth_bonus: float = 0.0,
) -> UBTreeBatch:
    """Build one fixed-size prefix-closed irregular tree per request on device."""

    if candidate_ids.ndim != 3 or edge_log_probs.ndim != 4:
        raise ValueError("candidate_ids/edge_log_probs must be rank 3/4")
    batch, depth_count, candidate_k = candidate_ids.shape
    if tuple(edge_log_probs.shape) != (
        batch,
        depth_count,
        candidate_k,
        candidate_k,
    ):
        raise ValueError("edge_log_probs must have shape [batch, depth, K, K]")
    if supertree_width <= 0 or supertree_width > candidate_k:
        raise ValueError("supertree_width must be in [1, candidate_k]")
    if budget <= 0 or budget > depth_count * supertree_width:
        raise ValueError("budget must be in [1, depth * supertree_width]")
    if depth_bonus > 0:
        raise ValueError("depth_bonus must be non-positive for prefix closure")

    device = candidate_ids.device
    width = int(supertree_width)
    max_nodes = 1 + depth_count * width
    selected_nodes = int(budget) + 1
    neg_inf = torch.finfo(torch.float32).min

    tokens = torch.zeros((batch, max_nodes), dtype=torch.long, device=device)
    parents = torch.full_like(tokens, -1)
    depths = torch.zeros_like(tokens)
    candidate_indices = torch.full_like(tokens, -1)
    scores = torch.full((batch, max_nodes), neg_inf, dtype=torch.float32, device=device)
    tokens[:, 0].copy_(anchor_tokens.reshape(batch))
    scores[:, 0] = 0

    frontier_nodes = torch.zeros((batch, width), dtype=torch.long, device=device)
    frontier_candidates = torch.zeros_like(frontier_nodes)
    frontier_scores = torch.full(
        (batch, width), neg_inf, dtype=torch.float32, device=device
    )
    frontier_scores[:, 0] = 0

    for depth in range(depth_count):
        edge_rows = edge_log_probs[:, depth].gather(
            1,
            frontier_candidates[:, :, None].expand(batch, width, candidate_k),
        )
        expanded_scores = frontier_scores[:, :, None] + edge_rows
        chosen_scores, chosen_flat = torch.topk(
            expanded_scores.reshape(batch, -1), width, dim=-1, sorted=True
        )
        parent_slots = torch.div(chosen_flat, candidate_k, rounding_mode="floor")
        successors = chosen_flat.remainder(candidate_k)
        start = 1 + depth * width
        stop = start + width
        parents[:, start:stop].copy_(frontier_nodes.gather(1, parent_slots))
        tokens[:, start:stop].copy_(candidate_ids[:, depth].gather(1, successors))
        depths[:, start:stop] = depth + 1
        candidate_indices[:, start:stop].copy_(successors)
        scores[:, start:stop].copy_(chosen_scores)
        frontier_nodes = torch.arange(start, stop, device=device).expand(batch, -1)
        frontier_candidates = successors
        frontier_scores = chosen_scores

    adjusted = scores[:, 1:] + float(depth_bonus) * depths[:, 1:].float()
    chosen_old = torch.argsort(adjusted, dim=-1, descending=True, stable=True)[
        :, :budget
    ].add(1)
    chosen_old = torch.sort(chosen_old, dim=-1).values
    old_nodes = torch.cat(
        (torch.zeros((batch, 1), dtype=torch.long, device=device), chosen_old), dim=1
    )
    selected_tokens = tokens.gather(1, old_nodes)
    selected_depths = depths.gather(1, old_nodes)
    selected_scores = scores.gather(1, old_nodes)
    selected_parent_old = parents.gather(1, old_nodes)

    old_to_new = torch.full_like(tokens, -1)
    new_ids = torch.arange(selected_nodes, device=device).expand(batch, -1)
    old_to_new.scatter_(1, old_nodes, new_ids)
    selected_parents = old_to_new.gather(1, selected_parent_old.clamp_min(0))
    selected_parents[:, 0] = -1

    visibility, retrieve_index, next_token, next_sibling = _tree_metadata(
        selected_parents, max_depth=depth_count + 1
    )
    return UBTreeBatch(
        tokens=selected_tokens,
        parent_ids=selected_parents,
        depths=selected_depths,
        path_scores=selected_scores,
        visibility=visibility,
        retrieve_index=retrieve_index,
        retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling,
    )


@triton.jit
def _build_best_first_tree_kernel(
    anchor_tokens,
    candidate_ids,
    edge_log_probs,
    tokens_out,
    parents_out,
    depths_out,
    scores_out,
    anchor_stride: tl.constexpr,
    candidate_batch_stride: tl.constexpr,
    candidate_depth_stride: tl.constexpr,
    candidate_stride: tl.constexpr,
    edge_batch_stride: tl.constexpr,
    edge_depth_stride: tl.constexpr,
    edge_parent_stride: tl.constexpr,
    edge_candidate_stride: tl.constexpr,
    num_depths: tl.constexpr,
    num_candidates: tl.constexpr,
    num_nodes: tl.constexpr,
    depth_bonus: tl.constexpr,
    node_block: tl.constexpr,
    pool_block: tl.constexpr,
):
    batch = tl.program_id(0)
    nodes = tl.arange(0, node_block)
    entries = tl.arange(0, pool_block)
    pool_parents = entries // num_candidates
    pool_candidates = entries % num_candidates
    edge_base = edge_log_probs + batch * edge_batch_stride
    candidate_base = candidate_ids + batch * candidate_batch_stride

    tokens = tl.where(nodes == 0, tl.load(anchor_tokens + batch * anchor_stride), 0)
    parents = tl.full((node_block,), -1, tl.int64)
    depths = tl.zeros((node_block,), tl.int32)
    scores = tl.zeros((node_block,), tl.float32)
    candidate_indices = tl.zeros((node_block,), tl.int32)
    active = entries < num_candidates
    pool = (
        tl.load(
            edge_base + pool_candidates * edge_candidate_stride,
            mask=active,
            other=-float("inf"),
        ).to(tl.float32)
        + depth_bonus
    )

    # A selected node exposes its children. Parent-before-child ordering makes
    # the output prefix closed without a separate prune/remap pass. The pool
    # tracks validity separately so zero-probability (-inf) edges remain valid
    # candidates when filling the fixed node budget.
    for node in range(1, num_nodes):
        best = tl.max(tl.where(active, pool, -float("inf")), 0)
        chosen = tl.min(tl.where(active & (pool == best), entries, pool_block), 0)
        parent = chosen // num_candidates
        candidate = chosen % num_candidates
        parent_depth = tl.sum(tl.where(nodes == parent, depths, 0), 0)
        parent_candidate = tl.sum(tl.where(nodes == parent, candidate_indices, 0), 0)
        parent_score = tl.sum(tl.where(nodes == parent, scores, 0), 0)
        edge = tl.load(
            edge_base
            + parent_depth * edge_depth_stride
            + parent_candidate * edge_parent_stride
            + candidate * edge_candidate_stride
        ).to(tl.float32)
        score = parent_score + edge
        depth = parent_depth + 1
        token = tl.load(
            candidate_base
            + parent_depth * candidate_depth_stride
            + candidate * candidate_stride
        )
        tokens = tl.where(nodes == node, token, tokens)
        parents = tl.where(nodes == node, parent.to(tl.int64), parents)
        depths = tl.where(nodes == node, depth, depths)
        scores = tl.where(nodes == node, score, scores)
        candidate_indices = tl.where(nodes == node, candidate, candidate_indices)

        expand = (pool_parents == node) & (depth < num_depths)
        child_edges = tl.load(
            edge_base
            + depth * edge_depth_stride
            + candidate * edge_parent_stride
            + pool_candidates * edge_candidate_stride,
            mask=expand,
            other=-float("inf"),
        ).to(tl.float32)
        child_scores = score + child_edges
        pool = tl.where(expand, child_scores + depth_bonus * (depth + 1), pool)
        active = (active & (entries != chosen)) | expand

    output = batch * num_nodes + nodes
    tl.store(tokens_out + output, tokens, mask=nodes < num_nodes)
    tl.store(parents_out + output, parents, mask=nodes < num_nodes)
    tl.store(depths_out + output, depths, mask=nodes < num_nodes)
    tl.store(scores_out + output, scores, mask=nodes < num_nodes)


@triton.jit
def _build_best_first_tree_cached_kernel(
    anchor_tokens,
    candidate_ids,
    edge_log_probs,
    tokens_out,
    parents_out,
    depths_out,
    scores_out,
    anchor_stride: tl.constexpr,
    candidate_batch_stride: tl.constexpr,
    candidate_depth_stride: tl.constexpr,
    candidate_stride: tl.constexpr,
    edge_batch_stride: tl.constexpr,
    edge_depth_stride: tl.constexpr,
    edge_parent_stride: tl.constexpr,
    edge_candidate_stride: tl.constexpr,
    num_depths: tl.constexpr,
    num_candidates: tl.constexpr,
    num_nodes: tl.constexpr,
    depth_bonus: tl.constexpr,
    node_block: tl.constexpr,
    candidate_block: tl.constexpr,
):
    # Each parent contributes only its best remaining child to the global
    # choice. Selecting a child changes two rows: its parent and the new node.
    # The uint64 mask indexes candidates, not tree nodes; N may exceed 64.
    tl.static_assert(num_candidates <= 64)
    batch = tl.program_id(0)
    nodes = tl.arange(0, node_block)
    candidates = tl.arange(0, candidate_block)
    valid = candidates < num_candidates
    bits = tl.full((candidate_block,), 1, tl.uint64) << candidates
    edge_base = edge_log_probs + batch * edge_batch_stride
    candidate_base = candidate_ids + batch * candidate_batch_stride
    tokens = tl.where(nodes == 0, tl.load(anchor_tokens + batch * anchor_stride), 0)
    parents = tl.full((node_block,), -1, tl.int64)
    depths = tl.zeros((node_block,), tl.int32)
    scores = tl.zeros((node_block,), tl.float32)
    candidate_indices = tl.zeros((node_block,), tl.int32)
    used = tl.zeros((node_block,), tl.uint64)
    root_edges = tl.load(
        edge_base + candidates * edge_candidate_stride,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    root_priorities = root_edges + depth_bonus
    root_best = tl.max(root_priorities, 0)
    root_candidate = tl.min(
        tl.where(valid & (root_priorities == root_best), candidates, num_candidates), 0
    )
    row_priorities = tl.where(nodes == 0, root_best, -float("inf"))
    row_candidates = tl.where(nodes == 0, root_candidate, 0)
    active = nodes == 0

    for node in range(1, num_nodes):
        best = tl.max(tl.where(active, row_priorities, -float("inf")), 0)
        parent = tl.min(
            tl.where(active & (row_priorities == best), nodes, num_nodes), 0
        )
        candidate = tl.sum(tl.where(nodes == parent, row_candidates, 0), 0)
        parent_depth = tl.sum(tl.where(nodes == parent, depths, 0), 0)
        parent_candidate = tl.sum(tl.where(nodes == parent, candidate_indices, 0), 0)
        parent_score = tl.sum(tl.where(nodes == parent, scores, 0), 0)
        parent_used = tl.sum(tl.where(nodes == parent, used, 0), 0)
        parent_used |= tl.full((), 1, tl.uint64) << candidate
        used = tl.where(nodes == parent, parent_used, used)
        parent_edges = tl.load(
            edge_base
            + parent_depth * edge_depth_stride
            + parent_candidate * edge_parent_stride
            + candidates * edge_candidate_stride,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        edge = tl.sum(tl.where(candidates == candidate, parent_edges, 0), 0)
        score = parent_score + edge
        depth = parent_depth + 1
        token = tl.load(
            candidate_base
            + parent_depth * candidate_depth_stride
            + candidate * candidate_stride
        )
        tokens = tl.where(nodes == node, token, tokens)
        parents = tl.where(nodes == node, parent.to(tl.int64), parents)
        depths = tl.where(nodes == node, depth, depths)
        scores = tl.where(nodes == node, score, scores)
        candidate_indices = tl.where(nodes == node, candidate, candidate_indices)

        remaining = valid & ((bits & parent_used) == 0)
        priorities = parent_score + parent_edges + depth_bonus * (parent_depth + 1)
        parent_best = tl.max(tl.where(remaining, priorities, -float("inf")), 0)
        parent_next = tl.min(
            tl.where(
                remaining & (priorities == parent_best), candidates, num_candidates
            ),
            0,
        )
        row_priorities = tl.where(nodes == parent, parent_best, row_priorities)
        row_candidates = tl.where(nodes == parent, parent_next, row_candidates)
        active = tl.where(
            nodes == parent, tl.sum(remaining.to(tl.int32), 0) > 0, active
        )

        child_edges = tl.load(
            edge_base
            + depth * edge_depth_stride
            + candidate * edge_parent_stride
            + candidates * edge_candidate_stride,
            mask=valid & (depth < num_depths),
            other=-float("inf"),
        ).to(tl.float32)
        child_priorities = score + child_edges + depth_bonus * (depth + 1)
        child_best = tl.max(tl.where(valid, child_priorities, -float("inf")), 0)
        child_next = tl.min(
            tl.where(
                valid & (child_priorities == child_best), candidates, num_candidates
            ),
            0,
        )
        row_priorities = tl.where(nodes == node, child_best, row_priorities)
        row_candidates = tl.where(nodes == node, child_next, row_candidates)
        active |= (nodes == node) & (depth < num_depths)

    output = batch * num_nodes + nodes
    tl.store(tokens_out + output, tokens, mask=nodes < num_nodes)
    tl.store(parents_out + output, parents, mask=nodes < num_nodes)
    tl.store(depths_out + output, depths, mask=nodes < num_nodes)
    tl.store(scores_out + output, scores, mask=nodes < num_nodes)


def _build_ubtree_batch_best_first(
    *,
    anchor_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    edge_log_probs: torch.Tensor,
    budget: int,
    depth_bonus: float,
) -> UBTreeBatch:
    """Select directly from the lattice; ties use (parent index, candidate index).

    Inputs are the same non-positive edge log probabilities as the beam builder.
    Candidate tokens must be unique at each depth, as provided by vocab top-k.
    Unlike beam construction, output nodes are in selection order, not depth order.
    """
    if not candidate_ids.is_cuda:
        raise ValueError("UBTree best-first construction requires CUDA tensors")
    batch, num_depths, num_candidates = candidate_ids.shape
    num_nodes = budget + 1
    anchor_tokens = anchor_tokens.reshape(batch)
    tokens = torch.empty(
        (batch, num_nodes), dtype=torch.long, device=candidate_ids.device
    )
    parents = torch.empty_like(tokens)
    depths = torch.empty_like(tokens)
    scores = torch.empty_like(tokens, dtype=torch.float32)
    if num_candidates <= 64:
        kernel = _build_best_first_tree_cached_kernel
        block_kwargs = {"candidate_block": triton.next_power_of_2(num_candidates)}
        num_warps = 1
    else:
        kernel = _build_best_first_tree_kernel
        block_kwargs = {
            "pool_block": triton.next_power_of_2(num_nodes * num_candidates)
        }
        num_warps = 4
    kernel[(batch,)](
        anchor_tokens,
        candidate_ids,
        edge_log_probs,
        tokens,
        parents,
        depths,
        scores,
        anchor_stride=anchor_tokens.stride(0),
        candidate_batch_stride=candidate_ids.stride(0),
        candidate_depth_stride=candidate_ids.stride(1),
        candidate_stride=candidate_ids.stride(2),
        edge_batch_stride=edge_log_probs.stride(0),
        edge_depth_stride=edge_log_probs.stride(1),
        edge_parent_stride=edge_log_probs.stride(2),
        edge_candidate_stride=edge_log_probs.stride(3),
        num_depths=num_depths,
        num_candidates=num_candidates,
        num_nodes=num_nodes,
        depth_bonus=float(depth_bonus),
        node_block=triton.next_power_of_2(num_nodes),
        **block_kwargs,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    visibility, retrieve_index, next_token, next_sibling = _tree_metadata(
        parents, max_depth=num_depths + 1
    )
    return UBTreeBatch(
        tokens=tokens,
        parent_ids=parents,
        depths=depths,
        path_scores=scores,
        visibility=visibility,
        retrieve_index=retrieve_index,
        retrieve_next_token=next_token,
        retrieve_next_sibling=next_sibling,
    )


@torch.inference_mode()
def build_ubtree_batch(
    *,
    anchor_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    edge_log_probs: torch.Tensor,
    budget: int,
    supertree_width: int,
    depth_bonus: float = 0.0,
) -> UBTreeBatch:
    """Build one fixed-size prefix-closed irregular tree per request on device."""

    if candidate_ids.ndim != 3 or edge_log_probs.ndim != 4:
        raise ValueError("candidate_ids/edge_log_probs must be rank 3/4")
    batch, depth_count, candidate_k = candidate_ids.shape
    if tuple(edge_log_probs.shape) != (
        batch,
        depth_count,
        candidate_k,
        candidate_k,
    ):
        raise ValueError("edge_log_probs must have shape [batch, depth, K, K]")
    if supertree_width <= 0 or supertree_width > candidate_k:
        raise ValueError("supertree_width must be in [1, candidate_k]")
    if budget <= 0 or budget > depth_count * supertree_width:
        raise ValueError("budget must be in [1, depth * supertree_width]")
    if depth_bonus > 0:
        raise ValueError("depth_bonus must be non-positive for prefix closure")
    if envs.SGLANG_USE_UBTREE_BEST_FIRST_TREE.get():
        return _build_ubtree_batch_best_first(
            anchor_tokens=anchor_tokens,
            candidate_ids=candidate_ids,
            edge_log_probs=edge_log_probs,
            budget=budget,
            depth_bonus=depth_bonus,
        )
    if candidate_ids.is_cuda:
        return _build_ubtree_batch_fused(
            anchor_tokens=anchor_tokens,
            candidate_ids=candidate_ids,
            edge_log_probs=edge_log_probs,
            budget=budget,
            supertree_width=supertree_width,
            depth_bonus=depth_bonus,
        )
    return _build_ubtree_batch_torch(
        anchor_tokens=anchor_tokens,
        candidate_ids=candidate_ids,
        edge_log_probs=edge_log_probs,
        budget=budget,
        supertree_width=supertree_width,
        depth_bonus=depth_bonus,
    )


def build_sglang_full_mask(
    visibility: torch.Tensor,
    prefix_lens: Sequence[int] | torch.Tensor,
    *,
    prefix_lens_sum: int | None = None,
) -> torch.Tensor:
    """Pack SGLang's per-request ``[query, prefix + tree]`` allow masks."""

    if isinstance(prefix_lens, torch.Tensor):
        if prefix_lens.device == visibility.device and prefix_lens_sum is not None:
            batch, nodes, _ = visibility.shape
            lengths = prefix_lens.to(torch.int64)
            total = nodes * (int(prefix_lens_sum) + batch * nodes)
            mask = torch.ones(total, dtype=torch.bool, device=visibility.device)
            starts = nodes * (
                torch.cumsum(lengths, dim=0)
                - lengths
                + torch.arange(batch, device=visibility.device) * nodes
            )
            query = torch.arange(nodes, device=visibility.device)[None, :, None]
            key = torch.arange(nodes, device=visibility.device)[None, None, :]
            tree_indices = (
                starts[:, None, None]
                + query * (lengths[:, None, None] + nodes)
                + lengths[:, None, None]
                + key
            )
            mask.scatter_(0, tree_indices.reshape(-1), visibility.reshape(-1))
            return mask
        if prefix_lens.device.type != "cpu":
            prefix_lens = prefix_lens.to(device="cpu")
        prefix_lens = prefix_lens.tolist()
    if len(prefix_lens) != visibility.shape[0]:
        raise ValueError("prefix_lens must contain one length per tree")
    masks = []
    nodes = visibility.shape[1]
    for batch_index, prefix_len in enumerate(prefix_lens):
        if int(prefix_len) < 0:
            raise ValueError("prefix lengths must be non-negative")
        prefix = torch.ones(
            (nodes, int(prefix_len)), dtype=torch.bool, device=visibility.device
        )
        masks.append(torch.cat((prefix, visibility[batch_index]), dim=1).reshape(-1))
    return torch.cat(masks) if masks else visibility.new_empty((0,))


__all__ = [
    "UBTreeBatch",
    "build_sglang_full_mask",
    "build_ubtree_batch",
    "compact_tree_values_to_front",
]
