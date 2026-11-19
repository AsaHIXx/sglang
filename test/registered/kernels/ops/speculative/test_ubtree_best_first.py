"""Best-first proposal selection against an independent priority-queue oracle."""

import heapq

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.ubtree_components.tree import (
    _retrieval_links,
    _visibility,
    build_ubtree_batch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _best_first_heap_reference(anchor, candidates, edges, budget, depth_bonus):
    """Independent CPU priority queue over the full candidate lattice."""
    tokens, parents, depths, scores = [int(anchor)], [-1], [0], [0.0]
    candidate_indices = [0]
    pending = []

    def expand(parent):
        depth = depths[parent]
        if depth == candidates.shape[0]:
            return
        for candidate in range(candidates.shape[1]):
            score = float(
                torch.tensor(scores[parent], dtype=torch.float32)
                + edges[depth, candidate_indices[parent], candidate]
            )
            priority = float(
                torch.tensor(score, dtype=torch.float32)
                + torch.tensor(depth_bonus, dtype=torch.float32) * (depth + 1)
            )
            heapq.heappush(pending, (-priority, parent, candidate, score))

    expand(0)
    for _ in range(budget):
        _, parent, candidate, score = heapq.heappop(pending)
        tokens.append(int(candidates[depths[parent], candidate]))
        parents.append(parent)
        depths.append(depths[parent] + 1)
        scores.append(score)
        candidate_indices.append(candidate)
        expand(len(tokens) - 1)
    return tokens, parents, depths, scores


def _assert_best_first_reference(tree, anchors, candidates, edges, depth_bonus=0.0):
    anchors, candidates, edges = anchors.cpu(), candidates.cpu(), edges.cpu()
    for batch in range(tree.batch_size):
        expected = _best_first_heap_reference(
            anchors[batch],
            candidates[batch],
            edges[batch],
            tree.node_count - 1,
            depth_bonus,
        )
        for field, values in zip(
            ("tokens", "parent_ids", "depths", "path_scores"), expected
        ):
            actual = getattr(tree, field)[batch].cpu()
            torch.testing.assert_close(
                actual, torch.tensor(values, dtype=actual.dtype), rtol=0, atol=0
            )
    parents = tree.parent_ids.cpu()
    assert torch.all(
        (parents[:, 1:] >= 0) & (parents[:, 1:] < torch.arange(1, tree.node_count))
    )
    assert torch.equal(tree.visibility.cpu(), _visibility(parents))
    for actual, expected in zip(
        (tree.retrieve_index, tree.retrieve_next_token, tree.retrieve_next_sibling),
        _retrieval_links(parents),
    ):
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel requires CUDA")
@pytest.mark.parametrize(
    "num_nodes,num_candidates,num_depths,batch",
    [
        (2, 1, 1, 1),
        (16, 1, 15, 2),
        (9, 3, 4, 2),
        (34, 33, 1, 2),
        (65, 64, 1, 1),
        (9, 65, 4, 2),
        (32, 64, 15, 1),
        (49, 64, 15, 2),
        (65, 64, 15, 8),
    ],
)
def test_best_first_matches_heap_and_metadata(
    num_nodes, num_candidates, num_depths, batch
):
    generator = torch.Generator().manual_seed(311 + num_nodes)
    anchors = torch.arange(batch * 16, device="cuda").view(batch, 16)[:, 0]
    candidates = torch.arange(
        batch * num_depths * num_candidates * 2, device="cuda"
    ).view(batch, num_depths, num_candidates * 2)[:, :, ::2]
    # Strided candidate and edge tensors exercise all input strides.
    edges = torch.empty(
        batch, num_depths, num_candidates, num_candidates * 2, device="cuda"
    )[..., ::2]
    edges.copy_(
        torch.log_softmax(
            torch.randn(
                batch, num_depths, num_candidates, num_candidates, generator=generator
            ),
            -1,
        )
    )
    for depth_bonus in (0.0, -0.25):
        with envs.SGLANG_USE_UBTREE_BEST_FIRST_TREE.override(True):
            tree = build_ubtree_batch(
                anchor_tokens=anchors,
                candidate_ids=candidates,
                edge_log_probs=edges,
                budget=num_nodes - 1,
                supertree_width=max(
                    min(12, num_candidates), (num_nodes - 2) // num_depths + 1
                ),
                depth_bonus=depth_bonus,
            )
        _assert_best_first_reference(tree, anchors, candidates, edges, depth_bonus)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel requires CUDA")
def test_best_first_can_keep_more_than_beam_width_at_one_depth():
    candidates = torch.arange(12, device="cuda").view(1, 4, 3)
    edges = torch.full((1, 4, 3, 3), -100.0, device="cuda")
    edges[:, 0, 0] = torch.tensor([-1.0, -1.1, -1.2], device="cuda")
    args = dict(
        anchor_tokens=torch.tensor([99], device="cuda"),
        candidate_ids=candidates,
        edge_log_probs=edges,
        budget=3,
        supertree_width=1,
    )
    with envs.SGLANG_USE_UBTREE_BEST_FIRST_TREE.override(False):
        beam = build_ubtree_batch(**args)
    with envs.SGLANG_USE_UBTREE_BEST_FIRST_TREE.override(True):
        best_first = build_ubtree_batch(**args)
    assert beam.depths.tolist() == [[0, 1, 2, 3]]
    assert best_first.depths.tolist() == [[0, 1, 1, 1]]
    assert best_first.tokens.tolist() == [[99, 0, 1, 2]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
@pytest.mark.parametrize("nodes", [9, 32, 65])
def test_best_first_graph_changes_topology(nodes):
    with envs.SGLANG_USE_UBTREE_BEST_FIRST_TREE.override(True):
        args = dict(
            anchor_tokens=torch.tensor([4], device="cuda"),
            candidate_ids=torch.arange(15 * 64, device="cuda").view(1, 15, 64),
            edge_log_probs=torch.full((1, 15, 64, 64), -4.0, device="cuda"),
            budget=nodes - 1,
            supertree_width=12,
        )
        build_ubtree_batch(**args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = build_ubtree_batch(**args)
        for replay in range(3):
            args["edge_log_probs"].fill_(-float("inf"))
            args["edge_log_probs"][..., replay : replay + 2] = -1.0
            args["candidate_ids"].add_(1)
            args["anchor_tokens"].add_(1)
            graph.replay()
            _assert_best_first_reference(
                result,
                args["anchor_tokens"],
                args["candidate_ids"],
                args["edge_log_probs"],
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
