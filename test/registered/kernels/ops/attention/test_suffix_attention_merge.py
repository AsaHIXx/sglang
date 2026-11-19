"""CUDA correctness tests for the fused suffix-attention merge."""

import math
import unittest

import torch

from sglang.kernels.ops.attention.suffix_attention_merge import (
    merge_suffix_attention_in_place,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=15,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSuffixAttentionMerge(CustomTestCase):
    def _case(self, *, num_queries: int, head_dim: int, dtype: torch.dtype):
        torch.manual_seed(7)
        device = torch.device("cuda")
        num_q_heads = 8
        num_kv_heads = 2
        num_slots = 2 * num_queries + 11

        q = torch.randn(num_queries, num_q_heads, head_dim, device=device, dtype=dtype)
        k_cache = torch.randn(
            num_slots, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        v_cache = torch.randn_like(k_cache)
        page_table = torch.stack(
            [
                torch.randperm(num_slots, device=device)[:num_queries]
                for _ in range(num_queries)
            ]
        ).to(torch.int32)
        suffix_lengths = (
            torch.arange(num_queries, device=device, dtype=torch.int32)
            .remainder(num_queries)
            .add_(1)
        )
        prefix = torch.randn_like(q)
        prefix_lse = torch.randn(
            num_q_heads, num_queries, device=device, dtype=torch.float32
        )
        scale = 1.0 / math.sqrt(head_dim)

        reference = prefix.float().clone()
        heads_per_kv = num_q_heads // num_kv_heads
        kv_heads = torch.arange(num_q_heads, device=device) // heads_per_kv
        for token in range(num_queries):
            length = int(suffix_lengths[token])
            slots = page_table[token, :length].long()
            keys = k_cache[slots][:, kv_heads].float()
            values = v_cache[slots][:, kv_heads].float()
            scores = torch.einsum("lhd,hd->lh", keys, q[token].float()) * scale
            maximum = torch.maximum(prefix_lse[:, token], scores.max(dim=0).values)
            prefix_weight = torch.exp(prefix_lse[:, token] - maximum)
            suffix_weights = torch.exp(scores - maximum)
            reference[token] = (
                reference[token] * prefix_weight[:, None]
                + torch.einsum("lh,lhd->hd", suffix_weights, values)
            ) / (prefix_weight + suffix_weights.sum(dim=0))[:, None]

        static_prefix = prefix.clone()
        merge_suffix_attention_in_place(
            q,
            k_cache,
            v_cache,
            page_table,
            suffix_lengths,
            static_prefix,
            prefix_lse,
            scale,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_prefix.copy_(prefix)
            merge_suffix_attention_in_place(
                q,
                k_cache,
                v_cache,
                page_table,
                suffix_lengths,
                static_prefix,
                prefix_lse,
                scale,
            )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            static_prefix.float(), reference, rtol=2e-2, atol=2e-2
        )

    def test_representative_shapes(self):
        cases = (
            (16, 64, torch.float16),
            (60, 128, torch.bfloat16),
        )
        for num_queries, head_dim, dtype in cases:
            with self.subTest(
                num_queries=num_queries,
                head_dim=head_dim,
                dtype=dtype,
            ):
                self._case(
                    num_queries=num_queries,
                    head_dim=head_dim,
                    dtype=dtype,
                )

    def test_ubtree_paths_and_graph_metadata_refresh(self):
        # Qwen3.8 TP4 has six query heads sharing one KV head, D=256.
        # Two requests have disjoint physical slots and different prefix states.
        for width in (9, 32, 49, 65):
            with self.subTest(width=width):
                torch.manual_seed(width)
                tokens, heads, dim = 2 * width, 6, 256
                q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(tokens * 3, 1, dim, device="cuda", dtype=q.dtype)
                v = torch.randn_like(k)
                slots = torch.randperm(k.shape[0], device="cuda")[:tokens]
                # Non-contiguous rows, with poisoned padding excluded by lengths.
                table = torch.full(
                    (tokens, width * 2), -100000, device="cuda", dtype=torch.int32
                )[:, ::2]
                paths = []
                for request in range(2):
                    for node in range(width):
                        path = [node]
                        while path[-1] > 0:
                            current = path[-1]
                            # A full-depth chain plus siblings branching from it.
                            path.append(
                                current - 1 if current < 16 else (current - 16) % 15
                            )
                        path = [request * width + i for i in reversed(path)]
                        paths.append(path)
                        table[request * width + node, : len(path)] = slots[path].int()
                lengths = torch.tensor(
                    [len(p) for p in paths], device="cuda", dtype=torch.int32
                )
                prefix = torch.randn_like(q)
                lse = torch.randn(tokens, heads, device="cuda").T
                # Empty committed prefix for the first request; the root remains visible.
                prefix[:width].zero_()
                lse[:, :width] = -float("inf")
                output = torch.empty_like(prefix)
                scale = dim**-0.5

                def run():
                    output.copy_(prefix)
                    merge_suffix_attention_in_place(
                        q,
                        k,
                        v,
                        table,
                        lengths,
                        output,
                        lse,
                        scale,
                        max_suffix_length=max(len(p) for p in paths),
                    )

                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for replay in range(3):
                    if replay:
                        q.normal_()
                        # Refresh both lengths and physical slots without recapture.
                        lengths.clamp_(max=replay)
                        table.copy_(
                            torch.where(table >= 0, (table + 1) % k.shape[0], table)
                        )
                    if replay == 2:
                        lse[:, :width] = float("inf")
                    graph.replay()
                    expected = torch.empty_like(output)
                    for token in range(tokens):
                        ids = table[token, : int(lengths[token])].long()
                        keys = k[ids, 0].float()
                        values = v[ids, 0].float()
                        scores = q[token].float() @ keys.T * scale
                        prefix_lse = lse[:, token, None]
                        prefix_lse = torch.where(
                            torch.isinf(prefix_lse), -float("inf"), prefix_lse
                        )
                        weights = torch.softmax(
                            torch.cat([prefix_lse, scores], dim=1), dim=1
                        )
                        expected[token] = (
                            weights[:, :1] * prefix[token].float()
                            + weights[:, 1:] @ values
                        )
                    self.assertTrue(torch.isfinite(output).all())
                    torch.testing.assert_close(output, expected, rtol=1e-2, atol=4e-3)


if __name__ == "__main__":
    unittest.main()
