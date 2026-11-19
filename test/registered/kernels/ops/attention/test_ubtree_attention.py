"""Mask, strided-layout and graph-replay checks for the experimental tree kernel."""

import pytest
import torch

from sglang.kernels.ops.attention.ubtree_attention import UBTreeAttention
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 PDL specialization",
)
@pytest.mark.parametrize("nodes,prefix", [(17, 0), (32, 128), (32, 1024), (65, 128)])
@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("h", [6, 3])
def test_tree_attention_mask_and_graph_refresh(nodes, prefix, with_gate, h):
    torch.manual_seed(6)
    d = 256
    q = torch.randn(nodes, h + 1, d, device="cuda", dtype=torch.bfloat16)[:, :h]
    k = torch.randn(prefix + nodes + 17, 1, d, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    gate = torch.randn_like(q) if with_gate else None
    slots = torch.randperm(k.shape[0], device="cuda")
    table = torch.zeros(nodes, nodes + 3, device="cuda", dtype=torch.int32)[:, :nodes]
    lengths = torch.empty(nodes, device="cuda", dtype=torch.int32)
    prefix_table = slots[: max(prefix, 1)].int()
    prefix_length = torch.tensor([prefix], device="cuda", dtype=torch.int32)
    for n in range(nodes):
        path = [n]
        while path[-1] > 0:
            path.append((path[-1] - 1) // 2)
        path.reverse()
        table[n, : len(path)] = slots[prefix + torch.tensor(path, device="cuda")].int()
        lengths[n] = len(path)
    kernel = UBTreeAttention()

    def reference():
        results = []
        for n in range(nodes):
            visible = torch.cat(
                (
                    prefix_table[: int(prefix_length.item())],
                    table[n, : int(lengths[n].item())],
                )
            ).long()
            logits = (
                torch.einsum("hd,kd->hk", q[n].float(), k[visible, 0].float()) * d**-0.5
            )
            o = (logits.softmax(-1) @ v[visible, 0].float()).to(q.dtype)
            if gate is not None:
                o = (o.float() * gate[n].float().sigmoid()).to(q.dtype)
            results.append(o)
        return torch.stack(results).reshape(nodes, -1)

    def candidate():
        return kernel.forward(
            q, k, v, prefix_table, prefix_length, table, lengths, gate, d**-0.5
        )

    actual = candidate()
    torch.testing.assert_close(actual, reference(), rtol=2e-2, atol=1e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    q.normal_()
    k.normal_()
    prefix_length.fill_(prefix // 2)
    # Switch the parent tree to a star while preserving every node's physical slot.
    table[:, 0] = slots[prefix]
    for n in range(1, nodes):
        table[n, 1] = slots[prefix + n]
    lengths[1:] = 2
    graph.replay()
    torch.testing.assert_close(captured, reference(), rtol=2e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
