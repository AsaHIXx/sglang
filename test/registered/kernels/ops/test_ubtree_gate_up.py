"""Numerical tolerance and graph-buffer coverage for the experimental SM90 MLP."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.ubtree_gate_up import UBTreeGateUp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 specialization",
)
@pytest.mark.parametrize("rows", [9, 17, 32, 49, 64])
@pytest.mark.parametrize("intermediate", [4352, 2176])
def test_gate_up_graph_and_weight_storage_refresh(rows, intermediate):
    torch.manual_seed(6)
    x = torch.randn(rows, 5128, device="cuda", dtype=torch.bfloat16)[:, :5120]
    weight = (
        torch.randn(2 * intermediate, 5120, device="cuda", dtype=x.dtype) / 5120**0.5
    )
    kernel = UBTreeGateUp()

    def reference():
        gate, up = F.linear(x, weight).chunk(2, -1)
        return (F.silu(gate.float()) * up.float()).to(x.dtype)

    actual = kernel.try_forward(x, weight)
    torch.testing.assert_close(actual, reference(), rtol=2e-2, atol=2e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.try_forward(x, weight)
    x.normal_()
    weight.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, reference(), rtol=2e-2, atol=2e-2)
    old_key = kernel._weight_key
    weight = weight.clone()
    actual = kernel.try_forward(x, weight)
    assert kernel._weight_key != old_key
    torch.testing.assert_close(actual, reference(), rtol=2e-2, atol=2e-2)
    assert (
        kernel.try_forward(torch.empty(65, 5120, device="cuda", dtype=x.dtype), weight)
        is None
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
