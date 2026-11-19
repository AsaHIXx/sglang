"""Node-parallel tree convolution: current-path parity, cache isolation, graphs."""

import pytest
import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.mamba.parallel_tree_conv import parallel_tree_conv_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def inputs(width, batch, channels, dtype):
    torch.manual_seed(width + batch)
    device = "cuda"
    x = torch.randn(batch, width, channels + 5, device=device, dtype=dtype)[
        ..., :channels
    ].transpose(1, 2)
    w = torch.randn(channels, 4, device=device, dtype=dtype) * 0.2
    bias = torch.randn(channels, device=device, dtype=dtype) * 0.1
    state = torch.randn(batch + 3, channels + 7, 3, device=device, dtype=dtype)[
        :, :channels
    ]
    slots = torch.arange(batch, device=device, dtype=torch.int32) + 1
    if batch > 1:
        slots[-1] = -1
    rows = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.int32)
    parents = torch.full((batch, width * 2), -1, device=device, dtype=torch.int64)[
        :, ::2
    ]
    for node in range(1, width):
        parents[:, node] = torch.randint(node, (batch,), device=device)
    raw = torch.full(
        (batch + 2, width + 3, channels + 7, 3), -7.0, device=device, dtype=dtype
    )[:, :, :channels]
    return x, w, bias, state, slots, raw, rows, parents


def reference(x, w, bias, state, slots, raw, rows, parents, activation):
    dummy = torch.full_like(parents, -1)
    return causal_conv1d_update(
        x,
        state,
        w,
        bias,
        activation,
        conv_state_indices=slots,
        intermediate_conv_window=raw,
        intermediate_state_indices=rows,
        retrieve_next_token=dummy,
        retrieve_next_sibling=dummy,
        retrieve_parent_token=parents,
        direct_parent=True,
        save_tree_raw_only=True,
    )


@pytest.mark.parametrize("width", [9, 17, 32, 49, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,channels,activation,has_bias", [(1, 95, "silu", True), (3, 257, None, False)]
)
def test_parallel_tree_conv_matches_existing_tree(
    width, dtype, batch, channels, activation, has_bias
):
    args = list(inputs(width, batch, channels, dtype))
    if not has_bias:
        args[2] = None
    initial_state = args[3].clone()
    initial_raw = args[5].clone()
    ref = reference(*args, activation)
    expected_raw = args[5].clone()
    args[5].copy_(initial_raw)
    out = parallel_tree_conv_update(*args, activation=activation)
    valid = args[4] >= 0
    assert torch.equal(out[valid], ref[valid])
    assert torch.equal(args[5], expected_raw)
    assert torch.equal(args[3], initial_state)
    # Independent raw-cache invariant: active requests write only their own
    # node inputs in the last window column; padded/unrelated storage is intact.
    manual = initial_raw.clone()
    for i in range(batch):
        if args[4][i] >= 0:
            manual[args[6][i], :width, :, 2] = args[0][i].T
    assert torch.equal(args[5], manual)


def test_parallel_tree_conv_graph_replay_refreshes_inputs():
    args = list(inputs(32, 2, 2560, torch.bfloat16))
    for _ in range(3):
        parallel_tree_conv_update(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = parallel_tree_conv_update(*args)
    args[0].normal_()
    args[3].normal_()
    args[7][:, 1:] = 0
    args[4][:] = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    graph.replay()
    actual = captured.clone()
    actual_raw = args[5].clone()
    expected = reference(*args, "silu")
    assert torch.equal(actual, expected)
    assert torch.equal(actual_raw, args[5])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
