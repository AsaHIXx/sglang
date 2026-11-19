"""Correctness tests for factorized GDN tree verification and accepted-path fold."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

try:
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
        commit_gdn_tree_all_layers,
    )

    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _tree(width: int, batch_size: int, device: str):
    """Build a bounded-depth tree and return parent/depth tensors."""
    parents = [-1]
    depths = [0]
    for node in range(1, width):
        parent = (node - 1) // 3
        parents.append(parent)
        depths.append(depths[parent] + 1)
    return (
        torch.tensor([parents] * batch_size, dtype=torch.int32, device=device),
        torch.tensor([depths] * batch_size, dtype=torch.int32, device=device),
    )


def _inputs(width: int, batch_size: int, *, seed: int = 20261008):
    torch.manual_seed(seed + width + batch_size)
    device = "cuda"
    num_query_heads, num_value_heads = 2, 6
    key_dim = value_dim = 128
    num_slots = batch_size + 3
    total = batch_size * width
    parents, depths = _tree(width, batch_size, device)
    tensors = {
        "A_log": torch.randn(num_value_heads, dtype=torch.float32, device=device),
        "dt_bias": torch.randn(num_value_heads, dtype=torch.bfloat16, device=device),
        "a": torch.randn(
            1, total, num_value_heads, dtype=torch.bfloat16, device=device
        ),
        "b": torch.randn(
            1, total, num_value_heads, dtype=torch.bfloat16, device=device
        ),
        "q": torch.randn(
            1,
            total,
            num_query_heads,
            key_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        "k": torch.randn(
            1,
            total,
            num_query_heads,
            key_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        "v": torch.randn(
            1,
            total,
            num_value_heads,
            value_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        "state": torch.randn(
            num_slots,
            num_value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=device,
        ),
        "state_indices": torch.arange(batch_size, dtype=torch.int32, device=device) + 1,
        "factor_indices": torch.arange(batch_size, dtype=torch.int32, device=device),
        "cu_seqlens": torch.arange(
            0, total + 1, width, dtype=torch.int32, device=device
        ),
        "parents": parents,
        "depths": depths,
    }
    return tensors


def _common(x, width: int):
    return dict(
        A_log=x["A_log"],
        a=x["a"],
        dt_bias=x["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=x["q"],
        k=x["k"],
        v=x["v"],
        b=x["b"],
        initial_state_source=x["state"],
        initial_state_indices=x["state_indices"],
        cu_seqlens=x["cu_seqlens"],
        is_kda=False,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        intermediate_state_indices=x["factor_indices"],
        cache_steps=width,
        retrieve_parent_token=x["parents"],
    )


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Triton GDN kernel is unavailable")
@pytest.mark.parametrize(
    "width,batch_size",
    [
        # Lower boundary and the two priority production widths.
        (8, 1),
        (9, 1),
        (17, 2),
        # Non-power-of-two widths and every Triton tile transition up to 65.
        (10, 1),
        (16, 1),
        (18, 1),
        (31, 1),
        (32, 1),
        (33, 1),
        (49, 1),
        (64, 1),
        (65, 1),
    ],
)
@pytest.mark.parametrize("use_tf32x3", [False, True])
def test_factorized_gdn_tree_matches_serial_and_commit(width, batch_size, use_tf32x3):
    x = _inputs(width, batch_size)
    num_slots, num_value_heads, value_dim, key_dim = x["state"].shape
    num_query_heads = x["k"].shape[2]
    serial_states = torch.full(
        (num_slots, width, num_value_heads, value_dim, key_dim),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    common = _common(x, width)
    serial_out = fused_sigmoid_gating_delta_rule_update(
        **common, intermediate_states_buffer=serial_states
    )

    query_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        key_dim,
        dtype=torch.float32,
        device="cuda",
    )
    value_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        width,
        dtype=torch.float32,
        device="cuda",
    )
    rawv = torch.full(
        (num_slots, num_value_heads, width, value_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    rawk = torch.full(
        (num_slots, num_query_heads, width, key_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = torch.full(
        (num_slots, num_value_heads, width),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    beta = torch.full_like(gate, float("nan"))
    factorized_out = fused_sigmoid_gating_delta_rule_update(
        **common,
        intermediate_states_buffer=None,
        tree_depths=x["depths"],
        tree_max_depth=int(x["depths"].max().item()) + 1,
        tree_query_factors=query_factors,
        tree_value_factors=value_factors,
        force_factorized_gdn_tree=True,
        gdn_verify_use_tf32x3=use_tf32x3,
        cache_ring=True,
        replayssm_rawv=rawv,
        replayssm_rawk=rawk,
        replayssm_g=gate,
        replayssm_beta=beta,
        factor_cache_indices=x["factor_indices"],
    )

    torch.testing.assert_close(factorized_out, serial_out, rtol=2e-2, atol=2e-2)

    if not use_tf32x3 and width < 16:
        legacy_tile_out = fused_sigmoid_gating_delta_rule_update(
            **common,
            intermediate_states_buffer=None,
            tree_depths=x["depths"],
            tree_max_depth=int(x["depths"].max().item()) + 1,
            tree_query_factors=query_factors,
            tree_value_factors=value_factors,
            force_factorized_gdn_tree=True,
            cache_ring=True,
            replayssm_rawv=rawv,
            replayssm_rawk=rawk,
            replayssm_g=gate,
            replayssm_beta=beta,
            factor_cache_indices=x["factor_indices"],
            factor_verify_bm=32,
            factor_verify_bv=32,
            factor_verify_num_warps=2,
            factor_verify_num_stages=1,
        )
        assert torch.equal(factorized_out, legacy_tile_out)

    # Commit one valid root-to-leaf path per request and compare against the
    # exact state snapshot produced by the original full-state tree path.
    paths = []
    for request in range(batch_size):
        local_last = width - 1 - request
        path = []
        node = local_last
        while node >= 0:
            path.append(node)
            node = int(x["parents"][request, node].item())
        path.reverse()
        paths.append(path)
    max_accept = max(map(len, paths))
    accept_index = torch.full(
        (batch_size, max_accept), -1, dtype=torch.int32, device="cuda"
    )
    for request, path in enumerate(paths):
        accept_index[request, : len(path)] = torch.tensor(
            [request * width + node for node in path],
            dtype=torch.int32,
            device="cuda",
        )
    accept_lens = torch.tensor(
        [len(path) for path in paths], dtype=torch.int32, device="cuda"
    )
    committed = x["state"].clone().unsqueeze(0)
    commit_gdn_tree_all_layers(
        checkpoint_state=committed,
        rawv_cache=rawv.unsqueeze(0),
        rawk_cache=rawk.unsqueeze(0),
        g_cache=gate.unsqueeze(0),
        beta_cache=beta.unsqueeze(0),
        ssm_state_indices=x["state_indices"],
        factor_cache_indices=x["factor_indices"],
        accept_index=accept_index,
        accept_lens=accept_lens,
        tree_width=width,
    )
    for request, path in enumerate(paths):
        state_slot = x["state_indices"][request]
        snapshot_slot = x["factor_indices"][request]
        torch.testing.assert_close(
            committed[0, state_slot],
            serial_states[snapshot_slot, path[-1]],
            rtol=1e-5,
            atol=1e-5,
        )

    expected_k = x["k"][0].reshape(batch_size, width, num_query_heads, key_dim)
    expected_v = x["v"][0].reshape(batch_size, width, num_value_heads, value_dim)
    for request, factor_slot in enumerate(x["factor_indices"].tolist()):
        torch.testing.assert_close(
            rawk[factor_slot], expected_k[request].transpose(0, 1)
        )
        torch.testing.assert_close(
            rawv[factor_slot], expected_v[request].transpose(0, 1)
        )
        assert torch.isfinite(gate[factor_slot]).all()
        assert torch.isfinite(beta[factor_slot]).all()


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Triton GDN kernel is unavailable")
@pytest.mark.parametrize(
    "width,use_tf32x3", [(9, False), (17, False), (32, True), (65, True)]
)
def test_factorized_gdn_tree_cuda_graph_replay_reads_new_inputs(width, use_tf32x3):
    """IEEE and TF32x3 paths must replay with graph-stable addresses."""
    batch_size = 1
    x = _inputs(width, batch_size, seed=20261009)
    _, num_value_heads, value_dim, key_dim = x["state"].shape
    num_query_heads = x["k"].shape[2]
    common = _common(x, width)
    tree_max_depth = int(x["depths"].max().item()) + 1

    query_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        key_dim,
        dtype=torch.float32,
        device="cuda",
    )
    value_factors = torch.empty(
        batch_size,
        num_value_heads,
        width,
        width,
        dtype=torch.float32,
        device="cuda",
    )
    rawv = torch.empty(
        batch_size,
        num_value_heads,
        width,
        value_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    rawk = torch.empty(
        batch_size,
        num_query_heads,
        width,
        key_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gate = torch.empty(
        batch_size,
        num_value_heads,
        width,
        dtype=torch.float32,
        device="cuda",
    )
    beta = torch.empty_like(gate)

    def run(qf, vf, rv, rk, g, bt):
        return fused_sigmoid_gating_delta_rule_update(
            **common,
            intermediate_states_buffer=None,
            tree_depths=x["depths"],
            tree_max_depth=tree_max_depth,
            tree_query_factors=qf,
            tree_value_factors=vf,
            force_factorized_gdn_tree=True,
            gdn_verify_use_tf32x3=use_tf32x3,
            cache_ring=True,
            replayssm_rawv=rv,
            replayssm_rawk=rk,
            replayssm_g=g,
            replayssm_beta=bt,
            factor_cache_indices=x["factor_indices"],
        )

    # Compile before capture, then capture with the exact production pointer layout.
    run(query_factors, value_factors, rawv, rawk, gate, beta)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run(query_factors, value_factors, rawv, rawk, gate, beta)

    # Mutate all recurrent inputs in place. Replay must consume the new values,
    # rather than factors or raw-cache contents left over from graph capture.
    for name in ("a", "b", "q", "k", "v"):
        tensor = x[name]
        tensor.copy_(torch.randn_like(tensor))
    graph.replay()
    replay_out = graph_out.clone()

    eager_out = run(
        torch.empty_like(query_factors),
        torch.empty_like(value_factors),
        torch.empty_like(rawv),
        torch.empty_like(rawk),
        torch.empty_like(gate),
        torch.empty_like(beta),
    )
    torch.testing.assert_close(replay_out, eager_out, rtol=0, atol=0)


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Triton GDN kernel is unavailable")
@pytest.mark.parametrize("width", [9, 17, 32, 49, 65])
def test_tf32x3_recovery_matches_fp64_with_ragged_rows(width):
    import triton

    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        _gdn_tree_factor_verify_kernel,
    )

    torch.manual_seed(41 + width)
    batch, heads, dim = 3, 4, 128
    lengths = [width, width - 3, 0]
    total = sum(lengths)
    cu = torch.tensor([0, width, total, total], device="cuda", dtype=torch.int32)
    indices = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
    # Pad slots and token rows: only the leading strides are supplied to the
    # kernel, while each head's [V, K] state and value dimension stay contiguous.
    state = torch.randn(5, heads + 1, dim, dim, device="cuda")[:, :heads]
    values = torch.randn(1, total, heads + 1, dim, device="cuda", dtype=torch.bfloat16)[
        :, :, :heads
    ]
    qf = torch.randn(batch, heads, width, dim, device="cuda")
    vf = torch.randn(batch, heads, width, width, device="cuda")
    for request, length in enumerate(lengths):
        qf[request, :, length:] = float("nan")
        vf[request, :, length:] = float("nan")
        vf[request, :, :length, length:] = float("nan")
    output = torch.full((1, total, heads, dim), float("nan"), device="cuda")
    _gdn_tree_factor_verify_kernel[
        (triton.cdiv(dim, 32), batch * triton.cdiv(width, 16), heads)
    ](
        v=values,
        o=output,
        h0_source=state,
        h0_indices=indices,
        stride_h0_source=state.stride(0),
        cu_seqlens=cu,
        tree_query_factors=qf,
        tree_value_factors=vf,
        stride_v=values.stride(1),
        TREE_WIDTH=width,
        H=heads,
        HV=heads,
        K=dim,
        V=dim,
        BK=dim,
        BV=32,
        BT=triton.next_power_of_2(width),
        BM=16,
        NM=triton.cdiv(width, 16),
        FACTORS_LOW_PRECISION=False,
        INPUT_PRECISION="tf32x3",
        num_warps=4,
        num_stages=1,
    )
    offset = 0
    for request, length in enumerate(lengths):
        if not length:
            continue
        query = qf[request, :, :length].double()
        initial = (
            state[indices[request]].double()
            if indices[request] >= 0
            else torch.zeros_like(state[0], dtype=torch.float64)
        )
        value = values[0, offset : offset + length].transpose(0, 1).double()
        reference = (
            query @ initial.transpose(-1, -2)
            + vf[request, :, :length, :length].double() @ value
        )
        torch.testing.assert_close(
            output[0, offset : offset + length].transpose(0, 1),
            reference.float(),
            rtol=2e-4,
            atol=2e-5,
        )
        offset += length


@pytest.mark.parametrize(
    "width,batch_size", [(9, 1), (17, 3), (32, 1), (49, 2), (65, 1)]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_shared_gdn_prep_preserves_factors_cache_and_graph(
    width, batch_size, normalize
):
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    x = _inputs(width, batch_size)
    # Padded token rows and non-contiguous gates are accepted by both consumers.
    for name in ("q", "k", "v", "a", "b"):
        original = x[name]
        padded = torch.empty(
            (*original.shape[:-1], original.shape[-1] * 2),
            dtype=original.dtype,
            device=original.device,
        )
        # Preserve contiguous head dimensions; introduce padding between tokens.
        if original.ndim == 4:
            padded = torch.empty(
                (1, original.shape[1], original.shape[2] + 1, original.shape[3]),
                dtype=original.dtype,
                device="cuda",
            )
            view = padded[:, :, : original.shape[2]]
        else:
            view = padded[..., : original.shape[-1]]
        view.copy_(original)
        x[name] = view
    active_tokens = batch_size * width
    if batch_size > 1:
        active_tokens -= 3
        x["cu_seqlens"][-1] = active_tokens
        x["factor_indices"][-1] = -1
    hv, k = x["v"].shape[2], x["q"].shape[-1]
    h = x["q"].shape[2]
    qf = torch.zeros(batch_size, hv, width, k, device="cuda")
    vf = torch.zeros(batch_size, hv, width, width, device="cuda")
    rawv = torch.zeros(
        batch_size + 3, hv, width + 3, k, dtype=x["v"].dtype, device="cuda"
    )
    rawk = torch.zeros(
        batch_size + 3, h, width + 3, k, dtype=x["k"].dtype, device="cuda"
    )
    gate = torch.zeros(batch_size + 3, hv, width + 3, device="cuda")
    beta = torch.zeros_like(gate)
    common = _common(x, width)
    common.update(
        use_qk_l2norm_in_kernel=normalize,
        softplus_beta=0.75,
        softplus_threshold=12.0,
        scale=0.17,
        tree_depths=x["depths"],
        tree_max_depth=16,
        tree_query_factors=qf,
        tree_value_factors=vf,
        force_factorized_gdn_tree=True,
        cache_ring=True,
        replayssm_rawv=rawv,
        replayssm_rawk=rawk,
        replayssm_g=gate,
        replayssm_beta=beta,
        factor_cache_indices=x["factor_indices"],
    )
    owner = TritonGDNKernel()
    prepared = owner._get_tree_prepared_scratch(x["q"], x["v"])
    assert owner._get_tree_prepared_scratch(x["q"], x["v"]) is prepared
    state_before = x["state"].clone()

    def reference():
        return fused_sigmoid_gating_delta_rule_update(**common)

    def candidate():
        return fused_sigmoid_gating_delta_rule_update(
            **common, gdn_tree_prepared=prepared
        )

    expected = reference().clone()
    expected_cache = [t.clone() for t in (qf, vf, rawv, rawk, gate, beta)]
    actual = candidate()
    assert torch.equal(expected[:, :active_tokens], actual[:, :active_tokens])
    for ref, got in zip(expected_cache, (qf, vf, rawv, rawk, gate, beta)):
        assert torch.equal(ref, got)
    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    # Replay with different input values, including gate threshold crossings.
    x["a"].uniform_(-24, 24)
    x["q"].normal_()
    x["k"].normal_()
    expected = reference().clone()
    expected_cache = [t.clone() for t in (qf, vf, rawv, rawk, gate, beta)]
    graph.replay()
    assert torch.equal(expected[:, :active_tokens], captured[:, :active_tokens])
    for ref, got in zip(expected_cache, (qf, vf, rawv, rawk, gate, beta)):
        assert torch.equal(ref, got)
    assert torch.equal(state_before, x["state"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


@pytest.mark.skipif(not KERNEL_AVAILABLE, reason="Triton GDN kernel is unavailable")
def test_gdn_tree_commit_graph_refreshes_gate_and_preserves_disarmed_state():
    """Capture/disarmed tails cannot commit stale factors or conv windows."""
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
        commit_gdn_tree_after_verify,
    )

    device = "cuda"
    width = 32
    temporal = torch.randn(2, 4, 6, 128, 128, device=device)
    conv = torch.randn(2, 4, 1280, 3, device=device, dtype=torch.bfloat16)
    state = SimpleNamespace(
        temporal=temporal,
        conv=[conv],
        tree_rawv=torch.zeros(2, 2, 6, width, 128, device=device, dtype=torch.bfloat16),
        tree_rawk=torch.zeros(2, 2, 2, width, 128, device=device, dtype=torch.bfloat16),
        tree_g=torch.zeros(2, 2, 6, width, device=device),
        tree_beta=torch.zeros(2, 2, 6, width, device=device),
        intermediate_conv_window=[
            torch.full((2, 2, width, 1280, 3), 2.0, device=device, dtype=torch.bfloat16)
        ],
    )
    kwargs = dict(
        spec_state=state,
        state_batch_indices=torch.tensor([1], device=device, dtype=torch.int32),
        factor_cache_indices=torch.tensor([0], device=device, dtype=torch.int32),
        accept_index=torch.zeros(1, 4, device=device, dtype=torch.int32),
        accept_lens=torch.zeros(1, device=device, dtype=torch.int32),
        tree_width=width,
        mamba_track_indices=torch.tensor([2], device=device, dtype=torch.int32),
        mamba_steps_to_track=torch.tensor([0], device=device, dtype=torch.int32),
    )
    before_state, before_conv = temporal.clone(), conv.clone()
    commit_gdn_tree_after_verify(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        commit_gdn_tree_after_verify(**kwargs)
    graph.replay()
    assert torch.equal(temporal, before_state)
    assert torch.equal(conv, before_conv)

    # With zero k/beta and log gate=0, SSM state is unchanged; the tracked
    # checkpoint copies it, and one accepted node shifts the conv window.
    kwargs["accept_lens"].fill_(1)
    graph.replay()
    expected_state = before_state.clone()
    expected_state[:, 2] = before_state[:, 1]
    expected_conv = before_conv.clone()
    window = torch.cat(
        (before_conv[:, 1, :, 1:], torch.full_like(before_conv[:, 1, :, :1], 2)), dim=-1
    )
    expected_conv[:, 1] = window
    expected_conv[:, 2] = window
    assert torch.equal(temporal, expected_state)
    assert torch.equal(conv, expected_conv)

    # A subsequent disarmed replay must not read the now-poisoned proposal.
    kwargs["accept_lens"].zero_()
    kwargs["accept_index"].fill_(-1)
    for tensor in (
        state.tree_rawv,
        state.tree_rawk,
        state.tree_g,
        state.tree_beta,
        *state.intermediate_conv_window,
    ):
        tensor.fill_(float("nan"))
    graph.replay()
    assert torch.equal(temporal, expected_state)
    assert torch.equal(conv, expected_conv)
