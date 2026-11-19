"""Node-parallel width-four convolution for compact GDN tree verification.

Nodes read only their input ancestors and the pre-tree state. Arithmetic and
raw-input cache layout match the ordinary tree convolution; the existing
accept-path fold remains the sole writer of committed convolution states.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _parallel_tree_conv_kernel(
    X,
    W,
    Bias,
    State,
    Slots,
    Raw,
    Rows,
    Parents,
    output_ptr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    sw0: tl.constexpr,
    sw1: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    sr0: tl.constexpr,
    sr1: tl.constexpr,
    sr2: tl.constexpr,
    sr3: tl.constexpr,
    sp0: tl.constexpr,
    sp1: tl.constexpr,
    so0: tl.constexpr,
    so1: tl.constexpr,
    so2: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BN: tl.constexpr,
    BF: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    batch = tl.program_id(0).to(tl.int64)
    feat = tl.program_id(1) * BF + tl.arange(0, BF)
    node = tl.program_id(2) * BN + tl.arange(0, BN)
    slot = tl.load(Slots + batch).to(tl.int64)
    if slot < 0:
        return
    row = tl.load(Rows + batch).to(tl.int64)
    valid = (node[:, None] < N) & (feat[None, :] < D)
    if HAS_BIAS:
        acc = tl.broadcast_to(
            tl.load(Bias + feat, feat < D, other=0).to(tl.float32)[None, :], (BN, BF)
        )
    else:
        acc = tl.full((BN, BF), 0, tl.float32)
    current = node
    for tap in tl.static_range(4):
        data = tl.load(
            X
            + batch * sx0
            + tl.maximum(current, 0)[:, None] * sx2
            + feat[None, :] * sx1,
            valid & (current[:, None] >= 0),
            other=0,
        )
        prior = tl.load(
            State + slot * ss0 + feat[None, :] * ss1 + (3 + current[:, None]) * ss2,
            valid & (current[:, None] < 0) & (current[:, None] >= -3),
            other=0,
        )
        value = tl.where(current[:, None] >= 0, data, prior)
        weight = tl.load(W + feat * sw0 + (3 - tap) * sw1, feat < D, other=0)
        acc += value * weight[None, :]
        if tap == 0:
            tl.store(
                Raw + row * sr0 + node[:, None] * sr1 + feat[None, :] * sr2 + 2 * sr3,
                data,
                valid & (row >= 0),
            )
        parent = tl.load(
            Parents + batch * sp0 + tl.maximum(current, 0) * sp1,
            (node < N) & (current > 0),
            other=0,
        )
        current = tl.where(current > 0, parent, current - 1)
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(
        output_ptr + batch * so0 + node[:, None] * so2 + feat[None, :] * so1, acc, valid
    )


def parallel_tree_conv_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    parents: torch.Tensor,
    *,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Read [batch, channels, nodes] and save raw x in the existing last slot.

    Only width-four, direct-parent, raw-input-cache tree verification is served.
    Negative state slots are padding and leave output/cache rows untouched.
    Parent indices are already validated by the proposal metadata producer.
    """
    batch, channels, num_nodes = x.shape
    assert weight.shape == (channels, 4)
    assert conv_state.shape[1] == channels and conv_state.shape[2] >= 3
    assert intermediate_conv_window.shape[1] >= num_nodes
    assert intermediate_conv_window.shape[2:] == (channels, 3)
    assert parents.shape == (batch, num_nodes)
    assert state_indices.stride(0) == intermediate_state_indices.stride(0) == 1
    assert x.dtype == weight.dtype == conv_state.dtype
    assert activation in (None, "silu", "swish")
    output = torch.empty_like(x)
    pdl_kwargs = {"launch_pdl": True} if is_arch_support_pdl() else {}
    _parallel_tree_conv_kernel[
        (batch, triton.cdiv(channels, 64), triton.cdiv(num_nodes, 4))
    ](
        x,
        weight,
        bias if bias is not None else weight,
        conv_state,
        state_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        parents,
        output,
        *x.stride(),
        *weight.stride(),
        *conv_state.stride(),
        *intermediate_conv_window.stride(),
        *parents.stride(),
        *output.stride(),
        N=num_nodes,
        D=channels,
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        BN=4,
        BF=64,
        USE_GDC=is_arch_support_pdl(),
        num_warps=4,
        **pdl_kwargs,
    )
    return output
