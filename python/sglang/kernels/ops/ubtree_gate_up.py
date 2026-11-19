# SPDX-License-Identifier: Apache-2.0
# Adapted from SpecPilot's h100_c1_qwen38-27b_dflash2 specpilot_x1_gemm.py.
# Changes: TP-local eligibility, instance-owned descriptor, native UBTree dispatch.
"""Experimental SM90 BF16 gate/up GEMM with a fused SiLU epilogue."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ubtree_gate_up_kernel(
    x,
    weight_descriptor,
    output,
    rows,
    intermediate,
    hidden,
    stride_x,
    stride_output,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    ns = block * BN + tl.arange(0, BN)
    ms = tl.arange(0, BM)
    ks = tl.arange(0, BK)
    xp = x + ms[None, :] * stride_x + ks[:, None]
    gate = tl.zeros((BN, BM), tl.float32)
    up = tl.zeros((BN, BM), tl.float32)
    for k in range(0, hidden, BK):
        values = tl.load(xp, ms[None, :] < rows, other=0.0)
        wg = weight_descriptor.load([block * BN, k])
        wu = weight_descriptor.load([intermediate + block * BN, k])
        gate = tl.dot(wg, values, gate)
        up = tl.dot(wu, values, up)
        xp += BK
    # Preserve the original BF16 GEMM boundary before SiLU/multiply.
    # A different reduction order can nevertheless change these rounded activations.
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    result = g * tl.sigmoid(g) * u
    tl.store(
        output + ms[None, :] * stride_output + ns[:, None],
        result,
        (ms[None, :] < rows) & (ns[:, None] < intermediate),
    )


class UBTreeGateUp:
    """One descriptor per owning MLP; unsupported shapes use its normal path."""

    def __init__(self):
        self._descriptor = None
        self._weight_key = None

    def try_forward(self, x, weight):
        if not (
            isinstance(x, torch.Tensor)
            and x.ndim == 2
            and x.is_cuda
            and x.dtype == weight.dtype == torch.bfloat16
            and weight.shape in ((8704, 5120), (4352, 5120))
            and weight.is_contiguous()
            and x.shape[1] == 5120
            and 1 <= x.shape[0] <= 64
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
        ):
            return None
        from triton.tools.tensor_descriptor import TensorDescriptor

        # Weight reloads that replace storage must refresh the descriptor.
        key = (weight.data_ptr(), weight.device, weight.shape, weight.stride())
        if key != self._weight_key:
            self._descriptor = TensorDescriptor.from_tensor(weight, [64, 128])
            self._weight_key = key
        rows = x.shape[0]
        intermediate = weight.shape[0] // 2
        output = torch.empty((rows, intermediate), device=x.device, dtype=x.dtype)
        _ubtree_gate_up_kernel[(triton.cdiv(intermediate, 64),)](
            x,
            self._descriptor,
            output,
            rows,
            intermediate,
            5120,
            x.stride(0),
            output.stride(0),
            BM=max(16, triton.next_power_of_2(rows)),
            BN=64,
            BK=128,
            num_warps=4,
            num_stages=5,
        )
        return output
