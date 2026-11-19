# SPDX-License-Identifier: Apache-2.0
# Adapted from SpecPilot h100_c1_qwen38-27b_dflash2/specpilot_d8_attn.py.
# Changes: instance-owned scratch, explicit native dispatch, TP4 tuning.
"""Experimental two-kernel tree attention with output-gate fusion."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ubtree_tree_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pt_ptr,
    plen_ptr,
    st_ptr,
    slen_ptr,
    pacc_ptr,
    pml_ptr,
    scale,
    sq_t,
    sq_h,
    sk_t,
    sk_h,
    spt,
    sst,
    P_W,
    NSLOT,
    N: tl.constexpr,
    NP: tl.constexpr,
    G: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    S: tl.constexpr,
    QB: tl.constexpr,
    PDL: tl.constexpr,
    SFX_F32: tl.constexpr = True,
):
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()  # the combine kernel launches and loads the gate meanwhile
    i_qb, i_s, i_kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = i_qb * BM + tl.arange(0, BM)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = tl.arange(0, D)
    q = tl.load(
        q_ptr + tok[:, None].to(tl.int64) * sq_t + hq[:, None] * sq_h + dd[None, :],
        mask=rvalid[:, None],
        other=0.0,
    )
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qk_scale = scale * 1.44269504088896  # exp2 domain
    if i_s < S:
        L = tl.minimum(
            tl.maximum(tl.load(plen_ptr), 0), P_W
        )  # memory-safe for capture-time metadata
        chunk = tl.cdiv(tl.cdiv(L, S), BN) * BN
        start = i_s * chunk
        end = tl.minimum(L, start + chunk)
        kj = start + tl.arange(0, BN)
        slot = tl.load(pt_ptr + kj * spt, mask=kj < end, other=0)
        slot = tl.minimum(tl.maximum(slot, 0), NSLOT - 1).to(tl.int64)
        for j in tl.range(start, end, BN):
            kval = kj < end
            kb = tl.load(
                k_ptr + slot[:, None] * sk_t + i_kv * sk_h + dd[None, :],
                mask=kval[:, None],
                other=0.0,
            )
            vb = tl.load(
                v_ptr + slot[:, None] * sk_t + i_kv * sk_h + dd[None, :],
                mask=kval[:, None],
                other=0.0,
            )
            kn = kj + BN
            slot = tl.load(
                pt_ptr + kn * spt, mask=kn < end, other=0
            )  # next block's slots
            slot = tl.minimum(tl.maximum(slot, 0), NSLOT - 1).to(tl.int64)
            s = tl.dot(q, tl.trans(kb)) * qk_scale
            s = tl.where(kval[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(vb.dtype), vb)
            m_i = m_new
            kj = kn
    else:  # tree suffix: node slots + per-row ancestor-or-self mask from the suffix page table
        nn = tl.arange(0, NP)
        nval = nn < N
        nlen = tl.minimum(tl.maximum(tl.load(slen_ptr + nn, mask=nval, other=1), 1), N)
        nslot = tl.load(st_ptr + nn * sst + (nlen - 1), mask=nval, other=-1)
        nslot = tl.where(nval, tl.minimum(tl.maximum(nslot, 0), NSLOT - 1), -1)
        rlen = tl.minimum(tl.load(slen_ptr + tok, mask=rvalid, other=0), N)
        vis = tl.zeros([BM, NP], tl.int1)
        for kk in tl.static_range(N):
            e = tl.load(st_ptr + tok * sst + kk, mask=rvalid & (kk < rlen), other=-2)
            vis = vis | (e[:, None] == nslot[None, :])
        vis = vis & nval[None, :]
        ns64 = tl.maximum(nslot, 0).to(tl.int64)
        kb = tl.load(
            k_ptr + ns64[:, None] * sk_t + i_kv * sk_h + dd[None, :],
            mask=nval[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(kb)) * qk_scale
        s = tl.where(vis, s, float("-inf"))
        m_i = tl.max(s, 1)
        m_i = tl.where(rvalid, m_i, 0.0)
        p = tl.exp2(s - m_i[:, None])
        p = tl.where(vis, p, 0.0)
        l_i = tl.sum(p, 1)
        vt = tl.load(
            v_ptr + ns64[:, None] * sk_t + i_kv * sk_h + dd[None, :],
            mask=nval[:, None],
            other=0.0,
        )
        if SFX_F32:  # P as a hi + lo bf16 pair: ~16 mantissa bits (V is bf16, products exact in fp32)
            p_hi = p.to(vt.dtype)
            p_lo = (p - p_hi.to(tl.float32)).to(vt.dtype)
            acc = tl.dot(p_hi, vt) + tl.dot(p_lo, vt)
        else:
            acc = tl.dot(p.to(vt.dtype), vt)
    # partial out
    pid = (i_kv * QB + i_qb) * (S + 1) + i_s
    rr = tl.arange(0, BM)
    tl.store(pacc_ptr + (pid * BM + rr[:, None]) * D + dd[None, :], acc)
    tl.store(pml_ptr + (pid * 2) * BM + rr, m_i)
    tl.store(pml_ptr + (pid * 2 + 1) * BM + rr, l_i)


@triton.jit
def _ubtree_tree_attn_combine_kernel(
    gate_ptr,
    out_ptr,
    pacc_ptr,
    pml_ptr,
    sg_t,
    sg_h,
    so_t,
    N: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    S: tl.constexpr,
    QB: tl.constexpr,
    RB: tl.constexpr,
    HAS_GATE: tl.constexpr,
    PDL: tl.constexpr,
    TRIGGER: tl.constexpr = False,
):
    if TRIGGER:  # a PDL-launched o_proj GEMM may pre-stage its weights now (it waits for this kernel's output)
        tl.extra.cuda.gdc_launch_dependents()
    # merge the S + 1 partials of RB rows of one KV head, sigmoid output gate, bf16 store
    i_r, i_kv = tl.program_id(0), tl.program_id(1)
    rows = i_r * RB + tl.arange(0, RB)
    rvalid = rows < N * G
    tok = rows // G
    hq = i_kv * G + rows % G
    dd = tl.arange(0, D)
    if HAS_GATE:
        gt = tl.load(
            gate_ptr
            + tok[:, None].to(tl.int64) * sg_t
            + hq[:, None] * sg_h
            + dd[None, :],
            mask=rvalid[:, None],
            other=0.0,
        ).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    qb = rows // BM
    rr = rows % BM
    base = (i_kv * QB + qb) * (S + 1)
    mx = tl.full([RB], float("-inf"), tl.float32)
    for t in tl.static_range(S + 1):
        mt = tl.load(
            pml_ptr + ((base + t) * 2) * BM + rr, mask=rvalid, other=float("-inf")
        )
        mx = tl.maximum(mx, mt)
    num = tl.zeros([RB, D], tl.float32)
    den = tl.zeros([RB], tl.float32)
    for t in tl.static_range(S + 1):
        mt = tl.load(
            pml_ptr + ((base + t) * 2) * BM + rr, mask=rvalid, other=float("-inf")
        )
        lt = tl.load(pml_ptr + ((base + t) * 2 + 1) * BM + rr, mask=rvalid, other=0.0)
        w = tl.where(mt == float("-inf"), 0.0, tl.exp2(mt - mx))
        at = tl.load(
            pacc_ptr + ((base + t) * BM + rr)[:, None] * D + dd[None, :],
            mask=rvalid[:, None],
            other=0.0,
        )
        num += w[:, None] * at
        den += w * lt
    o = num / tl.where(rvalid, den, 1.0)[:, None]
    o = o.to(out_ptr.dtype.element_ty).to(tl.float32)
    if HAS_GATE:
        o = o * tl.sigmoid(gt)
    tl.store(
        out_ptr + tok[:, None].to(tl.int64) * so_t + hq[:, None] * D + dd[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=rvalid[:, None],
    )


class UBTreeAttention:
    """Graph-stable scratch owned by one attention layer, with no global cache."""

    def __init__(self):
        self._scratch = {}

    def forward(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        prefix_len,
        suffix_table,
        suffix_len,
        gate,
        scale,
    ):
        n, hq, dim = q.shape
        assert (hq, dim) in ((6, 256), (3, 256)) and k_cache.shape[1:] == (1, 256)
        assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
        assert q.stride(-1) == k_cache.stride(-1) == 1
        assert k_cache.stride() == v_cache.stride()
        assert suffix_table.shape[0] >= n and suffix_len.shape[0] >= n
        assert gate is None or (gate.shape == q.shape and gate.stride(-1) == 1)
        hkv = k_cache.shape[1]
        group = hq // hkv
        splits, bm, bn, rb = 8, 32, 64, 8
        qb = triton.cdiv(n * group, bm)
        key = (q.device, n, hq, hkv, dim)
        if key not in self._scratch:
            self._scratch[key] = (
                torch.empty(
                    hkv * qb * (splits + 1),
                    bm,
                    dim,
                    device=q.device,
                    dtype=torch.float32,
                ),
                torch.empty(
                    hkv * qb * (splits + 1), 2, bm, device=q.device, dtype=torch.float32
                ),
            )
        partial, stats = self._scratch[key]
        out = q.new_empty(n, hq * dim)
        g = q if gate is None else gate
        _ubtree_tree_attn_kernel[(qb, splits + 1, hkv)](
            q,
            k_cache,
            v_cache,
            page_table,
            prefix_len,
            suffix_table,
            suffix_len,
            partial,
            stats,
            scale,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(-1),
            suffix_table.stride(0),
            page_table.shape[-1],
            k_cache.shape[0],
            N=n,
            NP=max(16, triton.next_power_of_2(n)),
            G=group,
            HKV=hkv,
            D=dim,
            BM=bm,
            BN=bn,
            S=splits,
            QB=qb,
            PDL=True,
            SFX_F32=True,
            num_warps=4,
            num_stages=2,
        )
        _ubtree_tree_attn_combine_kernel[(triton.cdiv(n * group, rb), hkv)](
            g,
            out,
            partial,
            stats,
            g.stride(0),
            g.stride(1),
            out.stride(0),
            N=n,
            G=group,
            D=dim,
            BM=bm,
            S=splits,
            QB=qb,
            RB=rb,
            HAS_GATE=gate is not None,
            PDL=True,
            TRIGGER=False,
            num_warps=4,
            launch_pdl=True,
        )
        return out
