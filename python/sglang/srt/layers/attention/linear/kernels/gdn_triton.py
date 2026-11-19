import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_xpu

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
        fused_recurrent_gdn_replayssm_decode,
    )
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

if is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent_decode_optimized import (
        fused_sigmoid_gating_delta_rule_update_decode_npu as fused_sigmoid_gating_delta_rule_update,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
elif is_cpu():
    from sgl_kernel.mamba import chunk_gated_delta_rule_cpu

    chunk_gated_delta_rule = chunk_gated_delta_rule_cpu
    fused_sigmoid_gating_delta_rule_update = (
        torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu
    )
elif is_xpu():
    from sglang.srt.hardware_backend.xpu.kernels.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )


class TritonGDNKernel(LinearAttnKernelBase):
    """Triton-based kernel for GDN (Gated Delta Network) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()
    supports_strided_target_verify_qkv: bool = True

    def __init__(self):
        # Graph-stable, GDN-only factor scratch.  Keep this independent from
        # TritonKDAKernel so enabling the GDN tree path cannot alter KDA state.
        self._tree_factor_scratch = {}
        self._tree_prepared_scratch = {}

    def _get_tree_factor_scratch(self, q, v, query_start_loc, tree_width):
        n = query_start_loc.shape[0] - 1
        hv, k = v.shape[2], q.shape[-1]
        key = (q.device.index, n, hv, tree_width, k)
        pair = self._tree_factor_scratch.get(key)
        if pair is None:
            pair = (
                torch.empty(
                    (n, hv, tree_width, k), device=q.device, dtype=torch.float32
                ),
                torch.empty(
                    (n, hv, tree_width, tree_width),
                    device=q.device,
                    dtype=torch.float32,
                ),
            )
            self._tree_factor_scratch[key] = pair
        return pair

    def _get_tree_prepared_scratch(self, q, v):
        tokens, hv, k = q.shape[1], v.shape[2], q.shape[-1]
        key = (q.device.index, tokens, hv, k)
        buffers = self._tree_prepared_scratch.get(key)
        if buffers is None:
            buffers = tuple(
                torch.empty(shape, device=q.device, dtype=torch.float32)
                for shape in [(tokens, hv, k)] * 2 + [(tokens, hv)] * 3
            )
            self._tree_prepared_scratch[key] = buffers
        return buffers

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: fuse QKV extraction + gating + recurrent
        update into a single Triton kernel, eliminating intermediate tensors
        and extra kernel launches.

        Args:
            mixed_qkv: [B, qkv_dim] packed projection output after conv1d.
            a, b: [B, HV] gating inputs.
            A_log: [HV] log-space decay parameter.
            dt_bias: [HV] time-step bias.
            scale: attention scale factor (typically head_k_dim ** -0.5).
            ssm_states: [num_slots, HV, V, K] full state pool.
            cache_indices: [B] per-request state slot indices.
            num_v_heads: number of value heads (after TP sharding).
            head_v_dim: dimension per value head.

        Returns:
            output tensor of shape [1, B, HV, V] matching the existing
            decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        # Packed kernel expects output shape [B, 1, HV, V]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # GDN ReplaySSM buffered decode (slice 1a). Drop-in for the packed
        # decode: same args plus the three per-layer ring caches and the
        # per-row write cursor. When any ring tensor / cursor is None (flag
        # off) we fall through to the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        # GDN ReplaySSM (slice 2b): optional per-row force-flush (radix track
        # boundary). None when radix tracking is off / flag off; the kernel
        # treats None as "no forced flush" (byte-identical to slice 1a/1b).
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            fused_recurrent_gdn_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
            )
            return out.transpose(0, 1)

        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Convert [B, 1, HV, V] → [1, B, HV, V] to match existing output
        # layout. transpose() returns a view — zero cost.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        inplace_update: bool = True,
        **kwargs,
    ) -> tuple:
        recurrent_state = ssm_states
        recurrent_state_indices_args = {"initial_state_indices": cache_indices}
        inplace_update_args = {"inplace_update": inplace_update}
        # Static tables of a CUDA-graph captured prefill (gdn_prefill_graph.py).
        graph_table_args = {}
        if kwargs.get("chunk_indices") is not None:
            graph_table_args = {
                "chunk_indices": kwargs["chunk_indices"],
                "chunk_offsets": kwargs["chunk_offsets"],
            }
        if is_cpu():
            if not inplace_update:
                raise NotImplementedError(
                    "GDN multi-item scoring is not supported by the CPU chunk kernel"
                )
            inplace_update_args = {}
        elif is_npu():
            if not inplace_update:
                raise NotImplementedError(
                    "GDN multi-item scoring is not supported by the NPU chunk kernel"
                )
            recurrent_state = ssm_states[cache_indices]
            recurrent_state_indices_args = {}
            # The external NPU kernel does not expose the optional write-back
            # control. Its existing behavior is equivalent to True.
            inplace_update_args = {}

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
            **inplace_update_args,
            **graph_table_args,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor | None,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor | None,
        tree_depths: torch.Tensor | None = None,
        tree_max_depth: int = 0,
        cache_ring: bool = False,
        replayssm_rawv: torch.Tensor | None = None,
        replayssm_rawk: torch.Tensor | None = None,
        replayssm_g: torch.Tensor | None = None,
        replayssm_beta: torch.Tensor | None = None,
        factor_cache_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        factorized_tree = (
            retrieve_parent_token is not None
            and 8 <= cache_steps <= 65
            and tree_max_depth > 0
            and cache_ring
            and factor_cache_indices is not None
            and replayssm_rawv is not None
            and replayssm_rawk is not None
            and replayssm_g is not None
            and replayssm_beta is not None
            and replayssm_g.ndim == 3
        )
        if factorized_tree:
            tree_query_factors, tree_value_factors = self._get_tree_factor_scratch(
                q, v, query_start_loc, cache_steps
            )
        else:
            tree_query_factors = tree_value_factors = None
        # Other platform wrappers keep their existing call signature.
        verify_kwargs = {}
        if factorized_tree and kwargs.get("gdn_verify_shared_prep", False):
            verify_kwargs["gdn_tree_prepared"] = self._get_tree_prepared_scratch(q, v)
        if factorized_tree and kwargs.get("gdn_verify_use_tf32x3", False):
            verify_kwargs["gdn_verify_use_tf32x3"] = True
        tree_kwargs = {}
        if factorized_tree:
            tree_kwargs = dict(
                tree_depths=tree_depths,
                tree_max_depth=tree_max_depth,
                tree_query_factors=tree_query_factors,
                tree_value_factors=tree_value_factors,
                force_factorized_gdn_tree=True,
                cache_ring=True,
                replayssm_rawv=replayssm_rawv,
                replayssm_rawk=replayssm_rawk,
                replayssm_g=replayssm_g,
                replayssm_beta=replayssm_beta,
                factor_cache_indices=factor_cache_indices,
                **verify_kwargs,
            )
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            # target_verify specific parameters
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            **tree_kwargs,
        )
