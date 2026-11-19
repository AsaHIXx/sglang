# UBTree Qwen GDN support

Dense Qwen3.5 targets (including the Qwen3.8-27B checkpoint) use the native
UBTree worker with compact GDN factors and accepted-path state commit. The
merge branch retains C1, CUDA, FA3, page size 1, PP1/DP1 and full decode graph
constraints. GDN verification currently supports 8–65 nodes with 128-dimensional
key/value state. KDA allocation and commit paths are unchanged.

Optional single-stage optimizations (all default off):

| Environment variable | Effect |
|---|---|
| `SGLANG_USE_UBTREE_BEST_FIRST_TREE` | Build the proposal directly to the node budget using global best-first selection. |
| `SGLANG_OPT_UBTREE_PARALLEL_TREE_CONV` | Compute independent tree convolution nodes in parallel. |
| `SGLANG_OPT_UBTREE_GDN_SHARED_PREP` | Reuse FP32 normalized q/k and gates during ancestor traversal. |
| `SGLANG_OPT_UBTREE_GDN_LAYOUT_VIEWS` | Reuse projection and post-convolution tensor views. |
| `SGLANG_OPT_UBTREE_FUSED_GATE_UP` | Fuse BF16 gate/up GEMM and SiLU on SM90, Qwen-27B TP4/TP8 shapes. |
| `SGLANG_OPT_UBTREE_FUSED_TREE_ATTENTION` | Fuse prefix/tree attention, merge and output gate on SM90, C1, Qwen-27B TP4/TP8 shapes. |
| `SGLANG_OPT_UBTREE_FUSED_SUFFIX_MERGE` | Fuse the FA3 tree suffix and prefix merge as a separate experimental alternative. |
| `SGLANG_OPT_UBTREE_GDN_VERIFY_TF32X3` | Use TF32x3 for GDN output recovery on SM90. |

Best-first changes proposal selection/tie order. GEMM, attention and TF32x3
fusions may change floating-point rounding and generated tokens. They do not
establish AR token parity or repair sampling acceptance length. Numerically
changed fusions are disabled under deterministic inference. Unsupported fused
MLP/attention shapes retain their ordinary model paths.

The TP8 comparison uses best-first and the first five optimization flags above;
suffix/merge and TF32x3 stay disabled. The merge branch requires
`SGLANG_OPT_UBTREE_FA3_TREE_ONLY_MASK=1`; the historical TP4 development run
used the full tree mask representation. No two-stage preview, W12, proposal
quantization or adaptive width policy is included.
