"""mlx-kquant: GGUF K-quant ops for MLX.

Custom Metal kernels packaged as an MLX C++ extension, installed on top of a
stock ``mlx`` wheel.

Current API:
  * ``codecs`` / ``metallib_dir`` / ``metallib_loads`` /
    ``cpu_neon_available`` - toolchain self-checks.
  * ``dequantize`` - GGUF K-quant wire bytes -> float array.
  * ``quantized_matmul`` - x @ dequant(w) for K-quant weights.
  * ``quantized_matmul_qmv_bias`` - decode-only (M=1) bias-fused variant of
    ``quantized_matmul``, q8_0 only.
  * ``quantized_matmul_qmv_hc_post`` - decode-only Q8_0 matvec with the
    DeepSeek-V4 hyper-connection post recombination fused into its epilogue.
  * ``quantized_matmul_qmv_add_hc_post`` - fixed-geometry shared-FFN Q8_0
    matvec with routed-add and DeepSeek-V4 hC-post fused into its epilogue.
  * ``gather_qmm`` - mixture-of-experts gathered quantized matmul.
  * ``gather_qmm_segments`` - descriptor-driven segmented MoE quantized GEMM.
  * ``gather_qmm_sorted`` - segmented MoE quantized GEMM over device-sorted
    expert ids, row ranges derived in-kernel (no host descriptor table).
  * ``gather_qmm_sorted_swiglu`` - gather_qmm_sorted over a combined gate/up
    expert stack with the SwiGLU fused into the kernel epilogue.
  * ``gather_qmv_pair_swiglu`` - decode-shaped fused gate/up + SwiGLU matvec
    with per-expert route weights baked into the intermediate.
  * ``gather_qmv_expert_sum`` - decode-shaped down matvec summing the token's
    routed experts inside the kernel.
  * ``quantize`` - encode a float tensor into K-quant wire bytes (CPU or Metal).
  * ``load_gguf`` - load a GGUF file's tensors + metadata (C++ mmap memcpy).
"""

# Import mlx.core first: it registers the nanobind type caster for
# ``mlx.core.array``, which every op here accepts and returns. Without it, the
# first array crossing the C++/Python boundary raises an opaque ``std::bad_cast``
# instead of working - so make ``import mlx_kquant`` sufficient on its own.
import mlx.core as _mx  # noqa: F401

from ._ext import (  # noqa: F401
    add_rmsnorm,
    codecs,
    cpu_neon_available,
    dequantize,
    gather_qmm,
    gather_qmm_segments,
    gather_qmm_sorted,
    gather_qmm_sorted_swiglu,
    gather_qmv_bias,
    gather_qmv_expert_sum,
    gather_qmv_kq,
    gather_qmv_mix_kq,
    gather_qmv_mix_ns_kq,
    gather_qmv_pair_swiglu,
    load_gguf,
    metallib_dir,
    metallib_loads,
    moe_glu_gather,
    moe_glu_gather_kq,
    moe_glu_gather_shexp_kq,
    moe_router_topk,
    quantize,
    quantized_matmul,
    quantized_matmul_qmv_add_hc_post,
    quantized_matmul_qmv_bias,
    quantized_matmul_qmv_hc_post,
    rmsnorm2_add,
    rmsnorm_multi3,
    sdpa_decode_gqa,
    sdpa_decode_q8,
    sdpa_fa_prefill,
    sdpa_fa_prefill_q8,
    sdpa_fa_verify,
    sdpa_q8_loader_debug,
    sdpa_vector,
    verify_zero_copy_views,
    zero_copy_view_count,
)
from ._version import __version__

HAS_SDPA_DECODE_Q8_DIMENSION_PARALLEL_MERGE = True

__all__ = [
    "HAS_SDPA_DECODE_Q8_DIMENSION_PARALLEL_MERGE",
    "__version__",
    "add_rmsnorm",
    "codecs",
    "cpu_neon_available",
    "dequantize",
    "gather_qmm",
    "gather_qmm_segments",
    "gather_qmm_sorted",
    "gather_qmm_sorted_swiglu",
    "gather_qmv_bias",
    "gather_qmv_expert_sum",
    "gather_qmv_kq",
    "gather_qmv_mix_kq",
    "gather_qmv_mix_ns_kq",
    "gather_qmv_pair_swiglu",
    "load_gguf",
    "metallib_dir",
    "metallib_loads",
    "moe_glu_gather",
    "moe_glu_gather_kq",
    "moe_glu_gather_shexp_kq",
    "moe_router_topk",
    "quantize",
    "quantized_matmul",
    "quantized_matmul_qmv_add_hc_post",
    "quantized_matmul_qmv_bias",
    "quantized_matmul_qmv_hc_post",
    "rmsnorm2_add",
    "rmsnorm_multi3",
    "sdpa_decode_gqa",
    "sdpa_decode_q8",
    "sdpa_fa_prefill",
    "sdpa_fa_prefill_q8",
    "sdpa_fa_verify",
    "sdpa_q8_loader_debug",
    "sdpa_vector",
    "verify_zero_copy_views",
    "zero_copy_view_count",
]
