// clang-format off
// Decode-shaped routed-MoE matvec kernel instantiations (fused pair+SwiGLU
// gate/up matvec with baked route weights; down matvec with the in-kernel sum
// over the token's routed experts). The kernel bodies live in
// kq_decode_moe.h; codec helpers and derived-code attribution live in the
// included kq_*.h headers and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_decode_moe.h"

// Instantiated for the served DS4 decode codec pair only: the combined
// gate/up pool ships iq2_xxs and the down projection q2_k. Other codecs stay
// uninstantiated to bound the metallib; the host op fails closed on them.
#define instantiate_kquant_decode_moe(func, type, gs, bits, codec)        \
  instantiate_kernel(                                                     \
      "kquant_" #codec "_" #func "_" #type "_gs_" #gs "_b_" #bits,        \
      kq_ ## codec ## _ ## func,                                          \
      type,                                                               \
      gs,                                                                 \
      bits)

#define instantiate_kquant_decode_moe_for_type(type)                          \
  instantiate_kquant_decode_moe(                                              \
      gather_qmv_pair_swiglu, type, 256, 2, iq2_xxs)                          \
  instantiate_kquant_decode_moe(gather_qmv_expert_sum, type, 256, 2, q2_k)

instantiate_kquant_decode_moe_for_type(float)
instantiate_kquant_decode_moe_for_type(bfloat16_t)
instantiate_kquant_decode_moe_for_type(float16_t)
    // clang-format on
