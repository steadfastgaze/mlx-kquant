// clang-format off
// Segmented quantized GEMM kernel instantiations. The codec traits structs
// (KqQ2_KExt, KqIq2_xxsExt, ...) and their deq_chunk16 helpers come from the
// kq_quantized* headers; derived-code attribution lives there and in
// mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_segments.h"

// One (codec, type, staging-type, tile) instantiation. The name carries the
// codec, I/O type token and tile tag so eval_gpu can build it from
// kquant_type + x.dtype() + the selected tile. WM=WN=2 -> 128 threads. StageT
// is the threadgroup tile precision (float for the f32-decode contract; the
// I/O type for the half-staging opt-in).
#define instantiate_segments(codec, ext, type, bm, bn, bk, staget, tag) \
  instantiate_kernel(                                                    \
      "kquant_" #codec "_gather_qmm_segments_" #type "_" #tag,           \
      kq_gather_qmm_segments_impl,                                       \
      type,                                                              \
      ext,                                                               \
      bm,                                                                \
      bn,                                                                \
      bk,                                                                \
      2,                                                                 \
      2,                                                                 \
      staget)

// Tile variants (tag matches KQ_SEG_TILE selection in eval_gpu). The default
// is 48x64x16; the others are A/B levers. Every float-staging tile stages float
// (StageT=float), the f32-decode contract that the dq_f32 parity tests require.
// BK=16 is the minimum tile depth (the weight decode helper works on 16-weight
// chunks); the small-BK tiles shrink the two staged float tiles to lift
// occupancy while keeping the f32 decode. BM in {32,48,64} at BK=16 trades
// activation-tile footprint against how many row-chunks a segment splits into.
#define instantiate_segments_type(codec, ext, type)                    \
  instantiate_segments(codec, ext, type, 64, 64, 32, float, t64x64x32) \
  instantiate_segments(codec, ext, type, 32, 64, 64, float, t32x64x64) \
  instantiate_segments(codec, ext, type, 64, 32, 64, float, t64x32x64) \
  instantiate_segments(codec, ext, type, 64, 64, 16, float, t64x64x16) \
  instantiate_segments(codec, ext, type, 32, 64, 16, float, t32x64x16) \
  instantiate_segments(codec, ext, type, 48, 64, 16, float, t48x64x16) \
  instantiate_segments(codec, ext, type, 128, 64, 32, float, t128x64x32)

#define instantiate_segments_type_f32(codec, ext)                       \
  instantiate_segments(codec, ext, float, 64, 64, 32, float, t64x64x32) \
  instantiate_segments(codec, ext, float, 32, 64, 64, float, t32x64x64) \
  instantiate_segments(codec, ext, float, 64, 32, 64, float, t64x32x64) \
  instantiate_segments(codec, ext, float, 64, 64, 16, float, t64x64x16) \
  instantiate_segments(codec, ext, float, 32, 64, 16, float, t32x64x16) \
  instantiate_segments(codec, ext, float, 48, 64, 16, float, t48x64x16) \
  instantiate_segments(codec, ext, float, 128, 64, 32, float, t128x64x32)

// Half-staging opt-in (tag suffix h). Weights round through the I/O half type
// before the multiply, so these trade the f32-decode contract for a smaller
// threadgroup footprint (better occupancy). Selected only via KQ_SEG_TILE; the
// default dispatch never picks them, so the parity tests keep the float tiles.
#define instantiate_segments_type_half(codec, ext, type)              \
  instantiate_segments(codec, ext, type, 64, 64, 32, type, t64x64x32h) \
  instantiate_segments(codec, ext, type, 32, 64, 64, type, t32x64x64h) \
  instantiate_segments(codec, ext, type, 64, 64, 64, type, t64x64x64h)

#define instantiate_segments_all(codec, ext)                     \
  instantiate_segments_type(codec, ext, float16_t)               \
  instantiate_segments_type(codec, ext, bfloat16_t)              \
  instantiate_segments_type_f32(codec, ext)                      \
  instantiate_segments_type_half(codec, ext, float16_t)          \
  instantiate_segments_type_half(codec, ext, bfloat16_t)

// K-quants (256-weight super-blocks).
instantiate_segments_all(q2_k, KqQ2_KExt)
instantiate_segments_all(q3_k, KqQ3_KExt)
instantiate_segments_all(q4_k, KqQ4_KExt)
instantiate_segments_all(q5_k, KqQ5_KExt)
instantiate_segments_all(q6_k, KqQ6_KExt)

// IQ codecs sharing the same deq_chunk16 signature.
instantiate_segments_all(iq4_xs, KqIq4_xsExt)
instantiate_segments_all(iq3_s, KqIq3_sExt)
instantiate_segments_all(iq3_xxs, KqIq3_xxsExt)
instantiate_segments_all(iq2_xxs, KqIq2_xxsExt)
instantiate_segments_all(iq2_xs, KqIq2_xsExt)
instantiate_segments_all(iq2_s, KqIq2_sExt)
instantiate_segments_all(iq1_s, KqIq1_sExt)
instantiate_segments_all(iq1_m, KqIq1_mExt)

// 32-weight-block codecs (legacy + q8_0 + iq4_nl). K must be a multiple of the
// codec block; the op validates that.
instantiate_segments_all(q8_0, KqQ8_0Ext)
instantiate_segments_all(q4_0, KqQ4_0Ext)
instantiate_segments_all(q4_1, KqQ4_1Ext)
instantiate_segments_all(q5_0, KqQ5_0Ext)
instantiate_segments_all(q5_1, KqQ5_1Ext)
instantiate_segments_all(iq4_nl, KqIq4_nlExt)
    // clang-format on
