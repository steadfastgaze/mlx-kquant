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
// I/O type for the half-staging opt-in). The trailing flag selects the GEMM
// body: false = both operands staged (steel BlockMMA), true = device-A
// (weight tile only, unpadded; see kq_segments.h). Every tile is instantiated
// for both entry points: the descriptor-table kernel (gather_qmm_segments)
// and the binary-search kernel over device-sorted ids (gather_qmm_sorted),
// which share the GEMM bodies.
#define instantiate_segments(codec, ext, type, bm, bn, bk, staget, deva, tag) \
  instantiate_kernel(                                                          \
      "kquant_" #codec "_gather_qmm_segments_" #type "_" #tag,                 \
      kq_gather_qmm_segments_impl,                                             \
      type,                                                                    \
      ext,                                                                     \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      2,                                                                       \
      2,                                                                       \
      staget,                                                                  \
      deva)                                                                    \
  instantiate_kernel(                                                          \
      "kquant_" #codec "_gather_qmm_sorted_" #type "_" #tag,                   \
      kq_gather_qmm_sorted_impl,                                               \
      type,                                                                    \
      ext,                                                                     \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      2,                                                                       \
      2,                                                                       \
      staget,                                                                  \
      deva)

// Tile variants (tag matches KQ_SEG_TILE selection in eval_gpu). The default
// is t48x128x16a: the device-A body (activation fragments read from device
// rows, weight tile unpadded), BN=128 to halve the per-row-chunk activation
// re-reads, BK=16 to keep the staged weight tile small (8 KB). On the sorted
// MoE bulk-prefill shape (S=23058, K=N=4096, E=256) it measures 89.3-91.0 ms
// for iq2_xxs against 105.6-106.5 ms for the best staged tile (t48x64x16,
// the previous default), byte-equal outputs, alternating matched arms. The
// sweep that produced it also ruled out: register-prefetch and
// double-buffered staging (register pressure and footprint regressed every
// tile they touched), 256-thread simdgroup layouts, BM=96 (halved dequant
// passes, register spill), BK=32 paired decode (halved decode headers,
// register spill), and a stashed-register paired decode. Every rejected
// variant was bit-identical; they lost on speed only.
//
// The staged float tiles stay as KQ_SEG_TILE A/B levers (t48x64x16 pins the
// previous default). BK=16 is the minimum tile depth (the weight decode
// helper works on 16-weight chunks). Only BN (the n-tile width) reaches the
// host - it sizes the grid; BM/BK and the body flag are internal to the
// kernel. The segments and sorted kernels share the selection so a
// segments/sorted parity pair always runs the same tile.
#define instantiate_segments_type(codec, ext, type)                              \
  instantiate_segments(codec, ext, type, 48, 128, 16, float, true, t48x128x16a)  \
  instantiate_segments(codec, ext, type, 64, 64, 32, float, false, t64x64x32)    \
  instantiate_segments(codec, ext, type, 32, 64, 64, float, false, t32x64x64)    \
  instantiate_segments(codec, ext, type, 64, 32, 64, float, false, t64x32x64)    \
  instantiate_segments(codec, ext, type, 64, 64, 16, float, false, t64x64x16)    \
  instantiate_segments(codec, ext, type, 32, 64, 16, float, false, t32x64x16)    \
  instantiate_segments(codec, ext, type, 48, 64, 16, float, false, t48x64x16)    \
  instantiate_segments(codec, ext, type, 128, 64, 32, float, false, t128x64x32)

// Half-staging opt-in (tag suffix h). Weights round through the I/O half type
// before the multiply, so these trade the f32-decode contract for a smaller
// threadgroup footprint (better occupancy). Selected only via KQ_SEG_TILE; the
// default dispatch never picks them, so the parity tests keep the float tiles.
#define instantiate_segments_type_half(codec, ext, type)                     \
  instantiate_segments(codec, ext, type, 64, 64, 32, type, false, t64x64x32h) \
  instantiate_segments(codec, ext, type, 32, 64, 64, type, false, t32x64x64h) \
  instantiate_segments(codec, ext, type, 64, 64, 64, type, false, t64x64x64h) \
  instantiate_segments(codec, ext, type, 48, 128, 16, half, true, t48x128x16ah)

// Half-staging with float32 I/O (tag suffix fh). Activations and weights
// round through half in the threadgroup tiles; I/O rows, the simdgroup
// accumulators, and the output stay float32. This reproduces the DS4-c Metal
// MoE contract exactly (float device rows, half operand fragments, float
// accumulate, float output), so a caller whose rows are float32 can gate the
// half-precision-operand question without also rounding its I/O.
//
// t48x128x16ah is the device-A body with only the weight tile in half: the
// activation fragments still read float straight from device rows, and the
// B-tile load casts half back to float fragments, so the MMA itself is
// unchanged and the only numeric effect is the weight decode rounding
// through half (measured rel ~2e-4 against an f64 reference, no boundary
// anomalies). The win is the halved weight-tile footprint and staging
// traffic: 136.7-137.6 ms against the float default's 143.7-144.1 ms on the
// sorted MoE bulk-prefill pair (iq2_xxs gate/up plus q2_k down, S=23058,
// alternating matched arms). Selected only via KQ_SEG_TILE.
#define instantiate_segments_type_fhalf(codec, ext)                            \
  instantiate_segments(codec, ext, float, 64, 64, 32, half, false, t64x64x32fh) \
  instantiate_segments(codec, ext, float, 32, 64, 64, half, false, t32x64x64fh) \
  instantiate_segments(codec, ext, float, 64, 64, 64, half, false, t64x64x64fh) \
  instantiate_segments(codec, ext, float, 48, 128, 16, half, true, t48x128x16ah)

#define instantiate_segments_all(codec, ext)                     \
  instantiate_segments_type(codec, ext, float16_t)               \
  instantiate_segments_type(codec, ext, bfloat16_t)              \
  instantiate_segments_type(codec, ext, float)                   \
  instantiate_segments_type_half(codec, ext, float16_t)          \
  instantiate_segments_type_half(codec, ext, bfloat16_t)         \
  instantiate_segments_type_fhalf(codec, ext)

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
