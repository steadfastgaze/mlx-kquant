// clang-format off
// Vector SDPA kernel instantiations for large head dims. Derived-code
// attribution lives in kq_sdpa.h and mlx_kquant/licenses/.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/loader.h"
#include "mlx/backend/metal/kernels/steel/attn/mma.h"
#include "mlx/backend/metal/kernels/kq_sdpa.h"

#define instantiate_kq_sdpa(type, partial_type, D)                    \
  instantiate_kernel(                                                 \
      "kq_sdpa_vector_2pass_1_" #type "_" #D,                         \
      kq_sdpa_vector_2pass_1,                                         \
      type,                                                           \
      partial_type,                                                   \
      D)                                                              \
  instantiate_kernel(                                                 \
      "kq_sdpa_vector_2pass_2_" #type "_" #D,                         \
      kq_sdpa_vector_2pass_2,                                         \
      type,                                                           \
      partial_type,                                                   \
      D)

// Float16 cannot represent every intermediate online-softmax partial at long
// context. Bfloat16 retains its compact partial buffer because its exponent
// range already covers those values.
instantiate_kq_sdpa(bfloat16_t, bfloat16_t, 256)
instantiate_kq_sdpa(bfloat16_t, bfloat16_t, 512)
instantiate_kq_sdpa(float16_t, float, 256)
instantiate_kq_sdpa(float16_t, float, 512)

#define instantiate_kq_sdpa_gqa(type, D, C)                           \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_1_" #type "_" #D "_c" #C,                    \
      kq_sdpa_gqa_2pass_1,                                            \
      type,                                                           \
      D,                                                              \
      C)

// NE (keys in flight per simdgroup) override: wide head dims halve NE so the
// per-lane query/output register slices stay at <= 64 floats.
#define instantiate_kq_sdpa_gqa_ne(type, D, C, NE)                    \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_1_" #type "_" #D "_c" #C,                    \
      kq_sdpa_gqa_2pass_1,                                            \
      type,                                                           \
      D,                                                              \
      C,                                                              \
      NE)

#define instantiate_kq_sdpa_gqa_merge(type, D)                        \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_2_" #type "_" #D,                            \
      kq_sdpa_gqa_2pass_2,                                            \
      type,                                                           \
      D)

// Exact-order split-128 merge for the fixed 16-head, head-dim-256 q8 decode
// geometry. Eight SIMD groups each own 32 output dimensions.
instantiate_kernel(
    "kq_sdpa_q8_merge_dim8", kq_sdpa_q8_merge_dim_parallel, 128, 256, 8)

// Tile C is bounded by threadgroup memory (2 * C * D * sizeof(T4)/4 bytes;
// 32 KB limit): D=64/128 stage 32 or 16 keys, D=256 stages 16 or 8.
instantiate_kq_sdpa_gqa(bfloat16_t, 64, 32)
instantiate_kq_sdpa_gqa(bfloat16_t, 64, 16)
instantiate_kq_sdpa_gqa(float16_t, 64, 32)
instantiate_kq_sdpa_gqa(float16_t, 64, 16)
instantiate_kq_sdpa_gqa(bfloat16_t, 128, 32)
instantiate_kq_sdpa_gqa(bfloat16_t, 128, 16)
instantiate_kq_sdpa_gqa(float16_t, 128, 32)
instantiate_kq_sdpa_gqa(float16_t, 128, 16)
instantiate_kq_sdpa_gqa(bfloat16_t, 256, 16)
instantiate_kq_sdpa_gqa(bfloat16_t, 256, 8)
instantiate_kq_sdpa_gqa(float16_t, 256, 16)
instantiate_kq_sdpa_gqa(float16_t, 256, 8)
instantiate_kq_sdpa_gqa_ne(bfloat16_t, 512, 8, 2)
instantiate_kq_sdpa_gqa_ne(float16_t, 512, 8, 2)

// Verify-width (qL 2..4) pair variants: two queries per simdgroup, so each
// staged element read feeds two dots (threadgroup-memory traffic halves) at
// two query/output register sets per thread. NE mirrors the qL==1 choice.
#define instantiate_kq_sdpa_gqa_p2(type, D, C, NE)                    \
  instantiate_kernel(                                                 \
      "kq_sdpa_gqa_2pass_1_" #type "_" #D "_c" #C "_p2",              \
      kq_sdpa_gqa_2pass_1,                                            \
      type,                                                           \
      D,                                                              \
      C,                                                              \
      NE,                                                             \
      2)

instantiate_kq_sdpa_gqa_p2(bfloat16_t, 512, 8, 2)
instantiate_kq_sdpa_gqa_p2(float16_t, 512, 8, 2)
instantiate_kq_sdpa_gqa_p2(bfloat16_t, 256, 16, 4)
instantiate_kq_sdpa_gqa_p2(float16_t, 256, 16, 4)
instantiate_kq_sdpa_gqa_p2(bfloat16_t, 128, 32, 4)
instantiate_kq_sdpa_gqa_p2(float16_t, 128, 32, 4)
instantiate_kq_sdpa_gqa_p2(bfloat16_t, 64, 32, 4)
instantiate_kq_sdpa_gqa_p2(float16_t, 64, 32, 4)

// Simdgroup-matrix FA verify pass 1 (folded GQA, one 32-row Q tile); the
// merge reuses kq_sdpa_gqa_2pass_2. head_dim 256 only for now: at 512 the
// Q + O fragment sets are ~256 floats/thread, so a 512 instantiation needs a
// BD-chunked (two-sweep) output accumulator first.
#define instantiate_kq_sdpa_fa_verify(type, D)                         \
  instantiate_kernel(                                                  \
      "kq_sdpa_fa_verify_2pass_1_" #type "_" #D,                       \
      kq_sdpa_fa_verify_2pass_1,                                       \
      type,                                                            \
      D)

instantiate_kq_sdpa_fa_verify(bfloat16_t, 256)
instantiate_kq_sdpa_fa_verify(float16_t, 256)

// Simdgroup-matrix FA prefill pass 1 (dense KV): the fa_verify tile plus an
// outer query-tile loop on the grid. QW is the query positions folded per tile
// (G * QW == 32); the merge reuses kq_sdpa_gqa_2pass_2 with grid (Hq, B, qL).
// float variant carries the served float32 query/key path; the half variants
// support the unit fences and mixed-dtype experiments.
#define instantiate_kq_sdpa_fa_prefill(type, D, QW, BK)                 \
  instantiate_kernel(                                                   \
      "kq_sdpa_fa_prefill_2pass_1_" #type "_" #D "_q" #QW,             \
      kq_sdpa_fa_prefill_2pass_1,                                       \
      type,                                                             \
      D,                                                                \
      QW,                                                               \
      BK)

// float32 K/V staging is 4 bytes/element, so BK=16 keeps the threadgroup tile
// under the 32 KB limit at D=256; the half-precision paths stage BK=32.
instantiate_kq_sdpa_fa_prefill(float, 256, 2, 16)
instantiate_kq_sdpa_fa_prefill(float, 256, 4, 16)
instantiate_kq_sdpa_fa_prefill(float, 256, 8, 16)
instantiate_kq_sdpa_fa_prefill(bfloat16_t, 256, 2, 32)
instantiate_kq_sdpa_fa_prefill(bfloat16_t, 256, 4, 32)
instantiate_kq_sdpa_fa_prefill(bfloat16_t, 256, 8, 32)
instantiate_kq_sdpa_fa_prefill(float16_t, 256, 2, 32)
instantiate_kq_sdpa_fa_prefill(float16_t, 256, 4, 32)
instantiate_kq_sdpa_fa_prefill(float16_t, 256, 8, 32)

// FA prefill pass 1, q8 past phase (the serving form): float32 queries and
// accumulators, past prefix dequantized from the QuantizedKVCache tuple into
// StageT threadgroup staging during load, dense self chunk cast alongside.
// The name carries the staging type; queries are always float32. The float
// staging instantiation (BK=16) is the full-precision verification arm and the
// memory-lever fallback form.
#define instantiate_kq_sdpa_fa_prefill_q8(stype, D, QW, BK, BQ)         \
  instantiate_kernel(                                                   \
      "kq_sdpa_fa_prefill_q8_2pass_1_" #stype "_" #D "_q" #QW "_b" #BQ  \
      "_k" #BK,                                                         \
      kq_sdpa_fa_prefill_q8_2pass_1,                                    \
      stype,                                                            \
      D,                                                                \
      QW,                                                               \
      BK,                                                               \
      BQ)

// BQ=64 (256-thread threadgroup) folds twice the query positions onto each
// staged KV tile and is the serving width for the 16:2 GQA, head-dim-256
// serving geometry (G=8, QW=8); the BQ=32 forms cover the other GQA folds and
// the verification arm.
// BK=48 stages half again more keys per tile within the 32 KB threadgroup
// budget (half staging only), cutting the per-tile barrier count at depth.
instantiate_kq_sdpa_fa_prefill_q8(bfloat16_t, 256, 2, 32, 32)
instantiate_kq_sdpa_fa_prefill_q8(bfloat16_t, 256, 4, 32, 32)
instantiate_kq_sdpa_fa_prefill_q8(bfloat16_t, 256, 8, 32, 32)
instantiate_kq_sdpa_fa_prefill_q8(bfloat16_t, 256, 8, 32, 64)
instantiate_kq_sdpa_fa_prefill_q8(bfloat16_t, 256, 8, 48, 64)
instantiate_kq_sdpa_fa_prefill_q8(float16_t, 256, 2, 32, 32)
instantiate_kq_sdpa_fa_prefill_q8(float16_t, 256, 4, 32, 32)
instantiate_kq_sdpa_fa_prefill_q8(float16_t, 256, 8, 32, 32)
instantiate_kq_sdpa_fa_prefill_q8(float16_t, 256, 8, 32, 64)
instantiate_kq_sdpa_fa_prefill_q8(float16_t, 256, 8, 48, 64)
// The float32 staging form pins BK=16: BK=32 at float32 needs 36 KB of
// threadgroup memory, over the 32 KB budget. So BQ is the only fold, and both
// widths stage the same 16-key tiles. BQ=64 folds twice the query positions
// onto each tile, halving the per-row staging and dequant work.
instantiate_kq_sdpa_fa_prefill_q8(float, 256, 4, 16, 32)
instantiate_kq_sdpa_fa_prefill_q8(float, 256, 8, 16, 64)

// Fused q8 decode pass 1 (the serving form of the KV-attention read): float32
// queries and accumulators, the whole q8 cache dequantized from the tuple into
// StageT threadgroup staging during load, one query row folded as BQ heads. The
// name carries the staging type; the float staging is the served full-precision
// form and the dequant-exactness verification arm. BQ is the GQA fold (8 for
// the 16:2 GQA, head-dim-256 serving geometry); the merge reuses
// kq_sdpa_gqa_2pass_2 on the float output.
#define instantiate_kq_sdpa_decode_q8(stype, D, BQ)                     \
  instantiate_kernel(                                                   \
      "kq_sdpa_decode_q8_2pass_1_" #stype "_" #D "_b" #BQ,             \
      kq_sdpa_decode_q8_2pass_1,                                        \
      stype,                                                            \
      D,                                                                \
      BQ)

instantiate_kq_sdpa_decode_q8(float, 256, 8)
instantiate_kq_sdpa_decode_q8(bfloat16_t, 256, 8)
instantiate_kq_sdpa_decode_q8(float16_t, 256, 8)

// SIMD-shuffle q8 decode pass 1 (the decode-latency form): float32 queries and
// staging, the whole q8 cache dequantized during the cooperative tile load, the
// GQA group folded per threadgroup. C is the staged tile height (float4 staging
// is 16 bytes per element, so C=8 stages 16 KB and C=16 stages 32 KB at D=256);
// NE is the keys in flight per simdgroup. C=16 has a scalar control and an
// aligned uint4 loaders; both retain the array-provided sequence strides. The
// default aligned arm converts each packed word through uchar4 while the
// control retains scalar byte extraction. The merge reuses
// kq_sdpa_gqa_2pass_2 on the float output.
#define instantiate_kq_sdpa_decode_gqa_q8(D, C, NE)                     \
  instantiate_kernel(                                                   \
      "kq_sdpa_decode_gqa_q8_2pass_1_" #D "_c" #C "_ne" #NE,           \
      kq_sdpa_decode_gqa_q8_2pass_1,                                    \
      D,                                                                \
      C,                                                                \
      NE)

#define instantiate_kq_sdpa_decode_gqa_q8_loader(                       \
    D, C, NE, uint4_load, vector_byte_unpack, suffix)                   \
  instantiate_kernel(                                                   \
      "kq_sdpa_decode_gqa_q8_2pass_1_" #D "_c" #C "_ne" #NE           \
      "_" #suffix,                                                     \
      kq_sdpa_decode_gqa_q8_2pass_1,                                    \
      D,                                                                \
      C,                                                                \
      NE,                                                               \
      uint4_load,                                                       \
      vector_byte_unpack)

instantiate_kq_sdpa_decode_gqa_q8(256, 8, 4)
instantiate_kq_sdpa_decode_gqa_q8_loader(
    256, 16, 4, false, false, scalar_dynamic)
instantiate_kq_sdpa_decode_gqa_q8_loader(
    256, 16, 4, true, false, uint4_dynamic)
instantiate_kq_sdpa_decode_gqa_q8_loader(
    256, 16, 4, true, true, uint4_byte_dynamic)

instantiate_kq_sdpa_gqa_merge(bfloat16_t, 64)
instantiate_kq_sdpa_gqa_merge(float16_t, 64)
instantiate_kq_sdpa_gqa_merge(bfloat16_t, 128)
instantiate_kq_sdpa_gqa_merge(float16_t, 128)
instantiate_kq_sdpa_gqa_merge(bfloat16_t, 256)
instantiate_kq_sdpa_gqa_merge(float16_t, 256)
instantiate_kq_sdpa_gqa_merge(float, 256)
instantiate_kq_sdpa_gqa_merge(bfloat16_t, 512)
instantiate_kq_sdpa_gqa_merge(float16_t, 512)
    // clang-format on
