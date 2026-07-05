// Descriptor-driven segmented quantized GEMM for sorted mixture-of-experts
// bulk prefill. Each threadgroup owns one (segment, n-tile) pair, reads its
// descriptor (expert, row_start, row_count) from the segments table, and walks
// its rows in chunks of BM. Weight tiles are dequantized into threadgroup
// tiles of the staging type StageT via the shared per-codec deq_chunk16 helper,
// the activation tile is copied into StageT, and the matmul runs on the steel
// BlockMMA with AccumType=float (simdgroup matrix, float32 accumulate). The
// output is cast to the store type T at store.
//
// StageT is the threadgroup tile precision, decoupled from the I/O type T.
// Default StageT=float decodes each weight to float32 before the multiply, the
// per-pair qmv/qmm decode contract; the simdgroup multiply-accumulate then runs
// entirely in float32. A float32 activation (T=float) also stages float, so it
// keeps full precision end to end, the variant the f32 down-projection seam
// needs. BlockMMA loads every simdgroup fragment as AccumType=float, so the
// matmul rate is the float32-simdgroup rate whatever StageT is (a float32
// simdgroup matmul measured 1.13x a float16 one at this shape, so the tile type
// is not the matmul cost); StageT only trades threadgroup footprint and
// staging-store bandwidth against decode precision.
//
// Codec is a traits struct (the same KqQ2_KExt / KqIq2_xxsExt structs the
// verify mat-vec kernels use) giving super-block geometry (superblock,
// block_bytes) and deq_chunk16(block, il, reg): dequantize the 16 contiguous
// weights of chunk il into a float4x4 in natural order [il*16, il*16+16). The
// weight for expert e, output feature n, input feature k lives at
//   w + ((e * N + n) * row_bytes) with row_bytes = (K / superblock) *
//   block_bytes
// and chunk ich = k / 16 covers super-block ich / (superblock/16) and the
// chunk-within cch = ich % (superblock/16).
//
// x is row-contiguous [S, K]; out is [S, N]. transpose=True only (each expert
// w is [N, K]).

// clang-format off
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

template <
    typename T,
    typename Codec,
    int BM = 32,
    int BN = 32,
    int BK = 32,
    int WM = 2,
    int WN = 2,
    typename StageT = float>
[[kernel]] void kq_gather_qmm_segments_impl(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* segments [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& S [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int TG_THREADS = WM * WN * 32;
  constexpr int chpb = Codec::superblock / 16; // 16-weight chunks per block
  // Pad the K stride to dodge threadgroup bank conflicts (matches qmm: one
  // 16-byte lane, so BK + 16/sizeof(StageT)). Deeper K tiles (BK >= 64) drop the
  // pad so a float32 staging pair still fits the 32 KiB threadgroup budget; half
  // staging has room to spare either way.
  constexpr int BK_padded = (BK >= 64) ? BK : (BK + 16 / sizeof(StageT));

  // Threadgroup tiles stage in StageT; BlockMMA loads each simdgroup fragment
  // as AccumType=float, so the multiply-accumulate is float32 regardless of the
  // tile type. The store type U is the I/O type T, so the epilogue casts the
  // float accumulator to T on write.
  using mma_t = mlx::steel::BlockMMA<
      /*T=*/StageT,
      /*U=*/T,
      BM,
      BN,
      BK,
      WM,
      WN,
      /*transpose_a=*/false,
      /*transpose_b=*/true,
      /*lda_tgp=*/BK_padded,
      /*ldb_tgp=*/BK_padded,
      /*AccumType=*/float>;

  // Threadgroup staging: activations [BM, BK_padded] then weights [BN,
  // BK_padded], both in the staging type StageT.
  threadgroup StageT Xs[BM * BK_padded];
  threadgroup StageT Ws[BN * BK_padded];

  // Descriptor for this threadgroup's segment: {expert, row_start, row_count}.
  const uint seg = tgpig.y;
  const uint expert = segments[seg * 3 + 0];
  const uint row_start = segments[seg * 3 + 1];
  const uint row_count = segments[seg * 3 + 2];
  const int n_tile = tgpig.x * BN; // first output feature this tile owns
  const short num_outs = min(BN, N - n_tile);

  const int nb = K / Codec::superblock; // super-blocks per weight row
  const int row_bytes = nb * Codec::block_bytes;
  // Base of this expert's weight matrix [N, K] wire bytes.
  const device uint8_t* w_expert =
      w + static_cast<int64_t>(expert) * N * row_bytes;

  // Walk this segment's rows in BM-row chunks. Each chunk is an independent
  // BM x BN x K GEMM tile.
  for (uint r0 = 0; r0 < row_count; r0 += BM) {
    const short num_rows = min((int)BM, (int)(row_count - r0));
    const device T* x_tile =
        x + static_cast<int64_t>(row_start + r0) * K;

    mma_t mma_op(simd_gid, simd_lid);

    for (int k0 = 0; k0 < K; k0 += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // --- Dequantize the weight tile [BN output rows, BK input cols] into Ws.
      // Each 16-wide chunk is dequantized by one thread; threads stride over
      // the BN*BK/16 chunks cooperatively. ---
      constexpr int wcols16 = BK / 16; // 16-weight chunks per row within BK
      constexpr int wchunks = BN * wcols16;
      for (int c = lid; c < wchunks; c += TG_THREADS) {
        const int wr = c / wcols16; // output-feature row within the tile
        const int wc16 = c % wcols16; // 16-wide chunk within the BK cols
        float4x4 reg;
        if (wr < num_outs) {
          const int n = n_tile + wr;
          const int k = k0 + wc16 * 16; // first input feature of this chunk
          const int ich = k / 16; // global 16-weight chunk index in the row
          const int ib = ich / chpb; // super-block index
          const int cch = ich % chpb; // chunk within the super-block
          const device uint8_t* block = w_expert +
              static_cast<int64_t>(n) * row_bytes +
              static_cast<int64_t>(ib) * Codec::block_bytes;
          Codec::deq_chunk16(block, cch, reg);
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            reg[i / 4][i % 4] = 0.0f;
          }
        }
        threadgroup StageT* dst = Ws + wr * BK_padded + wc16 * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          dst[i] = static_cast<StageT>(reg[i / 4][i % 4]);
        }
      }

      // --- Load the activation tile [BM rows, BK cols] into Xs (staging type
      // StageT). Out-of-range rows are zeroed. ---
      for (int e = lid; e < BM * BK; e += TG_THREADS) {
        const int xr = e / BK; // row within the BM tile
        const int xc = e % BK; // input feature within BK
        StageT v = StageT(0);
        if (xr < num_rows) {
          v = static_cast<StageT>(x_tile[static_cast<int64_t>(xr) * K + k0 + xc]);
        }
        Xs[xr * BK_padded + xc] = v;
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    }

    // --- Store this row chunk's outputs [num_rows, num_outs] to out. ---
    device T* y_tile =
        out + static_cast<int64_t>(row_start + r0) * N + n_tile;
    if (num_rows < BM || num_outs < BN) {
      mma_op.store_result_safe(y_tile, N, short2(num_outs, num_rows));
    } else {
      mma_op.store_result(y_tile, N);
    }
  }
}
// clang-format on
