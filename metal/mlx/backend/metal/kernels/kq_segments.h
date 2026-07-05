// Descriptor-driven segmented quantized GEMM for sorted mixture-of-experts
// bulk prefill. Each threadgroup owns one (segment, n-tile) pair, reads its
// descriptor (expert, row_start, row_count) from the segments table, and walks
// its rows in chunks of BM. Weight tiles are dequantized into threadgroup
// tiles of the staging type StageT via the shared per-codec deq_chunk16 helper
// and the matmul accumulates in float32 on simdgroup matrices. The output is
// cast to the store type T at store.
//
// StageT is the threadgroup tile precision, decoupled from the I/O type T.
// Default StageT=float decodes each weight to float32 before the multiply, the
// per-pair qmv/qmm decode contract; the simdgroup multiply-accumulate then runs
// entirely in float32. A float32 activation (T=float) also stages float, so it
// keeps full precision end to end, the variant the f32 down-projection seam
// needs. Every simdgroup fragment loads as float, so the matmul rate is the
// float32-simdgroup rate whatever StageT is (a float32 simdgroup matmul
// measured 1.13x a float16 one at this shape, so the tile type is not the
// matmul cost); StageT only trades threadgroup footprint and staging-store
// bandwidth against decode precision.
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
//
// Two GEMM bodies, selected per tile by the DEVA template flag:
//
//   * DEVA=0 (kq_segment_span_gemm): both operands staged through threadgroup
//     tiles and multiplied by the steel BlockMMA. The original body; the
//     legacy tiles and the half-staging opt-ins keep it.
//   * DEVA=1 (kq_segment_span_gemm_deva): only the weight tile is staged
//     (unpadded [BN, BK]); A fragments load straight from the device
//     activation rows through KqSegDevAMMA. The activation staging traffic,
//     its share of the threadgroup footprint, and the tile padding all
//     disappear, which lifts occupancy. Bit-identical to DEVA=0: the fragment
//     values are the same casts and tile_matmad runs the same call sequence,
//     so each output element accumulates in the same order. At the sorted MoE
//     bulk-prefill shape (S=23058, K=N=4096, E=256, iq2_xxs, f16 x) the
//     default deva tile measures 89.3-91.0 ms against 105.6-106.5 ms for the
//     best staged tile (alternating matched arms), with byte-equal outputs.
//
// Two entry points share the bodies, so any (expert, row range, n-tile)
// triple computes bit-identically whichever kernel dispatched it:
//   * kq_gather_qmm_segments_impl: grid (ceil(N/BN), T); each threadgroup reads
//     its {expert, row_start, row_count} descriptor from the host-built table.
//   * kq_gather_qmm_sorted_impl: grid (ceil(N/BN), E); no descriptor table.
//     The rows arrive pre-sorted by expert id (ascending, so equal ids are
//     contiguous) and each threadgroup derives its own row range with a binary
//     search over the device-resident sorted id array (lower_bound(e) and
//     lower_bound(e + 1)), then runs the same body. Threadgroups whose expert
//     owns no rows exit before touching x or w, so dispatching all E experts
//     costs only the search. Nothing about the segment structure ever reaches
//     the host, which is what lets a full prefill queue as one lazy graph.

// clang-format off
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"

// One segment x one n-tile GEMM: walk rows [row_start, row_start + row_count)
// in BM-row chunks against expert `expert`'s quantized [N, K] weight matrix,
// dequantizing weight tiles into the StageT threadgroup tiles and accumulating
// in float32 (see the header comment for the StageT contract).
template <
    typename T,
    typename Codec,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename StageT>
METAL_FUNC void kq_segment_span_gemm(
    const device uint8_t* w,
    const device T* x,
    device T* out,
    const int K,
    const int N,
    const uint expert,
    const uint row_start,
    const uint row_count,
    const int n_tile,
    threadgroup StageT* Xs,
    threadgroup StageT* Ws,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int TG_THREADS = WM * WN * 32;
  constexpr int chpb = Codec::superblock / 16; // 16-weight chunks per block
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

// Device-A variant of the steel BlockMMA: B fragments load from the staged
// threadgroup weight tile exactly as BlockMMA does, but A fragments load
// straight from the device activation rows, so no activation tile is staged.
// The fragment values are identical (the staged path writes x cast to the
// float tile and loads float; this path casts T to float in the fragment
// load, the same conversion) and tile_matmad runs the same call sequence, so
// outputs are bit-identical to the staged kernel. The weight tile is unpadded
// [BN, BK]: the B fragment reads showed no measurable bank penalty without
// the padding, and the smaller footprint (8 KB against the padded staged
// pair's 14 KB at the default tile) buys occupancy worth several ms at the
// bulk-prefill shape. Field and offset derivations mirror BlockMMA
// (transpose_a=false, transpose_b=true, AccumType=float).
template <typename T, int BM, int BN, int BK, int WM, int WN, typename StageT>
struct KqSegDevAMMA {
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = mlx::steel::BaseMMAFrag<float, 8, 8>;

  STEEL_CONST short TM_stride = kFragSize * WM;
  STEEL_CONST short TN_stride = kFragSize * WN;
  STEEL_CONST short TM = BM / (kFragSize * WM);
  STEEL_CONST short TN = BN / (kFragSize * WN);

  // Ws is unpadded [BN rows of n, BK cols of k] (transpose_b=true).
  STEEL_CONST short B_str_k = 1;
  STEEL_CONST short B_str_n = BK;
  STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

  mlx::steel::MMATile<float, TM, 1, MMAFrag_acc_t> Atile;
  mlx::steel::MMATile<float, 1, TN, MMAFrag_acc_t> Btile;
  mlx::steel::MMATile<float, TM, TN, MMAFrag_acc_t> Ctile;

  short sm;
  short sn;
  short Bs_offset;

  METAL_FUNC KqSegDevAMMA(ushort simd_group_id, ushort simd_lane_id) {
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n;

    sm += tm;
    sn += tn;
  }

  // One (BM, BK) x (BK, BN) step: A fragments from device rows x_tile
  // [num_rows, ld] at column offset k0 (rows past num_rows read as zero,
  // matching the staged tile's zero fill), B fragments from the staged
  // weight tile.
  METAL_FUNC void mma_device_a(
      const device T* x_tile,
      const int ld,
      const int k0,
      const short num_rows,
      const threadgroup StageT* Bs) {
    Bs += Bs_offset;

    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; ++i) {
        const short row = sm + i * TM_stride;
        thread auto& frag = Atile.frag_at(i, 0);
        if (row < num_rows) {
          const device T* src =
              x_tile + static_cast<int64_t>(row) * ld + k0 + kk + (sn % 8);
          frag[0] = static_cast<float>(src[0]);
          frag[1] = static_cast<float>(src[1]);
        } else {
          frag[0] = 0.0f;
          frag[1] = 0.0f;
        }
      }

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<StageT, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      mlx::steel::tile_matmad(Ctile, Atile, Btile, Ctile);

      Bs += tile_stride_b;
    }
  }

  METAL_FUNC void store_result(device T* D, const int ldd) {
    D += sm * ldd + sn;
    Ctile.template store<T, WM, WN>(D, ldd);
  }

  METAL_FUNC void
  store_result_safe(device T* D, const int ldd, short2 dst_tile_dims) {
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }
    Ctile.template store_safe<T, WM, WN>(D, ldd, dst_tile_dims);
  }
};

// Segment-span GEMM on the device-A MMA: the weight tile stages exactly as
// the synchronous kernel (same decode assignment, same per-thread chunk
// order) into the unpadded [BN, BK] layout; the activation tile is not
// staged at all. Bit-identical output (see KqSegDevAMMA).
template <
    typename T,
    typename Codec,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename StageT>
METAL_FUNC void kq_segment_span_gemm_deva(
    const device uint8_t* w,
    const device T* x,
    device T* out,
    const int K,
    const int N,
    const uint expert,
    const uint row_start,
    const uint row_count,
    const int n_tile,
    threadgroup StageT* Ws,
    uint lid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int TG_THREADS = WM * WN * 32;
  constexpr int chpb = Codec::superblock / 16;

  const short num_outs = min(BN, N - n_tile);
  const int nb = K / Codec::superblock;
  const int row_bytes = nb * Codec::block_bytes;
  const device uint8_t* w_expert =
      w + static_cast<int64_t>(expert) * N * row_bytes;

  for (uint r0 = 0; r0 < row_count; r0 += BM) {
    const short num_rows = min((int)BM, (int)(row_count - r0));
    const device T* x_tile = x + static_cast<int64_t>(row_start + r0) * K;

    KqSegDevAMMA<T, BM, BN, BK, WM, WN, StageT> mma_op(simd_gid, simd_lid);

    for (int k0 = 0; k0 < K; k0 += BK) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      constexpr int wcols16 = BK / 16;
      constexpr int wchunks = BN * wcols16;
      for (int c = lid; c < wchunks; c += TG_THREADS) {
        const int wr = c / wcols16;
        const int wc16 = c % wcols16;
        float4x4 reg;
        if (wr < num_outs) {
          const int n = n_tile + wr;
          const int k = k0 + wc16 * 16;
          const int ich = k / 16;
          const int ib = ich / chpb;
          const int cch = ich % chpb;
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
        threadgroup StageT* dst = Ws + wr * BK + wc16 * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          dst[i] = static_cast<StageT>(reg[i / 4][i % 4]);
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma_device_a(x_tile, K, k0, num_rows, Ws);
    }

    device T* y_tile =
        out + static_cast<int64_t>(row_start + r0) * N + n_tile;
    if (num_rows < BM || num_outs < BN) {
      mma_op.store_result_safe(y_tile, N, short2(num_outs, num_rows));
    } else {
      mma_op.store_result(y_tile, N);
    }
  }
}

template <
    typename T,
    typename Codec,
    int BM = 32,
    int BN = 32,
    int BK = 32,
    int WM = 2,
    int WN = 2,
    typename StageT = float,
    bool DEVA = false>
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
  constexpr int BK_padded = (BK >= 64) ? BK : (BK + 16 / sizeof(StageT));

  // Threadgroup staging. The staged body uses a padded activation tile and a
  // padded weight tile; the device-A body stages only the unpadded weight
  // tile, and Xs shrinks to a placeholder its branch never touches.
  threadgroup StageT Xs[DEVA ? 1 : BM * BK_padded];
  threadgroup StageT Ws[DEVA ? BN * BK : BN * BK_padded];

  // Descriptor for this threadgroup's segment: {expert, row_start, row_count}.
  const uint seg = tgpig.y;
  const uint expert = segments[seg * 3 + 0];
  const uint row_start = segments[seg * 3 + 1];
  const uint row_count = segments[seg * 3 + 2];
  const int n_tile = tgpig.x * BN; // first output feature this tile owns

  if constexpr (DEVA) {
    kq_segment_span_gemm_deva<T, Codec, BM, BN, BK, WM, WN, StageT>(
        w,
        x,
        out,
        K,
        N,
        expert,
        row_start,
        row_count,
        n_tile,
        Ws,
        lid,
        simd_gid,
        simd_lid);
  } else {
    kq_segment_span_gemm<T, Codec, BM, BN, BK, WM, WN, StageT>(
        w,
        x,
        out,
        K,
        N,
        expert,
        row_start,
        row_count,
        n_tile,
        Xs,
        Ws,
        lid,
        simd_gid,
        simd_lid);
  }
}

// First index in sorted_ids[0, S) whose value is >= key (S if none). All
// threads run the same search over the same device array, so control flow
// stays uniform and the touched cache lines are shared across the group.
METAL_FUNC uint kq_sorted_lower_bound(
    const device uint32_t* sorted_ids,
    const uint S,
    const uint32_t key) {
  uint lo = 0;
  uint hi = S;
  while (lo < hi) {
    const uint mid = lo + (hi - lo) / 2;
    if (sorted_ids[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <
    typename T,
    typename Codec,
    int BM = 32,
    int BN = 32,
    int BK = 32,
    int WM = 2,
    int WN = 2,
    typename StageT = float,
    bool DEVA = false>
[[kernel]] void kq_gather_qmm_sorted_impl(
    const device uint8_t* w [[buffer(0)]],
    const device T* x [[buffer(1)]],
    const device uint32_t* sorted_ids [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant int& S [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK >= 64) ? BK : (BK + 16 / sizeof(StageT));

  threadgroup StageT Xs[DEVA ? 1 : BM * BK_padded];
  threadgroup StageT Ws[DEVA ? BN * BK : BN * BK_padded];

  // This threadgroup's segment is the run of rows whose sorted id equals its
  // grid.y expert index; the ids are ascending so the run is exactly
  // [lower_bound(e), lower_bound(e + 1)).
  const uint expert = tgpig.y;
  const uint row_start = kq_sorted_lower_bound(sorted_ids, (uint)S, expert);
  const uint row_end = kq_sorted_lower_bound(sorted_ids, (uint)S, expert + 1);
  if (row_start >= row_end) {
    return; // uniform: no thread of this group has rows for this expert
  }
  const int n_tile = tgpig.x * BN;

  if constexpr (DEVA) {
    kq_segment_span_gemm_deva<T, Codec, BM, BN, BK, WM, WN, StageT>(
        w,
        x,
        out,
        K,
        N,
        expert,
        row_start,
        row_end - row_start,
        n_tile,
        Ws,
        lid,
        simd_gid,
        simd_lid);
  } else {
    kq_segment_span_gemm<T, Codec, BM, BN, BK, WM, WN, StageT>(
        w,
        x,
        out,
        K,
        N,
        expert,
        row_start,
        row_end - row_start,
        n_tile,
        Xs,
        Ws,
        lid,
        simd_gid,
        simd_lid);
  }
}
// clang-format on
