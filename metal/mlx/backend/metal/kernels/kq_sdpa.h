// Vector scaled-dot-product-attention kernels for large head dims (e.g. 512)
// that stock MLX's vector allowlist {64,96,128,256} excludes, forcing a
// materialized fallback. Derived from MLX's sdpa_vector_2pass (MIT); the
// query-transposed path is dropped (callers route a row-contiguous query). K
// and V are read in place via their head/seq strides so a strided KV-cache
// prefix needs no copy. An optional boolean column mask (broadcastable over
// batch/head/query via zero strides) gates keys in pass 1, and optional
// per-query-head attention sinks join the softmax denominator in pass 2.
//
// Two passes: pass 1 splits the keys into `blocks` chunks, each chunk computing
// a partial online-softmax output + running max + running sum; pass 2 reduces
// the per-chunk partials into the final output. `do_causal`, `blocks`,
// `has_mask` and `has_sinks` are Metal function constants so the key-stride
// loop specializes at pipeline build.

constant bool do_causal [[function_constant(0)]];
constant int blocks [[function_constant(1)]];
constant int gqa_splits [[function_constant(2)]];
constant bool gqa_has_sinks [[function_constant(3)]];
constant bool has_mask [[function_constant(4)]];
constant bool has_sinks [[function_constant(5)]];

template <typename T, typename PT, int D, int V = D>
[[kernel]] void kq_sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device PT* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const device bool* mask [[buffer(12), function_constant(has_mask)]],
    const constant size_t& m_batch_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant size_t& m_head_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant size_t& m_q_stride
    [[buffer(15), function_constant(has_mask)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // One threadgroup per (kv-head, batch, kv-block); the GQA group + query seq
  // are the threadgroup y,z dims so they share each block's K/V read.
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

  if (has_mask) {
    mask += batch_idx * m_batch_stride + q_head_idx * m_head_stride +
        q_seq_idx * m_q_stride;
  }
  queries += o_offset * D + simd_lid * qk_per_thread;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    }
    if (has_mask) {
      use_key = use_key && mask[i];
    }
    if (use_key) {
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * static_cast<U>(keys[j]);
      }
      score = simd_sum(score);
      U new_max = max(max_score, score);
      // fast::exp is safe here: both arguments are non-positive by the
      // running-max construction, so the exponent cannot overflow to +Inf;
      // the underflow edge returns 0.
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
  }

  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<PT>(o[i]);
  }
}

template <typename T, typename PT, int D>
[[kernel]] void kq_sdpa_vector_2pass_2(
    const device PT* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device float* sinks [[buffer(4), function_constant(has_sinks)]],
    const constant int& num_q_heads [[buffer(5), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  U sum_exp_score = 0;
  U max_score = Limits<U>::finite_min;

  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);
  // The sink is one extra softmax logit per query head with no value row: it
  // joins the global max before the block factors are formed (so a dominant
  // sink rescales every partial) and adds one term to the reduced denominator.
  U sink = Limits<U>::finite_min;
  if (has_sinks) {
    sink = static_cast<U>(sinks[head_idx % num_q_heads]);
    max_score = max(max_score, sink);
  }

  // fast::exp arguments below are non-positive (max_score majorizes every
  // per-block max and the sink), so no exponent can overflow to +Inf.
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  if (has_sinks) {
    sum_exp_score += fast::exp(sink - max_score);
  }

  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// Decode-time (qL == 1) GQA attention for head dims whose stock vector path
// leaves bandwidth on the table at long KV. Differs from kq_sdpa_vector_2pass
// in three ways: the key axis is split into `gqa_splits` CONTIGUOUS chunks
// (few, coarse partials; merge cost stays constant with depth); each chunk is
// streamed through threadgroup-staged K/V tiles shared by the whole GQA group
// (device reads KV once per kv-head instead of once per q-head); and scores
// run NE keys in flight per simdgroup with a log2(32/NE)-step lane reduction
// instead of a full simd_sum per key. Optional per-q-head attention sinks
// (an extra softmax logit with no value row) fold into the pass-2 denominator.
// Partials are float32.
//
// Pass-1 threadgroup: (32, gqa_factor, ceil(q_len / QPS)); one simdgroup per
// q-head x QPS-query group. Grid: (n_kv_heads, B, gqa_splits). q_len is 1 at
// decode and 2..4 at speculative-verify width: every simdgroup of the group
// shares each staged K/V tile, so device KV traffic stays one sweep per
// kv-head regardless of query count. QPS is the compile-time query count per
// simdgroup: each staged element is read from threadgroup memory once and
// dotted against QPS query slices, dividing the threadgroup-memory traffic
// (the qL>1 bottleneck) by QPS at a cost of QPS query/output register sets
// (QPS=2 at D=512 is ~140 floats/thread; QPS=4 spills). Verify queries are
// the sequence's trailing positions, causally clamped
// (key <= N - q_len + query index). Requires gqa_factor * ceil(q_len / QPS)
// <= 32 (1024-thread threadgroup) and gqa_splits <= 128 (pass-2 scratch).

template <typename T, int D, int C = 32, int NE = 4, int QPS = 1>
[[kernel]] void kq_sdpa_gqa_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& q_len [[buffer(12)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int D4 = D / 4;
  constexpr int NL = 32 / NE; // lanes per in-flight key
  constexpr int DP4 = D4 / NL; // float4s per lane per key row
  using T4 = metal::vec<T, 4>;

  threadgroup T4 sK[C * D4];
  threadgroup T4 sV[C * D4];

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int split_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int nq = q_len; // runtime query count (tptg.z * QPS >= nq)
  const int qz0 = tidtg.z * QPS; // first query of this simdgroup
  const int lane = tidtg.x;
  const int tx = lane % NL;
  const int ty = lane / NL;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;

  // Contiguous, C-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * C - 1) / (gqa_splits * C)) * C;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const device T* kbase =
      keys + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride;
  const device T* vbase =
      values + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * v_head_stride;

  // Pre-scaled query slices for this lane's key-row columns
  // ([B, Hq, q_len, D], row-contiguous). A simdgroup past the runtime query
  // count (odd q_len at QPS=2) zero-fills; its lanes compute but never write.
  const device T4* q4 =
      (const device T4*)(queries + ((size_t)q_batch_head_idx * nq + qz0) * D);
  float4 qf[QPS][DP4];
  int lim[QPS]; // highest key each query may attend (its causal position)
  for (short p = 0; p < QPS; p++) {
    const bool active = qz0 + p < nq;
    for (short ii = 0; ii < DP4; ii++) {
      qf[p][ii] = active ? scale * float4(q4[(size_t)p * D4 + ii * NL + tx])
                         : float4(0);
    }
    lim[p] = active ? N - nq + qz0 + p : -1;
  }

  float max_score[QPS];
  float sum_exp_score[QPS];
  float4 lo[QPS][DP4];
  for (short p = 0; p < QPS; p++) {
    max_score[p] = Limits<float>::finite_min;
    sum_exp_score[p] = 0;
    for (short ii = 0; ii < DP4; ii++) {
      lo[p][ii] = 0;
    }
  }

  const int flat = (tidtg.z * gqa_factor + tidtg.y) * 32 + lane;
  const int n_threads = 32 * gqa_factor * tptg.z;

  for (int kt = k0; kt < k1; kt += C) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative tile load; zero-fill the tail so stale threadgroup data
    // can never reach the accumulators.
    for (int i = flat; i < C * D4; i += n_threads) {
      const int row = i / D4;
      const int col = i % D4;
      const int kg = kt + row;
      if (kg < k1) {
        sK[i] = ((const device T4*)(kbase + (size_t)kg * k_seq_stride))[col];
        sV[i] = ((const device T4*)(vbase + (size_t)kg * v_seq_stride))[col];
      } else {
        sK[i] = T4(T(0));
        sV[i] = T4(T(0));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: NE keys in flight, NL lanes per key, then reduce + broadcast.
    // Each staged element is read once and dotted against all QPS queries.
    float mqk[QPS][C / NE];
    float m_tile[QPS];
    for (short p = 0; p < QPS; p++) {
      m_tile[p] = Limits<float>::finite_min;
    }
    for (short cc = 0; cc < C / NE; cc++) {
      float s[QPS];
      for (short p = 0; p < QPS; p++) {
        s[p] = 0;
      }
      for (short ii = 0; ii < DP4; ii++) {
        const float4 kk = float4(sK[(cc * NE + ty) * D4 + ii * NL + tx]);
        for (short p = 0; p < QPS; p++) {
          s[p] += dot(kk, qf[p][ii]);
        }
      }
      const int kg = kt + cc * NE + ty;
      for (short p = 0; p < QPS; p++) {
        for (short off = NL / 2; off > 0; off >>= 1) {
          s[p] += simd_shuffle_down(s[p], off);
        }
        s[p] = simd_shuffle(s[p], NL * ty);
        const bool valid = kg < k1 && kg <= lim[p];
        mqk[p][cc] = valid ? s[p] : Limits<float>::finite_min;
        m_tile[p] = max(m_tile[p], mqk[p][cc]);
      }
    }

    // Online softmax per query; each lane sums its ty-group's keys, so the
    // simd_sum counts every key NL times. A tile entirely beyond a query's
    // causal limit is skipped outright: with the running max still
    // finite_min, exp(finite_min - finite_min) == 1 would poison the sum
    // (can only happen at verify width; a decode query attends every key).
    float vs[QPS][C / NE];
    for (short p = 0; p < QPS; p++) {
      m_tile[p] = simd_max(m_tile[p]);
      if (m_tile[p] > Limits<float>::finite_min) {
        const float new_max = max(max_score[p], m_tile[p]);
        const float factor = fast::exp(max_score[p] - new_max);
        float vsum = 0;
        for (short cc = 0; cc < C / NE; cc++) {
          vs[p][cc] = fast::exp(mqk[p][cc] - new_max);
          vsum += vs[p][cc];
        }
        sum_exp_score[p] =
            sum_exp_score[p] * factor + simd_sum(vsum) * (1.0f / NL);
        max_score[p] = new_max;
        for (short ii = 0; ii < DP4; ii++) {
          lo[p][ii] *= factor;
        }
      } else {
        for (short cc = 0; cc < C / NE; cc++) {
          vs[p][cc] = 0;
        }
      }
    }

    for (short cc = 0; cc < C / NE; cc++) {
      for (short ii = 0; ii < DP4; ii++) {
        const float4 vv = float4(sV[(cc * NE + ty) * D4 + ii * NL + tx]);
        for (short p = 0; p < QPS; p++) {
          lo[p][ii] += vv * vs[p][cc];
        }
      }
    }
  }

  // Cross-ty reduction of the deferred output accumulators; the ty == 0 lane
  // group holds the chunk totals.
  for (short p = 0; p < QPS; p++) {
    if (qz0 + p >= nq) {
      continue;
    }
    for (short ii = 0; ii < DP4; ii++) {
      for (short off = (NE / 2) * NL; off >= NL; off >>= 1) {
        lo[p][ii][0] += simd_shuffle_down(lo[p][ii][0], off);
        lo[p][ii][1] += simd_shuffle_down(lo[p][ii][1], off);
        lo[p][ii][2] += simd_shuffle_down(lo[p][ii][2], off);
        lo[p][ii][3] += simd_shuffle_down(lo[p][ii][3], off);
      }
    }

    const size_t po =
        (((size_t)q_batch_head_idx * nq + qz0 + p) * gqa_splits + split_idx);
    if (ty == 0) {
      device float4* out4 = (device float4*)(out + po * D);
      for (short ii = 0; ii < DP4; ii++) {
        out4[ii * NL + tx] = lo[p][ii];
      }
    }
    if (lane == 0) {
      sums[po] = sum_exp_score[p];
      maxs[po] = max_score[p];
    }
  }
}

// Row-wise reduce/broadcast ops for the steel MMATile helpers below.
struct KQMaxOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return metal::max(x, y);
  }
};

struct KQSumOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x + y;
  }
};

struct KQMulOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x * y;
  }
};

struct KQExpSubOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return fast::exp2(x - y);
  }
};

// Simdgroup-matrix (steel MMA) speculative-verify attention, pass 1. The
// caller folds the GQA group into the query rows -- q [B, Hq, qL, D] becomes
// [B, Hkv, G*qL, D] with kv-major heads -- so the kernel sees an MHA problem
// whose n_rows = G*qL <= 32 queries fill exactly one BQ=32 tile, held in
// per-thread fragments (each thread owns one row of every 8x8 fragment, so
// the online-softmax row max/sum live in registers with no threadgroup
// round-trips). Grid (n_kv_heads, B, gqa_splits): each threadgroup streams
// its contiguous key chunk once through threadgroup-staged K/V tiles (one
// shared buffer, steel style), computing S = Q @ K^T and O += P @ V on
// simdgroup_matrix with float32 accumulators.
//
// Each folded row is causally clamped to key <= kL - qL + (row % qL), with qL
// a runtime buffer param. Only tiles reaching past kL - qL or the split tail
// take the mask branch; split-interior tiles run mask-free (verify rows share
// the whole prefix). Scores run in exp2 space (scale premultiplied by
// log2(e)); the P values and row sums are base-independent (2^(log2(e)*x) =
// e^x), so only the stored row max converts back to natural log and the
// partials [B, Hkv, n_rows, gqa_splits, D] merge through kq_sdpa_gqa_2pass_2
// unchanged. A tile entirely past a row's limit leaves the row's running max
// at finite_min and zeroes its P row (exp2(0) == 1 would otherwise poison the
// sum); an empty split writes (O = 0, sum = 0, max = finite_min) partials
// that merge with weight zero.
template <typename T, int D>
[[kernel]] void kq_sdpa_fa_verify_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& q_len [[buffer(12)]],
    const constant int& n_rows [[buffer(13)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int BQ = 32; // query rows (the whole fold, zero-padded)
  constexpr int BK = 32; // keys staged per tile
  constexpr int kNWarps = BQ / 8; // one 8-row fragment strip per simdgroup
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short TD = D / kFragSize;
  constexpr short kPad = 16 / sizeof(T);
  constexpr short LDK = BK + kPad; // Ks staged transposed [D][BK + pad]
  constexpr short LDV = D + kPad; // Vs staged row-major [BK][D + pad]
  constexpr int kSmemKV = (LDK * D > BK * LDV) ? LDK * D : BK * LDV;

  using MMAFrag_t = mlx::steel::BaseMMAFrag<float, kFragSize, kFragSize>;
  using KLoader = mlx::steel::BlockLoaderT<T, BK, D, 1, LDK, 0, kNWarps * 32>;
  using VLoader = mlx::steel::BlockLoaderT<T, BK, D, LDV, 1, 0, kNWarps * 32>;

  // K and V share the buffer (steel pattern): the S matmul consumes Ks
  // before the V load overwrites it, halving the threadgroup footprint.
  threadgroup T KV_smem[kSmemKV];

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int split_idx = tid.z;
  const int num_kv_heads = tpg.x;
  const int q_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

  // Contiguous, BK-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const device T* kbase = keys +
      (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride +
      (size_t)k0 * k_seq_stride;
  const device T* vbase = values +
      (size_t)(batch_idx * num_kv_heads + kv_head_idx) * v_head_stride +
      (size_t)k0 * v_seq_stride;

  KLoader loader_k(
      kbase, static_cast<int>(k_seq_stride), KV_smem, simd_gid, simd_lid);
  VLoader loader_v(
      vbase, static_cast<int>(v_seq_stride), KV_smem, simd_gid, simd_lid);

  // Fragment coordinates: this thread owns row (row0 + sm) and the column
  // pair at sn of every 8x8 fragment.
  const short2 sc = MMAFrag_t::get_coord(simd_lid);
  const short sm = sc.y;
  const short sn = sc.x;
  const int row = int(simd_gid) * kFragSize + sm;
  // Highest key this row attends. Padding rows (row >= n_rows) compute a
  // harmless in-range limit; their partials are never written.
  const int lim = N - q_len + (row % q_len);
  const int lim_min = N - q_len; // every real row attends at least this far

  // Q tile in float32 fragments (one device read; rows past n_rows
  // zero-fill, so padding rows score 0 everywhere).
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Qtile;
  {
    const device T* qrow =
        queries + ((size_t)q_batch_head_idx * n_rows + row) * D + sn;
    Qtile.template load_safe<T, 1, 1>(qrow, D, short2(D - sn, n_rows - row));
  }

  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Stile;
  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Ktile;
  mlx::steel::MMATile<float, 1, 1, MMAFrag_t> Vtile;
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Otile;
  Otile.clear();

  // exp2-space online softmax (steel): scores carry scale * log2(e).
  const float scale2 = scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0;

  for (int kt = k0; kt < k1; kt += BK) {
    const int krem = k1 - kt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (krem < BK) {
      loader_k.load_safe(short2(D, krem));
    } else {
      loader_k.load_unsafe();
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T, one 8-deep head-dim slab at a time.
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<T, 1, 1, LDK, 1>(
          &KV_smem[(dd * kFragSize + sm) * LDK + sn]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    // Scale in float32, then mask. Only the split tail or a tile reaching
    // past kL - qL can mask anything; interior tiles skip the branch.
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }
    if (krem < BK || kt + BK - 1 > lim_min) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int kg = kt + ik * kFragSize + sn;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          if (kg + jj >= k1 || kg + jj > lim) {
            Stile.frag_at(0, ik)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (krem < BK) {
      loader_v.load_safe(short2(D, krem));
    } else {
      loader_v.load_unsafe();
    }

    // Online softmax on this thread's row (registers only, overlapping the
    // V load). A row with no valid key yet keeps max at finite_min and
    // zeroes its P row instead of exponentiating.
    float new_max = max_score;
    Stile.template row_reduce<KQMaxOp>(&new_max);
    if (new_max > Limits<float>::finite_min) {
      Stile.template row_bin_op<KQExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      float tile_sum = 0;
      Stile.template row_reduce<KQSumOp>(&tile_sum);
      sum_score = sum_score * factor + tile_sum;
      max_score = new_max;
      Otile.template row_bin_op<KQMulOp>(&factor);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        Vtile.template load<T, 1, 1, LDV, 1>(
            &KV_smem[(ik * kFragSize + sm) * LDV + id * kFragSize + sn]);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }

    loader_k.next();
    loader_v.next();
  }

  // Unnormalized partials in the kq_sdpa_gqa_2pass_1 layout. Each row's four
  // owner threads hold the reduced row stats after the shuffle reductions;
  // the sn == 0 owner writes them, with the max converted to natural log for
  // the shared merge.
  if (row < n_rows) {
    const size_t po =
        ((size_t)q_batch_head_idx * n_rows + row) * gqa_splits + split_idx;
    device float* orow = out + po * D + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        orow[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
      }
    }
    if (sn == 0) {
      sums[po] = sum_score;
      maxs[po] = max_score == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score * M_LN2_F;
    }
  }
}

// Simdgroup-matrix (steel MMA) prefill attention, pass 1, dense KV. Extends the
// fa_verify tile to a full prefill chunk by tiling the query axis on the grid:
// the verify kernel packs one BQ=32 fold of G*qL <= 32 rows, this kernel walks
// ceil(qL / QW) query tiles, each a fold of the whole GQA group (G heads) with
// QW consecutive query positions (G * QW == BQ == 32). Row r of a tile maps to
// query head kv_head*G + r/QW and query position qtile*QW + r%QW; queries stay
// in their natural [B, Hq, qL, D] layout and each row loads its own device
// slice, so no host fold is needed. Grid (n_kv_heads, n_query_tiles,
// gqa_splits): each threadgroup streams its contiguous key chunk once through
// threadgroup-staged K/V tiles, computing S = Q @ K^T and O += P @ V on
// simdgroup_matrix with float32 accumulators, one query tile against one KV
// head and key split.
//
// Causal mask, prefill form. The chunk's qL queries occupy the trailing rows of
// the depth-N cache: query position p (0-based within the chunk) attends keys
// <= (N - qL) + p, so past-prefix keys are always in range and the self-chunk
// keys are causal. This is the single band the composed prefill path applies.
// Only tiles reaching past N - qL or the split tail take the mask branch.
// exp2-space online softmax as in fa_verify; partials merge through
// kq_sdpa_gqa_2pass_2 with grid (Hq, B, qL) and n_q_heads = Hq, so the output
// is the natural [B, Hq, qL, D].
//
// Dense KV only: both the past prefix and the self chunk are read from one
// contiguous key array. The q8 past-phase variant streams the past prefix from
// the quantized cache instead (kq_sdpa_fa_prefill_q8_2pass_1).
//
// BK is the keys staged per tile. A float32 K/V path stages 4 bytes per
// element, so BK=32 at D=256 overflows the 32 KB threadgroup limit (LDK*D*4 =
// 36 KB); the float instantiation uses BK=16 (LDK*D*4 = 20 KB). Half-precision
// paths keep BK=32.
template <typename T, int D, int QW, int BK = 32>
[[kernel]] void kq_sdpa_fa_prefill_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant size_t& k_head_stride [[buffer(7)]],
    const constant size_t& k_seq_stride [[buffer(8)]],
    const constant size_t& v_head_stride [[buffer(9)]],
    const constant size_t& v_seq_stride [[buffer(10)]],
    const constant float& scale [[buffer(11)]],
    const constant int& q_len [[buffer(12)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int BQ = 32; // query rows per tile (G * QW)
  constexpr int GQ = BQ / QW; // GQA heads folded into one tile
  constexpr int kNWarps = BQ / 8;
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short TD = D / kFragSize;
  constexpr short kPad = 16 / sizeof(T);
  constexpr short LDK = BK + kPad; // Ks staged transposed [D][BK + pad]
  constexpr short LDV = D + kPad; // Vs staged row-major [BK][D + pad]
  constexpr int kSmemKV = (LDK * D > BK * LDV) ? LDK * D : BK * LDV;

  using MMAFrag_t = mlx::steel::BaseMMAFrag<float, kFragSize, kFragSize>;
  using KLoader = mlx::steel::BlockLoaderT<T, BK, D, 1, LDK, 0, kNWarps * 32>;
  using VLoader = mlx::steel::BlockLoaderT<T, BK, D, LDV, 1, 0, kNWarps * 32>;

  threadgroup T KV_smem[kSmemKV];

  const int kv_head_idx = tid.x;
  const int qtile_idx = tid.y;
  const int split_idx = tid.z;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * GQ;
  const int q0 = qtile_idx * QW; // first query position of this tile

  // Contiguous, BK-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const device T* kbase =
      keys + (size_t)kv_head_idx * k_head_stride + (size_t)k0 * k_seq_stride;
  const device T* vbase =
      values + (size_t)kv_head_idx * v_head_stride + (size_t)k0 * v_seq_stride;

  KLoader loader_k(
      kbase, static_cast<int>(k_seq_stride), KV_smem, simd_gid, simd_lid);
  VLoader loader_v(
      vbase, static_cast<int>(v_seq_stride), KV_smem, simd_gid, simd_lid);

  const short2 sc = MMAFrag_t::get_coord(simd_lid);
  const short sm = sc.y;
  const short sn = sc.x;
  const int row = int(simd_gid) * kFragSize + sm;
  // Row -> (query head within kv group, query position in the chunk).
  const int gq_head = row / QW; // 0 .. GQ-1
  const int q_pos = q0 + (row % QW); // absolute query position in the chunk
  const int q_head_idx = kv_head_idx * GQ + gq_head;
  const bool row_active = q_pos < q_len;
  // Highest key this query attends: prefix + causal self-chunk.
  const int lim = (N - q_len) + q_pos;
  const int lim_min = N - q_len; // every real query attends at least this far

  // Q tile in float32 fragments; each row reads its own [q_head, q_pos] slice.
  // A row past the chunk (q_pos >= q_len) zero-fills and is never written.
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Qtile;
  {
    const device T* qrow = row_active
        ? queries + ((size_t)q_head_idx * q_len + q_pos) * D + sn
        : queries + sn;
    Qtile.template load_safe<T, 1, 1>(
        qrow, D, short2(D - sn, row_active ? 1 : 0));
  }

  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Stile;
  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Ktile;
  mlx::steel::MMATile<float, 1, 1, MMAFrag_t> Vtile;
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Otile;
  Otile.clear();

  const float scale2 = scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0;

  for (int kt = k0; kt < k1; kt += BK) {
    const int krem = k1 - kt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (krem < BK) {
      loader_k.load_safe(short2(D, krem));
    } else {
      loader_k.load_unsafe();
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T, one 8-deep head-dim slab at a time.
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<T, 1, 1, LDK, 1>(
          &KV_smem[(dd * kFragSize + sm) * LDK + sn]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }
    if (krem < BK || kt + BK - 1 > lim_min) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int kg = kt + ik * kFragSize + sn;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          if (kg + jj >= k1 || kg + jj > lim) {
            Stile.frag_at(0, ik)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (krem < BK) {
      loader_v.load_safe(short2(D, krem));
    } else {
      loader_v.load_unsafe();
    }

    float new_max = max_score;
    Stile.template row_reduce<KQMaxOp>(&new_max);
    if (new_max > Limits<float>::finite_min) {
      Stile.template row_bin_op<KQExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      float tile_sum = 0;
      Stile.template row_reduce<KQSumOp>(&tile_sum);
      sum_score = sum_score * factor + tile_sum;
      max_score = new_max;
      Otile.template row_bin_op<KQMulOp>(&factor);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        Vtile.template load<T, 1, 1, LDV, 1>(
            &KV_smem[(ik * kFragSize + sm) * LDV + id * kFragSize + sn]);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }

    loader_k.next();
    loader_v.next();
  }

  // Partials in the [B, Hq, qL, gqa_splits, D] layout the merge pass reads with
  // grid (Hq, B, qL); one B here.
  if (row_active) {
    const size_t po =
        ((size_t)q_head_idx * q_len + q_pos) * gqa_splits + split_idx;
    device float* orow = out + po * D + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        orow[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
      }
    }
    if (sn == 0) {
      sums[po] = sum_score;
      maxs[po] = max_score == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score * M_LN2_F;
    }
  }
}

// Buffer-parameter block for the q8-past prefill kernel: element strides for
// the six past-cache arrays (biases share the scale strides), the two dense
// self-chunk arrays, and the scalar geometry. One set_bytes on the host; the
// layout is plain C++ (twelve 8-byte strides then four 4-byte scalars).
struct KqPrefillQ8Params {
  size_t pk_head, pk_seq; // past K packed words
  size_t pks_head, pks_seq; // past K scales (and biases)
  size_t pv_head, pv_seq; // past V packed words
  size_t pvs_head, pvs_seq; // past V scales (and biases)
  size_t sk_head, sk_seq; // self-chunk K
  size_t sv_head, sv_seq; // self-chunk V
  int N; // total keys (past_len + q_len)
  int past_len;
  int q_len;
  float scale;
};

// Simdgroup-matrix prefill attention, pass 1, q8 past phase. The serving form
// of kq_sdpa_fa_prefill: float32 queries and float32 accumulators, with the
// past prefix read directly from the QuantizedKVCache tuple (packed uint32,
// float32 scales and biases, group 64, 8 bits) and the fresh self-chunk keys
// and values read dense. Both sources stage into StageT threadgroup tiles
// during the cooperative load: past tiles dequantize with fma(scale, q, bias),
// the exact mx dequantization form (verified bit-exact against mx.dequantize),
// and self tiles cast from float32. The matmuls run on simdgroup_matrix with
// float32 fragments and accumulators whatever StageT is, so the staging type
// trades only threadgroup footprint and staging rounding. StageT float (BK=16)
// keeps full precision end to end and doubles as the dequant-exactness
// verification arm and the memory-lever fallback form.
//
// Tile structure, causal band, online softmax, and the partials layout are
// identical to kq_sdpa_fa_prefill_2pass_1: query position p attends keys
// <= past_len + p on the combined key axis [0, N), so past keys are unmasked
// and the self chunk causal. Words never straddle a group-64 boundary
// (sixteen 4-element words per group), so each word's four weights share one
// scale and bias read.
//
// BQ is the query rows per threadgroup (G * QW; one 8-row fragment strip per
// simdgroup, so the threadgroup is 32 * BQ / 8 threads). BQ=64 folds twice the
// query positions onto each staged KV tile, halving the KV re-stream and the
// staging work per query row at unchanged per-thread register cost; it is the
// serving width. The staging loops are segmented by source (past, self, zero
// tail) so the per-word source branch is hoisted out of the hot loops, V-tile
// stores are 4-element vectors, and self-phase reads are float4.
template <typename StageT, int D, int QW, int BK, int BQ>
[[kernel]] void kq_sdpa_fa_prefill_q8_2pass_1(
    const device float* queries [[buffer(0)]],
    const device uint32_t* pk_w [[buffer(1)]],
    const device float* pk_s [[buffer(2)]],
    const device float* pk_b [[buffer(3)]],
    const device uint32_t* pv_w [[buffer(4)]],
    const device float* pv_s [[buffer(5)]],
    const device float* pv_b [[buffer(6)]],
    const device float* self_k [[buffer(7)]],
    const device float* self_v [[buffer(8)]],
    device float* out [[buffer(9)]],
    device float* sums [[buffer(10)]],
    device float* maxs [[buffer(11)]],
    const constant KqPrefillQ8Params& p [[buffer(12)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int GQ = BQ / QW; // GQA heads folded into one tile
  constexpr int kNWarps = BQ / 8;
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short TD = D / kFragSize;
  constexpr short kPad = 16 / sizeof(StageT);
  constexpr short LDK = BK + kPad; // Ks staged transposed [D][BK + pad]
  constexpr short LDV = D + kPad; // Vs staged row-major [BK][D + pad]
  constexpr int kSmemKV = (LDK * D > BK * LDV) ? LDK * D : BK * LDV;
  constexpr int kWPR = D / 4; // packed words per key row

  using MMAFrag_t = mlx::steel::BaseMMAFrag<float, kFragSize, kFragSize>;
  using Stage4 = metal::vec<StageT, 4>;

  threadgroup StageT KV_smem[kSmemKV];

  const int kv_head_idx = tid.x;
  const int qtile_idx = tid.y;
  const int split_idx = tid.z;
  const int q0 = qtile_idx * QW;
  const int N = p.N;
  const int Lp = p.past_len;
  const int q_len = p.q_len;

  // Contiguous, BK-aligned chunk of the combined key axis.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const short2 sc = MMAFrag_t::get_coord(simd_lid);
  const short sm = sc.y;
  const short sn = sc.x;
  const int row = int(simd_gid) * kFragSize + sm;
  const int gq_head = row / QW;
  const int q_pos = q0 + (row % QW);
  const int q_head_idx = kv_head_idx * GQ + gq_head;
  const bool row_active = q_pos < q_len;
  const int lim = Lp + q_pos; // past unmasked + causal self chunk
  const int lim_min = Lp;

  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Qtile;
  {
    const device float* qrow = row_active
        ? queries + ((size_t)q_head_idx * q_len + q_pos) * D + sn
        : queries + sn;
    Qtile.template load_safe<float, 1, 1>(
        qrow, D, short2(D - sn, row_active ? 1 : 0));
  }

  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Stile;
  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Ktile;
  mlx::steel::MMATile<float, 1, 1, MMAFrag_t> Vtile;
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Otile;
  Otile.clear();

  const float scale2 = p.scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0;

  const int flat = int(simd_gid) * 32 + int(simd_lid);
  const int n_threads = kNWarps * 32;
  const size_t hk = (size_t)kv_head_idx;

  for (int kt = k0; kt < k1; kt += BK) {
    // Per-tile row segmentation: rows [0, r_past) read the q8 cache, rows
    // [r_past, r_live) read the dense self chunk, rows [r_live, BK) zero-fill.
    const int r_past = metal::clamp(Lp - kt, 0, BK);
    const int r_live = metal::clamp(k1 - kt, 0, BK);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage the K tile transposed [D][LDK]: past rows dequantize with fma
    // (the exact mx form), self rows cast from float32, one branchless
    // segment per source. One packed word (four head-dim elements) per step.
    for (int w = flat; w < r_past * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      const int kg = kt + r;
      const uint32_t word = pk_w[hk * p.pk_head + (size_t)kg * p.pk_seq + c4];
      const size_t gb = hk * p.pks_head + (size_t)kg * p.pks_seq + (c4 >> 4);
      const float s8 = pk_s[gb];
      const float b8 = pk_b[gb];
      const int c = c4 * 4;
      KV_smem[(c + 0) * LDK + r] = StageT(fma(s8, float(word & 0xff), b8));
      KV_smem[(c + 1) * LDK + r] =
          StageT(fma(s8, float((word >> 8) & 0xff), b8));
      KV_smem[(c + 2) * LDK + r] =
          StageT(fma(s8, float((word >> 16) & 0xff), b8));
      KV_smem[(c + 3) * LDK + r] =
          StageT(fma(s8, float((word >> 24) & 0xff), b8));
    }
    for (int w = r_past * kWPR + flat; w < r_live * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      const float4 x =
          *(const device float4*)(self_k + hk * p.sk_head +
                                  (size_t)(kt + r - Lp) * p.sk_seq + c4 * 4);
      const int c = c4 * 4;
      KV_smem[(c + 0) * LDK + r] = StageT(x.x);
      KV_smem[(c + 1) * LDK + r] = StageT(x.y);
      KV_smem[(c + 2) * LDK + r] = StageT(x.z);
      KV_smem[(c + 3) * LDK + r] = StageT(x.w);
    }
    for (int w = r_live * kWPR + flat; w < BK * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c = (w % kWPR) * 4;
      KV_smem[(c + 0) * LDK + r] = StageT(0);
      KV_smem[(c + 1) * LDK + r] = StageT(0);
      KV_smem[(c + 2) * LDK + r] = StageT(0);
      KV_smem[(c + 3) * LDK + r] = StageT(0);
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T, one 8-deep head-dim slab at a time.
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<StageT, 1, 1, LDK, 1>(
          &KV_smem[(dd * kFragSize + sm) * LDK + sn]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    const int krem = k1 - kt;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }
    if (krem < BK || kt + BK - 1 > lim_min) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int kg = kt + ik * kFragSize + sn;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          if (kg + jj >= k1 || kg + jj > lim) {
            Stile.frag_at(0, ik)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage the V tile row-major [BK][LDV] from the same segmented sources;
    // the four dequantized elements of a word store as one vector (LDV keeps
    // the row base 4-element aligned).
    for (int w = flat; w < r_past * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      const int kg = kt + r;
      const uint32_t word = pv_w[hk * p.pv_head + (size_t)kg * p.pv_seq + c4];
      const size_t gb = hk * p.pvs_head + (size_t)kg * p.pvs_seq + (c4 >> 4);
      const float s8 = pv_s[gb];
      const float b8 = pv_b[gb];
      *(threadgroup Stage4*)(KV_smem + r * LDV + c4 * 4) = Stage4(
          StageT(fma(s8, float(word & 0xff), b8)),
          StageT(fma(s8, float((word >> 8) & 0xff), b8)),
          StageT(fma(s8, float((word >> 16) & 0xff), b8)),
          StageT(fma(s8, float((word >> 24) & 0xff), b8)));
    }
    for (int w = r_past * kWPR + flat; w < r_live * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      const float4 x =
          *(const device float4*)(self_v + hk * p.sv_head +
                                  (size_t)(kt + r - Lp) * p.sv_seq + c4 * 4);
      *(threadgroup Stage4*)(KV_smem + r * LDV + c4 * 4) =
          Stage4(StageT(x.x), StageT(x.y), StageT(x.z), StageT(x.w));
    }
    for (int w = r_live * kWPR + flat; w < BK * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      *(threadgroup Stage4*)(KV_smem + r * LDV + c4 * 4) = Stage4(StageT(0));
    }

    // Online softmax on this thread's row (registers, before the V barrier).
    float new_max = max_score;
    Stile.template row_reduce<KQMaxOp>(&new_max);
    if (new_max > Limits<float>::finite_min) {
      Stile.template row_bin_op<KQExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      float tile_sum = 0;
      Stile.template row_reduce<KQSumOp>(&tile_sum);
      sum_score = sum_score * factor + tile_sum;
      max_score = new_max;
      Otile.template row_bin_op<KQMulOp>(&factor);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        Vtile.template load<StageT, 1, 1, LDV, 1>(
            &KV_smem[(ik * kFragSize + sm) * LDV + id * kFragSize + sn]);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }
  }

  // Partials in the [B, Hq, qL, gqa_splits, D] layout the merge pass reads.
  if (row_active) {
    const size_t po =
        ((size_t)q_head_idx * q_len + q_pos) * gqa_splits + split_idx;
    device float* orow = out + po * D + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        orow[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
      }
    }
    if (sn == 0) {
      sums[po] = sum_score;
      maxs[po] = max_score == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score * M_LN2_F;
    }
  }
}

// Buffer-parameter block for the q8 decode kernel: element strides for the six
// past-cache arrays (biases share the scale strides) and the scalar geometry.
// The decode read has no dense self chunk (the fresh token is written into the
// q8 cache before this kernel reads it), so this is the prefill block minus the
// two self-chunk stride pairs and past_len (N is the whole key axis).
struct KqDecodeQ8Params {
  size_t pk_head, pk_seq; // past K packed words
  size_t pks_head, pks_seq; // past K scales (and biases)
  size_t pv_head, pv_seq; // past V packed words
  size_t pvs_head, pvs_seq; // past V scales (and biases)
  int N; // total keys (the whole q8 cache offset)
  float scale;
};

// Fused q8 decode attention, pass 1: the sibling composition of the gqa split-K
// grid/merge shape and the prefill_q8 staging idiom, specialized to one query
// row (qL == 1). The whole GQA group folds into a single BQ-row MMA query tile
// (BQ = G at qL == 1; 8 rows for the 16:2 GQA, head-dim-256 serving geometry,
// one 8-row fragment strip per simdgroup), the key axis splits into gqa_splits
// contiguous chunks, each threadgroup streams its chunk once through StageT
// threadgroup-staged K/V tiles with a per-tile online softmax, and the
// per-split partials merge through the shared kq_sdpa_gqa_2pass_2 pass
// unchanged.
//
// Differs from kq_sdpa_fa_prefill_q8_2pass_1 in two ways: there is no dense
// self chunk (the served decode cache already holds the fresh token as q8, read
// as part of the whole tuple), so every staged key comes from the q8 cache;
// and a
// decode query attends every key, so there is no causal band inside a split.
// Only the split tail past k1 zero-fills. The past tiles dequantize during the
// cooperative load with fma(scale, q, bias), the exact mx dequantization form
// (verified bit-exact against mx.dequantize); words never straddle a group-64
// boundary (sixteen 4-element words per group), so each word's four weights
// share one scale and bias read. Scores run in exp2 space (scale premultiplied
// by log2(e)); the stored row max converts back to natural log so the partials
// merge unchanged. An empty split (its chunk entirely past N) writes
// (O = 0, sum = 0, max = finite_min) partials that merge with weight zero.
template <typename StageT, int D, int BQ>
[[kernel]] void kq_sdpa_decode_q8_2pass_1(
    const device float* queries [[buffer(0)]],
    const device uint32_t* pk_w [[buffer(1)]],
    const device float* pk_s [[buffer(2)]],
    const device float* pk_b [[buffer(3)]],
    const device uint32_t* pv_w [[buffer(4)]],
    const device float* pv_s [[buffer(5)]],
    const device float* pv_b [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant KqDecodeQ8Params& p [[buffer(10)]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int GQ = BQ; // GQA heads folded into one tile (qL == 1)
  constexpr int BK = 16; // keys staged per tile (float32 staging budget)
  constexpr int kNWarps = BQ / 8;
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short TD = D / kFragSize;
  constexpr short kPad = 16 / sizeof(StageT);
  constexpr short LDK = BK + kPad; // Ks staged transposed [D][BK + pad]
  constexpr short LDV = D + kPad; // Vs staged row-major [BK][D + pad]
  constexpr int kSmemKV = (LDK * D > BK * LDV) ? LDK * D : BK * LDV;
  constexpr int kWPR = D / 4; // packed words per key row

  using MMAFrag_t = mlx::steel::BaseMMAFrag<float, kFragSize, kFragSize>;
  using Stage4 = metal::vec<StageT, 4>;

  threadgroup StageT KV_smem[kSmemKV];

  const int kv_head_idx = tid.x;
  const int split_idx = tid.z;
  const int N = p.N;

  // Contiguous, BK-aligned chunk of the key axis.
  const int chunk = ((N + gqa_splits * BK - 1) / (gqa_splits * BK)) * BK;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const short2 sc = MMAFrag_t::get_coord(simd_lid);
  const short sm = sc.y;
  const short sn = sc.x;
  const int row = int(simd_gid) * kFragSize + sm;
  const int gq_head = row; // qL == 1: one head per row
  const int q_head_idx = kv_head_idx * GQ + gq_head;
  const bool row_active = gq_head < GQ;

  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Qtile;
  {
    const device float* qrow =
        row_active ? queries + (size_t)q_head_idx * D + sn : queries + sn;
    Qtile.template load_safe<float, 1, 1>(
        qrow, D, short2(D - sn, row_active ? 1 : 0));
  }

  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Stile;
  mlx::steel::MMATile<float, 1, TK, MMAFrag_t> Ktile;
  mlx::steel::MMATile<float, 1, 1, MMAFrag_t> Vtile;
  mlx::steel::MMATile<float, 1, TD, MMAFrag_t> Otile;
  Otile.clear();

  const float scale2 = p.scale * M_LOG2E_F;
  float max_score = Limits<float>::finite_min;
  float sum_score = 0;

  const int flat = int(simd_gid) * 32 + int(simd_lid);
  const int n_threads = kNWarps * 32;
  const size_t hk = (size_t)kv_head_idx;

  for (int kt = k0; kt < k1; kt += BK) {
    // Rows [0, r_live) read the q8 cache; rows [r_live, BK) zero-fill the tail.
    const int r_live = metal::clamp(k1 - kt, 0, BK);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage the K tile transposed [D][LDK]: past rows dequantize with fma (the
    // exact mx form). One packed word (four head-dim elements) per step.
    for (int w = flat; w < r_live * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      const int kg = kt + r;
      const uint32_t word = pk_w[hk * p.pk_head + (size_t)kg * p.pk_seq + c4];
      const size_t gb = hk * p.pks_head + (size_t)kg * p.pks_seq + (c4 >> 4);
      const float s8 = pk_s[gb];
      const float b8 = pk_b[gb];
      const int c = c4 * 4;
      KV_smem[(c + 0) * LDK + r] = StageT(fma(s8, float(word & 0xff), b8));
      KV_smem[(c + 1) * LDK + r] =
          StageT(fma(s8, float((word >> 8) & 0xff), b8));
      KV_smem[(c + 2) * LDK + r] =
          StageT(fma(s8, float((word >> 16) & 0xff), b8));
      KV_smem[(c + 3) * LDK + r] =
          StageT(fma(s8, float((word >> 24) & 0xff), b8));
    }
    for (int w = r_live * kWPR + flat; w < BK * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c = (w % kWPR) * 4;
      KV_smem[(c + 0) * LDK + r] = StageT(0);
      KV_smem[(c + 1) * LDK + r] = StageT(0);
      KV_smem[(c + 2) * LDK + r] = StageT(0);
      KV_smem[(c + 3) * LDK + r] = StageT(0);
    }
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q @ K^T, one 8-deep head-dim slab at a time.
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Ktile.template load<StageT, 1, 1, LDK, 1>(
          &KV_smem[(dd * kFragSize + sm) * LDK + sn]);
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    const int krem = k1 - kt;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }
    // A decode query attends every key, so the only masked scores are the split
    // tail past k1 (a short final tile). Interior tiles run mask-free.
    if (krem < BK) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const int kg = kt + ik * kFragSize + sn;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          if (kg + jj >= k1) {
            Stile.frag_at(0, ik)[jj] = Limits<float>::finite_min;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage the V tile row-major [BK][LDV] from the q8 cache; the four
    // dequantized elements of a word store as one vector.
    for (int w = flat; w < r_live * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      const int kg = kt + r;
      const uint32_t word = pv_w[hk * p.pv_head + (size_t)kg * p.pv_seq + c4];
      const size_t gb = hk * p.pvs_head + (size_t)kg * p.pvs_seq + (c4 >> 4);
      const float s8 = pv_s[gb];
      const float b8 = pv_b[gb];
      *(threadgroup Stage4*)(KV_smem + r * LDV + c4 * 4) = Stage4(
          StageT(fma(s8, float(word & 0xff), b8)),
          StageT(fma(s8, float((word >> 8) & 0xff), b8)),
          StageT(fma(s8, float((word >> 16) & 0xff), b8)),
          StageT(fma(s8, float((word >> 24) & 0xff), b8)));
    }
    for (int w = r_live * kWPR + flat; w < BK * kWPR; w += n_threads) {
      const int r = w / kWPR;
      const int c4 = w % kWPR;
      *(threadgroup Stage4*)(KV_smem + r * LDV + c4 * 4) = Stage4(StageT(0));
    }

    // Online softmax on this thread's row (registers, before the V barrier).
    float new_max = max_score;
    Stile.template row_reduce<KQMaxOp>(&new_max);
    if (new_max > Limits<float>::finite_min) {
      Stile.template row_bin_op<KQExpSubOp>(&new_max);
      float factor = fast::exp2(max_score - new_max);
      float tile_sum = 0;
      Stile.template row_reduce<KQSumOp>(&tile_sum);
      sum_score = sum_score * factor + tile_sum;
      max_score = new_max;
      Otile.template row_bin_op<KQMulOp>(&factor);
    } else {
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = 0;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P @ V
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        Vtile.template load<StageT, 1, 1, LDV, 1>(
            &KV_smem[(ik * kFragSize + sm) * LDV + id * kFragSize + sn]);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }
  }

  // Partials in the [B, Hq, 1, gqa_splits, D] layout the merge pass reads with
  // grid (Hq, B, 1); one B here.
  if (row_active) {
    const size_t po = (size_t)q_head_idx * gqa_splits + split_idx;
    device float* orow = out + po * D + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        orow[id * kFragSize + jj] = Otile.frag_at(0, id)[jj];
      }
    }
    if (sn == 0) {
      sums[po] = sum_score;
      maxs[po] = max_score == Limits<float>::finite_min
          ? Limits<float>::finite_min
          : max_score * M_LN2_F;
    }
  }
}

// Fused q8 decode attention, pass 1, SIMD-shuffle compute. The decode-latency
// form: it takes kq_sdpa_gqa_2pass_1's shuffle-reduction compute (NE keys in
// flight per simdgroup, NL lanes per key, a log2(32/NE)-step lane reduction,
// the GQA group folded as the threadgroup y dim so device KV reads once per
// kv-head) rather than the matrix-unit tile, which underutilizes at one query
// row. The past K and V read directly from the QuantizedKVCache tuple and
// dequantize with fma(scale, q, bias) (the exact mx form) into float staging
// during the cooperative tile load; words never straddle a group-64 boundary,
// so each staged float4 shares one scale and bias read. Float32 queries, float
// staging, float32 accumulators. qL == 1: one query row folded as gqa_factor
// heads; a decode query attends every key (no causal band inside the past);
// only the split tail past N zero-fills. Partials merge through
// kq_sdpa_gqa_2pass_2 unchanged.
template <
    int D,
    int C = 8,
    int NE = 4,
    bool AlignedUint4Load = false,
    bool VectorByteUnpack = false>
[[kernel]] void kq_sdpa_decode_gqa_q8_2pass_1(
    const device float* queries [[buffer(0)]],
    const device uint32_t* pk_w [[buffer(1)]],
    const device float* pk_s [[buffer(2)]],
    const device float* pk_b [[buffer(3)]],
    const device uint32_t* pv_w [[buffer(4)]],
    const device float* pv_s [[buffer(5)]],
    const device float* pv_b [[buffer(6)]],
    device float* out [[buffer(7)]],
    device float* sums [[buffer(8)]],
    device float* maxs [[buffer(9)]],
    const constant KqDecodeQ8Params& p [[buffer(10)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  constexpr int D4 = D / 4;
  constexpr int NL = 32 / NE; // lanes per in-flight key
  constexpr int DP4 = D4 / NL; // float4s per lane per key row

  threadgroup float4 sK[C * D4];
  threadgroup float4 sV[C * D4];

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int split_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int lane = tidtg.x;
  const int tx = lane % NL;
  const int ty = lane / NL;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
  const int N = p.N;
  const float scale = p.scale;

  // Contiguous, C-aligned chunk of the key axis for this threadgroup.
  const int chunk = ((N + gqa_splits * C - 1) / (gqa_splits * C)) * C;
  const int k0 = split_idx * chunk;
  const int k1 = min(k0 + chunk, N);

  const size_t hk = (size_t)(batch_idx * num_kv_heads + kv_head_idx);

  // Pre-scaled query slices for this lane's key-row columns ([B, Hq, 1, D],
  // row-contiguous).
  const device float4* q4 =
      (const device float4*)(queries + (size_t)q_batch_head_idx * D);
  float4 qf[DP4];
  for (short ii = 0; ii < DP4; ii++) {
    qf[ii] = scale * q4[ii * NL + tx];
  }

  float max_score = Limits<float>::finite_min;
  float sum_exp_score = 0;
  float4 lo[DP4];
  for (short ii = 0; ii < DP4; ii++) {
    lo[ii] = 0;
  }

  const int flat = tidtg.y * 32 + lane;
  const int n_threads = 32 * gqa_factor;

  for (int kt = k0; kt < k1; kt += C) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Cooperative tile load, dequantizing the q8 tuple into float staging. One
    // float4 (four head-dim elements) per step from one packed word; the tail
    // past k1 zero-fills so stale threadgroup data never reaches the
    // accumulators.
    if constexpr (AlignedUint4Load) {
      // Four consecutive words cover sixteen values inside one group-64
      // scale/bias block. The row base and four-word column are 16-byte
      // aligned at the fixed D=256 serving geometry.
      constexpr int kWordsPerVector = 4;
      constexpr int kVectorsPerRow = D4 / kWordsPerVector;
      for (int vi = flat; vi < C * kVectorsPerRow; vi += n_threads) {
        const int row = vi / kVectorsPerRow;
        const int col = (vi % kVectorsPerRow) * kWordsPerVector;
        const int kg = kt + row;
        if (kg < k1) {
          const size_t kbase = hk * p.pk_head + (size_t)kg * p.pk_seq + col;
          const uint4 kw = *reinterpret_cast<const device uint4*>(pk_w + kbase);
          const size_t kgb =
              hk * p.pks_head + (size_t)kg * p.pks_seq + (col >> 4);
          const float ks = pk_s[kgb];
          const float kb = pk_b[kgb];
          const size_t vbase = hk * p.pv_head + (size_t)kg * p.pv_seq + col;
          const uint4 vw = *reinterpret_cast<const device uint4*>(pv_w + vbase);
          const size_t vgb =
              hk * p.pvs_head + (size_t)kg * p.pvs_seq + (col >> 4);
          const float vs_ = pv_s[vgb];
          const float vb = pv_b[vgb];
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < kWordsPerVector; jj++) {
            const uint32_t kwj = kw[jj];
            const uint32_t vwj = vw[jj];
            const int i = row * D4 + col + jj;
            if constexpr (VectorByteUnpack) {
              const float4 kc = float4(as_type<uchar4>(kwj));
              const float4 vc = float4(as_type<uchar4>(vwj));
              sK[i] = fma(float4(ks), kc, float4(kb));
              sV[i] = fma(float4(vs_), vc, float4(vb));
            } else {
              sK[i] = float4(
                  fma(ks, float(kwj & 0xff), kb),
                  fma(ks, float((kwj >> 8) & 0xff), kb),
                  fma(ks, float((kwj >> 16) & 0xff), kb),
                  fma(ks, float((kwj >> 24) & 0xff), kb));
              sV[i] = float4(
                  fma(vs_, float(vwj & 0xff), vb),
                  fma(vs_, float((vwj >> 8) & 0xff), vb),
                  fma(vs_, float((vwj >> 16) & 0xff), vb),
                  fma(vs_, float((vwj >> 24) & 0xff), vb));
            }
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < kWordsPerVector; jj++) {
            const int i = row * D4 + col + jj;
            sK[i] = float4(0);
            sV[i] = float4(0);
          }
        }
      }
    } else {
      for (int i = flat; i < C * D4; i += n_threads) {
        const int row = i / D4;
        const int col = i % D4;
        const int kg = kt + row;
        if (kg < k1) {
          const uint32_t kw =
              pk_w[hk * p.pk_head + (size_t)kg * p.pk_seq + col];
          const size_t kgb =
              hk * p.pks_head + (size_t)kg * p.pks_seq + (col >> 4);
          const float ks = pk_s[kgb];
          const float kb = pk_b[kgb];
          sK[i] = float4(
              fma(ks, float(kw & 0xff), kb),
              fma(ks, float((kw >> 8) & 0xff), kb),
              fma(ks, float((kw >> 16) & 0xff), kb),
              fma(ks, float((kw >> 24) & 0xff), kb));
          const uint32_t vw =
              pv_w[hk * p.pv_head + (size_t)kg * p.pv_seq + col];
          const size_t vgb =
              hk * p.pvs_head + (size_t)kg * p.pvs_seq + (col >> 4);
          const float vs_ = pv_s[vgb];
          const float vb = pv_b[vgb];
          sV[i] = float4(
              fma(vs_, float(vw & 0xff), vb),
              fma(vs_, float((vw >> 8) & 0xff), vb),
              fma(vs_, float((vw >> 16) & 0xff), vb),
              fma(vs_, float((vw >> 24) & 0xff), vb));
        } else {
          sK[i] = float4(0);
          sV[i] = float4(0);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: NE keys in flight, NL lanes per key, then reduce + broadcast.
    float mqk[C / NE];
    float m_tile = Limits<float>::finite_min;
    for (short cc = 0; cc < C / NE; cc++) {
      float s = 0;
      for (short ii = 0; ii < DP4; ii++) {
        s += dot(sK[(cc * NE + ty) * D4 + ii * NL + tx], qf[ii]);
      }
      const int kg = kt + cc * NE + ty;
      for (short off = NL / 2; off > 0; off >>= 1) {
        s += simd_shuffle_down(s, off);
      }
      s = simd_shuffle(s, NL * ty);
      const bool valid = kg < k1; // a decode query attends every key
      mqk[cc] = valid ? s : Limits<float>::finite_min;
      m_tile = max(m_tile, mqk[cc]);
    }

    // Online softmax; each lane sums its ty-group's keys, so the simd_sum
    // counts every key NL times (divided back by NL).
    float vs_row[C / NE];
    m_tile = simd_max(m_tile);
    if (m_tile > Limits<float>::finite_min) {
      const float new_max = max(max_score, m_tile);
      const float factor = fast::exp(max_score - new_max);
      float vsum = 0;
      for (short cc = 0; cc < C / NE; cc++) {
        vs_row[cc] = fast::exp(mqk[cc] - new_max);
        vsum += vs_row[cc];
      }
      sum_exp_score = sum_exp_score * factor + simd_sum(vsum) * (1.0f / NL);
      max_score = new_max;
      for (short ii = 0; ii < DP4; ii++) {
        lo[ii] *= factor;
      }
    } else {
      for (short cc = 0; cc < C / NE; cc++) {
        vs_row[cc] = 0;
      }
    }

    for (short cc = 0; cc < C / NE; cc++) {
      for (short ii = 0; ii < DP4; ii++) {
        lo[ii] += sV[(cc * NE + ty) * D4 + ii * NL + tx] * vs_row[cc];
      }
    }
  }

  // Cross-ty reduction of the deferred output accumulators; the ty == 0 lane
  // group holds the chunk totals.
  for (short ii = 0; ii < DP4; ii++) {
    for (short off = (NE / 2) * NL; off >= NL; off >>= 1) {
      lo[ii][0] += simd_shuffle_down(lo[ii][0], off);
      lo[ii][1] += simd_shuffle_down(lo[ii][1], off);
      lo[ii][2] += simd_shuffle_down(lo[ii][2], off);
      lo[ii][3] += simd_shuffle_down(lo[ii][3], off);
    }
  }

  const size_t po = (size_t)q_batch_head_idx * gqa_splits + split_idx;
  if (ty == 0) {
    device float4* out4 = (device float4*)(out + po * D);
    for (short ii = 0; ii < DP4; ii++) {
      out4[ii * NL + tx] = lo[ii];
    }
  }
  if (lane == 0) {
    sums[po] = sum_exp_score;
    maxs[po] = max_score;
  }
}

// Merge the per-split partials; one simdgroup per (q-head, batch, query).
// Grid z is the query axis (1 at decode; q_len at verify width). Sinks are
// a per-q-head extra logit with no value row: they raise the global max and
// add exp(sink - max) to the denominator; a query's own sink applies at
// every verify position identically.
template <typename T, int D>
[[kernel]] void kq_sdpa_gqa_2pass_2(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    const device float* sinks [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& n_q_heads [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int EPT = D / 32; // output elements per lane
  const int head_idx = tid.x;
  const int batch_idx = tid.y;
  const size_t base =
      ((size_t)batch_idx * n_q_heads + head_idx) * tpg.z + tid.z;

  partials += base * gqa_splits * D;
  sums += base * gqa_splits;
  maxs += base * gqa_splits;

  threadgroup float ws[128];

  float m = Limits<float>::finite_min;
  for (int s = simd_lid; s < gqa_splits; s += 32) {
    m = max(m, maxs[s]);
  }
  m = simd_max(m);
  if (gqa_has_sinks) {
    m = max(m, sinks[head_idx]);
  }

  float denom = 0;
  for (int s = simd_lid; s < gqa_splits; s += 32) {
    const float w = fast::exp(maxs[s] - m);
    ws[s] = w;
    denom += w * sums[s];
  }
  denom = simd_sum(denom);
  if (gqa_has_sinks) {
    denom += fast::exp(sinks[head_idx] - m);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[EPT] = {0};
  for (int s = 0; s < gqa_splits; s++) {
    const float w = ws[s];
    for (short e = 0; e < EPT; e++) {
      acc[e] += w * partials[s * D + e * 32 + simd_lid];
    }
  }
  out += base * D;
  for (short e = 0; e < EPT; e++) {
    out[e * 32 + simd_lid] = static_cast<T>(denom == 0 ? 0.0f : acc[e] / denom);
  }
}

// Fixed-geometry dimension-parallel form of the MLX-derived split merge above.
// SIMD group zero preserves the shared merge's lane mapping and reduction order
// for the global max, split weights, and denominator. The remaining SIMD groups
// only redistribute output dimensions, and each dimension walks the splits in
// the same increasing order as kq_sdpa_gqa_2pass_2.
template <int SPLITS, int D, int DIM_SIMDS>
[[kernel]] void kq_sdpa_q8_merge_dim_parallel(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    const device float* sinks [[buffer(3)]],
    device float* out [[buffer(4)]],
    const constant int& n_q_heads [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  static_assert(SPLITS == 128, "dimension-parallel merge requires 128 splits");
  static_assert(D == DIM_SIMDS * 32, "one SIMD group must cover 32 dimensions");

  const int head_idx = tid.x;
  const int batch_idx = tid.y;
  const size_t base =
      ((size_t)batch_idx * n_q_heads + head_idx) * tpg.z + tid.z;
  partials += base * SPLITS * D;
  sums += base * SPLITS;
  maxs += base * SPLITS;
  (void)sinks;

  threadgroup float ws[SPLITS];
  threadgroup float shared_denom;
  if (simd_gid == 0) {
    float m = Limits<float>::finite_min;
    for (int s = simd_lid; s < SPLITS; s += 32) {
      m = max(m, maxs[s]);
    }
    m = simd_max(m);

    float denom = 0;
    for (int s = simd_lid; s < SPLITS; s += 32) {
      const float w = fast::exp(maxs[s] - m);
      ws[s] = w;
      denom += w * sums[s];
    }
    denom = simd_sum(denom);
    if (simd_lid == 0) {
      shared_denom = denom;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int e = simd_gid;
  float acc = 0;
  for (int s = 0; s < SPLITS; s++) {
    acc += ws[s] * partials[s * D + e * 32 + simd_lid];
  }
  out[base * D + e * 32 + simd_lid] =
      shared_denom == 0 ? 0.0f : acc / shared_denom;
}
