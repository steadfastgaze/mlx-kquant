// Decode-only Q8_0 matvec with the DeepSeek-V4 hyper-connection post
// recombination fused into the output epilogue.

template <typename T, bool round_input>
[[kernel]] void kq_q8_0_qmv_fast_hc_post(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    const device float* residual [[buffer(4)]],
    const device float* post [[buffer(5)]],
    const device float* comb [[buffer(6)]],
    const constant int& in_vec_size [[buffer(7)]],
    const constant int& out_vec_size [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)scales;

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  thread float x_thread[values_per_thread];
  thread float result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int lane_k_offset = simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, float, values_per_thread>(x + k + lane_k_offset, x_thread);
    if constexpr (round_input) {
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        // Float32 is deliberately rounded here instead of at op construction.
        x_thread[i] = float(static_cast<bfloat16_t>(x_thread[i]));
      }
    }

#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base =
          w + static_cast<int64_t>(row_idx) * row_bytes;

      const int k_global = k + lane_k_offset;
      const int block_id = k_global / KQ_Q8_0_GROUP;
      const int within = k_global - block_id * KQ_Q8_0_GROUP;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;
      const float d = float(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;

      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        partial += x_thread[i] * float(q_ptr[i]);
      }
      result[row] += d * partial;
    }
  }

#pragma unroll
  for (int row = 0; row < results_per_simdgroup; row++) {
    float reduced = simd_sum(result[row]);
    if (simd_lid == 0) {
      const int n = out_row + row;
      // Match quantized_matmul's bfloat16 output materialization before the
      // served hC-post graph widens it back to float32.
      float qmv = float(static_cast<bfloat16_t>(reduced));

#pragma unroll
      for (int dst = 0; dst < 4; dst++) {
#pragma clang fp contract(off)
        float mixed =
            metal::fma(residual[0 * out_vec_size + n], comb[0 * 4 + dst], 0.0f);
        mixed = metal::fma(
            residual[1 * out_vec_size + n], comb[1 * 4 + dst], mixed);
        mixed = metal::fma(
            residual[2 * out_vec_size + n], comb[2 * 4 + dst], mixed);
        mixed = metal::fma(
            residual[3 * out_vec_size + n], comb[3 * 4 + dst], mixed);
        float scaled = qmv * post[dst];
        out[dst * out_vec_size + n] = scaled + mixed;
      }
    }
  }
}

template <typename T>
[[kernel]] void kq_q8_0_qmv_fast_add_hc_post(
    const device uint8_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device float* out [[buffer(3)]],
    const device half* routed [[buffer(4)]],
    const device float* residual [[buffer(5)]],
    const device float* post [[buffer(6)]],
    const device float* comb [[buffer(7)]],
    const constant int& in_vec_size [[buffer(8)]],
    const constant int& out_vec_size [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)scales;

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;

  thread float x_thread[values_per_thread];
  thread float result[results_per_simdgroup] = {0};

  const int row_bytes = in_vec_size * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int lane_k_offset = simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, float, values_per_thread>(x + k + lane_k_offset, x_thread);

#pragma unroll
    for (int row = 0; row < results_per_simdgroup; row++) {
      const int row_idx = out_row + row;
      const device uint8_t* row_base =
          w + static_cast<int64_t>(row_idx) * row_bytes;

      const int k_global = k + lane_k_offset;
      const int block_id = k_global / KQ_Q8_0_GROUP;
      const int within = k_global - block_id * KQ_Q8_0_GROUP;
      const device uint8_t* block_addr =
          row_base + block_id * KQ_Q8_0_BLOCK_BYTES;
      const float d = float(kq_q8_0_d(block_addr));
      const device int8_t* q_ptr = kq_q8_0_q_ptr(block_addr) + within;

      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < values_per_thread; i++) {
        partial += x_thread[i] * float(q_ptr[i]);
      }
      result[row] += d * partial;
    }
  }

#pragma unroll
  for (int row = 0; row < results_per_simdgroup; row++) {
    float reduced = simd_sum(result[row]);
    if (simd_lid == 0) {
#pragma clang fp contract(off)
      const int n = out_row + row;
      float shared = float(static_cast<bfloat16_t>(reduced));
      float merged = float(routed[n]) + shared;

#pragma unroll
      for (int dst = 0; dst < 4; dst++) {
#pragma clang fp contract(off)
        float mixed =
            metal::fma(residual[0 * out_vec_size + n], comb[0 * 4 + dst], 0.0f);
        mixed = metal::fma(
            residual[1 * out_vec_size + n], comb[1 * 4 + dst], mixed);
        mixed = metal::fma(
            residual[2 * out_vec_size + n], comb[2 * 4 + dst], mixed);
        mixed = metal::fma(
            residual[3 * out_vec_size + n], comb[3 * 4 + dst], mixed);
        float scaled = merged * post[dst];
        out[dst * out_vec_size + n] = scaled + mixed;
      }
    }
  }
}
