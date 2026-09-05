// Fixed-geometry Qwen gated-residual decode kernels for the released
// four-branch, width-2560, low-rank-320 graph.

METAL_FUNC bfloat16_t kq_qwen4_hc_sigmoid(bfloat16_t x) {
  const auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

[[kernel]] void kq_qwen4_hc_norm(
    const device bfloat16_t* residual [[buffer(0)]],
    const device bfloat16_t* norm_weight [[buffer(1)]],
    const device bfloat16_t* pending_output [[buffer(2)]],
    const device bfloat16_t* pending_injection [[buffer(3)]],
    device bfloat16_t* normalized [[buffer(4)]],
    device bfloat16_t* updated_residual [[buffer(5)]],
    const constant float& eps [[buffer(6)]],
    const constant int& has_pending [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int branches = 4;
  constexpr int width = 2560;

  constexpr int values_per_thread = 4;
  constexpr int threads_per_group = width / values_per_thread;
  constexpr int simdgroups_per_group = threads_per_group / SIMD_SIZE;
  threadgroup float partials[simdgroups_per_group * branches];
  threadgroup float inv_rms[branches];
  float sum_squares[branches] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int first = int(tid) * values_per_thread;
  for (int stream = 0; stream < branches; ++stream) {
    const int base = stream * width;
#pragma unroll
    for (int i = 0; i < values_per_thread; ++i) {
      const int d = first + i;
      bfloat16_t value = residual[base + d];
      if (has_pending != 0) {
        const bfloat16_t update = bfloat16_t(
            float(pending_output[d]) * float(pending_injection[stream]));
        value = bfloat16_t(float(value) + float(update));
      }
      updated_residual[base + d] = value;
      const float source = float(value);
      volatile float square = source * source;
      sum_squares[stream] += square;
    }
  }

  for (int stream = 0; stream < branches; ++stream) {
    sum_squares[stream] = simd_sum(sum_squares[stream]);
  }
  if (lane == 0) {
    for (int stream = 0; stream < branches; ++stream) {
      partials[sg * branches + stream] = sum_squares[stream];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    for (int stream = 0; stream < branches; ++stream) {
      const float cross_simd = simd_sum(
          lane < simdgroups_per_group ? partials[lane * branches + stream]
                                      : 0.0f);
      if (lane == 0) {
        volatile float mean_square = cross_simd * (1.0f / float(width));
        volatile float stabilized_mean = mean_square + eps;
        inv_rms[stream] = metal::precise::rsqrt(stabilized_mean);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int stream = 0; stream < branches; ++stream) {
    const int base = stream * width;
    const float inv = inv_rms[stream];
    for (int d = int(tid); d < width; d += threads_per_group) {
      const int index = base + d;
      const float source = float(updated_residual[index]);
      const float unit = source * inv;
      const float scale = 1.0f + float(norm_weight[index]);
      normalized[index] = bfloat16_t(unit * scale);
    }
  }
  (void)tg;
  (void)branches;
}

[[kernel]] void kq_qwen4_hc_front(
    const device bfloat16_t* normalized [[buffer(0)]],
    const device uint8_t* down_weight [[buffer(1)]],
    const device uint8_t* injection_weight [[buffer(2)]],
    device bfloat16_t* lowrank [[buffer(3)]],
    device bfloat16_t* injection [[buffer(4)]],
    const constant int& has_injection [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int expanded = 10240;
  constexpr int lowrank_width = 320;
  constexpr int down_groups = lowrank_width / 8;
  constexpr int q6_row_bytes =
      expanded * KQ_Q6_K_BLOCK_BYTES / KQ_Q6_K_SUPERBLOCK;
  constexpr int q6_blocks = expanded / KQ_Q6_K_SUPERBLOCK;
  constexpr int results_per_simdgroup = 4;
  constexpr int superblock_stride = 2;

  const int tid_lane = int(lane) / 2;
  const int ix = int(lane) % 2;
  const int ip = tid_lane / 8;
  const int il = tid_lane % 8;
  const int l0 = 4 * il;
  const int is = 8 * ip + l0 / 16;
  float values[16];

  if (int(tg.x) < down_groups) {
    float result[results_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int out_row = int(tg.x) * 8 + int(sg) * results_per_simdgroup;
    for (int ib = ix; ib < q6_blocks; ib += superblock_stride) {
      const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        values[4 * l + 0] = float(normalized[x_base + l + 0]);
        values[4 * l + 1] = float(normalized[x_base + l + 32]);
        values[4 * l + 2] = float(normalized[x_base + l + 64]);
        values[4 * l + 3] = float(normalized[x_base + l + 96]);
      }

#pragma unroll
      for (int row = 0; row < results_per_simdgroup; ++row) {
        const device uint8_t* block = down_weight +
            int64_t(out_row + row) * q6_row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;
        const device uint8_t* q1 = kq_q6_k_ql_ptr(block) + 64 * ip + l0;
        const device uint8_t* q2 = q1 + 32;
        const device uint8_t* qh = kq_q6_k_qh_ptr(block) + 32 * ip + l0;
        const device int8_t* scales = kq_q6_k_scales_ptr(block) + is;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const uint8_t q1l = q1[l];
          const uint8_t q2l = q2[l];
          const uint8_t qhl = qh[l];
          const int8_t v0 =
              int8_t((q1l & 0x0f) | ((qhl & 0x03) << 4)) - int8_t(32);
          const int8_t v1 =
              int8_t((q2l & 0x0f) | ((qhl & 0x0c) << 2)) - int8_t(32);
          const int8_t v2 =
              int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
          const int8_t v3 =
              int8_t((q2l >> 4) | ((qhl & 0xc0) >> 2)) - int8_t(32);
          sums[0] += values[4 * l + 0] * float(v0);
          sums[1] += values[4 * l + 1] * float(v1);
          sums[2] += values[4 * l + 2] * float(v2);
          sums[3] += values[4 * l + 3] * float(v3);
        }
        const float d = float(kq_q6_k_d(block));
        result[row] += d *
            (sums[0] * float(scales[0]) + sums[1] * float(scales[2]) +
             sums[2] * float(scales[4]) + sums[3] * float(scales[6]));
      }
    }

#pragma unroll
    for (int row = 0; row < results_per_simdgroup; ++row) {
      const float reduced = simd_sum(result[row]);
      if (lane == 0) {
        const bfloat16_t projected = bfloat16_t(reduced);
        const bfloat16_t scaled = bfloat16_t(float(projected) * 0.25f);
        const bfloat16_t gate = kq_qwen4_hc_sigmoid(scaled);
        lowrank[out_row + row] = bfloat16_t(float(scaled) * float(gate));
      }
    }
    return;
  }

  if (has_injection == 0 || sg != 0) {
    return;
  }

  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int ib = ix; ib < q6_blocks; ib += superblock_stride) {
    const int x_base = ib * KQ_Q6_K_SUPERBLOCK + 128 * ip + l0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      values[4 * l + 0] = float(normalized[x_base + l + 0]);
      values[4 * l + 1] = float(normalized[x_base + l + 32]);
      values[4 * l + 2] = float(normalized[x_base + l + 64]);
      values[4 * l + 3] = float(normalized[x_base + l + 96]);
    }

#pragma unroll
    for (int row = 0; row < 4; ++row) {
      const device uint8_t* block = injection_weight +
          int64_t(row) * q6_row_bytes + ib * KQ_Q6_K_BLOCK_BYTES;
      const device uint8_t* q1 = kq_q6_k_ql_ptr(block) + 64 * ip + l0;
      const device uint8_t* q2 = q1 + 32;
      const device uint8_t* qh = kq_q6_k_qh_ptr(block) + 32 * ip + l0;
      const device int8_t* scales = kq_q6_k_scales_ptr(block) + is;
      float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint8_t q1l = q1[l];
        const uint8_t q2l = q2[l];
        const uint8_t qhl = qh[l];
        const int8_t v0 =
            int8_t((q1l & 0x0f) | ((qhl & 0x03) << 4)) - int8_t(32);
        const int8_t v1 =
            int8_t((q2l & 0x0f) | ((qhl & 0x0c) << 2)) - int8_t(32);
        const int8_t v2 = int8_t((q1l >> 4) | ((qhl & 0x30) << 0)) - int8_t(32);
        const int8_t v3 = int8_t((q2l >> 4) | ((qhl & 0xc0) >> 2)) - int8_t(32);
        sums[0] += values[4 * l + 0] * float(v0);
        sums[1] += values[4 * l + 1] * float(v1);
        sums[2] += values[4 * l + 2] * float(v2);
        sums[3] += values[4 * l + 3] * float(v3);
      }
      const float d = float(kq_q6_k_d(block));
      result[row] += d *
          (sums[0] * float(scales[0]) + sums[1] * float(scales[2]) +
           sums[2] * float(scales[4]) + sums[3] * float(scales[6]));
    }
  }

#pragma unroll
  for (int row = 0; row < 4; ++row) {
    const float reduced = simd_sum(result[row]);
    if (lane == 0) {
      const bfloat16_t projected = bfloat16_t(reduced);
      const bfloat16_t scaled = bfloat16_t(float(projected) * 0.25f);
      const bfloat16_t gate = kq_qwen4_hc_sigmoid(scaled);
      injection[row] = bfloat16_t(2.0f * float(gate));
    }
  }
}

[[kernel]] void kq_qwen4_hc_epilogue(
    const device bfloat16_t* lowrank [[buffer(0)]],
    const device uint8_t* up_weight [[buffer(1)]],
    const device bfloat16_t* normalized [[buffer(2)]],
    device bfloat16_t* mixed [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int branches = 4;
  constexpr int width = 2560;
  constexpr int lowrank_width = 320;
  constexpr int q8_row_bytes =
      lowrank_width * KQ_Q8_0_BLOCK_BYTES / KQ_Q8_0_GROUP;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  const int d_out = int(tg.x) * 2 + int(sg);
  if (d_out >= width) {
    return;
  }

  float result[branches] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int lane_offset = int(lane) * values_per_thread;
  float x_thread[values_per_thread];
  for (int k = 0; k < lowrank_width; k += block_size) {
    const int remaining = lowrank_width - k - lane_offset;
    if (remaining >= values_per_thread) {
#pragma unroll
      for (int i = 0; i < values_per_thread; ++i) {
        x_thread[i] = float(lowrank[k + lane_offset + i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < values_per_thread; ++i) {
        x_thread[i] =
            i < remaining ? float(lowrank[k + lane_offset + i]) : 0.0f;
      }
    }
    const int active = remaining >= values_per_thread
        ? values_per_thread
        : (remaining > 0 ? remaining : 0);
    if (active == 0) {
      continue;
    }
    const int k_global = k + lane_offset;
    const int block_id = k_global / KQ_Q8_0_GROUP;
    const int within = k_global - block_id * KQ_Q8_0_GROUP;

#pragma unroll
    for (int stream = 0; stream < branches; ++stream) {
      const int row = stream * width + d_out;
      const device uint8_t* block = up_weight + int64_t(row) * q8_row_bytes +
          block_id * KQ_Q8_0_BLOCK_BYTES;
      const float scale = float(kq_q8_0_d(block));
      const device int8_t* quant = kq_q8_0_q_ptr(block) + within;
      float partial = 0.0f;
#pragma unroll
      for (int i = 0; i < values_per_thread; ++i) {
        if (i < active) {
          partial += x_thread[i] * float(quant[i]);
        }
      }
      result[stream] += scale * partial;
    }
  }

#pragma unroll
  for (int stream = 0; stream < branches; ++stream) {
    result[stream] = simd_sum(result[stream]);
  }
  if (lane == 0) {
    ushort total_bits = 0u;
    for (int stream = 0; stream < branches; ++stream) {
      const bfloat16_t projected = bfloat16_t(result[stream]);
      const bfloat16_t gate = kq_qwen4_hc_sigmoid(projected);
      const bfloat16_t product =
          bfloat16_t(float(gate) * float(normalized[stream * width + d_out]));
      const float sum = float(as_type<bfloat16_t>(total_bits)) + float(product);
      total_bits = as_type<ushort>(bfloat16_t(sum));
    }
    mixed[d_out] = bfloat16_t(float(as_type<bfloat16_t>(total_bits)) * 0.25f);
  }
}
