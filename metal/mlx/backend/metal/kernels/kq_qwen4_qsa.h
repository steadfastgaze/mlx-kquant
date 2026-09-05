// Fixed-geometry Qwen sparse-selection and mutable K4/V4 gather kernels.

[[kernel]] void kq_qwen4_qsa_project_rope(
    const device bfloat16_t* index_projection [[buffer(0)]],
    const device bfloat16_t* query_projection [[buffer(1)]],
    const device bfloat16_t* key_projection [[buffer(2)]],
    const device bfloat16_t* index_norm_weight [[buffer(3)]],
    const device bfloat16_t* query_norm_weight [[buffer(4)]],
    const device bfloat16_t* key_norm_weight [[buffer(5)]],
    const device bfloat16_t* rope_cosine [[buffer(6)]],
    const device bfloat16_t* rope_sine [[buffer(7)]],
    device bfloat16_t* index_queries [[buffer(8)]],
    device bfloat16_t* raw_index_key [[buffer(9)]],
    device bfloat16_t* queries [[buffer(10)]],
    device bfloat16_t* query_gate [[buffer(11)]],
    device bfloat16_t* keys [[buffer(12)]],
    const constant float& eps [[buffer(13)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int index_heads = 4;
  constexpr int query_heads = 24;
  constexpr int index_width = 128;
  constexpr int head_width = 256;
  constexpr int rotary_width = 64;

  const int logical_head = int(tg.x);
  const bool is_index = logical_head < index_heads;
  const bool is_query =
      logical_head >= index_heads && logical_head < index_heads + query_heads;
  const int local_head = is_index
      ? logical_head
      : (is_query ? logical_head - index_heads
                  : logical_head - index_heads - query_heads);
  const int width = is_index ? index_width : head_width;
  const int per_thread = width / 32;
  const device bfloat16_t* source = is_index
      ? index_projection + local_head * index_width
      : (is_query ? query_projection + local_head * 2 * head_width
                  : key_projection + local_head * head_width);
  const device bfloat16_t* weight = is_index
      ? index_norm_weight
      : (is_query ? query_norm_weight : key_norm_weight);
  device bfloat16_t* destination = is_index
      ? index_queries + local_head * index_width
      : (is_query ? queries + local_head * head_width
                  : keys + local_head * head_width);

  float values[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < per_thread) {
      const int dimension = int(tid) * per_thread + i;
      values[i] = float(source[dimension]);
    }
  }

  float partial = 0.0f;
  if (width == head_width) {
    // Match MLX's width-256 row reduction: each lane consumes four adjacent
    // values from each 128-value block before the simdgroup reduction.
#pragma unroll
    for (int block = 0; block < 2; ++block) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int dimension = block * 128 + int(tid) * 4 + i;
        const float value = float(source[dimension]);
        partial += value * value;
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      partial += values[i] * values[i];
    }
  }
  const float sum_squares = simd_sum(partial);
  const float inv_rms = metal::precise::rsqrt(sum_squares / float(width) + eps);

  threadgroup bfloat16_t normalized[head_width];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < per_thread) {
      const int dimension = int(tid) * per_thread + i;
      normalized[dimension] =
          bfloat16_t(values[i] * inv_rms * (1.0f + float(weight[dimension])));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < per_thread) {
      const int dimension = int(tid) * per_thread + i;
      bfloat16_t output = normalized[dimension];
      if (dimension < rotary_width) {
        const int partner = dimension < rotary_width / 2
            ? dimension + rotary_width / 2
            : dimension - rotary_width / 2;
        const bfloat16_t rotated_half = dimension < rotary_width / 2
            ? bfloat16_t(-float(normalized[partner]))
            : normalized[partner];
        const bfloat16_t first =
            bfloat16_t(normalized[dimension] * rope_cosine[dimension]);
        const bfloat16_t second =
            bfloat16_t(rotated_half * rope_sine[dimension]);
        output = bfloat16_t(first + second);
      }
      destination[dimension] = output;
    }
  }

  if (logical_head == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dimension = int(tid) * 4 + i;
      raw_index_key[dimension] =
          index_projection[index_heads * index_width + dimension];
    }
  }
  if (is_query) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int dimension = int(tid) * 8 + i;
      query_gate[local_head * head_width + dimension] =
          source[head_width + dimension];
    }
  }
}

[[kernel]] void kq_qwen4_qsa_stable_select(
    const device float* scores [[buffer(0)]],
    device int* selected [[buffer(1)]],
    device bool* valid [[buffer(2)]],
    const constant int& group_count_value [[buffer(3)]],
    const constant int& visible_count_value [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint group_count = uint(group_count_value);
  const uint visible_count = uint(visible_count_value);

  // Nonnegative IEEE-754 values retain their order as unsigned integers. The
  // inverted physical id makes the lower id win an exact score tie.
  threadgroup ulong ordered[4096];
  for (uint part = 0; part < 8u; ++part) {
    const uint slot = tid + part * 512u;
    if (slot < group_count) {
      const uint score_bits = as_type<uint>(scores[slot]);
      ordered[slot] = (ulong(score_bits) << 32u) | ulong(0xffffffffu - slot);
    } else {
      ordered[slot] = 0ul;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint width = 2u; width <= 4096u; width <<= 1u) {
    for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
      for (uint part = 0; part < 8u; ++part) {
        const uint left = tid + part * 512u;
        const uint right = left ^ stride;
        if (right <= left) {
          continue;
        }
        const ulong left_key = ordered[left];
        const ulong right_key = ordered[right];
        const bool ascending = (left & width) == 0u;
        const bool swap =
            ascending ? right_key < left_key : left_key < right_key;
        if (swap) {
          ordered[left] = right_key;
          ordered[right] = left_key;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const ulong selected_key = ordered[3584u + tid];
  ordered[tid] = ulong(0xffffffffu - uint(selected_key));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Restore the physical context order consumed by released attention.
  for (uint width = 2u; width <= 512u; width <<= 1u) {
    for (uint stride = width >> 1u; stride > 0u; stride >>= 1u) {
      const uint right = tid ^ stride;
      if (right > tid) {
        const ulong left_id = ordered[tid];
        const ulong right_id = ordered[right];
        const bool ascending = (tid & width) == 0u;
        const bool swap = ascending ? right_id < left_id : left_id < right_id;
        if (swap) {
          ordered[tid] = right_id;
          ordered[right] = left_id;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const int group = int(ordered[tid]);
  for (uint lane = 0; lane < 4u; ++lane) {
    const uint output_lane = tid * 4u + lane;
    selected[output_lane] = group * 4 + int(lane);
    valid[output_lane] = true;
  }
  if (tid < 3u) {
    const uint output_lane = 2048u + tid;
    const uint tail = group_count * 4u + tid;
    const bool tail_valid = tail < visible_count;
    selected[output_lane] = tail_valid ? int(tail) : -1;
    valid[output_lane] = tail_valid;
  }
}

[[kernel]] void kq_qwen4_qsa_k4v4_mutable_gather(
    const device uchar* records [[buffer(0)]],
    const device bfloat16_t* exact_sink_keys [[buffer(1)]],
    const device bfloat16_t* exact_sink_values [[buffer(2)]],
    const device bfloat16_t* exact_tail_keys [[buffer(3)]],
    const device bfloat16_t* exact_tail_values [[buffer(4)]],
    const device bfloat16_t* pending_keys [[buffer(5)]],
    const device bfloat16_t* pending_values [[buffer(6)]],
    const device int* logical_ids [[buffer(7)]],
    const device bool* valid_rows [[buffer(8)]],
    device bfloat16_t* gathered_keys [[buffer(9)]],
    device bfloat16_t* gathered_values [[buffer(10)]],
    const constant int& record_count [[buffer(11)]],
    const constant int& frontier [[buffer(12)]],
    const constant int& tail_tokens [[buffer(13)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  constexpr int sink_tokens = 128;
  constexpr int tile_tokens = 128;
  constexpr int heads = 2;
  constexpr int head_width = 256;
  constexpr int record_bytes = 35072;

  const uint row = tg.x;
  const uint head = tg.y;
  if (lane >= 16u || head >= uint(heads)) {
    return;
  }

  const int logical = logical_ids[row];
  const bool lane_valid = valid_rows[row];
  const int body_frontier = sink_tokens + record_count * tile_tokens;
  const ulong row_base = (ulong(row) * heads + head) * head_width;

  if (!lane_valid || logical < 0 || logical >= frontier + 1) {
    for (uint r = 0; r < 16u; ++r) {
      const uint channel = r * 16u + lane;
      gathered_keys[row_base + channel] = bfloat16_t(0.0f);
      gathered_values[row_base + channel] = bfloat16_t(0.0f);
    }
    return;
  }

  if (logical < sink_tokens) {
    for (uint r = 0; r < 16u; ++r) {
      const uint channel = r * 16u + lane;
      const ulong source =
          (ulong(head) * sink_tokens + uint(logical)) * head_width + channel;
      gathered_keys[row_base + channel] = exact_sink_keys[source];
      gathered_values[row_base + channel] = exact_sink_values[source];
    }
    return;
  }

  if (logical >= body_frontier) {
    int local;
    const device bfloat16_t* source_keys;
    const device bfloat16_t* source_values;
    int source_tokens;
    if (logical < frontier) {
      local = (logical - sink_tokens) % tail_tokens;
      source_keys = exact_tail_keys;
      source_values = exact_tail_values;
      source_tokens = tail_tokens;
    } else {
      local = logical - frontier;
      source_keys = pending_keys;
      source_values = pending_values;
      source_tokens = 1;
    }
    for (uint r = 0; r < 16u; ++r) {
      const uint channel = r * 16u + lane;
      const ulong source =
          (ulong(head) * uint(source_tokens) + uint(local)) * head_width +
          channel;
      gathered_keys[row_base + channel] = source_keys[source];
      gathered_values[row_base + channel] = source_values[source];
    }
    return;
  }

  const int body_local = logical - sink_tokens;
  const int tile = body_local / tile_tokens;
  const int token = body_local % tile_tokens;
  const device uchar* record =
      records + (ulong(tile) * heads + head) * record_bytes;
  threadgroup float key_buffer[head_width];
  threadgroup float value_buffer[head_width];

  for (uint r = 0; r < 16u; ++r) {
    const uint channel = r * 16u + lane;
    const uint code_column = channel >> 1u;
    const uint shift = (channel & 1u) * 4u;

    const uchar packed_k = record[ulong(token) * 128u + code_column];
    const float k_code = float((packed_k >> shift) & 0x0fu);
    const device half* k_scale =
        reinterpret_cast<const device half*>(record + 16384u);
    const device half* k_zero =
        reinterpret_cast<const device half*>(record + 16896u);
    const device half* k_token_scale =
        reinterpret_cast<const device half*>(record + 17408u);
    key_buffer[channel] =
        (k_code * float(k_scale[channel]) + float(k_zero[channel])) *
        float(k_token_scale[token]);

    const uchar packed_v = record[17664u + ulong(token) * 128u + code_column];
    const float v_code = float((packed_v >> shift) & 0x0fu);
    const device half* v_channel_scale =
        reinterpret_cast<const device half*>(record + 34048u);
    const device half* v_token_scale =
        reinterpret_cast<const device half*>(record + 34560u);
    const device half* v_zero =
        reinterpret_cast<const device half*>(record + 34816u);
    value_buffer[channel] =
        (v_code * float(v_token_scale[token]) + float(v_zero[token])) *
        float(v_channel_scale[channel]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float key_values[16];
  float value_values[16];
  short h = 1;
  for (short stage = 0; stage < 2; ++stage) {
    const short k = short(lane) & (h - 1);
    const short j = ((short(lane) - k) << 4) + k;
    for (short r = 0; r < 16; ++r) {
      key_values[r] = key_buffer[j + h * r];
      value_values[r] = value_buffer[j + h * r];
    }
    short local_h = 1;
    for (short local_stage = 0; local_stage < 4; ++local_stage) {
      for (short i = 0; i < 8; ++i) {
        const short local_k = i & (local_h - 1);
        const short local_j = ((i - local_k) << 1) + local_k;
        const float key_a = key_values[local_j];
        const float key_b = key_values[local_j + local_h];
        key_values[local_j] = key_a + key_b;
        key_values[local_j + local_h] = key_a - key_b;
        const float value_a = value_values[local_j];
        const float value_b = value_values[local_j + local_h];
        value_values[local_j] = value_a + value_b;
        value_values[local_j + local_h] = value_a - value_b;
      }
      local_h <<= 1;
    }
    for (short r = 0; r < 16; ++r) {
      key_buffer[j + h * r] = key_values[r];
      value_buffer[j + h * r] = value_values[r];
    }
    h <<= 4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint r = 0; r < 16u; ++r) {
    const uint channel = r * 16u + lane;
    gathered_keys[row_base + channel] =
        bfloat16_t(key_buffer[channel] * 0.0625f);
    gathered_values[row_base + channel] =
        bfloat16_t(value_buffer[channel] * 0.0625f);
  }
}
