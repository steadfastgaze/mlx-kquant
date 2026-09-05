// Fixed released BF16 [T, 512] router with stable top-10 selection.
[[kernel]] void kq_qwen4_router_fused_exact(
    const device bfloat16_t* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device bfloat16_t* selected_scores [[buffer(2)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint experts = 512;
  constexpr uint routes = 10;
  constexpr uint softmax_threads = experts / 4;

  threadgroup float probabilities[experts];
  threadgroup uint ids[experts];
  threadgroup float softmax_max[32];
  threadgroup float softmax_sum[32];
  threadgroup float chosen_scores[routes];
  threadgroup uint chosen_ids[routes];
  threadgroup float ranked_scores[routes];
  threadgroup uint ranked_ids[routes];
  threadgroup float selected_sum;

  float values[4] = {
      Limits<float>::finite_min,
      Limits<float>::finite_min,
      Limits<float>::finite_min,
      Limits<float>::finite_min,
  };
  if (simd_gid == 0) {
    softmax_max[simd_lid] = Limits<float>::min;
    softmax_sum[simd_lid] = 0.0f;
  }
  if (tid < softmax_threads) {
    const uint offset = tg.x * experts + tid * 4;
    for (uint i = 0; i < 4; ++i) {
      values[i] = float(logits[offset + i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < softmax_threads) {
    float local_max = Limits<float>::finite_min;
    for (uint i = 0; i < 4; ++i) {
      local_max = local_max < values[i] ? values[i] : local_max;
    }
    local_max = simd_max(local_max);
    if (simd_lid == 0) {
      softmax_max[simd_gid] = local_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    const float global_max = simd_max(softmax_max[simd_lid]);
    if (simd_lid == 0) {
      softmax_max[0] = global_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < softmax_threads) {
    const float global_max = softmax_max[0];
    float local_sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      values[i] = fast::exp(values[i] - global_max);
      local_sum += values[i];
    }
    local_sum = simd_sum(local_sum);
    if (simd_lid == 0) {
      softmax_sum[simd_gid] = local_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    const float global_sum = simd_sum(softmax_sum[simd_lid]);
    if (simd_lid == 0) {
      softmax_sum[0] = global_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < softmax_threads) {
    const float inverse_sum = 1.0f / softmax_sum[0];
    const uint offset = tid * 4;
    for (uint i = 0; i < 4; ++i) {
      probabilities[offset + i] = values[i] * inverse_sum;
    }
  }
  ids[tid] = tid;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint route = 0; route < routes; ++route) {
    for (uint stride = experts / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float left_value = probabilities[tid];
        const float right_value = probabilities[tid + stride];
        const uint left_id = ids[tid];
        const uint right_id = ids[tid + stride];
        const bool left_nan = metal::isnan(left_value);
        const bool right_nan = metal::isnan(right_value);
        const bool right_wins = (left_nan && right_nan && right_id > left_id) ||
            (!left_nan && right_nan) ||
            (!left_nan && !right_nan &&
             (right_value > left_value ||
              (right_value == left_value && right_id > left_id)));
        if (right_wins) {
          probabilities[tid] = right_value;
          probabilities[tid + stride] = left_value;
          ids[tid] = right_id;
          ids[tid + stride] = left_id;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
      chosen_scores[route] = probabilities[0];
      chosen_ids[route] = ids[0];
      probabilities[0] = -metal::numeric_limits<float>::infinity();
      ids[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid < routes) {
    const float score = chosen_scores[tid];
    const uint id = chosen_ids[tid];
    uint rank = 0;
    for (uint other = 0; other < routes; ++other) {
      const float other_score = chosen_scores[other];
      const uint other_id = chosen_ids[other];
      const bool score_nan = metal::isnan(score);
      const bool other_nan = metal::isnan(other_score);
      const bool before = (!score_nan && other_nan) ||
          (score_nan == other_nan &&
           (other_score > score || (other_score == score && other_id < id)));
      rank += uint(before);
    }
    ranked_ids[rank] = id;
    ranked_scores[rank] = score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    float denominator = 0.0f;
    for (uint route = 0; route < routes; ++route) {
      denominator = ranked_scores[route] + denominator;
    }
    selected_sum = denominator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < routes) {
    const uint output = tg.x * routes + tid;
    indices[output] = ranked_ids[tid];
    selected_scores[output] =
        static_cast<bfloat16_t>(ranked_scores[tid] / selected_sum);
  }
}
