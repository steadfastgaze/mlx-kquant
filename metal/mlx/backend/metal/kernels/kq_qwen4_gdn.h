// Fixed-geometry boundaries around the released Qwen gated-delta recurrence.

template <typename T>
METAL_FUNC T kq_qwen4_gdn_sigmoid(T x) {
  const auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

[[kernel]] void kq_qwen4_gdn_prepare(
    const device bfloat16_t* qkv [[buffer(0)]],
    const device bfloat16_t* beta_logits [[buffer(1)]],
    const device bfloat16_t* decay_logits [[buffer(2)]],
    const device bfloat16_t* conv_state [[buffer(3)]],
    const device bfloat16_t* conv_weight [[buffer(4)]],
    const device bfloat16_t* a_log [[buffer(5)]],
    const device bfloat16_t* dt_bias [[buffer(6)]],
    device bfloat16_t* query [[buffer(7)]],
    device bfloat16_t* key [[buffer(8)]],
    device bfloat16_t* value [[buffer(9)]],
    device bfloat16_t* beta [[buffer(10)]],
    device float* decay [[buffer(11)]],
    device bfloat16_t* next_conv_state [[buffer(12)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int conv_channels = 10240;
  constexpr int state_rows = 3;
  constexpr int kernel_width = 4;
  constexpr int key_heads = 16;
  constexpr int value_heads = 48;
  constexpr int head_width = 128;
  constexpr int key_width = key_heads * head_width;
  constexpr int values_per_thread = 4;

  const int prepared_head = int(tg.x);
  int source_base;
  device bfloat16_t* destination;
  bool normalize;
  bool scale_query;
  if (prepared_head < key_heads) {
    source_base = prepared_head * head_width;
    destination = query;
    normalize = true;
    scale_query = true;
  } else if (prepared_head < 2 * key_heads) {
    source_base = key_width + (prepared_head - key_heads) * head_width;
    destination = key;
    normalize = true;
    scale_query = false;
  } else {
    source_base = 2 * key_width + (prepared_head - 2 * key_heads) * head_width;
    destination = value;
    normalize = false;
    scale_query = false;
  }

  bfloat16_t activated[values_per_thread];
  const int first = int(tid) * values_per_thread;
#pragma unroll
  for (int i = 0; i < values_per_thread; ++i) {
    const int channel = source_base + first + i;
    float accumulator = 0.0f;
#pragma unroll
    for (int tap = 0; tap < kernel_width; ++tap) {
      const bfloat16_t source = tap < state_rows
          ? conv_state[tap * conv_channels + channel]
          : qkv[channel];
      accumulator +=
          float(source) * float(conv_weight[channel * kernel_width + tap]);
    }
    const bfloat16_t convolved = bfloat16_t(accumulator);
    activated[i] = bfloat16_t(convolved * kq_qwen4_gdn_sigmoid(convolved));
    next_conv_state[channel] = conv_state[conv_channels + channel];
    next_conv_state[conv_channels + channel] =
        conv_state[2 * conv_channels + channel];
    next_conv_state[2 * conv_channels + channel] = qkv[channel];
  }

  bfloat16_t inv_norm = bfloat16_t(1.0f);
  if (normalize) {
    bfloat16_t partial = bfloat16_t(0.0f);
#pragma unroll
    for (int i = 0; i < values_per_thread; ++i) {
      const bfloat16_t square = bfloat16_t(activated[i] * activated[i]);
      partial = bfloat16_t(partial + square);
    }
    const bfloat16_t sum_squares = simd_sum(partial);
    inv_norm = bfloat16_t(
        metal::precise::rsqrt(bfloat16_t(sum_squares + bfloat16_t(1e-6f))));
  }

  const int destination_base = normalize
      ? (prepared_head % key_heads) * head_width
      : (prepared_head - 2 * key_heads) * head_width;
  const bfloat16_t query_scale = bfloat16_t(0.08838834764831845f);
#pragma unroll
  for (int i = 0; i < values_per_thread; ++i) {
    bfloat16_t prepared =
        normalize ? bfloat16_t(activated[i] * inv_norm) : activated[i];
    if (scale_query) {
      prepared = bfloat16_t(prepared * query_scale);
    }
    destination[destination_base + first + i] = prepared;
  }

  if (prepared_head < value_heads && tid == 0) {
    beta[prepared_head] = kq_qwen4_gdn_sigmoid(beta_logits[prepared_head]);
    const float x =
        float(decay_logits[prepared_head]) + float(dt_bias[prepared_head]);
    const float maximum = metal::max(x, 0.0f);
    const float minimum = metal::min(x, 0.0f);
    const float softplus = maximum + log1p(metal::exp(minimum - maximum));
    const float rate =
        metal::precise::exp(float(a_log[prepared_head])) * softplus;
    decay[prepared_head] = metal::precise::exp(-rate);
  }
}

[[kernel]] void kq_qwen4_gdn_norm_gate(
    const device bfloat16_t* recurrence [[buffer(0)]],
    const device bfloat16_t* gate [[buffer(1)]],
    const device bfloat16_t* norm_weight [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    const constant float& eps [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int head_width = 128;
  constexpr int values_per_thread = 4;
  const int head = int(tg.x);
  const int head_base = head * head_width;
  const int first = int(tid) * values_per_thread;
  float source[values_per_thread];
  float partial = 0.0f;
#pragma unroll
  for (int i = 0; i < values_per_thread; ++i) {
    source[i] = float(recurrence[head_base + first + i]);
    partial += source[i] * source[i];
  }
  const float sum_squares = simd_sum(partial);
  const float inv_rms =
      metal::precise::rsqrt(sum_squares / float(head_width) + eps);
#pragma unroll
  for (int i = 0; i < values_per_thread; ++i) {
    const int dimension = first + i;
    const int index = head_base + dimension;
    const bfloat16_t normalized = bfloat16_t(source[i] * inv_rms);
    const bfloat16_t weighted = bfloat16_t(norm_weight[dimension] * normalized);
    output[index] =
        bfloat16_t(float(weighted) * kq_qwen4_gdn_sigmoid(float(gate[index])));
  }
}

[[kernel]] void kq_qwen4_gdn_recurrence(
    const device bfloat16_t* query [[buffer(0)]],
    const device bfloat16_t* key [[buffer(1)]],
    const device bfloat16_t* value [[buffer(2)]],
    const device float* decay [[buffer(3)]],
    const device bfloat16_t* beta [[buffer(4)]],
    const device float* state_in [[buffer(5)]],
    device bfloat16_t* output [[buffer(6)]],
    device float* state_out [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int key_heads = 16;
  constexpr int value_heads = 48;
  constexpr int head_width = 128;
  constexpr int state_values_per_thread = head_width / 32;
  const int value_head = int(gid.z);
  const int key_head = value_head / (value_heads / key_heads);
  const int value_dimension = int(gid.y);
  const int key_base = key_head * head_width;
  const int value_base = value_head * head_width;
  const int state_base =
      (value_head * head_width + value_dimension) * head_width;
  float state[state_values_per_thread];
#pragma unroll
  for (int i = 0; i < state_values_per_thread; ++i) {
    const int dimension = state_values_per_thread * int(tid.x) + i;
    state[i] = state_in[state_base + dimension];
  }
  float memory = 0.0f;
#pragma unroll
  for (int i = 0; i < state_values_per_thread; ++i) {
    const int dimension = state_values_per_thread * int(tid.x) + i;
    state[i] *= decay[value_head];
    memory += state[i] * float(key[key_base + dimension]);
  }
  memory = simd_sum(memory);
  const float delta = (float(value[value_base + value_dimension]) - memory) *
      float(beta[value_head]);
  float projected = 0.0f;
#pragma unroll
  for (int i = 0; i < state_values_per_thread; ++i) {
    const int dimension = state_values_per_thread * int(tid.x) + i;
    state[i] += float(key[key_base + dimension]) * delta;
    projected += state[i] * float(query[key_base + dimension]);
  }
  projected = simd_sum(projected);
  if (simd_lid == 0) {
    output[value_base + value_dimension] = bfloat16_t(projected);
  }
#pragma unroll
  for (int i = 0; i < state_values_per_thread; ++i) {
    const int dimension = state_values_per_thread * int(tid.x) + i;
    state_out[state_base + dimension] = state[i];
  }
}
