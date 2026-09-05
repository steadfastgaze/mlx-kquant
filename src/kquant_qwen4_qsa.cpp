// Fixed-geometry Qwen sparse selection and mutable K4/V4 row gather.
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kquant.h"
#include "kquant_internal.h"

#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h"
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

constexpr int kMinGroups = 512;
constexpr int kMaxGroups = 4096;
constexpr int kBlockBudget = 512;
constexpr int kCompressRatio = 4;
constexpr int kSelectedWidth =
    kBlockBudget * kCompressRatio + kCompressRatio - 1;
constexpr int kKVHeads = 2;
constexpr int kHeadWidth = 256;
constexpr int kSinkTokens = 128;
constexpr int kTailTokens = 1152;
constexpr int kTileTokens = 128;
constexpr int kRecordBytes = 35072;
constexpr int kMaxRecordCount =
    (std::numeric_limits<int>::max() - kSinkTokens) / kTileTokens;
constexpr int kHiddenWidth = 2560;
constexpr int kIndexQueryHeads = 4;
constexpr int kIndexKVHeads = 1;
constexpr int kIndexHeadWidth = 128;
constexpr int kIndexProjectionWidth =
    (kIndexQueryHeads + kIndexKVHeads) * kIndexHeadWidth;
constexpr int kQueryHeads = 24;
constexpr int kQueryProjectionWidth = 2 * kQueryHeads * kHeadWidth;
constexpr int kKVProjectionWidth = kKVHeads * kHeadWidth;
constexpr int kRotaryWidth = 64;
constexpr int kQ6HiddenRowBytes = kHiddenWidth * 210 / 256;
constexpr int kQ6AttentionRowBytes = (kQueryHeads * kHeadWidth) * 210 / 256;

void require_shape(
    const mx::array& value,
    mx::Dtype dtype,
    const mx::Shape& shape,
    const char* op,
    const char* what) {
  if (value.dtype() != dtype || value.shape() != shape) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " has incompatible shape or dtype.");
  }
}

void require_q6_wire(
    const mx::array& value,
    int rows,
    int row_bytes,
    const char* op,
    const char* what) {
  if (value.dtype() != mx::uint8 || value.ndim() != 2 ||
      value.shape(0) != rows || value.shape(1) != row_bytes) {
    throw std::invalid_argument(
        std::string(op) + " " + what +
        " must be a uint8 Q6_K wire with the released decode shape.");
  }
}

void require_scales(const mx::array& value, const char* op, const char* what) {
  if (value.dtype() != mx::uint8 || value.size() != 1) {
    throw std::invalid_argument(
        std::string(op) + " " + what +
        " must be the size-one uint8 K-quant placeholder.");
  }
}

mx::array row_contiguous(mx::array value, mx::Stream stream) {
  return mx::contiguous(value, false, stream);
}

} // namespace

std::vector<mx::Shape> KQuantQwen4QSAProjectRopeQ6::output_shapes(
    const std::vector<mx::array>&) {
  return {
      {1, 1, kIndexQueryHeads, kIndexHeadWidth},
      {1, 1, kIndexHeadWidth},
      {1, 1, kQueryHeads, kHeadWidth},
      {1, 1, kQueryHeads * kHeadWidth},
      {1, kKVHeads, 1, kHeadWidth},
      {1, kKVHeads, 1, kHeadWidth},
  };
}

bool KQuantQwen4QSAProjectRopeQ6::is_equivalent(
    const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQwen4QSAProjectRopeQ6&>(other);
  return eps_ == o.eps_;
}

void KQuantQwen4QSAProjectRopeQ6::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_qsa_project_rope_q6] requires Metal.");
}

std::vector<mx::Shape> KQuantQwen4QSASelectGatherK4V4::output_shapes(
    const std::vector<mx::array>&) {
  return {
      {1, 1, kSelectedWidth},
      {1, 1, kSelectedWidth},
      {kSelectedWidth, kKVHeads, kHeadWidth},
      {kSelectedWidth, kKVHeads, kHeadWidth},
  };
}

bool KQuantQwen4QSASelectGatherK4V4::is_equivalent(
    const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQwen4QSASelectGatherK4V4&>(other);
  return group_count_ == o.group_count_ && visible_count_ == o.visible_count_ &&
      record_count_ == o.record_count_ && tail_tokens_ == o.tail_tokens_ &&
      frontier_ == o.frontier_;
}

void KQuantQwen4QSASelectGatherK4V4::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_qsa_select_gather_k4v4] requires Metal.");
}

#ifdef _METAL_

namespace {

mx::array make_temporary(const mx::Shape& shape, mx::Dtype dtype) {
  mx::array value(shape, dtype, nullptr, {});
  value.set_data(mx::allocator::malloc(value.nbytes()));
  return value;
}

void dispatch_q6_qmv(
    mx::metal::Device& device,
    const mx::Stream& stream,
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    mx::array& output,
    int input_width,
    int output_width) {
  constexpr int group_size = 256;
  constexpr int bits = 6;
  constexpr int tile_width = 8;
  const std::string kernel_name = kq_kname_prefix("q6_k") + "qmv_fast_" +
      kq_type_string(x.dtype()) + "_gs_256_b_6_batch_0";
  auto kernel = kq_get_kernel(device, kernel_name);
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  encoder.set_input_array(weight, index++);
  encoder.set_input_array(scales, index++);
  encoder.set_input_array(x, index++);
  encoder.set_output_array(output, index++);
  encoder.set_bytes(input_width, index++);
  encoder.set_bytes(output_width, index++);
  add_strides_and_shapes(encoder, true, x, weight, scales, index);
  encoder.dispatch_threadgroups(
      MTL::Size(1, (output_width + tile_width - 1) / tile_width, 1),
      MTL::Size(32, 2, 1));
}

} // namespace

void KQuantQwen4QSAProjectRopeQ6::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  std::vector<mx::array> temporaries;
  temporaries.reserve(3);
  temporaries.push_back(
      make_temporary({1, 1, kIndexProjectionWidth}, mx::bfloat16));
  temporaries.push_back(
      make_temporary({1, 1, kQueryProjectionWidth}, mx::bfloat16));
  temporaries.push_back(
      make_temporary({1, 1, kKVProjectionWidth}, mx::bfloat16));
  auto& index_projection = temporaries[0];
  auto& query_projection = temporaries[1];
  auto& key_projection = temporaries[2];

  dispatch_q6_qmv(
      device,
      stream,
      inputs[0],
      inputs[1],
      inputs[2],
      index_projection,
      kHiddenWidth,
      kIndexProjectionWidth);
  dispatch_q6_qmv(
      device,
      stream,
      inputs[0],
      inputs[3],
      inputs[4],
      query_projection,
      kHiddenWidth,
      kQueryProjectionWidth);
  dispatch_q6_qmv(
      device,
      stream,
      inputs[0],
      inputs[5],
      inputs[6],
      key_projection,
      kHiddenWidth,
      kKVProjectionWidth);
  dispatch_q6_qmv(
      device,
      stream,
      inputs[0],
      inputs[7],
      inputs[8],
      outputs[5],
      kHiddenWidth,
      kKVProjectionWidth);

  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.barrier();
  auto prepare = kq_get_kernel(device, "kq_qwen4_qsa_project_rope");
  encoder.set_compute_pipeline_state(prepare);
  encoder.set_input_array(index_projection, 0);
  encoder.set_input_array(query_projection, 1);
  encoder.set_input_array(key_projection, 2);
  encoder.set_input_array(inputs[9], 3);
  encoder.set_input_array(inputs[10], 4);
  encoder.set_input_array(inputs[11], 5);
  encoder.set_input_array(inputs[12], 6);
  encoder.set_input_array(inputs[13], 7);
  for (int index = 0; index < 5; ++index) {
    encoder.set_output_array(outputs[index], 8 + index);
  }
  encoder.set_bytes(eps_, 13);
  encoder.dispatch_threadgroups(MTL::Size(30, 1, 1), MTL::Size(32, 1, 1));
  encoder.add_temporaries(std::move(temporaries));
}

void KQuantQwen4QSASelectGatherK4V4::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  auto& encoder = mx::metal::get_command_encoder(stream);
  auto selector = kq_get_kernel(device, "kq_qwen4_qsa_stable_select");
  encoder.set_compute_pipeline_state(selector);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_output_array(outputs[0], 1);
  encoder.set_output_array(outputs[1], 2);
  encoder.set_bytes(group_count_, 3);
  encoder.set_bytes(visible_count_, 4);
  encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(512, 1, 1));

  // The gather consumes selector outputs in this same compute encoder.
  encoder.barrier();

  auto gather = kq_get_kernel(device, "kq_qwen4_qsa_k4v4_mutable_gather");
  encoder.set_compute_pipeline_state(gather);
  encoder.set_input_array(inputs[1], 0);
  encoder.set_input_array(inputs[2], 1);
  encoder.set_input_array(inputs[3], 2);
  encoder.set_input_array(inputs[4], 3);
  encoder.set_input_array(inputs[5], 4);
  encoder.set_input_array(inputs[6], 5);
  encoder.set_input_array(inputs[7], 6);
  encoder.set_input_array(outputs[0], 7);
  encoder.set_input_array(outputs[1], 8);
  encoder.set_output_array(outputs[2], 9);
  encoder.set_output_array(outputs[3], 10);
  encoder.set_bytes(record_count_, 11);
  encoder.set_bytes(frontier_, 12);
  encoder.set_bytes(tail_tokens_, 13);
  encoder.dispatch_threadgroups(
      MTL::Size(kSelectedWidth, kKVHeads, 1), MTL::Size(16, 1, 1));
}

#else

void KQuantQwen4QSAProjectRopeQ6::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_qsa_project_rope_q6] requires Metal.");
}

void KQuantQwen4QSASelectGatherK4V4::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_qsa_select_gather_k4v4] requires Metal.");
}

#endif

std::vector<mx::array> qwen4_qsa_project_rope_q6(
    mx::array hidden,
    mx::array index_weight,
    mx::array index_scales,
    mx::array query_weight,
    mx::array query_scales,
    mx::array key_weight,
    mx::array key_scales,
    mx::array value_weight,
    mx::array value_scales,
    mx::array index_query_norm_weight,
    mx::array query_norm_weight,
    mx::array key_norm_weight,
    mx::array rope_cosine,
    mx::array rope_sine,
    mx::array position_ids,
    float eps,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_qsa_project_rope_q6]";
  auto stream = mx::to_stream(stream_or_device);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument(std::string(op) + " requires a Metal stream.");
  }
  require_shape(hidden, mx::bfloat16, {1, 1, kHiddenWidth}, op, "hidden");
  require_q6_wire(
      index_weight,
      kIndexProjectionWidth,
      kQ6HiddenRowBytes,
      op,
      "index_weight");
  require_q6_wire(
      query_weight,
      kQueryProjectionWidth,
      kQ6HiddenRowBytes,
      op,
      "query_weight");
  require_q6_wire(
      key_weight, kKVProjectionWidth, kQ6HiddenRowBytes, op, "key_weight");
  require_q6_wire(
      value_weight, kKVProjectionWidth, kQ6HiddenRowBytes, op, "value_weight");
  require_scales(index_scales, op, "index_scales");
  require_scales(query_scales, op, "query_scales");
  require_scales(key_scales, op, "key_scales");
  require_scales(value_scales, op, "value_scales");
  require_shape(
      index_query_norm_weight,
      mx::bfloat16,
      {kIndexHeadWidth},
      op,
      "index_query_norm_weight");
  require_shape(
      query_norm_weight, mx::bfloat16, {kHeadWidth}, op, "query_norm_weight");
  require_shape(
      key_norm_weight, mx::bfloat16, {kHeadWidth}, op, "key_norm_weight");
  require_shape(
      rope_cosine, mx::bfloat16, {1, 1, 1, kRotaryWidth}, op, "rope_cosine");
  require_shape(
      rope_sine, mx::bfloat16, {1, 1, 1, kRotaryWidth}, op, "rope_sine");
  if ((position_ids.dtype() != mx::int32 &&
       position_ids.dtype() != mx::int64) ||
      position_ids.shape() != mx::Shape{3, 1, 1}) {
    throw std::invalid_argument(
        std::string(op) + " position_ids must be int32 or int64 [3, 1, 1].");
  }
  if (!(eps > 0.0f)) {
    throw std::invalid_argument(std::string(op) + " eps must be positive.");
  }

  std::vector<mx::array> inputs;
  inputs.reserve(15);
  for (auto& input : std::vector<mx::array>{
           std::move(hidden),
           std::move(index_weight),
           std::move(index_scales),
           std::move(query_weight),
           std::move(query_scales),
           std::move(key_weight),
           std::move(key_scales),
           std::move(value_weight),
           std::move(value_scales),
           std::move(index_query_norm_weight),
           std::move(query_norm_weight),
           std::move(key_norm_weight),
           std::move(rope_cosine),
           std::move(rope_sine),
           std::move(position_ids),
       }) {
    inputs.push_back(row_contiguous(std::move(input), stream));
  }

  return mx::array::make_arrays(
      {
          {1, 1, kIndexQueryHeads, kIndexHeadWidth},
          {1, 1, kIndexHeadWidth},
          {1, 1, kQueryHeads, kHeadWidth},
          {1, 1, kQueryHeads * kHeadWidth},
          {1, kKVHeads, 1, kHeadWidth},
          {1, kKVHeads, 1, kHeadWidth},
      },
      {
          mx::bfloat16,
          mx::bfloat16,
          mx::bfloat16,
          mx::bfloat16,
          mx::bfloat16,
          mx::bfloat16,
      },
      std::make_shared<KQuantQwen4QSAProjectRopeQ6>(stream, eps),
      std::move(inputs));
}

std::vector<mx::array> qwen4_qsa_select_gather_k4v4(
    mx::array scores,
    mx::array records,
    mx::array exact_sink_keys,
    mx::array exact_sink_values,
    mx::array exact_tail_keys,
    mx::array exact_tail_values,
    mx::array pending_keys,
    mx::array pending_values,
    int visible_count,
    int frontier,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_qsa_select_gather_k4v4]";
  auto stream = mx::to_stream(stream_or_device);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument(std::string(op) + " requires a Metal stream.");
  }

  if (scores.dtype() != mx::float32 || scores.ndim() != 3 ||
      scores.shape(0) != 1 || scores.shape(1) != 1 ||
      scores.shape(2) < kMinGroups || scores.shape(2) > kMaxGroups) {
    throw std::invalid_argument(
        std::string(op) +
        " scores must be float32 [1, 1, G] for 512 <= G <= 4096.");
  }
  const int group_count = static_cast<int>(scores.shape(2));
  if (visible_count < 0 || visible_count / kCompressRatio != group_count ||
      visible_count - group_count * kCompressRatio >= kCompressRatio) {
    throw std::invalid_argument(
        std::string(op) + " visible_count is incompatible with score groups.");
  }
  if (frontier < kSinkTokens || visible_count != frontier + 1) {
    throw std::invalid_argument(
        std::string(op) +
        " requires one pending token immediately after the frontier.");
  }

  if (records.dtype() != mx::uint8 || records.ndim() != 3 ||
      records.shape(1) != kKVHeads || records.shape(2) != kRecordBytes) {
    throw std::invalid_argument(
        std::string(op) + " records must be uint8 [tiles, 2, 35072].");
  }
  const auto record_count_size = static_cast<std::size_t>(records.shape(0));
  if (record_count_size > static_cast<std::size_t>(kMaxRecordCount)) {
    throw std::invalid_argument(
        std::string(op) + " record count exceeds the safe frontier range.");
  }
  const int record_count = static_cast<int>(record_count_size);
  if (exact_tail_keys.dtype() != mx::bfloat16 || exact_tail_keys.ndim() != 4 ||
      exact_tail_keys.shape(0) != 1 || exact_tail_keys.shape(1) != kKVHeads ||
      exact_tail_keys.shape(2) < kTileTokens ||
      exact_tail_keys.shape(2) % kTileTokens != 0 ||
      exact_tail_keys.shape(2) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      exact_tail_keys.shape(3) != kHeadWidth) {
    throw std::invalid_argument(
        std::string(op) + " exact_tail_keys has incompatible shape or dtype.");
  }
  const int tail_tokens = static_cast<int>(exact_tail_keys.shape(2));
  const int body_frontier = kSinkTokens + record_count * kTileTokens;
  if (body_frontier > frontier || frontier - body_frontier > tail_tokens) {
    throw std::invalid_argument(
        std::string(op) + " records and exact tail do not cover the frontier.");
  }

  const mx::Shape sink_shape = {1, kKVHeads, kSinkTokens, kHeadWidth};
  const mx::Shape tail_shape = {1, kKVHeads, tail_tokens, kHeadWidth};
  const mx::Shape pending_shape = {1, kKVHeads, 1, kHeadWidth};
  require_shape(
      exact_sink_keys, mx::bfloat16, sink_shape, op, "exact_sink_keys");
  require_shape(
      exact_sink_values, mx::bfloat16, sink_shape, op, "exact_sink_values");
  require_shape(
      exact_tail_keys, mx::bfloat16, tail_shape, op, "exact_tail_keys");
  require_shape(
      exact_tail_values, mx::bfloat16, tail_shape, op, "exact_tail_values");
  require_shape(pending_keys, mx::bfloat16, pending_shape, op, "pending_keys");
  require_shape(
      pending_values, mx::bfloat16, pending_shape, op, "pending_values");

  std::vector<mx::array> inputs;
  inputs.reserve(8);
  for (auto& input : std::vector<mx::array>{
           std::move(scores),
           std::move(records),
           std::move(exact_sink_keys),
           std::move(exact_sink_values),
           std::move(exact_tail_keys),
           std::move(exact_tail_values),
           std::move(pending_keys),
           std::move(pending_values),
       }) {
    inputs.push_back(row_contiguous(std::move(input), stream));
  }

  return mx::array::make_arrays(
      {
          {1, 1, kSelectedWidth},
          {1, 1, kSelectedWidth},
          {kSelectedWidth, kKVHeads, kHeadWidth},
          {kSelectedWidth, kKVHeads, kHeadWidth},
      },
      {mx::int32, mx::bool_, mx::bfloat16, mx::bfloat16},
      std::make_shared<KQuantQwen4QSASelectGatherK4V4>(
          stream,
          group_count,
          visible_count,
          record_count,
          tail_tokens,
          frontier),
      std::move(inputs));
}

} // namespace mlx_kquant
