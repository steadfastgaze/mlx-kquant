// Fixed-geometry decode stages for the released Qwen gated-residual graph.
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kquant.h"

#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "kquant_metal_internal.h"
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

constexpr int kBranches = 4;
constexpr int kWidth = 2560;
constexpr int kExpanded = kBranches * kWidth;
constexpr int kLowRank = 320;
constexpr int kQ6RowBytes = kExpanded * 210 / 256;
constexpr int kQ8RowBytes = kLowRank * 34 / 32;

void require_bf16_one_row(
    const mx::array& value,
    int width,
    const char* op,
    const char* what) {
  if (value.dtype() != mx::bfloat16 || value.ndim() < 1 ||
      value.shape(-1) != width || value.size() != width) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be bfloat16 with exactly one " +
        "row of width " + std::to_string(width) + ".");
  }
}

void require_wire(
    const mx::array& value,
    int rows,
    int bytes,
    const char* op,
    const char* what) {
  if (value.dtype() != mx::uint8 || value.ndim() != 2 ||
      value.shape(0) != rows || value.shape(1) != bytes) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be a uint8 wire with shape [" +
        std::to_string(rows) + ", " + std::to_string(bytes) + "].");
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

mx::Shape replace_last(mx::Shape shape, int width) {
  shape.back() = width;
  return shape;
}

} // namespace

std::vector<mx::Shape> KQuantQwen4HCNorm::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape(), inputs[0].shape()};
}

bool KQuantQwen4HCNorm::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQwen4HCNorm&>(other);
  return eps_ == o.eps_ && has_pending_ == o.has_pending_;
}

void KQuantQwen4HCNorm::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_hc_norm] requires Metal.");
}

std::vector<mx::Shape> KQuantQwen4HCFront::output_shapes(
    const std::vector<mx::array>& inputs) {
  auto lowrank = replace_last(inputs[0].shape(), kLowRank);
  if (!has_injection_) {
    return {std::move(lowrank)};
  }
  return {std::move(lowrank), replace_last(inputs[0].shape(), kBranches)};
}

bool KQuantQwen4HCFront::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQwen4HCFront&>(other);
  return has_injection_ == o.has_injection_;
}

void KQuantQwen4HCFront::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_hc_front] requires Metal.");
}

std::vector<mx::Shape> KQuantQwen4HCEpilogue::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {replace_last(inputs[0].shape(), kWidth)};
}

bool KQuantQwen4HCEpilogue::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQuantQwen4HCEpilogue::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_hc_epilogue] requires Metal.");
}

#ifdef _METAL_

void KQuantQwen4HCNorm::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  auto kernel = kq_get_kernel(device, "kq_qwen4_hc_norm");
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_input_array(inputs[1], 1);
  encoder.set_input_array(has_pending_ ? inputs[2] : inputs[0], 2);
  encoder.set_input_array(has_pending_ ? inputs[3] : inputs[0], 3);
  encoder.set_output_array(outputs[0], 4);
  encoder.set_output_array(outputs[1], 5);
  encoder.set_bytes(eps_, 6);
  int has_pending = has_pending_ ? 1 : 0;
  encoder.set_bytes(has_pending, 7);
  encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(640, 1, 1));
}

void KQuantQwen4HCFront::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  auto kernel = kq_get_kernel(device, "kq_qwen4_hc_front");
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_input_array(inputs[1], 1);
  encoder.set_input_array(has_injection_ ? inputs[3] : inputs[1], 2);
  encoder.set_output_array(outputs[0], 3);
  encoder.set_output_array(has_injection_ ? outputs[1] : outputs[0], 4);
  int has_injection = has_injection_ ? 1 : 0;
  encoder.set_bytes(has_injection, 5);
  encoder.dispatch_threadgroups(
      MTL::Size(has_injection_ ? 41 : 40, 1, 1), MTL::Size(32, 2, 1));
}

void KQuantQwen4HCEpilogue::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  auto& output = outputs[0];
  output.set_data(mx::allocator::malloc(output.nbytes()));

  auto kernel = kq_get_kernel(device, "kq_qwen4_hc_epilogue");
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_input_array(inputs[1], 1);
  encoder.set_input_array(inputs[3], 2);
  encoder.set_output_array(output, 3);
  encoder.dispatch_threadgroups(
      MTL::Size(kWidth / 2, 1, 1), MTL::Size(64, 1, 1));
}

#else

void KQuantQwen4HCNorm::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_hc_norm] requires Metal.");
}

void KQuantQwen4HCFront::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_hc_front] requires Metal.");
}

void KQuantQwen4HCEpilogue::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_hc_epilogue] requires Metal.");
}

#endif

std::vector<mx::array> qwen4_hc_norm(
    mx::array residual,
    mx::array norm_weight,
    const std::optional<mx::array>& pending_output,
    const std::optional<mx::array>& pending_injection,
    float eps,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_hc_norm]";
  auto stream = mx::to_stream(stream_or_device);
  require_bf16_one_row(residual, kExpanded, op, "residual");
  require_bf16_one_row(norm_weight, kExpanded, op, "norm_weight");
  if (!(eps > 0.0f)) {
    throw std::invalid_argument(std::string(op) + " eps must be positive.");
  }
  if (pending_output.has_value() != pending_injection.has_value()) {
    throw std::invalid_argument(
        std::string(op) +
        " pending_output and pending_injection must be supplied together.");
  }

  std::vector<mx::array> inputs;
  inputs.reserve(pending_output.has_value() ? 4 : 2);
  inputs.push_back(row_contiguous(std::move(residual), stream));
  inputs.push_back(row_contiguous(std::move(norm_weight), stream));
  if (pending_output.has_value()) {
    require_bf16_one_row(*pending_output, kWidth, op, "pending_output");
    require_bf16_one_row(
        *pending_injection, kBranches, op, "pending_injection");
    inputs.push_back(row_contiguous(*pending_output, stream));
    inputs.push_back(row_contiguous(*pending_injection, stream));
  }

  return mx::array::make_arrays(
      {inputs[0].shape(), inputs[0].shape()},
      {mx::bfloat16, mx::bfloat16},
      std::make_shared<KQuantQwen4HCNorm>(
          stream, eps, pending_output.has_value()),
      std::move(inputs));
}

std::vector<mx::array> qwen4_hc_front(
    mx::array normalized,
    mx::array down_weight,
    mx::array down_scales,
    const std::optional<mx::array>& injection_weight,
    const std::optional<mx::array>& injection_scales,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_hc_front]";
  auto stream = mx::to_stream(stream_or_device);
  require_bf16_one_row(normalized, kExpanded, op, "normalized");
  require_wire(down_weight, kLowRank, kQ6RowBytes, op, "down_weight");
  require_scales(down_scales, op, "down_scales");
  if (injection_weight.has_value() != injection_scales.has_value()) {
    throw std::invalid_argument(
        std::string(op) +
        " injection_weight and injection_scales must be supplied together.");
  }

  std::vector<mx::array> inputs;
  inputs.reserve(injection_weight.has_value() ? 5 : 3);
  inputs.push_back(row_contiguous(std::move(normalized), stream));
  inputs.push_back(row_contiguous(std::move(down_weight), stream));
  inputs.push_back(row_contiguous(std::move(down_scales), stream));
  if (injection_weight.has_value()) {
    require_wire(
        *injection_weight, kBranches, kQ6RowBytes, op, "injection_weight");
    require_scales(*injection_scales, op, "injection_scales");
    inputs.push_back(row_contiguous(*injection_weight, stream));
    inputs.push_back(row_contiguous(*injection_scales, stream));
  }

  auto lowrank_shape = replace_last(inputs[0].shape(), kLowRank);
  if (!injection_weight.has_value()) {
    return mx::array::make_arrays(
        {std::move(lowrank_shape)},
        {mx::bfloat16},
        std::make_shared<KQuantQwen4HCFront>(stream, false),
        std::move(inputs));
  }
  return mx::array::make_arrays(
      {std::move(lowrank_shape), replace_last(inputs[0].shape(), kBranches)},
      {mx::bfloat16, mx::bfloat16},
      std::make_shared<KQuantQwen4HCFront>(stream, true),
      std::move(inputs));
}

mx::array qwen4_hc_epilogue(
    mx::array lowrank,
    mx::array up_weight,
    mx::array up_scales,
    mx::array normalized,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_hc_epilogue]";
  auto stream = mx::to_stream(stream_or_device);
  require_bf16_one_row(lowrank, kLowRank, op, "lowrank");
  require_wire(up_weight, kExpanded, kQ8RowBytes, op, "up_weight");
  require_scales(up_scales, op, "up_scales");
  require_bf16_one_row(normalized, kExpanded, op, "normalized");

  auto lowrank_c = row_contiguous(std::move(lowrank), stream);
  auto normalized_c = row_contiguous(std::move(normalized), stream);
  auto output_shape = replace_last(lowrank_c.shape(), kWidth);
  return mx::array(
      std::move(output_shape),
      mx::bfloat16,
      std::make_shared<KQuantQwen4HCEpilogue>(stream),
      {std::move(lowrank_c),
       row_contiguous(std::move(up_weight), stream),
       row_contiguous(std::move(up_scales), stream),
       std::move(normalized_c)});
}

} // namespace mlx_kquant
