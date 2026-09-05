// Fixed-geometry decode boundaries for the released Qwen gated-delta graph.
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

constexpr int kConvChannels = 10240;
constexpr int kConvStateRows = 3;
constexpr int kConvKernel = 4;
constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadWidth = 128;
constexpr int kValueWidth = kValueHeads * kHeadWidth;
constexpr int kPreparedHeads = 2 * kKeyHeads + kValueHeads;

void require_bf16_shape(
    const mx::array& value,
    const mx::Shape& shape,
    const char* op,
    const char* what) {
  if (value.dtype() != mx::bfloat16 || value.shape() != shape) {
    throw std::invalid_argument(
        std::string(op) + " " + what +
        " must be bfloat16 with the released decode shape.");
  }
}

mx::array row_contiguous(mx::array value, mx::Stream stream) {
  return mx::contiguous(value, false, stream);
}

} // namespace

std::vector<mx::Shape> KQuantQwen4GDNPrepare::output_shapes(
    const std::vector<mx::array>&) {
  return {
      {1, 1, kKeyHeads, kHeadWidth},
      {1, 1, kKeyHeads, kHeadWidth},
      {1, 1, kValueHeads, kHeadWidth},
      {1, 1, kValueHeads},
      {1, 1, kValueHeads},
      {1, kConvStateRows, kConvChannels},
  };
}

bool KQuantQwen4GDNPrepare::is_equivalent(const mx::Primitive&) const {
  return true;
}

void KQuantQwen4GDNPrepare::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_gdn_prepare] requires Metal.");
}

std::vector<mx::Shape> KQuantQwen4GDNNormGate::output_shapes(
    const std::vector<mx::array>&) {
  return {{1, 1, kValueWidth}};
}

bool KQuantQwen4GDNNormGate::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQwen4GDNNormGate&>(other);
  return eps_ == o.eps_;
}

void KQuantQwen4GDNNormGate::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_gdn_norm_gate] requires Metal.");
}

#ifdef _METAL_

void KQuantQwen4GDNPrepare::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  auto kernel = kq_get_kernel(device, "kq_qwen4_gdn_prepare");
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  for (int index = 0; index < 7; ++index) {
    encoder.set_input_array(inputs[index], index);
  }
  for (int index = 0; index < 6; ++index) {
    encoder.set_output_array(outputs[index], 7 + index);
  }
  encoder.dispatch_threadgroups(
      MTL::Size(kPreparedHeads, 1, 1), MTL::Size(32, 1, 1));
}

void KQuantQwen4GDNNormGate::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  auto& output = outputs[0];
  output.set_data(mx::allocator::malloc(output.nbytes()));

  auto kernel = kq_get_kernel(device, "kq_qwen4_gdn_norm_gate");
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_input_array(inputs[1], 1);
  encoder.set_input_array(inputs[2], 2);
  encoder.set_output_array(output, 3);
  encoder.set_bytes(eps_, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(kValueHeads, 1, 1), MTL::Size(32, 1, 1));
}

#else

void KQuantQwen4GDNPrepare::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_gdn_prepare] requires Metal.");
}

void KQuantQwen4GDNNormGate::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.qwen4_gdn_norm_gate] requires Metal.");
}

#endif

std::vector<mx::array> qwen4_gdn_prepare(
    mx::array qkv,
    mx::array beta_logits,
    mx::array decay_logits,
    mx::array conv_state,
    mx::array conv_weight,
    mx::array a_log,
    mx::array dt_bias,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_gdn_prepare]";
  auto stream = mx::to_stream(stream_or_device);
  require_bf16_shape(qkv, {1, 1, kConvChannels}, op, "qkv");
  require_bf16_shape(beta_logits, {1, 1, kValueHeads}, op, "beta_logits");
  require_bf16_shape(decay_logits, {1, 1, kValueHeads}, op, "decay_logits");
  require_bf16_shape(
      conv_state, {1, kConvStateRows, kConvChannels}, op, "conv_state");
  require_bf16_shape(
      conv_weight, {kConvChannels, kConvKernel, 1}, op, "conv_weight");
  require_bf16_shape(a_log, {kValueHeads}, op, "a_log");
  require_bf16_shape(dt_bias, {kValueHeads}, op, "dt_bias");

  std::vector<mx::array> inputs;
  inputs.reserve(7);
  inputs.push_back(row_contiguous(std::move(qkv), stream));
  inputs.push_back(row_contiguous(std::move(beta_logits), stream));
  inputs.push_back(row_contiguous(std::move(decay_logits), stream));
  inputs.push_back(row_contiguous(std::move(conv_state), stream));
  inputs.push_back(row_contiguous(std::move(conv_weight), stream));
  inputs.push_back(row_contiguous(std::move(a_log), stream));
  inputs.push_back(row_contiguous(std::move(dt_bias), stream));

  return mx::array::make_arrays(
      {
          {1, 1, kKeyHeads, kHeadWidth},
          {1, 1, kKeyHeads, kHeadWidth},
          {1, 1, kValueHeads, kHeadWidth},
          {1, 1, kValueHeads},
          {1, 1, kValueHeads},
          {1, kConvStateRows, kConvChannels},
      },
      {
          mx::bfloat16,
          mx::bfloat16,
          mx::bfloat16,
          mx::bfloat16,
          mx::float32,
          mx::bfloat16,
      },
      std::make_shared<KQuantQwen4GDNPrepare>(stream),
      std::move(inputs));
}

mx::array qwen4_gdn_norm_gate(
    mx::array recurrence,
    mx::array gate,
    mx::array norm_weight,
    float eps,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_gdn_norm_gate]";
  auto stream = mx::to_stream(stream_or_device);
  require_bf16_shape(
      recurrence, {1, 1, kValueHeads, kHeadWidth}, op, "recurrence");
  require_bf16_shape(gate, {1, 1, kValueWidth}, op, "gate");
  require_bf16_shape(norm_weight, {kHeadWidth}, op, "norm_weight");
  if (!(eps > 0.0f)) {
    throw std::invalid_argument(std::string(op) + " eps must be positive.");
  }

  return mx::array(
      {1, 1, kValueWidth},
      mx::bfloat16,
      std::make_shared<KQuantQwen4GDNNormGate>(stream, eps),
      {
          row_contiguous(std::move(recurrence), stream),
          row_contiguous(std::move(gate), stream),
          row_contiguous(std::move(norm_weight), stream),
      });
}

} // namespace mlx_kquant
