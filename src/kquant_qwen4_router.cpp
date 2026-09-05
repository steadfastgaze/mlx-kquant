// Fixed-geometry released Qwen router selection.
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

constexpr int kExperts = 512;
constexpr int kRoutes = 10;

void require_logits(const mx::array& logits, const char* op) {
  if (logits.dtype() != mx::bfloat16 || logits.ndim() != 2 ||
      logits.shape(0) <= 0 || logits.shape(1) != kExperts) {
    throw std::invalid_argument(
        std::string(op) + " logits must be nonempty bfloat16 [T, 512].");
  }
}

mx::array row_contiguous(mx::array value, mx::Stream stream) {
  return mx::contiguous(value, false, stream);
}

} // namespace

std::vector<mx::Shape> KQuantQwen4RouterFusedExact::output_shapes(
    const std::vector<mx::array>& inputs) {
  const int tokens = static_cast<int>(inputs[0].shape(0));
  return {{tokens, kRoutes}, {tokens, kRoutes}};
}

void KQuantQwen4RouterFusedExact::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_router_topk_fused_exact] requires Metal.");
}

void KQuantQwen4RouterFusedExact::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
#ifdef _METAL_
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  auto kernel = kq_get_kernel(device, "kq_qwen4_router_fused_exact");
  auto& encoder = mx::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_output_array(outputs[0], 1);
  encoder.set_output_array(outputs[1], 2);
  encoder.dispatch_threadgroups(
      MTL::Size(inputs[0].shape(0), 1, 1), MTL::Size(kExperts, 1, 1));
#else
  throw std::runtime_error(
      "[mlx_kquant.qwen4_router_topk_fused_exact] requires Metal.");
#endif
}

bool KQuantQwen4RouterFusedExact::is_equivalent(const mx::Primitive&) const {
  return true;
}

std::vector<mx::array> qwen4_router_topk_fused_exact(
    mx::array logits,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_router_topk_fused_exact]";
  auto stream = mx::to_stream(stream_or_device);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument(std::string(op) + " requires a Metal stream.");
  }
  require_logits(logits, op);
  const int tokens = static_cast<int>(logits.shape(0));
  return mx::array::make_arrays(
      {{tokens, kRoutes}, {tokens, kRoutes}},
      {mx::uint32, mx::bfloat16},
      std::make_shared<KQuantQwen4RouterFusedExact>(stream),
      {row_contiguous(std::move(logits), stream)});
}

} // namespace mlx_kquant
