// KQuantGatherQMVPairSwiglu / KQuantGatherQMVExpertSum primitives: the
// decode-shaped routed-MoE matvec pair (fused gate/up + SwiGLU with baked
// route weights; down matvec with the in-kernel sum over the token's routed
// experts). One dispatch each. The GPU paths fetch the per-codec kernels from
// the bundled metallib; only the codecs instantiated in kq_decode_moe.metal
// are reachable (the ops allow-list them). Inference-only (no CPU eval).
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string, kq_kname_prefix

#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

std::vector<mx::Shape> KQuantGatherQMVPairSwiglu::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& ids = inputs[3];
  return {mx::Shape{ids.shape(0), gate_out_}};
}

bool KQuantGatherQMVPairSwiglu::is_equivalent(
    const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVPairSwiglu&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_ && gate_out_ == o.gate_out_ &&
      swiglu_limit_ == o.swiglu_limit_;
}

#ifdef _METAL_

void KQuantGatherQMVPairSwiglu::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs: x (float16/bfloat16/float32, row-contiguous [1, K]), w (uint8
  // [E, 2 * gate_out, bytes_per_row], gate rows then up rows per expert),
  // scales (vestigial placeholder), ids (uint32 [B]), route_weights
  // (float32 [B]). Grid: one threadgroup per (4-row block, expert slot).
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& ids = inputs[3];
  const auto& route_weights = inputs[4];

  int K = x.shape(-1);
  int B = ids.shape(0);
  // 2 simdgroups x 2 results: must match the kernel's num_simdgroups *
  // results_per_simdgroup row block.
  int bn = 4;
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(1, (gate_out_ + bn - 1) / bn, B);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname = kq_kname_prefix(kquant_type_) +
      "gather_qmv_pair_swiglu_" + type_string + "_gs_" +
      std::to_string(group_size_) + "_b_" + std::to_string(bits_);

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(inputs[2], c++); // scales placeholder
  ce.set_input_array(x, c++);
  ce.set_input_array(ids, c++);
  ce.set_input_array(route_weights, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(gate_out_, c++);
  ce.set_bytes(swiglu_limit_, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantGatherQMVPairSwiglu::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_pair_swiglu] requires a Metal build.");
}

#endif

void KQuantGatherQMVPairSwiglu::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_pair_swiglu] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantGatherQMVExpertSum::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& w = inputs[1];
  return {mx::Shape{1, w.shape(-2)}};
}

bool KQuantGatherQMVExpertSum::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMVExpertSum&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

#ifdef _METAL_

void KQuantGatherQMVExpertSum::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs: x (float16/bfloat16/float32, row-contiguous [B, K]), w (uint8
  // [E, N, bytes_per_row]), scales (vestigial placeholder), ids (uint32
  // [B]). Grid: one threadgroup per 4-row output block; the kernel loops
  // the B experts internally.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& ids = inputs[3];

  int K = x.shape(-1);
  int B = ids.shape(0);
  int N = w.shape(-2);
  int bn = kquant_qmv_bn(kquant_type_);
  MTL::Size group_dims(32, 2, 1);
  MTL::Size grid_dims(1, (N + bn - 1) / bn, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname = kq_kname_prefix(kquant_type_) + "gather_qmv_expert_sum_" +
      type_string + "_gs_" + std::to_string(group_size_) + "_b_" +
      std::to_string(bits_);

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(inputs[2], c++); // scales placeholder
  ce.set_input_array(x, c++);
  ce.set_input_array(ids, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(B, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantGatherQMVExpertSum::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_expert_sum] requires a Metal build.");
}

#endif

void KQuantGatherQMVExpertSum::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmv_expert_sum] has no CPU implementation.");
}

} // namespace mlx_kquant
