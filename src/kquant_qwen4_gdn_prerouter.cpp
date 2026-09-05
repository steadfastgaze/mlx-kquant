// Fixed-geometry pre-router envelope for the released Qwen decode graph.
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

constexpr int kBranches = 4;
constexpr int kHiddenWidth = 2560;
constexpr int kExpandedWidth = kBranches * kHiddenWidth;
constexpr int kLowRank = 320;
constexpr int kConvStateRows = 3;
constexpr int kConvKernel = 4;
constexpr int kKeyHeads = 16;
constexpr int kValueHeads = 48;
constexpr int kHeadWidth = 128;
constexpr int kKeyWidth = kKeyHeads * kHeadWidth;
constexpr int kValueWidth = kValueHeads * kHeadWidth;
constexpr int kPreparedHeads = 2 * kKeyHeads + kValueHeads;
constexpr int kQ6HiddenRowBytes = kHiddenWidth * 210 / 256;
constexpr int kQ6ValueRowBytes = kValueWidth * 210 / 256;
constexpr int kHCQ6RowBytes = kExpandedWidth * 210 / 256;
constexpr int kHCQ8RowBytes = kLowRank * 34 / 32;

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

void require_float32_shape(
    const mx::array& value,
    const mx::Shape& shape,
    const char* op,
    const char* what) {
  if (value.dtype() != mx::float32 || value.shape() != shape) {
    throw std::invalid_argument(
        std::string(op) + " " + what +
        " must be float32 with the released decode shape.");
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
        std::string(op) + " " + what + " must be a uint8 Q6_K wire with " +
        "the released decode shape.");
  }
}

void require_q8_wire(
    const mx::array& value,
    int rows,
    int row_bytes,
    const char* op,
    const char* what) {
  if (value.dtype() != mx::uint8 || value.ndim() != 2 ||
      value.shape(0) != rows || value.shape(1) != row_bytes) {
    throw std::invalid_argument(
        std::string(op) + " " + what + " must be a uint8 Q8_0 wire with " +
        "the released decode shape.");
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

std::vector<mx::Shape> KQuantQwen4GDNPreRouterQ6::output_shapes(
    const std::vector<mx::array>&) {
  return {
      {1, 1, kHiddenWidth},
      {1, 1, kExpandedWidth},
      {1, 1, kBranches},
      {1, kConvStateRows, kExpandedWidth},
      {1, kValueHeads, kHeadWidth, kHeadWidth},
  };
}

bool KQuantQwen4GDNPreRouterQ6::is_equivalent(
    const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantQwen4GDNPreRouterQ6&>(other);
  return eps_ == o.eps_ && has_pending_ == o.has_pending_;
}

void KQuantQwen4GDNPreRouterQ6::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_gdn_prerouter_q6] requires Metal.");
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
    mx::metal::CommandEncoder& encoder,
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    mx::array& output,
    int input_width,
    int output_width) {
  constexpr int tile_width = 8;
  const std::string kernel_name = kq_kname_prefix("q6_k") + "qmv_fast_" +
      kq_type_string(x.dtype()) + "_gs_256_b_6_batch_0";
  auto kernel = kq_get_kernel(device, kernel_name);
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

void KQuantQwen4GDNPreRouterQ6::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& stream = this->stream();
  auto& device = mx::metal::device(stream.device);
  for (auto& output : outputs) {
    output.set_data(mx::allocator::malloc(output.nbytes()));
  }

  // All intermediates are explicitly retained by the command encoder until
  // the enclosing MLX command buffer completes. Their producer and consumer
  // dispatches are recorded on this same compute encoder in dependency order.
  std::vector<mx::array> temporaries;
  temporaries.reserve(19);
  temporaries.push_back(
      make_temporary({1, 1, kExpandedWidth}, mx::bfloat16)); // 0 attention norm
  temporaries.push_back(make_temporary(
      {1, 1, kExpandedWidth}, mx::bfloat16)); // 1 attention residual
  temporaries.push_back(
      make_temporary({1, 1, kLowRank}, mx::bfloat16)); // 2 attention low rank
  temporaries.push_back(
      make_temporary({1, 1, kBranches}, mx::bfloat16)); // 3 attention injection
  temporaries.push_back(
      make_temporary({1, 1, kHiddenWidth}, mx::bfloat16)); // 4 attention mixed
  temporaries.push_back(
      make_temporary({1, 1, kExpandedWidth}, mx::bfloat16)); // 5 qkv
  temporaries.push_back(
      make_temporary({1, 1, kValueWidth}, mx::bfloat16)); // 6 gate
  temporaries.push_back(
      make_temporary({1, 1, kValueHeads}, mx::bfloat16)); // 7 beta logits
  temporaries.push_back(
      make_temporary({1, 1, kValueHeads}, mx::bfloat16)); // 8 decay logits
  temporaries.push_back(
      make_temporary({1, 1, kKeyHeads, kHeadWidth}, mx::bfloat16)); // 9 query
  temporaries.push_back(
      make_temporary({1, 1, kKeyHeads, kHeadWidth}, mx::bfloat16)); // 10 key
  temporaries.push_back(make_temporary(
      {1, 1, kValueHeads, kHeadWidth}, mx::bfloat16)); // 11 value
  temporaries.push_back(
      make_temporary({1, 1, kValueHeads}, mx::bfloat16)); // 12 beta
  temporaries.push_back(
      make_temporary({1, 1, kValueHeads}, mx::float32)); // 13 decay
  temporaries.push_back(make_temporary(
      {1, 1, kValueHeads, kHeadWidth}, mx::bfloat16)); // 14 recurrence
  temporaries.push_back(
      make_temporary({1, 1, kValueWidth}, mx::bfloat16)); // 15 GDN norm gate
  temporaries.push_back(
      make_temporary({1, 1, kHiddenWidth}, mx::bfloat16)); // 16 mixer output
  temporaries.push_back(
      make_temporary({1, 1, kExpandedWidth}, mx::bfloat16)); // 17 MLP norm
  temporaries.push_back(
      make_temporary({1, 1, kLowRank}, mx::bfloat16)); // 18 MLP low rank

  auto& attention_normalized = temporaries[0];
  auto& attention_residual = temporaries[1];
  auto& attention_lowrank = temporaries[2];
  auto& attention_injection = temporaries[3];
  auto& attention_mixed = temporaries[4];
  auto& qkv = temporaries[5];
  auto& gate = temporaries[6];
  auto& beta_logits = temporaries[7];
  auto& decay_logits = temporaries[8];
  auto& query = temporaries[9];
  auto& key = temporaries[10];
  auto& value = temporaries[11];
  auto& beta = temporaries[12];
  auto& decay = temporaries[13];
  auto& recurrence = temporaries[14];
  auto& gdn_normalized = temporaries[15];
  auto& mixer_output = temporaries[16];
  auto& mlp_normalized = temporaries[17];
  auto& mlp_lowrank = temporaries[18];

  auto& encoder = mx::metal::get_command_encoder(stream);

  // Attention gated-residual read, optionally folding the preceding layer's
  // pending block output and injection pair.
  auto hc_norm = kq_get_kernel(device, "kq_qwen4_hc_norm");
  encoder.set_compute_pipeline_state(hc_norm);
  encoder.set_input_array(inputs[0], 0);
  encoder.set_input_array(inputs[1], 1);
  encoder.set_input_array(has_pending_ ? inputs[31] : inputs[0], 2);
  encoder.set_input_array(has_pending_ ? inputs[32] : inputs[0], 3);
  encoder.set_output_array(attention_normalized, 4);
  encoder.set_output_array(attention_residual, 5);
  encoder.set_bytes(eps_, 6);
  int has_pending = has_pending_ ? 1 : 0;
  encoder.set_bytes(has_pending, 7);
  encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(640, 1, 1));

  // Attention low-rank front and four-element injection gate.
  auto hc_front = kq_get_kernel(device, "kq_qwen4_hc_front");
  encoder.set_compute_pipeline_state(hc_front);
  encoder.set_input_array(attention_normalized, 0);
  encoder.set_input_array(inputs[2], 1);
  encoder.set_input_array(inputs[4], 2);
  encoder.set_output_array(attention_lowrank, 3);
  encoder.set_output_array(attention_injection, 4);
  int has_injection = 1;
  encoder.set_bytes(has_injection, 5);
  encoder.dispatch_threadgroups(MTL::Size(41, 1, 1), MTL::Size(32, 2, 1));

  // Attention hC epilogue produces the hidden row consumed by GDN.
  auto hc_epilogue = kq_get_kernel(device, "kq_qwen4_hc_epilogue");
  encoder.set_compute_pipeline_state(hc_epilogue);
  encoder.set_input_array(attention_lowrank, 0);
  encoder.set_input_array(inputs[6], 1);
  encoder.set_input_array(attention_normalized, 2);
  encoder.set_output_array(attention_mixed, 3);
  encoder.dispatch_threadgroups(
      MTL::Size(kHiddenWidth / 2, 1, 1), MTL::Size(64, 1, 1));

  // GDN input projections. The four calls reuse the accepted q6_k qmv wire
  // and dispatch shape from the released one-token GDN boundary.
  dispatch_q6_qmv(
      device,
      encoder,
      attention_mixed,
      inputs[15],
      inputs[16],
      qkv,
      kHiddenWidth,
      kExpandedWidth);
  dispatch_q6_qmv(
      device,
      encoder,
      attention_mixed,
      inputs[17],
      inputs[18],
      gate,
      kHiddenWidth,
      kValueWidth);
  dispatch_q6_qmv(
      device,
      encoder,
      attention_mixed,
      inputs[19],
      inputs[20],
      beta_logits,
      kHiddenWidth,
      kValueHeads);
  dispatch_q6_qmv(
      device,
      encoder,
      attention_mixed,
      inputs[21],
      inputs[22],
      decay_logits,
      kHiddenWidth,
      kValueHeads);

  // GDN convolution/state preparation.
  auto prepare = kq_get_kernel(device, "kq_qwen4_gdn_prepare");
  encoder.set_compute_pipeline_state(prepare);
  encoder.set_input_array(qkv, 0);
  encoder.set_input_array(beta_logits, 1);
  encoder.set_input_array(decay_logits, 2);
  encoder.set_input_array(inputs[23], 3);
  encoder.set_input_array(inputs[24], 4);
  encoder.set_input_array(inputs[25], 5);
  encoder.set_input_array(inputs[26], 6);
  encoder.set_output_array(query, 7);
  encoder.set_output_array(key, 8);
  encoder.set_output_array(value, 9);
  encoder.set_output_array(beta, 10);
  encoder.set_output_array(decay, 11);
  encoder.set_output_array(outputs[3], 12);
  encoder.dispatch_threadgroups(
      MTL::Size(kPreparedHeads, 1, 1), MTL::Size(32, 1, 1));

  // The recurrence and the next recurrent state remain the accepted FP32
  // gated-delta operation's exact wire and dispatch.
  auto recurrence_kernel = kq_get_kernel(device, "kq_qwen4_gdn_recurrence");
  encoder.set_compute_pipeline_state(recurrence_kernel);
  encoder.set_input_array(query, 0);
  encoder.set_input_array(key, 1);
  encoder.set_input_array(value, 2);
  encoder.set_input_array(decay, 3);
  encoder.set_input_array(beta, 4);
  encoder.set_input_array(inputs[27], 5);
  encoder.set_output_array(recurrence, 6);
  encoder.set_output_array(outputs[4], 7);
  encoder.dispatch_threads(
      MTL::Size(32, kHeadWidth, kValueHeads), MTL::Size(32, 4, 1));

  // GDN norm/gate and output projection.
  auto norm_gate = kq_get_kernel(device, "kq_qwen4_gdn_norm_gate");
  encoder.set_compute_pipeline_state(norm_gate);
  encoder.set_input_array(recurrence, 0);
  encoder.set_input_array(gate, 1);
  encoder.set_input_array(inputs[28], 2);
  encoder.set_output_array(gdn_normalized, 3);
  encoder.set_bytes(eps_, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(kValueHeads, 1, 1), MTL::Size(32, 1, 1));
  dispatch_q6_qmv(
      device,
      encoder,
      gdn_normalized,
      inputs[29],
      inputs[30],
      mixer_output,
      kValueWidth,
      kHiddenWidth);

  // The attention residual write is folded into the pending MLP hC norm. The
  // HC norm kernel performs the BF16 product, BF16 addition, and RMS lattice
  // in the same order as gated_residual_write followed by mlp_residual.
  encoder.set_compute_pipeline_state(hc_norm);
  encoder.set_input_array(attention_residual, 0);
  encoder.set_input_array(inputs[8], 1);
  encoder.set_input_array(mixer_output, 2);
  encoder.set_input_array(attention_injection, 3);
  encoder.set_output_array(mlp_normalized, 4);
  encoder.set_output_array(outputs[1], 5);
  encoder.set_bytes(eps_, 6);
  has_pending = 1;
  encoder.set_bytes(has_pending, 7);
  encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(640, 1, 1));

  // MLP gated-residual front and epilogue stay outside the router boundary;
  // their hidden row and injection are returned for the caller's router/MLP.
  encoder.set_compute_pipeline_state(hc_front);
  encoder.set_input_array(mlp_normalized, 0);
  encoder.set_input_array(inputs[9], 1);
  encoder.set_input_array(inputs[11], 2);
  encoder.set_output_array(mlp_lowrank, 3);
  encoder.set_output_array(outputs[2], 4);
  has_injection = 1;
  encoder.set_bytes(has_injection, 5);
  encoder.dispatch_threadgroups(MTL::Size(41, 1, 1), MTL::Size(32, 2, 1));

  encoder.set_compute_pipeline_state(hc_epilogue);
  encoder.set_input_array(mlp_lowrank, 0);
  encoder.set_input_array(inputs[13], 1);
  encoder.set_input_array(mlp_normalized, 2);
  encoder.set_output_array(outputs[0], 3);
  encoder.dispatch_threadgroups(
      MTL::Size(kHiddenWidth / 2, 1, 1), MTL::Size(64, 1, 1));

  encoder.add_temporaries(std::move(temporaries));
}

#else

void KQuantQwen4GDNPreRouterQ6::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.qwen4_gdn_prerouter_q6] requires Metal.");
}

#endif

std::vector<mx::array> qwen4_gdn_prerouter_q6(
    mx::array hidden,
    mx::array attention_norm_weight,
    mx::array attention_down_weight,
    mx::array attention_down_scales,
    mx::array attention_injection_weight,
    mx::array attention_injection_scales,
    mx::array attention_up_weight,
    mx::array attention_up_scales,
    mx::array mlp_norm_weight,
    mx::array mlp_down_weight,
    mx::array mlp_down_scales,
    mx::array mlp_injection_weight,
    mx::array mlp_injection_scales,
    mx::array mlp_up_weight,
    mx::array mlp_up_scales,
    mx::array qkv_weight,
    mx::array qkv_scales,
    mx::array gate_weight,
    mx::array gate_scales,
    mx::array beta_weight,
    mx::array beta_scales,
    mx::array decay_weight,
    mx::array decay_scales,
    mx::array conv_state,
    mx::array conv_weight,
    mx::array a_log,
    mx::array dt_bias,
    mx::array recurrent_state,
    mx::array gdn_norm_weight,
    mx::array gdn_out_weight,
    mx::array gdn_out_scales,
    const std::optional<mx::array>& pending_output,
    const std::optional<mx::array>& pending_injection,
    float eps,
    mx::StreamOrDevice stream_or_device) {
  constexpr const char* op = "[mlx_kquant.qwen4_gdn_prerouter_q6]";
  auto stream = mx::to_stream(stream_or_device);

  require_bf16_shape(hidden, {1, 1, kExpandedWidth}, op, "hidden");
  require_bf16_shape(
      attention_norm_weight, {kExpandedWidth}, op, "attention_norm_weight");
  require_q6_wire(
      attention_down_weight,
      kLowRank,
      kHCQ6RowBytes,
      op,
      "attention_down_weight");
  require_scales(attention_down_scales, op, "attention_down_scales");
  require_q6_wire(
      attention_injection_weight,
      kBranches,
      kHCQ6RowBytes,
      op,
      "attention_injection_weight");
  require_scales(attention_injection_scales, op, "attention_injection_scales");
  require_q8_wire(
      attention_up_weight,
      kExpandedWidth,
      kHCQ8RowBytes,
      op,
      "attention_up_weight");
  require_scales(attention_up_scales, op, "attention_up_scales");

  require_bf16_shape(mlp_norm_weight, {kExpandedWidth}, op, "mlp_norm_weight");
  require_q6_wire(
      mlp_down_weight, kLowRank, kHCQ6RowBytes, op, "mlp_down_weight");
  require_scales(mlp_down_scales, op, "mlp_down_scales");
  require_q6_wire(
      mlp_injection_weight,
      kBranches,
      kHCQ6RowBytes,
      op,
      "mlp_injection_weight");
  require_scales(mlp_injection_scales, op, "mlp_injection_scales");
  require_q8_wire(
      mlp_up_weight, kExpandedWidth, kHCQ8RowBytes, op, "mlp_up_weight");
  require_scales(mlp_up_scales, op, "mlp_up_scales");

  require_q6_wire(
      qkv_weight, kExpandedWidth, kQ6HiddenRowBytes, op, "qkv_weight");
  require_scales(qkv_scales, op, "qkv_scales");
  require_q6_wire(
      gate_weight, kValueWidth, kQ6HiddenRowBytes, op, "gate_weight");
  require_scales(gate_scales, op, "gate_scales");
  require_q6_wire(
      beta_weight, kValueHeads, kQ6HiddenRowBytes, op, "beta_weight");
  require_scales(beta_scales, op, "beta_scales");
  require_q6_wire(
      decay_weight, kValueHeads, kQ6HiddenRowBytes, op, "decay_weight");
  require_scales(decay_scales, op, "decay_scales");
  require_bf16_shape(
      conv_state, {1, kConvStateRows, kExpandedWidth}, op, "conv_state");
  require_bf16_shape(
      conv_weight, {kExpandedWidth, kConvKernel, 1}, op, "conv_weight");
  require_bf16_shape(a_log, {kValueHeads}, op, "a_log");
  require_bf16_shape(dt_bias, {kValueHeads}, op, "dt_bias");
  require_float32_shape(
      recurrent_state,
      {1, kValueHeads, kHeadWidth, kHeadWidth},
      op,
      "recurrent_state");
  require_bf16_shape(gdn_norm_weight, {kHeadWidth}, op, "gdn_norm_weight");
  require_q6_wire(
      gdn_out_weight, kHiddenWidth, kQ6ValueRowBytes, op, "gdn_out_weight");
  require_scales(gdn_out_scales, op, "gdn_out_scales");
  if (pending_output.has_value() != pending_injection.has_value()) {
    throw std::invalid_argument(
        std::string(op) +
        " pending_output and pending_injection must be supplied together.");
  }
  if (pending_output.has_value()) {
    require_bf16_shape(
        *pending_output, {1, 1, kHiddenWidth}, op, "pending_output");
    require_bf16_shape(
        *pending_injection, {1, 1, kBranches}, op, "pending_injection");
  }
  if (!(eps > 0.0f)) {
    throw std::invalid_argument(std::string(op) + " eps must be positive.");
  }

  std::vector<mx::array> inputs;
  inputs.reserve(pending_output.has_value() ? 33 : 31);
  for (auto& input : std::vector<mx::array>{
           std::move(hidden),
           std::move(attention_norm_weight),
           std::move(attention_down_weight),
           std::move(attention_down_scales),
           std::move(attention_injection_weight),
           std::move(attention_injection_scales),
           std::move(attention_up_weight),
           std::move(attention_up_scales),
           std::move(mlp_norm_weight),
           std::move(mlp_down_weight),
           std::move(mlp_down_scales),
           std::move(mlp_injection_weight),
           std::move(mlp_injection_scales),
           std::move(mlp_up_weight),
           std::move(mlp_up_scales),
           std::move(qkv_weight),
           std::move(qkv_scales),
           std::move(gate_weight),
           std::move(gate_scales),
           std::move(beta_weight),
           std::move(beta_scales),
           std::move(decay_weight),
           std::move(decay_scales),
           std::move(conv_state),
           std::move(conv_weight),
           std::move(a_log),
           std::move(dt_bias),
           std::move(recurrent_state),
           std::move(gdn_norm_weight),
           std::move(gdn_out_weight),
           std::move(gdn_out_scales),
       }) {
    inputs.push_back(row_contiguous(std::move(input), stream));
  }
  if (pending_output.has_value()) {
    inputs.push_back(row_contiguous(*pending_output, stream));
    inputs.push_back(row_contiguous(*pending_injection, stream));
  }

  return mx::array::make_arrays(
      {
          {1, 1, kHiddenWidth},
          {1, 1, kExpandedWidth},
          {1, 1, kBranches},
          {1, kConvStateRows, kExpandedWidth},
          {1, kValueHeads, kHeadWidth, kHeadWidth},
      },
      {mx::bfloat16, mx::bfloat16, mx::bfloat16, mx::bfloat16, mx::float32},
      std::make_shared<KQuantQwen4GDNPreRouterQ6>(
          stream, eps, pending_output.has_value()),
      std::move(inputs));
}

} // namespace mlx_kquant
