#include <dlfcn.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "kquant.h"
#include "kquant_codec.h"

#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

namespace {

// Matrix-contiguity helper. A "matrix-contiguous" tensor only needs
// its LAST TWO dims densely packed (stride(-2)==shape(-1) && stride(-1)==1); a
// strided LEADING/batch dim is fine because the qmv/qmm/gather kernels walk it
// via the strides passed in add_strides_and_shapes. Using full
// mx::contiguous() here instead would force a copy of the whole tensor whenever
// the leading dim is strided - e.g. the gate/up halves of a fused MoE
// gate_up_exps tensor are slices [..., :H, :] / [..., H:, :] whose expert
// (leading) stride is 2*H rows: matrix-contiguous but NOT row_contiguous, so
// full contiguify copies ~142 MB PER gather call (the decode MoE 5x
// regression).
//
// Limitation: this check runs at graph construction. An unevaluated array
// still carries the constructor's dense placeholder strides and default-true
// contiguity flags (the real layout exists only after evaluation), so a lazy
// strided view passes the check unconverted. This helper is therefore a
// cheap fast path, not a guarantee; a primitive whose kernels require dense
// rows must re-check the evaluated metadata in eval (KQuantMatmul does).
inline mx::array kq_ensure_row_contiguous_matrix(
    const mx::array& x,
    mx::StreamOrDevice s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == static_cast<int64_t>(x.shape(-1)) && stride_1 == 1) {
      return x;
    }
  }
  return mx::contiguous(x, false, s);
}

// Activation-side density guard for quantized_matmul. Strides and contiguity
// flags are real only once an array's eval has run (status != unscheduled);
// an unscheduled array still carries the constructor's dense placeholder
// metadata, so a lazy strided view sails through the matrix-contiguity check
// above unconverted, and the matmul kernels - which advance the M rows
// densely by K after the batch offset - then read the wrong elements on
// every multi-row call. That is the DS4 grouped-projection defect: a grouped
// activation's second-to-last-axis slice at M >= 2 measured rel error
// ~0.5-1.8 against the contiguized reference on every codec and both weight
// forms; float32 activations were laundered dense by the op's astype, and
// single-row calls have no row stride to misread, which is why decode text
// stayed coherent while every multi-row scorer read garbage.
//
// For an unscheduled multi-row activation, insert mx::contiguous: its eval
// is a zero-cost shared-buffer pass-through when the materialized layout is
// already row-contiguous and a real copy otherwise. Single-row activations
// pass through untouched - the row stride is never read at M = 1, and the
// inserted node would land on the decode hot path of every consumer.
inline mx::array kq_ensure_dense_rows_lazy_safe(
    const mx::array& x,
    mx::StreamOrDevice s) {
  if (x.status() != mx::array::Status::unscheduled) {
    return kq_ensure_row_contiguous_matrix(x, s);
  }
  int64_t rows = 1;
  for (int i = 0; i + 1 < static_cast<int>(x.ndim()); ++i) {
    rows *= x.shape(i);
  }
  if (rows <= 1) {
    return x;
  }
  return mx::contiguous(x, false, s);
}

} // namespace

// ----------------------------- ops -----------------------------

mx::array dequantize(
    const mx::array& w,
    const mx::array& scales,
    const std::string& kquant_type,
    std::optional<mx::Dtype> dtype,
    mx::StreamOrDevice s_) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] w must be uint8 (raw GGUF wire bytes).");
  }
  if (w.ndim() < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] w must have at least 1 dimension.");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (w.shape(-1) % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.dequantize] Last dim (" << w.shape(-1)
        << ") must be a multiple of bytes_per_block (" << codec->bytes_per_block
        << ") for codec '" << kquant_type << "'.";
    throw std::invalid_argument(msg.str());
  }

  mx::Dtype out_type = dtype.value_or(mx::float16);
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.dequantize] Only real floating output dtypes are "
        "supported.");
  }

  auto s = mx::to_stream(s_);

  auto out_shape = w.shape();
  out_shape.back() =
      (w.shape(-1) / codec->bytes_per_block) * codec->weights_per_block;

  // Row-contiguize at the op level so eval_gpu can assume dense inputs.
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);

  // For every kquant codec, group_size == weights_per_block (32 or 256).
  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantDequantize>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {w_c, scales});
}

mx::array quantized_matmul(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    bool transpose,
    mx::StreamOrDevice s_) {
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] w must be uint8 (raw GGUF wire bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] Unknown kquant_type: '" + kquant_type +
        "'.");
  }
  if (x.ndim() > 2 && w.ndim() > 2) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] batched (>2D) weights are not supported; "
        "use gather_qmm for mixture-of-experts.");
  }

  // Expand w's quantized geometry from the codec block layout.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int w_inner_dims = transpose ? weights_per_row : w.shape(-2);
  int w_outer_dims = transpose ? w.shape(-2) : weights_per_row;
  if (w_inner_dims != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized matrix inner dim ("
        << w_inner_dims << ") for codec '" << codec->name
        << "' with transpose=" << std::boolalpha << transpose << ".";
    throw std::invalid_argument(msg.str());
  }

  // Output dtype: x.dtype(), with float32 promoted to bfloat16 (the NAX matmul
  // path emits bf16 accumulation).
  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul] Only real floating x dtypes are "
        "supported.");
  }

  auto s = mx::to_stream(s_);

  // Cast x to the output dtype, then contiguize at the op level. The
  // activation takes the lazy-safe form (a lazy strided view carries dense
  // placeholder metadata that defeats the plain check); w and scales keep
  // the metadata check - the wire is evaluated in every real flow, and a
  // fresh row-slice of it materializes as a dense-rows offset view that the
  // kernels address correctly. KQuantMatmul::eval fails closed on anything
  // that still arrives without densely packed rows.
  auto x_c = kq_ensure_dense_rows_lazy_safe(mx::astype(x, out_type, s), s);
  auto w_c = kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);

  auto out_shape = x_c.shape();
  out_shape.back() = w_outer_dims;

  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantMatmul>(
          s, kquant_type, codec->weights_per_block, codec->bits, transpose),
      {x_c, w_c, scales_c});
}

mx::array quantized_matmul_qmv_bias(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array bias,
    const std::string& kquant_type,
    mx::StreamOrDevice s_) {
  if (kquant_type != "q8_0") {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] only kquant_type 'q8_0' is "
        "wired so far, got '" +
        kquant_type + "'.");
  }
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] w must be uint8 (raw GGUF "
        "wire bytes).");
  }
  if (w.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] w must be 2D (this op is not "
        "batched); use gather_qmm for mixture-of-experts.");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);

  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int N = w.shape(-2); // transpose=true only: w is [N, K]
  if (weights_per_row != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized matrix "
        << "inner dim (" << weights_per_row << ") for codec '" << codec->name
        << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (bias.ndim() != 1 || bias.shape(0) != N) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] bias must be 1D with "
        << "length " << N << " (the output dim), got shape " << bias.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  int64_t total_rows = x.size() / x.shape(-1);
  if (total_rows != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_bias] decode-only: x must carry "
        << "exactly one row (got " << total_rows << "). Use "
        << "quantized_matmul() + a separate add for prefill/verify shapes.";
    throw std::invalid_argument(msg.str());
  }

  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_bias] Only real floating x dtypes "
        "are supported.");
  }

  auto s = mx::to_stream(s_);

  auto x_c = kq_ensure_row_contiguous_matrix(mx::astype(x, out_type, s), s);
  auto w_c = kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);
  auto bias_c = mx::astype(bias, out_type, s);
  bias_c =
      bias_c.flags().row_contiguous ? bias_c : mx::contiguous(bias_c, false, s);

  auto out_shape = x_c.shape();
  out_shape.back() = N;

  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantQmvBias>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {x_c, w_c, scales_c, bias_c});
}

mx::array quantized_matmul_qmv_hc_post(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array residual,
    mx::array post,
    mx::array comb,
    const std::string& kquant_type,
    mx::StreamOrDevice s_) {
  constexpr int hc = 4;
  constexpr int qmv_n_align = 8;
  constexpr int qmv_k_align = 256;

  if (kquant_type != "q8_0") {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_hc_post] only kquant_type 'q8_0' "
        "is supported, got '" +
        kquant_type + "'.");
  }
  if ((x.dtype() != mx::float32 && x.dtype() != mx::bfloat16) || x.ndim() < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_hc_post] x must be float32 or "
        "bfloat16 with at least one dimension.");
  }
  if (w.dtype() != mx::uint8 || w.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_hc_post] w must be a 2D uint8 "
        "Q8_0 wire matrix.");
  }

  const KQuantCodec* codec = codec_by_name(kquant_type);
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row == 0 || w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_hc_post] w last dim ("
        << w_bytes_per_row << ") must be a positive multiple of "
        << codec->bytes_per_block << " Q8_0 bytes.";
    throw std::invalid_argument(msg.str());
  }
  int K = (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int N = w.shape(-2);
  if (N <= 0 || N % qmv_n_align != 0 || K % qmv_k_align != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_hc_post] fast-QMV alignment "
        << "requires positive N divisible by " << qmv_n_align
        << " and K divisible by " << qmv_k_align << "; got N=" << N
        << ", K=" << K << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.shape(-1) != K || static_cast<int64_t>(x.size()) / x.shape(-1) != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_hc_post] decode-only x must "
        << "contain exactly one row of length " << K << "; got shape "
        << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (residual.dtype() != mx::float32 || residual.shape() != mx::Shape{hc, N}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_hc_post] residual must be "
        << "float32 [4, " << N << "], got dtype " << residual.dtype()
        << " and shape " << residual.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (post.dtype() != mx::float32 || post.shape() != mx::Shape{hc}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_hc_post] post must be float32 "
        << "[4], got dtype " << post.dtype() << " and shape " << post.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (comb.dtype() != mx::float32 || comb.shape() != mx::Shape{hc, hc}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_hc_post] comb must be float32 "
        << "[4, 4], got dtype " << comb.dtype() << " and shape " << comb.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = mx::to_stream(s_);
  auto x_c = kq_ensure_row_contiguous_matrix(x, s);
  auto w_c = kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c =
      scales.flags().row_contiguous ? scales : mx::contiguous(scales, false, s);
  auto residual_c = kq_ensure_row_contiguous_matrix(residual, s);
  auto post_c =
      post.flags().row_contiguous ? post : mx::contiguous(post, false, s);
  auto comb_c = kq_ensure_row_contiguous_matrix(comb, s);

  return mx::array(
      {hc, N},
      mx::float32,
      std::make_shared<KQuantQmvHCPost>(s),
      {x_c, w_c, scales_c, residual_c, post_c, comb_c});
}

mx::array quantized_matmul_qmv_add_hc_post(
    mx::array x,
    mx::array w,
    mx::array scales,
    mx::array routed,
    mx::array residual,
    mx::array post,
    mx::array comb,
    const std::string& kquant_type,
    mx::StreamOrDevice s_) {
  constexpr int hc = 4;
  constexpr int K = 2048;
  constexpr int N = 4096;
  constexpr int w_bytes_per_row = 2176;

  if (kquant_type != "q8_0") {
    throw std::invalid_argument(
        "[mlx_kquant.quantized_matmul_qmv_add_hc_post] only kquant_type "
        "'q8_0' is supported, got '" +
        kquant_type + "'.");
  }
  if (x.dtype() != mx::bfloat16 || x.ndim() < 1 || x.shape(-1) != K ||
      static_cast<int64_t>(x.size()) / K != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_add_hc_post] x must be "
        << "bfloat16 with exactly one row of length " << K << "; got dtype "
        << x.dtype() << " and shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.dtype() != mx::uint8 || w.shape() != mx::Shape{N, w_bytes_per_row}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_add_hc_post] w must be uint8 "
        << "[" << N << ", " << w_bytes_per_row
        << "] Q8_0 wire bytes; got dtype " << w.dtype() << " and shape "
        << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (routed.dtype() != mx::float16 || routed.shape() != mx::Shape{N}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_add_hc_post] routed must be "
        << "float16 [" << N << "], got dtype " << routed.dtype()
        << " and shape " << routed.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (residual.dtype() != mx::float32 || residual.shape() != mx::Shape{hc, N}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_add_hc_post] residual must be "
        << "float32 [4, " << N << "], got dtype " << residual.dtype()
        << " and shape " << residual.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (post.dtype() != mx::float32 || post.shape() != mx::Shape{hc}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_add_hc_post] post must be "
        << "float32 [4], got dtype " << post.dtype() << " and shape "
        << post.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (comb.dtype() != mx::float32 || comb.shape() != mx::Shape{hc, hc}) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantized_matmul_qmv_add_hc_post] comb must be "
        << "float32 [4, 4], got dtype " << comb.dtype() << " and shape "
        << comb.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = mx::to_stream(s_);
  auto x_c = kq_ensure_row_contiguous_matrix(x, s);
  auto w_c = kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c =
      scales.flags().row_contiguous ? scales : mx::contiguous(scales, false, s);
  auto routed_c =
      routed.flags().row_contiguous ? routed : mx::contiguous(routed, false, s);
  auto residual_c = kq_ensure_row_contiguous_matrix(residual, s);
  auto post_c =
      post.flags().row_contiguous ? post : mx::contiguous(post, false, s);
  auto comb_c = kq_ensure_row_contiguous_matrix(comb, s);

  return mx::array(
      {hc, N},
      mx::float32,
      std::make_shared<KQuantQmvAddHCPost>(s),
      {x_c, w_c, scales_c, routed_c, residual_c, post_c, comb_c});
}

namespace {

// When indices are omitted, default to a flat arange over the leading (batch)
// dims.
mx::array kq_indices_or_default(
    const std::optional<mx::array>& indices,
    const mx::array& x,
    mx::StreamOrDevice s) {
  if (indices.has_value()) {
    return indices.value();
  }
  mx::Shape shape(x.shape().begin(), x.shape().end() - 2);
  int total = 1;
  for (auto dim : shape) {
    total *= dim;
  }
  return mx::reshape(mx::arange(total, mx::uint32, s), std::move(shape), s);
}

} // namespace

mx::array gather_qmm(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    std::optional<mx::array> lhs_indices_,
    std::optional<mx::array> rhs_indices_,
    bool transpose,
    bool sorted_indices,
    mx::StreamOrDevice s_) {
  // No indices at all -> a plain (non-gathered) quantized matmul.
  if (!lhs_indices_ && !rhs_indices_) {
    return quantized_matmul(
        std::move(x),
        std::move(w),
        std::move(scales),
        kquant_type,
        transpose,
        s_);
  }

  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm] w must be uint8 (raw GGUF wire bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm] Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (x.ndim() < 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm] x must have at least two dimensions but got "
        << "shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout. w is
  // [..., out_dims, bytes_per_row]; the last dim is whole K-quant blocks.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm] KQuant weight last dim (" << w_bytes_per_row
        << " bytes) is not a whole number of " << codec->bytes_per_block
        << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int weights_per_row =
      (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  int w_inner_dims = transpose ? weights_per_row : w.shape(-2);
  int w_outer_dims = transpose ? w.shape(-2) : weights_per_row;
  if (w_inner_dims != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized matrix inner dim ("
        << w_inner_dims << ") for codec '" << codec->name
        << "' with transpose=" << std::boolalpha << transpose << ".";
    throw std::invalid_argument(msg.str());
  }

  // Output dtype: x.dtype(), with float32 promoted to bfloat16.
  mx::Dtype out_type = (x.dtype() == mx::float32) ? mx::bfloat16 : x.dtype();
  if (!mx::issubdtype(out_type, mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm] Only real floating x dtypes are supported.");
  }

  auto s = mx::to_stream(s_);

  // Default + broadcast the indices, then cast to uint32.
  mx::array lhs_indices = kq_indices_or_default(lhs_indices_, x, s);
  mx::array rhs_indices = kq_indices_or_default(rhs_indices_, w, s);
  auto bc = mx::broadcast_arrays({lhs_indices, rhs_indices}, s);
  lhs_indices = mx::astype(bc[0], mx::uint32, s);
  rhs_indices = mx::astype(bc[1], mx::uint32, s);

  // Full output shape: lhs_indices batch dims, then [M, w_outer_dims].
  auto out_shape = lhs_indices.shape();
  out_shape.push_back(x.shape(-2));
  out_shape.push_back(w_outer_dims);

  // left_sorted_ / right_sorted_: a sort guarantee holds for the side whose
  // index was *defaulted* (broadcast from the other), so each "sorted" flag is
  // gated on the OPPOSITE index being
  // absent. left_sorted tracks the lhs (x) ordering -> gated on rhs absent;
  // right_sorted tracks the rhs (w/expert) ordering -> gated on lhs absent.
  // The MoE call passes rhs_indices only, so right_sorted_ == sorted_indices,
  // which arms the gather_qmm_rhs fast path in eval_gpu.
  bool left_sorted = sorted_indices && !rhs_indices_;
  bool right_sorted = sorted_indices && !lhs_indices_;

  // Cast x to the output dtype; row-contiguize x / w / scales at the op level.
  //
  // w needs FULL row-contiguity ONLY when the rhs_nax prefill leaf is reachable
  // (right_sorted && transpose && codec has a fused matmul kernel): that kernel
  // bakes dense expert packing into func consts and takes NO w strides, so the
  // rhs_nax leaf needs a fully row-contiguous w. Every
  // other leaf (qmv decode / qmm) walks w's strides, so matrix-contiguity (a
  // strided LEADING/expert dim, no copy) suffices and is what keeps the fused
  // gate_up_exps slice from being copied wholesale on every DECODE gather - the
  // 5x-sensitive hot path. x / scales always take the cheaper matrix check: the
  // eval_gpu rhs_nax gate already requires x.row_contiguous (skips rhs_nax
  // otherwise, routing to a strided-safe leaf), and scales is a (1,)
  // placeholder.
  bool rhs_nax_reachable =
      right_sorted && transpose && codec->has_matmul_kernel;
  auto x_c = kq_ensure_row_contiguous_matrix(mx::astype(x, out_type, s), s);
  auto w_c = rhs_nax_reachable
      ? (w.flags().row_contiguous ? w : mx::contiguous(w, false, s))
      : kq_ensure_row_contiguous_matrix(w, s);
  auto scales_c = kq_ensure_row_contiguous_matrix(scales, s);

  return mx::array(
      std::move(out_shape),
      out_type,
      std::make_shared<KQuantGatherQMM>(
          s,
          kquant_type,
          codec->weights_per_block,
          codec->bits,
          transpose,
          left_sorted,
          right_sorted),
      {x_c, w_c, scales_c, std::move(lhs_indices), std::move(rhs_indices)});
}

mx::array gather_qmm_segments(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array segments,
    bool transpose,
    mx::StreamOrDevice s_) {
  if (!transpose) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_segments] only transpose=True is supported "
        "(the MoE weight layout [n_experts, N, bytes_per_row]).");
  }
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_segments] w must be uint8 (raw GGUF wire "
        "bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_segments] Unknown kquant_type: '" +
        kquant_type + "'.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_segments] x must be float16, bfloat16, or "
        "float32.");
  }
  if (x.ndim() != 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_segments] x must be 2-D [S, K] but got shape "
        << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_segments] w must be 3-D "
        << "[n_experts, N, bytes_per_row] but got shape " << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (segments.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_segments] segments must be uint32.");
  }
  if (segments.ndim() != 2 || segments.shape(-1) != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_segments] segments must be [T, 3] but got "
        << "shape " << segments.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout. w is
  // [n_experts, N, bytes_per_row]; the last dim is whole K-quant blocks and
  // expands to K input features.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_segments] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int K = (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  if (K != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_segments] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized weight inner dim (" << K
        << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (K % codec->weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_segments] K (" << K
        << ") must be a multiple of the codec block ("
        << codec->weights_per_block << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }

  int N = w.shape(-2);

  auto s = mx::to_stream(s_);

  // Row-contiguize x / w / segments at the op level so eval_gpu can assume
  // dense inputs (the kernel indexes x[row*K], w[(e*N+n)*row_bytes], and reads
  // descriptors as a flat [T*3] uint32 buffer).
  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  auto seg_c = segments.flags().row_contiguous
      ? segments
      : mx::contiguous(segments, false, s);

  mx::Shape out_shape = {x.shape(0), N};

  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMMSegments>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {std::move(x_c), std::move(w_c), std::move(scales), std::move(seg_c)});
}

mx::array gather_qmm_sorted(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array sorted_ids,
    bool transpose,
    mx::StreamOrDevice s_) {
  if (!transpose) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted] only transpose=True is supported "
        "(the MoE weight layout [n_experts, N, bytes_per_row]).");
  }
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted] w must be uint8 (raw GGUF wire "
        "bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted] Unknown kquant_type: '" + kquant_type +
        "'.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted] x must be float16, bfloat16, or "
        "float32.");
  }
  if (x.ndim() != 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted] x must be 2-D [S, K] but got shape "
        << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted] w must be 3-D "
        << "[n_experts, N, bytes_per_row] but got shape " << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (sorted_ids.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted] sorted_ids must be uint32.");
  }
  if (sorted_ids.ndim() != 1 || sorted_ids.shape(0) != x.shape(0)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted] sorted_ids must be 1-D with one id "
        << "per x row (" << x.shape(0) << ") but got shape "
        << sorted_ids.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout, exactly as
  // gather_qmm_segments does.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int K = (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  if (K != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized weight inner dim (" << K
        << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }

  int N = w.shape(-2);

  auto s = mx::to_stream(s_);

  // Row-contiguize x / w / sorted_ids at the op level so eval_gpu can assume
  // dense inputs (the kernel indexes x[row*K], w[(e*N+n)*row_bytes], and
  // binary-searches sorted_ids as a flat [S] uint32 buffer).
  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  auto ids_c = sorted_ids.flags().row_contiguous
      ? sorted_ids
      : mx::contiguous(sorted_ids, false, s);

  mx::Shape out_shape = {x.shape(0), N};

  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMMSorted>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {std::move(x_c), std::move(w_c), std::move(scales), std::move(ids_c)});
}

mx::array gather_qmm_sorted_swiglu(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array sorted_ids,
    int gate_out,
    float swiglu_limit,
    bool transpose,
    mx::StreamOrDevice s_) {
  if (!transpose) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted_swiglu] only transpose=True is "
        "supported (the MoE weight layout [n_experts, N, bytes_per_row]).");
  }
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted_swiglu] w must be uint8 (raw GGUF "
        "wire bytes).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted_swiglu] Unknown kquant_type: '" +
        kquant_type + "'.");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted_swiglu] x must be float16, bfloat16, "
        "or float32.");
  }
  if (x.ndim() != 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted_swiglu] x must be 2-D [S, K] but "
        << "got shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted_swiglu] w must be 3-D "
        << "[n_experts, 2 * gate_out, bytes_per_row] but got shape "
        << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (sorted_ids.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmm_sorted_swiglu] sorted_ids must be uint32.");
  }
  if (sorted_ids.ndim() != 1 || sorted_ids.shape(0) != x.shape(0)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted_swiglu] sorted_ids must be 1-D "
        << "with one id per x row (" << x.shape(0) << ") but got shape "
        << sorted_ids.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (gate_out <= 0 || w.shape(-2) != 2 * gate_out) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted_swiglu] w must stack gate rows "
        << "then up rows per expert: expected " << 2 * gate_out
        << " rows for gate_out " << gate_out << " but w has " << w.shape(-2)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout, exactly as
  // gather_qmm_sorted does.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted_swiglu] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int K = (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  if (K != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmm_sorted_swiglu] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized weight inner dim (" << K
        << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }

  auto s = mx::to_stream(s_);

  // Row-contiguize x / w / sorted_ids at the op level so eval_gpu can assume
  // dense inputs, exactly as gather_qmm_sorted does.
  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  auto ids_c = sorted_ids.flags().row_contiguous
      ? sorted_ids
      : mx::contiguous(sorted_ids, false, s);

  mx::Shape out_shape = {x.shape(0), gate_out};

  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMMSortedSwiglu>(
          s,
          kquant_type,
          codec->weights_per_block,
          codec->bits,
          gate_out,
          swiglu_limit),
      {std::move(x_c), std::move(w_c), std::move(scales), std::move(ids_c)});
}

mx::array gather_qmv_pair_swiglu(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array ids,
    mx::array route_weights,
    int gate_out,
    float swiglu_limit,
    mx::StreamOrDevice s_) {
  // Only the codecs with an instantiated pair kernel are accepted; anything
  // else fails closed here rather than at kernel lookup.
  if (kquant_type != "iq2_xxs") {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_pair_swiglu] no pair+SwiGLU matvec kernel "
        "for codec '" +
        kquant_type + "' (instantiated: iq2_xxs).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_pair_swiglu] w must be uint8 (raw GGUF wire "
        "bytes).");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_pair_swiglu] x must be float16, bfloat16, or "
        "float32.");
  }
  if (x.ndim() != 2 || x.shape(0) != 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] x must be one activation row "
        << "[1, K] but got shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] w must be 3-D "
        << "[n_experts, 2 * gate_out, bytes_per_row] but got shape "
        << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (ids.dtype() != mx::uint32 || ids.ndim() != 1 || ids.shape(0) < 1) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] ids must be a non-empty 1-D "
        << "uint32 array but got " << ids.shape() << " " << ids.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (route_weights.dtype() != mx::float32 || route_weights.ndim() != 1 ||
      route_weights.shape(0) != ids.shape(0)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] route_weights must be 1-D "
        << "float32 with one weight per id (" << ids.shape(0)
        << ") but got shape " << route_weights.shape() << " "
        << route_weights.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (gate_out <= 0 || w.shape(-2) != 2 * gate_out) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] w must stack gate rows then "
        << "up rows per expert: expected " << 2 * gate_out
        << " rows for gate_out " << gate_out << " but w has " << w.shape(-2)
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (gate_out % 4 != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] gate_out (" << gate_out
        << ") must be a multiple of 4 (the qmv row block).";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout, exactly as
  // gather_qmm_sorted_swiglu does.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int K = (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  if (K != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_pair_swiglu] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized weight inner dim (" << K
        << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }

  auto s = mx::to_stream(s_);

  // Row-contiguize at the op level so eval_gpu can assume dense inputs (the
  // kernel indexes x[k], w[((id * 2 * gate_out) + n) * row_bytes], and reads
  // ids / route_weights as flat rows).
  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  auto ids_c = ids.flags().row_contiguous ? ids : mx::contiguous(ids, false, s);
  auto rw_c = route_weights.flags().row_contiguous
      ? route_weights
      : mx::contiguous(route_weights, false, s);

  mx::Shape out_shape = {ids.shape(0), gate_out};

  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVPairSwiglu>(
          s,
          kquant_type,
          codec->weights_per_block,
          codec->bits,
          gate_out,
          swiglu_limit),
      {std::move(x_c),
       std::move(w_c),
       std::move(scales),
       std::move(ids_c),
       std::move(rw_c)});
}

mx::array gather_qmv_expert_sum(
    mx::array x,
    mx::array w,
    mx::array scales,
    const std::string& kquant_type,
    mx::array ids,
    mx::StreamOrDevice s_) {
  // Only the codecs with an instantiated expert-sum kernel are accepted;
  // anything else fails closed here rather than at kernel lookup.
  if (kquant_type != "q2_k") {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_expert_sum] no expert-sum matvec kernel for "
        "codec '" +
        kquant_type + "' (instantiated: q2_k).");
  }
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (w.dtype() != mx::uint8) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_expert_sum] w must be uint8 (raw GGUF wire "
        "bytes).");
  }
  auto dt = x.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16 && dt != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.gather_qmv_expert_sum] x must be float16, bfloat16, or "
        "float32.");
  }
  if (x.ndim() != 2) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_expert_sum] x must be 2-D [B, K] but got "
        << "shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 3) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_expert_sum] w must be 3-D "
        << "[n_experts, N, bytes_per_row] but got shape " << w.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (ids.dtype() != mx::uint32 || ids.ndim() != 1 ||
      ids.shape(0) != x.shape(0)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_expert_sum] ids must be 1-D uint32 with "
        << "one id per x row (" << x.shape(0) << ") but got " << ids.shape()
        << " " << ids.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  int N = w.shape(-2);
  if (N % 4 != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_expert_sum] N (" << N
        << ") must be a multiple of 4 (the qmv row block).";
    throw std::invalid_argument(msg.str());
  }

  // Expand w's quantized geometry from the codec block layout, exactly as
  // gather_qmm_sorted does.
  int w_bytes_per_row = w.shape(-1);
  if (w_bytes_per_row % codec->bytes_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_expert_sum] KQuant weight last dim ("
        << w_bytes_per_row << " bytes) is not a whole number of "
        << codec->bytes_per_block << "-byte " << codec->name << " blocks.";
    throw std::invalid_argument(msg.str());
  }
  int K = (w_bytes_per_row / codec->bytes_per_block) * codec->weights_per_block;
  if (K != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[mlx_kquant.gather_qmv_expert_sum] x last dim (" << x.shape(-1)
        << ") does not match the expanded quantized weight inner dim (" << K
        << ") for codec '" << codec->name << "'.";
    throw std::invalid_argument(msg.str());
  }

  auto s = mx::to_stream(s_);

  // Row-contiguize at the op level so eval_gpu can assume dense inputs (the
  // kernel indexes x[b * K + k], w[((id * N) + n) * row_bytes], and reads
  // ids as a flat row).
  auto x_c = x.flags().row_contiguous ? x : mx::contiguous(x, false, s);
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  auto ids_c = ids.flags().row_contiguous ? ids : mx::contiguous(ids, false, s);

  mx::Shape out_shape = {1, N};

  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantGatherQMVExpertSum>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      {std::move(x_c), std::move(w_c), std::move(scales), std::move(ids_c)});
}

std::vector<mx::array> quantize(
    const mx::array& w,
    const std::string& kquant_type,
    const std::optional<mx::array>& imatrix,
    mx::StreamOrDevice s_) {
  const KQuantCodec* codec = codec_by_name(kquant_type);
  if (codec == nullptr) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] Unknown kquant_type: '" + kquant_type + "'.");
  }
  if (!codec->has_encode) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] Encode is not supported for codec '" +
        kquant_type + "'.");
  }
  if (!mx::issubdtype(w.dtype(), mx::floating)) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] w must be a real floating tensor (float32, "
        "float16, or bfloat16).");
  }
  if (w.ndim() < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] w must have at least 1 dimension.");
  }
  if (w.shape(-1) % codec->weights_per_block != 0) {
    std::ostringstream msg;
    msg << "[mlx_kquant.quantize] Last dim (" << w.shape(-1)
        << ") must be a multiple of weights_per_block ("
        << codec->weights_per_block << ") for codec '" << kquant_type << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (imatrix.has_value()) {
    const auto& im = *imatrix;
    if (im.dtype() != mx::float32) {
      throw std::invalid_argument(
          "[mlx_kquant.quantize] imatrix must be float32.");
    }
    if (im.ndim() != 1 || im.shape(-1) != w.shape(-1)) {
      std::ostringstream msg;
      msg << "[mlx_kquant.quantize] imatrix shape must be [K]=(" << w.shape(-1)
          << ",) but got " << im.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  if (codec->requires_imatrix && !imatrix.has_value()) {
    throw std::invalid_argument(
        "[mlx_kquant.quantize] codec '" + kquant_type +
        "' requires an importance matrix (imatrix); ggml rejects encoding it "
        "without one.");
  }

  auto s = mx::to_stream(s_);
  // IQ encode is CPU-only (ggml has no GPU IQ quantizer); force the op onto the
  // CPU stream UNCONDITIONALLY so KQuantQuantize::eval_cpu runs (it needs a CPU
  // command encoder) even if the caller passed stream=gpu -- there is no GPU IQ
  // encoder to honor. All nine IQ codecs -- and no K-quant -- start with "iq".
  if (kquant_type.rfind("iq", 0) == 0) {
    s = mx::default_stream(mx::Device::cpu);
  }

  auto wq_shape = w.shape();
  wq_shape.back() =
      (w.shape(-1) / codec->weights_per_block) * codec->bytes_per_block;
  mx::Shape s_shape = {1};

  // Row-contiguize w (+ imatrix) at the op level; eval_gpu trusts contiguity.
  auto w_c = w.flags().row_contiguous ? w : mx::contiguous(w, false, s);
  std::vector<mx::array> inputs = {w_c};
  if (imatrix.has_value()) {
    auto im_c = imatrix->flags().row_contiguous
        ? *imatrix
        : mx::contiguous(*imatrix, false, s);
    inputs.push_back(im_c);
  }

  return mx::array::make_arrays(
      {std::move(wq_shape), std::move(s_shape)},
      {mx::uint8, mx::uint8},
      std::make_shared<KQuantQuantize>(
          s, kquant_type, codec->weights_per_block, codec->bits),
      inputs);
}

// ----------------------------- toolchain self-checks
// -----------------------------

// Locate the directory of this shared object (where mlx_kquant.metallib lives).
std::string metallib_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&metallib_dir), &info)) {
      throw std::runtime_error(
          "mlx_kquant: unable to resolve binary directory");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

bool metallib_loads() {
#ifdef _METAL_
  auto& d = mx::metal::device(mx::Device::gpu);
  // get_library throws if <dir>/mlx_kquant.metallib is missing or unreadable.
  d.get_library("mlx_kquant", metallib_dir());
  return true;
#else
  return false;
#endif
}

} // namespace mlx_kquant
