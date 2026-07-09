// KQuantSDPA primitive: vector scaled-dot-product attention for large head dims
// (e.g. 512) that stock MLX's fused vector allowlist excludes. The GPU path
// dispatches the two-pass kernels (kq_sdpa_vector_2pass_1 / _2) from the
// bundled metallib. q is row-contiguous; k/v are read in place via their
// head/seq strides so a strided KV-cache prefix needs no copy. Inference-only
// (no CPU eval).
#include <stdexcept>
#include <string>

#include "kquant.h"
#include "kquant_internal.h" // kq_type_string

#include "mlx/ops.h" // contiguous
#include "mlx/utils.h" // to_stream

#ifdef _METAL_
#include "kquant_metal_internal.h" // kq_get_kernel
#include "mlx/backend/metal/device.h"
#endif

namespace mx = mlx::core;

namespace mlx_kquant {

#ifdef _METAL_

namespace {

using mx::array;
using mx::Stream;
using mx::metal::Device;

// Number of key-blocks to split the reduction across. Mirrors MLX's own
// sdpa_vector_2pass heuristic: more blocks only when there are enough
// simdgroups per kv-head (n_simds = gqa_factor * qL) to justify the extra
// partials.
int kq_sdpa_blocks(int N, int n_simds, Device& d) {
  char devc = d.get_architecture().back();
  int blocks;
  if (devc == 's') {
    blocks = 64;
    if (N > 1024 && n_simds > 4) {
      if (N <= 8192) {
        blocks = 128;
      } else if (N <= 32768) {
        blocks = 256;
      } else if (N <= 65536) {
        blocks = 512;
      } else {
        blocks = 1024;
      }
    }
  } else if (devc == 'd') {
    blocks = 128;
    if (n_simds <= 2 && N > 8192) {
      blocks = 256;
    } else if (n_simds >= 6) {
      if (N >= 16384 && N < 65536) {
        blocks = 512;
      } else if (N >= 65536) {
        blocks = 1024;
      }
    }
  } else {
    blocks = (n_simds >= 4) ? 64 : 32;
  }
  return blocks;
}

} // namespace

void KQuantSDPA::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // q row-contiguous [B, Hq, qL, D]; k/v [B, Hkv, kL, D], D contiguous.
  // Optional inputs follow in order: mask (bool), sinks (float32 [Hq]).
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  size_t next_input = 3;
  const mx::array* mask = has_mask_ ? &inputs[next_input++] : nullptr;
  const mx::array* sinks = has_sinks_ ? &inputs[next_input++] : nullptr;

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  int kL = k.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;

  size_t k_head_stride =
      static_cast<size_t>(k.shape(1) == 1 ? k.strides(0) : k.strides(1));
  size_t k_seq_stride = static_cast<size_t>(k.strides(2));
  size_t v_head_stride =
      static_cast<size_t>(v.shape(1) == 1 ? v.strides(0) : v.strides(1));
  size_t v_seq_stride = static_cast<size_t>(v.strides(2));

  // Wide-MQA dispatch: one threadgroup hosts the whole GQA group (32 * gqa *
  // qL threads), which overflows the 1024-thread cap for MQA models with many
  // query heads (e.g. 64 heads on one shared latent head). With a single kv
  // head every group reads the same K/V, so dispatch one query head per
  // threadgroup and broadcast K/V through a zero head stride instead. The
  // kernel folds the batch offset into the same stride, so this mode is
  // limited to B == 1 (the op validates that).
  if (32 * gqa_factor * qL > 1024 && n_kv_heads == 1 && B == 1) {
    n_kv_heads = n_q_heads;
    gqa_factor = 1;
    k_head_stride = 0;
    v_head_stride = 0;
  }
  int n_simds = gqa_factor * qL;
  int blocks = kq_sdpa_blocks(kL, n_simds, d);
  float scale = scale_;

  // Per-block partials + running max/sum, reduced by pass 2.
  mx::Shape part_shape = {B, n_q_heads, qL, blocks, D};
  mx::Shape red_shape = {B, n_q_heads, qL, blocks};
  array partials(part_shape, q.dtype(), nullptr, {});
  array sums(red_shape, mx::float32, nullptr, {});
  array maxs(red_shape, mx::float32, nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));
  sums.set_data(mx::allocator::malloc(sums.nbytes()));
  maxs.set_data(mx::allocator::malloc(maxs.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);
  ce.add_temporary(sums);
  ce.add_temporary(maxs);

  std::string ts = kq_type_string(q.dtype());
  bool causal = causal_;
  bool with_mask = has_mask_;
  bool with_sinks = has_sinks_;
  mx::metal::MTLFCList fc = {
      {&causal, MTL::DataType::DataTypeBool, 0},
      {&blocks, MTL::DataType::DataTypeInt, 1},
      {&with_mask, MTL::DataType::DataTypeBool, 4},
      {&with_sinks, MTL::DataType::DataTypeBool, 5},
  };

  // Broadcast dims read through zero strides; the column stride is 1 (the op
  // contiguizes the mask's last dim).
  size_t m_batch_stride = 0, m_head_stride = 0, m_q_stride = 0;
  if (mask != nullptr) {
    m_batch_stride =
        mask->shape(0) == 1 ? 0 : static_cast<size_t>(mask->strides(0));
    m_head_stride =
        mask->shape(1) == 1 ? 0 : static_cast<size_t>(mask->strides(1));
    m_q_stride =
        mask->shape(2) == 1 ? 0 : static_cast<size_t>(mask->strides(2));
  }

  // Pass 1: each (kv-head, batch, block) threadgroup computes a partial output.
  {
    std::string kname =
        "kq_sdpa_vector_2pass_1_" + ts + "_" + std::to_string(D);
    std::string hash = kname + (causal ? "_c1" : "_c0") + "_b" +
        std::to_string(blocks) + (with_mask ? "_m1" : "_m0") +
        (with_sinks ? "_s1" : "_s0");
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
    // Metal turns an oversized dispatch into silent garbage, not an error.
    const size_t tg = size_t(32) * gqa_factor * qL;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_vector] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(k, 1);
    ce.set_input_array(v, 2);
    ce.set_output_array(partials, 3);
    ce.set_output_array(sums, 4);
    ce.set_output_array(maxs, 5);
    ce.set_bytes(kL, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    if (mask != nullptr) {
      ce.set_input_array(*mask, 12);
      ce.set_bytes(m_batch_stride, 13);
      ce.set_bytes(m_head_stride, 14);
      ce.set_bytes(m_q_stride, 15);
    }
    MTL::Size group_dims(32, gqa_factor, qL);
    MTL::Size grid_dims(n_kv_heads, B, blocks);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: reduce per-block partials into the final output.
  {
    std::string kname =
        "kq_sdpa_vector_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_b" + std::to_string(blocks) +
        (with_sinks ? "_s1" : "_s0");
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    ce.set_output_array(out, 3);
    if (sinks != nullptr) {
      ce.set_input_array(*sinks, 4);
      ce.set_bytes(n_q_heads, 5);
    }
    MTL::Size group_dims(1024, 1, 1);
    MTL::Size grid_dims(B * n_q_heads, qL, 1);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void KQuantSDPAGQA::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const bool sinks = inputs.size() == 4;

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  int kL = k.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  // Auto splits: coarse buckets (a per-kL value would mint a new pipeline
  // specialization every decode step). Measured on M5 Max: more splits win as
  // depth grows; ~512-1024 keys per chunk is the sweet spot.
  int splits = splits_;
  if (splits == 0) {
    splits = kL <= 8192 ? 16 : kL <= 24576 ? 32 : kL <= 49152 ? 64 : 128;
  }

  size_t k_head_stride =
      static_cast<size_t>(k.shape(1) == 1 ? k.strides(0) : k.strides(1));
  size_t k_seq_stride = static_cast<size_t>(k.strides(2));
  size_t v_head_stride =
      static_cast<size_t>(v.shape(1) == 1 ? v.strides(0) : v.strides(1));
  size_t v_seq_stride = static_cast<size_t>(v.strides(2));
  float scale = scale_;

  // Coarse per-split partials (float32) + running max/sum, merged by pass 2.
  mx::Shape part_shape = {B, n_q_heads, qL, splits, D};
  mx::Shape red_shape = {B, n_q_heads, qL, splits};
  array partials(part_shape, mx::float32, nullptr, {});
  array sums(red_shape, mx::float32, nullptr, {});
  array maxs(red_shape, mx::float32, nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));
  sums.set_data(mx::allocator::malloc(sums.nbytes()));
  maxs.set_data(mx::allocator::malloc(maxs.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);
  ce.add_temporary(sums);
  ce.add_temporary(maxs);

  std::string ts = kq_type_string(q.dtype());
  bool has_sinks = sinks;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
  };

  // Pass 1: one threadgroup per (kv-head, batch, split); the whole GQA group
  // (and, at verify width, every query pair -- the threadgroup z axis) shares
  // each staged K/V tile. qL > 1 dispatches the _p2 (two queries per
  // simdgroup) instantiation.
  {
    std::string kname = "kq_sdpa_gqa_2pass_1_" + ts + "_" + std::to_string(D) +
        "_c" + std::to_string(tile_c_) + (qL > 1 ? "_p2" : "");
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
    // Metal turns an oversized dispatch into silent garbage, not an error.
    const size_t tg =
        size_t(32) * gqa_factor * (qL > 1 ? size_t((qL + 1) / 2) : 1);
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_gqa] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(k, 1);
    ce.set_input_array(v, 2);
    ce.set_output_array(partials, 3);
    ce.set_output_array(sums, 4);
    ce.set_output_array(maxs, 5);
    ce.set_bytes(kL, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    ce.set_bytes(qL, 12);
    MTL::Size group_dims(32, gqa_factor, qL > 1 ? (qL + 1) / 2 : 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: merge the per-split partials; sinks fold into the denominator.
  // Grid z is the query axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash =
        kname + "_s" + std::to_string(splits) + (has_sinks ? "_k1" : "_k0");
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    // Metal wants every buffer bound; without sinks, rebind sums as a dummy
    // (the read is compiled out via the function constant).
    ce.set_input_array(sinks ? inputs[3] : sums, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_q_heads, 5);
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_q_heads, B, qL);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void KQuantSDPAFAVerify::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // q is the GQA-folded query tile [B, Hkv, n_rows, D], row-contiguous;
  // k/v [B, Hkv, kL, D] are read in place via their head/seq strides.
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];

  int B = q.shape(0);
  int n_kv_heads = k.shape(1);
  int n_rows = q.shape(2);
  int D = q.shape(3);
  int kL = k.shape(2);
  int q_len = q_len_;
  // Same coarse split buckets as sdpa_decode_gqa (a per-kL value would mint
  // a new pipeline specialization every decode step).
  int splits = splits_;
  if (splits == 0) {
    splits = kL <= 8192 ? 16 : kL <= 24576 ? 32 : kL <= 49152 ? 64 : 128;
  }

  size_t k_head_stride =
      static_cast<size_t>(k.shape(1) == 1 ? k.strides(0) : k.strides(1));
  size_t k_seq_stride = static_cast<size_t>(k.strides(2));
  size_t v_head_stride =
      static_cast<size_t>(v.shape(1) == 1 ? v.strides(0) : v.strides(1));
  size_t v_seq_stride = static_cast<size_t>(v.strides(2));
  float scale = scale_;

  // Coarse per-split partials (float32) + running max/sum; the folded row
  // axis flattens identically to the unfolded [B, Hq, qL] layout, so the
  // kq_sdpa_gqa merge pass is reused unchanged.
  mx::Shape part_shape = {B, n_kv_heads, n_rows, splits, D};
  mx::Shape red_shape = {B, n_kv_heads, n_rows, splits};
  array partials(part_shape, mx::float32, nullptr, {});
  array sums(red_shape, mx::float32, nullptr, {});
  array maxs(red_shape, mx::float32, nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));
  sums.set_data(mx::allocator::malloc(sums.nbytes()));
  maxs.set_data(mx::allocator::malloc(maxs.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);
  ce.add_temporary(sums);
  ce.add_temporary(maxs);

  std::string ts = kq_type_string(q.dtype());
  bool has_sinks = false;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
  };

  // Pass 1: one 128-thread threadgroup per (kv-head, batch, split) streams
  // its key chunk through the simdgroup-matrix tile.
  {
    std::string kname =
        "kq_sdpa_fa_verify_2pass_1_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    // Register-heavy pipeline: some GPUs cap it below the dispatch width, and
    // Metal turns an oversized dispatch into silent garbage, not an error.
    const size_t tg = 128;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_fa_verify] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(k, 1);
    ce.set_input_array(v, 2);
    ce.set_output_array(partials, 3);
    ce.set_output_array(sums, 4);
    ce.set_output_array(maxs, 5);
    ce.set_bytes(kL, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    ce.set_bytes(q_len, 12);
    ce.set_bytes(n_rows, 13);
    MTL::Size group_dims(32, 4, 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: the shared kq_sdpa_gqa merge; grid z is the folded row axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_s" + std::to_string(splits) + "_k0";
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    // No sinks: rebind sums as a dummy (the read is compiled out via the
    // function constant).
    ce.set_input_array(sums, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_kv_heads, 5);
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_kv_heads, B, n_rows);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

void KQuantSDPAFAPrefill::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // q [1, Hq, qL, D] row-contiguous; k/v [1, Hkv, kL, D] read in place.
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = k.shape(1);
  int kL = k.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  int qw = qw_;
  if (qw == 0) {
    qw = 32 / gqa_factor;
  }
  int n_query_tiles = (qL + qw - 1) / qw;
  // Same coarse split buckets as sdpa_decode_gqa / fa_verify.
  int splits = splits_;
  if (splits == 0) {
    splits = kL <= 8192 ? 16 : kL <= 24576 ? 32 : kL <= 49152 ? 64 : 128;
  }

  size_t k_head_stride =
      static_cast<size_t>(k.shape(1) == 1 ? k.strides(0) : k.strides(1));
  size_t k_seq_stride = static_cast<size_t>(k.strides(2));
  size_t v_head_stride =
      static_cast<size_t>(v.shape(1) == 1 ? v.strides(0) : v.strides(1));
  size_t v_seq_stride = static_cast<size_t>(v.strides(2));
  float scale = scale_;

  // Per-split partials (float32) + running max/sum in the [B, Hq, qL, splits, D]
  // layout the shared merge reads with grid (Hq, B, qL).
  mx::Shape part_shape = {B, n_q_heads, qL, splits, D};
  mx::Shape red_shape = {B, n_q_heads, qL, splits};
  array partials(part_shape, mx::float32, nullptr, {});
  array sums(red_shape, mx::float32, nullptr, {});
  array maxs(red_shape, mx::float32, nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));
  sums.set_data(mx::allocator::malloc(sums.nbytes()));
  maxs.set_data(mx::allocator::malloc(maxs.nbytes()));

  auto& ce = mx::metal::get_command_encoder(s);
  ce.add_temporary(partials);
  ce.add_temporary(sums);
  ce.add_temporary(maxs);

  std::string ts = kq_type_string(q.dtype());
  bool has_sinks = false;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
  };

  // Pass 1: one 128-thread threadgroup per (kv-head, query-tile, split) streams
  // its key chunk through the simdgroup-matrix tile.
  {
    std::string kname = "kq_sdpa_fa_prefill_2pass_1_" + ts + "_" +
        std::to_string(D) + "_q" + std::to_string(qw);
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    const size_t tg = 128;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_fa_prefill] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(k, 1);
    ce.set_input_array(v, 2);
    ce.set_output_array(partials, 3);
    ce.set_output_array(sums, 4);
    ce.set_output_array(maxs, 5);
    ce.set_bytes(kL, 6);
    ce.set_bytes(k_head_stride, 7);
    ce.set_bytes(k_seq_stride, 8);
    ce.set_bytes(v_head_stride, 9);
    ce.set_bytes(v_seq_stride, 10);
    ce.set_bytes(scale, 11);
    ce.set_bytes(qL, 12);
    MTL::Size group_dims(32, 4, 1);
    MTL::Size grid_dims(n_kv_heads, n_query_tiles, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: the shared kq_sdpa_gqa merge; grid z is the query axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_" + ts + "_" + std::to_string(D);
    std::string hash = kname + "_s" + std::to_string(splits) + "_k0";
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    ce.set_input_array(sums, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_q_heads, 5);
    MTL::Size group_dims(32, 1, 1);
    MTL::Size grid_dims(n_q_heads, B, qL);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }
}

#else // !_METAL_

void KQuantSDPA::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[mlx_kquant.sdpa_vector] requires a Metal build.");
}

void KQuantSDPAGQA::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_gqa] requires a Metal build.");
}

void KQuantSDPAFAVerify::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_verify] requires a Metal build.");
}

void KQuantSDPAFAPrefill::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_prefill] requires a Metal build.");
}

#endif

void KQuantSDPA::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_vector] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPA::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPA::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPA&>(other);
  return scale_ == o.scale_ && causal_ == o.causal_ &&
      has_mask_ == o.has_mask_ && has_sinks_ == o.has_sinks_;
}

mx::array sdpa_vector(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    bool causal,
    std::optional<mx::array> mask,
    std::optional<mx::array> sinks,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if (v.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] value head dim must equal query head dim.");
  }
  if (D != 256 && D != 512) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] only head_dim 256 or 512 is supported.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] q, k, v must share a dtype.");
  }
  int n_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] n_q_heads must be a multiple of n_kv_heads.");
  }
  int qL = q.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  // pass-1 threadgroup is 32 * gqa_factor * qL threads, capped at the Metal
  // max. Above the cap only the B==1 single-kv-head (wide MQA) dispatch is
  // available: one query head per threadgroup with broadcast K/V.
  if (32 * gqa_factor * qL > 1024 &&
      !(n_kv_heads == 1 && q.shape(0) == 1 && 32 * qL <= 1024)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] gqa_factor * qL exceeds the 32-wide limit "
        "(wide MQA is supported only for B == 1, n_kv_heads == 1).");
  }
  if (qL > k.shape(2)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_vector] query length exceeds key length.");
  }

  int kL = k.shape(2);
  if (mask.has_value()) {
    if (mask->ndim() != 4 || mask->dtype() != mx::bool_) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_vector] mask must be a 4-D boolean array.");
    }
    if (mask->shape(3) != kL) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_vector] mask last dim must equal key length.");
    }
    auto bad_dim = [](int m, int full) { return m != 1 && m != full; };
    if (bad_dim(mask->shape(0), q.shape(0)) ||
        bad_dim(mask->shape(1), n_q_heads) || bad_dim(mask->shape(2), qL)) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_vector] mask must broadcast to [B, Hq, qL, kL].");
    }
  }
  if (sinks.has_value()) {
    if (sinks->size() != static_cast<size_t>(n_q_heads)) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_vector] sinks must hold one logit per query head.");
    }
  }

  // q small -> contiguize if needed (cheap). k/v: only the last (head) dim must
  // be contiguous; head/seq strides are read in place, so a strided KV-cache
  // prefix is passed through without a copy.
  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(k_c), std::move(v_c)};
  if (mask.has_value()) {
    auto m = mask->strides().back() == 1
        ? *mask
        : mx::contiguous(*mask, false, s);
    inputs.push_back(std::move(m));
  }
  if (sinks.has_value()) {
    inputs.push_back(mx::astype(
        mx::contiguous(mx::reshape(*sinks, {n_q_heads}, s), false, s),
        mx::float32,
        s));
  }

  auto out_shape = inputs[0].shape();
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantSDPA>(
          s, scale, causal, mask.has_value(), sinks.has_value()),
      std::move(inputs));
}

void KQuantSDPAGQA::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_gqa] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPAGQA::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPAGQA::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAGQA&>(other);
  return scale_ == o.scale_ && splits_ == o.splits_ && tile_c_ == o.tile_c_;
}

mx::array sdpa_decode_gqa(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    const std::optional<mx::array>& sinks,
    int splits,
    int tile_c,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if ((D != 64 && D != 128 && D != 256 && D != 512) || v.shape(-1) != D ||
      k.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] only head_dim 64/128/256/512 is "
        "supported.");
  }
  int qL = q.shape(2);
  if (qL < 1 || qL > 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] query length must be 1 (decode) "
        "to 4 (speculative-verify width).");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] q, k, v must share a dtype.");
  }
  int n_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] n_q_heads must be a multiple of "
        "n_kv_heads.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (gqa_factor > 16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] gqa_factor must be <= 16.");
  }
  // Pass-1 threadgroup is 32 * gqa_factor * ceil(qL / 2) threads (Metal max
  // 1024).
  if (gqa_factor * ((qL + 1) / 2) > 32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] gqa_factor * ceil(query length / 2) "
        "must be <= 32 (1024-thread threadgroup).");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] splits must be in [0, 128].");
  }
  if (tile_c == 0) {
    tile_c = D <= 128 ? 32 : D == 256 ? 16 : 8;
  }
  // Instantiated (D, C) pairs: threadgroup K+V tiles cap at 16 KB so two
  // threadgroups co-reside per core (D=64/128: C 32/16; 256: 16/8; 512: 8).
  const bool tile_ok = (D <= 128 && (tile_c == 32 || tile_c == 16)) ||
      (D == 256 && (tile_c == 16 || tile_c == 8)) || (D == 512 && tile_c == 8);
  if (!tile_ok) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] tile_c not instantiated for this "
        "head_dim (0 picks the default).");
  }
  if (k.shape(2) < qL) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_gqa] key length must be >= query length.");
  }

  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  std::vector<mx::array> inputs = {
      std::move(q_c), std::move(k_c), std::move(v_c)};
  if (sinks.has_value()) {
    auto sk = *sinks;
    if (sk.size() != static_cast<size_t>(n_q_heads)) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_gqa] sinks must have n_q_heads elements.");
    }
    sk = mx::astype(mx::reshape(sk, {n_q_heads}, s), mx::float32, s);
    inputs.push_back(mx::contiguous(sk, false, s));
  }

  auto out_shape = q.shape();
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantSDPAGQA>(s, scale, splits, tile_c),
      std::move(inputs));
}

void KQuantSDPAFAVerify::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_verify] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPAFAVerify::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPAFAVerify::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAFAVerify&>(other);
  return scale_ == o.scale_ && q_len_ == o.q_len_ && splits_ == o.splits_;
}

mx::array sdpa_fa_verify(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    int q_len,
    int splits,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  // head_dim 512 needs a BD-chunked output accumulator (see kq_sdpa.metal).
  if (D != 256 || k.shape(-1) != D || v.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] only head_dim 256 is supported.");
  }
  auto dt = q.dtype();
  if (dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q must be float16 or bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q, k, v must share a dtype.");
  }
  if (q.shape(0) != 1 || k.shape(0) != 1 || v.shape(0) != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] batch size must be 1.");
  }
  if (q.shape(1) != k.shape(1) || v.shape(1) != k.shape(1)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q must be GQA-folded: q heads must "
        "equal kv heads.");
  }
  if (v.shape(2) != k.shape(2)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] k and v must share a key length.");
  }
  if (q_len < 2 || q_len > 8) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] q_len must be in [2, 8].");
  }
  int n_rows = q.shape(2);
  if (n_rows < q_len || n_rows > 32 || n_rows % q_len != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] folded rows must be a multiple of q_len "
        "and <= 32.");
  }
  if (k.shape(2) < q_len) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] key length must be >= q_len.");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_verify] splits must be in [0, 128].");
  }

  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  auto out_shape = q_c.shape();
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantSDPAFAVerify>(s, scale, q_len, splits),
      {std::move(q_c), std::move(k_c), std::move(v_c)});
}

void KQuantSDPAFAPrefill::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_prefill] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPAFAPrefill::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPAFAPrefill::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAFAPrefill&>(other);
  return scale_ == o.scale_ && qw_ == o.qw_ && splits_ == o.splits_;
}

mx::array sdpa_fa_prefill(
    mx::array q,
    mx::array k,
    mx::array v,
    float scale,
    int qw,
    int splits,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] q, k, v must be 4-D [B, heads, L, D].");
  }
  int D = q.shape(-1);
  if (D != 256 || k.shape(-1) != D || v.shape(-1) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] only head_dim 256 is supported.");
  }
  auto dt = q.dtype();
  if (dt != mx::float32 && dt != mx::float16 && dt != mx::bfloat16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] q must be float32, float16, or "
        "bfloat16.");
  }
  if (k.dtype() != dt || v.dtype() != dt) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] q, k, v must share a dtype.");
  }
  if (q.shape(0) != 1 || k.shape(0) != 1 || v.shape(0) != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] batch size must be 1.");
  }
  int n_q_heads = q.shape(1);
  int n_kv_heads = k.shape(1);
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] n_q_heads must be a multiple of "
        "n_kv_heads.");
  }
  if (v.shape(1) != n_kv_heads || v.shape(2) != k.shape(2)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] k and v must share heads and key "
        "length.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (qw == 0) {
    qw = 32 / gqa_factor;
  }
  // One 32-row tile folds the whole GQA group with qw query positions.
  if (qw < 1 || gqa_factor * qw != 32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] gqa_factor * qw must equal 32 (qw 0 "
        "picks the default).");
  }
  int qL = q.shape(2);
  int kL = k.shape(2);
  if (kL < qL) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] key length must be >= query length.");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill] splits must be in [0, 128].");
  }

  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto k_c = k.strides().back() == 1 ? k : mx::contiguous(k, false, s);
  auto v_c = v.strides().back() == 1 ? v : mx::contiguous(v, false, s);

  auto out_shape = q_c.shape();
  return mx::array(
      std::move(out_shape),
      dt,
      std::make_shared<KQuantSDPAFAPrefill>(s, scale, qw, splits),
      {std::move(q_c), std::move(k_c), std::move(v_c)});
}

} // namespace mlx_kquant
