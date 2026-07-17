// KQuantSDPA primitive: vector scaled-dot-product attention for large head dims
// (e.g. 512) that stock MLX's fused vector allowlist excludes. The GPU path
// dispatches the two-pass kernels (kq_sdpa_vector_2pass_1 / _2) from the
// bundled metallib. q is row-contiguous; k/v are read in place via their
// head/seq strides so a strided KV-cache prefix needs no copy. Inference-only
// (no CPU eval).
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
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

namespace {

enum class Q8LoaderArm : size_t {
  ScalarDynamic = 0,
  Uint4Dynamic = 1,
  Uint4ByteDynamic = 2,
  OffContract = 3,
};

struct Q8LoaderConfig {
  const char* suffix;
  Q8LoaderArm arm;
};

const Q8LoaderConfig& q8_loader_config() {
  static const Q8LoaderConfig config = [] {
    const char* env = std::getenv("KQ_SDPA_Q8_UINT4_LOAD");
    if (env != nullptr && env[0] != '\0' && std::string(env) == "0") {
      return Q8LoaderConfig{"scalar_dynamic", Q8LoaderArm::ScalarDynamic};
    }
    if (env != nullptr && env[0] != '\0' && std::string(env) != "1") {
      throw std::runtime_error("KQ_SDPA_Q8_UINT4_LOAD must be 0 or 1.");
    }
    const char* unpack = std::getenv("KQ_SDPA_Q8_VECTOR_BYTE_UNPACK");
    if (
        unpack == nullptr || unpack[0] == '\0' ||
        std::string(unpack) == "1") {
      return Q8LoaderConfig{
          "uint4_byte_dynamic", Q8LoaderArm::Uint4ByteDynamic};
    }
    if (std::string(unpack) == "0") {
      return Q8LoaderConfig{"uint4_dynamic", Q8LoaderArm::Uint4Dynamic};
    }
    throw std::runtime_error("KQ_SDPA_Q8_VECTOR_BYTE_UNPACK must be 0 or 1.");
  }();
  return config;
}

std::array<std::atomic<uint64_t>, 4> q8_loader_dispatch_counts{};

void count_q8_loader_dispatch(Q8LoaderArm arm) {
  q8_loader_dispatch_counts[static_cast<size_t>(arm)].fetch_add(
      1, std::memory_order_relaxed);
}

} // namespace

std::string sdpa_q8_loader_arm() {
  return q8_loader_config().suffix;
}

std::vector<uint64_t> sdpa_q8_loader_dispatch_counts() {
  std::vector<uint64_t> counts;
  counts.reserve(q8_loader_dispatch_counts.size());
  for (const auto& count : q8_loader_dispatch_counts) {
    counts.push_back(count.load(std::memory_order_relaxed));
  }
  return counts;
}

#ifdef _METAL_

namespace {

using mx::array;
using mx::Stream;
using mx::metal::Device;

bool has_uint4_row_alignment(array a) {
  const size_t head_stride =
      static_cast<size_t>(a.shape(1) == 1 ? a.strides(0) : a.strides(1));
  const auto address = reinterpret_cast<uintptr_t>(a.buffer().raw_ptr()) +
      static_cast<uintptr_t>(a.offset());
  return address % 16 == 0 && head_stride % 4 == 0 &&
      static_cast<size_t>(a.strides(2)) % 4 == 0;
}

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

namespace {

// Host mirror of the kernel-side KqPrefillQ8Params (kq_sdpa.h): twelve 8-byte
// strides then four 4-byte scalars, no padding.
struct KqPrefillQ8ParamsHost {
  size_t pk_head, pk_seq;
  size_t pks_head, pks_seq;
  size_t pv_head, pv_seq;
  size_t pvs_head, pvs_seq;
  size_t sk_head, sk_seq;
  size_t sv_head, sv_seq;
  int N;
  int past_len;
  int q_len;
  float scale;
};
static_assert(
    sizeof(KqPrefillQ8ParamsHost) == 12 * sizeof(size_t) + 4 * 4,
    "KqPrefillQ8Params layout must match the kernel struct");

// Host mirror of the kernel-side KqDecodeQ8Params (kq_sdpa.h): eight 8-byte
// strides then one int and one float (the trailing 4 bytes pad to the 8-byte
// alignment). No dense self chunk and no separate past_len (N is the whole
// cache offset).
struct KqDecodeQ8ParamsHost {
  size_t pk_head, pk_seq;
  size_t pks_head, pks_seq;
  size_t pv_head, pv_seq;
  size_t pvs_head, pvs_seq;
  int N;
  float scale;
};
static_assert(
    sizeof(KqDecodeQ8ParamsHost) == 8 * sizeof(size_t) + 2 * 4,
    "KqDecodeQ8Params layout must match the kernel struct");

} // namespace

void KQuantSDPAFAPrefillQ8::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& q = inputs[0];
  const auto& pk_w = inputs[1];
  const auto& pk_s = inputs[2];
  const auto& pk_b = inputs[3];
  const auto& pv_w = inputs[4];
  const auto& pv_s = inputs[5];
  const auto& pv_b = inputs[6];
  const auto& self_k = inputs[7];
  const auto& self_v = inputs[8];

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  int D = q.shape(3);
  int n_kv_heads = pk_w.shape(1);
  int past_len = pk_w.shape(2);
  int N = past_len + qL;
  int gqa_factor = n_q_heads / n_kv_heads;
  int bq = bq_ == 0 ? 32 : bq_;
  int bk = bk_ != 0 ? bk_ : (stage_ == 2 ? 16 : 32);
  int qw = qw_;
  if (qw == 0) {
    qw = bq / gqa_factor;
  }
  int n_query_tiles = (qL + qw - 1) / qw;
  int splits = splits_;
  if (splits == 0) {
    // Prefill-shaped default. The query-tile grid already supplies
    // parallelism, so splits track the key depth (merge partials traffic
    // scales with the split count; measured optimum near one split per 2048
    // keys, capped at 16), with a floor that keeps the grid at least 256
    // threadgroups when the query axis is short.
    int by_depth = std::min(16, std::max(1, N / 2048));
    int tg_base = n_kv_heads * n_query_tiles;
    int for_occupancy = std::max(1, (256 + tg_base - 1) / tg_base);
    splits = std::min(128, std::max(by_depth, for_occupancy));
  }

  auto head_stride = [](const mx::array& a) {
    return static_cast<size_t>(a.shape(1) == 1 ? a.strides(0) : a.strides(1));
  };

  KqPrefillQ8ParamsHost p;
  p.pk_head = head_stride(pk_w);
  p.pk_seq = static_cast<size_t>(pk_w.strides(2));
  p.pks_head = head_stride(pk_s);
  p.pks_seq = static_cast<size_t>(pk_s.strides(2));
  p.pv_head = head_stride(pv_w);
  p.pv_seq = static_cast<size_t>(pv_w.strides(2));
  p.pvs_head = head_stride(pv_s);
  p.pvs_seq = static_cast<size_t>(pv_s.strides(2));
  p.sk_head = head_stride(self_k);
  p.sk_seq = static_cast<size_t>(self_k.strides(2));
  p.sv_head = head_stride(self_v);
  p.sv_seq = static_cast<size_t>(self_v.strides(2));
  p.N = N;
  p.past_len = past_len;
  p.q_len = qL;
  p.scale = scale_;

  // Per-split partials + running max/sum in the [B, Hq, qL, splits, D] layout
  // the shared merge reads with grid (Hq, B, qL). Output is float32.
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

  const std::string ts = stage_ == 0 ? "bfloat16_t"
      : stage_ == 1                  ? "float16_t"
                                     : "float";
  bool has_sinks = false;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
  };

  // Pass 1: one (32 * BQ / 8)-thread threadgroup per (kv-head, query-tile,
  // split).
  {
    std::string kname = "kq_sdpa_fa_prefill_q8_2pass_1_" + ts + "_" +
        std::to_string(D) + "_q" + std::to_string(qw) + "_b" +
        std::to_string(bq) + "_k" + std::to_string(bk);
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    const size_t tg = size_t(32) * (bq / 8);
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_fa_prefill_q8] threadgroup of " +
          std::to_string(tg) + " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(pk_w, 1);
    ce.set_input_array(pk_s, 2);
    ce.set_input_array(pk_b, 3);
    ce.set_input_array(pv_w, 4);
    ce.set_input_array(pv_s, 5);
    ce.set_input_array(pv_b, 6);
    ce.set_input_array(self_k, 7);
    ce.set_input_array(self_v, 8);
    ce.set_output_array(partials, 9);
    ce.set_output_array(sums, 10);
    ce.set_output_array(maxs, 11);
    ce.set_bytes(p, 12);
    MTL::Size group_dims(32, bq / 8, 1);
    MTL::Size grid_dims(n_kv_heads, n_query_tiles, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: the shared kq_sdpa_gqa merge on the float32 output; grid z is the
  // query axis.
  {
    std::string kname = "kq_sdpa_gqa_2pass_2_float_" + std::to_string(D);
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

void KQuantSDPADecodeQ8::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  const auto& q = inputs[0];
  const auto& pk_w = inputs[1];
  const auto& pk_s = inputs[2];
  const auto& pk_b = inputs[3];
  const auto& pv_w = inputs[4];
  const auto& pv_s = inputs[5];
  const auto& pv_b = inputs[6];

  int B = q.shape(0);
  int n_q_heads = q.shape(1);
  int D = q.shape(3);
  int n_kv_heads = pk_w.shape(1);
  int N = pk_w.shape(2);
  int gqa_factor = n_q_heads / n_kv_heads;
  const int bq = gqa_factor; // one query row folded as BQ heads (qL == 1)
  int splits = splits_;
  if (splits == 0) {
    // Coarse depth buckets (a per-N value would mint a new pipeline
    // specialization every decode step), with a floor that keeps the grid at
    // least 256 threadgroups so shallow caches still fill the GPU. Mirrors the
    // sdpa_decode_gqa depth ladder: more splits win as depth grows, roughly one
    // split per 1024-2048 keys.
    int by_depth = N <= 8192 ? 16 : N <= 24576 ? 32 : N <= 49152 ? 64 : 128;
    int tg_base = n_kv_heads;
    int for_occupancy = std::max(1, (256 + tg_base - 1) / tg_base);
    splits = std::min(128, std::max(by_depth, for_occupancy));
  }

  auto head_stride = [](const mx::array& a) {
    return static_cast<size_t>(a.shape(1) == 1 ? a.strides(0) : a.strides(1));
  };

  KqDecodeQ8ParamsHost p;
  p.pk_head = head_stride(pk_w);
  p.pk_seq = static_cast<size_t>(pk_w.strides(2));
  p.pks_head = head_stride(pk_s);
  p.pks_seq = static_cast<size_t>(pk_s.strides(2));
  p.pv_head = head_stride(pv_w);
  p.pv_seq = static_cast<size_t>(pv_w.strides(2));
  p.pvs_head = head_stride(pv_s);
  p.pvs_seq = static_cast<size_t>(pv_s.strides(2));
  p.N = N;
  p.scale = scale_;

  // Per-split partials + running max/sum in the [B, Hq, 1, splits, D] layout
  // the shared merge reads with grid (Hq, B, 1). Output is float32.
  mx::Shape part_shape = {B, n_q_heads, 1, splits, D};
  mx::Shape red_shape = {B, n_q_heads, 1, splits};
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

  const std::string ts = stage_ == 0 ? "bfloat16_t"
      : stage_ == 1                  ? "float16_t"
                                     : "float";
  bool has_sinks = false;
  mx::metal::MTLFCList fc = {
      {&splits, MTL::DataType::DataTypeInt, 2},
      {&has_sinks, MTL::DataType::DataTypeBool, 3},
  };

  // Pass 1. compute_ 1 (default) is the SIMD-shuffle reduction (the
  // decode-latency form); compute_ 0 is the matrix-unit tile (the prefill_q8
  // idiom). Both write the same float32 [B, Hq, 1, splits, D] partials.
  if (compute_ == 1) {
    int tile_c = tile_c_ != 0 ? tile_c_ : 8;
    const bool loader_variant_eligible =
        D == 256 && tile_c == 16 && gqa_factor == 8;
    Q8LoaderArm dispatched_loader = Q8LoaderArm::OffContract;
    bool alignment_fallback = false;
    std::string kname = "kq_sdpa_decode_gqa_q8_2pass_1_" + std::to_string(D) +
        "_c" + std::to_string(tile_c) + "_ne4";
    if (loader_variant_eligible) {
      const auto& selected_loader = q8_loader_config();
      const bool uint4_aligned =
          has_uint4_row_alignment(pk_w) && has_uint4_row_alignment(pv_w);
      const bool selected_uint4 =
          selected_loader.arm == Q8LoaderArm::Uint4Dynamic ||
          selected_loader.arm == Q8LoaderArm::Uint4ByteDynamic;
      if (selected_uint4 && !uint4_aligned) {
        kname += "_scalar_dynamic";
        dispatched_loader = Q8LoaderArm::ScalarDynamic;
        alignment_fallback = true;
      } else {
        kname += "_" + std::string(selected_loader.suffix);
        dispatched_loader = selected_loader.arm;
      }
    }
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    const size_t tg = size_t(32) * gqa_factor;
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_q8] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    count_q8_loader_dispatch(dispatched_loader);
    if (alignment_fallback) {
      count_q8_loader_dispatch(Q8LoaderArm::OffContract);
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(pk_w, 1);
    ce.set_input_array(pk_s, 2);
    ce.set_input_array(pk_b, 3);
    ce.set_input_array(pv_w, 4);
    ce.set_input_array(pv_s, 5);
    ce.set_input_array(pv_b, 6);
    ce.set_output_array(partials, 7);
    ce.set_output_array(sums, 8);
    ce.set_output_array(maxs, 9);
    ce.set_bytes(p, 10);
    MTL::Size group_dims(32, gqa_factor, 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  } else {
    std::string kname = "kq_sdpa_decode_q8_2pass_1_" + ts + "_" +
        std::to_string(D) + "_b" + std::to_string(bq);
    std::string hash = kname + "_s" + std::to_string(splits);
    auto kernel = kq_get_kernel(d, kname, hash, fc);
    const size_t tg = size_t(32) * (bq / 8);
    if (tg > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_q8] threadgroup of " + std::to_string(tg) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(q, 0);
    ce.set_input_array(pk_w, 1);
    ce.set_input_array(pk_s, 2);
    ce.set_input_array(pk_b, 3);
    ce.set_input_array(pv_w, 4);
    ce.set_input_array(pv_s, 5);
    ce.set_input_array(pv_b, 6);
    ce.set_output_array(partials, 7);
    ce.set_output_array(sums, 8);
    ce.set_output_array(maxs, 9);
    ce.set_bytes(p, 10);
    MTL::Size group_dims(32, bq / 8, 1);
    MTL::Size grid_dims(n_kv_heads, B, splits);
    ce.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Pass 2: either the shared kq_sdpa_gqa merge or its exact-order,
  // dimension-parallel fixed-geometry form. The latter changes only which
  // SIMD group owns each output dimension; max, denominator, and each output
  // dimension retain the shared merge's reduction order.
  {
    const bool use_dimension_parallel_merge = dimension_parallel_merge_ &&
        N >= 8192 && splits == 128 && stage_ == 2 && compute_ == 1 &&
        tile_c_ == 16 && B == 1 && n_q_heads == 16 && n_kv_heads == 2 &&
        D == 256;
    const std::string kname = use_dimension_parallel_merge
        ? "kq_sdpa_q8_merge_dim8"
        : "kq_sdpa_gqa_2pass_2_float_" + std::to_string(D);
    MTL::ComputePipelineState* kernel;
    if (use_dimension_parallel_merge) {
      kernel = kq_get_kernel(d, kname);
    } else {
      const std::string hash = kname + "_s" + std::to_string(splits) + "_k0";
      kernel = kq_get_kernel(d, kname, hash, fc);
    }
    const size_t merge_threads = use_dimension_parallel_merge ? 256 : 32;
    if (merge_threads > kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[mlx_kquant.sdpa_decode_q8] merge threadgroup of " +
          std::to_string(merge_threads) +
          " threads exceeds this GPU's pipeline limit (" +
          std::to_string(kernel->maxTotalThreadsPerThreadgroup()) + ").");
    }
    ce.set_compute_pipeline_state(kernel);
    ce.set_input_array(partials, 0);
    ce.set_input_array(sums, 1);
    ce.set_input_array(maxs, 2);
    ce.set_input_array(sums, 3);
    ce.set_output_array(out, 4);
    ce.set_bytes(n_q_heads, 5);
    MTL::Size group_dims(merge_threads, 1, 1);
    MTL::Size grid_dims(n_q_heads, B, 1);
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

void KQuantSDPAFAPrefillQ8::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_prefill_q8] requires a Metal build.");
}

void KQuantSDPADecodeQ8::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_q8] requires a Metal build.");
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

void KQuantSDPAFAPrefillQ8::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_fa_prefill_q8] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPAFAPrefillQ8::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPAFAPrefillQ8::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPAFAPrefillQ8&>(other);
  return scale_ == o.scale_ && qw_ == o.qw_ && bq_ == o.bq_ &&
      bk_ == o.bk_ && splits_ == o.splits_ && stage_ == o.stage_;
}

void KQuantSDPADecodeQ8::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.sdpa_decode_q8] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantSDPADecodeQ8::output_shapes(
    const std::vector<mx::array>& inputs) {
  return {inputs[0].shape()};
}

bool KQuantSDPADecodeQ8::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantSDPADecodeQ8&>(other);
  return scale_ == o.scale_ && splits_ == o.splits_ && stage_ == o.stage_ &&
      compute_ == o.compute_ && tile_c_ == o.tile_c_ &&
      dimension_parallel_merge_ == o.dimension_parallel_merge_;
}

mx::array sdpa_fa_prefill_q8(
    mx::array q,
    mx::array pk_w,
    mx::array pk_s,
    mx::array pk_b,
    mx::array pv_w,
    mx::array pv_s,
    mx::array pv_b,
    mx::array self_k,
    mx::array self_v,
    float scale,
    int group_size,
    int bits,
    int qw,
    int bq,
    int bk,
    int splits,
    int stage,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (group_size != 64 || bits != 8) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] only group_size 64 with 8 bits is "
        "supported.");
  }
  if (q.ndim() != 4 || self_k.ndim() != 4 || self_v.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] q and self k/v must be 4-D.");
  }
  int D = q.shape(-1);
  if (D != 256) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] only head_dim 256 is supported.");
  }
  if (q.dtype() != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] q must be float32 (the served "
        "prefill query dtype).");
  }
  if (q.shape(0) != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] batch size must be 1.");
  }
  int n_q_heads = q.shape(1);
  int qL = q.shape(2);
  for (const auto* a : {&pk_w, &pk_s, &pk_b, &pv_w, &pv_s, &pv_b}) {
    if (a->ndim() != 4 || a->shape(0) != 1) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_fa_prefill_q8] past cache arrays must be 4-D "
          "with batch 1.");
    }
  }
  if (pk_w.dtype() != mx::uint32 || pv_w.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] packed past K/V must be uint32.");
  }
  // Scales and biases follow the dense source dtype at quantize time (a
  // float32-key, bfloat16-value cache stores float32 K scales and bfloat16 V
  // scales). The kernel reads float32; casting is value-preserving for the
  // half formats, so accept any float and cast below.
  for (const auto* a : {&pk_s, &pk_b, &pv_s, &pv_b}) {
    if (!mx::issubdtype(a->dtype(), mx::floating)) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_fa_prefill_q8] past scales and biases must be "
          "floating point.");
    }
  }
  int n_kv_heads = pk_w.shape(1);
  int past_len = pk_w.shape(2);
  if (past_len < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] past length must be >= 1 (the dense "
        "kernel serves an empty cache).");
  }
  const int el_per_word = 4; // 8-bit values per uint32
  if (pk_w.shape(3) != D / el_per_word ||
      pk_s.shape(3) != D / group_size || pk_b.shape(3) != D / group_size) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] past K tuple last dims do not match "
        "head_dim 256 at group 64 / 8 bits.");
  }
  auto same_geom = [&](const mx::array& a, const mx::array& r) {
    return a.shape(1) == r.shape(1) && a.shape(2) == r.shape(2);
  };
  if (!same_geom(pk_s, pk_w) || !same_geom(pk_b, pk_w) ||
      !same_geom(pv_w, pk_w) || !same_geom(pv_s, pk_w) ||
      !same_geom(pv_b, pk_w) || pv_w.shape(3) != pk_w.shape(3) ||
      pv_s.shape(3) != pk_s.shape(3) || pv_b.shape(3) != pk_b.shape(3)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] past K and V tuples must share "
        "heads, length, and packing geometry.");
  }
  if (self_k.shape(1) != n_kv_heads || self_v.shape(1) != n_kv_heads ||
      self_k.shape(2) != qL || self_v.shape(2) != qL ||
      self_k.shape(3) != D || self_v.shape(3) != D) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] self k/v must be [1, n_kv_heads, "
        "qL, 256].");
  }
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] n_q_heads must be a multiple of "
        "n_kv_heads.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (bq == 0) {
    bq = 32;
  }
  if (bq != 32 && bq != 64) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] bq must be 32 or 64 (0 picks 32).");
  }
  if (qw == 0) {
    qw = bq / gqa_factor;
  }
  if (qw < 1 || gqa_factor * qw != bq) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] gqa_factor * qw must equal bq (qw 0 "
        "picks the default).");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] splits must be in [0, 128].");
  }
  if (stage < 0 || stage > 2) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] stage must be 0 (bfloat16), 1 "
        "(float16), or 2 (float32).");
  }
  // Instantiated key-tile widths: half staging 32 (any bq) or 48 (bq 64);
  // float32 staging 16.
  const bool bk_ok = bk == 0 ||
      (stage == 2 ? bk == 16 : (bk == 32 || (bk == 48 && bq == 64)));
  if (!bk_ok) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_fa_prefill_q8] bk not instantiated for this stage "
        "and bq (0 picks the default).");
  }

  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto lastdim_c = [&](mx::array a) {
    return a.strides().back() == 1 ? a : mx::contiguous(a, false, s);
  };
  // Self k/v read through float4 loads, so they must be row-contiguous (the
  // 16-byte alignment follows from a compact head_dim-256 float32 layout).
  auto rowc = [&](mx::array a) {
    return a.flags().row_contiguous ? a : mx::contiguous(a, false, s);
  };
  auto sk_c = rowc(mx::astype(self_k, mx::float32, s));
  auto sv_c = rowc(mx::astype(self_v, mx::float32, s));
  auto pk_w_c = lastdim_c(pk_w);
  auto pv_w_c = lastdim_c(pv_w);
  auto pk_s_c = lastdim_c(mx::astype(pk_s, mx::float32, s));
  auto pk_b_c = lastdim_c(mx::astype(pk_b, mx::float32, s));
  auto pv_s_c = lastdim_c(mx::astype(pv_s, mx::float32, s));
  auto pv_b_c = lastdim_c(mx::astype(pv_b, mx::float32, s));
  // The kernel reads biases through the scale strides; contiguize both when
  // the tuple's slicing left them different.
  if (pk_b_c.strides() != pk_s_c.strides()) {
    pk_s_c = mx::contiguous(pk_s_c, false, s);
    pk_b_c = mx::contiguous(pk_b_c, false, s);
  }
  if (pv_b_c.strides() != pv_s_c.strides()) {
    pv_s_c = mx::contiguous(pv_s_c, false, s);
    pv_b_c = mx::contiguous(pv_b_c, false, s);
  }

  auto out_shape = q_c.shape();
  return mx::array(
      std::move(out_shape),
      mx::float32,
      std::make_shared<KQuantSDPAFAPrefillQ8>(
          s, scale, qw, bq, bk, splits, stage),
      {std::move(q_c),
       std::move(pk_w_c),
       std::move(pk_s_c),
       std::move(pk_b_c),
       std::move(pv_w_c),
       std::move(pv_s_c),
       std::move(pv_b_c),
       std::move(sk_c),
       std::move(sv_c)});
}

mx::array sdpa_decode_q8(
    mx::array q,
    mx::array pk_w,
    mx::array pk_s,
    mx::array pk_b,
    mx::array pv_w,
    mx::array pv_s,
    mx::array pv_b,
    float scale,
    int group_size,
    int bits,
    int splits,
    int stage,
    int compute,
    int tile_c,
    bool dimension_parallel_merge,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);

  if (group_size != 64 || bits != 8) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] only group_size 64 with 8 bits is "
        "supported.");
  }
  if (compute != 0 && compute != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] compute must be 0 (matrix tile) or 1 "
        "(SIMD-shuffle).");
  }
  if (compute == 1 && tile_c != 0 && tile_c != 8 && tile_c != 16) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] SIMD-shuffle tile_c must be 8 or 16 (0 "
        "picks the default).");
  }
  if (q.ndim() != 4) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] q must be 4-D [1, n_q_heads, 1, 256].");
  }
  int D = q.shape(-1);
  if (D != 256) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] only head_dim 256 is supported.");
  }
  if (q.dtype() != mx::float32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] q must be float32 (the served decode "
        "query dtype).");
  }
  if (q.shape(0) != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] batch size must be 1.");
  }
  if (q.shape(2) != 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] query length must be 1 (decode shape; the "
        "prefill kernel serves qL > 1).");
  }
  int n_q_heads = q.shape(1);
  for (const auto* a : {&pk_w, &pk_s, &pk_b, &pv_w, &pv_s, &pv_b}) {
    if (a->ndim() != 4 || a->shape(0) != 1) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_q8] cache arrays must be 4-D with batch 1.");
    }
  }
  if (pk_w.dtype() != mx::uint32 || pv_w.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] packed K/V must be uint32.");
  }
  // Scales and biases follow the dense source dtype at quantize time. The
  // kernel reads float32; casting is value-preserving for the half formats,
  // so accept any float and cast below.
  for (const auto* a : {&pk_s, &pk_b, &pv_s, &pv_b}) {
    if (!mx::issubdtype(a->dtype(), mx::floating)) {
      throw std::invalid_argument(
          "[mlx_kquant.sdpa_decode_q8] scales and biases must be floating "
          "point.");
    }
  }
  int n_kv_heads = pk_w.shape(1);
  int past_len = pk_w.shape(2);
  if (past_len < 1) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] cache length must be >= 1.");
  }
  const int el_per_word = 4; // 8-bit values per uint32
  if (pk_w.shape(3) != D / el_per_word ||
      pk_s.shape(3) != D / group_size || pk_b.shape(3) != D / group_size) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] K tuple last dims do not match head_dim "
        "256 at group 64 / 8 bits.");
  }
  auto same_geom = [&](const mx::array& a, const mx::array& r) {
    return a.shape(1) == r.shape(1) && a.shape(2) == r.shape(2);
  };
  if (!same_geom(pk_s, pk_w) || !same_geom(pk_b, pk_w) ||
      !same_geom(pv_w, pk_w) || !same_geom(pv_s, pk_w) ||
      !same_geom(pv_b, pk_w) || pv_w.shape(3) != pk_w.shape(3) ||
      pv_s.shape(3) != pk_s.shape(3) || pv_b.shape(3) != pk_b.shape(3)) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] K and V tuples must share heads, length, "
        "and packing geometry.");
  }
  if (n_kv_heads == 0 || n_q_heads % n_kv_heads != 0) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] n_q_heads must be a multiple of "
        "n_kv_heads.");
  }
  int gqa_factor = n_q_heads / n_kv_heads;
  if (gqa_factor != 8) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] only the gqa factor 8 fold is "
        "instantiated (16 query heads over 2 kv heads).");
  }
  if (splits < 0 || splits > 128) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] splits must be in [0, 128].");
  }
  if (stage < 0 || stage > 2) {
    throw std::invalid_argument(
        "[mlx_kquant.sdpa_decode_q8] stage must be 0 (bfloat16), 1 (float16), "
        "or 2 (float32).");
  }

  auto q_c = q.flags().row_contiguous ? q : mx::contiguous(q, false, s);
  auto lastdim_c = [&](mx::array a) {
    return a.strides().back() == 1 ? a : mx::contiguous(a, false, s);
  };
  auto pk_w_c = lastdim_c(pk_w);
  auto pv_w_c = lastdim_c(pv_w);
  auto pk_s_c = lastdim_c(mx::astype(pk_s, mx::float32, s));
  auto pk_b_c = lastdim_c(mx::astype(pk_b, mx::float32, s));
  auto pv_s_c = lastdim_c(mx::astype(pv_s, mx::float32, s));
  auto pv_b_c = lastdim_c(mx::astype(pv_b, mx::float32, s));
  // The kernel reads biases through the scale strides; contiguize both when
  // the tuple's slicing left them different.
  if (pk_b_c.strides() != pk_s_c.strides()) {
    pk_s_c = mx::contiguous(pk_s_c, false, s);
    pk_b_c = mx::contiguous(pk_b_c, false, s);
  }
  if (pv_b_c.strides() != pv_s_c.strides()) {
    pv_s_c = mx::contiguous(pv_s_c, false, s);
    pv_b_c = mx::contiguous(pv_b_c, false, s);
  }

  auto out_shape = q_c.shape();
  return mx::array(
      std::move(out_shape),
      mx::float32,
      std::make_shared<KQuantSDPADecodeQ8>(
          s, scale, splits, stage, compute, tile_c, dimension_parallel_merge),
      {std::move(q_c),
       std::move(pk_w_c),
       std::move(pk_s_c),
       std::move(pk_b_c),
       std::move(pv_w_c),
       std::move(pv_s_c),
       std::move(pv_b_c)});
}

} // namespace mlx_kquant
