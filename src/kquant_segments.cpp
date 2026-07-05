// KQuantGatherQMMSegments primitive: descriptor-driven segmented (MoE)
// quantized GEMM. The GPU path dispatches the kq_<codec>_gather_qmm_segments
// kernel from the bundled metallib. Grid is (ceil(N/BN), T, 1): one threadgroup
// per (n-tile, segment). Each threadgroup reads its {expert, row_start,
// row_count} descriptor from the segments buffer and loops over its rows in
// BM-row chunks, dequantizing weight tiles into threadgroup float32 memory and
// accumulating in float32. Inference-only (no CPU eval). The segments table is
// host-built by the caller but read on the GPU (never dereferenced on the host)
// so no synchronization stalls the pipeline; grid sizing needs only T (from the
// segments array shape) and N.
//
// KQuantGatherQMMSorted is the table-free variant for barrier-free callers:
// grid (ceil(N/BN), E, 1) with E = w.shape(0), and each threadgroup binary
// searches the device-resident sorted per-row expert ids for its own row
// range. Grid sizing needs nothing about the segment structure, so the caller
// never reads the routing indices on the host at all.
#include <cstdlib>
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

namespace {

// Tile geometry must match a kernel instantiation in kq_segments.metal
// (WM=WN=2 -> WM*WN*32 = 128 threads per threadgroup). The tag is
// t{BM}x{BN}x{BK} with an optional trailing letter: 'a' for the device-A
// body (activation fragments read from device rows, unpadded weight tile;
// see kq_segments.h), 'h' for the half-staging opt-in. The default
// t48x128x16a keeps the f32-decode contract (float weight staging, float
// accumulate) and is bit-identical to the staged float tiles; on the sorted
// MoE bulk-prefill shape (S=23058, K=N=4096, E=256) it measures 89.3-91.0 ms
// per layer for iq2_xxs against 105.6-106.5 ms for the previous t48x64x16
// staged default (alternating matched arms). The win is footprint and
// traffic: dropping the staged activation tile and the weight-tile padding
// shrinks the threadgroup footprint from ~14 KB to 8 KB, and BN=128 halves
// how often each activation strip is re-read across n-tiles. BM=48 covers a
// 90-row segment in two row-chunks; BM=96 (halving the redundant weight
// decode) and every register-prefetch, double-buffered, paired-decode, and
// 256-thread variant measured slower and was dropped. The staged tiles stay
// as KQ_SEG_TILE A/B levers (t48x64x16 pins the previous default), plus the
// half-staging opt-in for callers whose quality gate allows f16-precision
// weight decode. Only BN (the n-tile width) reaches the host - it sizes the
// grid; BM/BK and the body flag are internal to the kernel. The segments and
// sorted kernels share the selection so a segments/sorted parity pair always
// runs the same tile.
const std::string& kq_seg_tile() {
  static const std::string tile = []() {
    const char* e = std::getenv("KQ_SEG_TILE");
    return std::string(e != nullptr ? e : "t48x128x16a");
  }();
  return tile;
}

// Parse BN (the middle field of t{BM}x{BN}x{BK}) so the grid width matches
// whichever tile the name selects.
int kq_seg_tile_bn(const std::string& tile) {
  int BN = 64;
  const auto first = tile.find('x');
  const auto second =
      first == std::string::npos ? first : tile.find('x', first + 1);
  if (first != std::string::npos && second != std::string::npos) {
    BN = std::stoi(tile.substr(first + 1, second - first - 1));
  }
  return BN;
}

} // namespace

std::vector<mx::Shape> KQuantGatherQMMSegments::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  return {mx::Shape{x.shape(0), w.shape(-2)}};
}

bool KQuantGatherQMMSegments::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMMSegments&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

#ifdef _METAL_

void KQuantGatherQMMSegments::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs: x (float16/bfloat16/float32, row-contiguous [S, K]), w (uint8
  // [E, N, bytes_per_row]), scales (vestigial placeholder), segments
  // (uint32 [T, 3]). The kernel name's type token (from kq_type_string)
  // selects the half or float staging instantiation.
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  const auto& segments = inputs[3];

  int S = x.shape(0);
  int K = x.shape(1);
  int N = w.shape(1);
  int T = segments.shape(0);

  // Tile selection is shared with the sorted kernel (kq_seg_tile above).
  const std::string& tile = kq_seg_tile();
  int BN = kq_seg_tile_bn(tile);
  constexpr int WM = 2, WN = 2;
  constexpr int TG_THREADS = WM * WN * 32;

  // Grid: one threadgroup per (n-tile, segment). The kernel loops over each
  // segment's rows internally, so the host needs no per-segment row counts.
  MTL::Size group_dims(TG_THREADS, 1, 1);
  MTL::Size grid_dims((N + BN - 1) / BN, T, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname = kq_kname_prefix(kquant_type_) + "gather_qmm_segments_" +
      type_string + "_" + tile;

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(segments, c++);
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(S, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantGatherQMMSegments::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmm_segments] requires a Metal build.");
}

#endif

void KQuantGatherQMMSegments::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmm_segments] has no CPU implementation.");
}

std::vector<mx::Shape> KQuantGatherQMMSorted::output_shapes(
    const std::vector<mx::array>& inputs) {
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  return {mx::Shape{x.shape(0), w.shape(-2)}};
}

bool KQuantGatherQMMSorted::is_equivalent(const mx::Primitive& other) const {
  const auto& o = static_cast<const KQuantGatherQMMSorted&>(other);
  return kquant_type_ == o.kquant_type_ && group_size_ == o.group_size_ &&
      bits_ == o.bits_;
}

#ifdef _METAL_

void KQuantGatherQMMSorted::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  // inputs: x (float16/bfloat16/float32, row-contiguous [S, K]), w (uint8
  // [E, N, bytes_per_row]), scales (vestigial placeholder), sorted_ids
  // (uint32 [S], ascending). Grid sizing needs only E (from w's shape) and N;
  // each threadgroup finds its row range in-kernel, so the id values are
  // never read on the host.
  const auto& x = inputs[0];
  const auto& w = inputs[1];

  int S = x.shape(0);
  int K = x.shape(1);
  int E = w.shape(0);
  int N = w.shape(1);

  const std::string& tile = kq_seg_tile();
  int BN = kq_seg_tile_bn(tile);
  constexpr int WM = 2, WN = 2;
  constexpr int TG_THREADS = WM * WN * 32;

  // Grid: one threadgroup per (n-tile, expert). Threadgroups whose expert has
  // no rows exit after the binary search.
  MTL::Size group_dims(TG_THREADS, 1, 1);
  MTL::Size grid_dims((N + BN - 1) / BN, E, 1);

  std::string type_string = kq_type_string(x.dtype());
  std::string kname = kq_kname_prefix(kquant_type_) + "gather_qmm_sorted_" +
      type_string + "_" + tile;

  auto kernel = kq_get_kernel(d, kname);
  auto& ce = mx::metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);

  int c = 0;
  ce.set_input_array(w, c++);
  ce.set_input_array(x, c++);
  ce.set_input_array(inputs[3], c++); // sorted_ids
  ce.set_output_array(out, c++);
  ce.set_bytes(K, c++);
  ce.set_bytes(N, c++);
  ce.set_bytes(S, c++);
  ce.dispatch_threadgroups(grid_dims, group_dims);
}

#else // !_METAL_

void KQuantGatherQMMSorted::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmm_sorted] requires a Metal build.");
}

#endif

void KQuantGatherQMMSorted::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error(
      "[mlx_kquant.gather_qmm_sorted] has no CPU implementation.");
}

} // namespace mlx_kquant
