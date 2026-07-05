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

  // Tile geometry must match a kernel instantiation in kq_segments.metal
  // (WM=WN=2 -> WM*WN*32 = 128 threads per threadgroup). The tag is
  // t{BM}x{BN}x{BK} with an optional trailing 'h' for the half-staging opt-in.
  // The default 48x64x16 (float staging, f32 decode) keeps the f32-decode
  // contract while shrinking the two staged float tiles enough to lift
  // occupancy. The staged footprint is two float [rows, BK_padded] tiles; at
  // BK=16 (the minimum, since the weight decode helper works on 16-weight
  // chunks) each tile is roughly a third of the 64x64x32 size, so several
  // threadgroups fit per SM instead of one. On the sorted MoE bulk-prefill shape
  // (S=23058, K=N=4096, E=256, uniform segments) this measures 106 ms/layer for
  // iq2_xxs and 108 ms for q2_k, against 169 ms for the old 64x64x32 float tile
  // and matching the 116 ms half-staging occupancy without rounding weights to
  // f16. BM=48 covers a 90-row segment in two row-chunks with a smaller
  // activation tile than BM=64 (which measured 134 ms) and less redundant weight
  // decode than BM=32 (109 ms). Other tiles stay as KQ_SEG_TILE A/B levers,
  // including the half-staging opt-in for callers whose quality gate allows
  // f16-precision weight decode. Only BN (the n-tile width) reaches the host - it
  // sizes the grid; BM/BK are internal to the kernel.
  static const std::string tile = []() {
    const char* e = std::getenv("KQ_SEG_TILE");
    return std::string(e != nullptr ? e : "t48x64x16");
  }();
  // Parse BN (the middle field of t{BM}x{BN}x{BK}) so the grid width matches
  // whichever tile the name selects.
  int BN = 64;
  {
    const auto first = tile.find('x');
    const auto second =
        first == std::string::npos ? first : tile.find('x', first + 1);
    if (first != std::string::npos && second != std::string::npos) {
      BN = std::stoi(tile.substr(first + 1, second - first - 1));
    }
  }
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

} // namespace mlx_kquant
