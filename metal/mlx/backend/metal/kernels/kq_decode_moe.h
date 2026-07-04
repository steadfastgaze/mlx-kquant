// Decode-shaped routed-MoE matvec kernels: one token, a handful of routed
// experts addressed by expert id. Two kernels port the DS4-c decode contract:
//
//   * kq_iq2_xxs_gather_qmv_pair_swiglu: per-expert matvec over a combined
//     [2 * gate_out, K] gate/up weight stack with the SwiGLU applied to the
//     float32 accumulators and the per-expert route weight baked into the
//     stored intermediate. Grid (1, ceil(gate_out / 4), B): tid.z selects the
//     expert slot (its id from `ids`, its route weight from `route_weights`)
//     and each simdgroup accumulates one gate row and its paired up row
//     (gate_out rows apart) per result, so weight traffic equals the plain
//     per-expert qmv over the same stack. The dot restructures
//     kq_iq2_xxs_qmv_impl's for the matvec rate: each thread owns a whole
//     32-weight sub-block (headers load once per row dot instead of once per
//     8-weight group) and the grid/sign decode vectorizes through
//     uchar4 -> float4 conversions; the per-element arithmetic stays the
//     float32 multiply-add chain. At the DS4 decode shape (K=4096,
//     gate_out=2048, six experts) the sub-block mapping measures ~1.3x the
//     l-group mapping (0.152 versus 0.196 ms pipelined).
//
//   * kq_q2_k_gather_qmv_expert_sum: down matvec that accumulates the routed
//     experts inside the kernel. x is [B, K], row b holding expert slot b's
//     (already route-weighted) activation; each simdgroup keeps one float32
//     accumulator per output row and loops the B experts around
//     kq_q2_k_qmv_fast_impl's super-block walk, so the per-expert dot math is
//     unchanged and the cross-expert sum happens on the float32 accumulators
//     (expert-major order). The separate weighted-sum reduction disappears.
//
// The weighted sum over experts therefore moves inside the kernels: the pair
// epilogue scales by the route weight before the store and the down kernel
// sums the experts in float32. Against the unfused composition (gather_qmv
// per projection, elementwise SwiGLU, route-weighted sum after the down
// matvec) the result is numerically equivalent but not bit-identical: the
// epilogue reads float32 accumulators the composition would round through the
// I/O dtype, and the cross-expert sum accumulates per output element instead
// of as a separate elementwise reduction.
//
// Both kernels require K to be whole super-blocks and the output width to be
// a multiple of 4 (2 simdgroups x 2 results, the qmv row-block); the host op
// validates both. Sub-block tables and helpers come from the kq_quantized
// headers; derived-code attribution lives there and in mlx_kquant/licenses/.

// clang-format off

// Fused gate/up + SwiGLU + route-weight matvec over one expert's combined
// [2 * gate_out, K] iq2_xxs stack. One threadgroup computes 4 output elements
// (2 simdgroups x 2 results); each result accumulates the gate row (out_row)
// and the paired up row (gate_out + out_row) against the shared x fragment,
// so x loads amortize over both halves. Epilogue on the float32 sums: with
// limit c > 0 the gate clamps from above only and up symmetrically (the jang
// _dsv4_swiglu contract, matching KqSegDevAPairMMA::apply_swiglu), then
// silu(gate) * up * route_weight, cast to T at store.
template <typename T, int group_size, int bits>
[[kernel]] void kq_iq2_xxs_gather_qmv_pair_swiglu(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    const device uint32_t* ids,
    const device float* route_weights,
    device T* y,
    const constant int& in_vec_size,
    const constant int& gate_out,
    const constant float& swiglu_limit,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_IQ2_XXS_SUPERBLOCK, "IQ2_XXS requires gs=256");
  static_assert(bits == 2, "IQ2_XXS requires bits=2");
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;
  typedef float U;

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= gate_out) {
    return;
  }
  const int active_rows = min(results_per_simdgroup, gate_out - out_row);
  const int row_bytes =
      in_vec_size * KQ_IQ2_XXS_BLOCK_BYTES / KQ_IQ2_XXS_SUPERBLOCK;
  const int nb = in_vec_size / KQ_IQ2_XXS_SUPERBLOCK;

  // Expert slot: id addresses the weight stack, the route weight scales the
  // epilogue, and the slot owns output row block tid.z of y ([B, gate_out]).
  const int slot = tid.z;
  const device uint8_t* w_expert = w +
      static_cast<int64_t>(ids[slot]) * 2 * gate_out * row_bytes;
  const float route_weight = route_weights[slot];
  y += static_cast<int64_t>(slot) * gate_out;

  // Thread mapping: each thread owns one whole 32-weight sub-block (its
  // super-block header, sign word, and four grid indices load once per row
  // dot instead of once per 8-weight l-group), and the simdgroup strides
  // four super-blocks per iteration. The per-element arithmetic is
  // kq_iq2_xxs_qmv_impl's float32 chain, vectorized: the grid uint64
  // converts through uchar4 -> float4 and the sign mask through a bool4
  // select, accumulating with float4 dots.
  const int ix = simd_lid / 8; // super-block offset within the stride
  const int is = simd_lid % 8; // sub-block within the super-block
  U result_gate[results_per_simdgroup] = {0};
  U result_up[results_per_simdgroup] = {0};
  for (int ib = ix; ib < nb; ib += sb_stride) {
    // This thread's 32 x values as eight float4 fragments.
    float4 xt[8];
    {
      const device T* xp = x + ib * KQ_IQ2_XXS_SUPERBLOCK + is * 32;
#pragma unroll
      for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          xt[i][j] = float(xp[4 * i + j]);
        }
      }
    }
    for (int row = 0; row < active_rows; row++) {
      // Gate row (hf == 0), then the paired up row gate_out rows below
      // (hf == 1).
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const device uint8_t* sb = w_expert +
            static_cast<int64_t>(hf * gate_out + out_row + row) * row_bytes +
            ib * KQ_IQ2_XXS_BLOCK_BYTES;
        const U d = U(float(*(const device half*)sb));
        // The 8-byte qs region (four grid indices, then the packed sign
        // word) as four uint16 loads; sb is 2-byte aligned (66-byte blocks
        // over an even row stride).
        const device uint16_t* qs16 = reinterpret_cast<const device uint16_t*>(
            sb + KQ_IQ2_XXS_QS_OFFSET + is * 8);
        const uint qidx = uint(qs16[0]) | (uint(qs16[1]) << 16);
        const uint signbits = uint(qs16[2]) | (uint(qs16[3]) << 16);
        const U db = d * (U(0.5f) + U(signbits >> 28)) * U(0.25f);
        U partial = 0;
#pragma unroll
        for (int l = 0; l < 4; l++) {
          const uint8_t signs = ksigns_iq2xs[(signbits >> (7 * l)) & 127];
          const uint64_t g = iq2xxs_grid[(qidx >> (8 * l)) & 0xff];
          const float4 v0 =
              float4(as_type<uchar4>(uint32_t(g & 0xffffffffull)));
          const float4 v1 = float4(as_type<uchar4>(uint32_t(g >> 32)));
          const float4 s0 = select(
              float4(1.0f), float4(-1.0f),
              bool4(uchar4(signs) & uchar4(1, 2, 4, 8)));
          const float4 s1 = select(
              float4(1.0f), float4(-1.0f),
              bool4(uchar4(signs) & uchar4(16, 32, 64, 128)));
          partial += dot(xt[2 * l], v0 * s0) + dot(xt[2 * l + 1], v1 * s1);
        }
        if (hf == 0) {
          result_gate[row] += db * partial;
        } else {
          result_up[row] += db * partial;
        }
      }
    }
  }
  for (int row = 0; row < results_per_simdgroup; row++) {
    U gate = simd_sum(result_gate[row]);
    U up = simd_sum(result_up[row]);
    if (simd_lid == 0 && row < active_rows) {
      if (swiglu_limit > 0.0f) {
        gate = metal::min(gate, U(swiglu_limit));
        up = metal::clamp(up, U(-swiglu_limit), U(swiglu_limit));
      }
      const U act = (gate / (U(1) + metal::exp(-gate))) * up;
      y[out_row + row] = static_cast<T>(act * U(route_weight));
    }
  }
}

// Down matvec with the sum over the token's routed experts inside the kernel:
// y[n] = sum_b x[b] . dequant(w[ids[b]])[n]. Grid (1, ceil(N / 4), 1); each
// simdgroup keeps 2 float32 output accumulators and loops the B experts
// around kq_q2_k_qmv_fast_impl's super-block walk (per-expert dot math
// unchanged; experts accumulate in slot order). x rows carry the route
// weights already (the pair kernel bakes them), so the sum is unweighted.
template <typename T, int group_size, int bits>
[[kernel]] void kq_q2_k_gather_qmv_expert_sum(
    const device uint8_t* w,
    const device uint8_t* /* scales */,
    const device T* x,
    const device uint32_t* ids,
    device T* y,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    const constant int& n_experts_per_token,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(
      group_size == KQ_Q2_K_SUPERBLOCK, "Q2_K kernel requires group_size=256");
  static_assert(bits == 2, "Q2_K kernel requires bits=2");

  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 2;
  constexpr int sb_stride = 4;

  typedef float U;
  thread U yl[32];
  thread U result[results_per_simdgroup] = {0};

  const int ix = simd_lid / 8;
  const int it = simd_lid % 8;
  const int iq = it / 4;
  const int ir = it % 4;
  const int is = (8 * ir) / 16;

  const int row_bytes = in_vec_size * KQ_Q2_K_BLOCK_BYTES / KQ_Q2_K_SUPERBLOCK;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }

  const int nb = in_vec_size / KQ_Q2_K_SUPERBLOCK;

  for (int b = 0; b < n_experts_per_token; b++) {
    const device T* xb = x + static_cast<int64_t>(b) * in_vec_size;
    const device uint8_t* w_expert = w +
        static_cast<int64_t>(ids[b]) * out_vec_size * row_bytes;

    for (int ib = ix; ib < nb; ib += sb_stride) {
      const int x_base = ib * KQ_Q2_K_SUPERBLOCK + 128 * iq + 8 * ir;
      U sumy[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
      for (int i = 0; i < 8; i++) {
        yl[i + 0] = U(xb[x_base + i + 0]);
        sumy[0] += yl[i + 0];
        yl[i + 8] = U(xb[x_base + i + 32]);
        sumy[1] += yl[i + 8];
        yl[i + 16] = U(xb[x_base + i + 64]);
        sumy[2] += yl[i + 16];
        yl[i + 24] = U(xb[x_base + i + 96]);
        sumy[3] += yl[i + 24];
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        const int row_idx = out_row + row;
        const device uint8_t* sb_addr = w_expert +
            static_cast<int64_t>(row_idx) * row_bytes +
            ib * KQ_Q2_K_BLOCK_BYTES;

        const device uint8_t* sc = kq_q2_k_scales_ptr(sb_addr) + 8 * iq + is;
        const device uint16_t* qs =
            reinterpret_cast<const device uint16_t*>(kq_q2_k_qs_ptr(sb_addr)) +
            16 * iq + 4 * ir;

        U acc1[4] = {U(0), U(0), U(0), U(0)};
        U acc2[4] = {U(0), U(0), U(0), U(0)};
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const uint16_t qs_i = qs[i / 2];
          acc1[0] += yl[i + 0] * U(qs_i & 0x0003);
          acc2[0] += yl[i + 1] * U(qs_i & 0x0300);
          acc1[1] += yl[i + 8] * U(qs_i & 0x000c);
          acc2[1] += yl[i + 9] * U(qs_i & 0x0c00);
          acc1[2] += yl[i + 16] * U(qs_i & 0x0030);
          acc2[2] += yl[i + 17] * U(qs_i & 0x3000);
          acc1[3] += yl[i + 24] * U(qs_i & 0x00c0);
          acc2[3] += yl[i + 25] * U(qs_i & 0xc000);
        }

        const U d = U(kq_q2_k_d(sb_addr));
        const U dmin = U(kq_q2_k_dmin(sb_addr));
        result[row] += d *
                ((acc1[0] + acc2[0] * (U(1) / U(256))) * U(sc[0] & 0x0F) +
                 (acc1[1] + acc2[1] * (U(1) / U(256))) * U(sc[2] & 0x0F) *
                     (U(1) / U(4)) +
                 (acc1[2] + acc2[2] * (U(1) / U(256))) * U(sc[4] & 0x0F) *
                     (U(1) / U(16)) +
                 (acc1[3] + acc2[3] * (U(1) / U(256))) * U(sc[6] & 0x0F) *
                     (U(1) / U(64))) -
            dmin * (U(1) / U(16)) *
                (sumy[0] * U(sc[0] & 0xF0) + sumy[1] * U(sc[2] & 0xF0) +
                 sumy[2] * U(sc[4] & 0xF0) + sumy[3] * U(sc[6] & 0xF0));
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[out_row + row] = static_cast<T>(result[row]);
    }
  }
}
// clang-format on
