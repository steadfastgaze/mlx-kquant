#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "kquant.h"
#include "kquant_codec.h"
#include "kquant_cpu_neon.h"
#include "kquant_gguf.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Convert a decoded GGUF metadata value to a Python object: scalars to
// int/float/bool/str, arrays to lists, monostate to None.
nb::object meta_to_py(const mlx_kquant::GgufMetaValue& v) {
  return std::visit(
      [](auto&& x) -> nb::object {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return nb::none();
        } else {
          return nb::cast(x);
        }
      },
      v);
}

} // namespace

NB_MODULE(_ext, m) {
  m.doc() =
      "mlx-kquant: standalone GGUF K-quant ops for MLX (custom Metal kernels).";

  // --- toolchain self-checks ---
  m.def(
      "codecs",
      &mlx_kquant::codec_names,
      "Return the list of supported K-quant codec names.");

  m.def(
      "metallib_dir",
      &mlx_kquant::metallib_dir,
      "Directory holding the bundled mlx_kquant.metallib.");

  m.def(
      "metallib_loads",
      &mlx_kquant::metallib_loads,
      "Load the bundled metallib via the Metal device (toolchain self-check).");

  m.def(
      "sdpa_q8_loader_debug",
      [] {
        const auto counts = mlx_kquant::sdpa_q8_loader_dispatch_counts();
        nb::dict state;
        state[nb::str("selected_arm")] =
            nb::str(mlx_kquant::sdpa_q8_loader_arm().c_str());
        state[nb::str("scalar_dynamic")] = nb::int_(counts.at(0));
        state[nb::str("uint4_dynamic")] = nb::int_(counts.at(1));
        state[nb::str("uint4_byte_dynamic")] = nb::int_(counts.at(2));
        state[nb::str("off_contract")] = nb::int_(counts.at(3));
        return state;
      },
      "Report the selected q8 loader arm and its process dispatch counters.");

  m.def(
      "cpu_neon_available",
      &mlx_kquant::kq_cpu_neon_available,
      "True when the arm64 NEON int8 CPU GEMV kernels can run here (arm64 "
      "build with the dotprod extension, not disabled via KQ_CPU_NEON=0).");

  // --- ops ---
  m.def(
      "dequantize",
      &mlx_kquant::dequantize,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Dequantize GGUF K-quant wire bytes to a float array.

        Args:
            w (array): uint8 wire bytes; last dim a multiple of the codec's
                bytes_per_block.
            scales (array): vestigial placeholder (K-quant scales live inside
                ``w``); ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"q4_k"``, ``"q8_0"``.
            dtype (Dtype, optional): output float dtype. Default ``float16``.

        Returns:
            array: the dequantized weights.
      )");

  m.def(
      "quantized_matmul",
      &mlx_kquant::quantized_matmul,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "transpose"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Quantized matmul: ``x @ dequant(w)`` for GGUF K-quant weights.

        Args:
            x (array): float activations.
            w (array): uint8 K-quant wire bytes (laid out [N, K] when
                transpose=True).
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"q4_k"``.
            transpose (bool): whether ``w`` is transposed ([N, K]). Default True.

        Returns:
            array: the matmul result (x.dtype, float32 promoted to bfloat16).
      )");

  m.def(
      "quantized_matmul_qmv_bias",
      &mlx_kquant::quantized_matmul_qmv_bias,
      "x"_a,
      "w"_a,
      "scales"_a,
      "bias"_a,
      "kquant_type"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Bias-fused quantized matmul: ``x @ dequant(w) + bias`` for GGUF
        K-quant weights, fusing the add into the matmul kernel dispatch.

        Decode-only: ``x`` must carry exactly one row (``x.shape[-2] == 1``
        after flattening leading batch dims) -- raises otherwise. Only
        ``kquant_type="q8_0"`` is wired so far. ``transpose`` is always True
        (the only regime this is used for). For any other shape or codec, use
        ``quantized_matmul`` followed by a separate ``+ bias``.

        Args:
            x (array): float activations, exactly one row.
            w (array): uint8 K-quant wire bytes, laid out [N, K].
            scales (array): vestigial placeholder; ignored by the kernel.
            bias (array): 1D, length N (the output dim).
            kquant_type (str): codec name; only ``"q8_0"`` is wired so far.

        Returns:
            array: the matmul-plus-bias result (x.dtype, float32 promoted to
            bfloat16).
      )");

  m.def(
      "sdpa_vector",
      &mlx_kquant::sdpa_vector,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a = true,
      "mask"_a = nb::none(),
      "sinks"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Vector scaled-dot-product attention for large head dims (256, 512) that
        stock MLX's fused vector allowlist excludes.

        Args:
            q (array): queries [B, n_q_heads, qL, D], float16/bfloat16.
            k (array): keys [B, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), D dim must be contiguous.
            v (array): values [B, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            causal (bool): apply an offset causal mask. Default True.
            mask (array, optional): boolean key mask broadcastable to
                [B, n_q_heads, qL, kL]; true means attend.
            sinks (array, optional): per-query-head attention-sink logit
                [n_q_heads] joining the softmax denominator with no value row.

        Returns:
            array: attention output [B, n_q_heads, qL, D].
      )");

  m.def(
      "sdpa_decode_gqa",
      &mlx_kquant::sdpa_decode_gqa,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "sinks"_a = nb::none(),
      "splits"_a = 0,
      "tile_c"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode/verify GQA attention tuned for long KV caches: the key axis
        is split into a fixed number of coarse contiguous chunks and each
        chunk is streamed through threadgroup-staged K/V tiles shared by the
        whole GQA group, so device memory reads the KV once per kv-head. At
        qL 2..4 (speculative-verify width) every query also shares the staged
        tiles, causally clamped to its own trailing position.

        Args:
            q (array): queries [B, n_q_heads, qL, D], float16/bfloat16;
                qL in 1..4, D in {64, 128, 256, 512}.
            k (array): keys [B, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), the head_dim must be contiguous.
            v (array): values [B, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            sinks (array, optional): per-q-head attention sinks, shape
                [n_q_heads] -- an extra softmax logit with no value row.
            splits (int): key-axis split count; 0 picks the default.
            tile_c (int): staged tile height, 8/16/32; 0 (default) picks by
                head_dim (32 up to D=128, 16 at D=256, 8 at D=512).

        Returns:
            array: attention output [B, n_q_heads, qL, D].
      )");

  m.def(
      "sdpa_fa_verify",
      &mlx_kquant::sdpa_fa_verify,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "q_len"_a,
      "splits"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Speculative-verify attention on the GPU matrix units for a GQA-folded
        query tile. Fold the GQA group into the query rows first --
        q [1, Hq, q_len, D] reshaped to [1, Hkv, G*q_len, D] with kv-major
        heads -- and pass the original q_len: folded row r is causally
        clamped to key <= kL - q_len + (r % q_len). The 32-row query tile
        streams each contiguous KV split once, computing S = Q @ K^T and
        O += P @ V on simdgroup_matrix with float32 accumulators and a
        per-row online softmax; per-split partials are merged by the same
        reduction pass as ``sdpa_decode_gqa``.

        Args:
            q (array): folded queries [1, n_kv_heads, G*q_len, D],
                float16/bfloat16; G*q_len <= 32, D = 256.
            k (array): keys [1, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), the head_dim must be contiguous.
            v (array): values [1, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            q_len (int): pre-fold query length (2..8); sets each folded
                row's causal clamp.
            splits (int): key-axis split count; 0 picks the default.

        Returns:
            array: attention output [1, n_kv_heads, G*q_len, D].
      )");

  m.def(
      "sdpa_fa_prefill",
      &mlx_kquant::sdpa_fa_prefill,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "qw"_a = 0,
      "splits"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Flash-style prefill attention on the GPU matrix units for a D=256 GQA
        chunk, dense KV. Queries stay in [1, Hq, qL, D]; the kernel tiles the
        query axis on the grid (each tile a fold of the whole GQA group with
        ``qw`` consecutive query positions, G * qw == 32) and streams each
        contiguous KV split once, computing S = Q @ K^T and O += P @ V on
        simdgroup_matrix with float32 accumulators and a per-row online softmax
        (no score tensor). Causal prefill mask: query position p attends
        keys <= (kL - qL) + p, so the past prefix is unmasked and the self
        chunk causal. Per-split partials merge through the same reduction pass
        as ``sdpa_decode_gqa``.

        Args:
            q (array): queries [1, n_q_heads, qL, D], float32/float16/bfloat16;
                D = 256.
            k (array): keys [1, n_kv_heads, kL, D]; head/seq strided is fine
                (read in place), the head_dim must be contiguous.
            v (array): values [1, n_kv_heads, kL, D].
            scale (float): query scale (typically 1/sqrt(D)).
            qw (int): query positions folded per 32-row tile; 0 picks the
                default 32 / (n_q_heads / n_kv_heads).
            splits (int): key-axis split count; 0 picks the depth default.

        Returns:
            array: attention output [1, n_q_heads, qL, D].
      )");

  m.def(
      "sdpa_fa_prefill_q8",
      &mlx_kquant::sdpa_fa_prefill_q8,
      "q"_a,
      "pk_w"_a,
      "pk_s"_a,
      "pk_b"_a,
      "pv_w"_a,
      "pv_s"_a,
      "pv_b"_a,
      "self_k"_a,
      "self_v"_a,
      "scale"_a,
      "group_size"_a = 64,
      "bits"_a = 8,
      "qw"_a = 0,
      "bq"_a = 0,
      "bk"_a = 0,
      "splits"_a = 0,
      "stage"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        The serving form of ``sdpa_fa_prefill``: float32 queries and
        accumulators, the past prefix read directly from a QuantizedKVCache
        tuple (packed uint32, float32 scales and biases, group 64, 8 bits) and
        the fresh self-chunk keys and values read dense. Past tiles dequantize
        with fma(scale, q, bias), the exact mx dequantization form, into the
        staging tiles during the cooperative load; self tiles cast from
        float32. Query position p attends keys <= past_len + p on the combined
        key axis, so the past prefix is unmasked and the self chunk causal.
        No score tensor reaches device memory.

        Args:
            q (array): queries [1, n_q_heads, qL, 256], float32.
            pk_w, pk_s, pk_b (arrays): past K quantized tuple
                [1, n_kv_heads, past_len, 64|4|4] (uint32, float32, float32).
            pv_w, pv_s, pv_b (arrays): past V quantized tuple, same geometry.
            self_k (array): fresh dense keys [1, n_kv_heads, qL, 256]; cast to
                float32 if needed.
            self_v (array): fresh dense values, same shape and treatment.
            scale (float): query scale (typically 1/sqrt(D)).
            group_size (int): must be 64.
            bits (int): must be 8.
            qw (int): query positions folded per tile; 0 picks the default
                bq / (n_q_heads / n_kv_heads).
            bq (int): query rows per threadgroup, 32 or 64; 0 picks 32. 64
                folds twice the query positions onto each staged KV tile.
            bk (int): keys staged per tile; 0 picks the stage default (half
                staging 32, float32 staging 16). 48 is instantiated for half
                staging at bq 64.
            splits (int): key-axis split count; 0 picks the depth default.
            stage (int): staging precision: 0 bfloat16, 1 float16, 2 float32
                (half tile width; the verification and memory-lever form).

        Returns:
            array: attention output [1, n_q_heads, qL, 256], float32.
      )");

  m.def(
      "sdpa_decode_q8",
      &mlx_kquant::sdpa_decode_q8,
      "q"_a,
      "pk_w"_a,
      "pk_s"_a,
      "pk_b"_a,
      "pv_w"_a,
      "pv_s"_a,
      "pv_b"_a,
      "scale"_a,
      "group_size"_a = 64,
      "bits"_a = 8,
      "splits"_a = 0,
      "stage"_a = 2,
      "compute"_a = 1,
      "tile_c"_a = 0,
      nb::kw_only(),
      "dimension_parallel_merge"_a = false,
      "stream"_a = nb::none(),
      R"(
        Fused q8 decode attention: the KV-attention read for one query row
        (qL == 1) over a whole QuantizedKVCache tuple (packed uint32, float32
        scales and biases, group 64, 8 bits). Float32 queries and accumulators.
        The whole GQA group folds into one query tile; the key axis splits into
        contiguous chunks each streamed once through threadgroup staging
        dequantized with fma(scale, q, bias) (the exact mx form), with a
        per-split online softmax and no score tensor; the partials merge through
        the same reduction pass as ``sdpa_decode_gqa``. A decode query attends
        every key (no causal cut inside the past).

        Args:
            q (array): queries [1, n_q_heads, 1, 256], float32.
            pk_w, pk_s, pk_b (arrays): K quantized tuple
                [1, n_kv_heads, N, 64|4|4] (uint32, float32, float32).
            pv_w, pv_s, pv_b (arrays): V quantized tuple, same geometry.
            scale (float): query scale (typically 1/sqrt(D)).
            group_size (int): must be 64.
            bits (int): must be 8.
            splits (int): key-axis split count; 0 picks the depth default.
            stage (int): matrix-tile staging precision: 0 bfloat16, 1 float16,
                2 float32 (used by compute 0 only).
            compute (int): compute pattern. 1 (default) is the SIMD-shuffle
                reduction, the decode-latency form; 0 is the matrix-unit tile.
            tile_c (int): SIMD-shuffle staged tile height, 8 or 16; 0 the
                default.
            dimension_parallel_merge (bool): use the exact-order,
                dimension-parallel split merge when the fixed 16:2,
                split-128, tile-16 serving geometry is present at a cache
                depth of at least 8192. Other geometries retain the shared
                merge. Defaults to false.

        Returns:
            array: attention output [1, n_q_heads, 1, 256], float32.
      )");

  m.def(
      "moe_glu_gather",
      &mlx_kquant::moe_glu_gather,
      "x"_a,
      "gate_w"_a,
      "gate_scales"_a,
      "gate_bias"_a,
      "up_w"_a,
      "up_scales"_a,
      "up_bias"_a,
      "indices"_a,
      "alpha"_a = 1.702f,
      "limit"_a = 7.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused MoE GLU gather on the MLX packed mxfp4 layout: gate and up
        expert matvecs (sharing each activation load), expert biases, and the
        clamped-SwiGLU epilogue
        ``(min(g, limit) * sigmoid(alpha * g)) * (clip(u, -limit, limit) + 1)``
        in one dispatch. Decode-shaped: one activation row per token, shared
        across that token's expert slots.

        Args:
            x (array): activations [T, K], float16/bfloat16.
            gate_w (array): packed gate weights uint32 [E, N, K/8].
            gate_scales (array): E8M0 group scales uint8 [E, N, K/32].
            gate_bias (array): gate biases [E, N].
            up_w / up_scales / up_bias: same layout for the up projection.
            indices (array): expert indices [T, R].
            alpha (float): sigmoid slope. Default 1.702.
            limit (float): activation clamp. Default 7.0.

        Returns:
            array: activated hidden states [T, R, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_bias",
      &mlx_kquant::gather_qmv_bias,
      "x"_a,
      "w"_a,
      "scales"_a,
      "bias"_a,
      "indices"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered matvec with the expert bias fused, on the MLX packed mxfp4
        layout (see moe_glu_gather). One activation row per expert slot.

        Args:
            x (array): activations [T, R, K], float16/bfloat16.
            w (array): packed weights uint32 [E, N, K/8].
            scales (array): E8M0 group scales uint8 [E, N, K/32].
            bias (array): biases [E, N].
            indices (array): expert indices [T, R].

        Returns:
            array: output [T, R, N] in x.dtype.
      )");

  m.def(
      "moe_glu_gather_kq",
      &mlx_kquant::moe_glu_gather_kq,
      "x"_a,
      "gate_w"_a,
      "up_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "act"_a = "silu",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused MoE GLU gather for K-quant expert stacks: gate and up expert
        matvecs share each activation load and the GLU epilogue act(g) * u is
        applied in the same dispatch. No biases. Decode-shaped.

        Args:
            x (array): activations [T, K], float16/bfloat16. K % 256 == 0.
            gate_w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            up_w (array): uint8 wire bytes, same shape as gate_w.
            kquant_type (str): codec with a fused kernel (full GGUF matrix).
            indices (array): expert indices [T, R].
            act (str): 'silu' (default) or 'gelu' (tanh approx).

        Returns:
            array: activated hidden states [T, R, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_kq",
      &mlx_kquant::gather_qmv_kq,
      "x"_a,
      "w"_a,
      "kquant_type"_a,
      "indices"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered matvec for K-quant expert stacks (the MoE down projection).
        One activation row per expert slot.

        Args:
            x (array): activations [T, R, K], float16/bfloat16. K % 256 == 0.
            w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            kquant_type (str): codec with a fused kernel (full GGUF matrix).
            indices (array): expert indices [T, R].

        Returns:
            array: output [T, R, N] in x.dtype.
      )");

  m.def(
      "moe_glu_gather_shexp_kq",
      &mlx_kquant::moe_glu_gather_shexp_kq,
      "x"_a,
      "gate_w"_a,
      "up_w"_a,
      "shexp_gate_w"_a,
      "shexp_up_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "act"_a = "silu",
      "shexp_kquant_type"_a = "",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        moe_glu_gather_kq with the block's shared expert folded in as one
        extra slot (the last), fed by single-expert 2-D wire-byte tensors
        row-shape-matched to the expert stack.

        Args:
            x (array): activations [T, K], float16/bfloat16. K % 256 == 0.
            gate_w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            up_w (array): uint8 wire bytes, same shape as gate_w.
            shexp_gate_w (array): uint8 wire bytes (N, bytes_per_row).
            shexp_up_w (array): uint8 wire bytes (N, bytes_per_row).
            kquant_type (str): expert codec with a fused kernel.
            indices (array): expert indices [T, R].
            act (str): 'silu' (default) or 'gelu' (tanh approx).
            shexp_kquant_type (str): shared-expert codec; '' (default) =
                kquant_type. Mixed combos must be q6_k or q8_0.

        Returns:
            array: activated hidden states [T, R + 1, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_mix_kq",
      &mlx_kquant::gather_qmv_mix_kq,
      "x"_a,
      "w"_a,
      "shexp_w"_a,
      "kquant_type"_a,
      "indices"_a,
      "scores"_a,
      "shexp_kquant_type"_a = "",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered down projection with the routing mix folded in: every slot
        (the last being the shared expert) is accumulated in f32 weighted by
        its score, replacing gather + (y * scores).sum + shared add.

        Args:
            x (array): activations [T, S, K], float16/bfloat16. K % 256 == 0.
            w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            shexp_w (array): uint8 wire bytes (N, bytes_per_row).
            kquant_type (str): expert codec with a fused kernel.
            indices (array): expert indices [T, S - 1].
            scores (array): mix weights [T, S]; cast to float32.
            shexp_kquant_type (str): shared-expert codec; '' (default) =
                kquant_type. Mixed combos must be q6_k or q8_0.

        Returns:
            array: mixed output [T, N] in x.dtype.
      )");

  m.def(
      "gather_qmv_mix_ns_kq",
      &mlx_kquant::gather_qmv_mix_ns_kq,
      "x"_a,
      "w"_a,
      "kquant_type"_a,
      "indices"_a,
      "scores"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gathered down projection with the routing mix folded in, no shared
        expert: each of the S routed slots is accumulated in f32 weighted by
        its score, replacing gather + (y * scores).sum.

        Args:
            x (array): activations [T, S, K], float16/bfloat16.
            w (array): uint8 wire bytes (n_experts, N, bytes_per_row).
            kquant_type (str): expert codec with a fused kernel.
            indices (array): expert indices [T, S].
            scores (array): mix weights [T, S]; cast to float32.

        Returns:
            array: mixed output [T, N] in x.dtype.
      )");

  m.def(
      "moe_router_topk",
      &mlx_kquant::moe_router_topk,
      "logits"_a,
      "top_k"_a,
      "norm_topk_prob"_a = true,
      "shared_gate"_a = true,
      "per_expert_scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Router top-k in one dispatch: f32 softmax over the first E columns,
        top_k selection (min-index tie-break), optional renormalization, an
        optional per-expert scale applied to the picked scores, and (when
        shared_gate) the sigmoid of column E (the shared-expert gate logit)
        in the last scores slot.

        Args:
            logits (array): router logits [T, E + shared_gate]; E <= 1024.
            top_k (int): experts per token, <= 16.
            norm_topk_prob (bool): renormalize picked probabilities.
            shared_gate (bool): logits carry a trailing shared-gate column.
            per_expert_scale (array, optional): [E] multiplier on picked
                scores, applied after renormalization; cast to float32.

        Returns:
            tuple: (indices [T, top_k] uint32,
            scores [T, top_k + shared_gate] float32).
      )");

  m.def(
      "add_rmsnorm",
      &mlx_kquant::add_rmsnorm,
      "h"_a,
      "residual"_a,
      "weight"_a,
      "eps"_a,
      "scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused post-norm residual: (residual + rms_norm(h, weight)) * scale
        in one dispatch, all math in f32.

        Args:
            h (array): [..., D], float16/bfloat16.
            residual (array): same shape and dtype as h.
            weight (array): [D] norm weight, same dtype as h.
            eps (float): rms_norm epsilon.
            scale (array, optional): size-1 epilogue scalar, same dtype as
                h; 1.0 when absent.

        Returns:
            array: same shape and dtype as h.
      )");

  m.def(
      "rmsnorm_multi3",
      &mlx_kquant::rmsnorm_multi3,
      "x"_a,
      "w0"_a,
      "w1"_a,
      "w2"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Three rms_norms of one tensor in one dispatch, sharing the
        mean-square reduction of x.

        Args:
            x (array): [..., D], float16/bfloat16.
            w0 (array): [D] norm weight, same dtype as x.
            w1 (array): [D] norm weight, same dtype as x.
            w2 (array): [D] norm weight, same dtype as x.
            eps (float): rms_norm epsilon.

        Returns:
            tuple: (rms_norm(x, w0), rms_norm(x, w1), rms_norm(x, w2)).
      )");

  m.def(
      "rmsnorm2_add",
      &mlx_kquant::rmsnorm2_add,
      "a"_a,
      "wa"_a,
      "b"_a,
      "wb"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused branch merge: rms_norm(a, wa) + rms_norm(b, wb) in one
        dispatch, all math in f32.

        Args:
            a (array): [..., D], float16/bfloat16.
            wa (array): [D] norm weight, same dtype as a.
            b (array): same shape and dtype as a.
            wb (array): [D] norm weight, same dtype as a.
            eps (float): rms_norm epsilon.

        Returns:
            array: same shape and dtype as a.
      )");

  m.def(
      "gather_qmm",
      &mlx_kquant::gather_qmm,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      "transpose"_a = true,
      "sorted_indices"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Gather (mixture-of-experts) quantized matmul for GGUF K-quant weights.

        Args:
            x (array): float activations, at least 2-D.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, out_dims, bytes_per_row).
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"q4_k"``.
            lhs_indices (array, optional): uint32 indices selecting x rows.
                Defaults to a plain arange.
            rhs_indices (array, optional): uint32 indices selecting expert
                weight matrices. Defaults to a plain arange.
            transpose (bool): whether each expert ``w`` is transposed
                ([out, in]). Default True.
            sorted_indices (bool): hint that the defaulted indices are sorted.

        Returns:
            array: the gathered matmul result (x.dtype, float32 -> bfloat16).
      )");

  m.def(
      "gather_qmm_segments",
      &mlx_kquant::gather_qmm_segments,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "segments"_a,
      "transpose"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Descriptor-driven segmented (mixture-of-experts) quantized GEMM for
        sorted token-expert pair rows.

        Args:
            x (array): float16/bfloat16 activations [S, K], row-contiguous. The
                sorted token-expert pair rows.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, N, bytes_per_row) - the transpose=True MoE layout,
                the same gather_qmm takes. N output features; K is derived from
                bytes_per_row via the codec.
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"iq2_xxs"``, ``"q2_k"``.
            segments (array): uint32 [T, 3] host-built descriptor table, rows
                (expert_index, row_start, row_count). Row ranges are disjoint,
                sorted, and their union covers [0, S). Every row in
                [row_start, row_start + row_count) multiplies against
                ``w[expert_index]``.
            transpose (bool): must be True (only the MoE transpose shape is
                supported).

        Returns:
            array: the segmented matmul result [S, N] (x.dtype). Weights are
            dequantized to float32 tiles and accumulated in float32, then cast
            to x.dtype.
      )");

  m.def(
      "gather_qmm_sorted",
      &mlx_kquant::gather_qmm_sorted,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "sorted_ids"_a,
      "transpose"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Sorted-ids segmented (mixture-of-experts) quantized GEMM with the row
        ranges derived in-kernel: gather_qmm_segments without the host-built
        descriptor table.

        Args:
            x (array): float16/bfloat16/float32 activations [S, K],
                row-contiguous. The sorted token-expert pair rows.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, N, bytes_per_row) - the transpose=True MoE layout,
                the same gather_qmm_segments takes.
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"iq2_xxs"``, ``"q2_k"``.
            sorted_ids (array): device uint32 [S] expert index per x row,
                ascending (equal ids contiguous), values in [0, n_experts).
                Each threadgroup binary-searches its expert's row range, so the
                ids are never read on the host and the call queues with zero
                synchronization.
            transpose (bool): must be True (only the MoE transpose shape is
                supported).

        Returns:
            array: the segmented matmul result [S, N] (x.dtype), bit-identical
            to gather_qmm_segments with the equivalent host-built table.
      )");

  m.def(
      "gather_qmm_sorted_swiglu",
      &mlx_kquant::gather_qmm_sorted_swiglu,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "sorted_ids"_a,
      "gate_out"_a,
      "swiglu_limit"_a,
      "transpose"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused gate/up + SwiGLU sorted-ids segmented (mixture-of-experts)
        quantized GEMM for combined gate/up weights: gather_qmm_sorted over a
        [2 * gate_out, K] expert stack with the SwiGLU applied in the kernel
        epilogue, so the [S, 2 * gate_out] intermediate is never materialized.

        For each row ``s`` with expert id ``sorted_ids[s]``:
        ``gate = x[s] @ dequant(w[e])[0:gate_out].T``, ``up = x[s] @
        dequant(w[e])[gate_out:].T``, and the output row is
        ``silu(gate) * up`` computed in float32, with ``swiglu_limit > 0``
        clamping ``gate`` from above only and ``up`` symmetrically first.

        Args:
            x (array): float16/bfloat16/float32 activations [S, K],
                row-contiguous. The sorted token-expert pair rows.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, 2 * gate_out, bytes_per_row); each expert stacks
                its gate rows first, then its up rows.
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name, e.g. ``"iq2_xxs"``, ``"q2_k"``.
            sorted_ids (array): device uint32 [S] expert index per x row,
                ascending (equal ids contiguous), values in [0, n_experts).
                Same contract as gather_qmm_sorted.
            gate_out (int): output features of the gate half; w must hold
                exactly ``2 * gate_out`` rows per expert.
            swiglu_limit (float): activation clamp. Positive values clamp the
                gate to at most ``swiglu_limit`` and the up value to
                ``[-swiglu_limit, swiglu_limit]`` before ``silu(gate) * up``;
                values <= 0 disable the clamps.
            transpose (bool): must be True (only the MoE transpose shape is
                supported).

        Returns:
            array: the fused result [S, gate_out] (x.dtype). The GEMMs and
            the activation compute in float32; against the unfused
            gather_qmm_sorted + activation composition the result is
            numerically equivalent but not bit-identical, because the fused
            epilogue skips the intermediate store in x.dtype.
      )");

  m.def(
      "gather_qmv_pair_swiglu",
      &mlx_kquant::gather_qmv_pair_swiglu,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "ids"_a,
      "route_weights"_a,
      "gate_out"_a,
      "swiglu_limit"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode-shaped fused gate/up + SwiGLU matvec over a combined gate/up
        mixture-of-experts weight stack, with the per-expert route weight
        baked into the stored intermediate. One dispatch computes every
        routed expert's activation row for a single token.

        For each expert slot ``b``: ``gate = x[0] @
        dequant(w[ids[b]])[0:gate_out].T``, ``up = x[0] @
        dequant(w[ids[b]])[gate_out:].T`` on float32 accumulators, and the
        output row is ``silu(gate) * up * route_weights[b]`` computed in
        float32 (``swiglu_limit > 0`` clamps the gate from above only and up
        symmetrically first, the DSV4 SwiGLU contract).

        Args:
            x (array): float16/bfloat16/float32 activation row [1, K],
                row-contiguous.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, 2 * gate_out, bytes_per_row); each expert stacks
                its gate rows first, then its up rows.
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name; only codecs with an instantiated
                pair kernel are accepted (``"iq2_xxs"``).
            ids (array): uint32 [B] expert index per routed slot, values in
                [0, n_experts).
            route_weights (array): float32 [B] route weight per slot, baked
                into the stored intermediate.
            gate_out (int): output features of the gate half; w must hold
                exactly ``2 * gate_out`` rows per expert, and gate_out must
                be a multiple of 4 (the qmv row block).
            swiglu_limit (float): activation clamp. Positive values clamp the
                gate to at most ``swiglu_limit`` and the up value to
                ``[-swiglu_limit, swiglu_limit]``; values <= 0 disable the
                clamps.

        Returns:
            array: the fused intermediate [B, gate_out] in x.dtype (float32
            stays float32; no bfloat16 promotion on this decode path).
            Numerically equivalent but not bit-identical to the unfused
            gather_qmm + activation + scale composition (the epilogue reads
            the float32 accumulators directly).
      )");

  m.def(
      "gather_qmv_expert_sum",
      &mlx_kquant::gather_qmv_expert_sum,
      "x"_a,
      "w"_a,
      "scales"_a,
      "kquant_type"_a,
      "ids"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Decode-shaped down matvec with the sum over the token's routed
        experts fused into the kernel: ``out[0, n] = sum_b x[b] @
        dequant(w[ids[b]])[n]``, each output element accumulating every
        expert's contribution in float32 in slot order. The separate
        route-weighted-sum reduction disappears; x rows are expected to
        carry the route weights already (gather_qmv_pair_swiglu bakes them).

        Args:
            x (array): float16/bfloat16/float32 activations [B, K],
                row-contiguous, one row per routed slot.
            w (array): uint8 K-quant wire bytes shaped
                (n_experts, N, bytes_per_row); N must be a multiple of 4
                (the qmv row block).
            scales (array): vestigial placeholder; ignored by the kernel.
            kquant_type (str): codec name; only codecs with an instantiated
                expert-sum kernel are accepted (``"q2_k"``).
            ids (array): uint32 [B] expert index per x row, values in
                [0, n_experts).

        Returns:
            array: the summed result [1, N] in x.dtype (float32 stays
            float32). Numerically equivalent but not bit-identical to the
            per-expert gather_qmm + sum composition (the cross-expert sum
            runs on the float32 accumulators inside the kernel).
      )");

  m.def(
      "quantize",
      &mlx_kquant::quantize,
      "w"_a,
      "kquant_type"_a,
      "imatrix"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Encode a float weight tensor into GGUF K-quant wire bytes (CPU or Metal).

        Args:
            w (array): float weights; last dim a multiple of the codec's
                weights_per_block.
            kquant_type (str): codec name, e.g. ``"q4_k"``, ``"q8_0"``.
            imatrix (array, optional): 1-D float32 importance vector of length
                K = ``w.shape[-1]`` to steer the encoder.

        Returns:
            tuple[array, array]: ``(wq, scales)`` where ``wq`` is the uint8 wire
            bytes and ``scales`` is a vestigial uint8 placeholder of shape [1]
            (K-quant scales live inside ``wq``).
      )");

  // --- GGUF loader ---
  m.def(
      "load_gguf",
      [](const std::string& path, bool zero_copy) {
        mlx_kquant::GgufLoadResult res = mlx_kquant::load_gguf(path, zero_copy);

        nb::dict arrays;
        for (auto& [name, arr] : res.arrays) {
          arrays[nb::str(name.c_str())] = nb::cast(arr);
        }
        nb::dict codecs;
        for (auto& [name, codec] : res.codecs) {
          codecs[nb::str(name.c_str())] = nb::str(codec.c_str());
        }
        nb::dict metadata;
        for (auto& [key, value] : res.metadata) {
          metadata[nb::str(key.c_str())] = meta_to_py(value);
        }
        nb::dict shapes;
        for (auto& [name, dims] : res.tensor_shapes) {
          shapes[nb::str(name.c_str())] = nb::cast(dims);
        }
        return nb::make_tuple(arrays, codecs, metadata, shapes);
      },
      "path"_a,
      "zero_copy"_a = true,
      R"(
        Load a GGUF file's tensors and metadata directly from gguflib's mmap.

        With ``zero_copy=True`` (default) each tensor array is a no-copy view
        over the mmap (a Metal newBufferWithBytesNoCopy per tensor, kept mapped
        until the last viewing array is freed) - no per-tensor allocation or
        byte-copy. With ``zero_copy=False`` every tensor is memcpy'd out of the
        mmap (~15 GB/s). Tensors that can't be wrapped no-copy fall back to the
        copy path transparently.

        Args:
            path (str): GGUF file path.
            zero_copy (bool): view the mmap instead of copying (default True).

        Returns:
            tuple[dict, dict, dict, dict]: ``(arrays, codecs, metadata, shapes)``:
              - arrays: tensor name -> mx.array. K-quant tensors are uint8 wire
                bytes (MLX axis order, byte-packed last dim) each with a 1-byte
                ``<prefix>.scales`` placeholder; F32/F16/BF16/I8/I16/I32 tensors
                keep their native dtype.
              - codecs: K-quant tensor name -> codec name ("q4_k", ...).
              - metadata: GGUF KV key -> decoded value (int/float/bool/str/list).
              - shapes: tensor name -> logical shape (GGUF native, innermost-first
                order; matches gguf-py ReaderTensor.shape).
      )");

  m.def(
      "zero_copy_view_count",
      &mlx_kquant::zero_copy_view_count,
      "Number of live zero-copy GGUF tensor views (registered mmap ranges).");

  m.def(
      "verify_zero_copy_views",
      [](nb::list items, std::vector<std::string> no_alias) {
        std::vector<std::pair<std::string, mx::array>> pairs;
        pairs.reserve(nb::len(items));
        for (nb::handle h : items) {
          auto t = nb::cast<nb::sequence>(h);
          pairs.emplace_back(
              nb::cast<std::string>(t[0]), nb::cast<mx::array>(t[1]));
        }
        return mlx_kquant::verify_zero_copy_views(pairs, no_alias);
      },
      "items"_a,
      "no_alias"_a = std::vector<std::string>{},
      R"(
        Check (name, array) pairs against the live zero-copy GGUF mappings.

        Returns a list of problem strings, one per violation: an array whose
        buffer sits inside a mapped GGUF tensor range but whose dtype differs
        from the wire dtype recorded at load (integer-to-integer reinterprets
        allowed), or an array named in ``no_alias`` that aliases any mapping
        (loader transforms must produce owned buffers). An empty list means
        clean. This detects buffer donation into the file mapping: a donated
        dtype-changing copy leaves an array typed X over wire bytes typed Y,
        and the write is dropped on read-only shared mappings. Arrays must be
        evaluated first. Metadata-only; no tensor data is read.

        Args:
            items (list[tuple[str, array]]): named arrays to check, e.g.
                ``mlx.utils.tree_flatten(model.parameters())``.
            no_alias (list[str]): names that must not alias any mapping.

        Returns:
            list[str]: problem descriptions; empty when clean.
      )");
}
