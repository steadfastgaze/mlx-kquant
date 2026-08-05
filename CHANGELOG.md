# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Rolling]

### Fixed
- `quantized_matmul` returned wrong values when an unevaluated (lazy)
  row-strided activation view - for example a grouped tensor sliced on the
  second-to-last axis - met any multi-row shape (M >= 2) at a half-precision
  dtype. The op-level matrix-contiguity check runs at graph construction,
  where an unevaluated array still carries the constructor's dense
  placeholder strides and default-true contiguity flags, so the strided view
  passed unconverted and the kernels, which advance the M rows densely by K
  after the batch offset, read the wrong elements (measured rel error
  ~0.5-1.8 against the contiguized reference on every codec and both weight
  forms; float32 activations were laundered dense by the op's astype;
  single-row calls have no row stride to misread; a pre-evaluated strided
  view was contiguized correctly, which made the failure depend on
  evaluation batching). The op now contiguizes unevaluated multi-row
  activations lazy-safely (`mx::contiguous` evaluates to a zero-cost
  shared-buffer pass-through when the materialized layout is already
  row-contiguous, so dense lazy activations gain no copy, and single-row
  decode calls keep the node-free path), and `KQuantMatmul` eval fails
  closed with a clear error on any operand that still arrives without
  densely packed rows instead of computing garbage. Outputs on every
  previously correct operand set are bit-identical; the previously wrong
  sets now match the contiguized-reference call bit-exactly.

### Added
- `sdpa_decode_q8(..., dimension_parallel_merge=True)` adds an opt-in
  dimension-parallel pass-two kernel for the 16-query-head, 2-KV-head,
  head-dim-256, split-128, tile-16 serving geometry at cache depths of at least
  8192. It preserves the shared merge's max, denominator, and per-output split
  order while assigning 32 output dimensions to each of eight SIMD groups.
  Off-contract calls retain the shared merge.
- **Fused q8 decode attention** (`kq.sdpa_decode_q8`): the served KV-attention
  read for one query row (qL == 1) over a whole QuantizedKVCache tuple (packed
  uint32, float32 scales and biases, group 64, 8 bits). The whole GQA group
  folds into one query tile, the key axis splits into contiguous chunks each
  streamed once through threadgroup staging dequantized with `fma(scale, q,
  bias)` (the exact mx form) with a per-split online softmax and no score
  tensor, and the partials merge through the shared `kq_sdpa_gqa_2pass_2`
  reduction. Two compute patterns behind the `compute` argument: 1 (default) is
  the SIMD-shuffle reduction (`sdpa_decode_gqa`'s decode-latency compute with
  the q8 dequant folded into the cooperative tile load, float staging, `tile_c`
  8 or 16); 0 is the matrix-unit tile (`sdpa_fa_prefill_q8`'s idiom, `stage`
  picks its staging precision). Float32 queries and accumulators; a decode query
  attends every key (no causal cut inside the past). Head dim 256, gqa factor 8
  (16 query heads over 2 kv heads).
- **Flash prefill attention, q8 past phase** (`kq.sdpa_fa_prefill_q8`): the
  serving form of the flash prefill kernel. Float32 queries and accumulators;
  the past prefix reads directly from a QuantizedKVCache tuple (packed uint32,
  float32 scales and biases, group 64, 8 bits) and dequantizes with
  `fma(scale, q, bias)`, the exact mx dequantization form, into half-precision
  threadgroup staging during the cooperative load; the fresh self chunk stages
  dense alongside. Staging precision selectable (bfloat16, float16, or float32
  at a narrower key tile for verification and memory-lever use), query rows
  per threadgroup 32 or 64, key tile 32 or 48, and a prefill-shaped split
  default (splits track key depth; the query-tile grid supplies parallelism).
  No score tensor reaches device memory.
- **Flash prefill attention** (`kq.sdpa_fa_prefill`): simdgroup-matrix
  head-dim-256 GQA prefill attention over a full query chunk with an online
  softmax and float32 accumulators, so the quadratic score tensor never
  materializes. Extends the `sdpa_fa_verify` tile with an outer query-tile grid
  loop; queries stay in `[1, Hq, qL, D]`, each 32-row tile folds the GQA group
  with `qw` query positions, and the causal band matches the composed prefill
  path (query p attends keys <= (kL - qL) + p). Float32 (the served query
  dtype), bfloat16, and float16; the float32 path stages 16 keys per tile to
  fit the threadgroup limit. Peak attention memory falls ~2x at long context;
  the float32 staging width measured about 3 percent behind the composed path
  on served long-context prefill (779 against 805 t/s at 37K), the time cost
  of its 16-key tile.
- `sdpa_vector` attention sinks and boolean masks, with a wide-MQA dispatch
  shape for single-KV-head decode (one query head block per simdgroup).
- **Segmented MoE quantized GEMM** (`kq.gather_qmm_segments`): bulk routed
  prefill over a descriptor of contiguous expert row ranges, float32 I/O and
  accumulate on the GGUF wire bytes.
- **Sorted-ids MoE quantized GEMM** (`kq.gather_qmm_sorted`): the segmented
  GEMM with the per-expert row ranges derived in-kernel from sorted expert
  ids, removing the host-side descriptor pass.
- Device-A fragment tiles for the segments/sorted GEMM family
  (`KQ_SEG_TILE`, default `t48x128x16a`): simdgroup-matrix staging of the
  dequantized A fragment; `fh`/`ah` half-staging variants remain opt-in.
- **Fused SwiGLU routed GEMM** (`kq.gather_qmm_sorted_swiglu`): combined
  gate/up expert stacks activate in the epilogue, halving the bulk routed
  dispatch count for GLU experts.
- Register-direct dequant body for the segments/sorted GEMM family as a
  parked opt-in (`KQ_SEG_TILE=...dr`); measured slower than threadgroup
  staging on M3, kept for newer-architecture probes.
- **Decode-shaped routed-MoE matvec pair** (`kq.gather_qmv_pair_swiglu`,
  `kq.gather_qmv_expert_sum`): one dispatch computes every routed expert's
  fused gate/up + SwiGLU activation row for a single token with the route
  weight baked into the stored intermediate (combined gate/up stack,
  iq2_xxs), and one dispatch computes the down matvec with the sum over the
  token's routed experts inside the kernel (q2_k), removing the separate
  route-weighted-sum reduction. Float32 accumulate; numerically equivalent
  but not bit-identical to the unfused gather_qmm composition. Instantiated
  for the served DS4 decode codec pair only; other codecs fail closed at the
  op level.

### Changed
- Tile-16 fused q8 decode attention loads each four-word packed K/V span with
  one aligned `uint4` read. The dequantization arithmetic, staging indices,
  dynamic cache strides, and reduction order are unchanged. Misaligned cache
  rows fall back to the scalar loader. `KQ_SDPA_Q8_UINT4_LOAD=0` retains the
  scalar path as an exact A/B control.
- SwiGLU epilogues pin `metal::precise::exp`; the softmax fast::exp sites
  carry an annotation for why the fast form stays.


## [0.3.1]

### Added
- `sdpa_fa_verify`: simdgroup-matrix speculative-verify attention for a
  GQA-folded query tile (G*q_len <= 32 rows, q_len 2..8, head_dim 256).
  Streams each contiguous KV split once through threadgroup-staged K/V tiles
  with S = Q@K^T and O += P@V on the matrix units, float32 accumulators, and a
  per-row offset-causal online softmax; reuses the `sdpa_decode_gqa`
  split-merge pass.
- `verify_zero_copy_views(items, no_alias=[])` and `zero_copy_view_count()`:
  post-load integrity check that arrays backed by a GGUF mapping still carry
  their wire dtype (integer reinterprets allowed) and that `no_alias` names
  own their buffers. Catches buffer donation into the file mapping: a donated
  dtype-changing copy leaves an array typed X over wire bytes typed Y, and
  the write is dropped on read-only maps. Metadata-only, O(#tensors).

### Changed
- `load_gguf` zero-copy mappings are now read-only shared (`PROT_READ` +
  `MAP_SHARED`) instead of writable private. Private mappings made the GPU
  write-fault every wired page, lazily copying the whole file into anonymous
  memory that can only be compressed, never dropped (a hidden full-model RAM
  copy on top of the page cache carrying the same bytes); read-only shared
  pages are wired in place and stay clean, evictable file cache.
  `KQ_GGUF_MMAP=private_rw` restores the old mode; `private_ro` is a
  diagnostic quadrant.

## [0.3.0]

### Added
- Fused MoE gather kernels: for quantized experts: `moe_glu_gather_kq` /
  `gather_qmv_kq` cover all 19 GGUF codecs (K-quant, legacy, IQ) plus
  mixed-codec shared experts; `moe_glu_gather` / `gather_qmv_bias` cover mxfp4.
  Each fuses the expert gather, dequant mat-vec, and GLU activation into one
  dispatch. Wide K-lane (NX=16/32) variants engage automatically for
  decode-scale two-stream GLU gathers.
- Fused MoE router: (`moe_router_topk`): softmax + top-k + weight norm +
  shared-gate sigmoid in one dispatch; no-shexp routing-mix gather and
  per-expert-scale support; q8_0 odd-K fallback.
- `sdpa_decode_gqa`: tile-staged GQA decode SDPA kernel (head_dim
  64/128/256/512, GQA factor 2..16, attention sinks folded into the merge
  pass), with pair variants accepting verify widths (q_len 2..4, subject to
  gqa_factor * q_len <= 32).
- Fused residual/rmsnorm glue ops (`add_rmsnorm`, `rmsnorm_multi3`,
  `rmsnorm2_add`) with register-cached 4-wide reads, row-sized threadgroups,
  and scalar CPU eval paths.

### Changed
- `KQuantEmbedding` output dtype defaults to bf16 (was f16).
- q8_0 decode `qmv`/`qmv_fast` fuse the bias add.

## [0.2.1]

### Added
- **Verify mat-vec kernels (`mul_mv_ext` port)** for speculative-decode verify
  and batched MTP. Flat-with-M kernels ported from ggml's `mul_mv_ext` for all
  K-quant, legacy, and IQ codecs via a shared `kq_mv_ext_impl` template. Gated
  at M>=3 for non-IQ codecs (verify_qmv wins at M<=2) and all-M for IQ. Extends
  to M=12 for batch MTP verify.
- **Compiled vector SDPA kernel (`kq.sdpa_vector`)** for head_dim 256 and 512
  with GQA and strided KV support.
- **Synthetic all-codec matmul test** (`test_matmul_synth.py`) - GGUF-free
  quantized-matmul validation across all codecs.

## [0.2.0]

### Added
- IQ codecs are now full-featured alongside the K-quant/legacy codecs. All nine
  (`iq4_nl, iq4_xs, iq3_s, iq3_xxs, iq2_xxs, iq2_xs, iq2_s, iq1_s, iq1_m`) gained:
  - **NAX (tensor-core) matmul** for prefill, matching the K-quant kernels
    (decode `qmv` unchanged).
  - **`quantize` (encode)** — a scalar CPU port of ggml's `quantize_row_iq*`
    quantizers (grid search + inverse-index neighbour tables), so `kq.quantize`,
    the `convert` driver, and the `mlx-kquant quantize` CLI now produce every IQ
    codec. IQ encode is CPU-only (ggml has no GPU IQ quantizer) and is routed to
    a CPU stream internally. `iq2_xxs`, `iq2_xs`, and `iq1_s` require an
    importance matrix (mirroring ggml) and reject a missing one.

## [0.1.2]

### Fixed
- `mlx-kquant lora`/`chat` require `--model` and no longer fall back to an mlx-lm
  default model, potentially fetched from hf hub.
- expanded cli help

## [0.1.1]

Docs: PyPI install instructions, no code changes

## [0.1.0]

First public release. A C++/Metal extension for a stock `mlx==0.31.2` wheel that
adds the K-quant superblock and per-block integer codecs as native MLX ops
(`kq.dequantize` / `quantized_matmul` / `gather_qmm` / `quantize`), with Metal and
portable CPU paths for all ten codecs. On top of the ops, the `mlx-kquant` CLI
quantizes an HF / mlx-lm model into a K-quant MLX safetensors checkpoint and runs,
chats with, LoRA-fine-tunes, and fuses it, with importance-matrix calibration and
per-tensor recipe inspection. A loader runs those checkpoints on stock mlx-lm.

### Notes
- `requires-python >= 3.10` (mlx 0.31.2 ships no cp39 wheel).
- The GPU path is macOS 26 (Tahoe) or later on Apple Silicon (Metal); the NAX
  matmul kernel needs the Metal 4 SDK (`MetalPerformancePrimitives`). Linux is
  supported CPU-only - build against `mlx[cpu]==0.31.2`; model forwards there also
  need `MLX_DISABLE_COMPILE=1` (an upstream MLX CPU-JIT limitation under GCC, not
  mlx-kquant).
