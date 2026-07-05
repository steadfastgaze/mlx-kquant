#!/usr/bin/env python3
"""Synthetic (GGUF-free) validation for gather_qmm_segments.

The descriptor-driven segmented GEMM multiplies each block of sorted pair rows
against its expert's quantized weight. Wire bytes are minted in-process by
kq.quantize (random uint8 would decode to NaN), so every codec gets coverage
with no model download.

Two references, both composed per-segment from the SAME dequantized weights the
kernel decodes:

  * ``kq.dequantize(w[e], f32) @ x_seg.T`` per segment, cast to the output dtype
    (the semantic reference; f16-output tolerance).
  * the identical float32 composition ("dq_f32 loop") the default float-staging
    tile reproduces; that tile decodes to f32 and accumulates in f32, so it must
    agree tightly (rel bound 1e-3).

A float32 activation stages float tiles too, so its parity is tight for the
same reason (``test_gather_qmm_segments_f32x``). The half-staging opt-in tile
(KQ_SEG_TILE suffix h) rounds weights through the f16 I/O type, so it holds only
the semantic bound, checked in a subprocess that pins the tile before dispatch.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

# imatrix-required codecs (ggml rejects encoding them without one).
REQ_IMAT = {"iq2_xxs", "iq2_xs", "iq1_s"}

CODECS = ["iq2_xxs", "q2_k"]
K = 512
N = 256
E = 6

# (name, list of (expert_index, row_count)) -> row_start filled by cumsum.
SEGMENT_CASES = {
    "ragged": [(0, 1), (1, 3), (2, 37), (3, 90), (4, 250)],
    "single_segment": [(2, 40)],
    "all_rows_one_expert": [(5, 128)],
}


def _build_experts(codec, rng):
    """Quantize E random expert weight matrices [N, K] into stacked wire bytes."""
    imat = None
    if codec in REQ_IMAT:
        imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    wq_list = []
    for _ in range(E):
        w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
        wq, _ = kq.quantize(mx.array(w_np), codec, imatrix=imat)
        mx.eval(wq)
        wq_list.append(np.ascontiguousarray(np.array(wq).astype(np.uint8)))
    return wq_list


def _segments_and_x(spec, rng):
    """Expand a (expert, row_count) spec into a [T,3] table + sorted x rows."""
    rows = []
    start = 0
    for expert, count in spec:
        rows.append((expert, start, count))
        start += count
    S = start
    seg = np.array(rows, dtype=np.uint32)
    x_np = (rng.standard_normal((S, K)) * 0.1).astype(np.float32)
    return seg, x_np, S


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("case", list(SEGMENT_CASES))
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gather_qmm_segments(codec, case, dtype):
    rng = np.random.default_rng(0)
    wq_list = _build_experts(codec, rng)
    w = mx.array(np.stack(wq_list))
    scales = mx.zeros((1,), dtype=mx.uint8)

    seg_np, x_np, S = _segments_and_x(SEGMENT_CASES[case], rng)
    seg = mx.array(seg_np)
    x = mx.array(x_np).astype(dtype)

    got = kq.gather_qmm_segments(x, w, scales, codec, seg)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))

    # Per-expert dequant (f32), shared by both references so a dequant bug
    # cannot cancel out.
    deq = [
        np.array(kq.dequantize(mx.array(wq_list[e]), scales, codec, mx.float32))
        for e in range(E)
    ]
    xn = np.array(x.astype(mx.float32))
    ref_f32 = np.zeros((S, N), dtype=np.float32)
    for expert, start, count in seg_np:
        ref_f32[start : start + count] = xn[start : start + count] @ deq[expert].T

    # Semantic reference: same math, then f16-output rounding tolerance.
    diff = np.abs(g - ref_f32)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(ref_f32) + 1e-3)).max())
    assert max_rel < 5e-2 or max_abs < 1e-2, (
        f"{codec} {case} {dtype}: max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
    )

    # The kernel decodes to f32 and accumulates in f32, the exact math of the
    # dq_f32 loop, differing only by the final cast to the output dtype. Round
    # the reference through the same cast and require agreement within one
    # output-dtype ULP (the float16 mantissa is 2^-11, bfloat16's is 2^-8, so
    # the "tight" rel bound scales with the output precision).
    ref_cast = np.array(mx.array(ref_f32).astype(dtype).astype(mx.float32))
    diff_cast = np.abs(g - ref_cast)
    tight = 1e-3 if dtype == mx.float16 else 1e-2
    tol = tight * (np.abs(ref_cast) + 1e-3)
    assert np.all(diff_cast <= tol), (
        f"{codec} {case} {dtype}: dq_f32 disagreement "
        f"max={float((diff_cast / (np.abs(ref_cast) + 1e-3)).max()):.3e}"
    )


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("case", list(SEGMENT_CASES))
def test_gather_qmm_segments_f32x(codec, case):
    """float32 activation -> float32 output, full-precision decode.

    A float32 x stages float tiles and decodes each weight to float32 before the
    multiply, so the kernel reproduces the per-segment ``dequantize(w, f32) @
    x.T`` composition to float rounding. This is the down-projection seam that
    needs f32 activation precision, so the bound is tight: the only slack is the
    matmul reduction order, not any narrowing of the operands.
    """
    rng = np.random.default_rng(0)
    wq_list = _build_experts(codec, rng)
    w = mx.array(np.stack(wq_list))
    scales = mx.zeros((1,), dtype=mx.uint8)

    seg_np, x_np, S = _segments_and_x(SEGMENT_CASES[case], rng)
    seg = mx.array(seg_np)
    x = mx.array(x_np)  # float32, unchanged

    got = kq.gather_qmm_segments(x, w, scales, codec, seg)
    mx.eval(got)
    assert got.dtype == mx.float32, f"{codec} {case}: expected f32 out, got {got.dtype}"
    g = np.array(got)

    deq = [
        np.array(kq.dequantize(mx.array(wq_list[e]), scales, codec, mx.float32))
        for e in range(E)
    ]
    ref = np.zeros((S, N), dtype=np.float32)
    for expert, start, count in seg_np:
        ref[start : start + count] = x_np[start : start + count] @ deq[expert].T

    diff = np.abs(g - ref)
    max_rel = float((diff / (np.abs(ref) + 1e-3)).max())
    max_abs = float(diff.max())
    assert max_rel < 1e-3 or max_abs < 1e-4, (
        f"{codec} {case}: f32-x parity max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
    )


def test_gather_qmm_segments_covers_all_rows():
    """A multi-expert table whose union is exactly [0, S) reconstructs fully."""
    rng = np.random.default_rng(1)
    codec = "q2_k"
    wq_list = _build_experts(codec, rng)
    w = mx.array(np.stack(wq_list))
    scales = mx.zeros((1,), dtype=mx.uint8)

    spec = [(0, 5), (3, 5), (1, 5), (5, 5)]  # experts out of order, disjoint
    seg_np, x_np, S = _segments_and_x(spec, rng)
    x = mx.array(x_np).astype(mx.float16)
    got = np.array(kq.gather_qmm_segments(x, w, scales, codec, mx.array(seg_np)))

    deq = [
        np.array(kq.dequantize(mx.array(wq_list[e]), scales, codec, mx.float32))
        for e in range(E)
    ]
    xn = x_np
    ref = np.zeros((S, N), dtype=np.float32)
    for expert, start, count in seg_np:
        ref[start : start + count] = xn[start : start + count] @ deq[expert].T
    diff = np.abs(got.astype(np.float32) - ref)
    assert float(diff.max()) < 1e-2, f"max_abs={float(diff.max()):.3e}"


def test_gather_qmm_segments_rejects_non_transpose():
    rng = np.random.default_rng(2)
    wq, _ = kq.quantize(
        mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32)), "q2_k"
    )
    mx.eval(wq)
    w = mx.array(np.stack([np.array(wq).astype(np.uint8)]))
    x = mx.array((rng.standard_normal((3, K)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    seg = mx.array(np.array([[0, 0, 3]], dtype=np.uint32))
    scales = mx.zeros((1,), dtype=mx.uint8)
    with pytest.raises(ValueError):
        out = kq.gather_qmm_segments(x, w, scales, "q2_k", seg, transpose=False)
        mx.eval(out)


# The half-staging opt-in tile (KQ_SEG_TILE suffix h) rounds weights through the
# f16 I/O type before the multiply, so it agrees with the semantic reference at
# f16-decode precision but not to the tight dq_f32 bound the float tiles hold.
# KQ_SEG_TILE is read once per process at first dispatch, so the tile must be
# chosen before any segments op runs; a subprocess is the reliable way to pin it.
_HALF_TILE_CHECK = textwrap.dedent(
    """
    import numpy as np
    import mlx.core as mx
    import mlx_kquant as kq

    rng = np.random.default_rng(0)
    K, N, E = 512, 256, 6
    imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    wq = []
    for _ in range(E):
        w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
        q, _ = kq.quantize(mx.array(w_np), "iq2_xxs", imatrix=imat)
        mx.eval(q)
        wq.append(np.ascontiguousarray(np.array(q).astype(np.uint8)))
    w = mx.array(np.stack(wq))
    scales = mx.zeros((1,), dtype=mx.uint8)
    spec = [(0, 1), (1, 3), (2, 37), (3, 90), (4, 250)]
    rows = []
    start = 0
    for e, c in spec:
        rows.append((e, start, c))
        start += c
    S = start
    seg = mx.array(np.array(rows, dtype=np.uint32))
    x_np = (rng.standard_normal((S, K)) * 0.1).astype(np.float32)
    x = mx.array(x_np).astype(mx.float16)
    out = kq.gather_qmm_segments(x, w, scales, "iq2_xxs", seg)
    got = np.array(out.astype(mx.float32))
    deq = [
        np.array(kq.dequantize(mx.array(wq[e]), scales, "iq2_xxs", mx.float32))
        for e in range(E)
    ]
    ref = np.zeros((S, N), np.float32)
    for e, st, c in np.array(seg):
        ref[st : st + c] = x_np[st : st + c] @ deq[e].T
    d = np.abs(got - ref)
    rel = float((d / (np.abs(ref) + 1e-3)).max())
    abs_max = float(d.max())
    assert rel < 5e-2 or abs_max < 1e-2, f"half rel={rel:.3e} abs={abs_max:.3e}"
    print("HALF_OK", rel)
    """
)


@pytest.mark.parametrize("tile", ["t32x64x64h", "t64x64x32h"])
def test_gather_qmm_segments_half_staging_optin(tile):
    if mx.default_device() == mx.cpu:
        pytest.skip("half-staging opt-in is a GPU kernel path")
    env = dict(os.environ, KQ_SEG_TILE=tile)
    proc = subprocess.run(
        [sys.executable, "-c", _HALF_TILE_CHECK],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{tile}: {proc.stderr}"
    assert "HALF_OK" in proc.stdout, f"{tile}: {proc.stdout}\n{proc.stderr}"
