#!/usr/bin/env python3
"""Synthetic (GGUF-free) validation for gather_qmm_sorted.

gather_qmm_sorted is gather_qmm_segments with the host-built descriptor table
replaced by an in-kernel binary search over the device-resident sorted per-row
expert ids. The two entry points share one segment-GEMM body, so the contract
is exact: for any ascending id array, the output must be bit-identical to
gather_qmm_segments called with the equivalent host-built table. The tests
here build that table on the host from the same spec and compare raw values,
not tolerances.

Wire bytes are minted in-process by kq.quantize (random uint8 would decode to
NaN), following the segments test. Coverage includes ragged segments that
cross the BM row-chunk boundary of every shipped tile (counts 90 and 250 span
several 48-row chunks), single-row segments, expert gaps (empty threadgroups
in the E-wide grid), and all three activation dtypes.
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
# Experts ascend within each spec; the sorted-ids contract requires it.
SEGMENT_CASES = {
    "ragged": [(0, 1), (1, 3), (2, 37), (3, 90), (4, 250)],
    "single_segment": [(2, 40)],
    "single_row_segments": [(0, 1), (2, 1), (3, 1), (5, 1)],
    "all_rows_one_expert": [(5, 128)],
    "expert_gaps": [(1, 47), (4, 49), (5, 90)],
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


def _spec_arrays(spec, rng):
    """Expand a (expert, row_count) spec into the sorted ids, the equivalent
    host-built segments table, and matching x rows."""
    rows = []
    ids = []
    start = 0
    for expert, count in spec:
        rows.append((expert, start, count))
        ids.extend([expert] * count)
        start += count
    S = start
    seg = np.array(rows, dtype=np.uint32)
    ids = np.array(ids, dtype=np.uint32)
    x_np = (rng.standard_normal((S, K)) * 0.1).astype(np.float32)
    return ids, seg, x_np, S


def _exact_equal(a: mx.array, b: mx.array) -> bool:
    """Bitwise equality for f16/bf16/f32 outputs (the f32 cast is lossless)."""
    return bool(
        np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))
    )


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("case", list(SEGMENT_CASES))
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_gather_qmm_sorted_bit_identical_to_segments(codec, case, dtype):
    rng = np.random.default_rng(0)
    wq_list = _build_experts(codec, rng)
    w = mx.array(np.stack(wq_list))
    scales = mx.zeros((1,), dtype=mx.uint8)

    ids_np, seg_np, x_np, S = _spec_arrays(SEGMENT_CASES[case], rng)
    x = mx.array(x_np).astype(dtype)

    got = kq.gather_qmm_sorted(x, w, scales, codec, mx.array(ids_np))
    ref = kq.gather_qmm_segments(x, w, scales, codec, mx.array(seg_np))
    mx.eval(got, ref)

    assert got.dtype == dtype and ref.dtype == dtype
    assert got.shape == (S, N)
    assert _exact_equal(got, ref), f"{codec} {case} {dtype}: outputs diverge"


@pytest.mark.parametrize("codec", CODECS)
def test_gather_qmm_sorted_matches_dq_f32_reference(codec):
    """One independent semantic reference so a shared kernel bug cannot pass
    the parity check by matching itself: the per-expert f32 dequantize + f32
    GEMM loop, at the tight f32-x bound the float-staging tiles hold."""
    rng = np.random.default_rng(0)
    wq_list = _build_experts(codec, rng)
    w = mx.array(np.stack(wq_list))
    scales = mx.zeros((1,), dtype=mx.uint8)

    ids_np, seg_np, x_np, S = _spec_arrays(SEGMENT_CASES["ragged"], rng)
    x = mx.array(x_np)  # float32

    got = kq.gather_qmm_sorted(x, w, scales, codec, mx.array(ids_np))
    mx.eval(got)
    assert got.dtype == mx.float32
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
        f"{codec}: dq_f32 parity max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
    )


def test_gather_qmm_sorted_rejects_non_transpose():
    rng = np.random.default_rng(2)
    wq, _ = kq.quantize(
        mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32)), "q2_k"
    )
    mx.eval(wq)
    w = mx.array(np.stack([np.array(wq).astype(np.uint8)]))
    x = mx.array((rng.standard_normal((3, K)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    ids = mx.array(np.zeros(3, dtype=np.uint32))
    scales = mx.zeros((1,), dtype=mx.uint8)
    with pytest.raises(ValueError):
        out = kq.gather_qmm_sorted(x, w, scales, "q2_k", ids, transpose=False)
        mx.eval(out)


# ---------------------------------------------------------------------------
# Fused gate/up + SwiGLU: gather_qmm_sorted_swiglu
# ---------------------------------------------------------------------------
#
# The fused op runs the sorted segment GEMM against a combined [2N, K]
# gate/up expert stack and applies the DSV4 SwiGLU in the kernel epilogue on
# the float32 accumulators. Parity against the unfused composition
# (gather_qmm_sorted, then the SwiGLU formula in float32) holds only to a
# tolerance: the epilogue's exp() differs from the host implementation, and
# for half-precision I/O the unfused composition rounds the gate/up GEMM
# outputs through the I/O dtype while the fused epilogue reads the unrounded
# accumulators. The bounds below sit ~4x above the measured worst case per
# dtype.

# The active limit must clip the test data (gate/up products at the
# 0.1-scale inputs reach ~0.3), asserted per case so the two limit arms are
# proven to differ; the large limit never clips, which is also asserted.
SWIGLU_LIMIT_ACTIVE = 0.2
SWIGLU_LIMIT_OFF = 1e6

# (rtol, atol) per I/O dtype for fused-vs-unfused parity.
SWIGLU_TOLS = {
    mx.float16: (1e-2, 1.2e-3),
    mx.bfloat16: (5e-2, 1e-2),
    mx.float32: (2e-6, 2e-7),
}

# Combined experts are expensive to mint (E quantize calls of [2N, K]) and
# shared by every fused parametrization, so cache them per codec.
_COMBINED_EXPERTS_CACHE: dict = {}


def _build_combined_experts(codec):
    """Quantize E random combined gate/up matrices [2N, K] (gate rows first)
    into stacked wire bytes, plus their f32 dequantized forms."""
    if codec not in _COMBINED_EXPERTS_CACHE:
        rng = np.random.default_rng(3)
        imat = None
        if codec in REQ_IMAT:
            imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
        scales = mx.zeros((1,), dtype=mx.uint8)
        wq_list = []
        deq_list = []
        for _ in range(E):
            w_np = (rng.standard_normal((2 * N, K)) * 0.1).astype(np.float32)
            wq, _ = kq.quantize(mx.array(w_np), codec, imatrix=imat)
            mx.eval(wq)
            wq_np = np.ascontiguousarray(np.array(wq).astype(np.uint8))
            wq_list.append(wq_np)
            deq_list.append(
                np.array(kq.dequantize(mx.array(wq_np), scales, codec, mx.float32))
            )
        _COMBINED_EXPERTS_CACHE[codec] = (np.stack(wq_list), deq_list)
    wq_stack, deq_list = _COMBINED_EXPERTS_CACHE[codec]
    return mx.array(wq_stack), deq_list


def _swiglu_f32(combined_f32, limit):
    """The jang _dsv4_swiglu formula in float32 numpy: gate is the first N
    columns, up the last N; a positive limit clamps gate from above only and
    up symmetrically, then silu(gate) * up."""
    gate = combined_f32[:, :N].copy()
    up = combined_f32[:, N:].copy()
    if limit > 0:
        up = np.clip(up, -limit, limit)
        gate = np.minimum(gate, limit)
    return gate / (1.0 + np.exp(-gate)) * up


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("case", ["ragged", "single_row_segments", "expert_gaps"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_gather_qmm_sorted_swiglu_matches_unfused(codec, case, dtype):
    """Fused output matches gather_qmm_sorted on the combined stack followed
    by the SwiGLU formula in float32, at both an active and an effectively
    disabled clamp. Case coverage: ragged (counts 90 and 250 cross every
    shipped BM row-chunk boundary), single-row segments, and expert gaps
    (empty threadgroups in the E-wide grid)."""
    rng = np.random.default_rng(1)
    w, _deq = _build_combined_experts(codec)
    scales = mx.zeros((1,), dtype=mx.uint8)

    ids_np, _seg, x_np, S = _spec_arrays(SEGMENT_CASES[case], rng)
    x = mx.array(x_np).astype(dtype)
    ids = mx.array(ids_np)

    comb = kq.gather_qmm_sorted(x, w, scales, codec, ids)
    mx.eval(comb)
    comb_f32 = np.array(comb.astype(mx.float32))
    rtol, atol = SWIGLU_TOLS[dtype]

    for limit in (SWIGLU_LIMIT_ACTIVE, SWIGLU_LIMIT_OFF):
        got = kq.gather_qmm_sorted_swiglu(x, w, scales, codec, ids, N, limit)
        mx.eval(got)
        assert got.dtype == dtype
        assert got.shape == (S, N)

        clips = bool(
            ((comb_f32[:, :N] > limit) | (np.abs(comb_f32[:, N:]) > limit)).any()
        )
        if limit == SWIGLU_LIMIT_ACTIVE:
            assert clips, f"{codec} {case}: active limit clipped nothing"
        else:
            assert not clips, f"{codec} {case}: 'off' limit clipped data"

        ref = _swiglu_f32(comb_f32, limit)
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32)),
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{codec} {case} {dtype} limit={limit}",
        )


@pytest.mark.parametrize("codec", CODECS)
def test_gather_qmm_sorted_swiglu_matches_dq_f32_reference(codec):
    """One independent semantic reference so a bug shared with
    gather_qmm_sorted cannot pass the parity check by matching itself: the
    per-expert f32 dequantize + f32 GEMM loop with the SwiGLU formula in
    numpy, at the tight f32-x bound."""
    rng = np.random.default_rng(1)
    w, deq = _build_combined_experts(codec)
    scales = mx.zeros((1,), dtype=mx.uint8)

    ids_np, seg_np, x_np, S = _spec_arrays(SEGMENT_CASES["ragged"], rng)
    x = mx.array(x_np)  # float32

    got = kq.gather_qmm_sorted_swiglu(
        x, w, scales, codec, mx.array(ids_np), N, SWIGLU_LIMIT_ACTIVE
    )
    mx.eval(got)
    assert got.dtype == mx.float32
    g = np.array(got)

    comb = np.zeros((S, 2 * N), dtype=np.float32)
    for expert, start, count in seg_np:
        comb[start : start + count] = x_np[start : start + count] @ deq[expert].T
    ref = _swiglu_f32(comb, SWIGLU_LIMIT_ACTIVE)

    diff = np.abs(g - ref)
    max_rel = float((diff / (np.abs(ref) + 1e-3)).max())
    max_abs = float(diff.max())
    assert max_rel < 1e-3 or max_abs < 1e-4, (
        f"{codec}: dq_f32 parity max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
    )


def test_gather_qmm_sorted_swiglu_deep_negative_gate_is_zero():
    """A gate accumulator past the float32 exp overflow point must produce
    exactly zero, not NaN: for gate below about -88, exp(-gate) overflows to
    +Inf and silu resolves as gate / Inf -> 0. The epilogue's precise::exp
    keeps that IEEE edge; a fast-math exp leaves the Inf class undefined.
    All-negative gate rows against all-positive x put every accumulator below
    -100, and the up rows stay large positive so a zero can only come from
    the silu factor."""
    rng = np.random.default_rng(9)
    scales = mx.zeros((1,), dtype=mx.uint8)
    gate_np = -(np.abs(rng.standard_normal((N, K))) * 0.5 + 0.5)
    up_np = np.abs(rng.standard_normal((N, K))) * 0.5 + 0.5
    w_np = np.concatenate([gate_np, up_np]).astype(np.float32)
    wq, _ = kq.quantize(mx.array(w_np), "q2_k")
    mx.eval(wq)
    wq_np = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    deq = np.array(kq.dequantize(mx.array(wq_np), scales, "q2_k", mx.float32))

    S = 4
    x_np = (np.abs(rng.standard_normal((S, K))) * 0.5 + 0.25).astype(np.float32)
    x64 = x_np.astype(np.float64)
    gate_ref = x64 @ deq[:N].astype(np.float64).T
    up_ref = x64 @ deq[N:].astype(np.float64).T
    assert gate_ref.max() < -100.0, "gate accumulators must sit past -100"
    assert np.abs(up_ref).min() > 1.0, "up factors must stay away from zero"

    w = mx.array(wq_np[None])
    ids = mx.array(np.zeros(S, dtype=np.uint32))
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        for limit in (0.0, SWIGLU_LIMIT_OFF):
            got = kq.gather_qmm_sorted_swiglu(
                mx.array(x_np).astype(dtype), w, scales, "q2_k", ids, N, limit
            )
            mx.eval(got)
            out = np.array(got.astype(mx.float32))
            assert not np.isnan(out).any(), f"{dtype} limit={limit}: NaN"
            assert (out == 0.0).all(), f"{dtype} limit={limit}: nonzero"


def test_gather_qmm_sorted_swiglu_rejects_bad_gate_out():
    """w must hold exactly 2 * gate_out rows per expert."""
    rng = np.random.default_rng(2)
    wq, _ = kq.quantize(
        mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32)),
        "q2_k",
    )
    mx.eval(wq)
    w = mx.array(np.stack([np.array(wq).astype(np.uint8)]))  # N rows, not 2N
    x = mx.array((rng.standard_normal((3, K)) * 0.1).astype(np.float32))
    ids = mx.array(np.zeros(3, dtype=np.uint32))
    scales = mx.zeros((1,), dtype=mx.uint8)
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted_swiglu(x, w, scales, "q2_k", ids, N, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted_swiglu(x, w, scales, "q2_k", ids, 0, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted_swiglu(
            x, w, scales, "q2_k", ids, N // 2, 0.2, transpose=False
        )


def test_gather_qmm_sorted_swiglu_rejects_bad_ids():
    rng = np.random.default_rng(2)
    wq, _ = kq.quantize(
        mx.array((rng.standard_normal((2 * N, K)) * 0.1).astype(np.float32)),
        "q2_k",
    )
    mx.eval(wq)
    w = mx.array(np.stack([np.array(wq).astype(np.uint8)]))
    x = mx.array((rng.standard_normal((3, K)) * 0.1).astype(np.float32))
    scales = mx.zeros((1,), dtype=mx.uint8)
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted_swiglu(
            x, w, scales, "q2_k", mx.array(np.zeros(3, dtype=np.int32)), N, 0.2
        )
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted_swiglu(
            x, w, scales, "q2_k", mx.array(np.zeros(4, dtype=np.uint32)), N, 0.2
        )


def test_gather_qmm_sorted_rejects_bad_ids():
    rng = np.random.default_rng(2)
    wq, _ = kq.quantize(
        mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32)), "q2_k"
    )
    mx.eval(wq)
    w = mx.array(np.stack([np.array(wq).astype(np.uint8)]))
    x = mx.array((rng.standard_normal((3, K)) * 0.1).astype(np.float32)).astype(
        mx.float16
    )
    scales = mx.zeros((1,), dtype=mx.uint8)
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted(
            x, w, scales, "q2_k", mx.array(np.zeros(3, dtype=np.int32))
        )
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted(
            x, w, scales, "q2_k", mx.array(np.zeros(4, dtype=np.uint32))
        )
    with pytest.raises(ValueError):
        kq.gather_qmm_sorted(
            x, w, scales, "q2_k", mx.array(np.zeros((3, 1), dtype=np.uint32))
        )


# The register-direct opt-in tiles (KQ_SEG_TILE suffix dr) decode B fragment
# values straight into registers with no threadgroup staging. The decode
# arithmetic, fragment values, and tile_matmad sequence match the float
# tiles, so the contract is byte-equality with the default tile, not a
# tolerance bound. KQ_SEG_TILE is read once per process, so each arm runs in
# its own subprocess and the raw output hashes are compared. N = 192
# exercises a partial final n-tile at BN=128; the ragged spec covers partial
# row chunks and an expert gap. The dr tiles are instantiated for iq2_xxs and
# q2_k only.
_DR_TILE_HASHES = textwrap.dedent(
    """
    import hashlib
    import numpy as np
    import mlx.core as mx
    import mlx_kquant as kq

    rng = np.random.default_rng(0)
    K, N, E = 512, 192, 6
    imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    experts = {}
    for codec in ("iq2_xxs", "q2_k"):
        wq = []
        for _ in range(E):
            w_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
            q, _ = kq.quantize(
                mx.array(w_np), codec,
                imatrix=imat if codec == "iq2_xxs" else None,
            )
            mx.eval(q)
            wq.append(np.ascontiguousarray(np.array(q).astype(np.uint8)))
        experts[codec] = mx.array(np.stack(wq))
    spec = [(0, 1), (1, 3), (2, 37), (4, 90), (5, 250)]
    ids = np.concatenate(
        [np.full(c, e, np.uint32) for e, c in spec]
    )
    S = int(ids.shape[0])
    ids = mx.array(ids)
    x_np = (rng.standard_normal((S, K)) * 0.1).astype(np.float32)
    scales = mx.zeros((1,), dtype=mx.uint8)
    for codec in ("iq2_xxs", "q2_k"):
        for dtype in (mx.float16, mx.bfloat16, mx.float32):
            x = mx.array(x_np).astype(dtype)
            out = kq.gather_qmm_sorted(x, experts[codec], scales, codec, ids)
            mx.eval(out)
            # bfloat16 has no numpy view; the f32 upcast is exact, so hashing
            # the upcast preserves byte-equality comparisons.
            h = hashlib.sha256(
                np.array(out.astype(mx.float32)).tobytes()
            ).hexdigest()
            print(codec, dtype, h)
    """
)

_DR_HASH_CACHE: dict = {}


def _dr_tile_hashes(tile):
    """Run the hash script with KQ_SEG_TILE pinned (None = default tile)."""
    if tile in _DR_HASH_CACHE:
        return _DR_HASH_CACHE[tile]
    env = dict(os.environ)
    env.pop("KQ_SEG_TILE", None)
    if tile is not None:
        env["KQ_SEG_TILE"] = tile
    proc = subprocess.run(
        [sys.executable, "-c", _DR_TILE_HASHES],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{tile}: {proc.stderr}"
    _DR_HASH_CACHE[tile] = proc.stdout
    return proc.stdout


@pytest.mark.parametrize("tile", ["t48x128x16dr", "t48x128x16dr2", "t48x64x16dr"])
def test_gather_qmm_sorted_devr_bit_identical(tile):
    if mx.default_device() == mx.cpu:
        pytest.skip("register-direct opt-in is a GPU kernel path")
    default = _dr_tile_hashes(None)
    got = _dr_tile_hashes(tile)
    assert got == default, (
        f"{tile} output hashes diverge from the default tile:\n{got}\nvs\n{default}"
    )
