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
        np.array_equal(
            np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
        )
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
