#!/usr/bin/env python3
"""Timing harness for gather_qmm_sorted vs gather_qmm_segments.

Recreates the sorted MoE bulk-prefill shape from a downstream speed log:
S=23058 token-expert pair rows, K=4096, N=4096, E=256, uniform segments.
gather_qmm_sorted replaces the host-built descriptor table with an in-kernel
binary search over the device-resident sorted ids; the value of the op is
barrier removal in the caller, so the acceptance bar here is parity with the
segments kernel (within 15%), not a kernel speedup. The codec is the first
argument (default iq2_xxs).

Run:
    uv run python scripts/bench_sorted.py [codec]
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np

import mlx_kquant as kq

S = 23058
K = 4096
N = 4096
E = 256
CODEC = sys.argv[1] if len(sys.argv) > 1 else "iq2_xxs"
ITERS = 5

# Activation dtype: float16 by default; set KQ_BENCH_XDTYPE=float32 to time the
# full-precision f32 variant (float32 in, float32 out) or bfloat16.
_XDTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
XDTYPE = _XDTYPES[os.environ.get("KQ_BENCH_XDTYPE", "float16")]

# Codecs whose encoder requires an importance matrix.
REQ_IMAT = {"iq2_xxs", "iq2_xs", "iq1_s"}

# Distinct experts to quantize; the pool is tiled across the E slots (see
# bench_segments.py for why this matches E distinct experts).
POOL = 8


def build():
    rng = np.random.default_rng(0)
    imat = None
    if CODEC in REQ_IMAT:
        imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    pool = []
    for _ in range(POOL):
        w_np = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
        wq, _ = kq.quantize(mx.array(w_np), CODEC, imatrix=imat)
        mx.eval(wq)
        pool.append(np.ascontiguousarray(np.array(wq).astype(np.uint8)))
    w = mx.array(np.stack([pool[e % POOL] for e in range(E)]))

    # Uniform segments: S rows spread evenly across E experts. The sorted-ids
    # array is the row-expanded form of the same table.
    base = S // E
    rem = S - base * E
    rows = []
    ids = []
    start = 0
    for e in range(E):
        count = base + (1 if e < rem else 0)
        rows.append((e, start, count))
        ids.extend([e] * count)
        start += count
    seg = mx.array(np.array(rows, dtype=np.uint32))
    ids = mx.array(np.array(ids, dtype=np.uint32))

    x = mx.array(
        (np.random.default_rng(1).standard_normal((S, K)) * 0.05).astype(np.float32)
    ).astype(XDTYPE)
    scales = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, seg, ids, x)
    return x, w, scales, seg, ids


def time_op(fn):
    out = fn()
    mx.eval(out)  # warmup
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
        mx.eval(out)
    dt = (time.perf_counter() - t0) / ITERS
    return dt, out


def main():
    print(
        f"shape: S={S} K={K} N={N} E={E} codec={CODEC} x={XDTYPE} "
        f"({ITERS} iters, uniform segments)"
    )
    x, w, scales, seg, ids = build()

    srt_dt, srt_out = time_op(
        lambda: kq.gather_qmm_sorted(x, w, scales, CODEC, ids))
    seg_dt, seg_out = time_op(
        lambda: kq.gather_qmm_segments(x, w, scales, CODEC, seg))

    # Bit-identity check: the two entry points share one segment-GEMM body.
    g = np.array(srt_out.astype(mx.float32))
    r = np.array(seg_out.astype(mx.float32))
    identical = bool(np.array_equal(g, r))
    max_abs = float(np.abs(g - r).max())

    print(f"gather_qmm_sorted   : {srt_dt * 1e3:8.1f} ms/layer")
    print(f"gather_qmm_segments : {seg_dt * 1e3:8.1f} ms/layer")
    print(f"sorted/segments     : {srt_dt / seg_dt:8.3f}x")
    print(f"bit-identical       : {identical} (max_abs {max_abs:.3e})")


if __name__ == "__main__":
    main()
