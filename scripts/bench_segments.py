#!/usr/bin/env python3
"""Timing harness for gather_qmm_segments vs the Python per-expert dq_f32 loop.

Recreates the sorted MoE bulk-prefill shape from a downstream speed log:
S=23058 token-expert pair rows, K=4096, N=4096, E=256, uniform segments.
Prints both the kernel time and the dq_f32 (f32 dequantize + f32 GEMM)
per-expert loop it is replacing. The codec is the first argument (default
iq2_xxs).

Run:
    uv run python scripts/bench_segments.py [codec]
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

# Activation dtype: float16 by default (the sorted-MoE prefill path); set
# KQ_BENCH_XDTYPE=float32 to time the full-precision f32 down-projection variant
# (float32 in, float32 out) or bfloat16.
_XDTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
XDTYPE = _XDTYPES[os.environ.get("KQ_BENCH_XDTYPE", "float16")]

# Codecs whose encoder requires an importance matrix.
REQ_IMAT = {"iq2_xxs", "iq2_xs", "iq1_s"}

# Distinct experts to quantize; the pool is tiled across the E slots. Decode
# work per block is content-independent and np.stack materializes E physical
# copies, so the measurement matches E distinct experts while the setup stays
# seconds instead of minutes (iq2_xxs encodes a 4096x4096 expert in ~3 s).
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

    # Uniform segments: S rows spread evenly across E experts.
    base = S // E
    rem = S - base * E
    rows = []
    start = 0
    for e in range(E):
        count = base + (1 if e < rem else 0)
        rows.append((e, start, count))
        start += count
    seg = mx.array(np.array(rows, dtype=np.uint32))

    x = mx.array(
        (np.random.default_rng(1).standard_normal((S, K)) * 0.05).astype(np.float32)
    ).astype(XDTYPE)
    scales = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(w, seg, x)
    return x, w, scales, seg, rows


def time_kernel(x, w, scales, seg):
    # Warmup.
    out = kq.gather_qmm_segments(x, w, scales, CODEC, seg)
    mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = kq.gather_qmm_segments(x, w, scales, CODEC, seg)
        mx.eval(out)
    dt = (time.perf_counter() - t0) / ITERS
    return dt, out


def time_dq_f32(x, w, scales, rows):
    xf = x.astype(mx.float32)

    def run():
        parts = []
        for expert, start, count in rows:
            if count == 0:
                continue
            deq = kq.dequantize(w[expert], scales, CODEC, mx.float32)
            parts.append(xf[start : start + count] @ deq.T)
        return mx.concatenate(parts, axis=0)

    out = run()
    mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = run()
        mx.eval(out)
    dt = (time.perf_counter() - t0) / ITERS
    return dt, out


def main():
    print(
        f"shape: S={S} K={K} N={N} E={E} codec={CODEC} x={XDTYPE} "
        f"({ITERS} iters, uniform segments)"
    )
    x, w, scales, seg, rows = build()

    k_dt, k_out = time_kernel(x, w, scales, seg)
    d_dt, d_out = time_dq_f32(x, w, scales, rows)

    # Agreement sanity check (f16 output).
    g = np.array(k_out.astype(mx.float32))
    r = np.array(d_out.astype(mx.float32))
    max_abs = float(np.abs(g - r).max())

    print(f"gather_qmm_segments : {k_dt * 1e3:8.1f} ms/layer")
    print(f"python dq_f32 loop  : {d_dt * 1e3:8.1f} ms/layer")
    print(f"speedup             : {d_dt / k_dt:8.2f}x")
    print(f"max_abs vs dq_f32   : {max_abs:.3e}")


if __name__ == "__main__":
    main()
