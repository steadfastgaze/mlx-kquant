#!/usr/bin/env python3
"""Measure the pure-GEMM matmul floors at the DS4 segments bench shape.

Separates the simdgroup-matmul rate question: is f32 simd matmul inherently
slower than f16, or is the segments kernel carrying reclaimable overhead? Times
mx.matmul at [S,K] x [K,N] for f16xf16, f32xf32, and bf16, plus a per-expert
f16 GEMM emulating the "pre-dequantized f16 GEMM" reference (weights already
f16 in DRAM, one matmul per expert segment).
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

S = 23058
K = 4096
N = 4096
E = 256
ITERS = 5


def timeit(fn):
    out = fn()
    mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / ITERS


def main():
    rng = np.random.default_rng(0)
    x16 = mx.array((rng.standard_normal((S, K)) * 0.05).astype(np.float32)).astype(
        mx.float16
    )
    w16 = mx.array((rng.standard_normal((N, K)) * 0.05).astype(np.float32)).astype(
        mx.float16
    )
    x32 = x16.astype(mx.float32)
    w32 = w16.astype(mx.float32)
    mx.eval(x16, w16, x32, w32)

    dt_f16 = timeit(lambda: mx.matmul(x16, w16.T))
    dt_f32 = timeit(lambda: mx.matmul(x32, w32.T))
    dt_bf16 = timeit(
        lambda: mx.matmul(x16.astype(mx.bfloat16), w16.astype(mx.bfloat16).T)
    )

    # Per-expert f16 GEMM: E segments of ~S/E rows each, one matmul apiece.
    base = S // E
    rows = []
    start = 0
    for e in range(E):
        count = base + (1 if e < (S - base * E) else 0)
        rows.append((start, count))
        start += count
    experts16 = [
        mx.array((rng.standard_normal((N, K)) * 0.05).astype(np.float32)).astype(
            mx.float16
        )
        for _ in range(8)
    ]
    mx.eval(experts16)

    def per_expert_f16():
        parts = []
        for i, (st, ct) in enumerate(rows):
            parts.append(mx.matmul(x16[st : st + ct], experts16[i % 8].T))
        return mx.concatenate(parts, axis=0)

    dt_pe16 = timeit(per_expert_f16)

    print(f"shape S={S} K={K} N={N} E={E} ({ITERS} iters)")
    print(f"mx.matmul f16xf16      : {dt_f16 * 1e3:8.1f} ms")
    print(f"mx.matmul bf16xbf16    : {dt_bf16 * 1e3:8.1f} ms")
    print(f"mx.matmul f32xf32      : {dt_f32 * 1e3:8.1f} ms")
    print(f"per-expert f16 GEMM    : {dt_pe16 * 1e3:8.1f} ms")
    print(f"f32/f16 ratio          : {dt_f32 / dt_f16:8.2f}x")


if __name__ == "__main__":
    main()
