#!/usr/bin/env python3
"""Vector-SDPA validation: kq.sdpa_vector vs an INDEPENDENT f32 materialized
attention reference (matmul -> masked softmax -> matmul, accumulated in float32).

The reference shares no code with the kernel, so a bug in the two-pass online
softmax cannot cancel out of both sides. Covers the head dims stock MLX's fused
vector path excludes (256, 512), both float dtypes, GQA, the decode (qL=1) and
speculative-verify (qL>1, offset-causal) widths, and a strided KV-cache prefix
(head stride > kL*D, the RotatingKVCache layout) which the op must read in place
without a copy.

The kernel is Metal-only (eval_cpu throws), so the module is skipped under
KQUANT_FORCE_CPU.

Usage:
    test_sdpa.py [--d 512] [--ql 1,5] [--kl 2048,8192]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest

import mlx_kquant as kq

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="kq.sdpa_vector is a Metal-only kernel; no CPU path.",
)

# rel-norm bound per dtype: bf16 has ~8 mantissa bits, fp16 ~11.
REL_BOUND = {mx.bfloat16: 5e-3, mx.float16: 2e-3}


def _ref_sdpa(q, k, v, scale, causal, mask=None, sinks=None):
    """f32 materialized attention. Offset-causal: query row i (of qL) attends
    keys <= kL - qL + i, matching the kernel's mask. `sinks` adds one extra
    softmax logit per query head with no value row."""
    g = q.shape[1] // k.shape[1]
    kr = mx.repeat(k, g, axis=1).astype(mx.float32)
    vr = mx.repeat(v, g, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) @ kr.swapaxes(-1, -2)) * scale  # [B,Hq,qL,kL]
    if causal:
        qL, kL = q.shape[2], k.shape[2]
        rows = mx.arange(kL - qL, kL).reshape(qL, 1)
        cols = mx.arange(kL).reshape(1, kL)
        s = mx.where(cols <= rows, s, float("-inf"))
    if mask is not None:
        s = mx.where(mask, s, float("-inf"))
    if sinks is not None:
        sink_col = mx.broadcast_to(
            sinks.astype(mx.float32).reshape(1, -1, 1, 1),
            (*s.shape[:-1], 1),
        )
        s = mx.concatenate([s, sink_col], axis=-1)
    w = mx.softmax(s, axis=-1)
    if sinks is not None:
        w = w[..., :-1]
    return (w @ vr).astype(q.dtype)


def _rel(a, b):
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.linalg.norm(af - bf) / (mx.linalg.norm(bf) + 1e-9))


def _make(B, Hq, Hkv, qL, kL, D, dtype, seed, strided):
    key = mx.random.key(seed)
    k0, k1, k2 = mx.random.split(key, 3)
    q = mx.random.normal((B, Hq, qL, D), key=k0).astype(dtype)
    if strided:
        # Allocate a longer seq dim and slice so the head stride is maxL*D (>
        # kL*D) and the seq stride stays D -- the RotatingKVCache prefix layout.
        maxL = kL + 256
        kf = mx.random.normal((B, Hkv, maxL, D), key=k1).astype(dtype)
        vf = mx.random.normal((B, Hkv, maxL, D), key=k2).astype(dtype)
        mx.eval(kf, vf)
        k, v = kf[:, :, :kL, :], vf[:, :, :kL, :]
    else:
        k = mx.random.normal((B, Hkv, kL, D), key=k1).astype(dtype)
        v = mx.random.normal((B, Hkv, kL, D), key=k2).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _eval_or_skip(*arrays):
    # Materialize op + reference; a device whose pipeline caps the dispatch
    # width raises the informative eval_gpu guard error -> capability skip,
    # never a silent-garbage numerics failure.
    try:
        mx.eval(*arrays)
    except RuntimeError as e:
        if "pipeline limit" in str(e):
            pytest.skip(str(e))
        raise


def _check(D, qL, kL, dtype, Hq=32, Hkv=16, strided=False):
    causal = qL > 1  # qL==1 attends all keys; offset-causal is the verify regime
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, Hq, Hkv, qL, kL, D, dtype, seed=qL * 7 + kL + D, strided=strided)
    got = kq.sdpa_vector(q, k, v, scale, causal=causal)
    ref = _ref_sdpa(q, k, v, scale, causal)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    tag = "strided" if strided else "contig"
    print(
        f"  [sdpa] D={D} qL={qL} kL={kL} {str(dtype)[9:]:>9} {tag}: "
        f"rel={rel:.3e} (bound {bound:.0e})"
    )
    assert rel < bound, f"D={D} qL={qL} kL={kL} {dtype} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("qL", [1, 2, 5])
def test_sdpa_vector(D, qL, dtype):
    _check(D, qL, kL=8192, dtype=dtype)


@pytest.mark.parametrize("D", [256, 512])
def test_sdpa_vector_strided_kv(D):
    # The no-copy strided-prefix path at the verify width (offset-causal).
    _check(D, qL=5, kL=4096, dtype=mx.bfloat16, strided=True)


@pytest.mark.parametrize("Hq,Hkv", [(32, 32), (32, 8)])
def test_sdpa_vector_gqa(Hq, Hkv):
    _check(512, qL=4, kL=2048, dtype=mx.bfloat16, Hq=Hq, Hkv=Hkv)


def _ref_sdpa_sinks(q, k, v, scale, sinks):
    """f32 reference with per-q-head sink logits: an extra softmax column
    with no value row (raises the max / adds to the denominator only).
    qL > 1 is offset-causal (query row i attends keys <= kL - qL + i)."""
    g = q.shape[1] // k.shape[1]
    kr = mx.repeat(k, g, axis=1).astype(mx.float32)
    vr = mx.repeat(v, g, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) @ kr.swapaxes(-1, -2)) * scale  # [B,Hq,qL,kL]
    qL, kL = q.shape[2], k.shape[2]
    if qL > 1:
        rows = mx.arange(kL - qL, kL).reshape(qL, 1)
        cols = mx.arange(kL).reshape(1, kL)
        s = mx.where(cols <= rows, s, float("-inf"))
    if sinks is not None:
        col = mx.broadcast_to(
            sinks.astype(mx.float32).reshape(1, -1, 1, 1),
            (*s.shape[:3], 1),
        )
        s = mx.concatenate([s, col], axis=-1)
    w = mx.softmax(s, axis=-1)
    if sinks is not None:
        w = w[..., :-1]
    return (w @ vr).astype(q.dtype)


def _check_gqa(
    D,
    kL,
    dtype,
    Hq=24,
    Hkv=4,
    tile_c=0,
    sinks=False,
    strided=False,
    splits=0,
    qL=1,
):
    scale = 1.0 / (D**0.5)
    q, k, v = _make(1, Hq, Hkv, qL, kL, D, dtype, seed=kL + D, strided=strided)
    sk = None
    if sinks:
        sk = mx.random.normal((Hq,), key=mx.random.key(D + 1)).astype(mx.float32)
        mx.eval(sk)
    got = kq.sdpa_decode_gqa(q, k, v, scale, sinks=sk, splits=splits, tile_c=tile_c)
    ref = _ref_sdpa_sinks(q, k, v, scale, sk)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    print(
        f"  [gqa] D={D} qL={qL} kL={kL} Hq/Hkv={Hq}/{Hkv} c={tile_c} "
        f"sinks={sinks} {str(dtype)[9:]:>9}: rel={rel:.3e}"
    )
    assert rel < bound, f"D={D} kL={kL} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "D,tile_c",
    [(64, 32), (64, 16), (128, 32), (128, 16), (256, 16), (256, 8), (512, 8)],
)
def test_sdpa_decode_gqa(D, tile_c, dtype):
    _check_gqa(D, kL=4096, dtype=dtype, tile_c=tile_c)


@pytest.mark.parametrize("Hq,Hkv", [(24, 4), (16, 4), (32, 8), (8, 8)])
def test_sdpa_decode_gqa_factors(Hq, Hkv):
    _check_gqa(256, kL=2048, dtype=mx.bfloat16, Hq=Hq, Hkv=Hkv)


@pytest.mark.parametrize("Hq,Hkv,D", [(16, 1, 512), (32, 4, 512)])
def test_sdpa_decode_gqa_wide_factor(Hq, Hkv, D):
    # gemma-4 12b/31b global-layer geometry (gqa 16 / 8 at hd512)
    _check_gqa(D, kL=2048, dtype=mx.bfloat16, Hq=Hq, Hkv=Hkv)


@pytest.mark.parametrize("D", [64, 128, 256, 512])
def test_sdpa_decode_gqa_sinks(D):
    _check_gqa(D, kL=2048, dtype=mx.bfloat16, sinks=True)


@pytest.mark.parametrize("D", [64, 256, 512])
def test_sdpa_decode_gqa_strided_unaligned(D):
    # strided KV-cache prefix + a key length off every tile/split boundary
    _check_gqa(D, kL=3071, dtype=mx.bfloat16, strided=True, splits=16)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("D", [64, 128, 256, 512])
@pytest.mark.parametrize("qL", [2, 3, 4])
def test_sdpa_gqa_verify(D, qL, dtype):
    # speculative-verify width: offset-causal queries share the staged tiles
    _check_gqa(D, kL=4096, dtype=dtype, qL=qL)


def test_sdpa_gqa_verify_gemma_geometry():
    # gemma-4-31b global layers at verify width (gqa 8 at hd512)
    _check_gqa(512, kL=8192, dtype=mx.bfloat16, Hq=32, Hkv=4, qL=4)


def test_sdpa_gqa_verify_sinks():
    _check_gqa(64, kL=2048, dtype=mx.bfloat16, sinks=True, qL=4)


@pytest.mark.parametrize("D", [64, 512])
def test_sdpa_gqa_verify_strided_unaligned(D):
    _check_gqa(D, kL=3071, dtype=mx.bfloat16, strided=True, splits=16, qL=4)


def test_sdpa_gqa_verify_short_kv():
    # kL small enough that most splits stage zero keys and whole tiles fall
    # beyond a query's causal limit: exercises the empty-split partials and
    # the fully-invalid-tile guard (finite_min max would otherwise poison
    # the sum with exp(0) terms).
    _check_gqa(64, kL=17, dtype=mx.bfloat16, splits=16, qL=4)


def _ref_sdpa_fold(q, k, v, scale, q_len):
    """f32 reference for the GQA-folded verify layout: q [B, Hkv, G*qL, D]
    attends its own kv head directly; folded row r is causally clamped to
    key <= kL - qL + (r % qL)."""
    s = (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)) * scale
    n_rows, kL = q.shape[2], k.shape[2]
    lims = (kL - q_len + mx.arange(n_rows) % q_len).reshape(n_rows, 1)
    cols = mx.arange(kL).reshape(1, kL)
    s = mx.where(cols <= lims, s, float("-inf"))
    w = mx.softmax(s, axis=-1)
    return (w @ v.astype(mx.float32)).astype(q.dtype)


def _check_fa(D, qL, kL, dtype, Hkv=4, G=6, strided=False, splits=0):
    n_rows = G * qL
    scale = 1.0 / (D**0.5)
    q, k, v = _make(
        1, Hkv, Hkv, n_rows, kL, D, dtype, seed=qL * 13 + kL + D, strided=strided
    )
    got = kq.sdpa_fa_verify(q, k, v, scale, q_len=qL, splits=splits)
    ref = _ref_sdpa_fold(q, k, v, scale, qL)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = REL_BOUND[dtype]
    tag = "strided" if strided else "contig"
    print(
        f"  [fa] D={D} qL={qL} G={G} kL={kL} Hkv={Hkv} {str(dtype)[9:]:>9} "
        f"{tag}: rel={rel:.3e}"
    )
    assert rel < bound, f"D={D} qL={qL} G={G} kL={kL} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("qL", [2, 3, 4, 5, 6])
def test_sdpa_fa_verify(qL, dtype):
    _check_fa(256, qL, kL=4096, dtype=dtype, G=4)


def test_sdpa_fa_verify_qwen_geometry():
    # qwen3.5/3.6 full-attn fold: 24 rows = G6 x qL4 at hd256
    _check_fa(256, 4, kL=8192, dtype=mx.bfloat16, Hkv=4, G=6)


def test_sdpa_fa_verify_full_tile():
    # n_rows == 32 fills the tile exactly (no padding rows), qL at the cap
    _check_fa(256, 8, kL=4096, dtype=mx.bfloat16, Hkv=2, G=4)


def test_sdpa_fa_verify_partial_warp():
    # n_rows == 18: the third simdgroup covers rows 16..17 plus padding
    _check_fa(256, 6, kL=2048, dtype=mx.bfloat16, G=3)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_sdpa_fa_verify_strided_unaligned(dtype):
    # strided KV-cache prefix + a key length off every tile/split boundary
    _check_fa(256, 4, kL=3071, dtype=dtype, strided=True, splits=16)


def test_sdpa_fa_verify_short_kv():
    # kL small enough that most splits stage zero keys: their empty partials
    # (max = finite_min, sum = 0) must merge as weight zero
    _check_fa(256, 4, kL=17, dtype=mx.bfloat16, splits=16, G=6)


def test_sdpa_fa_verify_min_kv():
    # kL == qL floor: row 0 attends exactly one key, every row masked hard
    _check_fa(256, 4, kL=4, dtype=mx.bfloat16, G=6)


def test_sdpa_fa_verify_causal_split_straddle():
    # the last qL keys straddle a split boundary (splits=128, kL=4098 puts
    # keys 4096..4097 alone in the final split): that split is entirely past
    # the low rows' causal limits, exercising the dead-row guard in a
    # non-empty split
    _check_fa(256, 4, kL=4098, dtype=mx.bfloat16, splits=128, G=6)


# --- FA prefill (dense KV) ---------------------------------------------------

# The prefill kernel keeps queries in [1, Hq, qL, D] and tiles the query axis on
# the grid, so it needs a full-chunk reference. Float32 is the served query
# dtype and gets the tightest bound (fp32-grade); the half arms carry the wider
# rounding bound.
PREFILL_REL_BOUND = {mx.float32: 5e-6, mx.bfloat16: 5e-3, mx.float16: 2e-3}


def _ref_prefill(q, k, v, scale):
    """f32 materialized prefill attention: query position p (of qL) attends
    keys <= (kL - qL) + p, so the past prefix is unmasked and the self chunk
    causal -- the single band the composed prefill path applies."""
    g = q.shape[1] // k.shape[1]
    kr = mx.repeat(k, g, axis=1).astype(mx.float32)
    vr = mx.repeat(v, g, axis=1).astype(mx.float32)
    qL, kL = q.shape[2], k.shape[2]
    s = (q.astype(mx.float32) @ kr.swapaxes(-1, -2)) * scale
    rows = ((kL - qL) + mx.arange(qL)).reshape(qL, 1)
    cols = mx.arange(kL).reshape(1, kL)
    s = mx.where(cols <= rows, s, float("-inf"))
    w = mx.softmax(s, axis=-1)
    return (w @ vr).astype(q.dtype)


def _check_prefill(D, qL, kL, dtype, Hq=16, Hkv=2, qw=0, strided=False, splits=0):
    scale = 1.0 / (D**0.5)
    q, k, v = _make(
        1, Hq, Hkv, qL, kL, D, dtype, seed=qL * 11 + kL + D, strided=strided
    )
    got = kq.sdpa_fa_prefill(q, k, v, scale, qw=qw, splits=splits)
    ref = _ref_prefill(q, k, v, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = PREFILL_REL_BOUND[dtype]
    tag = "strided" if strided else "contig"
    print(
        f"  [prefill] D={D} qL={qL} kL={kL} Hq={Hq} Hkv={Hkv} qw={qw} "
        f"{str(dtype)[9:]:>9} {tag}: rel={rel:.3e}"
    )
    assert rel < bound, f"D={D} qL={qL} kL={kL} rel {rel:.3e} >= {bound:.0e}"
    assert got.shape == q.shape


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize("kL", [2048, 4096, 8192])
def test_sdpa_fa_prefill_serving_geometry(dtype, kL):
    # The 16:2 GQA, head-dim-256 serving geometry: 16 q heads, 2 kv heads,
    # chunk 2048.
    _check_prefill(256, qL=2048, kL=kL, dtype=dtype)


def test_sdpa_fa_prefill_chunk0():
    # Chunk 0 form: the whole chunk attends only itself (qL == kL).
    _check_prefill(256, qL=2048, kL=2048, dtype=mx.float32)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sdpa_fa_prefill_unaligned(dtype):
    # qL and kL off every tile/split boundary (partial query tile, partial key
    # tile, partial split tail).
    _check_prefill(256, qL=100, kL=133, dtype=dtype, splits=16)
    _check_prefill(256, qL=37, kL=37, dtype=dtype, splits=16)


def test_sdpa_fa_prefill_decode_shape():
    # qL == 1 decode shape: one query attends the whole prefix (no causal cut).
    _check_prefill(256, qL=1, kL=8192, dtype=mx.float32)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_sdpa_fa_prefill_strided(dtype):
    # strided KV-cache prefix (head stride > kL*D) read in place without a copy.
    _check_prefill(256, qL=512, kL=3071, dtype=dtype, strided=True, splits=16)


def test_sdpa_fa_prefill_qw_fold():
    # Explicit qw at the 16:2 GQA serving fold (G=8 -> qw 4 fills a 32-row
    # tile).
    _check_prefill(256, qL=1000, kL=3000, dtype=mx.float32, qw=4)


def test_sdpa_fa_prefill_gqa_factors():
    # A 4:1 fold (G=4 -> qw default 8) and a 16:1 fold (G=16 -> qw default 2).
    _check_prefill(256, qL=300, kL=1000, dtype=mx.float32, Hq=8, Hkv=2)
    _check_prefill(256, qL=200, kL=900, dtype=mx.float32, Hq=16, Hkv=1)


def test_sdpa_fa_prefill_short_kv():
    # kL small enough that most splits stage zero keys: empty-split partials
    # (max = finite_min, sum = 0) must merge as weight zero.
    _check_prefill(256, qL=8, kL=17, dtype=mx.float32, splits=16)


def test_sdpa_fa_prefill_rejects_bad_geometry():
    scale = 1.0 / 16.0
    q = mx.zeros((1, 16, 8, 256))
    k = mx.zeros((1, 2, 32, 256))
    v = mx.zeros((1, 2, 32, 256))
    # head_dim off contract.
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill(mx.zeros((1, 16, 8, 128)), mx.zeros((1, 2, 32, 128)),
                           mx.zeros((1, 2, 32, 128)), scale)
    # batch > 1.
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill(mx.zeros((2, 16, 8, 256)), mx.zeros((2, 2, 32, 256)),
                           mx.zeros((2, 2, 32, 256)), scale)
    # qw that does not fill a 32-row tile for the GQA factor (G=8 needs qw 4).
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill(q, k, v, scale, qw=3)
    # key length shorter than the query chunk.
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill(mx.zeros((1, 16, 64, 256)), mx.zeros((1, 2, 8, 256)),
                           mx.zeros((1, 2, 8, 256)), scale)


# --- FA prefill, q8 past phase (the serving form) ----------------------------

# Staging-precision bounds: float32 staging carries no staging round, so it
# holds fp32 grade and doubles as the in-kernel dequant exactness check (a
# packing or fma defect would blow it up by orders of magnitude). The half
# stages carry their rounding.
Q8_STAGE_BOUND = {2: 5e-6, 1: 2e-3, 0: 8e-3}


def _make_q8_case(Hq, Hkv, qL, Lp, seed, strided):
    key = mx.random.key(seed)
    ks = mx.random.split(key, 6)
    q = mx.random.normal((1, Hq, qL, 256), key=ks[0]).astype(mx.float32)
    self_k = mx.random.normal((1, Hkv, qL, 256), key=ks[1]).astype(mx.float32)
    self_v = mx.random.normal((1, Hkv, qL, 256), key=ks[2]).astype(mx.bfloat16)
    if strided:
        # The QuantizedKVCache state form: a seq-sliced view of a longer
        # buffer, so the head stride exceeds Lp * row.
        cap = Lp + 256
        pk = tuple(x[..., :Lp, :] for x in mx.quantize(
            mx.random.normal((1, Hkv, cap, 256), key=ks[3]).astype(mx.float32),
            group_size=64, bits=8))
        pv = tuple(x[..., :Lp, :] for x in mx.quantize(
            mx.random.normal((1, Hkv, cap, 256), key=ks[4]).astype(mx.float32),
            group_size=64, bits=8))
    else:
        pk = mx.quantize(
            mx.random.normal((1, Hkv, Lp, 256), key=ks[3]).astype(mx.float32),
            group_size=64, bits=8)
        pv = mx.quantize(
            mx.random.normal((1, Hkv, Lp, 256), key=ks[4]).astype(mx.float32),
            group_size=64, bits=8)
    mx.eval(q, self_k, self_v, list(pk), list(pv))
    return q, pk, pv, self_k, self_v


def _ref_q8_prefill(q, pk, pv, self_k, self_v, scale):
    """f32 composed reference over the mx-dequantized past concatenated with
    the dense self chunk; query p attends keys <= Lp + p."""
    Hq, Hkv = q.shape[1], pk[0].shape[1]
    Lp, qL = pk[0].shape[2], q.shape[2]
    g = Hq // Hkv
    kf = mx.concatenate(
        [mx.dequantize(*pk, group_size=64, bits=8),
         self_k.astype(mx.float32)], axis=2)
    vf = mx.concatenate(
        [mx.dequantize(*pv, group_size=64, bits=8),
         self_v.astype(mx.float32)], axis=2)
    kr = mx.repeat(kf, g, axis=1)
    vr = mx.repeat(vf, g, axis=1)
    s = (q * scale) @ kr.swapaxes(-1, -2)
    kL = Lp + qL
    rows = (Lp + mx.arange(qL)).reshape(qL, 1)
    cols = mx.arange(kL).reshape(1, kL)
    s = mx.where(cols <= rows, s, float("-inf"))
    return mx.softmax(s, axis=-1, precise=True) @ vr


def _check_q8(Hq, Hkv, qL, Lp, stage, seed=0, strided=False, qw=0, bq=0,
              bk=0, splits=0):
    scale = 1.0 / 16.0
    q, pk, pv, self_k, self_v = _make_q8_case(Hq, Hkv, qL, Lp, seed, strided)
    got = kq.sdpa_fa_prefill_q8(q, *pk, *pv, self_k, self_v, scale,
                                qw=qw, bq=bq, bk=bk, splits=splits,
                                stage=stage)
    ref = _ref_q8_prefill(q, pk, pv, self_k, self_v, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    bound = Q8_STAGE_BOUND[stage]
    print(
        f"  [prefill-q8] Hq={Hq} Hkv={Hkv} qL={qL} Lp={Lp} stage={stage} "
        f"bq={bq or 32} bk={bk or 'auto'}: rel={rel:.3e}"
    )
    assert rel < bound, f"stage={stage} rel {rel:.3e} >= {bound:.0e}"
    assert got.dtype == mx.float32
    assert got.shape == q.shape


@pytest.mark.parametrize("bq", [32, 64])
def test_sdpa_fa_prefill_q8_dequant_exact(bq):
    # Float32 staging carries no staging round: fp32-grade agreement with the
    # mx.dequantize-based reference proves the in-kernel fma dequant matches
    # mx dequantization on real packed words. BQ=64 is the shipping-default
    # width for the float32 staging (BK pins at 16 for both widths).
    _check_q8(16, 2, qL=512, Lp=2048, stage=2, bq=bq)
    _check_q8(16, 2, qL=100, Lp=97, stage=2, bq=bq)


@pytest.mark.parametrize("stage", [0, 1])
def test_sdpa_fa_prefill_q8_half_stages(stage):
    _check_q8(16, 2, qL=512, Lp=2048, stage=stage)


@pytest.mark.parametrize("stage", [0, 1])
def test_sdpa_fa_prefill_q8_serving_width(stage):
    # BQ=64 with BK=48: the serving tile widths for the 16:2 GQA, head-dim-256
    # serving geometry.
    _check_q8(16, 2, qL=512, Lp=2048, stage=stage, bq=64, bk=48)
    _check_q8(16, 2, qL=100, Lp=97, stage=stage, bq=64, bk=48)


def test_sdpa_fa_prefill_q8_unaligned_and_strided():
    _check_q8(16, 2, qL=33, Lp=64, stage=1, splits=16)
    _check_q8(16, 2, qL=512, Lp=2048, stage=1, strided=True, bq=64, bk=48)


def test_sdpa_fa_prefill_q8_half_precision_scales():
    # A float32-key, bfloat16-value cache stores bfloat16 V scales and biases
    # (the to_quantized conversion path); the op casts them to float32, which
    # is value-preserving, so the result matches the reference over the same
    # dequantized values.
    key = mx.random.key(7)
    ks = mx.random.split(key, 5)
    q = mx.random.normal((1, 16, 128, 256), key=ks[0]).astype(mx.float32)
    self_k = mx.random.normal((1, 2, 128, 256), key=ks[1]).astype(mx.float32)
    self_v = mx.random.normal((1, 2, 128, 256), key=ks[2]).astype(mx.bfloat16)
    pk = mx.quantize(
        mx.random.normal((1, 2, 256, 256), key=ks[3]).astype(mx.float32),
        group_size=64, bits=8)
    pv = mx.quantize(
        mx.random.normal((1, 2, 256, 256), key=ks[4]).astype(mx.bfloat16),
        group_size=64, bits=8)
    assert pv[1].dtype == mx.bfloat16
    mx.eval(q, self_k, self_v, list(pk), list(pv))
    scale = 1.0 / 16.0
    got = kq.sdpa_fa_prefill_q8(q, *pk, *pv, self_k, self_v, scale,
                                bq=64, bk=48, stage=1)
    ref = _ref_q8_prefill(q, pk, pv, self_k, self_v, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    print(f"  [prefill-q8] bf16 V scales: rel={rel:.3e}")
    assert rel < Q8_STAGE_BOUND[1]


def test_sdpa_fa_prefill_q8_rejects_bad_contract():
    scale = 1.0 / 16.0
    q, pk, pv, self_k, self_v = _make_q8_case(16, 2, qL=8, Lp=64, seed=1,
                                              strided=False)
    # group size / bits off contract.
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill_q8(q, *pk, *pv, self_k, self_v, scale,
                              group_size=32)
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill_q8(q, *pk, *pv, self_k, self_v, scale, bits=4)
    # non-float32 queries.
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill_q8(q.astype(mx.bfloat16), *pk, *pv, self_k,
                              self_v, scale)
    # empty past (the dense kernel serves an empty cache).
    with pytest.raises(ValueError):
        empty_pk = tuple(x[..., :0, :] for x in pk)
        empty_pv = tuple(x[..., :0, :] for x in pv)
        kq.sdpa_fa_prefill_q8(q, *empty_pk, *empty_pv, self_k, self_v, scale)
    # bk not instantiated for the stage (float staging is bk 16 only).
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill_q8(q, *pk, *pv, self_k, self_v, scale, bk=48,
                              stage=2)
    # bk 48 requires the 64-row tile.
    with pytest.raises(ValueError):
        kq.sdpa_fa_prefill_q8(q, *pk, *pv, self_k, self_v, scale, bk=48,
                              bq=32, stage=1)


# --- fused q8 decode attention (the served KV-attention read) ----------------

# Staging-precision bounds: float32 staging carries no staging round, so it
# holds fp32 grade and doubles as the in-kernel dequant exactness check (a
# packing or fma defect blows it up by orders of magnitude). The half stages
# carry their rounding.
Q8_DECODE_BOUND = {2: 5e-6, 1: 2e-3, 0: 8e-3}


def _make_q8_decode_case(Hq, Hkv, N, seed, strided):
    key = mx.random.key(seed)
    ks = mx.random.split(key, 4)
    q = mx.random.normal((1, Hq, 1, 256), key=ks[0]).astype(mx.float32)
    if strided:
        # The QuantizedKVCache state form: a seq-sliced view of a longer buffer,
        # so the head stride exceeds N * row.
        cap = N + 256
        pk = tuple(x[..., :N, :] for x in mx.quantize(
            mx.random.normal((1, Hkv, cap, 256), key=ks[1]).astype(mx.float32),
            group_size=64, bits=8))
        pv = tuple(x[..., :N, :] for x in mx.quantize(
            mx.random.normal((1, Hkv, cap, 256), key=ks[2]).astype(mx.float32),
            group_size=64, bits=8))
    else:
        pk = mx.quantize(
            mx.random.normal((1, Hkv, N, 256), key=ks[1]).astype(mx.float32),
            group_size=64, bits=8)
        pv = mx.quantize(
            mx.random.normal((1, Hkv, N, 256), key=ks[2]).astype(mx.float32),
            group_size=64, bits=8)
    mx.eval(q, list(pk), list(pv))
    return q, pk, pv


def _ref_q8_decode(q, pk, pv, scale):
    """f32 composed reference over the mx-dequantized cache: one query attends
    every key (no causal cut)."""
    Hq, Hkv = q.shape[1], pk[0].shape[1]
    g = Hq // Hkv
    kf = mx.dequantize(*pk, group_size=64, bits=8)
    vf = mx.dequantize(*pv, group_size=64, bits=8)
    kr = mx.repeat(kf, g, axis=1)
    vr = mx.repeat(vf, g, axis=1)
    s = (q * scale) @ kr.swapaxes(-1, -2)
    return mx.softmax(s, axis=-1, precise=True) @ vr


def _check_q8_decode(
    N,
    stage=2,
    Hq=16,
    Hkv=2,
    seed=0,
    strided=False,
    splits=0,
    compute=1,
    tile_c=0,
    dimension_parallel_merge=False,
):
    scale = 1.0 / 16.0
    q, pk, pv = _make_q8_decode_case(Hq, Hkv, N, seed, strided)
    got = kq.sdpa_decode_q8(
        q,
        *pk,
        *pv,
        scale,
        splits=splits,
        stage=stage,
        compute=compute,
        tile_c=tile_c,
        dimension_parallel_merge=dimension_parallel_merge,
    )
    ref = _ref_q8_decode(q, pk, pv, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    # The SIMD-shuffle path (compute 1) always stages float; only compute 0
    # carries the stage-precision round, so the bound tracks the actual staging.
    bound = 5e-6 if compute == 1 else Q8_DECODE_BOUND[stage]
    print(
        f"  [decode-q8] N={N:6d} compute={compute} stage={stage} "
        f"splits={splits or 'auto'} tile_c={tile_c or 'auto'}: rel={rel:.3e}"
    )
    assert rel < bound, f"N={N} compute={compute} rel {rel:.3e} >= {bound:.0e}"
    assert got.dtype == mx.float32
    assert got.shape == q.shape


# Past lengths spanning group-64 boundaries, split-count edges, and depth. The
# float32 staging arm doubles as the in-kernel dequant bit-exactness check.
@pytest.mark.parametrize(
    "N", [63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 4096, 4123, 37000]
)
def test_sdpa_decode_q8_edges_f32(N):
    # Matrix-tile compute (compute 0), float32 staging: the dequant-exactness
    # arm and the split/group edge sweep.
    _check_q8_decode(N, stage=2, seed=N, compute=0)


# The SIMD-shuffle compute (compute 1) is the decode-latency serving form; it
# stages float and dequantizes during the cooperative tile load, so its
# fp32-grade agreement doubles as its own dequant-exactness check across the
# same group-64 and split-count edges.
@pytest.mark.parametrize(
    "N", [63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 4096, 4123, 37000]
)
def test_sdpa_decode_q8_edges_shuffle(N):
    _check_q8_decode(N, seed=N, compute=1)


@pytest.mark.parametrize("tile_c", [8, 16])
def test_sdpa_decode_q8_shuffle_tile_c(tile_c):
    _check_q8_decode(4096, seed=51 + tile_c, compute=1, tile_c=tile_c)
    _check_q8_decode(3071, seed=61 + tile_c, compute=1, tile_c=tile_c, splits=16)


def test_sdpa_decode_q8_shuffle_tile16_long_context_engages_loader():
    loader_before = kq.sdpa_q8_loader_debug()
    _check_q8_decode(
        37000,
        seed=33035,
        compute=1,
        tile_c=16,
        splits=128,
        dimension_parallel_merge=True,
    )
    loader_after = kq.sdpa_q8_loader_debug()
    selected = loader_before["selected_arm"]
    assert loader_after[selected] == loader_before[selected] + 1
    assert loader_after["off_contract"] == loader_before["off_contract"]


def test_sdpa_decode_q8_shuffle_strided_and_splits():
    _check_q8_decode(2048, compute=1, strided=True, seed=71)
    for splits in (1, 8, 16, 64):
        _check_q8_decode(3071, compute=1, seed=80 + splits, splits=splits)


def test_sdpa_decode_q8_shuffle_matches_matrix_tile():
    # Both compute paths attend the same q8 values at fp32 grade, so they agree
    # with each other within accumulation-order noise on served shapes.
    scale = 1.0 / 16.0
    q, pk, pv = _make_q8_decode_case(16, 2, 4096, seed=123, strided=False)
    got0 = kq.sdpa_decode_q8(q, *pk, *pv, scale, compute=0, stage=2)
    got1 = kq.sdpa_decode_q8(q, *pk, *pv, scale, compute=1)
    _eval_or_skip(got0, got1)
    rel = _rel(got1, got0)
    print(f"  [decode-q8] shuffle vs matrix-tile: rel={rel:.3e}")
    assert rel < 5e-6


def test_sdpa_decode_q8_dimension_parallel_merge_is_exact():
    N = 8192
    scale = 1.0 / 16.0
    q, pk, pv = _make_q8_decode_case(16, 2, N, seed=19001 + N, strided=False)
    shared = kq.sdpa_decode_q8(
        q, *pk, *pv, scale, splits=128, stage=2, compute=1, tile_c=16
    )
    parallel = kq.sdpa_decode_q8(
        q,
        *pk,
        *pv,
        scale,
        splits=128,
        stage=2,
        compute=1,
        tile_c=16,
        dimension_parallel_merge=True,
    )
    _eval_or_skip(shared, parallel)
    assert bool(mx.array_equal(shared, parallel))


def test_sdpa_decode_q8_uint4_alignment_fallback_is_exact():
    loader_before = kq.sdpa_q8_loader_debug()
    selected = loader_before["selected_arm"]
    if selected not in {"uint4_dynamic", "uint4_byte_dynamic"}:
        pytest.skip("the scalar kill switch is active")

    scale = 1.0 / 16.0
    q, pk, pv = _make_q8_decode_case(16, 2, 257, seed=33034, strided=False)

    def unaligned_words(words):
        prefix = mx.zeros((*words.shape[:-1], 1), dtype=words.dtype)
        return mx.concatenate((prefix, words), axis=-1)[..., 1:]

    unaligned_pk = (unaligned_words(pk[0]), *pk[1:])
    unaligned_pv = (unaligned_words(pv[0]), *pv[1:])
    mx.eval(*unaligned_pk, *unaligned_pv)

    kwargs = {"splits": 16, "stage": 2, "compute": 1, "tile_c": 16}
    aligned = kq.sdpa_decode_q8(q, *pk, *pv, scale, **kwargs)
    mx.eval(aligned)
    loader_after_aligned = kq.sdpa_q8_loader_debug()
    unaligned = kq.sdpa_decode_q8(
        q, *unaligned_pk, *unaligned_pv, scale, **kwargs
    )
    mx.eval(unaligned)
    loader_after_unaligned = kq.sdpa_q8_loader_debug()

    assert bool(mx.array_equal(aligned, unaligned))
    assert (
        loader_after_aligned[selected]
        == loader_before[selected] + 1
    )
    assert (
        loader_after_unaligned["scalar_dynamic"]
        == loader_after_aligned["scalar_dynamic"] + 1
    )
    assert (
        loader_after_unaligned["off_contract"]
        == loader_after_aligned["off_contract"] + 1
    )


_Q8_UNPACK_HASH = textwrap.dedent(
    """
    import hashlib

    import mlx.core as mx
    import mlx_kquant as kq
    import numpy as np

    length = 8191
    capacity = length + 17
    keys = mx.random.split(mx.random.key(39002), 3)
    query = mx.random.normal((1, 16, 1, 256), key=keys[0]).astype(mx.float32)
    key = mx.random.normal((1, 2, capacity, 256), key=keys[1]).astype(mx.float32)
    value = mx.random.normal((1, 2, capacity, 256), key=keys[2]).astype(mx.float32)
    packed_key = tuple(
        part[:, :, :length, :]
        for part in mx.quantize(key, group_size=64, bits=8)
    )
    packed_value = tuple(
        part[:, :, :length, :]
        for part in mx.quantize(value, group_size=64, bits=8)
    )
    counters_before = kq.sdpa_q8_loader_debug()
    output = kq.sdpa_decode_q8(
        query,
        *packed_key,
        *packed_value,
        1.0 / 16.0,
        group_size=64,
        bits=8,
        splits=128,
        stage=2,
        compute=1,
        tile_c=16,
        dimension_parallel_merge=True,
    )
    mx.eval(output)
    counters_after = kq.sdpa_q8_loader_debug()
    selected = counters_after["selected_arm"]
    print(selected)
    print(counters_after[selected] - counters_before[selected])
    print(counters_after["scalar_dynamic"] - counters_before["scalar_dynamic"])
    print(counters_after["off_contract"] - counters_before["off_contract"])
    print(hashlib.sha256(np.asarray(output).tobytes()).hexdigest())
    """
)


def _q8_unpack_hash(vector_unpack: str) -> tuple[str, int, int, int, str]:
    env = dict(os.environ)
    env["KQ_SDPA_Q8_UINT4_LOAD"] = "1"
    env["KQ_SDPA_Q8_VECTOR_BYTE_UNPACK"] = vector_unpack
    proc = subprocess.run(
        [sys.executable, "-c", _Q8_UNPACK_HASH],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    arm, selected_calls, scalar_calls, off_contract, output_hash = (
        proc.stdout.strip().splitlines()
    )
    return (
        arm,
        int(selected_calls),
        int(scalar_calls),
        int(off_contract),
        output_hash,
    )


def test_sdpa_decode_q8_vector_byte_unpack_is_bitwise_exact():
    if mx.default_device() == mx.cpu:
        pytest.skip("the q8 decode loader is a GPU kernel path")
    control_arm, control_calls, control_scalar, control_off, control_hash = (
        _q8_unpack_hash("0")
    )
    candidate_arm, candidate_calls, candidate_scalar, candidate_off, candidate_hash = (
        _q8_unpack_hash("1")
    )
    assert control_arm == "uint4_dynamic"
    assert candidate_arm == "uint4_byte_dynamic"
    assert control_calls == candidate_calls == 1
    assert control_scalar == candidate_scalar == 0
    assert control_off == candidate_off == 0
    assert candidate_hash == control_hash


@pytest.mark.parametrize(
    ("N", "kwargs"),
    [
        (8191, {"splits": 128, "stage": 2, "compute": 1, "tile_c": 16}),
        (8192, {"splits": 64, "stage": 2, "compute": 1, "tile_c": 16}),
        (8192, {"splits": 128, "stage": 2, "compute": 1, "tile_c": 8}),
        (8192, {"splits": 128, "stage": 1, "compute": 1, "tile_c": 16}),
    ],
)
def test_sdpa_decode_q8_dimension_parallel_merge_falls_back(N, kwargs):
    scale = 1.0 / 16.0
    q, pk, pv = _make_q8_decode_case(16, 2, N, seed=19002, strided=False)
    shared = kq.sdpa_decode_q8(q, *pk, *pv, scale, **kwargs)
    fallback = kq.sdpa_decode_q8(
        q, *pk, *pv, scale, dimension_parallel_merge=True, **kwargs
    )
    _eval_or_skip(shared, fallback)
    assert bool(mx.array_equal(shared, fallback))


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("N", [64, 2048, 4096])
def test_sdpa_decode_q8_half_stages(stage, N):
    # Matrix-tile staging precision (compute 0); the SIMD-shuffle path stages
    # float only.
    _check_q8_decode(N, stage=stage, seed=N + 3, compute=0)


def test_sdpa_decode_q8_strided():
    # A seq-sliced cache view (head stride > N * row): the op reads it in place.
    _check_q8_decode(2048, stage=2, strided=True, seed=91, compute=0)
    _check_q8_decode(4097, stage=1, strided=True, seed=92, compute=0)


def test_sdpa_decode_q8_explicit_splits():
    # Off-boundary depths at several fixed split counts (empty-split partials at
    # the tail must merge with weight zero).
    for splits in (1, 8, 16, 64):
        _check_q8_decode(3071, stage=2, seed=100 + splits, splits=splits,
                         compute=0)


def test_sdpa_decode_q8_dequant_exact():
    # Float32 staging carries no staging round: fp32-grade agreement with the
    # mx.dequantize-based reference proves the in-kernel fma dequant matches mx
    # dequantization on real packed words at a group boundary and past it.
    _check_q8_decode(64, stage=2, seed=7, compute=0)
    _check_q8_decode(65, stage=2, seed=8, compute=0)
    _check_q8_decode(37000, stage=2, seed=9, compute=0)


def test_sdpa_decode_q8_half_precision_scales():
    # A bfloat16-value cache stores bfloat16 V scales and biases (the
    # to_quantized conversion path); the op casts them to float32, which is
    # value-preserving, so the result matches the reference over the same
    # dequantized values.
    key = mx.random.key(21)
    ks = mx.random.split(key, 3)
    q = mx.random.normal((1, 16, 1, 256), key=ks[0]).astype(mx.float32)
    pk = mx.quantize(
        mx.random.normal((1, 2, 512, 256), key=ks[1]).astype(mx.float32),
        group_size=64, bits=8)
    pv = mx.quantize(
        mx.random.normal((1, 2, 512, 256), key=ks[2]).astype(mx.bfloat16),
        group_size=64, bits=8)
    assert pv[1].dtype == mx.bfloat16
    mx.eval(q, list(pk), list(pv))
    scale = 1.0 / 16.0
    got = kq.sdpa_decode_q8(q, *pk, *pv, scale, stage=1, compute=0)
    ref = _ref_q8_decode(q, pk, pv, scale)
    _eval_or_skip(got, ref)
    rel = _rel(got, ref)
    print(f"  [decode-q8] bf16 V scales: rel={rel:.3e}")
    assert rel < Q8_DECODE_BOUND[1]


def test_sdpa_decode_q8_rejects_bad_contract():
    scale = 1.0 / 16.0
    q, pk, pv = _make_q8_decode_case(16, 2, N=64, seed=1, strided=False)
    # group size / bits off contract.
    with pytest.raises(ValueError):
        kq.sdpa_decode_q8(q, *pk, *pv, scale, group_size=32)
    with pytest.raises(ValueError):
        kq.sdpa_decode_q8(q, *pk, *pv, scale, bits=4)
    # non-float32 queries.
    with pytest.raises(ValueError):
        kq.sdpa_decode_q8(q.astype(mx.bfloat16), *pk, *pv, scale)
    # qL > 1 is the prefill kernel's shape.
    with pytest.raises(ValueError):
        q2 = mx.zeros((1, 16, 2, 256), dtype=mx.float32)
        kq.sdpa_decode_q8(q2, *pk, *pv, scale)
    # empty cache.
    with pytest.raises(ValueError):
        empty_pk = tuple(x[..., :0, :] for x in pk)
        empty_pv = tuple(x[..., :0, :] for x in pv)
        kq.sdpa_decode_q8(q, *empty_pk, *empty_pv, scale)
    # off-contract fold (only gqa factor 8 is instantiated).
    with pytest.raises(ValueError):
        q4 = mx.zeros((1, 8, 1, 256), dtype=mx.float32)  # 8 q heads / 2 kv = 4
        kq.sdpa_decode_q8(q4, *pk, *pv, scale)
    # unknown compute mode.
    with pytest.raises(ValueError):
        kq.sdpa_decode_q8(q, *pk, *pv, scale, compute=2)
    # off-contract SIMD-shuffle tile height.
    with pytest.raises(ValueError):
        kq.sdpa_decode_q8(q, *pk, *pv, scale, compute=1, tile_c=32)


@pytest.mark.parametrize("D", [256, 512])
@pytest.mark.parametrize("qL", [1, 4])
def test_sdpa_vector_bool_mask(D, qL):
    """Boolean key mask (true = attend), broadcast over heads."""
    B, Hq, Hkv, kL = 1, 32, 1, 2048
    dtype = mx.float16
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, qL, kL, D, dtype, seed=3 * D + qL, strided=False)
    mkey = mx.random.key(11)
    mask = mx.random.uniform(shape=(B, 1, qL, kL), key=mkey) > 0.3
    mask = mask | (mx.arange(kL).reshape(1, 1, 1, kL) == 0)  # keep >= 1 key
    mx.eval(mask)
    got = kq.sdpa_vector(q, k, v, scale, causal=False, mask=mask)
    ref = _ref_sdpa(q, k, v, scale, causal=False, mask=mask)
    mx.eval(got, ref)
    rel = _rel(got, ref)
    print(f"  [sdpa/mask] D={D} qL={qL}: rel={rel:.3e}")
    assert rel < REL_BOUND[dtype]


@pytest.mark.parametrize("D", [256, 512])
def test_sdpa_vector_sinks(D):
    """Per-query-head attention sinks join the softmax denominator."""
    B, Hq, Hkv, qL, kL = 1, 32, 1, 1, 2048
    dtype = mx.float16
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, qL, kL, D, dtype, seed=D, strided=False)
    skey = mx.random.key(17)
    # Large positive sinks so the denominator term dominates: a sink bug shows
    # up as a big rel error, not noise.
    sinks = (mx.random.normal((Hq,), key=skey) * 2.0 + 4.0).astype(mx.float16)
    mx.eval(sinks)
    got = kq.sdpa_vector(q, k, v, scale, causal=False, sinks=sinks)
    ref = _ref_sdpa(q, k, v, scale, causal=False, sinks=sinks)
    mx.eval(got, ref)
    rel = _rel(got, ref)
    print(f"  [sdpa/sinks] D={D}: rel={rel:.3e}")
    assert rel < REL_BOUND[dtype]


def test_sdpa_vector_mask_and_sinks_mqa_512():
    """The DS4 decode shape: 64 query heads on one shared K=V latent head
    (wide MQA, one head per threadgroup), head dim 512, qL=1, a boolean
    pooled-column mask, and per-head sinks together."""
    B, Hq, Hkv, qL, kL, D = 1, 64, 1, 1, 1024, 512
    dtype = mx.float16
    scale = 1.0 / (D**0.5)
    q, k, v = _make(B, Hq, Hkv, qL, kL, D, dtype, seed=29, strided=True)
    mkey, skey = mx.random.split(mx.random.key(23), 2)
    mask = mx.random.uniform(shape=(B, 1, qL, kL), key=mkey) > 0.5
    mask = mask | (mx.arange(kL).reshape(1, 1, 1, kL) == 0)
    sinks = mx.random.normal((Hq,), key=skey).astype(mx.float16)
    mx.eval(mask, sinks)
    got = kq.sdpa_vector(q, k, v, scale, causal=False, mask=mask, sinks=sinks)
    ref = _ref_sdpa(q, k, v, scale, causal=False, mask=mask, sinks=sinks)
    mx.eval(got, ref)
    rel = _rel(got, ref)
    print(f"  [sdpa/mask+sinks] MQA D=512: rel={rel:.3e}")
    assert rel < REL_BOUND[dtype]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--ql", default="1,2,5")
    ap.add_argument("--kl", default="2048,8192")
    args = ap.parse_args()
    for dtype in (mx.bfloat16, mx.float16):
        for qL in (int(x) for x in args.ql.split(",")):
            for kL in (int(x) for x in args.kl.split(",")):
                _check(args.d, qL, kL, dtype)
    _check(args.d, qL=5, kL=4096, dtype=mx.bfloat16, strided=True)
    print("ok")


if __name__ == "__main__":
    sys.exit(main())
