"""Small CPU-side API checks for Metal-only SDPA primitives."""

from __future__ import annotations

import mlx.core as mx
import pytest

import mlx_kquant as kq


def _structural_q8_decode_case():
    q = mx.zeros((1, 16, 1, 256), dtype=mx.float32, stream=mx.cpu)
    packed = mx.zeros((1, 2, 1, 64), dtype=mx.uint32, stream=mx.cpu)
    params = mx.zeros((1, 2, 1, 4), dtype=mx.float32, stream=mx.cpu)
    return q, (packed, params, params), (packed, params, params)


def test_sdpa_decode_q8_dimension_parallel_merge_is_keyword_only():
    q, pk, pv = _structural_q8_decode_case()
    with pytest.raises(TypeError):
        kq.sdpa_decode_q8(
            q,
            *pk,
            *pv,
            1.0 / 16.0,
            64,
            8,
            128,
            2,
            1,
            16,
            True,
        )


def test_sdpa_decode_q8_dimension_parallel_merge_capability():
    assert kq.HAS_SDPA_DECODE_Q8_DIMENSION_PARALLEL_MERGE is True


def test_sdpa_decode_q8_dimension_parallel_merge_reaches_cpu_primitive():
    q, pk, pv = _structural_q8_decode_case()
    out = kq.sdpa_decode_q8(
        q,
        *pk,
        *pv,
        1.0 / 16.0,
        splits=128,
        stage=2,
        compute=1,
        tile_c=16,
        dimension_parallel_merge=True,
        stream=mx.cpu,
    )
    with pytest.raises(RuntimeError, match="has no CPU implementation"):
        mx.eval(out)
