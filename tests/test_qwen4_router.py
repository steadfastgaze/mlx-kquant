"""Exact released-geometry Qwen router parity tests."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

EXPERTS = 512
ROUTES = 10
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _lazy_strided(values):
    return mx.stack((values, mx.zeros_like(values)), axis=-1)[..., 0]


def _reference(logits):
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    candidates = mx.argpartition(probabilities, kth=-ROUTES, axis=-1)[..., -ROUTES:]
    candidate_scores = mx.take_along_axis(probabilities, candidates, axis=-1)
    rank = mx.argsort(-candidate_scores, axis=-1)
    indices = mx.take_along_axis(candidates, rank, axis=-1)
    scores = mx.take_along_axis(candidate_scores, rank, axis=-1)
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    return indices.astype(mx.uint32), scores.astype(logits.dtype)


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if actual.dtype == mx.bfloat16:
        assert np.array_equal(
            np.asarray(actual.view(mx.uint16)),
            np.asarray(expected.view(mx.uint16)),
        )
    else:
        assert np.array_equal(np.asarray(actual), np.asarray(expected))


def _assert_fused_exact(logits):
    expected = _reference(logits)
    actual = kq.qwen4_router_topk_fused_exact(logits)
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize("seed", [7411, 7412, 7413])
def test_fused_router_matches_softmax_ids_and_scores(seed):
    rng = np.random.default_rng(seed)
    logits = mx.array(rng.standard_normal((7, EXPERTS)).astype(np.float32)).astype(
        mx.bfloat16
    )
    _assert_fused_exact(logits)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_fused_router_preserves_released_tie_slice_and_order():
    logits_np = np.zeros((2, EXPERTS), dtype=np.float32)
    logits_np[1, 0] = 1.0
    logits = mx.array(logits_np).astype(mx.bfloat16)
    expected = _reference(logits)
    mx.eval(*expected)
    assert np.array_equal(np.asarray(expected[0])[0], np.arange(502, 512))
    _assert_fused_exact(logits)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_fused_router_matches_probability_rounding_boundaries():
    rows = np.zeros((4, EXPERTS), dtype=np.float32)
    rows[0] = np.linspace(-80.0, 0.0, EXPERTS, dtype=np.float32)
    rows[1] = np.repeat(np.linspace(-4.0, 4.0, 64, dtype=np.float32), 8)
    rows[2, 0] = 100.0
    rows[2, 1:] = np.linspace(-100.0, -80.0, EXPERTS - 1, dtype=np.float32)
    rows[3, np.asarray((7, 11, 19, 31, 47, 71, 101, 131, 173, 223, 281))] = 2.0
    _assert_fused_exact(mx.array(rows).astype(mx.bfloat16))


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_fused_router_accepts_lazy_strided_logits():
    rng = np.random.default_rng(7414)
    logits = mx.array(rng.standard_normal((7, EXPERTS)).astype(np.float32)).astype(
        mx.bfloat16
    )
    _assert_fused_exact(_lazy_strided(logits))


def test_fused_router_rejects_nonreleased_contract():
    bad = (
        mx.zeros((1, EXPERTS), dtype=mx.float32),
        mx.zeros((1, EXPERTS - 1), dtype=mx.bfloat16),
        mx.zeros((EXPERTS,), dtype=mx.bfloat16),
    )
    for logits in bad:
        with pytest.raises(ValueError):
            kq.qwen4_router_topk_fused_exact(logits)
