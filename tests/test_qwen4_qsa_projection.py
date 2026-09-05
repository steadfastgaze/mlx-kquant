"""Exact released-geometry gate for the Qwen QSA projection envelope."""

import time

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

HIDDEN = 2560
INDEX_HEADS = 4
INDEX_WIDTH = 128
INDEX_PROJECTION = 640
QUERY_HEADS = 24
KV_HEADS = 2
HEAD_WIDTH = 256
QUERY_PROJECTION = QUERY_HEADS * HEAD_WIDTH * 2
KV_PROJECTION = KV_HEADS * HEAD_WIDTH
ROTARY_WIDTH = 64
EPS = 1e-6
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _lazy_strided(values):
    return mx.stack((values, mx.zeros_like(values)), axis=-1)[..., 0]


def _bf16(rng, shape, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32)).astype(
        mx.bfloat16
    )


def _patterned_q6_wire(rng, rows, columns, scale, patterns=8):
    source = _bf16(rng, (patterns, columns), scale)
    encoded, placeholder = kq.quantize(source, "q6_k", stream=mx.cpu)
    repeats = (rows + patterns - 1) // patterns
    wire = mx.tile(encoded, (repeats, 1))[:rows]
    return mx.contiguous(wire), placeholder


def _norm(values, weight):
    source = values.astype(mx.float32)
    normalized = source * mx.rsqrt(
        mx.mean(mx.square(source), axis=-1, keepdims=True) + EPS
    )
    return (normalized * (1 + weight.astype(mx.float32))).astype(values.dtype)


def _rotate(values, cosine, sine):
    rotary = values[..., :ROTARY_WIDTH]
    half = ROTARY_WIDTH // 2
    rotated_half = mx.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
    rotated = rotary * cosine + rotated_half * sine
    return mx.concatenate([rotated, values[..., ROTARY_WIDTH:]], axis=-1)


def _reference(args):
    (
        hidden,
        index_weight,
        index_scales,
        query_weight,
        query_scales,
        key_weight,
        key_scales,
        value_weight,
        value_scales,
        index_norm,
        query_norm,
        key_norm,
        cosine,
        sine,
        _positions,
    ) = args
    index_projection = kq.quantized_matmul(hidden, index_weight, index_scales, "q6_k")
    index_queries = index_projection[..., : INDEX_HEADS * INDEX_WIDTH].reshape(
        1, 1, INDEX_HEADS, INDEX_WIDTH
    )
    raw_index_key = index_projection[..., INDEX_HEADS * INDEX_WIDTH :]
    index_queries = _rotate(_norm(index_queries, index_norm), cosine, sine)

    query_projection = kq.quantized_matmul(
        hidden, query_weight, query_scales, "q6_k"
    ).reshape(1, 1, QUERY_HEADS, HEAD_WIDTH * 2)
    queries = query_projection[..., :HEAD_WIDTH]
    gate = query_projection[..., HEAD_WIDTH:].reshape(1, 1, -1)
    queries = _rotate(_norm(queries, query_norm), cosine, sine)

    keys = kq.quantized_matmul(hidden, key_weight, key_scales, "q6_k").reshape(
        1, 1, KV_HEADS, HEAD_WIDTH
    )
    keys = _rotate(_norm(keys, key_norm), cosine, sine).transpose(0, 2, 1, 3)
    values = kq.quantized_matmul(hidden, value_weight, value_scales, "q6_k").reshape(
        1, 1, KV_HEADS, HEAD_WIDTH
    )
    values = values.transpose(0, 2, 1, 3)
    return index_queries, raw_index_key, queries, gate, keys, values


def _args(seed=7321, scale=0.12):
    rng = np.random.default_rng(seed)
    hidden = _bf16(rng, (1, 1, HIDDEN), scale)
    index_weight, index_scales = _patterned_q6_wire(
        rng, INDEX_PROJECTION, HIDDEN, 0.025
    )
    query_weight, query_scales = _patterned_q6_wire(
        rng, QUERY_PROJECTION, HIDDEN, 0.025
    )
    key_weight, key_scales = _patterned_q6_wire(rng, KV_PROJECTION, HIDDEN, 0.025)
    value_weight, value_scales = _patterned_q6_wire(rng, KV_PROJECTION, HIDDEN, 0.025)
    index_norm = _bf16(rng, (INDEX_WIDTH,), 0.2)
    query_norm = _bf16(rng, (HEAD_WIDTH,), 0.2)
    key_norm = _bf16(rng, (HEAD_WIDTH,), 0.2)
    angles = _bf16(rng, (1, 1, 1, ROTARY_WIDTH), 0.7).astype(mx.float32)
    cosine = mx.cos(angles).astype(mx.bfloat16)
    sine = mx.sin(angles).astype(mx.bfloat16)
    positions = mx.array([[[17]], [[17]], [[17]]], dtype=mx.int32)
    return (
        hidden,
        index_weight,
        index_scales,
        query_weight,
        query_scales,
        key_weight,
        key_scales,
        value_weight,
        value_scales,
        index_norm,
        query_norm,
        key_norm,
        cosine,
        sine,
        positions,
    )


def _candidate(args):
    return kq.qwen4_qsa_project_rope_q6(*args, eps=EPS)


def _immediate_consumers(outputs):
    bias = mx.array(0.25, dtype=mx.bfloat16)
    return tuple(output + bias for output in outputs)


def _assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(
        np.asarray(actual.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
    )


def _simd_sum_float32(values):
    work = np.asarray(values, dtype=np.float32).copy()
    for width in (16, 8, 4, 2, 1):
        work[:width] = (work[:width] + work[width : 2 * width]).astype(np.float32)
    return work[0]


def _width256_sum(values, *, mlx_order):
    partials = np.zeros(32, dtype=np.float32)
    for lane in range(32):
        dimensions = (
            [block * 128 + lane * 4 + i for block in range(2) for i in range(4)]
            if mlx_order
            else [lane * 8 + i for i in range(8)]
        )
        for dimension in dimensions:
            value = np.float32(values[dimension])
            partials[lane] = np.float32(partials[lane] + np.float32(value * value))
    return _simd_sum_float32(partials)


def test_width256_reduction_order_has_observable_rounding_boundaries():
    rng = np.random.default_rng(20260830)
    for _ in range(708):
        values = (rng.standard_normal(HEAD_WIDTH) * 0.1).astype(np.float32)
    mlx_sum = _width256_sum(values, mlx_order=True)
    contiguous_sum = _width256_sum(values, mlx_order=False)
    assert mlx_sum.view(np.uint32) == 0x4023B523
    assert contiguous_sum.view(np.uint32) == 0x4023B522


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize(("seed", "scale"), [(7321, 0.12), (7322, 0.004), (7323, 4.0)])
def test_qsa_projection_envelope_matches_component_lattice_exactly(seed, scale):
    args = _args(seed, scale)
    expected = _reference(args)
    actual = _candidate(args)
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qsa_projection_envelope_accepts_lazy_strided_hidden():
    args = list(_args(7324, 0.12))
    args[0] = _lazy_strided(args[0])
    reference_args = [mx.contiguous(args[0]), *args[1:]]
    expected = _reference(reference_args)
    actual = _candidate(args)
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize(("seed", "scale"), [(9321, 0.12), (9322, 4.0)])
def test_qsa_projection_envelope_is_exact_when_immediately_consumed(seed, scale):
    args = _args(seed, scale)
    expected = _immediate_consumers(_reference(args))
    actual = _immediate_consumers(_candidate(args))
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qsa_projection_envelope_twelve_immediate_consumers_share_one_graph():
    base = _args(10321, 0.12)
    expected = []
    actual = []
    for layer in range(12):
        hidden = base[0] + mx.array(layer / 128, dtype=mx.bfloat16)
        args = (hidden, *base[1:])
        expected.extend(_immediate_consumers(_reference(args)))
        actual.extend(_immediate_consumers(_candidate(args)))
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.slow
def test_qsa_projection_envelope_alternating_timing():
    args = _args(8321, 0.12)
    mx.eval(*_reference(args), *_candidate(args))

    reference = []
    candidate = []
    for _ in range(7):
        start = time.perf_counter()
        mx.eval(*_reference(args))
        reference.append(time.perf_counter() - start)

        start = time.perf_counter()
        mx.eval(*_candidate(args))
        candidate.append(time.perf_counter() - start)

    reference_median = float(np.median(reference[1:]))
    candidate_median = float(np.median(candidate[1:]))
    print(
        "qsa projection envelope alternating median: "
        f"components={reference_median * 1e3:.6f} ms "
        f"candidate={candidate_median * 1e3:.6f} ms "
        f"gain={(reference_median / candidate_median - 1) * 100:.3f}%"
    )
    assert candidate_median < reference_median


def test_qsa_projection_envelope_contract_fails_closed():
    args = list(_args())
    output = _candidate(args)
    assert [value.shape for value in output] == [
        (1, 1, INDEX_HEADS, INDEX_WIDTH),
        (1, 1, INDEX_WIDTH),
        (1, 1, QUERY_HEADS, HEAD_WIDTH),
        (1, 1, QUERY_HEADS * HEAD_WIDTH),
        (1, KV_HEADS, 1, HEAD_WIDTH),
        (1, KV_HEADS, 1, HEAD_WIDTH),
    ]
    assert all(value.dtype == mx.bfloat16 for value in output)

    args[0] = args[0].astype(mx.float32)
    with pytest.raises(ValueError, match="hidden"):
        _candidate(args)

    args = list(_args())
    args[12] = args[12].reshape(ROTARY_WIDTH)
    with pytest.raises(ValueError, match="rope_cosine"):
        _candidate(args)

    args = list(_args())
    args[14] = args[14].reshape(3)
    with pytest.raises(ValueError, match="position_ids"):
        _candidate(args)

    args = list(_args())
    with pytest.raises(ValueError, match="Metal stream"):
        kq.qwen4_qsa_project_rope_q6(*args, eps=EPS, stream=mx.cpu)
