"""Exact released-geometry gates for the Qwen GDN decode boundaries."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq

CONV_CHANNELS = 10240
CONV_STATE_ROWS = 3
CONV_KERNEL = 4
KEY_HEADS = 16
VALUE_HEADS = 48
HEAD_WIDTH = 128
KEY_WIDTH = KEY_HEADS * HEAD_WIDTH
VALUE_WIDTH = VALUE_HEADS * HEAD_WIDTH
EPS = 1e-6
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _lazy_strided(values):
    return mx.stack((values, mx.zeros_like(values)), axis=-1)[..., 0]


def _bf16(rng, shape, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32)).astype(
        mx.bfloat16
    )


def _l2_normalize(values):
    return values * mx.rsqrt(mx.sum(mx.square(values), axis=-1, keepdims=True) + EPS)


def _reference_prepare(
    qkv, beta_logits, decay_logits, conv_state, conv_weight, a_log, dt_bias
):
    conv = nn.Conv1d(
        in_channels=CONV_CHANNELS,
        out_channels=CONV_CHANNELS,
        kernel_size=CONV_KERNEL,
        groups=CONV_CHANNELS,
        bias=False,
    )
    conv.weight = conv_weight
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv_state = mx.contiguous(conv_input[:, -CONV_STATE_ROWS:, :])
    conv_output = nn.silu(conv(conv_input))
    query, key, value = [
        tensor.reshape(1, 1, heads, HEAD_WIDTH)
        for tensor, heads in zip(
            mx.split(conv_output, [KEY_WIDTH, 2 * KEY_WIDTH], axis=-1),
            [KEY_HEADS, KEY_HEADS, VALUE_HEADS],
            strict=True,
        )
    ]
    query = _l2_normalize(query) * (HEAD_WIDTH**-0.5)
    key = _l2_normalize(key)
    beta = mx.sigmoid(beta_logits)
    decay = mx.exp(
        -mx.exp(a_log.astype(mx.float32))
        * nn.softplus(decay_logits.astype(mx.float32) + dt_bias.astype(mx.float32))
    )
    return query, key, value, beta, decay, next_conv_state


def _reference_norm_gate(recurrence, gate, norm_weight):
    source = recurrence.astype(mx.float32)
    normalized = source * mx.rsqrt(
        mx.mean(mx.square(source), axis=-1, keepdims=True) + EPS
    )
    weighted = norm_weight * normalized.astype(mx.bfloat16)
    gated = weighted * mx.sigmoid(
        gate.reshape(1, 1, VALUE_HEADS, HEAD_WIDTH).astype(mx.float32)
    )
    return gated.astype(mx.bfloat16).reshape(1, 1, VALUE_WIDTH)


def _assert_exact(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    view = mx.uint32 if got.dtype == mx.float32 else mx.uint16
    assert np.array_equal(np.asarray(got.view(view)), np.asarray(expected.view(view)))


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize(
    ("seed", "projection_scale", "state_scale", "weight_scale"),
    [(3817, 0.2, 0.2, 0.05), (3818, 0.002, 0.002, 0.02), (3819, 8.0, 8.0, 0.25)],
)
def test_qwen4_gdn_prepare_matches_released_operation_lattice(
    seed, projection_scale, state_scale, weight_scale
):
    rng = np.random.default_rng(seed)
    qkv = _bf16(rng, (1, 1, CONV_CHANNELS), projection_scale)
    beta_logits = _bf16(rng, (1, 1, VALUE_HEADS), 0.7)
    decay_logits = _bf16(rng, (1, 1, VALUE_HEADS), 0.7)
    conv_state = _bf16(rng, (1, CONV_STATE_ROWS, CONV_CHANNELS), state_scale)
    conv_weight = _bf16(rng, (CONV_CHANNELS, CONV_KERNEL, 1), weight_scale)
    a_log = _bf16(rng, (VALUE_HEADS,), 0.5)
    dt_bias = _bf16(rng, (VALUE_HEADS,), 0.5)
    expected = _reference_prepare(
        qkv, beta_logits, decay_logits, conv_state, conv_weight, a_log, dt_bias
    )
    actual = kq.qwen4_gdn_prepare(
        qkv, beta_logits, decay_logits, conv_state, conv_weight, a_log, dt_bias
    )
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize(
    ("seed", "recurrence_scale", "gate_scale"),
    [(4817, 0.02, 0.5), (4818, 2.0, 4.0), (4819, 32.0, 16.0)],
)
def test_qwen4_gdn_norm_gate_matches_released_operation_lattice(
    seed, recurrence_scale, gate_scale
):
    rng = np.random.default_rng(seed)
    recurrence = _bf16(rng, (1, 1, VALUE_HEADS, HEAD_WIDTH), recurrence_scale)
    gate = _bf16(rng, (1, 1, VALUE_WIDTH), gate_scale)
    norm_weight = _bf16(rng, (HEAD_WIDTH,), 0.5)
    expected = _reference_norm_gate(recurrence, gate, norm_weight)
    actual = kq.qwen4_gdn_norm_gate(recurrence, gate, norm_weight, eps=EPS)
    mx.eval(expected, actual)
    _assert_exact(actual, expected)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qwen4_gdn_prepare_accepts_lazy_strided_projection():
    rng = np.random.default_rng(3820)
    qkv = _lazy_strided(_bf16(rng, (1, 1, CONV_CHANNELS), 0.2))
    beta_logits = _bf16(rng, (1, 1, VALUE_HEADS), 0.7)
    decay_logits = _bf16(rng, (1, 1, VALUE_HEADS), 0.7)
    conv_state = _bf16(rng, (1, CONV_STATE_ROWS, CONV_CHANNELS), 0.2)
    conv_weight = _bf16(rng, (CONV_CHANNELS, CONV_KERNEL, 1), 0.05)
    a_log = _bf16(rng, (VALUE_HEADS,), 0.5)
    dt_bias = _bf16(rng, (VALUE_HEADS,), 0.5)
    expected = _reference_prepare(
        qkv, beta_logits, decay_logits, conv_state, conv_weight, a_log, dt_bias
    )
    actual = kq.qwen4_gdn_prepare(
        qkv, beta_logits, decay_logits, conv_state, conv_weight, a_log, dt_bias
    )
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


def test_qwen4_gdn_prepare_contract_is_exact():
    qkv = mx.zeros((1, 1, CONV_CHANNELS), dtype=mx.bfloat16, stream=mx.cpu)
    logits = mx.zeros((1, 1, VALUE_HEADS), dtype=mx.bfloat16, stream=mx.cpu)
    state = mx.zeros(
        (1, CONV_STATE_ROWS, CONV_CHANNELS), dtype=mx.bfloat16, stream=mx.cpu
    )
    weight = mx.zeros((CONV_CHANNELS, CONV_KERNEL, 1), dtype=mx.bfloat16, stream=mx.cpu)
    head_values = mx.zeros((VALUE_HEADS,), dtype=mx.bfloat16, stream=mx.cpu)
    outputs = kq.qwen4_gdn_prepare(
        qkv, logits, logits, state, weight, head_values, head_values
    )
    assert [value.shape for value in outputs] == [
        (1, 1, KEY_HEADS, HEAD_WIDTH),
        (1, 1, KEY_HEADS, HEAD_WIDTH),
        (1, 1, VALUE_HEADS, HEAD_WIDTH),
        (1, 1, VALUE_HEADS),
        (1, 1, VALUE_HEADS),
        (1, CONV_STATE_ROWS, CONV_CHANNELS),
    ]
    assert outputs[4].dtype == mx.float32
    with pytest.raises(ValueError, match="released decode shape"):
        kq.qwen4_gdn_prepare(
            qkv.reshape(CONV_CHANNELS),
            logits,
            logits,
            state,
            weight,
            head_values,
            head_values,
        )


def test_qwen4_gdn_norm_gate_contract_is_exact():
    recurrence = mx.zeros(
        (1, 1, VALUE_HEADS, HEAD_WIDTH), dtype=mx.bfloat16, stream=mx.cpu
    )
    gate = mx.zeros((1, 1, VALUE_WIDTH), dtype=mx.bfloat16, stream=mx.cpu)
    weight = mx.zeros((HEAD_WIDTH,), dtype=mx.bfloat16, stream=mx.cpu)
    output = kq.qwen4_gdn_norm_gate(recurrence, gate, weight)
    assert output.shape == (1, 1, VALUE_WIDTH)
    assert output.dtype == mx.bfloat16
    with pytest.raises(ValueError, match="eps must be positive"):
        kq.qwen4_gdn_norm_gate(recurrence, gate, weight, eps=0.0)
