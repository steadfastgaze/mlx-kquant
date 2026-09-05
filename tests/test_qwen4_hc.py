"""Exact BF16 parity for the released Qwen gated-residual decode cell."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq

BRANCHES = 4
WIDTH = 2560
EXPANDED = BRANCHES * WIDTH
LOWRANK = 320
EPS = 1e-6
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _lazy_strided(values):
    return mx.stack((values, mx.zeros_like(values)), axis=-1)[..., 0]


def _bf16(rng, shape, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32)).astype(
        mx.bfloat16
    )


def _quantized_wires(rng):
    down = _bf16(rng, (LOWRANK, EXPANDED), 0.025)
    injection = _bf16(rng, (BRANCHES, EXPANDED), 0.025)
    up = _bf16(rng, (EXPANDED, LOWRANK), 0.025)
    down_wire, down_scales = kq.quantize(down, "q6_k")
    injection_wire, injection_scales = kq.quantize(injection, "q6_k")
    up_wire, up_scales = kq.quantize(up, "q8_0")
    mx.eval(
        down_wire,
        down_scales,
        injection_wire,
        injection_scales,
        up_wire,
        up_scales,
    )
    return (
        down_wire,
        down_scales,
        injection_wire,
        injection_scales,
        up_wire,
        up_scales,
    )


def _reference_norm(residual, norm_weight):
    source = residual.astype(mx.float32).reshape(1, 1, BRANCHES, WIDTH)
    normalized = source * mx.rsqrt(
        mx.mean(mx.square(source), axis=-1, keepdims=True) + EPS
    )
    normalized = normalized.reshape(residual.shape)
    return (normalized * (1 + norm_weight.astype(mx.float32))).astype(mx.bfloat16)


def _reference_write(residual, block_output, injection):
    update = block_output[..., None, :] * injection[..., :, None]
    return residual + update.reshape(residual.shape)


def _reference_front(
    normalized,
    down_wire,
    down_scales,
    injection_wire,
    injection_scales,
):
    down = kq.quantized_matmul(
        normalized, down_wire, down_scales, "q6_k", transpose=True
    )
    lowrank = nn.silu(down / BRANCHES)
    injection = 2 * mx.sigmoid(
        kq.quantized_matmul(
            normalized,
            injection_wire,
            injection_scales,
            "q6_k",
            transpose=True,
        )
        / BRANCHES
    )
    return lowrank, injection


def _reference_epilogue(lowrank, up_wire, up_scales, normalized):
    projected = kq.quantized_matmul(lowrank, up_wire, up_scales, "q8_0", transpose=True)
    mix = mx.sigmoid(projected).reshape(1, 1, BRANCHES, WIDTH)
    streams = normalized.reshape(1, 1, BRANCHES, WIDTH)
    return mx.mean(mix * streams, axis=-2)


def _assert_bf16_equal(got, expected):
    assert got.dtype == mx.bfloat16
    assert got.shape == expected.shape
    assert np.array_equal(
        np.asarray(got.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
    )


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize(
    ("seed", "residual_scale", "output_scale", "injection_scale"),
    [
        (16, 0.4, 0.1, 0.2),
        (1601, 2.0, 0.5, 0.75),
        (1602, 8.0, 2.0, 1.5),
        (1603, 32.0, 8.0, 3.0),
    ],
)
def test_qwen4_hc_norm_matches_grouped_reduction_tree(
    seed, residual_scale, output_scale, injection_scale
):
    rng = np.random.default_rng(seed)
    residual = _bf16(rng, (1, 1, EXPANDED), residual_scale)
    norm_weight = _bf16(rng, (EXPANDED,), 0.05)
    pending_output = _bf16(rng, (1, 1, WIDTH), output_scale)
    pending_injection = _bf16(rng, (1, 1, BRANCHES), injection_scale)

    expected_updated = _reference_write(residual, pending_output, pending_injection)
    expected_norm = _reference_norm(expected_updated, norm_weight)
    got_norm, got_updated = kq.qwen4_hc_norm(
        residual,
        norm_weight,
        pending_output,
        pending_injection,
        eps=EPS,
    )
    mx.eval(expected_updated, expected_norm, got_norm, got_updated)
    _assert_bf16_equal(got_updated, expected_updated)
    _assert_bf16_equal(got_norm, expected_norm)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qwen4_hc_three_dispatch_cell_bitwise_parity():
    rng = np.random.default_rng(3821)
    residual = _bf16(rng, (1, 1, EXPANDED), 0.4)
    norm_weight = _bf16(rng, (EXPANDED,), 0.05)
    pending_output = _bf16(rng, (1, 1, WIDTH), 0.1)
    pending_injection = _bf16(rng, (1, 1, BRANCHES), 0.2)
    wires = _quantized_wires(rng)

    expected_updated = _reference_write(residual, pending_output, pending_injection)
    expected_norm = _reference_norm(expected_updated, norm_weight)
    got_norm, got_updated = kq.qwen4_hc_norm(
        residual,
        norm_weight,
        pending_output,
        pending_injection,
        eps=EPS,
    )
    mx.eval(expected_updated, expected_norm, got_norm, got_updated)
    _assert_bf16_equal(got_updated, expected_updated)
    _assert_bf16_equal(got_norm, expected_norm)

    expected_lowrank, expected_injection = _reference_front(expected_norm, *wires[:4])
    got_lowrank, got_injection = kq.qwen4_hc_front(got_norm, *wires[:4])
    mx.eval(expected_lowrank, expected_injection, got_lowrank, got_injection)
    _assert_bf16_equal(got_lowrank, expected_lowrank)
    _assert_bf16_equal(got_injection, expected_injection)

    expected_mixed = _reference_epilogue(
        expected_lowrank, wires[4], wires[5], expected_norm
    )
    got_mixed = kq.qwen4_hc_epilogue(got_lowrank, wires[4], wires[5], got_norm)
    mx.eval(expected_mixed, got_mixed)
    _assert_bf16_equal(got_mixed, expected_mixed)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qwen4_hc_norm_accepts_lazy_strided_residual():
    rng = np.random.default_rng(3822)
    residual = _lazy_strided(_bf16(rng, (1, 1, EXPANDED), 0.4))
    norm_weight = _bf16(rng, (EXPANDED,), 0.05)
    expected = _reference_norm(residual, norm_weight)
    actual, updated = kq.qwen4_hc_norm(residual, norm_weight, eps=EPS)
    mx.eval(expected, actual, updated)
    _assert_bf16_equal(actual, expected)
    _assert_bf16_equal(updated, residual)


def test_qwen4_hc_released_geometry_fails_closed():
    residual = mx.zeros((1, 1, EXPANDED), dtype=mx.bfloat16, stream=mx.cpu)
    norm_weight = mx.zeros((EXPANDED,), dtype=mx.bfloat16, stream=mx.cpu)
    with pytest.raises(ValueError, match="exactly one row"):
        kq.qwen4_hc_norm(mx.broadcast_to(residual, (2, 1, EXPANDED)), norm_weight)
    with pytest.raises(ValueError, match="supplied together"):
        kq.qwen4_hc_norm(
            residual,
            norm_weight,
            mx.zeros((1, 1, WIDTH), dtype=mx.bfloat16, stream=mx.cpu),
        )

    down_wire = mx.zeros(
        (LOWRANK, EXPANDED * 210 // 256), dtype=mx.uint8, stream=mx.cpu
    )
    scales = mx.zeros((1,), dtype=mx.uint8, stream=mx.cpu)
    with pytest.raises(ValueError, match="injection_weight and injection_scales"):
        kq.qwen4_hc_front(
            residual,
            down_wire,
            scales,
            mx.zeros((BRANCHES, EXPANDED * 210 // 256), dtype=mx.uint8, stream=mx.cpu),
        )
    with pytest.raises(ValueError, match="up_weight must be"):
        kq.qwen4_hc_epilogue(
            mx.zeros((1, 1, LOWRANK), dtype=mx.bfloat16, stream=mx.cpu),
            mx.zeros((EXPANDED, LOWRANK * 34 // 32 - 1), dtype=mx.uint8, stream=mx.cpu),
            scales,
            residual,
        )
