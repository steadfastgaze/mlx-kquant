"""Exact DS4 shared-FFN Q8_0 matvec, routed-add, and hC-post parity."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

HC = 4
N = 4096
K = 2048
BLOCKS_PER_ROW = K // 32
WIRE_BYTES = BLOCKS_PER_ROW * 34
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _wire_bytes(rng):
    blocks = np.empty((N, BLOCKS_PER_ROW, 34), dtype=np.uint8)
    block_scales = rng.uniform(0.002, 0.02, (N, BLOCKS_PER_ROW)).astype(np.float16)
    blocks[:, :, :2] = block_scales.view(np.uint8).reshape(N, BLOCKS_PER_ROW, 2)
    blocks[:, :, 2:] = rng.integers(0, 256, (N, BLOCKS_PER_ROW, 32), dtype=np.uint8)
    return blocks.reshape(N, WIRE_BYTES)


def _metal_case():
    rng = np.random.default_rng(918)
    w = mx.array(_wire_bytes(rng))
    scales = mx.zeros((1,), dtype=mx.uint8)
    x = mx.array((0.15 * rng.standard_normal((1, 1, K))).astype(np.float32)).astype(
        mx.bfloat16
    )
    routed = mx.array((0.2 * rng.standard_normal((N,))).astype(np.float16))
    residual = mx.array((0.25 * rng.standard_normal((HC, N))).astype(np.float32))
    post = mx.array((0.2 * rng.standard_normal((HC,))).astype(np.float32))
    comb = mx.array((0.25 * rng.standard_normal((HC, HC))).astype(np.float32))
    mx.eval(w, scales, x, routed, residual, post, comb)
    return x, w, scales, routed, residual, post, comb


def _zero_case():
    x = mx.zeros((1, 1, K), dtype=mx.bfloat16, stream=mx.cpu)
    w = mx.zeros((N, WIRE_BYTES), dtype=mx.uint8, stream=mx.cpu)
    scales = mx.zeros((1,), dtype=mx.uint8, stream=mx.cpu)
    routed = mx.zeros((N,), dtype=mx.float16, stream=mx.cpu)
    residual = mx.zeros((HC, N), dtype=mx.float32, stream=mx.cpu)
    post = mx.zeros((HC,), dtype=mx.float32, stream=mx.cpu)
    comb = mx.zeros((HC, HC), dtype=mx.float32, stream=mx.cpu)
    return x, w, scales, routed, residual, post, comb


def _reference(x, w, scales, routed, residual, post, comb):
    shared = kq.quantized_matmul(x, w, scales, "q8_0", transpose=True)
    mx.eval(shared)
    merged = routed.astype(mx.float32) + shared.reshape(N).astype(mx.float32)
    mx.eval(merged)
    mixed = mx.matmul(mx.swapaxes(comb, -1, -2), residual)
    mx.eval(mixed)
    out = post[:, None] * merged[None, :] + mixed
    mx.eval(out)
    return out


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qmv_add_hc_post_bitwise_parity():
    x, w, scales, routed, residual, post, comb = _metal_case()
    ref = _reference(x, w, scales, routed, residual, post, comb)
    got = kq.quantized_matmul_qmv_add_hc_post(
        x, w, scales, routed, residual, post, comb, "q8_0"
    )
    mx.eval(got)

    assert got.dtype == mx.float32
    assert got.shape == (HC, N)
    assert np.array_equal(
        np.asarray(got).view(np.uint32),
        np.asarray(ref).view(np.uint32),
    )


def test_qmv_add_hc_post_validation():
    x, w, scales, routed, residual, post, comb = _zero_case()

    with pytest.raises(ValueError, match="only kquant_type"):
        kq.quantized_matmul_qmv_add_hc_post(
            x, w, scales, routed, residual, post, comb, "q4_k"
        )
    with pytest.raises(ValueError, match="bfloat16 with exactly one row"):
        kq.quantized_matmul_qmv_add_hc_post(
            x.astype(mx.float16),
            w,
            scales,
            routed,
            residual,
            post,
            comb,
            "q8_0",
        )
    with pytest.raises(ValueError, match="bfloat16 with exactly one row"):
        kq.quantized_matmul_qmv_add_hc_post(
            mx.broadcast_to(x, (2, 1, K)),
            w,
            scales,
            routed,
            residual,
            post,
            comb,
            "q8_0",
        )
    with pytest.raises(ValueError, match="Q8_0 wire bytes"):
        kq.quantized_matmul_qmv_add_hc_post(
            x, w[:, :-34], scales, routed, residual, post, comb, "q8_0"
        )
    with pytest.raises(ValueError, match="routed must be float16"):
        kq.quantized_matmul_qmv_add_hc_post(
            x,
            w,
            scales,
            routed.astype(mx.float32),
            residual,
            post,
            comb,
            "q8_0",
        )
    with pytest.raises(ValueError, match="residual must be float32"):
        kq.quantized_matmul_qmv_add_hc_post(
            x, w, scales, routed, residual[:3], post, comb, "q8_0"
        )
    with pytest.raises(ValueError, match="post must be float32"):
        kq.quantized_matmul_qmv_add_hc_post(
            x,
            w,
            scales,
            routed,
            residual,
            post.astype(mx.float16),
            comb,
            "q8_0",
        )
    with pytest.raises(ValueError, match="comb must be float32"):
        kq.quantized_matmul_qmv_add_hc_post(
            x, w, scales, routed, residual, post, comb[:3], "q8_0"
        )


def test_qmv_add_hc_post_cpu_throws():
    x, w, scales, routed, residual, post, comb = _zero_case()
    got = kq.quantized_matmul_qmv_add_hc_post(
        x,
        w,
        scales,
        routed,
        residual,
        post,
        comb,
        "q8_0",
        stream=mx.cpu,
    )
    with pytest.raises(RuntimeError, match="Metal-only"):
        mx.eval(got)
