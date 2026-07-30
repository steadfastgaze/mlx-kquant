"""Exact Q8_0 decode matvec plus DeepSeek-V4 hC-post parity."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

HC = 4
N = 32
K = 512
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _case(dtype):
    mx.random.seed(417)
    source_w = (0.15 * mx.random.normal((N, K))).astype(mx.float32)
    w, scales = kq.quantize(source_w, "q8_0")
    x = (0.2 * mx.random.normal((1, 1, K))).astype(dtype)
    residual = (0.3 * mx.random.normal((HC, N))).astype(mx.float32)
    post = (0.2 * mx.random.normal((HC,))).astype(mx.float32)
    comb = (0.25 * mx.random.normal((HC, HC))).astype(mx.float32)
    mx.eval(w, scales, x, residual, post, comb)
    return x, w, scales, residual, post, comb


def _reference(x, w, scales, residual, post, comb):
    qmv = kq.quantized_matmul(x, w, scales, "q8_0", transpose=True)
    mx.eval(qmv)
    qmv = qmv.reshape(N).astype(mx.float32)
    mixed = mx.matmul(mx.swapaxes(comb, -1, -2), residual)
    mx.eval(qmv, mixed)
    out = post[:, None] * qmv[None, :] + mixed
    mx.eval(out)
    return out


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qmv_hc_post_bitwise_parity(dtype):
    x, w, scales, residual, post, comb = _case(dtype)
    ref = _reference(x, w, scales, residual, post, comb)
    got = kq.quantized_matmul_qmv_hc_post(x, w, scales, residual, post, comb, "q8_0")
    mx.eval(got)

    assert got.dtype == mx.float32
    assert got.shape == (HC, N)
    assert np.array_equal(
        np.asarray(got).view(np.uint32),
        np.asarray(ref).view(np.uint32),
    )


def test_qmv_hc_post_validation():
    x, w, scales, residual, post, comb = _case(mx.float32)

    with pytest.raises(ValueError, match="only kquant_type 'q8_0'"):
        kq.quantized_matmul_qmv_hc_post(x, w, scales, residual, post, comb, "q4_k")
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        kq.quantized_matmul_qmv_hc_post(
            x.astype(mx.float16), w, scales, residual, post, comb, "q8_0"
        )
    with pytest.raises(ValueError, match="exactly one row"):
        kq.quantized_matmul_qmv_hc_post(
            mx.broadcast_to(x, (2, 1, K)),
            w,
            scales,
            residual,
            post,
            comb,
            "q8_0",
        )
    with pytest.raises(ValueError, match="residual must be float32"):
        kq.quantized_matmul_qmv_hc_post(x, w, scales, residual[:3], post, comb, "q8_0")


def test_qmv_hc_post_cpu_throws():
    x, w, scales, residual, post, comb = _case(mx.float32)
    got = kq.quantized_matmul_qmv_hc_post(
        x,
        w,
        scales,
        residual,
        post,
        comb,
        "q8_0",
        stream=mx.cpu,
    )
    with pytest.raises(RuntimeError, match="Metal-only"):
        mx.eval(got)
