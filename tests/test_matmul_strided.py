#!/usr/bin/env python3
"""Strided / view operand pins for quantized_matmul.

The op-level contiguity check runs at graph construction, where an
unevaluated array still carries dense placeholder metadata, so a lazy
row-strided activation view (a slice of a grouped tensor on the
second-to-last axis) used to reach the kernels unconverted; the kernels
advance rows densely by K, so every multi-row call on such a view read the
wrong elements (measured rel error ~0.5-1.8 against the contiguized
reference, all codecs, both weight forms, half-precision dtypes; float32
activations were laundered dense by the op's astype, and single-row calls
have no row stride to misread). The op now contiguizes unevaluated
multi-row activations lazy-safely, and KQuantMatmul::eval fails closed on
any operand that still arrives without densely packed rows.

Every test here computes the previously-wrong operand shape beside a
right-shape control (dense activation, or M=1) against the same contiguized
reference call, and requires bit-exact agreement: the lazy-safe contiguation
materializes the same dense layout the reference call passes, so the
dispatch is identical and there is no accumulation-order caveat.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

CODECS = ["q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k", "iq4_xs"]

DTYPES = {"bf16": mx.bfloat16, "f16": mx.float16}


def _f32(arr):
    return np.asarray(arr.astype(mx.float32))


def _reference(base, group, wire_rows, scales, codec):
    """The contiguized-operand call: same op, dense evaluated inputs."""
    x_ref = mx.contiguous(base[:, :, group, :])
    w_ref = mx.contiguous(wire_rows)
    mx.eval(x_ref, w_ref)
    ref = kq.quantized_matmul(x_ref, w_ref, scales, codec, transpose=True)
    mx.eval(ref)
    return ref


@pytest.mark.parametrize("codec", CODECS)
@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize("w_form", ["full", "view"])
@pytest.mark.parametrize("bsz", [1, 2])
def test_lazy_row_strided_activation_matches_contiguized_reference(
    codec, dtype_name, w_form, bsz
):
    rows, k = 16, 256
    rng = np.random.default_rng(11)
    dense = mx.array(rng.standard_normal((rows, k), dtype=np.float32))
    wire, scales = kq.quantize(dense, codec)
    mx.eval(wire, scales)
    w = wire[rows // 2 :, :] if w_form == "view" else wire

    for m in (2, 3, 4, 8, 64):
        base = mx.array(rng.standard_normal((bsz, m, 2, k), dtype=np.float32)).astype(
            DTYPES[dtype_name]
        )
        mx.eval(base)
        ref = _reference(base, 1, w, scales, codec)

        # The previously-wrong shape: a fresh (unevaluated) row-strided view.
        x_lazy = base[:, :, 1, :]
        out = kq.quantized_matmul(x_lazy, w, scales, codec, transpose=True)
        mx.eval(out)
        np.testing.assert_array_equal(
            _f32(out),
            _f32(ref),
            err_msg=(
                f"lazy strided view diverges: codec={codec} "
                f"dtype={dtype_name} w={w_form} bsz={bsz} M={m}"
            ),
        )

        # Right-shape control beside it: the dense activation, same values.
        x_dense = mx.contiguous(base[:, :, 1, :])
        mx.eval(x_dense)
        out_dense = kq.quantized_matmul(x_dense, w, scales, codec, transpose=True)
        mx.eval(out_dense)
        np.testing.assert_array_equal(_f32(out_dense), _f32(ref))

    # Single-row control: the decode shape never misread its (absent) row
    # stride and must stay on the no-copy path with the same values.
    base = mx.array(rng.standard_normal((1, 1, 2, k), dtype=np.float32)).astype(
        DTYPES[dtype_name]
    )
    mx.eval(base)
    ref = _reference(base, 1, w, scales, codec)
    out = kq.quantized_matmul(base[:, :, 1, :], w, scales, codec, transpose=True)
    mx.eval(out)
    np.testing.assert_array_equal(_f32(out), _f32(ref))


def test_recorded_ds4_dense_gate_repro_shape():
    # The exact shape recorded by the DS4 dense-gate diagnosis: a [16, 256]
    # q6_k wire's row-slice [8:, :] against a [1, 3, 2, 256] bf16 activation's
    # [:, :, 1, :] slice, multi-row. Pre-fix this read rel ~1.4 against the
    # dequantized matmul while each contiguized variant was clean.
    rng = np.random.default_rng(11)
    dense = mx.array(rng.standard_normal((16, 256), dtype=np.float32))
    wire, scales = kq.quantize(dense, "q6_k")
    mx.eval(wire, scales)
    base = mx.array(rng.standard_normal((1, 3, 2, 256), dtype=np.float32)).astype(
        mx.bfloat16
    )
    mx.eval(base)

    out = kq.quantized_matmul(
        base[:, :, 1, :], wire[8:, :], scales, "q6_k", transpose=True
    )
    mx.eval(out)

    ref = _reference(base, 1, wire[8:, :], scales, "q6_k")
    np.testing.assert_array_equal(_f32(out), _f32(ref))

    # Independent magnitude check against the dequantized matmul (bf16
    # activation rounding class).
    wf = kq.dequantize(mx.contiguous(wire[8:, :]), scales, "q6_k", dtype=mx.float32)
    fref = mx.matmul(
        mx.contiguous(base[:, :, 1, :]).astype(mx.float32).reshape(-1, 256), wf.T
    )
    mx.eval(fref)
    np.testing.assert_allclose(
        _f32(out).reshape(-1, 8), np.asarray(fref), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("m", [1, 2, 4, 5])
def test_ds4_wo_a_group_view_shape(m):
    # The DS4 grouped attention output projection per-group call: wo_a wire
    # [groups*rank, K] row-slice [g*rank:(g+1)*rank] against the grouped
    # activation [bsz, length, groups, K] slice [:, :, g, :]. n_heads=64,
    # head_dim=512, o_groups=8, o_lora_rank=1024 give K=4096, rank=1024.
    # M=1 is the decode shape (always clean); M in [2, 5] covers the drafter
    # verify widths that rode the defect.
    groups, rank, k = 8, 1024, 4096
    rng = np.random.default_rng(23)
    dense = mx.array(rng.standard_normal((groups * rank, k), dtype=np.float32) * 0.05)
    wire, scales = kq.quantize(dense, "q6_k")
    mx.eval(wire, scales)

    base = mx.array(
        rng.standard_normal((1, m, groups, k), dtype=np.float32) * 0.05
    ).astype(mx.bfloat16)
    mx.eval(base)
    for g in (0, 7):
        w = wire[g * rank : (g + 1) * rank, :]
        out = kq.quantized_matmul(base[:, :, g, :], w, scales, "q6_k", transpose=True)
        mx.eval(out)
        ref = _reference(base, g, w, scales, "q6_k")
        np.testing.assert_array_equal(
            _f32(out), _f32(ref), err_msg=f"wo_a group view diverges: M={m} g={g}"
        )


@pytest.mark.parametrize("m", [1, 4, 5])
def test_ds4_dense_verify_shape_stays_clean(m):
    # The dense verify class: a full-width dense weight against an already
    # dense activation at drafter-verify M (wo_b geometry, scaled to
    # K=8192 -> N=512 to keep the encode cheap). This class was never wrong;
    # the pin is that it tracks the dequantized reference.
    n, k = 512, 8192
    rng = np.random.default_rng(29)
    dense = mx.array(rng.standard_normal((n, k), dtype=np.float32) * 0.02)
    wire, scales = kq.quantize(dense, "q6_k")
    mx.eval(wire, scales)

    x = mx.array(rng.standard_normal((m, k), dtype=np.float32) * 0.02).astype(
        mx.bfloat16
    )
    mx.eval(x)
    out = kq.quantized_matmul(x, wire, scales, "q6_k", transpose=True)
    mx.eval(out)

    wf = kq.dequantize(wire, scales, "q6_k", dtype=mx.float32)
    fref = mx.matmul(x.astype(mx.float32), wf.T)
    mx.eval(fref)
    np.testing.assert_allclose(_f32(out), np.asarray(fref), rtol=2e-2, atol=2e-2)


def test_eval_fails_closed_on_non_dense_operands():
    # Operands the op-level guards cannot fix must throw at eval, never
    # compute garbage. Two lazy shapes reach eval non-dense: a single-row
    # activation with a strided last dim (single-row calls skip the
    # lazy-safe contiguation because their row stride is never read - a
    # strided last dim still is), and a multi-row weight whose row slice
    # keeps whole blocks but strides the rows.
    rng = np.random.default_rng(37)
    dense256 = mx.array(rng.standard_normal((16, 256), dtype=np.float32))
    wire256, scales256 = kq.quantize(dense256, "q6_k")
    dense512 = mx.array(rng.standard_normal((16, 512), dtype=np.float32))
    wire512, scales512 = kq.quantize(dense512, "q6_k")
    mx.eval(wire256, scales256, wire512, scales512)

    base = mx.array(rng.standard_normal((1, 1, 512), dtype=np.float32)).astype(
        mx.bfloat16
    )
    mx.eval(base)
    out = kq.quantized_matmul(
        base[..., ::2], wire256, scales256, "q6_k", transpose=True
    )
    with pytest.raises(Exception, match="densely packed rows"):
        mx.eval(out)

    x = mx.array(rng.standard_normal((3, 256), dtype=np.float32)).astype(mx.bfloat16)
    mx.eval(x)
    out = kq.quantized_matmul(x, wire512[:, :210], scales512, "q6_k", transpose=True)
    with pytest.raises(Exception, match="densely packed rows"):
        mx.eval(out)


def test_lazy_row_strided_activation_cpu_path():
    # The CPU implementation shares the dense-rows requirement; pin the same
    # lazy strided pair through the CPU device.
    default = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(11)
        dense = mx.array(rng.standard_normal((16, 256), dtype=np.float32))
        wire, scales = kq.quantize(dense, "q6_k")
        mx.eval(wire, scales)
        base = mx.array(rng.standard_normal((1, 3, 2, 256), dtype=np.float32)).astype(
            mx.bfloat16
        )
        mx.eval(base)
        out = kq.quantized_matmul(
            base[:, :, 1, :], wire[8:, :], scales, "q6_k", transpose=True
        )
        mx.eval(out)
        ref = _reference(base, 1, wire[8:, :], scales, "q6_k")
        np.testing.assert_array_equal(_f32(out), _f32(ref))
    finally:
        mx.set_default_device(default)
