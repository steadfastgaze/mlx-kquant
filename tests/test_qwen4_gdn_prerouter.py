"""Exact synthetic checks for the fixed Qwen pre-router layer envelope."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

BRANCHES = 4
HIDDEN = 2560
EXPANDED = BRANCHES * HIDDEN
LOWRANK = 320
CONV_STATE_ROWS = 3
CONV_KERNEL = 4
KEY_HEADS = 16
VALUE_HEADS = 48
HEAD_WIDTH = 128
VALUE_WIDTH = VALUE_HEADS * HEAD_WIDTH
EPS = 1e-6


def _mlx_array_runtime_available():
    previous = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        mx.eval(mx.zeros((1,), dtype=mx.float32, stream=mx.cpu))
    except RuntimeError:
        return False
    finally:
        mx.set_default_device(previous)
    return True


ARRAY_RUNTIME_AVAILABLE = _mlx_array_runtime_available()
METAL_AVAILABLE = (
    ARRAY_RUNTIME_AVAILABLE and hasattr(mx, "metal") and mx.metal.is_available()
)


def _lazy_strided(values):
    return mx.stack((values, mx.zeros_like(values)), axis=-1)[..., 0]


def _bf16(rng, shape, scale):
    return mx.array((scale * rng.standard_normal(shape)).astype(np.float32)).astype(
        mx.bfloat16
    )


def _repeated_wire(rng, codec, rows, columns, scale):
    source = _bf16(rng, (1, columns), scale)
    wire, scales = kq.quantize(source, codec, stream=mx.cpu)
    return mx.contiguous(mx.repeat(wire, rows, axis=0)), scales


def _reference(
    hidden,
    attention_norm_weight,
    attention_wires,
    mlp_norm_weight,
    mlp_wires,
    gdn_wires,
    conv_state,
    conv_weight,
    a_log,
    dt_bias,
    recurrent_state,
    gdn_norm_weight,
    pending_output=None,
    pending_injection=None,
):
    from mlx_lm.models.gated_delta import gated_delta_kernel

    attention_norm, attention_residual = kq.qwen4_hc_norm(
        hidden,
        attention_norm_weight,
        pending_output,
        pending_injection,
        eps=EPS,
    )
    attention_lowrank, attention_injection = kq.qwen4_hc_front(
        attention_norm, *attention_wires[:4]
    )
    attention_mixed = kq.qwen4_hc_epilogue(
        attention_lowrank,
        attention_wires[4],
        attention_wires[5],
        attention_norm,
    )

    qkv = kq.quantized_matmul(attention_mixed, gdn_wires[0], gdn_wires[1], "q6_k")
    gate = kq.quantized_matmul(attention_mixed, gdn_wires[2], gdn_wires[3], "q6_k")
    beta_logits = kq.quantized_matmul(
        attention_mixed, gdn_wires[4], gdn_wires[5], "q6_k"
    )
    decay_logits = kq.quantized_matmul(
        attention_mixed, gdn_wires[6], gdn_wires[7], "q6_k"
    )
    query, key, value, beta, decay, next_conv = kq.qwen4_gdn_prepare(
        qkv,
        beta_logits,
        decay_logits,
        conv_state,
        conv_weight,
        a_log,
        dt_bias,
    )
    recurrence, next_recurrent = gated_delta_kernel(
        query, key, value, decay, beta, recurrent_state, None
    )
    gdn_normalized = kq.qwen4_gdn_norm_gate(recurrence, gate, gdn_norm_weight, eps=EPS)
    mixer_output = kq.quantized_matmul(
        gdn_normalized, gdn_wires[8], gdn_wires[9], "q6_k"
    )

    mlp_norm, mlp_residual = kq.qwen4_hc_norm(
        attention_residual,
        mlp_norm_weight,
        mixer_output,
        attention_injection,
        eps=EPS,
    )
    mlp_lowrank, mlp_injection = kq.qwen4_hc_front(mlp_norm, *mlp_wires[:4])
    mlp_hidden = kq.qwen4_hc_epilogue(
        mlp_lowrank,
        mlp_wires[4],
        mlp_wires[5],
        mlp_norm,
    )
    return (
        mlp_hidden,
        mlp_residual,
        mlp_injection,
        next_conv,
        next_recurrent,
    )


def _assert_exact(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    view = mx.uint32 if got.dtype == mx.float32 else mx.uint16
    assert np.array_equal(np.asarray(got.view(view)), np.asarray(expected.view(view)))


def _build_case(seed):
    rng = np.random.default_rng(seed)
    hidden = _bf16(rng, (1, 1, EXPANDED), 0.4)
    attention_norm_weight = _bf16(rng, (EXPANDED,), 0.05)
    attention_wires = (
        *_repeated_wire(rng, "q6_k", LOWRANK, EXPANDED, 0.025),
        *_repeated_wire(rng, "q6_k", BRANCHES, EXPANDED, 0.025),
        *_repeated_wire(rng, "q8_0", EXPANDED, LOWRANK, 0.025),
    )
    mlp_norm_weight = _bf16(rng, (EXPANDED,), 0.05)
    mlp_wires = (
        *_repeated_wire(rng, "q6_k", LOWRANK, EXPANDED, 0.025),
        *_repeated_wire(rng, "q6_k", BRANCHES, EXPANDED, 0.025),
        *_repeated_wire(rng, "q8_0", EXPANDED, LOWRANK, 0.025),
    )
    gdn_wires = (
        *_repeated_wire(rng, "q6_k", EXPANDED, HIDDEN, 0.03),
        *_repeated_wire(rng, "q6_k", VALUE_WIDTH, HIDDEN, 0.03),
        *_repeated_wire(rng, "q6_k", VALUE_HEADS, HIDDEN, 0.03),
        *_repeated_wire(rng, "q6_k", VALUE_HEADS, HIDDEN, 0.03),
        *_repeated_wire(rng, "q6_k", HIDDEN, VALUE_WIDTH, 0.03),
    )
    conv_state = _bf16(rng, (1, CONV_STATE_ROWS, EXPANDED), 0.1)
    conv_weight = _bf16(rng, (EXPANDED, CONV_KERNEL, 1), 0.05)
    a_log = _bf16(rng, (VALUE_HEADS,), 0.5)
    dt_bias = _bf16(rng, (VALUE_HEADS,), 0.5)
    recurrent_state = mx.array(
        (0.02 * rng.standard_normal((1, VALUE_HEADS, HEAD_WIDTH, HEAD_WIDTH))).astype(
            np.float32
        )
    )
    gdn_norm_weight = _bf16(rng, (HEAD_WIDTH,), 0.5)
    pending_output = _bf16(rng, (1, 1, HIDDEN), 0.1)
    pending_injection = _bf16(rng, (1, 1, BRANCHES), 0.2)
    return {
        "hidden": hidden,
        "attention_norm_weight": attention_norm_weight,
        "attention_wires": attention_wires,
        "mlp_norm_weight": mlp_norm_weight,
        "mlp_wires": mlp_wires,
        "gdn_wires": gdn_wires,
        "conv_state": conv_state,
        "conv_weight": conv_weight,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "recurrent_state": recurrent_state,
        "gdn_norm_weight": gdn_norm_weight,
        "pending_output": pending_output,
        "pending_injection": pending_injection,
    }


def _invoke(case, with_pending):
    aw = case["attention_wires"]
    mw = case["mlp_wires"]
    gw = case["gdn_wires"]
    kwargs = {"eps": EPS}
    if with_pending:
        kwargs.update(
            pending_output=case["pending_output"],
            pending_injection=case["pending_injection"],
        )
    return kq.qwen4_gdn_prerouter_q6(
        case["hidden"],
        case["attention_norm_weight"],
        aw[0],
        aw[1],
        aw[2],
        aw[3],
        aw[4],
        aw[5],
        case["mlp_norm_weight"],
        mw[0],
        mw[1],
        mw[2],
        mw[3],
        mw[4],
        mw[5],
        gw[0],
        gw[1],
        gw[2],
        gw[3],
        gw[4],
        gw[5],
        gw[6],
        gw[7],
        case["conv_state"],
        case["conv_weight"],
        case["a_log"],
        case["dt_bias"],
        case["recurrent_state"],
        case["gdn_norm_weight"],
        gw[8],
        gw[9],
        **kwargs,
    )


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
@pytest.mark.parametrize("with_pending", [False, True])
def test_qwen4_gdn_prerouter_matches_staged_lattice(with_pending):
    case = _build_case(7811 + int(with_pending))
    expected = _reference(
        case["hidden"],
        case["attention_norm_weight"],
        case["attention_wires"],
        case["mlp_norm_weight"],
        case["mlp_wires"],
        case["gdn_wires"],
        case["conv_state"],
        case["conv_weight"],
        case["a_log"],
        case["dt_bias"],
        case["recurrent_state"],
        case["gdn_norm_weight"],
        *(
            (case["pending_output"], case["pending_injection"])
            if with_pending
            else (None, None)
        ),
    )
    got = _invoke(case, with_pending)
    mx.eval(*expected, *got)
    for actual, reference in zip(got, expected, strict=True):
        _assert_exact(actual, reference)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_qwen4_gdn_prerouter_accepts_lazy_strided_hidden():
    case = _build_case(7813)
    case["hidden"] = _lazy_strided(case["hidden"])
    expected = _reference(
        case["hidden"],
        case["attention_norm_weight"],
        case["attention_wires"],
        case["mlp_norm_weight"],
        case["mlp_wires"],
        case["gdn_wires"],
        case["conv_state"],
        case["conv_weight"],
        case["a_log"],
        case["dt_bias"],
        case["recurrent_state"],
        case["gdn_norm_weight"],
    )
    actual = _invoke(case, False)
    mx.eval(*expected, *actual)
    for got, wanted in zip(actual, expected, strict=True):
        _assert_exact(got, wanted)


def _zero_case():
    q6_hc = mx.zeros((LOWRANK, EXPANDED * 210 // 256), dtype=mx.uint8, stream=mx.cpu)
    q6_injection = mx.zeros(
        (BRANCHES, EXPANDED * 210 // 256), dtype=mx.uint8, stream=mx.cpu
    )
    q8_hc = mx.zeros((EXPANDED, LOWRANK * 34 // 32), dtype=mx.uint8, stream=mx.cpu)
    q6_gdn = mx.zeros((EXPANDED, HIDDEN * 210 // 256), dtype=mx.uint8, stream=mx.cpu)
    q6_gate = mx.zeros(
        (VALUE_WIDTH, HIDDEN * 210 // 256), dtype=mx.uint8, stream=mx.cpu
    )
    q6_head = mx.zeros(
        (VALUE_HEADS, HIDDEN * 210 // 256), dtype=mx.uint8, stream=mx.cpu
    )
    q6_out = mx.zeros((HIDDEN, VALUE_WIDTH * 210 // 256), dtype=mx.uint8, stream=mx.cpu)
    scales = mx.zeros((1,), dtype=mx.uint8, stream=mx.cpu)

    def zero(shape, dtype=mx.bfloat16):
        return mx.zeros(shape, dtype=dtype, stream=mx.cpu)

    return [
        zero((1, 1, EXPANDED)),
        zero((EXPANDED,)),
        q6_hc,
        scales,
        q6_injection,
        scales,
        q8_hc,
        scales,
        zero((EXPANDED,)),
        q6_hc,
        scales,
        q6_injection,
        scales,
        q8_hc,
        scales,
        q6_gdn,
        scales,
        q6_gate,
        scales,
        q6_head,
        scales,
        q6_head,
        scales,
        zero((1, CONV_STATE_ROWS, EXPANDED)),
        zero((EXPANDED, CONV_KERNEL, 1)),
        zero((VALUE_HEADS,)),
        zero((VALUE_HEADS,)),
        zero((1, VALUE_HEADS, HEAD_WIDTH, HEAD_WIDTH), mx.float32),
        zero((HEAD_WIDTH,)),
        q6_out,
        scales,
    ]


@pytest.mark.skipif(not ARRAY_RUNTIME_AVAILABLE, reason="requires an MLX array runtime")
def test_qwen4_gdn_prerouter_contract_and_optional_pair():
    args = _zero_case()
    outputs = kq.qwen4_gdn_prerouter_q6(*args)
    assert [value.shape for value in outputs] == [
        (1, 1, HIDDEN),
        (1, 1, EXPANDED),
        (1, 1, BRANCHES),
        (1, CONV_STATE_ROWS, EXPANDED),
        (1, VALUE_HEADS, HEAD_WIDTH, HEAD_WIDTH),
    ]
    assert [value.dtype for value in outputs] == [
        mx.bfloat16,
        mx.bfloat16,
        mx.bfloat16,
        mx.bfloat16,
        mx.float32,
    ]

    with pytest.raises(ValueError, match="supplied together"):
        kq.qwen4_gdn_prerouter_q6(
            *args,
            pending_output=mx.zeros((1, 1, HIDDEN), dtype=mx.bfloat16, stream=mx.cpu),
        )

    with pytest.raises(ValueError, match="pending_injection"):
        kq.qwen4_gdn_prerouter_q6(
            *args,
            pending_output=mx.zeros((1, 1, HIDDEN), dtype=mx.bfloat16, stream=mx.cpu),
            pending_injection=mx.zeros((BRANCHES,), dtype=mx.bfloat16, stream=mx.cpu),
        )


def test_qwen4_gdn_prerouter_source_contract():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "kquant_qwen4_gdn_prerouter.cpp"
    ).read_text()
    assert source.count("get_command_encoder(stream)") == 1
    assert source.count("encoder.add_temporaries") == 1
    assert "has_pending_ ? inputs[31] : inputs[0]" in source
    assert "has_pending_ ? 1 : 0" in source
    for kernel in (
        "kq_qwen4_hc_norm",
        "kq_qwen4_hc_front",
        "kq_qwen4_hc_epilogue",
        "kq_qwen4_gdn_prepare",
        "kq_qwen4_gdn_norm_gate",
    ):
        assert kernel in source
    assert 'kq_get_kernel(device, "kq_qwen4_gdn_recurrence")' in source
