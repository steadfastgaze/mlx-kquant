#!/usr/bin/env python3
"""Synthetic (GGUF-free) validation for the decode-shaped routed-MoE matvecs.

gather_qmv_pair_swiglu computes every routed expert's fused gate/up + SwiGLU
activation row for one token, with the route weight baked into the stored
intermediate; gather_qmv_expert_sum computes the down matvec with the sum
over the token's routed experts inside the kernel. Both accumulate in
float32, so parity against the unfused composition (per-expert gather_qmm,
the SwiGLU formula, the route-weighted sum) holds to a tolerance: the fused
epilogues read float32 accumulators the composition rounds through the I/O
dtype. An independent f32-dequantize reference pins the semantics so a bug
shared with gather_qmm cannot pass parity by matching itself.

Wire bytes are minted in-process by kq.quantize (random uint8 would decode
to NaN). Coverage includes duplicate expert ids in the routed set (the same
expert selected twice), a non-default routed-set width, all three activation
dtypes, and the fail-closed rejections (uninstantiated codecs, shape and
dtype contract violations).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

K = 512
GATE_OUT = 64
N_DOWN = 256
E = 8

SWIGLU_LIMIT_ACTIVE = 0.2
SWIGLU_LIMIT_OFF = 1e6

# (rtol, atol) per I/O dtype for fused-vs-reference parity; the half and
# bfloat16 bounds absorb the composition's intermediate rounding, following
# the gather_qmm_sorted_swiglu test bounds. The float32 bounds sit ~4x above
# the measured worst case of the kernels' float32 accumulation against the
# float64 reference (max abs ~8.5e-7 at K=512 over six experts).
PAIR_TOLS = {
    mx.float16: (1e-2, 1.2e-3),
    mx.bfloat16: (5e-2, 1e-2),
    mx.float32: (5e-6, 4e-6),
}

# Routed id sets: the DS4 decode width with a duplicate expert, and a
# non-default width to prove the kernels take the slot count from the inputs.
ID_CASES = {
    "six_with_duplicate": [3, 0, 5, 3, 7, 1],
    "four": [6, 2, 2, 4],
}


@pytest.fixture(scope="module")
def pair_experts():
    """Combined gate/up stacks [E, 2 * GATE_OUT, K] as iq2_xxs wire bytes,
    plus their f32 dequantized forms."""
    rng = np.random.default_rng(3)
    imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    scales = mx.zeros((1,), dtype=mx.uint8)
    wq_list = []
    deq_list = []
    for _ in range(E):
        w_np = (rng.standard_normal((2 * GATE_OUT, K)) * 0.1).astype(np.float32)
        wq, _ = kq.quantize(mx.array(w_np), "iq2_xxs", imatrix=imat)
        mx.eval(wq)
        wq_np = np.ascontiguousarray(np.array(wq).astype(np.uint8))
        wq_list.append(wq_np)
        deq_list.append(
            np.array(kq.dequantize(mx.array(wq_np), scales, "iq2_xxs", mx.float32)))
    return mx.array(np.stack(wq_list)), deq_list


@pytest.fixture(scope="module")
def down_experts():
    """Down stacks [E, N_DOWN, K] as q2_k wire bytes, plus their f32
    dequantized forms."""
    rng = np.random.default_rng(4)
    scales = mx.zeros((1,), dtype=mx.uint8)
    wq_list = []
    deq_list = []
    for _ in range(E):
        w_np = (rng.standard_normal((N_DOWN, K)) * 0.1).astype(np.float32)
        wq, _ = kq.quantize(mx.array(w_np), "q2_k")
        mx.eval(wq)
        wq_np = np.ascontiguousarray(np.array(wq).astype(np.uint8))
        wq_list.append(wq_np)
        deq_list.append(
            np.array(kq.dequantize(mx.array(wq_np), scales, "q2_k", mx.float32)))
    return mx.array(np.stack(wq_list)), deq_list


def _swiglu_rows(gate, up, limit, route_weights):
    """The jang _dsv4_swiglu formula per routed row in float64 numpy, scaled
    by the route weight (the fused epilogue's contract)."""
    gate = gate.astype(np.float64).copy()
    up = up.astype(np.float64).copy()
    if limit > 0:
        up = np.clip(up, -limit, limit)
        gate = np.minimum(gate, limit)
    return gate / (1.0 + np.exp(-gate)) * up * route_weights[:, None]


# ---------------------------------------------------------------------------
# gather_qmv_pair_swiglu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(ID_CASES))
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_pair_swiglu_matches_dq_reference(pair_experts, case, dtype):
    """Fused output matches the f32-dequantize + f64 matvec reference through
    the clamped SwiGLU and route-weight scale, at both an active and an
    effectively disabled clamp (both arms asserted to differ)."""
    w, deq = pair_experts
    scales = mx.zeros((1,), dtype=mx.uint8)
    rng = np.random.default_rng(1)

    ids_np = np.array(ID_CASES[case], dtype=np.uint32)
    B = len(ids_np)
    rw_np = (rng.standard_normal(B) * 0.5 + 1.0).astype(np.float32)
    x_np = (rng.standard_normal((1, K)) * 0.5).astype(np.float32)
    x = mx.array(x_np).astype(dtype)
    # The kernel reads x through the I/O dtype; the reference must see the
    # same rounded values.
    x_ref = np.array(x.astype(mx.float32), dtype=np.float64).reshape(K)

    gate = np.stack([deq[e][:GATE_OUT].astype(np.float64) @ x_ref for e in ids_np])
    up = np.stack([deq[e][GATE_OUT:].astype(np.float64) @ x_ref for e in ids_np])
    clips = bool(
        (gate > SWIGLU_LIMIT_ACTIVE).any()
        or (np.abs(up) > SWIGLU_LIMIT_ACTIVE).any())
    assert clips, f"{case}: active limit would clip nothing"

    rtol, atol = PAIR_TOLS[dtype]
    for limit in (SWIGLU_LIMIT_ACTIVE, SWIGLU_LIMIT_OFF):
        got = kq.gather_qmv_pair_swiglu(
            x, w, scales, "iq2_xxs", mx.array(ids_np), mx.array(rw_np),
            GATE_OUT, limit)
        mx.eval(got)
        assert got.dtype == dtype
        assert got.shape == (B, GATE_OUT)
        ref = _swiglu_rows(gate, up, limit, rw_np.astype(np.float64))
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32), dtype=np.float64),
            ref,
            rtol=rtol,
            atol=atol,
            err_msg=f"{case} {dtype} limit={limit}",
        )


def test_pair_swiglu_matches_unfused_composition(pair_experts):
    """Float32 I/O parity against the live composition the decode path
    replaces: gather_qmm over the combined stack, slice, the f32 SwiGLU,
    then the route-weight scale."""
    w, _deq = pair_experts
    scales = mx.zeros((1,), dtype=mx.uint8)
    rng = np.random.default_rng(2)

    ids_np = np.array(ID_CASES["six_with_duplicate"], dtype=np.uint32)
    B = len(ids_np)
    rw_np = (rng.standard_normal(B) * 0.5 + 1.0).astype(np.float32)
    x_np = (rng.standard_normal((1, K)) * 0.5).astype(np.float32)
    x = mx.array(x_np)

    got = kq.gather_qmv_pair_swiglu(
        x, w, scales, "iq2_xxs", mx.array(ids_np), mx.array(rw_np),
        GATE_OUT, SWIGLU_LIMIT_ACTIVE)
    mx.eval(got)

    comb = kq.gather_qmm(
        mx.broadcast_to(x[None, None], (B, 1, 1, K)),
        w, scales, "iq2_xxs",
        rhs_indices=mx.array(ids_np).reshape(B, 1),
        transpose=True,
    )
    mx.eval(comb)
    comb = np.array(
        comb.astype(mx.float32), dtype=np.float64).reshape(B, 2 * GATE_OUT)
    ref = _swiglu_rows(
        comb[:, :GATE_OUT], comb[:, GATE_OUT:], SWIGLU_LIMIT_ACTIVE,
        rw_np.astype(np.float64))

    # gather_qmm promotes float32 x to bfloat16 output, so the composition
    # itself carries bfloat16 rounding; the bound reflects that, not the
    # fused kernel (which the dq reference test pins tightly).
    np.testing.assert_allclose(
        np.array(got, dtype=np.float64), ref, rtol=5e-2, atol=1e-2)


def test_pair_swiglu_deep_negative_gate_is_zero():
    """A gate accumulator past the float32 exp overflow point must produce
    exactly zero, not NaN: for gate below about -88, exp(-gate) overflows to
    +Inf and silu resolves as gate / Inf -> 0. The epilogue's precise::exp
    keeps that IEEE edge; a fast-math exp leaves the Inf class undefined.
    All-negative gate rows against all-positive x put every accumulator below
    -100, and the up rows stay large positive so a zero can only come from
    the silu factor."""
    rng = np.random.default_rng(7)
    scales = mx.zeros((1,), dtype=mx.uint8)
    imat = mx.array((np.abs(rng.standard_normal(K)) + 0.1).astype(np.float32))
    gate_np = -(np.abs(rng.standard_normal((GATE_OUT, K))) * 0.5 + 0.5)
    up_np = np.abs(rng.standard_normal((GATE_OUT, K))) * 0.5 + 0.5
    w_np = np.concatenate([gate_np, up_np]).astype(np.float32)
    wq, _ = kq.quantize(mx.array(w_np), "iq2_xxs", imatrix=imat)
    mx.eval(wq)
    wq_np = np.ascontiguousarray(np.array(wq).astype(np.uint8))
    deq = np.array(
        kq.dequantize(mx.array(wq_np), scales, "iq2_xxs", mx.float32))

    x_np = (np.abs(rng.standard_normal((1, K))) * 0.5 + 0.25).astype(np.float32)
    x64 = x_np.astype(np.float64).reshape(K)
    gate_ref = deq[:GATE_OUT].astype(np.float64) @ x64
    up_ref = deq[GATE_OUT:].astype(np.float64) @ x64
    assert gate_ref.max() < -100.0, "gate accumulators must sit past -100"
    assert np.abs(up_ref).min() > 1.0, "up factors must stay away from zero"

    w = mx.array(wq_np[None])
    ids = mx.array(np.zeros(2, dtype=np.uint32))
    rw = mx.array(np.ones(2, dtype=np.float32))
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        for limit in (0.0, SWIGLU_LIMIT_OFF):
            got = kq.gather_qmv_pair_swiglu(
                mx.array(x_np).astype(dtype), w, scales, "iq2_xxs", ids, rw,
                GATE_OUT, limit)
            mx.eval(got)
            out = np.array(got.astype(mx.float32))
            assert not np.isnan(out).any(), f"{dtype} limit={limit}: NaN"
            assert (out == 0.0).all(), f"{dtype} limit={limit}: nonzero"


def test_pair_swiglu_rejects_bad_inputs(pair_experts):
    """Fail-closed contract: uninstantiated codec, bad ids/route_weights
    dtypes and shapes, a gate_out mismatch, and a K mismatch all raise."""
    w, _deq = pair_experts
    scales = mx.zeros((1,), dtype=mx.uint8)
    x = mx.zeros((1, K), dtype=mx.float32)
    ids = mx.array(np.zeros(6, dtype=np.uint32))
    rw = mx.array(np.ones(6, dtype=np.float32))

    with pytest.raises(ValueError, match="no pair"):
        kq.gather_qmv_pair_swiglu(
            x, w, scales, "q2_k", ids, rw, GATE_OUT, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmv_pair_swiglu(
            x, w, scales, "iq2_xxs",
            mx.array(np.zeros(6, dtype=np.int32)), rw, GATE_OUT, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmv_pair_swiglu(
            x, w, scales, "iq2_xxs", ids,
            mx.array(np.ones(4, dtype=np.float32)), GATE_OUT, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmv_pair_swiglu(
            x, w, scales, "iq2_xxs", ids,
            rw.astype(mx.float16), GATE_OUT, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmv_pair_swiglu(
            x, w, scales, "iq2_xxs", ids, rw, GATE_OUT // 2, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmv_pair_swiglu(
            mx.zeros((2, K), dtype=mx.float32), w, scales, "iq2_xxs", ids,
            rw, GATE_OUT, 0.2)
    with pytest.raises(ValueError):
        kq.gather_qmv_pair_swiglu(
            mx.zeros((1, K // 2), dtype=mx.float32), w, scales, "iq2_xxs",
            ids, rw, GATE_OUT, 0.2)


# ---------------------------------------------------------------------------
# gather_qmv_expert_sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(ID_CASES))
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_expert_sum_matches_dq_reference(down_experts, case, dtype):
    """Summed output matches the f32-dequantize + f64 per-expert matvec sum."""
    w, deq = down_experts
    scales = mx.zeros((1,), dtype=mx.uint8)
    rng = np.random.default_rng(5)

    ids_np = np.array(ID_CASES[case], dtype=np.uint32)
    B = len(ids_np)
    x_np = (rng.standard_normal((B, K)) * 0.3).astype(np.float32)
    x = mx.array(x_np).astype(dtype)
    x_ref = np.array(x.astype(mx.float32), dtype=np.float64)

    got = kq.gather_qmv_expert_sum(x, w, scales, "q2_k", mx.array(ids_np))
    mx.eval(got)
    assert got.dtype == dtype
    assert got.shape == (1, N_DOWN)

    ref = np.zeros(N_DOWN)
    for b, e in enumerate(ids_np):
        ref += deq[e].astype(np.float64) @ x_ref[b]

    rtol, atol = PAIR_TOLS[dtype]
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32), dtype=np.float64).reshape(-1),
        ref,
        rtol=rtol,
        atol=atol,
        err_msg=f"{case} {dtype}",
    )


def test_expert_sum_matches_unfused_composition(down_experts):
    """Float32 I/O parity against the live composition the decode path
    replaces: per-expert gather_qmm rows summed on the host."""
    w, _deq = down_experts
    scales = mx.zeros((1,), dtype=mx.uint8)
    rng = np.random.default_rng(6)

    ids_np = np.array(ID_CASES["six_with_duplicate"], dtype=np.uint32)
    B = len(ids_np)
    x_np = (rng.standard_normal((B, K)) * 0.3).astype(np.float32)
    x = mx.array(x_np)

    got = kq.gather_qmv_expert_sum(x, w, scales, "q2_k", mx.array(ids_np))
    mx.eval(got)

    per = kq.gather_qmm(
        x.reshape(B, 1, 1, K),
        w, scales, "q2_k",
        rhs_indices=mx.array(ids_np).reshape(B, 1),
        transpose=True,
    )
    mx.eval(per)
    ref = np.array(
        per.astype(mx.float32), dtype=np.float64).reshape(B, N_DOWN).sum(axis=0)

    # gather_qmm promotes float32 x to bfloat16 output before the host-side
    # sum, so the composition carries the rounding; the dq reference test
    # pins the fused kernel tightly.
    np.testing.assert_allclose(
        np.array(got, dtype=np.float64).reshape(-1), ref, rtol=5e-2, atol=1e-2)


def test_expert_sum_rejects_bad_inputs(down_experts):
    """Fail-closed contract: uninstantiated codec, bad ids, a row-count
    mismatch, and a K mismatch all raise."""
    w, _deq = down_experts
    scales = mx.zeros((1,), dtype=mx.uint8)
    x = mx.zeros((6, K), dtype=mx.float32)
    ids = mx.array(np.zeros(6, dtype=np.uint32))

    with pytest.raises(ValueError, match="no expert-sum"):
        kq.gather_qmv_expert_sum(x, w, scales, "iq2_xxs", ids)
    with pytest.raises(ValueError):
        kq.gather_qmv_expert_sum(
            x, w, scales, "q2_k", mx.array(np.zeros(6, dtype=np.int32)))
    with pytest.raises(ValueError):
        kq.gather_qmv_expert_sum(
            x, w, scales, "q2_k", mx.array(np.zeros(4, dtype=np.uint32)))
    with pytest.raises(ValueError):
        kq.gather_qmv_expert_sum(
            mx.zeros((6, K // 2), dtype=mx.float32), w, scales, "q2_k", ids)
    with pytest.raises(ValueError):
        kq.gather_qmv_expert_sum(
            mx.zeros((6, 1, K), dtype=mx.float32), w, scales, "q2_k", ids)
