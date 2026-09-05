"""Standalone exact checks for Qwen QSA selection and K4/V4 gathering."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq

GROUPS = 513
SELECTED_GROUPS = 512
COMPRESS_RATIO = 4
SELECTED_WIDTH = SELECTED_GROUPS * COMPRESS_RATIO + COMPRESS_RATIO - 1
KV_HEADS = 2
HEAD_WIDTH = 256
SINK_TOKENS = 128
TILE_TOKENS = 128
RECORD_BYTES = 35072
K_CODES = slice(0, 16384)
K_SCALE = 16384
K_ZERO = 16896
K_TOKEN_SCALE = 17408
V_CODES = slice(17664, 34048)
V_CHANNEL_SCALE = 34048
V_TOKEN_SCALE = 34560
V_ZERO = 34816
METAL_AVAILABLE = hasattr(mx, "metal") and mx.metal.is_available()


def _bf16(values: np.ndarray) -> mx.array:
    return mx.array(values.astype(np.float32)).astype(mx.bfloat16)


def _write_fp16(record: np.ndarray, offset: int, values: np.ndarray) -> None:
    payload = np.asarray(values, dtype="<f2").reshape(-1).view(np.uint8)
    record[offset : offset + payload.size] = payload


def _pack_nibbles(codes: np.ndarray) -> np.ndarray:
    return codes[..., 0::2] | (codes[..., 1::2] << np.uint8(4))


def _hand_authored_records() -> np.ndarray:
    """Build nonzero records whose metadata is exactly representable in FP16."""
    records = np.zeros((1, KV_HEADS, RECORD_BYTES), dtype=np.uint8)
    token = np.arange(TILE_TOKENS, dtype=np.uint16)[:, None]
    channel = np.arange(HEAD_WIDTH, dtype=np.uint16)[None, :]
    for head in range(KV_HEADS):
        record = records[0, head]
        key_codes = ((3 * token + 5 * channel + 7 * head) % 16).astype(np.uint8)
        value_codes = ((7 * token + 3 * channel + 5 * head + 1) % 16).astype(np.uint8)
        record[K_CODES] = _pack_nibbles(key_codes).reshape(-1)
        record[V_CODES] = _pack_nibbles(value_codes).reshape(-1)

        key_scale = 0.5 + 0.25 * (np.arange(HEAD_WIDTH) % 4)
        key_zero = 0.5 * ((np.arange(HEAD_WIDTH) % 3) - 1)
        key_token_scale = 0.5 + 0.5 * (np.arange(TILE_TOKENS) % 4)
        value_channel_scale = 0.5 + 0.25 * (np.arange(HEAD_WIDTH) % 4)
        value_token_scale = 0.25 + 0.25 * (np.arange(TILE_TOKENS) % 4)
        value_zero = 0.5 * ((np.arange(TILE_TOKENS) % 3) - 1)
        _write_fp16(record, K_SCALE, key_scale)
        _write_fp16(record, K_ZERO, key_zero)
        _write_fp16(record, K_TOKEN_SCALE, key_token_scale)
        _write_fp16(record, V_CHANNEL_SCALE, value_channel_scale)
        _write_fp16(record, V_TOKEN_SCALE, value_token_scale)
        _write_fp16(record, V_ZERO, value_zero)
    return records


def _case(*, tail_tokens: int = 2048):
    """Build a deterministic public synthetic layout covering every storage class."""
    visible_count = GROUPS * COMPRESS_RATIO + 1
    frontier = visible_count - 1
    scores = np.arange(GROUPS, dtype=np.float32).reshape(1, 1, GROUPS)
    scores[..., 0] = float(GROUPS + 1)
    records = mx.array(_hand_authored_records())
    sink = np.arange(KV_HEADS * SINK_TOKENS * HEAD_WIDTH, dtype=np.float32).reshape(
        1, KV_HEADS, SINK_TOKENS, HEAD_WIDTH
    )
    tail = (10_000_000 + np.arange(KV_HEADS * tail_tokens * HEAD_WIDTH)).reshape(
        1, KV_HEADS, tail_tokens, HEAD_WIDTH
    )
    pending = np.full((1, KV_HEADS, 1, HEAD_WIDTH), 20_000_000, dtype=np.float32)
    return (
        mx.array(scores),
        records,
        _bf16(sink),
        _bf16(sink + 500_000),
        _bf16(tail),
        _bf16(tail + 500_000),
        _bf16(pending),
        _bf16(pending + 500_000),
        visible_count,
        frontier,
    )


def _reference_selected(
    scores: np.ndarray, visible_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stable score selection, then ascending physical context order."""
    groups = scores.shape[-1]
    ids = np.arange(groups, dtype=np.int32)
    ranked = np.lexsort((ids, -scores.reshape(-1)))[:SELECTED_GROUPS]
    selected_groups = np.sort(ranked)
    body = (
        selected_groups[:, None] * COMPRESS_RATIO + np.arange(COMPRESS_RATIO)
    ).reshape(-1)
    tail = np.arange(
        groups * COMPRESS_RATIO, groups * COMPRESS_RATIO + 3, dtype=np.int32
    )
    logical = np.concatenate((body, tail))
    valid = logical < visible_count
    logical = np.where(valid, logical, -1).astype(np.int32)
    return logical.reshape(1, 1, -1), valid.reshape(1, 1, -1)


def _reference_gather(
    logical: np.ndarray,
    valid: np.ndarray,
    records: mx.array,
    sink: mx.array,
    tail: mx.array,
    pending: mx.array,
    *,
    frontier: int,
    values: bool,
) -> np.ndarray:
    """Independently reconstruct every selected storage class."""
    tail_tokens = tail.shape[2]
    record_host = np.asarray(records)
    body_frontier = SINK_TOKENS + record_host.shape[0] * TILE_TOKENS
    result = np.zeros((SELECTED_WIDTH, KV_HEADS, HEAD_WIDTH), dtype=np.float32)
    sink_host = np.asarray(sink.astype(mx.float32))[0]
    tail_host = np.asarray(tail.astype(mx.float32))[0]
    pending_host = np.asarray(pending.astype(mx.float32))[0]
    for row, token in enumerate(logical.reshape(-1)):
        if not valid.reshape(-1)[row]:
            continue
        if token < SINK_TOKENS:
            result[row] = sink_host[:, token]
        elif token < body_frontier:
            tile, local = divmod(token - SINK_TOKENS, TILE_TOKENS)
            for head in range(KV_HEADS):
                result[row, head] = _reference_record_row(
                    record_host[tile, head], local, values=values
                )
        elif token < frontier:
            result[row] = tail_host[:, (token - SINK_TOKENS) % tail_tokens]
        else:
            result[row] = pending_host[:, token - frontier]
    rounded = np.asarray(_bf16(result).astype(mx.float32))
    return rounded.transpose(1, 0, 2)[None, None]


def _read_fp16(record: np.ndarray, offset: int, count: int) -> np.ndarray:
    payload = record[offset : offset + count * 2]
    return np.frombuffer(payload.tobytes(), dtype="<f2").astype(np.float32)


def _inverse_sylvester(values: np.ndarray) -> np.ndarray:
    """Apply a normalized width-256 Sylvester transform with explicit butterflies."""
    work = np.asarray(values, dtype=np.float32).copy()
    width = 1
    while width < HEAD_WIDTH:
        groups = work.reshape(-1, 2 * width)
        left = groups[:, :width].copy()
        right = groups[:, width:].copy()
        groups[:, :width] = left + right
        groups[:, width:] = left - right
        width *= 2
    return work * np.float32(1.0 / np.sqrt(HEAD_WIDTH))


def _reference_record_row(
    record: np.ndarray, token: int, *, values: bool
) -> np.ndarray:
    if values:
        packed = record[V_CODES].reshape(TILE_TOKENS, HEAD_WIDTH // 2)[token]
        code = np.empty(HEAD_WIDTH, dtype=np.uint8)
        code[0::2] = packed & np.uint8(0x0F)
        code[1::2] = packed >> np.uint8(4)
        channel_scale = _read_fp16(record, V_CHANNEL_SCALE, HEAD_WIDTH)
        token_scale = _read_fp16(record, V_TOKEN_SCALE, TILE_TOKENS)[token]
        zero = _read_fp16(record, V_ZERO, TILE_TOKENS)[token]
        rotated = (code.astype(np.float32) * token_scale + zero) * channel_scale
    else:
        packed = record[K_CODES].reshape(TILE_TOKENS, HEAD_WIDTH // 2)[token]
        code = np.empty(HEAD_WIDTH, dtype=np.uint8)
        code[0::2] = packed & np.uint8(0x0F)
        code[1::2] = packed >> np.uint8(4)
        scale = _read_fp16(record, K_SCALE, HEAD_WIDTH)
        zero = _read_fp16(record, K_ZERO, HEAD_WIDTH)
        token_scale = _read_fp16(record, K_TOKEN_SCALE, TILE_TOKENS)[token]
        rotated = (code.astype(np.float32) * scale + zero) * token_scale
    return _inverse_sylvester(rotated)


def _invoke(args, *, stream=None):
    return kq.qwen4_qsa_select_gather_k4v4(
        *args[:8],
        visible_count=args[8],
        frontier=args[9],
        stream=stream,
    )


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_select_gather_matches_independent_selection_and_storage_reference():
    args = _case()
    expected_selected, expected_valid = _reference_selected(
        np.asarray(args[0]), args[-2]
    )
    expected_keys = _reference_gather(
        expected_selected,
        expected_valid,
        args[1],
        args[2],
        args[4],
        args[6],
        frontier=args[-1],
        values=False,
    )
    expected_values = _reference_gather(
        expected_selected,
        expected_valid,
        args[1],
        args[3],
        args[5],
        args[7],
        frontier=args[-1],
        values=True,
    )
    actual = _invoke(args)
    mx.eval(*actual)
    selected, valid, keys, values = actual
    assert np.array_equal(np.asarray(selected), expected_selected)
    assert np.array_equal(np.asarray(valid), expected_valid)
    assert np.array_equal(np.asarray(keys.astype(mx.float32)), expected_keys)
    assert np.array_equal(np.asarray(values.astype(mx.float32)), expected_values)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_select_gather_compacts_lazy_strided_scores():
    args = list(_case())
    score_storage = mx.stack((args[0], mx.zeros_like(args[0])), axis=-1)
    args[0] = score_storage[..., 0]
    scores = np.arange(GROUPS, dtype=np.float32).reshape(1, 1, GROUPS)
    scores[..., 0] = float(GROUPS + 1)
    expected_selected, expected_valid = _reference_selected(scores, args[-2])
    expected_keys = _reference_gather(
        expected_selected,
        expected_valid,
        args[1],
        args[2],
        args[4],
        args[6],
        frontier=args[-1],
        values=False,
    )
    expected_values = _reference_gather(
        expected_selected,
        expected_valid,
        args[1],
        args[3],
        args[5],
        args[7],
        frontier=args[-1],
        values=True,
    )
    actual = _invoke(args)
    mx.eval(*actual)
    assert np.array_equal(np.asarray(actual[0]), expected_selected)
    assert np.array_equal(np.asarray(actual[1]), expected_valid)
    assert np.array_equal(np.asarray(actual[2].astype(mx.float32)), expected_keys)
    assert np.array_equal(np.asarray(actual[3].astype(mx.float32)), expected_values)


@pytest.mark.skipif(not METAL_AVAILABLE, reason="requires Metal")
def test_select_gather_accepts_dynamic_exact_tail_capacity():
    args = _case(tail_tokens=8320)
    expected_selected, expected_valid = _reference_selected(
        np.asarray(args[0]), args[-2]
    )
    actual = _invoke(args)
    mx.eval(*actual)
    assert np.array_equal(np.asarray(actual[0]), expected_selected)
    assert np.array_equal(np.asarray(actual[1]), expected_valid)


def test_select_gather_fails_closed_before_graph_creation():
    args = list(_case())
    with pytest.raises(ValueError, match="scores must be float32"):
        bad = [args[0].astype(mx.bfloat16), *args[1:]]
        _invoke(bad)
    with pytest.raises(ValueError, match="one pending token"):
        bad = [*args[:-1], args[-1] - 1]
        _invoke(bad)
    with pytest.raises(ValueError, match="exact_tail_keys"):
        bad = [*args[:4], args[4][:, :, :-1], *args[5:]]
        _invoke(bad)
    with pytest.raises(ValueError, match="Metal stream"):
        _invoke(args, stream=mx.cpu)


def test_qsa_source_contract_binds_the_public_closure_kernels():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "kquant_qwen4_qsa.cpp").read_text()
    header = (
        root / "metal" / "mlx" / "backend" / "metal" / "kernels" / "kq_qwen4_qsa.h"
    ).read_text()
    assert "kq_qwen4_qsa_project_rope" in source
    assert "kq_qwen4_qsa_stable_select" in source
    assert "kq_qwen4_qsa_k4v4_mutable_gather" in source
    assert "[[kernel]] void kq_qwen4_qsa_project_rope" in header
    assert "[[kernel]] void kq_qwen4_qsa_stable_select" in header
    assert "[[kernel]] void kq_qwen4_qsa_k4v4_mutable_gather" in header
