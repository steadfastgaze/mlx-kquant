"""Source contracts for the q8 decode-loader implementation."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def _flat(path: str) -> str:
    return " ".join(_source(path).split())


def test_only_scalar_control_and_two_uint4_products_are_precompiled():
    metal = _flat("metal/kq_sdpa.metal")

    assert "256, 16, 4, false, false, scalar_dynamic" in metal
    assert "256, 16, 4, true, false, uint4_dynamic" in metal
    assert "256, 16, 4, true, true, uint4_byte_dynamic" in metal
    assert "scalar_fixed" not in metal
    assert "uint4_fixed" not in metal

    assert "instantiate_kq_sdpa_decode_gqa_q8(256, 8, 4)" in metal
    assert "instantiate_kq_sdpa_decode_gqa_q8(256, 16, 4)" not in metal


def test_loader_switch_changes_only_the_aligned_word_load_schedule():
    kernel = _flat("metal/mlx/backend/metal/kernels/kq_sdpa.h")

    assert "bool AlignedUint4Load = false" in kernel
    assert "bool VectorByteUnpack = false" in kernel
    assert "if constexpr (AlignedUint4Load)" in kernel
    assert "if constexpr (VectorByteUnpack)" in kernel
    assert "*reinterpret_cast<const device uint4*>(pk_w + kbase)" in kernel
    assert "*reinterpret_cast<const device uint4*>(pv_w + vbase)" in kernel
    assert "float4(as_type<uchar4>(kwj))" in kernel
    assert "float4(as_type<uchar4>(vwj))" in kernel
    assert "fma(float4(ks), kc, float4(kb))" in kernel
    assert "fma(float4(vs_), vc, float4(vb))" in kernel
    assert "FixedSequenceStrides" not in kernel
    assert "kPackedSequenceStride" not in kernel
    assert "kMetaSequenceStride" not in kernel
    assert "(size_t)kg * p.pk_seq + col" in kernel
    assert "(size_t)kg * p.pks_seq + (col >> 4)" in kernel
    assert "(size_t)kg * p.pv_seq + col" in kernel
    assert "(size_t)kg * p.pvs_seq + (col >> 4)" in kernel
    assert "fma(ks, float(kwj & 0xff), kb)" in kernel
    assert "fma(vs_, float(vwj & 0xff), vb)" in kernel


def test_served_geometry_scalar_and_uint4_assign_the_same_staging_indices():
    d4 = 64
    tile_rows = 16
    threads = 256

    scalar_indices = []
    for flat in range(threads):
        scalar_indices.extend(range(flat, tile_rows * d4, threads))

    vector_indices = []
    vectors_per_row = d4 // 4
    for flat in range(threads):
        for vi in range(flat, tile_rows * vectors_per_row, threads):
            row = vi // vectors_per_row
            col = (vi % vectors_per_row) * 4
            assert col % 4 == 0
            assert (col >> 4) == ((col + 3) >> 4)
            vector_indices.extend(row * d4 + col + lane for lane in range(4))

    expected = list(range(tile_rows * d4))
    assert sorted(scalar_indices) == expected
    assert sorted(vector_indices) == expected


def test_vector_byte_uint4_is_default_with_strict_kill_switches_and_counters():
    host = _flat("src/kquant_sdpa.cpp")
    binding = _flat("bindings.cpp")
    package = _source("mlx_kquant/__init__.py")

    assert 'std::getenv("KQ_SDPA_Q8_UINT4_LOAD")' in host
    assert 'std::string(env) == "0"' in host
    assert '"KQ_SDPA_Q8_UINT4_LOAD must be 0 or 1."' in host
    assert 'std::getenv("KQ_SDPA_Q8_VECTOR_BYTE_UNPACK")' in host
    assert 'std::string(unpack) == "1"' in host
    assert 'std::string(unpack) == "0"' in host
    assert '"KQ_SDPA_Q8_VECTOR_BYTE_UNPACK must be 0 or 1."' in host
    assert (
        'Q8LoaderConfig{ "uint4_byte_dynamic", Q8LoaderArm::Uint4ByteDynamic}'
    ) in host
    assert 'Q8LoaderConfig{"uint4_dynamic", Q8LoaderArm::Uint4Dynamic}' in host
    assert 'Q8LoaderConfig{"scalar_dynamic", Q8LoaderArm::ScalarDynamic}' in host
    assert 'kname += "_" + std::string(selected_loader.suffix)' in host
    assert "count_q8_loader_dispatch(" in host
    assert '"sdpa_q8_loader_debug"' in binding
    assert '"scalar_dynamic"' in binding
    assert '"uint4_dynamic"' in binding
    assert '"uint4_byte_dynamic"' in binding
    assert '"off_contract"' in binding
    assert '"scalar_fixed"' not in binding
    assert '"uint4_fixed"' not in binding
    assert "sdpa_q8_loader_debug" in package


def test_uint4_dispatch_falls_back_when_any_packed_row_is_not_aligned():
    host = _flat("src/kquant_sdpa.cpp")

    assert "bool has_uint4_row_alignment(array a)" in host
    assert "reinterpret_cast<uintptr_t>(a.buffer().raw_ptr())" in host
    assert "a.offset()" in host
    assert "address % 16 == 0" in host
    assert "head_stride % 4 == 0" in host
    assert "a.strides(2)) % 4 == 0" in host
    assert "has_uint4_row_alignment(pk_w)" in host
    assert "has_uint4_row_alignment(pv_w)" in host
    assert 'kname += "_scalar_dynamic"' in host
    assert "alignment_fallback = true" in host
    assert "count_q8_loader_dispatch(Q8LoaderArm::OffContract)" in host


def _loader_config_probe(
    uint4_load: str | None, vector_unpack: str | None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for name, value in (
        ("KQ_SDPA_Q8_UINT4_LOAD", uint4_load),
        ("KQ_SDPA_Q8_VECTOR_BYTE_UNPACK", vector_unpack),
    ):
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx_kquant as kq; print(kq.sdpa_q8_loader_debug()['selected_arm'])",
        ],
        env=env,
        capture_output=True,
        text=True,
    )


def test_q8_loader_environment_defaults_and_precedence():
    default = _loader_config_probe(None, None)
    assert default.returncode == 0, default.stderr
    assert default.stdout.strip() == "uint4_byte_dynamic"

    byte_control = _loader_config_probe("1", "0")
    assert byte_control.returncode == 0, byte_control.stderr
    assert byte_control.stdout.strip() == "uint4_dynamic"

    scalar_control = _loader_config_probe("0", "invalid")
    assert scalar_control.returncode == 0, scalar_control.stderr
    assert scalar_control.stdout.strip() == "scalar_dynamic"

    invalid_byte_setting = _loader_config_probe("1", "invalid")
    assert invalid_byte_setting.returncode != 0
    assert "KQ_SDPA_Q8_VECTOR_BYTE_UNPACK must be 0 or 1" in (
        invalid_byte_setting.stderr
    )
