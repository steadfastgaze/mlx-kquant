"""Source-level contracts for the opt-in q8 dimension-parallel merge."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def _flat(path: str) -> str:
    return " ".join(_source(path).split())


def test_dimension_parallel_merge_flag_reaches_primitive_identity():
    header = _flat("src/kquant.h")
    host = _flat("src/kquant_sdpa.cpp")
    binding = _flat("bindings.cpp")

    assert (
        '"tile_c"_a = 0, nb::kw_only(), '
        '"dimension_parallel_merge"_a = false, "stream"_a = nb::none()'
    ) in binding
    assert "bool dimension_parallel_merge = false" in header
    assert "dimension_parallel_merge_(dimension_parallel_merge)" in header
    assert "bool dimension_parallel_merge_;" in header
    assert "tile_c, dimension_parallel_merge)" in host
    assert ("dimension_parallel_merge_ == o.dimension_parallel_merge_") in host


def test_dimension_parallel_merge_selector_falls_back_off_contract():
    host = _flat("src/kquant_sdpa.cpp")
    selector = (
        "dimension_parallel_merge_ && N >= 8192 && splits == 128 && "
        "stage_ == 2 && compute_ == 1 && tile_c_ == 16 && B == 1 && "
        "n_q_heads == 16 && n_kv_heads == 2 && D == 256"
    )
    assert selector in host
    assert (
        'use_dimension_parallel_merge ? "kq_sdpa_q8_merge_dim8" : '
        '"kq_sdpa_gqa_2pass_2_float_" + std::to_string(D)'
    ) in host
    assert "use_dimension_parallel_merge ? 256 : 32" in host


def test_dimension_parallel_merge_is_precompiled_without_scout_surfaces():
    host = _source("src/kquant_sdpa.cpp")
    metal = _flat("metal/kq_sdpa.metal")
    kernel = _source("metal/mlx/backend/metal/kernels/kq_sdpa.h")

    assert (
        'instantiate_kernel( "kq_sdpa_q8_merge_dim8", '
        "kq_sdpa_q8_merge_dim_parallel, 128, 256, 8)"
    ) in metal
    assert "void kq_sdpa_q8_merge_dim_parallel(" in kernel
    assert "kq_get_kernel(d, kname);" in host
    assert 'R"metal(' not in host
    assert "KQ_SDPA_Q8_MERGE" not in host
    assert "_sdpa_decode_q8_partials" not in host
