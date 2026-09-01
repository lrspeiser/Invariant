from __future__ import annotations

import copy
import inspect
from functools import cache
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_heracles_s4g_model_lifted_3d_source_builder_v1 as builder,
)


def _config() -> dict[str, object]:
    return builder.load_config(verify_package=False)


@cache
def _profiles() -> dict[str, object]:
    return builder.build_profiles(_config())


def test_exact_public_source_and_geometry_dependencies() -> None:
    config = _config()
    acquisition, geometry, operator = builder._load_dependencies(config)
    paths = builder._source_paths(config, acquisition)

    assert len(paths) == 35
    assert sum(path.stat().st_size for path in paths.values()) == 90_449_926
    assert [row["object_id"] for row in geometry["objects"]] == list(builder._OBJECTS)
    assert operator["access_state"]["response_rows_opened"] == 0
    assert acquisition["scientific_boundary"]["response_rows_opened"] == 0


def test_each_builder_has_data_paper_and_independent_benchmark() -> None:
    anchors = _config()["published_anchor_contract"]
    assert {row["id"] for row in anchors["measurement_sources"]} == {
        "S4G_P5_STELLAR",
        "S4G_GEOMETRY",
        "THINGS_HI",
        "HERACLES_CO21",
    }
    assert {row["id"] for row in anchors["independent_benchmarks"]} == {
        "FREEMAN_EXPONENTIAL_DISK",
        "CASERTANO_FINITE_THICKNESS",
        "POINT_MASS_AND_SECH2",
    }
    assert len(anchors["mandatory_before_response"]) == 8
    assert "blocks response scoring" in anchors["failure_policy"]


def test_geometry_variants_precede_response_and_preserve_uncertainty() -> None:
    config = _config()
    _acquisition, geometry, _operator = builder._load_dependencies(config)
    counts = {
        row["object_id"]: len(builder.geometry_variants(config, row)) for row in geometry["objects"]
    }
    assert counts == {
        "NGC2903": 3,
        "NGC2976": 3,
        "NGC3198": 3,
        "NGC3521": 7,
        "NGC4214": 7,
    }
    assert config["cell_contract"]["response_based_cell_selection"] is False
    assert config["cell_contract"]["retain_every_failure"] is True


def test_analytic_and_published_operator_benchmarks_pass() -> None:
    report = _profiles()["benchmarks"]
    assert all(report["passed"].values())
    assert report["vertical_normalization"] == 1.0
    assert report["point_mass_far_field_relative_error"] < 3.0e-2
    assert report["freeman_max_relative_error"] < 8.0e-2
    assert 0.0 < report["finite_thickness_force_ratio_at_2p2rd"] < 1.0


def test_five_real_objects_and_all_393_cells_are_retained() -> None:
    profiles = _profiles()
    assert profiles["cell_count"] == 393
    assert [row["object_id"] for row in profiles["objects"]] == list(builder._OBJECTS)
    assert [len(row["cell_summaries"]) for row in profiles["objects"]] == [77, 77, 77, 81, 81]
    assert len(profiles["cell_summary_root_sha256"]) == 64
    assert len(profiles["cell_profile_root_sha256"]) == 64


def test_source_mass_beam_and_convergence_gates_pass_without_response() -> None:
    for row in _profiles()["objects"]:
        primary = row["primary_summary"]
        assert 1.0e7 < primary["stellar_mass_msun"] < 3.0e11
        assert 1.0e6 < primary["hi_helium_mass_msun"] < 1.0e11
        assert 20.0 < primary["target_fwhm_pc"] < 3000.0
        assert row["convergence"]["passed"] is True


def test_invalid_sip_sensitivity_cells_are_retained_not_hidden() -> None:
    failures = [
        (row["object_id"], cell["cell_id"], cell.get("failure_code"))
        for row in _profiles()["objects"]
        for cell in row["cell_summaries"]
        if cell.get("status") == "FAILED_RETAINED"
    ]
    assert failures == [
        (
            "NGC2976",
            "S4G_SIP_HEADER_SENSITIVITY_PRIMARY_PHYSICS",
            "S4G_SIP_WCS_NONCONVERGENCE",
        ),
        (
            "NGC4214",
            "S4G_SIP_HEADER_SENSITIVITY_PRIMARY_PHYSICS",
            "S4G_SIP_WCS_NONCONVERGENCE",
        ),
    ]


def test_scientific_boundary_is_source_only_and_model_lifted() -> None:
    config = _config()
    boundary = config["scientific_boundary"]
    assert boundary["source_files_opened"] == 35
    assert boundary["response_or_velocity_files_opened"] == 0
    assert boundary["response_rows_opened"] == 0
    assert boundary["scores_computed"] == 0
    assert boundary["models_fit"] == 0
    assert boundary["observed_full_3d_geometry"] is False
    assert boundary["model_lifted_2p5d_only"] is True


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("source_inventory", "exact_file_count"), 34),
        (("cell_contract", "response_based_cell_selection"), True),
        (("cell_contract", "retain_every_failure"), False),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("scientific_boundary", "observed_full_3d_geometry"), True),
        (("claims", "refracted_gravity_supported"), True),
        (("output_path",), "runs/gravity/forged.json"),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[object, ...], value: object) -> None:
    mutated = copy.deepcopy(_config())
    cursor: object = mutated
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(builder.SourceBuilderError):
        builder.validate_config(mutated)


def test_no_caller_selected_source_or_output_paths() -> None:
    assert tuple(inspect.signature(builder.write_packet).parameters) == ()
    assert tuple(inspect.signature(builder.check_packet).parameters) == ()
    assert builder._parser()._actions[1].choices == ("write", "check", "status")


def test_atomic_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert builder._atomic_no_clobber(path, b"one") == "CREATED"
    assert builder._atomic_no_clobber(path, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(builder.SourceBuilderError):
        builder._atomic_no_clobber(path, b"two")
    assert path.read_bytes() == b"one"
