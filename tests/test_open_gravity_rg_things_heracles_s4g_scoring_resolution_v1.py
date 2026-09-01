from __future__ import annotations

import copy
import inspect
from functools import cache
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_things_heracles_s4g_scoring_resolution_v1 as resolution,
)


def _config() -> dict[str, object]:
    return resolution.load_config(verify_package=False)


@cache
def _receipt() -> dict[str, object]:
    return resolution.build_receipt(_config())


def test_real_source_solver_and_primary_benchmark_bindings() -> None:
    predecessors = resolution.validate_predecessors(_config())
    assert set(predecessors) == {
        "FIVE_OBJECT_REAL_SOURCE_BUILDER",
        "AUDITED_HIGH_RESOLUTION_DST_PCG_MECHANICS",
        "PRIMARY_PAPER_RG_OPERATOR_BENCHMARK",
    }
    assert predecessors["FIVE_OBJECT_REAL_SOURCE_BUILDER"]["cell_count"] == 393
    assert (
        predecessors["AUDITED_HIGH_RESOLUTION_DST_PCG_MECHANICS"]["all_object_gates_pass"] is True
    )
    assert predecessors["PRIMARY_PAPER_RG_OPERATOR_BENCHMARK"]["benchmark_suite"]["failed"] == 0


def test_source_ledger_has_exact_five_objects_and_393_cells() -> None:
    source_config, _acquisition, geometry, paths, expected = resolution._source_evidence(_config())
    assert source_config["objects"] == list(resolution._OBJECTS)
    assert set(geometry) == set(resolution._OBJECTS)
    assert len(paths) == 35
    assert len(expected) == 393


def test_target_free_dst_and_pcg_benchmarks_pass() -> None:
    report = resolution.mechanics.run_target_free_benchmarks(_config())
    assert report["all_pass"] is True
    assert all(report["checks"].values())


def test_five_object_fine_convergence_radius_mask_passes() -> None:
    receipt = _receipt()
    assert receipt["status"] == "PASS_FIVE_OBJECT_SOURCE_ONLY_RG_SCORING_RESOLUTION_MASK"
    assert receipt["decision"] == "READY_FOR_FIXED_HELD_SPARC_RESPONSE_SCORE"
    assert receipt["all_object_gates_pass"] is True
    assert receipt["response_blind_radius_summary"] == {
        "registered_points": 1455,
        "eligible_points": 1315,
        "ineligible_points": 140,
        "selection_used_velocity_values": False,
    }


def test_every_object_passes_and_failed_radii_are_retained() -> None:
    receipt = _receipt()
    expected = {
        "NGC2903": (257, 34, 0.8),
        "NGC2976": (268, 23, 0.8),
        "NGC3198": (249, 42, 0.8),
        "NGC3521": (279, 12, 0.8),
        "NGC4214": (262, 29, 0.75),
    }
    for row in receipt["objects"]:
        eligible, ineligible, minimum = expected[row["object_id"]]
        assert row["all_object_gates_pass"] is True
        assert row["eligible_radius_count"] == eligible
        assert row["ineligible_radius_count"] == ineligible
        assert row["eligible_radius_min_kpc"] == minimum
        assert row["eligible_radius_max_kpc"] == 15.0
        assert eligible + ineligible == 291
        assert any(not point["response_scoring_eligible"] for point in row["radius_adjudication"])


def test_source_mass_solver_and_positive_field_gates_pass() -> None:
    for row in _receipt()["objects"]:
        assert all(row["object_gates"].values())
        for level in ("fine", "convergence"):
            solve = row[level]
            assert solve["source_builder_mass_relative_error"] <= 2.0e-9
            assert solve["solver_metrics"]["newton_relative_residual"] <= 1.0e-8
            assert solve["solver_metrics"]["refracted_gravity"]["converged"] is True
            assert solve["solver_metrics"]["refracted_gravity"]["relative_residual"] <= 1.0e-8


def test_no_response_or_tuning_access() -> None:
    boundary = _receipt()["scientific_boundary"]
    assert boundary["unique_source_files_opened_per_build"] == 35
    assert boundary["response_files_opened"] == 0
    assert boundary["response_rows_opened"] == 0
    assert boundary["response_values_opened"] == 0
    assert boundary["scores_computed"] == 0
    assert boundary["tuning_calls"] == 0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "CONFIRMED"),
        (("source_cell", "selection_from_response"), True),
        (("grid_contract", "fine_nodes_per_axis"), 121),
        (("operator_contract", "epsilon_0"), 0.5),
        (("operator_contract", "response_parameter_fitting"), True),
        (("benchmark_contract", "radius_gate_is_response_blind"), False),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("claim_boundary", "refracted_gravity_preferred"), True),
        (("output_path",), "runs/gravity/forged.json"),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[object, ...], value: object) -> None:
    mutated = copy.deepcopy(_config())
    cursor: object = mutated
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    with pytest.raises(resolution.ScoringResolutionError):
        resolution.validate_config(mutated)


def test_no_caller_selected_input_or_output_paths() -> None:
    assert tuple(inspect.signature(resolution.write_receipt).parameters) == ()
    assert tuple(inspect.signature(resolution.check_receipt).parameters) == ()
    assert resolution._parser()._actions[1].choices == ("write", "check", "status")


def test_atomic_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert resolution._atomic_no_clobber(path, b"one") == "CREATED"
    assert resolution._atomic_no_clobber(path, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(resolution.ScoringResolutionError):
        resolution._atomic_no_clobber(path, b"two")
    assert path.read_bytes() == b"one"
