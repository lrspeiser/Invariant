from __future__ import annotations

import copy
import json
from functools import lru_cache

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as subject,
)


@lru_cache(maxsize=1)
def _packet() -> dict[str, object]:
    return subject.build_receipt(subject.load_config(verify_package=False))


def _boundary_only(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[[0, -1], :, :] = values[[0, -1], :, :]
    result[:, [0, -1], :] = values[:, [0, -1], :]
    result[:, :, [0, -1]] = values[:, :, [0, -1]]
    return result


def test_config_freezes_real_source_paper_and_no_response_gate() -> None:
    config = subject.load_config(verify_package=False)
    subject.validate_config(config)
    assert config["admission_rule"]["real_input_required"] is True
    assert config["admission_rule"]["primary_paper_or_exact_analytic_benchmark_required"] is True
    assert config["grid_contract"]["fine_spacing_kpc"] == 0.25
    assert config["grid_contract"]["convergence_spacing_kpc"] == 0.3125
    assert config["scientific_boundary"]["response_files_opened"] == 0
    assert config["operator_contract"]["response_parameter_fitting"] is False


def test_all_predecessor_artifacts_and_receipts_are_exact() -> None:
    config = subject.load_config(verify_package=False)
    receipts = subject.validate_predecessors(config)
    assert set(receipts) == {
        "REAL_SOURCE_BUILDER",
        "FULL3D_SOLVER_BRIDGE",
        "REFRACTED_GRAVITY_PRIMARY_BENCHMARK",
        "REFRACTED_GRAVITY_225_BY_9_SOURCE_SCREEN",
    }


def test_dst_solver_recovers_discrete_manufactured_solution() -> None:
    grid = base.make_grid(25)
    exact = (
        np.cos(np.pi * grid.x / 2.0) * np.cos(np.pi * grid.y / 2.0) * np.cos(np.pi * grid.z / 2.0)
        + 0.07 * grid.z
    )
    rhs = base._constant_laplacian(exact, grid.spacing)
    solved, residual = subject.solve_poisson_dst(rhs, _boundary_only(exact), grid.spacing)
    assert np.max(np.abs(solved - exact)) < 1.0e-11
    assert residual < 1.0e-11


def test_pcg_solver_recovers_variable_coefficient_solution() -> None:
    grid = base.make_grid(21)
    exact = (
        np.cos(np.pi * grid.x / 2.0) * np.cos(np.pi * grid.y / 2.0) * np.cos(np.pi * grid.z / 2.0)
        + 0.03 * grid.x
    )
    coefficient = (
        0.661 + 0.339 * (1.0 + np.tanh(1.79 * (0.3 * grid.x - 0.2 * grid.y + 0.1 * grid.z))) / 2.0
    )
    rhs = base._variable_divergence(exact, coefficient, grid.spacing)
    solved, metrics = subject.solve_variable_pcg(
        rhs,
        _boundary_only(exact),
        coefficient,
        grid.spacing,
        relative_tolerance=1.0e-12,
        absolute_tolerance=0.0,
        max_iterations=100,
    )
    assert metrics["converged"] is True
    assert np.max(np.abs(solved - exact)) < 1.0e-9
    assert metrics["relative_residual"] < 1.0e-9


def test_target_free_benchmarks_all_pass() -> None:
    result = subject.run_target_free_benchmarks(subject.load_config(verify_package=False))
    assert result["all_pass"] is True
    assert all(result["checks"].values())
    assert result["metrics"]["invalid_coefficient_rejected"] is True


def test_real_source_high_resolution_packet_is_response_blind_and_retains_masks() -> None:
    packet = _packet()
    assert packet["target_free_benchmarks"]["all_pass"] is True
    assert packet["all_object_gates_pass"] is True
    assert len(packet["objects"]) == 3
    summary = packet["response_blind_radius_summary"]
    assert summary["registered_points"] == 873
    assert summary["eligible_points"] + summary["ineligible_points"] == 873
    assert summary["selection_used_velocity_values"] is False
    assert packet["scientific_boundary"]["response_files_opened"] == 0
    assert packet["scientific_boundary"]["response_rows_opened"] == 0
    assert packet["scientific_boundary"]["scores_computed"] == 0
    for row in packet["objects"]:
        assert row["fine"]["nodes_per_axis"] == 241
        assert row["convergence"]["nodes_per_axis"] == 193
        assert len(row["radius_adjudication"]) == 291
        assert row["eligible_radius_count"] > 0
        for radial in row["radius_adjudication"]:
            if radial["response_scoring_eligible"]:
                assert radial["positive_finite"] is True
                assert radial["fine_cells_per_radius"] >= 2.0
                assert max(radial["fine_vs_convergence_relative_difference"].values()) <= 0.05


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "PUBLICATION_READY"),
        (("admission_rule", "real_input_required"), False),
        (("grid_contract", "fine_nodes_per_axis"), 17),
        (("operator_contract", "epsilon_0"), 0.5),
        (("operator_contract", "response_parameter_fitting"), True),
        (("scientific_boundary", "response_files_opened"), 1),
        (("claim_boundary", "refracted_gravity_preferred"), True),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(subject.load_config(verify_package=False))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(subject.ScoringResolutionError):
        subject.validate_config(config)


def test_invalid_solver_inputs_fail_closed() -> None:
    zeros = np.zeros((5, 5, 5), dtype=np.float64)
    with pytest.raises(subject.ScoringResolutionError):
        subject.solve_variable_pcg(
            zeros,
            zeros,
            zeros,
            1.0,
            relative_tolerance=1.0e-10,
            absolute_tolerance=0.0,
            max_iterations=10,
        )
    with pytest.raises(subject.ScoringResolutionError):
        subject.solve_poisson_dst(zeros, np.zeros((7, 7, 7)), 1.0)


def test_receipt_self_hash_and_deterministic_rebuild() -> None:
    packet = _packet()
    assert packet["content_sha256"] == subject.content_sha256({**packet, "content_sha256": ""})
    rebuilt = subject.build_receipt(subject.load_config(verify_package=False))
    assert subject.canonical_bytes(rebuilt) == subject.canonical_bytes(packet)


def test_atomic_writer_refuses_nonidentical_existing_file(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    assert subject._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert subject._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(subject.ScoringResolutionError):
        subject._atomic_no_clobber(path, b"two\n")
    assert path.read_bytes() == b"one\n"


def test_receipt_mutation_is_rejected_without_reopening_scientific_inputs() -> None:
    packet = copy.deepcopy(_packet())
    packet["claim_boundary"]["observational_fit_tested"] = True
    packet["content_sha256"] = subject.content_sha256({**packet, "content_sha256": ""})
    with pytest.raises(subject.ScoringResolutionError):
        subject.validate_receipt_payload(subject.load_config(verify_package=False), packet)


def test_config_file_is_canonical_json_semantically() -> None:
    config = subject.load_config(verify_package=False)
    loaded = json.loads(subject._repo_path(subject.CONFIG_PATH).read_text(encoding="utf-8"))
    assert loaded == config
