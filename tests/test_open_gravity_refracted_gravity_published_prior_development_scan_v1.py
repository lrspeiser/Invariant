from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_published_prior_development_scan_v1 as subject,
)


@lru_cache(maxsize=1)
def _receipt() -> dict[str, object]:
    return subject.build_receipt(subject.load_config(verify_package=False))


def test_config_freezes_pre_response_cells_multiplicity_and_no_tuning() -> None:
    config = subject.load_config(verify_package=False)
    subject.validate_config(config)
    assert config["admission_rule"]["parameter_cells_frozen_before_this_response_scan"] is True
    assert config["admission_rule"]["new_parameter_generation_or_repair"] is False
    assert config["parameter_contract"]["registered_refracted_gravity_cells"] == 9
    assert config["parameter_contract"]["registered_cell_multiplicity_charged"] == 9
    assert config["parameter_contract"]["response_fitted_continuous_parameters"] == 0
    assert config["access_scope"]["tuning_calls"] == 0


def test_predecessor_negative_result_solver_and_primary_benchmark_are_exact() -> None:
    receipts = subject.validate_predecessors(subject.load_config(verify_package=False))
    assert (
        receipts["FIXED_MEDIAN_DEVELOPMENT_RESULT"]["status"]
        == "NO_DEVELOPMENT_SIGNAL_FOR_FIXED_PUBLISHED_RG_CONTROL"
    )
    assert receipts["SCORING_RESOLUTION_SOLVER_AND_SOURCES"]["all_object_gates_pass"] is True
    assert receipts["PUBLISHED_PARAMETER_AND_OPERATOR_BENCHMARK"]["benchmark_suite"]["failed"] == 0


def test_published_parameter_cells_are_exact_and_have_six_epsilon_families() -> None:
    config = subject.load_config(verify_package=False)
    cells = subject.parameter_cells(config)
    assert len(cells) == 9
    assert len({row["id"] for row in cells}) == 9
    assert sum(row["id"] == "DISKMASS_UNIVERSAL_MEDIAN" for row in cells) == 1
    density = np.asarray([0.0, 1.0e-27, 1.0e-25, 1.0e-23])
    hashes = {
        subject.array_sha256(
            subject.rg.published_permittivity(
                density,
                epsilon_0=float(row["epsilon_0"]),
                rho_c=10.0 ** float(row["log10_rho_c_g_cm3"]),
                q_slope=float(row["Q"]),
            )
        )
        for row in cells
    }
    assert len(hashes) == 6


def test_source_failures_stop_before_any_response_loader(monkeypatch) -> None:
    called = {"response": 0}

    def failed_fields(_config):
        return [{"object_id": "synthetic", "source_and_solver_gates_pass": False}]

    def forbidden_loader(_config):
        called["response"] += 1
        raise AssertionError("response loader should not run")

    monkeypatch.setattr(subject, "build_source_fields", failed_fields)
    monkeypatch.setattr(subject.fixed_score.responses, "_load_phangs_responses", forbidden_loader)
    with pytest.raises(subject.PriorScanError, match="source/solver gate failed"):
        subject.build_receipt(subject.load_config(verify_package=False))
    assert called["response"] == 0


def test_full_prior_grid_retains_all_cells_but_refuses_underresolved_ranking() -> None:
    receipt = _receipt()
    assert len(receipt["source_field_rows"]) == 3
    assert len(receipt["parameter_cells"]) == 9
    assert receipt["access_accounting"]["registered_rg_field_rows"] == 54
    assert receipt["access_accounting"]["unique_rg_linear_solves"] == 36
    assert receipt["access_accounting"]["registered_rg_multiplicity"] == 9
    assert receipt["access_accounting"]["continuous_parameter_fits"] == 0
    assert len(receipt["phangs_object_eligibility"]) == 3
    assert len(receipt["sparc_object_eligibility"]) == 1
    assert receipt["response_radius_gate_failures"]
    assert receipt["phangs_object_scores"] == []
    assert receipt["sparc_object_scores"] == []
    for object_row in receipt["phangs_object_eligibility"] + receipt["sparc_object_eligibility"]:
        assert object_row["scores"] == {}
        assert object_row["eligibility_used_velocity_values"] is False
    assert receipt["access_accounting"]["response_velocity_values_used_for_scoring"] == 0
    assert receipt["access_accounting"]["object_candidate_scores_computed"] == 0
    assert receipt["access_accounting"]["best_cell_selection_events"] == 0


def test_all_source_solver_cells_pass_before_scoring() -> None:
    receipt = _receipt()
    assert receipt["numerical_failures"] == []
    for object_row in receipt["source_field_rows"]:
        assert object_row["source_and_solver_gates_pass"] is True
        for grid_key in ("fine", "convergence"):
            grid = object_row[grid_key]
            assert grid["unique_rg_solves"] == 6
            assert len(grid["profiles"]) == 10
            assert grid["source_mass_gate"] is True
            assert grid["solver_gate"] is True


def test_adjudication_retains_multiplicity_but_performs_no_ranking() -> None:
    adjudication = _receipt()["adjudication"]
    assert adjudication["registered_rg_cells"] == 9
    assert adjudication["multiplicity_charge"] == 9
    assert adjudication["performed"] is False
    assert adjudication["reason"] == "INSUFFICIENT_PREDECLARED_COMMON_CONVERGED_RADII"
    assert adjudication["rg_ranking_by_phangs_loss"] == []
    assert adjudication["best_rg_parameter_id"] is None
    assert adjudication["best_cell_is_development_selection_not_confirmation"] is False
    assert adjudication["multiplicity_adjusted_global_discovery_claimed"] is False
    assert adjudication["development_signal"] is False


def test_access_and_claim_boundaries_remain_narrow() -> None:
    receipt = _receipt()
    access = receipt["access_accounting"]
    assert access["responses_opened_after_all_source_gates"] is True
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert access[key] == 0
    claims = receipt["claim_boundary"]
    assert claims["all_nine_preregistered_cells_scored"] is False
    assert claims["best_cell_is_development_selection"] is False
    assert claims["global_significance_established"] is False
    assert claims["source_systematic_score_robustness_established"] is False
    assert claims["publication_ready"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "PUBLICATION_READY"),
        (("admission_rule", "new_parameter_generation_or_repair"), True),
        (("admission_rule", "all_source_and_solver_gates_complete_before_response_open"), False),
        (("parameter_contract", "registered_refracted_gravity_cells"), 10),
        (("parameter_contract", "registered_cell_multiplicity_charged"), 1),
        (("parameter_contract", "response_fitted_continuous_parameters"), 3),
        (("grid_contract", "fine_vs_convergence_maximum_relative_difference"), 1.0),
        (("access_scope", "tuning_calls"), 1),
    ],
)
def test_semantic_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(subject.load_config(verify_package=False))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(subject.PriorScanError):
        subject.validate_config(config)


def test_receipt_self_hash() -> None:
    receipt = _receipt()
    assert receipt["content_sha256"] == subject.content_sha256({**receipt, "content_sha256": ""})


def test_atomic_no_clobber(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    assert subject._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert subject._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(subject.PriorScanError):
        subject._atomic_no_clobber(path, b"two\n")
