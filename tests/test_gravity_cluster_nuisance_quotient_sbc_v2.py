from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as v1
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / v2.CONFIG_PATH
RESULT = ROOT / v2.RESULT_PATH
RECEIPT = ROOT / v2.RECEIPT_PATH


def config_hash() -> str:
    return v2.file_sha256(CONFIG)


def result_summary() -> dict[str, object]:
    archive = np.load(RESULT, allow_pickle=False)
    return json.loads(str(archive["summary"].item()))


def test_v2_is_separately_preregistered_and_v1_is_hash_bound_unchanged() -> None:
    config = v2.load_config(CONFIG, config_hash())
    assert config["status"] == "separately_preregistered_before_bounded_v2_run"
    assert config["v1_evidence_bindings"] == {
        "implementation_source": {
            "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sbc.py",
            "file_sha256": (
                "0eaad327544454c3540bfd99e87530caf3f0ace5e9d0065798125d98de863ba3"
            ),
        },
        "config": {
            "path": "configs/gravity_cluster_nuisance_quotient_sbc_v1.json",
            "file_sha256": (
                "f9a9acf4ee4f1558ce97ab489423c7f43c09bb38af764c7d17de5109433b314c"
            ),
        },
        "result": {
            "path": (
                "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1/"
                "bounded-synthetic-sbc.npz"
            ),
            "file_sha256": (
                "90071721c53e4a635443a652541fc3eeae6c5fd9822bc84a77aa460ecab375f3"
            ),
        },
        "receipt": {
            "path": "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1.json",
            "file_sha256": (
                "afb4aa64850fc6449272d8ac2174c7c6c7dfa20c319259c48c5cecbd11dcaa11"
            ),
        },
    }
    for row in config["v1_evidence_bindings"].values():
        assert v2.file_sha256(ROOT / row["path"]) == row["file_sha256"]
    with pytest.raises(RuntimeError, match="config hash"):
        v2.load_config(CONFIG, "0" * 64)


def test_v2_changes_address_both_v1_failures_without_changing_regimes() -> None:
    config = v2.load_config(CONFIG, config_hash())
    assert [row["scenario_id"] for row in config["scenarios"]] == [
        row["scenario_id"] for row in v1.SCENARIOS
    ]
    assert [row["normalized_noise_sigma"] for row in config["scenarios"]] == [
        row["normalized_noise_sigma"] for row in v1.SCENARIOS
    ]
    assert config["candidate_inference"]["change_from_v1"] == {
        "adaptation_sweeps_multiplier": 4,
        "settling_sweeps_multiplier": 8,
        "retained_sweeps_multiplier": 8,
        "retained_draws_per_chain_multiplier": 4,
        "reason": "V1 maximum per-fit Rhat exceeded 3.5 in both scenarios",
    }
    reference_change = config["independent_reference"]["change_from_v1"]
    assert reference_change["finite_prior_importance_removed"] is True
    assert reference_change["adaptive_tempering_added"] is True
    assert config["gates"]["v1_thresholds_retroactively_changed"] is False


def test_finite_simulation_coverage_gates_follow_preregistered_formula() -> None:
    tolerances = v2.finite_simulation_coverage_tolerances(32)
    assert tolerances == {
        "0.5": pytest.approx(0.34060921676911454),
        "0.8": pytest.approx(0.27873737341529164),
        "0.9": pytest.approx(0.2168655300614687),
    }
    config = v2.load_config(CONFIG, config_hash())
    assert config["gates"]["coverage_tolerances_for_32_simulations"] == tolerances
    assert config["gates"]["finite_simulation_familywise_z"] == 3.5


def test_independent_reference_does_not_call_candidate_transition_or_orbits() -> None:
    source = inspect.getsource(v2.run_independent_smc_replicate)
    assert "active_transition" not in source
    assert "orbit_sweep" not in source
    assert "apply_orbit_move" not in source
    assert v2.REFERENCE["algorithmically_independent_of_candidate_kernel"] is True
    assert v2.REFERENCE["canonical_active_transition_called"] is False
    assert v2.REFERENCE["canonical_orbit_move_called"] is False


def test_v2_result_is_bounded_synthetic_and_failed_retained() -> None:
    archive = np.load(RESULT, allow_pickle=False)
    assert archive["truth_units"].shape == (64, 17)
    assert archive["candidate_normalized_ranks"].shape == (64, 10)
    assert archive["reference_normalized_ranks"].shape == (64, 10)
    assert archive["candidate_coverage"].shape == (64, 10, 3)
    summary = result_summary()
    assert summary["passed"] is False
    assert summary["decision"] == (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V2_FAILED_RESULT_RETAINED"
    )
    assert summary["v1_diagnosis"] == {
        "short_chain_mixing_failure": True,
        "finite_sobol_importance_collapse": True,
        "v1_result_or_thresholds_changed": False,
    }
    assert summary["data_boundary"] == v2.DATA_BOUNDARY
    assert summary["claim_boundary"] == v2.CLAIM_BOUNDARY


def test_independent_reference_fixed_v1_oracle_collapse_and_passed_its_gates() -> None:
    for scenario in result_summary()["scenario_results"]:
        assert scenario["reference_minimum_stage_conditional_ess_fraction"] >= 0.69
        assert (
            scenario["reference_maximum_standardized_replicate_mean_difference"]
            <= 0.35
        )
        assert scenario["reference_minimum_unique_initial_ancestor_fraction"] >= 0.05
        assert scenario["reference_coverage_passed"] is True
        assert scenario["reference_rank"]["maximum_absolute_mean_rank_z"] < 2.3


def test_candidate_calibration_agrees_but_mixing_still_fails() -> None:
    moderate, weak = result_summary()["scenario_results"]
    assert moderate["maximum_fit_rhat"] == pytest.approx(1.7675585521675952)
    assert weak["maximum_fit_rhat"] == pytest.approx(1.7521232008449714)
    assert moderate["minimum_fit_bulk_ess"] == pytest.approx(48.21302295073525)
    assert weak["minimum_fit_bulk_ess"] == pytest.approx(48.79898501935122)
    for scenario in (moderate, weak):
        assert scenario["candidate_coverage_passed"] is True
        assert scenario[
            "maximum_absolute_candidate_reference_mean_rank_difference"
        ] < 0.015
        assert scenario["maximum_fit_rhat"] > 1.2
        assert scenario["minimum_fit_bulk_ess"] < 50.0
        assert scenario["passed"] is False


def test_call_budget_and_orbit_control_are_exact() -> None:
    summary = result_summary()
    assert summary["call_accounting"] == {
        "actual_candidate_synthetic_likelihood_evaluations": 783053,
        "actual_independent_reference_synthetic_likelihood_evaluations": 9952518,
        "actual_total_synthetic_likelihood_evaluations": 10735571,
        "frozen_maximum_total_synthetic_likelihood_evaluations": 38275072,
        "real_forward_model_evaluations": 0,
    }
    assert summary["orbit_invariance_control"]["passed"] is True
    assert summary["orbit_invariance_control"][
        "maximum_absolute_composite_difference"
    ] < 1e-12


def test_receipt_is_valid_hash_bound_and_claim_safe() -> None:
    checked = v2.check(CONFIG, config_hash(), RECEIPT)
    assert checked == {
        "valid": True,
        "passed": False,
        "config_sha256": config_hash(),
        "receipt_sha256": v2.file_sha256(RECEIPT),
        "synthetic_likelihood_evaluations": 10735571,
        "real_forward_model_evaluations": 0,
        "candidate_production_runs": 0,
        "scientific_claim_allowed": False,
    }
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["status"] == "bounded_synthetic_sbc_v2_failed_result_retained"
    assert receipt["controls"]["reference_uses_candidate_transition"] is False
    assert receipt["controls"]["reference_uses_candidate_orbits"] is False
    assert receipt["claim_boundary"]["CP5_7_through_CP5_10_complete"] is False


def test_v2_files_have_no_development_workspace_binding() -> None:
    for path in (CONFIG, RECEIPT, Path(v2.__file__)):
        text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
        assert "work/" not in text
