from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_manuscript_evidence_package as package

ROOT = Path(__file__).resolve().parents[1]


def test_development_evidence_is_packaged_without_paper_overclaim() -> None:
    receipt = package.build_receipt(ROOT)
    assert receipt["decision"] == "DEVELOPMENT_MANUSCRIPT_EVIDENCE_PACKAGED_NOT_PAPER_READY"
    assert set(receipt["completed_goal_evidence"]) == {
        "CP12.2",
        "CP12.4",
        "CP12.5",
        "CP12.7",
        "CP12.8",
        "CP12.9",
    }
    assert set(receipt["blocked_goal_evidence"]) == {
        "CP12.1",
        "CP12.3",
        "CP12.6",
        "CP12.10",
        "CP12.11",
        "CP12.12",
    }
    assert receipt["claims"]["development_evidence"] is True
    assert receipt["claims"]["independent_replication"] is False
    assert receipt["claims"]["bounded_paper_ready"] is False


def test_all_candidate_rows_and_absolute_relative_summaries_are_retained() -> None:
    receipt = package.build_receipt(ROOT)
    assert receipt["counts"]["per_row_candidate_predictions"] == 233
    rows = receipt["per_row_candidate_predictions"]
    assert len({row["row_id"] for row in rows}) == 233
    assert {row["split"] for row in rows} == {
        "development_train",
        "development_holdout",
        "confirmation",
    }
    assert all(
        {"observed", "predicted", "error", "log_residual", "standardized_square"} <= set(row)
        for row in rows
    )
    assert set(receipt["split_summaries"]) == {
        "development_train",
        "development_holdout",
        "confirmation",
    }


def test_access_ledger_keeps_same_release_and_independent_access_separate() -> None:
    receipt = package.build_receipt(ROOT)
    access = receipt["access_ledger"]
    assert access["confirmation_response_files_opened_after_freeze"] == 8
    assert access["same_release_confirmation_rows"] == 77
    assert access["direct_lensing_likelihood_evaluations"] == 0
    assert access["inferred_total_mass_rows"] == 0
    assert access["independent_target_rows_opened"] == 0
    assert access["independent_observational_authorization"] is False


def test_negative_uncertainty_prior_art_and_claim_boundaries_are_all_present() -> None:
    receipt = package.build_receipt(ROOT)
    assert receipt["comparators_and_ablations"]["ablations"]
    assert receipt["negative_and_numerical_controls"]["synthetic_recovery"]
    assert receipt["negative_and_numerical_controls"]["false_selection"]
    assert receipt["uncertainty_and_alternative_cause_boundary"]["source_covariance_blockers"]
    calibration = receipt["quotient_sampler_calibration_and_newtonian_boundary"]
    assert calibration["v1_passed"] is False
    assert calibration["v2_passed"] is False
    assert calibration["v3_synthetic_sbc_passed"] is True
    assert calibration["newtonian_control_unlock"] is True
    assert calibration["candidate_production_unlock"] is False
    assert calibration["newtonian_external_approval_present"] is False
    covariance = receipt["development_pressure_covariance_boundary"]
    assert covariance["scoring_decision"] == "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS"
    assert covariance["reconstructed_matrices"] == 8
    assert covariance["scored_pressure_rows"] == 54
    assert covariance["CP5_2_through_CP5_6_complete"] is False
    assert receipt["prior_art_boundary"]["closest_behavioral_neighbor"]["source_id"] == (
        "PENNER_MODIFIED_GRAS_AQUAL_2026"
    )
    assert set(receipt["claim_tracks"]) == {
        "bounded_empirical_publication",
        "physical_mechanism",
        "universal_theory",
    }


def test_new_cross_scale_group_and_strata_evidence_keeps_claim_ceilings() -> None:
    receipt = package.build_receipt(ROOT)
    ben = receipt["shared_ben_synthetic_and_real_boundary"]
    assert ben["synthetic_raw_candidates"] == 240
    assert ben["synthetic_equivalence_classes"] == 60
    assert ben["synthetic_grammar_mechanics_validated"] is True
    assert ben["synthetic_recovery_is_scientific_evidence"] is False
    assert ben["local_sparc_confirmation_sealed_for_descendant"] is False
    assert ben["v2_blocked_before_payload_load"] is True
    assert ben["xcop_predictor_output_mapping_ready"] is False
    assert ben["v2_payload_loader_present"] is False
    assert ben["v2_real_scoring_executed"] is False
    group = receipt["group_scale_source_boundary"]
    assert group["candidate_lanes"] == 3
    assert group["ready_lanes"] == 0
    assert group["CP10_1_complete"] is False
    assert group["CP10_2_complete"] is False
    assert group["scientific_result_emitted"] is False
    strata = receipt["cluster_strata_boundary"]
    assert strata["development_clusters"] == 8
    assert strata["CP5_11_predictor_strata_frozen"] is True
    assert strata["candidate_absolute_gate_passed"] is False
    assert strata["candidate_cluster_wins"] == 4
    assert strata["minimum_cluster_wins"] == 5
    assert strata["candidate_object_win_gate_passed"] is False
    assert strata["frozen_stratum_explains_covariance_flips"] is False
    assert strata["CP5_13_complete"] is False
    assert strata["causal_variable_identified"] is False
    assert strata["scientific_claim_allowed"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["claim_boundary"].__setitem__("bounded_paper_ready", True),
            "claim boundary",
        ),
        (
            lambda value: value["environment_freeze"].__setitem__("numpy", "latest"),
            "environment",
        ),
        (
            lambda value: value["source_bindings"][0].__setitem__("content_sha256", "0" * 64),
            "content changed",
        ),
    ],
)
def test_claim_environment_and_evidence_mutations_fail_closed(mutation: object, match: str) -> None:
    config = copy.deepcopy(package.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    if match == "content changed":
        with pytest.raises(package.GravityClusterManuscriptPackageError, match=match):
            package._load_sources(ROOT, config)
    else:
        with pytest.raises(package.GravityClusterManuscriptPackageError, match=match):
            package.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / package.OUTPUT_PATH).read_text(encoding="utf-8"))
    package.validate_receipt(stored, ROOT)
    assert stored == package.build_receipt(ROOT)
