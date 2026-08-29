from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_cluster_pressure_covariance_scoring_pilot as pilot,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return pilot.build_receipt(ROOT)


def test_pilot_retains_failed_robustness_gate_and_advances_only_cp5_1(
    receipt: dict[str, object],
) -> None:
    assert receipt["decision"] == "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS"
    assert receipt["CP5_1_status"] == (
        "DEVELOPMENT_PRESSURE_COVARIANCE_SCORED_NOT_COMPONENT_COMPLETE"
    )
    assert receipt["advanced_goal_evidence"] == {
        "CP5.1": (
            "frozen_development_pressure_predictions_scored_with_reconstructed_"
            "released_correlation"
        )
    }
    assert receipt["completed_goal_evidence"] == {}
    assert receipt["claims"]["CP5_1_advances_to_development_scored"]
    assert not receipt["claims"]["CP5_1_complete"]
    assert not receipt["claims"]["independent_replication"]
    assert not receipt["claims"]["scientific_promotion_authorized"]


def test_exact_sample_and_no_access_expansion(receipt: dict[str, object]) -> None:
    assert receipt["sample_summary"] == {
        "clusters": list(pilot.CLUSTERS),
        "pressure_rows": 54,
        "development_train_pressure_rows": 30,
        "development_holdout_pressure_rows": 24,
        "unscored_outer_boundary_rows": 8,
        "same_release_confirmation_rows": 0,
        "independent_rows": 0,
        "lensing_rows": 0,
    }
    assert receipt["access_boundary"]["formula_refits"] == 0
    assert receipt["access_boundary"]["nuisance_refits"] == 0
    assert receipt["access_boundary"]["model_selection_operations"] == 0


def test_full_covariance_preserves_aggregate_ranking_but_fails_cluster_gate(
    receipt: dict[str, object],
) -> None:
    holdout = receipt["aggregates"]["development_holdout"]
    candidate = holdout["models"]["ITEM59_CROSS_SCALE_BOUNDARY"]
    nfw = holdout["models"]["GR_PLUS_NFW"]
    assert candidate["diagonal_score"] == pytest.approx(4.202682093406597)
    assert candidate["full_covariance_score"] == pytest.approx(3.8629233674315238)
    assert nfw["diagonal_score"] == pytest.approx(11.038980283164767)
    assert nfw["full_covariance_score"] == pytest.approx(12.62010166320426)
    assert holdout["candidate_advantage"]["full_covariance"] == pytest.approx(
        8.757178295772736
    )
    assert holdout["ranking_concordant"]
    assert holdout["candidate_cluster_wins"] == {
        "diagonal": 6,
        "full_covariance": 4,
    }
    assert receipt["robustness_gates"] == {
        "primary_full_covariance_candidate_advantage_positive": True,
        "primary_full_covariance_cluster_wins_meet_threshold": False,
        "primary_diagonal_and_full_ranking_concordant": True,
    }


def test_per_cluster_deltas_are_retained(receipt: dict[str, object]) -> None:
    holdout = {
        row["cluster"]: row
        for row in receipt["per_cluster"]
        if row["split"] == "development_holdout"
    }
    full_wins = {
        cluster
        for cluster, row in holdout.items()
        if row["candidate_advantage"]["full_covariance"] > 0.0
    }
    assert full_wins == {"A1644", "A2142", "A2319", "A3266"}
    assert holdout["A85"]["candidate_advantage"]["diagonal"] > 0.0
    assert holdout["A85"]["candidate_advantage"]["full_covariance"] < 0.0
    assert holdout["ZW1215"]["candidate_advantage"]["diagonal"] > 0.0
    assert holdout["ZW1215"]["candidate_advantage"]["full_covariance"] < 0.0


def test_conditioning_is_valid(receipt: dict[str, object]) -> None:
    conditioning = receipt["conditioning_summary"]
    thresholds = receipt["adjudication_thresholds"]
    assert conditioning["maximum_condition_number"] == pytest.approx(100.68596553838101)
    assert conditioning["maximum_condition_number"] < thresholds[
        "maximum_condition_number"
    ]
    assert conditioning["maximum_inverse_identity_residual"] < thresholds[
        "maximum_inverse_identity_residual"
    ]
    assert all(receipt["numerical_gates"].values())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["access_boundary"].__setitem__(
            "independent_target_rows_opened", 1
        ),
        lambda value: value["access_boundary"].__setitem__("formula_refits", 1),
        lambda value: value["sample_freeze"][0]["pressure_rows"].pop(),
        lambda value: value["model_freeze"]["candidate"].__setitem__("refit", True),
        lambda value: value["model_freeze"]["strongest_frozen_comparator"].__setitem__(
            "model_id", "GR_PLUS_EINASTO"
        ),
        lambda value: value["adjudication_thresholds"].__setitem__(
            "minimum_primary_full_covariance_cluster_wins", 4
        ),
        lambda value: value["claim_boundary"].__setitem__("CP5_1_complete", True),
    ],
)
def test_access_selection_threshold_and_claim_mutations_fail_closed(
    mutation: object,
) -> None:
    config = copy.deepcopy(pilot.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(pilot.GravityClusterPressureCovariancePilotError):
        pilot.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / pilot.OUTPUT_PATH).read_text(encoding="utf-8"))
    pilot.validate_receipt(stored, ROOT)
    assert stored == pilot.build_receipt(ROOT)
