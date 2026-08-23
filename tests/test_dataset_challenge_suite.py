from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import dataset_challenge_suite as D
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict[str, object]:
    return D.run_dataset_challenges(ROOT)


def test_all_four_dataset_kinds_execute_positive_and_mutation_controls() -> None:
    receipt = _receipt()
    D.validate_dataset_challenges(receipt, ROOT)
    assert receipt["summary"] == {
        "challenge_kinds": ["intervention", "noisy", "shifted", "unidentifiable"],
        "mutation_controls_rejected": 4,
        "positive_controls_passed": 4,
        "status": "PASS_EXECUTABLE_DATASET_CHALLENGES",
        "total": 4,
    }
    assert all(row["positive_control_passed"] for row in receipt["results"])
    assert all(row["mutation_control_rejected"] for row in receipt["results"])


def test_intervention_rows_execute_an_exact_do_contrast() -> None:
    by_kind = {row["kind"]: row for row in _receipt()["results"]}
    evidence = by_kind["intervention"]["evidence"]
    assert evidence["positive_exact_rows"] == evidence["actual_do_labeled_rows"] == 4
    assert evidence["constant_do_contrast"] == "3"
    assert evidence["mutation_exact_rows"] < evidence["actual_do_labeled_rows"]
    assert evidence["observational_rows_relabelled_as_interventions"] is False


def test_noise_intervals_retain_distinct_models_and_reject_an_outside_mutation() -> None:
    by_kind = {row["kind"]: row for row in _receipt()["results"]}
    evidence = by_kind["noisy"]["evidence"]
    assert evidence["distinct_point_predictions_retained"] == 2
    assert len(evidence["interval_compatible_models"]) == 2
    assert evidence["mutation_compatible_rows"] < evidence["interval_rows"]
    assert evidence["point_centers_treated_as_unique_truth"] is False


def test_shift_control_exposes_a_train_exact_but_deployment_wrong_alias() -> None:
    by_kind = {row["kind"]: row for row in _receipt()["results"]}
    evidence = by_kind["shifted"]["evidence"]
    assert evidence["train_only_alias_train_exact_rows"] == evidence["train_rows"]
    assert evidence["train_only_alias_deployment_exact_rows"] < evidence["deployment_rows"]
    assert evidence["deployment_exact_rows"] == evidence["deployment_rows"]


def test_unidentifiable_control_retains_both_observationally_equal_mechanisms() -> None:
    by_kind = {row["kind"]: row for row in _receipt()["results"]}
    evidence = by_kind["unidentifiable"]["evidence"]
    assert evidence["compatible_mechanisms"] == 2
    assert evidence["observational_predictions_equal"] is True
    assert evidence["intervention_predictions_diverge"] is True
    assert evidence["forced_unique_mechanism_rejected"] is True
    assert evidence["required_conclusion"] == "UNDERDETERMINED_RETAIN_MULTIPLE_MECHANISMS"


def test_resealed_receipt_cannot_turn_a_calibration_control_into_a_claim() -> None:
    changed = copy.deepcopy(_receipt())
    changed["claims"]["train_fit_establishes_deployment_validity"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(D.DatasetChallengeError, match="boundary"):
        D.validate_dataset_challenges(changed)


def test_resealed_receipt_cannot_rebind_its_source_path() -> None:
    changed = copy.deepcopy(_receipt())
    changed["source_bindings"]["source"] = dict(changed["source_bindings"]["config"])
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(D.DatasetChallengeError, match="source path"):
        D.validate_dataset_challenges(changed)


def test_stored_dataset_receipt_exactly_replays_current_sources() -> None:
    receipt = json.loads((ROOT / D.OUTPUT_PATH).read_text(encoding="utf-8"))
    D.validate_dataset_challenges(receipt, ROOT)
