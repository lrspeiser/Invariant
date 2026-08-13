from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.math_known_identity_pipeline_control import (
    OUTPUT_PATH,
    KnownIdentityControlError,
    build_result,
    validate_result,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH_PATH = "docs/KNOWN_ANSWER_SUCCESS_FAILURE_WALKTHROUGH.md"


def _artifact() -> dict[str, object]:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> None:
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = canonical_sha256(unsigned)


def test_checked_artifact_is_exact_immutable_replay() -> None:
    value = _artifact()
    validate_result(value, ROOT)
    assert value == build_result(ROOT)
    for binding in value["source_bindings"].values():
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]


def test_success_reaches_proof_and_ranking() -> None:
    value = _artifact()
    pipeline = value["pipeline_result"]
    correct_id = value["candidate_roles"]["correct"]["artifact_id"]
    evaluation = next(
        row for row in pipeline["evaluations"] if row["artifact"]["artifact_id"] == correct_id
    )
    row = next(
        row for row in pipeline["candidate_rows"] if row["candidate"]["artifact_id"] == correct_id
    )
    assert evaluation["status"] == "pass"
    assert evaluation["terminal"]["kind"] == "complete"
    assert evaluation["counts"]["attempted_by_phase"] == {
        "cheap": 2,
        "symbolic": 1,
        "formal": 2,
        "observational": 0,
    }
    assert row["all_required_gates_passed"] is True
    assert row["pareto_front"] == 1
    assert len(pipeline["metric_receipts"]) == 1


def test_failure_stops_at_counterexample_before_proof_or_ranking() -> None:
    value = _artifact()
    pipeline = value["pipeline_result"]
    wrong_id = value["candidate_roles"]["wrong"]["artifact_id"]
    evaluation = next(
        row for row in pipeline["evaluations"] if row["artifact"]["artifact_id"] == wrong_id
    )
    row = next(
        row for row in pipeline["candidate_rows"] if row["candidate"]["artifact_id"] == wrong_id
    )
    assert evaluation["status"] == "reject"
    assert evaluation["terminal"] == {
        "kind": "stage",
        "outcome_id": "counterexample_screened",
    }
    assert evaluation["skipped_steps"] == ["exactly_verified", "prior_art_checked"]
    assert evaluation["counts"]["attempted_by_phase"]["formal"] == 0
    assert row["ranking_eligible"] is False
    assert row["pareto_front"] is None
    assert all(
        receipt["candidate"]["artifact_id"] != wrong_id for receipt in pipeline["metric_receipts"]
    )


def test_control_keeps_scientific_claim_boundary_closed() -> None:
    value = _artifact()
    assert value["claims"] == {
        "known_answer_control_passed": True,
        "wrong_formula_rejected_by_counterexample": True,
        "soft_metric_admitted_only_after_all_hard_gates": True,
        "blind_rediscovery_proved": False,
        "general_formula_discovery_proved": False,
        "novelty_established": False,
        "promotion_authorized": False,
    }
    assert value["exact_counts"]["promotion_authorized"] == 0


def test_public_walkthrough_binds_the_checked_receipt_and_boundaries() -> None:
    artifact_path = ROOT / OUTPUT_PATH
    value = _artifact()
    text = (ROOT / WALKTHROUGH_PATH).read_text(encoding="utf-8")
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() in text
    assert value["content_sha256"] in text
    for required in (
        "W(2)=4",
        "exactly_verified",
        "recorded as skipped",
        "It does **not** establish",
        "Pareto rank authorizes publication or promotion",
    ):
        assert required in text


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["wrong_candidate"].__setitem__("formal_phase_attempted", True),
        lambda value: value["correct_candidate"].__setitem__("pareto_front", 2),
        lambda value: value["claims"].__setitem__("general_formula_discovery_proved", True),
        lambda value: value["pipeline_result"]["counts"].__setitem__("ranked_candidates", 2),
    ),
)
def test_resealed_semantic_tampering_fails_closed(mutation) -> None:
    value = copy.deepcopy(_artifact())
    mutation(value)
    _reseal(value)
    with pytest.raises((KnownIdentityControlError, ValueError)):
        validate_result(value, ROOT)
