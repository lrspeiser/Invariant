import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.galaxy_direct_observable_evaluator import (
    galaxy_direct_observable_evaluator,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY, evaluator_binding

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "configs" / "promotion_pipeline_fail_closed.json"
DESCRIPTOR = ROOT / "configs" / "promotion_galaxy_direct_observable_evaluator.json"
STATUS = ROOT / "runs" / "engine" / "galaxy-direct-observable-evaluator-status.json"


def _candidate() -> dict:
    return {
        "candidate_id": "STC2-sealed-galaxy-fixture",
        "data_eligibility": dict(ELIGIBILITY),
    }


def _context() -> dict:
    return {
        "input_lineage_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }


def test_missing_or_unregistered_prediction_bundle_never_opens_data() -> None:
    missing = galaxy_direct_observable_evaluator(_candidate(), _context())
    assert missing["decision"] == "blocked"
    assert missing["blocker"] == "missing_candidate_direct_observable_prediction_bundle"
    assert missing["observational_data_opened"] is False
    assert missing["source_registrations_loaded"] == 0

    candidate = _candidate()
    candidate["direct_observable_prediction_provenance"] = {
        "bundle_id": "not-registered",
        "bundle_binding_sha256": "b" * 64,
        "candidate_action_sha256": "c" * 64,
        "prediction_content_sha256": "d" * 64,
        "observable_contract_sha256": "e" * 64,
    }
    unregistered = galaxy_direct_observable_evaluator(candidate, _context())
    assert unregistered["decision"] == "blocked"
    assert unregistered["blocker"] == (
        "unregistered_candidate_direct_observable_prediction_bundle"
    )
    assert unregistered["registered_prediction_bundle_count"] == 0


def test_opened_eligibility_and_malformed_provenance_are_rejected() -> None:
    opened = _context()
    opened["data_eligibility"]["redshift_distance_inputs"] = True
    with pytest.raises(ValueError, match="eligibility"):
        galaxy_direct_observable_evaluator(_candidate(), opened)

    malformed = _candidate()
    malformed["direct_observable_prediction_provenance"] = {"bundle_id": "partial"}
    with pytest.raises(ValueError, match="fields are not exact"):
        galaxy_direct_observable_evaluator(malformed, _context())


def test_descriptor_is_exactly_hash_bound() -> None:
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    artifact = ROOT / descriptor["artifact_path"]
    assert descriptor["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    descriptor["artifact_path"] = str(artifact.resolve())
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
    galaxy_stage = pipeline["stages"][4]
    assert evaluator_binding(descriptor) == galaxy_stage[
        "required_evaluator_binding_sha256"
    ]


def test_status_artifact_records_a_fully_sealed_scaffold() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    body = {key: value for key, value in status.items() if key != "content_sha256"}
    assert status["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert status["decision_counts"] == {"blocked": 70}
    assert status["registered_prediction_bundle_count"] == 0
    assert status["source_registrations_loaded"] == 0
    assert status["observational_data_opened"] is False
