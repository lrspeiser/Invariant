from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.reviewed_g4_candidate_solar_evaluator import (
    ACTION_SHA256,
    CALIBRATION_IMPLEMENTATION_SHA256,
    CANDIDATE_ID,
    OUTPUT_CHANNELS,
    PARSER_HASHES,
    REQUIRED_REGISTRATION_HASHES,
    build_reviewed_g4_solar_evaluator_readiness,
    reviewed_g4_candidate_solar_evaluator,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "reviewed_g4_candidate_solar_evaluator_readiness.json"
DESCRIPTOR_PATH = ROOT / "configs" / "reviewed_g4_candidate_solar_evaluator.json"
ARTIFACT_PATH = (
    ROOT / "runs" / "engine" / "reviewed-g4-candidate-solar-evaluator-readiness.json"
)
SOURCE_PATH = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / "reviewed_g4_candidate_solar_evaluator.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate() -> dict:
    return {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "role": "generated_candidate",
        "data_eligibility": dict(ELIGIBILITY),
    }


def _future_context() -> dict:
    registration = {name: "1" * 64 for name in REQUIRED_REGISTRATION_HASHES}
    registration.update(PARSER_HASHES)
    registration[
        "raw_to_calibrated_transform_and_covariance_implementation_sha256"
    ] = CALIBRATION_IMPLEMENTATION_SHA256
    bundle = {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "path": "runs/engine/future-registered-g4-solar-prediction-bundle.json",
        "file_sha256": registration["prediction_bundle_file_sha256"],
        "content_sha256": registration["prediction_bundle_content_sha256"],
        "schema_version": "sigma-candidate-solar-prediction-bundle-1.0",
        "output_channels": OUTPUT_CHANNELS,
        "universal_parameter_count": 0,
        "object_specific_gravity_parameter_count": 0,
        "weak_field_solution_sha256": registration["weak_field_solution_sha256"],
        "state_estimation_contract_sha256": registration[
            "state_estimation_contract_sha256"
        ],
        "instrument_calibration_contract_sha256": (
            CALIBRATION_IMPLEMENTATION_SHA256
        ),
        "covariance_contract_sha256": registration["covariance_contract_sha256"],
        "likelihood_contract_sha256": registration["likelihood_contract_sha256"],
        "split_commitment_sha256": registration[
            "tracking_session_split_commitment_sha256"
        ],
        "stopping_rule_sha256": registration["stopping_rule_sha256"],
        "data_eligibility": dict(ELIGIBILITY),
        "observational_data_opened": False,
    }
    source = {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "decision": "pass",
        "interval_certificate_sha256": registration[
            "registered_real_source_interval_instantiation_certificate_sha256"
        ],
        "tail_profile_certificate_sha256": registration[
            "registered_trace_tail_profile_certificate_sha256"
        ],
        "observational_data_opened": False,
    }
    return {
        "data_eligibility": dict(ELIGIBILITY),
        "observational_opening_authorized": False,
        "registration_hashes": registration,
        "prediction_bundle": bundle,
        "real_source_certificate": source,
    }


def test_artifact_rebuilds_exactly_and_preserves_all_seals() -> None:
    config = _load(CONFIG_PATH)
    stored = _load(ARTIFACT_PATH)
    rebuilt = build_reviewed_g4_solar_evaluator_readiness(config, ROOT)
    assert rebuilt == stored
    assert stored["content_sha256"] == (
        "17e9410233a546f8ac946b3ec1993f00fbd564080eef1b7c40e41669894f5fc5"
    )
    assert _file_sha(ARTIFACT_PATH) == (
        "4fba90244ae58bae71192010356d53cf6d5c0f97517b97cef94a72a74a1cb49c"
    )
    assert stored["decision"] == "blocked"
    assert stored["observational_authorization"] is False
    assert stored["observational_data_opened"] is False
    assert stored["primary_record_access_count"] == 0
    assert stored["tracking_target_values_opened"] is False
    assert stored["paid_llm_spend_usd"] == 0.0
    assert stored["data_eligibility"] == ELIGIBILITY


def test_descriptor_binds_exact_candidate_action_callback_and_source() -> None:
    descriptor = _load(DESCRIPTOR_PATH)
    assert descriptor["candidate_id"] == CANDIDATE_ID
    assert descriptor["action_sha256"] == ACTION_SHA256
    assert descriptor["callback"].endswith(":reviewed_g4_candidate_solar_evaluator")
    assert descriptor["artifact_sha256"] == _file_sha(SOURCE_PATH)
    assert descriptor["data_eligibility"] == ELIGIBILITY
    artifact = _load(ARTIFACT_PATH)
    assert artifact["descriptor_implementation_ready"] is True
    assert artifact["real_source_prediction_bundle_registered"] is False
    assert artifact["candidate_use_authorized"] is False


def test_current_registration_fills_only_descriptor_and_blocks_every_other_hash() -> None:
    artifact = _load(ARTIFACT_PATH)
    decision = artifact["current_evaluator_decision"]
    expected_missing = sorted(
        set(REQUIRED_REGISTRATION_HASHES)
        - {"reviewed_candidate_solar_evaluator_descriptor_sha256"}
    )
    assert decision["decision"] == "blocked"
    assert decision["filled_registration_hash_count"] == 1
    assert decision["missing_registration_hashes"] == expected_missing
    assert artifact["unfilled_real_bundle_and_data_fields"] == expected_missing
    assert set(artifact["newly_filled_registration_fields"]) == {
        "reviewed_candidate_solar_evaluator_descriptor_sha256",
        "reviewed_candidate_solar_evaluator_implementation_readiness_sha256",
    }


def test_empty_or_partial_registration_fails_closed() -> None:
    context = {
        "data_eligibility": dict(ELIGIBILITY),
        "observational_opening_authorized": False,
    }
    empty = reviewed_g4_candidate_solar_evaluator(_candidate(), context)
    assert empty["decision"] == "blocked"
    assert len(empty["missing_registration_hashes"]) == len(
        REQUIRED_REGISTRATION_HASHES
    )
    context["registration_hashes"] = {
        "reviewed_candidate_solar_evaluator_descriptor_sha256": "2" * 64
    }
    partial = reviewed_g4_candidate_solar_evaluator(_candidate(), context)
    assert partial["decision"] == "blocked"
    assert partial["filled_registration_hash_count"] == 1
    assert "selected_primary_file_root_sha256" in partial[
        "missing_registration_hashes"
    ]


def test_fully_registered_synthetic_contract_still_needs_separate_opening() -> None:
    result = reviewed_g4_candidate_solar_evaluator(_candidate(), _future_context())
    assert result["decision"] == "blocked"
    assert result["blocker"] == "separate_observational_opening_authorization_required"
    assert result["readiness"] == "fully_registered_bundle_validated"
    assert result["missing_registration_hashes"] == []
    assert result["observational_data_opened"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("candidate", "candidate identity changed"),
        ("eligibility", "context eligibility changed"),
        ("authorization", "cannot authorize observation opening"),
        ("bundle", "future Solar bundle hash mismatch"),
        ("parser", "future Solar registration changed"),
    ],
)
def test_identity_policy_and_future_binding_tampering_is_rejected(
    mutation: str, message: str
) -> None:
    candidate = _candidate()
    context = _future_context()
    if mutation == "candidate":
        candidate["action_sha256"] = "0" * 64
    elif mutation == "eligibility":
        context["data_eligibility"] = {**ELIGIBILITY, "paid_llm_calls": True}
    elif mutation == "authorization":
        context["observational_opening_authorized"] = True
    elif mutation == "bundle":
        context["prediction_bundle"]["weak_field_solution_sha256"] = "3" * 64
    else:
        context["registration_hashes"]["verified_RSR_parser_sha256"] = "4" * 64
    with pytest.raises(ValueError, match=message):
        reviewed_g4_candidate_solar_evaluator(candidate, context)


def test_synthetic_fixtures_are_controls_not_candidate_evidence() -> None:
    fixtures = _load(ARTIFACT_PATH)["synthetic_fixtures"]
    gr = fixtures["GR_known_answer"]
    covariance = fixtures["covariance_propagation"]
    assert gr["decision"] == "pass"
    assert gr["role"] == "calibration_only_not_candidate_evidence"
    assert set(gr["golden_statuses"].values()) == {"pass"}
    assert len(gr["golden_statuses"]) == 5
    assert covariance["decision"] == "pass"
    assert covariance["cross_channel_covariance_nonzero"] is True
    assert covariance["shared_calibration_correlations_retained"] is True
    assert covariance["primary_target_records_opened"] is False


def test_bound_source_or_descriptor_tampering_cannot_rebuild() -> None:
    config = _load(CONFIG_PATH)
    for binding_name in ("candidate_dossier", "reviewed_evaluator_descriptor"):
        tampered = copy.deepcopy(config)
        tampered["source_bindings"][binding_name]["file_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="changed"):
            build_reviewed_g4_solar_evaluator_readiness(tampered, ROOT)
