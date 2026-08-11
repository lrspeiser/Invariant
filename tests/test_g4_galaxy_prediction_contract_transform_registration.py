from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.g4_galaxy_prediction_contract_transform_registration import (
    ACTION_SHA256,
    BRANCH_CONTRACT_SHA256,
    CANDIDATE_ID,
    PREDICTION_BUNDLE_CONTRACT_SHA256,
    TRANSFORM_CONTRACT_SHA256,
    apply_registered_linear_calibration,
    build_g4_galaxy_prediction_contract_transform_registration,
    validate_prediction_bundle_contract,
    validate_transform_contract,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_galaxy_prediction_contract_transform_registration.json"
BUNDLE_CONTRACT = ROOT / "configs" / "reviewed_g4_galaxy_prediction_bundle_contract.json"
TRANSFORM_CONTRACT = ROOT / "configs" / "g4_galaxy_raw_to_calibrated_transform_contract.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-galaxy-prediction-contract-transform-registration.json"
SOURCE = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / "g4_galaxy_prediction_contract_transform_registration.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context() -> dict:
    return {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "branch_and_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
        "shared_across_all_objects": True,
        "object_specific_gravity_parameters": {},
        "redshift_used_as_distance": False,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _apply(context: dict | None = None, channels: list[str] | None = None) -> dict:
    return apply_registered_linear_calibration(
        [10.0, 20.0],
        [[4.0, 1.0], [1.0, 9.0]],
        [[2.0, 0.0], [0.0, 3.0]],
        [1.0, 2.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.25, 0.05], [0.05, 0.36]],
        channels
        or [
            "stellar_light_calibrated_input_intensity_per_sr",
            "gas_line_calibrated_input_intensity_per_sr",
        ],
        ["calibrated_input_intensity_per_sr"] * 2,
        context or _context(),
    )


def test_existing_prediction_bundle_contract_is_exact_and_candidate_specific() -> None:
    contract = _load(BUNDLE_CONTRACT)
    validate_prediction_bundle_contract(contract)
    assert _sha(contract) == PREDICTION_BUNDLE_CONTRACT_SHA256
    properties = contract["properties"]
    assert properties["candidate_id"]["const"] == CANDIDATE_ID
    assert properties["action_sha256"]["const"] == ACTION_SHA256
    assert properties["object_specific_gravity_parameter_count"]["const"] == 0
    assert properties["data_eligibility"]["const"] == ELIGIBILITY
    assert properties["observational_data_opened"]["const"] is False


def test_transform_contract_is_exact_and_has_no_real_registration() -> None:
    contract = _load(TRANSFORM_CONTRACT)
    validate_transform_contract(contract)
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    assert _sha(body) == contract["content_sha256"] == TRANSFORM_CONTRACT_SHA256
    assert set(contract["current_registration_state"].values()) == {False}
    assert contract["transform"]["object_specific_gravity_parameter_count"] == 0
    assert "dark_matter_or_halo_label" in contract["forbidden_inputs_or_outputs"]
    assert "redshift_or_redshift_derived_distance" in contract["forbidden_inputs_or_outputs"]


def test_linear_transform_and_full_covariance_are_exact_on_synthetic_input() -> None:
    result = _apply()
    assert result["calibrated_values"] == [21.0, 62.0]
    assert np.allclose(result["joint_covariance"], [[16.25, 6.05], [6.05, 81.36]])
    assert result["joint_covariance"][0][1] != 0.0
    assert result["object_specific_gravity_parameter_count"] == 0
    assert result["redshift_distance_inputs"] is False
    assert result["observational_data_opened"] is False


@pytest.mark.parametrize("mutation", ["redshift", "gravity", "sharing"])
def test_redshift_hidden_gravity_and_object_specific_calibration_reject(mutation: str) -> None:
    context = _context()
    if mutation == "redshift":
        context["redshift_used_as_distance"] = True
    elif mutation == "gravity":
        context["object_specific_gravity_parameters"] = {"halo_mass": 1.0}
    else:
        context["shared_across_all_objects"] = False
    with pytest.raises(ValueError, match="sealed contract"):
        _apply(context=context)


def test_target_leakage_bad_covariance_and_negative_calibration_reject() -> None:
    with pytest.raises(ValueError, match="not admitted"):
        _apply(channels=["rotation_curve_target", "gas_line_calibrated_input_intensity_per_sr"])
    with pytest.raises(ValueError, match="positive semidefinite"):
        apply_registered_linear_calibration(
            [1.0],
            [[-1.0]],
            [[1.0]],
            [0.0],
            [[1.0]],
            [[1.0]],
            ["stellar_light_calibrated_input_intensity_per_sr"],
            ["calibrated_input_intensity_per_sr"],
            _context(),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        apply_registered_linear_calibration(
            [1.0],
            [[1.0]],
            [[1.0]],
            [-2.0],
            [[1.0]],
            [[1.0]],
            ["stellar_light_calibrated_input_intensity_per_sr"],
            ["calibrated_input_intensity_per_sr"],
            _context(),
        )


def test_artifact_rebuilds_and_advances_exactly_from_nine_to_eleven() -> None:
    stored = _load(ARTIFACT)
    assert build_g4_galaxy_prediction_contract_transform_registration(_load(CONFIG), ROOT) == stored
    assert (
        stored["content_sha256"]
        == "273a8d7fbbdb394724f5e7e3c07cf0c9dd7aeff802193f5edc3af961a89f8c31"
    )
    assert _file_sha(ARTIFACT) == "42dccf8603aa7e5d64d4d1e4f34fd3f07cbf09d62c01baa53836103452e988d9"
    assert stored["newly_filled_registration_fields"] == {
        "prediction_bundle_contract_sha256": PREDICTION_BUNDLE_CONTRACT_SHA256,
        "raw_to_calibrated_transform_sha256": stored["transform_implementation_readiness"][
            "content_sha256"
        ],
    }
    decision = stored["current_evaluator_decision"]
    assert decision["decision"] == "blocked"
    assert decision["filled_registration_hash_count"] == 11
    assert decision["missing_registration_hashes"] == [
        "dataset_manifest_independent_audit_sha256",
        "galaxy_split_commitment_sha256",
        "prediction_bundle_content_sha256",
        "prediction_bundle_file_sha256",
        "selected_primary_calibration_root_sha256",
        "selected_primary_imaging_and_spectroscopy_root_sha256",
        "training_only_checkpoint_sha256",
    ]
    assert stored["prediction_bundle_registered"] is False
    assert stored["real_transform_inputs_registered"] is False


def test_predecessor_contract_transform_source_and_authorization_tampering_reject() -> None:
    config = _load(CONFIG)
    assert config["source_bindings"]["registration_source"]["file_sha256"] == _file_sha(SOURCE)
    for key in ("predecessor", "prediction_bundle_contract", "transform_contract"):
        mutated = _load(CONFIG)
        mutated["source_bindings"][key]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content changed"):
            build_g4_galaxy_prediction_contract_transform_registration(mutated, ROOT)
    mutated = _load(CONFIG)
    mutated["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_galaxy_prediction_contract_transform_registration(mutated, ROOT)


def test_provenance_and_fail_closed_seals_are_exact() -> None:
    artifact = _load(ARTIFACT)
    provenance = artifact["provenance"]
    body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert _sha(body) == provenance["binding_sha256"]
    assert artifact["primary_record_access_count"] == 0
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["object_specific_gravity_parameter_count"] == 0
    assert artifact["paid_llm_spend_usd"] == 0.0
    assert artifact["data_eligibility"] == ELIGIBILITY
