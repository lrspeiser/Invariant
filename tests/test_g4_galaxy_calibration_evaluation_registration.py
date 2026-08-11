from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_galaxy_calibration_evaluation_registration import (
    CONTRACT_HASHES,
    SUITE_CONTENT_SHA256,
    assign_synthetic_split,
    build_g4_galaxy_calibration_evaluation_registration,
    joint_gaussian_nll,
    stopping_decision,
    validate_policy_suite,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_galaxy_calibration_evaluation_registration.json"
SUITE = ROOT / "configs" / "g4_galaxy_calibration_evaluation_policy_suite.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-galaxy-calibration-evaluation-registration.json"
SOURCE = ROOT / "src" / "sigma_theory_compiler" / "g4_galaxy_calibration_evaluation_registration.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_policy_suite_and_each_contract_are_exact_and_sealed() -> None:
    suite = _load(SUITE)
    validate_policy_suite(suite)
    assert suite["content_sha256"] == SUITE_CONTENT_SHA256
    assert (
        _sha({key: value for key, value in suite.items() if key != "content_sha256"})
        == SUITE_CONTENT_SHA256
    )
    for name, expected in CONTRACT_HASHES.items():
        contract = suite[name]
        assert contract["content_sha256"] == expected
        assert (
            _sha({key: value for key, value in contract.items() if key != "content_sha256"})
            == expected
        )
        assert contract["data_eligibility"] == ELIGIBILITY
    calibration = suite["calibration_hierarchy"]
    assert calibration["object_specific_gravity_parameter_count"] == 0
    assert calibration["posthoc_rescue_allowed"] is False
    assert calibration["actual_calibration_values_registered"] is False
    assert suite["joint_covariance_contract"]["actual_covariance_registered"] is False
    assert suite["likelihood_contract"]["actual_residuals_or_targets_registered"] is False


def test_split_policy_is_executable_but_not_a_real_split_commitment() -> None:
    suite = _load(SUITE)
    split = suite["held_out_split_policy"]
    assert split["split_unit"] == "whole_galaxy"
    assert split["galaxy_split_commitment_registration_admissible"] is False
    entry, salt = "1" * 64, "a" * 64
    assert assign_synthetic_split(entry, salt) == assign_synthetic_split(entry, salt)
    assert assign_synthetic_split(entry, salt) in split["roles"]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        assign_synthetic_split("galaxy-name", salt)


def test_joint_likelihood_and_stopping_controls_are_deterministic() -> None:
    nll = joint_gaussian_nll([1.0, -1.0], [[2.0, 0.5], [0.5, 1.0]])
    assert math.isclose(nll, 3.2605421032341995, rel_tol=0.0, abs_tol=1e-15)
    with pytest.raises(ValueError, match="positive definite"):
        joint_gaussian_nll([1.0, 2.0], [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="finite, square, and symmetric"):
        joint_gaussian_nll([1.0, 2.0], [[1.0, 0.5], [0.0, 1.0]])
    assert stopping_decision([1.0, 0.9], 20)["stop"] is False
    assert stopping_decision([1.0] + [1.0] * 20, 210)["reasons"] == ["validation_patience_checks"]
    assert "maximum_training_optimizer_steps" in stopping_decision([1.0], 500)["reasons"]


def test_artifact_rebuilds_and_advances_exactly_from_five_to_nine_filled() -> None:
    stored = _load(ARTIFACT)
    assert build_g4_galaxy_calibration_evaluation_registration(_load(CONFIG), ROOT) == stored
    assert (
        stored["content_sha256"]
        == "e8229a0a2aca013d7139fc80c3ae2430207539b3c04278d6c92398785d957dea"
    )
    assert _file_sha(ARTIFACT) == "798a8f9a247ca8a5d4fb8c07b8482047a27170b2047921866788f22dc7a3d375"
    assert set(stored["newly_filled_registration_fields"]) == {
        "baryonic_calibration_hierarchy_sha256",
        "joint_covariance_contract_sha256",
        "likelihood_contract_sha256",
        "stopping_rule_sha256",
    }
    decision = stored["current_evaluator_decision"]
    assert decision["decision"] == "blocked"
    assert decision["filled_registration_hash_count"] == 9
    assert len(decision["missing_registration_hashes"]) == 9
    assert "galaxy_split_commitment_sha256" in decision["missing_registration_hashes"]
    assert "training_only_checkpoint_sha256" in decision["missing_registration_hashes"]
    assert stored["non_registration_policy_hashes"] == {
        "held_out_split_policy_sha256": CONTRACT_HASHES["held_out_split_policy"]
    }
    assert stored["prediction_bundle_registered"] is False
    assert stored["observational_data_opened"] is False
    assert stored["object_specific_gravity_parameter_count"] == 0


def test_predecessor_suite_protocol_policy_source_and_authorization_tampering_reject() -> None:
    config = _load(CONFIG)
    assert config["source_bindings"]["registration_source"]["file_sha256"] == _file_sha(SOURCE)
    for key in ("predecessor", "policy_suite", "galaxy_protocol", "evidence_policy"):
        mutated = _load(CONFIG)
        mutated["source_bindings"][key]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content changed"):
            build_g4_galaxy_calibration_evaluation_registration(mutated, ROOT)
    mutated = _load(CONFIG)
    mutated["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_galaxy_calibration_evaluation_registration(mutated, ROOT)


def test_policy_tampering_rejects_hidden_per_galaxy_gravity_rescue() -> None:
    suite = _load(SUITE)
    suite["calibration_hierarchy"]["object_specific_gravity_parameter_count"] = 1
    with pytest.raises(ValueError, match="suite changed"):
        validate_policy_suite(suite)


def test_provenance_and_no_data_seals_are_exact() -> None:
    artifact = _load(ARTIFACT)
    provenance = artifact["provenance"]
    body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert _sha(body) == provenance["binding_sha256"]
    assert artifact["primary_record_access_count"] == 0
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["paid_llm_spend_usd"] == 0.0
    assert artifact["data_eligibility"] == ELIGIBILITY
