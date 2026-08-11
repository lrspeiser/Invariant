from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_galaxy_branch_distance_registration import (
    BRANCH_CONTRACT_SHA256,
    DISTANCE_CONTRACT_SHA256,
    GEOMETRY_REGISTRATION_SCHEMA,
    _synthetic_geometry_registration,
    build_g4_galaxy_branch_distance_registration,
    geometry_registration_to_forward_profile_fields,
    validate_branch_domain_contract,
    validate_distance_geometry_contract,
    validate_source_geometry_registration,
)
from sigma_theory_compiler.g4_scalar_free_galaxy_forward_model import (
    GEOMETRY_CONTRACT,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.reviewed_g4_candidate_galaxy_evaluator import (
    DESCRIPTOR_FIELD,
    REQUIRED_REGISTRATION_HASHES,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_galaxy_branch_distance_registration.json"
BRANCH = ROOT / "configs" / "g4_scalar_free_galaxy_branch_domain_contract.json"
DISTANCE = ROOT / "configs" / "g4_galaxy_nonredshift_distance_geometry_contract.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-galaxy-branch-distance-registration.json"
SOURCE = (
    ROOT
    / "src"
    / "sigma_theory_compiler"
    / "g4_galaxy_branch_distance_registration.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_branch_contract_is_exact_conditional_and_not_a_source_uniqueness_claim() -> None:
    contract = _load(BRANCH)
    validate_branch_domain_contract(contract)
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    assert _canonical_sha(body) == contract["content_sha256"] == BRANCH_CONTRACT_SHA256
    assert contract["field_domain"]["phi"] == "0_everywhere"
    assert set(contract["exact_residual_contract"].values()) == {"0"}
    conditionality = contract["conditionality"]
    assert conditionality["contract_status"] == "certified_exact_conditional_branch"
    assert conditionality["source_specific_branch_selection_proven"] is False
    assert conditionality["prediction_bundle_claimed"] is False
    assert conditionality["observational_evidence_claimed"] is False
    assert "not uniqueness" in conditionality["warning"]
    assert contract["forward_model_scope"]["object_specific_gravity_parameter_count"] == 0


def test_distance_contract_is_nonredshift_and_has_no_real_geometry_values() -> None:
    contract = _load(DISTANCE)
    validate_distance_geometry_contract(contract)
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    assert _canonical_sha(body) == contract["content_sha256"] == DISTANCE_CONTRACT_SHA256
    modes = contract["distance_modes"]
    assert modes["forward_model_mode"] == (
        "separately_registered_nonredshift_metric_distance"
    )
    assert modes["redshift_distance_allowed"] is False
    assert modes["cosmological_model_distance_allowed"] is False
    assert contract["forward_model_mapping"]["geometry_contract"] == GEOMETRY_CONTRACT
    assert set(contract["current_registration_state"].values()) == {False}
    assert contract["data_eligibility"] == ELIGIBILITY


def test_synthetic_geometry_registration_maps_exactly_to_forward_model_fields() -> None:
    registration = _synthetic_geometry_registration()
    assert registration["schema_version"] == GEOMETRY_REGISTRATION_SCHEMA
    validate_source_geometry_registration(registration)
    mapped = geometry_registration_to_forward_profile_fields(registration)
    assert mapped == {
        "nonredshift_lens_distance_m": 1.0e20,
        "lensing_distance_ratio": 0.5,
        "distance_provenance": "separately_registered_nonredshift_distance_required",
        "geometry_contract": GEOMETRY_CONTRACT,
    }
    assert mapped["nonredshift_lens_distance_m"] * 0.01 == 1.0e18
    assert registration["redshift_used_as_distance"] is False
    assert registration["observational_data_opened_by_contract"] is False
    assert registration["object_specific_gravity_parameters"] == {}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("redshift", "violates the contract"),
        ("ratio", "violates the contract"),
        ("distance", "violates the contract"),
        ("hash", "violates the contract"),
        ("geometry", "violates the contract"),
        ("object_gravity", "violates the contract"),
        ("opened", "violates the contract"),
        ("missing", "fields changed"),
    ],
)
def test_redshift_invalid_geometry_and_hidden_parameters_are_rejected(
    mutation: str, message: str
) -> None:
    registration = _synthetic_geometry_registration()
    if mutation == "redshift":
        registration["redshift_used_as_distance"] = True
    elif mutation == "ratio":
        registration["lensing_distance_ratio_D_ls_over_D_s"] = 1.1
    elif mutation == "distance":
        registration["nonredshift_lens_distance_m"] = -1.0
    elif mutation == "hash":
        registration["independent_nonredshift_distance_audit_sha256"] = "bad"
    elif mutation == "geometry":
        registration["geometry_contract"] = {
            **GEOMETRY_CONTRACT,
            "lensing": "unregistered_general_lens_solver",
        }
    elif mutation == "object_gravity":
        registration["object_specific_gravity_parameters"] = {"G": 2.0}
    elif mutation == "opened":
        registration["observational_data_opened_by_contract"] = True
    else:
        del registration["distance_and_geometry_covariance_sha256"]
    with pytest.raises(ValueError, match=message):
        validate_source_geometry_registration(registration)


def test_artifact_rebuilds_and_advances_exactly_from_three_to_five_filled() -> None:
    stored = _load(ARTIFACT)
    rebuilt = build_g4_galaxy_branch_distance_registration(_load(CONFIG), ROOT)
    assert rebuilt == stored
    assert stored["content_sha256"] == (
        "1a488f5b08421ded95c097dc2477bc2f34754af230dbc5f402f6e0ee07e4b2e7"
    )
    assert _file_sha(ARTIFACT) == (
        "8e073f85989e9ebf4b77bf137f5b687d30cad0ec8cb910ac8691e00a7746406c"
    )
    assert stored["newly_filled_registration_fields"] == {
        "branch_and_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
        "distance_mode_contract_sha256": DISTANCE_CONTRACT_SHA256,
    }
    assert set(stored["preserved_predecessor_registration_fields"]) == {
        DESCRIPTOR_FIELD,
        "rotation_prediction_implementation_sha256",
        "lensing_prediction_implementation_sha256",
    }
    expected_missing = sorted(
        set(REQUIRED_REGISTRATION_HASHES)
        - set(stored["newly_filled_registration_fields"])
        - set(stored["preserved_predecessor_registration_fields"])
    )
    assert stored["unfilled_registration_fields"] == expected_missing
    assert len(expected_missing) == 13
    assert stored["current_evaluator_decision"]["filled_registration_hash_count"] == 5
    assert stored["current_evaluator_decision"]["decision"] == "blocked"
    assert stored["prediction_bundle_registered"] is False
    assert stored["real_source_geometry_registered"] is False
    assert stored["source_specific_branch_selection_proven"] is False
    assert stored["observational_data_opened"] is False


def test_contract_predecessor_policy_source_and_authorization_tampering_reject() -> None:
    assert _load(CONFIG)["source_bindings"]["registration_source"][
        "file_sha256"
    ] == _file_sha(SOURCE)
    for key in (
        "scalar_free_branch",
        "forward_model",
        "branch_domain_contract",
        "distance_geometry_contract",
        "galaxy_protocol",
        "evidence_policy",
    ):
        config = _load(CONFIG)
        config["source_bindings"][key]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content changed"):
            build_g4_galaxy_branch_distance_registration(config, ROOT)
    config = _load(CONFIG)
    config["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_galaxy_branch_distance_registration(config, ROOT)


def test_provenance_and_fail_closed_seals_are_exact() -> None:
    artifact = _load(ARTIFACT)
    provenance = artifact["provenance"]
    body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert _canonical_sha(body) == provenance["binding_sha256"]
    assert provenance["branch_domain_contract_sha256"] == BRANCH_CONTRACT_SHA256
    assert provenance["distance_geometry_contract_sha256"] == DISTANCE_CONTRACT_SHA256
    assert artifact["synthetic_controls"]["real_source_geometry_registered"] is False
    assert artifact["primary_record_access_count"] == 0
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["object_specific_gravity_parameter_count"] == 0
    assert artifact["paid_llm_spend_usd"] == 0.0
    assert artifact["data_eligibility"] == ELIGIBILITY
