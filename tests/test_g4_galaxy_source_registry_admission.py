from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_galaxy_manifest_bundle_tooling import (
    RECEIPT_SCHEMA,
    _sha,
    _synthetic_manifest,
    audit_dataset_manifest,
)
from sigma_theory_compiler.g4_galaxy_source_registry_admission import (
    ACTION_SHA256,
    ALLOWED_CHANNELS,
    AUTHORIZATION_SCHEMA,
    CANDIDATE_ID,
    CONTRACT_SHA256,
    admit_source_registry,
    build_g4_galaxy_source_registry_admission_readiness,
    validate_admission_contract,
    validate_source_opening_authorization,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_galaxy_source_registry_admission.json"
CONTRACT = ROOT / "configs" / "g4_galaxy_source_registry_admission_contract.json"
LEDGER = ROOT / "runs" / "engine" / "g4-galaxy-prediction-contract-transform-registration.json"
ARTIFACT = ROOT / "runs" / "engine" / "g4-galaxy-source-registry-admission-readiness.json"
SOURCE = ROOT / "src" / "sigma_theory_compiler" / "g4_galaxy_source_registry_admission.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transform_sha() -> str:
    ledger = _load(LEDGER)
    return ledger["newly_filled_registration_fields"]["raw_to_calibrated_transform_sha256"]


def _receipt(manifest: dict) -> dict:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "manifest_content_sha256": manifest["content_sha256"],
        "source_registry_root_sha256": "a" * 64,
        "independent_reviewer_identity_sha256": "b" * 64,
        "reviewer_is_generator_operator": False,
        "observational_authorization": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _authorization(manifest: dict, audit: dict, receipt: dict) -> dict:
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "manifest_content_sha256": manifest["content_sha256"],
        "independent_manifest_audit_sha256": audit["content_sha256"],
        "independent_registry_receipt_sha256": receipt["content_sha256"],
        "selected_primary_imaging_and_spectroscopy_root_sha256": audit[
            "selected_primary_imaging_and_spectroscopy_root_sha256"
        ],
        "selected_primary_calibration_root_sha256": audit[
            "selected_primary_calibration_root_sha256"
        ],
        "galaxy_split_commitment_sha256": audit["galaxy_split_commitment_sha256"],
        "training_only_checkpoint_sha256": audit["training_only_checkpoint_sha256"],
        "allowed_data_classes": sorted(ALLOWED_CHANNELS),
        "source_input_opening_authorized": True,
        "target_opening_authorized": False,
        "candidate_use_authorized": False,
        "authorizer_identity_sha256": "c" * 64,
        "authorizer_is_generator_operator": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "object_specific_gravity_parameters": {},
    }
    return {**body, "content_sha256": _sha(body)}


def _future_inputs() -> tuple[dict, dict, dict, dict]:
    manifest = _synthetic_manifest(_transform_sha())
    receipt = _receipt(manifest)
    audit = audit_dataset_manifest(manifest, receipt)
    authorization = _authorization(manifest, audit, receipt)
    return manifest, receipt, audit, authorization


def _rehash(value: dict) -> None:
    value["content_sha256"] = _sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_admission_contract_is_exact_and_checked_in_state_is_disabled() -> None:
    contract = _load(CONTRACT)
    validate_admission_contract(contract)
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    assert _sha(body) == contract["content_sha256"] == CONTRACT_SHA256
    state = contract["checked_in_service_state"]
    assert state["enabled"] is False
    assert state["start_requested"] is False
    assert state["source_records_admitted"] == 0
    assert state["target_records_opened"] == 0
    assert state["registration_fields_filled"] == 0
    assert contract["data_eligibility"] == ELIGIBILITY


def test_valid_future_inputs_are_validated_but_disabled_service_admits_nothing() -> None:
    manifest, receipt, audit, authorization = _future_inputs()
    validate_source_opening_authorization(authorization)
    result = admit_source_registry(
        manifest,
        receipt,
        audit,
        authorization,
        service_enabled=False,
    )
    assert result == {
        "decision": "blocked",
        "blocker": "checked_in_source_registry_admission_service_disabled",
        "authorization_validated": True,
        "source_records_admitted": 0,
        "target_records_opened": 0,
        "registration_fields": {},
        "candidate_use_authorized": False,
        "observational_data_opened": False,
        "data_eligibility": ELIGIBILITY,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "source_not_authorized",
        "target_authorized",
        "generator",
        "halo",
        "redshift",
        "object_parameter",
    ],
)
def test_inadequate_leaking_or_forbidden_authorization_rejects(mutation: str) -> None:
    _, _, _, authorization = _future_inputs()
    if mutation == "source_not_authorized":
        authorization["source_input_opening_authorized"] = False
    elif mutation == "target_authorized":
        authorization["target_opening_authorized"] = True
    elif mutation == "generator":
        authorization["authorizer_is_generator_operator"] = True
    elif mutation == "halo":
        authorization["dark_matter_or_halo_inputs"] = True
    elif mutation == "redshift":
        authorization["redshift_distance_inputs"] = True
    else:
        authorization["object_specific_gravity_parameters"] = {"gravity_scale": 2.0}
    _rehash(authorization)
    with pytest.raises(ValueError, match="authorization is invalid"):
        validate_source_opening_authorization(authorization)


def test_root_split_checkpoint_audit_and_receipt_lineage_tampering_rejects() -> None:
    manifest, receipt, audit, authorization = _future_inputs()
    authorization["selected_primary_calibration_root_sha256"] = "0" * 64
    _rehash(authorization)
    with pytest.raises(ValueError, match="lineage mismatch"):
        admit_source_registry(manifest, receipt, audit, authorization, service_enabled=False)
    manifest, receipt, audit, authorization = _future_inputs()
    audit["whole_galaxy_group_count"] += 1
    _rehash(audit)
    authorization = _authorization(manifest, audit, receipt)
    with pytest.raises(ValueError, match="does not exactly reproduce"):
        admit_source_registry(manifest, receipt, audit, authorization, service_enabled=False)
    manifest, receipt, audit, authorization = _future_inputs()
    receipt["reviewer_is_generator_operator"] = True
    _rehash(receipt)
    with pytest.raises(ValueError, match="receipt is invalid"):
        admit_source_registry(manifest, receipt, audit, authorization, service_enabled=False)


def test_artifact_rebuilds_with_zero_records_and_no_ledger_advance() -> None:
    stored = _load(ARTIFACT)
    assert build_g4_galaxy_source_registry_admission_readiness(_load(CONFIG), ROOT) == stored
    assert stored["content_sha256"] == (
        "cb6d78ceda73b917615cd2b4a14d170c2e1cf32468c3e5dfa00067e3142d6c63"
    )
    assert _file_sha(ARTIFACT) == (
        "64dcc559ab84578d4b9fe8181ca8ac299a05d27abb9a037697ee697cc0b9a093"
    )
    assert stored["service_enabled"] is False
    assert stored["start_requested"] is False
    assert stored["source_records_admitted"] == 0
    assert stored["target_records_opened"] == 0
    assert stored["newly_filled_registration_fields"] == {}
    assert stored["filled_registration_hash_count"] == 11
    assert stored["missing_registration_hash_count"] == 7
    assert len(stored["unfilled_registration_fields"]) == 7
    assert stored["observation_opening_authorization_registered"] is False
    assert stored["prediction_bundle_registered"] is False


def test_bindings_config_enablement_authorization_provenance_and_seals() -> None:
    config = _load(CONFIG)
    assert config["source_bindings"]["admission_source"]["file_sha256"] == _file_sha(SOURCE)
    for key in ("ledger_predecessor", "manifest_bundle_tooling", "admission_contract"):
        mutated = _load(CONFIG)
        mutated["source_bindings"][key]["content_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="content changed"):
            build_g4_galaxy_source_registry_admission_readiness(mutated, ROOT)
    mutated = _load(CONFIG)
    mutated["service_enabled"] = True
    with pytest.raises(ValueError, match="must remain disabled"):
        build_g4_galaxy_source_registry_admission_readiness(mutated, ROOT)
    mutated = _load(CONFIG)
    mutated["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_g4_galaxy_source_registry_admission_readiness(mutated, ROOT)
    artifact = _load(ARTIFACT)
    provenance = artifact["provenance"]
    body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert _sha(body) == provenance["binding_sha256"]
    assert artifact["observational_data_opened"] is False
    assert artifact["primary_record_access_count"] == 0
    assert artifact["dark_matter_or_halo_inputs"] is False
    assert artifact["redshift_distance_inputs"] is False
    assert artifact["object_specific_gravity_parameter_count"] == 0
    assert artifact["paid_llm_spend_usd"] == 0.0
