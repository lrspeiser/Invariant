"""Checked-disabled source-registry admission gate for the G4 galaxy lane."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .g4_galaxy_manifest_bundle_tooling import (
    ALLOWED_CHANNELS,
    AUDIT_SCHEMA,
    audit_dataset_manifest,
)
from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    CANDIDATE_ID,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_galaxy_evaluator,
)

SCHEMA_VERSION = "sigma-g4-galaxy-source-registry-admission-readiness-1.0"
CONTRACT_SCHEMA = "sigma-g4-galaxy-source-registry-admission-contract-1.0"
CONTRACT_SHA256 = "6544e426f49a35dcc6703eb578ef0cb6480cb2af2630a774c8a66c67eac67d1e"
AUTHORIZATION_SCHEMA = "sigma-g4-galaxy-source-opening-authorization-1.0"
ADMISSION_RECEIPT_SCHEMA = "sigma-g4-galaxy-source-registry-admission-receipt-1.0"
MANIFEST_TOOLING_READINESS_SHA256 = (
    "28fe7f370c10cb749bc4cfb0cc5e1a1c29e75c6c3f37534f7f3b6eb8af68fa33"
)
CALLBACK = "sigma_theory_compiler.g4_galaxy_source_registry_admission:admit_source_registry"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound G4 source-admission artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or value.get("content_sha256", expected) != expected:
            raise ValueError(f"bound G4 source-admission content changed: {binding['path']}")
    return value


def validate_admission_contract(contract: dict[str, Any]) -> None:
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    authorization = contract.get("opening_authorization_contract", {})
    rules = contract.get("admission_rules", {})
    state = contract.get("checked_in_service_state", {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or contract.get("candidate_id") != CANDIDATE_ID
        or contract.get("action_sha256") != ACTION_SHA256
        or contract.get("manifest_audit_tooling_sha256") != MANIFEST_TOOLING_READINESS_SHA256
        or contract.get("content_sha256") != CONTRACT_SHA256
        or _sha(body) != CONTRACT_SHA256
        or authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or set(authorization.get("allowed_data_classes", [])) != ALLOWED_CHANNELS
        or authorization.get("source_input_opening_authorized") is not True
        or authorization.get("target_opening_authorized") is not False
        or authorization.get("candidate_use_authorized") is not False
        or authorization.get("authorizer_must_be_independent_of_generator") is not True
        or rules.get("whole_galaxy_grouping_must_have_one_split_role") is not True
        or rules.get("target_values_must_be_unopened") is not True
        or rules.get("object_specific_gravity_parameter_count") != 0
        or state
        != {
            "enabled": False,
            "start_requested": False,
            "source_records_admitted": 0,
            "target_records_opened": 0,
            "registration_fields_filled": 0,
            "observation_opening_authorization_registered": False,
            "prediction_bundle_claimed": False,
        }
        or contract.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("G4 source-registry admission contract changed")


def validate_source_opening_authorization(authorization: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "action_sha256",
        "manifest_content_sha256",
        "independent_manifest_audit_sha256",
        "independent_registry_receipt_sha256",
        "selected_primary_imaging_and_spectroscopy_root_sha256",
        "selected_primary_calibration_root_sha256",
        "galaxy_split_commitment_sha256",
        "training_only_checkpoint_sha256",
        "allowed_data_classes",
        "source_input_opening_authorized",
        "target_opening_authorized",
        "candidate_use_authorized",
        "authorizer_identity_sha256",
        "authorizer_is_generator_operator",
        "dark_matter_or_halo_inputs",
        "redshift_distance_inputs",
        "object_specific_gravity_parameters",
        "content_sha256",
    }
    body = {key: value for key, value in authorization.items() if key != "content_sha256"}
    if (
        set(authorization) != required
        or authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("candidate_id") != CANDIDATE_ID
        or authorization.get("action_sha256") != ACTION_SHA256
        or authorization.get("content_sha256") != _sha(body)
        or set(authorization.get("allowed_data_classes", [])) != ALLOWED_CHANNELS
        or authorization.get("source_input_opening_authorized") is not True
        or authorization.get("target_opening_authorized") is not False
        or authorization.get("candidate_use_authorized") is not False
        or authorization.get("authorizer_is_generator_operator") is not False
        or authorization.get("dark_matter_or_halo_inputs") is not False
        or authorization.get("redshift_distance_inputs") is not False
        or authorization.get("object_specific_gravity_parameters") != {}
    ):
        raise ValueError("G4 source-opening authorization is invalid")
    for name in required:
        if name.endswith("_sha256"):
            value = authorization[name]
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"G4 source-opening authorization hash is invalid: {name}")


def admit_source_registry(
    manifest: dict[str, Any],
    registry_receipt: dict[str, Any],
    independent_audit: dict[str, Any],
    opening_authorization: dict[str, Any],
    *,
    service_enabled: bool,
) -> dict[str, Any]:
    recomputed = audit_dataset_manifest(manifest, registry_receipt)
    if recomputed != independent_audit:
        raise ValueError("independent manifest audit does not exactly reproduce")
    if (
        independent_audit.get("schema_version") != AUDIT_SCHEMA
        or independent_audit.get("registration_admissible") is not True
        or independent_audit.get("group_leakage_found") is not False
        or independent_audit.get("target_values_opened") is not False
        or independent_audit.get("observational_authorization") is not False
        or independent_audit.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("independent manifest audit is not admissible")
    validate_source_opening_authorization(opening_authorization)
    lineage = {
        "manifest_content_sha256": manifest["content_sha256"],
        "independent_manifest_audit_sha256": independent_audit["content_sha256"],
        "independent_registry_receipt_sha256": registry_receipt["content_sha256"],
        "selected_primary_imaging_and_spectroscopy_root_sha256": independent_audit[
            "selected_primary_imaging_and_spectroscopy_root_sha256"
        ],
        "selected_primary_calibration_root_sha256": independent_audit[
            "selected_primary_calibration_root_sha256"
        ],
        "galaxy_split_commitment_sha256": independent_audit["galaxy_split_commitment_sha256"],
        "training_only_checkpoint_sha256": independent_audit["training_only_checkpoint_sha256"],
    }
    if any(opening_authorization[name] != value for name, value in lineage.items()):
        raise ValueError("source-opening authorization lineage mismatch")
    if not service_enabled:
        return {
            "decision": "blocked",
            "blocker": "checked_in_source_registry_admission_service_disabled",
            "authorization_validated": True,
            "source_records_admitted": 0,
            "target_records_opened": 0,
            "registration_fields": {},
            "candidate_use_authorized": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
        }
    registration_fields = {
        "dataset_manifest_independent_audit_sha256": independent_audit["content_sha256"],
        "galaxy_split_commitment_sha256": independent_audit["galaxy_split_commitment_sha256"],
        "selected_primary_calibration_root_sha256": independent_audit[
            "selected_primary_calibration_root_sha256"
        ],
        "selected_primary_imaging_and_spectroscopy_root_sha256": independent_audit[
            "selected_primary_imaging_and_spectroscopy_root_sha256"
        ],
        "training_only_checkpoint_sha256": independent_audit["training_only_checkpoint_sha256"],
    }
    receipt_body = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "manifest_content_sha256": manifest["content_sha256"],
        "independent_manifest_audit_sha256": independent_audit["content_sha256"],
        "opening_authorization_sha256": opening_authorization["content_sha256"],
        "source_record_count": independent_audit["entry_count"],
        "whole_galaxy_group_count": independent_audit["whole_galaxy_group_count"],
        "registration_fields": registration_fields,
        "target_records_opened": 0,
        "prediction_bundle_claimed": False,
        "candidate_use_authorized": False,
        "observational_data_opened_by_admission": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    receipt = {**receipt_body, "content_sha256": _sha(receipt_body)}
    return {
        "decision": "admitted_registered_source_metadata",
        "authorization_validated": True,
        "source_records_admitted": independent_audit["entry_count"],
        "target_records_opened": 0,
        "registration_fields": registration_fields,
        "admission_receipt": receipt,
        "candidate_use_authorized": False,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
    }


def build_g4_galaxy_source_registry_admission_readiness(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 source-admission readiness eligibility changed")
    if config.get("service_enabled") is not False:
        raise ValueError("checked-in G4 source-admission service must remain disabled")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 source-admission readiness opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["admission_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("G4 source-admission implementation changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key != "admission_source"
    }
    validate_admission_contract(sources["admission_contract"])
    tooling = sources["manifest_bundle_tooling"]
    if (
        tooling.get("tooling_readiness", {}).get("content_sha256")
        != MANIFEST_TOOLING_READINESS_SHA256
        or tooling.get("tooling_readiness", {}).get("enabled") is not False
        or tooling.get("dataset_manifest_registered") is not False
        or tooling.get("independent_registry_receipt_registered") is not False
        or tooling.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 manifest/bundle tooling readiness changed")
    predecessor = sources["ledger_predecessor"]
    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration.update(predecessor["preserved_predecessor_registration_fields"])
    registration.update(predecessor["newly_filled_registration_fields"])
    unchanged = reviewed_g4_candidate_galaxy_evaluator(
        {
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "role": "generated_candidate",
            "data_eligibility": dict(ELIGIBILITY),
        },
        {
            "data_eligibility": dict(ELIGIBILITY),
            "observational_opening_authorized": False,
            "registration_hashes": registration,
        },
    )
    if (
        unchanged.get("decision") != "blocked"
        or unchanged.get("filled_registration_hash_count") != 11
        or len(unchanged.get("missing_registration_hashes", [])) != 7
    ):
        raise ValueError("G4 source-admission readiness changed the 11/7 ledger")
    readiness_body = {
        "callback": CALLBACK,
        "admission_source_sha256": source_binding["file_sha256"],
        "admission_contract_sha256": CONTRACT_SHA256,
        "manifest_audit_tooling_sha256": MANIFEST_TOOLING_READINESS_SHA256,
        "service_enabled": False,
        "start_requested": False,
        "source_records_admitted": 0,
        "target_records_opened": 0,
        "registration_fields_filled": 0,
        "observation_opening_authorization_registered": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    readiness = {**readiness_body, "content_sha256": _sha(readiness_body)}
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "ledger_predecessor_sha256": bindings["ledger_predecessor"]["content_sha256"],
        "manifest_bundle_tooling_sha256": bindings["manifest_bundle_tooling"]["content_sha256"],
        "admission_contract_sha256": CONTRACT_SHA256,
        "admission_readiness_sha256": readiness["content_sha256"],
        "admission_source_sha256": source_binding["file_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "source_bindings": bindings,
        "admission_readiness": readiness,
        "newly_filled_registration_fields": {},
        "unchanged_evaluator_decision": unchanged,
        "filled_registration_hash_count": 11,
        "missing_registration_hash_count": 7,
        "unfilled_registration_fields": unchanged["missing_registration_hashes"],
        "service_enabled": False,
        "start_requested": False,
        "source_records_admitted": 0,
        "target_records_opened": 0,
        "observation_opening_authorization_registered": False,
        "prediction_bundle_registered": False,
        "candidate_use_authorized": False,
        "observational_authorization": False,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "object_specific_gravity_parameter_count": 0,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "decision": "blocked",
        "first_missing_premise": "explicit_registered_source_opening_authorization",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The deterministic admission callback is ready but checked-in execution is disabled. "
            "It admitted zero records, filled zero hashes, opened no targets or source records, and "
            "left the exact 11/7 galaxy ledger unchanged."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
