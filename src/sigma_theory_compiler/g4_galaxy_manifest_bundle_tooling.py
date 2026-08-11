"""Fail-closed manifest auditor and prediction-bundle tooling for the G4 galaxy lane."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .g4_galaxy_prediction_contract_transform_registration import (
    validate_prediction_bundle_contract,
)
from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    BUNDLE_SCHEMA,
    CANDIDATE_ID,
    FORMAL_PROVENANCE_SHA256,
    INPUT_CONTRACT,
    OUTPUT_CONTRACT,
    PREDICTION_BUNDLE_CONTRACT_SHA256,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_galaxy_evaluator,
)

SCHEMA_VERSION = "sigma-g4-galaxy-manifest-bundle-tooling-readiness-1.0"
TOOLING_CONTRACT_SCHEMA = "sigma-g4-galaxy-manifest-bundle-tooling-contract-1.0"
TOOLING_CONTRACT_SHA256 = "399880e499015e18d0dfd9d4654e9e9c059d1b95f2934e1aa52b42d4a1955c70"
MANIFEST_SCHEMA = "sigma-g4-galaxy-direct-source-manifest-1.0"
ENTRY_SCHEMA = "sigma-g4-galaxy-direct-source-manifest-entry-1.0"
RECEIPT_SCHEMA = "sigma-independent-galaxy-source-registry-receipt-1.0"
AUDIT_SCHEMA = "sigma-g4-galaxy-independent-dataset-manifest-audit-1.0"
BRANCH_CONTRACT_SHA256 = "a606219458c3eeabcbe940a608dbed758288b946bce8dae26dd59a1995acc405"
ALLOWED_CHANNELS = {
    "stellar_light_imaging",
    "gas_line_emission",
    "angular_geometry",
    "nonredshift_metric_geometry",
}
ALLOWED_ROLES = {
    "formula_search_training",
    "formula_selection_validation",
    "untouched_target_blind_test",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(value: Any) -> bytes:
    return _canonical(value).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound G4 manifest/bundle artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or value.get("content_sha256", expected) != expected:
            raise ValueError(f"bound G4 manifest/bundle content changed: {binding['path']}")
    return value


def validate_tooling_contract(contract: dict[str, Any]) -> None:
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    manifest = contract.get("dataset_manifest_contract", {})
    builder = contract.get("prediction_bundle_builder_contract", {})
    readiness = contract.get("checked_in_readiness", {})
    if (
        contract.get("schema_version") != TOOLING_CONTRACT_SCHEMA
        or contract.get("candidate_id") != CANDIDATE_ID
        or contract.get("action_sha256") != ACTION_SHA256
        or contract.get("content_sha256") != TOOLING_CONTRACT_SHA256
        or _sha(body) != TOOLING_CONTRACT_SHA256
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("entry_schema_version") != ENTRY_SCHEMA
        or set(manifest.get("allowed_source_channels", [])) != ALLOWED_CHANNELS
        or set(manifest.get("allowed_split_roles", [])) != ALLOWED_ROLES
        or manifest.get("target_values_must_be_unopened") is not True
        or manifest.get("independent_audit_requires_external_registry_receipt") is not True
        or builder.get("bundle_schema_version") != BUNDLE_SCHEMA
        or builder.get("bundle_contract_sha256") != PREDICTION_BUNDLE_CONTRACT_SHA256
        or builder.get("synthetic_drafts_registration_admissible") is not False
        or builder.get("builder_may_open_observations") is not False
        or builder.get("builder_may_authorize_candidate_use") is not False
        or any(value is not False for value in readiness.values())
        or contract.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("G4 manifest/bundle tooling contract changed")


def _validate_registry_receipt(receipt: dict[str, Any], manifest_sha256: str) -> None:
    required = {
        "schema_version",
        "manifest_content_sha256",
        "source_registry_root_sha256",
        "independent_reviewer_identity_sha256",
        "reviewer_is_generator_operator",
        "observational_authorization",
        "content_sha256",
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        set(receipt) != required
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("manifest_content_sha256") != manifest_sha256
        or _SHA256.fullmatch(str(receipt.get("source_registry_root_sha256"))) is None
        or _SHA256.fullmatch(str(receipt.get("independent_reviewer_identity_sha256"))) is None
        or receipt.get("reviewer_is_generator_operator") is not False
        or receipt.get("observational_authorization") is not False
        or receipt.get("content_sha256") != _sha(body)
    ):
        raise ValueError("independent source-registry receipt is invalid")


def audit_dataset_manifest(
    manifest: dict[str, Any], registry_receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "manifest_id_sha256",
        "entries",
        "split_salt_commitment_sha256",
        "selected_primary_imaging_and_spectroscopy_root_sha256",
        "selected_primary_calibration_root_sha256",
        "galaxy_split_commitment_sha256",
        "training_only_checkpoint_sha256",
        "raw_to_calibrated_transform_sha256",
        "target_values_opened",
        "dark_matter_or_halo_inputs",
        "redshift_distance_inputs",
        "object_specific_gravity_parameters",
        "content_sha256",
    }
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if (
        set(manifest) != required
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("content_sha256") != _sha(body)
        or manifest.get("target_values_opened") is not False
        or manifest.get("dark_matter_or_halo_inputs") is not False
        or manifest.get("redshift_distance_inputs") is not False
        or manifest.get("object_specific_gravity_parameters") != {}
    ):
        raise ValueError("galaxy dataset manifest violates the sealed schema")
    for name in (
        "manifest_id_sha256",
        "split_salt_commitment_sha256",
        "selected_primary_imaging_and_spectroscopy_root_sha256",
        "selected_primary_calibration_root_sha256",
        "galaxy_split_commitment_sha256",
        "training_only_checkpoint_sha256",
        "raw_to_calibrated_transform_sha256",
    ):
        _require_sha(manifest[name], name)
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("galaxy dataset manifest must contain entries")
    entry_required = {
        "schema_version",
        "galaxy_group_sha256",
        "primary_file_sha256",
        "calibration_file_sha256",
        "nonredshift_geometry_provenance_sha256",
        "source_channel",
        "split_role",
        "target_values_opened",
        "content_sha256",
    }
    groups: dict[str, str] = {}
    primary_hashes: set[str] = set()
    calibration_hashes: set[str] = set()
    entry_hashes: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != entry_required:
            raise ValueError("galaxy dataset manifest entry schema changed")
        entry_body = {key: value for key, value in entry.items() if key != "content_sha256"}
        if entry.get("schema_version") != ENTRY_SCHEMA or entry.get("content_sha256") != _sha(
            entry_body
        ):
            raise ValueError("galaxy dataset manifest entry content changed")
        for name in (
            "galaxy_group_sha256",
            "primary_file_sha256",
            "calibration_file_sha256",
            "nonredshift_geometry_provenance_sha256",
        ):
            _require_sha(entry[name], name)
        if (
            entry.get("source_channel") not in ALLOWED_CHANNELS
            or entry.get("split_role") not in ALLOWED_ROLES
            or entry.get("target_values_opened") is not False
        ):
            raise ValueError("galaxy dataset manifest entry leaks or changes a channel")
        group = entry["galaxy_group_sha256"]
        role = entry["split_role"]
        if group in groups and groups[group] != role:
            raise ValueError("whole-galaxy split leakage detected")
        groups[group] = role
        primary_hashes.add(entry["primary_file_sha256"])
        calibration_hashes.add(entry["calibration_file_sha256"])
        entry_hashes.append(entry["content_sha256"])
    primary_root = _sha(sorted(primary_hashes))
    calibration_root = _sha(sorted(calibration_hashes))
    assignments = [
        {"galaxy_group_sha256": group, "split_role": groups[group]} for group in sorted(groups)
    ]
    split_root = _sha(
        {
            "split_salt_commitment_sha256": manifest["split_salt_commitment_sha256"],
            "sorted_unique_whole_galaxy_assignments": assignments,
        }
    )
    if (
        manifest["selected_primary_imaging_and_spectroscopy_root_sha256"] != primary_root
        or manifest["selected_primary_calibration_root_sha256"] != calibration_root
        or manifest["galaxy_split_commitment_sha256"] != split_root
    ):
        raise ValueError("galaxy dataset manifest root or split commitment mismatch")
    if registry_receipt is not None:
        _validate_registry_receipt(registry_receipt, manifest["content_sha256"])
    audit_body = {
        "schema_version": AUDIT_SCHEMA,
        "manifest_content_sha256": manifest["content_sha256"],
        "entry_content_root_sha256": _sha(sorted(entry_hashes)),
        "selected_primary_imaging_and_spectroscopy_root_sha256": primary_root,
        "selected_primary_calibration_root_sha256": calibration_root,
        "galaxy_split_commitment_sha256": split_root,
        "training_only_checkpoint_sha256": manifest["training_only_checkpoint_sha256"],
        "raw_to_calibrated_transform_sha256": manifest["raw_to_calibrated_transform_sha256"],
        "whole_galaxy_group_count": len(groups),
        "entry_count": len(entries),
        "group_leakage_found": False,
        "target_values_opened": False,
        "registry_receipt_content_sha256": (
            registry_receipt["content_sha256"] if registry_receipt is not None else None
        ),
        "registration_admissible": registry_receipt is not None,
        "observational_authorization": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    return {**audit_body, "content_sha256": _sha(audit_body)}


def build_prediction_bundle_draft(
    registration: dict[str, str | None],
    manifest_audit: dict[str, Any],
    *,
    synthetic_control: bool,
) -> dict[str, Any]:
    if set(registration) != set(REQUIRED_REGISTRATION_HASHES):
        raise ValueError("prediction-bundle builder registration field set changed")
    for name, value in registration.items():
        if name in {"prediction_bundle_content_sha256", "prediction_bundle_file_sha256"}:
            if value is not None:
                raise ValueError(
                    "bundle content/file hashes must be empty before deterministic build"
                )
        else:
            _require_sha(value, name)
    if (
        registration["prediction_bundle_contract_sha256"] != PREDICTION_BUNDLE_CONTRACT_SHA256
        or registration["branch_and_domain_contract_sha256"] != BRANCH_CONTRACT_SHA256
    ):
        raise ValueError("prediction-bundle candidate contract lineage changed")
    audit_body = {key: value for key, value in manifest_audit.items() if key != "content_sha256"}
    if (
        manifest_audit.get("schema_version") != AUDIT_SCHEMA
        or manifest_audit.get("content_sha256") != _sha(audit_body)
        or manifest_audit.get("target_values_opened") is not False
        or manifest_audit.get("observational_authorization") is not False
        or manifest_audit.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("prediction-bundle builder manifest audit changed")
    if synthetic_control:
        if manifest_audit.get("registration_admissible") is not False:
            raise ValueError("synthetic bundle draft cannot consume a registrable audit")
    elif manifest_audit.get("registration_admissible") is not True:
        raise ValueError("registered bundle build requires an independent registry receipt")
    for audit_name, registration_name in (
        (
            "selected_primary_imaging_and_spectroscopy_root_sha256",
            "selected_primary_imaging_and_spectroscopy_root_sha256",
        ),
        ("selected_primary_calibration_root_sha256", "selected_primary_calibration_root_sha256"),
        ("galaxy_split_commitment_sha256", "galaxy_split_commitment_sha256"),
        ("training_only_checkpoint_sha256", "training_only_checkpoint_sha256"),
        ("raw_to_calibrated_transform_sha256", "raw_to_calibrated_transform_sha256"),
    ):
        if manifest_audit[audit_name] != registration[registration_name]:
            raise ValueError(f"prediction-bundle manifest lineage mismatch: {registration_name}")
    if (
        not synthetic_control
        and manifest_audit["content_sha256"]
        != registration["dataset_manifest_independent_audit_sha256"]
    ):
        raise ValueError("prediction-bundle independent audit registration mismatch")
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "formal_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "branch_and_domain_contract_sha256": registration["branch_and_domain_contract_sha256"],
        "input_contract": INPUT_CONTRACT,
        "output_contract": OUTPUT_CONTRACT,
        "universal_parameter_count": 0,
        "object_specific_gravity_parameter_count": 0,
        "rotation_prediction_implementation_sha256": registration[
            "rotation_prediction_implementation_sha256"
        ],
        "lensing_prediction_implementation_sha256": registration[
            "lensing_prediction_implementation_sha256"
        ],
        "baryonic_calibration_hierarchy_sha256": registration[
            "baryonic_calibration_hierarchy_sha256"
        ],
        "joint_covariance_contract_sha256": registration["joint_covariance_contract_sha256"],
        "likelihood_contract_sha256": registration["likelihood_contract_sha256"],
        "galaxy_split_commitment_sha256": registration["galaxy_split_commitment_sha256"],
        "training_only_checkpoint_sha256": registration["training_only_checkpoint_sha256"],
        "stopping_rule_sha256": registration["stopping_rule_sha256"],
        "distance_mode_contract_sha256": registration["distance_mode_contract_sha256"],
        "data_eligibility": dict(ELIGIBILITY),
        "observational_data_opened": False,
    }
    canonical = _canonical_bytes(bundle)
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        "bundle": bundle,
        "prediction_bundle_content_sha256": digest,
        "prediction_bundle_file_sha256": digest,
        "canonical_byte_count": len(canonical),
        "registration_admissible": not synthetic_control,
        "synthetic_control": synthetic_control,
        "observational_data_opened": False,
    }


def validate_prediction_bundle_draft(
    draft: dict[str, Any], registration: dict[str, str | None]
) -> None:
    required = {
        "bundle",
        "prediction_bundle_content_sha256",
        "prediction_bundle_file_sha256",
        "canonical_byte_count",
        "registration_admissible",
        "synthetic_control",
        "observational_data_opened",
    }
    if set(draft) != required or draft.get("observational_data_opened") is not False:
        raise ValueError("prediction-bundle draft fields changed")
    bundle = draft["bundle"]
    bundle_required = {
        "schema_version",
        "candidate_id",
        "action_sha256",
        "formal_provenance_sha256",
        "branch_and_domain_contract_sha256",
        "input_contract",
        "output_contract",
        "universal_parameter_count",
        "object_specific_gravity_parameter_count",
        "rotation_prediction_implementation_sha256",
        "lensing_prediction_implementation_sha256",
        "baryonic_calibration_hierarchy_sha256",
        "joint_covariance_contract_sha256",
        "likelihood_contract_sha256",
        "galaxy_split_commitment_sha256",
        "training_only_checkpoint_sha256",
        "stopping_rule_sha256",
        "distance_mode_contract_sha256",
        "data_eligibility",
        "observational_data_opened",
    }
    if (
        not isinstance(bundle, dict)
        or set(bundle) != bundle_required
        or bundle.get("schema_version") != BUNDLE_SCHEMA
        or bundle.get("candidate_id") != CANDIDATE_ID
        or bundle.get("action_sha256") != ACTION_SHA256
        or bundle.get("formal_provenance_sha256") != FORMAL_PROVENANCE_SHA256
        or bundle.get("input_contract") != INPUT_CONTRACT
        or bundle.get("output_contract") != OUTPUT_CONTRACT
        or not isinstance(bundle.get("universal_parameter_count"), int)
        or bundle["universal_parameter_count"] < 0
        or bundle.get("object_specific_gravity_parameter_count") != 0
        or bundle.get("data_eligibility") != ELIGIBILITY
        or bundle.get("observational_data_opened") is not False
    ):
        raise ValueError("prediction-bundle draft violates the exact bundle contract")
    digest = hashlib.sha256(_canonical_bytes(bundle)).hexdigest()
    if (
        draft.get("prediction_bundle_content_sha256") != digest
        or draft.get("prediction_bundle_file_sha256") != digest
        or draft.get("canonical_byte_count") != len(_canonical_bytes(bundle))
        or draft.get("registration_admissible") is not (not draft.get("synthetic_control"))
    ):
        raise ValueError("prediction-bundle draft hash or readiness changed")
    mapping = {
        "branch_and_domain_contract_sha256",
        "rotation_prediction_implementation_sha256",
        "lensing_prediction_implementation_sha256",
        "baryonic_calibration_hierarchy_sha256",
        "joint_covariance_contract_sha256",
        "likelihood_contract_sha256",
        "galaxy_split_commitment_sha256",
        "training_only_checkpoint_sha256",
        "stopping_rule_sha256",
        "distance_mode_contract_sha256",
    }
    if any(bundle[name] != registration[name] for name in mapping):
        raise ValueError("prediction-bundle draft registration lineage changed")


def _synthetic_manifest(transform_sha256: str) -> dict[str, Any]:
    entries = []
    for index, (group, role, channel) in enumerate(
        [
            ("1" * 64, "formula_search_training", "stellar_light_imaging"),
            ("1" * 64, "formula_search_training", "gas_line_emission"),
            ("2" * 64, "untouched_target_blind_test", "angular_geometry"),
        ]
    ):
        body = {
            "schema_version": ENTRY_SCHEMA,
            "galaxy_group_sha256": group,
            "primary_file_sha256": f"{index + 3:x}" * 64,
            "calibration_file_sha256": f"{index + 6:x}" * 64,
            "nonredshift_geometry_provenance_sha256": f"{index + 9:x}" * 64,
            "source_channel": channel,
            "split_role": role,
            "target_values_opened": False,
        }
        entries.append({**body, "content_sha256": _sha(body)})
    primary_root = _sha(sorted({entry["primary_file_sha256"] for entry in entries}))
    calibration_root = _sha(sorted({entry["calibration_file_sha256"] for entry in entries}))
    assignments = [
        {"galaxy_group_sha256": "1" * 64, "split_role": "formula_search_training"},
        {"galaxy_group_sha256": "2" * 64, "split_role": "untouched_target_blind_test"},
    ]
    salt = "c" * 64
    split = _sha(
        {
            "split_salt_commitment_sha256": salt,
            "sorted_unique_whole_galaxy_assignments": assignments,
        }
    )
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_id_sha256": "d" * 64,
        "entries": entries,
        "split_salt_commitment_sha256": salt,
        "selected_primary_imaging_and_spectroscopy_root_sha256": primary_root,
        "selected_primary_calibration_root_sha256": calibration_root,
        "galaxy_split_commitment_sha256": split,
        "training_only_checkpoint_sha256": "e" * 64,
        "raw_to_calibrated_transform_sha256": transform_sha256,
        "target_values_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "object_specific_gravity_parameters": {},
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_galaxy_manifest_bundle_tooling_readiness(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 manifest/bundle tooling eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 manifest/bundle tooling opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["tooling_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("G4 manifest/bundle tooling source changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key != "tooling_source"
    }
    validate_tooling_contract(sources["tooling_contract"])
    validate_prediction_bundle_contract(sources["prediction_bundle_contract"])
    predecessor = sources["predecessor"]
    decision = predecessor.get("current_evaluator_decision", {})
    if (
        predecessor.get("decision") != "blocked"
        or decision.get("filled_registration_hash_count") != 11
        or len(decision.get("missing_registration_hashes", [])) != 7
        or predecessor.get("prediction_bundle_registered") is not False
        or predecessor.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 11/7 predecessor ledger changed")
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
    if unchanged != decision:
        raise ValueError("G4 tooling readiness falsely changed the evaluator ledger")
    transform_sha = registration["raw_to_calibrated_transform_sha256"]
    assert isinstance(transform_sha, str)
    manifest = _synthetic_manifest(transform_sha)
    audit = audit_dataset_manifest(manifest)
    synthetic_registration = dict(registration)
    synthetic_registration.update(
        {
            "dataset_manifest_independent_audit_sha256": audit["content_sha256"],
            "galaxy_split_commitment_sha256": audit["galaxy_split_commitment_sha256"],
            "selected_primary_calibration_root_sha256": audit[
                "selected_primary_calibration_root_sha256"
            ],
            "selected_primary_imaging_and_spectroscopy_root_sha256": audit[
                "selected_primary_imaging_and_spectroscopy_root_sha256"
            ],
            "training_only_checkpoint_sha256": audit["training_only_checkpoint_sha256"],
        }
    )
    draft = build_prediction_bundle_draft(synthetic_registration, audit, synthetic_control=True)
    validate_prediction_bundle_draft(draft, synthetic_registration)
    controls_body = {
        "manifest_audit_sha256": audit["content_sha256"],
        "manifest_audit_registration_admissible": False,
        "bundle_draft_content_sha256": draft["prediction_bundle_content_sha256"],
        "bundle_draft_file_sha256": draft["prediction_bundle_file_sha256"],
        "bundle_draft_registration_admissible": False,
        "synthetic_values_promoted": False,
        "observational_data_opened": False,
    }
    controls = {**controls_body, "content_sha256": _sha(controls_body)}
    readiness_body = {
        "tooling_source_sha256": source_binding["file_sha256"],
        "tooling_contract_sha256": TOOLING_CONTRACT_SHA256,
        "prediction_bundle_contract_sha256": PREDICTION_BUNDLE_CONTRACT_SHA256,
        "synthetic_controls_sha256": controls["content_sha256"],
        "enabled": False,
        "registration_fields_filled": 0,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    readiness = {**readiness_body, "content_sha256": _sha(readiness_body)}
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "predecessor_content_sha256": bindings["predecessor"]["content_sha256"],
        "tooling_contract_sha256": TOOLING_CONTRACT_SHA256,
        "tooling_readiness_sha256": readiness["content_sha256"],
        "tooling_source_sha256": source_binding["file_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "source_bindings": bindings,
        "tooling_readiness": readiness,
        "synthetic_controls": controls,
        "newly_filled_registration_fields": {},
        "unchanged_evaluator_decision": unchanged,
        "unfilled_registration_fields": decision["missing_registration_hashes"],
        "filled_registration_hash_count": 11,
        "missing_registration_hash_count": 7,
        "dataset_manifest_registered": False,
        "independent_registry_receipt_registered": False,
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
        "first_missing_premise": "external_registered_source_manifest_and_independent_registry_receipt",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "Manifest-audit and deterministic prediction-bundle tools are executable, but checked-in "
            "readiness is disabled. Synthetic controls are non-registrable and the exact 11/7 ledger "
            "is unchanged; no source root, split, checkpoint, bundle, target, or observation was opened."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
