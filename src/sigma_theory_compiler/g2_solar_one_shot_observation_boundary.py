"""Fail-closed one-shot Solar observation boundary for two action-bound G2 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_SCHEMA = "sigma-g2-solar-one-shot-observation-boundary-config-1.0"
RESULT_SCHEMA = "sigma-g2-solar-one-shot-observation-boundary-result-1.0"
CAMPAIGN_ID = "g2-solar-one-shot-observation-boundary-001"
CONFIG_PATH = "configs/g2_solar_one_shot_observation_boundary.json"
SOURCE_PATH = "src/sigma_theory_compiler/g2_solar_one_shot_observation_boundary.py"
TEST_PATH = "tests/test_g2_solar_one_shot_observation_boundary.py"
OUTPUT_PATH = "runs/math/g2-solar-one-shot-observation-boundary/receipt.json"
SCOPE = (
    "Authorization and deterministic execution readiness for one atomic held-out Cassini SCE1 "
    "opening over two exact action-bound G2 candidates. A blocked receipt opens no observation "
    "and is not a data result, truth claim, model rejection, or promotion decision."
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_KEYS = {
    "campaign_id",
    "candidate_ids",
    "execution_contract",
    "policies",
    "required_opening_fields",
    "schema_version",
    "snapshot_sha256",
    "source_bindings",
}
_BINDING_KEYS = {"content_sha256", "file_sha256", "path"}
_REQUIRED_FIELDS = (
    "source_branch_domain_instantiation_sha256",
    "held_out_split_commitment_sha256",
    "selected_primary_record_roots_sha256",
    "observation_opening_authorization_sha256",
)
_CANDIDATES = (
    "G3A-2f8983c88f504150381064f2",
    "G3A-58e59412e5fe77cd54caf863",
)
_BUNDLE_ROLES = {
    "G3A-2f8983c88f504150381064f2": "bundle_G3A_2f8983",
    "G3A-58e59412e5fe77cd54caf863": "bundle_G3A_58e594",
}
_EXECUTION_CONTRACT = {
    "atomic_target_open_batches": 1,
    "candidate_evaluations": 2,
    "refits_after_open": 0,
    "promotion_actions": 0,
    "target_channels": [
        "two_way_round_trip_light_time",
        "coherent_carrier_frequency_or_phase_ratio",
        "relative_angular_separation",
    ],
    "allowed_quantity_classes": ["raw", "calibrated", "derived"],
    "forbidden_quantity_classes": ["latent", "model_dependent"],
    "terminal_outcomes": ["pass", "reject", "error"],
    "stopping_rule": "stop_after_one_atomic_open_and_one_evaluation_per_frozen_candidate",
}
_POLICIES = {
    "open_only_if_all_required_fields_present": True,
    "independent_authorization_required": True,
    "network_access": "forbidden",
    "live_sqlite_access": "forbidden",
    "runtime_process_control": "forbidden",
    "secret_access": "forbidden",
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "heldout_refit": False,
    "promotion_authorized": False,
}


class SolarOneShotBoundaryError(ValueError):
    """A provenance, authorization, or one-shot contract boundary changed."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SolarOneShotBoundaryError(f"{label} keys changed")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SolarOneShotBoundaryError("path must be portable and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise SolarOneShotBoundaryError("path escapes project root") from error
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise SolarOneShotBoundaryError(f"JSON source missing or oversized: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SolarOneShotBoundaryError(f"cannot read JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise SolarOneShotBoundaryError("JSON root must be an object")
    return value


def _validate_legacy_seal(value: Mapping[str, Any], label: str) -> str:
    content = value.get("content_sha256")
    if not isinstance(content, str) or _SHA256.fullmatch(content) is None:
        raise SolarOneShotBoundaryError(f"{label} content hash changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if _legacy_sha(body) != content:
        raise SolarOneShotBoundaryError(f"{label} content seal changed")
    return content


def _snapshot_body(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "source_bindings",
            "candidate_ids",
            "required_opening_fields",
            "execution_contract",
            "policies",
        )
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    _exact_keys(config, _CONFIG_KEYS, "config")
    if config["schema_version"] != CONFIG_SCHEMA or config["campaign_id"] != CAMPAIGN_ID:
        raise SolarOneShotBoundaryError("config identity changed")
    bindings = config["source_bindings"]
    if not isinstance(bindings, Mapping) or not bindings:
        raise SolarOneShotBoundaryError("source bindings changed")
    for role, descriptor in bindings.items():
        _exact_keys(descriptor, _BINDING_KEYS, f"binding {role}")
        if not isinstance(descriptor["path"], str) or not descriptor["path"]:
            raise SolarOneShotBoundaryError(f"binding path changed: {role}")
        if _SHA256.fullmatch(str(descriptor["file_sha256"])) is None:
            raise SolarOneShotBoundaryError(f"binding file hash changed: {role}")
        content = descriptor["content_sha256"]
        if content is not None and _SHA256.fullmatch(str(content)) is None:
            raise SolarOneShotBoundaryError(f"binding content hash changed: {role}")
    if config["candidate_ids"] != list(_CANDIDATES):
        raise SolarOneShotBoundaryError("candidate inventory changed")
    if config["required_opening_fields"] != list(_REQUIRED_FIELDS):
        raise SolarOneShotBoundaryError("opening field registry changed")
    if config["execution_contract"] != _EXECUTION_CONTRACT:
        raise SolarOneShotBoundaryError("one-shot execution contract changed")
    if config["policies"] != _POLICIES:
        raise SolarOneShotBoundaryError("policy boundary changed")
    if config["snapshot_sha256"] != canonical_sha256(_snapshot_body(config)):
        raise SolarOneShotBoundaryError("snapshot seal changed")


def _load_bound_sources(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    values: dict[str, dict[str, Any]] = {}
    for role, descriptor in sorted(config["source_bindings"].items()):
        path = _resolve(root, descriptor["path"])
        if not path.is_file() or _file_sha256(path) != descriptor["file_sha256"]:
            raise SolarOneShotBoundaryError(f"bound source changed: {role}")
        paths[role] = path
        if path.suffix == ".json":
            value = _load_json(path)
            values[role] = value
            if descriptor["content_sha256"] is not None:
                content = _validate_legacy_seal(value, role)
                if content != descriptor["content_sha256"]:
                    raise SolarOneShotBoundaryError(f"bound content changed: {role}")
    return paths, values


def _validate_policy_and_protocol(values: Mapping[str, Mapping[str, Any]]) -> None:
    policy = values["evidence_policy"]
    protocol = values["protocol"]
    audit = values["protocol_audit"]
    if (
        policy.get("status") != "frozen"
        or protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or audit.get("status") != "pass"
        or audit.get("observational_dataset_opened") is not False
        or audit.get("formula_search_authorized") is not False
        or audit.get("policy_sha256")
        != values["cassini_registration"]["bindings"]["evidence_policy_sha256"]
    ):
        raise SolarOneShotBoundaryError("policy or protocol boundary changed")
    quantity_classes = protocol.get("quantity_classes")
    if not isinstance(quantity_classes, Mapping):
        raise SolarOneShotBoundaryError("quantity class registry changed")
    if (
        quantity_classes.get("model_dependent", {}).get("allowed_as_prediction_truth") is not False
        or quantity_classes.get("latent", {}).get("allowed_as_input_or_target") is not False
    ):
        raise SolarOneShotBoundaryError("forbidden quantity class boundary changed")


def _validate_sealed_observation_sources(values: Mapping[str, Mapping[str, Any]]) -> None:
    registration = values["cassini_registration"]
    cassini_audit = values["cassini_audit"]
    parser = values["parser_artifact"]
    calibration = values["calibration_artifact"]
    if (
        registration.get("status") != "metadata_registered_data_sealed"
        or registration.get("data_opened") is not False
        or registration.get("candidate_use_authorized") is not False
        or registration.get("readiness", {}).get("dataset_ready") is not False
        or registration.get("readiness", {}).get("primary_files_downloaded") is not False
        or cassini_audit.get("dataset_ready") is not False
        or cassini_audit.get("candidate_use_authorized") is not False
        or cassini_audit.get("observational_dataset_opened") is not False
        or cassini_audit.get("registered_catalog_files") != 16
    ):
        raise SolarOneShotBoundaryError("Cassini metadata-only boundary changed")
    if (
        parser.get("status") != "parser_ready_labels_selected_primary_records_sealed"
        or parser.get("observational_authorization") is not False
        or parser.get("metadata_selection", {}).get("primary_record_access_count") != 0
        or parser.get("metadata_selection", {}).get("target_values_accessed") is not False
        or calibration.get("status") != "calibration_implementation_ready_primary_records_sealed"
        or calibration.get("observational_authorization") is not False
        or calibration.get("primary_record_access_count") != 0
    ):
        raise SolarOneShotBoundaryError("parser/calibration sealed boundary changed")


def _validate_transfer(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    transfer = values["transfer_artifact"]
    readiness = values["g2_readiness_artifact"]
    evaluator = values["evaluator_artifact"]
    if (
        transfer.get("candidate_count") != 2
        or transfer.get("decision_counts") != {"blocked": 2}
        or transfer.get("observational_authorization") is not False
        or transfer.get("observational_data_opened") is not False
        or transfer.get("held_out_target_access_count") != 0
        or transfer.get("primary_record_access_count") != 0
        or transfer.get("real_data_pass_count") != 0
        or readiness.get("candidate_count") != 2
        or readiness.get("real_solar_bundle_admissible_count") != 0
        or readiness.get("observational_data_opened") is not False
        or evaluator.get("data_eligibility", {}).get("observational_data_opened") is not False
    ):
        raise SolarOneShotBoundaryError("G2 transfer/readiness boundary changed")
    registrations = transfer.get("candidate_registrations")
    if not isinstance(registrations, list) or len(registrations) != 2:
        raise SolarOneShotBoundaryError("candidate registrations changed")
    by_id = {item.get("candidate_id"): item for item in registrations}
    if tuple(sorted(by_id)) != _CANDIDATES:
        raise SolarOneShotBoundaryError("candidate registration identity changed")
    return [by_id[candidate_id] for candidate_id in _CANDIDATES]


def _candidate_boundary(
    registration: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_id = registration["candidate_id"]
    _validate_legacy_seal(registration, f"registration {candidate_id}")
    _validate_legacy_seal(bundle, f"bundle {candidate_id}")
    descriptor = bundle.get("descriptor")
    if (
        bundle.get("candidate_id") != candidate_id
        or bundle.get("decision") != "blocked"
        or bundle.get("held_out_targets_opened") is not False
        or bundle.get("real_source_prediction_generated") is not False
        or not isinstance(descriptor, Mapping)
        or descriptor.get("candidate_id") != candidate_id
        or descriptor.get("action_sha256") != registration.get("action_sha256")
    ):
        raise SolarOneShotBoundaryError(f"action-bound bundle changed: {candidate_id}")
    remaining = registration.get("remaining_registration_fields")
    if remaining != list(_REQUIRED_FIELDS):
        raise SolarOneShotBoundaryError(f"remaining opening fields changed: {candidate_id}")
    obligations = []
    for field in _REQUIRED_FIELDS:
        value = descriptor.get(
            "observational_opening_authorization_sha256"
            if field == "observation_opening_authorization_sha256"
            else field
        )
        obligations.append(
            {
                "field": field,
                "present": isinstance(value, str) and _SHA256.fullmatch(value) is not None,
                "value_sha256": value if isinstance(value, str) else None,
            }
        )
    if any(item["present"] for item in obligations):
        raise SolarOneShotBoundaryError("current fixed snapshot unexpectedly authorizes opening")
    return {
        "candidate_id": candidate_id,
        "action_sha256": registration["action_sha256"],
        "registration_content_sha256": registration["content_sha256"],
        "bundle_content_sha256": bundle["content_sha256"],
        "evaluator_descriptor_sha256": descriptor["evaluator_descriptor_sha256"],
        "initial_state_contract_sha256": descriptor["initial_state_contract_sha256"],
        "nuisance_likelihood_stopping_sha256": descriptor["nuisance_likelihood_stopping_sha256"],
        "source_contract_sha256": descriptor["source_contract_sha256"],
        "opening_obligations": obligations,
        "decision": "block",
        "first_blocker": "source_branch_domain_instantiation_sha256",
        "observational_data_opened": False,
        "real_data_result": None,
    }


def _lane_bindings(root: Path, config_path: Path) -> dict[str, dict[str, str]]:
    paths = {
        "config": config_path,
        "source": _resolve(root, SOURCE_PATH),
        "test": _resolve(root, TEST_PATH),
    }
    return {
        role: {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "file_sha256": _file_sha256(path),
        }
        for role, path in sorted(paths.items())
    }


def build_boundary(
    config_path: str | Path = CONFIG_PATH, *, root: str | Path = "."
) -> dict[str, Any]:
    """Build the exact zero-data BLOCK receipt and one-shot execution contract."""

    project_root = Path(root).resolve()
    resolved_config = _resolve(project_root, Path(config_path).as_posix())
    config = _load_json(resolved_config)
    _validate_config(config)
    _, values = _load_bound_sources(config, project_root)
    _validate_policy_and_protocol(values)
    _validate_sealed_observation_sources(values)
    registrations = _validate_transfer(values)
    candidate_results = [
        _candidate_boundary(registration, values[_BUNDLE_ROLES[registration["candidate_id"]]])
        for registration in registrations
    ]
    missing = sorted(
        {
            obligation["field"]
            for result in candidate_results
            for obligation in result["opening_obligations"]
            if not obligation["present"]
        }
    )
    calibration_missing = values["calibration_artifact"].get("remaining_registration_fields")
    parser_missing = values["parser_artifact"].get("remaining_registration_fields")
    if not isinstance(calibration_missing, list) or not isinstance(parser_missing, list):
        raise SolarOneShotBoundaryError("parser/calibration missing-field ledgers changed")
    readiness_checks = {
        "independent_observation_opening_authorization": "block",
        "held_out_session_split_commitment": "block",
        "selected_primary_record_roots": "block",
        "real_source_branch_domain_instantiation": "block",
        "action_bound_candidate_bundles": "pass",
        "synthetic_parser_and_calibration_implementation": "pass",
        "raw_calibrated_derived_classification": "pass",
        "model_dependent_and_latent_exclusion": "pass",
        "dark_matter_and_redshift_exclusion": "pass",
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "scope": SCOPE,
        "decision": "block",
        "first_blocker": "independent_observation_opening_authorization_absent",
        "source_bindings": {
            "lane": _lane_bindings(project_root, resolved_config),
            "evidence": config["source_bindings"],
        },
        "snapshot_sha256": config["snapshot_sha256"],
        "readiness_checks": readiness_checks,
        "candidate_results": candidate_results,
        "missing_opening_fields": missing,
        "supporting_missing_fields": {
            "parser": parser_missing,
            "calibration": calibration_missing,
        },
        "one_shot_execution_contract": {
            **config["execution_contract"],
            "status": "sealed_not_executable_until_all_opening_obligations_are_bound",
            "pre_open_assertions": [
                "verify all four per-candidate opening hashes are non-null SHA-256 values",
                "verify independent authorization binds the exact candidate, action, split, and roots",
                "verify primary and calibration roots match the preregistered split without target inspection",
                "verify training-only state checkpoint and covariance lineage",
                "atomically open the held-out target batch exactly once",
            ],
            "evaluation_order": list(_CANDIDATES),
            "no_refit_after_open": True,
            "no_promotion_side_effect": True,
        },
        "data_boundary": {
            "observational_data_opened": False,
            "primary_record_access_count": 0,
            "held_out_target_access_count": 0,
            "real_data_evaluation_count": 0,
            "candidate_use_authorized": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "raw_records": "sealed",
            "calibrated_records": "implementation_ready_but_real_roots_unregistered",
            "derived_records": "not_computed",
            "model_dependent_records_used_as_truth": False,
            "latent_quantities_used_as_input_or_target": False,
        },
        "counts": {
            "candidates": 2,
            "candidate_blocks": 2,
            "candidate_passes": 0,
            "opening_obligations_per_candidate": 4,
            "missing_opening_obligations": 8,
            "unique_missing_opening_fields": len(missing),
            "registered_catalog_metadata_files": 16,
            "primary_record_accesses": 0,
            "held_out_target_accesses": 0,
            "real_data_evaluations": 0,
        },
        "claims": {
            "one_shot_execution_ready": False,
            "observation_opening_authorized": False,
            "observational_result_exists": False,
            "candidate_rejected_by_data": False,
            "candidate_supported_by_data": False,
            "truth_established": False,
            "promotion_authorized": False,
        },
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_boundary(
    result: Mapping[str, Any],
    *,
    config_path: str | Path = CONFIG_PATH,
    root: str | Path = ".",
) -> None:
    """Require exact live replay of all evidence and the derived BLOCK receipt."""

    expected = build_boundary(config_path, root=root)
    if canonical_json_bytes(result) != canonical_json_bytes(expected):
        raise SolarOneShotBoundaryError("boundary differs from exact live replay")


def write_boundary(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = OUTPUT_PATH,
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Build, validate, and write the canonical receipt."""

    project_root = Path(root).resolve()
    result = build_boundary(config_path, root=project_root)
    validate_boundary(result, config_path=config_path, root=project_root)
    output = _resolve(project_root, Path(output_path).as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result) + b"\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    result = write_boundary(args.config, args.output, root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_ID",
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "RESULT_SCHEMA",
    "SOURCE_PATH",
    "TEST_PATH",
    "SolarOneShotBoundaryError",
    "build_boundary",
    "validate_boundary",
    "write_boundary",
]
