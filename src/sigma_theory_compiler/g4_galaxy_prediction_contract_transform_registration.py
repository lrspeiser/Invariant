"""Audit the G4 galaxy bundle schema and register a sealed calibration adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

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

SCHEMA_VERSION = "sigma-g4-galaxy-prediction-contract-transform-registration-1.0"
TRANSFORM_CONTRACT_SCHEMA = "sigma-g4-galaxy-raw-to-calibrated-transform-contract-1.0"
TRANSFORM_CONTRACT_SHA256 = "bf16202dd820486812c544e4839cae1dcbfc4b3e960ece9cda3af92c747b1b6e"
BRANCH_CONTRACT_SHA256 = "a606219458c3eeabcbe940a608dbed758288b946bce8dae26dd59a1995acc405"
ALLOWED_OUTPUT_CHANNELS = {
    "stellar_light_calibrated_input_intensity_per_sr",
    "gas_line_calibrated_input_intensity_per_sr",
    "baryonic_point_calibrated_integrated_flux",
    "angular_radius_rad",
    "inclination_rad",
    "position_angle_rad",
}
ALLOWED_OUTPUT_UNITS = {
    "calibrated_input_intensity_per_sr",
    "calibrated_integrated_flux",
    "rad",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound G4 prediction/transform artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or value.get("content_sha256", expected) != expected:
            raise ValueError(f"bound G4 prediction/transform content changed: {binding['path']}")
    return value


def validate_prediction_bundle_contract(contract: dict[str, Any]) -> None:
    required = {
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
    properties = contract.get("properties", {})
    if (
        _sha(contract) != PREDICTION_BUNDLE_CONTRACT_SHA256
        or contract.get("type") != "object"
        or contract.get("additionalProperties") is not False
        or set(contract.get("required", [])) != required
        or set(properties) != required
        or properties.get("schema_version", {}).get("const") != BUNDLE_SCHEMA
        or properties.get("candidate_id", {}).get("const") != CANDIDATE_ID
        or properties.get("action_sha256", {}).get("const") != ACTION_SHA256
        or properties.get("formal_provenance_sha256", {}).get("const") != FORMAL_PROVENANCE_SHA256
        or properties.get("input_contract", {}).get("const") != INPUT_CONTRACT
        or properties.get("output_contract", {}).get("const") != OUTPUT_CONTRACT
        or properties.get("object_specific_gravity_parameter_count", {}).get("const") != 0
        or properties.get("data_eligibility", {}).get("const") != ELIGIBILITY
        or properties.get("observational_data_opened", {}).get("const") is not False
        or contract.get("$defs", {}).get("sha256", {}).get("pattern") != "^[0-9a-f]{64}$"
    ):
        raise ValueError("G4 prediction-bundle contract changed")


def validate_transform_contract(contract: dict[str, Any]) -> None:
    body = {key: value for key, value in contract.items() if key != "content_sha256"}
    state = contract.get("current_registration_state", {})
    transform = contract.get("transform", {})
    output = contract.get("output_contract", {})
    if (
        contract.get("schema_version") != TRANSFORM_CONTRACT_SCHEMA
        or contract.get("candidate_id") != CANDIDATE_ID
        or contract.get("action_sha256") != ACTION_SHA256
        or contract.get("branch_and_domain_contract_sha256") != BRANCH_CONTRACT_SHA256
        or contract.get("prediction_bundle_contract_sha256") != PREDICTION_BUNDLE_CONTRACT_SHA256
        or contract.get("content_sha256") != TRANSFORM_CONTRACT_SHA256
        or _sha(body) != TRANSFORM_CONTRACT_SHA256
        or transform.get("equation") != "y_cal=A_raw_to_cal*x_raw+b_cal"
        or transform.get("shared_or_preregistered_population_calibration_only") is not True
        or transform.get("object_specific_gravity_parameter_count") != 0
        or set(output.get("allowed_channels", [])) != ALLOWED_OUTPUT_CHANNELS
        or set(output.get("allowed_units", [])) != ALLOWED_OUTPUT_UNITS
        or output.get("full_cross_channel_covariance_retained") is not True
        or any(value is not False for value in state.values())
        or contract.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("G4 raw-to-calibrated transform contract changed")


def _covariance(value: Any, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (size, size)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
        or float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12
    ):
        raise ValueError(f"{name} must be finite, symmetric, and positive semidefinite")
    return matrix


def apply_registered_linear_calibration(
    raw_vector: Any,
    raw_covariance: Any,
    operator: Any,
    offset: Any,
    nuisance_jacobian: Any,
    nuisance_covariance: Any,
    output_channels: list[str],
    output_units: list[str],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Apply a future hash-registered linear calibration to synthetic or sealed inputs."""
    if context != {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "branch_and_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
        "shared_across_all_objects": True,
        "object_specific_gravity_parameters": {},
        "redshift_used_as_distance": False,
        "data_eligibility": ELIGIBILITY,
    }:
        raise ValueError("G4 calibration context violates the candidate-specific sealed contract")
    raw = np.asarray(raw_vector, dtype=float)
    transform = np.asarray(operator, dtype=float)
    bias = np.asarray(offset, dtype=float)
    nuisance = np.asarray(nuisance_jacobian, dtype=float)
    if (
        raw.ndim != 1
        or raw.size == 0
        or transform.ndim != 2
        or transform.shape[1] != raw.size
        or bias.shape != (transform.shape[0],)
        or nuisance.ndim != 2
        or nuisance.shape[0] != transform.shape[0]
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(transform))
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(nuisance))
    ):
        raise ValueError("G4 raw calibration arrays have incompatible shape or nonfinite values")
    if (
        len(output_channels) != transform.shape[0]
        or len(output_units) != transform.shape[0]
        or not set(output_channels).issubset(ALLOWED_OUTPUT_CHANNELS)
        or not set(output_units).issubset(ALLOWED_OUTPUT_UNITS)
        or len(set(output_channels)) != len(output_channels)
    ):
        raise ValueError("G4 calibrated output channel or unit is not admitted")
    raw_cov = _covariance(raw_covariance, raw.size, "raw covariance")
    nuisance_cov = _covariance(
        nuisance_covariance, nuisance.shape[1], "calibration nuisance covariance"
    )
    calibrated = transform @ raw + bias
    covariance = transform @ raw_cov @ transform.T + nuisance @ nuisance_cov @ nuisance.T
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(calibrated)) or np.any(calibrated < 0.0):
        raise ValueError("G4 calibrated direct-source values must be finite and nonnegative")
    _covariance(covariance, calibrated.size, "calibrated covariance")
    return {
        "calibrated_values": calibrated.tolist(),
        "joint_covariance": covariance.tolist(),
        "output_channels": list(output_channels),
        "output_units": list(output_units),
        "object_specific_gravity_parameter_count": 0,
        "redshift_distance_inputs": False,
        "observational_data_opened": False,
    }


def _synthetic_control() -> dict[str, Any]:
    context = {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "branch_and_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
        "shared_across_all_objects": True,
        "object_specific_gravity_parameters": {},
        "redshift_used_as_distance": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    result = apply_registered_linear_calibration(
        [10.0, 20.0],
        [[4.0, 1.0], [1.0, 9.0]],
        [[2.0, 0.0], [0.0, 3.0]],
        [1.0, 2.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.25, 0.05], [0.05, 0.36]],
        [
            "stellar_light_calibrated_input_intensity_per_sr",
            "gas_line_calibrated_input_intensity_per_sr",
        ],
        ["calibrated_input_intensity_per_sr"] * 2,
        context,
    )
    expected = {
        "values": [21.0, 62.0],
        "covariance": [[16.25, 6.05], [6.05, 81.36]],
    }
    if not np.allclose(result["calibrated_values"], expected["values"], rtol=0.0, atol=1e-12):
        raise ValueError("G4 synthetic calibration value control failed")
    if not np.allclose(result["joint_covariance"], expected["covariance"], rtol=0.0, atol=1e-12):
        raise ValueError("G4 synthetic calibration covariance control failed")
    body = {
        "role": "synthetic_linear_calibration_only_not_candidate_evidence",
        "expected": expected,
        "result": result,
        "cross_channel_covariance_retained": result["joint_covariance"][0][1] != 0.0,
        "real_operator_or_values_registered": False,
        "observational_data_opened": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_g4_galaxy_prediction_contract_transform_registration(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 prediction/transform eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 prediction/transform registration opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["registration_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("G4 prediction/transform registration source changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key not in {"registration_source", "forward_model_source"}
    }
    forward_path = root / bindings["forward_model_source"]["path"]
    if (
        not forward_path.is_file()
        or _file_sha(forward_path) != bindings["forward_model_source"]["file_sha256"]
    ):
        raise ValueError("G4 galaxy forward-model source changed")
    validate_prediction_bundle_contract(sources["prediction_bundle_contract"])
    validate_transform_contract(sources["transform_contract"])
    predecessor = sources["predecessor"]
    if (
        predecessor.get("decision") != "blocked"
        or predecessor.get("current_evaluator_decision", {}).get("filled_registration_hash_count")
        != 9
        or len(predecessor.get("unfilled_registration_fields", [])) != 9
        or predecessor.get("prediction_bundle_registered") is not False
        or predecessor.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 calibration/evaluation predecessor changed")

    control = _synthetic_control()
    readiness_body = {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "branch_and_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
        "prediction_bundle_contract_sha256": PREDICTION_BUNDLE_CONTRACT_SHA256,
        "transform_contract_sha256": TRANSFORM_CONTRACT_SHA256,
        "implementation_source_sha256": source_binding["file_sha256"],
        "forward_model_source_sha256": bindings["forward_model_source"]["file_sha256"],
        "synthetic_control_sha256": control["content_sha256"],
        "real_transform_inputs_registered": False,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    readiness = {**readiness_body, "content_sha256": _sha(readiness_body)}
    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration.update(predecessor["preserved_predecessor_registration_fields"])
    registration.update(predecessor["newly_filled_registration_fields"])
    registration.update(
        {
            "prediction_bundle_contract_sha256": PREDICTION_BUNDLE_CONTRACT_SHA256,
            "raw_to_calibrated_transform_sha256": readiness["content_sha256"],
        }
    )
    current = reviewed_g4_candidate_galaxy_evaluator(
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
    missing = sorted(name for name, value in registration.items() if value is None)
    if (
        current.get("decision") != "blocked"
        or current.get("filled_registration_hash_count") != 11
        or current.get("missing_registration_hashes") != missing
        or len(missing) != 7
    ):
        raise ValueError("G4 prediction/transform staged ledger changed")
    preserved = {
        **predecessor["preserved_predecessor_registration_fields"],
        **predecessor["newly_filled_registration_fields"],
    }
    newly_filled = {
        "prediction_bundle_contract_sha256": PREDICTION_BUNDLE_CONTRACT_SHA256,
        "raw_to_calibrated_transform_sha256": readiness["content_sha256"],
    }
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "predecessor_content_sha256": bindings["predecessor"]["content_sha256"],
        "prediction_bundle_contract_sha256": PREDICTION_BUNDLE_CONTRACT_SHA256,
        "transform_contract_sha256": TRANSFORM_CONTRACT_SHA256,
        "implementation_readiness_sha256": readiness["content_sha256"],
        "registration_source_sha256": source_binding["file_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "source_bindings": bindings,
        "newly_filled_registration_fields": newly_filled,
        "preserved_predecessor_registration_fields": {
            name: preserved[name] for name in sorted(preserved)
        },
        "unfilled_registration_fields": missing,
        "current_evaluator_decision": current,
        "transform_implementation_readiness": readiness,
        "synthetic_control": control,
        "real_transform_inputs_registered": False,
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
        "first_missing_premise": "registered_real_source_manifest_and_selected_primary_roots",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The already-reviewed prediction-bundle schema and candidate-bound linear raw-to-calibrated "
            "implementation are registered. No real raw vector, calibration operator, source root, split, "
            "checkpoint, prediction-bundle content/file, target, halo input, redshift distance, or observation exists."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
