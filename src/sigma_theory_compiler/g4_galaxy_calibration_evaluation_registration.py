"""Register sealed G4 galaxy calibration/evaluation contracts without data access."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    CANDIDATE_ID,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_galaxy_evaluator,
)

SCHEMA_VERSION = "sigma-g4-galaxy-calibration-evaluation-registration-1.0"
SUITE_SCHEMA = "sigma-g4-galaxy-calibration-evaluation-policy-suite-1.0"
CONTRACT_HASHES = {
    "calibration_hierarchy": "1a6b70215484f04128be3e7dd0a9f864a242e26ac80ebc8409823f48fdab6f71",
    "joint_covariance_contract": "34fc798655b16dd2e52f3b97797dfee91c9f7567517034cf52890cfc42d24026",
    "held_out_split_policy": "203291172afa26f6718919c5da94b639af5e37595b1a50cb2267ca380e45d101",
    "likelihood_contract": "2f0ec63de42b186afa26fc2afc2d88b44526e283d1409a90eedd292efeb8de33",
    "stopping_rule": "7989dca776506f697c8dc37c30fbaa455a8ea76a73d1471b7281a80d181b1cc9",
}
SUITE_CONTENT_SHA256 = "49726a432706e2f30a9e7e06a5d8dcc626b829a0f609eeed7d77f9e39fa015b5"
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
        raise ValueError(f"bound G4 calibration artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or value.get("content_sha256", expected) != expected:
            raise ValueError(f"bound G4 calibration content changed: {binding['path']}")
    return value


def validate_policy_suite(suite: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "candidate_id",
        "action_sha256",
        "calibration_hierarchy",
        "joint_covariance_contract",
        "held_out_split_policy",
        "likelihood_contract",
        "stopping_rule",
        "data_eligibility",
        "content_sha256",
    }
    if (
        set(suite) != expected_keys
        or suite.get("schema_version") != SUITE_SCHEMA
        or suite.get("candidate_id") != CANDIDATE_ID
        or suite.get("action_sha256") != ACTION_SHA256
        or suite.get("data_eligibility") != ELIGIBILITY
        or suite.get("content_sha256") != SUITE_CONTENT_SHA256
        or _sha({key: value for key, value in suite.items() if key != "content_sha256"})
        != SUITE_CONTENT_SHA256
    ):
        raise ValueError("G4 calibration/evaluation policy suite changed")
    for name, expected in CONTRACT_HASHES.items():
        contract = suite.get(name)
        if not isinstance(contract, dict):
            raise TypeError(f"G4 {name} contract is missing")
        body = {key: value for key, value in contract.items() if key != "content_sha256"}
        if contract.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"G4 {name} content changed")
        if contract.get("data_eligibility") != ELIGIBILITY:
            raise ValueError(f"G4 {name} eligibility changed")

    calibration = suite["calibration_hierarchy"]
    covariance = suite["joint_covariance_contract"]
    split = suite["held_out_split_policy"]
    likelihood = suite["likelihood_contract"]
    stopping = suite["stopping_rule"]
    if (
        calibration.get("sharing_rule")
        != "global_or_preregistered_population_hierarchy_never_target_tuned_per_galaxy"
        or calibration.get("object_specific_gravity_parameter_count") != 0
        or calibration.get("posthoc_rescue_allowed") is not False
        or calibration.get("actual_calibration_values_registered") is not False
        or calibration.get("primary_calibration_root_registered") is not False
        or covariance.get("propagation_rule") != "C_output=J_raw*C_raw*J_raw^T+J_cal*C_cal*J_cal^T"
        or covariance.get("actual_covariance_registered") is not False
        or covariance.get("raw_to_calibrated_transform_registered") is not False
        or split.get("split_unit") != "whole_galaxy"
        or split.get("group_leakage_forbidden") is not True
        or split.get("galaxy_split_commitment_registration_admissible") is not False
        or any(
            split.get(name) is not False
            for name in (
                "actual_manifest_root_registered",
                "actual_assignment_root_registered",
                "split_salt_commitment_registered",
            )
        )
        or likelihood.get("lensing_role")
        != "post_freeze_independent_falsification_only_never_formula_selection"
        or likelihood.get("halo_or_invisible_component_target_allowed") is not False
        or likelihood.get("redshift_distance_allowed") is not False
        or likelihood.get("actual_residuals_or_targets_registered") is not False
        or stopping.get("test_metric_may_stop_or_select") is not False
        or stopping.get("lensing_metric_may_stop_or_select") is not False
        or stopping.get("actual_training_run_started") is not False
        or stopping.get("training_checkpoint_registered") is not False
        or stopping.get("prediction_bundle_claimed") is not False
    ):
        raise ValueError("G4 calibration/evaluation fail-closed contract changed")


def assign_synthetic_split(entry_sha256: str, synthetic_salt: str) -> str:
    """Exercise the frozen split algorithm on non-observational hash fixtures."""
    if _SHA256.fullmatch(entry_sha256) is None or _SHA256.fullmatch(synthetic_salt) is None:
        raise ValueError("synthetic split inputs must be lowercase SHA-256 values")
    digest = hashlib.sha256((entry_sha256 + synthetic_salt).encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < 0.60:
        return "formula_search_training"
    if unit < 0.80:
        return "formula_selection_validation"
    return "untouched_target_blind_test"


def joint_gaussian_nll(residual: Sequence[float], covariance: Sequence[Sequence[float]]) -> float:
    vector = np.asarray(residual, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if (
        vector.ndim != 1
        or matrix.shape != (vector.size, vector.size)
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("joint Gaussian inputs must be finite, square, and symmetric")
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0 or float(np.min(np.linalg.eigvalsh(matrix))) <= 0.0:
        raise ValueError("scored covariance must be positive definite")
    quadratic = float(vector @ np.linalg.solve(matrix, vector))
    return 0.5 * (quadratic + float(logdet) + vector.size * math.log(2.0 * math.pi))


def stopping_decision(validation_nll: Sequence[float], optimizer_step: int) -> dict[str, Any]:
    values = [float(value) for value in validation_nll]
    if optimizer_step < 0 or any(not math.isfinite(value) for value in values):
        raise ValueError("stopping inputs must be finite and nonnegative")
    best = math.inf
    stale = 0
    for value in values:
        if best - value >= 1e-6:
            best = value
            stale = 0
        else:
            stale += 1
    reasons = []
    if optimizer_step >= 500:
        reasons.append("maximum_training_optimizer_steps")
    if len(values) >= 50:
        reasons.append("maximum_validation_checks")
    if stale >= 20:
        reasons.append("validation_patience_checks")
    return {
        "stop": bool(reasons),
        "reasons": reasons,
        "validation_checks": len(values),
        "stale_checks": stale,
    }


def build_g4_galaxy_calibration_evaluation_registration(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 calibration registration eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 calibration registration opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["registration_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("G4 calibration registration source changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key != "registration_source"
    }
    suite = sources["policy_suite"]
    validate_policy_suite(suite)
    predecessor = sources["predecessor"]
    if (
        predecessor.get("decision") != "blocked"
        or predecessor.get("current_evaluator_decision", {}).get("filled_registration_hash_count")
        != 5
        or len(predecessor.get("unfilled_registration_fields", [])) != 13
        or predecessor.get("prediction_bundle_registered") is not False
        or predecessor.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 branch-distance predecessor changed")
    protocol = sources["galaxy_protocol"]
    policy = sources["evidence_policy"]
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or policy.get("status") != "frozen"
        or "redshift-derived distance" not in protocol.get("prohibited_truth_or_rescue", [])
    ):
        raise ValueError("sealed galaxy evidence policy changed")

    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration.update(predecessor["preserved_predecessor_registration_fields"])
    registration.update(predecessor["newly_filled_registration_fields"])
    newly_filled = {
        "baryonic_calibration_hierarchy_sha256": CONTRACT_HASHES["calibration_hierarchy"],
        "joint_covariance_contract_sha256": CONTRACT_HASHES["joint_covariance_contract"],
        "likelihood_contract_sha256": CONTRACT_HASHES["likelihood_contract"],
        "stopping_rule_sha256": CONTRACT_HASHES["stopping_rule"],
    }
    registration.update(newly_filled)
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
    expected_missing = sorted(name for name, value in registration.items() if value is None)
    if (
        current.get("decision") != "blocked"
        or current.get("filled_registration_hash_count") != 9
        or current.get("missing_registration_hashes") != expected_missing
        or len(expected_missing) != 9
    ):
        raise ValueError("G4 calibration staged registration ledger changed")

    synthetic_entries = ["1" * 64, "2" * 64, "3" * 64]
    salt = "a" * 64
    assignments = [assign_synthetic_split(entry, salt) for entry in synthetic_entries]
    nll = joint_gaussian_nll([1.0, -1.0], [[2.0, 0.5], [0.5, 1.0]])
    stop = stopping_decision([1.0] + [1.0] * 20, 210)
    controls_body = {
        "synthetic_entry_count": 3,
        "deterministic_assignments": assignments,
        "repeat_assignment_exact": assignments
        == [assign_synthetic_split(entry, salt) for entry in synthetic_entries],
        "joint_gaussian_nll": nll,
        "patience_stop": stop,
        "real_manifest_used": False,
        "actual_target_or_residual_used": False,
        "observational_data_opened": False,
    }
    controls = {**controls_body, "content_sha256": _sha(controls_body)}
    preserved = {
        **predecessor["preserved_predecessor_registration_fields"],
        **predecessor["newly_filled_registration_fields"],
    }
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "predecessor_content_sha256": bindings["predecessor"]["content_sha256"],
        "policy_suite_sha256": SUITE_CONTENT_SHA256,
        "held_out_split_policy_sha256": CONTRACT_HASHES["held_out_split_policy"],
        "registration_source_sha256": source_binding["file_sha256"],
        "synthetic_controls_sha256": controls["content_sha256"],
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
        "non_registration_policy_hashes": {
            "held_out_split_policy_sha256": CONTRACT_HASHES["held_out_split_policy"]
        },
        "deliberately_unfilled_registration_fields": {
            "galaxy_split_commitment_sha256": "requires_registered_real_manifest_assignment_root_and_committed_salt",
            "training_only_checkpoint_sha256": "requires_completed_authorized_training_run_under_the_committed_split",
        },
        "unfilled_registration_fields": expected_missing,
        "current_evaluator_decision": current,
        "synthetic_controls": controls,
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
        "first_missing_premise": "registered_real_source_manifest_and_frozen_split_commitment",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "Shared calibration, covariance, likelihood, and stopping contracts are hash-bound. "
            "The split algorithm is policy evidence only: no real manifest, split commitment, "
            "checkpoint, prediction bundle, target, halo input, redshift distance, or observation was used."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
