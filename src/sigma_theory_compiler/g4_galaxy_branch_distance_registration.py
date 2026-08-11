"""Hash-bound scalar-free branch and non-redshift galaxy geometry contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .g4_scalar_free_galaxy_forward_model import GEOMETRY_CONTRACT
from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    CANDIDATE_ID,
    DESCRIPTOR_FIELD,
    FORMAL_PROVENANCE_SHA256,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_galaxy_evaluator,
)

SCHEMA_VERSION = "sigma-g4-galaxy-branch-distance-registration-1.0"
BRANCH_CONTRACT_SCHEMA = "sigma-g4-scalar-free-galaxy-branch-domain-contract-1.0"
DISTANCE_CONTRACT_SCHEMA = "sigma-g4-galaxy-nonredshift-distance-geometry-contract-1.0"
GEOMETRY_REGISTRATION_SCHEMA = "sigma-g4-galaxy-source-geometry-registration-1.0"
BRANCH_CONTRACT_SHA256 = (
    "a606219458c3eeabcbe940a608dbed758288b946bce8dae26dd59a1995acc405"
)
DISTANCE_CONTRACT_SHA256 = (
    "1edcdc75721633b039e9631d5dcec6f48c0229e89d8f1ccc885fb1ac2bc210bf"
)
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
        raise ValueError(f"bound G4 branch-distance artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or (
            "content_sha256" in value and value["content_sha256"] != expected
        ):
            raise ValueError(f"bound G4 branch-distance content changed: {binding['path']}")
    return value


def validate_branch_domain_contract(contract: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "action_sha256",
        "formal_provenance_sha256",
        "scalar_free_branch_certificate_sha256",
        "field_domain",
        "exact_residual_contract",
        "forward_model_scope",
        "conditionality",
        "forbidden_inputs",
        "data_eligibility",
        "content_sha256",
    }
    conditionality = contract.get("conditionality", {})
    if (
        set(contract) != required
        or contract.get("schema_version") != BRANCH_CONTRACT_SCHEMA
        or contract.get("candidate_id") != CANDIDATE_ID
        or contract.get("action_sha256") != ACTION_SHA256
        or contract.get("formal_provenance_sha256") != FORMAL_PROVENANCE_SHA256
        or contract.get("scalar_free_branch_certificate_sha256")
        != "2bca9d26343843231a8333bc9ac2396c395c388d24f55ae488c04c05f59256dc"
        or contract.get("field_domain")
        != {
            "phi": "0_everywhere",
            "phi_at_spatial_infinity": "0",
            "metric_branch": "universally_Jordan_coupled_Einstein_solution",
            "regime": "weak_field_stationary_baryonic_source",
            "matter_coupling": "universal_Jordan_metric",
        }
        or set(contract.get("exact_residual_contract", {}).values()) != {"0"}
        or contract.get("forward_model_scope", {}).get(
            "object_specific_gravity_parameter_count"
        )
        != 0
        or conditionality.get("contract_status")
        != "certified_exact_conditional_branch"
        or conditionality.get("source_specific_branch_selection_proven") is not False
        or conditionality.get("prediction_bundle_claimed") is not False
        or conditionality.get("observational_evidence_claimed") is not False
        or "not uniqueness" not in conditionality.get("warning", "")
        or contract.get("data_eligibility") != ELIGIBILITY
        or contract.get("content_sha256") != BRANCH_CONTRACT_SHA256
    ):
        raise ValueError("G4 scalar-free branch/domain contract changed")


def validate_distance_geometry_contract(contract: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate_id",
        "action_sha256",
        "distance_modes",
        "required_future_geometry_registration",
        "forward_model_mapping",
        "current_registration_state",
        "forbidden_inputs",
        "data_eligibility",
        "content_sha256",
    }
    modes = contract.get("distance_modes", {})
    mapping = contract.get("forward_model_mapping", {})
    current = contract.get("current_registration_state", {})
    if (
        set(contract) != required
        or contract.get("schema_version") != DISTANCE_CONTRACT_SCHEMA
        or contract.get("candidate_id") != CANDIDATE_ID
        or contract.get("action_sha256") != ACTION_SHA256
        or modes.get("forward_model_mode")
        != "separately_registered_nonredshift_metric_distance"
        or modes.get("redshift_distance_allowed") is not False
        or modes.get("cosmological_model_distance_allowed") is not False
        or mapping.get("geometry_contract") != GEOMETRY_CONTRACT
        or mapping.get("distance_provenance_literal")
        != "separately_registered_nonredshift_distance_required"
        or mapping.get("object_specific_gravity_parameter_count") != 0
        or any(value is not False for value in current.values())
        or "redshift_derived_distance_or_physical_size"
        not in contract.get("forbidden_inputs", [])
        or contract.get("data_eligibility") != ELIGIBILITY
        or contract.get("content_sha256") != DISTANCE_CONTRACT_SHA256
    ):
        raise ValueError("G4 non-redshift distance/geometry contract changed")


def validate_source_geometry_registration(registration: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "nonredshift_lens_distance_m",
        "lensing_distance_ratio_D_ls_over_D_s",
        "lens_distance_measurement_provenance_sha256",
        "lensing_geometry_measurement_provenance_sha256",
        "distance_and_geometry_covariance_sha256",
        "independent_nonredshift_distance_audit_sha256",
        "primary_source_manifest_root_sha256",
        "redshift_used_as_distance",
        "observational_data_opened_by_contract",
        "geometry_contract",
        "object_specific_gravity_parameters",
    }
    if set(registration) != required or registration.get(
        "schema_version"
    ) != GEOMETRY_REGISTRATION_SCHEMA:
        raise ValueError("future G4 galaxy geometry registration fields changed")
    distance = float(registration["nonredshift_lens_distance_m"])
    ratio = float(registration["lensing_distance_ratio_D_ls_over_D_s"])
    hashes = [value for key, value in registration.items() if key.endswith("_sha256")]
    if (
        not math.isfinite(distance)
        or distance <= 0.0
        or not math.isfinite(ratio)
        or not 0.0 <= ratio <= 1.0
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in hashes)
        or registration.get("redshift_used_as_distance") is not False
        or registration.get("observational_data_opened_by_contract") is not False
        or registration.get("geometry_contract") != GEOMETRY_CONTRACT
        or registration.get("object_specific_gravity_parameters") != {}
    ):
        raise ValueError("future G4 galaxy geometry registration violates the contract")


def geometry_registration_to_forward_profile_fields(
    registration: dict[str, Any]
) -> dict[str, Any]:
    validate_source_geometry_registration(registration)
    return {
        "nonredshift_lens_distance_m": float(
            registration["nonredshift_lens_distance_m"]
        ),
        "lensing_distance_ratio": float(
            registration["lensing_distance_ratio_D_ls_over_D_s"]
        ),
        "distance_provenance": "separately_registered_nonredshift_distance_required",
        "geometry_contract": dict(GEOMETRY_CONTRACT),
    }


def _synthetic_geometry_registration() -> dict[str, Any]:
    return {
        "schema_version": GEOMETRY_REGISTRATION_SCHEMA,
        "nonredshift_lens_distance_m": 1.0e20,
        "lensing_distance_ratio_D_ls_over_D_s": 0.5,
        "lens_distance_measurement_provenance_sha256": "1" * 64,
        "lensing_geometry_measurement_provenance_sha256": "2" * 64,
        "distance_and_geometry_covariance_sha256": "3" * 64,
        "independent_nonredshift_distance_audit_sha256": "4" * 64,
        "primary_source_manifest_root_sha256": "5" * 64,
        "redshift_used_as_distance": False,
        "observational_data_opened_by_contract": False,
        "geometry_contract": dict(GEOMETRY_CONTRACT),
        "object_specific_gravity_parameters": {},
    }


def build_g4_galaxy_branch_distance_registration(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G4 branch-distance registration eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("G4 branch-distance registration opened observations")
    bindings = config["source_bindings"]
    source_binding = bindings["registration_source"]
    source_path = root / source_binding["path"]
    if not source_path.is_file() or _file_sha(source_path) != source_binding["file_sha256"]:
        raise ValueError("G4 branch-distance registration source changed")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key != "registration_source"
    }
    validate_branch_domain_contract(sources["branch_domain_contract"])
    validate_distance_geometry_contract(sources["distance_geometry_contract"])
    protocol = sources["galaxy_protocol"]
    policy = sources["evidence_policy"]
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or "separately frozen non-redshift distance protocol"
        not in protocol.get("discovery_channel", {}).get("distance_exception", "")
        or "redshift-derived distance"
        not in protocol.get("prohibited_truth_or_rescue", [])
        or policy.get("status") != "frozen"
        or "treating redshift as a distance"
        not in policy.get("redshift", {}).get("not_allowed_by_default", [])
    ):
        raise ValueError("sealed galaxy non-redshift evidence policy changed")
    branch_record = sources["scalar_free_branch"]["candidate_records"][0]
    branch = branch_record["exact_scalar_free_branch_certificate"]
    if (
        branch_record.get("seed_id") != CANDIDATE_ID
        or branch_record.get("action_sha256") != ACTION_SHA256
        or branch.get("content_sha256")
        != sources["branch_domain_contract"]["scalar_free_branch_certificate_sha256"]
        or branch.get("exact_field_equation_residuals")
        != sources["branch_domain_contract"]["exact_residual_contract"]
        or branch.get("branch_selection_warning") is None
    ):
        raise ValueError("G4 branch certificate does not prove the conditional contract")
    forward = sources["forward_model"]
    predecessor = forward["current_evaluator_decision"]
    if (
        forward.get("decision") != "blocked"
        or predecessor.get("filled_registration_hash_count") != 3
        or len(forward.get("unfilled_registration_fields", [])) != 15
        or set(forward.get("newly_filled_registration_fields", {}))
        != {
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
        or forward.get("prediction_bundle_registered") is not False
        or forward.get("observational_data_opened") is not False
    ):
        raise ValueError("G4 forward-model predecessor registration changed")

    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration.update(forward["preserved_predecessor_registration_fields"])
    registration.update(forward["newly_filled_registration_fields"])
    registration["branch_and_domain_contract_sha256"] = BRANCH_CONTRACT_SHA256
    registration["distance_mode_contract_sha256"] = DISTANCE_CONTRACT_SHA256
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
    filled = {
        DESCRIPTOR_FIELD,
        "rotation_prediction_implementation_sha256",
        "lensing_prediction_implementation_sha256",
        "branch_and_domain_contract_sha256",
        "distance_mode_contract_sha256",
    }
    expected_missing = sorted(set(REQUIRED_REGISTRATION_HASHES) - filled)
    if (
        current.get("decision") != "blocked"
        or current.get("filled_registration_hash_count") != 5
        or current.get("missing_registration_hashes") != expected_missing
    ):
        raise ValueError("G4 branch-distance staged registration ledger changed")

    synthetic_registration = _synthetic_geometry_registration()
    mapped = geometry_registration_to_forward_profile_fields(synthetic_registration)
    if (
        mapped["nonredshift_lens_distance_m"] * 0.01 != 1.0e18
        or mapped["lensing_distance_ratio"] != 0.5
        or mapped["geometry_contract"] != GEOMETRY_CONTRACT
    ):
        raise ValueError("G4 non-redshift geometry synthetic mapping failed")
    controls_body = {
        "branch_exact_residuals": sources["branch_domain_contract"][
            "exact_residual_contract"
        ],
        "branch_conditionality_preserved": True,
        "nonredshift_geometry_mapping": mapped,
        "synthetic_physical_radius_m": 1.0e18,
        "real_source_geometry_registered": False,
        "observational_data_opened": False,
    }
    controls = {**controls_body, "content_sha256": _sha(controls_body)}
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "formal_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "scalar_free_branch_sha256": branch["content_sha256"],
        "branch_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
        "distance_geometry_contract_sha256": DISTANCE_CONTRACT_SHA256,
        "forward_model_predecessor_sha256": bindings["forward_model"][
            "content_sha256"
        ],
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
        "newly_filled_registration_fields": {
            "branch_and_domain_contract_sha256": BRANCH_CONTRACT_SHA256,
            "distance_mode_contract_sha256": DISTANCE_CONTRACT_SHA256,
        },
        "preserved_predecessor_registration_fields": {
            name: registration[name] for name in sorted(filled - {
                "branch_and_domain_contract_sha256",
                "distance_mode_contract_sha256",
            })
        },
        "unfilled_registration_fields": expected_missing,
        "current_evaluator_decision": current,
        "synthetic_controls": controls,
        "branch_contract_status": "certified_exact_conditional_branch",
        "source_specific_branch_selection_proven": False,
        "distance_geometry_contract_status": "certified_interface_no_real_values",
        "real_source_geometry_registered": False,
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
        "first_missing_premise": "registered_real_baryonic_source_and_geometry_manifest",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The exact conditional scalar-free branch interface and the non-redshift geometry "
            "interface are now hash-bound. This does not prove that a real galaxy selects the "
            "branch and supplies no distance value, source manifest, prediction bundle, split, "
            "likelihood, halo input, redshift distance, or observation."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
