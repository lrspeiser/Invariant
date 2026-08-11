"""Sealed candidate-specific galaxy direct-observable evaluator readiness."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-reviewed-g4-candidate-galaxy-evaluator-readiness-1.0"
DESCRIPTOR_SCHEMA = "sigma-candidate-galaxy-evaluator-descriptor-1.0"
BUNDLE_SCHEMA = "sigma-g4-galaxy-direct-observable-prediction-bundle-1.0"
CANDIDATE_ID = "G3-f9c598b70a77ea54009d8f18"
ACTION_SHA256 = "6ddd6502d110ead90ff494a6569213ec2e61a0b046dfa86344bb1980df6abc90"
FORMAL_PROVENANCE_SHA256 = (
    "201d7caf11473c618d7f4c494faa9f5eee37604863baf4cbe5d7af3de7c30dca"
)
PREDICTION_BUNDLE_CONTRACT_SHA256 = (
    "9ce0f7579375227d4ddf54a6157352e80cd168d75755fef63fbe983bd4f0cb4d"
)
CALLBACK = (
    "sigma_theory_compiler.reviewed_g4_candidate_galaxy_evaluator:"
    "reviewed_g4_candidate_galaxy_evaluator"
)
INPUT_CONTRACT = {
    "channels": [
        "angular_radius_and_relative_sky_position",
        "surface_brightness_or_image_pixels_with_calibration_and_covariance",
        "HI_Halpha_or_CO_line_emission_with_beam_and_line_spread_provenance",
        "angular_geometry_with_inclination_position_angle_image_provenance",
        "globally_frozen_or_hierarchically_shared_baryonic_calibration_nuisances",
    ],
    "distance_mode": "angular_dimensionless_or_separately_registered_nonredshift",
    "forbidden": [
        "galaxy_identity_as_formula_input",
        "dark_matter_or_halo_label",
        "rotation_curve_derived_mass_discrepancy",
        "redshift_derived_distance_or_environment",
        "lensing_derived_mass_or_convergence_truth",
        "per_galaxy_gravity_parameter",
    ],
}
OUTPUT_CONTRACT = {
    "rotation": [
        "held_out_spectral_line_centroid_wavelength_ratio",
        "reproducible_line_of_sight_Doppler_velocity_representation",
    ],
    "lensing": [
        "relative_image_or_arc_positions",
        "image_parity_and_topology",
        "directly_measured_time_delay_when_available",
    ],
    "same_covariant_action_and_universal_constants": True,
    "lensing_formula_selection_use": False,
}
DESCRIPTOR_FIELD = "reviewed_candidate_galaxy_evaluator_descriptor_sha256"
REQUIRED_REGISTRATION_HASHES = (
    "prediction_bundle_file_sha256",
    "prediction_bundle_content_sha256",
    "prediction_bundle_contract_sha256",
    "branch_and_domain_contract_sha256",
    "rotation_prediction_implementation_sha256",
    "lensing_prediction_implementation_sha256",
    "baryonic_calibration_hierarchy_sha256",
    "raw_to_calibrated_transform_sha256",
    "joint_covariance_contract_sha256",
    "likelihood_contract_sha256",
    "galaxy_split_commitment_sha256",
    "training_only_checkpoint_sha256",
    "stopping_rule_sha256",
    "distance_mode_contract_sha256",
    "dataset_manifest_independent_audit_sha256",
    "selected_primary_imaging_and_spectroscopy_root_sha256",
    "selected_primary_calibration_root_sha256",
    DESCRIPTOR_FIELD,
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
        raise ValueError(f"bound G4 galaxy readiness artifact changed: {binding['path']}")
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
            raise ValueError(f"bound G4 galaxy readiness content changed: {binding['path']}")
    return value


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def _validate_future_bundle(
    registration: dict[str, str], context: dict[str, Any]
) -> None:
    if (
        registration["prediction_bundle_contract_sha256"]
        != PREDICTION_BUNDLE_CONTRACT_SHA256
        or context.get("evaluator_descriptor_binding_sha256")
        != registration[DESCRIPTOR_FIELD]
    ):
        raise ValueError("future galaxy contract or evaluator descriptor binding changed")
    bundle = context.get("prediction_bundle")
    if not isinstance(bundle, dict):
        raise TypeError("fully registered galaxy context lacks prediction bundle")
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
    if (
        set(bundle) != required
        or bundle["schema_version"] != BUNDLE_SCHEMA
        or bundle["candidate_id"] != CANDIDATE_ID
        or bundle["action_sha256"] != ACTION_SHA256
        or bundle["formal_provenance_sha256"] != FORMAL_PROVENANCE_SHA256
        or bundle["input_contract"] != INPUT_CONTRACT
        or bundle["output_contract"] != OUTPUT_CONTRACT
        or not isinstance(bundle["universal_parameter_count"], int)
        or bundle["universal_parameter_count"] < 0
        or bundle["object_specific_gravity_parameter_count"] != 0
        or bundle["data_eligibility"] != ELIGIBILITY
        or bundle["observational_data_opened"] is not False
    ):
        raise ValueError("future G4 galaxy prediction bundle violates the sealed contract")
    mapping = {
        "branch_and_domain_contract_sha256": "branch_and_domain_contract_sha256",
        "rotation_prediction_implementation_sha256": (
            "rotation_prediction_implementation_sha256"
        ),
        "lensing_prediction_implementation_sha256": (
            "lensing_prediction_implementation_sha256"
        ),
        "baryonic_calibration_hierarchy_sha256": (
            "baryonic_calibration_hierarchy_sha256"
        ),
        "joint_covariance_contract_sha256": "joint_covariance_contract_sha256",
        "likelihood_contract_sha256": "likelihood_contract_sha256",
        "galaxy_split_commitment_sha256": "galaxy_split_commitment_sha256",
        "training_only_checkpoint_sha256": "training_only_checkpoint_sha256",
        "stopping_rule_sha256": "stopping_rule_sha256",
        "distance_mode_contract_sha256": "distance_mode_contract_sha256",
    }
    for registration_name, bundle_name in mapping.items():
        if registration[registration_name] != bundle[bundle_name]:
            raise ValueError(f"future galaxy bundle hash mismatch: {registration_name}")
    if _sha(bundle) != registration["prediction_bundle_content_sha256"]:
        raise ValueError("future galaxy prediction bundle content hash mismatch")


def reviewed_g4_candidate_galaxy_evaluator(
    candidate: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Validate registrations only; never access a galaxy record or return a data pass."""

    if candidate != {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "role": "generated_candidate",
        "data_eligibility": ELIGIBILITY,
    }:
        raise ValueError("reviewed G4 galaxy candidate identity changed")
    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("reviewed G4 galaxy context eligibility changed")
    if context.get("observational_opening_authorized") is not False:
        raise ValueError("reviewed G4 galaxy evaluator cannot authorize data opening")
    registration = context.get("registration_hashes")
    if registration is None:
        registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    if not isinstance(registration, dict):
        raise TypeError("galaxy registration hashes must be a mapping")
    unknown = sorted(set(registration) - set(REQUIRED_REGISTRATION_HASHES))
    if unknown:
        raise ValueError(f"unknown galaxy registration hashes: {unknown}")
    present = {name: value for name, value in registration.items() if value is not None}
    for name, value in present.items():
        _require_hash(value, name)
    missing = sorted(
        name for name in REQUIRED_REGISTRATION_HASHES if registration.get(name) is None
    )
    if missing:
        return {
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "decision": "blocked",
            "blocker": "missing_registered_galaxy_prediction_and_data_contracts",
            "filled_registration_hash_count": len(present),
            "missing_registration_hashes": missing,
            "candidate_use_authorized": False,
            "observational_opening_authorized": False,
            "observational_data_opened": False,
            "primary_record_access_count": 0,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "data_eligibility": dict(ELIGIBILITY),
        }
    if set(registration) != set(REQUIRED_REGISTRATION_HASHES):
        raise ValueError("future galaxy registration field set changed")
    _validate_future_bundle(registration, context)
    return {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "decision": "blocked",
        "blocker": "separate_observational_opening_authorization_required",
        "readiness": "fully_registered_prediction_bundle_validated",
        "filled_registration_hash_count": len(registration),
        "missing_registration_hashes": [],
        "candidate_use_authorized": False,
        "observational_opening_authorized": False,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _synthetic_shape_fixture() -> dict[str, Any]:
    angular_bins = np.linspace(0.1, 1.0, 5)
    surface_brightness = np.linspace(1.0, 0.2, 5)
    gas_channels = np.zeros((5, 3), dtype=float)
    angular_geometry = np.array([0.8, 0.2], dtype=float)
    rotation_output = np.zeros(5, dtype=float)
    lensing_output = np.zeros((3, 2), dtype=float)
    if (
        angular_bins.shape != surface_brightness.shape
        or gas_channels.shape[0] != angular_bins.size
        or angular_geometry.shape != (2,)
        or rotation_output.shape != (5,)
        or lensing_output.shape != (3, 2)
    ):
        raise ValueError("synthetic galaxy shape contract failed")
    body = {
        "role": "synthetic_shape_only_not_candidate_prediction",
        "decision": "pass",
        "input_shapes": {
            "angular_bins": [5],
            "surface_brightness": [5],
            "gas_line_channels": [5, 3],
            "angular_geometry": [2],
        },
        "output_shapes": {"rotation": [5], "lensing_relative_positions": [3, 2]},
        "object_specific_gravity_parameter_count": 0,
        "observational_data_opened": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _synthetic_covariance_fixture() -> dict[str, Any]:
    standard = np.linspace(1.0e-4, 3.0e-4, 11)
    shared = np.linspace(1.0e-5, 2.0e-5, 11)
    covariance = np.diag(standard**2) + np.outer(shared, shared)
    minimum = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum <= 0.0 or covariance[0, 5] == 0.0 or covariance.shape != (11, 11):
        raise ValueError("synthetic galaxy covariance contract failed")
    body = {
        "role": "synthetic_covariance_only_not_candidate_evidence",
        "decision": "pass",
        "joint_rotation_lensing_covariance_shape": [11, 11],
        "positive_definite": True,
        "cross_channel_covariance_nonzero": True,
        "observational_data_opened": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_reviewed_g4_galaxy_evaluator_readiness(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("reviewed G4 galaxy readiness eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("reviewed G4 galaxy readiness opened observations")
    bindings = config["source_bindings"]
    source_only = {"reviewed_evaluator_source", "generic_evaluator_source"}
    for key in source_only:
        binding = bindings[key]
        path = root / binding["path"]
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"bound G4 galaxy evaluator source changed: {binding['path']}")
    sources = {
        key: _load_bound(root, binding)
        for key, binding in bindings.items()
        if key not in source_only
    }

    dossiers = [
        item
        for item in sources["candidate_dossier"].get("dossiers", [])
        if item.get("dossier_id") == CANDIDATE_ID
    ]
    if len(dossiers) != 1 or dossiers[0].get("content_sha256") != bindings[
        "candidate_dossier"
    ]["candidate_dossier_sha256"]:
        raise ValueError("reviewed G4 galaxy dossier changed")
    nodes = {item["node_id"]: item for item in dossiers[0]["hierarchy_nodes"]}
    if (
        dossiers[0].get("overall_status") != "blocked_after_formal_pass"
        or nodes["defining_covariant_action"].get("action_sha256") != ACTION_SHA256
        or nodes["adm_dirac_obligation"].get("status") != "proven"
        or nodes["principal_symbol_obligation"].get("status") != "proven"
        or nodes["global_energy_obligation"].get("status") != "proven"
    ):
        raise ValueError("reviewed G4 galaxy formal dossier hierarchy changed")

    formal_records = [
        item
        for item in sources["formal_pass"].get("candidate_records", [])
        if item.get("seed_id") == CANDIDATE_ID
    ]
    if (
        len(formal_records) != 1
        or formal_records[0].get("action_sha256") != ACTION_SHA256
        or formal_records[0].get("decision") != "pass"
        or formal_records[0].get("first_missing_premise") is not None
        or formal_records[0].get("provenance", {}).get("binding_sha256")
        != FORMAL_PROVENANCE_SHA256
    ):
        raise ValueError("reviewed G4 galaxy formal pass changed")

    prediction_records = sources["scalar_free_ppn_prediction"].get(
        "candidate_records", []
    )
    if len(prediction_records) != 1:
        raise ValueError("reviewed G4 galaxy prediction record set changed")
    prediction = prediction_records[0]
    scalar_free = prediction.get("exact_scalar_free_branch_certificate", {})
    ppn = prediction.get("coupling_and_PPN_certificate", {})
    if (
        prediction.get("seed_id") != CANDIDATE_ID
        or prediction.get("action_sha256") != ACTION_SHA256
        or prediction.get("candidate_analytic_prediction_status")
        != "pass_on_declared_scalar_free_background"
        or scalar_free.get("content_sha256")
        != "2bca9d26343843231a8333bc9ac2396c395c388d24f55ae488c04c05f59256dc"
        or ppn.get("content_sha256")
        != "4e90877b2e49f682a6457e65f822c5b09d6773f5df041635cca70d1ecb8c12a2"
        or ppn.get("PPN_prediction", {}).get("gamma") != "1"
        or ppn.get("PPN_prediction", {}).get("beta") != "1"
        or scalar_free.get("branch_selection_warning") is None
    ):
        raise ValueError("reviewed G4 galaxy scalar-free/PPN binding changed")

    protocol = sources["galaxy_protocol"]
    policy = sources["evidence_policy"]
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or protocol.get("scoring_contract", {}).get(
            "object_specific_gravity_parameters"
        )
        != 0
        or "dark matter" not in protocol.get("prohibited_truth_or_rescue", [])
        or "redshift-derived distance"
        not in protocol.get("prohibited_truth_or_rescue", [])
        or policy.get("status") != "frozen"
        or policy.get("unobserved_components", {}).get("default_status")
        != "prohibited_as_truth_or_rescue"
    ):
        raise ValueError("sealed galaxy protocol or evidence policy changed")
    generic = sources["generic_evaluator_status"]
    if (
        generic.get("registered_prediction_bundle_count") != 0
        or generic.get("source_registrations_loaded") != 0
        or generic.get("observational_data_opened") is not False
        or generic.get("decision_counts") != {"blocked": 70}
    ):
        raise ValueError("generic galaxy evaluator seal changed")

    contract = sources["prediction_bundle_contract"]
    if (
        contract.get("$id")
        != "sigma://grammar-v3/g4-galaxy-direct-observable-prediction-bundle-1.0"
        or contract.get("additionalProperties") is not False
        or contract.get("properties", {}).get("input_contract", {}).get("const")
        != INPUT_CONTRACT
        or contract.get("properties", {}).get("output_contract", {}).get("const")
        != OUTPUT_CONTRACT
        or contract.get("properties", {})
        .get("object_specific_gravity_parameter_count", {})
        .get("const")
        != 0
    ):
        raise ValueError("G4 galaxy prediction bundle contract changed")

    descriptor = sources["reviewed_evaluator_descriptor"]
    expected_descriptor = {
        "schema_version": DESCRIPTOR_SCHEMA,
        "evaluator_id": "reviewed-g4-candidate-galaxy-readiness-v1",
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "callback": CALLBACK,
        "artifact_path": bindings["reviewed_evaluator_source"]["path"],
        "artifact_sha256": bindings["reviewed_evaluator_source"]["file_sha256"],
        "prediction_bundle_contract_path": bindings["prediction_bundle_contract"][
            "path"
        ],
        "prediction_bundle_contract_file_sha256": bindings[
            "prediction_bundle_contract"
        ]["file_sha256"],
        "prediction_bundle_contract_content_sha256": bindings[
            "prediction_bundle_contract"
        ]["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    if descriptor != expected_descriptor:
        raise ValueError("reviewed G4 galaxy evaluator descriptor changed")
    descriptor_binding_sha256 = _sha(descriptor)
    registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    registration[DESCRIPTOR_FIELD] = descriptor_binding_sha256
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
    expected_missing = sorted(set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD})
    if current["missing_registration_hashes"] != expected_missing:
        raise ValueError("current G4 galaxy missing-registration list changed")
    shape = _synthetic_shape_fixture()
    covariance = _synthetic_covariance_fixture()
    readiness_body = {
        "descriptor_binding_sha256": descriptor_binding_sha256,
        "callback_source_sha256": bindings["reviewed_evaluator_source"][
            "file_sha256"
        ],
        "prediction_bundle_contract_content_sha256": bindings[
            "prediction_bundle_contract"
        ]["content_sha256"],
        "synthetic_shape_fixture_sha256": shape["content_sha256"],
        "synthetic_covariance_fixture_sha256": covariance["content_sha256"],
    }
    implementation_readiness = {
        **readiness_body,
        "content_sha256": _sha(readiness_body),
    }
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "candidate_dossier_sha256": dossiers[0]["content_sha256"],
        "formal_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "scalar_free_branch_sha256": scalar_free["content_sha256"],
        "PPN_prediction_sha256": ppn["content_sha256"],
        "galaxy_protocol_content_sha256": bindings["galaxy_protocol"][
            "content_sha256"
        ],
        "evidence_policy_content_sha256": bindings["evidence_policy"][
            "content_sha256"
        ],
        "implementation_readiness_sha256": implementation_readiness[
            "content_sha256"
        ],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "role": "generated_candidate",
            "data_eligibility": dict(ELIGIBILITY),
        },
        "source_bindings": bindings,
        "implementation_readiness": implementation_readiness,
        "synthetic_controls": {"shape": shape, "covariance": covariance},
        "current_evaluator_decision": current,
        "newly_filled_registration_fields": {
            DESCRIPTOR_FIELD: descriptor_binding_sha256,
            "reviewed_candidate_galaxy_evaluator_implementation_readiness_sha256": (
                implementation_readiness["content_sha256"]
            ),
        },
        "unfilled_prediction_data_registration_fields": expected_missing,
        "descriptor_implementation_ready": True,
        "prediction_bundle_registered": False,
        "candidate_use_authorized": False,
        "observational_authorization": False,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "tracking_target_values_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "decision": "blocked",
        "first_missing_premise": "registered_action_bound_galaxy_prediction_bundle",
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The exact formal action and its declared scalar-free GR/PPN branch are bound as "
            "theory provenance, not galaxy evidence. Shape and covariance plumbing pass only "
            "synthetic controls. No prediction bundle, source manifest, target, halo label, or "
            "redshift-derived distance is registered or opened."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
