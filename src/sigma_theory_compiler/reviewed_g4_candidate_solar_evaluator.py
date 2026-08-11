from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .promotion_orchestrator import ELIGIBILITY
from .solar_direct_signal_calibration_readiness import calibrate_direct_signals
from .solar_promotion_evaluator import (
    GR_SOLAR_BUNDLE,
    bundle_binding,
    solar_known_answer_evaluator,
)

SCHEMA_VERSION = "sigma-reviewed-g4-candidate-solar-evaluator-readiness-1.0"
EVALUATOR_DESCRIPTOR_SCHEMA = "sigma-candidate-solar-evaluator-descriptor-1.0"
CANDIDATE_ID = "G3-f9c598b70a77ea54009d8f18"
ACTION_SHA256 = "6ddd6502d110ead90ff494a6569213ec2e61a0b046dfa86344bb1980df6abc90"
OUTPUT_CHANNELS = [
    "two_way_round_trip_light_time",
    "coherent_carrier_frequency_or_phase_ratio",
    "relative_angular_separation",
]
PARSER_HASHES = {
    "verified_ATDF_TDF_parser_sha256": (
        "7251db8d1e8edf33f2d66876f96e1392a8e79fde73839769075ccf1e8c736c20"
    ),
    "verified_RSR_parser_sha256": (
        "ddae01e336e1f19902e4552aaef36ce23b5abb5717cf776905e2f00d078180fe"
    ),
}
CALIBRATION_IMPLEMENTATION_SHA256 = (
    "b937071e25547df52ffac9f99b7879f171bf6901c08ef1cc1d6cd8fdfc38d27e"
)
REQUIRED_REGISTRATION_HASHES = (
    "registered_real_source_interval_instantiation_certificate_sha256",
    "registered_trace_tail_profile_certificate_sha256",
    "prediction_bundle_file_sha256",
    "prediction_bundle_content_sha256",
    "weak_field_solution_sha256",
    "state_estimation_contract_sha256",
    "selected_primary_file_root_sha256",
    "selected_PDS_label_and_calibration_file_root_sha256",
    "verified_ATDF_TDF_parser_sha256",
    "verified_RSR_parser_sha256",
    "raw_to_calibrated_transform_and_covariance_implementation_sha256",
    "covariance_contract_sha256",
    "likelihood_contract_sha256",
    "tracking_session_split_commitment_sha256",
    "training_only_initial_state_checkpoint_sha256",
    "stopping_rule_sha256",
    "reviewed_candidate_solar_evaluator_descriptor_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    path = root / descriptor["path"]
    if not path.is_file() or _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound evaluator-readiness file changed: {descriptor['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = descriptor.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or (
            "content_sha256" in value and value["content_sha256"] != expected
        ):
            raise ValueError(f"bound evaluator-readiness content changed: {descriptor['path']}")
    return value


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def _candidate_record(source: dict[str, Any], candidate_id: str = CANDIDATE_ID) -> dict[str, Any]:
    records = [
        item
        for item in source.get("candidate_records", [])
        if item.get("seed_id") == candidate_id
    ]
    if len(records) != 1:
        raise ValueError("candidate-specific Solar record is not unique")
    return records[0]


def _validate_future_registration(
    registration: dict[str, Any], context: dict[str, Any]
) -> None:
    if set(registration) != set(REQUIRED_REGISTRATION_HASHES):
        raise ValueError("future Solar registration field set changed")
    for name, value in registration.items():
        _require_hash(value, name)
    for name, expected in PARSER_HASHES.items():
        if registration[name] != expected:
            raise ValueError(f"future Solar registration changed {name}")
    if (
        registration[
            "raw_to_calibrated_transform_and_covariance_implementation_sha256"
        ]
        != CALIBRATION_IMPLEMENTATION_SHA256
    ):
        raise ValueError("future Solar registration changed calibration implementation")

    bundle = context.get("prediction_bundle")
    if not isinstance(bundle, dict):
        raise TypeError("fully registered Solar context lacks prediction bundle")
    required_bundle = {
        "candidate_id",
        "action_sha256",
        "path",
        "file_sha256",
        "content_sha256",
        "schema_version",
        "output_channels",
        "universal_parameter_count",
        "object_specific_gravity_parameter_count",
        "weak_field_solution_sha256",
        "state_estimation_contract_sha256",
        "instrument_calibration_contract_sha256",
        "covariance_contract_sha256",
        "likelihood_contract_sha256",
        "split_commitment_sha256",
        "stopping_rule_sha256",
        "data_eligibility",
        "observational_data_opened",
    }
    if (
        set(bundle) != required_bundle
        or bundle["candidate_id"] != CANDIDATE_ID
        or bundle["action_sha256"] != ACTION_SHA256
        or not isinstance(bundle["path"], str)
        or not bundle["path"].startswith("runs/engine/")
        or bundle["schema_version"] != "sigma-candidate-solar-prediction-bundle-1.0"
        or bundle["output_channels"] != OUTPUT_CHANNELS
        or bundle["universal_parameter_count"] != 0
        or bundle["object_specific_gravity_parameter_count"] != 0
        or bundle["data_eligibility"] != ELIGIBILITY
        or bundle["observational_data_opened"] is not False
    ):
        raise ValueError("future Solar prediction bundle violates the reviewed contract")
    bundle_matches = {
        "prediction_bundle_file_sha256": "file_sha256",
        "prediction_bundle_content_sha256": "content_sha256",
        "weak_field_solution_sha256": "weak_field_solution_sha256",
        "state_estimation_contract_sha256": "state_estimation_contract_sha256",
        "raw_to_calibrated_transform_and_covariance_implementation_sha256": (
            "instrument_calibration_contract_sha256"
        ),
        "covariance_contract_sha256": "covariance_contract_sha256",
        "likelihood_contract_sha256": "likelihood_contract_sha256",
        "tracking_session_split_commitment_sha256": "split_commitment_sha256",
        "stopping_rule_sha256": "stopping_rule_sha256",
    }
    for registration_name, bundle_name in bundle_matches.items():
        if registration[registration_name] != bundle[bundle_name]:
            raise ValueError(f"future Solar bundle hash mismatch: {registration_name}")

    source = context.get("real_source_certificate")
    if not isinstance(source, dict) or source != {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "decision": "pass",
        "interval_certificate_sha256": registration[
            "registered_real_source_interval_instantiation_certificate_sha256"
        ],
        "tail_profile_certificate_sha256": registration[
            "registered_trace_tail_profile_certificate_sha256"
        ],
        "observational_data_opened": False,
    }:
        raise ValueError("future Solar real-source certificate violates the reviewed contract")


def reviewed_g4_candidate_solar_evaluator(
    candidate: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Validate readiness only; never open records or return an observational pass."""

    if (
        candidate.get("candidate_id") != CANDIDATE_ID
        or candidate.get("action_sha256") != ACTION_SHA256
        or candidate.get("role") != "generated_candidate"
        or candidate.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("reviewed G4 Solar evaluator candidate identity changed")
    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("reviewed G4 Solar evaluator context eligibility changed")
    if context.get("observational_opening_authorized") is not False:
        raise ValueError("reviewed G4 Solar evaluator cannot authorize observation opening")

    registration = context.get("registration_hashes")
    if registration is None:
        registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    if not isinstance(registration, dict):
        raise TypeError("Solar registration hashes must be a mapping")
    missing = sorted(
        name
        for name in REQUIRED_REGISTRATION_HASHES
        if registration.get(name) is None
    )
    present = {name: value for name, value in registration.items() if value is not None}
    unknown = sorted(set(registration) - set(REQUIRED_REGISTRATION_HASHES))
    if unknown:
        raise ValueError(f"unknown Solar registration hashes: {unknown}")
    for name, value in present.items():
        _require_hash(value, name)
    if missing:
        return {
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "decision": "blocked",
            "blocker": "missing_fully_registered_real_source_and_prediction_bundle",
            "missing_registration_hashes": missing,
            "filled_registration_hash_count": len(present),
            "observational_opening_authorized": False,
            "observational_data_opened": False,
            "data_eligibility": dict(ELIGIBILITY),
        }
    _validate_future_registration(registration, context)
    return {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "decision": "blocked",
        "blocker": "separate_observational_opening_authorization_required",
        "readiness": "fully_registered_bundle_validated",
        "missing_registration_hashes": [],
        "filled_registration_hash_count": len(registration),
        "observational_opening_authorized": False,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
    }


def _synthetic_gr_fixture() -> dict[str, Any]:
    candidate = {
        "candidate_id": GR_SOLAR_BUNDLE["candidate_id"],
        "ordinal": 0,
        "correction_expression": "GR",
        "solar_control_provenance": {
            "bundle_id": GR_SOLAR_BUNDLE["bundle_id"],
            "bundle_binding_sha256": bundle_binding(GR_SOLAR_BUNDLE),
            "input_action_sha256": GR_SOLAR_BUNDLE["input_action_sha256"],
        },
        "data_eligibility": dict(ELIGIBILITY),
    }
    context = {
        "stage_name": "reviewed_g4_solar_evaluator_synthetic_fixture",
        "category": "calibration_only",
        "attempt": 1,
        "input_lineage_sha256": "a" * 64,
        "data_eligibility": dict(ELIGIBILITY),
    }
    result = solar_known_answer_evaluator(candidate, context)
    if result.get("decision") != "pass" or set(result.get("golden_statuses", {}).values()) != {
        "pass"
    }:
        raise ValueError("synthetic GR Solar fixture failed")
    body = {
        "role": "calibration_only_not_candidate_evidence",
        "decision": "pass",
        "golden_statuses": result["golden_statuses"],
        "input_action_sha256": result["input_action_sha256"],
        "observational_data_opened": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _synthetic_covariance_fixture() -> dict[str, Any]:
    tdf = {"raw_round_trip_light_time_s": 0.12, "utc_epoch_seconds": 1_025.0}
    rsr = {
        "carrier_phase_rad": 0.25,
        "measurement_covariance_matrix": [[1.0e-8, 2.0e-7], [2.0e-7, 4.0e-4]],
        "observation_duration_s": 0.19,
        "residual_frequency_hz": 5.0,
        "utc_epoch_seconds": 1_025.1,
    }
    labels = (
        "clock_offset_s",
        "instrument_phase_offset_rad",
        "ionosphere_delay_s",
        "oscillator_fractional_error",
        "propagation_frequency_shift_hz",
        "solar_plasma_delay_s",
        "station_geometry_delay_s",
        "station_geometry_frequency_shift_hz",
        "station_path_delay_s",
        "troposphere_delay_s",
    )
    nuisance = {label: 0.0 for label in labels}
    standard = np.array(
        [1e-10, 1e-4, 2e-10, 1e-13, 2e-3, 3e-10, 2e-10, 2e-3, 1e-10, 3e-10]
    )
    shared = np.array(
        [5e-11, 0.0, 8e-11, 5e-14, 5e-4, 9e-11, 7e-11, 3e-4, 4e-11, 8e-11]
    )
    covariance = np.diag(standard**2) + np.outer(shared, shared)
    result = calibrate_direct_signals(
        tdf,
        rsr,
        nuisance,
        covariance.tolist(),
        raw_tdf_variance_s2=4.0e-20,
        reference_frequency_hz=8.4e9,
        maximum_time_tag_separation_s=1.0,
    )
    output = np.asarray(result["covariance_matrix"], dtype=float)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(output)))
    if (
        minimum_eigenvalue < -1.0e-8
        or output[0, 2] == 0.0
        or output[1, 2] == 0.0
        or result["provenance"]["primary_target_records_opened"] is not False
    ):
        raise ValueError("synthetic covariance propagation fixture failed")
    body = {
        "role": "synthetic_covariance_implementation_fixture",
        "decision": "pass",
        "covariance_shape": list(output.shape),
        "cross_channel_covariance_nonzero": True,
        "minimum_eigenvalue_lower_check": "greater_than_or_equal_to_negative_1e-8",
        "shared_calibration_correlations_retained": result["provenance"][
            "shared_calibration_correlations_retained"
        ],
        "primary_target_records_opened": False,
    }
    return {**body, "content_sha256": _sha(body)}


def build_reviewed_g4_solar_evaluator_readiness(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("reviewed evaluator readiness eligibility changed")
    if config.get("observational_authorization") is not False:
        raise ValueError("reviewed evaluator readiness opened observations")
    bindings = config["source_bindings"]
    source_only = {
        "parser_source",
        "calibration_source",
        "gr_evaluator_source",
        "reviewed_evaluator_source",
    }
    for key in source_only:
        descriptor = bindings[key]
        path = root / descriptor["path"]
        if not path.is_file() or _file_sha(path) != descriptor["file_sha256"]:
            raise ValueError(f"bound evaluator source changed: {descriptor['path']}")
    sources = {
        key: _load_bound(root, value)
        for key, value in bindings.items()
        if key not in source_only
    }

    dossier = sources["candidate_dossier"]
    candidates = [
        item for item in dossier.get("dossiers", []) if item.get("dossier_id") == CANDIDATE_ID
    ]
    if len(candidates) != 1 or candidates[0].get("content_sha256") != bindings[
        "candidate_dossier"
    ]["candidate_dossier_sha256"]:
        raise ValueError("formal-pass candidate dossier changed")
    dossier_nodes = {item["node_id"]: item for item in candidates[0]["hierarchy_nodes"]}
    if (
        candidates[0].get("overall_status") != "blocked_after_formal_pass"
        or dossier_nodes["defining_covariant_action"].get("action_sha256")
        != ACTION_SHA256
        or dossier_nodes["exact_typed_operator_terms"].get("parameters")
        != {"G2": "X_phi", "G4": "1/2+(1/100)*phi^2", "phi_domain": "abs(phi)<=1"}
        or dossier_nodes["adm_dirac_obligation"]["status"] != "proven"
        or dossier_nodes["principal_symbol_obligation"]["status"] != "proven"
        or dossier_nodes["global_energy_obligation"]["status"] != "proven"
        or dossier_nodes["solar_prediction_obligation"]["status"] != "blocked"
    ):
        raise ValueError("candidate dossier formal/Solar hierarchy changed")
    protocol = sources["solar_protocol"]
    if protocol.get("status") != "sealed" or protocol.get("data_opened") is not False:
        raise ValueError("Solar protocol is not sealed")
    analytic = _candidate_record(sources["analytic_prediction"])
    source_class = _candidate_record(sources["source_class"])
    tail = sources["tail_theorem"]["candidate_records"][0]
    if (
        analytic.get("candidate_analytic_prediction_status")
        != "pass_on_declared_scalar_free_background"
        or analytic.get("action_sha256") != ACTION_SHA256
        or analytic.get("real_solar_admissibility", {}).get("admissible") is not False
        or source_class.get("source_class_theorem_decision") != "pass"
        or source_class.get("real_solar_bundle_admissible") is not False
        or tail.get("theorem_decision") != "pass"
        or tail.get("real_Sun_instantiation_decision") != "blocked"
    ):
        raise ValueError("analytic/source-class/tail readiness changed")

    parser = sources["parser_status"]
    calibration = sources["calibration_status"]
    if (
        parser.get("status") != "parser_ready_labels_selected_primary_records_sealed"
        or parser.get("filled_registration_fields") != PARSER_HASHES
        or parser.get("metadata_selection", {}).get("primary_record_access_count") != 0
        or calibration.get("status")
        != "calibration_implementation_ready_primary_records_sealed"
        or calibration.get("filled_registration_fields")
        != {
            **PARSER_HASHES,
            "raw_to_calibrated_transform_and_covariance_implementation_sha256": (
                CALIBRATION_IMPLEMENTATION_SHA256
            ),
        }
        or calibration.get("primary_record_access_count") != 0
    ):
        raise ValueError("parser or calibration implementation readiness changed")
    if (
        _file_sha(root / bindings["parser_source"]["path"])
        != parser["parser_source_sha256"]
        or _file_sha(root / bindings["calibration_source"]["path"])
        != calibration["implementation_source_sha256"]
    ):
        raise ValueError("parser or calibration source hash changed")

    descriptor = sources["reviewed_evaluator_descriptor"]
    source_path = root / descriptor["artifact_path"]
    if descriptor != {
        "schema_version": EVALUATOR_DESCRIPTOR_SCHEMA,
        "evaluator_id": "reviewed-g4-candidate-solar-readiness-v1",
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "callback": (
            "sigma_theory_compiler.reviewed_g4_candidate_solar_evaluator:"
            "reviewed_g4_candidate_solar_evaluator"
        ),
        "artifact_path": bindings["reviewed_evaluator_source"]["path"],
        "artifact_sha256": bindings["reviewed_evaluator_source"]["file_sha256"],
        "data_eligibility": ELIGIBILITY,
    } or _file_sha(source_path) != descriptor["artifact_sha256"]:
        raise ValueError("reviewed evaluator descriptor or callback source changed")

    descriptor_binding_sha256 = _sha(descriptor)
    current_registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
    current_registration["reviewed_candidate_solar_evaluator_descriptor_sha256"] = (
        descriptor_binding_sha256
    )
    blocked_context = {
        "data_eligibility": dict(ELIGIBILITY),
        "observational_opening_authorized": False,
        "registration_hashes": current_registration,
    }
    candidate = {
        "candidate_id": CANDIDATE_ID,
        "action_sha256": ACTION_SHA256,
        "role": "generated_candidate",
        "data_eligibility": dict(ELIGIBILITY),
    }
    current_decision = reviewed_g4_candidate_solar_evaluator(candidate, blocked_context)
    expected_missing = sorted(
        set(REQUIRED_REGISTRATION_HASHES)
        - {"reviewed_candidate_solar_evaluator_descriptor_sha256"}
    )
    if current_decision["missing_registration_hashes"] != expected_missing:
        raise ValueError("current registration fail-closed list changed")
    gr_fixture = _synthetic_gr_fixture()
    covariance_fixture = _synthetic_covariance_fixture()
    implementation_readiness_body = {
        "descriptor_file_sha256": bindings["reviewed_evaluator_descriptor"][
            "file_sha256"
        ],
        "descriptor_binding_sha256": descriptor_binding_sha256,
        "callback_source_sha256": bindings["reviewed_evaluator_source"]["file_sha256"],
        "parser_registration_hashes": PARSER_HASHES,
        "calibration_implementation_sha256": CALIBRATION_IMPLEMENTATION_SHA256,
        "synthetic_GR_fixture_sha256": gr_fixture["content_sha256"],
        "synthetic_covariance_fixture_sha256": covariance_fixture["content_sha256"],
    }
    implementation_readiness = {
        **implementation_readiness_body,
        "content_sha256": _sha(implementation_readiness_body),
    }
    provenance_body = {
        "action_sha256": ACTION_SHA256,
        "candidate_dossier_sha256": candidates[0]["content_sha256"],
        "analytic_prediction_sha256": analytic["provenance"]["binding_sha256"],
        "source_class_sha256": source_class["provenance"]["binding_sha256"],
        "tail_theorem_sha256": tail["provenance"]["binding_sha256"],
        "implementation_readiness_sha256": implementation_readiness["content_sha256"],
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "candidate": candidate,
        "source_bindings": bindings,
        "implementation_readiness": implementation_readiness,
        "synthetic_fixtures": {
            "GR_known_answer": gr_fixture,
            "covariance_propagation": covariance_fixture,
        },
        "current_evaluator_decision": current_decision,
        "newly_filled_registration_fields": {
            "reviewed_candidate_solar_evaluator_descriptor_sha256": (
                descriptor_binding_sha256
            ),
            "reviewed_candidate_solar_evaluator_implementation_readiness_sha256": (
                implementation_readiness["content_sha256"]
            ),
        },
        "already_filled_upstream_fields": {
            **PARSER_HASHES,
            "raw_to_calibrated_transform_and_covariance_implementation_sha256": (
                CALIBRATION_IMPLEMENTATION_SHA256
            ),
        },
        "unfilled_real_bundle_and_data_fields": current_decision[
            "missing_registration_hashes"
        ],
        "descriptor_implementation_ready": True,
        "real_source_prediction_bundle_registered": False,
        "candidate_use_authorized": False,
        "observational_authorization": False,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "tracking_target_values_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "decision": "blocked",
        "first_missing_premise": (
            "registered_real_source_interval_and_trace_tail_prediction_bundle"
        ),
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "interpretation": (
            "The reviewed candidate-specific callback and its descriptor are implemented and "
            "pass synthetic GR/covariance fixtures. Parser and calibration implementations are "
            "hash-bound, but no real-source certificate, prediction bundle, primary-record root, "
            "session split, or training checkpoint is filled. The callback remains blocked and "
            "cannot authorize observation opening."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
