"""Action-bound, data-sealed Solar held-out transfer registration for G2 candidates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-g2-solar-heldout-transfer-registration-config-1.0"
ARTIFACT_SCHEMA = "sigma-g2-solar-heldout-transfer-registration-1.0"
BUNDLE_SCHEMA = "sigma-g2-solar-action-bound-prediction-bundle-1.0"
EVALUATOR_SCHEMA = "sigma-g2-solar-heldout-transfer-evaluator-descriptor-1.0"
OUTPUT_CHANNELS = [
    "two_way_round_trip_light_time",
    "coherent_carrier_frequency_or_phase_ratio",
    "relative_angular_separation",
]
BASE_MISSING_FIELDS = [
    "candidate_specific_real_source_contract_sha256",
    "source_branch_domain_instantiation_sha256",
    "candidate_specific_evaluator_descriptor_sha256",
    "training_only_initial_state_sha256",
    "frozen_nuisance_likelihood_stopping_rule_sha256",
    "held_out_split_commitment_sha256",
    "action_bound_prediction_bundle_descriptor_sha256",
    "action_bound_prediction_bundle_file_sha256",
    "selected_primary_record_roots_sha256",
    "observation_opening_authorization_sha256",
]
FILLED_FIELDS = [
    "candidate_specific_real_source_contract_sha256",
    "candidate_specific_evaluator_descriptor_sha256",
    "training_only_initial_state_sha256",
    "frozen_nuisance_likelihood_stopping_rule_sha256",
    "action_bound_prediction_bundle_descriptor_sha256",
    "action_bound_prediction_bundle_file_sha256",
]
REMAINING_FIELDS = [
    "source_branch_domain_instantiation_sha256",
    "held_out_split_commitment_sha256",
    "selected_primary_record_roots_sha256",
    "observation_opening_authorization_sha256",
]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_bytes(value: dict[str, Any]) -> bytes:
    return (_canonical(value) + "\n").encode()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "adapter_source",
            "bindings",
            "bundle_output_directory",
            "registration_output_path",
            "observational_authorization",
            "data_eligibility",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("G2 held-out transfer config shape changed")
    if (
        config.get("observational_authorization") is not False
        or config.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("G2 held-out transfer config opened forbidden data")
    if set(config.get("bindings", {})) != {
        "g2_readiness",
        "solar_protocol",
        "source_registration",
        "evidence_policy",
        "protocol_audit",
    }:
        raise ValueError("G2 held-out transfer bindings changed")
    directory = config.get("bundle_output_directory")
    output = config.get("registration_output_path")
    if (
        not isinstance(directory, str)
        or not directory.startswith("runs/engine/")
        or not isinstance(output, str)
        or not output.startswith("runs/engine/")
    ):
        raise ValueError("G2 held-out transfer output path changed")


def _validate_inputs(config: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    loaded = {
        name: _load_bound(root, binding, name) for name, binding in config["bindings"].items()
    }
    readiness = loaded["g2_readiness"]
    if (
        readiness.get("schema_version") != "sigma-g2-scalable-solar-prediction-readiness-1.0"
        or readiness.get("candidate_count") != 2
        or readiness.get("decision_counts") != {"blocked": 2}
        or readiness.get("real_solar_bundle_count") != 0
        or readiness.get("observational_data_opened") is not False
        or readiness.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("G2 readiness boundary changed")
    for record in readiness.get("candidate_records", []):
        real = record.get("real_solar_readiness", {})
        if (
            real.get("missing_registration_fields") != BASE_MISSING_FIELDS
            or real.get("candidate_use_authorized") is not False
            or real.get("observational_inputs_opened_by_this_audit") is not False
        ):
            raise ValueError("G2 candidate registration boundary changed")
    protocol = loaded["solar_protocol"]
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or protocol.get("split_contract", {}).get("group_leakage_forbidden") is not True
        or protocol.get("scoring_contract", {}).get("object_specific_gravity_parameters") != 0
    ):
        raise ValueError("Solar protocol seal changed")
    source = loaded["source_registration"]
    if (
        source.get("status") != "metadata_registered_data_sealed"
        or source.get("data_opened") is not False
        or source.get("candidate_use_authorized") is not False
        or source.get("readiness", {}).get("primary_files_downloaded") is not False
        or source.get("readiness", {}).get("dataset_ready") is not False
    ):
        raise ValueError("Solar source seal changed")
    policy = loaded["evidence_policy"]
    if (
        policy.get("status") != "frozen"
        or policy.get("unobserved_components", {}).get("default_status")
        != "prohibited_as_truth_or_rescue"
    ):
        raise ValueError("observational evidence policy changed")
    audit = loaded["protocol_audit"]
    if (
        audit.get("status") != "pass"
        or audit.get("observational_dataset_opened") is not False
        or audit.get("formula_search_authorized") is not False
    ):
        raise ValueError("Solar protocol audit seal changed")
    return loaded


def _quantity_class_contract(source: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    classification = source["record_classification"]
    return {
        "raw": {
            "allowed": True,
            "classes": classification["direct_signal_records"],
            "retention": "original bytes, PDS label, file SHA-256, and record offsets",
        },
        "calibrated": {
            "allowed": True,
            "classes": classification["calibrated_or_compressed_records"],
            "rule": "bind every value to raw inputs, transform, nuisance values, and covariance",
        },
        "derived": {
            "allowed": True,
            "classes": protocol["quantity_classes"]["derived"]["examples"],
            "rule": protocol["quantity_classes"]["derived"]["rule"],
        },
        "model_dependent": {
            "allowed_as_input_or_target": False,
            "classes": classification["model_dependent_records"]["examples"],
        },
        "latent": {
            "allowed_as_input_or_target": False,
            "classes": classification["latent_quantities"]["examples"],
        },
    }


def _candidate_contracts(
    record: dict[str, Any], source: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = record["candidate_id"]
    action_sha256 = record["action_sha256"]
    prediction_sha256 = record["scalar_free_prediction_certificate"]["content_sha256"]
    source_contract = {
        "candidate_id": candidate_id,
        "action_sha256": action_sha256,
        "authority": source["source"]["authority"],
        "dataset_id": source["source"]["dataset_id"],
        "metadata_status": "metadata_registered_data_sealed",
        "allowed_direct_record_classes": source["record_classification"]["direct_signal_records"],
        "quantity_classes": _quantity_class_contract(source, protocol),
        "source_branch_domain_instantiation_sha256": None,
        "selected_primary_record_roots_sha256": None,
        "target_values_accessed": False,
        "status": "contract_registered_real_source_values_sealed",
    }
    initial_state = {
        "candidate_id": candidate_id,
        "action_sha256": action_sha256,
        "fit_role": "training_sessions_only",
        "estimated_state": [
            "spacecraft and relevant-body state with covariance",
            "station clock and oscillator offsets",
            "non-gravity spacecraft force nuisance state",
        ],
        "validation_or_test_refit": False,
        "checkpoint_sha256": None,
        "freeze_rule": "freeze posterior and covariance before validation and immutable test",
        "status": "contract_frozen_checkpoint_missing",
    }
    nuisance_likelihood_stopping = {
        "candidate_id": candidate_id,
        "action_sha256": action_sha256,
        "nuisance": {
            "gravity_nuisances": [],
            "object_specific_gravity_parameter_count": 0,
            "allowed": [
                "clock and oscillator offsets or drifts",
                "station and Earth-orientation covariance",
                "troposphere, ionosphere, and plasma delays",
                "instrument phase or range offsets",
                "non-gravity spacecraft force terms frozen from training",
            ],
            "post_hoc_rescue": "forbidden",
        },
        "likelihood": {
            "space": "joint direct signal space",
            "channels": OUTPUT_CHANNELS,
            "covariance_required": True,
            "candidate_selection_data": "training_and_validation_only",
            "model_dependent_residual_targets": False,
        },
        "stopping_rule": {
            "maximum_candidate_actions": 1,
            "maximum_test_openings": 1,
            "test_failure_rescue_iterations": 0,
            "stop_on_parser_covariance_or_source_branch_failure": True,
        },
        "status": "frozen_before_real_data_opening",
    }
    split_template = {
        "unit": "tracking pass or observing session",
        "group_keys": source["future_split_contract"]["group_keys"],
        "roles": [
            "state_and_calibration_training",
            "formula_selection_validation",
            "untouched_target_blind_test",
        ],
        "group_leakage_forbidden": True,
        "selection_before_target_access": True,
        "selected_session_ids": [],
        "commitment_sha256": None,
        "status": "template_frozen_actual_commitment_missing",
    }
    evaluator = {
        "schema_version": EVALUATOR_SCHEMA,
        "evaluator_id": f"g2-solar-heldout-transfer-{candidate_id}",
        "candidate_id": candidate_id,
        "action_sha256": action_sha256,
        "analytic_prediction_certificate_sha256": prediction_sha256,
        "output_channels": OUTPUT_CHANNELS,
        "required_registration_fields": BASE_MISSING_FIELDS,
        "remaining_external_fields": REMAINING_FIELDS,
        "decision_until_external_fields_and_authorization": "blocked",
        "observational_pass_possible_in_this_artifact": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    return {
        "source_contract": {**source_contract, "content_sha256": _sha(source_contract)},
        "training_only_initial_state": {
            **initial_state,
            "content_sha256": _sha(initial_state),
        },
        "nuisance_likelihood_stopping": {
            **nuisance_likelihood_stopping,
            "content_sha256": _sha(nuisance_likelihood_stopping),
        },
        "split_commitment_template": {
            **split_template,
            "content_sha256": _sha(split_template),
        },
        "evaluator_descriptor": {**evaluator, "content_sha256": _sha(evaluator)},
    }


def _prediction_bundle(record: dict[str, Any], contracts: dict[str, Any]) -> dict[str, Any]:
    prediction = record["scalar_free_prediction_certificate"]
    descriptor = {
        "candidate_id": record["candidate_id"],
        "action_sha256": record["action_sha256"],
        "formal_record_sha256": record["provenance"]["formal_record_sha256"],
        "analytic_prediction_certificate_sha256": prediction["content_sha256"],
        "source_contract_sha256": contracts["source_contract"]["content_sha256"],
        "initial_state_contract_sha256": contracts["training_only_initial_state"]["content_sha256"],
        "nuisance_likelihood_stopping_sha256": contracts["nuisance_likelihood_stopping"][
            "content_sha256"
        ],
        "evaluator_descriptor_sha256": contracts["evaluator_descriptor"]["content_sha256"],
        "output_channels": OUTPUT_CHANNELS,
        "universal_parameter_count": 0,
        "object_specific_gravity_parameter_count": 0,
        "source_branch_domain_instantiation_sha256": None,
        "held_out_split_commitment_sha256": None,
        "selected_primary_record_roots_sha256": None,
        "observational_opening_authorization_sha256": None,
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
    }
    body = {
        "schema_version": BUNDLE_SCHEMA,
        "candidate_id": record["candidate_id"],
        "action_sha256": record["action_sha256"],
        "descriptor": {**descriptor, "content_sha256": _sha(descriptor)},
        "analytic_forward_model": {
            "branch": prediction["background_and_boundary"],
            "Newtonian_prediction": prediction["Newtonian_prediction"],
            "PPN_prediction": prediction["PPN_prediction"],
            "vacuum_exterior": prediction["vacuum_exterior"],
            "known_answer_formulas": prediction["known_answer_formulas"],
            "role": "action_bound_analytic_forward_model_not_observational_evidence",
        },
        "real_source_prediction_generated": False,
        "held_out_targets_opened": False,
        "decision": "blocked",
        "blocker": "source_branch_domain_split_roots_and_authorization_missing",
    }
    return {**body, "content_sha256": _sha(body)}


def evaluate_registration(registration: dict[str, Any]) -> dict[str, Any]:
    """Validate the preregistration boundary; this callback cannot open observations."""

    if registration.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("held-out transfer eligibility changed")
    if registration.get("filled_registration_fields") != FILLED_FIELDS:
        raise ValueError("held-out transfer filled field set changed")
    values = registration.get("registration_hashes", {})
    if set(values) != set(BASE_MISSING_FIELDS):
        raise ValueError("held-out transfer registration field set changed")
    for field in FILLED_FIELDS:
        if not isinstance(values[field], str) or _SHA256.fullmatch(values[field]) is None:
            raise ValueError(f"held-out transfer invalid filled hash: {field}")
    if any(values[field] is not None for field in REMAINING_FIELDS):
        raise ValueError("external held-out fields cannot be filled by sealed preregistration")
    return {
        "candidate_id": registration["candidate_id"],
        "action_sha256": registration["action_sha256"],
        "decision": "blocked",
        "blocker": "source_branch_domain_split_roots_and_authorization_missing",
        "filled_registration_field_count": len(FILLED_FIELDS),
        "remaining_registration_fields": REMAINING_FIELDS,
        "candidate_use_authorized": False,
        "observational_data_opened": False,
        "real_data_pass": False,
        "data_eligibility": dict(ELIGIBILITY),
    }


def build_g2_solar_heldout_transfer_registration(
    config: dict[str, Any], project_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build action-bound bundles and registrations without opening primary records."""

    _validate_config(config)
    root = Path(project_root).resolve()
    source_path = (root / config["adapter_source"]["path"]).resolve()
    if _file_sha(source_path) != config["adapter_source"]["file_sha256"]:
        raise ValueError("G2 held-out transfer adapter source hash mismatch")
    loaded = _validate_inputs(config, root)
    readiness = loaded["g2_readiness"]
    source = loaded["source_registration"]
    protocol = loaded["solar_protocol"]
    candidate_material: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for record in readiness["candidate_records"]:
        contracts = _candidate_contracts(record, source, protocol)
        bundle = _prediction_bundle(record, contracts)
        candidate_material.append((record, contracts, bundle))
    bundle_files: dict[str, dict[str, Any]] = {}
    bundle_specs: dict[str, dict[str, str]] = {}
    for _, _, bundle in candidate_material:
        path = f"{config['bundle_output_directory']}/{bundle['candidate_id']}.json"
        bundle_files[path] = bundle
        bundle_specs[bundle["candidate_id"]] = {
            "path": path,
            "file_sha256": hashlib.sha256(_file_bytes(bundle)).hexdigest(),
            "content_sha256": bundle["content_sha256"],
        }
    registrations = []
    for record, contracts, bundle in candidate_material:
        hashes: dict[str, str | None] = {field: None for field in BASE_MISSING_FIELDS}
        hashes.update(
            {
                "candidate_specific_real_source_contract_sha256": contracts["source_contract"][
                    "content_sha256"
                ],
                "candidate_specific_evaluator_descriptor_sha256": contracts["evaluator_descriptor"][
                    "content_sha256"
                ],
                "training_only_initial_state_sha256": contracts["training_only_initial_state"][
                    "content_sha256"
                ],
                "frozen_nuisance_likelihood_stopping_rule_sha256": contracts[
                    "nuisance_likelihood_stopping"
                ]["content_sha256"],
                "action_bound_prediction_bundle_descriptor_sha256": bundle["descriptor"][
                    "content_sha256"
                ],
                "action_bound_prediction_bundle_file_sha256": bundle_specs[record["candidate_id"]][
                    "file_sha256"
                ],
            }
        )
        registration = {
            "candidate_id": record["candidate_id"],
            "action_sha256": record["action_sha256"],
            "base_missing_registration_fields": BASE_MISSING_FIELDS,
            "filled_registration_fields": FILLED_FIELDS,
            "remaining_registration_fields": REMAINING_FIELDS,
            "registration_hashes": hashes,
            "contracts": contracts,
            "prediction_bundle": bundle_specs[record["candidate_id"]],
            "candidate_use_authorized": False,
            "observational_data_opened": False,
            "real_data_pass": False,
            "data_eligibility": dict(ELIGIBILITY),
        }
        registration["evaluator_result"] = evaluate_registration(registration)
        registration["content_sha256"] = _sha(registration)
        registrations.append(registration)
    artifact_body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": config["bindings"],
        "prediction_bundles": [bundle_specs[key] for key in sorted(bundle_specs)],
        "candidate_count": len(registrations),
        "registration_advance_per_candidate": {
            "before_missing_field_count": len(BASE_MISSING_FIELDS),
            "filled_field_count": len(FILLED_FIELDS),
            "after_missing_field_count": len(REMAINING_FIELDS),
            "filled_fields": FILLED_FIELDS,
            "remaining_fields": REMAINING_FIELDS,
        },
        "candidate_registrations": registrations,
        "synthetic_controls": {
            "analytic_identity": {
                "candidate_count": len(registrations),
                "all_gamma_beta_equal_one": all(
                    bundle["analytic_forward_model"]["PPN_prediction"]["gamma"] == "1"
                    and bundle["analytic_forward_model"]["PPN_prediction"]["beta"] == "1"
                    for _, _, bundle in candidate_material
                ),
                "role": "symbolic_known_answer_control_not_real_data",
                "decision": "pass",
            },
            "fail_closed_evaluator": {
                "blocked_count": sum(
                    item["evaluator_result"]["decision"] == "blocked" for item in registrations
                ),
                "real_data_pass_count": 0,
                "role": "synthetic_registration_control",
                "decision": "pass",
            },
        },
        "first_missing_premise": (
            "candidate_specific_real_source_branch_domain_instantiation_and_"
            "metadata_only_session_split_commitment"
        ),
        "decision_counts": {"blocked": len(registrations)},
        "candidate_use_authorized": False,
        "observational_authorization": False,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "held_out_target_access_count": 0,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "real_data_pass_count": 0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Six theory/protocol-side registrations are now action-bound for each G2 candidate. "
            "Real source-domain facts, selected-session commitment, primary roots, and separate "
            "opening authorization remain absent; therefore no observation or real-data pass exists."
        ),
    }
    artifact = {**artifact_body, "content_sha256": _sha(artifact_body)}
    return bundle_files, artifact


def write_g2_solar_heldout_transfer_registration(
    config_path: str | Path, project_root: str | Path
) -> tuple[list[Path], Path]:
    root = Path(project_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    bundle_files, artifact = build_g2_solar_heldout_transfer_registration(config, root)
    artifact_path = root / config["registration_output_path"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_paths = []
    for relative, bundle in bundle_files.items():
        bundle_path = root / relative
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_bytes(_file_bytes(bundle))
        bundle_paths.append(bundle_path)
    artifact_path.write_bytes(_file_bytes(artifact))
    return bundle_paths, artifact_path


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    config_file = project / "configs" / "g2_solar_heldout_transfer_registration.json"
    bundle_outputs, registration_output = write_g2_solar_heldout_transfer_registration(
        config_file, project
    )
    for output in [*bundle_outputs, registration_output]:
        print(output)
