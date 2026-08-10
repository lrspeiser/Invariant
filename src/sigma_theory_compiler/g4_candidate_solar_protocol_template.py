"""Frozen, data-sealed candidate-use Solar protocol template for the G4 seed."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-g4-candidate-solar-protocol-template-config-1.0"
ARTIFACT_SCHEMA = "sigma-g4-candidate-solar-protocol-template-1.0"
DESCRIPTOR_SCHEMA = "sigma-grammar-v3-g4-solar-prediction-bundle-descriptor-1.0"
PREDICTION_BUNDLE_SCHEMA = "sigma-candidate-solar-prediction-bundle-1.0"
REGISTRY_SCHEMA = "sigma-g4-candidate-solar-template-registry-1.0"

CANDIDATE = {
    "candidate_id": "G3-f9c598b70a77ea54009d8f18",
    "role": "generated_candidate",
    "family_id": "CONFORMAL_G4_PHI_SCALAR_TENSOR",
    "action_sha256": "6ddd6502d110ead90ff494a6569213ec2e61a0b046dfa86344bb1980df6abc90",
    "seed_lineage_sha256": "f9c598b70a77ea54009d8f18723ff6c54974c8aceab680d2fc95f513a33b2aa7",
}

OUTPUT_CHANNELS = [
    "two_way_round_trip_light_time",
    "coherent_carrier_frequency_or_phase_ratio",
    "relative_angular_separation",
]

REMAINING_FIELDS = [
    "registered_real_source_interval_instantiation_certificate_sha256",
    "selected_primary_file_root_sha256",
    "selected_PDS_label_and_calibration_file_root_sha256",
    "verified_ATDF_TDF_parser_sha256",
    "verified_RSR_parser_sha256",
    "raw_to_calibrated_transform_and_covariance_implementation_sha256",
    "tracking_session_split_commitment_sha256",
    "training_only_initial_state_checkpoint_sha256",
    "reviewed_candidate_solar_evaluator_descriptor_sha256",
]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return value


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"candidate Solar template binding changed: {binding['path']}")
    value = _load(path)
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if _sha(body) != expected or (
            "content_sha256" in value and value["content_sha256"] != expected
        ):
            raise ValueError(f"candidate Solar template content changed: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "campaign_id",
        "candidate",
        "bindings",
        "source_selection",
        "budget",
        "data_eligibility",
        "observational_authorization",
    } or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("candidate Solar protocol template config is invalid")
    if config.get("candidate") != CANDIDATE:
        raise ValueError("candidate Solar protocol template identity changed")
    if (
        config.get("data_eligibility") != ELIGIBILITY
        or config.get("observational_authorization") is not False
    ):
        raise ValueError("candidate Solar protocol template opened forbidden data")
    bindings = config.get("bindings", {})
    if set(bindings) != {
        "prediction_audit",
        "source_class_uniqueness_audit",
        "promotion_service_status",
        "descriptor_contract",
        "solar_protocol",
        "solar_protocol_audit",
        "source_registration",
        "evidence_policy",
    }:
        raise ValueError("candidate Solar protocol template bindings are incomplete")
    selection = config.get("source_selection", {})
    if (
        selection.get("authority") != "NASA Planetary Data System"
        or selection.get("dataset_id") != "CO-SS-RSS-1-SCE1-V1.0"
        or selection.get("allowed_direct_record_classes")
        != [
            "ATDF/TDF closed-loop tracking records",
            "RSR open-loop receiver samples",
        ]
        or not 1 <= selection.get("maximum_primary_files", 0) <= 12
        or not 1 <= selection.get("maximum_primary_bytes", 0) <= 2 * 1024**3
        or selection.get("target_value_access_during_selection") is not False
        or selection.get("selected_primary_files") != []
        or selection.get("selected_primary_file_root_sha256") is not None
    ):
        raise ValueError("candidate Solar primary-file selection is not sealed")
    budget = config.get("budget", {})
    if budget != {
        "maximum_tasks": 12,
        "maximum_wall_seconds": 86400,
        "maximum_download_bytes": 2147483648,
        "maximum_checkpoint_bytes": 67108864,
        "maximum_paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("candidate Solar template budget changed")


def _validate_inputs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    bindings = config["bindings"]
    loaded = {name: _load_bound(root, item) for name, item in bindings.items()}
    audit = loaded["prediction_audit"]
    record = audit.get("candidate_records", [None])[0]
    if (
        len(audit.get("candidate_records", [])) != 1
        or not isinstance(record, dict)
        or record.get("seed_id") != CANDIDATE["candidate_id"]
        or record.get("action_sha256") != CANDIDATE["action_sha256"]
        or record.get("decision") != "blocked"
        or audit.get("gate_status_counts") != {"blocked": 2, "pass": 6}
        or audit.get("analytic_known_answer_bundle_count") != 1
        or audit.get("real_solar_bundle_count") != 0
        or audit.get("observational_data_opened") is not False
        or audit.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("candidate Solar prediction audit no longer has the reviewed boundary")
    uniqueness = loaded["source_class_uniqueness_audit"]
    uniqueness_records = uniqueness.get("candidate_records", [])
    if len(uniqueness_records) != 1:
        raise ValueError("candidate Solar source-class audit candidate set changed")
    uniqueness_record = uniqueness_records[0]
    if (
        uniqueness.get("schema_version")
        != "sigma-g4-source-class-scalar-uniqueness-audit-1.0"
        or uniqueness.get("decision_counts") != {"blocked": 1}
        or uniqueness.get("source_class_theorem_pass_count") != 1
        or uniqueness.get("real_source_instantiation_pass_count") != 0
        or uniqueness_record.get("seed_id") != CANDIDATE["candidate_id"]
        or uniqueness_record.get("action_sha256") != CANDIDATE["action_sha256"]
        or uniqueness_record.get("source_class_theorem_decision") != "pass"
        or uniqueness_record.get("decision") != "blocked"
        or uniqueness_record.get("first_missing_premise")
        != "registered_real_source_interval_certificate"
        or uniqueness_record.get("physics_side_real_source_blocker")
        != {
            "theorem_side": "closed_for_every_source_satisfying_the_explicit_class",
            "real_Sun_instantiation": "blocked_until_registered_facts_instantiate_the_class",
        }
        or uniqueness_record.get("provenance", {}).get("predecessor_content_sha256")
        != bindings["prediction_audit"]["content_sha256"]
        or uniqueness_record.get("provenance", {}).get("data_eligibility")
        != ELIGIBILITY
        or uniqueness.get("observational_data_opened") is not False
        or uniqueness.get("paid_llm_spend_usd") != 0.0
    ):
        raise ValueError("candidate Solar source-class theorem changed or overclaimed")
    service = loaded["promotion_service_status"]
    if (
        service.get("formal_pass_verified") is not True
        or service.get("prediction_bundle_descriptor_registered") is not False
        or service.get("reviewed_solar_evaluator_invoked") is not False
        or service.get("observational_data_opened") is not False
    ):
        raise ValueError("candidate Solar promotion service is no longer sealed")
    descriptor_contract = loaded["descriptor_contract"]
    if (
        descriptor_contract.get("properties", {})
        .get("schema_version", {})
        .get("const")
        != DESCRIPTOR_SCHEMA
        or descriptor_contract.get("additionalProperties") is not False
    ):
        raise ValueError("candidate Solar descriptor contract changed")
    protocol = loaded["solar_protocol"]
    if (
        protocol.get("status") != "sealed"
        or protocol.get("data_opened") is not False
        or protocol.get("scoring_contract", {}).get("object_specific_gravity_parameters")
        != 0
        or protocol.get("split_contract", {}).get("group_leakage_forbidden") is not True
    ):
        raise ValueError("candidate Solar observable protocol is not sealed")
    protocol_audit = loaded["solar_protocol_audit"]
    if (
        protocol_audit.get("status") != "pass"
        or protocol_audit.get("formula_search_authorized") is not False
        or protocol_audit.get("observational_dataset_opened") is not False
        or protocol_audit.get("object_specific_gravity_parameters") != 0
    ):
        raise ValueError("candidate Solar observable protocol audit changed")
    source = loaded["source_registration"]
    if (
        source.get("status") != "metadata_registered_data_sealed"
        or source.get("data_opened") is not False
        or source.get("candidate_use_authorized") is not False
        or source.get("source", {}).get("dataset_id")
        != config["source_selection"]["dataset_id"]
        or source.get("readiness", {}).get("dataset_ready") is not False
        or source.get("readiness", {}).get("primary_files_downloaded") is not False
        or source.get("readiness", {}).get("raw_parser_verified") is not False
    ):
        raise ValueError("candidate Solar source registration opened or changed")
    return loaded


def build_g4_candidate_solar_protocol_template(
    config: dict[str, Any], project_root: str | Path
) -> dict[str, Any]:
    """Build the frozen template without selecting or opening primary records."""

    _validate_config(config)
    root = Path(project_root).resolve()
    loaded = _validate_inputs(config, root)
    audit_record = loaded["prediction_audit"]["candidate_records"][0]
    uniqueness_record = loaded["source_class_uniqueness_audit"][
        "candidate_records"
    ][0]
    source = loaded["source_registration"]
    protocol = loaded["solar_protocol"]

    source_physics = {
        "analytic_scalar_free_branch_status": "pass",
        "analytic_scalar_free_branch_sha256": audit_record[
            "provenance"
        ]["scalar_free_branch_sha256"],
        "analytic_coupling_and_PPN_sha256": audit_record["provenance"][
            "coupling_PPN_sha256"
        ],
        "source_class_theorem": {
            "status": "pass",
            "source_class_sha256": uniqueness_record["provenance"][
                "source_class_sha256"
            ],
            "coercivity_sha256": uniqueness_record["provenance"][
                "coercivity_sha256"
            ],
            "global_coupling_sha256": uniqueness_record["provenance"][
                "global_coupling_sha256"
            ],
            "scope": "nonlinear static uniqueness and linear scalar stability for every source satisfying the exact registered interval class",
        },
        "real_source_interval_instantiation": {
            "required_fields": [
                item["id"]
                for item in uniqueness_record[
                    "minimal_real_source_instantiation_contract"
                ]["required_registered_facts"]
            ],
            "content_sha256": None,
            "status": "missing",
        },
        "candidate_use_status": "blocked_real_source_interval_instantiation_not_bound",
    }
    quantity_classes = {
        "raw": {
            "allowed": True,
            "classes": source["record_classification"]["direct_signal_records"],
            "retention": "retain original bytes, PDS label, file SHA-256, and record offsets",
        },
        "calibrated": {
            "allowed": True,
            "classes": source["record_classification"][
                "calibrated_or_compressed_records"
            ],
            "rule": "each value binds raw inputs, transformation version, nuisance values, and covariance",
        },
        "derived": {
            "allowed": True,
            "classes": protocol["quantity_classes"]["derived"]["examples"],
            "rule": protocol["quantity_classes"]["derived"]["rule"],
        },
        "model_dependent": {
            "allowed_as_input_or_target": False,
            "classes": source["record_classification"]["model_dependent_records"][
                "examples"
            ],
        },
        "latent": {
            "allowed_as_input_or_target": False,
            "classes": source["record_classification"]["latent_quantities"][
                "examples"
            ],
        },
    }
    parser_contract = {
        "ATDF_TDF": {
            "required": True,
            "implementation_sha256": None,
            "verification": "bit-level field offsets, signedness, scale, units, time tags, and endianness against each selected PDS label",
            "negative_controls": [
                "one-bit payload mutation must change decoded value or fail checksum",
                "label/record-layout mismatch must fail closed",
            ],
        },
        "RSR": {
            "required": True,
            "implementation_sha256": None,
            "verification": "bit-level header/sample layout, sample clock, channelization, units, and byte order against each selected PDS label",
            "negative_controls": [
                "truncated sample frame must fail",
                "inconsistent sample-count/header pair must fail",
            ],
        },
        "status": "blocked_unimplemented",
    }
    calibration_contract = {
        "inputs": [
            "station clock and oscillator calibration",
            "troposphere and ionosphere calibration",
            "solar-plasma propagation calibration",
            "station coordinates and Earth orientation",
            "antenna, receiver, and path-delay calibration",
        ],
        "raw_to_calibrated_transform_sha256": None,
        "calibration_file_root_sha256": None,
        "held_out_target_use": "forbidden",
        "freeze_rule": "model, priors, and transformation versions freeze before validation or test targets open",
        "status": "blocked_missing_selected_files_and_verified_transform",
    }
    initial_state_contract = {
        "fit_role": "training_sessions_only",
        "estimated_state": [
            "spacecraft and relevant-body state with covariance",
            "station clock/oscillator offsets",
            "non-gravity spacecraft force nuisance state",
        ],
        "validation_or_test_refit": False,
        "checkpoint_sha256": None,
        "freeze_rule": "posterior and covariance checkpoint before validation, then immutable before test",
        "status": "blocked_until_session_split_and_training_records_exist",
    }
    nuisance_contract = {
        "gravity_nuisances": [],
        "object_specific_gravity_parameter_count": 0,
        "allowed": [
            "clock and oscillator offsets/drifts",
            "station and Earth-orientation covariance",
            "troposphere, ionosphere, and plasma delays",
            "instrument phase/range offsets",
            "non-gravity spacecraft force terms fixed from training",
        ],
        "global_freeze": "all priors and sharing hierarchy frozen before test",
        "post_hoc_rescue": "forbidden",
    }
    covariance_contract = {
        "required_components": [
            "raw measurement covariance",
            "clock and oscillator covariance",
            "propagation calibration covariance",
            "station/geometry covariance",
            "training-state posterior covariance",
        ],
        "cross_session_rule": "retain declared shared calibration correlations; never assume independence silently",
        "implementation_sha256": None,
        "status": "specified_not_implemented",
    }
    likelihood_contract = {
        "space": "joint direct signal space",
        "channels": OUTPUT_CHANNELS,
        "family": "preregistered covariance-aware likelihood with per-channel units retained",
        "candidate_selection_data": "training and validation sessions only",
        "test_role": "single frozen evaluation; no action, parameter, nuisance, or stopping-rule update",
        "model_dependent_residual_targets": False,
        "implementation_sha256": None,
        "status": "specified_not_implemented",
    }
    split_contract = {
        "unit": "tracking pass or observing session",
        "group_keys": source["future_split_contract"]["group_keys"],
        "group_leakage_forbidden": True,
        "roles": [
            "state_and_calibration_training",
            "formula_selection_validation",
            "untouched_target_blind_test",
        ],
        "selection_before_target_access": True,
        "commitment_sha256": None,
        "status": "blocked_until_primary_files_are_selected_by_metadata",
    }
    stopping_contract = {
        "maximum_candidate_actions": 1,
        "maximum_test_openings": 1,
        "maximum_primary_files": config["budget"]["maximum_tasks"],
        "maximum_download_bytes": config["budget"]["maximum_download_bytes"],
        "test_failure_rescue_iterations": 0,
        "stop_conditions": [
            "budget exhausted",
            "parser or covariance verification failure",
            "source-class branch proof failure",
            "single preregistered held-out evaluation completed",
        ],
        "contract_sha256": None,
        "status": "frozen_text_pending_final_descriptor_hash",
    }
    restart_contract = {
        "checkpoint_unit": "one selected primary file or one complete tracking session",
        "checkpoint_contents": [
            "input file SHA-256 roots",
            "parser and calibration implementation hashes",
            "completed session IDs and roles",
            "initial-state checkpoint hash",
            "likelihood accumulator hash",
        ],
        "atomicity": "SQLite transaction or atomic manifest replacement after verification",
        "resume_rule": "reverify every bound file and implementation hash before consuming the next checkpoint",
        "replay_rule": "same immutable inputs, role commitment, and seed must reproduce the same per-session prediction/result hashes",
        "lease_recovery": "expired work returns to pending without changing session role or target visibility",
        "target_visibility_after_crash": "once any test target opens, resume cannot alter theory, nuisances, split, likelihood, or stopping rule",
    }
    frozen_contracts = {
        "source_physics": source_physics,
        "quantity_classes": quantity_classes,
        "parser_verification": parser_contract,
        "calibration": calibration_contract,
        "initial_state_inference": initial_state_contract,
        "nuisance_model": nuisance_contract,
        "covariance": covariance_contract,
        "likelihood": likelihood_contract,
        "session_split": split_contract,
        "stopping_rule": stopping_contract,
        "restart_and_replay": restart_contract,
    }
    contract_hashes = {
        name + "_sha256": _sha(contract)
        for name, contract in frozen_contracts.items()
    }
    prediction_bundle_projection = {
        "path": None,
        "file_sha256": None,
        "content_sha256": None,
        "schema_version": PREDICTION_BUNDLE_SCHEMA,
        "output_channels": OUTPUT_CHANNELS,
        "universal_parameter_count": 3,
        "object_specific_gravity_parameter_count": 0,
        "weak_field_solution_sha256": None,
        "state_estimation_contract_sha256": contract_hashes[
            "initial_state_inference_sha256"
        ],
        "instrument_calibration_contract_sha256": contract_hashes[
            "calibration_sha256"
        ],
        "covariance_contract_sha256": contract_hashes["covariance_sha256"],
        "likelihood_contract_sha256": contract_hashes["likelihood_sha256"],
        "split_commitment_sha256": None,
        "stopping_rule_sha256": contract_hashes["stopping_rule_sha256"],
    }
    descriptor_template = {
        "schema_version": DESCRIPTOR_SCHEMA,
        "descriptor_id": None,
        "candidate": CANDIDATE,
        "formal_pass_binding": loaded["promotion_service_status"][
            "formal_pass_binding"
        ],
        "prediction_audit_binding": loaded["promotion_service_status"][
            "reviewed_prediction_audit_binding"
        ],
        "solar_protocol_binding": {
            "path": config["bindings"]["solar_protocol"]["path"],
            "file_sha256": config["bindings"]["solar_protocol"]["file_sha256"],
            "content_sha256": _sha(loaded["solar_protocol"]),
        },
        "prediction_bundle": prediction_bundle_projection,
        "reviewed_evaluator": {
            "descriptor_path": None,
            "descriptor_file_sha256": None,
            "evaluator_binding_sha256": None,
            "callback": None,
        },
        "observational_opening": {
            "authorized": False,
            "requires_independent_dataset_manifest_audit": True,
            "requires_preregistered_session_split": True,
        },
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "status": "frozen_template_unregistered_ineligible",
        "candidate": CANDIDATE,
        "source_registration": {
            "authority": source["source"]["authority"],
            "dataset_id": source["source"]["dataset_id"],
            "metadata_status": source["status"],
            "remote_catalog_fingerprint_count": len(
                source["remote_catalog_fingerprints"]
            ),
            "primary_files_selected": 0,
            "primary_files_downloaded": 0,
            "target_values_accessed": False,
        },
        "direct_signal_channels": OUTPUT_CHANNELS,
        "frozen_contracts": frozen_contracts,
        "frozen_contract_hashes": contract_hashes,
        "descriptor_contract_binding": {
            **config["bindings"]["descriptor_contract"],
            "schema_version": DESCRIPTOR_SCHEMA,
        },
        "descriptor_template": descriptor_template,
        "descriptor_shape_status": "all_required_fields_present",
        "descriptor_registration_status": "blocked_required_values_unset",
        "remaining_registration_fields": REMAINING_FIELDS,
        "remaining_registration_field_count": len(REMAINING_FIELDS),
        "budget": config["budget"],
        "observational_authorization": False,
        "candidate_use_authorized": False,
        "formula_search_authorized": False,
        "observational_data_opened": False,
        "data_eligibility": ELIGIBILITY,
        "paid_llm_spend_usd": 0.0,
        "interpretation": (
            "This freezes how a candidate-specific Solar test must be built and resumed. It is "
            "not a registrable descriptor: real-source branch physics, selected primary-file "
            "hashes, verified parsers/calibration, a session split, and training-state checkpoint "
            "remain absent, so no target or evaluator may open."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


class G4CandidateSolarTemplateRegistry:
    """Idempotently checkpoint the immutable, still-ineligible template."""

    def __init__(self, database: str | Path, maximum_bytes: int = 4 * 1024**2) -> None:
        self.database = Path(database).resolve()
        if "campaign-v1-live.sqlite" in str(self.database).lower():
            raise ValueError("refusing to use the live campaign watchdog database")
        if not 4096 <= maximum_bytes <= 4 * 1024**2:
            raise ValueError("template registry disk budget is invalid")
        self.maximum_bytes = maximum_bytes
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS template_checkpoint ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                "schema_version TEXT NOT NULL,content_sha256 TEXT NOT NULL,"
                "template_json TEXT NOT NULL,state TEXT NOT NULL)"
            )
        self._enforce_budget()

    def _enforce_budget(self) -> int:
        consumed = sum(
            path.stat().st_size
            for path in (
                self.database,
                Path(str(self.database) + "-wal"),
                Path(str(self.database) + "-shm"),
            )
            if path.is_file()
        )
        if consumed > self.maximum_bytes:
            raise RuntimeError("candidate Solar template registry disk budget exhausted")
        return consumed

    def checkpoint(self, template: dict[str, Any]) -> dict[str, Any]:
        body = {key: item for key, item in template.items() if key != "content_sha256"}
        if (
            template.get("schema_version") != ARTIFACT_SCHEMA
            or template.get("content_sha256") != _sha(body)
            or template.get("status") != "frozen_template_unregistered_ineligible"
            or template.get("candidate_use_authorized") is not False
            or template.get("observational_data_opened") is not False
            or template.get("descriptor_registration_status")
            != "blocked_required_values_unset"
        ):
            raise ValueError("candidate Solar template checkpoint is not fail-closed")
        values = {
            "singleton": 1,
            "schema_version": REGISTRY_SCHEMA,
            "content_sha256": template["content_sha256"],
            "template_json": _canonical(template),
            "state": "unregistered_ineligible",
        }
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM template_checkpoint WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO template_checkpoint VALUES (1,?,?,?,?)",
                    tuple(values[key] for key in values if key != "singleton"),
                )
                replay = False
            elif dict(row) == values:
                replay = True
            else:
                raise ValueError("candidate Solar template replay changed")
        consumed = self._enforce_budget()
        result = {
            "schema_version": REGISTRY_SCHEMA,
            "state": "unregistered_ineligible",
            "content_sha256": template["content_sha256"],
            "replay": replay,
            "checkpoint_count": 1,
            "database_bytes": consumed,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return result

    def status(self) -> dict[str, Any]:
        with sqlite3.connect(self.database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM template_checkpoint WHERE singleton=1"
            ).fetchone()
        if row is None:
            return {
                "schema_version": REGISTRY_SCHEMA,
                "state": "empty",
                "checkpoint_count": 0,
                "observational_data_opened": False,
            }
        template = json.loads(row["template_json"])
        if template.get("content_sha256") != row["content_sha256"]:
            raise ValueError("candidate Solar template registry payload changed")
        return {
            "schema_version": REGISTRY_SCHEMA,
            "state": row["state"],
            "checkpoint_count": 1,
            "content_sha256": row["content_sha256"],
            "descriptor_registration_status": template[
                "descriptor_registration_status"
            ],
            "remaining_registration_field_count": template[
                "remaining_registration_field_count"
            ],
            "observational_data_opened": False,
            "database_bytes": self._enforce_budget(),
        }
