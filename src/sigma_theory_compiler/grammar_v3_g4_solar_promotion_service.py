"""Fail-closed formal-pass to Solar promotion service for the grammar-v3 G4 seed."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-g4-solar-promotion-service-config-1.0"
DESCRIPTOR_SCHEMA = "sigma-grammar-v3-g4-solar-prediction-bundle-descriptor-1.0"
PREDICTION_BUNDLE_SCHEMA = "sigma-candidate-solar-prediction-bundle-1.0"
EVALUATOR_DESCRIPTOR_SCHEMA = "sigma-candidate-solar-evaluator-descriptor-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-g4-solar-promotion-status-1.0"
TASK_SCHEMA = "sigma-grammar-v3-g4-solar-promotion-task-1.0"

G4_TASK_TYPES = {
    "g4_global_lapse_invertibility",
    "g4_global_positive_energy",
}
OUTPUT_CHANNELS = [
    "two_way_round_trip_light_time",
    "coherent_carrier_frequency_or_phase_ratio",
    "relative_angular_separation",
]
MISSING_DESCRIPTOR = "missing_candidate_specific_solar_prediction_bundle_descriptor"

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS service_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  config_json TEXT NOT NULL,
  lifecycle TEXT NOT NULL,
  cycle_count INTEGER NOT NULL,
  stop_requested INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS promotion_work (
  task_id TEXT PRIMARY KEY,
  lineage_sha256 TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role='generated_candidate'),
  state TEXT NOT NULL,
  blocker TEXT,
  descriptor_binding_sha256 TEXT,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS service_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _content_body(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "content_sha256"}


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound Solar promotion artifact changed: {binding['path']}")
    value = _load(path)
    expected_content = binding.get("content_sha256")
    if expected_content is not None:
        actual = (
            _sha(_content_body(value))
            if "content_sha256" in value
            else _sha(value)
        )
        if actual != expected_content or (
            "content_sha256" in value and value["content_sha256"] != expected_content
        ):
            raise ValueError(f"bound Solar promotion content changed: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "candidate",
        "formal_pass",
        "solar_prediction_audit",
        "solar_protocol",
        "gr_calibration",
        "prediction_bundle_descriptor_contract",
        "candidate_prediction_bundle_descriptor",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
        "observational_opening_authorized",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("G4 Solar promotion service config is invalid")
    candidate = config["candidate"]
    if candidate != {
        "candidate_id": "G3-f9c598b70a77ea54009d8f18",
        "role": "generated_candidate",
        "family_id": "CONFORMAL_G4_PHI_SCALAR_TENSOR",
        "action_sha256": "6ddd6502d110ead90ff494a6569213ec2e61a0b046dfa86344bb1980df6abc90",
        "seed_lineage_sha256": "f9c598b70a77ea54009d8f18723ff6c54974c8aceab680d2fc95f513a33b2aa7",
    }:
        raise ValueError("G4 Solar promotion candidate identity changed")
    if (
        config.get("data_eligibility") != ELIGIBILITY
        or config.get("external_paid_llm_calls") is not False
        or config.get("observational_opening_authorized") is not False
    ):
        raise ValueError("G4 Solar promotion opened a forbidden input")
    budget = config.get("budget", {})
    if (
        set(budget)
        != {
            "maximum_tasks",
            "maximum_service_cycles",
            "maximum_wall_seconds",
            "maximum_service_bytes",
        }
        or budget["maximum_tasks"] != 1
        or not 1 <= int(budget["maximum_service_cycles"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 30
        or not 4096 <= int(budget["maximum_service_bytes"]) <= 4 * 1024 * 1024
    ):
        raise ValueError("G4 Solar promotion budget is invalid")
    descriptor = config["candidate_prediction_bundle_descriptor"]
    if descriptor is not None and set(descriptor) != {"path", "file_sha256"}:
        raise ValueError("candidate Solar descriptor binding is invalid")


class GrammarV3G4SolarPromotionService:
    """Checkpoint one exact formal pass at the sealed Solar boundary."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        project_root: str | Path,
    ) -> None:
        _validate_config(config)
        self.config = config
        self.root = Path(project_root).resolve()
        self.directory = Path(directory).resolve()
        if "campaign-v1-live.sqlite" in str(self.directory).lower():
            raise ValueError("refusing to use the live campaign watchdog database")
        self.formal_binding = self._validate_formal_pass()
        self.prediction_audit_binding = self._validate_prediction_audit()
        self.protocol_binding = self._validate_protocol()
        self.gr_calibration = self._validate_gr_calibration()
        self.contract_binding = self._validate_contract()
        self.descriptor_binding = self._validate_candidate_descriptor()
        self.work_packet = self._build_packet()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "solar-promotion.sqlite"
        self._initialize()
        self._enforce_disk_budget()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _validate_formal_pass(self) -> dict[str, str]:
        formal = self.config["formal_pass"]
        if set(formal) != {"audit", "queue_status", "service_status"}:
            raise ValueError("G4 formal-pass binding is invalid")
        audit = _load_bound(self.root, formal["audit"])
        records = [item for item in audit["candidate_records"] if item.get("family") == "G4"]
        if len(records) != 1:
            raise ValueError("G4 formal audit candidate set changed")
        record = records[0]
        candidate = self.config["candidate"]
        if (
            record.get("seed_id") != candidate["candidate_id"]
            or record.get("action_sha256") != candidate["action_sha256"]
            or record.get("decision") != "pass"
            or record.get("first_missing_premise") is not None
            or record.get("gate_ledger", {})
            .get("formal_prerequisite_completion", {})
            .get("status")
            != "pass"
            or record.get("provenance", {}).get("binding_sha256")
            != formal["audit"]["candidate_provenance_sha256"]
            or record.get("provenance", {}).get("data_eligibility") != ELIGIBILITY
            or record.get("solar_bundle", {}).get("generated") is not False
            or audit.get("observational_data_opened") is not False
            or audit.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("G4 formal pass is not exact or remains Solar-ineligible")

        queue = _load_bound(self.root, formal["queue_status"])
        candidate_rows = [
            item
            for item in queue.get("work_records", [])
            if item.get("candidate_id") == candidate["candidate_id"]
        ]
        if (
            {item.get("task_type") for item in candidate_rows} != G4_TASK_TYPES
            or any(item.get("state") != "succeeded" for item in candidate_rows)
            or queue.get("followup_decision_counts") != {"blocked": 8, "pass": 2}
            or queue.get("missing_evaluator_count") != 0
            or queue.get("candidate_scientific_decisions_changed") != 1
            or queue.get("queue_registry_root_sha256")
            != formal["queue_status"]["queue_registry_root_sha256"]
            or queue.get("work_records_root_sha256")
            != formal["queue_status"]["completed_work_records_root_sha256"]
            or queue.get("observational_data_opened") is not False
            or queue.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("G4 final queue status does not prove the exact formal pass")

        service = _load_bound(self.root, formal["service_status"])
        if (
            service.get("lifecycle") != "idle"
            or service.get("processed_count") != 10
            or service.get("deferred_count") != 0
            or service.get("missing_evaluator_executions") != 0
            or service.get("candidate_scientific_decisions_changed") != 1
            or service.get("completed_work_records_root_sha256")
            != formal["queue_status"]["completed_work_records_root_sha256"]
            or service.get("queue_registry_root_sha256")
            != formal["queue_status"]["queue_registry_root_sha256"]
            or service.get("observational_data_opened") is not False
            or service.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("G4 final service status is not complete and sealed")
        return {
            "audit_content_sha256": formal["audit"]["content_sha256"],
            "candidate_provenance_sha256": formal["audit"][
                "candidate_provenance_sha256"
            ],
            "completed_work_records_root_sha256": formal["queue_status"][
                "completed_work_records_root_sha256"
            ],
            "queue_registry_root_sha256": formal["queue_status"][
                "queue_registry_root_sha256"
            ],
        }

    def _validate_prediction_audit(self) -> dict[str, Any]:
        binding = self.config["solar_prediction_audit"]
        audit = _load_bound(self.root, binding)
        records = audit.get("candidate_records", [])
        if len(records) != 1:
            raise ValueError("reviewed G4 Solar prediction audit candidate set changed")
        record = records[0]
        candidate = self.config["candidate"]
        admissibility = record.get("real_solar_admissibility", {})
        if (
            audit.get("schema_version")
            != "sigma-g4-conformal-solar-prediction-audit-1.0"
            or audit.get("decision_counts") != {"blocked": 1}
            or audit.get("gate_status_counts") != {"blocked": 2, "pass": 6}
            or audit.get("analytic_known_answer_bundle_count") != 1
            or audit.get("real_solar_bundle_count") != 0
            or audit.get("real_solar_bundle_admissible_count") != 0
            or record.get("seed_id") != candidate["candidate_id"]
            or record.get("action_sha256") != candidate["action_sha256"]
            or record.get("decision") != "blocked"
            or record.get("first_missing_premise")
            != "registered_candidate_specific_action_bound_Solar_bundle"
            or record.get("candidate_analytic_prediction_status")
            != "pass_on_declared_scalar_free_background"
            or record.get("provenance", {}).get("binding_sha256")
            != binding["candidate_provenance_sha256"]
            or record.get("provenance", {}).get("formal_predecessor_content_sha256")
            != self.formal_binding["audit_content_sha256"]
            or record.get("provenance", {}).get("formal_provenance_sha256")
            != self.formal_binding["candidate_provenance_sha256"]
            or record.get("provenance", {}).get("data_eligibility") != ELIGIBILITY
            or record.get("solar_bundle")
            != {
                "analytic_known_answer_bundle_generated": True,
                "real_observational_bundle_generated": False,
                "real_observational_bundle_admissible": False,
                "status": "blocked_before_data_opening",
            }
            or admissibility.get("admissible") is not False
            or admissibility.get("candidate_use_authorized") is not False
            or admissibility.get("dataset_ready") is not False
            or admissibility.get("primary_files_downloaded") is not False
            or admissibility.get("observational_inputs_opened_by_this_audit") is not False
            or audit.get("observational_data_opened") is not False
            or audit.get("paid_llm_spend_usd") != 0.0
            or audit.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("reviewed G4 Solar prediction audit changed or opened data")
        blockers = admissibility.get("blockers", [])
        required_blockers = {
            "real source stress/pressure/composition and scalar-branch uniqueness are not bound",
            "training-only initial state, nuisance, likelihood, and stopping rule are not frozen",
        }
        if not required_blockers.issubset(blockers):
            raise ValueError("reviewed G4 Solar blockers changed")
        return {
            "audit_content_sha256": binding["content_sha256"],
            "candidate_provenance_sha256": binding[
                "candidate_provenance_sha256"
            ],
            "analytic_bundle_count": 1,
            "real_bundle_count": 0,
            "decision": "blocked",
        }

    def _validate_protocol(self) -> dict[str, str]:
        binding = self.config["solar_protocol"]
        protocol = _load_bound(self.root, binding)
        if (
            protocol.get("schema_version") != "sigma-solar-observable-protocol-1.0"
            or protocol.get("status") != "sealed"
            or protocol.get("data_opened") is not False
            or protocol.get("scoring_contract", {}).get(
                "object_specific_gravity_parameters"
            )
            != 0
            or "dark matter or invisible halo"
            not in protocol.get("prohibited_truth_or_rescue", [])
            or "redshift-derived distance"
            not in protocol.get("prohibited_truth_or_rescue", [])
        ):
            raise ValueError("Solar observable protocol is not sealed and direct-observable")
        return {
            "path": binding["path"],
            "file_sha256": binding["file_sha256"],
            "content_sha256": binding["content_sha256"],
        }

    def _validate_gr_calibration(self) -> dict[str, Any]:
        binding = self.config["gr_calibration"]
        if (
            binding.get("role") != "calibration_only_control"
            or binding.get("promotion_eligible") is not False
        ):
            raise ValueError("GR Solar control leaked into candidate promotion")
        artifact = _load_bound(self.root, binding["status_artifact"])
        control = artifact.get("known_answer_control", {})
        unmapped = artifact.get("unmapped_candidate_control", {})
        statuses = control.get("golden_statuses", {})
        if (
            control.get("candidate_id") != binding["candidate_id"]
            or control.get("decision") != "pass"
            or control.get("bundle_binding_sha256")
            != binding["bundle_binding_sha256"]
            or len(statuses) != 5
            or set(statuses.values()) != {"pass"}
            or unmapped.get("decision") != "blocked"
            or control.get("data_eligibility") != ELIGIBILITY
            or artifact.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("GR Solar calibration artifact changed or became promotable")
        return {
            "candidate_id": binding["candidate_id"],
            "role": "calibration_only_control",
            "promotion_eligible": False,
            "decision": "pass",
            "exact_metrics": {
                "passed_control_count": 5,
                "total_control_count": 5,
            },
            "artifact_path": binding["status_artifact"]["path"],
            "artifact_content_sha256": binding["status_artifact"]["content_sha256"],
            "bundle_binding_sha256": binding["bundle_binding_sha256"],
        }

    def _validate_contract(self) -> dict[str, str]:
        binding = self.config["prediction_bundle_descriptor_contract"]
        contract = _load_bound(self.root, binding)
        if (
            contract.get("$id")
            != "sigma://grammar-v3/g4-solar-prediction-bundle-descriptor-1.0"
            or contract.get("additionalProperties") is not False
            or contract.get("properties", {})
            .get("schema_version", {})
            .get("const")
            != DESCRIPTOR_SCHEMA
        ):
            raise ValueError("candidate Solar descriptor contract changed")
        return dict(binding)

    def _validate_candidate_descriptor(self) -> dict[str, Any] | None:
        binding = self.config["candidate_prediction_bundle_descriptor"]
        if binding is None:
            return None
        descriptor = _load_bound(self.root, binding)
        required = {
            "schema_version",
            "descriptor_id",
            "candidate",
            "formal_pass_binding",
            "prediction_audit_binding",
            "solar_protocol_binding",
            "prediction_bundle",
            "reviewed_evaluator",
            "observational_opening",
            "data_eligibility",
        }
        if (
            set(descriptor) != required
            or descriptor.get("schema_version") != DESCRIPTOR_SCHEMA
            or descriptor.get("candidate") != self.config["candidate"]
            or descriptor.get("formal_pass_binding") != self.formal_binding
            or descriptor.get("prediction_audit_binding")
            != self.prediction_audit_binding
            or descriptor.get("solar_protocol_binding") != self.protocol_binding
            or descriptor.get("data_eligibility") != ELIGIBILITY
            or descriptor.get("observational_opening")
            != {
                "authorized": False,
                "requires_independent_dataset_manifest_audit": True,
                "requires_preregistered_session_split": True,
            }
        ):
            raise ValueError("candidate Solar descriptor violates the exact contract")
        bundle_binding = descriptor["prediction_bundle"]
        bundle = _load_bound(self.root, bundle_binding)
        bundle_required = {
            "schema_version",
            "candidate_id",
            "action_sha256",
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
            "content_sha256",
        }
        if (
            set(bundle) != bundle_required
            or bundle.get("schema_version") != PREDICTION_BUNDLE_SCHEMA
            or bundle.get("candidate_id") != self.config["candidate"]["candidate_id"]
            or bundle.get("action_sha256") != self.config["candidate"]["action_sha256"]
            or bundle.get("output_channels") != OUTPUT_CHANNELS
            or not isinstance(bundle.get("universal_parameter_count"), int)
            or bundle.get("universal_parameter_count") < 0
            or bundle.get("object_specific_gravity_parameter_count") != 0
            or bundle.get("data_eligibility") != ELIGIBILITY
            or bundle.get("observational_data_opened") is not False
        ):
            raise ValueError("candidate Solar prediction bundle changed or opened data")
        descriptor_bundle_fields = {
            key: value
            for key, value in bundle_binding.items()
            if key not in {"path", "file_sha256", "content_sha256"}
        }
        actual_bundle_fields = {
            key: bundle[key]
            for key in descriptor_bundle_fields
        }
        if descriptor_bundle_fields != actual_bundle_fields:
            raise ValueError("candidate Solar descriptor violates the exact contract")
        hashes = [
            bundle[key]
            for key in bundle_required
            if key.endswith("_sha256") and key != "content_sha256"
        ]
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("candidate Solar prediction component hash is invalid")
        evaluator_binding = descriptor["reviewed_evaluator"]
        evaluator_path = self.root / evaluator_binding["descriptor_path"]
        if (
            not evaluator_path.is_file()
            or _file_sha(evaluator_path) != evaluator_binding["descriptor_file_sha256"]
        ):
            raise ValueError("reviewed candidate Solar evaluator descriptor changed")
        evaluator = _load(evaluator_path)
        if (
            set(evaluator)
            != {
                "schema_version",
                "evaluator_id",
                "candidate_id",
                "action_sha256",
                "callback",
                "artifact_path",
                "artifact_sha256",
                "data_eligibility",
            }
            or evaluator.get("schema_version") != EVALUATOR_DESCRIPTOR_SCHEMA
            or evaluator.get("candidate_id") != self.config["candidate"]["candidate_id"]
            or evaluator.get("action_sha256") != self.config["candidate"]["action_sha256"]
            or evaluator.get("callback") != evaluator_binding["callback"]
            or evaluator.get("data_eligibility") != ELIGIBILITY
            or _sha(evaluator) != evaluator_binding["evaluator_binding_sha256"]
        ):
            raise ValueError("reviewed candidate Solar evaluator is not exactly allowlisted")
        callback_artifact = self.root / evaluator["artifact_path"]
        if (
            not callback_artifact.is_file()
            or _file_sha(callback_artifact) != evaluator["artifact_sha256"]
        ):
            raise ValueError("reviewed candidate Solar evaluator source changed")
        return {
            "descriptor_file_sha256": binding["file_sha256"],
            "descriptor_content_sha256": _sha(descriptor),
            "prediction_bundle_content_sha256": bundle_binding["content_sha256"],
            "evaluator_binding_sha256": evaluator_binding["evaluator_binding_sha256"],
        }

    def _build_packet(self) -> dict[str, Any]:
        candidate = self.config["candidate"]
        body = {
            "schema_version": TASK_SCHEMA,
            "candidate_id": candidate["candidate_id"],
            "role": "generated_candidate",
            "action_sha256": candidate["action_sha256"],
            "formal_pass_binding": self.formal_binding,
            "prediction_audit_binding": self.prediction_audit_binding,
            "descriptor_contract_file_sha256": self.contract_binding["file_sha256"],
            "data_eligibility": ELIGIBILITY,
        }
        lineage = _sha(body)
        return {
            **body,
            "task_id": f"G4SOL-{lineage[:24]}",
            "lineage_sha256": lineage,
        }

    def _initialize(self) -> None:
        config_sha = _sha(self.config)
        config_json = _canonical(self.config)
        descriptor_sha = (
            self.descriptor_binding["descriptor_content_sha256"]
            if self.descriptor_binding
            else None
        )
        state = (
            "ready_for_reviewed_solar_evaluator"
            if self.descriptor_binding
            else "deferred_missing_prediction_bundle_descriptor"
        )
        blocker = None if self.descriptor_binding else MISSING_DESCRIPTOR
        expected_work = {
            "task_id": self.work_packet["task_id"],
            "lineage_sha256": self.work_packet["lineage_sha256"],
            "candidate_id": self.work_packet["candidate_id"],
            "role": "generated_candidate",
            "state": state,
            "blocker": blocker,
            "descriptor_binding_sha256": descriptor_sha,
            "payload_json": _canonical(self.work_packet),
        }
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            row = connection.execute("SELECT * FROM service_state WHERE singleton=1").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO service_state VALUES (1,?,?,?,'created',0,0)",
                    (STATUS_SCHEMA, config_sha, config_json),
                )
                connection.execute(
                    "INSERT INTO promotion_work VALUES (?,?,?,?,?,?,?,?)",
                    tuple(expected_work.values()),
                )
                self._event(connection, "service_created", {"config_sha256": config_sha})
                return
            if row["schema_version"] != STATUS_SCHEMA:
                raise ValueError("Solar promotion service schema changed")
            old_config = json.loads(row["config_json"])
            if row["config_sha256"] != config_sha:
                old_without = {
                    key: value
                    for key, value in old_config.items()
                    if key != "candidate_prediction_bundle_descriptor"
                }
                new_without = {
                    key: value
                    for key, value in self.config.items()
                    if key != "candidate_prediction_bundle_descriptor"
                }
                if (
                    old_without != new_without
                    or old_config.get("candidate_prediction_bundle_descriptor") is not None
                    or self.config.get("candidate_prediction_bundle_descriptor") is None
                    or row["lifecycle"] not in {"stopped", "waiting_for_prediction_bundle"}
                ):
                    raise ValueError("refusing an unbound Solar promotion config transition")
                connection.execute(
                    "UPDATE service_state SET config_sha256=?,config_json=? WHERE singleton=1",
                    (config_sha, config_json),
                )
                connection.execute(
                    "UPDATE promotion_work SET state=?,blocker=?,descriptor_binding_sha256=? "
                    "WHERE task_id=?",
                    (state, blocker, descriptor_sha, self.work_packet["task_id"]),
                )
                self._event(
                    connection,
                    "prediction_bundle_descriptor_registered",
                    {"descriptor_binding_sha256": descriptor_sha},
                )
            work = connection.execute("SELECT * FROM promotion_work").fetchone()
            if work is None:
                raise ValueError("Solar promotion work checkpoint disappeared")
            actual = dict(work)
            if row["config_sha256"] == config_sha and actual != expected_work:
                raise ValueError("Solar promotion work lineage changed")

    @staticmethod
    def _event(
        connection: sqlite3.Connection, event_type: str, payload: dict[str, Any]
    ) -> None:
        connection.execute(
            "INSERT INTO service_events(event_type,payload_json) VALUES (?,?)",
            (event_type, _canonical(payload)),
        )

    def _database_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in (
                self.database,
                Path(str(self.database) + "-wal"),
                Path(str(self.database) + "-shm"),
            )
            if path.is_file()
        )

    def _enforce_disk_budget(self) -> int:
        consumed = self._database_bytes()
        if consumed > self.config["budget"]["maximum_service_bytes"]:
            raise RuntimeError("Solar promotion service disk budget exhausted")
        return consumed

    def _cycle(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM service_state").fetchone()
            if row["stop_requested"]:
                raise RuntimeError("Solar promotion service is stopped")
            if row["cycle_count"] >= self.config["budget"]["maximum_service_cycles"]:
                raise RuntimeError("Solar promotion service cycle budget exhausted")
            lifecycle = (
                "ready_for_reviewed_solar_evaluator"
                if self.descriptor_binding
                else "waiting_for_prediction_bundle"
            )
            connection.execute(
                "UPDATE service_state SET lifecycle=?,cycle_count=cycle_count+1 WHERE singleton=1",
                (lifecycle,),
            )
            self._event(
                connection,
                "service_cycle",
                {
                    "descriptor_registered": self.descriptor_binding is not None,
                    "solar_evaluator_invoked": False,
                    "observational_data_opened": False,
                },
            )
        self._enforce_disk_budget()
        return self.status()

    def start(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM service_state").fetchone()
            if row["lifecycle"] != "created":
                raise ValueError("Solar promotion start requires a new lifecycle")
            connection.execute(
                "UPDATE service_state SET lifecycle='running',stop_requested=0 WHERE singleton=1"
            )
            self._event(connection, "service_started", {})
        return self._cycle()

    def stop(self) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                "UPDATE service_state SET lifecycle='stopped',stop_requested=1 WHERE singleton=1"
            )
            self._event(connection, "service_stopped", {})
        return self.status()

    def resume(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM service_state").fetchone()
            if row["lifecycle"] not in {"stopped", "waiting_for_prediction_bundle"}:
                raise ValueError("Solar promotion resume requires stopped or waiting state")
            connection.execute(
                "UPDATE service_state SET lifecycle='running',stop_requested=0 WHERE singleton=1"
            )
            self._event(connection, "service_resumed", {})
        return self._cycle()

    def status(self) -> dict[str, Any]:
        with self._connect() as connection:
            state = dict(connection.execute("SELECT * FROM service_state").fetchone())
            work = dict(connection.execute("SELECT * FROM promotion_work").fetchone())
            event_count = connection.execute("SELECT COUNT(*) FROM service_events").fetchone()[0]
        candidate_row = {
            "candidate_id": work["candidate_id"],
            "role": "generated_candidate",
            "rank": None,
            "exact_metrics": None,
            "evidence_status": (
                "ready_for_reviewed_evaluator"
                if self.descriptor_binding
                else "untested"
            ),
            "data_class": "sealed_candidate_specific_solar_prediction",
            "gate_completeness": (
                "formal_complete_prediction_bundle_registered_solar_unopened"
                if self.descriptor_binding
                else "formal_complete_prediction_bundle_missing"
            ),
            "blocker": work["blocker"],
            "lineage_sha256": work["lineage_sha256"],
            "artifact_path": self.config["formal_pass"]["audit"]["path"],
            "artifact_content_sha256": self.formal_binding["audit_content_sha256"],
            "uncertainty": (
                "No candidate-specific Solar prediction bundle is registered; this is untested, "
                "not poor measured performance."
                if self.descriptor_binding is None
                else "The bundle is registered but no Solar evaluator or observation has run."
            ),
            "promotion_eligible": False,
        }
        gr_row = {
            **self.gr_calibration,
            "rank": 1,
            "evidence_status": "pass",
            "data_class": "sealed_solar_known_answer",
            "gate_completeness": "complete_for_calibration_only",
            "blocker": None,
            "uncertainty": (
                "GR validates the sealed reference solver only and cannot promote a generated "
                "candidate."
            ),
        }
        body = {
            "schema_version": STATUS_SCHEMA,
            "lifecycle": state["lifecycle"],
            "cycle_count": int(state["cycle_count"]),
            "task_count": 1,
            "work_state_counts": {work["state"]: 1},
            "formal_pass_verified": True,
            "prediction_bundle_descriptor_registered": self.descriptor_binding is not None,
            "reviewed_solar_evaluator_invoked": False,
            "solar_evaluator_opened": False,
            "observational_data_opened": False,
            "queue_root_sha256": _sha([self.work_packet]),
            "work_record_root_sha256": _sha(
                [
                    {
                        key: work[key]
                        for key in (
                            "task_id",
                            "lineage_sha256",
                            "candidate_id",
                            "role",
                            "state",
                            "blocker",
                            "descriptor_binding_sha256",
                        )
                    }
                ]
            ),
            "formal_pass_binding": self.formal_binding,
            "reviewed_prediction_audit_binding": self.prediction_audit_binding,
            "descriptor_contract_binding": self.contract_binding,
            "candidate_prediction_bundle_descriptor_binding": self.descriptor_binding,
            "category_leaderboard": {
                "category": "solar_known_answer",
                "ranking_rule": "completed comparable evidence within this category only",
                "ranked": [gr_row],
                "blocked_or_untested": [candidate_row],
                "scalar_truth_score": None,
            },
            "service_event_count": int(event_count),
            "database_bytes": self._enforce_disk_budget(),
            "budget": dict(self.config["budget"]),
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "The exact G4 formal pass is checkpointed at the Solar boundary. GR is calibration "
                "only. The generated candidate remains unranked and no Solar evaluator or data may "
                "open until its exact prediction-bundle descriptor is registered."
            ),
        }
        return {**body, "content_sha256": _sha(body)}

    def export(self) -> dict[str, Any]:
        status = self.status()
        body = {
            key: value
            for key, value in status.items()
            if key not in {"content_sha256", "database_bytes"}
        }
        return {**body, "content_sha256": _sha(body)}
