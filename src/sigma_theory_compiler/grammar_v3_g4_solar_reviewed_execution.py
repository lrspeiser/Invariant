"""Restart-safe execution of the reviewed, still-sealed G4 Solar callback."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_solar_evaluator import (
    ACTION_SHA256,
    CANDIDATE_ID,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_solar_evaluator,
)

CONFIG_SCHEMA = "sigma-grammar-v3-g4-solar-reviewed-execution-config-1.0"
PACKET_SCHEMA = "sigma-grammar-v3-g4-solar-reviewed-execution-packet-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-g4-solar-reviewed-execution-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-g4-solar-reviewed-execution-status-1.0"
EVALUATOR_DESCRIPTOR_SCHEMA = "sigma-candidate-solar-evaluator-descriptor-1.0"
CALLBACK = (
    "sigma_theory_compiler.reviewed_g4_candidate_solar_evaluator:"
    "reviewed_g4_candidate_solar_evaluator"
)
DESCRIPTOR_FIELD = "reviewed_candidate_solar_evaluator_descriptor_sha256"
BLOCKER = "missing_fully_registered_real_source_and_prediction_bundle"
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
        raise ValueError(f"bound G4 Solar execution artifact changed: {binding['path']}")
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
            raise ValueError(f"bound G4 Solar execution content changed: {binding['path']}")
    return value


def _expected_coordinator_identity(config: dict[str, Any], packet: dict[str, Any]) -> tuple[str, int]:
    seed_payload = {
        "master_seed": int(config["coordinator"]["determinism"]["master_seed"]),
        "ordinal": 0,
        "lane": "cpu",
        "payload": packet,
    }
    encoded = _canonical(seed_payload).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
    return f"PSW-{hashlib.sha256(encoded).hexdigest()[:24]}", seed


class GrammarV3G4SolarReviewedExecution:
    """One-candidate reviewed Solar readiness queue; never an observation opener."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        project_root: str | Path,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.directory = Path(directory).resolve()
        if "campaign-v1" in str(self.directory).lower():
            raise ValueError("refusing to use the live campaign-v1 database")
        self.config = config
        self._validate_config()
        self.queue_predecessor = self._validate_queue_predecessor()
        self.promotion_predecessor = self._validate_promotion_predecessor()
        self.descriptor = self._validate_descriptor_allowlist()
        self.readiness = self._validate_readiness()
        self.packet = self._build_packet()
        profile = _load_bound(self.root, self.config["resource_profile"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.coordinator = PersistentParallelSearch(
            self.directory / "reviewed-g4-solar.sqlite",
            self.config["coordinator"],
            profile,
        )
        self._bind_adapter_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _validate_config(self) -> None:
        required = {
            "schema_version",
            "candidate",
            "predecessor_followup_queue",
            "predecessor_solar_promotion_status",
            "reviewed_readiness",
            "evaluator_descriptor_allowlist",
            "executor_source",
            "resource_profile",
            "coordinator",
            "observational_opening_authorized",
            "external_paid_llm_calls",
            "data_eligibility",
        }
        if set(self.config) != required or self.config.get("schema_version") != CONFIG_SCHEMA:
            raise ValueError("reviewed G4 Solar execution config is invalid")
        if self.config["candidate"] != {
            "candidate_id": CANDIDATE_ID,
            "role": "generated_candidate",
            "action_sha256": ACTION_SHA256,
        }:
            raise ValueError("reviewed G4 Solar execution candidate changed")
        if (
            self.config.get("observational_opening_authorized") is not False
            or self.config.get("external_paid_llm_calls") is not False
            or self.config.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("reviewed G4 Solar execution opened a forbidden input")
        executor = self.config["executor_source"]
        executor_path = self.root / executor.get("path", "")
        if (
            executor.get("path")
            != "src/sigma_theory_compiler/grammar_v3_g4_solar_reviewed_execution.py"
            or not executor_path.is_file()
            or _file_sha(executor_path) != executor.get("file_sha256")
        ):
            raise ValueError("reviewed G4 Solar execution source changed")
        coordinator = self.config["coordinator"]
        if (
            coordinator.get("external_paid_llm_calls") is not False
            or coordinator.get("queue", {}).get("maximum_pending_work") != 1
            or coordinator.get("queue", {}).get("maximum_attempts") != 3
            or coordinator.get("budget", {}).get("maximum_tasks") != 1
            or coordinator.get("cpu", {}).get("maximum_workers") != 1
            or coordinator.get("supervisor", {}).get("cpu_workers") != 1
            or coordinator.get("supervisor", {}).get("gpu_workers") != 0
        ):
            raise ValueError("reviewed G4 Solar coordinator is not exactly bounded")

    def _validate_queue_predecessor(self) -> dict[str, str]:
        binding = self.config["predecessor_followup_queue"]
        status = _load_bound(self.root, binding)
        candidate_records = [
            item
            for item in status.get("work_records", [])
            if item.get("candidate_id") == CANDIDATE_ID
        ]
        if (
            status.get("queue_registry_root_sha256")
            != binding["queue_registry_root_sha256"]
            or status.get("work_records_root_sha256")
            != binding["completed_work_records_root_sha256"]
            or status.get("followup_decision_counts") != {"blocked": 8, "pass": 2}
            or {item.get("task_type") for item in candidate_records}
            != {"g4_global_lapse_invertibility", "g4_global_positive_energy"}
            or any(item.get("state") != "succeeded" for item in candidate_records)
            or status.get("observational_data_opened") is not False
            or status.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("reviewed G4 Solar follow-up predecessor changed")
        return {
            "queue_registry_root_sha256": binding["queue_registry_root_sha256"],
            "completed_work_records_root_sha256": binding[
                "completed_work_records_root_sha256"
            ],
            "content_sha256": binding["content_sha256"],
        }

    def _validate_promotion_predecessor(self) -> dict[str, str]:
        binding = self.config["predecessor_solar_promotion_status"]
        status = _load_bound(self.root, binding)
        if (
            status.get("lifecycle") != "waiting_for_prediction_bundle"
            or status.get("formal_pass_verified") is not True
            or status.get("reviewed_solar_evaluator_invoked") is not False
            or status.get("prediction_bundle_descriptor_registered") is not False
            or status.get("observational_data_opened") is not False
            or status.get("paid_llm_spend_usd") != 0.0
            or status.get("formal_pass_binding", {}).get("queue_registry_root_sha256")
            != self.queue_predecessor["queue_registry_root_sha256"]
            or status.get("formal_pass_binding", {}).get(
                "completed_work_records_root_sha256"
            )
            != self.queue_predecessor["completed_work_records_root_sha256"]
        ):
            raise ValueError("reviewed G4 Solar promotion predecessor changed")
        return {
            "content_sha256": binding["content_sha256"],
            "queue_root_sha256": status["queue_root_sha256"],
            "work_record_root_sha256": status["work_record_root_sha256"],
        }

    def _validate_descriptor_allowlist(self) -> dict[str, Any]:
        allowlist = self.config["evaluator_descriptor_allowlist"]
        if not isinstance(allowlist, list) or len(allowlist) != 1:
            raise ValueError("reviewed G4 Solar evaluator allowlist changed")
        binding = allowlist[0]
        descriptor = _load_bound(self.root, binding)
        expected = {
            "schema_version": EVALUATOR_DESCRIPTOR_SCHEMA,
            "evaluator_id": "reviewed-g4-candidate-solar-readiness-v1",
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "callback": CALLBACK,
            "artifact_path": "src/sigma_theory_compiler/reviewed_g4_candidate_solar_evaluator.py",
            "artifact_sha256": binding["callback_source_sha256"],
            "data_eligibility": ELIGIBILITY,
        }
        source = self.root / descriptor.get("artifact_path", "")
        if (
            descriptor != expected
            or _sha(descriptor) != binding["descriptor_binding_sha256"]
            or not source.is_file()
            or _file_sha(source) != descriptor["artifact_sha256"]
        ):
            raise ValueError("reviewed G4 Solar evaluator is not exactly allowlisted")
        return descriptor

    def _validate_readiness(self) -> dict[str, Any]:
        binding = self.config["reviewed_readiness"]
        readiness = _load_bound(self.root, binding)
        decision = readiness.get("current_evaluator_decision", {})
        expected_missing = sorted(set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD})
        if (
            readiness.get("candidate", {}).get("candidate_id") != CANDIDATE_ID
            or readiness.get("candidate", {}).get("action_sha256") != ACTION_SHA256
            or readiness.get("descriptor_implementation_ready") is not True
            or readiness.get("real_source_prediction_bundle_registered") is not False
            or readiness.get("candidate_use_authorized") is not False
            or readiness.get("observational_authorization") is not False
            or readiness.get("observational_data_opened") is not False
            or readiness.get("primary_record_access_count") != 0
            or decision.get("decision") != "blocked"
            or decision.get("blocker") != BLOCKER
            or decision.get("filled_registration_hash_count") != 1
            or decision.get("missing_registration_hashes") != expected_missing
            or readiness.get("implementation_readiness", {}).get(
                "descriptor_binding_sha256"
            )
            != self.config["evaluator_descriptor_allowlist"][0][
                "descriptor_binding_sha256"
            ]
            or readiness.get("data_eligibility") != ELIGIBILITY
            or readiness.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("reviewed G4 Solar readiness binding changed")
        return readiness

    def _build_packet(self) -> dict[str, Any]:
        registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
        registration[DESCRIPTOR_FIELD] = self.config["evaluator_descriptor_allowlist"][0][
            "descriptor_binding_sha256"
        ]
        body = {
            "schema_version": PACKET_SCHEMA,
            "ordinal": 0,
            "candidate": self.config["candidate"],
            "predecessor_followup_queue": self.queue_predecessor,
            "predecessor_solar_promotion_status": self.promotion_predecessor,
            "readiness_content_sha256": self.config["reviewed_readiness"][
                "content_sha256"
            ],
            "evaluator_descriptor_binding_sha256": registration[DESCRIPTOR_FIELD],
            "callback": CALLBACK,
            "registration_hashes": registration,
            "observational_opening_authorized": False,
            "data_eligibility": ELIGIBILITY,
        }
        lineage = _sha(body)
        return {
            **body,
            "followup_task_id": f"G4SOLREV-{lineage[:24]}",
            "followup_lineage_sha256": lineage,
        }

    def _bind_adapter_state(self) -> None:
        expected = {
            "schema_version": STATUS_SCHEMA,
            "config_sha256": _sha(self.config),
            "packet_root_sha256": _sha([self.packet]),
        }
        with self.coordinator.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS g4_solar_reviewed_adapter ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                "schema_version TEXT NOT NULL,config_sha256 TEXT NOT NULL,"
                "packet_root_sha256 TEXT NOT NULL)"
            )
            row = connection.execute("SELECT * FROM g4_solar_reviewed_adapter").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO g4_solar_reviewed_adapter VALUES (1,?,?,?)",
                    tuple(expected.values()),
                )
            elif {key: row[key] for key in expected} != expected:
                raise ValueError("refusing to resume a changed reviewed G4 Solar execution")

    def enqueue(self) -> dict[str, int]:
        result = self.coordinator.enqueue([self.packet], lane="cpu", max_attempts=3)
        return {**result, "requested": 1}

    def _validate_lease(self, lease: WorkLease) -> None:
        work_id, seed = _expected_coordinator_identity(self.config, self.packet)
        if (
            lease.payload != self.packet
            or lease.ordinal != 0
            or lease.lane != "cpu"
            or lease.work_id != work_id
            or lease.seed != seed
        ):
            raise ValueError("reviewed G4 Solar lease or lineage changed")

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        self._validate_lease(lease)
        context = {
            "data_eligibility": dict(ELIGIBILITY),
            "observational_opening_authorized": False,
            "registration_hashes": dict(self.packet["registration_hashes"]),
        }
        reviewed = reviewed_g4_candidate_solar_evaluator(
            {
                "candidate_id": CANDIDATE_ID,
                "action_sha256": ACTION_SHA256,
                "role": "generated_candidate",
                "data_eligibility": dict(ELIGIBILITY),
            },
            context,
        )
        expected_missing = sorted(set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD})
        if (
            reviewed.get("decision") != "blocked"
            or reviewed.get("blocker") != BLOCKER
            or reviewed.get("filled_registration_hash_count") != 1
            or reviewed.get("missing_registration_hashes") != expected_missing
            or reviewed.get("observational_opening_authorized") is not False
            or reviewed.get("observational_data_opened") is not False
            or reviewed.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("reviewed G4 Solar callback result changed or opened data")
        body = {
            "schema_version": RESULT_SCHEMA,
            "work_id": lease.work_id,
            "followup_task_id": self.packet["followup_task_id"],
            "followup_lineage_sha256": self.packet["followup_lineage_sha256"],
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "decision": "blocked",
            "blocker": BLOCKER,
            "reviewed_evaluator_invoked": True,
            "evaluator_descriptor_binding_sha256": self.packet[
                "evaluator_descriptor_binding_sha256"
            ],
            "filled_registration_hash_count": 1,
            "missing_registration_hashes": expected_missing,
            "candidate_use_authorized": False,
            "observational_opening_authorized": False,
            "observational_data_opened": False,
            "primary_record_access_count": 0,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self) -> dict[str, Any]:
        processed = 0
        lease = self.coordinator.claim("cpu", "reviewed-g4-solar-worker")
        if lease is not None:
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, "reviewed-g4-solar-worker", result):
                    raise RuntimeError("reviewed G4 Solar lease was lost")
                processed = 1
            except BaseException as error:
                self.coordinator.fail(
                    lease,
                    "reviewed-g4-solar-worker",
                    f"{type(error).__name__}: {error}",
                )
                raise
        if processed:
            self.coordinator.checkpoint()
        return {"processed": processed, "status": self.status()}

    def status(self) -> dict[str, Any]:
        expected_work_id, expected_seed = _expected_coordinator_identity(
            self.config, self.packet
        )
        records: list[dict[str, Any]] = []
        states: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        invoked = 0
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT work_id,ordinal,lane,seed,payload_json,state,attempt,result_json,"
                "error_text FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            if (
                payload != self.packet
                or row["work_id"] != expected_work_id
                or int(row["ordinal"]) != 0
                or row["lane"] != "cpu"
                or int(row["seed"]) != expected_seed
            ):
                raise ValueError("stored reviewed G4 Solar work lineage changed")
            state = str(row["state"])
            states[state] += 1
            result_sha = None
            if row["result_json"] is not None:
                result = json.loads(row["result_json"])
                body = {key: item for key, item in result.items() if key != "content_sha256"}
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("followup_lineage_sha256")
                    != self.packet["followup_lineage_sha256"]
                    or result.get("decision") != "blocked"
                    or result.get("blocker") != BLOCKER
                    or result.get("reviewed_evaluator_invoked") is not True
                    or result.get("observational_data_opened") is not False
                ):
                    raise ValueError("stored reviewed G4 Solar result changed")
                decisions[result["decision"]] += 1
                invoked += 1
                result_sha = result["content_sha256"]
            records.append(
                {
                    "work_id": row["work_id"],
                    "followup_task_id": payload["followup_task_id"],
                    "followup_lineage_sha256": payload["followup_lineage_sha256"],
                    "candidate_id": CANDIDATE_ID,
                    "ordinal": int(row["ordinal"]),
                    "coordinator_seed": int(row["seed"]),
                    "state": state,
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "error_text": row["error_text"],
                }
            )
        telemetry = self.coordinator.telemetry()
        body = {
            "schema_version": STATUS_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "predecessor_followup_queue": self.queue_predecessor,
            "predecessor_solar_promotion_status": self.promotion_predecessor,
            "evaluator_descriptor_allowlist_count": 1,
            "evaluator_descriptor_binding_sha256": self.packet[
                "evaluator_descriptor_binding_sha256"
            ],
            "queue_registry_root_sha256": _sha([self.packet]),
            "task_count": len(records),
            "work_state_counts": dict(sorted(states.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "reviewed_evaluator_invocation_count": invoked,
            "filled_registration_hash_count": 1,
            "missing_registration_hash_count": 16,
            "missing_registration_hashes": sorted(
                set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD}
            ),
            "work_records": records,
            "work_records_root_sha256": _sha(records),
            "checkpoint_sequence": telemetry["checkpoint_sequence"],
            "recovered_leases": telemetry["recovered_leases"],
            "restart_safe": True,
            "portable_status": True,
            "candidate_use_authorized": False,
            "observational_opening_authorized": False,
            "observational_data_opened": False,
            "primary_record_access_count": 0,
            "tracking_target_values_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "The exact reviewed G4 Solar callback ran through a durable one-candidate lease. "
                "It reproduced the readiness blocker with only the descriptor hash filled; no "
                "candidate prediction bundle, source registration, or observation was opened."
            ),
        }
        return {**body, "content_sha256": _sha(body)}

    def export(self) -> dict[str, Any]:
        return self.status()
