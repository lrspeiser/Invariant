from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .grammar_v3_followup_queue import GrammarV3FollowupQueue
from .persistent_parallel_search import PersistentParallelSearch
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-followup-service-config-1.0"
G3_EPOCH_CONFIG_SCHEMA = "sigma-grammar-v3-followup-service-config-g3-epoch-1.0"
G4_FINAL_CONFIG_SCHEMA = "sigma-grammar-v3-followup-service-config-g4-final-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-followup-service-status-1.0"

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS service_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  lifecycle TEXT NOT NULL,
  cycle_count INTEGER NOT NULL DEFAULT 0,
  admitted_count INTEGER NOT NULL DEFAULT 0,
  processed_count INTEGER NOT NULL DEFAULT 0,
  deferred_count INTEGER NOT NULL DEFAULT 0,
  stop_requested INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS deferred_packets (
  followup_task_id TEXT PRIMARY KEY,
  followup_lineage_sha256 TEXT NOT NULL,
  task_type TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state='deferred_missing_evaluator')
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


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "queue_config",
        "reviewed_report_revisions",
        "evaluator_descriptor_allowlist",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    schema = config.get("schema_version")
    if schema in {G3_EPOCH_CONFIG_SCHEMA, G4_FINAL_CONFIG_SCHEMA}:
        required.add("predecessor_service_epoch")
    if set(config) != required or schema not in {
        CONFIG_SCHEMA,
        G3_EPOCH_CONFIG_SCHEMA,
        G4_FINAL_CONFIG_SCHEMA,
    }:
        raise ValueError("grammar-v3 follow-up service config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 follow-up service eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 follow-up service enabled paid LLM calls")
    budget = config.get("budget", {})
    if (
        set(budget) != {
            "maximum_tasks",
            "maximum_wall_seconds",
            "maximum_service_cycles",
            "maximum_service_bytes",
        }
        or int(budget["maximum_tasks"]) != 10
        or int(budget["maximum_service_cycles"]) != (4 if schema == G4_FINAL_CONFIG_SCHEMA else 3)
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 4096 <= int(budget["maximum_service_bytes"]) <= 64 * 1024 * 1024
    ):
        raise ValueError("grammar-v3 follow-up service budget is invalid")
    revisions = config.get("reviewed_report_revisions")
    if not isinstance(revisions, list) or len(revisions) != 1:
        raise ValueError("service requires a finite immutable report revision allowlist")
    allowlist = config.get("evaluator_descriptor_allowlist")
    expected_evaluators = 7 if schema == G4_FINAL_CONFIG_SCHEMA else 5 if schema == G3_EPOCH_CONFIG_SCHEMA else 3
    if not isinstance(allowlist, list) or len(allowlist) != expected_evaluators:
        raise ValueError("service evaluator descriptor allowlist is invalid")
    if schema in {G3_EPOCH_CONFIG_SCHEMA, G4_FINAL_CONFIG_SCHEMA} and set(
        config["predecessor_service_epoch"]
    ) != {
        "config_path",
        "config_file_sha256",
        "config_sha256",
        "completed_work_records_root_sha256",
        "deferred_packet_root_sha256",
    }:
        raise ValueError("service predecessor epoch binding is invalid")


class GrammarV3FollowupService:
    """Restart-safe local lifecycle for reviewed grammar-v3 follow-up work."""

    def __init__(
        self,
        directory: str | Path,
        service_config: dict[str, Any],
        coordinator_config: dict[str, Any],
        resource_profile: dict[str, Any],
        project_root: str | Path,
    ) -> None:
        _validate_config(service_config)
        self.directory = Path(directory).resolve()
        if self.directory.name.lower() == "campaign-v1-live.sqlite":
            raise ValueError("refusing to use the live campaign watchdog database")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "service.sqlite"
        self.coordinator_database = self.directory / "coordinator.sqlite"
        self.config = service_config
        self.root = Path(project_root).resolve()
        self._validate_reviewed_inputs()
        if (
            coordinator_config.get("external_paid_llm_calls") is not False
            or int(coordinator_config["budget"]["maximum_tasks"]) != 10
            or float(coordinator_config["budget"]["maximum_wall_seconds"])
            > float(service_config["budget"]["maximum_wall_seconds"])
            or int(coordinator_config["queue"]["maximum_pending_work"]) < 10
        ):
            raise ValueError("coordinator does not preserve service task/wall/$0 bounds")
        self.coordinator = PersistentParallelSearch(
            self.coordinator_database, coordinator_config, resource_profile
        )
        if self.config["schema_version"] == G4_FINAL_CONFIG_SCHEMA:
            from .grammar_v3_followup_g4_epoch import (
                GrammarV3FollowupQueueG4FinalEpoch,
            )

            self.queue = GrammarV3FollowupQueueG4FinalEpoch(
                self.coordinator, self.queue_config, self.root
            )
        elif self.config["schema_version"] == G3_EPOCH_CONFIG_SCHEMA:
            from .grammar_v3_followup_g3_epoch import GrammarV3FollowupQueueG3Epoch

            self.queue = GrammarV3FollowupQueueG3Epoch(
                self.coordinator, self.queue_config, self.root
            )
        else:
            self.queue = GrammarV3FollowupQueue(
                self.coordinator, self.queue_config, self.root
            )
        self._initialize_state()
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

    def _validate_reviewed_inputs(self) -> None:
        queue_binding = self.config["queue_config"]
        queue_path = self.root / queue_binding["path"]
        if not queue_path.is_file() or _file_sha(queue_path) != queue_binding["file_sha256"]:
            raise ValueError("service queue config hash mismatch")
        self.queue_config = _load(queue_path)
        revision = self.config["reviewed_report_revisions"][0]
        report_path = self.root / revision["path"]
        if not report_path.is_file() or _file_sha(report_path) != revision["file_sha256"]:
            raise ValueError("service reviewed report revision file mismatch")
        report = _load(report_path)
        report_body = {key: item for key, item in report.items() if key != "content_sha256"}
        if (
            report.get("content_sha256") != revision["content_sha256"]
            or _sha(report_body) != revision["content_sha256"]
            or report.get("evidence_packet_registry_root_sha256")
            != revision["evidence_packet_registry_root_sha256"]
            or report.get("observational_data_opened") is not False
            or report.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("service reviewed report revision content changed")
        queue_report = self.queue_config["pareto_report"]
        if any(
            queue_report[key] != revision[key]
            for key in (
                "path",
                "file_sha256",
                "content_sha256",
                "evidence_packet_registry_root_sha256",
            )
        ):
            raise ValueError("queue config references an unreviewed report revision")
        allowed = {
            (item["path"], item["file_sha256"])
            for item in self.config["evaluator_descriptor_allowlist"]
        }
        configured = {
            (item["descriptor_path"], item["descriptor_file_sha256"])
            for item in self.queue_config["reviewed_evaluators"].values()
        }
        if configured != allowed:
            raise ValueError("queue evaluator descriptors differ from the service allowlist")
        for path_text, expected_hash in allowed:
            path = self.root / path_text
            if not path.is_file() or _file_sha(path) != expected_hash:
                raise ValueError("service evaluator descriptor hash mismatch")
        queue_budget = self.queue_config["budget"]
        service_budget = self.config["budget"]
        if (
            int(queue_budget["maximum_tasks"]) != int(service_budget["maximum_tasks"])
            or float(queue_budget["maximum_wall_seconds"])
            != float(service_budget["maximum_wall_seconds"])
            or int(queue_budget["maximum_database_bytes"])
            > int(service_budget["maximum_service_bytes"])
        ):
            raise ValueError("service changed the original queue budgets")

    def _initialize_state(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            row = connection.execute("SELECT * FROM service_state WHERE singleton=1").fetchone()
            expected_hash = _sha(self.config)
            if row is None:
                connection.execute(
                    "INSERT INTO service_state VALUES (1,?,?,'created',0,0,0,0,0)",
                    (STATUS_SCHEMA, expected_hash),
                )
                self._event(connection, "service_created", {"config_sha256": expected_hash})
            elif row["schema_version"] != STATUS_SCHEMA or row["config_sha256"] != expected_hash:
                if self.config["schema_version"] not in {
                    G3_EPOCH_CONFIG_SCHEMA,
                    G4_FINAL_CONFIG_SCHEMA,
                }:
                    raise ValueError("refusing to resume a changed grammar-v3 follow-up service")
                predecessor = self.config["predecessor_service_epoch"]
                predecessor_path = self.root / predecessor["config_path"]
                if (
                    not predecessor_path.is_file()
                    or _file_sha(predecessor_path) != predecessor["config_file_sha256"]
                    or _sha(_load(predecessor_path)) != predecessor["config_sha256"]
                    or row["config_sha256"] != predecessor["config_sha256"]
                    or row["lifecycle"] not in {"stopped", "waiting_for_reviewed_evaluators"}
                ):
                    raise ValueError("refusing an unbound follow-up service epoch transition")
                completed_root = self.queue.status()["work_records_root_sha256"]
                deferred = [
                    dict(item)
                    for item in connection.execute(
                        "SELECT followup_task_id,followup_lineage_sha256,task_type,candidate_id,state "
                        "FROM deferred_packets ORDER BY followup_task_id"
                    )
                ]
                if (
                    completed_root
                    != predecessor["completed_work_records_root_sha256"]
                    or _sha(deferred) != predecessor["deferred_packet_root_sha256"]
                ):
                    raise ValueError("follow-up service predecessor roots changed")
                connection.execute(
                    "UPDATE service_state SET config_sha256=? WHERE singleton=1",
                    (expected_hash,),
                )
                self._event(
                    connection,
                    "service_config_epoch_advanced",
                    {
                        "predecessor_config_sha256": predecessor["config_sha256"],
                        "config_sha256": expected_hash,
                    },
                )

    @staticmethod
    def _event(
        connection: sqlite3.Connection, event_type: str, payload: dict[str, Any]
    ) -> None:
        connection.execute(
            "INSERT INTO service_events(event_type,payload_json) VALUES (?,?)",
            (event_type, _canonical(payload)),
        )

    def _service_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for base in (self.database, self.coordinator_database)
            for path in (base, Path(str(base) + "-wal"), Path(str(base) + "-shm"))
            if path.is_file()
        )

    def _enforce_disk_budget(self) -> int:
        consumed = self._service_bytes()
        if consumed > int(self.config["budget"]["maximum_service_bytes"]):
            raise RuntimeError("grammar-v3 follow-up service disk budget exhausted")
        return consumed

    def _record_deferred(self, packets: list[dict[str, Any]]) -> tuple[int, int]:
        accepted = duplicate = 0
        with self._connect() as connection:
            for packet in packets:
                row = connection.execute(
                    "SELECT * FROM deferred_packets WHERE followup_task_id=?",
                    (packet["followup_task_id"],),
                ).fetchone()
                values = {
                    "followup_task_id": packet["followup_task_id"],
                    "followup_lineage_sha256": packet["followup_lineage_sha256"],
                    "task_type": packet["task_type"],
                    "candidate_id": packet["candidate_id"],
                    "payload_json": _canonical(packet),
                    "state": "deferred_missing_evaluator",
                }
                if row is None:
                    connection.execute(
                        "INSERT INTO deferred_packets VALUES (?,?,?,?,?,?)",
                        tuple(values.values()),
                    )
                    accepted += 1
                elif dict(row) == values:
                    duplicate += 1
                else:
                    raise ValueError("deferred follow-up packet lineage changed")
        return accepted, duplicate

    def _reconcile_deferred(self, packets: list[dict[str, Any]]) -> int:
        expected = {packet["followup_task_id"]: packet for packet in packets}
        resolved = 0
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM deferred_packets").fetchall()
            for row in rows:
                task_id = row["followup_task_id"]
                if task_id in expected:
                    packet = expected[task_id]
                    if (
                        row["followup_lineage_sha256"]
                        != packet["followup_lineage_sha256"]
                        or row["candidate_id"] != packet["candidate_id"]
                        or row["task_type"] != packet["task_type"]
                        or json.loads(row["payload_json"]) != packet
                    ):
                        raise ValueError("deferred follow-up packet lineage changed")
                elif row["task_type"] in self.queue.evaluators:
                    connection.execute(
                        "DELETE FROM deferred_packets WHERE followup_task_id=?", (task_id,)
                    )
                    resolved += 1
                else:
                    raise ValueError("service epoch lost an unevaluated deferred packet")
        return resolved

    def _cycle(self) -> dict[str, Any]:
        self._validate_reviewed_inputs()
        with self._connect() as connection:
            state = connection.execute("SELECT * FROM service_state").fetchone()
            if int(state["cycle_count"]) >= int(
                self.config["budget"]["maximum_service_cycles"]
            ):
                raise RuntimeError("grammar-v3 follow-up service cycle budget exhausted")
            if bool(state["stop_requested"]):
                raise RuntimeError("grammar-v3 follow-up service is stopped")
        ready = [
            packet
            for packet in self.queue.work_packets
            if packet["task_type"] in self.queue.evaluators
        ]
        deferred = [
            packet
            for packet in self.queue.work_packets
            if packet["task_type"] not in self.queue.evaluators
        ]
        deferred_resolved = self._reconcile_deferred(deferred)
        deferred_accepted, deferred_duplicate = self._record_deferred(deferred)
        admission = self.coordinator.enqueue(ready, lane="cpu", max_attempts=3)
        checkpoint = self.coordinator.checkpoint()
        execution = self.queue.run_bounded(worker_id="grammar-v3-followup-service")
        with self._connect() as connection:
            connection.execute(
                "UPDATE service_state SET lifecycle=?,cycle_count=cycle_count+1,"
                "admitted_count=admitted_count+?,processed_count=processed_count+?,"
                "deferred_count=? WHERE singleton=1",
                (
                    "waiting_for_reviewed_evaluators" if deferred else "idle",
                    admission["accepted"],
                    execution["executed"],
                    len(deferred),
                ),
            )
            self._event(
                connection,
                "service_cycle",
                {
                    "admission": admission,
                    "executed": execution["executed"],
                    "deferred_accepted": deferred_accepted,
                    "deferred_duplicate": deferred_duplicate,
                    "deferred_resolved": deferred_resolved,
                    "checkpoint_sha256": checkpoint["content_sha256"],
                },
            )
        self._enforce_disk_budget()
        return {
            "admission": admission,
            "executed": execution["executed"],
            "deferred_accepted": deferred_accepted,
            "deferred_duplicate": deferred_duplicate,
            "deferred_resolved": deferred_resolved,
            "status": self.status(),
        }

    def start(self) -> dict[str, Any]:
        with self._connect() as connection:
            state = connection.execute("SELECT * FROM service_state").fetchone()
            if state["lifecycle"] != "created":
                raise ValueError("service start is valid only for a new lifecycle")
            connection.execute(
                "UPDATE service_state SET lifecycle='running',stop_requested=0 WHERE singleton=1"
            )
            self._event(connection, "service_started", {})
        return self._cycle()

    def stop(self) -> dict[str, Any]:
        checkpoint = self.coordinator.checkpoint()
        with self._connect() as connection:
            connection.execute(
                "UPDATE service_state SET lifecycle='stopped',stop_requested=1 WHERE singleton=1"
            )
            self._event(
                connection, "service_stopped", {"checkpoint_sha256": checkpoint["content_sha256"]}
            )
        return self.status()

    def resume(self) -> dict[str, Any]:
        with self._connect() as connection:
            state = connection.execute("SELECT * FROM service_state").fetchone()
            if state["lifecycle"] not in {"stopped", "waiting_for_reviewed_evaluators"}:
                raise ValueError("service resume requires a stopped or waiting lifecycle")
            if int(state["cycle_count"]) >= int(
                self.config["budget"]["maximum_service_cycles"]
            ):
                raise RuntimeError("grammar-v3 follow-up service cycle budget exhausted")
            connection.execute(
                "UPDATE service_state SET lifecycle='running',stop_requested=0 WHERE singleton=1"
            )
            self._event(connection, "service_resumed", {})
        return self._cycle()

    def status(self) -> dict[str, Any]:
        queue_status = self.queue.status()
        with self._connect() as connection:
            state = dict(connection.execute("SELECT * FROM service_state").fetchone())
            deferred_rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT followup_task_id,followup_lineage_sha256,task_type,candidate_id,state "
                    "FROM deferred_packets ORDER BY followup_task_id"
                )
            ]
            event_count = connection.execute("SELECT COUNT(*) FROM service_events").fetchone()[0]
        packet_states = Counter(queue_status["work_state_counts"])
        packet_states["deferred_missing_evaluator"] = len(deferred_rows)
        body = {
            "schema_version": STATUS_SCHEMA,
            "lifecycle": state["lifecycle"],
            "cycle_count": int(state["cycle_count"]),
            "admitted_count": int(state["admitted_count"]),
            "processed_count": int(state["processed_count"]),
            "deferred_count": len(deferred_rows),
            "packet_state_counts": dict(sorted(packet_states.items())),
            "queue_registry_root_sha256": queue_status["queue_registry_root_sha256"],
            "completed_work_records_root_sha256": queue_status[
                "work_records_root_sha256"
            ],
            "deferred_packets": deferred_rows,
            "deferred_packet_root_sha256": _sha(deferred_rows),
            "reviewed_evaluator_invocation_count": queue_status[
                "reviewed_evaluator_invocation_count"
            ],
            "missing_evaluator_executions": queue_status["missing_evaluator_count"],
            "candidate_scientific_decisions_changed": queue_status[
                "candidate_scientific_decisions_changed"
            ],
            "coordinator_checkpoint_sequence": queue_status["checkpoint_sequence"],
            "service_event_count": int(event_count),
            "disk_bytes": self._enforce_disk_budget(),
            "budget": dict(self.config["budget"]),
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "Only hash-allowlisted evaluator work is admitted. Missing-evaluator packets are "
                "checkpointed as deferred and cannot alter scientific candidate decisions."
            ),
        }
        return {**body, "content_sha256": _sha(body)}

    def export(self) -> dict[str, Any]:
        status = self.status()
        body = {
            key: item
            for key, item in status.items()
            if key not in {"content_sha256", "disk_bytes"}
        }
        return {**body, "content_sha256": _sha(body)}
