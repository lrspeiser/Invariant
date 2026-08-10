from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-followup-queue-config-1.0"
PAYLOAD_SCHEMA = "sigma-grammar-v3-followup-work-packet-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-followup-work-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-followup-queue-status-1.0"

TASK_SPECS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "AETHER_K1234_PARAMETER_CELL": [
        (
            "aether_nonlinear_twist_energy",
            (
                "generic_nonlinear_hamiltonian_stability",
                "hypersurface_orthogonal_aether",
            ),
        )
    ],
    "KESSENCE_G2_CONVEX": [
        (
            "g2_global_boundary_dirac_contract",
            ("complete_distributed_dirac_boundary_contract",),
        ),
        ("g2_global_positive_mass", ("global_positive_energy",)),
    ],
    "CUBIC_HORNDESKI_G3_WEAK_CELL": [
        (
            "g3_uniform_interval_cell",
            (
                "principal_common_cone_center",
                "uniform_weak_cell_principal_common_cone",
            ),
        ),
        (
            "g3_global_lapse_dirac_contract",
            ("candidate_specific_distributed_dirac",),
        ),
    ],
    "CONFORMAL_G4_PHI_SCALAR_TENSOR": [
        (
            "g4_global_lapse_invertibility",
            ("global_lapse_operator_invertibility",),
        ),
        ("g4_global_positive_energy", ("global_positive_energy",)),
    ],
}
REVIEWED_TASK_TYPES = tuple(
    task_type
    for specifications in TASK_SPECS.values()
    for task_type, _ in specifications
)

STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS grammar_v3_followup_adapter (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  pareto_report_content_sha256 TEXT NOT NULL,
  queue_registry_root_sha256 TEXT NOT NULL
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


def _reject_scalar_truth_score(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "truth_score",
                "overall_score",
                "composite_score",
                "probability_of_truth",
            }:
                raise ValueError("scalar truth or composite score is forbidden")
            _reject_scalar_truth_score(item)
    elif isinstance(value, list):
        for item in value:
            _reject_scalar_truth_score(item)


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "pareto_report",
        "reviewed_task_types",
        "reviewed_evaluators",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 follow-up queue config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 follow-up queue eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 follow-up queue enabled paid LLM calls")
    if config.get("reviewed_task_types") != list(REVIEWED_TASK_TYPES):
        raise ValueError("grammar-v3 reviewed follow-up task allowlist changed")
    if config.get("reviewed_evaluators") != {}:
        raise ValueError("unreviewed follow-up evaluator descriptor was supplied")
    budget = config.get("budget", {})
    if (
        set(budget) != {"maximum_tasks", "maximum_wall_seconds", "maximum_database_bytes"}
        or int(budget["maximum_tasks"]) != 10
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 4096 <= int(budget["maximum_database_bytes"]) <= 64 * 1024 * 1024
    ):
        raise ValueError("grammar-v3 follow-up queue budget is invalid")


class GrammarV3FollowupQueue:
    """Durable reviewed-task queue derived from the immutable grammar-v3 Pareto report."""

    def __init__(
        self,
        coordinator: PersistentParallelSearch,
        config: dict[str, Any],
        project_root: str | Path,
    ) -> None:
        _validate_config(config)
        self.coordinator = coordinator
        self.config = config
        self.root = Path(project_root).resolve()
        if self.coordinator.database.name.lower() == "campaign-v1-live.sqlite":
            raise ValueError("refusing to use the live campaign watchdog database")
        if (
            coordinator.config.get("external_paid_llm_calls") is not False
            or int(coordinator.config["budget"]["maximum_tasks"]) != 10
            or float(coordinator.config["budget"]["maximum_wall_seconds"])
            > float(config["budget"]["maximum_wall_seconds"])
            or int(coordinator.config["queue"]["maximum_pending_work"]) < 10
        ):
            raise ValueError("coordinator does not preserve follow-up queue bounds")
        descriptor = config["pareto_report"]
        self.report_path = self.root / descriptor["path"]
        if not self.report_path.is_file() or _file_sha(self.report_path) != descriptor[
            "file_sha256"
        ]:
            raise ValueError("bound grammar-v3 Pareto report file mismatch")
        self.report = _load(self.report_path)
        body = {key: item for key, item in self.report.items() if key != "content_sha256"}
        if (
            self.report.get("content_sha256") != descriptor["content_sha256"]
            or _sha(body) != descriptor["content_sha256"]
            or self.report.get("evidence_packet_registry_root_sha256")
            != descriptor["evidence_packet_registry_root_sha256"]
            or self.report.get("candidate_decision_counts") != {"blocked": 6}
            or self.report.get("observational_data_opened") is not False
            or self.report.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
            or self.report.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("bound grammar-v3 Pareto report content or seals changed")
        _reject_scalar_truth_score(self.report)
        self.work_packets = self._build_work_packets()
        self.queue_registry_root_sha256 = _sha(self.work_packets)
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _build_work_packets(self) -> list[dict[str, Any]]:
        axes = self.report["priority_axes"]
        if axes != [
            "formal_pass_count",
            "candidate_evidence_packet_count",
            "source_lineage_depth",
            "blocker_reduction_margin",
        ]:
            raise ValueError("grammar-v3 Pareto axes changed")
        packets = []
        ordinals = set()
        for candidate in self.report["pareto_follow_up_queue"]:
            family = candidate["family_id"]
            if family not in TASK_SPECS or candidate["candidate_decision"] != "blocked":
                raise ValueError("Pareto candidate lacks a reviewed blocked-family task map")
            blockers_by_gate: dict[str, list[dict[str, Any]]] = {}
            for blocker in candidate["blocker_taxonomy"]:
                blockers_by_gate.setdefault(blocker["gate_id"], []).append(blocker)
            axis_values = {axis: candidate[axis] for axis in axes}
            blocker_root = _sha(candidate["blocker_taxonomy"])
            for task_type, target_gate_ids in TASK_SPECS[family]:
                targets = [
                    blocker
                    for gate_id in target_gate_ids
                    for blocker in blockers_by_gate.get(gate_id, [])
                ]
                if {item["gate_id"] for item in targets} != set(target_gate_ids):
                    raise ValueError(f"reviewed task {task_type} lost its exact blocker premise")
                task_body = {
                    "pareto_report_content_sha256": self.report["content_sha256"],
                    "evidence_packet_registry_root_sha256": self.report[
                        "evidence_packet_registry_root_sha256"
                    ],
                    "candidate_id": candidate["formula_id"],
                    "seed_lineage_sha256": candidate["seed_lineage_sha256"],
                    "action_sha256": candidate["action_sha256"],
                    "family_id": family,
                    "task_type": task_type,
                    "pareto_front": candidate["pareto_front"],
                    "pareto_axes": axes,
                    "pareto_axis_values": axis_values,
                    "candidate_evidence_packet_root_sha256": candidate[
                        "evidence_packet_root_sha256"
                    ],
                    "candidate_blocker_taxonomy_root_sha256": blocker_root,
                    "target_blockers": targets,
                    "reviewed_evaluator_binding_sha256": None,
                    "data_eligibility": dict(ELIGIBILITY),
                }
                lineage = _sha(task_body)
                ordinal = int(candidate["pareto_front"]) * 10**15 + int(lineage[:12], 16)
                if ordinal in ordinals:
                    raise ValueError("grammar-v3 follow-up deterministic ordinal collision")
                ordinals.add(ordinal)
                packets.append(
                    {
                        "schema_version": PAYLOAD_SCHEMA,
                        "ordinal": ordinal,
                        "followup_task_id": "G3F-" + lineage[:24],
                        "followup_lineage_sha256": lineage,
                        **task_body,
                    }
                )
        packets.sort(key=lambda item: (item["pareto_front"], item["candidate_id"], item["task_type"]))
        if len(packets) != 10:
            raise ValueError("grammar-v3 reviewed follow-up queue must contain exactly ten tasks")
        _reject_scalar_truth_score(packets)
        return packets

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "config_sha256": _sha(self.config),
            "pareto_report_content_sha256": self.report["content_sha256"],
            "queue_registry_root_sha256": self.queue_registry_root_sha256,
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SCHEMA)
            row = connection.execute(
                "SELECT * FROM grammar_v3_followup_adapter WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_followup_adapter VALUES (1,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume a changed grammar-v3 follow-up queue")
            rows = connection.execute("SELECT payload_json FROM work").fetchall()
            if any(json.loads(row[0]).get("schema_version") != PAYLOAD_SCHEMA for row in rows):
                raise ValueError("follow-up adapter requires a dedicated coordinator database")

    def _database_bytes(self) -> int:
        path = self.coordinator.database
        return sum(
            candidate.stat().st_size
            for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
            if candidate.is_file()
        )

    def _enforce_disk_budget(self) -> int:
        consumed = self._database_bytes()
        if consumed > int(self.config["budget"]["maximum_database_bytes"]):
            raise RuntimeError("grammar-v3 follow-up queue disk budget exhausted")
        return consumed

    def enqueue(self) -> dict[str, Any]:
        self._enforce_disk_budget()
        admitted = self.coordinator.enqueue(self.work_packets, lane="cpu", max_attempts=3)
        checkpoint = self.coordinator.checkpoint()
        return {
            **admitted,
            "requested": len(self.work_packets),
            "queue_registry_root_sha256": self.queue_registry_root_sha256,
            "checkpoint_sha256": checkpoint["content_sha256"],
        }

    def _expected_payload(self, lease: WorkLease) -> dict[str, Any]:
        expected = {packet["followup_task_id"]: packet for packet in self.work_packets}
        payload = lease.payload
        task_id = str(payload.get("followup_task_id"))
        if task_id not in expected or payload != expected[task_id]:
            raise ValueError("leased grammar-v3 follow-up packet or lineage mismatch")
        if lease.ordinal != payload["ordinal"]:
            raise ValueError("leased grammar-v3 follow-up ordinal mismatch")
        return payload

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        payload = self._expected_payload(lease)
        if payload["task_type"] not in REVIEWED_TASK_TYPES:
            raise ValueError("leased follow-up task type is not reviewed")
        result_body = {
            "schema_version": RESULT_SCHEMA,
            "work_id": lease.work_id,
            "followup_task_id": payload["followup_task_id"],
            "followup_lineage_sha256": payload["followup_lineage_sha256"],
            "candidate_id": payload["candidate_id"],
            "task_type": payload["task_type"],
            "decision": "blocked",
            "blocker": "reviewed_evaluator_missing",
            "evaluator_invoked": False,
            "scientific_candidate_decision_changed": False,
            "target_blocker_root_sha256": _sha(payload["target_blockers"]),
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**result_body, "content_sha256": _sha(result_body)}

    def run_bounded(self, *, worker_id: str = "grammar-v3-followup") -> dict[str, Any]:
        recovered_now = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(recovered_now[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        for _ in range(10):
            self._enforce_disk_budget()
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("grammar-v3 follow-up lease was lost")
            except Exception as error:
                self.coordinator.fail(lease, worker_id, f"{type(error).__name__}: {error}")
                raise
            executed += 1
        checkpoint = self.coordinator.checkpoint()
        return {
            "executed": executed,
            "recovered": recovered,
            "checkpoint_sha256": checkpoint["content_sha256"],
            "status": self.status(),
        }

    def status(self) -> dict[str, Any]:
        expected = {packet["followup_task_id"]: packet for packet in self.work_packets}
        records = []
        states: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT work_id,ordinal,seed,payload_json,state,attempt,result_json,error_text "
                "FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            task_id = payload.get("followup_task_id")
            if task_id not in expected or payload != expected[task_id]:
                raise ValueError("stored grammar-v3 follow-up packet is unregistered")
            state = str(row["state"])
            states[state] += 1
            result_sha = None
            if row["result_json"] is not None:
                result = json.loads(row["result_json"])
                body = {key: item for key, item in result.items() if key != "content_sha256"}
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("followup_lineage_sha256")
                    != payload["followup_lineage_sha256"]
                    or result.get("scientific_candidate_decision_changed") is not False
                ):
                    raise ValueError("stored grammar-v3 follow-up result lineage mismatch")
                decisions[result["decision"]] += 1
                result_sha = result["content_sha256"]
            records.append(
                {
                    "work_id": row["work_id"],
                    "followup_task_id": task_id,
                    "followup_lineage_sha256": payload["followup_lineage_sha256"],
                    "candidate_id": payload["candidate_id"],
                    "task_type": payload["task_type"],
                    "pareto_front": payload["pareto_front"],
                    "pareto_axis_values": payload["pareto_axis_values"],
                    "target_blocker_root_sha256": _sha(payload["target_blockers"]),
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
            "pareto_report_file_sha256": self.config["pareto_report"]["file_sha256"],
            "pareto_report_content_sha256": self.report["content_sha256"],
            "evidence_packet_registry_root_sha256": self.report[
                "evidence_packet_registry_root_sha256"
            ],
            "queue_registry_root_sha256": self.queue_registry_root_sha256,
            "reviewed_task_types": list(REVIEWED_TASK_TYPES),
            "task_count": len(self.work_packets),
            "work_state_counts": dict(sorted(states.items())),
            "followup_decision_counts": dict(sorted(decisions.items())),
            "missing_evaluator_count": decisions["blocked"],
            "candidate_scientific_decisions_changed": 0,
            "work_records": records,
            "work_records_root_sha256": _sha(records),
            "checkpoint_sequence": telemetry["checkpoint_sequence"],
            "recovered_leases": telemetry["recovered_leases"],
            "database_bytes": self._enforce_disk_budget(),
            "budget": dict(self.config["budget"]),
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "These packets order reviewed prerequisite work only. Missing evaluators remain "
                "blocked and cannot alter any scientific candidate decision."
            ),
        }
        _reject_scalar_truth_score(body)
        return {**body, "content_sha256": _sha(body)}


def portable_followup_status(status: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in status.items() if key not in {"content_sha256", "database_bytes"}}
    return {**body, "content_sha256": _sha(body)}
