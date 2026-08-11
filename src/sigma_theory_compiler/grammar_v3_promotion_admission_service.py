from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .grammar_v3_formal_preflight_service import GrammarV3FormalPreflightService
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-promotion-admission-config-1.0"
PAYLOAD_SCHEMA = "sigma-grammar-v3-promotion-admission-work-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-promotion-admission-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-promotion-admission-status-1.0"

STATE_SQL = """
CREATE TABLE IF NOT EXISTS grammar_v3_promotion_admission_service (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  immutable_config_sha256 TEXT NOT NULL,
  eligible_candidate_registry_root_sha256 TEXT NOT NULL,
  admission_adapter_registry_root_sha256 TEXT NOT NULL,
  preflight_status_content_sha256 TEXT NOT NULL
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
        raise TypeError(f"{path.name} must contain an object")
    return value


def _validate(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "execution_enabled",
        "preflight_status",
        "preflight_config",
        "coordinator_config",
        "resource_profile",
        "family_queue_adapters",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 promotion-admission config is invalid")
    if not isinstance(config.get("execution_enabled"), bool):
        raise TypeError("promotion-admission execution_enabled must be boolean")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("promotion-admission eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("promotion-admission enabled paid LLM calls")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_tasks",
        "chunk_size",
        "maximum_attempts",
        "maximum_wall_seconds",
        "maximum_disk_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_tasks"]) != 162
        or int(budget["chunk_size"]) != 32
        or not 1 <= int(budget["maximum_attempts"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("promotion-admission budget is invalid or unbounded")
    families = {item["family_id"] for item in config["family_queue_adapters"]}
    if families != {
        "AETHER_K1234_PARAMETER_CELL",
        "KESSENCE_G2_CONVEX",
        "CUBIC_HORNDESKI_G3_WEAK_CELL",
    }:
        raise ValueError("promotion-admission queue adapter registry is incomplete")


class GrammarV3PromotionAdmissionService:
    """Durably admits only exact preflight-pass receipts to sealed formal queues."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        repo_root: str | Path,
        *,
        unavailable_admission_families: frozenset[str] = frozenset(),
    ) -> None:
        _validate(config)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.preflight_status = self._bound_json("preflight_status", content=True)
        self.preflight_config = self._bound_json("preflight_config")
        self.base_coordinator = self._bound_json("coordinator_config")
        self.resource_profile = self._bound_json("resource_profile")
        self._validate_preflight_status()
        self.adapter_by_family = self._queue_adapters()
        unknown = unavailable_admission_families - set(self.adapter_by_family)
        if unknown:
            raise ValueError("unknown unavailable promotion-admission family")
        self.unavailable_admission_families = unavailable_admission_families
        self.admission_adapter_registry_root_sha256 = _sha(
            [
                {
                    **descriptor,
                    "state": (
                        "missing"
                        if descriptor["family_id"] in unavailable_admission_families
                        else "reviewed_bound"
                    ),
                }
                for descriptor in config["family_queue_adapters"]
            ]
        )
        self.preflight = GrammarV3FormalPreflightService(
            self.directory / "preflight-attestation",
            self.preflight_config,
            self.repo_root,
        )
        if (
            self.preflight.candidate_registry_root_sha256
            != self.preflight_status["candidate_registry_root_sha256"]
            or self.preflight.callback_registry_root_sha256
            != self.preflight_status["callback_registry_root_sha256"]
        ):
            raise ValueError("promotion-admission preflight registries changed")
        self.work_items = self._eligible_work_items()
        self.eligible_candidate_registry_root_sha256 = _sha(
            [
                [
                    item["candidate_id"],
                    item["typed_action_ir_sha256"],
                    item["preflight_result_sha256"],
                    item["preflight_adapter_binding_sha256"],
                    item["admission_adapter_binding_sha256"],
                ]
                for item in self.work_items
            ]
        )
        self.coordinator = PersistentParallelSearch(
            self.directory / "promotion-admission.sqlite",
            self._coordinator_config(),
            self.resource_profile,
        )
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _path(self, binding: dict[str, Any], label: str) -> Path:
        path = (self.repo_root / binding["path"]).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(f"promotion-admission {label} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"promotion-admission {label} file hash mismatch")
        return path

    def _bound_json(self, key: str, *, content: bool = False) -> dict[str, Any]:
        binding = self.config[key]
        value = _load(self._path(binding, key))
        if content:
            body = {name: item for name, item in value.items() if name != "content_sha256"}
            if value.get("content_sha256") != binding["content_sha256"] or _sha(body) != binding[
                "content_sha256"
            ]:
                raise ValueError(f"promotion-admission {key} content hash mismatch")
        return value

    def _validate_preflight_status(self) -> None:
        status = self.preflight_status
        if (
            status.get("candidate_count") != 163
            or status.get("decision_counts") != {"blocked": 1, "pass": 162}
            or status.get("family_decision_counts")
            != {
                "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
                "CONFORMAL_G4_PHI_SCALAR_TENSOR": {"blocked": 1},
                "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
                "KESSENCE_G2_CONVEX": {"pass": 2},
            }
            or status.get("expensive_adm_or_global_energy_run") is not False
            or status.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
            or status.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("promotion-admission preflight status is ineligible")

    def _queue_adapters(self) -> dict[str, dict[str, Any]]:
        adapters = {}
        for descriptor in self.config["family_queue_adapters"]:
            if set(descriptor) != {
                "family_id",
                "admission_adapter_id",
                "target_task_type",
                "required_bindings",
                "downstream_expensive_execution_enabled",
            }:
                raise ValueError("promotion-admission adapter descriptor fields are invalid")
            if (
                descriptor["required_bindings"]
                != [
                    "candidate_id",
                    "typed_action_ir_sha256",
                    "preflight_result_sha256",
                    "preflight_adapter_binding_sha256",
                ]
                or descriptor["downstream_expensive_execution_enabled"] is not False
            ):
                raise ValueError("promotion-admission adapter opens an unreviewed downstream path")
            adapters[descriptor["family_id"]] = descriptor
        return adapters

    def _preflight_adapter_bindings(self) -> dict[str, str]:
        return {
            descriptor["family_id"]: _sha(descriptor)
            for descriptor in self.preflight_config["reviewed_adapters"]
        }

    def _eligible_work_items(self) -> list[dict[str, Any]]:
        preflight_bindings = self._preflight_adapter_bindings()
        pass_families = set(self.adapter_by_family)
        items = []
        for source in self.preflight.work_items:
            if source["family_id"] not in pass_families:
                continue
            fake = WorkLease(
                work_id="attestation",
                ordinal=int(source["ordinal"]),
                lane="cpu",
                seed=0,
                attempt=1,
                max_attempts=1,
                payload=source,
            )
            result = self.preflight.execute_lease(fake)
            if result["decision"] != "pass":
                raise ValueError("preflight artifact pass candidate does not independently replay")
            adapter = self.adapter_by_family[source["family_id"]]
            body = {
                "schema_version": PAYLOAD_SCHEMA,
                "ordinal": len(items),
                "candidate_id": source["candidate_id"],
                "typed_action_ir_sha256": source["typed_action_ir_sha256"],
                "preflight_result_sha256": result["content_sha256"],
                "preflight_input_lineage_sha256": source["input_lineage_sha256"],
                "preflight_adapter_binding_sha256": preflight_bindings[source["family_id"]],
                "admission_adapter_binding_sha256": _sha(adapter),
                "family_id": source["family_id"],
                "target_task_type": adapter["target_task_type"],
                "preflight_status_content_sha256": self.preflight_status["content_sha256"],
                "data_eligibility": dict(ELIGIBILITY),
            }
            items.append({**body, "input_lineage_sha256": _sha(body)})
        if len(items) != int(self.config["budget"]["maximum_tasks"]):
            raise ValueError("promotion-admission eligible candidate count changed")
        return items

    def _coordinator_config(self) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_coordinator))
        maximum = int(self.config["budget"]["maximum_tasks"])
        config["queue"].update(
            maximum_pending_work=maximum,
            maximum_attempts=int(self.config["budget"]["maximum_attempts"]),
            lease_seconds=int(self.config["budget"]["maximum_wall_seconds"]),
            checkpoint_every_completions=int(self.config["budget"]["chunk_size"]),
        )
        config["budget"] = {
            "maximum_tasks": maximum,
            "maximum_wall_seconds": float(self.config["budget"]["maximum_wall_seconds"]),
        }
        config["cpu"]["maximum_workers"] = min(4, maximum)
        config["external_paid_llm_calls"] = False
        return config

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "immutable_config_sha256": _sha(self.config),
            "eligible_candidate_registry_root_sha256": self.eligible_candidate_registry_root_sha256,
            "admission_adapter_registry_root_sha256": self.admission_adapter_registry_root_sha256,
            "preflight_status_content_sha256": self.preflight_status["content_sha256"],
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SQL)
            row = connection.execute(
                "SELECT * FROM grammar_v3_promotion_admission_service WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_promotion_admission_service VALUES (1,?,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume changed promotion-admission service")
            for row in connection.execute("SELECT payload_json FROM work"):
                if json.loads(row[0]).get("schema_version") != PAYLOAD_SCHEMA:
                    raise ValueError("promotion-admission requires a dedicated coordinator DB")

    def _disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def _enforce_budget(self, started: float | None = None) -> None:
        if self._disk_bytes() > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("promotion-admission disk budget exhausted")
        if started is not None and time.monotonic() - started > float(
            self.config["budget"]["maximum_wall_seconds"]
        ):
            raise TimeoutError("promotion-admission wall budget exhausted")

    def enqueue(self) -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 promotion-admission is disabled by config")
        self._enforce_budget()
        admitted = self.coordinator.enqueue(
            self.work_items,
            lane="cpu",
            max_attempts=int(self.config["budget"]["maximum_attempts"]),
        )
        checkpoint = self.coordinator.checkpoint()
        return {
            **admitted,
            "requested": len(self.work_items),
            "eligible_candidate_registry_root_sha256": self.eligible_candidate_registry_root_sha256,
            "checkpoint_sha256": checkpoint["content_sha256"],
        }

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        if lease.ordinal >= len(self.work_items) or lease.payload != self.work_items[lease.ordinal]:
            raise ValueError("promotion-admission leased payload binding changed")
        payload = lease.payload
        missing = payload["family_id"] in self.unavailable_admission_families
        decision = "blocked" if missing else "pass"
        blocker = "reviewed_promotion_admission_adapter_missing" if missing else None
        body = {
            "schema_version": RESULT_SCHEMA,
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "preflight_result_sha256": payload["preflight_result_sha256"],
            "preflight_adapter_binding_sha256": payload["preflight_adapter_binding_sha256"],
            "admission_adapter_binding_sha256": payload["admission_adapter_binding_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "family_id": payload["family_id"],
            "target_task_type": payload["target_task_type"],
            "decision": decision,
            "queue_state": "ready_for_reviewed_evaluator" if not missing else "blocked",
            "blocker": blocker,
            "downstream_expensive_execution_started": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self, *, worker_id: str = "grammar-v3-promotion-admission") -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 promotion-admission is disabled by config")
        started = time.monotonic()
        current = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(current[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        for _ in range(int(self.config["budget"]["maximum_tasks"])):
            self._enforce_budget(started)
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("promotion-admission lease was lost")
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
        work_counts: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        family_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        queue_counts: Counter[str] = Counter()
        queue_records: defaultdict[str, list[list[str]]] = defaultdict(list)
        records = []
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT ordinal,payload_json,state,attempt,result_json,error_text FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            ordinal = int(row["ordinal"])
            payload = json.loads(row["payload_json"])
            if ordinal >= len(self.work_items) or payload != self.work_items[ordinal]:
                raise ValueError("stored promotion-admission payload was tampered")
            work_counts[str(row["state"])] += 1
            result_sha = None
            decision = None
            blocker = None
            if row["result_json"]:
                result = json.loads(row["result_json"])
                body = {key: value for key, value in result.items() if key != "content_sha256"}
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("candidate_id") != payload["candidate_id"]
                    or result.get("typed_action_ir_sha256") != payload["typed_action_ir_sha256"]
                    or result.get("preflight_result_sha256") != payload["preflight_result_sha256"]
                    or result.get("input_lineage_sha256") != payload["input_lineage_sha256"]
                ):
                    raise ValueError("stored promotion-admission result binding changed")
                decision = result["decision"]
                decisions[decision] += 1
                family_counts[payload["family_id"]][decision] += 1
                if decision == "pass":
                    queue_counts[payload["target_task_type"]] += 1
                    queue_records[payload["target_task_type"]].append(
                        [
                            payload["candidate_id"],
                            payload["typed_action_ir_sha256"],
                            payload["preflight_result_sha256"],
                            result["content_sha256"],
                        ]
                    )
                result_sha = result["content_sha256"]
                blocker = result["blocker"]
            records.append(
                {
                    "candidate_id": payload["candidate_id"],
                    "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
                    "preflight_result_sha256": payload["preflight_result_sha256"],
                    "family_id": payload["family_id"],
                    "target_task_type": payload["target_task_type"],
                    "state": row["state"],
                    "attempt": int(row["attempt"]),
                    "decision": decision,
                    "result_sha256": result_sha,
                    "blocker": blocker,
                    "error_text": row["error_text"],
                }
            )
        body = {
            "schema_version": STATUS_SCHEMA,
            "execution_enabled": self.config["execution_enabled"],
            "preflight_status_binding": self.config["preflight_status"],
            "preflight_config_binding": self.config["preflight_config"],
            "preflight_candidate_count": 163,
            "preflight_pass_count": 162,
            "preflight_blocked_excluded_count": 1,
            "eligible_candidate_count": len(self.work_items),
            "eligible_candidate_registry_root_sha256": self.eligible_candidate_registry_root_sha256,
            "admission_adapter_registry_root_sha256": self.admission_adapter_registry_root_sha256,
            "work_state_counts": dict(sorted(work_counts.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "family_decision_counts": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(family_counts.items())
            },
            "target_queue_counts": dict(sorted(queue_counts.items())),
            "target_queue_registry_roots": {
                queue: _sha(queue_records[queue]) for queue in sorted(queue_records)
            },
            "record_registry_root_sha256": _sha(records),
            "checkpoint_sequence": self.coordinator.telemetry()["checkpoint_sequence"],
            "disk_bytes": self._disk_bytes(),
            "budget": self.config["budget"],
            "downstream_expensive_execution_started": False,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}


def portable_status(status: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in status.items()
        if key not in {"content_sha256", "checkpoint_sequence", "disk_bytes"}
    }
    return {**body, "content_sha256": _sha(body)}
