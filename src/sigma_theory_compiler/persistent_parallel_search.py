from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-persistent-parallel-search-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _capability(value: str) -> tuple[int, int]:
    major, _, minor = value.partition(".")
    return int(major), int(minor or 0)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported persistent-search schema_version")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("external paid LLM calls must remain disabled")
    positive = {
        "queue.maximum_pending_work": config["queue"]["maximum_pending_work"],
        "queue.maximum_attempts": config["queue"]["maximum_attempts"],
        "queue.lease_seconds": config["queue"]["lease_seconds"],
        "budget.maximum_tasks": config["budget"]["maximum_tasks"],
        "budget.maximum_wall_seconds": config["budget"]["maximum_wall_seconds"],
        "gpu.estimated_bytes_per_candidate": config["gpu"]["estimated_bytes_per_candidate"],
        "gpu.maximum_batch_candidates": config["gpu"]["maximum_batch_candidates"],
        "supervisor.worker_poll_seconds": config["supervisor"]["worker_poll_seconds"],
        "supervisor.telemetry_interval_seconds": config["supervisor"][
            "telemetry_interval_seconds"
        ],
        "supervisor.refill_interval_seconds": config["supervisor"][
            "refill_interval_seconds"
        ],
        "supervisor.maximum_wall_seconds_per_run": config["supervisor"][
            "maximum_wall_seconds_per_run"
        ],
        "supervisor.shutdown_grace_seconds": config["supervisor"]["shutdown_grace_seconds"],
        "supervisor.maximum_telemetry_bytes": config["supervisor"][
            "maximum_telemetry_bytes"
        ],
    }
    if any(float(value) <= 0 for value in positive.values()):
        raise ValueError("persistent-search resource limits must be positive")
    nonnegative = {
        "queue.checkpoint_every_completions": config["queue"][
            "checkpoint_every_completions"
        ],
        "cpu.maximum_workers": config["cpu"]["maximum_workers"],
        "gpu.minimum_memory_mib": config["gpu"]["minimum_memory_mib"],
        "gpu.reserve_memory_mib": config["gpu"]["reserve_memory_mib"],
        "supervisor.cpu_workers": config["supervisor"]["cpu_workers"],
        "supervisor.gpu_workers": config["supervisor"]["gpu_workers"],
        "supervisor.maximum_process_restarts": config["supervisor"][
            "maximum_process_restarts"
        ],
    }
    if any(int(value) < 0 for value in nonnegative.values()):
        raise ValueError("persistent-search counts must be nonnegative")


def plan_parallel_capacity(
    resource_profile: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Make a deterministic CPU/GPU lane and batch plan from measured hardware."""

    _validate_config(config)
    hardware = resource_profile["hardware"]
    production = resource_profile["production_lanes"]
    safety = resource_profile["safety"]
    cpu = config["cpu"]
    gpu = config["gpu"]
    logical = int(hardware["logical_processors"])
    reserve = int(safety["reserve_cpu_cores_for_os_database_and_gpu_feeder"])
    measured_cpu = int(production["cpu_symbolic"]["sustained_workers"])
    cpu_workers = min(
        int(cpu["maximum_workers"]), measured_cpu, max(1, logical - reserve)
    )

    gpu_capability = _capability(str(hardware["cuda_compute_capability"]))
    minimum_capability = _capability(str(gpu["minimum_compute_capability"]))
    gpu_memory = int(hardware["gpu_memory_mib"])
    gpu_available = (
        gpu_capability >= minimum_capability
        and gpu_memory >= int(gpu["minimum_memory_mib"])
    )
    usable_bytes = max(
        0,
        gpu_memory - int(gpu["reserve_memory_mib"]),
    ) * 1024**2
    memory_batch = usable_bytes // int(gpu["estimated_bytes_per_candidate"])
    gpu_batch = (
        min(int(gpu["maximum_batch_candidates"]), int(memory_batch))
        if gpu_available
        else 0
    )
    gpu_workers = 1 if gpu_available else 0
    measured_gpu = resource_profile.get("measured_gpu_benchmark", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_profile_sha256": _sha(resource_profile),
        "cpu": {
            "workers": cpu_workers,
            "logical_processors": logical,
            "reserved_processors": reserve,
            "measured_sustained_workers": measured_cpu,
            "task_types": list(production["cpu_symbolic"]["task_types"]),
        },
        "gpu": {
            "available": gpu_available,
            "workers": gpu_workers,
            "device": hardware["gpu"],
            "memory_mib": gpu_memory,
            "compute_capability": str(hardware["cuda_compute_capability"]),
            "batch_candidates": gpu_batch,
            "measured_candidates_per_second": float(
                measured_gpu.get("candidates_per_second", 0.0)
            ),
            "single_owner_reason": (
                "one CUDA owner batches candidates without duplicating VRAM"
            ),
        },
        "simultaneous_lane_owners": cpu_workers + gpu_workers,
        "paid_llm_workers": 0,
    }


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS execution (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  config_json TEXT NOT NULL,
  config_sha256 TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  deadline_utc TEXT NOT NULL,
  submitted INTEGER NOT NULL DEFAULT 0,
  completed INTEGER NOT NULL DEFAULT 0,
  failed INTEGER NOT NULL DEFAULT 0,
  recovered INTEGER NOT NULL DEFAULT 0,
  checkpoint_sequence INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS work (
  work_id TEXT PRIMARY KEY,
  ordinal INTEGER NOT NULL,
  lane TEXT NOT NULL CHECK(lane IN ('cpu','gpu')),
  seed INTEGER NOT NULL,
  priority REAL NOT NULL,
  payload_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL,
  leased_by TEXT,
  lease_expires_utc TEXT,
  result_json TEXT,
  error_text TEXT,
  created_utc TEXT NOT NULL,
  started_utc TEXT,
  completed_utc TEXT,
  UNIQUE(ordinal,lane)
);
CREATE TABLE IF NOT EXISTS checkpoints (
  sequence INTEGER PRIMARY KEY,
  created_utc TEXT NOT NULL,
  snapshot_json TEXT NOT NULL,
  snapshot_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS execution_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc TEXT NOT NULL,
  event_type TEXT NOT NULL,
  work_id TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_claim ON work(lane,state,priority,ordinal);
"""


@dataclass(frozen=True)
class WorkLease:
    work_id: str
    ordinal: int
    lane: str
    seed: int
    attempt: int
    max_attempts: int
    payload: dict[str, Any]


class PersistentParallelSearch:
    """Durable bounded coordinator for CPU symbolic and one-owner CUDA work."""

    def __init__(
        self,
        database: str | Path,
        config: dict[str, Any],
        resource_profile: dict[str, Any],
    ) -> None:
        _validate_config(config)
        self.database = Path(database).resolve()
        self.config = config
        self.plan = plan_parallel_capacity(resource_profile, config)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            existing = connection.execute(
                "SELECT config_sha256,plan_json FROM execution WHERE singleton=1"
            ).fetchone()
            config_hash = _sha(config)
            if existing is not None and existing[0] != config_hash:
                raise ValueError("refusing to resume with a different execution config")
            if existing is not None and existing[1] != _canonical(self.plan):
                raise ValueError("refusing to resume with a different hardware plan")
            if existing is None:
                now = _now()
                deadline = now + timedelta(
                    seconds=float(config["budget"]["maximum_wall_seconds"])
                )
                connection.execute(
                    "INSERT INTO execution VALUES (1,?,?,?,?,?,0,0,0,0,0)",
                    (
                        _canonical(config),
                        config_hash,
                        _canonical(self.plan),
                        now.isoformat(),
                        deadline.isoformat(),
                    ),
                )
                self._event(connection, "execution_created", None, self.plan)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        event_type: str,
        work_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        connection.execute(
            "INSERT INTO execution_events(created_utc,event_type,work_id,payload_json) "
            "VALUES (?,?,?,?)",
            (_now().isoformat(), event_type, work_id, _canonical(payload)),
        )

    def _budget_open(self, row: sqlite3.Row) -> bool:
        return (
            int(row["submitted"]) < int(self.config["budget"]["maximum_tasks"])
            and _now() < datetime.fromisoformat(row["deadline_utc"])
        )

    def enqueue(
        self,
        items: list[dict[str, Any]],
        *,
        lane: str,
        priority: float = 0.0,
        max_attempts: int | None = None,
    ) -> dict[str, int]:
        if lane not in {"cpu", "gpu"}:
            raise ValueError("lane must be cpu or gpu")
        if lane == "gpu" and not self.plan["gpu"]["available"]:
            raise ValueError("GPU lane is unavailable")
        accepted = duplicate = backpressured = budget_rejected = 0
        maximum_queue = int(self.config["queue"]["maximum_pending_work"])
        attempts = int(max_attempts or self.config["queue"]["maximum_attempts"])
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            execution = connection.execute("SELECT * FROM execution").fetchone()
            pending = connection.execute(
                "SELECT COUNT(*) FROM work WHERE state IN ('queued','running')"
            ).fetchone()[0]
            for item in items:
                ordinal = int(item["ordinal"])
                seed_payload = {
                    "master_seed": int(self.config["determinism"]["master_seed"]),
                    "ordinal": ordinal,
                    "lane": lane,
                    "payload": item,
                }
                digest = hashlib.sha256(_canonical(seed_payload).encode()).digest()
                seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
                work_id = f"PSW-{hashlib.sha256(_canonical(seed_payload).encode()).hexdigest()[:24]}"
                if connection.execute(
                    "SELECT 1 FROM work WHERE ordinal=? AND lane=?", (ordinal, lane)
                ).fetchone():
                    duplicate += 1
                    continue
                if not self._budget_open(execution):
                    budget_rejected += 1
                    continue
                if pending >= maximum_queue:
                    backpressured += 1
                    continue
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO work VALUES "
                    "(?,?,?,?,?,?,'queued',0,?,NULL,NULL,NULL,NULL,?,NULL,NULL)",
                    (
                        work_id,
                        ordinal,
                        lane,
                        seed,
                        float(priority),
                        _canonical(item),
                        attempts,
                        _now().isoformat(),
                    ),
                )
                if cursor.rowcount:
                    accepted += 1
                    pending += 1
                    connection.execute(
                        "UPDATE execution SET submitted=submitted+1 WHERE singleton=1"
                    )
                    execution = connection.execute("SELECT * FROM execution").fetchone()
                    self._event(connection, "work_enqueued", work_id, {"lane": lane, "seed": seed})
                else:
                    duplicate += 1
        return {
            "accepted": accepted,
            "duplicate": duplicate,
            "backpressured": backpressured,
            "budget_rejected": budget_rejected,
        }

    def claim(self, lane: str, worker_id: str, lease_seconds: int | None = None) -> WorkLease | None:
        capacity = int(self.plan[lane]["workers"])
        if capacity <= 0:
            return None
        lease = int(lease_seconds or self.config["queue"]["lease_seconds"])
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            execution = connection.execute("SELECT * FROM execution").fetchone()
            if not self._budget_open(execution) and _now() >= datetime.fromisoformat(
                execution["deadline_utc"]
            ):
                return None
            active = connection.execute(
                "SELECT COUNT(*) FROM work WHERE lane=? AND state='running'", (lane,)
            ).fetchone()[0]
            if active >= capacity:
                return None
            row = connection.execute(
                "SELECT * FROM work WHERE lane=? AND state='queued' "
                "ORDER BY priority DESC,ordinal ASC LIMIT 1",
                (lane,),
            ).fetchone()
            if row is None:
                return None
            now = _now()
            expiry = now + timedelta(seconds=lease)
            connection.execute(
                "UPDATE work SET state='running',attempt=attempt+1,leased_by=?,"
                "lease_expires_utc=?,started_utc=COALESCE(started_utc,?) WHERE work_id=?",
                (worker_id, expiry.isoformat(), now.isoformat(), row["work_id"]),
            )
            self._event(connection, "work_claimed", row["work_id"], {"worker_id": worker_id})
            return WorkLease(
                row["work_id"],
                row["ordinal"],
                lane,
                row["seed"],
                row["attempt"] + 1,
                row["max_attempts"],
                json.loads(row["payload_json"]),
            )

    def heartbeat(self, lease: WorkLease, worker_id: str) -> bool:
        expiry = _now() + timedelta(seconds=int(self.config["queue"]["lease_seconds"]))
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE work SET lease_expires_utc=? WHERE work_id=? AND state='running' "
                "AND leased_by=?",
                (expiry.isoformat(), lease.work_id, worker_id),
            )
            return bool(cursor.rowcount)

    def finish(self, lease: WorkLease, worker_id: str, result: dict[str, Any]) -> bool:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE work SET state='succeeded',result_json=?,completed_utc=?,"
                "lease_expires_utc=NULL WHERE work_id=? AND state='running' AND leased_by=?",
                (_canonical(result), _now().isoformat(), lease.work_id, worker_id),
            )
            if cursor.rowcount:
                connection.execute("UPDATE execution SET completed=completed+1 WHERE singleton=1")
                self._event(connection, "work_succeeded", lease.work_id, result)
            return bool(cursor.rowcount)

    def fail(self, lease: WorkLease, worker_id: str, error: str) -> str:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work WHERE work_id=? AND state='running' AND leased_by=?",
                (lease.work_id, worker_id),
            ).fetchone()
            if row is None:
                return "stale"
            retry = int(row["attempt"]) < int(row["max_attempts"])
            state = "queued" if retry else "failed"
            connection.execute(
                "UPDATE work SET state=?,error_text=?,leased_by=NULL,lease_expires_utc=NULL,"
                "completed_utc=? WHERE work_id=?",
                (state, error, None if retry else _now().isoformat(), lease.work_id),
            )
            if not retry:
                connection.execute("UPDATE execution SET failed=failed+1 WHERE singleton=1")
            self._event(connection, f"work_{state}", lease.work_id, {"error": error})
            return state

    def recover_expired(self) -> dict[str, int]:
        recovered = failed = 0
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM work WHERE state='running' AND lease_expires_utc<?",
                (_now().isoformat(),),
            ).fetchall()
            for row in rows:
                retry = int(row["attempt"]) < int(row["max_attempts"])
                state = "queued" if retry else "failed"
                connection.execute(
                    "UPDATE work SET state=?,leased_by=NULL,lease_expires_utc=NULL,"
                    "error_text='expired lease',completed_utc=? WHERE work_id=?",
                    (state, None if retry else _now().isoformat(), row["work_id"]),
                )
                if retry:
                    recovered += 1
                else:
                    failed += 1
            connection.execute(
                "UPDATE execution SET recovered=recovered+?,failed=failed+? WHERE singleton=1",
                (recovered, failed),
            )
            if rows:
                self._event(connection, "expired_leases_recovered", None, {"recovered": recovered, "failed": failed})
        return {"recovered": recovered, "failed": failed}

    def checkpoint(self) -> dict[str, Any]:
        telemetry = self.telemetry()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = connection.execute(
                "SELECT checkpoint_sequence+1 FROM execution"
            ).fetchone()[0]
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "sequence": sequence,
                "config_sha256": _sha(self.config),
                "telemetry": telemetry,
                "work_state_root": self._work_state_root(connection),
            }
            snapshot_hash = _sha(snapshot)
            connection.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?)",
                (sequence, _now().isoformat(), _canonical(snapshot), snapshot_hash),
            )
            connection.execute(
                "UPDATE execution SET checkpoint_sequence=? WHERE singleton=1",
                (sequence,),
            )
            self._event(connection, "checkpoint_written", None, {"sequence": sequence, "sha256": snapshot_hash})
        return {**snapshot, "content_sha256": snapshot_hash}

    @staticmethod
    def _work_state_root(connection: sqlite3.Connection) -> str:
        rows = connection.execute(
            "SELECT work_id,state,attempt,seed,result_json,error_text FROM work ORDER BY work_id"
        ).fetchall()
        return _sha([dict(row) for row in rows])

    def telemetry(self) -> dict[str, Any]:
        with self.connect() as connection:
            execution = connection.execute("SELECT * FROM execution").fetchone()
            counts = {
                row["state"]: row["count"]
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM work GROUP BY state"
                )
            }
            lane_rows = {
                row["lane"]: dict(row)
                for row in connection.execute(
                    "SELECT lane,SUM(state='queued') AS queued,SUM(state='running') AS running,"
                    "SUM(state='succeeded') AS succeeded,SUM(state='failed') AS failed "
                    "FROM work GROUP BY lane"
                )
            }
            pending = counts.get("queued", 0) + counts.get("running", 0)
            lanes: dict[str, Any] = {}
            warnings: list[str] = []
            for lane in ("cpu", "gpu"):
                capacity = int(self.plan[lane]["workers"])
                row = lane_rows.get(lane, {})
                running = int(row.get("running") or 0)
                queued = int(row.get("queued") or 0)
                utilization = running / capacity if capacity else 0.0
                reason = None
                if capacity and running < capacity:
                    reason = "workers_not_claiming" if queued else "queue_starved"
                    warnings.append(f"{lane}:{reason}")
                lanes[lane] = {
                    "capacity": capacity,
                    "running": running,
                    "queued": queued,
                    "utilization": utilization,
                    "underutilization_reason": reason,
                }
            gpu_batch = int(self.plan["gpu"]["batch_candidates"])
            queued_gpu_candidates = connection.execute(
                "SELECT payload_json FROM work WHERE lane='gpu' AND state='queued'"
            ).fetchall()
            queued_candidates = sum(
                int(json.loads(row[0]).get("candidate_count", 1))
                for row in queued_gpu_candidates
            )
            if gpu_batch and 0 < queued_candidates < gpu_batch:
                warnings.append("gpu:batch_underfilled")
            created = datetime.fromisoformat(execution["created_utc"])
            elapsed = max((_now() - created).total_seconds(), 1e-9)
            return {
                "schema_version": SCHEMA_VERSION,
                "plan": self.plan,
                "counts": counts,
                "queue": {
                    "pending": pending,
                    "capacity": int(self.config["queue"]["maximum_pending_work"]),
                    "fill_fraction": pending / int(self.config["queue"]["maximum_pending_work"]),
                },
                "lanes": lanes,
                "gpu": {
                    "planned_batch_candidates": gpu_batch,
                    "queued_candidates": queued_candidates,
                    "queued_batch_fill_fraction": (
                        min(1.0, queued_candidates / gpu_batch) if gpu_batch else 0.0
                    ),
                },
                "budget": {
                    "submitted": int(execution["submitted"]),
                    "maximum_tasks": int(self.config["budget"]["maximum_tasks"]),
                    "remaining_tasks": max(
                        0,
                        int(self.config["budget"]["maximum_tasks"])
                        - int(execution["submitted"]),
                    ),
                    "deadline_utc": execution["deadline_utc"],
                },
                "throughput": {
                    "completed_per_second": int(execution["completed"]) / elapsed,
                    "measured_gpu_candidates_per_second": self.plan["gpu"][
                        "measured_candidates_per_second"
                    ],
                },
                "recovered_leases": int(execution["recovered"]),
                "checkpoint_sequence": int(execution["checkpoint_sequence"]),
                "underutilization_warnings": warnings,
                "paid_llm_calls_enabled": False,
            }
