from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .gravity_engine_service import _hardware_telemetry
from .high_throughput import total_search_count
from .persistent_parallel_search import PersistentParallelSearch
from .persistent_parallel_supervisor import PersistentParallelSupervisor
from .rust_streaming_search import (
    ELIGIBILITY,
    HARD_MAXIMUM_FORMULAS,
    MANIFEST_ALLOWANCE_BYTES,
    PROMOTION_RECORD,
    _directory_bytes,
    _file_sha,
    _sha,
    _verify_stream_manifest,
    configure_rust_streaming_execution,
    validate_binary_result,
)
from .survivors import HEADER, RECORD

SCHEMA_VERSION = "sigma-rust-parallel-streaming-1.0"
REPORT_SCHEMA = "sigma-rust-parallel-streaming-report-1.0"

PARALLEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS rust_parallel_source (
  source_id TEXT PRIMARY KEY,
  identity_sha256 TEXT NOT NULL,
  start_ordinal INTEGER NOT NULL,
  stop_ordinal INTEGER NOT NULL,
  chunk_size INTEGER NOT NULL,
  deadline_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rust_parallel_chunks (
  source_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  start_ordinal INTEGER NOT NULL,
  end_ordinal_exclusive INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('available','generating','verified','enqueued')),
  owner_id TEXT,
  lease_expires_utc TEXT,
  manifest_path TEXT NOT NULL,
  manifest_sha256 TEXT,
  block_path TEXT,
  block_sha256 TEXT,
  block_size_bytes INTEGER,
  record_count INTEGER,
  generation_started_time_ns INTEGER,
  generation_ended_time_ns INTEGER,
  producer_stdout TEXT,
  PRIMARY KEY(source_id,sequence)
);
"""


class ParallelRustRangeScheduler:
    """Lease disjoint Rust ranges concurrently and feed one persistent GPU queue."""

    def __init__(
        self,
        coordinator: PersistentParallelSearch,
        config: dict[str, Any],
        *,
        scheduler_id: str | None = None,
    ) -> None:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported parallel Rust streaming schema")
        if config.get("external_paid_llm_calls") is not False:
            raise ValueError("paid LLM calls must remain disabled")
        if config.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("parallel source eligibility is not fail-closed")
        self.coordinator = coordinator
        self.config = config
        self.scheduler_id = scheduler_id or f"parallel-{os.getpid()}-{uuid.uuid4().hex[:16]}"
        self.generator_path = Path(config["generator_config_path"]).resolve()
        self.binary_path = Path(config["generator_binary_path"]).resolve()
        self.output_directory = Path(config["output_directory"]).resolve()
        self.promotion_directory = Path(
            config.get("promotion_directory", self.output_directory / "promotion")
        ).resolve()
        if not self.generator_path.is_file() or not self.binary_path.is_file():
            raise FileNotFoundError("trusted generator config or Rust binary is unavailable")
        generator_raw = self.generator_path.read_bytes()
        self.generator = json.loads(generator_raw)
        if self.generator.get("observational_data_opened") is not False:
            raise ValueError("Rust generator config opens observational data")
        self.generator_sha = hashlib.sha256(generator_raw).hexdigest()
        self.binary_sha = _file_sha(self.binary_path)
        self.start = int(config["start_ordinal"])
        self.formula_count = int(config["formula_count"])
        self.chunk_size = int(config["chunk_formula_count"])
        maximum = int(config["maximum_formula_count"])
        if not 1 <= self.formula_count <= maximum <= HARD_MAXIMUM_FORMULAS:
            raise ValueError("parallel formula budget exceeds one billion")
        if self.chunk_size <= 0 or self.start % self.chunk_size:
            raise ValueError("parallel start must align to a positive chunk size")
        self.stop = self.start + self.formula_count
        declared = total_search_count(
            int(self.generator["basis_count"]), int(self.generator["max_action_terms"])
        )
        if self.start < 0 or self.stop > declared:
            raise ValueError("parallel interval exceeds the generator search space")
        self.worker_count = int(config["producer_workers"])
        if not 1 <= self.worker_count <= 24:
            raise ValueError("parallel producer workers must be between one and 24")
        if not 1 <= int(config["threads_per_producer"]) <= 24:
            raise ValueError("threads per Rust producer must be between one and 24")
        self.target_pending = int(config["target_pending_chunks"])
        if self.target_pending < self.worker_count:
            raise ValueError("pending target must cover all Rust producers")
        self.lease_seconds = float(config["producer_chunk_lease_seconds"])
        wall = float(config["maximum_wall_seconds"])
        if self.lease_seconds <= 0 or wall <= 0:
            raise ValueError("parallel lease and wall limits must be positive")
        chunks = (self.formula_count + self.chunk_size - 1) // self.chunk_size
        worst = self.formula_count * (RECORD.size + PROMOTION_RECORD.size) + chunks * (
            HEADER.size + MANIFEST_ALLOWANCE_BYTES
        )
        self.disk_budget = int(config["maximum_disk_bytes"])
        if self.disk_budget <= 0 or worst > self.disk_budget:
            raise ValueError("parallel disk budget fails worst-case preflight")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.promotion_directory.mkdir(parents=True, exist_ok=True)
        identity = {
            "generator_sha256": self.generator_sha,
            "binary_sha256": self.binary_sha,
            "start": self.start,
            "stop": self.stop,
            "chunk_size": self.chunk_size,
            "disk_budget": self.disk_budget,
            "eligibility": ELIGIBILITY,
        }
        self.identity_sha = _sha(identity)
        self.source_id = f"RUSTPAR-{self.identity_sha[:24]}"
        with coordinator.connect() as connection:
            connection.executescript(PARALLEL_SCHEMA)
            row = connection.execute(
                "SELECT * FROM rust_parallel_source WHERE source_id=?", (self.source_id,)
            ).fetchone()
            if row is None:
                deadline = datetime.now(UTC) + timedelta(seconds=wall)
                connection.execute(
                    "INSERT INTO rust_parallel_source VALUES (?,?,?,?,?,?)",
                    (
                        self.source_id,
                        self.identity_sha,
                        self.start,
                        self.stop,
                        self.chunk_size,
                        deadline.isoformat(),
                    ),
                )
                for sequence in range(chunks):
                    start = self.start + sequence * self.chunk_size
                    end = min(self.stop, start + self.chunk_size)
                    manifest = self.output_directory / (
                        f"parallel-{sequence:08}-{start}-{end}.json"
                    )
                    connection.execute(
                        "INSERT INTO rust_parallel_chunks "
                        "(source_id,sequence,start_ordinal,end_ordinal_exclusive,state,"
                        "manifest_path) VALUES (?,?,?,?,?,?)",
                        (self.source_id, sequence, start, end, "available", str(manifest)),
                    )
            elif row["identity_sha256"] != self.identity_sha:
                raise ValueError("refusing to resume a changed parallel Rust source")
        self._recover_expired()
        self._recover_generated_files()
        self._executor = ThreadPoolExecutor(
            max_workers=self.worker_count, thread_name_prefix="sigma-rust-producer"
        )
        self._futures: dict[Future[dict[str, Any]], tuple[int, str]] = {}
        self._lock = threading.Lock()
        self._peak_active = 0
        self._launch_count = 0

    def _recover_expired(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self.coordinator.connect() as connection:
            cursor = connection.execute(
                "UPDATE rust_parallel_chunks SET state='available',owner_id=NULL,"
                "lease_expires_utc=NULL WHERE source_id=? AND state='generating' "
                "AND lease_expires_utc<=?",
                (self.source_id, now),
            )
        return cursor.rowcount

    def _verify(self, row: sqlite3.Row) -> dict[str, Any]:
        audit = _verify_stream_manifest(
            Path(row["manifest_path"]),
            self.output_directory,
            self.generator_path,
            expected_start=int(row["start_ordinal"]),
            expected_end=int(row["end_ordinal_exclusive"]),
            equivalence_samples_per_block=int(self.config["equivalence_samples_per_chunk"]),
        )
        if len(audit["blocks"]) != 1:
            raise ValueError("parallel chunk must produce exactly one SGSURV2 block")
        block = audit["blocks"][0]
        block_path = self.output_directory / block["file"]
        return {
            "manifest_sha256": audit["manifest_sha256"],
            "block_path": str(block_path),
            "block_sha256": block["file_sha256"],
            "block_size_bytes": block_path.stat().st_size,
            "record_count": block["record_count"],
        }

    def _recover_generated_files(self) -> None:
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rust_parallel_chunks WHERE source_id=? AND state='available' "
                "ORDER BY sequence",
                (self.source_id,),
            ).fetchall()
        for row in rows:
            if not Path(row["manifest_path"]).is_file():
                continue
            try:
                verified = self._verify(row)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            with self.coordinator.connect() as connection:
                connection.execute(
                    "UPDATE rust_parallel_chunks SET state='verified',manifest_sha256=?,"
                    "block_path=?,block_sha256=?,block_size_bytes=?,record_count=?,owner_id=NULL,"
                    "lease_expires_utc=NULL WHERE source_id=? AND sequence=? AND state='available'",
                    (
                        verified["manifest_sha256"],
                        verified["block_path"],
                        verified["block_sha256"],
                        verified["block_size_bytes"],
                        verified["record_count"],
                        self.source_id,
                        row["sequence"],
                    ),
                )

    def _lease(self) -> tuple[sqlite3.Row, str] | None:
        owner = f"{self.scheduler_id}-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        expiry = now + timedelta(seconds=self.lease_seconds)
        with self.coordinator.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM rust_parallel_chunks WHERE source_id=? AND state='available' "
                "ORDER BY sequence LIMIT 1",
                (self.source_id,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE rust_parallel_chunks SET state='generating',owner_id=?,"
                "lease_expires_utc=?,generation_started_time_ns=? WHERE source_id=? "
                "AND sequence=? AND state='available'",
                (owner, expiry.isoformat(), time.time_ns(), self.source_id, row["sequence"]),
            )
        return row, owner

    def _heartbeat(self, sequence: int, owner: str) -> None:
        expiry = datetime.now(UTC) + timedelta(seconds=self.lease_seconds)
        with self.coordinator.connect() as connection:
            cursor = connection.execute(
                "UPDATE rust_parallel_chunks SET lease_expires_utc=? WHERE source_id=? "
                "AND sequence=? AND state='generating' AND owner_id=?",
                (expiry.isoformat(), self.source_id, sequence, owner),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("parallel Rust producer lost its range lease")

    def _generate(self, row: sqlite3.Row, owner: str) -> dict[str, Any]:
        sequence = int(row["sequence"])
        start, end = int(row["start_ordinal"]), int(row["end_ordinal_exclusive"])
        manifest_path = Path(row["manifest_path"])
        command = [
            str(self.binary_path),
            "run",
            "--config",
            str(self.generator_path),
            "--output",
            str(manifest_path),
            "--start",
            str(start),
            "--limit",
            str(end - start),
            "--threads",
            str(int(self.config["threads_per_producer"])),
            "--block-size",
            str(self.chunk_size),
            "--survivor-dir",
            str(self.output_directory),
        ]
        with self.coordinator.connect() as connection:
            deadline = datetime.fromisoformat(
                connection.execute(
                    "SELECT deadline_utc FROM rust_parallel_source WHERE source_id=?",
                    (self.source_id,),
                ).fetchone()[0]
            )
        process = subprocess.Popen(
            command,
            cwd=self.output_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        heartbeat = max(0.1, self.lease_seconds / 3)
        while True:
            remaining = (deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                process.kill()
                process.communicate()
                raise TimeoutError("parallel Rust streaming wall budget exhausted")
            try:
                stdout, stderr = process.communicate(timeout=min(heartbeat, remaining))
                break
            except subprocess.TimeoutExpired:
                self._heartbeat(sequence, owner)
        if process.returncode != 0:
            raise RuntimeError(f"parallel Rust producer failed: {stderr[-2000:]}")
        with self.coordinator.connect() as connection:
            current = connection.execute(
                "SELECT * FROM rust_parallel_chunks WHERE source_id=? AND sequence=?",
                (self.source_id, sequence),
            ).fetchone()
        verified = self._verify(current)
        if _directory_bytes(self.output_directory) > self.disk_budget:
            raise ValueError("parallel Rust stream exceeded its disk budget")
        ended = time.time_ns()
        with self.coordinator.connect() as connection:
            cursor = connection.execute(
                "UPDATE rust_parallel_chunks SET state='verified',owner_id=NULL,"
                "lease_expires_utc=NULL,manifest_sha256=?,block_path=?,block_sha256=?,"
                "block_size_bytes=?,record_count=?,generation_ended_time_ns=?,producer_stdout=? "
                "WHERE source_id=? AND sequence=? AND state='generating' AND owner_id=?",
                (
                    verified["manifest_sha256"],
                    verified["block_path"],
                    verified["block_sha256"],
                    verified["block_size_bytes"],
                    verified["record_count"],
                    ended,
                    stdout[-4000:],
                    self.source_id,
                    sequence,
                    owner,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("parallel range completion lost its lease")
        return {"sequence": sequence, **verified}

    def _payload(self, row: sqlite3.Row) -> dict[str, Any]:
        manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
        block = manifest["blocks"][0]
        normalization = (
            "omitted-zero-survive-sampled-static-gate"
            if int(manifest.get("survivor_count", -1)) == 0
            and "survive_sampled_static" not in manifest.get("gate_counts", {})
            else None
        )
        return {
            "ordinal": int(block["block_index"]),
            "source_id": self.source_id,
            "sequence": int(row["sequence"]),
            "manifest_path": row["manifest_path"],
            "manifest_sha256": row["manifest_sha256"],
            "generator_config_path": str(self.generator_path),
            "generator_config_sha256": self.generator_sha,
            "block_path": row["block_path"],
            "block_sha256": row["block_sha256"],
            "block_size_bytes": int(row["block_size_bytes"]),
            "block_index": int(block["block_index"]),
            "start_ordinal": int(block["start_ordinal"]),
            "end_ordinal_exclusive": int(block["end_ordinal_exclusive"]),
            "record_count": int(row["record_count"]),
            "basis_count": int(self.generator["basis_count"]),
            "max_action_terms": int(self.generator["max_action_terms"]),
            "equivalence_samples": int(self.config["equivalence_samples_per_chunk"]),
            "ambiguity_guard": float(self.config["ambiguity_guard"]),
            "promotion_directory": str(self.promotion_directory),
            "manifest_verification_normalization": normalization,
            "data_eligibility": ELIGIBILITY,
        }

    def _reap(self) -> None:
        with self._lock:
            completed = [future for future in self._futures if future.done()]
            for future in completed:
                sequence, _ = self._futures.pop(future)
                future.result()
                if sequence < 0:
                    raise AssertionError("invalid parallel sequence")

    def _enqueue_verified(self) -> tuple[int, int, int]:
        accepted = duplicates = backpressured = 0
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rust_parallel_chunks WHERE source_id=? AND state='verified' "
                "ORDER BY sequence",
                (self.source_id,),
            ).fetchall()
        for row in rows:
            outcome = self.coordinator.enqueue([self._payload(row)], lane="gpu")
            if outcome["accepted"] or outcome["duplicate"]:
                accepted += outcome["accepted"]
                duplicates += outcome["duplicate"]
                with self.coordinator.connect() as connection:
                    connection.execute(
                        "UPDATE rust_parallel_chunks SET state='enqueued' WHERE source_id=? "
                        "AND sequence=? AND state='verified'",
                        (self.source_id, row["sequence"]),
                    )
            else:
                backpressured += outcome["backpressured"]
                break
        return accepted, duplicates, backpressured

    def refill(self) -> dict[str, Any]:
        self._recover_expired()
        self._reap()
        accepted, duplicates, backpressured = self._enqueue_verified()
        telemetry = self.coordinator.telemetry()
        with self._lock:
            capacity = self.worker_count - len(self._futures)
            queued_or_running = int(telemetry["queue"]["pending"])
            launches = min(capacity, max(0, self.target_pending - queued_or_running))
            for _ in range(launches):
                leased = self._lease()
                if leased is None:
                    break
                row, owner = leased
                future = self._executor.submit(self._generate, row, owner)
                self._futures[future] = (int(row["sequence"]), owner)
                self._launch_count += 1
            self._peak_active = max(self._peak_active, len(self._futures))
        return {
            "accepted_chunks": accepted,
            "duplicate_chunks": duplicates,
            "backpressured_chunks": backpressured,
            "active_producers": len(self._futures),
            "peak_active_producers": self._peak_active,
            "launched_producers": self._launch_count,
            "cursor": self.status(),
        }

    def status(self) -> dict[str, Any]:
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT state,COUNT(*) AS count FROM rust_parallel_chunks WHERE source_id=? "
                "GROUP BY state",
                (self.source_id,),
            ).fetchall()
            chunks = connection.execute(
                "SELECT sequence,start_ordinal,end_ordinal_exclusive,state FROM "
                "rust_parallel_chunks WHERE source_id=? ORDER BY sequence",
                (self.source_id,),
            ).fetchall()
        counts = {row["state"]: int(row["count"]) for row in rows}
        next_ordinal = self.start
        for chunk in chunks:
            if int(chunk["start_ordinal"]) != next_ordinal or chunk["state"] != "enqueued":
                break
            next_ordinal = int(chunk["end_ordinal_exclusive"])
        exhausted = counts.get("enqueued", 0) == len(chunks)
        return {
            "source_id": self.source_id,
            "next_contiguous_ordinal": next_ordinal,
            "stop_ordinal_exclusive": self.stop,
            "chunk_counts": counts,
            "active_producers": len(self._futures),
            "peak_active_producers": self._peak_active,
            "exhausted": exhausted,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def _overlap(intervals_a: list[tuple[int, int]], intervals_b: list[tuple[int, int]]) -> float:
    def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        output: list[list[int]] = []
        for start, end in sorted(intervals):
            if not output or start > output[-1][1]:
                output.append([start, end])
            else:
                output[-1][1] = max(output[-1][1], end)
        return [(start, end) for start, end in output]

    left, right = merged(intervals_a), merged(intervals_b)
    return sum(
        max(0, min(a1, b1) - max(a0, b0))
        for a0, a1 in left
        for b0, b1 in right
    ) / 1e9


def _hardware_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    gpu = [sample["gpu"] for sample in samples if sample.get("gpu", {}).get("available")]
    cpu = [sample["cpu"] for sample in samples if sample.get("cpu", {}).get("available")]

    def stats(values: list[float]) -> dict[str, float] | None:
        return (
            {"minimum": min(values), "mean": sum(values) / len(values), "peak": max(values)}
            if values
            else None
        )

    return {
        "semantics": "periodic physical sensor samples, distinct from scheduler occupancy",
        "sample_count": len(samples),
        "gpu_available_samples": len(gpu),
        "gpu_utilization_percent": stats([float(item["utilization_percent"]) for item in gpu]),
        "gpu_memory_controller_percent": stats(
            [float(item["memory_controller_utilization_percent"]) for item in gpu]
        ),
        "gpu_power_watts": stats(
            [float(item["power_watts"]) for item in gpu if item.get("power_watts") is not None]
        ),
        "cpu_available_samples": len(cpu),
        "cpu_utilization_percent": stats([float(item["utilization_percent"]) for item in cpu]),
    }


def run_parallel_rust_streaming_search(
    database: str | Path,
    execution_config: dict[str, Any],
    resource_profile: dict[str, Any],
    parallel_config: dict[str, Any],
    telemetry_path: str | Path,
    *,
    external_stop_path: str | Path | None = None,
    stop_reason_callback: Any = None,
    status_callback: Any = None,
) -> dict[str, Any]:
    stream_view = {
        **parallel_config,
        "schema_version": "sigma-rust-streaming-search-1.0",
        "producer_lease_seconds": parallel_config["producer_chunk_lease_seconds"],
    }
    configured = configure_rust_streaming_execution(execution_config, stream_view)
    coordinator = PersistentParallelSearch(database, configured, resource_profile)
    scheduler = ParallelRustRangeScheduler(coordinator, parallel_config)
    hardware_samples: list[dict[str, Any]] = []

    def periodic(value: dict[str, Any]) -> None:
        hardware = _hardware_telemetry()
        hardware_samples.append(hardware)
        if status_callback is not None:
            status_callback({**value, "hardware": hardware})

    started = time.time_ns()
    try:
        supervisor = PersistentParallelSupervisor(
            database, configured, resource_profile, telemetry_path
        ).run(
            refill_callback=scheduler.refill,
            external_stop_path=external_stop_path,
            stop_reason_callback=stop_reason_callback,
            status_callback=periodic,
        )
    finally:
        scheduler.close()
    ended = time.time_ns()
    with coordinator.connect() as connection:
        chunks = connection.execute(
            "SELECT * FROM rust_parallel_chunks WHERE source_id=? ORDER BY sequence",
            (scheduler.source_id,),
        ).fetchall()
        work = connection.execute(
            "SELECT state,payload_json,result_json FROM work ORDER BY ordinal"
        ).fetchall()
    producer_intervals = [
        (int(row["generation_started_time_ns"]), int(row["generation_ended_time_ns"]))
        for row in chunks
        if row["generation_started_time_ns"] and row["generation_ended_time_ns"]
    ]
    consumer_intervals: list[tuple[int, int]] = []
    records = formulas_screened = cache_reused = 0
    backends: Counter[str] = Counter()
    lineage: list[dict[str, Any]] = []
    for row in work:
        if row["state"] != "succeeded" or not row["result_json"]:
            continue
        payload, result = json.loads(row["payload_json"]), json.loads(row["result_json"])
        validate_binary_result(result, payload)
        promotion = result.get("promotion_survivor_export")
        if (
            not isinstance(promotion, dict)
            or promotion.get("source_block_sha256") != payload["block_sha256"]
            or promotion.get("source_manifest_sha256") != payload["manifest_sha256"]
            or _file_sha(Path(promotion.get("path", ""))) != promotion.get("sha256")
        ):
            raise ValueError("parallel promotion lineage verification failed")
        lineage.append(
            {
                "sequence": int(payload["sequence"]),
                "start_ordinal": int(payload["start_ordinal"]),
                "end_ordinal_exclusive": int(payload["end_ordinal_exclusive"]),
                "manifest_sha256": payload["manifest_sha256"],
                "block_sha256": payload["block_sha256"],
                "status_root_sha256": result["status_root_sha256"],
                "promotion_sha256": promotion["sha256"],
                "promotion_record_count": int(promotion["record_count"]),
            }
        )
        records += int(result["block"]["record_count"])
        formulas_screened += int(result["block"]["end_ordinal_exclusive"]) - int(
            result["block"]["start_ordinal"]
        )
        cache_reused += result["cuda_assets_reused"] is True
        backends[result["backend"]] += 1
        timing = result["streaming_timing"]
        consumer_intervals.append(
            (int(timing["started_time_ns"]), int(timing["ended_time_ns"]))
        )
    wall = (ended - started) / 1e9
    producer_busy = sum(end - start for start, end in producer_intervals) / 1e9
    consumer_busy = sum(end - start for start, end in consumer_intervals) / 1e9
    status = scheduler.status()
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_id": scheduler.source_id,
        "formula_count": scheduler.formula_count,
        "chunk_count": len(chunks),
        "producer": {
            "workers": scheduler.worker_count,
            "peak_active": scheduler._peak_active,
            "aggregate_busy_seconds": producer_busy,
            "wall_utilization_fraction": (
                producer_busy / (wall * scheduler.worker_count) if wall else 0.0
            ),
            "formulas_per_busy_second": (
                scheduler.formula_count / producer_busy if producer_busy else None
            ),
            "source_formulas_per_wall_second": (
                scheduler.formula_count / wall if wall else None
            ),
        },
        "consumer": {
            "busy_seconds": consumer_busy,
            "survivor_records": records,
            "records_per_second": records / consumer_busy if consumer_busy else None,
            "cache_reused_chunks": cache_reused,
            "backend_counts": dict(backends),
        },
        "combined": {
            "wall_seconds": wall,
            "source_formulas_screened": formulas_screened,
            "source_formulas_per_second": formulas_screened / wall if wall else None,
            "producer_consumer_overlap_seconds": _overlap(
                producer_intervals, consumer_intervals
            ),
        },
        "cursor": status,
        "lineage": {
            "verified_chunk_count": len(lineage),
            "promotion_identity_count": sum(
                int(item["promotion_record_count"]) for item in lineage
            ),
            "root_sha256": _sha(sorted(lineage, key=lambda item: item["sequence"])),
        },
        "supervisor": supervisor,
        "hardware": _hardware_summary(hardware_samples),
        "all_work_succeeded": (
            status["exhausted"]
            and len(work) == len(chunks)
            and all(row["state"] == "succeeded" for row in work)
        ),
        "disk_bytes": _directory_bytes(scheduler.output_directory),
        "maximum_disk_bytes": scheduler.disk_budget,
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    report["content_sha256"] = _sha(report)
    return report
