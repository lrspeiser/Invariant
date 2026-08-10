from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import time
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .binary_formula_execution import (
    _assets,
    _cuda_assets,
    _load_block,
    _make_result,
    _screen_cpu,
    configure_binary_evaluators,
    validate_binary_result,
)
from .bounded_survivor_corpus import HEADER, RECORD, verify_generated_manifest
from .high_throughput import total_search_count
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .persistent_parallel_supervisor import PersistentParallelSupervisor

SCHEMA_VERSION = "sigma-rust-streaming-search-1.0"
REPORT_SCHEMA = "sigma-rust-streaming-search-report-1.0"
ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
}
HARD_MAXIMUM_FORMULAS = 1_000_000_000
MANIFEST_ALLOWANCE_BYTES = 2 * 1024 * 1024
PROMOTION_MAGIC = b"SGPROM1\0"
PROMOTION_HEADER = struct.Struct("<8sHHQQQ")
PROMOTION_RECORD = struct.Struct("<BQBBH6H")


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


def _directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _verify_stream_manifest(
    manifest_path: Path,
    survivor_directory: Path,
    generator_path: Path,
    *,
    expected_start: int,
    expected_end: int,
    equivalence_samples_per_block: int,
) -> dict[str, Any]:
    """Verify Rust output, accepting only its omitted zero-valued survivor gate key."""
    try:
        return verify_generated_manifest(
            manifest_path,
            survivor_directory,
            generator_path,
            expected_start=expected_start,
            expected_end=expected_end,
            equivalence_samples_per_block=equivalence_samples_per_block,
        )
    except ValueError as error:
        if str(error) != "Rust survivor corpus accounting mismatch":
            raise
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    gates = manifest.get("gate_counts")
    blocks = manifest.get("blocks")
    zero_survivor_omission = (
        isinstance(gates, dict)
        and "survive_sampled_static" not in gates
        and int(manifest.get("survivor_count", -1)) == 0
        and isinstance(blocks, list)
        and bool(blocks)
        and all(
            int(block.get("survivors", -1)) == 0
            and isinstance(block.get("survivor_export"), dict)
            and int(block["survivor_export"].get("record_count", -1)) == 0
            for block in blocks
        )
        and sum(int(value) for value in gates.values()) == expected_end - expected_start
    )
    if not zero_survivor_omission:
        raise ValueError("Rust survivor corpus accounting mismatch")
    normalized = json.loads(json.dumps(manifest))
    normalized["gate_counts"]["survive_sampled_static"] = 0
    audit_path = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.zero-survivor-audit.json"
    )
    try:
        audit_path.write_text(json.dumps(normalized), encoding="utf-8")
        audit = verify_generated_manifest(
            audit_path,
            survivor_directory,
            generator_path,
            expected_start=expected_start,
            expected_end=expected_end,
            equivalence_samples_per_block=equivalence_samples_per_block,
        )
    finally:
        audit_path.unlink(missing_ok=True)
    audit["manifest_path"] = str(manifest_path)
    audit["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    audit["verification_normalization"] = {
        "kind": "omitted-zero-survive-sampled-static-gate",
        "original_manifest_sha256": audit["manifest_sha256"],
        "accepted_value": 0,
    }
    return audit


def streaming_gpu_block_evaluator(lease: WorkLease) -> dict[str, Any]:
    """Existing binary GPU screen plus a bounded CPU/GPU status equivalence sample."""
    started_ns = time.time_ns()
    payload = lease.payload
    config, _, hessians = _assets(
        str(payload["generator_config_path"]), str(payload["generator_config_sha256"])
    )
    records, equivalence = _load_block(payload)
    import cupy as cp

    assets, cache_reused = _cuda_assets(str(payload["generator_config_sha256"]), hessians)
    count = len(records)
    if count:
        device_terms = cp.asarray(np.ascontiguousarray(records["term_ids"]))
        device_counts = cp.asarray(np.ascontiguousarray(records["term_count"]))
        device_masks = cp.asarray(np.ascontiguousarray(records["sign_mask"]))
        gpu_statuses = cp.empty(count, dtype=cp.uint8)
        fail_samples = cp.empty(count, dtype=cp.uint16)
        gpu_margins = cp.empty(count, dtype=cp.float64)
        tolerance = float(config["convexity_tolerance"])
        guard = float(payload["ambiguity_guard"])
        assets.kernel(
            ((count + 255) // 256,),
            (256,),
            (
                device_terms,
                device_counts,
                device_masks,
                assets.hessians,
                np.int32(assets.sample_count),
                np.float64(config["coupling_magnitude"]),
                np.float64(tolerance - guard),
                np.float64(tolerance + guard),
                gpu_statuses,
                fail_samples,
                gpu_margins,
                np.int32(count),
            ),
        )
        cp.cuda.Device().synchronize()
        host_gpu = cp.asnumpy(gpu_statuses)
        host_margins = cp.asnumpy(gpu_margins)
    else:
        host_gpu = np.empty(0, dtype=np.uint8)
        host_margins = np.empty(0, dtype=np.float64)
    result = _make_result(
        payload,
        "gpu_cupy_binary_cached",
        host_gpu,
        host_margins,
        equivalence,
        (time.time_ns() - started_ns) / 1e9,
        cache_reused,
    )
    positions = list(equivalence["positions"])
    sampled = np.array(records[positions], dtype=records.dtype) if positions else records[:0]
    cpu_statuses, _ = _screen_cpu(
        sampled,
        hessians,
        float(config["coupling_magnitude"]),
        float(config["convexity_tolerance"]),
        float(payload["ambiguity_guard"]),
    )
    if len(sampled):
        device_terms = cp.asarray(np.ascontiguousarray(sampled["term_ids"]))
        device_counts = cp.asarray(np.ascontiguousarray(sampled["term_count"]))
        device_masks = cp.asarray(np.ascontiguousarray(sampled["sign_mask"]))
        gpu_statuses = cp.empty(len(sampled), dtype=cp.uint8)
        fail_samples = cp.empty(len(sampled), dtype=cp.uint16)
        margins = cp.empty(len(sampled), dtype=cp.float64)
        tolerance = float(config["convexity_tolerance"])
        guard = float(payload["ambiguity_guard"])
        assets.kernel(
            (1,),
            (256,),
            (
                device_terms,
                device_counts,
                device_masks,
                assets.hessians,
                np.int32(assets.sample_count),
                np.float64(config["coupling_magnitude"]),
                np.float64(tolerance - guard),
                np.float64(tolerance + guard),
                gpu_statuses,
                fail_samples,
                margins,
                np.int32(len(sampled)),
            ),
        )
        cp.cuda.Device().synchronize()
        host_gpu = cp.asnumpy(gpu_statuses)
    else:
        host_gpu = np.empty(0, dtype=np.uint8)
    cpu_root = hashlib.sha256(cpu_statuses.tobytes()).hexdigest()
    gpu_root = hashlib.sha256(host_gpu.tobytes()).hexdigest()
    if cpu_root != gpu_root or not np.array_equal(cpu_statuses, host_gpu):
        raise ValueError("streaming CPU/GPU status sample mismatch")
    result["streaming_cpu_gpu_equivalence"] = {
        "sample_count": len(sampled),
        "positions": positions,
        "cpu_status_root_sha256": cpu_root,
        "gpu_status_root_sha256": gpu_root,
        "all_equal": True,
    }
    promotion_directory = Path(payload["promotion_directory"]).resolve()
    promotion_directory.mkdir(parents=True, exist_ok=True)
    promotion_path = promotion_directory / (
        f"promotion-{int(payload['sequence']):08}-"
        f"{int(payload['start_ordinal'])}-{int(payload['end_ordinal_exclusive'])}.bin"
    )
    temporary = promotion_path.with_suffix(promotion_path.suffix + ".tmp")
    selected = np.flatnonzero(host_gpu != 0)
    with temporary.open("wb") as handle:
        handle.write(
            PROMOTION_HEADER.pack(
                PROMOTION_MAGIC,
                1,
                PROMOTION_RECORD.size,
                int(payload["start_ordinal"]),
                int(payload["end_ordinal_exclusive"]),
                len(selected),
            )
        )
        for position in selected:
            record = records[int(position)]
            handle.write(
                PROMOTION_RECORD.pack(
                    int(host_gpu[int(position)]),
                    int(record["ordinal"]),
                    int(record["term_count"]),
                    int(record["sign_mask"]),
                    0,
                    *map(int, record["term_ids"]),
                )
            )
    temporary.replace(promotion_path)
    promotion_sha = _file_sha(promotion_path)
    result["promotion_survivor_export"] = {
        "schema_version": "sigma-promotion-survivor-block-1.0",
        "path": str(promotion_path),
        "sha256": promotion_sha,
        "record_format": "SGPROM1/1 fixed-25-byte status-plus-SGSURV2-identity",
        "record_count": len(selected),
        "pass_count": int(np.count_nonzero(host_gpu == 1)),
        "ambiguous_count": int(np.count_nonzero(host_gpu == 2)),
        "source_block_sha256": payload["block_sha256"],
        "source_manifest_sha256": payload["manifest_sha256"],
        "generator_config_sha256": payload["generator_config_sha256"],
        "manifest_verification_normalization": payload.get(
            "manifest_verification_normalization"
        ),
    }
    result["manifest_verification_normalization"] = payload.get(
        "manifest_verification_normalization"
    )
    result["streaming_timing"] = {
        "started_time_ns": started_ns,
        "ended_time_ns": time.time_ns(),
    }
    validate_binary_result(result, payload)
    return result


STREAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS rust_stream_source (
  source_id TEXT PRIMARY KEY,
  identity_sha256 TEXT NOT NULL,
  next_ordinal INTEGER NOT NULL,
  stop_ordinal INTEGER NOT NULL,
  sequence INTEGER NOT NULL,
  deadline_utc TEXT NOT NULL,
  owner_id TEXT,
  owner_lease_expires_utc TEXT
);
CREATE TABLE IF NOT EXISTS rust_stream_chunks (
  source_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  start_ordinal INTEGER NOT NULL,
  end_ordinal_exclusive INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('generating','verified','enqueued')),
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


class RustStreamingProducer:
    """One-owner restart-safe Rust chunk producer feeding binary GPU work."""

    def __init__(
        self,
        coordinator: PersistentParallelSearch,
        config: dict[str, Any],
        *,
        owner_id: str | None = None,
    ) -> None:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported Rust streaming schema")
        if config.get("external_paid_llm_calls") is not False:
            raise ValueError("paid LLM calls must remain disabled")
        if config.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("streaming source eligibility is not fail-closed")
        self.coordinator = coordinator
        self.config = config
        self.owner_id = owner_id or f"producer-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self.generator_path = Path(config["generator_config_path"]).resolve()
        self.binary_path = Path(config["generator_binary_path"]).resolve()
        self.output_directory = Path(config["output_directory"]).resolve()
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
            raise ValueError("streaming formula budget exceeds one billion")
        if self.chunk_size <= 0 or self.start % self.chunk_size:
            raise ValueError("streaming start must align to a positive chunk size")
        self.stop = self.start + self.formula_count
        total = total_search_count(
            int(self.generator["basis_count"]), int(self.generator["max_action_terms"])
        )
        if self.start < 0 or self.stop > total:
            raise ValueError("streaming interval exceeds the generator search space")
        self.disk_budget = int(config["maximum_disk_bytes"])
        chunks = (self.formula_count + self.chunk_size - 1) // self.chunk_size
        worst_case = self.formula_count * (RECORD.size + PROMOTION_RECORD.size) + chunks * (
            HEADER.size + MANIFEST_ALLOWANCE_BYTES
        )
        if self.disk_budget <= 0 or worst_case > self.disk_budget:
            raise ValueError("streaming disk budget fails worst-case preflight")
        maximum_wall = float(config["maximum_wall_seconds"])
        if maximum_wall <= 0:
            raise ValueError("streaming wall budget must be positive")
        if float(config["producer_lease_seconds"]) <= 0:
            raise ValueError("producer lease must be positive")
        if int(config["target_pending_chunks"]) <= 0:
            raise ValueError("streaming backpressure target must be positive")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.promotion_directory = Path(
            config.get("promotion_directory", self.output_directory / "promotion")
        ).resolve()
        self.promotion_directory.mkdir(parents=True, exist_ok=True)
        identity = {
            "generator_sha256": self.generator_sha,
            "binary_sha256": self.binary_sha,
            "start": self.start,
            "stop": self.stop,
            "chunk_size": self.chunk_size,
            "threads": int(config["threads"]),
            "disk_budget": self.disk_budget,
            "eligibility": ELIGIBILITY,
        }
        self.identity_sha = _sha(identity)
        self.source_id = f"RUSTSTREAM-{self.identity_sha[:24]}"
        with coordinator.connect() as connection:
            connection.executescript(STREAM_SCHEMA)
            row = connection.execute(
                "SELECT * FROM rust_stream_source WHERE source_id=?", (self.source_id,)
            ).fetchone()
            if row is None:
                deadline = datetime.now(UTC) + timedelta(
                    seconds=float(config["maximum_wall_seconds"])
                )
                connection.execute(
                    "INSERT INTO rust_stream_source VALUES (?,?,?,?,0,?,NULL,NULL)",
                    (self.source_id, self.identity_sha, self.start, self.stop, deadline.isoformat()),
                )
            elif row["identity_sha256"] != self.identity_sha or int(row["stop_ordinal"]) != self.stop:
                raise ValueError("refusing to resume a changed Rust stream")
        self._recover_chunks()

    def _acquire_owner(self) -> None:
        lease_seconds = float(self.config["producer_lease_seconds"])
        if lease_seconds <= 0:
            raise ValueError("producer lease must be positive")
        now = datetime.now(UTC)
        expiry = now + timedelta(seconds=lease_seconds)
        with self.coordinator.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT owner_id,owner_lease_expires_utc FROM rust_stream_source WHERE source_id=?",
                (self.source_id,),
            ).fetchone()
            active = (
                row["owner_id"]
                and row["owner_id"] != self.owner_id
                and row["owner_lease_expires_utc"]
                and datetime.fromisoformat(row["owner_lease_expires_utc"]) > now
            )
            if active:
                raise RuntimeError("another Rust streaming producer owns the source")
            connection.execute(
                "UPDATE rust_stream_source SET owner_id=?,owner_lease_expires_utc=? "
                "WHERE source_id=?",
                (self.owner_id, expiry.isoformat(), self.source_id),
            )

    def release_owner(self) -> None:
        with self.coordinator.connect() as connection:
            connection.execute(
                "UPDATE rust_stream_source SET owner_id=NULL,owner_lease_expires_utc=NULL "
                "WHERE source_id=? AND owner_id=?",
                (self.source_id, self.owner_id),
            )

    def _manifest_path(self, sequence: int, start: int, end: int) -> Path:
        return self.output_directory / f"stream-{sequence:06}-{start}-{end}.json"

    def _verify_chunk(self, sequence: int, start: int, end: int, manifest_path: Path) -> dict[str, Any]:
        audit = _verify_stream_manifest(
            manifest_path,
            self.output_directory,
            self.generator_path,
            expected_start=start,
            expected_end=end,
            equivalence_samples_per_block=int(self.config["equivalence_samples_per_chunk"]),
        )
        if len(audit["blocks"]) != 1:
            raise ValueError("streaming chunk must produce exactly one SGSURV2 block")
        block = audit["blocks"][0]
        return {
            "sequence": sequence,
            "manifest_sha256": audit["manifest_sha256"],
            "block_path": str(self.output_directory / block["file"]),
            "block_sha256": block["file_sha256"],
            "block_size_bytes": (self.output_directory / block["file"]).stat().st_size,
            "record_count": block["record_count"],
            "block_index": block["block_index"],
        }

    def _recover_chunks(self) -> None:
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rust_stream_chunks WHERE source_id=? AND state='generating'",
                (self.source_id,),
            ).fetchall()
        for row in rows:
            manifest = Path(row["manifest_path"])
            try:
                verified = self._verify_chunk(
                    int(row["sequence"]),
                    int(row["start_ordinal"]),
                    int(row["end_ordinal_exclusive"]),
                    manifest,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            with self.coordinator.connect() as connection:
                connection.execute(
                    "UPDATE rust_stream_chunks SET state='verified',manifest_sha256=?,block_path=?,"
                    "block_sha256=?,block_size_bytes=?,record_count=? WHERE source_id=? AND sequence=?",
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

    def _generate(self, sequence: int, start: int, end: int) -> dict[str, Any]:
        manifest_path = self._manifest_path(sequence, start, end)
        with self.coordinator.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO rust_stream_chunks "
                "(source_id,sequence,start_ordinal,end_ordinal_exclusive,state,manifest_path,"
                "generation_started_time_ns) VALUES (?,?,?,?,?,?,?)",
                (
                    self.source_id,
                    sequence,
                    start,
                    end,
                    "generating",
                    str(manifest_path),
                    time.time_ns(),
                ),
            )
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
            str(int(self.config["threads"])),
            "--block-size",
            str(self.chunk_size),
            "--survivor-dir",
            str(self.output_directory),
        ]
        with self.coordinator.connect() as connection:
            deadline = datetime.fromisoformat(
                connection.execute(
                    "SELECT deadline_utc FROM rust_stream_source WHERE source_id=?",
                    (self.source_id,),
                ).fetchone()[0]
            )
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("Rust streaming wall budget exhausted")
        process = subprocess.Popen(
            command,
            cwd=self.output_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        heartbeat_seconds = max(0.1, float(self.config["producer_lease_seconds"]) / 3.0)
        while True:
            remaining = (deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                process.kill()
                process.communicate()
                raise TimeoutError("Rust streaming wall budget exhausted")
            try:
                stdout, stderr = process.communicate(timeout=min(heartbeat_seconds, remaining))
                break
            except subprocess.TimeoutExpired:
                # Refreshing under BEGIN IMMEDIATE ensures no second producer
                # can take ownership during a long bounded Rust chunk.
                self._acquire_owner()
        ended_ns = time.time_ns()
        if process.returncode != 0:
            raise RuntimeError(f"Rust stream producer failed: {stderr[-2000:]}")
        verified = self._verify_chunk(sequence, start, end, manifest_path)
        if _directory_bytes(self.output_directory) > self.disk_budget:
            raise ValueError("Rust stream exceeded its disk budget")
        with self.coordinator.connect() as connection:
            connection.execute(
                "UPDATE rust_stream_chunks SET state='verified',manifest_sha256=?,block_path=?,"
                "block_sha256=?,block_size_bytes=?,record_count=?,generation_ended_time_ns=?,"
                "producer_stdout=? WHERE source_id=? AND sequence=?",
                (
                    verified["manifest_sha256"],
                    verified["block_path"],
                    verified["block_sha256"],
                    verified["block_size_bytes"],
                    verified["record_count"],
                    ended_ns,
                    stdout[-4000:],
                    self.source_id,
                    sequence,
                ),
            )
        return verified

    def _payload(self, row: sqlite3.Row) -> dict[str, Any]:
        manifest_path = Path(row["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            "manifest_path": str(manifest_path),
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

    def refill(self) -> dict[str, Any]:
        self._acquire_owner()
        target = int(self.config["target_pending_chunks"])
        accepted = duplicates = generated = backpressured = waiting_for_consumer = 0
        while True:
            telemetry = self.coordinator.telemetry()
            if int(telemetry["queue"]["pending"]) >= target:
                break
            # Do not materialize the next chunk merely to fill a queue.  After
            # the first enqueue, wait until the GPU owner has claimed it; Rust
            # generation of the next exact interval then overlaps that lease.
            gpu_lane = telemetry["lanes"]["gpu"]
            if int(gpu_lane["queued"]) and not int(gpu_lane["running"]):
                waiting_for_consumer += 1
                break
            with self.coordinator.connect() as connection:
                source = connection.execute(
                    "SELECT * FROM rust_stream_source WHERE source_id=?", (self.source_id,)
                ).fetchone()
                start = int(source["next_ordinal"])
                stop = int(source["stop_ordinal"])
                sequence = int(source["sequence"])
                row = connection.execute(
                    "SELECT * FROM rust_stream_chunks WHERE source_id=? AND sequence=?",
                    (self.source_id, sequence),
                ).fetchone()
            if start >= stop:
                break
            end = min(stop, start + self.chunk_size)
            if row is None or row["state"] == "generating":
                self._generate(sequence, start, end)
                generated += 1
                with self.coordinator.connect() as connection:
                    row = connection.execute(
                        "SELECT * FROM rust_stream_chunks WHERE source_id=? AND sequence=?",
                        (self.source_id, sequence),
                    ).fetchone()
            outcome = self.coordinator.enqueue([self._payload(row)], lane="gpu")
            if outcome["accepted"] or outcome["duplicate"]:
                accepted += outcome["accepted"]
                duplicates += outcome["duplicate"]
                with self.coordinator.connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "UPDATE rust_stream_chunks SET state='enqueued' WHERE source_id=? AND sequence=?",
                        (self.source_id, sequence),
                    )
                    connection.execute(
                        "UPDATE rust_stream_source SET next_ordinal=?,sequence=sequence+1 "
                        "WHERE source_id=? AND next_ordinal=?",
                        (end, self.source_id, start),
                    )
            else:
                backpressured += outcome["backpressured"]
                break
        return {
            "accepted_chunks": accepted,
            "duplicate_chunks": duplicates,
            "generated_chunks": generated,
            "backpressured_chunks": backpressured,
            "waiting_for_consumer": waiting_for_consumer,
            "cursor": self.status(),
        }

    def status(self) -> dict[str, Any]:
        with self.coordinator.connect() as connection:
            source = connection.execute(
                "SELECT * FROM rust_stream_source WHERE source_id=?", (self.source_id,)
            ).fetchone()
            counts = {
                row["state"]: int(row["count"])
                for row in connection.execute(
                    "SELECT state,COUNT(*) AS count FROM rust_stream_chunks "
                    "WHERE source_id=? GROUP BY state",
                    (self.source_id,),
                )
            }
        return {
            "source_id": self.source_id,
            "next_ordinal": int(source["next_ordinal"]),
            "stop_ordinal_exclusive": int(source["stop_ordinal"]),
            "exhausted": int(source["next_ordinal"]) >= int(source["stop_ordinal"]),
            "chunk_counts": counts,
            "owner_id": source["owner_id"],
        }


def _overlap_seconds(
    producer_intervals: list[tuple[int, int]], consumer_intervals: list[tuple[int, int]]
) -> float:
    overlap_ns = 0
    for producer_start, producer_end in producer_intervals:
        for consumer_start, consumer_end in consumer_intervals:
            overlap_ns += max(0, min(producer_end, consumer_end) - max(producer_start, consumer_start))
    return overlap_ns / 1e9


def configure_rust_streaming_execution(
    execution_config: dict[str, Any], stream_config: dict[str, Any]
) -> dict[str, Any]:
    configured = configure_binary_evaluators(execution_config, {"lane_cycle": ["gpu"]})
    configured["supervisor"]["gpu_evaluator"] = (
        "sigma_theory_compiler.rust_streaming_search:streaming_gpu_block_evaluator"
    )
    chunk_count = (
        int(stream_config["formula_count"]) + int(stream_config["chunk_formula_count"]) - 1
    ) // int(stream_config["chunk_formula_count"])
    configured["budget"]["maximum_tasks"] = min(
        int(configured["budget"]["maximum_tasks"]), chunk_count
    )
    configured["budget"]["maximum_wall_seconds"] = min(
        float(configured["budget"]["maximum_wall_seconds"]),
        float(stream_config["maximum_wall_seconds"]),
    )
    configured["supervisor"]["maximum_wall_seconds_per_run"] = configured["budget"][
        "maximum_wall_seconds"
    ]
    return configured


def run_rust_streaming_search(
    database: str | Path,
    execution_config: dict[str, Any],
    resource_profile: dict[str, Any],
    stream_config: dict[str, Any],
    telemetry_path: str | Path,
    *,
    output_report: str | Path | None = None,
    external_stop_path: str | Path | None = None,
    stop_reason_callback: Any = None,
    status_callback: Any = None,
) -> dict[str, Any]:
    configured = configure_rust_streaming_execution(execution_config, stream_config)
    coordinator = PersistentParallelSearch(database, configured, resource_profile)
    producer = RustStreamingProducer(coordinator, stream_config)
    started_ns = time.time_ns()
    try:
        supervisor = PersistentParallelSupervisor(
            database, configured, resource_profile, telemetry_path
        ).run(
            refill_callback=producer.refill,
            external_stop_path=external_stop_path,
            stop_reason_callback=stop_reason_callback,
            status_callback=status_callback,
        )
    finally:
        producer.release_owner()
    ended_ns = time.time_ns()
    with coordinator.connect() as connection:
        chunks = connection.execute(
            "SELECT * FROM rust_stream_chunks WHERE source_id=? ORDER BY sequence",
            (producer.source_id,),
        ).fetchall()
        work = connection.execute(
            "SELECT state,payload_json,result_json FROM work ORDER BY ordinal"
        ).fetchall()
    expected_start = producer.start
    chunk_chain: list[dict[str, Any]] = []
    exact_interval_complete = True
    for row in chunks:
        start = int(row["start_ordinal"])
        end = int(row["end_ordinal_exclusive"])
        exact_interval_complete &= start == expected_start and start < end <= producer.stop
        expected_start = end
        chunk_chain.append(
            {
                "sequence": int(row["sequence"]),
                "start_ordinal": start,
                "end_ordinal_exclusive": end,
                "manifest_sha256": row["manifest_sha256"],
                "block_sha256": row["block_sha256"],
            }
        )
    exact_interval_complete &= expected_start == producer.stop
    cumulative_producer_intervals = [
        (int(row["generation_started_time_ns"]), int(row["generation_ended_time_ns"]))
        for row in chunks
        if row["generation_started_time_ns"] and row["generation_ended_time_ns"]
    ]
    producer_intervals = [
        interval for interval in cumulative_producer_intervals if interval[0] >= started_ns
    ]
    generated_this_run = sum(
        int(row["end_ordinal_exclusive"]) - int(row["start_ordinal"])
        for row in chunks
        if row["generation_started_time_ns"]
        and int(row["generation_started_time_ns"]) >= started_ns
    )
    consumer_intervals: list[tuple[int, int]] = []
    backend_counts: Counter[str] = Counter()
    records = records_this_run = source_formulas_screened_this_run = 0
    equivalence_passed = True
    cache_reused = 0
    for row in work:
        if row["state"] != "succeeded" or not row["result_json"]:
            continue
        payload = json.loads(row["payload_json"])
        result = json.loads(row["result_json"])
        validate_binary_result(result, payload)
        backend_counts[result["backend"]] += 1
        records += int(result["block"]["record_count"])
        equivalence_passed &= result["streaming_cpu_gpu_equivalence"]["all_equal"]
        cache_reused += result["cuda_assets_reused"] is True
        timing = result["streaming_timing"]
        interval = (int(timing["started_time_ns"]), int(timing["ended_time_ns"]))
        if interval[0] >= started_ns:
            consumer_intervals.append(interval)
            records_this_run += int(result["block"]["record_count"])
            source_formulas_screened_this_run += int(result["block"][
                "end_ordinal_exclusive"
            ]) - int(result["block"]["start_ordinal"])
    producer_seconds = sum(end - start for start, end in producer_intervals) / 1e9
    consumer_seconds = sum(end - start for start, end in consumer_intervals) / 1e9
    wall_seconds = (ended_ns - started_ns) / 1e9
    overlap = _overlap_seconds(producer_intervals, consumer_intervals)
    cursor = producer.status()
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_id": producer.source_id,
        "provenance": {
            "generator_config_sha256": producer.generator_sha,
            "generator_binary_sha256": producer.binary_sha,
            "verified_chunk_chain_sha256": _sha(chunk_chain),
        },
        "formula_count": int(stream_config["formula_count"]),
        "chunk_count": len(chunks),
        "exact_interval": {
            "start_ordinal": producer.start,
            "end_ordinal_exclusive": producer.stop,
            "complete": exact_interval_complete,
        },
        "survivor_records": records,
        "producer": {
            "busy_seconds": producer_seconds,
            "formulas_generated_this_run": generated_this_run,
            "formulas_generated_cumulative": sum(
                int(row["end_ordinal_exclusive"]) - int(row["start_ordinal"])
                for row in chunks
            ),
            "formulas_per_second": (
                generated_this_run / producer_seconds if producer_seconds else None
            ),
            "single_owner_utilization": producer_seconds / wall_seconds if wall_seconds else 0.0,
        },
        "consumer": {
            "busy_seconds": consumer_seconds,
            "survivor_records_screened_this_run": records_this_run,
            "survivor_records_screened_cumulative": records,
            "records_per_second": (
                records_this_run / consumer_seconds if consumer_seconds else None
            ),
            "single_gpu_owner_utilization": consumer_seconds / wall_seconds if wall_seconds else 0.0,
            "cached_chunks": cache_reused,
            "backend_counts": dict(backend_counts),
        },
        "combined": {
            "wall_seconds": wall_seconds,
            "source_formulas_screened_this_run": source_formulas_screened_this_run,
            "formulas_per_second": (
                source_formulas_screened_this_run / wall_seconds if wall_seconds else None
            ),
            "producer_consumer_overlap_seconds": overlap,
            "overlap_observed": overlap > 0,
        },
        "cursor": cursor,
        "supervisor": supervisor,
        "all_work_succeeded": (
            exact_interval_complete
            and bool(work)
            and len(work) == len(chunks)
            and all(row["state"] == "succeeded" for row in work)
        ),
        "cpu_gpu_equivalence_passed": equivalence_passed,
        "disk_bytes": _directory_bytes(Path(stream_config["output_directory"]).resolve()),
        "maximum_disk_bytes": int(stream_config["maximum_disk_bytes"]),
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    report["content_sha256"] = _sha(report)
    if output_report is not None:
        output = Path(output_report).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    return report
