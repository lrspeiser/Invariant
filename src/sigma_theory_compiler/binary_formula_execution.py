from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_screen import KERNEL
from .high_throughput import build_basis, decode_ordinal, total_search_count
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .persistent_parallel_supervisor import PersistentParallelSupervisor
from .real_formula_execution import _assets, cuda_available
from .survivors import HEADER, MAGIC, RECORD

CURSOR_SCHEMA = "sigma-binary-block-refill-1.0"
RESULT_SCHEMA = "sigma-binary-formula-block-result-1.0"
ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
}
RECORD_DTYPE = np.dtype(
    [
        ("ordinal", "<u8"),
        ("term_count", "u1"),
        ("sign_mask", "u1"),
        ("reserved", "<u2"),
        ("term_ids", "<u2", (6,)),
    ]
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_directory(manifest_path: Path, value: str | Path) -> Path:
    requested = Path(value)
    candidates = [requested]
    if not requested.is_absolute():
        candidates = [manifest_path.parent / requested, Path.cwd() / requested]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(f"survivor export directory is unavailable: {value}")


def _sample_positions(count: int, requested: int, block_sha256: str) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    limit = min(count, requested)
    positions = {0, count - 1}
    cursor = bytes.fromhex(block_sha256)
    sequence = 0
    while len(positions) < limit:
        cursor = hashlib.sha256(cursor + sequence.to_bytes(8, "little")).digest()
        positions.add(int.from_bytes(cursor[:8], "little") % count)
        sequence += 1
    return sorted(positions)[:limit]


def _load_block(payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if payload.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("binary block data eligibility is not fail-closed")
    path = Path(str(payload["block_path"]))
    expected_size = int(payload["block_size_bytes"])
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("binary block size mismatch")
    block_sha = _sha256_path(path)
    if block_sha != payload["block_sha256"]:
        raise ValueError("binary block hash mismatch")
    with path.open("rb") as handle:
        raw = handle.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError("truncated binary block header")
    magic, version, record_size, block_index, start, end, count = HEADER.unpack(raw)
    expected_header = (
        MAGIC,
        1,
        RECORD.size,
        int(payload["block_index"]),
        int(payload["start_ordinal"]),
        int(payload["end_ordinal_exclusive"]),
        int(payload["record_count"]),
    )
    if (magic, version, record_size, block_index, start, end, count) != expected_header:
        raise ValueError("binary block header mismatch")
    records = (
        np.memmap(path, dtype=RECORD_DTYPE, mode="r", offset=HEADER.size, shape=(count,))
        if count
        else np.empty(0, dtype=RECORD_DTYPE)
    )
    if RECORD_DTYPE.itemsize != RECORD.size:
        raise RuntimeError("survivor NumPy dtype does not match the Rust record")
    if count:
        ordinals = records["ordinal"]
        term_counts = records["term_count"]
        term_ids = records["term_ids"]
        positions = np.arange(6)[None, :]
        active = positions < term_counts[:, None]
        if (
            np.any(records["reserved"] != 0)
            or np.any((term_counts < 1) | (term_counts > 6))
            or np.any(ordinals[1:] <= ordinals[:-1])
            or int(ordinals[0]) < start
            or int(ordinals[-1]) >= end
            or np.any(term_ids[active] >= int(payload["basis_count"]))
            or np.any(term_ids[~active] != 0xFFFF)
        ):
            raise ValueError("invalid binary survivor records")
    samples = _sample_positions(
        count, int(payload["equivalence_samples"]), str(payload["block_sha256"])
    )
    for position in samples:
        record = records[position]
        decoded = decode_ordinal(
            int(payload["basis_count"]),
            int(payload["max_action_terms"]),
            int(record["ordinal"]),
        )
        decoded_mask = sum(
            (1 << index) for index, sign in enumerate(decoded["signs"]) if sign > 0
        )
        count_at_position = int(record["term_count"])
        if (
            decoded["term_ids"]
            != [int(value) for value in record["term_ids"][:count_at_position]]
            or decoded_mask != int(record["sign_mask"])
        ):
            raise ValueError("binary survivor differs from ordinal decoder")
    return records, {
        "method": "sha256-seeded record positions",
        "requested": int(payload["equivalence_samples"]),
        "checked": len(samples),
        "positions": samples,
        "all_equal": True,
    }


def _screen_cpu(
    records: np.ndarray,
    hessians: np.ndarray,
    coupling: float,
    tolerance: float,
    guard: float,
) -> tuple[np.ndarray, np.ndarray]:
    statuses = np.empty(len(records), dtype=np.uint8)
    margins = np.empty(len(records), dtype=np.float64)
    low, high = tolerance - guard, tolerance + guard
    for row, record in enumerate(records):
        count = int(record["term_count"])
        mask = int(record["sign_mask"])
        signs = np.array([1.0 if mask & (1 << position) else -1.0 for position in range(count)])
        candidate = np.array([1.0, 0.0, 0.0]) + coupling * np.sum(
            hessians[record["term_ids"][:count]] * signs[:, None, None], axis=0
        )
        hdd, hdp, hpp = candidate.T
        eigenvalue = 0.5 * (
            hdd + hpp - np.sqrt(np.maximum((hdd - hpp) ** 2 + 4 * hdp**2, 0.0))
        )
        margin = float(np.min(eigenvalue))
        margins[row] = margin
        statuses[row] = 0 if not np.isfinite(margin) or margin <= low else (2 if margin <= high else 1)
    return statuses, margins


@dataclass(frozen=True)
class _CudaAssets:
    hessians: Any
    kernel: Any
    sample_count: int


_CUDA_CACHE: dict[tuple[str, int], _CudaAssets] = {}


def _cuda_assets(config_sha256: str, hessians: np.ndarray) -> tuple[_CudaAssets, bool]:
    import cupy as cp

    device = int(cp.cuda.Device().id)
    key = (config_sha256, device)
    cached = key in _CUDA_CACHE
    if not cached:
        _CUDA_CACHE[key] = _CudaAssets(
            cp.asarray(np.ascontiguousarray(hessians)),
            cp.RawKernel(KERNEL, "dense_screen"),
            int(hessians.shape[1]),
        )
    return _CUDA_CACHE[key], cached


def _make_result(
    payload: dict[str, Any],
    backend: str,
    statuses: np.ndarray,
    margins: np.ndarray,
    equivalence: dict[str, Any],
    elapsed: float,
    cuda_cache_reused: bool | None,
) -> dict[str, Any]:
    counts = Counter(map(int, statuses))
    result = {
        "schema_version": RESULT_SCHEMA,
        "backend": backend,
        "source": {
            "source_id": payload["source_id"],
            "manifest_sha256": payload["manifest_sha256"],
            "block_sha256": payload["block_sha256"],
            "generator_config_sha256": payload["generator_config_sha256"],
        },
        "block": {
            "block_index": int(payload["block_index"]),
            "start_ordinal": int(payload["start_ordinal"]),
            "end_ordinal_exclusive": int(payload["end_ordinal_exclusive"]),
            "record_count": int(payload["record_count"]),
        },
        "counts": {"reject": counts[0], "pass": counts[1], "ambiguous": counts[2]},
        "status_root_sha256": hashlib.sha256(statuses.tobytes()).hexdigest(),
        "reported_margin_minimum": float(np.min(margins)) if len(margins) else None,
        "ordinal_equivalence": equivalence,
        "cuda_assets_reused": cuda_cache_reused,
        "elapsed_seconds": elapsed,
        "records_per_second": len(statuses) / elapsed if elapsed > 0 else None,
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
        "interpretation": (
            "A pass survives only the frozen 343-point sampled-static convexity screen; "
            "it is not a covariant-health or observational claim."
        ),
    }
    validate_binary_result(result, payload)
    return result


def validate_binary_result(result: dict[str, Any], payload: dict[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unsupported binary formula result schema")
    if result.get("backend") not in {"cpu_numpy_binary", "gpu_cupy_binary_cached"}:
        raise ValueError("unsupported binary formula backend")
    expected_source = {
        "source_id": payload["source_id"],
        "manifest_sha256": payload["manifest_sha256"],
        "block_sha256": payload["block_sha256"],
        "generator_config_sha256": payload["generator_config_sha256"],
    }
    if result.get("source") != expected_source:
        raise ValueError("binary result source identity mismatch")
    block = result.get("block", {})
    if block != {
        "block_index": int(payload["block_index"]),
        "start_ordinal": int(payload["start_ordinal"]),
        "end_ordinal_exclusive": int(payload["end_ordinal_exclusive"]),
        "record_count": int(payload["record_count"]),
    }:
        raise ValueError("binary result block identity mismatch")
    counts = result.get("counts", {})
    if set(counts) != {"reject", "pass", "ambiguous"} or any(
        not isinstance(value, int) or value < 0 for value in counts.values()
    ):
        raise ValueError("binary result counts are invalid")
    if sum(counts.values()) != int(payload["record_count"]):
        raise ValueError("binary result accounting mismatch")
    equivalence = result.get("ordinal_equivalence", {})
    if equivalence.get("all_equal") is not True or equivalence.get("checked", 0) < 0:
        raise ValueError("binary result lacks ordinal equivalence")
    if result.get("data_eligibility") != {**ELIGIBILITY, "paid_llm_calls": False, "passed": True}:
        raise ValueError("binary result violates data eligibility")
    root = result.get("status_root_sha256")
    if not isinstance(root, str) or len(root) != 64:
        raise ValueError("binary result lacks a status root")


def cpu_binary_block_evaluator(lease: WorkLease) -> dict[str, Any]:
    started = time.perf_counter()
    payload = lease.payload
    config, _, hessians = _assets(
        str(payload["generator_config_path"]), str(payload["generator_config_sha256"])
    )
    records, equivalence = _load_block(payload)
    statuses, margins = _screen_cpu(
        records,
        hessians,
        float(config["coupling_magnitude"]),
        float(config["convexity_tolerance"]),
        float(payload["ambiguity_guard"]),
    )
    return _make_result(
        payload,
        "cpu_numpy_binary",
        statuses,
        margins,
        equivalence,
        time.perf_counter() - started,
        None,
    )


def gpu_binary_block_evaluator(lease: WorkLease) -> dict[str, Any]:
    started = time.perf_counter()
    available, reason = cuda_available()
    if not available:
        raise RuntimeError(f"CUDA evaluator unavailable: {reason}")
    import cupy as cp

    payload = lease.payload
    config, _, hessians = _assets(
        str(payload["generator_config_path"]), str(payload["generator_config_sha256"])
    )
    records, equivalence = _load_block(payload)
    assets, cache_reused = _cuda_assets(str(payload["generator_config_sha256"]), hessians)
    count = len(records)
    if count:
        device_terms = cp.asarray(np.ascontiguousarray(records["term_ids"]))
        device_counts = cp.asarray(np.ascontiguousarray(records["term_count"]))
        device_masks = cp.asarray(np.ascontiguousarray(records["sign_mask"]))
        statuses = cp.empty(count, dtype=cp.uint8)
        fail_samples = cp.empty(count, dtype=cp.uint16)
        margins = cp.empty(count, dtype=cp.float64)
        threads = 256
        tolerance = float(config["convexity_tolerance"])
        guard = float(payload["ambiguity_guard"])
        assets.kernel(
            ((count + threads - 1) // threads,),
            (threads,),
            (
                device_terms,
                device_counts,
                device_masks,
                assets.hessians,
                np.int32(assets.sample_count),
                np.float64(config["coupling_magnitude"]),
                np.float64(tolerance - guard),
                np.float64(tolerance + guard),
                statuses,
                fail_samples,
                margins,
                np.int32(count),
            ),
        )
        cp.cuda.Device().synchronize()
        host_statuses = cp.asnumpy(statuses)
        host_margins = cp.asnumpy(margins)
    else:
        host_statuses = np.empty(0, dtype=np.uint8)
        host_margins = np.empty(0, dtype=np.float64)
    return _make_result(
        payload,
        "gpu_cupy_binary_cached",
        host_statuses,
        host_margins,
        equivalence,
        time.perf_counter() - started,
        cache_reused,
    )


class BinaryBlockQueueRefill:
    """Restart-safe cursor over hash-bound SGSURV2 blocks."""

    def __init__(self, coordinator: PersistentParallelSearch, config: dict[str, Any]) -> None:
        if config.get("schema_version") != CURSOR_SCHEMA:
            raise ValueError("unsupported binary block refill schema")
        if config.get("external_paid_llm_calls") is not False:
            raise ValueError("paid LLM calls must remain disabled")
        if config.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("binary source data eligibility is not fail-closed")
        self.coordinator = coordinator
        self.config = config
        self.manifest_path = Path(config["manifest_path"]).resolve()
        manifest_raw = self.manifest_path.read_bytes()
        self.manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        self.manifest = json.loads(manifest_raw)
        if self.manifest.get("observational_data_opened") is not False:
            raise ValueError("survivor manifest opens observational data")
        self.generator_path = Path(config["generator_config_path"]).resolve()
        generator_raw = self.generator_path.read_bytes()
        self.generator_sha = hashlib.sha256(generator_raw).hexdigest()
        self.generator = json.loads(generator_raw)
        if (
            self.generator.get("observational_data_opened") is not False
            or self.generator_sha != self.manifest.get("config_sha256")
        ):
            raise ValueError("manifest and generator config identity mismatch")
        basis_payload = json.dumps(
            build_basis(int(self.generator["basis_count"])),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        if (
            self.manifest.get("protocol_version") != self.generator.get("protocol_version")
            or int(self.manifest.get("basis_count", -1))
            != int(self.generator["basis_count"])
            or int(self.manifest.get("max_action_terms", -1))
            != int(self.generator["max_action_terms"])
            or self.manifest.get("basis_library_sha256")
            != hashlib.sha256(basis_payload).hexdigest()
            or int(self.manifest.get("total_declared_actions", -1))
            != total_search_count(
                int(self.generator["basis_count"]),
                int(self.generator["max_action_terms"]),
            )
        ):
            raise ValueError("manifest generator-space provenance mismatch")
        root = hashlib.sha256()
        root.update(b"SIGMA-GENERATOR-V2-ROOT\0")
        processed = survivors = 0
        for block in self.manifest.get("blocks", []):
            start, end = int(block["start_ordinal"]), int(block["end_ordinal_exclusive"])
            if int(block.get("processed", -1)) != end - start:
                raise ValueError("manifest block accounting mismatch")
            digest = str(block.get("digest_sha256", ""))
            if len(digest) != 64:
                raise ValueError("manifest block digest is invalid")
            root.update(int(block["block_index"]).to_bytes(8, "little"))
            root.update(start.to_bytes(8, "little"))
            root.update(end.to_bytes(8, "little"))
            root.update(digest.encode("ascii"))
            processed += end - start
            survivors += int(block.get("survivors", 0))
        gate_counts = self.manifest.get("gate_counts", {})
        if (
            root.hexdigest() != self.manifest.get("blocks_root_sha256")
            or processed != int(self.manifest.get("processed_actions", -1))
            or survivors != int(self.manifest.get("survivor_count", -1))
            or sum(int(value) for value in gate_counts.values()) != processed
            or int(gate_counts.get("survive_sampled_static", -1)) != survivors
        ):
            raise ValueError("manifest root or gate accounting mismatch")
        self.directory = _resolve_directory(
            self.manifest_path,
            config.get("survivor_directory", self.manifest["survivor_export_directory"]),
        )
        self.blocks = [block for block in self.manifest["blocks"] if block.get("survivor_export")]
        start = int(config["start_export_block"])
        stop = min(int(config["stop_export_block_exclusive"]), len(self.blocks))
        if not 0 <= start < stop <= len(self.blocks):
            raise ValueError("invalid binary block interval")
        self.start, self.stop = start, stop
        identity = {
            "manifest": self.manifest_sha,
            "generator": self.generator_sha,
            "directory": str(self.directory),
            "start": start,
            "stop": stop,
        }
        self.source_id = f"BINARY-{hashlib.sha256(_canonical(identity).encode()).hexdigest()[:24]}"
        with coordinator.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS binary_block_cursor ("
                "source_id TEXT PRIMARY KEY,manifest_sha256 TEXT NOT NULL,generator_sha256 TEXT NOT NULL,"
                "next_position INTEGER NOT NULL,stop_position INTEGER NOT NULL,sequence INTEGER NOT NULL,"
                "lane_index INTEGER NOT NULL)"
            )
            row = connection.execute(
                "SELECT * FROM binary_block_cursor WHERE source_id=?", (self.source_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO binary_block_cursor VALUES (?,?,?,?,?,0,0)",
                    (self.source_id, self.manifest_sha, self.generator_sha, start, stop),
                )
            elif (
                row["manifest_sha256"] != self.manifest_sha
                or row["generator_sha256"] != self.generator_sha
                or int(row["stop_position"]) != stop
            ):
                raise ValueError("refusing to resume a changed binary source")

    def status(self) -> dict[str, Any]:
        with self.coordinator.connect() as connection:
            row = connection.execute(
                "SELECT * FROM binary_block_cursor WHERE source_id=?", (self.source_id,)
            ).fetchone()
        return {
            "source_id": self.source_id,
            "next_export_block": int(row["next_position"]),
            "stop_export_block_exclusive": int(row["stop_position"]),
            "sequence": int(row["sequence"]),
            "exhausted": int(row["next_position"]) >= int(row["stop_position"]),
        }

    def refill(self) -> dict[str, Any]:
        lanes = list(self.config["lane_cycle"])
        if not lanes or any(lane not in {"cpu", "gpu"} for lane in lanes):
            raise ValueError("lane_cycle must contain cpu/gpu lanes")
        accepted = duplicates = backpressured = 0
        target = int(self.config["target_pending_blocks"])
        while int(self.coordinator.telemetry()["queue"]["pending"]) < target:
            with self.coordinator.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM binary_block_cursor WHERE source_id=?", (self.source_id,)
                ).fetchone()
            position, stop = int(row["next_position"]), int(row["stop_position"])
            if position >= stop:
                break
            block = self.blocks[position]
            export = block["survivor_export"]
            path = self.directory / export["file"]
            payload = {
                "ordinal": int(block["block_index"]),
                "source_id": self.source_id,
                "sequence": int(row["sequence"]),
                "manifest_path": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha,
                "generator_config_path": str(self.generator_path),
                "generator_config_sha256": self.generator_sha,
                "block_path": str(path),
                "block_sha256": export["file_sha256"],
                "block_size_bytes": int(export["file_size_bytes"]),
                "block_index": int(block["block_index"]),
                "start_ordinal": int(block["start_ordinal"]),
                "end_ordinal_exclusive": int(block["end_ordinal_exclusive"]),
                "record_count": int(export["record_count"]),
                "basis_count": int(self.manifest["basis_count"]),
                "max_action_terms": int(self.manifest["max_action_terms"]),
                "equivalence_samples": int(self.config["equivalence_samples_per_block"]),
                "ambiguity_guard": float(self.config["ambiguity_guard"]),
                "data_eligibility": self.config["data_eligibility"],
            }
            lane_index = int(row["lane_index"])
            outcome = self.coordinator.enqueue([payload], lane=lanes[lane_index % len(lanes)])
            if outcome["accepted"] or outcome["duplicate"]:
                accepted += outcome["accepted"]
                duplicates += outcome["duplicate"]
                with self.coordinator.connect() as connection:
                    connection.execute(
                        "UPDATE binary_block_cursor SET next_position=?,sequence=sequence+1,"
                        "lane_index=lane_index+1 WHERE source_id=? AND next_position=?",
                        (position + 1, self.source_id, position),
                    )
            else:
                backpressured += outcome["backpressured"]
                break
        return {
            "accepted_blocks": accepted,
            "duplicate_blocks": duplicates,
            "backpressured_blocks": backpressured,
            "cursor": self.status(),
        }


def configure_binary_evaluators(
    execution_config: dict[str, Any], adapter_config: dict[str, Any]
) -> dict[str, Any]:
    configured = json.loads(json.dumps(execution_config))
    configured["supervisor"]["cpu_evaluator"] = (
        "sigma_theory_compiler.binary_formula_execution:cpu_binary_block_evaluator"
    )
    configured["supervisor"]["gpu_evaluator"] = (
        "sigma_theory_compiler.binary_formula_execution:gpu_binary_block_evaluator"
    )
    if "cpu" not in adapter_config["lane_cycle"]:
        configured["supervisor"]["cpu_workers"] = 0
    if "gpu" not in adapter_config["lane_cycle"]:
        configured["supervisor"]["gpu_workers"] = 0
    return configured


def run_binary_block_search(
    database: str | Path,
    execution_config: dict[str, Any],
    resource_profile: dict[str, Any],
    adapter_config: dict[str, Any],
    telemetry_path: str | Path,
) -> dict[str, Any]:
    """Consume one bounded refill with persistent lane owners and validate storage."""
    configured = configure_binary_evaluators(execution_config, adapter_config)
    coordinator = PersistentParallelSearch(database, configured, resource_profile)
    refill = BinaryBlockQueueRefill(coordinator, adapter_config)
    refill_report = refill.refill()
    if not refill_report["cursor"]["exhausted"]:
        raise ValueError(
            "target_pending_blocks must cover this bounded source so the GPU cache "
            "persists for the entire run"
        )
    supervisor_report = PersistentParallelSupervisor(
        database, configured, resource_profile, telemetry_path
    ).run()
    with coordinator.connect() as connection:
        rows = connection.execute(
            "SELECT state,payload_json,result_json FROM work ORDER BY ordinal,lane"
        ).fetchall()
    backend_counts: Counter[str] = Counter()
    records = 0
    valid = 0
    for row in rows:
        if row["state"] != "succeeded" or not row["result_json"]:
            continue
        payload = json.loads(row["payload_json"])
        result = json.loads(row["result_json"])
        validate_binary_result(result, payload)
        backend_counts[result["backend"]] += 1
        records += int(result["block"]["record_count"])
        valid += 1
    return {
        "schema_version": "sigma-binary-block-search-run-1.0",
        "cursor": refill.status(),
        "work_counts": coordinator.telemetry()["counts"],
        "valid_result_blocks": valid,
        "processed_records": records,
        "backend_block_counts": dict(backend_counts),
        "all_work_succeeded": bool(rows) and all(row["state"] == "succeeded" for row in rows),
        "supervisor": supervisor_report,
        "data_eligibility_passed": True,
        "paid_llm_calls_made": 0,
    }
