from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_screen import KERNEL, dense_grid, precompute_dense_hessians
from .high_throughput import (
    build_basis,
    decode_ordinal,
    total_search_count,
)
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .persistent_parallel_supervisor import PersistentParallelSupervisor

RESULT_SCHEMA = "sigma-real-formula-batch-result-1.0"
CURSOR_SCHEMA = "sigma-finite-formula-refill-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def cuda_available() -> tuple[bool, str]:
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        return count > 0, f"cupy={cp.__version__};devices={count}"
    except Exception as error:  # noqa: BLE001 - capability probe must fail closed
        return False, f"{type(error).__name__}: {error}"


@cache
def _assets(config_path: str, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    path = Path(config_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("generator config hash mismatch")
    config = json.loads(raw)
    if config.get("observational_data_opened") is not False:
        raise ValueError("generator must keep observational data closed")
    basis = build_basis(int(config["basis_count"]))
    hessians = precompute_dense_hessians(basis, dense_grid())
    return config, basis, hessians


def _batch_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = int(payload["start_ordinal"])
    end = int(payload["end_ordinal_exclusive"])
    if end <= start or end - start != int(payload["candidate_count"]):
        raise ValueError("invalid formula batch interval")
    basis_count = int(payload["basis_count"])
    max_terms = int(payload["max_action_terms"])
    ordinals = np.arange(start, end, dtype=np.uint64)
    term_ids = np.zeros((len(ordinals), 6), dtype=np.uint16)
    term_counts = np.zeros(len(ordinals), dtype=np.uint8)
    sign_masks = np.zeros(len(ordinals), dtype=np.uint8)
    for row, ordinal in enumerate(map(int, ordinals)):
        decoded = decode_ordinal(basis_count, max_terms, ordinal)
        count = len(decoded["term_ids"])
        term_ids[row, :count] = decoded["term_ids"]
        term_counts[row] = count
        mask = 0
        for position, sign in enumerate(decoded["signs"]):
            if sign > 0:
                mask |= 1 << position
        sign_masks[row] = mask
    return ordinals, term_ids, term_counts, sign_masks


def _validate_payload(payload: dict[str, Any], generator: dict[str, Any]) -> None:
    expected_eligibility = {
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
    }
    if payload.get("data_eligibility") != expected_eligibility:
        raise ValueError("formula batch data eligibility is not fail-closed")
    if (
        payload.get("protocol_version") != generator.get("protocol_version")
        or int(payload["basis_count"]) != int(generator["basis_count"])
        or int(payload["max_action_terms"]) != int(generator["max_action_terms"])
    ):
        raise ValueError("formula batch generator identity mismatch")


def _result(
    payload: dict[str, Any], backend: str, statuses: np.ndarray, margins: np.ndarray, elapsed: float
) -> dict[str, Any]:
    counts = Counter(map(int, statuses))
    candidate_count = int(payload["candidate_count"])
    result = {
        "schema_version": RESULT_SCHEMA,
        "backend": backend,
        "protocol_version": payload["protocol_version"],
        "source_config_sha256": payload["generator_config_sha256"],
        "batch": {
            "start_ordinal": int(payload["start_ordinal"]),
            "end_ordinal_exclusive": int(payload["end_ordinal_exclusive"]),
            "candidate_count": candidate_count,
        },
        "counts": {
            "reject": counts[0],
            "pass": counts[1],
            "ambiguous": counts[2],
        },
        "status_root_sha256": hashlib.sha256(statuses.tobytes()).hexdigest(),
        "reported_margin_minimum": float(np.min(margins)) if len(margins) else None,
        "margin_semantics": (
            "minimum over samples for passes; the CUDA kernel reports the first failing "
            "sample for rejects, so rejected margins are witnesses rather than global minima"
        ),
        "elapsed_seconds": elapsed,
        "candidates_per_second": candidate_count / elapsed if elapsed > 0 else None,
        "data_eligibility": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_calls": False,
            "passed": True,
        },
        "interpretation": (
            "A pass survives only the declared 343-point sampled-static convexity screen; "
            "it is not a covariant-health or observational claim."
        ),
    }
    validate_formula_batch_result(result, payload)
    return result


def validate_formula_batch_result(result: dict[str, Any], payload: dict[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unsupported real-formula result schema")
    if result.get("backend") not in {"cpu_numpy", "gpu_cupy_raw_kernel"}:
        raise ValueError("unsupported formula evaluator backend")
    batch = result.get("batch", {})
    if (
        batch.get("start_ordinal") != int(payload["start_ordinal"])
        or batch.get("end_ordinal_exclusive") != int(payload["end_ordinal_exclusive"])
        or batch.get("candidate_count") != int(payload["candidate_count"])
    ):
        raise ValueError("formula result batch identity mismatch")
    counts = result.get("counts", {})
    if set(counts) != {"reject", "pass", "ambiguous"} or any(
        not isinstance(value, int) or value < 0 for value in counts.values()
    ):
        raise ValueError("formula result counts are invalid")
    if sum(counts.values()) != int(payload["candidate_count"]):
        raise ValueError("formula result accounting mismatch")
    if result.get("source_config_sha256") != payload["generator_config_sha256"]:
        raise ValueError("formula result source hash mismatch")
    eligibility = result.get("data_eligibility", {})
    if eligibility != {
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
        "passed": True,
    }:
        raise ValueError("formula result violates the data-eligibility contract")
    root = result.get("status_root_sha256")
    if not isinstance(root, str) or len(root) != 64:
        raise ValueError("formula result lacks a status root")


def cpu_formula_batch_evaluator(lease: WorkLease) -> dict[str, Any]:
    started = time.perf_counter()
    payload = lease.payload
    config, _, hessians = _assets(
        str(payload["generator_config_path"]), str(payload["generator_config_sha256"])
    )
    _validate_payload(payload, config)
    _, term_ids, term_counts, sign_masks = _batch_arrays(payload)
    candidate_count = len(term_counts)
    statuses = np.empty(candidate_count, dtype=np.uint8)
    margins = np.empty(candidate_count, dtype=np.float64)
    tolerance = float(config["convexity_tolerance"])
    guard = float(payload["ambiguity_guard"])
    low, high = tolerance - guard, tolerance + guard
    coupling = float(config["coupling_magnitude"])
    for row in range(candidate_count):
        count = int(term_counts[row])
        signs = np.array(
            [1.0 if int(sign_masks[row]) & (1 << position) else -1.0 for position in range(count)]
        )
        candidate = np.array([1.0, 0.0, 0.0]) + coupling * np.sum(
            hessians[term_ids[row, :count]] * signs[:, None, None], axis=0
        )
        hdd, hdp, hpp = candidate.T
        minimum = 0.5 * (
            hdd + hpp - np.sqrt(np.maximum((hdd - hpp) ** 2 + 4 * hdp**2, 0.0))
        )
        margin = float(np.min(minimum))
        margins[row] = margin
        statuses[row] = 0 if not np.isfinite(margin) or margin <= low else (2 if margin <= high else 1)
    return _result(payload, "cpu_numpy", statuses, margins, time.perf_counter() - started)


def gpu_formula_batch_evaluator(lease: WorkLease) -> dict[str, Any]:
    started = time.perf_counter()
    available, reason = cuda_available()
    if not available:
        raise RuntimeError(f"CUDA evaluator unavailable: {reason}")
    import cupy as cp

    payload = lease.payload
    config, _, hessians = _assets(
        str(payload["generator_config_path"]), str(payload["generator_config_sha256"])
    )
    _validate_payload(payload, config)
    _, term_ids, term_counts, sign_masks = _batch_arrays(payload)
    candidate_count = len(term_counts)
    device_hessians = cp.asarray(hessians)
    device_terms = cp.asarray(np.ascontiguousarray(term_ids))
    device_counts = cp.asarray(term_counts)
    device_masks = cp.asarray(sign_masks)
    statuses = cp.empty(candidate_count, dtype=cp.uint8)
    fail_samples = cp.empty(candidate_count, dtype=cp.uint16)
    margins = cp.empty(candidate_count, dtype=cp.float64)
    kernel = cp.RawKernel(KERNEL, "dense_screen")
    tolerance = float(config["convexity_tolerance"])
    guard = float(payload["ambiguity_guard"])
    threads = 256
    kernel(
        ((candidate_count + threads - 1) // threads,),
        (threads,),
        (
            device_terms,
            device_counts,
            device_masks,
            device_hessians,
            np.int32(hessians.shape[1]),
            np.float64(config["coupling_magnitude"]),
            np.float64(tolerance - guard),
            np.float64(tolerance + guard),
            statuses,
            fail_samples,
            margins,
            np.int32(candidate_count),
        ),
    )
    host_statuses = cp.asnumpy(statuses)
    host_margins = cp.asnumpy(margins)
    cp.cuda.Device().synchronize()
    return _result(
        payload,
        "gpu_cupy_raw_kernel",
        host_statuses,
        host_margins,
        time.perf_counter() - started,
    )


class FiniteFormulaQueueRefill:
    """Single-owner persistent cursor over the frozen finite ordinal generator."""

    def __init__(
        self,
        coordinator: PersistentParallelSearch,
        adapter_config: dict[str, Any],
    ) -> None:
        if adapter_config.get("schema_version") != CURSOR_SCHEMA:
            raise ValueError("unsupported finite-refill schema")
        if adapter_config.get("external_paid_llm_calls") is not False:
            raise ValueError("paid LLM calls must remain disabled")
        eligibility = adapter_config.get("data_eligibility", {})
        if eligibility != {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
        }:
            raise ValueError("finite generator data eligibility is not fail-closed")
        self.coordinator = coordinator
        self.config = adapter_config
        self.generator_path = Path(adapter_config["generator_config_path"]).resolve()
        raw = self.generator_path.read_bytes()
        self.generator_sha = hashlib.sha256(raw).hexdigest()
        self.generator = json.loads(raw)
        if self.generator.get("observational_data_opened") is not False:
            raise ValueError("generator config opens observational data")
        total = total_search_count(
            int(self.generator["basis_count"]), int(self.generator["max_action_terms"])
        )
        start = int(adapter_config["start_ordinal"])
        stop = min(int(adapter_config["stop_ordinal_exclusive"]), total)
        if not 0 <= start < stop <= total:
            raise ValueError("invalid finite generator interval")
        self.source_id = f"FORMULA-{_sha({'path': str(self.generator_path), 'sha': self.generator_sha, 'start': start, 'stop': stop})[:24]}"
        with coordinator.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS formula_generator_cursor ("
                "source_id TEXT PRIMARY KEY,source_sha256 TEXT NOT NULL,next_ordinal INTEGER NOT NULL,"
                "stop_ordinal INTEGER NOT NULL,batch_sequence INTEGER NOT NULL,lane_index INTEGER NOT NULL)"
            )
            row = connection.execute(
                "SELECT * FROM formula_generator_cursor WHERE source_id=?", (self.source_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO formula_generator_cursor VALUES (?,?,?,?,0,0)",
                    (self.source_id, self.generator_sha, start, stop),
                )
            elif row["source_sha256"] != self.generator_sha or row["stop_ordinal"] != stop:
                raise ValueError("refusing to resume a changed finite generator")

    def status(self) -> dict[str, Any]:
        with self.coordinator.connect() as connection:
            row = connection.execute(
                "SELECT * FROM formula_generator_cursor WHERE source_id=?", (self.source_id,)
            ).fetchone()
        return {
            "source_id": self.source_id,
            "next_ordinal": int(row["next_ordinal"]),
            "stop_ordinal_exclusive": int(row["stop_ordinal"]),
            "batch_sequence": int(row["batch_sequence"]),
            "exhausted": int(row["next_ordinal"]) >= int(row["stop_ordinal"]),
        }

    def refill(self) -> dict[str, Any]:
        target = int(self.config["target_pending_batches"])
        lane_cycle = list(self.config["lane_cycle"])
        if not lane_cycle or any(lane not in {"cpu", "gpu"} for lane in lane_cycle):
            raise ValueError("lane_cycle must contain cpu/gpu lanes")
        accepted = duplicates = backpressured = 0
        while int(self.coordinator.telemetry()["queue"]["pending"]) < target:
            with self.coordinator.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM formula_generator_cursor WHERE source_id=?", (self.source_id,)
                ).fetchone()
            start = int(row["next_ordinal"])
            stop = int(row["stop_ordinal"])
            if start >= stop:
                break
            lane_index = int(row["lane_index"])
            lane = lane_cycle[lane_index % len(lane_cycle)]
            batch_size = int(self.config[f"{lane}_batch_candidates"])
            end = min(stop, start + batch_size)
            payload = {
                "ordinal": start,
                "source_id": self.source_id,
                "batch_sequence": int(row["batch_sequence"]),
                "start_ordinal": start,
                "end_ordinal_exclusive": end,
                "candidate_count": end - start,
                "generator_config_path": str(self.generator_path),
                "generator_config_sha256": self.generator_sha,
                "protocol_version": self.generator["protocol_version"],
                "basis_count": int(self.generator["basis_count"]),
                "max_action_terms": int(self.generator["max_action_terms"]),
                "ambiguity_guard": float(self.config["ambiguity_guard"]),
                "data_eligibility": self.config["data_eligibility"],
            }
            outcome = self.coordinator.enqueue([payload], lane=lane)
            if outcome["accepted"] or outcome["duplicate"]:
                accepted += outcome["accepted"]
                duplicates += outcome["duplicate"]
                with self.coordinator.connect() as connection:
                    connection.execute(
                        "UPDATE formula_generator_cursor SET next_ordinal=?,batch_sequence=batch_sequence+1,"
                        "lane_index=lane_index+1 WHERE source_id=? AND next_ordinal=?",
                        (end, self.source_id, start),
                    )
            else:
                backpressured += outcome["backpressured"]
                break
        return {
            "accepted_batches": accepted,
            "duplicate_batches": duplicates,
            "backpressured_batches": backpressured,
            "cursor": self.status(),
        }


def configure_real_evaluators(
    execution_config: dict[str, Any], adapter_config: dict[str, Any]
) -> dict[str, Any]:
    configured = json.loads(json.dumps(execution_config))
    configured["supervisor"]["cpu_evaluator"] = (
        "sigma_theory_compiler.real_formula_execution:cpu_formula_batch_evaluator"
    )
    configured["supervisor"]["gpu_evaluator"] = (
        "sigma_theory_compiler.real_formula_execution:gpu_formula_batch_evaluator"
    )
    if "gpu" not in adapter_config["lane_cycle"]:
        configured["supervisor"]["gpu_workers"] = 0
    return configured


def run_finite_formula_search(
    database: str | Path,
    execution_config: dict[str, Any],
    resource_profile: dict[str, Any],
    adapter_config: dict[str, Any],
    telemetry_path: str | Path,
    *,
    maximum_waves: int = 100,
) -> dict[str, Any]:
    configured = configure_real_evaluators(execution_config, adapter_config)
    coordinator = PersistentParallelSearch(database, configured, resource_profile)
    refill = FiniteFormulaQueueRefill(coordinator, adapter_config)
    waves: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(maximum_waves):
        refill_report = refill.refill()
        if coordinator.telemetry()["queue"]["pending"] == 0:
            break
        run_report = PersistentParallelSupervisor(
            database, configured, resource_profile, telemetry_path
        ).run()
        waves.append({"refill": refill_report, "run": run_report})
        if run_report["stop_reason"] != "queue_drained":
            break
    cursor = refill.status()
    with coordinator.connect() as connection:
        rows = connection.execute(
            "SELECT result_json,payload_json,state FROM work ORDER BY ordinal,lane"
        ).fetchall()
    valid_results = 0
    backend_counts: Counter[str] = Counter()
    for row in rows:
        if row["state"] != "succeeded" or not row["result_json"]:
            continue
        result = json.loads(row["result_json"])
        validate_formula_batch_result(result, json.loads(row["payload_json"]))
        backend_counts[result["backend"]] += 1
        valid_results += 1
    elapsed = time.perf_counter() - started
    processed_candidates = sum(
        json.loads(row["result_json"])["batch"]["candidate_count"]
        for row in rows
        if row["state"] == "succeeded" and row["result_json"]
    )
    return {
        "schema_version": "sigma-finite-real-formula-search-run-1.0",
        "cursor": cursor,
        "waves": len(waves),
        "work_counts": coordinator.telemetry()["counts"],
        "valid_result_batches": valid_results,
        "backend_batch_counts": dict(backend_counts),
        "processed_candidates": processed_candidates,
        "elapsed_seconds": elapsed,
        "candidates_per_second": processed_candidates / elapsed if elapsed else None,
        "generator_exhausted": cursor["exhausted"],
        "all_work_succeeded": bool(rows) and all(row["state"] == "succeeded" for row in rows),
        "data_eligibility_passed": True,
        "paid_llm_calls_made": 0,
    }
