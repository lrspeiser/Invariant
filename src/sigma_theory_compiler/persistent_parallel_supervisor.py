from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .persistent_parallel_search import (
    PersistentParallelSearch,
    WorkLease,
)

Evaluator = Callable[[WorkLease], dict[str, Any]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def synthetic_cpu_evaluator(lease: WorkLease) -> dict[str, Any]:
    """Small deterministic evaluator used by tests and installation smoke checks."""

    sleep_ms = int(lease.payload.get("sleep_ms", 0))
    if sleep_ms:
        time.sleep(sleep_ms / 1000)
    if int(lease.payload.get("hard_crash_on_attempt", -1)) == lease.attempt:
        os._exit(73)
    digest = hashlib.sha256(
        f"{lease.seed}:{_canonical(lease.payload)}".encode()
    ).hexdigest()
    return {
        "evaluator": "synthetic_cpu",
        "ordinal": lease.ordinal,
        "seed": lease.seed,
        "digest": digest,
        "score_u64": int(digest[:16], 16),
    }


def synthetic_gpu_owner_evaluator(lease: WorkLease) -> dict[str, Any]:
    """Deterministic batched callback with the same ownership shape as CUDA screening."""

    sleep_ms = int(lease.payload.get("sleep_ms", 0))
    if sleep_ms:
        time.sleep(sleep_ms / 1000)
    candidate_count = int(lease.payload["candidate_count"])
    digest = hashlib.sha256(
        f"gpu:{lease.seed}:{_canonical(lease.payload)}".encode()
    ).hexdigest()
    return {
        "evaluator": "synthetic_gpu_owner",
        "ordinal": lease.ordinal,
        "seed": lease.seed,
        "candidate_count": candidate_count,
        "survivors": int(digest[:16], 16) % (candidate_count + 1),
        "digest": digest,
    }


BUILTIN_EVALUATORS: dict[str, Evaluator] = {
    "synthetic_cpu": synthetic_cpu_evaluator,
    "synthetic_gpu_owner": synthetic_gpu_owner_evaluator,
}


def resolve_evaluator(reference: str) -> Evaluator:
    if reference in BUILTIN_EVALUATORS:
        return BUILTIN_EVALUATORS[reference]
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("evaluator must be a built-in name or module:function")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError("resolved evaluator is not callable")
    return callback


@dataclass(frozen=True)
class WorkerSpec:
    lane: str
    evaluator: str
    slot: int


def _heartbeat_loop(
    coordinator: PersistentParallelSearch,
    lease: WorkLease,
    worker_id: str,
    done: threading.Event,
    interval_seconds: float,
) -> None:
    while not done.wait(interval_seconds):
        if not coordinator.heartbeat(lease, worker_id):
            return


def _worker_main(
    database: str,
    config: dict[str, Any],
    resource_profile: dict[str, Any],
    spec: WorkerSpec,
    stop_event: Any,
) -> None:
    coordinator = PersistentParallelSearch(database, config, resource_profile)
    evaluator = resolve_evaluator(spec.evaluator)
    worker_id = f"{spec.lane}-{spec.slot}-pid-{os.getpid()}"
    poll = float(config["supervisor"]["worker_poll_seconds"])
    heartbeat_interval = max(
        0.01, float(config["queue"]["lease_seconds"]) / 3
    )
    while not stop_event.is_set():
        lease = coordinator.claim(spec.lane, worker_id)
        if lease is None:
            stop_event.wait(poll)
            continue
        heartbeat_done = threading.Event()
        heartbeat = threading.Thread(
            target=_heartbeat_loop,
            args=(coordinator, lease, worker_id, heartbeat_done, heartbeat_interval),
            daemon=True,
        )
        heartbeat.start()
        try:
            result = evaluator(lease)
            coordinator.finish(lease, worker_id, result)
        except Exception as error:  # noqa: BLE001 - isolate arbitrary evaluator failures
            coordinator.fail(lease, worker_id, f"{type(error).__name__}: {error}")
        finally:
            heartbeat_done.set()
            heartbeat.join(timeout=heartbeat_interval * 2)


class PersistentParallelSupervisor:
    """Spawn-safe supervisor for durable CPU workers and one GPU owner."""

    def __init__(
        self,
        database: str | Path,
        config: dict[str, Any],
        resource_profile: dict[str, Any],
        telemetry_path: str | Path,
    ) -> None:
        if config.get("external_paid_llm_calls") is not False:
            raise ValueError("paid LLM calls must remain disabled")
        self.database = Path(database).resolve()
        self.config = config
        self.resource_profile = resource_profile
        self.telemetry_path = Path(telemetry_path).resolve()
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.coordinator = PersistentParallelSearch(
            self.database, config, resource_profile
        )
        supervisor = config["supervisor"]
        requested_cpu = int(supervisor["cpu_workers"])
        if not 0 <= requested_cpu <= int(self.coordinator.plan["cpu"]["workers"]):
            raise ValueError("requested CPU workers exceed the resource plan")
        requested_gpu = int(supervisor["gpu_workers"])
        if requested_gpu not in {0, 1} or requested_gpu > int(
            self.coordinator.plan["gpu"]["workers"]
        ):
            raise ValueError("GPU worker count must respect single-owner capacity")
        self.specs = [
            WorkerSpec("cpu", str(supervisor["cpu_evaluator"]), slot)
            for slot in range(requested_cpu)
        ] + [
            WorkerSpec("gpu", str(supervisor["gpu_evaluator"]), 0)
            for _ in range(requested_gpu)
        ]
        for spec in self.specs:
            resolve_evaluator(spec.evaluator)

    def _append_telemetry(
        self, sequence: int, snapshot: dict[str, Any], process_state: dict[str, Any]
    ) -> bool:
        record = {
            "schema_version": "sigma-parallel-supervisor-telemetry-1.0",
            "sequence": sequence,
            "created_utc": datetime.now(UTC).isoformat(),
            "execution": snapshot,
            "processes": process_state,
        }
        payload = (_canonical(record) + "\n").encode("utf-8")
        maximum = int(self.config["supervisor"]["maximum_telemetry_bytes"])
        existing = self.telemetry_path.stat().st_size if self.telemetry_path.exists() else 0
        if existing + len(payload) > maximum:
            return False
        with self.telemetry_path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
        return True

    def run(self, maximum_wall_seconds: float | None = None) -> dict[str, Any]:
        context = mp.get_context("spawn")
        stop_event = context.Event()
        supervisor_config = self.config["supervisor"]
        maximum_wall = float(
            maximum_wall_seconds
            if maximum_wall_seconds is not None
            else supervisor_config["maximum_wall_seconds_per_run"]
        )
        telemetry_interval = float(supervisor_config["telemetry_interval_seconds"])
        maximum_restarts = int(supervisor_config["maximum_process_restarts"])
        checkpoint_every = int(self.config["queue"]["checkpoint_every_completions"])
        start = time.monotonic()
        self.coordinator.recover_expired()
        processes: dict[int, tuple[Any, WorkerSpec]] = {}
        starts = restarts = crashes = telemetry_records = telemetry_records_dropped = 0
        checkpoint_completed = self.coordinator.telemetry()["counts"].get(
            "succeeded", 0
        )
        utilization_samples: dict[str, list[float]] = {"cpu": [], "gpu": []}

        def spawn(spec: WorkerSpec) -> None:
            nonlocal starts
            process = context.Process(
                target=_worker_main,
                args=(
                    str(self.database),
                    self.config,
                    self.resource_profile,
                    spec,
                    stop_event,
                ),
                name=f"sigma-{spec.lane}-{spec.slot}",
            )
            process.start()
            processes[process.pid] = (process, spec)
            starts += 1

        for spec in self.specs:
            spawn(spec)
        sequence = sum(1 for _ in self.telemetry_path.open("r", encoding="utf-8")) if self.telemetry_path.exists() else 0
        stop_reason = "queue_drained"
        try:
            while True:
                self.coordinator.recover_expired()
                snapshot = self.coordinator.telemetry()
                for lane, samples in utilization_samples.items():
                    samples.append(
                        float(snapshot["lanes"][lane]["utilization"])
                    )
                process_state = {
                    "alive": sum(process.is_alive() for process, _ in processes.values()),
                    "starts": starts,
                    "restarts": restarts,
                    "crashes": crashes,
                }
                sequence += 1
                if self._append_telemetry(sequence, snapshot, process_state):
                    telemetry_records += 1
                else:
                    telemetry_records_dropped += 1

                for pid, (process, spec) in list(processes.items()):
                    if process.is_alive() or process.exitcode is None:
                        continue
                    process.join(timeout=0)
                    del processes[pid]
                    if process.exitcode != 0:
                        crashes += 1
                    pending = snapshot["queue"]["pending"]
                    if pending and restarts < maximum_restarts:
                        restarts += 1
                        spawn(spec)

                completed = int(snapshot["counts"].get("succeeded", 0))
                if checkpoint_every > 0 and completed - checkpoint_completed >= checkpoint_every:
                    self.coordinator.checkpoint()
                    checkpoint_completed = completed
                if snapshot["queue"]["pending"] == 0:
                    break
                if time.monotonic() - start >= maximum_wall:
                    stop_reason = "run_wall_time_reached"
                    break
                time.sleep(telemetry_interval)
        finally:
            stop_event.set()
            grace = float(supervisor_config["shutdown_grace_seconds"])
            for process, _ in processes.values():
                process.join(timeout=grace)
            for process, _ in processes.values():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=grace)
        checkpoint = self.coordinator.checkpoint()
        final = self.coordinator.telemetry()
        utilization = {
            lane: {
                "samples": len(values),
                "mean": sum(values) / len(values) if values else 0.0,
                "peak": max(values, default=0.0),
            }
            for lane, values in utilization_samples.items()
        }
        return {
            "schema_version": "sigma-parallel-supervisor-run-1.0",
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - start,
            "process_starts": starts,
            "process_restarts": restarts,
            "process_crashes": crashes,
            "telemetry_records_written": telemetry_records,
            "telemetry_records_dropped_by_disk_cap": telemetry_records_dropped,
            "telemetry_path": str(self.telemetry_path),
            "utilization": utilization,
            "checkpoint": checkpoint,
            "final_telemetry": final,
            "paid_llm_calls_made": 0,
        }
