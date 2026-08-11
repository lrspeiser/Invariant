"""Foreground-only deferred handoff to the reviewed set-indexed GPU scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import kastner_schlatter_deferred_gpu_ownership as deferred
from .kastner_schlatter_set_indexed_gpu_scheduler_adapter import run_scheduler

CONFIG_SCHEMA = "sigma-kastner-schlatter-deferred-gpu-handoff-config-1.0"
READINESS_SCHEMA = "sigma-kastner-schlatter-deferred-gpu-handoff-readiness-1.0"
CHECKPOINT_SCHEMA = "sigma-kastner-schlatter-deferred-gpu-handoff-checkpoint-1.0"
LEASE_SCHEMA = "sigma-kastner-schlatter-deferred-gpu-handoff-lease-1.0"
EXPECTED_RUNTIME = (
    "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-runtime/deferred-handoff-service"
)
EXPECTED_EVALUATOR = (
    "sigma_theory_compiler.kastner_schlatter_set_indexed_gpu_scheduler_adapter:"
    "gpu_reviewed_workload_evaluator"
)
EXPECTED_SEALS = {
    "waiting_cuda_context": False,
    "handoff_service_direct_sqlite_access": False,
    "live_campaign_sqlite_access": False,
    "existing_process_signaled": False,
    "detached_launch": False,
    "arbitrary_callable_or_subprocess_injection": False,
    "observations_opened": False,
    "scientific_or_readiness_promotion": False,
    "paid_llm_calls": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("deferred handoff path escapes repository") from error
    return target


def _bounded_json(path: Path, maximum_bytes: int) -> dict[str, Any]:
    if path.is_symlink() or path.stat().st_size > maximum_bytes:
        raise RuntimeError("deferred handoff state symlink or size bound violated")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("deferred handoff state is not an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], maximum_bytes: int) -> None:
    payload = (_canonical(value) + "\n").encode()
    if len(payload) > maximum_bytes:
        raise RuntimeError("deferred handoff state exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("deferred handoff symlink rejected")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "service_id",
        "service_epoch",
        "runtime_directory",
        "service_lease_name",
        "service_recovery_name",
        "checkpoint_name",
        "stop_name",
        "output_path",
        "maximum_service_cycles",
        "maximum_service_seconds",
        "maximum_wait_polls_per_cycle",
        "poll_interval_seconds",
        "required_consecutive_safe_samples",
        "maximum_gpu_utilization_percent",
        "minimum_free_gpu_memory_mib",
        "scheduler_slice_seconds",
        "maximum_state_bytes",
        "bindings",
        "seals",
    }
    if set(config) != expected or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported deferred handoff config")
    if (
        config.get("runtime_directory") != EXPECTED_RUNTIME
        or config.get("service_lease_name") != "handoff-service.lease.json"
        or config.get("service_recovery_name") != "handoff-service.lease.recovery.json"
        or config.get("checkpoint_name") != "handoff-service-checkpoint.json"
        or config.get("stop_name") != "handoff-service.stop.request"
        or config.get("output_path")
        != "runs/engine/kastner-schlatter-deferred-gpu-handoff-readiness.json"
        or config.get("seals") != EXPECTED_SEALS
    ):
        raise ValueError("deferred handoff closed contract changed")
    if (
        int(config["maximum_service_cycles"]) > 24
        or int(config["maximum_service_seconds"]) > 86400
        or int(config["maximum_wait_polls_per_cycle"]) > 721
        or float(config["poll_interval_seconds"]) > 5
        or int(config["required_consecutive_safe_samples"]) < 3
        or int(config["maximum_gpu_utilization_percent"]) > 20
        or int(config["minimum_free_gpu_memory_mib"]) < 8192
        or float(config["scheduler_slice_seconds"]) > 120
        or int(config["maximum_state_bytes"]) > 131072
        or min(
            float(config[key])
            for key in (
                "maximum_service_cycles",
                "maximum_service_seconds",
                "maximum_wait_polls_per_cycle",
                "poll_interval_seconds",
                "required_consecutive_safe_samples",
                "minimum_free_gpu_memory_mib",
                "scheduler_slice_seconds",
                "maximum_state_bytes",
            )
        )
        <= 0
    ):
        raise ValueError("deferred handoff safety bound widened")
    expected_bindings = {
        "deferred_config",
        "deferred_source",
        "deferred_readiness",
        "scheduler_config",
        "scheduler_source",
        "scheduler_readiness",
        "gitignore",
    }
    if set(config.get("bindings", {})) != expected_bindings:
        raise ValueError("deferred handoff binding set changed")
    for name, binding in config["bindings"].items():
        bound = _inside(root, binding["path"])
        if _file_sha(bound) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
        if "content_sha256" in binding:
            value = json.loads(bound.read_text(encoding="utf-8"))
            if (
                value.get("content_sha256") != binding["content_sha256"]
                or _content_sha(value) != binding["content_sha256"]
            ):
                raise ValueError(f"{name} content hash mismatch")
    deferred_config, _ = deferred.load_config(
        _inside(root, config["bindings"]["deferred_config"]["path"])
    )
    if any(
        int(config[key]) != int(deferred_config[key])
        for key in (
            "poll_interval_seconds",
            "required_consecutive_safe_samples",
            "maximum_gpu_utilization_percent",
            "minimum_free_gpu_memory_mib",
        )
    ):
        raise ValueError("deferred handoff threshold drift")
    scheduler = json.loads(
        _inside(root, config["bindings"]["scheduler_config"]["path"]).read_text(encoding="utf-8")
    )
    supervisor = scheduler.get("persistent_config", {}).get("supervisor", {})
    if (
        scheduler.get("reviewed_workload_id")
        != "kastner-schlatter-set-indexed-cuda-falsification-001"
        or supervisor.get("gpu_workers") != 1
        or supervisor.get("cpu_workers") != 0
        or supervisor.get("gpu_evaluator") != EXPECTED_EVALUATOR
    ):
        raise ValueError("reviewed one-GPU scheduler boundary changed")
    ignored = _inside(root, config["bindings"]["gitignore"]["path"]).read_text(encoding="utf-8")
    if (
        "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-runtime/"
        not in ignored.splitlines()
    ):
        raise ValueError("deferred handoff runtime is not ignored")
    return config, root


def _process_identity(argv: list[str]) -> str:
    return _sha(argv) if argv else ""


def _current_identity() -> str:
    try:
        import psutil

        return _process_identity(psutil.Process(os.getpid()).cmdline())
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("deferred handoff process identity unavailable") from error


def _owner_state(pid: int, identity: str) -> str:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("deferred handoff process inventory unavailable") from error
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return "dead"
        return "match" if _process_identity(process.cmdline()) == identity else "mismatch"
    except psutil.NoSuchProcess:
        return "dead"
    except (psutil.AccessDenied, psutil.ZombieProcess, OSError) as error:
        raise RuntimeError("deferred handoff owner is unverifiable") from error


def _lease(config: Mapping[str, Any], attempted_cycles: int) -> dict[str, Any]:
    body = {
        "schema_version": LEASE_SCHEMA,
        "service_id": config["service_id"],
        "service_epoch": config["service_epoch"],
        "pid": os.getpid(),
        "process_argv_sha256": _current_identity(),
        "attempted_cycles": attempted_cycles,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    return {**body, "content_sha256": _content_sha(body)}


def _acquire_service_lease(
    runtime: Path, config: Mapping[str, Any], attempted_cycles: int
) -> tuple[Path, dict[str, Any]]:
    runtime.mkdir(parents=True, exist_ok=True)
    if runtime.is_symlink() or any(runtime.glob("*.sqlite*")):
        raise RuntimeError("deferred handoff runtime is unsafe or contains SQLite")
    maximum = int(config["maximum_state_bytes"])
    path = runtime / config["service_lease_name"]
    value = _lease(config, attempted_cycles)
    payload = (_canonical(value) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _bounded_json(path, maximum)
        if existing.get("content_sha256") != _content_sha(existing):
            raise RuntimeError("deferred handoff service lease tamper detected") from None
        if (
            _owner_state(
                int(existing.get("pid", -1)),
                str(existing.get("process_argv_sha256", "")),
            )
            == "match"
        ):
            raise RuntimeError("deferred handoff service already active") from None
        _atomic_json(runtime / config["service_recovery_name"], existing, maximum)
        path.unlink()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path, value


def _release_service_lease(path: Path, owned: Mapping[str, Any], maximum: int) -> None:
    if not path.exists():
        return
    current = _bounded_json(path, maximum)
    if current.get("pid") == owned.get("pid") and current.get("process_argv_sha256") == owned.get(
        "process_argv_sha256"
    ):
        path.unlink()


def _checkpoint(
    config_path: Path,
    config: Mapping[str, Any],
    runtime: Path,
    *,
    state: str,
    started_utc: str,
    attempted_cycles: int,
    executed_cycles: int,
    polls: int,
    consecutive_safe: int,
    sample: Mapping[str, Any] | None,
    scheduler_receipt: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "service_id": config["service_id"],
        "service_epoch": config["service_epoch"],
        "state": state,
        "pid": os.getpid() if state in {"starting", "waiting", "reserved", "running"} else None,
        "process_argv_sha256": _current_identity(),
        "config_file_sha256": _file_sha(config_path),
        "started_utc": started_utc,
        "attempted_cycles": attempted_cycles,
        "executed_cycles": executed_cycles,
        "polls_in_cycle": polls,
        "consecutive_safe_samples": consecutive_safe,
        "last_nvml_sample": dict(sample) if sample is not None else None,
        "last_scheduler_receipt": dict(scheduler_receipt) if scheduler_receipt else None,
        "error": error,
        "waiting_cuda_context_created": False,
        "handoff_service_direct_sqlite_accessed": False,
        "live_campaign_sqlite_accessed": False,
        "existing_process_signaled": False,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    value = {**body, "content_sha256": _content_sha(body)}
    _atomic_json(runtime / config["checkpoint_name"], value, int(config["maximum_state_bytes"]))
    return value


def _load_checkpoint(path: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = _bounded_json(path, int(config["maximum_state_bytes"]))
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA
        or value.get("content_sha256") != _content_sha(value)
        or value.get("service_id") != config["service_id"]
        or value.get("service_epoch") != config["service_epoch"]
        or value.get("config_file_sha256") != _file_sha(config_path)
    ):
        raise ValueError("deferred handoff checkpoint validation failed")
    for key in ("attempted_cycles", "executed_cycles", "polls_in_cycle"):
        if not isinstance(value.get(key), int) or int(value[key]) < 0:
            raise ValueError("deferred handoff checkpoint counter invalid")
    return value


def _safe(sample: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    return int(sample["gpu_utilization_percent"]) <= int(
        config["maximum_gpu_utilization_percent"]
    ) and int(sample["memory_free_mib"]) >= int(config["minimum_free_gpu_memory_mib"])


def _reserve_gpu_owner(
    config: Mapping[str, Any],
    root: Path,
    polls: int,
    sample: Mapping[str, Any],
    runtime_override: Path | None,
) -> deferred.DeferredOwnershipToken:
    deferred_path = _inside(root, config["bindings"]["deferred_config"]["path"])
    deferred_config, _ = deferred.load_config(deferred_path)
    owner_runtime = (
        runtime_override
        if runtime_override is not None
        else _inside(root, deferred_config["runtime_directory"])
    )
    lease_path, _ = deferred._acquire_waiter(owner_runtime, deferred_config)
    owned = deferred._lease_value(deferred_config, "gpu_owner_reserved", polls)
    deferred._atomic_json(lease_path, owned, int(deferred_config["maximum_checkpoint_bytes"]))
    checkpoint = deferred._write_checkpoint(
        owner_runtime,
        deferred_config,
        state="reserved",
        polls=polls,
        consecutive_safe=int(config["required_consecutive_safe_samples"]),
        sample=sample,
        error=None,
    )
    return deferred.DeferredOwnershipToken(
        owner_runtime, deferred_config, lease_path, owned, checkpoint
    )


def _scheduler_receipt(
    result: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    if (
        result.get("immutable_artifact_written") is not False
        or result.get("live_campaign_sqlite_accessed") is not False
    ):
        raise ValueError("reviewed scheduler boundary claim changed")
    scheduler_config = json.loads(
        _inside(root, config["bindings"]["scheduler_config"]["path"]).read_text(encoding="utf-8")
    )
    runtime = _inside(root, scheduler_config["runtime_directory"])
    database = Path(str(result.get("database", ""))).resolve()
    telemetry = Path(str(result.get("telemetry", ""))).resolve()
    if (
        database != runtime / scheduler_config["database_name"]
        or telemetry != runtime / scheduler_config["telemetry_name"]
    ):
        raise ValueError("reviewed scheduler runtime path changed")
    supervisor = result.get("supervisor")
    if not isinstance(supervisor, Mapping):
        raise TypeError("reviewed scheduler report missing")
    final = supervisor.get("final_telemetry", {})
    counts = final.get("counts", {}) if isinstance(final, Mapping) else {}
    queue = final.get("queue", {}) if isinstance(final, Mapping) else {}
    body = {
        "scheduler_result_sha256": _sha(result),
        "enqueue": result.get("enqueue"),
        "supervisor_stop_reason": supervisor.get("stop_reason"),
        "final_counts": dict(counts) if isinstance(counts, Mapping) else {},
        "final_queue": dict(queue) if isinstance(queue, Mapping) else {},
        "reviewed_workload_succeeded": int(counts.get("succeeded", 0)) >= 1,
        "queue_pending": int(queue.get("pending", 0)),
        "scheduler_owned_isolated_sqlite_queue": database.relative_to(root).as_posix(),
        "handoff_service_direct_sqlite_accessed": False,
        "live_campaign_sqlite_accessed": False,
        "immutable_artifact_written": False,
    }
    return {**body, "content_sha256": _content_sha(body)}


def run_service(
    config_path: str | Path,
    *,
    runtime_override: str | Path | None = None,
    gpu_owner_runtime_override: str | Path | None = None,
    maximum_cycles_override: int | None = None,
    maximum_wait_polls_override: int | None = None,
) -> dict[str, Any]:
    """Run foreground; tests may only narrow bounds and redirect service JSON to temp."""
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    raw_override = Path(runtime_override) if runtime_override is not None else None
    if raw_override is not None and raw_override.is_symlink():
        raise ValueError("deferred handoff runtime override symlink rejected")
    runtime = raw_override.resolve() if raw_override else _inside(root, config["runtime_directory"])
    if raw_override is not None:
        try:
            runtime.relative_to(Path(tempfile.gettempdir()).resolve())
        except ValueError as error:
            raise ValueError("runtime override restricted to temporary tests") from error
    owner_override = (
        Path(gpu_owner_runtime_override).resolve()
        if gpu_owner_runtime_override is not None
        else None
    )
    if owner_override is not None:
        try:
            owner_override.relative_to(Path(tempfile.gettempdir()).resolve())
        except ValueError as error:
            raise ValueError("GPU owner runtime override restricted to temporary tests") from error
    maximum_cycles = int(
        maximum_cycles_override
        if maximum_cycles_override is not None
        else config["maximum_service_cycles"]
    )
    maximum_polls = int(
        maximum_wait_polls_override
        if maximum_wait_polls_override is not None
        else config["maximum_wait_polls_per_cycle"]
    )
    if (
        maximum_cycles < 1
        or maximum_cycles > int(config["maximum_service_cycles"])
        or maximum_polls < int(config["required_consecutive_safe_samples"])
        or maximum_polls > int(config["maximum_wait_polls_per_cycle"])
    ):
        raise ValueError("test bound override widens or defeats handoff safety")
    checkpoint_path = runtime / config["checkpoint_name"]
    previous = (
        _load_checkpoint(checkpoint_path, config_path, config) if checkpoint_path.exists() else None
    )
    attempted = int(previous["attempted_cycles"]) if previous else 0
    executed = int(previous["executed_cycles"]) if previous else 0
    started_utc = str(previous["started_utc"]) if previous else datetime.now(UTC).isoformat()
    lease_path, owned = _acquire_service_lease(runtime, config, attempted)
    sample: dict[str, Any] | None = None
    receipt = previous.get("last_scheduler_receipt") if previous else None
    checkpoint: dict[str, Any] | None = None
    deadline = datetime.fromisoformat(started_utc).timestamp() + float(
        config["maximum_service_seconds"]
    )
    try:
        checkpoint = _checkpoint(
            config_path,
            config,
            runtime,
            state="starting",
            started_utc=started_utc,
            attempted_cycles=attempted,
            executed_cycles=executed,
            polls=0,
            consecutive_safe=0,
            sample=None,
            scheduler_receipt=receipt,
            error=None,
        )
        if (
            receipt
            and receipt.get("reviewed_workload_succeeded")
            and receipt.get("queue_pending") == 0
        ):
            return _checkpoint(
                config_path,
                config,
                runtime,
                state="completed",
                started_utc=started_utc,
                attempted_cycles=attempted,
                executed_cycles=executed,
                polls=0,
                consecutive_safe=0,
                sample=None,
                scheduler_receipt=receipt,
                error=None,
            )
        while attempted < maximum_cycles and time.time() < deadline:
            if (runtime / config["stop_name"]).exists():
                break
            consecutive = 0
            polls = 0
            token: deferred.DeferredOwnershipToken | None = None
            while polls < maximum_polls and time.time() < deadline:
                if (runtime / config["stop_name"]).exists():
                    break
                sample = deferred.sample_nvml()
                polls += 1
                consecutive = consecutive + 1 if _safe(sample, config) else 0
                checkpoint = _checkpoint(
                    config_path,
                    config,
                    runtime,
                    state="waiting",
                    started_utc=started_utc,
                    attempted_cycles=attempted,
                    executed_cycles=executed,
                    polls=polls,
                    consecutive_safe=consecutive,
                    sample=sample,
                    scheduler_receipt=receipt,
                    error=None,
                )
                if consecutive >= int(config["required_consecutive_safe_samples"]):
                    token = _reserve_gpu_owner(config, root, polls, sample, owner_override)
                    post_reservation = deferred.sample_nvml()
                    if not _safe(post_reservation, config):
                        token.release()
                        token = None
                        sample = post_reservation
                        consecutive = 0
                    else:
                        sample = post_reservation
                        break
                if polls < maximum_polls:
                    time.sleep(float(config["poll_interval_seconds"]))
            attempted += 1
            owned = _lease(config, attempted)
            _atomic_json(lease_path, owned, int(config["maximum_state_bytes"]))
            if token is None:
                continue
            with token:
                checkpoint = _checkpoint(
                    config_path,
                    config,
                    runtime,
                    state="reserved",
                    started_utc=started_utc,
                    attempted_cycles=attempted,
                    executed_cycles=executed,
                    polls=polls,
                    consecutive_safe=consecutive,
                    sample=sample,
                    scheduler_receipt=receipt,
                    error=None,
                )
                scheduler_path = _inside(root, config["bindings"]["scheduler_config"]["path"])
                result = run_scheduler(
                    scheduler_path,
                    maximum_wall_seconds=float(config["scheduler_slice_seconds"]),
                )
                receipt = _scheduler_receipt(result, config, root)
                executed += 1
                checkpoint = _checkpoint(
                    config_path,
                    config,
                    runtime,
                    state="running",
                    started_utc=started_utc,
                    attempted_cycles=attempted,
                    executed_cycles=executed,
                    polls=polls,
                    consecutive_safe=consecutive,
                    sample=sample,
                    scheduler_receipt=receipt,
                    error=None,
                )
            if receipt["reviewed_workload_succeeded"] and receipt["queue_pending"] == 0:
                break
        state = (
            "completed"
            if receipt
            and receipt.get("reviewed_workload_succeeded")
            and receipt.get("queue_pending") == 0
            else "stopped"
        )
        return _checkpoint(
            config_path,
            config,
            runtime,
            state=state,
            started_utc=started_utc,
            attempted_cycles=attempted,
            executed_cycles=executed,
            polls=int(checkpoint.get("polls_in_cycle", 0)) if checkpoint else 0,
            consecutive_safe=int(checkpoint.get("consecutive_safe_samples", 0))
            if checkpoint
            else 0,
            sample=sample,
            scheduler_receipt=receipt,
            error=None,
        )
    except Exception as error:
        _checkpoint(
            config_path,
            config,
            runtime,
            state="failed",
            started_utc=started_utc,
            attempted_cycles=attempted,
            executed_cycles=executed,
            polls=int(checkpoint.get("polls_in_cycle", 0)) if checkpoint else 0,
            consecutive_safe=int(checkpoint.get("consecutive_safe_samples", 0))
            if checkpoint
            else 0,
            sample=sample,
            scheduler_receipt=receipt,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        _release_service_lease(lease_path, owned, int(config["maximum_state_bytes"]))


def build_readiness(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    sample = deferred.sample_nvml()
    runtime = _inside(root, config["runtime_directory"])
    source = root / "src/sigma_theory_compiler/kastner_schlatter_deferred_gpu_handoff_service.py"
    test = root / "tests/test_kastner_schlatter_deferred_gpu_handoff_service.py"
    result: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA,
        "service_id": config["service_id"],
        "source_bindings": {
            **config["bindings"],
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source),
            },
            "test": {"path": test.relative_to(root).as_posix(), "file_sha256": _file_sha(test)},
        },
        "handoff_contract": {
            "waiting_backend": "NVML only; no CUDA context",
            "required_consecutive_safe_samples": config["required_consecutive_safe_samples"],
            "maximum_gpu_utilization_percent": config["maximum_gpu_utilization_percent"],
            "minimum_free_gpu_memory_mib": config["minimum_free_gpu_memory_mib"],
            "maximum_wait_polls_per_cycle": config["maximum_wait_polls_per_cycle"],
            "maximum_service_cycles": config["maximum_service_cycles"],
            "maximum_service_seconds": config["maximum_service_seconds"],
            "scheduler_slice_seconds": config["scheduler_slice_seconds"],
            "exact_service_pid_argv_lease": config["service_lease_name"],
            "stale_service_lease_recovery": config["service_recovery_name"],
            "shared_gpu_owner_lease_acquired_only_after_safe_samples": "deferred-gpu-owner.lease.json",
            "post_reservation_nvml_safe_recheck": True,
            "atomic_checkpoint": config["checkpoint_name"],
            "restart_resume_preserves_cycle_and_result_receipt": True,
            "completed_queue_resume_is_idempotent": True,
            "external_stop_control": config["stop_name"],
            "fixed_scheduler_evaluator": EXPECTED_EVALUATOR,
            "gpu_workers": 1,
            "cpu_workers": 0,
            "scheduler_owned_isolated_durable_queue_uses_sqlite": True,
            "handoff_service_direct_sqlite_surface": False,
            "automatic_detached_launch": False,
        },
        "current_runtime_audit": {
            "nvml_sample": sample,
            "single_sample_safe": _safe(sample, config),
            "runtime_exists": runtime.exists(),
            "service_lease_exists": (runtime / config["service_lease_name"]).exists(),
            "service_started_by_readiness": False,
            "gpu_owner_reserved_by_readiness": False,
            "scheduler_started_by_readiness": False,
        },
        "decision": "deferred_handoff_ready_current_device_occupied_not_started"
        if not _safe(sample, config)
        else "deferred_handoff_ready_single_sample_safe_not_started",
        "seals": config["seals"],
        "observations_opened": False,
        "scientific_test_pass": False,
        "readiness_advanced": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_readiness(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, root = load_config(config_path)
    if (
        result.get("schema_version") != READINESS_SCHEMA
        or result.get("content_sha256") != _content_sha(result)
        or result.get("seals") != EXPECTED_SEALS
    ):
        raise ValueError("deferred handoff readiness validation failed")
    sample = result.get("current_runtime_audit", {}).get("nvml_sample", {})
    expected_decision = (
        "deferred_handoff_ready_single_sample_safe_not_started"
        if _safe(sample, config)
        else "deferred_handoff_ready_current_device_occupied_not_started"
    )
    if result.get("decision") != expected_decision:
        raise ValueError("deferred handoff readiness decision changed")
    for name, binding in config["bindings"].items():
        if result.get("source_bindings", {}).get(name) != binding:
            raise ValueError(f"deferred handoff {name} binding changed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"deferred handoff {name} hash mismatch")
    audit = result["current_runtime_audit"]
    if any(
        audit.get(key) is not False
        for key in (
            "service_started_by_readiness",
            "gpu_owner_reserved_by_readiness",
            "scheduler_started_by_readiness",
        )
    ) or any(
        result.get(key) is not False
        for key in ("observations_opened", "scientific_test_pass", "readiness_advanced")
    ):
        raise ValueError("deferred handoff readiness execution or claim changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--write-readiness", action="store_true")
    parser.add_argument("--validate-readiness")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if sum((args.write_readiness, bool(args.validate_readiness), args.run)) != 1:
        raise ValueError("select exactly one deferred handoff operation")
    if args.validate_readiness:
        validate_readiness(json.loads(Path(args.validate_readiness).read_text()), args.config)
        return 0
    if args.run:
        print(_canonical(run_service(args.config)))
        return 0
    result = build_readiness(args.config)
    config, root = load_config(args.config)
    output = _inside(root, config["output_path"])
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
