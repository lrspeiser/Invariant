"""Closed durable GPU scheduler adapter for one reviewed Kastner--Schlatter workload."""

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

from .kastner_schlatter_set_indexed_cuda_falsification_campaign import (
    CONFIG_SCHEMA as WORKLOAD_CONFIG_SCHEMA,
)
from .kastner_schlatter_set_indexed_cuda_falsification_campaign import execute_gpu_workload
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .persistent_parallel_supervisor import PersistentParallelSupervisor

SCHEMA = "sigma-kastner-schlatter-set-indexed-gpu-scheduler-adapter-readiness-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-set-indexed-gpu-scheduler-adapter-config-1.0"
REVIEWED_WORKLOAD_ID = "kastner-schlatter-set-indexed-cuda-falsification-001"
FIXED_EVALUATOR = (
    "sigma_theory_compiler.kastner_schlatter_set_indexed_gpu_scheduler_adapter:"
    "gpu_reviewed_workload_evaluator"
)
EXPECTED_RUNTIME_DIRECTORY = "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-runtime"
EXPECTED_DATABASE_NAME = "durable-gpu-queue.sqlite"
EXPECTED_TELEMETRY_NAME = "supervisor-telemetry.jsonl"
EXPECTED_SEALS = {
    "arbitrary_callable_injection": False,
    "arbitrary_subprocess_injection": False,
    "cpu_worker_execution": False,
    "live_campaign_sqlite_access": False,
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
        raise ValueError("adapter path escapes repository") from error
    return target


def load_adapter_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "adapter_id", "reviewed_workload_id", "runtime_directory",
        "database_name", "telemetry_name", "output_path", "bindings", "persistent_config",
        "service", "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported scheduler-adapter config")
    if config.get("reviewed_workload_id") != REVIEWED_WORKLOAD_ID:
        raise ValueError("unreviewed workload ID rejected")
    if (
        config.get("runtime_directory") != EXPECTED_RUNTIME_DIRECTORY
        or config.get("database_name") != EXPECTED_DATABASE_NAME
        or config.get("telemetry_name") != EXPECTED_TELEMETRY_NAME
    ):
        raise ValueError("scheduler runtime path contract changed")
    if config.get("seals") != EXPECTED_SEALS:
        raise ValueError("scheduler safety seals changed")
    expected_bindings = {
        "workload_config", "workload_source", "persistent_search_source",
        "persistent_supervisor_source", "resource_profile", "gitignore",
    }
    if set(config.get("bindings", {})) != expected_bindings:
        raise ValueError("scheduler binding set changed")
    for name, binding in config["bindings"].items():
        if set(binding) != {"path", "file_sha256"}:
            raise ValueError(f"invalid scheduler binding: {name}")
        bound_path = _inside(root, binding["path"])
        if _file_sha(bound_path) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    workload = json.loads(
        _inside(root, config["bindings"]["workload_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        workload.get("schema_version") != WORKLOAD_CONFIG_SCHEMA
        or workload.get("campaign_id") != REVIEWED_WORKLOAD_ID
        or workload.get("synthetic_only") is not True
        or workload.get("observations_opened") is not False
    ):
        raise ValueError("reviewed workload boundary changed")
    ignored_runtime = f"{config['runtime_directory'].rstrip('/')}/"
    gitignore = _inside(root, config["bindings"]["gitignore"]["path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    if ignored_runtime not in gitignore:
        raise ValueError("scheduler runtime directory is not Git-ignored")
    persistent = config["persistent_config"]
    supervisor = persistent.get("supervisor", {})
    if (
        persistent.get("external_paid_llm_calls") is not False
        or persistent.get("budget", {}).get("maximum_tasks") != 1
        or persistent.get("queue", {}).get("maximum_pending_work") != 1
        or persistent.get("cpu", {}).get("maximum_workers") != 0
        or supervisor.get("cpu_workers") != 0
        or supervisor.get("gpu_workers") != 1
        or supervisor.get("gpu_evaluator") != FIXED_EVALUATOR
        or supervisor.get("maximum_process_restarts") != 2
    ):
        raise ValueError("single-owner closed supervisor contract changed")
    service = config["service"]
    expected_service_keys = {
        "service_epoch", "lease_name", "checkpoint_name", "stop_name",
        "idle_poll_seconds", "supervisor_slice_seconds", "maximum_service_cycles",
        "maximum_start_gpu_utilization_percent", "minimum_start_free_gpu_memory_mib",
    }
    if (
        set(service) != expected_service_keys
        or not str(service.get("service_epoch", "")).startswith(
            "kastner-schlatter-set-indexed-gpu-service-"
        )
        or service.get("lease_name") != "service.lease.json"
        or service.get("checkpoint_name") != "service-checkpoint.json"
        or service.get("stop_name") != "stop.request"
        or any(
            float(service[key]) <= 0
            for key in (
                "idle_poll_seconds", "supervisor_slice_seconds", "maximum_service_cycles",
                "maximum_start_gpu_utilization_percent", "minimum_start_free_gpu_memory_mib",
            )
        )
        or float(service["maximum_start_gpu_utilization_percent"]) > 20
    ):
        raise ValueError("continuous GPU service contract changed")
    _inside(root, config["runtime_directory"])
    _inside(root, config["output_path"])
    return config, root


def _reviewed_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    workload = config["bindings"]["workload_config"]
    source = config["bindings"]["workload_source"]
    body = {
        "ordinal": 0,
        "candidate_count": 1,
        "workload_id": REVIEWED_WORKLOAD_ID,
        "workload_config_path": workload["path"],
        "workload_config_file_sha256": workload["file_sha256"],
        "workload_source_path": source["path"],
        "workload_source_file_sha256": source["file_sha256"],
        "allowed_evaluator": FIXED_EVALUATOR,
        "artifact_write_allowed": False,
        "live_campaign_sqlite_access_allowed": False,
    }
    return {**body, "idempotency_key": _sha(body)}


def _validate_payload(payload: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    expected = _reviewed_payload(config)
    if dict(payload) != expected:
        raise ValueError("unreviewed or tampered GPU work payload rejected")
    forbidden = {"command", "argv", "subprocess", "callable", "module", "function"}
    if forbidden.intersection(payload):
        raise ValueError("callable or subprocess injection rejected")


def gpu_reviewed_workload_evaluator(lease: WorkLease) -> dict[str, Any]:
    """Fixed spawned-worker evaluator; its only side effect is the coordinator result commit."""
    if lease.lane != "gpu" or lease.ordinal != 0:
        raise ValueError("reviewed workload requires the sole GPU ordinal")
    config_path_text = lease.payload.get("adapter_config_path")
    if not isinstance(config_path_text, str):
        raise TypeError("adapter config binding missing")
    adapter_config_sha = lease.payload.get("adapter_config_file_sha256")
    if not isinstance(adapter_config_sha, str) or _file_sha(
        Path(config_path_text).resolve()
    ) != adapter_config_sha:
        raise ValueError("adapter config hash mismatch")
    config, root = load_adapter_config(config_path_text)
    payload = {
        key: item
        for key, item in lease.payload.items()
        if key not in {"adapter_config_path", "adapter_config_file_sha256"}
    }
    _validate_payload(payload, config)
    workload_path = _inside(root, payload["workload_config_path"])
    source_path = _inside(root, payload["workload_source_path"])
    if (
        _file_sha(workload_path) != payload["workload_config_file_sha256"]
        or _file_sha(source_path) != payload["workload_source_file_sha256"]
    ):
        raise ValueError("reviewed workload changed after lease creation")
    workload_config = json.loads(workload_path.read_text(encoding="utf-8"))
    computed = execute_gpu_workload(workload_config)
    return {
        "schema_version": "sigma-reviewed-set-indexed-gpu-queue-result-1.0",
        "workload_id": REVIEWED_WORKLOAD_ID,
        "work_id": lease.work_id,
        "ordinal": lease.ordinal,
        "attempt": lease.attempt,
        "seed": lease.seed,
        "idempotency_key": payload["idempotency_key"],
        "workload_config_file_sha256": payload["workload_config_file_sha256"],
        "workload_source_file_sha256": payload["workload_source_file_sha256"],
        "compute_result": computed,
        "immutable_artifact_written": False,
        "live_campaign_sqlite_accessed": False,
        "scientific_or_readiness_promotion": False,
    }


def create_coordinator(
    config_path: str | Path, *, runtime_override: str | Path | None = None
) -> tuple[PersistentParallelSearch, dict[str, Any], Path, Path]:
    config, root = load_adapter_config(config_path)
    runtime = (
        Path(runtime_override).resolve()
        if runtime_override is not None
        else _inside(root, config["runtime_directory"])
    )
    live_campaign_directory = (root / "runs/campaigns").resolve()
    try:
        runtime.relative_to(live_campaign_directory)
    except ValueError:
        pass
    else:
        raise ValueError("live campaign runtime override rejected")
    database = runtime / config["database_name"]
    telemetry = runtime / config["telemetry_name"]
    resource = json.loads(
        _inside(root, config["bindings"]["resource_profile"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    coordinator = PersistentParallelSearch(database, config["persistent_config"], resource)
    if coordinator.plan["gpu"]["workers"] != 1 or coordinator.plan["cpu"]["workers"] != 0:
        raise ValueError("hardware plan does not provide exactly one GPU owner")
    return coordinator, config, root, telemetry


def enqueue_reviewed_workload(
    coordinator: PersistentParallelSearch,
    config: Mapping[str, Any],
    config_path: str | Path,
) -> dict[str, int]:
    payload = _reviewed_payload(config)
    payload["adapter_config_path"] = str(Path(config_path).resolve())
    payload["adapter_config_file_sha256"] = _file_sha(Path(config_path).resolve())
    return coordinator.enqueue([payload], lane="gpu", max_attempts=3)


def run_scheduler(
    config_path: str | Path,
    *,
    runtime_override: str | Path | None = None,
    maximum_wall_seconds: float | None = None,
) -> dict[str, Any]:
    coordinator, config, root, telemetry = create_coordinator(
        config_path, runtime_override=runtime_override
    )
    enqueue = enqueue_reviewed_workload(coordinator, config, config_path)
    resource = json.loads(
        _inside(root, config["bindings"]["resource_profile"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    supervisor = PersistentParallelSupervisor(
        coordinator.database,
        config["persistent_config"],
        resource,
        telemetry,
    )
    report = supervisor.run(maximum_wall_seconds=maximum_wall_seconds)
    return {
        "enqueue": enqueue,
        "supervisor": report,
        "database": str(coordinator.database),
        "telemetry": str(telemetry),
        "immutable_artifact_written": False,
        "live_campaign_sqlite_accessed": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("scheduler service symlink target rejected")
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
        if temporary.exists():
            temporary.unlink()


def _process_identity(argv: list[str]) -> str:
    try:
        module_index = argv.index("-m")
    except ValueError:
        return ""
    return _sha(argv[module_index:])


def _current_process_identity() -> str:
    try:
        import psutil

        return _process_identity(psutil.Process(os.getpid()).cmdline())
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("scheduler service process identity unavailable") from error


def _owner_matches(pid: int, identity: str) -> bool:
    try:
        import psutil

        process = psutil.Process(pid)
        return process.is_running() and _process_identity(process.cmdline()) == identity
    except (ImportError, OSError, ValueError):
        return False
    except psutil.Error:
        return False


def _lease_document(config: Mapping[str, Any], *, cycle: int = 0) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sigma-set-indexed-gpu-service-lease-1.0",
        "service_epoch": config["service"]["service_epoch"],
        "pid": os.getpid(),
        "process_argv_sha256": _current_process_identity(),
        "cycle": cycle,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    return {**body, "content_sha256": _content_sha(body)}


def _acquire_service_lease(runtime: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    runtime.mkdir(parents=True, exist_ok=True)
    if runtime.is_symlink():
        raise RuntimeError("scheduler runtime symlink rejected")
    lease_path = runtime / config["service"]["lease_name"]
    document = _lease_document(config)
    payload = (_canonical(document) + "\n").encode()
    try:
        descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = json.loads(lease_path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") != _content_sha(existing):
            raise RuntimeError("scheduler service lease tamper detected") from None
        pid = existing.get("pid")
        identity = existing.get("process_argv_sha256")
        if isinstance(pid, int) and isinstance(identity, str) and _owner_matches(pid, identity):
            raise RuntimeError("scheduler service already active") from None
        recovery = lease_path.with_name("service.lease.recovery.json")
        _atomic_json(recovery, existing)
        lease_path.unlink()
        descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return lease_path, document


def _release_service_lease(lease_path: Path, owned: Mapping[str, Any]) -> None:
    if not lease_path.exists():
        return
    current = json.loads(lease_path.read_text(encoding="utf-8"))
    if (
        current.get("pid") == owned.get("pid")
        and current.get("process_argv_sha256") == owned.get("process_argv_sha256")
    ):
        lease_path.unlink()


def gpu_start_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless device 0 is quiet enough for a new exclusive campaign owner."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mib = int(memory.free // 1024**2)
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        raise RuntimeError(f"GPU start gate unavailable: {type(error).__name__}: {error}") from error
    maximum = int(config["service"]["maximum_start_gpu_utilization_percent"])
    minimum_free = int(config["service"]["minimum_start_free_gpu_memory_mib"])
    if utilization > maximum or free_mib < minimum_free:
        raise RuntimeError(
            f"GPU start gate blocked: utilization={utilization}% free_mib={free_mib}"
        )
    return {
        "device_index": 0,
        "gpu_utilization_percent": utilization,
        "free_memory_mib": free_mib,
        "maximum_start_gpu_utilization_percent": maximum,
        "minimum_start_free_gpu_memory_mib": minimum_free,
        "decision": "start_allowed",
        "counter_scope": "device-wide instantaneous NVML preflight; external owners can appear later",
    }


def _service_checkpoint(
    config_path: Path,
    config: Mapping[str, Any],
    runtime: Path,
    *,
    state: str,
    cycle: int,
    start_gate: Mapping[str, Any],
    queue: Mapping[str, Any] | None,
    last_run: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sigma-set-indexed-gpu-service-checkpoint-1.0",
        "service_epoch": config["service"]["service_epoch"],
        "state": state,
        "pid": os.getpid() if state not in {"stopped", "failed"} else None,
        "process_argv_sha256": _current_process_identity(),
        "adapter_config_file_sha256": _file_sha(config_path),
        "adapter_source_file_sha256": _file_sha(Path(__file__).resolve()),
        "cycle": cycle,
        "start_gate": dict(start_gate),
        "queue": dict(queue) if queue is not None else None,
        "last_supervisor_run": dict(last_run) if last_run is not None else None,
        "error": error,
        "immutable_artifact_written": False,
        "live_campaign_sqlite_accessed": False,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    value = {**body, "content_sha256": _content_sha(body)}
    _atomic_json(runtime / config["service"]["checkpoint_name"], value)
    return value


def run_continuous_service(
    config_path: str | Path,
    *,
    runtime_override: str | Path | None = None,
    maximum_cycles_override: int | None = None,
) -> dict[str, Any]:
    """Run a foreground restart-safe service; no detached process is launched."""
    config_path = Path(config_path).resolve()
    config, root = load_adapter_config(config_path)
    start_gate = gpu_start_gate(config)
    runtime = (
        Path(runtime_override).resolve()
        if runtime_override is not None
        else _inside(root, config["runtime_directory"])
    )
    live_campaign_directory = (root / "runs/campaigns").resolve()
    try:
        runtime.relative_to(live_campaign_directory)
    except ValueError:
        pass
    else:
        raise ValueError("live campaign runtime override rejected")
    lease_path, owned = _acquire_service_lease(runtime, config)
    cycle = 0
    last_run: dict[str, Any] | None = None
    maximum_cycles = int(
        maximum_cycles_override
        if maximum_cycles_override is not None
        else config["service"]["maximum_service_cycles"]
    )
    checkpoint: dict[str, Any] | None = None
    try:
        checkpoint = _service_checkpoint(
            config_path, config, runtime, state="starting", cycle=cycle,
            start_gate=start_gate, queue=None, last_run=None, error=None,
        )
        while cycle < maximum_cycles:
            if (runtime / config["service"]["stop_name"]).exists():
                break
            coordinator, _, _, _ = create_coordinator(
                config_path, runtime_override=runtime
            )
            enqueue_reviewed_workload(coordinator, config, config_path)
            telemetry = coordinator.telemetry()
            if int(telemetry["queue"]["pending"]) > 0:
                last_run = run_scheduler(
                    config_path,
                    runtime_override=runtime,
                    maximum_wall_seconds=float(
                        config["service"]["supervisor_slice_seconds"]
                    ),
                )
                telemetry = coordinator.telemetry()
            cycle += 1
            owned = _lease_document(config, cycle=cycle)
            _atomic_json(lease_path, owned)
            checkpoint = _service_checkpoint(
                config_path,
                config,
                runtime,
                state="running" if telemetry["queue"]["pending"] else "idle",
                cycle=cycle,
                start_gate=start_gate,
                queue=telemetry,
                last_run=last_run,
                error=None,
            )
            if cycle >= maximum_cycles:
                break
            time.sleep(float(config["service"]["idle_poll_seconds"]))
        checkpoint = _service_checkpoint(
            config_path,
            config,
            runtime,
            state="stopped",
            cycle=cycle,
            start_gate=start_gate,
            queue=checkpoint.get("queue") if checkpoint else None,
            last_run=last_run,
            error=None,
        )
        return checkpoint
    except Exception as error:
        checkpoint = _service_checkpoint(
            config_path,
            config,
            runtime,
            state="failed",
            cycle=cycle,
            start_gate=start_gate,
            queue=checkpoint.get("queue") if checkpoint else None,
            last_run=last_run,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        _release_service_lease(lease_path, owned)


def durable_results(coordinator: PersistentParallelSearch) -> list[dict[str, Any]]:
    with coordinator.connect() as connection:
        rows = connection.execute(
            "SELECT work_id,state,attempt,result_json,error_text FROM work ORDER BY ordinal"
        ).fetchall()
    return [
        {
            "work_id": row["work_id"],
            "state": row["state"],
            "attempt": row["attempt"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error_text"],
        }
        for row in rows
    ]


def build_readiness(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, root = load_adapter_config(config_path)
    source = root / "src/sigma_theory_compiler/kastner_schlatter_set_indexed_gpu_scheduler_adapter.py"
    test = root / "tests/test_kastner_schlatter_set_indexed_gpu_scheduler_adapter.py"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "adapter_id": config["adapter_id"],
        "source_bindings": {
            **config["bindings"],
            "config": {"path": config_path.relative_to(root).as_posix(), "file_sha256": _file_sha(config_path)},
            "source": {"path": source.relative_to(root).as_posix(), "file_sha256": _file_sha(source)},
            "test": {"path": test.relative_to(root).as_posix(), "file_sha256": _file_sha(test)},
        },
        "scheduler_contract": {
            "coordinator": "PersistentParallelSearch",
            "supervisor": "PersistentParallelSupervisor",
            "durable_queue_result_column": "work.result_json",
            "gpu_owner_count": 1,
            "cpu_worker_count": 0,
            "maximum_queue_items": 1,
            "maximum_attempts": 3,
            "lease_seconds": 120,
            "process_restart_budget": 2,
            "idempotency": "fixed ordinal/lane plus deterministic reviewed payload",
            "crash_recovery": "expired leases requeue until maximum attempts",
            "fixed_evaluator": FIXED_EVALUATOR,
            "pure_compute_hook": "execute_gpu_workload(config: Mapping[str, Any]) -> dict[str, Any]",
            "arbitrary_callable_or_subprocess_surface": False,
        },
        "continuous_service_contract": {
            "foreground_only_no_detached_launcher": True,
            "service_epoch": config["service"]["service_epoch"],
            "exclusive_pid_argv_lease": config["service"]["lease_name"],
            "atomic_checkpoint": config["service"]["checkpoint_name"],
            "external_stop_request": config["service"]["stop_name"],
            "stale_lease_recovery_requires_owner_nonmatch": True,
            "idempotent_queue_resume_each_cycle": True,
            "maximum_service_cycles": config["service"]["maximum_service_cycles"],
            "supervisor_slice_seconds": config["service"]["supervisor_slice_seconds"],
            "gpu_start_gate": {
                "maximum_device_wide_utilization_percent": config["service"]["maximum_start_gpu_utilization_percent"],
                "minimum_free_memory_mib": config["service"]["minimum_start_free_gpu_memory_mib"],
                "fails_closed_if_nvml_unavailable": True,
            },
            "runtime_outputs_gitignored": True,
        },
        "execution_state": {
            "runtime_created_by_readiness": False,
            "scheduler_started_by_readiness": False,
            "worker_result_created_by_readiness": False,
        },
        "decision": "durable_single_owner_gpu_continuous_service_ready_start_gated_not_started",
        "seals": config["seals"],
        "observations_opened": False,
        "scientific_test_pass": False,
        "readiness_advanced": False,
        "live_campaign_sqlite_accessed": False,
        "paid_llm_calls": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_readiness(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, root = load_adapter_config(config_path)
    if result.get("schema_version") != SCHEMA or result.get("content_sha256") != _content_sha(result):
        raise ValueError("readiness schema or content hash mismatch")
    if result.get("decision") != "durable_single_owner_gpu_continuous_service_ready_start_gated_not_started":
        raise ValueError("readiness decision changed")
    if result.get("seals") != EXPECTED_SEALS:
        raise ValueError("readiness seals changed")
    for key in (
        "observations_opened", "scientific_test_pass", "readiness_advanced",
        "live_campaign_sqlite_accessed", "paid_llm_calls",
    ):
        if result.get(key) is not False:
            raise ValueError(f"readiness claim changed: {key}")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"readiness {name} hash mismatch")
    if result["source_bindings"]["workload_config"] != config["bindings"]["workload_config"]:
        raise ValueError("reviewed workload config binding changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--write-readiness", action="store_true")
    parser.add_argument("--validate-readiness")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--service", action="store_true")
    args = parser.parse_args()
    selected = sum(
        (args.write_readiness, bool(args.validate_readiness), args.run, args.service)
    )
    if selected != 1:
        raise ValueError("select exactly one closed adapter operation")
    if args.validate_readiness:
        validate_readiness(
            json.loads(Path(args.validate_readiness).read_text(encoding="utf-8")), args.config
        )
        return 0
    if args.run:
        print(_canonical(run_scheduler(args.config)))
        return 0
    if args.service:
        print(_canonical(run_continuous_service(args.config)))
        return 0
    result = build_readiness(args.config)
    config, root = load_adapter_config(args.config)
    output = _inside(root, config["output_path"])
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
