from __future__ import annotations

import hashlib
import html
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .binary_formula_execution import (
    BinaryBlockQueueRefill,
    configure_binary_evaluators,
)
from .persistent_parallel_search import PersistentParallelSearch, plan_parallel_capacity
from .persistent_parallel_supervisor import PersistentParallelSupervisor
from .process_health import pid_alive
from .real_formula_execution import FiniteFormulaQueueRefill, configure_real_evaluators

SERVICE_SCHEMA = "sigma-gravity-engine-service-1.0"
STATUS_SCHEMA = "sigma-gravity-engine-status-1.0"
EXPORT_SCHEMA = "sigma-gravity-engine-export-1.0"
ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _hardware_telemetry() -> dict[str, Any]:
    """Sample host CPU and NVIDIA device load without making either a dependency."""

    sample: dict[str, Any] = {
        "semantics": "instantaneous host/device sensor sample; distinct from lease occupancy",
        "cpu": {"available": False},
        "gpu": {"available": False},
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        sample["cpu"] = {
            "available": True,
            "utilization_percent": psutil.cpu_percent(interval=0.05),
            "logical_processors": psutil.cpu_count(logical=True),
            "memory_used_bytes": int(memory.used),
            "memory_total_bytes": int(memory.total),
        }
    except (ImportError, OSError, RuntimeError) as error:
        sample["cpu"]["error"] = type(error).__name__
    try:
        import pynvml
    except ImportError as error:
        sample["gpu"]["error"] = type(error).__name__
    else:
        try:
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                sample["gpu"] = {
                    "available": True,
                    "device": name.decode() if isinstance(name, bytes) else str(name),
                    "utilization_percent": int(utilization.gpu),
                    "memory_controller_utilization_percent": int(utilization.memory),
                    "memory_used_bytes": int(memory.used),
                    "memory_total_bytes": int(memory.total),
                }
                try:
                    sample["gpu"]["power_watts"] = (
                        float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    )
                except pynvml.NVMLError:
                    sample["gpu"]["power_watts"] = None
            finally:
                pynvml.nvmlShutdown()
        except (pynvml.NVMLError, OSError, RuntimeError) as error:
            sample["gpu"]["error"] = type(error).__name__
    return sample


def _pid_alive(pid: int | None) -> bool:
    return pid_alive(pid)


def _resolve_adapter_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    normalized = json.loads(json.dumps(config))
    for key in ("generator_config_path", "manifest_path", "survivor_directory"):
        if key not in normalized:
            continue
        value = Path(normalized[key])
        if value.is_absolute():
            resolved = value.resolve()
        else:
            candidates = [
                (Path.cwd() / value).resolve(),
                (config_path.parent / value).resolve(),
                (config_path.parent.parent / value).resolve(),
            ]
            resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        normalized[key] = str(resolved)
    return normalized


def _service_paths(service_directory: str | Path) -> dict[str, Path]:
    root = Path(service_directory).resolve()
    return {
        "root": root,
        "service": root / "service.json",
        "execution": root / "execution-config.json",
        "resource": root / "resource-profile.json",
        "adapter": root / "adapter-config.json",
        "database": root / "engine.sqlite",
        "telemetry": root / "telemetry.jsonl",
        "stop": root / "stop.request",
        "status": root / "status-summary.json",
        "last_run": root / "last-run.json",
        "dashboard": root / "dashboard.html",
        "log": root / "service.log",
    }


def _configure(
    mode: str, execution: dict[str, Any], adapter: dict[str, Any]
) -> dict[str, Any]:
    if mode == "real":
        return configure_real_evaluators(execution, adapter)
    if mode == "binary":
        return configure_binary_evaluators(execution, adapter)
    raise ValueError("engine mode must be real or binary")


def initialize_service(
    service_directory: str | Path,
    execution_config_path: str | Path,
    resource_profile_path: str | Path,
    adapter_config_path: str | Path,
    *,
    mode: str,
    maximum_tasks: int | None = None,
    maximum_wall_seconds: float | None = None,
    maximum_disk_bytes: int = 8 * 1024**3,
) -> dict[str, Any]:
    paths = _service_paths(service_directory)
    if paths["service"].exists():
        raise FileExistsError("engine service already exists; use engine-resume")
    if maximum_disk_bytes <= 0:
        raise ValueError("engine disk budget must be positive")
    execution_path = Path(execution_config_path).resolve()
    resource_path = Path(resource_profile_path).resolve()
    adapter_path = Path(adapter_config_path).resolve()
    execution = _load(execution_path)
    resource = _load(resource_path)
    adapter = _resolve_adapter_paths(_load(adapter_path), adapter_path)
    if execution.get("external_paid_llm_calls") is not False:
        raise ValueError("paid LLM calls must remain disabled")
    if adapter.get("external_paid_llm_calls") is not False:
        raise ValueError("adapter paid LLM calls must remain disabled")
    if adapter.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("engine data eligibility is not fail-closed")
    configured = _configure(mode, execution, adapter)
    if maximum_tasks is not None:
        if maximum_tasks <= 0:
            raise ValueError("maximum tasks must be positive")
        configured["budget"]["maximum_tasks"] = min(
            int(configured["budget"]["maximum_tasks"]), maximum_tasks
        )
    if maximum_wall_seconds is not None:
        if maximum_wall_seconds <= 0:
            raise ValueError("maximum wall seconds must be positive")
        configured["budget"]["maximum_wall_seconds"] = min(
            float(configured["budget"]["maximum_wall_seconds"]), maximum_wall_seconds
        )
    configured["supervisor"]["maximum_wall_seconds_per_run"] = float(
        configured["budget"]["maximum_wall_seconds"]
    )
    configured["supervisor"]["maximum_telemetry_bytes"] = min(
        int(configured["supervisor"]["maximum_telemetry_bytes"]),
        max(1, maximum_disk_bytes // 4),
    )
    plan_parallel_capacity(resource, configured)
    snapshot_estimate = sum(
        len(json.dumps(value, indent=2, sort_keys=True).encode()) + 1
        for value in (configured, resource, adapter)
    ) + 16 * 1024
    if snapshot_estimate >= maximum_disk_bytes:
        raise ValueError("engine disk budget cannot hold its configuration and state")
    paths["root"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["execution"], configured)
    _write_json(paths["resource"], resource)
    _write_json(paths["adapter"], adapter)
    identity = {
        "mode": mode,
        "execution_sha256": _sha(configured),
        "resource_sha256": _sha(resource),
        "adapter_sha256": _sha(adapter),
        "maximum_disk_bytes": int(maximum_disk_bytes),
    }
    now = datetime.now(UTC).isoformat()
    service = {
        "schema_version": SERVICE_SCHEMA,
        "service_id": f"SGE-{_sha(identity)[:24]}",
        "identity": identity,
        "mode": mode,
        "state": "initialized",
        "pid": None,
        "created_utc": now,
        "updated_utc": now,
        "last_stop_reason": None,
        "cost_budget": {
            "paid_llm_calls_enabled": False,
            "maximum_paid_llm_spend_usd": 0.0,
            "paid_llm_spend_usd": 0.0,
        },
        "paths": {key: str(value) for key, value in paths.items() if key != "root"},
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    _write_json(paths["service"], service)
    return service


def _update_service(paths: dict[str, Path], **changes: Any) -> dict[str, Any]:
    service = _load(paths["service"])
    service.update(changes)
    service["updated_utc"] = datetime.now(UTC).isoformat()
    _write_json(paths["service"], service)
    return service


def _source_status(database: Path, mode: str) -> dict[str, Any] | None:
    if not database.exists():
        return None
    table = "formula_generator_cursor" if mode == "real" else "binary_block_cursor"
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        row = connection.execute(f"SELECT * FROM {table} ORDER BY source_id LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    finally:
        if "connection" in locals():
            connection.close()
    if row is None:
        return None
    value = dict(row)
    if mode == "real":
        value["exhausted"] = int(value["next_ordinal"]) >= int(value["stop_ordinal"])
    else:
        value["exhausted"] = int(value["next_position"]) >= int(value["stop_position"])
    return value


def _latest_telemetry(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64 * 1024))
        lines = handle.read().splitlines()
    return json.loads(lines[-1]) if lines else None


def _write_dashboard(path: Path, status: dict[str, Any]) -> None:
    payload = html.escape(json.dumps(status, indent=2, sort_keys=True))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>Sigma Gravity Engine</title><style>
body{{font:15px system-ui;margin:2rem;background:#10151c;color:#e8eef6}}
main{{max-width:1100px;margin:auto}}pre{{padding:1rem;background:#17202b;overflow:auto}}
.ok{{color:#78e08f}}.warn{{color:#f6b93b}}
</style></head><body><main><h1>Sigma Gravity Engine</h1>
<p class="{'ok' if status.get('alive') else 'warn'}">state={html.escape(str(status.get('state')))} · alive={status.get('alive')}</p>
<pre>{payload}</pre></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def service_status(service_directory: str | Path, *, write_artifacts: bool = True) -> dict[str, Any]:
    paths = _service_paths(service_directory)
    if not paths["service"].exists():
        raise FileNotFoundError("engine service does not exist")
    service = _load(paths["service"])
    execution = _load(paths["execution"])
    resource = _load(paths["resource"])
    telemetry = None
    if paths["database"].exists():
        coordinator = PersistentParallelSearch(paths["database"], execution, resource)
        telemetry = coordinator.telemetry()
    alive = _pid_alive(service.get("pid"))
    status = {
        "schema_version": STATUS_SCHEMA,
        "service_id": service["service_id"],
        "mode": service["mode"],
        "state": service["state"],
        "pid": service.get("pid"),
        "alive": alive,
        "stop_requested": paths["stop"].exists(),
        "last_stop_reason": service.get("last_stop_reason"),
        "cost_budget": service["cost_budget"],
        "source": _source_status(paths["database"], service["mode"]),
        "execution": telemetry,
        "hardware": _hardware_telemetry(),
        "last_periodic_telemetry": _latest_telemetry(paths["telemetry"]),
        "disk": {
            "used_bytes": _directory_bytes(paths["root"]),
            "maximum_bytes": int(service["identity"]["maximum_disk_bytes"]),
        },
        "data_eligibility": service["data_eligibility"],
    }
    if write_artifacts:
        _write_json(paths["status"], status)
        _write_dashboard(paths["dashboard"], status)
    return status


def run_service_worker(service_directory: str | Path) -> dict[str, Any]:
    paths = _service_paths(service_directory)
    service = _load(paths["service"])
    execution = _load(paths["execution"])
    resource = _load(paths["resource"])
    adapter = _load(paths["adapter"])
    if service.get("data_eligibility") != {**ELIGIBILITY, "paid_llm_calls": False, "passed": True}:
        raise ValueError("stored engine eligibility is invalid")
    if service.get("cost_budget") != {
        "paid_llm_calls_enabled": False,
        "maximum_paid_llm_spend_usd": 0.0,
        "paid_llm_spend_usd": 0.0,
    }:
        raise ValueError("stored engine paid-LLM budget is not hard-zero")
    if service["identity"] != {
        "mode": service["mode"],
        "execution_sha256": _sha(execution),
        "resource_sha256": _sha(resource),
        "adapter_sha256": _sha(adapter),
        "maximum_disk_bytes": int(service["identity"]["maximum_disk_bytes"]),
    }:
        raise ValueError("stored engine configuration identity mismatch")
    _update_service(paths, state="running", pid=os.getpid(), last_stop_reason=None)
    coordinator = PersistentParallelSearch(paths["database"], execution, resource)
    source = (
        FiniteFormulaQueueRefill(coordinator, adapter)
        if service["mode"] == "real"
        else BinaryBlockQueueRefill(coordinator, adapter)
    )

    def disk_reason() -> str | None:
        return (
            "disk_budget_exhausted"
            if _directory_bytes(paths["root"])
            >= int(service["identity"]["maximum_disk_bytes"])
            else None
        )

    def periodic(value: dict[str, Any]) -> None:
        summary = {
            "schema_version": STATUS_SCHEMA,
            "service_id": service["service_id"],
            "mode": service["mode"],
            "state": "running",
            "pid": os.getpid(),
            "alive": True,
            "stop_requested": paths["stop"].exists(),
            "cost_budget": service["cost_budget"],
            "source": value.get("refill", {}).get("cursor"),
            "execution": value["execution"],
            "processes": value["processes"],
            "hardware": _hardware_telemetry(),
            "disk": {
                "used_bytes": _directory_bytes(paths["root"]),
                "maximum_bytes": int(service["identity"]["maximum_disk_bytes"]),
            },
            "data_eligibility": service["data_eligibility"],
        }
        _write_json(paths["status"], summary)
        _write_dashboard(paths["dashboard"], summary)

    try:
        report = PersistentParallelSupervisor(
            paths["database"], execution, resource, paths["telemetry"]
        ).run(
            refill_callback=source.refill,
            external_stop_path=paths["stop"],
            stop_reason_callback=disk_reason,
            status_callback=periodic,
        )
        cursor = source.status()
        completed = report["stop_reason"] == "queue_drained" and cursor["exhausted"]
        state = "completed" if completed else "stopped"
        _write_json(paths["last_run"], report)
        _update_service(
            paths,
            state=state,
            pid=None,
            last_stop_reason=report["stop_reason"],
        )
        service_status(paths["root"])
        return report
    except BaseException as error:
        _update_service(
            paths,
            state="failed",
            pid=None,
            last_stop_reason=f"{type(error).__name__}: {error}",
        )
        raise


def _spawn_worker(paths: dict[str, Path]) -> int:
    command = [
        sys.executable,
        "-m",
        "sigma_theory_compiler.cli",
        "engine-worker",
        "--service-dir",
        str(paths["root"]),
    ]
    log = paths["log"].open("ab")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    log.close()
    _update_service(paths, state="starting", pid=process.pid)
    return process.pid


def start_service(
    service_directory: str | Path,
    execution_config_path: str | Path,
    resource_profile_path: str | Path,
    adapter_config_path: str | Path,
    *,
    mode: str,
    foreground: bool,
    maximum_tasks: int | None = None,
    maximum_wall_seconds: float | None = None,
    maximum_disk_bytes: int = 8 * 1024**3,
) -> dict[str, Any]:
    initialize_service(
        service_directory,
        execution_config_path,
        resource_profile_path,
        adapter_config_path,
        mode=mode,
        maximum_tasks=maximum_tasks,
        maximum_wall_seconds=maximum_wall_seconds,
        maximum_disk_bytes=maximum_disk_bytes,
    )
    paths = _service_paths(service_directory)
    if foreground:
        report = run_service_worker(paths["root"])
        return {"foreground": True, "run": report, "status": service_status(paths["root"])}
    pid = _spawn_worker(paths)
    return {"foreground": False, "pid": pid, "status": service_status(paths["root"])}


def stop_service(service_directory: str | Path, *, wait_seconds: float = 10.0) -> dict[str, Any]:
    paths = _service_paths(service_directory)
    service = _load(paths["service"])
    paths["stop"].write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while _pid_alive(service.get("pid")) and time.monotonic() < deadline:
        time.sleep(0.05)
        service = _load(paths["service"])
    return service_status(paths["root"])


def resume_service(service_directory: str | Path, *, foreground: bool) -> dict[str, Any]:
    paths = _service_paths(service_directory)
    service = _load(paths["service"])
    if _pid_alive(service.get("pid")):
        raise RuntimeError("engine service is already running")
    paths["stop"].unlink(missing_ok=True)
    if foreground:
        report = run_service_worker(paths["root"])
        return {"foreground": True, "run": report, "status": service_status(paths["root"])}
    pid = _spawn_worker(paths)
    return {"foreground": False, "pid": pid, "status": service_status(paths["root"])}


def export_service(service_directory: str | Path, output_path: str | Path) -> dict[str, Any]:
    paths = _service_paths(service_directory)
    status = service_status(paths["root"])
    backend_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    candidate_counts_by_lane: Counter[str] = Counter()
    roots: list[str] = []
    succeeded = 0
    if paths["database"].exists():
        connection = sqlite3.connect(paths["database"])
        rows = connection.execute(
            "SELECT lane,result_json FROM work "
            "WHERE state='succeeded' AND result_json IS NOT NULL"
        ).fetchall()
        connection.close()
        for lane, raw in rows:
            result = json.loads(raw)
            backend_counts[str(result.get("backend", result.get("evaluator", "unknown")))] += 1
            status_counts.update(
                {str(key): int(value) for key, value in result.get("counts", {}).items()}
            )
            candidate_count = int(
                result.get("batch", {}).get(
                    "candidate_count", result.get("block", {}).get("record_count", 0)
                )
            )
            candidate_counts_by_lane[str(lane)] += candidate_count
            root = result.get("status_root_sha256")
            if isinstance(root, str):
                roots.append(root)
            succeeded += 1
    report = {
        "schema_version": EXPORT_SCHEMA,
        "service": status,
        "results": {
            "succeeded_work_items": succeeded,
            "backend_counts": dict(backend_counts),
            "processed_candidates": sum(candidate_counts_by_lane.values()),
            "candidate_counts_by_lane": dict(candidate_counts_by_lane),
            "status_counts": dict(status_counts),
            "status_roots_root_sha256": hashlib.sha256(
                "".join(sorted(roots)).encode()
            ).hexdigest(),
        },
        "last_run": _load(paths["last_run"]) if paths["last_run"].exists() else None,
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    _write_json(Path(output_path).resolve(), report)
    return report
