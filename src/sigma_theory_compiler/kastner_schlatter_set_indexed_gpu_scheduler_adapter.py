"""Closed durable GPU scheduler adapter for one reviewed Kastner--Schlatter workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
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
        "seals",
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
        "persistent_supervisor_source", "resource_profile",
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
        "execution_state": {
            "runtime_created_by_readiness": False,
            "scheduler_started_by_readiness": False,
            "worker_result_created_by_readiness": False,
        },
        "decision": "durable_single_owner_gpu_adapter_ready_not_started",
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
    if result.get("decision") != "durable_single_owner_gpu_adapter_ready_not_started":
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
    args = parser.parse_args()
    selected = sum((args.write_readiness, bool(args.validate_readiness), args.run))
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
    result = build_readiness(args.config)
    config, root = load_adapter_config(args.config)
    output = _inside(root, config["output_path"])
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
