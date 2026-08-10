from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .gravity_engine_service import _hardware_telemetry
from .persistent_parallel_search import PersistentParallelSearch
from .rust_parallel_streaming_search import SCHEMA_VERSION as PARALLEL_SCHEMA_VERSION
from .rust_parallel_streaming_search import (
    ParallelRustRangeScheduler,
    run_parallel_rust_streaming_search,
)
from .rust_streaming_search import (
    ELIGIBILITY,
    PROMOTION_HEADER,
    PROMOTION_MAGIC,
    PROMOTION_RECORD,
    RustStreamingProducer,
    configure_rust_streaming_execution,
    run_rust_streaming_search,
)

SERVICE_SCHEMA = "sigma-rust-streaming-service-1.0"
STATUS_SCHEMA = "sigma-rust-streaming-service-status-1.0"
EXPORT_SCHEMA = "sigma-rust-streaming-promotion-export-1.0"


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
            # SQLite WAL/SHM files can disappear between enumeration and stat.
            continue
    return total


def _paths(service_directory: str | Path) -> dict[str, Path]:
    root = Path(service_directory).resolve()
    return {
        "root": root,
        "service": root / "service.json",
        "execution": root / "execution-config.json",
        "resource": root / "resource-profile.json",
        "stream": root / "stream-config.json",
        "database": root / "stream.sqlite",
        "telemetry": root / "telemetry.jsonl",
        "hardware": root / "hardware.jsonl",
        "stop": root / "stop.request",
        "status": root / "status.json",
        "last_run": root / "last-run.json",
        "log": root / "service.log",
        "chunks": root / "chunks",
        "promotion": root / "promotion",
    }


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _resolve_existing(path_value: str, config_path: Path) -> Path:
    path = Path(path_value)
    candidates = [path.resolve()] if path.is_absolute() else [
        (Path.cwd() / path).resolve(),
        (config_path.parent / path).resolve(),
        (config_path.parent.parent / path).resolve(),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def initialize_service(
    service_directory: str | Path,
    execution_config_path: str | Path,
    resource_profile_path: str | Path,
    stream_config_path: str | Path,
    *,
    maximum_tasks: int | None = None,
) -> dict[str, Any]:
    paths = _paths(service_directory)
    if paths["service"].exists():
        raise FileExistsError("Rust streaming service already exists; use resume")
    execution_path = Path(execution_config_path).resolve()
    resource_path = Path(resource_profile_path).resolve()
    stream_path = Path(stream_config_path).resolve()
    execution = _load(execution_path)
    resource = _load(resource_path)
    stream = _load(stream_path)
    if execution.get("external_paid_llm_calls") is not False:
        raise ValueError("paid LLM calls must remain disabled")
    if stream.get("external_paid_llm_calls") is not False or stream.get(
        "data_eligibility"
    ) != ELIGIBILITY:
        raise ValueError("streaming service eligibility is not fail-closed")
    stream["generator_config_path"] = str(
        _resolve_existing(stream["generator_config_path"], stream_path)
    )
    stream["generator_binary_path"] = str(
        _resolve_existing(stream["generator_binary_path"], stream_path)
    )
    stream["output_directory"] = str(paths["chunks"])
    stream["promotion_directory"] = str(paths["promotion"])
    scheduler_mode = (
        "parallel" if stream.get("schema_version") == PARALLEL_SCHEMA_VERSION else "serial"
    )
    stream_view = (
        {
            **stream,
            "schema_version": "sigma-rust-streaming-search-1.0",
            "producer_lease_seconds": stream["producer_chunk_lease_seconds"],
        }
        if scheduler_mode == "parallel"
        else stream
    )
    configured = configure_rust_streaming_execution(execution, stream_view)
    if maximum_tasks is not None:
        if maximum_tasks <= 0:
            raise ValueError("maximum tasks must be positive")
        configured["budget"]["maximum_tasks"] = min(
            int(configured["budget"]["maximum_tasks"]), int(maximum_tasks)
        )
    maximum_disk = int(stream["maximum_disk_bytes"])
    if maximum_disk <= 0:
        raise ValueError("service disk cap must be positive")
    configured["supervisor"]["maximum_telemetry_bytes"] = min(
        int(configured["supervisor"]["maximum_telemetry_bytes"]),
        max(1, maximum_disk // 16),
    )
    paths["root"].mkdir(parents=True, exist_ok=True)
    _write_json(paths["execution"], configured)
    _write_json(paths["resource"], resource)
    _write_json(paths["stream"], stream)
    coordinator = PersistentParallelSearch(paths["database"], configured, resource)
    # This validates the full ordinal/disk/wall/lease contract before a worker
    # can be launched and persists the restart cursor atomically.
    if scheduler_mode == "parallel":
        scheduler = ParallelRustRangeScheduler(
            coordinator, stream, scheduler_id="service-initializer"
        )
        scheduler.close()
    else:
        RustStreamingProducer(coordinator, stream, owner_id="service-initializer").release_owner()
    identity = {
        "execution_sha256": _sha(configured),
        "resource_sha256": _sha(resource),
        "stream_sha256": _sha(stream),
        "maximum_disk_bytes": maximum_disk,
    }
    now = datetime.now(UTC).isoformat()
    service = {
        "schema_version": SERVICE_SCHEMA,
        "service_id": f"SGRS-{_sha(identity)[:24]}",
        "scheduler_mode": scheduler_mode,
        "identity": identity,
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
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    _write_json(paths["service"], service)
    return service


def _update(paths: dict[str, Path], **changes: Any) -> dict[str, Any]:
    service = _load(paths["service"])
    service.update(changes)
    service["updated_utc"] = datetime.now(UTC).isoformat()
    _write_json(paths["service"], service)
    return service


def _source(database: Path, scheduler_mode: str = "serial") -> dict[str, Any] | None:
    if not database.exists():
        return None
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        table = "rust_parallel_source" if scheduler_mode == "parallel" else "rust_stream_source"
        row = connection.execute(f"SELECT * FROM {table} ORDER BY source_id LIMIT 1").fetchone()
        count_rows = (
            connection.execute(
                "SELECT state,COUNT(*) FROM rust_parallel_chunks WHERE source_id=? GROUP BY state",
                (row["source_id"],),
            ).fetchall()
            if scheduler_mode == "parallel" and row is not None
            else []
        )
    except sqlite3.Error:
        return None
    finally:
        if "connection" in locals():
            connection.close()
    if row is None:
        return None
    value = dict(row)
    if scheduler_mode == "parallel":
        counts = {state: int(count) for state, count in count_rows}
        value["chunk_counts"] = counts
        value["exhausted"] = counts.get("enqueued", 0) == sum(counts.values())
    else:
        value["exhausted"] = int(value["next_ordinal"]) >= int(value["stop_ordinal"])
    return value


def service_status(service_directory: str | Path) -> dict[str, Any]:
    paths = _paths(service_directory)
    service = _load(paths["service"])
    scheduler_mode = service.get("scheduler_mode", "serial")
    execution = _load(paths["execution"])
    resource = _load(paths["resource"])
    telemetry = None
    if paths["database"].exists():
        telemetry = PersistentParallelSearch(paths["database"], execution, resource).telemetry()
    status = {
        "schema_version": STATUS_SCHEMA,
        "service_id": service["service_id"],
        "scheduler_mode": scheduler_mode,
        "state": service["state"],
        "pid": service.get("pid"),
        "alive": _pid_alive(service.get("pid")),
        "stop_requested": paths["stop"].exists(),
        "last_stop_reason": service.get("last_stop_reason"),
        "source": _source(paths["database"], scheduler_mode),
        "execution": telemetry,
        "hardware": _hardware_telemetry(),
        "disk": {
            "used_bytes": _directory_bytes(paths["root"]),
            "maximum_bytes": int(service["identity"]["maximum_disk_bytes"]),
        },
        "cost_budget": service["cost_budget"],
        "data_eligibility": service["data_eligibility"],
    }
    _write_json(paths["status"], status)
    return status


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
        "semantics": "periodic physical sensor samples, distinct from SQLite lease occupancy",
        "sample_count": len(samples),
        "gpu_available_samples": len(gpu),
        "gpu_utilization_percent": stats([float(item["utilization_percent"]) for item in gpu]),
        "gpu_memory_controller_percent": stats(
            [float(item["memory_controller_utilization_percent"]) for item in gpu]
        ),
        "gpu_memory_used_bytes": stats([float(item["memory_used_bytes"]) for item in gpu]),
        "gpu_power_watts": stats(
            [float(item["power_watts"]) for item in gpu if item.get("power_watts") is not None]
        ),
        "cpu_available_samples": len(cpu),
        "cpu_utilization_percent": stats([float(item["utilization_percent"]) for item in cpu]),
    }


def run_service_worker(service_directory: str | Path) -> dict[str, Any]:
    paths = _paths(service_directory)
    service = _load(paths["service"])
    execution = _load(paths["execution"])
    resource = _load(paths["resource"])
    stream = _load(paths["stream"])
    scheduler_mode = service.get(
        "scheduler_mode",
        "parallel" if stream.get("schema_version") == PARALLEL_SCHEMA_VERSION else "serial",
    )
    expected_identity = {
        "execution_sha256": _sha(execution),
        "resource_sha256": _sha(resource),
        "stream_sha256": _sha(stream),
        "maximum_disk_bytes": int(service["identity"]["maximum_disk_bytes"]),
    }
    if service["identity"] != expected_identity:
        raise ValueError("stored streaming service identity mismatch")
    if service["data_eligibility"] != {
        **ELIGIBILITY,
        "paid_llm_calls": False,
        "passed": True,
    } or service["cost_budget"]["maximum_paid_llm_spend_usd"] != 0.0:
        raise ValueError("stored service is not fail-closed")
    _update(paths, state="running", pid=os.getpid(), last_stop_reason=None)
    samples: list[dict[str, Any]] = []

    def disk_reason() -> str | None:
        return (
            "disk_budget_exhausted"
            if _directory_bytes(paths["root"]) >= int(service["identity"]["maximum_disk_bytes"])
            else None
        )

    def periodic(value: dict[str, Any]) -> None:
        hardware = value.get("hardware") or _hardware_telemetry()
        samples.append(hardware)
        status = {
            "schema_version": STATUS_SCHEMA,
            "service_id": service["service_id"],
            "scheduler_mode": scheduler_mode,
            "state": "running",
            "pid": os.getpid(),
            "alive": True,
            "stop_requested": paths["stop"].exists(),
            "source": value.get("refill", {}).get("cursor"),
            "execution": value["execution"],
            "processes": value["processes"],
            "hardware": hardware,
            "disk": {
                "used_bytes": _directory_bytes(paths["root"]),
                "maximum_bytes": int(service["identity"]["maximum_disk_bytes"]),
            },
            "cost_budget": service["cost_budget"],
            "data_eligibility": service["data_eligibility"],
        }
        _write_json(paths["status"], status)

    try:
        runner = (
            run_parallel_rust_streaming_search
            if scheduler_mode == "parallel"
            else run_rust_streaming_search
        )
        report = runner(
            paths["database"], execution, resource, stream, paths["telemetry"],
            external_stop_path=paths["stop"], stop_reason_callback=disk_reason,
            status_callback=periodic,
        )
        envelope = {
            "schema_version": "sigma-rust-streaming-service-run-1.0",
            "streaming": report,
            "hardware": _hardware_summary(samples),
        }
        envelope["content_sha256"] = _sha(envelope)
        _write_json(paths["last_run"], envelope)
        completed = report["supervisor"]["stop_reason"] == "queue_drained" and report["cursor"][
            "exhausted"
        ]
        _update(
            paths,
            state="completed" if completed else "stopped",
            pid=None,
            last_stop_reason=report["supervisor"]["stop_reason"],
        )
        service_status(paths["root"])
        return envelope
    except BaseException as error:
        _update(
            paths,
            state="failed",
            pid=None,
            last_stop_reason=f"{type(error).__name__}: {error}",
        )
        raise


def _spawn(paths: dict[str, Path]) -> int:
    command = [
        sys.executable,
        "-m",
        "sigma_theory_compiler.rust_streaming_service",
        "worker",
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
    _update(paths, state="starting", pid=process.pid)
    return process.pid


def start_service(
    service_directory: str | Path,
    execution_config_path: str | Path,
    resource_profile_path: str | Path,
    stream_config_path: str | Path,
    *,
    foreground: bool,
    maximum_tasks: int | None = None,
) -> dict[str, Any]:
    initialize_service(
        service_directory,
        execution_config_path,
        resource_profile_path,
        stream_config_path,
        maximum_tasks=maximum_tasks,
    )
    paths = _paths(service_directory)
    if foreground:
        return {"foreground": True, "run": run_service_worker(paths["root"])}
    return {"foreground": False, "pid": _spawn(paths), "status": service_status(paths["root"])}


def resume_service(service_directory: str | Path, *, foreground: bool) -> dict[str, Any]:
    paths = _paths(service_directory)
    service = _load(paths["service"])
    if _pid_alive(service.get("pid")):
        raise RuntimeError("Rust streaming service is already running")
    stale_pid = service.get("pid")
    if stale_pid and paths["database"].exists():
        # A hard-killed worker cannot execute the producer's finally block.
        # Reclaim only the lease whose structured owner id proves it belonged
        # to that now-dead service PID; unrelated owners remain protected.
        connection = sqlite3.connect(paths["database"])
        if service.get("scheduler_mode", "serial") == "parallel":
            connection.execute(
                "UPDATE rust_parallel_chunks SET state='available',owner_id=NULL,"
                "lease_expires_utc=NULL WHERE state='generating' AND owner_id LIKE ?",
                (f"parallel-{int(stale_pid)}-%",),
            )
        else:
            connection.execute(
                "UPDATE rust_stream_source SET owner_id=NULL,owner_lease_expires_utc=NULL "
                "WHERE owner_id LIKE ?",
                (f"producer-{int(stale_pid)}-%",),
            )
        connection.commit()
        connection.close()
    paths["stop"].unlink(missing_ok=True)
    if foreground:
        return {"foreground": True, "run": run_service_worker(paths["root"])}
    return {"foreground": False, "pid": _spawn(paths), "status": service_status(paths["root"])}


def stop_service(service_directory: str | Path, *, wait_seconds: float = 10.0) -> dict[str, Any]:
    paths = _paths(service_directory)
    service = _load(paths["service"])
    paths["stop"].write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while _pid_alive(service.get("pid")) and time.monotonic() < deadline:
        time.sleep(0.05)
        service = _load(paths["service"])
    return service_status(paths["root"])


def _verify_promotion(path: Path, export: dict[str, Any], payload: dict[str, Any]) -> None:
    if _file_sha(path) != export["sha256"]:
        raise ValueError("promotion survivor block SHA mismatch")
    with path.open("rb") as handle:
        header = handle.read(PROMOTION_HEADER.size)
        magic, version, record_size, start, end, count = PROMOTION_HEADER.unpack(header)
        if (magic, version, record_size) != (PROMOTION_MAGIC, 1, PROMOTION_RECORD.size):
            raise ValueError("invalid promotion survivor header")
        if (start, end, count) != (
            int(payload["start_ordinal"]),
            int(payload["end_ordinal_exclusive"]),
            int(export["record_count"]),
        ):
            raise ValueError("promotion survivor interval mismatch")
        previous = start - 1 if start else -1
        statuses: Counter[int] = Counter()
        for _ in range(count):
            raw = handle.read(PROMOTION_RECORD.size)
            if len(raw) != PROMOTION_RECORD.size:
                raise ValueError("truncated promotion survivor block")
            status, ordinal, term_count, sign_mask, reserved, *term_ids = PROMOTION_RECORD.unpack(raw)
            if (
                status not in (1, 2)
                or reserved
                or not previous < ordinal < end
                or ordinal < start
                or not 1 <= term_count <= 6
                or sign_mask >= 1 << term_count
                or any(value == 0xFFFF for value in term_ids[:term_count])
                or any(value != 0xFFFF for value in term_ids[term_count:])
            ):
                raise ValueError("invalid promotion survivor identity")
            previous = ordinal
            statuses[status] += 1
        if handle.read(1):
            raise ValueError("promotion survivor block has trailing bytes")
    if statuses[1] != int(export["pass_count"]) or statuses[2] != int(
        export["ambiguous_count"]
    ):
        raise ValueError("promotion survivor status accounting mismatch")
    if export["source_block_sha256"] != payload["block_sha256"]:
        raise ValueError("promotion survivor source lineage mismatch")


def export_service(
    service_directory: str | Path,
    output_path: str | Path,
    *,
    maximum_export_bytes: int | None = None,
) -> dict[str, Any]:
    paths = _paths(service_directory)
    output = Path(output_path).resolve()
    artifact_directory = output.parent / f"{output.stem}-survivors"
    service = _load(paths["service"])
    limit = int(maximum_export_bytes or service["identity"]["maximum_disk_bytes"])
    if limit <= 0:
        raise ValueError("export disk cap must be positive")
    rows: list[tuple[str, str]] = []
    if paths["database"].exists():
        connection = sqlite3.connect(paths["database"])
        rows = connection.execute(
            "SELECT payload_json,result_json FROM work WHERE state='succeeded' "
            "AND result_json IS NOT NULL ORDER BY ordinal"
        ).fetchall()
        connection.close()
    blocks: list[dict[str, Any]] = []
    total_bytes = 0
    for payload_raw, result_raw in rows:
        payload, result = json.loads(payload_raw), json.loads(result_raw)
        export = result.get("promotion_survivor_export")
        if not export:
            raise ValueError("successful result lacks promotion survivor identities")
        source = Path(export["path"])
        _verify_promotion(source, export, payload)
        total_bytes += source.stat().st_size
        if total_bytes > limit:
            raise ValueError("promotion export exceeds its disk cap")
        artifact_directory.mkdir(parents=True, exist_ok=True)
        destination = artifact_directory / source.name
        shutil.copyfile(source, destination)
        if _file_sha(destination) != export["sha256"]:
            raise ValueError("copied promotion survivor block SHA mismatch")
        blocks.append(
            {
                **{key: value for key, value in export.items() if key != "path"},
                "file": f"{artifact_directory.name}/{destination.name}",
                "start_ordinal": int(payload["start_ordinal"]),
                "end_ordinal_exclusive": int(payload["end_ordinal_exclusive"]),
                "source_id": payload["source_id"],
                "result_status_root_sha256": result["status_root_sha256"],
            }
        )
    report = {
        "schema_version": EXPORT_SCHEMA,
        "service_id": service["service_id"],
        "source": _source(
            paths["database"],
            service.get("scheduler_mode", "serial"),
        ),
        "blocks": blocks,
        "block_count": len(blocks),
        "survivor_identity_count": sum(int(block["record_count"]) for block in blocks),
        "pass_count": sum(int(block["pass_count"]) for block in blocks),
        "ambiguous_count": sum(int(block["ambiguous_count"]) for block in blocks),
        "artifact_bytes": total_bytes,
        "maximum_export_bytes": limit,
        "blocks_root_sha256": _sha(blocks),
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
        "promotion_contract": (
            "Identities survived only the sampled-static screen and remain sealed from "
            "observational claims until separately hash-bound promotion gates exist."
        ),
    }
    report["content_sha256"] = _sha(report)
    _write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded restart-safe Rust SGSURV2 service")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--service-dir", required=True)
    start.add_argument("--execution-config", required=True)
    start.add_argument("--resource-profile", required=True)
    start.add_argument("--stream-config", required=True)
    start.add_argument("--maximum-tasks", type=int)
    start.add_argument("--foreground", action="store_true")
    for name in ("status", "stop", "resume", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--service-dir", required=True)
        if name == "resume":
            command.add_argument("--foreground", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--service-dir", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--maximum-export-bytes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "start":
        value = start_service(
            args.service_dir,
            args.execution_config,
            args.resource_profile,
            args.stream_config,
            foreground=args.foreground,
            maximum_tasks=args.maximum_tasks,
        )
    elif args.command == "status":
        value = service_status(args.service_dir)
    elif args.command == "stop":
        value = stop_service(args.service_dir)
    elif args.command == "resume":
        value = resume_service(args.service_dir, foreground=args.foreground)
    elif args.command == "worker":
        value = run_service_worker(args.service_dir)
    else:
        value = export_service(
            args.service_dir,
            args.output,
            maximum_export_bytes=args.maximum_export_bytes,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
