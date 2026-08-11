from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .process_health import pid_alive
from .scientific_leaderboards import build_scientific_leaderboards, load_leaderboard_config
from .unified_engine_dashboard import render_dashboard
from .unified_engine_status import build_unified_snapshot, load_config

CONFIG_SCHEMA = "sigma-unified-engine-live-service-config-1.0"
CHECKPOINT_SCHEMA = "sigma-unified-engine-live-service-checkpoint-1.0"
READINESS_SCHEMA = "sigma-unified-engine-live-service-readiness-1.0"
EXPECTED_SEALS = {
    "observations_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "paid_llm_calls": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve_inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("live-service paths must be portable relative paths")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("live-service path escapes the project root") from error
    return resolved


def load_live_config(root: Path, path: Path) -> dict[str, Any]:
    config = _load_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported live-service config schema")
    if config.get("enabled") is not True:
        raise ValueError("live-service config is checked disabled")
    for key in (
        "refresh_interval_seconds",
        "maximum_refreshes",
        "maximum_consecutive_failures",
        "maximum_output_bytes",
    ):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"invalid positive live-service bound: {key}")
    if config["refresh_interval_seconds"] < 10:
        raise ValueError("live-service refresh interval is too aggressive")
    if config.get("data_seals") != EXPECTED_SEALS:
        raise ValueError("live-service data seals are not fail-closed")
    for key in (
        "unified_config_path",
        "leaderboard_config_path",
        "runtime_directory",
    ):
        if not isinstance(config.get(key), str):
            raise TypeError(f"missing portable live-service path: {key}")
        _resolve_inside(root, config[key])
    return config


def _paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    runtime = _resolve_inside(root, str(config["runtime_directory"]))
    return {
        "runtime": runtime,
        "checkpoint": runtime / "checkpoint.json",
        "snapshot": runtime / "unified-engine-status-live.json",
        "dashboard": runtime / "dashboard.html",
        "stop": runtime / "stop.request",
        "log": runtime / "service.log",
    }


def _atomic_write(path: Path, payload: bytes, maximum_bytes: int) -> None:
    if len(payload) > maximum_bytes:
        raise RuntimeError("bounded live-service artifact exceeds maximum bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checkpoint_payload(value: Mapping[str, Any]) -> bytes:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    body["content_sha256"] = _sha(body)
    return (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()


def _validate_checkpoint(value: Mapping[str, Any], config_file_sha256: str) -> None:
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid live-service checkpoint schema")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _sha(body):
        raise ValueError("live-service checkpoint content hash mismatch")
    if value.get("config_file_sha256") != config_file_sha256:
        raise ValueError("live-service checkpoint config binding mismatch")
    for key in ("refresh_count", "consecutive_failures"):
        if not isinstance(value.get(key), int) or value[key] < 0:
            raise ValueError("invalid live-service checkpoint counter")


def refresh_once(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(root, config)
    unified_path = _resolve_inside(root, str(config["unified_config_path"]))
    leaderboard_path = _resolve_inside(root, str(config["leaderboard_config_path"]))
    snapshot = build_unified_snapshot(root, load_config(unified_path))
    previous = None
    if paths["snapshot"].is_file():
        prior = _load_json(paths["snapshot"])
        previous = prior.get("core", {}).get("scientific_leaderboards")
    snapshot["core"]["scientific_leaderboards"] = build_scientific_leaderboards(
        root, load_leaderboard_config(leaderboard_path), previous
    )
    snapshot["core_content_sha256"] = _sha(snapshot["core"])
    if snapshot["core"].get("data_seals") != {
        "dark_matter_or_halo_inputs": False,
        "observations_opened": False,
        "paid_llm_in_streaming_promotion_grammar": False,
        "redshift_distance_inputs": False,
    }:
        raise ValueError("live snapshot opened a protected data seal")
    json_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    html_bytes = render_dashboard(snapshot).encode()
    maximum = int(config["maximum_output_bytes"])
    _atomic_write(paths["snapshot"], json_bytes, maximum)
    _atomic_write(paths["dashboard"], html_bytes, maximum)
    return {
        "core_content_sha256": snapshot["core_content_sha256"],
        "snapshot_file_sha256": hashlib.sha256(json_bytes).hexdigest(),
        "dashboard_file_sha256": hashlib.sha256(html_bytes).hexdigest(),
        "json_bytes": len(json_bytes),
        "dashboard_bytes": len(html_bytes),
        "sampled_at_utc": snapshot["volatile"]["sampled_at_utc"],
    }


def run_worker(root: Path, config_path: Path) -> int:
    config = load_live_config(root, config_path)
    paths = _paths(root, config)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    config_sha = _file_sha(config_path)
    if paths["checkpoint"].is_file():
        checkpoint = _load_json(paths["checkpoint"])
        _validate_checkpoint(checkpoint, config_sha)
    else:
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "config_file_sha256": config_sha,
            "state": "starting",
            "pid": os.getpid(),
            "refresh_count": 0,
            "consecutive_failures": 0,
            "last_error": None,
            "last_refresh": None,
            "stop_reason": None,
            "updated_utc": datetime.now(UTC).isoformat(),
        }
    checkpoint.update(state="running", pid=os.getpid(), stop_reason=None)
    _atomic_write(
        paths["checkpoint"],
        _checkpoint_payload(checkpoint),
        int(config["maximum_output_bytes"]),
    )
    stop_reason = "maximum_refreshes_reached"
    while checkpoint["refresh_count"] < int(config["maximum_refreshes"]):
        if paths["stop"].exists():
            stop_reason = "external_stop_requested"
            break
        try:
            result = refresh_once(root, config)
            checkpoint["refresh_count"] += 1
            checkpoint["consecutive_failures"] = 0
            checkpoint["last_error"] = None
            checkpoint["last_refresh"] = result
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            checkpoint["consecutive_failures"] += 1
            checkpoint["last_error"] = f"{type(error).__name__}: {error}"
            if checkpoint["consecutive_failures"] >= int(config["maximum_consecutive_failures"]):
                stop_reason = "consecutive_failure_limit_reached"
                break
        checkpoint["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_write(
            paths["checkpoint"],
            _checkpoint_payload(checkpoint),
            int(config["maximum_output_bytes"]),
        )
        if checkpoint["refresh_count"] >= int(config["maximum_refreshes"]):
            break
        time.sleep(int(config["refresh_interval_seconds"]))
    checkpoint.update(
        state="stopped",
        pid=None,
        stop_reason=stop_reason,
        updated_utc=datetime.now(UTC).isoformat(),
    )
    _atomic_write(
        paths["checkpoint"],
        _checkpoint_payload(checkpoint),
        int(config["maximum_output_bytes"]),
    )
    return 0


def service_status(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_live_config(root, config_path)
    paths = _paths(root, config)
    if not paths["checkpoint"].is_file():
        return {"state": "not_started", "alive": False, "runtime": str(paths["runtime"])}
    checkpoint = _load_json(paths["checkpoint"])
    _validate_checkpoint(checkpoint, _file_sha(config_path))
    return {
        **checkpoint,
        "alive": pid_alive(checkpoint.get("pid")),
        "runtime": str(paths["runtime"]),
    }


def start_service(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_live_config(root, config_path)
    paths = _paths(root, config)
    if paths["checkpoint"].is_file():
        status = service_status(root, config_path)
        if status["alive"]:
            return status
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    paths["stop"].unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "sigma_theory_compiler.unified_engine_live_service",
        "worker",
        "--project-root",
        str(root),
        "--config",
        str(config_path.relative_to(root)),
    ]
    log = paths["log"].open("ab")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | 0x08000000
        )
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, cwd=root, **kwargs)
    log.close()
    return {"state": "starting", "alive": True, "pid": process.pid}


def stop_service(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_live_config(root, config_path)
    paths = _paths(root, config)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    _atomic_write(paths["stop"], b"stop\n", 1024)
    return service_status(root, config_path)


def build_readiness(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_live_config(root, config_path)
    source_path = Path(__file__).resolve()
    body = {
        "schema_version": READINESS_SCHEMA,
        "decision": "ready_enabled_read_only_bounded",
        "config_file_sha256": _file_sha(config_path),
        "source_file_sha256": _file_sha(source_path),
        "refresh_interval_seconds": config["refresh_interval_seconds"],
        "maximum_refreshes": config["maximum_refreshes"],
        "maximum_consecutive_failures": config["maximum_consecutive_failures"],
        "maximum_output_bytes": config["maximum_output_bytes"],
        "lifecycle_commands": ["start", "status", "stop", "refresh-once", "worker"],
        "atomic_snapshot_write": True,
        "atomic_dashboard_write": True,
        "hash_bound_checkpoint": True,
        "reads_live_campaign_database": True,
        "writes_live_campaign_database": False,
        "immutable_snapshot_overwritten": False,
        "data_seals": config["data_seals"],
    }
    return {**body, "content_sha256": _sha(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded read-only unified dashboard service")
    parser.add_argument(
        "command",
        choices=("start", "status", "stop", "refresh-once", "readiness", "worker"),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/unified_engine_live_service.json")
    parser.add_argument(
        "--output", default="runs/engine/unified-engine-live-service-readiness.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(arguments.project_root).resolve()
    config_path = _resolve_inside(root, arguments.config)
    if arguments.command == "worker":
        return run_worker(root, config_path)
    if arguments.command == "start":
        result = start_service(root, config_path)
    elif arguments.command == "stop":
        result = stop_service(root, config_path)
    elif arguments.command == "refresh-once":
        result = refresh_once(root, load_live_config(root, config_path))
    elif arguments.command == "readiness":
        result = build_readiness(root, config_path)
        output_path = _resolve_inside(root, arguments.output)
        _atomic_write(
            output_path,
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
            int(load_live_config(root, config_path)["maximum_output_bytes"]),
        )
    else:
        result = service_status(root, config_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
