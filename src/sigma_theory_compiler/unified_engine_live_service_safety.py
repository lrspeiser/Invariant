"""Hardened, epoch-bound supervisor for unified live-dashboard refreshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .process_health import pid_alive
from .scientific_leaderboards import build_scientific_leaderboards, load_leaderboard_config
from .unified_engine_dashboard import render_dashboard
from .unified_engine_status import build_unified_snapshot, load_config

CONFIG_SCHEMA = "sigma-unified-engine-live-service-safety-config-1.0"
CHECKPOINT_SCHEMA = "sigma-unified-engine-live-service-safety-checkpoint-1.0"
LEASE_SCHEMA = "sigma-unified-engine-live-service-cutover-lease-1.0"
ARTIFACT_SCHEMA = "sigma-unified-engine-live-service-safety-readiness-1.0"
EXPECTED_RUNTIME_DIRECTORY = "runs/engine/unified-live-dashboard-safety-service"
EXPECTED_LEGACY_RUNTIME_DIRECTORY = "runs/engine/unified-live-dashboard-service"
EXPECTED_SEALS = {
    "observations_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "paid_llm_calls": False,
}


class ControlSignal(RuntimeError):
    """A supervised stop or configuration-change signal."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _resolve_inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("safety-service paths must be portable relative paths")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("safety-service path escapes project root") from error
    return resolved


def _load_bound(root: Path, binding: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    path = _resolve_inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{name} predecessor file hash mismatch")
    if path.suffix != ".json":
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_content = binding.get("content_sha256")
    if expected_content is not None and (
        value.get("content_sha256") != expected_content or _content_sha(value) != expected_content
    ):
        raise ValueError(f"{name} predecessor content hash mismatch")
    return value


def load_safety_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "enabled",
        "runtime_directory",
        "runtime_epoch",
        "legacy_runtime_directory",
        "cutover_lease_path",
        "startup_identity_timeout_seconds",
        "refresh_interval_seconds",
        "control_poll_interval_seconds",
        "maximum_refreshes",
        "maximum_consecutive_failures",
        "maximum_output_bytes",
        "unified_config_path",
        "leaderboard_config_path",
        "output_path",
        "predecessors",
        "data_seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported safety-service config")
    if config.get("enabled") is not True or config.get("data_seals") != EXPECTED_SEALS:
        raise ValueError("safety-service disabled or data seals opened")
    if config.get("runtime_directory") != EXPECTED_RUNTIME_DIRECTORY:
        raise ValueError("runtime_directory epoch drift rejected")
    if config.get("legacy_runtime_directory") != EXPECTED_LEGACY_RUNTIME_DIRECTORY:
        raise ValueError("legacy runtime_directory drift rejected")
    if not isinstance(config.get("runtime_epoch"), str) or len(config["runtime_epoch"]) < 16:
        raise ValueError("invalid runtime epoch")
    for key in (
        "refresh_interval_seconds",
        "control_poll_interval_seconds",
        "maximum_refreshes",
        "maximum_consecutive_failures",
        "maximum_output_bytes",
        "startup_identity_timeout_seconds",
    ):
        if not isinstance(config.get(key), (int, float)) or config[key] <= 0:
            raise ValueError(f"invalid safety-service bound: {key}")
    if config["control_poll_interval_seconds"] > 1.0:
        raise ValueError("control polling latency exceeds one second")
    for key in (
        "runtime_directory",
        "legacy_runtime_directory",
        "cutover_lease_path",
        "unified_config_path",
        "leaderboard_config_path",
        "output_path",
    ):
        _resolve_inside(root, str(config[key]))
    expected_predecessors = {"service_source", "service_config", "service_test", "readiness"}
    if set(config.get("predecessors", {})) != expected_predecessors:
        raise ValueError("safety-service predecessor set changed")
    for name, binding in config["predecessors"].items():
        _load_bound(root, binding, name)
    return config


def _paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    runtime = _resolve_inside(root, str(config["runtime_directory"]))
    legacy_runtime = _resolve_inside(root, str(config["legacy_runtime_directory"]))
    lease = _resolve_inside(root, str(config["cutover_lease_path"]))
    return {
        "runtime": runtime,
        "checkpoint": runtime / "checkpoint.json",
        "snapshot": runtime / "unified-engine-status-live.json",
        "dashboard": runtime / "dashboard.html",
        "stop": runtime / "stop.request",
        "log": runtime / "service.log",
        "lease": lease,
        "lease_recovery": lease.with_name(f"{lease.name}.recovery"),
        "legacy_checkpoint": legacy_runtime / "checkpoint.json",
    }


def _safe_atomic_write(path: Path, payload: bytes, maximum_bytes: int) -> None:
    if len(payload) > maximum_bytes:
        raise RuntimeError("bounded safety-service artifact exceeds maximum bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("safety-service target symlink rejected")
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


def _checkpoint_payload(value: Mapping[str, Any]) -> bytes:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    body["content_sha256"] = _sha(body)
    return (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()


def _lease_payload(value: Mapping[str, Any]) -> bytes:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    body["content_sha256"] = _sha(body)
    return (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()


def _validate_lease(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if value.get("schema_version") != LEASE_SCHEMA:
        raise ValueError("invalid cutover lease schema")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _sha(body):
        raise ValueError("cutover lease content hash mismatch")
    if value.get("runtime_epoch") != config["runtime_epoch"]:
        raise ValueError("cutover lease epoch mismatch")
    if value.get("runtime_directory") != config["runtime_directory"]:
        raise ValueError("cutover lease runtime drift")
    if value.get("owner_kind") not in {"starter", "worker"}:
        raise ValueError("invalid cutover lease owner kind")
    if not isinstance(value.get("owner_pid"), int) or value["owner_pid"] <= 0:
        raise ValueError("invalid cutover lease owner PID")
    for key in ("owner_argv_sha256", "worker_argv_sha256"):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 64:
            raise ValueError("invalid cutover lease argv identity")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("cutover lease symlink rejected")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_checkpoint(value: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid safety-service checkpoint schema")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _sha(body):
        raise ValueError("safety-service checkpoint content hash mismatch")
    if value.get("runtime_epoch") != config["runtime_epoch"]:
        raise ValueError("safety-service checkpoint epoch mismatch")
    if value.get("runtime_directory") != config["runtime_directory"]:
        raise ValueError("safety-service checkpoint runtime drift")
    for key in ("refresh_count", "consecutive_failures", "reload_count"):
        if not isinstance(value.get(key), int) or value[key] < 0:
            raise ValueError("invalid safety-service checkpoint counter")


def _worker_tail(root: Path, config_path: Path) -> list[str]:
    return [
        "-m",
        "sigma_theory_compiler.unified_engine_live_service_safety",
        "worker",
        "--project-root",
        str(root.resolve()),
        "--config",
        config_path.resolve().relative_to(root.resolve()).as_posix(),
    ]


def _worker_command(root: Path, config_path: Path) -> list[str]:
    return [sys.executable, *_worker_tail(root, config_path)]


def _argv_identity(argv: list[str]) -> str:
    try:
        module_index = argv.index("-m")
    except ValueError:
        return ""
    return _sha(argv[module_index:])


def _exact_argv_identity(argv: list[str]) -> str:
    return _sha(argv)


def _process_identity_state(
    pid: int | None, expected_identity: str, *, normalized_worker: bool
) -> str:
    if not pid_alive(pid):
        return "dead"
    try:
        import psutil

        command = psutil.Process(int(pid)).cmdline()
    except (ImportError, OSError, ValueError):
        return "unverifiable"
    except psutil.NoSuchProcess:
        return "dead"
    except psutil.Error:
        return "unverifiable"
    actual = _argv_identity(command) if normalized_worker else _exact_argv_identity(command)
    return "match" if actual == expected_identity else "mismatch"


def _pid_matches_worker(pid: int | None, expected_identity: str) -> bool:
    return (
        _process_identity_state(pid, expected_identity, normalized_worker=True)
        == "match"
    )


def _current_exact_argv_identity() -> str:
    try:
        import psutil

        command = psutil.Process(os.getpid()).cmdline()
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("starter PID/argv identity unavailable") from error
    except psutil.Error as error:
        raise RuntimeError("starter PID/argv identity unavailable") from error
    return _exact_argv_identity(command)


def _worker_pids(expected_identity: str) -> list[int]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("worker process inventory unavailable") from error
    matches: list[int] = []
    try:
        processes = psutil.process_iter(["pid", "name", "cmdline"])
        for process in processes:
            try:
                info = process.info
                command = info.get("cmdline")
                if not command:
                    continue
                if _argv_identity(list(command)) == expected_identity:
                    matches.append(int(info["pid"]))
            except psutil.NoSuchProcess:
                continue
            except psutil.Error as error:
                raise RuntimeError("worker process inventory unavailable") from error
    except psutil.Error as error:
        raise RuntimeError("worker process inventory unavailable") from error
    return sorted(set(matches))


def _legacy_worker_command(root: Path, config: Mapping[str, Any]) -> list[str]:
    legacy_config = _resolve_inside(
        root, str(config["predecessors"]["service_config"]["path"])
    )
    return [
        sys.executable,
        "-m",
        "sigma_theory_compiler.unified_engine_live_service",
        "worker",
        "--project-root",
        str(root.resolve()),
        "--config",
        os.fspath(legacy_config.relative_to(root.resolve())),
    ]


def _assert_no_legacy_worker(root: Path, config: Mapping[str, Any]) -> None:
    identity = _argv_identity(_legacy_worker_command(root, config))
    paths = _paths(root, config)
    matches = set(_worker_pids(identity))
    if paths["legacy_checkpoint"].is_file():
        raw = paths["legacy_checkpoint"].read_bytes()
        checkpoint = json.loads(raw)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("legacy checkpoint is not an object")
        body = {
            key: item for key, item in checkpoint.items() if key != "content_sha256"
        }
        legacy_content_sha = hashlib.sha256(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        legacy_config = _resolve_inside(
            root, str(config["predecessors"]["service_config"]["path"])
        )
        if (
            checkpoint.get("schema_version")
            != "sigma-unified-engine-live-service-checkpoint-1.0"
            or checkpoint.get("content_sha256") != legacy_content_sha
            or checkpoint.get("config_file_sha256") != _file_sha(legacy_config)
        ):
            raise RuntimeError("legacy checkpoint validation failed")
        pid = checkpoint.get("pid")
        if checkpoint.get("state") in {"starting", "running"} and pid is not None:
            state = _process_identity_state(pid, identity, normalized_worker=True)
            if state == "match":
                matches.add(int(pid))
            elif state == "unverifiable":
                raise RuntimeError("legacy worker PID/argv identity unavailable")
    if matches:
        raise RuntimeError(
            f"legacy unified live-service worker present: {sorted(matches)}"
        )


def _lease_owner_state(lease: Mapping[str, Any]) -> str:
    return _process_identity_state(
        lease.get("owner_pid"),
        str(lease.get("owner_argv_sha256", "")),
        normalized_worker=lease.get("owner_kind") == "worker",
    )


def _read_lease(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("cutover lease is not an object")
    _validate_lease(value, config)
    return value


def _recover_stale_lease(paths: Mapping[str, Path], config: Mapping[str, Any]) -> None:
    recovery = paths["lease_recovery"]
    try:
        _exclusive_write(recovery, b"recover\n")
    except FileExistsError as error:
        raise RuntimeError("cutover lease recovery already in progress") from error
    try:
        lease = _read_lease(paths["lease"], config)
        state = _lease_owner_state(lease)
        if state in {"match", "unverifiable"}:
            raise RuntimeError(f"cutover lease owner is {state}")
        paths["lease"].unlink()
    finally:
        recovery.unlink(missing_ok=True)


def _acquire_start_lease(
    paths: Mapping[str, Path], config: Mapping[str, Any], worker_identity: str
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    body = {
        "schema_version": LEASE_SCHEMA,
        "runtime_epoch": config["runtime_epoch"],
        "runtime_directory": config["runtime_directory"],
        "worker_argv_sha256": worker_identity,
        "owner_kind": "starter",
        "owner_pid": os.getpid(),
        "owner_argv_sha256": _current_exact_argv_identity(),
        "created_utc": now,
        "updated_utc": now,
    }
    lease = json.loads(_lease_payload(body))
    try:
        _exclusive_write(paths["lease"], _lease_payload(lease))
        return lease
    except FileExistsError:
        existing = _read_lease(paths["lease"], config)
        state = _lease_owner_state(existing)
        if state in {"match", "unverifiable"}:
            raise RuntimeError(f"cutover lease already active: {state}")
        _recover_stale_lease(paths, config)
        try:
            _exclusive_write(paths["lease"], _lease_payload(lease))
        except FileExistsError as error:
            raise RuntimeError("cutover lease reacquisition lost") from error
        return lease


def _transfer_lease_to_worker(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    starter_lease: Mapping[str, Any],
    worker_pid: int,
    worker_identity: str,
) -> dict[str, Any]:
    current = _read_lease(paths["lease"], config)
    if (
        current.get("owner_kind") != "starter"
        or current.get("owner_pid") != os.getpid()
        or current.get("owner_argv_sha256") != starter_lease.get("owner_argv_sha256")
        or current.get("worker_argv_sha256") != worker_identity
    ):
        raise RuntimeError("starter no longer owns cutover lease")
    current.update(
        owner_kind="worker",
        owner_pid=worker_pid,
        owner_argv_sha256=worker_identity,
        updated_utc=datetime.now(UTC).isoformat(),
    )
    _safe_atomic_write(paths["lease"], _lease_payload(current), 16384)
    return json.loads(paths["lease"].read_text(encoding="utf-8"))


def _claim_worker_lease(
    paths: Mapping[str, Path], config: Mapping[str, Any], worker_identity: str
) -> None:
    current = _read_lease(paths["lease"], config)
    if current.get("worker_argv_sha256") != worker_identity:
        raise RuntimeError("cutover lease worker identity drift")
    if (
        current.get("owner_kind") == "worker"
        and current.get("owner_pid") == os.getpid()
        and current.get("owner_argv_sha256") == worker_identity
    ):
        return
    if current.get("owner_kind") != "starter":
        raise RuntimeError("cutover lease owned by another worker")
    if _pid_matches_worker(os.getpid(), worker_identity) is not True:
        raise RuntimeError("worker self PID/argv identity mismatch")
    current.update(
        owner_kind="worker",
        owner_pid=os.getpid(),
        owner_argv_sha256=worker_identity,
        updated_utc=datetime.now(UTC).isoformat(),
    )
    _safe_atomic_write(paths["lease"], _lease_payload(current), 16384)


def _release_owned_lease(paths: Mapping[str, Path], worker_identity: str) -> None:
    if not paths["lease"].is_file():
        return
    try:
        lease = json.loads(paths["lease"].read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return
    if (
        lease.get("owner_kind") == "worker"
        and lease.get("owner_pid") == os.getpid()
        and lease.get("owner_argv_sha256") == worker_identity
    ):
        paths["lease"].unlink(missing_ok=True)


def _wait_for_worker_identity(
    pid: int, worker_identity: str, timeout_seconds: float, poll_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = _process_identity_state(pid, worker_identity, normalized_worker=True)
        if state == "match":
            return
        if state in {"dead", "mismatch"}:
            raise RuntimeError(f"spawned worker PID/argv identity {state}")
        if time.monotonic() >= deadline:
            raise RuntimeError("spawned worker PID/argv identity unavailable at timeout")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _dependency_manifest(
    root: Path, unified_path: Path, leaderboard_path: Path
) -> dict[str, Any]:
    unified = json.loads(unified_path.read_text(encoding="utf-8"))
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    files = {
        unified_path.relative_to(root).as_posix(): _file_sha(unified_path),
        leaderboard_path.relative_to(root).as_posix(): _file_sha(leaderboard_path),
    }
    for spec in unified.get("sources", []):
        path = _resolve_inside(root, str(spec["path"]))
        actual = _file_sha(path)
        if actual != spec["file_sha256"]:
            raise ValueError(f"unified projection source drift: {spec['label']}")
        files[path.relative_to(root).as_posix()] = actual
    for label, spec in leaderboard.get("sources", {}).items():
        path = _resolve_inside(root, str(spec["path"]))
        actual = _file_sha(path)
        if actual != spec["file_sha256"]:
            raise ValueError(f"leaderboard projection source drift: {label}")
        files[path.relative_to(root).as_posix()] = actual
    ordered = dict(sorted(files.items()))
    return {"file_sha256": ordered, "manifest_sha256": _sha(ordered)}


def safe_refresh_once(
    root: Path,
    config: Mapping[str, Any],
    guard: Callable[[], None],
    *,
    snapshot_builder: Callable[..., dict[str, Any]] = build_unified_snapshot,
    leaderboard_builder: Callable[..., dict[str, Any]] = build_scientific_leaderboards,
    dashboard_renderer: Callable[[Mapping[str, Any]], str] = render_dashboard,
) -> dict[str, Any]:
    paths = _paths(root, config)
    unified_path = _resolve_inside(root, str(config["unified_config_path"]))
    leaderboard_path = _resolve_inside(root, str(config["leaderboard_config_path"]))
    manifest_before = _dependency_manifest(root, unified_path, leaderboard_path)
    guard()
    snapshot = snapshot_builder(root, load_config(unified_path))
    guard()
    previous = None
    if paths["snapshot"].is_file():
        previous = json.loads(paths["snapshot"].read_text(encoding="utf-8")).get("core", {}).get(
            "scientific_leaderboards"
        )
    snapshot["core"]["scientific_leaderboards"] = leaderboard_builder(
        root, load_leaderboard_config(leaderboard_path), previous
    )
    snapshot["core_content_sha256"] = _sha(snapshot["core"])
    snapshot["live_projection_input_manifest_sha256"] = manifest_before["manifest_sha256"]
    guard()
    json_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    html_bytes = dashboard_renderer(snapshot).encode()
    manifest_after = _dependency_manifest(root, unified_path, leaderboard_path)
    if manifest_after != manifest_before:
        raise ControlSignal("projection_inputs_changed_during_refresh")
    guard()
    maximum = int(config["maximum_output_bytes"])
    _safe_atomic_write(paths["snapshot"], json_bytes, maximum)
    _safe_atomic_write(paths["dashboard"], html_bytes, maximum)
    return {
        "core_content_sha256": snapshot["core_content_sha256"],
        "projection_input_manifest_sha256": manifest_before["manifest_sha256"],
        "snapshot_file_sha256": hashlib.sha256(json_bytes).hexdigest(),
        "dashboard_file_sha256": hashlib.sha256(html_bytes).hexdigest(),
    }


def _control_guard(paths: Mapping[str, Path], config_path: Path, config_sha: str) -> None:
    if paths["stop"].exists():
        raise ControlSignal("external_stop_requested")
    try:
        current = _file_sha(config_path)
    except OSError as error:
        raise ControlSignal(f"configuration_reload_error:{type(error).__name__}") from error
    if current != config_sha:
        raise ControlSignal("configuration_changed")


def _interruptible_wait(
    seconds: float,
    poll_seconds: float,
    guard: Callable[[], None],
) -> None:
    deadline = time.monotonic() + seconds
    while True:
        guard()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(poll_seconds, remaining))


def _reload_preserving_epoch(
    root: Path,
    config_path: Path,
    current: Mapping[str, Any],
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    reloaded = load_safety_config(root, config_path)
    if (
        reloaded["runtime_epoch"] != current["runtime_epoch"]
        or reloaded["runtime_directory"] != current["runtime_directory"]
    ):
        raise ValueError("runtime epoch or directory drift rejected")
    config_sha = _file_sha(config_path)
    checkpoint["config_file_sha256"] = config_sha
    checkpoint["reload_count"] += 1
    checkpoint["last_reload_utc"] = datetime.now(UTC).isoformat()
    return reloaded, config_sha


def _starting_checkpoint(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
    config_sha: str,
    worker_identity: str,
) -> dict[str, Any]:
    if paths["checkpoint"].is_file():
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        _validate_checkpoint(checkpoint, config)
    else:
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA,
            "runtime_epoch": config["runtime_epoch"],
            "runtime_directory": config["runtime_directory"],
            "refresh_count": 0,
            "consecutive_failures": 0,
            "reload_count": 0,
            "last_reload_utc": None,
            "last_refresh": None,
        }
    checkpoint.update(
        config_file_sha256=config_sha,
        worker_argv_sha256=worker_identity,
        state="starting",
        pid=None,
        last_error=None,
        stop_reason=None,
        updated_utc=datetime.now(UTC).isoformat(),
    )
    _safe_atomic_write(
        paths["checkpoint"],
        _checkpoint_payload(checkpoint),
        int(config["maximum_output_bytes"]),
    )
    return checkpoint


def run_worker(root: Path, config_path: Path) -> int:
    config = load_safety_config(root, config_path)
    paths = _paths(root, config)
    config_sha = _file_sha(config_path)
    identity = _argv_identity(_worker_command(root, config_path))
    if not paths["checkpoint"].is_file():
        raise RuntimeError("atomic starting checkpoint missing")
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    _validate_checkpoint(checkpoint, config)
    stop_reason = "maximum_refreshes_reached"
    lease_owned = False
    try:
        _claim_worker_lease(paths, config, identity)
        lease_owned = True
        _assert_no_legacy_worker(root, config)
        checkpoint.update(
            state="running",
            pid=os.getpid(),
            worker_argv_sha256=identity,
            config_file_sha256=config_sha,
            last_error=None,
            stop_reason=None,
            updated_utc=datetime.now(UTC).isoformat(),
        )
        _safe_atomic_write(
            paths["checkpoint"],
            _checkpoint_payload(checkpoint),
            int(config["maximum_output_bytes"]),
        )
        while checkpoint["refresh_count"] < int(config["maximum_refreshes"]):
            guard = lambda active_sha=config_sha: _control_guard(
                paths, config_path, active_sha
            )
            try:
                checkpoint["last_refresh"] = safe_refresh_once(root, config, guard)
                checkpoint["refresh_count"] += 1
                checkpoint["consecutive_failures"] = 0
                checkpoint["last_error"] = None
            except ControlSignal as signal:
                if signal.reason == "configuration_changed":
                    try:
                        config, config_sha = _reload_preserving_epoch(
                            root, config_path, config, checkpoint
                        )
                        continue
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        checkpoint["last_error"] = f"{type(error).__name__}: {error}"
                        stop_reason = "configuration_reload_failed"
                        break
                stop_reason = signal.reason
                break
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                checkpoint["consecutive_failures"] += 1
                checkpoint["last_error"] = f"{type(error).__name__}: {error}"
                if checkpoint["consecutive_failures"] >= int(
                    config["maximum_consecutive_failures"]
                ):
                    stop_reason = "consecutive_failure_limit_reached"
                    break
            checkpoint["updated_utc"] = datetime.now(UTC).isoformat()
            _safe_atomic_write(
                paths["checkpoint"],
                _checkpoint_payload(checkpoint),
                int(config["maximum_output_bytes"]),
            )
            if checkpoint["refresh_count"] >= int(config["maximum_refreshes"]):
                break
            try:
                _interruptible_wait(
                    float(config["refresh_interval_seconds"]),
                    float(config["control_poll_interval_seconds"]),
                    lambda active_sha=config_sha: _control_guard(
                        paths, config_path, active_sha
                    ),
                )
            except ControlSignal as signal:
                if signal.reason == "configuration_changed":
                    try:
                        config, config_sha = _reload_preserving_epoch(
                            root, config_path, config, checkpoint
                        )
                        continue
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        checkpoint["last_error"] = f"{type(error).__name__}: {error}"
                        stop_reason = "configuration_reload_failed"
                        break
                stop_reason = signal.reason
                break
    finally:
        checkpoint.update(
            state="stopped",
            pid=None,
            stop_reason=stop_reason,
            updated_utc=datetime.now(UTC).isoformat(),
        )
        try:
            _safe_atomic_write(
                paths["checkpoint"],
                _checkpoint_payload(checkpoint),
                int(config["maximum_output_bytes"]),
            )
        finally:
            if lease_owned:
                _release_owned_lease(paths, identity)
    return 0


def service_status(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_safety_config(root, config_path)
    paths = _paths(root, config)
    if not paths["checkpoint"].is_file():
        return {"state": "not_started", "alive": False, "runtime": str(paths["runtime"])}
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    _validate_checkpoint(checkpoint, config)
    expected_identity = _argv_identity(_worker_command(root, config_path))
    return {
        **checkpoint,
        "alive": _pid_matches_worker(checkpoint.get("pid"), expected_identity),
        "config_current": checkpoint.get("config_file_sha256") == _file_sha(config_path),
        "runtime": str(paths["runtime"]),
    }


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("safety-service log symlink rejected")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    return os.fdopen(descriptor, "ab", buffering=0)


def start_service(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_safety_config(root, config_path)
    paths = _paths(root, config)
    paths["runtime"].mkdir(parents=True, exist_ok=True)
    command = _worker_command(root, config_path)
    identity = _argv_identity(command)
    starter_lease = _acquire_start_lease(paths, config, identity)
    checkpoint: dict[str, Any] | None = None
    process: subprocess.Popen | None = None
    try:
        _assert_no_legacy_worker(root, config)
        paths["stop"].unlink(missing_ok=True)
        checkpoint = _starting_checkpoint(
            paths, config, _file_sha(config_path), identity
        )
        _assert_no_legacy_worker(root, config)
        log = _open_log(paths["log"])
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": log,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | 0x08000000
            )
        else:
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(command, cwd=root, shell=False, **kwargs)
        finally:
            log.close()
        _transfer_lease_to_worker(
            paths, config, starter_lease, process.pid, identity
        )
        _wait_for_worker_identity(
            process.pid,
            identity,
            float(config["startup_identity_timeout_seconds"]),
            float(config["control_poll_interval_seconds"]),
        )
    except Exception as error:
        if process is None:
            if checkpoint is not None:
                checkpoint.update(
                    state="stopped",
                    pid=None,
                    last_error=f"{type(error).__name__}: {error}",
                    stop_reason="start_failed_before_spawn",
                    updated_utc=datetime.now(UTC).isoformat(),
                )
                _safe_atomic_write(
                    paths["checkpoint"],
                    _checkpoint_payload(checkpoint),
                    int(config["maximum_output_bytes"]),
                )
            lease = _read_lease(paths["lease"], config)
            if (
                lease.get("owner_kind") == "starter"
                and lease.get("owner_pid") == os.getpid()
                and lease.get("owner_argv_sha256")
                == starter_lease.get("owner_argv_sha256")
            ):
                paths["lease"].unlink(missing_ok=True)
        raise
    return {
        "state": "starting",
        "alive": True,
        "pid": process.pid,
        "worker_argv_sha256": identity,
        "runtime_epoch": config["runtime_epoch"],
        "cutover_lease_acquired": True,
    }


def stop_service(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_safety_config(root, config_path)
    paths = _paths(root, config)
    _safe_atomic_write(paths["stop"], b"stop\n", 1024)
    return service_status(root, config_path)


def build_safety_readiness(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_safety_config(root, config_path)
    source = root / "src/sigma_theory_compiler/unified_engine_live_service_safety.py"
    test = root / "tests/test_unified_engine_live_service_safety.py"
    gitignore = root / ".gitignore"
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "decision": "hardened_service_ready_not_started",
        "source_bindings": {
            "config": {"path": config_path.relative_to(root).as_posix(), "file_sha256": _file_sha(config_path)},
            "source": {"path": source.relative_to(root).as_posix(), "file_sha256": _file_sha(source)},
            "test": {"path": test.relative_to(root).as_posix(), "file_sha256": _file_sha(test)},
            "gitignore": {
                "path": gitignore.relative_to(root).as_posix(),
                "file_sha256": _file_sha(gitignore),
            },
            "predecessors": config["predecessors"],
        },
        "safety_contract": {
            "windows_argv_list_shell_false": True,
            "worker_pid_bound_to_normalized_argv": True,
            "runtime_directory_fixed": config["runtime_directory"],
            "runtime_epoch": config["runtime_epoch"],
            "counters_preserved_across_compatible_reload": True,
            "reload_failures_checkpointed_in_worker_finally": True,
            "control_poll_interval_seconds": config["control_poll_interval_seconds"],
            "refresh_phase_control_guards": 4,
            "unpredictable_exclusive_temp_files": True,
            "target_and_log_symlinks_rejected": True,
            "pre_and_post_projection_input_manifest_required": True,
            "stale_projection_publication_allowed": False,
            "legacy_worker_absence_required_before_start": True,
            "legacy_runtime_directory": config["legacy_runtime_directory"],
            "cross_start_exclusive_lease": config["cutover_lease_path"],
            "atomic_starting_checkpoint_before_spawn": True,
            "stale_lease_recovery_requires_pid_argv_nonmatch": True,
            "worker_finally_releases_owned_lease": True,
            "repeated_start_launch_allowed": False,
            "first_refresh_observes_running_checkpoint": True,
            "startup_identity_timeout_seconds": config[
                "startup_identity_timeout_seconds"
            ],
            "runtime_outputs_gitignored": True,
        },
        "remaining_limitations": [
            "a stop or drift request cannot interrupt the interior of one third-party snapshot or leaderboard builder call; it is observed at the next guarded phase boundary",
            "snapshot and dashboard target replacement is individually atomic but not a single cross-file filesystem transaction",
            "PID command-line verification depends on psutil availability and fails closed when unavailable",
            "the legacy service does not consume the cutover lease, so an operator must keep legacy start disabled after the exact pre-spawn absence checks",
            "the hardened service is isolated and not started or substituted for the existing integrated live service by this lane",
        ],
        "data_seals": config["data_seals"],
        "live_database_opened_by_readiness": False,
        "supervisor_outputs_opened_by_readiness": False,
        "service_started": False,
    }
    return {**body, "content_sha256": _sha(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("start", "status", "stop", "worker", "readiness")
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/unified_engine_live_service_safety.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    config_path = _resolve_inside(root, args.config)
    if args.command == "worker":
        return run_worker(root, config_path)
    if args.command == "start":
        result = start_service(root, config_path)
    elif args.command == "stop":
        result = stop_service(root, config_path)
    elif args.command == "status":
        result = service_status(root, config_path)
    else:
        result = build_safety_readiness(root, config_path)
        config = load_safety_config(root, config_path)
        _safe_atomic_write(
            _resolve_inside(root, config["output_path"]),
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
            int(config["maximum_output_bytes"]),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
