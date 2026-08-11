from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .process_health import pid_alive
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _hash_matches as _continuation_hash_matches,
)
from .quartic_tc2_mixed_third_jet_parallel_epoch import run_parallel_epoch

SCHEMA_VERSION = "sigma-quartic-tc2-mixed-third-jet-parallel-supervisor-1.0"
STATE_SCHEMA_VERSION = "sigma-quartic-tc2-mixed-third-jet-parallel-supervisor-state-1.0"
EXPORT_SCHEMA_VERSION = "sigma-quartic-tc2-mixed-third-jet-parallel-supervisor-export-1.0"
PARALLEL_POLICY = "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
FALSE_CLAIMS = {
    "full_mixed_sector_closed": False,
    "full_tube_Sylvester_identity": False,
    "CK1_closed": False,
    "CK3_closed": False,
    "TC2_closed": False,
    "B7_closed": False,
    "global_H7_closed": False,
    "lifespan_proved": False,
}

EpochRunner = Callable[..., dict[str, Any]]


class QuarticTC2MixedThirdJetParallelSupervisorError(ValueError):
    """Raised when the bounded parallel supervisor cannot advance safely."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _body(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "content_sha256"}


def _with_hash(value: dict[str, Any]) -> dict[str, Any]:
    body = _body(value)
    return {**body, "content_sha256": _sha(body)}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _sha(_body(value))


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            f"expected JSON object: {path}"
        )
    return value, data


def _resolve_under(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            f"{label} escapes project root"
        )
    return path


def _pid_alive(pid: int | None) -> bool:
    return pid_alive(pid)


def _paths(project_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    service_root = _resolve_under(
        project_root, str(config["supervisor_output_path"]), "supervisor output"
    )
    return {
        "root": service_root,
        "state": service_root / "supervisor-state.json",
        "status": service_root / "supervisor-status.json",
        "lock": service_root / "supervisor.lock",
        "stop": service_root / "stop.request",
        "log": service_root / "supervisor.log",
        "epoch_output": _resolve_under(
            project_root, str(config["epoch_output_path"]), "epoch output"
        ),
        "epoch_config": _resolve_under(
            project_root, str(config["epoch_config_path"]), "epoch config"
        ),
    }


def _validate_config(
    project_root: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    config, _ = _load(config_path)
    if config.get("schema_version") != SCHEMA_VERSION or not _hash_matches(config):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "parallel supervisor config hash/schema mismatch"
        )
    numeric = {
        "maximum_epochs_per_run": (1, 256),
        "maximum_wall_seconds": (1, 14 * 24 * 3600),
        "maximum_history_records": (1, 256),
        "poll_interval_seconds": (0, 60),
    }
    for key, (lower, upper) in numeric.items():
        value = float(config.get(key, -1))
        if not lower <= value <= upper:
            raise QuarticTC2MixedThirdJetParallelSupervisorError(
                f"invalid supervisor budget: {key}"
            )
    if (
        int(config.get("parallel_worker_count", 0)) != 8
        or config.get("parallel_execution_policy") != PARALLEL_POLICY
        or config.get("stop_on_first_obstruction") is not True
        or config.get("orphan_recovery_policy") != "validate_and_adopt"
        or config.get("resume_policy") != "record_sha256_chain"
        or config.get("external_paid_llm_calls") is not False
        or config.get("observational_data_opened") is not False
        or config.get("dark_matter_or_halo_inputs") is not False
        or config.get("redshift_distance_inputs") is not False
        or any(config.get(key) != "fail_closed" for key in FALSE_CLAIMS)
    ):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "parallel supervisor safety contract mismatch"
        )
    paths = _paths(project_root, config)
    epoch_config, epoch_data = _load(paths["epoch_config"])
    if (
        _file_sha(epoch_data) != config.get("epoch_config_file_sha256")
        or epoch_config.get("content_sha256")
        != config.get("epoch_config_content_sha256")
        or not _hash_matches(epoch_config)
        or int(epoch_config.get("parallel_worker_count", 0)) != 8
        or epoch_config.get("parallel_execution_policy") != PARALLEL_POLICY
        or int(epoch_config.get("max_chunks_per_invocation", 0)) != 1
        or epoch_config.get("obstruction_policy") != "permanent_stop"
    ):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "parallel epoch config binding mismatch"
        )
    return config, paths


def _read_epoch_checkpoint(path: Path) -> tuple[dict[str, Any], bytes]:
    checkpoint, data = _load(path)
    if (
        not _continuation_hash_matches(checkpoint)
        or any(checkpoint.get("claims", {}).values())
        or bool(checkpoint.get("permanently_stopped"))
        != (checkpoint.get("stop_reason") == "exact_obstruction")
        or len(checkpoint.get("history", [])) != checkpoint.get("completed_chunks")
    ):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "parallel epoch checkpoint contract mismatch"
        )
    return checkpoint, data


def _initial_state(
    config: dict[str, Any], checkpoint: dict[str, Any], checkpoint_data: bytes
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return _with_hash(
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "config_content_sha256": config["content_sha256"],
            "epoch_config_file_sha256": config["epoch_config_file_sha256"],
            "epoch_config_content_sha256": config["epoch_config_content_sha256"],
            "state": "initialized",
            "pid": None,
            "started_utc": None,
            "updated_utc": now.isoformat(),
            "deadline_utc": None,
            "epochs_completed": 0,
            "chunks_advanced": 0,
            "next_offset": checkpoint["next_offset"],
            "remaining_mixed_triples": checkpoint["remaining_mixed_triples"],
            "prior_resume_sha256": checkpoint["prior_resume_sha256"],
            "epoch_checkpoint_file_sha256": _file_sha(checkpoint_data),
            "epoch_checkpoint_content_sha256": checkpoint["content_sha256"],
            "stop_reason": None,
            "history": [],
            "claims": dict(FALSE_CLAIMS),
        }
    )


def _load_state(
    path: Path,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_data: bytes,
) -> dict[str, Any]:
    if not path.exists():
        return _initial_state(config, checkpoint, checkpoint_data)
    state, _ = _load(path)
    history = state.get("history", [])
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or not _hash_matches(state)
        or state.get("config_content_sha256") != config["content_sha256"]
        or state.get("epoch_config_file_sha256")
        != config["epoch_config_file_sha256"]
        or state.get("epoch_config_content_sha256")
        != config["epoch_config_content_sha256"]
        or len(history) != state.get("epochs_completed")
        or len(history) > int(config["maximum_history_records"])
        or state.get("chunks_advanced")
        != sum(int(row["chunks_advanced"]) for row in history)
        or any(state.get("claims", {}).values())
        or state.get("next_offset") != checkpoint.get("next_offset")
        or state.get("remaining_mixed_triples")
        != checkpoint.get("remaining_mixed_triples")
        or state.get("prior_resume_sha256") != checkpoint.get("prior_resume_sha256")
        or state.get("epoch_checkpoint_file_sha256") != _file_sha(checkpoint_data)
        or state.get("epoch_checkpoint_content_sha256")
        != checkpoint.get("content_sha256")
    ):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "parallel supervisor state/checkpoint mismatch"
        )
    return state


def _status(state: dict[str, Any], *, alive: bool) -> dict[str, Any]:
    return _with_hash(
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "state": state["state"],
            "pid": state["pid"],
            "alive": alive,
            "epochs_completed": state["epochs_completed"],
            "chunks_advanced": state["chunks_advanced"],
            "next_offset": state["next_offset"],
            "remaining_mixed_triples": state["remaining_mixed_triples"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "stop_reason": state["stop_reason"],
            "claims": state["claims"],
        }
    )


def _write_state(paths: dict[str, Path], state: dict[str, Any]) -> None:
    _atomic_write(paths["state"], _json_bytes(state))
    _atomic_write(paths["status"], _json_bytes(_status(state, alive=_pid_alive(state["pid"]))))


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QuarticTC2MixedThirdJetParallelSupervisorError(
                "malformed supervisor lock"
            ) from error
        if _pid_alive(lock.get("pid")):
            raise QuarticTC2MixedThirdJetParallelSupervisorError(
                "parallel supervisor is already running"
            )
        path.unlink()
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags)
    try:
        os.write(
            descriptor,
            _json_bytes({"pid": os.getpid(), "created_utc": datetime.now(UTC).isoformat()}),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_supervisor(
    project_root: Path,
    config_path: Path,
    *,
    epoch_runner: EpochRunner = run_parallel_epoch,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config, paths = _validate_config(project_root, config_path)
    checkpoint_path = paths["epoch_output"] / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "parallel epoch checkpoint is missing"
        )
    _acquire_lock(paths["lock"])
    try:
        checkpoint, checkpoint_data = _read_epoch_checkpoint(checkpoint_path)
        state = _load_state(
            paths["state"], config, checkpoint, checkpoint_data
        )
        if paths["stop"].exists():
            state = _with_hash(
                {
                    **_body(state),
                    "state": "stopped",
                    "pid": None,
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "stop_reason": "external_stop_requested",
                }
            )
            _write_state(paths, state)
            return state
        now = datetime.now(UTC)
        deadline = (
            datetime.fromisoformat(state["deadline_utc"])
            if state.get("deadline_utc")
            else now + timedelta(seconds=float(config["maximum_wall_seconds"]))
        )
        state = _with_hash(
            {
                **_body(state),
                "state": "running",
                "pid": os.getpid(),
                "started_utc": state.get("started_utc") or now.isoformat(),
                "updated_utc": now.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "stop_reason": None,
            }
        )
        _write_state(paths, state)
        start = monotonic()
        epochs_this_run = 0
        while epochs_this_run < int(config["maximum_epochs_per_run"]):
            if paths["stop"].exists():
                reason = "external_stop_requested"
                break
            if len(state["history"]) >= int(config["maximum_history_records"]):
                reason = "supervisor_history_limit"
                break
            if monotonic() - start >= float(config["maximum_wall_seconds"]):
                reason = "run_wall_time_reached"
                break
            if datetime.now(UTC) >= deadline:
                reason = "execution_deadline_reached"
                break
            before, before_data = _read_epoch_checkpoint(checkpoint_path)
            result = epoch_runner(
                project_root,
                config_path=paths["epoch_config"],
                output_path=paths["epoch_output"],
            )
            after, after_data = _read_epoch_checkpoint(checkpoint_path)
            advanced = int(result.get("chunks_advanced", -1))
            if advanced not in (0, 1):
                raise QuarticTC2MixedThirdJetParallelSupervisorError(
                    "parallel epoch advanced an invalid chunk count"
                )
            if advanced == 1:
                if (
                    after["completed_chunks"] != before["completed_chunks"] + 1
                    or after["next_offset"] <= before["next_offset"]
                    or after["remaining_mixed_triples"] >= before["remaining_mixed_triples"]
                    or after["prior_resume_sha256"] == before["prior_resume_sha256"]
                    or any(after.get("claims", {}).values())
                ):
                    raise QuarticTC2MixedThirdJetParallelSupervisorError(
                        "parallel epoch did not advance the exact chain"
                    )
            elif after_data != before_data:
                raise QuarticTC2MixedThirdJetParallelSupervisorError(
                    "zero-advance epoch mutated its checkpoint"
                )
            epochs_this_run += 1
            history_record = {
                "epoch_index": int(state["epochs_completed"]),
                "before_offset": before["next_offset"],
                "after_offset": after["next_offset"],
                "before_remaining": before["remaining_mixed_triples"],
                "after_remaining": after["remaining_mixed_triples"],
                "before_resume_sha256": before["prior_resume_sha256"],
                "after_resume_sha256": after["prior_resume_sha256"],
                "chunks_advanced": advanced,
                "epoch_reason": result.get("reason"),
                "checkpoint_file_sha256": _file_sha(after_data),
                "checkpoint_content_sha256": after["content_sha256"],
            }
            state = _with_hash(
                {
                    **_body(state),
                    "updated_utc": datetime.now(UTC).isoformat(),
                    "epochs_completed": int(state["epochs_completed"]) + 1,
                    "chunks_advanced": int(state["chunks_advanced"]) + advanced,
                    "next_offset": after["next_offset"],
                    "remaining_mixed_triples": after["remaining_mixed_triples"],
                    "prior_resume_sha256": after["prior_resume_sha256"],
                    "epoch_checkpoint_file_sha256": _file_sha(after_data),
                    "epoch_checkpoint_content_sha256": after["content_sha256"],
                    "history": [*state["history"], history_record],
                }
            )
            _write_state(paths, state)
            if after["permanently_stopped"]:
                reason = "exact_obstruction"
                break
            if after["remaining_mixed_triples"] <= 0:
                reason = "mixed_selector_complete_full_tube_still_open"
                break
            if advanced == 0:
                reason = str(result.get("reason") or "epoch_no_progress")
                break
            sleep(float(config["poll_interval_seconds"]))
        else:
            reason = "epoch_limit"
        terminal = reason in {
            "exact_obstruction",
            "mixed_selector_complete_full_tube_still_open",
        }
        state = _with_hash(
            {
                **_body(state),
                "state": "complete" if terminal else "stopped",
                "pid": None,
                "updated_utc": datetime.now(UTC).isoformat(),
                "stop_reason": reason,
            }
        )
        _write_state(paths, state)
        return state
    finally:
        if paths["lock"].exists():
            paths["lock"].unlink()


def start_supervisor(
    project_root: Path,
    config_path: Path,
    *,
    foreground: bool,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    config, paths = _validate_config(project_root, config_path)
    if paths["lock"].exists():
        lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
        if _pid_alive(lock.get("pid")):
            raise QuarticTC2MixedThirdJetParallelSupervisorError(
                "parallel supervisor is already running"
            )
    if paths["stop"].exists():
        paths["stop"].unlink()
    if foreground:
        return run_supervisor(project_root, config_path)
    paths["root"].mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sigma_theory_compiler.quartic_tc2_mixed_third_jet_parallel_supervisor",
        "run",
        "--project-root",
        str(project_root),
        "--config",
        str(config_path),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with paths["log"].open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )
    return {
        "state": "starting",
        "pid": process.pid,
        "supervisor_output_path": str(paths["root"]),
        "epoch_output_path": str(paths["epoch_output"]),
        "maximum_epochs_per_run": config["maximum_epochs_per_run"],
    }


def request_stop(project_root: Path, config_path: Path) -> dict[str, Any]:
    config, paths = _validate_config(project_root.resolve(), config_path.resolve())
    del config
    paths["root"].mkdir(parents=True, exist_ok=True)
    _atomic_write(
        paths["stop"],
        _json_bytes(
            {"requested_utc": datetime.now(UTC).isoformat(), "graceful": True}
        ),
    )
    return {"state": "stop_requested", "path": str(paths["stop"])}


def supervisor_status(project_root: Path, config_path: Path) -> dict[str, Any]:
    config, paths = _validate_config(project_root.resolve(), config_path.resolve())
    checkpoint, checkpoint_data = _read_epoch_checkpoint(
        paths["epoch_output"] / "checkpoint.json"
    )
    state = _load_state(
        paths["state"], config, checkpoint, checkpoint_data
    )
    status = _status(state, alive=_pid_alive(state.get("pid")))
    _atomic_write(paths["status"], _json_bytes(status))
    return status


def export_supervisor(
    project_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    if output_path != project_root and project_root not in output_path.parents:
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "supervisor export escapes project root"
        )
    config, paths = _validate_config(project_root, config_path)
    checkpoint, checkpoint_data = _read_epoch_checkpoint(
        paths["epoch_output"] / "checkpoint.json"
    )
    state = _load_state(paths["state"], config, checkpoint, checkpoint_data)
    if _pid_alive(state.get("pid")):
        raise QuarticTC2MixedThirdJetParallelSupervisorError(
            "refusing to export a changing live supervisor"
        )
    state_data = _json_bytes(state)
    source_data = Path(__file__).read_bytes()
    body = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "status": "portable_stopped_checkpoint",
        "supervisor_source": {
            "path": "src/sigma_theory_compiler/quartic_tc2_mixed_third_jet_parallel_supervisor.py",
            "file_sha256": _file_sha(source_data),
        },
        "supervisor_config": {
            "path": config_path.relative_to(project_root).as_posix(),
            "file_sha256": _file_sha(config_path.read_bytes()),
            "content_sha256": config["content_sha256"],
        },
        "epoch_config": {
            "path": config["epoch_config_path"],
            "file_sha256": config["epoch_config_file_sha256"],
            "content_sha256": config["epoch_config_content_sha256"],
        },
        "supervisor_checkpoint": {
            "file_sha256": _file_sha(state_data),
            "content_sha256": state["content_sha256"],
        },
        "epoch_checkpoint": {
            "file_sha256": _file_sha(checkpoint_data),
            "content_sha256": checkpoint["content_sha256"],
        },
        "lifecycle": {
            "state": state["state"],
            "stop_reason": state["stop_reason"],
            "epochs_completed": state["epochs_completed"],
            "chunks_advanced": state["chunks_advanced"],
            "next_offset": state["next_offset"],
            "remaining_mixed_triples": state["remaining_mixed_triples"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "graceful_stop_observed": state["stop_reason"]
            == "external_stop_requested",
            "resume_available": state["state"] == "stopped",
        },
        "parallel_contract": {
            "parallel_worker_count": config["parallel_worker_count"],
            "parallel_execution_policy": config["parallel_execution_policy"],
            "stop_on_first_obstruction": config["stop_on_first_obstruction"],
            "orphan_recovery_policy": config["orphan_recovery_policy"],
            "resume_policy": config["resume_policy"],
        },
        "claims": dict(FALSE_CLAIMS),
        "data_eligibility": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_calls": False,
        },
    }
    artifact = _with_hash(body)
    _atomic_write(output_path, _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervise bounded exact 8-worker mixed-third-jet epochs."
    )
    parser.add_argument(
        "command", choices=("start", "run", "status", "stop", "resume", "export")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/backgrounds/quartic_tc2_mixed_third_jet_parallel_supervisor.json"
        ),
    )
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/engine/quartic-tc2-mixed-third-jet-parallel-supervisor-readiness.json"
        ),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    if args.command == "run":
        result = run_supervisor(project_root, config_path)
    elif args.command in {"start", "resume"}:
        result = start_supervisor(
            project_root, config_path, foreground=args.foreground
        )
    elif args.command == "stop":
        result = request_stop(project_root, config_path)
    elif args.command == "status":
        result = supervisor_status(project_root, config_path)
    else:
        output_path = args.output if args.output.is_absolute() else project_root / args.output
        result = export_supervisor(project_root, config_path, output_path)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
