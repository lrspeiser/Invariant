"""Bounded NVML-only deferred ownership for the reviewed RTX workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

SCHEMA = "sigma-kastner-schlatter-deferred-gpu-ownership-readiness-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-deferred-gpu-ownership-config-1.0"
EXPECTED_RUNTIME = (
    "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-runtime/deferred-ownership"
)
EXPECTED_SEALS = {
    "cuda_context_created_while_waiting": False,
    "detached_process_launch": False,
    "existing_gpu_process_signaled": False,
    "arbitrary_callable_or_subprocess_injection": False,
    "sqlite_access": False,
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


def _read_json_bounded(path: Path, maximum_bytes: int) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError("deferred ownership symlink rejected")
    if path.stat().st_size > maximum_bytes:
        raise RuntimeError("deferred ownership state exceeds bound")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("deferred ownership state is not an object")
    return value


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("deferred ownership path escapes repository") from error
    return target


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "mechanism_id",
        "runtime_directory",
        "lease_name",
        "checkpoint_name",
        "recovery_name",
        "stop_name",
        "output_path",
        "poll_interval_seconds",
        "maximum_wait_seconds",
        "required_consecutive_safe_samples",
        "maximum_gpu_utilization_percent",
        "minimum_free_gpu_memory_mib",
        "maximum_checkpoint_bytes",
        "bindings",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported deferred GPU ownership config")
    if (
        config.get("runtime_directory") != EXPECTED_RUNTIME
        or config.get("lease_name") != "deferred-gpu-owner.lease.json"
        or config.get("checkpoint_name") != "deferred-gpu-owner-checkpoint.json"
        or config.get("recovery_name") != "deferred-gpu-owner-lease-recovery.json"
        or config.get("stop_name") != "deferred-gpu-owner.stop.request"
        or config.get("seals") != EXPECTED_SEALS
    ):
        raise ValueError("deferred ownership closed contract changed")
    for key in (
        "poll_interval_seconds",
        "maximum_wait_seconds",
        "required_consecutive_safe_samples",
        "maximum_gpu_utilization_percent",
        "minimum_free_gpu_memory_mib",
        "maximum_checkpoint_bytes",
    ):
        if float(config[key]) <= 0:
            raise ValueError("deferred ownership bounds must be positive")
    if (
        float(config["maximum_gpu_utilization_percent"]) > 20
        or int(config["required_consecutive_safe_samples"]) < 3
        or float(config["maximum_wait_seconds"]) > 3600
    ):
        raise ValueError("deferred ownership safety bound widened")
    expected_bindings = {"scheduler_config", "scheduler_source", "scheduler_readiness", "gitignore"}
    if set(config.get("bindings", {})) != expected_bindings:
        raise ValueError("deferred ownership binding set changed")
    for name, binding in config["bindings"].items():
        path_bound = _inside(root, binding["path"])
        if _file_sha(path_bound) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
        if "content_sha256" in binding:
            value = json.loads(path_bound.read_text(encoding="utf-8"))
            if (
                value.get("content_sha256") != binding["content_sha256"]
                or _content_sha(value) != binding["content_sha256"]
            ):
                raise ValueError(f"{name} content hash mismatch")
    scheduler = json.loads(
        _inside(root, config["bindings"]["scheduler_config"]["path"]).read_text(encoding="utf-8")
    )
    if (
        scheduler.get("reviewed_workload_id")
        != "kastner-schlatter-set-indexed-cuda-falsification-001"
        or scheduler.get("persistent_config", {}).get("supervisor", {}).get("gpu_workers") != 1
        or scheduler.get("persistent_config", {}).get("supervisor", {}).get("cpu_workers") != 0
    ):
        raise ValueError("reviewed single-owner scheduler boundary changed")
    ignored = (
        _inside(root, config["bindings"]["gitignore"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    ignored_parent = "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-runtime/"
    if ignored_parent not in ignored:
        raise ValueError("deferred ownership runtime is not Git-ignored")
    return config, root


def sample_nvml() -> dict[str, Any]:
    """Read device-wide counters without creating a CUDA context."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            utilization = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_mw = int(pynvml.nvmlDeviceGetPowerUsage(handle))
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        raise RuntimeError(
            f"NVML ownership sample unavailable: {type(error).__name__}: {error}"
        ) from error
    return {
        "device_index": 0,
        "device_name": name.decode() if isinstance(name, bytes) else str(name),
        "gpu_utilization_percent": utilization,
        "memory_used_mib": int(memory.used // 1024**2),
        "memory_free_mib": int(memory.free // 1024**2),
        "memory_total_mib": int(memory.total // 1024**2),
        "power_watts": power_mw / 1000,
        "sampled_utc": datetime.now(UTC).isoformat(),
        "scope": "device-wide instantaneous NVML sample; no CUDA context and no process signal",
    }


def _safe(sample: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    return int(sample["gpu_utilization_percent"]) <= int(
        config["maximum_gpu_utilization_percent"]
    ) and int(sample["memory_free_mib"]) >= int(config["minimum_free_gpu_memory_mib"])


def _atomic_json(path: Path, value: Mapping[str, Any], maximum_bytes: int) -> None:
    payload = (_canonical(value) + "\n").encode()
    if len(payload) > maximum_bytes:
        raise RuntimeError("deferred ownership checkpoint exceeds bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("deferred ownership symlink rejected")
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
    """Bind a lease to the complete argv, not a family/module label."""
    return _sha(argv) if argv else ""


def _current_identity() -> str:
    try:
        import psutil

        return _process_identity(psutil.Process(os.getpid()).cmdline())
    except (ImportError, OSError, ValueError) as error:
        raise RuntimeError("deferred ownership process identity unavailable") from error


def _owner_matches(pid: int, identity: str) -> bool:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("cannot import process inventory for deferred lease") from error

    try:
        process = psutil.Process(pid)
        return process.is_running() and _process_identity(process.cmdline()) == identity
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, psutil.ZombieProcess, OSError) as error:
        raise RuntimeError("cannot verify deferred lease owner identity") from error


def _lease_value(config: Mapping[str, Any], role: str, polls: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sigma-deferred-gpu-owner-lease-1.0",
        "mechanism_id": config["mechanism_id"],
        "role": role,
        "pid": os.getpid(),
        "process_argv_sha256": _current_identity(),
        "polls": polls,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    return {**body, "content_sha256": _content_sha(body)}


def _acquire_waiter(runtime: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    runtime.mkdir(parents=True, exist_ok=True)
    if runtime.is_symlink() or any(runtime.glob("*.sqlite*")):
        raise RuntimeError("deferred ownership runtime is unsafe or contains SQLite")
    lease_path = runtime / config["lease_name"]
    value = _lease_value(config, "waiting", 0)
    payload = (_canonical(value) + "\n").encode()
    try:
        descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _read_json_bounded(lease_path, int(config["maximum_checkpoint_bytes"]))
        if existing.get("content_sha256") != _content_sha(existing):
            raise RuntimeError("deferred owner lease tamper detected") from None
        if _owner_matches(existing.get("pid", -1), existing.get("process_argv_sha256", "")):
            raise RuntimeError("deferred GPU owner already active") from None
        _atomic_json(
            runtime / config["recovery_name"], existing, int(config["maximum_checkpoint_bytes"])
        )
        lease_path.unlink()
        descriptor = os.open(lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return lease_path, value


def _write_checkpoint(
    runtime: Path,
    config: Mapping[str, Any],
    *,
    state: str,
    polls: int,
    consecutive_safe: int,
    sample: Mapping[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "sigma-deferred-gpu-owner-checkpoint-1.0",
        "mechanism_id": config["mechanism_id"],
        "state": state,
        "pid": os.getpid() if state in {"waiting", "reserved"} else None,
        "polls": polls,
        "consecutive_safe_samples": consecutive_safe,
        "required_consecutive_safe_samples": config["required_consecutive_safe_samples"],
        "last_nvml_sample": dict(sample) if sample is not None else None,
        "error": error,
        "cuda_context_created": False,
        "sqlite_accessed": False,
        "existing_process_signaled": False,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    value = {**body, "content_sha256": _content_sha(body)}
    _atomic_json(
        runtime / config["checkpoint_name"], value, int(config["maximum_checkpoint_bytes"])
    )
    return value


@dataclass
class DeferredOwnershipToken:
    runtime: Path
    config: dict[str, Any]
    lease_path: Path
    owned: dict[str, Any]
    checkpoint: dict[str, Any]
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        if self.lease_path.exists():
            current = _read_json_bounded(
                self.lease_path, int(self.config["maximum_checkpoint_bytes"])
            )
            if current.get("pid") == self.owned.get("pid") and current.get(
                "process_argv_sha256"
            ) == self.owned.get("process_argv_sha256"):
                self.lease_path.unlink()
        self.checkpoint = _write_checkpoint(
            self.runtime,
            self.config,
            state="released",
            polls=int(self.checkpoint["polls"]),
            consecutive_safe=int(self.checkpoint["consecutive_safe_samples"]),
            sample=self.checkpoint["last_nvml_sample"],
            error=None,
        )
        self.released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def wait_for_ownership(
    config_path: str | Path,
    *,
    runtime_override: str | Path | None = None,
    maximum_polls_override: int | None = None,
) -> DeferredOwnershipToken:
    """Wait without CUDA/SQLite and return a held ownership token after safe samples."""
    config, root = load_config(config_path)
    override_path = Path(runtime_override) if runtime_override is not None else None
    if override_path is not None and override_path.is_symlink():
        raise ValueError("deferred ownership runtime override symlink rejected")
    runtime = (
        override_path.resolve()
        if override_path is not None
        else _inside(root, config["runtime_directory"])
    )
    live_campaign = (root / "runs/campaigns").resolve()
    try:
        runtime.relative_to(live_campaign)
    except ValueError:
        pass
    else:
        raise ValueError("live campaign runtime override rejected")
    if override_path is not None:
        try:
            runtime.relative_to(Path(tempfile.gettempdir()).resolve())
        except ValueError as error:
            raise ValueError("runtime override is restricted to temporary tests") from error
    lease_path, owned = _acquire_waiter(runtime, config)
    polls = consecutive = 0
    sample: dict[str, Any] | None = None
    maximum_polls = int(
        maximum_polls_override
        if maximum_polls_override is not None
        else math.floor(
            float(config["maximum_wait_seconds"]) / float(config["poll_interval_seconds"])
        )
        + 1
    )
    configured_maximum_polls = (
        math.floor(float(config["maximum_wait_seconds"]) / float(config["poll_interval_seconds"]))
        + 1
    )
    if maximum_polls < 1 or maximum_polls > configured_maximum_polls:
        raise ValueError("deferred ownership poll bound override rejected")
    try:
        _write_checkpoint(
            runtime,
            config,
            state="waiting",
            polls=0,
            consecutive_safe=0,
            sample=None,
            error=None,
        )
        while polls < maximum_polls:
            if (runtime / config["stop_name"]).exists():
                raise RuntimeError("deferred GPU ownership stop requested")
            sample = sample_nvml()
            polls += 1
            consecutive = consecutive + 1 if _safe(sample, config) else 0
            owned = _lease_value(config, "waiting", polls)
            _atomic_json(lease_path, owned, int(config["maximum_checkpoint_bytes"]))
            checkpoint = _write_checkpoint(
                runtime,
                config,
                state="waiting",
                polls=polls,
                consecutive_safe=consecutive,
                sample=sample,
                error=None,
            )
            if consecutive >= int(config["required_consecutive_safe_samples"]):
                owned = _lease_value(config, "gpu_owner_reserved", polls)
                _atomic_json(lease_path, owned, int(config["maximum_checkpoint_bytes"]))
                checkpoint = _write_checkpoint(
                    runtime,
                    config,
                    state="reserved",
                    polls=polls,
                    consecutive_safe=consecutive,
                    sample=sample,
                    error=None,
                )
                return DeferredOwnershipToken(runtime, config, lease_path, owned, checkpoint)
            if polls < maximum_polls:
                time.sleep(float(config["poll_interval_seconds"]))
        raise TimeoutError("bounded deferred GPU ownership wait expired")
    except Exception as error:
        _write_checkpoint(
            runtime,
            config,
            state="timed_out" if isinstance(error, TimeoutError) else "failed",
            polls=polls,
            consecutive_safe=consecutive,
            sample=sample,
            error=f"{type(error).__name__}: {error}",
        )
        if lease_path.exists():
            current = _read_json_bounded(lease_path, int(config["maximum_checkpoint_bytes"]))
            if current.get("pid") == owned.get("pid"):
                lease_path.unlink()
        raise


def build_readiness(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    current_sample = sample_nvml()
    current_safe = _safe(current_sample, config)
    runtime = _inside(root, config["runtime_directory"])
    source = root / "src/sigma_theory_compiler/kastner_schlatter_deferred_gpu_ownership.py"
    test = root / "tests/test_kastner_schlatter_deferred_gpu_ownership.py"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "mechanism_id": config["mechanism_id"],
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
        "ownership_contract": {
            "poll_backend": "NVML only; no CUDA context",
            "poll_interval_seconds": config["poll_interval_seconds"],
            "maximum_wait_seconds": config["maximum_wait_seconds"],
            "maximum_polls": math.floor(
                config["maximum_wait_seconds"] / config["poll_interval_seconds"]
            )
            + 1,
            "required_consecutive_safe_samples": config["required_consecutive_safe_samples"],
            "maximum_gpu_utilization_percent": config["maximum_gpu_utilization_percent"],
            "minimum_free_gpu_memory_mib": config["minimum_free_gpu_memory_mib"],
            "exclusive_pid_argv_waiter_and_owner_lease": config["lease_name"],
            "stale_lease_recovery_requires_owner_nonmatch": True,
            "detached_launch_or_process_signal_surface": False,
            "sqlite_surface": False,
            "handoff_to_scheduler_automatic": False,
        },
        "current_runtime_audit": {
            "nvml_sample": current_sample,
            "single_sample_safe": current_safe,
            "ownership_reservable_now": False,
            "reason": "mechanism was not started; reservation additionally requires consecutive safe samples",
            "runtime_directory_exists": runtime.exists(),
            "lease_exists": (runtime / config["lease_name"]).exists(),
            "checkpoint_exists": (runtime / config["checkpoint_name"]).exists(),
        },
        "execution_state": {
            "waiter_started_by_readiness": False,
            "gpu_owner_reserved_by_readiness": False,
            "existing_gpu_process_signaled": False,
            "sqlite_accessed": False,
        },
        "decision": (
            "deferred_gpu_ownership_ready_current_single_sample_safe_not_started"
            if current_safe
            else "deferred_gpu_ownership_ready_current_device_occupied_not_started"
        ),
        "seals": config["seals"],
        "observations_opened": False,
        "scientific_test_pass": False,
        "readiness_advanced": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_readiness(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, root = load_config(config_path)
    if result.get("schema_version") != SCHEMA or result.get("content_sha256") != _content_sha(
        result
    ):
        raise ValueError("deferred readiness schema or content hash mismatch")
    recorded_sample = result.get("current_runtime_audit", {}).get("nvml_sample", {})
    recorded_safe = _safe(recorded_sample, config)
    expected_decision = (
        "deferred_gpu_ownership_ready_current_single_sample_safe_not_started"
        if recorded_safe
        else "deferred_gpu_ownership_ready_current_device_occupied_not_started"
    )
    if result.get("decision") != expected_decision:
        raise ValueError("deferred readiness decision changed")
    if result["current_runtime_audit"].get("single_sample_safe") is not recorded_safe:
        raise ValueError("deferred readiness sample classification changed")
    if result.get("seals") != EXPECTED_SEALS:
        raise ValueError("deferred readiness seals changed")
    if any(
        result.get(key) is not False
        for key in ("observations_opened", "scientific_test_pass", "readiness_advanced")
    ):
        raise ValueError("deferred readiness claim changed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"deferred readiness {name} hash mismatch")
    for name, binding in config["bindings"].items():
        if result["source_bindings"].get(name) != binding:
            raise ValueError(f"deferred readiness {name} binding changed")
    if result["ownership_contract"]["sqlite_surface"] is not False:
        raise ValueError("SQLite surface opened")
    if result["execution_state"] != {
        "waiter_started_by_readiness": False,
        "gpu_owner_reserved_by_readiness": False,
        "existing_gpu_process_signaled": False,
        "sqlite_accessed": False,
    }:
        raise ValueError("deferred readiness execution state changed")
    if config["runtime_directory"] != EXPECTED_RUNTIME:
        raise ValueError("deferred runtime changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--write-readiness", action="store_true")
    parser.add_argument("--validate-readiness")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    selected = sum((args.write_readiness, bool(args.validate_readiness), args.wait))
    if selected != 1:
        raise ValueError("select exactly one deferred ownership operation")
    if args.validate_readiness:
        validate_readiness(
            json.loads(Path(args.validate_readiness).read_text(encoding="utf-8")), args.config
        )
        return 0
    if args.wait:
        with wait_for_ownership(args.config) as token:
            print(_canonical(token.checkpoint))
        return 0
    result = build_readiness(args.config)
    config, root = load_config(args.config)
    output = _inside(root, config["output_path"])
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
