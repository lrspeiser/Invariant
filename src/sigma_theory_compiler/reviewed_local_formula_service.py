"""Restart-safe standalone lifecycle for the bounded reviewed local formula epoch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .campaign import CampaignStore, ClaimedTask, stable_id
from .campaign_engine import CampaignEngine, WorkerOutcome
from .llm_formula_proposal_adapter import canonical_sha256, sha256_file
from .reviewed_local_formula_epoch import run_bounded_mock_epoch

TASK_TYPE = "reviewed_local_formula_epoch_run"
CHECKPOINT_SCHEMA = "sigma-reviewed-local-formula-service-checkpoint-1.0"


class ReviewedLocalServiceError(ValueError):
    """Raised for lifecycle, budget, configuration, or checkpoint violations."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_hash(payload: dict[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != "checkpoint_sha256"})


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    value = dict(payload)
    value["checkpoint_sha256"] = _checkpoint_hash(value)
    _atomic_json(path, value)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReviewedLocalServiceError("service checkpoint is missing")
    value = _read_json(path)
    if value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ReviewedLocalServiceError("service checkpoint schema mismatch")
    if value.get("checkpoint_sha256") != _checkpoint_hash(value):
        raise ReviewedLocalServiceError("service checkpoint hash mismatch")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    required = {
        "budgets",
        "component_sha256",
        "execution_enabled",
        "network_allowed",
        "schema_version",
    }
    if set(raw) != required or raw["schema_version"] != "sigma-reviewed-local-formula-service-config-1.0":
        raise ReviewedLocalServiceError("service config shape or version mismatch")
    if not isinstance(raw["execution_enabled"], bool) or raw["network_allowed"] is not False:
        raise ReviewedLocalServiceError("service execution/network flags are invalid")
    budgets = raw["budgets"]
    if set(budgets) != {
        "maximum_attempts",
        "maximum_disk_bytes",
        "maximum_tasks",
        "maximum_wall_seconds",
    }:
        raise ReviewedLocalServiceError("service budget shape mismatch")
    attempts = int(budgets["maximum_attempts"])
    disk = int(budgets["maximum_disk_bytes"])
    tasks = int(budgets["maximum_tasks"])
    wall = float(budgets["maximum_wall_seconds"])
    if not 1 <= attempts <= 4 or tasks != 1 or not 1_000_000 <= disk <= 1_000_000_000:
        raise ReviewedLocalServiceError("service task/retry/disk budget is outside bounds")
    if not 5 <= wall <= 300:
        raise ReviewedLocalServiceError("service wall budget is outside bounds")
    return raw


def _validate_components(repo_root: Path, config: dict[str, Any]) -> str:
    for relative, expected in config["component_sha256"].items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReviewedLocalServiceError(f"service component binding mismatch: {relative}")
    callback_contract = {
        "callbacks": [
            "bounded_local_mock_provider",
            "bounded_local_prompt_resolver",
            "bounded_local_quarantine_resolver",
            "bounded_local_covariant_compiler",
            "bounded_local_action_resolver",
            "bounded_local_policy_adapter",
            "policy_validate",
        ],
        "component_sha256": config["component_sha256"],
        "network_allowed": False,
        "paid_spend_usd": "0.000000",
    }
    return canonical_sha256(callback_contract)


def _disk_usage(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _service_campaign_config(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    budgets = config["budgets"]
    return {
        "name": "reviewed local formula standalone service",
        "project_root": str(root),
        "output_root": str(root / "service-output"),
        "budget": {
            "duration_days": 1,
            "max_tasks": int(budgets["maximum_attempts"]),
            "max_failures": int(budgets["maximum_attempts"]),
            "max_cycles": 0,
        },
        "runtime": {"lease_seconds": 2},
        "scientific_contract": {
            "observations_authorized": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
        },
    }


def start_service(
    repo_root: Path,
    service_root: Path,
    config_path: Path,
    *,
    allow_bounded_test_harness: bool = False,
    foreground: bool = False,
) -> dict[str, Any]:
    config = _load_config(config_path)
    if config["execution_enabled"] and not allow_bounded_test_harness:
        raise ReviewedLocalServiceError("enabled service config requires bounded test authorization")
    if not config["execution_enabled"]:
        raise ReviewedLocalServiceError("checked-in service execution is disabled")
    service_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = service_root / "checkpoint.json"
    if checkpoint_path.exists():
        raise ReviewedLocalServiceError("service already exists; use resume")
    callback_root = _validate_components(repo_root, config)
    config_sha = sha256_file(config_path)
    store = CampaignStore(service_root / "service.sqlite")
    store.initialize()
    campaign_id = store.create_campaign(_service_campaign_config(service_root, config))
    task_id = store.add_task(
        campaign_id,
        TASK_TYPE,
        stage=0,
        payload={
            "callback_registry_sha256": callback_root,
            "config_sha256": config_sha,
        },
        max_attempts=int(config["budgets"]["maximum_attempts"]),
        idempotency_key=stable_id("LOCALEPOCHSERVICE", config_sha, callback_root),
    )
    _write_checkpoint(
        checkpoint_path,
        {
            "callback_registry_sha256": callback_root,
            "campaign_id": campaign_id,
            "config_sha256": config_sha,
            "last_stop_reason": "awaiting_worker",
            "schema_version": CHECKPOINT_SCHEMA,
            "service_state": "ready",
            "task_id": task_id,
        },
    )
    if _disk_usage(service_root) > int(config["budgets"]["maximum_disk_bytes"]):
        raise ReviewedLocalServiceError("service disk budget exceeded during start")
    return resume_service(repo_root, service_root, config_path) if foreground else status_service(
        repo_root, service_root, config_path
    )


def _validate_epoch_status(status: dict[str, Any]) -> None:
    expected = {
        "candidate_count": 1,
        "compiler_receipt_pass_count": 2,
        "decision_counts": {"block": 1, "dedup": 1, "pass": 1, "reject": 1},
        "lineage_preserved": True,
        "network_calls": 0,
        "next_stage_enqueue_count": 1,
        "paid_spend_usd": "0.000000",
        "policy_pass_count": 1,
        "proposal_quarantine_count": 4,
    }
    if any(status.get(key) != value for key, value in expected.items()):
        raise ReviewedLocalServiceError("bounded epoch status contract mismatch")


def resume_service(repo_root: Path, service_root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    checkpoint_path = service_root / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    if sha256_file(config_path) != checkpoint["config_sha256"]:
        raise ReviewedLocalServiceError("service config changed since start")
    callback_root = _validate_components(repo_root, config)
    if callback_root != checkpoint["callback_registry_sha256"]:
        raise ReviewedLocalServiceError("callback registry root changed since start")
    if not config["execution_enabled"]:
        raise ReviewedLocalServiceError("service execution remains disabled")
    store = CampaignStore(service_root / "service.sqlite")
    campaign = store.campaign(checkpoint["campaign_id"])
    if campaign["state"] == "paused":
        store.set_campaign_state(checkpoint["campaign_id"], "active", "explicit resume")
    store.recover_expired_leases(checkpoint["campaign_id"])
    wall_limit = float(config["budgets"]["maximum_wall_seconds"])
    disk_limit = int(config["budgets"]["maximum_disk_bytes"])

    def handler(task: ClaimedTask) -> WorkerOutcome:
        started = time.monotonic()
        attempt_root = service_root / "attempts" / f"attempt-{task.attempt:02d}"
        if attempt_root.exists():
            raise ReviewedLocalServiceError("attempt directory already exists")
        status = run_bounded_mock_epoch(repo_root, attempt_root)
        _validate_epoch_status(status)
        elapsed = time.monotonic() - started
        if elapsed > wall_limit:
            raise ReviewedLocalServiceError("service wall budget exceeded")
        if _disk_usage(service_root) > disk_limit:
            raise ReviewedLocalServiceError("service disk budget exceeded")
        return WorkerOutcome(
            result={
                "attempt": task.attempt,
                "callback_registry_sha256": callback_root,
                "epoch_status": status,
            }
        )

    engine = CampaignEngine(
        store,
        checkpoint["campaign_id"],
        "reviewed-local-service-worker",
        {TASK_TYPE},
    )
    engine.handlers[TASK_TYPE] = handler
    report = engine.run(max_tasks=1, duration_seconds=wall_limit)
    task_status, result = _service_task(store, checkpoint["task_id"])
    if task_status == "succeeded":
        store.set_campaign_state(checkpoint["campaign_id"], "complete", "bounded epoch complete")
        state, reason = "complete", "bounded_epoch_complete"
    elif task_status == "queued":
        state, reason = "ready", "retry_scheduled"
    elif task_status == "running":
        state, reason = "running", "owned_by_other_worker"
    else:
        state, reason = "failed", f"task_{task_status}"
    _write_checkpoint(
        checkpoint_path,
        {
            **{key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"},
            "last_stop_reason": reason,
            "service_state": state,
        },
    )
    return {
        **status_service(repo_root, service_root, config_path),
        "processed_tasks": report["processed_tasks"],
        "result_available": bool(result),
    }


def _service_task(store: CampaignStore, task_id: str) -> tuple[str, dict[str, Any]]:
    with store.connect() as connection:
        row = connection.execute("SELECT status,result_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    if row is None:
        raise ReviewedLocalServiceError("service task is missing")
    return row["status"], json.loads(row["result_json"] or "{}")


def stop_service(repo_root: Path, service_root: Path, config_path: Path) -> dict[str, Any]:
    checkpoint_path = service_root / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    config = _load_config(config_path)
    if sha256_file(config_path) != checkpoint["config_sha256"]:
        raise ReviewedLocalServiceError("service config changed since start")
    _validate_components(repo_root, config)
    store = CampaignStore(service_root / "service.sqlite")
    campaign = store.campaign(checkpoint["campaign_id"])
    if campaign["state"] == "active":
        store.set_campaign_state(checkpoint["campaign_id"], "paused", "explicit stop")
    _write_checkpoint(
        checkpoint_path,
        {
            **{key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"},
            "last_stop_reason": "explicit_stop",
            "service_state": "stopped",
        },
    )
    return status_service(repo_root, service_root, config_path)


def status_service(repo_root: Path, service_root: Path, config_path: Path) -> dict[str, Any]:
    checkpoint = _load_checkpoint(service_root / "checkpoint.json")
    config = _load_config(config_path)
    if sha256_file(config_path) != checkpoint["config_sha256"]:
        raise ReviewedLocalServiceError("service config changed since start")
    callback_root = _validate_components(repo_root, config)
    if callback_root != checkpoint["callback_registry_sha256"]:
        raise ReviewedLocalServiceError("callback registry root changed since start")
    store = CampaignStore(service_root / "service.sqlite")
    task_status, result = _service_task(store, checkpoint["task_id"])
    return {
        "callback_registry_sha256": callback_root,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "disk_bytes": _disk_usage(service_root),
        "last_stop_reason": checkpoint["last_stop_reason"],
        "result_core_sha256": result.get("epoch_status", {}).get("core_sha256"),
        "schema_version": "sigma-reviewed-local-formula-service-status-1.0",
        "service_state": checkpoint["service_state"],
        "task_status": task_status,
    }


def export_service(repo_root: Path, service_root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    status = status_service(repo_root, service_root, config_path)
    if status["task_status"] != "succeeded":
        raise ReviewedLocalServiceError("service result is not complete")
    checkpoint = _load_checkpoint(service_root / "checkpoint.json")
    store = CampaignStore(service_root / "service.sqlite")
    _, result = _service_task(store, checkpoint["task_id"])
    epoch = result["epoch_status"]
    exported: dict[str, Any] = {
        "callback_registry_sha256": checkpoint["callback_registry_sha256"],
        "config_sha256": checkpoint["config_sha256"],
        "epoch_core_sha256": epoch["core_sha256"],
        "epoch_status": epoch,
        "schema_version": "sigma-reviewed-local-formula-service-export-1.0",
    }
    exported["content_sha256"] = canonical_sha256(exported)
    if output.exists() and _read_json(output) != exported:
        raise ReviewedLocalServiceError("refusing to overwrite a different service export")
    _atomic_json(output, exported)
    return exported


def build_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    if config["execution_enabled"]:
        raise ReviewedLocalServiceError("checked-in service config must remain disabled")
    callback_root = _validate_components(repo_root, config)
    artifact: dict[str, Any] = {
        "budgets": config["budgets"],
        "callback_registry_sha256": callback_root,
        "config_sha256": sha256_file(config_path),
        "default_execution_enabled": False,
        "deterministic_export": True,
        "network_allowed": False,
        "paid_spend_usd": "0.000000",
        "schema_version": "sigma-reviewed-local-formula-service-readiness-1.0",
        "source_sha256": sha256_file(
            repo_root / "src/sigma_theory_compiler/reviewed_local_formula_service.py"
        ),
        "status": "ready_disabled_bounded_local_only",
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "stop", "resume", "export"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--allow-bounded-test-harness", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "start":
        result = start_service(
            args.repo_root,
            args.service_root,
            args.config,
            allow_bounded_test_harness=args.allow_bounded_test_harness,
            foreground=args.foreground,
        )
    elif args.command == "status":
        result = status_service(args.repo_root, args.service_root, args.config)
    elif args.command == "stop":
        result = stop_service(args.repo_root, args.service_root, args.config)
    elif args.command == "resume":
        result = resume_service(args.repo_root, args.service_root, args.config)
    else:
        if args.output is None:
            raise ReviewedLocalServiceError("export requires --output")
        result = export_service(args.repo_root, args.service_root, args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
