"""Current read-only operational audit plus isolated multi-task recovery exercise."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .persistent_parallel_search import PersistentParallelSearch
from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "sigma-current-operational-scratch-recovery-config-1.0"
RESULT_SCHEMA = "sigma-current-operational-scratch-recovery-result-1.0"
CONFIG_REL = "configs/current_operational_scratch_recovery_campaign.json"
SOURCE_REL = "src/sigma_theory_compiler/current_operational_scratch_recovery_campaign.py"
TEST_REL = "tests/test_current_operational_scratch_recovery_campaign.py"
RESULT_REL = "runs/engine/current-operational-scratch-recovery-campaign/result.json"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("campaign path is not portable")
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("campaign JSON must be an object")
    return value


def load_config(root: Path, path: Path | None = None) -> dict[str, Any]:
    expected = _resolve(root, CONFIG_REL)
    actual = expected if path is None else path.resolve()
    if actual != expected:
        raise ValueError("campaign config path changed")
    value = _load(actual)
    if set(value) != {
        "campaign_id",
        "live_state_audit",
        "metadata_snapshot",
        "resource_admission",
        "schema_version",
        "scratch_contract",
    }:
        raise ValueError("campaign config keys changed")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["campaign_id"] != "current-operational-scratch-recovery-campaign-v1"
        or value["resource_admission"]
        != {
            "cpu_utilization_strictly_below_percent": 92,
            "minimum_available_ram_mib": 32768,
            "require_every_sample_admitted": True,
            "sample_count": 3,
        }
        or value["scratch_contract"]
        != {
            "cpu_workers": 1,
            "gpu_workers": 0,
            "maximum_attempts": 2,
            "network_access": False,
            "scratch_must_be_outside_repository_runs": True,
            "tasks": 3,
        }
    ):
        raise ValueError("campaign config contract changed")
    expected_live = {
        "batch_cursor": "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/cumulative-cursor.json",
        "batch_result": "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/result.json",
        "live_database": "runs/campaigns/campaign-v1-live.sqlite",
        "terminal_service_result": "runs/engine/continuous-scientific-pipeline-service-result.json",
    }
    if value["live_state_audit"] != expected_live:
        raise ValueError("live-state audit inventory changed")
    expected_snapshot_keys = set(expected_live)
    snapshot = value["metadata_snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != expected_snapshot_keys:
        raise ValueError("metadata snapshot inventory changed")
    for label, relative in expected_live.items():
        row = snapshot[label]
        if (
            not isinstance(row, dict)
            or set(row) != {"modified_utc", "path", "size_bytes"}
            or row["path"] != relative
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] <= 0
        ):
            raise ValueError("metadata snapshot schema changed")
        parsed = datetime.fromisoformat(row["modified_utc"])
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("metadata snapshot timestamp is not explicit UTC")
    return value


def _normalize_samples(
    samples: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if len(samples) != contract["sample_count"]:
        raise ValueError("resource sample count changed")
    rows = []
    for sequence, sample in enumerate(samples):
        if set(sample) != {
            "available_ram_mib",
            "cpu_utilization_percent",
            "sampled_at",
            "sequence",
        }:
            raise ValueError("resource sample schema changed")
        cpu, ram = sample["cpu_utilization_percent"], sample["available_ram_mib"]
        try:
            cpu_value = float(cpu)
        except (TypeError, ValueError) as error:
            raise ValueError("resource CPU sample is not numeric") from error
        if (
            sample["sequence"] != sequence
            or isinstance(cpu, bool)
            or not 0 <= cpu_value <= 100
            or isinstance(ram, bool)
            or not isinstance(ram, int)
            or ram <= 0
        ):
            raise ValueError("resource sample value changed")
        parsed = datetime.fromisoformat(sample["sampled_at"])
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("resource sample timestamp is not explicit UTC")
        admitted = (
            cpu_value < contract["cpu_utilization_strictly_below_percent"]
            and ram >= contract["minimum_available_ram_mib"]
        )
        rows.append(
            {
                **dict(sample),
                "cpu_utilization_percent": format(cpu_value, ".1f"),
                "admitted": admitted,
            }
        )
    if not all(row["admitted"] for row in rows):
        raise ValueError("scratch recovery was not resource-admitted")
    return rows


def capture_resource_samples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Observe device CPU/RAM without controlling any external process."""
    import psutil

    rows = []
    for sequence in range(config["resource_admission"]["sample_count"]):
        rows.append(
            {
                "sequence": sequence,
                "sampled_at": datetime.now(UTC).isoformat(),
                "cpu_utilization_percent": float(psutil.cpu_percent(interval=1.0)),
                "available_ram_mib": int(psutil.virtual_memory().available // (1024 * 1024)),
            }
        )
    return rows


def _json_binding(
    path: Path, relative: str, snapshot: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _load(path)
    claimed = value.get("content_sha256")
    if claimed != canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    ):
        raise ValueError(f"live-state JSON content seal changed: {relative}")
    if path.stat().st_size != snapshot["size_bytes"]:
        raise ValueError(f"live-state JSON size changed: {relative}")
    return value, {
        "path": relative,
        "file_sha256": _file_sha(path),
        "content_sha256": claimed,
        "modified_utc": snapshot["modified_utc"],
        "size_bytes": snapshot["size_bytes"],
    }


def _audit_live_state(root: Path, config: Mapping[str, Any], observed_at: str) -> dict[str, Any]:
    inventory = config["live_state_audit"]
    snapshots = config["metadata_snapshot"]
    cursor, cursor_binding = _json_binding(
        _resolve(root, inventory["batch_cursor"]),
        inventory["batch_cursor"],
        snapshots["batch_cursor"],
    )
    batch, batch_binding = _json_binding(
        _resolve(root, inventory["batch_result"]),
        inventory["batch_result"],
        snapshots["batch_result"],
    )
    service, service_binding = _json_binding(
        _resolve(root, inventory["terminal_service_result"]),
        inventory["terminal_service_result"],
        snapshots["terminal_service_result"],
    )
    database_path = _resolve(root, inventory["live_database"])
    database_stat = database_path.stat()
    database_snapshot = snapshots["live_database"]
    if database_stat.st_size != database_snapshot["size_bytes"]:
        raise ValueError("live scheduler database size changed since the metadata snapshot")
    counts = cursor["counts"]
    if (
        batch["cumulative_ledger_binding"]["content_sha256"] != cursor["content_sha256"]
        or batch["decision"]
        != "formal_receipt_cursor_advanced_by_bounded_multi_leaf_batch_no_promotion"
        or service["runtime_binding"]["terminal_state"] != "bounded_complete"
        or counts["remaining_pending_formal_receipts"] != 11_023
        or counts["cumulative_formally_checked_candidates"] != 226
        or counts["candidate_promotions"] != 0
        or counts["rank_assignments"] != 0
    ):
        raise ValueError("authoritative operational state changed")
    observed = datetime.fromisoformat(observed_at)
    live_modified = datetime.fromisoformat(database_snapshot["modified_utc"])
    return {
        "observed_at": observed_at,
        "batch0003": {
            "binding": batch_binding,
            "cursor_binding": cursor_binding,
            "decision": batch["decision"],
            "executed_leaf_indices": batch["batch_leaf_catalog_indices"],
            "counts": counts,
            "first_remaining_blocker": cursor["first_remaining_blocker"],
            "complete_comparable_evidence": cursor["complete_comparable_evidence"],
        },
        "terminal_service": {
            "binding": service_binding,
            "decision": service["decision"],
            "terminal_state": service["runtime_binding"]["terminal_state"],
            "unique_formula_count": service["coverage"]["unique_formula_count"],
        },
        "live_scheduler_metadata_only": {
            "path": inventory["live_database"],
            "size_bytes": database_snapshot["size_bytes"],
            "modified_utc": live_modified.isoformat(),
            "age_seconds_at_observation": int((observed - live_modified).total_seconds()),
            "sqlite_opened": False,
            "wal_or_shm_opened": False,
            "lease_rows_read": False,
            "fresh_live_lease_claimed": False,
            "interpretation": "metadata_only_cannot_establish_current_lease_freshness",
        },
    }


def _scratch_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load(root / "configs/persistent_parallel_search_5090.json")
    profile = _load(root / "configs/resource_profile_5090.json")
    config["external_paid_llm_calls"] = False
    config["queue"] = {
        **config["queue"],
        "checkpoint_every_completions": 1,
        "lease_seconds": 1,
        "maximum_attempts": 2,
        "maximum_pending_work": 3,
    }
    config["budget"] = {"maximum_tasks": 3, "maximum_wall_seconds": 60}
    config["cpu"] = {"maximum_workers": 1}
    config["supervisor"] = {**config["supervisor"], "cpu_workers": 1, "gpu_workers": 0}
    profile["hardware"] = {**profile["hardware"], "gpu_memory_mib": 0}
    return config, profile


def _run_scratch(root: Path, scratch: Path) -> dict[str, Any]:
    repo_runs = (root / "runs").resolve()
    scratch = scratch.resolve()
    try:
        scratch.relative_to(repo_runs)
    except ValueError:
        pass
    else:
        raise ValueError("scratch campaign may not use repository runtime")
    if scratch.exists() and any(scratch.iterdir()):
        raise ValueError("scratch campaign directory must be empty")
    scratch.mkdir(parents=True, exist_ok=True)
    config, profile = _scratch_config(root)
    database = scratch / "caller-owned-recovery.sqlite"
    coordinator = PersistentParallelSearch(database, config, profile)
    payloads = [
        {
            "ordinal": index,
            "control_id": f"scratch-recovery-{index}",
            "synthetic_control_only": True,
        }
        for index in range(3)
    ]
    admission = coordinator.enqueue(payloads, lane="cpu", max_attempts=2)
    interrupted = coordinator.claim("cpu", "caller-owned-interrupted", lease_seconds=-1)
    if admission["accepted"] != 3 or interrupted is None or interrupted.attempt != 1:
        raise RuntimeError("scratch multi-task admission failed")
    resumed = PersistentParallelSearch(database, config, profile)
    recovery = resumed.recover_expired()
    completed = []
    for position in range(3):
        lease = resumed.claim("cpu", "caller-owned-replacement")
        if lease is None:
            raise RuntimeError("scratch task claim failed")
        expected_attempt = 2 if lease.work_id == interrupted.work_id else 1
        if lease.attempt != expected_attempt:
            raise RuntimeError("scratch recovery attempt lineage changed")
        result = {
            "ordinal": lease.ordinal,
            "synthetic_control_only": True,
            "scientific_pass": False,
        }
        if not resumed.finish(lease, "caller-owned-replacement", result):
            raise RuntimeError("scratch task completion failed")
        completed.append(
            {"ordinal": lease.ordinal, "attempt": lease.attempt, "work_id": lease.work_id}
        )
    checkpoint = resumed.checkpoint()
    telemetry = resumed.telemetry()
    if recovery != {"recovered": 1, "failed": 0} or telemetry["counts"] != {"succeeded": 3}:
        raise RuntimeError("scratch recovery terminal counts changed")
    return {
        "database_name": database.name,
        "database_inside_repository_runs": False,
        "tasks_admitted": admission,
        "recovery": recovery,
        "completed": sorted(completed, key=lambda row: row["ordinal"]),
        "terminal_counts": telemetry["counts"],
        "checkpoint": {
            "sequence": checkpoint["sequence"],
            "config_sha256": checkpoint["config_sha256"],
            "work_state_root": checkpoint["work_state_root"],
        },
        "workers": {"cpu": 1, "gpu": 0},
    }


def build_campaign(
    root: Path,
    scratch: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    observed_at: str,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    rows = _normalize_samples(samples, config["resource_admission"])
    live_state = _audit_live_state(root, config, observed_at)
    scratch_result = _run_scratch(root, scratch)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "decision": "pass_current_resources_admitted_isolated_three_task_recovery_complete",
        "resource_admission": {
            "contract": config["resource_admission"],
            "samples": rows,
            "maximum_cpu_utilization_percent": format(
                max(float(row["cpu_utilization_percent"]) for row in rows), ".1f"
            ),
            "minimum_available_ram_mib": min(row["available_ram_mib"] for row in rows),
            "all_samples_admitted": True,
        },
        "operational_audit": live_state,
        "scratch_recovery": scratch_result,
        "claims": {
            "production_scheduler_freshness_established": False,
            "production_namespace_written": False,
            "live_sqlite_opened": False,
            "external_process_signals": False,
            "gpu_or_cuda_access": False,
            "network_access": False,
            "observations_opened": False,
            "promotion_or_rank_write": False,
            "scientific_pass": False,
            "synthetic_control_only": True,
        },
        "bindings": {
            label: {"path": relative, "file_sha256": _file_sha(_resolve(root, relative))}
            for label, relative in (
                ("config", CONFIG_REL),
                ("source", SOURCE_REL),
                ("test", TEST_REL),
            )
        },
        "scope": (
            "current resource and immutable operational-state audit plus a caller-owned scratch "
            "three-task lease recovery control; no production scheduler or scientific result claim"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_campaign(value: Mapping[str, Any], root: Path) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != RESULT_SCHEMA or value.get(
        "content_sha256"
    ) != canonical_sha256(body):
        raise ValueError("scratch recovery campaign seal changed")
    if (
        value.get("decision")
        != "pass_current_resources_admitted_isolated_three_task_recovery_complete"
    ):
        raise ValueError("scratch recovery decision changed")
    config = load_config(root.resolve())
    expected_bindings = {
        label: {"path": relative, "file_sha256": _file_sha(_resolve(root, relative))}
        for label, relative in (
            ("config", CONFIG_REL),
            ("source", SOURCE_REL),
            ("test", TEST_REL),
        )
    }
    if value.get("bindings") != expected_bindings:
        raise ValueError("scratch recovery source binding changed")
    expected_audit = _audit_live_state(
        root.resolve(), config, value["operational_audit"]["observed_at"]
    )
    if value.get("operational_audit") != expected_audit:
        raise ValueError("scratch recovery operational snapshot changed")
    _normalize_samples(
        [
            {key: item for key, item in row.items() if key != "admitted"}
            for row in value["resource_admission"]["samples"]
        ],
        config["resource_admission"],
    )
    scratch = value["scratch_recovery"]
    if (
        scratch["recovery"] != {"recovered": 1, "failed": 0}
        or scratch["terminal_counts"] != {"succeeded": 3}
        or [row["attempt"] for row in scratch["completed"]] != [2, 1, 1]
        or scratch["workers"] != {"cpu": 1, "gpu": 0}
        or scratch["database_inside_repository_runs"] is not False
        or any(
            value["claims"][key] for key in value["claims"] if key not in {"synthetic_control_only"}
        )
        or value["claims"]["synthetic_control_only"] is not True
    ):
        raise ValueError("scratch recovery campaign boundary changed")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError("immutable scratch recovery artifact differs")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


__all__ = [
    "CONFIG_REL",
    "RESULT_REL",
    "build_campaign",
    "capture_resource_samples",
    "load_config",
    "validate_campaign",
    "write_immutable",
]
