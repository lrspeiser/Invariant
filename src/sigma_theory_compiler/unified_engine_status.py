"""Read-only, hash-bound status model for the local gravity-search engine.

The module deliberately does not expose a write-capable campaign connection.  A
snapshot has a deterministic ``core`` (for a fixed set of source revisions) and
an explicitly separate ``volatile`` hardware/freshness observation.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import urllib.parse
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-unified-engine-status-1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_RE = re.compile(r"(api.?key|password|secret|token|authorization)", re.IGNORECASE)
_HOST_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/Users/|/home/)", re.IGNORECASE)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_bound_json(project_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = (project_root / str(spec["path"])).resolve()
    root = project_root.resolve()
    if root not in path.parents:
        raise ValueError("status source escapes project root")
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != spec["file_sha256"]:
        raise ValueError(f"status source file hash mismatch: {spec['label']}")
    value = json.loads(raw)
    claimed = value.get("content_sha256")
    without_hash = dict(value)
    without_hash.pop("content_sha256", None)
    calculated = _sha(without_hash) if claimed is not None else _sha(value)
    if claimed is not None and (not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed)):
        raise ValueError(f"invalid content hash: {spec['label']}")
    if calculated != (claimed or spec["content_sha256"]) or calculated != spec["content_sha256"]:
        raise ValueError(f"status source content hash mismatch: {spec['label']}")
    return value


def _counts(rows: list[sqlite3.Row], key: str = "status") -> dict[str, int]:
    return {str(row[key]): int(row["count"]) for row in rows}


def _read_campaign_mode_ro(database: Path) -> dict[str, Any]:
    # URI mode=ro is the security boundary.  A normal sqlite connection is never
    # used here, and query_only adds a second fail-closed guard.
    uri = "file:" + urllib.parse.quote(database.resolve().as_posix(), safe="/:") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        campaign = connection.execute(
            "SELECT state, deadline_utc, max_tasks, tasks_started, tasks_succeeded, "
            "tasks_failed, max_cycles, cycles_completed, stop_reason FROM campaigns "
            "ORDER BY campaign_id LIMIT 1"
        ).fetchone()
        if campaign is None:
            raise ValueError("campaign watchdog database has no campaign")
        tasks = _counts(
            connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status ORDER BY status"
            ).fetchall()
        )
        candidates = _counts(
            connection.execute(
                "SELECT status, COUNT(*) AS count FROM candidates GROUP BY status ORDER BY status"
            ).fetchall()
        )
        evidence = _counts(
            connection.execute(
                "SELECT outcome AS status, COUNT(*) AS count FROM evidence "
                "GROUP BY outcome ORDER BY outcome"
            ).fetchall()
        )
        task_types = {
            (str(row["task_type"]), str(row["status"])): int(row["count"])
            for row in connection.execute(
                "SELECT task_type, status, COUNT(*) AS count FROM tasks "
                "GROUP BY task_type, status ORDER BY task_type, status"
            ).fetchall()
        }
        budget = connection.execute(
            "SELECT limit_microusd, reserved_microusd, spent_microusd, max_calls, "
            "calls_started, calls_completed FROM llm_budgets LIMIT 1"
        ).fetchone()
        latest = connection.execute("SELECT MAX(created_utc) AS latest FROM events").fetchone()
        connection.execute("COMMIT")
    finally:
        connection.close()
    return {
        "campaign": dict(campaign),
        "task_counts": tasks,
        "candidate_counts": candidates,
        "evidence_counts": evidence,
        "task_type_state_counts": task_types,
        "llm_budget": dict(budget) if budget else None,
        "latest_event_utc": latest["latest"] if latest else None,
    }


def _scheduler_lanes(watchdog: Mapping[str, Any], resource: Mapping[str, Any]) -> dict[str, Any]:
    production = resource["production_lanes"]
    states = watchdog["task_type_state_counts"]
    active_states = {"leased", "running"}
    lanes: dict[str, Any] = {}
    claimed_types: set[str] = set()
    for name in ("cpu_symbolic", "gpu_dense", "llm_research", "housekeeping"):
        lane = production[name]
        allowed = set(lane.get("task_types", []))
        claimed_types |= allowed
        capacity = int(lane.get("sustained_workers", lane.get("workers", 0)))
        running = sum(
            count for (task_type, state), count in states.items()
            if task_type in allowed and state in active_states
        )
        queued = sum(
            count for (task_type, state), count in states.items()
            if task_type in allowed and state == "queued"
        )
        lanes[name] = {
            "capacity": capacity,
            "running": running,
            "queued": queued,
            "scheduler_occupancy_fraction": running / capacity if capacity else None,
        }
    lanes["unclassified"] = {
        "capacity": None,
        "running": sum(
            count for (task_type, state), count in states.items()
            if task_type not in claimed_types and state in active_states
        ),
        "queued": sum(
            count for (task_type, state), count in states.items()
            if task_type not in claimed_types and state == "queued"
        ),
        "scheduler_occupancy_fraction": None,
    }
    return lanes


def sample_nvidia_smi() -> dict[str, Any]:
    """Take one bounded physical-GPU sample; never infer this from queue occupancy."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        line = result.stdout.strip().splitlines()[0]
        name, utilization, used, total, power = [part.strip() for part in line.split(",")]
        return {
            "availability": "available",
            "source": "nvidia-smi_nvml",
            "device_class": name,
            "utilization_percent": float(utilization),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
            "power_draw_watts": float(power),
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "availability": "unavailable",
            "source": "nvidia-smi_nvml",
            "reason": type(exc).__name__,
        }


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).astimezone(UTC)


def _assert_redacted(value: Any) -> None:
    encoded = json.dumps(value, sort_keys=True)
    if _HOST_PATH_RE.search(encoded):
        raise ValueError("snapshot leaks a host-specific path")
    for key in _walk_keys(value):
        if _SECRET_RE.search(key):
            raise ValueError("snapshot contains a secret-bearing field")


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def build_unified_snapshot(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    physical_gpu: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one consistent read-only snapshot from hash-bound source revisions."""
    root = project_root.resolve()
    sources = {spec["label"]: _read_bound_json(root, spec) for spec in config["sources"]}
    watchdog_path = (root / str(config["watchdog_database"])).resolve()
    if root not in watchdog_path.parents:
        raise ValueError("watchdog database escapes project root")
    watchdog = _read_campaign_mode_ro(watchdog_path)

    streaming = sources["billion_streaming"]
    promotion = sources["promotion_overlay"]
    parameter = sources["grammar_parameter_cells"]
    pareto = sources["evidence_pareto"]
    followup = sources["followup_service"]
    followup_queue = sources["followup_queue"]
    resource = sources["resource_profile"]

    blocker_gates: Counter[str] = Counter()
    for row in pareto["pareto_follow_up_queue"]:
        for blocker in row["blocker_taxonomy"]:
            if blocker["outcome_class"] == "blocked":
                blocker_gates[str(blocker["gate_id"])] += 1

    llm = watchdog["llm_budget"]
    llm_summary = {
        "configured_budget_usd": (llm["limit_microusd"] / 1_000_000) if llm else 0.0,
        "reserved_usd": (llm["reserved_microusd"] / 1_000_000) if llm else 0.0,
        "spent_usd": (llm["spent_microusd"] / 1_000_000) if llm else 0.0,
        "calls_started": int(llm["calls_started"]) if llm else 0,
        "calls_completed": int(llm["calls_completed"]) if llm else 0,
        "sealed_subsystems_spend_usd": {
            "billion_streaming": 0.0,
            "promotion_overlay": float(promotion["paid_llm_spend_usd"]),
            "grammar_parameter_cells": float(parameter["paid_llm_spend_usd"]),
            "evidence_pareto": float(pareto["paid_llm_spend_usd"]),
            "followup_service": float(followup["paid_llm_spend_usd"]),
        },
    }
    core = {
        "schema_version": SCHEMA_VERSION,
        "source_revisions": {
            label: {
                "file_sha256": spec["file_sha256"],
                "content_sha256": spec["content_sha256"],
                "freshness_class": "immutable_completed_snapshot",
                "stale_source_reason": None,
            }
            for label, spec in sorted((s["label"], s) for s in config["sources"])
        },
        "campaign_watchdog": {
            "read_contract": "sqlite_uri_mode_ro_plus_query_only_transaction",
            "state": watchdog["campaign"]["state"],
            "task_counts": watchdog["task_counts"],
            "candidate_counts": watchdog["candidate_counts"],
            "evidence_outcome_counts": watchdog["evidence_counts"],
            "normalized_evidence_outcomes": {
                "pass": watchdog["evidence_counts"].get("pass", 0),
                "reject": watchdog["evidence_counts"].get("reject", 0),
                "block": watchdog["evidence_counts"].get("unresolved", 0),
            },
            "scientific_outcome_semantics": {"pass": "pass", "reject": "reject", "unresolved": "block"},
            "deadline_utc": watchdog["campaign"]["deadline_utc"],
            "stop_reason": watchdog["campaign"]["stop_reason"],
            "budget": {
                "maximum_tasks": int(watchdog["campaign"]["max_tasks"]),
                "tasks_started": int(watchdog["campaign"]["tasks_started"]),
                "tasks_succeeded": int(watchdog["campaign"]["tasks_succeeded"]),
                "tasks_failed": int(watchdog["campaign"]["tasks_failed"]),
                "maximum_cycles": int(watchdog["campaign"]["max_cycles"]),
                "cycles_completed": int(watchdog["campaign"]["cycles_completed"]),
            },
        },
        "scheduler_lanes": _scheduler_lanes(watchdog, resource),
        "billion_formula_streaming": {
            "completed": streaming["execution"]["completed"],
            "source_formula_count": streaming["execution"]["formula_count"],
            "chunks": {
                "succeeded": streaming["execution"]["chunks_succeeded"],
                "total": streaming["execution"]["chunk_count"],
            },
            "sampled_static_stage": {
                "pass": streaming["portable_export"]["pass_count"],
                "ambiguous_block": streaming["portable_export"]["ambiguous_count"],
                "survivor_identities": streaming["portable_export"]["survivor_identity_count"],
                "normalized_outcomes": {
                    "pass": streaming["portable_export"]["pass_count"],
                    "reject": None,
                    "block": streaming["portable_export"]["ambiguous_count"],
                },
                "reject_count_reason": "not reported across heterogeneous Rust and CUDA screening phases",
            },
            "promotion_stage": {
                "lift_reject": streaming["promotion"]["covariant_lift_rejected"],
                "lift_block": streaming["promotion"]["covariant_lift_blocked"],
                "formal_reached": streaming["promotion"]["adm_dirac_principal_reached"],
            },
            "historical_physical_gpu_utilization_percent": {
                "mean": streaming["screening"]["resumed_wave"]["physical_gpu_utilization_mean_percent"],
                "peak": streaming["screening"]["resumed_wave"]["physical_gpu_utilization_peak_percent"],
            },
            "deadline": "completed_artifact_no_live_deadline",
        },
        "promotion_overlay": {
            "state": promotion["state"],
            "lift": {"pass": promotion["lift_passed_count"], "reject": promotion["upstream_terminal_candidate_count"]},
            "formal": {
                "pass": promotion["formal_passed_count"],
                "reject": promotion["formal_rejected_count"],
                "block": promotion["remaining_formal_blocked_count"],
            },
            "observational_opened": promotion["solar_opened_count"] + promotion["galaxy_opened_count"],
            "deadline": "completed_artifact_no_live_deadline",
        },
        "grammar_parameter_cells": {
            "task_state_counts": parameter["work_state_counts"],
            "scientific_decision_counts": parameter["decision_counts"],
            "normalized_scientific_outcomes": {"pass": 0, "reject": 0, "block": 6},
            "maximum_tasks": parameter["budget"]["maximum_tasks"],
            "deadline": "bounded_completed_artifact_no_live_deadline",
            "next_scaling_hook": parameter["next_scaling_hook"],
        },
        "evidence_pareto": {
            "candidate_decision_counts": pareto["candidate_decision_counts"],
            "normalized_candidate_outcomes": {"pass": 0, "reject": 0, "block": 6},
            "candidate_evidence_packet_counts": {
                key: value for key, value in pareto["evidence_packet_outcome_counts"].items() if key == "blocked"
            },
            "calibration_control_counts": pareto["calibration_outcome_counts"],
            "blocker_gate_counts": dict(sorted(blocker_gates.items())),
            "scalar_truth_score": "forbidden",
        },
        "followup_service": {
            "lifecycle": followup["lifecycle"],
            "packet_state_counts": followup["packet_state_counts"],
            "followup_decision_counts": followup_queue["followup_decision_counts"],
            "normalized_followup_outcomes": {"pass": 0, "reject": 0, "block": 10},
            "processed": followup["processed_count"],
            "deferred": followup["deferred_count"],
            "current_missing_evaluator_blockers": dict(sorted(Counter(
                packet["task_type"] for packet in followup["deferred_packets"]
            ).items())),
            "deadline": "bounded_waiting_service_no_live_deadline",
        },
        "llm": llm_summary,
        "data_seals": {
            "observations_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_in_streaming_promotion_grammar": False,
        },
        "cross_pipeline_total": {
            "status": "not_computed",
            "reason": "pipelines overlap candidates and use different gate semantics; per-stage exact counts must not be summed",
        },
    }
    timestamp = (now_utc or datetime.now(UTC)).astimezone(UTC)
    latest = _parse_utc(watchdog["latest_event_utc"])
    age = (timestamp - latest).total_seconds() if latest else None
    threshold = int(config["watchdog_stale_after_seconds"])
    volatile = {
        "sampled_at_utc": timestamp.isoformat(),
        "physical_gpu": dict(physical_gpu) if physical_gpu is not None else sample_nvidia_smi(),
        "campaign_watchdog_freshness": {
            "latest_event_utc": watchdog["latest_event_utc"],
            "age_seconds": age,
            "stale": age is None or age > threshold,
            "stale_source_reason": (
                "no_events" if age is None else
                f"latest_event_older_than_{threshold}_seconds" if age > threshold else None
            ),
        },
        "deadline_state": (
            "unavailable" if not _parse_utc(watchdog["campaign"]["deadline_utc"])
            else "expired" if timestamp > _parse_utc(watchdog["campaign"]["deadline_utc"])
            else "open"
        ),
        "note": "volatile fields are excluded from core_content_sha256",
    }
    snapshot = {"core": core, "core_content_sha256": _sha(core), "volatile": volatile}
    _assert_redacted(snapshot)
    return snapshot


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("content_sha256", None)
    if claimed != _sha(value):
        raise ValueError("unified status config content hash mismatch")
    value["content_sha256"] = claimed
    return value


def write_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
