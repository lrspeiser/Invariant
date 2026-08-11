"""Read-only, hash-bound status model for the local gravity-search engine.

The module deliberately does not expose a write-capable campaign connection.  A
snapshot has a deterministic ``core`` (for a fixed set of source revisions) and
an explicitly separate ``volatile`` hardware/freshness observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import urllib.parse
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
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
        task_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        queued_task_schedule = (
            [
                {
                    "task_type": str(row["task_type"]),
                    "not_before_utc": row["not_before_utc"],
                    "count": int(row["count"]),
                }
                for row in connection.execute(
                    "SELECT task_type, not_before_utc, COUNT(*) AS count FROM tasks "
                    "WHERE status = 'queued' GROUP BY task_type, not_before_utc "
                    "ORDER BY task_type, not_before_utc"
                ).fetchall()
            ]
            if "not_before_utc" in task_columns
            else []
        )
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
        "queued_task_schedule": queued_task_schedule,
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


def _scheduler_readiness(
    watchdog: Mapping[str, Any],
    resource: Mapping[str, Any],
    sampled_at: datetime,
) -> dict[str, Any]:
    """Classify queued work without treating future not-before tasks as runnable."""
    production = resource["production_lanes"]
    core_lanes = _scheduler_lanes(watchdog, resource)
    schedules = list(watchdog.get("queued_task_schedule", []))
    claimed_types: set[str] = set()
    readiness: dict[str, Any] = {}
    for name in ("cpu_symbolic", "gpu_dense", "llm_research", "housekeeping"):
        allowed = set(production[name].get("task_types", []))
        claimed_types |= allowed
        relevant = [
            (row, _parse_utc(row.get("not_before_utc")))
            for row in schedules
            if row["task_type"] in allowed
        ]
        delayed_rows = [
            (row, ready_at)
            for row, ready_at in relevant
            if ready_at is not None and ready_at > sampled_at
        ]
        delayed = sum(int(row["count"]) for row, _ in delayed_rows)
        future_times = sorted(ready_at for _, ready_at in delayed_rows)
        readiness[name] = {
            "queued_total": core_lanes[name]["queued"],
            "runnable_now": core_lanes[name]["queued"] - delayed,
            "delayed_until_not_before": delayed,
            "earliest_future_not_before_utc": (
                future_times[0].isoformat() if future_times else None
            ),
        }
    relevant = [
        (row, _parse_utc(row.get("not_before_utc")))
        for row in schedules
        if row["task_type"] not in claimed_types
    ]
    delayed_rows = [
        (row, ready_at)
        for row, ready_at in relevant
        if ready_at is not None and ready_at > sampled_at
    ]
    delayed = sum(int(row["count"]) for row, _ in delayed_rows)
    future_times = sorted(ready_at for _, ready_at in delayed_rows)
    readiness["unclassified"] = {
        "queued_total": core_lanes["unclassified"]["queued"],
        "runnable_now": core_lanes["unclassified"]["queued"] - delayed,
        "delayed_until_not_before": delayed,
        "earliest_future_not_before_utc": (
            future_times[0].isoformat() if future_times else None
        ),
    }
    return readiness


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
    timestamp = (now_utc or datetime.now(UTC)).astimezone(UTC)
    sources = {spec["label"]: _read_bound_json(root, spec) for spec in config["sources"]}
    watchdog_path = (root / str(config["watchdog_database"])).resolve()
    if root not in watchdog_path.parents:
        raise ValueError("watchdog database escapes project root")
    watchdog = _read_campaign_mode_ro(watchdog_path)

    streaming = sources["billion_streaming"]
    promotion = sources["promotion_overlay"]
    parameter = sources["grammar_parameter_cells"]
    parameter_expansion = sources["grammar_parameter_cell_expansion_service"]
    grammar_seed_manifest = sources["grammar_v3_seed_manifest"]
    parameter_manifest = sources["grammar_parameter_cell_manifest"]
    parameter_compilation = sources["grammar_parameter_cell_compilation"]
    formal_preflight = sources["grammar_v3_formal_preflight"]
    promotion_admission = sources["grammar_v3_promotion_admission"]
    g2_candidate_formal = sources["grammar_v3_g2_candidate_formal"]
    g2_nonmaximal_followup = sources[
        "grammar_v3_g2_nonmaximal_positive_mass_followup"
    ]
    g2_solar_readiness = sources["grammar_v3_g2_solar_readiness"]
    g2_solar_heldout_transfer = sources[
        "grammar_v3_g2_solar_heldout_transfer"
    ]
    scalable_campaign_epoch = sources["scalable_campaign_epoch"]
    scalable_future_parameter_chunk = sources["scalable_future_parameter_chunk"]
    scalable_future_formal_preflight = sources["scalable_future_formal_preflight"]
    future_aether_formal_followup = sources["future_aether_formal_followup"]
    future_aether_constraint_followup = sources["future_aether_constraint_followup"]
    future_g3_domain_followup = sources["future_g3_domain_followup"]
    future_g3_action_bound_followup = sources["future_g3_action_bound_followup"]
    future_candidate_action_dossier = sources["future_candidate_action_dossier"]
    g3_candidate_formal = sources["grammar_v3_g3_candidate_formal"]
    g4_scalable_formal_followup = sources[
        "grammar_v3_g4_scalable_formal_followup"
    ]
    aether_candidate_formal = sources["grammar_v3_aether_candidate_formal"]
    scalable_structural_metrics = sources["scalable_structural_metrics"]
    scalable_explanation_dossiers = sources["scalable_explanation_dossiers"]
    pareto = sources["evidence_pareto"]
    followup = sources["followup_service"]
    followup_queue = sources["followup_queue"]
    resource = sources["resource_profile"]
    llm_adapter = sources["llm_proposal_adapter"]
    llm_bridge = sources["llm_campaign_bridge"]
    g4_solar = sources["g4_solar_evaluator"]
    g4_solar_execution = sources["g4_solar_execution"]
    g4_galaxy = sources["g4_galaxy_evaluator"]
    g4_galaxy_execution = sources["g4_galaxy_execution"]
    typed_admission = sources["typed_dsl_admission"]
    compiler_registry = sources["compiler_registry_bridge"]
    local_formula_epoch = sources["reviewed_local_formula_epoch"]
    local_formula_service = sources["reviewed_local_formula_service"]
    g4_galaxy_forward_model = sources["g4_galaxy_forward_model"]
    g4_galaxy_branch_distance = sources["g4_galaxy_branch_distance"]
    g4_galaxy_calibration_evaluation = sources[
        "g4_galaxy_calibration_evaluation"
    ]
    g4_galaxy_prediction_contract_transform = sources[
        "g4_galaxy_prediction_contract_transform"
    ]
    g4_galaxy_manifest_bundle_tooling = sources[
        "g4_galaxy_manifest_bundle_tooling"
    ]
    g4_galaxy_source_registry_admission = sources[
        "g4_galaxy_source_registry_admission"
    ]
    quartic_tc2_quadratic_deltak = sources[
        "quartic_tc2_quadratic_deltak_extension"
    ]
    quartic_tc2_diagonal_third_jet = sources[
        "quartic_tc2_diagonal_third_jet"
    ]
    quartic_tc2_mixed_third_jet_chunk = sources[
        "quartic_tc2_mixed_third_jet_chunk"
    ]
    quartic_tc2_mixed_third_jet_chunk_64 = sources[
        "quartic_tc2_mixed_third_jet_chunk_64"
    ]
    quartic_tc2_mixed_third_jet_checkpoint = sources[
        "quartic_tc2_mixed_third_jet_checkpoint"
    ]
    quartic_tc2_mixed_third_jet_continuation_status = sources[
        "quartic_tc2_mixed_third_jet_continuation_status"
    ]
    if (
        scalable_structural_metrics.get("candidate_count") != 163
        or scalable_structural_metrics.get("alias_count") != 93
        or scalable_structural_metrics.get("formal_decision_counts")
        != {"blocked": 158, "pass": 3, "reject": 2}
        or scalable_structural_metrics.get("structural_measurement_counts")
        != {"measured": 163}
        or scalable_structural_metrics.get("simplicity_pareto_front", {}).get(
            "candidate_count"
        )
        != 2
        or scalable_structural_metrics.get("observational_data_opened") is not False
        or scalable_structural_metrics.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("scalable structural metrics export is inconsistent")
    if (
        scalable_explanation_dossiers.get("candidate_count") != 163
        or scalable_explanation_dossiers.get("alias_count") != 93
        or scalable_explanation_dossiers.get("formal_decision_counts")
        != {"blocked": 158, "pass": 3, "reject": 2}
        or scalable_explanation_dossiers.get("hierarchy_node_status_counts")
        != {
            "blocked": 321,
            "calibration_only": 163,
            "proven": 166,
            "rejected": 2,
        }
        or scalable_explanation_dossiers.get("observational_authorization") is not False
        or scalable_explanation_dossiers.get("observational_data_opened") is not False
        or scalable_explanation_dossiers.get("paid_llm_spend_usd") != 0.0
    ):
        raise ValueError("scalable explanation dossier bridge is inconsistent")
    if (
        followup.get("lifecycle") != "idle"
        or followup.get("processed_count") != 10
        or followup.get("deferred_count") != 0
        or followup.get("packet_state_counts")
        != {"deferred_missing_evaluator": 0, "succeeded": 10}
        or followup.get("candidate_scientific_decisions_changed") != 1
        or followup.get("reviewed_evaluator_invocation_count") != 10
        or followup.get("missing_evaluator_executions") != 0
        or followup.get("deferred_packets") != []
        or followup_queue.get("followup_decision_counts")
        != {"blocked": 8, "pass": 2}
        or followup_queue.get("work_state_counts") != {"succeeded": 10}
        or followup_queue.get("candidate_scientific_decisions_changed") != 1
        or followup_queue.get("reviewed_evaluator_invocation_count") != 10
        or followup_queue.get("missing_evaluator_count") != 0
        or followup.get("queue_registry_root_sha256")
        != followup_queue.get("queue_registry_root_sha256")
        or followup.get("completed_work_records_root_sha256")
        != followup_queue.get("work_records_root_sha256")
    ):
        raise ValueError("final grammar-v3 follow-up epoch is inconsistent")
    if (
        parameter_expansion.get("execution_enabled") is not True
        or parameter_expansion.get("parameter_cell_count") != 6
        or parameter_expansion.get("chunk_count") != 3
        or parameter_expansion.get("work_state_counts") != {"succeeded": 3}
        or parameter_expansion.get("decision_counts") != {"blocked": 6}
        or parameter_expansion.get("paid_llm_spend_usd") != 0.0
        or parameter_expansion.get("observational_data_opened") is not False
        or parameter_expansion.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
        or parameter_expansion.get("source_manifest_content_sha256")
        != grammar_seed_manifest.get("content_sha256")
    ):
        raise ValueError("grammar parameter-cell expansion service is inconsistent")
    if (
        parameter_manifest.get("parameter_cell_count") != 256
        or parameter_manifest.get("family_cell_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": 128,
            "CONFORMAL_G4_PHI_SCALAR_TENSOR": 32,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
            "KESSENCE_G2_CONVEX": 64,
        }
        or len(parameter_manifest.get("chunks", [])) != 8
        or parameter_manifest.get("formal_evaluation_performed") is not False
        or parameter_manifest.get("scientific_decision_counts") != {}
        or parameter_manifest.get("evaluator_semantics_changed") is not False
        or parameter_manifest.get("negative_control_counts") != {"reject": 6}
        or parameter_manifest.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
        or parameter_manifest.get("source_seed_manifest_content_sha256")
        != grammar_seed_manifest.get("content_sha256")
    ):
        raise ValueError("grammar parameter-cell manifest is inconsistent")
    if (
        parameter_compilation.get("input_parameter_cell_count") != 256
        or parameter_compilation.get("compiled_action_ir_count") != 256
        or parameter_compilation.get("unique_candidate_count") != 163
        or parameter_compilation.get("equivalent_duplicate_count") != 93
        or parameter_compilation.get("candidate_decision_counts")
        != {"blocked": 0, "pass": 163, "reject": 0}
        or parameter_compilation.get("formal_decision_counts") != {}
        or parameter_compilation.get("expensive_formal_campaign_run") is not False
        or parameter_compilation.get("cell_disposition_counts")
        != {"compiled_representative": 163, "deduplicated_equivalent": 93}
        or set(parameter_compilation.get("structural_gate_pass_counts", {}).values())
        != {256}
        or parameter_compilation.get("negative_control_counts") != {"reject": 5}
        or parameter_compilation.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
        or parameter_compilation.get("parameter_cell_manifest_binding", {}).get(
            "content_sha256"
        )
        != parameter_manifest.get("content_sha256")
    ):
        raise ValueError("grammar parameter-cell compilation is inconsistent")
    if (
        formal_preflight.get("execution_enabled") is not True
        or formal_preflight.get("candidate_count") != 163
        or formal_preflight.get("work_state_counts") != {"succeeded": 163}
        or formal_preflight.get("decision_counts") != {"blocked": 1, "pass": 162}
        or formal_preflight.get("gate_counts")
        != {
            "family_prerequisite": {"blocked": 1, "pass": 162},
            "receipt_binding": {"pass": 163},
        }
        or formal_preflight.get("family_decision_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
            "CONFORMAL_G4_PHI_SCALAR_TENSOR": {"blocked": 1},
            "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
            "KESSENCE_G2_CONVEX": {"pass": 2},
        }
        or formal_preflight.get("candidate_registry_root_sha256")
        != parameter_compilation.get("unique_candidate_registry_root_sha256")
        or formal_preflight.get("compilation_campaign_content_sha256")
        != parameter_compilation.get("content_sha256")
        or formal_preflight.get("expensive_adm_or_global_energy_run") is not False
        or formal_preflight.get("paid_llm_spend_usd") != 0.0
        or formal_preflight.get("observational_data_opened") is not False
        or formal_preflight.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("grammar-v3 formal preflight is inconsistent")
    if (
        promotion_admission.get("execution_enabled") is not True
        or promotion_admission.get("preflight_candidate_count") != 163
        or promotion_admission.get("preflight_pass_count") != 162
        or promotion_admission.get("preflight_blocked_excluded_count") != 1
        or promotion_admission.get("eligible_candidate_count") != 162
        or promotion_admission.get("work_state_counts") != {"succeeded": 162}
        or promotion_admission.get("decision_counts") != {"pass": 162}
        or promotion_admission.get("family_decision_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
            "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
            "KESSENCE_G2_CONVEX": {"pass": 2},
        }
        or promotion_admission.get("target_queue_counts")
        != {
            "grammar_v3_aether_candidate_adm_formal": 128,
            "grammar_v3_g2_candidate_adm_formal": 2,
            "grammar_v3_g3_candidate_adm_formal": 32,
        }
        or set(promotion_admission.get("target_queue_registry_roots", {}))
        != {
            "grammar_v3_aether_candidate_adm_formal",
            "grammar_v3_g2_candidate_adm_formal",
            "grammar_v3_g3_candidate_adm_formal",
        }
        or promotion_admission.get("downstream_expensive_execution_started")
        is not False
        or promotion_admission.get("preflight_status_binding")
        != {
            "content_sha256": formal_preflight.get("content_sha256"),
            "file_sha256": "03387c868b9074d71ca79691ed92d0ba293dffea2dfb3169d342c67c1a4fc210",
            "path": "runs/engine/grammar-v3-formal-preflight-status.json",
        }
        or promotion_admission.get("preflight_config_binding")
        != {
            "file_sha256": "48253e2fefec23887435b567e1ee52c8d4e7b7257933980d9a9110684b33c7b2",
            "path": "configs/grammar_v3_formal_preflight_service.json",
        }
        or promotion_admission.get("paid_llm_spend_usd") != 0.0
        or promotion_admission.get("observational_data_opened") is not False
        or promotion_admission.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("grammar-v3 promotion admission is inconsistent")
    if (
        g2_candidate_formal.get("execution_enabled") is not True
        or g2_candidate_formal.get("candidate_count") != 2
        or g2_candidate_formal.get("work_state_counts") != {"succeeded": 2}
        or g2_candidate_formal.get("decision_counts") != {"blocked": 2}
        or g2_candidate_formal.get("full_formal_pass_count") != 0
        or g2_candidate_formal.get("blocker_counts")
        != {"hash_bound_general_nonmaximal_positive_mass_theorem": 2}
        or g2_candidate_formal.get("general_nonmaximal_global_positive_mass_proved")
        is not False
        or g2_candidate_formal.get("gate_counts", {}).get(
            "candidate_action_preflight_admission_binding"
        )
        != {"pass": 2}
        or g2_candidate_formal.get("gate_counts", {}).get(
            "restricted_maximal_slice_positive_mass"
        )
        != {"pass": 2}
        or g2_candidate_formal.get("gate_counts", {}).get(
            "general_nonmaximal_positive_mass"
        )
        != {"blocked": 2}
        or g2_candidate_formal.get("promotion_status_binding", {}).get(
            "content_sha256"
        )
        != promotion_admission.get("content_sha256")
        or g2_candidate_formal.get("paid_llm_spend_usd") != 0.0
        or g2_candidate_formal.get("observational_data_opened") is not False
        or g2_candidate_formal.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("grammar-v3 G2 candidate formal service is inconsistent")
    if (
        g2_nonmaximal_followup.get("candidate_count") != 2
        or g2_nonmaximal_followup.get("decision_counts") != {"pass": 2}
        or g2_nonmaximal_followup.get("full_formal_pass_count") != 2
        or g2_nonmaximal_followup.get(
            "general_nonmaximal_positive_mass_pass_count"
        )
        != 2
        or g2_nonmaximal_followup.get("solar_bundle_count") != 0
        or g2_nonmaximal_followup.get("observational_data_opened") is not False
        or g2_nonmaximal_followup.get("paid_llm_spend_usd") != 0.0
        or g2_nonmaximal_followup.get("source_bindings", {})
        .get("g2_status", {})
        .get("content_sha256")
        != g2_candidate_formal.get("content_sha256")
        or len(g2_nonmaximal_followup.get("candidate_records", [])) != 2
        or any(
            record.get("decision") != "pass"
            or record.get("previous_blocker_closed")
            != "hash_bound_general_nonmaximal_positive_mass_theorem"
            or record.get("actual_initial_data_set_instantiated") is not False
            or record.get("cell_preservation_or_global_evolution_proved") is not False
            or record.get("nonlinear_asymptotic_stability_proved") is not False
            for record in g2_nonmaximal_followup.get("candidate_records", [])
        )
    ):
        raise ValueError("grammar-v3 G2 nonmaximal follow-up is inconsistent")
    if (
        g2_solar_readiness.get("candidate_count") != 2
        or g2_solar_readiness.get("decision_counts") != {"blocked": 2}
        or g2_solar_readiness.get("candidate_analytic_prediction_pass_count") != 2
        or g2_solar_readiness.get("conditional_static_source_class_pass_count")
        != 2
        or g2_solar_readiness.get("real_source_registration_pass_count") != 0
        or g2_solar_readiness.get("real_solar_bundle_count") != 0
        or g2_solar_readiness.get("real_solar_bundle_admissible_count") != 0
        or g2_solar_readiness.get("observational_data_opened") is not False
        or g2_solar_readiness.get("paid_llm_spend_usd") != 0.0
        or len(g2_solar_readiness.get("candidate_records", [])) != 2
        or any(
            record.get("decision") != "blocked"
            or record.get("candidate_analytic_prediction_status")
            != "pass_on_exact_constant_phi_branch"
            or record.get("real_solar_readiness", {}).get(
                "observational_inputs_opened_by_this_audit"
            )
            is not False
            or len(
                record.get("real_solar_readiness", {}).get(
                    "missing_registration_fields", []
                )
            )
            != 10
            for record in g2_solar_readiness.get("candidate_records", [])
        )
    ):
        raise ValueError("grammar-v3 G2 Solar readiness is inconsistent")
    if (
        g2_solar_heldout_transfer.get("candidate_count") != 2
        or g2_solar_heldout_transfer.get("decision_counts") != {"blocked": 2}
        or g2_solar_heldout_transfer.get("registration_advance_per_candidate")
        != {
            "after_missing_field_count": 4,
            "before_missing_field_count": 10,
            "filled_field_count": 6,
            "filled_fields": [
                "candidate_specific_real_source_contract_sha256",
                "candidate_specific_evaluator_descriptor_sha256",
                "training_only_initial_state_sha256",
                "frozen_nuisance_likelihood_stopping_rule_sha256",
                "action_bound_prediction_bundle_descriptor_sha256",
                "action_bound_prediction_bundle_file_sha256",
            ],
            "remaining_fields": [
                "source_branch_domain_instantiation_sha256",
                "held_out_split_commitment_sha256",
                "selected_primary_record_roots_sha256",
                "observation_opening_authorization_sha256",
            ],
        }
        or g2_solar_heldout_transfer.get("primary_record_access_count") != 0
        or g2_solar_heldout_transfer.get("held_out_target_access_count") != 0
        or g2_solar_heldout_transfer.get("real_data_pass_count") != 0
        or g2_solar_heldout_transfer.get("observational_authorization") is not False
        or g2_solar_heldout_transfer.get("observational_data_opened") is not False
        or g2_solar_heldout_transfer.get("dark_matter_or_halo_inputs") is not False
        or g2_solar_heldout_transfer.get("redshift_distance_inputs") is not False
        or g2_solar_heldout_transfer.get("paid_llm_spend_usd") != 0.0
        or g2_solar_heldout_transfer.get("source_bindings", {})
        .get("g2_readiness", {})
        .get("content_sha256")
        != g2_solar_readiness.get("content_sha256")
        or len(g2_solar_heldout_transfer.get("candidate_registrations", [])) != 2
        or any(
            record.get("evaluator_result", {}).get("decision") != "blocked"
            or record.get("candidate_use_authorized") is not False
            or record.get("observational_data_opened") is not False
            or record.get("real_data_pass") is not False
            or len(record.get("filled_registration_fields", [])) != 6
            or len(record.get("remaining_registration_fields", [])) != 4
            for record in g2_solar_heldout_transfer.get(
                "candidate_registrations", []
            )
        )
    ):
        raise ValueError("grammar-v3 G2 Solar held-out transfer is inconsistent")
    if (
        scalable_campaign_epoch.get("stage_count") != 10
        or scalable_campaign_epoch.get("sealed_epoch_counts")
        != {
            "aliases": 93,
            "decisions": {"blocked": 158, "pass": 3, "reject": 2},
            "parameter_cells": 256,
            "unique_candidates": 163,
        }
        or scalable_campaign_epoch.get("next_epoch_readiness", {}).get("state")
        != "blocked"
        or scalable_campaign_epoch.get("next_epoch_readiness", {}).get(
            "scientific_compilation_started"
        )
        is not False
        or scalable_campaign_epoch.get("paid_llm_spend_usd") != 0.0
    ):
        raise ValueError("scalable campaign epoch is inconsistent")
    if (
        scalable_future_parameter_chunk.get("input_cell_count") != 32
        or scalable_future_parameter_chunk.get("disposition_counts")
        != {
            "admitted_new_candidate": 19,
            "deduplicated_existing_candidate": 13,
        }
        or scalable_future_parameter_chunk.get("next_blocker")
        != "reviewed_formal_preflight_not_run_for_new_candidates"
        or scalable_future_parameter_chunk.get("paid_llm_spend_usd") != 0.0
        or scalable_future_parameter_chunk.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("scalable future parameter chunk is inconsistent")
    if (
        scalable_future_formal_preflight.get("candidate_count") != 19
        or scalable_future_formal_preflight.get("source_input_cell_count") != 32
        or scalable_future_formal_preflight.get("source_new_candidate_count") != 19
        or scalable_future_formal_preflight.get(
            "source_deduplicated_candidate_count"
        )
        != 13
        or scalable_future_formal_preflight.get("family_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": 16,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
        }
        or scalable_future_formal_preflight.get("decision_counts")
        != {"blocked": 3, "pass": 14, "reject": 2}
        or scalable_future_formal_preflight.get("family_decision_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": {"pass": 14, "reject": 2},
            "CUBIC_HORNDESKI_G3_WEAK_CELL": {"blocked": 3},
        }
        or scalable_future_formal_preflight.get("first_blocker_counts")
        != {
            "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing": 3,
            "nonpositive_spin0_principal_numerator_c123": 2,
        }
        or scalable_future_formal_preflight.get("formal_preflight_completed")
        is not True
        or scalable_future_formal_preflight.get(
            "full_candidate_specific_formal_completion_claimed"
        )
        is not False
        or scalable_future_formal_preflight.get("promotion")
        != {
            "automatic_downstream_enqueue_performed": False,
            "blocked_pending_exact_domain_registration": 3,
            "eligible_for_candidate_specific_formal_queue": 14,
            "rejected_before_candidate_specific_formal_queue": 2,
        }
        or scalable_future_formal_preflight.get("source_status_binding", {}).get(
            "content_sha256"
        )
        != scalable_future_parameter_chunk.get("content_sha256")
        or scalable_future_formal_preflight.get("observational_data_opened")
        is not False
        or scalable_future_formal_preflight.get("dark_matter_or_halo_inputs")
        is not False
        or scalable_future_formal_preflight.get("redshift_distance_inputs")
        is not False
        or scalable_future_formal_preflight.get("paid_llm_spend_usd") != 0.0
        or len(scalable_future_formal_preflight.get("candidate_records", [])) != 19
    ):
        raise ValueError("scalable future formal preflight is inconsistent")
    if (
        future_aether_formal_followup.get("candidate_count") != 14
        or future_aether_formal_followup.get("input_preflight_pass_count") != 14
        or future_aether_formal_followup.get("decision_counts") != {"blocked": 14}
        or future_aether_formal_followup.get("formal_pass_count") != 0
        or future_aether_formal_followup.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_formal_followup.get(
            "exact_negative_local_twist_witness_count"
        )
        != 14
        or future_aether_formal_followup.get("witness_tilt_squared_counts")
        != {"1": 8, "2": 4, "8": 2}
        or future_aether_formal_followup.get("global_tilt_strata_counts")
        != {
            "finite_characteristic_foliation_present": 13,
            "globally_noncharacteristic_for_finite_unit_tilt": 1,
        }
        or future_aether_formal_followup.get("first_blocker_counts")
        != {"full_constraint_embedding_of_negative_static_twist_jet": 14}
        or future_aether_formal_followup.get("reviewed_adapter_replay_count") != 6
        or future_aether_formal_followup.get("source_preflight_binding", {}).get(
            "content_sha256"
        )
        != scalable_future_formal_preflight.get("content_sha256")
        or future_aether_formal_followup.get(
            "full_candidate_specific_formal_completion_claimed"
        )
        is not False
        or future_aether_formal_followup.get(
            "automatic_downstream_enqueue_performed"
        )
        is not False
        or future_aether_formal_followup.get("observational_data_opened") is not False
        or future_aether_formal_followup.get("dark_matter_or_halo_inputs") is not False
        or future_aether_formal_followup.get("redshift_distance_inputs") is not False
        or future_aether_formal_followup.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_formal_followup.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether formal follow-up is inconsistent")
    if (
        future_aether_constraint_followup.get("candidate_count") != 14
        or future_aether_constraint_followup.get("decision_counts")
        != {"blocked": 14}
        or future_aether_constraint_followup.get(
            "explicit_affine_ansatz_constraint_reject_count"
        )
        != 14
        or future_aether_constraint_followup.get(
            "nonzero_Hamiltonian_constraint_residual_count"
        )
        != 14
        or future_aether_constraint_followup.get(
            "nonzero_momentum_constraint_residual_count"
        )
        != 14
        or future_aether_constraint_followup.get(
            "undefined_AE_boundary_contribution_count"
        )
        != 14
        or future_aether_constraint_followup.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_constraint_followup.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_constraint_followup.get("formal_pass_count") != 0
        or future_aether_constraint_followup.get("first_blocker_counts")
        != {
            "constraint_satisfying_asymptotically_Euclidean_completion_of_negative_twist_witness": 14
        }
        or future_aether_constraint_followup.get("source_followup_binding", {}).get(
            "content_sha256"
        )
        != future_aether_formal_followup.get("content_sha256")
        or future_aether_constraint_followup.get("observational_data_opened") is not False
        or future_aether_constraint_followup.get("dark_matter_or_halo_inputs") is not False
        or future_aether_constraint_followup.get("redshift_distance_inputs") is not False
        or future_aether_constraint_followup.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_constraint_followup.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether constraint follow-up is inconsistent")
    if (
        future_g3_domain_followup.get("candidate_count") != 3
        or future_g3_domain_followup.get("decision_counts") != {"blocked": 3}
        or future_g3_domain_followup.get("all_direction_single_center_pass_count")
        != 3
        or future_g3_domain_followup.get("full_Delta_N_derivation_pass_count") != 3
        or future_g3_domain_followup.get("nonzero_componentwise_box_pass_count")
        != 0
        or future_g3_domain_followup.get("uniform_principal_common_cone_pass_count")
        != 0
        or future_g3_domain_followup.get("uniform_Delta_N_coercivity_pass_count")
        != 0
        or future_g3_domain_followup.get("periodic_distributed_Dirac_pass_count")
        != 0
        or future_g3_domain_followup.get("asymptotically_flat_Dirac_pass_count")
        != 0
        or future_g3_domain_followup.get("full_formal_pass_count") != 0
        or future_g3_domain_followup.get("first_blocker_counts")
        != {
            "candidate_bound_nonzero_componentwise_normalized_local_jet_box_values": 3
        }
        or future_g3_domain_followup.get("source_bindings", {})
        .get("preflight", {})
        .get("content_sha256")
        != scalable_future_formal_preflight.get("content_sha256")
        or future_g3_domain_followup.get("observational_data_opened") is not False
        or future_g3_domain_followup.get("dark_matter_or_halo_inputs") is not False
        or future_g3_domain_followup.get("redshift_distance_inputs") is not False
        or future_g3_domain_followup.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_domain_followup.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 domain follow-up is inconsistent")
    if (
        future_g3_action_bound_followup.get("candidate_count") != 3
        or future_g3_action_bound_followup.get("decision_counts") != {"blocked": 3}
        or future_g3_action_bound_followup.get("domain_registration_filled_field_count")
        != 36
        or future_g3_action_bound_followup.get("domain_registration_missing_field_count")
        != 0
        or future_g3_action_bound_followup.get("nonzero_componentwise_box_pass_count")
        != 3
        or future_g3_action_bound_followup.get(
            "uniform_principal_common_cone_pass_count"
        )
        != 3
        or future_g3_action_bound_followup.get("full_Delta_N_derivation_pass_count")
        != 3
        or future_g3_action_bound_followup.get("uniform_Delta_N_coercivity_pass_count")
        != 3
        or future_g3_action_bound_followup.get(
            "periodic_distributed_Dirac_pass_count"
        )
        != 3
        or future_g3_action_bound_followup.get("asymptotically_flat_Dirac_pass_count")
        != 0
        or future_g3_action_bound_followup.get("global_energy_pass_count") != 0
        or future_g3_action_bound_followup.get("full_formal_pass_count") != 0
        or future_g3_action_bound_followup.get("first_blocker_counts")
        != {"asymptotically_flat_or_global_energy_domain_missing": 3}
        or future_g3_action_bound_followup.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_domain_followup.get("content_sha256")
        or future_g3_action_bound_followup.get("observational_data_opened") is not False
        or future_g3_action_bound_followup.get("dark_matter_or_halo_inputs") is not False
        or future_g3_action_bound_followup.get("redshift_distance_inputs") is not False
        or future_g3_action_bound_followup.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_action_bound_followup.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 action-bound follow-up is inconsistent")
    if (
        future_candidate_action_dossier.get("candidate_count") != 19
        or future_candidate_action_dossier.get("family_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": 16,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
        }
        or future_candidate_action_dossier.get("decision_counts")
        != {"blocked": 17, "reject": 2}
        or future_candidate_action_dossier.get("ranked_candidate_count") != 0
        or future_candidate_action_dossier.get("observational_authorization")
        is not False
        or future_candidate_action_dossier.get("observational_data_opened") is not False
        or future_candidate_action_dossier.get("paid_llm_spend_usd") != 0.0
        or future_candidate_action_dossier.get("source_roots", {}).get(
            "preflight_content_sha256"
        )
        != scalable_future_formal_preflight.get("content_sha256")
        or future_candidate_action_dossier.get("source_roots", {}).get(
            "aether_followup_content_sha256"
        )
        != future_aether_constraint_followup.get("content_sha256")
        or future_candidate_action_dossier.get("source_roots", {}).get(
            "g3_followup_content_sha256"
        )
        != future_g3_action_bound_followup.get("content_sha256")
        or len(future_candidate_action_dossier.get("dossiers", [])) != 19
        or any(
            record.get("comparison_contract", {}).get("rank") is not None
            or record.get("comparison_contract", {}).get("rank_eligible") is not False
            or not record.get("action", {}).get("ordered_operator_densities")
            or record.get("action", {})
            .get("human_readable_action", {})
            .get("display_kind")
            != "verbatim_ordered_covariant_density_concatenation"
            for record in future_candidate_action_dossier.get("dossiers", [])
        )
    ):
        raise ValueError("future candidate action dossier is inconsistent")
    if (
        aether_candidate_formal.get("candidate_count") != 128
        or aether_candidate_formal.get("input_preflight_pass_count") != 128
        or aether_candidate_formal.get("decision_counts")
        != {"blocked": 126, "reject": 2}
        or aether_candidate_formal.get("formal_pass_count") != 0
        or aether_candidate_formal.get("solar_bundle_count") != 0
        or aether_candidate_formal.get("gate_finding_counts")
        != {
            "finite_characteristic_slicing_present": 121,
            "finite_negative_local_density_witness": 79,
            "globally_noncharacteristic_for_finite_unit_tilt": 5,
            "positive_at_every_finite_tilt_but_no_uniform_gap": 8,
            "principal_spin0_degeneracy_reject": 2,
            "uniform_positive_static_local_twist_gap": 39,
        }
        or len(aether_candidate_formal.get("candidate_bindings", [])) != 128
        or len(
            {
                row.get("candidate_id")
                for row in aether_candidate_formal.get("candidate_bindings", [])
            }
        )
        != 128
        or aether_candidate_formal.get("provenance", {}).get(
            "formal_preflight_status_sha256"
        )
        != formal_preflight.get("content_sha256")
        or aether_candidate_formal.get("provenance", {}).get(
            "compilation_campaign_sha256"
        )
        != parameter_compilation.get("content_sha256")
        or aether_candidate_formal.get("paid_llm_spend_usd") != 0.0
        or aether_candidate_formal.get("observational_data_opened") is not False
        or aether_candidate_formal.get("dark_matter_or_halo_inputs") is not False
        or aether_candidate_formal.get("redshift_distance_inputs") is not False
        or aether_candidate_formal.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("grammar-v3 Aether candidate formal campaign is inconsistent")
    if (
        g3_candidate_formal.get("execution_enabled") is not True
        or g3_candidate_formal.get("candidate_count") != 32
        or g3_candidate_formal.get("work_state_counts") != {"succeeded": 32}
        or g3_candidate_formal.get("decision_counts") != {"blocked": 32}
        or g3_candidate_formal.get("full_formal_pass_count") != 0
        or g3_candidate_formal.get("necessary_condition_rejection_count") != 0
        or g3_candidate_formal.get("blocker_counts")
        != {"uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain": 32}
        or g3_candidate_formal.get("gate_counts", {}).get(
            "candidate_action_preflight_admission_binding"
        )
        != {"pass": 32}
        or g3_candidate_formal.get("gate_counts", {}).get(
            "uniform_local_principal_symbol"
        )
        != {"pass": 32}
        or g3_candidate_formal.get("gate_counts", {}).get(
            "distributed_Dirac_on_periodic_cell"
        )
        != {"pass": 32}
        or g3_candidate_formal.get("gate_counts", {}).get(
            "af_uniform_lapse_Dirac_invertibility"
        )
        != {"blocked": 32}
        or g3_candidate_formal.get("promotion_status_binding", {}).get(
            "content_sha256"
        )
        != promotion_admission.get("content_sha256")
        or g3_candidate_formal.get("af_global_constraint_solution_proved")
        is not False
        or g3_candidate_formal.get("global_positive_energy_proved") is not False
        or g3_candidate_formal.get("paid_llm_spend_usd") != 0.0
        or g3_candidate_formal.get("observational_data_opened") is not False
        or g3_candidate_formal.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("grammar-v3 G3 candidate formal service is inconsistent")
    g4_equivalence = g4_scalable_formal_followup.get(
        "equivalence_certificate", {}
    )
    if (
        g4_scalable_formal_followup.get("candidate_count") != 1
        or g4_scalable_formal_followup.get("candidate_id")
        != "G3A-e0eff4150989e3522dc6ba03"
        or g4_scalable_formal_followup.get("preflight_decision") != "blocked"
        or g4_scalable_formal_followup.get("preflight_blocker")
        != "family_prerequisite_not_passed"
        or g4_scalable_formal_followup.get("formal_followup_decision") != "pass"
        or g4_scalable_formal_followup.get("decision_counts") != {"pass": 1}
        or g4_scalable_formal_followup.get("formal_pass_count") != 1
        or g4_scalable_formal_followup.get(
            "necessary_condition_rejection_count"
        )
        != 0
        or g4_scalable_formal_followup.get(
            "equivalent_parameter_cell_alias_count"
        )
        != 32
        or g4_equivalence.get("action_density_projection_equal") is not True
        or g4_equivalence.get("operator_densities_equal") is not True
        or g4_equivalence.get("universal_matter_coupling_equal") is not True
        or g4_equivalence.get("representative_domain_is_subset") is not True
        or g4_equivalence.get("all_alias_domains_inside_reviewed_domain")
        is not True
        or g4_equivalence.get("family_label_used_as_equivalence_evidence")
        is not False
        or g4_scalable_formal_followup.get("source_bindings", {})
        .get("compilation_campaign", {})
        .get("content_sha256")
        != parameter_compilation.get("content_sha256")
        or g4_scalable_formal_followup.get("source_bindings", {})
        .get("formal_preflight_status", {})
        .get("content_sha256")
        != formal_preflight.get("content_sha256")
        or g4_scalable_formal_followup.get("solar_bundle_count") != 0
        or g4_scalable_formal_followup.get("paid_llm_spend_usd") != 0.0
        or g4_scalable_formal_followup.get("observational_data_opened") is not False
        or g4_scalable_formal_followup.get("dark_matter_or_halo_inputs") is not False
        or g4_scalable_formal_followup.get("redshift_distance_inputs") is not False
        or g4_scalable_formal_followup.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("grammar-v3 scalable G4 formal follow-up is inconsistent")
    if (
        llm_adapter.get("status") != "ready_disabled_no_network_no_spend"
        or llm_adapter.get("default_paid_calls_enabled") is not False
        or llm_adapter.get("maximum_total_usd") != "500.000000"
        or llm_adapter.get("network_calls_made") != 0
        or llm_adapter.get("paid_spend_usd") != "0.000000"
        or llm_adapter.get("output_status")
        != "quarantine_until_downstream_validation"
    ):
        raise ValueError("LLM proposal adapter readiness is not fail-closed")
    if (
        llm_bridge.get("status") != "ready_disabled_quarantine_only"
        or llm_bridge.get("default_execution_enabled") is not False
        or llm_bridge.get("maximum_total_usd") != "500.000000"
        or llm_bridge.get("network_calls_made") != 0
        or llm_bridge.get("paid_spend_usd") != "0.000000"
        or llm_bridge.get("compiler_tasks_enqueued") != 0
        or llm_bridge.get("raw_body_persistence") is not False
        or llm_bridge.get("provider_callback_registered") is not False
        or llm_bridge.get("prompt_resolver_callback_registered") is not False
    ):
        raise ValueError("LLM campaign bridge readiness is not fail-closed")
    solar_decision = g4_solar.get("current_evaluator_decision", {})
    if (
        g4_solar.get("decision") != "blocked"
        or g4_solar.get("descriptor_implementation_ready") is not True
        or g4_solar.get("candidate_use_authorized") is not False
        or g4_solar.get("observational_data_opened") is not False
        or g4_solar.get("primary_record_access_count") != 0
        or solar_decision.get("decision") != "blocked"
        or solar_decision.get("filled_registration_hash_count") != 1
        or len(solar_decision.get("missing_registration_hashes", [])) != 16
    ):
        raise ValueError("reviewed G4 Solar evaluator readiness is not fail-closed")
    if (
        g4_solar_execution.get("decision_counts") != {"blocked": 1}
        or g4_solar_execution.get("work_state_counts") != {"succeeded": 1}
        or g4_solar_execution.get("reviewed_evaluator_invocation_count") != 1
        or g4_solar_execution.get("filled_registration_hash_count") != 1
        or g4_solar_execution.get("missing_registration_hash_count") != 16
        or g4_solar_execution.get("observational_data_opened") is not False
        or g4_solar_execution.get("primary_record_access_count") != 0
    ):
        raise ValueError("reviewed G4 Solar execution status is not fail-closed")
    galaxy_decision = g4_galaxy.get("current_evaluator_decision", {})
    if (
        g4_galaxy.get("decision") != "blocked"
        or g4_galaxy.get("descriptor_implementation_ready") is not True
        or g4_galaxy.get("prediction_bundle_registered") is not False
        or g4_galaxy.get("candidate_use_authorized") is not False
        or g4_galaxy.get("observational_data_opened") is not False
        or g4_galaxy.get("primary_record_access_count") != 0
        or galaxy_decision.get("filled_registration_hash_count") != 1
        or len(galaxy_decision.get("missing_registration_hashes", [])) != 17
        or g4_galaxy["synthetic_controls"]["shape"].get(
            "object_specific_gravity_parameter_count"
        )
        != 0
    ):
        raise ValueError("reviewed G4 galaxy evaluator readiness is not fail-closed")
    if (
        g4_galaxy_execution.get("decision_counts") != {"blocked": 1}
        or g4_galaxy_execution.get("work_state_counts") != {"succeeded": 1}
        or g4_galaxy_execution.get("reviewed_evaluator_invocation_count") != 1
        or g4_galaxy_execution.get("filled_registration_hash_count") != 1
        or g4_galaxy_execution.get("missing_registration_hash_count") != 17
        or g4_galaxy_execution.get("prediction_bundle_registered") is not False
        or g4_galaxy_execution.get("object_specific_gravity_parameter_count") != 0
        or g4_galaxy_execution.get("observational_data_opened") is not False
    ):
        raise ValueError("reviewed G4 galaxy execution status is not fail-closed")
    if (
        typed_admission.get("status") != "ready_disabled_hash_only"
        or typed_admission.get("default_execution_enabled") is not False
        or typed_admission.get("formula_body_persistence") is not False
        or typed_admission.get("paid_spend_usd") != "0.000000"
        or typed_admission.get("fixture_expected_counts")
        != {"block": 1, "enqueue": 1, "pass": 1, "reject": 9}
    ):
        raise ValueError("typed DSL admission readiness is not fail-closed")
    if (
        compiler_registry.get("status") != "ready_disabled_hash_only"
        or compiler_registry.get("default_execution_enabled") is not False
        or compiler_registry.get("candidate_body_persistence") is not False
        or compiler_registry.get("next_stage_adapter_registered") is not False
        or compiler_registry.get("novelty_claim_allowed") is not False
        or compiler_registry.get("fixture_expected_counts")
        != {"block": 1, "dedup": 1, "enqueue": 1, "pass": 1, "reject": 7}
    ):
        raise ValueError("compiler receipt registry readiness is not fail-closed")
    expected_epoch = {
        "candidate_count": 1,
        "compiler_receipt_pass_count": 2,
        "decision_counts": {"block": 1, "dedup": 1, "pass": 1, "reject": 1},
        "network_calls": 0,
        "next_stage_enqueue_count": 1,
        "paid_spend_usd": "0.000000",
        "policy_pass_count": 1,
        "proposal_quarantine_count": 4,
    }
    if (
        local_formula_epoch.get("status") != "ready_disabled_bounded_mock_only"
        or local_formula_epoch.get("default_execution_enabled") is not False
        or local_formula_epoch.get("formula_body_persistence") is not False
        or local_formula_epoch.get("maximum_total_usd") != "500.000000"
        or local_formula_epoch.get("network_calls") != 0
        or local_formula_epoch.get("paid_spend_usd") != "0.000000"
        or local_formula_epoch.get("expected_bounded_status") != expected_epoch
    ):
        raise ValueError("reviewed local formula epoch readiness is not fail-closed")
    if (
        local_formula_service.get("status")
        != "ready_disabled_bounded_local_only"
        or local_formula_service.get("default_execution_enabled") is not False
        or local_formula_service.get("network_allowed") is not False
        or local_formula_service.get("paid_spend_usd") != "0.000000"
        or local_formula_service.get("deterministic_export") is not True
        or local_formula_service.get("budgets")
        != {
            "maximum_attempts": 3,
            "maximum_disk_bytes": 100_000_000,
            "maximum_tasks": 1,
            "maximum_wall_seconds": 120,
        }
    ):
        raise ValueError("reviewed local formula service readiness is not fail-closed")
    forward_decision = g4_galaxy_forward_model.get("current_evaluator_decision", {})
    forward_controls = g4_galaxy_forward_model.get("synthetic_controls", {})
    if (
        g4_galaxy_forward_model.get("decision") != "blocked"
        or g4_galaxy_forward_model.get("prediction_bundle_registered") is not False
        or g4_galaxy_forward_model.get("candidate_use_authorized") is not False
        or g4_galaxy_forward_model.get("observational_data_opened") is not False
        or g4_galaxy_forward_model.get("primary_record_access_count") != 0
        or g4_galaxy_forward_model.get("object_specific_gravity_parameter_count") != 0
        or g4_galaxy_forward_model.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_forward_model.get("redshift_distance_inputs") is not False
        or forward_decision.get("filled_registration_hash_count") != 3
        or len(forward_decision.get("missing_registration_hashes", [])) != 15
        or set(g4_galaxy_forward_model.get("newly_filled_registration_fields", {}))
        != {
            "lensing_prediction_implementation_sha256",
            "rotation_prediction_implementation_sha256",
        }
        or set(forward_controls.get("analytic_known_answers", {}).values())
        != {"pass"}
        or forward_controls.get("covariance", {}).get("decision") != "pass"
    ):
        raise ValueError("G4 galaxy forward-model readiness is not fail-closed")
    branch_distance_decision = g4_galaxy_branch_distance.get(
        "current_evaluator_decision", {}
    )
    if (
        g4_galaxy_branch_distance.get("decision") != "blocked"
        or g4_galaxy_branch_distance.get("prediction_bundle_registered") is not False
        or g4_galaxy_branch_distance.get("candidate_use_authorized") is not False
        or g4_galaxy_branch_distance.get("observational_data_opened") is not False
        or g4_galaxy_branch_distance.get("primary_record_access_count") != 0
        or g4_galaxy_branch_distance.get("real_source_geometry_registered") is not False
        or g4_galaxy_branch_distance.get("source_specific_branch_selection_proven")
        is not False
        or g4_galaxy_branch_distance.get("object_specific_gravity_parameter_count")
        != 0
        or g4_galaxy_branch_distance.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_branch_distance.get("redshift_distance_inputs") is not False
        or branch_distance_decision.get("filled_registration_hash_count") != 5
        or len(branch_distance_decision.get("missing_registration_hashes", [])) != 13
        or set(g4_galaxy_branch_distance.get("newly_filled_registration_fields", {}))
        != {
            "branch_and_domain_contract_sha256",
            "distance_mode_contract_sha256",
        }
        or set(
            g4_galaxy_branch_distance.get(
                "preserved_predecessor_registration_fields", {}
            )
        )
        != {
            "reviewed_candidate_galaxy_evaluator_descriptor_sha256",
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
        or g4_galaxy_branch_distance.get("branch_contract_status")
        != "certified_exact_conditional_branch"
        or g4_galaxy_branch_distance.get("distance_geometry_contract_status")
        != "certified_interface_no_real_values"
        or g4_galaxy_branch_distance.get("provenance", {}).get(
            "forward_model_predecessor_sha256"
        )
        != g4_galaxy_forward_model.get("content_sha256")
    ):
        raise ValueError("G4 galaxy branch/distance registration is not fail-closed")
    calibration_decision = g4_galaxy_calibration_evaluation.get(
        "current_evaluator_decision", {}
    )
    if (
        g4_galaxy_calibration_evaluation.get("decision") != "blocked"
        or g4_galaxy_calibration_evaluation.get("prediction_bundle_registered")
        is not False
        or g4_galaxy_calibration_evaluation.get("candidate_use_authorized")
        is not False
        or g4_galaxy_calibration_evaluation.get("observational_data_opened")
        is not False
        or g4_galaxy_calibration_evaluation.get("primary_record_access_count") != 0
        or g4_galaxy_calibration_evaluation.get(
            "object_specific_gravity_parameter_count"
        )
        != 0
        or g4_galaxy_calibration_evaluation.get("dark_matter_or_halo_inputs")
        is not False
        or g4_galaxy_calibration_evaluation.get("redshift_distance_inputs")
        is not False
        or g4_galaxy_calibration_evaluation.get("paid_llm_spend_usd") != 0.0
        or calibration_decision.get("filled_registration_hash_count") != 9
        or len(calibration_decision.get("missing_registration_hashes", [])) != 9
        or set(
            g4_galaxy_calibration_evaluation.get(
                "newly_filled_registration_fields", {}
            )
        )
        != {
            "baryonic_calibration_hierarchy_sha256",
            "joint_covariance_contract_sha256",
            "likelihood_contract_sha256",
            "stopping_rule_sha256",
        }
        or set(
            g4_galaxy_calibration_evaluation.get(
                "preserved_predecessor_registration_fields", {}
            )
        )
        != {
            "branch_and_domain_contract_sha256",
            "distance_mode_contract_sha256",
            "reviewed_candidate_galaxy_evaluator_descriptor_sha256",
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
        or set(
            g4_galaxy_calibration_evaluation.get(
                "non_registration_policy_hashes", {}
            )
        )
        != {"held_out_split_policy_sha256"}
        or set(
            g4_galaxy_calibration_evaluation.get(
                "deliberately_unfilled_registration_fields", {}
            )
        )
        != {
            "galaxy_split_commitment_sha256",
            "training_only_checkpoint_sha256",
        }
        or g4_galaxy_calibration_evaluation.get("provenance", {}).get(
            "predecessor_content_sha256"
        )
        != g4_galaxy_branch_distance.get("content_sha256")
    ):
        raise ValueError("G4 galaxy calibration/evaluation registration is not fail-closed")
    transform_decision = g4_galaxy_prediction_contract_transform.get(
        "current_evaluator_decision", {}
    )
    if (
        g4_galaxy_prediction_contract_transform.get("decision") != "blocked"
        or g4_galaxy_prediction_contract_transform.get(
            "prediction_bundle_registered"
        )
        is not False
        or g4_galaxy_prediction_contract_transform.get("candidate_use_authorized")
        is not False
        or g4_galaxy_prediction_contract_transform.get("observational_data_opened")
        is not False
        or g4_galaxy_prediction_contract_transform.get("primary_record_access_count")
        != 0
        or g4_galaxy_prediction_contract_transform.get(
            "object_specific_gravity_parameter_count"
        )
        != 0
        or g4_galaxy_prediction_contract_transform.get("dark_matter_or_halo_inputs")
        is not False
        or g4_galaxy_prediction_contract_transform.get("redshift_distance_inputs")
        is not False
        or g4_galaxy_prediction_contract_transform.get("paid_llm_spend_usd") != 0.0
        or g4_galaxy_prediction_contract_transform.get(
            "real_transform_inputs_registered"
        )
        is not False
        or transform_decision.get("filled_registration_hash_count") != 11
        or len(transform_decision.get("missing_registration_hashes", [])) != 7
        or set(
            g4_galaxy_prediction_contract_transform.get(
                "newly_filled_registration_fields", {}
            )
        )
        != {
            "prediction_bundle_contract_sha256",
            "raw_to_calibrated_transform_sha256",
        }
        or len(
            g4_galaxy_prediction_contract_transform.get(
                "preserved_predecessor_registration_fields", {}
            )
        )
        != 9
        or g4_galaxy_prediction_contract_transform.get("provenance", {}).get(
            "predecessor_content_sha256"
        )
        != g4_galaxy_calibration_evaluation.get("content_sha256")
        or g4_galaxy_prediction_contract_transform.get("synthetic_control", {}).get(
            "cross_channel_covariance_retained"
        )
        is not True
        or g4_galaxy_prediction_contract_transform.get("synthetic_control", {}).get(
            "real_operator_or_values_registered"
        )
        is not False
    ):
        raise ValueError("G4 galaxy prediction/transform registration is not fail-closed")
    tooling_decision = g4_galaxy_manifest_bundle_tooling.get(
        "unchanged_evaluator_decision", {}
    )
    tooling_controls = g4_galaxy_manifest_bundle_tooling.get(
        "synthetic_controls", {}
    )
    if (
        g4_galaxy_manifest_bundle_tooling.get("decision") != "blocked"
        or g4_galaxy_manifest_bundle_tooling.get("candidate_use_authorized")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("observational_data_opened")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("primary_record_access_count") != 0
        or g4_galaxy_manifest_bundle_tooling.get("prediction_bundle_registered")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("dataset_manifest_registered")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get(
            "independent_registry_receipt_registered"
        )
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("dark_matter_or_halo_inputs")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("redshift_distance_inputs")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("paid_llm_spend_usd") != 0.0
        or g4_galaxy_manifest_bundle_tooling.get("filled_registration_hash_count")
        != 11
        or g4_galaxy_manifest_bundle_tooling.get("missing_registration_hash_count")
        != 7
        or g4_galaxy_manifest_bundle_tooling.get("newly_filled_registration_fields")
        != {}
        or tooling_decision.get("filled_registration_hash_count") != 11
        or len(tooling_decision.get("missing_registration_hashes", [])) != 7
        or tooling_controls.get("manifest_audit_registration_admissible") is not False
        or tooling_controls.get("bundle_draft_registration_admissible") is not False
        or tooling_controls.get("synthetic_values_promoted") is not False
        or g4_galaxy_manifest_bundle_tooling.get("tooling_readiness", {}).get(
            "enabled"
        )
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("provenance", {}).get(
            "predecessor_content_sha256"
        )
        != g4_galaxy_prediction_contract_transform.get("content_sha256")
    ):
        raise ValueError("G4 galaxy manifest/bundle tooling is not fail-closed")
    source_registry_decision = g4_galaxy_source_registry_admission.get(
        "unchanged_evaluator_decision", {}
    )
    source_registry_readiness = g4_galaxy_source_registry_admission.get(
        "admission_readiness", {}
    )
    if (
        g4_galaxy_source_registry_admission.get("decision") != "blocked"
        or g4_galaxy_source_registry_admission.get("service_enabled") is not False
        or g4_galaxy_source_registry_admission.get("start_requested") is not False
        or g4_galaxy_source_registry_admission.get("source_records_admitted") != 0
        or g4_galaxy_source_registry_admission.get("target_records_opened") != 0
        or g4_galaxy_source_registry_admission.get("primary_record_access_count") != 0
        or g4_galaxy_source_registry_admission.get(
            "observation_opening_authorization_registered"
        )
        is not False
        or g4_galaxy_source_registry_admission.get("prediction_bundle_registered")
        is not False
        or g4_galaxy_source_registry_admission.get("observational_data_opened")
        is not False
        or g4_galaxy_source_registry_admission.get("dark_matter_or_halo_inputs")
        is not False
        or g4_galaxy_source_registry_admission.get("redshift_distance_inputs")
        is not False
        or g4_galaxy_source_registry_admission.get(
            "object_specific_gravity_parameter_count"
        )
        != 0
        or g4_galaxy_source_registry_admission.get("paid_llm_spend_usd") != 0.0
        or g4_galaxy_source_registry_admission.get("filled_registration_hash_count")
        != 11
        or g4_galaxy_source_registry_admission.get("missing_registration_hash_count")
        != 7
        or g4_galaxy_source_registry_admission.get(
            "newly_filled_registration_fields"
        )
        != {}
        or source_registry_decision.get("filled_registration_hash_count") != 11
        or len(source_registry_decision.get("missing_registration_hashes", [])) != 7
        or source_registry_readiness.get("service_enabled") is not False
        or source_registry_readiness.get("start_requested") is not False
        or source_registry_readiness.get("registration_fields_filled") != 0
        or g4_galaxy_source_registry_admission.get("provenance", {}).get(
            "manifest_bundle_tooling_sha256"
        )
        != g4_galaxy_manifest_bundle_tooling.get("content_sha256")
        or g4_galaxy_source_registry_admission.get("provenance", {}).get(
            "ledger_predecessor_sha256"
        )
        != g4_galaxy_prediction_contract_transform.get("content_sha256")
    ):
        raise ValueError("G4 galaxy source-registry admission is not fail-closed")

    quartic_counts = quartic_tc2_quadratic_deltak.get("counts", {})
    quartic_pairs = quartic_tc2_quadratic_deltak.get("pair_partition", {})
    quartic_control = quartic_tc2_quadratic_deltak.get(
        "generic_quadratic_sylvester_jet_control", {}
    )
    if (
        quartic_tc2_quadratic_deltak.get("status")
        != "pass_all_12_complete_reference_quadratic_deltaK_two_jets_full_identity_fail_closed"
        or quartic_pairs.get("total_unordered_coordinate_pairs") != 11_781
        or quartic_pairs.get("canonical_active_exact_pairs") != 861
        or quartic_pairs.get("excluded_exact_obligations") != 2_675
        or quartic_pairs.get("entrywise_zero_chain_rule_pairs") != 8_245
        or quartic_pairs.get("coverage_complete") is not True
        or quartic_counts.get("selected") != 12
        or quartic_counts.get("reference_two_jets_closed") != 12
        or any(
            quartic_counts.get(key) != 0
            for key in (
                "full_tube_Sylvester_identities",
                "full_variable_CK1_closures",
                "CK3_closures",
                "TC2_closures",
                "B7_closures",
                "global_H7_closures",
                "lifespans_proved",
            )
        )
        or quartic_control.get("reference_jet_orders_closed") != [0, 1, 2]
        or quartic_control.get("full_tube_Sylvester_identity_closed") is not False
    ):
        raise ValueError("quartic TC2 quadratic deltaK extension is inconsistent")
    quartic_third_counts = quartic_tc2_diagonal_third_jet.get("counts", {})
    quartic_third_slice = quartic_tc2_diagonal_third_jet.get(
        "slice_contract", {}
    )
    quartic_third_blocker = quartic_tc2_diagonal_third_jet.get(
        "first_remaining_blocker", {}
    )
    if (
        quartic_tc2_diagonal_third_jet.get("status")
        != "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_mixed_triples_full_tube_global_H7_fail_closed"
        or quartic_third_slice.get("active_coordinate_directions") != 41
        or quartic_third_slice.get("diagonal_triples") != 41
        or quartic_third_slice.get("full_symmetric_triples_in_41_direction_sector")
        != 12_341
        or quartic_third_slice.get("mixed_AAB_ABB_ABC_triples") != 0
        or quartic_third_counts.get("candidates") != 12
        or quartic_third_counts.get("symbolic_parameter_diagonal_third_jet_passes")
        != 41
        or quartic_third_counts.get("candidate_direction_evaluations") != 492
        or quartic_third_counts.get("candidate_direction_solvable") != 492
        or quartic_third_counts.get("candidate_direction_obstructed") != 0
        or quartic_third_counts.get(
            "candidates_all_41_diagonal_third_jets_closed"
        )
        != 12
        or any(
            quartic_third_counts.get(key) != 0
            for key in (
                "mixed_third_jet_closures",
                "full_tube_Sylvester_identities",
                "TC2_closures",
                "B7_closures",
                "global_H7_closures",
                "lifespans_proved",
            )
        )
        or quartic_third_blocker.get("closed") is not False
        or "12,300 polarized mixed triples"
        not in quartic_third_blocker.get("required", "")
    ):
        raise ValueError("quartic TC2 diagonal third-jet slice is inconsistent")
    quartic_mixed_counts = quartic_tc2_mixed_third_jet_chunk.get("counts", {})
    quartic_mixed_contract = quartic_tc2_mixed_third_jet_chunk.get(
        "chunk_contract", {}
    )
    quartic_mixed_ledger = quartic_tc2_mixed_third_jet_chunk.get(
        "closure_ledger", {}
    )
    if (
        quartic_tc2_mixed_third_jet_chunk.get("status")
        != "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
        or quartic_mixed_contract.get("chunk_offset") != 0
        or quartic_mixed_contract.get("processed_count") != 64
        or quartic_mixed_contract.get("next_offset") != 64
        or quartic_mixed_contract.get("global_mixed_triple_count") != 12_300
        or quartic_mixed_contract.get("stopped_early") is not False
        or quartic_mixed_counts.get("selected") != 64
        or quartic_mixed_counts.get("symbolic_parameter_compatible") != 64
        or quartic_mixed_counts.get("candidate_evaluations") != 768
        or quartic_mixed_counts.get("candidate_solvable") != 768
        or quartic_mixed_counts.get("candidate_obstructed") != 0
        or quartic_mixed_counts.get("triple_kind_counts")
        != {"AAB": 40, "ABB": 1, "ABC": 23}
        or quartic_mixed_counts.get("mixed_triples_remaining") != 12_236
        or quartic_tc2_mixed_third_jet_chunk.get("first_exact_obstruction")
        is not None
        or quartic_tc2_mixed_third_jet_chunk.get("upstream_sha256", {}).get(
            "diagonal_third_jet"
        )
        != quartic_tc2_diagonal_third_jet.get("content_sha256")
        or quartic_mixed_ledger.get("processed_mixed_third_jets_closed") != 64
        or any(
            quartic_mixed_ledger.get(key) is not False
            for key in (
                "all_12_300_mixed_third_jets_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
    ):
        raise ValueError("quartic TC2 mixed third-jet chunk is inconsistent")
    quartic_mixed_64_counts = quartic_tc2_mixed_third_jet_chunk_64.get(
        "counts", {}
    )
    quartic_mixed_64_contract = quartic_tc2_mixed_third_jet_chunk_64.get(
        "chunk_contract", {}
    )
    quartic_mixed_64_ledger = quartic_tc2_mixed_third_jet_chunk_64.get(
        "closure_ledger", {}
    )
    if (
        quartic_tc2_mixed_third_jet_chunk_64.get("status")
        != "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
        or quartic_mixed_64_contract.get("chunk_offset") != 64
        or quartic_mixed_64_contract.get("processed_count") != 64
        or quartic_mixed_64_contract.get("next_offset") != 128
        or quartic_mixed_64_contract.get("global_mixed_triple_count") != 12_300
        or quartic_mixed_64_contract.get("prior_resume_sha256")
        != quartic_mixed_contract.get("resume_tip_sha256")
        or quartic_mixed_64_contract.get("stopped_early") is not False
        or quartic_mixed_64_counts.get("selected") != 64
        or quartic_mixed_64_counts.get("symbolic_parameter_compatible") != 64
        or quartic_mixed_64_counts.get("candidate_evaluations") != 768
        or quartic_mixed_64_counts.get("candidate_solvable") != 768
        or quartic_mixed_64_counts.get("candidate_obstructed") != 0
        or quartic_mixed_64_counts.get("triple_kind_counts")
        != {"ABB": 2, "ABC": 62}
        or quartic_mixed_64_counts.get("mixed_triples_remaining") != 12_172
        or quartic_tc2_mixed_third_jet_chunk_64.get("first_exact_obstruction")
        is not None
        or quartic_mixed_64_ledger.get("processed_mixed_third_jets_closed") != 64
        or any(
            quartic_mixed_64_ledger.get(key) is not False
            for key in (
                "all_12_300_mixed_third_jets_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
        or quartic_tc2_mixed_third_jet_checkpoint.get("completed_chunks") != 1
        or quartic_tc2_mixed_third_jet_checkpoint.get("next_offset") != 128
        or quartic_tc2_mixed_third_jet_checkpoint.get("remaining_mixed_triples")
        != 12_172
        or quartic_tc2_mixed_third_jet_checkpoint.get(
            "current_artifact_content_sha256"
        )
        != quartic_tc2_mixed_third_jet_chunk_64.get("content_sha256")
        or quartic_tc2_mixed_third_jet_checkpoint.get("prior_resume_sha256")
        != quartic_mixed_64_contract.get("resume_tip_sha256")
        or quartic_tc2_mixed_third_jet_continuation_status.get(
            "checkpoint_content_sha256"
        )
        != quartic_tc2_mixed_third_jet_checkpoint.get("content_sha256")
        or quartic_tc2_mixed_third_jet_continuation_status.get("next_offset")
        != 128
        or quartic_tc2_mixed_third_jet_continuation_status.get(
            "remaining_mixed_triples"
        )
        != 12_172
        or quartic_tc2_mixed_third_jet_continuation_status.get("decision")
        != "checkpointed"
        or quartic_tc2_mixed_third_jet_continuation_status.get(
            "permanently_stopped"
        )
        is not False
        or any(
            quartic_tc2_mixed_third_jet_continuation_status.get("claims", {}).get(
                key
            )
            is not False
            for key in (
                "full_mixed_sector_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
    ):
        raise ValueError("quartic TC2 mixed third-jet continuation is inconsistent")

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
        "proposal_adapter": {
            "default_paid_calls_enabled": False,
            "maximum_call_usd": llm_adapter["maximum_call_usd"],
            "maximum_total_usd": llm_adapter["maximum_total_usd"],
            "network_calls_made": llm_adapter["network_calls_made"],
            "output_status": llm_adapter["output_status"],
            "paid_spend_usd": llm_adapter["paid_spend_usd"],
            "status": llm_adapter["status"],
        },
        "campaign_bridge": {
            "admission_callback_configured": llm_bridge["admission_callback_configured"],
            "campaign_task_type": llm_bridge["campaign_task_type"],
            "compiler_tasks_enqueued": llm_bridge["compiler_tasks_enqueued"],
            "default_execution_enabled": llm_bridge["default_execution_enabled"],
            "network_calls_made": llm_bridge["network_calls_made"],
            "paid_spend_usd": llm_bridge["paid_spend_usd"],
            "raw_body_persistence": llm_bridge["raw_body_persistence"],
            "status": llm_bridge["status"],
        },
        "typed_dsl_admission": {
            "compiler_queue_task_type": typed_admission["compiler_queue_task_type"],
            "default_execution_enabled": typed_admission["default_execution_enabled"],
            "fixture_expected_counts": typed_admission["fixture_expected_counts"],
            "formula_body_persistence": typed_admission["formula_body_persistence"],
            "status": typed_admission["status"],
        },
        "compiler_registry_bridge": {
            "candidate_body_persistence": compiler_registry[
                "candidate_body_persistence"
            ],
            "default_execution_enabled": compiler_registry[
                "default_execution_enabled"
            ],
            "fixture_expected_counts": compiler_registry["fixture_expected_counts"],
            "next_stage_adapter_registered": compiler_registry[
                "next_stage_adapter_registered"
            ],
            "novelty_claim_allowed": compiler_registry["novelty_claim_allowed"],
            "status": compiler_registry["status"],
        },
        "reviewed_local_epoch": {
            "default_execution_enabled": local_formula_epoch[
                "default_execution_enabled"
            ],
            "expected_bounded_status": local_formula_epoch["expected_bounded_status"],
            "formula_body_persistence": local_formula_epoch[
                "formula_body_persistence"
            ],
            "network_calls": local_formula_epoch["network_calls"],
            "paid_spend_usd": local_formula_epoch["paid_spend_usd"],
            "status": local_formula_epoch["status"],
        },
        "reviewed_local_service": {
            "budgets": local_formula_service["budgets"],
            "default_execution_enabled": local_formula_service[
                "default_execution_enabled"
            ],
            "deterministic_export": local_formula_service[
                "deterministic_export"
            ],
            "network_allowed": local_formula_service["network_allowed"],
            "paid_spend_usd": local_formula_service["paid_spend_usd"],
            "status": local_formula_service["status"],
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
        "g4_solar_evaluator": {
            "candidate_id": g4_solar["candidate"]["candidate_id"],
            "decision": g4_solar["decision"],
            "descriptor_implementation_ready": g4_solar[
                "descriptor_implementation_ready"
            ],
            "filled_registration_hash_count": solar_decision[
                "filled_registration_hash_count"
            ],
            "missing_registration_hash_count": len(
                solar_decision["missing_registration_hashes"]
            ),
            "first_missing_premise": g4_solar["first_missing_premise"],
            "observational_data_opened": g4_solar["observational_data_opened"],
            "primary_record_access_count": g4_solar["primary_record_access_count"],
            "synthetic_GR_golden_pass_count": sum(
                status == "pass"
                for status in g4_solar["synthetic_fixtures"]["GR_known_answer"][
                    "golden_statuses"
                ].values()
            ),
            "durable_execution": {
                "decision_counts": g4_solar_execution["decision_counts"],
                "reviewed_evaluator_invocation_count": g4_solar_execution[
                    "reviewed_evaluator_invocation_count"
                ],
                "task_count": g4_solar_execution["task_count"],
                "work_state_counts": g4_solar_execution["work_state_counts"],
            },
        },
        "g4_galaxy_evaluator": {
            "candidate_id": g4_galaxy["candidate"]["candidate_id"],
            "decision": g4_galaxy["decision"],
            "descriptor_implementation_ready": g4_galaxy[
                "descriptor_implementation_ready"
            ],
            "filled_registration_hash_count": galaxy_decision[
                "filled_registration_hash_count"
            ],
            "first_missing_premise": g4_galaxy["first_missing_premise"],
            "missing_registration_hash_count": len(
                galaxy_decision["missing_registration_hashes"]
            ),
            "object_specific_gravity_parameter_count": g4_galaxy[
                "synthetic_controls"
            ]["shape"]["object_specific_gravity_parameter_count"],
            "observational_data_opened": g4_galaxy["observational_data_opened"],
            "prediction_bundle_registered": g4_galaxy[
                "prediction_bundle_registered"
            ],
            "primary_record_access_count": g4_galaxy["primary_record_access_count"],
            "synthetic_control_decisions": {
                key: value["decision"]
                for key, value in sorted(g4_galaxy["synthetic_controls"].items())
            },
            "durable_execution": {
                "decision_counts": g4_galaxy_execution["decision_counts"],
                "reviewed_evaluator_invocation_count": g4_galaxy_execution[
                    "reviewed_evaluator_invocation_count"
                ],
                "task_count": g4_galaxy_execution["task_count"],
                "work_state_counts": g4_galaxy_execution["work_state_counts"],
            },
            "forward_model": {
                "analytic_known_answer_pass_count": sum(
                    status == "pass"
                    for status in forward_controls["analytic_known_answers"].values()
                ),
                "covariance_control": forward_controls["covariance"]["decision"],
                "decision": g4_galaxy_forward_model["decision"],
                "filled_registration_hash_count": forward_decision[
                    "filled_registration_hash_count"
                ],
                "first_missing_premise": g4_galaxy_forward_model[
                    "first_missing_premise"
                ],
                "missing_registration_hash_count": len(
                    forward_decision["missing_registration_hashes"]
                ),
                "newly_filled_fields": sorted(
                    g4_galaxy_forward_model["newly_filled_registration_fields"]
                ),
                "object_specific_gravity_parameter_count": 0,
                "observational_data_opened": False,
                "prediction_bundle_registered": False,
            },
            "registration": {
                "branch_contract_status": g4_galaxy_branch_distance[
                    "branch_contract_status"
                ],
                "decision": g4_galaxy_prediction_contract_transform["decision"],
                "distance_geometry_contract_status": g4_galaxy_branch_distance[
                    "distance_geometry_contract_status"
                ],
                "filled_registration_hash_count": transform_decision[
                    "filled_registration_hash_count"
                ],
                "first_missing_premise": g4_galaxy_prediction_contract_transform[
                    "first_missing_premise"
                ],
                "missing_registration_hash_count": len(
                    transform_decision["missing_registration_hashes"]
                ),
                "newly_filled_fields": sorted(
                    g4_galaxy_prediction_contract_transform[
                        "newly_filled_registration_fields"
                    ]
                ),
                "held_out_split_policy_registered_as_evidence": True,
                "real_split_commitment_registered": False,
                "object_specific_gravity_parameter_count": 0,
                "observational_data_opened": False,
                "prediction_bundle_registered": False,
                "real_transform_inputs_registered": False,
                "real_source_geometry_registered": False,
                "source_specific_branch_selection_proven": False,
                "manifest_bundle_tooling": {
                    "decision": g4_galaxy_manifest_bundle_tooling["decision"],
                    "enabled": g4_galaxy_manifest_bundle_tooling[
                        "tooling_readiness"
                    ]["enabled"],
                    "filled_registration_hash_count": g4_galaxy_manifest_bundle_tooling[
                        "filled_registration_hash_count"
                    ],
                    "first_missing_premise": g4_galaxy_manifest_bundle_tooling[
                        "first_missing_premise"
                    ],
                    "missing_registration_hash_count": g4_galaxy_manifest_bundle_tooling[
                        "missing_registration_hash_count"
                    ],
                    "newly_filled_fields": [],
                    "synthetic_bundle_registration_admissible": False,
                    "synthetic_manifest_registration_admissible": False,
                },
                "source_registry_admission": {
                    "decision": g4_galaxy_source_registry_admission["decision"],
                    "enabled": False,
                    "filled_registration_hash_count": g4_galaxy_source_registry_admission[
                        "filled_registration_hash_count"
                    ],
                    "first_missing_premise": g4_galaxy_source_registry_admission[
                        "first_missing_premise"
                    ],
                    "missing_registration_hash_count": g4_galaxy_source_registry_admission[
                        "missing_registration_hash_count"
                    ],
                    "newly_filled_fields": [],
                    "source_opening_permission_registered": False,
                    "source_records_admitted": 0,
                    "target_records_opened": 0,
                },
            },
        },
        "grammar_parameter_cells": {
            "seed_execution": {
                "task_state_counts": parameter["work_state_counts"],
                "scientific_decision_counts": parameter["decision_counts"],
                "normalized_scientific_outcomes": {
                    "pass": 0,
                    "reject": 0,
                    "block": 6,
                },
                "maximum_tasks": parameter["budget"]["maximum_tasks"],
                "deadline": "bounded_completed_artifact_no_live_deadline",
                "next_scaling_hook": parameter["next_scaling_hook"],
                "candidate_universe": "six reviewed deterministic seed actions",
            },
            "scalable_unique_action_formal_outcomes": {
                "pass": 3,
                "reject": 2,
                "block": 158,
            },
            "scalable_admitted_family_formal_outcomes": {
                "pass": 2,
                "reject": 2,
                "block": 158,
            },
            "scalable_preflight_blocked_excluded_count": 1,
            "scalable_preflight_blocked_followup_resolved_count": 1,
            "expansion_service": {
                "chunk_count": parameter_expansion["chunk_count"],
                "decision_counts": parameter_expansion["decision_counts"],
                "parameter_cell_count": parameter_expansion["parameter_cell_count"],
                "scientific_scope": parameter_expansion["scientific_scope"],
                "work_state_counts": parameter_expansion["work_state_counts"],
            },
            "reviewed_manifest": {
                "chunk_count": len(parameter_manifest["chunks"]),
                "family_cell_counts": parameter_manifest["family_cell_counts"],
                "formal_evaluation_performed": False,
                "parameter_cell_count": parameter_manifest["parameter_cell_count"],
                "scientific_decision_counts": {},
                "next_execution_hook": parameter_manifest["next_execution_hook"],
                "compilation": {
                    "candidate_decision_counts": parameter_compilation[
                        "candidate_decision_counts"
                    ],
                    "compiled_action_ir_count": parameter_compilation[
                        "compiled_action_ir_count"
                    ],
                    "equivalent_duplicate_count": parameter_compilation[
                        "equivalent_duplicate_count"
                    ],
                    "expensive_formal_campaign_run": False,
                    "formal_decision_counts": {},
                    "unique_candidate_count": parameter_compilation[
                        "unique_candidate_count"
                    ],
                    "formal_preflight": {
                        "candidate_count": formal_preflight["candidate_count"],
                        "decision_counts": formal_preflight["decision_counts"],
                        "expensive_adm_or_global_energy_run": False,
                        "family_decision_counts": formal_preflight[
                            "family_decision_counts"
                        ],
                        "gate_counts": formal_preflight["gate_counts"],
                        "next_promotion_hook": formal_preflight[
                            "next_promotion_hook"
                        ],
                        "work_state_counts": formal_preflight[
                            "work_state_counts"
                        ],
                        "promotion_admission": {
                            "decision_counts": promotion_admission[
                                "decision_counts"
                            ],
                            "downstream_expensive_execution_started": False,
                            "eligible_candidate_count": promotion_admission[
                                "eligible_candidate_count"
                            ],
                            "preflight_blocked_excluded_count": promotion_admission[
                                "preflight_blocked_excluded_count"
                            ],
                            "target_queue_counts": promotion_admission[
                                "target_queue_counts"
                            ],
                            "work_state_counts": promotion_admission[
                                "work_state_counts"
                            ],
                            "family_formal_execution": {
                                "aether": {
                                    "candidate_count": aether_candidate_formal[
                                        "candidate_count"
                                    ],
                                    "decision_counts": aether_candidate_formal[
                                        "decision_counts"
                                    ],
                                    "formal_pass_count": aether_candidate_formal[
                                        "formal_pass_count"
                                    ],
                                    "gate_finding_counts": aether_candidate_formal[
                                        "gate_finding_counts"
                                    ],
                                },
                                "g2": {
                                    "predecessor_blocker_counts": g2_candidate_formal[
                                        "blocker_counts"
                                    ],
                                    "candidate_count": g2_candidate_formal[
                                        "candidate_count"
                                    ],
                                    "predecessor_decision_counts": g2_candidate_formal[
                                        "decision_counts"
                                    ],
                                    "decision_counts": g2_nonmaximal_followup[
                                        "decision_counts"
                                    ],
                                    "full_formal_pass_count": g2_nonmaximal_followup[
                                        "full_formal_pass_count"
                                    ],
                                    "general_nonmaximal_positive_mass_pass_count": g2_nonmaximal_followup[
                                        "general_nonmaximal_positive_mass_pass_count"
                                    ],
                                    "actual_initial_data_set_instantiated": False,
                                    "cell_preservation_or_global_evolution_proved": False,
                                    "solar_readiness": {
                                        "analytic_prediction_pass_count": g2_solar_readiness[
                                            "candidate_analytic_prediction_pass_count"
                                        ],
                                        "conditional_static_source_class_pass_count": g2_solar_readiness[
                                            "conditional_static_source_class_pass_count"
                                        ],
                                        "decision_counts": g2_solar_readiness[
                                            "decision_counts"
                                        ],
                                        "real_solar_bundle_count": g2_solar_readiness[
                                            "real_solar_bundle_count"
                                        ],
                                        "observational_data_opened": False,
                                        "registration_advance": g2_solar_heldout_transfer[
                                            "registration_advance_per_candidate"
                                        ],
                                        "held_out_target_access_count": g2_solar_heldout_transfer[
                                            "held_out_target_access_count"
                                        ],
                                        "primary_record_access_count": g2_solar_heldout_transfer[
                                            "primary_record_access_count"
                                        ],
                                        "real_data_pass_count": g2_solar_heldout_transfer[
                                            "real_data_pass_count"
                                        ],
                                        "first_missing_premise": g2_solar_heldout_transfer[
                                            "first_missing_premise"
                                        ],
                                    },
                                    "work_state_counts": g2_candidate_formal[
                                        "work_state_counts"
                                    ],
                                },
                                "g3": {
                                    "blocker_counts": g3_candidate_formal[
                                        "blocker_counts"
                                    ],
                                    "candidate_count": g3_candidate_formal[
                                        "candidate_count"
                                    ],
                                    "decision_counts": g3_candidate_formal[
                                        "decision_counts"
                                    ],
                                    "full_formal_pass_count": g3_candidate_formal[
                                        "full_formal_pass_count"
                                    ],
                                    "gate_counts": g3_candidate_formal[
                                        "gate_counts"
                                    ],
                                    "work_state_counts": g3_candidate_formal[
                                        "work_state_counts"
                                    ],
                                },
                                "g4_followup": {
                                    "candidate_count": g4_scalable_formal_followup[
                                        "candidate_count"
                                    ],
                                    "decision_counts": g4_scalable_formal_followup[
                                        "decision_counts"
                                    ],
                                    "equivalent_parameter_cell_alias_count": g4_scalable_formal_followup[
                                        "equivalent_parameter_cell_alias_count"
                                    ],
                                    "formal_followup_decision": g4_scalable_formal_followup[
                                        "formal_followup_decision"
                                    ],
                                    "original_preflight_decision": g4_scalable_formal_followup[
                                        "preflight_decision"
                                    ],
                                    "transfer_method": g4_equivalence["method"],
                                },
                            },
                        },
                    },
                },
            },
            "structural_metrics": {
                "alias_count": scalable_structural_metrics["alias_count"],
                "candidate_count": scalable_structural_metrics["candidate_count"],
                "formal_decision_counts": scalable_structural_metrics[
                    "formal_decision_counts"
                ],
                "measurement_counts": scalable_structural_metrics[
                    "structural_measurement_counts"
                ],
                "simplicity_pareto_front": scalable_structural_metrics[
                    "simplicity_pareto_front"
                ],
                "simplicity_top10": scalable_structural_metrics["simplicity_top10"],
                "alias_multiplicity_top10": scalable_structural_metrics[
                    "alias_multiplicity_top10"
                ],
                "scientific_validity_inference": False,
            },
            "explanation_dossiers": {
                "alias_count": scalable_explanation_dossiers["alias_count"],
                "candidate_count": scalable_explanation_dossiers["candidate_count"],
                "formal_decision_counts": scalable_explanation_dossiers[
                    "formal_decision_counts"
                ],
                "hierarchy_node_status_counts": scalable_explanation_dossiers[
                    "hierarchy_node_status_counts"
                ],
                "observational_data_opened": False,
                "dossier_registry_root_sha256": scalable_explanation_dossiers[
                    "provenance"
                ]["dossier_registry_root_sha256"],
            },
            "staged_epoch": {
                "stage_count": scalable_campaign_epoch["stage_count"],
                "sealed_epoch_counts": scalable_campaign_epoch[
                    "sealed_epoch_counts"
                ],
                "next_epoch_readiness": scalable_campaign_epoch[
                    "next_epoch_readiness"
                ],
                "reviewed_future_chunk": {
                    "input_cell_count": scalable_future_parameter_chunk[
                        "input_cell_count"
                    ],
                    "disposition_counts": scalable_future_parameter_chunk[
                        "disposition_counts"
                    ],
                    "preflight": {
                        "candidate_count": scalable_future_formal_preflight[
                            "candidate_count"
                        ],
                        "decision_counts": scalable_future_formal_preflight[
                            "decision_counts"
                        ],
                        "family_counts": scalable_future_formal_preflight[
                            "family_counts"
                        ],
                        "first_blocker_counts": scalable_future_formal_preflight[
                            "first_blocker_counts"
                        ],
                        "full_candidate_specific_formal_completion_claimed": False,
                        "promotion": scalable_future_formal_preflight["promotion"],
                    },
                    "family_followup": {
                        "aether": {
                            "candidate_count": future_aether_constraint_followup[
                                "candidate_count"
                            ],
                            "decision_counts": future_aether_constraint_followup[
                                "decision_counts"
                            ],
                            "formal_pass_count": 0,
                            "exact_negative_local_twist_witness_count": future_aether_formal_followup[
                                "exact_negative_local_twist_witness_count"
                            ],
                            "witness_tilt_squared_counts": future_aether_formal_followup[
                                "witness_tilt_squared_counts"
                            ],
                            "global_tilt_strata_counts": future_aether_formal_followup[
                                "global_tilt_strata_counts"
                            ],
                            "explicit_affine_ansatz_constraint_reject_count": future_aether_constraint_followup[
                                "explicit_affine_ansatz_constraint_reject_count"
                            ],
                            "nonzero_Hamiltonian_constraint_residual_count": future_aether_constraint_followup[
                                "nonzero_Hamiltonian_constraint_residual_count"
                            ],
                            "nonzero_momentum_constraint_residual_count": future_aether_constraint_followup[
                                "nonzero_momentum_constraint_residual_count"
                            ],
                            "undefined_AE_boundary_contribution_count": future_aether_constraint_followup[
                                "undefined_AE_boundary_contribution_count"
                            ],
                            "constraint_satisfying_negative_total_energy_datum_count": 0,
                            "first_blocker_counts": future_aether_constraint_followup[
                                "first_blocker_counts"
                            ],
                            "candidate_rejection_authorized_count": 0,
                        },
                        "g3": {
                            "candidate_count": future_g3_action_bound_followup[
                                "candidate_count"
                            ],
                            "decision_counts": future_g3_action_bound_followup[
                                "decision_counts"
                            ],
                            "all_direction_single_center_pass_count": future_g3_domain_followup[
                                "all_direction_single_center_pass_count"
                            ],
                            "domain_registration_filled_field_count": future_g3_action_bound_followup[
                                "domain_registration_filled_field_count"
                            ],
                            "domain_registration_missing_field_count": future_g3_action_bound_followup[
                                "domain_registration_missing_field_count"
                            ],
                            "full_Delta_N_derivation_pass_count": future_g3_action_bound_followup[
                                "full_Delta_N_derivation_pass_count"
                            ],
                            "nonzero_componentwise_box_pass_count": future_g3_action_bound_followup[
                                "nonzero_componentwise_box_pass_count"
                            ],
                            "uniform_principal_common_cone_pass_count": future_g3_action_bound_followup[
                                "uniform_principal_common_cone_pass_count"
                            ],
                            "uniform_Delta_N_coercivity_pass_count": future_g3_action_bound_followup[
                                "uniform_Delta_N_coercivity_pass_count"
                            ],
                            "periodic_distributed_Dirac_pass_count": future_g3_action_bound_followup[
                                "periodic_distributed_Dirac_pass_count"
                            ],
                            "asymptotically_flat_Dirac_pass_count": future_g3_action_bound_followup[
                                "asymptotically_flat_Dirac_pass_count"
                            ],
                            "global_energy_pass_count": future_g3_action_bound_followup[
                                "global_energy_pass_count"
                            ],
                            "full_formal_pass_count": future_g3_action_bound_followup[
                                "full_formal_pass_count"
                            ],
                            "first_blocker_counts": future_g3_action_bound_followup[
                                "first_blocker_counts"
                            ],
                        },
                    },
                    "action_dossiers": {
                        "candidate_count": future_candidate_action_dossier[
                            "candidate_count"
                        ],
                        "decision_counts": future_candidate_action_dossier[
                            "decision_counts"
                        ],
                        "ranked_candidate_count": 0,
                        "dossier_registry_root_sha256": future_candidate_action_dossier[
                            "dossier_registry_root_sha256"
                        ],
                        "records": future_candidate_action_dossier["dossiers"],
                    },
                },
            },
        },
        "quartic_nonlinear_closure": {
            "candidate_count": quartic_counts["selected"],
            "coordinate_pair_partition": quartic_pairs,
            "quadratic_deltaK_two_jet": {
                "closed_candidate_count": quartic_counts[
                    "reference_two_jets_closed"
                ],
                "closed_derivative_orders": quartic_control[
                    "reference_jet_orders_closed"
                ],
                "D2_coordinate_linf_to_Frobenius_ceiling": quartic_tc2_quadratic_deltak[
                    "quadratic_D2_envelopes"
                ][0]["D2_deltaK_coordinate_linf_to_Frobenius_integer_ceiling"],
                "full_tube_Sylvester_identity_closed": False,
            },
            "diagonal_third_jet": {
                "active_direction_count": quartic_third_slice[
                    "active_coordinate_directions"
                ],
                "diagonal_triples_closed": quartic_third_counts[
                    "symbolic_parameter_diagonal_third_jet_passes"
                ],
                "candidate_direction_evaluations": quartic_third_counts[
                    "candidate_direction_evaluations"
                ],
                "candidate_direction_solvable": quartic_third_counts[
                    "candidate_direction_solvable"
                ],
                "candidate_direction_obstructed": quartic_third_counts[
                    "candidate_direction_obstructed"
                ],
                "full_active_symmetric_triple_count": quartic_third_slice[
                    "full_symmetric_triples_in_41_direction_sector"
                ],
                "remaining_mixed_triples": 12_172,
                "mixed_third_jet_closures": 128,
            },
            "mixed_third_jet_chunk": {
                "chunk_offset": quartic_mixed_64_contract["chunk_offset"],
                "latest_chunk_processed_count": quartic_mixed_64_contract[
                    "processed_count"
                ],
                "processed_count": 128,
                "next_offset": quartic_mixed_64_contract["next_offset"],
                "triple_kind_counts": quartic_mixed_64_counts[
                    "triple_kind_counts"
                ],
                "symbolic_parameter_compatible": 128,
                "latest_candidate_evaluations": quartic_mixed_64_counts[
                    "candidate_evaluations"
                ],
                "candidate_evaluations": 1_536,
                "candidate_solvable": 1_536,
                "candidate_obstructed": 0,
                "remaining_mixed_triples": quartic_mixed_64_counts[
                    "mixed_triples_remaining"
                ],
                "resume_tip_sha256": quartic_mixed_64_contract[
                    "resume_tip_sha256"
                ],
                "service_decision": quartic_tc2_mixed_third_jet_continuation_status[
                    "decision"
                ],
                "full_mixed_sector_closed": False,
            },
            "closure_counts": {
                key: quartic_counts[key]
                for key in (
                    "full_tube_Sylvester_identities",
                    "full_variable_CK1_closures",
                    "CK3_closures",
                    "TC2_closures",
                    "B7_closures",
                    "global_H7_closures",
                    "lifespans_proved",
                )
            },
            "first_missing_premise": (
                "remaining_12172_polarized_mixed_third_sylvester_jets_then_"
                "fourth_and_higher_remainder_or_nonlinear_range_theorem"
            ),
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
            "normalized_followup_outcomes": {"pass": 2, "reject": 0, "block": 8},
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
    latest = _parse_utc(watchdog["latest_event_utc"])
    age = (timestamp - latest).total_seconds() if latest else None
    threshold = int(config["watchdog_stale_after_seconds"])
    scheduler_readiness = _scheduler_readiness(watchdog, resource, timestamp)
    next_ready_values = sorted(
        _parse_utc(lane["earliest_future_not_before_utc"])
        for lane in scheduler_readiness.values()
        if lane["earliest_future_not_before_utc"] is not None
    )
    next_ready = next_ready_values[0] if next_ready_values else None
    scheduled_times = sorted(
        parsed
        for row in watchdog.get("queued_task_schedule", [])
        if (parsed := _parse_utc(row.get("not_before_utc"))) is not None
    )
    scheduled_event_anchor = next(
        (
            value
            for value in scheduled_times
            if latest is None or value > latest
        ),
        None,
    )
    freshness_deadline = (
        latest + timedelta(seconds=threshold) if latest is not None else None
    )
    if scheduled_event_anchor is not None:
        scheduled_deadline = scheduled_event_anchor + timedelta(seconds=threshold)
        freshness_deadline = (
            scheduled_deadline
            if freshness_deadline is None
            else max(freshness_deadline, scheduled_deadline)
        )
    freshness_stale = freshness_deadline is None or timestamp > freshness_deadline
    volatile = {
        "sampled_at_utc": timestamp.isoformat(),
        "physical_gpu": dict(physical_gpu) if physical_gpu is not None else sample_nvidia_smi(),
        "campaign_watchdog_freshness": {
            "latest_event_utc": watchdog["latest_event_utc"],
            "age_seconds": age,
            "expected_next_event_not_before_utc": (
                scheduled_event_anchor.isoformat()
                if scheduled_event_anchor is not None
                else None
            ),
            "freshness_deadline_utc": (
                freshness_deadline.isoformat()
                if freshness_deadline is not None
                else None
            ),
            "state": (
                "stale"
                if freshness_stale
                else "scheduled_idle"
                if next_ready is not None and next_ready > timestamp
                else "fresh"
            ),
            "stale": freshness_stale,
            "stale_source_reason": (
                "no_events_or_scheduled_work"
                if freshness_deadline is None
                else f"no_event_by_{freshness_deadline.isoformat()}"
                if freshness_stale
                else None
            ),
        },
        "scheduler_readiness": scheduler_readiness,
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


def _resolve_inside(project_root: Path, value: str) -> Path:
    root = project_root.resolve()
    candidate = (root / value).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("output or input path escapes project root")
    return candidate


def _bounded_write(project_root: Path, path: Path, payload: bytes, maximum_bytes: int) -> None:
    if not 4096 <= maximum_bytes <= 4 * 1024 * 1024:
        raise ValueError("maximum output bytes must be between 4096 and 4194304")
    target = path.resolve()
    root = project_root.resolve()
    if target == root or root not in target.parents:
        raise ValueError("output path escapes project root")
    if len(payload) > maximum_bytes:
        raise RuntimeError(
            f"bounded output exceeds maximum: {len(payload)} > {maximum_bytes} bytes"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def _load_snapshot(path: Path) -> dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "core",
        "core_content_sha256",
        "volatile",
    }:
        raise ValueError("unified status snapshot shape is invalid")
    if _sha(snapshot["core"]) != snapshot["core_content_sha256"]:
        raise ValueError("unified status snapshot core hash mismatch")
    _assert_redacted(snapshot)
    return snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Sigma engine status exporter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh = subparsers.add_parser("refresh", help="refresh JSON and optional HTML")
    refresh.add_argument("--project-root", default=".")
    refresh.add_argument("--config", default="configs/unified_engine_status.json")
    refresh.add_argument("--output", default="runs/engine/unified-engine-status-refresh.json")
    refresh.add_argument("--dashboard-output")
    refresh.add_argument(
        "--leaderboard-config", default="configs/scientific_leaderboards.json"
    )
    refresh.add_argument("--disable-leaderboards", action="store_true")
    refresh.add_argument("--maximum-output-bytes", type=int, default=3_145_728)
    refresh.add_argument("--disable-gpu-sample", action="store_true")
    refresh.add_argument("--sampled-at-utc")
    dashboard = subparsers.add_parser("export-dashboard", help="render existing JSON as HTML")
    dashboard.add_argument("--project-root", default=".")
    dashboard.add_argument("--snapshot", default="runs/engine/unified-engine-status-refresh.json")
    dashboard.add_argument("--output", default="runs/engine/unified-engine-dashboard.html")
    dashboard.add_argument("--maximum-output-bytes", type=int, default=3_145_728)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .unified_engine_dashboard import render_dashboard, validate_dashboard_input

    arguments = _parser().parse_args(argv)
    root = Path(arguments.project_root).resolve()
    if arguments.command == "refresh":
        config_path = _resolve_inside(root, arguments.config)
        output_path = _resolve_inside(root, arguments.output)
        timestamp = _parse_utc(arguments.sampled_at_utc) if arguments.sampled_at_utc else None
        gpu = (
            {"availability": "disabled", "source": "disabled_by_operator"}
            if arguments.disable_gpu_sample
            else None
        )
        snapshot = build_unified_snapshot(
            root, load_config(config_path), now_utc=timestamp, physical_gpu=gpu
        )
        leaderboard_path = _resolve_inside(root, arguments.leaderboard_config)
        if not arguments.disable_leaderboards and leaderboard_path.is_file():
            from .scientific_leaderboards import (
                build_scientific_leaderboards,
                load_leaderboard_config,
            )

            previous = None
            if output_path.is_file():
                try:
                    previous_snapshot = _load_snapshot(output_path)
                    previous = previous_snapshot["core"].get("scientific_leaderboards")
                except (OSError, ValueError, json.JSONDecodeError):
                    previous = None
            snapshot["core"]["scientific_leaderboards"] = build_scientific_leaderboards(
                root, load_leaderboard_config(leaderboard_path), previous
            )
            snapshot["core_content_sha256"] = _sha(snapshot["core"])
            _assert_redacted(snapshot)
        json_bytes = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
        html_bytes = None
        dashboard_path = None
        if arguments.dashboard_output:
            dashboard_path = _resolve_inside(root, arguments.dashboard_output)
            html_bytes = render_dashboard(snapshot).encode()
            if len(html_bytes) > arguments.maximum_output_bytes:
                raise RuntimeError("bounded dashboard output exceeds maximum")
        if len(json_bytes) > arguments.maximum_output_bytes:
            raise RuntimeError("bounded JSON output exceeds maximum")
        if len(json_bytes) + (len(html_bytes) if html_bytes is not None else 0) > (
            arguments.maximum_output_bytes
        ):
            raise RuntimeError("bounded combined output exceeds maximum")
        _bounded_write(root, output_path, json_bytes, arguments.maximum_output_bytes)
        if dashboard_path is not None and html_bytes is not None:
            _bounded_write(root, dashboard_path, html_bytes, arguments.maximum_output_bytes)
        print(json.dumps({
            "core_content_sha256": snapshot["core_content_sha256"],
            "json_bytes": len(json_bytes),
            "dashboard_bytes": len(html_bytes) if html_bytes is not None else 0,
        }, sort_keys=True))
        return 0
    snapshot_path = _resolve_inside(root, arguments.snapshot)
    output_path = _resolve_inside(root, arguments.output)
    snapshot = _load_snapshot(snapshot_path)
    validate_dashboard_input(snapshot, _sha(snapshot["core"]))
    html_bytes = render_dashboard(snapshot).encode()
    _bounded_write(root, output_path, html_bytes, arguments.maximum_output_bytes)
    print(json.dumps({
        "core_content_sha256": snapshot["core_content_sha256"],
        "dashboard_bytes": len(html_bytes),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
