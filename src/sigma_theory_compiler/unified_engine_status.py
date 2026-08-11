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


def _read_mixed_third_jet_supervisor_status(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    relative = config.get("mixed_third_jet_supervisor_status")
    if relative is None:
        return {"availability": "not_configured"}
    root = project_root.resolve()
    path = (root / str(relative)).resolve()
    if root not in path.parents:
        raise ValueError("mixed-third-jet supervisor status escapes project root")
    if not path.is_file():
        return {"availability": "configured_not_started"}
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("mixed-third-jet supervisor status is not an object")
    claimed = value.get("content_sha256")
    without_hash = dict(value)
    without_hash.pop("content_sha256", None)
    if (
        not isinstance(claimed, str)
        or not _SHA256_RE.fullmatch(claimed)
        or _sha(without_hash) != claimed
        or any(value.get("claims", {}).values())
        or value.get("chunks_advanced", 0) < 0
        or value.get("epochs_completed", 0) < value.get("chunks_advanced", 0)
        or value.get("next_offset", 0) < 0
        or value.get("remaining_mixed_triples", 0) < 0
    ):
        raise ValueError("mixed-third-jet supervisor status is not fail-closed")
    return {
        "availability": "available",
        "state": value.get("state"),
        "alive": value.get("alive"),
        "pid": value.get("pid"),
        "epochs_completed": value.get("epochs_completed"),
        "chunks_advanced": value.get("chunks_advanced"),
        "next_offset": value.get("next_offset"),
        "remaining_mixed_triples": value.get("remaining_mixed_triples"),
        "prior_resume_sha256": value.get("prior_resume_sha256"),
        "stop_reason": value.get("stop_reason"),
        "content_sha256": claimed,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "claims": value.get("claims"),
    }


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
            str(row["name"]) for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
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
            count
            for (task_type, state), count in states.items()
            if task_type in allowed and state in active_states
        )
        queued = sum(
            count
            for (task_type, state), count in states.items()
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
            count
            for (task_type, state), count in states.items()
            if task_type not in claimed_types and state in active_states
        ),
        "queued": sum(
            count
            for (task_type, state), count in states.items()
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
        "earliest_future_not_before_utc": (future_times[0].isoformat() if future_times else None),
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


def sample_host_cpu() -> dict[str, Any]:
    """Take one bounded host CPU/RAM sample, separate from scheduler leases."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "availability": "available",
            "source": "psutil_host_sample",
            "utilization_percent": float(psutil.cpu_percent(interval=0.05)),
            "logical_processors": int(psutil.cpu_count(logical=True) or 0),
            "physical_cores": int(psutil.cpu_count(logical=False) or 0),
            "memory_used_bytes": int(memory.used),
            "memory_total_bytes": int(memory.total),
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return {
            "availability": "unavailable",
            "source": "psutil_host_sample",
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
    physical_cpu: Mapping[str, Any] | None = None,
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
    mixed_third_jet_supervisor = _read_mixed_third_jet_supervisor_status(root, config)

    streaming = sources["billion_streaming"]
    promotion = sources["promotion_overlay"]
    parameter = sources["grammar_parameter_cells"]
    parameter_expansion = sources["grammar_parameter_cell_expansion_service"]
    grammar_seed_manifest = sources["grammar_v3_seed_manifest"]
    parameter_manifest = sources["grammar_parameter_cell_manifest"]
    parameter_compilation = sources["grammar_parameter_cell_compilation"]
    generated_candidate_formal_export = sources["generated_candidate_formal_export"]
    generated_candidate_metric_variation = sources[
        "generated_candidate_metric_variation_specialization"
    ]
    formal_preflight = sources["grammar_v3_formal_preflight"]
    promotion_admission = sources["grammar_v3_promotion_admission"]
    g2_candidate_formal = sources["grammar_v3_g2_candidate_formal"]
    g2_nonmaximal_followup = sources["grammar_v3_g2_nonmaximal_positive_mass_followup"]
    g2_solar_readiness = sources["grammar_v3_g2_solar_readiness"]
    g2_solar_heldout_transfer = sources["grammar_v3_g2_solar_heldout_transfer"]
    scalable_campaign_epoch = sources["scalable_campaign_epoch"]
    scalable_future_parameter_chunk = sources["scalable_future_parameter_chunk"]
    scalable_future_formal_preflight = sources["scalable_future_formal_preflight"]
    future_aether_formal_followup = sources["future_aether_formal_followup"]
    future_aether_constraint_followup = sources["future_aether_constraint_followup"]
    future_aether_pure_twist_ae_no_go = sources["future_aether_pure_twist_ae_no_go"]
    future_aether_weak_field_ae_constraint_gate = sources[
        "future_aether_weak_field_ae_constraint_gate"
    ]
    future_aether_finite_amplitude_negative_seed_gate = sources[
        "future_aether_finite_amplitude_negative_seed_gate"
    ]
    future_aether_nonlinear_lift_characteristic_gate = sources[
        "future_aether_nonlinear_lift_characteristic_gate"
    ]
    future_aether_regular_adm_inverse_margin_gate = sources[
        "future_aether_regular_adm_inverse_margin_gate"
    ]
    future_aether_weighted_ift_contract_gate = sources["future_aether_weighted_ift_contract_gate"]
    future_aether_weighted_reference_operator_gate = sources[
        "future_aether_weighted_reference_operator_gate"
    ]
    future_aether_fixed_free_data_principal_gate = sources[
        "future_aether_fixed_free_data_principal_gate"
    ]
    future_aether_finite_tilt_york_symbol_gate = sources[
        "future_aether_finite_tilt_york_symbol_gate"
    ]
    future_aether_principal_inverse_fredholm_gate = sources[
        "future_aether_principal_inverse_fredholm_gate"
    ]
    future_g3_domain_followup = sources["future_g3_domain_followup"]
    future_g3_action_bound_followup = sources["future_g3_action_bound_followup"]
    future_g3_af_transition_obstruction = sources["future_g3_af_transition_obstruction"]
    future_g3_nonunitary_af_constraint_gate = sources["future_g3_nonunitary_af_constraint_gate"]
    future_g3_radial_conformal_constraint_reduction = sources[
        "future_g3_radial_conformal_constraint_reduction"
    ]
    future_g3_radial_lichnerowicz_bvp_no_go = sources["future_g3_radial_lichnerowicz_bvp_no_go"]
    future_g3_nonradial_york_bounded_mean_curvature_no_go = sources[
        "future_g3_nonradial_york_bounded_mean_curvature_no_go"
    ]
    future_g3_york_mean_curvature_frontier = sources["future_g3_york_mean_curvature_frontier"]
    future_g3_york_analytic_threshold = sources["future_g3_york_analytic_threshold"]
    future_g3_york_tracefree_compensation = sources["future_g3_york_tracefree_compensation"]
    future_g3_general_geometry_curvature_shortfall = sources[
        "future_g3_general_geometry_curvature_shortfall"
    ]
    future_g3_general_geometry_surplus_mismatch = sources[
        "future_g3_general_geometry_surplus_mismatch"
    ]
    future_candidate_action_dossier = sources["future_candidate_action_dossier"]
    g3_candidate_formal = sources["grammar_v3_g3_candidate_formal"]
    g4_scalable_formal_followup = sources["grammar_v3_g4_scalable_formal_followup"]
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
    g4_galaxy_calibration_evaluation = sources["g4_galaxy_calibration_evaluation"]
    g4_galaxy_prediction_contract_transform = sources["g4_galaxy_prediction_contract_transform"]
    g4_galaxy_manifest_bundle_tooling = sources["g4_galaxy_manifest_bundle_tooling"]
    g4_galaxy_source_registry_admission = sources["g4_galaxy_source_registry_admission"]
    quartic_tc2_quadratic_deltak = sources["quartic_tc2_quadratic_deltak_extension"]
    quartic_tc2_diagonal_third_jet = sources["quartic_tc2_diagonal_third_jet"]
    quartic_tc2_mixed_third_jet_basis_reduction = sources[
        "quartic_tc2_mixed_third_jet_basis_reduction"
    ]
    quartic_tc2_mixed_third_jet_reranked_reduction = sources[
        "quartic_tc2_mixed_third_jet_reranked_reduction"
    ]
    quartic_tc2_fourth_jet_range_obligations = sources[
        "quartic_tc2_fourth_jet_range_obligations"
    ]
    quartic_tc2_fourth_jet_chunk_0 = sources["quartic_tc2_fourth_jet_chunk_0"]
    quartic_tc2_fourth_jet_chunk_32 = sources["quartic_tc2_fourth_jet_chunk_32"]
    quartic_tc2_fourth_jet_chunk_64 = sources["quartic_tc2_fourth_jet_chunk_64"]
    quartic_tc2_fourth_jet_chunk_96 = sources["quartic_tc2_fourth_jet_chunk_96"]
    quartic_tc2_fourth_jet_checkpoint = sources["quartic_tc2_fourth_jet_checkpoint"]
    quartic_tc2_fourth_jet_status = sources["quartic_tc2_fourth_jet_status"]
    quartic_tc2_reranked_obligation_chunks = tuple(
        sources[f"quartic_tc2_reranked_obligation_chunk_{offset}"]
        for offset in (0, 64, 128, 192, 256, 320, 384)
    )
    quartic_tc2_reranked_obligation_checkpoint = sources[
        "quartic_tc2_reranked_obligation_checkpoint"
    ]
    quartic_tc2_reranked_obligation_status = sources["quartic_tc2_reranked_obligation_status"]
    quartic_tc2_mixed_third_jet_chunk = sources["quartic_tc2_mixed_third_jet_chunk"]
    quartic_tc2_mixed_third_jet_chunk_64 = sources["quartic_tc2_mixed_third_jet_chunk_64"]
    quartic_tc2_mixed_third_jet_chunk_128 = sources["quartic_tc2_mixed_third_jet_chunk_128"]
    quartic_tc2_mixed_third_jet_checkpoint = sources["quartic_tc2_mixed_third_jet_checkpoint"]
    quartic_tc2_mixed_third_jet_continuation_status = sources[
        "quartic_tc2_mixed_third_jet_continuation_status"
    ]
    quartic_tc2_mixed_third_jet_parallel_chunks = tuple(
        sources[f"quartic_tc2_mixed_third_jet_parallel_chunk_{offset}"]
        for offset in range(192, 1_600, 64)
    )
    quartic_tc2_mixed_third_jet_parallel_checkpoint = sources[
        "quartic_tc2_mixed_third_jet_parallel_checkpoint"
    ]
    quartic_tc2_mixed_third_jet_parallel_status = sources[
        "quartic_tc2_mixed_third_jet_parallel_status"
    ]
    quartic_tc2_mixed_third_jet_parallel_supervisor_readiness = sources[
        "quartic_tc2_mixed_third_jet_parallel_supervisor_readiness"
    ]
    if (
        scalable_structural_metrics.get("candidate_count") != 163
        or scalable_structural_metrics.get("alias_count") != 93
        or scalable_structural_metrics.get("formal_decision_counts")
        != {"blocked": 158, "pass": 3, "reject": 2}
        or scalable_structural_metrics.get("structural_measurement_counts") != {"measured": 163}
        or scalable_structural_metrics.get("simplicity_pareto_front", {}).get("candidate_count")
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
        or followup.get("packet_state_counts") != {"deferred_missing_evaluator": 0, "succeeded": 10}
        or followup.get("candidate_scientific_decisions_changed") != 1
        or followup.get("reviewed_evaluator_invocation_count") != 10
        or followup.get("missing_evaluator_executions") != 0
        or followup.get("deferred_packets") != []
        or followup_queue.get("followup_decision_counts") != {"blocked": 8, "pass": 2}
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
        or set(parameter_compilation.get("structural_gate_pass_counts", {}).values()) != {256}
        or parameter_compilation.get("negative_control_counts") != {"reject": 5}
        or parameter_compilation.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
        or parameter_compilation.get("parameter_cell_manifest_binding", {}).get("content_sha256")
        != parameter_manifest.get("content_sha256")
    ):
        raise ValueError("grammar parameter-cell compilation is inconsistent")
    if (
        generated_candidate_formal_export.get("candidate_count") != 163
        or generated_candidate_formal_export.get("action_export_counts")
        != {"exact_rendered": 163, "rejected": 0, "sandbox_parsed_and_canonicalised": 163}
        or generated_candidate_formal_export.get("metric_variation_counts")
        != {
            "executed_by_this_campaign": 0,
            "formal_passes_inferred": 0,
            "reviewed_adapter_routes_bound": 163,
        }
        or generated_candidate_formal_export.get("sandbox_receipt", {}).get("status") != "pass"
        or generated_candidate_formal_export.get("sandbox_receipt", {}).get(
            "network_namespace_created"
        )
        is not True
        or generated_candidate_formal_export.get("sandbox_receipt", {}).get(
            "user_namespace_created"
        )
        is not True
        or generated_candidate_formal_export.get("sandbox_receipt", {}).get("shell_invoked")
        is not False
        or generated_candidate_formal_export.get("sandbox_receipt", {}).get("marker_count") != 1
        or generated_candidate_formal_export.get("observational_data_opened") is not False
        or generated_candidate_formal_export.get("paid_llm_spend_usd") != 0.0
        or generated_candidate_formal_export.get("data_eligibility")
        != {
            "dark_matter_or_halo_inputs": False,
            "observational_data_opened": False,
            "paid_llm_calls": False,
            "passed": True,
            "redshift_distance_inputs": False,
        }
        or len(generated_candidate_formal_export.get("candidate_records", [])) != 163
        or generated_candidate_formal_export.get("first_missing_premise")
        != "candidate_specific_metric_variation_execution_from_the_generated_action_export_for_each_action_hash_and_future_operator_family"
    ):
        raise ValueError("generated candidate formal action export is inconsistent")
    metric_counts = generated_candidate_metric_variation.get(
        "metric_variation_execution_counts", {}
    )
    if (
        generated_candidate_metric_variation.get("candidate_count") != 163
        or metric_counts
        != {
            "aether_formal_control_bound": 128,
            "blocked": 0,
            "candidate_action_hashes_specialized": 163,
            "candidate_euler_expressions_materialized": 163,
            "formal_passes_inferred": 0,
            "rejected": 0,
        }
        or generated_candidate_metric_variation.get("current_operator_families_complete")
        is not True
        or generated_candidate_metric_variation.get(
            "future_unregistered_operator_families_complete"
        )
        is not False
        or generated_candidate_metric_variation.get("observational_data_opened") is not False
        or generated_candidate_metric_variation.get("paid_llm_spend_usd") != 0.0
        or len(generated_candidate_metric_variation.get("candidate_records", [])) != 163
        or any(
            record.get("generic_metric_variation_theorem_bound") is not True
            or record.get("candidate_specialized_euler_expression_materialized") is not True
            or record.get("formal_pass_inferred") is not False
            for record in generated_candidate_metric_variation.get("candidate_records", [])
        )
    ):
        raise ValueError("generated candidate metric specialization is inconsistent")
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
        or promotion_admission.get("downstream_expensive_execution_started") is not False
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
        or g2_candidate_formal.get("general_nonmaximal_global_positive_mass_proved") is not False
        or g2_candidate_formal.get("gate_counts", {}).get(
            "candidate_action_preflight_admission_binding"
        )
        != {"pass": 2}
        or g2_candidate_formal.get("gate_counts", {}).get("restricted_maximal_slice_positive_mass")
        != {"pass": 2}
        or g2_candidate_formal.get("gate_counts", {}).get("general_nonmaximal_positive_mass")
        != {"blocked": 2}
        or g2_candidate_formal.get("promotion_status_binding", {}).get("content_sha256")
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
        or g2_nonmaximal_followup.get("general_nonmaximal_positive_mass_pass_count") != 2
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
        or g2_solar_readiness.get("conditional_static_source_class_pass_count") != 2
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
            or len(record.get("real_solar_readiness", {}).get("missing_registration_fields", []))
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
            for record in g2_solar_heldout_transfer.get("candidate_registrations", [])
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
        or scalable_campaign_epoch.get("next_epoch_readiness", {}).get("state") != "blocked"
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
        or scalable_future_formal_preflight.get("source_deduplicated_candidate_count") != 13
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
        or scalable_future_formal_preflight.get("formal_preflight_completed") is not True
        or scalable_future_formal_preflight.get("full_candidate_specific_formal_completion_claimed")
        is not False
        or scalable_future_formal_preflight.get("promotion")
        != {
            "automatic_downstream_enqueue_performed": False,
            "blocked_pending_exact_domain_registration": 3,
            "eligible_for_candidate_specific_formal_queue": 14,
            "rejected_before_candidate_specific_formal_queue": 2,
        }
        or scalable_future_formal_preflight.get("source_status_binding", {}).get("content_sha256")
        != scalable_future_parameter_chunk.get("content_sha256")
        or scalable_future_formal_preflight.get("observational_data_opened") is not False
        or scalable_future_formal_preflight.get("dark_matter_or_halo_inputs") is not False
        or scalable_future_formal_preflight.get("redshift_distance_inputs") is not False
        or scalable_future_formal_preflight.get("paid_llm_spend_usd") != 0.0
        or len(scalable_future_formal_preflight.get("candidate_records", [])) != 19
    ):
        raise ValueError("scalable future formal preflight is inconsistent")
    if (
        future_aether_formal_followup.get("candidate_count") != 14
        or future_aether_formal_followup.get("input_preflight_pass_count") != 14
        or future_aether_formal_followup.get("decision_counts") != {"blocked": 14}
        or future_aether_formal_followup.get("formal_pass_count") != 0
        or future_aether_formal_followup.get("candidate_rejection_authorized_count") != 0
        or future_aether_formal_followup.get("exact_negative_local_twist_witness_count") != 14
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
        or future_aether_formal_followup.get("source_preflight_binding", {}).get("content_sha256")
        != scalable_future_formal_preflight.get("content_sha256")
        or future_aether_formal_followup.get("full_candidate_specific_formal_completion_claimed")
        is not False
        or future_aether_formal_followup.get("automatic_downstream_enqueue_performed") is not False
        or future_aether_formal_followup.get("observational_data_opened") is not False
        or future_aether_formal_followup.get("dark_matter_or_halo_inputs") is not False
        or future_aether_formal_followup.get("redshift_distance_inputs") is not False
        or future_aether_formal_followup.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_formal_followup.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether formal follow-up is inconsistent")
    if (
        future_aether_constraint_followup.get("candidate_count") != 14
        or future_aether_constraint_followup.get("decision_counts") != {"blocked": 14}
        or future_aether_constraint_followup.get("explicit_affine_ansatz_constraint_reject_count")
        != 14
        or future_aether_constraint_followup.get("nonzero_Hamiltonian_constraint_residual_count")
        != 14
        or future_aether_constraint_followup.get("nonzero_momentum_constraint_residual_count") != 14
        or future_aether_constraint_followup.get("undefined_AE_boundary_contribution_count") != 14
        or future_aether_constraint_followup.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_constraint_followup.get("candidate_rejection_authorized_count") != 0
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
        future_aether_pure_twist_ae_no_go.get("candidate_count") != 14
        or future_aether_pure_twist_ae_no_go.get("decision_counts") != {"blocked": 14}
        or future_aether_pure_twist_ae_no_go.get("bounded_pure_twist_AE_no_go_audit_completed")
        is not True
        or future_aether_pure_twist_ae_no_go.get(
            "flat_static_global_pure_twist_AE_completion_obstructed_count"
        )
        != 14
        or future_aether_pure_twist_ae_no_go.get(
            "compact_cutoff_non_pure_twist_transition_required_count"
        )
        != 14
        or future_aether_pure_twist_ae_no_go.get(
            "normalized_transition_symmetric_gradient_norm_squared_counts"
        )
        != {"6": 8, "10": 4, "34": 2}
        or future_aether_pure_twist_ae_no_go.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_pure_twist_ae_no_go.get("candidate_rejection_authorized_count") != 0
        or future_aether_pure_twist_ae_no_go.get("formal_pass_count") != 0
        or future_aether_pure_twist_ae_no_go.get("first_blocker_counts")
        != {
            "candidate_bound_AE_coupled_constraint_solution_beyond_flat_static_global_pure_twist_class_with_negative_completed_boundary_energy": 14
        }
        or future_aether_pure_twist_ae_no_go.get("symbolic_obstruction_control", {})
        .get("differentiated_Killing_system", {})
        .get("coefficient_rank")
        != 18
        or future_aether_pure_twist_ae_no_go.get("symbolic_obstruction_control", {})
        .get("differentiated_Killing_system", {})
        .get("unknown_count")
        != 18
        or future_aether_pure_twist_ae_no_go.get("symbolic_obstruction_control", {})
        .get("differentiated_Killing_system", {})
        .get("kernel_dimension")
        != 0
        or future_aether_pure_twist_ae_no_go.get("source_embedding_binding", {}).get(
            "content_sha256"
        )
        != future_aether_constraint_followup.get("content_sha256")
        or future_aether_pure_twist_ae_no_go.get("observational_data_opened") is not False
        or future_aether_pure_twist_ae_no_go.get("dark_matter_or_halo_inputs") is not False
        or future_aether_pure_twist_ae_no_go.get("redshift_distance_inputs") is not False
        or future_aether_pure_twist_ae_no_go.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_pure_twist_ae_no_go.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether pure-twist AE no-go is inconsistent")
    if (
        future_aether_weak_field_ae_constraint_gate.get("candidate_count") != 14
        or future_aether_weak_field_ae_constraint_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_weak_field_ae_constraint_gate.get(
            "bounded_weak_field_AE_constraint_gate_completed"
        )
        is not True
        or future_aether_weak_field_ae_constraint_gate.get(
            "weak_field_linearized_constraint_completion_count"
        )
        != 14
        or future_aether_weak_field_ae_constraint_gate.get(
            "strictly_positive_compact_quadratic_energy_count"
        )
        != 14
        or future_aether_weak_field_ae_constraint_gate.get(
            "weak_field_negative_completed_energy_direction_count"
        )
        != 0
        or future_aether_weak_field_ae_constraint_gate.get(
            "finite_amplitude_nonlinear_constraint_completion_count"
        )
        != 0
        or future_aether_weak_field_ae_constraint_gate.get("c2_plus_c3_counts")
        != {"0": 2, "1/16": 3, "1/32": 1, "1/4": 1, "1/8": 3, "3/16": 2, "3/32": 1, "5/32": 1}
        or future_aether_weak_field_ae_constraint_gate.get("formal_pass_count") != 0
        or future_aether_weak_field_ae_constraint_gate.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_weak_field_ae_constraint_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_weak_field_ae_constraint_gate.get("first_blocker_counts")
        != {
            "finite_amplitude_candidate_bound_nonlinear_AE_coupled_constraint_solution_with_negative_completed_boundary_energy_beyond_positive_weak_field_quadratic_regime": 14
        }
        or future_aether_weak_field_ae_constraint_gate.get("source_no_go_binding", {}).get(
            "content_sha256"
        )
        != future_aether_pure_twist_ae_no_go.get("content_sha256")
        or future_aether_weak_field_ae_constraint_gate.get("observational_data_opened") is not False
        or future_aether_weak_field_ae_constraint_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_weak_field_ae_constraint_gate.get("redshift_distance_inputs") is not False
        or future_aether_weak_field_ae_constraint_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_weak_field_ae_constraint_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether weak-field AE constraint gate is inconsistent")
    if (
        future_aether_finite_amplitude_negative_seed_gate.get("candidate_count") != 14
        or future_aether_finite_amplitude_negative_seed_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "compact_finite_amplitude_Aether_seed_count"
        )
        != 14
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "exact_negative_static_source_monopole_count"
        )
        != 14
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "frozen_source_linearized_constraint_completion_count"
        )
        != 14
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "negative_linearized_completed_boundary_energy_coefficient_count"
        )
        != 14
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "full_nonlinear_constraint_completion_count"
        )
        != 0
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "sign_preserving_nonlinear_boundary_completion_count"
        )
        != 0
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_finite_amplitude_negative_seed_gate.get("formal_pass_count") != 0
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_finite_amplitude_negative_seed_gate.get(
            "source_weak_field_binding", {}
        ).get("content_sha256")
        != future_aether_weak_field_ae_constraint_gate.get("content_sha256")
        or future_aether_finite_amplitude_negative_seed_gate.get("observational_data_opened")
        is not False
        or future_aether_finite_amplitude_negative_seed_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_finite_amplitude_negative_seed_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_finite_amplitude_negative_seed_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_finite_amplitude_negative_seed_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether finite-amplitude negative-seed gate is inconsistent")
    if (
        future_aether_nonlinear_lift_characteristic_gate.get("candidate_count") != 14
        or future_aether_nonlinear_lift_characteristic_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "registered_seed_characteristic_crossing_count"
        )
        != 13
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "negative_source_family_forced_characteristic_crossing_count"
        )
        != 11
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "certified_negative_characteristic_free_amplitude_window_count"
        )
        != 2
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "globally_noncharacteristic_candidate_count"
        )
        != 1
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "regular_ADM_implicit_lift_prerequisite_pass_count"
        )
        != 3
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "full_nonlinear_constraint_completion_count"
        )
        != 0
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_nonlinear_lift_characteristic_gate.get("formal_pass_count") != 0
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_nonlinear_lift_characteristic_gate.get("first_blocker_counts")
        != {
            "candidate_bound_weighted_nonlinear_Einstein_Aether_constraint_map_inverse_and_remainder_bound_with_completed_boundary_sign_persistence": 3,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_nonlinear_lift_characteristic_gate.get(
            "source_negative_seed_binding", {}
        ).get("content_sha256")
        != future_aether_finite_amplitude_negative_seed_gate.get("content_sha256")
        or future_aether_nonlinear_lift_characteristic_gate.get("observational_data_opened")
        is not False
        or future_aether_nonlinear_lift_characteristic_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_nonlinear_lift_characteristic_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_nonlinear_lift_characteristic_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_nonlinear_lift_characteristic_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether nonlinear-lift characteristic gate is inconsistent")
    if (
        future_aether_regular_adm_inverse_margin_gate.get("candidate_count") != 14
        or future_aether_regular_adm_inverse_margin_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_regular_adm_inverse_margin_gate.get("regular_ADM_candidate_count") != 3
        or future_aether_regular_adm_inverse_margin_gate.get(
            "forced_characteristic_candidate_count"
        )
        != 11
        or future_aether_regular_adm_inverse_margin_gate.get(
            "uniform_Aether_Legendre_block_inverse_pass_count"
        )
        != 3
        or future_aether_regular_adm_inverse_margin_gate.get(
            "strict_negative_source_margin_pass_count"
        )
        != 3
        or future_aether_regular_adm_inverse_margin_gate.get(
            "weighted_full_constraint_operator_isomorphism_pass_count"
        )
        != 0
        or future_aether_regular_adm_inverse_margin_gate.get(
            "nonlinear_Frechet_remainder_bound_pass_count"
        )
        != 0
        or future_aether_regular_adm_inverse_margin_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_regular_adm_inverse_margin_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_regular_adm_inverse_margin_gate.get("formal_pass_count") != 0
        or future_aether_regular_adm_inverse_margin_gate.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_regular_adm_inverse_margin_gate.get("first_blocker_counts")
        != {
            "candidate_bound_weighted_elliptic_Einstein_Aether_constraint_operator_isomorphism_and_nonlinear_remainder_bound_with_completed_boundary_sign_persistence": 3,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_regular_adm_inverse_margin_gate.get(
            "source_characteristic_binding", {}
        ).get("content_sha256")
        != future_aether_nonlinear_lift_characteristic_gate.get("content_sha256")
        or future_aether_regular_adm_inverse_margin_gate.get("observational_data_opened")
        is not False
        or future_aether_regular_adm_inverse_margin_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_regular_adm_inverse_margin_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_regular_adm_inverse_margin_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_regular_adm_inverse_margin_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether regular-ADM inverse-margin gate is inconsistent")
    if (
        future_aether_weighted_ift_contract_gate.get("candidate_count") != 14
        or future_aether_weighted_ift_contract_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_weighted_ift_contract_gate.get("regular_ADM_candidate_count") != 3
        or future_aether_weighted_ift_contract_gate.get("forced_characteristic_candidate_count")
        != 11
        or future_aether_weighted_ift_contract_gate.get(
            "reference_conformal_York_Aether_block_control_count"
        )
        != 3
        or future_aether_weighted_ift_contract_gate.get(
            "typed_weighted_operator_contract_complete_count"
        )
        != 0
        or future_aether_weighted_ift_contract_gate.get(
            "full_weighted_operator_isomorphism_pass_count"
        )
        != 0
        or future_aether_weighted_ift_contract_gate.get("nonlinear_remainder_bound_pass_count") != 0
        or future_aether_weighted_ift_contract_gate.get("completed_boundary_sign_persistence_count")
        != 0
        or future_aether_weighted_ift_contract_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_weighted_ift_contract_gate.get("formal_pass_count") != 0
        or future_aether_weighted_ift_contract_gate.get("candidate_rejection_authorized_count") != 0
        or future_aether_weighted_ift_contract_gate.get("first_blocker_counts")
        != {
            "candidate_bound_gauge_fixed_weighted_constraint_operator_norm_contract_with_nonlinear_remainder_and_completed_boundary_majorants": 3,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or set(future_aether_weighted_ift_contract_gate.get("missing_contract_field_counts", {}))
        != {
            "weight_delta",
            "domain_space",
            "codomain_space",
            "gauge_fixing",
            "full_linearized_constraint_map",
            "reference_inverse_norm",
            "operator_perturbation_norm",
            "seed_nonlinear_constraint_residual_norm",
            "nonlinear_second_derivative_majorant",
            "completed_boundary_first_derivative_bound",
            "completed_boundary_second_derivative_bound",
        }
        or any(
            count != 3
            for count in future_aether_weighted_ift_contract_gate.get(
                "missing_contract_field_counts", {}
            ).values()
        )
        or future_aether_weighted_ift_contract_gate.get("source_inverse_margin_binding", {}).get(
            "content_sha256"
        )
        != future_aether_regular_adm_inverse_margin_gate.get("content_sha256")
        or future_aether_weighted_ift_contract_gate.get("observational_data_opened") is not False
        or future_aether_weighted_ift_contract_gate.get("dark_matter_or_halo_inputs") is not False
        or future_aether_weighted_ift_contract_gate.get("redshift_distance_inputs") is not False
        or future_aether_weighted_ift_contract_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_weighted_ift_contract_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether weighted-IFT contract gate is inconsistent")
    weighted_reference_records = future_aether_weighted_reference_operator_gate.get(
        "candidate_records", []
    )
    regular_weighted_reference_records = [
        record
        for record in weighted_reference_records
        if record.get("first_blocker")
        == "candidate_bound_Aether_constraint_variable_block_and_off_diagonal_principal_symbol_on_declared_weighted_spaces"
    ]
    if (
        future_aether_weighted_reference_operator_gate.get("candidate_count") != 14
        or future_aether_weighted_reference_operator_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_weighted_reference_operator_gate.get("regular_ADM_candidate_count") != 3
        or future_aether_weighted_reference_operator_gate.get(
            "forced_characteristic_candidate_count"
        )
        != 11
        or future_aether_weighted_reference_operator_gate.get(
            "declared_metric_weighted_contract_count"
        )
        != 3
        or future_aether_weighted_reference_operator_gate.get(
            "metric_reference_principal_ellipticity_pass_count"
        )
        != 3
        or future_aether_weighted_reference_operator_gate.get(
            "metric_reference_trivial_kernel_pass_count"
        )
        != 3
        or future_aether_weighted_reference_operator_gate.get(
            "registered_compact_source_right_inverse_count"
        )
        != 3
        or future_aether_weighted_reference_operator_gate.get(
            "candidate_Aether_constraint_principal_block_pass_count"
        )
        != 0
        or future_aether_weighted_reference_operator_gate.get(
            "full_coupled_Fredholm_operator_defined_count"
        )
        != 0
        or future_aether_weighted_reference_operator_gate.get(
            "full_weighted_operator_isomorphism_pass_count"
        )
        != 0
        or future_aether_weighted_reference_operator_gate.get("computable_full_inverse_norm_count")
        != 0
        or future_aether_weighted_reference_operator_gate.get(
            "nonlinear_remainder_bound_pass_count"
        )
        != 0
        or future_aether_weighted_reference_operator_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_weighted_reference_operator_gate.get("formal_pass_count") != 0
        or future_aether_weighted_reference_operator_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_weighted_reference_operator_gate.get("first_blocker_counts")
        != {
            "candidate_bound_Aether_constraint_variable_block_and_off_diagonal_principal_symbol_on_declared_weighted_spaces": 3,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_weighted_reference_operator_gate.get(
            "source_weighted_ift_binding", {}
        ).get("content_sha256")
        != future_aether_weighted_ift_contract_gate.get("content_sha256")
        or len(weighted_reference_records) != 14
        or len(regular_weighted_reference_records) != 3
        or any(
            record.get("weighted_reference_operator_certificate", {})
            .get("declared_metric_weighted_contract", {})
            .get("weight_delta")
            != "-1/2"
            or record.get("weighted_reference_operator_certificate", {})
            .get("principal_symbol_certificates", {})
            .get("axial_unit_covector", {})
            .get("combined_symbol_eigenvalues")
            != ["2", "2", "8/3", "4"]
            or record.get("weighted_reference_operator_certificate", {})
            .get("decaying_kernel_certificate", {})
            .get("reference_metric_kernel_trivial")
            is not True
            or record.get("weighted_reference_operator_certificate", {}).get(
                "candidate_Aether_constraint_principal_block_derived"
            )
            is not False
            or record.get("candidate_rejection_authorized") is not False
            for record in regular_weighted_reference_records
        )
        or future_aether_weighted_reference_operator_gate.get("observational_data_opened")
        is not False
        or future_aether_weighted_reference_operator_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_weighted_reference_operator_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_weighted_reference_operator_gate.get("paid_llm_spend_usd") != 0.0
    ):
        raise ValueError("future Aether weighted reference-operator gate is inconsistent")
    if (
        future_g3_domain_followup.get("candidate_count") != 3
        or future_g3_domain_followup.get("decision_counts") != {"blocked": 3}
        or future_g3_domain_followup.get("all_direction_single_center_pass_count") != 3
        or future_g3_domain_followup.get("full_Delta_N_derivation_pass_count") != 3
        or future_g3_domain_followup.get("nonzero_componentwise_box_pass_count") != 0
        or future_g3_domain_followup.get("uniform_principal_common_cone_pass_count") != 0
        or future_g3_domain_followup.get("uniform_Delta_N_coercivity_pass_count") != 0
        or future_g3_domain_followup.get("periodic_distributed_Dirac_pass_count") != 0
        or future_g3_domain_followup.get("asymptotically_flat_Dirac_pass_count") != 0
        or future_g3_domain_followup.get("full_formal_pass_count") != 0
        or future_g3_domain_followup.get("first_blocker_counts")
        != {"candidate_bound_nonzero_componentwise_normalized_local_jet_box_values": 3}
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
        or future_g3_action_bound_followup.get("domain_registration_filled_field_count") != 36
        or future_g3_action_bound_followup.get("domain_registration_missing_field_count") != 0
        or future_g3_action_bound_followup.get("nonzero_componentwise_box_pass_count") != 3
        or future_g3_action_bound_followup.get("uniform_principal_common_cone_pass_count") != 3
        or future_g3_action_bound_followup.get("full_Delta_N_derivation_pass_count") != 3
        or future_g3_action_bound_followup.get("uniform_Delta_N_coercivity_pass_count") != 3
        or future_g3_action_bound_followup.get("periodic_distributed_Dirac_pass_count") != 3
        or future_g3_action_bound_followup.get("asymptotically_flat_Dirac_pass_count") != 0
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
        future_g3_af_transition_obstruction.get("candidate_count") != 3
        or future_g3_af_transition_obstruction.get("decision_counts") != {"blocked": 3}
        or future_g3_af_transition_obstruction.get("AF_decaying_gradient_profile_pass_count") != 3
        or future_g3_af_transition_obstruction.get("AF_principal_common_cone_profile_pass_count")
        != 3
        or future_g3_af_transition_obstruction.get("flat_reference_constraint_ansatz_reject_count")
        != 3
        or future_g3_af_transition_obstruction.get("AF_unitary_lapse_Dirac_pass_count") != 0
        or future_g3_af_transition_obstruction.get("AF_Einstein_constraint_solution_pass_count")
        != 0
        or future_g3_af_transition_obstruction.get("global_hamiltonian_energy_pass_count") != 0
        or future_g3_af_transition_obstruction.get("full_formal_pass_count") != 0
        or future_g3_af_transition_obstruction.get("first_blocker_counts")
        != {"bounded_global_unitary_Delta_N_inverse_on_candidate_AF_transition_profile": 3}
        or future_g3_af_transition_obstruction.get("synthetic_fixture_role") != "none_used"
        or future_g3_af_transition_obstruction.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_action_bound_followup.get("content_sha256")
        or future_g3_af_transition_obstruction.get("observational_data_opened") is not False
        or future_g3_af_transition_obstruction.get("dark_matter_or_halo_inputs") is not False
        or future_g3_af_transition_obstruction.get("redshift_distance_inputs") is not False
        or future_g3_af_transition_obstruction.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_af_transition_obstruction.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 AF transition obstruction is inconsistent")
    if (
        future_g3_nonunitary_af_constraint_gate.get("candidate_count") != 3
        or future_g3_nonunitary_af_constraint_gate.get("decision_counts") != {"blocked": 3}
        or future_g3_nonunitary_af_constraint_gate.get(
            "nonunitary_formulation_registration_pass_count"
        )
        != 3
        or future_g3_nonunitary_af_constraint_gate.get("nonunitary_AF_principal_pass_count") != 3
        or future_g3_nonunitary_af_constraint_gate.get(
            "flat_nontrivial_reference_constraint_ansatz_reject_count"
        )
        != 3
        or future_g3_nonunitary_af_constraint_gate.get(
            "actual_AF_vacuum_constraint_reference_pass_count"
        )
        != 3
        or future_g3_nonunitary_af_constraint_gate.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_nonunitary_af_constraint_gate.get("global_hamiltonian_energy_pass_count") != 0
        or future_g3_nonunitary_af_constraint_gate.get("full_formal_pass_count") != 0
        or future_g3_nonunitary_af_constraint_gate.get("first_blocker_counts")
        != {
            "candidate_specific_nontrivial_AF_Einstein_constraint_solution_on_decaying_gradient_domain_in_nonunitary_formulation": 3
        }
        or future_g3_nonunitary_af_constraint_gate.get("synthetic_fixture_role") != "none_used"
        or future_g3_nonunitary_af_constraint_gate.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_af_transition_obstruction.get("content_sha256")
        or future_g3_nonunitary_af_constraint_gate.get("observational_data_opened") is not False
        or future_g3_nonunitary_af_constraint_gate.get("dark_matter_or_halo_inputs") is not False
        or future_g3_nonunitary_af_constraint_gate.get("redshift_distance_inputs") is not False
        or future_g3_nonunitary_af_constraint_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_nonunitary_af_constraint_gate.get("candidate_records", [])) != 3
        or any(
            record.get("theory_rejected") is not False
            for record in future_g3_nonunitary_af_constraint_gate.get("candidate_records", [])
        )
    ):
        raise ValueError("future G3 nonunitary AF constraint gate is inconsistent")
    if (
        future_g3_radial_conformal_constraint_reduction.get("candidate_count") != 3
        or future_g3_radial_conformal_constraint_reduction.get("decision_counts") != {"blocked": 3}
        or future_g3_radial_conformal_constraint_reduction.get(
            "radial_pure_trace_momentum_constraint_reduction_pass_count"
        )
        != 3
        or future_g3_radial_conformal_constraint_reduction.get(
            "positive_Hamiltonian_source_registration_pass_count"
        )
        != 3
        or future_g3_radial_conformal_constraint_reduction.get(
            "radial_Lichnerowicz_BVP_registration_pass_count"
        )
        != 3
        or future_g3_radial_conformal_constraint_reduction.get(
            "flat_pure_trace_completion_ansatz_reject_count"
        )
        != 3
        or future_g3_radial_conformal_constraint_reduction.get(
            "positive_global_radial_Lichnerowicz_solution_pass_count"
        )
        != 0
        or future_g3_radial_conformal_constraint_reduction.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_radial_conformal_constraint_reduction.get(
            "global_hamiltonian_energy_pass_count"
        )
        != 0
        or future_g3_radial_conformal_constraint_reduction.get("full_formal_pass_count") != 0
        or future_g3_radial_conformal_constraint_reduction.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_nonunitary_af_constraint_gate.get("content_sha256")
        or future_g3_radial_conformal_constraint_reduction.get("synthetic_fixture_role")
        != "none_used"
        or future_g3_radial_conformal_constraint_reduction.get("observational_data_opened")
        is not False
        or len(future_g3_radial_conformal_constraint_reduction.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 radial conformal constraint reduction is inconsistent")
    if (
        future_g3_radial_lichnerowicz_bvp_no_go.get("candidate_count") != 3
        or future_g3_radial_lichnerowicz_bvp_no_go.get("decision_counts") != {"blocked": 3}
        or future_g3_radial_lichnerowicz_bvp_no_go.get("exact_comparison_inequality_pass_count")
        != 3
        or future_g3_radial_lichnerowicz_bvp_no_go.get(
            "positive_global_radial_Lichnerowicz_solution_nonexistence_count"
        )
        != 3
        or future_g3_radial_lichnerowicz_bvp_no_go.get(
            "radial_conformal_pure_trace_ansatz_reject_count"
        )
        != 3
        or future_g3_radial_lichnerowicz_bvp_no_go.get(
            "positive_global_radial_Lichnerowicz_solution_pass_count"
        )
        != 0
        or future_g3_radial_lichnerowicz_bvp_no_go.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_radial_lichnerowicz_bvp_no_go.get("global_hamiltonian_energy_pass_count") != 0
        or future_g3_radial_lichnerowicz_bvp_no_go.get("full_formal_pass_count") != 0
        or future_g3_radial_lichnerowicz_bvp_no_go.get("first_blocker_counts")
        != {
            "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_radial_conformal_pure_trace_ansatz": 3
        }
        or future_g3_radial_lichnerowicz_bvp_no_go.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_radial_conformal_constraint_reduction.get("content_sha256")
        or future_g3_radial_lichnerowicz_bvp_no_go.get("synthetic_fixture_role") != "none_used"
        or future_g3_radial_lichnerowicz_bvp_no_go.get("observational_data_opened") is not False
        or future_g3_radial_lichnerowicz_bvp_no_go.get("dark_matter_or_halo_inputs") is not False
        or future_g3_radial_lichnerowicz_bvp_no_go.get("redshift_distance_inputs") is not False
        or future_g3_radial_lichnerowicz_bvp_no_go.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_radial_lichnerowicz_bvp_no_go.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 radial Lichnerowicz BVP no-go is inconsistent")
    if (
        future_g3_nonradial_york_bounded_mean_curvature_no_go.get("candidate_count") != 3
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("decision_counts")
        != {"blocked": 3}
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get(
            "nonradial_York_Hamiltonian_reduction_pass_count"
        )
        != 3
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get(
            "bounded_mean_curvature_green_comparison_pass_count"
        )
        != 3
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get(
            "conformally_flat_bounded_mean_curvature_York_class_reject_count"
        )
        != 3
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get(
            "momentum_constraint_solution_pass_count"
        )
        != 0
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get(
            "global_hamiltonian_energy_pass_count"
        )
        != 0
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("full_formal_pass_count") != 0
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("theory_reject_count") != 0
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("first_blocker_counts")
        != {
            "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_conformally_flat_bounded_mean_curvature_York_class": 3
        }
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_radial_lichnerowicz_bvp_no_go.get("content_sha256")
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("synthetic_fixture_role")
        != "none_used"
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("observational_data_opened")
        is not False
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("dark_matter_or_halo_inputs")
        is not False
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("redshift_distance_inputs")
        is not False
        or future_g3_nonradial_york_bounded_mean_curvature_no_go.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_nonradial_york_bounded_mean_curvature_no_go.get("candidate_records", []))
        != 3
    ):
        raise ValueError("future G3 nonradial York no-go is inconsistent")
    if (
        future_g3_york_mean_curvature_frontier.get("candidate_count") != 3
        or future_g3_york_mean_curvature_frontier.get("decision_counts") != {"blocked": 3}
        or future_g3_york_mean_curvature_frontier.get(
            "candidate_millicap_frontier_registration_pass_count"
        )
        != 3
        or future_g3_york_mean_curvature_frontier.get(
            "strict_extension_beyond_kappa_6_over_5_pass_count"
        )
        != 3
        or future_g3_york_mean_curvature_frontier.get("expanded_nonradial_York_class_reject_count")
        != 3
        or future_g3_york_mean_curvature_frontier.get("next_grid_cap_inconclusive_count") != 3
        or future_g3_york_mean_curvature_frontier.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_york_mean_curvature_frontier.get("global_hamiltonian_energy_pass_count") != 0
        or future_g3_york_mean_curvature_frontier.get("full_formal_pass_count") != 0
        or future_g3_york_mean_curvature_frontier.get("theory_reject_count") != 0
        or future_g3_york_mean_curvature_frontier.get("first_blocker_counts")
        != {
            "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_registered_millicap_conformally_flat_York_class": 3
        }
        or future_g3_york_mean_curvature_frontier.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_nonradial_york_bounded_mean_curvature_no_go.get("content_sha256")
        or future_g3_york_mean_curvature_frontier.get("synthetic_fixture_role") != "none_used"
        or future_g3_york_mean_curvature_frontier.get("observational_data_opened") is not False
        or future_g3_york_mean_curvature_frontier.get("dark_matter_or_halo_inputs") is not False
        or future_g3_york_mean_curvature_frontier.get("redshift_distance_inputs") is not False
        or future_g3_york_mean_curvature_frontier.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_york_mean_curvature_frontier.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 York mean-curvature frontier is inconsistent")
    if (
        future_g3_york_analytic_threshold.get("candidate_count") != 3
        or future_g3_york_analytic_threshold.get("decision_counts") != {"blocked": 3}
        or future_g3_york_analytic_threshold.get("exact_algebraic_threshold_pass_count") != 3
        or future_g3_york_analytic_threshold.get("closed_threshold_endpoint_reject_count") != 3
        or future_g3_york_analytic_threshold.get(
            "above_threshold_negative_control_inconclusive_count"
        )
        != 3
        or future_g3_york_analytic_threshold.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_york_analytic_threshold.get("global_hamiltonian_energy_pass_count") != 0
        or future_g3_york_analytic_threshold.get("full_formal_pass_count") != 0
        or future_g3_york_analytic_threshold.get("theory_reject_count") != 0
        or future_g3_york_analytic_threshold.get("first_blocker_counts")
        != {
            "candidate_specific_nontrivial_AF_Einstein_constraint_solution_beyond_analytic_conformally_flat_York_threshold": 3
        }
        or future_g3_york_analytic_threshold.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_york_mean_curvature_frontier.get("content_sha256")
        or future_g3_york_analytic_threshold.get("synthetic_fixture_role") != "none_used"
        or future_g3_york_analytic_threshold.get("observational_data_opened") is not False
        or future_g3_york_analytic_threshold.get("dark_matter_or_halo_inputs") is not False
        or future_g3_york_analytic_threshold.get("redshift_distance_inputs") is not False
        or future_g3_york_analytic_threshold.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_york_analytic_threshold.get("candidate_records", [])) != 3
        or any(
            record.get("York_analytic_threshold_certificate", {})
            .get("exact_algebraic_threshold", {})
            .get("unique_positive_root")
            is not True
            or record.get("York_analytic_threshold_certificate", {})
            .get("endpoint_certificate", {})
            .get("threshold_endpoint_excluded")
            is not True
            or record.get("York_analytic_threshold_certificate", {})
            .get("above_threshold_negative_control", {})
            .get("status")
            != "comparison_inconclusive"
            or record.get("theory_rejected") is not False
            for record in future_g3_york_analytic_threshold.get("candidate_records", [])
        )
    ):
        raise ValueError("future G3 York analytic threshold is inconsistent")
    if (
        future_aether_fixed_free_data_principal_gate.get("candidate_count") != 14
        or future_aether_fixed_free_data_principal_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_fixed_free_data_principal_gate.get(
            "positive_unit_branch_constraint_variable_classification_count"
        )
        != 3
        or future_aether_fixed_free_data_principal_gate.get(
            "zero_dimensional_Aether_constraint_diagonal_block_count"
        )
        != 3
        or future_aether_fixed_free_data_principal_gate.get(
            "zero_Aether_second_order_off_diagonal_columns_count"
        )
        != 3
        or future_aether_fixed_free_data_principal_gate.get(
            "augmented_Aether_unknown_nonelliptic_negative_control_count"
        )
        != 3
        or future_aether_fixed_free_data_principal_gate.get("formal_pass_count") != 0
        or future_aether_fixed_free_data_principal_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_fixed_free_data_principal_gate.get("observational_data_opened")
        is not False
    ):
        raise ValueError("future Aether fixed-free-data principal gate is inconsistent")
    if (
        future_aether_finite_tilt_york_symbol_gate.get("candidate_count") != 14
        or future_aether_finite_tilt_york_symbol_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_finite_tilt_york_symbol_gate.get(
            "finite_tilt_metric_York_symbol_derived_count"
        )
        != 3
        or future_aether_finite_tilt_york_symbol_gate.get(
            "uniform_fixed_free_data_principal_ellipticity_pass_count"
        )
        != 1
        or future_aether_finite_tilt_york_symbol_gate.get(
            "exact_nonelliptic_York_shell_count"
        )
        != 2
        or future_aether_finite_tilt_york_symbol_gate.get("York_ansatz_reject_count") != 2
        or future_aether_finite_tilt_york_symbol_gate.get(
            "weighted_Fredholm_isomorphism_pass_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "lower_order_coefficient_bound_pass_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "computable_full_inverse_norm_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "nonlinear_remainder_bound_pass_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get("formal_pass_count") != 0
        or future_aether_finite_tilt_york_symbol_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "candidate_bound_weighted_Fredholm_isomorphism_lower_order_coefficient_and_inverse_norm_bounds_for_finite_tilt_York_operator": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_finite_tilt_york_symbol_gate.get(
            "source_fixed_free_data_binding", {}
        ).get("content_sha256")
        != future_aether_fixed_free_data_principal_gate.get("content_sha256")
        or future_aether_finite_tilt_york_symbol_gate.get("observational_data_opened")
        is not False
        or future_aether_finite_tilt_york_symbol_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_finite_tilt_york_symbol_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_finite_tilt_york_symbol_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_finite_tilt_york_symbol_gate.get("candidate_records", []))
        != 14
    ):
        raise ValueError("future Aether finite-tilt York symbol gate is inconsistent")
    if (
        future_aether_principal_inverse_fredholm_gate.get("candidate_count") != 14
        or future_aether_principal_inverse_fredholm_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_principal_inverse_fredholm_gate.get(
            "uniformly_elliptic_candidate_count"
        )
        != 1
        or future_aether_principal_inverse_fredholm_gate.get(
            "uniform_principal_symbol_inverse_bound_pass_count"
        )
        != 1
        or future_aether_principal_inverse_fredholm_gate.get(
            "principal_elliptic_homotopy_to_reference_pass_count"
        )
        != 1
        or future_aether_principal_inverse_fredholm_gate.get(
            "distributed_lower_order_coefficient_registry_complete_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get(
            "weighted_Fredholm_isomorphism_pass_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get(
            "full_operator_inverse_norm_pass_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get(
            "nonlinear_remainder_bound_pass_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get("formal_pass_count") != 0
        or future_aether_principal_inverse_fredholm_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "candidate_bound_spatially_distributed_lower_order_linearized_constraint_coefficient_registry_on_weighted_spaces": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_principal_inverse_fredholm_gate.get(
            "source_York_symbol_binding", {}
        ).get("content_sha256")
        != future_aether_finite_tilt_york_symbol_gate.get("content_sha256")
        or future_aether_principal_inverse_fredholm_gate.get("observational_data_opened")
        is not False
        or future_aether_principal_inverse_fredholm_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_principal_inverse_fredholm_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_principal_inverse_fredholm_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_principal_inverse_fredholm_gate.get("candidate_records", []))
        != 14
    ):
        raise ValueError("future Aether principal-inverse Fredholm gate is inconsistent")
    if (
        future_g3_york_tracefree_compensation.get("candidate_count") != 3
        or future_g3_york_tracefree_compensation.get("decision_counts") != {"blocked": 3}
        or future_g3_york_tracefree_compensation.get(
            "exact_tracefree_compensation_bound_pass_count"
        )
        != 3
        or future_g3_york_tracefree_compensation.get(
            "tracefree_compensated_York_class_reject_count"
        )
        != 3
        or future_g3_york_tracefree_compensation.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_york_tracefree_compensation.get("theory_reject_count") != 0
        or future_g3_york_tracefree_compensation.get("observational_data_opened") is not False
    ):
        raise ValueError("future G3 tracefree-compensated York gate is inconsistent")
    if (
        future_g3_general_geometry_curvature_shortfall.get("candidate_count") != 3
        or future_g3_general_geometry_curvature_shortfall.get("decision_counts")
        != {"blocked": 3}
        or future_g3_general_geometry_curvature_shortfall.get(
            "general_geometry_pointwise_theorem_pass_count"
        )
        != 3
        or future_g3_general_geometry_curvature_shortfall.get(
            "curvature_shortfall_constraint_class_reject_count"
        )
        != 3
        or future_g3_general_geometry_curvature_shortfall.get(
            "exact_curvature_endpoint_inconclusive_count"
        )
        != 3
        or future_g3_general_geometry_curvature_shortfall.get(
            "above_threshold_not_excluded_control_count"
        )
        != 3
        or future_g3_general_geometry_curvature_shortfall.get(
            "nonconformally_flat_metric_construction_pass_count"
        )
        != 0
        or future_g3_general_geometry_curvature_shortfall.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_general_geometry_curvature_shortfall.get(
            "first_blocker_counts"
        )
        != {
            "candidate_specific_AF_Einstein_constraint_datum_outside_general_geometry_curvature_shortfall_class": 3
        }
        or future_g3_general_geometry_curvature_shortfall.get("theory_reject_count") != 0
        or future_g3_general_geometry_curvature_shortfall.get("observational_data_opened")
        is not False
        or future_g3_general_geometry_curvature_shortfall.get("dark_matter_or_halo_inputs")
        is not False
        or future_g3_general_geometry_curvature_shortfall.get("redshift_distance_inputs")
        is not False
        or future_g3_general_geometry_curvature_shortfall.get("paid_llm_spend_usd") != 0.0
        or future_g3_general_geometry_curvature_shortfall.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_york_tracefree_compensation.get("content_sha256")
        or len(future_g3_general_geometry_curvature_shortfall.get("candidate_records", []))
        != 3
    ):
        raise ValueError("future G3 general-geometry curvature shortfall gate is inconsistent")
    if (
        future_g3_general_geometry_surplus_mismatch.get("candidate_count") != 3
        or future_g3_general_geometry_surplus_mismatch.get("decision_counts")
        != {"blocked": 3}
        or future_g3_general_geometry_surplus_mismatch.get(
            "exact_surplus_identity_pass_count"
        )
        != 3
        or future_g3_general_geometry_surplus_mismatch.get(
            "above_threshold_surplus_mismatch_class_reject_count"
        )
        != 3
        or future_g3_general_geometry_surplus_mismatch.get(
            "matched_surplus_necessary_control_count"
        )
        != 3
        or future_g3_general_geometry_surplus_mismatch.get(
            "overcurvature_not_excluded_control_count"
        )
        != 3
        or future_g3_general_geometry_surplus_mismatch.get(
            "registered_AF_metric_York_datum_pass_count"
        )
        != 0
        or future_g3_general_geometry_surplus_mismatch.get(
            "momentum_constraint_solution_pass_count"
        )
        != 0
        or future_g3_general_geometry_surplus_mismatch.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_general_geometry_surplus_mismatch.get("theory_reject_count") != 0
        or future_g3_general_geometry_surplus_mismatch.get("first_blocker_counts")
        != {
            "candidate_specific_AF_metric_and_York_data_with_pointwise_curvature_surplus_matching_and_momentum_solution": 3
        }
        or future_g3_general_geometry_surplus_mismatch.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_general_geometry_curvature_shortfall.get("content_sha256")
        or future_g3_general_geometry_surplus_mismatch.get("observational_data_opened")
        is not False
        or future_g3_general_geometry_surplus_mismatch.get("dark_matter_or_halo_inputs")
        is not False
        or future_g3_general_geometry_surplus_mismatch.get("redshift_distance_inputs")
        is not False
        or future_g3_general_geometry_surplus_mismatch.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_general_geometry_surplus_mismatch.get("candidate_records", []))
        != 3
    ):
        raise ValueError("future G3 general-geometry surplus mismatch gate is inconsistent")
    if (
        future_candidate_action_dossier.get("candidate_count") != 19
        or future_candidate_action_dossier.get("family_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": 16,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
        }
        or future_candidate_action_dossier.get("decision_counts") != {"blocked": 17, "reject": 2}
        or future_candidate_action_dossier.get("ranked_candidate_count") != 0
        or future_candidate_action_dossier.get("observational_authorization") is not False
        or future_candidate_action_dossier.get("observational_data_opened") is not False
        or future_candidate_action_dossier.get("paid_llm_spend_usd") != 0.0
        or future_candidate_action_dossier.get("source_roots", {}).get("preflight_content_sha256")
        != scalable_future_formal_preflight.get("content_sha256")
        or future_candidate_action_dossier.get("source_roots", {}).get(
            "aether_followup_content_sha256"
        )
        != future_aether_principal_inverse_fredholm_gate.get("content_sha256")
        or future_candidate_action_dossier.get("source_roots", {}).get("g3_followup_content_sha256")
        != future_g3_general_geometry_surplus_mismatch.get("content_sha256")
        or len(future_candidate_action_dossier.get("dossiers", [])) != 19
        or any(
            record.get("comparison_contract", {}).get("rank") is not None
            or record.get("comparison_contract", {}).get("rank_eligible") is not False
            or not record.get("action", {}).get("ordered_operator_densities")
            or record.get("action", {}).get("human_readable_action", {}).get("display_kind")
            != "verbatim_ordered_covariant_density_concatenation"
            for record in future_candidate_action_dossier.get("dossiers", [])
        )
    ):
        raise ValueError("future candidate action dossier is inconsistent")
    if (
        aether_candidate_formal.get("candidate_count") != 128
        or aether_candidate_formal.get("input_preflight_pass_count") != 128
        or aether_candidate_formal.get("decision_counts") != {"blocked": 126, "reject": 2}
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
        or aether_candidate_formal.get("provenance", {}).get("formal_preflight_status_sha256")
        != formal_preflight.get("content_sha256")
        or aether_candidate_formal.get("provenance", {}).get("compilation_campaign_sha256")
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
        or g3_candidate_formal.get("gate_counts", {}).get("uniform_local_principal_symbol")
        != {"pass": 32}
        or g3_candidate_formal.get("gate_counts", {}).get("distributed_Dirac_on_periodic_cell")
        != {"pass": 32}
        or g3_candidate_formal.get("gate_counts", {}).get("af_uniform_lapse_Dirac_invertibility")
        != {"blocked": 32}
        or g3_candidate_formal.get("promotion_status_binding", {}).get("content_sha256")
        != promotion_admission.get("content_sha256")
        or g3_candidate_formal.get("af_global_constraint_solution_proved") is not False
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
    g4_equivalence = g4_scalable_formal_followup.get("equivalence_certificate", {})
    if (
        g4_scalable_formal_followup.get("candidate_count") != 1
        or g4_scalable_formal_followup.get("candidate_id") != "G3A-e0eff4150989e3522dc6ba03"
        or g4_scalable_formal_followup.get("preflight_decision") != "blocked"
        or g4_scalable_formal_followup.get("preflight_blocker") != "family_prerequisite_not_passed"
        or g4_scalable_formal_followup.get("formal_followup_decision") != "pass"
        or g4_scalable_formal_followup.get("decision_counts") != {"pass": 1}
        or g4_scalable_formal_followup.get("formal_pass_count") != 1
        or g4_scalable_formal_followup.get("necessary_condition_rejection_count") != 0
        or g4_scalable_formal_followup.get("equivalent_parameter_cell_alias_count") != 32
        or g4_equivalence.get("action_density_projection_equal") is not True
        or g4_equivalence.get("operator_densities_equal") is not True
        or g4_equivalence.get("universal_matter_coupling_equal") is not True
        or g4_equivalence.get("representative_domain_is_subset") is not True
        or g4_equivalence.get("all_alias_domains_inside_reviewed_domain") is not True
        or g4_equivalence.get("family_label_used_as_equivalence_evidence") is not False
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
        or llm_adapter.get("output_status") != "quarantine_until_downstream_validation"
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
        or g4_galaxy["synthetic_controls"]["shape"].get("object_specific_gravity_parameter_count")
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
        local_formula_service.get("status") != "ready_disabled_bounded_local_only"
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
        or set(forward_controls.get("analytic_known_answers", {}).values()) != {"pass"}
        or forward_controls.get("covariance", {}).get("decision") != "pass"
    ):
        raise ValueError("G4 galaxy forward-model readiness is not fail-closed")
    branch_distance_decision = g4_galaxy_branch_distance.get("current_evaluator_decision", {})
    if (
        g4_galaxy_branch_distance.get("decision") != "blocked"
        or g4_galaxy_branch_distance.get("prediction_bundle_registered") is not False
        or g4_galaxy_branch_distance.get("candidate_use_authorized") is not False
        or g4_galaxy_branch_distance.get("observational_data_opened") is not False
        or g4_galaxy_branch_distance.get("primary_record_access_count") != 0
        or g4_galaxy_branch_distance.get("real_source_geometry_registered") is not False
        or g4_galaxy_branch_distance.get("source_specific_branch_selection_proven") is not False
        or g4_galaxy_branch_distance.get("object_specific_gravity_parameter_count") != 0
        or g4_galaxy_branch_distance.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_branch_distance.get("redshift_distance_inputs") is not False
        or branch_distance_decision.get("filled_registration_hash_count") != 5
        or len(branch_distance_decision.get("missing_registration_hashes", [])) != 13
        or set(g4_galaxy_branch_distance.get("newly_filled_registration_fields", {}))
        != {
            "branch_and_domain_contract_sha256",
            "distance_mode_contract_sha256",
        }
        or set(g4_galaxy_branch_distance.get("preserved_predecessor_registration_fields", {}))
        != {
            "reviewed_candidate_galaxy_evaluator_descriptor_sha256",
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
        or g4_galaxy_branch_distance.get("branch_contract_status")
        != "certified_exact_conditional_branch"
        or g4_galaxy_branch_distance.get("distance_geometry_contract_status")
        != "certified_interface_no_real_values"
        or g4_galaxy_branch_distance.get("provenance", {}).get("forward_model_predecessor_sha256")
        != g4_galaxy_forward_model.get("content_sha256")
    ):
        raise ValueError("G4 galaxy branch/distance registration is not fail-closed")
    calibration_decision = g4_galaxy_calibration_evaluation.get("current_evaluator_decision", {})
    if (
        g4_galaxy_calibration_evaluation.get("decision") != "blocked"
        or g4_galaxy_calibration_evaluation.get("prediction_bundle_registered") is not False
        or g4_galaxy_calibration_evaluation.get("candidate_use_authorized") is not False
        or g4_galaxy_calibration_evaluation.get("observational_data_opened") is not False
        or g4_galaxy_calibration_evaluation.get("primary_record_access_count") != 0
        or g4_galaxy_calibration_evaluation.get("object_specific_gravity_parameter_count") != 0
        or g4_galaxy_calibration_evaluation.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_calibration_evaluation.get("redshift_distance_inputs") is not False
        or g4_galaxy_calibration_evaluation.get("paid_llm_spend_usd") != 0.0
        or calibration_decision.get("filled_registration_hash_count") != 9
        or len(calibration_decision.get("missing_registration_hashes", [])) != 9
        or set(g4_galaxy_calibration_evaluation.get("newly_filled_registration_fields", {}))
        != {
            "baryonic_calibration_hierarchy_sha256",
            "joint_covariance_contract_sha256",
            "likelihood_contract_sha256",
            "stopping_rule_sha256",
        }
        or set(
            g4_galaxy_calibration_evaluation.get("preserved_predecessor_registration_fields", {})
        )
        != {
            "branch_and_domain_contract_sha256",
            "distance_mode_contract_sha256",
            "reviewed_candidate_galaxy_evaluator_descriptor_sha256",
            "rotation_prediction_implementation_sha256",
            "lensing_prediction_implementation_sha256",
        }
        or set(g4_galaxy_calibration_evaluation.get("non_registration_policy_hashes", {}))
        != {"held_out_split_policy_sha256"}
        or set(
            g4_galaxy_calibration_evaluation.get("deliberately_unfilled_registration_fields", {})
        )
        != {
            "galaxy_split_commitment_sha256",
            "training_only_checkpoint_sha256",
        }
        or g4_galaxy_calibration_evaluation.get("provenance", {}).get("predecessor_content_sha256")
        != g4_galaxy_branch_distance.get("content_sha256")
    ):
        raise ValueError("G4 galaxy calibration/evaluation registration is not fail-closed")
    transform_decision = g4_galaxy_prediction_contract_transform.get(
        "current_evaluator_decision", {}
    )
    if (
        g4_galaxy_prediction_contract_transform.get("decision") != "blocked"
        or g4_galaxy_prediction_contract_transform.get("prediction_bundle_registered") is not False
        or g4_galaxy_prediction_contract_transform.get("candidate_use_authorized") is not False
        or g4_galaxy_prediction_contract_transform.get("observational_data_opened") is not False
        or g4_galaxy_prediction_contract_transform.get("primary_record_access_count") != 0
        or g4_galaxy_prediction_contract_transform.get("object_specific_gravity_parameter_count")
        != 0
        or g4_galaxy_prediction_contract_transform.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_prediction_contract_transform.get("redshift_distance_inputs") is not False
        or g4_galaxy_prediction_contract_transform.get("paid_llm_spend_usd") != 0.0
        or g4_galaxy_prediction_contract_transform.get("real_transform_inputs_registered")
        is not False
        or transform_decision.get("filled_registration_hash_count") != 11
        or len(transform_decision.get("missing_registration_hashes", [])) != 7
        or set(g4_galaxy_prediction_contract_transform.get("newly_filled_registration_fields", {}))
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
    tooling_decision = g4_galaxy_manifest_bundle_tooling.get("unchanged_evaluator_decision", {})
    tooling_controls = g4_galaxy_manifest_bundle_tooling.get("synthetic_controls", {})
    if (
        g4_galaxy_manifest_bundle_tooling.get("decision") != "blocked"
        or g4_galaxy_manifest_bundle_tooling.get("candidate_use_authorized") is not False
        or g4_galaxy_manifest_bundle_tooling.get("observational_data_opened") is not False
        or g4_galaxy_manifest_bundle_tooling.get("primary_record_access_count") != 0
        or g4_galaxy_manifest_bundle_tooling.get("prediction_bundle_registered") is not False
        or g4_galaxy_manifest_bundle_tooling.get("dataset_manifest_registered") is not False
        or g4_galaxy_manifest_bundle_tooling.get("independent_registry_receipt_registered")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_manifest_bundle_tooling.get("redshift_distance_inputs") is not False
        or g4_galaxy_manifest_bundle_tooling.get("paid_llm_spend_usd") != 0.0
        or g4_galaxy_manifest_bundle_tooling.get("filled_registration_hash_count") != 11
        or g4_galaxy_manifest_bundle_tooling.get("missing_registration_hash_count") != 7
        or g4_galaxy_manifest_bundle_tooling.get("newly_filled_registration_fields") != {}
        or tooling_decision.get("filled_registration_hash_count") != 11
        or len(tooling_decision.get("missing_registration_hashes", [])) != 7
        or tooling_controls.get("manifest_audit_registration_admissible") is not False
        or tooling_controls.get("bundle_draft_registration_admissible") is not False
        or tooling_controls.get("synthetic_values_promoted") is not False
        or g4_galaxy_manifest_bundle_tooling.get("tooling_readiness", {}).get("enabled")
        is not False
        or g4_galaxy_manifest_bundle_tooling.get("provenance", {}).get("predecessor_content_sha256")
        != g4_galaxy_prediction_contract_transform.get("content_sha256")
    ):
        raise ValueError("G4 galaxy manifest/bundle tooling is not fail-closed")
    source_registry_decision = g4_galaxy_source_registry_admission.get(
        "unchanged_evaluator_decision", {}
    )
    source_registry_readiness = g4_galaxy_source_registry_admission.get("admission_readiness", {})
    if (
        g4_galaxy_source_registry_admission.get("decision") != "blocked"
        or g4_galaxy_source_registry_admission.get("service_enabled") is not False
        or g4_galaxy_source_registry_admission.get("start_requested") is not False
        or g4_galaxy_source_registry_admission.get("source_records_admitted") != 0
        or g4_galaxy_source_registry_admission.get("target_records_opened") != 0
        or g4_galaxy_source_registry_admission.get("primary_record_access_count") != 0
        or g4_galaxy_source_registry_admission.get("observation_opening_authorization_registered")
        is not False
        or g4_galaxy_source_registry_admission.get("prediction_bundle_registered") is not False
        or g4_galaxy_source_registry_admission.get("observational_data_opened") is not False
        or g4_galaxy_source_registry_admission.get("dark_matter_or_halo_inputs") is not False
        or g4_galaxy_source_registry_admission.get("redshift_distance_inputs") is not False
        or g4_galaxy_source_registry_admission.get("object_specific_gravity_parameter_count") != 0
        or g4_galaxy_source_registry_admission.get("paid_llm_spend_usd") != 0.0
        or g4_galaxy_source_registry_admission.get("filled_registration_hash_count") != 11
        or g4_galaxy_source_registry_admission.get("missing_registration_hash_count") != 7
        or g4_galaxy_source_registry_admission.get("newly_filled_registration_fields") != {}
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
    quartic_third_slice = quartic_tc2_diagonal_third_jet.get("slice_contract", {})
    quartic_third_blocker = quartic_tc2_diagonal_third_jet.get("first_remaining_blocker", {})
    if (
        quartic_tc2_diagonal_third_jet.get("status")
        != "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_mixed_triples_full_tube_global_H7_fail_closed"
        or quartic_third_slice.get("active_coordinate_directions") != 41
        or quartic_third_slice.get("diagonal_triples") != 41
        or quartic_third_slice.get("full_symmetric_triples_in_41_direction_sector") != 12_341
        or quartic_third_slice.get("mixed_AAB_ABB_ABC_triples") != 0
        or quartic_third_counts.get("candidates") != 12
        or quartic_third_counts.get("symbolic_parameter_diagonal_third_jet_passes") != 41
        or quartic_third_counts.get("candidate_direction_evaluations") != 492
        or quartic_third_counts.get("candidate_direction_solvable") != 492
        or quartic_third_counts.get("candidate_direction_obstructed") != 0
        or quartic_third_counts.get("candidates_all_41_diagonal_third_jets_closed") != 12
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
        or "12,300 polarized mixed triples" not in quartic_third_blocker.get("required", "")
    ):
        raise ValueError("quartic TC2 diagonal third-jet slice is inconsistent")
    quartic_mixed_counts = quartic_tc2_mixed_third_jet_chunk.get("counts", {})
    quartic_mixed_contract = quartic_tc2_mixed_third_jet_chunk.get("chunk_contract", {})
    quartic_mixed_ledger = quartic_tc2_mixed_third_jet_chunk.get("closure_ledger", {})
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
        or quartic_mixed_counts.get("triple_kind_counts") != {"AAB": 40, "ABB": 1, "ABC": 23}
        or quartic_mixed_counts.get("mixed_triples_remaining") != 12_236
        or quartic_tc2_mixed_third_jet_chunk.get("first_exact_obstruction") is not None
        or quartic_tc2_mixed_third_jet_chunk.get("upstream_sha256", {}).get("diagonal_third_jet")
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
    quartic_mixed_64_counts = quartic_tc2_mixed_third_jet_chunk_64.get("counts", {})
    quartic_mixed_64_contract = quartic_tc2_mixed_third_jet_chunk_64.get("chunk_contract", {})
    quartic_mixed_64_ledger = quartic_tc2_mixed_third_jet_chunk_64.get("closure_ledger", {})
    quartic_mixed_128_counts = quartic_tc2_mixed_third_jet_chunk_128.get("counts", {})
    quartic_mixed_128_contract = quartic_tc2_mixed_third_jet_chunk_128.get("chunk_contract", {})
    quartic_mixed_128_ledger = quartic_tc2_mixed_third_jet_chunk_128.get("closure_ledger", {})
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
        or quartic_mixed_64_counts.get("triple_kind_counts") != {"ABB": 2, "ABC": 62}
        or quartic_mixed_64_counts.get("mixed_triples_remaining") != 12_172
        or quartic_tc2_mixed_third_jet_chunk_64.get("first_exact_obstruction") is not None
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
        or quartic_tc2_mixed_third_jet_chunk_128.get("status")
        != "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
        or quartic_mixed_128_contract.get("chunk_offset") != 128
        or quartic_mixed_128_contract.get("processed_count") != 64
        or quartic_mixed_128_contract.get("next_offset") != 192
        or quartic_mixed_128_contract.get("global_mixed_triple_count") != 12_300
        or quartic_mixed_128_contract.get("prior_resume_sha256")
        != quartic_mixed_64_contract.get("resume_tip_sha256")
        or quartic_mixed_128_contract.get("stopped_early") is not False
        or quartic_mixed_128_counts.get("selected") != 64
        or quartic_mixed_128_counts.get("symbolic_parameter_compatible") != 64
        or quartic_mixed_128_counts.get("candidate_evaluations") != 768
        or quartic_mixed_128_counts.get("candidate_solvable") != 768
        or quartic_mixed_128_counts.get("candidate_obstructed") != 0
        or quartic_mixed_128_counts.get("triple_kind_counts") != {"ABB": 1, "ABC": 63}
        or quartic_mixed_128_counts.get("mixed_triples_remaining") != 12_108
        or quartic_tc2_mixed_third_jet_chunk_128.get("first_exact_obstruction") is not None
        or quartic_mixed_128_ledger.get("processed_mixed_third_jets_closed") != 64
        or any(
            quartic_mixed_128_ledger.get(key) is not False
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
        or quartic_tc2_mixed_third_jet_checkpoint.get("completed_chunks") != 2
        or quartic_tc2_mixed_third_jet_checkpoint.get("next_offset") != 192
        or quartic_tc2_mixed_third_jet_checkpoint.get("remaining_mixed_triples") != 12_108
        or quartic_tc2_mixed_third_jet_checkpoint.get("current_artifact_content_sha256")
        != quartic_tc2_mixed_third_jet_chunk_128.get("content_sha256")
        or quartic_tc2_mixed_third_jet_checkpoint.get("prior_resume_sha256")
        != quartic_mixed_128_contract.get("resume_tip_sha256")
        or quartic_tc2_mixed_third_jet_continuation_status.get("checkpoint_content_sha256")
        != quartic_tc2_mixed_third_jet_checkpoint.get("content_sha256")
        or quartic_tc2_mixed_third_jet_continuation_status.get("next_offset") != 192
        or quartic_tc2_mixed_third_jet_continuation_status.get("remaining_mixed_triples") != 12_108
        or quartic_tc2_mixed_third_jet_continuation_status.get("decision") != "checkpointed"
        or quartic_tc2_mixed_third_jet_continuation_status.get("permanently_stopped") is not False
        or any(
            quartic_tc2_mixed_third_jet_continuation_status.get("claims", {}).get(key) is not False
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

    prior_resume = quartic_mixed_128_contract.get("resume_tip_sha256")
    for offset, artifact in zip(
        range(192, 1_600, 64),
        quartic_tc2_mixed_third_jet_parallel_chunks,
        strict=True,
    ):
        counts = artifact.get("counts", {})
        contract = artifact.get("chunk_contract", {})
        ledger = artifact.get("closure_ledger", {})
        if (
            artifact.get("status") != "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
            or contract.get("chunk_offset") != offset
            or contract.get("processed_count") != 64
            or contract.get("next_offset") != offset + 64
            or contract.get("global_mixed_triple_count") != 12_300
            or contract.get("prior_resume_sha256") != prior_resume
            or contract.get("parallel_worker_count") != 8
            or contract.get("parallel_execution_policy")
            != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
            or contract.get("bounded_speculative_evaluations_may_finish_after_first_obstruction")
            is not True
            or contract.get("records_after_first_obstruction_committed_or_inferred") != 0
            or contract.get("stopped_early") is not False
            or counts.get("selected") != 64
            or counts.get("symbolic_parameter_compatible") != 64
            or counts.get("candidate_evaluations") != 768
            or counts.get("candidate_solvable") != 768
            or counts.get("candidate_obstructed") != 0
            or sum(counts.get("triple_kind_counts", {}).values()) != 64
            or counts.get("mixed_triples_remaining") != 12_300 - (offset + 64)
            or artifact.get("first_exact_obstruction") is not None
            or ledger.get("processed_mixed_third_jets_closed") != 64
            or any(
                ledger.get(key) is not False
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
            raise ValueError(
                f"quartic TC2 parallel mixed third-jet offset {offset} is inconsistent"
            )
        prior_resume = contract.get("resume_tip_sha256")

    quartic_parallel_latest = quartic_tc2_mixed_third_jet_parallel_chunks[-1]
    quartic_parallel_counts = quartic_parallel_latest["counts"]
    quartic_parallel_contract = quartic_parallel_latest["chunk_contract"]
    if (
        quartic_tc2_mixed_third_jet_parallel_checkpoint.get("completed_chunks") != 22
        or quartic_tc2_mixed_third_jet_parallel_checkpoint.get("next_offset") != 1_600
        or quartic_tc2_mixed_third_jet_parallel_checkpoint.get("remaining_mixed_triples") != 10_700
        or quartic_tc2_mixed_third_jet_parallel_checkpoint.get("current_artifact_content_sha256")
        != quartic_parallel_latest.get("content_sha256")
        or quartic_tc2_mixed_third_jet_parallel_checkpoint.get("prior_resume_sha256")
        != prior_resume
        or quartic_tc2_mixed_third_jet_parallel_status.get("checkpoint_content_sha256")
        != quartic_tc2_mixed_third_jet_parallel_checkpoint.get("content_sha256")
        or quartic_tc2_mixed_third_jet_parallel_status.get("next_offset") != 1_600
        or quartic_tc2_mixed_third_jet_parallel_status.get("remaining_mixed_triples") != 10_700
        or quartic_tc2_mixed_third_jet_parallel_status.get("decision") != "checkpointed"
        or quartic_tc2_mixed_third_jet_parallel_status.get("permanently_stopped") is not False
        or any(
            quartic_tc2_mixed_third_jet_parallel_status.get("claims", {}).get(key) is not False
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
        or quartic_tc2_mixed_third_jet_parallel_supervisor_readiness.get("status")
        != "portable_stopped_checkpoint"
        or quartic_tc2_mixed_third_jet_parallel_supervisor_readiness.get("lifecycle", {}).get(
            "next_offset"
        )
        != 576
        or quartic_tc2_mixed_third_jet_parallel_supervisor_readiness.get("lifecycle", {}).get(
            "remaining_mixed_triples"
        )
        != 11_724
        or quartic_tc2_mixed_third_jet_parallel_supervisor_readiness.get(
            "parallel_contract", {}
        ).get("parallel_worker_count")
        != 8
        or any(
            quartic_tc2_mixed_third_jet_parallel_supervisor_readiness.get("claims", {}).get(key)
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
        raise ValueError("quartic TC2 parallel mixed third-jet checkpoint is inconsistent")
    quartic_basis_counts = quartic_tc2_mixed_third_jet_basis_reduction.get("counts", {})
    quartic_basis_reduction = quartic_tc2_mixed_third_jet_basis_reduction.get(
        "exact_active_direction_reduction", {}
    )
    if (
        quartic_tc2_mixed_third_jet_basis_reduction.get("status")
        != "pass_exact_15_direction_basis_reduction_560_obligations_no_inferred_passes_global_closure_fail_closed"
        or quartic_basis_counts.get("stable_mixed_triples_evaluated") != 576
        or quartic_basis_counts.get("reduced_exact_obligations") != 560
        or quartic_basis_counts.get("reduced_obligations_evaluated") != 0
        or quartic_basis_counts.get("remaining_active_triples_inferred_passed") != 0
        or quartic_basis_reduction.get("active_direction_rank") != 15
        or quartic_basis_reduction.get("symmetric_cubic_dimension") != 680
        or quartic_basis_reduction.get("combined_evidence_functional_rank") != 120
        or quartic_basis_reduction.get("completion_rank") != 680
        or quartic_basis_reduction.get("drop_final_obligation_rank") != 679
        or any(
            quartic_tc2_mixed_third_jet_basis_reduction.get("closure_ledger", {}).get(key)
            is not False
            for key in (
                "all_12_300_mixed_third_jets_closed",
                "all_560_reduced_obligations_closed",
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
        raise ValueError("quartic TC2 mixed third-jet basis reduction is inconsistent")
    quartic_rerank_counts = quartic_tc2_mixed_third_jet_reranked_reduction.get("counts", {})
    quartic_rerank = quartic_tc2_mixed_third_jet_reranked_reduction.get("exact_reranking", {})
    quartic_rerank_evidence = quartic_tc2_mixed_third_jet_reranked_reduction.get(
        "stable_evidence", {}
    )
    if (
        quartic_tc2_mixed_third_jet_reranked_reduction.get("status")
        != "pass_exact_stopped_chain_1600_rerank_447_obligations_no_inferred_passes_global_closure_fail_closed"
        or quartic_rerank_counts.get("stable_mixed_triples_evaluated") != 1_600
        or quartic_rerank_counts.get("stable_mixed_triples_remaining") != 10_700
        or quartic_rerank_counts.get("reranked_exact_obligations") != 447
        or quartic_rerank_counts.get("reranked_obligations_evaluated") != 0
        or quartic_rerank_counts.get("remaining_active_triples_inferred_passed") != 0
        or quartic_rerank.get("active_direction_rank") != 15
        or quartic_rerank.get("symmetric_cubic_dimension") != 680
        or quartic_rerank.get("diagonal_evidence_rank") != 16
        or quartic_rerank.get("prior_576_prefix_rank") != 105
        or quartic_rerank.get("stable_1600_prefix_rank") != 219
        or quartic_rerank.get("stable_combined_evidence_rank") != 233
        or quartic_rerank.get("rank_gain_over_prior_reduction") != 113
        or quartic_rerank.get("reranked_obligation_kind_counts")
        != {"AAB": 77, "ABB": 81, "ABC": 289}
        or quartic_rerank.get("reranked_obligation_count") != 447
        or quartic_rerank.get("first_selector_index") != 1_634
        or quartic_rerank.get("last_selector_index") != 12_269
        or quartic_rerank.get("completion_rank") != 680
        or quartic_rerank.get("drop_final_obligation_rank") != 679
        or quartic_tc2_mixed_third_jet_reranked_reduction.get(
            "reranked_obligation_selector", {}
        ).get("candidate_evaluations_if_all_obligations_are_run")
        != 5_364
        or quartic_rerank_evidence.get("chunk_count") != 25
        or quartic_rerank_evidence.get("mixed_prefix_records") != 1_600
        or quartic_rerank_evidence.get("mixed_candidate_evaluations") != 19_200
        or quartic_rerank_evidence.get("mixed_candidate_solvable") != 19_200
        or quartic_rerank_evidence.get("mixed_candidate_obstructed") != 0
        or quartic_rerank_evidence.get("checkpoint_content_sha256")
        != quartic_tc2_mixed_third_jet_parallel_checkpoint.get("content_sha256")
        or quartic_rerank_evidence.get("service_status_content_sha256")
        != quartic_tc2_mixed_third_jet_parallel_status.get("content_sha256")
        or quartic_rerank_evidence.get("supervisor_stop_reason") != "epoch_limit"
        or any(
            quartic_tc2_mixed_third_jet_reranked_reduction.get("closure_ledger", {}).get(key)
            is not False
            for key in (
                "all_12_300_mixed_third_jets_closed",
                "all_447_reranked_obligations_closed",
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
        raise ValueError("quartic TC2 mixed third-jet reranked reduction is inconsistent")
    reranked_prior_resume = "1c979dd6037a9841ef6dc7d506ecb2b51d3b42bec18a5655d0c7bc6acbe7fb5b"
    reranked_total_processed = 0
    reranked_candidate_evaluations = 0
    reranked_offsets = (0, 64, 128, 192, 256, 320, 384)
    for offset, artifact in zip(
        reranked_offsets, quartic_tc2_reranked_obligation_chunks, strict=True
    ):
        contract = artifact.get("chunk_contract", {})
        counts = artifact.get("counts", {})
        ledger = artifact.get("closure_ledger", {})
        processed_count = 63 if offset == 384 else 64
        next_offset = offset + processed_count
        final_tail = next_offset == 447
        if (
            artifact.get("status")
            != (
                "pass_reranked_obligation_exact_final_tail_63_fail_closed"
                if final_tail
                else "pass_reranked_obligation_chunk_64_fail_closed"
            )
            or contract.get("global_obligation_count") != 447
            or contract.get("obligation_offset") != offset
            or contract.get("processed_count") != processed_count
            or contract.get("requested_chunk_size") != processed_count
            or contract.get("next_obligation_offset") != next_offset
            or contract.get("exact_final_partial_tail", False) is not final_tail
            or contract.get("prior_resume_sha256") != reranked_prior_resume
            or contract.get("parallel_worker_count") != 8
            or contract.get("parallel_execution_policy")
            != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
            or contract.get("bounded_speculative_evaluations_may_finish_after_first_obstruction")
            is not True
            or contract.get("records_after_first_obstruction_committed_or_inferred") != 0
            or counts.get("selected") != processed_count
            or counts.get("symbolic_parameter_compatible") != processed_count
            or counts.get("candidate_evaluations") != processed_count * 12
            or counts.get("candidate_solvable") != processed_count * 12
            or counts.get("candidate_obstructed") != 0
            or counts.get("reranked_obligations_remaining") != 447 - next_offset
            or artifact.get("first_exact_obstruction") is not None
            or artifact.get("upstream_sha256", {}).get("reranked_reduction")
            != quartic_tc2_mixed_third_jet_reranked_reduction.get("content_sha256")
            or artifact.get("obligation_manifest", [])[0].get("previous_record_sha256")
            != contract.get("resume_seed_sha256")
            or artifact.get("obligation_manifest", [])[-1].get("record_sha256")
            != contract.get("resume_tip_sha256")
            or ledger.get("processed_reranked_obligations_closed") != processed_count
            or ledger.get("all_12_300_mixed_third_jets_closed") is not final_tail
            or ledger.get("all_447_reranked_obligations_closed") is not final_tail
            or any(
                ledger.get(key) is not False
                for key in (
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
            raise ValueError(f"quartic TC2 reranked obligation offset {offset} is inconsistent")
        reranked_prior_resume = contract["resume_tip_sha256"]
        reranked_total_processed += processed_count
        reranked_candidate_evaluations += counts["candidate_evaluations"]
    reranked_latest = quartic_tc2_reranked_obligation_chunks[-1]
    if (
        reranked_total_processed != 447
        or reranked_candidate_evaluations != 5_364
        or quartic_tc2_reranked_obligation_checkpoint.get("completed_chunks") != 7
        or quartic_tc2_reranked_obligation_checkpoint.get("next_obligation_offset") != 447
        or quartic_tc2_reranked_obligation_checkpoint.get("remaining_obligations") != 0
        or quartic_tc2_reranked_obligation_checkpoint.get("current_artifact_content_sha256")
        != reranked_latest.get("content_sha256")
        or quartic_tc2_reranked_obligation_checkpoint.get("prior_resume_sha256")
        != reranked_prior_resume
        or quartic_tc2_reranked_obligation_checkpoint.get("reranked_reduction_content_sha256")
        != quartic_tc2_mixed_third_jet_reranked_reduction.get("content_sha256")
        or quartic_tc2_reranked_obligation_status.get("checkpoint_content_sha256")
        != quartic_tc2_reranked_obligation_checkpoint.get("content_sha256")
        or quartic_tc2_reranked_obligation_status.get("next_obligation_offset") != 447
        or quartic_tc2_reranked_obligation_status.get("remaining_obligations") != 0
        or quartic_tc2_reranked_obligation_status.get("decision") != "completed"
        or quartic_tc2_reranked_obligation_status.get("reason")
        != "reranked_selector_complete_full_tube_still_open"
        or quartic_tc2_reranked_obligation_status.get("permanently_stopped") is not False
        or quartic_tc2_reranked_obligation_status.get("claims", {}).get("full_mixed_sector_closed")
        is not True
        or any(
            quartic_tc2_reranked_obligation_status.get("claims", {}).get(key) is not False
            for key in (
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
        raise ValueError("quartic TC2 reranked obligation checkpoint is inconsistent")
    fourth_counts = quartic_tc2_fourth_jet_range_obligations.get("counts", {})
    fourth_chunk_counts_0 = quartic_tc2_fourth_jet_chunk_0.get("counts", {})
    fourth_chunk_counts_32 = quartic_tc2_fourth_jet_chunk_32.get("counts", {})
    fourth_chunk_counts_64 = quartic_tc2_fourth_jet_chunk_64.get("counts", {})
    fourth_chunk_counts_96 = quartic_tc2_fourth_jet_chunk_96.get("counts", {})
    if (
        quartic_tc2_fourth_jet_range_obligations.get("status")
        != "pass_exact_fourth_jet_minimal_selector_manifest_no_evaluations_tube_fail_closed"
        or fourth_counts.get("active_directions") != 15
        or fourth_counts.get("fourth_selector_records") != 3_060
        or fourth_counts.get("candidate_fourth_jet_obligations") != 36_720
        or fourth_counts.get("fourth_jet_obligations_evaluated") != 0
        or quartic_tc2_fourth_jet_chunk_0.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_0.get("selected") != 32
        or fourth_chunk_counts_0.get("candidate_solvable") != 384
        or fourth_chunk_counts_0.get("candidate_obstructed") != 0
        or fourth_chunk_counts_0.get("fourth_obligations_remaining") != 3_028
        or fourth_chunk_counts_0.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_32.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_32.get("selected") != 32
        or fourth_chunk_counts_32.get("candidate_solvable") != 384
        or fourth_chunk_counts_32.get("candidate_obstructed") != 0
        or fourth_chunk_counts_32.get("fourth_obligations_remaining") != 2_996
        or fourth_chunk_counts_32.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_32.get("chunk_contract", {}).get(
            "prior_resume_sha256"
        )
        != quartic_tc2_fourth_jet_chunk_0.get("chunk_contract", {}).get(
            "resume_tip_sha256"
        )
        or quartic_tc2_fourth_jet_chunk_64.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_64.get("selected") != 32
        or fourth_chunk_counts_64.get("candidate_solvable") != 384
        or fourth_chunk_counts_64.get("candidate_obstructed") != 0
        or fourth_chunk_counts_64.get("fourth_obligations_remaining") != 2_964
        or fourth_chunk_counts_64.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_64.get("chunk_contract", {}).get(
            "prior_resume_sha256"
        )
        != quartic_tc2_fourth_jet_chunk_32.get("chunk_contract", {}).get(
            "resume_tip_sha256"
        )
        or quartic_tc2_fourth_jet_chunk_96.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_96.get("selected") != 32
        or fourth_chunk_counts_96.get("candidate_solvable") != 384
        or fourth_chunk_counts_96.get("candidate_obstructed") != 0
        or fourth_chunk_counts_96.get("fourth_obligations_remaining") != 2_932
        or fourth_chunk_counts_96.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_96.get("chunk_contract", {}).get(
            "prior_resume_sha256"
        )
        != quartic_tc2_fourth_jet_chunk_64.get("chunk_contract", {}).get(
            "resume_tip_sha256"
        )
        or quartic_tc2_fourth_jet_checkpoint.get("next_obligation_offset") != 128
        or quartic_tc2_fourth_jet_checkpoint.get("remaining_obligations") != 2_932
        or quartic_tc2_fourth_jet_checkpoint.get("current_artifact_content_sha256")
        != quartic_tc2_fourth_jet_chunk_96.get("content_sha256")
        or quartic_tc2_fourth_jet_checkpoint.get("completed_chunks") != 4
        or len(quartic_tc2_fourth_jet_checkpoint.get("history", [])) != 4
        or quartic_tc2_fourth_jet_status.get("checkpoint_content_sha256")
        != quartic_tc2_fourth_jet_checkpoint.get("content_sha256")
        or quartic_tc2_fourth_jet_status.get("decision") != "checkpointed"
        or any(
            quartic_tc2_fourth_jet_status.get("claims", {}).get(key) is not False
            for key in (
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
        raise ValueError("quartic TC2 fourth-jet obligation service is inconsistent")

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
            "candidate_body_persistence": compiler_registry["candidate_body_persistence"],
            "default_execution_enabled": compiler_registry["default_execution_enabled"],
            "fixture_expected_counts": compiler_registry["fixture_expected_counts"],
            "next_stage_adapter_registered": compiler_registry["next_stage_adapter_registered"],
            "novelty_claim_allowed": compiler_registry["novelty_claim_allowed"],
            "status": compiler_registry["status"],
        },
        "reviewed_local_epoch": {
            "default_execution_enabled": local_formula_epoch["default_execution_enabled"],
            "expected_bounded_status": local_formula_epoch["expected_bounded_status"],
            "formula_body_persistence": local_formula_epoch["formula_body_persistence"],
            "network_calls": local_formula_epoch["network_calls"],
            "paid_spend_usd": local_formula_epoch["paid_spend_usd"],
            "status": local_formula_epoch["status"],
        },
        "reviewed_local_service": {
            "budgets": local_formula_service["budgets"],
            "default_execution_enabled": local_formula_service["default_execution_enabled"],
            "deterministic_export": local_formula_service["deterministic_export"],
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
            "scientific_outcome_semantics": {
                "pass": "pass",
                "reject": "reject",
                "unresolved": "block",
            },
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
                "mean": streaming["screening"]["resumed_wave"][
                    "physical_gpu_utilization_mean_percent"
                ],
                "peak": streaming["screening"]["resumed_wave"][
                    "physical_gpu_utilization_peak_percent"
                ],
            },
            "deadline": "completed_artifact_no_live_deadline",
        },
        "promotion_overlay": {
            "state": promotion["state"],
            "lift": {
                "pass": promotion["lift_passed_count"],
                "reject": promotion["upstream_terminal_candidate_count"],
            },
            "formal": {
                "pass": promotion["formal_passed_count"],
                "reject": promotion["formal_rejected_count"],
                "block": promotion["remaining_formal_blocked_count"],
            },
            "observational_opened": promotion["solar_opened_count"]
            + promotion["galaxy_opened_count"],
            "deadline": "completed_artifact_no_live_deadline",
        },
        "g4_solar_evaluator": {
            "candidate_id": g4_solar["candidate"]["candidate_id"],
            "decision": g4_solar["decision"],
            "descriptor_implementation_ready": g4_solar["descriptor_implementation_ready"],
            "filled_registration_hash_count": solar_decision["filled_registration_hash_count"],
            "missing_registration_hash_count": len(solar_decision["missing_registration_hashes"]),
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
            "descriptor_implementation_ready": g4_galaxy["descriptor_implementation_ready"],
            "filled_registration_hash_count": galaxy_decision["filled_registration_hash_count"],
            "first_missing_premise": g4_galaxy["first_missing_premise"],
            "missing_registration_hash_count": len(galaxy_decision["missing_registration_hashes"]),
            "object_specific_gravity_parameter_count": g4_galaxy["synthetic_controls"]["shape"][
                "object_specific_gravity_parameter_count"
            ],
            "observational_data_opened": g4_galaxy["observational_data_opened"],
            "prediction_bundle_registered": g4_galaxy["prediction_bundle_registered"],
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
                "first_missing_premise": g4_galaxy_forward_model["first_missing_premise"],
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
                "branch_contract_status": g4_galaxy_branch_distance["branch_contract_status"],
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
                    g4_galaxy_prediction_contract_transform["newly_filled_registration_fields"]
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
                    "enabled": g4_galaxy_manifest_bundle_tooling["tooling_readiness"]["enabled"],
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
                    "candidate_decision_counts": parameter_compilation["candidate_decision_counts"],
                    "compiled_action_ir_count": parameter_compilation["compiled_action_ir_count"],
                    "equivalent_duplicate_count": parameter_compilation[
                        "equivalent_duplicate_count"
                    ],
                    "expensive_formal_campaign_run": False,
                    "formal_decision_counts": {},
                    "unique_candidate_count": parameter_compilation["unique_candidate_count"],
                    "generated_action_export": {
                        "candidate_count": generated_candidate_formal_export["candidate_count"],
                        "action_export_counts": generated_candidate_formal_export[
                            "action_export_counts"
                        ],
                        "metric_variation_counts": generated_candidate_formal_export[
                            "metric_variation_counts"
                        ],
                        "sandbox_backend": generated_candidate_formal_export["sandbox_receipt"][
                            "backend_mode"
                        ],
                        "network_namespace_created": generated_candidate_formal_export[
                            "sandbox_receipt"
                        ]["network_namespace_created"],
                        "action_export_historical_first_missing_premise": generated_candidate_formal_export[
                            "first_missing_premise"
                        ],
                        "first_missing_premise": generated_candidate_metric_variation[
                            "first_missing_premise"
                        ],
                        "candidate_metric_specialization": {
                            "candidate_count": generated_candidate_metric_variation[
                                "candidate_count"
                            ],
                            "counts": generated_candidate_metric_variation[
                                "metric_variation_execution_counts"
                            ],
                            "first_missing_premise": generated_candidate_metric_variation[
                                "first_missing_premise"
                            ],
                            "scope": generated_candidate_metric_variation["scope"],
                        },
                    },
                    "formal_preflight": {
                        "candidate_count": formal_preflight["candidate_count"],
                        "decision_counts": formal_preflight["decision_counts"],
                        "expensive_adm_or_global_energy_run": False,
                        "family_decision_counts": formal_preflight["family_decision_counts"],
                        "gate_counts": formal_preflight["gate_counts"],
                        "next_promotion_hook": formal_preflight["next_promotion_hook"],
                        "work_state_counts": formal_preflight["work_state_counts"],
                        "promotion_admission": {
                            "decision_counts": promotion_admission["decision_counts"],
                            "downstream_expensive_execution_started": False,
                            "eligible_candidate_count": promotion_admission[
                                "eligible_candidate_count"
                            ],
                            "preflight_blocked_excluded_count": promotion_admission[
                                "preflight_blocked_excluded_count"
                            ],
                            "target_queue_counts": promotion_admission["target_queue_counts"],
                            "work_state_counts": promotion_admission["work_state_counts"],
                            "family_formal_execution": {
                                "aether": {
                                    "candidate_count": aether_candidate_formal["candidate_count"],
                                    "decision_counts": aether_candidate_formal["decision_counts"],
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
                                    "candidate_count": g2_candidate_formal["candidate_count"],
                                    "predecessor_decision_counts": g2_candidate_formal[
                                        "decision_counts"
                                    ],
                                    "decision_counts": g2_nonmaximal_followup["decision_counts"],
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
                                        "decision_counts": g2_solar_readiness["decision_counts"],
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
                                    "work_state_counts": g2_candidate_formal["work_state_counts"],
                                },
                                "g3": {
                                    "blocker_counts": g3_candidate_formal["blocker_counts"],
                                    "candidate_count": g3_candidate_formal["candidate_count"],
                                    "decision_counts": g3_candidate_formal["decision_counts"],
                                    "full_formal_pass_count": g3_candidate_formal[
                                        "full_formal_pass_count"
                                    ],
                                    "gate_counts": g3_candidate_formal["gate_counts"],
                                    "work_state_counts": g3_candidate_formal["work_state_counts"],
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
                "formal_decision_counts": scalable_structural_metrics["formal_decision_counts"],
                "measurement_counts": scalable_structural_metrics["structural_measurement_counts"],
                "simplicity_pareto_front": scalable_structural_metrics["simplicity_pareto_front"],
                "simplicity_top10": scalable_structural_metrics["simplicity_top10"],
                "alias_multiplicity_top10": scalable_structural_metrics["alias_multiplicity_top10"],
                "scientific_validity_inference": False,
            },
            "explanation_dossiers": {
                "alias_count": scalable_explanation_dossiers["alias_count"],
                "candidate_count": scalable_explanation_dossiers["candidate_count"],
                "formal_decision_counts": scalable_explanation_dossiers["formal_decision_counts"],
                "hierarchy_node_status_counts": scalable_explanation_dossiers[
                    "hierarchy_node_status_counts"
                ],
                "observational_data_opened": False,
                "dossier_registry_root_sha256": scalable_explanation_dossiers["provenance"][
                    "dossier_registry_root_sha256"
                ],
            },
            "staged_epoch": {
                "stage_count": scalable_campaign_epoch["stage_count"],
                "sealed_epoch_counts": scalable_campaign_epoch["sealed_epoch_counts"],
                "next_epoch_readiness": scalable_campaign_epoch["next_epoch_readiness"],
                "reviewed_future_chunk": {
                    "input_cell_count": scalable_future_parameter_chunk["input_cell_count"],
                    "disposition_counts": scalable_future_parameter_chunk["disposition_counts"],
                    "preflight": {
                        "candidate_count": scalable_future_formal_preflight["candidate_count"],
                        "decision_counts": scalable_future_formal_preflight["decision_counts"],
                        "family_counts": scalable_future_formal_preflight["family_counts"],
                        "first_blocker_counts": scalable_future_formal_preflight[
                            "first_blocker_counts"
                        ],
                        "full_candidate_specific_formal_completion_claimed": False,
                        "promotion": scalable_future_formal_preflight["promotion"],
                    },
                    "family_followup": {
                        "aether": {
                            "candidate_count": future_aether_principal_inverse_fredholm_gate[
                                "candidate_count"
                            ],
                            "decision_counts": future_aether_principal_inverse_fredholm_gate[
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
                            "flat_static_global_pure_twist_AE_completion_obstructed_count": future_aether_pure_twist_ae_no_go[
                                "flat_static_global_pure_twist_AE_completion_obstructed_count"
                            ],
                            "compact_cutoff_non_pure_twist_transition_required_count": future_aether_pure_twist_ae_no_go[
                                "compact_cutoff_non_pure_twist_transition_required_count"
                            ],
                            "normalized_transition_symmetric_gradient_norm_squared_counts": future_aether_pure_twist_ae_no_go[
                                "normalized_transition_symmetric_gradient_norm_squared_counts"
                            ],
                            "differentiated_Killing_system": future_aether_pure_twist_ae_no_go[
                                "symbolic_obstruction_control"
                            ]["differentiated_Killing_system"],
                            "constraint_satisfying_negative_total_energy_datum_count": future_aether_nonlinear_lift_characteristic_gate[
                                "constraint_satisfying_negative_total_energy_datum_count"
                            ],
                            "weak_field_linearized_constraint_completion_count": future_aether_weak_field_ae_constraint_gate[
                                "weak_field_linearized_constraint_completion_count"
                            ],
                            "strictly_positive_compact_quadratic_energy_count": future_aether_weak_field_ae_constraint_gate[
                                "strictly_positive_compact_quadratic_energy_count"
                            ],
                            "weak_field_negative_completed_energy_direction_count": future_aether_weak_field_ae_constraint_gate[
                                "weak_field_negative_completed_energy_direction_count"
                            ],
                            "compact_finite_amplitude_Aether_seed_count": future_aether_finite_amplitude_negative_seed_gate[
                                "compact_finite_amplitude_Aether_seed_count"
                            ],
                            "exact_negative_static_source_monopole_count": future_aether_finite_amplitude_negative_seed_gate[
                                "exact_negative_static_source_monopole_count"
                            ],
                            "frozen_source_linearized_constraint_completion_count": future_aether_finite_amplitude_negative_seed_gate[
                                "frozen_source_linearized_constraint_completion_count"
                            ],
                            "negative_linearized_completed_boundary_energy_coefficient_count": future_aether_finite_amplitude_negative_seed_gate[
                                "negative_linearized_completed_boundary_energy_coefficient_count"
                            ],
                            "registered_seed_characteristic_crossing_count": future_aether_nonlinear_lift_characteristic_gate[
                                "registered_seed_characteristic_crossing_count"
                            ],
                            "negative_source_family_forced_characteristic_crossing_count": future_aether_nonlinear_lift_characteristic_gate[
                                "negative_source_family_forced_characteristic_crossing_count"
                            ],
                            "certified_negative_characteristic_free_amplitude_window_count": future_aether_nonlinear_lift_characteristic_gate[
                                "certified_negative_characteristic_free_amplitude_window_count"
                            ],
                            "globally_noncharacteristic_candidate_count": future_aether_nonlinear_lift_characteristic_gate[
                                "globally_noncharacteristic_candidate_count"
                            ],
                            "regular_ADM_implicit_lift_prerequisite_pass_count": future_aether_nonlinear_lift_characteristic_gate[
                                "regular_ADM_implicit_lift_prerequisite_pass_count"
                            ],
                            "uniform_Aether_Legendre_block_inverse_pass_count": future_aether_regular_adm_inverse_margin_gate[
                                "uniform_Aether_Legendre_block_inverse_pass_count"
                            ],
                            "strict_negative_source_margin_pass_count": future_aether_regular_adm_inverse_margin_gate[
                                "strict_negative_source_margin_pass_count"
                            ],
                            "typed_weighted_operator_contract_complete_count": future_aether_weighted_ift_contract_gate[
                                "typed_weighted_operator_contract_complete_count"
                            ],
                            "declared_metric_weighted_contract_count": future_aether_weighted_reference_operator_gate[
                                "declared_metric_weighted_contract_count"
                            ],
                            "metric_reference_principal_ellipticity_pass_count": future_aether_weighted_reference_operator_gate[
                                "metric_reference_principal_ellipticity_pass_count"
                            ],
                            "metric_reference_trivial_kernel_pass_count": future_aether_weighted_reference_operator_gate[
                                "metric_reference_trivial_kernel_pass_count"
                            ],
                            "registered_compact_source_right_inverse_count": future_aether_weighted_reference_operator_gate[
                                "registered_compact_source_right_inverse_count"
                            ],
                            "candidate_Aether_constraint_principal_block_pass_count": future_aether_weighted_reference_operator_gate[
                                "candidate_Aether_constraint_principal_block_pass_count"
                            ],
                            "full_coupled_Fredholm_operator_defined_count": future_aether_weighted_reference_operator_gate[
                                "full_coupled_Fredholm_operator_defined_count"
                            ],
                            "weighted_full_constraint_operator_isomorphism_pass_count": future_aether_weighted_reference_operator_gate[
                                "full_weighted_operator_isomorphism_pass_count"
                            ],
                            "nonlinear_Frechet_remainder_bound_pass_count": future_aether_weighted_reference_operator_gate[
                                "nonlinear_remainder_bound_pass_count"
                            ],
                            "completed_boundary_sign_persistence_count": future_aether_weighted_reference_operator_gate[
                                "completed_boundary_sign_persistence_count"
                            ],
                            "fixed_free_data_constraint_variable_classification_count": future_aether_fixed_free_data_principal_gate[
                                "positive_unit_branch_constraint_variable_classification_count"
                            ],
                            "zero_dimensional_Aether_constraint_diagonal_block_count": future_aether_fixed_free_data_principal_gate[
                                "zero_dimensional_Aether_constraint_diagonal_block_count"
                            ],
                            "zero_Aether_second_order_off_diagonal_columns_count": future_aether_fixed_free_data_principal_gate[
                                "zero_Aether_second_order_off_diagonal_columns_count"
                            ],
                            "augmented_Aether_unknown_nonelliptic_negative_control_count": future_aether_fixed_free_data_principal_gate[
                                "augmented_Aether_unknown_nonelliptic_negative_control_count"
                            ],
                            "finite_tilt_metric_York_symbol_derived_count": future_aether_finite_tilt_york_symbol_gate[
                                "finite_tilt_metric_York_symbol_derived_count"
                            ],
                            "uniform_fixed_free_data_principal_ellipticity_pass_count": future_aether_finite_tilt_york_symbol_gate[
                                "uniform_fixed_free_data_principal_ellipticity_pass_count"
                            ],
                            "exact_nonelliptic_York_shell_count": future_aether_finite_tilt_york_symbol_gate[
                                "exact_nonelliptic_York_shell_count"
                            ],
                            "York_ansatz_reject_count": future_aether_finite_tilt_york_symbol_gate[
                                "York_ansatz_reject_count"
                            ],
                            "finite_tilt_weighted_Fredholm_isomorphism_pass_count": future_aether_finite_tilt_york_symbol_gate[
                                "weighted_Fredholm_isomorphism_pass_count"
                            ],
                            "uniform_principal_symbol_inverse_bound_pass_count": future_aether_principal_inverse_fredholm_gate[
                                "uniform_principal_symbol_inverse_bound_pass_count"
                            ],
                            "principal_elliptic_homotopy_to_reference_pass_count": future_aether_principal_inverse_fredholm_gate[
                                "principal_elliptic_homotopy_to_reference_pass_count"
                            ],
                            "distributed_lower_order_coefficient_registry_complete_count": future_aether_principal_inverse_fredholm_gate[
                                "distributed_lower_order_coefficient_registry_complete_count"
                            ],
                            "full_operator_inverse_norm_pass_count": future_aether_principal_inverse_fredholm_gate[
                                "full_operator_inverse_norm_pass_count"
                            ],
                            "missing_weighted_contract_field_counts": future_aether_weighted_ift_contract_gate[
                                "missing_contract_field_counts"
                            ],
                            "finite_amplitude_nonlinear_constraint_completion_count": future_aether_nonlinear_lift_characteristic_gate[
                                "full_nonlinear_constraint_completion_count"
                            ],
                            "c2_plus_c3_counts": future_aether_weak_field_ae_constraint_gate[
                                "c2_plus_c3_counts"
                            ],
                            "first_blocker_counts": future_aether_principal_inverse_fredholm_gate[
                                "first_blocker_counts"
                            ],
                            "candidate_rejection_authorized_count": future_aether_principal_inverse_fredholm_gate[
                                "candidate_rejection_authorized_count"
                            ],
                        },
                        "g3": {
                            "candidate_count": future_g3_general_geometry_surplus_mismatch[
                                "candidate_count"
                            ],
                            "decision_counts": future_g3_general_geometry_surplus_mismatch[
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
                            "AF_decaying_gradient_profile_pass_count": future_g3_af_transition_obstruction[
                                "AF_decaying_gradient_profile_pass_count"
                            ],
                            "AF_principal_common_cone_profile_pass_count": future_g3_af_transition_obstruction[
                                "AF_principal_common_cone_profile_pass_count"
                            ],
                            "flat_reference_constraint_ansatz_reject_count": future_g3_af_transition_obstruction[
                                "flat_reference_constraint_ansatz_reject_count"
                            ],
                            "nonunitary_formulation_registration_pass_count": future_g3_nonunitary_af_constraint_gate[
                                "nonunitary_formulation_registration_pass_count"
                            ],
                            "nonunitary_AF_principal_pass_count": future_g3_nonunitary_af_constraint_gate[
                                "nonunitary_AF_principal_pass_count"
                            ],
                            "flat_nontrivial_reference_constraint_ansatz_reject_count": future_g3_nonunitary_af_constraint_gate[
                                "flat_nontrivial_reference_constraint_ansatz_reject_count"
                            ],
                            "actual_AF_vacuum_constraint_reference_pass_count": future_g3_nonunitary_af_constraint_gate[
                                "actual_AF_vacuum_constraint_reference_pass_count"
                            ],
                            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count": future_g3_nonunitary_af_constraint_gate[
                                "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
                            ],
                            "radial_pure_trace_momentum_constraint_reduction_pass_count": future_g3_radial_conformal_constraint_reduction[
                                "radial_pure_trace_momentum_constraint_reduction_pass_count"
                            ],
                            "radial_Lichnerowicz_BVP_registration_pass_count": future_g3_radial_conformal_constraint_reduction[
                                "radial_Lichnerowicz_BVP_registration_pass_count"
                            ],
                            "positive_global_radial_Lichnerowicz_solution_nonexistence_count": future_g3_radial_lichnerowicz_bvp_no_go[
                                "positive_global_radial_Lichnerowicz_solution_nonexistence_count"
                            ],
                            "radial_conformal_pure_trace_ansatz_reject_count": future_g3_radial_lichnerowicz_bvp_no_go[
                                "radial_conformal_pure_trace_ansatz_reject_count"
                            ],
                            "nonradial_York_Hamiltonian_reduction_pass_count": future_g3_nonradial_york_bounded_mean_curvature_no_go[
                                "nonradial_York_Hamiltonian_reduction_pass_count"
                            ],
                            "bounded_mean_curvature_green_comparison_pass_count": future_g3_nonradial_york_bounded_mean_curvature_no_go[
                                "bounded_mean_curvature_green_comparison_pass_count"
                            ],
                            "conformally_flat_bounded_mean_curvature_York_class_reject_count": future_g3_nonradial_york_bounded_mean_curvature_no_go[
                                "conformally_flat_bounded_mean_curvature_York_class_reject_count"
                            ],
                            "candidate_millicap_frontier_registration_pass_count": future_g3_york_mean_curvature_frontier[
                                "candidate_millicap_frontier_registration_pass_count"
                            ],
                            "strict_extension_beyond_kappa_6_over_5_pass_count": future_g3_york_mean_curvature_frontier[
                                "strict_extension_beyond_kappa_6_over_5_pass_count"
                            ],
                            "expanded_nonradial_York_class_reject_count": future_g3_york_mean_curvature_frontier[
                                "expanded_nonradial_York_class_reject_count"
                            ],
                            "next_grid_cap_inconclusive_count": future_g3_york_mean_curvature_frontier[
                                "next_grid_cap_inconclusive_count"
                            ],
                            "exact_algebraic_threshold_pass_count": future_g3_york_analytic_threshold[
                                "exact_algebraic_threshold_pass_count"
                            ],
                            "closed_threshold_endpoint_reject_count": future_g3_york_analytic_threshold[
                                "closed_threshold_endpoint_reject_count"
                            ],
                            "above_threshold_control_inconclusive_count": future_g3_york_analytic_threshold[
                                "above_threshold_negative_control_inconclusive_count"
                            ],
                            "tracefree_compensation_bound_pass_count": future_g3_york_tracefree_compensation[
                                "exact_tracefree_compensation_bound_pass_count"
                            ],
                            "tracefree_compensated_York_class_reject_count": future_g3_york_tracefree_compensation[
                                "tracefree_compensated_York_class_reject_count"
                            ],
                            "undercompensated_control_inconclusive_count": future_g3_york_tracefree_compensation[
                                "undercompensated_negative_control_inconclusive_count"
                            ],
                            "general_geometry_pointwise_theorem_pass_count": future_g3_general_geometry_curvature_shortfall[
                                "general_geometry_pointwise_theorem_pass_count"
                            ],
                            "curvature_shortfall_constraint_class_reject_count": future_g3_general_geometry_curvature_shortfall[
                                "curvature_shortfall_constraint_class_reject_count"
                            ],
                            "exact_curvature_endpoint_inconclusive_count": future_g3_general_geometry_curvature_shortfall[
                                "exact_curvature_endpoint_inconclusive_count"
                            ],
                            "above_threshold_not_excluded_control_count": future_g3_general_geometry_curvature_shortfall[
                                "above_threshold_not_excluded_control_count"
                            ],
                            "nonconformally_flat_metric_construction_pass_count": future_g3_general_geometry_curvature_shortfall[
                                "nonconformally_flat_metric_construction_pass_count"
                            ],
                            "exact_surplus_identity_pass_count": future_g3_general_geometry_surplus_mismatch[
                                "exact_surplus_identity_pass_count"
                            ],
                            "above_threshold_surplus_mismatch_class_reject_count": future_g3_general_geometry_surplus_mismatch[
                                "above_threshold_surplus_mismatch_class_reject_count"
                            ],
                            "matched_surplus_necessary_control_count": future_g3_general_geometry_surplus_mismatch[
                                "matched_surplus_necessary_control_count"
                            ],
                            "overcurvature_not_excluded_control_count": future_g3_general_geometry_surplus_mismatch[
                                "overcurvature_not_excluded_control_count"
                            ],
                            "registered_AF_metric_York_datum_pass_count": future_g3_general_geometry_surplus_mismatch[
                                "registered_AF_metric_York_datum_pass_count"
                            ],
                            "asymptotically_flat_Dirac_pass_count": future_g3_af_transition_obstruction[
                                "AF_unitary_lapse_Dirac_pass_count"
                            ],
                            "AF_Einstein_constraint_solution_pass_count": future_g3_general_geometry_surplus_mismatch[
                                "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
                            ],
                            "global_energy_pass_count": future_g3_general_geometry_surplus_mismatch[
                                "global_hamiltonian_energy_pass_count"
                            ],
                            "full_formal_pass_count": future_g3_general_geometry_surplus_mismatch[
                                "full_formal_pass_count"
                            ],
                            "first_blocker_counts": future_g3_general_geometry_surplus_mismatch[
                                "first_blocker_counts"
                            ],
                        },
                    },
                    "action_dossiers": {
                        "candidate_count": future_candidate_action_dossier["candidate_count"],
                        "decision_counts": future_candidate_action_dossier["decision_counts"],
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
                "closed_candidate_count": quartic_counts["reference_two_jets_closed"],
                "closed_derivative_orders": quartic_control["reference_jet_orders_closed"],
                "D2_coordinate_linf_to_Frobenius_ceiling": quartic_tc2_quadratic_deltak[
                    "quadratic_D2_envelopes"
                ][0]["D2_deltaK_coordinate_linf_to_Frobenius_integer_ceiling"],
                "full_tube_Sylvester_identity_closed": False,
            },
            "diagonal_third_jet": {
                "active_direction_count": quartic_third_slice["active_coordinate_directions"],
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
                "remaining_mixed_triples": 10_700,
                "mixed_third_jet_closures": 1_600,
            },
            "mixed_third_jet_chunk": {
                "chunk_offset": quartic_parallel_contract["chunk_offset"],
                "latest_chunk_processed_count": quartic_parallel_contract["processed_count"],
                "processed_count": 1_600,
                "next_offset": quartic_parallel_contract["next_offset"],
                "triple_kind_counts": quartic_parallel_counts["triple_kind_counts"],
                "symbolic_parameter_compatible": 1_600,
                "latest_candidate_evaluations": quartic_parallel_counts["candidate_evaluations"],
                "candidate_evaluations": 19_200,
                "candidate_solvable": 19_200,
                "candidate_obstructed": 0,
                "remaining_mixed_triples": quartic_parallel_counts["mixed_triples_remaining"],
                "resume_tip_sha256": quartic_parallel_contract["resume_tip_sha256"],
                "service_decision": quartic_tc2_mixed_third_jet_parallel_status["decision"],
                "parallel_worker_count": quartic_parallel_contract["parallel_worker_count"],
                "parallel_execution_policy": quartic_parallel_contract["parallel_execution_policy"],
                "sequential_predecessor_processed_count": 192,
                "full_mixed_sector_closed": quartic_tc2_reranked_obligation_status["claims"][
                    "full_mixed_sector_closed"
                ],
            },
            "mixed_third_jet_reduction": {
                "active_direction_rank": quartic_rerank["active_direction_rank"],
                "symmetric_cubic_dimension": quartic_rerank["symmetric_cubic_dimension"],
                "stable_combined_evidence_rank": quartic_rerank["stable_combined_evidence_rank"],
                "rank_gain_over_prior_reduction": quartic_rerank["rank_gain_over_prior_reduction"],
                "reranked_exact_obligations": quartic_rerank["reranked_obligation_count"],
                "reranked_obligation_kind_counts": quartic_rerank[
                    "reranked_obligation_kind_counts"
                ],
                "candidate_evaluation_budget": quartic_tc2_mixed_third_jet_reranked_reduction[
                    "reranked_obligation_selector"
                ]["candidate_evaluations_if_all_obligations_are_run"],
                "first_selector_index": quartic_rerank["first_selector_index"],
                "last_selector_index": quartic_rerank["last_selector_index"],
                "brute_force_unevaluated_triples": quartic_rerank_counts[
                    "stable_mixed_triples_remaining"
                ],
                "obligations_evaluated": reranked_total_processed,
                "obligations_remaining": 447 - reranked_total_processed,
                "candidate_evaluations": reranked_candidate_evaluations,
                "candidate_solvable": reranked_candidate_evaluations,
                "candidate_obstructed": 0,
                "next_obligation_offset": quartic_tc2_reranked_obligation_status[
                    "next_obligation_offset"
                ],
                "resume_tip_sha256": quartic_tc2_reranked_obligation_status["prior_resume_sha256"],
                "obligations_inferred_passed": quartic_rerank_counts[
                    "remaining_active_triples_inferred_passed"
                ],
                "completion_rank": quartic_rerank["completion_rank"],
                "drop_final_obligation_rank": quartic_rerank["drop_final_obligation_rank"],
            },
            "fourth_jet_range_obligations": {
                "active_direction_rank": fourth_counts["active_directions"],
                "selector_obligations": fourth_counts["fourth_selector_records"],
                "candidate_obligation_budget": fourth_counts[
                    "candidate_fourth_jet_obligations"
                ],
                "obligations_evaluated": (
                    fourth_chunk_counts_0["selected"]
                    + fourth_chunk_counts_32["selected"]
                    + fourth_chunk_counts_64["selected"]
                    + fourth_chunk_counts_96["selected"]
                ),
                "obligations_remaining": fourth_chunk_counts_96[
                    "fourth_obligations_remaining"
                ],
                "candidate_evaluations": (
                    fourth_chunk_counts_0["candidate_evaluations"]
                    + fourth_chunk_counts_32["candidate_evaluations"]
                    + fourth_chunk_counts_64["candidate_evaluations"]
                    + fourth_chunk_counts_96["candidate_evaluations"]
                ),
                "candidate_solvable": (
                    fourth_chunk_counts_0["candidate_solvable"]
                    + fourth_chunk_counts_32["candidate_solvable"]
                    + fourth_chunk_counts_64["candidate_solvable"]
                    + fourth_chunk_counts_96["candidate_solvable"]
                ),
                "candidate_obstructed": (
                    fourth_chunk_counts_0["candidate_obstructed"]
                    + fourth_chunk_counts_32["candidate_obstructed"]
                    + fourth_chunk_counts_64["candidate_obstructed"]
                    + fourth_chunk_counts_96["candidate_obstructed"]
                ),
                "directional_evaluations": (
                    fourth_chunk_counts_0["directional_evaluations"]
                    + fourth_chunk_counts_32["directional_evaluations"]
                    + fourth_chunk_counts_64["directional_evaluations"]
                    + fourth_chunk_counts_96["directional_evaluations"]
                ),
                "next_obligation_offset": quartic_tc2_fourth_jet_status[
                    "next_obligation_offset"
                ],
                "resume_tip_sha256": quartic_tc2_fourth_jet_status["prior_resume_sha256"],
                "parallel_worker_count": 8,
                "full_fourth_jet_range_closed": False,
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
            "first_missing_premise": "remaining_2932_exact_fourth_jet_range_obligations_then_all_order_remainder_or_nonlinear_range_theorem",
        },
        "evidence_pareto": {
            "candidate_decision_counts": pareto["candidate_decision_counts"],
            "normalized_candidate_outcomes": {"pass": 0, "reject": 0, "block": 6},
            "candidate_evidence_packet_counts": {
                key: value
                for key, value in pareto["evidence_packet_outcome_counts"].items()
                if key == "blocked"
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
            "current_missing_evaluator_blockers": dict(
                sorted(
                    Counter(packet["task_type"] for packet in followup["deferred_packets"]).items()
                )
            ),
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
        (value for value in scheduled_times if latest is None or value > latest),
        None,
    )
    freshness_deadline = latest + timedelta(seconds=threshold) if latest is not None else None
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
        "physical_cpu": (dict(physical_cpu) if physical_cpu is not None else sample_host_cpu()),
        "physical_gpu": dict(physical_gpu) if physical_gpu is not None else sample_nvidia_smi(),
        "campaign_watchdog_freshness": {
            "latest_event_utc": watchdog["latest_event_utc"],
            "age_seconds": age,
            "expected_next_event_not_before_utc": (
                scheduled_event_anchor.isoformat() if scheduled_event_anchor is not None else None
            ),
            "freshness_deadline_utc": (
                freshness_deadline.isoformat() if freshness_deadline is not None else None
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
        "mixed_third_jet_supervisor": mixed_third_jet_supervisor,
        "deadline_state": (
            "unavailable"
            if not _parse_utc(watchdog["campaign"]["deadline_utc"])
            else "expired"
            if timestamp > _parse_utc(watchdog["campaign"]["deadline_utc"])
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
    refresh.add_argument("--leaderboard-config", default="configs/scientific_leaderboards.json")
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
        print(
            json.dumps(
                {
                    "core_content_sha256": snapshot["core_content_sha256"],
                    "json_bytes": len(json_bytes),
                    "dashboard_bytes": len(html_bytes) if html_bytes is not None else 0,
                },
                sort_keys=True,
            )
        )
        return 0
    snapshot_path = _resolve_inside(root, arguments.snapshot)
    output_path = _resolve_inside(root, arguments.output)
    snapshot = _load_snapshot(snapshot_path)
    validate_dashboard_input(snapshot, _sha(snapshot["core"]))
    html_bytes = render_dashboard(snapshot).encode()
    _bounded_write(root, output_path, html_bytes, arguments.maximum_output_bytes)
    print(
        json.dumps(
            {
                "core_content_sha256": snapshot["core_content_sha256"],
                "dashboard_bytes": len(html_bytes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
