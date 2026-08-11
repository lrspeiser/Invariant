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

from .process_health import pid_alive

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


def _read_unified_live_service_status(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    relative = config.get("unified_live_service_checkpoint")
    if relative is None:
        return {"availability": "not_configured", "alive": False}
    root = project_root.resolve()
    path = (root / str(relative)).resolve()
    if root not in path.parents:
        raise ValueError("unified live-service checkpoint escapes project root")
    if not path.is_file():
        return {"availability": "configured_not_started", "alive": False}
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("unified live-service checkpoint is not an object")
    claimed = value.get("content_sha256")
    without_hash = dict(value)
    without_hash.pop("content_sha256", None)
    schema = value.get("schema_version")
    implementations = {
        "sigma-unified-engine-live-service-checkpoint-1.0": "legacy",
        "sigma-unified-engine-live-service-safety-checkpoint-1.0": "hardened_safety",
    }
    if (
        schema not in implementations
        or not isinstance(claimed, str)
        or not _SHA256_RE.fullmatch(claimed)
        or _sha(without_hash) != claimed
        or not isinstance(value.get("refresh_count"), int)
        or value.get("refresh_count", -1) < 0
        or not isinstance(value.get("consecutive_failures"), int)
        or value.get("consecutive_failures", -1) < 0
    ):
        raise ValueError("unified live-service checkpoint is invalid")
    implementation = implementations[str(schema)]
    config_current = None
    pid_identity_verified = None
    if implementation == "hardened_safety":
        runtime_epoch = value.get("runtime_epoch")
        runtime_directory = value.get("runtime_directory")
        worker_identity = value.get("worker_argv_sha256")
        config_relative = config.get("unified_live_service_config")
        if (
            not isinstance(runtime_epoch, str)
            or len(runtime_epoch) < 16
            or runtime_directory != "runs/engine/unified-live-dashboard-safety-service"
            or not isinstance(worker_identity, str)
            or not _SHA256_RE.fullmatch(worker_identity)
            or config_relative is None
        ):
            raise ValueError("hardened unified live-service checkpoint is invalid")
        config_path = (root / str(config_relative)).resolve()
        if root not in config_path.parents or not config_path.is_file():
            raise ValueError("hardened unified live-service config is unavailable")
        config_current = hashlib.sha256(config_path.read_bytes()).hexdigest() == value.get(
            "config_file_sha256"
        )
        expected_tail = [
            "-m",
            "sigma_theory_compiler.unified_engine_live_service_safety",
            "worker",
            "--project-root",
            str(root),
            "--config",
            config_path.relative_to(root).as_posix(),
        ]
        expected_identity = _sha(expected_tail)
        if worker_identity != expected_identity:
            raise ValueError("hardened unified live-service worker identity drift")
        pid_identity_verified = False
        try:
            import psutil

            command = psutil.Process(int(value.get("pid"))).cmdline()
            module_index = command.index("-m")
            pid_identity_verified = _sha(command[module_index:]) == expected_identity
        except (ImportError, OSError, ValueError, TypeError):
            pid_identity_verified = False
        except psutil.Error:
            pid_identity_verified = False
    last_refresh = value.get("last_refresh")
    if last_refresh is not None and (
        not isinstance(last_refresh, dict)
        or not _SHA256_RE.fullmatch(str(last_refresh.get("core_content_sha256", "")))
        or not _SHA256_RE.fullmatch(str(last_refresh.get("snapshot_file_sha256", "")))
        or not _SHA256_RE.fullmatch(str(last_refresh.get("dashboard_file_sha256", "")))
    ):
        raise ValueError("unified live-service refresh receipt is invalid")
    return {
        "availability": "available",
        "implementation": implementation,
        "state": value.get("state"),
        "alive": (
            pid_identity_verified
            if implementation == "hardened_safety"
            else pid_alive(value.get("pid"))
        ),
        "pid_identity_verified": pid_identity_verified,
        "config_current": config_current,
        "pid": value.get("pid"),
        "refresh_count": value.get("refresh_count"),
        "consecutive_failures": value.get("consecutive_failures"),
        "last_error_kind": (
            str(value.get("last_error")).split(":", 1)[0] if value.get("last_error") else None
        ),
        "last_refresh": last_refresh,
        "stop_reason": value.get("stop_reason"),
        "checkpoint_content_sha256": claimed,
        "checkpoint_file_sha256": hashlib.sha256(raw).hexdigest(),
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
    unified_live_service = _read_unified_live_service_status(root, config)

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
    generated_candidate_formula_gpu_stress = sources["generated_candidate_formula_gpu_stress"]
    transactional_gravity_intake = sources["kastner_schlatter_transactional_gravity_intake"]
    transactional_gravity_graph = sources["kastner_schlatter_equation_graph_admission"]
    transactional_gravity_cuda = sources["kastner_schlatter_cuda_consequence_campaign"]
    transactional_gravity_observational = sources[
        "kastner_schlatter_observational_readiness_contract"
    ]
    transactional_gravity_falsification = sources["kastner_schlatter_cuda_falsification_design"]
    transactional_gravity_candidate_action = sources[
        "kastner_schlatter_candidate_action_completion"
    ]
    transactional_gravity_equivalence = sources["kastner_schlatter_action_equivalence_audit"]
    transactional_gravity_formal = sources["kastner_schlatter_candidate_action_formal_admission"]
    transactional_gravity_de_sitter = sources["kastner_schlatter_de_sitter_energy_prerequisite"]
    transactional_gravity_poisson_action = sources["kastner_schlatter_poisson_action_compatibility"]
    transactional_gravity_positive_intensity = sources[
        "kastner_schlatter_positive_intensity_preservation"
    ]
    transactional_gravity_positive_reparameterization = sources[
        "kastner_schlatter_positive_reparameterization"
    ]
    transactional_gravity_point_process_measure = sources[
        "kastner_schlatter_covariant_point_process_measure"
    ]
    transactional_gravity_poisson_selector = sources["kastner_schlatter_poisson_selector_contract"]
    transactional_gravity_conditional_poisson = sources[
        "kastner_schlatter_conditional_poisson_kernel_completion"
    ]
    transactional_gravity_actualization_history = sources[
        "kastner_schlatter_actualization_history_map_audit"
    ]
    transactional_gravity_qed_poisson_derivation = sources[
        "kastner_schlatter_qed_actualization_poisson_derivation"
    ]
    transactional_gravity_deterministic_compensator = sources[
        "kastner_schlatter_deterministic_compensator_admission"
    ]
    transactional_gravity_observable_exposure = sources[
        "kastner_schlatter_transaction_event_observable_exposure"
    ]
    transactional_gravity_poisson_cox_power = sources["kastner_schlatter_poisson_cox_cuda_power"]
    transactional_gravity_set_indexed_cuda = sources[
        "kastner_schlatter_set_indexed_cuda_falsification"
    ]
    transactional_gravity_gpu_scheduler_adapter = sources[
        "kastner_schlatter_set_indexed_gpu_scheduler_adapter"
    ]
    transactional_gravity_deferred_gpu_ownership = sources[
        "kastner_schlatter_deferred_gpu_ownership"
    ]
    transactional_gravity_scalar_cuda = sources[
        "kastner_schlatter_scalar_intensity_cuda_falsification"
    ]
    transactional_gravity_extended_geometry = sources[
        "kastner_schlatter_extended_geometry_cuda_stress"
    ]
    generic_g4_b4_termwise_normalization = sources["generic_g4_b4_termwise_normalization"]
    einstein_aether_coupling_boundary_kkt = sources["einstein_aether_coupling_boundary_kkt"]
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
    future_aether_lower_order_coefficient_contract_gate = sources[
        "future_aether_lower_order_coefficient_contract_gate"
    ]
    future_aether_canonical_seed_constraint_dag_gate = sources[
        "future_aether_canonical_seed_constraint_dag_gate"
    ]
    future_aether_characteristic_shell_hcore_gate = sources[
        "future_aether_characteristic_shell_hcore_gate"
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
    future_g3_flat_radial_matched_constraints_asymptotic_no_go = sources[
        "future_g3_flat_radial_matched_constraints_asymptotic_no_go"
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
    quartic_tc2_fourth_jet_range_obligations = sources["quartic_tc2_fourth_jet_range_obligations"]
    quartic_tc2_fourth_jet_chunk_0 = sources["quartic_tc2_fourth_jet_chunk_0"]
    quartic_tc2_fourth_jet_chunk_32 = sources["quartic_tc2_fourth_jet_chunk_32"]
    quartic_tc2_fourth_jet_chunk_64 = sources["quartic_tc2_fourth_jet_chunk_64"]
    quartic_tc2_fourth_jet_chunk_96 = sources["quartic_tc2_fourth_jet_chunk_96"]
    quartic_tc2_fourth_jet_chunk_128 = sources["quartic_tc2_fourth_jet_chunk_128"]
    quartic_tc2_fourth_jet_chunk_160 = sources["quartic_tc2_fourth_jet_chunk_160"]
    quartic_tc2_fourth_jet_chunk_192 = sources["quartic_tc2_fourth_jet_chunk_192"]
    quartic_tc2_fourth_jet_chunk_224 = sources["quartic_tc2_fourth_jet_chunk_224"]
    quartic_tc2_fourth_jet_checkpoint = sources["quartic_tc2_fourth_jet_checkpoint"]
    quartic_tc2_fourth_jet_status = sources["quartic_tc2_fourth_jet_status"]
    quartic_tc2_d4_obstruction_certificate = sources[
        "quartic_tc2_d4_obstruction_cokernel_certificate"
    ]
    quartic_tc2_d4_homogeneous_freedom_reduction = sources[
        "quartic_tc2_d4_homogeneous_freedom_reduction"
    ]
    quartic_tc2_d4_minimal_tc2_escape = sources["quartic_tc2_d4_minimal_tc2_escape"]
    quartic_tc2_d4_registered_operator_origin_no_go = sources[
        "quartic_tc2_d4_registered_operator_origin_no_go"
    ]
    quartic_tc2_d4_topology_changing_origin = sources[
        "quartic_tc2_d4_topology_changing_origin_classification"
    ]
    quartic_tc2_d4_curl_constraint_admission = sources["quartic_tc2_d4_curl_constraint_admission"]
    quartic_tc2_d4_curl_companion_range = sources["quartic_tc2_d4_curl_companion_range"]
    quartic_tc2_d4_axis2_base_rhs = sources["quartic_tc2_d4_axis2_base_rhs"]
    quartic_tc2_d4_spatial_gradient_no_go = sources[
        "quartic_tc2_d4_spatial_gradient_annihilator_no_go"
    ]
    quartic_tc2_d4_full_linear_gradient_no_go = sources[
        "quartic_tc2_d4_full_linear_gradient_annihilator_no_go"
    ]
    quartic_tc2_d4_parity_cubic_escape = sources[
        "quartic_tc2_d4_parity_cubic_angular_escape"
    ]
    quartic_tc2_d4_parity_cubic_generic_direction = sources[
        "quartic_tc2_d4_parity_cubic_generic_direction"
    ]
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
    unified_live_dashboard_service_readiness = sources["unified_live_dashboard_service_readiness"]
    unified_live_dashboard_service_safety = sources[
        "unified_live_dashboard_service_safety_readiness"
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
            "candidate_backend_variations_executed": 0,
            "candidate_euler_expressions_materialized": 163,
            "candidate_specializations_symbolically_verified": 163,
            "exact_formula_domains_validated": 163,
            "formal_passes_inferred": 0,
            "rejected": 0,
            "typed_action_hashes_replayed": 163,
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
            or record.get("candidate_action_hash_replayed") is not True
            or record.get("candidate_formula_domain_validated") is not True
            or record.get("candidate_specialized_euler_expression_materialized") is not True
            or record.get("candidate_backend_metric_variation_executed") is not False
            or record.get("formal_pass_inferred") is not False
            or record.get("formal_decision_changed") is not False
            or record.get("observational_data_opened") is not False
            for record in generated_candidate_metric_variation.get("candidate_records", [])
        )
    ):
        raise ValueError("generated candidate metric specialization is inconsistent")
    if (
        generic_g4_b4_termwise_normalization.get("status")
        != "pass_exact_24_term_generic_nonlinear_G4X_metric_Euler_normalization_to_KYY_B4"
        or generic_g4_b4_termwise_normalization.get("canonical_term_count") != 24
        or generic_g4_b4_termwise_normalization.get("matched_term_count") != 24
        or generic_g4_b4_termwise_normalization.get("nonzero_residual_count") != 0
        or generic_g4_b4_termwise_normalization.get("metric_variation_normalization_pass")
        is not True
        or generic_g4_b4_termwise_normalization.get("primary_source", {}).get("equation") != "B.4"
        or generic_g4_b4_termwise_normalization.get("primary_source", {}).get("arxiv_id")
        != "1105.5723v4"
        or generic_g4_b4_termwise_normalization.get("primary_source_transcription")
        != {
            "path": "formal/sources/kyy_1105.5723v4_eq_B4_canonical_coefficients.json",
            "file_sha256": ("497042978c3c0eed8ec02b49c5ceb2c258e60416c23152e34d44cde4ae53d32f"),
        }
        or len(generic_g4_b4_termwise_normalization.get("term_records", [])) != 24
        or any(
            record.get("residual") != "0"
            or record.get("B4_coefficient") != record.get("cadabra_coefficient")
            for record in generic_g4_b4_termwise_normalization.get("term_records", [])
        )
        or generic_g4_b4_termwise_normalization.get("negative_controls")
        != {
            "flip_R_pp_sign_rejected": True,
            "omit_G4_XX_QQ_rejected": True,
            "wrong_source_equation_rejected": True,
        }
        or generic_g4_b4_termwise_normalization.get("full_candidate_formal_pass_inferred")
        is not False
        or generic_g4_b4_termwise_normalization.get("global_energy_inferred") is not False
        or generic_g4_b4_termwise_normalization.get("observational_data_opened") is not False
        or generic_g4_b4_termwise_normalization.get("dark_matter_or_halo_inputs") is not False
        or generic_g4_b4_termwise_normalization.get("redshift_distance_inputs") is not False
        or generic_g4_b4_termwise_normalization.get("paid_llm_spend_usd") != 0.0
    ):
        raise ValueError("generic G4 B.4 termwise normalization is inconsistent")
    expected_aether_boundary_counts = {
        "D_only_ambient_singular_constrained_full_rank_witnesses": 2,
        "candidate_or_theory_reject": 0,
        "five_mode_linear_positivity_chart_bindings": 1,
        "generic_symbolic_determinant_identities_pass": 5,
        "global_nonlinear_stability_pass": 0,
        "observational_pass": 0,
        "true_constrained_rank_boundary_witnesses": 3,
    }
    aether_boundary_factorization = einstein_aether_coupling_boundary_kkt.get(
        "symbolic_factorization", {}
    )
    aether_boundary_witnesses = einstein_aether_coupling_boundary_kkt.get("exact_witnesses", {})
    if (
        einstein_aether_coupling_boundary_kkt.get("decision")
        != "constrained_coupling_boundary_rank_closed_nonlinear_stability_blocked"
        or einstein_aether_coupling_boundary_kkt.get("decision_counts")
        != {"blocked": 1, "pass": 0, "reject": 0}
        or einstein_aether_coupling_boundary_kkt.get("gate_counts")
        != expected_aether_boundary_counts
        or einstein_aether_coupling_boundary_kkt.get("first_blocker")
        != "generic_nonlinear_constraint_reduced_Hamiltonian_boundedness_and_boundary_completion_not_proven"
        or set(aether_boundary_factorization.get("identity_checks", {}).values()) != {True}
        or aether_boundary_factorization.get("constrained_KKT_11x11_determinant")
        != "-(c1 + c4)**3*(-M2 + c1 + c3)**5*(2*M2 + c1 + 3*c2 + c3)/128"
        or aether_boundary_factorization.get("tangent_9x9_determinant")
        != "(c1 + c4)**3*(-M2 + c1 + c3)**5*(2*M2 + c1 + 3*c2 + c3)/512"
        or aether_boundary_witnesses.get("D_only_inside_five_mode_positivity_chart", {}).get(
            "five_mode_chart_checks"
        )
        != {
            "1_minus_c13_positive": True,
            "c123_times_trace_positive": True,
            "vector_gradient_positive": True,
            "zero_less_c14_less_two": True,
        }
        or aether_boundary_witnesses.get("D_only_inside_five_mode_positivity_chart", {}).get(
            "ambient_rank"
        )
        != 9
        or aether_boundary_witnesses.get("D_only_inside_five_mode_positivity_chart", {}).get(
            "tangent_rank"
        )
        != 9
        or aether_boundary_witnesses.get("D_only_inside_five_mode_positivity_chart", {}).get(
            "KKT_rank"
        )
        != 11
        or {
            name: {key: witness.get(key) for key in ("ambient_rank", "tangent_rank", "KKT_rank")}
            for name, witness in aether_boundary_witnesses.get(
                "true_constrained_boundaries", {}
            ).items()
        }
        != {
            "M2_minus_c13_equals_zero": {
                "ambient_rank": 5,
                "tangent_rank": 4,
                "KKT_rank": 6,
            },
            "c14_equals_zero": {
                "ambient_rank": 7,
                "tangent_rank": 6,
                "KKT_rank": 8,
            },
            "two_M2_plus_c13_plus_3c2_equals_zero": {
                "ambient_rank": 10,
                "tangent_rank": 8,
                "KKT_rank": 10,
            },
        }
        or einstein_aether_coupling_boundary_kkt.get("reduced_five_mode_chart_binding", {}).get(
            "D_is_not_a_reduced_five_mode_boundary"
        )
        is not True
        or any(einstein_aether_coupling_boundary_kkt.get("claim_seals", {}).values())
        or any(einstein_aether_coupling_boundary_kkt.get("data_seals", {}).values())
    ):
        raise ValueError("Einstein-Aether coupling-boundary KKT gate is inconsistent")
    if (
        generated_candidate_formula_gpu_stress.get("campaign_decision")
        != "completed_numerical_stress_control_only"
        or generated_candidate_formula_gpu_stress.get("counts")
        != {
            "candidate_count": 163,
            "cpu_exact_rational_crosschecks": 5216,
            "cpu_full_projection_evaluations": 5341184,
            "family_count": 4,
            "formal_passes_inferred": 0,
            "gpu_measured_candidate_formula_evaluations": 87509958656,
            "gpu_measured_repetitions": 16384,
            "gpu_projection_dispatches": 16392,
            "gpu_warmup_repetitions": 8,
            "observational_records_accessed": 0,
            "paid_llm_calls": 0,
            "synthetic_points_per_candidate": 32768,
            "unique_candidate_point_pairs": 5341184,
        }
        or generated_candidate_formula_gpu_stress.get("family_counts")
        != {
            "AETHER_K1234_PARAMETER_CELL": 128,
            "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
            "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
            "KESSENCE_G2_CONVEX": 2,
        }
        or generated_candidate_formula_gpu_stress.get("source_bindings", {})
        .get("metric_variation_artifact", {})
        .get("content_sha256")
        != generated_candidate_metric_variation.get("content_sha256")
        or generated_candidate_formula_gpu_stress.get("exact_cpu_control", {}).get("within_bound")
        is not True
        or generated_candidate_formula_gpu_stress.get("exact_cpu_control", {}).get(
            "crosscheck_count"
        )
        != 5216
        or generated_candidate_formula_gpu_stress.get("gpu_cpu_comparison", {}).get("within_bounds")
        is not True
        or generated_candidate_formula_gpu_stress.get("gpu_cpu_comparison", {}).get(
            "comparison_count"
        )
        != 5341184
        or generated_candidate_formula_gpu_stress.get("gpu_cpu_comparison", {}).get(
            "violating_point_count"
        )
        != 0
        or generated_candidate_formula_gpu_stress.get("runtime_measurement", {})
        .get("device", {})
        .get("device_name")
        != "NVIDIA GeForce RTX 5090"
        or "device-wide NVML"
        not in generated_candidate_formula_gpu_stress.get("runtime_measurement", {})
        .get("utilization", {})
        .get("counter_scope", "")
        or generated_candidate_formula_gpu_stress.get("synthetic_only") is not True
        or generated_candidate_formula_gpu_stress.get("formal_pass_inferred") is not False
        or generated_candidate_formula_gpu_stress.get("field_equations_proven") is not False
        or generated_candidate_formula_gpu_stress.get("candidate_backend_metric_variation_executed")
        is not False
        or generated_candidate_formula_gpu_stress.get("candidate_rejection_authorized") is not False
        or generated_candidate_formula_gpu_stress.get("scientific_ranking_authorized") is not False
        or generated_candidate_formula_gpu_stress.get("observations_opened") is not False
        or generated_candidate_formula_gpu_stress.get("dark_matter_or_halo_inputs") is not False
        or generated_candidate_formula_gpu_stress.get("redshift_distance_inputs") is not False
        or generated_candidate_formula_gpu_stress.get("paid_llm_calls") is not False
    ):
        raise ValueError("generated candidate GPU formula stress control is inconsistent")
    transactional_claim_seals = {
        "fundamental_action_registered": False,
        "formal_gr_equivalence_proven": False,
        "dark_matter_elimination_proven": False,
        "dark_energy_elimination_proven": False,
        "observational_pass": False,
        "theory_validity_claimed": False,
        "cuda_execution_performed": False,
        "automatic_downstream_enqueue_performed": False,
    }
    if (
        transactional_gravity_intake.get("decision") != "blocked"
        or transactional_gravity_intake.get("first_blocker")
        != "no_candidate_bound_fundamental_action_or_complete_variational_field_system"
        or transactional_gravity_intake.get("synthetic_preflight_counts")
        != {"pass": 7, "reject": 0, "block": 1}
        or transactional_gravity_intake.get("claim_seals") != transactional_claim_seals
        or transactional_gravity_intake.get("source_binding", {}).get("official_pdf_sha256")
        != "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
    ):
        raise ValueError("Kastner-Schlatter transactional-gravity intake is inconsistent")
    graph_claim_seals = {
        "fundamental_action_registered": False,
        "variational_derivation_registered": False,
        "formal_gr_equivalence_proven": False,
        "dark_matter_elimination_proven": False,
        "dark_energy_elimination_proven": False,
        "observational_pass": False,
        "theory_validity_claimed": False,
        "automatic_downstream_enqueue_performed": False,
    }
    if (
        transactional_gravity_graph.get("decision") != "blocked"
        or transactional_gravity_graph.get("first_blocker")
        != transactional_gravity_intake.get("first_blocker")
        or transactional_gravity_graph.get("source_lineage", {}).get("source_intake_content_sha256")
        != transactional_gravity_intake.get("content_sha256")
        or transactional_gravity_graph.get("graph_counts")
        != {
            "nodes": 54,
            "edges": 137,
            "formula_nodes": 25,
            "assumption_nodes": 12,
            "domain_nodes": 6,
            "source_nodes": 2,
            "action_contract_nodes": 1,
            "absent_capability_nodes": 8,
            "dependency_edges": 18,
            "assumption_edges": 35,
            "semantic_algebraic_equivalence_edges": 1,
            "exact_duplicate_edges": 0,
            "theory_equivalence_edges": 0,
            "absent_action_edges": 33,
        }
        or transactional_gravity_graph.get("claim_seals") != graph_claim_seals
    ):
        raise ValueError("Kastner-Schlatter equation graph admission is inconsistent")
    expected_transactional_cuda_counts = {
        "cosmology_locked_a0_values_emitted": 0,
        "formal_or_theory_passes_inferred": 0,
        "gpu_kernel_dispatches": 24600,
        "gpu_measured_consequence_evaluations": 17179869184,
        "gpu_measured_repetitions": 8192,
        "gpu_warmup_repetitions": 8,
        "lambda_values_emitted": 0,
        "lorentz_equal_volume_frames": 2,
        "mond_cases": 262144,
        "observational_records_accessed": 0,
        "ontology_passes_inferred": 0,
        "paid_llm_calls": 0,
        "poisson_means": 3,
        "poisson_samples": 1572864,
        "sds_cases": 262144,
    }
    if (
        transactional_gravity_cuda.get("decision")
        != "synthetic_consequences_executed_equation_35_and_physics_claims_blocked"
        or transactional_gravity_cuda.get("first_blocker")
        != "authoritative_equation_35_h_versus_hbar_normalization_clarification"
        or transactional_gravity_cuda.get("predecessor_binding", {}).get("content_sha256")
        != transactional_gravity_intake.get("content_sha256")
        or transactional_gravity_cuda.get("counts") != expected_transactional_cuda_counts
        or transactional_gravity_cuda.get("equation_35_normalization_gate", {}).get(
            "exact_ratio_middle_to_printed"
        )
        != "2"
        or transactional_gravity_cuda.get("equation_35_normalization_gate", {}).get(
            "lambda_values_emitted"
        )
        != 0
        or transactional_gravity_cuda.get("gpu_cpu_bindings", {}).get("poisson_output_byte_equal")
        is not True
        or transactional_gravity_cuda.get("poisson_four_volume_control", {}).get(
            "all_statistical_controls_closed"
        )
        is not True
        or transactional_gravity_cuda.get("sds_root_domain_control", {}).get("decision")
        != "synthetic_domain_control_closed"
        or transactional_gravity_cuda.get("mond_btfr_control", {}).get("decision")
        != "conditional_synthetic_asymptote_control_closed"
        or transactional_gravity_cuda.get("synthetic_only") is not True
        or transactional_gravity_cuda.get("observations_opened") is not False
        or transactional_gravity_cuda.get("ontology_pass") is not False
        or transactional_gravity_cuda.get("theory_pass") is not False
        or transactional_gravity_cuda.get("formal_pass") is not False
        or transactional_gravity_cuda.get("dark_matter_or_halo_inputs") is not False
        or transactional_gravity_cuda.get("redshift_or_cosmology_inputs") is not False
        or transactional_gravity_cuda.get("paid_llm_calls") is not False
        or "device-wide NVML"
        not in transactional_gravity_cuda.get("runtime_measurement", {})
        .get("utilization", {})
        .get("counter_scope", "")
    ):
        raise ValueError("Kastner-Schlatter CUDA consequence campaign is inconsistent")
    transactional_registration_counts = transactional_gravity_observational.get(
        "registration_counts", {}
    )
    if (
        transactional_gravity_observational.get("decision")
        != "blocked_registration_incomplete_observations_sealed"
        or transactional_gravity_observational.get("source_lineage", {}).get(
            "source_intake_content_sha256"
        )
        != transactional_gravity_intake.get("content_sha256")
        or transactional_gravity_observational.get("source_lineage", {}).get(
            "equation_graph_content_sha256"
        )
        != transactional_gravity_graph.get("content_sha256")
        or transactional_gravity_observational.get("source_lineage", {}).get(
            "cuda_consequence_content_sha256"
        )
        != transactional_gravity_cuda.get("content_sha256")
        or transactional_registration_counts.get("total_fields") != 88
        or transactional_registration_counts.get("by_status")
        != {
            "forbidden": 7,
            "missing_required": 58,
            "source_blocked": 4,
            "source_registered": 19,
        }
        or transactional_registration_counts.get("by_lane")
        != {
            "lambda_relation": 10,
            "mond_btfr": 34,
            "sds_clock_acceleration": 22,
            "transaction_poisson": 22,
        }
        or transactional_gravity_observational.get("observational_access_count") != 0
        or transactional_gravity_observational.get("real_data_bundle_count") != 0
        or transactional_gravity_observational.get("real_data_pass_count") != 0
        or transactional_gravity_observational.get("theory_or_ontology_pass_count") != 0
        or transactional_gravity_observational.get("data_seals")
        != {
            "dark_matter_or_halo_inputs_opened": False,
            "observations_opened": False,
            "paid_llm_calls": False,
            "redshift_or_cosmology_inputs_opened": False,
            "synthetic_only": True,
            "transaction_event_observations_opened": False,
        }
        or not transactional_gravity_observational.get("negative_controls")
        or any(
            value is not True
            for value in transactional_gravity_observational["negative_controls"].values()
        )
        or transactional_gravity_observational.get("synthetic_positive_control", {}).get(
            "observational_pass"
        )
        is not False
        or transactional_gravity_observational.get("synthetic_positive_control", {}).get(
            "real_data_eligibility"
        )
        is not False
    ):
        raise ValueError("Kastner-Schlatter observational readiness contract is inconsistent")
    expected_transactional_falsification_counts = {
        "btfr_synthetic_residual_values": 2_097_152,
        "gpu_measured_repetitions": 16_384,
        "gpu_measured_value_evaluations": 103_079_215_104,
        "observational_records_accessed": 0,
        "poisson_synthetic_count_values": 4_194_304,
        "readiness_fields_advanced": 0,
        "scientific_tests_passed": 0,
    }
    if (
        transactional_gravity_falsification.get("decision")
        != "synthetic_falsification_design_controls_closed_observational_lanes_still_blocked"
        or transactional_gravity_falsification.get("source_bindings", {})
        .get("cuda_consequence", {})
        .get("content_sha256")
        != transactional_gravity_cuda.get("content_sha256")
        or transactional_gravity_falsification.get("source_bindings", {})
        .get("observational_readiness", {})
        .get("content_sha256")
        != transactional_gravity_observational.get("content_sha256")
        or transactional_gravity_falsification.get("counts")
        != expected_transactional_falsification_counts
        or transactional_gravity_falsification.get("poisson_power_control", {}).get(
            "empirical_null_rejection_rate"
        )
        != 0.000244140625
        or transactional_gravity_falsification.get("poisson_power_control", {}).get(
            "empirical_alternative_detection_rate"
        )
        != 1.0
        or transactional_gravity_falsification.get("btfr_power_control", {}).get(
            "empirical_null_rejection_rate"
        )
        != 0.0
        or transactional_gravity_falsification.get("btfr_power_control", {}).get(
            "empirical_alternative_detection_rate"
        )
        != 0.999267578125
        or transactional_gravity_falsification.get("btfr_power_control", {}).get(
            "extended_galaxy_geometry_tested"
        )
        is not False
        or transactional_gravity_falsification.get("gpu_cpu_crosscheck", {}).get(
            "all_rejection_decisions_byte_equal"
        )
        is not True
        or transactional_gravity_falsification.get("gpu_cpu_crosscheck", {}).get(
            "maximum_absolute_statistic_error"
        )
        > transactional_gravity_falsification.get("gpu_cpu_crosscheck", {}).get("error_bound")
        or transactional_gravity_falsification.get("observational_bridge", {}).get(
            "registration_fields_advanced"
        )
        != 0
        or transactional_gravity_falsification.get("observational_bridge", {}).get(
            "real_bundle_fields_filled"
        )
        != 0
        or transactional_gravity_falsification.get("synthetic_only") is not True
        or transactional_gravity_falsification.get("observations_opened") is not False
        or transactional_gravity_falsification.get("ontology_pass") is not False
        or transactional_gravity_falsification.get("theory_pass") is not False
        or transactional_gravity_falsification.get("scientific_test_pass") is not False
        or transactional_gravity_falsification.get("dark_matter_or_halo_inputs") is not False
        or transactional_gravity_falsification.get("redshift_or_cosmology_inputs") is not False
        or transactional_gravity_falsification.get("paid_llm_calls") is not False
        or "device-wide NVML"
        not in transactional_gravity_falsification.get("runtime_measurement", {})
        .get("utilization", {})
        .get("counter_scope", "")
    ):
        raise ValueError("Kastner-Schlatter CUDA falsification design is inconsistent")
    expected_candidate_action_counts = {
        "complete_local_deterministic_action_hypotheses": 2,
        "conditional_exact_eq35_branch_matches": 2,
        "normalization_branches": 2,
        "normalization_branches_selected_as_fact": 0,
        "observational_or_theory_passes": 0,
        "paper_derived_actions": 0,
    }
    action_hypotheses = transactional_gravity_candidate_action.get("completion_hypotheses", [])
    if (
        transactional_gravity_candidate_action.get("decision")
        != "candidate_completions_registered_paper_derivation_and_physics_claims_blocked"
        or transactional_gravity_candidate_action.get("first_blocker")
        != "no_paper_derivation_of_candidate_action_or_transaction_intensity_dynamics"
        or transactional_gravity_candidate_action.get("counts") != expected_candidate_action_counts
        or transactional_gravity_candidate_action.get("source_bindings", {})
        .get("intake", {})
        .get("content_sha256")
        != transactional_gravity_intake.get("content_sha256")
        or transactional_gravity_candidate_action.get("source_bindings", {})
        .get("equation_graph", {})
        .get("content_sha256")
        != transactional_gravity_graph.get("content_sha256")
        or transactional_gravity_candidate_action.get("dimensions", {}).get(
            "all_declared_terms_dimensionally_closed"
        )
        is not True
        or len(action_hypotheses) != 2
        or [(row.get("branch_id"), row.get("beta")) for row in action_hypotheses]
        != [("eq35_middle_h", "1/2"), ("eq35_printed_planck", "1/4")]
        or any(
            row.get("paper_authorship_or_derivation") is not False
            or row.get("candidate_action", {}).get("local_deterministic_action_complete")
            is not True
            or row.get("candidate_action", {}).get("stochastic_law_derived_by_action") is not False
            or row.get("conditional_stochastic_completion", {}).get(
                "derived_from_QED_actualization"
            )
            is not False
            or row.get("matching", {}).get("exact_coefficient_match") is not True
            or row.get("matching", {}).get("normalization_selected_as_fact") is not False
            or row.get("noether_bianchi", {}).get("on_shell_covariant_conservation") is not True
            for row in action_hypotheses
        )
        or any(
            value is not False
            for value in transactional_gravity_candidate_action.get("claim_seals", {}).values()
        )
        or any(
            value is not False
            for value in transactional_gravity_candidate_action.get("data_seals", {}).values()
        )
    ):
        raise ValueError("Kastner-Schlatter candidate-action completion is inconsistent")
    if (
        transactional_gravity_equivalence.get("decision")
        != "canonical_dynamic_class_identified_paper_and_physics_claims_blocked"
        or transactional_gravity_equivalence.get("counts")
        != {
            "candidate_action_branches": 2,
            "canonical_dynamic_class_matches": 2,
            "full_action_equalities_to_constant_free_control": 0,
            "new_propagating_gravity_operator_classes": 0,
            "distinct_vacuum_normalization_branches": 2,
            "paper_or_qed_derived_actions": 0,
            "literature_novelty_claims": 0,
            "observational_or_theory_passes": 0,
        }
        or transactional_gravity_equivalence.get("source_bindings", {})
        .get("candidate_action", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or transactional_gravity_equivalence.get("branch_comparison", {}).get(
            "same_propagating_operator_class"
        )
        is not True
        or transactional_gravity_equivalence.get("branch_comparison", {}).get(
            "same_constant_vacuum_energy"
        )
        is not False
        or any(
            item.get("propagating_dynamic_operator_equivalent") is not True
            or item.get("full_action_equal_to_constant_free_control") is not False
            or item.get("paper_or_qed_derivation_inferred") is not False
            or item.get("literature_novelty_inferred") is not False
            for item in transactional_gravity_equivalence.get("equivalence_certificates", [])
        )
        or any(
            value
            for section in ("claim_seals", "data_seals")
            for value in transactional_gravity_equivalence.get(section, {}).values()
        )
    ):
        raise ValueError("Kastner-Schlatter action equivalence audit is inconsistent")
    if (
        transactional_gravity_formal.get("decision")
        != "local_formal_gates_pass_global_boundary_energy_and_full_admission_blocked"
        or transactional_gravity_formal.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_formal.get("formal_counts")
        != {
            "candidate_actions": 2,
            "covariant_variation_pass": 2,
            "formal_admission_pass": 0,
            "gauge_fixed_local_hyperbolicity_pass": 2,
            "ghost_gradient_tachyon_pass": 2,
            "global_positive_energy_pass": 0,
            "paper_or_QED_derived_actions": 0,
            "regular_ADM_Dirac_pass": 2,
            "scalar_Hamiltonian_positive_pass": 2,
            "three_local_DOF_pass": 2,
        }
        or transactional_gravity_formal.get("first_blocker")
        != "global_de_Sitter_boundary_charge_and_nonlinear_positive_energy_theorem_not_registered"
        or transactional_gravity_formal.get("source_bindings", {})
        .get("completion_artifact", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or any(transactional_gravity_formal.get("claim_seals", {}).values())
        or any(transactional_gravity_formal.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter formal admission is inconsistent")
    if (
        transactional_gravity_de_sitter.get("decision")
        != "charge_interface_and_fixed_scalar_energy_pass_coupled_global_energy_blocked"
        or transactional_gravity_de_sitter.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_de_sitter.get("prerequisite_counts")
        != {
            "candidate_actions": 2,
            "closed_slice_empty_boundary_control_pass": 2,
            "covariant_charge_interface_pass": 2,
            "exact_de_Sitter_background_radius_pass": 2,
            "fixed_background_scalar_positive_energy_pass": 2,
            "full_formal_admission_pass": 0,
            "nonlinear_coupled_positive_energy_pass": 0,
            "nontrivial_integrable_coupled_charge_pass": 0,
            "paper_or_QED_derived_actions": 0,
        }
        or transactional_gravity_de_sitter.get("first_blocker")
        != "candidate_bound_de_Sitter_boundary_conditions_zero_symplectic_flux_and_integrable_coupled_charge_not_registered"
        or transactional_gravity_de_sitter.get("source_bindings", {})
        .get("formal_admission_artifact", {})
        .get("content_sha256")
        != transactional_gravity_formal.get("content_sha256")
        or any(transactional_gravity_de_sitter.get("claim_seals", {}).values())
        or any(transactional_gravity_de_sitter.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter de Sitter energy prerequisite is inconsistent")
    expected_poisson_action_counts = {
        "candidate_action_branches": 2,
        "conditional_covariant_point_process_interfaces": 2,
        "stationary_homogeneous_poisson_matches": 2,
        "action_derived_point_process_measures": 0,
        "positive_intensity_preservation_theorems": 0,
        "qed_actualization_derivations": 0,
        "fluctuating_intensity_homogeneous_poisson_closures": 0,
        "observational_or_theory_passes": 0,
    }
    if (
        transactional_gravity_poisson_action.get("decision")
        != "stationary_conditional_poisson_interface_closed_dynamic_derivation_blocked"
        or transactional_gravity_poisson_action.get("counts") != expected_poisson_action_counts
        or transactional_gravity_poisson_action.get("first_blocker")
        != "no_action_derived_covariant_point_process_measure_or_positive_intensity_dynamics"
        or transactional_gravity_poisson_action.get("source_bindings", {})
        .get("candidate_action", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or transactional_gravity_poisson_action.get("exact_mixed_poisson_control", {}).get(
            "Fano_factor"
        )
        != "3/2"
        or transactional_gravity_poisson_action.get("exact_mixed_poisson_control", {}).get(
            "homogeneous_poisson_rejected_for_fluctuating_intensity"
        )
        is not True
        or any(transactional_gravity_poisson_action.get("claim_seals", {}).values())
        or any(transactional_gravity_poisson_action.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter Poisson action compatibility is inconsistent")
    expected_positive_intensity_counts = {
        "action_derived_point_process_measure_pass": 0,
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "exact_crossing_witnesses": 2,
        "fully_coupled_constraint_satisfying_witnesses": 2,
        "paper_QED_ontology_observational_pass": 0,
        "positive_reparameterized_action_pass": 0,
        "restricted_invariant_nonnegative_cone_pass": 0,
        "stationary_conditional_Poisson_interface_pass": 2,
        "unrestricted_positive_intensity_preservation_pass": 0,
        "unrestricted_positive_intensity_preservation_reject": 2,
    }
    if (
        transactional_gravity_positive_intensity.get("decision")
        != "unrestricted_intensity_positivity_rejected_actions_and_stationary_interfaces_blocked"
        or transactional_gravity_positive_intensity.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_positive_intensity.get("gate_counts")
        != expected_positive_intensity_counts
        or transactional_gravity_positive_intensity.get("first_blocker")
        != "no_candidate_bound_positive_intensity_reparameterization_or_proven_invariant_nonnegative_initial_data_cone"
        or transactional_gravity_positive_intensity.get("source_bindings", {})
        .get("candidate_action_completion", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or transactional_gravity_positive_intensity.get("source_bindings", {})
        .get("poisson_action_compatibility", {})
        .get("content_sha256")
        != transactional_gravity_poisson_action.get("content_sha256")
        or len(transactional_gravity_positive_intensity.get("candidate_records", [])) != 2
        or any(
            record.get("decision") != "blocked"
            or record.get("candidate_action_rejection_authorized") is not False
            or record.get("stationary_conditional_interface_preserved") is not True
            or record.get("unrestricted_positive_intensity_preservation") is not False
            or record.get("exact_crossing_witness", {}).get("crossing_exists") is not True
            or record.get("exact_crossing_witness", {}).get("crossing_time_bound")
            != "0<Tau_cross<=q0/v"
            or record.get("exact_crossing_witness", {})
            .get("initial_data", {})
            .get("Hamiltonian_constraint_residual")
            != "0"
            or record.get("exact_crossing_witness", {})
            .get("initial_data", {})
            .get("momentum_constraint_residual")
            != "0"
            for record in transactional_gravity_positive_intensity.get("candidate_records", [])
        )
        or any(transactional_gravity_positive_intensity.get("claim_seals", {}).values())
        or any(transactional_gravity_positive_intensity.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter positive-intensity preservation gate is inconsistent")
    expected_positive_reparameterization_counts = {
        "EL_equivalence_on_positive_sector_pass": 2,
        "action_derived_point_process_measure_pass": 0,
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "exact_positive_field_diffeomorphism_pass": 2,
        "exact_reparameterized_action_pass": 2,
        "original_unrestricted_phase_space_positivity_reject": 2,
        "paper_QED_ontology_observational_pass": 0,
        "paper_or_QED_positive_sector_selection_pass": 0,
        "regular_solution_strict_positivity_pass": 2,
    }
    if (
        transactional_gravity_positive_reparameterization.get("decision")
        != "positive_coordinate_actions_closed_physical_sector_selection_blocked"
        or transactional_gravity_positive_reparameterization.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_positive_reparameterization.get("gate_counts")
        != expected_positive_reparameterization_counts
        or transactional_gravity_positive_reparameterization.get("first_blocker")
        != "no_paper_or_QED_derived_selection_of_the_positive_field_sector_and_no_action_derived_point_process_probability_measure"
        or transactional_gravity_positive_reparameterization.get("source_bindings", {})
        .get("positive_intensity_predecessor", {})
        .get("content_sha256")
        != transactional_gravity_positive_intensity.get("content_sha256")
        or transactional_gravity_positive_reparameterization.get("source_bindings", {})
        .get("candidate_action_completion", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or len(transactional_gravity_positive_reparameterization.get("candidate_records", [])) != 2
        or any(
            record.get("candidate_decision") != "blocked"
            or record.get("candidate_action_rejection_authorized") is not False
            or record.get("field_diffeomorphism", {}).get("map") != "q=q0*exp(phi)"
            or record.get("field_diffeomorphism", {}).get("global_on_declared_open_sector")
            is not True
            or record.get("reparameterized_action", {}).get("exact_EL_equivalence_on_q_positive")
            is not True
            or record.get("positivity_theorem", {}).get(
                "original_q_in_R_nonnegative_cone_invariant"
            )
            is not False
            for record in transactional_gravity_positive_reparameterization.get(
                "candidate_records", []
            )
        )
        or any(transactional_gravity_positive_reparameterization.get("claim_seals", {}).values())
        or any(transactional_gravity_positive_reparameterization.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter positive reparameterization gate is inconsistent")
    expected_point_process_measure_counts = {
        "action_only_Poisson_derivation_pass": 0,
        "action_only_Poisson_derivation_reject": 2,
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "covariant_intensity_measure_pass": 2,
        "exact_covariant_nonidentifiability_witnesses": 2,
        "external_Poisson_postulate_well_posed": 2,
        "minimal_probability_measure_contracts_registered": 1,
        "paper_QED_ontology_observational_pass": 0,
        "paper_or_QED_Poisson_derivation_pass": 0,
    }
    if (
        transactional_gravity_point_process_measure.get("decision")
        != "covariant_intensity_closed_action_only_Poisson_derivation_rejected"
        or transactional_gravity_point_process_measure.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_point_process_measure.get("gate_counts")
        != expected_point_process_measure_counts
        or transactional_gravity_point_process_measure.get("first_blocker")
        != "no_registered_stochastic_generating_functional_or_QED_event_kernel_to_select_Poisson_over_a_covariant_Cox_competitor"
        or transactional_gravity_point_process_measure.get("source_bindings", {})
        .get("positive_reparameterization_predecessor", {})
        .get("content_sha256")
        != transactional_gravity_positive_reparameterization.get("content_sha256")
        or transactional_gravity_point_process_measure.get("source_bindings", {})
        .get("positive_intensity_predecessor", {})
        .get("content_sha256")
        != transactional_gravity_positive_intensity.get("content_sha256")
        or transactional_gravity_point_process_measure.get("exact_nonidentifiability_witness", {})
        .get("exact_separation", {})
        .get("same_first_moment")
        is not True
        or transactional_gravity_point_process_measure.get("deterministic_controls", {}).get(
            "exact_moment_separation_control", {}
        )
        != {
            "Cox_mean": "2",
            "Cox_second_factorial_moment": "5",
            "Cox_variance": "3",
            "Poisson_mean": "2",
            "Poisson_second_factorial_moment": "4",
            "Poisson_variance": "2",
            "epsilon": "1/2",
            "mu": "2",
            "separates_laws_exactly": True,
        }
        or len(transactional_gravity_point_process_measure.get("candidate_records", [])) != 2
        or any(
            record.get("candidate_decision") != "blocked"
            or record.get("candidate_action_rejection_authorized") is not False
            or record.get("covariant_intensity_measure_construction") != "pass"
            or record.get("external_poisson_postulate_is_action_derived") is not False
            for record in transactional_gravity_point_process_measure.get("candidate_records", [])
        )
        or any(transactional_gravity_point_process_measure.get("claim_seals", {}).values())
        or any(transactional_gravity_point_process_measure.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter covariant point-process gate is inconsistent")
    expected_poisson_selector_counts = {
        "Poisson_Laplace_functional_derivation_pass": 0,
        "QED_counting_measure_kernel_derivation_pass": 0,
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "independent_increment_derivation_pass": 0,
        "minimal_sufficient_selector_contracts": 3,
        "paper_QED_ontology_observational_pass": 0,
        "registered_action_derivation_edges_to_PMF": 0,
        "registered_equations_imply_selector_reject": 2,
        "registered_scalar_Poisson_PMF_assertions": 1,
        "registered_selector_nodes": 0,
    }
    poisson_selector_audit = transactional_gravity_poisson_selector.get(
        "registered_dependency_audit", {}
    )
    if (
        transactional_gravity_poisson_selector.get("decision")
        != "Poisson_selector_contract_registered_no_registered_derivation_path"
        or transactional_gravity_poisson_selector.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_poisson_selector.get("gate_counts")
        != expected_poisson_selector_counts
        or transactional_gravity_poisson_selector.get("first_blocker")
        != "no_registered_derivation_of_a_set_indexed_Poisson_Laplace_functional_independent_increment_family_or_QED_counting_measure_kernel"
        or transactional_gravity_poisson_selector.get("source_bindings", {})
        .get("point_process_measure_gate", {})
        .get("content_sha256")
        != transactional_gravity_point_process_measure.get("content_sha256")
        or transactional_gravity_poisson_selector.get("source_bindings", {})
        .get("equation_graph", {})
        .get("content_sha256")
        != transactional_gravity_graph.get("content_sha256")
        or poisson_selector_audit.get("closed_world_counts") != {"edges": 137, "nodes": 54}
        or poisson_selector_audit.get("registered_scalar_Poisson_PMF_nodes") != 1
        or poisson_selector_audit.get("PMF_not_derived_from_action_edge") is not True
        or any(
            poisson_selector_audit.get(key) is not False
            for key in (
                "registered_equations_imply_Poisson_Laplace_functional",
                "registered_equations_imply_QED_counting_measure_kernel",
                "registered_equations_imply_independent_increments",
            )
        )
        or len(transactional_gravity_poisson_selector.get("candidate_records", [])) != 2
        or any(
            record.get("candidate_decision") != "blocked"
            or record.get("candidate_action_rejection_authorized") is not False
            or record.get("registered_action_stochastic_outputs", {}).get(
                "positive_intensity_measure"
            )
            is not True
            or any(
                record.get("registered_action_stochastic_outputs", {}).get(key) is not False
                for key in (
                    "Laplace_functional",
                    "Mecke_or_QED_event_kernel",
                    "independent_increment_family",
                    "probability_kernel",
                )
            )
            for record in transactional_gravity_poisson_selector.get("candidate_records", [])
        )
        or any(transactional_gravity_poisson_selector.get("claim_seals", {}).values())
        or any(transactional_gravity_poisson_selector.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter Poisson selector contract is inconsistent")
    expected_conditional_poisson_counts = {
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "compiler_authored_conditional_kernels": 2,
        "conditional_Laplace_selector_pass": 2,
        "conditional_Mecke_identity_pass": 2,
        "conditional_independent_increment_pass": 2,
        "deterministic_action_derivation_pass": 0,
        "diffeomorphism_covariance_pass": 2,
        "paper_or_QED_actualization_derivation_pass": 0,
        "stationary_Poisson_PMF_recovery_pass": 2,
        "theory_ontology_observational_pass": 0,
    }
    conditional_poisson_contract = transactional_gravity_conditional_poisson.get(
        "conditional_Poisson_kernel_contract", {}
    )
    if (
        transactional_gravity_conditional_poisson.get("decision")
        != "conditional_Poisson_kernel_mathematically_closed_physical_selection_blocked"
        or transactional_gravity_conditional_poisson.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_conditional_poisson.get("gate_counts")
        != expected_conditional_poisson_counts
        or transactional_gravity_conditional_poisson.get("first_blocker")
        != "no_paper_or_QED_derived_actualization_history_to_counting_measure_map_or_principle_selecting_the_compiler_authored_conditional_Poisson_kernel"
        or transactional_gravity_conditional_poisson.get("source_bindings", {})
        .get("Poisson_selector_predecessor", {})
        .get("content_sha256")
        != transactional_gravity_poisson_selector.get("content_sha256")
        or transactional_gravity_conditional_poisson.get("source_bindings", {})
        .get("point_process_measure_gate", {})
        .get("content_sha256")
        != transactional_gravity_point_process_measure.get("content_sha256")
        or conditional_poisson_contract.get("status")
        != "compiler_authored_external_conditional_stochastic_completion"
        or conditional_poisson_contract.get("Laplace_functional", {}).get(
            "uniquely_selects_conditional_Poisson_law"
        )
        is not True
        or conditional_poisson_contract.get("joint_disjoint_set_PGF", {}).get(
            "independent_increment_factorization"
        )
        is not True
        or conditional_poisson_contract.get("Mecke_identity", {}).get(
            "characterizes_same_conditional_Poisson_law"
        )
        is not True
        or conditional_poisson_contract.get("diffeomorphism_covariance", {}).get("pass") is not True
        or len(transactional_gravity_conditional_poisson.get("candidate_records", [])) != 2
        or any(
            record.get("candidate_decision") != "blocked"
            or record.get("compiler_authored_conditional_kernel") is not True
            or record.get("action_derived") is not False
            or record.get("paper_or_QED_derived") is not False
            or record.get("candidate_action_rejection_authorized") is not False
            for record in transactional_gravity_conditional_poisson.get("candidate_records", [])
        )
        or any(transactional_gravity_conditional_poisson.get("claim_seals", {}).values())
        or any(transactional_gravity_conditional_poisson.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter conditional Poisson kernel is inconsistent")
    expected_actualization_history_counts = {
        "blocked_or_absent": 6,
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "closed_by_paper_semantics": 4,
        "compiler_conditional_count_maps": 1,
        "paper_complete_history_to_counting_measure_maps": 0,
        "paper_or_QED_Poisson_kernel_selections": 0,
        "paper_source_clauses_audited": 5,
        "partially_specified": 2,
        "theory_ontology_observational_pass": 0,
        "typed_map_obligations": 12,
    }
    if (
        transactional_gravity_actualization_history.get("decision")
        != "paper_supplies_partial_event_semantics_but_no_typed_history_map_or_kernel_selection"
        or transactional_gravity_actualization_history.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_actualization_history.get("gate_counts")
        != expected_actualization_history_counts
        or transactional_gravity_actualization_history.get("first_blocker")
        != "no_paper_registered_locally_finite_measurable_actualization_history_space_or_set_indexed_counting_map"
        or transactional_gravity_actualization_history.get("source_bindings", {})
        .get("conditional_poisson_kernel", {})
        .get("content_sha256")
        != transactional_gravity_conditional_poisson.get("content_sha256")
        or transactional_gravity_actualization_history.get("source_bindings", {})
        .get("equation_graph", {})
        .get("content_sha256")
        != transactional_gravity_graph.get("content_sha256")
        or len(transactional_gravity_actualization_history.get("typed_map_obligations", [])) != 12
        or transactional_gravity_actualization_history.get(
            "compiler_conditional_count_map", {}
        ).get("attribution")
        != "compiler_authored_formal_completion_not_printed_or_derived_in_the_paper"
        or transactional_gravity_actualization_history.get("compiler_conditional_count_map", {})
        .get("theorem", {})
        .get("countably_additive")
        is not True
        or transactional_gravity_actualization_history.get("compiler_conditional_count_map", {})
        .get("theorem", {})
        .get("locally_finite_by_declared_domain")
        is not True
        or any(transactional_gravity_actualization_history.get("claim_seals", {}).values())
        or any(transactional_gravity_actualization_history.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter actualization history map audit is inconsistent")
    expected_qed_poisson_counts = {
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "compiler_conditional_sufficient_theorems": 1,
        "exact_same_rate_non_Poisson_witnesses": 2,
        "microscopic_derivation_obligations": 12,
        "microscopic_obligations_absent": 10,
        "microscopic_obligations_closed": 0,
        "microscopic_obligations_partial": 2,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "paper_or_QED_channel_kernels_registered": 0,
        "source_evidence_clauses": 5,
        "theory_ontology_observational_pass": 0,
    }
    qed_poisson_theorem = transactional_gravity_qed_poisson_derivation.get(
        "independent_rare_channel_Poisson_limit", {}
    )
    qed_poisson_controls = transactional_gravity_qed_poisson_derivation.get(
        "exact_controls", {}
    )
    if (
        transactional_gravity_qed_poisson_derivation.get("decision")
        != "conditional_rare_channel_theorem_closed_paper_QED_derivation_blocked"
        or transactional_gravity_qed_poisson_derivation.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_qed_poisson_derivation.get("gate_counts")
        != expected_qed_poisson_counts
        or transactional_gravity_qed_poisson_derivation.get("first_blocker")
        != "no_registered_QED_actualization_channel_probability_array_or_predictable_hazard_kernel"
        or transactional_gravity_qed_poisson_derivation.get("source_bindings", {})
        .get("conditional_poisson_kernel", {})
        .get("content_sha256")
        != transactional_gravity_conditional_poisson.get("content_sha256")
        or transactional_gravity_qed_poisson_derivation.get("source_bindings", {})
        .get("operational_event_exposure", {})
        .get("content_sha256")
        != transactional_gravity_observable_exposure.get("content_sha256")
        or transactional_gravity_qed_poisson_derivation.get("source_bindings", {})
        .get("set_indexed_synthetic_campaign", {})
        .get("content_sha256")
        != transactional_gravity_set_indexed_cuda.get("content_sha256")
        or transactional_gravity_qed_poisson_derivation.get("source_bindings", {})
        .get("equation_graph", {})
        .get("content_sha256")
        != transactional_gravity_graph.get("content_sha256")
        or transactional_gravity_qed_poisson_derivation.get("source_bindings", {}).get(
            "primary_pdf_sha256"
        )
        != "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
        or len(
            transactional_gravity_qed_poisson_derivation.get(
                "microscopic_derivation_obligations", []
            )
        )
        != 12
        or qed_poisson_theorem.get("finite_row_joint_PGF")
        != "G_m(z)=product_k[1+sum_i p_mki*(z_i-1)]"
        or qed_poisson_theorem.get("limit_joint_PGF")
        != "exp(sum_i mu_i*(z_i-1))"
        or qed_poisson_theorem.get("paper_or_registered_QED_closes_conditions") is not False
        or qed_poisson_controls.get("paired_cluster_same_rate_no_go", {}).get(
            "limit_variance"
        )
        != "2*mu"
        or qed_poisson_controls.get("paired_cluster_same_rate_no_go", {}).get(
            "Poisson_conclusion_rejected"
        )
        is not True
        or qed_poisson_controls.get("two_cell_common_shock_no_go", {}).get(
            "cross_covariance"
        )
        != "1/2"
        or len(transactional_gravity_qed_poisson_derivation.get("candidate_records", [])) != 2
        or any(
            record.get("candidate_decision") != "blocked"
            or record.get("compiler_conditional_rare_channel_theorem") is not True
            or record.get("paper_or_QED_channel_kernel_registered") is not False
            or record.get("paper_or_QED_Poisson_derivation_pass") is not False
            or record.get("candidate_action_rejection_authorized") is not False
            for record in transactional_gravity_qed_poisson_derivation.get(
                "candidate_records", []
            )
        )
        or transactional_gravity_qed_poisson_derivation.get("synthetic_only") is not True
        or any(transactional_gravity_qed_poisson_derivation.get("claim_seals", {}).values())
        or any(transactional_gravity_qed_poisson_derivation.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter QED Poisson derivation audit is inconsistent")
    expected_compensator_counts = {
        "action_or_QED_compensator_identities": 0,
        "candidate_action_reject": 0,
        "candidate_actions": 2,
        "compiler_compensator_theorem_interfaces": 2,
        "evidence_absent": 8,
        "evidence_closed_by_compiler_hypotheses": 2,
        "evidence_obligations": 10,
        "exact_same_action_alternative_law_witnesses": 2,
        "paper_or_QED_Poisson_derivation_pass": 0,
        "positive_candidate_mean_measures": 2,
        "registered_causal_filtrations": 0,
        "theory_ontology_observational_pass": 0,
    }
    compensator_theorem = transactional_gravity_deterministic_compensator.get(
        "deterministic_compensator_Poisson_characterization", {}
    )
    compensator_controls = transactional_gravity_deterministic_compensator.get(
        "exact_controls", {}
    )
    if (
        transactional_gravity_deterministic_compensator.get("decision")
        != "conditional_compensator_characterization_closed_physical_identity_blocked"
        or transactional_gravity_deterministic_compensator.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or transactional_gravity_deterministic_compensator.get("gate_counts")
        != expected_compensator_counts
        or transactional_gravity_deterministic_compensator.get("first_blocker")
        != "no_registered_QED_probability_space_causal_filtration_or_deterministic_compensator_martingale_identity"
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("qed_actualization_audit", {})
        .get("content_sha256")
        != transactional_gravity_qed_poisson_derivation.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("conditional_poisson_kernel", {})
        .get("content_sha256")
        != transactional_gravity_conditional_poisson.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("positive_reparameterization", {})
        .get("content_sha256")
        != transactional_gravity_positive_reparameterization.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("candidate_action_completion", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("candidate_formal_admission", {})
        .get("content_sha256")
        != transactional_gravity_formal.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("paper_intake", {})
        .get("content_sha256")
        != transactional_gravity_intake.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {})
        .get("equation_graph", {})
        .get("content_sha256")
        != transactional_gravity_graph.get("content_sha256")
        or transactional_gravity_deterministic_compensator.get("source_bindings", {}).get(
            "primary_pdf_sha256"
        )
        != "c2f671293d07b21397e745da00a3ce1a2193c00da647a2ebf4147612b76c1780"
        or len(transactional_gravity_deterministic_compensator.get("evidence_gap_ledger", []))
        != 10
        or compensator_theorem.get("candidate_action_or_paper_supplies_compensator_identity")
        is not False
        or compensator_theorem.get("exponential_martingale")
        != "Z_t(f)=exp(-Integral_{W_t} f*dN+Integral_{W_t}(1-exp(-f))*dmu)"
        or compensator_controls.get("same_action_Poisson_Cox_nonidentifiability", {}).get(
            "same_deterministic_candidate_action"
        )
        is not True
        or compensator_controls.get("same_action_Poisson_Cox_nonidentifiability", {}).get(
            "Cox_variance_on_B"
        )
        != "mu_B+mu_B^2/4"
        or len(transactional_gravity_deterministic_compensator.get("candidate_records", []))
        != 2
        or any(
            record.get("candidate_decision") != "blocked"
            or record.get("positive_mean_measure_on_regular_finite_phi_patch") is not True
            or record.get("compiler_compensator_theorem_interface") != "pass"
            or record.get("action_or_QED_compensator_identity_derived") is not False
            or record.get("candidate_action_rejection_authorized") is not False
            for record in transactional_gravity_deterministic_compensator.get(
                "candidate_records", []
            )
        )
        or any(transactional_gravity_deterministic_compensator.get("claim_seals", {}).values())
        or any(transactional_gravity_deterministic_compensator.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter deterministic compensator gate is inconsistent")
    expected_poisson_cox_power_counts = {
        "finite_sample_evaluation_replicate_tests": 294_912,
        "gpu_generated_count_values": 110_100_480,
        "metric_replicate_values_cpu_gpu_checked": 1_769_472,
        "null_calibration_replicates": 49_152,
        "observational_records_accessed": 0,
        "paper_or_qed_inferences": 0,
        "readiness_fields_advanced": 0,
        "registered_witness_scenario_cells": 12,
        "scenario_cells": 144,
        "scientific_tests_passed": 0,
    }
    poisson_cox_crosscheck = transactional_gravity_poisson_cox_power.get("gpu_cpu_crosscheck", {})
    poisson_cox_runtime = transactional_gravity_poisson_cox_power.get("runtime_measurement", {})
    if (
        transactional_gravity_poisson_cox_power.get("decision")
        != "synthetic_poisson_cox_falsification_power_measured_scientific_and_observational_claims_blocked"
        or transactional_gravity_poisson_cox_power.get("counts")
        != expected_poisson_cox_power_counts
        or transactional_gravity_poisson_cox_power.get("source_bindings", {})
        .get("registered_poisson_cox_witness", {})
        .get("content_sha256")
        != transactional_gravity_poisson_action.get("content_sha256")
        or transactional_gravity_poisson_cox_power.get("registered_witness_exact_sentinel")
        != {
            "conditional_rate_support": ["1", "3"],
            "count_variance": "3",
            "factorial_excess": "1",
            "factorial_second_moment": "5",
            "fano_factor": "3/2",
            "latent_rate_variance": "1",
            "mean_count": "2",
            "poisson_null_factorial_second_moment": "4",
            "void_probability_symbolic": "(exp(-1)+exp(-3))/2",
            "weights": ["1/2", "1/2"],
        }
        or poisson_cox_crosscheck.get("replicate_metric_values_checked") != 1_769_472
        or poisson_cox_crosscheck.get("all_rejection_decisions_byte_equal") is not True
        or poisson_cox_crosscheck.get("exact_registered_witness_sentinel_passed") is not True
        or poisson_cox_crosscheck.get("maximum_absolute_error")
        > poisson_cox_crosscheck.get("absolute_error_bound")
        or poisson_cox_crosscheck.get("maximum_relative_error_scaled_at_one")
        > poisson_cox_crosscheck.get("relative_error_bound")
        or "device-wide NVML"
        not in poisson_cox_runtime.get("utilization", {}).get("counter_scope", "")
        or transactional_gravity_poisson_cox_power.get("synthetic_only") is not True
        or any(
            transactional_gravity_poisson_cox_power.get(key) is not False
            for key in (
                "observational_test_pass",
                "observations_opened",
                "ontology_pass",
                "paper_pass",
                "qed_pass",
                "readiness_advanced",
                "redshift_or_cosmology_inputs",
                "scientific_test_pass",
                "theory_pass",
                "dark_matter_or_halo_inputs",
                "paid_llm_calls",
            )
        )
    ):
        raise ValueError("Kastner-Schlatter Poisson-Cox CUDA power campaign is inconsistent")
    expected_set_indexed_cuda_counts = {
        "cpu_gpu_sentinel_metric_values": 4_608,
        "cpu_gpu_sentinel_pgf_values": 6_144,
        "gpu_generated_unique_count_values": 1_887_436_800,
        "joint_pgf_terms_evaluated": 8_053_063_680,
        "observational_records_accessed": 0,
        "paper_or_qed_inferences": 0,
        "prior_poisson_cox_scenario_cells": 144,
        "projection_multiply_adds": 322_122_547_200,
        "readiness_fields_advanced": 0,
        "scenario_cells": 48,
        "scientific_tests_passed": 0,
    }
    set_indexed_crosscheck = transactional_gravity_set_indexed_cuda.get("gpu_cpu_crosscheck", {})
    set_indexed_runtime = transactional_gravity_set_indexed_cuda.get("runtime_measurement", {})
    if (
        transactional_gravity_set_indexed_cuda.get("decision")
        != "synthetic_set_indexed_independence_falsification_power_measured_all_scientific_claims_blocked"
        or transactional_gravity_set_indexed_cuda.get("counts") != expected_set_indexed_cuda_counts
        or transactional_gravity_set_indexed_cuda.get("source_bindings", {})
        .get("selector_contract", {})
        .get("content_sha256")
        != transactional_gravity_poisson_selector.get("content_sha256")
        or transactional_gravity_set_indexed_cuda.get("source_bindings", {})
        .get("poisson_cox_power", {})
        .get("content_sha256")
        != transactional_gravity_poisson_cox_power.get("content_sha256")
        or transactional_gravity_set_indexed_cuda.get("set_indexed_contract", {}).get(
            "operational_transaction_events_defined"
        )
        is not False
        or set_indexed_crosscheck.get("all_heldout_decisions_byte_equal") is not True
        or set_indexed_crosscheck.get("maximum_absolute_error")
        > set_indexed_crosscheck.get("absolute_error_bound")
        or set_indexed_crosscheck.get("maximum_relative_error_scaled_at_one")
        > set_indexed_crosscheck.get("relative_error_bound")
        or "device-wide" not in set_indexed_runtime.get("scope", "")
        or transactional_gravity_set_indexed_cuda.get("synthetic_only") is not True
        or any(
            transactional_gravity_set_indexed_cuda.get(key) is not False
            for key in (
                "observations_opened",
                "observational_test_pass",
                "ontology_pass",
                "paid_llm_calls",
                "paper_pass",
                "qed_pass",
                "readiness_advanced",
                "redshift_or_cosmology_inputs",
                "scientific_test_pass",
                "theory_pass",
                "dark_matter_or_halo_inputs",
            )
        )
    ):
        raise ValueError("Kastner-Schlatter set-indexed CUDA campaign is inconsistent")
    observable_counts = transactional_gravity_observable_exposure.get("gate_counts", {})
    if (
        transactional_gravity_observable_exposure.get("decision")
        != "operational_operator_specified_but_data_bridge_and_latent_identification_blocked"
        or transactional_gravity_observable_exposure.get("decision_counts")
        != {"blocked": 2, "pass": 0, "reject": 0}
        or observable_counts
        != {
            "candidate_action_reject": 0,
            "candidate_actions": 2,
            "compiler_observation_operator_contracts": 1,
            "exact_nonidentifiability_witnesses": 4,
            "latent_rate_identification_pass": 0,
            "operational_obligations": 11,
            "operational_obligations_missing": 10,
            "operational_obligations_registered": 0,
            "operational_obligations_sealed": 1,
            "real_observation_bundles": 0,
            "theory_ontology_observational_pass": 0,
        }
        or transactional_gravity_observable_exposure.get("source_bindings", {})
        .get("actualization_history_map", {})
        .get("content_sha256")
        != transactional_gravity_actualization_history.get("content_sha256")
        or transactional_gravity_observable_exposure.get("first_blocker")
        != "no_registered_detector_level_transaction_event_schema_exposure_response_background_or_calibration_manifest"
        or len(transactional_gravity_observable_exposure.get("operational_obligation_ledger", []))
        != 11
        or transactional_gravity_observable_exposure.get("identifiability_theorem", {}).get(
            "current_contract_satisfies_conditions"
        )
        is not False
        or transactional_gravity_observable_exposure.get("synthetic_only") is not True
        or any(transactional_gravity_observable_exposure.get("claim_seals", {}).values())
        or any(transactional_gravity_observable_exposure.get("data_seals", {}).values())
    ):
        raise ValueError("Kastner-Schlatter operational exposure gate is inconsistent")
    scheduler_contract = transactional_gravity_gpu_scheduler_adapter.get("scheduler_contract", {})
    continuous_gpu_contract = transactional_gravity_gpu_scheduler_adapter.get(
        "continuous_service_contract", {}
    )
    if (
        transactional_gravity_gpu_scheduler_adapter.get("decision")
        != "durable_single_owner_gpu_continuous_service_ready_start_gated_not_started"
        or scheduler_contract.get("coordinator") != "PersistentParallelSearch"
        or scheduler_contract.get("supervisor") != "PersistentParallelSupervisor"
        or scheduler_contract.get("gpu_owner_count") != 1
        or scheduler_contract.get("cpu_worker_count") != 0
        or scheduler_contract.get("maximum_queue_items") != 1
        or scheduler_contract.get("maximum_attempts") != 3
        or scheduler_contract.get("process_restart_budget") != 2
        or scheduler_contract.get("arbitrary_callable_or_subprocess_surface") is not False
        or continuous_gpu_contract.get("service_epoch")
        != "kastner-schlatter-set-indexed-gpu-service-20260811-v1"
        or continuous_gpu_contract.get("exclusive_pid_argv_lease") != "service.lease.json"
        or continuous_gpu_contract.get("atomic_checkpoint") != "service-checkpoint.json"
        or continuous_gpu_contract.get("external_stop_request") != "stop.request"
        or continuous_gpu_contract.get("maximum_service_cycles") != 241_920
        or continuous_gpu_contract.get("supervisor_slice_seconds") != 120
        or continuous_gpu_contract.get("foreground_only_no_detached_launcher") is not True
        or continuous_gpu_contract.get("idempotent_queue_resume_each_cycle") is not True
        or continuous_gpu_contract.get("runtime_outputs_gitignored") is not True
        or continuous_gpu_contract.get("stale_lease_recovery_requires_owner_nonmatch") is not True
        or continuous_gpu_contract.get("gpu_start_gate")
        != {
            "fails_closed_if_nvml_unavailable": True,
            "maximum_device_wide_utilization_percent": 20,
            "minimum_free_memory_mib": 8192,
        }
        or transactional_gravity_gpu_scheduler_adapter.get("source_bindings", {})
        .get("workload_source", {})
        .get("file_sha256")
        != "4f66e8a19025f75156a92f5772fc5be6cb5046dddd4c962e5532528e055935fb"
        or transactional_gravity_gpu_scheduler_adapter.get("source_bindings", {})
        .get("source", {})
        .get("file_sha256")
        != "75e46510e40e71a4c18ad78c8a2eedcdaf43a957421b97977cb94574fba927e7"
        or transactional_gravity_gpu_scheduler_adapter.get("source_bindings", {})
        .get("gitignore", {})
        .get("file_sha256")
        != "33d2ada3d63a31cb62cbfbc1ac26b7d08b7b7b23419c7cac8271e1b50c47d289"
        or any(transactional_gravity_gpu_scheduler_adapter.get("execution_state", {}).values())
        or any(transactional_gravity_gpu_scheduler_adapter.get("seals", {}).values())
        or transactional_gravity_gpu_scheduler_adapter.get("scientific_test_pass") is not False
        or transactional_gravity_gpu_scheduler_adapter.get("readiness_advanced") is not False
    ):
        raise ValueError("Kastner-Schlatter durable GPU scheduler adapter is inconsistent")
    deferred_gpu_contract = transactional_gravity_deferred_gpu_ownership.get(
        "ownership_contract", {}
    )
    deferred_gpu_runtime = transactional_gravity_deferred_gpu_ownership.get(
        "current_runtime_audit", {}
    )
    deferred_gpu_sample = deferred_gpu_runtime.get("nvml_sample", {})
    if (
        transactional_gravity_deferred_gpu_ownership.get("decision")
        != "deferred_gpu_ownership_ready_current_device_occupied_not_started"
        or deferred_gpu_contract
        != {
            "detached_launch_or_process_signal_surface": False,
            "exclusive_pid_argv_waiter_and_owner_lease": "deferred-gpu-owner.lease.json",
            "handoff_to_scheduler_automatic": False,
            "maximum_gpu_utilization_percent": 20,
            "maximum_polls": 721,
            "maximum_wait_seconds": 3600,
            "minimum_free_gpu_memory_mib": 8192,
            "poll_backend": "NVML only; no CUDA context",
            "poll_interval_seconds": 5,
            "required_consecutive_safe_samples": 3,
            "sqlite_surface": False,
            "stale_lease_recovery_requires_owner_nonmatch": True,
        }
        or deferred_gpu_runtime.get("runtime_directory_exists") is not False
        or deferred_gpu_runtime.get("lease_exists") is not False
        or deferred_gpu_runtime.get("checkpoint_exists") is not False
        or deferred_gpu_runtime.get("single_sample_safe") is not False
        or deferred_gpu_runtime.get("ownership_reservable_now") is not False
        or deferred_gpu_sample.get("device_name") != "NVIDIA GeForce RTX 5090"
        or deferred_gpu_sample.get("gpu_utilization_percent") != 99
        or deferred_gpu_sample.get("memory_free_mib") != 8083
        or "device-wide" not in deferred_gpu_sample.get("scope", "")
        or transactional_gravity_deferred_gpu_ownership.get("source_bindings", {})
        .get("scheduler_readiness", {})
        .get("content_sha256")
        != transactional_gravity_gpu_scheduler_adapter.get("content_sha256")
        or transactional_gravity_deferred_gpu_ownership.get("source_bindings", {})
        .get("scheduler_source", {})
        .get("file_sha256")
        != "75e46510e40e71a4c18ad78c8a2eedcdaf43a957421b97977cb94574fba927e7"
        or any(transactional_gravity_deferred_gpu_ownership.get("execution_state", {}).values())
        or any(transactional_gravity_deferred_gpu_ownership.get("seals", {}).values())
        or transactional_gravity_deferred_gpu_ownership.get("observations_opened") is not False
        or transactional_gravity_deferred_gpu_ownership.get("readiness_advanced") is not False
        or transactional_gravity_deferred_gpu_ownership.get("scientific_test_pass") is not False
    ):
        raise ValueError("Kastner-Schlatter deferred GPU ownership gate is inconsistent")
    expected_scalar_cuda_counts = {
        "compiler_action_hypotheses": 2,
        "exact_sentinel_groups": 4,
        "gpu_kernel_dispatches": 32_772,
        "gpu_measured_parameter_sample_pairs": 68_719_476_736,
        "gpu_measured_repetitions": 32_768,
        "gpu_measured_scalar_consequence_evaluations": 137_438_953_472,
        "gpu_warmup_repetitions": 4,
        "negative_parameter_controls": 5,
        "observational_records_accessed": 0,
        "paper_qed_or_theory_passes": 0,
        "parameter_cases": 2_048,
        "spectral_and_radial_samples_per_parameter": 1_024,
        "unique_parameter_sample_pairs": 2_097_152,
        "unique_scalar_consequence_values": 4_194_304,
    }
    if (
        transactional_gravity_scalar_cuda.get("decision")
        != "compiler_hypothesis_scalar_controls_closed_paper_qed_and_theory_claims_blocked"
        or transactional_gravity_scalar_cuda.get("counts") != expected_scalar_cuda_counts
        or transactional_gravity_scalar_cuda.get("source_bindings", {})
        .get("candidate_action_completion", {})
        .get("content_sha256")
        != transactional_gravity_candidate_action.get("content_sha256")
        or transactional_gravity_scalar_cuda.get("gpu_cpu_crosscheck", {}).get(
            "maximum_absolute_error"
        )
        > transactional_gravity_scalar_cuda.get("gpu_cpu_crosscheck", {}).get(
            "absolute_error_bound"
        )
        or transactional_gravity_scalar_cuda.get("gpu_cpu_crosscheck", {}).get(
            "maximum_relative_error"
        )
        > transactional_gravity_scalar_cuda.get("gpu_cpu_crosscheck", {}).get(
            "relative_error_bound"
        )
        or transactional_gravity_scalar_cuda.get("dispersion_control", {}).get(
            "maximum_gpu_relative_equation_residual"
        )
        > transactional_gravity_scalar_cuda.get("gpu_cpu_crosscheck", {}).get(
            "dispersion_relative_residual_bound"
        )
        or "device-wide NVML"
        not in transactional_gravity_scalar_cuda.get("runtime_measurement", {})
        .get("utilization", {})
        .get("counter_scope", "")
        or transactional_gravity_scalar_cuda.get("synthetic_only") is not True
        or any(
            transactional_gravity_scalar_cuda.get(key) is not False
            for key in (
                "observations_opened",
                "ontology_pass",
                "paper_or_qed_pass",
                "theory_pass",
                "dark_matter_or_halo_inputs",
                "redshift_or_cosmology_inputs",
                "paid_llm_calls",
            )
        )
    ):
        raise ValueError("Kastner-Schlatter scalar CUDA campaign is inconsistent")
    expected_extended_geometry_counts = {
        "evaluation_radii_per_case": 2_048,
        "extended_source_laws_registered": 0,
        "geometry_resolution_cases": 20,
        "gpu_kernel_dispatches": 1_320,
        "gpu_measured_repetitions": 64,
        "gpu_measured_source_evaluation_interactions": 2_860_515_328,
        "gpu_warmup_repetitions": 2,
        "lensing_cases_executed": 0,
        "observational_records_accessed": 0,
        "physical_or_theory_passes": 0,
        "source_resolutions": 5,
        "synthetic_geometry_classes": 4,
        "unique_source_evaluation_interactions": 44_695_552,
    }
    extended_hypotheses = transactional_gravity_extended_geometry.get("completion_hypotheses", {})
    local_superposition = extended_hypotheses.get("H_local_superposition", {})
    if (
        transactional_gravity_extended_geometry.get("decision")
        != "no_source_supported_extended_completion_local_superposition_rejected_enclosed_mass_blocked"
        or transactional_gravity_extended_geometry.get("counts")
        != expected_extended_geometry_counts
        or transactional_gravity_extended_geometry.get("source_bindings", {})
        .get("equation_graph", {})
        .get("content_sha256")
        != transactional_gravity_graph.get("content_sha256")
        or transactional_gravity_extended_geometry.get("source_bindings", {})
        .get("cuda_falsification_design", {})
        .get("content_sha256")
        != transactional_gravity_falsification.get("content_sha256")
        or transactional_gravity_extended_geometry.get("source_bindings", {})
        .get("observational_readiness", {})
        .get("content_sha256")
        != transactional_gravity_observational.get("content_sha256")
        or transactional_gravity_extended_geometry.get("paper_boundary", {}).get(
            "extended_source_operator_registered"
        )
        is not False
        or transactional_gravity_extended_geometry.get("paper_boundary", {}).get(
            "covariant_extended_metric_registered"
        )
        is not False
        or transactional_gravity_extended_geometry.get("paper_boundary", {}).get(
            "lensing_deflection_operator_registered"
        )
        is not False
        or extended_hypotheses.get("H_enclosed_mass", {}).get("decision")
        != "blocked_not_a_registered_extended_source_law"
        or extended_hypotheses.get("H_enclosed_mass", {}).get("geometry_blind_shell_ring_control")
        != "fails_to_distinguish_thin_shell_from_thin_ring_with_same_radial_mass_support"
        or local_superposition.get("decision")
        != "hypothesis_rejected_by_exact_splitting_and_pair_balance_controls"
        or local_superposition.get("point_mass_aggregation_invariant") is not False
        or [
            row.get("exact_local_superposition_coincident_split_ratio")
            for row in local_superposition.get("coincident_split_controls", [])
        ]
        != [4.0, 8.0, 16.0, 32.0, 64.0]
        or local_superposition.get("unequal_pair_matter_force_control", {}).get(
            "action_reaction_balance"
        )
        is not False
        or local_superposition.get("unequal_pair_matter_force_control", {}).get(
            "exact_net_matter_force"
        )
        != "(3-sqrt(3))/8"
        or transactional_gravity_extended_geometry.get("lensing_rotation_consistency_gate", {}).get(
            "executed"
        )
        is not False
        or transactional_gravity_extended_geometry.get("lensing_rotation_consistency_gate", {}).get(
            "first_missing_field"
        )
        != "source_supported_covariant_extended_metric_and_null_geodesic_deflection_operator"
        or transactional_gravity_extended_geometry.get("gpu_cpu_crosscheck", {}).get(
            "maximum_absolute_component_error"
        )
        > transactional_gravity_extended_geometry.get("gpu_cpu_crosscheck", {}).get(
            "absolute_error_bound"
        )
        or transactional_gravity_extended_geometry.get("gpu_cpu_crosscheck", {}).get(
            "maximum_far_coefficient_relative_error_to_sqrt_N"
        )
        > transactional_gravity_extended_geometry.get("gpu_cpu_crosscheck", {}).get(
            "far_coefficient_relative_error_bound"
        )
        or "device-wide NVML"
        not in transactional_gravity_extended_geometry.get("runtime_measurement", {})
        .get("utilization", {})
        .get("counter_scope", "")
        or transactional_gravity_extended_geometry.get("synthetic_only") is not True
        or transactional_gravity_extended_geometry.get("observations_opened") is not False
        or transactional_gravity_extended_geometry.get("ontology_pass") is not False
        or transactional_gravity_extended_geometry.get("theory_pass") is not False
        or transactional_gravity_extended_geometry.get("physical_pass") is not False
        or transactional_gravity_extended_geometry.get("dark_matter_or_halo_inputs") is not False
        or transactional_gravity_extended_geometry.get("redshift_or_cosmology_inputs") is not False
        or transactional_gravity_extended_geometry.get("paid_llm_calls") is not False
    ):
        raise ValueError("Kastner-Schlatter extended-geometry CUDA stress is inconsistent")
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
        or future_aether_fixed_free_data_principal_gate.get("decision_counts") != {"blocked": 14}
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
        or future_aether_fixed_free_data_principal_gate.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_fixed_free_data_principal_gate.get("observational_data_opened")
        is not False
    ):
        raise ValueError("future Aether fixed-free-data principal gate is inconsistent")
    if (
        future_aether_finite_tilt_york_symbol_gate.get("candidate_count") != 14
        or future_aether_finite_tilt_york_symbol_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_finite_tilt_york_symbol_gate.get(
            "finite_tilt_metric_York_symbol_derived_count"
        )
        != 3
        or future_aether_finite_tilt_york_symbol_gate.get(
            "uniform_fixed_free_data_principal_ellipticity_pass_count"
        )
        != 1
        or future_aether_finite_tilt_york_symbol_gate.get("exact_nonelliptic_York_shell_count") != 2
        or future_aether_finite_tilt_york_symbol_gate.get("York_ansatz_reject_count") != 2
        or future_aether_finite_tilt_york_symbol_gate.get(
            "weighted_Fredholm_isomorphism_pass_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "lower_order_coefficient_bound_pass_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get("computable_full_inverse_norm_count") != 0
        or future_aether_finite_tilt_york_symbol_gate.get("nonlinear_remainder_bound_pass_count")
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_finite_tilt_york_symbol_gate.get("formal_pass_count") != 0
        or future_aether_finite_tilt_york_symbol_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "candidate_bound_weighted_Fredholm_isomorphism_lower_order_coefficient_and_inverse_norm_bounds_for_finite_tilt_York_operator": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_finite_tilt_york_symbol_gate.get("source_fixed_free_data_binding", {}).get(
            "content_sha256"
        )
        != future_aether_fixed_free_data_principal_gate.get("content_sha256")
        or future_aether_finite_tilt_york_symbol_gate.get("observational_data_opened") is not False
        or future_aether_finite_tilt_york_symbol_gate.get("dark_matter_or_halo_inputs") is not False
        or future_aether_finite_tilt_york_symbol_gate.get("redshift_distance_inputs") is not False
        or future_aether_finite_tilt_york_symbol_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_finite_tilt_york_symbol_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether finite-tilt York symbol gate is inconsistent")
    if (
        future_aether_principal_inverse_fredholm_gate.get("candidate_count") != 14
        or future_aether_principal_inverse_fredholm_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_principal_inverse_fredholm_gate.get("uniformly_elliptic_candidate_count")
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
        or future_aether_principal_inverse_fredholm_gate.get("nonlinear_remainder_bound_pass_count")
        != 0
        or future_aether_principal_inverse_fredholm_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_principal_inverse_fredholm_gate.get("formal_pass_count") != 0
        or future_aether_principal_inverse_fredholm_gate.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_principal_inverse_fredholm_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "candidate_bound_spatially_distributed_lower_order_linearized_constraint_coefficient_registry_on_weighted_spaces": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_principal_inverse_fredholm_gate.get("source_York_symbol_binding", {}).get(
            "content_sha256"
        )
        != future_aether_finite_tilt_york_symbol_gate.get("content_sha256")
        or future_aether_principal_inverse_fredholm_gate.get("observational_data_opened")
        is not False
        or future_aether_principal_inverse_fredholm_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_principal_inverse_fredholm_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_principal_inverse_fredholm_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_principal_inverse_fredholm_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether principal-inverse Fredholm gate is inconsistent")
    if (
        future_aether_lower_order_coefficient_contract_gate.get("candidate_count") != 14
        or future_aether_lower_order_coefficient_contract_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_lower_order_coefficient_contract_gate.get(
            "bounded_lower_order_coefficient_contract_gate_completed"
        )
        is not True
        or future_aether_lower_order_coefficient_contract_gate.get(
            "forced_characteristic_candidate_count"
        )
        != 11
        or future_aether_lower_order_coefficient_contract_gate.get(
            "York_symbol_shell_candidate_count"
        )
        != 2
        or future_aether_lower_order_coefficient_contract_gate.get(
            "uniformly_elliptic_candidate_count"
        )
        != 1
        or future_aether_lower_order_coefficient_contract_gate.get(
            "compact_profile_C3_weighted_jet_bound_pass_count"
        )
        != 1
        or future_aether_lower_order_coefficient_contract_gate.get(
            "lower_order_coefficient_contract_declared_count"
        )
        != 1
        or future_aether_lower_order_coefficient_contract_gate.get(
            "full_canonical_background_point_registered_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "distributed_lower_order_coefficient_registry_complete_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "weighted_relative_lower_order_bound_pass_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "weighted_Fredholm_isomorphism_pass_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "full_operator_inverse_norm_pass_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "nonlinear_remainder_bound_pass_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get("formal_pass_count") != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_lower_order_coefficient_contract_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "candidate_bound_full_canonical_seed_point_including_pi_and_p_A_and_distributed_H_D_coefficient_DAG": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_lower_order_coefficient_contract_gate.get(
            "source_principal_inverse_binding", {}
        ).get("content_sha256")
        != future_aether_principal_inverse_fredholm_gate.get("content_sha256")
        or future_aether_lower_order_coefficient_contract_gate.get(
            "source_compact_seed_binding", {}
        ).get("content_sha256")
        != future_aether_finite_amplitude_negative_seed_gate.get("content_sha256")
        or future_aether_lower_order_coefficient_contract_gate.get(
            "automatic_downstream_enqueue_performed"
        )
        is not False
        or future_aether_lower_order_coefficient_contract_gate.get("observational_data_opened")
        is not False
        or future_aether_lower_order_coefficient_contract_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_lower_order_coefficient_contract_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_lower_order_coefficient_contract_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_lower_order_coefficient_contract_gate.get("candidate_records", []))
        != 14
    ):
        raise ValueError("future Aether lower-order coefficient-contract gate is inconsistent")
    if (
        future_aether_canonical_seed_constraint_dag_gate.get("candidate_count") != 14
        or future_aether_canonical_seed_constraint_dag_gate.get("decision_counts")
        != {"blocked": 14}
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "bounded_canonical_seed_constraint_DAG_gate_completed"
        )
        is not True
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "forced_characteristic_candidate_count"
        )
        != 11
        or future_aether_canonical_seed_constraint_dag_gate.get("York_symbol_shell_candidate_count")
        != 2
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "uniformly_elliptic_candidate_count"
        )
        != 1
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "full_canonical_background_point_registered_count"
        )
        != 1
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "candidate_bound_flat_chart_D_residual_DAG_registered_count"
        )
        != 1
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "spatially_distributed_canonical_H_core_registered_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "metric_covariantized_H_D_Frechet_DAG_registered_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "distributed_lower_order_coefficient_registry_complete_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "weighted_relative_lower_order_bound_pass_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "weighted_Fredholm_isomorphism_pass_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "full_operator_inverse_norm_pass_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "nonlinear_remainder_bound_pass_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "completed_boundary_sign_persistence_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get("formal_pass_count") != 0
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "candidate_rejection_authorized_count"
        )
        != 0
        or future_aether_canonical_seed_constraint_dag_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "candidate_bound_spatially_distributed_canonical_H_core_and_metric_covariantized_H_D_Frechet_DAG_off_flat_seed_chart": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "source_lower_order_binding", {}
        ).get("content_sha256")
        != future_aether_lower_order_coefficient_contract_gate.get("content_sha256")
        or future_aether_canonical_seed_constraint_dag_gate.get(
            "automatic_downstream_enqueue_performed"
        )
        is not False
        or future_aether_canonical_seed_constraint_dag_gate.get("observational_data_opened")
        is not False
        or future_aether_canonical_seed_constraint_dag_gate.get("dark_matter_or_halo_inputs")
        is not False
        or future_aether_canonical_seed_constraint_dag_gate.get("redshift_distance_inputs")
        is not False
        or future_aether_canonical_seed_constraint_dag_gate.get("paid_llm_spend_usd") != 0.0
        or len(future_aether_canonical_seed_constraint_dag_gate.get("candidate_records", [])) != 14
    ):
        raise ValueError("future Aether canonical-seed constraint DAG gate is inconsistent")
    aether_shell_records = future_aether_characteristic_shell_hcore_gate.get(
        "candidate_records", []
    )
    aether_shell_target = next(
        (
            record
            for record in aether_shell_records
            if record.get("candidate_id") == "G3A-5e9f93eda83935f288c19571"
        ),
        None,
    )
    aether_shell_control = (
        aether_shell_target.get("characteristic_shell_H_core_certificate", {}).get(
            "characteristic_shell_control", {}
        )
        if aether_shell_target
        else {}
    )
    if (
        future_aether_characteristic_shell_hcore_gate.get("candidate_count") != 14
        or future_aether_characteristic_shell_hcore_gate.get("decision_counts") != {"blocked": 14}
        or future_aether_characteristic_shell_hcore_gate.get("first_blocker_counts")
        != {
            "alternative_canonical_momentum_variable_or_gauge_avoiding_exact_finite_tilt_York_symbol_shell": 2,
            "declared_compact_seed_crosses_candidate_bound_Legendre_characteristic_shell_F2_eq31": 1,
            "noncharacteristic_foliation_or_compact_negative_seed_avoiding_forced_ADM_Legendre_characteristic_crossing": 11,
        }
        or future_aether_characteristic_shell_hcore_gate.get(
            "regular_stratum_flat_chart_H_core_contract_registered_count"
        )
        != 1
        or future_aether_characteristic_shell_hcore_gate.get(
            "declared_profile_global_flat_chart_H_core_registered_count"
        )
        != 0
        or future_aether_characteristic_shell_hcore_gate.get(
            "off_flat_metric_covariantization_registered_count"
        )
        != 0
        or future_aether_characteristic_shell_hcore_gate.get(
            "metric_covariantized_H_D_Frechet_DAG_registered_count"
        )
        != 0
        or future_aether_characteristic_shell_hcore_gate.get("formal_pass_count") != 0
        or future_aether_characteristic_shell_hcore_gate.get("candidate_rejection_authorized_count")
        != 0
        or future_aether_characteristic_shell_hcore_gate.get(
            "constraint_satisfying_negative_total_energy_datum_count"
        )
        != 0
        or future_aether_characteristic_shell_hcore_gate.get("source_lineage", {}).get(
            "canonical_artifact_content_sha256"
        )
        != future_aether_canonical_seed_constraint_dag_gate.get("content_sha256")
        or len(aether_shell_records) != 14
        or aether_shell_target is None
        or aether_shell_target.get("first_blocker")
        != "declared_compact_seed_crosses_candidate_bound_Legendre_characteristic_shell_F2_eq31"
        or aether_shell_target.get("decision") != "blocked"
        or aether_shell_target.get("formal_pass") is not False
        or aether_shell_target.get("candidate_rejection_authorized") is not False
        or aether_shell_control.get("only_real_characteristic_condition") != "F**2=31"
        or aether_shell_control.get("hessian_determinant")
        != "-31*(F**2 - 31)**2*(33*F**2 + 65)*(61*F**2 + 124)**2/(8796093022208*(F**2 + 1))"
        or aether_shell_control.get("declared_profile_characteristic_shell", {}).get("hessian_rank")
        != 7
        or aether_shell_control.get("declared_profile_characteristic_shell", {}).get(
            "hessian_nullity"
        )
        != 2
        or aether_shell_control.get("seed_legendre_image", {}).get(
            "shell_primary_compatibility_residuals"
        )
        != ["0", "0"]
        or aether_shell_control.get("incompatible_momentum_negative_control", {}).get(
            "null_projection_residuals"
        )
        != ["0", "1"]
        or aether_shell_control.get("incompatible_momentum_negative_control", {}).get("rejected")
        is not True
        or aether_shell_control.get("noncrossing_profile_control", {}).get(
            "distance_to_characteristic_F_squared"
        )
        != "6"
        or aether_shell_control.get("regular_stratum_H_core", {}).get("registered") is not True
        or aether_shell_control.get("regular_stratum_H_core", {}).get("global_on_declared_profile")
        is not False
        or future_aether_characteristic_shell_hcore_gate.get(
            "automatic_downstream_enqueue_performed"
        )
        is not False
        or future_aether_characteristic_shell_hcore_gate.get("observational_data_opened")
        is not False
        or future_aether_characteristic_shell_hcore_gate.get("data_eligibility")
        != {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_calls": False,
        }
    ):
        raise ValueError("future Aether characteristic-shell H_core gate is inconsistent")
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
        or future_g3_general_geometry_curvature_shortfall.get("decision_counts") != {"blocked": 3}
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
        or future_g3_general_geometry_curvature_shortfall.get("first_blocker_counts")
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
        or len(future_g3_general_geometry_curvature_shortfall.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 general-geometry curvature shortfall gate is inconsistent")
    if (
        future_g3_general_geometry_surplus_mismatch.get("candidate_count") != 3
        or future_g3_general_geometry_surplus_mismatch.get("decision_counts") != {"blocked": 3}
        or future_g3_general_geometry_surplus_mismatch.get("exact_surplus_identity_pass_count") != 3
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
        or future_g3_general_geometry_surplus_mismatch.get("observational_data_opened") is not False
        or future_g3_general_geometry_surplus_mismatch.get("dark_matter_or_halo_inputs")
        is not False
        or future_g3_general_geometry_surplus_mismatch.get("redshift_distance_inputs") is not False
        or future_g3_general_geometry_surplus_mismatch.get("paid_llm_spend_usd") != 0.0
        or len(future_g3_general_geometry_surplus_mismatch.get("candidate_records", [])) != 3
    ):
        raise ValueError("future G3 general-geometry surplus mismatch gate is inconsistent")
    if (
        future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("candidate_count") != 3
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("decision_counts")
        != {"blocked": 3}
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "radial_momentum_leading_order_pass_count"
        )
        != 3
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "flat_Hamiltonian_leading_order_pass_count"
        )
        != 3
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "joint_real_asymptotic_coefficient_solution_count"
        )
        != 0
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "flat_radial_matched_constraint_class_reject_count"
        )
        != 3
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "registered_AF_metric_York_datum_pass_count"
        )
        != 0
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
        )
        != 0
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("theory_reject_count")
        != 0
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "global_hamiltonian_energy_pass_count"
        )
        != 0
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("full_formal_pass_count")
        != 0
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("first_blocker_counts")
        != {
            "candidate_specific_AF_metric_York_data_beyond_flat_radial_r_minus_2_asymptotic_class": 3
        }
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("source_bindings", {})
        .get("predecessor", {})
        .get("content_sha256")
        != future_g3_general_geometry_surplus_mismatch.get("content_sha256")
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("synthetic_fixture_role")
        != "deterministic_symbolic_negative_controls_only"
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "observational_data_opened"
        )
        is not False
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "dark_matter_or_halo_inputs"
        )
        is not False
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get(
            "redshift_distance_inputs"
        )
        is not False
        or future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("paid_llm_spend_usd")
        != 0.0
        or len(
            future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("candidate_records", [])
        )
        != 3
    ):
        raise ValueError("future G3 flat-radial matched-constraint no-go is inconsistent")
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
        != future_aether_characteristic_shell_hcore_gate.get("content_sha256")
        or future_candidate_action_dossier.get("source_roots", {}).get("g3_followup_content_sha256")
        != future_g3_flat_radial_matched_constraints_asymptotic_no_go.get("content_sha256")
        or len(future_candidate_action_dossier.get("dossiers", [])) != 19
        or next(
            record
            for record in future_candidate_action_dossier.get("dossiers", [])
            if record.get("candidate_id") == "G3A-5e9f93eda83935f288c19571"
        ).get("first_blocker")
        != "declared_compact_seed_crosses_candidate_bound_Legendre_characteristic_shell_F2_eq31"
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
    fourth_chunk_counts_128 = quartic_tc2_fourth_jet_chunk_128.get("counts", {})
    fourth_chunk_counts_160 = quartic_tc2_fourth_jet_chunk_160.get("counts", {})
    fourth_chunk_counts_192 = quartic_tc2_fourth_jet_chunk_192.get("counts", {})
    fourth_chunk_counts_224 = quartic_tc2_fourth_jet_chunk_224.get("counts", {})
    fourth_obstruction = quartic_tc2_fourth_jet_chunk_224.get("first_exact_obstruction", {})
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
        or quartic_tc2_fourth_jet_chunk_32.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_0.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_64.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_64.get("selected") != 32
        or fourth_chunk_counts_64.get("candidate_solvable") != 384
        or fourth_chunk_counts_64.get("candidate_obstructed") != 0
        or fourth_chunk_counts_64.get("fourth_obligations_remaining") != 2_964
        or fourth_chunk_counts_64.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_64.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_32.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_96.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_96.get("selected") != 32
        or fourth_chunk_counts_96.get("candidate_solvable") != 384
        or fourth_chunk_counts_96.get("candidate_obstructed") != 0
        or fourth_chunk_counts_96.get("fourth_obligations_remaining") != 2_932
        or fourth_chunk_counts_96.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_96.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_64.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_128.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_128.get("selected") != 32
        or fourth_chunk_counts_128.get("candidate_solvable") != 384
        or fourth_chunk_counts_128.get("candidate_obstructed") != 0
        or fourth_chunk_counts_128.get("fourth_obligations_remaining") != 2_900
        or fourth_chunk_counts_128.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_128.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_96.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_160.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_160.get("selected") != 32
        or fourth_chunk_counts_160.get("candidate_solvable") != 384
        or fourth_chunk_counts_160.get("candidate_obstructed") != 0
        or fourth_chunk_counts_160.get("fourth_obligations_remaining") != 2_868
        or fourth_chunk_counts_160.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_160.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_128.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_192.get("status")
        != "pass_exact_fourth_jet_chunk_32_tube_fail_closed"
        or fourth_chunk_counts_192.get("selected") != 32
        or fourth_chunk_counts_192.get("candidate_solvable") != 384
        or fourth_chunk_counts_192.get("candidate_obstructed") != 0
        or fourth_chunk_counts_192.get("directional_evaluations") != 456
        or fourth_chunk_counts_192.get("fourth_obligations_remaining") != 2_836
        or fourth_chunk_counts_192.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_192.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_160.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_224.get("status")
        != "stop_first_exact_fourth_jet_obstruction"
        or fourth_chunk_counts_224.get("selected") != 21
        or fourth_chunk_counts_224.get("symbolic_parameter_compatible") != 20
        or fourth_chunk_counts_224.get("candidate_evaluations") != 252
        or fourth_chunk_counts_224.get("candidate_solvable") != 240
        or fourth_chunk_counts_224.get("candidate_obstructed") != 12
        or fourth_chunk_counts_224.get("directional_evaluations") != 251
        or fourth_chunk_counts_224.get("fourth_obligations_remaining") != 2_816
        or fourth_chunk_counts_224.get("fourth_obligations_inferred_passed") != 0
        or quartic_tc2_fourth_jet_chunk_224.get("chunk_contract", {}).get("prior_resume_sha256")
        != quartic_tc2_fourth_jet_chunk_192.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_chunk_224.get("chunk_contract", {}).get("stopped_early")
        is not True
        or quartic_tc2_fourth_jet_chunk_224.get("chunk_contract", {}).get(
            "records_after_first_obstruction_committed_or_inferred"
        )
        != 0
        or fourth_obstruction.get("obligation_offset") != 244
        or fourth_obstruction.get("active_indices") != [0, 2, 3, 9]
        or fourth_obstruction.get("gate") != "fourth-order equal-eigenspace Sylvester compatibility"
        or len(fourth_obstruction.get("obstructed_candidate_ids", [])) != 12
        or fourth_obstruction.get("record_sha256")
        != quartic_tc2_fourth_jet_chunk_224.get("chunk_contract", {}).get("resume_tip_sha256")
        or quartic_tc2_fourth_jet_checkpoint.get("next_obligation_offset") != 245
        or quartic_tc2_fourth_jet_checkpoint.get("remaining_obligations") != 2_816
        or quartic_tc2_fourth_jet_checkpoint.get("current_artifact_content_sha256")
        != quartic_tc2_fourth_jet_chunk_224.get("content_sha256")
        or quartic_tc2_fourth_jet_checkpoint.get("completed_chunks") != 8
        or len(quartic_tc2_fourth_jet_checkpoint.get("history", [])) != 8
        or quartic_tc2_fourth_jet_checkpoint.get("permanently_stopped") is not True
        or quartic_tc2_fourth_jet_checkpoint.get("stop_reason") != "exact_obstruction"
        or quartic_tc2_fourth_jet_status.get("checkpoint_content_sha256")
        != quartic_tc2_fourth_jet_checkpoint.get("content_sha256")
        or quartic_tc2_fourth_jet_status.get("decision") != "stopped"
        or quartic_tc2_fourth_jet_status.get("reason") != "exact_obstruction"
        or quartic_tc2_fourth_jet_status.get("permanently_stopped") is not True
        or quartic_tc2_fourth_jet_status.get("next_obligation_offset") != 245
        or quartic_tc2_fourth_jet_status.get("remaining_obligations") != 2_816
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
    d4_certificate_counts = quartic_tc2_d4_obstruction_certificate.get("counts", {})
    d4_certificate_claims = quartic_tc2_d4_obstruction_certificate.get("claims", {})
    d4_symbolic = quartic_tc2_d4_obstruction_certificate.get("exact_symbolic_certificate", {})
    d4_zero_compression = d4_symbolic.get("equal_eigenspace_compressions", {}).get(
        "zero_eigenspace", {}
    )
    d4_range = d4_symbolic.get("range_certificate", {})
    d4_gap = d4_symbolic.get("exact_candidate_gap", {})
    d4_candidates = d4_symbolic.get("candidate_classification", [])
    if (
        quartic_tc2_d4_obstruction_certificate.get("status")
        != "pass_exact_canonical_d4_obstruction_cokernel_classification"
        or d4_certificate_counts
        != {
            "candidate_compatibilities_certified": 0,
            "candidate_obstructions_certified": 12,
            "candidate_specializations_checked": 12,
            "compression_generic_rank": 2,
            "compression_nonzero_entries": 4,
            "directional_polarization_evaluations": 15,
            "inferred_passes": 0,
            "negative_controls": 4,
            "nonzero_reference_eigenspace_compressions": 1,
            "selector_obligations_classified": 1,
        }
        or quartic_tc2_d4_obstruction_certificate.get("selector_binding", {}).get(
            "obligation_offset"
        )
        != 244
        or quartic_tc2_d4_obstruction_certificate.get("selector_binding", {}).get("active_indices")
        != [0, 2, 3, 9]
        or quartic_tc2_d4_obstruction_certificate.get("source_bindings", {})
        .get("obstruction_chunk", {})
        .get("content_sha256")
        != quartic_tc2_fourth_jet_chunk_224.get("content_sha256")
        or d4_zero_compression.get("factorization") != "(34816/15)*alpha^5*W"
        or d4_zero_compression.get("generic_rank") != 2
        or d4_zero_compression.get("nonzero_entries") != 4
        or d4_zero_compression.get("sha256")
        != "6dcc21e22a450b41d624a739c7db4e5d9753a3848f1a9578730f10d77db125f2"
        or d4_range.get("compatibility_iff_over_Q_or_R") != "alpha=0"
        or d4_range.get("independent_of_c20") is not True
        or d4_gap.get("interval") != "[1088/15,34816/15]"
        or len(d4_candidates) != 12
        or any(row.get("compatible") is not False for row in d4_candidates)
        or {row.get("a10") for row in d4_candidates} != {"-1", "-1/2", "1/2", "1"}
        or d4_certificate_claims.get("canonical_D4_obligation_244_classified") is not True
        or d4_certificate_claims.get("canonical_D4_obligation_244_compatible") is not False
        or d4_certificate_claims.get("all_12_registered_candidates_canonically_obstructed")
        is not True
        or d4_certificate_claims.get("alternative_lower_jet_homogeneous_completion_ruled_out")
        is not False
        or d4_certificate_claims.get("c20_can_remove_canonical_obstruction") is not False
        or any(
            d4_certificate_claims.get(key) is not False
            for key in (
                "full_fourth_jet_range_closed",
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
        raise ValueError("quartic TC2 D4 obstruction certificate is inconsistent")
    d4_homogeneous_counts = quartic_tc2_d4_homogeneous_freedom_reduction.get("counts", {})
    d4_homogeneous_claims = quartic_tc2_d4_homogeneous_freedom_reduction.get("claims", {})
    d4_homogeneous_reduction = quartic_tc2_d4_homogeneous_freedom_reduction.get(
        "homogeneous_freedom_reduction", {}
    )
    if (
        quartic_tc2_d4_homogeneous_freedom_reduction.get("status")
        != "pass_exact_d4_obstruction_invariant_under_all_lower_homogeneous_freedom"
        or quartic_tc2_d4_homogeneous_freedom_reduction.get("selector_binding", {}).get(
            "obligation_offset"
        )
        != 244
        or quartic_tc2_d4_homogeneous_freedom_reduction.get("selector_binding", {}).get(
            "active_indices"
        )
        != [0, 2, 3, 9]
        or quartic_tc2_d4_homogeneous_freedom_reduction.get("selector_binding", {}).get(
            "canonical_witness_sha256"
        )
        != d4_zero_compression.get("sha256")
        or quartic_tc2_d4_homogeneous_freedom_reduction.get("source_bindings", {})
        .get("obstruction_certificate", {})
        .get("content_sha256")
        != quartic_tc2_d4_obstruction_certificate.get("content_sha256")
        or d4_homogeneous_counts.get("polarization_directions_checked") != 15
        or d4_homogeneous_counts.get("Taylor_orders_per_direction_checked") != 5
        or d4_homogeneous_counts.get("stationary_block_checks") != 75
        or d4_homogeneous_counts.get("projector_algebra_checks") != 225
        or d4_homogeneous_counts.get("total_exact_zero_projector_checks") != 300
        or d4_homogeneous_counts.get("lower_jet_reference_kernel_slots_covered_by_identity")
        != 20_842
        or d4_homogeneous_counts.get("induced_cokernel_map_rank") != 0
        or d4_homogeneous_counts.get("candidate_specializations_checked") != 12
        or d4_homogeneous_counts.get("candidate_obstructions_invariant") != 12
        or d4_homogeneous_counts.get("candidate_cancellations") != 0
        or d4_homogeneous_counts.get("inferred_passes") != 0
        or d4_homogeneous_reduction.get("induced_D4_zero_eigenspace_map_rank") != 0
        or d4_homogeneous_reduction.get("induced_D4_zero_eigenspace_map_image_dimension") != 0
        or d4_homogeneous_reduction.get("canonical_rank_two_witness_in_image") is not False
        or d4_homogeneous_reduction.get(
            "total_lower_jet_reference_kernel_slots_before_cross_order_constraints"
        )
        != 20_842
        or d4_homogeneous_claims.get(
            "alternative_lower_jet_homogeneous_completion_ruled_out_for_obligation_244"
        )
        is not True
        or d4_homogeneous_claims.get("all_12_registered_candidates_D4_obstructed_at_obligation_244")
        is not True
        or d4_homogeneous_claims.get("all_3060_fourth_jet_obligations_evaluated") is not False
        or any(
            d4_homogeneous_claims.get(key) is not False
            for key in (
                "full_fourth_jet_range_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
        or len(quartic_tc2_d4_homogeneous_freedom_reduction.get("candidate_classification", []))
        != 12
        or any(
            row.get("D4_compatible_after_all_lower_homogeneous_freedom") is not False
            or row.get("cancellation_possible") is not False
            or row.get("homogeneous_D4_zero_eigenspace_correction") != "0"
            for row in quartic_tc2_d4_homogeneous_freedom_reduction.get(
                "candidate_classification", []
            )
        )
    ):
        raise ValueError("quartic TC2 D4 homogeneous-freedom reduction is inconsistent")
    d4_escape_counts = quartic_tc2_d4_minimal_tc2_escape.get("counts", {})
    d4_escape_claims = quartic_tc2_d4_minimal_tc2_escape.get("claims", {})
    d4_escape = quartic_tc2_d4_minimal_tc2_escape.get("exact_escape", {})
    d4_escape_ansatz = d4_escape.get("correction_ansatz", {})
    d4_escape_map = d4_escape.get("induced_cokernel_map", {})
    d4_escape_rows = d4_escape.get("candidate_classification", [])
    if (
        quartic_tc2_d4_minimal_tc2_escape.get("status")
        != "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only"
        or quartic_tc2_d4_minimal_tc2_escape.get("selector_binding", {}).get("obligation_offset")
        != 244
        or quartic_tc2_d4_minimal_tc2_escape.get("selector_binding", {}).get("active_indices")
        != [0, 2, 3, 9]
        or quartic_tc2_d4_minimal_tc2_escape.get("selector_binding", {}).get(
            "canonical_compression_sha256"
        )
        != "6dcc21e22a450b41d624a739c7db4e5d9753a3848f1a9578730f10d77db125f2"
        or quartic_tc2_d4_minimal_tc2_escape.get("source_bindings", {})
        .get("homogeneous_freedom_reduction", {})
        .get("content_sha256")
        != quartic_tc2_d4_homogeneous_freedom_reduction.get("content_sha256")
        or quartic_tc2_d4_minimal_tc2_escape.get("source_bindings", {})
        .get("obstruction_certificate", {})
        .get("content_sha256")
        != quartic_tc2_d4_obstruction_certificate.get("content_sha256")
        or d4_escape_counts
        != {
            "candidate_D4_obstructions_after_tuning": 0,
            "candidate_D4_solutions_after_tuning": 12,
            "candidate_specializations_checked": 12,
            "correction_basis_dimension": 1,
            "correction_block_rank": 1,
            "distinct_candidate_tunings": 4,
            "induced_cokernel_map_rank": 1,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "selector_obligations_touched": 1,
            "target_cokernel_line_dimension": 1,
        }
        or d4_escape_ansatz.get("V_rank") != 1
        or d4_escape_ansatz.get("V_nonzero_entries") != 6
        or d4_escape_ansatz.get("V_sha256")
        != "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
        or d4_escape_ansatz.get("W_rank") != 2
        or d4_escape_ansatz.get("W_nonzero_entries") != 4
        or d4_escape_ansatz.get("W_sha256")
        != "e44c769b1eaf44c6e0ffc411007d98f9de24c6e8a20bac112d9a0a062e913500"
        or d4_escape_ansatz.get("covariant_or_action_derived") is not False
        or d4_escape_map.get("formula") != "eta -> eta*W"
        or d4_escape_map.get("rank") != 1
        or d4_escape_map.get("image_dimension") != 1
        or d4_escape_map.get("unique_solvability_condition") != "eta=-(34816/15)*alpha^5"
        or d4_escape.get("distinct_candidate_eta_values")
        != ["-34816/15", "-1088/15", "1088/15", "34816/15"]
        or len(d4_escape_rows) != 12
        or any(
            row.get("corrected_D4_Sylvester_solvable") is not True
            or row.get("corrected_D4_Sylvester_residual_zero") is not True
            or row.get("corrected_deltaK_Hermitian") is not True
            or row.get("corrected_equal_eigenspace_compressions_zero") is not True
            or row.get("covariant_operator_origin_proved") is not False
            for row in d4_escape_rows
        )
        or d4_escape_claims.get("obligation_244_minimal_algebraic_TC2_escape_constructed")
        is not True
        or d4_escape_claims.get("candidate_specific_tuned_D4_compatibility_count") != 12
        or d4_escape_claims.get("correction_covariant_or_action_derived") is not False
        or d4_escape_claims.get("correction_gauge_constraint_compatible") is not False
        or d4_escape_claims.get("corrected_candidate_family_registered") is not False
        or d4_escape_claims.get("single_universal_eta_closes_all_12") is not False
        or any(
            d4_escape_claims.get(key) is not False
            for key in (
                "full_fourth_jet_range_closed",
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
        raise ValueError("quartic TC2 D4 minimal algebraic escape is inconsistent")
    origin_no_go = quartic_tc2_d4_registered_operator_origin_no_go.get("exact_no_go", {})
    origin_map = origin_no_go.get("induced_cokernel_map", {})
    origin_support = origin_no_go.get("constraint_support_audit", {})
    origin_claims = quartic_tc2_d4_registered_operator_origin_no_go.get("claims", {})
    if (
        quartic_tc2_d4_registered_operator_origin_no_go.get("status")
        != "pass_exact_no_go_for_registered_support_preserving_TC2_operator_class"
        or quartic_tc2_d4_registered_operator_origin_no_go.get("source_bindings", {})
        .get("minimal_escape", {})
        .get("content_sha256")
        != quartic_tc2_d4_minimal_tc2_escape.get("content_sha256")
        or quartic_tc2_d4_registered_operator_origin_no_go.get("counts")
        != {
            "broad_induced_cokernel_map_rank": 0,
            "broad_support_preserving_domain_dimension": 55,
            "inferred_global_passes": 0,
            "negative_controls": 5,
            "positive_controls": 1,
            "registered_TC2_blocks_checked": 4,
            "registered_action_terms_checked": 1,
            "target_augmented_rank": 1,
        }
        or origin_map.get("domain_dimension") != 55
        or origin_map.get("rank") != 0
        or origin_map.get("image_dimension") != 0
        or origin_map.get("augmented_rank") != 1
        or origin_map.get("target_in_image") is not False
        or origin_support.get("registered_right_support_columns") != [54]
        or origin_support.get("escape_V_right_support_columns") != [21]
        or origin_support.get("support_intersection_empty") is not True
        or origin_claims.get("registered_linear_X_quartic_Horndeski_TC2_origin_ruled_out")
        is not True
        or origin_claims.get("registered_support_preserving_gauge_deformation_ruled_out")
        is not True
        or any(
            origin_claims.get(key) is not False
            for key in (
                "arbitrary_covariant_quartic_operator_ruled_out",
                "arbitrary_gauge_fixed_operator_deformation_ruled_out",
                "constraint_topology_changing_realization_constructed",
                "corrected_candidate_family_registered",
                "covariant_realization_constructed",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
    ):
        raise ValueError("quartic TC2 registered operator origin no-go is inconsistent")
    topology_direct = quartic_tc2_d4_topology_changing_origin.get("exact_classification", {}).get(
        "direct_action_origin_no_go", {}
    )
    topology_joint = topology_direct.get("joint_result", {})
    topology_selectors = quartic_tc2_d4_topology_changing_origin.get(
        "exact_classification", {}
    ).get("explicit_TC2_selector_classification", {})
    topology_selector_counts = topology_selectors.get("canonical_counts", {})
    topology_claims = quartic_tc2_d4_topology_changing_origin.get("claims", {})
    if (
        quartic_tc2_d4_topology_changing_origin.get("status")
        != "pass_exact_direct_action_origin_no_go_and_complete_selector_classification"
        or quartic_tc2_d4_topology_changing_origin.get("source_bindings", {})
        .get("registered_operator_no_go", {})
        .get("content_sha256")
        != quartic_tc2_d4_registered_operator_origin_no_go.get("content_sha256")
        or quartic_tc2_d4_topology_changing_origin.get("counts")
        != {
            "canonical_cokernel_capable_selectors": 5,
            "canonical_kernel_selectors": 16,
            "canonical_nonzero_incapable_selectors": 34,
            "canonical_selectors_checked": 55,
            "constructive_rank_one_cokernel_blocks": 5,
            "direct_deltaP_domain_dimension": 605,
            "direct_joint_cokernel_map_rank": 0,
            "direct_joint_domain_dimension": 2145,
            "direct_symmetric_deltaK_domain_dimension": 1540,
            "inferred_global_passes": 0,
            "negative_controls": 6,
        }
        or topology_joint.get("direct_action_principal_origin_ruled_out") is not True
        or topology_joint.get("joint_map_rank") != 0
        or topology_joint.get("target_W_in_image") is not False
        or topology_selector_counts
        != {
            "cokernel_capable_selectors": 5,
            "nonzero_projection_incapable_selectors": 34,
            "selectors_checked": 55,
            "zero_projection_selectors": 16,
        }
        or topology_selectors.get("canonical_capable_indices") != [21, 44, 48, 51, 53]
        or topology_selectors.get("registered_selector_control", {}).get("selector_index") != 54
        or topology_selectors.get("registered_selector_control", {}).get("projection_zero")
        is not True
        or topology_claims.get("direct_second_order_action_principal_origin_ruled_out") is not True
        or topology_claims.get("all_55_canonical_TC2_input_selectors_classified") is not True
        or topology_claims.get("explicit_constraint_row_covariant_origin_constructed") is not False
        or topology_claims.get("constraint_propagation_for_topology_change_proved") is not False
        or any(
            topology_claims.get(key) is not False
            for key in (
                "remaining_D4_selector_closed",
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
        raise ValueError("quartic TC2 topology-changing origin classification is inconsistent")
    curl_counts = quartic_tc2_d4_curl_constraint_admission.get("counts", {})
    curl_claims = quartic_tc2_d4_curl_constraint_admission.get("claims", {})
    curl_admission = quartic_tc2_d4_curl_constraint_admission.get("exact_admission", {})
    curl_result = curl_admission.get("admission_result", {})
    curl_propagation = curl_admission.get("constraint_propagation", {})
    curl_operator = curl_admission.get("gauge_fixed_operator", {})
    curl_equivalence = curl_admission.get("physical_reduction_equivalence", {})
    curl_coefficient = curl_admission.get("coefficient_jet", {})
    if (
        quartic_tc2_d4_curl_constraint_admission.get("status")
        != "pass_exact_gauge_fixed_curl_constraint_admission_for_minimal_V"
        or quartic_tc2_d4_curl_constraint_admission.get("source_bindings", {})
        .get("topology_classification", {})
        .get("content_sha256")
        != quartic_tc2_d4_topology_changing_origin.get("content_sha256")
        or curl_counts
        != {
            "candidate_reference_D4_solutions_inherited": 12,
            "curl_constraints_propagated": 33,
            "definition_constraints_propagated": 33,
            "direction_blocks": 2,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "ordered_fourth_coefficient_derivatives_checked": 256,
            "ordered_lower_coefficient_derivatives_checked": 85,
            "output_nonzero_coefficients": 6,
            "source_curl_constraints": 1,
        }
        or curl_result.get("gauge_fixed_constraint_operator_constructed") is not True
        or curl_result.get("canonical_definition_constraint_surface_invariant") is not True
        or curl_result.get("canonical_curl_constraint_surface_invariant") is not True
        or curl_result.get("variable_coefficient_constraint_surface_invariant") is not True
        or curl_result.get("reference_direction_minimal_escape_physically_equivalent") is not True
        or curl_result.get("covariant_action_derived") is not False
        or curl_result.get("spatially_covariant_tensor_completion_proved") is not False
        or curl_result.get("all_direction_Sylvester_compatibility_proved") is not False
        or curl_operator.get("direction_1_block_equals_V") is not True
        or curl_operator.get("direction_1_block_sha256")
        != "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
        or curl_operator.get("direction_2_companion_sha256")
        != "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
        or curl_operator.get("direction_3_block_zero") is not True
        or curl_operator.get("minimal_direction_block_count") != 2
        or curl_equivalence.get("directional_operator_times_gradient_lift_zero") is not True
        or curl_equivalence.get("physical_second_order_solutions_unchanged") is not True
        or curl_propagation.get("definition_constraint_propagation", {}).get("constraint_count")
        != 33
        or curl_propagation.get("definition_constraint_propagation", {}).get("map_rank") != 1
        or curl_propagation.get("curl_constraint_propagation", {}).get("constraint_count") != 33
        or curl_propagation.get("curl_constraint_propagation", {}).get("map_rank") != 3
        or curl_coefficient.get("orders_0_through_3_zero") is not True
        or curl_coefficient.get("canonical_D_0_D_2_D_3_D_9_value") != "1"
        or curl_admission.get("reference_D4_binding", {}).get("candidate_count") != 12
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_curl_constraint_admission.get(
                "negative_controls", {}
            ).values()
        )
        or curl_claims.get("minimal_V_gauge_fixed_curl_constraint_realized") is not True
        or curl_claims.get("canonical_constraint_surface_invariance_proved") is not True
        or any(
            curl_claims.get(key) is not False
            for key in (
                "covariant_action_origin_constructed",
                "spatially_covariant_tensor_completion_proved",
                "all_spatial_direction_compatibility_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
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
        raise ValueError("quartic TC2 D4 curl-constraint admission is inconsistent")
    companion_counts = quartic_tc2_d4_curl_companion_range.get("counts", {})
    companion_claims = quartic_tc2_d4_curl_companion_range.get("claims", {})
    companion_audit = quartic_tc2_d4_curl_companion_range.get("exact_companion_audit", {})
    companion_eigenspaces = companion_audit.get("equal_eigenspace_audit", {})
    companion_range = companion_audit.get("pure_curl_completion_range", {})
    if (
        quartic_tc2_d4_curl_companion_range.get("status")
        != "pass_exact_axis2_companion_obstruction_and_pure_curl_range_no_go"
        or quartic_tc2_d4_curl_companion_range.get("source_bindings", {})
        .get("curl_admission", {})
        .get("content_sha256")
        != quartic_tc2_d4_curl_constraint_admission.get("content_sha256")
        or quartic_tc2_d4_curl_companion_range.get("source_bindings", {})
        .get("topology_classification", {})
        .get("content_sha256")
        != quartic_tc2_d4_topology_changing_origin.get("content_sha256")
        or companion_counts
        != {
            "candidate_companion_witnesses": 12,
            "companion_compression_nonzero_entries": 10,
            "companion_compression_rank": 2,
            "distinct_scaled_witnesses": 4,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "nonzero_equal_eigenspace_compressions": 1,
            "pure_C23_effective_parameters": 363,
            "pure_C23_range_rank": 297,
            "pure_C23_raw_parameters": 605,
            "reference_eigenspaces_checked": 7,
            "target_augmented_rank": 298,
        }
        or companion_audit.get("axis_2_reference", {}).get("direct_axis_2_P55_matches_rotation")
        is not True
        or companion_audit.get("axis_2_reference", {}).get("axis_swap_orthogonal") is not True
        or companion_audit.get("axis_2_reference", {}).get("axis_swap_state_sha256")
        != "db8252cef6f0505fe21d0be1150ae476f12089a79cacedffdbf144145bc6cc55"
        or quartic_tc2_d4_curl_companion_range.get("selector_binding")
        != {
            "active_indices": [0, 2, 3, 9],
            "companion_block_sha256": (
                "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
            ),
            "companion_input_selector": 54,
            "obligation_offset": 244,
            "reference_direction": "e2",
        }
        or companion_audit.get("companion_block", {}).get("rank") != 1
        or companion_audit.get("companion_block", {}).get("nonzero_entries") != 6
        or companion_audit.get("companion_block", {}).get("sha256")
        != "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
        or companion_eigenspaces.get("companion_block_alone_Sylvester_compatible") is not False
        or companion_eigenspaces.get("companion_compression_rank") != 2
        or companion_eigenspaces.get("companion_compression_nonzero_entries") != 10
        or companion_eigenspaces.get("sole_nonzero_eigenvalue") != "0"
        or companion_eigenspaces.get("companion_compression_sha256")
        != "def5dc985fa3356a9a21b2b06d4ebe0f0365058403e3e762eab161d7fb2822be"
        or companion_audit.get("rotation_control", {}).get("companion_and_rotated_W_span_dimension")
        != 2
        or companion_audit.get("rotation_control", {}).get("companion_is_rotated_W_multiple")
        is not False
        or companion_range.get("exact_range_map", {}).get("matrix_shape") != [528, 363]
        or companion_range.get("exact_range_map", {}).get("rank") != 297
        or companion_range.get("exact_range_map", {}).get("target_augmented_rank") != 298
        or companion_range.get("exact_range_map", {}).get("target_in_image") is not False
        or companion_range.get("result", {}).get("pure_curl_self_compatible_completion_exists")
        is not False
        or companion_audit.get("necessary_full_D4_condition", {}).get("base_D4_RHS_computed")
        is not False
        or companion_audit.get("necessary_full_D4_condition", {}).get("condition_verified")
        is not False
        or companion_audit.get("necessary_full_D4_condition", {}).get("condition_refuted")
        is not False
        or len(companion_audit.get("candidate_companion_witnesses", [])) != 12
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_curl_companion_range.get("negative_controls", {}).values()
        )
        or companion_claims.get("axis2_companion_all_eigenspaces_audited") is not True
        or companion_claims.get("pure_curl_self_compatible_completion_ruled_out") is not True
        or any(
            companion_claims.get(key) is not False
            for key in (
                "companion_block_alone_Sylvester_compatible",
                "full_axis2_base_D4_RHS_evaluated",
                "full_axis2_D4_compatibility_proved",
                "full_axis2_D4_obstruction_proved",
                "spatially_covariant_tensor_completion_proved",
                "all_spatial_direction_compatibility_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
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
        raise ValueError("quartic TC2 D4 curl companion range is inconsistent")
    axis2_counts = quartic_tc2_d4_axis2_base_rhs.get("counts", {})
    axis2_claims = quartic_tc2_d4_axis2_base_rhs.get("claims", {})
    axis2_audit = quartic_tc2_d4_axis2_base_rhs.get("exact_axis2_base_D4_audit", {})
    axis2_result = axis2_audit.get("result", {})
    axis2_base = axis2_audit.get("polarized_base_D4", {})
    if (
        quartic_tc2_d4_axis2_base_rhs.get("status")
        != "pass_exact_all_12_axis2_D4_companion_obstructions"
        or quartic_tc2_d4_axis2_base_rhs.get("source_bindings", {})
        .get("companion_range", {})
        .get("content_sha256")
        != quartic_tc2_d4_curl_companion_range.get("content_sha256")
        or axis2_counts
        != {
            "candidate_conditions_checked": 12,
            "corrected_axis2_D4_compatibilities": 0,
            "corrected_axis2_D4_obstructions": 12,
            "directional_evaluations": 15,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "zero_speed_cancellations_exact": 0,
        }
        or quartic_tc2_d4_axis2_base_rhs.get("selector_binding")
        != {
            "active_indices": [0, 2, 3, 9],
            "obligation_offset": 244,
            "reference_direction": "e2",
            "same_active_tensor_component_inputs": True,
            "selector_record_sha256": (
                "337daa86bf740ae9e66dbef0829df30297c02e22b8baeb6b90328d608fa66c87"
            ),
        }
        or axis2_audit.get("directional_evaluations") != 15
        or axis2_base.get("RHS_base_nonzero_entries") != 0
        or {
            axis2_base.get(key)
            for key in (
                "D4K55_sha256",
                "D4P55_sha256",
                "D4TC2_sha256",
                "RHS_base_sha256",
            )
        }
        != {"d4d121c239c4905e1183840e887b102fd0a0c9dc588820df4b858291c70cb4ad"}
        or axis2_audit.get("companion_correction")
        != {
            "block_sha256": ("9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"),
            "compression_nonzero_entries": 10,
            "compression_rank": 2,
            "compression_sha256": (
                "def5dc985fa3356a9a21b2b06d4ebe0f0365058403e3e762eab161d7fb2822be"
            ),
        }
        or axis2_result
        != {
            "base_D4_RHS_identically_zero": True,
            "base_D4_Sylvester_solvable": True,
            "base_D4_deltaK_zero": True,
            "base_D4_residual_zero": True,
            "candidate_conditions_checked": 12,
            "corrected_axis2_D4_compatibilities": 0,
            "corrected_axis2_D4_obstructions": 12,
            "full_axis2_base_D4_RHS_evaluated": True,
            "wrong_sign_companion_compatibilities": 0,
            "zero_speed_cancellations_exact": 0,
        }
        or len(axis2_audit.get("candidate_comparison", [])) != 12
        or any(
            record.get("corrected_axis2_D4_Sylvester_solvable") is not False
            or record.get("zero_speed_cancellation_exact") is not False
            or record.get("corrected_nonzero_equal_eigenspace_compressions", {})
            .get("0", {})
            .get("rank")
            != 2
            for record in axis2_audit.get("candidate_comparison", [])
        )
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_axis2_base_rhs.get("negative_controls", {}).values()
        )
        or axis2_claims.get("full_axis2_base_D4_RHS_evaluated") is not True
        or axis2_claims.get("all_12_axis2_D4_obstructions_proved") is not True
        or axis2_claims.get("fixed_chart_curl_completion_axis2_D4_rejected") is not True
        or any(
            axis2_claims.get(key) is not False
            for key in (
                "all_12_axis2_D4_compatibilities_proved",
                "all_spatial_direction_compatibility_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
                "spatially_covariant_tensor_completion_proved",
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
        raise ValueError("quartic TC2 axis-two base D4 RHS gate is inconsistent")
    spatial_no_go_counts = quartic_tc2_d4_spatial_gradient_no_go.get("counts", {})
    spatial_no_go_exact = quartic_tc2_d4_spatial_gradient_no_go.get("exact_classification", {})
    if (
        quartic_tc2_d4_spatial_gradient_no_go.get("status")
        != "pass_exact_exhaustive_spatial_gradient_annihilator_completion_no_go"
        or quartic_tc2_d4_spatial_gradient_no_go.get("selector_binding")
        != {
            "active_indices": [0, 2, 3, 9],
            "obligation_offset": 244,
            "reference_direction": "e2",
            "same_active_tensor_component_inputs": True,
        }
        or spatial_no_go_counts
        != {
            "candidate_no_go_results": 12,
            "effective_projected_parameters": 363,
            "field_dimension": 11,
            "independent_affine_constraints": 3025,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "non_gradient_input_columns": 22,
            "projected_range_rank": 297,
            "raw_affine_dimension": 605,
            "raw_unknown_coefficients_after_fixing_B1": 3630,
            "spatial_directions": 3,
            "spatial_input_columns": 33,
            "state_dimension": 55,
            "target_augmented_rank": 298,
        }
        or quartic_tc2_d4_spatial_gradient_no_go.get("source_bindings", {})
        .get("axis2_base_rhs", {})
        .get("content_sha256")
        != quartic_tc2_d4_axis2_base_rhs.get("content_sha256")
        or quartic_tc2_d4_spatial_gradient_no_go.get("source_bindings", {})
        .get("companion_range", {})
        .get("content_sha256")
        != quartic_tc2_d4_curl_companion_range.get("content_sha256")
        or spatial_no_go_exact.get("exact_affine_solution", {}).get("general_solution")
        != "B2*E3=A and B3*E2=-A; every other free spatial block is zero"
        or spatial_no_go_exact.get("axis2_projected_range", {}).get("target_in_image") is not False
        or spatial_no_go_exact.get("candidate_consequence", {}).get(
            "candidate_completions_in_declared_class"
        )
        != 0
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_spatial_gradient_no_go.get(
                "negative_controls", {}
            ).values()
        )
    ):
        raise ValueError("quartic TC2 spatial-gradient annihilator no-go is inconsistent")
    full_no_go_counts = quartic_tc2_d4_full_linear_gradient_no_go.get("counts", {})
    full_no_go_exact = quartic_tc2_d4_full_linear_gradient_no_go.get("exact_classification", {})
    qv_partition = full_no_go_exact.get("canonical_qv_selector_partition", {})
    combined_range = full_no_go_exact.get("combined_axis2_free_B2_range", {})
    if (
        quartic_tc2_d4_full_linear_gradient_no_go.get("status")
        != "pass_exact_full_linear_gradient_annihilator_completion_no_go"
        or quartic_tc2_d4_full_linear_gradient_no_go.get("selector_binding")
        != quartic_tc2_d4_spatial_gradient_no_go.get("selector_binding")
        or full_no_go_counts
        != {
            "candidate_no_go_results": 12,
            "canonical_qv_capable_selectors": 0,
            "canonical_qv_kernel_selectors": 11,
            "canonical_qv_nonzero_incapable_selectors": 11,
            "canonical_qv_selectors_checked": 22,
            "combined_free_B2_selector_subspace_rank": 22,
            "combined_free_B2_wedge_range_rank": 473,
            "combined_target_augmented_rank": 474,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "qv_selector_subspace_rank": 11,
            "qv_target_augmented_rank": 298,
            "qv_wedge_range_rank": 297,
        }
        or quartic_tc2_d4_full_linear_gradient_no_go.get("source_bindings", {})
        .get("spatial_gradient_no_go", {})
        .get("content_sha256")
        != quartic_tc2_d4_spatial_gradient_no_go.get("content_sha256")
        or qv_partition.get("capable_indices") != []
        or qv_partition.get("kernel_indices") != list(range(33, 44))
        or qv_partition.get("nonzero_incapable_indices") != list(range(11))
        or combined_range.get("selector_projection_rank") != 22
        or combined_range.get("wedge_range_rank") != 473
        or combined_range.get("target_augmented_rank") != 474
        or combined_range.get("target_in_image") is not False
        or full_no_go_exact.get("candidate_consequence", {}).get(
            "candidate_linear_gradient_annihilator_completions"
        )
        != 0
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_full_linear_gradient_no_go.get(
                "negative_controls", {}
            ).values()
        )
        or quartic_tc2_d4_full_linear_gradient_no_go.get("claims", {}).get(
            "all_operator_classes_ruled_out"
        )
        is not False
    ):
        raise ValueError("quartic TC2 full-linear gradient-annihilator no-go is inconsistent")
    cubic_escape_counts = quartic_tc2_d4_parity_cubic_escape.get("counts", {})
    cubic_escape = quartic_tc2_d4_parity_cubic_escape.get("exact_escape", {})
    cubic_symbol = cubic_escape.get("exact_symbol", {})
    cubic_two_axis = cubic_escape.get("two_axis_D4_consequence", {})
    cubic_claims = quartic_tc2_d4_parity_cubic_escape.get("claims", {})
    if (
        quartic_tc2_d4_parity_cubic_escape.get("status")
        != "pass_exact_minimal_parity_preserving_cubic_angular_two_axis_escape"
        or quartic_tc2_d4_parity_cubic_escape.get("selector_binding")
        != {
            "active_indices": [0, 2, 3, 9],
            "newly_closed_companion_direction": "e2",
            "obligation_offset": 244,
            "reference_direction": "e1",
        }
        or cubic_escape_counts
        != {
            "bound_predecessors": 6,
            "candidate_specializations": 12,
            "generic_direction_D4_compatibilities_proved": 0,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "new_axis2_D4_compatibilities": 12,
            "new_axis2_D4_obstructions": 0,
            "nonzero_polynomial_coefficient_blocks": 2,
            "reference_e1_D4_solutions_inherited": 12,
            "scalar_multiplier_degree": 2,
            "total_angular_polynomial_degree": 3,
        }
        or quartic_tc2_d4_parity_cubic_escape.get("source_bindings", {})
        .get("full_linear_no_go", {})
        .get("content_sha256")
        != quartic_tc2_d4_full_linear_gradient_no_go.get("content_sha256")
        or quartic_tc2_d4_parity_cubic_escape.get("source_bindings", {})
        .get("axis2_base_rhs", {})
        .get("content_sha256")
        != quartic_tc2_d4_axis2_base_rhs.get("content_sha256")
        or quartic_tc2_d4_parity_cubic_escape.get("source_bindings", {})
        .get("curl_admission", {})
        .get("content_sha256")
        != quartic_tc2_d4_curl_constraint_admission.get("content_sha256")
        or cubic_escape.get("minimality", {}).get("canonical_multiplier") != "a(n)=n1^2"
        or cubic_escape.get("minimality", {}).get(
            "minimal_total_angular_polynomial_degree"
        )
        != 3
        or cubic_escape.get("minimality", {}).get("constant_even_multiplier_impossible")
        is not True
        or cubic_symbol.get("definition")
        != "B_cubic(n)=n1^2*(n1*V+n2*C_companion)"
        or cubic_symbol.get("antipodal_odd") is not True
        or cubic_symbol.get("sphere_multiplier_interval") != "0<=n1^2<=1"
        or cubic_symbol.get("e2_block_zero") is not True
        or cubic_symbol.get("symbol_sha256")
        != "1a9e8b9f5101ccf6a59cd81181ff02943182b06fea2207fc01814aa32ee713ca"
        or cubic_escape.get("physical_gradient_lift_equivalence", {}).get("residual_zero")
        is not True
        or cubic_escape.get("pseudodifferential_constraint_admission", {}).get(
            "M1_fourier_symbol"
        )
        != "xi1^2/|xi|^2=n1^2"
        or cubic_escape.get("pseudodifferential_constraint_admission", {}).get(
            "periodic_or_Schwartz_constraint_surface_invariant"
        )
        is not True
        or cubic_escape.get("pseudodifferential_constraint_admission", {}).get(
            "local_differential_operator_realization_proved"
        )
        is not False
        or cubic_two_axis.get("reference_e1_solutions_inherited") != 12
        or cubic_two_axis.get("axis2_D4_compatibilities") != 12
        or cubic_two_axis.get("axis2_D4_obstructions") != 0
        or cubic_two_axis.get("axis2_base_D4_RHS_identically_zero") is not True
        or cubic_two_axis.get("all_direction_D4_compatibility_proved") is not False
        or len(cubic_two_axis.get("candidate_records", [])) != 12
        or any(
            not record.get("e1_D4_Sylvester_solvable_inherited")
            or not record.get("e2_D4_Sylvester_solvable")
            or record.get("all_direction_D4_Sylvester_solvable") is not False
            or record.get("e2_angular_multiplier") != "0"
            for record in cubic_two_axis.get("candidate_records", [])
        )
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_parity_cubic_escape.get(
                "negative_controls", {}
            ).values()
        )
        or cubic_claims.get("minimal_parity_preserving_cubic_scalar_multiplier_constructed")
        is not True
        or cubic_claims.get("all_12_axis2_D4_compatibilities_proved_for_cubic_symbol")
        is not True
        or cubic_claims.get("generic_direction_D4_compatibility_proved") is not False
        or any(
            cubic_claims.get(key) is not False
            for key in (
                "local_differential_operator_origin_proved",
                "covariant_action_origin_proved",
                "spatially_covariant_tensor_completion_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
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
        raise ValueError("quartic TC2 parity-cubic angular escape is inconsistent")
    generic_direction_counts = quartic_tc2_d4_parity_cubic_generic_direction.get(
        "counts", {}
    )
    generic_direction_audit = quartic_tc2_d4_parity_cubic_generic_direction.get(
        "exact_generic_direction_audit", {}
    )
    generic_direction_records = generic_direction_audit.get("direction_records", [])
    generic_direction_claims = quartic_tc2_d4_parity_cubic_generic_direction.get(
        "claims", {}
    )
    if (
        quartic_tc2_d4_parity_cubic_generic_direction.get("status")
        != "pass_exact_generic_direction_obstruction_of_parity_cubic_escape"
        or quartic_tc2_d4_parity_cubic_generic_direction.get("selector_binding")
        != {
            "active_indices": [0, 2, 3, 9],
            "frequency_selector": "ordered_three_frame_rational_generic_direction_slice",
            "obligation_offset": 244,
        }
        or generic_direction_counts
        != {
            "candidate_direction_compatibilities": 0,
            "candidate_direction_obstructions": 12,
            "candidate_direction_systems_evaluated": 12,
            "declared_rational_frames": 3,
            "directional_recurrence_evaluations": 15,
            "frames_evaluated": 1,
            "frames_unevaluated_after_stop": 2,
            "inferred_global_passes": 0,
            "negative_controls": 6,
        }
        or quartic_tc2_d4_parity_cubic_generic_direction.get("source_bindings", {})
        .get("parity_cubic_escape", {})
        .get("content_sha256")
        != quartic_tc2_d4_parity_cubic_escape.get("content_sha256")
        or generic_direction_audit.get("selector", {}).get("stop_reason")
        != "first_exact_generic_direction_obstruction"
        or generic_direction_audit.get("selector", {}).get("frames_evaluated") != 1
        or generic_direction_audit.get("selector", {}).get(
            "frames_unevaluated_after_stop"
        )
        != 2
        or generic_direction_audit.get("selector", {}).get(
            "not_an_interpolation_basis_for_the_full_sphere"
        )
        is not True
        or len(generic_direction_records) != 1
        or generic_direction_records[0].get("direction") != ["3/5", "4/5", "0"]
        or generic_direction_records[0].get("directional_evaluations") != 15
        or generic_direction_records[0].get("all_seven_eigenspaces_checked_per_candidate")
        is not True
        or generic_direction_records[0].get("base_D4_RHS_nonzero_entries") != 64
        or generic_direction_records[0].get("base_D4_RHS_sha256")
        != "1c6c07e82d619fa24a46bf33033f7370e34e442c50c33f179356189142491972"
        or generic_direction_records[0].get("cubic_correction_block_rank") != 1
        or generic_direction_records[0].get("cubic_correction_block_nonzero_entries") != 12
        or generic_direction_records[0].get("cubic_correction_block_sha256")
        != "bbbec5a07976b06d20bc36e5c09ef2c1158ac4f40f0e909b5b093fe702e21399"
        or generic_direction_records[0].get("cubic_correction_skew_rank") != 2
        or generic_direction_records[0].get("candidate_compatibilities") != 0
        or generic_direction_records[0].get("candidate_obstructions") != 12
        or len(generic_direction_records[0].get("candidate_records", [])) != 12
        or any(
            record.get("D4_Sylvester_solvable") is not False
            or record.get("nonzero_equal_eigenspace_compressions", {}).get("0", {}).get(
                "rank"
            )
            != 2
            or record.get("nonzero_equal_eigenspace_compressions", {}).get("0", {}).get(
                "nonzero_entries"
            )
            != 14
            for record in generic_direction_records[0].get("candidate_records", [])
        )
        or generic_direction_audit.get("result")
        != {
            "all_evaluated_generic_directions_compatible": False,
            "candidate_direction_compatibilities": 0,
            "candidate_direction_obstructions": 12,
            "candidate_direction_systems_evaluated": 12,
            "cubic_escape_all_direction_completion_rejected": True,
            "full_generic_direction_sphere_classified": False,
        }
        or generic_direction_claims.get(
            "exact_rational_generic_direction_D4_recurrence_evaluated"
        )
        is not True
        or generic_direction_claims.get("parity_cubic_all_direction_completion_rejected")
        is not True
        or any(
            generic_direction_claims.get(key) is not False
            for key in (
                "generic_direction_D4_compatibility_proved",
                "full_generic_direction_sphere_classified",
                "local_differential_operator_origin_proved",
                "covariant_action_origin_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
        or any(
            control.get("rejected") is not True
            for control in quartic_tc2_d4_parity_cubic_generic_direction.get(
                "negative_controls", {}
            ).values()
        )
    ):
        raise ValueError("quartic TC2 parity-cubic generic-direction audit is inconsistent")
    if (
        unified_live_dashboard_service_readiness.get("decision")
        != "ready_enabled_read_only_bounded"
        or unified_live_dashboard_service_readiness.get("reads_live_campaign_database") is not True
        or unified_live_dashboard_service_readiness.get("writes_live_campaign_database")
        is not False
        or unified_live_dashboard_service_readiness.get("atomic_snapshot_write") is not True
        or unified_live_dashboard_service_readiness.get("atomic_dashboard_write") is not True
        or unified_live_dashboard_service_readiness.get("hash_bound_checkpoint") is not True
        or unified_live_dashboard_service_readiness.get("immutable_snapshot_overwritten")
        is not False
        or unified_live_dashboard_service_readiness.get("refresh_interval_seconds") != 300
        or unified_live_dashboard_service_readiness.get("maximum_refreshes") != 4_032
        or unified_live_dashboard_service_readiness.get("maximum_consecutive_failures") != 12
        or unified_live_dashboard_service_readiness.get("maximum_output_bytes") != 3_145_728
        or unified_live_dashboard_service_readiness.get("data_seals")
        != {
            "dark_matter_or_halo_inputs": False,
            "observations_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
    ):
        raise ValueError("unified live dashboard service readiness is inconsistent")
    safety_contract = unified_live_dashboard_service_safety.get("safety_contract", {})
    if (
        unified_live_dashboard_service_safety.get("decision")
        != "hardened_service_ready_not_started"
        or unified_live_dashboard_service_safety.get("service_started") is not False
        or unified_live_dashboard_service_safety.get("live_database_opened_by_readiness")
        is not False
        or unified_live_dashboard_service_safety.get("supervisor_outputs_opened_by_readiness")
        is not False
        or unified_live_dashboard_service_safety.get("data_seals")
        != {
            "dark_matter_or_halo_inputs": False,
            "observations_opened": False,
            "paid_llm_calls": False,
            "redshift_distance_inputs": False,
        }
        or safety_contract.get("windows_argv_list_shell_false") is not True
        or safety_contract.get("worker_pid_bound_to_normalized_argv") is not True
        or safety_contract.get("legacy_worker_absence_required_before_start") is not True
        or safety_contract.get("cross_start_exclusive_lease")
        != "runs/engine/unified-live-dashboard-cutover.lease.json"
        or safety_contract.get("atomic_starting_checkpoint_before_spawn") is not True
        or safety_contract.get("repeated_start_launch_allowed") is not False
        or safety_contract.get("worker_finally_releases_owned_lease") is not True
        or safety_contract.get("stale_lease_recovery_requires_pid_argv_nonmatch") is not True
        or safety_contract.get("first_refresh_observes_running_checkpoint") is not True
        or safety_contract.get("runtime_outputs_gitignored") is not True
        or safety_contract.get("counters_preserved_across_compatible_reload") is not True
        or safety_contract.get("reload_failures_checkpointed_in_worker_finally") is not True
        or safety_contract.get("pre_and_post_projection_input_manifest_required") is not True
        or safety_contract.get("leaderboard_history_seed_checked_snapshot")
        != "runs/engine/unified-engine-status.json"
        or safety_contract.get("leaderboard_history_seed_core_and_content_hash_validated")
        is not True
        or safety_contract.get("leaderboard_history_seed_legacy_fallback_hash_bound") is not True
        or safety_contract.get("leaderboard_history_seed_pre_and_post_hash_guarded") is not True
        or safety_contract.get("leaderboard_history_seed_source_revisions_structurally_validated")
        is not True
        or safety_contract.get("maximum_seed_history_entries") != 64
        or safety_contract.get("maximum_seed_history_bytes") != 65_536
        or safety_contract.get("stale_projection_publication_allowed") is not False
        or safety_contract.get("unpredictable_exclusive_temp_files") is not True
        or safety_contract.get("target_and_log_symlinks_rejected") is not True
        or safety_contract.get("control_poll_interval_seconds") != 0.25
        or safety_contract.get("refresh_phase_control_guards") != 4
    ):
        raise ValueError("unified live dashboard safety readiness is inconsistent")

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
        "continuous_dashboard": {
            "decision": unified_live_dashboard_service_readiness["decision"],
            "read_contract": "sqlite_uri_mode_ro_plus_query_only_transaction",
            "reads_live_campaign_database": True,
            "writes_live_campaign_database": False,
            "refresh_interval_seconds": unified_live_dashboard_service_readiness[
                "refresh_interval_seconds"
            ],
            "maximum_refreshes": unified_live_dashboard_service_readiness["maximum_refreshes"],
            "maximum_consecutive_failures": unified_live_dashboard_service_readiness[
                "maximum_consecutive_failures"
            ],
            "maximum_output_bytes": unified_live_dashboard_service_readiness[
                "maximum_output_bytes"
            ],
            "hash_bound_checkpoint": True,
            "atomic_snapshot_write": True,
            "atomic_dashboard_write": True,
            "immutable_snapshot_overwritten": False,
            "safety_hardening": {
                "decision": unified_live_dashboard_service_safety["decision"],
                "service_started": unified_live_dashboard_service_safety["service_started"],
                "safety_contract": unified_live_dashboard_service_safety["safety_contract"],
                "remaining_limitations": unified_live_dashboard_service_safety[
                    "remaining_limitations"
                ],
            },
        },
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
        "einstein_aether_coupling_boundary_kkt": {
            "decision": einstein_aether_coupling_boundary_kkt["decision"],
            "decision_counts": einstein_aether_coupling_boundary_kkt["decision_counts"],
            "gate_counts": einstein_aether_coupling_boundary_kkt["gate_counts"],
            "first_blocker": einstein_aether_coupling_boundary_kkt["first_blocker"],
            "aligned_contract": einstein_aether_coupling_boundary_kkt["aligned_contract"],
            "symbolic_factorization": einstein_aether_coupling_boundary_kkt[
                "symbolic_factorization"
            ],
            "exact_witnesses": einstein_aether_coupling_boundary_kkt["exact_witnesses"],
            "reduced_five_mode_chart_binding": einstein_aether_coupling_boundary_kkt[
                "reduced_five_mode_chart_binding"
            ],
            "scope": einstein_aether_coupling_boundary_kkt["scope"],
        },
        "transactional_gravity_proposal": {
            "primary_source": {
                "arxiv_id": transactional_gravity_intake["primary_source"]["arxiv_id"],
                "title": transactional_gravity_intake["primary_source"]["title"],
                "authors": transactional_gravity_intake["primary_source"]["authors"],
                "pdf_sha256": transactional_gravity_intake["primary_source"]["pdf_sha256"],
            },
            "decision": "blocked",
            "first_blocker": transactional_gravity_candidate_action["first_blocker"],
            "equation_preflight_counts": transactional_gravity_intake["synthetic_preflight_counts"],
            "equation_35_normalization_gate": transactional_gravity_cuda[
                "equation_35_normalization_gate"
            ],
            "equation_graph": {
                "counts": transactional_gravity_graph["graph_counts"],
                "graph_sha256": transactional_gravity_graph["graph_sha256"],
                "decision": transactional_gravity_graph["decision"],
                "first_blocker": transactional_gravity_graph["first_blocker"],
            },
            "cuda_consequence_campaign": {
                "decision": transactional_gravity_cuda["decision"],
                "first_blocker": transactional_gravity_cuda["first_blocker"],
                "counts": transactional_gravity_cuda["counts"],
                "poisson": transactional_gravity_cuda["poisson_four_volume_control"],
                "sds": transactional_gravity_cuda["sds_root_domain_control"],
                "mond_btfr": transactional_gravity_cuda["mond_btfr_control"],
                "gpu_cpu_bindings": transactional_gravity_cuda["gpu_cpu_bindings"],
                "runtime_measurement": transactional_gravity_cuda["runtime_measurement"],
            },
            "observational_readiness": {
                "decision": transactional_gravity_observational["decision"],
                "registration_counts": transactional_gravity_observational["registration_counts"],
                "lane_decisions": transactional_gravity_observational["lane_decisions"],
                "observational_access_count": transactional_gravity_observational[
                    "observational_access_count"
                ],
                "real_data_bundle_count": transactional_gravity_observational[
                    "real_data_bundle_count"
                ],
                "real_data_pass_count": transactional_gravity_observational["real_data_pass_count"],
                "theory_or_ontology_pass_count": transactional_gravity_observational[
                    "theory_or_ontology_pass_count"
                ],
                "contract_scope": transactional_gravity_observational["contract_scope"],
                "data_seals": transactional_gravity_observational["data_seals"],
            },
            "cuda_falsification_design": {
                "decision": transactional_gravity_falsification["decision"],
                "counts": transactional_gravity_falsification["counts"],
                "poisson_power_control": transactional_gravity_falsification[
                    "poisson_power_control"
                ],
                "btfr_power_control": transactional_gravity_falsification["btfr_power_control"],
                "gpu_cpu_crosscheck": transactional_gravity_falsification["gpu_cpu_crosscheck"],
                "observational_bridge": transactional_gravity_falsification["observational_bridge"],
                "runtime_measurement": transactional_gravity_falsification["runtime_measurement"],
                "workload_priority": transactional_gravity_falsification["workload_priority"],
                "synthetic_only": transactional_gravity_falsification["synthetic_only"],
                "scientific_test_pass": transactional_gravity_falsification["scientific_test_pass"],
            },
            "candidate_action_completion": {
                "decision": transactional_gravity_candidate_action["decision"],
                "first_blocker": transactional_gravity_candidate_action["first_blocker"],
                "counts": transactional_gravity_candidate_action["counts"],
                "completion_hypotheses": transactional_gravity_candidate_action[
                    "completion_hypotheses"
                ],
                "dimensions": transactional_gravity_candidate_action["dimensions"],
                "secondary_blockers": transactional_gravity_candidate_action["secondary_blockers"],
                "scope": transactional_gravity_candidate_action["scope"],
            },
            "action_equivalence_audit": {
                "decision": transactional_gravity_equivalence["decision"],
                "first_blocker": transactional_gravity_equivalence["first_blocker"],
                "counts": transactional_gravity_equivalence["counts"],
                "branch_comparison": transactional_gravity_equivalence["branch_comparison"],
                "equivalence_certificates": transactional_gravity_equivalence[
                    "equivalence_certificates"
                ],
                "scope": transactional_gravity_equivalence["scope"],
            },
            "candidate_action_formal_admission": {
                "decision": transactional_gravity_formal["decision"],
                "decision_counts": transactional_gravity_formal["decision_counts"],
                "formal_counts": transactional_gravity_formal["formal_counts"],
                "first_blocker": transactional_gravity_formal["first_blocker"],
                "formal_domain": transactional_gravity_formal["formal_domain"],
                "scope": transactional_gravity_formal["scope"],
            },
            "de_sitter_energy_prerequisite": {
                "decision": transactional_gravity_de_sitter["decision"],
                "decision_counts": transactional_gravity_de_sitter["decision_counts"],
                "prerequisite_counts": transactional_gravity_de_sitter["prerequisite_counts"],
                "first_blocker": transactional_gravity_de_sitter["first_blocker"],
                "declared_charge_framework": transactional_gravity_de_sitter[
                    "declared_charge_framework"
                ],
                "scope": transactional_gravity_de_sitter["scope"],
            },
            "poisson_action_compatibility": {
                "decision": transactional_gravity_poisson_action["decision"],
                "counts": transactional_gravity_poisson_action["counts"],
                "first_blocker": transactional_gravity_poisson_action["first_blocker"],
                "covariant_point_process_contract": transactional_gravity_poisson_action[
                    "covariant_point_process_contract"
                ],
                "mixed_poisson_theorem": transactional_gravity_poisson_action[
                    "mixed_poisson_theorem"
                ],
                "exact_mixed_poisson_control": transactional_gravity_poisson_action[
                    "exact_mixed_poisson_control"
                ],
                "scope": transactional_gravity_poisson_action["scope"],
            },
            "positive_intensity_preservation": {
                "decision": transactional_gravity_positive_intensity["decision"],
                "decision_counts": transactional_gravity_positive_intensity["decision_counts"],
                "gate_counts": transactional_gravity_positive_intensity["gate_counts"],
                "first_blocker": transactional_gravity_positive_intensity["first_blocker"],
                "witness_domain": transactional_gravity_positive_intensity["witness_domain"],
                "secondary_blockers": transactional_gravity_positive_intensity[
                    "secondary_blockers"
                ],
                "scope": transactional_gravity_positive_intensity["scope"],
            },
            "positive_reparameterization": {
                "decision": transactional_gravity_positive_reparameterization["decision"],
                "decision_counts": transactional_gravity_positive_reparameterization[
                    "decision_counts"
                ],
                "gate_counts": transactional_gravity_positive_reparameterization["gate_counts"],
                "first_blocker": transactional_gravity_positive_reparameterization["first_blocker"],
                "field_space_contract": transactional_gravity_positive_reparameterization[
                    "field_space_contract"
                ],
                "secondary_blockers": transactional_gravity_positive_reparameterization[
                    "secondary_blockers"
                ],
                "scope": transactional_gravity_positive_reparameterization["scope"],
            },
            "covariant_point_process_measure": {
                "decision": transactional_gravity_point_process_measure["decision"],
                "decision_counts": transactional_gravity_point_process_measure["decision_counts"],
                "gate_counts": transactional_gravity_point_process_measure["gate_counts"],
                "first_blocker": transactional_gravity_point_process_measure["first_blocker"],
                "measure_domain": transactional_gravity_point_process_measure["measure_domain"],
                "minimal_covariant_probability_measure_contract": (
                    transactional_gravity_point_process_measure[
                        "minimal_covariant_probability_measure_contract"
                    ]
                ),
                "exact_nonidentifiability_witness": transactional_gravity_point_process_measure[
                    "exact_nonidentifiability_witness"
                ],
                "scope": transactional_gravity_point_process_measure["scope"],
            },
            "poisson_selector_contract": {
                "decision": transactional_gravity_poisson_selector["decision"],
                "decision_counts": transactional_gravity_poisson_selector["decision_counts"],
                "gate_counts": transactional_gravity_poisson_selector["gate_counts"],
                "first_blocker": transactional_gravity_poisson_selector["first_blocker"],
                "minimal_Poisson_selector_contract": (
                    transactional_gravity_poisson_selector["minimal_Poisson_selector_contract"]
                ),
                "registered_dependency_audit": transactional_gravity_poisson_selector[
                    "registered_dependency_audit"
                ],
                "scalar_marginal_nonimplication_theorem": (
                    transactional_gravity_poisson_selector["scalar_marginal_nonimplication_theorem"]
                ),
                "scope": transactional_gravity_poisson_selector["scope"],
            },
            "conditional_poisson_kernel_completion": {
                "decision": transactional_gravity_conditional_poisson["decision"],
                "decision_counts": transactional_gravity_conditional_poisson["decision_counts"],
                "gate_counts": transactional_gravity_conditional_poisson["gate_counts"],
                "first_blocker": transactional_gravity_conditional_poisson["first_blocker"],
                "conditional_Poisson_kernel_contract": (
                    transactional_gravity_conditional_poisson["conditional_Poisson_kernel_contract"]
                ),
                "conditional_kernel_domain": transactional_gravity_conditional_poisson[
                    "conditional_kernel_domain"
                ],
                "scope": transactional_gravity_conditional_poisson["scope"],
            },
            "actualization_history_map_audit": {
                "decision": transactional_gravity_actualization_history["decision"],
                "decision_counts": transactional_gravity_actualization_history["decision_counts"],
                "gate_counts": transactional_gravity_actualization_history["gate_counts"],
                "first_blocker": transactional_gravity_actualization_history["first_blocker"],
                "paper_evidence_ledger": transactional_gravity_actualization_history[
                    "paper_evidence_ledger"
                ],
                "compiler_conditional_count_map": (
                    transactional_gravity_actualization_history["compiler_conditional_count_map"]
                ),
                "scope": transactional_gravity_actualization_history["scope"],
            },
            "qed_actualization_poisson_derivation_audit": {
                "decision": transactional_gravity_qed_poisson_derivation["decision"],
                "decision_counts": transactional_gravity_qed_poisson_derivation[
                    "decision_counts"
                ],
                "gate_counts": transactional_gravity_qed_poisson_derivation["gate_counts"],
                "first_blocker": transactional_gravity_qed_poisson_derivation[
                    "first_blocker"
                ],
                "independent_rare_channel_Poisson_limit": (
                    transactional_gravity_qed_poisson_derivation[
                        "independent_rare_channel_Poisson_limit"
                    ]
                ),
                "microscopic_derivation_obligations": (
                    transactional_gravity_qed_poisson_derivation[
                        "microscopic_derivation_obligations"
                    ]
                ),
                "exact_controls": transactional_gravity_qed_poisson_derivation[
                    "exact_controls"
                ],
                "scope": transactional_gravity_qed_poisson_derivation["scope"],
            },
            "deterministic_compensator_admission": {
                "decision": transactional_gravity_deterministic_compensator["decision"],
                "decision_counts": transactional_gravity_deterministic_compensator[
                    "decision_counts"
                ],
                "gate_counts": transactional_gravity_deterministic_compensator[
                    "gate_counts"
                ],
                "first_blocker": transactional_gravity_deterministic_compensator[
                    "first_blocker"
                ],
                "theorem_domain": transactional_gravity_deterministic_compensator[
                    "theorem_domain"
                ],
                "deterministic_compensator_Poisson_characterization": (
                    transactional_gravity_deterministic_compensator[
                        "deterministic_compensator_Poisson_characterization"
                    ]
                ),
                "evidence_gap_ledger": transactional_gravity_deterministic_compensator[
                    "evidence_gap_ledger"
                ],
                "exact_controls": transactional_gravity_deterministic_compensator[
                    "exact_controls"
                ],
                "secondary_blockers": transactional_gravity_deterministic_compensator[
                    "secondary_blockers"
                ],
                "scope": transactional_gravity_deterministic_compensator["scope"],
            },
            "transaction_event_observable_exposure": {
                "decision": transactional_gravity_observable_exposure["decision"],
                "decision_counts": transactional_gravity_observable_exposure["decision_counts"],
                "gate_counts": observable_counts,
                "first_blocker": transactional_gravity_observable_exposure["first_blocker"],
                "minimal_operational_contract": transactional_gravity_observable_exposure[
                    "minimal_operational_contract"
                ],
                "identifiability_theorem": transactional_gravity_observable_exposure[
                    "identifiability_theorem"
                ],
                "scope": transactional_gravity_observable_exposure["scope"],
            },
            "poisson_cox_cuda_power": {
                "decision": transactional_gravity_poisson_cox_power["decision"],
                "counts": transactional_gravity_poisson_cox_power["counts"],
                "design": transactional_gravity_poisson_cox_power["design"],
                "registered_witness_exact_sentinel": (
                    transactional_gravity_poisson_cox_power["registered_witness_exact_sentinel"]
                ),
                "gpu_cpu_crosscheck": transactional_gravity_poisson_cox_power["gpu_cpu_crosscheck"],
                "runtime_measurement": transactional_gravity_poisson_cox_power[
                    "runtime_measurement"
                ],
                "registered_witness_scenario_results": [
                    row
                    for row in transactional_gravity_poisson_cox_power["scenario_results"]
                    if row["null_mean"] == "2" and row["mixing_delta"] == "1/2"
                ],
                "synthetic_only": transactional_gravity_poisson_cox_power["synthetic_only"],
            },
            "set_indexed_cuda_falsification": {
                "decision": transactional_gravity_set_indexed_cuda["decision"],
                "counts": transactional_gravity_set_indexed_cuda["counts"],
                "design": transactional_gravity_set_indexed_cuda["design"],
                "set_indexed_contract": transactional_gravity_set_indexed_cuda[
                    "set_indexed_contract"
                ],
                "exact_common_shock_sentinel": transactional_gravity_set_indexed_cuda[
                    "exact_common_shock_sentinel"
                ],
                "gpu_cpu_crosscheck": transactional_gravity_set_indexed_cuda["gpu_cpu_crosscheck"],
                "runtime_measurement": transactional_gravity_set_indexed_cuda[
                    "runtime_measurement"
                ],
                "synthetic_only": transactional_gravity_set_indexed_cuda["synthetic_only"],
            },
            "set_indexed_gpu_scheduler_adapter": {
                "decision": transactional_gravity_gpu_scheduler_adapter["decision"],
                "scheduler_contract": scheduler_contract,
                "continuous_service_contract": continuous_gpu_contract,
                "execution_state": transactional_gravity_gpu_scheduler_adapter["execution_state"],
                "scientific_test_pass": transactional_gravity_gpu_scheduler_adapter[
                    "scientific_test_pass"
                ],
                "readiness_advanced": transactional_gravity_gpu_scheduler_adapter[
                    "readiness_advanced"
                ],
            },
            "deferred_gpu_ownership": {
                "decision": transactional_gravity_deferred_gpu_ownership["decision"],
                "ownership_contract": deferred_gpu_contract,
                "current_runtime_audit": deferred_gpu_runtime,
                "execution_state": transactional_gravity_deferred_gpu_ownership[
                    "execution_state"
                ],
                "scientific_test_pass": transactional_gravity_deferred_gpu_ownership[
                    "scientific_test_pass"
                ],
                "readiness_advanced": transactional_gravity_deferred_gpu_ownership[
                    "readiness_advanced"
                ],
            },
            "scalar_intensity_cuda_falsification": {
                "decision": transactional_gravity_scalar_cuda["decision"],
                "first_blocker": transactional_gravity_scalar_cuda["first_blocker"],
                "counts": transactional_gravity_scalar_cuda["counts"],
                "linearized_operator": transactional_gravity_scalar_cuda["linearized_operator"],
                "dispersion_control": transactional_gravity_scalar_cuda["dispersion_control"],
                "gpu_cpu_crosscheck": transactional_gravity_scalar_cuda["gpu_cpu_crosscheck"],
                "runtime_measurement": transactional_gravity_scalar_cuda["runtime_measurement"],
                "synthetic_only": transactional_gravity_scalar_cuda["synthetic_only"],
            },
            "extended_geometry_cuda_stress": {
                "decision": transactional_gravity_extended_geometry["decision"],
                "counts": transactional_gravity_extended_geometry["counts"],
                "paper_boundary": transactional_gravity_extended_geometry["paper_boundary"],
                "completion_hypotheses": {
                    "enclosed_mass": transactional_gravity_extended_geometry[
                        "completion_hypotheses"
                    ]["H_enclosed_mass"],
                    "local_superposition": {
                        "decision": transactional_gravity_extended_geometry[
                            "completion_hypotheses"
                        ]["H_local_superposition"]["decision"],
                        "point_mass_aggregation_invariant": transactional_gravity_extended_geometry[
                            "completion_hypotheses"
                        ]["H_local_superposition"]["point_mass_aggregation_invariant"],
                        "coincident_split_controls": transactional_gravity_extended_geometry[
                            "completion_hypotheses"
                        ]["H_local_superposition"]["coincident_split_controls"],
                        "unequal_pair_matter_force_control": transactional_gravity_extended_geometry[
                            "completion_hypotheses"
                        ]["H_local_superposition"]["unequal_pair_matter_force_control"],
                    },
                },
                "lensing_rotation_consistency_gate": transactional_gravity_extended_geometry[
                    "lensing_rotation_consistency_gate"
                ],
                "gpu_cpu_crosscheck": transactional_gravity_extended_geometry["gpu_cpu_crosscheck"],
                "runtime_measurement": transactional_gravity_extended_geometry[
                    "runtime_measurement"
                ],
                "interpretation": transactional_gravity_extended_geometry["interpretation"],
            },
            "claim_seals": {
                "fundamental_action_registered": False,
                "variational_derivation_registered": False,
                "compiler_candidate_action_hypotheses_registered": 2,
                "compiler_candidate_variational_systems_registered": 2,
                "paper_fundamental_action_registered": False,
                "paper_transaction_intensity_dynamics_derived": False,
                "formal_gr_equivalence_proven": False,
                "dark_matter_elimination_proven": False,
                "dark_energy_elimination_proven": False,
                "observational_pass": False,
                "theory_or_ontology_pass": False,
                "automatic_downstream_enqueue_performed": False,
            },
            "data_seals": {
                "synthetic_only": True,
                "observations_opened": False,
                "dark_matter_or_halo_inputs": False,
                "redshift_or_cosmology_inputs": False,
                "paid_llm_calls": False,
            },
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
                        "gpu_synthetic_formula_stress": {
                            "campaign_decision": generated_candidate_formula_gpu_stress[
                                "campaign_decision"
                            ],
                            "counts": generated_candidate_formula_gpu_stress["counts"],
                            "exact_cpu_control": generated_candidate_formula_gpu_stress[
                                "exact_cpu_control"
                            ],
                            "gpu_cpu_comparison": generated_candidate_formula_gpu_stress[
                                "gpu_cpu_comparison"
                            ],
                            "runtime_measurement": generated_candidate_formula_gpu_stress[
                                "runtime_measurement"
                            ],
                            "synthetic_only": True,
                            "formal_pass_inferred": False,
                            "observations_opened": False,
                            "scope": generated_candidate_formula_gpu_stress["scope"],
                            "interpretation": generated_candidate_formula_gpu_stress[
                                "interpretation"
                            ],
                        },
                        "generic_g4_B4_termwise_normalization": {
                            "status": generic_g4_b4_termwise_normalization["status"],
                            "primary_source": {
                                key: generic_g4_b4_termwise_normalization["primary_source"][key]
                                for key in (
                                    "arxiv_id",
                                    "authors",
                                    "equation",
                                    "title",
                                )
                            },
                            "primary_source_transcription": generic_g4_b4_termwise_normalization[
                                "primary_source_transcription"
                            ],
                            "canonical_term_count": generic_g4_b4_termwise_normalization[
                                "canonical_term_count"
                            ],
                            "matched_term_count": generic_g4_b4_termwise_normalization[
                                "matched_term_count"
                            ],
                            "nonzero_residual_count": generic_g4_b4_termwise_normalization[
                                "nonzero_residual_count"
                            ],
                            "metric_variation_normalization_pass": True,
                            "full_candidate_formal_pass_inferred": False,
                            "scope": generic_g4_b4_termwise_normalization["interpretation"],
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
                            "candidate_count": future_aether_characteristic_shell_hcore_gate[
                                "candidate_count"
                            ],
                            "decision_counts": future_aether_characteristic_shell_hcore_gate[
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
                            "weighted_full_constraint_operator_isomorphism_pass_count": future_aether_canonical_seed_constraint_dag_gate[
                                "weighted_Fredholm_isomorphism_pass_count"
                            ],
                            "nonlinear_Frechet_remainder_bound_pass_count": future_aether_canonical_seed_constraint_dag_gate[
                                "nonlinear_remainder_bound_pass_count"
                            ],
                            "completed_boundary_sign_persistence_count": future_aether_canonical_seed_constraint_dag_gate[
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
                            "compact_profile_C3_weighted_jet_bound_pass_count": future_aether_lower_order_coefficient_contract_gate[
                                "compact_profile_C3_weighted_jet_bound_pass_count"
                            ],
                            "lower_order_coefficient_contract_declared_count": future_aether_lower_order_coefficient_contract_gate[
                                "lower_order_coefficient_contract_declared_count"
                            ],
                            "full_canonical_background_point_registered_count": future_aether_canonical_seed_constraint_dag_gate[
                                "full_canonical_background_point_registered_count"
                            ],
                            "candidate_bound_flat_chart_D_residual_DAG_registered_count": future_aether_canonical_seed_constraint_dag_gate[
                                "candidate_bound_flat_chart_D_residual_DAG_registered_count"
                            ],
                            "spatially_distributed_canonical_H_core_registered_count": future_aether_canonical_seed_constraint_dag_gate[
                                "spatially_distributed_canonical_H_core_registered_count"
                            ],
                            "metric_covariantized_H_D_Frechet_DAG_registered_count": future_aether_canonical_seed_constraint_dag_gate[
                                "metric_covariantized_H_D_Frechet_DAG_registered_count"
                            ],
                            "distributed_lower_order_coefficient_registry_complete_count": future_aether_canonical_seed_constraint_dag_gate[
                                "distributed_lower_order_coefficient_registry_complete_count"
                            ],
                            "weighted_relative_lower_order_bound_pass_count": future_aether_canonical_seed_constraint_dag_gate[
                                "weighted_relative_lower_order_bound_pass_count"
                            ],
                            "full_operator_inverse_norm_pass_count": future_aether_canonical_seed_constraint_dag_gate[
                                "full_operator_inverse_norm_pass_count"
                            ],
                            "regular_stratum_flat_chart_H_core_contract_registered_count": future_aether_characteristic_shell_hcore_gate[
                                "regular_stratum_flat_chart_H_core_contract_registered_count"
                            ],
                            "declared_profile_global_flat_chart_H_core_registered_count": future_aether_characteristic_shell_hcore_gate[
                                "declared_profile_global_flat_chart_H_core_registered_count"
                            ],
                            "off_flat_metric_covariantization_registered_count": future_aether_characteristic_shell_hcore_gate[
                                "off_flat_metric_covariantization_registered_count"
                            ],
                            "characteristic_shell_condition": aether_shell_control[
                                "only_real_characteristic_condition"
                            ],
                            "characteristic_shell_rank": aether_shell_control[
                                "declared_profile_characteristic_shell"
                            ]["hessian_rank"],
                            "characteristic_shell_nullity": aether_shell_control[
                                "declared_profile_characteristic_shell"
                            ]["hessian_nullity"],
                            "noncrossing_control_F_squared_margin": aether_shell_control[
                                "noncrossing_profile_control"
                            ]["distance_to_characteristic_F_squared"],
                            "missing_weighted_contract_field_counts": future_aether_weighted_ift_contract_gate[
                                "missing_contract_field_counts"
                            ],
                            "finite_amplitude_nonlinear_constraint_completion_count": future_aether_nonlinear_lift_characteristic_gate[
                                "full_nonlinear_constraint_completion_count"
                            ],
                            "c2_plus_c3_counts": future_aether_weak_field_ae_constraint_gate[
                                "c2_plus_c3_counts"
                            ],
                            "first_blocker_counts": future_aether_characteristic_shell_hcore_gate[
                                "first_blocker_counts"
                            ],
                            "candidate_rejection_authorized_count": future_aether_characteristic_shell_hcore_gate[
                                "candidate_rejection_authorized_count"
                            ],
                        },
                        "g3": {
                            "candidate_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "candidate_count"
                            ],
                            "decision_counts": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
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
                            "radial_momentum_leading_order_pass_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "radial_momentum_leading_order_pass_count"
                            ],
                            "flat_Hamiltonian_leading_order_pass_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "flat_Hamiltonian_leading_order_pass_count"
                            ],
                            "joint_real_asymptotic_coefficient_solution_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "joint_real_asymptotic_coefficient_solution_count"
                            ],
                            "flat_radial_matched_constraint_class_reject_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "flat_radial_matched_constraint_class_reject_count"
                            ],
                            "registered_AF_metric_York_datum_pass_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "registered_AF_metric_York_datum_pass_count"
                            ],
                            "asymptotically_flat_Dirac_pass_count": future_g3_af_transition_obstruction[
                                "AF_unitary_lapse_Dirac_pass_count"
                            ],
                            "AF_Einstein_constraint_solution_pass_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
                            ],
                            "global_energy_pass_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "global_hamiltonian_energy_pass_count"
                            ],
                            "full_formal_pass_count": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
                                "full_formal_pass_count"
                            ],
                            "first_blocker_counts": future_g3_flat_radial_matched_constraints_asymptotic_no_go[
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
                "candidate_obligation_budget": fourth_counts["candidate_fourth_jet_obligations"],
                "obligations_evaluated": (
                    fourth_chunk_counts_0["selected"]
                    + fourth_chunk_counts_32["selected"]
                    + fourth_chunk_counts_64["selected"]
                    + fourth_chunk_counts_96["selected"]
                    + fourth_chunk_counts_128["selected"]
                    + fourth_chunk_counts_160["selected"]
                    + fourth_chunk_counts_192["selected"]
                    + fourth_chunk_counts_224["selected"]
                ),
                "obligations_closed": 244,
                "obligations_remaining": fourth_chunk_counts_224["fourth_obligations_remaining"],
                "candidate_evaluations": (
                    fourth_chunk_counts_0["candidate_evaluations"]
                    + fourth_chunk_counts_32["candidate_evaluations"]
                    + fourth_chunk_counts_64["candidate_evaluations"]
                    + fourth_chunk_counts_96["candidate_evaluations"]
                    + fourth_chunk_counts_128["candidate_evaluations"]
                    + fourth_chunk_counts_160["candidate_evaluations"]
                    + fourth_chunk_counts_192["candidate_evaluations"]
                    + fourth_chunk_counts_224["candidate_evaluations"]
                ),
                "candidate_solvable": (
                    fourth_chunk_counts_0["candidate_solvable"]
                    + fourth_chunk_counts_32["candidate_solvable"]
                    + fourth_chunk_counts_64["candidate_solvable"]
                    + fourth_chunk_counts_96["candidate_solvable"]
                    + fourth_chunk_counts_128["candidate_solvable"]
                    + fourth_chunk_counts_160["candidate_solvable"]
                    + fourth_chunk_counts_192["candidate_solvable"]
                    + fourth_chunk_counts_224["candidate_solvable"]
                ),
                "candidate_obstructed": (
                    fourth_chunk_counts_0["candidate_obstructed"]
                    + fourth_chunk_counts_32["candidate_obstructed"]
                    + fourth_chunk_counts_64["candidate_obstructed"]
                    + fourth_chunk_counts_96["candidate_obstructed"]
                    + fourth_chunk_counts_128["candidate_obstructed"]
                    + fourth_chunk_counts_160["candidate_obstructed"]
                    + fourth_chunk_counts_192["candidate_obstructed"]
                    + fourth_chunk_counts_224["candidate_obstructed"]
                ),
                "directional_evaluations": (
                    fourth_chunk_counts_0["directional_evaluations"]
                    + fourth_chunk_counts_32["directional_evaluations"]
                    + fourth_chunk_counts_64["directional_evaluations"]
                    + fourth_chunk_counts_96["directional_evaluations"]
                    + fourth_chunk_counts_128["directional_evaluations"]
                    + fourth_chunk_counts_160["directional_evaluations"]
                    + fourth_chunk_counts_192["directional_evaluations"]
                    + fourth_chunk_counts_224["directional_evaluations"]
                ),
                "next_obligation_offset": quartic_tc2_fourth_jet_status["next_obligation_offset"],
                "resume_tip_sha256": quartic_tc2_fourth_jet_status["prior_resume_sha256"],
                "parallel_worker_count": 8,
                "permanently_stopped": True,
                "stop_reason": "exact_obstruction",
                "first_exact_obstruction": fourth_obstruction,
                "canonical_obstruction_certificate": {
                    "status": quartic_tc2_d4_obstruction_certificate["status"],
                    "selector_obligations_classified": d4_certificate_counts[
                        "selector_obligations_classified"
                    ],
                    "candidate_specializations_checked": d4_certificate_counts[
                        "candidate_specializations_checked"
                    ],
                    "candidate_compatibilities_certified": d4_certificate_counts[
                        "candidate_compatibilities_certified"
                    ],
                    "candidate_obstructions_certified": d4_certificate_counts[
                        "candidate_obstructions_certified"
                    ],
                    "obligation_offset": quartic_tc2_d4_obstruction_certificate["selector_binding"][
                        "obligation_offset"
                    ],
                    "active_indices": quartic_tc2_d4_obstruction_certificate["selector_binding"][
                        "active_indices"
                    ],
                    "zero_eigenspace_factorization": d4_zero_compression["factorization"],
                    "zero_eigenspace_compression_rank": d4_zero_compression["generic_rank"],
                    "zero_eigenspace_compression_sha256": d4_zero_compression["sha256"],
                    "compatibility_iff_over_Q_or_R": d4_range["compatibility_iff_over_Q_or_R"],
                    "independent_of_c20": d4_range["independent_of_c20"],
                    "exact_candidate_witness_gap": d4_gap["interval"],
                    "alternative_lower_jet_homogeneous_completion_ruled_out": d4_homogeneous_claims[
                        "alternative_lower_jet_homogeneous_completion_ruled_out_for_obligation_244"
                    ],
                    "homogeneous_freedom_reduction": {
                        "status": quartic_tc2_d4_homogeneous_freedom_reduction["status"],
                        "polarization_directions_checked": d4_homogeneous_counts[
                            "polarization_directions_checked"
                        ],
                        "Taylor_orders_per_direction_checked": d4_homogeneous_counts[
                            "Taylor_orders_per_direction_checked"
                        ],
                        "total_exact_zero_projector_checks": d4_homogeneous_counts[
                            "total_exact_zero_projector_checks"
                        ],
                        "lower_jet_reference_kernel_slots_covered_by_identity": d4_homogeneous_counts[
                            "lower_jet_reference_kernel_slots_covered_by_identity"
                        ],
                        "induced_D4_zero_eigenspace_map_rank": d4_homogeneous_reduction[
                            "induced_D4_zero_eigenspace_map_rank"
                        ],
                        "candidate_obstructions_invariant": d4_homogeneous_counts[
                            "candidate_obstructions_invariant"
                        ],
                        "candidate_cancellations": d4_homogeneous_counts["candidate_cancellations"],
                        "exact_identity": d4_homogeneous_reduction["exact_identity"],
                    },
                    "minimal_algebraic_TC2_escape": {
                        "status": quartic_tc2_d4_minimal_tc2_escape["status"],
                        "correction_basis_dimension": d4_escape_counts[
                            "correction_basis_dimension"
                        ],
                        "correction_block_rank": d4_escape_counts["correction_block_rank"],
                        "induced_cokernel_map_rank": d4_escape_counts["induced_cokernel_map_rank"],
                        "target_cokernel_line_dimension": d4_escape_counts[
                            "target_cokernel_line_dimension"
                        ],
                        "candidate_D4_solutions_after_tuning": d4_escape_counts[
                            "candidate_D4_solutions_after_tuning"
                        ],
                        "candidate_D4_obstructions_after_tuning": d4_escape_counts[
                            "candidate_D4_obstructions_after_tuning"
                        ],
                        "distinct_candidate_eta_values": d4_escape["distinct_candidate_eta_values"],
                        "correction_ansatz": {
                            "definition": d4_escape_ansatz["V_definition"],
                            "V_rank": d4_escape_ansatz["V_rank"],
                            "V_nonzero_entries": d4_escape_ansatz["V_nonzero_entries"],
                            "energy_skew_definition": d4_escape_ansatz["energy_skew_definition"],
                            "covariant_or_action_derived": d4_escape_ansatz[
                                "covariant_or_action_derived"
                            ],
                        },
                        "induced_cokernel_map": d4_escape_map,
                        "corrected_candidate_family_registered": d4_escape_claims[
                            "corrected_candidate_family_registered"
                        ],
                        "correction_gauge_constraint_compatible": d4_escape_claims[
                            "correction_gauge_constraint_compatible"
                        ],
                        "scope": quartic_tc2_d4_minimal_tc2_escape["scope"],
                    },
                    "registered_operator_origin_no_go": {
                        "status": quartic_tc2_d4_registered_operator_origin_no_go["status"],
                        "counts": quartic_tc2_d4_registered_operator_origin_no_go["counts"],
                        "declared_operator_class": {
                            key: origin_no_go["declared_operator_class"][key]
                            for key in (
                                "name",
                                "general_block",
                                "domain_dimension_at_one_jet_monomial",
                                "fixed_input_state",
                                "scope_limit",
                            )
                        },
                        "induced_cokernel_map": {
                            key: origin_map[key]
                            for key in (
                                "domain_dimension",
                                "rank",
                                "image_dimension",
                                "augmented_rank",
                                "target_W_rank",
                                "target_W_nonzero_entries",
                                "target_in_image",
                            )
                        },
                        "constraint_support_audit": {
                            key: origin_support[key]
                            for key in (
                                "registered_right_support_columns",
                                "escape_V_right_support_columns",
                                "support_intersection_empty",
                                "zero_projector_rank",
                                "interpretation",
                            )
                        },
                        "sharp_result": origin_no_go["sharp_result"],
                        "scope": quartic_tc2_d4_registered_operator_origin_no_go["scope"],
                    },
                    "topology_changing_origin_classification": {
                        "status": quartic_tc2_d4_topology_changing_origin["status"],
                        "counts": quartic_tc2_d4_topology_changing_origin["counts"],
                        "direct_action_origin_no_go": topology_direct,
                        "explicit_TC2_selector_classification": topology_selectors,
                        "scope": quartic_tc2_d4_topology_changing_origin["scope"],
                    },
                    "curl_constraint_admission": {
                        "status": quartic_tc2_d4_curl_constraint_admission["status"],
                        "counts": curl_counts,
                        "gauge_fixed_operator": curl_admission["gauge_fixed_operator"],
                        "physical_reduction_equivalence": curl_admission[
                            "physical_reduction_equivalence"
                        ],
                        "constraint_propagation": curl_propagation,
                        "coefficient_jet": curl_admission["coefficient_jet"],
                        "reference_D4_binding": curl_admission["reference_D4_binding"],
                        "admission_result": curl_result,
                        "scope": quartic_tc2_d4_curl_constraint_admission["scope"],
                    },
                    "curl_companion_range": {
                        "status": quartic_tc2_d4_curl_companion_range["status"],
                        "counts": companion_counts,
                        "selector_binding": quartic_tc2_d4_curl_companion_range["selector_binding"],
                        "axis_2_reference": companion_audit["axis_2_reference"],
                        "companion_block": companion_audit["companion_block"],
                        "equal_eigenspace_audit": companion_eigenspaces,
                        "rotation_control": companion_audit["rotation_control"],
                        "pure_curl_completion_range": companion_range,
                        "necessary_full_D4_condition": companion_audit[
                            "necessary_full_D4_condition"
                        ],
                        "scope": quartic_tc2_d4_curl_companion_range["scope"],
                    },
                    "axis2_base_D4_RHS": {
                        "status": quartic_tc2_d4_axis2_base_rhs["status"],
                        "counts": axis2_counts,
                        "selector_binding": quartic_tc2_d4_axis2_base_rhs["selector_binding"],
                        "axis_2_reference": axis2_audit["axis_2_reference"],
                        "polarized_base_D4": axis2_base,
                        "companion_correction": axis2_audit["companion_correction"],
                        "result": axis2_result,
                        "claims": axis2_claims,
                        "scope": quartic_tc2_d4_axis2_base_rhs["scope"],
                    },
                    "spatial_gradient_annihilator_no_go": {
                        "status": quartic_tc2_d4_spatial_gradient_no_go["status"],
                        "counts": spatial_no_go_counts,
                        "selector_binding": quartic_tc2_d4_spatial_gradient_no_go[
                            "selector_binding"
                        ],
                        "declared_operator_class": spatial_no_go_exact["declared_operator_class"],
                        "exact_affine_solution": spatial_no_go_exact["exact_affine_solution"],
                        "axis2_projected_range": spatial_no_go_exact["axis2_projected_range"],
                        "candidate_consequence": spatial_no_go_exact["candidate_consequence"],
                        "escape_boundary": spatial_no_go_exact["escape_boundary"],
                        "claims": quartic_tc2_d4_spatial_gradient_no_go["claims"],
                        "scope": quartic_tc2_d4_spatial_gradient_no_go["scope"],
                    },
                    "full_linear_gradient_annihilator_no_go": {
                        "status": quartic_tc2_d4_full_linear_gradient_no_go["status"],
                        "counts": full_no_go_counts,
                        "selector_binding": quartic_tc2_d4_full_linear_gradient_no_go[
                            "selector_binding"
                        ],
                        "declared_operator_class": full_no_go_exact["declared_operator_class"],
                        "full_affine_dimension": full_no_go_exact["full_affine_dimension"],
                        "canonical_qv_selector_partition": qv_partition,
                        "qv_subspace_range": full_no_go_exact["qv_subspace_range"],
                        "c23_subspace_range": full_no_go_exact["c23_subspace_range"],
                        "combined_axis2_free_B2_range": combined_range,
                        "candidate_consequence": full_no_go_exact["candidate_consequence"],
                        "escape_boundary": full_no_go_exact["escape_boundary"],
                        "claims": quartic_tc2_d4_full_linear_gradient_no_go["claims"],
                        "scope": quartic_tc2_d4_full_linear_gradient_no_go["scope"],
                    },
                    "parity_cubic_angular_escape": {
                        "status": quartic_tc2_d4_parity_cubic_escape["status"],
                        "counts": cubic_escape_counts,
                        "selector_binding": quartic_tc2_d4_parity_cubic_escape[
                            "selector_binding"
                        ],
                        "declared_escape_class": cubic_escape["declared_escape_class"],
                        "minimality": cubic_escape["minimality"],
                        "exact_symbol": cubic_symbol,
                        "physical_gradient_lift_equivalence": cubic_escape[
                            "physical_gradient_lift_equivalence"
                        ],
                        "pseudodifferential_constraint_admission": cubic_escape[
                            "pseudodifferential_constraint_admission"
                        ],
                        "two_axis_D4_consequence": cubic_two_axis,
                        "claims": cubic_claims,
                        "scope": quartic_tc2_d4_parity_cubic_escape["scope"],
                    },
                    "parity_cubic_generic_direction_audit": {
                        "status": quartic_tc2_d4_parity_cubic_generic_direction["status"],
                        "counts": generic_direction_counts,
                        "selector_binding": quartic_tc2_d4_parity_cubic_generic_direction[
                            "selector_binding"
                        ],
                        "exact_generic_direction_audit": generic_direction_audit,
                        "claims": generic_direction_claims,
                        "scope": quartic_tc2_d4_parity_cubic_generic_direction["scope"],
                    },
                    "next_gate": quartic_tc2_d4_parity_cubic_generic_direction["next_gate"],
                },
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
            "first_missing_premise": (
                "matrix_valued_or_covariant_local_generic_direction_D4_completion_beyond_rejected_parity_cubic_scalar_escape"
            ),
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
        "unified_live_dashboard_service": unified_live_service,
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
