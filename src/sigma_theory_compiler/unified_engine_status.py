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
    parameter_expansion = sources["grammar_parameter_cell_expansion_service"]
    grammar_seed_manifest = sources["grammar_v3_seed_manifest"]
    parameter_manifest = sources["grammar_parameter_cell_manifest"]
    parameter_compilation = sources["grammar_parameter_cell_compilation"]
    formal_preflight = sources["grammar_v3_formal_preflight"]
    promotion_admission = sources["grammar_v3_promotion_admission"]
    g2_candidate_formal = sources["grammar_v3_g2_candidate_formal"]
    g3_candidate_formal = sources["grammar_v3_g3_candidate_formal"]
    aether_candidate_formal = sources["grammar_v3_aether_candidate_formal"]
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
                "pass": 0,
                "reject": 2,
                "block": 161,
            },
            "scalable_admitted_family_formal_outcomes": {
                "pass": 0,
                "reject": 2,
                "block": 160,
            },
            "scalable_preflight_blocked_excluded_count": 1,
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
                                    "blocker_counts": g2_candidate_formal[
                                        "blocker_counts"
                                    ],
                                    "candidate_count": g2_candidate_formal[
                                        "candidate_count"
                                    ],
                                    "decision_counts": g2_candidate_formal[
                                        "decision_counts"
                                    ],
                                    "full_formal_pass_count": g2_candidate_formal[
                                        "full_formal_pass_count"
                                    ],
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
                                }
                            },
                        },
                    },
                },
            },
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
    refresh.add_argument("--maximum-output-bytes", type=int, default=1_048_576)
    refresh.add_argument("--disable-gpu-sample", action="store_true")
    refresh.add_argument("--sampled-at-utc")
    dashboard = subparsers.add_parser("export-dashboard", help="render existing JSON as HTML")
    dashboard.add_argument("--project-root", default=".")
    dashboard.add_argument("--snapshot", default="runs/engine/unified-engine-status-refresh.json")
    dashboard.add_argument("--output", default="runs/engine/unified-engine-dashboard.html")
    dashboard.add_argument("--maximum-output-bytes", type=int, default=1_048_576)
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
