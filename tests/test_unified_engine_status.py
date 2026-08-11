from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sigma_theory_compiler.unified_engine_status import (
    build_unified_snapshot,
    load_config,
    main,
)

REPO = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "runs/engine/rust-streaming-billion-status.json",
    "runs/engine/composite-promotion-overlay-production-status.json",
    "runs/engine/grammar-v3-parameter-cell-execution-status.json",
    "runs/engine/grammar-v3-parameter-cell-expansion-service-status.json",
    "runs/engine/covariant-grammar-v3-seed-manifest.json",
    "runs/engine/grammar-v3-parameter-cell-manifest.json",
    "runs/engine/grammar-v3-parameter-cell-compilation-campaign.json",
    "runs/engine/grammar-v3-formal-preflight-status.json",
    "runs/engine/grammar-v3-promotion-admission-status.json",
    "runs/engine/grammar-v3-g2-candidate-formal-status.json",
    "runs/engine/grammar-v3-g3-candidate-formal-status.json",
    "runs/engine/aether-parameter-cell-formal-gate-status.json",
    "runs/engine/grammar-v3-evidence-pareto-report.json",
    "runs/engine/grammar-v3-followup-service-status.json",
    "runs/engine/grammar-v3-followup-queue-status.json",
    "configs/resource_profile_5090.json",
    "runs/engine/llm-formula-proposal-adapter-readiness.json",
    "runs/engine/campaign-llm-proposal-bridge-readiness.json",
    "runs/engine/reviewed-g4-candidate-solar-evaluator-readiness.json",
    "runs/engine/grammar-v3-g4-solar-reviewed-execution-status.json",
    "runs/engine/reviewed-g4-candidate-galaxy-evaluator-readiness.json",
    "runs/engine/grammar-v3-g4-galaxy-reviewed-execution-status.json",
    "runs/engine/typed-dsl-campaign-admission-readiness.json",
    "runs/engine/compiler-receipt-registry-bridge-readiness.json",
    "runs/engine/reviewed-local-formula-epoch-status.json",
    "runs/engine/reviewed-local-formula-service-readiness.json",
    "runs/engine/g4-scalar-free-galaxy-forward-model.json",
    "runs/engine/g4-galaxy-branch-distance-registration.json",
    "runs/engine/g4-galaxy-calibration-evaluation-registration.json",
    "runs/engine/g4-galaxy-prediction-contract-transform-registration.json",
    "runs/engine/g4-galaxy-manifest-bundle-tooling-readiness.json",
    "runs/engine/g4-galaxy-source-registry-admission-readiness.json",
]
LABELS = [
    "billion_streaming",
    "promotion_overlay",
    "grammar_parameter_cells",
    "grammar_parameter_cell_expansion_service",
    "grammar_v3_seed_manifest",
    "grammar_parameter_cell_manifest",
    "grammar_parameter_cell_compilation",
    "grammar_v3_formal_preflight",
    "grammar_v3_promotion_admission",
    "grammar_v3_g2_candidate_formal",
    "grammar_v3_g3_candidate_formal",
    "grammar_v3_aether_candidate_formal",
    "evidence_pareto",
    "followup_service",
    "followup_queue",
    "resource_profile",
    "llm_proposal_adapter",
    "llm_campaign_bridge",
    "g4_solar_evaluator",
    "g4_solar_execution",
    "g4_galaxy_evaluator",
    "g4_galaxy_execution",
    "typed_dsl_admission",
    "compiler_registry_bridge",
    "reviewed_local_formula_epoch",
    "reviewed_local_formula_service",
    "g4_galaxy_forward_model",
    "g4_galaxy_branch_distance",
    "g4_galaxy_calibration_evaluation",
    "g4_galaxy_prediction_contract_transform",
    "g4_galaxy_manifest_bundle_tooling",
    "g4_galaxy_source_registry_admission",
]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    specs = []
    for label, rel in zip(LABELS, SOURCE_PATHS, strict=True):
        source = REPO / rel
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        raw = target.read_bytes()
        value = json.loads(raw)
        claimed = value.get("content_sha256")
        if claimed is None:
            claimed = hashlib.sha256(_canonical(value)).hexdigest()
        specs.append({
            "label": label,
            "path": rel,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": claimed,
        })
    database = tmp_path / "runs/campaigns/watchdog.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE campaigns (
          campaign_id TEXT, state TEXT, deadline_utc TEXT, max_tasks INTEGER,
          tasks_started INTEGER, tasks_succeeded INTEGER, tasks_failed INTEGER,
          max_cycles INTEGER, cycles_completed INTEGER, stop_reason TEXT
        );
        CREATE TABLE tasks (task_type TEXT, status TEXT);
        CREATE TABLE candidates (status TEXT);
        CREATE TABLE evidence (outcome TEXT);
        CREATE TABLE llm_budgets (
          limit_microusd INTEGER, reserved_microusd INTEGER, spent_microusd INTEGER,
          max_calls INTEGER, calls_started INTEGER, calls_completed INTEGER
        );
        CREATE TABLE events (created_utc TEXT);
        INSERT INTO campaigns VALUES
          ('fixture','active','2026-08-21T00:00:00+00:00',100,4,1,0,8,1,NULL);
        INSERT INTO tasks VALUES ('covariant_lift','queued'),('llm_research','running'),
          ('candidate_dossier','deferred');
        INSERT INTO candidates VALUES ('active'),('rejected'),('deferred');
        INSERT INTO evidence VALUES ('pass'),('reject'),('unresolved');
        INSERT INTO llm_budgets VALUES (500000000,0,1250000,250,2,2);
        INSERT INTO events VALUES ('2026-08-10T20:00:00+00:00');
    """)
    connection.commit()
    connection.close()
    config = {
        "watchdog_database": "runs/campaigns/watchdog.sqlite",
        "watchdog_stale_after_seconds": 1800,
        "sources": specs,
    }
    return tmp_path, config, database


def test_read_only_snapshot_is_deterministic_and_does_not_mutate_database(tmp_path: Path) -> None:
    root, config, database = _fixture(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    sampled_at = datetime(2026, 8, 10, 20, 10, tzinfo=UTC)
    first = build_unified_snapshot(
        root,
        config,
        now_utc=sampled_at,
        physical_gpu={"availability": "available", "utilization_percent": 4.0},
    )
    second = build_unified_snapshot(
        root,
        config,
        now_utc=sampled_at,
        physical_gpu={"availability": "available", "utilization_percent": 99.0},
    )
    after = hashlib.sha256(database.read_bytes()).hexdigest()

    assert before == after
    assert first["core"] == second["core"]
    assert first["core_content_sha256"] == second["core_content_sha256"]
    assert first["volatile"] != second["volatile"]
    watchdog = first["core"]["campaign_watchdog"]
    assert watchdog["read_contract"] == "sqlite_uri_mode_ro_plus_query_only_transaction"
    assert watchdog["candidate_counts"] == {"active": 1, "deferred": 1, "rejected": 1}
    assert watchdog["evidence_outcome_counts"] == {"pass": 1, "reject": 1, "unresolved": 1}
    assert first["core"]["scheduler_lanes"]["llm_research"] == {
        "capacity": 4,
        "running": 1,
        "queued": 0,
        "scheduler_occupancy_fraction": 0.25,
    }
    assert first["volatile"]["campaign_watchdog_freshness"]["stale"] is False
    assert first["core"]["llm"]["spent_usd"] == 1.25
    assert first["core"]["llm"]["proposal_adapter"] == {
        "default_paid_calls_enabled": False,
        "maximum_call_usd": "5.000000",
        "maximum_total_usd": "500.000000",
        "network_calls_made": 0,
        "output_status": "quarantine_until_downstream_validation",
        "paid_spend_usd": "0.000000",
        "status": "ready_disabled_no_network_no_spend",
    }
    assert first["core"]["llm"]["campaign_bridge"] == {
        "admission_callback_configured": False,
        "campaign_task_type": "reviewed_llm_formula_proposal",
        "compiler_tasks_enqueued": 0,
        "default_execution_enabled": False,
        "network_calls_made": 0,
        "paid_spend_usd": "0.000000",
        "raw_body_persistence": False,
        "status": "ready_disabled_quarantine_only",
    }
    assert first["core"]["llm"]["typed_dsl_admission"] == {
        "compiler_queue_task_type": "reviewed_covariant_compiler_admission",
        "default_execution_enabled": False,
        "fixture_expected_counts": {
            "block": 1,
            "enqueue": 1,
            "pass": 1,
            "reject": 9,
        },
        "formula_body_persistence": False,
        "status": "ready_disabled_hash_only",
    }
    assert first["core"]["llm"]["compiler_registry_bridge"] == {
        "candidate_body_persistence": False,
        "default_execution_enabled": False,
        "fixture_expected_counts": {
            "block": 1,
            "dedup": 1,
            "enqueue": 1,
            "pass": 1,
            "reject": 7,
        },
        "next_stage_adapter_registered": False,
        "novelty_claim_allowed": False,
        "status": "ready_disabled_hash_only",
    }
    assert first["core"]["llm"]["reviewed_local_epoch"] == {
        "default_execution_enabled": False,
        "expected_bounded_status": {
            "candidate_count": 1,
            "compiler_receipt_pass_count": 2,
            "decision_counts": {
                "block": 1,
                "dedup": 1,
                "pass": 1,
                "reject": 1,
            },
            "network_calls": 0,
            "next_stage_enqueue_count": 1,
            "paid_spend_usd": "0.000000",
            "policy_pass_count": 1,
            "proposal_quarantine_count": 4,
        },
        "formula_body_persistence": False,
        "network_calls": 0,
        "paid_spend_usd": "0.000000",
        "status": "ready_disabled_bounded_mock_only",
    }
    assert first["core"]["llm"]["reviewed_local_service"] == {
        "budgets": {
            "maximum_attempts": 3,
            "maximum_disk_bytes": 100_000_000,
            "maximum_tasks": 1,
            "maximum_wall_seconds": 120,
        },
        "default_execution_enabled": False,
        "deterministic_export": True,
        "network_allowed": False,
        "paid_spend_usd": "0.000000",
        "status": "ready_disabled_bounded_local_only",
    }
    assert first["core"]["g4_solar_evaluator"] == {
        "candidate_id": "G3-f9c598b70a77ea54009d8f18",
        "decision": "blocked",
        "descriptor_implementation_ready": True,
        "durable_execution": {
            "decision_counts": {"blocked": 1},
            "reviewed_evaluator_invocation_count": 1,
            "task_count": 1,
            "work_state_counts": {"succeeded": 1},
        },
        "filled_registration_hash_count": 1,
        "first_missing_premise": (
            "registered_real_source_interval_and_trace_tail_prediction_bundle"
        ),
        "missing_registration_hash_count": 16,
        "observational_data_opened": False,
        "primary_record_access_count": 0,
        "synthetic_GR_golden_pass_count": 5,
    }
    assert first["core"]["g4_galaxy_evaluator"] == {
        "candidate_id": "G3-f9c598b70a77ea54009d8f18",
        "decision": "blocked",
        "descriptor_implementation_ready": True,
        "durable_execution": {
            "decision_counts": {"blocked": 1},
            "reviewed_evaluator_invocation_count": 1,
            "task_count": 1,
            "work_state_counts": {"succeeded": 1},
        },
        "filled_registration_hash_count": 1,
        "first_missing_premise": "registered_action_bound_galaxy_prediction_bundle",
        "missing_registration_hash_count": 17,
        "object_specific_gravity_parameter_count": 0,
        "observational_data_opened": False,
        "prediction_bundle_registered": False,
        "primary_record_access_count": 0,
        "synthetic_control_decisions": {"covariance": "pass", "shape": "pass"},
        "forward_model": {
            "analytic_known_answer_pass_count": 3,
            "covariance_control": "pass",
            "decision": "blocked",
            "filled_registration_hash_count": 3,
            "first_missing_premise": "registered_baryonic_source_and_data_contracts",
            "missing_registration_hash_count": 15,
            "newly_filled_fields": [
                "lensing_prediction_implementation_sha256",
                "rotation_prediction_implementation_sha256",
            ],
            "object_specific_gravity_parameter_count": 0,
            "observational_data_opened": False,
            "prediction_bundle_registered": False,
        },
        "registration": {
            "branch_contract_status": "certified_exact_conditional_branch",
            "decision": "blocked",
            "distance_geometry_contract_status": (
                "certified_interface_no_real_values"
            ),
            "filled_registration_hash_count": 11,
            "first_missing_premise": (
                "registered_real_source_manifest_and_selected_primary_roots"
            ),
            "missing_registration_hash_count": 7,
            "newly_filled_fields": [
                "prediction_bundle_contract_sha256",
                "raw_to_calibrated_transform_sha256",
            ],
            "held_out_split_policy_registered_as_evidence": True,
            "object_specific_gravity_parameter_count": 0,
            "observational_data_opened": False,
            "prediction_bundle_registered": False,
            "real_source_geometry_registered": False,
            "real_split_commitment_registered": False,
            "real_transform_inputs_registered": False,
            "source_specific_branch_selection_proven": False,
            "manifest_bundle_tooling": {
                "decision": "blocked",
                "enabled": False,
                "filled_registration_hash_count": 11,
                "first_missing_premise": (
                    "external_registered_source_manifest_and_independent_registry_receipt"
                ),
                "missing_registration_hash_count": 7,
                "newly_filled_fields": [],
                "synthetic_bundle_registration_admissible": False,
                "synthetic_manifest_registration_admissible": False,
            },
            "source_registry_admission": {
                "decision": "blocked",
                "enabled": False,
                "filled_registration_hash_count": 11,
                "first_missing_premise": (
                    "explicit_registered_source_opening_authorization"
                ),
                "missing_registration_hash_count": 7,
                "newly_filled_fields": [],
                "source_opening_permission_registered": False,
                "source_records_admitted": 0,
                "target_records_opened": 0,
            },
        },
    }
    assert "C:\\" not in json.dumps(first)


def test_stage_counts_and_missing_evaluator_blockers_are_not_collapsed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    result = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 22, tzinfo=UTC),
        physical_gpu={"availability": "unavailable", "reason": "fixture"},
    )
    core = result["core"]
    assert core["billion_formula_streaming"]["sampled_static_stage"]["pass"] == 5855
    assert core["billion_formula_streaming"]["sampled_static_stage"]["normalized_outcomes"] == {
        "pass": 5855,
        "reject": None,
        "block": 0,
    }
    assert core["promotion_overlay"]["formal"] == {"pass": 0, "reject": 70, "block": 0}
    assert core["grammar_parameter_cells"]["seed_execution"] == {
        "candidate_universe": "six reviewed deterministic seed actions",
        "deadline": "bounded_completed_artifact_no_live_deadline",
        "maximum_tasks": 6,
        "next_scaling_hook": (
            "a new hash-reviewed campaign result must register additional parameter "
            "cells before this finite range may expand beyond six"
        ),
        "normalized_scientific_outcomes": {"pass": 0, "reject": 0, "block": 6},
        "scientific_decision_counts": {"blocked": 6},
        "task_state_counts": {"succeeded": 6},
    }
    assert core["grammar_parameter_cells"]["scalable_unique_action_formal_outcomes"] == {
        "pass": 0,
        "reject": 2,
        "block": 161,
    }
    assert core["grammar_parameter_cells"]["scalable_admitted_family_formal_outcomes"] == {
        "pass": 0,
        "reject": 2,
        "block": 160,
    }
    assert core["grammar_parameter_cells"]["scalable_preflight_blocked_excluded_count"] == 1
    assert core["grammar_parameter_cells"]["expansion_service"] == {
        "chunk_count": 3,
        "decision_counts": {"blocked": 6},
        "parameter_cell_count": 6,
        "scientific_scope": (
            "execution scaling only; no cells beyond the reviewed manifest are inferred"
        ),
        "work_state_counts": {"succeeded": 3},
    }
    reviewed_manifest = core["grammar_parameter_cells"]["reviewed_manifest"]
    assert reviewed_manifest["parameter_cell_count"] == 256
    assert reviewed_manifest["chunk_count"] == 8
    assert reviewed_manifest["family_cell_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 32,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 64,
    }
    assert reviewed_manifest["formal_evaluation_performed"] is False
    assert reviewed_manifest["scientific_decision_counts"] == {}
    assert reviewed_manifest["compilation"] == {
        "candidate_decision_counts": {"blocked": 0, "pass": 163, "reject": 0},
        "compiled_action_ir_count": 256,
        "equivalent_duplicate_count": 93,
        "expensive_formal_campaign_run": False,
        "formal_decision_counts": {},
        "unique_candidate_count": 163,
        "formal_preflight": {
            "candidate_count": 163,
            "decision_counts": {"blocked": 1, "pass": 162},
            "expensive_adm_or_global_energy_run": False,
            "family_decision_counts": {
                "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
                "CONFORMAL_G4_PHI_SCALAR_TENSOR": {"blocked": 1},
                "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
                "KESSENCE_G2_CONVEX": {"pass": 2},
            },
            "gate_counts": {
                "family_prerequisite": {"blocked": 1, "pass": 162},
                "receipt_binding": {"pass": 163},
            },
            "next_promotion_hook": (
                "enqueue only preflight-pass candidates into separately reviewed "
                "family-specific ADM/formal campaigns bound to "
                "candidate_id+typed_action_ir_sha256"
            ),
            "work_state_counts": {"succeeded": 163},
            "promotion_admission": {
                "decision_counts": {"pass": 162},
                "downstream_expensive_execution_started": False,
                "eligible_candidate_count": 162,
                "preflight_blocked_excluded_count": 1,
                "target_queue_counts": {
                    "grammar_v3_aether_candidate_adm_formal": 128,
                    "grammar_v3_g2_candidate_adm_formal": 2,
                    "grammar_v3_g3_candidate_adm_formal": 32,
                },
                "work_state_counts": {"succeeded": 162},
                "family_formal_execution": {
                    "aether": {
                        "candidate_count": 128,
                        "decision_counts": {"blocked": 126, "reject": 2},
                        "formal_pass_count": 0,
                        "gate_finding_counts": {
                            "finite_characteristic_slicing_present": 121,
                            "finite_negative_local_density_witness": 79,
                            "globally_noncharacteristic_for_finite_unit_tilt": 5,
                            "positive_at_every_finite_tilt_but_no_uniform_gap": 8,
                            "principal_spin0_degeneracy_reject": 2,
                            "uniform_positive_static_local_twist_gap": 39,
                        },
                    },
                    "g2": {
                        "blocker_counts": {
                            "hash_bound_general_nonmaximal_positive_mass_theorem": 2
                        },
                        "candidate_count": 2,
                        "decision_counts": {"blocked": 2},
                        "full_formal_pass_count": 0,
                        "work_state_counts": {"succeeded": 2},
                    },
                    "g3": {
                        "blocker_counts": {
                            "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain": 32
                        },
                        "candidate_count": 32,
                        "decision_counts": {"blocked": 32},
                        "full_formal_pass_count": 0,
                        "gate_counts": {
                            "adm_primary_degeneracy": {"pass": 32},
                            "af_Einstein_constraint_solution": {"blocked": 32},
                            "af_finite_scalar_energy_tail": {"pass": 32},
                            "af_reference_principal_common_cone": {"pass": 32},
                            "af_uniform_lapse_Dirac_invertibility": {"blocked": 32},
                            "all_spatial_covector_directions": {"pass": 32},
                            "candidate_action_preflight_admission_binding": {"pass": 32},
                            "covariant_G2_G3_variation_noether": {"pass": 32},
                            "distributed_Dirac_on_periodic_cell": {"pass": 32},
                            "exact_parameter_cell_and_weak_envelope": {"pass": 32},
                            "full_candidate_lapse_operator_derivation": {"pass": 32},
                            "full_formal_completion": {"blocked": 32},
                            "global_hamiltonian_energy": {"blocked": 32},
                            "periodic_lapse_coercivity_and_zero_mode_exclusion": {"pass": 32},
                            "uniform_local_common_time_and_BSSN_cone": {"pass": 32},
                            "uniform_local_principal_symbol": {"pass": 32},
                        },
                        "work_state_counts": {"succeeded": 32},
                    }
                },
            },
        },
    }
    assert core["evidence_pareto"]["calibration_control_counts"] == {"pass": 13, "reject": 1}
    assert core["followup_service"]["followup_decision_counts"] == {"blocked": 10}
    assert core["followup_service"]["current_missing_evaluator_blockers"] == {
        "g3_global_lapse_dirac_contract": 1,
        "g3_uniform_interval_cell": 1,
        "g4_global_lapse_invertibility": 1,
        "g4_global_positive_energy": 1,
    }
    assert core["cross_pipeline_total"]["status"] == "not_computed"
    assert result["volatile"]["campaign_watchdog_freshness"]["stale"] is True
    assert result["volatile"]["campaign_watchdog_freshness"]["stale_source_reason"]


def test_hash_tamper_fails_closed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    target = root / SOURCE_PATHS[0]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


def test_portable_artifact_core_and_config_are_hash_bound() -> None:
    config = load_config(REPO / "configs/unified_engine_status.json")
    artifact = json.loads((REPO / "runs/engine/unified-engine-status.json").read_text())
    assert config["schema_version"] == "sigma-unified-engine-status-config-1.0"
    assert artifact["core"]["schema_version"] == "sigma-unified-engine-status-1.0"
    assert hashlib.sha256(_canonical(artifact["core"])).hexdigest() == artifact["core_content_sha256"]
    assert artifact["core"]["data_seals"] == {
        "dark_matter_or_halo_inputs": False,
        "observations_opened": False,
        "paid_llm_in_streaming_promotion_grammar": False,
        "redshift_distance_inputs": False,
    }
    live = json.loads(
        (REPO / "runs/engine/unified-engine-status-live-refresh.json").read_text()
    )
    dashboard = (REPO / "runs/engine/unified-engine-dashboard.html").read_text(
        encoding="utf-8"
    )
    assert hashlib.sha256(_canonical(live["core"])).hexdigest() == live[
        "core_content_sha256"
    ]
    assert live["core_content_sha256"] in dashboard
    leaderboards = live["core"]["scientific_leaderboards"]
    assert len(leaderboards["categories"]) == 9
    assert len(leaderboards["history"]) >= 1
    assert all(
        "top10" in category and "full_ranked" in category
        for category in leaderboards["categories"].values()
    )
    assert "Theory formula" in dashboard
    assert "Conformal scalar–tensor gravity" in dashboard
    assert "φ²/100" in dashboard
    assert "Derived operator terms / evidence scope" in dashboard
    assert "Proof and test hierarchy" in dashboard
    assert "How to read a candidate theory" in dashboard
    assert "compact master formula" in dashboard
    assert "solar_prediction_obligation" in dashboard
    assert "LLM budget and proposal quarantine" in dashboard
    assert "quarantine_until_downstream_validation" in dashboard
    assert len(dashboard.encode()) < 131072
    assert "C:\\" not in dashboard


def _write_fixture_config(root: Path, config: dict[str, object]) -> Path:
    body = {
        "schema_version": "sigma-unified-engine-status-config-1.0",
        **config,
    }
    body["content_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path = root / "configs/unified_engine_status.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def test_standalone_refresh_and_dashboard_keep_watchdog_database_read_only(
    tmp_path: Path,
) -> None:
    root, config, database = _fixture(tmp_path)
    _write_fixture_config(root, config)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert main([
        "refresh",
        "--project-root",
        str(root),
        "--output",
        "runs/engine/refreshed.json",
        "--dashboard-output",
        "runs/engine/dashboard.html",
        "--maximum-output-bytes",
        "131072",
        "--disable-gpu-sample",
        "--sampled-at-utc",
        "2026-08-10T20:10:00+00:00",
    ]) == 0
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert before == after

    snapshot_path = root / "runs/engine/refreshed.json"
    dashboard_path = root / "runs/engine/dashboard.html"
    snapshot = json.loads(snapshot_path.read_text())
    dashboard = dashboard_path.read_text(encoding="utf-8")
    assert snapshot["volatile"]["physical_gpu"] == {
        "availability": "disabled",
        "source": "disabled_by_operator",
    }
    assert len(snapshot_path.read_bytes()) < 131072
    assert len(dashboard_path.read_bytes()) < 131072
    assert "Scheduler lanes" in dashboard
    assert "Physical hardware sample" in dashboard
    assert "C:\\" not in dashboard

    assert main([
        "export-dashboard",
        "--project-root",
        str(root),
        "--snapshot",
        "runs/engine/refreshed.json",
        "--output",
        "runs/engine/dashboard-replay.html",
        "--maximum-output-bytes",
        "131072",
    ]) == 0
    assert (root / "runs/engine/dashboard-replay.html").read_bytes() == dashboard_path.read_bytes()


def test_standalone_output_budget_and_path_escape_fail_closed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    _write_fixture_config(root, config)
    output = root / "runs/engine/too-large.json"
    with pytest.raises(RuntimeError, match="bounded JSON output"):
        main([
            "refresh",
            "--project-root",
            str(root),
            "--output",
            "runs/engine/too-large.json",
            "--maximum-output-bytes",
            "4096",
            "--disable-gpu-sample",
        ])
    assert not output.exists()
    with pytest.raises(ValueError, match="escapes project root"):
        main([
            "refresh",
            "--project-root",
            str(root),
            "--output",
            "../escaped.json",
            "--disable-gpu-sample",
        ])
