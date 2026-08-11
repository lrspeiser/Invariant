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
    "runs/engine/g2-scalable-nonmaximal-positive-mass-audit.json",
    "runs/engine/g2-scalable-solar-prediction-readiness.json",
    "runs/engine/g2-solar-heldout-transfer-registration.json",
    "runs/engine/scalable-campaign-staged-epoch-status.json",
    "runs/engine/scalable-future-parameter-chunk-001-status.json",
    "runs/engine/reviewed-future-parameter-formal-preflight-001.json",
    "runs/engine/future-aether-candidate-formal-followup.json",
    "runs/engine/future-aether-constraint-boundary-embedding-audit.json",
    "runs/engine/future-g3-componentwise-domain-contract-campaign.json",
    "runs/engine/future-g3-action-bound-jet-box-campaign.json",
    "runs/engine/future-candidate-action-dossier.json",
    "runs/engine/grammar-v3-g3-candidate-formal-status.json",
    "runs/engine/g4-scalable-action-formal-followup.json",
    "runs/engine/aether-parameter-cell-formal-gate-status.json",
    "runs/engine/scalable-candidate-structural-metrics.json",
    "runs/engine/scalable-candidate-explanation-dossier-bridge.json",
    "runs/engine/grammar-v3-evidence-pareto-report.json",
    "runs/engine/grammar-v3-followup-service-g4-final-status.json",
    "runs/engine/grammar-v3-followup-queue-g4-final-status.json",
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
    "runs/physics-language/quartic-tc2-ck1-p55-tube-envelope-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-quadratic-deltak-extension-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-diagonal-third-jet-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-chunk-campaign/campaign.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/chunks/offset-000064.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/checkpoint.json",
    "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/service-status.json",
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
    "grammar_v3_g2_nonmaximal_positive_mass_followup",
    "grammar_v3_g2_solar_readiness",
    "grammar_v3_g2_solar_heldout_transfer",
    "scalable_campaign_epoch",
    "scalable_future_parameter_chunk",
    "scalable_future_formal_preflight",
    "future_aether_formal_followup",
    "future_aether_constraint_followup",
    "future_g3_domain_followup",
    "future_g3_action_bound_followup",
    "future_candidate_action_dossier",
    "grammar_v3_g3_candidate_formal",
    "grammar_v3_g4_scalable_formal_followup",
    "grammar_v3_aether_candidate_formal",
    "scalable_structural_metrics",
    "scalable_explanation_dossiers",
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
    "quartic_ck1_p55_tube_envelope",
    "quartic_tc2_quadratic_deltak_extension",
    "quartic_tc2_diagonal_third_jet",
    "quartic_tc2_mixed_third_jet_chunk",
    "quartic_tc2_mixed_third_jet_chunk_64",
    "quartic_tc2_mixed_third_jet_checkpoint",
    "quartic_tc2_mixed_third_jet_continuation_status",
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


def test_future_not_before_work_is_scheduled_idle_then_stale(tmp_path: Path) -> None:
    root, config, database = _fixture(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE tasks ADD COLUMN not_before_utc TEXT")
    connection.execute(
        "UPDATE tasks SET not_before_utc = ? "
        "WHERE task_type = 'covariant_lift' AND status = 'queued'",
        ("2026-08-10T21:00:00+00:00",),
    )
    connection.commit()
    connection.close()

    scheduled = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 20, 40, tzinfo=UTC),
        physical_gpu={"availability": "unavailable"},
    )
    overdue = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 21, 40, tzinfo=UTC),
        physical_gpu={"availability": "unavailable"},
    )

    assert scheduled["core"] == overdue["core"]
    assert scheduled["core_content_sha256"] == overdue["core_content_sha256"]
    cpu_before = scheduled["volatile"]["scheduler_readiness"]["cpu_symbolic"]
    assert cpu_before == {
        "queued_total": 1,
        "runnable_now": 0,
        "delayed_until_not_before": 1,
        "earliest_future_not_before_utc": "2026-08-10T21:00:00+00:00",
    }
    freshness_before = scheduled["volatile"]["campaign_watchdog_freshness"]
    assert freshness_before["state"] == "scheduled_idle"
    assert freshness_before["stale"] is False
    assert freshness_before["expected_next_event_not_before_utc"] == (
        "2026-08-10T21:00:00+00:00"
    )
    assert freshness_before["freshness_deadline_utc"] == (
        "2026-08-10T21:30:00+00:00"
    )

    cpu_after = overdue["volatile"]["scheduler_readiness"]["cpu_symbolic"]
    assert cpu_after == {
        "queued_total": 1,
        "runnable_now": 1,
        "delayed_until_not_before": 0,
        "earliest_future_not_before_utc": None,
    }
    freshness_after = overdue["volatile"]["campaign_watchdog_freshness"]
    assert freshness_after["state"] == "stale"
    assert freshness_after["stale"] is True
    assert freshness_after["expected_next_event_not_before_utc"] == (
        "2026-08-10T21:00:00+00:00"
    )
    assert freshness_after["freshness_deadline_utc"] == (
        "2026-08-10T21:30:00+00:00"
    )
    assert freshness_after["stale_source_reason"] == (
        "no_event_by_2026-08-10T21:30:00+00:00"
    )


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
        "pass": 3,
        "reject": 2,
        "block": 158,
    }
    assert core["grammar_parameter_cells"]["scalable_admitted_family_formal_outcomes"] == {
        "pass": 2,
        "reject": 2,
        "block": 158,
    }
    assert core["grammar_parameter_cells"]["scalable_preflight_blocked_excluded_count"] == 1
    assert core["grammar_parameter_cells"][
        "scalable_preflight_blocked_followup_resolved_count"
    ] == 1
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
    future_chunk = core["grammar_parameter_cells"]["staged_epoch"][
        "reviewed_future_chunk"
    ]
    assert {
        key: future_chunk[key]
        for key in (
            "input_cell_count",
            "disposition_counts",
            "preflight",
            "family_followup",
        )
    } == {
        "input_cell_count": 32,
        "disposition_counts": {
            "admitted_new_candidate": 19,
            "deduplicated_existing_candidate": 13,
        },
        "preflight": {
            "candidate_count": 19,
            "decision_counts": {"blocked": 3, "pass": 14, "reject": 2},
            "family_counts": {
                "AETHER_K1234_PARAMETER_CELL": 16,
                "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
            },
            "first_blocker_counts": {
                "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing": 3,
                "nonpositive_spin0_principal_numerator_c123": 2,
            },
            "full_candidate_specific_formal_completion_claimed": False,
            "promotion": {
                "automatic_downstream_enqueue_performed": False,
                "blocked_pending_exact_domain_registration": 3,
                "eligible_for_candidate_specific_formal_queue": 14,
                "rejected_before_candidate_specific_formal_queue": 2,
            },
        },
        "family_followup": {
            "aether": {
                "candidate_count": 14,
                "decision_counts": {"blocked": 14},
                "formal_pass_count": 0,
                "exact_negative_local_twist_witness_count": 14,
                "witness_tilt_squared_counts": {"1": 8, "2": 4, "8": 2},
                "global_tilt_strata_counts": {
                    "finite_characteristic_foliation_present": 13,
                    "globally_noncharacteristic_for_finite_unit_tilt": 1,
                },
                "explicit_affine_ansatz_constraint_reject_count": 14,
                "nonzero_Hamiltonian_constraint_residual_count": 14,
                "nonzero_momentum_constraint_residual_count": 14,
                "undefined_AE_boundary_contribution_count": 14,
                "constraint_satisfying_negative_total_energy_datum_count": 0,
                "first_blocker_counts": {
                    "constraint_satisfying_asymptotically_Euclidean_completion_of_negative_twist_witness": 14
                },
                "candidate_rejection_authorized_count": 0,
            },
            "g3": {
                "candidate_count": 3,
                "decision_counts": {"blocked": 3},
                "all_direction_single_center_pass_count": 3,
                "domain_registration_filled_field_count": 36,
                "domain_registration_missing_field_count": 0,
                "full_Delta_N_derivation_pass_count": 3,
                "nonzero_componentwise_box_pass_count": 3,
                "uniform_principal_common_cone_pass_count": 3,
                "uniform_Delta_N_coercivity_pass_count": 3,
                "periodic_distributed_Dirac_pass_count": 3,
                "asymptotically_flat_Dirac_pass_count": 0,
                "global_energy_pass_count": 0,
                "full_formal_pass_count": 0,
                "first_blocker_counts": {
                    "asymptotically_flat_or_global_energy_domain_missing": 3
                },
            },
        },
    }
    future_dossiers = future_chunk["action_dossiers"]
    assert future_dossiers["candidate_count"] == 19
    assert future_dossiers["decision_counts"] == {"blocked": 17, "reject": 2}
    assert future_dossiers["ranked_candidate_count"] == 0
    assert len(future_dossiers["records"]) == 19
    assert all(
        record["comparison_contract"]["rank"] is None
        and record["comparison_contract"]["rank_eligible"] is False
        and record["action"]["human_readable_action"]["display_kind"]
        == "verbatim_ordered_covariant_density_concatenation"
        for record in future_dossiers["records"]
    )
    structural = core["grammar_parameter_cells"]["structural_metrics"]
    assert structural["candidate_count"] == 163
    assert structural["alias_count"] == 93
    assert structural["measurement_counts"] == {"measured": 163}
    assert structural["formal_decision_counts"] == {
        "blocked": 158,
        "pass": 3,
        "reject": 2,
    }
    assert structural["simplicity_pareto_front"]["candidate_ids"] == [
        "G3A-2f8983c88f504150381064f2",
        "G3A-58e59412e5fe77cd54caf863",
    ]
    assert structural["scientific_validity_inference"] is False
    explanations = core["grammar_parameter_cells"]["explanation_dossiers"]
    assert explanations["candidate_count"] == 163
    assert explanations["alias_count"] == 93
    assert explanations["formal_decision_counts"] == {
        "blocked": 158,
        "pass": 3,
        "reject": 2,
    }
    assert explanations["hierarchy_node_status_counts"] == {
        "blocked": 321,
        "calibration_only": 163,
        "proven": 166,
        "rejected": 2,
    }
    assert explanations["observational_data_opened"] is False
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
                        "predecessor_blocker_counts": {
                            "hash_bound_general_nonmaximal_positive_mass_theorem": 2
                        },
                        "candidate_count": 2,
                        "predecessor_decision_counts": {"blocked": 2},
                        "decision_counts": {"pass": 2},
                        "full_formal_pass_count": 2,
                        "general_nonmaximal_positive_mass_pass_count": 2,
                        "actual_initial_data_set_instantiated": False,
                        "cell_preservation_or_global_evolution_proved": False,
                        "solar_readiness": {
                            "analytic_prediction_pass_count": 2,
                            "conditional_static_source_class_pass_count": 2,
                            "decision_counts": {"blocked": 2},
                            "real_solar_bundle_count": 0,
                            "observational_data_opened": False,
                            "registration_advance": {
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
                            },
                            "held_out_target_access_count": 0,
                            "primary_record_access_count": 0,
                            "real_data_pass_count": 0,
                            "first_missing_premise": (
                                "candidate_specific_real_source_branch_domain_"
                                "instantiation_and_metadata_only_session_split_"
                                "commitment"
                            ),
                        },
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
                    },
                    "g4_followup": {
                        "candidate_count": 1,
                        "decision_counts": {"pass": 1},
                        "equivalent_parameter_cell_alias_count": 32,
                        "formal_followup_decision": "pass",
                        "original_preflight_decision": "blocked",
                        "transfer_method": (
                            "exact_typed_density_projection_and_rational_domain_inclusion"
                        ),
                    },
                },
            },
        },
    }
    assert core["evidence_pareto"]["calibration_control_counts"] == {"pass": 13, "reject": 1}
    assert core["followup_service"]["followup_decision_counts"] == {
        "blocked": 8,
        "pass": 2,
    }
    assert core["followup_service"]["normalized_followup_outcomes"] == {
        "block": 8,
        "pass": 2,
        "reject": 0,
    }
    assert core["followup_service"]["processed"] == 10
    assert core["followup_service"]["deferred"] == 0
    assert core["followup_service"]["current_missing_evaluator_blockers"] == {}
    assert core["quartic_nonlinear_closure"] == {
        "candidate_count": 12,
        "coordinate_pair_partition": {
            "canonical_active_exact_pairs": 861,
            "coverage_complete": True,
            "entrywise_zero_chain_rule_pairs": 8245,
            "excluded_exact_obligations": 2675,
            "global_pair_index_set_sha256": (
                "d300bb318a6475e88d7dfccd6ef4df9ff991e1e1d8cc535ef555c817723168ef"
            ),
            "total_unordered_coordinate_pairs": 11781,
        },
        "quadratic_deltaK_two_jet": {
            "closed_candidate_count": 12,
            "closed_derivative_orders": [0, 1, 2],
            "D2_coordinate_linf_to_Frobenius_ceiling": 16472172,
            "full_tube_Sylvester_identity_closed": False,
        },
        "diagonal_third_jet": {
            "active_direction_count": 41,
            "diagonal_triples_closed": 41,
            "candidate_direction_evaluations": 492,
            "candidate_direction_solvable": 492,
            "candidate_direction_obstructed": 0,
            "full_active_symmetric_triple_count": 12341,
            "remaining_mixed_triples": 12172,
            "mixed_third_jet_closures": 128,
        },
        "mixed_third_jet_chunk": {
            "chunk_offset": 64,
            "latest_chunk_processed_count": 64,
            "processed_count": 128,
            "next_offset": 128,
            "triple_kind_counts": {"ABB": 2, "ABC": 62},
            "symbolic_parameter_compatible": 128,
            "latest_candidate_evaluations": 768,
            "candidate_evaluations": 1536,
            "candidate_solvable": 1536,
            "candidate_obstructed": 0,
            "remaining_mixed_triples": 12172,
            "resume_tip_sha256": (
                "6a179f60d665e2aa73b2e1c07ec36e7ae49490342af7fb14ca305704fda7e191"
            ),
            "service_decision": "checkpointed",
            "full_mixed_sector_closed": False,
        },
        "closure_counts": {
            "full_tube_Sylvester_identities": 0,
            "full_variable_CK1_closures": 0,
            "CK3_closures": 0,
            "TC2_closures": 0,
            "B7_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "first_missing_premise": (
            "remaining_12172_polarized_mixed_third_sylvester_jets_then_"
            "fourth_and_higher_remainder_or_nonlinear_range_theorem"
        ),
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
    assert live["core"]["followup_service"]["followup_decision_counts"] == {
        "blocked": 8,
        "pass": 2,
    }
    assert live["core"]["followup_service"]["deferred"] == 0
    assert live["core"]["followup_service"][
        "current_missing_evaluator_blockers"
    ] == {}
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
    assert "1 rejected, 1 blocked, 1 calibration-only" in dashboard
    assert "Formal decision: pass" in dashboard
    assert "Overall: pass" not in dashboard
    assert "How to read a candidate theory" in dashboard
    assert "compact master formula" in dashboard
    assert "G3A-e0eff4150989e3522dc6ba03" in dashboard
    assert "current exact formal tally is 3 pass, 2 reject, and 158 blocked" in dashboard
    assert "G2 formal passes" in dashboard
    assert "G2 Solar analytic branches" in dashboard
    assert "G2 Solar fields remaining" in dashboard
    assert "Future preflight passes" in dashboard
    assert "two exact Aether principal-mode rejects" in dashboard
    assert "Future Aether blocked" in dashboard
    assert "Future G3 center checks" in dashboard
    assert "Future G3 uniform boxes" in dashboard
    assert "all 14 Aether survivors blocked" in dashboard
    assert "every completion fails both coupled constraints" in dashboard
    assert "pass uniform all-direction principal/common-cone" in dashboard
    assert "asymptotically-flat/global-energy domain" in dashboard
    assert "Staged future candidate formulas (unranked)" in dashboard
    assert "These master actions are recompiled from the exact typed cells" in dashboard
    assert "G3A-8555e529226d13e2e9dacad5" in dashboard
    assert "S = integral d^4x" in dashboard
    assert "blocked and rejected staged actions never enter a scientific ranking" in dashboard
    assert "Future reviewed cells" in dashboard
    assert "Future new candidates" in dashboard
    assert "19 new action classes and 13 exact deduplications" in dashboard
    assert "Quartic nonlinear closure" in dashboard
    assert "Diagonal third jets" in dashboard
    assert "Mixed third jets closed" in dashboard
    assert "128/128 lexicographic AAB/ABB/ABC triples" in dashboard
    assert "Mixed triples remaining" in dashboard
    assert "12,172 polarized mixed triples remain" in dashboard
    assert "Two restart-safe mixed chunks now close 128/128" in dashboard
    assert "CK1, CK3, TC2, B7, global H7, and lifespan remain fail-closed" in dashboard
    assert "No full formal pass is inferred" not in dashboard
    assert "class #1" in dashboard
    assert "g4_global_positive_energy: 1" not in dashboard
    assert "completed in separate evidence classes" in dashboard
    assert "solar_prediction_obligation" in dashboard
    assert "LLM budget and proposal quarantine" in dashboard
    assert "quarantine_until_downstream_validation" in dashboard
    assert len(dashboard.encode()) < 524288
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
        "1048576",
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
    assert len(snapshot_path.read_bytes()) < 1048576
    assert len(dashboard_path.read_bytes()) < 1048576
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
