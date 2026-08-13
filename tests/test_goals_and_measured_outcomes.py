from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/GOALS_AND_MEASURED_OUTCOMES.md"
MATTER_RECEIPT = ROOT / "runs/math/universal-matter-coupled-pde-control-pack/receipt.json"
P55_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-flat-reference-p55-spatial-pencil-registration-gate/campaign.json"
)
P55_RESULT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-p55-checkpointable-materializer/result.json"
)
P55_RECURRENCE_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-p55-registration/"
    "campaign.json"
)
P55_NORMALIZATION_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-candidate-normalization-registration/"
    "campaign.json"
)
P55_SPHERE_REDUCER_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-sphere-normal-form-reducer-registration/"
    "campaign.json"
)
P55_TAYLOR_ZERO_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-p55-taylor-order-zero-registration/"
    "campaign.json"
)
K55_SERIALIZATION_AUDIT_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-k55-taylor-order-zero-serialization-audit/"
    "campaign.json"
)
FLAT_ACTION_METRIC_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-flat-action-metric-registration/campaign.json"
)
BATCH_RECEIPT = ROOT / (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/result.json"
)
NATIVE_RECEIPT = ROOT / "runs/math/native-newton-blind-polynomial-tournament/receipt.json"
MAXWELL_RECEIPT = ROOT / "runs/math/maxwell-hilbert-noether-interface-gate/receipt.json"
MAXWELL_ARBITRARY_RECEIPT = ROOT / (
    "runs/math/maxwell-arbitrary-background-stress-divergence-gate/receipt.json"
)
FLUID_RECEIPT = ROOT / "runs/math/barotropic-irrotational-action-gate/receipt.json"
FLUID_STRESS_RECEIPT = ROOT / (
    "runs/math/barotropic-irrotational-stress-conservation-gate/receipt.json"
)
FLUID_HYPERBOLICITY_RECEIPT = ROOT / (
    "runs/math/barotropic-irrotational-hyperbolicity-gate/receipt.json"
)
FLUID_CONSTRAINT_RECEIPT = ROOT / (
    "runs/math/barotropic-irrotational-constraint-propagation-gate/receipt.json"
)
CURRENT_OPERATIONAL_RECEIPT = ROOT / (
    "runs/engine/current-operational-scratch-recovery-campaign/result.json"
)
COMBINED_MATTER_RECEIPT = ROOT / (
    "runs/math/combined-scalar-maxwell-fluid-gravity-interface-gate/receipt.json"
)
MATTER_COUPLING_CENSUS_RECEIPT = ROOT / (
    "runs/math/quartic-twelve-candidate-matter-coupling-registration-census/receipt.json"
)
TOTAL_MATTER_ACTION_RECEIPT = ROOT / (
    "runs/math/quartic-twelve-candidate-total-matter-action-binding/receipt.json"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_registry_contains_each_goal_once_and_retains_incomplete_boundary() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    goal_numbers = [int(value) for value in re.findall(r"^\| (\d+) \|", text, re.MULTILINE)]
    assert goal_numbers == list(range(1, 42))
    assert "strong alpha; not yet scientifically or operationally complete" in text
    assert "It is not yet a comprehensive theorem-discovery\nsystem" in text
    assert "39/39 successful jobs" in text
    assert "31746356515" in text
    assert "zero failures, zero cancellations, and zero timeouts" in text
    assert "any subsequent change must earn a new terminal 39/39 receipt" in text


def test_matter_and_p55_counts_are_projected_from_checked_receipts() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    matter = _load(MATTER_RECEIPT)
    assert matter["counts"] == {
        "exact_symbolic_replays": 1,
        "floating_point_operations": 0,
        "formal_controls_bound": 6,
        "gate_blocks": 5,
        "gate_passes": 7,
        "gate_rejects": 0,
        "gates": 12,
        "sector_blocks": 2,
        "sector_passes": 1,
        "sector_rejects": 0,
        "sectors": 3,
    }
    assert "combined three-sector gate passes" in text

    p55 = _load(P55_RECEIPT)
    assert p55["counts"]["required_matrix_packets"] == 3
    assert p55["counts"]["missing_matrix_packets"] == 3
    assert p55["counts"]["required_dense_entries"] == 9_075
    assert p55["counts"]["registered_sparse_entries"] == 0
    assert p55["counts"]["minimal_polynomial_entries_reduced"] == 0
    assert "prior BLOCK receipt remains historical evidence" in text

    p55_result = _load(P55_RESULT)
    assert p55_result["counts"]["matrix_packets"] == 3
    assert p55_result["counts"]["sparse_entries"] == 144
    assert p55_result["counts"]["linearity_entries_certified"] == 3_025
    assert p55_result["counts"]["minimal_polynomial_entries_reduced"] == 3_025
    assert p55_result["counts"]["minimal_polynomial_nonzero_remainders"] == 0
    assert p55_result["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "3/3 exact 55x55 P55 axis matrices" in text

    recurrence = _load(P55_RECURRENCE_RECEIPT)
    assert recurrence["status"] == (
        "block_coordinate_free_D4_recurrence_emitter_missing_298_symbolic_packets"
    )
    assert recurrence["counts"]["required_symbolic_input_packets"] == 304
    assert recurrence["counts"]["registered_symbolic_input_packets"] == 6
    assert recurrence["counts"]["missing_symbolic_input_packets"] == 298
    assert recurrence["counts"]["required_output_rows"] == 117_180
    assert recurrence["counts"]["emitted_output_rows"] == 0
    assert recurrence["counts"]["phase_two_solve_attempts"] == 0
    assert recurrence["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "recurrence manifest to 34/304" in text
    assert "0/117,180 coefficient rows" in text

    normalization = _load(P55_NORMALIZATION_RECEIPT)
    assert normalization["status"] == (
        "block_coordinate_free_D4_recurrence_emitter_missing_286_symbolic_packets"
    )
    assert normalization["counts"]["new_candidate_normalization_packets_registered"] == 12
    assert normalization["counts"]["registered_symbolic_input_packets"] == 18
    assert normalization["counts"]["missing_symbolic_input_packets"] == 286
    assert normalization["counts"]["common_shape_factorization_residuals_checked"] == 12
    assert normalization["counts"]["common_shape_factorization_nonzero_residuals"] == 0
    assert normalization["counts"]["emitted_output_rows"] == 0
    assert normalization["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "Candidate normalization, sphere reduction" in text
    sphere = _load(P55_SPHERE_REDUCER_RECEIPT)
    assert sphere["status"] == (
        "block_coordinate_free_D4_recurrence_emitter_missing_285_symbolic_packets"
    )
    assert sphere["counts"]["registered_symbolic_input_packets"] == 19
    assert sphere["counts"]["missing_symbolic_input_packets"] == 285
    assert sphere["counts"]["odd_sphere_remainder_modes"] == 210
    assert sphere["counts"]["basis_unit_replays"] == 210
    assert sphere["counts"]["sphere_generator_multiple_replays"] == 615
    assert sphere["counts"]["nonzero_replay_remainders"] == 0
    assert sphere["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "Candidate normalization, sphere reduction" in text
    taylor_zero = _load(P55_TAYLOR_ZERO_RECEIPT)
    assert taylor_zero["status"] == (
        "block_coordinate_free_D4_recurrence_emitter_missing_270_symbolic_packets"
    )
    assert taylor_zero["counts"]["P55_axis_sparse_entries_consumed"] == 144
    assert taylor_zero["counts"]["new_P55_Taylor_order_zero_packets_registered"] == 15
    assert taylor_zero["counts"]["registered_symbolic_input_packets"] == 34
    assert taylor_zero["counts"]["missing_symbolic_input_packets"] == 270
    assert taylor_zero["counts"]["P55_Taylor_orders_missing"] == 4
    assert taylor_zero["claims"]["P55_Taylor_orders_one_through_four_registered"] is False
    assert "advance the recurrence manifest to 34/304" in text
    assert "BLOCKED on 270" in text
    k55 = _load(K55_SERIALIZATION_AUDIT_RECEIPT)
    assert k55["status"] == "block_K55_Taylor_order_zero_missing_exact_reference_action_metric"
    assert k55["counts"]["K55_named_paths_audited"] == 66
    assert k55["counts"]["exact_sparse_55x55_K55_packets_found"] == 0
    assert k55["counts"]["exact_sparse_22x22_action_metric_packets_found"] == 0
    assert k55["counts"]["registered_symbolic_input_packets"] == 34
    assert k55["claims"]["minimal_K55_order_zero_serialization_contract_closed"] is True
    flat_metric = _load(FLAT_ACTION_METRIC_RECEIPT)
    assert flat_metric["status"] == "pass_exact_flat_action_metric_h_plus_0_registration"
    assert flat_metric["counts"]["A_0_nonzero_entries"] == 10
    assert flat_metric["counts"]["B_0_nonzero_entries"] == 8
    assert flat_metric["counts"]["h_plus_0_nonzero_entries"] == 28
    assert flat_metric["counts"]["full_symbol_build_calls"] == 0
    assert flat_metric["claims"]["flat_action_metric_h_plus_0_registered"] is True
    assert flat_metric["claims"]["K55_Taylor_order_zero_packets_registered"] is False
    assert "symmetric 22x22 `h_plus_0` with 28 nonzero entries" in text


def test_continuous_cursor_counts_and_product_milestones_are_current() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    batch = _load(BATCH_RECEIPT)
    counts = batch["counts"]
    assert counts["cumulative_formally_checked_candidates"] == 226
    assert counts["cumulative_newly_processed_candidates"] == 224
    assert counts["cumulative_reconciled_preserved_candidates"] == 2
    assert counts["cumulative_candidate_rejects"] == 226
    assert counts["remaining_pending_formal_receipts"] == 11_023
    assert "226 checked, 224 new, two reconciled" in text
    assert "11,023 pending" in text

    assert (ROOT / "src/sigma_theory_compiler/formula_discovery_job.py").is_file()
    assert (ROOT / "src/sigma_theory_compiler/formula_discovery_cli.py").is_file()
    assert (ROOT / "docs/formula-discovery-cli-walkthrough.md").is_file()
    assert "Product milestone achieved" in text


def test_native_independent_construction_result_is_measured_and_narrow() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    native = _load(NATIVE_RECEIPT)
    assert native["counts"] == {
        "candidate_blocks": 0,
        "candidate_passes": 1,
        "candidate_rejects": 2,
        "candidates": 3,
        "exact_counterexamples": 2,
        "exact_identity_certificates": 1,
        "floating_point_operations": 0,
        "generator_families": 3,
        "generic_exact_solver_invocations": 0,
        "native_formula_constructions": 1,
        "worlds": 1,
    }
    winner = native["candidate_results"][0]
    assert winner["candidate"]["family"] == "symbolic_newton"
    assert winner["candidate"]["coefficients_constant_first"] == [11, -3, 0, 2, -1]
    assert winner["status"] == "PASS"
    assert "1 PASS, 2 REJECT" in text
    assert "Multi-world, non-polynomial" in text


def test_maxwell_followup_records_profile_success_and_arbitrary_background_block() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    maxwell = _load(MAXWELL_RECEIPT)
    assert maxwell["decision"] == "BOUNDED_PASS_WITH_TYPED_BLOCK"
    assert maxwell["counts"]["exact_noether_residuals"] == 12
    assert maxwell["counts"]["exact_structural_residuals"] == 1
    assert maxwell["counts"]["blocks"] == 1
    assert maxwell["claims"]["dedicated_maxwell_registered_profile_interface_closed"] is True
    assert maxwell["claims"]["dedicated_maxwell_arbitrary_background_interface_closed"] is False
    assert "Maxwell has an arbitrary-background Hilbert-stress identity" in text

    arbitrary = _load(MAXWELL_ARBITRARY_RECEIPT)
    assert arbitrary["decision"] == "PASS_ARBITRARY_BACKGROUND_MAXWELL_STRESS_DIVERGENCE"
    assert arbitrary["counts"]["independent_field_strength_components"] == 6
    assert arbitrary["counts"]["independent_potential_second_jets"] == 40
    assert arbitrary["counts"]["stress_identity_components"] == 4
    assert arbitrary["counts"]["stress_identity_residual_monomials"] == 0
    assert arbitrary["counts"]["negative_residual_components"] == 4
    assert (
        arbitrary["claims"]["arbitrary_background_maxwell_hilbert_stress_divergence_closed"] is True
    )
    assert arbitrary["claims"]["coupled_gravity_matter_pde_closed"] is False
    assert "Maxwell has an arbitrary-background Hilbert-stress identity" in text


def test_fluid_followups_close_action_and_stress_only() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    fluid = _load(FLUID_RECEIPT)
    assert fluid["decision"] == "PASS_EARLIEST_GATE_ONLY"
    assert [item["outcome"] for item in fluid["gate_results"]] == [
        "PASS",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert fluid["exact_replay"]["equation_of_state"] == "p=rho/3"
    assert fluid["claims"]["vortical_flows_covered"] is False
    assert fluid["claims"]["universal_matter_closure_established"] is False
    assert "vortical matter" in text

    stress = _load(FLUID_STRESS_RECEIPT)
    assert stress["decision"] == "PASS_SECOND_GATE_ONLY"
    assert stress["counts"] == {
        "blocks": 0,
        "gates_not_evaluated": 2,
        "negative_controls": 2,
        "new_gates_passed": 1,
        "predecessor_gates": 1,
        "registered_exact_residuals": 4,
        "rejects": 0,
        "sectors": 1,
        "specialized_exact_residual_coefficients": 3,
    }
    assert [item["outcome"] for item in stress["gate_results"]] == [
        "PREDECESSOR_PASS",
        "PASS",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert stress["claims"]["stress_conservation_gate_closed"] is True
    assert stress["claims"]["hyperbolicity_gate_closed"] is False
    assert stress["claims"]["constraint_propagation_gate_closed"] is False
    assert "pass their four bounded gates" in text
    assert "six-component polynomial" in text

    hyperbolicity = _load(FLUID_HYPERBOLICITY_RECEIPT)
    assert hyperbolicity["decision"] == "PASS_THIRD_GATE_ONLY"
    assert hyperbolicity["counts"] == {
        "blocks": 0,
        "exact_registered_residuals": 4,
        "exact_specialized_residuals": 2,
        "gates_not_evaluated": 1,
        "new_gates_passed": 1,
        "predecessor_gates": 2,
        "registered_negative_controls": 4,
        "rejects": 0,
        "sectors": 1,
        "specialized_negative_controls": 2,
    }
    assert hyperbolicity["claims"]["irrotational_matter_hyperbolicity_gate_closed"] is True
    assert hyperbolicity["claims"]["constraint_propagation_gate_closed"] is False
    assert hyperbolicity["claims"]["coupled_gravity_matter_hyperbolicity_established"] is False
    assert "`(|k|^2-omega^2)^5(|k|^2-3 omega^2)`" in text

    constraint = _load(FLUID_CONSTRAINT_RECEIPT)
    assert constraint["decision"] == "PASS_FOURTH_GATE_ZERO_INDEPENDENT_CONSTRAINTS"
    assert constraint["counts"]["independent_primary_matter_constraints"] == 0
    assert constraint["counts"]["independent_matter_gauge_generators"] == 0
    assert constraint["counts"]["definitional_identities_replayed"] == 3
    assert constraint["claims"]["matter_constraint_propagation_gate_closed_not_applicable"] is True
    assert constraint["claims"]["coupled_gravity_matter_constraint_algebra_established"] is False
    assert "irrotational `P(X)=kappa X^2` sector pass" in text

    combined = _load(COMBINED_MATTER_RECEIPT)
    assert combined["decision"] == "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK"
    assert combined["counts"] == {
        "acoustic_cone_components": 1,
        "combined_interface_passes": 4,
        "exact_combined_residuals": 4,
        "gravity_interface_blocks": 1,
        "internal_matter_constraints": 1,
        "light_cone_components": 5,
        "matter_second_order_components": 6,
        "matter_sectors": 3,
        "negative_controls": 2,
        "rejects": 0,
    }
    assert combined["claims"]["combined_three_sector_matter_interface_closed"] is True
    assert combined["claims"]["full_coupled_gravity_matter_principal_system_closed"] is False
    assert combined["claims"]["gravity_constraint_propagation_closed"] is False
    census = _load(MATTER_COUPLING_CENSUS_RECEIPT)
    assert census["decision"] == "TYPED_BLOCK_CENSUS_NO_CANDIDATE_COUPLED_REGISTRATION"
    assert census["counts"] == {
        "candidates_audited": 12,
        "candidates_blocked_at_first_item": 12,
        "candidates_fully_registered": 0,
        "contract_items_audited": 72,
        "contract_passes": 0,
        "prerequisite_passes": 72,
        "rejects": 0,
        "typed_blocks": 72,
    }
    assert census["claims"]["any_candidate_coupled_registration_complete"] is False
    assert "72/72 vacuum and matter-side prerequisites" in text
    total_action = _load(TOTAL_MATTER_ACTION_RECEIPT)
    assert total_action["decision"] == "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY"
    assert total_action["counts"]["total_action_hash_bindings_passed"] == 12
    assert total_action["counts"]["unique_gravity_action_hashes"] == 12
    assert total_action["counts"]["unique_total_action_hashes"] == 12
    assert total_action["counts"]["omitted_fluid_hash_negatives_passed"] == 12
    assert total_action["counts"]["sourced_euler_bindings_passed"] == 0
    assert total_action["claims"]["all_twelve_total_actions_compositionally_hash_bound"] is True
    assert total_action["claims"]["sourced_gauge_fixed_euler_bound_to_total_action"] is False
    assert "12 unique total-action hashes" in text
    assert "Sourced Euler insertion is the next typed BLOCK" in text


def test_current_scratch_recovery_is_measured_without_production_freshness_claim() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    receipt = _load(CURRENT_OPERATIONAL_RECEIPT)
    assert receipt["decision"] == (
        "pass_current_resources_admitted_isolated_three_task_recovery_complete"
    )
    assert receipt["resource_admission"]["all_samples_admitted"] is True
    assert receipt["resource_admission"]["maximum_cpu_utilization_percent"] == "5.7"
    assert receipt["resource_admission"]["minimum_available_ram_mib"] == 73_443
    assert receipt["scratch_recovery"]["tasks_admitted"]["accepted"] == 3
    assert receipt["scratch_recovery"]["recovery"] == {"failed": 0, "recovered": 1}
    assert receipt["scratch_recovery"]["terminal_counts"] == {"succeeded": 3}
    assert [item["attempt"] for item in receipt["scratch_recovery"]["completed"]] == [2, 1, 1]
    assert receipt["scratch_recovery"]["checkpoint"]["sequence"] == 1
    assert receipt["claims"]["live_sqlite_opened"] is False
    assert receipt["claims"]["production_scheduler_freshness_established"] is False
    assert "attempts `[2,1,1]`" in text
    assert "production freshness still open" in text
