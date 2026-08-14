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
K55_TAYLOR_ZERO_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-k55-taylor-order-zero-registration/campaign.json"
)
TC2_TAYLOR_ZERO_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-tc2-taylor-order-zero-registration/campaign.json"
)
ORDER_ONE_FRONTIER_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-order-one-serialization-frontier-gate/campaign.json"
)
P55_ORDER_ONE_MATERIALIZER_RESULT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-order-one-p55-checkpointable-materializer/result.json"
)
P55_ORDER_ONE_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-p55-taylor-order-one-registration/campaign.json"
)
COORDINATE_FREE_K0_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k0-directional-lift/campaign.json"
)
COORDINATE_FREE_K0_POLYNOMIAL_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k0-polynomial-packet/campaign.json"
)
COORDINATE_FREE_K55_ORDER_ONE_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-k55-order-one-registration/campaign.json"
)
COORDINATE_FREE_TC2_ORDER_ONE_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-coordinate-free-tc2-order-one-registration/campaign.json"
)
D2_CROSS_DIRECTION_RECEIPT = ROOT / (
    "runs/physics-language/quartic-registered-direction-cross-leaf-d2-replay-gate/campaign.json"
)
D2_COORDINATE_COMPLEMENT_RECEIPT = ROOT / (
    "runs/physics-language/quartic-full-coordinate-tangent-complement-checkpoint-gate/campaign.json"
)
BATCH_RECEIPT = ROOT / (
    "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0005/result.json"
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
SOURCED_METRIC_EULER_RECEIPT = ROOT / (
    "runs/math/quartic-twelve-candidate-sourced-metric-euler-binding/receipt.json"
)
COUPLED_PRINCIPAL_RECEIPT = ROOT / (
    "runs/math/quartic-twelve-candidate-coupled-principal-matrix-gate/receipt.json"
)
MAXWELL_MIXED_PRINCIPAL_RECEIPT = ROOT / (
    "runs/math/quartic-maxwell-metric-mixed-principal-completion-gate/receipt.json"
)
FIRST_ORDER_85_RECEIPT = ROOT / (
    "runs/math/quartic-twelve-candidate-85-state-first-order-reduction/receipt.json"
)
COMMON_TIME_SYMMETRIZER_RECEIPT = ROOT / (
    "runs/math/quartic-twelve-candidate-85-state-common-time-symmetrizer-gate/receipt.json"
)
RESONANT_COMPATIBILITY_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-resonant-projector-compatibility-gate/receipt.json"
)
FULL_SPHERE_RESONANT_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-full-sphere-resonant-compatibility-gate/receipt.json"
)
NONRESONANT_SYLVESTER_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-nonresonant-sylvester-complement-gate/receipt.json"
)
BOUNDED_B_SYMMETRIZER_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-bounded-B-schur-symmetrizer-gate/receipt.json"
)
SOURCED_GRAVITY_CONSTRAINT_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-sourced-gravity-constraint-propagation-gate/receipt.json"
)
GRAVITY_CONSTRAINT_BASIS_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-candidate-gravity-constraint-basis-gate/receipt.json"
)
OFF_SHELL_EULER_DIVERGENCE_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-off-shell-gauge-fixed-euler-divergence-gate/receipt.json"
)
DIFFERENTIATED_GAUGE_READINESS_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-differentiated-gauge-map-readiness/receipt.json"
)
DIFFERENTIATED_GAUGE_MATERIALIZER_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-differentiated-gauge-map-materializer/receipt.json"
)
GAUGE_SCALAR_EXPANSION_RECEIPT = ROOT / (
    "runs/math/quartic-85-state-gauge-map-scalar-coefficient-expansion-gate/receipt.json"
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
    assert "one typed-partition byte-authority failure" in text
    assert "one 30-minute shard timeout" in text
    assert "A new 39/39 clean-clone receipt is mandatory before merge" in text
    assert "75 exact paths" in text


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
    assert "combined three-sector gate passes four matter-interface checks" in text

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
    assert "first advanced the recurrence manifest to 34/304" in text
    assert "registered all 15 K55 order-zero packets" in text
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
    assert "symmetric 22x22 `h_plus_0`" in text
    k55_zero = _load(K55_TAYLOR_ZERO_RECEIPT)
    assert (
        k55_zero["status"]
        == "block_coordinate_free_D4_recurrence_emitter_missing_255_symbolic_packets"
    )
    assert k55_zero["counts"]["new_K55_Taylor_order_zero_packets_registered"] == 15
    assert k55_zero["counts"]["registered_symbolic_input_packets"] == 49
    assert k55_zero["counts"]["missing_symbolic_input_packets"] == 255
    assert k55_zero["counts"]["K0_sparse_entries"] == 131
    assert k55_zero["counts"]["full_symbol_build_calls"] == 0
    assert k55_zero["claims"]["all_15_K55_Taylor_order_zero_packets_registered"] is True
    tc2_zero = _load(TC2_TAYLOR_ZERO_RECEIPT)
    assert (
        tc2_zero["status"]
        == "block_coordinate_free_D4_recurrence_emitter_missing_240_symbolic_packets"
    )
    assert tc2_zero["counts"]["new_TC2_Taylor_order_zero_packets_registered"] == 15
    assert tc2_zero["counts"]["registered_symbolic_input_packets"] == 64
    assert tc2_zero["counts"]["missing_symbolic_input_packets"] == 240
    assert tc2_zero["counts"]["unit_TC2_sparse_linear_coefficients"] == 8
    assert tc2_zero["claims"]["all_15_TC2_Taylor_order_zero_packets_registered"] is True
    assert tc2_zero["claims"]["full_direction_sphere_D4_compatibility_proved"] is False
    assert "manifest to 79/304" in text
    assert "BLOCKED on 195 packets" in text
    frontier = _load(ORDER_ONE_FRONTIER_RECEIPT)
    assert frontier["status"] == "block_coordinate_free_P55_Taylor_order_one_serialization_absent"
    assert frontier["counts"]["target_P55_Taylor_order_one_packets"] == 15
    assert frontier["counts"]["serialized_P55_Taylor_order_one_packets"] == 0
    assert frontier["counts"]["serialized_M0_inverse_packets"] == 0
    assert frontier["counts"]["serialized_M1_packets"] == 0
    assert frontier["counts"]["serialized_E1_packets"] == 0
    assert frontier["counts"]["admissible_coordinate_free_deltaK_order_zero_packets"] == 0
    assert frontier["claims"]["order_one_serialization_frontier_exactly_measured"] is True
    assert frontier["claims"]["P55_Taylor_order_one_registered"] is False
    assert "P_1=M_0^-1(E_1-M_1P_0)" in text
    materializer = _load(P55_ORDER_ONE_MATERIALIZER_RESULT)
    assert materializer["status"] == "pass_exact_15_P55_Taylor_order_one_packets_materialized"
    assert materializer["counts"]["basis_jet_packets"] == 4
    assert materializer["counts"]["basis_axis_matrices"] == 12
    assert materializer["counts"]["P55_Taylor_order_one_packets"] == 15
    assert materializer["counts"]["manifest_registered_after"] == 79
    order_one = _load(P55_ORDER_ONE_RECEIPT)
    assert order_one["counts"]["new_P55_Taylor_order_one_packets_registered"] == 15
    assert order_one["counts"]["registered_symbolic_input_packets"] == 79
    assert order_one["counts"]["missing_symbolic_input_packets"] == 225
    assert order_one["counts"]["P55_Taylor_order_one_distinct_matrix_cells"] == 440
    assert order_one["claims"]["all_15_P55_Taylor_order_one_packets_registered"] is True
    k0_directional = _load(COORDINATE_FREE_K0_RECEIPT)
    assert k0_directional["status"] == "pass_exact_coordinate_free_K0_directional_lift_formula"
    assert k0_directional["counts"]["e1_reference_matrix_entries_compared"] == 3_025
    assert k0_directional["counts"]["e1_reference_matrix_mismatches"] == 0
    assert k0_directional["counts"]["exact_direction_controls_passed"] == 6
    assert k0_directional["claims"]["coordinate_free_K0_directional_lift_formula_constructed"]
    assert k0_directional["claims"]["expanded_55x55_polynomial_K0_packet_emitted"] is False
    assert "basis-free directional lift reproduces all 3,025 sealed e1 entries" in text
    k0_polynomial = _load(COORDINATE_FREE_K0_POLYNOMIAL_RECEIPT)
    assert k0_polynomial["counts"]["K0_polynomial_nonzero_entries"] == 847
    assert k0_polynomial["counts"]["K0_polynomial_normal_form_terms"] == 2_732
    assert k0_polynomial["counts"]["K0_polynomial_maximum_total_degree"] == 6
    assert k0_polynomial["counts"]["sphere_identity_entries_reduced"] == 3_025
    assert k0_polynomial["counts"]["sphere_identity_nonzero_remainders"] == 0
    assert (
        k0_polynomial["claims"]["all_15_K55_order_one_packets_authorized_for_construction"] is True
    )
    assert k0_polynomial["claims"]["K55_Taylor_order_one_registered"] is False
    assert "847 nonzero polynomial entries, 2,732 exact terms" in text
    k55_order_one = _load(COORDINATE_FREE_K55_ORDER_ONE_RECEIPT)
    assert k55_order_one["counts"]["K55_order_one_packets_registered"] == 15
    assert k55_order_one["counts"]["K55_order_one_nonzero_polynomial_entries_total"] == 2_688
    assert k55_order_one["counts"]["K55_order_one_normal_form_terms_total"] == 17_704
    assert k55_order_one["counts"]["differentiated_identity_matrix_entries_reduced"] == 45_375
    assert k55_order_one["counts"]["differentiated_identity_nonzero_remainders"] == 0
    assert k55_order_one["counts"]["manifest_registered_after"] == 94
    assert k55_order_one["counts"]["manifest_missing_after"] == 210
    assert (
        k55_order_one["claims"]["all_15_coordinate_free_K55_Taylor_order_one_packets_registered"]
        is True
    )
    tc2_order_one = _load(COORDINATE_FREE_TC2_ORDER_ONE_RECEIPT)
    assert tc2_order_one["counts"]["TC2_order_one_packets_registered"] == 15
    assert tc2_order_one["counts"]["TC2_order_one_zero_packets"] == 15
    assert tc2_order_one["counts"]["product_rule_nonzero_remainders"] == 0
    assert tc2_order_one["counts"]["manifest_registered_after"] == 109
    assert tc2_order_one["counts"]["manifest_missing_after"] == 195
    assert tc2_order_one["claims"]["all_15_coordinate_free_TC2_Taylor_order_one_packets_registered"]
    assert "The manifest is now 109/304" in text
    assert "BLOCKED on 195 packets" in text

    d2_cross = _load(D2_CROSS_DIRECTION_RECEIPT)
    assert d2_cross["gate_counts"]["new_off_diagonal_records_all_candidates"] == 60_984
    assert d2_cross["gate_counts"]["new_off_diagonal_records_per_candidate"] == 5_082
    assert d2_cross["gate_counts"]["registered_per_candidate"] == 5_324
    assert d2_cross["gate_counts"]["remaining_per_candidate"] == 252_175
    assert d2_cross["claim_seals"]["complete_D2F"] is False
    assert "5,324 of 257,499" in text
    complement = _load(D2_COORDINATE_COMPLEMENT_RECEIPT)
    assert complement["gate_counts"]["existing_formal_direction_records_per_candidate"] == 22
    assert complement["gate_counts"]["existing_unique_coordinate_vectors_per_candidate"] == 20
    assert complement["gate_counts"]["new_coordinate_tangent_certificates_per_candidate"] == 133
    assert complement["gate_counts"]["new_coordinate_tangent_certificates_all_candidates"] == 1_596
    assert complement["gate_counts"]["checkpoint_receipts"] == 96
    assert (
        complement["gate_counts"]["unique_coordinate_vectors_per_candidate_after_extension"] == 153
    )
    assert complement["gate_counts"]["new_D2_entries_registered_per_candidate"] == 0
    assert complement["claim_seals"]["physical_covariant_component_projection_registered"] is False
    assert "all 153 unique coordinate directions are now sealed" in text
    assert "physical coordinate-to-covariant projection" in text


def test_continuous_cursor_counts_and_product_milestones_are_current() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    batch = _load(BATCH_RECEIPT)
    counts = batch["counts"]
    assert counts["cumulative_formally_checked_candidates"] == 314
    assert counts["cumulative_newly_processed_candidates"] == 312
    assert counts["cumulative_reconciled_preserved_candidates"] == 2
    assert counts["cumulative_candidate_rejects"] == 314
    assert counts["remaining_pending_formal_receipts"] == 10_935
    assert "314 checked, 312 new, two reconciled" in text
    assert "10,935 pending" in text

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
    assert "All 12 candidates have unique total-action hashes" in text
    sourced = _load(SOURCED_METRIC_EULER_RECEIPT)
    assert sourced["decision"] == "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY"
    assert sourced["counts"]["sourced_metric_euler_bindings_passed"] == 12
    assert sourced["counts"]["unique_sourced_metric_euler_hashes"] == 12
    assert sourced["counts"]["total_registered_gravity_rows"] == 132
    assert sourced["counts"]["sourced_acceleration_solutions"] == 0
    assert sourced["claims"]["all_twelve_sourced_metric_euler_equations_hash_bound"] is True
    assert sourced["claims"]["full_coupled_principal_system_closed"] is False
    assert "covering 132 gravity rows" in text
    principal = _load(COUPLED_PRINCIPAL_RECEIPT)
    assert principal["decision"] == "TYPED_BLOCK_MISSING_MAXWELL_METRIC_MIXED_PRINCIPAL_BLOCK"
    assert principal["counts"]["determined_entries_per_candidate"] == 249
    assert principal["counts"]["unresolved_entries_per_candidate"] == 40
    assert principal["counts"]["determined_entries_total"] == 2_988
    assert principal["counts"]["unresolved_entries_total"] == 480
    assert principal["counts"]["full_matrices_passed"] == 0
    assert principal["claims"]["full_coupled_principal_matrix_any_candidate"] is False
    assert principal["minimal_registration_contract"]["required_shape"] == [4, 10]
    mixed = _load(MAXWELL_MIXED_PRINCIPAL_RECEIPT)
    assert mixed["decision"] == "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL"
    assert mixed["counts"]["mixed_block_entries"] == 40
    assert mixed["counts"]["completed_17_field_principal_matrices"] == 12
    assert mixed["counts"]["determined_entries_total"] == 3_468
    assert mixed["counts"]["unresolved_entries_total"] == 0
    assert mixed["claims"]["all_twelve_17_field_second_order_principal_matrices_closed"] is True
    assert mixed["claims"]["full_85_state_first_order_reduction_closed"] is False
    assert "completing all 12 coupled 17-field principal matrices" in text
    assert "3,468 globally, zero unresolved" in text
    first_order = _load(FIRST_ORDER_85_RECEIPT)
    assert first_order["decision"] == "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE"
    assert first_order["counts"]["reductions_passed"] == 12
    assert first_order["counts"]["first_order_states_per_candidate"] == 85
    assert first_order["counts"]["first_order_state_entries_total"] == 1_020
    assert first_order["counts"]["lift_residual_entries"] == 0
    assert first_order["claims"]["all_twelve_exact_85_state_first_order_reductions_closed"] is True
    assert first_order["claims"]["full_coupled_symmetrizer_closed"] is False
    assert "closes all 12 exact first-order reductions at 85 states each" in text
    symmetrizer = _load(COMMON_TIME_SYMMETRIZER_RECEIPT)
    assert symmetrizer["decision"] == (
        "TYPED_BLOCK_RESONANT_SYLVESTER_AND_SCHUR_DOMAIN_UNREGISTERED"
    )
    assert symmetrizer["counts"]["vacuum_K55_prerequisites_passed"] == 12
    assert symmetrizer["counts"]["matter_common_time_prerequisites_passed"] == 12
    assert symmetrizer["counts"]["sylvester_unknown_entries_per_candidate"] == 1_650
    assert symmetrizer["counts"]["coupled_symmetrizers_passed"] == 0
    assert symmetrizer["counts"]["typed_blocks"] == 12
    assert symmetrizer["claims"]["nonzero_coupling_witness_diagonalizable"] is True
    assert symmetrizer["claims"]["physical_jordan_obstruction_established"] is False
    resonant = _load(RESONANT_COMPATIBILITY_RECEIPT)
    assert resonant["decision"] == "PASS_EXACT_FLAT_E1_RESONANT_COMPATIBILITY_ALL_B_COMPONENTS"
    assert resonant["counts"]["resonant_component_checks"] == 8
    assert resonant["counts"]["resonant_projection_entries_checked"] == 13_200
    assert resonant["counts"]["resonant_projection_nonzero_entries"] == 0
    assert resonant["counts"]["full_coupled_symmetrizers"] == 0
    assert resonant["claims"]["exact_resonant_compatibility_all_four_potential_components"] is True
    assert resonant["claims"]["all_spatial_directions_closed"] is False
    sphere_resonant = _load(FULL_SPHERE_RESONANT_RECEIPT)
    assert (
        sphere_resonant["decision"]
        == "PASS_EXACT_FULL_SPHERE_RESONANT_COMPATIBILITY_FLAT_REFERENCE"
    )
    assert sphere_resonant["counts"]["full_vacuum_projectors"] == 2
    assert sphere_resonant["counts"]["full_maxwell_cross_coefficients"] == 12
    assert sphere_resonant["counts"]["resonant_sphere_reductions"] == 8
    assert sphere_resonant["counts"]["resonant_normal_form_nonzero_entries"] == 0
    assert sphere_resonant["counts"]["nonresonant_sylvester_solutions"] == 0
    assert sphere_resonant["claims"]["exact_unit_sphere_resonant_compatibility_closed"] is True
    assert sphere_resonant["claims"]["full_coupled_symmetrizer_closed"] is False
    nonresonant = _load(NONRESONANT_SYLVESTER_RECEIPT)
    assert nonresonant["decision"] == "PASS_EXACT_FLAT_SPHERE_NONRESONANT_SYLVESTER_COMPLEMENT"
    assert nonresonant["counts"]["potential_component_solutions"] == 4
    assert nonresonant["counts"]["spectral_block_records"] == 48
    assert nonresonant["counts"]["exact_sylvester_residual_nonzero_entries"] == 0
    assert nonresonant["counts"]["uniform_solution_bounds"] == 4
    assert nonresonant["counts"]["bounded_B_schur_domains"] == 0
    assert nonresonant["claims"]["exact_flat_sphere_nonresonant_sylvester_solution_closed"] is True
    assert nonresonant["claims"]["full_coupled_symmetrizer_closed"] is False
    assert "592 polynomial entries and 1,918 normal-form terms each" in text
    assert "companion-restricted packet serialization is explicitly non-authoritative" in text
    assert "resonant and nonresonant Sylvester equations are closed" in text
    bounded = _load(BOUNDED_B_SYMMETRIZER_RECEIPT)
    assert bounded["decision"] == "PASS_EXACT_FLAT_SPHERE_FULL_SYMMETRIZER_BOUNDED_B"
    assert bounded["counts"]["bounded_nonzero_potential_domains"] == 1
    assert bounded["counts"]["flat_reference_full_symmetrizers"] == 1
    assert bounded["counts"]["full_85_state_symmetry_residual_nonzero_entries"] == 0
    assert bounded["claims"]["bounded_B_uniform_positive_lower_bound_closed"] is True
    assert bounded["claims"]["candidate_jet_uniformity_closed"] is False
    sourced_constraint = _load(SOURCED_GRAVITY_CONSTRAINT_RECEIPT)
    assert sourced_constraint["counts"]["candidates_with_exact_matter_source_cancellation"] == 12
    assert sourced_constraint["counts"]["sourced_gravity_constraint_propagation_passes"] == 0
    basis = _load(GRAVITY_CONSTRAINT_BASIS_RECEIPT)
    assert basis["counts"]["state_coordinates"] == 85
    assert basis["counts"]["physical_gravity_constraint_rows_registered"] == 0
    assert basis["counts"]["physical_gravity_constraint_rows_required"] == 96
    divergence = _load(OFF_SHELL_EULER_DIVERGENCE_RECEIPT)
    assert divergence["counts"]["candidate_common_formula_hashes"] == 12
    assert divergence["claims"]["common_off_shell_covariant_sourced_identity_closed"] is True
    readiness = _load(DIFFERENTIATED_GAUGE_READINESS_RECEIPT)
    assert readiness["counts"]["missing_primitive_jet_families"] == 5
    assert readiness["counts"]["missing_primitive_slots"] == 780
    assert readiness["counts"]["primitive_resume_chunks"] == 48
    assert readiness["claims"]["differentiated_gauge_map_in_85_state_coordinates_closed"] is False
    gauge_materializer = _load(DIFFERENTIATED_GAUGE_MATERIALIZER_RECEIPT)
    assert gauge_materializer["decision"] == (
        "PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS"
    )
    assert gauge_materializer["counts"]["total_primitive_slots"] == 780
    assert gauge_materializer["counts"]["checkpoint_packets"] == 48
    assert gauge_materializer["counts"]["indexed_formula_templates"] == 17
    assert gauge_materializer["counts"]["fully_expanded_85_state_coefficient_rows"] == 0
    assert gauge_materializer["claims"]["exact_indexed_differentiated_gauge_map_closed"]
    assert gauge_materializer["claims"]["external_formulation_jet_values_certified"] is False
    scalar_expansion = _load(GAUGE_SCALAR_EXPANSION_RECEIPT)
    assert scalar_expansion["counts"]["flat_gravity_constraint_rows_expanded"] == 4
    assert scalar_expansion["counts"]["nonzero_scalar_coefficients_total"] == 112
    assert scalar_expansion["counts"]["candidate_flat_row_manifests"] == 12
    assert scalar_expansion["counts"]["required_general_scalar_values_before_domain"] == 1_010
    assert scalar_expansion["claims"]["exact_flat_reference_scalar_coefficient_rows_closed"]
    assert scalar_expansion["claims"]["general_external_jet_scalar_expansion_closed"] is False
    assert "max|B_mu| <= 8/38505" in text
    assert "48/96 candidate-bound gauge rows" in text
    assert "General expansion remains blocked" in text
    assert "all 780 readiness slots in 48 chained packets" in text
    assert "17 exact indexed tensor templates" in text
    assert "four gravity rows into 112 exact coefficients" in text
    assert "complete 1,010-value packet" in text
    assert "This is not a physical Jordan no-go" in text


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
