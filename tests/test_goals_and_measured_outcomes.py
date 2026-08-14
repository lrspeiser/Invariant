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
HIGHER_P55_RESULT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-higher-p55-checkpointable-materializer/result.json"
)
HIGHER_H_STAR_RESULT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-higher-h-star-checkpointable-materializer/result.json"
)
ALTERNATIVE_SYMMETRIZER_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-alternative-symmetrizer-recurrence-audit/campaign.json"
)
ALL_POLARIZATION_ALTERNATIVE_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-all-polarization-alternative-k55-recurrence/campaign.json"
)
INDEPENDENT_G2_ALTERNATIVE_RECEIPT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-independent-g2-alternative-k55-recurrence/campaign.json"
)
SYSTEM9_GLOBAL_H7_RECEIPT = ROOT / (
    "runs/physics-language/quartic-candidate-complete-global-h7-lifespan-gate/campaign.json"
)
SYSTEM9_SUCCESSOR_RECEIPT = ROOT / (
    "runs/physics-language/"
    "quartic-candidate-complete-global-h7-lifespan-system8-successor-gate/campaign.json"
)
D2_CROSS_DIRECTION_RECEIPT = ROOT / (
    "runs/physics-language/quartic-registered-direction-cross-leaf-d2-replay-gate/campaign.json"
)
D2_COORDINATE_COMPLEMENT_RECEIPT = ROOT / (
    "runs/physics-language/quartic-full-coordinate-tangent-complement-checkpoint-gate/campaign.json"
)
D2_PRINCIPAL_PROJECTION_RECEIPT = ROOT / (
    "runs/physics-language/quartic-principal-second-jet-covariant-projection-gate/campaign.json"
)
D2_LOWER_PROJECTION_RECEIPT = ROOT / (
    "runs/physics-language/quartic-lower-coordinate-covariant-projection-gate/campaign.json"
)
D2_REACHABLE_LEAF_COMPLETION_RECEIPT = ROOT / (
    "runs/physics-language/quartic-reachable-leaf-derivative-completion-gate/campaign.json"
)
D2_PGRADIENT_LEAF_RECEIPT = ROOT / (
    "runs/physics-language/quartic-pgradient-abc-leaf-authority-gate/campaign.json"
)
D2_LEAF_SUCCESSOR_CHAIN = (
    (
        "runs/physics-language/quartic-remaining-scalar-hessian-abc-leaf-authority-gate/campaign.json",
        30,
    ),
    ("runs/physics-language/quartic-s03-metric-abc-leaf-authority-gate/campaign.json", 40),
    (
        "runs/physics-language/quartic-s01-metric-complement-abc-leaf-authority-gate/campaign.json",
        49,
    ),
    (
        "runs/physics-language/quartic-s02-metric-complement-abc-leaf-authority-gate/campaign.json",
        58,
    ),
    (
        "runs/physics-language/quartic-s12-metric-complement-abc-leaf-authority-gate/campaign.json",
        67,
    ),
    (
        "runs/physics-language/quartic-s13-metric-complement-abc-leaf-authority-gate/campaign.json",
        76,
    ),
    (
        "runs/physics-language/quartic-s23-metric-complement-abc-leaf-authority-gate/campaign.json",
        85,
    ),
    (
        "runs/physics-language/quartic-s33-metric-complement-abc-leaf-authority-gate/campaign.json",
        93,
    ),
    (
        "runs/physics-language/quartic-s11-metric-complement-abc-leaf-authority-gate/campaign.json",
        99,
    ),
    (
        "runs/physics-language/quartic-s22-metric-complement-abc-leaf-authority-gate/campaign.json",
        105,
    ),
    ("runs/physics-language/quartic-p0-metric-lower-abc-leaf-authority-gate/campaign.json", 115),
    ("runs/physics-language/quartic-p1-metric-lower-abc-leaf-authority-gate/campaign.json", 125),
    ("runs/physics-language/quartic-p2-metric-lower-abc-leaf-authority-gate/campaign.json", 135),
    ("runs/physics-language/quartic-p3-metric-lower-abc-leaf-authority-gate/campaign.json", 145),
)
FORMAL_LOWER_DIRECTION_ALIASES = ("s11[10]", "s22[10]")
Q_LEAF_RECEIPTS = tuple(
    ROOT / path
    for path in (
        "runs/physics-language/quartic-q-metric-lower-abc-leaf-authority-gate/campaign.json",
        "runs/physics-language/quartic-q-metric-leaf-unit-000-gate/campaign.json",
        "runs/physics-language/quartic-q-metric-leaf-candidate0-batch-gate/campaign.json",
        "runs/physics-language/quartic-q-metric-leaf-candidate1-batch-gate/campaign.json",
    )
)
SYSTEM11_SOLAR_AUTHORIZATION_RECEIPT = ROOT / (
    "runs/math/system11-g2-solar-authorization-closure/receipt.json"
)
SYSTEM11_SOLAR_LIKELIHOOD_RECEIPT = ROOT / (
    "runs/math/system11-g2-solar-likelihood-executor-contract/receipt.json"
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
DURABLE_TWO_HOST_CONFIG = ROOT / "configs/durable_two_host_campaign.json"
DURABLE_TWO_HOST_RECEIPT = ROOT / (
    "runs/engine/formula-discovery-durable-two-host-001/duration-receipt.json"
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
SYSTEM10_CYLINDRICAL_ROWS_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-sourced-constraint-row-materializer/receipt.json"
)
SYSTEM10_R_POSITIVE_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-domain-lift/receipt.json"
)
SYSTEM10_PROPAGATION_ATTEMPT_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-constraint-propagation-attempt/receipt.json"
)
SYSTEM10_DIVQ_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-divq-row-materializer/receipt.json"
)
SYSTEM10_KINEMATIC_RHS_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-kinematic-rhs-materializer/receipt.json"
)
SYSTEM10_MATTER_RHS_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-matter-dynamic-rhs-materializer/receipt.json"
)
SYSTEM10_MAXWELL_RHS_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-maxwell-dynamic-rhs-materializer/receipt.json"
)
SYSTEM10_GRAVITY_SCALAR_RHS_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-gravity-scalar-dynamic-rhs-readiness/receipt.json"
)
SYSTEM10_GRAVITY_SCALAR_AW_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-readiness/receipt.json"
)
SYSTEM10_GRAVITY_SCALAR_AW_ROWS = tuple(
    ROOT
    / f"runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-materializer/row-{row:02d}.json"
    for row in range(11)
)
SYSTEM10_GRAVITY_SCALAR_AW_INVERTIBILITY_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-invertibility/receipt.json"
)
SYSTEM10_GRAVITY_SCALAR_AW_TUBE_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-nonsingular-tube/receipt.json"
)
SYSTEM10_TWELVE_CANDIDATE_AW_PACKETS = tuple(
    ROOT
    / f"runs/math/system10-cylindrical-r-positive-twelve-candidate-aw/candidate-{candidate:02d}.json"
    for candidate in range(12)
)
SYSTEM10_TWELVE_CANDIDATE_AW_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-twelve-candidate-aw/receipt.json"
)
SYSTEM10_TWELVE_CANDIDATE_AW_TUBE_SOLUTIONS = tuple(
    ROOT
    / f"runs/math/system10-cylindrical-r-positive-twelve-candidate-aw-tube-solve/solution-{candidate:02d}.json"
    for candidate in range(12)
)
SYSTEM10_TWELVE_CANDIDATE_AW_TUBE_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-twelve-candidate-aw-tube-solve/receipt.json"
)
SYSTEM10_COMMON_TUBE_FULL_RHS_PACKETS = tuple(
    ROOT / f"runs/math/system10-cylindrical-common-tube-full-rhs/candidate-{candidate:02d}.json"
    for candidate in range(12)
)
SYSTEM10_COMMON_TUBE_FULL_RHS_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-common-tube-full-rhs/receipt.json"
)
SYSTEM10_COMMON_TUBE_PROPAGATION_RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-common-tube-propagation-audit/receipt.json"
)
SYSTEM10_OPEN_R_RECEIPTS = {
    name: ROOT / f"runs/math/{slug}/receipt.json"
    for name, slug in {
        "radial": "system10-cylindrical-open-r-twelve-candidate-rhs-jets",
        "tangential": "system10-cylindrical-open-r-tangential-rhs-jets",
        "decomposition": "system10-cylindrical-open-r-coordinate-decomposition",
        "normalization": "system10-cylindrical-open-r-euler-normalization-bridge",
        "factorization": "system10-cylindrical-open-r-divq-c-factorization",
        "initial_data": "system10-cylindrical-open-r-subsidiary-initial-data-map",
    }.items()
}


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
    assert "BLOCKED on 150 packets" in text
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
    assert "basis-free directional lift" in text
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
    higher_p55 = _load(HIGHER_P55_RESULT)
    assert higher_p55["counts"]["P55_higher_packets_registered"] == 45
    assert higher_p55["counts"]["exact_recurrence_matrix_entries_reduced"] == 408_375
    assert higher_p55["counts"]["exact_recurrence_nonzero_remainders"] == 0
    assert higher_p55["counts"]["manifest_registered_after"] == 154
    assert higher_p55["counts"]["manifest_missing_after"] == 150
    higher_h_star = _load(HIGHER_H_STAR_RESULT)
    assert higher_h_star["counts"]["H_star_plus_higher_packets"] == 45
    assert higher_h_star["counts"]["zero_packets_exactly_derived"] == 45
    assert higher_h_star["counts"]["symmetry_remainder_entries"] == 0
    alternative = _load(ALTERNATIVE_SYMMETRIZER_RECEIPT)
    assert alternative["decision"] == "BLOCK_SERIALIZATION"
    assert alternative["counts"]["constructive_witness_local_alternatives"] == 1
    assert alternative["counts"]["global_coordinate_free_alternatives"] == 0
    assert alternative["counts"]["remaining_packets"] == 150
    assert alternative["counts"]["remaining_rows"] == 117_180
    all_polarization = _load(ALL_POLARIZATION_ALTERNATIVE_RECEIPT)
    assert all_polarization["decision"] == "BLOCK_SERIALIZATION"
    assert all_polarization["counts"]["required_evaluations"] == 15
    assert all_polarization["counts"]["evaluations_audited"] == 6
    assert all_polarization["counts"]["evaluations_solved"] == 5
    assert all_polarization["first_exact_incompatibility"]["evaluation_id"] == "subset_02"
    assert all_polarization["first_exact_incompatibility"]["joint_coefficient_rank"] == 4
    assert all_polarization["first_exact_incompatibility"]["joint_augmented_rank"] == 5
    assert all_polarization["counts"]["manifest_registered_after"] == 154
    assert all_polarization["claims"]["manifest_advanced"] is False
    independent_g2 = _load(INDEPENDENT_G2_ALTERNATIVE_RECEIPT)
    assert independent_g2["decision"] == "BLOCK_SERIALIZATION"
    assert independent_g2["counts"]["required_evaluations"] == 15
    assert independent_g2["counts"]["transport_evaluations_audited"] == 15
    assert independent_g2["counts"]["transport_evaluations_solved"] == 15
    assert independent_g2["counts"]["55_state_lifts_passed"] == 2
    assert independent_g2["counts"]["positive_tubes_proved"] == 0
    assert independent_g2["counts"]["manifest_registered_after"] == 154
    assert independent_g2["counts"]["remaining_packets"] == 150
    assert independent_g2["claims"]["all_15_broader_transports_proved"]
    assert independent_g2["claims"]["all_15_exact_55_state_lifts_proved"] is False
    assert independent_g2["first_exact_55_state_lift_failure"] == {
        "K55_symmetrizer_remainder_entries": [0, 0, 72],
        "K55_symmetry_remainder_entries": [0, 0, 0],
        "evaluation_id": "subset_2",
        "first_missing_primitive": (
            "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
        ),
    }
    assert "manifest atomically to 154/304" in text
    assert "all 15/15 transport systems exactly" in text
    assert "symmetrizer remainders are `[0,0,72]`" in text
    assert "BLOCKED on 150 packets" in text
    system9 = _load(SYSTEM9_GLOBAL_H7_RECEIPT)
    assert system9["decision"] == "BLOCK_SYSTEM9"
    assert system9["counts"]["selected_candidates"] == 12
    assert system9["counts"]["candidate_blocks"] == 12
    assert system9["counts"]["closed_global_H7_proofs"] == 0
    assert system9["counts"]["closed_bootstraps"] == 0
    assert system9["counts"]["positive_lifespans"] == 0
    assert system9["counts"]["finite_unmodified_Sobolev_obstructions"] == 12
    assert system9["counts"]["completion_grade_obstructions"] == 0
    assert system9["counts"]["accepted_full_direction_recurrence_evaluations"] == 0
    assert system9["counts"]["missing_full_direction_alternative_evaluations"] == 14
    assert system9["claims"]["completion_grade_obstruction_proved"] is False
    assert (
        system9["exact_remaining_contract"]["first_missing_primitive"]
        == "candidate_bound_full_tensor_source_good_unknown_B7_bound_or_completion_grade_full_direction_obstruction"
    )
    assert "All 12 candidates remain honestly BLOCKED" in text
    assert "not candidate-bound" in text
    system9_successor = _load(SYSTEM9_SUCCESSOR_RECEIPT)
    assert system9_successor["decision"] == "BLOCK_SYSTEM9"
    assert system9_successor["counts"]["selected_candidates"] == 12
    assert system9_successor["counts"]["candidate_blocks"] == 12
    assert system9_successor["counts"]["proved_independent_G2_transport_evaluations"] == 15
    assert system9_successor["counts"]["exact_55_state_lifts_passed"] == 2
    assert system9_successor["counts"]["candidates_upgraded_to_completion_grade"] == 0
    assert system9_successor["counts"]["completion_grade_obstructions_after"] == 0
    assert system9_successor["claims"]["completion_grade_obstruction_proved"] is False
    assert (
        system9_successor["exact_remaining_contract"]["first_System9_completion_primitive"]
        == "candidate_bound_full_tensor_source_good_unknown_B7_bound_or_all_closure_strategy_completion_grade_obstruction"
    )
    assert "upgrades 0/12 candidates to completion-grade" in text

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
    projection = _load(D2_PRINCIPAL_PROJECTION_RECEIPT)
    assert projection["gate_counts"]["principal_second_jet_projection_directions"] == 99
    assert (
        projection["gate_counts"]["new_unique_projection_directions_registered_per_candidate"] == 79
    )
    assert projection["gate_counts"]["remaining_lower_jet_projection_directions"] == 54
    assert projection["gate_counts"]["D2_entries_registered_per_candidate"] == 5_324
    assert projection["claim_seals"]["formal_22_slot_to_20_unique_alias_reconciled"] is True
    assert projection["claim_seals"]["D2_entry_count_advanced"] is False
    assert "all 99 principal second-jet directions" in text
    lower_projection = _load(D2_LOWER_PROJECTION_RECEIPT)
    assert lower_projection["gate_counts"]["lower_projection_directions_registered"] == 54
    assert lower_projection["gate_counts"]["candidate_bound_lower_projection_certificates"] == 648
    assert lower_projection["gate_counts"]["unique_coordinate_directions_projected"] == 153
    assert lower_projection["gate_counts"]["new_D2_entries_registered_per_candidate"] == 0
    assert lower_projection["D1_DAG_audit"]["missing_candidate_bound_leaf_derivatives"] == 31_680
    assert lower_projection["claim_seals"]["all_54_lower_coordinate_directions_projected"]
    assert lower_projection["claim_seals"]["D2_entry_count_advanced"] is False
    assert "remaining 54 q/p directions" in text
    leaf_completion = _load(D2_REACHABLE_LEAF_COMPLETION_RECEIPT)
    leaf_counts = leaf_completion["gate_counts"]
    assert leaf_counts["reachable_leaf_derivative_obligations"] == 31_680
    assert leaf_counts["registered_exact_leaf_derivative_roots"] == 31_680
    assert leaf_counts["nonzero_leaf_derivative_roots"] == 396
    assert leaf_counts["exact_zero_leaf_derivative_roots"] == 31_284
    assert leaf_counts["bounded_ordered_D2_roots_registered"] == 264
    assert leaf_counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 131
    assert leaf_completion["claim_seals"]["all_31680_reachable_leaf_derivative_roots_registered"]
    assert leaf_completion["claim_seals"]["D2_entry_count_advanced"] is False
    assert "all 31,680 A/B/C component-input roots" in text
    assert "checks all 264 bounded ordered-D2 Merkle records" in text
    pgradient = _load(D2_PGRADIENT_LEAF_RECEIPT)
    pgradient_counts = pgradient["gate_counts"]
    assert pgradient_counts["new_leaf_derivative_roots_all_candidates"] == 126_720
    assert pgradient_counts["nonzero_leaf_derivative_roots"] == 13_728
    assert pgradient_counts["exact_zero_leaf_derivative_roots"] == 112_992
    assert pgradient_counts["new_scalar_gradient_coordinate_columns"] == 4
    assert pgradient_counts["registered_coordinate_columns_after"] == 26
    assert pgradient_counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 127
    assert pgradient_counts["potential_candidate_bound_D2_records_blocked"] == 1_056
    assert pgradient["claim_seals"]["D2_entry_count_advanced"] is False
    previous = 26
    for relative_path, expected_after in D2_LEAF_SUCCESSOR_CHAIN:
        successor = _load(ROOT / relative_path)
        counts = successor["gate_counts"]
        assert counts["previous_registered_coordinate_columns"] == previous
        assert counts["registered_coordinate_columns_after"] == expected_after
        assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == (
            153 - expected_after
        )
        assert counts["registered_D2_entries_per_candidate_before"] == 5_324
        assert counts["registered_D2_entries_per_candidate_after"] == 5_324
        assert successor["claim_seals"]["D2_entry_count_advanced"] is False
        previous = expected_after
    assert previous == 145
    assert previous - len(FORMAL_LOWER_DIRECTION_ALIASES) == 143
    assert "145 formal coordinate registrations" in text
    assert "143/153 unique A/B/C leaf columns" in text
    assert "`s11[10]` and `s22[10]` are duplicate aliases" in text
    assert "all ten `q[0..9]` columns" in text
    assert "D2 remains 5,324/257,499" in text
    checkpoint, unit_000, candidate_0, candidate_1 = [_load(path) for path in Q_LEAF_RECEIPTS]
    assert checkpoint["gate_counts"]["materialized_q_tangent_scalar_values"] == 200
    assert checkpoint["gate_counts"]["planned_leaf_roots_all_candidates"] == 316_800
    assert checkpoint["gate_counts"]["unique_registered_coordinate_columns_after"] == 143
    assert unit_000["gate_counts"]["materialized_leaf_roots"] == 2_640
    assert unit_000["gate_counts"]["nonzero_leaf_derivative_roots"] == 311
    assert unit_000["gate_counts"]["exact_zero_leaf_derivative_roots"] == 2_329
    assert candidate_0["gate_counts"]["new_leaf_roots"] == 23_760
    assert candidate_0["gate_counts"]["nonzero_leaf_roots"] == 2_799
    assert candidate_0["gate_counts"]["exact_zero_leaf_roots"] == 20_961
    final_q_counts = candidate_1["gate_counts"]
    assert final_q_counts["new_leaf_roots"] == 26_400
    assert final_q_counts["nonzero_leaf_roots"] == 3_110
    assert final_q_counts["exact_zero_leaf_roots"] == 23_290
    assert final_q_counts["cumulative_leaf_roots"] == 52_800
    assert final_q_counts["cumulative_completed_units"] == 20
    assert final_q_counts["remaining_leaf_roots"] == 264_000
    assert final_q_counts["remaining_units"] == 100
    assert final_q_counts["unique_registered_coordinate_columns_after"] == 143
    assert final_q_counts["registered_D2_entries_per_candidate_after"] == 5_324
    assert "52,800 of 316,800 exact q-family leaf roots" in text
    assert "20 of 120 checkpoint units" in text
    assert "264,000 roots and 100 checkpoint units remaining" in text
    assert "no q column is yet promoted" in text


def test_system11_authorization_closure_is_current_and_target_free() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    receipt = _load(SYSTEM11_SOLAR_AUTHORIZATION_RECEIPT)
    counts = receipt["counts"]
    assert receipt["decision"] == "block"
    assert counts["bound_metadata_artifacts_audited"] == 10
    assert counts["already_authorized_observational_artifacts"] == 0
    assert counts["missing_external_opening_obligations"] == 8
    assert counts["primary_record_accesses"] == 0
    assert counts["held_out_target_accesses"] == 0
    assert counts["real_data_evaluations"] == 0
    assert (
        receipt["execution_implementation_audit"]["action_bound_real_record_likelihood_executor"]
        == "block_not_registered"
    )
    assert "audits 10 bound metadata artifacts" in text
    executor = _load(SYSTEM11_SOLAR_LIKELIHOOD_RECEIPT)
    assert executor["counts"]["missing_external_opening_obligations"] == 8
    assert executor["counts"]["primary_record_accesses"] == 0
    assert executor["counts"]["real_data_evaluations"] == 0
    assert executor["counts"]["synthetic_controls"] == 4
    assert executor["counts"]["synthetic_pass_controls"] == 2
    assert executor["counts"]["synthetic_reject_controls"] == 2
    assert executor["claims"]["observation_opened"] is False
    assert executor["claims"]["observational_result_exists"] is False
    assert "action-bound real-record likelihood executor contract" in text
    assert "remaining blocker is exactly the eight external opening obligations" in text


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
    assert "irrotational `P(X)=kappa X^2` sector" in text

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
    cylindrical_rows = _load(SYSTEM10_CYLINDRICAL_ROWS_RECEIPT)
    assert cylindrical_rows["decision"] == (
        "BOUNDED_PASS_48_EXACT_CYLINDRICAL_HAMILTONIAN_MOMENTUM_ROWS_NO_PROPAGATION_CLAIM"
    )
    assert cylindrical_rows["counts"]["candidates"] == 12
    assert cylindrical_rows["counts"]["hamiltonian_rows_closed"] == 12
    assert cylindrical_rows["counts"]["momentum_rows_closed"] == 36
    assert cylindrical_rows["counts"]["hamiltonian_momentum_rows_closed"] == 48
    assert cylindrical_rows["counts"]["specialized_physical_gravity_rows_closed"] == 96
    assert (
        cylindrical_rows["materialization"]["acceleration_and_integrability_proof"][
            "partial_0_v_nonzero_coefficients"
        ]
        == 0
    )
    assert (
        cylindrical_rows["materialization"]["acceleration_and_integrability_proof"][
            "forbidden_time_differential_atoms_after_replacement"
        ]
        == 0
    )
    assert cylindrical_rows["claims"]["sourced_constraint_propagation_closed"] is False
    r_positive = _load(SYSTEM10_R_POSITIVE_RECEIPT)
    assert r_positive["decision"] == (
        "BOUNDED_PASS_FIXED_CYLINDRICAL_R_POSITIVE_1010_JETS_AND_96_ROWS_NO_PROPAGATION_CLAIM"
    )
    assert r_positive["counts"]["formulation_jet_rational_functions"] == 1_010
    assert r_positive["counts"]["physical_gravity_rows_closed"] == 96
    assert r_positive["counts"]["exact_r1_replays"] == 1_062
    assert r_positive["claims"]["fixed_cylindrical_profile_r_positive_closed"]
    assert r_positive["claims"]["sourced_constraint_propagation_closed"] is False
    propagation = _load(SYSTEM10_PROPAGATION_ATTEMPT_RECEIPT)
    assert propagation["counts"]["fully_expanded_divQ_rows_required"] == 4
    assert propagation["counts"]["fully_expanded_divQ_rows_registered"] == 0
    assert propagation["counts"]["sourced_constraint_propagation_proofs"] == 0
    divq = _load(SYSTEM10_DIVQ_RECEIPT)
    assert divq["counts"]["divq_rows_required"] == 4
    assert divq["counts"]["divq_rows_registered"] == 4
    assert divq["counts"]["total_nonzero_operator_terms"] == 191
    assert divq["claims"]["four_divq_rows_closed"]
    assert divq["claims"]["constraint_propagation_closed"] is False
    kinematic = _load(SYSTEM10_KINEMATIC_RHS_RECEIPT)
    assert kinematic["counts"]["common_kinematic_rows_registered"] == 68
    assert kinematic["counts"]["full_85_state_rhs_candidates_closed"] == 0
    matter_rhs = _load(SYSTEM10_MATTER_RHS_RECEIPT)
    assert matter_rhs["counts"]["matter_dynamic_rows_registered"] == 2
    assert matter_rhs["counts"]["total_rhs_rows_closed_per_candidate"] == 70
    maxwell_rhs = _load(SYSTEM10_MAXWELL_RHS_RECEIPT)
    assert maxwell_rhs["counts"]["maxwell_dynamic_rows_registered"] == 4
    assert maxwell_rhs["counts"]["total_rhs_rows_closed_per_candidate"] == 74
    assert maxwell_rhs["counts"]["candidate_dynamic_rows_remaining"] == 132
    gravity_scalar = _load(SYSTEM10_GRAVITY_SCALAR_RHS_RECEIPT)
    assert gravity_scalar["counts"]["predecessor_rhs_rows_per_candidate"] == 74
    assert gravity_scalar["counts"]["total_rhs_rows_closed_per_candidate"] == 74
    assert gravity_scalar["counts"]["dynamic_rows_remaining_per_candidate"] == 11
    assert gravity_scalar["counts"]["candidate_dynamic_rows_remaining"] == 132
    assert gravity_scalar["claims"]["full_85_state_rhs_closed"] is False
    aw = _load(SYSTEM10_GRAVITY_SCALAR_AW_RECEIPT)
    assert aw["counts"]["rhs_rows_closed_per_candidate"] == 74
    assert aw["counts"]["semantic_A_entries_manifested"] == 121
    assert aw["counts"]["semantic_W_entries_manifested"] == 11
    assert aw["counts"]["coordinate_arithmetic_A_entries"] == 0
    assert aw["counts"]["coordinate_arithmetic_W_entries"] == 0
    assert aw["counts"]["candidate_dynamic_rows_remaining"] == 132
    assert aw["claims"]["coordinate_arithmetic_A_W_materialized"] is False
    aw_rows = [_load(path) for path in SYSTEM10_GRAVITY_SCALAR_AW_ROWS]
    assert [row["row"] for row in aw_rows] == list(range(11))
    assert sum(len(row["A_entries"]) for row in aw_rows) == 121
    assert sum(1 for row in aw_rows if row["W_entry"]) == 11
    assert all(row["certificates"]["affine_residual"] == "0" for row in aw_rows)
    assert all(row["claims"]["solved_acceleration_row"] is False for row in aw_rows)
    invertibility = _load(SYSTEM10_GRAVITY_SCALAR_AW_INVERTIBILITY_RECEIPT)
    assert invertibility["decision"] == "BLOCK_GLOBAL_ACCELERATION_SOLVE_EXACT_SINGULAR_A_WITNESS"
    assert invertibility["exact_singular_witness"]["rank"] == 10
    assert invertibility["exact_singular_witness"]["domain_certificate"] == "r=1>0"
    assert (
        invertibility["conclusion"]["global_invertibility_over_fixed_r_positive_state_domain"]
        is False
    )
    tube = _load(SYSTEM10_GRAVITY_SCALAR_AW_TUBE_RECEIPT)
    assert tube["decision"] == (
        "BOUNDED_PASS_REPRESENTATIVE_A_W_SOLVE_ON_PREREGISTERED_NONSINGULAR_TUBE"
    )
    assert tube["preregistered_tube"]["real_v_10_interval"] == ["-1/2", "1/2"]
    assert tube["invertibility_certificate"]["exact_absolute_lower_bound"] == (
        "3486784401/268435456"
    )
    assert tube["claims"]["representative_tube_all_11_accelerations_solved"] is True
    assert tube["claims"]["representative_tube_all_11_residuals_replayed"] is True
    assert tube["claims"]["other_candidates_solved"] is False
    all_aw = _load(SYSTEM10_TWELVE_CANDIDATE_AW_RECEIPT)
    assert all_aw["decision"] == "BOUNDED_PASS_ALL_TWELVE_A_W_PACKETS_AND_COMMON_LOCAL_TUBE"
    assert all_aw["counts"] == {
        "A_entries": 1452,
        "W_entries": 132,
        "candidate_packets": 12,
        "rows": 132,
        "tube_admitted_candidates": 12,
    }
    assert all_aw["claims"]["all_twelve_candidate_A_W_packets_materialized"] is True
    assert all_aw["claims"]["common_local_tube_admitted"] is True
    assert all_aw["claims"]["global_candidate_domains_invertible"] is False
    assert all_aw["claims"]["full_rhs"] is False
    assert len([_load(path) for path in SYSTEM10_TWELVE_CANDIDATE_AW_PACKETS]) == 12
    all_tube = _load(SYSTEM10_TWELVE_CANDIDATE_AW_TUBE_RECEIPT)
    assert all_tube["decision"] == (
        "BOUNDED_PASS_ALL_TWELVE_COMMON_TUBE_ACCELERATION_SOLVES_AND_RESIDUALS"
    )
    assert all_tube["counts"] == {
        "accelerations_solved": 132,
        "candidate_blocks": 0,
        "candidate_passes": 12,
        "candidate_solution_packets": 12,
        "zero_residuals_replayed": 132,
    }
    assert all_tube["common_tube"]["r"] == "1"
    assert all_tube["common_tube"]["real_v_10_interval"] == ["-1/4", "1/4"]
    assert all_tube["claims"]["all_twelve_common_tube_accelerations_solved"] is True
    assert all_tube["claims"]["all_twelve_common_tube_residuals_replayed"] is True
    assert all_tube["claims"]["all_twelve_global_domains_solved"] is False
    assert all_tube["claims"]["full_rhs"] is False
    assert len([_load(path) for path in SYSTEM10_TWELVE_CANDIDATE_AW_TUBE_SOLUTIONS]) == 12
    full_rhs = _load(SYSTEM10_COMMON_TUBE_FULL_RHS_RECEIPT)
    assert full_rhs["decision"] == (
        "BOUNDED_PASS_12_CANDIDATES_EXACT_85_OF_85_LINKED_RHS_ON_COMMON_TUBE"
    )
    assert full_rhs["counts"]["candidate_passes"] == 12
    assert full_rhs["counts"]["total_rhs_rows_per_candidate"] == 85
    assert full_rhs["counts"]["total_rhs_row_instances"] == 1020
    assert full_rhs["counts"]["equation_origin_seals"] == 1020
    assert full_rhs["counts"]["new_dynamic_row_instances"] == 132
    assert full_rhs["counts"]["new_exact_zero_residual_replays"] == 132
    assert full_rhs["common_tube"]["r"] == "1"
    assert full_rhs["common_tube"]["real_v_10_interval"] == ["-1/4", "1/4"]
    assert full_rhs["claims"]["all_twelve_common_tube_exact_85_state_rhs_closed"] is True
    assert full_rhs["claims"]["fixed_r_positive_domain_full_rhs_closed"] is False
    assert full_rhs["claims"]["global_domain_full_rhs_closed"] is False
    assert full_rhs["claims"]["constraint_propagation_closed"] is False
    assert len([_load(path) for path in SYSTEM10_COMMON_TUBE_FULL_RHS_PACKETS]) == 12
    propagation = _load(SYSTEM10_COMMON_TUBE_PROPAGATION_RECEIPT)
    assert propagation["decision"] == (
        "BLOCK_COMMON_TUBE_RHS_HAS_NO_RADIAL_JET_FOR_CONSTRAINT_PROPAGATION"
    )
    assert propagation["counts"]["full_rhs_candidate_packets_bound"] == 12
    assert propagation["counts"]["full_rhs_rows_bound_per_candidate"] == 85
    assert propagation["counts"]["constraint_propagation_proofs"] == 0
    assert propagation["counts"]["candidate_subsidiary_systems_closed"] == 0
    assert propagation["counts"]["subsidiary_energy_estimates"] == 0
    missing = propagation["materialization"]["first_missing_primitive"]
    assert missing["primitive"] == (
        "candidate_bound_radial_first_jet_of_all_11_solved_dynamic_rhs_rows"
    )
    assert missing["required_domain"] == (
        "an open radial neighborhood of r=1 with the same state tube"
    )
    witness = propagation["materialization"]["radial_jet_nonidentifiability_witness"]
    assert witness["same_registered_tube_value"] is True
    assert witness["radial_derivative_delta_at_r_1"] == "1"
    assert witness["exact_constraint_time_derivative_delta"] == "1/128"
    assert propagation["claims"]["full_85_state_rhs_closed_on_common_tube"] is True
    assert propagation["claims"]["constraint_propagation_closed_on_common_tube"] is False
    assert propagation["claims"]["global_theorem_established"] is False
    radial = _load(SYSTEM10_OPEN_R_RECEIPTS["radial"])
    assert radial["counts"]["candidate_passes"] == 12
    assert radial["counts"]["open_r_rhs_rows"] == 132
    assert radial["counts"]["open_r_radial_rhs_jets"] == 132
    assert radial["counts"]["open_r_zero_residuals"] == 132
    assert radial["counts"]["radially_differentiated_zero_residuals"] == 132
    assert radial["counts"]["r1_rhs_replays"] == 132
    assert radial["claims"]["constraint_propagation_closed"] is False
    tangential = _load(SYSTEM10_OPEN_R_RECEIPTS["tangential"])
    assert tangential["counts"]["tangential_rhs_jets"] == 264
    assert tangential["counts"]["direction_2_rhs_jets"] == 132
    assert tangential["counts"]["direction_3_rhs_jets"] == 132
    assert tangential["counts"]["differentiated_zero_residuals"] == 264
    assert tangential["counts"]["base_rhs_row_replays"] == 132
    assert tangential["counts"]["unclassified_W_atoms"] == 0
    decomposition = _load(SYSTEM10_OPEN_R_RECEIPTS["decomposition"])
    assert decomposition["counts"]["coordinate_decomposition_rows_closed"] == 48
    assert decomposition["counts"]["symbolic_coordinate_zero_residuals"] == 48
    assert decomposition["counts"]["physical_constraint_rows_bound"] == 96
    assert decomposition["counts"]["all_first_spatial_rhs_jets_bound"] == 396
    assert decomposition["counts"]["full_rhs_rows_bound"] == 1020
    normalization = _load(SYSTEM10_OPEN_R_RECEIPTS["normalization"])
    assert normalization["counts"]["normalization_bridges_closed"] == 12
    assert normalization["counts"]["spatial_metric_euler_normalizations"] == 72
    assert normalization["counts"]["gravity_scalar_euler_normalizations"] == 12
    assert normalization["counts"]["matter_force_sector_normalizations"] == 36
    assert normalization["counts"]["divQ_normalizations"] == 48
    factorization = _load(SYSTEM10_OPEN_R_RECEIPTS["factorization"])
    assert factorization["counts"]["divQ_to_C_factorization_rows_closed"] == 4
    assert factorization["counts"]["exact_termwise_replays"] == 4
    assert factorization["counts"]["expanded_operator_terms"] == 191
    assert factorization["claims"]["constraint_propagation_closed"] is False
    initial_data = _load(SYSTEM10_OPEN_R_RECEIPTS["initial_data"])
    assert initial_data["counts"]["candidate_subsidiary_initial_data_maps_closed"] == 12
    assert initial_data["counts"]["gravity_normal_derivative_maps_closed"] == 48
    assert initial_data["counts"]["maxwell_normal_derivative_maps_closed"] == 12
    assert initial_data["counts"]["homogeneous_gravity_subsidiary_equations_closed"] == 4
    assert initial_data["counts"]["homogeneous_maxwell_subsidiary_equations_closed"] == 1
    gravity_map = initial_data["materialization"]["gravity_normal_derivative_map"]
    assert gravity_map["det_A"] == "-6561/(256*r**2)"
    assert gravity_map["det_A_nonzero_on_r_positive"] is True
    current_missing = initial_data["materialization"]["propagation_audit"][
        "first_missing_primitive"
    ]
    assert current_missing["primitive"] == (
        "common_tube_homogeneous_subsidiary_cauchy_uniqueness_certificate"
    )
    assert current_missing["registered_energy_estimates"] == 0
    assert initial_data["claims"]["all_twelve_subsidiary_initial_data_maps_closed"] is True
    assert initial_data["claims"]["subsidiary_energy_or_uniqueness_closed"] is False
    assert initial_data["claims"]["constraint_propagation_closed"] is False
    assert "max|B_mu| <= 8/38505" in text
    assert "all 96 physical gravity rows" in text
    assert "all 96 physical gravity rows over `r>0`" in text
    assert "30,884 exact sparse-polynomial terms" in text
    assert "Arbitrary formulation functions" in text
    assert "all 780 readiness slots in 48 chained packets" in text
    assert "17 exact indexed tensor templates" in text
    assert "four gravity rows into 112 exact coefficients" in text
    assert "complete 1,010-value packet" in text
    assert "74/85 RHS rows per candidate" in text
    assert "11 gravity/scalar rows remain" in text
    assert "closes 4/4 rows with 191 nonzero operator terms" in text
    assert "132 semantic A/W roots (121 A and 11 W)" in text
    assert "all 121 A and 11 W entries across 11 exact rows" in text
    assert "rank-10 singular witness" in text
    assert "exact absolute lower bound `3486784401/268435456`" in text
    assert "all 11 accelerations solve with 11/11 zero residuals" in text
    assert "12 candidate A/W packets with 132 rows" in text
    assert "132 exact acceleration formulas" in text
    assert "132/132 zero residuals" in text
    assert "exact 85/85 rows for all 12 candidates on that same narrow tube" in text
    assert "1,020 row instances and equation-origin seals" in text
    assert "132 new dynamic row instances with 132 exact zero-residual replays" in text
    assert "Fixed-`r` positive and global-domain full RHS remain false" in text
    assert "396/396 first spatial jets total" in text
    assert "48/48 candidate-by-lower-`nu` coordinate decompositions" in text
    assert "12/12 Euler-normalization bridges" in text
    assert "43, 55, 50, and 43 terms (191 total)" in text
    assert "subsidiary initial-data successor closes 12/12 candidate maps" in text
    assert "determinant `-6561/(256r^2)`" in text
    assert "checked coercive energy/Cauchy-uniqueness estimate" in text
    assert "No propagation, hyperbolicity, or global theorem" in text
    assert "physical Jordan no-go" in text


def test_system12_real_duration_receipt_preserves_its_honest_scope() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    config = _load(DURABLE_TWO_HOST_CONFIG)
    receipt = _load(DURABLE_TWO_HOST_RECEIPT)
    assert config["duration"]["required_credited_seconds"] == 21_600
    assert config["logical_hosts"] == ["host-a", "host-b"]
    assert config["storage"]["maximum_sqlite_family_bytes"] == 536_870_912
    assert receipt["decision"] == "PASS"
    assert receipt["duration"]["credited_wall_seconds"] == 21_600.000388
    assert receipt["duration"]["credited_seconds_by_host"] == {
        "host-a": 17_227.661285,
        "host-b": 21_600.000388,
    }
    assert len(receipt["duration"]["cleanly_closed_sessions"]) == 2
    assert receipt["event_count"] == 16
    assert receipt["event_chain_root_sha256"] == (
        "e082114afebb223e154633416691c7c896e900052869c50d2789c47eace4eed6"
    )
    assert receipt["storage"]["total_sqlite_family_bytes"] == 53_248
    assert receipt["storage"]["maximum_sqlite_family_bytes"] == 536_870_912
    assert receipt["storage"]["within_ceiling"] is True
    assert receipt["work_counts"] == {"succeeded": 2}
    assert receipt["claims"]["real_credited_wall_seconds_at_least_six_hours"] is True
    assert receipt["claims"]["dead_session_runtime_credited"] is False
    assert receipt["claims"]["overlapping_host_runtime_double_counted"] is False
    assert receipt["claims"]["two_logical_hosts_observed"] is True
    assert receipt["claims"]["two_physical_machines_established"] is False
    assert receipt["claims"]["scientific_result_inferred"] is False
    assert "21,600.000388 credited union seconds" in text
    assert "two cleanly stopped logical-host sessions" in text
    assert "no physical two-machine or scientific-validity claim" in text


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
    assert "legacy production freshness remain open" in text
