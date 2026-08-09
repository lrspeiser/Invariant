from sigma_theory_compiler.horndeski import (
    quartic_horndeski_arbitrary_curvature_scalar_principal_control,
    quartic_horndeski_boundary_and_flrw_noether_control,
    quartic_horndeski_coupled_formulation_hyperbolicity_control,
    quartic_horndeski_scalar_euler_reduction_control,
)
from sigma_theory_compiler.horndeski_principal import (
    quartic_horndeski_baseline_riesz_symmetrizer_control,
    quartic_horndeski_full_local_principal_control,
    quartic_horndeski_x2_kessence_extension_control,
)


def test_quartic_horndeski_scalar_euler_equation_is_second_order() -> None:
    passed, result = quartic_horndeski_scalar_euler_reduction_control()
    assert passed, result
    assert result["reduction_residual"] == "0"
    assert result["fourth_derivative_coefficient"] == "0"
    assert result["curvature_gradient_coefficient"] == "0"
    assert result["expected_equation"] == "G^(mu nu) nabla_mu nabla_nu(phi)=0"
    assert result["wrong_completion_negative_control"][
        "higher_metric_derivative_restored"
    ]


def test_quartic_horndeski_boundary_equivalence_and_flrw_noether_identity() -> None:
    passed, result = quartic_horndeski_boundary_and_flrw_noether_control()
    assert passed, result
    assert result["boundary_equivalence"]["boundary_residual"] == "0"
    assert result["lapse_flrw"]["noether_residual"] == "0"
    assert result["lapse_flrw"]["negative_control_rejected"]


def test_quartic_horndeski_arbitrary_curvature_scalar_effective_cone() -> None:
    passed, result = quartic_horndeski_arbitrary_curvature_scalar_principal_control()
    assert passed, result
    assert result["euler_reduction_residual"] == "0"
    assert result["healthy_diagonal_witness"]["speed_squared"] == [
        "3/4",
        "7/8",
        "5/6",
    ]
    assert result["healthy_oblique_witness"]["x_characteristic_speeds"] == [
        "1",
        "-2/3",
    ]
    assert all(
        item["rejected"] for item in result["negative_controls"].values()
    )
    assert result["metric_cone_comparison"]["exceeds_metric_light_cone"]
    assert not result["metric_cone_comparison"][
        "health_rejection_without_declared_cone_policy"
    ]
    assert result["full_coupled_metric_scalar_principal_status"] == "unresolved"


def test_quartic_horndeski_coupled_formulation_is_fail_closed() -> None:
    passed, result = quartic_horndeski_coupled_formulation_hyperbolicity_control()
    assert passed, result
    assert result["generalized_harmonic"]["status"] == "reject"
    assert not result["generalized_harmonic"][
        "strongly_hyperbolic_for_this_action_class"
    ]
    modified = result["modified_harmonic"]
    assert modified["theorem_status"] == "conditional_pass"
    assert modified["exact_nonempty_auxiliary_cone_witness"]["all_lorentzian"]
    assert modified["flat_action_witness"]["auxiliary_cones_disjoint"]
    assert modified["flat_action_witness"]["minimum_squared_speed_gap"] == (
        "19/36"
    )
    robustness = modified["exact_cone_robustness_budget"]
    assert robustness["status"] == "conditional_pass"
    assert robustness["spectral_perturbation_budget"] == "19/72"
    assert robustness["exact_safe_witness"] == {
        "perturbation_norm": "1/4",
        "remaining_minimum_gap": "5/18",
        "inside_budget": True,
    }
    assert robustness["exact_collision_witness"][
        "remaining_minimum_gap"
    ] == "0"
    assert robustness["full_correction_norm_status"] == "unresolved"
    assert result["action_specific_application"]["status"] == "unresolved"
    assert len(result["action_specific_application"]["missing"]) == 4


def test_quartic_horndeski_full_local_principal_matrix_is_extracted() -> None:
    passed, result = quartic_horndeski_full_local_principal_control()
    assert passed, result
    assert result["matrix_shape"] == [11, 11]
    assert result["extraction_status"] == "pass"
    assert all(result["block_certificates"].values())
    flat = result["flat_constant_timelike_gradient_reduction"]
    assert flat["tensor_polarizations_match"]
    assert flat["adm_tensor_residual"] == "0"
    assert flat["scalar_polynomial"] == "-(k - omega)*(k + omega)"
    assert flat["metric_scalar_mixing_zero"]
    first_order = result["first_order_generalized_pencil"]
    assert first_order["status"] == "pass"
    assert first_order["mass_matrix_shape"] == [22, 22]
    assert first_order["evolution_matrix_shape"] == [22, 22]
    assert first_order["einstein_scalar_flat_unit_direction"]["residual"] == "0"
    assert first_order["einstein_scalar_flat_unit_direction"][
        "mode_multiplicities_per_sign"
    ] == {
        "physical_speed_1": 3,
        "pure_gauge_speed_1/2": 4,
        "gauge_violating_speed_1/3": 4,
    }
    time_block = first_order["time_block_invertibility"]
    assert time_block["status"] == "conditional_pass"
    assert time_block["baseline_general_determinant"] == "6561*M2**10/4096"
    assert time_block["baseline_general_determinant"] == time_block[
        "expected_baseline_general_determinant"
    ]
    assert time_block["baseline_minimum_singular_value"] == "Min(1, M2/4)"
    assert time_block["sum_of_squares_residual"] == "0"
    assert "Min(1, M2/4)**2" in time_block["sufficient_condition"]
    assert time_block["declared_gradient_only_domain_status"] == "unresolved"
    assert len(time_block["missing_uniform_background_bounds"]) == 3
    collapse = time_block["curvature_collapse_negative_control"]
    assert collapse["A_determinant"] == "0"
    assert collapse["A_rank"] == 10
    assert "A_star_squared=0" in collapse["gradient_domain_inequality"]
    assert first_order["generic_A_invertibility_status"] == (
        "conditional_pass_with_missing_domain_bounds"
    )
    assert result["uniform_symmetrizer_and_norm_status"] == "unresolved"


def test_quartic_horndeski_x2_kessence_extension_changes_only_scalar_block() -> None:
    passed, result = quartic_horndeski_x2_kessence_extension_control()
    assert passed, result
    assert result["matrix_shape"] == [11, 11]
    assert result["extension_location"] == "scalar-scalar entry only"
    assert result["canonical_c20_zero_matrix_residual_zero"]
    assert result["arbitrary_covector_effective_metric_residual"] == "0"
    assert result["flat_constant_timelike_gradient"]["residual"] == "0"
    assert result["negative_control"]["rejected"]


def test_quartic_baseline_has_exact_six_group_positive_symmetrizer() -> None:
    passed, result = quartic_horndeski_baseline_riesz_symmetrizer_control()
    assert passed, result
    assert result["status"] == (
        "pass_exact_baseline_and_quantitative_physical_group_contract"
    )
    assert result["projector_sum_residual_zero"]
    assert result["pairwise_projector_products_zero"]
    assert result["symmetrizer"]["K_M_minus_M_T_K_zero"]
    assert result["symmetrizer"]["all_LDL_pivots_positive"]
    assert result["symmetrizer"]["decomposition_norm_lower_bound"] == "1/24"
    contract = result["quantitative_physical_group_perturbation_contract"]
    assert contract["required_companion_2_norm_perturbation_upper_numeric"] > 0
    assert contract["implied_Riesz_projector_drift_upper"] == "1/32"
    assert contract["certified_physical_H_star_margin"] == "181/4096"
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert result["full_quartic_candidate_status"] == (
        "unresolved_uniform_matrix_and_hat_block_bounds"
    )
