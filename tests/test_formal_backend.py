from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.formal_backend import (
    load_field_contract,
    run_formal_control_suite,
    validate_covariant_action_spec,
    validate_field_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "covariant_field_contract.json"


def test_covariant_field_contract_is_complete_and_fail_closed_for_z_b() -> None:
    contract = load_field_contract(CONTRACT_PATH)
    result = validate_field_contract(contract)
    assert result["valid"]
    assert result["z_b_definition"] == "-g_mu_nu J_b^mu J_b^nu/n_0^2 = n_b^2/n_0^2"
    assert result["z_b_generator_status"] == "forbidden"


def test_baryonic_action_atom_is_rejected_despite_one_metric() -> None:
    contract = load_field_contract(CONTRACT_PATH)
    result = validate_covariant_action_spec(
        {
            "matter_metric": "g_mu_nu",
            "invariants": ["z_b"],
            "action": "sqrt(-g) F(z_b)",
            "static_dictionary_status": "derived",
        },
        contract,
    )
    assert not result["valid"]
    assert "z_b" in result["prohibited_diagnostic_invariants"]
    assert any("baryon-specific" in error for error in result["errors"])


def test_declared_gravitational_invariants_require_derived_static_dictionary() -> None:
    contract = load_field_contract(CONTRACT_PATH)
    base = {
        "matter_metric": "g_mu_nu",
        "invariants": ["K1_u", "K2_u"],
        "action": "sqrt(-g) F(K1_u,K2_u)",
    }
    unresolved = validate_covariant_action_spec(
        {**base, "static_dictionary_status": "claimed"}, contract
    )
    derived = validate_covariant_action_spec(
        {**base, "static_dictionary_status": "derived"}, contract
    )
    assert not unresolved["valid"]
    assert derived["valid"]


def test_all_formal_known_answer_controls_pass(tmp_path) -> None:
    report = run_formal_control_suite(CONTRACT_PATH, PROJECT_ROOT)
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if report["backends"]["cadabra2"]["available"]:
        assert report["counts"] == {"total": 118, "passed": 118, "failed": 0}
    else:
        assert report["counts"] == {"total": 106, "passed": 106, "failed": 0}
    checks = {item["name"]: item for item in report["checks"]}
    geometric_jet = checks[
        "quartic_linear_x_nonlinear_geometric_state_to_jet_map"
    ]
    assert geometric_jet["status"] == "pass"
    assert geometric_jet["evidence"]["artifact_hash_matches_reexecution"]
    assert geometric_jet["evidence"]["curvilinear_flat_control"] == {
        "metric": "diag(-1,1,r^2,1)",
        "scalar": "r",
        "connection_nonzero": True,
        "riemann_zero": True,
        "hessian_theta_theta_residual": "0",
    }
    assert all(
        item["rejected"]
        for item in geometric_jet["evidence"]["negative_controls"].values()
    )
    nonlinear_source = checks[
        "quartic_linear_x_gauge_fixed_nonlinear_evolution_source"
    ]
    assert nonlinear_source["status"] == "pass"
    assert nonlinear_source["evidence"]["artifact_hash_matches_reexecution"]
    assert nonlinear_source["evidence"]["time_acceleration_affine_residual_zero"]
    assert nonlinear_source["evidence"][
        "independent_principal_time_block_residual_zero"
    ]
    assert nonlinear_source["evidence"]["sample_solution"]["solution_residual_zero"]
    assert all(
        item["rejected"]
        for item in nonlinear_source["evidence"]["negative_controls"].values()
    )
    nonquasilinear_pde = checks[
        "quartic_linear_x_full_nonquasilinear_pde_symmetrizer"
    ]
    assert nonquasilinear_pde["status"] == "pass"
    assert nonquasilinear_pde["evidence"]["artifact_hash_matches_reexecution"]
    assert nonquasilinear_pde["evidence"]["generic_nonquasilinear_control"][
        "passed"
    ]
    assert nonquasilinear_pde["evidence"]["generic_full_symmetrizer_lift_control"][
        "K55_M55_minus_M55_dagger_K55_zero"
    ]
    assert (
        nonquasilinear_pde["evidence"]["representative_uniform_bounds"][
            "K55_2_lower_numeric"
        ]
        > 0
    )
    assert nonquasilinear_pde["evidence"]["characteristic_gap_negative"][
        "status"
    ] == "reject"
    coordinate_tube = checks[
        "quartic_linear_x_coordinate_jet_hyperbolicity_tube"
    ]
    assert coordinate_tube["status"] == "pass"
    assert coordinate_tube["evidence"]["artifact_hash_matches_reexecution"]
    assert coordinate_tube["evidence"]["bounded_coordinate_atoms"]["total"] == 153
    assert all(
        item["strict_margin_numeric"] > 0
        for item in coordinate_tube["evidence"][
            "covariant_hyperbolicity_components"
        ].values()
    )
    assert coordinate_tube["evidence"]["negative_control"]["rejected"]
    assert coordinate_tube["evidence"]["configuration_radius_negative"][
        "status"
    ] == "reject"
    euler_remainder = checks["quartic_linear_x_euler_remainder_majorants"]
    assert euler_remainder["status"] == "pass"
    assert euler_remainder["evidence"]["artifact_hash_matches_reexecution"]
    assert euler_remainder["evidence"]["term_counts"] == {
        "quartic_metric_lower": 8,
        "G2_metric_lower": 2,
        "scalar_euler": 3,
        "modified_harmonic_gauge": 4,
    }
    assert euler_remainder["evidence"]["representative_acceleration_upper"] > 0
    assert euler_remainder["evidence"]["term_inventory_negative"]["status"] == "reject"
    solved_source = checks["quartic_linear_x_solved_source_moser_envelopes"]
    assert solved_source["status"] == "pass"
    assert solved_source["evidence"]["artifact_hash_matches_reexecution"]
    assert solved_source["evidence"]["generic_composition_control"]["passed"]
    assert (
        solved_source["evidence"]["solved_source_fourth_order_range"]["minimum"]
        > 1e19
    )
    assert (
        solved_source["evidence"]["solved_source_fourth_order_range"]["maximum"]
        < 2e20
    )
    assert solved_source["evidence"]["insufficient_order_negative"]["status"] == "reject"
    full_symmetrizer = checks["quartic_linear_x_full_symmetrizer_moser_envelopes"]
    assert full_symmetrizer["status"] == "pass"
    assert full_symmetrizer["evidence"]["artifact_hash_matches_reexecution"]
    assert full_symmetrizer["evidence"]["generic_symmetrizer_derivative_control"][
        "passed"
    ]
    assert full_symmetrizer["evidence"]["K55_coordinate_fourth_order_range"][
        "minimum"
    ] > 1e54
    assert full_symmetrizer["evidence"]["K55_coordinate_fourth_order_range"][
        "maximum"
    ] < 5e55
    assert full_symmetrizer["evidence"]["insufficient_order_negative"]["status"] == "reject"
    symbol_moser = checks["quartic_linear_x_symmetrizer_symbol_moser_envelopes"]
    assert symbol_moser["status"] == "pass"
    assert symbol_moser["evidence"]["artifact_hash_matches_reexecution"]
    assert symbol_moser["evidence"]["mixed_multiindex_count"] == 15
    assert symbol_moser["evidence"]["generic_bivariate_symbol_derivative_control"][
        "negative_control"
    ]["rejected"]
    assert symbol_moser["evidence"]["K55_total_order_four_range"][
        "minimum"
    ] > 8e44
    assert symbol_moser["evidence"]["K55_total_order_four_range"][
        "maximum"
    ] < 9e44
    assert symbol_moser["evidence"]["raw_direction_degree_witnesses"]["B_0_2"][
        "exact"
    ] == "0"
    assert symbol_moser["evidence"]["raw_direction_degree_witnesses"][
        "H_star_0_2"
    ]["exact"] == "0"
    assert symbol_moser["evidence"]["raw_direction_degree_witnesses"]["C_0_2"][
        "numeric"
    ] > 0
    assert symbol_moser["evidence"]["insufficient_order_negative"]["status"] == "reject"
    frequency_symbol = checks[
        "quartic_linear_x_homogeneous_frequency_symbol_envelopes"
    ]
    assert frequency_symbol["status"] == "pass"
    assert frequency_symbol["evidence"]["artifact_hash_matches_reexecution"]
    frequency_control = frequency_symbol["evidence"][
        "generic_homogeneous_frequency_chain_rule_control"
    ]
    assert frequency_control["normalization_map_Frechet_majorants"] == {
        "0": 1,
        "1": 2,
        "2": 6,
        "3": 36,
        "4": 300,
    }
    assert frequency_control["negative_control"]["rejected"]
    assert frequency_symbol["evidence"][
        "K55_homogeneous_frequency_total_order_four_range"
    ]["minimum"] > 8e44
    assert frequency_symbol["evidence"][
        "K55_homogeneous_frequency_total_order_four_range"
    ]["maximum"] < 9e44
    assert frequency_symbol["evidence"]["representative_frequency_bounds"]["0,4"][
        "coordinate_multiindices_covered"
    ] == 15
    assert frequency_symbol["evidence"]["insufficient_order_negative"]["status"] == (
        "reject"
    )
    low_frequency = checks["quartic_linear_x_low_frequency_symbol_extension"]
    assert low_frequency["status"] == "pass"
    assert low_frequency["evidence"]["artifact_hash_matches_reexecution"]
    low_frequency_control = low_frequency["evidence"][
        "generic_low_frequency_extension_control"
    ]
    assert set(low_frequency_control["endpoint_C4_residuals"].values()) == {"0"}
    assert low_frequency_control["radial_cutoff_Frechet_majorants"] == {
        "0": 1,
        "1": 10080,
        "2": 80640,
        "3": 735840,
        "4": 7650720,
    }
    assert low_frequency_control["negative_control"]["rejected"]
    assert low_frequency["evidence"]["K55_global_lower_range"]["minimum"] > 4e-26
    assert low_frequency["evidence"]["K55_global_total_order_four_range"][
        "maximum"
    ] < 9e44
    assert low_frequency["evidence"]["representative_global_bounds"]["0,4"][
        "coordinate_multiindices_covered"
    ] == 15
    assert low_frequency["evidence"]["insufficient_order_negative"]["status"] == (
        "reject"
    )
    positive_quantization = checks[
        "quartic_linear_x_positive_symmetrizer_quantization"
    ]
    assert positive_quantization["status"] == "pass"
    assert positive_quantization["evidence"]["artifact_hash_matches_reexecution"]
    anti_wick = positive_quantization["evidence"][
        "generic_gaussian_anti_wick_control"
    ]
    assert anti_wick["window_norm_squared"] == "1"
    assert anti_wick["resolution_of_identity_coefficient"] == "1"
    assert anti_wick["negative_control"]["rejected"]
    assert positive_quantization["evidence"]["operator_energy_lower_range"][
        "minimum"
    ] > 4e-26
    assert positive_quantization["evidence"]["operator_energy_upper_range"][
        "minimum"
    ] > 1e21
    assert positive_quantization["evidence"]["wrong_state_dimension_negative"][
        "status"
    ] == "reject"
    evolution_symbol = checks[
        "quartic_linear_x_full_evolution_symbol_envelopes"
    ]
    assert evolution_symbol["status"] == "pass"
    assert evolution_symbol["evidence"]["artifact_hash_matches_reexecution"]
    evolution_control = evolution_symbol["evidence"][
        "generic_degree_one_evolution_symbol_control"
    ]
    assert evolution_control["block_scalar_residual"] == "0"
    assert evolution_control["radius_map_Frechet_majorants"] == {
        "0": 1,
        "1": 1,
        "2": 2,
        "3": 6,
        "4": 36,
    }
    assert evolution_control["negative_control"]["rejected"]
    assert evolution_symbol["evidence"]["P55_scaled_zeroth_order_range"][
        "minimum"
    ] > 0
    assert evolution_symbol["evidence"]["P55_scaled_total_order_four_range"][
        "minimum"
    ] > 0
    assert evolution_symbol["evidence"]["representative_principal_symbol_bounds"][
        "0,4"
    ]["coordinate_multiindices_covered"] == 15
    assert evolution_symbol["evidence"]["insufficient_order_negative"][
        "status"
    ] == "reject"
    r3_sobolev = checks["quartic_linear_x_R3_H6_symbol_spatialization"]
    assert r3_sobolev["status"] == "pass"
    assert r3_sobolev["evidence"]["artifact_hash_matches_reexecution"]
    r3_control = r3_sobolev["evidence"]["generic_R3_sobolev_chain_control"]
    assert r3_control["H6_embedding_constant_squares"] == {
        "0": "7/(1024*pi)",
        "1": "3/(1024*pi)",
        "2": "3/(1024*pi)",
        "3": "7/(1024*pi)",
        "4": "63/(1024*pi)",
    }
    assert set(r3_control["spatial_chain_residuals"].values()) == {"0"}
    assert set(r3_control["time_chain_residuals"].values()) == {"0"}
    assert r3_control["negative_control"]["rejected"]
    assert r3_sobolev["evidence"]["sufficient_H6_tube_radius_range"][
        "minimum"
    ] > 2e-12
    assert r3_sobolev["evidence"]["representative_coordinate_map_ceilings"] == {
        "1": "481",
        "2": "26860",
        "3": "991862",
        "4": "34142034",
    }
    assert len(r3_sobolev["evidence"]["representative_spatialized_K55_bounds"]) == 15
    assert len(r3_sobolev["evidence"]["representative_spatialized_P55_bounds"]) == 15
    assert len(r3_sobolev["evidence"]["representative_time_K55_bounds"]) == 10
    assert r3_sobolev["evidence"]["insufficient_order_negative"]["status"] == "reject"
    time_atoms = checks["quartic_linear_x_H7_coordinate_time_atom_budget"]
    assert time_atoms["status"] == "pass"
    assert time_atoms["evidence"]["artifact_hash_matches_reexecution"]
    atom_control = time_atoms["evidence"][
        "generic_coordinate_atom_time_evolution_control"
    ]
    assert atom_control["coordinate_atom_counts"]["total"] == 153
    assert atom_control["minimal_integer_state_sobolev_order"] == 7
    assert set(atom_control["commuting_partial_residuals"].values()) == {"0"}
    assert atom_control["insufficient_H6_negative"]["rejected"]
    time_chain = time_atoms["evidence"]["generic_marked_time_chain_control"]
    assert set(time_chain["source_spatial_residuals"].values()) == {"0"}
    assert set(time_chain["marked_time_residuals"].values()) == {"0"}
    assert time_atoms["evidence"]["sufficient_H7_state_radius_range"][
        "minimum"
    ] > 2e-12
    assert len(
        time_atoms["evidence"]["representative_source_spatial_chain_bounds"]
    ) == 5
    assert len(
        time_atoms["evidence"]["representative_closed_coordinate_atom_time_jets"]
    ) == 4
    assert len(time_atoms["evidence"]["representative_closed_time_K55_bounds"]) == 10
    assert time_atoms["evidence"]["insufficient_H6_state_negative"]["status"] == "reject"

    bounded = checks["quartic_linear_x_compact_physical_frequency_defect"]
    assert bounded["status"] == "pass"
    assert bounded["evidence"]["artifact_hash_matches_reexecution"]
    assert bounded["evidence"]["standalone_artifact_validator_passed"]
    bounded_control = bounded["evidence"]["generic_compact_frequency_defect_control"]
    assert bounded_control["compact_symbol_Schur_lemma"]["exact_coefficient"] == "4/3"
    assert bounded_control["physical_scale_contract"]["high_shell_defect_zero"]
    assert bounded["evidence"]["wrong_physical_scale_negative"]["status"] == "reject"

    dyadic = checks["quartic_linear_x_H7_dyadic_localization_audit"]
    assert dyadic["status"] == "pass"
    assert dyadic["evidence"]["artifact_hash_matches_reexecution"]
    assert dyadic["evidence"]["standalone_artifact_validator_passed"]
    dyadic_control = dyadic["evidence"]["generic_dyadic_localization_control"]
    assert dyadic_control["partition"]["maximum_nonzero_ordinary_multipliers"] == 2
    assert dyadic_control["partition"][
        "ordinary_shells_interacting_with_one_enlarged_shell"
    ] == 5
    assert dyadic_control["partition"][
        "maximum_simultaneous_enlarged_multiplier_overlap"
    ] == 4
    assert dyadic_control["derivative_loss_negative"]["growth_exponent"] == 1
    assert dyadic["evidence"]["incompatible_regularity_contract_negative"][
        "status"
    ] == "reject"

    composition = checks["quartic_linear_x_anti_wick_composition_derivative_audit"]
    assert composition["status"] == "pass"
    assert composition["evidence"]["artifact_hash_matches_reexecution"]
    assert composition["evidence"]["standalone_artifact_validator_passed"]
    composition_control = composition["evidence"]["generic_anti_wick_composition_audit"]
    assert composition_control["anti_wick_to_weyl"]["heat_time"] == "h/4"
    assert composition_control["anti_wick_to_weyl"][
        "frequency_heat_transform_residual"
    ] == "0"
    assert composition_control["exact_composition_amplitude"][
        "FTOC_polynomial_residual"
    ] == "0"
    assert composition_control["amplitude_Schur_lemma"]["exact_coefficient"] == "1/(8*pi)"
    assert composition_control["derivative_audit"]["required_maximum_mixed_total_order"] == 6
    assert composition["evidence"]["false_C4_closure_negative"]["status"] == "reject"
    annular = checks["quartic_linear_x_annular_K55_C6_principal_composition"]
    assert annular["status"] == "pass"
    assert annular["evidence"]["standalone_artifact_validator_passed"]
    assert annular["evidence"]["counts"]["targeted_C6_bounds_passed"] == 12
    assert annular["evidence"]["counts"]["full_dyadic_energies_closed"] == 0
    assert annular["evidence"]["wrong_derivative_order_negative"]["status"] == "reject"
    assert checks["projected_aether_q_fixed_metric_first_variation"]["status"] == "pass"
    assert checks["projected_aether_q_fixed_metric_first_variation"]["evidence"][
        "projector_and_B_first_variation_residual"
    ] == "0"
    q_adm = checks["projected_aether_q_generic_3plus1_decomposition"]
    assert q_adm["status"] == "pass"
    assert q_adm["evidence"]["q_contraction_residual"] == "0"
    assert q_adm["evidence"]["generic_tilt_normal_normal_entry"] != "0"
    q_dirac = checks["projected_aether_q_aligned_auxiliary_dirac"]
    assert q_dirac["status"] == "pass"
    assert q_dirac["evidence"]["constraint_surface_rank"] == 4
    assert q_dirac["evidence"]["physical_dof_per_polarization"] == 1
    q_tilt = checks["projected_aether_q_constant_tilt_root_audit"]
    assert q_tilt["status"] == "pass"
    assert q_tilt["evidence"]["expanded_quartic_exact_real_root_count"] == 2
    assert q_tilt["evidence"]["expanded_quartic_nonreal_root_count"] == 2
    assert q_tilt["evidence"]["generic_tilt_hyperbolicity_status"] == "reject"
    assert q_tilt["evidence"]["negative_control"]["rejected"]
    x_family = checks["nonlinear_aether_acceleration_global_convexity"]
    assert x_family["status"] == "pass"
    assert x_family["evidence"]["negative_control"]["rejected"]
    assert x_family["evidence"]["high_field_F_over_X_limit_at_p_2_3"] == "0"
    x_no_go = checks["static_null_k14_multiplicative_completion_no_go"]
    assert x_no_go["status"] == "pass"
    assert x_no_go["evidence"]["symbolic_identity_residual"] == "0"
    assert all(
        item["negative_exactly"]
        for item in x_no_go["evidence"]["registered_matched_weight_witnesses"].values()
    )
    assert x_no_go["evidence"]["constant_weight_escape"]["rejected"]
    generic_horndeski_adm = checks[
        "generic_horndeski_l2_l4_unitary_adm_primary_degeneracy"
    ]
    assert generic_horndeski_adm["status"] == "pass"
    assert generic_horndeski_adm["evidence"]["adm_cancellation_residual"] == "0"
    assert generic_horndeski_adm["evidence"]["velocity_hessian_rank_on_regular_patch"] == 6
    assert generic_horndeski_adm["evidence"]["velocity_hessian_nullity_on_regular_patch"] == 1
    assert generic_horndeski_adm["evidence"]["negative_control"][
        "extra_kinetic_direction_restored"
    ]
    generic_horndeski_dirac = checks[
        "generic_horndeski_l2_l4_unitary_distributed_dirac_closure"
    ]
    assert generic_horndeski_dirac["status"] == "pass"
    assert generic_horndeski_dirac["evidence"][
        "constraint_matrix_rank_on_regular_patch"
    ] == 2
    assert generic_horndeski_dirac["evidence"]["constraint_count"]["physical_dof"] == 3
    assert generic_horndeski_dirac["evidence"]["singular_negative_control"][
        "three_mode_count_rejected"
    ]
    horndeski = checks["quartic_horndeski_covariant_adm_degeneracy"]
    assert horndeski["status"] == "pass"
    assert horndeski["evidence"]["adm_cancellation_residual"] == "0"
    assert horndeski["evidence"]["primary_null_vector"] == ["1", "0", "0"]
    assert horndeski["evidence"]["action_binding_passed"]
    assert len(horndeski["evidence"]["input_action_sha256"]) == 64
    horndeski_dirac = checks["quartic_horndeski_unitary_flrw_dirac_chain"]
    assert horndeski_dirac["status"] == "pass"
    assert horndeski_dirac["evidence"]["action_binding_passed"]
    assert horndeski_dirac["evidence"]["constraint_matrix_rank"] == 2
    assert horndeski_dirac["evidence"]["minisuperspace_physical_dof"] == 1
    assert (
        horndeski_dirac["evidence"]["conditional_full_field_count"]["status"]
        == "not_yet_a_distributed_closure_proof"
    )
    horndeski_distributed = checks[
        "quartic_horndeski_unitary_distributed_dirac_closure"
    ]
    assert horndeski_distributed["status"] == "pass"
    assert horndeski_distributed["evidence"]["action_binding_passed"]
    assert horndeski_distributed["evidence"]["constraint_count"]["physical_dof"] == 3
    assert horndeski_distributed["evidence"]["lapse_pair"]["rank_on_regular_patch"] == 2
    horndeski_principal = checks["quartic_horndeski_timelike_flat_principal_symbol"]
    assert horndeski_principal["status"] == "pass"
    assert horndeski_principal["evidence"]["action_binding_passed"]
    assert horndeski_principal["evidence"]["healthy_witness"]["values"][
        "tensor_speed_squared"
    ] == "7/9"
    assert all(
        item["rejected"]
        for item in horndeski_principal["evidence"]["negative_controls"].values()
    )
    horndeski_curved_scalar = checks[
        "quartic_horndeski_arbitrary_curvature_scalar_principal"
    ]
    assert horndeski_curved_scalar["status"] == "pass"
    assert horndeski_curved_scalar["evidence"]["action_binding_passed"]
    assert horndeski_curved_scalar["evidence"]["healthy_diagonal_witness"][
        "speed_squared"
    ] == ["3/4", "7/8", "5/6"]
    assert horndeski_curved_scalar["evidence"][
        "full_coupled_metric_scalar_principal_status"
    ] == "unresolved"
    horndeski_formulation = checks[
        "quartic_horndeski_coupled_formulation_hyperbolicity"
    ]
    assert horndeski_formulation["status"] == "pass"
    assert horndeski_formulation["evidence"]["action_binding_passed"]
    assert horndeski_formulation["evidence"]["generalized_harmonic"][
        "status"
    ] == "reject"
    assert horndeski_formulation["evidence"]["modified_harmonic"][
        "theorem_status"
    ] == "conditional_pass"
    assert horndeski_formulation["evidence"]["modified_harmonic"][
        "exact_cone_robustness_budget"
    ]["spectral_perturbation_budget"] == "19/72"
    assert horndeski_formulation["evidence"]["modified_harmonic"][
        "exact_cone_robustness_budget"
    ]["full_correction_norm_status"] == "unresolved"
    assert horndeski_formulation["evidence"]["action_specific_application"][
        "status"
    ] == "unresolved"
    horndeski_full_principal = checks[
        "quartic_horndeski_full_local_principal_extraction"
    ]
    assert horndeski_full_principal["status"] == "pass"
    assert horndeski_full_principal["evidence"]["action_binding_passed"]
    assert horndeski_full_principal["evidence"]["matrix_shape"] == [11, 11]
    assert all(
        horndeski_full_principal["evidence"]["block_certificates"].values()
    )
    assert horndeski_full_principal["evidence"][
        "flat_constant_timelike_gradient_reduction"
    ]["adm_tensor_residual"] == "0"
    assert horndeski_full_principal["evidence"][
        "first_order_generalized_pencil"
    ]["status"] == "pass"
    assert horndeski_full_principal["evidence"][
        "first_order_generalized_pencil"
    ]["einstein_scalar_flat_unit_direction"]["residual"] == "0"
    formal_time_block = horndeski_full_principal["evidence"][
        "first_order_generalized_pencil"
    ]["time_block_invertibility"]
    assert formal_time_block["status"] == "conditional_pass"
    assert formal_time_block["baseline_general_determinant"] == (
        "6561*M2**10/4096"
    )
    assert formal_time_block["sum_of_squares_residual"] == "0"
    assert formal_time_block["declared_gradient_only_domain_status"] == "unresolved"
    assert formal_time_block["curvature_collapse_negative_control"][
        "A_determinant"
    ] == "0"
    assert horndeski_full_principal["evidence"][
        "uniform_symmetrizer_and_norm_status"
    ] == "unresolved"
    horndeski_hamiltonian = checks[
        "quartic_horndeski_timelike_flat_physical_hamiltonian"
    ]
    assert horndeski_hamiltonian["status"] == "pass"
    assert horndeski_hamiltonian["evidence"]["action_binding_passed"]
    assert horndeski_hamiltonian["evidence"]["legendre_transform_residual"] == "0"
    assert horndeski_hamiltonian["evidence"]["healthy_witness"]["strictly_positive"]
    horndeski_no_go = checks["quartic_horndeski_global_timelike_gradient_no_go"]
    assert horndeski_no_go["status"] == "pass"
    assert horndeski_no_go["evidence"]["action_binding_passed"]
    assert not horndeski_no_go["evidence"][
        "global_all_amplitude_domain_exists_for_nonzero_alpha"
    ]
    assert horndeski_no_go["evidence"]["equivalent_bound"] == (
        "A_star^2 < M2/abs(alpha)"
    )
    horndeski_crossing = checks[
        "quartic_horndeski_flrw_background_domain_crossing"
    ]
    assert horndeski_crossing["status"] == "pass"
    assert horndeski_crossing["evidence"]["action_binding_passed"]
    assert horndeski_crossing["evidence"]["crossing_witness"][
        "healthy_boundary_time_derivative"
    ] == "sqrt(6)/2"
    assert not horndeski_crossing["evidence"][
        "forward_invariant_under_unrestricted_flrw_evolution"
    ]
    assert horndeski["evidence"]["negative_control"][
        "extra_kinetic_direction_restored"
    ]
    horndeski_variation = checks["quartic_horndeski_scalar_covariant_variation"]
    assert horndeski_variation["status"] == "pass"
    assert horndeski_variation["evidence"]["second_order_reduction"][
        "reduction_residual"
    ] == "0"
    horndeski_metric = checks["quartic_horndeski_metric_variation_and_noether"]
    assert horndeski_metric["status"] == "pass"
    assert horndeski_metric["evidence"]["boundary_and_flrw_noether"]["lapse_flrw"][
        "noether_residual"
    ] == "0"
    assert checks["proca_adm_dirac"]["evidence"]["physical_vector_dof"] == 3
    assert checks["einstein_hilbert_linearized_bianchi"]["evidence"]["residuals"] == [
        "0",
        "0",
        "0",
        "0",
    ]
    assert checks["einstein_hilbert_linearized_adm"]["evidence"]["physical_dof"] == 2
    nonlinear_adm = checks["nonlinear_adm_hamiltonian_constraint_algebra"]["evidence"]
    assert nonlinear_adm["passed"]
    assert nonlinear_adm["cross_contraction_residual"] == "0"
    assert nonlinear_adm["boundary_reduction_residual"] == "0"
    assert nonlinear_adm["wrong_dewitt_trace_negative_control"]["rejected"]
    assert nonlinear_adm["wrong_curvature_sign_negative_control"]["rejected"]
    curvature_diffeomorphism = checks["spatial_curvature_density_diffeomorphism_covariance"][
        "evidence"
    ]
    assert curvature_diffeomorphism["passed"]
    assert curvature_diffeomorphism["residual"] == "0"
    assert curvature_diffeomorphism["omitted_density_weight_negative_control"]["rejected"]
    assert checks["canonical_scalar_noether_identity"]["evidence"]["residuals"] == [
        "0",
        "0",
        "0",
        "0",
    ]
    generic_g2 = checks["generic_g2_variation_noether_identity"]
    assert generic_g2["status"] == "pass"
    assert generic_g2["evidence"]["local_jet_residuals"] == ["0", "0", "0", "0"]
    assert generic_g2["evidence"]["corrupted_sign_rejected"]
    generic_g3 = checks["generic_g3_variation_noether_identity"]
    assert generic_g3["status"] == "pass"
    assert generic_g3["evidence"]["third_derivative_cancellation_residual"] == "0"
    assert generic_g3["evidence"]["flat_arbitrary_third_jet_noether_residuals"] == [
        "0",
        "0",
        "0",
        "0",
    ]
    assert generic_g3["evidence"]["noether_residuals"] == ["0", "0", "0", "0"]
    assert generic_g3["evidence"]["omitted_braiding_stress_rejected"]
    assert generic_g3["evidence"]["omitted_ricci_commutator_rejected"]
    generic_g4_phi = checks["generic_g4_phi_variation_noether_identity"]
    assert generic_g4_phi["status"] == "pass"
    assert generic_g4_phi["evidence"]["noether_residuals"] == ["0"] * 4
    assert generic_g4_phi["evidence"]["omitted_metric_completion_rejected"]
    assert generic_g4_phi["evidence"]["wrong_scalar_sign_rejected"]
    generic_g4_scalar = checks["generic_g4_fixed_metric_scalar_variation"]
    assert generic_g4_scalar["status"] == "pass"
    assert generic_g4_scalar["evidence"]["flat_arbitrary_third_jet"][
        "surviving_third_derivatives"
    ] == {}
    assert generic_g4_scalar["evidence"]["flat_arbitrary_metric_scalar_noether"][
        "combined_noether_residuals"
    ] == ["0"] * 4
    assert generic_g4_scalar["evidence"]["flat_arbitrary_metric_scalar_noether"][
        "omitted_G4_XX_q_mu_q_nu_rejected"
    ]
    assert generic_g4_scalar["evidence"]["curved_linear_x_reduction"][
        "wrong_completion_rejected"
    ]
    generic_g4_curved = checks["generic_g4_curved_rnc_exact_witnesses"]
    assert generic_g4_curved["status"] == "pass"
    assert generic_g4_curved["evidence"]["generic_all_jet_theorem"].startswith(
        "proved_by_"
    )
    for witness in generic_g4_curved["evidence"]["witnesses"]:
        assert witness["combined_noether_residuals"] == ["0"] * 4
        assert witness["omitted_term_rejected"]
    generic_g4_symbolic = checks["generic_g4_curved_symbolic_all_jet_noether"]
    assert generic_g4_symbolic["status"] == "pass"
    assert generic_g4_symbolic["evidence"]["independent_local_data"][
        "total_independent_symbols"
    ] == 345
    assert generic_g4_symbolic["evidence"]["combined_noether_residuals"] == ["0"] * 4
    generic_g4_cadabra = checks["cadabra_generic_g4_metric_raw_variation"]
    assert generic_g4_cadabra["status"] == "pass"
    assert generic_g4_cadabra["evidence"]["return_code"] == 0
    assert "SIGMA_GENERIC_G4_METRIC_VARIATION_POLARIZATION_CERTIFIED" in generic_g4_cadabra[
        "evidence"
    ]["expected_fragments"]
    assert "SIGMA_GENERIC_G4_METRIC_VARIATION_OMITTED_PALATINI_REJECTED" in (
        generic_g4_cadabra["evidence"]["expected_fragments"]
    )
    assert "SIGMA_GENERIC_G4_METRIC_VARIATION_THIRD_DERIVATIVES_CANCELLED" in (
        generic_g4_cadabra["evidence"]["expected_fragments"]
    )
    scalar_constraint = checks["canonical_scalar_gravity_cross_constraint_identities"]["evidence"]
    assert scalar_constraint["passed"]
    assert scalar_constraint["local_frame_density_weight_residual"] == "0"
    assert scalar_constraint["gravity_matter_cross_hh_antisymmetry_residual"] == "0"
    proca_constraint = checks["proca_reduced_smeared_constraint_algebra"]["evidence"]
    assert proca_constraint["passed"]
    assert proca_constraint["primary_secondary_bracket"] == "-m_A**2*sqrt_h"
    assert proca_constraint["equal_modulo_spatial_boundary"]
    assert proca_constraint["euler_boundary_residuals"] == ["0"] * 8
    dhost = checks["dhost_degenerate_kinetic_block"]["evidence"]
    assert dhost["rank"] == 1
    assert dhost["constraint_matrix_rank"] == 2
    assert dhost["second_class_constraints"] == 2
    assert dhost["physical_scalar_dof"] == 1
    assert dhost["closure"]
    anisotropic = checks["anisotropic_principal_symbol_directions"]["evidence"]
    assert anisotropic["passed"]
    assert anisotropic["negative_control_axes_pass"]
    assert not anisotropic["negative_control_oblique_failure"]["gradient_stable"]
    extracted = checks["reduced_lagrangian_principal_extraction"]["evidence"]
    assert extracted["passed"]
    assert extracted["canonical_scalar"]["passed"]
    assert extracted["time_space_mixed_characteristics"]["status"] == "pass"
    uniform_scalar = checks["uniform_scalar_anisotropy_sphere"]["evidence"]
    assert uniform_scalar["passed"]
    assert uniform_scalar["stable_anisotropic"]["minimum_speed_squared"] == "1/4"
    assert not uniform_scalar["off_axis_unstable"]["gradient_stable"]
    uniform_multifield = checks["uniform_multifield_block_certificate"]["evidence"]
    assert uniform_multifield["passed"]
    assert uniform_multifield["stable_two_field"]["status"] == "pass"
    assert uniform_multifield["off_axis_inconclusive_negative_control"]["status"] == ("unresolved")
    assert uniform_multifield["superluminal_inconclusive_negative_control"]["status"] == (
        "unresolved"
    )
    assert checks["curved_background_principal_controls"]["evidence"]["passed"]
    assert checks["proca_divergence_identity"]["evidence"]["maxwell_divergence_residual"] == "0"
    assert checks["proca_stress_noether_identity"]["evidence"]["residuals"] == [
        "0",
        "0",
        "0",
        "0",
    ]
    curved_proca = checks["proca_curved_background_noether_identity"]["evidence"]
    assert curved_proca["flrw"]["residuals"] == ["0", "0", "0", "0"]
    assert curved_proca["static_spherical"]["residuals"] == ["0", "0", "0", "0"]
    aether_flrw = checks["einstein_aether_flrw_variation_noether"]["evidence"]
    assert aether_flrw["noether_residual"] == "0"
    assert aether_flrw["hessian_determinant"] == "0"
    assert aether_flrw["gauge_null_residual"] == ["0", "0", "0"]
    aether_adm = checks["einstein_aether_adm_kinetic_hessian"]["evidence"]
    assert aether_adm["aligned"]["hessian_rank"] == 10
    assert aether_adm["tilted"]["hessian_rank"] == 10
    assert aether_adm["aligned"]["expected_c14"] == "3/20"
    aether_diffeomorphism = checks["einstein_aether_spatial_diffeomorphism_algebra"]["evidence"]
    assert aether_diffeomorphism["passed"]
    assert aether_diffeomorphism["metric_sector_passed"]
    assert aether_diffeomorphism["canonical_coordinate_residuals"] == ["0"] * 4
    assert aether_diffeomorphism["canonical_momentum_residuals"] == ["0"] * 4
    assert aether_diffeomorphism["commutator_coordinate_residuals"] == ["0"] * 4
    assert aether_diffeomorphism["commutator_momentum_residuals"] == ["0"] * 4
    unit_chain = checks["unit_timelike_vector_dirac_chain"]["evidence"]
    assert unit_chain["passed"]
    assert unit_chain["constraint_matrix_rank"] == 4
    assert unit_chain["second_class_constraints"] == 4
    assert unit_chain["physical_dof"] == 3
    assert unit_chain["closure"]
    holonomic = checks["regular_holonomic_multiplier_dirac_theorem"]["evidence"]
    assert holonomic["passed"]
    assert holonomic["poisson_determinant"] == f"({holonomic['normality']})**4"
    assert holonomic["constraint_matrix_rank_on_regular_patch"] == 4
    assert holonomic["second_class_constraints"] == 4
    surface_rank = checks["dirac_constraint_surface_poisson_rank"]["evidence"]
    assert surface_rank["off_surface_rank"] == 2
    assert surface_rank["constraint_surface_rank"] == 0
    assert surface_rank["independent_constraints"] == 2
    tertiary = checks["dirac_tertiary_constraint_chain"]["evidence"]
    assert len(tertiary["primary_constraints"]) == 2
    assert len(tertiary["secondary_constraints"]) == 2
    assert len(tertiary["higher_generation_constraints"]) == 2
    assert tertiary["constraint_matrix_rank"] == 6
    assert tertiary["physical_dof"] == 0
    assert tertiary["closure"]
    field_dirac = checks["field_theory_smeared_constraint_algebra"]["evidence"]
    assert field_dirac["passed"]
    assert field_dirac["spatial_dimension"] == 1
    for case in field_dirac["cases"].values():
        assert case["equal_modulo_spatial_boundary"]
    three_dimensional = checks["three_spatial_dimensional_smeared_brackets"]["evidence"]
    assert three_dimensional["passed"]
    assert three_dimensional["spatial_dimension"] == 3
    assert three_dimensional["hamiltonian_hamiltonian"]["residual"] == "0"
    assert three_dimensional["diffeomorphism_diffeomorphism"]["equal_modulo_spatial_boundary"]
    metric_diffeomorphism = checks["canonical_metric_diffeomorphism_algebra"]["evidence"]
    assert metric_diffeomorphism["passed"]
    assert len(metric_diffeomorphism["canonical_pairs"]) == 6
    assert metric_diffeomorphism["metric_generator_residuals"] == ["0"] * 6
    assert metric_diffeomorphism["momentum_generator_residuals"] == ["0"] * 6
    assert metric_diffeomorphism["metric_commutator_residuals"] == ["0"] * 6
    assert metric_diffeomorphism["momentum_commutator_residuals"] == ["0"] * 6
    dewitt_kinetic = checks["canonical_metric_dewitt_kinetic_covariance"]["evidence"]
    assert dewitt_kinetic["passed"]
    assert dewitt_kinetic["canonical_metric_components"] == 6
    assert dewitt_kinetic["weight_one_lie_residual"] == "0"
    aether_2d = checks["einstein_aether_inhomogeneous_2d_noether"]["evidence"]
    assert aether_2d["expected_terms_present"]
    assert aether_2d["script_hash_matches"]
    assert set(aether_2d["term_residuals"]) == {"K1", "K2", "K3", "K4", "unit_constraint"}
    assert all(residuals == ["0", "0"] for residuals in aether_2d["term_residuals"].values())
    aether_4d = checks["einstein_aether_inhomogeneous_4d_numeric_noether"]["evidence"]
    assert aether_4d["expected_terms_present"]
    assert aether_4d["script_hash_matches"]
    assert aether_4d["sample_count"] == 15
    assert aether_4d["maximum_absolute_residual"] < 1.0e-12
    assert aether_4d["negative_control"]["rejected"]
    aether_energy = checks["einstein_aether_linearized_physical_energy"]
    assert aether_energy["status"] == "pass"
    assert aether_energy["evidence"]["passed"]
    assert sum(aether_energy["evidence"]["physical_modes"].values()) == 5
    assert all(
        item["positive_speed_negative_energy"]
        for item in aether_energy["evidence"]["speed_only_negative_controls"].values()
    )
    nonlinear_energy = checks["einstein_aether_restricted_nonlinear_total_energy"]
    assert nonlinear_energy["status"] == "pass"
    assert nonlinear_energy["evidence"]["passed"]
    assert nonlinear_energy["evidence"]["conformal_residual"] == "0"
    assert nonlinear_energy["evidence"]["boundary_charge_residual"] == "0"
    assert nonlinear_energy["evidence"]["generic_status"] == "unresolved"
    aether_principal = checks["einstein_aether_reduced_five_mode_principal_domain"]
    assert aether_principal["status"] == "pass"
    assert aether_principal["evidence"]["passed"]
    assert aether_principal["evidence"]["mode_count"] == 5
    assert len(aether_principal["evidence"]["necessary_and_sufficient_regular_domain"]) == 4
    assert len(aether_principal["evidence"]["singular_strata"]) == 6
    assert all(
        item["rejected"] for item in aether_principal["evidence"]["negative_controls"].values()
    )
    aether_tilt = checks["einstein_aether_global_tilt_legendre_strata"]
    assert aether_tilt["status"] == "pass"
    assert aether_tilt["evidence"]["passed"]
    assert aether_tilt["evidence"]["velocity_count"] == 9
    assert aether_tilt["evidence"]["determinant_residual"] == "0"
    assert aether_tilt["evidence"]["superluminal_tensor_threshold_control"][
        "rejected_as_regular_at_threshold"
    ]
    aether_covariant_hyperbolicity = checks[
        "einstein_aether_covariant_arbitrary_background_hyperbolicity"
    ]
    assert aether_covariant_hyperbolicity["status"] == "pass"
    assert aether_covariant_hyperbolicity["evidence"]["passed"]
    assert aether_covariant_hyperbolicity["evidence"]["physical_mode_count"] == 5
    assert aether_covariant_hyperbolicity["evidence"]["healthy_arbitrary_background_control"][
        "strongly_hyperbolic"
    ]
    if report["backends"]["cadabra2"]["available"]:
        aether_exact_4d = checks["einstein_aether_arbitrary_background_4d_noether"]
        assert aether_exact_4d["status"] == "pass"
        assert aether_exact_4d["evidence"]["passed"]
        assert "2 nabla^a E^(g)_ab" in aether_exact_4d["evidence"]["identity"]
        assert set(aether_exact_4d["evidence"]["scripts"]) == {
            "metric_euler",
            "vector_multiplier_euler",
            "euler_noether_coefficient",
            "action_diffeomorphism_covariance",
        }
        assert all(
            item["passed"] and item["sha256"]
            for item in aether_exact_4d["evidence"]["scripts"].values()
        )
        assert checks["cadabra_metric_contraction"]["status"] == "pass"
        assert checks["cadabra_canonical_scalar_variation"]["status"] == "pass"
        assert checks["cadabra_proca_variation"]["status"] == "pass"
        assert checks["cadabra_einstein_aether_vector_variation"]["status"] == "pass"
        assert checks["cadabra_einstein_aether_metric_variation"]["status"] == "pass"
        assert checks["cadabra_einstein_hilbert_metric_variation"]["status"] == "pass"
        curvature = checks["cadabra_adm_spatial_curvature_variation"]
        assert curvature["status"] == "pass"
        assert curvature["evidence"]["script_sha256"]
        assert "nabla^2 N" in curvature["evidence"]["euler_coefficient"]
        assert checks["cadabra_nonlinear_contracted_bianchi"]["status"] == "pass"
        assert checks["cadabra_canonical_scalar_metric_variation"]["status"] == "pass"
        assert checks["cadabra_proca_metric_variation"]["status"] == "pass"
    assert report["candidate_readiness"]["observational_gates_unsealed"] is False
