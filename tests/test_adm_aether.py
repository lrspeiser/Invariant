from __future__ import annotations

from sigma_theory_compiler.adm_aether import (
    einstein_aether_adm_kinetic_control,
    einstein_aether_coupled_unit_normal_control,
    einstein_aether_covariant_strong_hyperbolicity_control,
    einstein_aether_3plus1_decomposition_control,
    einstein_aether_spatial_diffeomorphism_control,
    einstein_aether_lapse_shift_constraint_seed_control,
    einstein_aether_generic_dh_covariance_control,
    einstein_aether_generic_hh_deformation_control,
    einstein_aether_global_tilt_legendre_control,
    einstein_aether_linearized_energy_control,
    einstein_aether_nonlinear_positive_energy_theorem_control,
    einstein_aether_reduced_principal_domain_control,
    maxwell_unit_aether_nonlinear_hamiltonian_control,
    unit_timelike_vector_dirac_chain_control,
)


def test_einstein_aether_pointwise_kinetic_hessian_control() -> None:
    result = einstein_aether_adm_kinetic_control()
    assert result["passed"]
    assert result["aligned"]["hessian_rank"] == 10
    assert result["tilted"]["hessian_rank"] == 10


def test_einstein_aether_mixed_unit_normal_remains_second_class() -> None:
    result = einstein_aether_coupled_unit_normal_control()
    assert result["passed"], result
    assert set(result["patches"]) == {"aligned", "axis_tilted", "oblique_tilted"}
    for patch in result["patches"].values():
        assert patch["unit_norm"] == "-1"
        assert patch["hessian_rank"] == 10
        assert patch["second_class_regular"]
        assert patch["constraint_normality"] != "0"
    negative = result["singular_coupling_negative_control"]
    assert negative["rejected"]
    assert negative["hessian_rank"] < negative["expected_full_rank"]


def test_einstein_aether_spatial_diffeomorphism_cotangent_lift_closes() -> None:
    result = einstein_aether_spatial_diffeomorphism_control()
    assert result["passed"]
    assert result["metric_sector_passed"]
    for key in (
        "canonical_coordinate_residuals",
        "canonical_momentum_residuals",
        "commutator_coordinate_residuals",
        "commutator_momentum_residuals",
    ):
        assert result[key] == ["0"] * 4
    assert result["omitted_momentum_density_weight_negative_control"]["rejected"]


def test_unit_timelike_vector_dirac_chain_has_four_second_class_constraints() -> None:
    result = unit_timelike_vector_dirac_chain_control()
    assert result["passed"], result
    assert [len(generation) for generation in result["constraint_generations"]] == [1] * 4
    assert result["constraint_matrix_rank"] == 4
    assert result["first_class_constraints"] == 0
    assert result["second_class_constraints"] == 4
    assert result["physical_dof"] == 3
    assert result["closure"]


def test_maxwell_unit_aether_closes_with_five_modes_but_is_unbounded() -> None:
    result = maxwell_unit_aether_nonlinear_hamiltonian_control()
    assert result["passed"], result
    assert result["hh_residual"] == ["0"] * 3
    assert result["normalization_negative_control"]["rejected"]
    assert result["degree_count"]["physical_dof"] == 5
    assert result["hamiltonian_stability"]["status"] == "reject"
    assert result["hamiltonian_stability"]["negative_control_point_verified"]


def test_generic_aether_exact_3plus1_blocks_and_legendre_patch() -> None:
    result = einstein_aether_3plus1_decomposition_control()
    assert result["passed"], result
    assert result["invariant_block_residuals"] == ["0"] * 4
    assert result["unit_branch"]["u_dot_acceleration_residual"] == "0"
    assert result["omitted_KA_transport_negative_control"]["rejected"]
    assert result["tilted_inhomogeneous_patch"]["hessian_rank"] == 9
    assert result["tilted_inhomogeneous_patch"]["legendre_residual"] == "0"
    lapse = result["lapse_linearity"]
    assert lapse["unit_normal_derivative_residual"] == "0"
    assert lapse["electric_lagrangian_acceleration_residuals"] == ["0"] * 3
    assert lapse["integration_residual"] == "0"


def test_generic_aether_lapse_shift_seed_constraints_without_overclaim() -> None:
    result = einstein_aether_lapse_shift_constraint_seed_control()
    assert result["passed"], result
    assert result["verified_constraint_generations"] == {
        "primary": 4,
        "secondary_seeds": 4,
    }
    assert result["classification"]["spatial_momentum_sector"].startswith(
        "first-class"
    )
    assert result["classification"]["hamiltonian_sector"].startswith("unresolved")
    assert result["classification"]["physical_dof"].startswith("unresolved")


def test_generic_aether_legendre_density_closes_dh_bracket() -> None:
    result = einstein_aether_generic_dh_covariance_control()
    assert result["passed"], result
    assert result["lagrangian_density_weight_residual"] == "0"
    assert result["canonical_pairing_weight_residual"] == "0"
    assert result["legendre_hamiltonian_weight_residual"] == "0"
    assert result["local_lie_divergence_residual"] == "0"
    for negative in result["negative_controls"].values():
        assert negative["rejected"]


def test_generic_aether_normal_deformation_closes_hh_on_regular_patches() -> None:
    result = einstein_aether_generic_hh_deformation_control()
    assert result["passed"], result
    for residual in result["kinematic_residuals"].values():
        assert residual in {
            "0",
            "Matrix([[0, 0, 0]])",
            "Matrix([[0], [0], [0]])",
            "Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])",
            "Matrix([[0], [0], [0], [0]])",
        }
    assert result["degree_count"]["reduced_positive_unit_branch"]["physical_dof"] == 5
    assert result["degree_count"]["unreduced_multiplier_chart"]["physical_dof"] == 5
    for negative in result["negative_controls"].values():
        assert negative["rejected"]


def test_aether_physical_mode_energy_rejects_speed_only_ghosts() -> None:
    result = einstein_aether_linearized_energy_control()
    assert result["passed"], result
    assert sum(result["physical_modes"].values()) == 5
    for value in result["healthy_control_point"]["energy_coefficients"].values():
        assert value not in {"0"}
    for witness in result["speed_only_negative_controls"].values():
        assert witness["positive_speed_negative_energy"]
    nonlinear = result["nonlinear_positive_energy_subsector"]
    assert nonlinear["status"] == "pass_in_restricted_subsector"
    assert nonlinear["coupling_domain"] == ["0 <= c14 <= 2", "c13 <= 1"]


def test_aether_restricted_nonlinear_total_energy_theorem_is_executable_and_scoped() -> None:
    result = einstein_aether_nonlinear_positive_energy_theorem_control()
    assert result["passed"], result
    assert result["energy_not_local_density"]
    assert result["conformal_residual"] == "0"
    assert result["boundary_charge_residual"] == "0"
    assert result["interior_positivity_parameterization"]["positive"]
    assert all(value == "0" for value in result["endpoint_coefficients"].values())
    assert result["theorem_status"] == "pass_in_restricted_subsector"
    assert result["generic_status"] == "unresolved"
    assert all(
        item["theorem_premise_rejected"]
        for item in result["out_of_domain_controls"].values()
    )


def test_aether_five_mode_principal_domain_is_exact_and_fail_closed() -> None:
    result = einstein_aether_reduced_principal_domain_control()
    assert result["passed"], result
    assert result["mode_count"] == 5
    assert result["propagation_residual"] == "Matrix([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])"
    assert result["necessary_and_sufficient_regular_domain"] == [
        "1-c13 > 0",
        "0 < c14 < 2",
        "2c1-c1^2+c3^2 > 0",
        "c123(2+c13+3c2) > 0",
    ]
    assert all(result["positivity_certificate"].values())
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert set(result["singular_strata"]) == {
        "tensor_legendre",
        "vector_legendre",
        "scalar_trace_legendre",
        "spin_1_gradient",
        "spin_0_amplitude",
        "spin_0_gradient",
    }


def test_aether_global_tilt_legendre_strata_match_characteristic_cones() -> None:
    result = einstein_aether_global_tilt_legendre_control()
    assert result["passed"]
    assert result["velocity_count"] == 9
    assert result["determinant_residual"] == "0"
    assert result["aligned_determinant_residual"] == "0"
    assert result["sector_multiplicities"] == {"spin_2": 2, "spin_1": 2, "spin_0": 1}
    assert all(value == "0" for value in result["threshold_identity_residuals"].values())
    threshold = result["superluminal_tensor_threshold_control"]
    assert threshold["thresholds"]["spin_2"] == "9"
    assert threshold["ranks"] == {"0": 9, "8": 9, "9": 7, "10": 9}
    assert threshold["rejected_as_regular_at_threshold"]
    assert result["globally_subluminal_control"]["globally_noncharacteristic"]


def test_aether_covariant_arbitrary_background_hyperbolicity_is_sufficient_and_fail_closed() -> None:
    result = einstein_aether_covariant_strong_hyperbolicity_control()
    assert result["passed"], result
    assert result["physical_mode_count"] == 5
    assert all(value == "0" for value in result["cone_residuals"].values())
    assert result["lorentz_covariance_control"]["metric_residual"] == "Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])"
    assert result["lorentz_covariance_control"]["cone_scalar_residual"] == "0"
    assert result["quasilinear_principal_structure"][
        "exact_action_hessian_velocity_independent"
    ]
    assert result["healthy_arbitrary_background_control"]["strongly_hyperbolic"]
    assert all(
        witness["excluded_by_sufficient_theorem"]
        and witness["theory_status"] == "unresolved_by_this_formulation_not_rejected"
        for witness in result["formulation_boundary_controls"].values()
    )
    assert all(item["rejected"] for item in result["instability_controls"].values())
