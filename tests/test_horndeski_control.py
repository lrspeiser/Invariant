from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.adm_ir import compile_adm_ir
from sigma_theory_compiler.dhost import (
    quartic_horndeski_covariant_adm_control,
    quartic_horndeski_unitary_flrw_dirac_control,
)
from sigma_theory_compiler.dirac_ir import compile_dirac_ir
from sigma_theory_compiler.horndeski import (
    generic_cubic_horndeski_bssn_hyperbolicity_control,
    generic_horndeski_l2_l4_flrw_scalar_reduction_control,
    generic_horndeski_l2_l4_tensor_stability_control,
    generic_horndeski_l2_l4_unitary_dirac_control,
    generic_kessence_nonlinear_adm_legendre_control,
    generic_kessence_timelike_principal_hamiltonian_control,
    quartic_horndeski_flrw_domain_crossing_control,
    quartic_horndeski_global_timelike_gradient_no_go_control,
    quartic_horndeski_timelike_flat_principal_control,
    quartic_horndeski_unitary_distributed_dirac_control,
)


def test_generic_kessence_effective_metric_and_hamiltonian_are_exact() -> None:
    passed, result = generic_kessence_timelike_principal_hamiltonian_control()
    assert passed, result
    assert result["scalar_kinetic"] == "G2_X + 2*G2_XX*X"
    assert result["scalar_gradient"] == "G2_X"
    assert result["canonical_momentum_residual"] == "0"
    assert result["effective_metric_determinant_residual"] == "0"
    assert result["legendre_residual"] == "0"
    assert result["hamiltonian_hessian_residual"] == "Matrix([[0, 0], [0, 0]])"
    assert result["healthy_witness"]["values"] == {
        "scalar_kinetic": "3",
        "scalar_gradient": "2",
        "speed_squared": "2/3",
    }
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert "route to modified harmonic" in result["formulation_boundary"][
        "nonzero_canonical_G3_or_G4_X"
    ]


def test_generic_kessence_nonlinear_adm_legendre_map_is_exact() -> None:
    passed, result = generic_kessence_nonlinear_adm_legendre_control()
    assert passed, result
    assert result["canonical_momentum_density"] == "G2_X*v_n"
    assert result["legendre_jacobian"] == "G2_X + G2_XX*v_n**2"
    assert result["hamiltonian_density"] == "-G2 + G2_X*v_n**2"
    assert result["dH_dp_residual"] == "0"
    assert result["inverse_hessian_residual"] == "0"
    assert result["canonical_scalar_control"]["hamiltonian"] == (
        "(s_squared + v_n**2)/2"
    )
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert result["capability_boundary"]["global_gravitational_positive_energy"] == (
        "unresolved"
    )


def test_generic_cubic_horndeski_bssn_hyperbolicity_contract_is_exact() -> None:
    passed, result = generic_cubic_horndeski_bssn_hyperbolicity_control()
    assert passed, result
    assert result["source_conditions"]["momentum_constraint_parameter"] == "m > 1/4"
    assert result["source_conditions"]["slicing_parameter"] == (
        "suitable sigma > 1/2"
    )
    assert result["healthy_parameter_witness"]["squared_speeds"] == {
        "transverse": "1",
        "momentum": "1",
        "slicing": "2",
        "longitudinal": "1",
        "scalar": "1",
    }
    assert len(result["weak_field_derivative_ledger"]["G2"]) == 6
    assert len(result["weak_field_derivative_ledger"]["G3"]) == 9
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert result["candidate_contract"][
        "universal_numeric_threshold_for_much_less_than"
    ] == "not supplied by source"


def test_generic_horndeski_l2_l4_flrw_scalar_constraints_reduce_exactly() -> None:
    passed, result = generic_horndeski_l2_l4_flrw_scalar_reduction_control()
    assert passed, result
    assert result["physical_basis"] == ["curvature_scalar_zeta"]
    assert result["constraints"]["residuals"] == {
        "lapse": "0",
        "shift": "0",
        "lapse_on_solution": "0",
        "shift_on_solution": "0",
    }
    assert result["before_integration_by_parts_residual"] == "0"
    assert result["integration_by_parts_residual"] == "0"
    assert result["legendre_residual"] == "0"
    assert result["hamiltonian_hessian_residual"] == "Matrix([[0, 0], [0, 0]])"
    assert result["healthy_witness"]["values"] == {
        "G_S": "2",
        "F_S": "3",
        "c_S_squared": "3/2",
    }
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert result["capability_boundary"]["candidate_background_sign_proof"] == (
        "required"
    )


def test_generic_horndeski_l2_l4_tensor_block_is_healthy_on_its_regular_patch() -> None:
    passed, result = generic_horndeski_l2_l4_tensor_stability_control()
    assert passed, result
    assert result["physical_basis"] == ["tensor_plus", "tensor_cross"]
    assert result["G_T"] == "2*(G4 - 2*G4_X*X)"
    assert result["F_T"] == "2*G4"
    assert result["tensor_speed_squared"] == "G4/(G4 - 2*G4_X*X)"
    assert result["canonical_momentum_residual"] == "Matrix([[0], [0]])"
    assert result["legendre_residual"] == "0"
    assert result["hamiltonian_hessian_residual"] == "Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])"
    assert result["healthy_witness"]["values"] == {
        "G_T": "5",
        "F_T": "4",
        "c_T_squared": "4/5",
    }
    assert all(item["rejected"] for item in result["negative_controls"].values())
    assert result["capability_boundary"]["generic_scalar_principal_symbol"] == (
        "unresolved"
    )


def test_generic_horndeski_l2_l4_dirac_chain_closes_on_regular_lapse_patches() -> None:
    passed, result = generic_horndeski_l2_l4_unitary_dirac_control()
    assert passed, result
    assert result["adm_primary_input"]["hessian_rank"] == 6
    assert result["constraint_matrix_rank_on_regular_patch"] == 2
    assert result["constraint_count"]["physical_dof"] == 3
    assert result["spatial_diffeomorphism_residuals"] == {
        "lapse": "0",
        "lapse_momentum": "0",
        "D_D_lapse": "0",
        "D_D_lapse_momentum": "0",
        "D_C_secondary_density": "0",
    }
    assert result["regular_family_witness"]["G2_lapse_hessian"] == "N**(-3)"
    assert result["singular_negative_control"]["three_mode_count_rejected"]
    assert result["capability_boundary"]["global_Delta_N_invertibility"] == "unresolved"
from sigma_theory_compiler.legendre_ir import compile_legendre_ir
from sigma_theory_compiler.stability_ir import compile_stability_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"
SPEC = ROOT / "configs" / "actions" / "quartic_horndeski_control.json"


def test_named_horndeski_action_is_compiled_and_bound_to_its_covariant_invariant() -> None:
    action = compile_action_file(SPEC, GRAMMAR, CONTRACT)
    assert action["valid"], action["errors"]
    assert action["canonical"]["matter_metric"] == "g_mu_nu"
    assert action["canonical"]["background_domain"]["variables"][0]["id"] == (
        "A_star_squared"
    )
    assert action["canonical"]["background_domain"]["preservation"]["status"] == (
        "rejected"
    )
    terms = {item["id"]: item for item in action["canonical"]["terms"]}
    assert terms["HORNDESKI_L4_LINEAR_X"]["invariant"] == "H4_phi"
    assert terms["HORNDESKI_L4_LINEAR_X"]["maximum_derivatives_per_field"] == 2


def test_named_horndeski_action_has_verified_adm_and_legendre_degeneracy() -> None:
    action = compile_action_file(SPEC, GRAMMAR, CONTRACT)
    control = quartic_horndeski_covariant_adm_control()
    adm = compile_adm_ir(
        action,
        {
            "quartic_horndeski_covariant_adm_degeneracy": control["passed"],
            "canonical_scalar": True,
            "cadabra_adm_spatial_curvature_variation": True,
            "nonlinear_adm_hamiltonian_constraint_algebra": True,
        },
    )
    legendre = compile_legendre_ir(action, adm)
    assert adm["status"] == "pass"
    assert "V_star" not in adm["velocity_channels"]
    assert adm["primary_constraint_seeds"] == ["p_N=0", "p_(N^i)=0 (three components)"]
    assert legendre["status"] == "pass"
    assert legendre["velocity_order"][-1] == "V_star"
    assert legendre["generic_hessian_rank"] == 6
    assert legendre["generic_hessian_nullity"] == 1
    assert legendre["legendre_status"] == "degenerate_primary_verified"
    assert legendre["kinetic_primary_constraints"] == ["p_V_star"]
    assert legendre["unsupported_kinetic_terms"] == []
    dirac = compile_dirac_ir(
        action,
        adm,
        legendre,
        {
            "quartic_horndeski_covariant_adm_degeneracy": True,
            "quartic_horndeski_unitary_flrw_dirac_chain": True,
            "quartic_horndeski_unitary_distributed_dirac_closure": True,
        },
    )
    stability = compile_stability_ir(action, dirac, {})
    assert dirac["status"] == "pass"
    assert dirac["local_canonical_transform"]["status"] == "pass"
    assert dirac["local_canonical_transform"]["primary_constraints"] == ["P_A_star"]
    assert dirac["distributed_constraint_closure"]["family"] == "quartic_horndeski"
    assert dirac["distributed_constraint_closure"]["missing_or_failed_controls"] == []
    assert dirac["distributed_constraint_closure"]["status"] == "pass"
    assert (
        dirac["distributed_constraint_closure"]["constraint_surface_rank"]["physical_dof"]
        == 3
    )
    assert dirac["unitary_gauge_local_dirac_control"]["passed"]
    assert stability["status"] == "unresolved"
    assert stability["family"] == "quartic_horndeski"
    assert stability["condition_certificate"]["pointwise_status"] == "pass"
    assert stability["condition_certificate"]["status"] == "unresolved"
    assert stability["background_domain"]["variables"][0]["id"] == (
        "A_star_squared"
    )
    preservation = stability["condition_certificate"][
        "background_domain_preservation"
    ]
    assert preservation["status"] == "unresolved"
    assert preservation["missing_or_failed_controls"] == [
        "quartic_horndeski_flrw_background_domain_crossing",
        "quartic_horndeski_metric_variation_and_noether",
    ]


def test_named_horndeski_curved_flrw_lapse_chain_is_second_class_on_regular_patch() -> None:
    control = quartic_horndeski_unitary_flrw_dirac_control()
    assert control["passed"]
    assert control["velocity_hessian_rank"] == 1
    assert control["primary_constraint"] == "p_h4_0"
    assert control["constraint_matrix_rank"] == 2
    assert control["first_class_constraints"] == 0
    assert control["second_class_constraints"] == 2
    assert control["minisuperspace_physical_dof"] == 1
    assert control["closure"]
    assert control["canonical_scalar_boundary_control"]["surface_pairing"] != "0"
    assert (
        control["conditional_full_field_count"][
            "physical_dof_if_distributed_spatial_chain_closes"
        ]
        == 3
    )
    assert (
        control["conditional_full_field_count"]["status"]
        == "not_yet_a_distributed_closure_proof"
    )


def test_named_horndeski_flat_timelike_principal_block_has_three_healthy_modes() -> None:
    passed, control = quartic_horndeski_timelike_flat_principal_control()
    assert passed
    assert control["physical_basis"] == ["tensor_plus", "tensor_cross", "scalar"]
    assert control["healthy_witness"]["values"]["tensor_speed_squared"] == "7/9"
    assert control["healthy_witness"]["values"]["scalar_kinetic"] == "1"
    assert all(item["rejected"] for item in control["negative_controls"].values())
    assert "arbitrary-background strong hyperbolicity remain separate" in control["scope"]


def test_named_horndeski_unitary_distributed_dirac_closes_on_regular_lapse_patches() -> None:
    passed, control = quartic_horndeski_unitary_distributed_dirac_control()
    assert passed
    assert control["spatial_diffeomorphism"]["metric_cotangent_lift_passed"]
    assert control["secondary_density_covariance"]["residual"] == "0"
    assert control["lapse_pair"]["rank_on_regular_patch"] == 2
    assert control["lapse_pair"]["higher_constraints"] == []
    assert control["constraint_count"]["first_class_constraints"] == 6
    assert control["constraint_count"]["second_class_constraints"] == 2
    assert control["constraint_count"]["physical_dof"] == 3
    assert control["singular_negative_control"]["rejected_as_regular_patch"]
    assert "global operator invertibility" in control["scope"]


def test_named_horndeski_has_no_global_all_timelike_gradient_domain_for_nonzero_alpha() -> None:
    passed, control = quartic_horndeski_global_timelike_gradient_no_go_control()
    assert passed
    assert not control["global_all_amplitude_domain_exists_for_nonzero_alpha"]
    assert control["equivalent_bound"] == "A_star^2 < M2/abs(alpha)"
    assert control["positive_alpha_branch"]["tensor_kinetic_at_boundary"] == "0"
    assert control["positive_alpha_branch"]["tensor_kinetic_above_boundary"] == "-M2/2"
    assert control["negative_alpha_branch"]["tensor_gradient_at_boundary"] == "0"
    assert control["negative_alpha_branch"]["tensor_gradient_above_boundary"] == "-M2/2"


def test_named_horndeski_flrw_evolution_crosses_the_healthy_domain_boundary() -> None:
    passed, control = quartic_horndeski_flrw_domain_crossing_control()
    assert passed
    witness = control["crossing_witness"]
    assert witness["lapse_constraint_residual"] == "0"
    assert witness["scale_euler_residual"] == "0"
    assert witness["constraint_flow_residual"] == "0"
    assert witness["evolution_matrix_determinant"] == "2112"
    assert witness["healthy_boundary_function"] == "0"
    assert witness["healthy_boundary_time_derivative"] == "sqrt(6)/2"
    assert witness["tensor_kinetic_at_boundary"] == "1"
    assert witness["tensor_gradient_at_boundary"] == "0"
    assert witness["tensor_gradient_time_derivative"] == "-sqrt(6)/4"
    assert not control["forward_invariant_under_unrestricted_flrw_evolution"]
