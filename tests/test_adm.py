from __future__ import annotations

from sigma_theory_compiler.adm import (
    linearized_einstein_hilbert_adm_control,
    nonlinear_adm_hamiltonian_constraint_control,
    spatial_curvature_density_diffeomorphism_control,
)


def test_linearized_einstein_hilbert_adm_control_has_two_healthy_modes() -> None:
    result = linearized_einstein_hilbert_adm_control()
    assert result["passed"], result
    assert result["hessian_rank"] == 6
    assert result["hessian_nullity"] == 4
    assert len(result["primary_constraints"]) == 4
    assert len(result["secondary_constraints"]) == 4
    assert result["constraint_matrix_rank"] == 0
    assert result["first_class_constraints"] == 8
    assert result["second_class_constraints"] == 0
    assert result["physical_dof"] == 2
    assert result["tt_hamiltonian_positive_definite"]
    assert result["tt_principal_symbol"]["passed"]


def test_nonlinear_adm_hamiltonian_constraints_close_with_metric_structure_function() -> None:
    result = nonlinear_adm_hamiltonian_constraint_control()
    assert result["passed"], result
    assert result["cross_contraction_residual"] == "0"
    assert result["antisymmetric_gradient_residual"] == "0"
    assert result["boundary_reduction_residual"] == "0"
    assert result["wrong_dewitt_trace_negative_control"]["rejected"]
    assert result["wrong_curvature_sign_negative_control"]["rejected"]


def test_spatial_curvature_potential_is_a_weight_one_density() -> None:
    result = spatial_curvature_density_diffeomorphism_control()
    assert result["passed"]
    assert result["residual"] == "0"
    assert result["omitted_density_weight_negative_control"]["rejected"]
