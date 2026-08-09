from __future__ import annotations

from sigma_theory_compiler.dhost import (
    dhost_reduced_dirac_control,
    generic_horndeski_l2_l4_unitary_adm_control,
    quartic_horndeski_covariant_adm_control,
)


def test_generic_horndeski_l2_l4_has_one_primary_adm_null_direction() -> None:
    result = generic_horndeski_l2_l4_unitary_adm_control()
    assert result["passed"], result
    assert result["adm_cancellation_residual"] == "0"
    assert result["normal_hessian_velocity_coefficient"] == "0"
    assert result["velocity_hessian_rank_on_regular_patch"] == 6
    assert result["velocity_hessian_nullity_on_regular_patch"] == 1
    assert result["primary_null_vector"] == ["1", "0", "0", "0", "0", "0", "0"]
    assert result["primary_constraint"] == "p_V_star=0"
    assert result["regular_patch"] == "G4-2 X G4_X != 0"
    assert result["negative_control"]["extra_kinetic_direction_restored"]
    assert (
        result["capability_boundary"]["secondary_constraint"]
        == "unresolved_for_arbitrary_G2_G3_G4"
    )


def test_named_quartic_horndeski_covariant_action_has_the_adm_primary_null_direction() -> None:
    result = quartic_horndeski_covariant_adm_control()
    assert result["passed"], result
    assert result["adm_cancellation_residual"] == "0"
    assert result["normal_hessian_velocity_coefficient"] == "0"
    assert result["velocity_hessian_determinant"] == "0"
    assert result["velocity_hessian_rank"] == 2
    assert result["primary_null_vector"] == ["1", "0", "0"]
    assert result["negative_control"]["determinant_identity_residual"] == "0"
    assert result["negative_control"]["extra_kinetic_direction_restored"]


def test_dhost_degeneracy_generates_second_class_pair_and_removes_extra_mode() -> None:
    result = dhost_reduced_dirac_control()
    assert result["passed"], result
    assert result["hessian_determinant"] == "0"
    assert result["hessian_rank"] == 1
    assert result["hessian_null_vector"] == ["-alpha", "1"]
    assert [len(generation) for generation in result["constraint_generations"]] == [1, 1]
    assert result["constraint_matrix_rank"] == 2
    assert result["first_class_constraints"] == 0
    assert result["second_class_constraints"] == 2
    assert result["physical_scalar_dof"] == 1
    assert result["closure"]
    assert result["primary_secondary_bracket"] == result["regularity_factor"]
    assert all(
        coefficient not in {"0", "-1"}
        for coefficient in result["reduced_hamiltonian_positive_coefficients"].values()
    )


def test_nondegenerate_dhost_control_restores_extra_scalar_mode() -> None:
    negative = dhost_reduced_dirac_control()["nondegenerate_negative_control"]
    assert negative["hessian_determinant"] == "C*epsilon"
    assert negative["hessian_rank"] == 2
    assert negative["primary_constraints"] == []
    assert negative["physical_scalar_dof"] == 2
    assert negative["rejected_as_degenerate"]
