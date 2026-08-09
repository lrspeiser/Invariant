from __future__ import annotations

import sympy as sp

from sigma_theory_compiler.dirac import (
    analyze_quadratic_lagrangian,
    partial_velocity_solution,
    poisson_bracket,
    proca_fourier_dirac_control,
    reduce_poisson_matrix_on_constraint_surface,
    regular_holonomic_multiplier_dirac_control,
)


def test_dirac_multipliers_cannot_collide_with_physical_u_coordinates() -> None:
    u0, u1, u2, u3, lam = sp.symbols("u0 u1 u2 u3 lam")
    v0, v1, v2, v3, vlam = sp.symbols("v0 v1 v2 v3 vlam")
    unit_norm = -u0**2 + u1**2 + u2**2 + u3**2 + 1
    lagrangian = (v0**2 + v1**2 + v2**2 + v3**2) / 2 + lam * unit_norm
    result = analyze_quadratic_lagrangian(
        lagrangian,
        (u0, u1, u2, u3, lam),
        (v0, v1, v2, v3, vlam),
        max_constraint_generations=8,
    )
    assert len(result.constraint_generations) >= 4
    assert result.secondary_constraints == (unit_norm,)
    assert all(symbol not in (u0, u1, u2, u3, lam) for symbol in result.multiplier_solution)


def test_poisson_bracket_has_canonical_sign() -> None:
    q, p = sp.symbols("q p")
    assert poisson_bracket(q, p, (q,), (p,)) == 1
    assert poisson_bracket(p, q, (q,), (p,)) == -1


def test_constraint_surface_reduction_removes_structure_function_rank() -> None:
    q1, q2, p1, p2 = sp.symbols("q1 q2 p1 p2")
    constraints = (p1, p2 + q1 * p1)
    off_surface, on_surface, independent = reduce_poisson_matrix_on_constraint_surface(
        constraints, (q1, q2), (p1, p2)
    )
    assert off_surface.rank() == 2
    assert on_surface == sp.zeros(2)
    assert on_surface.rank() == 0
    assert independent == 2


def test_partial_legendre_transform_keeps_regular_kinetic_direction() -> None:
    q1, v1, v2, p1, p2 = sp.symbols("q1 v1 v2 p1 p2")
    hessian = sp.Matrix([[1, q1], [q1, q1**2]])
    solution, unresolved = partial_velocity_solution(
        hessian, sp.zeros(2, 1), (v1, v2), (p1, p2)
    )
    assert solution == {v1: p1 - q1 * v2}
    assert unresolved == (v2,)


def test_dirac_algorithm_iterates_through_tertiary_constraints() -> None:
    q1, q2, q3 = sp.symbols("q1 q2 q3")
    v1, v2, v3 = sp.symbols("v1 v2 v3")
    lagrangian = (v1 - q2) ** 2 / 2 + q3 * q1
    result = analyze_quadratic_lagrangian(
        lagrangian, (q1, q2, q3), (v1, v2, v3)
    )
    assert len(result.primary_constraints) == 2
    assert len(result.secondary_constraints) == 2
    assert len(result.higher_generation_constraints) == 2
    assert len(result.constraint_generations) == 3
    assert result.independent_constraints == 6
    assert result.second_class_constraints == 6
    assert result.first_class_constraints == 0
    assert result.physical_dof == 0
    assert result.closure


def test_regular_oscillator_has_no_constraints_and_one_dof() -> None:
    q, velocity, frequency = sp.symbols("q v omega", positive=True)
    result = analyze_quadratic_lagrangian(
        (velocity**2 - frequency**2 * q**2) / 2,
        (q,),
        (velocity,),
    )
    assert result.velocity_hessian.rank() == 1
    assert result.primary_constraints == ()
    assert result.secondary_constraints == ()
    assert result.physical_dof == 1
    assert result.closure


def test_nondegenerate_higher_derivative_oscillator_keeps_ostrogradsky_mode() -> None:
    q, auxiliary, multiplier = sp.symbols("q Q lambda_Q")
    velocity_q, velocity_auxiliary, velocity_multiplier = sp.symbols(
        "v_q v_Q v_lambda_Q"
    )
    frequency = sp.symbols("omega", positive=True)
    # First-order auxiliary form of L=ddot(q)^2/2-omega^2 q^2/2 with Q=dot(q).
    lagrangian = (
        velocity_auxiliary**2 / 2
        + multiplier * (velocity_q - auxiliary)
        - frequency**2 * q**2 / 2
    )
    result = analyze_quadratic_lagrangian(
        lagrangian,
        (q, auxiliary, multiplier),
        (velocity_q, velocity_auxiliary, velocity_multiplier),
        max_constraint_generations=8,
    )

    assert len(result.primary_constraints) == 2
    assert result.secondary_constraints == ()
    assert result.constraint_matrix_rank == 2
    assert result.second_class_constraints == 2
    assert result.physical_dof == 2
    assert result.closure
    p_q, p_auxiliary, _ = result.momenta
    reduced_hamiltonian = sp.factor(
        result.canonical_hamiltonian.subs(multiplier, p_q)
    )
    assert sp.diff(reduced_hamiltonian, p_q) == auxiliary
    assert sp.diff(reduced_hamiltonian, p_q, 2) == 0
    assert sp.simplify(
        reduced_hamiltonian
        - (2 * auxiliary * p_q + p_auxiliary**2 + frequency**2 * q**2) / 2
    ) == 0


def test_regular_holonomic_multiplier_chain_survives_arbitrary_kinetic_mixing() -> None:
    result = regular_holonomic_multiplier_dirac_control()
    assert result["passed"], result
    assert [len(generation) for generation in result["constraint_generations"]] == [1] * 4
    assert result["consistency_residuals"] == ["0", "0", "0"]
    assert result["poisson_determinant"] == f"({result['normality']})**4"
    assert result["constraint_matrix_rank_on_regular_patch"] == 4
    assert result["first_class_constraints"] == 0
    assert result["second_class_constraints"] == 4
    assert result["physical_dof_for_n_regular_coordinates"] == "n-1"


def test_proca_dirac_algorithm_closes_and_reduced_hamiltonian_is_positive() -> None:
    result = proca_fourier_dirac_control()
    assert result["hessian_rank"] == 3
    assert result["hessian_nullity"] == 1
    assert len(result["primary_constraints"]) == 1
    assert len(result["secondary_constraints"]) == 1
    assert result["constraint_matrix_rank"] == 2
    assert result["first_class_constraints"] == 0
    assert result["second_class_constraints"] == 2
    assert result["physical_dof"] == 3
    assert result["closure"]
    assert result["hamiltonian_positive_definite"]
