from __future__ import annotations

import sympy as sp

from sigma_theory_compiler.field_dirac import (
    canonical_metric_dewitt_kinetic_control,
    canonical_metric_diffeomorphism_control,
    euler_operator,
    euler_operator_nd,
    proca_reduced_smeared_constraint_control,
    smeared_poisson_density,
    three_dimensional_smeared_bracket_control,
    virasoro_constraint_algebra_control,
)


def test_euler_operator_annihilates_total_spatial_derivative() -> None:
    x = sp.symbols("x")
    field = sp.Function("q")(x)
    density = sp.diff(field**2 * sp.diff(field, x), x)
    assert euler_operator(density, field, x) == 0


def test_smeared_canonical_bracket_uses_variational_derivatives() -> None:
    x = sp.symbols("x")
    field = sp.Function("q")(x)
    momentum = sp.Function("p")(x)
    left = sp.Function("N")(x) * field
    right = sp.Function("M")(x) * momentum
    assert smeared_poisson_density(left, right, (field,), (momentum,), x) == (
        sp.Function("M")(x) * sp.Function("N")(x)
    )


def test_exact_smeared_virasoro_constraint_algebra_closes() -> None:
    result = virasoro_constraint_algebra_control()
    assert result["passed"]
    assert set(result["cases"]) == {
        "diffeomorphism_diffeomorphism",
        "hamiltonian_hamiltonian",
        "diffeomorphism_hamiltonian",
    }
    for case in result["cases"].values():
        assert case["equal_modulo_spatial_boundary"]
        assert case["euler_boundary_residuals"] == ["0"] * 6


def test_multidimensional_euler_operator_annihilates_divergence() -> None:
    x, y, z = sp.symbols("x y z")
    field = sp.Function("q")(x, y, z)
    density = sp.diff(field * sp.diff(field, y), x) + sp.diff(field**2, z)
    assert euler_operator_nd(density, field, (x, y, z)) == 0


def test_three_spatial_dimensional_smeared_brackets_close() -> None:
    result = three_dimensional_smeared_bracket_control()
    assert result["passed"]
    assert result["spatial_dimension"] == 3
    assert result["hamiltonian_hamiltonian"]["residual"] == "0"
    diffeomorphism = result["diffeomorphism_diffeomorphism"]
    assert diffeomorphism["equal_modulo_spatial_boundary"]
    assert diffeomorphism["euler_boundary_residuals"] == ["0"] * 8


def test_reduced_massive_proca_smeared_hh_bracket_closes() -> None:
    result = proca_reduced_smeared_constraint_control()
    assert result["passed"]
    assert result["primary_secondary_bracket"] == "-m_A**2*sqrt_h"
    assert result["equal_modulo_spatial_boundary"]
    assert result["euler_boundary_residuals"] == ["0"] * 8
    assert result["physical_vector_dof"] == 3
    assert result["reduced_hamiltonian_positive"]


def test_canonical_metric_diffeomorphism_generator_and_algebra_close() -> None:
    result = canonical_metric_diffeomorphism_control()
    assert result["passed"]
    assert result["spatial_dimension"] == 3
    assert len(result["canonical_pairs"]) == 6
    for key in (
        "metric_generator_residuals",
        "momentum_generator_residuals",
        "metric_commutator_residuals",
        "momentum_commutator_residuals",
    ):
        assert result[key] == ["0"] * 6


def test_dewitt_kinetic_density_is_a_weight_one_spatial_density() -> None:
    result = canonical_metric_dewitt_kinetic_control()
    assert result["passed"]
    assert result["canonical_metric_components"] == 6
    assert result["weight_one_lie_residual"] == "0"
