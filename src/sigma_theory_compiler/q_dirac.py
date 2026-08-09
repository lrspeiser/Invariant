from __future__ import annotations

from typing import Any

import sympy as sp


def _poisson(left: sp.Expr, right: sp.Expr, q: tuple[sp.Symbol, ...], p: tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.factor(
        sum(
            sp.diff(left, coordinate) * sp.diff(right, momentum)
            - sp.diff(left, momentum) * sp.diff(right, coordinate)
            for coordinate, momentum in zip(q, p, strict=True)
        )
    )


def projected_aether_q_aligned_auxiliary_dirac_control() -> tuple[bool, dict[str, Any]]:
    """Complete Dirac reduction for one aligned Fourier polarization of the lifted Q sector."""

    kinetic, dispersive, gradient, wave_number = sp.symbols(
        "K0 K2 G k", positive=True, real=True
    )
    effective_kinetic = kinetic + dispersive * wave_number**2
    v, b, multiplier = sp.symbols("v b r", real=True)
    dot_v, dot_b, dot_multiplier = sp.symbols("dot_v dot_b dot_r", real=True)
    p_v, p_b, p_multiplier = sp.symbols("p_v p_b p_r", real=True)
    coordinates = (v, b, multiplier)
    momenta = (p_v, p_b, p_multiplier)
    lagrangian = (
        effective_kinetic * b**2 / 2
        - gradient * wave_number**2 * v**2 / 2
        + multiplier * (b - dot_v)
    )
    canonical_momenta = (
        sp.diff(lagrangian, dot_v),
        sp.diff(lagrangian, dot_b),
        sp.diff(lagrangian, dot_multiplier),
    )
    primary = (p_v + multiplier, p_b, p_multiplier)
    canonical_hamiltonian = (
        -effective_kinetic * b**2 / 2
        + gradient * wave_number**2 * v**2 / 2
        - multiplier * b
    )
    secondary = effective_kinetic * b + multiplier
    constraints = (*primary, secondary)
    poisson_matrix = sp.Matrix(
        [
            [_poisson(left, right, coordinates, momenta) for right in constraints]
            for left in constraints
        ]
    )
    poisson_determinant = sp.factor(poisson_matrix.det())
    reduced_hamiltonian = sp.factor(
        canonical_hamiltonian.subs(
            {multiplier: -p_v, b: p_v / effective_kinetic}
        )
    )
    expected_reduced = (
        p_v**2 / (2 * effective_kinetic)
        + gradient * wave_number**2 * v**2 / 2
    )
    dispersion = sp.factor(gradient * wave_number**2 / effective_kinetic)
    consistency_roles = {
        "p_v+r": "fixes the p_r multiplier",
        "p_b": "generates K(k)b+r=0",
        "p_r": "fixes the p_v+r multiplier to b",
        "K(k)b+r": "fixes the p_b multiplier; chain closes",
    }
    passed = (
        canonical_momenta == (-multiplier, 0, 0)
        and poisson_determinant == effective_kinetic**2
        and poisson_matrix.rank() == 4
        and sp.factor(reduced_hamiltonian - expected_reduced) == 0
    )
    return passed, {
        "quadratic_lagrangian": str(lagrangian),
        "effective_kinetic": str(effective_kinetic),
        "canonical_momenta": [str(value) for value in canonical_momenta],
        "primary_constraints": [str(value) for value in primary],
        "secondary_constraint": str(secondary),
        "consistency_roles": consistency_roles,
        "constraint_poisson_matrix": str(poisson_matrix),
        "constraint_poisson_determinant": str(poisson_determinant),
        "constraint_surface_rank": int(poisson_matrix.rank()),
        "second_class_constraints": 4,
        "first_class_constraints": 0,
        "phase_space_dimension": 6,
        "physical_dof_per_polarization": 1,
        "reduced_hamiltonian": str(reduced_hamiltonian),
        "positivity_domain": "K0>0, K2>0, G>0, k^2>=0",
        "dispersion_omega_squared": str(dispersion),
        "claim_limit": (
            "complete for one aligned frozen-metric Fourier polarization of the auxiliary "
            "quadratic sector; generic tilt, metric mixing, nonlinear coefficients, and "
            "distributed gravitational constraints are excluded"
        ),
    }
