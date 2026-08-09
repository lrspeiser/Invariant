from __future__ import annotations

from typing import Any

import sympy as sp

from .dirac import analyze_quadratic_lagrangian, poisson_bracket


def generic_horndeski_l2_l4_unitary_adm_control() -> dict[str, Any]:
    """Exact local ADM kinetic theorem for arbitrary smooth Horndeski L2--L4.

    At one unitary-gauge point write ``F=G4(phi,X)`` and ``B=G4_X(phi,X)``.
    ``G2`` is velocity-algebraic and the integrated ``G3``/``G4_phi`` pieces are at
    most linear in the extrinsic curvature, so neither changes the velocity Hessian.
    The only apparent extra velocity is ``V_star=L_n A_star``.  Its mixing with the
    trace of ``K_ij`` cancels between the Gauss--Codazzi boundary term and the
    Horndeski completion for arbitrary local values of F and B.
    """

    f, b = sp.symbols("F B", finite=True, real=True)
    a_star = sp.Symbol("A_star", positive=True, finite=True)
    v_star = sp.Symbol("V_star", real=True)
    k11, k22, k33, k12, k13, k23 = sp.symbols(
        "K11 K22 K33 K12 K13 K23", real=True
    )
    linear_k = sp.Symbol("L_K", real=True)
    beta = sp.Symbol("beta", real=True)
    x = sp.factor(a_star**2 / 2)
    trace_k = k11 + k22 + k33
    kij_squared = k11**2 + k22**2 + k33**2 + 2 * (k12**2 + k13**2 + k23**2)
    dewitt_kinetic = sp.expand(kij_squared - trace_k**2)

    gauss_kinetic = sp.expand(f * dewitt_kinetic)
    curvature_boundary_after_parts = sp.expand(-2 * b * a_star * v_star * trace_k)
    horndeski_completion = sp.expand(
        b * (2 * a_star * v_star * trace_k - a_star**2 * dewitt_kinetic)
    )
    lower_derivative_linear_terms = linear_k * trace_k
    combined = sp.factor(
        gauss_kinetic
        + curvature_boundary_after_parts
        + horndeski_completion
        + lower_derivative_linear_terms
    )
    regularity_factor = sp.factor(f - 2 * x * b)
    expected = sp.factor(regularity_factor * dewitt_kinetic + linear_k * trace_k)
    cancellation_residual = sp.expand(combined - expected)
    vstar_coefficient = sp.factor(sp.diff(combined, v_star))

    metric_velocities = (k11, k22, k33, k12, k13, k23)
    velocities = (v_star, *metric_velocities)
    hessian = sp.hessian(combined, velocities)
    metric_hessian = sp.hessian(combined, metric_velocities)
    nullspace = hessian.nullspace()
    metric_determinant = sp.factor(metric_hessian.det())
    expected_metric_determinant = sp.factor(-1024 * regularity_factor**6)

    deformed = sp.factor(
        gauss_kinetic
        + curvature_boundary_after_parts
        + beta * horndeski_completion
        + lower_derivative_linear_terms
    )
    deformed_hessian = sp.hessian(deformed, velocities)
    deformed_determinant = sp.factor(deformed_hessian.det())
    deformed_regularity = sp.factor(f - beta * a_star**2 * b)
    expected_deformed_determinant = sp.factor(
        -3072
        * deformed_regularity**5
        * b**2
        * a_star**2
        * (beta - 1) ** 2
    )
    deformation_residual = sp.factor(
        deformed_determinant - expected_deformed_determinant
    )
    negative_point = {
        f: sp.Integer(2),
        b: sp.Rational(1, 4),
        a_star: sp.Integer(1),
        beta: sp.Integer(2),
    }
    negative_determinant = sp.factor(deformed_determinant.subs(negative_point))

    passed = (
        cancellation_residual == 0
        and vstar_coefficient == 0
        and hessian.det() == 0
        and hessian.rank() == 6
        and metric_hessian.rank() == 6
        and metric_determinant == expected_metric_determinant
        and len(nullspace) == 1
        and list(nullspace[0]) == [1, 0, 0, 0, 0, 0, 0]
        and deformation_residual == 0
        and negative_determinant != 0
    )
    return {
        "passed": passed,
        "control": "generic Horndeski L2-L4 unitary-gauge ADM kinetic degeneracy",
        "primary_source": "https://arxiv.org/abs/1105.5723",
        "covariant_family": (
            "G2(phi,X)-G3(phi,X) box(phi)+G4(phi,X) R+"
            "G4_X[(box(phi))^2-phi_(mu nu)phi^(mu nu)]"
        ),
        "unitary_gauge_dictionary": {
            "gradient": "nabla_mu(phi)=-A_star n_mu",
            "X": str(x),
            "normal_hessian_velocity": "V_star=L_n A_star",
            "metric_velocities": [str(item) for item in metric_velocities],
        },
        "arbitrary_local_function_jet": {
            "G4": str(f),
            "G4_X": str(b),
            "G2_G3_G4_phi_hessian_contribution": (
                "velocity-algebraic or linear in K_ij; represented by L_K trace(K)"
            ),
        },
        "curvature_boundary_after_integration_by_parts": str(
            curvature_boundary_after_parts
        ),
        "horndeski_second_derivative_adm_kinetic": str(horndeski_completion),
        "combined_adm_kinetic": str(combined),
        "expected_adm_kinetic": str(expected),
        "adm_cancellation_residual": str(cancellation_residual),
        "normal_hessian_velocity_coefficient": str(vstar_coefficient),
        "velocity_order": [str(item) for item in velocities],
        "velocity_hessian": str(hessian),
        "velocity_hessian_rank_on_regular_patch": int(hessian.rank()),
        "velocity_hessian_nullity_on_regular_patch": len(velocities) - int(hessian.rank()),
        "primary_null_vector": [str(item) for item in nullspace[0]],
        "primary_constraint": "p_V_star=0",
        "metric_velocity_hessian_determinant": str(metric_determinant),
        "metric_velocity_hessian_determinant_residual": str(
            sp.factor(metric_determinant - expected_metric_determinant)
        ),
        "regularity_factor": str(regularity_factor),
        "regular_patch": "G4-2 X G4_X != 0",
        "singular_stratum": "G4-2 X G4_X = 0",
        "negative_control": {
            "deformation": "multiply only the Horndeski Hessian completion by beta",
            "deformed_hessian_determinant": str(deformed_determinant),
            "determinant_identity_residual": str(deformation_residual),
            "witness": {"F": "2", "B": "1/4", "A_star": "1", "beta": "2"},
            "witness_determinant": str(negative_determinant),
            "extra_kinetic_direction_restored": negative_determinant != 0,
        },
        "capability_boundary": {
            "hessian_rank": "pass_on_regular_patch",
            "primary_constraint": "pass",
            "secondary_constraint": "unresolved_for_arbitrary_G2_G3_G4",
            "distributed_poisson_closure": "unresolved_for_arbitrary_G2_G3_G4",
            "physical_dof": "not_inferred_from_primary_degeneracy_alone",
            "reduced_hamiltonian_stability": "unresolved",
        },
        "scope": (
            "exact pointwise seven-velocity ADM Hessian theorem for arbitrary smooth Horndeski "
            "L2-L4 on unitary-gauge patches with timelike scalar gradient and "
            "G4-2 X G4_X nonzero. It proves the primary degeneracy but does not substitute for "
            "the action-dependent secondary constraint, distributed Poisson algebra, degree "
            "count, or Hamiltonian boundedness"
        ),
    }


def quartic_horndeski_covariant_adm_control() -> dict[str, Any]:
    """Exact covariant-to-ADM degeneracy control for a named quartic Horndeski action.

    The covariant density is ``G4(X) R + G4_X[(box phi)^2-phi_mn phi^mn]`` with
    ``G4=M2/2+alpha X``.  In unitary gauge ``nabla_mu phi=-A_* n_mu`` the normal Hessian
    velocity is ``V_*=L_n A_*``.  The Gauss-Codazzi boundary integration and Horndeski
    second-derivative term cancel every ``V_* K`` contribution exactly.
    """

    m2 = sp.Symbol("M2", positive=True, finite=True)
    alpha = sp.Symbol("alpha", nonzero=True, finite=True, real=True)
    a_star = sp.Symbol("A_star", positive=True, finite=True)
    v_star, trace_k, shear = sp.symbols("V_star K T", real=True)
    x = sp.factor(a_star**2 / 2)
    g4 = sp.factor(m2 / 2 + alpha * x)
    kij_squared = sp.factor(shear**2 + trace_k**2 / 3)
    gauss_kinetic = sp.factor(g4 * (kij_squared - trace_k**2))
    curvature_boundary_after_parts = sp.factor(-2 * alpha * a_star * v_star * trace_k)
    horndeski_second_derivative = sp.factor(
        alpha
        * (
            2 * a_star * v_star * trace_k
            + a_star**2 * (trace_k**2 - kij_squared)
        )
    )
    combined = sp.factor(
        gauss_kinetic + curvature_boundary_after_parts + horndeski_second_derivative
    )
    expected = sp.factor(
        (g4 - 2 * x * alpha) * (kij_squared - trace_k**2)
    )
    cancellation_residual = sp.simplify(combined - expected)
    vstar_coefficient = sp.factor(sp.diff(combined, v_star))
    velocities = (v_star, trace_k, shear)
    hessian = sp.hessian(combined, velocities)
    nullspace = hessian.nullspace()

    beta = sp.Symbol("beta", real=True)
    deformed = sp.factor(
        gauss_kinetic
        + curvature_boundary_after_parts
        + beta * horndeski_second_derivative
    )
    deformed_hessian = sp.hessian(deformed, velocities)
    deformed_determinant = sp.factor(deformed_hessian.det())
    expected_deformed_determinant = sp.factor(
        -8
        * alpha**2
        * a_star**2
        * (beta - 1) ** 2
        * (g4 - beta * alpha * a_star**2)
    )
    deformation_residual = sp.simplify(
        deformed_determinant - expected_deformed_determinant
    )
    negative_point = {
        m2: sp.Integer(2),
        alpha: sp.Rational(1, 4),
        a_star: sp.Integer(1),
        beta: sp.Integer(2),
    }
    negative_determinant = sp.factor(deformed_determinant.subs(negative_point))

    regular_factor = sp.factor(g4 - 2 * x * alpha)
    passed = (
        cancellation_residual == 0
        and vstar_coefficient == 0
        and hessian.det() == 0
        and hessian.rank() == 2
        and len(nullspace) == 1
        and list(nullspace[0]) == [1, 0, 0]
        and deformation_residual == 0
        and negative_determinant != 0
    )
    return {
        "passed": passed,
        "control": "named covariant quartic-Horndeski ADM degeneracy",
        "primary_source": "https://arxiv.org/abs/1105.5723",
        "covariant_action_density": (
            "sqrt(-g){Lambda_phi^4 X_phi+G4(X) R + "
            "G4_X[(box(phi))^2-phi_(mu nu)phi^(mu nu)]}, G4(X)=M2/2+alpha X"
        ),
        "unitary_gauge_dictionary": {
            "gradient": "nabla_mu phi=-A_star n_mu",
            "X": str(x),
            "normal_hessian_velocity": "V_star=L_n A_star",
            "Kij_squared": str(kij_squared),
        },
        "covariant_curvature_adm_kinetic": str(gauss_kinetic),
        "curvature_boundary_after_integration_by_parts": str(
            curvature_boundary_after_parts
        ),
        "horndeski_second_derivative_adm_kinetic": str(
            horndeski_second_derivative
        ),
        "combined_adm_kinetic": str(combined),
        "expected_adm_kinetic": str(expected),
        "adm_cancellation_residual": str(cancellation_residual),
        "normal_hessian_velocity_coefficient": str(vstar_coefficient),
        "velocity_order": [str(item) for item in velocities],
        "velocity_hessian": str(hessian),
        "velocity_hessian_determinant": str(sp.factor(hessian.det())),
        "velocity_hessian_rank": int(hessian.rank()),
        "primary_null_vector": [str(item) for item in nullspace[0]],
        "regularity_factor": str(regular_factor),
        "negative_control": {
            "deformation": "multiply the Horndeski second-derivative completion by beta",
            "deformed_hessian_determinant": str(deformed_determinant),
            "determinant_identity_residual": str(deformation_residual),
            "witness": {"M2": "2", "alpha": "1/4", "A_star": "1", "beta": "2"},
            "witness_determinant": str(negative_determinant),
            "extra_kinetic_direction_restored": negative_determinant != 0,
        },
        "interpretation": (
            "The named Horndeski coefficient relation removes the scalar normal-Hessian/lapse "
            "velocity before the Legendre transform. Breaking that relation makes the local "
            "three-velocity Hessian nondegenerate and restores the unwanted kinetic direction."
        ),
        "scope": (
            "exact unitary-gauge local kinetic ADM identity for G4=M2/2+alpha X; spatial "
            "curvature, distributed diffeomorphism constraints, covariant Euler variation, "
            "and physical scalar/tensor stability remain separate gates. The positive canonical "
            "scalar term is algebraic in A_star in unitary gauge and does not change this Hessian"
        ),
    }


def quartic_horndeski_unitary_flrw_dirac_control() -> dict[str, Any]:
    """Exact gauge-fixed lapse Dirac chain for the named quartic Horndeski action.

    The scalar is used as the time coordinate, ``phi=t``, on a curved FLRW slice. The
    lapse then carries no velocity, but the Horndeski interaction makes its equation
    nonlinear. Consequently ``p_N`` and lapse preservation form a generic second-class
    pair. This is an exact minisuperspace control, not the distributed field algebra.
    """

    lapse, scale_factor = sp.symbols("N a", positive=True, finite=True)
    dot_lapse, dot_scale = sp.symbols("dot_N dot_a", real=True)
    m2 = sp.Symbol("M2", positive=True, finite=True)
    alpha, spatial_curvature = sp.symbols(
        "alpha k", nonzero=True, finite=True, real=True
    )
    lagrangian = sp.factor(
        -3 * m2 * scale_factor * dot_scale**2 / lapse
        + 3 * alpha * scale_factor * dot_scale**2 / lapse**3
        + 3 * m2 * spatial_curvature * scale_factor * lapse
        + 3 * alpha * spatial_curvature * scale_factor / lapse
        + scale_factor**3 / (2 * lapse)
    )
    result = analyze_quadratic_lagrangian(
        lagrangian,
        (lapse, scale_factor),
        (dot_lapse, dot_scale),
        momentum_prefix="p_h4_",
        max_constraint_generations=8,
    )
    primary = result.primary_constraints[0]
    secondary = result.secondary_constraints[0]
    surface_pairing = sp.factor(result.constraint_matrix[0, 1])
    pair_determinant = sp.factor(result.constraint_matrix.det())
    expected_pair_determinant = sp.factor(surface_pairing**2)
    kinetic_rank_factor = sp.factor(m2 * lapse**2 - alpha)
    pairing_numerator = sp.factor(sp.together(surface_pairing).as_numer_denom()[0])
    canonical_scalar_boundary_pairing = sp.factor(surface_pairing.subs(alpha, 0))
    passed = (
        result.velocity_hessian.rank() == 1
        and result.velocity_hessian.det() == 0
        and primary == result.momenta[0]
        and len(result.secondary_constraints) == 1
        and not result.higher_generation_constraints
        and sp.factor(pair_determinant - expected_pair_determinant) == 0
        and surface_pairing != 0
        and result.constraint_matrix_rank == 2
        and result.first_class_constraints == 0
        and result.second_class_constraints == 2
        and result.physical_dof == 1
        and result.closure
        and canonical_scalar_boundary_pairing != 0
    )
    return {
        "passed": passed,
        "control": "named quartic-Horndeski curved-FLRW unitary-gauge lapse Dirac chain",
        "covariant_parent": (
            "sqrt(-g){(M2/2)R+Lambda_phi^4 X_phi+"
            "alpha[X_c R+(box phi)^2-phi_(mu nu)phi^(mu nu)]}"
        ),
        "gauge_and_background": "phi=t, FLRW with N(t), a(t), and nonzero spatial curvature k",
        "reduced_lagrangian": str(lagrangian),
        "velocity_order": [str(item) for item in result.velocities],
        "velocity_hessian": str(result.velocity_hessian),
        "velocity_hessian_rank": int(result.velocity_hessian.rank()),
        "primary_constraint": str(primary),
        "canonical_hamiltonian": str(result.canonical_hamiltonian),
        "secondary_constraint": str(secondary),
        "constraint_generations": [
            [str(item) for item in generation]
            for generation in result.constraint_generations
        ],
        "constraint_surface_poisson_matrix": str(result.constraint_matrix),
        "primary_secondary_pairing_on_surface": str(surface_pairing),
        "constraint_matrix_determinant": str(pair_determinant),
        "constraint_matrix_rank": result.constraint_matrix_rank,
        "first_class_constraints": result.first_class_constraints,
        "second_class_constraints": result.second_class_constraints,
        "minisuperspace_physical_dof": result.physical_dof,
        "closure": result.closure,
        "multiplier_solution": {
            str(key): str(value) for key, value in result.multiplier_solution.items()
        },
        "regular_patch": [
            "N > 0",
            "a > 0",
            "M2 > 0",
            "alpha != 0",
            "k != 0",
            "p_h4_1 != 0",
            f"{kinetic_rank_factor} != 0",
            f"{pairing_numerator} != 0",
        ],
        "canonical_scalar_boundary_control": {
            "substitution": "alpha=0",
            "surface_pairing": str(canonical_scalar_boundary_pairing),
            "interpretation": (
                "the unitary-clock lapse pair remains second class because the positive canonical "
                "scalar kinetic term remains; setting alpha=0 recovers canonical scalar gravity"
            ),
        },
        "conditional_full_field_count": {
            "configuration_variables_after_unitary_gauge": 10,
            "spatial_diffeomorphism_first_class_constraints": 6,
            "lapse_second_class_constraints": 2,
            "physical_dof_if_distributed_spatial_chain_closes": 3,
            "status": "not_yet_a_distributed_closure_proof",
        },
        "interpretation": (
            "The cancelled lapse velocity gives p_N=0. On a generic curved-FLRW unitary-gauge "
            "patch its preservation produces one secondary constraint with a nonzero Poisson "
            "pairing, leaving one minisuperspace degree of freedom."
        ),
        "scope": (
            "exact action-specific curved-FLRW unitary-gauge Dirac chain and quotient-surface "
            "constraint rank; inhomogeneous momentum constraints, distributed H-H closure, "
            "three physical field modes, and Hamiltonian boundedness remain unproved"
        ),
    }


def dhost_reduced_dirac_control() -> dict[str, Any]:
    """Exact Dirac chain for the scalar kinetic block used in quadratic DHOST analyses.

    ``A`` represents the scalar normal derivative, ``Q`` a scalar metric/extrinsic-curvature
    direction, and their velocities represent ``V_*`` and ``K``.  This is a finite-point ADM
    scalar-sector control: it verifies the degeneracy-to-primary-to-secondary mechanism and the
    local degree reduction.  It is not a complete covariant DHOST classification or a proof of
    the distributed gravitational constraint algebra.
    """

    a_star, metric_scalar = sp.symbols("A_star Q_metric", real=True)
    v_star, trace_k = sp.symbols("V_star K", real=True)
    alpha = sp.symbols("alpha", nonzero=True, real=True)
    coefficient, mass, frequency = sp.symbols(
        "C m omega", positive=True, finite=True
    )
    kinetic = coefficient * (v_star + alpha * trace_k) ** 2 / 2
    potential = (mass**2 * a_star**2 + frequency**2 * metric_scalar**2) / 2
    lagrangian = kinetic - potential

    result = analyze_quadratic_lagrangian(
        lagrangian,
        (a_star, metric_scalar),
        (v_star, trace_k),
        momentum_prefix="p_dhost_",
        max_constraint_generations=8,
    )
    p_star, p_metric = result.momenta
    expected_primary = sp.factor(-alpha * p_star + p_metric)
    expected_secondary = sp.factor(
        alpha * mass**2 * a_star - frequency**2 * metric_scalar
    )
    second_class_pairing = sp.factor(
        poisson_bracket(
            expected_primary,
            expected_secondary,
            result.coordinates,
            result.momenta,
        )
    )
    regularity_factor = sp.factor(alpha**2 * mass**2 + frequency**2)

    solved_metric = sp.solve(sp.Eq(expected_secondary, 0), metric_scalar)
    reduced_hamiltonian = sp.factor(
        result.canonical_hamiltonian.subs(metric_scalar, solved_metric[0])
    )
    expected_reduced_hamiltonian = sp.factor(
        p_star**2 / (2 * coefficient)
        + mass**2 * a_star**2 / 2
        + alpha**2 * mass**4 * a_star**2 / (2 * frequency**2)
    )
    positivity_coefficients = {
        "p_star_squared": sp.factor(
            sp.diff(reduced_hamiltonian, p_star, 2) / 2
        ),
        "A_star_squared": sp.factor(
            sp.diff(reduced_hamiltonian, a_star, 2) / 2
        ),
    }

    epsilon = sp.symbols("epsilon", positive=True, finite=True)
    nondegenerate_lagrangian = (
        kinetic + epsilon * trace_k**2 / 2 - potential
    )
    nondegenerate = analyze_quadratic_lagrangian(
        nondegenerate_lagrangian,
        (a_star, metric_scalar),
        (v_star, trace_k),
        momentum_prefix="p_nondegenerate_",
    )
    nondegenerate_hessian = nondegenerate.velocity_hessian

    passed = (
        result.velocity_hessian.det() == 0
        and result.velocity_hessian.rank() == 1
        and result.primary_constraints == (expected_primary,)
        and result.secondary_constraints == (expected_secondary,)
        and result.constraint_matrix_rank == 2
        and result.first_class_constraints == 0
        and result.second_class_constraints == 2
        and result.physical_dof == 1
        and result.closure
        and sp.factor(second_class_pairing - regularity_factor) == 0
        and sp.factor(reduced_hamiltonian - expected_reduced_hamiltonian) == 0
        and all(item.is_positive is True for item in positivity_coefficients.values())
        and sp.factor(nondegenerate_hessian.det() - coefficient * epsilon) == 0
        and nondegenerate.primary_constraints == ()
        and nondegenerate.physical_dof == 2
    )
    return {
        "passed": passed,
        "control": "reduced quadratic-DHOST ADM scalar Dirac chain",
        "primary_source": "https://arxiv.org/abs/1512.06820",
        "kinetic_form": str(kinetic),
        "potential_control": str(potential),
        "hessian": str(result.velocity_hessian),
        "hessian_determinant": str(sp.factor(result.velocity_hessian.det())),
        "hessian_rank": int(result.velocity_hessian.rank()),
        "determinant": str(sp.factor(result.velocity_hessian.det())),
        "rank": int(result.velocity_hessian.rank()),
        "hessian_null_vector": [str(item) for item in result.velocity_hessian.nullspace()[0]],
        "null_vector": [str(item) for item in result.velocity_hessian.nullspace()[0]],
        "canonical_momenta": [str(item) for item in result.canonical_momenta],
        "constraint_generations": [
            [str(item) for item in generation]
            for generation in result.constraint_generations
        ],
        "primary_constraint": str(expected_primary),
        "secondary_constraint": str(expected_secondary),
        "primary_secondary_bracket": str(second_class_pairing),
        "regularity_factor": str(regularity_factor),
        "regular_domain": ["C > 0", "m > 0", "omega > 0", "alpha != 0"],
        "constraint_matrix": str(result.constraint_matrix),
        "constraint_matrix_rank": result.constraint_matrix_rank,
        "first_class_constraints": result.first_class_constraints,
        "second_class_constraints": result.second_class_constraints,
        "physical_scalar_dof": result.physical_dof,
        "closure": result.closure,
        "multiplier_solution": {
            str(key): str(value) for key, value in result.multiplier_solution.items()
        },
        "reduced_hamiltonian": str(reduced_hamiltonian),
        "reduced_hamiltonian_positive_coefficients": {
            name: str(value) for name, value in positivity_coefficients.items()
        },
        "nondegenerate_negative_control": {
            "kinetic_deformation": str(epsilon * trace_k**2 / 2),
            "hessian_determinant": str(sp.factor(nondegenerate_hessian.det())),
            "hessian_rank": int(nondegenerate_hessian.rank()),
            "primary_constraints": [
                str(item) for item in nondegenerate.primary_constraints
            ],
            "physical_scalar_dof": nondegenerate.physical_dof,
            "rejected_as_degenerate": (
                nondegenerate_hessian.det() != 0
                and nondegenerate.physical_dof == 2
            ),
            "interpretation": (
                "Breaking the kinetic degeneracy restores the extra local scalar mode.  This "
                "finite-point block does not by itself prove the full field theory is "
                "Ostrogradsky-unstable."
            ),
        },
        "nondegenerate_negative_control_determinant": str(
            sp.factor(nondegenerate_hessian.det())
        ),
        "singular_stratum": (
            "alpha^2 m^2 + omega^2 = 0 makes the primary-secondary pairing singular; "
            "it lies outside the declared positive control domain"
        ),
        "interpretation": (
            "Kinetic degeneracy produces one primary constraint.  Its preservation produces a "
            "secondary constraint; the pair is second class on the regular domain and removes "
            "the extra scalar mode, leaving one positive reduced scalar Hamiltonian direction."
        ),
        "scope": (
            "exact finite-point ADM scalar-sector Dirac mechanism corresponding to the DHOST "
            "kinetic block; tensor/momentum constraints, spatial derivatives, arbitrary DHOST "
            "functions, and the full covariant classification remain separate"
        ),
    }
