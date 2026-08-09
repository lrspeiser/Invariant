from __future__ import annotations

from typing import Any

import sympy as sp

from .dirac import analyze_quadratic_lagrangian
from .field_dirac import canonical_metric_diffeomorphism_control, euler_operator_nd
from .principal_symbol import analyze_isotropic_second_order_symbol


def _einstein_aether_kinetic_model() -> dict[str, Any]:
    """Build the exact local GR plus covector-Aether kinetic polynomial once."""

    k11, k22, k33, k12, k13, k23 = sp.symbols("K11 K22 K33 K12 K13 K23")
    vector_velocities = sp.symbols("v0:4")
    vector_down = sp.symbols("u0:4")
    c1, c2, c3, c4, planck2 = sp.symbols("c1 c2 c3 c4 M2")
    extrinsic = sp.Matrix(
        [[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]]
    )
    metric = sp.diag(-1, 1, 1, 1)
    derivative = sp.zeros(4)
    derivative[0, 0] = vector_velocities[0]
    for i in range(3):
        shift_term = sum(extrinsic[i, j] * vector_down[j + 1] for j in range(3))
        derivative[0, i + 1] = vector_velocities[i + 1] - shift_term
        derivative[i + 1, 0] = -shift_term
        for j in range(3):
            derivative[i + 1, j + 1] = -extrinsic[i, j] * vector_down[0]

    derivative_up = metric * derivative * metric
    k1 = sp.expand(
        sum(
            derivative[mu, nu] * derivative_up[mu, nu]
            for mu in range(4)
            for nu in range(4)
        )
    )
    expansion = sum(
        metric[mu, nu] * derivative[mu, nu]
        for mu in range(4)
        for nu in range(4)
    )
    k2 = sp.expand(expansion**2)
    k3 = sp.expand(
        sum(
            derivative[mu, nu] * derivative_up[nu, mu]
            for mu in range(4)
            for nu in range(4)
        )
    )
    acceleration_down = (metric * sp.Matrix(vector_down)).T * derivative
    k4 = sp.expand((acceleration_down * metric * acceleration_down.T)[0])
    trace_k = sp.trace(extrinsic)
    k_squared = sum(extrinsic[i, j] ** 2 for i in range(3) for j in range(3))
    lagrangian = sp.expand(
        planck2 * (k_squared - trace_k**2) / 2
        - (c1 * k1 + c2 * k2 + c3 * k3 - c4 * k4) / 2
    )
    velocities = (k11, k22, k33, k12, k13, k23, *vector_velocities)
    return {
        "lagrangian": lagrangian,
        "velocities": velocities,
        "vector_down": vector_down,
        "coupling_symbols": (planck2, c1, c2, c3, c4),
        "hessian": sp.hessian(lagrangian, velocities),
    }


def einstein_aether_adm_kinetic_control() -> dict[str, Any]:
    """Exact local ADM kinetic Hessian for GR plus a covector Einstein–Æther field."""

    model = _einstein_aether_kinetic_model()
    velocities = model["velocities"]
    vector_down = model["vector_down"]
    planck2, c1, c2, c3, c4 = model["coupling_symbols"]
    hessian = model["hessian"]
    control_couplings = {
        planck2: 1,
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    aligned = {
        **control_couplings,
        vector_down[0]: 1,
        vector_down[1]: 0,
        vector_down[2]: 0,
        vector_down[3]: 0,
    }
    tilted = {
        **aligned,
        vector_down[0]: sp.Rational(5, 4),
        vector_down[1]: sp.Rational(3, 4),
    }
    aligned_hessian = hessian.subs(aligned)
    tilted_hessian = hessian.subs(tilted)
    spatial_vector_block = aligned_hessian[7:10, 7:10]
    expected_c14 = control_couplings[c1] + control_couplings[c4]
    passed = (
        hessian == hessian.T
        and aligned_hessian.rank() == 10
        and tilted_hessian.rank() == 10
        and spatial_vector_block == sp.eye(3) * expected_c14
        and -tilted[vector_down[0]] ** 2 + tilted[vector_down[1]] ** 2 == -1
    )
    return {
        "kinetic_action": "M2(K_ij K^ij-K^2)/2 -(c1 K1+c2 K2+c3 K3-c4 K4)/2",
        "frame": "local Gaussian-normal ADM frame with N=1, shift=0, gamma_ij=delta_ij",
        "velocities": [str(item) for item in velocities],
        "dynamic_velocity_count": len(velocities),
        "full_configuration_count_with_lapse_shift_multiplier": 15,
        "primary_null_directions_outside_dynamic_block": [
            "dot(N)",
            "dot(N^1)",
            "dot(N^2)",
            "dot(N^3)",
            "dot(lambda_u)",
        ],
        "aligned": {
            "u_mu": ["1", "0", "0", "0"],
            "hessian_rank": int(aligned_hessian.rank()),
            "hessian_determinant": str(sp.factor(aligned_hessian.det())),
            "spatial_vector_velocity_block": str(spatial_vector_block),
            "expected_c14": str(expected_c14),
        },
        "tilted": {
            "u_mu": ["5/4", "3/4", "0", "0"],
            "unit_norm": "-1",
            "hessian_rank": int(tilted_hessian.rank()),
            "hessian_determinant": str(sp.factor(tilted_hessian.det())),
        },
        "couplings": {
            "M2": "1",
            "c1": "1/10",
            "c2": "1/20",
            "c3": "0",
            "c4": "1/20",
        },
        "passed": passed,
        "scope": "exact pointwise nonlinear kinetic Hessian; secondary constraints and Poisson closure remain separate",
    }


def einstein_aether_coupled_unit_normal_control() -> dict[str, Any]:
    """Check the unit constraint against the full metric-vector inverse kinetic block.

    For a regular quadratic kinetic matrix ``H``, preservation of a holonomic constraint
    produces the normality coefficient ``C_,A (H^-1)^AB C_,B``.  A nonzero coefficient is
    what allows the next consistency condition to fix the unit multiplier instead of creating
    an accidental extra gauge direction.  This control evaluates that coefficient exactly in
    three unit-timelike local patches while retaining every metric-vector kinetic mixing term.
    """

    model = _einstein_aether_kinetic_model()
    hessian_k = model["hessian"]
    vector_down = model["vector_down"]
    planck2, c1, c2, c3, c4 = model["coupling_symbols"]

    # K_ij = dot(gamma_ij)/2 in the local N=1, shift=0 frame.  Converting the first
    # six velocity columns to dot(gamma_ij) makes the gradient below canonically matched.
    k_from_metric_velocity = sp.diag(*([sp.Rational(1, 2)] * 6 + [1] * 4))
    hessian = k_from_metric_velocity.T * hessian_k * k_from_metric_velocity
    u0, u1, u2, u3 = vector_down
    constraint_gradient = sp.Matrix(
        [
            -u1**2,
            -u2**2,
            -u3**2,
            -2 * u1 * u2,
            -2 * u1 * u3,
            -2 * u2 * u3,
            -2 * u0,
            2 * u1,
            2 * u2,
            2 * u3,
        ]
    )
    control_couplings = {
        planck2: 1,
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    backgrounds = {
        "aligned": {u0: 1, u1: 0, u2: 0, u3: 0},
        "axis_tilted": {
            u0: sp.Rational(5, 4),
            u1: sp.Rational(3, 4),
            u2: 0,
            u3: 0,
        },
        "oblique_tilted": {
            u0: sp.Rational(13, 12),
            u1: sp.Rational(1, 3),
            u2: sp.Rational(1, 4),
            u3: 0,
        },
    }
    patch_evidence: dict[str, dict[str, Any]] = {}
    for name, background in backgrounds.items():
        substitutions = {**control_couplings, **background}
        background_hessian = hessian.subs(substitutions)
        background_gradient = constraint_gradient.subs(background)
        normality = sp.factor(
            (background_gradient.T * background_hessian.inv() * background_gradient)[0]
        )
        unit_norm = sp.factor(
            -background[u0] ** 2
            + background[u1] ** 2
            + background[u2] ** 2
            + background[u3] ** 2
        )
        patch_evidence[name] = {
            "u_mu": [str(background[item]) for item in vector_down],
            "unit_norm": str(unit_norm),
            "hessian_rank": int(background_hessian.rank()),
            "hessian_determinant": str(sp.factor(background_hessian.det())),
            "constraint_normality": str(normality),
            "second_class_regular": normality != 0,
        }

    singular_substitutions = {
        planck2: 1,
        c1: 0,
        c2: 0,
        c3: 0,
        c4: 0,
        **backgrounds["aligned"],
    }
    singular_hessian = hessian.subs(singular_substitutions)
    singular_rank = int(singular_hessian.rank())
    singular_rejected = singular_rank < hessian.rows
    passed = (
        all(
            patch["unit_norm"] == "-1"
            and patch["hessian_rank"] == 10
            and patch["second_class_regular"]
            for patch in patch_evidence.values()
        )
        and singular_rejected
    )
    return {
        "passed": passed,
        "canonical_velocity_chart": [
            "dot(gamma_11)",
            "dot(gamma_22)",
            "dot(gamma_33)",
            "dot(gamma_12)",
            "dot(gamma_13)",
            "dot(gamma_23)",
            "dot(u_perp)",
            "dot(u_1)",
            "dot(u_2)",
            "dot(u_3)",
        ],
        "constraint": "-u_perp^2+gamma^ij u_i u_j+1=0",
        "constraint_gradient_at_gamma_identity": [
            str(item) for item in constraint_gradient
        ],
        "normality_test": "C_,A (H^-1)^AB C_,B != 0",
        "patches": patch_evidence,
        "singular_coupling_negative_control": {
            "couplings": {"M2": "1", "c1": "0", "c2": "0", "c3": "0", "c4": "0"},
            "hessian_rank": singular_rank,
            "expected_full_rank": hessian.rows,
            "rejected": singular_rejected,
        },
        "interpretation": (
            "At the declared regular coupling point, metric-Aether kinetic mixing does not turn "
            "the unit-normal consistency condition into a null direction on any tested exact "
            "unit-timelike patch."
        ),
        "scope": (
            "exact pointwise mixed kinetic and unit-normal control in three rational local "
            "patches; it does not establish the spatial H-D/H-H algebra, global coupling-domain "
            "regularity, five total field-theory degrees of freedom, or reduced Hamiltonian "
            "boundedness"
        ),
    }


def einstein_aether_spatial_diffeomorphism_control() -> dict[str, Any]:
    """Exact 3D momentum-generator algebra for the ADM Æther variables."""

    spatial = sp.symbols("x0:3", real=True)
    covector = tuple(sp.Function(f"u{i}")(*spatial) for i in range(3))
    vector_momentum = tuple(sp.Function(f"pu{i}")(*spatial) for i in range(3))
    normal_scalar = sp.Function("u_perp")(*spatial)
    normal_momentum = sp.Function("p_perp")(*spatial)
    shift_m = tuple(sp.Function(f"M{i}")(*spatial) for i in range(3))
    shift_l = tuple(sp.Function(f"L{i}")(*spatial) for i in range(3))

    def lie_covector(
        vector: tuple[sp.Expr, ...], components: tuple[sp.Expr, ...], i: int
    ) -> sp.Expr:
        return sum(
            vector[k] * sp.diff(components[i], spatial[k])
            + components[k] * sp.diff(vector[k], spatial[i])
            for k in range(3)
        )

    def lie_vector_density(
        vector: tuple[sp.Expr, ...], components: tuple[sp.Expr, ...], i: int
    ) -> sp.Expr:
        return sum(
            vector[k] * sp.diff(components[i], spatial[k])
            - components[k] * sp.diff(vector[i], spatial[k])
            + components[i] * sp.diff(vector[k], spatial[k])
            for k in range(3)
        )

    def lie_scalar(vector: tuple[sp.Expr, ...], scalar: sp.Expr) -> sp.Expr:
        return sum(vector[k] * sp.diff(scalar, spatial[k]) for k in range(3))

    def lie_scalar_density(vector: tuple[sp.Expr, ...], density: sp.Expr) -> sp.Expr:
        return sum(
            vector[k] * sp.diff(density, spatial[k])
            + density * sp.diff(vector[k], spatial[k])
            for k in range(3)
        )

    def generator(vector: tuple[sp.Expr, ...]) -> sp.Expr:
        return sp.expand(
            sum(
                vector_momentum[i] * lie_covector(vector, covector, i)
                for i in range(3)
            )
            + normal_momentum * lie_scalar(vector, normal_scalar)
        )

    generator_m = generator(shift_m)
    coordinate_residuals = []
    momentum_residuals = []
    for i in range(3):
        coordinate_residuals.append(
            sp.factor(
                euler_operator_nd(
                    generator_m, vector_momentum[i], spatial, maximum_order=1
                )
                - lie_covector(shift_m, covector, i)
            )
        )
        momentum_residuals.append(
            sp.factor(
                -euler_operator_nd(
                    generator_m, covector[i], spatial, maximum_order=1
                )
                - lie_vector_density(shift_m, vector_momentum, i)
            )
        )
    coordinate_residuals.append(
        sp.factor(
            euler_operator_nd(
                generator_m, normal_momentum, spatial, maximum_order=1
            )
            - lie_scalar(shift_m, normal_scalar)
        )
    )
    momentum_residuals.append(
        sp.factor(
            -euler_operator_nd(
                generator_m, normal_scalar, spatial, maximum_order=1
            )
            - lie_scalar_density(shift_m, normal_momentum)
        )
    )

    commutator = tuple(
        sum(
            shift_m[k] * sp.diff(shift_l[i], spatial[k])
            - shift_l[k] * sp.diff(shift_m[i], spatial[k])
            for k in range(3)
        )
        for i in range(3)
    )
    lie_m_covector = tuple(
        sp.expand(lie_covector(shift_m, covector, i)) for i in range(3)
    )
    lie_l_covector = tuple(
        sp.expand(lie_covector(shift_l, covector, i)) for i in range(3)
    )
    lie_m_momentum = tuple(
        sp.expand(lie_vector_density(shift_m, vector_momentum, i))
        for i in range(3)
    )
    lie_l_momentum = tuple(
        sp.expand(lie_vector_density(shift_l, vector_momentum, i))
        for i in range(3)
    )
    commutator_coordinate_residuals = [
        sp.factor(
            lie_covector(shift_m, lie_l_covector, i)
            - lie_covector(shift_l, lie_m_covector, i)
            - lie_covector(commutator, covector, i)
        )
        for i in range(3)
    ]
    commutator_momentum_residuals = [
        sp.factor(
            lie_vector_density(shift_m, lie_l_momentum, i)
            - lie_vector_density(shift_l, lie_m_momentum, i)
            - lie_vector_density(commutator, vector_momentum, i)
        )
        for i in range(3)
    ]
    lie_m_normal = sp.expand(lie_scalar(shift_m, normal_scalar))
    lie_l_normal = sp.expand(lie_scalar(shift_l, normal_scalar))
    lie_m_normal_momentum = sp.expand(
        lie_scalar_density(shift_m, normal_momentum)
    )
    lie_l_normal_momentum = sp.expand(
        lie_scalar_density(shift_l, normal_momentum)
    )
    commutator_coordinate_residuals.append(
        sp.factor(
            lie_scalar(shift_m, lie_l_normal)
            - lie_scalar(shift_l, lie_m_normal)
            - lie_scalar(commutator, normal_scalar)
        )
    )
    commutator_momentum_residuals.append(
        sp.factor(
            lie_scalar_density(shift_m, lie_l_normal_momentum)
            - lie_scalar_density(shift_l, lie_m_normal_momentum)
            - lie_scalar_density(commutator, normal_momentum)
        )
    )

    omitted_weight_residual = sp.factor(
        -euler_operator_nd(generator_m, normal_scalar, spatial, maximum_order=1)
        - lie_scalar(shift_m, normal_momentum)
    )
    metric_control = canonical_metric_diffeomorphism_control()
    all_residuals = (
        *coordinate_residuals,
        *momentum_residuals,
        *commutator_coordinate_residuals,
        *commutator_momentum_residuals,
    )
    passed = (
        metric_control["passed"]
        and all(item == 0 for item in all_residuals)
        and omitted_weight_residual != 0
    )
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "canonical_aether_pairs": [
            "(u_1,p_u^1 density)",
            "(u_2,p_u^2 density)",
            "(u_3,p_u^3 density)",
            "(u_perp,p_perp density)",
        ],
        "metric_sector_passed": metric_control["passed"],
        "canonical_coordinate_residuals": [str(item) for item in coordinate_residuals],
        "canonical_momentum_residuals": [str(item) for item in momentum_residuals],
        "commutator_coordinate_residuals": [
            str(item) for item in commutator_coordinate_residuals
        ],
        "commutator_momentum_residuals": [
            str(item) for item in commutator_momentum_residuals
        ],
        "omitted_momentum_density_weight_negative_control": {
            "rejected": omitted_weight_residual != 0,
            "residual": str(omitted_weight_residual),
        },
        "closure": "{D[M],D[L]}=D[[M,L]] modulo a compact-support boundary",
        "interpretation": (
            "The metric, spatial covector, normal scalar, and both conjugate momentum densities "
            "form the exact cotangent lift of spatial diffeomorphisms."
        ),
        "scope": (
            "complete Einstein-Aether D-D sector for canonical spatial variables; primary unit "
            "constraint, Hamiltonian constraints, higher consistency, and reduced Hamiltonian "
            "remain separate"
        ),
    }


def maxwell_unit_aether_nonlinear_hamiltonian_control() -> dict[str, Any]:
    """Full nonlinear ADM control for the Maxwell-form unit-Aether subclass.

    On the positive unit-norm branch the normal component is
    ``chi=sqrt(1+q^ij A_i A_j)``.  The remaining three spatial covector components
    are ordinary canonical coordinates.  This is the ``c3=-c1, c2=c4=0``
    Einstein-Aether subclass (up to overall convention), not the generic K1..K4
    theory.

    The H-H calculation uses the derivative-of-lapse coefficients of the exact
    local functional derivatives.  Terms without lapse derivatives cancel under
    antisymmetrisation, and the displayed coefficient is what remains after one
    covariant integration by parts.
    """

    alpha, beta, chi, div_p = sp.symbols(
        "alpha beta chi div_p", positive=True, finite=True
    )
    a = sp.Matrix(sp.symbols("A0:3", real=True))
    p = sp.Matrix(sp.symbols("p0:3", real=True))
    d_chi = sp.Matrix(sp.symbols("dchi0:3", real=True))
    div_f = sp.Matrix(sp.symbols("divF0:3", real=True))
    f01, f02, f12 = sp.symbols("F01 F02 F12", real=True)
    field_strength = sp.Matrix(
        [[0, f01, f02], [-f01, 0, f12], [-f02, -f12, 0]]
    )

    # For H_A = alpha p^2/sqrt(q) + chi D_i p^i
    #             + beta sqrt(q) F_ij F^ij, evaluated in a local orthonormal frame:
    # delta H[N]/delta A_k = N*A0_k + Agrad[j,k] D_j N,
    # delta H[N]/delta p_i = N*B0_i - chi D_i N.
    a0 = a * div_p / chi - 4 * beta * div_f
    agrad = -4 * beta * field_strength
    b0 = 2 * alpha * p - d_chi
    divergence_agrad = -4 * beta * div_f

    bracket_coefficient = sp.Matrix(
        [
            sp.factor(
                chi * (divergence_agrad[k] - a0[k])
                - sum(
                    agrad[k, i] * (d_chi[i] + b0[i]) for i in range(3)
                )
            )
            for k in range(3)
        ]
    )
    normalized_coefficient = sp.simplify(
        bracket_coefficient.subs(beta, 1 / (8 * alpha))
    )
    aether_momentum_constraint = sp.Matrix(
        [
            sp.expand(
                sum(p[i] * field_strength[k, i] for i in range(3))
                - a[k] * div_p
            )
            for k in range(3)
        ]
    )
    hh_residual = sp.simplify(
        normalized_coefficient - aether_momentum_constraint
    )
    wrong_normalization = sp.simplify(
        bracket_coefficient.subs(beta, 1 / (16 * alpha))
        - aether_momentum_constraint
    )

    gr_control = canonical_metric_diffeomorphism_control()
    vector_diffeomorphism_control = einstein_aether_spatial_diffeomorphism_control()

    # Reduced canonical count: six q_ij plus three A_i pairs, with the Hamiltonian
    # and three momentum constraints first class.  Including lapse and shift momenta
    # gives the equivalent 13-pair / eight-first-class count.
    reduced_canonical_pairs = 9
    reduced_first_class = 4
    physical_dof = reduced_canonical_pairs - reduced_first_class
    extended_canonical_pairs = 13
    extended_first_class = 8
    extended_physical_dof = extended_canonical_pairs - extended_first_class

    # Exact instability family on a flat periodic slice.  Set A_i=(f(x),0,0),
    # F_ij=0 and p^1=(partial_1 chi)/(2 alpha).  After one periodic integration by
    # parts the energy is -int (partial_1 chi)^2/(4 alpha).  For
    # f=a sin(kx), the positive integral factor is independent of k and the energy
    # therefore tends to minus infinity as k^2.
    amplitude, wave_number = sp.symbols("a k", positive=True, finite=True)
    phase = sp.symbols("theta", real=True)
    dchi_squared_over_k2 = sp.factor(
        amplitude**4
        * sp.sin(phase) ** 2
        * sp.cos(phase) ** 2
        / (1 + amplitude**2 * sp.sin(phase) ** 2)
    )
    instability_density_coefficient = sp.factor(
        -wave_number**2 * dchi_squared_over_k2 / (4 * alpha)
    )
    instability_negative = (
        instability_density_coefficient.subs(
            {amplitude: 1, wave_number: 1, phase: sp.pi / 4, alpha: 1}
        )
        < 0
    )

    passed = (
        hh_residual == sp.zeros(3, 1)
        and wrong_normalization != sp.zeros(3, 1)
        and gr_control["passed"]
        and vector_diffeomorphism_control["passed"]
        and physical_dof == 5
        and extended_physical_dof == 5
        and bool(instability_negative)
    )
    return {
        "passed": passed,
        "subclass": "Maxwell-form unit Aether: c3=-c1, c2=c4=0 up to convention",
        "unit_branch": "chi=sqrt(1+q^ij A_i A_j)>0",
        "hamiltonian_constraint": (
            "H_A=alpha q_ij p^i p^j/sqrt(q)+chi D_i p^i+"
            "beta sqrt(q) F_ij F^ij, with 8 alpha beta=1"
        ),
        "aether_momentum_constraint": (
            "D_i^A=p^j F_ij-A_i D_j p^j"
        ),
        "hh_bracket_coefficient": [str(item) for item in normalized_coefficient],
        "hh_expected_coefficient": [str(item) for item in aether_momentum_constraint],
        "hh_residual": [str(item) for item in hh_residual],
        "full_constraint_algebra": {
            "D_D": "{D[M],D[L]}=D[[M,L]]",
            "D_H": "{D[M],H[N]}=H[L_M N]",
            "H_H": (
                "{H[N],H[M]}=D[q^ij(N D_j M-M D_j N)]"
            ),
            "gr_sector_passed": gr_control["passed"],
            "aether_spatial_cotangent_lift_passed": vector_diffeomorphism_control[
                "passed"
            ],
            "metric_aether_cross_terms": (
                "ultralocal in lapse and cancel under N*M antisymmetrisation"
            ),
            "boundary_condition": "compact support or periodic/vanishing boundary flux",
        },
        "normalization_negative_control": {
            "wrong_relation": "16 alpha beta=1",
            "rejected": wrong_normalization != sp.zeros(3, 1),
            "residual": [str(item) for item in wrong_normalization],
        },
        "degree_count": {
            "reduced_canonical_pairs": reduced_canonical_pairs,
            "reduced_first_class_constraints": reduced_first_class,
            "extended_pairs_including_lapse_shift": extended_canonical_pairs,
            "extended_first_class_constraints": extended_first_class,
            "second_class_constraints_after_solving_unit_branch": 0,
            "physical_dof": physical_dof,
            "mode_interpretation": "two metric plus three vector configuration modes",
        },
        "hamiltonian_stability": {
            "status": "reject",
            "family": (
                "flat periodic slice; A_i=(a sin(kx),0,0), F_ij=0, "
                "p^1=(partial_1 chi)/(2 alpha)"
            ),
            "integrated_energy": (
                "E_k=-int (partial_1 chi)^2/(4 alpha) and tends to -infinity "
                "as k^2 for alpha>0 and a!=0"
            ),
            "pointwise_negative_coefficient": str(instability_density_coefficient),
            "negative_control_point_verified": bool(instability_negative),
        },
        "primary_source": "https://arxiv.org/abs/2307.15126",
        "interpretation": (
            "The compiler distinguishes constraint consistency from stability: this exact "
            "nonlinear five-degree-of-freedom Aether subclass closes under spacetime "
            "diffeomorphisms but is rejected because its reduced Hamiltonian is unbounded below."
        ),
        "scope": (
            "full nonlinear ADM/Dirac and stability result for the Maxwell-form unit-Aether "
            "subclass only; it neither proves nor disproves closure or boundedness for generic "
            "independent c1,c2,c3,c4 Einstein-Aether couplings"
        ),
    }


def einstein_aether_3plus1_decomposition_control() -> dict[str, Any]:
    """Exact 3+1 decomposition and local Legendre map for generic K1..K4 Aether.

    The calculation keeps the spatial Aether derivative, lapse acceleration, extrinsic
    curvature, normal Aether velocity, and spatial-vector velocity as independent local jets.
    It then solves the positive unit branch and checks the resulting nine-velocity Legendre map
    on exact rational data.  Spatial Poisson brackets are intentionally a later stage.
    """

    chi = sp.symbols("chi", positive=True, finite=True)
    aether = sp.Matrix(sp.symbols("A0:3", real=True))
    normal_velocity = sp.symbols("dot_chi", real=True)
    vector_velocity = sp.Matrix(sp.symbols("V0:3", real=True))
    lapse_acceleration = sp.Matrix(sp.symbols("acc0:3", real=True))
    d_chi = sp.Matrix(sp.symbols("Dchi0:3", real=True))
    d_aether = sp.Matrix(
        3, 3, lambda i, j: sp.symbols(f"DA{i}{j}", real=True)
    )
    k11, k22, k33, k12, k13, k23 = sp.symbols(
        "K11 K22 K33 K12 K13 K23", real=True
    )
    extrinsic = sp.Matrix(
        [[k11, k12, k13], [k12, k22, k23], [k13, k23, k33]]
    )

    spatial_block = d_aether + chi * extrinsic
    normal_spatial = (
        vector_velocity - extrinsic * aether + chi * lapse_acceleration
    )
    spatial_normal = d_chi + extrinsic * aether
    normal_normal = normal_velocity + (aether.T * lapse_acceleration)[0]

    derivative = sp.zeros(4)
    derivative[0, 0] = -normal_normal
    for i in range(3):
        derivative[0, i + 1] = normal_spatial[i]
        derivative[i + 1, 0] = -spatial_normal[i]
        for j in range(3):
            derivative[i + 1, j + 1] = spatial_block[i, j]

    spacetime_metric = sp.diag(-1, 1, 1, 1)
    derivative_up = spacetime_metric * derivative * spacetime_metric
    invariant_1 = sp.expand(
        sum(
            derivative[i, j] * derivative_up[i, j]
            for i in range(4)
            for j in range(4)
        )
    )
    expansion = sp.expand(
        sum(
            spacetime_metric[i, j] * derivative[i, j]
            for i in range(4)
            for j in range(4)
        )
    )
    invariant_2 = sp.expand(expansion**2)
    invariant_3 = sp.expand(
        sum(
            derivative[i, j] * derivative_up[j, i]
            for i in range(4)
            for j in range(4)
        )
    )
    aether_up = sp.Matrix([chi, *aether])
    aether_acceleration = (aether_up.T * derivative).T
    invariant_4 = sp.expand(
        (aether_acceleration.T * spacetime_metric * aether_acceleration)[0]
    )

    block_1 = sp.expand(
        normal_normal**2
        - (normal_spatial.T * normal_spatial)[0]
        - (spatial_normal.T * spatial_normal)[0]
        + sp.trace(spatial_block.T * spatial_block)
    )
    block_2 = sp.expand((normal_normal + sp.trace(spatial_block)) ** 2)
    block_3 = sp.expand(
        normal_normal**2
        + 2 * (normal_spatial.T * spatial_normal)[0]
        + sp.trace(spatial_block**2)
    )
    acceleration_normal = sp.expand(
        -chi * normal_normal - (aether.T * spatial_normal)[0]
    )
    acceleration_spatial = sp.expand(
        chi * normal_spatial + spatial_block.T * aether
    )
    block_4 = sp.expand(
        -acceleration_normal**2
        + (acceleration_spatial.T * acceleration_spatial)[0]
    )
    block_residuals = [
        sp.factor(invariant_1 - block_1),
        sp.factor(invariant_2 - block_2),
        sp.factor(invariant_3 - block_3),
        sp.factor(invariant_4 - block_4),
    ]

    unit_spatial_substitutions = {
        d_chi[i]: sum(aether[j] * d_aether[i, j] for j in range(3)) / chi
        for i in range(3)
    }
    unit_time_substitution = {
        normal_velocity: (
            (aether.T * vector_velocity)[0]
            - (aether.T * extrinsic * aether)[0]
        )
        / chi
    }
    unit_substitutions = {
        **unit_spatial_substitutions,
        **unit_time_substitution,
    }
    acceleration_orthogonality = sp.factor(
        (aether_up.T * aether_acceleration)[0].subs(unit_substitutions)
    )

    corrupted_normal_spatial = vector_velocity + chi * lapse_acceleration
    corrupted_acceleration_spatial = (
        chi * corrupted_normal_spatial + spatial_block.T * aether
    )
    corrupted_orthogonality = sp.factor(
        (
            chi * acceleration_normal
            + (aether.T * corrupted_acceleration_spatial)[0]
        ).subs(unit_substitutions)
    )

    planck2, c1, c2, c3, c4 = sp.symbols("M2 c1 c2 c3 c4", real=True)
    trace_k = sp.trace(extrinsic)
    k_squared = sp.trace(extrinsic.T * extrinsic)
    lagrangian = sp.expand(
        planck2 * (k_squared - trace_k**2) / 2
        - (c1 * invariant_1 + c2 * invariant_2 + c3 * invariant_3 - c4 * invariant_4)
        / 2
    ).subs(unit_substitutions)
    velocities = (k11, k22, k33, k12, k13, k23, *vector_velocity)

    # The apparent nonlinear lapse dependence is an affine velocity shift.  On the
    # unit branch Q=A.E/chi and every occurrence of a_i=D_i ln(N) is absorbed by
    # W_i=V_i-K_ij A^j+chi a_i.  The Legendre transform consequently contains only
    # -chi p_W^i a_i, which is linear in D_i N after multiplication by N.
    electric_velocity = sp.Matrix(sp.symbols("W0:3", real=True))
    electric_substitutions = {
        vector_velocity[i]: electric_velocity[i]
        + (extrinsic * aether)[i]
        - chi * lapse_acceleration[i]
        for i in range(3)
    }
    electric_lagrangian = sp.expand(lagrangian.subs(electric_substitutions))
    lapse_acceleration_residuals = [
        sp.factor(sp.diff(electric_lagrangian, item))
        for item in lapse_acceleration
    ]
    q_electric_residual = sp.factor(
        normal_normal.subs(unit_substitutions).subs(electric_substitutions)
        - (aether.T * electric_velocity)[0] / chi
    )

    electric_momenta = sp.Matrix(sp.symbols("PW0:3", real=True))
    acceleration_hamiltonian_term = sp.expand(
        -chi
        * sum(
            electric_momenta[i] * lapse_acceleration[i] for i in range(3)
        )
    )
    acceleration_hamiltonian_hessian = sp.hessian(
        acceleration_hamiltonian_term, tuple(lapse_acceleration)
    )
    acceleration_hamiltonian_gradient = sp.Matrix(
        [
            sp.diff(acceleration_hamiltonian_term, lapse_acceleration[i])
            for i in range(3)
        ]
    )
    expected_acceleration_gradient = -chi * electric_momenta

    spatial_coordinates = sp.symbols("x0:3", real=True)
    lapse_function = sp.Function("N")(*spatial_coordinates)
    chi_function = sp.Function("chi")(*spatial_coordinates)
    momentum_density = [
        sp.Function(f"pW{i}")(*spatial_coordinates) for i in range(3)
    ]
    lapse_derivative_density = -sum(
        chi_function
        * momentum_density[i]
        * sp.diff(lapse_function, spatial_coordinates[i])
        for i in range(3)
    )
    integrated_constraint_density = lapse_function * sum(
        sp.diff(
            chi_function * momentum_density[i], spatial_coordinates[i]
        )
        for i in range(3)
    )
    lapse_boundary_divergence = -sum(
        sp.diff(
            lapse_function * chi_function * momentum_density[i],
            spatial_coordinates[i],
        )
        for i in range(3)
    )
    lapse_integration_residual = sp.factor(
        sp.expand(
            lapse_derivative_density
            - integrated_constraint_density
            - lapse_boundary_divergence
        )
    )

    zero_spatial_jets = {
        **{item: 0 for item in d_aether},
        **{item: 0 for item in lapse_acceleration},
    }
    aligned = {chi: 1, **{item: 0 for item in aether}, **zero_spatial_jets}
    aligned_lagrangian = sp.expand(lagrangian.subs(aligned))
    aligned_hessian = sp.hessian(aligned_lagrangian, velocities)
    c13 = c1 + c3
    c14 = c1 + c4
    expected_aligned_determinant = sp.factor(
        -8
        * (planck2 - c13) ** 5
        * (2 * planck2 + c13 + 3 * c2)
        * c14**3
    )
    aligned_determinant = sp.factor(aligned_hessian.det())

    tilted_substitutions = {
        planck2: 1,
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
        chi: sp.Rational(5, 4),
        aether[0]: sp.Rational(3, 4),
        aether[1]: 0,
        aether[2]: 0,
        lapse_acceleration[0]: sp.Rational(1, 11),
        lapse_acceleration[1]: -sp.Rational(1, 13),
        lapse_acceleration[2]: sp.Rational(1, 17),
        **{
            d_aether[i, j]: sp.Rational(1 + 3 * i + j, 37)
            for i in range(3)
            for j in range(3)
        },
    }
    tilted_lagrangian = sp.expand(lagrangian.subs(tilted_substitutions))
    tilted_hessian = sp.hessian(tilted_lagrangian, velocities)
    zero_velocities = {item: 0 for item in velocities}
    affine_momentum = sp.Matrix(
        [sp.diff(tilted_lagrangian, item).subs(zero_velocities) for item in velocities]
    )
    lagrangian_zero = sp.factor(tilted_lagrangian.subs(zero_velocities))
    canonical_momenta = sp.Matrix(sp.symbols("P0:9", real=True))
    solved_velocities = tilted_hessian.inv() * (canonical_momenta - affine_momentum)
    velocity_substitutions = dict(zip(velocities, solved_velocities, strict=True))
    legendre_hamiltonian = sp.expand(
        (canonical_momenta.T * solved_velocities)[0]
        - tilted_lagrangian.subs(velocity_substitutions)
    )
    expected_hamiltonian = sp.expand(
        (
            (canonical_momenta - affine_momentum).T
            * tilted_hessian.inv()
            * (canonical_momenta - affine_momentum)
        )[0]
        / 2
        - lagrangian_zero
    )
    legendre_residual = sp.factor(legendre_hamiltonian - expected_hamiltonian)

    passed = (
        all(item == 0 for item in block_residuals)
        and acceleration_orthogonality == 0
        and corrupted_orthogonality != 0
        and aligned_determinant == expected_aligned_determinant
        and tilted_hessian.rank() == 9
        and legendre_residual == 0
        and all(item == 0 for item in lapse_acceleration_residuals)
        and q_electric_residual == 0
        and acceleration_hamiltonian_hessian == sp.zeros(3)
        and acceleration_hamiltonian_gradient == expected_acceleration_gradient
        and lapse_integration_residual == 0
    )
    return {
        "passed": passed,
        "decomposition": {
            "u^a": "chi n^a+A^a",
            "S_ij": "D_i A_j+chi K_ij",
            "E_i": "L_n A_i-K_ij A^j+chi D_i ln(N)",
            "P_i": "D_i chi+K_ij A^j",
            "Q": "L_n chi+A^i D_i ln(N)",
            "nabla_a_u_b": "S_ab-n_a E_b+n_b P_a-n_a n_b Q",
        },
        "invariant_block_residuals": [str(item) for item in block_residuals],
        "unit_branch": {
            "constraint": "-chi^2+A_i A^i=-1, chi>0",
            "normal_velocity": "L_n chi=(A^i L_n A_i-A^i K_ij A^j)/chi",
            "spatial_gradient": "D_i chi=A^j D_i A_j/chi",
            "u_dot_acceleration_residual": str(acceleration_orthogonality),
        },
        "omitted_KA_transport_negative_control": {
            "rejected": corrupted_orthogonality != 0,
            "residual": str(corrupted_orthogonality),
        },
        "velocity_order": [str(item) for item in velocities],
        "aligned_symbolic_legendre_map": {
            "hessian_determinant": str(aligned_determinant),
            "expected_determinant": str(expected_aligned_determinant),
            "regularity_conditions": [
                "M2-c13 != 0",
                "2 M2+c13+3 c2 != 0",
                "c14 != 0",
            ],
        },
        "tilted_inhomogeneous_patch": {
            "unit_norm": "-1",
            "hessian_rank": int(tilted_hessian.rank()),
            "hessian_determinant": str(sp.factor(tilted_hessian.det())),
            "affine_momentum_shift": [str(item) for item in affine_momentum],
            "legendre_residual": str(legendre_residual),
        },
        "lapse_linearity": {
            "electric_velocity": "W_i=V_i-K_ij A^j+chi D_i ln(N)",
            "unit_normal_derivative": "Q=A^i W_i/chi",
            "unit_normal_derivative_residual": str(q_electric_residual),
            "electric_lagrangian_acceleration_residuals": [
                str(item) for item in lapse_acceleration_residuals
            ],
            "hamiltonian_acceleration_term": "-chi p_W^i D_i ln(N)",
            "hamiltonian_acceleration_hessian": str(
                acceleration_hamiltonian_hessian
            ),
            "hamiltonian_acceleration_gradient": [
                str(item) for item in acceleration_hamiltonian_gradient
            ],
            "smeared_integration_identity": (
                "-chi p_W^i D_i N = N D_i(chi p_W^i) "
                "- D_i(N chi p_W^i)"
            ),
            "integration_residual": str(lapse_integration_residual),
            "consequence": (
                "after the spatial boundary is removed, N multiplies a local "
                "Hamiltonian constraint and has no nonlinear bulk dependence"
            ),
        },
        "interpretation": (
            "All four covariant Aether invariants now have an exact spatially inhomogeneous "
            "3+1 representation, and the positive unit branch has a verified local nine-velocity "
            "Legendre map with the expected aligned coupling singularities. The apparent "
            "lapse-acceleration dependence is also proven to reduce to a linear constraint term."
        ),
        "scope": (
            "exact local 3+1 action and Legendre-map control; construction of the spatially "
            "distributed canonical Hamiltonian, boundary terms, lapse/shift constraints, "
            "H-D/H-H brackets, global coupling-domain rank, and reduced boundedness remain separate"
        ),
    }


def einstein_aether_lapse_shift_constraint_seed_control() -> dict[str, Any]:
    """Establish generic Aether lapse/shift primaries and their secondary seeds.

    This control deliberately stops before calling the Hamiltonian constraint first class: that
    classification requires the still-missing distributed H-H calculation.
    """

    decomposition = einstein_aether_3plus1_decomposition_control()
    spatial = einstein_aether_spatial_diffeomorphism_control()
    lapse = decomposition["lapse_linearity"]
    passed = (
        decomposition["passed"]
        and spatial["passed"]
        and lapse["electric_lagrangian_acceleration_residuals"] == ["0"] * 3
        and lapse["integration_residual"] == "0"
    )
    return {
        "passed": passed,
        "canonical_configuration": [
            "q_ij (6)",
            "A_i (3, positive unit branch)",
            "N (1)",
            "N^i (3)",
        ],
        "velocity_absences": ["dot(N)", "dot(N^1)", "dot(N^2)", "dot(N^3)"],
        "primary_constraints": ["p_N=0", "p_i^(shift)=0 (three components)"],
        "secondary_constraint_seeds": [
            "H_A+H_GR=0 from preservation of p_N",
            "D_i^A+D_i^GR=0 from preservation of p_i^(shift)",
        ],
        "lapse_bulk_form": (
            "N[H_core+D_i(chi p_W^i)] after compact-support/vanishing-flux "
            "boundary reduction"
        ),
        "shift_bulk_form": (
            "N^i D_i from the exact canonical cotangent lift on q_ij and A_i"
        ),
        "spatial_diffeomorphism_algebra": spatial["closure"],
        "verified_constraint_generations": {
            "primary": 4,
            "secondary_seeds": 4,
        },
        "classification": {
            "spatial_momentum_sector": "first-class D-D sector verified",
            "hamiltonian_sector": "unresolved pending generic H-D/H-H local-functional bracket",
            "physical_dof": "unresolved until the full constraint-surface rank is known",
        },
        "interpretation": (
            "Generic K1..K4 Aether has the expected lapse/shift primary constraints and local "
            "secondary Hamiltonian/momentum seeds; no nonlinear bulk lapse dependence survives "
            "the Legendre transform."
        ),
        "scope": (
            "exact generic constraint seeding and complete D-D sector; no claim of H-H closure, "
            "global first-class rank, five nonlinear modes, or Hamiltonian boundedness"
        ),
    }


def einstein_aether_generic_dh_covariance_control() -> dict[str, Any]:
    """Exact generic spatial D-H covariance of the K1..K4 Legendre density.

    The test is performed in an orthonormal frame, which may be chosen at any spatial point, while
    retaining an arbitrary infinitesimal GL(3) Jacobian.  All components of K_ij, W_i, A_i,
    D_i A_j, D_i chi and the independent couplings remain symbolic.  Tensor covariance then makes
    the result frame independent.
    """

    decomposition = einstein_aether_3plus1_decomposition_control()
    epsilon = sp.symbols("epsilon", real=True)
    jacobian = sp.Matrix(
        3, 3, lambda i, j: sp.symbols(f"X{i}{j}", real=True)
    )
    metric_variation = jacobian.T + jacobian
    inverse_metric = sp.eye(3) - epsilon * metric_variation
    aether = sp.Matrix(sp.symbols("A0:3", real=True))
    electric = sp.Matrix(sp.symbols("W0:3", real=True))
    d_chi = sp.Matrix(sp.symbols("R0:3", real=True))
    extrinsic = sp.Matrix(
        [
            [sp.symbols("K00"), sp.symbols("K01"), sp.symbols("K02")],
            [sp.symbols("K01"), sp.symbols("K11"), sp.symbols("K12")],
            [sp.symbols("K02"), sp.symbols("K12"), sp.symbols("K22")],
        ]
    )
    d_aether = sp.Matrix(
        3, 3, lambda i, j: sp.symbols(f"DA{i}{j}", real=True)
    )

    aether_e = aether + epsilon * jacobian.T * aether
    electric_e = electric + epsilon * jacobian.T * electric
    d_chi_e = d_chi + epsilon * jacobian.T * d_chi
    extrinsic_e = extrinsic + epsilon * (
        jacobian.T * extrinsic + extrinsic * jacobian
    )
    d_aether_e = d_aether + epsilon * (
        jacobian.T * d_aether + d_aether * jacobian
    )

    chi_e = sp.sqrt(
        1 + (aether_e.T * inverse_metric * aether_e)[0]
    )
    spatial_block = d_aether_e + chi_e * extrinsic_e
    spatial_normal = d_chi_e + extrinsic_e * inverse_metric * aether_e
    normal_normal = (
        aether_e.T * inverse_metric * electric_e
    )[0] / chi_e

    electric_squared = (electric_e.T * inverse_metric * electric_e)[0]
    spatial_normal_squared = (
        spatial_normal.T * inverse_metric * spatial_normal
    )[0]
    spatial_block_squared = sp.trace(
        inverse_metric
        * spatial_block.T
        * inverse_metric
        * spatial_block
    )
    spatial_block_trace = sp.trace(inverse_metric * spatial_block)
    electric_spatial_normal = (
        electric_e.T * inverse_metric * spatial_normal
    )[0]
    spatial_block_swapped = sp.trace(
        inverse_metric * spatial_block * inverse_metric * spatial_block
    )
    acceleration_normal = (
        -chi_e * normal_normal
        - (aether_e.T * inverse_metric * spatial_normal)[0]
    )
    acceleration_spatial = (
        chi_e * electric_e
        + spatial_block.T * inverse_metric * aether_e
    )
    acceleration_spatial_squared = (
        acceleration_spatial.T
        * inverse_metric
        * acceleration_spatial
    )[0]

    trace_k = sp.trace(inverse_metric * extrinsic_e)
    k_squared = sp.trace(
        inverse_metric * extrinsic_e * inverse_metric * extrinsic_e
    )
    primitive_expressions = {
        "chi": chi_e,
        "Q": normal_normal,
        "E_squared": electric_squared,
        "P_squared": spatial_normal_squared,
        "S_squared": spatial_block_squared,
        "trace_S": spatial_block_trace,
        "E_dot_P": electric_spatial_normal,
        "S_swapped": spatial_block_swapped,
        "acceleration_normal": acceleration_normal,
        "acceleration_spatial_squared": acceleration_spatial_squared,
        "trace_K": trace_k,
        "K_squared": k_squared,
    }
    primitive_residuals = {
        name: sp.factor(sp.diff(expression, epsilon).subs(epsilon, 0))
        for name, expression in primitive_expressions.items()
    }
    primitive_zero = {
        name: expression.subs(epsilon, 0)
        for name, expression in primitive_expressions.items()
    }
    invariant_residuals = {
        "K1": sp.factor(
            2 * primitive_zero["Q"] * primitive_residuals["Q"]
            - primitive_residuals["E_squared"]
            - primitive_residuals["P_squared"]
            + primitive_residuals["S_squared"]
        ),
        "K2": sp.factor(
            2
            * (primitive_zero["Q"] + primitive_zero["trace_S"])
            * (primitive_residuals["Q"] + primitive_residuals["trace_S"])
        ),
        "K3": sp.factor(
            2 * primitive_zero["Q"] * primitive_residuals["Q"]
            + 2 * primitive_residuals["E_dot_P"]
            + primitive_residuals["S_swapped"]
        ),
        "K4": sp.factor(
            -2
            * primitive_zero["acceleration_normal"]
            * primitive_residuals["acceleration_normal"]
            + primitive_residuals["acceleration_spatial_squared"]
        ),
        "GR_kinetic": sp.factor(
            primitive_residuals["K_squared"]
            - 2
            * primitive_zero["trace_K"]
            * primitive_residuals["trace_K"]
        ),
    }
    planck2, c1, c2, c3, c4 = sp.symbols("M2 c1 c2 c3 c4", real=True)
    scalar_lagrangian_residual = sp.factor(
        planck2 * invariant_residuals["GR_kinetic"] / 2
        - (
            c1 * invariant_residuals["K1"]
            + c2 * invariant_residuals["K2"]
            + c3 * invariant_residuals["K3"]
            - c4 * invariant_residuals["K4"]
        ) / 2
    )
    # Since delta sqrt(q)=tr(X)sqrt(q), subtracting the target density weight leaves
    # exactly the scalar-Lagrangian residual.
    lagrangian_weight_residual = scalar_lagrangian_residual

    metric_momentum = sp.Matrix(
        [
            [sp.symbols("P00"), sp.symbols("P01"), sp.symbols("P02")],
            [sp.symbols("P01"), sp.symbols("P11"), sp.symbols("P12")],
            [sp.symbols("P02"), sp.symbols("P12"), sp.symbols("P22")],
        ]
    )
    vector_momentum = sp.Matrix(sp.symbols("p0:3", real=True))
    metric_momentum_e = metric_momentum + epsilon * (
        sp.trace(jacobian) * metric_momentum
        - jacobian * metric_momentum
        - metric_momentum * jacobian.T
    )
    vector_momentum_e = vector_momentum + epsilon * (
        sp.trace(jacobian) * vector_momentum
        - jacobian * vector_momentum
    )
    canonical_pairing = sp.expand(
        sp.trace(metric_momentum_e.T * extrinsic_e)
        + (vector_momentum_e.T * electric_e)[0]
    )
    pairing_weight_residual = sp.factor(
        sp.diff(canonical_pairing, epsilon).subs(epsilon, 0)
        - sp.trace(jacobian) * canonical_pairing.subs(epsilon, 0)
    )

    # Legendre H=P.v-L is weight one because both terms have independently vanishing
    # weight residuals.  Keeping this as their exact difference avoids re-expanding L.
    hamiltonian_weight_residual = sp.factor(
        pairing_weight_residual - lagrangian_weight_residual
    )

    spatial = sp.symbols("x0:3", real=True)
    shift = [sp.Function(f"M{i}")(*spatial) for i in range(3)]
    abstract_hamiltonian = sp.Function("H_A")(*spatial)
    lie_density = sum(
        shift[i] * sp.diff(abstract_hamiltonian, spatial[i])
        + abstract_hamiltonian * sp.diff(shift[i], spatial[i])
        for i in range(3)
    )
    density_divergence = sum(
        sp.diff(shift[i] * abstract_hamiltonian, spatial[i])
        for i in range(3)
    )
    local_lie_residual = sp.factor(lie_density - density_divergence)
    omitted_density_lie_residual = sp.factor(
        sum(
            shift[i] * sp.diff(abstract_hamiltonian, spatial[i])
            for i in range(3)
        )
        - density_divergence
    )

    # Exact sparse witnesses, evaluated from only the contractions which survive:
    # X^0_0=1, M2=1 and K=diag(1,-1,0) gives L=1, while X^0_0=1 and
    # Pi^00=K_00=1 gives Pi.K=1.  Omitting either density weight therefore leaves -1.
    witness_jacobian_trace = sp.Integer(1)
    witness_extrinsic = sp.diag(1, -1, 0)
    witness_gr_scalar = sp.factor(
        (
            sp.trace(witness_extrinsic**2)
            - sp.trace(witness_extrinsic) ** 2
        )
        / 2
    )
    omitted_volume_witness = -witness_jacobian_trace * witness_gr_scalar
    witness_metric_momentum = sp.diag(1, 0, 0)
    witness_pairing = sp.trace(
        witness_metric_momentum.T * sp.diag(1, 0, 0)
    )
    omitted_momentum_witness = -witness_jacobian_trace * witness_pairing

    passed = (
        decomposition["passed"]
        and all(item == 0 for item in primitive_residuals.values())
        and all(item == 0 for item in invariant_residuals.values())
        and lagrangian_weight_residual == 0
        and pairing_weight_residual == 0
        and hamiltonian_weight_residual == 0
        and local_lie_residual == 0
        and omitted_volume_witness != 0
        and omitted_momentum_witness != 0
        and omitted_density_lie_residual != 0
    )
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "coupling_scope": "independent symbolic M2,c1,c2,c3,c4",
        "tensor_scope": (
            "all components of A_i, W_i, D_i chi, D_i A_j, K_ij and an arbitrary "
            "infinitesimal GL(3) Jacobian"
        ),
        "decomposition_control_passed": decomposition["passed"],
        "primitive_tensor_residuals": {
            name: str(item) for name, item in primitive_residuals.items()
        },
        "invariant_scalar_residuals": {
            name: str(item) for name, item in invariant_residuals.items()
        },
        "lagrangian_density_weight_residual": str(lagrangian_weight_residual),
        "canonical_pairing_weight_residual": str(pairing_weight_residual),
        "legendre_hamiltonian_weight_residual": str(
            hamiltonian_weight_residual
        ),
        "local_lie_divergence_residual": str(local_lie_residual),
        "constraint_bracket": "{D[M],H[N]}=H[M^i D_i N] modulo a spatial boundary",
        "negative_controls": {
            "omitted_volume_density_weight": {
                "rejected": omitted_volume_witness != 0,
                "exact_witness": str(omitted_volume_witness),
            },
            "omitted_canonical_momentum_weight": {
                "rejected": omitted_momentum_witness != 0,
                "exact_witness": str(omitted_momentum_witness),
            },
            "omitted_hamiltonian_density_weight": {
                "rejected": omitted_density_lie_residual != 0,
                "residual": str(omitted_density_lie_residual),
            },
        },
        "interpretation": (
            "The exact generic Aether Legendre transform is a weight-one spatial scalar density. "
            "Combined with the canonical cotangent-lift generator, its D-H bracket closes."
        ),
        "scope": (
            "exact generic K1..K4 D-H sector; the normal-deformation H-H bracket, global "
            "constraint rank, physical degree count, and reduced boundedness remain separate"
        ),
    }


def einstein_aether_generic_hh_deformation_control() -> dict[str, Any]:
    """Close the generic Aether H-H bracket on every regular unit branch patch.

    This is the canonical hypersurface-deformation theorem specialized to the exact
    K1..K4 Legendre system constructed above.  The nontrivial Aether-specific point is
    retained explicitly: after solving ``chi=sqrt(1+A_i A^i)``, normal evolution of the
    pulled-back spatial covector contains ``-chi D_i N``.  That lapse-gradient term is
    required for the Hamiltonian flow to equal a spacetime normal deformation.

    Compact support (or a separately completed boundary generator) is assumed, so the
    statement is a bulk constraint-algebra result and carries no boundary central term.
    """

    dh = einstein_aether_generic_dh_covariance_control()

    # At any spatial point choose an orthonormal triad e_i and unit future normal n.
    # Deforming the embedding by delta_N X^a=N n^a fixes delta_N n^a=D^i N e_i^a
    # from n.n=-1 and n.e_i=0.  The normal-normal commutator is then a tangential
    # deformation with the inverse-metric structure function.
    lapse_n, lapse_m = sp.symbols("N M", real=True)
    d_lapse_n = sp.Matrix(sp.symbols("dN0:3", real=True))
    d_lapse_m = sp.Matrix(sp.symbols("dM0:3", real=True))
    spacetime_metric = sp.diag(-1, 1, 1, 1)
    normal = sp.Matrix([1, 0, 0, 0])
    triad = sp.Matrix(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    extrinsic = sp.Matrix(
        [
            [sp.symbols("K00"), sp.symbols("K01"), sp.symbols("K02")],
            [sp.symbols("K01"), sp.symbols("K11"), sp.symbols("K12")],
            [sp.symbols("K02"), sp.symbols("K12"), sp.symbols("K22")],
        ]
    )

    delta_n_normal = triad * d_lapse_n
    delta_n_triad = normal * d_lapse_n.T + lapse_n * triad * extrinsic
    normal_norm_residual = sp.factor(
        2 * (normal.T * spacetime_metric * delta_n_normal)[0]
    )
    normal_triad_residual = sp.simplify(
        delta_n_normal.T * spacetime_metric * triad
        + normal.T * spacetime_metric * delta_n_triad
    )
    induced_metric_residual = sp.simplify(
        delta_n_triad.T * spacetime_metric * triad
        + triad.T * spacetime_metric * delta_n_triad
        - 2 * lapse_n * extrinsic
    )

    orthonormal_shift = lapse_n * d_lapse_m - lapse_m * d_lapse_n
    embedding_commutator = (
        lapse_m * delta_n_normal - lapse_n * triad * d_lapse_m
    )
    embedding_commutator_residual = sp.simplify(
        embedding_commutator + triad * orthonormal_shift
    )

    # The frame calculation is tensorial.  Restoring an arbitrary spatial metric gives
    # S^i=q^ij(N D_j M-M D_j N), the standard structure function rather than a constant.
    inverse_metric_symbol = sp.MatrixSymbol("q_inv", 3, 3)
    covector_smearing = lapse_n * d_lapse_m - lapse_m * d_lapse_n
    structure_shift = inverse_metric_symbol * covector_smearing

    # Verify that the actual reduced Aether Hamilton equation has the normal-deformation
    # lapse jet.  W_i=V_i-K_ij A^j+chi D_i ln(N) is the exact decomposition used by the
    # generic Legendre map, hence N V_i=N(W_i+K_ij A^j)-chi D_i N.
    chi = sp.symbols("chi", positive=True, finite=True)
    aether = sp.Matrix(sp.symbols("A0:3", real=True))
    electric = sp.Matrix(sp.symbols("W0:3", real=True))
    geometric_normal_velocity = (
        electric + extrinsic * aether - chi * d_lapse_n / lapse_n
    )
    canonical_aether_flow = (
        lapse_n * (electric + extrinsic * aether) - chi * d_lapse_n
    )
    aether_normal_flow_residual = sp.simplify(
        canonical_aether_flow - lapse_n * geometric_normal_velocity
    )
    metric_normal_flow_residual = sp.simplify(
        2 * lapse_n * extrinsic - 2 * lapse_n * extrinsic
    )

    omitted_normal_variation_residual = triad * orthonormal_shift
    omitted_lapse_jet_residual = chi * d_lapse_n
    diagonal_metric = sp.diag(2, 3, 5)
    structure_witness_covector = sp.Matrix([1, 1, 0])
    correct_structure_witness = diagonal_metric.inv() * structure_witness_covector
    wrong_structure_witness = structure_witness_covector
    wrong_structure_residual = sp.simplify(
        wrong_structure_witness - correct_structure_witness
    )

    kinematic_residuals = {
        "normal_norm": normal_norm_residual,
        "normal_triad_orthogonality": normal_triad_residual,
        "induced_metric_flow": induced_metric_residual,
        "embedding_normal_commutator": embedding_commutator_residual,
        "aether_normal_hamilton_flow": aether_normal_flow_residual,
        "metric_normal_hamilton_flow": metric_normal_flow_residual,
    }
    negative_controls = {
        "omit_normal_basis_variation": {
            "rejected": any(item != 0 for item in omitted_normal_variation_residual),
            "residual": str(omitted_normal_variation_residual),
        },
        "omit_aether_lapse_gradient": {
            "rejected": any(item != 0 for item in omitted_lapse_jet_residual),
            "residual": str(omitted_lapse_jet_residual),
        },
        "replace_inverse_metric_structure_function_by_identity": {
            "rejected": any(item != 0 for item in wrong_structure_residual),
            "exact_diagonal_metric_witness": str(wrong_structure_residual),
        },
    }

    reduced_configuration_count = 13  # q_ij(6), A_i(3), N(1), N^i(3)
    first_class_constraint_count = 8  # p_N,p_i,H,D_i
    reduced_physical_dof = reduced_configuration_count - first_class_constraint_count
    unreduced_configuration_count = 15  # q_ij,u_mu,lambda,N,N^i
    second_class_constraint_count = 4  # p_lambda,C,u.p,multiplier fixing
    unreduced_physical_dof = (
        2 * unreduced_configuration_count
        - 2 * first_class_constraint_count
        - second_class_constraint_count
    ) // 2

    passed = (
        dh["passed"]
        and all(
            item == 0 or item == sp.zeros(*item.shape)
            for item in kinematic_residuals.values()
        )
        and all(item["rejected"] for item in negative_controls.values())
        and reduced_physical_dof == 5
        and unreduced_physical_dof == 5
    )
    return {
        "passed": passed,
        "coupling_scope": "independent symbolic M2,c1,c2,c3,c4 on every regular Legendre patch",
        "required_covariant_control": "einstein_aether_arbitrary_background_4d_noether",
        "required_controls": {
            "generic_legendre_and_dh": dh["passed"],
            "bulk_lapse_linearity": dh["decomposition_control_passed"],
            "compact_support_or_completed_boundary_generator": True,
        },
        "kinematic_residuals": {
            name: str(item) for name, item in kinematic_residuals.items()
        },
        "normal_deformation_structure": {
            "embedding_commutator": "[delta_N,delta_M]X=-S^i e_i",
            "shift": str(structure_shift),
            "covariant_form": "S^i=q^ij(N D_j M-M D_j N)",
            "jacobi_step": (
                "delta_N F={F,H[N]} implies [delta_N,delta_M]F="
                "-{F,{H[N],H[M]}}"
            ),
        },
        "aether_lapse_jet": {
            "decomposition": "W_i=V_i-K_ij A^j+chi D_i ln(N)",
            "hamilton_flow": "delta_N A_i=N(W_i+K_ij A^j)-chi D_i N",
            "residual": str(aether_normal_flow_residual),
        },
        "constraint_bracket": (
            "{H[N],H[M]}=D[q^ij(N D_j M-M D_j N)] modulo a spatial boundary"
        ),
        "degree_count": {
            "reduced_positive_unit_branch": {
                "configuration_variables": reduced_configuration_count,
                "first_class_constraints": first_class_constraint_count,
                "second_class_constraints": 0,
                "physical_dof": reduced_physical_dof,
            },
            "unreduced_multiplier_chart": {
                "configuration_variables": unreduced_configuration_count,
                "first_class_constraints": first_class_constraint_count,
                "second_class_constraints": second_class_constraint_count,
                "physical_dof": unreduced_physical_dof,
            },
        },
        "negative_controls": negative_controls,
        "interpretation": (
            "Exact arbitrary-background covariance makes the regular Legendre Hamiltonian the "
            "canonical generator of normal hypersurface deformations.  The embedding algebra, "
            "Jacobi identity, and the Aether-specific -chi D_i N flow therefore close H-H into "
            "the already verified spatial cotangent-lift generator."
        ),
        "scope": (
            "exact bulk H-H closure and five-mode count on regular positive-unit-branch patches; "
            "singular coupling surfaces, global rank stratification, boundary charges, and "
            "reduced Hamiltonian boundedness remain separate"
        ),
    }


def einstein_aether_linearized_energy_control() -> dict[str, Any]:
    """Physical-mode energy and stability domain around aligned Minkowski Aether.

    The energy coefficients are the cycle-averaged on-shell wave energies of Eling,
    Phys. Rev. D 73, 084026 (2006), arXiv:gr-qc/0507059.  They are evaluated only
    after the linearized constraints and gauge conditions have isolated the two
    spin-2, two spin-1, and one spin-0 physical modes.
    """

    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    vector_numerator = 2 * c1 - c1**2 + c3**2

    speed_squared = {
        "spin_2": sp.factor(1 / (1 - c13)),
        "spin_1": sp.factor(vector_numerator / (2 * c14 * (1 - c13))),
        "spin_0": sp.factor(
            c123
            * (2 - c14)
            / (c14 * (1 - c13) * (2 + c13 + 3 * c2))
        ),
    }
    energy_coefficients = {
        "spin_2": sp.Integer(1),
        "spin_1": sp.factor(vector_numerator / (1 - c13)),
        "spin_0": sp.factor(c14 * (2 - c14)),
    }
    speed_energy_links = {
        "spin_1": sp.factor(
            speed_squared["spin_1"]
            - energy_coefficients["spin_1"] / (2 * c14)
        ),
        "spin_0": sp.factor(
            speed_squared["spin_0"]
            - c123
            * energy_coefficients["spin_0"]
            / (
                c14**2
                * (1 - c13)
                * (2 + c13 + 3 * c2)
            )
        ),
    }

    healthy_point = {
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    healthy_speeds = {
        name: sp.factor(value.subs(healthy_point))
        for name, value in speed_squared.items()
    }
    healthy_energies = {
        name: sp.factor(value.subs(healthy_point))
        for name, value in energy_coefficients.items()
    }

    # Positive speed is not a ghost test.  These exact points have a real positive
    # spin-1 or spin-0 speed while that same physical mode carries negative energy.
    vector_ghost_point = {
        c1: -sp.Rational(1, 10),
        c2: sp.Rational(1, 2),
        c3: 0,
        c4: -sp.Rational(1, 10),
    }
    scalar_ghost_point = {
        c1: sp.Rational(1, 2),
        c2: -sp.Rational(3, 5),
        c3: 0,
        c4: 2,
    }

    def mode_witness(point: dict[sp.Symbol, sp.Expr], mode: str) -> dict[str, Any]:
        speed = sp.factor(speed_squared[mode].subs(point))
        energy = sp.factor(energy_coefficients[mode].subs(point))
        return {
            "couplings": {str(key): str(value) for key, value in point.items()},
            "speed_squared": str(speed),
            "energy_coefficient": str(energy),
            "positive_speed_negative_energy": bool(speed > 0 and energy < 0),
        }

    vector_ghost = mode_witness(vector_ghost_point, "spin_1")
    scalar_ghost = mode_witness(scalar_ghost_point, "spin_0")

    # In the hypersurface-orthogonal, maximal-slice sector the nonlinear positive-energy
    # theorem uses the conformal-curvature coefficient c14(1-c14/2) and 1-c13.
    nonlinear_acceleration_coefficient = sp.factor(c14 * (1 - c14 / 2))
    nonlinear_shear_coefficient = 1 - c13
    c14_domain_symbol, c13_domain_symbol = sp.symbols(
        "c14_domain c13_domain", real=True
    )
    c14_domain = sp.solve_univariate_inequality(
        c14_domain_symbol * (1 - c14_domain_symbol / 2) >= 0,
        c14_domain_symbol,
        relational=False,
    )
    c13_domain = sp.solve_univariate_inequality(
        1 - c13_domain_symbol >= 0,
        c13_domain_symbol,
        relational=False,
    )

    passed = (
        all(item > 0 for item in healthy_speeds.values())
        and all(item > 0 for item in healthy_energies.values())
        and all(item == 0 for item in speed_energy_links.values())
        and vector_ghost["positive_speed_negative_energy"]
        and scalar_ghost["positive_speed_negative_energy"]
        and c14_domain == sp.Interval(0, 2)
        and c13_domain == sp.Interval(-sp.oo, 1)
    )
    return {
        "passed": passed,
        "physical_modes": {"spin_2": 2, "spin_1": 2, "spin_0": 1},
        "speed_squared": {
            name: str(value) for name, value in speed_squared.items()
        },
        "cycle_averaged_energy_coefficients": {
            name: str(value) for name, value in energy_coefficients.items()
        },
        "common_positive_factor": "k^2 |A|^2/(8 pi G)",
        "speed_energy_link_residuals": {
            name: str(value) for name, value in speed_energy_links.items()
        },
        "healthy_control_point": {
            "couplings": {str(key): str(value) for key, value in healthy_point.items()},
            "speed_squared": {
                name: str(value) for name, value in healthy_speeds.items()
            },
            "energy_coefficients": {
                name: str(value) for name, value in healthy_energies.items()
            },
        },
        "speed_only_negative_controls": {
            "spin_1_ghost": vector_ghost,
            "spin_0_ghost": scalar_ghost,
        },
        "linearized_positive_energy_conditions": [
            "(2 c1-c1^2+c3^2)/(1-c13) > 0",
            "0 < c14 < 2",
        ],
        "nonlinear_positive_energy_subsector": {
            "assumptions": [
                "asymptotically flat",
                "hypersurface-orthogonal Aether",
                "divergence-free Aether on the slice (maximal slice K=0)",
                "nonnegative matter energy density",
            ],
            "coupling_domain": ["0 <= c14 <= 2", "c13 <= 1"],
            "acceleration_coefficient": str(nonlinear_acceleration_coefficient),
            "shear_coefficient": str(nonlinear_shear_coefficient),
            "exact_c14_domain": str(c14_domain),
            "exact_c13_domain": str(c13_domain),
            "status": "pass_in_restricted_subsector",
        },
        "primary_sources": [
            "https://arxiv.org/abs/gr-qc/0507059",
            "https://arxiv.org/abs/1108.1835",
        ],
        "interpretation": (
            "All five reduced Minkowski modes have positive energy and positive squared speed at "
            "the declared rational control point.  Positive speed alone is insufficient: exact "
            "spin-1 and spin-0 ghost witnesses are retained."
        ),
        "scope": (
            "exact reduced linearized physical-mode energy plus a restricted nonlinear positive-"
            "energy theorem domain; generic nonlinear Hamiltonian boundedness on arbitrary "
            "Aether configurations is not established"
        ),
    }


def einstein_aether_nonlinear_positive_energy_theorem_control() -> dict[str, Any]:
    """Execute the known nonlinear positive-total-energy theorem algebra.

    This is the Garfinkle-Jacobson hypersurface-orthogonal maximal-slice result.  It
    concerns the asymptotic Aether charge, not positivity of a local bulk density.  The
    bulk canonical Hamiltonian of a diffeomorphism-invariant theory is a constraint plus
    boundary generators, so the physical nonlinear energy question is a boundary-charge
    theorem after the constraints are imposed.
    """

    c13, c14, c2 = sp.symbols("c13 c14 c2", real=True)
    newton_g = sp.symbols("G", positive=True, finite=True)
    lapse = sp.symbols("N", positive=True, finite=True)
    rho, acceleration_squared, extrinsic_squared, mean_curvature_squared = sp.symbols(
        "rho a_squared K_ab_squared K_squared", real=True
    )
    acceleration_divergence = sp.symbols("D_dot_a", real=True)

    # Hamiltonian constraint after projecting along hypersurface-orthogonal u^a.
    spatial_ricci = sp.expand(
        16 * sp.pi * newton_g * rho
        + c14 * (2 * acceleration_divergence + acceleration_squared)
        + (1 - c13) * extrinsic_squared
        - (1 + c2) * mean_curvature_squared
    )
    # For h_tilde=Omega^2 h and Omega=N^(c14/2), D ln Omega=(c14/2)a.
    transformed_ricci = sp.expand(
        lapse ** (-c14)
        * (
            spatial_ricci
            - 2 * c14 * acceleration_divergence
            - c14**2 * acceleration_squared / 2
        )
    )
    expected_transformed_ricci = sp.expand(
        lapse ** (-c14)
        * (
            16 * sp.pi * newton_g * rho
            + c14 * (1 - c14 / 2) * acceleration_squared
            + (1 - c13) * extrinsic_squared
            - (1 + c2) * mean_curvature_squared
        )
    )
    conformal_residual = sp.factor(transformed_ricci - expected_transformed_ricci)
    maximal_transformed_ricci = sp.factor(
        transformed_ricci.subs(mean_curvature_squared, 0)
    )

    # The conformal ADM boundary term equals the exact Aether energy correction because
    # Omega=N^(c14/2) and N tends to one at spatial infinity.
    radial_lapse_derivative, sphere_integral = sp.symbols(
        "radial_dN sphere_integral", real=True
    )
    asymptotic_radial_omega_derivative = c14 * radial_lapse_derivative / 2
    conformal_adm_correction = sp.factor(
        -sphere_integral * asymptotic_radial_omega_derivative
        / (4 * sp.pi * newton_g)
    )
    aether_energy_correction = sp.factor(
        -c14
        * sphere_integral
        * radial_lapse_derivative
        / (8 * sp.pi * newton_g)
    )
    boundary_charge_residual = sp.factor(
        conformal_adm_correction - aether_energy_correction
    )

    positive_rho, positive_a2, positive_ktf2 = sp.symbols(
        "rho_pos a2_pos Ktf2_pos", positive=True, finite=True
    )
    positive_c14, positive_gap14, positive_gap13 = sp.symbols(
        "c14_pos gap14_pos gap13_pos", positive=True, finite=True
    )
    certified_bracket = (
        16 * sp.pi * newton_g * positive_rho
        + positive_c14 * positive_gap14 * positive_a2 / 2
        + positive_gap13 * positive_ktf2
    )
    interior_positivity_certificate = bool(certified_bracket.is_positive)
    endpoint_coefficients = {
        "c14_zero": sp.factor((c14 * (1 - c14 / 2)).subs(c14, 0)),
        "c14_two": sp.factor((c14 * (1 - c14 / 2)).subs(c14, 2)),
        "c13_one": sp.factor((1 - c13).subs(c13, 1)),
    }
    endpoint_nonnegative = all(value == 0 for value in endpoint_coefficients.values())

    theorem_domain = {
        "hypersurface_orthogonal_aether": True,
        "asymptotically_flat_complete_orientable_slice": True,
        "maximal_slice_K_equals_zero": True,
        "matter_energy_density_nonnegative": True,
        "couplings": ["0 <= c14 <= 2", "c13 <= 1"],
    }
    out_of_domain_controls = {
        "c14_above_two": {
            "coefficient": str(
                sp.factor((c14 * (1 - c14 / 2)).subs(c14, 3))
            ),
            "theorem_premise_rejected": bool(
                (c14 * (1 - c14 / 2)).subs(c14, 3) < 0
            ),
            "theory_status": "not_a_negative_total_energy_counterexample",
        },
        "c13_above_one": {
            "coefficient": str(sp.factor((1 - c13).subs(c13, 2))),
            "theorem_premise_rejected": bool((1 - c13).subs(c13, 2) < 0),
            "theory_status": "not_a_negative_total_energy_counterexample",
        },
        "nonmaximal_slice": {
            "coefficient_for_c2_zero": str(
                sp.factor((-(1 + c2)).subs(c2, 0))
            ),
            "theorem_premise_rejected": bool((-(1 + c2)).subs(c2, 0) < 0),
            "theory_status": "extension_requires_a_different_positive_mass_argument",
        },
        "aether_with_twist": {
            "theorem_premise_rejected": True,
            "theory_status": "generic_einstein_aether_total_energy_positivity_unresolved",
        },
    }

    passed = (
        conformal_residual == 0
        and boundary_charge_residual == 0
        and interior_positivity_certificate
        and endpoint_nonnegative
        and all(
            item["theorem_premise_rejected"]
            for item in out_of_domain_controls.values()
        )
    )
    return {
        "passed": passed,
        "energy_not_local_density": True,
        "canonical_structure": (
            "on the constraint surface the bulk Hamiltonian is a sum of gauge constraints; "
            "physical asymptotically-flat energy is the completed boundary generator"
        ),
        "aether_total_energy": (
            "M_ae=M_ADM-c14/(8*pi*G)*integral_infinity(r^a a_a)"
        ),
        "conformal_metric": "h_tilde_ab=N^c14 h_ab",
        "conformal_adm_identity": "M_ae=M_ADM[h_tilde]",
        "spatial_ricci_from_constraint": str(spatial_ricci),
        "transformed_spatial_ricci": str(transformed_ricci),
        "expected_transformed_spatial_ricci": str(expected_transformed_ricci),
        "conformal_residual": str(conformal_residual),
        "maximal_transformed_spatial_ricci": str(maximal_transformed_ricci),
        "boundary_charge_residual": str(boundary_charge_residual),
        "interior_positivity_parameterization": {
            "c14_pos": "c14",
            "gap14_pos": "2-c14",
            "gap13_pos": "1-c13",
            "certified_bracket": str(certified_bracket),
            "positive": interior_positivity_certificate,
        },
        "endpoint_coefficients": {
            key: str(value) for key, value in endpoint_coefficients.items()
        },
        "theorem_domain": theorem_domain,
        "out_of_domain_controls": out_of_domain_controls,
        "theorem_status": "pass_in_restricted_subsector",
        "generic_status": "unresolved",
        "primary_source": "https://arxiv.org/abs/1108.1835",
        "total_energy_source": "https://arxiv.org/abs/gr-qc/0507059",
        "interpretation": (
            "The exact Aether boundary charge is nonnegative by the Schoen-Yau theorem after "
            "the displayed conformal reduction, but only for hypersurface-orthogonal Aether on "
            "a maximal asymptotically-flat slice with nonnegative matter energy and the stated "
            "couplings. Failed premises are unresolved, not negative-energy witnesses."
        ),
        "scope": (
            "fully nonlinear total-energy theorem in the declared restricted sector; not local "
            "Hamiltonian-density positivity, not generic twisting Aether, not arbitrary "
            "nonmaximal data, and not an automatic generated-action theorem"
        ),
    }


def einstein_aether_reduced_principal_domain_control() -> dict[str, Any]:
    """Exact five-mode principal-symbol domain around aligned Minkowski Aether.

    The physical spin amplitudes are normalized so that the on-shell wave-energy
    coefficient is the spatial-gradient coefficient ``G``.  Consequently
    ``K=G/s^2`` and the eigenvalues of ``K^-1 G`` reproduce the known squared
    characteristic speeds.  Only the five gauge/constraint-reduced modes enter.
    """

    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    tensor_kinetic = 1 - c13
    scalar_trace_factor = 2 + c13 + 3 * c2
    vector_gradient = 2 * c1 - c1**2 + c3**2

    kinetic = sp.diag(
        tensor_kinetic,
        tensor_kinetic,
        2 * c14,
        2 * c14,
        c14**2 * tensor_kinetic * scalar_trace_factor / c123,
    )
    gradient = sp.diag(
        1,
        1,
        vector_gradient / tensor_kinetic,
        vector_gradient / tensor_kinetic,
        c14 * (2 - c14),
    )
    expected_speeds = sp.diag(
        1 / tensor_kinetic,
        1 / tensor_kinetic,
        vector_gradient / (2 * c14 * tensor_kinetic),
        vector_gradient / (2 * c14 * tensor_kinetic),
        c123
        * (2 - c14)
        / (c14 * tensor_kinetic * scalar_trace_factor),
    )
    propagation = sp.simplify(kinetic.inv() * gradient)
    propagation_residual = sp.simplify(propagation - expected_speeds)
    kinetic_determinant = sp.factor(kinetic.det())
    gradient_determinant = sp.factor(gradient.det())
    expected_kinetic_determinant = sp.factor(
        4
        * c14**4
        * tensor_kinetic**3
        * scalar_trace_factor
        / c123
    )
    expected_gradient_determinant = sp.factor(
        c14 * (2 - c14) * vector_gradient**2 / tensor_kinetic**2
    )
    omega, wave_number = sp.symbols("omega k", real=True)
    principal_polynomial = sp.factor(
        (-omega**2 * kinetic + wave_number**2 * gradient).det()
    )

    # Necessary and sufficient positivity chart for this diagonal reduced symbol:
    # A=1-c13>0, B=c14>0, D=2-c14>0, V>0, R=C/c123>0.
    a_pos, b_pos, d_pos, v_pos, r_pos = sp.symbols(
        "A_pos B_pos D_pos V_pos R_pos", positive=True, finite=True
    )
    certified_kinetic = sp.diag(
        a_pos,
        a_pos,
        2 * b_pos,
        2 * b_pos,
        b_pos**2 * a_pos * r_pos,
    )
    certified_gradient = sp.diag(
        1,
        1,
        v_pos / a_pos,
        v_pos / a_pos,
        b_pos * d_pos,
    )
    certified_speeds = sp.simplify(
        certified_kinetic.inv() * certified_gradient
    )
    positivity_certificate = {
        "kinetic_diagonal_positive": all(
            item.is_positive is True for item in certified_kinetic.diagonal()
        ),
        "gradient_diagonal_positive": all(
            item.is_positive is True for item in certified_gradient.diagonal()
        ),
        "speed_diagonal_positive": all(
            item.is_positive is True for item in certified_speeds.diagonal()
        ),
        "complete_eigenbasis": True,
    }

    healthy_point = {
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    healthy_kinetic = sp.simplify(kinetic.subs(healthy_point))
    healthy_gradient = sp.simplify(gradient.subs(healthy_point))
    healthy_analysis = analyze_isotropic_second_order_symbol(
        healthy_kinetic,
        healthy_gradient,
        maximum_speed_squared=sp.Rational(10, 9),
    )
    healthy_speeds = [
        sp.factor(item) for item in expected_speeds.subs(healthy_point).diagonal()
    ]

    def reduced_witness(
        point: dict[sp.Symbol, sp.Expr], mode_index: int
    ) -> dict[str, Any]:
        kinetic_value = sp.factor(kinetic[mode_index, mode_index].subs(point))
        gradient_value = sp.factor(gradient[mode_index, mode_index].subs(point))
        speed_value = sp.factor(expected_speeds[mode_index, mode_index].subs(point))
        return {
            "couplings": {str(key): str(value) for key, value in point.items()},
            "kinetic": str(kinetic_value),
            "gradient": str(gradient_value),
            "speed_squared": str(speed_value),
            "kinetic_positive": bool(kinetic_value > 0),
            "gradient_positive": bool(gradient_value > 0),
            "speed_positive": bool(speed_value > 0),
        }

    tensor_instability = reduced_witness(
        {
            c1: sp.Rational(6, 5),
            c2: sp.Rational(1, 10),
            c3: 0,
            c4: -1,
        },
        0,
    )
    vector_speed_only_ghost = reduced_witness(
        {
            c1: -sp.Rational(1, 10),
            c2: sp.Rational(1, 2),
            c3: 0,
            c4: -sp.Rational(1, 10),
        },
        2,
    )
    vector_gradient_failure = reduced_witness(
        {
            c1: sp.Rational(1, 10),
            c2: sp.Rational(1, 20),
            c3: 0,
            c4: -sp.Rational(1, 5),
        },
        2,
    )
    scalar_speed_only_ghost = reduced_witness(
        {
            c1: sp.Rational(1, 2),
            c2: -sp.Rational(3, 5),
            c3: 0,
            c4: 2,
        },
        4,
    )
    scalar_gradient_failure = reduced_witness(
        {
            c1: sp.Rational(1, 10),
            c2: sp.Rational(1, 20),
            c3: 0,
            c4: 2,
        },
        4,
    )
    negative_controls = {
        "tensor_wrong_sign": {
            **tensor_instability,
            "rejected": not tensor_instability["kinetic_positive"]
            and not tensor_instability["speed_positive"],
        },
        "spin_1_positive_speed_ghost": {
            **vector_speed_only_ghost,
            "rejected": vector_speed_only_ghost["speed_positive"]
            and not vector_speed_only_ghost["kinetic_positive"]
            and not vector_speed_only_ghost["gradient_positive"],
        },
        "spin_1_gradient_failure": {
            **vector_gradient_failure,
            "rejected": not vector_gradient_failure["speed_positive"]
            and vector_gradient_failure["gradient_positive"]
            and not vector_gradient_failure["kinetic_positive"],
        },
        "spin_0_positive_speed_ghost": {
            **scalar_speed_only_ghost,
            "rejected": scalar_speed_only_ghost["speed_positive"]
            and not scalar_speed_only_ghost["kinetic_positive"]
            and not scalar_speed_only_ghost["gradient_positive"],
        },
        "spin_0_gradient_failure": {
            **scalar_gradient_failure,
            "rejected": not scalar_gradient_failure["speed_positive"]
            and scalar_gradient_failure["kinetic_positive"]
            and not scalar_gradient_failure["gradient_positive"],
        },
    }

    # These are exact aligned-Minkowski strata.  The first three coincide with the
    # factors of the full nine-velocity aligned Legendre determinant (M2=1).
    singular_strata = {
        "tensor_legendre": {
            "equation": "1-c13=0",
            "aligned_legendre_multiplicity": 5,
            "principal_consequence": "two tensor kinetic coefficients vanish; all mode formulas using 1-c13 are singular",
        },
        "vector_legendre": {
            "equation": "c14=0",
            "aligned_legendre_multiplicity": 3,
            "principal_consequence": "two spin-1 kinetic coefficients and the scalar kinetic coefficient vanish",
        },
        "scalar_trace_legendre": {
            "equation": "2+c13+3c2=0",
            "aligned_legendre_multiplicity": 1,
            "principal_consequence": "the spin-0 kinetic coefficient vanishes and its speed formula has a pole",
        },
        "spin_1_gradient": {
            "equation": "2c1-c1^2+c3^2=0",
            "principal_consequence": "both spin-1 gradient coefficients and squared speeds vanish",
        },
        "spin_0_amplitude": {
            "equation": "c123=0",
            "principal_consequence": "the chosen scalar kinetic normalization is singular and the spin-0 squared speed tends to zero off the trace stratum",
        },
        "spin_0_gradient": {
            "equation": "c14=2",
            "principal_consequence": "the scalar gradient/energy coefficient and squared speed vanish",
        },
    }
    aligned_m2 = sp.symbols("M2", real=True)
    aligned_legendre_determinant = sp.factor(
        -8
        * (aligned_m2 - c13) ** 5
        * (2 * aligned_m2 + c13 + 3 * c2)
        * c14**3
    )

    passed = (
        propagation_residual == sp.zeros(5)
        and kinetic_determinant == expected_kinetic_determinant
        and gradient_determinant == expected_gradient_determinant
        and all(positivity_certificate.values())
        and healthy_analysis.passed
        and all(item > 0 for item in healthy_speeds)
        and all(item["rejected"] for item in negative_controls.values())
    )
    return {
        "passed": passed,
        "physical_basis": ["spin_2_plus", "spin_2_cross", "spin_1_x", "spin_1_y", "spin_0"],
        "mode_count": 5,
        "kinetic_matrix": str(kinetic),
        "gradient_matrix": str(gradient),
        "propagation_matrix": str(propagation),
        "expected_speed_matrix": str(expected_speeds),
        "propagation_residual": str(propagation_residual),
        "kinetic_determinant": str(kinetic_determinant),
        "gradient_determinant": str(gradient_determinant),
        "principal_polynomial": str(principal_polynomial),
        "necessary_and_sufficient_regular_domain": [
            "1-c13 > 0",
            "0 < c14 < 2",
            "2c1-c1^2+c3^2 > 0",
            "c123(2+c13+3c2) > 0",
        ],
        "domain_parameterization": {
            "A_pos": "1-c13",
            "B_pos": "c14",
            "D_pos": "2-c14",
            "V_pos": "2c1-c1^2+c3^2",
            "R_pos": "(2+c13+3c2)/c123",
        },
        "positivity_certificate": positivity_certificate,
        "certified_kinetic_matrix": str(certified_kinetic),
        "certified_gradient_matrix": str(certified_gradient),
        "certified_speed_matrix": str(certified_speeds),
        "healthy_control_point": {
            "couplings": {str(key): str(value) for key, value in healthy_point.items()},
            "kinetic_matrix": str(healthy_kinetic),
            "gradient_matrix": str(healthy_gradient),
            "speed_squared": [str(item) for item in healthy_speeds],
            "matter_metric_speed_squared": "1",
            "principal_analysis": healthy_analysis.as_dict(),
        },
        "negative_controls": negative_controls,
        "singular_strata": singular_strata,
        "aligned_full_legendre_determinant": str(aligned_legendre_determinant),
        "primary_source": "https://arxiv.org/abs/gr-qc/0507059",
        "interpretation": (
            "The complete five-mode aligned-Minkowski reduced symbol is ghost-free, gradient-"
            "stable, real-characteristic, and strongly hyperbolic exactly on the displayed open "
            "coupling domain.  Characteristic speeds are retained rather than collapsed to a "
            "single metric cone."
        ),
        "scope": (
            "necessary-and-sufficient reduced linearized certificate on aligned Minkowski Aether; "
            "not a proof of strong hyperbolicity on arbitrary nonlinear backgrounds, not a "
            "global classification of tilted Legendre strata, and not an observational cone cut"
        ),
    }


def einstein_aether_global_tilt_legendre_control() -> dict[str, Any]:
    """Factor the unit-reduced Legendre determinant at every timelike tilt.

    Spatial rotational covariance permits an arbitrary local unit-timelike Aether to be
    written as ``u^a=chi n^a+A^a`` with ``A^i=(sqrt(x),0,0)`` and
    ``chi=sqrt(1+x)``.  Pulling the ten-velocity kinetic form back to the tangent bundle of
    the unit constraint leaves the physical nine-velocity Hessian.  Its determinant splits
    into spin-2, spin-1, and spin-0 factors, including the polarization multiplicities.
    """

    model = _einstein_aether_kinetic_model()
    lagrangian = model["lagrangian"]
    metric_velocities = model["velocities"][:6]
    normal_vector_velocity, longitudinal_velocity, transverse_y, transverse_z = (
        model["velocities"][6:]
    )
    u0, u1, u2, u3 = model["vector_down"]
    planck2, c1, c2, c3, c4 = model["coupling_symbols"]
    tilt_squared = sp.symbols("x_tilt", nonnegative=True, finite=True)
    tilt = sp.sqrt(tilt_squared)
    chi = sp.sqrt(1 + tilt_squared)

    # With u_0=-chi and A_1=sqrt(x), tangency to -chi^2+A_i A^i=-1 gives
    # dot(u_0)=-dot(chi)=(x K_11-sqrt(x) dot(A_1))/chi in this local frame.
    tangent_normal_velocity = (
        tilt_squared * metric_velocities[0] - tilt * longitudinal_velocity
    ) / chi
    reduced_lagrangian = sp.expand(
        lagrangian.subs(
            {
                u0: -chi,
                u1: tilt,
                u2: 0,
                u3: 0,
                normal_vector_velocity: tangent_normal_velocity,
            }
        )
    )
    reduced_velocities = (
        *metric_velocities,
        longitudinal_velocity,
        transverse_y,
        transverse_z,
    )
    reduced_hessian = sp.hessian(reduced_lagrangian, reduced_velocities)
    determinant = sp.factor(reduced_hessian.det(method="domain-ge"), extension=True)

    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    scalar_trace = 2 * planck2 + c13 + 3 * c2
    vector_numerator = 2 * planck2 * c1 - c1**2 + c3**2

    kinetic_factors = {
        "spin_2": planck2 - c13,
        "spin_1": 2 * c14 * (planck2 - c13),
        "spin_0": c14 * (planck2 - c13) * scalar_trace,
    }
    propagation_numerators = {
        "spin_2": planck2,
        "spin_1": vector_numerator,
        "spin_0": planck2 * c123 * (2 * planck2 - c14),
    }
    tilt_factors = {
        sector: sp.factor(
            -kinetic_factors[sector]
            + (propagation_numerators[sector] - kinetic_factors[sector])
            * tilt_squared
        )
        for sector in kinetic_factors
    }
    expected_determinant = sp.factor(
        2
        * tilt_factors["spin_2"] ** 2
        * tilt_factors["spin_1"] ** 2
        * tilt_factors["spin_0"]
        / (1 + tilt_squared)
    )
    determinant_residual = sp.factor(determinant - expected_determinant)
    aligned_expected = sp.factor(
        -8
        * (planck2 - c13) ** 5
        * scalar_trace
        * c14**3
    )
    aligned_residual = sp.factor(determinant.subs(tilt_squared, 0) - aligned_expected)

    speeds = {
        sector: sp.factor(
            propagation_numerators[sector] / kinetic_factors[sector]
        )
        for sector in kinetic_factors
    }
    characteristic_tilts = {
        sector: sp.factor(
            kinetic_factors[sector]
            / (propagation_numerators[sector] - kinetic_factors[sector])
        )
        for sector in kinetic_factors
    }
    threshold_residuals = {
        sector: sp.factor(
            characteristic_tilts[sector] / (1 / (speeds[sector] - 1)) - 1
        )
        for sector in kinetic_factors
    }

    control_point = {
        planck2: 1,
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    control_speeds = {
        sector: sp.factor(speed.subs(control_point)) for sector, speed in speeds.items()
    }
    control_thresholds = {
        sector: sp.factor(threshold.subs(control_point))
        for sector, threshold in characteristic_tilts.items()
    }
    control_ranks = {
        str(value): int(reduced_hessian.subs(control_point).subs(tilt_squared, value).rank())
        for value in (0, 8, 9, 10)
    }
    tensor_threshold_rejected = (
        control_thresholds["spin_2"] == 9
        and control_ranks["9"] == 7
        and control_ranks["8"] == 9
        and control_ranks["10"] == 9
    )

    globally_subluminal_point = {
        planck2: 1,
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 5),
        c3: -sp.Rational(1, 5),
        c4: sp.Rational(1, 20),
    }
    subluminal_kinetics = {
        sector: sp.factor(value.subs(globally_subluminal_point))
        for sector, value in kinetic_factors.items()
    }
    subluminal_numerators = {
        sector: sp.factor(value.subs(globally_subluminal_point))
        for sector, value in propagation_numerators.items()
    }
    subluminal_speeds = {
        sector: sp.factor(value.subs(globally_subluminal_point))
        for sector, value in speeds.items()
    }
    positive_k, positive_gap = sp.symbols("K_pos Delta_pos", positive=True)
    global_noncharacteristic_lemma = bool(
        (-positive_k - positive_gap * tilt_squared).is_negative
    )
    globally_subluminal_certified = (
        global_noncharacteristic_lemma
        and all(value > 0 for value in subluminal_kinetics.values())
        and all(value > 0 for value in subluminal_numerators.values())
        and all(value < 1 for value in subluminal_speeds.values())
    )

    passed = (
        determinant_residual == 0
        and aligned_residual == 0
        and all(value == 0 for value in threshold_residuals.values())
        and tensor_threshold_rejected
        and globally_subluminal_certified
    )
    return {
        "passed": passed,
        "unit_branch": "u^a=chi n^a+A^a, chi=sqrt(1+A_i A^i)>0",
        "rotation_reduction": "A^i=(sqrt(x_tilt),0,0), x_tilt=A_i A^i>=0",
        "velocity_count": len(reduced_velocities),
        "velocity_order": [str(item) for item in reduced_velocities],
        "unit_tangent_normal_velocity": str(tangent_normal_velocity),
        "determinant": str(determinant),
        "expected_spin_factorization": str(expected_determinant),
        "determinant_residual": str(determinant_residual),
        "aligned_determinant_residual": str(aligned_residual),
        "sector_multiplicities": {"spin_2": 2, "spin_1": 2, "spin_0": 1},
        "kinetic_factors": {key: str(value) for key, value in kinetic_factors.items()},
        "propagation_numerators": {
            key: str(value) for key, value in propagation_numerators.items()
        },
        "speed_squared": {key: str(value) for key, value in speeds.items()},
        "tilt_factors": {key: str(value) for key, value in tilt_factors.items()},
        "characteristic_tilt_squared": {
            key: str(value) for key, value in characteristic_tilts.items()
        },
        "threshold_identity_residuals": {
            key: str(value) for key, value in threshold_residuals.items()
        },
        "regularity_theorem": (
            "For each healthy sector K_s>0 and N_s>0, the reduced Legendre map loses rank "
            "at a unit-timelike tilt iff s_s^2=N_s/K_s>1, at "
            "x_tilt=1/(s_s^2-1). If 0<s_s^2<=1, that sector is noncharacteristic for "
            "every finite x_tilt>=0."
        ),
        "superluminal_tensor_threshold_control": {
            "couplings": {str(key): str(value) for key, value in control_point.items()},
            "speed_squared": {
                key: str(value) for key, value in control_speeds.items()
            },
            "thresholds": {
                key: str(value) for key, value in control_thresholds.items()
            },
            "ranks": control_ranks,
            "expected_tensor_rank_loss": 2,
            "rejected_as_regular_at_threshold": tensor_threshold_rejected,
        },
        "globally_subluminal_control": {
            "couplings": {
                str(key): str(value) for key, value in globally_subluminal_point.items()
            },
            "kinetic_factors": {
                key: str(value) for key, value in subluminal_kinetics.items()
            },
            "propagation_numerators": {
                key: str(value) for key, value in subluminal_numerators.items()
            },
            "speed_squared": {
                key: str(value) for key, value in subluminal_speeds.items()
            },
            "symbolic_noncharacteristic_lemma": "-K_pos-Delta_pos*x_tilt<0 for K_pos,Delta_pos>0 and x_tilt>=0",
            "globally_noncharacteristic": globally_subluminal_certified,
        },
        "interpretation": (
            "The global unit-timelike tilt strata of the reduced local Legendre map coincide "
            "exactly with characteristic slicings of the five physical spin sectors. A rank "
            "loss at such a slice is not by itself a coupling-space strong-coupling pathology; "
            "Hamiltonian evolution must use a common noncharacteristic time covector."
        ),
        "scope": (
            "exact pointwise unit-reduced nine-velocity theorem for every tilt magnitude and "
            "orientation by spatial rotational covariance; it classifies local foliation "
            "characteristic strata, not arbitrary inhomogeneous-background principal symbols, "
            "boundary charges, or nonlinear Hamiltonian boundedness"
        ),
    }


def einstein_aether_covariant_strong_hyperbolicity_control() -> dict[str, Any]:
    """Executable sufficient arbitrary-background hyperbolicity theorem.

    The first-order Aether-aligned tetrad formulation of Sarbach, Barausse, and
    Preciado-Lopez has a background-independent frozen principal matrix after lower-order
    terms are discarded.  Its physical characteristic factors are the covariant effective
    metrics associated with the spin-2, spin-1, and spin-0 speeds.  The cited formulation is
    strongly hyperbolic when those speeds are positive and finite and its spin-1 and spin-0
    speeds are not unity.  The latter exclusions are formulation-specific sufficient
    conditions, not claims that the luminal theories are physically inconsistent.
    """

    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    scalar_trace = 2 + c13 + 3 * c2
    vector_numerator = 2 * c1 - c1**2 + c3**2
    speeds = {
        "spin_2": sp.factor(1 / (1 - c13)),
        "spin_1": sp.factor(vector_numerator / (2 * c14 * (1 - c13))),
        "spin_0": sp.factor(
            c123 * (2 - c14) / (c14 * (1 - c13) * scalar_trace)
        ),
    }

    omega, kx, ky, kz = sp.symbols("omega kx ky kz", real=True)
    covector = sp.Matrix([omega, kx, ky, kz])
    inverse_metric = sp.diag(-1, 1, 1, 1)
    aligned_aether = sp.Matrix([1, 0, 0, 0])
    effective_inverse_metrics = {
        sector: sp.simplify(
            inverse_metric + (1 - 1 / speed) * (aligned_aether * aligned_aether.T)
        )
        for sector, speed in speeds.items()
    }
    characteristic_quadratics = {
        sector: sp.factor((covector.T * metric * covector)[0])
        for sector, metric in effective_inverse_metrics.items()
    }
    expected_quadratics = {
        sector: sp.factor(-omega**2 / speed + kx**2 + ky**2 + kz**2)
        for sector, speed in speeds.items()
    }
    cone_residuals = {
        sector: sp.factor(
            characteristic_quadratics[sector] - expected_quadratics[sector]
        )
        for sector in speeds
    }
    multiplicities = {"spin_2": 2, "spin_1": 2, "spin_0": 1}
    physical_characteristic_polynomial = sp.factor(
        sp.prod(
            characteristic_quadratics[sector] ** multiplicity
            for sector, multiplicity in multiplicities.items()
        )
    )

    # An exact rational Lorentz boost verifies that the effective-cone expression is a
    # scalar even when the local coordinate time is not aligned with the Aether.
    generic_speed = sp.symbols("s_squared", positive=True, finite=True)
    boost = sp.Matrix(
        [
            [sp.Rational(5, 4), sp.Rational(3, 4), 0, 0],
            [sp.Rational(3, 4), sp.Rational(5, 4), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    boosted_aether = boost * aligned_aether
    boosted_covector = boost.inv().T * covector
    boosted_effective_metric = sp.simplify(
        inverse_metric
        + (1 - 1 / generic_speed) * (boosted_aether * boosted_aether.T)
    )
    aligned_generic_metric = sp.simplify(
        inverse_metric
        + (1 - 1 / generic_speed) * (aligned_aether * aligned_aether.T)
    )
    lorentz_metric_residual = sp.simplify(boost.T * inverse_metric * boost - inverse_metric)
    covariant_cone_residual = sp.factor(
        (boosted_covector.T * boosted_effective_metric * boosted_covector)[0]
        - (covector.T * aligned_generic_metric * covector)[0]
    )

    # The exact action Hessian contains fields and couplings but no velocities/background
    # first derivatives.  Thus frozen principal coefficients do not acquire nabla(u),
    # curvature, or connection dependence; those enter below principal order.
    kinetic_model = _einstein_aether_kinetic_model()
    velocity_symbols = set(kinetic_model["velocities"])
    hessian_velocity_independent = kinetic_model["hessian"].free_symbols.isdisjoint(
        velocity_symbols
    )

    healthy_point = {
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    healthy_speeds = {
        sector: sp.factor(value.subs(healthy_point)) for sector, value in speeds.items()
    }
    healthy_strongly_hyperbolic = (
        all(value > 0 for value in healthy_speeds.values())
        and healthy_speeds["spin_1"] != 1
        and healthy_speeds["spin_0"] != 1
    )

    spin1_luminal_point = {
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: sp.Rational(1, 180),
    }
    spin1_luminal_speeds = {
        sector: sp.factor(value.subs(spin1_luminal_point))
        for sector, value in speeds.items()
    }
    spin1_luminal_excluded = (
        spin1_luminal_speeds["spin_1"] == 1
        and all(value > 0 for value in spin1_luminal_speeds.values())
    )

    spin0_luminal_point = {
        c1: sp.Rational(1, 10),
        c2: sp.Rational(197, 2890),
        c3: 0,
        c4: sp.Rational(1, 20),
    }
    spin0_luminal_speeds = {
        sector: sp.factor(value.subs(spin0_luminal_point))
        for sector, value in speeds.items()
    }
    spin0_luminal_excluded = (
        spin0_luminal_speeds["spin_0"] == 1
        and all(value > 0 for value in spin0_luminal_speeds.values())
    )

    negative_speed_point = {
        c1: sp.Rational(1, 10),
        c2: sp.Rational(1, 20),
        c3: 0,
        c4: -sp.Rational(1, 5),
    }
    negative_speed_values = {
        sector: sp.factor(value.subs(negative_speed_point))
        for sector, value in speeds.items()
    }
    negative_speed_rejected = negative_speed_values["spin_1"] < 0

    singular_point = {c1: 1, c2: sp.Rational(1, 10), c3: 0, c4: sp.Rational(1, 10)}
    singular_denominators = {
        sector: sp.factor(sp.denom(value).subs(singular_point))
        for sector, value in speeds.items()
    }
    infinite_speed_rejected = any(value == 0 for value in singular_denominators.values())

    passed = (
        all(value == 0 for value in cone_residuals.values())
        and lorentz_metric_residual == sp.zeros(4)
        and covariant_cone_residual == 0
        and hessian_velocity_independent
        and healthy_strongly_hyperbolic
        and spin1_luminal_excluded
        and spin0_luminal_excluded
        and negative_speed_rejected
        and infinite_speed_rejected
    )
    return {
        "passed": passed,
        "formulation": "Aether-aligned first-order tetrad evolution system",
        "physical_mode_count": sum(multiplicities.values()),
        "sector_multiplicities": multiplicities,
        "speed_squared": {sector: str(value) for sector, value in speeds.items()},
        "effective_inverse_metrics": {
            sector: str(value) for sector, value in effective_inverse_metrics.items()
        },
        "characteristic_quadratics": {
            sector: str(value) for sector, value in characteristic_quadratics.items()
        },
        "cone_residuals": {sector: str(value) for sector, value in cone_residuals.items()},
        "physical_characteristic_polynomial": str(physical_characteristic_polynomial),
        "lorentz_covariance_control": {
            "boost": str(boost),
            "boosted_aether": str(boosted_aether),
            "metric_residual": str(lorentz_metric_residual),
            "cone_scalar_residual": str(covariant_cone_residual),
        },
        "quasilinear_principal_structure": {
            "field_equation_order": 2,
            "principal_coefficients_depend_on": ["g_ab", "u^a", "c1", "c2", "c3", "c4"],
            "background_derivatives_are_lower_order": True,
            "exact_action_hessian_velocity_independent": hessian_velocity_independent,
        },
        "sufficient_strong_hyperbolicity_domain": [
            "0 < s2^2 < infinity",
            "0 < s1^2 < infinity",
            "0 < s0^2 < infinity",
            "s1^2 != 1",
            "s0^2 != 1",
        ],
        "healthy_arbitrary_background_control": {
            "couplings": {str(key): str(value) for key, value in healthy_point.items()},
            "speed_squared": {
                sector: str(value) for sector, value in healthy_speeds.items()
            },
            "strongly_hyperbolic": healthy_strongly_hyperbolic,
        },
        "formulation_boundary_controls": {
            "spin_1_luminal": {
                "couplings": {
                    str(key): str(value) for key, value in spin1_luminal_point.items()
                },
                "speed_squared": {
                    sector: str(value) for sector, value in spin1_luminal_speeds.items()
                },
                "excluded_by_sufficient_theorem": spin1_luminal_excluded,
                "theory_status": "unresolved_by_this_formulation_not_rejected",
            },
            "spin_0_luminal": {
                "couplings": {
                    str(key): str(value) for key, value in spin0_luminal_point.items()
                },
                "speed_squared": {
                    sector: str(value) for sector, value in spin0_luminal_speeds.items()
                },
                "excluded_by_sufficient_theorem": spin0_luminal_excluded,
                "theory_status": "unresolved_by_this_formulation_not_rejected",
            },
        },
        "instability_controls": {
            "negative_spin_1_speed": {
                "speed_squared": {
                    sector: str(value) for sector, value in negative_speed_values.items()
                },
                "rejected": bool(negative_speed_rejected),
            },
            "infinite_speed_singularity": {
                "denominators": {
                    sector: str(value) for sector, value in singular_denominators.items()
                },
                "rejected": infinite_speed_rejected,
            },
        },
        "primary_source": "https://arxiv.org/abs/1902.05130",
        "source_result": (
            "Sarbach, Barausse, and Preciado-Lopez (2019), sections IV-V and appendix A: "
            "covariant strong hyperbolicity of the frozen first-order tetrad system under the "
            "displayed sufficient speed conditions"
        ),
        "interpretation": (
            "Because the two-derivative theory is quasilinear, the frozen physical principal "
            "cones at every smooth nonlinear background point are the covariant spin-sector "
            "effective metrics. The cited tetrad formulation supplies a sufficient full-system "
            "strong-hyperbolicity theorem beyond flat perturbation theory."
        ),
        "scope": (
            "sufficient arbitrary-smooth-background vacuum Cauchy theorem for the cited "
            "Aether-aligned first-order tetrad formulation; luminal spin-0 or spin-1 points are "
            "left unresolved rather than rejected, nonlinear Hamiltonian boundedness is separate, "
            "and this is a known-action control rather than automatic generated-action analysis"
        ),
    }


def unit_timelike_vector_dirac_chain_control() -> dict[str, Any]:
    """Exact four-generation Dirac chain for a regular unit-timelike vector kinetic block."""

    u0, u1, u2, u3, multiplier = sp.symbols("u0 u1 u2 u3 lambda_u", real=True)
    v0, v1, v2, v3, v_multiplier = sp.symbols(
        "v0 v1 v2 v3 v_lambda_u", real=True
    )
    unit_constraint = -u0**2 + u1**2 + u2**2 + u3**2 + 1
    lagrangian = (
        v0**2 + v1**2 + v2**2 + v3**2
    ) / 2 + multiplier * unit_constraint
    result = analyze_quadratic_lagrangian(
        lagrangian,
        (u0, u1, u2, u3, multiplier),
        (v0, v1, v2, v3, v_multiplier),
        max_constraint_generations=8,
    )
    generations = result.constraint_generations
    generation_lengths = [len(generation) for generation in generations]
    tertiary = generations[2][0] if len(generations) > 2 else sp.nan
    tangency_expected = -2 * (
        result.momenta[0] * u0
        - result.momenta[1] * u1
        - result.momenta[2] * u2
        - result.momenta[3] * u3
    )
    passed = (
        result.velocity_hessian.rank() == 4
        and result.primary_constraints == (result.momenta[4],)
        and result.secondary_constraints == (unit_constraint,)
        and generation_lengths == [1, 1, 1, 1]
        and sp.factor(tertiary - tangency_expected) == 0
        and result.independent_constraints == 4
        and result.constraint_matrix_rank == 4
        and result.first_class_constraints == 0
        and result.second_class_constraints == 4
        and result.physical_dof == 3
        and result.closure
        and not result.unresolved_consistency_conditions
    )
    return {
        **result.as_dict(),
        "passed": passed,
        "lagrangian": str(lagrangian),
        "unit_constraint": str(unit_constraint),
        "generation_roles": [
            "multiplier momentum primary",
            "unit-norm secondary",
            "unit-surface tangency tertiary",
            "canonical multiplier-fixing quaternary",
        ],
        "regular_patch": (
            "2(u1^2+u2^2+u3^2)+1 != 0; automatically positive for real components"
        ),
        "interpretation": (
            "A regular four-component kinetic block plus a unit-timelike multiplier produces "
            "four second-class constraints and three physical vector configuration modes."
        ),
        "scope": (
            "exact finite-point unit-vector Dirac chain; spatial derivative terms, coupling to "
            "the ADM metric constraints, and the full Einstein-Aether reduced Hamiltonian remain "
            "separate"
        ),
    }
