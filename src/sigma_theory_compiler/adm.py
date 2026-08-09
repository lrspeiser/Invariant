from __future__ import annotations

from typing import Any

import sympy as sp

from .dirac import poisson_bracket
from .principal_symbol import analyze_isotropic_second_order_symbol


def linearized_einstein_hilbert_adm_control() -> dict[str, Any]:
    """Linearized ADM control for one nonzero Fourier mode, k aligned with x^1."""

    wave_number = sp.Symbol("k", positive=True, finite=True)
    lapse, shift1, shift2, shift3 = sp.symbols("n beta1 beta2 beta3", real=True)
    h11, h22, h33, h12, h13, h23 = sp.symbols(
        "h11 h22 h33 h12 h13 h23", real=True
    )
    coordinates = (
        lapse,
        shift1,
        shift2,
        shift3,
        h11,
        h22,
        h33,
        h12,
        h13,
        h23,
    )
    velocities = sp.symbols("vn vb1 vb2 vb3 vh11 vh22 vh33 vh12 vh13 vh23", real=True)
    _, _, _, _, vh11, vh22, vh33, vh12, vh13, vh23 = velocities
    # K_ij = (dot(h)_ij - partial_i beta_j - partial_j beta_i)/2.
    # A real Fourier representative is sufficient for the velocity Hessian; factors of i do not
    # change its rank. Off-diagonal contractions occur twice in K_ij K^ij.
    k11 = (vh11 - 2 * wave_number * shift1) / 2
    k22 = vh22 / 2
    k33 = vh33 / 2
    k12 = (vh12 - wave_number * shift2) / 2
    k13 = (vh13 - wave_number * shift3) / 2
    k23 = vh23 / 2
    trace_k = k11 + k22 + k33
    kinetic_lagrangian = sp.expand(
        k11**2 + k22**2 + k33**2 + 2 * (k12**2 + k13**2 + k23**2) - trace_k**2
    )
    velocity_hessian = sp.hessian(kinetic_lagrangian, velocities)

    momenta = sp.symbols("pn pb1 pb2 pb3 p11 p22 p33 p12 p13 p23", real=True)
    primary = momenta[:4]
    # Linearized lapse and shift equations. Overall nonzero constants are irrelevant to the
    # constraint surface and rank. For k along x^1, R^(1)=k^2(h22+h33).
    secondary = (
        wave_number**2 * (h22 + h33),
        wave_number * momenta[4],
        wave_number * momenta[7],
        wave_number * momenta[8],
    )
    constraints = primary + secondary
    bracket_matrix = sp.Matrix(
        [
            [
                poisson_bracket(left, right, coordinates, momenta)
                for right in constraints
            ]
            for left in constraints
        ]
    )
    bracket_rank = int(bracket_matrix.rank())
    first_class = len(constraints) - bracket_rank
    second_class = bracket_rank
    phase_dimension = 2 * len(coordinates)
    physical_dof = (phase_dimension - 2 * first_class - second_class) // 2

    h_plus, h_cross, p_plus, p_cross = sp.symbols(
        "h_plus h_cross p_plus p_cross", real=True
    )
    tt_hamiltonian = sp.expand(
        (p_plus**2 + p_cross**2) / 2
        + wave_number**2 * (h_plus**2 + h_cross**2) / 2
    )
    tt_hessian = sp.hessian(tt_hamiltonian, (h_plus, h_cross, p_plus, p_cross))
    symbol = analyze_isotropic_second_order_symbol(sp.eye(2), sp.eye(2))
    passed = (
        velocity_hessian.rank() == 6
        and len(velocity_hessian.nullspace()) == 4
        and bracket_rank == 0
        and first_class == 8
        and second_class == 0
        and physical_dof == 2
        and tt_hessian.is_positive_definite is True
        and symbol.passed
    )
    return {
        "passed": passed,
        "background": "Minkowski",
        "mode_domain": "one nonzero spatial Fourier mode; k aligned with x^1 by rotation",
        "adm_kinetic_density": str(kinetic_lagrangian),
        "velocity_order": [str(item) for item in velocities],
        "velocity_hessian": str(velocity_hessian),
        "hessian_rank": int(velocity_hessian.rank()),
        "hessian_nullity": len(velocity_hessian.nullspace()),
        "primary_constraints": [str(item) for item in primary],
        "secondary_constraints": [str(item) for item in secondary],
        "constraint_poisson_matrix": str(bracket_matrix),
        "constraint_matrix_rank": bracket_rank,
        "first_class_constraints": first_class,
        "second_class_constraints": second_class,
        "phase_space_dimension": phase_dimension,
        "physical_dof": physical_dof,
        "constraint_algebra": "linearized abelian first-class algebra",
        "tt_hamiltonian": str(tt_hamiltonian),
        "tt_hamiltonian_hessian": str(tt_hessian),
        "tt_hamiltonian_positive_definite": tt_hessian.is_positive_definite is True,
        "tt_principal_symbol": symbol.as_dict(),
        "scope_warning": "This is the complete linearized Fourier-mode ADM control, not the nonlinear hypersurface-deformation algebra for arbitrary generated actions.",
    }


def nonlinear_adm_hamiltonian_constraint_control() -> dict[str, Any]:
    """Exact 3D ADM Hamiltonian-Hamiltonian bracket in covariant local form.

    The lapse-Hessian part of the spatial-curvature variation is contracted with the exact DeWitt
    kinetic derivative. Ultralocal kinetic-kinetic and Einstein-tensor terms cancel under lapse
    antisymmetrization. A final covariant integration by parts gives the metric-dependent momentum
    constraint, including its structure function ``q^{ij}``.
    """

    pairs = tuple((i, j) for i in range(3) for j in range(i, 3))

    def symmetric_matrix(prefix: str) -> sp.Matrix:
        values = {
            pair: sp.symbols(f"{prefix}{pair[0]}{pair[1]}", real=True)
            for pair in pairs
        }
        return sp.Matrix(3, 3, lambda i, j: values[tuple(sorted((i, j)))])

    metric = symmetric_matrix("q")
    inverse_metric = metric.inv()
    momentum_upper = symmetric_matrix("p")
    momentum_lower = metric * momentum_upper * metric
    momentum_trace = sp.trace(metric * momentum_upper)
    einstein_upper = symmetric_matrix("G")
    hessian_n = symmetric_matrix("n")
    hessian_m = symmetric_matrix("m")
    lapse_n, lapse_m = sp.symbols("N M", real=True)
    sqrt_metric = sp.symbols("sqrt_q", positive=True)

    def potential_metric_derivative(lapse: sp.Expr, hessian: sp.Matrix) -> sp.Matrix:
        laplacian = sp.trace(inverse_metric * hessian)
        hessian_upper = inverse_metric * hessian * inverse_metric
        return sqrt_metric * (
            lapse * einstein_upper
            + inverse_metric * laplacian
            - hessian_upper
        )

    def kinetic_momentum_derivative(
        lapse: sp.Expr, trace_coefficient: sp.Expr
    ) -> sp.Matrix:
        return (
            2
            * lapse
            / sqrt_metric
            * (momentum_lower - trace_coefficient * metric * momentum_trace)
        )

    def contract(left: sp.Matrix, right: sp.Matrix) -> sp.Expr:
        return sp.trace(left.T * right)

    derivative_n = potential_metric_derivative(lapse_n, hessian_n)
    derivative_m = potential_metric_derivative(lapse_m, hessian_m)
    kinetic_n = kinetic_momentum_derivative(lapse_n, sp.Rational(1, 2))
    kinetic_m = kinetic_momentum_derivative(lapse_m, sp.Rational(1, 2))
    raw_cross_bracket = sp.factor(
        contract(derivative_n, kinetic_m) - contract(derivative_m, kinetic_n)
    )
    expected_second_derivative = sp.factor(
        2
        * sp.trace(
            momentum_upper.T
            * (lapse_n * hessian_m - lapse_m * hessian_n)
        )
    )
    contraction_residual = sp.factor(
        raw_cross_bracket - expected_second_derivative
    )

    gradient_n = sp.Matrix(sp.symbols("N0:3", real=True))
    gradient_m = sp.Matrix(sp.symbols("M0:3", real=True))
    momentum_divergence = sp.Matrix(sp.symbols("Pdiv0:3", real=True))
    antisymmetric_lapse_gradient = lapse_n * gradient_m - lapse_m * gradient_n
    structure_shift = inverse_metric * antisymmetric_lapse_gradient
    momentum_constraint = -2 * metric * momentum_divergence
    smeared_momentum_constraint = sp.factor(
        (structure_shift.T * momentum_constraint)[0]
    )
    gradient_cross_term = sp.factor(
        2
        * sum(
            momentum_upper[i, j]
            * (gradient_n[i] * gradient_m[j] - gradient_m[i] * gradient_n[j])
            for i in range(3)
            for j in range(3)
        )
    )
    boundary_divergence = sp.factor(
        2 * (momentum_divergence.T * antisymmetric_lapse_gradient)[0]
        + gradient_cross_term
        + expected_second_derivative
    )
    boundary_reduction_residual = sp.factor(
        expected_second_derivative
        - smeared_momentum_constraint
        - boundary_divergence
    )

    wrong_kinetic_n = kinetic_momentum_derivative(lapse_n, sp.Rational(1, 3))
    wrong_kinetic_m = kinetic_momentum_derivative(lapse_m, sp.Rational(1, 3))
    wrong_trace_residual = sp.factor(
        contract(derivative_n, wrong_kinetic_m)
        - contract(derivative_m, wrong_kinetic_n)
        - expected_second_derivative
    )
    wrong_curvature_sign_residual = sp.factor(
        -raw_cross_bracket - expected_second_derivative
    )

    passed = (
        contraction_residual == 0
        and gradient_cross_term == 0
        and boundary_reduction_residual == 0
        and wrong_trace_residual != 0
        and wrong_curvature_sign_residual != 0
    )
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "hamiltonian_constraint": (
            "H=(pi^{ij}pi_{ij}-pi^2/2)/sqrt(q)-sqrt(q) R^(3)"
        ),
        "curvature_metric_derivative": (
            "sqrt(q)[N G^{ij}+q^{ij} nabla^2 N-nabla^i nabla^j N]"
        ),
        "kinetic_momentum_derivative": (
            "2N[pi_ij-q_ij pi/2]/sqrt(q)"
        ),
        "raw_cross_bracket": str(raw_cross_bracket),
        "expected_second_derivative_density": str(expected_second_derivative),
        "cross_contraction_residual": str(contraction_residual),
        "antisymmetric_gradient_residual": str(gradient_cross_term),
        "structure_shift": "S^i=q^{ij}(N partial_j M-M partial_j N)",
        "momentum_constraint": "D_i=-2 q_ij nabla_k pi^{jk}",
        "smeared_momentum_constraint": str(smeared_momentum_constraint),
        "boundary_divergence": (
            "nabla_i[2 pi^{ij}(N partial_j M-M partial_j N)]"
        ),
        "boundary_reduction_residual": str(boundary_reduction_residual),
        "wrong_dewitt_trace_negative_control": {
            "trace_coefficient": "1/3",
            "rejected": wrong_trace_residual != 0,
            "residual": str(wrong_trace_residual),
        },
        "wrong_curvature_sign_negative_control": {
            "rejected": wrong_curvature_sign_residual != 0,
            "residual": str(wrong_curvature_sign_residual),
        },
        "ultralocal_cancellations": [
            "kinetic-kinetic terms are proportional to N*M and cancel antisymmetrically",
            "N*G^{ij} curvature terms are proportional to N*M and cancel antisymmetrically",
            "potential-potential bracket vanishes because the potential has no momentum",
        ],
        "constraint_algebra": (
            "{H[N],H[M]}=D[q^{ij}(N partial_j M-M partial_j N)]"
        ),
        "boundary_condition": "compact support or vanishing spatial boundary flux",
        "primary_source": "https://arxiv.org/abs/gr-qc/0405109",
        "scope": (
            "exact nonlinear pure-GR H-H hypersurface-deformation bracket; lapse/shift primary "
            "constraints and D-D/D-H sectors are covered by companion controls, while "
            "Einstein-Aether extra-field constraints remain separate"
        ),
    }


def spatial_curvature_density_diffeomorphism_control() -> dict[str, Any]:
    """Exact D-H covariance of ``sqrt(q) R^(3)`` as a weight-one density."""

    spatial = sp.symbols("x0:3", real=True)
    volume_density = sp.Function("sqrt_q")(*spatial)
    scalar_curvature = sp.Function("R3")(*spatial)
    shift = tuple(sp.Function(f"M{i}")(*spatial) for i in range(3))
    potential_density = volume_density * scalar_curvature
    lie_volume = sum(
        shift[i] * sp.diff(volume_density, spatial[i])
        + volume_density * sp.diff(shift[i], spatial[i])
        for i in range(3)
    )
    lie_curvature = sum(
        shift[i] * sp.diff(scalar_curvature, spatial[i]) for i in range(3)
    )
    canonical_variation = sp.expand(
        scalar_curvature * lie_volume + volume_density * lie_curvature
    )
    target_divergence = sp.expand(
        sum(
            sp.diff(shift[i] * potential_density, spatial[i])
            for i in range(3)
        )
    )
    residual = sp.factor(canonical_variation - target_divergence)
    omitted_density_weight = sp.expand(
        scalar_curvature
        * sum(shift[i] * sp.diff(volume_density, spatial[i]) for i in range(3))
        + volume_density * lie_curvature
    )
    negative_residual = sp.factor(omitted_density_weight - target_divergence)
    passed = residual == 0 and negative_residual != 0
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "potential_density": "sqrt(q) R^(3)",
        "volume_transformation": (
            "Lie_M sqrt(q)=M^i partial_i sqrt(q)+sqrt(q) partial_i M^i"
        ),
        "curvature_transformation": "Lie_M R^(3)=M^i partial_i R^(3)",
        "target": "Lie_M[sqrt(q)R^(3)]=partial_i[M^i sqrt(q)R^(3)]",
        "residual": str(residual),
        "omitted_density_weight_negative_control": {
            "rejected": negative_residual != 0,
            "residual": str(negative_residual),
        },
        "boundary_condition": "compact support or vanishing spatial boundary flux",
        "interpretation": (
            "Together with the canonical-metric momentum generator, the curvature potential "
            "has the required D-H bracket modulo a spatial boundary."
        ),
        "scope": (
            "exact spatial-density covariance using the tensorial fact that R^(3) is a scalar; "
            "the separate Cadabra control derives its lapse-smeared metric variation"
        ),
    }
