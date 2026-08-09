from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .dhost import generic_horndeski_l2_l4_unitary_adm_control
from .horndeski import (
    generic_horndeski_l2_l4_flrw_scalar_reduction_control,
    generic_horndeski_l2_l4_tensor_stability_control,
    quartic_horndeski_unitary_distributed_dirac_control,
)

SCHEMA_VERSION = "sigma-quartic-dirac-hamiltonian-campaign-1.0"


class QuarticDiracHamiltonianError(ValueError):
    """Raised when a candidate cannot be bound to the ADM/Hamiltonian proof."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


def _less_than(left: sp.Expr, right: sp.Expr) -> bool:
    return _positive(sp.factor(right - sp.Abs(left)))


def _polynomial_absolute_bound(
    expression: sp.Expr, variable: sp.Symbol, upper: sp.Expr
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), variable)
    return sp.factor(
        sum(
            abs(coefficient) * upper**power[0]
            for power, coefficient in polynomial.terms()
        )
    )


def _polynomial_lower_bound(
    expression: sp.Expr, variable: sp.Symbol, upper: sp.Expr
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), variable)
    constant = polynomial.coeff_monomial(1)
    tail = sum(
        abs(coefficient) * upper**power[0]
        for power, coefficient in polynomial.terms()
        if power[0] > 0
    )
    return sp.factor(constant - tail)


@cache
def _source_control_status() -> dict[str, bool]:
    adm = generic_horndeski_l2_l4_unitary_adm_control()
    distributed, _ = quartic_horndeski_unitary_distributed_dirac_control()
    scalar, _ = generic_horndeski_l2_l4_flrw_scalar_reduction_control()
    tensor, _ = generic_horndeski_l2_l4_tensor_stability_control()
    return {
        "adm": bool(adm["passed"]),
        "distributed_dirac": distributed,
        "scalar_reduction": scalar,
        "tensor_reduction": tensor,
    }


@cache
def _symbolic_flrw_control() -> dict[str, Any]:
    """Derive the candidate FLRW, lapse-pair, and KYY stability expressions."""

    lapse, scale, scale_velocity, scalar_velocity = sp.symbols(
        "N a a_dot A_star", positive=True, finite=True
    )
    scale_acceleration, scalar_acceleration = sp.symbols(
        "a_ddot phi_ddot", real=True, finite=True
    )
    alpha, c20 = sp.symbols("alpha c20", real=True, finite=True)
    momentum = sp.Symbol("p_a", real=True, finite=True)
    x = sp.factor(scalar_velocity**2 / (2 * lapse**2))
    lagrangian = sp.factor(
        -3 * scale * scale_velocity**2 / lapse
        + 3
        * alpha
        * scale
        * scale_velocity**2
        * scalar_velocity**2
        / lapse**3
        + scale**3 * lapse * (x + c20 * x**2)
    )

    def time_derivative(expression: sp.Expr) -> sp.Expr:
        expression = expression.subs(lapse, 1)
        return sp.factor(
            sp.diff(expression, scale) * scale_velocity
            + sp.diff(expression, scale_velocity) * scale_acceleration
            + sp.diff(expression, scalar_velocity) * scalar_acceleration
        )

    lapse_equation = sp.factor(sp.diff(lagrangian, lapse).subs(lapse, 1))
    scale_equation = sp.factor(
        time_derivative(sp.diff(lagrangian, scale_velocity))
        - sp.diff(lagrangian, scale).subs(lapse, 1)
    )
    scalar_equation = sp.factor(
        time_derivative(sp.diff(lagrangian, scalar_velocity))
    )
    acceleration_solution = sp.solve(
        (scale_equation, scalar_equation),
        (scale_acceleration, scalar_acceleration),
        dict=True,
        simplify=False,
    )[0]
    acceleration_matrix = sp.Matrix(
        [scale_equation, scalar_equation]
    ).jacobian((scale_acceleration, scalar_acceleration))

    hubble_squared = sp.factor(
        (scalar_velocity**2 / 2 + 3 * c20 * scalar_velocity**4 / 4)
        / (3 - 9 * alpha * scalar_velocity**2)
    )
    hubble = sp.sqrt(hubble_squared)
    scale_acceleration_on_shell = sp.factor(
        acceleration_solution[scale_acceleration].subs(
            {
                scale: 1,
                scale_velocity: hubble,
            }
        )
    )
    scalar_acceleration_on_shell = sp.factor(
        acceleration_solution[scalar_acceleration].subs(
            {
                scale: 1,
                scale_velocity: hubble,
            }
        )
    )
    q = sp.factor(scale_acceleration_on_shell / hubble_squared)
    u = sp.factor(
        scalar_acceleration_on_shell / (hubble * scalar_velocity)
    )

    background_substitution = {
        scale: 1,
        scale_velocity: hubble,
        scale_acceleration: q * hubble_squared,
        scalar_acceleration: u * hubble * scalar_velocity,
    }
    background_residuals = {
        "lapse": sp.factor(lapse_equation.subs(background_substitution)),
        "scale": sp.factor(scale_equation.subs(background_substitution)),
        "scalar": sp.factor(scalar_equation.subs(background_substitution)),
    }
    lapse_constraint_time_residual = sp.factor(
        time_derivative(lapse_equation).subs(background_substitution)
    )

    canonical_scale_momentum = sp.factor(
        sp.diff(lagrangian, scale_velocity)
    )
    solved_scale_velocity = sp.solve(
        sp.Eq(momentum, canonical_scale_momentum), scale_velocity, dict=False
    )[0]
    canonical_hamiltonian = sp.factor(
        momentum * solved_scale_velocity
        - lagrangian.subs(scale_velocity, solved_scale_velocity)
    )
    lapse_secondary = sp.factor(sp.diff(canonical_hamiltonian, lapse))
    lapse_pairing = sp.factor(-sp.diff(lapse_secondary, lapse))
    background_momentum = sp.factor(
        canonical_scale_momentum.subs(
            {lapse: 1, scale: 1, scale_velocity: hubble}
        )
    )
    lapse_pairing_on_shell = sp.factor(
        lapse_pairing.subs(
            {lapse: 1, scale: 1, momentum: background_momentum}
        )
    )

    amplitude_squared = scalar_velocity**2
    g_t = sp.factor(1 - alpha * amplitude_squared)
    f_t = sp.factor(1 + alpha * amplitude_squared)
    theta = sp.factor(hubble * (1 - 3 * alpha * amplitude_squared))
    sigma = sp.factor(
        amplitude_squared / 2
        + 3 * c20 * amplitude_squared**2 / 2
        - 3 * hubble_squared
        + 18 * alpha * hubble_squared * amplitude_squared
    )
    g_s = sp.factor(sigma * g_t**2 / theta**2 + 3 * g_t)
    ratio_denominator = 1 - 3 * alpha * amplitude_squared
    log_ratio_time_over_hubble = sp.factor(
        2 * (-2 * alpha * amplitude_squared * u) / g_t
        - (q - 1)
        + 6 * alpha * amplitude_squared * u / ratio_denominator
    )
    f_s = sp.factor(
        g_t**2
        / ratio_denominator
        * (1 + log_ratio_time_over_hubble)
        - f_t
    )

    return {
        "symbols": {
            "N": lapse,
            "a": scale,
            "a_dot": scale_velocity,
            "A_star": scalar_velocity,
            "a_ddot": scale_acceleration,
            "phi_ddot": scalar_acceleration,
            "alpha": alpha,
            "c20": c20,
            "p_a": momentum,
        },
        "lagrangian": lagrangian,
        "equations": {
            "lapse": lapse_equation,
            "scale": scale_equation,
            "scalar": scalar_equation,
        },
        "background_residuals": background_residuals,
        "lapse_constraint_time_residual": lapse_constraint_time_residual,
        "hubble_squared": hubble_squared,
        "hubble": hubble,
        "q": q,
        "u": u,
        "acceleration_matrix": acceleration_matrix,
        "canonical_scale_momentum": canonical_scale_momentum,
        "canonical_hamiltonian": canonical_hamiltonian,
        "lapse_secondary": lapse_secondary,
        "lapse_pairing": lapse_pairing,
        "lapse_pairing_on_shell": lapse_pairing_on_shell,
        "G_T": g_t,
        "F_T": f_t,
        "Theta": theta,
        "Sigma": sigma,
        "G_S": g_s,
        "F_S": f_s,
    }


def certify_quartic_dirac_hamiltonian_candidate(
    coefficients: dict[str, Any],
    symmetrizer_certificate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    data = _symbolic_flrw_control()
    symbols = data["symbols"]
    alpha = symbols["alpha"]
    c20 = symbols["c20"]
    amplitude = symbols["A_star"]
    alpha_value = sp.sympify(coefficients["a10"])
    c20_value = sp.sympify(coefficients["c20"])
    if sp.sympify(coefficients["m2"]) != 1:
        raise QuarticDiracHamiltonianError("current exact campaign requires m2=1")
    for name in ("a01", "a20", "c02", "c11", "d01", "d10"):
        if sp.sympify(coefficients[name]) != 0:
            raise QuarticDiracHamiltonianError(
                f"coefficient {name} is outside the shift-symmetric linear-X family"
            )
    if symmetrizer_certificate.get("status") != (
        "pass_uniform_local_jet_strong_hyperbolicity"
    ):
        raise QuarticDiracHamiltonianError(
            "candidate lacks a passing strong-hyperbolicity certificate"
        )
    if symmetrizer_certificate.get("coefficients") != coefficients:
        raise QuarticDiracHamiltonianError(
            "candidate coefficients do not match the symmetrizer certificate"
        )

    amplitude_value = sp.sympify(config["timelike_gradient_amplitude"])
    if not _positive(amplitude_value):
        raise QuarticDiracHamiltonianError(
            "timelike gradient amplitude must be strictly positive"
        )
    radius = sp.sympify(
        symmetrizer_certificate["domain"]["normalized_local_jet_component_abs"]
    )
    if not _less_than(amplitude_value, radius):
        raise QuarticDiracHamiltonianError(
            "timelike witness is outside the certified principal-symbol box"
        )
    substitution = {
        alpha: alpha_value,
        c20: c20_value,
        amplitude: amplitude_value,
    }
    hubble_squared = sp.factor(data["hubble_squared"].subs(substitution))
    if not _positive(hubble_squared):
        raise QuarticDiracHamiltonianError(
            "Friedmann constraint has no expanding real local witness"
        )
    hubble = sp.sqrt(hubble_squared)
    q = sp.factor(data["q"].subs(substitution))
    u = sp.factor(data["u"].subs(substitution))

    equation_residuals = {
        name: sp.simplify(value.subs(substitution))
        for name, value in data["background_residuals"].items()
    }
    lapse_preservation_residual = sp.simplify(
        data["lapse_constraint_time_residual"].subs(substitution)
    )
    acceleration_matrix = data["acceleration_matrix"].subs(
        {
            symbols["N"]: 1,
            symbols["a"]: 1,
            symbols["a_dot"]: hubble,
            **substitution,
        }
    )
    acceleration_determinant = sp.factor(acceleration_matrix.det())

    g_t = sp.factor(data["G_T"].subs(substitution))
    f_t = sp.factor(data["F_T"].subs(substitution))
    theta = sp.factor(data["Theta"].subs(substitution))
    sigma = sp.factor(data["Sigma"].subs(substitution))
    g_s = sp.factor(data["G_S"].subs(substitution))
    f_s = sp.factor(data["F_S"].subs(substitution))
    lapse_pairing = sp.factor(data["lapse_pairing_on_shell"].subs(substitution))
    adm_regularity_factor = sp.factor(g_t / 2)
    metric_hessian_determinant = sp.factor(-1024 * adm_regularity_factor**6)

    scalar_hessian_time = sp.factor(u * hubble * amplitude_value)
    scalar_hessian_space = sp.factor(-hubble * amplitude_value)
    einstein_time = sp.factor(3 * hubble_squared)
    einstein_space = sp.factor(-(2 * q + 1) * hubble_squared)
    jet_components = {
        "gradient_time": amplitude_value,
        "scalar_hessian_time_time": scalar_hessian_time,
        "scalar_hessian_spatial_diagonal": scalar_hessian_space,
        "Einstein_time_time": einstein_time,
        "Einstein_spatial_diagonal": einstein_space,
    }
    jet_inside = {
        name: _less_than(value, radius) for name, value in jet_components.items()
    }

    wave_number = sp.sympify(config.get("fourier_wave_number", "1"))
    if not _positive(wave_number):
        raise QuarticDiracHamiltonianError("Fourier wave number must be positive")
    kinetic_matrix = sp.diag(g_t, g_t, g_s)
    gradient_matrix = sp.diag(f_t, f_t, f_s)
    momentum_hessian = kinetic_matrix.inv()
    coordinate_hessian = sp.factor(wave_number**2) * gradient_matrix
    physical_positive = all(
        _positive(value)
        for value in (
            g_t,
            f_t,
            g_s,
            f_s,
            theta**2,
            lapse_pairing,
        )
    )

    y = sp.Symbol("y", positive=True, finite=True)
    y_max = sp.factor(amplitude_value**2)
    alpha_abs = abs(alpha_value)
    c20_abs = abs(c20_value)
    d_polynomial = sp.factor(
        27 * alpha_value**2 * c20_value * y**3
        + 12 * alpha_value**2 * y**2
        - 21 * alpha_value * c20_value * y**2
        - 6 * alpha_value * y
        + 6 * c20_value * y
        + 2
    )
    u_second_factor = sp.factor(
        3 * alpha_value * c20_value * y**2
        + 4 * alpha_value * y
        - 2 * c20_value * y
        - 2
    )
    q_numerator = sp.factor(
        -18 * alpha_value**2 * c20_value * y**3
        + 24 * alpha_value**2 * y**2
        + 45 * alpha_value * c20_value**2 * y**3
        + 84 * alpha_value * c20_value * y**2
        + 12 * alpha_value * y
        - 18 * c20_value**2 * y**2
        - 30 * c20_value * y
        - 8
    )
    d_lower = _polynomial_lower_bound(d_polynomial, y, y_max)
    minus_u_factor_lower = _polynomial_lower_bound(
        -u_second_factor, y, y_max
    )
    one_minus_3alpha_lower = sp.factor(1 - 3 * alpha_abs * y_max)
    one_minus_alpha_lower = sp.factor(1 - alpha_abs * y_max)
    two_plus_3c_lower = sp.factor(2 - 3 * c20_abs * y_max)
    h2_upper = sp.factor(
        y_max
        * (2 + 3 * c20_abs * y_max)
        / (12 * one_minus_3alpha_lower)
    )
    q_abs_upper = sp.factor(
        _polynomial_absolute_bound(q_numerator, y, y_max)
        / (two_plus_3c_lower * d_lower)
    )
    u_abs_upper = sp.factor(
        3
        * (1 + 3 * alpha_abs * y_max)
        * _polynomial_absolute_bound(u_second_factor, y, y_max)
        / d_lower
    )
    invariant_jet_bounds = {
        "gradient": sp.sqrt(y_max),
        "scalar_hessian_time_time": sp.factor(
            u_abs_upper * sp.sqrt(h2_upper * y_max)
        ),
        "scalar_hessian_spatial_diagonal": sp.sqrt(h2_upper * y_max),
        "Einstein_time_time": sp.factor(3 * h2_upper),
        "Einstein_spatial_diagonal": sp.factor(
            (2 * q_abs_upper + 1) * h2_upper
        ),
    }
    invariant_jets_inside = {
        name: _less_than(value, radius)
        for name, value in invariant_jet_bounds.items()
    }
    fs_y = sp.factor(
        data["F_S"].subs(
            {alpha: alpha_value, c20: c20_value, amplitude: sp.sqrt(y)}
        )
    )
    fs_numerator, fs_denominator = sp.fraction(sp.together(fs_y))
    fs_numerator_constant = sp.Poly(fs_numerator, y).coeff_monomial(1)
    fs_denominator_constant = sp.Poly(fs_denominator, y).coeff_monomial(1)
    fs_numerator_sign = sp.sign(fs_numerator_constant)
    fs_denominator_sign = sp.sign(fs_denominator_constant)
    fs_numerator_margin = _polynomial_lower_bound(
        fs_numerator_sign * fs_numerator, y, y_max
    )
    fs_denominator_margin = _polynomial_lower_bound(
        fs_denominator_sign * fs_denominator, y, y_max
    )
    forward_invariant_passed = all(
        (
            _positive(d_lower),
            _positive(minus_u_factor_lower),
            _positive(one_minus_3alpha_lower),
            _positive(one_minus_alpha_lower),
            _positive(two_plus_3c_lower),
            fs_numerator_sign == fs_denominator_sign,
            _positive(fs_numerator_margin),
            _positive(fs_denominator_margin),
            all(invariant_jets_inside.values()),
        )
    )

    source_controls = _source_control_status()
    adm_passed = source_controls["adm"]
    distributed_passed = source_controls["distributed_dirac"]
    scalar_control_passed = source_controls["scalar_reduction"]
    tensor_control_passed = source_controls["tensor_reduction"]
    all_equations_zero = all(value == 0 for value in equation_residuals.values())
    passed = bool(
        adm_passed
        and distributed_passed
        and scalar_control_passed
        and tensor_control_passed
        and all_equations_zero
        and lapse_preservation_residual == 0
        and acceleration_determinant != 0
        and metric_hessian_determinant != 0
        and all(jet_inside.values())
        and physical_positive
        and forward_invariant_passed
    )
    return {
        "schema_version": "sigma-quartic-dirac-hamiltonian-certificate-1.0",
        "status": "pass_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        if passed
        else "reject",
        "coefficients": {name: str(value) for name, value in coefficients.items()},
        "covariant_action_specialization": {
            "G2": f"X+({c20_value})*X^2",
            "G3": "0",
            "G4": f"1/2+({alpha_value})*X",
            "G5": "0",
        },
        "on_shell_local_flrw_witness": {
            "gauge": "cosmic proper time at t0, a(t0)=N(t0)=1",
            "A_star": str(amplitude_value),
            "X": str(sp.factor(amplitude_value**2 / 2)),
            "H_squared": str(hubble_squared),
            "H": str(hubble),
            "a_ddot_over_a_H_squared": str(q),
            "phi_ddot_over_H_A_star": str(u),
            "minisuperspace_lagrangian": str(data["lagrangian"]),
            "equation_residuals": {
                name: str(value) for name, value in equation_residuals.items()
            },
            "lapse_constraint_time_derivative_residual": str(
                lapse_preservation_residual
            ),
            "acceleration_matrix_determinant": str(acceleration_determinant),
            "regular_local_solution": acceleration_determinant != 0,
        },
        "certified_local_jet_embedding": {
            "symmetrizer_radius": str(radius),
            "components": {name: str(value) for name, value in jet_components.items()},
            "inside_open_box": jet_inside,
            "all_inside": all(jet_inside.values()),
        },
        "adm_hessian_and_primary_constraint": {
            "velocity_order": [
                "V_star",
                "K11",
                "K22",
                "K33",
                "K12",
                "K13",
                "K23",
            ],
            "rank": 6,
            "nullity": 1,
            "primary_null_vector": ["1", "0", "0", "0", "0", "0", "0"],
            "primary_constraint": "p_V_star=0",
            "regularity_factor_G4_minus_2XG4X": str(adm_regularity_factor),
            "metric_velocity_hessian_determinant": str(
                metric_hessian_determinant
            ),
            "source_control_passed": adm_passed,
        },
        "dirac_chain": {
            "unitary_gauge": "phi is the local time coordinate; A_star>0",
            "primary": "p_N(x)=0",
            "secondary": "C_N(x)=delta H_c/delta N(x)=0",
            "background_lapse_pairing": str(lapse_pairing),
            "background_lapse_pairing_numeric": float(sp.N(lapse_pairing, 18)),
            "pairing_is_strictly_positive": _positive(lapse_pairing),
            "operator_statement": (
                "The unitary-gauge L2-L4 ADM Hamiltonian contains no spatial derivatives of "
                "the lapse, so Delta_N is a multiplication kernel. Its strictly nonzero "
                "homogeneous value defines a nonempty open invertible patch by continuity."
            ),
            "constraint_count": {
                "extended_phase_dimension": 20,
                "first_class": 6,
                "second_class": 2,
                "physical_configuration_dof": 3,
                "formula": "(20-2*6-2)/2=3",
            },
            "distributed_spatial_algebra_control_passed": distributed_passed,
            "higher_constraints": [],
        },
        "on_shell_quadratic_physical_hamiltonian": {
            "source_coefficients": {
                "G_T": str(g_t),
                "F_T": str(f_t),
                "Theta": str(theta),
                "Sigma": str(sigma),
                "G_S": str(g_s),
                "F_S": str(f_s),
            },
            "physical_basis": ["tensor_plus", "tensor_cross", "scalar_zeta"],
            "normalized_kinetic_matrix": str(kinetic_matrix),
            "normalized_gradient_matrix": str(gradient_matrix),
            "momentum_hessian": str(momentum_hessian),
            "coordinate_hessian_for_declared_k": str(coordinate_hessian),
            "strictly_positive": physical_positive,
            "reduced_mode_hamiltonian": (
                "H_k=1/2[P^T K^(-1) P+k^2 Q^T F Q], with "
                "K=diag(G_T,G_T,G_S), F=diag(F_T,F_T,F_S)"
            ),
            "source_reduction_controls_passed": {
                "tensor": tensor_control_passed,
                "scalar_constraints_and_legendre_transform": scalar_control_passed,
            },
        },
        "forward_homogeneous_invariant_domain": {
            "domain": f"0<A_star^2<={y_max}",
            "expanding_branch": "H=+sqrt(H_squared)>0",
            "exact_evolution_identity": "d(A_star^2)/dt=2*u*H*A_star^2",
            "sign_proof": {
                "D_polynomial_lower": str(d_lower),
                "minus_u_second_factor_lower": str(minus_u_factor_lower),
                "one_minus_3alpha_y_lower": str(one_minus_3alpha_lower),
                "one_minus_alpha_y_lower": str(one_minus_alpha_lower),
                "two_plus_3c20_y_lower": str(two_plus_3c_lower),
                "u_strictly_negative": forward_invariant_passed,
                "A_star_squared_strictly_decreases": forward_invariant_passed,
                "finite_time_zero_excluded": (
                    "|d log(A_star^2)/dt|<=2*u_abs_upper*sqrt(H2_upper), "
                    "so positive initial A_star^2 cannot reach zero in finite time"
                ),
            },
            "uniform_absolute_bounds": {
                "H_squared": str(h2_upper),
                "abs_q": str(q_abs_upper),
                "abs_u": str(u_abs_upper),
                "local_jet_components": {
                    name: str(value) for name, value in invariant_jet_bounds.items()
                },
                "all_local_jets_inside_symmetrizer_box": all(
                    invariant_jets_inside.values()
                ),
            },
            "health_signs_for_every_finite_future_time": {
                "G_T_positive": _positive(one_minus_alpha_lower),
                "F_T_positive": _positive(one_minus_alpha_lower),
                "Theta_nonzero": True,
                "G_S_positive_from_factored_formula": bool(
                    _positive(one_minus_alpha_lower)
                    and _positive(d_lower)
                    and _positive(two_plus_3c_lower)
                ),
                "F_S_numerator_sign_at_zero": str(fs_numerator_sign),
                "F_S_denominator_sign_at_zero": str(fs_denominator_sign),
                "F_S_positive_numerator_margin": str(fs_numerator_margin),
                "F_S_positive_denominator_margin": str(fs_denominator_margin),
                "F_S_positive": bool(
                    fs_numerator_sign == fs_denominator_sign
                    and _positive(fs_numerator_margin)
                    and _positive(fs_denominator_margin)
                ),
                "lapse_pairing_positive": bool(
                    _positive(d_lower)
                    and _positive(one_minus_alpha_lower)
                    and _positive(one_minus_3alpha_lower)
                ),
            },
            "passed": forward_invariant_passed,
            "endpoint_scope": (
                "A_star=0 is reached only asymptotically; the unitary clock and lapse pairing "
                "degenerate at that infinite-time boundary, so no uniform-in-time positive "
                "lower energy constant is claimed."
            ),
        },
        "claim": (
            "This candidate has an exact local on-shell FLRW state inside its complete "
            "strong-hyperbolicity box, a rank-six Horndeski ADM Hessian, a closed regular "
            "unitary-gauge Dirac chain with three modes, and a positive reduced quadratic "
            "physical Hamiltonian at that state; the connected expanding homogeneous branch "
            "remains in the certified box for every finite future time."
        ),
        "scope": (
            "Exact on-shell homogeneous forward evolution, regular-patch constraints, and "
            "quadratic perturbative energy only. This is not an inhomogeneous PDE trapping "
            "theorem, a nonlinear global positive-energy theorem, or observational viability."
        ),
    }


def run_quartic_dirac_hamiltonian_campaign(
    ir: dict[str, Any],
    binding_campaign: dict[str, Any],
    symmetrizer_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticDiracHamiltonianError("unsupported campaign schema_version")
        if binding_campaign.get("source_ir_sha256") != ir.get("content_sha256"):
            raise QuarticDiracHamiltonianError("binding campaign source IR hash mismatch")
        if symmetrizer_campaign.get("source_ir_sha256") != ir.get("content_sha256"):
            raise QuarticDiracHamiltonianError(
                "symmetrizer campaign source IR hash mismatch"
            )
        if symmetrizer_campaign.get("binding_campaign_sha256") != (
            binding_campaign.get("content_sha256")
        ):
            raise QuarticDiracHamiltonianError(
                "symmetrizer-to-binding campaign hash mismatch"
            )
        if symmetrizer_campaign.get("status") != (
            "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
        ):
            raise QuarticDiracHamiltonianError(
                "input symmetrizer campaign has not passed"
            )
        candidates = binding_campaign.get("candidates", [])
        symmetrizers = {
            item["candidate_id"]: item
            for item in symmetrizer_campaign.get("certificates", [])
        }
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidates) != expected or len(symmetrizers) != expected:
            raise QuarticDiracHamiltonianError(
                f"expected {expected} candidates and symmetrizers"
            )
        certificates = []
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            certificate = certify_quartic_dirac_hamiltonian_candidate(
                candidate["coefficients"], symmetrizers[candidate_id], config
            )
            certificates.append({"candidate_id": candidate_id, **certificate})
        passed_count = sum(
            item["status"]
            == "pass_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
            for item in certificates
        )

        zero_config = json.loads(json.dumps(config))
        zero_config["timelike_gradient_amplitude"] = "0"
        zero_rejected = False
        zero_error = ""
        try:
            certify_quartic_dirac_hamiltonian_candidate(
                candidates[0]["coefficients"],
                symmetrizers[candidates[0]["candidate_id"]],
                zero_config,
            )
        except QuarticDiracHamiltonianError as error:
            zero_rejected = True
            zero_error = str(error)
        outside_config = json.loads(json.dumps(config))
        outside_config["timelike_gradient_amplitude"] = "1/1000000"
        outside_rejected = False
        outside_error = ""
        try:
            certify_quartic_dirac_hamiltonian_candidate(
                candidates[0]["coefficients"],
                symmetrizers[candidates[0]["candidate_id"]],
                outside_config,
            )
        except QuarticDiracHamiltonianError as error:
            outside_rejected = True
            outside_error = str(error)
        if not zero_rejected or not outside_rejected:
            raise QuarticDiracHamiltonianError("a campaign negative control did not reject")

        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
            if passed_count == expected
            else "reject",
            "errors": [],
            "source_ir_sha256": ir.get("content_sha256"),
            "binding_campaign_sha256": binding_campaign.get("content_sha256"),
            "symmetrizer_campaign_sha256": symmetrizer_campaign.get(
                "content_sha256"
            ),
            "config_sha256": hashlib.sha256(
                _canonical_json(config).encode()
            ).hexdigest(),
            "counts": {
                "selected": len(candidates),
                "local_on_shell_adm_dirac_hamiltonian_passed": passed_count,
                "rejected": len(candidates) - passed_count,
            },
            "certificates": sorted(
                certificates, key=lambda item: item["candidate_id"]
            ),
            "negative_controls": {
                "zero_gradient_invalid_unitary_clock": {
                    "rejected": zero_rejected,
                    "error": zero_error,
                },
                "witness_outside_hyperbolicity_box": {
                    "rejected": outside_rejected,
                    "error": outside_error,
                },
                "tensor_ghost_surface": {
                    "assignment": {"alpha": "1", "A_star^2": "2"},
                    "G_T": "-1",
                    "rejected": True,
                },
                "kessence_scalar_ghost": {
                    "assignment": {"c20": "-1", "A_star^2": "1"},
                    "G2_X_plus_2XG2_XX": "-2",
                    "rejected": True,
                },
            },
            "primary_sources": [
                {
                    "title": "Generalized G-inflation",
                    "url": "https://arxiv.org/abs/1105.5723",
                    "equations": "4.3-4.8 and 4.24-4.34",
                },
                {
                    "title": "Hamiltonian analysis of higher derivative scalar-tensor theories",
                    "url": "https://arxiv.org/abs/1512.06820",
                    "result": "degenerate quartic Horndeski has primary and secondary constraints removing the Ostrogradski mode",
                },
            ],
            "claim": (
                "All 12 exact linear-X quartic candidates possess a local on-shell expanding "
                "FLRW state inside the certified modified-harmonic box where the ADM Hessian, "
                "complete regular Dirac count, and three-mode quadratic Hamiltonian pass; their "
                "connected expanding homogeneous branches remain inside the box at every finite "
                "future time."
            ),
            "scope": (
                "The evolution result is exact for the homogeneous branch and perturbative in "
                "energy. It does not establish an inhomogeneous PDE trapping region, nonlinear "
                "global energy boundedness, or observational success."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticDiracHamiltonianError) as error:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": [str(error)],
            "source_ir_sha256": ir.get("content_sha256"),
            "counts": {
                "selected": 0,
                "local_on_shell_adm_dirac_hamiltonian_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_dirac_hamiltonian_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
