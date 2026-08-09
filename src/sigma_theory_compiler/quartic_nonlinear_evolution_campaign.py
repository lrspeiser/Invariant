from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    build_quartic_horndeski_x2_kessence_modified_harmonic_symbol,
)
from .quartic_geometric_jet_campaign import (
    DIMENSION,
    FIELD_COUNT,
    SYMMETRIC_METRIC_PAIRS,
    _coordinate_state,
    state_to_covariant_geometry,
)

SCHEMA_VERSION = "sigma-quartic-nonlinear-evolution-campaign-1.0"


class QuarticNonlinearEvolutionError(ValueError):
    """Raised when the nonlinear quartic evolution adapter cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _zero_tensor(shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [sp.Integer(0) for _ in range(shape[0])]
    return [_zero_tensor(shape[1:]) for _ in range(shape[0])]


def _factor_tensor(value: Any) -> Any:
    if isinstance(value, list):
        return [_factor_tensor(item) for item in value]
    return sp.factor(sp.simplify(value))


def quartic_action_euler_tensors(
    geometry: dict[str, Any],
    *,
    m2: sp.Expr,
    alpha: sp.Expr,
    c20: sp.Expr,
    include_riemann_gradient: bool = True,
) -> dict[str, Any]:
    """Return the exact G2=X+c20 X^2, G4=m2/2+alpha X Euler tensors.

    The metric coefficient is the variation with respect to the inverse metric.
    It is the linear-X specialization of KYY equation B.4; the scalar coefficient
    combines the G2 current with ``-2 alpha G^{mu nu} H_mu_nu``.
    """

    metric: sp.Matrix = geometry["metric"]
    inverse: sp.Matrix = geometry["inverse_metric"]
    p_down = sp.Matrix(geometry["scalar_gradient"])
    p_up = inverse * p_down
    hessian = geometry["scalar_hessian"]
    ricci = geometry["ricci"]
    einstein = geometry["einstein"]
    curvature = geometry["scalar_curvature"]
    riemann_up = geometry["riemann_up"]

    x_scalar = -sum(p_down[index] * p_up[index] for index in range(DIMENSION)) / 2
    theta = sum(
        inverse[left, right] * hessian[left][right]
        for left in range(DIMENSION)
        for right in range(DIMENSION)
    )
    hessian_squared = sum(
        inverse[left, upper]
        * inverse[right, lower]
        * hessian[left][right]
        * hessian[upper][lower]
        for left in range(DIMENSION)
        for right in range(DIMENSION)
        for upper in range(DIMENSION)
        for lower in range(DIMENSION)
    )
    hessian_difference = theta**2 - hessian_squared
    ricci_pp = sum(
        p_up[left] * p_up[right] * ricci[left][right]
        for left in range(DIMENSION)
        for right in range(DIMENSION)
    )
    function = m2 / 2 + alpha * x_scalar
    g2 = x_scalar + c20 * x_scalar**2
    g2_x = 1 + 2 * c20 * x_scalar

    quartic_metric = _zero_tensor((DIMENSION, DIMENSION))
    g2_metric = _zero_tensor((DIMENSION, DIMENSION))
    for mu in range(DIMENSION):
        for nu in range(DIMENSION):
            hessian_product = sum(
                inverse[left, right]
                * hessian[left][mu]
                * hessian[right][nu]
                for left in range(DIMENSION)
                for right in range(DIMENSION)
            )
            ricci_gradient = sum(
                p_up[index]
                * (
                    ricci[index][mu] * p_down[nu]
                    + ricci[index][nu] * p_down[mu]
                )
                for index in range(DIMENSION)
            )
            riemann_gradient = sum(
                p_up[first]
                * p_up[second]
                * sum(
                    metric[mu, raised]
                    * riemann_up[raised][first][nu][second]
                    for raised in range(DIMENSION)
                )
                for first in range(DIMENSION)
                for second in range(DIMENSION)
            )
            quartic_metric[mu][nu] = (
                function * einstein[mu][nu]
                - alpha * curvature * p_down[mu] * p_down[nu] / 2
                - alpha * theta * hessian[mu][nu]
                + alpha * hessian_product
                + metric[mu, nu] * alpha * hessian_difference / 2
                + alpha * ricci_gradient
                - metric[mu, nu] * alpha * ricci_pp
                + (alpha * riemann_gradient if include_riemann_gradient else 0)
            )
            g2_metric[mu][nu] = -(
                metric[mu, nu] * g2 + g2_x * p_down[mu] * p_down[nu]
            ) / 2

    einstein_upper = inverse * sp.Matrix(einstein) * inverse
    hessian_matrix = sp.Matrix(hessian)
    scalar_euler = sum(
        (
            g2_x * inverse[mu, nu]
            - 2 * c20 * p_up[mu] * p_up[nu]
            - 2 * alpha * einstein_upper[mu, nu]
        )
        * hessian_matrix[mu, nu]
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    )
    total_metric = [
        [quartic_metric[mu][nu] + g2_metric[mu][nu] for nu in range(DIMENSION)]
        for mu in range(DIMENSION)
    ]
    return {
        "x": sp.factor(x_scalar),
        "G2": sp.factor(g2),
        "G2_X": sp.factor(g2_x),
        "theta": sp.factor(theta),
        "hessian_squared": sp.factor(hessian_squared),
        "quartic_metric_lower": _factor_tensor(quartic_metric),
        "g2_metric_lower": _factor_tensor(g2_metric),
        "metric_euler_lower": _factor_tensor(total_metric),
        "scalar_euler": sp.factor(sp.simplify(scalar_euler)),
    }


def _projector(
    lower_index: int,
    derivative_index: int,
    first_metric_index: int,
    second_metric_index: int,
    inverse_metric: sp.Matrix,
) -> sp.Expr:
    return (
        int(lower_index == first_metric_index)
        * inverse_metric[second_metric_index, derivative_index]
        + int(lower_index == second_metric_index)
        * inverse_metric[first_metric_index, derivative_index]
        - int(lower_index == derivative_index)
        * inverse_metric[first_metric_index, second_metric_index]
    ) / 2


def modified_harmonic_gauge_tensor(
    geometry: dict[str, Any],
    *,
    m2: sp.Expr,
    tilde_inverse_metric: sp.Matrix,
    hat_inverse_metric: sp.Matrix,
    tilde_inverse_first: Sequence[Sequence[Sequence[sp.Expr]]] | None = None,
    reference_connection: Sequence[Sequence[Sequence[sp.Expr]]] | None = None,
    reference_connection_first: Sequence[
        Sequence[Sequence[Sequence[sp.Expr]]]
    ]
    | None = None,
    gauge_source_lower: Sequence[sp.Expr] | None = None,
    gauge_source_first: Sequence[Sequence[sp.Expr]] | None = None,
) -> dict[str, Any]:
    """Evaluate the covariant modified-harmonic gauge completion.

    ``Delta Gamma=Gamma-barGamma`` is a tensor.  The prescribed auxiliary inverse
    metrics, reference connection, and gauge source are nondynamical formulation
    fields; setting the latter two to zero is valid in a Cartesian reference chart.
    """

    if tilde_inverse_metric.shape != (DIMENSION, DIMENSION) or hat_inverse_metric.shape != (
        DIMENSION,
        DIMENSION,
    ):
        raise QuarticNonlinearEvolutionError("auxiliary inverse metrics must be 4 by 4")
    tilde_first = (
        tilde_inverse_first
        if tilde_inverse_first is not None
        else _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    )
    reference = (
        reference_connection
        if reference_connection is not None
        else _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    )
    reference_first = (
        reference_connection_first
        if reference_connection_first is not None
        else _zero_tensor((DIMENSION, DIMENSION, DIMENSION, DIMENSION))
    )
    source = list(gauge_source_lower) if gauge_source_lower is not None else [sp.Integer(0)] * 4
    source_first = (
        gauge_source_first
        if gauge_source_first is not None
        else _zero_tensor((DIMENSION, DIMENSION))
    )
    metric: sp.Matrix = geometry["metric"]
    inverse: sp.Matrix = geometry["inverse_metric"]
    metric_first = geometry["metric_first"]
    connection = geometry["connection"]
    connection_first = geometry["connection_first"]
    delta_up = _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    delta_lower = _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    delta_lower_first = _zero_tensor((DIMENSION, DIMENSION, DIMENSION, DIMENSION))
    for upper in range(DIMENSION):
        for left in range(DIMENSION):
            for right in range(DIMENSION):
                delta_up[upper][left][right] = (
                    connection[upper][left][right] - reference[upper][left][right]
                )
    for lower in range(DIMENSION):
        for left in range(DIMENSION):
            for right in range(DIMENSION):
                delta_lower[lower][left][right] = sum(
                    metric[lower, upper] * delta_up[upper][left][right]
                    for upper in range(DIMENSION)
                )
                for derivative in range(DIMENSION):
                    delta_lower_first[derivative][lower][left][right] = sum(
                        metric_first[derivative][lower][upper]
                        * delta_up[upper][left][right]
                        + metric[lower, upper]
                        * (
                            connection_first[derivative][upper][left][right]
                            - reference_first[derivative][upper][left][right]
                        )
                        for upper in range(DIMENSION)
                    )
    constraint = [
        sum(
            tilde_inverse_metric[left, right] * delta_lower[lower][left][right]
            for left in range(DIMENSION)
            for right in range(DIMENSION)
        )
        - source[lower]
        for lower in range(DIMENSION)
    ]
    constraint_first = _zero_tensor((DIMENSION, DIMENSION))
    constraint_covariant_first = _zero_tensor((DIMENSION, DIMENSION))
    for derivative in range(DIMENSION):
        for lower in range(DIMENSION):
            constraint_first[derivative][lower] = sum(
                tilde_first[derivative][left][right]
                * delta_lower[lower][left][right]
                + tilde_inverse_metric[left, right]
                * delta_lower_first[derivative][lower][left][right]
                for left in range(DIMENSION)
                for right in range(DIMENSION)
            ) - source_first[derivative][lower]
            constraint_covariant_first[derivative][lower] = (
                constraint_first[derivative][lower]
                - sum(
                    connection[upper][derivative][lower] * constraint[upper]
                    for upper in range(DIMENSION)
                )
            )
    gauge_upper = _zero_tensor((DIMENSION, DIMENSION))
    for mu in range(DIMENSION):
        for nu in range(DIMENSION):
            gauge_upper[mu][nu] = -m2 / 2 * sum(
                _projector(alpha, derivative, mu, nu, hat_inverse_metric)
                * inverse[alpha, lower]
                * constraint_covariant_first[derivative][lower]
                for alpha in range(DIMENSION)
                for derivative in range(DIMENSION)
                for lower in range(DIMENSION)
            )
    return {
        "connection_difference_up": _factor_tensor(delta_up),
        "constraint_lower": _factor_tensor(constraint),
        "constraint_covariant_first": _factor_tensor(constraint_covariant_first),
        "metric_euler_upper": _factor_tensor(gauge_upper),
        "formulation_fields": {
            "tilde_inverse_metric": "prescribed smooth Lorentzian contravariant tensor",
            "hat_inverse_metric": "prescribed smooth Lorentzian contravariant tensor",
            "reference_connection": "prescribed torsion-free affine connection",
            "gauge_source_lower": "prescribed covector field",
            "dynamical": False,
        },
    }


def _assemble_equations(
    geometry: dict[str, Any],
    action: dict[str, Any],
    gauge_upper: sp.Matrix,
) -> tuple[sp.Matrix, sp.Matrix]:
    inverse: sp.Matrix = geometry["inverse_metric"]
    action_upper = inverse * sp.Matrix(action["metric_euler_lower"]) * inverse
    total_upper = action_upper + gauge_upper
    metric_equations = []
    for left, right in SYMMETRIC_METRIC_PAIRS:
        if left == right:
            metric_equations.append(sp.factor(total_upper[left, right]))
        else:
            metric_equations.append(sp.factor(sp.sqrt(2) * total_upper[left, right]))
    # The symmetric 11-field variational convention uses -E_phi as the scalar
    # row; with this sign its mixed principal block is the transpose of E_g.
    equations = sp.Matrix([*metric_equations, -action["scalar_euler"]])
    return total_upper.applyfunc(sp.factor), equations.applyfunc(sp.factor)


def gauge_fixed_euler_from_state(
    state: Sequence[sp.Expr],
    state_derivative: Sequence[Sequence[sp.Expr]],
    *,
    m2: sp.Expr,
    alpha: sp.Expr,
    c20: sp.Expr,
    tilde_inverse_metric: sp.Matrix,
    hat_inverse_metric: sp.Matrix,
    include_gauge: bool = True,
    include_riemann_gradient: bool = True,
) -> dict[str, Any]:
    geometry = state_to_covariant_geometry(state, state_derivative)
    action = quartic_action_euler_tensors(
        geometry,
        m2=m2,
        alpha=alpha,
        c20=c20,
        include_riemann_gradient=include_riemann_gradient,
    )
    gauge = modified_harmonic_gauge_tensor(
        geometry,
        m2=m2,
        tilde_inverse_metric=tilde_inverse_metric,
        hat_inverse_metric=hat_inverse_metric,
    )
    total_upper, equations = _assemble_equations(
        geometry,
        action,
        sp.Matrix(gauge["metric_euler_upper"]) if include_gauge else sp.zeros(4)
    )
    return {
        "geometry": geometry,
        "action": action,
        "gauge": gauge,
        "metric_euler_upper": total_upper,
        "equations": equations,
    }


def _gradient_index(derivative: int, field: int) -> int:
    return 11 + field if derivative == 0 else 22 + 11 * (derivative - 1) + field


def _exact_local_witness() -> tuple[list[sp.Expr], list[list[sp.Expr]], tuple[sp.Symbol, ...]]:
    epsilon = sp.Rational(1, 10**12)
    state = [sp.Integer(0)] * 55
    for index, value in {0: -1, 4: 1, 7: 1, 9: 1}.items():
        state[index] = sp.Integer(value)
    for derivative in range(DIMENSION):
        state[_gradient_index(derivative, 10)] = (derivative + 1) * epsilon
    state_derivative = [[sp.Integer(0)] * 55 for _ in range(DIMENSION)]
    for derivative in range(DIMENSION):
        for field in range(FIELD_COUNT):
            state_derivative[derivative][field] = state[
                _gradient_index(derivative, field)
            ]
    accelerations = sp.symbols("Y_0:11", real=True)
    for field in range(FIELD_COUNT):
        for left in range(DIMENSION):
            for right in range(left, DIMENSION):
                if left == 0 and right == 0:
                    value = accelerations[field]
                else:
                    numerator = ((3 * field + 5 * left + 7 * right + 1) % 9) - 4
                    value = numerator * epsilon
                state_derivative[left][_gradient_index(right, field)] = value
                state_derivative[right][_gradient_index(left, field)] = value
    return state, state_derivative, accelerations


@cache
def _nonlinear_witness_data() -> dict[str, Any]:
    data = build_quartic_horndeski_x2_kessence_modified_harmonic_symbol()
    state, state_derivative, accelerations = _exact_local_witness()
    tilde = sp.diag(-4, 1, 1, 1)
    hat = sp.diag(-9, 1, 1, 1)
    result = gauge_fixed_euler_from_state(
        state,
        state_derivative,
        m2=data["m2"],
        alpha=data["alpha"],
        c20=data["c20"],
        tilde_inverse_metric=tilde,
        hat_inverse_metric=hat,
    )
    equations = result["equations"]
    acceleration_vector = sp.Matrix(accelerations)
    zero_acceleration = {symbol: 0 for symbol in accelerations}
    acceleration_matrix = equations.jacobian(accelerations).applyfunc(sp.factor)
    source = equations.subs(zero_acceleration).applyfunc(sp.factor)
    affine_residual = (
        equations - acceleration_matrix * acceleration_vector - source
    ).applyfunc(sp.factor)

    geometry = result["geometry"]
    inverse: sp.Matrix = geometry["inverse_metric"]
    einstein_upper = inverse * sp.Matrix(geometry["einstein"]) * inverse
    substitutions: dict[sp.Expr, sp.Expr] = {
        data["xi_lower"][0]: 1,
        data["xi_lower"][1]: 0,
        data["xi_lower"][2]: 0,
        data["xi_lower"][3]: 0,
    }
    for index in range(DIMENSION):
        substitutions[data["gradient_lower"][index]] = geometry["scalar_gradient"][
            index
        ].subs(zero_acceleration)
    for left, right in SYMMETRIC_METRIC_PAIRS:
        substitutions[data["hessian_lower"][left, right]] = geometry[
            "scalar_hessian"
        ][left][right].subs(zero_acceleration)
        substitutions[data["einstein_upper"][left, right]] = einstein_upper[
            left, right
        ].subs(zero_acceleration)
    expected_time_block = data["full_symbol"].subs(substitutions).applyfunc(sp.factor)
    principal_residual = (acceleration_matrix - expected_time_block).applyfunc(
        sp.factor
    )

    no_gauge_upper, no_gauge_equations = _assemble_equations(
        geometry, result["action"], sp.zeros(DIMENSION)
    )
    del no_gauge_upper
    no_gauge_matrix = no_gauge_equations.jacobian(accelerations).applyfunc(sp.factor)
    no_gauge_residual = (no_gauge_matrix - expected_time_block).applyfunc(sp.factor)

    omitted_riemann_action = quartic_action_euler_tensors(
        geometry,
        m2=data["m2"],
        alpha=data["alpha"],
        c20=data["c20"],
        include_riemann_gradient=False,
    )
    _, omitted_riemann_equations = _assemble_equations(
        geometry,
        omitted_riemann_action,
        sp.Matrix(result["gauge"]["metric_euler_upper"]),
    )
    omitted_riemann_matrix = omitted_riemann_equations.jacobian(
        accelerations
    ).applyfunc(sp.factor)
    omitted_riemann_residual = (
        omitted_riemann_matrix - expected_time_block
    ).applyfunc(sp.factor)

    canonical_action = quartic_action_euler_tensors(
        geometry, m2=data["m2"], alpha=0, c20=0
    )
    p_down = sp.Matrix(geometry["scalar_gradient"])
    canonical_expected = [
        [
            data["m2"] * geometry["einstein"][mu][nu] / 2
            - (
                geometry["metric"][mu, nu] * canonical_action["x"]
                + p_down[mu] * p_down[nu]
            )
            / 2
            for nu in range(DIMENSION)
        ]
        for mu in range(DIMENSION)
    ]
    canonical_metric_residual = (
        sp.Matrix(canonical_action["metric_euler_lower"])
        - sp.Matrix(canonical_expected)
    ).applyfunc(sp.factor)
    canonical_scalar_residual = sp.factor(
        canonical_action["scalar_euler"] - canonical_action["theta"]
    )

    pure_gr_geometry = dict(geometry)
    pure_gr_geometry["scalar_gradient"] = [sp.Integer(0)] * DIMENSION
    pure_gr_geometry["scalar_hessian"] = _zero_tensor((DIMENSION, DIMENSION))
    pure_gr_action = quartic_action_euler_tensors(
        pure_gr_geometry, m2=data["m2"], alpha=0, c20=0
    )
    pure_gr_metric_residual = (
        sp.Matrix(pure_gr_action["metric_euler_lower"])
        - data["m2"] * sp.Matrix(geometry["einstein"]) / 2
    ).applyfunc(sp.factor)

    collapse = {
        data["m2"]: 2,
        data["alpha"]: 1,
        data["c20"]: 0,
        **{item: 0 for item in data["gradient_lower"]},
        **{item: 0 for item in data["hessian_lower"].free_symbols},
        **{item: 0 for item in data["einstein_upper"].free_symbols},
        data["einstein_upper"][0, 0]: -sp.Rational(1, 2),
    }
    collapse_a = data["first_order"]["A"].subs(collapse)
    return {
        "symbol_data": data,
        "state": state,
        "state_derivative": state_derivative,
        "accelerations": accelerations,
        "geometry": geometry,
        "equations": equations,
        "acceleration_matrix": acceleration_matrix,
        "source": source,
        "affine_residual": affine_residual,
        "expected_time_block": expected_time_block,
        "principal_residual": principal_residual,
        "no_gauge_residual": no_gauge_residual,
        "omitted_riemann_residual": omitted_riemann_residual,
        "canonical_metric_residual": canonical_metric_residual,
        "canonical_scalar_residual": canonical_scalar_residual,
        "pure_gr_metric_residual": pure_gr_metric_residual,
        "pure_gr_scalar_residual": sp.factor(pure_gr_action["scalar_euler"]),
        "collapse_a_determinant": sp.factor(collapse_a.det()),
        "collapse_a_rank": collapse_a.rank(),
    }


@cache
def nonlinear_evolution_source_control() -> tuple[bool, dict[str, Any]]:
    witness = _nonlinear_witness_data()
    data = witness["symbol_data"]
    sample = {data["m2"]: 1, data["alpha"]: sp.Rational(1, 2), data["c20"]: 1}
    matrix = witness["acceleration_matrix"].subs(sample)
    source = witness["source"].subs(sample)
    solution = matrix.inv() * (-source)
    solution_residual = (matrix * solution + source).applyfunc(sp.factor)
    nonzero_source = any(item != 0 for item in source)

    time, radius, angle, height = sp.symbols("t r theta z", real=True)
    coordinates = (time, radius, angle, height)
    cylindrical_state = _coordinate_state(
        sp.diag(-1, 1, radius**2, 1), radius, coordinates
    )
    cylindrical_geometry = state_to_covariant_geometry(*cylindrical_state)
    reference_gauge = modified_harmonic_gauge_tensor(
        cylindrical_geometry,
        m2=1,
        tilde_inverse_metric=cylindrical_geometry["inverse_metric"],
        hat_inverse_metric=cylindrical_geometry["inverse_metric"],
        tilde_inverse_first=cylindrical_geometry["inverse_metric_first"],
        reference_connection=cylindrical_geometry["connection"],
        reference_connection_first=cylindrical_geometry["connection_first"],
    )
    omitted_reference_gauge = modified_harmonic_gauge_tensor(
        cylindrical_geometry,
        m2=1,
        tilde_inverse_metric=cylindrical_geometry["inverse_metric"],
        hat_inverse_metric=cylindrical_geometry["inverse_metric"],
        tilde_inverse_first=cylindrical_geometry["inverse_metric_first"],
    )
    reference_constraint_zero = all(
        item == 0 for item in reference_gauge["constraint_lower"]
    )
    reference_gauge_zero = all(
        item == 0
        for row in reference_gauge["metric_euler_upper"]
        for item in row
    )
    omitted_reference_nonzero = any(
        item != 0
        for row in omitted_reference_gauge["metric_euler_upper"]
        for item in row
    )
    passed = bool(
        witness["affine_residual"].is_zero_matrix
        and witness["principal_residual"].is_zero_matrix
        and not witness["no_gauge_residual"].is_zero_matrix
        and not witness["omitted_riemann_residual"].is_zero_matrix
        and matrix.det() != 0
        and solution_residual.is_zero_matrix
        and nonzero_source
        and witness["collapse_a_determinant"] == 0
        and witness["collapse_a_rank"] == 10
        and witness["canonical_metric_residual"].is_zero_matrix
        and witness["canonical_scalar_residual"] == 0
        and witness["pure_gr_metric_residual"].is_zero_matrix
        and witness["pure_gr_scalar_residual"] == 0
        and reference_constraint_zero
        and reference_gauge_zero
        and omitted_reference_nonzero
    )
    formula_contract = {
        "action": "G2=X+c20 X^2; G4=M2/2+alpha X; G3=G5=0",
        "metric_euler": "linear-X specialization of KYY 2011 equation B.4 plus G2 Hilbert tensor",
        "scalar_euler": "nabla_mu(G2_X nabla^mu phi)-2 alpha G^mu_nu nabla_mu nabla_nu phi",
        "gauge_constraint": "C_beta=tilde_g^rho_sigma(Delta Gamma)_beta_rho_sigma-H_beta",
        "gauge_completion": "-M2/2 hat_P_alpha^(gamma mu nu) g^(alpha beta) nabla_gamma C_beta",
        "evolution": "A_AB partial_0^2 q_B + S_A=0; partial_0^2 q=-A^{-1}S",
    }
    return passed, {
        "control": "exact gauge-fixed nonlinear quartic time-acceleration elimination",
        "formula_contract": formula_contract,
        "formula_contract_sha256": hashlib.sha256(
            _canonical_json(formula_contract).encode()
        ).hexdigest(),
        "equation_count": 11,
        "acceleration_count": 11,
        "time_acceleration_affine_residual_zero": witness[
            "affine_residual"
        ].is_zero_matrix,
        "independent_principal_time_block_residual_zero": witness[
            "principal_residual"
        ].is_zero_matrix,
        "nonzero_lower_order_source": nonzero_source,
        "known_answer_reductions": {
            "alpha_0_c20_0": {
                "theory": "Einstein-Hilbert plus canonical scalar",
                "metric_residual_zero": witness[
                    "canonical_metric_residual"
                ].is_zero_matrix,
                "scalar_box_residual": str(witness["canonical_scalar_residual"]),
            },
            "constant_scalar": {
                "theory": "pure Einstein-Hilbert",
                "metric_residual_zero": witness[
                    "pure_gr_metric_residual"
                ].is_zero_matrix,
                "scalar_residual": str(witness["pure_gr_scalar_residual"]),
            },
        },
        "sample_solution": {
            "coefficients": {"M2": "1", "alpha": "1/2", "c20": "1"},
            "time_block_determinant_nonzero": matrix.det() != 0,
            "solution_residual_zero": solution_residual.is_zero_matrix,
            "maximum_abs_acceleration_numeric": max(
                abs(float(sp.N(item, 17))) for item in solution
            ),
        },
        "formulation_fields": {
            "physical_fields": "g_mu_nu and phi",
            "matter_coupling": "all matter couples only to g_mu_nu; no matter appears in this vacuum source control",
            "tilde_inverse_metric": "prescribed smooth Lorentzian contravariant tensor with nested auxiliary cone",
            "hat_inverse_metric": "prescribed smooth Lorentzian contravariant tensor with a second nested auxiliary cone",
            "reference_connection": "prescribed torsion-free affine connection; Delta Gamma is a tensor",
            "gauge_source": "prescribed covector H_beta and first derivative",
        },
        "curvilinear_reference_connection_control": {
            "metric": "diag(-1,1,r^2,1)",
            "physical_connection_nonzero": any(
                item != 0
                for upper in cylindrical_geometry["connection"]
                for row in upper
                for item in row
            ),
            "Delta_Gamma_zero_with_matching_flat_reference": reference_constraint_zero,
            "gauge_completion_zero": reference_gauge_zero,
            "omitted_reference_connection_nonzero": omitted_reference_nonzero,
        },
        "negative_controls": {
            "omit_modified_harmonic_gauge_completion": {
                "nonzero_matrix_entries": sum(
                    item != 0 for item in witness["no_gauge_residual"]
                ),
                "rejected": not witness["no_gauge_residual"].is_zero_matrix,
            },
            "omit_riemann_gradient_metric_term": {
                "nonzero_matrix_entries": sum(
                    item != 0 for item in witness["omitted_riemann_residual"]
                ),
                "rejected": not witness["omitted_riemann_residual"].is_zero_matrix,
            },
            "singular_time_block": {
                "witness": "M2=2, alpha=1, c20=0, p=H=0, G^00=-1/2",
                "determinant": str(witness["collapse_a_determinant"]),
                "rank": witness["collapse_a_rank"],
                "rejected": witness["collapse_a_determinant"] == 0,
            },
        },
        "passed": passed,
        "scope": (
            "This supplies the exact local gauge-fixed vacuum Euler source and solves all 11 "
            "second-time partials wherever the certified time block is invertible. It does not "
            "yet prove that nonlinear evolution preserves the local-jet box, bound derivatives "
            "of the 55-variable symmetrizer/source, close commuted Sobolev estimates, include "
            "matter sources, establish boundary conditions, or prove a PDE bootstrap."
        ),
    }


def _certify_candidate(
    prerequisite: dict[str, Any], witness: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    coefficients = prerequisite.get("coefficients", {})
    data = witness["symbol_data"]
    substitution = {
        data["m2"]: sp.sympify(coefficients["m2"]),
        data["alpha"]: sp.sympify(coefficients["a10"]),
        data["c20"]: sp.sympify(coefficients["c20"]),
    }
    matrix = witness["acceleration_matrix"].subs(substitution)
    source = witness["source"].subs(substitution)
    if matrix.det() == 0:
        raise QuarticNonlinearEvolutionError("candidate time block is singular at the source witness")
    solution = matrix.inv() * (-source)
    residual = (matrix * solution + source).applyfunc(sp.factor)
    if not residual.is_zero_matrix:
        raise QuarticNonlinearEvolutionError("candidate acceleration solution has nonzero residual")
    bound = sp.Rational(1, 5_000_000_000)
    solved_geometry = {
        "p": witness["geometry"]["scalar_gradient"],
        "H": witness["geometry"]["scalar_hessian"],
        "G": witness["geometry"]["einstein"],
    }
    acceleration_substitution = dict(zip(witness["accelerations"], solution, strict=True))
    jet_values = [
        sp.factor(item.subs(acceleration_substitution))
        for family in solved_geometry.values()
        for row in family
        for item in (row if isinstance(row, list) else [row])
    ]
    maximum_jet = max(abs(float(sp.N(item, 17))) for item in jet_values)
    if maximum_jet >= float(bound):
        raise QuarticNonlinearEvolutionError("solved witness leaves the certified local-jet box")
    return {
        "schema_version": "sigma-quartic-nonlinear-evolution-certificate-1.0",
        "status": "pass_exact_local_nonlinear_time_acceleration_elimination",
        "candidate_id": prerequisite["candidate_id"],
        "coefficients": coefficients,
        "source_geometric_formula_contract_sha256": prerequisite[
            "geometric_formula_contract_sha256"
        ],
        "evolution_formula_contract_sha256": control["formula_contract_sha256"],
        "time_block_determinant_nonzero": True,
        "acceleration_solution_residual_zero": True,
        "nonzero_source": any(item != 0 for item in source),
        "certified_local_jet_bound": str(bound),
        "maximum_solved_jet_component_numeric": maximum_jet,
        "maximum_abs_acceleration_numeric": max(
            abs(float(sp.N(item, 17))) for item in solution
        ),
        "resolved_predecessor_gate": "quartic_gauge_fixed_nonlinear_evolution_source",
        "remaining_gate": "nonlinear_source_symmetrizer_derivative_bounds_and_pde_bootstrap",
        "scope": control["scope"],
    }


def run_quartic_nonlinear_evolution_campaign(
    geometric_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticNonlinearEvolutionError("unsupported campaign schema_version")
        if geometric_campaign.get("status") != (
            "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps"
        ):
            raise QuarticNonlinearEvolutionError("geometric-jet campaign prerequisite failed")
        expected = int(config.get("expected_candidate_count", 12))
        prerequisites = geometric_campaign.get("certificates", [])
        if len(prerequisites) != expected:
            raise QuarticNonlinearEvolutionError("unexpected geometric candidate count")
        control_passed, control = nonlinear_evolution_source_control()
        if not control_passed:
            raise QuarticNonlinearEvolutionError("nonlinear evolution source control failed")
        witness = _nonlinear_witness_data()
        certificates = [
            _certify_candidate(item, witness, control)
            for item in sorted(prerequisites, key=lambda value: value["candidate_id"])
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
            "errors": [],
            "geometric_campaign_sha256": geometric_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "counts": {
                "selected": len(certificates),
                "nonlinear_time_acceleration_eliminations_passed": len(certificates),
                "rejected": 0,
            },
            "nonlinear_evolution_control": control,
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates are bound to the exact gauge-fixed nonlinear vacuum "
                "Euler source and an exact local solution for their 11 second-time partials at "
                "a nonzero source witness inside the certified local-jet box."
            ),
            "scope": control["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticNonlinearEvolutionError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "geometric_campaign_sha256": geometric_campaign.get("content_sha256"),
            "counts": {
                "selected": 0,
                "nonlinear_time_acceleration_eliminations_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_nonlinear_evolution_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
