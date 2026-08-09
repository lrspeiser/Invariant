from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-geometric-jet-campaign-1.0"
DIMENSION = 4
FIELD_COUNT = 11
SYMMETRIC_METRIC_PAIRS = tuple(
    (left, right)
    for left in range(DIMENSION)
    for right in range(left, DIMENSION)
)


class QuarticGeometricJetError(ValueError):
    """Raised when a first-order campaign cannot be bound to the geometric jet map."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _zero_tensor(shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [sp.Integer(0) for _ in range(shape[0])]
    return [_zero_tensor(shape[1:]) for _ in range(shape[0])]


def _simplify_tensor(value: Any) -> Any:
    if isinstance(value, list):
        return [_simplify_tensor(item) for item in value]
    return sp.factor(sp.simplify(value))


def reconstruct_covariant_geometry(
    metric: sp.Matrix,
    metric_first: Sequence[Sequence[Sequence[sp.Expr]]],
    metric_second: Sequence[Sequence[Sequence[Sequence[sp.Expr]]]],
    scalar_first: Sequence[sp.Expr],
    scalar_second: Sequence[Sequence[sp.Expr]],
) -> dict[str, Any]:
    """Reconstruct connection, scalar Hessian, curvature, and Einstein tensor from a 2-jet.

    The inputs are coordinate partial derivatives.  No normal-coordinate assumption is made,
    so all connection and connection-squared lower-order terms remain present.
    """

    if metric.shape != (DIMENSION, DIMENSION):
        raise QuarticGeometricJetError("the geometric map requires a four-dimensional metric")
    inverse = metric.inv().applyfunc(sp.factor)
    inverse_first = _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    for derivative in range(DIMENSION):
        for upper in range(DIMENSION):
            for right in range(DIMENSION):
                inverse_first[derivative][upper][right] = -sum(
                    inverse[upper, left]
                    * metric_first[derivative][left][lower]
                    * inverse[lower, right]
                    for left in range(DIMENSION)
                    for lower in range(DIMENSION)
                )

    connection = _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    connection_first = _zero_tensor((DIMENSION, DIMENSION, DIMENSION, DIMENSION))
    for upper in range(DIMENSION):
        for left in range(DIMENSION):
            for right in range(DIMENSION):
                metric_bracket = [
                    metric_first[left][contracted][right]
                    + metric_first[right][contracted][left]
                    - metric_first[contracted][left][right]
                    for contracted in range(DIMENSION)
                ]
                connection[upper][left][right] = sum(
                    inverse[upper, contracted] * metric_bracket[contracted]
                    for contracted in range(DIMENSION)
                ) / 2
                for derivative in range(DIMENSION):
                    bracket_first = [
                        metric_second[derivative][left][contracted][right]
                        + metric_second[derivative][right][contracted][left]
                        - metric_second[derivative][contracted][left][right]
                        for contracted in range(DIMENSION)
                    ]
                    connection_first[derivative][upper][left][right] = sum(
                        inverse_first[derivative][upper][contracted]
                        * metric_bracket[contracted]
                        + inverse[upper, contracted] * bracket_first[contracted]
                        for contracted in range(DIMENSION)
                    ) / 2

    hessian = _zero_tensor((DIMENSION, DIMENSION))
    for left in range(DIMENSION):
        for right in range(DIMENSION):
            hessian[left][right] = scalar_second[left][right] - sum(
                connection[upper][left][right] * scalar_first[upper]
                for upper in range(DIMENSION)
            )

    riemann_up = _zero_tensor((DIMENSION, DIMENSION, DIMENSION, DIMENSION))
    for upper in range(DIMENSION):
        for lowered in range(DIMENSION):
            for left in range(DIMENSION):
                for right in range(DIMENSION):
                    riemann_up[upper][lowered][left][right] = (
                        connection_first[left][upper][right][lowered]
                        - connection_first[right][upper][left][lowered]
                        + sum(
                            connection[upper][left][contracted]
                            * connection[contracted][right][lowered]
                            - connection[upper][right][contracted]
                            * connection[contracted][left][lowered]
                            for contracted in range(DIMENSION)
                        )
                    )

    ricci = _zero_tensor((DIMENSION, DIMENSION))
    for left in range(DIMENSION):
        for right in range(DIMENSION):
            ricci[left][right] = sum(
                riemann_up[upper][left][upper][right]
                for upper in range(DIMENSION)
            )
    scalar_curvature = sum(
        inverse[left, right] * ricci[left][right]
        for left in range(DIMENSION)
        for right in range(DIMENSION)
    )
    einstein = _zero_tensor((DIMENSION, DIMENSION))
    for left in range(DIMENSION):
        for right in range(DIMENSION):
            einstein[left][right] = (
                ricci[left][right] - metric[left, right] * scalar_curvature / 2
            )

    return {
        "inverse_metric": inverse,
        "inverse_metric_first": _simplify_tensor(inverse_first),
        "connection": _simplify_tensor(connection),
        "connection_first": _simplify_tensor(connection_first),
        "scalar_gradient": list(scalar_first),
        "scalar_hessian": _simplify_tensor(hessian),
        "riemann_up": _simplify_tensor(riemann_up),
        "ricci": _simplify_tensor(ricci),
        "scalar_curvature": sp.factor(sp.simplify(scalar_curvature)),
        "einstein": _simplify_tensor(einstein),
    }


def state_to_covariant_geometry(
    state: Sequence[sp.Expr],
    state_derivative: Sequence[Sequence[sp.Expr]],
) -> dict[str, Any]:
    """Map U=(q_A,v_A,w_iA) and coordinate derivatives of U to covariant jets."""

    if len(state) != 55 or len(state_derivative) != DIMENSION:
        raise QuarticGeometricJetError("state map requires U[55] and partial_mu U[4][55]")
    if any(len(row) != 55 for row in state_derivative):
        raise QuarticGeometricJetError("every state-derivative row must contain 55 entries")

    metric = sp.zeros(DIMENSION)
    for field, (left, right) in enumerate(SYMMETRIC_METRIC_PAIRS):
        metric[left, right] = state[field]
        metric[right, left] = state[field]

    def gradient_index(derivative: int, field: int) -> int:
        return 11 + field if derivative == 0 else 22 + 11 * (derivative - 1) + field

    first = [
        [state[gradient_index(derivative, field)] for field in range(FIELD_COUNT)]
        for derivative in range(DIMENSION)
    ]
    second = [
        [
            [state_derivative[left][gradient_index(right, field)] for field in range(FIELD_COUNT)]
            for right in range(DIMENSION)
        ]
        for left in range(DIMENSION)
    ]
    metric_first = _zero_tensor((DIMENSION, DIMENSION, DIMENSION))
    metric_second = _zero_tensor((DIMENSION, DIMENSION, DIMENSION, DIMENSION))
    for field, (left, right) in enumerate(SYMMETRIC_METRIC_PAIRS):
        for derivative in range(DIMENSION):
            metric_first[derivative][left][right] = first[derivative][field]
            metric_first[derivative][right][left] = first[derivative][field]
            for second_derivative in range(DIMENSION):
                value = second[derivative][second_derivative][field]
                metric_second[derivative][second_derivative][left][right] = value
                metric_second[derivative][second_derivative][right][left] = value

    geometry = reconstruct_covariant_geometry(
        metric,
        metric_first,
        metric_second,
        [first[derivative][10] for derivative in range(DIMENSION)],
        [
            [second[left][right][10] for right in range(DIMENSION)]
            for left in range(DIMENSION)
        ],
    )
    geometry["partial_jet"] = {
        "field_first": first,
        "field_second": second,
        "integrability_residuals": [
            sp.factor(second[left][right][field] - second[right][left][field])
            for field in range(FIELD_COUNT)
            for left in range(DIMENSION)
            for right in range(left + 1, DIMENSION)
        ],
    }
    return geometry


def _coordinate_state(
    metric: sp.Matrix,
    scalar: sp.Expr,
    coordinates: Sequence[sp.Symbol],
) -> tuple[list[sp.Expr], list[list[sp.Expr]]]:
    fields = [metric[left, right] for left, right in SYMMETRIC_METRIC_PAIRS] + [scalar]
    state = list(fields)
    for derivative in range(DIMENSION):
        state.extend(sp.diff(field, coordinates[derivative]) for field in fields)
    state_derivative = [
        [sp.diff(component, coordinate) for component in state]
        for coordinate in coordinates
    ]
    return state, state_derivative


def _all_zero(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return all(_all_zero(item) for item in value)
    if isinstance(value, sp.MatrixBase):
        return all(sp.factor(sp.simplify(item)) == 0 for item in value)
    return sp.factor(sp.simplify(value)) == 0


@cache
def geometric_state_to_jet_control() -> tuple[bool, dict[str, Any]]:
    """Exercise the nonlinear map on exact curvilinear-flat and curved geometries."""

    time, radius, angle, height = sp.symbols("t r theta z", real=True)
    cylindrical_coordinates = (time, radius, angle, height)
    cylindrical_metric = sp.diag(-1, 1, radius**2, 1)
    cylindrical_state = _coordinate_state(
        cylindrical_metric, radius, cylindrical_coordinates
    )
    cylindrical = state_to_covariant_geometry(*cylindrical_state)
    cylindrical_connection_nonzero = any(
        item != 0
        for upper in cylindrical["connection"]
        for left in upper
        for item in left
    )
    flat_curvature_zero = _all_zero(cylindrical["riemann_up"])
    cylindrical_hessian_residual = sp.factor(
        cylindrical["scalar_hessian"][2][2] - radius
    )
    omitted_connection_hessian_residual = sp.factor(0 - radius)
    omitted_gamma_squared_riemann_residual = sp.factor(
        cylindrical["connection_first"][1][1][2][2]
        - cylindrical["connection_first"][2][1][1][2]
    )

    curved_time, x, y, z = sp.symbols("tau x y z", real=True)
    scale = sp.Function("a")(curved_time)
    flrw_coordinates = (curved_time, x, y, z)
    flrw_metric = sp.diag(-1, scale**2, scale**2, scale**2)
    flrw_state = _coordinate_state(flrw_metric, curved_time, flrw_coordinates)
    flrw = state_to_covariant_geometry(*flrw_state)
    expected_diagonal = [
        3 * sp.diff(scale, curved_time) ** 2 / scale**2,
        -(2 * scale * sp.diff(scale, curved_time, 2) + sp.diff(scale, curved_time) ** 2),
        -(2 * scale * sp.diff(scale, curved_time, 2) + sp.diff(scale, curved_time) ** 2),
        -(2 * scale * sp.diff(scale, curved_time, 2) + sp.diff(scale, curved_time) ** 2),
    ]
    flrw_einstein_residuals = [
        sp.factor(
            flrw["einstein"][left][right]
            - (expected_diagonal[left] if left == right else 0)
        )
        for left in range(DIMENSION)
        for right in range(DIMENSION)
    ]
    inverse_compatibility = [
        sp.factor(
            flrw["inverse_metric_first"][derivative][upper][right]
            + sum(
                flrw["inverse_metric"][upper, left]
                * sp.diff(flrw_metric[left, lower], flrw_coordinates[derivative])
                * flrw["inverse_metric"][lower, right]
                for left in range(DIMENSION)
                for lower in range(DIMENSION)
            )
        )
        for derivative in range(DIMENSION)
        for upper in range(DIMENSION)
        for right in range(DIMENSION)
    ]
    integrability_zero = _all_zero(
        cylindrical["partial_jet"]["integrability_residuals"]
    ) and _all_zero(flrw["partial_jet"]["integrability_residuals"])

    passed = bool(
        cylindrical_connection_nonzero
        and flat_curvature_zero
        and cylindrical_hessian_residual == 0
        and omitted_connection_hessian_residual != 0
        and omitted_gamma_squared_riemann_residual != 0
        and _all_zero(flrw_einstein_residuals)
        and _all_zero(inverse_compatibility)
        and integrability_zero
    )
    formula_contract = {
        "connection": "Gamma^rho_mu_nu=1/2 g^rho_sigma(d_mu g_sigma_nu+d_nu g_sigma_mu-d_sigma g_mu_nu)",
        "inverse_derivative": "d_lambda g^mu_nu=-g^mu_alpha(d_lambda g_alpha_beta)g^beta_nu",
        "scalar_hessian": "nabla_mu nabla_nu phi=d_mu d_nu phi-Gamma^rho_mu_nu d_rho phi",
        "riemann": "R^rho_sigma_mu_nu=d_mu Gamma^rho_nu_sigma-d_nu Gamma^rho_mu_sigma+Gamma^rho_mu_lam Gamma^lam_nu_sigma-Gamma^rho_nu_lam Gamma^lam_mu_sigma",
        "einstein": "G_mu_nu=R_mu_nu-1/2 g_mu_nu R",
    }
    return passed, {
        "control": "exact nonlinear 55-state to covariant geometric 2-jet map",
        "state": {
            "U_dimension": 55,
            "q_metric_components": 10,
            "q_scalar_components": 1,
            "coordinate_gradient_components": 44,
            "input_derivative_shape": [4, 55],
            "spacetime_integrability_residual_count": 66,
        },
        "outputs": [
            "g^mu_nu",
            "partial_lambda g^mu_nu",
            "Gamma^rho_mu_nu",
            "partial_lambda Gamma^rho_mu_nu",
            "nabla_mu phi",
            "nabla_mu nabla_nu phi",
            "R^rho_sigma_mu_nu",
            "R_mu_nu",
            "R",
            "G_mu_nu",
        ],
        "formula_contract": formula_contract,
        "formula_contract_sha256": hashlib.sha256(
            _canonical_json(formula_contract).encode()
        ).hexdigest(),
        "curvilinear_flat_control": {
            "metric": "diag(-1,1,r^2,1)",
            "scalar": "r",
            "connection_nonzero": cylindrical_connection_nonzero,
            "riemann_zero": flat_curvature_zero,
            "hessian_theta_theta_residual": str(cylindrical_hessian_residual),
        },
        "curved_control": {
            "metric": "diag(-1,a(t)^2,a(t)^2,a(t)^2)",
            "einstein_residuals": [str(item) for item in flrw_einstein_residuals],
            "inverse_derivative_compatibility_residuals": [
                str(item) for item in inverse_compatibility
            ],
        },
        "integrability_residuals_zero_on_coordinate_jets": integrability_zero,
        "negative_controls": {
            "omit_connection_from_scalar_hessian": {
                "exact_residual": str(omitted_connection_hessian_residual),
                "rejected": omitted_connection_hessian_residual != 0,
            },
            "omit_gamma_squared_from_riemann": {
                "exact_residual": str(omitted_gamma_squared_riemann_residual),
                "rejected": omitted_gamma_squared_riemann_residual != 0,
            },
        },
        "passed": passed,
        "scope": (
            "This is an exact local coordinate map from the first-order state and its first "
            "derivatives to covariant geometric tensors. It does not yet substitute the quartic "
            "evolution equations for the 11 time-time partials, generate the full gauge-fixed "
            "nonlinear source S(U,partial U), add gauge-driver variables, bound symmetrizer "
            "derivatives, or close commuted Sobolev energy estimates and a PDE bootstrap."
        ),
    }


def run_quartic_geometric_jet_campaign(
    first_order_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticGeometricJetError("unsupported campaign schema_version")
        if first_order_campaign.get("status") != (
            "pass_all_12_exact_55_variable_principal_first_order_reductions"
        ):
            raise QuarticGeometricJetError("first-order campaign prerequisite failed")
        expected = int(config.get("expected_candidate_count", 12))
        prerequisites = first_order_campaign.get("certificates", [])
        if len(prerequisites) != expected:
            raise QuarticGeometricJetError("unexpected first-order candidate count")
        control_passed, control = geometric_state_to_jet_control()
        if not control_passed:
            raise QuarticGeometricJetError("nonlinear geometric state-to-jet control failed")
        certificates = []
        for prerequisite in prerequisites:
            if prerequisite.get("status") != (
                "pass_exact_55_variable_principal_first_order_reduction"
            ):
                raise QuarticGeometricJetError("candidate first-order prerequisite failed")
            certificates.append(
                {
                    "schema_version": "sigma-quartic-geometric-jet-certificate-1.0",
                    "status": "pass_exact_nonlinear_geometric_state_to_jet_map",
                    "candidate_id": prerequisite["candidate_id"],
                    "coefficients": prerequisite["coefficients"],
                    "source_spatial_block_sha256": prerequisite[
                        "source_spatial_block_sha256"
                    ],
                    "geometric_formula_contract_sha256": control[
                        "formula_contract_sha256"
                    ],
                    "state_dimension": 55,
                    "covariant_outputs": control["outputs"],
                    "resolved_predecessor_gate": (
                        "incidence_defined_nonlinear_formula_map_unresolved"
                    ),
                    "remaining_gate": "quartic_gauge_fixed_nonlinear_evolution_source",
                    "scope": control["scope"],
                }
            )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps",
            "errors": [],
            "first_order_campaign_sha256": first_order_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "counts": {
                "selected": len(certificates),
                "geometric_state_to_jet_maps_passed": len(certificates),
                "rejected": 0,
            },
            "geometric_control": control,
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates now share an exact nonlinear coordinate map from "
                "their 55-variable first-order state and its first derivatives to the "
                "connection, scalar Hessian, curvature, and Einstein tensor."
            ),
            "scope": control["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticGeometricJetError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "first_order_campaign_sha256": first_order_campaign.get("content_sha256"),
            "counts": {
                "selected": 0,
                "geometric_state_to_jet_maps_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_geometric_jet_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
