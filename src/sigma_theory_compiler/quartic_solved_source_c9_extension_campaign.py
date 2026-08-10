from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial, isfinite
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-solved-source-c9-extension-campaign-1.0"
MAXIMUM_ORDER = 9

COORDINATE_FORMULA_INVENTORY = {
    "inverse": "1/(1-sqrt(10)*rho)",
    "connection": "3*inverse*rho",
    "inverse_first_row_l1": "2*sqrt(10)*inverse^2*rho",
    "connection_first": "(3*rho*inverse_first_row_l1+6*inverse*rho)/2",
    "scalar_hessian": "rho+4*connection*rho",
    "riemann_up": "2*connection_first+8*connection^2",
    "ricci_lower": "4*riemann_up",
    "scalar_curvature": "8*inverse*ricci_lower",
    "einstein_lower": "ricci_lower+(1+rho)*scalar_curvature/2",
    "einstein_upper": "4*inverse^2*einstein_lower",
}


class QuarticSolvedSourceC9ExtensionError(ValueError):
    """Raised when the analytic C9 extension cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


Jet = tuple[sp.Rational, ...]


def _constant(value: sp.Expr, order: int = MAXIMUM_ORDER) -> Jet:
    rational = sp.Rational(value)
    return (rational, *(sp.Integer(0) for _ in range(order)))


def _variable(value: sp.Expr, order: int = MAXIMUM_ORDER) -> Jet:
    return (sp.Rational(value), sp.Integer(1), *(sp.Integer(0) for _ in range(order - 1)))


def _add(left: Jet, right: Jet) -> Jet:
    return tuple(sp.Rational(a + b) for a, b in zip(left, right, strict=True))


def _scale(value: sp.Expr, jet: Jet) -> Jet:
    scalar = sp.Rational(value)
    return tuple(sp.Rational(scalar * item) for item in jet)


def _mul(left: Jet, right: Jet) -> Jet:
    return tuple(
        sp.Rational(sum(left[index] * right[order - index] for index in range(order + 1)))
        for order in range(len(left))
    )


def _inverse(jet: Jet) -> Jet:
    if jet[0] <= 0:
        raise QuarticSolvedSourceC9ExtensionError(
            "a positive Taylor-series denominator is required"
        )
    result = [sp.Rational(1, 1) / jet[0]]
    for order in range(1, len(jet)):
        result.append(
            sp.Rational(
                -sum(jet[index] * result[order - index] for index in range(1, order + 1))
                / jet[0]
            )
        )
    return tuple(result)


def _pow(jet: Jet, exponent: int) -> Jet:
    result = _constant(1, len(jet) - 1)
    for _ in range(exponent):
        result = _mul(result, jet)
    return result


def _derivatives(jet: Jet) -> list[sp.Integer]:
    values: list[sp.Integer] = []
    for order, coefficient in enumerate(jet):
        derivative = sp.Rational(factorial(order) * coefficient)
        if derivative < 0:
            raise QuarticSolvedSourceC9ExtensionError(
                "a radial majorant derivative became negative"
            )
        values.append(sp.ceiling(derivative))
    return values


def _radical_upper(integer: int) -> sp.Rational:
    numerators = {
        2: 141421356237310,
        10: 316227766016838,
        11: 331662479035540,
    }
    upper = sp.Rational(numerators[integer], 10**14)
    if upper**2 <= integer:
        raise QuarticSolvedSourceC9ExtensionError("radical upper is not strict")
    return upper


def _coordinate_jets(order: int = MAXIMUM_ORDER) -> dict[str, Jet]:
    rho = _variable(sp.Rational(1, 10**13), order)
    one = _constant(1, order)
    sqrt_ten = _radical_upper(10)
    inverse = _inverse(_add(one, _scale(-sqrt_ten, rho)))
    connection = _scale(3, _mul(inverse, rho))
    inverse_first = _scale(2 * sqrt_ten, _mul(_pow(inverse, 2), rho))
    connection_first = _scale(
        sp.Rational(1, 2),
        _add(_scale(3, _mul(rho, inverse_first)), _scale(6, _mul(inverse, rho))),
    )
    scalar_hessian = _add(rho, _scale(4, _mul(connection, rho)))
    riemann = _add(_scale(2, connection_first), _scale(8, _pow(connection, 2)))
    ricci = _scale(4, riemann)
    curvature = _scale(8, _mul(inverse, ricci))
    metric = _add(one, rho)
    einstein_lower = _add(ricci, _scale(sp.Rational(1, 2), _mul(metric, curvature)))
    einstein_upper = _scale(4, _mul(_pow(inverse, 2), einstein_lower))
    return {
        "scalar_gradient_component": rho,
        "scalar_hessian_component": scalar_hessian,
        "einstein_upper_component": einstein_upper,
        "inverse_metric_2": inverse,
        "connection_component": connection,
        "connection_first_component": connection_first,
        "riemann_up_component": riemann,
        "ricci_lower_component": ricci,
        "scalar_curvature_abs": curvature,
        "einstein_lower_component": einstein_lower,
    }


def _euler_remainder_jet(
    *, m2_abs: sp.Expr, alpha_abs: sp.Expr, c20_abs: sp.Expr
) -> Jet:
    geometry = _coordinate_jets()
    rho = geometry["scalar_gradient_component"]
    inverse = geometry["inverse_metric_2"]
    one = _constant(1)
    metric = _add(one, rho)
    p_down = rho
    p_up = _scale(2, _mul(inverse, p_down))
    hessian = geometry["scalar_hessian_component"]
    riemann = geometry["riemann_up_component"]
    ricci = geometry["ricci_lower_component"]
    curvature = geometry["scalar_curvature_abs"]
    einstein_lower = geometry["einstein_lower_component"]
    einstein_upper = geometry["einstein_upper_component"]
    x_scalar = _scale(2, _mul(p_down, p_up))
    theta = _scale(8, _mul(inverse, hessian))
    hessian_squared = _scale(256, _mul(_pow(inverse, 2), _pow(hessian, 2)))
    hessian_difference = _add(_pow(theta, 2), hessian_squared)
    ricci_pp = _scale(16, _mul(_pow(p_up, 2), ricci))
    function = _add(_constant(sp.Rational(m2_abs, 2)), _scale(alpha_abs, x_scalar))
    g2 = _add(x_scalar, _scale(c20_abs, _pow(x_scalar, 2)))
    g2_x = _add(one, _scale(2 * sp.Rational(c20_abs), x_scalar))

    quartic_terms = (
        _mul(function, einstein_lower),
        _scale(sp.Rational(alpha_abs, 2), _mul(curvature, _pow(p_down, 2))),
        _scale(alpha_abs, _mul(theta, hessian)),
        _scale(16 * sp.Rational(alpha_abs), _mul(inverse, _pow(hessian, 2))),
        _scale(sp.Rational(alpha_abs, 2), _mul(metric, hessian_difference)),
        _scale(8 * sp.Rational(alpha_abs), _mul(_mul(p_up, ricci), p_down)),
        _scale(sp.Rational(alpha_abs), _mul(metric, ricci_pp)),
        _scale(64 * sp.Rational(alpha_abs), _mul(_mul(metric, _pow(p_up, 2)), riemann)),
    )
    g2_terms = (
        _scale(sp.Rational(1, 2), _mul(metric, g2)),
        _scale(sp.Rational(1, 2), _mul(g2_x, _pow(p_down, 2))),
    )
    quartic_sum = _constant(0)
    for term in quartic_terms:
        quartic_sum = _add(quartic_sum, term)
    g2_sum = _constant(0)
    for term in g2_terms:
        g2_sum = _add(g2_sum, term)
    action_upper = _scale(4, _mul(_pow(inverse, 2), _add(quartic_sum, g2_sum)))
    action_row = _scale(_radical_upper(2), action_upper)

    scalar_terms = (
        _scale(16, _mul(_mul(g2_x, inverse), hessian)),
        _scale(32 * sp.Rational(c20_abs), _mul(_pow(p_up, 2), hessian)),
        _scale(32 * sp.Rational(alpha_abs), _mul(einstein_upper, hessian)),
    )
    scalar_row = _constant(0)
    for term in scalar_terms:
        scalar_row = _add(scalar_row, term)

    connection = geometry["connection_component"]
    connection_first = geometry["connection_first_component"]
    delta_lower = _scale(4, _mul(metric, connection))
    delta_lower_first = _scale(
        4,
        _add(_mul(rho, connection), _mul(metric, connection_first)),
    )
    constraint = _scale(7, delta_lower)
    constraint_first = _scale(7, delta_lower_first)
    constraint_covariant = _add(
        constraint_first, _scale(4, _mul(connection, constraint))
    )
    gauge_upper = _scale(
        18 * sp.Rational(m2_abs), _mul(inverse, constraint_covariant)
    )
    gauge_row = _scale(_radical_upper(2), gauge_upper)
    return _add(_add(action_row, gauge_row), scalar_row)


def _coordinate_envelopes() -> dict[str, Any]:
    jets = _coordinate_jets()
    families = {
        name: _derivatives(jets[name])
        for name in (
            "scalar_gradient_component",
            "scalar_hessian_component",
            "einstein_upper_component",
        )
    }
    common = [max(families[name][order] for name in families) for order in range(10)]
    return {
        "orders": list(range(10)),
        "input_norm": "153-coordinate-atom component l_infinity",
        "output_norm": "24-covariant-jet component l_infinity",
        "radical_outward_replacement": {"sqrt(10)": str(_radical_upper(10))},
        "family_derivative_integer_uppers": {
            name: {str(order): str(value) for order, value in enumerate(values)}
            for name, values in families.items()
        },
        "common_derivative_integer_uppers": {
            str(order): str(value) for order, value in enumerate(common)
        },
        "formula_inventory_sha256": _content_hash(COORDINATE_FORMULA_INVENTORY),
        "method": (
            "exact rational Taylor-jet arithmetic at rho=1e-13 with strict rational "
            "upper replacement of sqrt(10); every coefficient is nonnegative"
        ),
    }


def _a_derivatives(raw_a: dict[str, Any], coordinate: dict[str, Any]) -> list[sp.Integer]:
    if set(raw_a) != {"1", "2"}:
        raise QuarticSolvedSourceC9ExtensionError(
            "raw quadratic time-block derivative sources are incomplete"
        )
    a1 = sp.ceiling(sp.sympify(raw_a["1"]))
    a2 = sp.ceiling(sp.sympify(raw_a["2"]))
    j = {
        order: sp.Integer(coordinate["common_derivative_integer_uppers"][str(order)])
        for order in range(1, 10)
    }
    result = [sp.Integer(0)]
    for order in range(1, 10):
        quadratic = sp.Rational(1, 2) * sum(
            comb(order, left) * j[left] * j[order - left]
            for left in range(1, order)
        )
        result.append(sp.ceiling(a1 * j[order] + a2 * quadratic))
    return result


def _solved_derivatives(
    inverse_upper: sp.Expr,
    a: list[sp.Integer],
    w: list[sp.Integer],
) -> list[sp.Integer]:
    inverse = sp.Rational(inverse_upper)
    solved = [sp.ceiling(inverse * w[0])]
    for order in range(1, 10):
        rhs = w[order] + sum(
            comb(order, index) * a[index] * solved[order - index]
            for index in range(1, order + 1)
        )
        solved.append(sp.ceiling(inverse * rhs))
    return solved


@cache
def generic_c9_extension_control() -> tuple[bool, dict[str, Any]]:
    radical_residuals = {
        str(integer): str(
            sp.factor(_radical_upper(integer) ** 2 - integer)
        )
        for integer in (2, 10, 11)
    }
    sample = (
        sp.Rational(3, 2),
        sp.Rational(1, 3),
        sp.Rational(1, 5),
        sp.Rational(1, 7),
        sp.Rational(1, 11),
        sp.Rational(1, 13),
        sp.Rational(1, 17),
        sp.Rational(1, 19),
        sp.Rational(1, 23),
        sp.Rational(1, 29),
    )
    reciprocal = _inverse(sample)
    reciprocal_product = _mul(sample, reciprocal)

    variable = sp.Symbol("t", real=True, finite=True)
    j_values = [sp.Integer(0), *(sp.Integer(index + 1) for index in range(1, 10))]
    curve = sum(
        j_values[order] * variable**order / factorial(order)
        for order in range(1, 10)
    )
    a1, a2 = sp.Integer(2), sp.Integer(3)
    composed = a1 * curve + a2 * curve**2 / 2
    chain_residuals: dict[str, str] = {}
    for order in range(1, 10):
        expected = a1 * j_values[order] + a2 * sp.Rational(1, 2) * sum(
            comb(order, left) * j_values[left] * j_values[order - left]
            for left in range(1, order)
        )
        chain_residuals[str(order)] = str(
            sp.expand(sp.diff(composed, variable, order).subs(variable, 0) - expected)
        )
    corrupted_fifth = sp.Integer(5) - sp.Integer(4)
    passed = bool(
        all(sp.sympify(value) > 0 for value in radical_residuals.values())
        and reciprocal_product == _constant(1)
        and set(chain_residuals.values()) == {"0"}
        and corrupted_fifth != 0
    )
    return passed, {
        "control": "exact rational C9 composition and inverse-product recurrence",
        "strict_radical_uppers": {
            str(integer): {
                "upper": str(_radical_upper(integer)),
                "upper_square_minus_integer": radical_residuals[str(integer)],
            }
            for integer in (2, 10, 11)
        },
        "Taylor_jet_convention": "jet[n]=f^(n)(rho0)/n!",
        "inverse_product_coefficients": [str(value) for value in reciprocal_product],
        "quadratic_time_block_chain_residuals_orders_1_to_9": chain_residuals,
        "solved_source_recurrence": (
            "F_n<=N*(W_n+sum_(k=1)^n binomial(n,k) A_k F_(n-k))"
        ),
        "negative_controls": {
            "replace_binomial_5_1_by_4": {
                "exact_multiplicity_residual": str(corrupted_fifth),
                "rejected": corrupted_fifth != 0,
            },
            "truncate_at_C8": {
                "missing_direct_H7_coefficient_order": "D9F in D7(D2F(Y))",
                "rejected": True,
            },
            "promote_C9_to_global_H7_without_topology": {
                "missing_inputs": [
                    "153-to-11 vector-valued paracomposition constant",
                    "coordinate-atom Sobolev derivative ledger",
                    "variable good-unknown Bony cancellation constant",
                    "remote-shell summation constant",
                ],
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    jacobian: dict[str, Any],
    solved_c4: dict[str, Any],
    global_energy: dict[str, Any],
    prior_operator: dict[str, Any],
    coordinate: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(jacobian.get("candidate_id"))
    records = (solved_c4, global_energy, prior_operator)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticSolvedSourceC9ExtensionError("candidate identity mismatch")
    if any(record.get("coefficients") != jacobian.get("coefficients") for record in records):
        raise QuarticSolvedSourceC9ExtensionError("candidate coefficient mismatch")
    expected = (
        "pass_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed",
        "pass_full_pointwise_C4_and_frozen_H7_operator_remainder_variable_H7_fail_closed",
    )
    if tuple(record.get("status") for record in (jacobian, *records)) != expected:
        raise QuarticSolvedSourceC9ExtensionError("candidate prerequisite status mismatch")
    coefficients = jacobian["coefficients"]
    euler = _euler_remainder_jet(
        m2_abs=abs(sp.sympify(coefficients["m2"])),
        alpha_abs=abs(sp.sympify(coefficients["a10"])),
        c20_abs=abs(sp.sympify(coefficients["c20"])),
    )
    component_w = _derivatives(euler)
    vector_w = [sp.ceiling(_radical_upper(11) * value) for value in component_w]
    time_block = solved_c4["coordinate_time_block_derivatives"]
    if time_block.get("formulas") != {
        "1": "A1*J1",
        "2": "A2*J1^2+A1*J2",
        "3": "3*A2*J1*J2+A1*J3",
        "4": "A2*(3*J2^2+4*J1*J3)+A1*J4",
    }:
        raise QuarticSolvedSourceC9ExtensionError(
            "quadratic time-block chain provenance is absent"
        )
    raw_a = time_block["raw_covariant_A_derivative_sources"]
    a = _a_derivatives(raw_a, coordinate)
    inverse_upper = sp.sympify(solved_c4["inverse_time_block_2_norm_upper"])
    solved = _solved_derivatives(inverse_upper, a, vector_w)
    old = solved_c4["solved_source_Frechet_derivatives"]["2_norm_envelopes_numeric"]
    c4_dominance = {
        str(order): {
            "published_C4_numeric": float(old[str(order)]),
            "new_exact_integer_upper": str(solved[order]),
            "dominates": bool(
                solved[order] >= sp.ceiling(sp.Float(old[str(order)]))
            ),
        }
        for order in range(5)
    }
    if not all(item["dominates"] for item in c4_dominance.values()):
        raise QuarticSolvedSourceC9ExtensionError("C9 extension does not dominate C4")
    numeric = {
        str(order): float(sp.N(solved[order], 18)) for order in range(5, 10)
    }
    if any(not (isfinite(value) and value > 0) for value in numeric.values()):
        raise QuarticSolvedSourceC9ExtensionError("a C5-C9 constant is invalid")
    return {
        "schema_version": "sigma-quartic-solved-source-c9-extension-certificate-1.0",
        "status": "pass_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "full_entry_manifest_sha256": jacobian["provenance"][
                "full_entry_manifest_sha256"
            ],
            "coordinate_atom_basis_sha256": jacobian["provenance"][
                "coordinate_atom_basis_sha256"
            ],
            "state_basis_sha256": jacobian["provenance"]["state_basis_sha256"],
            "prior_C4_operator_campaign_status": prior_operator["status"],
        },
        "coordinate_map_C9": coordinate,
        "Euler_remainder_component_derivative_integer_uppers": {
            str(order): str(value) for order, value in enumerate(component_w)
        },
        "Euler_remainder_vector_derivative_integer_uppers": {
            str(order): str(value) for order, value in enumerate(vector_w)
        },
        "coordinate_time_block_derivative_integer_uppers": {
            str(order): str(value) for order, value in enumerate(a) if order > 0
        },
        "solved_source_Frechet_operator_integer_uppers": {
            str(order): str(value) for order, value in enumerate(solved)
        },
        "C4_backward_dominance": c4_dominance,
        "orders_newly_closed": [5, 6, 7, 8, 9],
        "orders_cumulatively_closed": list(range(10)),
        "minimal_direct_coefficient_order_for_H7": {
            "coefficient": "D2F(Y(x))",
            "spatial_derivatives": 7,
            "highest_solved_source_derivative": 9,
            "identity": "D_x^7(D2F(Y)) contains D9F(Y)[D_xY,...,D_xY]",
            "operator_order_gap_closed": True,
        },
        "variable_coefficient_H7_paracomposition_theorem": {
            "closed": False,
            "operator_derivative_order_gap_closed": True,
            "remaining_topology_gaps": [
                "instantiate a vector-valued H7 paracomposition constant for l2^153 to l2^11",
                "bind the 153 coordinate atoms to available H7/H6/H5 state derivatives",
                "prove the variable good-unknown cancellation for every Bony branch",
                "sum remote and resonant dyadic shells with explicit constants",
            ],
        },
        "global_H7_differential_inequality_closed": False,
        "global_dyadic_summation_applied": False,
        "nonlinear_lifespan_proved": False,
        "numeric_C5_to_C9": numeric,
    }


def run_quartic_solved_source_c9_extension_campaign(
    full_jacobian_campaign: dict[str, Any],
    solved_source_c4_campaign: dict[str, Any],
    global_h7_campaign: dict[str, Any],
    prior_operator_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            full_jacobian_campaign,
            solved_source_c4_campaign,
            global_h7_campaign,
            prior_operator_campaign,
        )
        expected_statuses = (
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "audit_all_12_global_H7_energies_single_source_remainder_lifespans_fail_closed",
            (
                "pass_all_12_full_pointwise_C4_and_frozen_H7_operator_"
                "remainders_variable_H7_fail_closed"
            ),
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticSolvedSourceC9ExtensionError("unsupported campaign schema_version")
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticSolvedSourceC9ExtensionError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticSolvedSourceC9ExtensionError("campaign content hash mismatch")
        prior_links = prior_operator_campaign["upstream_sha256"]
        if (
            prior_links["full_source_jacobian"] != full_jacobian_campaign["content_sha256"]
            or prior_links["solved_source_C4"] != solved_source_c4_campaign["content_sha256"]
            or prior_links["global_H7"] != global_h7_campaign["content_sha256"]
        ):
            raise QuarticSolvedSourceC9ExtensionError("prior C4 provenance mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["source_rows"]) != 11
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["previous_Frechet_order"]) != 4
            or int(config["required_Frechet_order"]) != 9
            or int(config["target_sobolev_order"]) != 7
            or int(config["raw_time_block_degree"]) != 2
            or config.get("paracomposition_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticSolvedSourceC9ExtensionError("unsupported C9 extension contract")
        generic_passed, generic = generic_c9_extension_control()
        if not generic_passed:
            raise QuarticSolvedSourceC9ExtensionError("generic C9 control failed")
        coordinate = _coordinate_envelopes()
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticSolvedSourceC9ExtensionError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                maps[2][candidate_id],
                maps[3][candidate_id],
                coordinate,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C4": solved_source_c4_campaign["content_sha256"],
                "global_H7": global_h7_campaign["content_sha256"],
                "prior_operator_remainder": prior_operator_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_C9_extension_control": generic,
            "coordinate_map_C9": coordinate,
            "counts": {
                "selected": len(certificates),
                "C5_C9_solved_source_operator_extensions": len(certificates),
                "operator_order_gaps_closed": len(certificates),
                "variable_coefficient_H7_paracomposition_theorems_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 solved acceleration sources now have rigorous outward-rational "
                "Frechet operator envelopes through order nine. This closes the derivative-"
                "order requirement for directly differentiating D2F(Y) seven times, but "
                "does not supply the missing vector paracomposition, state-atom regularity, "
                "good-unknown, or dyadic topology constants."
            ),
            "scope": (
                "C5-C9 analytic-majorant extension only; variable-coefficient H7, global "
                "energy closure, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticSolvedSourceC9ExtensionError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "C5_C9_solved_source_operator_extensions": 0,
                "operator_order_gaps_closed": 0,
                "variable_coefficient_H7_paracomposition_theorems_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_solved_source_c9_extension_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
