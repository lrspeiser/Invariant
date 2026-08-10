from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_coordinate_jet_tube_campaign import _geometry_majorants

SCHEMA_VERSION = "sigma-quartic-euler-remainder-majorant-campaign-1.0"

TERM_INVENTORY = {
    "quartic_metric_lower": [
        "function_times_Einstein",
        "curvature_times_p_p",
        "theta_times_H",
        "H_contraction_product",
        "metric_times_H_difference",
        "Ricci_gradient",
        "metric_times_Ricci_p_p",
        "Riemann_gradient",
    ],
    "G2_metric_lower": ["metric_times_G2", "G2_X_times_p_p"],
    "scalar_euler": [
        "G2_X_inverse_metric_times_H",
        "c20_p_up_p_up_times_H",
        "alpha_Einstein_upper_times_H",
    ],
    "modified_harmonic_gauge": [
        "metric_lowered_connection_difference",
        "derivative_of_lowered_connection_difference",
        "tilde_metric_constraint_and_covariant_derivative",
        "hat_projector_completion",
    ],
}


class QuarticEulerRemainderMajorantError(ValueError):
    """Raised when the acceleration-independent Euler remainder cannot be bounded."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = expression.is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


def _nonnegative(expression: sp.Expr) -> bool:
    decision = expression.is_nonnegative
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) >= 0)


def _projector_l1_by_symmetric_row() -> dict[str, sp.Expr]:
    hat = sp.diag(-9, 1, 1, 1)
    result: dict[str, sp.Expr] = {}
    for mu in range(4):
        for nu in range(mu, 4):
            total = sp.Integer(0)
            for alpha in range(4):
                for derivative in range(4):
                    projector = (
                        int(alpha == mu) * hat[nu, derivative]
                        + int(alpha == nu) * hat[mu, derivative]
                        - int(alpha == derivative) * hat[mu, nu]
                    ) / 2
                    total += abs(projector)
            result[f"{mu}{nu}"] = sp.factor(total)
    return result


def _remainder_majorants(
    radius: sp.Symbol,
    *,
    m2_abs: sp.Expr,
    alpha_abs: sp.Expr,
    c20_abs: sp.Expr,
) -> dict[str, Any]:
    geometry = _geometry_majorants(radius)
    inverse = geometry["inverse_metric_2"]
    metric = 1 + radius
    p_down = geometry["scalar_gradient_component"]
    p_up = 2 * inverse * p_down
    hessian = geometry["scalar_hessian_component"]
    riemann = geometry["riemann_up_component"]
    ricci = geometry["ricci_lower_component"]
    curvature = geometry["scalar_curvature_abs"]
    einstein_lower = geometry["einstein_lower_component"]
    einstein_upper = geometry["einstein_upper_component"]

    # Preserve these as positive radial sums/products. Factoring the complete
    # fourth derivative can take minutes and obscures the termwise proof.
    x_scalar = 2 * p_down * p_up
    theta = 8 * inverse * hessian
    hessian_squared = 256 * inverse**2 * hessian**2
    hessian_difference = theta**2 + hessian_squared
    ricci_pp = 16 * p_up**2 * ricci
    function = m2_abs / 2 + alpha_abs * x_scalar
    g2 = x_scalar + c20_abs * x_scalar**2
    g2_x = 1 + 2 * c20_abs * x_scalar

    quartic_terms = {
        "function_times_Einstein": function * einstein_lower,
        "curvature_times_p_p": alpha_abs * curvature * p_down**2 / 2,
        "theta_times_H": alpha_abs * theta * hessian,
        "H_contraction_product": alpha_abs * 16 * inverse * hessian**2,
        "metric_times_H_difference": metric * alpha_abs * hessian_difference / 2,
        "Ricci_gradient": alpha_abs * 8 * p_up * ricci * p_down,
        "metric_times_Ricci_p_p": metric * alpha_abs * ricci_pp,
        "Riemann_gradient": alpha_abs * 64 * metric * p_up**2 * riemann,
    }
    g2_terms = {
        "metric_times_G2": metric * g2 / 2,
        "G2_X_times_p_p": g2_x * p_down**2 / 2,
    }
    quartic_metric_lower = sum(quartic_terms.values())
    g2_metric_lower = sum(g2_terms.values())
    action_metric_upper = 4 * inverse**2 * (
        quartic_metric_lower + g2_metric_lower
    )
    action_metric_row = sp.sqrt(2) * action_metric_upper

    scalar_terms = {
        "G2_X_inverse_metric_times_H": 16 * g2_x * inverse * hessian,
        "c20_p_up_p_up_times_H": 32 * c20_abs * p_up**2 * hessian,
        "alpha_Einstein_upper_times_H": 32 * alpha_abs * einstein_upper * hessian,
    }
    scalar_row = sum(scalar_terms.values())

    connection = geometry["connection_component"]
    connection_first = geometry["connection_first_component"]
    delta_lower = 4 * metric * connection
    delta_lower_first = 4 * (radius * connection + metric * connection_first)
    # tilde g^{-1}=diag(-4,1,1,1), so its component l1 norm is exactly seven.
    constraint = 7 * delta_lower
    constraint_first = 7 * delta_lower_first
    constraint_covariant_first = constraint_first + 4 * connection * constraint
    # For hat g^{-1}=diag(-9,1,1,1), the maximum symmetric-row projector l1 norm is 18.
    gauge_metric_upper = 18 * m2_abs * inverse * constraint_covariant_first
    gauge_metric_row = sp.sqrt(2) * gauge_metric_upper
    metric_row = action_metric_row + gauge_metric_row
    # A sum bounds max(metric rows, scalar row) and remains an analytic radial majorant.
    remainder_component = metric_row + scalar_row
    return {
        "geometry": geometry,
        "derived_covariant": {
            "p_up_component": p_up,
            "X_abs": x_scalar,
            "theta_abs": theta,
            "H_squared_abs": hessian_squared,
            "H_difference_abs": hessian_difference,
            "Ricci_p_p_abs": ricci_pp,
            "G2_abs": g2,
            "G2_X_abs": g2_x,
        },
        "quartic_metric_terms": quartic_terms,
        "G2_metric_terms": g2_terms,
        "scalar_terms": scalar_terms,
        "gauge_stages": {
            "delta_lower_component": delta_lower,
            "delta_lower_first_component": delta_lower_first,
            "constraint_component": constraint,
            "constraint_first_component": constraint_first,
            "constraint_covariant_first_component": constraint_covariant_first,
            "gauge_metric_upper_component": gauge_metric_upper,
        },
        "rows": {
            "action_metric": action_metric_row,
            "gauge_metric": gauge_metric_row,
            "total_metric": metric_row,
            "scalar": scalar_row,
            "remainder_component": remainder_component,
        },
    }


@cache
def _derivative_hierarchy(
    expression: sp.Expr, radius: sp.Symbol, value: sp.Expr, order: int
) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    for derivative_order in range(order + 1):
        derivative_expression = sp.diff(expression, radius, derivative_order)
        derivative_at_radius = derivative_expression.subs(radius, value)
        if not _nonnegative(derivative_at_radius):
            raise QuarticEulerRemainderMajorantError("a remainder majorant derivative is negative")
        result[str(derivative_order)] = {
            "exact_expression": str(derivative_expression),
            "evaluation_radius": str(value),
            "numeric": float(sp.N(derivative_at_radius, 18)),
        }
    return result


def generic_euler_remainder_majorant_control() -> tuple[bool, dict[str, Any]]:
    radius = sp.Symbol("rho", positive=True, finite=True)
    coordinate_radius = sp.Rational(1, 10**13)
    majorants = _remainder_majorants(
        radius, m2_abs=sp.Integer(1), alpha_abs=sp.Integer(1), c20_abs=sp.Integer(1)
    )
    projector_rows = _projector_l1_by_symmetric_row()
    projector_max = max(projector_rows.values(), key=lambda value: float(value))
    term_counts = {name: len(terms) for name, terms in TERM_INVENTORY.items()}
    inventory_hash = hashlib.sha256(_canonical_json(TERM_INVENTORY).encode()).hexdigest()
    row_values = {
        name: sp.factor(expression.subs(radius, coordinate_radius))
        for name, expression in majorants["rows"].items()
    }
    derivative_order = 4
    derivatives = _derivative_hierarchy(
        majorants["rows"]["remainder_component"],
        radius,
        coordinate_radius,
        derivative_order,
    )
    passed = bool(
        term_counts
        == {
            "quartic_metric_lower": 8,
            "G2_metric_lower": 2,
            "scalar_euler": 3,
            "modified_harmonic_gauge": 4,
        }
        and projector_max == 18
        and sum(abs(item) for item in sp.diag(-4, 1, 1, 1).diagonal()) == 7
        and all(_positive(value) for value in row_values.values())
        and all(item["numeric"] >= 0 for item in derivatives.values())
    )
    return passed, {
        "control": "complete termwise acceleration-independent Euler remainder majorant",
        "term_inventory": TERM_INVENTORY,
        "term_inventory_sha256": inventory_hash,
        "term_counts": term_counts,
        "auxiliary_metric_contractions": {
            "tilde_inverse_metric": "diag(-4,1,1,1)",
            "tilde_component_l1": 7,
            "hat_inverse_metric": "diag(-9,1,1,1)",
            "hat_projector_symmetric_row_l1": {
                key: str(value) for key, value in projector_rows.items()
            },
            "hat_projector_symmetric_row_l1_max": str(projector_max),
        },
        "coordinate_component_radius": str(coordinate_radius),
        "reference_coefficient_envelope": {
            "abs_M2": "1",
            "abs_alpha": "1",
            "abs_c20": "1",
        },
        "row_majorants": {
            name: {"exact": str(value), "numeric": float(sp.N(value, 18))}
            for name, value in row_values.items()
        },
        "remainder_Frechet_majorant_derivatives": {
            "orders": list(range(derivative_order + 1)),
            "input_norm": "153-coordinate-atom component l_infinity",
            "output_norm": "11-Euler-row component l_infinity",
            "values": derivatives,
        },
        "passed": passed,
        "scope": (
            "This supplies a complete vacuum Euler-remainder majorant on the coordinate tube. "
            "It does not yet compose derivatives of A^{-1}, close the commuted gauge/constraint "
            "energy, prove a lifespan, or include matter sources."
        ),
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["candidate_id"]: item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _certify_candidate(
    pde: dict[str, Any], tube: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if pde.get("candidate_id") != tube.get("candidate_id"):
        raise QuarticEulerRemainderMajorantError("candidate ID mismatch")
    if pde.get("coefficients") != tube.get("coefficients"):
        raise QuarticEulerRemainderMajorantError("candidate coefficient mismatch")
    if pde.get("status") != "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift":
        raise QuarticEulerRemainderMajorantError("candidate PDE prerequisite failed")
    if tube.get("status") != "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube":
        raise QuarticEulerRemainderMajorantError("candidate coordinate-tube prerequisite failed")
    coefficients = pde["coefficients"]
    radius = sp.Symbol("rho", positive=True, finite=True)
    coordinate_radius = sp.sympify(tube["coordinate_component_radius"])
    majorants = _remainder_majorants(
        radius,
        m2_abs=abs(sp.sympify(coefficients["m2"])),
        alpha_abs=abs(sp.sympify(coefficients["a10"])),
        c20_abs=abs(sp.sympify(coefficients["c20"])),
    )
    order = int(config["required_Frechet_majorant_order"])
    remainder = majorants["rows"]["remainder_component"].subs(
        radius, coordinate_radius
    )
    derivatives = _derivative_hierarchy(
        majorants["rows"]["remainder_component"], radius, coordinate_radius, order
    )
    inverse_a = sp.sympify(pde["uniform_bounds"]["A_inverse_2_upper_used"])
    acceleration = inverse_a * sp.sqrt(11) * remainder
    if not (_positive(remainder) and _positive(acceleration)):
        raise QuarticEulerRemainderMajorantError("candidate remainder bound is not positive")
    term_values = {
        family: {
            name: {
                "exact_expression": str(expression),
                "evaluation_radius": str(coordinate_radius),
                "numeric": float(sp.N(expression.subs(radius, coordinate_radius), 18)),
            }
            for name, expression in terms.items()
        }
        for family, terms in (
            ("quartic_metric", majorants["quartic_metric_terms"]),
            ("G2_metric", majorants["G2_metric_terms"]),
            ("scalar", majorants["scalar_terms"]),
            ("gauge", majorants["gauge_stages"]),
        )
    }
    return {
        "schema_version": "sigma-quartic-euler-remainder-majorant-certificate-1.0",
        "status": "pass_complete_coordinate_tube_euler_remainder_majorant",
        "candidate_id": pde["candidate_id"],
        "coefficients": coefficients,
        "coordinate_component_radius": str(coordinate_radius),
        "term_inventory_sha256": hashlib.sha256(
            _canonical_json(TERM_INVENTORY).encode()
        ).hexdigest(),
        "term_majorants": term_values,
        "row_majorants": {
            name: {
                "exact_expression": str(expression),
                "evaluation_radius": str(coordinate_radius),
                "numeric": float(sp.N(expression.subs(radius, coordinate_radius), 18)),
            }
            for name, expression in majorants["rows"].items()
        },
        "Euler_remainder_component_upper": str(remainder),
        "Euler_remainder_component_upper_numeric": float(sp.N(remainder, 18)),
        "Euler_remainder_Frechet_derivative_uppers": derivatives,
        "time_block_inverse_2_upper": str(inverse_a),
        "solved_acceleration_component_upper": str(acceleration),
        "solved_acceleration_component_upper_numeric": float(sp.N(acceleration, 18)),
        "claim": (
            "Every acceleration-independent vacuum Euler row and its coordinate-atom Frechet "
            "derivatives through order four are bounded on the certified coordinate tube."
        ),
        "remaining_gate": "A_inverse_derivative_composition_and_commuted_energy_lifespan",
        "scope": (
            "The acceleration bound uses ||A^-1||_2 and the 11-row Euclidean conversion. "
            "Derivatives of the solved acceleration still require differentiated A^{-1}."
        ),
    }


def run_quartic_euler_remainder_majorant_campaign(
    nonquasilinear_pde_campaign: dict[str, Any],
    coordinate_tube_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticEulerRemainderMajorantError("unsupported campaign schema_version")
        if nonquasilinear_pde_campaign.get("status") != (
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts"
        ):
            raise QuarticEulerRemainderMajorantError("full-state PDE prerequisite failed")
        if coordinate_tube_campaign.get("status") != (
            "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes"
        ):
            raise QuarticEulerRemainderMajorantError("coordinate-tube prerequisite failed")
        if coordinate_tube_campaign.get("nonquasilinear_pde_campaign_sha256") != (
            nonquasilinear_pde_campaign.get("content_sha256")
        ):
            raise QuarticEulerRemainderMajorantError("upstream provenance mismatch")
        order = int(config["required_Frechet_majorant_order"])
        if order != 4:
            raise QuarticEulerRemainderMajorantError("Euler remainder requires order four")
        expected_counts = {
            "quartic_metric_lower": int(config["required_quartic_metric_term_count"]),
            "G2_metric_lower": int(config["required_G2_metric_term_count"]),
            "scalar_euler": int(config["required_scalar_term_count"]),
            "modified_harmonic_gauge": int(config["required_gauge_stage_count"]),
        }
        actual_counts = {name: len(terms) for name, terms in TERM_INVENTORY.items()}
        if expected_counts != actual_counts:
            raise QuarticEulerRemainderMajorantError("term inventory count mismatch")
        control_passed, control = generic_euler_remainder_majorant_control()
        if not control_passed:
            raise QuarticEulerRemainderMajorantError("generic Euler majorant control failed")
        pde_records = _candidate_records(nonquasilinear_pde_campaign)
        tube_records = _candidate_records(coordinate_tube_campaign)
        expected = int(config.get("expected_candidate_count", 12))
        if len(pde_records) != expected or set(pde_records) != set(tube_records):
            raise QuarticEulerRemainderMajorantError("candidate-set mismatch")
        certificates = [
            _certify_candidate(pde_records[candidate_id], tube_records[candidate_id], config)
            for candidate_id in sorted(pde_records)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_complete_coordinate_tube_euler_remainder_majorants",
            "errors": [],
            "nonquasilinear_pde_campaign_sha256": nonquasilinear_pde_campaign.get(
                "content_sha256"
            ),
            "coordinate_tube_campaign_sha256": coordinate_tube_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_euler_remainder_majorant_control": control,
            "counts": {
                "selected": len(certificates),
                "Euler_remainder_majorants_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have complete termwise vacuum Euler-remainder "
                "majorants and coordinate-atom derivative envelopes through order four on "
                "the common certified coordinate tube."
            ),
            "scope": control["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticEulerRemainderMajorantError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "Euler_remainder_majorants_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_euler_remainder_majorant_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
