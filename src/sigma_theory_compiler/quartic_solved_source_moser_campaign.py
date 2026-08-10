from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_quasilinear_moser_campaign import (
    generic_inverse_product_derivative_control,
)

SCHEMA_VERSION = "sigma-quartic-solved-source-moser-campaign-1.0"

JET_FAMILIES = (
    "scalar_gradient_component",
    "scalar_hessian_component",
    "einstein_upper_component",
)


class QuarticSolvedSourceMoserError(ValueError):
    """Raised when the differentiated solved source cannot be bounded."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = expression.is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


@cache
def generic_quadratic_composition_control() -> tuple[bool, dict[str, Any]]:
    """Verify the fourth-order chain rule for a quadratic matrix coefficient."""

    u, y = sp.symbols("u y", real=True, finite=True)
    a0, a1, a2, j0, j1, j2, j3, j4 = sp.symbols(
        "a0 a1 a2 j0 j1 j2 j3 j4", real=True, finite=True
    )
    coefficient = a0 + a1 * y + a2 * y**2 / 2
    jet = j0 + j1 * u + j2 * u**2 / 2 + j3 * u**3 / 6 + j4 * u**4 / 24
    composed = coefficient.subs(y, jet)
    coefficient_first = sp.diff(coefficient, y).subs(y, j0)
    coefficient_second = sp.diff(coefficient, y, 2)
    expected = {
        1: coefficient_first * j1,
        2: coefficient_second * j1**2 + coefficient_first * j2,
        3: 3 * coefficient_second * j1 * j2 + coefficient_first * j3,
        4: coefficient_second * (3 * j2**2 + 4 * j1 * j3)
        + coefficient_first * j4,
    }
    residuals = {
        str(order): sp.factor(sp.diff(composed, u, order).subs(u, 0) - expression)
        for order, expression in expected.items()
    }
    corrupted_fourth = sp.factor(
        sp.diff(composed, u, 4).subs(u, 0)
        - (
            coefficient_second * (2 * j2**2 + 4 * j1 * j3)
            + coefficient_first * j4
        )
    )
    witness = {
        a0: 1,
        a1: 2,
        a2: 3,
        j0: 5,
        j1: 7,
        j2: 11,
        j3: 13,
        j4: 17,
    }
    corrupted_witness = sp.factor(corrupted_fourth.subs(witness))
    inverse_passed, inverse_control = generic_inverse_product_derivative_control()
    passed = bool(
        all(residual == 0 for residual in residuals.values())
        and corrupted_witness != 0
        and inverse_passed
    )
    return passed, {
        "control": "quadratic coefficient composition and inverse-product recurrence",
        "quadratic_composition_formulas": {
            "1": "A1*J1",
            "2": "A2*J1^2+A1*J2",
            "3": "3*A2*J1*J2+A1*J3",
            "4": "A2*(3*J2^2+4*J1*J3)+A1*J4",
        },
        "quadratic_composition_residuals": {
            order: str(residual) for order, residual in residuals.items()
        },
        "inverse_product_identity": inverse_control["identity"],
        "inverse_product_residuals": inverse_control["residuals"],
        "negative_control": {
            "corruption": "replace the D4 coefficient 3 on A2*(J2)^2 by 2",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": corrupted_witness != 0,
        },
        "passed": passed,
        "scope": (
            "Exact scalar representatives verify the Banach-space chain-rule and product-rule "
            "multiplicities used by the matrix/vector norm recurrence."
        ),
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["candidate_id"]: item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _coordinate_jet_derivatives(
    coordinate_tube_campaign: dict[str, Any], order: int
) -> dict[str, Any]:
    control = coordinate_tube_campaign["generic_coordinate_jet_majorant_control"]
    derivative_data = control["Frechet_majorant_derivatives"]
    if derivative_data.get("orders") != list(range(order + 1)):
        raise QuarticSolvedSourceMoserError("coordinate-jet derivative order mismatch")
    families = derivative_data["families"]
    envelopes: dict[str, Any] = {}
    for derivative_order in range(1, order + 1):
        available = {
            name: families[name][str(derivative_order)] for name in JET_FAMILIES
        }
        dominant_name = max(
            available,
            key=lambda name: float(available[name]["numeric"]),
        )
        dominant = available[dominant_name]
        dominant_exact = sp.sympify(dominant["exact"])
        if any(
            not _positive(dominant_exact - sp.sympify(item["exact"]))
            and dominant_exact != sp.sympify(item["exact"])
            for item in available.values()
        ):
            raise QuarticSolvedSourceMoserError(
                "selected coordinate-jet derivative is not a common envelope"
            )
        envelopes[str(derivative_order)] = {
            "dominant_family": dominant_name,
            "exact": str(dominant_exact),
            "numeric": float(sp.N(dominant_exact, 18)),
            "covered_families": list(JET_FAMILIES),
        }
    return {
        "input_norm": "153-coordinate-atom component l_infinity",
        "output_norm": "24-covariant-jet component l_infinity",
        "orders": list(range(1, order + 1)),
        "envelopes": envelopes,
        "justification": (
            "The maximum of the scalar-gradient, scalar-Hessian, and upper-Einstein "
            "component envelopes bounds the 24-component covariant jet map."
        ),
    }


@cache
def _evaluated_derivative(
    exact_expression: str, evaluation_radius: str
) -> sp.Expr:
    expression = sp.sympify(exact_expression)
    radius_symbols = [symbol for symbol in expression.free_symbols if str(symbol) == "rho"]
    if len(radius_symbols) > 1:
        raise QuarticSolvedSourceMoserError("ambiguous Euler-remainder radius symbol")
    if radius_symbols:
        expression = expression.subs(radius_symbols[0], sp.sympify(evaluation_radius))
    return expression


def _coordinate_a_derivatives(
    raw_a: dict[str, Any], jet: dict[str, Any]
) -> tuple[list[sp.Expr], dict[str, Any]]:
    a1 = sp.sympify(raw_a["1"])
    a2 = sp.sympify(raw_a["2"])
    if sp.sympify(raw_a["3"]) != 0 or sp.sympify(raw_a["4"]) != 0:
        raise QuarticSolvedSourceMoserError("raw time block is not quadratic")
    j = {
        order: sp.sympify(jet["envelopes"][str(order)]["exact"])
        for order in range(1, 5)
    }
    derivatives = [
        sp.Integer(0),
        a1 * j[1],
        a2 * j[1] ** 2 + a1 * j[2],
        3 * a2 * j[1] * j[2] + a1 * j[3],
        a2 * (3 * j[2] ** 2 + 4 * j[1] * j[3]) + a1 * j[4],
    ]
    if not all(_positive(value) for value in derivatives[1:]):
        raise QuarticSolvedSourceMoserError("coordinate time-block derivative is not positive")
    return derivatives, {
        "raw_covariant_A_derivative_sources": {"1": raw_a["1"], "2": raw_a["2"]},
        "formulas": {
            "1": "A1*J1",
            "2": "A2*J1^2+A1*J2",
            "3": "3*A2*J1*J2+A1*J3",
            "4": "A2*(3*J2^2+4*J1*J3)+A1*J4",
        },
        "2_norm_envelopes_numeric": {
            str(order): float(sp.N(derivatives[order], 18))
            for order in range(1, 5)
        },
    }


def _solved_source_derivatives(
    inverse_a: sp.Expr,
    a_derivatives: list[sp.Expr],
    remainder_derivatives: dict[str, Any],
    order: int,
    state_dimension: int,
) -> tuple[list[sp.Expr], dict[str, Any]]:
    vector_conversion = sp.sqrt(state_dimension)
    remainder_exact = [
        _evaluated_derivative(
            remainder_derivatives[str(derivative_order)]["exact_expression"],
            remainder_derivatives[str(derivative_order)]["evaluation_radius"],
        )
        for derivative_order in range(order + 1)
    ]
    # Work at high precision after exact source evaluation. The artifact is provenance-bound
    # to the exact upstream expressions and records the recurrence rather than expanding a
    # multi-megabyte algebraic expression for every candidate and derivative order.
    a_numeric = [sp.N(value, 80) for value in a_derivatives]
    w_numeric = [sp.N(value, 80) for value in remainder_exact]
    inverse_numeric = sp.N(inverse_a, 80)
    solved = [inverse_numeric * vector_conversion * w_numeric[0]]
    term_breakdown: dict[str, Any] = {
        "0": {
            "formula": "N*sqrt(11)*W0",
            "remainder_term_numeric": float(sp.N(vector_conversion * w_numeric[0], 18)),
            "coefficient_product_terms_numeric": {},
        }
    }
    for derivative_order in range(1, order + 1):
        remainder_term = vector_conversion * w_numeric[derivative_order]
        products = {
            str(a_order): (
                comb(derivative_order, a_order)
                * a_numeric[a_order]
                * solved[derivative_order - a_order]
            )
            for a_order in range(1, derivative_order + 1)
        }
        rhs = remainder_term + sum(products.values())
        solved.append(inverse_numeric * rhs)
        term_breakdown[str(derivative_order)] = {
            "formula": (
                f"N*(sqrt(11)*W{derivative_order}+sum_{{k=1}}^{derivative_order} "
                f"binomial({derivative_order},k)*Ak*F{derivative_order}-k)"
            ),
            "remainder_term_numeric": float(sp.N(remainder_term, 18)),
            "coefficient_product_terms_numeric": {
                key: float(sp.N(value, 18)) for key, value in products.items()
            },
        }
    if not all(_positive(value) for value in solved):
        raise QuarticSolvedSourceMoserError("solved-source derivative is not positive")
    return solved, {
        "recurrence": (
            "F_n <= N*(sqrt(11)*W_n + sum_{k=1}^n binomial(n,k)*A_k*F_{n-k})"
        ),
        "input_norm": "153-coordinate-atom component l_infinity",
        "output_norm": "11-acceleration-vector Euclidean 2-norm",
        "orders": list(range(order + 1)),
        "2_norm_envelopes_numeric": {
            str(derivative_order): float(sp.N(value, 18))
            for derivative_order, value in enumerate(solved)
        },
        "term_breakdown": term_breakdown,
    }


def _certify_candidate(
    moser: dict[str, Any],
    pde: dict[str, Any],
    tube: dict[str, Any],
    euler: dict[str, Any],
    coordinate_jet: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates = (moser, pde, tube, euler)
    candidate_id = moser.get("candidate_id")
    if any(item.get("candidate_id") != candidate_id for item in candidates):
        raise QuarticSolvedSourceMoserError("candidate ID mismatch")
    if any(item.get("coefficients") != moser.get("coefficients") for item in candidates[1:]):
        raise QuarticSolvedSourceMoserError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_quasilinear_coefficient_derivative_envelopes",
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "pass_complete_coordinate_tube_euler_remainder_majorant",
    )
    if any(item.get("status") != status for item, status in zip(candidates, expected_statuses)):
        raise QuarticSolvedSourceMoserError("candidate prerequisite failed")
    order = int(config["required_Frechet_majorant_order"])
    state_dimension = int(config["state_dimension"])
    raw_a = moser["raw_Frechet_derivative_2_norm_envelopes"]["A"]
    a_derivatives, a_evidence = _coordinate_a_derivatives(raw_a, coordinate_jet)
    inverse_a = sp.sympify(moser["inverse_time_block_2_norm_rational_ceiling"])
    if inverse_a != sp.sympify(pde["uniform_bounds"]["A_inverse_2_upper_used"]):
        raise QuarticSolvedSourceMoserError("inverse time-block ceiling mismatch")
    if inverse_a != sp.sympify(euler["time_block_inverse_2_upper"]):
        raise QuarticSolvedSourceMoserError("Euler inverse time-block ceiling mismatch")
    solved, solved_evidence = _solved_source_derivatives(
        inverse_a,
        a_derivatives,
        euler["Euler_remainder_Frechet_derivative_uppers"],
        order,
        state_dimension,
    )
    existing_order_zero = float(euler["solved_acceleration_component_upper_numeric"])
    recomputed_order_zero = float(sp.N(solved[0], 18))
    relative_residual = abs(recomputed_order_zero - existing_order_zero) / max(
        recomputed_order_zero, existing_order_zero
    )
    if relative_residual > 1e-12:
        raise QuarticSolvedSourceMoserError("order-zero acceleration cross-check failed")
    return {
        "schema_version": "sigma-quartic-solved-source-moser-certificate-1.0",
        "status": "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "candidate_id": candidate_id,
        "coefficients": moser["coefficients"],
        "coordinate_component_radius": tube["coordinate_component_radius"],
        "inverse_time_block_2_norm_upper": str(inverse_a),
        "coordinate_time_block_derivatives": a_evidence,
        "Euler_remainder_derivative_source": {
            "orders": list(range(order + 1)),
            "upstream_candidate_status": euler["status"],
        },
        "solved_source_Frechet_derivatives": solved_evidence,
        "order_zero_acceleration_crosscheck": {
            "upstream_numeric": existing_order_zero,
            "recomputed_numeric": recomputed_order_zero,
            "relative_residual": relative_residual,
        },
        "claim": (
            "The solved 11-component acceleration source has finite coordinate-atom "
            "Frechet envelopes through order four on the common coordinate tube."
        ),
        "remaining_gate": "full_symmetrizer_derivatives_commuted_energy_and_lifespan",
        "scope": (
            "This closes the A^{-1}W source-composition derivative gate. It does not yet "
            "differentiate the full 55-state symmetrizer, close the commuted gauge/constraint "
            "energy inequality, prove a quantitative lifespan, or add matter."
        ),
    }


def run_quartic_solved_source_moser_campaign(
    moser_campaign: dict[str, Any],
    nonquasilinear_pde_campaign: dict[str, Any],
    coordinate_tube_campaign: dict[str, Any],
    euler_remainder_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticSolvedSourceMoserError("unsupported campaign schema_version")
        if int(config["required_Frechet_majorant_order"]) != 4:
            raise QuarticSolvedSourceMoserError("solved source requires order four")
        if int(config["state_dimension"]) != 11:
            raise QuarticSolvedSourceMoserError("solved source requires eleven Euler rows")
        expected_campaign_statuses = (
            (
                moser_campaign,
                "pass_all_12_quasilinear_coefficient_derivative_envelopes",
            ),
            (
                nonquasilinear_pde_campaign,
                "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
            ),
            (
                coordinate_tube_campaign,
                "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            ),
            (
                euler_remainder_campaign,
                "pass_all_12_complete_coordinate_tube_euler_remainder_majorants",
            ),
        )
        if any(campaign.get("status") != status for campaign, status in expected_campaign_statuses):
            raise QuarticSolvedSourceMoserError("campaign prerequisite failed")
        pde_hash = nonquasilinear_pde_campaign.get("content_sha256")
        tube_hash = coordinate_tube_campaign.get("content_sha256")
        if nonquasilinear_pde_campaign.get("upstream_sha256", {}).get("moser") != (
            moser_campaign.get("content_sha256")
        ):
            raise QuarticSolvedSourceMoserError("Moser provenance mismatch")
        if coordinate_tube_campaign.get("nonquasilinear_pde_campaign_sha256") != pde_hash:
            raise QuarticSolvedSourceMoserError("coordinate-tube provenance mismatch")
        if (
            euler_remainder_campaign.get("nonquasilinear_pde_campaign_sha256")
            != pde_hash
            or euler_remainder_campaign.get("coordinate_tube_campaign_sha256")
            != tube_hash
        ):
            raise QuarticSolvedSourceMoserError("Euler-remainder provenance mismatch")
        control_passed, control = generic_quadratic_composition_control()
        if not control_passed:
            raise QuarticSolvedSourceMoserError("generic composition control failed")
        coordinate_jet = _coordinate_jet_derivatives(
            coordinate_tube_campaign, int(config["required_Frechet_majorant_order"])
        )
        record_sets = [
            _candidate_records(campaign)
            for campaign in (
                moser_campaign,
                nonquasilinear_pde_campaign,
                coordinate_tube_campaign,
                euler_remainder_campaign,
            )
        ]
        expected = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(record_sets[0])
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in record_sets[1:]
        ):
            raise QuarticSolvedSourceMoserError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                record_sets[0][candidate_id],
                record_sets[1][candidate_id],
                record_sets[2][candidate_id],
                record_sets[3][candidate_id],
                coordinate_jet,
                config,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "errors": [],
            "upstream_sha256": {
                "moser": moser_campaign.get("content_sha256"),
                "nonquasilinear_pde": pde_hash,
                "coordinate_tube": tube_hash,
                "euler_remainder": euler_remainder_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_composition_control": control,
            "coordinate_jet_Frechet_envelopes": coordinate_jet,
            "counts": {
                "selected": len(certificates),
                "solved_source_moser_envelopes_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have fourth-order coordinate-atom Frechet "
                "envelopes for the complete solved acceleration source A^{-1}W."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticSolvedSourceMoserError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "solved_source_moser_envelopes_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_solved_source_moser_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
