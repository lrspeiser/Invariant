from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    _first_order_generalized_pencil,
    quartic_horndeski_baseline_riesz_symmetrizer_control,
)
from .quartic_quasilinear_moser_campaign import (
    _jet_and_direction_symbols,
    _matrix_derivative_tensor_bound,
    _symbol_data,
)

SCHEMA_VERSION = "sigma-quartic-full-symmetrizer-moser-campaign-1.0"


class QuarticFullSymmetrizerMoserError(ValueError):
    """Raised when full-state symmetrizer derivatives cannot be bounded."""


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


def _multinomial(total: int, *parts: int) -> int:
    if sum(parts) != total:
        raise QuarticFullSymmetrizerMoserError("invalid multinomial partition")
    result = factorial(total)
    for part in parts:
        result //= factorial(part)
    return result


@cache
def generic_symmetrizer_derivative_control() -> tuple[bool, dict[str, Any]]:
    """Verify the resolvent, triple-product, and fourth-order chain recurrences."""

    u, z = sp.symbols("u z", real=True, finite=True)
    m = sp.symbols("m0:5", real=True, finite=True)
    matrix_symbol = sum(m[order] * u**order / factorial(order) for order in range(5))
    resolvent = 1 / (z - matrix_symbol)
    resolvent_residuals: dict[str, str] = {}
    for order in range(1, 5):
        recurrence = resolvent * sum(
            sp.binomial(order, derivative_order)
            * sp.diff(matrix_symbol, u, derivative_order)
            * sp.diff(resolvent, u, order - derivative_order)
            for derivative_order in range(1, order + 1)
        )
        residual = sp.factor(sp.diff(resolvent, u, order) - recurrence)
        resolvent_residuals[str(order)] = str(residual)

    left = sp.symbols("l0:5", real=True, finite=True)
    middle = sp.symbols("k0:5", real=True, finite=True)
    right = sp.symbols("q0:5", real=True, finite=True)
    left_series = sum(left[order] * u**order / factorial(order) for order in range(5))
    middle_series = sum(
        middle[order] * u**order / factorial(order) for order in range(5)
    )
    right_series = sum(right[order] * u**order / factorial(order) for order in range(5))
    triple = left_series * middle_series * right_series
    triple_residuals: dict[str, str] = {}
    for order in range(5):
        recurrence = sum(
            _multinomial(order, a_order, b_order, order - a_order - b_order)
            * left[a_order]
            * middle[b_order]
            * right[order - a_order - b_order]
            for a_order in range(order + 1)
            for b_order in range(order - a_order + 1)
        )
        triple_residuals[str(order)] = str(
            sp.factor(sp.diff(triple, u, order).subs(u, 0) - recurrence)
        )

    y = sp.Symbol("y", real=True, finite=True)
    s = sp.symbols("s0:5", real=True, finite=True)
    j = sp.symbols("j0:5", real=True, finite=True)
    outer = sum(s[order] * y**order / factorial(order) for order in range(5))
    inner = sum(j[order] * u**order / factorial(order) for order in range(5))
    composed = outer.subs(y, inner - j[0])
    expected_chain = {
        1: s[1] * j[1],
        2: s[2] * j[1] ** 2 + s[1] * j[2],
        3: s[3] * j[1] ** 3 + 3 * s[2] * j[1] * j[2] + s[1] * j[3],
        4: s[4] * j[1] ** 4
        + 6 * s[3] * j[1] ** 2 * j[2]
        + s[2] * (3 * j[2] ** 2 + 4 * j[1] * j[3])
        + s[1] * j[4],
    }
    chain_residuals = {
        str(order): str(
            sp.factor(
                sp.diff(composed, u, order).subs(u, 0) - expected_chain[order]
            )
        )
        for order in range(1, 5)
    }
    corrupted_fourth = sp.factor(
        sp.diff(composed, u, 4).subs(u, 0)
        - (
            s[4] * j[1] ** 4
            + 5 * s[3] * j[1] ** 2 * j[2]
            + s[2] * (3 * j[2] ** 2 + 4 * j[1] * j[3])
            + s[1] * j[4]
        )
    )
    corrupted_witness = corrupted_fourth.subs(
        {
            **{symbol: index + 2 for index, symbol in enumerate(s)},
            **{symbol: index + 7 for index, symbol in enumerate(j)},
        }
    )
    passed = bool(
        set(resolvent_residuals.values()) == {"0"}
        and set(triple_residuals.values()) == {"0"}
        and set(chain_residuals.values()) == {"0"}
        and corrupted_witness != 0
    )
    return passed, {
        "control": "full symmetrizer derivative recurrence through order four",
        "resolvent_identity": (
            "R_n=R*sum_{k=1}^n binomial(n,k) M_k R_{n-k}, "
            "R=(zI-M)^{-1}"
        ),
        "resolvent_residuals": resolvent_residuals,
        "triple_product_identity": (
            "D^n(L K Q)=sum_{a+b+c=n} n!/(a!b!c!) L_a K_b Q_c"
        ),
        "triple_product_residuals": triple_residuals,
        "fourth_order_chain_rule": (
            "S4 J1^4+6 S3 J1^2 J2+S2(3 J2^2+4 J1 J3)+S1 J4"
        ),
        "chain_rule_residuals": chain_residuals,
        "negative_control": {
            "corruption": "replace the fourth-order coefficient 6 on S3*J1^2*J2 by 5",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": corrupted_witness != 0,
        },
        "passed": passed,
        "scope": (
            "Exact scalar representatives verify the multiplicities used after taking "
            "operator norms of the matrix resolvent, product, and composition formulas."
        ),
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["candidate_id"]: item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


@cache
def _physical_h_star_derivatives(
    a10: str, c20: str, m2: str, radius: str, order: int
) -> tuple[sp.Expr, ...]:
    data = _symbol_data()
    jets, directions = _jet_and_direction_symbols(data)
    all_variables = jets + directions
    bounds = {
        **{symbol: sp.sympify(radius) for symbol in jets},
        **{symbol: sp.Integer(1) for symbol in directions},
    }
    substitutions = {
        data["alpha"]: sp.sympify(a10),
        data["c20"]: sp.sympify(c20),
        data["m2"]: sp.sympify(m2),
    }
    action_first_order = _first_order_generalized_pencil(
        data["action_symbol"], data["xi_lower"][0]
    )
    action_a = action_first_order["A"].subs(substitutions).applyfunc(sp.expand)
    action_b = action_first_order["B"].subs(substitutions).applyfunc(sp.expand)
    h_star = action_b.row_join(action_a).col_join(
        action_a.row_join(sp.zeros(11))
    )
    derivatives = [sp.Rational(9, 8)]
    for derivative_order in range(1, order + 1):
        if derivative_order > 2:
            derivatives.append(sp.Integer(0))
            continue
        bound, _ = _matrix_derivative_tensor_bound(
            h_star,
            derivative_order,
            jets,
            all_variables,
            bounds,
        )
        derivatives.append(bound)
    return tuple(derivatives)


def _inverse_product_bounds(
    inverse_upper: sp.Expr,
    coefficient_derivatives: list[sp.Expr],
    right_derivatives: list[sp.Expr],
    order: int,
) -> list[sp.Expr]:
    inverse_numeric = sp.N(inverse_upper, 80)
    coefficient_numeric = [sp.N(value, 80) for value in coefficient_derivatives]
    right_numeric = [sp.N(value, 80) for value in right_derivatives]
    result = [inverse_numeric * right_numeric[0]]
    for derivative_order in range(1, order + 1):
        rhs = right_numeric[derivative_order] + sum(
            comb(derivative_order, coefficient_order)
            * coefficient_numeric[coefficient_order]
            * result[derivative_order - coefficient_order]
            for coefficient_order in range(1, derivative_order + 1)
        )
        result.append(inverse_numeric * rhs)
    return result


def _resolvent_and_projector_derivatives(
    baseline_resolvent: sp.Expr,
    companion_perturbation: sp.Expr,
    companion_derivatives: list[sp.Expr],
    contour_radius: sp.Expr,
    order: int,
) -> tuple[list[sp.Expr], list[sp.Expr], sp.Expr]:
    denominator = 1 - baseline_resolvent * companion_perturbation
    if not _positive(denominator):
        raise QuarticFullSymmetrizerMoserError("candidate resolvent Neumann margin failed")
    candidate_resolvent = sp.N(baseline_resolvent / denominator, 80)
    companion_numeric = [sp.N(value, 80) for value in companion_derivatives]
    resolvent = [candidate_resolvent]
    for derivative_order in range(1, order + 1):
        rhs = sum(
            comb(derivative_order, coefficient_order)
            * companion_numeric[coefficient_order]
            * resolvent[derivative_order - coefficient_order]
            for coefficient_order in range(1, derivative_order + 1)
        )
        resolvent.append(candidate_resolvent * rhs)
    projector = [sp.Integer(0)] + [
        sp.N(contour_radius, 80) * resolvent[derivative_order]
        for derivative_order in range(1, order + 1)
    ]
    return resolvent, projector, denominator


def _k22_derivatives(
    projector_zero_by_group: dict[str, sp.Expr],
    projector_derivatives: list[sp.Expr],
    h_star_derivatives: tuple[sp.Expr, ...],
    k22_zero: sp.Expr,
    order: int,
) -> list[sp.Expr]:
    result = [sp.N(k22_zero, 80)]
    physical_groups = {"1", "-1"}
    for derivative_order in range(1, order + 1):
        total = sp.Float(0, 80)
        for group, projector_zero in projector_zero_by_group.items():
            for left_order in range(derivative_order + 1):
                for middle_order in range(derivative_order - left_order + 1):
                    right_order = derivative_order - left_order - middle_order
                    if middle_order > 0 and group not in physical_groups:
                        continue
                    left = (
                        sp.N(projector_zero, 80)
                        if left_order == 0
                        else projector_derivatives[left_order]
                    )
                    right = (
                        sp.N(projector_zero, 80)
                        if right_order == 0
                        else projector_derivatives[right_order]
                    )
                    if middle_order == 0:
                        middle = sp.Rational(9, 8) if group in physical_groups else 1
                    else:
                        middle = sp.N(h_star_derivatives[middle_order], 80)
                    total += (
                        _multinomial(
                            derivative_order,
                            left_order,
                            middle_order,
                            right_order,
                        )
                        * left
                        * middle
                        * right
                    )
        result.append(total)
    return result


def _triple_product_bounds(
    left: list[sp.Expr], middle: list[sp.Expr], right: list[sp.Expr], order: int
) -> list[sp.Expr]:
    result: list[sp.Expr] = []
    for derivative_order in range(order + 1):
        total = sp.Float(0, 80)
        for left_order in range(derivative_order + 1):
            for middle_order in range(derivative_order - left_order + 1):
                right_order = derivative_order - left_order - middle_order
                total += (
                    _multinomial(
                        derivative_order,
                        left_order,
                        middle_order,
                        right_order,
                    )
                    * left[left_order]
                    * middle[middle_order]
                    * right[right_order]
                )
        result.append(total)
    return result


def _compose_with_coordinate_jet(
    symmetrizer_derivatives: list[sp.Expr], coordinate_jet: dict[str, Any]
) -> list[sp.Expr]:
    j = [sp.Integer(0)] + [
        sp.N(
            sp.sympify(coordinate_jet["envelopes"][str(order)]["exact"]),
            80,
        )
        for order in range(1, 5)
    ]
    s = symmetrizer_derivatives
    return [
        s[0],
        s[1] * j[1],
        s[2] * j[1] ** 2 + s[1] * j[2],
        s[3] * j[1] ** 3 + 3 * s[2] * j[1] * j[2] + s[1] * j[3],
        s[4] * j[1] ** 4
        + 6 * s[3] * j[1] ** 2 * j[2]
        + s[2] * (3 * j[2] ** 2 + 4 * j[1] * j[3])
        + s[1] * j[4],
    ]


def _numeric_hierarchy(values: list[sp.Expr]) -> dict[str, float]:
    return {
        str(order): float(sp.N(value, 18)) for order, value in enumerate(values)
    }


def _certify_candidate(
    symmetrizer: dict[str, Any],
    moser: dict[str, Any],
    pde: dict[str, Any],
    tube: dict[str, Any],
    solved_source: dict[str, Any],
    baseline_contract: dict[str, Any],
    coordinate_jet: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates = (symmetrizer, moser, pde, tube, solved_source)
    candidate_id = symmetrizer.get("candidate_id")
    if any(item.get("candidate_id") != candidate_id for item in candidates):
        raise QuarticFullSymmetrizerMoserError("candidate ID mismatch")
    if any(
        item.get("coefficients") != symmetrizer.get("coefficients")
        for item in candidates[1:]
    ):
        raise QuarticFullSymmetrizerMoserError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_uniform_local_jet_strong_hyperbolicity",
        "pass_quasilinear_coefficient_derivative_envelopes",
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
    )
    if any(
        item.get("status") != status
        for item, status in zip(candidates, expected_statuses)
    ):
        raise QuarticFullSymmetrizerMoserError("candidate prerequisite failed")
    order = int(config["required_Frechet_majorant_order"])
    contour_radius = sp.sympify(config["contour_radius"])
    if contour_radius != sp.sympify(baseline_contract["contour_radius"]):
        raise QuarticFullSymmetrizerMoserError("Riesz contour radius mismatch")

    companion = [
        sp.sympify(moser["companion_Frechet_derivative_2_norm_envelopes"][str(n)])
        for n in range(order + 1)
    ]
    baseline_resolvent = sp.sympify(
        baseline_contract["maximum_resolvent_upper_bound"]
    )
    companion_perturbation = sp.sympify(
        symmetrizer["uniform_matrix_bounds"]["companion_2_norm_perturbation_upper"]
    )
    resolvent, projector, resolvent_margin = _resolvent_and_projector_derivatives(
        baseline_resolvent,
        companion_perturbation,
        companion,
        contour_radius,
        order,
    )
    projector_zero_by_group = {
        name: sp.sympify(value)
        for name, value in pde["uniform_bounds"]["derivation"][
            "candidate_projector_2_norm_uppers"
        ].items()
    }
    coefficients = symmetrizer["coefficients"]
    h_star = _physical_h_star_derivatives(
        coefficients["a10"],
        coefficients["c20"],
        coefficients["m2"],
        symmetrizer["domain"]["normalized_local_jet_component_abs"],
        order,
    )
    k22 = _k22_derivatives(
        projector_zero_by_group,
        projector,
        h_star,
        sp.sympify(pde["uniform_bounds"]["K22_2_upper"]),
        order,
    )
    inverse_m22 = _inverse_product_bounds(
        sp.sympify(pde["uniform_bounds"]["M22_inverse_2_upper"]),
        companion,
        [sp.Integer(1)] + [sp.Integer(0)] * order,
        order,
    )
    raw = moser["raw_Frechet_derivative_2_norm_envelopes"]
    raw_a = [sp.sympify(raw["A"][str(n)]) for n in range(order + 1)]
    raw_c = [sp.sympify(raw["C"][str(n)]) for n in range(order + 1)]
    transverse_l = _inverse_product_bounds(
        sp.sympify(moser["inverse_time_block_2_norm_rational_ceiling"]),
        raw_a,
        raw_c,
        order,
    )
    transverse_l[0] = sp.N(sp.sympify(pde["uniform_bounds"]["L_2_upper"]), 80)
    inverse_m22[0] = sp.N(
        sp.sympify(pde["uniform_bounds"]["M22_inverse_2_upper"]), 80
    )
    k22[0] = sp.N(sp.sympify(pde["uniform_bounds"]["K22_2_upper"]), 80)
    cross_f = _triple_product_bounds(transverse_l, k22, inverse_m22, order)
    cross_f[0] = sp.N(sp.sympify(pde["uniform_bounds"]["F_2_upper"]), 80)
    k55 = [sp.N(sp.sympify(pde["uniform_bounds"]["K55_2_upper"]), 80)] + [
        2 * cross_f[n] + k22[n] for n in range(1, order + 1)
    ]
    coordinate_k55 = _compose_with_coordinate_jet(k55, coordinate_jet)
    commutator_multipliers = [sp.Integer(0)] + [
        sum(
            comb(n, derivative_order) * coordinate_k55[derivative_order]
            for derivative_order in range(1, n + 1)
        )
        for n in range(1, order + 1)
    ]
    if not all(
        _positive(value)
        for hierarchy in (
            resolvent,
            projector[1:],
            k22,
            inverse_m22,
            transverse_l,
            cross_f,
            k55,
            coordinate_k55,
            commutator_multipliers[1:],
        )
        for value in hierarchy
    ) or not all(_nonnegative(value) for value in h_star):
        raise QuarticFullSymmetrizerMoserError("a symmetrizer derivative bound is not positive")
    return {
        "schema_version": "sigma-quartic-full-symmetrizer-moser-certificate-1.0",
        "status": "pass_full_K55_coordinate_atom_C4_derivative_envelopes",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "domain": {
            "coordinate_component_radius": tube["coordinate_component_radius"],
            "covariant_component_radius": tube["covariant_component_radius"],
            "spatial_direction": symmetrizer["domain"]["spatial_direction"],
        },
        "Riesz_resolvent": {
            "contour_radius": str(contour_radius),
            "baseline_resolvent_2_upper": str(baseline_resolvent),
            "candidate_Neumann_denominator_lower": str(resolvent_margin),
            "Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
                resolvent
            ),
        },
        "Riesz_projector_Frechet_derivative_2_norm_envelopes_numeric": {
            str(n): float(sp.N(projector[n], 18)) for n in range(1, order + 1)
        },
        "physical_H_star_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
            list(h_star)
        ),
        "K22_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(k22),
        "M22_inverse_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
            inverse_m22
        ),
        "L_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
            transverse_l
        ),
        "F_cross_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
            cross_f
        ),
        "K55_covariant_jet_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
            k55
        ),
        "K55_coordinate_atom_Frechet_derivative_2_norm_envelopes_numeric": _numeric_hierarchy(
            coordinate_k55
        ),
        "Leibniz_commutator_coefficient_multipliers_numeric": {
            str(n): float(sp.N(commutator_multipliers[n], 18))
            for n in range(1, order + 1)
        },
        "energy_equivalence": {
            "K55_2_lower": pde["uniform_bounds"]["K55_2_lower"],
            "K55_2_upper": pde["uniform_bounds"]["K55_2_upper"],
            "K55_inverse_2_upper": str(
                1 / sp.sympify(pde["uniform_bounds"]["K55_2_lower"])
            ),
        },
        "claim": (
            "The actual lifted 55-state Riesz symmetrizer has finite state and "
            "coordinate-atom Frechet envelopes through order four."
        ),
        "remaining_gate": (
            "direction_symbol_derivatives_Sobolev_embedding_constants_and_energy_lifespan"
        ),
        "scope": (
            "This differentiates the Riesz projectors, physical H_star blocks, M22 inverse, "
            "lift coupling L, cross block F, and complete K55. The reported Leibniz values "
            "are coefficient multipliers, not a closed Sobolev energy estimate: direction-"
            "symbol derivatives, Sobolev embedding/product constants, boundary terms, a "
            "lifespan, and matter remain unresolved."
        ),
    }


def run_quartic_full_symmetrizer_moser_campaign(
    symmetrizer_campaign: dict[str, Any],
    moser_campaign: dict[str, Any],
    nonquasilinear_pde_campaign: dict[str, Any],
    coordinate_tube_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticFullSymmetrizerMoserError("unsupported campaign schema_version")
        order = int(config["required_Frechet_majorant_order"])
        if order != 4:
            raise QuarticFullSymmetrizerMoserError("full symmetrizer requires order four")
        if int(config["full_state_dimension"]) != 55:
            raise QuarticFullSymmetrizerMoserError("full symmetrizer requires 55 states")
        expected_statuses = (
            (
                symmetrizer_campaign,
                "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes",
            ),
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
                solved_source_campaign,
                "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            ),
        )
        if any(campaign.get("status") != status for campaign, status in expected_statuses):
            raise QuarticFullSymmetrizerMoserError("campaign prerequisite failed")
        pde_hash = nonquasilinear_pde_campaign.get("content_sha256")
        tube_hash = coordinate_tube_campaign.get("content_sha256")
        if nonquasilinear_pde_campaign.get("upstream_sha256", {}).get(
            "symmetrizer"
        ) != symmetrizer_campaign.get("content_sha256") or (
            nonquasilinear_pde_campaign.get("upstream_sha256", {}).get("moser")
            != moser_campaign.get("content_sha256")
        ):
            raise QuarticFullSymmetrizerMoserError("PDE provenance mismatch")
        if coordinate_tube_campaign.get("nonquasilinear_pde_campaign_sha256") != pde_hash:
            raise QuarticFullSymmetrizerMoserError("coordinate-tube provenance mismatch")
        solved_upstream = solved_source_campaign.get("upstream_sha256", {})
        if (
            solved_upstream.get("moser") != moser_campaign.get("content_sha256")
            or solved_upstream.get("nonquasilinear_pde") != pde_hash
            or solved_upstream.get("coordinate_tube") != tube_hash
        ):
            raise QuarticFullSymmetrizerMoserError("solved-source provenance mismatch")
        control_passed, control = generic_symmetrizer_derivative_control()
        if not control_passed:
            raise QuarticFullSymmetrizerMoserError("generic derivative control failed")
        baseline_passed, baseline = quartic_horndeski_baseline_riesz_symmetrizer_control()
        if not baseline_passed:
            raise QuarticFullSymmetrizerMoserError("baseline Riesz control failed")
        baseline_contract = baseline["quantitative_physical_group_perturbation_contract"]
        coordinate_jet = solved_source_campaign["coordinate_jet_Frechet_envelopes"]
        record_sets = [
            _candidate_records(campaign)
            for campaign in (
                symmetrizer_campaign,
                moser_campaign,
                nonquasilinear_pde_campaign,
                coordinate_tube_campaign,
                solved_source_campaign,
            )
        ]
        expected = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(record_sets[0])
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in record_sets[1:]
        ):
            raise QuarticFullSymmetrizerMoserError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                record_sets[0][candidate_id],
                record_sets[1][candidate_id],
                record_sets[2][candidate_id],
                record_sets[3][candidate_id],
                record_sets[4][candidate_id],
                baseline_contract,
                coordinate_jet,
                config,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
            "errors": [],
            "upstream_sha256": {
                "symmetrizer": symmetrizer_campaign.get("content_sha256"),
                "moser": moser_campaign.get("content_sha256"),
                "nonquasilinear_pde": pde_hash,
                "coordinate_tube": tube_hash,
                "solved_source": solved_source_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_symmetrizer_derivative_control": control,
            "counts": {
                "selected": len(certificates),
                "full_K55_C4_derivative_envelopes_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have fourth-order coordinate-atom derivative "
                "envelopes for the actual lifted 55-state Riesz symmetrizer."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticFullSymmetrizerMoserError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "full_K55_C4_derivative_envelopes_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_full_symmetrizer_moser_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
