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
from .quartic_full_symmetrizer_moser_campaign import (
    _candidate_records,
    _multinomial,
)
from .quartic_quasilinear_moser_campaign import (
    _jet_and_direction_symbols,
    _symbol_data,
)

SCHEMA_VERSION = "sigma-quartic-symmetrizer-symbol-moser-campaign-1.0"


class QuarticSymmetrizerSymbolMoserError(ValueError):
    """Raised when mixed state-direction symmetrizer bounds cannot be certified."""


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


def _pairs(maximum_total_order: int) -> list[tuple[int, int]]:
    return [
        (state_order, direction_order)
        for total_order in range(maximum_total_order + 1)
        for state_order in range(total_order + 1)
        for direction_order in (total_order - state_order,)
    ]


def _key(state_order: int, direction_order: int) -> str:
    return f"{state_order},{direction_order}"


def _falling_factorial(degree: int, order: int) -> int:
    if degree < order:
        return 0
    return factorial(degree) // factorial(degree - order)


@cache
def generic_bivariate_symbol_derivative_control() -> tuple[bool, dict[str, Any]]:
    """Verify bivariate inverse and triple-product recurrences through total order four."""

    u, v, z = sp.symbols("u v z", real=True, finite=True)
    maximum_order = 4
    pairs = _pairs(maximum_order)

    def exact_polynomial(offset: int) -> sp.Expr:
        return sum(
            (offset + index + 1)
            * u**a
            * v**b
            / (factorial(a) * factorial(b))
            for index, (a, b) in enumerate(pairs)
        )

    matrix_symbol = exact_polynomial(1)
    resolvent = 1 / (z - matrix_symbol)
    witness = {z: 101, u: sp.Rational(1, 10), v: sp.Rational(-1, 13)}
    inverse_residuals: dict[str, str] = {}
    for state_order, direction_order in pairs[1:]:
        recurrence = resolvent * sum(
            comb(state_order, left_state)
            * comb(direction_order, left_direction)
            * sp.diff(
                matrix_symbol,
                u,
                left_state,
                v,
                left_direction,
            )
            * sp.diff(
                resolvent,
                u,
                state_order - left_state,
                v,
                direction_order - left_direction,
            )
            for left_state in range(state_order + 1)
            for left_direction in range(direction_order + 1)
            if (left_state, left_direction) != (0, 0)
        )
        residual = sp.cancel(
            (
                sp.diff(resolvent, u, state_order, v, direction_order)
                - recurrence
            ).subs(witness)
        )
        inverse_residuals[_key(state_order, direction_order)] = str(residual)

    left = exact_polynomial(20)
    middle = exact_polynomial(40)
    right = exact_polynomial(60)
    triple = left * middle * right
    triple_residuals: dict[str, str] = {}
    for state_order, direction_order in pairs:
        recurrence = sp.Integer(0)
        for left_state in range(state_order + 1):
            for middle_state in range(state_order - left_state + 1):
                right_state = state_order - left_state - middle_state
                for left_direction in range(direction_order + 1):
                    for middle_direction in range(
                        direction_order - left_direction + 1
                    ):
                        right_direction = (
                            direction_order - left_direction - middle_direction
                        )
                        recurrence += (
                            _multinomial(
                                state_order,
                                left_state,
                                middle_state,
                                right_state,
                            )
                            * _multinomial(
                                direction_order,
                                left_direction,
                                middle_direction,
                                right_direction,
                            )
                            * sp.diff(
                                left, u, left_state, v, left_direction
                            ).subs(witness)
                            * sp.diff(
                                middle,
                                u,
                                middle_state,
                                v,
                                middle_direction,
                            ).subs(witness)
                            * sp.diff(
                                right, u, right_state, v, right_direction
                            ).subs(witness)
                        )
        residual = sp.cancel(
            sp.diff(triple, u, state_order, v, direction_order).subs(witness)
            - recurrence
        )
        triple_residuals[_key(state_order, direction_order)] = str(residual)

    target = (1, 2)
    correct = resolvent * sum(
        comb(target[0], left_state)
        * comb(target[1], left_direction)
        * sp.diff(matrix_symbol, u, left_state, v, left_direction)
        * sp.diff(
            resolvent,
            u,
            target[0] - left_state,
            v,
            target[1] - left_direction,
        )
        for left_state in range(target[0] + 1)
        for left_direction in range(target[1] + 1)
        if (left_state, left_direction) != (0, 0)
    )
    corrupted = resolvent * sum(
        comb(target[0], left_state)
        * (1 if left_direction == 1 else comb(target[1], left_direction))
        * sp.diff(matrix_symbol, u, left_state, v, left_direction)
        * sp.diff(
            resolvent,
            u,
            target[0] - left_state,
            v,
            target[1] - left_direction,
        )
        for left_state in range(target[0] + 1)
        for left_direction in range(target[1] + 1)
        if (left_state, left_direction) != (0, 0)
    )
    corrupted_witness = sp.cancel((correct - corrupted).subs(witness))
    passed = bool(
        set(inverse_residuals.values()) == {"0"}
        and set(triple_residuals.values()) == {"0"}
        and corrupted_witness != 0
    )
    return passed, {
        "control": "mixed state-direction symbol derivative recurrence",
        "multiindices": [_key(*pair) for pair in _pairs(maximum_order)],
        "inverse_recurrence": (
            "R_(a,b)=R_00 sum_(i,j)!=(0,0) binomial(a,i) binomial(b,j) "
            "M_(i,j) R_(a-i,b-j)"
        ),
        "inverse_residuals": inverse_residuals,
        "triple_product_recurrence": (
            "Separate state and direction multinomials multiply for D_U^a D_n^b(L K Q)."
        ),
        "triple_product_residuals": triple_residuals,
        "negative_control": {
            "corruption": "replace binomial(2,1)=2 by 1 in the (1,2) inverse recurrence",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": corrupted_witness != 0,
        },
        "passed": passed,
        "scope": (
            "Exact scalar representatives verify every bivariate product multiplicity through "
            "total order four before operator norms are applied."
        ),
    }


def _mixed_polynomial_matrix_bound(
    matrix: sp.Matrix,
    state_order: int,
    direction_order: int,
    jets: list[sp.Symbol],
    directions: list[sp.Symbol],
    coefficient_symbols: list[sp.Symbol],
    state_radius: sp.Expr,
) -> sp.Expr:
    """Exact ordered-tensor l1 envelope using monomial falling factorials."""

    variables = jets + directions + coefficient_symbols
    total = sp.Integer(0)
    jet_end = len(jets)
    direction_end = jet_end + len(directions)
    for entry in matrix:
        polynomial = sp.Poly(sp.expand(entry), *variables)
        for powers, coefficient in polynomial.terms():
            state_degree = sum(powers[:jet_end])
            direction_degree = sum(powers[jet_end:direction_end])
            state_factor = _falling_factorial(state_degree, state_order)
            direction_factor = _falling_factorial(
                direction_degree, direction_order
            )
            if state_factor == 0 or direction_factor == 0:
                continue
            total += (
                abs(coefficient)
                * state_factor
                * direction_factor
                * state_radius ** (state_degree - state_order)
            )
    return sp.factor(total)


@cache
def _uniform_raw_mixed_envelopes(
    state_radius: str, maximum_total_order: int
) -> dict[str, dict[tuple[int, int], sp.Expr]]:
    data = _symbol_data()
    jets, directions = _jet_and_direction_symbols(data)
    coefficient_symbols = [data["alpha"], data["c20"]]
    matrices = {
        name: data["first_order"][name].subs(data["m2"], 1).applyfunc(sp.expand)
        for name in ("A", "B", "C")
    }
    action_first_order = _first_order_generalized_pencil(
        data["action_symbol"], data["xi_lower"][0]
    )
    action_a = action_first_order["A"].subs(data["m2"], 1).applyfunc(sp.expand)
    action_b = action_first_order["B"].subs(data["m2"], 1).applyfunc(sp.expand)
    matrices["H_star"] = action_b.row_join(action_a).col_join(
        action_a.row_join(sp.zeros(11))
    )
    radius = sp.sympify(state_radius)
    return {
        name: {
            pair: _mixed_polynomial_matrix_bound(
                matrix,
                pair[0],
                pair[1],
                jets,
                directions,
                coefficient_symbols,
                radius,
            )
            for pair in _pairs(maximum_total_order)
        }
        for name, matrix in matrices.items()
    }


def _bivariate_inverse_product(
    inverse_zero: sp.Expr,
    coefficient: dict[tuple[int, int], sp.Expr],
    right: dict[tuple[int, int], sp.Expr],
    maximum_total_order: int,
    *,
    product_zero: sp.Expr | None = None,
) -> dict[tuple[int, int], sp.Expr]:
    inverse = sp.N(inverse_zero, 80)
    result: dict[tuple[int, int], sp.Expr] = {
        (0, 0): (
            sp.N(product_zero, 80)
            if product_zero is not None
            else inverse * sp.N(right[(0, 0)], 80)
        )
    }
    for state_order, direction_order in _pairs(maximum_total_order)[1:]:
        rhs = sp.N(right[(state_order, direction_order)], 80)
        for left_state in range(state_order + 1):
            for left_direction in range(direction_order + 1):
                if (left_state, left_direction) == (0, 0):
                    continue
                rhs += (
                    comb(state_order, left_state)
                    * comb(direction_order, left_direction)
                    * sp.N(coefficient[(left_state, left_direction)], 80)
                    * result[
                        (
                            state_order - left_state,
                            direction_order - left_direction,
                        )
                    ]
                )
        result[(state_order, direction_order)] = inverse * rhs
    return result


def _bivariate_resolvent(
    resolvent_zero: sp.Expr,
    companion: dict[tuple[int, int], sp.Expr],
    maximum_total_order: int,
) -> dict[tuple[int, int], sp.Expr]:
    result = {(0, 0): sp.N(resolvent_zero, 80)}
    for state_order, direction_order in _pairs(maximum_total_order)[1:]:
        rhs = sp.Float(0, 80)
        for left_state in range(state_order + 1):
            for left_direction in range(direction_order + 1):
                if (left_state, left_direction) == (0, 0):
                    continue
                rhs += (
                    comb(state_order, left_state)
                    * comb(direction_order, left_direction)
                    * companion[(left_state, left_direction)]
                    * result[
                        (
                            state_order - left_state,
                            direction_order - left_direction,
                        )
                    ]
                )
        result[(state_order, direction_order)] = result[(0, 0)] * rhs
    return result


def _bivariate_triple_product(
    left: dict[tuple[int, int], sp.Expr],
    middle: dict[tuple[int, int], sp.Expr],
    right: dict[tuple[int, int], sp.Expr],
    maximum_total_order: int,
) -> dict[tuple[int, int], sp.Expr]:
    result: dict[tuple[int, int], sp.Expr] = {}
    for state_order, direction_order in _pairs(maximum_total_order):
        total = sp.Float(0, 80)
        for left_state in range(state_order + 1):
            for middle_state in range(state_order - left_state + 1):
                right_state = state_order - left_state - middle_state
                for left_direction in range(direction_order + 1):
                    for middle_direction in range(
                        direction_order - left_direction + 1
                    ):
                        right_direction = (
                            direction_order - left_direction - middle_direction
                        )
                        total += (
                            _multinomial(
                                state_order,
                                left_state,
                                middle_state,
                                right_state,
                            )
                            * _multinomial(
                                direction_order,
                                left_direction,
                                middle_direction,
                                right_direction,
                            )
                            * left[(left_state, left_direction)]
                            * middle[(middle_state, middle_direction)]
                            * right[(right_state, right_direction)]
                        )
        result[(state_order, direction_order)] = total
    return result


def _bivariate_k22(
    projector_zero_by_group: dict[str, sp.Expr],
    projector: dict[tuple[int, int], sp.Expr],
    h_star: dict[tuple[int, int], sp.Expr],
    k22_zero: sp.Expr,
    maximum_total_order: int,
) -> dict[tuple[int, int], sp.Expr]:
    result = {(0, 0): sp.N(k22_zero, 80)}
    physical_groups = {"1", "-1"}
    for state_order, direction_order in _pairs(maximum_total_order)[1:]:
        total = sp.Float(0, 80)
        for group, projector_zero in projector_zero_by_group.items():
            for left_state in range(state_order + 1):
                for middle_state in range(state_order - left_state + 1):
                    right_state = state_order - left_state - middle_state
                    for left_direction in range(direction_order + 1):
                        for middle_direction in range(
                            direction_order - left_direction + 1
                        ):
                            right_direction = (
                                direction_order
                                - left_direction
                                - middle_direction
                            )
                            middle_pair = (middle_state, middle_direction)
                            if middle_pair != (0, 0) and group not in physical_groups:
                                continue
                            left_pair = (left_state, left_direction)
                            right_pair = (right_state, right_direction)
                            left_value = (
                                sp.N(projector_zero, 80)
                                if left_pair == (0, 0)
                                else projector[left_pair]
                            )
                            right_value = (
                                sp.N(projector_zero, 80)
                                if right_pair == (0, 0)
                                else projector[right_pair]
                            )
                            if middle_pair == (0, 0):
                                middle_value = (
                                    sp.Rational(9, 8)
                                    if group in physical_groups
                                    else sp.Integer(1)
                                )
                            else:
                                middle_value = h_star[middle_pair]
                            total += (
                                _multinomial(
                                    state_order,
                                    left_state,
                                    middle_state,
                                    right_state,
                                )
                                * _multinomial(
                                    direction_order,
                                    left_direction,
                                    middle_direction,
                                    right_direction,
                                )
                                * left_value
                                * middle_value
                                * right_value
                            )
        result[(state_order, direction_order)] = total
    return result


def _numeric_hierarchy(
    values: dict[tuple[int, int], sp.Expr]
) -> dict[str, float]:
    return {_key(*pair): float(sp.N(value, 18)) for pair, value in values.items()}


def _certify_candidate(
    symmetrizer: dict[str, Any],
    moser: dict[str, Any],
    pde: dict[str, Any],
    tube: dict[str, Any],
    solved_source: dict[str, Any],
    full_symmetrizer: dict[str, Any],
    raw: dict[str, dict[tuple[int, int], sp.Expr]],
    baseline_contract: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates = (symmetrizer, moser, pde, tube, solved_source, full_symmetrizer)
    candidate_id = symmetrizer.get("candidate_id")
    if any(item.get("candidate_id") != candidate_id for item in candidates):
        raise QuarticSymmetrizerSymbolMoserError("candidate ID mismatch")
    if any(
        item.get("coefficients") != symmetrizer.get("coefficients")
        for item in candidates[1:]
    ):
        raise QuarticSymmetrizerSymbolMoserError("candidate coefficient mismatch")
    maximum_order = int(config["maximum_total_derivative_order"])
    inverse_a = sp.sympify(moser["inverse_time_block_2_norm_rational_ceiling"])
    b_product = _bivariate_inverse_product(
        inverse_a, raw["A"], raw["B"], maximum_order
    )
    c_companion = _bivariate_inverse_product(
        inverse_a, raw["A"], raw["C"], maximum_order
    )
    transverse_l = _bivariate_inverse_product(
        inverse_a,
        raw["A"],
        raw["C"],
        maximum_order,
        product_zero=sp.sympify(pde["uniform_bounds"]["L_2_upper"]),
    )
    companion = {
        pair: b_product[pair]
        + c_companion[pair]
        + (sp.Integer(4) if pair == (0, 0) else sp.Integer(0))
        for pair in _pairs(maximum_order)
    }
    # Candidate-specific state-only bounds are tighter and already source-derived.
    for state_order in range(maximum_order + 1):
        pair = (state_order, 0)
        candidate_bound = sp.N(
            sp.sympify(
                moser["companion_Frechet_derivative_2_norm_envelopes"][
                    str(state_order)
                ]
            ),
            80,
        )
        uniform_numeric = float(sp.N(companion[pair], 18))
        candidate_numeric = float(sp.N(candidate_bound, 18))
        if uniform_numeric < candidate_numeric * (1 - 1e-14):
            raise QuarticSymmetrizerSymbolMoserError(
                "uniform companion envelope misses candidate state derivative"
            )

    baseline_resolvent = sp.sympify(
        baseline_contract["maximum_resolvent_upper_bound"]
    )
    perturbation = sp.sympify(
        symmetrizer["uniform_matrix_bounds"]["companion_2_norm_perturbation_upper"]
    )
    denominator = 1 - baseline_resolvent * perturbation
    if not _positive(denominator):
        raise QuarticSymmetrizerSymbolMoserError("candidate resolvent margin failed")
    resolvent_zero = baseline_resolvent / denominator
    resolvent = _bivariate_resolvent(resolvent_zero, companion, maximum_order)
    contour_radius = sp.sympify(config["contour_radius"])
    projector = {
        pair: (
            sp.Integer(0)
            if pair == (0, 0)
            else sp.N(contour_radius, 80) * resolvent[pair]
        )
        for pair in _pairs(maximum_order)
    }
    projector_zero_by_group = {
        name: sp.sympify(value)
        for name, value in pde["uniform_bounds"]["derivation"][
            "candidate_projector_2_norm_uppers"
        ].items()
    }
    h_star = {
        pair: (
            sp.Rational(9, 8)
            if pair == (0, 0)
            else sp.N(raw["H_star"][pair], 80)
        )
        for pair in _pairs(maximum_order)
    }
    k22 = _bivariate_k22(
        projector_zero_by_group,
        projector,
        h_star,
        sp.sympify(pde["uniform_bounds"]["K22_2_upper"]),
        maximum_order,
    )
    identity = {
        pair: sp.Integer(1) if pair == (0, 0) else sp.Integer(0)
        for pair in _pairs(maximum_order)
    }
    inverse_m22 = _bivariate_inverse_product(
        sp.sympify(pde["uniform_bounds"]["M22_inverse_2_upper"]),
        companion,
        identity,
        maximum_order,
        product_zero=sp.sympify(pde["uniform_bounds"]["M22_inverse_2_upper"]),
    )
    cross_f = _bivariate_triple_product(
        transverse_l, k22, inverse_m22, maximum_order
    )
    cross_f[(0, 0)] = sp.N(sp.sympify(pde["uniform_bounds"]["F_2_upper"]), 80)
    k55 = {
        pair: (
            sp.N(sp.sympify(pde["uniform_bounds"]["K55_2_upper"]), 80)
            if pair == (0, 0)
            else 2 * cross_f[pair] + k22[pair]
        )
        for pair in _pairs(maximum_order)
    }
    previous_state = full_symmetrizer[
        "K55_covariant_jet_Frechet_derivative_2_norm_envelopes_numeric"
    ]
    state_crosscheck = {
        str(state_order): {
            "uniform_mixed_envelope": float(sp.N(k55[(state_order, 0)], 18)),
            "candidate_state_only_envelope": float(previous_state[str(state_order)]),
            "covers": float(sp.N(k55[(state_order, 0)], 18))
            >= float(previous_state[str(state_order)]) * (1 - 1e-14),
        }
        for state_order in range(maximum_order + 1)
    }
    if not all(item["covers"] for item in state_crosscheck.values()):
        raise QuarticSymmetrizerSymbolMoserError(
            "mixed hierarchy does not cover the state-only certificate"
        )
    if not all(_positive(value) for value in k55.values()):
        raise QuarticSymmetrizerSymbolMoserError("a mixed K55 envelope is not positive")
    total_four = {
        _key(*pair): float(sp.N(k55[pair], 18))
        for pair in _pairs(maximum_order)
        if sum(pair) == maximum_order
    }
    return {
        "schema_version": "sigma-quartic-symmetrizer-symbol-moser-certificate-1.0",
        "status": "pass_full_K55_mixed_state_direction_C4_symbol_envelopes",
        "candidate_id": candidate_id,
        "coefficients": symmetrizer["coefficients"],
        "domain": {
            "covariant_state_component_radius": symmetrizer["domain"][
                "normalized_local_jet_component_abs"
            ],
            "direction_component_abs": "1",
            "direction_domain": "Euclidean unit sphere inside the component cube",
            "coefficient_envelope": {"abs_alpha": "1", "abs_c20": "1"},
        },
        "companion_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(
            companion
        ),
        "resolvent_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(
            resolvent
        ),
        "projector_mixed_Frechet_2_norm_envelopes_numeric": {
            _key(*pair): float(sp.N(value, 18))
            for pair, value in projector.items()
            if pair != (0, 0)
        },
        "H_star_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(h_star),
        "K22_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(k22),
        "M22_inverse_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(
            inverse_m22
        ),
        "L_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(
            transverse_l
        ),
        "F_cross_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(
            cross_f
        ),
        "K55_mixed_Frechet_2_norm_envelopes_numeric": _numeric_hierarchy(k55),
        "K55_total_order_four_envelopes_numeric": total_four,
        "state_only_coverage_crosscheck": state_crosscheck,
        "claim": (
            "The actual lifted K55 symbol has finite mixed covariant-state and direction "
            "Frechet envelopes for every multiindex a+b<=4."
        ),
        "remaining_gate": (
            "homogeneous_frequency_chart_constants_Sobolev_calculus_and_energy_lifespan"
        ),
        "scope": (
            "This closes mixed state/direction regularity on the unit-direction component "
            "cube. It does not yet convert n-derivatives to homogeneous xi-derivatives, "
            "choose a pseudodifferential quantization, supply Calderon-Vaillancourt or "
            "Sobolev product constants, close the energy inequality, prove a lifespan, "
            "or include matter."
        ),
    }


def run_quartic_symmetrizer_symbol_moser_campaign(
    symmetrizer_campaign: dict[str, Any],
    moser_campaign: dict[str, Any],
    nonquasilinear_pde_campaign: dict[str, Any],
    coordinate_tube_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    full_symmetrizer_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticSymmetrizerSymbolMoserError("unsupported campaign schema_version")
        maximum_order = int(config["maximum_total_derivative_order"])
        if maximum_order != 4:
            raise QuarticSymmetrizerSymbolMoserError(
                "symmetrizer symbol requires total order four"
            )
        if sp.sympify(config["coefficient_abs_envelope"]) != 1:
            raise QuarticSymmetrizerSymbolMoserError(
                "coefficient envelope must cover absolute value one"
            )
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
            (
                full_symmetrizer_campaign,
                "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
            ),
        )
        if any(campaign.get("status") != status for campaign, status in expected_statuses):
            raise QuarticSymmetrizerSymbolMoserError("campaign prerequisite failed")
        pde_hash = nonquasilinear_pde_campaign.get("content_sha256")
        tube_hash = coordinate_tube_campaign.get("content_sha256")
        if nonquasilinear_pde_campaign.get("upstream_sha256", {}).get(
            "symmetrizer"
        ) != symmetrizer_campaign.get("content_sha256") or (
            nonquasilinear_pde_campaign.get("upstream_sha256", {}).get("moser")
            != moser_campaign.get("content_sha256")
        ):
            raise QuarticSymmetrizerSymbolMoserError("PDE provenance mismatch")
        if coordinate_tube_campaign.get("nonquasilinear_pde_campaign_sha256") != pde_hash:
            raise QuarticSymmetrizerSymbolMoserError("coordinate-tube provenance mismatch")
        full_upstream = full_symmetrizer_campaign.get("upstream_sha256", {})
        if (
            full_upstream.get("symmetrizer")
            != symmetrizer_campaign.get("content_sha256")
            or full_upstream.get("moser") != moser_campaign.get("content_sha256")
            or full_upstream.get("nonquasilinear_pde") != pde_hash
            or full_upstream.get("coordinate_tube") != tube_hash
            or full_upstream.get("solved_source")
            != solved_source_campaign.get("content_sha256")
        ):
            raise QuarticSymmetrizerSymbolMoserError(
                "full-symmetrizer provenance mismatch"
            )
        control_passed, control = generic_bivariate_symbol_derivative_control()
        if not control_passed:
            raise QuarticSymmetrizerSymbolMoserError("generic bivariate control failed")
        baseline_passed, baseline = quartic_horndeski_baseline_riesz_symmetrizer_control()
        if not baseline_passed:
            raise QuarticSymmetrizerSymbolMoserError("baseline Riesz control failed")
        baseline_contract = baseline["quantitative_physical_group_perturbation_contract"]
        contour_radius = sp.sympify(config["contour_radius"])
        if contour_radius != sp.sympify(baseline_contract["contour_radius"]):
            raise QuarticSymmetrizerSymbolMoserError("Riesz contour radius mismatch")
        state_radius = str(config["covariant_state_component_radius"])
        if sp.sympify(state_radius) != sp.Rational(1, 5_000_000_000):
            raise QuarticSymmetrizerSymbolMoserError("covariant state radius mismatch")
        raw = _uniform_raw_mixed_envelopes(state_radius, maximum_order)
        record_sets = [
            _candidate_records(campaign)
            for campaign in (
                symmetrizer_campaign,
                moser_campaign,
                nonquasilinear_pde_campaign,
                coordinate_tube_campaign,
                solved_source_campaign,
                full_symmetrizer_campaign,
            )
        ]
        expected = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(record_sets[0])
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in record_sets[1:]
        ):
            raise QuarticSymmetrizerSymbolMoserError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                record_sets[0][candidate_id],
                record_sets[1][candidate_id],
                record_sets[2][candidate_id],
                record_sets[3][candidate_id],
                record_sets[4][candidate_id],
                record_sets[5][candidate_id],
                raw,
                baseline_contract,
                config,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        raw_evidence = {
            name: {
                _key(*pair): {
                    "exact": str(value),
                    "numeric": float(sp.N(value, 18)),
                }
                for pair, value in hierarchy.items()
            }
            for name, hierarchy in raw.items()
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes",
            "errors": [],
            "upstream_sha256": {
                "symmetrizer": symmetrizer_campaign.get("content_sha256"),
                "moser": moser_campaign.get("content_sha256"),
                "nonquasilinear_pde": pde_hash,
                "coordinate_tube": tube_hash,
                "solved_source": solved_source_campaign.get("content_sha256"),
                "full_symmetrizer": full_symmetrizer_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_bivariate_symbol_derivative_control": control,
            "uniform_raw_mixed_derivative_envelopes": raw_evidence,
            "counts": {
                "selected": len(certificates),
                "mixed_symbol_envelopes_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have mixed state/direction Frechet envelopes "
                "through total order four for the actual lifted K55 symbol."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticSymmetrizerSymbolMoserError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "mixed_symbol_envelopes_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_symmetrizer_symbol_moser_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
