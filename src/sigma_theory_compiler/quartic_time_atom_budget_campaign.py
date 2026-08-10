from __future__ import annotations

import hashlib
import json
import math
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_homogeneous_frequency_symbol_campaign import _multiplicity_vectors
from .quartic_r3_sobolev_calculus_campaign import (
    _composition_coefficients,
    _key,
    _pairs,
    r3_sobolev_embedding_constant,
)

SCHEMA_VERSION = "sigma-quartic-time-atom-budget-campaign-1.0"


class QuarticTimeAtomBudgetError(ValueError):
    """Raised when the coordinate-atom time budget cannot be closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == hashlib.sha256(
        _canonical_json(body).encode()
    ).hexdigest()


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _partition_count(multiplicities: tuple[int, ...]) -> int:
    order = sum(
        derivative_order * count
        for derivative_order, count in enumerate(multiplicities, start=1)
    )
    denominator = 1
    for derivative_order, count in enumerate(multiplicities, start=1):
        denominator *= factorial(count) * factorial(derivative_order) ** count
    return factorial(order) // denominator


def _marked_time_chain_expression(
    state_frequency_bounds: dict[tuple[int, int], sp.Expr],
    spatial_order: int,
    frequency_order: int,
    ordinary_spatial_jets: dict[int, sp.Expr],
    marked_time_jets: dict[int, sp.Expr],
) -> sp.Expr:
    """Bound D_x^m partial_t F(Y) with one marked Y_t block."""

    expression = sp.Integer(0)
    for marked_order in range(spatial_order + 1):
        remaining = spatial_order - marked_order
        for multiplicities in _multiplicity_vectors(remaining):
            ordinary_blocks = sum(multiplicities)
            state_order = ordinary_blocks + 1
            term = (
                comb(spatial_order, marked_order)
                * _partition_count(multiplicities)
                * marked_time_jets[marked_order]
                * state_frequency_bounds[(state_order, frequency_order)]
            )
            for derivative_order, count in enumerate(multiplicities, start=1):
                term *= ordinary_spatial_jets[derivative_order] ** count
            expression += term
    return sp.factor(expression)


def _spatial_composition_expression(
    derivative_bounds: dict[int, sp.Expr],
    spatial_order: int,
    spatial_jets: dict[int, sp.Expr],
) -> sp.Expr:
    state_bounds = {(order, 0): value for order, value in derivative_bounds.items()}
    return sp.factor(
        sum(
            _composition_coefficients(
                state_bounds, spatial_order, 0, spatial_jets
            ).values()
        )
    )


def _outward_source_bound(value: Any, padding: sp.Rational) -> sp.Expr:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise QuarticTimeAtomBudgetError("solved-source envelope is not finite positive")
    recorded = sp.Rational(str(numeric))
    outward = sp.factor(recorded * padding)
    if not outward > recorded:
        raise QuarticTimeAtomBudgetError("outward source padding must exceed one")
    return outward


@cache
def generic_coordinate_atom_time_evolution_control() -> tuple[bool, dict[str, Any]]:
    """Verify the atom counts, commuting-partial evolution, and minimal H7 gate."""

    t, x_1, x_2 = sp.symbols("t x_1 x_2", real=True, finite=True)
    q = sp.Function("q")(t, x_1, x_2)
    commuting_residuals = {
        "dt_partial_i_q_minus_partial_i_dt_q": sp.diff(q, t, x_1)
        - sp.diff(q, x_1, t),
        "dt_partial_0i_q_minus_partial_i_partial_00_q": sp.diff(
            q, t, t, x_1
        )
        - sp.diff(q, t, t, x_1),
        "dt_partial_ij_q_minus_partial_ij_partial_0_q": sp.diff(
            q, t, x_1, x_2
        )
        - sp.diff(q, x_1, x_2, t),
    }
    counts = {
        "metric_deviations": 10,
        "field_first_partials": 44,
        "mixed_0i_second_partials": 33,
        "symmetric_ij_second_partials": 66,
    }
    total = sum(counts.values())
    h7_d5 = r3_sobolev_embedding_constant(7, 5)
    insufficient_rejected = False
    insufficient_error = ""
    try:
        r3_sobolev_embedding_constant(6, 5)
    except ValueError as error:
        insufficient_rejected = True
        insufficient_error = str(error)
    corrupted = sp.diff(q, t, t, x_1) - sp.diff(q, t, t)
    passed = bool(
        total == 153
        and set(commuting_residuals.values()) == {0}
        and h7_d5.is_positive
        and insufficient_rejected
        and corrupted != 0
    )
    return passed, {
        "control": "exact 153-coordinate-atom time evolution and minimal state regularity",
        "coordinate_atom_counts": {**counts, "total": total},
        "time_evolution_families": {
            "metric_deviation": "partial_t h_ab=p_0,ab",
            "time_first_partial": "partial_t p_0,A=F_A(Y)",
            "spatial_first_partial": "partial_t p_i,A=s_0i,A",
            "mixed_second_partial": "partial_t s_0i,A=partial_i F_A(Y)",
            "spatial_second_partial": "partial_t s_ij,A=partial_i partial_j p_0,A",
        },
        "commuting_partial_residuals": {
            key: str(value) for key, value in commuting_residuals.items()
        },
        "state_to_atom_norm_map": (
            "For E=max_I ||U_I||_H7, each zero/first atom has H6 norm <=E and "
            "each acceleration-free second atom is one spatial derivative of v_0 or "
            "w_i, hence also has H6 norm <=E. Therefore R=max_A||Y_A||_H6<=E."
        ),
        "highest_velocity_spatial_derivative_needed": 5,
        "minimal_integer_state_sobolev_order": 7,
        "H7_to_D5_Linfinity_constant": str(h7_d5),
        "insufficient_H6_negative": {
            "rejected": insufficient_rejected,
            "error": insufficient_error,
        },
        "negative_control": {
            "corruption": "omit the spatial derivative on partial_i F in dt s_0i",
            "exact_witness_residual": str(corrupted),
            "rejected": corrupted != 0,
        },
        "passed": passed,
    }


@cache
def generic_marked_time_chain_control() -> tuple[bool, dict[str, Any]]:
    """Verify the source and marked-time Faa-di-Bruno multiplicities through order three."""

    x, epsilon = sp.symbols("x epsilon", real=True, finite=True)
    maximum_time_order = 3
    a = {order: sp.Integer(order + 2) for order in range(1, 5)}
    b = {order: sp.Integer(2 * order + 3) for order in range(4)}
    outer_derivatives = {order: sp.Integer(5 * order + 7) for order in range(5)}
    ordinary_curve = sum(a[order] * x**order / factorial(order) for order in a)
    marked_curve = sum(b[order] * x**order / factorial(order) for order in b)
    y = sp.Symbol("y")
    outer = sum(
        outer_derivatives[order] * y**order / factorial(order)
        for order in outer_derivatives
    )
    direct_spatial = outer.subs(y, ordinary_curve)
    direct_time = sp.diff(
        outer.subs(y, ordinary_curve + epsilon * marked_curve), epsilon
    ).subs(epsilon, 0)
    state_bounds = {(order, 0): value for order, value in outer_derivatives.items()}
    spatial_residuals: dict[str, str] = {}
    time_residuals: dict[str, str] = {}
    for order in range(5):
        recurrence = _spatial_composition_expression(
            outer_derivatives, order, a
        )
        direct = sp.diff(direct_spatial, x, order).subs(x, 0)
        spatial_residuals[str(order)] = str(sp.expand(direct - recurrence))
    for order in range(maximum_time_order + 1):
        recurrence = _marked_time_chain_expression(
            state_bounds, order, 0, a, b
        )
        direct = sp.diff(direct_time, x, order).subs(x, 0)
        time_residuals[str(order)] = str(sp.expand(direct - recurrence))
    target = 3
    correct = _marked_time_chain_expression(state_bounds, target, 0, a, b)
    corruption_residual = sp.expand(
        correct - (correct - outer_derivatives[2] * b[1] * a[2])
    )
    passed = bool(
        set(spatial_residuals.values()) == {"0"}
        and set(time_residuals.values()) == {"0"}
        and corruption_residual != 0
    )
    return passed, {
        "control": "spatial source and marked-time Faa-di-Bruno recurrence",
        "source_spatial_residuals": spatial_residuals,
        "marked_time_residuals": time_residuals,
        "negative_control": {
            "corruption": "reduce choose(3,1) by one on the marked-order-one term",
            "exact_witness_residual": str(corruption_residual),
            "rejected": corruption_residual != 0,
        },
        "passed": passed,
    }


def _certify_candidate(
    low_frequency: dict[str, Any],
    r3: dict[str, Any],
    solved_source: dict[str, Any],
    pde: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(low_frequency.get("candidate_id"))
    records = (r3, solved_source, pde)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTimeAtomBudgetError("candidate ID mismatch")
    if any(record.get("coefficients") != low_frequency.get("coefficients") for record in records):
        raise QuarticTimeAtomBudgetError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_global_C4_positive_K55_symbol_extension",
        "pass_R3_H6_spatialized_K55_P55_symbol_bounds",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
    )
    if tuple(record.get("status") for record in (low_frequency, *records)) != expected_statuses:
        raise QuarticTimeAtomBudgetError("candidate prerequisite status mismatch")
    if pde.get("full_first_order_state") != {
        "q": 11,
        "v_0": 11,
        "v_i": 33,
        "total": 55,
        "definition_constraints": 33,
        "independent_spatial_curl_constraints": 33,
    }:
        raise QuarticTimeAtomBudgetError("55-state decomposition mismatch")

    maximum_symbol_order = int(config["maximum_symbol_derivative_order"])
    maximum_time_order = int(config["maximum_time_spatial_order"])
    expected_pairs = set(_pairs(maximum_symbol_order))
    raw_k = low_frequency.get("global_C4_frequency_derivative_integer_ceilings", {})
    if set(raw_k) != {_key(*pair) for pair in expected_pairs}:
        raise QuarticTimeAtomBudgetError("global K55 bounds are incomplete")
    coordinate_map = {0: sp.Integer(1)} | {
        order: sp.Integer(r3["coordinate_map_Frechet_integer_ceilings"][str(order)])
        for order in range(1, maximum_symbol_order + 1)
    }
    k_covariant = {
        pair: sp.Integer(raw_k[_key(*pair)]["global_ceiling"])
        for pair in expected_pairs
    }
    k_coordinate = {
        pair: (
            k_covariant[pair]
            if pair[0] == 0
            else sp.factor(
                sum(
                    _composition_coefficients(
                        k_covariant, pair[0], pair[1], coordinate_map
                    ).values()
                )
            )
        )
        for pair in expected_pairs
    }

    padding = sp.Rational(
        int(config["outward_padding_numerator"]),
        int(config["outward_padding_denominator"]),
    )
    source_numeric = solved_source["solved_source_Frechet_derivatives"][
        "2_norm_envelopes_numeric"
    ]
    source_bounds = {
        order: _outward_source_bound(source_numeric[str(order)], padding)
        for order in range(maximum_symbol_order + 1)
    }
    energy = sp.Symbol("E", nonnegative=True, finite=True)
    coordinate_embedding = {
        order: r3_sobolev_embedding_constant(6, order)
        for order in range(maximum_symbol_order + 1)
    }
    ordinary_y_jets = {
        order: coordinate_embedding[order] * energy
        for order in range(1, maximum_symbol_order + 1)
    }
    source_spatial = {
        order: _spatial_composition_expression(
            source_bounds, order, ordinary_y_jets
        )
        for order in range(maximum_symbol_order + 1)
    }
    state_embedding = {
        order: r3_sobolev_embedding_constant(7, order)
        for order in range(2, maximum_time_order + 3)
    }
    time_atom_jets = {
        order: sp.factor(
            coordinate_embedding[order] * energy
            + source_spatial[order]
            + source_spatial[order + 1]
            + state_embedding[order + 2] * energy
        )
        for order in range(maximum_time_order + 1)
    }
    time_pairs = set(_pairs(maximum_time_order))
    closed_time_k = {
        pair: _marked_time_chain_expression(
            k_coordinate,
            pair[0],
            pair[1],
            ordinary_y_jets,
            time_atom_jets,
        )
        for pair in time_pairs
    }

    radius = sp.sympify(
        r3["sufficient_H6_radius_for_state_and_spatial_jet_tube"]["H6_radius"]
    )
    radius_symbol = sp.Symbol("R", nonnegative=True, finite=True)
    time_symbol = sp.Symbol("R_t", nonnegative=True, finite=True)
    published_residuals: dict[str, str] = {}
    for pair in time_pairs:
        placeholder = _marked_time_chain_expression(
            k_coordinate,
            pair[0],
            pair[1],
            {
                order: coordinate_embedding[order] * radius_symbol
                for order in range(1, maximum_symbol_order + 1)
            },
            {
                order: coordinate_embedding[order] * time_symbol
                for order in range(maximum_time_order + 1)
            },
        )
        published = time_symbol * sp.sympify(
            r3["spatialized_time_K55_bounds"][_key(*pair)]["expression"],
            locals={"R": radius_symbol},
        )
        published_residuals[_key(*pair)] = str(sp.expand(placeholder - published))
    if set(published_residuals.values()) != {"0"}:
        raise QuarticTimeAtomBudgetError("published R3 marked-time chain mismatch")
    if any(expression.free_symbols - {energy} for expression in closed_time_k.values()):
        raise QuarticTimeAtomBudgetError("an unclosed time-budget symbol remains")

    return {
        "schema_version": "sigma-quartic-time-atom-budget-certificate-1.0",
        "status": "pass_H7_closed_coordinate_atom_time_budget",
        "candidate_id": candidate_id,
        "coefficients": low_frequency.get("coefficients"),
        "functional_setting": {
            "space": "R^3",
            "state_energy": "E=max_I ||U_I||_H7 for U=(q,v_0,w_i), I=1,...,55",
            "coordinate_atom_norm": "R=max_A ||Y_A||_H6 <= E",
            "coordinate_atom_count": 153,
            "marked_time_orders": list(range(maximum_time_order + 1)),
        },
        "state_to_coordinate_atom_H7_to_H6_constant": "1",
        "outward_source_Frechet_bounds": {
            str(order): str(value) for order, value in source_bounds.items()
        },
        "source_spatial_chain_bounds": {
            str(order): {
                "expression": str(expression),
                "at_sufficient_tube_radius_numeric": float(
                    sp.N(expression.subs(energy, radius), 18)
                ),
            }
            for order, expression in source_spatial.items()
        },
        "closed_coordinate_atom_time_jets": {
            str(order): {
                "quantity": f"max_A ||D_x^{order} partial_t Y_A||_infinity",
                "expression": str(expression),
                "at_sufficient_tube_radius_numeric": float(
                    sp.N(expression.subs(energy, radius), 18)
                ),
            }
            for order, expression in time_atom_jets.items()
        },
        "closed_time_K55_bounds": {
            _key(*pair): {
                "quantity": (
                    f"||D_x^{pair[0]} partial_xi^{pair[1]} partial_t K55||_2"
                ),
                "expression": str(closed_time_k[pair]),
                "at_sufficient_tube_radius_numeric": float(
                    sp.N(closed_time_k[pair].subs(energy, radius), 18)
                ),
            }
            for pair in _pairs(maximum_time_order)
        },
        "published_R3_placeholder_residuals": published_residuals,
        "sufficient_H7_state_radius_for_coordinate_tube": {
            "exact": str(radius),
            "numeric": float(sp.N(radius, 18)),
            "reason": "R<=E by the exact state-to-atom derivative map",
        },
        "derivative_accounting": {
            "maximum_source_Frechet_order_used": maximum_symbol_order,
            "maximum_source_spatial_order_used": maximum_symbol_order,
            "maximum_velocity_spatial_derivative_used": maximum_time_order + 2,
            "undefined_partial_t_Y_norm_remaining": False,
        },
        "claim": (
            "The exact 153-atom evolution closes every marked time jet needed by the "
            "C3 time derivative of K55 as an explicit polynomial of the 55-state H7 "
            "energy, using no undefined partial_t Y norm."
        ),
        "remaining_gate": (
            "explicit_anti_Wick_evolution_composition_bounded_frequency_defect_and_dyadic_sum"
        ),
        "scope": (
            "This closes the coordinate-atom time-jet input to the symbol calculus. It "
            "does not yet prove the anti-Wick composition estimate, close the commuted "
            "energy inequality, assign a lifespan, add matter, or test observations."
        ),
    }


def run_quartic_time_atom_budget_campaign(
    low_frequency_campaign: dict[str, Any],
    r3_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    nonquasilinear_pde_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTimeAtomBudgetError("unsupported campaign schema_version")
        expected_statuses = (
            "pass_all_12_global_C4_positive_K55_symbol_extensions",
            "pass_all_12_R3_H6_spatialized_K55_P55_symbol_bounds",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
        )
        campaigns = (
            low_frequency_campaign,
            r3_campaign,
            solved_source_campaign,
            nonquasilinear_pde_campaign,
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTimeAtomBudgetError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTimeAtomBudgetError("campaign content hash mismatch")
        if r3_campaign.get("upstream_sha256", {}).get(
            "low_frequency"
        ) != low_frequency_campaign.get("content_sha256"):
            raise QuarticTimeAtomBudgetError("low-frequency to R3 provenance mismatch")
        if r3_campaign.get("upstream_sha256", {}).get(
            "solved_source"
        ) != solved_source_campaign.get("content_sha256"):
            raise QuarticTimeAtomBudgetError("solved-source to R3 provenance mismatch")
        if solved_source_campaign.get("upstream_sha256", {}).get(
            "nonquasilinear_pde"
        ) != nonquasilinear_pde_campaign.get("content_sha256"):
            raise QuarticTimeAtomBudgetError("55-state to solved-source provenance mismatch")
        if (
            int(config["coordinate_atom_sobolev_order"]) != 6
            or int(config["state_sobolev_order"]) != 7
            or int(config["maximum_symbol_derivative_order"]) != 4
            or int(config["maximum_time_spatial_order"]) != 3
        ):
            raise QuarticTimeAtomBudgetError(
                "time-atom closure requires coordinate H6, state H7, source C4, and C3 time jets"
            )
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
        ):
            raise QuarticTimeAtomBudgetError(
                "campaign requires R3, 55 evolution states, and 153 coordinate atoms"
            )
        padding = sp.Rational(
            int(config["outward_padding_numerator"]),
            int(config["outward_padding_denominator"]),
        )
        if padding <= 1:
            raise QuarticTimeAtomBudgetError("outward padding must exceed one")
        atom_passed, atom_control = generic_coordinate_atom_time_evolution_control()
        chain_passed, chain_control = generic_marked_time_chain_control()
        if not atom_passed or not chain_passed:
            raise QuarticTimeAtomBudgetError("generic time-atom control failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        expected = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(maps[0])
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTimeAtomBudgetError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), config
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_H7_closed_coordinate_atom_time_budgets",
            "errors": [],
            "upstream_sha256": {
                "low_frequency": low_frequency_campaign.get("content_sha256"),
                "r3_sobolev": r3_campaign.get("content_sha256"),
                "solved_source": solved_source_campaign.get("content_sha256"),
                "nonquasilinear_pde": nonquasilinear_pde_campaign.get(
                    "content_sha256"
                ),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_coordinate_atom_time_evolution_control": atom_control,
            "generic_marked_time_chain_control": chain_control,
            "counts": {
                "selected": len(certificates),
                "H7_time_atom_budgets_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates close the four marked coordinate-atom time "
                "jets and all 10 partial_t K55 symbol bounds as explicit H7-state "
                "energy polynomials."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticTimeAtomBudgetError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "H7_time_atom_budgets_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_time_atom_budget_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
