from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import quartic_horndeski_baseline_riesz_symmetrizer_control
from .quartic_homogeneous_frequency_symbol_campaign import (
    _composed_bound,
    _multiplicity_vectors,
    normalization_map_frechet_majorants,
)
from .quartic_low_frequency_symbol_extension_campaign import (
    radius_map_frechet_majorants,
)
from .quartic_r3_sobolev_calculus_campaign import r3_sobolev_embedding_constant
from .quartic_symmetrizer_symbol_moser_campaign import (
    _bivariate_inverse_product,
    _bivariate_k22,
    _bivariate_resolvent,
    _bivariate_triple_product,
    _positive,
    _uniform_raw_mixed_envelopes,
)
from .recovery_artifact_validation import (
    DATA_SEALS,
    load_bound_inputs,
    validate_bound_inputs,
    validate_exact_rebuild,
)

SCHEMA_VERSION = "sigma-quartic-annular-k55-c6-campaign-1.0"
CONFIG_KEYS = {
    "schema_version",
    "expected_candidate_count",
    "maximum_total_derivative_order",
    "spatial_dimension",
    "state_dimension",
    "covariant_state_component_radius",
    "contour_radius",
    "annular_support_radius_lower",
    "annular_support_radius_upper",
    "annular_ramp_width",
    "semiclassical_h_maximum",
}
DEFAULT_ANNULAR_RADIUS_LOWER = sp.Rational(5, 2)
DEFAULT_ANNULAR_RAMP_WIDTH = sp.Rational(1, 2)


class QuarticAnnularK55C6Error(ValueError):
    """Raised when the targeted annular C6 composition bounds cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return (
        campaign.get("content_sha256") == hashlib.sha256(_canonical_json(body).encode()).hexdigest()
    )


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _pairs(maximum_total_order: int) -> list[tuple[int, int]]:
    return [
        (left, total - left)
        for total in range(maximum_total_order + 1)
        for left in range(total + 1)
    ]


def _key(left: int, right: int) -> str:
    return f"{left},{right}"


def _partition_count(multiplicities: tuple[int, ...]) -> int:
    order = sum(
        derivative_order * count for derivative_order, count in enumerate(multiplicities, start=1)
    )
    denominator = 1
    for derivative_order, count in enumerate(multiplicities, start=1):
        denominator *= factorial(count) * factorial(derivative_order) ** count
    return factorial(order) // denominator


def _chi6() -> sp.Expr:
    t = sp.Symbol("t", real=True)
    return (
        1716 * t**7
        - 9009 * t**8
        + 20020 * t**9
        - 24024 * t**10
        + 16380 * t**11
        - 6006 * t**12
        + 924 * t**13
    )


def _chi_derivative_majorants(maximum_order: int) -> dict[int, int]:
    t = sp.Symbol("t", real=True)
    cutoff = _chi6()
    return {
        order: int(
            sum(
                abs(coefficient) for _, coefficient in sp.Poly(sp.diff(cutoff, t, order), t).terms()
            )
        )
        for order in range(maximum_order + 1)
    }


def annular_cutoff_frechet_majorants(
    maximum_order: int = 6,
    radius_lower: sp.Expr = DEFAULT_ANNULAR_RADIUS_LOWER,
    ramp_width: sp.Expr = DEFAULT_ANNULAR_RAMP_WIDTH,
) -> dict[int, sp.Expr]:
    """Bound coordinate Frechet derivatives of the radial C6 annular cutoff."""

    chi = _chi_derivative_majorants(maximum_order)
    radial = radius_map_frechet_majorants(maximum_order)
    scale = 1 / ramp_width
    result: dict[int, sp.Expr] = {0: sp.Integer(1)}
    for order in range(1, maximum_order + 1):
        total = sp.Integer(0)
        for multiplicities in _multiplicity_vectors(order):
            outer_order = sum(multiplicities)
            term = _partition_count(multiplicities) * chi[outer_order] * scale**outer_order
            for derivative_order, count in enumerate(multiplicities, start=1):
                derivative_bound = radial[derivative_order] * radius_lower ** (1 - derivative_order)
                term *= derivative_bound**count
            total += term
        result[order] = sp.factor(total)
    return result


@cache
def generic_annular_k55_c6_control() -> tuple[bool, dict[str, Any]]:
    """Verify every recurrence and cutoff constant used by the targeted C6 gate."""

    maximum_order = 6
    pairs = _pairs(maximum_order)
    u, v = sp.symbols("u v", real=True, finite=True)
    coefficient = {pair: sp.Integer(7 + 10 * pair[0] + pair[1]) for pair in pairs}
    inverse_zero = sp.Integer(2)
    inverse = {(0, 0): inverse_zero}
    for state_order, direction_order in pairs[1:]:
        inverse[(state_order, direction_order)] = inverse_zero * sum(
            comb(state_order, left_state)
            * comb(direction_order, left_direction)
            * coefficient[(left_state, left_direction)]
            * inverse[(state_order - left_state, direction_order - left_direction)]
            for left_state in range(state_order + 1)
            for left_direction in range(direction_order + 1)
            if (left_state, left_direction) != (0, 0)
        )
    coefficient_series = sum(
        coefficient[pair] * u ** pair[0] * v ** pair[1] / (factorial(pair[0]) * factorial(pair[1]))
        for pair in pairs
        if pair != (0, 0)
    )
    inverse_series = sum(
        inverse[pair] * u ** pair[0] * v ** pair[1] / (factorial(pair[0]) * factorial(pair[1]))
        for pair in pairs
    )
    inverse_residuals = {
        _key(*pair): str(
            sp.expand(
                sp.diff(
                    (1 / inverse_zero - coefficient_series) * inverse_series - 1,
                    u,
                    pair[0],
                    v,
                    pair[1],
                ).subs({u: 0, v: 0})
            )
        )
        for pair in pairs
    }

    normalization = normalization_map_frechet_majorants(maximum_order)
    n = sp.Symbol("n", real=True, finite=True)
    outer = sum(
        sp.Integer(3 + 10 * pair[0] + pair[1])
        * u ** pair[0]
        * (n - 1) ** pair[1]
        / (factorial(pair[0]) * factorial(pair[1]))
        for pair in pairs
    )
    frequency_curve = sum(
        sp.Integer(normalization[order]) * v**order / factorial(order)
        for order in range(maximum_order + 1)
    )
    bell_residuals: dict[str, str] = {}
    direction_bounds = {pair: sp.Integer(3 + 10 * pair[0] + pair[1]) for pair in pairs}
    for state_order, frequency_order in pairs:
        direct = sp.diff(
            outer.subs(n, frequency_curve),
            u,
            state_order,
            v,
            frequency_order,
        ).subs({u: 0, v: 0})
        recurrence = _composed_bound(direction_bounds, state_order, frequency_order, normalization)
        bell_residuals[_key(state_order, frequency_order)] = str(sp.expand(direct - recurrence))

    t = sp.Symbol("t", real=True)
    cutoff = _chi6()
    cutoff_residual = sp.factor(sp.diff(cutoff, t) - 12012 * t**6 * (1 - t) ** 6)
    endpoint_residuals = [
        sp.diff(cutoff, t, order).subs(t, endpoint) - (1 if endpoint == 1 and order == 0 else 0)
        for endpoint in (0, 1)
        for order in range(7)
    ]
    cutoff_majorants = annular_cutoff_frechet_majorants(maximum_order)

    j1, j2, c1, c2, energy = sp.symbols("J1 J2 C1 C2 E", nonnegative=True, finite=True)
    k1, k2 = sp.symbols("K1 K2", nonnegative=True, finite=True)
    coordinate_curve = j1 * u + j2 * u**2 / 2
    outer_curve = k1 * v + k2 * v**2 / 2
    second_direct = sp.diff(outer_curve.subs(v, coordinate_curve), u, 2).subs(u, 0)
    second_coordinate_residual = sp.expand(second_direct - (k2 * j1**2 + k1 * j2))
    spatial_curve = c1 * energy * u + c2 * energy * u**2 / 2
    second_spatial_direct = sp.diff(outer_curve.subs(v, spatial_curve), u, 2).subs(u, 0)
    second_spatial_residual = sp.expand(
        second_spatial_direct - (k2 * (c1 * energy) ** 2 + k1 * c2 * energy)
    )

    corrupted = dict(normalization)
    corrupted[6] -= 1
    corruption = _composed_bound(direction_bounds, 0, 6, normalization) - _composed_bound(
        direction_bounds, 0, 6, corrupted
    )
    passed = bool(
        set(inverse_residuals.values()) == {"0"}
        and set(bell_residuals.values()) == {"0"}
        and cutoff_residual == 0
        and set(endpoint_residuals) == {0}
        and all(value > 0 for value in cutoff_majorants.values())
        and second_coordinate_residual == 0
        and second_spatial_residual == 0
        and corruption != 0
    )
    return passed, {
        "control": "targeted annular K55 C6 recurrence and spatialization",
        "maximum_total_order": maximum_order,
        "bivariate_inverse_residuals": inverse_residuals,
        "normalization_map_Frechet_majorants": {
            str(order): value for order, value in normalization.items()
        },
        "homogeneous_Bell_residuals": bell_residuals,
        "chi6_derivative_residual": str(cutoff_residual),
        "chi6_endpoint_residuals": [str(value) for value in endpoint_residuals],
        "annular_cutoff_Frechet_majorants": {
            str(order): str(value) for order, value in cutoff_majorants.items()
        },
        "coordinate_second_chain_residual": str(second_coordinate_residual),
        "spatial_second_chain_residual": str(second_spatial_residual),
        "required_maximal_spatial_frequency_pairs": [[2, 4], [0, 6], [0, 5], [1, 4]],
        "negative_control": {
            "corruption": "decrease the sixth normalization-map majorant by one",
            "exact_witness_residual": str(sp.expand(corruption)),
            "rejected": corruption != 0,
        },
        "passed": passed,
    }


def _covariant_direction_k55_c6(
    symmetrizer: dict[str, Any],
    moser: dict[str, Any],
    pde: dict[str, Any],
    raw: dict[str, dict[tuple[int, int], sp.Expr]],
    baseline: dict[str, Any],
    contour_radius: sp.Expr,
) -> dict[tuple[int, int], sp.Expr]:
    maximum_order = 6
    inverse_a = sp.sympify(moser["inverse_time_block_2_norm_rational_ceiling"])
    b_product = _bivariate_inverse_product(inverse_a, raw["A"], raw["B"], maximum_order)
    c_product = _bivariate_inverse_product(inverse_a, raw["A"], raw["C"], maximum_order)
    transverse_l = _bivariate_inverse_product(
        inverse_a,
        raw["A"],
        raw["C"],
        maximum_order,
        product_zero=sp.sympify(pde["uniform_bounds"]["L_2_upper"]),
    )
    companion = {
        pair: b_product[pair]
        + c_product[pair]
        + (sp.Integer(4) if pair == (0, 0) else sp.Integer(0))
        for pair in _pairs(maximum_order)
    }
    for order in range(5):
        published = sp.sympify(moser["companion_Frechet_derivative_2_norm_envelopes"][str(order)])
        if sp.N(companion[(order, 0)], 80) < sp.N(published, 80) * (1 - sp.Float("1e-14")):
            raise QuarticAnnularK55C6Error(
                "uniform companion hierarchy misses a published state bound"
            )

    baseline_resolvent = sp.sympify(baseline["maximum_resolvent_upper_bound"])
    perturbation = sp.sympify(
        symmetrizer["uniform_matrix_bounds"]["companion_2_norm_perturbation_upper"]
    )
    denominator = 1 - baseline_resolvent * perturbation
    if not _positive(denominator):
        raise QuarticAnnularK55C6Error("candidate resolvent margin failed")
    resolvent = _bivariate_resolvent(baseline_resolvent / denominator, companion, maximum_order)
    projector = {
        pair: (sp.Integer(0) if pair == (0, 0) else sp.N(contour_radius, 80) * resolvent[pair])
        for pair in _pairs(maximum_order)
    }
    projector_zero = {
        name: sp.sympify(value)
        for name, value in pde["uniform_bounds"]["derivation"][
            "candidate_projector_2_norm_uppers"
        ].items()
    }
    h_star = {
        pair: (sp.Rational(9, 8) if pair == (0, 0) else sp.N(raw["H_star"][pair], 80))
        for pair in _pairs(maximum_order)
    }
    k22 = _bivariate_k22(
        projector_zero,
        projector,
        h_star,
        sp.sympify(pde["uniform_bounds"]["K22_2_upper"]),
        maximum_order,
    )
    identity = {
        pair: sp.Integer(1) if pair == (0, 0) else sp.Integer(0) for pair in _pairs(maximum_order)
    }
    inverse_m22 = _bivariate_inverse_product(
        sp.sympify(pde["uniform_bounds"]["M22_inverse_2_upper"]),
        companion,
        identity,
        maximum_order,
        product_zero=sp.sympify(pde["uniform_bounds"]["M22_inverse_2_upper"]),
    )
    cross_f = _bivariate_triple_product(transverse_l, k22, inverse_m22, maximum_order)
    cross_f[(0, 0)] = sp.N(sp.sympify(pde["uniform_bounds"]["F_2_upper"]), 80)
    k55 = {
        pair: (
            sp.N(sp.sympify(pde["uniform_bounds"]["K55_2_upper"]), 80)
            if pair == (0, 0)
            else 2 * cross_f[pair] + k22[pair]
        )
        for pair in _pairs(maximum_order)
    }
    if not all(_positive(value) for value in k55.values()):
        raise QuarticAnnularK55C6Error("a C6 K55 direction bound is not positive")
    return k55


def _certify_candidate(
    symmetrizer: dict[str, Any],
    moser: dict[str, Any],
    pde: dict[str, Any],
    tube: dict[str, Any],
    solved: dict[str, Any],
    full: dict[str, Any],
    symbol_c4: dict[str, Any],
    r3: dict[str, Any],
    anti_wick: dict[str, Any],
    raw: dict[str, dict[tuple[int, int], sp.Expr]],
    baseline: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = (symmetrizer, moser, pde, tube, solved, full, symbol_c4, r3, anti_wick)
    candidate_id = str(symmetrizer.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticAnnularK55C6Error("candidate identity mismatch")
    if any(record.get("coefficients") != symmetrizer.get("coefficients") for record in records[1:]):
        raise QuarticAnnularK55C6Error("candidate coefficient mismatch")
    expected_statuses = (
        "pass_uniform_local_jet_strong_hyperbolicity",
        "pass_quasilinear_coefficient_derivative_envelopes",
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "pass_full_K55_coordinate_atom_C4_derivative_envelopes",
        "pass_full_K55_mixed_state_direction_C4_symbol_envelopes",
        "pass_R3_H6_spatialized_K55_P55_symbol_bounds",
        "fail_closed_requires_C6_spatial_frequency_symbol_bounds",
    )
    if tuple(record.get("status") for record in records) != expected_statuses:
        raise QuarticAnnularK55C6Error("candidate prerequisite status mismatch")

    contour_radius = sp.sympify(config["contour_radius"])
    covariant_direction = _covariant_direction_k55_c6(
        symmetrizer, moser, pde, raw, baseline, contour_radius
    )
    direction_ceilings = {
        pair: sp.Integer(sp.ceiling(value)) for pair, value in covariant_direction.items()
    }
    published_c4 = symbol_c4["K55_mixed_Frechet_2_norm_envelope_integer_ceilings"]
    c4_residuals = {
        _key(*pair): str(direction_ceilings[pair] - sp.Integer(published_c4[_key(*pair)]))
        for pair in _pairs(4)
    }
    if set(c4_residuals.values()) != {"0"}:
        raise QuarticAnnularK55C6Error("C6 hierarchy does not reproduce published C4 bounds")

    normalization = normalization_map_frechet_majorants(6)
    homogeneous = {
        pair: _composed_bound(direction_ceilings, pair[0], pair[1], normalization)
        for pair in _pairs(6)
    }
    coordinate_map = {
        order: sp.Integer(r3["coordinate_map_Frechet_integer_ceilings"][str(order)])
        for order in (1, 2)
    }
    coordinate: dict[tuple[int, int], sp.Expr] = {}
    for frequency_order in range(7):
        coordinate[(0, frequency_order)] = homogeneous[(0, frequency_order)]
    for frequency_order in range(6):
        coordinate[(1, frequency_order)] = homogeneous[(1, frequency_order)] * coordinate_map[1]
    for frequency_order in range(5):
        coordinate[(2, frequency_order)] = (
            homogeneous[(2, frequency_order)] * coordinate_map[1] ** 2
            + homogeneous[(1, frequency_order)] * coordinate_map[2]
        )

    energy = sp.Symbol("E", nonnegative=True, finite=True)
    c1 = r3_sobolev_embedding_constant(6, 1)
    c2 = r3_sobolev_embedding_constant(6, 2)
    spatial: dict[tuple[int, int], sp.Expr] = {}
    for frequency_order in range(7):
        spatial[(0, frequency_order)] = coordinate[(0, frequency_order)]
    for frequency_order in range(6):
        spatial[(1, frequency_order)] = sp.factor(coordinate[(1, frequency_order)] * c1 * energy)
    for frequency_order in range(5):
        spatial[(2, frequency_order)] = sp.factor(
            coordinate[(2, frequency_order)] * (c1 * energy) ** 2
            + coordinate[(1, frequency_order)] * c2 * energy
        )

    radius_lower = sp.sympify(config["annular_support_radius_lower"])
    radius_upper = sp.sympify(config["annular_support_radius_upper"])
    cutoff_bounds = annular_cutoff_frechet_majorants(
        6, radius_lower, sp.sympify(config["annular_ramp_width"])
    )
    volume = sp.factor(4 * sp.pi * (radius_upper**3 - radius_lower**3) / 3)
    localized_l1: dict[tuple[int, int], sp.Expr] = {}
    for spatial_order in range(3):
        for frequency_order in range(7 - spatial_order):
            localized_l1[(spatial_order, frequency_order)] = sp.factor(
                volume
                * sum(
                    comb(frequency_order, cutoff_order)
                    * cutoff_bounds[cutoff_order]
                    * spatial[(spatial_order, frequency_order - cutoff_order)]
                    / radius_lower ** (frequency_order - cutoff_order)
                    for cutoff_order in range(frequency_order + 1)
                )
            )

    sufficient_radius = sp.sympify(
        r3["sufficient_H6_radius_for_state_and_spatial_jet_tube"]["H6_radius"]
    )
    evaluated_l1 = {
        pair: sp.factor(value.subs(energy, sufficient_radius))
        for pair, value in localized_l1.items()
    }
    d = {
        order: sp.factor(
            sp.Rational(3, 4) * (evaluated_l1[(2, order)] + evaluated_l1[(0, order + 2)])
        )
        for order in range(5)
    }
    q = radius_upper + sp.sqrt(3 * sp.sympify(config["semiclassical_h_maximum"]) / 2)
    r_symbol = sp.Symbol("R", nonnegative=True, finite=True)
    p_bounds = {
        pair: sp.sympify(
            r3["spatialized_dyadic_P55_bounds"][_key(*pair)]["expression"],
            locals={"R": r_symbol},
        ).subs(r_symbol, sufficient_radius)
        for pair in ((0, 1), (1, 1))
    }
    a0 = p_bounds[(0, 1)]
    a1 = p_bounds[(1, 1)]
    s_bound = {
        order: sp.factor(2 * a0 * (3 * q * d[order] + (order * d[order - 1] if order else 0)))
        for order in (0, 2, 4)
    }
    c_bound = {
        order: sp.factor(
            9 * a1 * (q * evaluated_l1[(0, order + 1)] + (order + 1) * evaluated_l1[(0, order)])
        )
        for order in (0, 2, 4)
    }
    t_bound = {
        order: sp.factor(3 * (a0 * evaluated_l1[(1, order)] + a1 * evaluated_l1[(0, order)]))
        for order in (0, 2, 4)
    }
    amplitude = {
        order: sp.factor(s_bound[order] + c_bound[order] + t_bound[order]) for order in (0, 2, 4)
    }
    composition = sp.factor((amplitude[0] + 6 * amplitude[2] + 9 * amplitude[4]) / (8 * sp.pi))
    composition_numeric = float(sp.N(composition, 18))
    if not (composition_numeric > 0 and sp.Float(composition_numeric).is_finite):
        raise QuarticAnnularK55C6Error("composition constant is not finite positive")

    required_pairs = ((2, 4), (0, 6), (0, 5), (1, 4))
    return {
        "schema_version": "sigma-quartic-annular-k55-c6-certificate-1.0",
        "status": "pass_targeted_annular_K55_C6_principal_composition_constant",
        "candidate_id": candidate_id,
        "coefficients": symmetrizer.get("coefficients"),
        "C4_reproduction_residuals": c4_residuals,
        "required_spatial_frequency_K55_bounds": {
            _key(*pair): {
                "expression": str(spatial[pair]),
                "at_sufficient_H6_radius": str(spatial[pair].subs(energy, sufficient_radius)),
                "numeric": float(sp.N(spatial[pair].subs(energy, sufficient_radius), 18)),
            }
            for pair in required_pairs
        },
        "annular_cutoff": {
            "support": f"{radius_lower}<=|xi|<={radius_upper}",
            "volume": str(volume),
            "Frechet_majorants": {str(order): str(value) for order, value in cutoff_bounds.items()},
        },
        "localized_symbol_L1_frequency_bounds": {
            _key(*pair): str(evaluated_l1[pair])
            for pair in sorted(evaluated_l1)
            if pair[0] <= 2 and pair[1] <= 6 - pair[0]
        },
        "composition_inputs": {
            "sufficient_H6_radius": str(sufficient_radius),
            "A0_P_0_1": str(a0),
            "A1_P_1_1": str(a1),
            "Q": str(q),
            "D_m": {str(order): str(d[order]) for order in range(5)},
            "S_m": {str(order): str(value) for order, value in s_bound.items()},
            "C_m": {str(order): str(value) for order, value in c_bound.items()},
            "T_m": {str(order): str(value) for order, value in t_bound.items()},
            "R_m": {str(order): str(value) for order, value in amplitude.items()},
        },
        "principal_anti_wick_composition_constant": {
            "exact": str(composition),
            "numeric": composition_numeric,
            "bound": "||(i/h)(OpAW(rho*K)OpW(P)-adjoint)||<=C_comp",
        },
        "anti_wick_principal_composition_remainder_instantiated": True,
        "full_dyadic_energy_closed": False,
        "remaining_gate": (
            "insert_C_comp_into_frequency_localized_evolution_energy_with_"
            "projection_commutators_finite_low_modes_and_dyadic_sum"
        ),
        "scope": (
            "This instantiates only the high-frequency principal anti-Wick composition "
            "constant at the certified tube radius. It does not close projection "
            "commutators, bounded frequencies, lower-order sources, the dyadic sum, "
            "a nonlinear lifespan, matter, or observations."
        ),
    }


def run_quartic_annular_k55_c6_campaign(
    symmetrizer_campaign: dict[str, Any],
    moser_campaign: dict[str, Any],
    pde_campaign: dict[str, Any],
    tube_campaign: dict[str, Any],
    solved_campaign: dict[str, Any],
    full_campaign: dict[str, Any],
    symbol_c4_campaign: dict[str, Any],
    r3_campaign: dict[str, Any],
    anti_wick_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticAnnularK55C6Error("unsupported campaign schema_version")
        campaigns = {
            "symmetrizer": symmetrizer_campaign,
            "moser": moser_campaign,
            "pde": pde_campaign,
            "tube": tube_campaign,
            "solved": solved_campaign,
            "full": full_campaign,
            "symbol_c4": symbol_c4_campaign,
            "r3": r3_campaign,
            "anti_wick": anti_wick_campaign,
        }
        validate_bound_inputs(config, campaigns, CONFIG_KEYS)
        expected_statuses = {
            "symmetrizer": "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes",
            "moser": "pass_all_12_quasilinear_coefficient_derivative_envelopes",
            "pde": "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
            "tube": "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            "solved": "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "full": "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
            "symbol_c4": "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes",
            "r3": "pass_all_12_R3_H6_spatialized_K55_P55_symbol_bounds",
            "anti_wick": "pass_exact_anti_wick_composition_prerequisite_audit_C6_required",
        }
        for name, campaign in campaigns.items():
            if campaign.get("status") != expected_statuses[name]:
                raise QuarticAnnularK55C6Error(f"{name} prerequisite status mismatch")
            if not _content_hash_matches(campaign):
                raise QuarticAnnularK55C6Error(f"{name} campaign content hash mismatch")
        upstream = {name: campaign.get("content_sha256") for name, campaign in campaigns.items()}
        pde_upstream = pde_campaign.get("upstream_sha256", {})
        if (
            pde_upstream.get("symmetrizer") != upstream["symmetrizer"]
            or pde_upstream.get("moser") != upstream["moser"]
        ):
            raise QuarticAnnularK55C6Error("PDE provenance mismatch")
        if tube_campaign.get("nonquasilinear_pde_campaign_sha256") != upstream["pde"]:
            raise QuarticAnnularK55C6Error("tube provenance mismatch")
        solved_upstream = solved_campaign.get("upstream_sha256", {})
        if any(
            solved_upstream.get(name) != upstream[target]
            for name, target in (
                ("moser", "moser"),
                ("nonquasilinear_pde", "pde"),
                ("coordinate_tube", "tube"),
            )
        ):
            raise QuarticAnnularK55C6Error("solved-source provenance mismatch")
        for campaign_name, expected_links in (
            (
                "full",
                {
                    "symmetrizer": "symmetrizer",
                    "moser": "moser",
                    "nonquasilinear_pde": "pde",
                    "coordinate_tube": "tube",
                    "solved_source": "solved",
                },
            ),
            (
                "symbol_c4",
                {
                    "symmetrizer": "symmetrizer",
                    "moser": "moser",
                    "nonquasilinear_pde": "pde",
                    "coordinate_tube": "tube",
                    "solved_source": "solved",
                    "full_symmetrizer": "full",
                },
            ),
        ):
            links = campaigns[campaign_name].get("upstream_sha256", {})
            if any(links.get(key) != upstream[value] for key, value in expected_links.items()):
                raise QuarticAnnularK55C6Error(f"{campaign_name} provenance mismatch")
        if r3_campaign.get("upstream_sha256", {}).get("solved_source") != upstream["solved"]:
            raise QuarticAnnularK55C6Error("R3 provenance mismatch")
        if anti_wick_campaign.get("upstream_sha256", {}).get("r3_sobolev") != upstream["r3"]:
            raise QuarticAnnularK55C6Error("anti-Wick provenance mismatch")
        if (
            int(config["maximum_total_derivative_order"]) != 6
            or int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or sp.sympify(config["annular_support_radius_lower"]) != sp.Rational(5, 2)
            or sp.sympify(config["annular_support_radius_upper"]) != sp.Rational(35, 2)
            or sp.sympify(config["annular_ramp_width"]) != sp.Rational(1, 2)
            or sp.sympify(config["semiclassical_h_maximum"]) != sp.Rational(1, 16)
        ):
            raise QuarticAnnularK55C6Error("unsupported targeted C6 contract")
        control_passed, control = generic_annular_k55_c6_control()
        if not control_passed:
            raise QuarticAnnularK55C6Error("generic targeted C6 control failed")
        baseline_passed, baseline = quartic_horndeski_baseline_riesz_symmetrizer_control()
        if not baseline_passed:
            raise QuarticAnnularK55C6Error("baseline Riesz control failed")
        baseline_contract = baseline["quantitative_physical_group_perturbation_contract"]
        contour_radius = sp.sympify(config["contour_radius"])
        if contour_radius != sp.sympify(baseline_contract["contour_radius"]):
            raise QuarticAnnularK55C6Error("Riesz contour mismatch")
        raw = _uniform_raw_mixed_envelopes(str(config["covariant_state_component_radius"]), 6)
        record_maps = {name: _candidate_records(campaign) for name, campaign in campaigns.items()}
        candidate_ids = set(record_maps["symmetrizer"])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in record_maps.values()
        ):
            raise QuarticAnnularK55C6Error("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(record_maps[name][candidate_id] for name in campaigns),
                raw,
                baseline_contract,
                config,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_targeted_annular_K55_C6_principal_composition_constants",
            "errors": [],
            "upstream_sha256": upstream,
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_annular_k55_c6_control": control,
            "counts": {
                "selected": len(certificates),
                "targeted_C6_bounds_passed": len(certificates),
                "principal_composition_constants_instantiated": len(certificates),
                "full_dyadic_energies_closed": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have the targeted annular K55 C6 derivatives and a "
                "finite explicit principal anti-Wick composition constant. This is an "
                "operator-remainder input, not a closed dyadic or nonlinear energy theorem."
            ),
            "scope": certificates[0]["scope"],
            "data_seals": DATA_SEALS,
        }
    except (KeyError, TypeError, ValueError, QuarticAnnularK55C6Error) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "targeted_C6_bounds_passed": 0,
                "principal_composition_constants_instantiated": 0,
                "full_dyadic_energies_closed": 0,
                "rejected": 0,
            },
            "data_seals": DATA_SEALS,
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def validate_quartic_annular_k55_c6_artifact(
    artifact: dict[str, Any], root: Path, config: dict[str, Any]
) -> None:
    labels = (
        "symmetrizer",
        "moser",
        "pde",
        "tube",
        "solved",
        "full",
        "symbol_c4",
        "r3",
        "anti_wick",
    )
    loaded = load_bound_inputs(root, config, labels)
    rebuilt = run_quartic_annular_k55_c6_campaign(*(loaded[label] for label in labels), config)
    validate_exact_rebuild(artifact, rebuilt)


def write_quartic_annular_k55_c6_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
