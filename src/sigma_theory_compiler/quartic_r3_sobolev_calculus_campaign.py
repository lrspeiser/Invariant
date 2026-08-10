from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_homogeneous_frequency_symbol_campaign import _multiplicity_vectors

SCHEMA_VERSION = "sigma-quartic-r3-sobolev-calculus-campaign-1.0"


class QuarticR3SobolevCalculusError(ValueError):
    """Raised when the R3 Sobolev symbol calculus cannot be certified."""


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


def _pairs(maximum_total_order: int) -> list[tuple[int, int]]:
    return [
        (spatial_order, frequency_order)
        for total_order in range(maximum_total_order + 1)
        for spatial_order in range(total_order + 1)
        for frequency_order in (total_order - spatial_order,)
    ]


def _key(left: int, right: int) -> str:
    return f"{left},{right}"


def _partition_count(multiplicities: tuple[int, ...]) -> int:
    order = sum(
        derivative_order * count
        for derivative_order, count in enumerate(multiplicities, start=1)
    )
    denominator = 1
    for derivative_order, count in enumerate(multiplicities, start=1):
        denominator *= factorial(count) * factorial(derivative_order) ** count
    return factorial(order) // denominator


def r3_sobolev_embedding_constant(order: int, derivative_order: int) -> sp.Expr:
    """C in ||D^m f||_infinity <= C ||f||_H^s for the declared Fourier convention."""

    if order <= derivative_order + sp.Rational(3, 2):
        raise QuarticR3SobolevCalculusError(
            "Sobolev order must exceed derivative order plus 3/2"
        )
    square = sp.factor(
        sp.gamma(sp.Rational(2 * derivative_order + 3, 2))
        * sp.gamma(sp.Rational(2 * order - 2 * derivative_order - 3, 2))
        / (4 * sp.pi**2 * sp.gamma(order))
    )
    return sp.sqrt(square)


def ordered_spatial_jet_constants(
    order: int, maximum_derivative_order: int
) -> dict[int, sp.Expr]:
    return {
        derivative_order: sp.factor(
            sp.Integer(3) ** sp.Rational(derivative_order, 2)
            * r3_sobolev_embedding_constant(order, derivative_order)
        )
        for derivative_order in range(maximum_derivative_order + 1)
    }


def r3_sobolev_algebra_constant(order: int) -> sp.Expr:
    if order <= sp.Rational(3, 2):
        raise QuarticR3SobolevCalculusError("H^s is not an algebra at this order")
    return sp.factor(2**order * r3_sobolev_embedding_constant(order, 0))


def dyadic_ball_bernstein_constant(derivative_order: int) -> sp.Expr:
    """C_m for Fourier support |xi|<=2/h: ||D^m u||inf<=C_m h^-m-3/2||u||2."""

    return sp.factor(
        (2 * sp.pi) ** (-sp.Rational(3, 2))
        * sp.sqrt(sp.Rational(4, 3) * sp.pi)
        * 2 ** sp.Rational(2 * derivative_order + 3, 2)
    )


def _composition_coefficients(
    state_frequency_bounds: dict[tuple[int, int], sp.Integer],
    spatial_order: int,
    frequency_order: int,
    jet_constants: dict[int, sp.Expr],
) -> dict[int, sp.Expr]:
    if spatial_order == 0:
        return {0: state_frequency_bounds[(0, frequency_order)]}
    coefficients: dict[int, sp.Expr] = {}
    for multiplicities in _multiplicity_vectors(spatial_order):
        state_order = sum(multiplicities)
        coefficient = sp.Integer(_partition_count(multiplicities))
        for derivative_order, count in enumerate(multiplicities, start=1):
            coefficient *= jet_constants[derivative_order] ** count
        coefficient *= state_frequency_bounds[(state_order, frequency_order)]
        coefficients[state_order] = sp.factor(
            coefficients.get(state_order, sp.Integer(0)) + coefficient
        )
    return coefficients


def _time_composition_coefficients(
    state_frequency_bounds: dict[tuple[int, int], sp.Integer],
    spatial_order: int,
    frequency_order: int,
    jet_constants: dict[int, sp.Expr],
    time_jet_constants: dict[int, sp.Expr] | None = None,
) -> dict[int, sp.Expr]:
    """Coefficients of R_t sum_p c_p R^p for D_x^m partial_t F(U)."""

    coefficients: dict[int, sp.Expr] = {}
    marked_constants = time_jet_constants or jet_constants
    for time_block_spatial_order in range(spatial_order + 1):
        remaining = spatial_order - time_block_spatial_order
        for multiplicities in _multiplicity_vectors(remaining):
            ordinary_blocks = sum(multiplicities)
            state_order = ordinary_blocks + 1
            coefficient = (
                comb(spatial_order, time_block_spatial_order)
                * _partition_count(multiplicities)
                * marked_constants[time_block_spatial_order]
                * state_frequency_bounds[(state_order, frequency_order)]
            )
            for derivative_order, count in enumerate(multiplicities, start=1):
                coefficient *= jet_constants[derivative_order] ** count
            coefficients[ordinary_blocks] = sp.factor(
                coefficients.get(ordinary_blocks, sp.Integer(0)) + coefficient
            )
    return coefficients


def _polynomial_payload(coefficients: dict[int, sp.Expr]) -> dict[str, Any]:
    radius = sp.Symbol("R", nonnegative=True, finite=True)
    expression = sp.factor(
        sum(value * radius**power for power, value in coefficients.items())
    )
    return {
        "variable": "R=max_A ||Y_A||_H6 for the 153 coordinate-atom field Y",
        "coefficients_by_power": {
            str(power): str(value) for power, value in sorted(coefficients.items())
        },
        "expression": str(expression),
    }


@cache
def generic_r3_sobolev_chain_control() -> tuple[bool, dict[str, Any]]:
    """Verify exact R3 constants and spatial/time Faa-di-Bruno multiplicities."""

    order = 6
    maximum_order = 4
    embedding = {
        derivative_order: r3_sobolev_embedding_constant(order, derivative_order)
        for derivative_order in range(maximum_order + 1)
    }
    expected_squares = {
        0: sp.Rational(7, 1024) / sp.pi,
        1: sp.Rational(3, 1024) / sp.pi,
        2: sp.Rational(3, 1024) / sp.pi,
        3: sp.Rational(7, 1024) / sp.pi,
        4: sp.Rational(63, 1024) / sp.pi,
    }
    embedding_residuals = {
        str(derivative_order): str(
            sp.factor(embedding[derivative_order] ** 2 - expected_squares[derivative_order])
        )
        for derivative_order in range(maximum_order + 1)
    }
    binomial_weight_left = sum(
        comb(order, index) * sp.Rational(index, order)
        for index in range(order + 1)
    )
    binomial_weight_right = sum(
        comb(order, index) * sp.Rational(order - index, order)
        for index in range(order + 1)
    )

    t, epsilon = sp.symbols("t epsilon", real=True, finite=True)
    state_coefficients = [sp.Integer(index + 2) for index in range(maximum_order + 1)]
    time_coefficients = [sp.Integer(2 * index + 3) for index in range(maximum_order + 1)]
    outer_coefficients = [sp.Integer(5 * index + 7) for index in range(maximum_order + 2)]
    state_curve = sum(
        state_coefficients[index] * t**index / factorial(index)
        for index in range(1, maximum_order + 1)
    )
    time_curve = sum(
        time_coefficients[index] * t**index / factorial(index)
        for index in range(maximum_order + 1)
    )
    outer = sum(
        outer_coefficients[index] * sp.Symbol("q") ** index / factorial(index)
        for index in range(maximum_order + 2)
    )
    q = sp.Symbol("q")
    spatial_residuals: dict[str, str] = {}
    time_residuals: dict[str, str] = {}
    model_bounds = {
        (state_order, 0): outer_coefficients[state_order]
        for state_order in range(maximum_order + 2)
    }
    model_jets = {
        derivative_order: state_coefficients[derivative_order]
        for derivative_order in range(maximum_order + 1)
    }
    model_time_jets = {
        derivative_order: time_coefficients[derivative_order]
        for derivative_order in range(maximum_order + 1)
    }
    direct_spatial = outer.subs(q, state_curve)
    direct_time = sp.diff(
        outer.subs(q, state_curve + epsilon * time_curve), epsilon
    ).subs(epsilon, 0)
    for derivative_order in range(maximum_order + 1):
        if derivative_order == 0:
            recurrence = outer_coefficients[0]
        else:
            recurrence = sum(
                value
                for value in _composition_coefficients(
                    model_bounds, derivative_order, 0, model_jets
                ).values()
            )
        spatial_residuals[str(derivative_order)] = str(
            sp.expand(
                sp.diff(direct_spatial, t, derivative_order).subs(t, 0)
                - recurrence
            )
        )
        time_recurrence = sum(
            value
            for value in _time_composition_coefficients(
                model_bounds,
                derivative_order,
                0,
                model_jets,
                model_time_jets,
            ).values()
        )
        time_residuals[str(derivative_order)] = str(
            sp.expand(
                sp.diff(direct_time, t, derivative_order).subs(t, 0)
                - time_recurrence
            )
        )

    target = 3
    correct = sum(
        value
        for value in _time_composition_coefficients(
            model_bounds, target, 0, model_jets, model_time_jets
        ).values()
    )
    corrupted = correct - (
        comb(target, 1) - 1
    ) * outer_coefficients[2] * time_coefficients[1] * state_coefficients[2]
    corruption_residual = sp.expand(correct - corrupted)
    passed = bool(
        set(embedding_residuals.values()) == {"0"}
        and binomial_weight_left == 2 ** (order - 1)
        and binomial_weight_right == 2 ** (order - 1)
        and set(spatial_residuals.values()) == {"0"}
        and set(time_residuals.values()) == {"0"}
        and corruption_residual != 0
    )
    return passed, {
        "fourier_convention": {
            "forward": "fhat(xi)=integral exp(-i x.xi) f(x) dx",
            "inverse": "f(x)=(2*pi)^-3 integral exp(i x.xi) fhat(xi) dxi",
            "H6_norm_squared": "(2*pi)^-3 integral (1+|xi|^2)^6 |fhat|^2 dxi",
        },
        "embedding_identity": (
            "C_(s,m)^2=Gamma(m+3/2)Gamma(s-m-3/2)/(4*pi^2 Gamma(s))"
        ),
        "H6_embedding_constant_squares": {
            str(index): str(value) for index, value in expected_squares.items()
        },
        "embedding_residuals": embedding_residuals,
        "H6_algebra_constant": str(r3_sobolev_algebra_constant(order)),
        "weighted_convolution_binomial_sums": {
            "left": str(binomial_weight_left),
            "right": str(binomial_weight_right),
        },
        "spatial_chain_residuals": spatial_residuals,
        "time_chain_residuals": time_residuals,
        "negative_control": {
            "corruption": "replace choose(3,1)=3 by 1 for the marked time block",
            "exact_witness_residual": str(corruption_residual),
            "rejected": corruption_residual != 0,
        },
        "passed": passed,
        "scope": (
            "Exact Fourier integrals and scalar representatives verify the R3 embedding, "
            "algebra, spatial composition, and marked-time composition constants."
        ),
    }


def _certify_candidate(
    low_frequency: dict[str, Any],
    evolution: dict[str, Any],
    tube: dict[str, Any],
    solved_source: dict[str, Any],
    coordinate_jet: dict[str, Any],
    sobolev_order: int,
    maximum_order: int,
) -> dict[str, Any]:
    candidate_id = str(low_frequency.get("candidate_id"))
    candidates = (evolution, tube, solved_source)
    if any(item.get("candidate_id") != candidate_id for item in candidates):
        raise QuarticR3SobolevCalculusError("candidate ID mismatch")
    if any(
        item.get("coefficients") != low_frequency.get("coefficients")
        for item in candidates
    ):
        raise QuarticR3SobolevCalculusError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_global_C4_positive_K55_symbol_extension",
        "pass_full_55_state_degree_one_evolution_symbol_C4_bounds",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
    )
    if tuple(
        item.get("status")
        for item in (low_frequency, evolution, tube, solved_source)
    ) != expected_statuses:
        raise QuarticR3SobolevCalculusError("candidate prerequisite status mismatch")

    expected_pairs = set(_pairs(maximum_order))
    k_raw = low_frequency.get(
        "global_C4_frequency_derivative_integer_ceilings", {}
    )
    p_raw = evolution.get("homogeneous_principal_P55_bounds", {})
    if set(k_raw) != {_key(*pair) for pair in expected_pairs}:
        raise QuarticR3SobolevCalculusError("global K55 bounds are incomplete")
    if set(p_raw) != {_key(*pair) for pair in expected_pairs}:
        raise QuarticR3SobolevCalculusError("homogeneous P55 bounds are incomplete")
    k_covariant_bounds = {
        pair: sp.Integer(k_raw[_key(*pair)]["global_ceiling"])
        for pair in expected_pairs
    }
    p_covariant_bounds = {
        pair: sp.Integer(p_raw[_key(*pair)]["scaled_integer_ceiling"])
        * (2 if pair[1] == 0 else 1)
        for pair in expected_pairs
    }
    if coordinate_jet.get("input_norm") != "153-coordinate-atom component l_infinity":
        raise QuarticR3SobolevCalculusError("coordinate-jet input norm mismatch")
    coordinate_map_constants = {0: sp.Integer(1)} | {
        order: sp.ceiling(
            sp.sympify(coordinate_jet["envelopes"][str(order)]["exact"])
        )
        for order in range(1, maximum_order + 1)
    }
    k_bounds = {
        pair: (
            k_covariant_bounds[pair]
            if pair[0] == 0
            else sp.factor(
                sum(
                    _composition_coefficients(
                        k_covariant_bounds,
                        pair[0],
                        pair[1],
                        coordinate_map_constants,
                    ).values()
                )
            )
        )
        for pair in expected_pairs
    }
    p_bounds = {
        pair: (
            p_covariant_bounds[pair]
            if pair[0] == 0
            else sp.factor(
                sum(
                    _composition_coefficients(
                        p_covariant_bounds,
                        pair[0],
                        pair[1],
                        coordinate_map_constants,
                    ).values()
                )
            )
        )
        for pair in expected_pairs
    }
    jet_constants = {
        order: r3_sobolev_embedding_constant(sobolev_order, order)
        for order in range(maximum_order + 1)
    }
    k_spatial = {
        pair: _composition_coefficients(
            k_bounds, pair[0], pair[1], jet_constants
        )
        for pair in expected_pairs
    }
    p_spatial = {
        pair: _composition_coefficients(
            p_bounds, pair[0], pair[1], jet_constants
        )
        for pair in expected_pairs
    }
    time_pairs = set(_pairs(maximum_order - 1))
    k_time = {
        pair: _time_composition_coefficients(
            k_bounds, pair[0], pair[1], jet_constants
        )
        for pair in time_pairs
    }
    component_radius = sp.sympify(tube["coordinate_component_radius"])
    tube_h6_radius = sp.factor(
        component_radius / max(jet_constants[0], jet_constants[1], key=lambda x: float(sp.N(x)))
    )
    if not tube_h6_radius.is_positive:
        raise QuarticR3SobolevCalculusError("the sufficient H6 tube radius is not positive")
    return {
        "schema_version": "sigma-quartic-r3-sobolev-calculus-certificate-1.0",
        "status": "pass_R3_H6_spatialized_K55_P55_symbol_bounds",
        "candidate_id": candidate_id,
        "coefficients": low_frequency.get("coefficients"),
        "functional_setting": {
            "space": "R^3",
                "coordinate_atom_field": (
                    "Y=(metric deviation, first partials, acceleration-free second "
                    "partials), 153 components each in H^6(R^3)"
                ),
                "componentwise_norm": "R=max_A ||Y_A||_H6",
            "sobolev_order": sobolev_order,
            "maximum_L_infinity_spatial_derivative": maximum_order,
            "dyadic_semiclassical_shell": "1<=|xi|<=2",
        },
        "coordinate_map_Frechet_integer_ceilings": {
            str(index): str(value)
            for index, value in coordinate_map_constants.items()
            if index > 0
        },
        "per_coordinate_multiindex_spatial_embedding_constants": {
            str(index): str(value) for index, value in jet_constants.items()
        },
        "sufficient_H6_radius_for_state_and_spatial_jet_tube": {
            "coordinate_component_radius": str(component_radius),
            "H6_radius": str(tube_h6_radius),
            "H6_radius_numeric": float(sp.N(tube_h6_radius, 18)),
            "controlled_atoms": "153 coordinate atoms and each first spatial partial",
        },
        "spatialized_global_K55_bounds": {
            _key(*pair): _polynomial_payload(k_spatial[pair])
            for pair in _pairs(maximum_order)
        },
        "spatialized_dyadic_P55_bounds": {
            _key(*pair): _polynomial_payload(p_spatial[pair])
            for pair in _pairs(maximum_order)
        },
        "spatialized_time_K55_bounds": {
            _key(*pair): {
                **_polynomial_payload(k_time[pair]),
                "full_bound": (
                    "max_A ||partial_t Y_A||_H6 times the emitted polynomial"
                ),
            }
            for pair in _pairs(maximum_order - 1)
        },
        "claim": (
            "On R3, the candidate's global K55 and dyadic-shell P55 symbols have explicit "
            "L-infinity spatial/frequency derivative polynomials through total order four; "
            "the K55 time derivative has an explicit polynomial linear in the "
            "componentwise coordinate-atom time H6 budget."
        ),
        "remaining_gate": (
            "close_U_t_H6_from_evolution_then_explicit_operator_composition_and_dyadic_sum"
        ),
        "scope": (
            "This resolves the prior torus/R3 Sobolev mismatch and spatializes the certified "
            "symbol derivatives. It does not yet close the coordinate-atom time H6 budget, "
            "prove a Calderon-"
            "Vaillancourt/composition remainder, sum dyadic energies, or prove lifespan."
        ),
    }


def run_quartic_r3_sobolev_calculus_campaign(
    low_frequency_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    tube_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticR3SobolevCalculusError("unsupported campaign schema_version")
        expected_statuses = (
            "pass_all_12_global_C4_positive_K55_symbol_extensions",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
        )
        if tuple(
            campaign.get("status")
            for campaign in (
                low_frequency_campaign,
                evolution_campaign,
                tube_campaign,
                solved_source_campaign,
            )
        ) != expected_statuses:
            raise QuarticR3SobolevCalculusError("campaign prerequisite status mismatch")
        if not all(
            _content_hash_matches(campaign)
            for campaign in (
                low_frequency_campaign,
                evolution_campaign,
                tube_campaign,
                solved_source_campaign,
            )
        ):
            raise QuarticR3SobolevCalculusError("campaign content hash mismatch")
        if evolution_campaign.get("symbol_campaign_sha256") != (
            low_frequency_campaign.get("upstream_sha256", {}).get("symbol")
        ):
            raise QuarticR3SobolevCalculusError("K55/P55 symbol provenance mismatch")
        if solved_source_campaign.get("upstream_sha256", {}).get(
            "coordinate_tube"
        ) != tube_campaign.get("content_sha256"):
            raise QuarticR3SobolevCalculusError(
                "coordinate-tube to solved-source provenance mismatch"
            )
        sobolev_order = int(config["sobolev_order"])
        maximum_order = int(config["maximum_total_derivative_order"])
        if sobolev_order != 6 or maximum_order != 4:
            raise QuarticR3SobolevCalculusError(
                "R3 C4 symbol spatialization requires H6 and total order four"
            )
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
        ):
            raise QuarticR3SobolevCalculusError(
                "campaign requires R3, 55 evolution states, and 153 coordinate atoms"
            )
        control_passed, control = generic_r3_sobolev_chain_control()
        if not control_passed:
            raise QuarticR3SobolevCalculusError("generic R3 Sobolev control failed")
        maps = tuple(
            _candidate_records(campaign)
            for campaign in (
                low_frequency_campaign,
                evolution_campaign,
                tube_campaign,
                solved_source_campaign,
            )
        )
        expected = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(maps[0])
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticR3SobolevCalculusError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps),
                solved_source_campaign["coordinate_jet_Frechet_envelopes"],
                sobolev_order,
                maximum_order,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_R3_H6_spatialized_K55_P55_symbol_bounds",
            "errors": [],
            "upstream_sha256": {
                "low_frequency": low_frequency_campaign.get("content_sha256"),
                "evolution": evolution_campaign.get("content_sha256"),
                "coordinate_tube": tube_campaign.get("content_sha256"),
                "solved_source": solved_source_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_R3_sobolev_chain_control": control,
            "sobolev_constants": {
                "H6_algebra": str(r3_sobolev_algebra_constant(sobolev_order)),
                "H6_C4_embedding": {
                    str(index): str(
                        r3_sobolev_embedding_constant(sobolev_order, index)
                    )
                    for index in range(maximum_order + 1)
                },
                "dyadic_ball_Bernstein": {
                    str(index): str(dyadic_ball_bernstein_constant(index))
                    for index in range(maximum_order + 1)
                },
            },
            "counts": {
                "selected": len(certificates),
                "R3_symbol_spatializations_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have explicit Fourier-normalized R3 H6 "
                "spatialization polynomials for K55, dyadic P55, and d_t K55."
            ),
            "scope": certificates[0]["scope"],
            "primary_references": [
                {
                    "title": "Pseudo-differential operators with isotropic symbols, Wick and anti-Wick operators, and hypoellipticity",
                    "url": "https://arxiv.org/abs/2011.00313",
                },
                {
                    "title": "Integral formulas for the Weyl and anti-Wick symbols",
                    "url": "https://arxiv.org/abs/1806.04898",
                },
            ],
        }
    except (KeyError, TypeError, ValueError, QuarticR3SobolevCalculusError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "R3_symbol_spatializations_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_r3_sobolev_calculus_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
