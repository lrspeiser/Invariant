from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_homogeneous_frequency_symbol_campaign import (
    _composed_bound,
    normalization_map_frechet_majorants,
)
from .quartic_low_frequency_symbol_extension_campaign import (
    radius_map_frechet_majorants,
)

SCHEMA_VERSION = "sigma-quartic-evolution-symbol-campaign-1.0"


class QuarticEvolutionSymbolError(ValueError):
    """Raised when the complete first-order evolution symbol cannot be bounded."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == hashlib.sha256(
        _canonical_json(body).encode()
    ).hexdigest()


def _pairs(maximum_total_order: int) -> list[tuple[int, int]]:
    return [
        (state_order, frequency_order)
        for total_order in range(maximum_total_order + 1)
        for state_order in range(total_order + 1)
        for frequency_order in (total_order - state_order,)
    ]


def _key(state_order: int, frequency_order: int) -> str:
    return f"{state_order},{frequency_order}"


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _principal_symbol_bound(
    normalized_bounds: dict[tuple[int, int], sp.Integer],
    state_order: int,
    frequency_order: int,
    radius_bounds: dict[int, int],
) -> sp.Integer:
    """Bound D_U^a D_xi^b(|xi| M(U,xi/|xi|)) with exact Leibniz factors."""

    return sp.Integer(
        sum(
            comb(frequency_order, radial_order)
            * radius_bounds[radial_order]
            * normalized_bounds[(state_order, frequency_order - radial_order)]
            for radial_order in range(frequency_order + 1)
        )
    )


@cache
def generic_degree_one_evolution_symbol_control() -> tuple[bool, dict[str, Any]]:
    """Verify the 55-block norm and radial product recurrence through order four."""

    maximum_order = 4
    radius_bounds = radius_map_frechet_majorants(maximum_order)
    normalization_bounds = normalization_map_frechet_majorants(maximum_order)

    ell, em, x, y = sp.symbols("L M x y", nonnegative=True, finite=True)
    block_residual = sp.expand(
        (ell + em) ** 2 * (x**2 + y**2) - (ell * x + em * y) ** 2
    )
    block_positive_decomposition = sp.expand(
        (em * x - ell * y) ** 2 + 2 * ell * em * (x**2 + y**2)
    )

    u, t = sp.symbols("u t", real=True, finite=True)
    model = sum(
        sp.Integer(10 * state_order + frequency_order + 1)
        * u**state_order
        * t**frequency_order
        / (factorial(state_order) * factorial(frequency_order))
        for state_order, frequency_order in _pairs(maximum_order)
    )
    radius = sum(
        sp.Integer(radius_bounds[order]) * t**order / factorial(order)
        for order in range(maximum_order + 1)
    )
    residuals: dict[str, str] = {}
    for state_order, frequency_order in _pairs(maximum_order):
        direct = sp.diff(
            radius * model, u, state_order, t, frequency_order
        ).subs({u: 0, t: 0})
        recurrence = sum(
            comb(frequency_order, radial_order)
            * radius_bounds[radial_order]
            * sp.diff(
                model,
                u,
                state_order,
                t,
                frequency_order - radial_order,
            ).subs({u: 0, t: 0})
            for radial_order in range(frequency_order + 1)
        )
        residuals[_key(state_order, frequency_order)] = str(
            sp.expand(direct - recurrence)
        )

    target_state, target_frequency = 1, 2
    correct = sum(
        comb(target_frequency, radial_order)
        * radius_bounds[radial_order]
        * sp.diff(
            model,
            u,
            target_state,
            t,
            target_frequency - radial_order,
        ).subs({u: 0, t: 0})
        for radial_order in range(target_frequency + 1)
    )
    corrupted = sum(
        (1 if radial_order == 1 else comb(target_frequency, radial_order))
        * radius_bounds[radial_order]
        * sp.diff(
            model,
            u,
            target_state,
            t,
            target_frequency - radial_order,
        ).subs({u: 0, t: 0})
        for radial_order in range(target_frequency + 1)
    )
    corruption_residual = sp.expand(correct - corrupted)
    passed = bool(
        radius_bounds == {0: 1, 1: 1, 2: 2, 3: 6, 4: 36}
        and normalization_bounds == {0: 1, 1: 2, 2: 6, 3: 36, 4: 300}
        and sp.expand(block_residual - block_positive_decomposition) == 0
        and set(residuals.values()) == {"0"}
        and corruption_residual != 0
    )
    return passed, {
        "full_symbol": "M55(U,n)=[[0,0],[L(U,n),M22(U,n)]]",
        "principal_symbol": "P55(U,xi)=|xi| M55(U,xi/|xi|)",
        "block_operator_norm_bound": "||M55||_2<=||L||_2+||M22||_2",
        "block_scalar_residual": str(
            sp.expand(block_residual - block_positive_decomposition)
        ),
        "block_positive_decomposition": str(block_positive_decomposition),
        "radius_map_Frechet_majorants": {
            str(order): value for order, value in radius_bounds.items()
        },
        "normalization_map_Frechet_majorants": {
            str(order): value for order, value in normalization_bounds.items()
        },
        "radial_Leibniz_residuals": residuals,
        "negative_control": {
            "corruption": "replace binomial(2,1)=2 by 1 in D_xi^2(|xi|M)",
            "exact_witness_residual": str(corruption_residual),
            "rejected": corruption_residual != 0,
        },
        "passed": passed,
        "scope": (
            "Exact block and degree-one homogeneous recurrences. Candidate bounds are "
            "supplied separately from the certified L and M22 derivative envelopes."
        ),
    }


def _certify_candidate(
    first_order: dict[str, Any],
    nonquasilinear: dict[str, Any],
    symbol: dict[str, Any],
    maximum_order: int,
    normalization_bounds: dict[int, int],
    radius_bounds: dict[int, int],
) -> dict[str, Any]:
    candidate_id = str(symbol.get("candidate_id"))
    if first_order.get("candidate_id") != candidate_id or nonquasilinear.get(
        "candidate_id"
    ) != candidate_id:
        raise QuarticEvolutionSymbolError("candidate ID mismatch")
    if any(
        item.get("coefficients") != symbol.get("coefficients")
        for item in (first_order, nonquasilinear)
    ):
        raise QuarticEvolutionSymbolError("candidate coefficient mismatch")
    if first_order.get("status") != (
        "pass_exact_55_variable_principal_first_order_reduction"
    ):
        raise QuarticEvolutionSymbolError(
            f"candidate {candidate_id} lacks the exact physical-space reduction"
        )
    if symbol.get("status") != (
        "pass_full_K55_mixed_state_direction_C4_symbol_envelopes"
    ):
        raise QuarticEvolutionSymbolError(
            f"candidate {candidate_id} lacks mixed lifted-symbol envelopes"
        )
    if nonquasilinear.get("status") != (
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift"
    ):
        raise QuarticEvolutionSymbolError(
            f"candidate {candidate_id} lacks the exact 55-state lift"
        )
    dimensions = first_order.get("state_dimensions", {})
    if dimensions.get("physical_space_first_order") != 55:
        raise QuarticEvolutionSymbolError("physical-space state dimension is not 55")
    if dimensions.get("zero_speed_auxiliary") != 33:
        raise QuarticEvolutionSymbolError("zero/transverse block dimension is not 33")

    expected_pairs = set(_pairs(maximum_order))
    companion_raw = symbol.get(
        "companion_mixed_Frechet_2_norm_envelope_integer_ceilings", {}
    )
    transverse_raw = symbol.get(
        "L_mixed_Frechet_2_norm_envelope_integer_ceilings", {}
    )
    if set(companion_raw) != {_key(*pair) for pair in expected_pairs}:
        raise QuarticEvolutionSymbolError("companion integer ceilings are incomplete")
    if set(transverse_raw) != {_key(*pair) for pair in expected_pairs}:
        raise QuarticEvolutionSymbolError("transverse integer ceilings are incomplete")
    direction_bounds = {
        pair: sp.Integer(companion_raw[_key(*pair)])
        + sp.Integer(transverse_raw[_key(*pair)])
        for pair in expected_pairs
    }
    if any(value < 0 for value in direction_bounds.values()) or direction_bounds[
        (0, 0)
    ] <= 0:
        raise QuarticEvolutionSymbolError("a directional M55 bound is invalid")
    normalized_bounds = {
        pair: _composed_bound(
            direction_bounds,
            pair[0],
            pair[1],
            normalization_bounds,
        )
        for pair in expected_pairs
    }
    principal_bounds = {
        pair: _principal_symbol_bound(
            normalized_bounds,
            pair[0],
            pair[1],
            radius_bounds,
        )
        for pair in expected_pairs
    }
    if any(value < 0 for value in principal_bounds.values()) or principal_bounds[
        (0, 0)
    ] <= 0:
        raise QuarticEvolutionSymbolError("a homogeneous P55 bound is invalid")
    state_crosscheck = {
        str(order): {
            "M55_direction_integer_ceiling": str(direction_bounds[(order, 0)]),
            "P55_scaled_integer_ceiling": str(principal_bounds[(order, 0)]),
            "equal": direction_bounds[(order, 0)] == principal_bounds[(order, 0)],
        }
        for order in range(maximum_order + 1)
    }
    return {
        "schema_version": "sigma-quartic-evolution-symbol-certificate-1.0",
        "status": "pass_full_55_state_degree_one_evolution_symbol_C4_bounds",
        "candidate_id": candidate_id,
        "coefficients": symbol.get("coefficients"),
        "exact_reduction_provenance": {
            "source_spatial_block_sha256": first_order.get(
                "source_spatial_block_sha256"
            ),
            "state_dimension": 55,
            "zero_and_transverse_block_dimension": 33,
            "directional_companion_block_dimension": 22,
            "nonzero_characteristic_lift_residual_zero": first_order.get(
                "nonzero_characteristic_lift_residual_zero"
            ),
        },
        "directional_M55_integer_ceilings": {
            _key(*pair): str(direction_bounds[pair])
            for pair in _pairs(maximum_order)
        },
        "homogeneous_principal_P55_bounds": {
            _key(*pair): {
                "scaled_integer_ceiling": str(principal_bounds[pair]),
                "scaled_quantity": (
                    "|xi|^(|beta|-1) ||D_U^a partial_xi^beta P55(U,xi)||_2"
                ),
                "homogeneous_degree_after_frequency_derivatives": 1 - pair[1],
                "coordinate_multiindices_covered": comb(pair[1] + 2, 2),
            }
            for pair in _pairs(maximum_order)
        },
        "state_only_crosscheck": state_crosscheck,
        "claim": (
            "The exact lifted 55-state first-order principal symbol is degree one in xi "
            "and has outward integer mixed state/frequency bounds for every a+|beta|<=4."
        ),
        "remaining_gate": (
            "anti_wick_composition_time_commutator_bounded_frequency_dyadic_energy"
        ),
        "scope": (
            "This binds the actual 55-state principal evolution pencil to explicit symbol "
            "bounds. It does not yet estimate anti-Wick composition, time derivatives, "
            "lower-order nonlinear sources, dyadic summation, lifespan, or matter."
        ),
    }


def run_quartic_evolution_symbol_campaign(
    first_order_campaign: dict[str, Any],
    nonquasilinear_campaign: dict[str, Any],
    symbol_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticEvolutionSymbolError("unsupported campaign schema_version")
        if first_order_campaign.get("status") != (
            "pass_all_12_exact_55_variable_principal_first_order_reductions"
        ):
            raise QuarticEvolutionSymbolError("first-order campaign prerequisite failed")
        if symbol_campaign.get("status") != (
            "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes"
        ):
            raise QuarticEvolutionSymbolError("symbol campaign prerequisite failed")
        if nonquasilinear_campaign.get("status") != (
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts"
        ):
            raise QuarticEvolutionSymbolError(
                "nonquasilinear campaign prerequisite failed"
            )
        if not _content_hash_matches(first_order_campaign):
            raise QuarticEvolutionSymbolError("first-order campaign content hash mismatch")
        if not _content_hash_matches(nonquasilinear_campaign):
            raise QuarticEvolutionSymbolError(
                "nonquasilinear campaign content hash mismatch"
            )
        if not _content_hash_matches(symbol_campaign):
            raise QuarticEvolutionSymbolError("symbol campaign content hash mismatch")
        if nonquasilinear_campaign.get("upstream_sha256", {}).get(
            "first_order"
        ) != first_order_campaign.get("content_sha256"):
            raise QuarticEvolutionSymbolError(
                "first-order to nonquasilinear provenance mismatch"
            )
        if symbol_campaign.get("upstream_sha256", {}).get(
            "nonquasilinear_pde"
        ) != nonquasilinear_campaign.get("content_sha256"):
            raise QuarticEvolutionSymbolError(
                "nonquasilinear to symbol provenance mismatch"
            )
        maximum_order = int(config["maximum_total_derivative_order"])
        if maximum_order != 4:
            raise QuarticEvolutionSymbolError("evolution symbol requires total order four")
        if int(config["spatial_dimension"]) != 3:
            raise QuarticEvolutionSymbolError("evolution symbol requires three dimensions")
        if int(config["state_dimension"]) != 55:
            raise QuarticEvolutionSymbolError("evolution symbol requires 55 states")
        control_passed, control = generic_degree_one_evolution_symbol_control()
        if not control_passed:
            raise QuarticEvolutionSymbolError("generic evolution-symbol control failed")
        first_records = _candidate_records(first_order_campaign)
        nonquasilinear_records = _candidate_records(nonquasilinear_campaign)
        symbol_records = _candidate_records(symbol_campaign)
        expected = int(config.get("expected_candidate_count", 12))
        if (
            len(first_records) != expected
            or set(first_records) != set(nonquasilinear_records)
            or set(first_records) != set(symbol_records)
        ):
            raise QuarticEvolutionSymbolError("candidate-set mismatch")
        normalization_bounds = normalization_map_frechet_majorants(maximum_order)
        radius_bounds = radius_map_frechet_majorants(maximum_order)
        certificates = [
            _certify_candidate(
                first_records[candidate_id],
                nonquasilinear_records[candidate_id],
                symbol_records[candidate_id],
                maximum_order,
                normalization_bounds,
                radius_bounds,
            )
            for candidate_id in sorted(first_records)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "errors": [],
            "first_order_campaign_sha256": first_order_campaign.get("content_sha256"),
            "nonquasilinear_campaign_sha256": nonquasilinear_campaign.get(
                "content_sha256"
            ),
            "symbol_campaign_sha256": symbol_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_degree_one_evolution_symbol_control": control,
            "counts": {
                "selected": len(certificates),
                "evolution_symbol_bounds_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates bind the exact 55-state physical-space "
                "reduction to degree-one homogeneous principal-symbol C4 bounds."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticEvolutionSymbolError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "evolution_symbol_bounds_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_evolution_symbol_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
