from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_homogeneous_frequency_symbol_campaign import (
    _bell_coefficient,
    _multiplicity_vectors,
)

SCHEMA_VERSION = "sigma-quartic-low-frequency-symbol-extension-campaign-1.0"


class QuarticLowFrequencySymbolExtensionError(ValueError):
    """Raised when a global finite-regularity symmetrizer symbol cannot be certified."""


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


def smoothstep_polynomial() -> sp.Expr:
    t = sp.Symbol("t", real=True)
    return 126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9


def cutoff_derivative_majorants(maximum_order: int = 4) -> dict[int, int]:
    """Coefficient-l1 bounds for the transition polynomial on 0<=t<=1."""

    t = sp.Symbol("t", real=True)
    cutoff = smoothstep_polynomial()
    result = {0: 1}
    for order in range(1, maximum_order + 1):
        polynomial = sp.Poly(sp.diff(cutoff, t, order), t)
        result[order] = int(sum(abs(value) for _, value in polynomial.terms()))
    return result


def radius_map_frechet_majorants(maximum_order: int = 4) -> dict[int, int]:
    """Bound D^k |xi| by R_k |xi|^(1-k) in Euclidean operator norm."""

    result: dict[int, int] = {}
    for order in range(maximum_order + 1):
        total = sp.Rational(0)
        for pair_blocks in range(order // 2 + 1):
            singleton_blocks = order - 2 * pair_blocks
            block_count = singleton_blocks + pair_blocks
            falling = sp.prod(
                sp.Rational(1, 2) - index for index in range(block_count)
            )
            partition_count = factorial(order) // (
                factorial(singleton_blocks)
                * factorial(pair_blocks)
                * 2**pair_blocks
            )
            total += partition_count * abs(falling) * 2**block_count
        result[order] = int(total)
    return result


def radial_cutoff_frechet_majorants(maximum_order: int = 4) -> dict[int, int]:
    cutoff = cutoff_derivative_majorants(maximum_order)
    radius = radius_map_frechet_majorants(maximum_order)
    result = {0: 1}
    for order in range(1, maximum_order + 1):
        total = 0
        for multiplicities in _multiplicity_vectors(order):
            outer_order = sum(multiplicities)
            product = 1
            for derivative_order, count in enumerate(multiplicities, start=1):
                product *= radius[derivative_order] ** count
            total += (
                _bell_coefficient(multiplicities)
                * cutoff[outer_order]
                * product
            )
        result[order] = total
    return result


@cache
def generic_low_frequency_extension_control() -> tuple[bool, dict[str, Any]]:
    """Verify endpoint gluing, positivity weights, and radial chain-rule constants."""

    maximum_order = 4
    t, r = sp.symbols("t r", real=True)
    cutoff = smoothstep_polynomial()
    derivative = sp.factor(sp.diff(cutoff, t))
    endpoint_residuals = {
        "value_at_inner": str(cutoff.subs(t, 0)),
        "value_at_outer_minus_one": str(sp.expand(cutoff.subs(t, 1) - 1)),
        **{
            f"derivative_{order}_at_inner": str(
                sp.diff(cutoff, t, order).subs(t, 0)
            )
            for order in range(1, maximum_order + 1)
        },
        **{
            f"derivative_{order}_at_outer": str(
                sp.diff(cutoff, t, order).subs(t, 1)
            )
            for order in range(1, maximum_order + 1)
        },
    }
    cutoff_bounds = cutoff_derivative_majorants(maximum_order)
    radius_bounds = radius_map_frechet_majorants(maximum_order)
    radial_bounds = radial_cutoff_frechet_majorants(maximum_order)

    scalar_outer = sum(
        sp.Integer(10 + order) * t**order / factorial(order)
        for order in range(maximum_order + 1)
    )
    scalar_radius = sum(
        sp.Integer(radius_bounds[order]) * r**order / factorial(order)
        for order in range(maximum_order + 1)
    )
    residuals: dict[str, str] = {}
    for order in range(maximum_order + 1):
        direct = sp.diff(scalar_outer.subs(t, scalar_radius), r, order).subs(r, 0)
        if order == 0:
            recurrence = scalar_outer.subs(t, radius_bounds[0])
        else:
            recurrence = sp.Integer(0)
            for multiplicities in _multiplicity_vectors(order):
                outer_order = sum(multiplicities)
                product = sp.Integer(1)
                for derivative_order, count in enumerate(multiplicities, start=1):
                    product *= radius_bounds[derivative_order] ** count
                recurrence += (
                    _bell_coefficient(multiplicities)
                    * sp.diff(scalar_outer, t, outer_order).subs(
                        t, radius_bounds[0]
                    )
                    * product
                )
        residuals[str(order)] = str(sp.expand(direct - recurrence))

    corrupted = sp.expand(cutoff - t**9)
    corrupted_endpoint = sp.expand(corrupted.subs(t, 1) - 1)
    passed = bool(
        set(endpoint_residuals.values()) == {"0"}
        and derivative == 630 * t**4 * (t - 1) ** 4
        and cutoff_bounds
        == {0: 1, 1: 10080, 2: 60480, 3: 312480, 4: 1360800}
        and radius_bounds == {0: 1, 1: 1, 2: 2, 3: 6, 4: 36}
        and radial_bounds == {0: 1, 1: 10080, 2: 80640, 3: 735840, 4: 7650720}
        and set(residuals.values()) == {"0"}
        and corrupted_endpoint != 0
    )
    return passed, {
        "cutoff": (
            "chi(t)=0 for t<=0; 126t^5-420t^6+540t^7-315t^8+70t^9 "
            "for 0<t<1; chi(t)=1 for t>=1"
        ),
        "transition_variable": "t=|xi|-1",
        "cutoff_derivative_factorization": str(derivative),
        "monotonicity_and_range": (
            "chi'(t)=630 t^4(1-t)^4>=0 and chi(0)=0, chi(1)=1, so 0<=chi<=1"
        ),
        "endpoint_C4_residuals": endpoint_residuals,
        "cutoff_derivative_majorants": {
            str(order): value for order, value in cutoff_bounds.items()
        },
        "radius_map_Frechet_majorants": {
            str(order): value for order, value in radius_bounds.items()
        },
        "radial_cutoff_Frechet_majorants": {
            str(order): value for order, value in radial_bounds.items()
        },
        "bell_chain_rule_residuals": residuals,
        "negative_control": {
            "corruption": "replace the t^9 coefficient 70 by 69",
            "outer_endpoint_residual": str(corrupted_endpoint),
            "rejected": corrupted_endpoint != 0,
        },
        "passed": passed,
        "scope": (
            "Exact C4 radial gluing and conservative Euclidean Frechet majorants; no "
            "pseudodifferential operator theorem is asserted here."
        ),
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _certify_candidate(
    homogeneous: dict[str, Any],
    full_symmetrizer: dict[str, Any],
    maximum_order: int,
    radial_cutoff_bounds: dict[int, int],
) -> dict[str, Any]:
    candidate_id = str(homogeneous.get("candidate_id"))
    if full_symmetrizer.get("candidate_id") != candidate_id:
        raise QuarticLowFrequencySymbolExtensionError("candidate ID mismatch")
    if homogeneous.get("status") != "pass_full_K55_homogeneous_frequency_C4_bounds":
        raise QuarticLowFrequencySymbolExtensionError(
            f"candidate {candidate_id} lacks homogeneous frequency bounds"
        )
    if full_symmetrizer.get("status") != (
        "pass_full_K55_coordinate_atom_C4_derivative_envelopes"
    ):
        raise QuarticLowFrequencySymbolExtensionError(
            f"candidate {candidate_id} lacks full symmetrizer energy bounds"
        )
    source = homogeneous.get("homogeneous_frequency_K55_bounds", {})
    high_bounds = {
        pair: sp.Integer(source[_key(*pair)]["scaled_integer_ceiling"])
        for pair in _pairs(maximum_order)
    }
    global_bounds: dict[tuple[int, int], sp.Integer] = {}
    transition_bounds: dict[tuple[int, int], sp.Integer] = {}
    for state_order, frequency_order in _pairs(maximum_order):
        if frequency_order == 0:
            transition = high_bounds[(state_order, 0)]
        else:
            transition = sp.Integer(0)
            for cutoff_order in range(frequency_order + 1):
                symbol_order = frequency_order - cutoff_order
                symbol_bound = (
                    2 * high_bounds[(state_order, 0)]
                    if symbol_order == 0
                    else high_bounds[(state_order, symbol_order)]
                )
                transition += (
                    comb(frequency_order, cutoff_order)
                    * radial_cutoff_bounds[cutoff_order]
                    * symbol_bound
                )
        transition_bounds[(state_order, frequency_order)] = sp.Integer(transition)
        global_bounds[(state_order, frequency_order)] = max(
            high_bounds[(state_order, frequency_order)], sp.Integer(transition)
        )

    energy = full_symmetrizer.get("energy_equivalence", {})
    lower_numeric = float(sp.N(sp.sympify(energy["K55_2_lower"]), 18))
    upper_numeric = float(sp.N(sp.sympify(energy["K55_2_upper"]), 18))
    if not (lower_numeric > 0 and upper_numeric >= lower_numeric):
        raise QuarticLowFrequencySymbolExtensionError(
            f"candidate {candidate_id} lacks positive energy equivalence"
        )
    return {
        "schema_version": "sigma-quartic-low-frequency-symbol-extension-certificate-1.0",
        "status": "pass_global_C4_positive_K55_symbol_extension",
        "candidate_id": candidate_id,
        "coefficients": homogeneous.get("coefficients"),
        "extension_definition": {
            "reference_direction": "e_1=(1,0,0)",
            "inner_region": "K_ext(U,xi)=K55(U,e_1) for |xi|<=1",
            "transition_region": (
                "K_ext=(1-chi(|xi|-1))K55(U,e_1)+"
                "chi(|xi|-1)K55(U,xi/|xi|) for 1<|xi|<2"
            ),
            "outer_region": "K_ext(U,xi)=K55(U,xi/|xi|) for |xi|>=2",
            "regularity": "C4 in xi and C4 in the certified state variables",
        },
        "energy_equivalence": {
            "K55_2_lower": energy["K55_2_lower"],
            "K55_2_lower_numeric": lower_numeric,
            "K55_2_upper": energy["K55_2_upper"],
            "K55_2_upper_numeric": upper_numeric,
            "preservation_reason": (
                "The extension is a convex combination of Hermitian symmetrizers with "
                "the same uniform lower and upper bounds."
            ),
        },
        "global_C4_frequency_derivative_integer_ceilings": {
            _key(*pair): {
                "high_frequency_ceiling": str(high_bounds[pair]),
                "transition_ceiling": str(transition_bounds[pair]),
                "global_ceiling": str(global_bounds[pair]),
                "numeric": float(sp.N(global_bounds[pair], 18)),
                "coordinate_multiindices_covered": comb(pair[1] + 2, 2),
            }
            for pair in _pairs(maximum_order)
        },
        "claim": (
            "K_ext is a globally defined positive Hermitian C4 symbol with explicit mixed "
            "state/frequency bounds; it exactly equals the directional symmetrizer for |xi|>=2."
        ),
        "remaining_gate": (
            "positive_quantization_operator_constants_commuted_energy_and_lifespan"
        ),
        "scope": (
            "The principal symmetrization identity is exact only for |xi|>=2; its defect is "
            "confined to bounded frequencies. This campaign does not yet choose a positive "
            "quantization, prove an operator norm or Gårding estimate, close an energy "
            "inequality, prove a lifespan, or include matter."
        ),
    }


def run_quartic_low_frequency_symbol_extension_campaign(
    homogeneous_campaign: dict[str, Any],
    symbol_campaign: dict[str, Any],
    full_symmetrizer_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticLowFrequencySymbolExtensionError(
                "unsupported campaign schema_version"
            )
        expected_statuses = (
            (
                homogeneous_campaign,
                "pass_all_12_full_K55_homogeneous_frequency_C4_bounds",
            ),
            (
                symbol_campaign,
                "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes",
            ),
            (
                full_symmetrizer_campaign,
                "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
            ),
        )
        if any(campaign.get("status") != status for campaign, status in expected_statuses):
            raise QuarticLowFrequencySymbolExtensionError("campaign prerequisite failed")
        if not all(
            _content_hash_matches(campaign)
            for campaign in (
                homogeneous_campaign,
                symbol_campaign,
                full_symmetrizer_campaign,
            )
        ):
            raise QuarticLowFrequencySymbolExtensionError(
                "campaign content hash mismatch"
            )
        if homogeneous_campaign.get("symbol_campaign_sha256") != symbol_campaign.get(
            "content_sha256"
        ):
            raise QuarticLowFrequencySymbolExtensionError(
                "homogeneous-symbol provenance mismatch"
            )
        if symbol_campaign.get("upstream_sha256", {}).get(
            "full_symmetrizer"
        ) != full_symmetrizer_campaign.get("content_sha256"):
            raise QuarticLowFrequencySymbolExtensionError(
                "full-symmetrizer provenance mismatch"
            )
        maximum_order = int(config["maximum_total_derivative_order"])
        if maximum_order != 4:
            raise QuarticLowFrequencySymbolExtensionError(
                "low-frequency extension requires total order four"
            )
        if int(config["spatial_dimension"]) != 3:
            raise QuarticLowFrequencySymbolExtensionError(
                "the 55-state reduction requires three spatial dimensions"
            )
        inner_radius = sp.sympify(config["inner_radius"])
        outer_radius = sp.sympify(config["outer_radius"])
        if inner_radius != 1 or outer_radius != 2:
            raise QuarticLowFrequencySymbolExtensionError(
                "the certified cutoff requires inner radius one and outer radius two"
            )
        if int(config["cutoff_matching_order"]) != 4:
            raise QuarticLowFrequencySymbolExtensionError(
                "cutoff must match through fourth order"
            )
        control_passed, control = generic_low_frequency_extension_control()
        if not control_passed:
            raise QuarticLowFrequencySymbolExtensionError(
                "generic low-frequency extension control failed"
            )
        homogeneous_records = _candidate_records(homogeneous_campaign)
        full_records = _candidate_records(full_symmetrizer_campaign)
        expected = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(homogeneous_records)
        if (
            len(candidate_ids) != expected
            or set(full_records) != candidate_ids
        ):
            raise QuarticLowFrequencySymbolExtensionError("candidate-set mismatch")
        radial_bounds = radial_cutoff_frechet_majorants(maximum_order)
        certificates = [
            _certify_candidate(
                homogeneous_records[candidate_id],
                full_records[candidate_id],
                maximum_order,
                radial_bounds,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_global_C4_positive_K55_symbol_extensions",
            "errors": [],
            "upstream_sha256": {
                "homogeneous_frequency": homogeneous_campaign.get("content_sha256"),
                "symbol": symbol_campaign.get("content_sha256"),
                "full_symmetrizer": full_symmetrizer_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_low_frequency_extension_control": control,
            "counts": {
                "selected": len(certificates),
                "global_C4_positive_symbol_extensions_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have a global positive Hermitian C4 K55 symbol "
                "extension with explicit mixed state/frequency bounds."
            ),
            "scope": certificates[0]["scope"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticLowFrequencySymbolExtensionError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "global_C4_positive_symbol_extensions_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_low_frequency_symbol_extension_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
