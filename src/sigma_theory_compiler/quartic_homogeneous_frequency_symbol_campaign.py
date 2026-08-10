from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-homogeneous-frequency-symbol-campaign-1.0"


class QuarticHomogeneousFrequencySymbolError(ValueError):
    """Raised when the homogeneous frequency chart cannot be certified."""


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


def _odd_double_factorial(order: int) -> int:
    if order <= 0:
        return 1
    result = 1
    for value in range(1, order + 1, 2):
        result *= value
    return result


def _multiplicity_vectors(order: int) -> list[tuple[int, ...]]:
    """Return m_j with sum(j*m_j)=order, in j=1,...,order coordinates."""

    if order == 0:
        return [()]
    result: list[tuple[int, ...]] = []

    def visit(j: int, remaining: int, values: list[int]) -> None:
        if j == 0:
            if remaining == 0:
                result.append(tuple(reversed(values)))
            return
        for count in range(remaining // j + 1):
            visit(j - 1, remaining - count * j, [*values, count])

    visit(order, order, [])
    return result


def _bell_coefficient(multiplicities: tuple[int, ...]) -> int:
    order = sum((index + 1) * count for index, count in enumerate(multiplicities))
    denominator = 1
    for index, count in enumerate(multiplicities, start=1):
        denominator *= factorial(count) * factorial(index) ** count
    return factorial(order) // denominator


def inverse_radius_frechet_majorants(maximum_order: int = 4) -> dict[int, int]:
    """Bound D^k |xi|^-1 by c_k |xi|^(-k-1) in Euclidean operator norm."""

    majorants: dict[int, int] = {}
    for order in range(maximum_order + 1):
        total = 0
        for pair_blocks in range(order // 2 + 1):
            singleton_blocks = order - 2 * pair_blocks
            block_count = singleton_blocks + pair_blocks
            partition_count = factorial(order) // (
                factorial(singleton_blocks)
                * factorial(pair_blocks)
                * 2**pair_blocks
            )
            total += partition_count * _odd_double_factorial(2 * block_count - 1)
        majorants[order] = total
    return majorants


def normalization_map_frechet_majorants(maximum_order: int = 4) -> dict[int, int]:
    """Bound D^k(xi/|xi|) by N_k |xi|^-k in Euclidean operator norm."""

    inverse_radius = inverse_radius_frechet_majorants(maximum_order)
    return {
        0: 1,
        **{
            order: inverse_radius[order] + order * inverse_radius[order - 1]
            for order in range(1, maximum_order + 1)
        },
    }


def _composed_bound(
    direction_bounds: dict[tuple[int, int], sp.Integer],
    state_order: int,
    frequency_order: int,
    normalization_bounds: dict[int, int],
) -> sp.Integer:
    if frequency_order == 0:
        return direction_bounds[(state_order, 0)]
    total = sp.Integer(0)
    for multiplicities in _multiplicity_vectors(frequency_order):
        direction_order = sum(multiplicities)
        product = sp.Integer(1)
        for derivative_order, count in enumerate(multiplicities, start=1):
            product *= normalization_bounds[derivative_order] ** count
        total += (
            _bell_coefficient(multiplicities)
            * direction_bounds[(state_order, direction_order)]
            * product
        )
    return sp.Integer(total)


@cache
def generic_homogeneous_frequency_chain_rule_control() -> tuple[bool, dict[str, Any]]:
    """Verify normalization constants and Bell multiplicities through total order four."""

    maximum_order = 4
    inverse_radius = inverse_radius_frechet_majorants(maximum_order)
    normalization = normalization_map_frechet_majorants(maximum_order)
    expected_inverse = {0: 1, 1: 1, 2: 4, 3: 24, 4: 204}
    expected_normalization = {0: 1, 1: 2, 2: 6, 3: 36, 4: 300}

    u, t, n = sp.symbols("u t n", real=True)
    normalized_curve = sum(
        sp.Integer(normalization[order]) * t**order / factorial(order)
        for order in range(maximum_order + 1)
    )
    outer = sum(
        sp.Integer(10 * state_order + direction_order + 1)
        * u**state_order
        * n**direction_order
        / (factorial(state_order) * factorial(direction_order))
        for state_order, direction_order in _pairs(maximum_order)
    )
    residuals: dict[str, str] = {}
    for state_order, frequency_order in _pairs(maximum_order):
        direct = sp.diff(
            outer.subs(n, normalized_curve),
            u,
            state_order,
            t,
            frequency_order,
        ).subs({u: 0, t: 0})
        if frequency_order == 0:
            recurrence = sp.diff(outer, u, state_order).subs({u: 0, n: 1})
        else:
            recurrence = sp.Integer(0)
            for multiplicities in _multiplicity_vectors(frequency_order):
                direction_order = sum(multiplicities)
                product = sp.Integer(1)
                for derivative_order, count in enumerate(multiplicities, start=1):
                    product *= normalization[derivative_order] ** count
                recurrence += (
                    _bell_coefficient(multiplicities)
                    * sp.diff(
                        outer,
                        u,
                        state_order,
                        n,
                        direction_order,
                    ).subs({u: 0, n: 1})
                    * product
                )
        residuals[_key(state_order, frequency_order)] = str(sp.expand(direct - recurrence))

    corrupted = dict(normalization)
    corrupted[4] -= 1
    direct_fourth = sp.diff(outer.subs(n, normalized_curve), t, 4).subs(
        {u: 0, t: 0}
    )
    corrupt_curve = sum(
        sp.Integer(corrupted[order]) * t**order / factorial(order)
        for order in range(maximum_order + 1)
    )
    corrupted_fourth = sp.diff(outer.subs(n, corrupt_curve), t, 4).subs(
        {u: 0, t: 0}
    )
    corruption_residual = sp.expand(direct_fourth - corrupted_fourth)
    passed = bool(
        inverse_radius == expected_inverse
        and normalization == expected_normalization
        and set(residuals.values()) == {"0"}
        and corruption_residual != 0
    )
    return passed, {
        "inverse_radius_Frechet_majorants": {
            str(order): value for order, value in inverse_radius.items()
        },
        "normalization_map_Frechet_majorants": {
            str(order): value for order, value in normalization.items()
        },
        "normalization_identity": "n(xi)=xi/|xi|",
        "inverse_radius_derivation": (
            "Set partitions of D^k f(xi.xi) have singleton and pair blocks; "
            "their exact absolute coefficient sum gives c_k."
        ),
        "normalization_product_rule": "N_k=c_k+k*c_(k-1)",
        "bell_chain_rule_residuals": residuals,
        "multiindex_count": len(_pairs(maximum_order)),
        "negative_control": {
            "corruption": "replace the derived fourth normalization majorant 300 by 299",
            "exact_witness_residual": str(corruption_residual),
            "rejected": corruption_residual != 0,
        },
        "passed": passed,
        "scope": (
            "Exact set-partition and scalar Bell-polynomial controls validate the operator-norm "
            "majorant composition used for xi != 0."
        ),
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _certify_candidate(
    candidate: dict[str, Any],
    maximum_order: int,
    frequency_radius_lower: sp.Expr,
    normalization_bounds: dict[int, int],
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id"))
    if candidate.get("status") != (
        "pass_full_K55_mixed_state_direction_C4_symbol_envelopes"
    ):
        raise QuarticHomogeneousFrequencySymbolError(
            f"candidate {candidate_id} lacks mixed symbol envelopes"
        )
    raw_ceilings = candidate.get(
        "K55_mixed_Frechet_2_norm_envelope_integer_ceilings"
    )
    raw_numeric = candidate.get("K55_mixed_Frechet_2_norm_envelopes_numeric")
    if not isinstance(raw_ceilings, dict):
        raise QuarticHomogeneousFrequencySymbolError(
            f"candidate {candidate_id} lacks outward integer ceilings"
        )
    if not isinstance(raw_numeric, dict):
        raise QuarticHomogeneousFrequencySymbolError(
            f"candidate {candidate_id} lacks numeric crosschecks"
        )
    expected_pairs = set(_pairs(maximum_order))
    direction_bounds = {
        pair: sp.Integer(raw_ceilings[_key(*pair)]) for pair in expected_pairs
    }
    if any(value <= 0 for value in direction_bounds.values()):
        raise QuarticHomogeneousFrequencySymbolError(
            f"candidate {candidate_id} has a nonpositive direction ceiling"
        )
    if any(
        direction_bounds[pair] * 10**15
        < sp.Rational(str(raw_numeric[_key(*pair)])) * (10**15 - 1)
        for pair in expected_pairs
    ):
        raise QuarticHomogeneousFrequencySymbolError(
            f"candidate {candidate_id} has a non-outward integer ceiling"
        )
    homogeneous_bounds = {
        pair: _composed_bound(
            direction_bounds,
            pair[0],
            pair[1],
            normalization_bounds,
        )
        for pair in expected_pairs
    }
    if any(value <= 0 for value in homogeneous_bounds.values()):
        raise QuarticHomogeneousFrequencySymbolError(
            f"candidate {candidate_id} has a nonpositive xi bound"
        )
    radius_bounds = {
        pair: sp.factor(value / frequency_radius_lower ** pair[1])
        for pair, value in homogeneous_bounds.items()
    }
    state_crosscheck = {
        str(order): {
            "direction_integer_ceiling": str(direction_bounds[(order, 0)]),
            "frequency_integer_ceiling": str(homogeneous_bounds[(order, 0)]),
            "equal": direction_bounds[(order, 0)] == homogeneous_bounds[(order, 0)],
        }
        for order in range(maximum_order + 1)
    }
    return {
        "schema_version": "sigma-quartic-homogeneous-frequency-symbol-certificate-1.0",
        "status": "pass_full_K55_homogeneous_frequency_C4_bounds",
        "candidate_id": candidate_id,
        "coefficients": candidate.get("coefficients"),
        "frequency_domain": {
            "spatial_dimension": 3,
            "frequency_radius_lower": str(frequency_radius_lower),
            "normalization": "n=xi/|xi|",
            "excluded_point": "xi=0",
        },
        "homogeneous_frequency_K55_bounds": {
            _key(*pair): {
                "scaled_integer_ceiling": str(homogeneous_bounds[pair]),
                "radius_lower_bound_ceiling": str(radius_bounds[pair]),
                "numeric_at_radius_lower": float(sp.N(radius_bounds[pair], 18)),
                "coordinate_multiindices_covered": comb(pair[1] + 2, 2),
            }
            for pair in _pairs(maximum_order)
        },
        "state_only_crosscheck": state_crosscheck,
        "claim": (
            "For every coordinate multiindex beta with a+|beta|<=4 and |xi|>=1, "
            "|xi|^|beta| ||D_U^a partial_xi^beta K55(U,xi/|xi|)|| is bounded "
            "by the emitted integer ceiling."
        ),
        "remaining_gate": (
            "low_frequency_extension_pseudodifferential_quantization_Sobolev_constants_"
            "energy_lifespan"
        ),
        "scope": (
            "This is a rigorous high-frequency homogeneous chart for xi != 0. It does not "
            "define a smooth low-frequency extension, choose a quantization, prove a "
            "Calderon-Vaillancourt estimate, close a Sobolev energy inequality, prove a "
            "lifespan, or include matter."
        ),
    }


def run_quartic_homogeneous_frequency_symbol_campaign(
    symbol_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticHomogeneousFrequencySymbolError(
                "unsupported campaign schema_version"
            )
        if symbol_campaign.get("status") != (
            "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes"
        ):
            raise QuarticHomogeneousFrequencySymbolError(
                "mixed symbol campaign prerequisite failed"
            )
        if not _content_hash_matches(symbol_campaign):
            raise QuarticHomogeneousFrequencySymbolError(
                "mixed symbol campaign content hash mismatch"
            )
        maximum_order = int(config["maximum_total_derivative_order"])
        if maximum_order != 4:
            raise QuarticHomogeneousFrequencySymbolError(
                "homogeneous frequency chart requires total order four"
            )
        spatial_dimension = int(config["spatial_dimension"])
        if spatial_dimension != 3:
            raise QuarticHomogeneousFrequencySymbolError(
                "the 55-state reduction requires three spatial dimensions"
            )
        frequency_radius_lower = sp.sympify(config["frequency_radius_lower"])
        if frequency_radius_lower < 1:
            raise QuarticHomogeneousFrequencySymbolError(
                "high-frequency radius lower bound must be at least one"
            )
        control_passed, control = generic_homogeneous_frequency_chain_rule_control()
        if not control_passed:
            raise QuarticHomogeneousFrequencySymbolError(
                "generic homogeneous frequency control failed"
            )
        normalization_bounds = normalization_map_frechet_majorants(maximum_order)
        records = _candidate_records(symbol_campaign)
        expected = int(config.get("expected_candidate_count", 12))
        if len(records) != expected:
            raise QuarticHomogeneousFrequencySymbolError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                records[candidate_id],
                maximum_order,
                frequency_radius_lower,
                normalization_bounds,
            )
            for candidate_id in sorted(records)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_full_K55_homogeneous_frequency_C4_bounds",
            "errors": [],
            "symbol_campaign_sha256": symbol_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_homogeneous_frequency_chain_rule_control": control,
            "counts": {
                "selected": len(certificates),
                "homogeneous_frequency_bounds_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have homogeneous xi-derivative bounds through "
                "total state/frequency order four for the actual lifted K55 symbol."
            ),
            "scope": certificates[0]["scope"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticHomogeneousFrequencySymbolError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "homogeneous_frequency_bounds_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_homogeneous_frequency_symbol_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
