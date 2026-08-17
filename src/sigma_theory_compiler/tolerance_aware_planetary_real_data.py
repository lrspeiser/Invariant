"""B8 on real measured data: published planetary elements, not re-anchored ones.

The blind planetary campaign could not hand B1 the published semi-major axes and sidereal
periods.  B1 solves exactly, and the published pairs are not exactly anything: the
campaign's own ``anchor_fidelity`` block records a residual of up to 6.7e-4 between the
published periods and the exact two-body power law.  So the campaign re-anchored the rows
-- it quantised the square root of each published axis, then *rebuilt* the axis and the
period from that root -- and the deviation went away by construction.  That re-anchoring
is the gate this module walks through.  It asks the same question of the numbers as
published, with the deviation left in, and lets the tolerance decide.

Four cohorts, all at coverage factor k = 1 over half-ulp intervals:

``published_kepler``
    The published ``(a, T)`` pairs.  Does ``T = a^(3/2)`` pass through every published
    interval at once?
``two_body_counterfactual_kepler``
    The campaign's own sealed generative rule -- ``a = s^2``, ``T = s^3`` -- rounded back
    to the *same number of published digits* as each real value.  This is the matched
    control: same bodies, same printed precision, same interval widths, only the physics
    differs.  Whatever the published cohort does, this cohort shows what it would have
    done in a strictly two-body solar system published to the same precision.
``published_inverse_square`` / ``two_body_counterfactual_inverse_square``
    The same two cohorts recast as ``g = 4*pi^2*a/T^2`` against ``r = a``, so the recovered
    exponent should be -2 instead of 3/2.

Sigma convention, stated because it is a choice.  Neither cited table publishes an
uncertainty.  What each does publish is a number of digits, and a correctly-rounded
decimal carries a hard bound: the true value lies within half a unit in the last published
place.  Every row therefore uses ``half_ulp_of_last_published_digit``, which
:mod:`.tolerance_aware_fitting` re-derives from the printed string rather than accepting
from this module.  That bound is a *rounding* bound, not a measurement error bar: it is
what the source's own precision entitles a reader to assume, and it is narrower than any
physical uncertainty would be.  A verdict of INFEASIBLE under it means the published
numbers disagree with the model by more than their own printed precision allows -- which
is a statement about the published numbers and the model together, and nothing more.

Boundary.  This run reads the already-public provenance block of the blind campaign's
target fixture to obtain the published anchors and the quantised roots; it participates in
no blinding protocol and asserts nothing about blinding.  It opens no observational
archive, performs no fit of its own outside :mod:`.tolerance_aware_fitting`, and claims no
novelty: Kepler's third law and the inverse-square law are the two most thoroughly settled
results in the subject, and rediscovering them is a test of the instrument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256
from .tolerance_aware_fitting import (
    MeasuredRow,
    ToleranceFittingError,
    build_ladder,
    certified_exponent_bracket,
    decide_power_law,
    decimal_string,
    fit_measured,
    forbidden_receipt_keys,
    parse_rows,
    write_immutable,
)

RESULT_SCHEMA = "invariant-tolerance-aware-planetary-real-data-result-1.0"
COHORT_SCHEMA = "invariant-tolerance-aware-cohort-receipt-1.0"
CAMPAIGN_ID = "tolerance-aware-planetary-real-data-001"

ANCHOR_FIXTURE_PATH = "configs/backgrounds/blind_planetary_targets_v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/tolerance_aware_planetary_real_data.py"
OUTPUT_PATH = "runs/math/tolerance-fitting/planetary-real-data-v1.json"
RECEIPT_DIRECTORY = "runs/math/tolerance-fitting"

COVERAGE_K = "1"
#: Outward decimal places used when a derived column's exact interval is re-expressed as a
#: centre and a half-width.  Rounding is outward, so the stored box always contains the
#: exact propagated interval.
DERIVED_COLUMN_PLACES = 30
EXPONENT_BRACKET_PLACES = 30
EXCLUSION_OFFSET = "1/100"

SIGMA_CONVENTION = (
    "half_ulp_of_last_published_digit: a value printed to d decimal places is treated as "
    "correctly rounded, so its true value lies within 5*10^-(d+1). Neither cited table "
    "publishes an uncertainty; the digit count is the precision the source does state. "
    "This is a rounding bound, not a physical error bar, and it is re-derived inside "
    "tolerance_aware_fitting from the printed string rather than supplied by this module."
)

INSTRUMENT_PASS_POLICY = (
    "Declared before the run and about the instrument, not about nature. PASS requires: "
    "(1) the two-body counterfactual Kepler cohort returns FEASIBLE_MINIMAL at exponent "
    "3/2 with a constant interval containing 1; (2) the two-body counterfactual "
    "inverse-square cohort returns FEASIBLE_MINIMAL at exponent -2; (3) both counterfactual "
    "cohorts certify their target exponent plus and minus 1/100 as INFEASIBLE, so the "
    "recovered exponent is pinned inside +/-0.01; (4) every published cohort reaches a "
    "decided verdict -- FEASIBLE_MINIMAL or INFEASIBLE_ALL_LADDER, never BLOCKED. An "
    "INFEASIBLE verdict on a published cohort is a finding about the published numbers, "
    "not a failure of the run, and does not affect this policy."
)

SCOPE = (
    "Tolerance-aware interval fitting applied to published planetary semi-major axes and "
    "sidereal periods as printed, alongside a matched two-body counterfactual built from "
    "the same sealed generative rule and rounded to the same published digits. Every "
    "verdict is about the declared half-ulp intervals at coverage factor 1 and the "
    "declared bounded exponent ladder; INFEASIBLE means no exponent in that ladder admits "
    "a single constant reaching every published interval, never that no relation exists. "
    "The published values are transcribed from the sealed anchor fixture's provenance "
    "block, which cites JPL Solar System Dynamics and NASA NSSDCA; this run opens no "
    "observational archive and re-derives no element. Kepler's third law and the "
    "inverse-square law are classical results and no novelty is asserted."
)

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "measured_values_used_as_published": True,
    "novelty_claimed": False,
    "observational_archive_opened": False,
    "rediscovery_of_classical_results": True,
    "re_anchored_values_used_in_the_published_cohorts": False,
    "sigma_derived_from_the_published_digit_convention": True,
    "uncertainties_adjusted_to_obtain_a_verdict": False,
}


class PlanetaryRealDataError(ValueError):
    """Raised on fixture drift, propagation failure, or receipt tamper."""


# ---------------------------------------------------------------------------
# Published anchors, read from the already-public sealed fixture
# ---------------------------------------------------------------------------


def _decimal_places(text: str) -> int:
    _, _, decimals = text.partition(".")
    return len(decimals)


def _exact_decimal(text: str) -> Fraction:
    whole, _, decimals = text.partition(".")
    sign = -1 if whole.startswith("-") else 1
    return Fraction(sign * int(whole.lstrip("-") + decimals), 10 ** len(decimals))


def load_anchors(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Published anchors, declared constants, and the fixture's byte identity."""

    path = (root / ANCHOR_FIXTURE_PATH).resolve()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PlanetaryRealDataError("anchor fixture unavailable") from error
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    fixture = json.loads(normalized.decode("utf-8"))
    provenance = fixture.get("provenance")
    if not isinstance(provenance, Mapping) or "anchors" not in provenance:
        raise PlanetaryRealDataError("anchor fixture provenance changed")
    anchors: list[dict[str, Any]] = []
    for anchor in provenance["anchors"]:
        required = {
            "body",
            "eccentricity",
            "quantized_root_of_semi_major_axis",
            "semi_major_axis_au",
            "sidereal_orbit_period_yr",
        }
        if set(anchor) != required:
            raise PlanetaryRealDataError("anchor record schema changed")
        anchors.append(dict(anchor))
    if len(anchors) < 4:
        raise PlanetaryRealDataError("anchor inventory shrank below a usable cohort")
    return anchors, dict(provenance["declared_constants"]), hashlib.sha256(normalized).hexdigest()


def _label(body: str) -> str:
    return body.lower().replace(" ", "_")


def published_kepler_rows(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``(a, T)`` exactly as printed in the cited tables."""

    return [
        {
            "label": _label(anchor["body"]),
            "point": anchor["semi_major_axis_au"],
            "point_sigma_rule": "half_ulp_of_last_published_digit",
            "source": (
                "semi-major axis: JPL Solar System Dynamics, Keplerian Elements for "
                "Approximate Positions of the Major Planets (Ceres: JPL Small-Body "
                "Database); sidereal period: NASA NSSDCA Planetary Fact Sheet; both as "
                f"transcribed in {ANCHOR_FIXTURE_PATH}"
            ),
            "value": anchor["sidereal_orbit_period_yr"],
            "value_sigma_rule": "half_ulp_of_last_published_digit",
        }
        for anchor in anchors
    ]


def counterfactual_pairs(anchors: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    """``a = s^2``, ``T = s^3`` rounded back to each body's own published digit count.

    The rule and the roots are the campaign's, unchanged; only the final rounding is added,
    so the control differs from the published cohort in physics alone and not in precision.
    """

    pairs: list[tuple[str, str, str]] = []
    for anchor in anchors:
        root = _exact_decimal(anchor["quantized_root_of_semi_major_axis"])
        pairs.append(
            (
                _label(anchor["body"]),
                decimal_string(root * root, _decimal_places(anchor["semi_major_axis_au"])),
                decimal_string(root**3, _decimal_places(anchor["sidereal_orbit_period_yr"])),
            )
        )
    return pairs


def counterfactual_kepler_rows(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "point": axis,
            "point_sigma_rule": "half_ulp_of_last_published_digit",
            "source": (
                "constructed control: the sealed generative rule a = s^2, T = s^3 of "
                f"{ANCHOR_FIXTURE_PATH}, rounded to the published digit count of the "
                "corresponding real value; not a measurement"
            ),
            "value": period,
            "value_sigma_rule": "half_ulp_of_last_published_digit",
        }
        for label, axis, period in counterfactual_pairs(anchors)
    ]


# ---------------------------------------------------------------------------
# The derived inverse-square column, propagated outward
# ---------------------------------------------------------------------------


def _four_pi_squared_interval(constants: Mapping[str, Any]) -> tuple[Fraction, Fraction]:
    text = constants["four_pi_squared_50dp"]
    places = _decimal_places(text)
    if places != 50:
        raise PlanetaryRealDataError("declared circle constant precision changed")
    value = _exact_decimal(text)
    half_ulp = Fraction(5, 10 ** (places + 1))
    return value - half_ulp, value + half_ulp


def inverse_square_rows(
    pairs: Sequence[tuple[str, str, str]],
    constants: Mapping[str, Any],
    *,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ``(r, g)`` rows with ``g = 4*pi^2*a/T^2`` as an outward-rounded box.

    The exact range of ``g`` over the declared ``(a, T)`` box is computed first -- every
    factor is monotone, so the extremes sit at the corners -- and only then re-expressed as
    a centre and a half-width, rounding *outward* at both ends.  Containment of the exact
    range inside the stored box is checked here and recorded, so the widening can never
    quietly shrink.
    """

    constant_low, constant_high = _four_pi_squared_interval(constants)
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    scale = 10**DERIVED_COLUMN_PLACES
    for label, axis_text, period_text in pairs:
        axis = _exact_decimal(axis_text)
        period = _exact_decimal(period_text)
        axis_pad = Fraction(5, 10 ** (_decimal_places(axis_text) + 1))
        period_pad = Fraction(5, 10 ** (_decimal_places(period_text) + 1))
        axis_low, axis_high = axis - axis_pad, axis + axis_pad
        period_low, period_high = period - period_pad, period + period_pad
        if axis_low <= 0 or period_low <= 0:
            raise PlanetaryRealDataError(f"declared interval is not positive: {label}")
        exact_low = constant_low * axis_low / (period_high * period_high)
        exact_high = constant_high * axis_high / (period_low * period_low)
        outward_low = Fraction((exact_low * scale).numerator // (exact_low * scale).denominator, scale)
        upper_scaled = exact_high * scale
        outward_high = Fraction(-((-upper_scaled.numerator) // upper_scaled.denominator), scale)
        if outward_low > exact_low or outward_high < exact_high:
            raise PlanetaryRealDataError(f"outward rounding lost the exact range: {label}")
        centre = (outward_low + outward_high) / 2
        half_width = (outward_high - outward_low) / 2
        if half_width <= 0:
            raise PlanetaryRealDataError(f"derived column half-width vanished: {label}")
        rows.append(
            {
                "label": label,
                "point": axis_text,
                "point_sigma_rule": "half_ulp_of_last_published_digit",
                "source": source,
                "value": {"numerator": centre.numerator, "denominator": centre.denominator},
                "value_sigma": {
                    "numerator": half_width.numerator,
                    "denominator": half_width.denominator,
                },
                "value_sigma_citation": (
                    "propagated outward from the declared half-ulp intervals of a and T "
                    "through g = 4*pi^2*a/T^2, with 4*pi^2 enclosed from its declared "
                    "50-decimal value plus and minus half a unit in the last place; every "
                    "factor is monotone so the exact range sits at the box corners, and the "
                    f"result is rounded outward to {DERIVED_COLUMN_PLACES} decimal places"
                ),
                "value_sigma_rule": "propagated_outward",
            }
        )
        audit.append(
            {
                "exact_range_contained": True,
                "row": label,
                "stored_half_width": decimal_string(half_width, DERIVED_COLUMN_PLACES),
                "stored_centre": decimal_string(centre, DERIVED_COLUMN_PLACES),
            }
        )
    return rows, audit


# ---------------------------------------------------------------------------
# Cohort execution
# ---------------------------------------------------------------------------


def _constant_interval(result: Mapping[str, Any]) -> dict[str, Any] | None:
    accepted = result.get("result")
    if not accepted:
        return None
    interval = accepted["coefficient_intervals"][0]
    lower = Fraction(interval["lower"]["numerator"], interval["lower"]["denominator"])
    upper = Fraction(interval["upper"]["numerator"], interval["upper"]["denominator"])
    return {
        "exact": {"lower": interval["lower"], "upper": interval["upper"]},
        "lower_decimal": decimal_string(lower, 18),
        "term": interval["term"],
        "upper_decimal": decimal_string(upper, 18),
    }


def _witness_summary(result: Mapping[str, Any], entry_id: str) -> dict[str, Any] | None:
    for entry in result["minimality_certificate"]["strictly_simpler_entries_rejected"]:
        if entry["entry_id"] != entry_id:
            continue
        witness = entry.get("witness")
        if not witness:
            return {"reason": entry.get("reason"), "verdict": entry["verdict"]}
        low = witness["requires_at_least"]
        high = witness["requires_at_most"]
        return {
            "entry_id": entry_id,
            "kind": witness["kind"],
            "reading": witness["reading"],
            "requires_at_least": {
                "bound_decimal": decimal_string(
                    Fraction(low["bound"]["numerator"], low["bound"]["denominator"]), 18
                ),
                "row": low["row"],
            },
            "requires_at_most": {
                "bound_decimal": decimal_string(
                    Fraction(high["bound"]["numerator"], high["bound"]["denominator"]), 18
                ),
                "row": high["row"],
            },
            "verdict": entry["verdict"],
        }
    return None


def run_cohort(
    cohort_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    target_exponent: str,
    question: str,
    boundary: str,
) -> dict[str, Any]:
    """Fit one cohort, probe the +/-0.01 competitors, and bracket the exponent."""

    ladder = build_ladder(include_linear=False)
    offset = Fraction(EXCLUSION_OFFSET)
    target = Fraction(target_exponent)
    probes = (str(target - offset), str(target + offset))
    result = fit_measured(
        list(rows), coverage_k=COVERAGE_K, ladder=ladder, exponent_probes=probes
    )
    parsed = parse_rows(list(rows))
    bracket = certified_exponent_bracket(
        parsed,
        coverage_k=COVERAGE_K,
        centre=target_exponent,
        outer_offset=EXCLUSION_OFFSET,
        places=EXPONENT_BRACKET_PLACES,
    )
    accepted = result.get("result")
    body = {
        "boundary": boundary,
        "certified_exponent_bracket": bracket,
        "cohort_id": cohort_id,
        "constant_interval": _constant_interval(result),
        "decision": result["decision"],
        "exclusion_probes": result["exponent_probes"],
        "fit_receipt": result,
        "question": question,
        "recovered_exponent": None if not accepted else accepted["exponent"],
        "rows": len(rows),
        "schema_version": COHORT_SCHEMA,
        "simpler_exponents_rejected": result["counts"]["entries_rejected_before_acceptance"],
        "target_exponent": target_exponent,
        "target_exponent_witness": _witness_summary(result, f"power_law:{target}"),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Diagnostics: how large is the residual the tolerance is refusing to absorb?
# ---------------------------------------------------------------------------


COVERAGE_BRACKET_PLACES = 6


def coverage_factor_bracket(
    rows: Sequence[MeasuredRow], exponent: Fraction, *, iterations: int = 60
) -> dict[str, Any]:
    """Smallest interval widening at which a single constant becomes admissible.

    This is a *diagnostic*, never a decision.  It reports how many declared half-widths of
    disagreement sit between the rows, in the only currency the receipt has -- the declared
    uncertainty itself.  No verdict anywhere in this campaign consumes it, and widening the
    coverage factor to obtain a verdict is exactly what the sigma guard exists to prevent.
    """

    ceiling = min(
        min(
            row.point / row.point_sigma if row.point_sigma else Fraction(10**9),
            row.value / row.value_sigma if row.value_sigma else Fraction(10**9),
        )
        for row in rows
    )
    low = Fraction(1)
    high = Fraction(1)
    while high < ceiling and decide_power_law(rows, exponent, high)["verdict"] != "FEASIBLE":
        low, high = high, high * 2
    if high >= ceiling:
        return {
            "bracketed": False,
            "is_a_decision": False,
            "positivity_ceiling_k": decimal_string(ceiling, COVERAGE_BRACKET_PLACES),
            "reason": "declared intervals stop being positive before any constant admits them",
        }
    # Bisect on values that are exactly representable at the reporting precision, so the
    # two endpoints printed below are the endpoints that were actually decided; rounding a
    # bisection bound afterwards can carry it across the threshold it is meant to bracket.
    step = Fraction(1, 10**COVERAGE_BRACKET_PLACES)
    for _ in range(iterations):
        if high - low <= step:
            break
        middle = Fraction(decimal_string((low + high) / 2, COVERAGE_BRACKET_PLACES))
        if middle in (low, high):
            break
        if decide_power_law(rows, exponent, middle)["verdict"] == "FEASIBLE":
            high = middle
        else:
            low = middle
    return {
        "bracketed": True,
        "certified_feasible_at_k": decimal_string(high, COVERAGE_BRACKET_PLACES),
        "certified_infeasible_at_k": decimal_string(low, COVERAGE_BRACKET_PLACES),
        "is_a_decision": False,
        "positivity_ceiling_k": decimal_string(ceiling, COVERAGE_BRACKET_PLACES),
        "reading": (
            "every declared interval must be widened by this factor before one constant "
            "reaches all of them; the campaign decides at k = 1 and never consumes this "
            "number"
        ),
    }


def multi_body_residual(
    anchors: Sequence[Mapping[str, Any]], declared_fidelity: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-body deviation of the published period from the sealed two-body rule."""

    bodies: list[dict[str, Any]] = []
    worst = Fraction(0)
    worst_body = ""
    for anchor in anchors:
        root = _exact_decimal(anchor["quantized_root_of_semi_major_axis"])
        published = _exact_decimal(anchor["sidereal_orbit_period_yr"])
        two_body = root**3
        deviation = abs(published - two_body) / published
        half_ulp = Fraction(5, 10 ** (_decimal_places(anchor["sidereal_orbit_period_yr"]) + 1))
        if deviation > worst:
            worst, worst_body = deviation, _label(anchor["body"])
        bodies.append(
            {
                "deviation_in_units_of_the_declared_half_width": decimal_string(
                    abs(published - two_body) / half_ulp, 3
                ),
                "published_period_yr": anchor["sidereal_orbit_period_yr"],
                "relative_deviation_from_the_two_body_rule": decimal_string(deviation, 20),
                "row": _label(anchor["body"]),
                "two_body_period_yr": decimal_string(two_body, 12),
            }
        )
    declared = declared_fidelity["max_relative_deviation_sidereal_period"]
    recomputed = decimal_string(worst, _decimal_places(declared))
    return {
        "bodies": bodies,
        "note": declared_fidelity["note"],
        "reproduces_the_sealed_campaign_fidelity_figure": recomputed == declared,
        "sealed_campaign_declared_max_relative_deviation": declared,
        "this_run_recomputed_max_relative_deviation": recomputed,
        "worst_body": worst_body,
    }


# ---------------------------------------------------------------------------
# Campaign assembly
# ---------------------------------------------------------------------------


def build_campaign(root: Path) -> dict[str, Any]:
    """Run all four cohorts and seal the campaign receipt."""

    root = root.resolve()
    anchors, constants, fixture_sha256 = load_anchors(root)
    fixture = json.loads(
        (root / ANCHOR_FIXTURE_PATH)
        .read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .decode("utf-8")
    )
    declared_fidelity = fixture["provenance"]["fidelity"]

    published_rows = published_kepler_rows(anchors)
    counterfactual_rows = counterfactual_kepler_rows(anchors)
    published_pairs = [
        (row["label"], row["point"], row["value"]) for row in published_rows
    ]
    published_inverse, published_audit = inverse_square_rows(
        published_pairs,
        constants,
        source=(
            "derived column: g = 4*pi^2*a/T^2 from the published a and T of the cited "
            f"tables as transcribed in {ANCHOR_FIXTURE_PATH}"
        ),
    )
    counterfactual_inverse, counterfactual_audit = inverse_square_rows(
        counterfactual_pairs(anchors),
        constants,
        source=(
            "derived column: g = 4*pi^2*a/T^2 from the constructed two-body control "
            "values; not a measurement"
        ),
    )

    correlated_note = (
        "the g column is derived from the same published a and T that form the point "
        "column, so its box treats correlated rounding as independent. That widens the "
        "admissible region, which makes an INFEASIBLE verdict here rigorous and a FEASIBLE "
        "verdict weaker than it looks; this cohort is a re-parameterisation of the Kepler "
        "cohort and is not independent evidence"
    )
    cohorts = [
        run_cohort(
            "published_kepler",
            published_rows,
            target_exponent="3/2",
            question="does T = a^(3/2) pass through every published interval at once?",
            boundary=(
                "values exactly as printed in the cited tables; no re-anchoring, no "
                "refitting, no adjustment of the declared uncertainties"
            ),
        ),
        run_cohort(
            "two_body_counterfactual_kepler",
            counterfactual_rows,
            target_exponent="3/2",
            question=(
                "what would the published cohort have done in a strictly two-body solar "
                "system printed to the same precision?"
            ),
            boundary=(
                "constructed control, not a measurement: the campaign's own sealed rule "
                "rounded to the published digit count of each real value"
            ),
        ),
        run_cohort(
            "published_inverse_square",
            published_inverse,
            target_exponent="-2",
            question="does g = C*r^(-2) pass through every propagated interval at once?",
            boundary=correlated_note,
        ),
        run_cohort(
            "two_body_counterfactual_inverse_square",
            counterfactual_inverse,
            target_exponent="-2",
            question="the same question on the matched two-body control",
            boundary=f"constructed control, not a measurement; {correlated_note}",
        ),
    ]

    published_parsed = parse_rows(published_rows)
    witness = cohorts[0]["target_exponent_witness"] or {}
    witness_labels = {
        (witness.get("requires_at_least") or {}).get("row"),
        (witness.get("requires_at_most") or {}).get("row"),
    }
    witness_rows = [row for row in published_parsed if row.label in witness_labels]
    coverage_diagnostic = {
        "all_published_rows": coverage_factor_bracket(published_parsed, Fraction(3, 2)),
        "policy": (
            "diagnostic only. The campaign decides at k = 1 and no verdict consumes these "
            "numbers; they measure the size of the disagreement in units of the declared "
            "uncertainty"
        ),
        "witness_pair": {
            **coverage_factor_bracket(witness_rows, Fraction(3, 2)),
            "rows": sorted(row.label for row in witness_rows),
        }
        if len(witness_rows) == 2
        else None,
    }

    decided = {"FEASIBLE_MINIMAL", "INFEASIBLE_ALL_LADDER"}
    by_id = {cohort["cohort_id"]: cohort for cohort in cohorts}
    kepler_control = by_id["two_body_counterfactual_kepler"]
    inverse_control = by_id["two_body_counterfactual_inverse_square"]

    def _excluded(cohort: Mapping[str, Any]) -> bool:
        return all(probe["verdict"] == "INFEASIBLE" for probe in cohort["exclusion_probes"])

    def _contains_one(cohort: Mapping[str, Any]) -> bool:
        interval = cohort["constant_interval"]
        if interval is None:
            return False
        lower = Fraction(interval["exact"]["lower"]["numerator"], interval["exact"]["lower"]["denominator"])
        upper = Fraction(interval["exact"]["upper"]["numerator"], interval["exact"]["upper"]["denominator"])
        return lower <= 1 <= upper

    checks = {
        "counterfactual_inverse_square_recovers_minus_two": (
            inverse_control["decision"] == "FEASIBLE_MINIMAL"
            and inverse_control["recovered_exponent"] == "-2"
        ),
        "counterfactual_kepler_constant_interval_contains_one": _contains_one(kepler_control),
        "counterfactual_kepler_recovers_three_halves": (
            kepler_control["decision"] == "FEASIBLE_MINIMAL"
            and kepler_control["recovered_exponent"] == "3/2"
        ),
        "every_cohort_reached_a_decided_verdict": all(
            cohort["decision"] in decided for cohort in cohorts
        ),
        "target_exponent_pinned_within_one_hundredth_on_both_controls": (
            _excluded(kepler_control) and _excluded(inverse_control)
        ),
    }
    decision = "PASS" if all(checks.values()) else "BLOCK"

    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "checks": checks,
        "claims": CLAIMS,
        "cohort_summaries": [
            {
                "cohort_id": cohort["cohort_id"],
                "constant_interval_decimal": (
                    None
                    if cohort["constant_interval"] is None
                    else [
                        cohort["constant_interval"]["lower_decimal"],
                        cohort["constant_interval"]["upper_decimal"],
                    ]
                ),
                "decision": cohort["decision"],
                "exclusion_probe_verdicts": {
                    probe["exponent"]: probe["verdict"] for probe in cohort["exclusion_probes"]
                },
                "exponent_bracket": (
                    {
                        side: {
                            key: data[key]
                            for key in ("certified_feasible_at", "certified_infeasible_at")
                            if key in data
                        }
                        for side, data in cohort["certified_exponent_bracket"]["sides"].items()
                    }
                ),
                "recovered_exponent": cohort["recovered_exponent"],
                "rows": cohort["rows"],
                "simpler_exponents_rejected": cohort["simpler_exponents_rejected"],
                "target_exponent": cohort["target_exponent"],
                "target_exponent_verdict_at_centre": cohort["certified_exponent_bracket"][
                    "centre_verdict"
                ],
                "witness": cohort["target_exponent_witness"],
            }
            for cohort in cohorts
        ],
        "coverage_factor": COVERAGE_K,
        "coverage_factor_diagnostic": coverage_diagnostic,
        "decision": decision,
        "derived_column_audit": {
            "published_inverse_square": published_audit,
            "two_body_counterfactual_inverse_square": counterfactual_audit,
        },
        "first_blocker": None
        if decision == "PASS"
        else "declared instrument policy not satisfied",
        "instrument_pass_policy": INSTRUMENT_PASS_POLICY,
        "multi_body_residual": multi_body_residual(anchors, declared_fidelity),
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "sigma_convention": SIGMA_CONVENTION,
        "source_bindings": {
            "anchor_fixture": {"file_sha256": fixture_sha256, "path": ANCHOR_FIXTURE_PATH},
        },
    }
    campaign = {**body, "content_sha256": canonical_sha256(body)}
    return {"campaign": campaign, "cohorts": {c["cohort_id"]: c for c in cohorts}}


def validate_artifacts(
    campaign: Mapping[str, Any], cohorts: Mapping[str, Mapping[str, Any]], *, root: Path
) -> None:
    """Reject tamper or drift by exact deterministic replay of the whole campaign."""

    if campaign.get("schema_version") != RESULT_SCHEMA:
        raise PlanetaryRealDataError("campaign receipt schema changed")
    body = {key: item for key, item in campaign.items() if key != "content_sha256"}
    if campaign.get("content_sha256") != canonical_sha256(body):
        raise PlanetaryRealDataError("campaign receipt seal changed")
    if campaign.get("claims") != CLAIMS:
        raise PlanetaryRealDataError("claims changed")
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise PlanetaryRealDataError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    for cohort in cohorts.values():
        cohort_body = {key: item for key, item in cohort.items() if key != "content_sha256"}
        if cohort.get("content_sha256") != canonical_sha256(cohort_body):
            raise PlanetaryRealDataError("cohort receipt seal changed")
    replayed = build_campaign(root)
    if dict(campaign) != replayed["campaign"]:
        raise PlanetaryRealDataError("campaign receipt exact replay changed")
    if {key: dict(value) for key, value in cohorts.items()} != replayed["cohorts"]:
        raise PlanetaryRealDataError("cohort receipt exact replay changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if args.validate_checked:
        campaign = json.loads(output.read_text(encoding="utf-8"))
        cohorts = {
            summary["cohort_id"]: json.loads(
                (root / f"{RECEIPT_DIRECTORY}/cohort-{summary['cohort_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            for summary in campaign["cohort_summaries"]
        }
        validate_artifacts(campaign, cohorts, root=root)
        print(json.dumps({"validated": True, "decision": campaign["decision"]}, indent=2))
        return 0
    artifacts = build_campaign(root)
    write_immutable(output, artifacts["campaign"])
    for cohort_id, receipt in artifacts["cohorts"].items():
        write_immutable((root / f"{RECEIPT_DIRECTORY}/cohort-{cohort_id}.json").resolve(), receipt)
    validate_artifacts(artifacts["campaign"], artifacts["cohorts"], root=root)
    print(
        json.dumps(
            {
                "cohorts": [
                    {
                        "cohort_id": summary["cohort_id"],
                        "decision": summary["decision"],
                        "recovered_exponent": summary["recovered_exponent"],
                    }
                    for summary in artifacts["campaign"]["cohort_summaries"]
                ],
                "decision": artifacts["campaign"]["decision"],
            },
            indent=2,
        )
    )
    return 0 if artifacts["campaign"]["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIMS",
    "OUTPUT_PATH",
    "RESULT_SCHEMA",
    "PlanetaryRealDataError",
    "ToleranceFittingError",
    "build_campaign",
    "counterfactual_kepler_rows",
    "coverage_factor_bracket",
    "inverse_square_rows",
    "load_anchors",
    "multi_body_residual",
    "published_kepler_rows",
    "run_cohort",
    "validate_artifacts",
]
