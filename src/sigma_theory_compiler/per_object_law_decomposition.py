"""Tier 7 R1 and R2: fit every galaxy on its own, then diagnose the parameter.

``real_data_gravity_confrontation`` pools 214 published SPARC points from six galaxies into
one linear system whose width is the number of *universal* constants, and returns a single
word.  That word is INFEASIBLE for Newtonian baryons alone and INFEASIBLE for all twelve
surviving screened-gravity families, which is two very different failures reported
identically: a family whose one free amplitude wants a different value in every galaxy and
a family whose amplitude is the same everywhere but simply too small are indistinguishable
once they have been pooled.

This module separates them.

**R1 -- per-object decomposition.**  Each galaxy is fitted *alone*, over the same declared
law space, with its own copy of the free parameter, and the whole population of local
solutions is kept.  Nothing is ranked and nothing is discarded: the population *is* the
result.

**R2 -- parameter-variation diagnosis.**  For each parameter and each object an exact
rational interval is emitted, and the question asked of the population is the only one that
matters: does a single value lie inside every interval?  If it does, the universal law is
already in hand and the pooled INFEASIBLE was about the law's *shape*, not about its
constancy.  If it does not, the variation is real, it is bounded below by an exhibited pair
of disjoint intervals, and that bound is a measurement.

The arithmetic that makes both exact
-----------------------------------

With one free parameter the model on row ``i`` is ``c_i + theta * b_i`` and the declared
constraint is ``v_i - k sigma_i <= c_i + theta b_i <= v_i + k sigma_i``.  Writing
``d_i = v_i - c_i``, ``alpha_i = d_i / b_i`` and ``beta_i = sigma_i / |b_i|`` (both exact
rationals), that constraint is exactly ``alpha_i - beta_i k <= theta <= alpha_i + beta_i k``.
So the feasible set at coverage ``k`` is the interval

    [ max_i (alpha_i - beta_i k) , min_j (alpha_j + beta_j k) ]

whose left endpoint falls and whose right endpoint rises, both affinely in ``k``.  It is
non-empty exactly when ``k >= (alpha_i - alpha_j) / (beta_i + beta_j)`` for every ordered
pair, so the **smallest coverage factor at which this object admits any solution at all** is

    k*(object) = max over ordered pairs (i, j) of (alpha_i - alpha_j) / (beta_i + beta_j)

which is an exact rational, is attained at a named pair of published rows, and is the entire
per-object result: at ``k*`` the interval collapses onto one point, and below ``k*`` those
two rows contradict each other.  Nothing here is iterated, tolerated, or optimised
numerically -- it is a maximum over a finite set of rationals.

Two derived quantities carry the diagnosis:

``k_pop = max over objects of k*(object)``
    the smallest coverage at which the *population* exists -- every object individually
    feasible, each with its own parameter.

``k_common``
    the same maximum taken over ordered pairs drawn from *all* objects at once, which is
    the smallest coverage at which **one shared value** fits every object.  Because the
    pooled pair set contains every within-object pair, ``k_common >= k_pop`` always, and
    equality holds exactly when the binding pair lies inside a single object.

Their ratio is the **price of universality**.  It is 1 when the per-object parameter is
already constant within the declared uncertainties, and larger than 1 by exactly the factor
by which the intervals must be widened before one value fits everything.  It is *not* a
goodness measure and it does not rank laws: a badly wrong law whose parameter is uselessly
unconstrained also prices universality at 1, and the deliberately-wrong-law control in this
receipt demonstrates precisely that.  ``k_pop`` says how well any object can be fitted;
the price says whether the objects agree.  Both are needed and neither substitutes.

The discipline, stated before the result
----------------------------------------

Per-object knobs are exactly what the sealed trial forbids, and a population of per-object
fits can absorb almost anything.  So this receipt is *hypothesis generation only*:

* the exploration/confirmation split is computed from a declared salted hash of the galaxy
  *names* -- never from their data -- and is written into the receipt **before** any fit
  runs, with its own digest;
* every fitting entry point takes the split and raises :class:`ConfirmationSetTouched` if a
  confirmation-set galaxy reaches it, so "we did not look" is enforced by the call graph
  rather than promised in prose;
* the receipt declares ``trial_type: exploratory`` and refuses to be cited as confirmation.

One honesty note that belongs in the module and not only in the receipt: the confirmation
set here is *not* virgin data.  The predecessor receipt
``runs/gpu-baryonic-screen/real-data-exploratory-v1.json`` already scanned its declared grid
against all six galaxies.  The split therefore protects the Tier 7 *reconciliation* claim,
which is what R4 and R5 will consume, and it does not restore untouched-data status to the
two withheld galaxies.  That is recorded in the receipt as a limitation, not a footnote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .real_data_gravity_confrontation import (
    A0_GRID,
    COVERAGE_GRID,
    LENGTH_UNIT_GRID,
    QUADRATURE,
    REFERENCE_GRID_POINT,
    ColumnCache,
    Family,
    Galaxy,
    _family_columns,
    load_families,
    load_galaxies,
    measured_rows,
    prepare_galaxy,
    select_best_family,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .tolerance_aware_fitting import (
    FEASIBLE,
    INFEASIBLE,
    LinearSystem,
    MeasuredRow,
    decide_system,
    forbidden_receipt_keys,
)

RESULT_SCHEMA = "invariant-per-object-law-decomposition-result-1.0"
RECEIPT_PATH = "runs/gpu-baryonic-screen/per-object-decomposition-v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/per_object_law_decomposition.py"
TEST_PATH = "tests/test_per_object_law_decomposition.py"

#: This receipt fits.  It may never be cited as a confirmation of anything.
TRIAL_TYPE = "exploratory"

# ---------------------------------------------------------------------------
# The split.  Declared here, in the source, before any datum is read.
# ---------------------------------------------------------------------------

#: The salt that fixes the exploration/confirmation split.  It is a constant of this
#: module, not an argument, so the split cannot be re-rolled until it looks convenient.
SPLIT_SALT = "invariant-tier7-per-object-decomposition-v1"

#: How many of the declared galaxies are withheld.  Two of six.
CONFIRMATION_COUNT = 2

SPLIT_RULE = (
    "sort the declared galaxy names by sha256(SPLIT_SALT + '|' + name) and withhold the "
    "first CONFIRMATION_COUNT of them. The rule reads galaxy *names* only -- never a "
    "radius, a velocity, or an uncertainty -- so the split cannot be steered by the data "
    "it is protecting, and it is recomputable from this source file alone"
)

# ---------------------------------------------------------------------------
# Declared rules, all fixed before the run
# ---------------------------------------------------------------------------

#: A residual pattern counts as structured when the sign changes at most once per this many
#: consecutive points.  Declared before the fit; it is an integer test on integer counts and
#: no probability is claimed anywhere.
STRUCTURE_RUN_DIVISOR = 4

STRUCTURE_RULE = (
    "order each galaxy's residuals by published radius and count sign changes. The pattern "
    f"is called structured when (sign changes) * {STRUCTURE_RUN_DIVISOR} <= (points - 1), "
    "that is, when the residual keeps its sign for long stretches instead of alternating. "
    "This is an integer comparison on integer counts: no distribution is assumed and no "
    "probability is claimed"
)

#: The control that must come out worse.  Declared before the run so it cannot be chosen
#: after seeing which way the comparison fell.
WRONG_LAW_EXPECTATION = (
    "the deliberately wrong law -- the same family with its local factor inverted, so the "
    "modification grows where gravity is strong -- must need a strictly larger k_pop than "
    "every screened family on the same exploration galaxies. If it ever needs less, the "
    "per-object machinery is measuring nothing and this receipt is void"
)

#: Two free arm weights, per object, for the declared best family.  Reported as coordinate
#: projections of the exact feasible polytope.
TWO_PARAMETER_CAVEAT = (
    "these are coordinate projections of each object's exact feasible polytope. A single "
    "value lying in every object's interval for w_yukawa AND a single value lying in every "
    "object's interval for w_power does NOT imply that one shared pair (w_yukawa, w_power) "
    "fits every object: the projections of a set do not determine the set. R2 asks the "
    "per-parameter question and is answered per parameter; the joint question is decided "
    "separately below by an exact pooled linear program, and both answers are published"
)

CLAIMS = {
    "confirmation_set_fitted": False,
    "every_fit_is_single_object": True,
    "exact_rational_certificates": True,
    "population_kept_not_ranked": True,
    "split_sealed_before_fitting": True,
}

SCOPE = (
    "Tier 7 R1 and R2 on the declared SPARC subset. R1 fits each exploration galaxy alone "
    "with its own copy of one free parameter over each declared law space and keeps every "
    "local solution. R2 emits an exact interval per parameter per object and decides "
    "whether one value lies in all of them. R3 (channel discovery), R4 (reconciliation "
    "search) and R5 (held-out confirmation) are out of scope here and no channel is named, "
    "measured, or hinted at anywhere in this module -- naming one before it is measured is "
    "the R3 falsifier."
)

ASSUMPTIONS = {
    "columns_are_frozen_floats": (
        "the design entries c_i and b_i are produced by the predecessor module's declared "
        "quadrature in float64 and then frozen exactly onto a 15-significant-digit decimal "
        "grid. Every step from that freeze onward -- alpha, beta, k*, every interval, every "
        "certificate -- is exact rational arithmetic. The freeze is the boundary and it is "
        "declared, not hidden"
    ),
    "coverage_factor_is_a_tolerance_not_a_probability": (
        "k multiplies the published e_Vobs column propagated outward into v^2. It is a "
        "declared widening of a declared interval. No sampling distribution is assumed, so "
        "k is not a number of standard deviations in any inferential sense"
    ),
    "published_sigmas_are_random_errors_only": (
        "the SPARC e_Vobs column is the published random error from non-circular motions "
        "and kinematic asymmetries. It excludes inclination and distance systematics, so "
        "the k values reported here are large for every law tried, including the ones that "
        "are conventionally regarded as successful. That is a property of the declared "
        "uncertainty budget and it applies identically to every law compared here"
    ),
    "the_parameter_is_the_same_slot_in_every_law_space": (
        "each law space offers exactly one free scalar per object, multiplying the part of "
        "the model that the law adds to the published baryons. theta = 1 always means 'the "
        "law exactly as declared', so the populations are directly comparable"
    ),
}


class PerObjectError(ValueError):
    """Raised on malformed input, a guard violation, or a failed self-check."""


class ConfirmationSetTouched(PerObjectError):
    """Raised when a confirmation-set object reaches a fitting entry point."""


# ---------------------------------------------------------------------------
# Numeric helpers.  Decimals are for reading; every decision is on a Fraction.
# ---------------------------------------------------------------------------

EMITTED_DIGITS = 9


def _num(value: Fraction) -> str:
    """Render an exact rational for reading.  Never consumed by a decision."""

    try:
        return f"{float(value):.{EMITTED_DIGITS - 1}e}"
    except OverflowError:  # pragma: no cover - defensive; values here are O(100)
        return "out_of_float_range"


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def _fraction_block(value: Fraction) -> dict[str, Any]:
    return {"decimal": _num(value), "exact": _fraction_data(value)}


def _from_block(block: Mapping[str, Any]) -> Fraction:
    exact = block["exact"]
    return Fraction(int(exact["numerator"]), int(exact["denominator"]))


# ---------------------------------------------------------------------------
# Step 0 -- the split, sealed before anything is fitted
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Split:
    """Which objects may be fitted, and which may not be touched.

    ``count``, ``salt`` and ``rule`` default to this module's declared constants, so the
    six-galaxy receipt is unchanged.  They are fields rather than globals because a wider
    population -- the full published sample -- must declare its own withholding fraction
    and its own salt, and must do so in *its* source rather than by editing this one.
    """

    exploration: tuple[str, ...]
    confirmation: tuple[str, ...]
    digests: tuple[tuple[str, str], ...]
    count: int = CONFIRMATION_COUNT
    salt: str = SPLIT_SALT
    rule: str = SPLIT_RULE

    def block(self) -> dict[str, Any]:
        """The receipt block.  Written before pass 1 runs, and hashed on its own."""

        body = {
            "confirmation": list(self.confirmation),
            "confirmation_count": self.count,
            "exploration": list(self.exploration),
            "name_digests": {name: digest for name, digest in self.digests},
            "rule": self.rule,
            "salt": self.salt,
            "sealed_before_any_fit": True,
        }
        return {**body, "split_sha256": canonical_sha256(body)}

    def guard(self, galaxies: Sequence[Galaxy]) -> None:
        """Refuse, structurally, to fit anything the split withheld."""

        allowed = set(self.exploration)
        for galaxy in galaxies:
            if galaxy.name in self.confirmation:
                raise ConfirmationSetTouched(
                    f"{galaxy.name} is in the confirmation set and may not be fitted"
                )
            if galaxy.name not in allowed:
                raise ConfirmationSetTouched(
                    f"{galaxy.name} is not in the declared exploration set"
                )


def declare_split(
    names: Sequence[str],
    *,
    count: int | None = None,
    salt: str | None = None,
    rule: str | None = None,
) -> Split:
    """Compute the split from the declared salt and the object *names* alone.

    Omitting an argument reproduces the six-galaxy split byte for byte.  A caller fitting a
    wider population passes its own declared ``count``, ``salt`` and ``rule``; the
    arithmetic is identical and still reads names only, never a radius, a velocity or an
    uncertainty.  The defaults are resolved here rather than in the signature so that the
    declared constants are read at call time -- a control that rebinds ``SPLIT_SALT`` to
    check the partition really moves with it must keep working.
    """

    count = CONFIRMATION_COUNT if count is None else count
    salt = SPLIT_SALT if salt is None else salt
    rule = SPLIT_RULE if rule is None else rule
    unique = sorted(set(names))
    if len(unique) != len(names):
        raise PerObjectError("duplicate object name in the declared population")
    if count < 1:
        raise PerObjectError("the split must withhold at least one object")
    if len(unique) <= count:
        raise PerObjectError("the split would leave nothing to explore")
    digested = sorted(
        (hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(), name) for name in unique
    )
    confirmation = tuple(name for _, name in digested[:count])
    exploration = tuple(name for _, name in digested[count:])
    if set(confirmation) & set(exploration):
        raise PerObjectError("the split is not a partition")
    return Split(
        confirmation=tuple(sorted(confirmation)),
        exploration=tuple(sorted(exploration)),
        digests=tuple(sorted((name, digest) for digest, name in digested)),
        count=count,
        salt=salt,
        rule=rule,
    )


# ---------------------------------------------------------------------------
# Step 1 -- one object, one free parameter, exact
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Axis:
    """One object's rows reduced to the theta axis: theta in [alpha - beta k, alpha + beta k]."""

    alphas: tuple[Fraction, ...]
    betas: tuple[Fraction, ...]
    labels: tuple[str, ...]
    #: Rows whose free column vanished: they constrain k directly and theta not at all.
    blind_floor: Fraction
    blind_label: str | None


def build_axis(
    offsets: Sequence[Fraction],
    slopes: Sequence[Fraction],
    rows: Sequence[MeasuredRow],
) -> Axis:
    """Reduce ``c_i + theta b_i in [v_i - k s_i, v_i + k s_i]`` to an interval in theta."""

    alphas: list[Fraction] = []
    betas: list[Fraction] = []
    labels: list[str] = []
    floor = Fraction(0)
    floor_label: str | None = None
    for offset, slope, row in zip(offsets, slopes, rows, strict=True):
        if row.value_sigma <= 0:
            raise PerObjectError(f"row {row.label!r} carries a non-positive declared sigma")
        gap = row.value - offset
        if slope == 0:
            # theta cannot move this row at all; it fixes a floor under k on its own.
            need = abs(gap) / row.value_sigma
            if need > floor:
                floor, floor_label = need, row.label
            continue
        alphas.append(gap / slope)
        betas.append(row.value_sigma / abs(slope))
        labels.append(row.label)
    if not alphas:
        raise PerObjectError("every row was blind to the free parameter")
    return Axis(
        alphas=tuple(alphas),
        betas=tuple(betas),
        labels=tuple(labels),
        blind_floor=floor,
        blind_label=floor_label,
    )


def critical_coverage(axis: Axis) -> tuple[Fraction, tuple[str, str] | None]:
    """The exact smallest coverage at which this axis admits any theta, and the binding pair.

    ``max`` over ordered pairs of ``(alpha_i - alpha_j) / (beta_i + beta_j)``.  Every term is
    a rational, the set is finite, and the maximum is attained -- so the answer is a
    certificate, not an optimisation.
    """

    best = axis.blind_floor
    pair: tuple[str, str] | None = None
    count = len(axis.alphas)
    for i in range(count):
        alpha_i, beta_i = axis.alphas[i], axis.betas[i]
        for j in range(count):
            value = (alpha_i - axis.alphas[j]) / (beta_i + axis.betas[j])
            if value > best:
                best, pair = value, (axis.labels[i], axis.labels[j])
    return best, pair


def interval_at(axis: Axis, coverage: Fraction) -> dict[str, Any]:
    """The exact feasible theta interval at a declared coverage, or an empty verdict."""

    if coverage < axis.blind_floor:
        return {
            "blocking_row": axis.blind_label,
            "empty": True,
            "reason": "a row blind to the free parameter is already outside its interval",
        }
    lower = None
    upper = None
    lower_row = ""
    upper_row = ""
    for alpha, beta, label in zip(axis.alphas, axis.betas, axis.labels, strict=True):
        low = alpha - beta * coverage
        high = alpha + beta * coverage
        if lower is None or low > lower:
            lower, lower_row = low, label
        if upper is None or high < upper:
            upper, upper_row = high, label
    if lower is None or upper is None:  # pragma: no cover - build_axis forbids an empty axis
        raise PerObjectError("the axis carried no row that the free parameter can move")
    if lower > upper:
        return {
            "certificate": {
                "gap": _fraction_block(lower - upper),
                "kind": "two_row_contradiction",
                "lower_bound_from": lower_row,
                "reading": (
                    "these two published rows force theta above and below values that do "
                    "not meet at this coverage, so no single parameter satisfies both"
                ),
                "upper_bound_from": upper_row,
            },
            "empty": True,
            "reason": "two rows of this object contradict each other",
        }
    return {
        "empty": False,
        "lower": _fraction_block(lower),
        "lower_bound_from": lower_row,
        "pinned_to_a_point": lower == upper,
        "upper": _fraction_block(upper),
        "upper_bound_from": upper_row,
        "width": _fraction_block(upper - lower),
    }


def residual_structure(
    offsets: Sequence[Fraction],
    slopes: Sequence[Fraction],
    rows: Sequence[MeasuredRow],
    theta: Fraction,
) -> dict[str, Any]:
    """Sign pattern of the exact residuals at a given theta, ordered by published radius.

    Integer counts on exact rational comparisons.  A law whose residuals keep one sign for
    long stretches is failing systematically; a law whose residuals alternate is failing at
    random.  The two are different diagnoses and this tells them apart without assuming any
    distribution.
    """

    signs: list[int] = []
    for offset, slope, row in zip(offsets, slopes, rows, strict=True):
        residual = row.value - (offset + theta * slope)
        signs.append(1 if residual > 0 else (-1 if residual < 0 else 0))
    changes = sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])
    longest = 1 if signs else 0
    current = 1
    for index in range(1, len(signs)):
        current = current + 1 if signs[index] == signs[index - 1] else 1
        longest = max(longest, current)
    return {
        "longest_same_sign_run": longest,
        "points": len(signs),
        "positive_residuals": sum(1 for value in signs if value > 0),
        "rule": STRUCTURE_RULE,
        "sign_changes": changes,
        "structured": changes * STRUCTURE_RUN_DIVISOR <= max(len(signs) - 1, 0),
    }


# ---------------------------------------------------------------------------
# Step 2 -- declared law spaces
# ---------------------------------------------------------------------------


#: A law space hands back, for one galaxy, the fixed offset column ``c`` and the column
#: ``b`` that the single free per-object parameter multiplies.
LawColumns = Callable[[Galaxy], tuple[list[Fraction], list[Fraction]]]


@dataclass(frozen=True, slots=True)
class LawSpace:
    """One declared law with exactly one free scalar per object."""

    name: str
    parameter: str
    meaning: str
    columns: LawColumns
    is_control: bool = False


def newtonian_columns(prepared: Mapping[str, dict[str, Any]]) -> LawColumns:
    """v^2 = theta * V_bar^2: theta is a per-object rescale of the published baryons."""

    def build(galaxy: Galaxy) -> tuple[list[Fraction], list[Fraction]]:
        exact = prepared[galaxy.name]["v_bar_squared_exact"]
        return [Fraction(0)] * galaxy.count, list(exact)

    return build


def family_amplitude_columns(
    family: Family,
    prepared: Mapping[str, dict[str, Any]],
    cache: ColumnCache,
    a0_text: str,
    unit_text: str,
    *,
    wrong_law: bool = False,
) -> LawColumns:
    """v^2 = local(r) + theta * (w_Y B_Y + w_P B_P) at the family's own enumerated weights.

    theta = 1 is the family exactly as enumerated, so the population is directly readable
    as "how far from its own declared law does each object have to move".
    """

    weight_yukawa = Fraction(family.parameters["w_yukawa"])
    weight_power = Fraction(family.parameters["w_power"])

    def build(galaxy: Galaxy) -> tuple[list[Fraction], list[Fraction]]:
        entry = prepared[galaxy.name]
        columns = _family_columns(
            galaxy,
            family,
            a0_text,
            unit_text,
            cache,
            entry["v_bar_squared"],
            entry["radius"],
            free_arms=True,
            wrong_law=wrong_law,
        )
        offsets = [row[0] for row in columns]
        slopes = [weight_yukawa * row[1] + weight_power * row[2] for row in columns]
        return offsets, slopes

    return build


def family_arm_columns(
    family: Family,
    prepared: Mapping[str, dict[str, Any]],
    cache: ColumnCache,
    a0_text: str,
    unit_text: str,
) -> Callable[[Galaxy], tuple[list[Fraction], list[list[Fraction]]]]:
    """The same law with *both* arm weights free per object: a two-parameter law space."""

    def build(galaxy: Galaxy) -> tuple[list[Fraction], list[list[Fraction]]]:
        entry = prepared[galaxy.name]
        columns = _family_columns(
            galaxy,
            family,
            a0_text,
            unit_text,
            cache,
            entry["v_bar_squared"],
            entry["radius"],
            free_arms=True,
        )
        return [row[0] for row in columns], [[row[1], row[2]] for row in columns]

    return build


# ---------------------------------------------------------------------------
# Step 3 -- R1: the population
# ---------------------------------------------------------------------------


#: How the smallest feasible coverage is computed from an axis.  ``critical_coverage`` is
#: the reference: an explicit maximum over every ordered pair.  A caller fitting thousands
#: of pooled rows may supply an exactly equivalent solver that does not enumerate pairs.
CoverageSolver = Callable[[Axis], tuple[Fraction, tuple[str, str] | None]]


def fit_object(
    law: LawSpace,
    galaxy: Galaxy,
    rows: Sequence[MeasuredRow],
    coverages: Sequence[str],
    *,
    solver: CoverageSolver = critical_coverage,
) -> tuple[Axis, dict[str, Any]]:
    """One object, alone, with its own copy of the free parameter.  Nothing is pooled."""

    offsets, slopes = law.columns(galaxy)
    axis = build_axis(offsets, slopes, rows)
    critical, pair = solver(axis)
    at_critical = interval_at(axis, critical)
    if at_critical["empty"]:
        raise PerObjectError(
            f"{galaxy.name}: the interval is empty at its own critical coverage"
        )
    theta = _from_block(at_critical["lower"])
    return axis, {
        "at_critical_coverage": at_critical,
        "binding_pair": (
            {
                "pushes_theta_down": pair[1],
                "pushes_theta_up": pair[0],
                "reading": (
                    "below the critical coverage these two published rows demand values of "
                    "the free parameter that do not overlap; at it they meet at one point"
                ),
            }
            if pair is not None
            else None
        ),
        "critical_coverage": _fraction_block(critical),
        "declared_coverage_intervals": {
            text: interval_at(axis, Fraction(text)) for text in coverages
        },
        "object": galaxy.name,
        "points": galaxy.count,
        "residual_structure_at_critical_theta": residual_structure(
            offsets, slopes, rows, theta
        ),
        "theta_at_critical_coverage": _fraction_block(theta),
    }


def pooled_axis(
    law: LawSpace, galaxies: Sequence[Galaxy], rows_by_object: Mapping[str, Sequence[MeasuredRow]]
) -> Axis:
    """Every object's rows on one theta axis: the shared-parameter question."""

    alphas: list[Fraction] = []
    betas: list[Fraction] = []
    labels: list[str] = []
    floor = Fraction(0)
    floor_label: str | None = None
    for galaxy in galaxies:
        offsets, slopes = law.columns(galaxy)
        axis = build_axis(offsets, slopes, rows_by_object[galaxy.name])
        alphas.extend(axis.alphas)
        betas.extend(axis.betas)
        labels.extend(axis.labels)
        if axis.blind_floor > floor:
            floor, floor_label = axis.blind_floor, axis.blind_label
    return Axis(
        alphas=tuple(alphas),
        betas=tuple(betas),
        labels=tuple(labels),
        blind_floor=floor,
        blind_label=floor_label,
    )


def decompose(
    law: LawSpace,
    galaxies: Sequence[Galaxy],
    rows_by_object: Mapping[str, Sequence[MeasuredRow]],
    split: Split,
    coverages: Sequence[str] = COVERAGE_GRID,
    *,
    solver: CoverageSolver = critical_coverage,
) -> dict[str, Any]:
    """R1 for one law space: every exploration object fitted alone, whole population kept."""

    split.guard(galaxies)
    fitted = [
        fit_object(law, galaxy, rows_by_object[galaxy.name], coverages, solver=solver)
        for galaxy in galaxies
    ]
    fitted.sort(key=lambda item: item[1]["object"])
    population = [entry for _, entry in fitted]
    k_pop = max(_from_block(entry["critical_coverage"]) for entry in population)
    worst = max(
        population, key=lambda entry: (_from_block(entry["critical_coverage"]), entry["object"])
    )
    # Every object's exact interval at the one coverage where the whole population exists.
    # This is the interval R2 adjudicates: below it some object has no solution at all, so
    # the constancy question would be about an incomplete population.
    for axis, entry in fitted:
        interval = interval_at(axis, k_pop)
        if interval["empty"]:
            raise PerObjectError(
                f"{entry['object']}: empty at the population coverage, which is at least "
                "its own critical coverage and therefore cannot be empty"
            )
        entry["interval_at_population_coverage"] = interval
    common, common_pair = solver(pooled_axis(law, galaxies, rows_by_object))
    if common < k_pop:
        raise PerObjectError(
            "the shared-parameter coverage fell below the per-object one, which is "
            "arithmetically impossible: the pooled pair set contains every per-object pair"
        )
    return {
        "law": law.name,
        "meaning": law.meaning,
        "objects_fitted": [entry["object"] for entry in population],
        "parameter": law.parameter,
        "population": population,
        "population_kept_not_ranked": True,
        "shared_parameter_coverage": _fraction_block(common),
        "shared_parameter_binding_pair": (
            {"pushes_theta_down": common_pair[1], "pushes_theta_up": common_pair[0]}
            if common_pair is not None
            else None
        ),
        "smallest_coverage_with_a_population": _fraction_block(k_pop),
        "smallest_coverage_with_a_population_set_by": worst["object"],
        "is_control": law.is_control,
    }


# ---------------------------------------------------------------------------
# Step 4 -- R2: does one value lie in every interval?
# ---------------------------------------------------------------------------

CONSTANT = "CONSTANT"
VARIES = "VARIES"
NO_POPULATION = "NO_POPULATION"


def adjudicate(intervals: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Decide constancy from a set of per-object intervals, exactly, with a certificate.

    Three outcomes and no fourth: some object has no solution at all at this coverage, so
    the question is not yet askable; one value lies in every interval, exhibited; or two
    intervals are disjoint, exhibited with the exact gap between them.
    """

    missing = sorted(name for name, entry in intervals.items() if entry.get("empty"))
    if missing:
        return {
            "objects_without_any_solution": missing,
            "reading": (
                "at least one object admits no value of the parameter at all at this "
                "coverage, so there is no population to diagnose. Widening is required "
                "before the constancy question means anything"
            ),
            "verdict": NO_POPULATION,
        }
    if not intervals:
        raise PerObjectError("no intervals were offered to the adjudicator")
    lower_name, lower = max(
        ((name, _from_block(entry["lower"])) for name, entry in intervals.items()),
        key=lambda item: (item[1], item[0]),
    )
    upper_name, upper = min(
        ((name, _from_block(entry["upper"])) for name, entry in intervals.items()),
        key=lambda item: (item[1], item[0]),
    )
    if lower <= upper:
        return {
            "reading": (
                "this single value lies inside every object's exact interval, so the "
                "parameter is constant within the declared uncertainties at this coverage "
                "and the law needs no per-object freedom here"
            ),
            "shared_value_range": {
                "highest": _fraction_block(upper),
                "lowest": _fraction_block(lower),
            },
            "verdict": CONSTANT,
            "witness": _fraction_block(lower),
            "witness_verified_in_every_interval": True,
        }
    return {
        "certificate": {
            "gap": _fraction_block(lower - upper),
            "highest_lower_bound_from": lower_name,
            "kind": "disjoint_object_intervals",
            "lowest_upper_bound_from": upper_name,
            "reading": (
                f"{lower_name} requires the parameter at or above its lower bound and "
                f"{upper_name} requires it at or below its upper bound, and the first "
                "exceeds the second by the exact gap recorded here, so no single value "
                "reaches both objects"
            ),
        },
        "reading": (
            "no single value lies in every object's interval: the per-object parameter is "
            "not constant within the declared uncertainties at this coverage, and the gap "
            "below is a measured lower bound on how far apart the objects insist on being"
        ),
        "verdict": VARIES,
    }


def verify_adjudication(
    record: Mapping[str, Any], intervals: Mapping[str, Mapping[str, Any]]
) -> None:
    """Re-check a constancy verdict against the intervals it claims to summarise.

    This is the R2 falsifier turned into a guard: declaring a parameter constant when no
    single value lies in all intervals, or variable when one does, must raise here rather
    than reach a receipt.
    """

    verdict = record.get("verdict")
    empty = {name for name, entry in intervals.items() if entry.get("empty")}
    if verdict == NO_POPULATION:
        if not empty:
            raise PerObjectError("NO_POPULATION claimed but every object has a solution")
        return
    if empty:
        raise PerObjectError(f"{verdict} claimed while {sorted(empty)} have no solution")
    if verdict == CONSTANT:
        witness = _from_block(record["witness"])
        for name, entry in sorted(intervals.items()):
            low = _from_block(entry["lower"])
            high = _from_block(entry["upper"])
            if not low <= witness <= high:
                raise PerObjectError(
                    f"CONSTANT claimed but the witness is outside {name}'s interval"
                )
        return
    if verdict == VARIES:
        certificate = record["certificate"]
        low = _from_block(intervals[certificate["highest_lower_bound_from"]]["lower"])
        high = _from_block(intervals[certificate["lowest_upper_bound_from"]]["upper"])
        if low <= high:
            raise PerObjectError(
                "VARIES claimed but the exhibited intervals are not disjoint"
            )
        if _from_block(certificate["gap"]) != low - high:
            raise PerObjectError("the exhibited gap does not equal the exhibited endpoints")
        for name, entry in sorted(intervals.items()):
            if _from_block(entry["lower"]) > low or _from_block(entry["upper"]) < high:
                raise PerObjectError(
                    f"VARIES exhibited a pair that {name} already beats; the certificate "
                    "does not name the binding objects"
                )
        return
    raise PerObjectError(f"unknown constancy verdict {verdict!r}")


def diagnose(decomposition: Mapping[str, Any], coverages: Sequence[str] = COVERAGE_GRID) -> dict[str, Any]:
    """R2 for one law space: the constancy question at every declared coverage and at k_pop."""

    population = decomposition["population"]
    by_coverage: dict[str, Any] = {}
    for text in coverages:
        intervals = {
            entry["object"]: entry["declared_coverage_intervals"][text] for entry in population
        }
        record = adjudicate(intervals)
        verify_adjudication(record, intervals)
        record["objects_offered"] = len(intervals)
        record["objects_with_a_solution"] = sum(
            1 for entry in intervals.values() if not entry.get("empty")
        )
        by_coverage[text] = record

    k_pop = _from_block(decomposition["smallest_coverage_with_a_population"])
    common = _from_block(decomposition["shared_parameter_coverage"])
    price = common / k_pop if k_pop > 0 else Fraction(1)

    # R2 proper: the exact interval per object at the coverage where the population exists,
    # and the question of whether one value lies in every one of them.  The verdict is read
    # off the intervals, never off the price; the two must agree and a disagreement raises.
    at_population = {
        entry["object"]: entry["interval_at_population_coverage"] for entry in population
    }
    population_record = adjudicate(at_population)
    verify_adjudication(population_record, at_population)
    if (population_record["verdict"] == CONSTANT) != (price == 1):
        raise PerObjectError(
            "the constancy verdict read off the intervals disagrees with the price of "
            "universality read off the certificates; one of the two derivations is wrong"
        )
    thetas = {
        entry["object"]: _from_block(entry["theta_at_critical_coverage"])
        for entry in population
    }
    values = sorted(thetas.values())
    return {
        "at_declared_coverage_factors": by_coverage,
        "at_population_coverage": {
            **population_record,
            "coverage_factor": decomposition["smallest_coverage_with_a_population"],
            "intervals": at_population,
            "reading": (
                "this is R2's own test, run at the exact coverage where every object first "
                "admits a solution: one interval per object, and the question of whether a "
                "single value lies inside all of them. Below this coverage the population "
                "is incomplete and the question is not yet askable"
            ),
        },
        "law": decomposition["law"],
        "parameter": decomposition["parameter"],
        "per_object_point_estimates": {
            name: _fraction_block(value) for name, value in sorted(thetas.items())
        },
        "point_estimate_span": _fraction_block(values[-1] - values[0]),
        "price_of_universality": {
            "exact": _fraction_data(price),
            "decimal": _num(price),
            "is_one": price == 1,
            "reading": (
                "the factor by which the declared intervals must be widened, beyond what "
                "the objects already need individually, before one shared value fits every "
                "object. Exactly 1 means universality is free: the binding pair lies inside "
                "a single object and the per-object parameter is already constant. It is "
                "not a measure of how well the law fits -- a law too weak to constrain its "
                "own parameter also prices universality at 1, which the control in this "
                "receipt demonstrates"
            ),
            "shared_parameter_coverage": decomposition["shared_parameter_coverage"],
            "smallest_coverage_with_a_population": (
                decomposition["smallest_coverage_with_a_population"]
            ),
        },
        "residual_structure": {
            entry["object"]: entry["residual_structure_at_critical_theta"]
            for entry in population
        },
        "structured_object_count": sum(
            1
            for entry in population
            if entry["residual_structure_at_critical_theta"]["structured"]
        ),
        "verdict_at_population_coverage": population_record["verdict"],
    }


# ---------------------------------------------------------------------------
# Step 5 -- the vetted instrument, used as an independent check
# ---------------------------------------------------------------------------


def offset_system(
    offsets: Sequence[Fraction],
    columns: Sequence[Sequence[Fraction]],
    rows: Sequence[MeasuredRow],
    coverage: Fraction,
) -> LinearSystem:
    """``A x <= b`` for ``c_i + sum_j A_ij x_j`` inside each declared interval."""

    matrix: list[tuple[Fraction, ...]] = []
    rhs: list[Fraction] = []
    origin: list[tuple[str, str]] = []
    for offset, column, row in zip(offsets, columns, rows, strict=True):
        low, high = row.value_interval(coverage)
        matrix.append(tuple(column))
        rhs.append(high - offset)
        origin.append((row.label, "upper"))
        matrix.append(tuple(-cell for cell in column))
        rhs.append(-(low - offset))
        origin.append((row.label, "lower"))
    return LinearSystem(tuple(matrix), tuple(rhs), tuple(origin))


def crosscheck_against_simplex(
    law: LawSpace,
    galaxies: Sequence[Galaxy],
    rows_by_object: Mapping[str, Sequence[MeasuredRow]],
    decomposition: Mapping[str, Any],
) -> dict[str, Any]:
    """The closed form's verdict at and just below k* must match the audited simplex.

    The interval arithmetic above is a closed form, and a closed form that is wrong is
    wrong silently.  So each object's critical coverage is handed to the repository's own
    exact rational simplex twice: at ``k*``, where feasibility is claimed, and at a declared
    rational strictly below it, where infeasibility is claimed and a Farkas witness must
    appear.  A single disagreement voids the receipt.
    """

    checks: list[dict[str, Any]] = []
    for entry in decomposition["population"]:
        galaxy = next(item for item in galaxies if item.name == entry["object"])
        offsets, slopes = law.columns(galaxy)
        columns = [[value] for value in slopes]
        rows = rows_by_object[galaxy.name]
        critical = _from_block(entry["critical_coverage"])
        below = critical * Fraction(999, 1000)
        at = decide_system(offset_system(offsets, columns, rows, critical))
        under = decide_system(offset_system(offsets, columns, rows, below))
        if at["verdict"] != FEASIBLE:
            raise PerObjectError(
                f"{galaxy.name}: the simplex refuses the closed form's critical coverage"
            )
        if under["verdict"] != INFEASIBLE:
            raise PerObjectError(
                f"{galaxy.name}: the simplex finds a solution below the closed form's "
                "critical coverage, so the closed form is not the minimum it claims to be"
            )
        witness = under["witness"]
        checks.append(
            {
                "at_critical_coverage": at["verdict"],
                "below_critical_coverage": under["verdict"],
                "below_critical_coverage_factor": "999/1000",
                "farkas_rows_below": witness["unreachable_rows"],
                "farkas_term_count_below": len(witness["terms"]),
                "object": galaxy.name,
            }
        )
    checks.sort(key=lambda item: item["object"])
    return {
        "agreements": len(checks),
        "checks": checks,
        "disagreements": 0,
        "instrument": "sigma_theory_compiler.tolerance_aware_fitting.decide_system",
        "reading": (
            "the exact rational simplex confirms, for every object, that the closed form's "
            "critical coverage is feasible and that a strictly smaller declared coverage is "
            "not. The two derivations are independent: one is a maximum over ordered pairs "
            "of rationals, the other a phase-one simplex with a re-verified Farkas witness"
        ),
    }


# ---------------------------------------------------------------------------
# Step 6 -- the two-parameter law space, by exact projection
# ---------------------------------------------------------------------------


def project_box(
    offsets: Sequence[Fraction],
    columns: Sequence[Sequence[Fraction]],
    rows: Sequence[MeasuredRow],
    coverage: Fraction,
    names: Sequence[str],
) -> dict[str, Any]:
    """Exact coordinate projections of one object's feasible polytope, by exact LP."""

    system = offset_system(offsets, columns, rows, coverage)
    verdict = decide_system(system)
    if verdict["verdict"] != FEASIBLE:
        return {
            "empty": True,
            "reason": "this object admits no parameter pair at this coverage",
            "unreachable_rows": verdict["witness"]["unreachable_rows"][:8],
        }
    intervals: dict[str, Any] = {}
    for index, name in enumerate(names):
        objective = [Fraction(1) if position == index else Fraction(0) for position in range(len(names))]
        bounds: dict[str, Any] = {}
        for direction, maximize in (("lower", False), ("upper", True)):
            fresh = decide_system(offset_system(offsets, columns, rows, coverage))
            status, value = fresh["simplex"].optimize(objective, maximize=maximize)
            if status != "OPTIMAL":
                bounds[direction] = None
                bounds[f"{direction}_status"] = status
            else:
                bounds[direction] = _fraction_block(value)
                bounds[f"{direction}_status"] = status
        intervals[name] = bounds
    return {"empty": False, "intervals": intervals}


def two_parameter_diagnosis(
    family: Family,
    galaxies: Sequence[Galaxy],
    rows_by_object: Mapping[str, Sequence[MeasuredRow]],
    prepared: Mapping[str, dict[str, Any]],
    cache: ColumnCache,
    split: Split,
    a0_text: str,
    unit_text: str,
    coverages: Sequence[tuple[str, Fraction]],
) -> dict[str, Any]:
    """R2 with two free parameters per object: per-coordinate answer *and* the joint one."""

    split.guard(galaxies)
    names = ("w_yukawa", "w_power")
    builder = family_arm_columns(family, prepared, cache, a0_text, unit_text)
    declared = {
        "w_power": Fraction(family.parameters["w_power"]),
        "w_yukawa": Fraction(family.parameters["w_yukawa"]),
    }
    by_coverage: dict[str, Any] = {}
    for text, coverage in coverages:
        boxes: dict[str, Any] = {}
        pooled_offsets: list[Fraction] = []
        pooled_columns: list[list[Fraction]] = []
        pooled_rows: list[MeasuredRow] = []
        for galaxy in galaxies:
            offsets, columns = builder(galaxy)
            rows = rows_by_object[galaxy.name]
            boxes[galaxy.name] = project_box(offsets, columns, rows, coverage, names)
            pooled_offsets.extend(offsets)
            pooled_columns.extend(columns)
            pooled_rows.extend(rows)
        per_parameter: dict[str, Any] = {}
        for name in names:
            intervals = {
                galaxy: (
                    {"empty": True}
                    if box["empty"]
                    else {
                        "lower": box["intervals"][name]["lower"],
                        "upper": box["intervals"][name]["upper"],
                    }
                )
                for galaxy, box in boxes.items()
            }
            record = adjudicate(intervals)
            verify_adjudication(record, intervals)
            record["objects_offered"] = len(intervals)
            record["objects_with_a_solution"] = sum(
                1 for entry in intervals.values() if not entry.get("empty")
            )
            record["declared_family_value"] = {
                "decimal": _num(declared[name]),
                "exact": _fraction_data(declared[name]),
                "inside_every_object_interval": all(
                    (not box["empty"])
                    and _from_block(box["intervals"][name]["lower"])
                    <= declared[name]
                    <= _from_block(box["intervals"][name]["upper"])
                    for box in boxes.values()
                ),
            }
            per_parameter[name] = record
        joint = decide_system(offset_system(pooled_offsets, pooled_columns, pooled_rows, coverage))
        by_coverage[text] = {
            "coordinate_projections": boxes,
            "joint_shared_pair": {
                "reading": (
                    "one (w_yukawa, w_power) shared by every object, decided by the exact "
                    "rational simplex over the pooled rows"
                ),
                "unreachable_rows": (
                    joint["witness"]["unreachable_rows"][:8]
                    if joint["verdict"] == INFEASIBLE
                    else None
                ),
                "verdict": joint["verdict"],
            },
            "per_parameter": per_parameter,
            "per_parameter_constant_but_no_joint_pair": (
                all(record["verdict"] == CONSTANT for record in per_parameter.values())
                and joint["verdict"] == INFEASIBLE
            ),
        }
    return {
        "at_coverage_factors": by_coverage,
        "caveat": TWO_PARAMETER_CAVEAT,
        "coverage_factors": [
            {"exact": _fraction_data(value), "label": label} for label, value in coverages
        ],
        "family_ordinal": family.ordinal,
        "grid_point": {"a0": a0_text, "length_unit": unit_text},
        "parameters": list(names),
        "widened_coverage_source": (
            "the declared grid, plus the one-parameter population coverage of the same "
            "family. That widening is certificate-derived, not chosen: the one-parameter "
            "law is the two-parameter law restricted to the ray (theta w_Y, theta w_P), so "
            "the two-parameter feasible set contains the one-parameter one at the same "
            "coverage and every object is guaranteed non-empty there"
        ),
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_law_spaces(
    families: Sequence[Family],
    prepared: Mapping[str, dict[str, Any]],
    cache: ColumnCache,
    a0_text: str,
    unit_text: str,
) -> list[LawSpace]:
    """Every declared law space, in a fixed order, with the control last."""

    spaces = [
        LawSpace(
            columns=newtonian_columns(prepared),
            meaning=(
                "the published baryons alone, with one per-object rescale of the declared "
                "universal mass-to-light ratios. theta = 1 is the declared convention, so "
                "the population reads directly as how much invisible mass each galaxy would "
                "have to hide in its own visible matter"
            ),
            name="newtonian_baryons_only",
            parameter="baryonic_rescale",
        )
    ]
    for family in families:
        spaces.append(
            LawSpace(
                columns=family_amplitude_columns(family, prepared, cache, a0_text, unit_text),
                meaning=(
                    "the family's own screened nonlocal boost, at its own enumerated arm "
                    "weights, with one per-object amplitude on top of the law's local "
                    "factor. theta = 1 is the family exactly as enumerated"
                ),
                name=f"family_{family.ordinal}_amplitude",
                parameter="boost_amplitude",
            )
        )
    best = select_best_family(families)
    spaces.append(
        LawSpace(
            columns=family_amplitude_columns(
                best, prepared, cache, a0_text, unit_text, wrong_law=True
            ),
            is_control=True,
            meaning=(
                "the declared best family with its local factor inverted, so the "
                "modification grows where gravity is strong: the same complexity, the wrong "
                "asymptotics. It must need a strictly larger k_pop than every real family"
            ),
            name="deliberately_wrong_law",
            parameter="boost_amplitude",
        )
    )
    return spaces


def build_receipt(root: Path) -> dict[str, Any]:
    """R1 and R2 over the declared law spaces, sealed and replayable."""

    root = root.resolve()
    galaxies, provenance = load_galaxies(root)
    families = load_families(root)

    # The split is declared and sealed here, before a single column is built.
    split = declare_split([galaxy.name for galaxy in galaxies])
    split_block = split.block()

    exploration = [galaxy for galaxy in galaxies if galaxy.name in set(split.exploration)]
    if len(exploration) != len(split.exploration):
        raise PerObjectError("the exploration set does not match the declared split")

    convention = provenance["mass_to_light_convention"]
    upsilon_disk = Fraction(convention["disk_3_6um"])
    upsilon_bul = Fraction(convention["bulge_3_6um"])
    source = (
        f"{provenance['source']['primary_citation']}; {provenance['source']['table']}; "
        f"{provenance['source']['dataset_doi']}"
    )
    prepared = {
        galaxy.name: prepare_galaxy(galaxy, upsilon_disk, upsilon_bul, QUADRATURE)
        for galaxy in exploration
    }
    rows_by_object = {galaxy.name: measured_rows(galaxy, source) for galaxy in exploration}

    a0_text = REFERENCE_GRID_POINT["a0"]
    unit_text = REFERENCE_GRID_POINT["length_unit"]
    cache = ColumnCache(prepared)
    spaces = build_law_spaces(families, prepared, cache, a0_text, unit_text)

    decompositions: dict[str, Any] = {}
    diagnoses: dict[str, Any] = {}
    for law in spaces:
        decomposition = decompose(law, exploration, rows_by_object, split)
        decompositions[law.name] = decomposition
        diagnoses[law.name] = diagnose(decomposition)

    best = select_best_family(families)
    checked = ("newtonian_baryons_only", f"family_{best.ordinal}_amplitude", "deliberately_wrong_law")
    crosschecks = {
        name: crosscheck_against_simplex(
            next(law for law in spaces if law.name == name),
            exploration,
            rows_by_object,
            decompositions[name],
        )
        for name in checked
    }

    best_space = f"family_{best.ordinal}_amplitude"
    widened = _from_block(
        decompositions[best_space]["smallest_coverage_with_a_population"]
    )
    two_parameter = two_parameter_diagnosis(
        best,
        exploration,
        rows_by_object,
        prepared,
        cache,
        split,
        a0_text,
        unit_text,
        [(text, Fraction(text)) for text in COVERAGE_GRID]
        + [("one_parameter_population_coverage", widened)],
    )

    control = decompositions["deliberately_wrong_law"]
    control_k = _from_block(control["smallest_coverage_with_a_population"])
    family_names = [f"family_{family.ordinal}_amplitude" for family in families]
    beaten_by = sorted(
        name
        for name in family_names
        if _from_block(decompositions[name]["smallest_coverage_with_a_population"]) >= control_k
    )
    if beaten_by:
        raise PerObjectError(
            "the deliberately wrong law did not need a strictly larger coverage than "
            f"{beaten_by[0]}; the per-object machinery is not discriminating"
        )

    def _ladder(name: str) -> dict[str, int]:
        """Objects admitting any solution at all, at each declared coverage factor."""

        return {
            text: diagnoses[name]["at_declared_coverage_factors"][text][
                "objects_with_a_solution"
            ]
            for text in COVERAGE_GRID
        }

    newtonian = diagnoses["newtonian_baryons_only"]
    best_diagnosis = diagnoses[best_space]
    contrast = {
        "control_law": {
            "law": "deliberately_wrong_law",
            "objects_with_a_solution_by_declared_coverage": _ladder("deliberately_wrong_law"),
            "price_of_universality": diagnoses["deliberately_wrong_law"][
                "price_of_universality"
            ]["decimal"],
            "reading": (
                "the wrong law needs a strictly larger coverage than every real family "
                "before any object admits a solution, and it still prices universality at "
                "1. That is the point of publishing it: a constant parameter is not "
                "evidence for a law, it is evidence about the parameter"
            ),
            "smallest_coverage_with_a_population": control["smallest_coverage_with_a_population"][
                "decimal"
            ],
            "verdict_at_population_coverage": diagnoses["deliberately_wrong_law"][
                "verdict_at_population_coverage"
            ],
        },
        "reading": (
            "This is the separation the pooled INFEASIBLE could not make. Newtonian baryons "
            "and the screened families both fail when pooled, and they fail for different "
            "reasons: the per-object baryonic rescale refuses to take one value, while the "
            "screened boost amplitude takes one value everywhere and is simply the wrong "
            "size. The first is a dead end -- the parameter is not a constant of nature on "
            "these objects. The second is a lead, because a law whose parameter is already "
            "constant is a law with nothing left to fit, and what remains is its shape."
        ),
        "screened_family": {
            "law": best_space,
            "objects_with_a_solution_by_declared_coverage": _ladder(best_space),
            "per_object_point_estimates": {
                name: block["decimal"]
                for name, block in best_diagnosis["per_object_point_estimates"].items()
            },
            "price_of_universality": best_diagnosis["price_of_universality"]["decimal"],
            "smallest_coverage_with_a_population": decompositions[best_space][
                "smallest_coverage_with_a_population"
            ]["decimal"],
            "structured_object_count": best_diagnosis["structured_object_count"],
            "verdict_at_population_coverage": best_diagnosis["verdict_at_population_coverage"],
        },
        "newtonian_baryons": {
            "law": "newtonian_baryons_only",
            "objects_with_a_solution_by_declared_coverage": _ladder("newtonian_baryons_only"),
            "per_object_point_estimates": {
                name: block["decimal"]
                for name, block in newtonian["per_object_point_estimates"].items()
            },
            "price_of_universality": newtonian["price_of_universality"]["decimal"],
            "smallest_coverage_with_a_population": decompositions["newtonian_baryons_only"][
                "smallest_coverage_with_a_population"
            ]["decimal"],
            "structured_object_count": newtonian["structured_object_count"],
            "variation_certificate": newtonian["at_population_coverage"].get("certificate"),
            "verdict_at_population_coverage": newtonian["verdict_at_population_coverage"],
        },
    }

    body: dict[str, Any] = {
        "assumptions": dict(ASSUMPTIONS),
        "claims": dict(CLAIMS),
        "contrast": contrast,
        "counts": {
            "confirmation_objects": len(split.confirmation),
            "exploration_objects": len(exploration),
            "exploration_points": sum(galaxy.count for galaxy in exploration),
            "law_spaces": len(spaces),
            "one_parameter_fits": len(spaces) * len(exploration),
        },
        "coverage_factors": list(COVERAGE_GRID),
        "data_provenance": {
            "columns": provenance["columns"],
            "data_sha256": provenance["data_sha256"],
            "mass_to_light_convention": provenance["mass_to_light_convention"],
            "source": provenance["source"],
        },
        "decision": _decision(contrast, split),
        "exploration_confirmation_split": split_block,
        "exploratory_caveat": {
            "confirmation_set_is_not_virgin_data": (
                "the predecessor receipt runs/gpu-baryonic-screen/real-data-exploratory-v1"
                ".json already scanned its declared universal-constant grid against all six "
                "galaxies, including the two withheld here. This split protects the Tier 7 "
                "reconciliation claim that R4 and R5 will consume; it does not restore "
                "untouched-data status to the withheld objects, and any later sealed trial "
                "must say so"
            ),
            "may_be_cited_as_confirmation": False,
            "sealed_no_refit_trial": False,
            "statement": (
                "Every fit here gives each object its own copy of a free parameter. That is "
                "exactly the freedom the sealed trial forbids, and it is why this receipt "
                "generates hypotheses and confirms nothing. The confirmation set was never "
                "handed to a fitting function: the guard raises rather than declines"
            ),
        },
        "grid_point": {
            "a0_kms2_per_kpc": a0_text,
            "declared_before_the_fit": True,
            "length_unit_kpc": unit_text,
            "reading": (
                "the reference point of the predecessor module's declared grid, chosen "
                "there before any data was opened and reused unchanged here so that the "
                "per-object populations are not a scan over grid points"
            ),
        },
        "instrument_crosscheck": crosschecks,
        "method": {
            "critical_coverage": (
                "k*(object) = max over ordered pairs (i, j) of "
                "(alpha_i - alpha_j) / (beta_i + beta_j), with alpha = (v - c) / b and "
                "beta = sigma / |b|. A maximum over a finite set of exact rationals: the "
                "argmax pair is the certificate and there is nothing to converge"
            ),
            "price_of_universality": (
                "k_common / k_pop, where k_pop is the largest per-object critical coverage "
                "and k_common is the same maximum taken over pairs drawn from all objects "
                "at once. It is at least 1 by construction because the pooled pair set "
                "contains every per-object pair"
            ),
            "structure_rule": STRUCTURE_RULE,
            "wrong_law_expectation": WRONG_LAW_EXPECTATION,
        },
        "per_object_decomposition_r1": decompositions,
        "parameter_variation_diagnosis_r2": diagnoses,
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "trial_type": TRIAL_TYPE,
        "two_parameter_diagnosis_r2": two_parameter,
    }
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise PerObjectError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    return {**body, "content_sha256": canonical_sha256(body)}


def _decision(contrast: Mapping[str, Any], split: Split) -> str:
    newtonian = contrast["newtonian_baryons"]
    screened = contrast["screened_family"]
    return (
        f"EXPLORATORY: over {len(split.exploration)} exploration galaxies fitted one at a "
        f"time, the Newtonian per-object baryonic rescale is {newtonian['verdict_at_population_coverage']} "
        f"(price of universality {newtonian['price_of_universality']}), while the screened "
        f"family's per-object boost amplitude is {screened['verdict_at_population_coverage']} "
        f"(price {screened['price_of_universality']}). Both laws are refused by the pooled "
        "confrontation, and this receipt says why they are refused differently. The "
        f"{len(split.confirmation)} confirmation galaxies were never handed to a fitting "
        "function. Nothing here may be cited as a confirmation."
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Reject tamper or drift by exact deterministic replay."""

    if receipt.get("schema_version") != RESULT_SCHEMA:
        raise PerObjectError("receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise PerObjectError("receipt seal changed")
    if receipt.get("claims") != CLAIMS:
        raise PerObjectError("claims changed")
    if receipt.get("trial_type") != TRIAL_TYPE:
        raise PerObjectError("trial type changed")
    split = receipt.get("exploration_confirmation_split")
    if not isinstance(split, Mapping):
        raise PerObjectError("the receipt carries no exploration/confirmation split")
    sealed = {key: value for key, value in split.items() if key != "split_sha256"}
    if split.get("split_sha256") != canonical_sha256(sealed):
        raise PerObjectError("the split seal changed")
    if split.get("salt") != SPLIT_SALT or split.get("rule") != SPLIT_RULE:
        raise PerObjectError("the split rule changed")
    withheld = set(split.get("confirmation", ()))
    fitted = {
        name
        for decomposition in receipt.get("per_object_decomposition_r1", {}).values()
        for name in decomposition.get("objects_fitted", ())
    }
    if withheld & fitted:
        raise PerObjectError("a confirmation-set object appears in a fitted population")
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise PerObjectError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    replayed = build_receipt(root)
    if dict(receipt) != replayed:
        raise PerObjectError("receipt exact replay changed")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    """Write a receipt once; a differing rewrite is refused rather than silently accepted."""

    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise PerObjectError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tier 7 R1/R2 per-object decomposition.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=RECEIPT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if args.validate_checked:
        validate_receipt(_load_json(output), root=root)
        return 0
    receipt = build_receipt(root)
    write_immutable(output, receipt)
    validate_receipt(receipt, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A0_GRID",
    "ASSUMPTIONS",
    "CLAIMS",
    "CONFIRMATION_COUNT",
    "CONSTANT",
    "COVERAGE_GRID",
    "LENGTH_UNIT_GRID",
    "NO_POPULATION",
    "RECEIPT_PATH",
    "RESULT_SCHEMA",
    "SCOPE",
    "SPLIT_RULE",
    "SPLIT_SALT",
    "STRUCTURE_RULE",
    "TRIAL_TYPE",
    "VARIES",
    "Axis",
    "ConfirmationSetTouched",
    "CoverageSolver",
    "LawSpace",
    "PerObjectError",
    "Split",
    "adjudicate",
    "build_axis",
    "build_law_spaces",
    "build_receipt",
    "critical_coverage",
    "crosscheck_against_simplex",
    "declare_split",
    "decompose",
    "diagnose",
    "fit_object",
    "interval_at",
    "main",
    "offset_system",
    "pooled_axis",
    "project_box",
    "residual_structure",
    "two_parameter_diagnosis",
    "validate_receipt",
    "verify_adjudication",
    "write_immutable",
]
