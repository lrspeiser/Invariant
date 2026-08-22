"""Reconciliation search (Tier 7, R4): one law for the whole population, no per-object knobs.

Tier 7 splits a global fit into two passes.  Pass 1 (R1) fits every object *independently*
and keeps the whole population of local solutions; pass 2 (R2) turns each per-object
parameter into an exact interval.  This module is pass 3.  It asks the only question that
makes the first two worth running:

    is there a **single** law, drawn from a declared law space, whose prediction for every
    object's parameters lands inside every one of R2's intervals, with **no free parameter
    left per object**?

Two answers are allowed and both are deliverables.

``RECONCILED``
    a universal coefficient vector, re-checked against every declared row in exact rational
    arithmetic, plus the exact coefficient box (the extremes of each universal constant over
    the whole feasible polytope) and a per-object table showing each prediction inside its
    R2 interval.  The width of the linear system is the number of *universal* coefficients
    and does not depend on how many objects are present; :func:`freedom_report` is tested
    against exactly that.

``OBSTRUCTED``
    an exact obstruction, not a shrug.  Every form in the declared space carries its own
    **Farkas certificate**: nonnegative multipliers ``lambda`` with ``lambda^T A = 0`` and
    ``lambda^T b < 0``, so a nonnegative combination of R2's own interval inequalities adds
    up to ``0 <= a strictly negative number``.  The certificate machinery is not reinvented
    here: rows go through :func:`sigma_theory_compiler.tolerance_aware_fitting.build_system`
    and verdicts through :func:`~sigma_theory_compiler.tolerance_aware_fitting.decide_system`,
    the same instrument :mod:`~sigma_theory_compiler.real_data_gravity_confrontation` uses,
    and the zero-per-object-freedom structure is checked with that module's own
    :class:`~sigma_theory_compiler.real_data_gravity_confrontation.Design` and
    :func:`~sigma_theory_compiler.real_data_gravity_confrontation.universal_parameter_width`.

**The obstruction is a measurement, not a verdict word.**  Only ``b`` depends on the
coverage factor ``k`` that scales R2's halfwidths, and it does so affinely, so the *same*
multipliers keep certifying infeasibility for every ``k`` below an exactly computable
break-even.  Taking the minimum over the whole enumerated space gives
``space_break_even_coverage``: **R2's intervals would every one of them have to be at least
this many times wider before any law in the declared space could reconcile them.**  That is
the "which parameter refused to stay constant, and by how much" that Tier 7 asks for, and it
is read off the certificate rather than fitted.

**Where a fake reconciliation would come from, and what stops it.**  R4's falsifier is "a
reconciled law that quietly retains a per-object parameter".  A per-object knob does not
arrive labelled; it arrives as a covariate column handed over by pass 1 that happens to be
an indicator of one object.  Two independent gates stop it:

*Width independence.*  The universal width is recomputed on a sub-population and on the full
population and must be identical.  A knob added per object fails here.

*Column support.*  A universal coefficient whose design column is nonzero for exactly one
object can only ever move that one object, which is a per-object parameter whatever it is
called.  Such a form is **refused**, and the refusal records the linear program's own verdict
on it -- in the twin control that verdict is ``FEASIBLE``, so the receipt shows the gate
turning a would-be reconciliation into an obstruction.  Width independence alone would miss
this, because one indicator for one named object does not grow with the population.

**The Tier 7 discipline is enforced structurally.**  A population that declares no
exploration/confirmation split, or that hands this search an object sitting in the
confirmation set, is refused before a single row is built.  Tier 7 states plainly that a
reconciliation reported without that split is worthless; here it is not reportable.

**The negative control is catalogue-independent.**  Two objects with *identical* covariates
and *disjoint* parameter intervals cannot be reconciled by any function of covariates
whatsoever -- ``f(o1) = f(o2)`` for every such ``f``, and the intervals do not meet.  That
argument is computed directly, without the linear program, and cross-checked against the
program's verdict on every enumerated form.  So the obstruction control does not rest on the
search failing to find something.

Nothing on a certificate path is a float.  Covariates, channels, intervals, design columns,
multipliers, and break-even coverages are all :class:`fractions.Fraction`; the only decimals
in the receipt are renderings produced by exact round-half-even integer division, and no
decision consumes one.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

from .real_data_gravity_confrontation import Design, universal_parameter_width
from .sigma_core import canonical_sha256
from .tolerance_aware_fitting import (
    FEASIBLE,
    INFEASIBLE,
    MeasuredRow,
    _Simplex,
    build_system,
    decide_system,
    decimal_string,
)

RESULT_SCHEMA = "invariant-reconciliation-search-1.0"
RECEIPT_PATH = "runs/math/reconciliation/search-v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/reconciliation_search.py"
TEST_PATH = "tests/test_reconciliation_search.py"

#: The verdict when one law in the declared space reaches every R2 interval with no
#: per-object freedom left.
RECONCILED = "RECONCILED"
#: The verdict when every form in the declared space carries a re-verified Farkas witness.
OBSTRUCTED = "OBSTRUCTED"

#: Why a form was thrown out before it could be a candidate.  A refusal is not an
#: infeasibility: the form was never allowed to compete.
REFUSAL_PER_OBJECT_FREEDOM = "column_supported_by_a_single_object"
REFUSAL_WIDTH_GROWS = "universal_width_depends_on_the_population"
REFUSAL_UNDEFINED_CHANNEL = "channel_undefined_on_a_population_member"

REFUSAL_REASONS = (
    REFUSAL_PER_OBJECT_FREEDOM,
    REFUSAL_UNDEFINED_CHANNEL,
    REFUSAL_WIDTH_GROWS,
)

#: The channel every form carries.  A law with no constant term is a declaration that the
#: parameter vanishes when every covariate does, which is a physical claim nobody made here.
CONSTANT_CHANNEL = "one"

#: Declared, finite caps.  Every one of them is checked, and a violation stops the run.
SEARCH_CAPS = {
    "max_covariates": 6,
    "max_extra_channels_per_parameter": 3,
    "max_forms": 4096,
    "max_objects": 64,
    "max_parameters": 4,
    "max_rows": 512,
    "max_universal_width": 12,
}

#: Decimal places for rendered-only numbers.  The exact rational is always carried beside
#: them and every comparison in this module consumes the rational.
RENDER_PLACES = 12

#: How many forms carry a **fully transcribed** record into the receipt.  Declared before the
#: run, not chosen after seeing one: the Occam-first forms, plus the form the space-level
#: obstruction is actually read from (the tightest break-even) and, when the search succeeds,
#: the reconciled form.  Every other form still carries its status, its core, its break-even
#: and the SHA-256 of its full record, so nothing is dropped -- it is addressed rather than
#: quoted, and :func:`verify_receipt` rebuilds all of it and compares.
TRANSCRIBED_FORMS = 4

#: The multiplier at which the search reads R2's intervals.  ``1`` means "exactly the
#: interval R2 published"; the certificate then reports for itself how far that could be
#: widened before the obstruction dissolves.
DECLARED_COVERAGE = Fraction(1)


class ReconciliationError(ValueError):
    """Raised on a malformed population, a cap violation, or a discipline violation."""


# ---------------------------------------------------------------------------
# Exact rational helpers
# ---------------------------------------------------------------------------


def _q(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _unq(value: Mapping[str, Any]) -> Fraction:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"numerator", "denominator"}
        or not isinstance(value["numerator"], int)
        or not isinstance(value["denominator"], int)
        or isinstance(value["numerator"], bool)
        or isinstance(value["denominator"], bool)
        or value["denominator"] <= 0
    ):
        raise ReconciliationError("exact rational block malformed")
    return Fraction(value["numerator"], value["denominator"])


def _rendered(value: Fraction) -> dict[str, Any]:
    """An exact rational plus a rendering.  The rendering is never read by a decision."""

    return {"decimal": decimal_string(value, RENDER_PLACES), "exact": _q(value)}


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in body.items() if key != "content_sha256"}
    return {**payload, "content_sha256": canonical_sha256(payload)}


# ---------------------------------------------------------------------------
# The declared law space
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Channel:
    """One universal channel: an exact monomial in the declared covariates.

    ``powers`` is a sorted tuple of ``(covariate, integer exponent)``.  The empty tuple is
    the constant channel.  A channel is a function of *measured* quantities only; object
    identity is not a covariate and cannot enter here, which is the first reason a per-object
    parameter has nowhere to hide.
    """

    name: str
    powers: tuple[tuple[str, int], ...]

    def evaluate(self, covariates: Mapping[str, Fraction]) -> Fraction:
        value = Fraction(1)
        for covariate, exponent in self.powers:
            if covariate not in covariates:
                raise ReconciliationError(f"channel {self.name} needs undeclared {covariate!r}")
            base = covariates[covariate]
            if base == 0 and exponent < 0:
                raise ReconciliationError(
                    f"channel {self.name} is undefined where {covariate} is zero"
                )
            value *= base**exponent
        return value

    def declaration(self) -> dict[str, Any]:
        return {"name": self.name, "powers": [[name, power] for name, power in self.powers]}


def _channel_name(covariate: str, exponent: int) -> str:
    if exponent == 1:
        return covariate
    if exponent < 0:
        return f"{covariate}^m{-exponent}"
    return f"{covariate}^{exponent}"


def channel_catalogue(
    covariate_names: Sequence[str], exponents: Sequence[int]
) -> tuple[Channel, ...]:
    """The declared catalogue: the constant plus one monomial per (covariate, exponent)."""

    if not covariate_names:
        raise ReconciliationError("a law space needs at least one covariate")
    if len(covariate_names) > SEARCH_CAPS["max_covariates"]:
        raise ReconciliationError("covariate count exceeds the declared cap")
    if len(set(covariate_names)) != len(covariate_names):
        raise ReconciliationError("covariate names must be distinct")
    if not exponents or 0 in exponents or len(set(exponents)) != len(exponents):
        raise ReconciliationError("exponents must be distinct and nonzero")
    channels = [Channel(CONSTANT_CHANNEL, ())]
    for covariate in covariate_names:
        for exponent in exponents:
            channels.append(Channel(_channel_name(covariate, exponent), ((covariate, exponent),)))
    return tuple(channels)


@dataclass(frozen=True, slots=True)
class Form:
    """One candidate law shape: for each parameter, the channels its prediction may use."""

    assignment: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def extra_count(self) -> int:
        return sum(len(names) - 1 for _, names in self.assignment)

    @property
    def width(self) -> int:
        return sum(len(names) for _, names in self.assignment)

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return tuple(
            f"{parameter}:{channel}" for parameter, names in self.assignment for channel in names
        )

    def identifier(self) -> str:
        return "|".join(
            f"{parameter}={'+'.join(names)}" for parameter, names in self.assignment
        )

    def declaration(self) -> dict[str, Any]:
        return {
            "assignment": [[parameter, list(names)] for parameter, names in self.assignment],
            "extra_channel_count": self.extra_count,
            "id": self.identifier(),
            "universal_width": self.width,
        }


@dataclass(frozen=True, slots=True)
class LawSpace:
    """A declared, finite, counted space of law shapes."""

    name: str
    covariate_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    channels: tuple[Channel, ...]
    max_extra_channels: int

    def __post_init__(self) -> None:
        if not self.parameter_names:
            raise ReconciliationError("a law space needs at least one parameter")
        if len(self.parameter_names) > SEARCH_CAPS["max_parameters"]:
            raise ReconciliationError("parameter count exceeds the declared cap")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ReconciliationError("parameter names must be distinct")
        if not 0 <= self.max_extra_channels <= SEARCH_CAPS["max_extra_channels_per_parameter"]:
            raise ReconciliationError("extra-channel budget outside the declared cap")
        names = [channel.name for channel in self.channels]
        if names.count(CONSTANT_CHANNEL) != 1 or len(set(names)) != len(names):
            raise ReconciliationError("catalogue must carry exactly one constant and no repeats")

    def channel(self, name: str) -> Channel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise ReconciliationError(f"undeclared channel {name!r}")

    def parameter_options(self) -> tuple[tuple[str, ...], ...]:
        extras = tuple(name for name in (c.name for c in self.channels) if name != CONSTANT_CHANNEL)
        options: list[tuple[str, ...]] = []
        for size in range(self.max_extra_channels + 1):
            for combination in itertools.combinations(extras, size):
                options.append((CONSTANT_CHANNEL, *combination))
        return tuple(options)

    def forms(self) -> tuple[Form, ...]:
        options = self.parameter_options()
        found = [
            Form(tuple(zip(self.parameter_names, choice, strict=True)))
            for choice in itertools.product(options, repeat=len(self.parameter_names))
        ]
        found.sort(key=lambda form: (form.extra_count, form.identifier()))
        if len(found) > SEARCH_CAPS["max_forms"]:
            raise ReconciliationError("declared law space exceeds the declared form cap")
        return tuple(found)

    def coverage_certificate(self) -> dict[str, Any]:
        """A counting argument, not a spot check: enumeration equals the closed-form count."""

        extras = len(self.channels) - 1
        budget = self.max_extra_channels
        parameters = len(self.parameter_names)
        per_parameter = sum(comb(extras, size) for size in range(budget + 1))
        declared = per_parameter**parameters
        forms = self.forms()
        identifiers = {form.identifier() for form in forms}
        return {
            "argument": (
                f"each parameter takes the constant channel plus a subset of the {extras} "
                f"non-constant channels of size at most {budget}, so the count per parameter "
                f"is sum_(s=0..{budget}) C({extras}, s) = {per_parameter}; parameters choose "
                f"independently, so the space has {per_parameter}^{parameters} = {declared} "
                "forms"
            ),
            "declared_cardinality": declared,
            "distinct_traversed": len(identifiers),
            "enumeration_equals_declared": len(forms) == declared == len(identifiers),
            "options_per_parameter": per_parameter,
            "traversed_cardinality": len(forms),
        }

    def declaration(self) -> dict[str, Any]:
        return {
            "channels": [channel.declaration() for channel in self.channels],
            "covariate_names": list(self.covariate_names),
            "max_extra_channels_per_parameter": self.max_extra_channels,
            "name": self.name,
            "object_identity_is_not_a_covariate": True,
            "parameter_names": list(self.parameter_names),
        }


# ---------------------------------------------------------------------------
# The population interface (what R1 and R2 hand over)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PopulationObject:
    """One object's R1 fit reduced to R2's exact per-parameter intervals."""

    label: str
    covariates: tuple[tuple[str, Fraction], ...]
    intervals: tuple[tuple[str, Fraction, Fraction], ...]

    def covariate_map(self) -> dict[str, Fraction]:
        return dict(self.covariates)

    def interval(self, parameter: str) -> tuple[Fraction, Fraction]:
        for name, lower, upper in self.intervals:
            if name == parameter:
                return lower, upper
        raise ReconciliationError(f"object {self.label} declares no interval for {parameter!r}")

    def declaration(self) -> dict[str, Any]:
        return {
            "covariates": [[name, _q(value)] for name, value in self.covariates],
            "intervals": [
                {"lower": _q(lower), "parameter": name, "upper": _q(upper)}
                for name, lower, upper in self.intervals
            ],
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class Population:
    """The exploration half of a sealed split, plus how it was produced.

    ``confirmation`` is the sealed other half.  It is carried so the search can *refuse* to
    look at it: Tier 7's own discipline says a reconciliation reported without the split is
    worthless, so a population with no confirmation set, or one whose objects intrude on it,
    never reaches the linear program.
    """

    name: str
    covariate_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    objects: tuple[PopulationObject, ...]
    confirmation: tuple[str, ...]
    provenance: str
    generator: tuple[tuple[str, Any], ...] = ()

    def labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.objects)

    def declaration(self) -> dict[str, Any]:
        return {
            "confirmation_set": list(self.confirmation),
            "covariate_names": list(self.covariate_names),
            "exploration_set": list(self.labels()),
            "generator": [[key, value] for key, value in self.generator],
            "name": self.name,
            "objects": [item.declaration() for item in self.objects],
            "parameter_names": list(self.parameter_names),
            "provenance": self.provenance,
        }

    def binding(self) -> str:
        return canonical_sha256(self.declaration())


def validate_population(population: Population) -> None:
    """Fail closed on anything ambiguous, and on any breach of the Tier 7 split."""

    if not population.objects:
        raise ReconciliationError("population is empty")
    if len(population.objects) > SEARCH_CAPS["max_objects"]:
        raise ReconciliationError("object count exceeds the declared cap")
    labels = population.labels()
    if len(set(labels)) != len(labels):
        raise ReconciliationError("object labels must be distinct")
    if not population.provenance.strip():
        raise ReconciliationError("a population must declare its provenance")
    if not population.confirmation:
        # Tier 7, stated in the goals document: pass 1 and pass 2 are refit machines, and a
        # population of per-object fits absorbs almost anything.  Without a sealed
        # confirmation set there is nothing left to test the reconciled law on, so the result
        # would be unpublishable by construction.  Refuse before building a single row.
        raise ReconciliationError("no confirmation set declared: the Tier 7 split is missing")
    if len(set(population.confirmation)) != len(population.confirmation):
        raise ReconciliationError("confirmation labels must be distinct")
    intrusion = sorted(set(labels) & set(population.confirmation))
    if intrusion:
        raise ReconciliationError(
            f"exploration/confirmation split violated by: {', '.join(intrusion)}"
        )
    for item in population.objects:
        covariates = item.covariate_map()
        if tuple(sorted(covariates)) != tuple(sorted(population.covariate_names)):
            raise ReconciliationError(f"object {item.label} does not declare the covariate set")
        parameters = tuple(name for name, _, _ in item.intervals)
        if tuple(sorted(parameters)) != tuple(sorted(population.parameter_names)):
            raise ReconciliationError(f"object {item.label} does not declare the parameter set")
        if len(set(parameters)) != len(parameters):
            raise ReconciliationError(f"object {item.label} repeats a parameter")
        for name, lower, upper in item.intervals:
            if lower > upper:
                raise ReconciliationError(f"object {item.label} interval {name} is inverted")


def check_space_matches(population: Population, space: LawSpace) -> None:
    if tuple(sorted(space.covariate_names)) != tuple(sorted(population.covariate_names)):
        raise ReconciliationError("law space and population declare different covariates")
    if tuple(space.parameter_names) != tuple(population.parameter_names):
        raise ReconciliationError("law space and population declare different parameters")


# ---------------------------------------------------------------------------
# From a form and a population to a pooled design with universal width
# ---------------------------------------------------------------------------


def _interval_row(label: str, lower: Fraction, upper: Fraction, source: str) -> MeasuredRow:
    """R2's interval as one declared row.  ``value_interval(1)`` reproduces it exactly."""

    centre = (lower + upper) / 2
    halfwidth = (upper - lower) / 2
    row = MeasuredRow(
        label=label,
        point=Fraction(0),
        point_sigma=Fraction(0),
        point_sigma_rule="exact",
        value=centre,
        value_sigma=halfwidth,
        value_sigma_rule="cited_absolute",
        source=source,
        point_citation=None,
        value_citation="R2 per-object parameter interval",
        point_declared=0,
        value_declared=_q(centre),
    )
    if row.value_interval(DECLARED_COVERAGE) != (lower, upper):
        raise ReconciliationError("interval row does not reproduce the declared R2 interval")
    return row


def build_design(population: Population, space: LawSpace, form: Form) -> Design:
    """One pooled design.  Its width is the universal coefficient count, and only that.

    Reuses :class:`~sigma_theory_compiler.real_data_gravity_confrontation.Design` because the
    structure is identical: one column per universal parameter, rows contributed by every
    object, and the width structurally independent of how many objects contributed.
    """

    check_space_matches(population, space)
    coefficient_names = form.coefficient_names
    if len(coefficient_names) > SEARCH_CAPS["max_universal_width"]:
        raise ReconciliationError("universal width exceeds the declared cap")
    slots: list[tuple[str, str]] = [
        (parameter, channel) for parameter, names in form.assignment for channel in names
    ]
    columns: list[tuple[Fraction, ...]] = []
    rows: list[MeasuredRow] = []
    object_of_row: list[str] = []
    for item in population.objects:
        covariates = item.covariate_map()
        for parameter, names in form.assignment:
            lower, upper = item.interval(parameter)
            design = []
            for slot_parameter, slot_channel in slots:
                if slot_parameter != parameter or slot_channel not in names:
                    design.append(Fraction(0))
                else:
                    design.append(space.channel(slot_channel).evaluate(covariates))
            columns.append(tuple(design))
            rows.append(
                _interval_row(f"{item.label}::{parameter}", lower, upper, population.provenance)
            )
            object_of_row.append(item.label)
    if len(rows) > SEARCH_CAPS["max_rows"]:
        raise ReconciliationError("pooled row count exceeds the declared cap")
    return Design(
        parameter_names=coefficient_names,
        columns=tuple(columns),
        rows=tuple(rows),
        galaxy_of_row=tuple(object_of_row),
    )


def freedom_report(population: Population, space: LawSpace, form: Form) -> dict[str, Any]:
    """The zero-per-object-freedom audit.  Two gates, and they catch different things."""

    design = build_design(population, space, form)
    width = universal_parameter_width(design)
    support: list[dict[str, Any]] = []
    refusals: list[str] = []
    for column, name in enumerate(design.parameter_names):
        objects = sorted(
            {
                design.galaxy_of_row[index]
                for index, line in enumerate(design.columns)
                if line[column] != 0
            }
        )
        support.append({"coefficient": name, "objects": objects, "support_size": len(objects)})
        if len(objects) <= 1 and len(population.objects) > 1:
            refusals.append(REFUSAL_PER_OBJECT_FREEDOM)
    # A knob that arrives once per object shows up here; a knob that arrives once, aimed at a
    # named object, does not -- which is why the support gate above exists as well.
    head = Population(
        name=population.name,
        covariate_names=population.covariate_names,
        parameter_names=population.parameter_names,
        objects=population.objects[:1],
        confirmation=population.confirmation,
        provenance=population.provenance,
        generator=population.generator,
    )
    head_width = universal_parameter_width(build_design(head, space, form))
    if head_width != width:
        refusals.append(REFUSAL_WIDTH_GROWS)
    return {
        "column_support": support,
        "free_parameters_per_object": 0 if not refusals else None,
        "refusals": sorted(set(refusals)),
        "row_count": len(design.rows),
        "single_object_width": head_width,
        "universal_width": width,
        "width_is_population_independent": head_width == width,
    }


# ---------------------------------------------------------------------------
# Deciding one form: FEASIBLE with a re-checked witness, or a Farkas obstruction
# ---------------------------------------------------------------------------


def _predict(column: Sequence[Fraction], point: Sequence[Fraction]) -> Fraction:
    return sum((cell * value for cell, value in zip(column, point, strict=True)), Fraction(0))


def satisfies_every_row(design: Design, point: Sequence[Fraction]) -> list[dict[str, Any]]:
    """Every violated row, exactly.  An empty list is the re-check a witness must pass."""

    violations: list[dict[str, Any]] = []
    for index, row in enumerate(design.rows):
        lower, upper = row.value_interval(DECLARED_COVERAGE)
        predicted = _predict(design.columns[index], point)
        if predicted < lower or predicted > upper:
            violations.append(
                {
                    "interval": {"lower": _q(lower), "upper": _q(upper)},
                    "predicted": _q(predicted),
                    "row": row.label,
                }
            )
    return violations


def coefficient_box(design: Design) -> list[dict[str, Any]]:
    """Exact extremes of each universal constant over the whole feasible polytope."""

    box: list[dict[str, Any]] = []
    width = len(design.parameter_names)
    system = build_system(design.columns, design.rows, DECLARED_COVERAGE)
    for column, name in enumerate(design.parameter_names):
        objective = [Fraction(1) if index == column else Fraction(0) for index in range(width)]
        low_simplex = _Simplex(system.matrix, system.rhs)
        low_simplex.solve_phase_one()
        high_simplex = _Simplex(system.matrix, system.rhs)
        high_simplex.solve_phase_one()
        low = low_simplex.optimize(objective, maximize=False)
        high = high_simplex.optimize(objective, maximize=True)
        box.append(
            {
                "coefficient": name,
                "lower": None if low[0] != "OPTIMAL" else _q(low[1]),
                "unbounded": low[0] != "OPTIMAL" or high[0] != "OPTIMAL",
                "upper": None if high[0] != "OPTIMAL" else _q(high[1]),
            }
        )
    return box


def certificate_break_even(design: Design, terms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The coverage factor at which this Farkas witness stops being a contradiction.

    The argument is the one
    :mod:`~sigma_theory_compiler.real_data_gravity_confrontation` uses on rotation curves,
    carried over unchanged.  Only ``b`` depends on the factor ``k`` that scales R2's
    halfwidths, and affinely: an upper row contributes ``centre + k*halfwidth`` and a lower
    row ``-centre + k*halfwidth``.  So ``lambda^T b(k) = A0 + k*B0`` with
    ``B0 = sum lambda_i halfwidth_i >= 0``, and the *same* nonnegative multipliers keep
    certifying infeasibility for every ``k`` strictly below ``-A0/B0``.
    """

    lookup = {row.label: row for row in design.rows}
    offset = Fraction(0)
    slope = Fraction(0)
    for term in terms:
        row = lookup.get(str(term["row"]))
        if row is None:
            raise ReconciliationError("certificate names a row that is not in the design")
        multiplier = _unq(term["multiplier"])
        if multiplier < 0:
            raise ReconciliationError("Farkas multiplier is negative")
        sign = Fraction(1) if term["bound"] == "upper" else Fraction(-1)
        offset += multiplier * sign * row.value
        slope += multiplier * row.value_sigma
    if slope <= 0:
        return {
            "coverage_independent": True,
            "reading": (
                "every interval in this witness has zero width, so no widening of R2's "
                "intervals dissolves the contradiction at any factor"
            ),
        }
    break_even = -offset / slope
    return {
        "coverage_independent": False,
        "reading": (
            "the same nonnegative multipliers certify infeasibility for every factor "
            "strictly below this value, so scaling every R2 halfwidth by less than it "
            "cannot make this form reconcilable"
        ),
        "value": _rendered(break_even),
    }


def recheck_farkas(design: Design, terms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Re-derive the witness from the design and the receipt's own multipliers.

    This is what makes an ``OBSTRUCTED`` verdict independent of the solver: a reader rebuilds
    ``lambda^T A`` and ``lambda^T b`` from the transcribed numbers and needs nothing else.
    """

    width = len(design.parameter_names)
    index_of: dict[tuple[str, str], int] = {}
    for index, row in enumerate(design.rows):
        index_of[(row.label, "upper")] = 2 * index
        index_of[(row.label, "lower")] = 2 * index + 1
    system = build_system(design.columns, design.rows, DECLARED_COVERAGE)
    combination = [Fraction(0)] * width
    total = Fraction(0)
    nonnegative = True
    for term in terms:
        key = (str(term["row"]), str(term["bound"]))
        if key not in index_of:
            raise ReconciliationError("certificate names a row/bound that is not in the design")
        multiplier = _unq(term["multiplier"])
        if multiplier < 0:
            nonnegative = False
        line = system.matrix[index_of[key]]
        for column in range(width):
            combination[column] += multiplier * line[column]
        total += multiplier * system.rhs[index_of[key]]
    return {
        "annihilates_the_model": all(value == 0 for value in combination),
        "combined_left_hand_side": [_q(value) for value in combination],
        "combined_right_hand_side": _q(total),
        "multipliers_nonnegative": nonnegative,
        "reaches_a_contradiction": total < 0,
        "valid": nonnegative and all(value == 0 for value in combination) and total < 0,
    }


def decide_form(population: Population, space: LawSpace, form: Form) -> dict[str, Any]:
    """One form's verdict: refused, reconciled, or obstructed with an exact certificate."""

    try:
        design = build_design(population, space, form)
    except ReconciliationError as error:
        if "undefined where" not in str(error):
            raise
        return {
            "form": form.declaration(),
            "refusal_detail": str(error),
            "refusals": [REFUSAL_UNDEFINED_CHANNEL],
            "status": "REFUSED",
        }
    freedom = freedom_report(population, space, form)
    system = build_system(design.columns, design.rows, DECLARED_COVERAGE)
    decision = decide_system(system)
    if freedom["refusals"]:
        # Record what the linear program *would* have said.  When that is FEASIBLE the gate
        # has just turned a would-be reconciliation into an obstruction, and the receipt
        # shows it rather than asserting it.
        return {
            "form": form.declaration(),
            "freedom": freedom,
            "linear_program_verdict": decision["verdict"],
            "refusals": freedom["refusals"],
            "status": "REFUSED",
        }
    if decision["verdict"] == FEASIBLE:
        point = [Fraction(value) for value in decision["point"]]
        violations = satisfies_every_row(design, point)
        if violations:
            raise ReconciliationError("a FEASIBLE witness failed the exact per-row re-check")
        return {
            "form": form.declaration(),
            "freedom": freedom,
            "status": FEASIBLE,
            "witness": [
                {"coefficient": name, "value": _q(value)}
                for name, value in zip(design.parameter_names, point, strict=True)
            ],
        }
    witness = decision["witness"]
    recheck = recheck_farkas(design, witness["terms"])
    if not recheck["valid"]:
        raise ReconciliationError("the solver returned a Farkas witness that does not verify")
    core = sorted(
        {
            design.galaxy_of_row[index]
            for index, row in enumerate(design.rows)
            if row.label in set(witness["unreachable_rows"])
        }
    )
    return {
        "break_even": certificate_break_even(design, witness["terms"]),
        "certificate": {
            "kind": witness["kind"],
            "reading": witness["reading"],
            "recheck": recheck,
            "terms": witness["terms"],
            "unreachable_rows": list(witness["unreachable_rows"]),
        },
        "form": form.declaration(),
        "freedom": freedom,
        "irreconcilable_core": core,
        "status": INFEASIBLE,
    }


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def identical_covariate_obstruction(population: Population) -> dict[str, Any] | None:
    """A catalogue-independent obstruction, computed without the linear program.

    If two objects carry identical covariates then every function of covariates -- every law
    in this space and in any other space built on these covariates -- gives them the same
    prediction.  If their R2 intervals for some parameter are disjoint, no such function
    exists.  This does not depend on which channels were declared, and it does not depend on
    a search having failed to find something.
    """

    for left, right in itertools.combinations(population.objects, 2):
        if left.covariate_map() != right.covariate_map():
            continue
        for parameter in population.parameter_names:
            low_left, high_left = left.interval(parameter)
            low_right, high_right = right.interval(parameter)
            if high_left < low_right or high_right < low_left:
                gap = (
                    low_right - high_left if high_left < low_right else low_left - high_right
                )
                return {
                    "gap": _rendered(gap),
                    "holds_for_every_law_space_on_these_covariates": True,
                    "objects": [left.label, right.label],
                    "parameter": parameter,
                    "reading": (
                        "these two objects declare identical covariates, so every function "
                        "of covariates predicts the same value for both; their declared "
                        "intervals for this parameter are disjoint, so no such function "
                        "exists"
                    ),
                }
    return None


def reconcile(population: Population, space: LawSpace) -> dict[str, Any]:
    """Search the declared space for one law that reproduces the whole population."""

    validate_population(population)
    check_space_matches(population, space)
    coverage = space.coverage_certificate()
    if not coverage["enumeration_equals_declared"]:
        raise ReconciliationError("law-space traversal does not match its declared cardinality")
    records = [decide_form(population, space, form) for form in space.forms()]
    feasible = [record for record in records if record["status"] == FEASIBLE]
    infeasible = [record for record in records if record["status"] == INFEASIBLE]
    refused = [record for record in records if record["status"] == "REFUSED"]
    result: dict[str, Any] = {
        "coverage": coverage,
        "declared_coverage_factor": _q(DECLARED_COVERAGE),
        "forms": records,
        "law_space": space.declaration(),
        "population": population.declaration(),
        "population_binding_sha256": population.binding(),
        "reconciled_form_count": len(feasible),
        "refused_form_count": len(refused),
    }
    if feasible:
        best = feasible[0]
        form = _form_from_declaration(best["form"], space)
        design = build_design(population, space, form)
        point = [_unq(entry["value"]) for entry in best["witness"]]
        result["verdict"] = RECONCILED
        result["reconciliation"] = {
            "checked_against_every_row": True,
            "coefficient_box": coefficient_box(design),
            "form": best["form"],
            "free_parameters_per_object": 0,
            "per_object_predictions": [
                {
                    "inside": True,
                    "interval": {
                        "lower": _q(row.value_interval(DECLARED_COVERAGE)[0]),
                        "upper": _q(row.value_interval(DECLARED_COVERAGE)[1]),
                    },
                    "object": design.galaxy_of_row[index],
                    "predicted": _q(_predict(design.columns[index], point)),
                    "row": row.label,
                }
                for index, row in enumerate(design.rows)
            ],
            "universal_width": form.width,
            "witness": best["witness"],
        }
        result["obstruction"] = None
        return result

    result["verdict"] = OBSTRUCTED
    result["reconciliation"] = None
    breaks = [
        (_unq(record["break_even"]["value"]["exact"]), record["form"]["id"])
        for record in infeasible
        if not record["break_even"]["coverage_independent"]
    ]
    cores = [set(record["irreconcilable_core"]) for record in infeasible]
    universal = sorted(set.intersection(*cores)) if cores else []
    space_break: dict[str, Any]
    if not infeasible:
        space_break = {
            "coverage_independent": True,
            "reading": "every form was refused before it could be decided",
        }
    elif not breaks:
        space_break = {
            "coverage_independent": True,
            "reading": (
                "every certificate in the space is coverage-independent, so no widening of "
                "R2's intervals reconciles this population at any factor"
            ),
        }
    else:
        floor_value, floor_form = min(breaks)
        space_break = {
            "coverage_independent": False,
            "reading": (
                "no law in the declared space can reconcile this population until every R2 "
                "halfwidth is scaled by at least this factor; below it every enumerated form "
                "carries a nonnegative combination of R2's own inequalities summing to a "
                "contradiction"
            ),
            "tightest_form": floor_form,
            "value": _rendered(floor_value),
        }
    result["obstruction"] = {
        "declared_form_count": coverage["declared_cardinality"],
        "identical_covariate_argument": identical_covariate_obstruction(population),
        "infeasible_form_count": len(infeasible),
        "kind": "farkas_exhaustion_over_the_declared_law_space",
        "reading": (
            "every form in the enumerated space is either refused for retaining per-object "
            "freedom or carries a re-verified Farkas witness; the space is exhausted by the "
            "counting argument in coverage, not by sampling"
        ),
        "refused_form_count": len(refused),
        "space_break_even_coverage": space_break,
        "universally_irreconcilable_objects": universal,
    }
    return result


def _compact_form_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """One form reduced to what a reader needs plus the digest of everything else."""

    break_even = record.get("break_even") or {}
    return {
        "break_even": None if not break_even else (
            {"coverage_independent": True}
            if break_even.get("coverage_independent")
            else {"coverage_independent": False, "value": break_even["value"]}
        ),
        "certificate_term_count": len(record.get("certificate", {}).get("terms", [])),
        "form_id": record["form"]["id"],
        "irreconcilable_core": record.get("irreconcilable_core", []),
        "record_sha256": canonical_sha256(record),
        "refusals": record.get("refusals", []),
        "status": record["status"],
        "universal_width": record["form"]["universal_width"],
    }


def transcribe(result: Mapping[str, Any], limit: int = TRANSCRIBED_FORMS) -> dict[str, Any]:
    """Receipt shape: every form addressed by digest, a declared few quoted in full."""

    if limit < 1:
        raise ReconciliationError("the transcription budget must be at least one form")
    records = list(result["forms"])
    chosen = [record["form"]["id"] for record in records[:limit]]
    obstruction = result.get("obstruction")
    if obstruction:
        tightest = obstruction["space_break_even_coverage"].get("tightest_form")
        if tightest is not None and tightest not in chosen:
            chosen.append(tightest)
    reconciliation = result.get("reconciliation")
    if reconciliation and reconciliation["form"]["id"] not in chosen:
        chosen.append(reconciliation["form"]["id"])
    smuggled = [
        record["form"]["id"]
        for record in records
        if record["status"] == "REFUSED" and record.get("linear_program_verdict") == FEASIBLE
    ]
    if smuggled and smuggled[0] not in chosen:
        chosen.append(smuggled[0])
    order = {identifier: index for index, identifier in enumerate(chosen)}
    return {
        **{key: value for key, value in result.items() if key != "forms"},
        "forms": [_compact_form_record(record) for record in records],
        "transcribed_forms": sorted(
            (record for record in records if record["form"]["id"] in order),
            key=lambda record: order[record["form"]["id"]],
        ),
        "transcription_rule": (
            f"the first {limit} forms in Occam order, plus the tightest-break-even form, the "
            "reconciled form when one exists, and the first form the freedom gate refused "
            "after the linear program called it feasible; every other form carries the "
            "SHA-256 of its full record"
        ),
    }


def _form_from_declaration(declaration: Mapping[str, Any], space: LawSpace) -> Form:
    assignment = tuple(
        (str(parameter), tuple(str(name) for name in names))
        for parameter, names in declaration["assignment"]
    )
    form = Form(assignment)
    if form.identifier() != declaration["id"]:
        raise ReconciliationError("form declaration does not match its identifier")
    for _, names in assignment:
        if names[0] != CONSTANT_CHANNEL:
            raise ReconciliationError("every form carries the constant channel first")
        for name in names:
            space.channel(name)
    return form


# ---------------------------------------------------------------------------
# Synthetic populations (the controls)
# ---------------------------------------------------------------------------

#: A deterministic exact integer stream.  No float, no ``random`` module, no platform
#: dependence: the population is a function of its declared seed and nothing else.
_LCG_MULTIPLIER = 1103515245
_LCG_INCREMENT = 12345
_LCG_MODULUS = 2**31

#: The law the ``universal`` generator actually used.  The positive control's whole job is to
#: get this back, so it is declared here as data and compared coefficient by coefficient.
TRUE_LAW: tuple[tuple[str, tuple[tuple[str, Fraction], ...]], ...] = (
    ("alpha", ((CONSTANT_CHANNEL, Fraction(3)), ("x", Fraction(2)))),
    ("beta", ((CONSTANT_CHANNEL, Fraction(5)), ("y", Fraction(-4)))),
)

TRUE_FORM = Form((("alpha", (CONSTANT_CHANNEL, "x")), ("beta", (CONSTANT_CHANNEL, "y"))))

#: Halfwidth of every synthetic R2 interval.
DEFAULT_HALFWIDTH = Fraction(1, 64)

GENERATOR_KINDS = ("universal", "twin", "scrambled")


def _lcg(seed: int, count: int) -> tuple[int, ...]:
    state = seed % _LCG_MODULUS
    stream: list[int] = []
    for _ in range(count):
        state = (_LCG_MULTIPLIER * state + _LCG_INCREMENT) % _LCG_MODULUS
        stream.append(state)
    return tuple(stream)


def _true_value(parameter: str, covariates: Mapping[str, Fraction]) -> Fraction:
    for name, terms in TRUE_LAW:
        if name != parameter:
            continue
        total = Fraction(0)
        for channel, coefficient in terms:
            total += coefficient * (
                Fraction(1) if channel == CONSTANT_CHANNEL else covariates[channel]
            )
        return total
    raise ReconciliationError(f"the declared true law has no parameter {parameter!r}")


def synthetic_population(
    *,
    kind: str,
    object_count: int,
    seed: int,
    halfwidth: Fraction = DEFAULT_HALFWIDTH,
    tagged: bool = False,
) -> Population:
    """Build a population whose answer is known before the search runs.

    ``universal``
        every object's parameters come from :data:`TRUE_LAW` plus an offset strictly inside
        half the interval halfwidth, so the true law lies inside every declared interval and
        a correct search must return it.
    ``twin``
        the first two objects carry identical covariates and disjoint intervals for
        ``alpha``.  No function of covariates reconciles them, so this is a genuinely
        per-object population and the only correct answer is an obstruction.
    ``scrambled``
        the parameters carry per-object offsets many halfwidths wide, drawn from the exact
        integer stream and tracked by nothing in the catalogue.

    ``tagged`` adds a covariate that is ``1`` on exactly one object and ``0`` elsewhere: a
    per-object knob wearing a covariate's clothes.  It exists so the freedom gate has
    something real to refuse.
    """

    if kind not in GENERATOR_KINDS:
        raise ReconciliationError(f"undeclared generator kind {kind!r}")
    if object_count < 2 or object_count > SEARCH_CAPS["max_objects"]:
        raise ReconciliationError("object count outside the declared range")
    if halfwidth <= 0:
        raise ReconciliationError("halfwidth must be strictly positive")
    stream = _lcg(seed, 4 * object_count)
    covariate_names = ("tag", "x", "y") if tagged else ("x", "y")
    objects: list[PopulationObject] = []
    for index in range(object_count):
        raw_x, raw_y, raw_alpha, raw_beta = stream[4 * index : 4 * index + 4]
        x_value = Fraction(1) + Fraction(raw_x % 97, 8)
        y_value = Fraction(1) + Fraction(raw_y % 89, 7)
        if kind == "twin" and index == 1:
            previous = objects[0].covariate_map()
            x_value, y_value = previous["x"], previous["y"]
        covariates = {"x": x_value, "y": y_value}
        if tagged:
            covariates["tag"] = Fraction(1) if index == 1 else Fraction(0)
        intervals: list[tuple[str, Fraction, Fraction]] = []
        for parameter, raw in (("alpha", raw_alpha), ("beta", raw_beta)):
            centre = _true_value(parameter, covariates)
            if kind == "scrambled":
                centre += halfwidth * Fraction((raw % 41) - 20, 1)
            else:
                centre += halfwidth * Fraction((raw % 21) - 10, 20)
            if kind == "twin" and index == 1 and parameter == "alpha":
                centre += 4 * halfwidth
            intervals.append((parameter, centre - halfwidth, centre + halfwidth))
        objects.append(
            PopulationObject(
                label=f"OBJ-{index:03d}",
                covariates=tuple(sorted((name, value) for name, value in covariates.items())),
                intervals=tuple(intervals),
            )
        )
    return Population(
        name=f"{kind}-{object_count}{'-tagged' if tagged else ''}",
        covariate_names=covariate_names,
        parameter_names=("alpha", "beta"),
        objects=tuple(objects),
        confirmation=tuple(f"HOLD-{index:03d}" for index in range(4)),
        provenance=(
            "synthetic R1/R2 population generated by "
            "sigma_theory_compiler.reconciliation_search.synthetic_population"
        ),
        generator=(
            ("halfwidth", [halfwidth.numerator, halfwidth.denominator]),
            ("kind", kind),
            ("module", SOURCE_PATH),
            ("object_count", object_count),
            ("seed", seed),
            ("tagged", tagged),
        ),
    )


def scale_population_intervals(population: Population, factor: Fraction) -> Population:
    """Every R2 interval widened about its own centre by an exact factor.

    This is what turns ``space_break_even_coverage`` from a number in a receipt into a
    checkable prediction: below the break-even the obstruction must survive this operation,
    because the certificate says the *same* multipliers still work.  The result carries no
    generator, because a widened population is no longer the one R2 published.
    """

    if factor <= 0:
        raise ReconciliationError("interval scale factor must be strictly positive")
    objects = []
    for item in population.objects:
        intervals = []
        for name, lower, upper in item.intervals:
            centre = (lower + upper) / 2
            halfwidth = (upper - lower) / 2 * factor
            intervals.append((name, centre - halfwidth, centre + halfwidth))
        objects.append(
            PopulationObject(
                label=item.label, covariates=item.covariates, intervals=tuple(intervals)
            )
        )
    return Population(
        name=f"{population.name}-scaled-{factor.numerator}-{factor.denominator}",
        covariate_names=population.covariate_names,
        parameter_names=population.parameter_names,
        objects=tuple(objects),
        confirmation=population.confirmation,
        provenance=f"{population.provenance} (every R2 halfwidth scaled by {factor})",
        generator=(),
    )


def regenerate_population(generator: Sequence[Sequence[Any]]) -> Population:
    """Rebuild a population from a receipt's declared generator block, exactly."""

    fields = {str(key): value for key, value in generator}
    if fields.get("module") != SOURCE_PATH:
        raise ReconciliationError("population generator is not this module")
    halfwidth = fields["halfwidth"]
    return synthetic_population(
        kind=str(fields["kind"]),
        object_count=int(fields["object_count"]),
        seed=int(fields["seed"]),
        halfwidth=Fraction(int(halfwidth[0]), int(halfwidth[1])),
        tagged=bool(fields["tagged"]),
    )


def default_space(covariate_names: Sequence[str]) -> LawSpace:
    """The declared law space: constant plus linear and quadratic monomials, budget two."""

    return LawSpace(
        name="monomial-degree-2-budget-2",
        covariate_names=tuple(covariate_names),
        parameter_names=("alpha", "beta"),
        channels=channel_catalogue(sorted(covariate_names), (1, 2)),
        max_extra_channels=2,
    )


def tagged_space() -> LawSpace:
    """A smaller space carrying the smuggled ``tag`` covariate, so the gate has a target."""

    return LawSpace(
        name="monomial-degree-1-budget-2-with-tag",
        covariate_names=("tag", "x", "y"),
        parameter_names=("alpha", "beta"),
        channels=channel_catalogue(("tag", "x", "y"), (1,)),
        max_extra_channels=2,
    )


# ---------------------------------------------------------------------------
# Controls: every positive answer has a control that must fail
# ---------------------------------------------------------------------------


def _true_law_point(form: Form) -> list[Fraction] | None:
    """The generating coefficient vector expressed in this form's coefficient order."""

    values: list[Fraction] = []
    truth = {name: dict(terms) for name, terms in TRUE_LAW}
    for parameter, names in form.assignment:
        if parameter not in truth:
            return None
        for channel in names:
            values.append(truth[parameter].get(channel, Fraction(0)))
    return values


def reconciliation_controls() -> dict[str, Any]:
    """The battery.  Positives must be accepted; every probe must be rejected."""

    honest: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []

    universal = synthetic_population(kind="universal", object_count=8, seed=20260819)
    universal_space = default_space(universal.covariate_names)
    universal_result = reconcile(universal, universal_space)

    twin = synthetic_population(kind="twin", object_count=6, seed=20260819)
    twin_space = default_space(twin.covariate_names)
    twin_result = reconcile(twin, twin_space)

    scrambled = synthetic_population(kind="scrambled", object_count=8, seed=20260819)
    scrambled_space = default_space(scrambled.covariate_names)
    scrambled_result = reconcile(scrambled, scrambled_space)

    smuggled = synthetic_population(kind="twin", object_count=6, seed=20260819, tagged=True)
    smuggled_space = tagged_space()
    smuggled_result = reconcile(smuggled, smuggled_space)

    # ---- honest: a population built from a universal law is reconciled back to it --------
    recovered_form = universal_result["reconciliation"]["form"]["id"] if (
        universal_result["verdict"] == RECONCILED
    ) else None
    truth_point = _true_law_point(TRUE_FORM)
    truth_design = build_design(universal, universal_space, TRUE_FORM)
    truth_violations = satisfies_every_row(truth_design, truth_point or [])
    box = (
        universal_result["reconciliation"]["coefficient_box"]
        if universal_result["verdict"] == RECONCILED
        else []
    )
    truth_in_box = bool(box) and recovered_form == TRUE_FORM.identifier() and all(
        entry["lower"] is not None
        and entry["upper"] is not None
        and _unq(entry["lower"]) <= value <= _unq(entry["upper"])
        for entry, value in zip(box, truth_point or [], strict=False)
    )
    honest.append(
        {
            "accepted": (
                universal_result["verdict"] == RECONCILED
                and recovered_form == TRUE_FORM.identifier()
                and truth_violations == []
                and truth_in_box
                and universal_result["reconciliation"]["free_parameters_per_object"] == 0
            ),
            "detail": {
                "generating_form": TRUE_FORM.identifier(),
                "generating_law_violates_no_row": truth_violations == [],
                "generating_law_inside_the_coefficient_box": truth_in_box,
                "recovered_form": recovered_form,
            },
            "name": "universal_population_is_reconciled_back_to_its_own_law",
        }
    )

    # ---- honest: per-object populations return an obstruction, not a reconciliation ------
    for label, outcome in (("twin", twin_result), ("scrambled", scrambled_result)):
        obstruction = outcome["obstruction"]
        certificates = [row for row in outcome["forms"] if row["status"] == INFEASIBLE]
        honest.append(
            {
                "accepted": (
                    outcome["verdict"] == OBSTRUCTED
                    and outcome["reconciliation"] is None
                    and outcome["reconciled_form_count"] == 0
                    and bool(certificates)
                    and all(row["certificate"]["recheck"]["valid"] for row in certificates)
                    and obstruction["infeasible_form_count"] + obstruction["refused_form_count"]
                    == obstruction["declared_form_count"]
                ),
                "detail": {
                    "certificate_count": len(certificates),
                    "declared_form_count": obstruction["declared_form_count"],
                    "space_break_even": obstruction["space_break_even_coverage"],
                },
                "name": f"{label}_population_returns_an_exact_obstruction",
            }
        )

    # ---- honest: the identical-covariate argument agrees with the program ----------------
    identical = identical_covariate_obstruction(twin)
    honest.append(
        {
            "accepted": identical is not None and twin_result["verdict"] == OBSTRUCTED,
            "detail": identical,
            "name": "identical_covariate_argument_needs_no_search",
        }
    )

    # ---- probe: the smuggled per-object knob is refused, and it *would* have fitted ------
    refused = [row for row in smuggled_result["forms"] if row["status"] == "REFUSED"]
    would_have_fitted = [
        row for row in refused if row.get("linear_program_verdict") == FEASIBLE
    ]
    probes.append(
        {
            "detail": {
                "refused_form_count": len(refused),
                "refused_forms_the_program_called_feasible": len(would_have_fitted),
                "verdict": smuggled_result["verdict"],
            },
            "name": "smuggled_per_object_knob_is_refused_not_reconciled",
            "rejected": (
                smuggled_result["verdict"] == OBSTRUCTED
                and bool(would_have_fitted)
                and all(
                    REFUSAL_PER_OBJECT_FREEDOM in row["refusals"] for row in would_have_fitted
                )
            ),
        }
    )

    # ---- probe: a forged witness must fail the exact per-row re-check --------------------
    forged_design = build_design(twin, twin_space, TRUE_FORM)
    forged_point = _true_law_point(TRUE_FORM) or []
    probes.append(
        {
            "detail": {"violated_rows": len(satisfies_every_row(forged_design, forged_point))},
            "name": "forged_reconciliation_of_a_per_object_population",
            "rejected": satisfies_every_row(forged_design, forged_point) != [],
        }
    )

    # ---- probe: a witness nudged off the polytope must fail ------------------------------
    good_design = build_design(universal, universal_space, TRUE_FORM)
    good_point = _true_law_point(TRUE_FORM) or []
    nudged = list(good_point)
    nudged[0] = nudged[0] + 1
    probes.append(
        {
            "detail": {"violated_rows": len(satisfies_every_row(good_design, nudged))},
            "name": "nudged_witness_leaves_the_declared_intervals",
            "rejected": satisfies_every_row(good_design, nudged) != [],
        }
    )

    # ---- honest: a Farkas witness is a cone, so a positive rescaling still certifies -----
    # This one is here so the battery cannot pass by rejecting everything put in front of it.
    first_certificate = next(row for row in twin_result["forms"] if row["status"] == INFEASIBLE)
    certificate_form = _form_from_declaration(first_certificate["form"], twin_space)
    design = build_design(twin, twin_space, certificate_form)
    terms = first_certificate["certificate"]["terms"]
    if len(terms) < 2:
        raise ReconciliationError("the tamper probes need a witness with at least two terms")
    halved = [{**term, "multiplier": _q(_unq(term["multiplier"]) / 2)} for term in terms]
    honest.append(
        {
            "accepted": recheck_farkas(design, halved)["valid"],
            "detail": {
                "form": first_certificate["form"]["id"],
                "reading": (
                    "lambda >= 0 with lambda^T A = 0 and lambda^T b < 0 is closed under "
                    "multiplication by a positive rational, so halving every multiplier "
                    "must still verify"
                ),
            },
            "name": "positively_scaled_farkas_witness_still_certifies",
        }
    )

    # ---- probe: tampered Farkas multipliers must not verify ------------------------------
    negated = [
        {**term, "multiplier": _q(-_unq(term["multiplier"]))} if index == 0 else term
        for index, term in enumerate(terms)
    ]
    # Doubling exactly one multiplier shifts lambda^T A by that row's design vector, which is
    # never the zero vector because every form carries the constant channel.  So this probe
    # breaks annihilation by construction rather than by luck.
    unbalanced = [
        {**term, "multiplier": _q(_unq(term["multiplier"]) * 2)} if index == 0 else term
        for index, term in enumerate(terms)
    ]
    probes.append(
        {
            "detail": recheck_farkas(design, negated),
            "name": "negated_farkas_multiplier_is_not_a_certificate",
            "rejected": not recheck_farkas(design, negated)["valid"],
        }
    )
    probes.append(
        {
            "detail": recheck_farkas(design, unbalanced),
            "name": "unbalanced_farkas_witness_no_longer_annihilates_the_model",
            "rejected": not recheck_farkas(design, unbalanced)["valid"],
        }
    )

    # ---- probe: the Tier 7 split is not optional -----------------------------------------
    for name, mutation in (
        ("population_without_a_confirmation_set", {"confirmation": ()}),
        (
            "population_that_intrudes_on_the_confirmation_set",
            {"confirmation": (universal.labels()[0], "HOLD-000")},
        ),
    ):
        broken = Population(
            name=universal.name,
            covariate_names=universal.covariate_names,
            parameter_names=universal.parameter_names,
            objects=universal.objects,
            confirmation=mutation["confirmation"],
            provenance=universal.provenance,
            generator=universal.generator,
        )
        try:
            reconcile(broken, universal_space)
            rejected, detail = False, "accepted"
        except ReconciliationError as error:
            rejected, detail = True, str(error)
        probes.append({"detail": detail, "name": name, "rejected": rejected})

    return {
        "all_honest_accepted": all(row["accepted"] for row in honest),
        "all_probes_rejected": all(row["rejected"] for row in probes),
        "honest": honest,
        "probes": probes,
    }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _scenario(population: Population, space: LawSpace) -> dict[str, Any]:
    return {
        "law_space_name": space.name,
        "population_name": population.name,
        "result": transcribe(reconcile(population, space)),
    }


def build_receipt() -> dict[str, Any]:
    """A sealed receipt: three populations, one law space each, plus the control battery."""

    universal = synthetic_population(kind="universal", object_count=8, seed=20260819)
    twin = synthetic_population(kind="twin", object_count=6, seed=20260819)
    scrambled = synthetic_population(kind="scrambled", object_count=8, seed=20260819)
    smuggled = synthetic_population(kind="twin", object_count=6, seed=20260819, tagged=True)
    scenarios = [
        _scenario(universal, default_space(universal.covariate_names)),
        _scenario(twin, default_space(twin.covariate_names)),
        _scenario(scrambled, default_space(scrambled.covariate_names)),
        _scenario(smuggled, tagged_space()),
    ]
    controls = reconciliation_controls()
    body = {
        "claims": {
            "exact_arithmetic_only": True,
            "may_be_cited_as_confirmation": False,
            "observational_data_opened": False,
            "sealed_no_refit_trial": False,
            "synthetic_populations_only": True,
            "tier": "R4",
            "trial_type": "exploratory",
        },
        "controls": controls,
        "decision": (
            "RECONCILIATION_SEARCH_CERTIFIED"
            if controls["all_honest_accepted"] and controls["all_probes_rejected"]
            else "CONTROLS_FAILED"
        ),
        "scenarios": scenarios,
        "schema_version": RESULT_SCHEMA,
        "source_path": SOURCE_PATH,
        "test_path": TEST_PATH,
    }
    return _seal(body)


def _verify_scenario(scenario: Mapping[str, Any], name: str) -> list[str]:
    """Re-derive one scenario from its own declarations.  Returns findings, never raises."""

    findings: list[str] = []
    result = scenario["result"]
    try:
        population = regenerate_population(result["population"]["generator"])
    except (ReconciliationError, KeyError, TypeError, ValueError) as error:
        return [f"{name}: population is not regenerable: {error}"]
    if population.binding() != result.get("population_binding_sha256"):
        return [f"{name}: declared population does not match its generator"]
    if population.declaration() != result["population"]:
        # The binding can only catch a *stated* sha that moved.  This catches the interval
        # that was widened after the fact while the sha beside it was left alone.
        return [f"{name}: declared population does not match its generator"]
    space = (
        tagged_space()
        if result["law_space"]["name"] == tagged_space().name
        else default_space(population.covariate_names)
    )
    if space.declaration() != result["law_space"]:
        return [f"{name}: law-space declaration drift"]
    compact = {row["form_id"]: row for row in result["forms"]}
    for record in result["transcribed_forms"]:
        form = _form_from_declaration(record["form"], space)
        identifier = record["form"]["id"]
        if compact.get(identifier) != _compact_form_record(record):
            findings.append(f"{name}/{identifier}: compact row does not address this record")
        if record["status"] != INFEASIBLE:
            continue
        design = build_design(population, space, form)
        recheck = recheck_farkas(design, record["certificate"]["terms"])
        if not recheck["valid"]:
            findings.append(f"{name}/{identifier}: Farkas witness does not verify")
        if recheck != record["certificate"]["recheck"]:
            findings.append(f"{name}/{identifier}: transcribed recheck drift")
        if certificate_break_even(design, record["certificate"]["terms"]) != record["break_even"]:
            findings.append(f"{name}/{identifier}: break-even was not recomputed")
    verdict = result["verdict"]
    reconciliation = result.get("reconciliation")
    if verdict == RECONCILED:
        if not reconciliation:
            return [*findings, f"{name}: RECONCILED with no reconciliation block"]
        form = _form_from_declaration(reconciliation["form"], space)
        design = build_design(population, space, form)
        point = [_unq(entry["value"]) for entry in reconciliation["witness"]]
        violations = satisfies_every_row(design, point)
        if violations:
            findings.append(f"{name}: reconciled witness misses {len(violations)} rows")
        if freedom_report(population, space, form)["refusals"]:
            findings.append(f"{name}: reconciled form retains per-object freedom")
    elif verdict == OBSTRUCTED:
        if reconciliation is not None:
            findings.append(f"{name}: OBSTRUCTED but a reconciliation block is present")
    else:
        findings.append(f"{name}: undeclared verdict {verdict!r}")
    return findings


def verify_receipt(
    receipt: Mapping[str, Any], *, regenerated: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Two jobs: re-derive every claim, then rebuild the whole body and require a match.

    The first job kills a receipt that is *wrong* -- a witness that misses a row, a Farkas
    combination that does not annihilate the model, a break-even that was not computed from
    the multipliers it quotes.  The second kills a receipt that is merely *misleading*: a
    population whose intervals were widened after the fact, a form identifier that names a
    shape the search never decided, a verdict edited without resealing.

    ``regenerated`` accepts an already-built body for the second job.  Passing one changes no
    check; it exists so a caller that verifies several receipts does not rebuild the same
    body once per receipt.
    """

    findings: list[str] = []
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if canonical_sha256(body) != receipt.get("content_sha256"):
        findings.append("seal does not match the body")
    if receipt.get("schema_version") != RESULT_SCHEMA:
        findings.append("schema version drift")

    for scenario in receipt.get("scenarios", []):
        name = scenario.get("population_name", "<unnamed>")
        # A verifier is fed hostile input by definition, so a malformed receipt must come
        # back as a finding rather than as a traceback.
        try:
            findings.extend(_verify_scenario(scenario, name))
        except (ReconciliationError, KeyError, TypeError, ValueError, IndexError) as error:
            findings.append(f"{name}: receipt is malformed: {type(error).__name__}: {error}")

    fresh = build_receipt() if regenerated is None else regenerated
    matches = {
        key: fresh.get(key) == receipt.get(key)
        for key in ("claims", "controls", "decision", "scenarios", "schema_version")
    }
    for key, ok in matches.items():
        if not ok:
            findings.append(f"regeneration mismatch in {key!r}")
    return {"findings": findings, "regeneration": matches, "verified": not findings}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="write the sealed receipt here")
    parser.add_argument(
        "--verify", default=None, help="verify an existing receipt instead of writing one"
    )
    arguments = parser.parse_args(argv)
    if arguments.verify:
        loaded = json.loads(Path(arguments.verify).read_text(encoding="utf-8"))
        report = verify_receipt(loaded)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if report["verified"] else 1
    receipt = build_receipt()
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False)
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0 if receipt["decision"] == "RECONCILIATION_SEARCH_CERTIFIED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONSTANT_CHANNEL",
    "DECLARED_COVERAGE",
    "DEFAULT_HALFWIDTH",
    "GENERATOR_KINDS",
    "OBSTRUCTED",
    "RECEIPT_PATH",
    "RECONCILED",
    "REFUSAL_PER_OBJECT_FREEDOM",
    "REFUSAL_REASONS",
    "REFUSAL_UNDEFINED_CHANNEL",
    "REFUSAL_WIDTH_GROWS",
    "RESULT_SCHEMA",
    "SEARCH_CAPS",
    "SOURCE_PATH",
    "TEST_PATH",
    "TRANSCRIBED_FORMS",
    "TRUE_FORM",
    "TRUE_LAW",
    "Channel",
    "Form",
    "LawSpace",
    "Population",
    "PopulationObject",
    "ReconciliationError",
    "build_design",
    "build_receipt",
    "certificate_break_even",
    "channel_catalogue",
    "coefficient_box",
    "decide_form",
    "default_space",
    "freedom_report",
    "identical_covariate_obstruction",
    "main",
    "recheck_farkas",
    "reconcile",
    "reconciliation_controls",
    "regenerate_population",
    "satisfies_every_row",
    "scale_population_intervals",
    "synthetic_population",
    "tagged_space",
    "transcribe",
    "validate_population",
    "verify_receipt",
]
