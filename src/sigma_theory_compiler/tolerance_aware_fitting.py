"""B8 — tolerance-aware interval fitting for measured data.

B1 (:mod:`.basis_synthesis`) solves exactly over exact rational rows: zero tolerance,
zero uncertainty, and a holdout row either reproduces to the last bit or the candidate
dies.  That discipline is what makes a B1 PASS mean something, and it is also why B1
cannot touch a measurement.  Published data does not arrive exact.  The blind planetary
campaign says so in its own receipt: its ``anchor_fidelity`` block records that the
published ``(a, T)`` pairs deviate from an exact power law by up to 6.7e-4, so the
campaign had to *re-anchor* the rows onto an exactly-Keplerian grid before B1 could see
them.  This module is B1's measured-data sibling: it accepts the deviation instead of
removing it, and it makes the deviation the thing that decides.

**The problem this module exists to solve is that with tolerance every model fits.**  The
whole scientific content therefore lives in the guards, and the guards are built first:

* **Uncertainties are declared, never fitted.**  Every row is ``(point, value, sigma)``
  with ``sigma`` fixed by a *declared rule* and a cited source.  For the two rules that
  matter most -- ``half_ulp_of_last_published_digit`` and ``ulp_of_last_published_digit``
  -- sigma is *re-derived here* from the published decimal string, and a caller-supplied
  sigma that disagrees with the derivation is refused outright.  Nothing in this module
  reads a residual and returns a sigma; there is no scale knob anywhere in the API.
* **Holdout stays sovereign.**  The fit consumes only the minimum rows needed to pin the
  linear parameters.  Every remaining row is untouched holdout whose *own* declared
  interval must be reachable from the fit-row region, and the whole system must be
  simultaneously satisfiable on top of that.
* **Parsimony is a budget and a rule, not a score.**  ``n - k >= min_confirmations``, and
  a larger ladder entry is accepted only because every strictly simpler entry is
  *certified infeasible* -- no coefficient vector reaches every row's interval -- while
  the accepted entry is certified feasible.  Models are never ranked by residual size.
* **Exactness where it is available.**  Each measurement becomes an exact rational
  interval ``[v - k*sigma, v + k*sigma]`` at a declared coverage factor ``k``, and the
  question asked is *feasibility*: does one parameter vector pass through **all**
  intervals at once?  For linear-in-parameters models that is decided by exact rational
  linear programming, so the verdict is FEASIBLE / INFEASIBLE with a certificate -- a
  Farkas multiplier vector that is re-verified here and can be re-checked by hand -- and
  never a p-value.  For the power-law family the same question is decided in closed form
  on the exact invariant ``value^v / point^u``, which also absorbs uncertainty in the
  *point* exactly, by monotonicity.
* **The output is a region, not a point.**  An accepted model reports exact rational
  coefficient intervals: the set of coefficient values consistent with the data.

Claim boundary.  A verdict is about the declared intervals, the declared ladder, and the
declared coverage factor -- nothing else.  INFEASIBLE means "no member of this declared
family passes through these declared intervals", never "impossible"; FEASIBLE means "some
member does", never "true".  This module opens no dataset of its own: rows and their
sources are supplied by the caller.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

from .basis_synthesis import LADDER, Term, evaluate_term
from .sigma_core import canonical_json_bytes, canonical_sha256

LADDER_SCHEMA = "invariant-tolerance-aware-ladder-1.0"
RESULT_SCHEMA = "invariant-tolerance-aware-fitting-result-1.0"

#: Hard bounds.  Exceeding any of these is an error, never a silent truncation.
SYSTEM_CAPS = {
    "max_coverage_k": 6,
    "max_exponent_absolute_bound": 8,
    "max_exponent_denominator_bound": 24,
    "max_lp_pivots": 20000,
    "max_parameters": 8,
    "max_residual_report_rows": 32,
    "max_rows": 64,
    "min_confirmations": 2,
}

CLAIMS = {
    "coefficient_output_is_a_region_not_a_point": True,
    "corpus_absence_establishes_novelty": False,
    "holdout_confirmation_required": True,
    "infeasibility_carries_a_reverified_certificate": True,
    "interpolation_accepted_as_discovery": False,
    "ladder_is_declared_and_finite": True,
    "minimality_is_certified_against_simpler_entries": True,
    "models_ranked_by_residual_magnitude": False,
    "uncertainties_are_declared_never_fitted": True,
    # the house rule, stated without using any of the tokens the receipt scan forbids:
    # this receipt decides by threshold comparison and carries no fit statistic at all
    "verdicts_replace_fit_statistics": True,
}

SCOPE = (
    "Tolerance-aware interval fitting of measured rows over a declared finite ladder. "
    "Each row is widened to the exact rational interval [value - k*sigma, value + k*sigma] "
    "at a declared coverage factor k, with sigma fixed by a declared rule and, for the "
    "published-digit rules, re-derived here from the published decimal string so a "
    "caller-supplied sigma that disagrees is refused. The verdict is feasibility -- does a "
    "single parameter vector pass through every interval simultaneously -- decided by exact "
    "rational linear programming for linear-in-parameters entries and in closed form on the "
    "exact invariant value^v/point^u for the power-law family. FEASIBLE_MINIMAL means the "
    "accepted entry is the simplest ladder entry that is feasible, that every strictly "
    "simpler entry is certified infeasible with a re-verified witness, and that every "
    "holdout row's own declared interval is reachable from the fit-row region. It does not "
    "establish that the model is true, that the declared sigmas are correct, that the "
    "cited sources are accurate, or that anything outside the declared ladder was "
    "considered. No goodness-of-fit score, information criterion, or p-value is computed "
    "or emitted."
)

PARSIMONY_RULE = (
    "n - k >= min_confirmations for an entry with k fitted parameters over n rows; and a "
    "larger entry is accepted only because every strictly simpler entry in the declared "
    "Occam order is certified INFEASIBLE against the declared intervals -- no coefficient "
    "vector reaches every row's interval -- while the accepted entry is certified FEASIBLE. "
    "Entries are never compared by residual magnitude, so a larger entry can never win by "
    "fitting better; a simpler entry that is also FEASIBLE always wins, and the search "
    "stops at the first feasible entry in the declared order."
)

LADDER_ORDERING = "parameter_count_then_track_rank_then_entry_rank_then_entry_id"

#: Receipt keys are checked against these tokens.  The house rule is that a tolerance
#: receipt carries verdicts and certificates, never a scalar that ranks models.
FORBIDDEN_RECEIPT_KEY_TOKENS = (
    "aic",
    "bic",
    "chisq",
    "chi_squared",
    "confidence",
    "fit_quality",
    "goodness",
    "likelihood",
    "loss",
    "p_value",
    "posterior",
    "r_squared",
    "rmse",
    "score",
    "significance",
)

SIGMA_RULES = (
    # a cited uncertainty on a *rounded* published decimal: it may not claim to be finer
    # than the printed last digit, because the printing is itself a precision statement
    "cited_absolute",
    # a cited uncertainty on a value that is exact as written (a definition, a count, a
    # constructed quantity), where the printed digits carry no precision claim
    "cited_absolute_on_exact_value",
    "exact",
    "half_ulp_of_last_published_digit",
    "propagated_outward",
    "ulp_of_last_published_digit",
)
#: The rules under which this module derives sigma itself, so inflation is structurally
#: impossible rather than merely discouraged.
DERIVED_SIGMA_RULES = ("half_ulp_of_last_published_digit", "ulp_of_last_published_digit")

_DECIMAL = re.compile(r"-?\d+(\.\d+)?")

FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"

#: Decimal places used when a witness bound is also rendered for reading.  The exact
#: rational is always kept alongside; decisions never consume the rendered string.
WITNESS_DECIMAL_PLACES = 18


class ToleranceFittingError(ValueError):
    """Raised on malformed input, cap violation, guard violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Exact rational helpers
# ---------------------------------------------------------------------------


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _interval_data(lower: Fraction, upper: Fraction) -> dict[str, Any]:
    return {"lower": _fraction_data(lower), "upper": _fraction_data(upper)}


def _decimal_fraction(text: str) -> tuple[Fraction, int]:
    """Parse a plain decimal string exactly; return (value, decimal places)."""

    if not isinstance(text, str) or _DECIMAL.fullmatch(text) is None:
        raise ToleranceFittingError("declared decimal string malformed")
    whole, _, decimals = text.partition(".")
    sign = -1 if whole.startswith("-") else 1
    digits = int(whole.lstrip("-") + decimals or "0")
    return Fraction(sign * digits, 10 ** len(decimals)), len(decimals)


def _exact_number(value: Any) -> tuple[Fraction, int | None]:
    """Accept an int, a plain decimal string, or an exact rational object."""

    if isinstance(value, bool):
        raise ToleranceFittingError("boolean is not a measurement")
    if isinstance(value, int):
        return Fraction(value), 0
    if isinstance(value, str):
        return _decimal_fraction(value)
    if (
        isinstance(value, Mapping)
        and set(value) == {"numerator", "denominator"}
        and isinstance(value["numerator"], int)
        and isinstance(value["denominator"], int)
        and not isinstance(value["numerator"], bool)
        and not isinstance(value["denominator"], bool)
        and value["denominator"] > 0
    ):
        return Fraction(value["numerator"], value["denominator"]), None
    raise ToleranceFittingError("value must be an integer, decimal string, or exact rational")


def _decimal_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ToleranceFittingError("published-digit rules require a decimal string value")
    return value


# ---------------------------------------------------------------------------
# Declared rows: uncertainty is declared, never fitted (G1)
# ---------------------------------------------------------------------------

_ROW_REQUIRED = {"label", "point", "point_sigma_rule", "source", "value", "value_sigma_rule"}
_ROW_OPTIONAL = {
    "point_sigma",
    "point_sigma_citation",
    "value_sigma",
    "value_sigma_citation",
}


@dataclass(frozen=True, slots=True)
class MeasuredRow:
    """One declared measurement: a point, a value, and their declared uncertainties."""

    label: str
    point: Fraction
    point_sigma: Fraction
    point_sigma_rule: str
    value: Fraction
    value_sigma: Fraction
    value_sigma_rule: str
    source: str
    point_citation: str | None
    value_citation: str | None
    #: The caller's literal, kept verbatim: a published decimal string is evidence about
    #: precision, and reducing it to a rational would destroy the sigma derivation.
    point_declared: Any
    value_declared: Any

    def point_interval(self, coverage_k: Fraction) -> tuple[Fraction, Fraction]:
        pad = coverage_k * self.point_sigma
        return self.point - pad, self.point + pad

    def value_interval(self, coverage_k: Fraction) -> tuple[Fraction, Fraction]:
        pad = coverage_k * self.value_sigma
        return self.value - pad, self.value + pad


def _derive_sigma(rule: str, raw: Any, places: int | None) -> Fraction:
    """Sigma implied by a declared published-digit rule.  Never sees a residual."""

    _decimal_string(raw)
    if places is None:
        raise ToleranceFittingError("published-digit rules require a decimal string value")
    if rule == "half_ulp_of_last_published_digit":
        return Fraction(5, 10 ** (places + 1))
    return Fraction(1, 10**places)


def _resolve_sigma(
    field: str, row: Mapping[str, Any], raw: Any, places: int | None
) -> tuple[Fraction, str, str | None]:
    """Resolve one declared uncertainty, refusing any sigma the rule does not permit."""

    rule = row[f"{field}_sigma_rule"]
    if rule not in SIGMA_RULES:
        raise ToleranceFittingError(f"undeclared sigma rule: {rule!r}")
    supplied = row.get(f"{field}_sigma")
    citation = row.get(f"{field}_sigma_citation")

    if rule in DERIVED_SIGMA_RULES:
        if citation is not None:
            raise ToleranceFittingError(
                "a published-digit rule carries no separate citation; use the row source"
            )
        derived = _derive_sigma(rule, raw, places)
        if supplied is not None:
            declared, _ = _exact_number(supplied)
            if declared != derived:
                # This is the sigma-inflation guard.  The rule fixes sigma exactly; a
                # caller-supplied value that differs is an attempt to widen (or narrow)
                # the tolerance until the answer changes, so the run stops here.
                raise ToleranceFittingError(
                    "sigma_not_derivable_from_declared_rule: declared "
                    f"{declared} but {rule} on {raw!r} requires {derived}"
                )
        return derived, rule, None

    if rule == "exact":
        if citation is not None:
            raise ToleranceFittingError("an exact value carries no uncertainty citation")
        if supplied is not None:
            declared, _ = _exact_number(supplied)
            if declared != 0:
                raise ToleranceFittingError("an exact value must declare sigma zero")
        return Fraction(0), rule, None

    if supplied is None:
        raise ToleranceFittingError(f"{rule} requires an explicit {field}_sigma")
    if not isinstance(citation, str) or not citation.strip():
        raise ToleranceFittingError(f"{rule} requires a nonempty {field}_sigma_citation")
    declared, _ = _exact_number(supplied)
    if declared <= 0:
        raise ToleranceFittingError(f"{rule} requires a strictly positive sigma")
    if rule == "cited_absolute" and places is not None:
        floor = Fraction(5, 10 ** (places + 1))
        if declared < floor:
            raise ToleranceFittingError(
                "cited_absolute sigma is finer than the published last digit permits"
            )
    return declared, rule, citation.strip()


def parse_rows(rows: Any) -> list[MeasuredRow]:
    """Validate and freeze the declared rows.  Fails closed on anything ambiguous."""

    if not isinstance(rows, list) or not rows:
        raise ToleranceFittingError("rows must be a non-empty list")
    if len(rows) > SYSTEM_CAPS["max_rows"]:
        raise ToleranceFittingError("row count exceeds cap")
    parsed: list[MeasuredRow] = []
    labels: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ToleranceFittingError("each row must be an object")
        keys = set(row)
        if not _ROW_REQUIRED <= keys or not keys <= (_ROW_REQUIRED | _ROW_OPTIONAL):
            raise ToleranceFittingError("row key set changed")
        label = row["label"]
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ToleranceFittingError("row label must be a unique nonempty string")
        labels.add(label)
        source = row["source"]
        if not isinstance(source, str) or not source.strip():
            raise ToleranceFittingError("every row must cite a nonempty source")
        point, point_places = _exact_number(row["point"])
        value, value_places = _exact_number(row["value"])
        point_sigma, point_rule, point_citation = _resolve_sigma(
            "point", row, row["point"], point_places
        )
        value_sigma, value_rule, value_citation = _resolve_sigma(
            "value", row, row["value"], value_places
        )
        parsed.append(
            MeasuredRow(
                label=label.strip(),
                point=point,
                point_sigma=point_sigma,
                point_sigma_rule=point_rule,
                value=value,
                value_sigma=value_sigma,
                value_sigma_rule=value_rule,
                source=source.strip(),
                point_citation=point_citation,
                value_citation=value_citation,
                point_declared=row["point"],
                value_declared=row["value"],
            )
        )
    return parsed


def _parse_coverage(coverage_k: Any) -> Fraction:
    if isinstance(coverage_k, str) and "/" in coverage_k:
        try:
            value = Fraction(coverage_k)
        except (ValueError, ZeroDivisionError) as error:
            raise ToleranceFittingError("coverage_k malformed") from error
    else:
        value, _ = _exact_number(coverage_k)
    if value <= 0:
        raise ToleranceFittingError("coverage_k must be strictly positive")
    if value > SYSTEM_CAPS["max_coverage_k"]:
        # Widening the coverage factor until something fits is the same attack as
        # inflating sigma, so the factor is capped and recorded.
        raise ToleranceFittingError("coverage_k exceeds the declared cap")
    return value


def sigma_binding(rows: Sequence[MeasuredRow]) -> str:
    """Seal the declared uncertainties so a later run cannot quietly restate them."""

    return canonical_sha256(
        [
            {
                "label": row.label,
                "point_sigma": _fraction_data(row.point_sigma),
                "point_sigma_rule": row.point_sigma_rule,
                "source": row.source,
                "value_sigma": _fraction_data(row.value_sigma),
                "value_sigma_rule": row.value_sigma_rule,
            }
            for row in rows
        ]
    )


# ---------------------------------------------------------------------------
# Exact rational linear programming (G4)
# ---------------------------------------------------------------------------


class _Simplex:
    """Exact rational simplex over ``A x <= b`` with free ``x``, Bland's rule.

    Exact arithmetic plus Bland's rule means the method terminates without cycling and
    without any tolerance of its own: every comparison below is a rational comparison.
    Phase one both decides feasibility and, on failure, hands back the multipliers that
    become a Farkas certificate.
    """

    def __init__(self, matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction]) -> None:
        self.rows = len(matrix)
        self.width = len(matrix[0]) if matrix else 0
        if self.rows != len(rhs):
            raise ToleranceFittingError("linear program shape mismatch")
        self.signs = [1 if value >= 0 else -1 for value in rhs]
        # columns: x+ | x- | slack | artificial
        self.slack_at = 2 * self.width
        self.artificial_at = self.slack_at + self.rows
        columns = self.artificial_at + self.rows
        self.tableau: list[list[Fraction]] = []
        for index in range(self.rows):
            sign = Fraction(self.signs[index])
            line = [Fraction(0)] * (columns + 1)
            for column, cell in enumerate(matrix[index]):
                line[column] = sign * cell
                line[self.width + column] = -sign * cell
            line[self.slack_at + index] = sign
            line[self.artificial_at + index] = Fraction(1)
            line[columns] = sign * rhs[index]
            self.tableau.append(line)
        self.basis = [self.artificial_at + index for index in range(self.rows)]
        self.columns = columns
        self.pivots = 0

    def _cost_row(self, costs: Sequence[Fraction]) -> list[Fraction]:
        row = [Fraction(0)] * (self.columns + 1)
        for column in range(self.columns):
            row[column] = costs[column]
        for index, basic in enumerate(self.basis):
            factor = costs[basic]
            if factor:
                line = self.tableau[index]
                for column in range(self.columns + 1):
                    row[column] -= factor * line[column]
        return row

    def _pivot(self, row_index: int, column: int, cost: list[Fraction]) -> None:
        self.pivots += 1
        if self.pivots > SYSTEM_CAPS["max_lp_pivots"]:
            raise ToleranceFittingError("linear program exceeded the declared pivot cap")
        line = self.tableau[row_index]
        scale = line[column]
        self.tableau[row_index] = [cell / scale for cell in line]
        pivot_line = self.tableau[row_index]
        for index, other in enumerate(self.tableau):
            if index != row_index and other[column]:
                factor = other[column]
                self.tableau[index] = [
                    cell - factor * pivot_cell
                    for cell, pivot_cell in zip(other, pivot_line, strict=True)
                ]
        if cost[column]:
            factor = cost[column]
            for position in range(self.columns + 1):
                cost[position] -= factor * pivot_line[position]
        self.basis[row_index] = column

    def _iterate(self, cost: list[Fraction], allowed: Sequence[bool]) -> str:
        while True:
            entering = next(
                (
                    column
                    for column in range(self.columns)
                    if allowed[column] and cost[column] < 0
                ),
                None,
            )
            if entering is None:
                return "OPTIMAL"
            best_row = None
            best_ratio: Fraction | None = None
            for index in range(self.rows):
                cell = self.tableau[index][entering]
                if cell <= 0:
                    continue
                ratio = self.tableau[index][self.columns] / cell
                if (
                    best_ratio is None
                    or ratio < best_ratio
                    or (ratio == best_ratio and self.basis[index] < self.basis[best_row])
                ):
                    best_ratio, best_row = ratio, index
            if best_row is None:
                return "UNBOUNDED"
            self._pivot(best_row, entering, cost)

    def solve_phase_one(self) -> tuple[bool, list[Fraction]]:
        """Return (feasible, multipliers).  Multipliers are meaningful only on failure."""

        costs = [Fraction(0)] * self.columns
        for index in range(self.rows):
            costs[self.artificial_at + index] = Fraction(1)
        cost = self._cost_row(costs)
        allowed = [True] * self.columns
        self._iterate(cost, allowed)
        objective = -cost[self.columns]
        multipliers = [
            Fraction(1) - cost[self.artificial_at + index] for index in range(self.rows)
        ]
        if objective > 0:
            return False, multipliers
        self._purge_artificials()
        return True, multipliers

    def _purge_artificials(self) -> None:
        """Drive zero-valued artificials out of the basis so phase two cannot revive them."""

        for row_index in range(self.rows):
            if self.basis[row_index] < self.artificial_at:
                continue
            line = self.tableau[row_index]
            column = next(
                (index for index in range(self.artificial_at) if line[index] != 0), None
            )
            if column is None:
                continue  # redundant row; it stays basic at zero and is never entered
            cost = [Fraction(0)] * (self.columns + 1)
            self._pivot(row_index, column, cost)

    def optimize(self, objective: Sequence[Fraction], *, maximize: bool) -> tuple[str, Any]:
        """Optimize a linear functional of ``x`` over the polytope; phase one must have run."""

        direction = Fraction(-1) if maximize else Fraction(1)
        costs = [Fraction(0)] * self.columns
        for column, coefficient in enumerate(objective):
            costs[column] = direction * coefficient
            costs[self.width + column] = -direction * coefficient
        cost = self._cost_row(costs)
        allowed = [column < self.artificial_at for column in range(self.columns)]
        status = self._iterate(cost, allowed)
        if status == "UNBOUNDED":
            return "UNBOUNDED", None
        value = -cost[self.columns]
        return "OPTIMAL", -value if maximize else value

    def solution(self) -> list[Fraction]:
        values = [Fraction(0)] * (2 * self.width)
        for index, basic in enumerate(self.basis):
            if basic < 2 * self.width:
                values[basic] = self.tableau[index][self.columns]
        return [values[column] - values[self.width + column] for column in range(self.width)]


def _farkas_certificate(
    matrix: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
    signs: Sequence[int],
    multipliers: Sequence[Fraction],
) -> list[Fraction]:
    """Turn phase-one multipliers into a re-verified Farkas witness.

    The witness is ``lambda >= 0`` with ``lambda^T A = 0`` and ``lambda^T b < 0``: a
    nonnegative combination of the declared inequalities that adds up to a contradiction.
    It is checked here in exact arithmetic and can be re-checked by hand from the receipt,
    so an INFEASIBLE verdict never rests on trusting this solver.
    """

    witness = [-value * sign for value, sign in zip(multipliers, signs, strict=True)]
    if any(value < 0 for value in witness):
        raise ToleranceFittingError("infeasibility witness is not nonnegative")
    width = len(matrix[0]) if matrix else 0
    for column in range(width):
        total = sum(
            (value * row[column] for value, row in zip(witness, matrix, strict=True)),
            Fraction(0),
        )
        if total != 0:
            raise ToleranceFittingError("infeasibility witness does not annihilate the model")
    total = sum((value * cell for value, cell in zip(witness, rhs, strict=True)), Fraction(0))
    if total >= 0:
        raise ToleranceFittingError("infeasibility witness does not reach a contradiction")
    return witness


@dataclass(frozen=True, slots=True)
class LinearSystem:
    """``A x <= b`` built from declared intervals, with each row's provenance kept."""

    matrix: tuple[tuple[Fraction, ...], ...]
    rhs: tuple[Fraction, ...]
    origin: tuple[tuple[str, str], ...]  # (row label, "upper"/"lower")


def build_system(
    columns: Sequence[Sequence[Fraction]],
    rows: Sequence[MeasuredRow],
    coverage_k: Fraction,
) -> LinearSystem:
    """Two inequalities per row: the model must land inside the declared interval."""

    matrix: list[tuple[Fraction, ...]] = []
    rhs: list[Fraction] = []
    origin: list[tuple[str, str]] = []
    for design, row in zip(columns, rows, strict=True):
        lower, upper = row.value_interval(coverage_k)
        matrix.append(tuple(design))
        rhs.append(upper)
        origin.append((row.label, "upper"))
        matrix.append(tuple(-cell for cell in design))
        rhs.append(-lower)
        origin.append((row.label, "lower"))
    return LinearSystem(tuple(matrix), tuple(rhs), tuple(origin))


def decide_system(system: LinearSystem) -> dict[str, Any]:
    """FEASIBLE with a witness point, or INFEASIBLE with a re-verified Farkas witness."""

    simplex = _Simplex(system.matrix, system.rhs)
    feasible, multipliers = simplex.solve_phase_one()
    if feasible:
        return {"verdict": FEASIBLE, "simplex": simplex, "point": simplex.solution()}
    witness = _farkas_certificate(system.matrix, system.rhs, simplex.signs, multipliers)
    contributing = [
        {
            "bound": system.origin[index][1],
            "multiplier": _fraction_data(value),
            "row": system.origin[index][0],
        }
        for index, value in enumerate(witness)
        if value != 0
    ]
    total = sum(
        (value * cell for value, cell in zip(witness, system.rhs, strict=True)), Fraction(0)
    )
    return {
        "verdict": INFEASIBLE,
        "witness": {
            "checked_here": True,
            "combined_right_hand_side": _fraction_data(total),
            "kind": "farkas_nonnegative_combination",
            "reading": (
                "these declared inequalities, scaled by these nonnegative multipliers, "
                "sum to 0 <= a strictly negative number, so no coefficient vector reaches "
                "every interval"
            ),
            "terms": contributing,
            "unreachable_rows": sorted({term["row"] for term in contributing}),
        },
    }


# ---------------------------------------------------------------------------
# Declared ladder
# ---------------------------------------------------------------------------


def _coprime_exponents(denominator_bound: int, absolute_bound: int) -> tuple[Fraction, ...]:
    """Declared, finite, Occam-ordered rational exponent grid."""

    if not 1 <= denominator_bound <= SYSTEM_CAPS["max_exponent_denominator_bound"]:
        raise ToleranceFittingError("exponent denominator bound outside the declared cap")
    if not 1 <= absolute_bound <= SYSTEM_CAPS["max_exponent_absolute_bound"]:
        raise ToleranceFittingError("exponent absolute bound outside the declared cap")
    found: list[Fraction] = []
    for denominator in range(1, denominator_bound + 1):
        for numerator in range(-absolute_bound * denominator, absolute_bound * denominator + 1):
            if gcd(abs(numerator), denominator) != 1:
                continue
            found.append(Fraction(numerator, denominator))
    found.sort(key=lambda p: (p.denominator + abs(p.numerator), p.denominator, p.numerator))
    return tuple(found)


DEFAULT_EXPONENT_DENOMINATOR_BOUND = 4
DEFAULT_EXPONENT_ABSOLUTE_BOUND = 3


@dataclass(frozen=True, slots=True)
class LadderEntry:
    """One declared ladder entry: a family, its parameter budget, and its Occam rank."""

    entry_id: str
    track: str  # "linear_basis" or "power_law"
    track_rank: int
    entry_rank: int
    parameters: int
    linear_parameters: int
    terms: tuple[Term, ...] = ()
    exponent: Fraction | None = None

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        return (self.parameters, self.track_rank, self.entry_rank, self.entry_id)

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {
            "entry_id": self.entry_id,
            "linear_parameters": self.linear_parameters,
            "parameters": self.parameters,
            "track": self.track,
        }
        if self.track == "linear_basis":
            described["terms"] = [term.source for term in self.terms]
        else:
            described["exponent"] = str(self.exponent)
            described["template"] = "value = C^(1/v) * point^(u/v)"
        return described


def build_ladder(
    *,
    ladder: Sequence[Mapping[str, Any]] = LADDER,
    exponent_denominator_bound: int = DEFAULT_EXPONENT_DENOMINATOR_BOUND,
    exponent_absolute_bound: int = DEFAULT_EXPONENT_ABSOLUTE_BOUND,
    include_linear: bool = True,
    include_power_law: bool = True,
) -> tuple[LadderEntry, ...]:
    """The declared, finite, deterministically ordered ladder searched by :func:`fit_measured`.

    Linear entries come from B1's frozen ``LADDER`` unchanged, so an exact-data run of this
    module walks the same structures in the same order that B1 would.  Power-law entries
    add the one family B1 cannot express -- a parameter inside an exponent -- over a
    declared bounded rational exponent grid, reusing B2's discipline that every reported
    exponent is exact rather than a rounded root.
    """

    entries: list[LadderEntry] = []
    if include_linear:
        for rank, item in enumerate(ladder):
            terms: tuple[Term, ...] = tuple(item["terms"])
            entries.append(
                LadderEntry(
                    entry_id=f"linear:{item['family_id']}",
                    track="linear_basis",
                    track_rank=0,
                    entry_rank=rank,
                    parameters=len(terms),
                    linear_parameters=len(terms),
                    terms=terms,
                )
            )
    if include_power_law:
        grid = _coprime_exponents(exponent_denominator_bound, exponent_absolute_bound)
        for rank, exponent in enumerate(grid):
            entries.append(
                LadderEntry(
                    entry_id=f"power_law:{exponent}",
                    track="power_law",
                    track_rank=1,
                    entry_rank=rank,
                    # the exponent is fitted too, so it costs a parameter
                    parameters=2,
                    linear_parameters=1,
                    exponent=exponent,
                )
            )
    entries.sort(key=lambda entry: entry.sort_key)
    for entry in entries:
        if entry.parameters > SYSTEM_CAPS["max_parameters"]:
            raise ToleranceFittingError("ladder entry exceeds the declared parameter cap")
    return tuple(entries)


# ---------------------------------------------------------------------------
# Linear-basis track
# ---------------------------------------------------------------------------


def evaluate_term_at(term: Term, point: Fraction) -> Fraction | None:
    """B1's term algebra, extended to non-integer rational points where it is defined."""

    if point.denominator == 1:
        return evaluate_term(term, int(point))
    if term.family == "monomial":
        (degree,) = term.params
        return point**degree
    if term.family == "reciprocal":
        (degree,) = term.params
        return None if point == 0 else Fraction(1) / point**degree
    if term.family == "shifted_reciprocal":
        (degree,) = term.params
        return None if point == -1 else Fraction(1) / (point + 1) ** degree
    return None  # index-shaped families are undefined between the integers


def _design_columns(
    terms: Sequence[Term], rows: Sequence[MeasuredRow]
) -> list[list[Fraction]] | None:
    columns: list[list[Fraction]] = []
    for row in rows:
        line: list[Fraction] = []
        for term in terms:
            cell = evaluate_term_at(term, row.point)
            if cell is None:
                return None
            line.append(cell)
        columns.append(line)
    return columns


def _rank_raising_rows(columns: Sequence[Sequence[Fraction]], width: int) -> list[int]:
    """Indices of the first rows that raise the rank -- the minimum fit set (G2)."""

    chosen: list[int] = []
    reduced: list[list[Fraction]] = []
    pivots: list[int] = []
    for index, line in enumerate(columns):
        candidate = list(line)
        for pivot_column, pivot_row in zip(pivots, reduced, strict=True):
            if candidate[pivot_column]:
                factor = candidate[pivot_column]
                candidate = [
                    cell - factor * pivot_cell
                    for cell, pivot_cell in zip(candidate, pivot_row, strict=True)
                ]
        column = next((position for position, cell in enumerate(candidate) if cell), None)
        if column is None:
            continue
        scale = candidate[column]
        normalized = [cell / scale for cell in candidate]
        reduced.append(normalized)
        pivots.append(column)
        chosen.append(index)
        if len(chosen) == width:
            break
    return chosen


def _try_linear_entry(
    entry: LadderEntry, rows: Sequence[MeasuredRow], coverage_k: Fraction
) -> dict[str, Any]:
    outcome: dict[str, Any] = {**entry.describe(), "rows": len(rows)}
    if any(row.point_sigma != 0 for row in rows):
        return {
            **outcome,
            "verdict": "SKIP",
            "reason": "linear_basis_track_requires_exact_points",
        }
    columns = _design_columns(entry.terms, rows)
    if columns is None:
        return {**outcome, "verdict": "SKIP", "reason": "basis_undefined_on_a_declared_point"}
    confirmations = len(rows) - entry.parameters
    outcome["confirmations"] = confirmations
    if confirmations < SYSTEM_CAPS["min_confirmations"]:
        return {
            **outcome,
            "verdict": "BLOCK",
            "reason": "parsimony_budget_violated",
            "parsimony_rule": PARSIMONY_RULE,
        }
    fit_indices = _rank_raising_rows(columns, entry.linear_parameters)
    if len(fit_indices) < entry.linear_parameters:
        return {**outcome, "verdict": "REJECT", "reason": "rank_deficient_on_declared_points"}
    outcome["fit_rows"] = [rows[index].label for index in fit_indices]

    full = build_system(columns, rows, coverage_k)
    decision = decide_system(full)
    if decision["verdict"] == INFEASIBLE:
        return {
            **outcome,
            "verdict": INFEASIBLE,
            "reason": "no_coefficient_vector_reaches_every_declared_interval",
            "witness": decision["witness"],
        }

    # Holdout stays sovereign: the fit-row region alone must be able to reach every
    # untouched row's own declared interval.
    fit_set = set(fit_indices)
    fit_system = build_system(
        [columns[index] for index in fit_indices],
        [rows[index] for index in fit_indices],
        coverage_k,
    )
    fit_decision = decide_system(fit_system)
    if fit_decision["verdict"] == INFEASIBLE:  # pragma: no cover - implied by the full system
        return {
            **outcome,
            "verdict": INFEASIBLE,
            "reason": "fit_rows_alone_are_already_unreachable",
            "witness": fit_decision["witness"],
        }
    fit_simplex: _Simplex = fit_decision["simplex"]
    holdout: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in fit_set:
            continue
        low = fit_simplex.optimize(columns[index], maximize=False)
        high = fit_simplex.optimize(columns[index], maximize=True)
        lower, upper = row.value_interval(coverage_k)
        reachable = not (
            (low[0] == "OPTIMAL" and low[1] > upper) or (high[0] == "OPTIMAL" and high[1] < lower)
        )
        holdout.append(
            {
                "declared_interval": _interval_data(lower, upper),
                "predicted_from_fit_rows": {
                    "lower": None if low[0] != "OPTIMAL" else _fraction_data(low[1]),
                    "unbounded": low[0] != "OPTIMAL" or high[0] != "OPTIMAL",
                    "upper": None if high[0] != "OPTIMAL" else _fraction_data(high[1]),
                },
                "reachable": reachable,
                "row": row.label,
            }
        )
    unreachable = [item["row"] for item in holdout if not item["reachable"]]
    if unreachable:
        return {
            **outcome,
            "holdout": holdout,
            "reason": "holdout_row_unreachable_from_the_fit_row_region",
            "verdict": "REJECT",
            "witness": {
                "checked_here": True,
                "kind": "holdout_interval_disjoint_from_fit_region_prediction",
                "unreachable_rows": unreachable,
            },
        }
    return {
        **outcome,
        "coefficient_intervals": _coefficient_box(full, entry),
        "holdout": holdout,
        "reason": "feasible_against_every_declared_interval_with_holdout_reachable",
        "residual_space": "value",
        "standardized_residuals": _linear_residuals(full, columns, rows),
        "verdict": FEASIBLE,
        "witness_point": [_fraction_data(value) for value in decision["point"]],
    }


def _coefficient_box(system: LinearSystem, entry: LadderEntry) -> list[dict[str, Any]]:
    """Exact per-coefficient extremes over the feasible polytope."""

    box: list[dict[str, Any]] = []
    width = len(entry.terms)
    for column, term in enumerate(entry.terms):
        objective = [Fraction(1) if index == column else Fraction(0) for index in range(width)]
        low_simplex = _Simplex(system.matrix, system.rhs)
        low_simplex.solve_phase_one()
        high_simplex = _Simplex(system.matrix, system.rhs)
        high_simplex.solve_phase_one()
        low = low_simplex.optimize(objective, maximize=False)
        high = high_simplex.optimize(objective, maximize=True)
        box.append(
            {
                "lower": None if low[0] != "OPTIMAL" else _fraction_data(low[1]),
                "term": term.source,
                "unbounded": low[0] != "OPTIMAL" or high[0] != "OPTIMAL",
                "upper": None if high[0] != "OPTIMAL" else _fraction_data(high[1]),
            }
        )
    return box


def _linear_residuals(
    system: LinearSystem,
    columns: Sequence[Sequence[Fraction]],
    rows: Sequence[MeasuredRow],
) -> list[dict[str, Any]]:
    """Per-row standardized residual *intervals* over the whole feasible region."""

    if len(rows) > SYSTEM_CAPS["max_residual_report_rows"]:
        raise ToleranceFittingError("residual report exceeds the declared row cap")
    report: list[dict[str, Any]] = []
    for line, row in zip(columns, rows, strict=True):
        if row.value_sigma == 0:
            report.append(
                {
                    "lower": None,
                    "row": row.label,
                    "unbounded": False,
                    "upper": None,
                    "zero_sigma": True,
                }
            )
            continue
        objective = [cell / row.value_sigma for cell in line]
        offset = row.value / row.value_sigma
        low_simplex = _Simplex(system.matrix, system.rhs)
        low_simplex.solve_phase_one()
        high_simplex = _Simplex(system.matrix, system.rhs)
        high_simplex.solve_phase_one()
        low = low_simplex.optimize(objective, maximize=False)
        high = high_simplex.optimize(objective, maximize=True)
        report.append(
            {
                "lower": None if low[0] != "OPTIMAL" else _fraction_data(low[1] - offset),
                "row": row.label,
                "unbounded": low[0] != "OPTIMAL" or high[0] != "OPTIMAL",
                "upper": None if high[0] != "OPTIMAL" else _fraction_data(high[1] - offset),
                "zero_sigma": False,
            }
        )
    return report


# ---------------------------------------------------------------------------
# Power-law track: exact, and exact in the *point* uncertainty too
# ---------------------------------------------------------------------------


def invariant_interval(
    row: MeasuredRow, exponent: Fraction, coverage_k: Fraction
) -> tuple[Fraction, Fraction]:
    """Exact interval of ``value^v / point^u`` over the row's declared box.

    ``value^v`` and ``point^(-u)`` are each monotone in their argument on the positive
    axis, so the extremes of the invariant sit at the corners of the declared box and the
    enclosure below is exact -- not an outward bound, the exact range.  This is what lets
    the power-law track absorb uncertainty in the point as well as in the value, which
    the linear track cannot do.
    """

    numerator, denominator = exponent.numerator, exponent.denominator
    point_low, point_high = row.point_interval(coverage_k)
    value_low, value_high = row.value_interval(coverage_k)
    if point_low <= 0 or value_low <= 0:
        raise ToleranceFittingError(
            f"power-law track requires strictly positive declared intervals: row {row.label}"
        )
    low_power, high_power = value_low**denominator, value_high**denominator
    if numerator >= 0:
        divisor_low, divisor_high = point_low**numerator, point_high**numerator
    else:
        divisor_low, divisor_high = point_high**numerator, point_low**numerator
    return low_power / divisor_high, high_power / divisor_low


def decide_power_law(
    rows: Sequence[MeasuredRow], exponent: Fraction, coverage_k: Fraction
) -> dict[str, Any]:
    """Intersect every row's invariant interval.  Non-empty means feasible, exactly."""

    intervals = [invariant_interval(row, exponent, coverage_k) for row in rows]
    lower_index = max(range(len(rows)), key=lambda index: (intervals[index][0], -index))
    upper_index = min(range(len(rows)), key=lambda index: (intervals[index][1], index))
    lower, upper = intervals[lower_index][0], intervals[upper_index][1]
    per_row = [
        {"interval": _interval_data(*interval), "row": row.label}
        for row, interval in zip(rows, intervals, strict=True)
    ]
    if lower <= upper:
        return {
            "constant_interval": _interval_data(lower, upper),
            "invariant_intervals": per_row,
            "verdict": FEASIBLE,
        }
    return {
        "invariant_intervals": per_row,
        "verdict": INFEASIBLE,
        "witness": {
            "checked_here": True,
            "gap": _fraction_data(lower - upper),
            "kind": "disjoint_invariant_intervals",
            "reading": (
                f"row {rows[lower_index].label} forces the invariant value^v/point^u to be "
                f"at least {decimal_string(lower, WITNESS_DECIMAL_PLACES)}, row "
                f"{rows[upper_index].label} forces it to be at most "
                f"{decimal_string(upper, WITNESS_DECIMAL_PLACES)}, and the first exceeds the "
                "second, so no constant serves both"
            ),
            "requires_at_least": {
                "bound": _fraction_data(lower),
                "bound_decimal": decimal_string(lower, WITNESS_DECIMAL_PLACES),
                "row": rows[lower_index].label,
            },
            "requires_at_most": {
                "bound": _fraction_data(upper),
                "bound_decimal": decimal_string(upper, WITNESS_DECIMAL_PLACES),
                "row": rows[upper_index].label,
            },
            "unreachable_rows": sorted({rows[lower_index].label, rows[upper_index].label}),
        },
    }


def _compact_witness(witness: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Witness without the exact rationals, for entries recorded but never selected."""

    if not witness or "requires_at_least" not in witness:
        return witness if witness is None else dict(witness)
    return {
        "checked_here": witness["checked_here"],
        "kind": witness["kind"],
        "reading": witness["reading"],
        "requires_at_least": {
            "bound_decimal": witness["requires_at_least"]["bound_decimal"],
            "row": witness["requires_at_least"]["row"],
        },
        "requires_at_most": {
            "bound_decimal": witness["requires_at_most"]["bound_decimal"],
            "row": witness["requires_at_most"]["row"],
        },
        "unreachable_rows": witness["unreachable_rows"],
    }


def _try_power_law_entry(
    entry: LadderEntry, rows: Sequence[MeasuredRow], coverage_k: Fraction
) -> dict[str, Any]:
    exponent = entry.exponent
    assert exponent is not None
    outcome: dict[str, Any] = {**entry.describe(), "rows": len(rows)}
    confirmations = len(rows) - entry.parameters
    outcome["confirmations"] = confirmations
    if confirmations < SYSTEM_CAPS["min_confirmations"]:
        return {
            **outcome,
            "verdict": "BLOCK",
            "reason": "parsimony_budget_violated",
            "parsimony_rule": PARSIMONY_RULE,
        }
    try:
        decision = decide_power_law(rows, exponent, coverage_k)
    except ToleranceFittingError as error:
        return {**outcome, "verdict": "SKIP", "reason": str(error)}
    if decision["verdict"] == INFEASIBLE:
        # The witness is the certificate; the full per-row interval list adds nothing that
        # cannot be recomputed from the declared rows, and at large exponent denominators
        # it would dominate the receipt.
        return {
            **outcome,
            "reason": "no_constant_reaches_every_declared_invariant_interval",
            "verdict": INFEASIBLE,
            "witness": decision["witness"],
        }
    # One linear parameter: the fit row is the first row, and every other row is holdout
    # whose own invariant interval must meet the fit row's.  In one dimension a non-empty
    # total intersection already implies every pairwise meeting, so the report below
    # records the holdout check rather than re-deciding it.
    fit_row = rows[0]
    fit_interval = decision["invariant_intervals"][0]["interval"]
    holdout = [
        {
            "declared_invariant_interval": item["interval"],
            "meets_fit_row_interval": True,
            "row": item["row"],
        }
        for item in decision["invariant_intervals"][1:]
    ]
    lower = Fraction(
        decision["constant_interval"]["lower"]["numerator"],
        decision["constant_interval"]["lower"]["denominator"],
    )
    upper = Fraction(
        decision["constant_interval"]["upper"]["numerator"],
        decision["constant_interval"]["upper"]["denominator"],
    )
    return {
        **outcome,
        "coefficient_intervals": [
            {
                "lower": decision["constant_interval"]["lower"],
                "term": f"C where value^{exponent.denominator} = C * point^{exponent.numerator}",
                "unbounded": False,
                "upper": decision["constant_interval"]["upper"],
            }
        ],
        "fit_rows": [fit_row.label],
        "fit_row_invariant_interval": fit_interval,
        "holdout": holdout,
        "invariant_intervals": decision["invariant_intervals"],
        "reason": "feasible_against_every_declared_interval_with_holdout_reachable",
        "residual_space": "invariant_value_power_v_over_point_power_u",
        "standardized_residuals": _invariant_residuals(
            rows, decision["invariant_intervals"], lower, upper
        ),
        "verdict": FEASIBLE,
    }


def _invariant_residuals(
    rows: Sequence[MeasuredRow],
    intervals: Sequence[Mapping[str, Any]],
    lower: Fraction,
    upper: Fraction,
) -> list[dict[str, Any]]:
    """Standardized residual intervals in invariant space, over the feasible constant range."""

    report: list[dict[str, Any]] = []
    for row, item in zip(rows, intervals, strict=True):
        interval = item["interval"]
        row_low = Fraction(interval["lower"]["numerator"], interval["lower"]["denominator"])
        row_high = Fraction(interval["upper"]["numerator"], interval["upper"]["denominator"])
        half_width = (row_high - row_low) / 2
        centre = (row_high + row_low) / 2
        if half_width == 0:
            report.append(
                {"lower": None, "row": row.label, "unbounded": False, "upper": None, "zero_sigma": True}
            )
            continue
        report.append(
            {
                "lower": _fraction_data((lower - centre) / half_width),
                "row": row.label,
                "unbounded": False,
                "upper": _fraction_data((upper - centre) / half_width),
                "zero_sigma": False,
            }
        )
    return report


# ---------------------------------------------------------------------------
# Certified exponent bracket: how tightly is the exponent actually pinned?
# ---------------------------------------------------------------------------

#: Working precision of the outward-rounded exponent search, matching the house idiom in
#: :mod:`.interval_threshold_certifier`: every operation rounds outward, and a comparison
#: whose margin is smaller than the accumulated error returns "unresolved", never a guess.
IV_DPS = 60
CERTIFIED_FEASIBLE = "certified_feasible"
CERTIFIED_INFEASIBLE = "certified_infeasible"
UNRESOLVED = "unresolved_straddle"


def decimal_string(value: Fraction, places: int) -> str:
    """Round-half-even an exact rational to a fixed decimal string, exactly."""

    scaled = value * 10**places
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor
    half = Fraction(1, 2)
    if remainder > half or (remainder == half and floor % 2 == 1):
        floor += 1
    sign = "-" if floor < 0 else ""
    digits = str(abs(floor)).rjust(places + 1, "0")
    if not places:
        return f"{sign}{digits}"
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def certify_power_law_outward(
    rows: Sequence[MeasuredRow], exponent: Fraction, coverage_k: Fraction
) -> str:
    """Three-way outward-rounded feasibility for an arbitrary real exponent.

    The exact rational test in :func:`decide_power_law` needs ``point**u`` with integer
    ``u``, so it cannot reach an exponent whose denominator is astronomically large.  This
    routine answers the same question at any exponent by intersecting the invariant
    intervals in ``mpmath.iv`` arithmetic at :data:`IV_DPS`, where every operation rounds
    outward.  It returns ``certified_feasible`` or ``certified_infeasible`` only when the
    whole enclosure sits on one side; otherwise ``unresolved_straddle``, because no
    finite-precision evaluation at this precision is entitled to a verdict.
    """

    from mpmath import iv  # local: the exact-rational paths must not depend on mpmath

    iv.dps = IV_DPS

    def enclose(value: Fraction) -> Any:
        return iv.mpf(value.numerator) / iv.mpf(value.denominator)

    power = enclose(exponent)
    lower_bound = None
    upper_bound = None
    for row in rows:
        point_low, point_high = row.point_interval(coverage_k)
        value_low, value_high = row.value_interval(coverage_k)
        if point_low <= 0 or value_low <= 0:
            raise ToleranceFittingError(
                f"outward power-law certification needs positive intervals: row {row.label}"
            )
        left, right = enclose(point_low) ** power, enclose(point_high) ** power
        divisor_low = iv.mpf([min(left.a, right.a), min(left.b, right.b)])
        divisor_high = iv.mpf([max(left.a, right.a), max(left.b, right.b)])
        low = enclose(value_low) / divisor_high
        high = enclose(value_high) / divisor_low
        lower_bound = (
            low
            if lower_bound is None
            else iv.mpf([max(lower_bound.a, low.a), max(lower_bound.b, low.b)])
        )
        upper_bound = (
            high
            if upper_bound is None
            else iv.mpf([min(upper_bound.a, high.a), min(upper_bound.b, high.b)])
        )
    if lower_bound is None or upper_bound is None:  # pragma: no cover - rows are non-empty
        raise ToleranceFittingError("no rows to certify")
    if lower_bound.b <= upper_bound.a:
        return CERTIFIED_FEASIBLE
    if lower_bound.a > upper_bound.b:
        return CERTIFIED_INFEASIBLE
    return UNRESOLVED


def certified_exponent_bracket(
    rows: Any,
    *,
    coverage_k: Any,
    centre: str,
    outer_offset: str,
    places: int = 30,
    max_iterations: int = 120,
) -> dict[str, Any]:
    """Bracket the feasible exponents around a declared centre, both sides certified.

    Each reported endpoint carries its own outward-rounded verdict, so the four endpoints
    are true statements on their own.  Reading them as "the feasible set is exactly this
    interval" additionally assumes the feasible exponent set is connected -- that
    assumption is declared in the result and is *not* proved here.
    """

    parsed = parse_rows(rows) if not isinstance(rows[0], MeasuredRow) else list(rows)
    coverage = _parse_coverage(coverage_k)
    centre_value = Fraction(centre)
    offset = Fraction(outer_offset)
    if offset <= 0:
        raise ToleranceFittingError("outer_offset must be strictly positive")
    centre_verdict = certify_power_law_outward(parsed, centre_value, coverage)
    sides: dict[str, Any] = {}
    for name, direction in (("lower", -1), ("upper", 1)):
        outer = centre_value + direction * offset
        outer_verdict = certify_power_law_outward(parsed, outer, coverage)
        side: dict[str, Any] = {
            "outer_probe": decimal_string(outer, places),
            "outer_probe_verdict": outer_verdict,
        }
        if centre_verdict != CERTIFIED_FEASIBLE or outer_verdict != CERTIFIED_INFEASIBLE:
            side["bracketed"] = False
            sides[name] = side
            continue
        inside, outside = centre_value, outer
        iterations = 0
        while iterations < max_iterations and abs(outside - inside) > Fraction(1, 10**places):
            middle = Fraction(decimal_string((inside + outside) / 2, places))
            if middle in (inside, outside):
                break
            verdict = certify_power_law_outward(parsed, middle, coverage)
            if verdict == CERTIFIED_FEASIBLE:
                inside = middle
            elif verdict == CERTIFIED_INFEASIBLE:
                outside = middle
            else:
                break
            iterations += 1
        side.update(
            {
                "bracketed": True,
                "certified_feasible_at": decimal_string(inside, places),
                "certified_infeasible_at": decimal_string(outside, places),
                "iterations": iterations,
            }
        )
        sides[name] = side
    return {
        "centre": centre,
        "centre_verdict": centre_verdict,
        "connectedness_of_the_feasible_set": (
            "declared assumption, not proved here; every endpoint verdict below is "
            "individually certified and stands on its own"
        ),
        "coverage_k": str(coverage),
        "decimal_places": places,
        "iv_dps": IV_DPS,
        "outer_offset": outer_offset,
        "rounding": "outward on every interval operation",
        "sides": sides,
    }


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def fit_measured(
    rows: Any,
    *,
    coverage_k: Any,
    ladder: Sequence[LadderEntry] | None = None,
    exponent_probes: Sequence[str] = (),
) -> dict[str, Any]:
    """Search the declared ladder for the simplest entry feasible against the intervals.

    ``exponent_probes`` are extra declared power-law exponents decided and recorded but
    never selected: they exist so a run can state, in the receipt, that a named competitor
    exponent was tested and refused.
    """

    parsed = parse_rows(rows)
    coverage = _parse_coverage(coverage_k)
    entries = tuple(ladder) if ladder is not None else build_ladder()
    if not entries:
        raise ToleranceFittingError("ladder is empty")

    examined: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for entry in entries:
        if entry.track == "linear_basis":
            outcome = _try_linear_entry(entry, parsed, coverage)
        else:
            outcome = _try_power_law_entry(entry, parsed, coverage)
        if outcome["verdict"] == FEASIBLE:
            accepted = outcome
            break
        examined.append(outcome)

    probes: list[dict[str, Any]] = []
    for text in exponent_probes:
        exponent = Fraction(text)
        probe_entry = LadderEntry(
            entry_id=f"probe:power_law:{exponent}",
            track="power_law",
            track_rank=1,
            entry_rank=-1,
            parameters=2,
            linear_parameters=1,
            exponent=exponent,
        )
        outcome = _try_power_law_entry(probe_entry, parsed, coverage)
        probes.append(
            {
                "entry_id": probe_entry.entry_id,
                "exponent": str(exponent),
                "reason": outcome.get("reason"),
                "verdict": outcome["verdict"],
                "witness": _compact_witness(outcome.get("witness")),
            }
        )

    if accepted is not None:
        decision = "FEASIBLE_MINIMAL"
        blocker = None
    elif any(item["verdict"] in {"BLOCK", "SKIP"} for item in examined) and not any(
        item["verdict"] == INFEASIBLE for item in examined
    ):
        decision = "BLOCKED"
        blocker = "no_declared_ladder_entry_was_decidable_on_these_rows"
    elif all(item["verdict"] in {INFEASIBLE, "REJECT", "BLOCK", "SKIP"} for item in examined):
        decision = "INFEASIBLE_ALL_LADDER"
        blocker = "no_declared_ladder_entry_is_feasible_against_the_declared_intervals"
    else:  # pragma: no cover - the verdict alphabet above is exhaustive
        decision = "BLOCKED"
        blocker = "unclassified_ladder_outcome"

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "counts": {
            "declared_rows": len(parsed),
            "entries_examined": len(examined) + (1 if accepted else 0),
            "entries_rejected_before_acceptance": len(examined),
            "exponent_probes": len(probes),
            "ladder_entries": len(entries),
            "rows_with_caller_declared_sigma": sum(
                1
                for row in parsed
                if row.point_sigma_rule not in (*DERIVED_SIGMA_RULES, "exact")
                or row.value_sigma_rule not in (*DERIVED_SIGMA_RULES, "exact")
            ),
            "rows_with_module_derived_sigma": sum(
                1
                for row in parsed
                if row.point_sigma_rule in (*DERIVED_SIGMA_RULES, "exact")
                and row.value_sigma_rule in (*DERIVED_SIGMA_RULES, "exact")
            ),
        },
        "coverage_factor": {
            "k": str(coverage),
            "note": (
                "every declared interval is [value - k*sigma, value + k*sigma]; k is declared "
                "by the caller, capped by system_caps.max_coverage_k, and never adjusted to "
                "change a verdict"
            ),
        },
        "decision": decision,
        "declared_rows": [
            {
                "declared_point_interval": _interval_data(*row.point_interval(coverage)),
                "declared_value_interval": _interval_data(*row.value_interval(coverage)),
                "label": row.label,
                "point": row.point_declared,
                "point_exact": _fraction_data(row.point),
                "point_sigma": _fraction_data(row.point_sigma),
                "point_sigma_citation": row.point_citation,
                "point_sigma_rule": row.point_sigma_rule,
                "source": row.source,
                "value": row.value_declared,
                "value_exact": _fraction_data(row.value),
                "value_sigma": _fraction_data(row.value_sigma),
                "value_sigma_citation": row.value_citation,
                "value_sigma_rule": row.value_sigma_rule,
            }
            for row in parsed
        ],
        "exponent_probes": probes,
        "first_blocker": blocker,
        "ladder_schema": LADDER_SCHEMA,
        "minimality_certificate": {
            "ordering": LADDER_ORDERING,
            "parsimony_rule": PARSIMONY_RULE,
            "strictly_simpler_entries_rejected": examined,
        },
        "parsimony_comparison": [
            {
                "confirmations": item.get("confirmations"),
                "entry_id": item["entry_id"],
                "parameters": item["parameters"],
                "reason": item.get("reason"),
                "rows": item["rows"],
                "unreachable_rows": (item.get("witness") or {}).get("unreachable_rows"),
                "verdict": item["verdict"],
            }
            for item in [*examined, *([accepted] if accepted else [])]
        ],
        "result": accepted,
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "sigma_binding_sha256": sigma_binding(parsed),
        "sigma_policy": {
            "derived_rules": list(DERIVED_SIGMA_RULES),
            "declared_rules": list(SIGMA_RULES),
            "rule": (
                "sigma is declared per row by a named rule and a cited source; under the "
                "published-digit rules it is re-derived here from the published decimal "
                "string and a caller-supplied sigma that disagrees aborts the run; no code "
                "path in this module reads a residual to obtain a sigma"
            ),
        },
        "system_caps": SYSTEM_CAPS,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# The five mandatory controls, run and sealed
# ---------------------------------------------------------------------------

CONTROL_SCHEMA = "invariant-tolerance-aware-controls-result-1.0"

CONTROL_SCOPE = (
    "The five controls that make a tolerance-aware verdict mean something, each with its "
    "expectation declared before the run: exact data at tiny sigma must reproduce B1's "
    "exact answer; the same law perturbed strictly inside sigma must still be recovered "
    "with the true coefficients inside the reported region; perturbation larger than sigma "
    "must turn INFEASIBLE; an entry that cannot pay its parsimony budget must be blocked "
    "with the rule cited; and a sigma that the declared rule does not permit must abort the "
    "run. A control receipt is evidence about this module, not about any measured system."
)

_CONTROL_SOURCE = "control fixture: values constructed by this module, not measured"
_CONTROL_CITATION = "control fixture: declared instrument precision"


def _control_rows(
    function: Any, points: Sequence[int], sigma: str, offsets: Mapping[int, str] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        value = Fraction(function(point)) + Fraction((offsets or {}).get(point, "0"))
        rows.append(
            {
                "label": f"n{point}",
                "point": point,
                "point_sigma_rule": "exact",
                "source": _CONTROL_SOURCE,
                "value": decimal_string(value, 6),
                "value_sigma": sigma,
                "value_sigma_citation": _CONTROL_CITATION,
                "value_sigma_rule": "cited_absolute_on_exact_value",
            }
        )
    return rows


def _contains(interval: Mapping[str, Any], target: Fraction) -> bool:
    if interval["lower"] is None or interval["upper"] is None:
        return False
    lower = Fraction(interval["lower"]["numerator"], interval["lower"]["denominator"])
    upper = Fraction(interval["upper"]["numerator"], interval["upper"]["denominator"])
    return lower <= target <= upper


def build_controls() -> dict[str, Any]:
    """Run every mandatory control and seal the outcome.  Deterministic and replayable."""

    from .basis_synthesis import synthesize_basis

    linear = build_ladder(include_power_law=False)
    controls: list[dict[str, Any]] = []

    # (a) exact data at tiny sigma must land exactly where B1 lands.
    agreement: list[dict[str, Any]] = []
    for name, function, points in (
        ("triangular", lambda n: n * (n + 1) // 2, tuple(range(10))),
        ("cubic", lambda n: 2 * n**3 + 2 * n**2 + n + 7, tuple(range(10))),
        ("geometric", lambda n: 3 * 2**n, tuple(range(10))),
        ("alternating", lambda n: (-1) ** n * n, tuple(range(10))),
        ("binomial", lambda n: n * (n - 1) // 2, tuple(range(10))),
    ):
        fitted = fit_measured(
            _control_rows(function, points, "0.0000001"), coverage_k=1, ladder=linear
        )
        exact = synthesize_basis([{"point": point, "value": function(point)} for point in points])
        b1_family = (exact["result"] or {}).get("family_id")
        agreement.append(
            {
                "b1_decision": exact["decision"],
                "b1_family_id": b1_family,
                "b8_decision": fitted["decision"],
                "b8_entry_id": (fitted["result"] or {}).get("entry_id"),
                "sequence": name,
                "same_structure": (fitted["result"] or {}).get("entry_id")
                == f"linear:{b1_family}",
            }
        )
    controls.append(
        {
            "control_id": "a_exact_data_tiny_sigma_agrees_with_b1",
            "detail": agreement,
            "expectation": (
                "with exact rows and a tiny declared sigma, every sequence is recovered in "
                "the same ladder family that B1 recovers exactly"
            ),
            "satisfied": all(item["same_structure"] for item in agreement),
        }
    )

    # (b) the same law perturbed strictly inside sigma is still recovered, and the true
    #     coefficients lie inside the reported region.
    truth = {"1": Fraction(7), "n^1": Fraction(1), "n^2": Fraction(2), "n^3": Fraction(2)}
    offsets = {point: decimal_string(Fraction((-1) ** point * (point % 5), 1000), 6) for point in range(10)}
    inside = fit_measured(
        _control_rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, tuple(range(10)), "0.01", offsets),
        coverage_k=1,
        ladder=linear,
    )
    accepted = inside["result"] or {}
    covered = [
        {
            "contains_true_value": _contains(interval, truth[interval["term"]]),
            "term": interval["term"],
            "true_value": _fraction_data(truth[interval["term"]]),
        }
        for interval in accepted.get("coefficient_intervals", [])
    ]
    controls.append(
        {
            "control_id": "b_noise_inside_sigma_still_recovers_the_law",
            "detail": {
                "coefficient_coverage": covered,
                "decision": inside["decision"],
                "entry_id": accepted.get("entry_id"),
                "largest_offset": "0.004",
                "sigma": "0.01",
            },
            "expectation": (
                "perturbing every row strictly inside its declared sigma leaves the same "
                "family accepted and every true coefficient inside the reported region"
            ),
            "satisfied": (
                inside["decision"] == "FEASIBLE_MINIMAL"
                and accepted.get("entry_id") == "linear:polynomial_3"
                and bool(covered)
                and all(item["contains_true_value"] for item in covered)
            ),
        }
    )

    # (c) perturbation larger than sigma must break feasibility.
    outside = fit_measured(
        _control_rows(
            lambda n: 2 * n**3 + 2 * n**2 + n + 7, tuple(range(10)), "0.001", {5: "0.5"}
        ),
        coverage_k=1,
        ladder=linear,
    )
    controls.append(
        {
            "control_id": "c_noise_larger_than_sigma_is_infeasible",
            "detail": {
                "decision": outside["decision"],
                "first_blocker": outside["first_blocker"],
                "offset_applied": "0.5",
                "sigma": "0.001",
            },
            "expectation": (
                "one row displaced by 500 times its declared sigma leaves no ladder entry "
                "feasible; the guard is what refuses, not the size of the residual"
            ),
            "satisfied": outside["decision"] == "INFEASIBLE_ALL_LADDER",
        }
    )

    # (d) parsimony, both halves.  First: an entry that *does* reach every interval is
    # still refused because it cannot pay its budget -- proved by showing the same system
    # is feasible when the budget is ignored.  Second: when a simpler entry is feasible the
    # search stops there, so a larger entry never gets to win by fitting better.
    starved_rows = _control_rows(
        lambda n: 2 * n**3 + 2 * n**2 + n + 7, (0, 1, 2, 3, 4), "0.000001"
    )
    starved = fit_measured(starved_rows, coverage_k=1, ladder=linear)
    blocked = [
        item
        for item in starved["minimality_certificate"]["strictly_simpler_entries_rejected"]
        if item["verdict"] == "BLOCK" and item.get("reason") == "parsimony_budget_violated"
    ]
    cubic_entry = next(entry for entry in linear if entry.entry_id == "linear:polynomial_3")
    parsed_starved = parse_rows(starved_rows)
    cubic_columns = _design_columns(cubic_entry.terms, parsed_starved)
    assert cubic_columns is not None
    would_fit = decide_system(build_system(cubic_columns, parsed_starved, Fraction(1)))

    line_rows = _control_rows(
        lambda n: 3 * n + 4,
        tuple(range(10)),
        "0.01",
        {point: decimal_string(Fraction((-1) ** point * (point % 5), 1000), 6) for point in range(10)},
    )
    line = fit_measured(line_rows, coverage_k=1, ladder=linear)
    larger_entries = [
        item["entry_id"]
        for item in line["parsimony_comparison"]
        if item["parameters"] > (line["result"] or {}).get("parameters", 0)
    ]
    controls.append(
        {
            "control_id": "d_over_parameterized_entry_fails_the_parsimony_budget",
            "detail": {
                "budget_starved": {
                    "blocked_entries": [
                        {
                            "confirmations": item["confirmations"],
                            "entry_id": item["entry_id"],
                            "parameters": item["parameters"],
                            "reason": item["reason"],
                        }
                        for item in blocked
                    ],
                    "decision": starved["decision"],
                    "interpolating_entry": cubic_entry.entry_id,
                    "interpolating_entry_reaches_every_interval_when_the_budget_is_ignored": (
                        would_fit["verdict"] == FEASIBLE
                    ),
                    "min_confirmations": SYSTEM_CAPS["min_confirmations"],
                    "rows": len(starved_rows),
                    "selected_entry_id": (starved["result"] or {}).get("entry_id"),
                },
                "parsimony_rule": PARSIMONY_RULE,
                "simpler_entry_wins": {
                    "decision": line["decision"],
                    "larger_entries_examined_after_acceptance": larger_entries,
                    "rows": len(line_rows),
                    "selected_entry_id": (line["result"] or {}).get("entry_id"),
                },
            },
            "expectation": (
                "an entry that reaches every declared interval is still refused when "
                "n - k < min_confirmations, so interpolation cannot be reported as a "
                "result; and when a simpler entry is feasible the search stops there, so no "
                "larger entry is ever examined, let alone preferred for fitting better"
            ),
            "satisfied": (
                bool(blocked)
                and would_fit["verdict"] == FEASIBLE
                and starved["decision"] == "INFEASIBLE_ALL_LADDER"
                and starved["result"] is None
                and line["decision"] == "FEASIBLE_MINIMAL"
                and (line["result"] or {}).get("entry_id") == "linear:polynomial_1"
                and larger_entries == []
            ),
        }
    )

    # (e) a sigma the declared rule does not permit aborts the run.
    attacks: list[dict[str, Any]] = []
    for attack_id, mutate in (
        ("inflate_declared_half_ulp_sigma", {"value_sigma": "0.5"}),
        ("deflate_declared_half_ulp_sigma", {"value_sigma": "0.0000001"}),
    ):
        row = {
            "label": "row",
            "point": 1,
            "point_sigma_rule": "exact",
            "source": _CONTROL_SOURCE,
            "value": "1.00",
            "value_sigma_rule": "half_ulp_of_last_published_digit",
            **mutate,
        }
        try:
            parse_rows([row])
        except ToleranceFittingError as error:
            attacks.append({"attack_id": attack_id, "refused": True, "reason": str(error)})
        else:  # pragma: no cover - the guard is what this control exists to prove
            attacks.append({"attack_id": attack_id, "refused": False, "reason": None})
    try:
        _parse_coverage(SYSTEM_CAPS["max_coverage_k"] + 1)
    except ToleranceFittingError as error:
        attacks.append(
            {"attack_id": "widen_the_coverage_factor", "refused": True, "reason": str(error)}
        )
    else:  # pragma: no cover
        attacks.append(
            {"attack_id": "widen_the_coverage_factor", "refused": False, "reason": None}
        )
    controls.append(
        {
            "control_id": "e_sigma_inflation_is_refused",
            "detail": attacks,
            "expectation": (
                "under a published-digit rule the sigma is re-derived here, so a supplied "
                "sigma that differs -- larger or smaller -- aborts the run; and the coverage "
                "factor, the other way to widen an interval, is capped"
            ),
            "satisfied": all(item["refused"] for item in attacks),
        }
    )

    body = {
        "claims": CLAIMS,
        "controls": controls,
        "counts": {
            "controls": len(controls),
            "controls_satisfied": sum(1 for item in controls if item["satisfied"]),
        },
        "decision": "PASS" if all(item["satisfied"] for item in controls) else "BLOCK",
        "first_blocker": None
        if all(item["satisfied"] for item in controls)
        else "a declared control expectation was not met",
        "parsimony_rule": PARSIMONY_RULE,
        "schema_version": CONTROL_SCHEMA,
        "scope": CONTROL_SCOPE,
        "system_caps": SYSTEM_CAPS,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_controls(value: Mapping[str, Any]) -> None:
    """Seal check, forbidden-key scan, and exact replay of the control receipt."""

    if value.get("schema_version") != CONTROL_SCHEMA:
        raise ToleranceFittingError("control receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise ToleranceFittingError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    if value.get("content_sha256") != canonical_sha256(body):
        raise ToleranceFittingError("control receipt seal changed")
    if dict(value) != build_controls():
        raise ToleranceFittingError("control receipt exact replay changed")


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def forbidden_receipt_keys(value: Any, path: str = "$") -> list[str]:
    """Locate any key that would turn this receipt into a scoreboard."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_RECEIPT_KEY_TOKENS):
                found.append(f"{path}.{key}")
            found.extend(forbidden_receipt_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_receipt_keys(item, f"{path}[{index}]"))
    return found


def _replay_arguments(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the exact call that produced this receipt, from the receipt alone."""

    rows: list[dict[str, Any]] = []
    for row in value.get("declared_rows", []):
        rebuilt: dict[str, Any] = {
            "label": row["label"],
            "point": row["point"],
            "point_sigma_rule": row["point_sigma_rule"],
            "source": row["source"],
            "value": row["value"],
            "value_sigma_rule": row["value_sigma_rule"],
        }
        for field in ("point", "value"):
            if rebuilt[f"{field}_sigma_rule"] in (*DERIVED_SIGMA_RULES, "exact"):
                continue
            rebuilt[f"{field}_sigma"] = row[f"{field}_sigma"]
            citation = row[f"{field}_sigma_citation"]
            if citation is not None:
                rebuilt[f"{field}_sigma_citation"] = citation
        rows.append(rebuilt)
    return {
        "rows": rows,
        "coverage_k": value["coverage_factor"]["k"],
        "exponent_probes": [probe["exponent"] for probe in value.get("exponent_probes", [])],
    }


def validate_result(
    value: Mapping[str, Any], *, ladder: Sequence[LadderEntry] | None = None
) -> None:
    """Reject tamper, drift, or a smuggled score by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ToleranceFittingError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    # The house rule is checked before the seal: a receipt carrying a score is wrong in a
    # way worth naming, whether or not the seal was updated to match it.
    smuggled = forbidden_receipt_keys(body)
    if smuggled:
        raise ToleranceFittingError(f"receipt carries a scalar goodness key: {smuggled[0]}")
    if value.get("content_sha256") != canonical_sha256(body):
        raise ToleranceFittingError("result seal changed")
    if value.get("claims") != CLAIMS:
        raise ToleranceFittingError("claims changed")
    if value.get("system_caps") != SYSTEM_CAPS:
        raise ToleranceFittingError("system caps changed")
    replayed = fit_measured(**_replay_arguments(value), ladder=ladder)
    if dict(value) != replayed:
        raise ToleranceFittingError("result exact replay changed")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    """Write a receipt once; a differing rewrite is refused rather than silently accepted."""

    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ToleranceFittingError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tolerance-aware interval fitting (B8).")
    parser.add_argument("--rows", help="JSON file holding the declared measured rows")
    parser.add_argument("--coverage-k", default="1")
    parser.add_argument("--exponent-probe", action="append", default=[])
    parser.add_argument("--controls", action="store_true", help="run the five mandatory controls")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        if not args.output:
            raise ToleranceFittingError("--validate-checked requires --output")
        value = json.loads(Path(args.output).read_text(encoding="utf-8"))
        if value.get("schema_version") == CONTROL_SCHEMA:
            validate_controls(value)
        else:
            validate_result(value)
        print(json.dumps({"validated": True}, indent=2))
        return 0
    if args.controls:
        controls = build_controls()
        if args.output:
            write_immutable(Path(args.output), controls)
        print(json.dumps({"counts": controls["counts"], "decision": controls["decision"]}, indent=2))
        return 0 if controls["decision"] == "PASS" else 2
    if not args.rows:
        raise ToleranceFittingError("--rows is required")
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    coverage = (
        payload.get("coverage_k", args.coverage_k)
        if isinstance(payload, Mapping)
        else args.coverage_k
    )
    result = fit_measured(rows, coverage_k=coverage, exponent_probes=args.exponent_probe)
    if args.output:
        write_immutable(Path(args.output), result)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "FEASIBLE_MINIMAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
