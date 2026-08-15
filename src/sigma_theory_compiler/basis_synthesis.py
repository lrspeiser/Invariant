"""B1 — exact basis synthesis from public data rows.

The existing exact solver (`formula_discovery_job`, `solver.kind =
exact_linear_basis_v1`) requires the caller to declare the basis.  That makes the
caller's choice of representation the discovery and reduces this module's job to
bookkeeping.  This module removes the per-problem basis declaration: it takes only
public `(n, value)` rows and searches a declared, enumerable ladder of basis
families for the *simplest* basis that both solves exactly and predicts rows it
never saw.

Two rules keep this honest.

**Interpolation is not discovery.**  A basis with `k` terms can always be fitted to
`k` points.  The solve therefore consumes only as many rows as it needs to reach
full rank; every remaining row is untouched holdout that the solution must predict
exactly.  A candidate with zero confirmations is refused.

**Minimality is certified, not asserted.**  The ladder is ordered by (term count,
family rank).  A result carries exact rejection evidence for every strictly simpler
ladder entry, so "this is the simplest structure that explains the data" is a
checked statement rather than an artifact of search order.

Claim boundary: this moves basis declaration from per-problem to per-system.  It is
not unbounded representation invention.  A target outside the declared ladder stays
unreachable, and a BLOCK means "not in this declared ladder", never "impossible".
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb, factorial
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

LADDER_SCHEMA = "invariant-basis-synthesis-ladder-1.0"
RESULT_SCHEMA = "invariant-basis-synthesis-result-1.0"

#: Hard bounds.  Exceeding any of these is an error, never a silent truncation.
SYSTEM_CAPS = {
    "max_rows": 64,
    "max_basis_terms": 8,
    "max_abs_point": 64,
    "max_numerator_bits": 512,
    "min_confirmations": 2,
}

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "holdout_confirmation_required": True,
    "interpolation_accepted_as_discovery": False,
    "ladder_is_declared_and_finite": True,
    "minimality_is_certified_against_simpler_entries": True,
    "unbounded_representation_invention": False,
}


class BasisSynthesisError(ValueError):
    """Raised when synthesis input, caps, or receipt integrity are violated."""


# ---------------------------------------------------------------------------
# Term algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Term:
    """One basis term.  `source` is the canonical display form."""

    family: str
    params: tuple[int, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "params": list(self.params), "source": self.source}


def evaluate_term(term: Term, point: int) -> Fraction | None:
    """Exact value of `term` at `point`, or None where the term is undefined."""

    family, params = term.family, term.params
    if family == "monomial":
        (degree,) = params
        return Fraction(point) ** degree if degree >= 0 or point != 0 else None
    if family == "alternating_monomial":
        (degree,) = params
        return Fraction(-1 if point % 2 else 1) * Fraction(point) ** degree
    if family == "geometric":
        (base,) = params
        return Fraction(base) ** point
    if family == "geometric_monomial":
        base, degree = params
        return Fraction(base) ** point * Fraction(point) ** degree
    if family == "reciprocal":
        (degree,) = params
        return None if point == 0 else Fraction(1, point**degree)
    if family == "shifted_reciprocal":
        (degree,) = params
        return None if point == -1 else Fraction(1, (point + 1) ** degree)
    if family == "harmonic":
        if point < 0:
            return None
        total = Fraction(0)
        for index in range(1, point + 1):
            total += Fraction(1, index)
        return total
    if family == "factorial":
        return None if point < 0 else Fraction(factorial(point))
    if family == "binomial":
        (choose,) = params
        return None if point < 0 else Fraction(comb(point, choose))
    raise BasisSynthesisError(f"unsupported term family: {family}")


def _monomials(degree: int) -> tuple[Term, ...]:
    return tuple(
        Term("monomial", (index,), "1" if index == 0 else f"n^{index}")
        for index in range(degree + 1)
    )


def _ladder_entries() -> tuple[dict[str, Any], ...]:
    """The declared, finite, deterministically ordered basis ladder.

    Ordering is (term count, family rank, parameter).  Search returns the first
    qualifying entry, so ordering *is* the Occam preference and must stay frozen.
    """

    entries: list[dict[str, Any]] = []

    def add(family_id: str, rank: int, terms: Sequence[Term]) -> None:
        entries.append({"family_id": family_id, "family_rank": rank, "terms": tuple(terms)})

    add("constant", 0, _monomials(0))
    for degree in range(1, 6):
        add(f"polynomial_{degree}", 1, _monomials(degree))
    for base in (2, 3):
        add(
            f"geometric_{base}",
            2,
            (Term("monomial", (0,), "1"), Term("geometric", (base,), f"{base}^n")),
        )
    for base in (2, 3):
        add(
            f"geometric_linear_{base}",
            3,
            (
                Term("monomial", (0,), "1"),
                Term("geometric", (base,), f"{base}^n"),
                Term("geometric_monomial", (base, 1), f"n*{base}^n"),
            ),
        )
    for degree in range(1, 4):
        add(
            f"alternating_{degree}",
            4,
            (Term("monomial", (0,), "1"),)
            + tuple(
                Term(
                    "alternating_monomial",
                    (index,),
                    "(-1)^n" if index == 0 else f"(-1)^n*n^{index}",
                )
                for index in range(degree)
            ),
        )
    add(
        "harmonic",
        5,
        (Term("monomial", (0,), "1"), Term("harmonic", (), "H_n")),
    )
    add(
        "harmonic_linear",
        5,
        (
            Term("monomial", (0,), "1"),
            Term("monomial", (1,), "n"),
            Term("harmonic", (), "H_n"),
        ),
    )
    for degree in range(1, 3):
        add(
            f"reciprocal_{degree}",
            6,
            (Term("monomial", (0,), "1"),)
            + tuple(
                Term("reciprocal", (index,), f"1/n^{index}") for index in range(1, degree + 1)
            ),
        )
    add(
        "shifted_reciprocal_1",
        6,
        (
            Term("monomial", (0,), "1"),
            Term("shifted_reciprocal", (1,), "1/(n+1)"),
        ),
    )
    add(
        "factorial",
        7,
        (Term("monomial", (0,), "1"), Term("factorial", (), "n!")),
    )
    for choose in (2, 3):
        add(
            f"binomial_{choose}",
            7,
            (Term("monomial", (0,), "1"), Term("binomial", (choose,), f"C(n,{choose})")),
        )

    entries.sort(key=lambda entry: (len(entry["terms"]), entry["family_rank"], entry["family_id"]))
    return tuple(entries)


LADDER = _ladder_entries()


# ---------------------------------------------------------------------------
# Exact rational linear algebra
# ---------------------------------------------------------------------------


def _row_reduce(
    matrix: list[list[Fraction]], vector: list[Fraction]
) -> tuple[int, list[int], bool]:
    """Fraction-exact Gauss-Jordan.  Returns (rank, pivot_rows, consistent)."""

    rows, columns = len(matrix), (len(matrix[0]) if matrix else 0)
    pivot_rows: list[int] = []
    row = 0
    for column in range(columns):
        pivot = next((index for index in range(row, rows) if matrix[index][column] != 0), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        vector[row], vector[pivot] = vector[pivot], vector[row]
        scale = matrix[row][column]
        matrix[row] = [value / scale for value in matrix[row]]
        vector[row] = vector[row] / scale
        for index in range(rows):
            if index != row and matrix[index][column] != 0:
                factor = matrix[index][column]
                matrix[index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(matrix[index], matrix[row], strict=True)
                ]
                vector[index] = vector[index] - factor * vector[row]
        pivot_rows.append(column)
        row += 1
        if row == rows:
            break
    consistent = all(
        vector[index] == 0 for index in range(row, rows) if all(v == 0 for v in matrix[index])
    )
    return row, pivot_rows, consistent


def _solve_unique(
    matrix: list[list[Fraction]], vector: list[Fraction], width: int
) -> list[Fraction] | None:
    """Return the unique exact solution, or None when it is not unique."""

    work = [list(row) for row in matrix]
    target = list(vector)
    rank, pivots, consistent = _row_reduce(work, target)
    if not consistent or rank != width or len(pivots) != width:
        return None
    solution = [Fraction(0)] * width
    for index, column in enumerate(pivots):
        solution[column] = target[index]
    return solution


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _parse_rows(rows: Any) -> list[tuple[int, Fraction]]:
    if not isinstance(rows, list) or not rows:
        raise BasisSynthesisError("rows must be a non-empty list")
    if len(rows) > SYSTEM_CAPS["max_rows"]:
        raise BasisSynthesisError("row count exceeds cap")
    parsed: list[tuple[int, Fraction]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"point", "value"}:
            raise BasisSynthesisError("each row needs exactly point and value")
        point = row["point"]
        if not isinstance(point, int) or isinstance(point, bool):
            raise BasisSynthesisError("point must be an integer")
        if abs(point) > SYSTEM_CAPS["max_abs_point"]:
            raise BasisSynthesisError("point exceeds cap")
        if point in seen:
            raise BasisSynthesisError("duplicate point")
        seen.add(point)
        value = row["value"]
        if isinstance(value, int) and not isinstance(value, bool):
            parsed.append((point, Fraction(value)))
            continue
        if (
            isinstance(value, Mapping)
            and set(value) == {"numerator", "denominator"}
            and isinstance(value["numerator"], int)
            and isinstance(value["denominator"], int)
            and not isinstance(value["numerator"], bool)
            and value["denominator"] != 0
        ):
            parsed.append((point, Fraction(value["numerator"], value["denominator"])))
            continue
        raise BasisSynthesisError("value must be an integer or exact rational object")
    parsed.sort(key=lambda item: item[0])
    return parsed


def _try_entry(
    entry: Mapping[str, Any], rows: Sequence[tuple[int, Fraction]]
) -> dict[str, Any]:
    """Attempt one ladder entry.  Never raises on ordinary failure."""

    terms: tuple[Term, ...] = entry["terms"]
    width = len(terms)
    outcome: dict[str, Any] = {
        "family_id": entry["family_id"],
        "term_count": width,
        "terms": [term.source for term in terms],
    }
    if width > SYSTEM_CAPS["max_basis_terms"]:
        return {**outcome, "status": "SKIP", "reason": "basis_exceeds_term_cap"}

    evaluated: list[tuple[int, list[Fraction], Fraction]] = []
    for point, value in rows:
        columns: list[Fraction] = []
        for term in terms:
            cell = evaluate_term(term, point)
            if cell is None:
                columns = []
                break
            columns.append(cell)
        if columns:
            evaluated.append((point, columns, value))
    if len(evaluated) < width:
        return {**outcome, "status": "SKIP", "reason": "insufficient_defined_rows"}

    # Consume rows in order, keeping only those that raise the rank.  Everything
    # left over is untouched holdout.
    fit_matrix: list[list[Fraction]] = []
    fit_vector: list[Fraction] = []
    fit_points: list[int] = []
    rank = 0
    for point, columns, value in evaluated:
        trial = [list(row) for row in fit_matrix] + [list(columns)]
        trial_vector = list(fit_vector) + [value]
        trial_rank, _, _ = _row_reduce(trial, trial_vector)
        if trial_rank > rank:
            fit_matrix.append(list(columns))
            fit_vector.append(value)
            fit_points.append(point)
            rank = trial_rank
        if rank == width:
            break
    if rank < width:
        return {**outcome, "status": "REJECT", "reason": "rank_deficient_on_public_rows"}

    solution = _solve_unique(fit_matrix, fit_vector, width)
    if solution is None:
        return {**outcome, "status": "REJECT", "reason": "no_unique_exact_solution"}

    holdout = [item for item in evaluated if item[0] not in set(fit_points)]
    if len(holdout) < SYSTEM_CAPS["min_confirmations"]:
        return {
            **outcome,
            "status": "BLOCK",
            "reason": "insufficient_holdout_confirmations",
            "confirmations": len(holdout),
        }

    mismatch: dict[str, Any] | None = None
    for point, columns, value in holdout:
        predicted = sum(
            (coefficient * cell for coefficient, cell in zip(solution, columns, strict=True)),
            Fraction(0),
        )
        if predicted != value:
            mismatch = {
                "point": point,
                "predicted": _fraction_data(predicted),
                "observed": _fraction_data(value),
                "residual": _fraction_data(predicted - value),
            }
            break
    if mismatch is not None:
        return {
            **outcome,
            "status": "REJECT",
            "reason": "holdout_counterexample",
            "counterexample": mismatch,
        }

    return {
        **outcome,
        "status": "PASS",
        "reason": "unique_exact_solution_confirmed_on_holdout",
        "coefficients": [_fraction_data(value) for value in solution],
        "fit_points": fit_points,
        "confirmations": len(holdout),
        "compression_ratio": {"rows": len(evaluated), "terms": width},
        "expression": _render(terms, solution),
    }


def fit_entry(
    entry: Mapping[str, Any], rows: Sequence[tuple[int, Fraction]]
) -> tuple[list[Fraction], list[int]] | None:
    """Solve one ladder entry on its minimal fit rows, ignoring holdout agreement.

    B1 itself never accepts such a fit; the holdout gate is what makes a result
    meaningful.  B7 needs the raw solution so it can take a residual and repair the
    *structure* rather than the coefficients.
    """

    terms: tuple[Term, ...] = entry["terms"]
    width = len(terms)
    evaluated: list[tuple[int, list[Fraction], Fraction]] = []
    for point, value in rows:
        columns: list[Fraction] = []
        for term in terms:
            cell = evaluate_term(term, point)
            if cell is None:
                columns = []
                break
            columns.append(cell)
        if columns:
            evaluated.append((point, columns, value))
    if len(evaluated) < width:
        return None
    fit_matrix: list[list[Fraction]] = []
    fit_vector: list[Fraction] = []
    fit_points: list[int] = []
    rank = 0
    for point, columns, value in evaluated:
        trial = [list(row) for row in fit_matrix] + [list(columns)]
        trial_vector = list(fit_vector) + [value]
        trial_rank, _, _ = _row_reduce(trial, trial_vector)
        if trial_rank > rank:
            fit_matrix.append(list(columns))
            fit_vector.append(value)
            fit_points.append(point)
            rank = trial_rank
        if rank == width:
            break
    if rank < width:
        return None
    solution = _solve_unique(fit_matrix, fit_vector, width)
    return None if solution is None else (solution, fit_points)


def evaluate_basis(
    terms: Sequence[Term], solution: Sequence[Fraction], point: int
) -> Fraction | None:
    """Exact value of a fitted basis at `point`, or None where any term is undefined."""

    total = Fraction(0)
    for coefficient, term in zip(solution, terms, strict=True):
        cell = evaluate_term(term, point)
        if cell is None:
            return None
        total += coefficient * cell
    return total


def render_basis(terms: Sequence[Term], solution: Sequence[Fraction]) -> str:
    """Public canonical rendering of a fitted basis."""

    return _render(terms, solution)


def _render(terms: Sequence[Term], solution: Sequence[Fraction]) -> str:
    parts: list[str] = []
    for coefficient, term in zip(solution, terms, strict=True):
        if coefficient == 0:
            continue
        if term.source == "1":
            parts.append(str(coefficient))
        elif coefficient == 1:
            parts.append(term.source)
        elif coefficient == -1:
            parts.append(f"-{term.source}")
        else:
            parts.append(f"{coefficient}*{term.source}")
    return " + ".join(parts).replace("+ -", "- ") if parts else "0"


def synthesize_basis(rows: Any) -> dict[str, Any]:
    """Search the declared ladder for the simplest confirmed basis.

    Returns a sealed result.  `decision` is PASS with a certified-minimal basis,
    or BLOCK when no ladder entry qualifies.  BLOCK is never a guess.
    """

    parsed = _parse_rows(rows)
    rejected: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for entry in LADDER:
        outcome = _try_entry(entry, parsed)
        if outcome["status"] == "PASS":
            accepted = outcome
            break
        rejected.append(outcome)

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "counts": {
            "ladder_entries": len(LADDER),
            "entries_examined": len(rejected) + (1 if accepted else 0),
            "entries_rejected_before_acceptance": len(rejected),
            "public_rows": len(parsed),
        },
        "decision": "PASS" if accepted else "BLOCK",
        "first_blocker": None if accepted else "no_qualifying_basis_in_declared_ladder",
        "ladder_schema": LADDER_SCHEMA,
        "minimality_certificate": {
            "ordering": "term_count_then_family_rank_then_family_id",
            "strictly_simpler_entries_rejected": rejected,
        },
        "public_rows": [
            {"point": point, "value": _fraction_data(value)} for point, value in parsed
        ],
        "result": accepted,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Exact basis synthesis over a declared finite ladder of basis families. "
            "PASS means the returned basis is the simplest ladder entry that solves "
            "exactly and predicts every untouched holdout row. It does not establish "
            "novelty, scientific significance, or reachability of targets outside "
            "this declared ladder."
        ),
        "system_caps": SYSTEM_CAPS,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(value: Mapping[str, Any]) -> None:
    """Reject tamper or drift by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise BasisSynthesisError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise BasisSynthesisError("result seal changed")
    rows = [
        {"point": row["point"], "value": row["value"]} for row in value.get("public_rows", [])
    ]
    if dict(value) != synthesize_basis(rows):
        raise BasisSynthesisError("result exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise BasisSynthesisError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact basis synthesis (B1).")
    parser.add_argument("--rows", required=True, help="JSON file holding the public rows")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    if args.validate_checked:
        validate_result(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = synthesize_basis(rows)
    if args.output:
        _write_immutable(Path(args.output), result)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
