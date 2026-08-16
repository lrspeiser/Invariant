"""Holonomic (P-finite) recurrence guessing from exact terms.

The linear-recurrence statement kind is constant-coefficient: it can say
`a(n) = a(n-1) + a(n-2)` but not `(n+2)*a(n+1) = (4n+2)*a(n)`.  That gap is not
cosmetic — most of the classical combinatorial sequences (Catalan, Motzkin,
derangements, factorials) are *holonomic*: they satisfy a linear recurrence whose
coefficients are polynomials in `n`.  This module guesses such operators.

The method is brute force backwards: massive exact linear algebra over a declared
ladder of `(order, degree)` cells recovers exact structure from raw terms.  For each
cell the unknowns are the `(order+1)*(degree+1)` polynomial coefficients, the fitted
system uses only the minimum rows needed, and the resulting operator must then
annihilate **every remaining term exactly**.  An operator that merely interpolates its
own fitting window is refused; a cell whose fitting window plus mandatory holdout does
not fit in the data is skipped, never fitted.

Honesty rules, in the house style:

**Fitting is not discovery.**  A cell with `m` unknowns can always be solved on `m`
equations.  Acceptance requires exact annihilation on at least
``min_term_holdout`` equations the solver never saw.

**The ladder is declared and finite.**  `NO_ANNIHILATOR` means "no operator in this
declared ladder", never "the sequence is not holonomic".

**A guess is a guess.**  A surviving operator carries no proof certificate and no
novelty claim.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from math import gcd, lcm
from pathlib import Path
from typing import Any

from .basis_synthesis import _row_reduce
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-holonomic-guess-result-1.0"

#: Declared (order r, degree d) search cells.  The ordering follows parameter count
#: (r+1)*(d+1) with the adjacent-count tie order frozen exactly as declared here;
#: search returns the first accepting cell, so this ordering *is* the Occam
#: preference and must stay fixed.
LADDER: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
)

SYSTEM_CAPS = {
    "max_terms": 6000,
    "min_terms": 8,
    "min_term_holdout": 6,
    "max_order": 4,
    "max_degree": 4,
}

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "fitted_on_minimum_rows_only": True,
    "guess_is_a_proof": False,
    "interpolation_accepted_as_discovery": False,
    "ladder_is_declared_and_finite": True,
    "operator_must_annihilate_all_remaining_terms": True,
}


class HolonomicGuessError(ValueError):
    """Raised on malformed input, cap violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Exact polynomial helpers (ascending Fraction coefficient lists)
# ---------------------------------------------------------------------------


def _poly_trim(poly: list[Fraction]) -> list[Fraction]:
    while poly and poly[-1] == 0:
        poly.pop()
    return poly


def _poly_mod(numerator: list[Fraction], denominator: list[Fraction]) -> list[Fraction]:
    remainder = list(numerator)
    while len(remainder) >= len(denominator) and remainder:
        scale = remainder[-1] / denominator[-1]
        offset = len(remainder) - len(denominator)
        for index, coefficient in enumerate(denominator):
            remainder[offset + index] -= scale * coefficient
        _poly_trim(remainder)
    return remainder


def _poly_gcd(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    first, second = list(left), list(right)
    while second:
        first, second = second, _poly_mod(first, second)
    if first:
        leading = first[-1]
        first = [coefficient / leading for coefficient in first]
    return first


def _poly_divexact(numerator: list[Fraction], denominator: list[Fraction]) -> list[Fraction]:
    remainder = list(numerator)
    quotient = [Fraction(0)] * (len(numerator) - len(denominator) + 1)
    while len(remainder) >= len(denominator) and remainder:
        scale = remainder[-1] / denominator[-1]
        offset = len(remainder) - len(denominator)
        quotient[offset] = scale
        for index, coefficient in enumerate(denominator):
            remainder[offset + index] -= scale * coefficient
        _poly_trim(remainder)
    if remainder:
        raise HolonomicGuessError("polynomial content division left a remainder")
    return _poly_trim(quotient)


def _poly_eval_int(coefficients: Sequence[int], point: int) -> int:
    total = 0
    for coefficient in reversed(coefficients):
        total = total * point + coefficient
    return total


# ---------------------------------------------------------------------------
# Exact rational nullspace (extends basis_synthesis._solve_unique's elimination)
# ---------------------------------------------------------------------------


def _nullspace(matrix: Sequence[Sequence[Fraction]], width: int) -> list[list[Fraction]]:
    """Reduced-echelon basis of the exact rational nullspace of `matrix`."""

    work = [list(row) for row in matrix]
    zeros = [Fraction(0)] * len(work)
    _, pivot_columns, _ = _row_reduce(work, zeros)
    pivot_set = set(pivot_columns)
    basis: list[list[Fraction]] = []
    for free in range(width):
        if free in pivot_set:
            continue
        vector = [Fraction(0)] * width
        vector[free] = Fraction(1)
        for row_index, pivot in enumerate(pivot_columns):
            vector[pivot] = -work[row_index][free]
        basis.append(vector)
    return basis


# ---------------------------------------------------------------------------
# Operator normalization and rendering
# ---------------------------------------------------------------------------


def _vector_to_operator(
    vector: Sequence[Fraction], order: int, degree: int
) -> list[list[Fraction]]:
    width = degree + 1
    return [
        _poly_trim([vector[shift * width + power] for power in range(width)])
        for shift in range(order + 1)
    ]


def _normalize_operator(operator: Sequence[Sequence[Fraction]]) -> list[list[int]]:
    """Canonical integer form: polynomial-content-free, integer-content-free,
    leading coefficient of the top-shift polynomial positive."""

    nonzero = [list(poly) for poly in operator if poly]
    if not nonzero:
        raise HolonomicGuessError("cannot normalize the zero operator")
    content = nonzero[0]
    for poly in nonzero[1:]:
        if len(content) == 1:
            break
        content = _poly_gcd(content, poly)
    reduced = [
        _poly_divexact(list(poly), content) if poly and len(content) > 1 else list(poly)
        for poly in operator
    ]
    scale = 1
    for poly in reduced:
        for coefficient in poly:
            scale = lcm(scale, coefficient.denominator)
    integered = [[int(coefficient * scale) for coefficient in poly] for poly in reduced]
    divisor = 0
    for poly in integered:
        for coefficient in poly:
            divisor = gcd(divisor, abs(coefficient))
    integered = [[coefficient // divisor for coefficient in poly] for poly in integered]
    top = integered[-1]
    if top and top[-1] < 0:
        integered = [[-coefficient for coefficient in poly] for poly in integered]
    return integered


def _poly_text(coefficients: Sequence[int], latex: bool = False) -> str:
    parts: list[str] = []
    for power in range(len(coefficients) - 1, -1, -1):
        coefficient = coefficients[power]
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        if power == 0:
            body = str(magnitude)
        else:
            variable = "n" if power == 1 else (f"n^{{{power}}}" if latex else f"n^{power}")
            head = "" if magnitude == 1 else (f"{magnitude} " if latex else f"{magnitude}*")
            body = f"{head}{variable}"
        if not parts:
            parts.append(body if coefficient > 0 else f"-{body}")
        else:
            parts.append(f"+ {body}" if coefficient > 0 else f"- {body}")
    return " ".join(parts) if parts else "0"


def _shift_name(shift: int) -> str:
    return "a(n)" if shift == 0 else f"a(n+{shift})"


def _term_text(coefficients: Sequence[int], shift: int, latex: bool) -> tuple[str, str]:
    """(sign, body) for one `P_i(n)*a(n+i)` term; sign is '+' or '-'."""

    join = r"\," if latex else "*"
    name = _shift_name(shift)
    nonzero = [c for c in coefficients if c != 0]
    if len(nonzero) == 1 and len(coefficients) == 1:
        constant = coefficients[0]
        magnitude = abs(constant)
        body = name if magnitude == 1 else f"{magnitude}{join}{name}"
        return ("+" if constant > 0 else "-", body)
    if all(c <= 0 for c in coefficients):
        flipped = [-c for c in coefficients]
        return ("-", f"({_poly_text(flipped, latex)}){join}{name}")
    return ("+", f"({_poly_text(list(coefficients), latex)}){join}{name}")


def render_operator(
    operator: Sequence[Sequence[int]], start_point: int, latex: bool = False
) -> str:
    parts: list[str] = []
    for shift in range(len(operator) - 1, -1, -1):
        coefficients = operator[shift]
        if not coefficients or all(c == 0 for c in coefficients):
            continue
        sign, body = _term_text(coefficients, shift, latex)
        if not parts:
            parts.append(body if sign == "+" else f"-{body}")
        else:
            parts.append(f"{sign} {body}")
    geq = r"\ge" if latex else ">="
    return f"{' '.join(parts)} = 0 for n {geq} {start_point}"


# ---------------------------------------------------------------------------
# The guesser
# ---------------------------------------------------------------------------


def _parse_values(values: Sequence[Any]) -> list[Fraction]:
    parsed: list[Fraction] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise HolonomicGuessError("terms must be exact integers or Fractions")
        parsed.append(Fraction(value))
    return parsed


def _operator_complexity(operator: Sequence[Sequence[int]]) -> tuple[Any, ...]:
    max_degree = max((len(poly) - 1 for poly in operator if poly), default=-1)
    nonzero = sum(1 for poly in operator for coefficient in poly if coefficient != 0)
    flattened = tuple(tuple(poly) for poly in operator)
    return (max_degree, nonzero, flattened)


def guess_recurrence(
    values: Sequence[Any],
    *,
    start_index: int = 0,
    min_term_holdout: int | None = None,
    ladder: Sequence[tuple[int, int]] = LADDER,
) -> dict[str, Any]:
    """Search the declared ladder for the first exactly-confirmed annihilator.

    Returns ``{"decision": "OPERATOR_FOUND" | "NO_ANNIHILATOR", "operator": ...,
    "ladder_trace": [...]}``.  Every number involved is exact; there is no float
    anywhere in this search.
    """

    terms = _parse_values(values)
    count = len(terms)
    if count > SYSTEM_CAPS["max_terms"]:
        raise HolonomicGuessError("term count exceeds cap")
    if count < SYSTEM_CAPS["min_terms"]:
        raise HolonomicGuessError("too few terms to guess against")
    holdout_floor = (
        SYSTEM_CAPS["min_term_holdout"] if min_term_holdout is None else min_term_holdout
    )
    if holdout_floor < 1:
        raise HolonomicGuessError("min_term_holdout must be at least one")

    trace: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for order, degree in ladder:
        if order > SYSTEM_CAPS["max_order"] or degree > SYSTEM_CAPS["max_degree"]:
            raise HolonomicGuessError("ladder cell exceeds declared caps")
        width = (order + 1) * (degree + 1)
        equations = count - order
        cell: dict[str, Any] = {
            "order": order,
            "degree": degree,
            "parameter_count": width,
        }
        if equations < width + holdout_floor:
            trace.append(
                {
                    **cell,
                    "status": "SKIPPED_INSUFFICIENT_TERMS",
                    "equations_available": equations,
                    "equations_required": width + holdout_floor,
                }
            )
            continue

        # Fit on exactly the minimum rows: the first `width` recurrence instances.
        # Unknown ordering is shift-major, degree-minor: column (i, j) multiplies
        # n^j * a(n+i).
        matrix: list[list[Fraction]] = []
        for base in range(width):
            point = start_index + base
            row: list[Fraction] = []
            for shift in range(order + 1):
                for power in range(degree + 1):
                    row.append(Fraction(point) ** power * terms[base + shift])
            matrix.append(row)

        basis = _nullspace(matrix, width)
        if not basis:
            trace.append({**cell, "status": "EMPTY_NULLSPACE", "fitted_equations": width})
            continue

        candidates: list[list[list[int]]] = []
        first_failure: dict[str, Any] | None = None
        ineligible = 0
        for vector in basis:
            operator = _vector_to_operator(vector, order, degree)
            if not operator[-1] or not operator[0]:
                ineligible += 1
                continue
            normalized = _normalize_operator(operator)
            failure: dict[str, Any] | None = None
            for base in range(width, equations):
                point = start_index + base
                residual = Fraction(0)
                for shift in range(order + 1):
                    residual += _poly_eval_int(normalized[shift], point) * terms[base + shift]
                if residual != 0:
                    failure = {
                        "base_point": point,
                        "residual": {
                            "numerator": residual.numerator,
                            "denominator": residual.denominator,
                        },
                    }
                    break
            if failure is None:
                candidates.append(normalized)
            elif first_failure is None:
                first_failure = failure

        if candidates:
            unique = list(
                dict.fromkeys(tuple(tuple(poly) for poly in item) for item in candidates)
            )
            best = min(unique, key=lambda item: _operator_complexity(item))
            chosen = [list(poly) for poly in best]
            accepted = {
                **cell,
                "status": "ACCEPTED",
                "fitted_equations": width,
                "verified_equations": equations - width,
                "nullspace_dimension": len(basis),
                "operator": chosen,
            }
            trace.append(accepted)
            break
        trace.append(
            {
                **cell,
                "status": "NULLSPACE_FAILS_HOLDOUT" if first_failure else "NO_ELIGIBLE_VECTOR",
                "fitted_equations": width,
                "nullspace_dimension": len(basis),
                "ineligible_vectors": ineligible,
                "first_failure": first_failure,
            }
        )

    if accepted is None:
        return {"decision": "NO_ANNIHILATOR", "operator": None, "ladder_trace": trace}
    operator = accepted["operator"]
    return {
        "decision": "OPERATOR_FOUND",
        "operator": {
            "order": accepted["order"],
            "degree": accepted["degree"],
            "parameter_count": accepted["parameter_count"],
            "coefficients": operator,
            "fitted_equations": accepted["fitted_equations"],
            "verified_equations": accepted["verified_equations"],
            "nullspace_dimension": accepted["nullspace_dimension"],
            "start_point": start_index,
            "statement": render_operator(operator, start_index),
            "latex": render_operator(operator, start_index, latex=True),
        },
        "ladder_trace": trace,
    }


# ---------------------------------------------------------------------------
# Sealed receipts
# ---------------------------------------------------------------------------


def _parse_rows(rows: Any) -> list[tuple[int, Fraction]]:
    if not isinstance(rows, list) or not rows:
        raise HolonomicGuessError("rows must be a non-empty list")
    if len(rows) > SYSTEM_CAPS["max_terms"]:
        raise HolonomicGuessError("row count exceeds cap")
    parsed: list[tuple[int, Fraction]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"point", "value"}:
            raise HolonomicGuessError("each row needs exactly point and value")
        point = row["point"]
        if not isinstance(point, int) or isinstance(point, bool):
            raise HolonomicGuessError("point must be an integer")
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
        raise HolonomicGuessError("value must be an integer or exact rational object")
    parsed.sort(key=lambda item: item[0])
    for (left, _), (right, _) in pairwise(parsed):
        if right != left + 1:
            raise HolonomicGuessError("points must be consecutive integers")
    return parsed


def guess_receipt(rows: Any, sequence_label: str = "") -> dict[str, Any]:
    parsed = _parse_rows(rows)
    result = guess_recurrence(
        [value for _, value in parsed], start_index=parsed[0][0]
    )
    body: dict[str, Any] = {
        "claims": CLAIMS,
        "decision": result["decision"],
        "ladder": [list(cell) for cell in LADDER],
        "ladder_trace": result["ladder_trace"],
        "operator": result["operator"],
        "public_rows": [
            {
                "point": point,
                "value": {"numerator": value.numerator, "denominator": value.denominator},
            }
            for point, value in parsed
        ],
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Holonomic (P-finite) recurrence guessing over a declared finite ladder of "
            "(order, degree) cells. OPERATOR_FOUND means the first ladder cell whose exact "
            "rational nullspace produced an operator that annihilates every remaining term "
            "exactly; the operator is a guess, not a proof, and NO_ANNIHILATOR means 'not "
            "in this declared ladder', never 'the sequence is not holonomic'."
        ),
        "sequence_label": sequence_label,
        "system_caps": SYSTEM_CAPS,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Reject tamper or drift by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise HolonomicGuessError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise HolonomicGuessError("result seal changed")
    rows = [
        {"point": row["point"], "value": row["value"]} for row in value.get("public_rows", [])
    ]
    if dict(value) != guess_receipt(rows, value.get("sequence_label", "")):
        raise HolonomicGuessError("result exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise HolonomicGuessError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="Holonomic recurrence guesser.")
    parser.add_argument("--rows", required=True, help="JSON file holding the public rows")
    parser.add_argument("--label", default="")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    result = guess_receipt(rows, args.label)
    if args.output:
        _write_immutable(Path(args.output), result)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "OPERATOR_FOUND" else 2


if __name__ == "__main__":
    raise SystemExit(main())
