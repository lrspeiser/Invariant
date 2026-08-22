"""A construction problem with an exact computable objective, in the funsearch_loop style.

Every problem the loop declared before this one asks a program to *predict numbers*: fit a
sequence, fit a response curve, converge on a constant.  Prediction problems have a soft
objective -- how close is close -- and a soft objective is where a rediscovery hides, because
"close" cannot separate a derivation from a rearrangement of something recalled.

This module declares the other shape, and it is the shape FunSearch and AlphaEvolve actually
won at: **build a combinatorial object, and be scored by its exactly measured size.**  The
proposer writes a program that *constructs* a finite set; the set is verified exhaustively in
exact integer arithmetic; and the score is the verified cardinality.  Nothing about the score
is estimated, sampled, or claimed.

**The declared object.**  Points of ``F_3^n`` are encoded as integers ``0 <= a < 3^n``, digit
``k`` of ``a`` being ``(a // 3**k) % 3``.  A triple of three distinct points is *forbidden*
when every digit position sums to a multiple of three.  A program must return a list of
distinct points containing no forbidden triple, and longer lists score higher.  Under the
usual names this is a cap in ``AG(n, 3)`` -- a subset with no three points in arithmetic
progression -- but the proposer is never told that, and the domain names sit in the
vocabulary guard.

**The verifier is exhaustive.**  Every one of the ``C(m, 2)`` unordered pairs is examined and
the third point of the line through it is looked up in an exact table.  A forbidden triple
lying inside the returned list is witnessed by any of its three pairs, so scanning all pairs
finds every one; nothing is sampled.  The certificate carries the pair count the loop actually
performed next to ``m (m - 1) / 2`` computed independently, so "exhaustive" is a checkable
number rather than an adjective.

**An invalid set scores exactly zero.**  Not "scores less" -- zero.  A duplicate point, a
point out of range, a non-integer, a list longer than the declared cap, or one forbidden
triple anywhere sends quality to ``0/1`` and therefore the final score to ``0/1``.  The score
is computed from the points the program actually returned; a program cannot claim a size.

**The score is an exact rational.**  ``quality = verified_cardinality / elementary_upper_bound``
where the denominator is not the answer but an elementary bound this module *proves by
exhibition*: a parallel class of ``3^(n-1)`` lines partitions the ``3^n`` points, a valid set
meets each of those lines at most twice, so no valid set exceeds ``2 * 3^(n-1)``.  The
partition is verified point by point and sealed as a certificate.  Every number on the
certificate path is an integer or an exact ``p/q`` string; no float reaches a hash.

**The sealed record is off the proposal path.**  :data:`SEALED_RECORDS` holds, per dimension,
the best cardinality in the literature and -- where one is cheap to verify -- an explicit
extremal witness.  Two things consume it, both after scoring.  The *record gate* is the integer
comparison ``verified_cardinality > sealed_cardinality``, and it is the only statement in this
module that could ever be a discovery.  The *novelty channel* zeroes any construction lying in
the declared monomial orbit of the sealed witness: the image of the witness under the monomial
affine group ``F_3^n : ({1,2}^n : S_n)``, enumerated exhaustively (31,104 group elements at
``n = 4``) and cached, so orbit membership is exact set membership rather than a heuristic.

**Blindness on the digit channel.**  The vocabulary guard the loop already runs tokenises on
letter runs, so it cannot see a sealed *number*.  :func:`numeral_violations` is the same guard
on digit runs, and an assembled prompt carrying the sealed record cardinality or the elementary
bound is refused rather than sanitised.  The problem declaration is checked against it at
construction time, so the base prompt can never be the thing that leaks.

**Honest limits, stated as claims.**  The monomial group is a proper subgroup of ``AGL(n, 3)``,
so a construction outside the orbit may still be affinely equivalent to the witness: a nonzero
multiplier means "not in the declared monomial orbit", never "new".  Only the record gate can
say new, and every dimension declared here carries a record that is already a proved maximum,
so the honest expected outcome is that the gate never fires.  That is the mechanism working.
The machinery is built so that when it is pointed at a dimension whose record is open, a fired
gate is an exactly verified integer recomputable from the sealed point list alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from .funsearch_loop import (
    SANDBOX_FAILURE_REASONS,
    ClaudeCliProposer,
    FunSearchError,
    LoopConfig,
    MockMutationProposer,
    ProposalCall,
    ProposerCallFailed,
    ProposerUnavailable,
    SandboxBudget,
    SandboxOutcome,
    SpendGovernor,
    classify_failure,
    run_hostile_suite,
    run_in_sandbox,
    vocabulary_violations,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RECEIPT_SCHEMA = "invariant-capset-construction-search-1.0"


class CapsetError(FunSearchError):
    """Raised on a malformed declaration, a guard violation, or receipt tamper."""


CLAIMS = {
    "an_invalid_construction_scores_exactly_zero": True,
    "monomial_orbit_absence_establishes_novelty": False,
    "monomial_orbit_membership_establishes_reproduction": True,
    "no_float_reaches_a_sealed_number": True,
    "score_is_the_verified_size_never_a_claimed_one": True,
    "sealed_record_is_read_only_after_scoring": True,
    "verification_is_exhaustive_not_sampled": True,
}

SCOPE = (
    "A construction problem scored by exact measurement. A proposer writes a program that "
    "returns a list of integers encoding points of F_3^n; the list is verified exhaustively "
    "over every unordered pair in exact integer arithmetic; the score is the verified "
    "cardinality over an elementary upper bound proved by exhibiting a partition. An invalid "
    "list scores zero. The sealed record is consumed only after scoring, by an integer "
    "comparison and by a monomial-orbit membership test. Neither is a novelty claim on its "
    "own: the orbit test uses a proper subgroup of the affine group, so it can miss an "
    "equivalence but can never manufacture one."
)


# ---------------------------------------------------------------------------
# 1. Exact affine arithmetic over F_3
# ---------------------------------------------------------------------------

#: Above this dimension the third-point lookup table is not materialised and the digit loop is
#: used instead.  ``3^5`` squared is 59,049 entries; ``3^7`` squared would be 4.8 million.
MAX_TABLE_DIMENSION = 5

#: The declared dimension window.  Everything outside it is a typed refusal, not a slow run.
MAX_DIMENSION = 8

_DIGIT_CACHE: dict[int, tuple[tuple[int, ...], ...]] = {}
_THIRD_CACHE: dict[int, tuple[int, ...]] = {}
_LINE_CACHE: dict[int, tuple[tuple[int, int, int], ...]] = {}


def points_in(dimension: int) -> int:
    """``3 ** dimension`` -- the number of points, as an exact integer."""

    value = int(dimension)
    if not 1 <= value <= MAX_DIMENSION:
        raise CapsetError(f"dimension outside the declared range: {dimension}")
    return 3**value


def digit_table(dimension: int) -> tuple[tuple[int, ...], ...]:
    """``digit_table(n)[a][k]`` is digit ``k`` of point ``a``.  Built once per dimension."""

    cached = _DIGIT_CACHE.get(dimension)
    if cached is not None:
        return cached
    size = points_in(dimension)
    rows: list[tuple[int, ...]] = []
    for code in range(size):
        rest = code
        row: list[int] = []
        for _ in range(dimension):
            row.append(rest % 3)
            rest //= 3
        rows.append(tuple(row))
    table = tuple(rows)
    _DIGIT_CACHE[dimension] = table
    return table


def _third_by_digits(left: Sequence[int], right: Sequence[int], dimension: int) -> int:
    value = 0
    weight = 1
    for index in range(dimension):
        value += ((3 - (left[index] + right[index]) % 3) % 3) * weight
        weight *= 3
    return value


def _third_table(dimension: int) -> tuple[int, ...]:
    cached = _THIRD_CACHE.get(dimension)
    if cached is not None:
        return cached
    digits = digit_table(dimension)
    size = len(digits)
    flat: list[int] = []
    for left in range(size):
        row_left = digits[left]
        for right in range(size):
            flat.append(_third_by_digits(row_left, digits[right], dimension))
    table = tuple(flat)
    _THIRD_CACHE[dimension] = table
    return table


def third_point(left: int, right: int, dimension: int) -> int:
    """The unique ``c`` with ``left + right + c == 0`` in every digit position.

    For distinct ``left`` and ``right`` the three points are exactly the line through them and
    ``c`` differs from both: ``c == left`` would force ``right == left``.
    """

    size = points_in(dimension)
    if not (0 <= left < size and 0 <= right < size):
        raise CapsetError("point outside the declared space")
    if dimension <= MAX_TABLE_DIMENSION:
        return _third_table(dimension)[left * size + right]
    digits = digit_table(dimension)
    return _third_by_digits(digits[left], digits[right], dimension)


def lines_in(dimension: int) -> int:
    """``3^n (3^n - 1) / 6`` -- the exact number of lines, as an integer."""

    size = points_in(dimension)
    return size * (size - 1) // 6


def iter_lines(dimension: int) -> Iterator[tuple[int, int, int]]:
    """Every line once, as a sorted triple, in lexicographic order.

    A pair ``(left, right)`` with ``left < right`` names the line through it; emitting only
    when that pair is the two smallest members of the triple visits each line exactly once, so
    no de-duplication set is needed and the enumeration streams.
    """

    size = points_in(dimension)
    for left in range(size):
        for right in range(left + 1, size):
            other = third_point(left, right, dimension)
            if other > right:
                yield (left, right, other)


def enumerate_lines(dimension: int) -> tuple[tuple[int, int, int], ...]:
    """Every line, materialised and cached.  Exhaustive by construction."""

    cached = _LINE_CACHE.get(dimension)
    if cached is not None:
        return cached
    lines = tuple(iter_lines(dimension))
    _LINE_CACHE[dimension] = lines
    return lines


# ---------------------------------------------------------------------------
# 2. The exhaustive verifier
# ---------------------------------------------------------------------------

#: Every way a returned list can fail.  A failure is typed, and every one of them scores zero.
INVALID_REASONS = (
    "contains_a_forbidden_triple",
    "duplicate_point",
    "non_integer_point",
    "point_out_of_range",
    "too_many_points",
)


@dataclass(frozen=True, slots=True)
class CapCertificate:
    """The exact, exhaustive verdict on one returned list.

    ``pairs_examined`` and ``pairs_expected`` are produced by two different routes -- one by
    counting the loop's iterations, the other as ``m (m - 1) / 2`` -- so a verifier that
    silently skipped work cannot emit a certificate that checks out.
    """

    dimension: int
    valid: bool
    reason: str
    cardinality: int
    pairs_examined: int
    pairs_expected: int
    violating_pairs: int
    first_forbidden_triple: tuple[int, int, int] | None
    points: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.reason and self.reason not in INVALID_REASONS:
            raise CapsetError(f"undeclared invalidity reason: {self.reason}")
        if self.valid and self.reason:
            raise CapsetError("a valid certificate carries an invalidity reason")
        if self.valid and self.pairs_examined != self.pairs_expected:
            raise CapsetError("a valid certificate did not examine every pair")
        if self.violating_pairs % 3:
            # A forbidden triple inside the list is detected by all three of its pairs, so the
            # violating-pair count is always three times the number of forbidden triples.  A
            # remainder here would mean the scan was not exhaustive.
            raise CapsetError("violating pairs are not a whole number of forbidden triples")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "valid": self.valid,
            "reason": self.reason,
            "cardinality": self.cardinality,
            "pairs_examined": self.pairs_examined,
            "pairs_expected": self.pairs_expected,
            "exhaustive": self.pairs_examined == self.pairs_expected,
            "violating_pairs": self.violating_pairs,
            "forbidden_triples_found": self.violating_pairs // 3,
            "first_forbidden_triple": (
                list(self.first_forbidden_triple) if self.first_forbidden_triple else None
            ),
            "points": list(self.points),
            "points_sha256": canonical_sha256(list(self.points)),
        }


def read_points(outputs: Sequence[str]) -> tuple[list[int], str]:
    """Parse the sandbox's rendered numbers as points.  Returns ``(points, reason)``.

    The sandbox renders every returned number with ``.17g``, so an integer arrives as its exact
    decimal digits.  Anything that is not an exact integer is a typed rejection here rather
    than a silently rounded point downstream.
    """

    points: list[int] = []
    for raw in outputs:
        text = raw.strip()
        if not re.fullmatch(r"-?\d+", text):
            return [], "non_integer_point"
        points.append(int(text))
    return points, ""


def verify_cap(points: Sequence[int], dimension: int, *, max_points: int) -> CapCertificate:
    """Exhaustively verify a list of points.  Every pair is examined; nothing is sampled."""

    size = points_in(dimension)
    ordered = tuple(int(item) for item in points)
    count = len(ordered)
    expected = count * (count - 1) // 2
    if count > max_points:
        return CapCertificate(
            dimension, False, "too_many_points", count, 0, expected, 0, None, ordered
        )
    for item in ordered:
        if not 0 <= item < size:
            return CapCertificate(
                dimension, False, "point_out_of_range", count, 0, expected, 0, None, ordered
            )
    membership = set(ordered)
    if len(membership) != count:
        return CapCertificate(
            dimension, False, "duplicate_point", count, 0, expected, 0, None, ordered
        )
    examined = 0
    violations = 0
    witness: tuple[int, int, int] | None = None
    for index in range(count):
        left = ordered[index]
        for offset in range(index + 1, count):
            right = ordered[offset]
            examined += 1
            other = third_point(left, right, dimension)
            if other in membership:
                violations += 1
                low, middle, high = sorted((left, right, other))
                triple = (low, middle, high)
                if witness is None or triple < witness:
                    witness = triple
    if violations:
        return CapCertificate(
            dimension,
            False,
            "contains_a_forbidden_triple",
            count,
            examined,
            expected,
            violations,
            witness,
            ordered,
        )
    return CapCertificate(dimension, True, "", count, examined, expected, 0, None, ordered)


# ---------------------------------------------------------------------------
# 3. The elementary upper bound, proved by exhibition
# ---------------------------------------------------------------------------


def parallel_class(dimension: int) -> tuple[tuple[int, int, int], ...]:
    """The ``3^(n-1)`` lines in the direction of the top digit, in ascending order.

    Together they partition the space, which :func:`upper_bound_certificate` verifies point by
    point rather than asserting.
    """

    size = points_in(dimension)
    step = size // 3
    return tuple((base, base + step, base + 2 * step) for base in range(step))


def upper_bound_certificate(dimension: int) -> dict[str, Any]:
    """An exactly checked proof that no valid set exceeds ``2 * 3^(n-1)``.

    Exhibit a partition of the space into lines; a valid set meets each line at most twice,
    because meeting one three times *is* a forbidden triple; therefore the cardinality is at
    most twice the number of parts.  Everything here is verified, including that each exhibited
    part really is a line and that the parts really do cover every point exactly once.
    """

    size = points_in(dimension)
    parts = parallel_class(dimension)
    covered: dict[int, int] = {}
    every_part_is_a_line = True
    for left, middle, right in parts:
        if third_point(left, middle, dimension) != right:
            every_part_is_a_line = False
        for point in (left, middle, right):
            covered[point] = covered.get(point, 0) + 1
    covers_every_point_once = len(covered) == size and set(covered.values()) == {1}
    bound = 2 * (size // 3)
    return {
        "dimension": dimension,
        "points": size,
        "parts": len(parts),
        "part_size": 3,
        "every_part_is_a_line": every_part_is_a_line,
        "covers_every_point_exactly_once": covers_every_point_once,
        "points_covered": len(covered),
        "max_points_per_part_in_a_valid_set": 2,
        "elementary_upper_bound": bound,
        "argument": (
            "the exhibited parts are lines and partition the space; a valid set meets a line "
            "at most twice, because meeting it three times is itself a forbidden triple; "
            "so cardinality <= 2 * parts"
        ),
        "verified": every_part_is_a_line and covers_every_point_once,
    }


def elementary_upper_bound(dimension: int) -> int:
    """``2 * 3^(n-1)``, but only after the exhibited partition passes verification."""

    certificate = upper_bound_certificate(dimension)
    if not certificate["verified"]:
        raise CapsetError(f"the upper-bound partition failed verification at n={dimension}")
    return int(certificate["elementary_upper_bound"])


# ---------------------------------------------------------------------------
# 4. The sealed records.  Nothing on the proposal path may read this table.
# ---------------------------------------------------------------------------

#: The lexicographically first maximum-size valid set at ``n = 3``, cardinality 9.
WITNESS_DIMENSION_3: tuple[int, ...] = (0, 1, 3, 4, 9, 10, 14, 17, 23)

#: The lexicographically first maximum-size valid set at ``n = 4``, cardinality 20.
WITNESS_DIMENSION_4: tuple[int, ...] = (
    0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 32, 35, 38, 47, 59, 65, 66, 67, 71, 77,
)

#: Per dimension: the best cardinality in the literature, whether it is proved maximal, and an
#: explicit witness where one is cheap enough to seal and to orbit.  Read only after scoring.
SEALED_RECORDS: dict[int, dict[str, Any]] = {
    1: {"cardinality": 2, "status": "proved_maximal", "witness": (0, 1)},
    2: {"cardinality": 4, "status": "proved_maximal", "witness": (0, 1, 3, 4)},
    3: {"cardinality": 9, "status": "proved_maximal", "witness": WITNESS_DIMENSION_3},
    4: {"cardinality": 20, "status": "proved_maximal", "witness": WITNESS_DIMENSION_4},
    5: {"cardinality": 45, "status": "proved_maximal", "witness": ()},
    6: {"cardinality": 112, "status": "proved_maximal", "witness": ()},
}


def sealed_record(dimension: int) -> dict[str, Any]:
    """The sealed row for one dimension, in receipt form.  Consumed only after scoring."""

    record = SEALED_RECORDS.get(int(dimension))
    if record is None:
        raise CapsetError(f"no sealed record declared for dimension {dimension}")
    return {
        "dimension": int(dimension),
        "cardinality": int(record["cardinality"]),
        "status": str(record["status"]),
        "witness": [int(item) for item in record["witness"]],
        "witness_declared": bool(record["witness"]),
    }


# ---------------------------------------------------------------------------
# 5. The monomial affine group and the orbit novelty channel
# ---------------------------------------------------------------------------

_ORBIT_CACHE: dict[tuple[int, tuple[int, ...]], frozenset[int]] = {}


def monomial_group_order(dimension: int) -> int:
    """``3^n * 2^n * n!`` -- the exact order of the declared group, as an integer."""

    factorial = 1
    for index in range(2, dimension + 1):
        factorial *= index
    return points_in(dimension) * (2**dimension) * factorial


def _permutations(count: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    def walk(prefix: list[int], rest: list[int]) -> None:
        if not rest:
            result.append(tuple(prefix))
            return
        for index, item in enumerate(rest):
            walk([*prefix, item], rest[:index] + rest[index + 1 :])

    walk([], list(range(count)))
    return result


def linear_parts(dimension: int) -> list[tuple[int, ...]]:
    """Every ``code -> code`` map given by a coordinate permutation and per-digit scaling.

    There are ``2^n n!`` of them and they are exactly the linear part of the monomial affine
    group: a permutation matrix with a nonzero scalar in each row.
    """

    digits = digit_table(dimension)
    size = len(digits)
    weights = [3**index for index in range(dimension)]
    scalings = [
        tuple(1 + ((mask >> index) & 1) for index in range(dimension))
        for mask in range(2**dimension)
    ]
    parts: list[tuple[int, ...]] = []
    for permutation in _permutations(dimension):
        for scaling in scalings:
            mapped: list[int] = []
            for code in range(size):
                row = digits[code]
                value = 0
                for position in range(dimension):
                    value += (scaling[position] * row[permutation[position]] % 3) * weights[
                        position
                    ]
                mapped.append(value)
            parts.append(tuple(mapped))
    return parts


def translate(code: int, shift: int, dimension: int) -> int:
    """Digit-wise addition of two points, in exact integer arithmetic."""

    digits = digit_table(dimension)
    left, right = digits[code], digits[shift]
    value = 0
    weight = 1
    for index in range(dimension):
        value += ((left[index] + right[index]) % 3) * weight
        weight *= 3
    return value


def monomial_orbit(witness: Sequence[int], dimension: int) -> frozenset[int]:
    """Every image of ``witness`` under the monomial affine group, as bitmasks.  Exhaustive.

    Each group element is applied to the witness and the image recorded as a bitmask over the
    ``3^n`` points, so membership is exact integer set membership and the distance to the orbit
    is a popcount.  Computed once per (dimension, witness) and cached.
    """

    key = (dimension, tuple(witness))
    cached = _ORBIT_CACHE.get(key)
    if cached is not None:
        return cached
    size = points_in(dimension)
    shift_tables = [
        tuple(translate(code, shift, dimension) for code in range(size)) for shift in range(size)
    ]
    masks: set[int] = set()
    for part in linear_parts(dimension):
        image = [part[point] for point in witness]
        for table in shift_tables:
            mask = 0
            for point in image:
                mask |= 1 << table[point]
            masks.add(mask)
    orbit = frozenset(masks)
    _ORBIT_CACHE[key] = orbit
    return orbit


def points_mask(points: Sequence[int]) -> int:
    """The bitmask of a point list, one bit per point of the space."""

    mask = 0
    for point in points:
        mask |= 1 << int(point)
    return mask


@dataclass(frozen=True, slots=True)
class OrbitPolicy:
    """The declared rule turning orbit distance into a multiplier.  Exact rationals only.

    The zero threshold is exactly zero on purpose: the multiplier is zero **iff** the returned
    set is literally one of the enumerated images of the sealed witness.  There is no epsilon
    to argue about and no fitted parameter anywhere in the channel.
    """

    saturation: Fraction = Fraction(1, 4)

    def __post_init__(self) -> None:
        if not Fraction(0) < self.saturation <= Fraction(1):
            raise CapsetError("orbit saturation must lie in (0, 1]")

    def multiplier_from_distance(self, distance: Fraction) -> Fraction:
        if distance <= 0:
            return Fraction(0)
        if distance >= self.saturation:
            return Fraction(1)
        return distance / self.saturation

    def to_dict(self) -> dict[str, Any]:
        return {
            "zero_threshold": "0/1",
            "saturation": _rational(self.saturation),
            "distance_formula": (
                "min over the enumerated orbit of popcount(returned XOR image) / "
                "(|returned| + |witness|)"
            ),
            "multiplier_rule": (
                "0 when the returned set is exactly an image of the sealed witness; "
                "else min(1, distance / saturation)"
            ),
        }


def orbit_novelty(
    certificate: CapCertificate, witness: Sequence[int], policy: OrbitPolicy
) -> dict[str, Any]:
    """The multiplier for one construction, from its exact distance to the sealed orbit."""

    if not certificate.valid:
        return {
            "multiplier": Fraction(0),
            "reason": "invalid_construction",
            "detail": {"invalidity": certificate.reason},
        }
    if not witness:
        return {
            "multiplier": Fraction(1),
            "reason": "no_sealed_witness_declared",
            "detail": {
                "note": (
                    "this arm declares a record cardinality but no explicit witness, so no "
                    "reproduction can be detected on this channel and the multiplier is 1 by "
                    "declaration"
                )
            },
        }
    denominator = certificate.cardinality + len(witness)
    gap = abs(certificate.cardinality - len(witness))
    lower_bound = Fraction(gap, denominator)
    if lower_bound >= policy.saturation:
        # |A XOR B| >= ||A| - |B||, so the cardinality gap alone already saturates.  Skipping
        # the orbit scan here is an exact shortcut, not an approximation: the recorded bound is
        # recheckable by hand from the two integers printed beside it.
        return {
            "multiplier": Fraction(1),
            "reason": "cardinality_gap_saturates_the_orbit_distance",
            "detail": {
                "returned_cardinality": certificate.cardinality,
                "witness_cardinality": len(witness),
                "distance_lower_bound": _rational(lower_bound),
                "saturation": _rational(policy.saturation),
                "orbit_scan_skipped": True,
            },
        }
    orbit = monomial_orbit(witness, certificate.dimension)
    mask = points_mask(certificate.points)
    best = min((mask ^ image).bit_count() for image in orbit)
    distance = Fraction(best, denominator)
    multiplier = policy.multiplier_from_distance(distance)
    reason = (
        "monomial_orbit_of_the_sealed_witness"
        if multiplier == 0
        else "distance_from_the_sealed_witness_orbit"
    )
    return {
        "multiplier": multiplier,
        "reason": reason,
        "detail": {
            "returned_cardinality": certificate.cardinality,
            "witness_cardinality": len(witness),
            "orbit_size": len(orbit),
            "group_order": monomial_group_order(certificate.dimension),
            "minimum_symmetric_difference": best,
            "distance": _rational(distance),
            "saturation": _rational(policy.saturation),
            "orbit_scan_skipped": False,
        },
    }


# ---------------------------------------------------------------------------
# 6. Exact rendering.  No float reaches a sealed number.
# ---------------------------------------------------------------------------


def _rational(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _decimal(value: Fraction, places: int = 12) -> str:
    """A decimal string produced by integer division and half-up rounding.  Never via float."""

    if value < 0:
        raise CapsetError("only non-negative quantities are rendered as decimals here")
    scale = 10**places
    whole, remainder = divmod(value.numerator * scale, value.denominator)
    if 2 * remainder >= value.denominator:
        whole += 1
    integral, fractional = divmod(whole, scale)
    return f"{integral}.{fractional:0{places}d}"


def _parse_rational(text: str) -> Fraction:
    numerator, _, denominator = str(text).partition("/")
    return Fraction(int(numerator), int(denominator))


# ---------------------------------------------------------------------------
# 7. The declared problem
# ---------------------------------------------------------------------------

FORBIDDEN_CONSTRUCTION_VOCABULARY: tuple[str, ...] = (
    "affine",
    "arithmetic",
    "cap",
    "capset",
    "collinear",
    "combinatorial",
    "davenport",
    "ellenberg",
    "geometry",
    "gijswijt",
    "hill",
    "meshulam",
    "pellegrino",
    "progression",
    "roth",
    "sperner",
    "szemeredi",
    "tao",
)


def numeral_violations(text: str, forbidden: Sequence[int]) -> list[str]:
    """Forbidden numerals in ``text``, tokenised on digit runs only.

    The word-channel guard the loop already runs cannot see a sealed *number*: it tokenises on
    letter runs, so a sealed cardinality reaches a prompt unflagged.  This is the same guard on
    the digit channel, matching whole runs so ``120`` does not read as a leak of ``20`` while a
    bare ``20`` does.
    """

    tokens = set(re.findall(r"\d+", text))
    return sorted(tokens & {str(int(item)) for item in forbidden})


@dataclass(frozen=True, slots=True)
class ConstructionProblem:
    """A construction problem: a signature, a weak seed, an exhaustive verifier, a sealed record.

    The proposer sees ``signature_text``, ``entry``, the seed program, the observed forbidden
    triples, and the scores of previous programs.  It does not see the sealed record, the
    sealed witness, the elementary bound, or any cardinality.
    """

    problem_id: str
    dimension: int
    entry: str = "build"
    orbit_channel_enabled: bool = True
    observed_triples_shown: int = 12
    sandbox: SandboxBudget = field(default_factory=lambda: SandboxBudget(wall_seconds=6.0))
    orbit_policy: OrbitPolicy = field(default_factory=OrbitPolicy)
    mutation_bank: tuple[str, ...] = ()
    forbidden_vocabulary: tuple[str, ...] = FORBIDDEN_CONSTRUCTION_VOCABULARY

    def __post_init__(self) -> None:
        if self.dimension not in SEALED_RECORDS:
            raise CapsetError(f"no sealed record declared for dimension {self.dimension}")
        declaration = "\n".join(
            [
                self.signature_text(),
                self.seed_program(),
                *(" ".join(str(item) for item in triple) for triple in self.observed_triples()),
            ]
        )
        leaks = vocabulary_violations(declaration, self.forbidden_vocabulary)
        if leaks:
            raise CapsetError(f"the problem declaration leaks vocabulary: {leaks}")
        numerals = numeral_violations(declaration, self.forbidden_numerals())
        if numerals:
            raise CapsetError(f"the problem declaration leaks a sealed numeral: {numerals}")

    # -- the proposer-visible surface --------------------------------------------------

    @property
    def points(self) -> int:
        return points_in(self.dimension)

    @property
    def max_points(self) -> int:
        """The longest list the verifier will look at.  Beyond it the score is zero."""

        return self.points

    def signature_text(self) -> str:
        return (
            f"build() -> list of distinct integers, each in [0, {self.points}). "
            f"Digit k of an integer a is (a // 3 ** k) % 3, for k = 0 .. {self.dimension - 1}. "
            "Three distinct returned integers are BAD when, in every digit position k, their "
            "three digits sum to a multiple of 3. Return as many integers as you can with no "
            "BAD triple among them. Any BAD triple, any repeat, or any integer out of range "
            "scores zero."
        )

    @staticmethod
    def digit_expression(index: int) -> str:
        """Digit ``index`` of ``a``, written without any power of three as a literal.

        ``a // 3 // 3 % 3`` rather than ``a // 9 % 3``.  The two are equal, and the first is
        the one that can be written at ``n = 3`` without the seed program carrying the sealed
        record cardinality ``9`` in its bytes.  The blindness guard would otherwise refuse
        every prompt the problem could assemble, which is a guard defeating its own problem.
        """

        return "a" + " // 3" * index + " % 3"

    def seed_program(self) -> str:
        """A deliberately weak but valid founder: the single point whose digits are all zero.

        The whole computation sits inside the returned expression on purpose.  The declared
        mutation grammar draws its rewrite slots from ``return`` statements, so a seed that
        computes into a local and returns a bare name gives the mutator nothing to hold: it is
        the difference between a search and a random walk, and it is measurable -- the earlier
        loop-and-append seed never once improved on itself across a full campaign.

        The climb the seed opens is real and short.  Relaxing each ``< 1`` to ``< 2`` admits one
        more digit value, and relaxing all of them lands on the digits-in-{0,1} construction of
        size ``2^n`` -- a genuine construction, and still well under the sealed record.
        """

        condition = " and ".join(
            f"{self.digit_expression(index)} < 1" for index in range(self.dimension)
        )
        return f"def build():\n    return [a for a in range({self.points}) if {condition}]\n"

    def observed_triples(self) -> tuple[tuple[int, int, int], ...]:
        """The lexicographically first BAD triples that carry no sealed numeral.

        Observations that make sense for a construction problem are instances of the
        *constraint*, not values of an answer.  The sealed-numeral filter is part of the
        declaration and is applied by the same guard that screens assembled prompts, so the
        base prompt can never be the thing that leaks.
        """

        forbidden = self.forbidden_numerals()
        chosen: list[tuple[int, int, int]] = []
        for triple in iter_lines(self.dimension):
            if len(chosen) >= self.observed_triples_shown:
                break
            if numeral_violations(" ".join(str(item) for item in triple), forbidden):
                continue
            chosen.append(triple)
        return tuple(chosen)

    def forbidden_numerals(self) -> tuple[int, ...]:
        """The sealed numbers no prompt may carry: the record and the elementary bound."""

        size = points_in(self.dimension)
        return (int(SEALED_RECORDS[self.dimension]["cardinality"]), 2 * (size // 3))

    def sealed_witness(self) -> tuple[int, ...]:
        if not self.orbit_channel_enabled:
            return ()
        return tuple(int(item) for item in SEALED_RECORDS[self.dimension]["witness"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "dimension": self.dimension,
            "entry": self.entry,
            "points": self.points,
            "lines": lines_in(self.dimension),
            "max_points": self.max_points,
            "signature_text": self.signature_text(),
            "seed_program": self.seed_program(),
            "observed_triples": [list(item) for item in self.observed_triples()],
            "orbit_channel_enabled": self.orbit_channel_enabled,
            "orbit_policy": self.orbit_policy.to_dict(),
            "sealed_witness_declared": bool(self.sealed_witness()),
            "elementary_upper_bound": elementary_upper_bound(self.dimension),
            "upper_bound_certificate": upper_bound_certificate(self.dimension),
            "forbidden_vocabulary": list(self.forbidden_vocabulary),
            "mutation_bank": list(self.mutation_bank),
            "sandbox": self.sandbox.to_dict(),
        }


def mutation_bank_for(dimension: int) -> tuple[str, ...]:
    """The declared mutation grammar: expressions the mock mutator may splice into a program.

    Every digit of a point is here as an expression, written with repeated ``// 3`` so no power
    of three appears as a literal and the bank cannot smuggle a sealed numeral into a program
    and from there into a prompt.
    """

    digits = tuple(
        ConstructionProblem.digit_expression(index) for index in range(max(dimension, 1))
    )
    return (
        *digits,
        "a",
        "a % 2",
        "a // 3",
        "0",
        "1",
        "2",
        "3",
        f"range({points_in(dimension)})",
        str(dimension),
    )


def declared_problems() -> dict[str, ConstructionProblem]:
    """The three declared construction problems plus the control that isolates the multiplier."""

    problems: dict[str, ConstructionProblem] = {}
    for dimension in (3, 4, 5):
        problems[f"capset_dimension_{dimension}"] = ConstructionProblem(
            problem_id=f"capset_dimension_{dimension}",
            dimension=dimension,
            mutation_bank=mutation_bank_for(dimension),
        )
    problems["capset_dimension_4_open_orbit"] = ConstructionProblem(
        problem_id="capset_dimension_4_open_orbit",
        dimension=4,
        orbit_channel_enabled=False,
        mutation_bank=mutation_bank_for(4),
    )
    return problems


#: Run labels name arms.  The control is its own declared problem, differing from the
#: dimension-4 arm in exactly one field: the sealed witness orbit is off.
RUN_LABEL_PROBLEM = {
    "capset_dimension_3": "capset_dimension_3",
    "capset_dimension_4": "capset_dimension_4",
    "capset_dimension_4_open_orbit": "capset_dimension_4_open_orbit",
    "capset_dimension_5": "capset_dimension_5",
}


# ---------------------------------------------------------------------------
# 8. Scoring one program end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoredConstruction:
    """One executed program: its certificate, its exact score, and why it scored that way."""

    program_sha256: str
    source: str
    origin: str
    generation: int
    island: int
    sandbox: SandboxOutcome
    certificate: CapCertificate | None
    quality: Fraction
    novelty_multiplier: Fraction
    novelty_reason: str
    novelty_detail: dict[str, Any]
    final: Fraction

    @property
    def final_score(self) -> float:
        """A float **only** for softmax sampling.  It never reaches a sealed record."""

        return float(self.final)

    @property
    def cardinality(self) -> int:
        return self.certificate.cardinality if self.certificate and self.certificate.valid else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_sha256": self.program_sha256,
            "source": self.source,
            "origin": self.origin,
            "generation": self.generation,
            "island": self.island,
            "sandbox": self.sandbox.classification(),
            "certificate": self.certificate.to_dict() if self.certificate else None,
            "quality": _rational(self.quality),
            "quality_decimal": _decimal(self.quality),
            "novelty": {
                "multiplier": _rational(self.novelty_multiplier),
                "multiplier_decimal": _decimal(self.novelty_multiplier),
                "reason": self.novelty_reason,
                "detail": self.novelty_detail,
            },
            "final_score": _rational(self.final),
            "final_score_decimal": _decimal(self.final),
        }


def score_construction(
    problem: ConstructionProblem,
    source: str,
    *,
    origin: str = "proposed",
    generation: int = 0,
    island: int = 0,
) -> ScoredConstruction:
    """Execute, verify exhaustively, score exactly.  Never raises on the program."""

    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    outcome = run_in_sandbox(source, problem.entry, ((),), problem.sandbox, output_width=0)
    zero = Fraction(0)
    if not outcome.ok:
        return ScoredConstruction(
            digest,
            source,
            origin,
            generation,
            island,
            outcome,
            None,
            zero,
            zero,
            "not_executable",
            {"sandbox_reason": outcome.reason},
            zero,
        )
    points, parse_reason = read_points(outcome.outputs)
    if parse_reason:
        certificate = CapCertificate(
            problem.dimension, False, parse_reason, len(outcome.outputs), 0, 0, 0, None, ()
        )
    else:
        certificate = verify_cap(points, problem.dimension, max_points=problem.max_points)
    bound = elementary_upper_bound(problem.dimension)
    quality = Fraction(certificate.cardinality, bound) if certificate.valid else zero
    verdict = orbit_novelty(certificate, problem.sealed_witness(), problem.orbit_policy)
    multiplier = verdict["multiplier"]
    return ScoredConstruction(
        digest,
        source,
        origin,
        generation,
        island,
        outcome,
        certificate,
        quality,
        multiplier,
        str(verdict["reason"]),
        dict(verdict["detail"]),
        quality * multiplier,
    )


# ---------------------------------------------------------------------------
# 9. The blind prompt
# ---------------------------------------------------------------------------

_CONSTRUCTION_INSTRUCTION = (
    "You are given a function signature and several example implementations, each with a "
    "score between 0 and 1 that was measured by executing it and checking its output "
    "exhaustively. Write ONE new implementation of the same signature that would score "
    "higher. Return only Python source for a single function. Use no imports, no input or "
    "output, and no attribute names beginning with an underscore. You are not told what the "
    "numbers mean and you are not asked to explain anything; only the measured score counts."
)


def guard_construction_prompt(text: str, problem: ConstructionProblem) -> str:
    """Return ``text`` unchanged, or refuse on either channel.  The refusal is the point."""

    words = vocabulary_violations(text, problem.forbidden_vocabulary)
    if words:
        raise CapsetError(f"prompt leaked forbidden vocabulary: {', '.join(words)}")
    numerals = numeral_violations(text, problem.forbidden_numerals())
    if numerals:
        raise CapsetError(f"prompt leaked a sealed numeral: {', '.join(numerals)}")
    return text


def build_construction_prompt(
    problem: ConstructionProblem, examples: Sequence[ScoredConstruction]
) -> str:
    """Assemble the blind prompt, then refuse it if a sealed word or number got in."""

    lines = [
        _CONSTRUCTION_INSTRUCTION,
        "",
        f"signature: {problem.signature_text()}",
        "import allowlist: none",
        f"wall clock limit: {problem.sandbox.wall_seconds} seconds",
        "",
        "# examples of BAD triples, which no returned list may contain:",
    ]
    for triple in problem.observed_triples():
        lines.append(f"#   {triple[0]}, {triple[1]}, {triple[2]}")
    lines.append("")
    for index, example in enumerate(examples):
        lines.append(f"# example {index} scored {_decimal(example.final, 9)}")
        lines.append(example.source)
        lines.append("")
    lines.append("# your implementation:")
    return guard_construction_prompt("\n".join(lines), problem)


# ---------------------------------------------------------------------------
# 10. The loop
# ---------------------------------------------------------------------------


def _stable_hash(text: str) -> int:
    """A seed that does not move between processes; ``hash()`` of a str is randomised."""

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _sample_examples(
    island: Sequence[ScoredConstruction], count: int, temperature: float, rng: random.Random
) -> list[ScoredConstruction]:
    """FunSearch's score-weighted sampling: softmax over the final score within one island."""

    if not island:
        return []
    top = max(item.final_score for item in island)
    pool = list(island)
    weights = [math.exp((item.final_score - top) / max(temperature, 1e-9)) for item in pool]
    chosen: list[ScoredConstruction] = []
    for _ in range(min(count, len(pool))):
        pick = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        chosen.append(pool.pop(pick))
        weights.pop(pick)
    return chosen


def _reset_islands(
    islands: list[list[ScoredConstruction]], rng: random.Random
) -> list[list[ScoredConstruction]]:
    """FunSearch's island reset: the weakest half is reseeded from the strongest half."""

    ranked = sorted(
        range(len(islands)),
        key=lambda index: -max((item.final for item in islands[index]), default=Fraction(0)),
    )
    strong = ranked[: max(1, len(islands) // 2)]
    for index in ranked[max(1, len(islands) // 2) :]:
        donor = islands[rng.choice(strong)]
        if donor:
            islands[index] = [max(donor, key=lambda item: item.final)]
    return islands


def _brief(item: ScoredConstruction) -> dict[str, Any]:
    return {
        "program_sha256": item.program_sha256,
        "origin": item.origin,
        "source": item.source,
        "verified_cardinality": item.cardinality,
        "valid": bool(item.certificate and item.certificate.valid),
        "quality": _rational(item.quality),
        "novelty_multiplier": _rational(item.novelty_multiplier),
        "novelty_reason": item.novelty_reason,
        "final_score": _rational(item.final),
    }


def run_construction_problem(
    problem: ConstructionProblem,
    config: LoopConfig,
    proposer: Any,
    governor: SpendGovernor,
    *,
    extra_programs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """One construction problem, start to finish.  Returns the arm's block of the receipt.

    ``extra_programs`` are scored **after** the search and never enter the population, which is
    what lets the declared probes reproduce the sealed witness without ever leaking it into a
    prompt.  That is the one structural difference from the response problem's probes, and it
    exists because a construction's answer is a literal list of coordinates: showing it to the
    proposer would not be a temptation to recall, it would be the whole task.
    """

    rng = random.Random(config.seed ^ _stable_hash(problem.problem_id))
    founder = score_construction(problem, problem.seed_program(), origin="seed")
    islands: list[list[ScoredConstruction]] = [[founder] for _ in range(config.islands)]
    sealed: dict[str, ScoredConstruction] = {founder.program_sha256: founder}
    history: list[dict[str, Any]] = []
    calls: list[ProposalCall] = []
    incidents: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    if not founder.sandbox.ok:
        incidents.append(
            {"program_sha256": founder.program_sha256, **founder.sandbox.classification()}
        )
    halt_reason = "generations_exhausted"

    steps = [
        (generation, island)
        for generation in range(config.generations)
        for island in (
            range(config.islands) if config.sweep_islands else (generation % config.islands,)
        )
    ]
    for generation, index in steps:
        if not governor.may_call():
            halt_reason = governor.halt_reason
            break
        examples = _sample_examples(
            islands[index], config.examples_per_prompt, config.temperature, rng
        )
        try:
            prompt = build_construction_prompt(problem, examples)
        except FunSearchError as error:
            refusals.append(
                {
                    "generation": generation,
                    "island": index,
                    "detail": str(error)[:200],
                    "examples": [item.program_sha256[:12] for item in examples],
                }
            )
            calls.append(
                ProposalCall(
                    getattr(proposer, "proposer_id", "unknown"),
                    "",
                    0,
                    0,
                    False,
                    "prompt_refused_by_the_blindness_guard",
                    str(error)[:200],
                )
            )
            continue
        governor.charge()
        call = proposer.propose(prompt, examples, config.proposals_per_call)
        calls.append(call)
        attempt = 0
        while not call.ok:
            kind = classify_failure(call)
            if kind == "persistent" or attempt >= config.transient_retries:
                raise ProposerCallFailed(
                    f"proposal call failed at generation {generation}, island {index}: "
                    f"{call.reason} / {call.detail[:200]}.  Classified {kind}.  Stopping the "
                    "run rather than sealing a receipt that looks like a full campaign."
                )
            attempt += 1
            time.sleep(config.retry_backoff_seconds * attempt)
            call = proposer.propose(prompt, examples, config.proposals_per_call)
            calls.append(call)
        produced: list[ScoredConstruction] = []
        for source in proposer.programs():
            scored = score_construction(
                problem, source, origin="proposed", generation=generation, island=index
            )
            produced.append(scored)
            sealed.setdefault(scored.program_sha256, scored)
            if not scored.sandbox.ok:
                incidents.append(
                    {"program_sha256": scored.program_sha256, **scored.sandbox.classification()}
                )
        islands[index] = sorted(
            {item.program_sha256: item for item in [*islands[index], *produced]}.values(),
            key=lambda item: (-item.final, item.program_sha256),
        )[: config.island_capacity]
        history.append(
            {
                "generation": generation,
                "island": index,
                "examples": [item.program_sha256[:12] for item in examples],
                "proposed": len(produced),
                "best_final_score_in_island": _rational(
                    max((item.final for item in islands[index]), default=Fraction(0))
                ),
                "best_verified_cardinality_in_island": max(
                    (item.cardinality for item in islands[index]), default=0
                ),
            }
        )
        if config.reset_period and (generation + 1) % config.reset_period == 0:
            islands = _reset_islands(islands, rng)

    for label, source in extra_programs:
        scored = score_construction(problem, source, origin=label)
        sealed.setdefault(scored.program_sha256, scored)
        if not scored.sandbox.ok:
            incidents.append(
                {"program_sha256": scored.program_sha256, **scored.sandbox.classification()}
            )

    record = sealed_record(problem.dimension)
    ordered = sorted(
        sealed.values(), key=lambda item: (-item.final, -item.quality, item.program_sha256)
    )
    valid = [item for item in sealed.values() if item.certificate and item.certificate.valid]
    best_card = max((item.cardinality for item in valid), default=0)
    # The declared probes are hand-written and reproduce the sealed witness, so a headline that
    # mixed them into "best found" would read as a search result.  Search-discovered means the
    # seed and whatever the proposer wrote, and nothing else.
    searched = [item for item in valid if item.origin in ("seed", "proposed")]
    best_searched = max((item.cardinality for item in searched), default=0)
    by_reason: dict[str, int] = {}
    for item in sealed.values():
        if item.certificate and not item.certificate.valid:
            by_reason[item.certificate.reason] = by_reason.get(item.certificate.reason, 0) + 1
    zeroed = [
        item
        for item in sealed.values()
        if item.novelty_reason == "monomial_orbit_of_the_sealed_witness"
    ]
    return {
        "problem": problem.to_dict(),
        "sealed_record_sha256": canonical_sha256(record),
        "sealed_record_revealed_after_scoring": record,
        "loop": config.to_dict(),
        "generations_run": len(history),
        "halt_reason": halt_reason,
        "population_history": history,
        "proposal_calls": [item.to_dict() for item in calls],
        "blindness_refusals": refusals,
        "sandbox_incidents": incidents,
        "sealed_programs": [item.to_dict() for item in ordered],
        "headline": {
            "programs_sealed": len(sealed),
            "valid_constructions": len(valid),
            "invalid_constructions_by_reason": dict(sorted(by_reason.items())),
            "best_verified_cardinality": best_card,
            "best_verified_cardinality_from_the_search": best_searched,
            "sealed_record_cardinality": record["cardinality"],
            "sealed_record_status": record["status"],
            "beats_sealed_record": best_card > int(record["cardinality"]),
            "beats_sealed_record_from_the_search": best_searched > int(record["cardinality"]),
            "ties_sealed_record": best_card == int(record["cardinality"]),
            "elementary_upper_bound": elementary_upper_bound(problem.dimension),
            "constructions_in_the_sealed_orbit": len(zeroed),
            "prompts_refused_by_the_blindness_guard": len(refusals),
            "best_by_final_score": _brief(ordered[0]) if ordered else None,
            "best_by_cardinality": (
                _brief(max(valid, key=lambda item: (item.cardinality, item.program_sha256)))
                if valid
                else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# 11. The declared probe programs
# ---------------------------------------------------------------------------


def _program(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def literal_witness_program(witness: Sequence[int]) -> str:
    """A program that simply writes the sealed witness down."""

    body = ", ".join(str(int(item)) for item in witness)
    return _program("def build():", f"    return [{body}]")


def translated_witness_program(witness: Sequence[int], shift: int, dimension: int) -> str:
    """A translate of the sealed witness: same orbit, entirely different bytes."""

    return literal_witness_program(sorted(translate(point, shift, dimension) for point in witness))


def scaled_witness_program(witness: Sequence[int], dimension: int) -> str:
    """The sealed witness under a coordinate permutation and scaling: same orbit again."""

    part = linear_parts(dimension)[-1]
    return literal_witness_program(sorted(part[point] for point in witness))


def search_witness_program(dimension: int, target: int) -> str:
    """A program that *finds* an extremal construction instead of listing one.

    This is the probe that matters.  Its bytes contain no coordinate of the sealed witness
    anywhere -- it runs a deterministic depth-first search and returns whatever that finds --
    so a novelty channel matching text rather than behaviour would let it straight through.
    """

    return _program(
        "def build():",
        f"    n = {dimension}",
        f"    size = {3 ** dimension}",
        f"    target = {target}",
        "    pow3 = []",
        "    weight = 1",
        "    for k in range(n):",
        "        pow3.append(weight)",
        "        weight = weight * 3",
        "    digits = []",
        "    for a in range(size):",
        "        row = []",
        "        rest = a",
        "        for k in range(n):",
        "            row.append(rest % 3)",
        "            rest = rest // 3",
        "        digits.append(row)",
        "    third = []",
        "    for a in range(size):",
        "        da = digits[a]",
        "        row = []",
        "        for b in range(size):",
        "            db = digits[b]",
        "            c = 0",
        "            for k in range(n):",
        "                c = c + ((3 - (da[k] + db[k]) % 3) % 3) * pow3[k]",
        "            row.append(c)",
        "        third.append(row)",
        "",
        "    def dfs(start, chosen, blocked):",
        "        if len(chosen) >= target:",
        "            return list(chosen)",
        "        if len(chosen) + (size - start) < target:",
        "            return None",
        "        for x in range(start, size):",
        "            if x in blocked:",
        "                continue",
        "            extra = set()",
        "            for y in chosen:",
        "                extra.add(third[x][y])",
        "            chosen.append(x)",
        "            found = dfs(x + 1, chosen, blocked | extra | set([x]))",
        "            if found is not None:",
        "                return found",
        "            chosen.pop()",
        "        return None",
        "",
        "    found = dfs(0, [], set())",
        "    if found is None:",
        "        return [0, 1]",
        "    return found",
    )


def invalid_probe_program(dimension: int) -> str:
    """A program returning a list that contains a forbidden triple.  It must score zero."""

    triple = (0, 1, third_point(0, 1, dimension))
    return _program("def build():", f"    return [{', '.join(str(item) for item in triple)}]")


def probe_programs(dimension: int) -> tuple[tuple[str, str], ...]:
    """The declared probes for one dimension.  Scored after the search, never seeded into it."""

    record = SEALED_RECORDS[dimension]
    witness = tuple(int(item) for item in record["witness"])
    probes: list[tuple[str, str]] = [
        ("probe_invalid_contains_a_forbidden_triple", invalid_probe_program(dimension)),
    ]
    if witness:
        probes.append(("probe_known_literal", literal_witness_program(witness)))
        probes.append(
            ("probe_known_translated", translated_witness_program(witness, 1, dimension))
        )
        probes.append(("probe_known_scaled", scaled_witness_program(witness, dimension)))
        probes.append(
            ("probe_known_searched", search_witness_program(dimension, int(record["cardinality"])))
        )
    return tuple(probes)


# ---------------------------------------------------------------------------
# 12. The campaign
# ---------------------------------------------------------------------------

#: The declared campaign.  The generation count is not decoration: the mutation that opens the
#: climb -- relaxing one digit threshold -- lands on roughly one proposal in forty, and four of
#: them in a row are needed to reach the digits-in-{0,1} construction.  At eighteen generations
#: the measured search stalled at cardinality four; at eighty it reaches sixteen.
CAMPAIGN_CONFIG = LoopConfig(
    islands=3,
    island_capacity=8,
    generations=80,
    examples_per_prompt=3,
    proposals_per_call=6,
    reset_period=10,
    temperature=0.12,
    seed=20260819,
)

SEARCH_ARMS: tuple[str, ...] = (
    "capset_dimension_3",
    "capset_dimension_4",
    "capset_dimension_5",
)


def run_campaign(
    *,
    config: LoopConfig | None = None,
    ledger_path: str | Path = "runs/math/capset/spend-ledger.json",
    max_calls: int = 400,
    max_dollars_hundredths: int = 400,
    charge_per_call_hundredths: int = 1,
    proposer_kind: str = "mock",
    claude_executable: str = "claude",
    include_hostile_suite: bool = True,
) -> dict[str, Any]:
    """Run every declared arm and seal one receipt."""

    started = time.perf_counter()
    settings = config or CAMPAIGN_CONFIG
    problems = declared_problems()
    if proposer_kind == "mock":
        use_live = False
    else:
        probe = ClaudeCliProposer(claude_executable)
        call = probe.propose("Return a function named build that returns the list [0, 1].", (), 1)
        if not call.ok:
            raise ProposerUnavailable(
                f"requested proposer {proposer_kind!r} is unreachable: {call.reason} / "
                f"{call.detail[:200]}.  There is no degraded mode: pass proposer_kind='mock' to "
                "run the deterministic mutator deliberately, which is a different experiment "
                "and is labelled as one."
            )
        use_live = True

    governor = SpendGovernor(
        Path(ledger_path), max_calls, max_dollars_hundredths, charge_per_call_hundredths
    )

    def make_proposer(problem: ConstructionProblem, salt: int) -> Any:
        if use_live:
            return ClaudeCliProposer(claude_executable)
        return MockMutationProposer(settings.seed + salt, problem.mutation_bank)

    blocks: list[dict[str, Any]] = []
    for salt, label in enumerate(SEARCH_ARMS):
        problem = problems[label]
        block = run_construction_problem(
            problem,
            settings,
            make_proposer(problem, salt * 977),
            governor,
            extra_programs=probe_programs(problem.dimension),
        )
        block["run_label"] = label
        blocks.append(block)

    # The control.  It searches nothing: every program the dimension-4 arm sealed is re-scored
    # against a problem identical in every declared field except one -- the sealed witness orbit
    # is off.  Same programs, same verifier, same bound, so a score that moves can only have
    # moved because of the multiplier.
    live = next(block for block in blocks if block["run_label"] == "capset_dimension_4")
    replicas = tuple((record["origin"], record["source"]) for record in live["sealed_programs"])
    control = run_construction_problem(
        problems["capset_dimension_4_open_orbit"],
        replace(settings, generations=0),
        make_proposer(problems["capset_dimension_4_open_orbit"], 3 * 977),
        governor,
        extra_programs=replicas,
    )
    control["run_label"] = "capset_dimension_4_open_orbit"
    control["control_note"] = (
        "no search was run here. Every program sealed by the dimension-4 arm was re-scored "
        "against a problem identical in every declared field except one: the sealed witness "
        "orbit is disabled."
    )
    blocks.append(control)

    hostile = run_hostile_suite() if include_hostile_suite else []
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "lane": "capset-construction-search",
        "claims": CLAIMS,
        "config": settings.to_dict(),
        "config_sha256": canonical_sha256(settings.to_dict()),
        "proposer": {
            "requested": proposer_kind,
            "used": "claude_cli_oauth" if use_live else "deterministic_mock_mutator",
            "note": (
                "the proposer supplies source code only; every cardinality in this receipt was "
                "produced by executing a program and verifying its output exhaustively"
            ),
        },
        "sandbox": {
            "declaration": SandboxBudget(wall_seconds=6.0).to_dict(),
            "hostile_suite": hostile,
            "hostile_suite_run": bool(hostile),
            "all_hostile_programs_contained": all(item["contained"] for item in hostile),
            "incident_log": _incident_log(blocks),
        },
        "budget": governor.to_dict(),
        "problems": blocks,
        "headline": _headline(blocks),
        "scope": SCOPE,
    }
    body["result_core_sha256"] = canonical_sha256(body)
    body["measurement"] = {"elapsed_seconds": format(time.perf_counter() - started, ".3f")}
    return {**body, "content_sha256": canonical_sha256(body)}


def _incident_log(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every program that failed to execute, counted by typed reason."""

    counts: dict[str, int] = {}
    total = 0
    for block in blocks:
        for incident in block["sandbox_incidents"]:
            counts[incident["reason"]] = counts.get(incident["reason"], 0) + 1
            total += 1
    return {
        "incidents": total,
        "by_reason": dict(sorted(counts.items())),
        "every_reason_is_declared": all(name in SANDBOX_FAILURE_REASONS for name in counts),
        "loop_survived_every_incident": True,
    }


def _headline(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "run_label": block["run_label"],
            "dimension": block["problem"]["dimension"],
            "programs_sealed": block["headline"]["programs_sealed"],
            "valid_constructions": block["headline"]["valid_constructions"],
            "best_verified_cardinality": block["headline"]["best_verified_cardinality"],
            "best_verified_cardinality_from_the_search": block["headline"][
                "best_verified_cardinality_from_the_search"
            ],
            "sealed_record_cardinality": block["headline"]["sealed_record_cardinality"],
            "beats_sealed_record": block["headline"]["beats_sealed_record"],
            "beats_sealed_record_from_the_search": block["headline"][
                "beats_sealed_record_from_the_search"
            ],
            "constructions_in_the_sealed_orbit": block["headline"][
                "constructions_in_the_sealed_orbit"
            ],
        }
        for block in blocks
    ]
    return {
        "arms": rows,
        "any_arm_beats_its_sealed_record": any(row["beats_sealed_record"] for row in rows),
        "any_search_beats_its_sealed_record": any(
            row["beats_sealed_record_from_the_search"] for row in rows
        ),
        "record_gate_is_an_integer_comparison": True,
        "note": (
            "every declared dimension carries a record that is already a proved maximum, so the "
            "honest expected value of the record gate is false. A true here would be an exactly "
            "verified integer rather than a claim, recomputable from the sealed point list "
            "alone. best_verified_cardinality counts the hand-written probes, which reproduce "
            "the sealed witness on purpose; best_verified_cardinality_from_the_search counts "
            "only the seed and what the proposer wrote, and it is the number a search result "
            "would have to move"
        ),
    }


# ---------------------------------------------------------------------------
# 13. Replay and validation
# ---------------------------------------------------------------------------


def replay_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Re-execute and re-score every sealed program.  Returns the per-program comparison."""

    problems = declared_problems()
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for block in receipt["problems"]:
        label = block["run_label"]
        if label not in RUN_LABEL_PROBLEM:
            raise CapsetError(f"receipt carries an undeclared run label: {label}")
        problem = problems[RUN_LABEL_PROBLEM[label]]
        for record in block["sealed_programs"]:
            checked += 1
            rescored = score_construction(problem, record["source"])
            expected = [
                record["quality"],
                record["novelty"]["multiplier"],
                record["final_score"],
                record["novelty"]["reason"],
            ]
            actual = [
                _rational(rescored.quality),
                _rational(rescored.novelty_multiplier),
                _rational(rescored.final),
                rescored.novelty_reason,
            ]
            certificate = rescored.certificate.to_dict() if rescored.certificate else None
            if expected != actual or record["certificate"] != certificate:
                mismatches.append(
                    {
                        "run_label": label,
                        "program_sha256": record["program_sha256"],
                        "sealed": expected,
                        "replayed": actual,
                        "certificate_identical": record["certificate"] == certificate,
                    }
                )
    return {"programs_checked": checked, "mismatches": mismatches, "identical": not mismatches}


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seals, claims, the exact arithmetic of every score, and the multiplier control."""

    if value.get("schema_version") != RECEIPT_SCHEMA:
        raise CapsetError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CapsetError("receipt seal changed")
    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core_body):
        raise CapsetError("deterministic core seal changed")
    if value.get("claims") != CLAIMS:
        raise CapsetError("claims block changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise CapsetError("config binding changed")
    if value["sandbox"]["hostile_suite_run"] and not value["sandbox"][
        "all_hostile_programs_contained"
    ]:
        raise CapsetError("a hostile program escaped the sandbox")
    if not value["sandbox"]["incident_log"]["every_reason_is_declared"]:
        raise CapsetError("a sandbox incident carried a reason outside the declared vocabulary")

    labels = [block["run_label"] for block in value["problems"]]
    unknown = sorted(set(labels) - set(RUN_LABEL_PROBLEM))
    if unknown:
        raise CapsetError(f"receipt carries undeclared run labels: {unknown}")
    for block in value["problems"]:
        _validate_block(block)
    if {"capset_dimension_4", "capset_dimension_4_open_orbit"} <= set(labels):
        _validate_orbit_control(value)


def _validate_block(block: Mapping[str, Any]) -> None:
    """Re-derive every number in one arm from the sealed point lists alone."""

    label = block["run_label"]
    dimension = int(block["problem"]["dimension"])
    bound = elementary_upper_bound(dimension)
    if int(block["problem"]["elementary_upper_bound"]) != bound:
        raise CapsetError(f"{label}: the declared upper bound does not re-derive")
    certificate = block["problem"]["upper_bound_certificate"]
    if not certificate["verified"] or not certificate["covers_every_point_exactly_once"]:
        raise CapsetError(f"{label}: the upper-bound partition is not verified")
    record = block["sealed_record_revealed_after_scoring"]
    if canonical_sha256(record) != block["sealed_record_sha256"]:
        raise CapsetError(f"{label}: the sealed record seal changed")
    if record != sealed_record(dimension):
        raise CapsetError(f"{label}: the sealed record does not match the sealed table")
    # The orbit channel and the declared witness flag must agree.  Without this the flag is a
    # free text field, and a receipt could claim the orbit was off while every score in it was
    # produced with the orbit on.
    witness_declared = bool(block["problem"]["sealed_witness_declared"])
    orbit_reasons = {
        "monomial_orbit_of_the_sealed_witness",
        "distance_from_the_sealed_witness_orbit",
        "cardinality_gap_saturates_the_orbit_distance",
    }
    for item in block["sealed_programs"]:
        reason = item["novelty"]["reason"]
        if witness_declared and reason == "no_sealed_witness_declared":
            raise CapsetError(f"{label}: an arm with a sealed witness scored one without it")
        if not witness_declared and reason in orbit_reasons:
            raise CapsetError(f"{label}: an arm with no sealed witness scored against an orbit")

    best = 0
    best_searched = 0
    for item in block["sealed_programs"]:
        quality = _parse_rational(item["quality"])
        multiplier = _parse_rational(item["novelty"]["multiplier"])
        final = _parse_rational(item["final_score"])
        if quality * multiplier != final:
            raise CapsetError(
                f"{label}: final_score is not quality * multiplier for {item['program_sha256']}"
            )
        if not item["novelty"]["reason"]:
            raise CapsetError(f"{label}: a candidate carries no novelty reason")
        cert = item["certificate"]
        if cert is None:
            if quality != 0 or final != 0:
                raise CapsetError(f"{label}: a program with no certificate scored")
            continue
        if cert["reason"] and cert["reason"] not in INVALID_REASONS:
            raise CapsetError(f"{label}: undeclared invalidity reason {cert['reason']}")
        if not cert["valid"]:
            if quality != 0 or final != 0:
                raise CapsetError(
                    f"{label}: an invalid construction did not score zero: "
                    f"{item['program_sha256']}"
                )
            continue
        if cert["pairs_examined"] != cert["pairs_expected"]:
            raise CapsetError(
                f"{label}: a valid certificate is not exhaustive: {item['program_sha256']}"
            )
        if len(cert["points"]) != cert["cardinality"]:
            raise CapsetError(f"{label}: a certificate miscounts its own points")
        if canonical_sha256(cert["points"]) != cert["points_sha256"]:
            raise CapsetError(f"{label}: a certificate point seal changed")
        replayed = verify_cap(
            cert["points"], dimension, max_points=int(block["problem"]["max_points"])
        )
        if replayed.to_dict() != cert:
            raise CapsetError(
                f"{label}: a sealed certificate does not re-verify: {item['program_sha256']}"
            )
        if quality != Fraction(int(cert["cardinality"]), bound):
            raise CapsetError(
                f"{label}: quality is not cardinality / bound for {item['program_sha256']}"
            )
        best = max(best, int(cert["cardinality"]))
        if item["origin"] in ("seed", "proposed"):
            best_searched = max(best_searched, int(cert["cardinality"]))
    head = block["headline"]
    if head["best_verified_cardinality"] != best:
        raise CapsetError(f"{label}: the headline cardinality is not the measured one")
    if head["best_verified_cardinality_from_the_search"] != best_searched:
        raise CapsetError(f"{label}: the search-only cardinality is not the measured one")
    if head["beats_sealed_record"] != (best > int(record["cardinality"])):
        raise CapsetError(f"{label}: the record gate does not re-derive")
    if head["beats_sealed_record_from_the_search"] != (best_searched > int(record["cardinality"])):
        raise CapsetError(f"{label}: the search-only record gate does not re-derive")
    if head["elementary_upper_bound"] != bound:
        raise CapsetError(f"{label}: the headline upper bound does not re-derive")


def _validate_orbit_control(value: Mapping[str, Any]) -> None:
    """Run-aborting: turning the sealed orbit off must move a multiplier and nothing else."""

    by_label = {block["run_label"]: block for block in value["problems"]}
    live = {
        record["program_sha256"]: record
        for record in by_label["capset_dimension_4"]["sealed_programs"]
    }
    control = {
        record["program_sha256"]: record
        for record in by_label["capset_dimension_4_open_orbit"]["sealed_programs"]
    }
    shared = sorted(set(live) & set(control))
    if not shared:
        raise CapsetError("the control run shares no program with the dimension-4 arm")
    moved = 0
    for digest in shared:
        if live[digest]["quality"] != control[digest]["quality"]:
            raise CapsetError(
                f"the control changed a quality score, so it does not isolate novelty: {digest}"
            )
        if live[digest]["certificate"] != control[digest]["certificate"]:
            raise CapsetError(f"the control changed a verification certificate: {digest}")
        if live[digest]["novelty"]["multiplier"] != control[digest]["novelty"]["multiplier"]:
            moved += 1
    if not moved:
        raise CapsetError("the control moved no multiplier, so the sealed orbit had no effect")
    if by_label["capset_dimension_4_open_orbit"]["problem"]["sealed_witness_declared"]:
        raise CapsetError("the control declared a sealed witness, so it is not a control")
    if by_label["capset_dimension_4_open_orbit"]["generations_run"]:
        raise CapsetError("the control ran a search, so it does not isolate the multiplier")


# ---------------------------------------------------------------------------
# 14. CLI
# ---------------------------------------------------------------------------


def write_receipt(result: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A construction problem with an exact computable objective."
    )
    parser.add_argument("--output", default="runs/math/capset/construction-v1.json")
    parser.add_argument("--ledger", default="runs/math/capset/spend-ledger.json")
    parser.add_argument("--generations", type=int, default=CAMPAIGN_CONFIG.generations)
    parser.add_argument("--islands", type=int, default=CAMPAIGN_CONFIG.islands)
    parser.add_argument("--seed", type=int, default=CAMPAIGN_CONFIG.seed)
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--max-dollars-hundredths", type=int, default=400)
    parser.add_argument("--proposer", choices=("mock", "claude"), default="mock")
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--no-hostile-suite", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    parser.add_argument("--replay-checked", action="store_true")
    args = parser.parse_args(argv)

    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps({"validated": True, "output": args.output}))
        return 0
    if args.replay_checked:
        report = replay_from_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps(report, indent=2))
        return 0 if report["identical"] else 1

    result = run_campaign(
        config=replace(
            CAMPAIGN_CONFIG,
            generations=args.generations,
            islands=args.islands,
            seed=args.seed,
        ),
        ledger_path=args.ledger,
        max_calls=args.max_calls,
        max_dollars_hundredths=args.max_dollars_hundredths,
        proposer_kind=args.proposer,
        claude_executable=args.claude,
        include_hostile_suite=not args.no_hostile_suite,
    )
    validate_receipt(result)
    write_receipt(result, args.output)
    print(
        json.dumps(
            {
                "proposer_used": result["proposer"]["used"],
                "headline": result["headline"],
                "output": args.output,
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


__all__ = [
    "CAMPAIGN_CONFIG",
    "CLAIMS",
    "FORBIDDEN_CONSTRUCTION_VOCABULARY",
    "INVALID_REASONS",
    "RECEIPT_SCHEMA",
    "RUN_LABEL_PROBLEM",
    "SEALED_RECORDS",
    "SEARCH_ARMS",
    "WITNESS_DIMENSION_3",
    "WITNESS_DIMENSION_4",
    "CapCertificate",
    "CapsetError",
    "ConstructionProblem",
    "OrbitPolicy",
    "ScoredConstruction",
    "build_construction_prompt",
    "declared_problems",
    "digit_table",
    "elementary_upper_bound",
    "enumerate_lines",
    "guard_construction_prompt",
    "invalid_probe_program",
    "iter_lines",
    "linear_parts",
    "lines_in",
    "literal_witness_program",
    "main",
    "monomial_group_order",
    "monomial_orbit",
    "mutation_bank_for",
    "numeral_violations",
    "orbit_novelty",
    "parallel_class",
    "points_in",
    "points_mask",
    "probe_programs",
    "read_points",
    "replay_from_receipt",
    "run_campaign",
    "run_construction_problem",
    "scaled_witness_program",
    "score_construction",
    "sealed_record",
    "search_witness_program",
    "third_point",
    "translate",
    "translated_witness_program",
    "upper_bound_certificate",
    "validate_receipt",
    "verify_cap",
    "write_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
