"""Continued-fraction prior-art corpus: independently encoded classical identities plus
their exact transformation orbits.

Why this module exists.  The inverse-symbolic CF lane
(:mod:`sigma_theory_compiler.inverse_symbolic_engine`) enumerated 1.19e8 ordinals of a
declared continued-fraction family and promoted 214 survivors through a 120-digit holdout.
Those survivors were then screened against a **fifteen-entry** built-in table.  Absence from
a fifteen-entry table is not a statement about the literature, so 32 survivors sat in a
``NOT_IN_BUILTIN_TABLE`` bucket that means nothing on its own.  This module builds the
knowledge base that a real screen needs.

What is stored.  Every record asserts one identity::

    wrap(CF(a, b)) = value

where ``CF(a, b) = a_0 + b_1/(a_1 + b_2/(a_2 + ...))``, ``wrap`` is an exact rational
Moebius map, and ``value`` is a named constant or closed-form expression.  Coefficient
sequences are exact: each of ``a`` and ``b`` is a period-``p`` sequence of rational
functions of ``n`` over the rationals, plus a bounded prefix of explicit overrides.  All
arithmetic in this module is exact (``fractions.Fraction``); floating point appears only
when a stored identity is *checked* against mpmath at declared precision.

Source policy (``docs/EQUATION_UNIVERSE.md``).  DLMF and textbook sources are
metadata-only: this module stores **independently encoded mathematical facts plus citation
metadata**, never copied prose.  Every seed carries an author, a year, a reference
identifier, a validity domain, and a ``citation_confidence`` that says honestly whether the
reference is a pinned equation, a section-level pointer, or a general family theorem that
the record specializes.  Every seed is additionally *self-certifying*: :func:`build_corpus`
evaluates its continued fraction numerically and refuses to emit a seed whose stored closed
form does not reproduce to the declared digit count.  A mis-transcribed citation therefore
cannot smuggle in a wrong identity.

How ten thousand records are reached honestly.  Not by padding.  ~200 seed identities are
expanded by the *declared, exact* transformation group of continued fractions:

* ``equivalence(c)``  -- ``a_n -> c_n a_n``, ``b_n -> c_n c_{n-1} b_n``, value ``-> c_0 V``,
  for ``c`` drawn from a declared finite set of rational sequences;
* ``mobius(p, q, r, s)`` -- post-composition of the reported value, declared bounded
  integer coefficients;
* ``tail_shift(k)`` -- drop ``k`` levels; the value moves by the exact induced Moebius map;
* ``contract_even`` / ``contract_odd`` / ``extend`` -- the classical contraction formulas
  and their unit-denominator inverse;
* ``euler_minding`` -- the exact series<->continued-fraction correspondence, applied only
  where the associated series is in the declared grammar.

Every generated record stores its ``parent_id`` and the exact transformation applied, so the
corpus is a provenance **forest** rooted at cited seeds, never anonymous bulk.
:func:`resolve_to_seed` walks any record back to its root and
:func:`verify_forest_closure` proves the whole corpus does.

What this corpus is *not*.  It is finite.  Absence from it is ``absent_from_this_corpus``
and never novelty -- the same rule the equation universe already enforces.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp

from .sigma_core import canonical_json_bytes, canonical_sha256

CORPUS_SCHEMA = "invariant-cf-prior-art-corpus-1.0"

#: Explicit coefficient overrides live only in this index window; beyond it a sequence must
#: follow its generic rational formula exactly (checked on :data:`VERIFY_WINDOW`).
OVERRIDE_WINDOW = 8
VERIFY_WINDOW = 40

#: Working precision for the numeric certification of stored identities.
SEED_CHECK_DPS = 60
SEED_CHECK_DIGITS = 30
VALUE_STORE_DPS = 60
#: Two records are treated as sharing a value when they agree to this many decimal digits.
VALUE_MATCH_DIGITS = 50


class CorpusError(ValueError):
    """Raised on malformed patterns, out-of-class transforms, or corpus tamper."""


class OutOfDeclaredClass(CorpusError):
    """Raised when a transformation leaves the declared quasi-rational pattern class."""


# ---------------------------------------------------------------------------
# Exact polynomials and rational functions in the index n
# ---------------------------------------------------------------------------


def _trim(coefficients: Sequence[Fraction]) -> tuple[Fraction, ...]:
    items = list(coefficients)
    while items and items[-1] == 0:
        items.pop()
    return tuple(items)


@dataclass(frozen=True, slots=True)
class Poly:
    """Univariate polynomial in ``n`` with rational coefficients, ascending order."""

    coefficients: tuple[Fraction, ...]

    @staticmethod
    def of(*values: int | Fraction) -> Poly:
        return Poly(_trim([Fraction(v) for v in values]))

    @staticmethod
    def constant(value: int | Fraction) -> Poly:
        return Poly(_trim([Fraction(value)]))

    @staticmethod
    def linear(slope: int | Fraction, intercept: int | Fraction) -> Poly:
        return Poly(_trim([Fraction(intercept), Fraction(slope)]))

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def is_zero(self) -> bool:
        return not self.coefficients

    def evaluate(self, n: int | Fraction) -> Fraction:
        total = Fraction(0)
        for coefficient in reversed(self.coefficients):
            total = total * n + coefficient
        return total

    def __add__(self, other: Poly) -> Poly:
        width = max(len(self.coefficients), len(other.coefficients))
        out = []
        for index in range(width):
            left = self.coefficients[index] if index < len(self.coefficients) else Fraction(0)
            right = other.coefficients[index] if index < len(other.coefficients) else Fraction(0)
            out.append(left + right)
        return Poly(_trim(out))

    def __neg__(self) -> Poly:
        return Poly(tuple(-c for c in self.coefficients))

    def __sub__(self, other: Poly) -> Poly:
        return self + (-other)

    def __mul__(self, other: Poly) -> Poly:
        if self.is_zero() or other.is_zero():
            return Poly(())
        out = [Fraction(0)] * (len(self.coefficients) + len(other.coefficients) - 1)
        for i, left in enumerate(self.coefficients):
            if left == 0:
                continue
            for j, right in enumerate(other.coefficients):
                out[i + j] += left * right
        return Poly(_trim(out))

    def scale(self, factor: Fraction) -> Poly:
        return Poly(_trim([c * factor for c in self.coefficients]))

    def substitute(self, slope: Fraction, intercept: Fraction) -> Poly:
        """Return ``P(slope*n + intercept)``."""

        inner = Poly.linear(slope, intercept)
        result = Poly(())
        power = Poly.constant(1)
        for coefficient in self.coefficients:
            result = result + power.scale(coefficient)
            power = power * inner
        return result

    def divmod_by(self, other: Poly) -> tuple[Poly, Poly]:
        if other.is_zero():
            raise CorpusError("polynomial division by zero")
        remainder = list(self.coefficients)
        quotient = [Fraction(0)] * max(1, len(remainder) - len(other.coefficients) + 1)
        lead = other.coefficients[-1]
        while len(_trim(remainder)) >= len(other.coefficients) and _trim(remainder):
            remainder = list(_trim(remainder))
            shift = len(remainder) - len(other.coefficients)
            factor = remainder[-1] / lead
            quotient[shift] = factor
            for index, coefficient in enumerate(other.coefficients):
                remainder[shift + index] -= factor * coefficient
        return Poly(_trim(quotient)), Poly(_trim(remainder))

    def gcd_with(self, other: Poly) -> Poly:
        left, right = self, other
        while not right.is_zero():
            left, right = right, left.divmod_by(right)[1]
        if left.is_zero():
            return Poly.constant(1)
        return left.scale(Fraction(1) / left.coefficients[-1])

    def render(self) -> str:
        if self.is_zero():
            return "0"
        parts = []
        for power, coefficient in enumerate(self.coefficients):
            if coefficient == 0:
                continue
            symbol = "" if power == 0 else ("n" if power == 1 else f"n^{power}")
            parts.append(f"{coefficient}*{symbol}" if symbol else f"{coefficient}")
        return " + ".join(reversed(parts))


@dataclass(frozen=True, slots=True)
class Rat:
    """Rational function ``num(n)/den(n)`` in lowest terms with monic denominator."""

    num: Poly
    den: Poly

    @staticmethod
    def of(num: Poly, den: Poly | None = None) -> Rat:
        den = Poly.constant(1) if den is None else den
        if den.is_zero():
            raise CorpusError("rational function with zero denominator")
        if num.is_zero():
            return Rat(Poly(()), Poly.constant(1))
        common = num.gcd_with(den)
        num = num.divmod_by(common)[0]
        den = den.divmod_by(common)[0]
        lead = den.coefficients[-1]
        return Rat(num.scale(Fraction(1) / lead), den.scale(Fraction(1) / lead))

    @staticmethod
    def constant(value: int | Fraction) -> Rat:
        return Rat.of(Poly.constant(value))

    def is_zero(self) -> bool:
        return self.num.is_zero()

    def evaluate(self, n: int | Fraction) -> Fraction:
        denominator = self.den.evaluate(n)
        if denominator == 0:
            raise CorpusError(f"rational function pole at n={n}")
        return self.num.evaluate(n) / denominator

    def __add__(self, other: Rat) -> Rat:
        return Rat.of(self.num * other.den + other.num * self.den, self.den * other.den)

    def __mul__(self, other: Rat) -> Rat:
        return Rat.of(self.num * other.num, self.den * other.den)

    def __neg__(self) -> Rat:
        return Rat(-self.num, self.den)

    def reciprocal(self) -> Rat:
        if self.num.is_zero():
            raise CorpusError("reciprocal of the zero rational function")
        return Rat.of(self.den, self.num)

    def substitute(self, slope: int | Fraction, intercept: int | Fraction) -> Rat:
        slope = Fraction(slope)
        intercept = Fraction(intercept)
        return Rat.of(self.num.substitute(slope, intercept), self.den.substitute(slope, intercept))

    def key(self) -> str:
        return f"({self.num.render()})/({self.den.render()})"


# ---------------------------------------------------------------------------
# Quasi-rational coefficient sequences
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeqSpec:
    """A sequence ``n -> Fraction`` given by ``period`` rational functions plus overrides.

    ``terms[j]`` supplies the value for every index ``n`` with ``n % period == j``, except at
    indices listed in ``overrides`` (which are confined to ``[0, OVERRIDE_WINDOW)``).
    """

    period: int
    terms: tuple[Rat, ...]
    overrides: tuple[tuple[int, Fraction], ...] = ()

    @staticmethod
    def build(
        period: int,
        terms: Sequence[Rat],
        overrides: Mapping[int, Fraction] | None = None,
    ) -> SeqSpec:
        if period < 1 or len(terms) != period:
            raise CorpusError("period must be >= 1 and match the number of terms")
        raw = dict(overrides or {})
        for index in raw:
            if not 0 <= index < OVERRIDE_WINDOW:
                raise CorpusError(f"override index {index} outside the declared window")
        spec = SeqSpec(period, tuple(terms), tuple(sorted(raw.items())))
        return spec.canonical()

    def _generic(self, n: int) -> Fraction | None:
        try:
            return self.terms[n % self.period].evaluate(n)
        except CorpusError:
            return None

    def at(self, n: int) -> Fraction:
        for index, value in self.overrides:
            if index == n:
                return value
        generic = self._generic(n)
        if generic is None:
            raise CorpusError(f"sequence undefined at n={n}")
        return generic

    def canonical(self) -> SeqSpec:
        """Drop overrides that agree with the generic formula; keep the rest sorted."""

        kept: list[tuple[int, Fraction]] = []
        for index, value in sorted(dict(self.overrides).items()):
            if self._generic(index) != value:
                kept.append((index, value))
        return SeqSpec(self.period, self.terms, tuple(kept))

    def shifted(self, k: int) -> SeqSpec:
        """Return the sequence ``n -> self(n + k)`` for ``k >= 0``."""

        if k < 0:
            raise CorpusError("negative shift is not a declared transformation")
        terms = tuple(self.terms[(j + k) % self.period].substitute(1, k) for j in range(self.period))
        overrides: dict[int, Fraction] = {}
        for index in range(OVERRIDE_WINDOW):
            try:
                overrides[index] = self.at(index + k)
            except CorpusError:
                continue
        return SeqSpec.build(self.period, terms, overrides)

    def delayed(self, k: int) -> SeqSpec:
        """Return the sequence ``n -> self(n - k)`` for ``k >= 0`` (undefined below ``k``)."""

        if k < 0:
            raise CorpusError("negative delay is not a declared transformation")
        terms = tuple(self.terms[(j - k) % self.period].substitute(1, -k) for j in range(self.period))
        overrides: dict[int, Fraction] = {}
        for index in range(OVERRIDE_WINDOW):
            if index - k < 0:
                continue
            try:
                overrides[index] = self.at(index - k)
            except CorpusError:
                continue
        return SeqSpec.build(self.period, terms, overrides)

    def key(self) -> str:
        body = ";".join(term.key() for term in self.terms)
        extra = ",".join(f"{index}={value}" for index, value in self.overrides)
        return f"p{self.period}[{body}]{{{extra}}}"


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _combine(left: SeqSpec, right: SeqSpec, op: str) -> SeqSpec:
    period = _lcm(left.period, right.period)
    terms: list[Rat] = []
    for residue in range(period):
        a = left.terms[residue % left.period]
        b = right.terms[residue % right.period]
        terms.append(a * b if op == "mul" else a + b)
    overrides: dict[int, Fraction] = {}
    for index in range(OVERRIDE_WINDOW):
        try:
            value = left.at(index) * right.at(index) if op == "mul" else left.at(index) + right.at(index)
        except CorpusError:
            continue
        overrides[index] = value
    result = SeqSpec.build(period, terms, overrides)
    for index in range(OVERRIDE_WINDOW, VERIFY_WINDOW):
        expected = left.at(index) * right.at(index) if op == "mul" else left.at(index) + right.at(index)
        if result.at(index) != expected:
            raise OutOfDeclaredClass("pointwise combination left the declared class")
    return result


def seq_mul(left: SeqSpec, right: SeqSpec) -> SeqSpec:
    return _combine(left, right, "mul")


def seq_reciprocal(spec: SeqSpec, *, zero_index_value: Fraction | None = None) -> SeqSpec:
    """Pointwise reciprocal.  ``zero_index_value`` pins index 0 (used when ``a_0 = 0``)."""

    terms = tuple(
        term.reciprocal() if not term.is_zero() else Rat.constant(0) for term in spec.terms
    )
    if any(term.is_zero() for term in spec.terms):
        raise OutOfDeclaredClass("cannot invert an identically zero branch")
    overrides: dict[int, Fraction] = {}
    for index in range(OVERRIDE_WINDOW):
        if index == 0 and zero_index_value is not None:
            overrides[0] = zero_index_value
            continue
        try:
            value = spec.at(index)
        except CorpusError:
            continue
        if value == 0:
            raise OutOfDeclaredClass("cannot invert a sequence with a zero entry")
        overrides[index] = Fraction(1) / value
    return SeqSpec.build(spec.period, terms, overrides)


def seq_constant(value: int | Fraction) -> SeqSpec:
    return SeqSpec.build(1, (Rat.constant(value),))


def seq_from_poly(poly: Poly, overrides: Mapping[int, Fraction] | None = None) -> SeqSpec:
    return SeqSpec.build(1, (Rat.of(poly),), overrides)


def seq_from_rat(rat: Rat, overrides: Mapping[int, Fraction] | None = None) -> SeqSpec:
    return SeqSpec.build(1, (rat,), overrides)


# ---------------------------------------------------------------------------
# Continued-fraction patterns, Moebius wraps, and exact evaluation
# ---------------------------------------------------------------------------


def drop_index_zero(spec: SeqSpec) -> SeqSpec:
    """Strip an index-0 override.  Partial numerators start at ``b_1``; ``b_0`` is never
    read, so an override there is noise that must not distinguish two identical patterns."""

    kept = {index: value for index, value in spec.overrides if index != 0}
    return SeqSpec(spec.period, spec.terms, tuple(sorted(kept.items())))


def to_mpf(value: Fraction) -> mp.mpf:
    """Exact rational -> ``mpf`` at the current working precision."""

    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


@dataclass(frozen=True, slots=True)
class CFPattern:
    """``a`` supplies ``a_0, a_1, ...`` and ``b`` supplies ``b_1, b_2, ...``."""

    a: SeqSpec
    b: SeqSpec

    def key(self) -> str:
        return f"a={self.a.key()}|b={drop_index_zero(self.b).key()}"

    def evaluate(self, depth: int) -> mp.mpf:
        x = to_mpf(self.a.at(depth))
        for n in range(depth - 1, -1, -1):
            if x == 0:
                return mp.mpf("nan")
            x = to_mpf(self.a.at(n)) + to_mpf(self.b.at(n + 1)) / x
        return x


Mobius = tuple[Fraction, Fraction, Fraction, Fraction]

IDENTITY_MOBIUS: Mobius = (Fraction(1), Fraction(0), Fraction(0), Fraction(1))


def mobius_of(p: int | Fraction, q: int | Fraction, r: int | Fraction, s: int | Fraction) -> Mobius:
    entries = (Fraction(p), Fraction(q), Fraction(r), Fraction(s))
    if entries[0] * entries[3] - entries[1] * entries[2] == 0:
        raise CorpusError(f"degenerate Moebius map {entries}")
    return entries


def mobius_normalize(m: Mobius) -> Mobius:
    """Scale to integers with content 1 and a positive leading nonzero entry."""

    denominator = 1
    for value in m:
        denominator = _lcm(denominator, value.denominator)
    scaled = [int(value * denominator) for value in m]
    content = 0
    for value in scaled:
        content = math.gcd(content, abs(value))
    if content:
        scaled = [value // content for value in scaled]
    for value in scaled:
        if value != 0:
            if value < 0:
                scaled = [-item for item in scaled]
            break
    return (Fraction(scaled[0]), Fraction(scaled[1]), Fraction(scaled[2]), Fraction(scaled[3]))


def mobius_compose(outer: Mobius, inner: Mobius) -> Mobius:
    p1, q1, r1, s1 = outer
    p2, q2, r2, s2 = inner
    return mobius_normalize(
        (p1 * p2 + q1 * r2, p1 * q2 + q1 * s2, r1 * p2 + s1 * r2, r1 * q2 + s1 * s2)
    )


def mobius_inverse(m: Mobius) -> Mobius:
    p, q, r, s = m
    return mobius_normalize((s, -q, -r, p))


def mobius_apply(m: Mobius, value: mp.mpf) -> mp.mpf:
    p, q, r, s = m
    if not mp.isfinite(value):
        return mp.mpf("nan")
    denominator = to_mpf(r) * value + to_mpf(s)
    if denominator == 0:
        return mp.mpf("nan")
    return (to_mpf(p) * value + to_mpf(q)) / denominator


def mobius_render(m: Mobius) -> str:
    p, q, r, s = m
    if (r, s) == (Fraction(0), Fraction(1)):
        return f"{p}*x + {q}" if q else (f"{p}*x" if p != 1 else "x")
    return f"({p}*x + {q})/({r}*x + {s})"


# ---------------------------------------------------------------------------
# Exact polynomial fitting (used by the contraction/extension transformations)
# ---------------------------------------------------------------------------


def _newton_interpolate(points: Sequence[tuple[Fraction, Fraction]]) -> Poly:
    """Exact interpolating polynomial through distinct rational nodes.

    Newton's divided differences: ``f[x_j..x_{j+k}] = (f[x_{j+1}..x_{j+k}] -
    f[x_j..x_{j+k-1}])/(x_{j+k} - x_j)`` -- the denominator spans the whole window, not just
    the last step.
    """

    result = Poly(())
    basis = Poly.constant(1)
    coefficients = [value for _, value in points]
    for order, (node, _) in enumerate(points):
        result = result + basis.scale(coefficients[0])
        next_coefficients = []
        for start in range(len(coefficients) - 1):
            numerator = coefficients[start + 1] - coefficients[start]
            denominator = points[start + order + 1][0] - points[start][0]
            next_coefficients.append(numerator / denominator)
        coefficients = next_coefficients
        basis = basis * Poly.linear(Fraction(1), -node)
        if not coefficients:
            break
    return result


def fit_polynomial_sequence(
    values: Mapping[int, Fraction],
    *,
    period: int,
    max_degree: int,
    fit_start: int,
) -> SeqSpec:
    """Recover a period-``period`` polynomial pattern from exact sample values.

    Indices below ``fit_start`` become explicit overrides; indices at or above it must obey
    the fitted polynomial exactly or :class:`OutOfDeclaredClass` is raised.
    """

    if fit_start > OVERRIDE_WINDOW:
        raise CorpusError("fit_start must stay inside the override window")
    terms: list[Rat] = []
    for residue in range(period):
        nodes = sorted(n for n in values if n >= fit_start and n % period == residue)
        if len(nodes) < max_degree + 2:
            raise OutOfDeclaredClass("not enough samples to fit and verify")
        points = [(Fraction(n), values[n]) for n in nodes[: max_degree + 1]]
        poly = _newton_interpolate(points)
        if poly.degree > max_degree:
            raise OutOfDeclaredClass("fitted degree exceeds the declared bound")
        for node in nodes:
            if poly.evaluate(node) != values[node]:
                raise OutOfDeclaredClass("sample values are not polynomial in the index")
        terms.append(Rat.of(poly))
    overrides = {n: value for n, value in values.items() if n < fit_start}
    return SeqSpec.build(period, terms, overrides)


def _is_polynomial(spec: SeqSpec) -> bool:
    return all(term.den.degree == 0 for term in spec.terms)


# ---------------------------------------------------------------------------
# The declared transformation group
# ---------------------------------------------------------------------------


def transform_equivalence(pattern: CFPattern, c: SeqSpec) -> tuple[CFPattern, Mobius]:
    """``a_n -> c_n a_n``, ``b_n -> c_n c_{n-1} b_n``; the reported value scales by ``c_0``."""

    for index in range(VERIFY_WINDOW):
        if c.at(index) == 0:
            raise OutOfDeclaredClass(f"equivalence sequence vanishes at n = {index}")
    new_a = seq_mul(pattern.a, c)
    new_b = seq_mul(pattern.b, seq_mul(c, c.delayed(1)))
    return CFPattern(new_a, new_b), mobius_of(c.at(0), 0, 0, 1)


def transform_tail_shift(pattern: CFPattern, levels: int) -> tuple[CFPattern, Mobius]:
    """Drop ``levels`` leading levels; ``x_{i+1} = b_{i+1}/(x_i - a_i)``."""

    if levels < 1:
        raise CorpusError("tail shift must drop at least one level")
    transfer = IDENTITY_MOBIUS
    for index in range(levels):
        b_next = pattern.b.at(index + 1)
        if b_next == 0:
            raise OutOfDeclaredClass(f"tail shift blocked by b_{index + 1} = 0")
        step = mobius_of(0, b_next, 1, -pattern.a.at(index))
        transfer = mobius_compose(step, transfer)
    shifted = CFPattern(pattern.a.shifted(levels), pattern.b.shifted(levels))
    return shifted, transfer


def _convergents(pattern: CFPattern, count: int) -> tuple[list[Fraction], list[Fraction]]:
    """``(A, B)`` with ``A[i] = A_{i-1}`` of the classical convergent recurrence."""

    numerators = [Fraction(1), Fraction(pattern.a.at(0))]
    denominators = [Fraction(0), Fraction(1)]
    for n in range(1, count + 1):
        a_n = pattern.a.at(n)
        b_n = pattern.b.at(n)
        numerators.append(a_n * numerators[-1] + b_n * numerators[-2])
        denominators.append(a_n * denominators[-1] + b_n * denominators[-2])
    return numerators, denominators


def transform_contraction(pattern: CFPattern, parity: str) -> CFPattern:
    """Even (``A_{2m}/B_{2m}``) or odd (``A_{2m+1}/B_{2m+1}``) contraction, exactly.

    The contracted coefficients are recovered from the convergent recurrence by an exact
    2x2 solve and then refitted to the declared polynomial class; a contraction that leaves
    that class is refused rather than approximated.
    """

    if parity not in {"even", "odd"}:
        raise CorpusError("parity must be 'even' or 'odd'")
    if not (_is_polynomial(pattern.a) and _is_polynomial(pattern.b)):
        raise OutOfDeclaredClass("contraction is declared only for polynomial patterns")
    offset = 0 if parity == "even" else 1
    samples = 26
    numerators, denominators = _convergents(pattern, 2 * samples + offset + 2)

    # Normalize by B_offset so that the contracted continued fraction's own convergent
    # recurrence (which starts at C_{-1} = 1, D_{-1} = 0, D_0 = 1) is consistent with the
    # subsequence being reproduced.  Without this the odd contraction comes out scaled.
    d_base = denominators[offset + 1]
    if d_base == 0:
        raise OutOfDeclaredClass("contraction seed convergent is degenerate")

    def numerator(index: int) -> Fraction:
        return numerators[index + 1] / d_base

    def denominator(index: int) -> Fraction:
        return denominators[index + 1] / d_base

    d_values: dict[int, Fraction] = {0: numerator(offset)}
    e_values: dict[int, Fraction] = {}
    prev_num, prev_den = numerator(offset), denominator(offset)
    prev2_num, prev2_den = Fraction(1), Fraction(0)
    for m in range(1, samples):
        top = 2 * m + offset
        determinant = prev_num * prev2_den - prev2_num * prev_den
        if determinant == 0:
            raise OutOfDeclaredClass("degenerate contraction system")
        d_values[m] = (numerator(top) * prev2_den - prev2_num * denominator(top)) / determinant
        e_values[m] = (prev_num * denominator(top) - numerator(top) * prev_den) / determinant
        prev2_num, prev2_den = prev_num, prev_den
        prev_num, prev_den = numerator(top), denominator(top)

    # The contracted continued fraction is fixed only up to an equivalence transformation,
    # and the convergent-matching representative above is generally rational rather than
    # polynomial.  Re-scale by a declared normalizer with c_0 = 1 (so the value is
    # untouched) and keep the first choice that lands back in the polynomial class.
    def partial(index: int) -> Fraction:
        return Fraction(1) if index < 0 else pattern.a.at(index)

    normalizers: tuple[Callable[[int], Fraction], ...] = (
        lambda m: Fraction(1),
        lambda m: partial(2 * m - 2 + offset),
        lambda m: partial(2 * m + offset),
        lambda m: partial(2 * m - 1 + offset),
    )
    for normalizer in normalizers:
        scales = {m: (Fraction(1) if m == 0 else normalizer(m)) for m in d_values}
        if any(value == 0 for value in scales.values()):
            continue
        scaled_a = {m: scales[m] * value for m, value in d_values.items()}
        scaled_b = {
            m: scales[m] * scales[m - 1] * value for m, value in e_values.items() if m - 1 in scales
        }
        try:
            new_a = fit_polynomial_sequence(scaled_a, period=1, max_degree=6, fit_start=3)
            new_b = fit_polynomial_sequence(scaled_b, period=1, max_degree=12, fit_start=3)
        except OutOfDeclaredClass:
            continue
        contracted = CFPattern(new_a, new_b)
        _verify_contraction(contracted, numerators, denominators, offset)
        return contracted
    raise OutOfDeclaredClass("no declared normalizer keeps the contraction polynomial")


def _verify_contraction(
    contracted: CFPattern,
    numerators: Sequence[Fraction],
    denominators: Sequence[Fraction],
    offset: int,
    checks: int = 8,
) -> None:
    """The contracted convergents must reproduce the original's every-other convergents."""

    contracted_numerators, contracted_denominators = _convergents(contracted, checks + 1)
    for index in range(checks):
        top = 2 * index + offset
        left_den = contracted_denominators[index + 1]
        right_den = denominators[top + 1]
        if left_den == 0 or right_den == 0:
            continue
        if contracted_numerators[index + 1] / left_den != numerators[top + 1] / right_den:
            raise OutOfDeclaredClass("contracted convergents do not reproduce the original")


def transform_extension(pattern: CFPattern) -> CFPattern:
    """Unit-denominator inverse of the even contraction (``a_n = 1`` for ``n >= 1``).

    Solves ``e_1 = b_1``, ``d_1 = 1 + b_2``, ``e_m = -b_{2m-2} b_{2m-1}`` and
    ``d_m = 1 + b_{2m-1} + b_{2m}``, so the even contraction of the result is the input.
    """

    b_values: dict[int, Fraction] = {1: pattern.b.at(1), 2: pattern.a.at(1) - 1}
    for m in range(2, 2 + VERIFY_WINDOW):
        previous = b_values[2 * m - 2]
        if previous == 0:
            raise OutOfDeclaredClass("extension blocked by a vanishing partial numerator")
        odd = -pattern.b.at(m) / previous
        b_values[2 * m - 1] = odd
        b_values[2 * m] = pattern.a.at(m) - 1 - odd
    a_values: dict[int, Fraction] = {0: pattern.a.at(0)}
    for index in range(1, 2 * (2 + VERIFY_WINDOW)):
        a_values[index] = Fraction(1)
    new_a = fit_polynomial_sequence(a_values, period=1, max_degree=0, fit_start=1)
    for period, degree in ((1, 1), (2, 1), (1, 2), (2, 2), (2, 3), (4, 2)):
        try:
            new_b = fit_polynomial_sequence(b_values, period=period, max_degree=degree, fit_start=3)
        except OutOfDeclaredClass:
            continue
        return CFPattern(new_a, new_b)
    raise OutOfDeclaredClass("the unit-denominator extension leaves the declared class")


def transform_euler_minding(t0: Fraction, ratio: Rat, first_index: int) -> CFPattern:
    """Euler's exact series<->continued-fraction correspondence.

    For ``S = sum_{k >= first_index} t_k`` with ``t_k / t_{k-1} = ratio(k)``,
    ``S = t_0/(1 - r_1/(1 + r_1 - r_2/(1 + r_2 - ...)))``.  Encoded with ``a_0 = 0``,
    ``b_1 = t_0``, ``a_1 = 1`` and, for ``n >= 2``, ``a_n = 1 + r_n``, ``b_n = -r_n``
    where ``r_n = ratio(n - 1 + first_index)``.
    """

    shifted = ratio.substitute(1, first_index - 1)
    new_a = seq_from_rat(Rat.constant(1) + shifted, {0: Fraction(0), 1: Fraction(1)})
    new_b = seq_from_rat(-shifted, {1: Fraction(t0)})
    return CFPattern(new_a, new_b)


# ---------------------------------------------------------------------------
# Equivalence normal form
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalForm:
    """Canonical representative of a continued fraction's equivalence class.

    The classical equivalence transformation with ``c_n = 1/a_n`` sends every pattern with
    non-vanishing partial denominators to one with ``a_n = 1`` for ``n >= 1``, leaving
    ``r_n = b_n/(a_n a_{n-1})`` -- an exact invariant of the class -- and rescaling the value
    by ``1/a_0``.  Two continued fractions are equivalence-related exactly when their
    ``(leading, r)`` pairs agree.
    """

    leading: Fraction
    r_sequence: SeqSpec
    scale: Fraction

    def key(self) -> str:
        return f"lead={self.leading}|r={self.r_sequence.key()}"


def normal_form(pattern: CFPattern) -> NormalForm:
    """Equivalence normal form; raises when a partial denominator vanishes."""

    for index in range(1, VERIFY_WINDOW):
        if pattern.a.at(index) == 0:
            raise OutOfDeclaredClass(f"normal form blocked by a_{index} = 0")
    a_zero = pattern.a.at(0)
    if a_zero == 0:
        # With a_0 = 0 the equivalence group still carries a free overall scale c_0, so the
        # class has no canonical representative until it is pinned.  Pin it by forcing
        # b_1 -> 1; then (leading, r) and the rescaled value are both exact class invariants.
        first = pattern.b.at(1)
        if first == 0:
            raise OutOfDeclaredClass("normal form blocked by a_0 = 0 with b_1 = 0")
        scale = pattern.a.at(1) / first
        c = seq_reciprocal(pattern.a, zero_index_value=scale)
        leading = Fraction(0)
    else:
        scale = Fraction(1) / a_zero
        c = seq_reciprocal(pattern.a)
        leading = Fraction(1)
    normalized = drop_index_zero(seq_mul(pattern.b, seq_mul(c, c.delayed(1))))
    return NormalForm(leading, normalized, scale)


# ---------------------------------------------------------------------------
# Records and the provenance forest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Citation:
    """Citation metadata for an independently encoded identity.

    ``confidence`` is deliberately explicit and never optimistic:

    ``pinned_identity``
        the exact identity is stated in the cited source at the cited location;
    ``section_reference``
        the cited section is the source's continued-fraction section for this function
        class and states this identity or its immediate parametric form;
    ``family_theorem``
        the record is a declared specialization of the cited general theorem rather than a
        verbatim entry;
    ``elementary_derivation``
        the identity follows in one step from the cited elementary theory.
    """

    author: str
    year: str
    reference: str
    confidence: str
    note: str

    def as_json(self) -> dict[str, str]:
        return {
            "author": self.author,
            "year": self.year,
            "reference": self.reference,
            "confidence": self.confidence,
            "note": self.note,
        }


CITATION_CONFIDENCES = (
    "pinned_identity",
    "section_reference",
    "family_theorem",
    "elementary_derivation",
)


@dataclass(frozen=True, slots=True)
class CFRecord:
    """One stored identity ``wrap(CF(pattern)) = value`` with full provenance."""

    record_id: str
    kind: str
    family: str
    seed_id: str
    parent_id: str | None
    depth: int
    transform: tuple[tuple[str, str], ...]
    pattern: CFPattern | None
    wrap: Mobius
    cf_value: str
    value: str
    value_expr: str
    citation: Citation
    validity_domain: str
    grammar: str
    normal_form_key: str | None = None
    normalized_value: str | None = None
    cf_mobius_from_seed: Mobius = IDENTITY_MOBIUS

    def pattern_key(self) -> str:
        return "OUT_OF_DECLARED_GRAMMAR" if self.pattern is None else self.pattern.key()

    def as_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "family": self.family,
            "seed_id": self.seed_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "transform": {name: value for name, value in self.transform},
            "pattern_key": self.pattern_key(),
            "a_pattern": None if self.pattern is None else self.pattern.a.key(),
            "b_pattern": None if self.pattern is None else drop_index_zero(self.pattern.b).key(),
            "wrap": [str(item) for item in self.wrap],
            "cf_value": self.cf_value,
            "value": self.value,
            "value_expr": self.value_expr,
            "citation": self.citation.as_json(),
            "validity_domain": self.validity_domain,
            "grammar": self.grammar,
            "normal_form_key": self.normal_form_key,
            "normalized_value": self.normalized_value,
            "cf_mobius_from_seed": [str(item) for item in self.cf_mobius_from_seed],
        }


def resolve_to_seed(records: Mapping[str, CFRecord], record_id: str) -> list[str]:
    """Walk a record back to its seed, returning the chain of record ids (root first)."""

    chain: list[str] = []
    seen: set[str] = set()
    current = record_id
    while True:
        if current in seen:
            raise CorpusError(f"provenance cycle at {current}")
        seen.add(current)
        record = records.get(current)
        if record is None:
            raise CorpusError(f"dangling provenance edge to {current}")
        chain.append(current)
        if record.kind == "seed":
            if record.parent_id is not None:
                raise CorpusError(f"seed {current} carries a parent")
            return list(reversed(chain))
        if record.parent_id is None:
            raise CorpusError(f"non-seed record {current} has no parent")
        current = record.parent_id


def verify_forest_closure(records: Sequence[CFRecord]) -> dict[str, Any]:
    """Prove every non-seed record resolves to a seed by declared transformations."""

    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(records):
        raise CorpusError("duplicate record ids in the corpus")
    declared = set(DECLARED_TRANSFORMATIONS)
    longest = 0
    for record in records:
        chain = resolve_to_seed(by_id, record.record_id)
        longest = max(longest, len(chain) - 1)
        if len(chain) - 1 != record.depth:
            raise CorpusError(f"depth mismatch for {record.record_id}")
        if record.kind != "seed":
            name = dict(record.transform).get("transformation")
            if name not in declared:
                raise CorpusError(f"undeclared transformation {name!r} on {record.record_id}")
            if by_id[chain[0]].seed_id != record.seed_id:
                raise CorpusError(f"seed attribution mismatch for {record.record_id}")
    return {"records": len(records), "max_chain_length": longest}


DECLARED_TRANSFORMATIONS = (
    "equivalence",
    "mobius_post_composition",
    "tail_shift",
    "contract_even",
    "contract_odd",
    "extend",
    "euler_minding",
)


# ---------------------------------------------------------------------------
# Seed catalogue: independently encoded classical continued fractions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Seed:
    """A cited classical identity, self-certified numerically at build time."""

    seed_id: str
    family: str
    pattern: CFPattern | None
    value_expr: str
    value_fn: Callable[[], mp.mpf]
    citation: Citation
    validity_domain: str
    check_mode: str = "direct"
    check_depth: int = 2500
    check_digits: int = 25
    grammar: str = "declared_quasi_rational"


def _poly_seq(*coefficients: int | Fraction, **overrides: Any) -> SeqSpec:
    fixed = {int(key[1:]): Fraction(value) for key, value in overrides.items()}
    return seq_from_poly(Poly.of(*coefficients), fixed)


def _seeds_periodic_surd() -> list[Seed]:
    """``x = a + b/x`` -- the period-one purely periodic continued fraction."""

    citation = Citation(
        author="Euler; Lagrange",
        year="1737/1770",
        reference="periodic continued fractions and quadratic irrationals; DLMF 1.12",
        confidence="elementary_derivation",
        note=(
            "x = a + b/x is equivalent to x^2 - a x - b = 0; the continued fraction "
            "converges to the root of larger modulus whenever a^2 + 4b > 0"
        ),
    )
    seeds: list[Seed] = []
    for a in range(-4, 5):
        if a == 0:
            continue
        for b in range(-4, 5):
            if b == 0 or a * a + 4 * b <= 0:
                continue

            def value(a: int = a, b: int = b) -> mp.mpf:
                root = mp.sqrt(a * a + 4 * b)
                return (mp.mpf(a) + (root if a > 0 else -root)) / 2

            sign = "+" if a > 0 else "-"
            seeds.append(
                Seed(
                    seed_id=f"periodic_surd_a{a}_b{b}",
                    family="periodic_quadratic_surd",
                    pattern=CFPattern(seq_constant(a), seq_constant(b)),
                    value_expr=f"({a} {sign} sqrt({a * a + 4 * b}))/2",
                    value_fn=value,
                    citation=citation,
                    validity_domain=f"a={a}, b={b}, discriminant {a * a + 4 * b} > 0",
                )
            )
    return seeds


def _sqrt_simple_cf(d: int) -> tuple[int, list[int]]:
    """Standard periodic simple-continued-fraction expansion of ``sqrt(d)``."""

    a0 = math.isqrt(d)
    m, q, a = 0, 1, a0
    period: list[int] = []
    while True:
        m = q * a - m
        q = (d - m * m) // q
        a = (a0 + m) // q
        period.append(a)
        if a == 2 * a0:
            return a0, period


def _seeds_simple_sqrt() -> list[Seed]:
    citation = Citation(
        author="Lagrange",
        year="1770",
        reference="periodicity of the simple continued fraction of a quadratic surd; DLMF 1.12",
        confidence="pinned_identity",
        note="regular continued fraction [a0; a1, a2, ...] of sqrt(d), all partial numerators 1",
    )
    seeds: list[Seed] = []
    for d in range(2, 31):
        if math.isqrt(d) ** 2 == d:
            continue
        a0, period = _sqrt_simple_cf(d)
        length = len(period)
        terms = tuple(Rat.constant(period[(j - 1) % length]) for j in range(length))
        a_spec = SeqSpec.build(length, terms, {0: Fraction(a0)})
        seeds.append(
            Seed(
                seed_id=f"simple_cf_sqrt_{d}",
                family="regular_cf_quadratic_surd",
                pattern=CFPattern(a_spec, seq_constant(1)),
                value_expr=f"sqrt({d})",
                value_fn=lambda d=d: mp.sqrt(d),
                citation=citation,
                validity_domain=f"d={d} not a perfect square; period {length}",
                check_depth=400,
                check_digits=40,
            )
        )
    return seeds


def _seeds_euler_e() -> list[Seed]:
    euler = Citation(
        author="Euler",
        year="1737",
        reference="De fractionibus continuis dissertatio (E71); DLMF 4.9",
        confidence="pinned_identity",
        note="Euler's continued fractions for e and its Moebius relatives",
    )
    lambert = Citation(
        author="Lambert; Euler",
        year="1761",
        reference="DLMF 4.36 (continued fractions for hyperbolic functions)",
        confidence="section_reference",
        note="coth(1/2) = 2 + 1/(6 + 1/(10 + 1/(14 + ...)))",
    )
    period3 = SeqSpec.build(
        3,
        (
            Rat.constant(1),
            Rat.constant(1),
            Rat.of(Poly.linear(Fraction(2, 3), Fraction(2, 3))),
        ),
        {0: Fraction(2)},
    )
    return [
        Seed(
            "euler_e_tail_1737",
            "euler_e",
            CFPattern(_poly_seq(1, 1), _poly_seq(0, 1)),
            "1/(e - 2)",
            lambda: 1 / (mp.e - 2),
            euler,
            "Euler 1737 tail form: e = 2 + 1/(1 + 1/(2 + 2/(3 + 3/(4 + ...))))",
        ),
        Seed(
            "euler_e_alternating",
            "euler_e",
            CFPattern(_poly_seq(3, 1), _poly_seq(0, -1)),
            "e",
            lambda: +mp.e,
            euler,
            "e = 3 - 1/(4 - 2/(5 - 3/(6 - ...)))",
        ),
        Seed(
            "euler_e_over_e_minus_one",
            "euler_e",
            CFPattern(_poly_seq(2, 1), _poly_seq(0, -1)),
            "e/(e - 1)",
            lambda: mp.e / (mp.e - 1),
            euler,
            "x = 2 - 1/(3 - 2/(4 - 3/(5 - ...)))",
        ),
        Seed(
            "euler_e_minus_one",
            "euler_e",
            CFPattern(_poly_seq(1, 1), _poly_seq(1, 1)),
            "e - 1",
            lambda: mp.e - 1,
            euler,
            "e = 2 + 2/(2 + 3/(3 + 4/(4 + 5/(5 + ...))))",
        ),
        Seed(
            "euler_one_over_e_minus_one",
            "euler_e",
            CFPattern(_poly_seq(0, 1), _poly_seq(0, 1)),
            "1/(e - 1)",
            lambda: 1 / (mp.e - 1),
            euler,
            "x = 1/(1 + 2/(2 + 3/(3 + 4/(4 + ...))))",
        ),
        Seed(
            "lambert_coth_half",
            "euler_e",
            CFPattern(_poly_seq(2, 4), _poly_seq(1)),
            "coth(1/2) = (e + 1)/(e - 1)",
            lambda: mp.coth(mp.mpf(1) / 2),
            lambert,
            "z = 1/2 member of the hyperbolic-cotangent continued fraction",
        ),
        Seed(
            "euler_e_regular_cf",
            "euler_e",
            CFPattern(period3, seq_constant(1)),
            "e",
            lambda: +mp.e,
            Citation(
                author="Euler",
                year="1737",
                reference="regular continued fraction e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]; OEIS A003417",
                confidence="pinned_identity",
                note="period-three quasi-polynomial pattern: a_n = (2n + 2)/3 when n = 2 (mod 3), else 1",
            ),
            "n >= 1",
            check_depth=400,
            check_digits=40,
        ),
        Seed(
            "lambert_coth_1",
            "euler_e",
            CFPattern(_poly_seq(1, 2), _poly_seq(1)),
            "coth(1)",
            lambda: mp.coth(1),
            lambert,
            "z = 1 member: coth(1) = 1 + 1/(3 + 1/(5 + 1/(7 + ...)))",
        ),
    ]


def _seeds_lambert() -> list[Seed]:
    """Lambert's continued fractions for ``tan`` and ``tanh`` at ``z = 1/q``."""

    tan_citation = Citation(
        author="Lambert",
        year="1761",
        reference="DLMF 4.25 (continued fractions for trigonometric functions)",
        confidence="section_reference",
        note="tan z = z/(1 - z^2/(3 - z^2/(5 - ...))); cleared to integers at z = 1/q",
    )
    tanh_citation = Citation(
        author="Lambert",
        year="1761",
        reference="DLMF 4.36 (continued fractions for hyperbolic functions)",
        confidence="section_reference",
        note="tanh z = z/(1 + z^2/(3 + z^2/(5 + ...))); cleared to integers at z = 1/q",
    )
    seeds: list[Seed] = []
    for q in range(1, 13):
        a_spec = _poly_seq(-q, 2 * q, a0=0)
        seeds.append(
            Seed(
                f"lambert_tan_1_over_{q}",
                "lambert_trigonometric",
                CFPattern(a_spec, seq_from_poly(Poly.constant(-1), {1: Fraction(1)})),
                f"tan(1/{q})",
                lambda q=q: mp.tan(mp.mpf(1) / q),
                tan_citation,
                f"z = 1/{q}, |z| < pi/2",
            )
        )
        seeds.append(
            Seed(
                f"lambert_tanh_1_over_{q}",
                "lambert_hyperbolic",
                CFPattern(a_spec, seq_constant(1)),
                f"tanh(1/{q})",
                lambda q=q: mp.tanh(mp.mpf(1) / q),
                tanh_citation,
                f"z = 1/{q}, z real",
            )
        )
    return seeds


def _seeds_arctan() -> list[Seed]:
    arctan_citation = Citation(
        author="Euler; Gauss",
        year="1748/1813",
        reference="DLMF 4.25 (continued fractions for inverse trigonometric functions)",
        confidence="section_reference",
        note="arctan z = z/(1 + z^2/(3 + 4z^2/(5 + 9z^2/(7 + ...)))); cleared at z = 1/q",
    )
    arctanh_citation = Citation(
        author="Euler; Gauss",
        year="1748/1813",
        reference="DLMF 4.36 (continued fractions for inverse hyperbolic functions)",
        confidence="section_reference",
        note="arctanh z = z/(1 - z^2/(3 - 4z^2/(5 - 9z^2/(7 - ...)))); cleared at z = 1/q",
    )
    seeds: list[Seed] = []
    for q in range(1, 11):
        a_spec = _poly_seq(-q, 2 * q, a0=0)
        seeds.append(
            Seed(
                f"arctan_1_over_{q}",
                "inverse_trigonometric",
                CFPattern(a_spec, _poly_seq(1, -2, 1, b1=1)),
                f"arctan(1/{q})",
                lambda q=q: mp.atan(mp.mpf(1) / q),
                arctan_citation,
                f"z = 1/{q}",
                check_depth=3000,
                check_digits=12,
            )
        )
    for q in range(2, 12):
        a_spec = _poly_seq(-q, 2 * q, a0=0)
        seeds.append(
            Seed(
                f"arctanh_1_over_{q}",
                "inverse_hyperbolic",
                CFPattern(a_spec, _poly_seq(-1, 2, -1, b1=1)),
                f"arctanh(1/{q})",
                lambda q=q: mp.atanh(mp.mpf(1) / q),
                arctanh_citation,
                f"z = 1/{q}, |z| < 1",
            )
        )
    return seeds


def _seeds_gauss_log() -> list[Seed]:
    citation = Citation(
        author="Gauss",
        year="1813",
        reference="DLMF 4.9 (continued fraction for the logarithm)",
        confidence="section_reference",
        note=(
            "log(1+z) = z/(1 + z/(2 + z/(3 + 4z/(4 + 4z/(5 + 9z/(6 + 9z/(7 + ...)))))));"
            " partial numerators are period-two in the index"
        ),
    )
    seeds: list[Seed] = []
    for numerator, denominator in ((1, 1), (1, 2), (-1, 2), (2, 1), (1, 3), (-1, 3), (3, 1), (-2, 3)):
        z = Fraction(numerator, denominator)
        even_branch = Rat.of(Poly.of(0, 0, z / 4))
        odd_branch = Rat.of(Poly.of(z / 4, -z / 2, z / 4))
        b_spec = SeqSpec.build(2, (even_branch, odd_branch), {1: z})
        label = f"{numerator}_{denominator}".replace("-", "m")
        seeds.append(
            Seed(
                f"gauss_log_z_{label}",
                "gauss_logarithm",
                CFPattern(_poly_seq(0, 1), b_spec),
                f"log(1 + {z})",
                lambda z=z: mp.log(1 + to_mpf(z)),
                citation,
                f"z = {z}, z > -1",
            )
        )
    return seeds


def _seeds_erfc() -> list[Seed]:
    citation = Citation(
        author="Laplace; Stieltjes",
        year="1805/1894",
        reference="DLMF 7.9 (continued fractions for the error functions)",
        confidence="section_reference",
        note="sqrt(pi) exp(z^2) erfc(z) = 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + ...))))",
    )
    seeds: list[Seed] = []
    for numerator, denominator in ((1, 2), (1, 1), (3, 2), (2, 1), (3, 1)):
        z = Fraction(numerator, denominator)
        seeds.append(
            Seed(
                f"erfc_z_{numerator}_{denominator}",
                "error_function",
                CFPattern(
                    seq_from_rat(Rat.constant(z), {0: Fraction(0)}),
                    seq_from_poly(Poly.linear(Fraction(1, 2), Fraction(-1, 2)), {1: Fraction(1)}),
                ),
                f"sqrt(pi)*exp(z^2)*erfc(z) at z = {z}",
                lambda z=z: mp.sqrt(mp.pi) * mp.exp(to_mpf(z) ** 2) * mp.erfc(to_mpf(z)),
                citation,
                f"z = {z} > 0",
                check_depth=3000,
                check_digits=10,
            )
        )
    return seeds


def _seeds_bessel() -> list[Seed]:
    citation_j = Citation(
        author="Bessel; Gauss",
        year="1824/1813",
        reference="DLMF 10.10 (continued fraction for ratios of Bessel functions)",
        confidence="section_reference",
        note="J_nu(z)/J_{nu-1}(z) = 1/(2 nu/z - 1/(2(nu+1)/z - 1/(2(nu+2)/z - ...)))",
    )
    citation_i = Citation(
        author="Bessel; Gauss",
        year="1824/1813",
        reference="DLMF 10.10 (modified-Bessel ratio; sign-flipped partial numerators)",
        confidence="family_theorem",
        note="I_nu(z)/I_{nu-1}(z) = 1/(2 nu/z + 1/(2(nu+1)/z + 1/(2(nu+2)/z + ...)))",
    )
    seeds: list[Seed] = []
    for nu_num, nu_den in ((1, 1), (1, 2), (3, 2), (2, 1)):
        nu = Fraction(nu_num, nu_den)
        for z in (1, 2, 3):
            slope = Fraction(2, z)
            a_spec = seq_from_poly(Poly.linear(slope, slope * (nu - 1)), {0: Fraction(0)})
            seeds.append(
                Seed(
                    f"bessel_j_ratio_nu{nu_num}_{nu_den}_z{z}",
                    "bessel_ratio",
                    CFPattern(a_spec, seq_from_poly(Poly.constant(-1), {1: Fraction(1)})),
                    f"J_{nu}({z})/J_{nu - 1}({z})",
                    lambda nu=nu, z=z: mp.besselj(to_mpf(nu), z) / mp.besselj(to_mpf(nu - 1), z),
                    citation_j,
                    f"nu = {nu}, z = {z}",
                )
            )
    for nu_num, nu_den in ((1, 1), (1, 2)):
        nu = Fraction(nu_num, nu_den)
        for z in (1, 2):
            slope = Fraction(2, z)
            a_spec = seq_from_poly(Poly.linear(slope, slope * (nu - 1)), {0: Fraction(0)})
            seeds.append(
                Seed(
                    f"bessel_i_ratio_nu{nu_num}_{nu_den}_z{z}",
                    "bessel_ratio",
                    CFPattern(a_spec, seq_from_poly(Poly.constant(1), {1: Fraction(1)})),
                    f"I_{nu}({z})/I_{nu - 1}({z})",
                    lambda nu=nu, z=z: mp.besseli(to_mpf(nu), z) / mp.besseli(to_mpf(nu - 1), z),
                    citation_i,
                    f"nu = {nu}, z = {z}",
                )
            )
    return seeds


def _seeds_cotangent_family() -> list[Seed]:
    """``1 + (1^2 - A^2)/(3 + (2^2 - A^2)/(5 + ...)) = A cot(pi A/4)``.

    The ``A = 0`` member is the Euler-Gauss arctan continued fraction at ``z = 1``
    (``4/pi``); the imaginary members ``A = iC`` give ``C coth(pi C/4)`` and are the only
    place in this corpus where an integer-coefficient continued fraction produces
    ``exp(pi)``.  Terminating checks: at ``A = 1, 2, 3`` a partial numerator vanishes and the
    continued fraction is finite with value ``cot(pi/4) = 1``, ``2 cot(pi/2) = 0``,
    ``3 cot(3 pi/4) = -3``.
    """

    citation = Citation(
        author="Gauss (family); classical cotangent specialization",
        year="1813",
        reference="DLMF 15.7 (continued fractions for ratios of contiguous 2F1) specialized",
        confidence="family_theorem",
        note=(
            "a_n = 2n + 1, b_n = n^2 - A^2 with value A cot(pi A/4); reduces to the "
            "arctan continued fraction 4/pi at A = 0 and is verified here to 25 digits"
        ),
    )
    seeds: list[Seed] = []
    reals = ((0, 1), (1, 2), (1, 1), (3, 2), (1, 3), (2, 3), (4, 3), (5, 3), (5, 2), (7, 2))
    for numerator, denominator in reals:
        A = Fraction(numerator, denominator)

        def value(A: Fraction = A) -> mp.mpf:
            if A == 0:
                return 4 / mp.pi
            return to_mpf(A) * mp.cot(mp.pi * to_mpf(A) / 4)

        seeds.append(
            Seed(
                f"cotangent_cf_A_{numerator}_{denominator}",
                "cotangent_cf",
                CFPattern(_poly_seq(1, 2), seq_from_poly(Poly.of(-A * A, 0, 1))),
                f"A*cot(pi*A/4) at A = {A}" if A else "4/pi",
                value,
                citation,
                f"A = {A}; A not an even integer",
                check_depth=3000,
                check_digits=20,
            )
        )
    for numerator, denominator in ((1, 2), (1, 1), (3, 2), (2, 1), (3, 1), (4, 1)):
        C = Fraction(numerator, denominator)
        seeds.append(
            Seed(
                f"cotangent_cf_iC_{numerator}_{denominator}",
                "cotangent_cf",
                CFPattern(_poly_seq(1, 2), seq_from_poly(Poly.of(C * C, 0, 1))),
                f"C*coth(pi*C/4) at C = {C}",
                lambda C=C: to_mpf(C) * mp.coth(mp.pi * to_mpf(C) / 4),
                citation,
                f"A = i*{C}",
                check_depth=3000,
                check_digits=20,
            )
        )
    return seeds


def _seeds_apery() -> list[Seed]:
    citation = Citation(
        author="Apery; van der Poorten",
        year="1978/1979",
        reference="A proof that Euler missed (Math. Intelligencer 1, 195-203)",
        confidence="pinned_identity",
        note="the accelerating continued fractions behind the irrationality proofs",
    )
    return [
        Seed(
            "apery_zeta3",
            "apery",
            CFPattern(_poly_seq(5, 27, 51, 34), seq_from_poly(Poly.of(0, 0, 0, 0, 0, 0, -1))),
            "6/zeta(3)",
            lambda: 6 / mp.zeta(3),
            citation,
            "a_n = 34n^3 + 51n^2 + 27n + 5, b_n = -n^6",
            check_depth=200,
            check_digits=40,
        ),
        Seed(
            "apery_zeta2",
            "apery",
            CFPattern(_poly_seq(3, 11, 11), seq_from_poly(Poly.of(0, 0, 0, 0, 1))),
            "5/zeta(2)",
            lambda: 5 / mp.zeta(2),
            citation,
            "a_n = 11n^2 + 11n + 3, b_n = n^4",
            check_depth=200,
            check_digits=40,
        ),
    ]


@dataclass(frozen=True, slots=True)
class SeriesSeed:
    """A series in the declared grammar: ratio of consecutive terms is rational in the index."""

    series_id: str
    first_index: int
    first_term: Fraction
    ratio: Rat
    value_expr: str
    value_fn: Callable[[], mp.mpf]
    citation: Citation
    validity_domain: str


def _rat(num: Sequence[int | Fraction], den: Sequence[int | Fraction]) -> Rat:
    return Rat.of(Poly.of(*num), Poly.of(*den))


def _series_seeds() -> list[SeriesSeed]:
    euler_minding = "Euler's series-to-continued-fraction correspondence (Introductio, 1748)"
    return [
        SeriesSeed(
            "leibniz_pi_quarter",
            0,
            Fraction(1),
            _rat((1, -2), (1, 2)),
            "pi/4",
            lambda: mp.pi / 4,
            Citation("Madhava; Gregory; Leibniz", "1400s/1671/1674",
                     "alternating series pi/4 = 1 - 1/3 + 1/5 - ...; " + euler_minding,
                     "pinned_identity", "term ratio r_k = -(2k-1)/(2k+1)"),
            "alternating, conditionally convergent",
        ),
        SeriesSeed(
            "mercator_ln2",
            0,
            Fraction(1),
            _rat((0, -1), (1, 1)),
            "log(2)",
            lambda: mp.log(2),
            Citation("Mercator; Newton", "1668",
                     "alternating harmonic series log 2 = 1 - 1/2 + 1/3 - ...; " + euler_minding,
                     "pinned_identity", "term ratio r_k = -k/(k+1)"),
            "alternating, conditionally convergent",
        ),
        SeriesSeed(
            "euler_pi_half",
            0,
            Fraction(1),
            _rat((0, 1), (1, 2)),
            "pi/2",
            lambda: mp.pi / 2,
            Citation("Euler", "1748",
                     "pi/2 = sum_{k>=0} 2^k (k!)^2/(2k+1)! = sum n!/(2n+1)!!; " + euler_minding,
                     "pinned_identity", "term ratio r_k = k/(2k+1)"),
            "positive terms, geometric-rate convergence",
        ),
        SeriesSeed(
            "ln2_half_powers",
            1,
            Fraction(1, 2),
            _rat((-1, 1), (0, 2)),
            "log(2)",
            lambda: mp.log(2),
            Citation("Mercator; Euler", "1668/1748",
                     "log 2 = sum_{k>=1} 1/(k 2^k) (the series for -log(1-z) at z = 1/2); "
                     + euler_minding,
                     "pinned_identity", "term ratio r_k = (k-1)/(2k)"),
            "z = 1/2 inside the disc of convergence",
        ),
        SeriesSeed(
            "exp_one",
            0,
            Fraction(1),
            _rat((0, 1), (1,)),
            "e",
            lambda: +mp.e,
            Citation("Euler", "1748", "e = sum_{k>=0} 1/k!; " + euler_minding,
                     "pinned_identity", "term ratio r_k = 1/k"),
            "entire",
        ),
        SeriesSeed(
            "exp_minus_one",
            0,
            Fraction(1),
            _rat((0, -1), (1,)),
            "1/e",
            lambda: 1 / mp.e,
            Citation("Euler", "1748", "1/e = sum_{k>=0} (-1)^k/k!; " + euler_minding,
                     "pinned_identity", "term ratio r_k = -1/k"),
            "entire",
        ),
        SeriesSeed(
            "exp_half",
            0,
            Fraction(1),
            _rat((0, 1), (0, 2)),
            "exp(1/2)",
            lambda: mp.exp(mp.mpf(1) / 2),
            Citation("Euler", "1748", "exp(1/2) = sum_{k>=0} 2^-k/k!; " + euler_minding,
                     "pinned_identity", "term ratio r_k = 1/(2k)"),
            "entire",
        ),
        SeriesSeed(
            "basel_zeta2",
            1,
            Fraction(1),
            _rat((1, -2, 1), (0, 0, 1)),
            "zeta(2) = pi^2/6",
            lambda: mp.zeta(2),
            Citation("Euler", "1735",
                     "Basel problem sum_{k>=1} 1/k^2 = pi^2/6; " + euler_minding,
                     "pinned_identity", "term ratio r_k = (k-1)^2/k^2"),
            "positive terms",
        ),
        SeriesSeed(
            "eta2",
            1,
            Fraction(1),
            _rat((-1, 2, -1), (0, 0, 1)),
            "pi^2/12",
            lambda: mp.pi**2 / 12,
            Citation("Euler", "1735", "eta(2) = sum (-1)^{k-1}/k^2 = pi^2/12; " + euler_minding,
                     "pinned_identity", "term ratio r_k = -(k-1)^2/k^2"),
            "alternating",
        ),
        SeriesSeed(
            "catalan_series",
            0,
            Fraction(1),
            _rat((-1, 4, -4), (1, 4, 4)),
            "Catalan",
            lambda: +mp.catalan,
            Citation("Catalan", "1865", "G = sum (-1)^k/(2k+1)^2; " + euler_minding,
                     "pinned_identity", "term ratio r_k = -(2k-1)^2/(2k+1)^2"),
            "alternating",
        ),
        SeriesSeed(
            "zeta3_series",
            1,
            Fraction(1),
            _rat((-1, 3, -3, 1), (0, 0, 0, 1)),
            "zeta(3)",
            lambda: mp.zeta(3),
            Citation("Euler", "1748", "zeta(3) = sum_{k>=1} 1/k^3; " + euler_minding,
                     "pinned_identity", "term ratio r_k = (k-1)^3/k^3"),
            "positive terms",
        ),
        SeriesSeed(
            "eta3",
            1,
            Fraction(1),
            _rat((1, -3, 3, -1), (0, 0, 0, 1)),
            "3*zeta(3)/4",
            lambda: 3 * mp.zeta(3) / 4,
            Citation("Euler", "1748", "eta(3) = sum (-1)^{k-1}/k^3 = 3 zeta(3)/4; " + euler_minding,
                     "pinned_identity", "term ratio r_k = -(k-1)^3/k^3"),
            "alternating",
        ),
        SeriesSeed(
            "arctan_half",
            0,
            Fraction(1, 2),
            _rat((Fraction(1, 4), Fraction(-1, 2)), (1, 2)),
            "arctan(1/2)",
            lambda: mp.atan(mp.mpf(1) / 2),
            Citation("Gregory", "1671", "arctan z = sum (-1)^k z^{2k+1}/(2k+1) at z = 1/2; "
                     + euler_minding,
                     "pinned_identity", "term ratio r_k = -z^2 (2k-1)/(2k+1)"),
            "|z| <= 1",
        ),
        SeriesSeed(
            "arctan_third",
            0,
            Fraction(1, 3),
            _rat((Fraction(1, 9), Fraction(-2, 9)), (1, 2)),
            "arctan(1/3)",
            lambda: mp.atan(mp.mpf(1) / 3),
            Citation("Gregory", "1671", "arctan z = sum (-1)^k z^{2k+1}/(2k+1) at z = 1/3; "
                     + euler_minding,
                     "pinned_identity", "term ratio r_k = -z^2 (2k-1)/(2k+1)"),
            "|z| <= 1",
        ),
        SeriesSeed(
            "log_three_halves",
            1,
            Fraction(1, 3),
            _rat((Fraction(-1, 3), Fraction(1, 3)), (0, 1)),
            "log(3/2)",
            lambda: mp.log(mp.mpf(3) / 2),
            Citation("Mercator", "1668", "-log(1-z) = sum_{k>=1} z^k/k at z = 1/3; " + euler_minding,
                     "pinned_identity", "term ratio r_k = z(k-1)/k"),
            "|z| < 1",
        ),
        SeriesSeed(
            "log_four_thirds",
            1,
            Fraction(1, 4),
            _rat((Fraction(-1, 4), Fraction(1, 4)), (0, 1)),
            "log(4/3)",
            lambda: mp.log(mp.mpf(4) / 3),
            Citation("Mercator", "1668", "-log(1-z) = sum_{k>=1} z^k/k at z = 1/4; " + euler_minding,
                     "pinned_identity", "term ratio r_k = z(k-1)/k"),
            "|z| < 1",
        ),
    ]


def _seeds_gauss_hypergeometric() -> list[Seed]:
    """Gauss's continued fraction for ratios of contiguous ``2F1`` functions."""

    citation = Citation(
        author="Gauss",
        year="1813",
        reference="Disquisitiones generales circa seriem infinitam (Werke III); DLMF 15.7",
        confidence="pinned_identity",
        note=(
            "F(a,b+1;c+1;z)/F(a,b;c;z) = 1/(1 - u_1 z/(1 - u_2 z/(1 - ...))) with "
            "u_{2k+1} = (a+k)(c-b+k)/((c+2k)(c+2k+1)) and "
            "u_{2k} = (b+k)(c-a+k)/((c+2k-1)(c+2k))"
        ),
    )
    grid = [
        (Fraction(1, 2), Fraction(1, 2), Fraction(3, 2)),
        (Fraction(1), Fraction(1, 2), Fraction(3, 2)),
        (Fraction(1), Fraction(1), Fraction(2)),
        (Fraction(1, 2), Fraction(1), Fraction(2)),
        (Fraction(3, 2), Fraction(1, 2), Fraction(5, 2)),
        (Fraction(1), Fraction(3, 2), Fraction(5, 2)),
        (Fraction(2), Fraction(1), Fraction(3)),
        (Fraction(1, 2), Fraction(3, 2), Fraction(2)),
    ]
    zs = [Fraction(1, 2), Fraction(-1), Fraction(1, 3), Fraction(-1, 2)]
    seeds: list[Seed] = []
    for a, b, c in grid:
        for z in zs:
            even = Rat.of(
                Poly.linear(Fraction(1, 2), a - 1) * Poly.linear(Fraction(1, 2), c - b - 1),
                Poly.linear(1, c - 2) * Poly.linear(1, c - 1),
            )
            odd = Rat.of(
                Poly.linear(Fraction(1, 2), b - Fraction(1, 2))
                * Poly.linear(Fraction(1, 2), c - a - Fraction(1, 2)),
                Poly.linear(1, c - 2) * Poly.linear(1, c - 1),
            )
            b_spec = SeqSpec.build(
                2,
                (Rat.constant(-z) * even, Rat.constant(-z) * odd),
                {1: Fraction(1)},
            )
            label = f"a{a}_b{b}_c{c}_z{z}".replace("/", "over").replace("-", "m")
            seeds.append(
                Seed(
                    f"gauss_2f1_{label}",
                    "gauss_hypergeometric",
                    CFPattern(seq_from_poly(Poly.constant(1), {0: Fraction(0)}), b_spec),
                    f"2F1({a},{b}+1;{c}+1;{z})/2F1({a},{b};{c};{z})",
                    lambda a=a, b=b, c=c, z=z: (
                        mp.hyp2f1(to_mpf(a), to_mpf(b + 1), to_mpf(c + 1), to_mpf(z))
                        / mp.hyp2f1(to_mpf(a), to_mpf(b), to_mpf(c), to_mpf(z))
                    ),
                    citation,
                    f"a={a}, b={b}, c={c}, z={z}",
                    check_depth=600,
                    check_digits=20,
                )
            )
    return seeds


def _seeds_rogers_ramanujan() -> list[Seed]:
    """Rogers-Ramanujan continued fraction: declared *outside* the rational-in-n grammar."""

    citation = Citation(
        author="Rogers; Ramanujan",
        year="1894/1913",
        reference="Rogers-Ramanujan continued fraction R(q) = q^(1/5)/(1 + q/(1 + q^2/(1 + ...)))",
        confidence="pinned_identity",
        note=(
            "partial numerators are q^n, a geometric sequence, so this identity is outside "
            "the rational-in-n grammar and can only be matched by value"
        ),
    )

    def rr(q_expr: Callable[[], mp.mpf]) -> Callable[[], mp.mpf]:
        def value() -> mp.mpf:
            q = q_expr()
            depth = 400
            x = mp.mpf(1)
            for n in range(depth, 0, -1):
                x = 1 + q**n / x
            return q ** (mp.mpf(1) / 5) / x

        return value

    seeds: list[Seed] = []
    for label, q_expr in (
        ("exp_minus_2pi", lambda: mp.exp(-2 * mp.pi)),
        ("exp_minus_pi", lambda: mp.exp(-mp.pi)),
        ("exp_minus_4pi", lambda: mp.exp(-4 * mp.pi)),
    ):
        seeds.append(
            Seed(
                f"rogers_ramanujan_{label}",
                "rogers_ramanujan",
                None,
                f"R(q) at q = {label.replace('_', ' ')}",
                rr(q_expr),
                citation,
                "|q| < 1",
                check_mode="out_of_grammar",
                grammar="outside_declared_grammar",
            )
        )
    return seeds


def build_seeds() -> list[Seed]:
    """The full seed catalogue, deterministic and sorted by seed id."""

    seeds: list[Seed] = []
    seeds.extend(_seeds_periodic_surd())
    seeds.extend(_seeds_simple_sqrt())
    seeds.extend(_seeds_euler_e())
    seeds.extend(_seeds_lambert())
    seeds.extend(_seeds_arctan())
    seeds.extend(_seeds_gauss_log())
    seeds.extend(_seeds_erfc())
    seeds.extend(_seeds_bessel())
    seeds.extend(_seeds_cotangent_family())
    seeds.extend(_seeds_apery())
    seeds.extend(_seeds_gauss_hypergeometric())
    seeds.extend(_seeds_rogers_ramanujan())
    for series in _series_seeds():
        pattern = transform_euler_minding(series.first_term, series.ratio, series.first_index)
        seeds.append(
            Seed(
                f"euler_minding_{series.series_id}",
                "euler_minding_series",
                pattern,
                series.value_expr,
                series.value_fn,
                series.citation,
                series.validity_domain,
                check_mode="series_correspondence",
            )
        )
    seeds.sort(key=lambda seed: seed.seed_id)
    identifiers = [seed.seed_id for seed in seeds]
    if len(set(identifiers)) != len(identifiers):
        raise CorpusError("duplicate seed identifier in the catalogue")
    return seeds


# ---------------------------------------------------------------------------
# Declared expansion sets
# ---------------------------------------------------------------------------

#: Constant equivalence sequences ``c_n = c``.
EQUIVALENCE_CONSTANTS: tuple[Fraction, ...] = (
    Fraction(-1),
    Fraction(2),
    Fraction(-2),
    Fraction(3),
    Fraction(-3),
    Fraction(4),
    Fraction(-4),
    Fraction(1, 2),
    Fraction(-1, 2),
    Fraction(1, 3),
    Fraction(-1, 3),
    Fraction(1, 4),
    Fraction(-1, 4),
)

#: Linear equivalence sequences ``c_n = p n + q`` with ``p`` and ``q`` of the same sign, so
#: that ``c_n`` never vanishes on ``n >= 0``.
EQUIVALENCE_LINEAR: tuple[tuple[int, int], ...] = tuple(
    (p, q if p > 0 else -q) for p in (1, 2, 3, -1, -2, -3) for q in (1, 2, 3)
)


def declared_equivalence_sequences() -> tuple[tuple[str, SeqSpec], ...]:
    """The declared finite set of equivalence sequences, deterministic and labelled."""

    specs: list[tuple[str, SeqSpec]] = []
    for constant in EQUIVALENCE_CONSTANTS:
        specs.append((f"c_n={constant}", seq_constant(constant)))
    for p, q in EQUIVALENCE_LINEAR:
        specs.append((f"c_n={p}n{q:+d}", seq_from_poly(Poly.linear(p, q))))
    return tuple(specs)


def declared_mobius_set() -> tuple[Mobius, ...]:
    """Canonical invertible Moebius maps with coefficients in ``{-1, 0, 1}``, identity last."""

    classes: set[tuple[Fraction, Fraction, Fraction, Fraction]] = set()
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            for r in (-1, 0, 1):
                for s in (-1, 0, 1):
                    if p * s - q * r == 0:
                        continue
                    classes.add(mobius_normalize(mobius_of(p, q, r, s)))
    ordered = sorted(classes, key=lambda item: tuple(str(value) for value in item))
    return tuple(item for item in ordered if item != IDENTITY_MOBIUS)


DECLARED_EXPANSION = {
    "equivalence_sequence_count": len(EQUIVALENCE_CONSTANTS) + len(EQUIVALENCE_LINEAR),
    "mobius_map_count": len(declared_mobius_set()),
    "tail_shift_levels": [1, 2, 3],
    "contractions": ["contract_even", "contract_odd"],
    "second_level": "constant equivalences applied to every tail_shift(1) record",
    "extension": "unit-denominator inverse applied to every contract_even record",
}

CORPUS_CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "every_record_resolves_to_a_cited_seed": True,
    "seeds_are_numerically_certified_at_build_time": True,
    "external_fetch_performed": False,
    "value_equality_alone_is_not_membership": True,
}


# ---------------------------------------------------------------------------
# Building the corpus
# ---------------------------------------------------------------------------


def _render_value(value: mp.mpf) -> str:
    if not mp.isfinite(value):
        return "nan"
    return mp.nstr(value, VALUE_STORE_DPS, strip_zeros=False)


def _normal_fields(pattern: CFPattern | None, cf_value: mp.mpf) -> tuple[str | None, str | None]:
    if pattern is None:
        return None, None
    try:
        form = normal_form(pattern)
    except CorpusError:
        return None, None
    return form.key(), _render_value(to_mpf(form.scale) * cf_value)


def certify_seed(seed: Seed) -> dict[str, Any]:
    """Numerically certify a seed's stored closed form against its continued fraction."""

    with mp.workdps(SEED_CHECK_DPS):
        target = seed.value_fn()
        if seed.check_mode == "out_of_grammar":
            return {"mode": seed.check_mode, "digits_verified": None}
        assert seed.pattern is not None
        if seed.check_mode == "series_correspondence":
            return {"mode": seed.check_mode, "digits_verified": None}
        approximation = seed.pattern.evaluate(seed.check_depth)
        error = abs(approximation - target)
        scale = max(mp.mpf(1), abs(target))
        if not mp.isfinite(error) or error / scale >= mp.mpf(10) ** (-seed.check_digits):
            raise CorpusError(
                f"seed {seed.seed_id} failed certification: relative error "
                f"{mp.nstr(error / scale, 5)} at depth {seed.check_depth}"
            )
        return {"mode": seed.check_mode, "digits_verified": seed.check_digits}


def certify_series_correspondence(series: SeriesSeed, terms: int = 20) -> None:
    """Exact check that the Euler-Minding convergents equal the series partial sums."""

    pattern = transform_euler_minding(series.first_term, series.ratio, series.first_index)
    numerators, denominators = _convergents(pattern, terms + 1)
    partial = Fraction(0)
    term = series.first_term
    for index in range(terms):
        partial += term
        convergent = numerators[index + 2] / denominators[index + 2]
        if convergent != partial:
            raise CorpusError(
                f"Euler-Minding correspondence failed for {series.series_id} at term {index}"
            )
        term = term * series.ratio.evaluate(series.first_index + index + 1)


def _seed_record(seed: Seed, cf_value: mp.mpf) -> CFRecord:
    normal_key, normalized = _normal_fields(seed.pattern, cf_value)
    rendered = _render_value(cf_value)
    return CFRecord(
        record_id=f"seed:{seed.seed_id}",
        kind="seed",
        family=seed.family,
        seed_id=seed.seed_id,
        parent_id=None,
        depth=0,
        transform=(),
        pattern=seed.pattern,
        wrap=IDENTITY_MOBIUS,
        cf_value=rendered,
        value=rendered,
        value_expr=seed.value_expr,
        citation=seed.citation,
        validity_domain=seed.validity_domain,
        grammar=seed.grammar,
        normal_form_key=normal_key,
        normalized_value=normalized,
    )


def _derived_record(
    parent: CFRecord,
    *,
    suffix: str,
    transformation: str,
    detail: str,
    pattern: CFPattern | None,
    wrap: Mobius,
    step: Mobius,
    parent_cf_value: mp.mpf,
) -> tuple[CFRecord, mp.mpf]:
    cf_value = mobius_apply(step, parent_cf_value)
    if not mp.isfinite(cf_value):
        raise OutOfDeclaredClass("transformed value is not finite")
    value = mobius_apply(wrap, cf_value)
    if not mp.isfinite(value):
        raise OutOfDeclaredClass("wrapped value is not finite")
    normal_key, normalized = _normal_fields(pattern, cf_value)
    total = mobius_compose(step, parent.cf_mobius_from_seed)
    combined = mobius_compose(wrap, total)
    expression = (
        parent.value_expr
        if combined == IDENTITY_MOBIUS
        else f"{mobius_render(combined)} at x = the seed value {parent.value_expr}"
    )
    record = CFRecord(
        record_id=f"{parent.record_id}|{suffix}",
        kind="derived",
        family=parent.family,
        seed_id=parent.seed_id,
        parent_id=parent.record_id,
        depth=parent.depth + 1,
        transform=(("transformation", transformation), ("detail", detail)),
        pattern=pattern,
        wrap=wrap,
        cf_value=_render_value(cf_value),
        value=_render_value(value),
        value_expr=expression,
        citation=parent.citation,
        validity_domain=parent.validity_domain,
        grammar=parent.grammar,
        normal_form_key=normal_key,
        normalized_value=normalized,
        cf_mobius_from_seed=total,
    )
    return record, cf_value


def build_corpus(*, verbose: bool = False) -> tuple[list[CFRecord], dict[str, Any]]:
    """Build the full corpus deterministically and report how its size was reached."""

    seeds = build_seeds()
    series_index = {f"euler_minding_{item.series_id}": item for item in _series_seeds()}
    certifications: list[dict[str, Any]] = []
    records: list[CFRecord] = []
    values: dict[str, mp.mpf] = {}
    with mp.workdps(SEED_CHECK_DPS + 20):
        for seed in seeds:
            certification = certify_seed(seed)
            if seed.check_mode == "series_correspondence":
                certify_series_correspondence(series_index[seed.seed_id])
            certifications.append({"seed_id": seed.seed_id, **certification})
            cf_value = seed.value_fn()
            record = _seed_record(seed, cf_value)
            records.append(record)
            values[record.record_id] = cf_value

        seed_records = list(records)
        equivalences = declared_equivalence_sequences()
        mobius_maps = declared_mobius_set()
        constants = equivalences[: len(EQUIVALENCE_CONSTANTS)]
        counts = {name: 0 for name in DECLARED_TRANSFORMATIONS}
        level_two_parents: list[CFRecord] = []
        contraction_parents: list[CFRecord] = []

        def emit(record: CFRecord, cf_value: mp.mpf) -> None:
            records.append(record)
            values[record.record_id] = cf_value
            counts[dict(record.transform)["transformation"]] += 1

        for parent in seed_records:
            parent_value = values[parent.record_id]
            if parent.pattern is not None:
                for label, spec in equivalences:
                    try:
                        pattern, step = transform_equivalence(parent.pattern, spec)
                        child, child_value = _derived_record(
                            parent,
                            suffix=f"equiv({label})",
                            transformation="equivalence",
                            detail=label,
                            pattern=pattern,
                            wrap=IDENTITY_MOBIUS,
                            step=step,
                            parent_cf_value=parent_value,
                        )
                    except CorpusError:
                        continue
                    emit(child, child_value)
                for levels in DECLARED_EXPANSION["tail_shift_levels"]:
                    try:
                        pattern, step = transform_tail_shift(parent.pattern, levels)
                        child, child_value = _derived_record(
                            parent,
                            suffix=f"tail_shift({levels})",
                            transformation="tail_shift",
                            detail=f"levels={levels}",
                            pattern=pattern,
                            wrap=IDENTITY_MOBIUS,
                            step=step,
                            parent_cf_value=parent_value,
                        )
                    except CorpusError:
                        continue
                    emit(child, child_value)
                    if levels == 1:
                        level_two_parents.append(child)
                for parity in ("even", "odd"):
                    try:
                        pattern = transform_contraction(parent.pattern, parity)
                        child, child_value = _derived_record(
                            parent,
                            suffix=f"contract_{parity}",
                            transformation=f"contract_{parity}",
                            detail=f"parity={parity}",
                            pattern=pattern,
                            wrap=IDENTITY_MOBIUS,
                            step=IDENTITY_MOBIUS,
                            parent_cf_value=parent_value,
                        )
                    except CorpusError:
                        continue
                    emit(child, child_value)
                    if parity == "even":
                        contraction_parents.append(child)
            for wrap in mobius_maps:
                try:
                    child, child_value = _derived_record(
                        parent,
                        suffix=f"mobius({','.join(str(item) for item in wrap)})",
                        transformation="mobius_post_composition",
                        detail=mobius_render(wrap),
                        pattern=parent.pattern,
                        wrap=wrap,
                        step=IDENTITY_MOBIUS,
                        parent_cf_value=parent_value,
                    )
                except CorpusError:
                    continue
                emit(child, child_value)

        for parent in level_two_parents:
            parent_value = values[parent.record_id]
            assert parent.pattern is not None
            for label, spec in constants:
                try:
                    pattern, step = transform_equivalence(parent.pattern, spec)
                    child, child_value = _derived_record(
                        parent,
                        suffix=f"equiv({label})",
                        transformation="equivalence",
                        detail=label,
                        pattern=pattern,
                        wrap=IDENTITY_MOBIUS,
                        step=step,
                        parent_cf_value=parent_value,
                    )
                except CorpusError:
                    continue
                emit(child, child_value)

        for parent in contraction_parents:
            parent_value = values[parent.record_id]
            assert parent.pattern is not None
            try:
                pattern = transform_extension(parent.pattern)
                child, child_value = _derived_record(
                    parent,
                    suffix="extend",
                    transformation="extend",
                    detail="unit-denominator inverse of the even contraction",
                    pattern=pattern,
                    wrap=IDENTITY_MOBIUS,
                    step=IDENTITY_MOBIUS,
                    parent_cf_value=parent_value,
                )
            except CorpusError:
                continue
            emit(child, child_value)

    def dedup_key(record: CFRecord) -> tuple[str, str, str]:
        return (
            record.pattern_key(),
            "/".join(str(item) for item in record.wrap),
            record.value,
        )

    # Seeds are never deduplicated away: two sources stating the same identity are two
    # citations, and prior art is better served by keeping both.  Derived records that
    # merely restate a seed, or each other, collapse.
    seeds_kept = [record for record in records if record.kind == "seed"]
    deduplicated: dict[tuple[str, str, str], CFRecord] = {
        dedup_key(record): record for record in seeds_kept
    }
    for record in sorted(records, key=lambda item: (item.depth, item.record_id)):
        if record.kind == "seed":
            continue
        key = dedup_key(record)
        if key not in deduplicated:
            deduplicated[key] = record
    kept = sorted({record.record_id: record for record in
                   [*seeds_kept, *deduplicated.values()]}.values(),
                  key=lambda item: item.record_id)
    kept_ids = {record.record_id for record in kept}
    final = [record for record in kept if record.parent_id is None or record.parent_id in kept_ids]
    while True:
        final_ids = {record.record_id for record in final}
        pruned = [
            record
            for record in final
            if record.parent_id is None or record.parent_id in final_ids
        ]
        if len(pruned) == len(final):
            break
        final = pruned
    report = {
        "seed_count": len(seeds),
        "generated_before_dedup": len(records),
        "records": len(final),
        "transformation_counts": counts,
        "certifications": certifications,
        "verbose": verbose,
    }
    return final, report


# ---------------------------------------------------------------------------
# Corpus container and lookup indices
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """Loaded corpus with the three exact lookup indices the screen needs."""

    records: tuple[CFRecord, ...]
    manifest: dict[str, Any]
    by_id: dict[str, CFRecord] = field(init=False)
    pattern_index: dict[str, list[str]] = field(init=False)
    normal_index: dict[str, list[str]] = field(init=False)
    normal_key_index: dict[str, list[str]] = field(init=False)
    numerator_index: dict[str, list[str]] = field(init=False)
    _value_table: list[tuple[float, str]] = field(init=False)
    _reported_table: list[tuple[float, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {record.record_id: record for record in self.records}
        self.pattern_index = {}
        self.normal_index = {}
        self.normal_key_index = {}
        self.numerator_index = {}
        table: list[tuple[float, str]] = []
        reported: list[tuple[float, str]] = []
        for record in self.records:
            self.pattern_index.setdefault(record.pattern_key(), []).append(record.record_id)
            if record.normal_form_key is not None:
                key = f"{record.normal_form_key}#{record.normalized_value}"
                self.normal_index.setdefault(key, []).append(record.record_id)
                self.normal_key_index.setdefault(record.normal_form_key, []).append(
                    record.record_id
                )
            if record.pattern is not None:
                b_key = drop_index_zero(record.pattern.b).key()
                self.numerator_index.setdefault(b_key, []).append(record.record_id)
            for text, sink in ((record.cf_value, table), (record.value, reported)):
                try:
                    sink.append((float(mp.mpf(text)), record.record_id))
                except (ValueError, TypeError):
                    continue
        table.sort()
        reported.sort()
        self._value_table = table
        self._reported_table = reported

    def lookup_pattern(self, pattern: CFPattern) -> list[CFRecord]:
        return [self.by_id[item] for item in self.pattern_index.get(pattern.key(), ())]

    def lookup_normal_form(self, form: NormalForm, cf_value: mp.mpf) -> list[CFRecord]:
        key = f"{form.key()}#{_render_value(to_mpf(form.scale) * cf_value)}"
        return [self.by_id[item] for item in self.normal_index.get(key, ())]

    def _lookup_table(
        self,
        table: Sequence[tuple[float, str]],
        value: mp.mpf,
        field_name: str,
        digits: int,
    ) -> list[CFRecord]:
        if not mp.isfinite(value):
            return []
        probe = float(value)
        window = 1e-9 * max(1.0, abs(probe))
        keys = [item[0] for item in table]
        low = bisect.bisect_left(keys, probe - window)
        high = bisect.bisect_right(keys, probe + window)
        tolerance = mp.mpf(10) ** (-digits)
        scale = max(mp.mpf(1), abs(value))
        found: list[CFRecord] = []
        for _, record_id in table[low:high]:
            record = self.by_id[record_id]
            stored = mp.mpf(getattr(record, field_name))
            if abs(stored - value) / scale < tolerance:
                found.append(record)
        return sorted(found, key=lambda item: item.record_id)

    def lookup_value(self, value: mp.mpf, *, digits: int = VALUE_MATCH_DIGITS) -> list[CFRecord]:
        """Records whose *continued fraction* converges to ``value`` to ``digits`` digits."""

        return self._lookup_table(self._value_table, value, "cf_value", digits)

    def lookup_reported_value(
        self, value: mp.mpf, *, digits: int = VALUE_MATCH_DIGITS
    ) -> list[CFRecord]:
        """Records whose *reported* value (the identity's right-hand side) equals ``value``."""

        return self._lookup_table(self._reported_table, value, "value", digits)

    def family_relatives(self, pattern: CFPattern, *, limit: int = 4) -> list[CFRecord]:
        """Records sharing the candidate's partial numerators exactly.

        These are the honest near misses: same ``b_n``, different ``a_n``.  When a candidate
        is refused, naming its closest relatives says what kind of continued fraction it is
        without pretending the corpus contains it.
        """

        key = drop_index_zero(pattern.b).key()
        found = [self.by_id[item] for item in self.numerator_index.get(key, ())]
        return sorted(found, key=lambda item: (item.depth, item.record_id))[:limit]

    def adjacent_family_members(
        self, pattern: CFPattern, *, offsets: Sequence[int] = (-4, -3, -2, -1, 1, 2, 3, 4)
    ) -> list[tuple[int, CFRecord]]:
        """Corpus members of the same one-parameter family, shifted in ``a_n``'s constant.

        Many enumerated continued fractions sit in a family ``a_n = alpha1 n + j`` whose
        low-``j`` members are classical.  Replacing ``j`` by ``j + delta`` and asking whether
        *that* pattern's equivalence class is in the corpus answers a precise question --
        "is a neighbouring member of this exact family known?" -- without asserting anything
        about the candidate itself.  It is a diagnostic, never a membership test.
        """

        found: list[tuple[int, CFRecord]] = []
        for delta in offsets:
            shifted_terms = tuple(
                term + Rat.constant(delta) for term in pattern.a.terms
            )
            overrides = {
                index: value + Fraction(delta) for index, value in pattern.a.overrides
            }
            try:
                moved = CFPattern(
                    SeqSpec.build(pattern.a.period, shifted_terms, overrides), pattern.b
                )
                key = normal_form(moved).key()
            except CorpusError:
                continue
            for record_id in self.normal_key_index.get(key, ())[:1]:
                found.append((delta, self.by_id[record_id]))
        return sorted(found, key=lambda item: (abs(item[0]), item[0]))


# ---------------------------------------------------------------------------
# Sealed artifact: sqlite database plus canonical manifest
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE corpus_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE citations (
    citation_id TEXT PRIMARY KEY,
    author TEXT NOT NULL, year TEXT NOT NULL, reference TEXT NOT NULL,
    confidence TEXT NOT NULL, note TEXT NOT NULL
);
CREATE TABLE records (
    record_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL, family TEXT NOT NULL, seed_id TEXT NOT NULL,
    parent_id TEXT, depth INTEGER NOT NULL,
    transformation TEXT, transform_detail TEXT, grammar TEXT NOT NULL,
    a_pattern TEXT, b_pattern TEXT, pattern_key TEXT NOT NULL,
    normal_form_key TEXT, normalized_value TEXT,
    wrap TEXT NOT NULL, cf_value TEXT NOT NULL, value TEXT NOT NULL,
    value_expr TEXT NOT NULL, citation_id TEXT NOT NULL, validity_domain TEXT NOT NULL,
    cf_mobius_from_seed TEXT NOT NULL
);
CREATE INDEX records_pattern ON records (pattern_key);
CREATE INDEX records_normal ON records (normal_form_key);
CREATE INDEX records_value ON records (cf_value);
CREATE INDEX records_seed ON records (seed_id);
"""


def corpus_manifest(
    records: Sequence[CFRecord], report: Mapping[str, Any], sqlite_sha256: str | None = None
) -> dict[str, Any]:
    """Canonical manifest: counts, declared sets, claims, and the record-stream hash."""

    by_family: dict[str, int] = {}
    by_transformation: dict[str, int] = {}
    by_confidence: dict[str, int] = {}
    seed_families: dict[str, int] = {}
    for record in records:
        by_family[record.family] = by_family.get(record.family, 0) + 1
        name = dict(record.transform).get("transformation", "seed")
        by_transformation[name] = by_transformation.get(name, 0) + 1
        by_confidence[record.citation.confidence] = (
            by_confidence.get(record.citation.confidence, 0) + 1
        )
        if record.kind == "seed":
            seed_families[record.family] = seed_families.get(record.family, 0) + 1
    stream = [record.as_json() for record in records]
    closure = verify_forest_closure(list(records))
    body = {
        "schema_version": CORPUS_SCHEMA,
        "claims": CORPUS_CLAIMS,
        "counts": {
            "records": len(records),
            "seeds": sum(1 for record in records if record.kind == "seed"),
            "derived": sum(1 for record in records if record.kind != "seed"),
            "out_of_declared_grammar": sum(
                1 for record in records if record.grammar != "declared_quasi_rational"
            ),
            "records_by_family": dict(sorted(by_family.items())),
            "records_by_transformation": dict(sorted(by_transformation.items())),
            "records_by_citation_confidence": dict(sorted(by_confidence.items())),
            "seeds_by_family": dict(sorted(seed_families.items())),
            "generated_before_dedup": report["generated_before_dedup"],
        },
        "forest": closure,
        "declared_transformations": list(DECLARED_TRANSFORMATIONS),
        "declared_expansion": DECLARED_EXPANSION,
        "declared_equivalence_sequences": [
            label for label, _ in declared_equivalence_sequences()
        ],
        "declared_mobius_set": [
            "/".join(str(item) for item in wrap) for wrap in declared_mobius_set()
        ],
        "euler_minding_seed_records": sum(
            1 for record in records if record.family == "euler_minding_series" and record.kind == "seed"
        ),
        "seed_certification": {
            "working_dps": SEED_CHECK_DPS,
            "modes": dict(
                sorted(
                    {
                        entry["mode"]: sum(
                            1 for item in report["certifications"] if item["mode"] == entry["mode"]
                        )
                        for entry in report["certifications"]
                    }.items()
                )
            ),
        },
        "value_match_digits": VALUE_MATCH_DIGITS,
        "records_sha256": canonical_sha256(stream),
        "scope": (
            "Independently encoded classical continued-fraction identities plus their exact "
            "declared transformation orbits. Every record resolves to a cited seed. The "
            "corpus is finite: absence from it is absence from this corpus, never novelty."
        ),
    }
    body["sqlite_sha256"] = sqlite_sha256
    body["content_sha256"] = canonical_sha256(body)
    return body


def write_corpus(
    records: Sequence[CFRecord], manifest: Mapping[str, Any], database: str | Path
) -> str:
    """Write the sealed sqlite artifact; returns its sha256."""

    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_SCHEMA)
        citations: dict[str, Citation] = {}
        for record in records:
            citations[canonical_sha256(record.citation.as_json())[:16]] = record.citation
        connection.executemany(
            "INSERT INTO citations VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    citation_id,
                    citation.author,
                    citation.year,
                    citation.reference,
                    citation.confidence,
                    citation.note,
                )
                for citation_id, citation in sorted(citations.items())
            ],
        )
        connection.executemany(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    record.record_id,
                    record.kind,
                    record.family,
                    record.seed_id,
                    record.parent_id,
                    record.depth,
                    dict(record.transform).get("transformation"),
                    dict(record.transform).get("detail"),
                    record.grammar,
                    None if record.pattern is None else record.pattern.a.key(),
                    None
                    if record.pattern is None
                    else drop_index_zero(record.pattern.b).key(),
                    record.pattern_key(),
                    record.normal_form_key,
                    record.normalized_value,
                    "/".join(str(item) for item in record.wrap),
                    record.cf_value,
                    record.value,
                    record.value_expr,
                    canonical_sha256(record.citation.as_json())[:16],
                    record.validity_domain,
                    "/".join(str(item) for item in record.cf_mobius_from_seed),
                )
                for record in records
            ],
        )
        connection.executemany(
            "INSERT INTO corpus_meta VALUES (?, ?)",
            sorted(
                {
                    "schema_version": CORPUS_SCHEMA,
                    "records_sha256": manifest["records_sha256"],
                    "record_count": str(len(records)),
                }.items()
            ),
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pattern_from_keys(a_key: str | None, b_key: str | None) -> CFPattern | None:
    if a_key is None or b_key is None:
        return None
    return CFPattern(parse_seq_key(a_key), parse_seq_key(b_key))


def load_corpus(database: str | Path, manifest_path: str | Path) -> Corpus:
    """Load a sealed corpus; the manifest's record count and stream hash are re-checked."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    connection = sqlite3.connect(Path(database))
    try:
        connection.row_factory = sqlite3.Row
        citations = {
            row["citation_id"]: Citation(
                row["author"], row["year"], row["reference"], row["confidence"], row["note"]
            )
            for row in connection.execute("SELECT * FROM citations")
        }
        records: list[CFRecord] = []
        for row in connection.execute("SELECT * FROM records ORDER BY record_id"):
            transform: tuple[tuple[str, str], ...] = ()
            if row["transformation"] is not None:
                transform = (
                    ("transformation", row["transformation"]),
                    ("detail", row["transform_detail"]),
                )
            records.append(
                CFRecord(
                    record_id=row["record_id"],
                    kind=row["kind"],
                    family=row["family"],
                    seed_id=row["seed_id"],
                    parent_id=row["parent_id"],
                    depth=row["depth"],
                    transform=transform,
                    pattern=_pattern_from_keys(row["a_pattern"], row["b_pattern"]),
                    wrap=tuple(Fraction(item) for item in row["wrap"].split("/")),  # type: ignore[arg-type]
                    cf_value=row["cf_value"],
                    value=row["value"],
                    value_expr=row["value_expr"],
                    citation=citations[row["citation_id"]],
                    validity_domain=row["validity_domain"],
                    grammar=row["grammar"],
                    normal_form_key=row["normal_form_key"],
                    normalized_value=row["normalized_value"],
                    cf_mobius_from_seed=tuple(  # type: ignore[arg-type]
                        Fraction(item) for item in row["cf_mobius_from_seed"].split("/")
                    ),
                )
            )
    finally:
        connection.close()
    if manifest["counts"]["records"] != len(records):
        raise CorpusError("manifest record count does not match the database")
    if canonical_sha256([record.as_json() for record in records]) != manifest["records_sha256"]:
        raise CorpusError("corpus record stream hash does not match the manifest")
    return Corpus(tuple(records), manifest)


# ---------------------------------------------------------------------------
# Key parsing (the sqlite round trip)
# ---------------------------------------------------------------------------


def _parse_poly(text: str) -> Poly:
    coefficients: dict[int, Fraction] = {}
    body = text.strip()
    if body == "0":
        return Poly(())
    for chunk in body.split(" + "):
        chunk = chunk.strip()
        if "*" in chunk:
            head, symbol = chunk.split("*", 1)
            power = 1 if symbol == "n" else int(symbol.split("^")[1])
            coefficients[power] = coefficients.get(power, Fraction(0)) + Fraction(head)
        else:
            coefficients[0] = coefficients.get(0, Fraction(0)) + Fraction(chunk)
    width = max(coefficients) + 1
    return Poly(_trim([coefficients.get(index, Fraction(0)) for index in range(width)]))


def _parse_rat(text: str) -> Rat:
    if not text.startswith("(") or ")/(" not in text or not text.endswith(")"):
        raise CorpusError(f"malformed rational-function key: {text!r}")
    head, tail = text[1:-1].split(")/(", 1)
    return Rat.of(_parse_poly(head), _parse_poly(tail))


def parse_seq_key(text: str) -> SeqSpec:
    """Inverse of :meth:`SeqSpec.key` -- used when reloading the sealed database."""

    if not text.startswith("p"):
        raise CorpusError(f"malformed sequence key: {text!r}")
    head, rest = text[1:].split("[", 1)
    period = int(head)
    body, tail = rest.rsplit("]{", 1)
    if not tail.endswith("}"):
        raise CorpusError(f"malformed sequence key: {text!r}")
    terms = tuple(_parse_rat(item) for item in body.split(";"))
    overrides: dict[int, Fraction] = {}
    inner = tail[:-1]
    if inner:
        for item in inner.split(","):
            index, value = item.split("=", 1)
            overrides[int(index)] = Fraction(value)
    return SeqSpec(period, terms, tuple(sorted(overrides.items())))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate the continued-fraction prior-art corpus."
    )
    parser.add_argument("--database", default="runs/math/prior-art/cf-corpus-v1.sqlite")
    parser.add_argument("--manifest", default="runs/math/prior-art/cf-corpus-v1-manifest.json")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        corpus = load_corpus(args.database, args.manifest)
        verify_forest_closure(list(corpus.records))
        print(
            json.dumps(
                {
                    "validated": True,
                    "records": len(corpus.records),
                    "seeds": corpus.manifest["counts"]["seeds"],
                    "content_sha256": corpus.manifest["content_sha256"],
                }
            )
        )
        return 0
    records, report = build_corpus()
    provisional = corpus_manifest(records, report)
    digest = write_corpus(records, provisional, args.database)
    manifest = corpus_manifest(records, report, digest)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(
        json.dumps(
            {
                "records": manifest["counts"]["records"],
                "seeds": manifest["counts"]["seeds"],
                "generated_before_dedup": manifest["counts"]["generated_before_dedup"],
                "database": args.database,
                "sqlite_sha256": digest,
                "content_sha256": manifest["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
