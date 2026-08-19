"""Exact rational bracketing of the closed-form constants that appear in certificates.

Certificates in this repository carry their constants as *closed forms* -- nested radicals,
``exp(-8)``, ``pi`` -- because that is the only representation an exact-arithmetic ledger can
accept.  Ranking candidates by such a constant needs one operation the closed form does not
directly provide: **deciding which of two of them is larger**, without ever touching a float.

This module provides it.  Every real number that appears in the global-H7 certificates is
enclosed in an interval whose two endpoints are :class:`fractions.Fraction` -- exact rationals.
The enclosure is built bottom-up through the expression tree, and every node rounds its own
interval *outward* onto a dyadic grid, so containment is preserved at every step while the
endpoints stay small.  Comparison is then a proof, not an estimate: when the bracket of ``a``
lies strictly below the bracket of ``b``, the two rationals that witness it are transcribed
into the receipt and can be re-checked by anyone with integer arithmetic alone.

Three outcomes, and only three:

``EQUAL``
    The two expressions are the *same* canonical object.  Nothing weaker is ever called equal:
    a bracket can never prove two distinct closed forms equal, and this module does not pretend
    otherwise.
``LESS`` / ``GREATER``
    Separated at some precision on the declared ladder, with the separating rationals attached.
``UNSEPARATED``
    The declared precision budget ran out with the brackets still overlapping.  This is a real
    answer -- it says the ordering is not established -- and callers must not turn it into one.

No :class:`float` is constructed anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Any

import sympy as sp

#: Declared precision ladder, in significant bits, used when a caller does not supply one.
DEFAULT_LADDER = (64, 128, 256, 512, 1024)

EQUAL = "EQUAL"
LESS = "LESS"
GREATER = "GREATER"
UNSEPARATED = "UNSEPARATED"

_ZERO = Fraction(0)
_ONE = Fraction(1)


class ExactBracketError(ValueError):
    """Raised when an expression falls outside the supported exact-bracketing grammar."""


@dataclass(frozen=True)
class Bracket:
    """A closed rational interval ``[lo, hi]`` that provably contains a real number."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.lo, Fraction) or not isinstance(self.hi, Fraction):
            raise ExactBracketError("bracket endpoints must be exact rationals")
        if self.lo > self.hi:
            raise ExactBracketError("bracket is empty")

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def contains(self, value: Fraction) -> bool:
        return self.lo <= value <= self.hi

    def is_positive(self) -> bool:
        return self.lo > _ZERO

    def as_strings(self) -> dict[str, str]:
        """Endpoints as exact ``numerator/denominator`` strings for receipt transcription."""

        return {"lo": str(self.lo), "hi": str(self.hi)}


def _exp2(power: int) -> Fraction:
    if power >= 0:
        return Fraction(1 << power, 1)
    return Fraction(1, 1 << -power)


def _magnitude_exponent(value: Fraction) -> int:
    """An integer ``e`` with ``|value| < 2**e``; used only to place the rounding grid."""

    if value == 0:
        return 0
    numerator = abs(value.numerator)
    return numerator.bit_length() - value.denominator.bit_length() + 1


def _round_out(bracket: Bracket, bits: int) -> Bracket:
    """Widen ``bracket`` outward onto a dyadic grid carrying about ``bits`` significant bits."""

    magnitude = max(abs(bracket.lo), abs(bracket.hi))
    if magnitude == 0:
        return bracket
    grid = _exp2(_magnitude_exponent(magnitude) - bits)
    low_units = bracket.lo / grid
    high_units = bracket.hi / grid
    lo = (low_units.numerator // low_units.denominator) * grid
    hi = (-((-high_units.numerator) // high_units.denominator)) * grid
    return Bracket(lo, hi)


def _add(left: Bracket, right: Bracket) -> Bracket:
    return Bracket(left.lo + right.lo, left.hi + right.hi)


def _mul(left: Bracket, right: Bracket) -> Bracket:
    corners = (
        left.lo * right.lo,
        left.lo * right.hi,
        left.hi * right.lo,
        left.hi * right.hi,
    )
    return Bracket(min(corners), max(corners))


def _reciprocal(value: Bracket) -> Bracket:
    if value.lo <= _ZERO <= value.hi:
        raise ExactBracketError("cannot invert a bracket that straddles zero")
    return Bracket(_ONE / value.hi, _ONE / value.lo)


def _integer_power(value: Bracket, exponent: int, bits: int) -> Bracket:
    if exponent < 0:
        return _reciprocal(_integer_power(value, -exponent, bits))
    result = Bracket(_ONE, _ONE)
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _round_out(_mul(result, base), bits)
        remaining >>= 1
        if remaining:
            base = _round_out(_mul(base, base), bits)
    return result


def _sqrt_rational(value: Fraction, bits: int) -> Bracket:
    """Exact rational bracket for ``sqrt(value)`` with ``value >= 0``."""

    if value < 0:
        raise ExactBracketError("cannot take the square root of a negative bracket")
    if value == 0:
        return Bracket(_ZERO, _ZERO)
    numerator = value.numerator
    denominator = value.denominator
    shift = bits + 2
    scaled = (numerator * denominator) << (2 * shift)
    root = isqrt(scaled)
    divisor = denominator << shift
    return Bracket(Fraction(root, divisor), Fraction(root + 1, divisor))


def _sqrt(value: Bracket, bits: int) -> Bracket:
    if value.lo < _ZERO:
        if value.hi < _ZERO:
            raise ExactBracketError("cannot take the square root of a negative bracket")
        low = Bracket(_ZERO, _ZERO)
    else:
        low = _sqrt_rational(value.lo, bits)
    high = _sqrt_rational(value.hi, bits)
    return Bracket(low.lo, high.hi)


def arctan_reciprocal_bracket(inverse: int, bits: int) -> Bracket:
    """Bracket ``arctan(1/inverse)`` for an integer ``inverse >= 2``.

    The series is alternating with strictly decreasing terms, so two consecutive partial sums
    bracket the limit exactly.  No tail estimate is needed beyond that.
    """

    if inverse < 2:
        raise ExactBracketError("arctan series requires an argument of at most 1/2")
    target = _exp2(-bits - 4)
    total = Fraction(0)
    term_index = 0
    while True:
        power = 2 * term_index + 1
        term = Fraction(1, power * inverse**power)
        total += term if term_index % 2 == 0 else -term
        next_power = 2 * term_index + 3
        next_term = Fraction(1, next_power * inverse**next_power)
        if next_term < target:
            if term_index % 2 == 0:
                return Bracket(total - next_term, total)
            return Bracket(total, total + next_term)
        term_index += 1


def pi_bracket(bits: int) -> Bracket:
    """Bracket ``pi`` from Machin's formula ``pi = 16*arctan(1/5) - 4*arctan(1/239)``."""

    five = arctan_reciprocal_bracket(5, bits + 8)
    two39 = arctan_reciprocal_bracket(239, bits + 8)
    return _round_out(
        Bracket(16 * five.lo - 4 * two39.hi, 16 * five.hi - 4 * two39.lo), bits
    )


def exp_bracket(argument: Fraction, bits: int) -> Bracket:
    """Bracket ``exp(argument)`` for any rational ``argument``, from the Taylor series."""

    if argument < 0:
        return _reciprocal(exp_bracket(-argument, bits + 8))
    if argument == 0:
        return Bracket(_ONE, _ONE)
    target = _exp2(-bits - 4)
    total = Fraction(1)
    term = Fraction(1)
    index = 0
    while True:
        index += 1
        term = term * argument / index
        total += term
        # Once ``argument < index + 1`` the tail is dominated by a geometric series with
        # ratio ``argument / (index + 1)``; that bound is exact rational arithmetic.
        if argument < index + 1:
            ratio = argument / (index + 1)
            tail = term * ratio / (1 - ratio)
            if tail < target * total:
                return _round_out(Bracket(total, total + tail), bits)


def log_bracket(argument: Fraction, bits: int) -> Bracket:
    """Bracket ``log(argument)`` for a positive rational, via ``2*atanh((x-1)/(x+1))``."""

    if argument <= 0:
        raise ExactBracketError("log requires a positive rational argument")
    if argument == 1:
        return Bracket(_ZERO, _ZERO)
    if argument < 1:
        inverted = log_bracket(1 / argument, bits + 8)
        return Bracket(-inverted.hi, -inverted.lo)
    # Halve the argument until it lands in (1, 2] so that y = (x-1)/(x+1) <= 1/3.  The
    # halving count is carried as an exact multiple of the directly-summed log(2).
    halvings = 0
    reduced = argument
    while reduced > 2:
        reduced = reduced / 2
        halvings += 1
    y = (reduced - 1) / (reduced + 1)
    if y == 0:
        core = Bracket(_ZERO, _ZERO)
    else:
        target = _exp2(-bits - 4)
        y_squared = y * y
        total = Fraction(0)
        power = y
        index = 0
        while True:
            total += power / (2 * index + 1)
            power = power * y_squared
            index += 1
            tail = power / ((2 * index + 1) * (1 - y_squared))
            if tail < target:
                core = Bracket(2 * total, 2 * (total + tail))
                break
    if halvings == 0:
        return _round_out(core, bits)
    two = log_bracket(Fraction(2), bits + 8)
    return _round_out(
        Bracket(core.lo + halvings * two.lo, core.hi + halvings * two.hi), bits
    )


def bracket_expression(expression: Any, bits: int) -> Bracket:
    """Enclose ``expression`` in an exact rational interval carrying ~``bits`` bits.

    Supported grammar: rational literals, ``Add``, ``Mul``, ``Pow`` with an integer or
    half-integer exponent, ``exp`` of a rational, ``log`` of a positive rational, and ``pi``.
    Anything else raises, so an unsupported constant can never be silently mis-bracketed.
    """

    if bits < 8:
        raise ExactBracketError("precision budget must be at least 8 bits")
    expression = sp.sympify(expression)
    return _round_out(_bracket(expression, bits), bits)


def _bracket(expression: Any, bits: int) -> Bracket:
    if expression.is_Rational:
        value = Fraction(int(expression.p), int(expression.q))
        return Bracket(value, value)
    if expression.is_Integer:
        value = Fraction(int(expression))
        return Bracket(value, value)
    if expression is sp.pi:
        return pi_bracket(bits)
    if isinstance(expression, sp.Add):
        total = Bracket(_ZERO, _ZERO)
        for term in expression.args:
            total = _round_out(_add(total, _bracket(term, bits)), bits)
        return total
    if isinstance(expression, sp.Mul):
        product = Bracket(_ONE, _ONE)
        for factor in expression.args:
            product = _round_out(_mul(product, _bracket(factor, bits)), bits)
        return product
    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        if not exponent.is_Rational:
            raise ExactBracketError(f"unsupported exponent: {exponent}")
        numerator = int(exponent.p)
        denominator = int(exponent.q)
        if denominator not in (1, 2):
            raise ExactBracketError(f"unsupported exponent denominator: {exponent}")
        inner = _round_out(_bracket(base, bits + 8), bits + 8)
        if denominator == 2:
            inner = _round_out(_sqrt(inner, bits + 8), bits + 8)
        return _round_out(_integer_power(inner, numerator, bits + 8), bits)
    if isinstance(expression, sp.exp):
        (argument,) = expression.args
        if not argument.is_Rational:
            raise ExactBracketError(f"unsupported exp argument: {argument}")
        return exp_bracket(Fraction(int(argument.p), int(argument.q)), bits)
    if isinstance(expression, sp.log):
        (argument,) = expression.args
        if not argument.is_Rational:
            raise ExactBracketError(f"unsupported log argument: {argument}")
        return log_bracket(Fraction(int(argument.p), int(argument.q)), bits)
    raise ExactBracketError(f"unsupported expression node: {type(expression).__name__}")


@dataclass(frozen=True)
class Comparison:
    """The verdict of an exact comparison, with the rationals that witness it."""

    verdict: str
    bits: int | None
    left: Bracket | None
    right: Bracket | None

    def separated(self) -> bool:
        return self.verdict in (LESS, GREATER)

    def as_receipt(self) -> dict[str, Any]:
        witness: dict[str, Any] = {"verdict": self.verdict, "separating_bits": self.bits}
        if self.left is not None and self.right is not None:
            witness["left_bracket"] = self.left.as_strings()
            witness["right_bracket"] = self.right.as_strings()
        return witness


def compare_expressions(
    left: Any, right: Any, ladder: tuple[int, ...] = DEFAULT_LADDER
) -> Comparison:
    """Decide the order of two closed-form reals, or report that the budget did not decide it.

    Equality is claimed only for expressions that are the same canonical object; a rational
    bracket is incapable of proving two distinct closed forms equal, and this refuses to guess.
    """

    left_expression = sp.sympify(left)
    right_expression = sp.sympify(right)
    if left_expression == right_expression:
        return Comparison(EQUAL, None, None, None)
    for bits in ladder:
        left_bracket = bracket_expression(left_expression, bits)
        right_bracket = bracket_expression(right_expression, bits)
        if left_bracket.hi < right_bracket.lo:
            return Comparison(LESS, bits, left_bracket, right_bracket)
        if right_bracket.hi < left_bracket.lo:
            return Comparison(GREATER, bits, left_bracket, right_bracket)
    return Comparison(UNSEPARATED, None, None, None)
