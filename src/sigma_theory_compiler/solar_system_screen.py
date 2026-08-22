"""The solar-system screen: cheap, one-sided, and almost devoid of evidential weight.

**Read the framing before the verdicts.**  This module is a *screen*, not a discovery.  Solar
accelerations run from about 3.96e-2 m/s^2 at Mercury down to about 6.56e-6 m/s^2 at Neptune --
between four and eight orders of magnitude above the MOND scale ``a0 = 1.2e-10 m/s^2``.  Every
MOND-class response law in the literature is *constructed* to reduce to Newton in exactly that
regime, so a PASS here confirms only that the author of the law did the thing every author of
such a law does.  It is necessary and it is nearly free of information.

What the regime *does* have is refutation power, and a lot of it.  A law that deviates
measurably at solar accelerations is dead, and it dies in milliseconds rather than after a
rotation-curve campaign.  That asymmetry is the whole point: this file is a cheap gate in front
of expensive lanes, and the receipt says so in ``evidential_framing`` in those words.

The screen is one-sided by construction.  Three separate mechanisms all point the same way --
towards under-rejecting rather than over-rejecting:

* **The bounds are deliberately loose.**  Every entry in :data:`BOUND_SCHEDULE` is a round
  number chosen to be *weaker* than the published ephemeris constraint it stands for, and each
  carries ``provenance_class: "conservative_round_number"`` saying so.  No verdict in this
  module depends on a decimal transcribed out of a paper's table.
* **A declared modelling slack.**  The precession channel uses the near-circular apsidal
  formula, which is an O(1) approximation at planetary eccentricities.  Bounds are multiplied
  by :data:`MODEL_SLACK` before any comparison, so an O(1) modelling error cannot manufacture a
  refutation.
* **A FAIL is a proof, not an estimate.**  Every quantity is an exact rational *bracket*.  A
  cell is called ``EXCEEDS`` only when the bracket's lower endpoint is already above the
  effective bound, and ``WITHIN`` only when its upper endpoint is already below.  A bracket
  straddling the bound returns ``UNRESOLVED`` and escalates precision; if the declared ladder
  runs out, the screen returns UNRESOLVED and refuses to call it a pass.

**The law grammar.**  A candidate response law is

    nu(y) = (P(u) / Q(u))^beta,     u = y^(-1/2),     y = g_N / a0

with ``P``, ``Q`` rational-coefficient polynomials in ``u`` and ``beta`` an exact rational.  The
observed acceleration is ``g = nu(y) * g_N``.  This is the interpolation grammar the GPU
baryonic screen already enumerates, widened in one deliberate way: the constant terms are free,
so laws that *fail* to reduce to Newton are expressible.  A grammar in which every member
reduces to Newton could not host the negative controls this screen exists to run.

**Two channels, and one of them had to be repaired to be honest.**

``fractional_deviation``
    A *constant* ``nu`` is not observable.  It rescales ``GM_Sun`` and nothing else, and orbit
    fitting absorbs it exactly.  So the channel does not test ``|nu - 1|``; it tests
    ``|nu(y_i) / nu(y_ref) - 1|`` against the most Newtonian anchor, which is the part a
    rescaling cannot hide.  :func:`is_gm_degenerate` decides constancy exactly on the
    coefficient tuples, and a constant law is reported as ``gm_degenerate`` with its pass
    explicitly marked worthless rather than being silently failed.

``perihelion_precession``
    For ``a(r) = g_N(r) nu(g_N/a0)`` the near-circular apsidal angle is ``pi/sqrt(3 + s)`` with
    ``s = d ln a / d ln r``, so the advance per orbit is ``2 pi (1/sqrt(3+s) - 1)``.  Since
    ``g_N ~ r^-2`` gives ``s = -2 - 2L`` with ``L = d ln nu / d ln y``, the advance per orbit is
    ``2 pi (1/sqrt(1 - 2L) - 1)`` -- exactly zero when ``L = 0``, which is Newton.  Converting
    radians to milliarcseconds divides by ``pi`` again, so **pi cancels completely** and the
    whole channel is algebraic:

        advance_mas_per_century = 1296000000 * (1/sqrt(1 - 2L) - 1) * (100 / T_years)

    ``L`` needs no logarithm either: ``L = -(beta/2) * (u P'(u)/P(u) - u Q'(u)/Q(u))``.  Nothing
    transcendental is evaluated anywhere on the certificate path.

**What this screen measured.**  The local response factor carried by all twelve families that
survived ``nonlocal-localization-v1`` is ``sqrt(1 + a0/g_N)``, which is ``P = 1 + u^2``,
``Q = 1``, ``beta = 1/2`` in this grammar.  It FAILS, on six of the eight anchors::

    mercury    816 mas/cy   bound 10        EXCEEDS
    venus     1115 mas/cy   bound 10        EXCEEDS
    earth     1311 mas/cy   bound 10        EXCEEDS
    mars      1619 mas/cy   bound 10        EXCEEDS
    jupiter   2992 mas/cy   bound 1000      EXCEEDS
    saturn    4050 mas/cy   bound 100       EXCEEDS
    uranus    5747 mas/cy   bound 1000000   WITHIN
    neptune   7195 mas/cy   bound 10000000  WITHIN

That is reported as what it is: a refutation of the *local factor taken alone*.  Those families
also carry a curvature-screening factor whose solar behaviour this module does not evaluate, and
the receipt states that scope restriction rather than claiming the families themselves are dead.

**And what it hands back.**  A gate that only says no steers nothing, so the receipt also
carries the *admissible region*: for ``nu = 1 + c y^-p``, the largest coefficient the solar
system still tolerates, bisected exactly and witnessed on both sides::

    p = 1   c < 0.0030889...     (linear recovery: the simple function's c = 1 is out by ~324x)
    p = 2   c < 3354.16...       (quadratic recovery: the standard function's c = 1/2 is safe)
    p = 3   c < 1215006720
    p = 4   above the declared search cap of 2^40

The bisection is a proof rather than a sample because the verdict is monotone in ``c``, which
:func:`recovery_frontier` states and the tests check on a ladder.

Nothing on a certificate path is a float.  Constants are declared as exact literals -- a decimal
with an optional exponent, or an integer ratio -- every intermediate is a
:class:`~sigma_theory_compiler.exact_real_bracket.Bracket` with :class:`fractions.Fraction`
endpoints, and every comparison is integer arithmetic.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .exact_real_bracket import Bracket, ExactBracketError, pi_bracket
from .sigma_core import canonical_sha256

SCREEN_SCHEMA = "invariant-solar-system-screen-1.0"
RECEIPT_PATH = "runs/gpu-baryonic-screen/solar-system-screen-v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/solar_system_screen.py"

PASS = "PASS"
FAIL = "FAIL"
UNRESOLVED = "UNRESOLVED"

WITHIN = "WITHIN"
EXCEEDS = "EXCEEDS"
UNSTABLE = "UNSTABLE_CIRCULAR_ORBIT"

CHANNEL_FRACTIONAL = "fractional_deviation"
CHANNEL_PRECESSION = "perihelion_precession_mas_per_century"
CHANNELS = (CHANNEL_FRACTIONAL, CHANNEL_PRECESSION)

#: Declared precision ladder, in significant bits.  A cell that cannot be decided at the last
#: rung is UNRESOLVED; it is never rounded into a pass.
PRECISION_LADDER = (64, 128, 256, 512)

#: Milliarcseconds per radian times ``2 pi`` -- the exact integer ``2 * 648000000`` that the
#: precession channel is left with once ``pi`` cancels between the apsidal formula and the
#: radian-to-milliarcsecond conversion.
MAS_PER_CENTURY_SCALE = 1296000000

#: Multiplies every declared bound before comparison, absorbing the O(1) error of the
#: near-circular apsidal approximation at planetary eccentricities.  Declared here, before any
#: candidate is evaluated, and never tuned against a result.
MODEL_SLACK = Fraction(10)

#: Universal constants, as exact decimal strings.  ``GM_SUN`` is the IAU 2015 Resolution B3
#: nominal solar mass parameter and ``ASTRONOMICAL_UNIT`` is the IAU 2012 definition; both are
#: defined constants, not measurements, so neither carries an uncertainty here.  ``A0`` is the
#: conventional MOND acceleration scale that the whole screen is positioned against.
GM_SUN_M3_PER_S2 = "1.3271244e20"
ASTRONOMICAL_UNIT_M = "1.495978707e11"
A0_M_PER_S2 = "1.2e-10"

#: Declared orbital anchors: nominal semi-major axis in AU and sidereal period in Julian years.
#: Sub-percent changes to either column move no verdict in this module -- the margins are
#: orders of magnitude -- and :func:`kepler_consistency` checks the two columns against each
#: other so a transcription slip cannot pass silently.
ANCHORS: tuple[dict[str, str], ...] = (
    {"name": "mercury", "semi_major_axis_au": "0.38709927", "period_years": "0.2408467"},
    {"name": "venus", "semi_major_axis_au": "0.72333566", "period_years": "0.61519726"},
    {"name": "earth", "semi_major_axis_au": "1.00000261", "period_years": "1.0000174"},
    {"name": "mars", "semi_major_axis_au": "1.52371034", "period_years": "1.8808476"},
    {"name": "jupiter", "semi_major_axis_au": "5.20288700", "period_years": "11.862615"},
    {"name": "saturn", "semi_major_axis_au": "9.53667594", "period_years": "29.447498"},
    {"name": "uranus", "semi_major_axis_au": "19.18916464", "period_years": "84.016846"},
    {"name": "neptune", "semi_major_axis_au": "30.06992276", "period_years": "164.79132"},
)

#: Relative tolerance for the Kepler third-law self-check on :data:`ANCHORS`.  It has to admit
#: real physics the two-body reduction leaves out: ``T^2 = 4 pi^2 a^3 / (G (M + m))`` means
#: Jupiter's own mass alone shifts the ratio by about 1e-3, and the elements are osculating
#: rather than mean.  It is still far tighter than a mistyped digit, which is what it is for.
KEPLER_TOLERANCE = "3e-3"

#: Seconds in a Julian year, exactly ``365.25 * 86400``.
SECONDS_PER_JULIAN_YEAR = 31557600

#: The declared observational bounds.  **Every one of these is a deliberately loose round
#: number.**  Published planetary-ephemeris analyses constrain supplementary perihelion
#: advances far more tightly than this -- sub-milliarcsecond per century for the inner planets,
#: and Cassini ranging does the same for Saturn -- while Uranus and Neptune are weak.  Loose
#: round numbers are used on purpose: the screen may then only under-reject, and no verdict
#: rests on a decimal that could have been mis-transcribed from a table.  Tightening these
#: values can only make the screen sharper, never reverse a FAIL already issued.
BOUND_SCHEDULE: dict[str, dict[str, str]] = {
    "mercury": {CHANNEL_FRACTIONAL: "1e-8", CHANNEL_PRECESSION: "1"},
    "venus": {CHANNEL_FRACTIONAL: "1e-8", CHANNEL_PRECESSION: "1"},
    "earth": {CHANNEL_FRACTIONAL: "1e-8", CHANNEL_PRECESSION: "1"},
    "mars": {CHANNEL_FRACTIONAL: "1e-8", CHANNEL_PRECESSION: "1"},
    "jupiter": {CHANNEL_FRACTIONAL: "1e-7", CHANNEL_PRECESSION: "100"},
    "saturn": {CHANNEL_FRACTIONAL: "1e-8", CHANNEL_PRECESSION: "10"},
    "uranus": {CHANNEL_FRACTIONAL: "1e-5", CHANNEL_PRECESSION: "100000"},
    "neptune": {CHANNEL_FRACTIONAL: "1e-4", CHANNEL_PRECESSION: "1000000"},
}

BOUND_PROVENANCE = {
    "provenance_class": "conservative_round_number",
    "transcribed_from_a_published_table": False,
    "declared_looser_than_published_constraints": True,
    "one_sided_consequence": (
        "the screen may fail to reject a law that published data would reject; it cannot "
        "reject a law that published data would admit"
    ),
    "tightening_cannot_reverse_a_fail": True,
}

EVIDENTIAL_FRAMING = {
    "artifact_kind": "screen",
    "is_a_discovery": False,
    "pass_is_necessary": True,
    "pass_is_sufficient": False,
    "pass_evidential_weight": "almost none",
    "why": (
        "solar accelerations sit 4 to 8 orders of magnitude above a0, and every MOND-class "
        "response law is constructed to reduce to Newton there; passing this screen confirms "
        "the construction, not the law"
    ),
    "fail_evidential_weight": "decisive for the law as declared",
    "observational_data_opened": False,
    "sealed_no_refit_trial": False,
    "may_be_cited_as_confirmation": False,
}


class SolarScreenError(ValueError):
    """Raised on a malformed law, a malformed declaration, or a tampered certificate."""


# ---------------------------------------------------------------------------
# Exact rational interval arithmetic
# ---------------------------------------------------------------------------
#
# ``Bracket`` and its invariants are reused from :mod:`.exact_real_bracket`; the operations
# below are the ones this screen needs and that module does not export.  Every one rounds its
# result *outward*, so containment survives every step while endpoints stay small.

_ZERO = Fraction(0)
_ONE = Fraction(1)


def _exp2(power: int) -> Fraction:
    return Fraction(1 << power, 1) if power >= 0 else Fraction(1, 1 << -power)


def _round_out(value: Bracket, bits: int) -> Bracket:
    """Widen ``value`` outward onto a dyadic grid carrying about ``bits`` significant bits."""

    magnitude = max(abs(value.lo), abs(value.hi))
    if magnitude == 0:
        return value
    exponent = abs(magnitude.numerator).bit_length() - magnitude.denominator.bit_length() + 1
    grid = _exp2(exponent - bits)
    low = value.lo / grid
    high = value.hi / grid
    lo = (low.numerator // low.denominator) * grid
    hi = (-((-high.numerator) // high.denominator)) * grid
    return Bracket(lo, hi)


def _exact(value: Fraction) -> Bracket:
    return Bracket(value, value)


def _add(left: Bracket, right: Bracket) -> Bracket:
    return Bracket(left.lo + right.lo, left.hi + right.hi)


def _sub(left: Bracket, right: Bracket) -> Bracket:
    return Bracket(left.lo - right.hi, left.hi - right.lo)


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


def _div(left: Bracket, right: Bracket) -> Bracket:
    return _mul(left, _reciprocal(right))


def _integer_power(value: Bracket, exponent: int, bits: int) -> Bracket:
    if exponent < 0:
        return _reciprocal(_integer_power(value, -exponent, bits))
    result = _exact(_ONE)
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _round_out(_mul(result, base), bits)
        remaining >>= 1
        if remaining:
            base = _round_out(_mul(base, base), bits)
    return result


def integer_nth_root(value: int, degree: int) -> int:
    """``floor(value ** (1/degree))`` for ``value >= 0`` and ``degree >= 1``, exactly."""

    if degree < 1:
        raise SolarScreenError("root degree must be at least one")
    if value < 0:
        raise SolarScreenError("cannot take a real root of a negative integer here")
    if degree == 1 or value < 2:
        return value
    guess = 1 << ((value.bit_length() + degree - 1) // degree)
    while True:
        following = ((degree - 1) * guess + value // guess ** (degree - 1)) // degree
        if following >= guess:
            return guess
        guess = following


def _nth_root_rational(value: Fraction, degree: int, bits: int) -> Bracket:
    """Bracket ``value ** (1/degree)`` for a non-negative rational, exactly."""

    if value < 0:
        raise SolarScreenError("cannot take a real root of a negative rational")
    if value == 0:
        return _exact(_ZERO)
    shift = bits + 4
    scaled = (value.numerator * value.denominator ** (degree - 1)) << (degree * shift)
    root = integer_nth_root(scaled, degree)
    divisor = value.denominator << shift
    return Bracket(Fraction(root, divisor), Fraction(root + 1, divisor))


def _nth_root(value: Bracket, degree: int, bits: int) -> Bracket:
    if value.lo < _ZERO:
        raise SolarScreenError("cannot take a real root of a bracket reaching below zero")
    low = _nth_root_rational(value.lo, degree, bits)
    high = _nth_root_rational(value.hi, degree, bits)
    return Bracket(low.lo, high.hi)


def _rational_power(value: Bracket, exponent: Fraction, bits: int) -> Bracket:
    """Bracket ``value ** exponent`` for a strictly positive bracket and rational exponent."""

    if not value.is_positive():
        raise SolarScreenError("rational powers require a strictly positive base bracket")
    raised = _integer_power(value, exponent.numerator, bits + 16)
    if exponent.denominator == 1:
        return _round_out(raised, bits)
    return _round_out(_nth_root(raised, exponent.denominator, bits + 16), bits)


def _absolute(value: Bracket) -> Bracket:
    if value.lo >= _ZERO:
        return value
    if value.hi <= _ZERO:
        return Bracket(-value.hi, -value.lo)
    return Bracket(_ZERO, max(-value.lo, value.hi))


# ---------------------------------------------------------------------------
# Exact decimal parsing and deterministic rendering
# ---------------------------------------------------------------------------

_RENDER = Context(prec=12, rounding=ROUND_HALF_EVEN)


def exact_rational(text: str) -> Fraction:
    """Parse a declared constant into an exact :class:`Fraction`, or refuse it.

    Two spellings are accepted and both are exact: a decimal literal with an optional exponent
    (``"1.2e-10"``) and an integer ratio (``"1/300"``).  Nothing else is, so a declaration can
    never enter this module as an approximation.
    """

    if not isinstance(text, str) or not text.strip():
        raise SolarScreenError("declared constants must be non-empty exact literals")
    stripped = text.strip()
    try:
        return Fraction(stripped) if "/" in stripped else Fraction(Decimal(stripped))
    except (ArithmeticError, ValueError) as error:
        raise SolarScreenError(f"not an exact literal: {text!r}") from error


def _render(value: Fraction) -> str:
    """A deterministic 12-significant-digit rendering, for reading only, never for deciding."""

    quotient = _RENDER.divide(Decimal(value.numerator), Decimal(value.denominator))
    return str(quotient)


def _bracket_block(value: Bracket) -> dict[str, str]:
    return {
        "lo": str(value.lo),
        "hi": str(value.hi),
        "lo_decimal": _render(value.lo),
        "hi_decimal": _render(value.hi),
    }


# ---------------------------------------------------------------------------
# The candidate response law
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseLaw:
    """``nu(y) = (P(u)/Q(u))^beta`` with ``u = y^(-1/2)`` and ``y = g_N / a0``.

    Coefficients run from the constant term upward: ``numerator[k]`` multiplies ``u**k``.  The
    constant terms are unconstrained on purpose, so a law that never reduces to Newton is
    expressible and can be run as a negative control.
    """

    name: str
    numerator: tuple[Fraction, ...]
    denominator: tuple[Fraction, ...]
    beta: Fraction
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise SolarScreenError("a response law needs a name")
        for label, coefficients in (
            ("numerator", self.numerator),
            ("denominator", self.denominator),
        ):
            if not coefficients:
                raise SolarScreenError(f"{label} needs at least one coefficient")
            if any(not isinstance(value, Fraction) for value in coefficients):
                raise SolarScreenError(f"{label} coefficients must be exact rationals")
        if not isinstance(self.beta, Fraction) or self.beta == 0:
            raise SolarScreenError("beta must be a non-zero exact rational")

    def declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numerator": [str(value) for value in self.numerator],
            "denominator": [str(value) for value in self.denominator],
            "beta": str(self.beta),
            "note": self.note,
            "rendered": self.rendered(),
        }

    def rendered(self) -> str:
        return f"nu(y) = [({_polynomial(self.numerator)}) / ({_polynomial(self.denominator)})]" \
            f"^({self.beta}),  u = y^(-1/2),  y = g_N/a0"


def _polynomial(coefficients: Sequence[Fraction]) -> str:
    parts: list[str] = []
    for power, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        if power == 0:
            parts.append(str(coefficient))
            continue
        symbol = "u" if power == 1 else f"u^{power}"
        parts.append(symbol if coefficient == 1 else f"{coefficient}*{symbol}")
    return " + ".join(parts) if parts else "0"


def law_from_declaration(declaration: Mapping[str, Any]) -> ResponseLaw:
    """Rebuild a law from its receipt block; the inverse of :meth:`ResponseLaw.declaration`."""

    try:
        return ResponseLaw(
            name=str(declaration["name"]),
            numerator=tuple(Fraction(str(value)) for value in declaration["numerator"]),
            denominator=tuple(Fraction(str(value)) for value in declaration["denominator"]),
            beta=Fraction(str(declaration["beta"])),
            note=str(declaration.get("note", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SolarScreenError(f"malformed law declaration: {error}") from error


def is_gm_degenerate(law: ResponseLaw) -> bool:
    """Is ``nu`` constant in ``u``?  Then it rescales ``GM_Sun`` and is not observable.

    Decided exactly on the coefficient tuples: ``P/Q`` is constant precisely when ``P`` and
    ``Q`` are proportional, which is ``P_k * Q_0 == Q_k * P_0`` for every ``k`` once the leading
    constants are non-zero.  No evaluation, no tolerance.
    """

    width = max(len(law.numerator), len(law.denominator))
    top = [law.numerator[k] if k < len(law.numerator) else _ZERO for k in range(width)]
    bottom = [law.denominator[k] if k < len(law.denominator) else _ZERO for k in range(width)]
    return all(top[k] * bottom[0] == bottom[k] * top[0] for k in range(width))


def _evaluate_polynomial(
    coefficients: Sequence[Fraction], point: Bracket, bits: int
) -> Bracket:
    total = _exact(_ZERO)
    for coefficient in reversed(coefficients):
        total = _round_out(_add(_mul(total, point), _exact(coefficient)), bits)
    return total


def _derivative(coefficients: Sequence[Fraction]) -> tuple[Fraction, ...]:
    return tuple(coefficient * power for power, coefficient in enumerate(coefficients))[1:] or (
        _ZERO,
    )


# ---------------------------------------------------------------------------
# The declared solar anchors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Anchor:
    """One declared solar-system probe point, with its exact Newtonian acceleration."""

    name: str
    semi_major_axis_au: Fraction
    period_years: Fraction
    newtonian_acceleration: Fraction
    y: Fraction


def build_anchors() -> tuple[Anchor, ...]:
    """The declared anchor table with ``g_N = GM/r^2`` and ``y = g_N/a0`` in exact rationals."""

    gm = exact_rational(GM_SUN_M3_PER_S2)
    au = exact_rational(ASTRONOMICAL_UNIT_M)
    a0 = exact_rational(A0_M_PER_S2)
    anchors: list[Anchor] = []
    for row in ANCHORS:
        axis = exact_rational(row["semi_major_axis_au"])
        radius = axis * au
        acceleration = gm / (radius * radius)
        anchors.append(
            Anchor(
                name=row["name"],
                semi_major_axis_au=axis,
                period_years=exact_rational(row["period_years"]),
                newtonian_acceleration=acceleration,
                y=acceleration / a0,
            )
        )
    return tuple(anchors)


def kepler_consistency(anchors: Sequence[Anchor], bits: int = 128) -> list[dict[str, Any]]:
    """Check the declared axes against the declared periods: ``T^2 GM / (4 pi^2 a^3) = 1``.

    This is the one place ``pi`` appears, and it decides nothing about any candidate -- it is a
    guard on the anchor table itself, so a mistyped digit cannot ride into a verdict.
    """

    gm = exact_rational(GM_SUN_M3_PER_S2)
    au = exact_rational(ASTRONOMICAL_UNIT_M)
    tolerance = exact_rational(KEPLER_TOLERANCE)
    pi_squared = _round_out(_mul(pi_bracket(bits), pi_bracket(bits)), bits)
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        seconds = anchor.period_years * SECONDS_PER_JULIAN_YEAR
        radius = anchor.semi_major_axis_au * au
        ratio = _div(
            _exact(seconds * seconds * gm),
            _round_out(_mul(_exact(4 * radius**3), pi_squared), bits),
        )
        residual = _absolute(_sub(ratio, _exact(_ONE)))
        rows.append(
            {
                "anchor": anchor.name,
                "kepler_ratio": _bracket_block(_round_out(ratio, 64)),
                "residual_upper_bound": str(residual.hi),
                "residual_upper_bound_decimal": _render(residual.hi),
                "within_tolerance": residual.hi <= tolerance,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Evaluating a law at one anchor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchorEvaluation:
    """``nu``, the logarithmic slope ``L``, and the precession, all as exact brackets."""

    anchor: str
    nu: Bracket
    log_slope: Bracket
    precession_mas_per_century: Bracket
    stable_circular_orbit: bool


def evaluate_at_anchor(law: ResponseLaw, anchor: Anchor, bits: int) -> AnchorEvaluation:
    """Evaluate ``law`` at ``anchor`` in exact interval arithmetic at ``bits`` of precision."""

    working = bits + 32
    u = _round_out(_nth_root_rational(_ONE / anchor.y, 2, working), working)
    top = _evaluate_polynomial(law.numerator, u, working)
    bottom = _evaluate_polynomial(law.denominator, u, working)
    if not top.is_positive() or not bottom.is_positive():
        raise SolarScreenError(
            f"law {law.name!r} is not provably positive at anchor {anchor.name!r}; "
            "the screen refuses to evaluate a response it cannot bound away from zero"
        )
    ratio = _round_out(_div(top, bottom), working)
    nu = _rational_power(ratio, law.beta, bits)

    top_derivative = _evaluate_polynomial(_derivative(law.numerator), u, working)
    bottom_derivative = _evaluate_polynomial(_derivative(law.denominator), u, working)
    logarithmic = _sub(
        _round_out(_div(_mul(u, top_derivative), top), working),
        _round_out(_div(_mul(u, bottom_derivative), bottom), working),
    )
    slope = _round_out(_mul(_exact(-law.beta / 2), logarithmic), bits)

    discriminant = _sub(_exact(_ONE), _mul(_exact(Fraction(2)), slope))
    if not discriminant.is_positive():
        return AnchorEvaluation(anchor.name, nu, slope, Bracket(_ZERO, _ZERO), False)
    root = _nth_root(discriminant, 2, working)
    advance = _sub(_reciprocal(root), _exact(_ONE))
    centuries = _exact(Fraction(100) / anchor.period_years)
    precession = _round_out(
        _mul(_mul(advance, _exact(Fraction(MAS_PER_CENTURY_SCALE))), centuries), bits
    )
    return AnchorEvaluation(anchor.name, nu, slope, precession, True)


# ---------------------------------------------------------------------------
# Deciding a cell against a declared bound
# ---------------------------------------------------------------------------


def decide_cell(magnitude: Bracket, effective_bound: Fraction) -> str:
    """``EXCEEDS`` only when proved, ``WITHIN`` only when proved, ``UNRESOLVED`` otherwise."""

    if magnitude.lo > effective_bound:
        return EXCEEDS
    if magnitude.hi <= effective_bound:
        return WITHIN
    return UNRESOLVED


def _margin(magnitude: Bracket, effective_bound: Fraction, status: str) -> Fraction:
    """A gradient, not a verdict: how far the cell sits from its bound, as a ratio."""

    if effective_bound == 0:
        return Fraction(0)
    if status == EXCEEDS:
        return magnitude.lo / effective_bound
    return magnitude.hi / effective_bound


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def screen_law(
    law: ResponseLaw,
    anchors: Sequence[Anchor] | None = None,
    bounds: Mapping[str, Mapping[str, str]] | None = None,
    ladder: Sequence[int] = PRECISION_LADDER,
) -> dict[str, Any]:
    """Run the solar screen on one candidate law and return its certificate body."""

    anchors = anchors if anchors is not None else build_anchors()
    bounds = bounds if bounds is not None else BOUND_SCHEDULE
    if not ladder:
        raise SolarScreenError("the precision ladder is empty; there is nothing to decide with")
    degenerate = is_gm_degenerate(law)
    reference = max(anchors, key=lambda item: item.y)

    cells: list[dict[str, Any]] = []
    verdict = PASS
    for bits in ladder:
        cells = []
        verdict = PASS
        evaluations = {
            anchor.name: evaluate_at_anchor(law, anchor, bits) for anchor in anchors
        }
        reference_nu = evaluations[reference.name].nu
        for anchor in anchors:
            evaluation = evaluations[anchor.name]
            declared = bounds.get(anchor.name)
            if declared is None:
                raise SolarScreenError(f"no declared bound for anchor {anchor.name!r}")
            deviation = _absolute(
                _sub(_round_out(_div(evaluation.nu, reference_nu), bits), _exact(_ONE))
            )
            precession = _absolute(evaluation.precession_mas_per_century)
            for channel, magnitude in (
                (CHANNEL_FRACTIONAL, deviation),
                (CHANNEL_PRECESSION, precession),
            ):
                bound = exact_rational(declared[channel])
                effective = bound * MODEL_SLACK
                if channel == CHANNEL_PRECESSION and not evaluation.stable_circular_orbit:
                    status = UNSTABLE
                else:
                    status = decide_cell(magnitude, effective)
                cells.append(
                    {
                        "anchor": anchor.name,
                        "channel": channel,
                        "y": _render(anchor.y),
                        "newtonian_acceleration_m_per_s2": _render(
                            anchor.newtonian_acceleration
                        ),
                        "magnitude": _bracket_block(magnitude),
                        "declared_bound": declared[channel],
                        "model_slack": str(MODEL_SLACK),
                        "effective_bound": str(effective),
                        "effective_bound_decimal": _render(effective),
                        "status": status,
                        "margin_ratio": _render(_margin(magnitude, effective, status)),
                    }
                )
                if status in (EXCEEDS, UNSTABLE):
                    verdict = FAIL
                elif status == UNRESOLVED and verdict != FAIL:
                    verdict = UNRESOLVED
        if verdict != UNRESOLVED:
            break

    decisive = [cell for cell in cells if cell["status"] in (EXCEEDS, UNSTABLE)]
    worst = max(
        (cell for cell in cells if cell["status"] != UNSTABLE),
        key=lambda cell: Fraction(cell["margin_ratio"]),
        default=None,
    )
    return {
        "law": law.declaration(),
        "verdict": verdict,
        "precision_bits_used": bits,
        "gm_degenerate": degenerate,
        "gm_degeneracy_note": (
            "nu is constant in u, so it rescales GM_Sun and nothing else; orbit fitting "
            "absorbs it exactly.  This PASS carries no information whatsoever."
            if degenerate
            else "nu varies across the anchor set, so the fractional channel has power here"
        ),
        "reference_anchor": reference.name,
        "cells": cells,
        "decisive_cells": [
            {"anchor": cell["anchor"], "channel": cell["channel"], "status": cell["status"]}
            for cell in decisive
        ],
        "worst_margin_ratio": worst["margin_ratio"] if worst else None,
        "worst_cell": (
            {"anchor": worst["anchor"], "channel": worst["channel"]} if worst else None
        ),
    }


# ---------------------------------------------------------------------------
# Declared control laws
# ---------------------------------------------------------------------------


def _law(name: str, numerator, denominator, beta, note: str) -> ResponseLaw:
    return ResponseLaw(
        name=name,
        numerator=tuple(Fraction(value) for value in numerator),
        denominator=tuple(Fraction(value) for value in denominator),
        beta=Fraction(beta),
        note=note,
    )


def control_laws() -> dict[str, list[ResponseLaw]]:
    """The declared control battery.  Every positive is paired with a negative that must fail."""

    return {
        "must_pass": [
            _law(
                "newton",
                (1,),
                (1,),
                1,
                "nu == 1 identically: the null law, and the one that must never be rejected",
            ),
            _law(
                "standard_like_quadratic_recovery",
                (1, 0, 0, 0, 1),
                (1,),
                1,
                "nu = 1 + y^-2, the recovery rate of the standard interpolating function; the "
                "deviation falls off fast enough to be invisible at solar accelerations",
            ),
            _law(
                "steep_cubic_recovery",
                (1, 0, 0, 0, 0, 0, 1),
                (1,),
                1,
                "nu = 1 + y^-3: steeper still, and comfortably inside every bound",
            ),
            _law(
                "rational_quadratic_recovery",
                (1, 0, 0, 0, 2),
                (1, 0, 0, 0, 1),
                1,
                "(1 + 2y^-2)/(1 + y^-2): a genuine ratio rather than a polynomial, and the "
                "screen must not reject a law merely for having a denominator",
            ),
            _law(
                "threshold_pair_inside",
                (1, 0, Fraction(1, 400)),
                (1,),
                1,
                "nu = 1 + y^-1/400.  Half of a matched pair straddling the screen's threshold: "
                "this one survives, and only one coefficient separates it from the law below",
            ),
        ],
        "must_fail": [
            _law(
                "deep_mond_everywhere",
                (0, 1),
                (1,),
                1,
                "nu = y^-1/2 at every acceleration: the deliberately non-reducing control, "
                "which never approaches Newton however strong gravity gets",
            ),
            _law(
                "simple_like_linear_recovery",
                (1, 0, 1),
                (1,),
                1,
                "nu = 1 + y^-1, the recovery rate of the simple interpolating function; it "
                "does reduce to Newton, just far too slowly",
            ),
            _law(
                "surviving_family_local_factor",
                (1, 0, 1),
                (1,),
                Fraction(1, 2),
                "sqrt(1 + a0/g_N): the local response factor carried by all twelve families "
                "that survived nonlocal-localization-v1, screened here on its own",
            ),
            _law(
                "inverted_screening",
                (1, 0, 1),
                (0, 0, 1),
                Fraction(1, 2),
                "sqrt(1 + g_N/a0): the repository's own declared wrong law, in which the "
                "enhancement grows where gravity is strong instead of where it is weak",
            ),
            _law(
                "threshold_pair_outside",
                (1, 0, Fraction(1, 300)),
                (1,),
                1,
                "nu = 1 + y^-1/300.  The other half of the matched pair: one coefficient moved "
                "by a third, and the verdict flips",
            ),
        ],
        "must_pass_but_worthless": [
            _law(
                "constant_offset",
                (1001,),
                (1000,),
                1,
                "nu == 1.001 everywhere: not observable, because it is exactly a rescaling of "
                "GM_Sun.  The screen must pass it and must say the pass is worthless",
            ),
        ],
    }


def run_controls() -> dict[str, Any]:
    """Run the declared battery and report, per law, whether it landed where it must."""

    anchors = build_anchors()
    battery = control_laws()
    rows: list[dict[str, Any]] = []
    for group, expected in (
        ("must_pass", PASS),
        ("must_fail", FAIL),
        ("must_pass_but_worthless", PASS),
    ):
        for law in battery[group]:
            result = screen_law(law, anchors=anchors)
            row = {
                "group": group,
                "law": law.name,
                "expected": expected,
                "verdict": result["verdict"],
                "as_expected": result["verdict"] == expected,
                "gm_degenerate": result["gm_degenerate"],
                "decisive_cells": result["decisive_cells"],
                "worst_margin_ratio": result["worst_margin_ratio"],
                "worst_cell": result["worst_cell"],
            }
            if group == "must_pass_but_worthless":
                row["as_expected"] = row["as_expected"] and result["gm_degenerate"]
            rows.append(row)
    return {
        "rows": rows,
        "all_as_expected": all(row["as_expected"] for row in rows),
        "positives": sum(1 for row in rows if row["expected"] == PASS),
        "negatives": sum(1 for row in rows if row["expected"] == FAIL),
    }


# ---------------------------------------------------------------------------
# The admissible region, not just a yes/no oracle
# ---------------------------------------------------------------------------
#
# A filter that only says no tells a generator nothing about where to look next.  The frontier
# below converts the same screen into a *constructive* statement: for each recovery order ``p``,
# the largest coefficient the solar system still tolerates in ``nu = 1 + c y^-p``.  That is a
# region the search may propose inside, and it is the useful output of a screen.

#: Recovery orders swept by :func:`recovery_frontier`.  ``p = 1`` is the simple interpolating
#: function's rate, ``p = 2`` the standard one's.
FRONTIER_EXPONENTS = (1, 2, 3, 4)

#: Bisection depth and the upper cap on the coefficient search, both declared before the sweep.
FRONTIER_DEPTH = 20
FRONTIER_CAP = Fraction(1 << 40)


def recovery_law(exponent: int, coefficient: Fraction) -> ResponseLaw:
    """``nu = 1 + coefficient * y^-exponent``, the one-parameter family the frontier sweeps."""

    numerator = [_ZERO] * (2 * exponent + 1)
    numerator[0] = _ONE
    numerator[2 * exponent] = coefficient
    return ResponseLaw(
        name=f"recovery_p{exponent}_c{coefficient}",
        numerator=tuple(numerator),
        denominator=(_ONE,),
        beta=_ONE,
        note=f"nu = 1 + {coefficient} * y^-{exponent}",
    )


def recovery_frontier(
    exponents: Sequence[int] = FRONTIER_EXPONENTS,
    depth: int = FRONTIER_DEPTH,
    anchors: Sequence[Anchor] | None = None,
) -> list[dict[str, Any]]:
    """Bisect the largest coefficient that survives, for each declared recovery order.

    The bisection is a proof rather than a sampling, because the screen is monotone in the
    coefficient: for ``nu = 1 + c y^-p`` the logarithmic slope is
    ``L = -p c y^-p / (1 + c y^-p)``, whose magnitude increases with ``c`` at every anchor, and
    the fractional channel does the same.  So a passing ``c`` implies every smaller one passes,
    and the two witnesses bracket the frontier exactly.
    """

    anchors = anchors if anchors is not None else build_anchors()
    rows: list[dict[str, Any]] = []
    for exponent in exponents:
        low = _ZERO
        high = _ONE
        while (
            screen_law(recovery_law(exponent, high), anchors=anchors)["verdict"] == PASS
            and high <= FRONTIER_CAP
        ):
            low = high
            high *= 2
        if screen_law(recovery_law(exponent, high), anchors=anchors)["verdict"] == PASS:
            rows.append(
                {
                    "exponent": exponent,
                    "bracketed": False,
                    "note": "the declared cap passes; the frontier is above the search range",
                    "cap": str(FRONTIER_CAP),
                }
            )
            continue
        for _ in range(depth):
            middle = (low + high) / 2
            if screen_law(recovery_law(exponent, middle), anchors=anchors)["verdict"] == PASS:
                low = middle
            else:
                high = middle
        passing = screen_law(recovery_law(exponent, low), anchors=anchors)
        failing = screen_law(recovery_law(exponent, high), anchors=anchors)
        rows.append(
            {
                "exponent": exponent,
                "bracketed": True,
                "largest_passing_coefficient": str(low),
                "largest_passing_coefficient_decimal": _render(low),
                "smallest_failing_coefficient": str(high),
                "smallest_failing_coefficient_decimal": _render(high),
                "passing_witness": {
                    "law": passing["law"]["rendered"],
                    "verdict": passing["verdict"],
                    "worst_margin_ratio": passing["worst_margin_ratio"],
                    "worst_cell": passing["worst_cell"],
                },
                "failing_witness": {
                    "law": failing["law"]["rendered"],
                    "verdict": failing["verdict"],
                    "worst_margin_ratio": failing["worst_margin_ratio"],
                    "worst_cell": failing["worst_cell"],
                },
            }
        )
    return rows


#: Memo for the default frontier sweep, keyed by a digest of every declaration that can move
#: it.  Monkeypatching any of them invalidates the entry rather than serving a stale one.
_FRONTIER_MEMO: dict[str, str] = {}


def _frontier_key() -> str:
    return canonical_sha256(
        {
            "anchors": [dict(sorted(row.items())) for row in ANCHORS],
            "bounds": {name: dict(sorted(row.items())) for name, row in BOUND_SCHEDULE.items()},
            "constants": [GM_SUN_M3_PER_S2, ASTRONOMICAL_UNIT_M, A0_M_PER_S2],
            "slack": str(MODEL_SLACK),
            "ladder": list(PRECISION_LADDER),
            "exponents": list(FRONTIER_EXPONENTS),
            "depth": FRONTIER_DEPTH,
            "cap": str(FRONTIER_CAP),
        }
    )


def default_frontier() -> list[dict[str, Any]]:
    """:func:`recovery_frontier` on the declared defaults, memoised on those declarations."""

    key = _frontier_key()
    if key not in _FRONTIER_MEMO:
        _FRONTIER_MEMO.clear()
        _FRONTIER_MEMO[key] = json.dumps(recovery_frontier(), sort_keys=True)
    return json.loads(_FRONTIER_MEMO[key])


def default_candidates() -> list[ResponseLaw]:
    """Two laws screened as candidates rather than as controls, to exercise the lane."""

    return [
        _law(
            "two_term_simple_expansion",
            (1, 0, 1, 0, -1),
            (1,),
            1,
            "nu = 1 + y^-1 - y^-2: the next order of the simple interpolating function's "
            "expansion, screened to show that a higher-order term does not rescue a law whose "
            "leading recovery is linear",
        ),
        _law(
            "generous_quadratic_recovery",
            (1, 0, 0, 0, 100),
            (1,),
            1,
            "nu = 1 + 100 y^-2: a hundred times the standard function's coefficient, screened "
            "to show how much room the quadratic-recovery region actually has",
        ),
    ]


# ---------------------------------------------------------------------------
# The sealed certificate
# ---------------------------------------------------------------------------


def _declarations() -> dict[str, Any]:
    anchors = build_anchors()
    return {
        "constants": {
            "GM_Sun_m3_per_s2": GM_SUN_M3_PER_S2,
            "astronomical_unit_m": ASTRONOMICAL_UNIT_M,
            "a0_m_per_s2": A0_M_PER_S2,
            "seconds_per_julian_year": SECONDS_PER_JULIAN_YEAR,
            "mas_per_century_scale": MAS_PER_CENTURY_SCALE,
            "model_slack": str(MODEL_SLACK),
            "precision_ladder": list(PRECISION_LADDER),
            "provenance": (
                "GM_Sun is the IAU 2015 Resolution B3 nominal solar mass parameter and the "
                "astronomical unit is the IAU 2012 definition; both are defined constants.  a0 "
                "is the conventional MOND acceleration scale"
            ),
        },
        "anchors": [
            {
                "name": anchor.name,
                "semi_major_axis_au": str(anchor.semi_major_axis_au),
                "period_years": str(anchor.period_years),
                "newtonian_acceleration_m_per_s2": str(anchor.newtonian_acceleration),
                "newtonian_acceleration_decimal": _render(anchor.newtonian_acceleration),
                "y_is_g_over_a0": _render(anchor.y),
            }
            for anchor in anchors
        ],
        "acceleration_range": {
            "strongest_m_per_s2": _render(
                max(anchor.newtonian_acceleration for anchor in anchors)
            ),
            "weakest_m_per_s2": _render(
                min(anchor.newtonian_acceleration for anchor in anchors)
            ),
            "weakest_in_units_of_a0": _render(min(anchor.y for anchor in anchors)),
            "note": (
                "even the weakest anchor sits more than four orders of magnitude above a0, "
                "which is why this regime refutes and does not discover"
            ),
        },
        "bounds": {
            "schedule": {
                name: dict(sorted(channels.items())) for name, channels in BOUND_SCHEDULE.items()
            },
            "provenance": BOUND_PROVENANCE,
        },
        "channels": {
            CHANNEL_FRACTIONAL: (
                "|nu(y_i)/nu(y_ref) - 1| against the most Newtonian anchor.  Not |nu - 1|: a "
                "constant nu rescales GM_Sun and is not observable"
            ),
            CHANNEL_PRECESSION: (
                "1296000000 * (1/sqrt(1 - 2L) - 1) * (100/T_years) with L = d ln nu / d ln y, "
                "the near-circular apsidal advance; pi cancels and nothing transcendental is "
                "evaluated"
            ),
        },
        "approximations": {
            "near_circular_apsidal_formula": (
                "the precession channel uses the near-circular apsidal angle pi/sqrt(3+s), an "
                "O(1) approximation at planetary eccentricities; MODEL_SLACK absorbs it"
            ),
            "single_body_keplerian_anchors": (
                "each planet is treated as a test body on a Keplerian orbit about the Sun; "
                "planetary perturbations are not modelled and are not needed at these margins"
            ),
        },
        "kepler_self_check": kepler_consistency(anchors),
    }


def build_receipt(extra_laws: Sequence[ResponseLaw] | None = None) -> dict[str, Any]:
    """Assemble and seal the screen's receipt."""

    anchors = build_anchors()
    declarations = _declarations()
    controls = run_controls()
    kepler_ok = all(row["within_tolerance"] for row in declarations["kepler_self_check"])
    laws = default_candidates() if extra_laws is None else list(extra_laws)
    screened = [screen_law(law, anchors=anchors) for law in laws]
    body = {
        "schema_version": SCREEN_SCHEMA,
        "source_path": SOURCE_PATH,
        "evidential_framing": EVIDENTIAL_FRAMING,
        "declarations": declarations,
        "controls": controls,
        "admissible_region": {
            "family": "nu = 1 + c * y^-p",
            "monotone_in_c": True,
            "bisection_depth": FRONTIER_DEPTH,
            "search_cap": str(FRONTIER_CAP),
            "note": (
                "the constructive output of the screen: a region the generator may propose "
                "inside, rather than an oracle that only ever says no"
            ),
            "frontier": default_frontier(),
        },
        "screened_candidates": screened,
        "scope_restriction": (
            "surviving_family_local_factor screens sqrt(1 + a0/g_N) on its own.  The twelve "
            "families of nonlocal-localization-v1 also carry a curvature-screening factor "
            "whose solar behaviour is NOT evaluated here, so this is a refutation of the local "
            "factor taken alone and not of those families as published"
        ),
        "decision": (
            "SCREEN_OPERATIONAL"
            if controls["all_as_expected"] and kepler_ok
            else "CONTROLS_FAILED"
        ),
    }
    body["certificate_sha256"] = canonical_sha256(body)
    return body


def verify_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every claim from the receipt's own declarations and re-seal.

    Two jobs, not one.  The recomputation kills a receipt that is *wrong*: laws are rebuilt
    from their declaration blocks and re-screened, and every verdict, every cell status and
    every bracket must come back identical.  The reseal kills a receipt that is merely
    *edited*: any byte moved anywhere in the body changes the hash.
    """

    problems: list[str] = []
    if receipt.get("schema_version") != SCREEN_SCHEMA:
        problems.append("schema_version does not match this module")

    body = {key: value for key, value in receipt.items() if key != "certificate_sha256"}
    if canonical_sha256(body) != receipt.get("certificate_sha256"):
        problems.append("certificate_sha256 does not seal this body")

    declared = receipt.get("declarations", {})
    if declared != _declarations():
        problems.append("declarations block does not match this module's declarations")

    if receipt.get("evidential_framing") != EVIDENTIAL_FRAMING:
        problems.append("evidential_framing has been altered")

    anchors = build_anchors()
    for row in receipt.get("controls", {}).get("rows", []):
        group = row.get("group")
        expected = {"must_pass": PASS, "must_fail": FAIL, "must_pass_but_worthless": PASS}.get(
            str(group)
        )
        if expected is None:
            problems.append(f"unknown control group {group!r}")
            continue
        if row.get("expected") != expected:
            problems.append(f"control {row.get('law')!r} declares the wrong expectation")
    recomputed = run_controls()
    if receipt.get("controls") != recomputed:
        problems.append("controls block does not reproduce")

    region = receipt.get("admissible_region", {})
    if region.get("frontier") != default_frontier():
        problems.append("admissible_region frontier does not reproduce")

    for entry in receipt.get("screened_candidates", []):
        law = law_from_declaration(entry.get("law", {}))
        again = screen_law(law, anchors=anchors)
        if again != entry:
            problems.append(f"screened candidate {law.name!r} does not reproduce")

    decision = (
        "SCREEN_OPERATIONAL"
        if recomputed["all_as_expected"]
        and all(row["within_tolerance"] for row in _declarations()["kepler_self_check"])
        else "CONTROLS_FAILED"
    )
    if receipt.get("decision") != decision:
        problems.append("decision does not follow from the controls")

    return {"accepted": not problems, "problems": problems}


def tamper_probes(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Forge the receipt in several ways; every forgery must be rejected."""

    def reseal(body: dict[str, Any]) -> dict[str, Any]:
        body.pop("certificate_sha256", None)
        body["certificate_sha256"] = canonical_sha256(body)
        return body

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(receipt))

    probes: list[dict[str, Any]] = []

    def probe(name: str, forged: Mapping[str, Any]) -> None:
        outcome = verify_receipt(forged)
        probes.append(
            {"probe": name, "rejected": not outcome["accepted"], "problems": outcome["problems"]}
        )

    unsealed = clone()
    for row in unsealed["controls"]["rows"]:
        if row["group"] == "must_fail":
            row["verdict"] = PASS
            row["as_expected"] = False
            break
    probe("unsealed_verdict_flip", unsealed)

    resealed = clone()
    for row in resealed["controls"]["rows"]:
        if row["group"] == "must_fail":
            row["verdict"] = PASS
            row["as_expected"] = True
            break
    probe("resealed_verdict_flip", reseal(resealed))

    loosened = clone()
    loosened["declarations"]["bounds"]["schedule"]["mercury"][CHANNEL_PRECESSION] = "1e9"
    probe("loosened_mercury_bound", reseal(loosened))

    slack = clone()
    slack["declarations"]["constants"]["model_slack"] = "1000000"
    probe("inflated_model_slack", reseal(slack))

    framing = clone()
    framing["evidential_framing"] = dict(EVIDENTIAL_FRAMING)
    framing["evidential_framing"]["pass_is_sufficient"] = True
    probe("framing_upgraded_to_confirmation", reseal(framing))

    anchor = clone()
    anchor["declarations"]["anchors"][0]["period_years"] = "24.08467"
    probe("mistyped_anchor_period", reseal(anchor))

    if receipt.get("screened_candidates"):
        candidate = clone()
        candidate["screened_candidates"][0]["verdict"] = (
            FAIL if candidate["screened_candidates"][0]["verdict"] == PASS else PASS
        )
        probe("screened_candidate_verdict_flip", reseal(candidate))

        swapped = clone()
        swapped["screened_candidates"][0]["law"]["numerator"] = ["1"]
        probe("screened_candidate_law_swapped_under_its_result", reseal(swapped))

    if receipt.get("admissible_region", {}).get("frontier"):
        widened = clone()
        for row in widened["admissible_region"]["frontier"]:
            if row.get("bracketed"):
                row["largest_passing_coefficient"] = "1000000"
                break
        probe("widened_admissible_region", reseal(widened))

    honest = verify_receipt(receipt)
    return {
        "honest_receipt_accepted": honest["accepted"],
        "honest_problems": honest["problems"],
        "probes": probes,
        "all_probes_rejected": all(row["rejected"] for row in probes),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="write the sealed receipt here")
    parser.add_argument(
        "--verify", default=None, help="verify an existing receipt instead of building one"
    )
    arguments = parser.parse_args(argv)

    if arguments.verify:
        receipt = json.loads(Path(arguments.verify).read_text(encoding="utf-8"))
        outcome = verify_receipt(receipt)
        print(json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if outcome["accepted"] else 1

    receipt = build_receipt()
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False)
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0 if receipt["decision"] == "SCREEN_OPERATIONAL" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
