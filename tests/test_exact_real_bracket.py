"""The bracketing engine has one job: never lie about where a real number is.

Every test here checks containment against an independently computed high-precision value, or
checks that the engine refuses rather than guesses.  ``mpmath`` appears only in the tests -- it
is the independent oracle, never the certificate path.
"""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import mpmath as mp
import pytest
import sympy as sp

from sigma_theory_compiler.exact_real_bracket import (
    EQUAL,
    GREATER,
    LESS,
    UNSEPARATED,
    Bracket,
    ExactBracketError,
    arctan_reciprocal_bracket,
    bracket_expression,
    compare_expressions,
    exp_bracket,
    log_bracket,
    pi_bracket,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs" / "physics-language" / "quartic-global-h7-energy-campaign" / "campaign.json"


def _oracle(value: Fraction) -> mp.mpf:
    """A rational rendered at whatever working precision the calling test has set."""

    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _encloses(bracket: Bracket, truth: mp.mpf) -> bool:
    return _oracle(bracket.lo) <= truth <= _oracle(bracket.hi)


def test_endpoints_are_exact_rationals_and_no_float_appears() -> None:
    bracket = bracket_expression("sqrt(2) + pi/3", 128)
    assert isinstance(bracket.lo, Fraction)
    assert isinstance(bracket.hi, Fraction)
    assert not isinstance(bracket.lo, float)
    assert bracket.lo < bracket.hi


@pytest.mark.parametrize("bits", [16, 64, 256, 512])
def test_pi_bracket_encloses_pi_at_every_precision(bits: int) -> None:
    mp.mp.dps = 400
    assert _encloses(pi_bracket(bits), mp.pi)


@pytest.mark.parametrize("bits", [16, 64, 256])
@pytest.mark.parametrize("argument", ["-8", "8", "0", "3/7", "-11/5"])
def test_exp_bracket_encloses_exp(bits: int, argument: str) -> None:
    mp.mp.dps = 400
    value = Fraction(argument)
    truth = mp.e ** (mp.mpf(value.numerator) / mp.mpf(value.denominator))
    assert _encloses(exp_bracket(value, bits), truth)


@pytest.mark.parametrize("bits", [16, 64, 256])
@pytest.mark.parametrize("argument", ["2", "4", "1", "1/7", "1000", "1024", "3/2"])
def test_log_bracket_encloses_log(bits: int, argument: str) -> None:
    mp.mp.dps = 400
    value = Fraction(argument)
    truth = mp.log(mp.mpf(value.numerator) / mp.mpf(value.denominator))
    assert _encloses(log_bracket(value, bits), truth)


def test_bracket_width_shrinks_as_the_budget_grows() -> None:
    expression = "sqrt(8640*sqrt(3) + 2304*sqrt(10645) + 512*sqrt(249510) + 254406007)"
    widths = [bracket_expression(expression, bits).width for bits in (32, 64, 128, 256)]
    assert all(later < earlier for earlier, later in pairwise(widths))


@pytest.mark.parametrize(
    "expression",
    [
        "1",
        "-3/4",
        "sqrt(2)",
        "sqrt(3)/7 - sqrt(5)/11",
        "pi**2",
        "1/sqrt(pi)",
        "exp(-8)*sqrt(2)",
        "(sqrt(2) + sqrt(3))**2",
        "sqrt(8640*sqrt(3) + 2304*sqrt(10645) + 512*sqrt(249510) + 254406007)",
        "(1 + sqrt(pi))**(-2)",
        "2*log(2)/sqrt(7)",
    ],
)
def test_bracket_encloses_an_independently_evaluated_value(expression: str) -> None:
    mp.mp.dps = 120
    truth = mp.mpf(str(sp.N(sp.sympify(expression), 100)))
    assert _encloses(bracket_expression(expression, 256), truth)


def test_alternating_arctan_series_brackets_the_limit() -> None:
    mp.mp.dps = 200
    for inverse in (5, 239, 2):
        bracket = arctan_reciprocal_bracket(inverse, 128)
        assert _encloses(bracket, mp.atan(mp.mpf(1) / inverse))


def test_unsupported_nodes_are_refused_rather_than_approximated() -> None:
    with pytest.raises(ExactBracketError):
        bracket_expression("sin(1)", 64)
    with pytest.raises(ExactBracketError):
        bracket_expression("x + 1", 64)
    with pytest.raises(ExactBracketError):
        bracket_expression("2**(1/3)", 64)
    with pytest.raises(ExactBracketError):
        bracket_expression("exp(sqrt(2))", 64)
    with pytest.raises(ExactBracketError):
        bracket_expression("sqrt(2)", 4)


def test_a_bracket_straddling_zero_cannot_be_inverted() -> None:
    # sqrt(2) and this decimal differ by about 5e-17, so a 16-bit bracket of the difference
    # straddles zero and inverting it would be a fabrication.  It has to raise instead.
    near_zero = "1/(sqrt(2) - 14142135623730951/10000000000000000)"
    with pytest.raises(ExactBracketError):
        bracket_expression(near_zero, 16)
    # With a budget that resolves the difference the same expression brackets fine.
    assert bracket_expression(near_zero, 256).hi < 0


def test_empty_brackets_are_rejected() -> None:
    with pytest.raises(ExactBracketError):
        Bracket(Fraction(3), Fraction(2))


def test_equality_is_claimed_only_for_the_same_object() -> None:
    assert compare_expressions("sqrt(2)", "sqrt(2)").verdict == EQUAL


def test_equal_values_in_different_forms_are_never_given_an_order() -> None:
    # sqrt(2)+sqrt(3) and sqrt(5+2*sqrt(6)) are the same number, and sympy does not fold one
    # into the other.  No bracket can prove them equal, so the only sound answer is to refuse.
    left, right = "sqrt(2) + sqrt(3)", "sqrt(5 + 2*sqrt(6))"
    assert sp.sympify(left) != sp.sympify(right)
    for ladder in ((64,), (64, 256), (64, 256, 1024)):
        assert compare_expressions(left, right, ladder).verdict == UNSEPARATED
        assert compare_expressions(right, left, ladder).verdict == UNSEPARATED


def test_strict_order_is_witnessed_by_two_separating_rationals() -> None:
    comparison = compare_expressions("sqrt(2)", "sqrt(3)")
    assert comparison.verdict == LESS
    assert comparison.left.hi < comparison.right.lo
    receipt = comparison.as_receipt()
    assert Fraction(receipt["left_bracket"]["hi"]) < Fraction(receipt["right_bracket"]["lo"])
    assert comparison.separated()
    assert compare_expressions("sqrt(3)", "sqrt(2)").verdict == GREATER


def test_a_starved_ladder_reports_unseparated_instead_of_guessing() -> None:
    # These differ in the 13th significant digit; 16 bits cannot see it and must say so.
    close = ("10000000000000", "10000000000001")
    assert compare_expressions(*close, ladder=(16,)).verdict == UNSEPARATED
    assert compare_expressions(*close, ladder=(16, 64, 128)).verdict == LESS


def test_certificate_constants_are_bracketed_and_ordered_exactly() -> None:
    mp.mp.dps = 120
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    growths = {
        certificate["candidate_id"]: certificate["strongest_global_differential_inequality"][
            "A_known"
        ]
        for certificate in campaign["certificates"]
    }
    distinct = sorted({expression for expression in growths.values()})
    assert len(distinct) == 4
    for expression in distinct:
        truth = mp.mpf(str(sp.N(sp.sympify(expression), 100)))
        assert _encloses(bracket_expression(expression, 256), truth)
    ordered = sorted(distinct, key=lambda item: mp.mpf(str(sp.N(sp.sympify(item), 60))))
    for smaller, larger in pairwise(ordered):
        assert compare_expressions(smaller, larger).verdict == LESS
