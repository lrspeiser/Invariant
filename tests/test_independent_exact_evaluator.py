from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.independent_exact_evaluator import (
    IndependentEvaluationError,
    evaluate_expression,
    evaluate_recurrence,
)


def test_exact_expression_uses_only_fraction_arithmetic() -> None:
    assert evaluate_expression(
        "x0*(x0+1)*(2*x0+1)/6", {"x0": Fraction(9)}
    ) == Fraction(285)
    assert evaluate_expression(
        "(1/2)*x0*x1**2", {"x0": Fraction(7), "x1": Fraction(3, 2)}
    ) == Fraction(63, 8)


def test_recurrence_replays_in_a_separate_implementation() -> None:
    assert evaluate_recurrence(
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
        (0, 1, 2, 3, 4, 5),
    ) == tuple(Fraction(item) for item in (0, 1, 1, 2, 3, 5))


@pytest.mark.parametrize(
    "expression",
    (
        "__import__('os')",
        "x0.attr",
        "x0[0]",
        "[x0 for x0 in (1,)]",
        "x0**9",
        "1/0",
    ),
)
def test_independent_grammar_fails_closed(expression: str) -> None:
    with pytest.raises(IndependentEvaluationError):
        evaluate_expression(expression, {"x0": Fraction(1)})
