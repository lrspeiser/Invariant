from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.independent_exact_evaluator import (
    IndependentEvaluationError,
    evaluate_comparison,
    evaluate_expression,
    evaluate_recurrence,
)


def test_exact_expression_uses_only_fraction_arithmetic() -> None:
    assert evaluate_expression("x0*(x0+1)*(2*x0+1)/6", {"x0": Fraction(9)}) == Fraction(285)
    assert evaluate_expression(
        "(1/2)*x0*x1**2", {"x0": Fraction(7), "x1": Fraction(3, 2)}
    ) == Fraction(63, 8)


def test_recurrence_replays_in_a_separate_implementation() -> None:
    assert evaluate_recurrence(
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1)),
        (0, 1, 2, 3, 4, 5),
    ) == tuple(Fraction(item) for item in (0, 1, 1, 2, 3, 5))


def test_exact_comparison_uses_fraction_semantics() -> None:
    variables = {"x0": Fraction(1, 3)}
    assert evaluate_comparison("3*x0", "eq", "1", variables)
    assert evaluate_comparison("x0", "lt", "1/2", variables)
    assert not evaluate_comparison("x0", "ge", "1/2", variables)
    with pytest.raises(IndependentEvaluationError, match="operator"):
        evaluate_comparison("x0", "approximately", "1/2", variables)


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
