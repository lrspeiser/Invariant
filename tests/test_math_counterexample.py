from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.math_counterexample import (
    SearchStatus,
    SearchStrategy,
    UnsupportedEvaluation,
    evaluate_formula,
    find_counterexample,
)
from sigma_theory_compiler.math_expression_ir import (
    Equation,
    Inequality,
    InequalityRelation,
    Recurrence,
    call,
    literal,
    symbol,
)
from sigma_theory_compiler.math_types import INTEGER, REAL, SequenceType


def test_exact_assignment_returns_a_structured_counterexample() -> None:
    x = symbol("x", INTEGER)
    report = find_counterexample(
        Equation(x**2, literal(1)),
        {"x": INTEGER},
        exact_assignments=({"x": 1}, {"x": 2}),
        strategies=(SearchStrategy.EXACT,),
        seed=17,
    )

    assert report.status is SearchStatus.COUNTEREXAMPLE_FOUND
    assert report.counterexample is not None
    assert report.counterexample.assignment == (("x", 2),)
    assert report.counterexample.strategy is SearchStrategy.EXACT
    assert report.proves_formula is False


def test_adversarial_and_random_schedules_are_deterministic() -> None:
    x = symbol("x", REAL)
    formula = Equation(call("sqrt", x**2), x)
    adversarial = find_counterexample(
        formula,
        {"x": REAL},
        strategies=(SearchStrategy.ADVERSARIAL,),
        seed=9,
    )
    first = find_counterexample(
        Inequality(x, literal(100), InequalityRelation.LESS),
        {"x": REAL},
        strategies=(SearchStrategy.RANDOM,),
        random_trials=12,
        seed=12345,
    )
    second = find_counterexample(
        Inequality(x, literal(100), InequalityRelation.LESS),
        {"x": REAL},
        strategies=(SearchStrategy.RANDOM,),
        random_trials=12,
        seed=12345,
    )

    assert adversarial.status is SearchStatus.COUNTEREXAMPLE_FOUND
    assert adversarial.counterexample is not None
    assert adversarial.counterexample.assignment == (("x", Fraction(-1)),)
    assert first == second
    assert first.status is SearchStatus.INCONCLUSIVE_WITHIN_BUDGET
    assert first.proves_formula is False


def test_recurrence_uses_a_finite_function_table() -> None:
    n = symbol("n", INTEGER)
    recurrence = Recurrence(
        "a",
        n,
        1,
        Equation(call("a", n + 1), call("a", n) + 1),
        ((0, literal(0)),),
    )
    assert evaluate_formula(recurrence, {"n": 1, "a": {0: 0, 1: 1, 2: 2}})
    assert not evaluate_formula(recurrence, {"n": 1, "a": {0: 0, 1: 1, 2: 3}})


def test_unsupported_cases_fail_closed_instead_of_claiming_success() -> None:
    x = symbol("x")
    with pytest.raises(UnsupportedEvaluation, match="finite function table"):
        evaluate_formula(Equation(call("mystery", x), literal(0)), {"x": 0})

    report = find_counterexample(
        Equation(x, x),
        {"x": SequenceType(INTEGER)},
        strategies=(SearchStrategy.ADVERSARIAL, SearchStrategy.RANDOM),
        seed=2,
    )
    assert report.status is SearchStatus.UNSUPPORTED
    assert report.counterexample is None
    assert report.unsupported_reason
    assert report.proves_formula is False
