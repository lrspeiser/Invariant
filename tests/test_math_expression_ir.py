from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.math_expression_ir import (
    Equation,
    ExpressionIRError,
    Inequality,
    InequalityRelation,
    Recurrence,
    call,
    expression_to_data,
    formula_to_data,
    literal,
    symbol,
)
from sigma_theory_compiler.math_types import INTEGER


def test_expression_builders_create_immutable_typed_ir() -> None:
    x = symbol("x", INTEGER)
    expression = (x + Fraction(1, 2)) * (x - 2) ** 2
    data = expression_to_data(expression)

    assert expression.operation == "multiply"
    assert data["arguments"][0]["operation"] == "add"
    assert data["arguments"][0]["arguments"][0]["math_type"] == repr(INTEGER)
    assert data["arguments"][0]["arguments"][1]["value"] == {
        "kind": "rational",
        "numerator": 1,
        "denominator": 2,
    }


def test_equation_inequality_and_recurrence_are_serializable_formulas() -> None:
    n = symbol("n", INTEGER)
    recurrence = Recurrence(
        sequence="a",
        index=n,
        order=2,
        equation=Equation(call("a", n + 2), call("a", n + 1) + call("a", n)),
        initial_conditions=((0, literal(0)), (1, literal(1))),
    )
    recurrence_data = formula_to_data(recurrence)
    inequality_data = formula_to_data(Inequality(n, literal(0), InequalityRelation.GREATER_EQUAL))

    assert recurrence_data["kind"] == "recurrence"
    assert recurrence_data["order"] == 2
    assert recurrence_data["equation"]["kind"] == "equation"
    assert inequality_data["relation"] == ">="


def test_invalid_ir_fails_at_construction() -> None:
    with pytest.raises(ExpressionIRError):
        symbol("not a name")
    with pytest.raises(ExpressionIRError):
        Equation(literal(0), 0)  # type: ignore[arg-type]
    with pytest.raises(ExpressionIRError):
        Inequality(literal(0), literal(1), "<")  # type: ignore[arg-type]
    with pytest.raises(ExpressionIRError):
        Recurrence("a", literal(0), 1, Equation(literal(0), literal(0)))
    with pytest.raises(ExpressionIRError):
        Recurrence(
            "a",
            symbol("n"),
            1,
            Equation(literal(0), literal(0)),
            ((0, literal(1)), (0, literal(2))),
        )
