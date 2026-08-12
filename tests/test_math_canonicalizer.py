from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.math_canonicalizer import (
    CanonicalizationError,
    canonical_data,
    canonical_sha256,
    canonicalize_expression,
    canonicalize_formula,
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


def test_exact_rational_function_canonicalization_is_stable() -> None:
    x = symbol("x")
    first = (x**2 - 1) / (x - 1)
    second = x + 1

    assert canonical_data(first) == canonical_data(second)
    assert canonical_sha256(first) == canonical_sha256(second)
    assert canonicalize_expression(x + x) == canonicalize_expression(2 * x)


def test_equations_normalize_nonzero_rational_scalar_multiples() -> None:
    x = symbol("x")
    first = Equation(2 * x + 2, literal(0))
    second = Equation(-3 * x - 3, literal(0))

    assert canonicalize_formula(first) == canonicalize_formula(second)


def test_equation_that_reduces_identically_to_zero_canonicalizes() -> None:
    n = symbol("n")
    equation = Equation(n * (n + 1) / 2, (n**2 + n) / 2)

    assert canonicalize_formula(equation) == Equation(literal(0), literal(0))
    assert len(canonical_sha256(equation)) == 64


def test_inequality_sign_and_recurrence_initial_order_are_canonical() -> None:
    x = symbol("x")
    inequality = Inequality(-x, literal(0), InequalityRelation.LESS)
    assert canonicalize_formula(inequality) == Inequality(x, literal(0), InequalityRelation.GREATER)

    n = symbol("n")
    recurrence = Recurrence(
        "a",
        n,
        1,
        Equation(call("a", n + 1), call("a", n) + Fraction(2, 2)),
        ((1, literal(2)), (0, literal(1))),
    )
    canonical = canonicalize_formula(recurrence)
    assert isinstance(canonical, Recurrence)
    assert [index for index, _ in canonical.initial_conditions] == [0, 1]
    assert canonical.equation == canonicalize_formula(Equation(call("a", n + 1), call("a", n) + 1))


def test_floating_point_inputs_fail_closed() -> None:
    with pytest.raises(CanonicalizationError, match="floating-point"):
        canonicalize_expression(literal(0.1))
