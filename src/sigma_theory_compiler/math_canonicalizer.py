"""Deterministic exact-algebra canonicalization for Math Pack v1 IR."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

import sympy as sp

from .math_expression_ir import (
    Equation,
    Expression,
    Formula,
    Inequality,
    Recurrence,
    add,
    call,
    expression_to_data,
    formula_to_data,
    literal,
    multiply,
    power,
    symbol,
)
from .math_types import ExactComplex

_KNOWN_FUNCTIONS: dict[str, Any] = {
    "abs": sp.Abs,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "sqrt": sp.sqrt,
}


class CanonicalizationError(ValueError):
    """Raised when exact canonicalization cannot safely represent an input."""


def to_sympy(expression: Expression) -> sp.Expr:
    """Translate supported IR to a SymPy expression without parsing source text."""

    if expression.operation == "literal":
        value = expression.value
        if isinstance(value, (bool, float)):
            raise CanonicalizationError("floating-point literals are not exact algebraic inputs")
        if isinstance(value, int):
            return sp.Integer(value)
        if isinstance(value, Fraction):
            return sp.Rational(value.numerator, value.denominator)
        if isinstance(value, ExactComplex):
            return sp.Rational(value.real.numerator, value.real.denominator) + sp.I * sp.Rational(
                value.imaginary.numerator, value.imaginary.denominator
            )
        raise CanonicalizationError(f"unsupported literal: {value!r}")
    if expression.operation == "symbol":
        return sp.Symbol(str(expression.value))
    arguments = tuple(to_sympy(argument) for argument in expression.arguments)
    if expression.operation == "add":
        return sp.Add(*arguments)
    if expression.operation == "multiply":
        return sp.Mul(*arguments)
    if expression.operation == "power":
        return sp.Pow(*arguments)
    if expression.operation == "negate":
        return -arguments[0]
    if expression.operation == "call":
        function = _KNOWN_FUNCTIONS.get(str(expression.value), sp.Function(str(expression.value)))
        return function(*arguments)
    raise CanonicalizationError(f"unsupported operation: {expression.operation}")


def _symbol_types(expression: Expression) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if expression.operation == "symbol" and expression.math_type is not None:
        found[str(expression.value)] = expression.math_type
    for argument in expression.arguments:
        for name, math_type in _symbol_types(argument).items():
            if name in found and found[name] != math_type:
                raise CanonicalizationError(f"symbol {name!r} has conflicting mathematical types")
            found[name] = math_type
    return found


def _from_sympy(value: sp.Expr, symbol_types: dict[str, Any]) -> Expression:
    if value.is_Integer:
        return literal(int(value))
    if value.is_Rational:
        return literal(Fraction(int(value.p), int(value.q)))
    if value.is_number and not value.free_symbols:
        real, imaginary = value.as_real_imag()
        if real.is_Rational and imaginary.is_Rational:
            return literal(
                ExactComplex(
                    Fraction(int(real.p), int(real.q)),
                    Fraction(int(imaginary.p), int(imaginary.q)),
                )
            )
        raise CanonicalizationError(f"non-rational numeric constant is unsupported: {value}")
    if value.is_Symbol:
        name = str(value)
        return symbol(name, symbol_types.get(name))
    if value.is_Add:
        return add(*(_from_sympy(argument, symbol_types) for argument in value.args))
    if value.is_Mul:
        return multiply(*(_from_sympy(argument, symbol_types) for argument in value.args))
    if value.is_Pow:
        return power(*(_from_sympy(argument, symbol_types) for argument in value.args))
    if value.is_Function:
        return call(
            value.func.__name__,
            *(_from_sympy(argument, symbol_types) for argument in value.args),
        )
    raise CanonicalizationError(f"SymPy node cannot be represented in Math Pack IR: {value}")


def _canonical_sympy(expression: Expression) -> sp.Expr:
    value = to_sympy(expression)
    numerator, denominator = sp.fraction(sp.cancel(sp.together(value)))
    numerator = sp.expand(numerator)
    denominator = sp.expand(denominator)
    return sp.cancel(numerator / denominator)


def canonicalize_expression(expression: Expression) -> Expression:
    """Canonicalize an expression in the exact rational-function algebra."""

    return _from_sympy(_canonical_sympy(expression), _symbol_types(expression))


def _monic_zero_expression(value: sp.Expr) -> sp.Expr:
    numerator, denominator = sp.fraction(sp.cancel(value))
    if numerator == 0:
        return sp.Integer(0)
    variables = sorted(numerator.free_symbols, key=sp.default_sort_key)
    try:
        polynomial = sp.Poly(numerator, *variables)
    except sp.PolynomialError:
        return value
    if polynomial.is_zero:
        return sp.Integer(0)
    leading = polynomial.LC()
    if not leading.is_Rational:
        return value
    return sp.cancel((numerator / leading) / denominator)


def canonicalize_formula(formula: Formula) -> Formula:
    """Canonicalize equations, inequalities, and recurrence relations."""

    if isinstance(formula, Equation):
        symbol_types = {**_symbol_types(formula.left), **_symbol_types(formula.right)}
        difference = _monic_zero_expression(_canonical_sympy(formula.left - formula.right))
        return Equation(_from_sympy(difference, symbol_types), literal(0))
    if isinstance(formula, Inequality):
        symbol_types = {**_symbol_types(formula.left), **_symbol_types(formula.right)}
        difference = _canonical_sympy(formula.left - formula.right)
        relation = formula.relation
        if difference.could_extract_minus_sign():
            difference = -difference
            relation = relation.flipped()
        return Inequality(_from_sympy(difference, symbol_types), literal(0), relation)
    if isinstance(formula, Recurrence):
        equation = canonicalize_formula(formula.equation)
        if not isinstance(equation, Equation):
            raise TypeError("recurrence equation canonicalized to a non-equation")
        return Recurrence(
            sequence=formula.sequence,
            index=formula.index,
            order=formula.order,
            equation=equation,
            initial_conditions=tuple(
                (index, canonicalize_expression(value))
                for index, value in sorted(formula.initial_conditions)
            ),
        )
    raise CanonicalizationError(f"unsupported formula: {type(formula).__name__}")


def canonical_data(value: Expression | Formula) -> dict[str, Any]:
    canonical = (
        canonicalize_expression(value)
        if isinstance(value, Expression)
        else canonicalize_formula(value)
    )
    return (
        expression_to_data(canonical)
        if isinstance(canonical, Expression)
        else formula_to_data(canonical)
    )


def canonical_sha256(value: Expression | Formula) -> str:
    encoded = json.dumps(canonical_data(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
