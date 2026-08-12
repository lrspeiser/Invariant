"""Small immutable expression and formula IR for Math Pack v1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

from .math_types import ExactComplex, MathType

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_OPERATIONS = {"literal", "symbol", "add", "multiply", "power", "negate", "call"}


class ExpressionIRError(ValueError):
    """Raised when an expression or formula is structurally invalid."""


@dataclass(frozen=True, slots=True)
class Expression:
    operation: str
    arguments: tuple[Expression, ...] = ()
    value: int | Fraction | float | ExactComplex | str | None = None
    math_type: MathType | None = None

    def __post_init__(self) -> None:
        if self.operation not in _OPERATIONS:
            raise ExpressionIRError(f"unsupported expression operation: {self.operation}")
        if self.operation == "literal":
            if (
                self.arguments
                or isinstance(self.value, bool)
                or not isinstance(self.value, (int, Fraction, float, ExactComplex))
            ):
                raise ExpressionIRError("literal requires one supported scalar value")
        elif self.operation == "symbol":
            if self.arguments or not isinstance(self.value, str) or not _NAME.fullmatch(self.value):
                raise ExpressionIRError("symbol requires a valid identifier")
        elif self.operation in {"add", "multiply"}:
            if self.value is not None or not self.arguments:
                raise ExpressionIRError(f"{self.operation} requires one or more arguments")
        elif self.operation == "power":
            if self.value is not None or len(self.arguments) != 2:
                raise ExpressionIRError("power requires base and exponent")
        elif self.operation == "negate":
            if self.value is not None or len(self.arguments) != 1:
                raise ExpressionIRError("negate requires one argument")
        elif self.operation == "call" and (
            not isinstance(self.value, str) or not _NAME.fullmatch(self.value)
        ):
            raise ExpressionIRError("function call requires a valid function name")

    def __add__(self, other: ExpressionLike) -> Expression:
        return add(self, other)

    def __radd__(self, other: ExpressionLike) -> Expression:
        return add(other, self)

    def __sub__(self, other: ExpressionLike) -> Expression:
        return add(self, negate(as_expression(other)))

    def __rsub__(self, other: ExpressionLike) -> Expression:
        return add(other, negate(self))

    def __mul__(self, other: ExpressionLike) -> Expression:
        return multiply(self, other)

    def __rmul__(self, other: ExpressionLike) -> Expression:
        return multiply(other, self)

    def __truediv__(self, other: ExpressionLike) -> Expression:
        return multiply(self, power(other, -1))

    def __rtruediv__(self, other: ExpressionLike) -> Expression:
        return multiply(other, power(self, -1))

    def __pow__(self, exponent: ExpressionLike) -> Expression:
        return power(self, exponent)

    def __neg__(self) -> Expression:
        return negate(self)


ExpressionLike = Expression | int | Fraction | float | ExactComplex


def literal(
    value: Fraction | float | ExactComplex, math_type: MathType | None = None
) -> Expression:
    return Expression("literal", value=value, math_type=math_type)


def symbol(name: str, math_type: MathType | None = None) -> Expression:
    return Expression("symbol", value=name, math_type=math_type)


def as_expression(value: ExpressionLike) -> Expression:
    return value if isinstance(value, Expression) else literal(value)


def add(*arguments: ExpressionLike) -> Expression:
    return Expression("add", tuple(as_expression(argument) for argument in arguments))


def multiply(*arguments: ExpressionLike) -> Expression:
    return Expression("multiply", tuple(as_expression(argument) for argument in arguments))


def power(base: ExpressionLike, exponent: ExpressionLike) -> Expression:
    return Expression("power", (as_expression(base), as_expression(exponent)))


def negate(argument: ExpressionLike) -> Expression:
    return Expression("negate", (as_expression(argument),))


def call(name: str, *arguments: ExpressionLike, math_type: MathType | None = None) -> Expression:
    return Expression(
        "call",
        tuple(as_expression(argument) for argument in arguments),
        value=name,
        math_type=math_type,
    )


class Formula:
    """Marker base for immutable formula nodes."""


@dataclass(frozen=True, slots=True)
class Equation(Formula):
    left: Expression
    right: Expression

    def __post_init__(self) -> None:
        if not isinstance(self.left, Expression) or not isinstance(self.right, Expression):
            raise ExpressionIRError("equation sides must be expressions")


class InequalityRelation(str, Enum):
    LESS = "<"
    LESS_EQUAL = "<="
    GREATER = ">"
    GREATER_EQUAL = ">="
    NOT_EQUAL = "!="

    def flipped(self) -> InequalityRelation:
        return {
            self.LESS: self.GREATER,
            self.LESS_EQUAL: self.GREATER_EQUAL,
            self.GREATER: self.LESS,
            self.GREATER_EQUAL: self.LESS_EQUAL,
            self.NOT_EQUAL: self.NOT_EQUAL,
        }[self]


@dataclass(frozen=True, slots=True)
class Inequality(Formula):
    left: Expression
    right: Expression
    relation: InequalityRelation

    def __post_init__(self) -> None:
        if not isinstance(self.left, Expression) or not isinstance(self.right, Expression):
            raise ExpressionIRError("inequality sides must be expressions")
        if not isinstance(self.relation, InequalityRelation):
            raise ExpressionIRError("inequality relation must be an InequalityRelation")


@dataclass(frozen=True, slots=True)
class Recurrence(Formula):
    sequence: str
    index: Expression
    order: int
    equation: Equation
    initial_conditions: tuple[tuple[int, Expression], ...] = ()

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.sequence):
            raise ExpressionIRError("recurrence sequence must be a valid identifier")
        if self.index.operation != "symbol":
            raise ExpressionIRError("recurrence index must be a symbol")
        if self.order < 1:
            raise ExpressionIRError("recurrence order must be positive")
        if not isinstance(self.equation, Equation):
            raise ExpressionIRError("recurrence body must be an equation")
        if any(not isinstance(value, Expression) for _, value in self.initial_conditions):
            raise ExpressionIRError("recurrence initial values must be expressions")
        indices = [index for index, _ in self.initial_conditions]
        if len(indices) != len(set(indices)):
            raise ExpressionIRError("recurrence initial-condition indices must be unique")


def _literal_data(value: Any) -> dict[str, Any]:
    if isinstance(value, Fraction):
        return {"kind": "rational", "numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, ExactComplex):
        return {
            "kind": "complex",
            "real": _literal_data(value.real),
            "imaginary": _literal_data(value.imaginary),
        }
    if isinstance(value, float):
        return {"kind": "real", "value": value.hex()}
    return {"kind": "integer", "value": value}


def expression_to_data(expression: Expression) -> dict[str, Any]:
    result: dict[str, Any] = {
        "operation": expression.operation,
        "arguments": [expression_to_data(argument) for argument in expression.arguments],
    }
    if expression.operation == "literal":
        result["value"] = _literal_data(expression.value)
    elif expression.value is not None:
        result["value"] = expression.value
    if expression.math_type is not None:
        result["math_type"] = repr(expression.math_type)
    return result


def formula_to_data(formula: Formula) -> dict[str, Any]:
    if isinstance(formula, Equation):
        return {
            "kind": "equation",
            "left": expression_to_data(formula.left),
            "right": expression_to_data(formula.right),
        }
    if isinstance(formula, Inequality):
        return {
            "kind": "inequality",
            "left": expression_to_data(formula.left),
            "right": expression_to_data(formula.right),
            "relation": formula.relation.value,
        }
    if isinstance(formula, Recurrence):
        return {
            "kind": "recurrence",
            "sequence": formula.sequence,
            "index": expression_to_data(formula.index),
            "order": formula.order,
            "equation": formula_to_data(formula.equation),
            "initial_conditions": [
                {"index": index, "value": expression_to_data(value)}
                for index, value in formula.initial_conditions
            ],
        }
    raise TypeError(f"unsupported formula node: {type(formula).__name__}")
