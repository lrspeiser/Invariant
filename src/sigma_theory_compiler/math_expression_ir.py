"""Small immutable expression and formula IR for Math Pack v1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import prod
from typing import Any

from .math_types import ExactComplex, MathType

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_OPERATIONS = {"literal", "symbol", "add", "multiply", "power", "negate", "call"}
_MAX_EXPRESSION_NODES = 512
_MAX_FINITE_TERMS = 64
_MAX_PIECEWISE_BRANCHES = 8
_MAX_TENSOR_COMPONENTS = 256


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


def _expression_nodes(expression: Expression) -> int:
    return 1 + sum(_expression_nodes(argument) for argument in expression.arguments)


def _expression_symbols(expression: Expression) -> set[str]:
    found = {str(expression.value)} if expression.operation == "symbol" else set()
    for argument in expression.arguments:
        found.update(_expression_symbols(argument))
    return found


def _bounded_expression(expression: Expression, label: str) -> None:
    if not isinstance(expression, Expression):
        raise ExpressionIRError(f"{label} must be an expression")
    if _expression_nodes(expression) > _MAX_EXPRESSION_NODES:
        raise ExpressionIRError(f"{label} exceeds the expression-node budget")


def _index_symbol(expression: Expression, label: str) -> None:
    _bounded_expression(expression, label)
    if expression.operation != "symbol":
        raise ExpressionIRError(f"{label} must be a symbol")


def _bounded_range(lower: int, upper: int, label: str) -> None:
    if (
        isinstance(lower, bool)
        or not isinstance(lower, int)
        or isinstance(upper, bool)
        or not isinstance(upper, int)
        or lower > upper
        or upper - lower + 1 > _MAX_FINITE_TERMS
    ):
        raise ExpressionIRError(f"{label} is invalid or exceeds the finite-term budget")


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
        _bounded_expression(self.equation.left, "recurrence left side")
        _bounded_expression(self.equation.right, "recurrence right side")
        if any(not isinstance(value, Expression) for _, value in self.initial_conditions):
            raise ExpressionIRError("recurrence initial values must be expressions")
        for _, value in self.initial_conditions:
            _bounded_expression(value, "recurrence initial value")
        indices = [index for index, _ in self.initial_conditions]
        if len(indices) != len(set(indices)):
            raise ExpressionIRError("recurrence initial-condition indices must be unique")


@dataclass(frozen=True, slots=True)
class FiniteSum(Formula):
    index: Expression
    lower: int
    upper: int
    summand: Expression
    claimed_value: Expression

    def __post_init__(self) -> None:
        _index_symbol(self.index, "finite-sum index")
        _bounded_range(self.lower, self.upper, "finite-sum range")
        _bounded_expression(self.summand, "finite-sum summand")
        _bounded_expression(self.claimed_value, "finite-sum claimed value")
        if str(self.index.value) in _expression_symbols(self.claimed_value):
            raise ExpressionIRError("finite-sum claimed value contains its bound index")


@dataclass(frozen=True, slots=True)
class FiniteProduct(Formula):
    index: Expression
    lower: int
    upper: int
    factor: Expression
    claimed_value: Expression

    def __post_init__(self) -> None:
        _index_symbol(self.index, "finite-product index")
        _bounded_range(self.lower, self.upper, "finite-product range")
        _bounded_expression(self.factor, "finite-product factor")
        _bounded_expression(self.claimed_value, "finite-product claimed value")
        if str(self.index.value) in _expression_symbols(self.claimed_value):
            raise ExpressionIRError("finite-product claimed value contains its bound index")


@dataclass(frozen=True, slots=True)
class GeneratingFunction(Formula):
    sequence: str
    variable: Expression
    coefficients: tuple[Expression, ...]
    claimed_value: Expression

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, str) or _NAME.fullmatch(self.sequence) is None:
            raise ExpressionIRError("generating-function sequence must be a valid identifier")
        _index_symbol(self.variable, "generating-function variable")
        if not 1 <= len(self.coefficients) <= _MAX_FINITE_TERMS:
            raise ExpressionIRError("generating-function truncation exceeds the term budget")
        for coefficient in self.coefficients:
            _bounded_expression(coefficient, "generating-function coefficient")
            if str(self.variable.value) in _expression_symbols(coefficient):
                raise ExpressionIRError(
                    "generating-function coefficient contains the series variable"
                )
        _bounded_expression(self.claimed_value, "generating-function claimed value")


@dataclass(frozen=True, slots=True)
class ModularRelation(Formula):
    left: Expression
    right: Expression
    modulus: int

    def __post_init__(self) -> None:
        _bounded_expression(self.left, "modular left side")
        _bounded_expression(self.right, "modular right side")
        if (
            isinstance(self.modulus, bool)
            or not isinstance(self.modulus, int)
            or not 2 <= self.modulus <= 2**31 - 1
        ):
            raise ExpressionIRError("modulus must be an integer in [2, 2^31-1]")


class PiecewiseComparator(str, Enum):
    LESS = "lt"
    LESS_EQUAL = "le"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    GREATER_EQUAL = "ge"
    GREATER = "gt"


@dataclass(frozen=True, slots=True)
class PiecewiseBranch:
    left: Expression
    comparator: PiecewiseComparator
    right: Expression
    expression: Expression

    def __post_init__(self) -> None:
        _bounded_expression(self.left, "piecewise left predicate operand")
        _bounded_expression(self.right, "piecewise right predicate operand")
        _bounded_expression(self.expression, "piecewise branch expression")
        if not isinstance(self.comparator, PiecewiseComparator):
            raise ExpressionIRError("piecewise comparator must be a PiecewiseComparator")


@dataclass(frozen=True, slots=True)
class PiecewiseRelation(Formula):
    branches: tuple[PiecewiseBranch, ...]
    default_expression: Expression
    claimed_value: Expression

    def __post_init__(self) -> None:
        if (
            not isinstance(self.branches, tuple)
            or not 1 <= len(self.branches) <= _MAX_PIECEWISE_BRANCHES
            or any(not isinstance(branch, PiecewiseBranch) for branch in self.branches)
        ):
            raise ExpressionIRError("piecewise branches are invalid or outside the branch budget")
        _bounded_expression(self.default_expression, "piecewise default expression")
        _bounded_expression(self.claimed_value, "piecewise claimed value")


@dataclass(frozen=True, slots=True)
class TensorIdentity(Formula):
    tensor_name: str
    shape: tuple[int, ...]
    variance: tuple[str, ...]
    left_components: tuple[Expression, ...]
    right_components: tuple[Expression, ...]
    symmetries: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tensor_name, str) or _NAME.fullmatch(self.tensor_name) is None:
            raise ExpressionIRError("tensor name must be a valid identifier")
        if (
            not 1 <= len(self.shape) <= 4
            or any(isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 8 for size in self.shape)
            or prod(self.shape) > _MAX_TENSOR_COMPONENTS
        ):
            raise ExpressionIRError("tensor shape is invalid or exceeds the component budget")
        if len(self.variance) != len(self.shape) or any(
            item not in {"covariant", "contravariant"} for item in self.variance
        ):
            raise ExpressionIRError("tensor variance does not match its rank")
        component_count = prod(self.shape)
        if (
            len(self.left_components) != component_count
            or len(self.right_components) != component_count
        ):
            raise ExpressionIRError("tensor component coverage does not match its shape")
        for component in (*self.left_components, *self.right_components):
            _bounded_expression(component, "tensor component")
        if self.symmetries != tuple(sorted(set(self.symmetries))):
            raise ExpressionIRError("tensor symmetries must be sorted and unique")
        for left_axis, right_axis, sign in self.symmetries:
            if (
                isinstance(left_axis, bool)
                or isinstance(right_axis, bool)
                or not isinstance(left_axis, int)
                or not isinstance(right_axis, int)
                or not 0 <= left_axis < right_axis < len(self.shape)
                or self.shape[left_axis] != self.shape[right_axis]
                or isinstance(sign, bool)
                or sign not in {-1, 1}
            ):
                raise ExpressionIRError("tensor symmetry is invalid")


@dataclass(frozen=True, slots=True)
class VariationalFunctional(Formula):
    field: str
    coordinate: str
    first_derivative: str
    second_derivative: str
    integrand: Expression
    claimed_euler_lagrange: Expression

    def __post_init__(self) -> None:
        names = (self.field, self.coordinate, self.first_derivative, self.second_derivative)
        if any(not isinstance(name, str) or _NAME.fullmatch(name) is None for name in names):
            raise ExpressionIRError("variational symbols must be valid identifiers")
        if len(set(names)) != len(names):
            raise ExpressionIRError("variational symbols must be distinct")
        _bounded_expression(self.integrand, "variational integrand")
        _bounded_expression(self.claimed_euler_lagrange, "Euler-Lagrange claim")
        if self.second_derivative in _expression_symbols(self.integrand):
            raise ExpressionIRError(
                "first-order variational integrand contains the second derivative"
            )


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
    if isinstance(formula, FiniteSum):
        return {
            "kind": "finite_sum",
            "index": expression_to_data(formula.index),
            "lower": formula.lower,
            "upper": formula.upper,
            "summand": expression_to_data(formula.summand),
            "claimed_value": expression_to_data(formula.claimed_value),
        }
    if isinstance(formula, FiniteProduct):
        return {
            "kind": "finite_product",
            "index": expression_to_data(formula.index),
            "lower": formula.lower,
            "upper": formula.upper,
            "factor": expression_to_data(formula.factor),
            "claimed_value": expression_to_data(formula.claimed_value),
        }
    if isinstance(formula, GeneratingFunction):
        return {
            "kind": "generating_function",
            "sequence": formula.sequence,
            "variable": expression_to_data(formula.variable),
            "coefficients": [expression_to_data(item) for item in formula.coefficients],
            "claimed_value": expression_to_data(formula.claimed_value),
        }
    if isinstance(formula, ModularRelation):
        return {
            "kind": "modular_relation",
            "left": expression_to_data(formula.left),
            "right": expression_to_data(formula.right),
            "modulus": formula.modulus,
        }
    if isinstance(formula, PiecewiseRelation):
        return {
            "kind": "piecewise_relation",
            "branches": [
                {
                    "left": expression_to_data(branch.left),
                    "comparator": branch.comparator.value,
                    "right": expression_to_data(branch.right),
                    "expression": expression_to_data(branch.expression),
                }
                for branch in formula.branches
            ],
            "default_expression": expression_to_data(formula.default_expression),
            "claimed_value": expression_to_data(formula.claimed_value),
        }
    if isinstance(formula, TensorIdentity):
        return {
            "kind": "tensor_identity",
            "tensor_name": formula.tensor_name,
            "shape": list(formula.shape),
            "variance": list(formula.variance),
            "left_components": [expression_to_data(item) for item in formula.left_components],
            "right_components": [expression_to_data(item) for item in formula.right_components],
            "symmetries": [
                {"left_axis": left, "right_axis": right, "sign": sign}
                for left, right, sign in formula.symmetries
            ],
        }
    if isinstance(formula, VariationalFunctional):
        return {
            "kind": "variational_functional",
            "field": formula.field,
            "coordinate": formula.coordinate,
            "first_derivative": formula.first_derivative,
            "second_derivative": formula.second_derivative,
            "integrand": expression_to_data(formula.integrand),
            "claimed_euler_lagrange": expression_to_data(
                formula.claimed_euler_lagrange
            ),
        }
    raise TypeError(f"unsupported formula node: {type(formula).__name__}")
