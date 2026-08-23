"""Pure-Python exact evaluator independent of SymPy and the primary grammar evaluator."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from fractions import Fraction
from math import prod

from .independent_exact_evaluator import IndependentEvaluationError, evaluate_expression
from .math_expression_ir import (
    Expression,
    FiniteProduct,
    FiniteSum,
    Formula,
    GeneratingFunction,
    ModularRelation,
    TensorIdentity,
    VariationalFunctional,
)


class ExpandedIndependentEvaluationError(ValueError):
    """The formula escaped the independent exact grammar."""


Polynomial = dict[tuple[int, ...], Fraction]


def _literal_text(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ExpandedIndependentEvaluationError("independent evaluator requires exact rationals")
    rational = Fraction(value)
    return f"({rational.numerator}/{rational.denominator})"


def _expression_text(expression: Expression) -> str:
    if expression.operation == "literal":
        return _literal_text(expression.value)
    if expression.operation == "symbol":
        return str(expression.value)
    arguments = [_expression_text(item) for item in expression.arguments]
    if expression.operation == "add":
        return "(" + "+".join(arguments) + ")"
    if expression.operation == "multiply":
        return "(" + "*".join(arguments) + ")"
    if expression.operation == "negate":
        return f"(-{arguments[0]})"
    if expression.operation == "power":
        exponent = expression.arguments[1]
        if (
            exponent.operation != "literal"
            or isinstance(exponent.value, bool)
            or not isinstance(exponent.value, int)
        ):
            raise ExpandedIndependentEvaluationError("power exponent must be an integer literal")
        return f"({arguments[0]}**{exponent.value})"
    raise ExpandedIndependentEvaluationError("independent expression operation is unsupported")


def _exact(
    expression: Expression, assignment: Mapping[str, int | Fraction]
) -> Fraction:
    variables = {}
    for name, value in assignment.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or isinstance(value, bool)
            or not isinstance(value, (int, Fraction))
        ):
            raise ExpandedIndependentEvaluationError("independent assignment is malformed")
        variables[name] = Fraction(value)
    try:
        return evaluate_expression(_expression_text(expression), variables)
    except IndependentEvaluationError as error:
        raise ExpandedIndependentEvaluationError(str(error)) from error


def _coordinates(shape: tuple[int, ...]):
    return itertools.product(*(range(size) for size in shape))


def _offset(shape: tuple[int, ...], coordinates: tuple[int, ...]) -> int:
    offset = 0
    for size, coordinate in zip(shape, coordinates, strict=True):
        offset = offset * size + coordinate
    return offset


def _tensor_symmetries(
    values: tuple[Fraction, ...],
    shape: tuple[int, ...],
    symmetries: tuple[tuple[int, int, int], ...],
) -> bool:
    if len(values) != prod(shape):
        return False
    for coordinates in _coordinates(shape):
        coordinates = tuple(coordinates)
        source = values[_offset(shape, coordinates)]
        for left_axis, right_axis, sign in symmetries:
            swapped = list(coordinates)
            swapped[left_axis], swapped[right_axis] = swapped[right_axis], swapped[left_axis]
            if source != sign * values[_offset(shape, tuple(swapped))]:
                return False
    return True


def _clean(polynomial: Polynomial) -> Polynomial:
    return {powers: coefficient for powers, coefficient in polynomial.items() if coefficient}


def _constant(value: int | Fraction, dimension: int) -> Polynomial:
    rational = Fraction(value)
    return {} if rational == 0 else {(0,) * dimension: rational}


def _variable(index: int, dimension: int) -> Polynomial:
    powers = [0] * dimension
    powers[index] = 1
    return {tuple(powers): Fraction(1)}


def _add(*polynomials: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for polynomial in polynomials:
        for powers, coefficient in polynomial.items():
            result[powers] = result.get(powers, Fraction(0)) + coefficient
    return _clean(result)


def _scale(polynomial: Polynomial, scalar: int | Fraction) -> Polynomial:
    return _clean({powers: Fraction(scalar) * value for powers, value in polynomial.items()})


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_powers, left_value in left.items():
        for right_powers, right_value in right.items():
            powers = tuple(
                left_power + right_power
                for left_power, right_power in zip(left_powers, right_powers, strict=True)
            )
            result[powers] = result.get(powers, Fraction(0)) + left_value * right_value
    return _clean(result)


def _power(polynomial: Polynomial, exponent: int, dimension: int) -> Polynomial:
    if not 0 <= exponent <= 8:
        raise ExpandedIndependentEvaluationError("polynomial exponent is outside [0, 8]")
    result = _constant(1, dimension)
    base = polynomial
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _multiply(result, base)
        base = _multiply(base, base)
        remaining //= 2
    return result


def _differentiate(polynomial: Polynomial, variable: int) -> Polynomial:
    result: Polynomial = {}
    for powers, coefficient in polynomial.items():
        exponent = powers[variable]
        if exponent:
            reduced = list(powers)
            reduced[variable] -= 1
            result[tuple(reduced)] = coefficient * exponent
    return _clean(result)


def _polynomial(
    expression: Expression,
    names: tuple[str, ...],
    parameters: Mapping[str, int | Fraction],
) -> Polynomial:
    dimension = len(names)
    if expression.operation == "literal":
        if isinstance(expression.value, bool) or not isinstance(expression.value, (int, Fraction)):
            raise ExpandedIndependentEvaluationError("polynomial literal is not exact rational")
        return _constant(expression.value, dimension)
    if expression.operation == "symbol":
        name = str(expression.value)
        if name in names:
            return _variable(names.index(name), dimension)
        if (
            name in parameters
            and not isinstance(parameters[name], bool)
            and isinstance(parameters[name], (int, Fraction))
        ):
            return _constant(Fraction(parameters[name]), dimension)
        raise ExpandedIndependentEvaluationError("polynomial contains an undeclared symbol")
    arguments = [_polynomial(item, names, parameters) for item in expression.arguments]
    if expression.operation == "add":
        return _add(*arguments)
    if expression.operation == "multiply":
        result = _constant(1, dimension)
        for argument in arguments:
            result = _multiply(result, argument)
        return result
    if expression.operation == "negate":
        return _scale(arguments[0], -1)
    if expression.operation == "power":
        exponent = expression.arguments[1]
        if (
            exponent.operation != "literal"
            or isinstance(exponent.value, bool)
            or not isinstance(exponent.value, int)
        ):
            raise ExpandedIndependentEvaluationError("polynomial power is not a literal integer")
        return _power(arguments[0], exponent.value, dimension)
    raise ExpandedIndependentEvaluationError("variational polynomial operation is unsupported")


def _variational_truth(
    formula: VariationalFunctional, parameters: Mapping[str, int | Fraction]
) -> bool:
    names = (
        formula.coordinate,
        formula.field,
        formula.first_derivative,
        formula.second_derivative,
    )
    integrand = _polynomial(formula.integrand, names, parameters)
    claimed = _polynomial(formula.claimed_euler_lagrange, names, parameters)
    d_l_d_first = _differentiate(integrand, 2)
    total_derivative = _add(
        _differentiate(d_l_d_first, 0),
        _multiply(_variable(2, 4), _differentiate(d_l_d_first, 1)),
        _multiply(_variable(3, 4), _differentiate(d_l_d_first, 2)),
    )
    euler_lagrange = _add(_differentiate(integrand, 1), _scale(total_derivative, -1))
    return _clean(_add(euler_lagrange, _scale(claimed, -1))) == {}


def evaluate(
    formula: Formula, assignment: Mapping[str, int | Fraction] | None = None
) -> bool:
    """Return exact truth using only Python AST/Fraction and sparse-polynomial operations."""

    assignment = dict(assignment or {})
    if isinstance(formula, FiniteSum):
        index = str(formula.index.value)
        total = sum(
            (
                _exact(formula.summand, {**assignment, index: value})
                for value in range(formula.lower, formula.upper + 1)
            ),
            Fraction(0),
        )
        return total == _exact(formula.claimed_value, assignment)
    if isinstance(formula, FiniteProduct):
        index = str(formula.index.value)
        total = Fraction(1)
        for value in range(formula.lower, formula.upper + 1):
            total *= _exact(formula.factor, {**assignment, index: value})
        return total == _exact(formula.claimed_value, assignment)
    if isinstance(formula, GeneratingFunction):
        variable = str(formula.variable.value)
        if variable not in assignment:
            raise ExpandedIndependentEvaluationError("generating-function point is missing")
        point = _exact(formula.variable, assignment)
        total = sum(
            (
                _exact(coefficient, assignment) * point**index
                for index, coefficient in enumerate(formula.coefficients)
            ),
            Fraction(0),
        )
        return total == _exact(formula.claimed_value, assignment)
    if isinstance(formula, ModularRelation):
        left = _exact(formula.left, assignment)
        right = _exact(formula.right, assignment)
        if left.denominator != 1 or right.denominator != 1:
            raise ExpandedIndependentEvaluationError("modular relation requires integers")
        _, remainder = divmod(left.numerator - right.numerator, formula.modulus)
        return remainder == 0
    if isinstance(formula, TensorIdentity):
        left = tuple(_exact(item, assignment) for item in formula.left_components)
        right = tuple(_exact(item, assignment) for item in formula.right_components)
        return (
            left == right
            and _tensor_symmetries(left, formula.shape, formula.symmetries)
            and _tensor_symmetries(right, formula.shape, formula.symmetries)
        )
    if isinstance(formula, VariationalFunctional):
        return _variational_truth(formula, assignment)
    raise ExpandedIndependentEvaluationError(
        f"unsupported expanded formula: {type(formula).__name__}"
    )


__all__ = ["ExpandedIndependentEvaluationError", "evaluate"]
