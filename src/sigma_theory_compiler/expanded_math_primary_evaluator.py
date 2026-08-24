"""SymPy-backed exact evaluator for the expanded immutable mathematical grammar."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from fractions import Fraction
from math import prod

import sympy as sp

from .math_canonicalizer import to_sympy
from .math_expression_ir import (
    FiniteProduct,
    FiniteSum,
    Formula,
    GeneratingFunction,
    ModularRelation,
    PiecewiseComparator,
    PiecewiseRelation,
    TensorIdentity,
    VariationalFunctional,
)


class ExpandedPrimaryEvaluationError(ValueError):
    """The formula escaped the bounded exact primary evaluator."""


def _substitutions(assignment: Mapping[str, int | Fraction]) -> dict[sp.Symbol, sp.Rational]:
    result = {}
    for name, value in assignment.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or isinstance(value, bool)
            or not isinstance(value, (int, Fraction))
        ):
            raise ExpandedPrimaryEvaluationError("primary assignment is malformed")
        rational = Fraction(value)
        result[sp.Symbol(name)] = sp.Rational(rational.numerator, rational.denominator)
    return result


def _exact(expression, assignment: Mapping[str, int | Fraction]) -> sp.Rational:
    value = sp.cancel(to_sympy(expression).subs(_substitutions(assignment)))
    if value.free_symbols or value.is_Rational is not True:
        raise ExpandedPrimaryEvaluationError("primary expression is not an exact rational")
    return sp.Rational(value)


def _coordinates(shape: tuple[int, ...]):
    return itertools.product(*(range(size) for size in shape))


def _offset(shape: tuple[int, ...], coordinates: tuple[int, ...]) -> int:
    offset = 0
    for size, coordinate in zip(shape, coordinates, strict=True):
        offset = offset * size + coordinate
    return offset


def _tensor_symmetries(
    values: tuple[sp.Rational, ...],
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


def evaluate(
    formula: Formula, assignment: Mapping[str, int | Fraction] | None = None
) -> bool:
    """Return exact truth for one registered expanded formula or fail closed."""

    assignment = dict(assignment or {})
    if isinstance(formula, FiniteSum):
        index = str(formula.index.value)
        total = sum(
            (_exact(formula.summand, {**assignment, index: value}) for value in range(formula.lower, formula.upper + 1)),
            sp.Integer(0),
        )
        return sp.cancel(total - _exact(formula.claimed_value, assignment)) == 0
    if isinstance(formula, FiniteProduct):
        index = str(formula.index.value)
        total = sp.Integer(1)
        for value in range(formula.lower, formula.upper + 1):
            total *= _exact(formula.factor, {**assignment, index: value})
        return sp.cancel(total - _exact(formula.claimed_value, assignment)) == 0
    if isinstance(formula, GeneratingFunction):
        variable = str(formula.variable.value)
        if variable not in assignment:
            raise ExpandedPrimaryEvaluationError("generating-function point is missing")
        point = _exact(formula.variable, assignment)
        total = sum(
            (
                _exact(coefficient, assignment) * point**index
                for index, coefficient in enumerate(formula.coefficients)
            ),
            sp.Integer(0),
        )
        return sp.cancel(total - _exact(formula.claimed_value, assignment)) == 0
    if isinstance(formula, ModularRelation):
        left = _exact(formula.left, assignment)
        right = _exact(formula.right, assignment)
        if left.q != 1 or right.q != 1:
            raise ExpandedPrimaryEvaluationError("modular relation requires integers")
        return (int(left) - int(right)) % formula.modulus == 0
    if isinstance(formula, PiecewiseRelation):
        selected = formula.default_expression
        for branch in formula.branches:
            left = _exact(branch.left, assignment)
            right = _exact(branch.right, assignment)
            conditions = {
                PiecewiseComparator.LESS: left < right,
                PiecewiseComparator.LESS_EQUAL: left <= right,
                PiecewiseComparator.EQUAL: left == right,
                PiecewiseComparator.NOT_EQUAL: left != right,
                PiecewiseComparator.GREATER_EQUAL: left >= right,
                PiecewiseComparator.GREATER: left > right,
            }
            if conditions[branch.comparator]:
                selected = branch.expression
                break
        return _exact(selected, assignment) == _exact(formula.claimed_value, assignment)
    if isinstance(formula, TensorIdentity):
        left = tuple(_exact(item, assignment) for item in formula.left_components)
        right = tuple(_exact(item, assignment) for item in formula.right_components)
        return (
            left == right
            and _tensor_symmetries(left, formula.shape, formula.symmetries)
            and _tensor_symmetries(right, formula.shape, formula.symmetries)
        )
    if isinstance(formula, VariationalFunctional):
        coordinate = sp.Symbol(formula.coordinate)
        field = sp.Symbol(formula.field)
        first = sp.Symbol(formula.first_derivative)
        second = sp.Symbol(formula.second_derivative)
        reserved = {coordinate, field, first, second}
        substitutions = {
            key: value for key, value in _substitutions(assignment).items() if key not in reserved
        }
        integrand = sp.expand(to_sympy(formula.integrand).subs(substitutions))
        claimed = sp.expand(to_sympy(formula.claimed_euler_lagrange).subs(substitutions))
        if (integrand.free_symbols | claimed.free_symbols) - reserved:
            raise ExpandedPrimaryEvaluationError("variational expression has undeclared parameters")
        total_derivative = (
            sp.diff(sp.diff(integrand, first), coordinate)
            + first * sp.diff(sp.diff(integrand, first), field)
            + second * sp.diff(sp.diff(integrand, first), first)
        )
        euler_lagrange = sp.diff(integrand, field) - total_derivative
        return sp.expand(euler_lagrange - claimed) == 0
    raise ExpandedPrimaryEvaluationError(
        f"unsupported expanded formula: {type(formula).__name__}"
    )


__all__ = ["ExpandedPrimaryEvaluationError", "evaluate"]
