"""Deterministic, fail-closed counterexample evaluation for Math Pack v1."""

from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

import sympy as sp

from .math_expression_ir import Equation, Expression, Formula, Inequality, Recurrence, call, literal
from .math_types import (
    ComplexType,
    ExactComplex,
    IntegerType,
    MathType,
    RationalType,
    RealType,
    validate_value,
)


class UnsupportedEvaluation(ValueError):
    """Raised when evaluation cannot establish an exact truth value."""


class SearchStrategy(str, Enum):
    EXACT = "exact"
    ADVERSARIAL = "adversarial"
    RANDOM = "random"


class SearchStatus(str, Enum):
    COUNTEREXAMPLE_FOUND = "counterexample_found"
    INCONCLUSIVE_WITHIN_BUDGET = "inconclusive_within_budget"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class Counterexample:
    assignment: tuple[tuple[str, Any], ...]
    strategy: SearchStrategy
    trial_index: int


@dataclass(frozen=True, slots=True)
class CounterexampleReport:
    status: SearchStatus
    seed: int
    trials_run: int
    counterexample: Counterexample | None = None
    unsupported_reason: str | None = None
    proves_formula: bool = False

    def __post_init__(self) -> None:
        if self.proves_formula:
            raise ValueError("bounded counterexample search cannot prove a formula")
        if (self.status is SearchStatus.COUNTEREXAMPLE_FOUND) != (self.counterexample is not None):
            raise ValueError("counterexample payload disagrees with search status")
        if self.status is SearchStatus.UNSUPPORTED and not self.unsupported_reason:
            raise ValueError("unsupported reports require a reason")


def _exact_value(value: Any) -> sp.Expr:
    if isinstance(value, bool):
        raise UnsupportedEvaluation("boolean values are not mathematical scalars")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    if isinstance(value, ExactComplex):
        return sp.Rational(value.real.numerator, value.real.denominator) + sp.I * sp.Rational(
            value.imaginary.numerator, value.imaginary.denominator
        )
    raise UnsupportedEvaluation(f"non-exact scalar value: {value!r}")


def evaluate_expression(expression: Expression, assignment: Mapping[str, Any]) -> sp.Expr:
    """Evaluate an expression to an exact SymPy value or fail closed."""

    if expression.operation == "literal":
        return _exact_value(expression.value)
    if expression.operation == "symbol":
        name = str(expression.value)
        if name not in assignment:
            raise UnsupportedEvaluation(f"missing symbol assignment: {name}")
        return _exact_value(assignment[name])
    values = tuple(evaluate_expression(argument, assignment) for argument in expression.arguments)
    if expression.operation == "add":
        return sp.Add(*values)
    if expression.operation == "multiply":
        return sp.Mul(*values)
    if expression.operation == "negate":
        return -values[0]
    if expression.operation == "power":
        if values[0] == 0 and values[1].is_negative:
            raise UnsupportedEvaluation("division by zero")
        return sp.Pow(*values)
    if expression.operation == "call":
        name = str(expression.value)
        builtins = {
            "abs": sp.Abs,
            "cos": sp.cos,
            "exp": sp.exp,
            "log": sp.log,
            "sin": sp.sin,
            "sqrt": sp.sqrt,
        }
        if name in builtins:
            return builtins[name](*values)
        table = assignment.get(name)
        if not isinstance(table, Mapping):
            raise UnsupportedEvaluation(f"missing finite function table: {name}")
        keys = tuple(_python_exact(value) for value in values)
        key: Any = keys[0] if len(keys) == 1 else keys
        if key not in table:
            raise UnsupportedEvaluation(f"function table {name} has no entry for {key!r}")
        return _exact_value(table[key])
    raise UnsupportedEvaluation(f"unsupported expression operation: {expression.operation}")


def _python_exact(value: sp.Expr) -> Any:
    if value.is_Integer:
        return int(value)
    if value.is_Rational:
        return Fraction(int(value.p), int(value.q))
    real, imaginary = value.as_real_imag()
    if real.is_Rational and imaginary.is_Rational:
        return ExactComplex(
            Fraction(int(real.p), int(real.q)), Fraction(int(imaginary.p), int(imaginary.q))
        )
    raise UnsupportedEvaluation(f"function argument is not exact rational/complex: {value}")


def _equation_truth(equation: Equation, assignment: Mapping[str, Any]) -> bool:
    difference = sp.cancel(
        evaluate_expression(equation.left, assignment)
        - evaluate_expression(equation.right, assignment)
    )
    if difference.is_zero is True:
        return True
    if difference.is_zero is False:
        return False
    raise UnsupportedEvaluation(f"equation truth is undecidable at assignment: {difference}")


def evaluate_formula(formula: Formula, assignment: Mapping[str, Any]) -> bool:
    """Evaluate a formula exactly; never coerce an indeterminate result to truth."""

    if isinstance(formula, Equation):
        return _equation_truth(formula, assignment)
    if isinstance(formula, Inequality):
        difference = sp.cancel(
            evaluate_expression(formula.left, assignment)
            - evaluate_expression(formula.right, assignment)
        )
        if difference.is_real is not True:
            raise UnsupportedEvaluation("inequality requires a provably real value")
        predicates = {
            "<": difference.is_negative,
            "<=": difference.is_nonpositive,
            ">": difference.is_positive,
            ">=": difference.is_nonnegative,
            "!=": False
            if difference.is_zero is True
            else True
            if difference.is_zero is False
            else None,
        }
        result = predicates[formula.relation.value]
        if result is None:
            raise UnsupportedEvaluation(f"inequality truth is undecidable: {difference}")
        return bool(result)
    if isinstance(formula, Recurrence):
        if not _equation_truth(formula.equation, assignment):
            return False
        for index, expected in formula.initial_conditions:
            initial = Equation(call(formula.sequence, literal(index)), expected)
            if not _equation_truth(initial, assignment):
                return False
        return True
    raise UnsupportedEvaluation(f"unsupported formula node: {type(formula).__name__}")


def _adversarial_values(type_spec: MathType) -> tuple[Any, ...]:
    if isinstance(type_spec, IntegerType):
        candidates: list[Any] = [0, 1, -1]
        if type_spec.minimum is not None:
            candidates.append(type_spec.minimum)
        if type_spec.maximum is not None:
            candidates.append(type_spec.maximum)
    elif isinstance(type_spec, RationalType):
        candidates = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
        if type_spec.minimum is not None:
            candidates.append(type_spec.minimum)
        if type_spec.maximum is not None:
            candidates.append(type_spec.maximum)
    elif isinstance(type_spec, RealType):
        candidates = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
        if type_spec.minimum is not None and not isinstance(type_spec.minimum, float):
            candidates.append(type_spec.minimum)
        if type_spec.maximum is not None and not isinstance(type_spec.maximum, float):
            candidates.append(type_spec.maximum)
    elif isinstance(type_spec, ComplexType):
        candidates = [
            ExactComplex(),
            ExactComplex(1),
            ExactComplex(-1),
            ExactComplex(0, 1),
            ExactComplex(0, -1),
        ]
    else:
        raise UnsupportedEvaluation(
            f"automatic adversarial sampling does not support {type(type_spec).__name__}"
        )
    return tuple(dict.fromkeys(value for value in candidates if validate_value(type_spec, value)))


def _random_value(type_spec: MathType, generator: random.Random) -> Any:
    if isinstance(type_spec, IntegerType):
        minimum = type_spec.minimum if type_spec.minimum is not None else -10
        maximum = type_spec.maximum if type_spec.maximum is not None else 10
        return generator.randint(minimum, maximum)
    if isinstance(type_spec, (RationalType, RealType)):
        for _ in range(256):
            value = Fraction(generator.randint(-20, 20), generator.randint(1, 12))
            if validate_value(type_spec, value):
                return value
        raise UnsupportedEvaluation("could not sample a bounded rational/real type")
    if isinstance(type_spec, ComplexType):
        return ExactComplex(
            Fraction(generator.randint(-10, 10), generator.randint(1, 8)),
            Fraction(generator.randint(-10, 10), generator.randint(1, 8)),
        )
    raise UnsupportedEvaluation(f"random sampling does not support {type(type_spec).__name__}")


def _validate_assignment(variables: Mapping[str, MathType], assignment: Mapping[str, Any]) -> None:
    for name, type_spec in variables.items():
        if name not in assignment:
            raise UnsupportedEvaluation(f"missing typed variable: {name}")
        if not validate_value(type_spec, assignment[name]):
            raise UnsupportedEvaluation(f"assignment for {name} does not inhabit {type_spec!r}")


def find_counterexample(
    formula: Formula,
    variables: Mapping[str, MathType],
    *,
    exact_assignments: Sequence[Mapping[str, Any]] = (),
    strategies: tuple[SearchStrategy, ...] = (
        SearchStrategy.EXACT,
        SearchStrategy.ADVERSARIAL,
        SearchStrategy.RANDOM,
    ),
    random_trials: int = 64,
    adversarial_limit: int = 256,
    seed: int = 0,
) -> CounterexampleReport:
    """Search a finite deterministic schedule; an exhausted search remains inconclusive."""

    if random_trials < 0 or adversarial_limit < 0:
        raise ValueError("search budgets cannot be negative")
    if len(set(strategies)) != len(strategies):
        raise ValueError("search strategies must be unique")
    generator = random.Random(seed)
    trials = 0
    unsupported: str | None = None

    def attempt(candidate: Mapping[str, Any], strategy: SearchStrategy) -> Counterexample | None:
        nonlocal trials, unsupported
        try:
            _validate_assignment(variables, candidate)
            truth = evaluate_formula(formula, candidate)
        except (UnsupportedEvaluation, ZeroDivisionError) as error:
            unsupported = str(error)
            return None
        trial_index = trials
        trials += 1
        if truth:
            return None
        return Counterexample(tuple(sorted(candidate.items())), strategy, trial_index)

    for strategy in strategies:
        try:
            if strategy is SearchStrategy.EXACT:
                candidates = exact_assignments
            elif strategy is SearchStrategy.ADVERSARIAL:
                names = sorted(variables)
                products = itertools.product(
                    *(_adversarial_values(variables[name]) for name in names)
                )
                candidates = (
                    dict(zip(names, values, strict=True))
                    for values in itertools.islice(products, adversarial_limit)
                )
            elif strategy is SearchStrategy.RANDOM:
                names = sorted(variables)
                candidates = (
                    {name: _random_value(variables[name], generator) for name in names}
                    for _ in range(random_trials)
                )
            else:
                raise UnsupportedEvaluation(f"unknown search strategy: {strategy}")
            for candidate in candidates:
                counterexample = attempt(candidate, strategy)
                if counterexample is not None:
                    return CounterexampleReport(
                        SearchStatus.COUNTEREXAMPLE_FOUND, seed, trials, counterexample
                    )
        except UnsupportedEvaluation as error:
            unsupported = str(error)
    if unsupported is not None:
        return CounterexampleReport(
            SearchStatus.UNSUPPORTED,
            seed,
            trials,
            unsupported_reason=unsupported,
        )
    return CounterexampleReport(SearchStatus.INCONCLUSIVE_WITHIN_BUDGET, seed, trials)
