"""A small, independent exact evaluator for creativity-campaign reproduction.

This module intentionally does not import SymPy, Z3, mpmath, or the campaign
implementation.  It supplies a second executable interpretation of the admitted
arithmetic grammar using only Python's AST and :class:`fractions.Fraction`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction


class IndependentEvaluationError(ValueError):
    """Raised when an expression escapes the independent arithmetic grammar."""


_DECIMAL_LITERAL = re.compile(
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)
_MAX_EXACT_LITERAL = 10**12


def _decimal_fraction(source: str, node: ast.Constant) -> Fraction:
    text = ast.get_source_segment(source, node)
    if text is None or _DECIMAL_LITERAL.fullmatch(text) is None:
        raise IndependentEvaluationError("decimal literal is not canonical")
    try:
        value = Fraction(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise IndependentEvaluationError("decimal literal is not finite") from error
    if abs(value.numerator) > _MAX_EXACT_LITERAL or value.denominator > _MAX_EXACT_LITERAL:
        raise IndependentEvaluationError("exact literal exceeds the coefficient budget")
    return value


def _round_half_even(value: Fraction) -> Fraction:
    lower = value.numerator // value.denominator
    remainder = value - lower
    if remainder < Fraction(1, 2):
        return Fraction(lower)
    if remainder > Fraction(1, 2):
        return Fraction(lower + 1)
    return Fraction(lower if lower % 2 == 0 else lower + 1)


def _integer_constant(node: ast.AST) -> int:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _integer_constant(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise IndependentEvaluationError("power exponent must be a literal integer")


def evaluate_expression(expression: str, variables: Mapping[str, Fraction]) -> Fraction:
    """Evaluate the closed arithmetic DSL without using the primary symbolic stack."""

    if not expression or len(expression.encode("utf-8")) > 4096:
        raise IndependentEvaluationError("expression is empty or exceeds the byte limit")
    try:
        root = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise IndependentEvaluationError("expression is not valid arithmetic syntax") from error

    def compare(node: ast.Compare) -> bool:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise IndependentEvaluationError("only one exact comparison is admitted")
        left = visit(node.left)
        right = visit(node.comparators[0])
        operator = node.ops[0]
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.Gt):
            return left > right
        raise IndependentEvaluationError("comparison operator is unsupported")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name):
            try:
                return variables[node.id]
            except KeyError as error:
                raise IndependentEvaluationError("expression uses an unknown variable") from error
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                if abs(node.value) > _MAX_EXACT_LITERAL:
                    raise IndependentEvaluationError("integer literal exceeds the coefficient budget")
                return Fraction(node.value)
            if isinstance(node.value, float):
                return _decimal_fraction(expression, node)
            raise IndependentEvaluationError("only exact numeric constants are admitted")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            if isinstance(node.op, ast.Pow):
                exponent = _integer_constant(node.right)
                if not -8 <= exponent <= 8:
                    raise IndependentEvaluationError("power exponent is outside [-8, 8]")
                try:
                    return left**exponent
                except ZeroDivisionError as error:
                    raise IndependentEvaluationError(
                        "negative power of zero is undefined"
                    ) from error
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise IndependentEvaluationError("division by zero")
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise IndependentEvaluationError("floor division by zero")
                return Fraction(left // right)
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise IndependentEvaluationError("modulo by zero")
                return left % right
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.keywords
                or len(node.args) != 1
            ):
                raise IndependentEvaluationError("function call is outside the exact DSL")
            value = visit(node.args[0])
            if node.func.id == "floor":
                return Fraction(value.numerator // value.denominator)
            if node.func.id == "round":
                return _round_half_even(value)
            raise IndependentEvaluationError("function is outside the exact DSL")
        if isinstance(node, ast.IfExp):
            if not isinstance(node.test, ast.Compare):
                raise IndependentEvaluationError("conditional test must be one exact comparison")
            return visit(node.body if compare(node.test) else node.orelse)
        raise IndependentEvaluationError("expression contains an unsupported syntax node")

    return visit(root)


def evaluate_comparison(
    left_expression: str,
    comparator: str,
    right_expression: str,
    variables: Mapping[str, Fraction],
) -> bool:
    """Evaluate one exact comparison without the primary symbolic stack."""

    left = evaluate_expression(left_expression, variables)
    right = evaluate_expression(right_expression, variables)
    if comparator == "lt":
        return left < right
    if comparator == "le":
        return left <= right
    if comparator == "eq":
        return left == right
    if comparator == "ne":
        return left != right
    if comparator == "ge":
        return left >= right
    if comparator == "gt":
        return left > right
    raise IndependentEvaluationError("comparison operator is unsupported")


def evaluate_recurrence(
    coefficients: Sequence[Fraction],
    seed: Sequence[Fraction],
    indices: Sequence[int],
) -> tuple[Fraction, ...]:
    """Evaluate a constant-coefficient recurrence in a separate exact implementation."""

    if not coefficients or len(coefficients) != len(seed):
        raise IndependentEvaluationError("recurrence coefficients and seed must have equal order")
    if any(index < 0 for index in indices):
        raise IndependentEvaluationError("recurrence indices must be nonnegative")
    values = list(seed)
    maximum = max(indices, default=-1)
    while len(values) <= maximum:
        values.append(
            sum(coefficient * values[-offset] for offset, coefficient in enumerate(coefficients, 1))
        )
    return tuple(values[index] for index in indices)


__all__ = [
    "IndependentEvaluationError",
    "evaluate_comparison",
    "evaluate_expression",
    "evaluate_recurrence",
]
