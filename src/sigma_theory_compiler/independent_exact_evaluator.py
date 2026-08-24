"""A small, independent exact evaluator for creativity-campaign reproduction.

This module intentionally does not import SymPy, Z3, mpmath, or the campaign
implementation.  It supplies a second executable interpretation of the admitted
arithmetic grammar using only Python's AST and :class:`fractions.Fraction`.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from fractions import Fraction


class IndependentEvaluationError(ValueError):
    """Raised when an expression escapes the independent arithmetic grammar."""


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
                return Fraction(node.value)
            raise IndependentEvaluationError("only integer constants are admitted")
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
