from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp


X, Q, Z = sp.symbols("x q z", nonnegative=True, finite=True)
SYMBOLS = {"x": X, "q": Q, "z": Z}


@dataclass(frozen=True)
class GrammarExpression:
    expression: sp.Expr
    complexity: int
    canonical: str


def canonicalize(expression: sp.Expr) -> tuple[sp.Expr, str]:
    simplified = sp.factor(sp.cancel(sp.together(expression)))
    return simplified, sp.srepr(simplified)


def _unary(name: str, value: sp.Expr) -> sp.Expr:
    if name == "sqrt1p_minus1":
        return sp.sqrt(1 + value) - 1
    if name == "saturate":
        return value / (1 + value)
    raise ValueError(f"Unknown unary operator: {name}")


def enumerate_expressions(
    atoms: list[str], unary_operators: list[str], binary_operators: list[str], max_complexity: int
) -> tuple[list[GrammarExpression], dict[str, int]]:
    """Enumerate every expression in the declared grammar up to an exact node budget.

    Atom cost is 1, unary cost is 1 + child cost, and binary cost is
    1 + left cost + right cost. Commutative operands are ordered before insertion.
    Algebraically equivalent expressions are collapsed by their SymPy canonical form.
    """

    if max_complexity < 1:
        return [], {"generated_before_deduplication": 0, "unique": 0, "duplicates_removed": 0}
    unknown = set(atoms) - set(SYMBOLS)
    if unknown:
        raise ValueError(f"Unknown atoms: {sorted(unknown)}")

    by_cost: dict[int, dict[str, sp.Expr]] = {}
    raw_count = 0
    atom_bucket: dict[str, sp.Expr] = {}
    for atom in atoms:
        raw_count += 1
        expr, key = canonicalize(SYMBOLS[atom])
        atom_bucket[key] = expr
    by_cost[1] = atom_bucket

    for cost in range(2, max_complexity + 1):
        bucket: dict[str, sp.Expr] = {}
        for operator in unary_operators:
            for child in by_cost.get(cost - 1, {}).values():
                raw_count += 1
                expr, key = canonicalize(_unary(operator, child))
                bucket.setdefault(key, expr)

        for left_cost in range(1, cost - 1):
            right_cost = cost - 1 - left_cost
            for left, right in product(
                by_cost.get(left_cost, {}).values(), by_cost.get(right_cost, {}).values()
            ):
                for operator in binary_operators:
                    raw_count += 1
                    if operator == "add":
                        combined = left + right
                    elif operator == "multiply":
                        combined = left * right
                    else:
                        raise ValueError(f"Unknown binary operator: {operator}")
                    expr, key = canonicalize(combined)
                    bucket.setdefault(key, expr)
        by_cost[cost] = bucket

    # The same algebraic expression can occur at several costs. Keep its cheapest proof.
    unique: dict[str, GrammarExpression] = {}
    for cost in range(1, max_complexity + 1):
        for key, expression in by_cost.get(cost, {}).items():
            unique.setdefault(key, GrammarExpression(expression, cost, key))

    result = sorted(unique.values(), key=lambda item: (item.complexity, item.canonical))
    return result, {
        "generated_before_deduplication": raw_count,
        "unique": len(result),
        "duplicates_removed": raw_count - len(result),
    }

