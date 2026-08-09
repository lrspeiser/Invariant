import sympy as sp

from sigma_theory_compiler.grammar import Q, X, enumerate_expressions


def test_enumeration_is_deterministic_and_deduplicated() -> None:
    first, first_counts = enumerate_expressions(
        ["x", "q"], ["saturate"], ["add", "multiply"], 4
    )
    second, second_counts = enumerate_expressions(
        ["x", "q"], ["saturate"], ["add", "multiply"], 4
    )
    assert [item.canonical for item in first] == [item.canonical for item in second]
    assert len({item.canonical for item in first}) == len(first)
    assert first_counts == second_counts
    assert first_counts["duplicates_removed"] > 0


def test_commutative_duplicates_collapse() -> None:
    expressions, _ = enumerate_expressions(["x", "q"], [], ["add"], 3)
    sums = [item for item in expressions if sp.expand(item.expression) == Q + X]
    assert len(sums) == 1
