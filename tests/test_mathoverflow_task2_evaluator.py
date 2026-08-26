from __future__ import annotations

from sigma_theory_compiler.mathoverflow_task2_evaluator import (
    PROMISING_GENERATOR_IDS,
    _generator_checks,
    union_closure,
)


def test_all_promising_llm_generator_families_fail_exactly():
    checks = _generator_checks()

    assert set(checks) == PROMISING_GENERATOR_IDS
    assert all(row["union_closed"] for row in checks.values())
    assert all(not row["exact_counterexample_valid"] for row in checks.values())


def test_union_closure_is_deterministic_and_complete():
    family = union_closure([{1, 2}, {2, 3}])

    assert family == [[1, 2], [2, 3], [1, 2, 3]]
