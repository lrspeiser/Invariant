from __future__ import annotations

import copy
import math

import pytest

from sigma_theory_compiler.anonymous_monomial_discovery import (
    AnonymousMonomialError,
    discover,
    enumerate_exponents,
    fit_candidate,
    relation_expression,
    score_frozen_candidate,
    validate_search,
)


def _rows(count: int = 40) -> list[dict]:
    rows: list[dict] = []
    for index in range(1, count + 1):
        x1 = 1.0 + index / 7
        x2 = 0.8 + ((index * 17) % 29) / 11
        x0 = x1 * x2
        rows.append(
            {
                "label": f"r{index:03d}",
                "values": [str(x0), str(x1), str(x2)],
                "uncertainties": [str(x0 * 1e-6), str(x1 * 1e-6), str(x2 * 1e-6)],
            }
        )
    return rows


def test_complete_space_is_primitive_oriented_and_deduplicated() -> None:
    vectors = enumerate_exponents(3, 12)
    assert len(vectors) == 6152
    assert len(set(vectors)) == len(vectors)
    assert all(vector[0] > 0 for vector in vectors)
    assert all(sum(value != 0 for value in vector) >= 2 for vector in vectors)
    assert all(math.gcd(*[abs(value) for value in vector]) == 1 for vector in vectors)
    assert all(tuple(-value for value in vector) not in vectors for vector in vectors)


def test_new_search_recovers_a_three_column_relation_old_pairwise_cannot_express() -> None:
    rows = _rows()
    common = {"arity": 3, "candidate_budget": 256, "exponent_bound": 12}
    new = discover(rows, strategy="new_occam", **common)
    old = discover(rows, strategy="old_pairwise", **common)
    assert new["best_candidate"]["exponents"] == [1, -1, -1]
    assert float(new["best_candidate"]["fit_metrics"]["maximum_absolute_log_residual"]) < 1e-12
    assert old["best_candidate"]["support_size"] == 2
    assert float(old["best_candidate"]["fit_metrics"]["median_absolute_log_residual"]) > 0.01


def test_unit_rescaling_preserves_the_discovered_exponents() -> None:
    rows = _rows()
    scaled = [
        {
            "label": row["label"],
            "values": [
                str(float(row["values"][0]) * 7),
                str(float(row["values"][1]) * 11),
                str(float(row["values"][2]) * 13),
            ],
            "uncertainties": [
                str(float(row["uncertainties"][0]) * 7),
                str(float(row["uncertainties"][1]) * 11),
                str(float(row["uncertainties"][2]) * 13),
            ],
        }
        for row in rows
    ]
    common = {"arity": 3, "candidate_budget": 256, "exponent_bound": 12}
    original = discover(rows, strategy="new_occam", **common)
    converted = discover(scaled, strategy="new_occam", **common)
    assert original["best_candidate"]["exponents"] == converted["best_candidate"]["exponents"]
    assert original["best_candidate"]["fit_log_constant"] != converted["best_candidate"]["fit_log_constant"]


def test_random_order_is_seeded_replayable_and_budget_matched() -> None:
    rows = _rows()
    common = {
        "arity": 3,
        "candidate_budget": 256,
        "exponent_bound": 12,
        "strategy": "uniform_random",
    }
    first = discover(rows, random_seed=101, **common)
    replay = discover(rows, random_seed=101, **common)
    second = discover(rows, random_seed=103, **common)
    assert first == replay
    assert first["candidate_budget"] == second["candidate_budget"] == 256
    assert first["full_pool_sha256"] == second["full_pool_sha256"]
    assert first["evaluated_exponent_sha256"] != second["evaluated_exponent_sha256"]


def test_frozen_constant_is_not_refit_on_holdout() -> None:
    training = _rows()
    candidate = fit_candidate(training, [1, -1, -1])
    shifted = copy.deepcopy(training)
    for row in shifted:
        row["values"][0] = str(float(row["values"][0]) * 2)
    score = score_frozen_candidate(shifted, candidate)
    assert float(score["median_absolute_log_residual"]) == pytest.approx(math.log(2))
    assert float(score["median_absolute_response_log_error"]) == pytest.approx(math.log(2))


def test_search_receipt_replays_and_tamper_fails() -> None:
    rows = _rows()
    receipt = discover(
        rows,
        arity=3,
        exponent_bound=12,
        candidate_budget=256,
        strategy="new_occam",
    )
    assert validate_search(receipt, rows, exponent_bound=12) == receipt
    drifted = copy.deepcopy(receipt)
    drifted["best_candidate"]["fit_log_constant"] = "123"
    with pytest.raises(AnonymousMonomialError, match="does not replay"):
        validate_search(drifted, rows, exponent_bound=12)


def test_invalid_rows_and_oversized_budgets_fail_closed() -> None:
    rows = _rows()
    rows[0]["values"][0] = "0"
    with pytest.raises(AnonymousMonomialError, match="non-positive"):
        discover(
            rows,
            arity=3,
            exponent_bound=12,
            candidate_budget=256,
            strategy="new_occam",
        )
    with pytest.raises(AnonymousMonomialError, match="exceeds"):
        discover(
            _rows(),
            arity=3,
            exponent_bound=2,
            candidate_budget=10_000,
            strategy="old_pairwise",
        )


def test_expression_is_neutral_and_explicit() -> None:
    assert relation_expression([2, -3, 1]) == "x0^2*x1^-3*x2 = constant"
    text = relation_expression([1, 0, -2]).lower()
    assert not any(word in text for word in ("orbit", "mass", "planet", "period"))
