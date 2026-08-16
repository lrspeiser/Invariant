"""B3 conjecture-generation gates.

A conjecture generator is trivially "successful" if it is allowed to look at all the
data before speaking.  These tests fix the prefix/holdout discipline, the refutation
path, and the rule that a surviving conjecture is still not a proof.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.conjecture_generation import (
    CLAIMS,
    STATEMENT_KINDS,
    STATEMENT_KINDS_V1,
    SYSTEM_CAPS,
    ConjectureGenerationError,
    generate_conjectures,
    validate_result,
)


def _rows(function, points):
    rows = []
    for point in points:
        value = Fraction(function(point))
        rows.append(
            {
                "point": point,
                "value": {"numerator": value.numerator, "denominator": value.denominator},
            }
        )
    return rows


def _fibonacci(count: int) -> list[int]:
    values = [0, 1]
    while len(values) < count:
        values.append(values[-1] + values[-2])
    return values


def _kinds(result, status):
    return {entry["kind"] for entry in result["conjectures"] if entry.get("status") == status}


# ---------------------------------------------------------------------------
# Discovery of statement kinds
# ---------------------------------------------------------------------------


def test_recurrence_is_found_where_no_closed_form_exists():
    """Fibonacci has no rational closed form; its recurrence is the real statement."""

    values = _fibonacci(24)
    result = generate_conjectures(_rows(lambda n: values[n], range(16)))
    assert result["decision"] == "PROPOSED"
    assert "linear_recurrence" in _kinds(result, "SURVIVED")
    assert "closed_form" not in _kinds(result, "SURVIVED")
    recurrence = next(
        entry for entry in result["conjectures"] if entry["kind"] == "linear_recurrence"
    )
    assert recurrence["statement"] == "a(n) = (1)*a(n-1) + (1)*a(n-2)"
    assert recurrence["support"] == result["counts"]["holdout_rows"]


def test_multiple_distinct_statement_kinds_are_proposed():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    survived = _kinds(result, "SURVIVED")
    assert {"closed_form", "linear_recurrence", "divisibility", "monotonicity"} <= survived


def test_divisibility_is_detected_and_confirmed():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    entry = next(item for item in result["conjectures"] if item["kind"] == "divisibility")
    assert entry["statement"] == "6 divides a(n)"
    assert entry["status"] == "SURVIVED"


def test_partial_sum_closed_form_is_a_separate_statement():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    entry = next(
        item for item in result["conjectures"] if item["kind"] == "partial_sum_closed_form"
    )
    assert entry["statement"].startswith("sum_(i<=n) a(i) =")
    assert entry["status"] == "SURVIVED"


# ---------------------------------------------------------------------------
# Refutation — the load-bearing behaviour
# ---------------------------------------------------------------------------


def test_conjecture_from_a_misleading_prefix_is_refuted_on_holdout():
    values = [5, 3, 9, 2, 7, 1, 8, 4, 6, 0, 11, 13, 17, 19]
    result = generate_conjectures(_rows(lambda n: values[n], range(14)))
    assert result["decision"] == "NONE_SURVIVED"
    assert result["counts"]["refuted"] >= 1
    assert result["pareto_front"] == []


def test_refutation_records_an_exact_witness():
    values = [5, 3, 9, 2, 7, 1, 8, 4, 6, 0, 11, 13, 17, 19]
    result = generate_conjectures(_rows(lambda n: values[n], range(14)))
    refuted = [entry for entry in result["conjectures"] if entry.get("status") == "REFUTED"]
    assert refuted
    for entry in refuted:
        assert entry["refutation_witness"] is not None
        assert "point" in entry["refutation_witness"]


def test_a_survivor_never_carries_its_own_refutation_witness():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    for entry in result["conjectures"]:
        if entry.get("status") == "SURVIVED":
            assert entry["refutation_witness"] is None


# ---------------------------------------------------------------------------
# Prefix / holdout discipline
# ---------------------------------------------------------------------------


def test_holdout_is_never_empty_and_meets_the_declared_minimum():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    assert result["counts"]["holdout_rows"] >= SYSTEM_CAPS["min_holdout_rows"]
    assert result["counts"]["prefix_rows"] >= SYSTEM_CAPS["min_prefix_rows"]
    assert (
        result["counts"]["prefix_rows"] + result["counts"]["holdout_rows"]
        == len(result["public_rows"])
    )


def test_support_can_never_exceed_the_holdout_size():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    for entry in result["conjectures"]:
        if entry.get("status") in {"SURVIVED", "REFUTED"}:
            assert entry["support"] <= result["counts"]["holdout_rows"]


def test_too_few_rows_to_split_is_refused():
    with pytest.raises(ConjectureGenerationError):
        generate_conjectures(_rows(lambda n: n, range(5)))


# ---------------------------------------------------------------------------
# Ranking honesty
# ---------------------------------------------------------------------------


def test_pareto_front_is_reported_without_a_scalar_score():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    assert result["pareto_front"]
    assert result["claims"]["scalar_truth_or_probability_score"] is False
    for entry in result["conjectures"]:
        assert "score" not in entry
        assert "probability" not in entry
        assert "confidence" not in entry


def test_pareto_front_drops_dominated_conjectures():
    """Divisibility by 6 dominates congruence mod 2 at equal support and simplicity."""

    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    survived = [e for e in result["conjectures"] if e.get("status") == "SURVIVED"]
    assert len(result["pareto_front"]) < len(survived)


def test_every_conjecture_is_marked_unproved():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    for entry in result["conjectures"]:
        if entry.get("status") in {"SURVIVED", "REFUTED"}:
            assert entry["proved"] is False
    assert result["claims"]["conjecture_is_a_proof"] is False
    assert result["claims"]["survival_on_holdout_establishes_truth"] is False


def test_declared_statement_kinds_are_finite_and_bound():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    assert result["statement_kinds"] == list(STATEMENT_KINDS)
    assert result["counts"]["declared_statement_kinds"] == len(STATEMENT_KINDS)


def test_statement_kind_count_extended_deliberately():
    """v1 froze eight kinds; the current profile adds spectral_bias and
    holonomic_recurrence.  This count moves only by deliberate extension here."""

    assert len(STATEMENT_KINDS_V1) == 8
    assert len(STATEMENT_KINDS) == 10
    assert STATEMENT_KINDS[:8] == STATEMENT_KINDS_V1
    assert STATEMENT_KINDS[8:] == ("spectral_bias", "holonomic_recurrence")


# ---------------------------------------------------------------------------
# The v2 statement kinds: spectral bias and holonomic recurrence
# ---------------------------------------------------------------------------


def test_holonomic_recurrence_is_found_where_no_closed_form_exists():
    """Catalan numbers have no basis-ladder closed form and no constant-coefficient
    recurrence; the polynomial-coefficient recurrence is the real statement, found
    directly instead of only through the derived-ratio route."""

    from math import comb

    catalan = [comb(2 * n, n) // (n + 1) for n in range(24)]
    result = generate_conjectures(_rows(lambda n: catalan[n], range(24)))
    kinds = _kinds(result, "SURVIVED")
    assert "holonomic_recurrence" in kinds
    assert "closed_form" not in kinds
    assert "linear_recurrence" not in kinds
    entry = next(
        item for item in result["conjectures"] if item["kind"] == "holonomic_recurrence"
    )
    assert entry["statement"] == "(n + 2)*a(n+1) - (4*n + 2)*a(n) = 0 for n >= 0"
    assert entry["proved"] is False
    assert entry["support"] == result["counts"]["holdout_rows"]


def test_holonomic_conjecture_is_refuted_by_a_broken_suffix():
    from math import comb

    catalan = [comb(2 * n, n) // (n + 1) for n in range(24)]
    catalan[22] += 1  # deep inside the holdout
    result = generate_conjectures(_rows(lambda n: catalan[n], range(24)))
    entry = next(
        item for item in result["conjectures"] if item["kind"] == "holonomic_recurrence"
    )
    assert entry["status"] == "REFUTED"
    assert entry["refutation_witness"]["residual"]["numerator"] != 0


def test_spectral_bias_needs_a_wide_enough_prefix():
    """Fourteen rows leave an eight-row prefix: far below the declared spectral
    floor, so the kind must abstain rather than read tea leaves."""

    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    entry = next(item for item in result["conjectures"] if item["kind"] == "spectral_bias")
    assert entry["status"] == "NOT_PROPOSED"


@pytest.mark.empirical_validation
def test_spectral_bias_survives_on_the_ulam_window():
    """The exact 64-row Ulam window whose shipped receipt declared
    statement_kinds_too_weak now yields a surviving Steinerberger-form statement."""

    from sigma_theory_compiler.dozen_unsolved_progress_campaign import _ulam_terms

    terms = _ulam_terms(64)
    result = generate_conjectures(
        [{"point": index + 1, "value": value} for index, value in enumerate(terms)]
    )
    entry = next(item for item in result["conjectures"] if item["kind"] == "spectral_bias")
    assert entry["status"] == "SURVIVED"
    assert entry["peak_is_not_proof"] is True
    assert entry["statement"].startswith("cos(2.56")
    holonomic = next(
        item for item in result["conjectures"] if item["kind"] == "holonomic_recurrence"
    )
    assert holonomic["status"] == "NOT_PROPOSED"  # honest: Ulam is not known holonomic


# ---------------------------------------------------------------------------
# Frozen statement-kind profiles
# ---------------------------------------------------------------------------


def test_v1_profile_still_generates_and_replays():
    rows = _rows(lambda n: 6 * n + 12, range(14))
    result = generate_conjectures(rows, statement_kinds=STATEMENT_KINDS_V1)
    assert result["statement_kinds"] == list(STATEMENT_KINDS_V1)
    assert result["counts"]["declared_statement_kinds"] == 8
    assert {entry["kind"] for entry in result["conjectures"]} == set(STATEMENT_KINDS_V1)
    assert "spectral_min_prefix_rows" not in result["system_caps"]
    validate_result(result)


def test_free_form_statement_kind_subsets_are_refused():
    rows = _rows(lambda n: 6 * n + 12, range(14))
    with pytest.raises(ConjectureGenerationError):
        generate_conjectures(rows, statement_kinds=("sign", "monotonicity"))


def test_receipt_declaring_an_unknown_profile_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["statement_kinds"] = list(STATEMENT_KINDS) + ["astrology"]
    with pytest.raises(ConjectureGenerationError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_result_is_deterministic_and_replays():
    rows = _rows(lambda n: 6 * n + 12, range(14))
    first = generate_conjectures(rows)
    assert first == generate_conjectures(rows)
    validate_result(first)


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["decision"] = "NONE_SURVIVED"
    with pytest.raises(ConjectureGenerationError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


def test_claim_boundary_is_bound_into_every_result():
    result = generate_conjectures(_rows(lambda n: 6 * n + 12, range(14)))
    assert result["claims"] == CLAIMS
    assert "not proof" in result["scope"] or "not proof" in result["scope"].lower()
