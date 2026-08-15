"""Collatz case study — an engine capability probe against an unsolved problem.

The Collatz conjecture (every positive integer reaches 1 under `n -> n/2` for even `n`
and `n -> 3n+1` for odd `n`) has been open since 1937.  **Nothing in this file solves it
or claims progress toward it.**  It is used because it is a genuine open problem that
emits integer sequences, which makes it a good probe for what the engine cannot see.

The first run failed completely: B1 BLOCK, B7 BLOCK, and B3 proposed nothing at all.
That was correct — the total stopping time has no known closed form — but it was also
*uninformative*, because two elementary true statements about the same data were sitting
right there and the engine was blind to both:

* `sigma(2^k) = k`, which needs a sparse sub-domain plus reindexing by the exponent;
* `sigma(2n) = sigma(n) + 1`, which relates a term to its multiplicative-index partner.

These tests pin both discoveries, and pin the boundary that neither one is a result
about the conjecture itself.
"""

from __future__ import annotations

import pytest

from sigma_theory_compiler.basis_synthesis import synthesize_basis
from sigma_theory_compiler.conjecture_generation import generate_conjectures
from sigma_theory_compiler.structural_repair import repair_structure


def _stopping_time(value: int) -> int:
    steps = 0
    while value != 1:
        value = value // 2 if value % 2 == 0 else 3 * value + 1
        steps += 1
    return steps


def _rows(limit: int) -> list[dict[str, object]]:
    return [
        {"point": point, "value": {"numerator": _stopping_time(point), "denominator": 1}}
        for point in range(1, limit + 1)
    ]


# ---------------------------------------------------------------------------
# The honest baseline: the hard part stays hard
# ---------------------------------------------------------------------------


def test_no_global_closed_form_is_ever_claimed():
    """Collatz stopping time has no known closed form; the engine must not invent one."""

    result = synthesize_basis(_rows(30))
    assert result["decision"] == "BLOCK"
    assert result["result"] is None


def test_unrestricted_repair_still_finds_no_global_law():
    result = repair_structure(_rows(30))
    repair = result["repair"]
    # Any PASS must be a restricted-domain statement, never a global one.
    if result["decision"] == "PASS":
        assert repair["strategy"] == "domain_restriction"
        assert repair["rows_excluded"] > 0


# ---------------------------------------------------------------------------
# Gap A, fixed: sparse sub-domain with reindexing
# ---------------------------------------------------------------------------


def test_powers_of_two_identity_is_discovered():
    """sigma(2^k) = k, found by restricting to a sparse set and reindexing by k."""

    result = repair_structure(_rows(64))
    assert result["decision"] == "PASS"
    repair = result["repair"]
    assert repair["strategy"] == "domain_restriction"
    assert repair["restriction_id"] == "geometric_index_2"
    assert repair["reindexed"] is True
    assert repair["index_variable"] == "k"
    assert repair["expression"] == "k^1"


def test_powers_of_two_identity_is_holdout_confirmed():
    repair = repair_structure(_rows(64))["repair"]
    assert repair["confirmations"] >= 2
    assert repair["rows_excluded"] > repair["rows_in_restricted_domain"]


def test_reindexed_result_declares_that_it_is_reindexed():
    """A statement about k must never be readable as a statement about n."""

    repair = repair_structure(_rows(64))["repair"]
    assert repair["reindexed"] is True
    assert "restated in k" in repair["restricted_domain"]


# ---------------------------------------------------------------------------
# Gap B, fixed: multiplicative-index relation
# ---------------------------------------------------------------------------


def test_halving_relation_is_discovered():
    """sigma(2n) = sigma(n) + 1, a relation no previous statement kind could express."""

    result = generate_conjectures(_rows(64))
    assert result["decision"] == "PROPOSED"
    scaling = next(
        entry
        for entry in result["conjectures"]
        if entry.get("kind") == "index_scaling_relation"
    )
    assert scaling["status"] == "SURVIVED"
    assert scaling["statement"] == "a(2n) = (1)*a(n) + (1)"


def test_halving_relation_survives_many_holdout_confirmations():
    result = generate_conjectures(_rows(64))
    scaling = next(
        entry for entry in result["conjectures"] if entry.get("kind") == "index_scaling_relation"
    )
    assert scaling["support"] >= 20


def test_index_scaling_is_refuted_where_it_does_not_hold():
    """The relation must be falsifiable, not a formality that always survives."""

    rows = [
        {"point": point, "value": {"numerator": point * point, "denominator": 1}}
        for point in range(1, 33)
    ]
    # a(2n) = 4a(n) holds for n^2, so perturb one scaled partner to force refutation.
    rows[23]["value"] = {"numerator": 999999, "denominator": 1}
    result = generate_conjectures(rows)
    scaling = [
        entry
        for entry in result["conjectures"]
        if entry.get("kind") == "index_scaling_relation"
    ]
    if scaling and scaling[0].get("status") == "SURVIVED":
        pytest.fail("a perturbed scaled partner must not survive")


def test_index_scaling_recovers_a_nonunit_multiplier():
    """`a(n) = n^2` satisfies a(2n) = 4a(n); the multiplier must be found exactly."""

    rows = [
        {"point": point, "value": {"numerator": point * point, "denominator": 1}}
        for point in range(1, 33)
    ]
    result = generate_conjectures(rows)
    scaling = next(
        entry for entry in result["conjectures"] if entry.get("kind") == "index_scaling_relation"
    )
    assert scaling["statement"] == "a(2n) = (4)*a(n)"
    assert scaling["status"] == "SURVIVED"


# ---------------------------------------------------------------------------
# The claim boundary
# ---------------------------------------------------------------------------


def test_nothing_here_is_marked_proved():
    """Both discoveries are empirical. Neither carries a proof certificate."""

    result = generate_conjectures(_rows(64))
    for entry in result["conjectures"]:
        if entry.get("status") in {"SURVIVED", "REFUTED"}:
            assert entry["proved"] is False
    assert result["claims"]["survival_on_holdout_establishes_truth"] is False


def test_engine_makes_no_termination_claim():
    """The open part of Collatz is termination. Neither discovery addresses it.

    `sigma(2n) = sigma(n) + 1` presupposes that both stopping times exist; it says
    nothing about whether they exist for every n. No statement kind in the engine can
    express a termination claim, so none can accidentally imply one.
    """

    result = generate_conjectures(_rows(64))
    statements = " ".join(
        entry.get("statement", "") for entry in result["conjectures"]
    ).lower()
    for forbidden in ("terminat", "halts", "reaches 1", "for all n"):
        assert forbidden not in statements
