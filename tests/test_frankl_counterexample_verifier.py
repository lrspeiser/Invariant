from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.frankl_counterexample_verifier import (
    FranklVerifierError,
    validate_receipt,
    verify_family,
)

REFERENCE_FAMILY = [
    [0, 1], [0, 1, 3], [2, 3], [0, 2, 3], [0, 1, 2, 3], [1, 3, 4],
    [0, 1, 3, 4], [2, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4],
    [0, 1, 5], [0, 2, 5], [0, 1, 2, 5], [0, 1, 3, 5], [0, 2, 3, 5],
    [0, 1, 2, 3, 5], [4, 5], [1, 4, 5], [0, 1, 4, 5], [2, 4, 5],
    [0, 2, 4, 5], [1, 2, 4, 5], [0, 1, 2, 4, 5], [1, 3, 4, 5],
    [0, 1, 3, 4, 5], [2, 3, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
]


def test_reference_family_is_exact_counterexample():
    receipt = verify_family(REFERENCE_FAMILY)
    validate_receipt(receipt)

    assert receipt["family_size"] == 30
    assert receipt["union_closed"] is True
    assert receipt["delta"] == 19
    assert set(receipt["degrees"].values()) == {19}
    assert set(receipt["residual_delta"].values()) == {10}
    assert receipt["exact_counterexample_valid"] is True


def test_six_set_llm_proposal_is_rejected_by_exact_residual_count():
    universe = set(range(1, 6))
    family = [sorted(universe - {x}) for x in universe] + [sorted(universe)]
    receipt = verify_family(family)

    assert receipt["family_size"] == 6
    assert receipt["delta"] == 5
    assert set(receipt["residual_delta"].values()) == {1}
    assert receipt["exact_counterexample_valid"] is False


def test_non_union_closed_family_is_rejected():
    receipt = verify_family([[1, 2], [2, 3]])

    assert receipt["union_closed"] is False
    assert receipt["missing_unions"] == [[1, 2, 3]]
    assert receipt["exact_counterexample_valid"] is False


def test_tampered_receipt_fails_closed():
    receipt = verify_family(REFERENCE_FAMILY)
    tampered = copy.deepcopy(receipt)
    tampered["delta"] = 18

    with pytest.raises(FranklVerifierError, match="seal"):
        validate_receipt(tampered)
