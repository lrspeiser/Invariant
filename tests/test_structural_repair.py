"""B7 structural-repair gates.

Repair is the most dangerous capability in this stack: given enough freedom to change
structure, anything can be fitted.  The tests that matter are the ones proving the
repair budget is real and that a restricted-domain result is never sold as a global one.
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import pytest

from sigma_theory_compiler.basis_synthesis import synthesize_basis
from sigma_theory_compiler.structural_repair import (
    CLAIMS,
    RESTRICTIONS,
    SYSTEM_CAPS,
    StructuralRepairError,
    repair_structure,
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


def _harmonic(point: int) -> Fraction:
    return sum((Fraction(1, index) for index in range(1, point + 1)), Fraction(0))


# ---------------------------------------------------------------------------
# Basis union — structure the unrepaired search cannot reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "function", "points"),
    [
        ("polynomial_plus_geometric", lambda n: n**2 + 2**n, range(11)),
        ("factorial_plus_polynomial", lambda n: factorial(n) + n**2, range(9)),
    ],
)
def test_basis_union_recovers_cross_family_structure(name, function, points):
    rows = _rows(function, points)
    assert synthesize_basis(rows)["decision"] == "BLOCK", f"{name} must be out of reach for B1"
    result = repair_structure(rows)
    assert result["decision"] == "PASS"
    assert result["repair"]["strategy"] == "basis_union"
    assert len(result["repair"]["extended_from"]) == 2


def test_repair_is_not_invoked_when_unrepaired_search_succeeds():
    rows = _rows(lambda n: _harmonic(n) + n, range(1, 11))
    result = repair_structure(rows)
    assert result["decision"] == "PASS"
    assert result["repair"]["strategy"] == "none_required"
    assert result["unrepaired_decision"] == "PASS"


def test_union_result_reproduces_every_public_row():
    rows = _rows(lambda n: n**2 + 2**n, range(11))
    result = repair_structure(rows)
    confirmations = result["repair"]["confirmations"]
    assert confirmations >= SYSTEM_CAPS["min_confirmations"]
    assert result["repair"]["total_parameters"] + confirmations == len(rows)


# ---------------------------------------------------------------------------
# Domain restriction — the honesty boundary
# ---------------------------------------------------------------------------


def test_domain_restriction_recovers_a_restricted_theorem():
    exceptions = {0: 99, 1: -4, 2: 17}
    function = lambda n: exceptions.get(n, n**2)
    rows = _rows(function, range(12))
    assert synthesize_basis(rows)["decision"] == "BLOCK"
    result = repair_structure(rows)
    assert result["decision"] == "PASS"
    assert result["repair"]["strategy"] == "domain_restriction"
    assert result["repair"]["rows_excluded"] == 3


def test_restricted_result_always_carries_its_restriction():
    """A restricted result must never be presentable as a global identity."""

    exceptions = {0: 99, 1: -4, 2: 17}
    rows = _rows(lambda n: exceptions.get(n, n**2), range(12))
    repair = repair_structure(rows)["repair"]
    assert repair["restricted_domain"]
    assert repair["restriction_id"]
    assert repair["rows_excluded"] > 0


def test_claims_forbid_promoting_a_restricted_domain_to_a_global_claim():
    rows = _rows(lambda n: n**2 + 2**n, range(11))
    result = repair_structure(rows)
    assert result["claims"]["global_claim_from_restricted_domain"] is False
    assert result["claims"]["coefficient_only_repair"] is False
    assert "never promoted to" in result["scope"]


def test_declared_restrictions_are_finite_and_described():
    assert RESTRICTIONS
    for restriction in RESTRICTIONS:
        assert restriction["restriction_id"]
        assert restriction["description"]


# ---------------------------------------------------------------------------
# The repair budget
# ---------------------------------------------------------------------------


def test_repair_cannot_consume_the_confirming_evidence():
    """A union wide enough to interpolate the data must be refused."""

    rows = _rows(lambda n: n**2 + 2**n, range(5))
    result = repair_structure(rows)
    assert result["decision"] == "BLOCK"
    assert result["first_blocker"] == "no_declared_repair_recovered_structure"


def test_structureless_data_survives_every_declared_repair():
    values = [5, 3, 9, 2, 7, 1, 8, 4, 6, 0, 11, 13]
    result = repair_structure(_rows(lambda n: values[n], range(12)))
    assert result["decision"] == "BLOCK"
    assert result["repair"] is None
    assert result["counts"]["strategies_attempted"] == 2


def test_one_perturbed_row_inside_the_restricted_domain_still_fails():
    exceptions = {0: 99, 1: -4, 2: 17}
    rows = _rows(lambda n: exceptions.get(n, n**2), range(12))
    rows[8]["value"] = {"numerator": 12345, "denominator": 1}
    assert repair_structure(rows)["decision"] == "BLOCK"


# ---------------------------------------------------------------------------
# Input contract and receipt integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [[], [{"point": 0}], [{"point": 0, "value": 1.5}], [{"point": 0, "value": 1}, {"point": 0, "value": 2}]],
)
def test_malformed_input_raises_one_error_type(rows):
    with pytest.raises(StructuralRepairError):
        repair_structure(rows)


def test_result_is_deterministic_and_replays():
    rows = _rows(lambda n: n**2 + 2**n, range(11))
    first = repair_structure(rows)
    assert first == repair_structure(rows)
    validate_result(first)


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = repair_structure(_rows(lambda n: n**2 + 2**n, range(11)))
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["decision"] = "REJECT"
    with pytest.raises(StructuralRepairError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


def test_unrepaired_receipt_is_bound_for_provenance():
    result = repair_structure(_rows(lambda n: n**2 + 2**n, range(11)))
    assert result["unrepaired_decision"] == "BLOCK"
    assert len(result["unrepaired_receipt_sha256"]) == 64
    assert result["claims"] == CLAIMS
