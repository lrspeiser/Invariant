"""B1 basis-synthesis gates.

The load-bearing tests are the negative ones.  Recovering a formula is easy to fake
by interpolation; these fix the boundary that makes a PASS mean something.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

import pytest

from sigma_theory_compiler.basis_synthesis import (
    CLAIMS,
    LADDER,
    SYSTEM_CAPS,
    BasisSynthesisError,
    synthesize_basis,
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
# Positive recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "function", "points", "family_id"),
    [
        ("triangular", lambda n: n * (n + 1) // 2, range(10), "polynomial_2"),
        ("cubic_recurrence", lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(10), "polynomial_3"),
        ("geometric", lambda n: 3 * 2**n, range(10), "geometric_2"),
        ("harmonic", _harmonic, range(1, 10), "harmonic"),
        ("alternating", lambda n: (-1) ** n * n, range(10), "alternating_2"),
        ("factorial", lambda n: factorial(n) + 1, range(9), "factorial"),
        ("binomial", lambda n: comb(n, 2), range(10), "binomial_2"),
        ("shifted_reciprocal", lambda n: Fraction(1, n + 1), range(10), "shifted_reciprocal_1"),
    ],
)
def test_declared_ladder_recovers_structure(name, function, points, family_id):
    result = synthesize_basis(_rows(function, points))
    assert result["decision"] == "PASS", name
    assert result["result"]["family_id"] == family_id
    assert result["result"]["confirmations"] >= SYSTEM_CAPS["min_confirmations"]


def test_recovered_coefficients_reproduce_every_public_row():
    """A PASS must reproduce the data exactly, not approximately."""

    result = synthesize_basis(_rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(10)))
    coefficients = [
        Fraction(item["numerator"], item["denominator"])
        for item in result["result"]["coefficients"]
    ]
    for point in range(10):
        predicted = sum(
            coefficient * Fraction(point) ** degree
            for degree, coefficient in enumerate(coefficients)
        )
        assert predicted == 2 * point**3 + 2 * point**2 + point + 7


# ---------------------------------------------------------------------------
# The interpolation guard
# ---------------------------------------------------------------------------


def test_exactly_determined_system_is_refused_as_interpolation():
    """k points and a k-term basis is interpolation; it must never PASS."""

    result = synthesize_basis(_rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(4)))
    assert result["decision"] == "BLOCK"
    assert result["first_blocker"] == "no_qualifying_basis_in_declared_ladder"


def test_single_confirmation_is_still_refused():
    result = synthesize_basis(_rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(5)))
    assert result["decision"] == "BLOCK"


def test_minimum_confirmations_is_the_exact_threshold():
    result = synthesize_basis(_rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(6)))
    assert result["decision"] == "PASS"
    assert result["result"]["confirmations"] == SYSTEM_CAPS["min_confirmations"]


def test_structureless_data_fails_closed():
    """No ladder entry may be stretched to cover arbitrary values."""

    values = [5, 3, 9, 2, 7, 1, 8, 4, 6, 0]
    result = synthesize_basis(_rows(lambda n: values[n], range(10)))
    assert result["decision"] == "BLOCK"
    assert result["result"] is None
    assert result["counts"]["entries_examined"] == len(LADDER)


def test_one_perturbed_row_destroys_the_pass():
    """Structure plus a single wrong value is not structure."""

    rows = _rows(lambda n: n * (n + 1) // 2, range(10))
    rows[7]["value"] = {"numerator": 999, "denominator": 1}
    assert synthesize_basis(rows)["decision"] == "BLOCK"


# ---------------------------------------------------------------------------
# Minimality
# ---------------------------------------------------------------------------


def test_minimality_prefers_the_simplest_qualifying_entry():
    result = synthesize_basis(_rows(lambda n: 3 * n + 4, range(10)))
    assert result["result"]["family_id"] == "polynomial_1"
    assert result["result"]["term_count"] == 2


def test_minimality_certificate_records_every_simpler_rejection():
    """The claim 'simplest' must be backed by exact rejections, not search order."""

    result = synthesize_basis(_rows(lambda n: 2 * n**3 + 2 * n**2 + n + 7, range(10)))
    rejected = result["minimality_certificate"]["strictly_simpler_entries_rejected"]
    accepted_terms = result["result"]["term_count"]
    assert rejected, "a degree-3 recovery must reject simpler entries"
    for entry in rejected:
        assert entry["status"] in {"REJECT", "BLOCK", "SKIP"}
        assert entry["reason"]
        assert entry["term_count"] <= accepted_terms


def test_ladder_ordering_is_frozen_and_nondecreasing_in_term_count():
    counts = [len(entry["terms"]) for entry in LADDER]
    assert counts == sorted(counts)


# ---------------------------------------------------------------------------
# Input caps and fail-closed parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"point": 0}],
        [{"point": 0, "value": 1, "extra": 2}],
        [{"point": True, "value": 1}],
        [{"point": 0, "value": 1}, {"point": 0, "value": 2}],
        [{"point": 0, "value": {"numerator": 1, "denominator": 0}}],
        [{"point": 0, "value": 1.5}],
        [{"point": SYSTEM_CAPS["max_abs_point"] + 1, "value": 1}],
    ],
)
def test_malformed_input_raises_rather_than_guessing(rows):
    with pytest.raises(BasisSynthesisError):
        synthesize_basis(rows)


def test_row_cap_is_enforced():
    rows = [{"point": point, "value": point} for point in range(SYSTEM_CAPS["max_rows"] + 1)]
    with pytest.raises(BasisSynthesisError):
        synthesize_basis(rows)


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_result_is_deterministic_and_replays():
    rows = _rows(lambda n: n * (n + 1) // 2, range(10))
    first = synthesize_basis(rows)
    assert first == synthesize_basis(rows)
    validate_result(first)


def test_reseal_after_tamper_fails_closed():
    result = synthesize_basis(_rows(lambda n: n * (n + 1) // 2, range(10)))
    tampered = dict(result)
    tampered["decision"] = "REJECT"
    with pytest.raises(BasisSynthesisError):
        validate_result(tampered)


def test_resealed_body_still_fails_replay():
    """Re-hashing a tampered body must not launder it."""

    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = synthesize_basis(_rows(lambda n: n * (n + 1) // 2, range(10)))
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["decision"] = "REJECT"
    with pytest.raises(BasisSynthesisError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


def test_claim_boundary_is_bound_into_every_result():
    result = synthesize_basis(_rows(lambda n: n * (n + 1) // 2, range(10)))
    assert result["claims"] == CLAIMS
    assert result["claims"]["interpolation_accepted_as_discovery"] is False
    assert result["claims"]["unbounded_representation_invention"] is False
    assert "does not establish novelty" in result["scope"]
