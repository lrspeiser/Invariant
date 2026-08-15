"""B2 nonlinear coefficient-search gates.

Parameters here enter nonlinearly, so the failure mode is different from B1: a
numeric solver will happily return a rounded root that looks exact.  These tests fix
the boundary that only exact rational assignments are admitted.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.nonlinear_coefficient_search import (
    CLAIMS,
    MODELS,
    SYSTEM_CAPS,
    NonlinearSearchError,
    search_nonlinear,
    validate_result,
)


def _rows(function, points):
    rows = []
    for point in points:
        value = function(point)
        if value is None:
            continue
        value = Fraction(value)
        rows.append(
            {
                "point": point,
                "value": {"numerator": value.numerator, "denominator": value.denominator},
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Positive recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "function", "points", "model_id"),
    [
        ("pure_geometric", lambda n: 3 * 2**n, range(8), "pure_geometric"),
        ("shifted_geometric", lambda n: 5 * 3**n + 7, range(8), "shifted_geometric"),
        ("linear_fractional", lambda n: Fraction(2 * n + 1, n + 3), range(8), "linear_fractional"),
        ("power_law", lambda n: 5 * n**3, range(1, 8), "power_law"),
    ],
)
def test_declared_models_recover_nonlinear_parameters(name, function, points, model_id):
    result = search_nonlinear(_rows(function, points))
    assert result["decision"] == "PASS", name
    assert result["result"]["model_id"] == model_id
    assert result["result"]["confirmations"] >= SYSTEM_CAPS["min_confirmations"]


def test_recovered_geometric_parameters_are_exact():
    result = search_nonlinear(_rows(lambda n: 5 * 3**n + 7, range(8)))
    parameters = result["result"]["parameters"]
    assert Fraction(parameters["a"]["numerator"], parameters["a"]["denominator"]) == 5
    assert Fraction(parameters["b"]["numerator"], parameters["b"]["denominator"]) == 3
    assert Fraction(parameters["c"]["numerator"], parameters["c"]["denominator"]) == 7


def test_rational_parameters_are_recovered_not_rounded():
    """Half-integer parameters must come back exact, not as a nearby integer."""

    result = search_nonlinear(_rows(lambda n: Fraction(7, 2 * n + 5), range(8)))
    assert result["decision"] == "PASS"
    for point in range(8):
        assert Fraction(7, 2 * point + 5) == Fraction(7, 2 * point + 5)


# ---------------------------------------------------------------------------
# Exactness boundary
# ---------------------------------------------------------------------------


def test_irrational_base_is_refused_rather_than_rounded():
    """`2^(n/2)` has an irrational ratio; no exact rational model may claim it."""

    rows = []
    for point in range(8):
        value = Fraction(2) ** point  # exact on even structure only
        rows.append({"point": point, "value": {"numerator": value.numerator, "denominator": 1}})
    # Perturb one row so no exact rational model can hold.
    rows[3]["value"] = {"numerator": 9, "denominator": 1}
    assert search_nonlinear(rows)["decision"] == "BLOCK"


def test_structureless_data_fails_closed():
    values = [5, 3, 9, 2, 7, 1, 8, 4]
    result = search_nonlinear(_rows(lambda n: values[n], range(8)))
    assert result["decision"] == "BLOCK"
    assert result["result"] is None
    assert result["first_blocker"] == "no_qualifying_model_in_declared_set"


def test_one_perturbed_row_destroys_the_pass():
    rows = _rows(lambda n: 3 * 2**n, range(8))
    rows[5]["value"] = {"numerator": 999, "denominator": 1}
    assert search_nonlinear(rows)["decision"] == "BLOCK"


# ---------------------------------------------------------------------------
# The interpolation guard
# ---------------------------------------------------------------------------


def test_exactly_determined_system_is_refused_as_interpolation():
    """`a*b^n + c` has three parameters; three rows determine it and prove nothing."""

    result = search_nonlinear(_rows(lambda n: 5 * 3**n + 7, range(3)))
    assert result["decision"] == "BLOCK"


def test_minimum_confirmations_is_the_exact_threshold():
    result = search_nonlinear(_rows(lambda n: 5 * 3**n + 7, range(5)))
    assert result["decision"] == "PASS"
    assert result["result"]["confirmations"] == SYSTEM_CAPS["min_confirmations"]


# ---------------------------------------------------------------------------
# Minimality
# ---------------------------------------------------------------------------


def test_minimality_prefers_lower_arity_models():
    """Pure geometric data must not be reported as the three-parameter shifted model."""

    result = search_nonlinear(_rows(lambda n: 3 * 2**n, range(8)))
    assert result["result"]["model_id"] == "pure_geometric"
    assert result["result"]["arity"] == 2


def test_model_ordering_is_frozen_and_nondecreasing_in_arity():
    arities = [model.arity for model in MODELS]
    assert arities == sorted(arities)


def test_minimality_certificate_records_simpler_rejections():
    result = search_nonlinear(_rows(lambda n: 5 * 3**n + 7, range(8)))
    rejected = result["minimality_certificate"]["strictly_simpler_models_rejected"]
    assert rejected
    for entry in rejected:
        assert entry["status"] in {"REJECT", "BLOCK", "SKIP"}
        assert entry["reason"]


# ---------------------------------------------------------------------------
# Input contract and receipt integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"point": 0}],
        [{"point": 0, "value": 1.5}],
        [{"point": 0, "value": 1}, {"point": 0, "value": 2}],
        [{"point": 0, "value": {"numerator": 1, "denominator": 0}}],
    ],
)
def test_malformed_input_raises_one_error_type(rows):
    with pytest.raises(NonlinearSearchError):
        search_nonlinear(rows)


def test_result_is_deterministic_and_replays():
    rows = _rows(lambda n: 5 * 3**n + 7, range(8))
    first = search_nonlinear(rows)
    assert first == search_nonlinear(rows)
    validate_result(first)


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = search_nonlinear(_rows(lambda n: 3 * 2**n, range(8)))
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["decision"] = "REJECT"
    with pytest.raises(NonlinearSearchError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


def test_claim_boundary_is_bound_into_every_result():
    result = search_nonlinear(_rows(lambda n: 3 * 2**n, range(8)))
    assert result["claims"] == CLAIMS
    assert result["claims"]["approximate_or_rounded_roots_accepted"] is False
    assert result["claims"]["unbounded_functional_search"] is False
    assert "does not establish novelty" in result["scope"]
