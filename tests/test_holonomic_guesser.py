"""Holonomic-guesser gates.

The failure mode of a recurrence guesser is silent interpolation: enough free
coefficients fit anything.  These tests pin the controls (classical holonomic
sequences recovered exactly), the fail-closed behaviour on random data, the
interpolation guard (a cell whose fit plus mandatory holdout does not fit in the
data is skipped, never fitted), and receipt integrity.
"""

from __future__ import annotations

import random
from math import comb, factorial

import pytest

from sigma_theory_compiler.holonomic_guesser import (
    CLAIMS,
    LADDER,
    HolonomicGuessError,
    guess_receipt,
    guess_recurrence,
    render_operator,
    validate_receipt,
)


def _catalan(count: int) -> list[int]:
    return [comb(2 * n, n) // (n + 1) for n in range(count)]


def _derangements(count: int) -> list[int]:
    values = [1, 0]
    while len(values) < count:
        n = len(values) - 1
        values.append(n * (values[-1] + values[-2]))
    return values


def _motzkin(count: int) -> list[int]:
    values = [1, 1]
    while len(values) < count:
        n = len(values) - 1
        values.append(((2 * n + 3) * values[-1] + 3 * n * values[-2]) // (n + 3))
    return values


def _fibonacci(count: int) -> list[int]:
    values = [0, 1]
    while len(values) < count:
        values.append(values[-1] + values[-2])
    return values


def _rows(values: list[int], start: int = 0) -> list[dict[str, object]]:
    return [{"point": start + index, "value": value} for index, value in enumerate(values)]


# ---------------------------------------------------------------------------
# Controls: classical holonomic sequences recovered exactly
# ---------------------------------------------------------------------------


def test_catalan_direct_recurrence_is_recovered():
    result = guess_recurrence(_catalan(24))
    assert result["decision"] == "OPERATOR_FOUND"
    operator = result["operator"]
    assert operator["statement"] == "(n + 2)*a(n+1) - (4*n + 2)*a(n) = 0 for n >= 0"
    assert (operator["order"], operator["degree"]) == (1, 1)
    assert operator["coefficients"] == [[-2, -4], [2, 1]]


def test_factorial_is_recovered_at_the_first_cell():
    result = guess_recurrence([factorial(n) for n in range(20)])
    assert result["decision"] == "OPERATOR_FOUND"
    operator = result["operator"]
    assert operator["statement"] == "a(n+1) - (n + 1)*a(n) = 0 for n >= 0"
    assert (operator["order"], operator["degree"]) == (1, 1)


def test_derangements_holonomic_form_is_recovered():
    """`a(n) = n*a(n-1) + (-1)^n` has the homogeneous holonomic equivalent
    `a(n+2) = (n+1)*(a(n+1) + a(n))`."""

    result = guess_recurrence(_derangements(22))
    assert result["decision"] == "OPERATOR_FOUND"
    operator = result["operator"]
    assert operator["statement"] == "a(n+2) - (n + 1)*a(n+1) - (n + 1)*a(n) = 0 for n >= 0"
    assert (operator["order"], operator["degree"]) == (2, 1)


def test_motzkin_is_recovered():
    result = guess_recurrence(_motzkin(22))
    assert result["decision"] == "OPERATOR_FOUND"
    operator = result["operator"]
    assert (
        operator["statement"]
        == "(n + 4)*a(n+2) - (2*n + 5)*a(n+1) - (3*n + 3)*a(n) = 0 for n >= 0"
    )


def test_fibonacci_is_recovered_as_the_degenerate_constant_coefficient_case():
    """The (2,1) cell's nullspace contains Q(n)-multiples of the minimal operator;
    polynomial-content normalization must collapse them to the constant form."""

    result = guess_recurrence(_fibonacci(24))
    assert result["decision"] == "OPERATOR_FOUND"
    operator = result["operator"]
    assert operator["statement"] == "a(n+2) - a(n+1) - a(n) = 0 for n >= 0"
    assert operator["coefficients"] == [[-1], [-1], [1]]
    assert operator["degree"] == 1  # the search cell; the operator itself is degree 0


def test_verification_consumes_every_remaining_term():
    result = guess_recurrence(_catalan(30))
    operator = result["operator"]
    assert operator["fitted_equations"] == 4
    assert operator["verified_equations"] == (30 - 1) - 4


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------


def test_random_sequence_yields_no_annihilator_through_the_whole_ladder():
    rng = random.Random(20260816)
    values = [rng.randrange(1, 10**6) for _ in range(40)]
    result = guess_recurrence(values)
    assert result["decision"] == "NO_ANNIHILATOR"
    assert result["operator"] is None
    assert len(result["ladder_trace"]) == len(LADDER)
    for cell in result["ladder_trace"]:
        assert cell["status"] != "ACCEPTED"


def test_interpolation_guard_skips_cells_instead_of_fitting_them():
    """Nine terms cannot host any cell's fit plus the mandatory holdout."""

    result = guess_recurrence(_catalan(9))
    assert result["decision"] == "NO_ANNIHILATOR"
    for cell in result["ladder_trace"]:
        assert cell["status"] == "SKIPPED_INSUFFICIENT_TERMS"
        assert cell["equations_available"] < cell["equations_required"]


def test_too_few_terms_is_refused_outright():
    with pytest.raises(HolonomicGuessError):
        guess_recurrence(_catalan(7))


def test_floats_are_refused():
    with pytest.raises(HolonomicGuessError):
        guess_recurrence([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


# ---------------------------------------------------------------------------
# Normalization and rendering
# ---------------------------------------------------------------------------


def test_operator_is_content_free_with_positive_leading_coefficient():
    for values in (_catalan(24), _derangements(22), _motzkin(22), _fibonacci(24)):
        operator = guess_recurrence(values)["operator"]
        coefficients = operator["coefficients"]
        top = coefficients[-1]
        assert top and top[-1] > 0
        from math import gcd

        divisor = 0
        for poly in coefficients:
            for coefficient in poly:
                divisor = gcd(divisor, abs(coefficient))
        assert divisor == 1


def test_start_index_shifts_the_rendered_statement():
    result = guess_recurrence(_catalan(24)[1:], start_index=1)
    assert result["decision"] == "OPERATOR_FOUND"
    assert result["operator"]["statement"].endswith("for n >= 1")


def test_latex_rendering_is_emitted():
    operator = guess_recurrence(_catalan(24))["operator"]
    assert operator["latex"] == r"(n + 2)\,a(n+1) - (4 n + 2)\,a(n) = 0 for n \ge 0"


def test_render_operator_handles_mixed_sign_polynomials():
    rendered = render_operator([[3, -1], [0, 2]], 0)
    assert rendered == "(2*n)*a(n+1) + (-n + 3)*a(n) = 0 for n >= 0"


# ---------------------------------------------------------------------------
# The declared ladder
# ---------------------------------------------------------------------------


def test_ladder_is_exactly_the_declared_cells_in_order():
    assert LADDER == (
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (3, 3),
        (2, 4),
        (4, 2),
    )


def test_first_accepting_cell_wins_and_search_stops():
    trace = guess_recurrence(_catalan(24))["ladder_trace"]
    assert [cell["status"] for cell in trace] == ["ACCEPTED"]


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_receipt_is_deterministic_and_replays():
    rows = _rows(_catalan(24))
    first = guess_receipt(rows, "catalan-control")
    assert first == guess_receipt(rows, "catalan-control")
    validate_receipt(first)
    assert first["claims"] == CLAIMS
    assert first["claims"]["guess_is_a_proof"] is False


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    receipt = guess_receipt(_rows(_catalan(24)), "catalan-control")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    body["decision"] = "NO_ANNIHILATOR"
    with pytest.raises(HolonomicGuessError):
        validate_receipt({**body, "content_sha256": canonical_sha256(body)})


def test_nonconsecutive_points_are_refused():
    rows = _rows(_catalan(24))
    rows[5]["point"] = 99
    with pytest.raises(HolonomicGuessError):
        guess_receipt(rows)


# ---------------------------------------------------------------------------
# Committed receipts (the real guesses this module shipped)
# ---------------------------------------------------------------------------


def _committed(relative: str) -> dict:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "runs" / "math" / "holonomic" / relative).read_text(encoding="utf-8")
    )


def test_committed_ulam_and_recaman_guesses_are_honestly_empty():
    for name in ("ulam-holonomic-v1.json", "recaman-appearance-holonomic-v1.json"):
        receipt = _committed(name)
        validate_receipt(receipt)
        assert receipt["decision"] == "NO_ANNIHILATOR"
        assert receipt["operator"] is None


def test_committed_catalan_world_receipt_states_the_direct_recurrence():
    """The sealed blind campaign reached Catalan only through the declared
    adjacent-term-ratio route; this receipt records the expressiveness jump: the
    same public rows now yield the recurrence directly."""

    receipt = _committed("solved-dozen/catalan_ratio.json")
    validate_receipt(receipt)
    assert receipt["decision"] == "OPERATOR_FOUND"
    assert (
        receipt["operator"]["statement"]
        == "(n + 2)*a(n+1) - (4*n + 2)*a(n) = 0 for n >= 0"
    )
    assert "expressiveness comparison" in receipt["sequence_label"]


def test_committed_world_receipts_all_validate_and_find_operators():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "runs" / "math" / "holonomic" / "solved-dozen").glob("*.json"))
    assert len(paths) == 12
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(receipt)
        assert receipt["decision"] == "OPERATOR_FOUND"
