"""Spectral-scan gates.

The scan's failure mode is astrology: peaks exist in any finite spectrum, so the
load-bearing tests are the negative controls (structureless data proposes nothing),
the exactness discipline (the lambda printed in a statement is exactly the lambda
that is counted, and every count is exact), and the empirical validation that the
declared brute-force sweep really does rediscover the one hidden frequency in the
literature's canonical example.
"""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.dozen_unsolved_progress_campaign import _ulam_terms
from sigma_theory_compiler.spectral_signal_scan import (
    CLAIMS,
    PROFILES,
    THRESHOLD_LADDER,
    SpectralScanError,
    cos_sign,
    lambda_decimal,
    propose_bias_from_prefix,
    scan_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
STEINERBERGER_LAMBDA = 2.5714474995


def _rows(values: list[int], start: int = 0) -> list[dict[str, object]]:
    return [{"point": start + index, "value": value} for index, value in enumerate(values)]


def _planted(count: int, frequency: float = 2.5) -> list[int]:
    """Integers whose phases `frequency * a_n` cluster at pi: a planted bias."""

    return [round((n + 0.5) * 2 * math.pi / frequency) for n in range(count)]


# ---------------------------------------------------------------------------
# Peak discovery and the Steinerberger statement form
# ---------------------------------------------------------------------------


def test_planted_frequency_is_recovered_with_a_bias_statement():
    proposals = propose_bias_from_prefix(_planted(28), "engine")
    assert proposals
    top = proposals[0]
    assert abs(Fraction(top["lambda"]) - Fraction(5, 2)) < Fraction(2, 100)
    assert top["statement"] is not None
    assert top["statement"].startswith(f"cos({top['lambda']} * a(n)) < 0 for at least ")
    assert top["prefix_negative_terms"] > top["prefix_terms"] // 2


def test_statement_lambda_is_exactly_the_tested_lambda():
    """The decimal in the statement parses back to the exact rational whose cos
    signs produced every count: no hidden higher-precision lambda."""

    top = propose_bias_from_prefix(_planted(28), "engine")[0]
    lam = Fraction(top["lambda"])
    assert lam == top["lambda_fraction"]
    negative = sum(
        1 for value in _planted(28) if cos_sign(lam, Fraction(value), 50) < 0
    )
    assert negative == top["prefix_negative_terms"]


def test_threshold_comes_from_the_declared_ladder_only():
    top = propose_bias_from_prefix(_planted(28), "engine")[0]
    threshold = top["threshold_fraction"]
    assert threshold in THRESHOLD_LADDER
    measured = Fraction(top["prefix_negative_terms"], top["prefix_terms"])
    assert measured >= threshold
    assert all(entry <= threshold for entry in THRESHOLD_LADDER if measured >= entry)


# ---------------------------------------------------------------------------
# Negative controls — structureless data must propose nothing
# ---------------------------------------------------------------------------


def test_linear_data_proposes_no_bias():
    assert propose_bias_from_prefix(list(range(1, 40)), "engine") == []


def test_random_data_proposes_no_bias():
    rng = random.Random(20260816)
    values = [rng.randrange(1, 10**6) for _ in range(38)]
    assert propose_bias_from_prefix(values, "engine") == []


def test_tiny_prefixes_cannot_reach_the_scan():
    with pytest.raises(SpectralScanError):
        scan_receipt(_rows(_planted(20)))  # prefix would fall below the floor


# ---------------------------------------------------------------------------
# Empirical validation against the literature
# ---------------------------------------------------------------------------


@pytest.mark.empirical_validation
def test_ulam_hidden_frequency_is_rediscovered_to_1e_minus_3():
    """Steinerberger (2017): cos(2.5714474995... * a_n) < 0 for almost all Ulam
    terms.  The dense declared sweep on a 360-term prefix must land within 1e-3.
    A matching peak is a rediscovery of a measured constant, not a proof."""

    terms = _ulam_terms(600)
    proposals = propose_bias_from_prefix(terms[:360], "dense")
    assert proposals
    top = proposals[0]
    assert abs(float(Fraction(top["lambda"])) - STEINERBERGER_LAMBDA) < 1e-3
    assert top["threshold_fraction"] == Fraction(49, 50)


@pytest.mark.empirical_validation
def test_ulam_bias_survives_the_holdout_suffix():
    receipt = scan_receipt(_rows(_ulam_terms(160), start=1), profile_name="engine")
    assert receipt["decision"] == "SPECTRAL_BIAS_SURVIVED"
    survivors = [peak for peak in receipt["peaks"] if peak["status"] == "SURVIVED"]
    assert survivors
    top = survivors[0]
    assert abs(float(Fraction(top["lambda"])) - STEINERBERGER_LAMBDA) < 5e-3
    assert top["holdout"]["negative_terms"] > top["holdout"]["rows"] * 9 // 10


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_receipt_is_deterministic_and_replays():
    rows = _rows(_planted(40))
    first = scan_receipt(rows, "planted-control", profile_name="engine")
    assert first == scan_receipt(rows, "planted-control", profile_name="engine")
    validate_receipt(first)
    assert first["claims"] == CLAIMS
    assert first["claims"]["peak_is_not_proof"] is True


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    receipt = scan_receipt(_rows(_planted(40)), "planted-control", profile_name="engine")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    survivor = next(p for p in body["peaks"] if p["status"] == "SURVIVED")
    survivor["holdout"]["negative_terms"] += 1
    with pytest.raises(SpectralScanError):
        validate_receipt({**body, "content_sha256": canonical_sha256(body)})


def test_lambda_decimal_rendering_is_exact():
    assert lambda_decimal(Fraction(5, 2)) == "2.500000000000"
    assert lambda_decimal(Fraction(2571447499516, 10**12)) == "2.571447499516"
    with pytest.raises(SpectralScanError):
        lambda_decimal(Fraction(1, 3))


def test_profiles_are_declared_and_finite():
    assert set(PROFILES) == {"engine", "dense"}
    dense = PROFILES["dense"]
    assert dense["grid_denominator"] == 100000
    assert dense["stop_k"] - dense["start_k"] + 1 == 313160  # the declared ~314k sweep


# ---------------------------------------------------------------------------
# Committed receipts (the real scans this module shipped)
# ---------------------------------------------------------------------------


def _committed(name: str) -> dict:
    return json.loads(
        (ROOT / "runs" / "math" / "spectral" / name).read_text(encoding="utf-8")
    )


@pytest.mark.empirical_validation
def test_committed_ulam_receipt_validates_and_matches_the_literature_lambda():
    receipt = _committed("ulam-signal-v1.json")
    validate_receipt(receipt)
    top = next(peak for peak in receipt["peaks"] if peak["status"] == "SURVIVED")
    assert abs(float(Fraction(top["lambda"])) - STEINERBERGER_LAMBDA) < 1e-3
    assert receipt["decision"] == "SPECTRAL_BIAS_SURVIVED"


def test_committed_side_scans_validate():
    for name in ("recaman-appearance-signal-v1.json", "untouchable-head-signal-v1.json"):
        receipt = _committed(name)
        validate_receipt(receipt)
        assert receipt["novelty"]["claimed"] is False
