"""Gates for the outward-rounded interval threshold certifier.

The module exists to remove one failure mode: an fp64 screening verdict whose margin
is smaller than fp64 rounding error.  The tests therefore pin the three-way verdict
machinery with controls on both sides of the boundary and ON the boundary — a clear
survivor certifies, a clear violator certified-fails, and an exact threshold tie must
come back unresolved_straddle rather than pass or fail.  Receipt determinism, tamper
fail-closure (including a re-sealed tamper), source-byte binding, and the claim
boundary are sealed the same way as the screen's own gates.  All CPU, no GPU needed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import mpmath as mp
import pytest
from mpmath.libmp import mpf_le

from sigma_theory_compiler.gpu_baryonic_interpolation_screen import (
    SCREEN_CONFIG,
    _disk_gbar,
    build_probe_grid,
    encode_candidate,
    verify_candidate_exact,
)
from sigma_theory_compiler.interval_threshold_certifier import (
    CERTIFIED_FAIL,
    CERTIFIED_PASS,
    CLAIMS,
    DISK_GRID_PATH,
    IV_DPS,
    RESULT_SCHEMA,
    UNRESOLVED,
    IntervalCertifierError,
    build_interval_grid,
    certify_coefficients,
    certify_receipt,
    main,
    validate_result,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

RECEIPT_PATH = Path(__file__).resolve().parents[1] / "runs/gpu-baryonic-screen/billion-v1.json"

CRITERIA_NAMES = {
    "defined",
    "newton_near",
    "newton_far",
    "monotone",
    "flat_disk_0",
    "flat_disk_1",
    "flat_disk_2",
    "btfr_slope",
}


@pytest.fixture(scope="module")
def grid():
    return build_interval_grid()


@pytest.fixture(scope="module")
def result():
    return certify_receipt(RECEIPT_PATH)


# ---------------------------------------------------------------------------
# (a) The real receipt: all 64 Pareto candidates certify cleanly
# ---------------------------------------------------------------------------


def test_all_sixty_four_pareto_candidates_certify(result):
    assert result["schema_version"] == RESULT_SCHEMA
    assert result["counts"] == {
        "candidates": 64,
        "certified": 64,
        "uncertified": 0,
        "unresolved": 0,
    }
    for row in result["candidates"]:
        assert row["certified"] is True
        assert set(row["criteria"]) == CRITERIA_NAMES
        assert all(verdict == CERTIFIED_PASS for verdict in row["criteria"].values())
        assert mp.mpf(row["min_margin"]) > 0


def test_result_binds_the_source_receipt_bytes(result):
    assert result["source_receipt_sha256"] == hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest()
    source = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert result["source_content_sha256"] == source["content_sha256"]
    ordinals = [row["ordinal"] for row in result["candidates"]]
    assert ordinals == [entry["ordinal"] for entry in source["pareto_front"]]


def test_result_states_the_disk_grid_path(result):
    assert result["disk_grid_path"] == DISK_GRID_PATH
    assert result["iv_dps"] == IV_DPS
    assert "dps80" in DISK_GRID_PATH  # the iv context has no working Bessel functions


# ---------------------------------------------------------------------------
# (b) Near-threshold control: a tie must straddle, never pass or fail
# ---------------------------------------------------------------------------


def test_exact_threshold_tie_is_unresolved_not_pass_or_fail(grid):
    """nu = 1 + 2u has |nu(1e6) - 1| == 2e-3, the newton_far threshold, exactly.

    The margin is 0 — within 1e-18 of the threshold — so no finite-precision
    evaluation may call it: fp64 would decide this by rounding luck, the interval
    layer must refuse.
    """

    row = certify_coefficients(2, [2, 0, 0, 0, 0], [0] * 5, grid=grid)
    assert row["criteria"]["newton_far"] == UNRESOLVED
    assert row["criteria"]["newton_far"] not in (CERTIFIED_PASS, CERTIFIED_FAIL)
    assert row["certified"] is False


def test_sub_ulp_threshold_shift_still_straddles(grid):
    """A margin far below one ulp at dps 60 (~1e-63) must stay unresolved."""

    nudged = "2." + "0" * 69 + "1e-3"  # 2e-3 * (1 + 5e-71)
    row = certify_coefficients(
        2, [2, 0, 0, 0, 0], [0] * 5, grid=grid, thresholds={"newton_far": nudged}
    )
    assert row["criteria"]["newton_far"] == UNRESOLVED


def test_resolvable_margins_resolve_on_both_sides(grid):
    """The straddle verdict is sharp, not lazy: real margins certify either way."""

    passing = certify_coefficients(
        2, [2, 0, 0, 0, 0], [0] * 5, grid=grid, thresholds={"newton_far": "2.1e-3"}
    )
    assert passing["criteria"]["newton_far"] == CERTIFIED_PASS
    failing = certify_coefficients(
        2, [2, 0, 0, 0, 0], [0] * 5, grid=grid, thresholds={"newton_far": "1.9e-3"}
    )
    assert failing["criteria"]["newton_far"] == CERTIFIED_FAIL


# ---------------------------------------------------------------------------
# (c) Certified-fail controls: clear violators fail with certainty
# ---------------------------------------------------------------------------


def test_newton_only_is_a_certified_failure(grid):
    """nu = 1 is the dark-matter problem itself: certainly no flat curves."""

    row = certify_coefficients(2, [0] * 5, [0] * 5, grid=grid)
    assert row["certified"] is False
    for disk in ("flat_disk_0", "flat_disk_1", "flat_disk_2"):
        assert row["criteria"][disk] == CERTIFIED_FAIL
    assert row["criteria"]["btfr_slope"] == CERTIFIED_FAIL
    assert mp.mpf(row["min_margin"]) < 0


@pytest.mark.parametrize(
    ("name", "beta_index", "a", "expected"),
    [
        ("overboost", 2, [0, 1, 0, 0, 0], "flat_disk_0"),
        ("beta_two_no_newton", 3, [1, 0, 0, 0, 0], "newton_far"),
    ],
)
def test_known_violators_certified_fail(grid, name, beta_index, a, expected):
    row = certify_coefficients(beta_index, a, [0] * 5, grid=grid)
    assert row["certified"] is False, name
    assert row["criteria"][expected] == CERTIFIED_FAIL, name


def test_interval_verdicts_agree_with_the_exact_layer(grid):
    """On well-separated controls the certifier must match 50-digit mpmath."""

    controls = {
        "newton_only": encode_candidate(2, [0] * 5, [0] * 5),
        "sqrt_family": encode_candidate(1, [0, 1, 0, 0, 0], [0] * 5),
        "linear_u": encode_candidate(2, [1, 0, 0, 0, 0], [0] * 5),
        "cbrt_family": encode_candidate(0, [0, 0, 1, 0, 0], [0] * 5),
        "overboost": encode_candidate(2, [0, 1, 0, 0, 0], [0] * 5),
    }
    float_grid = build_probe_grid()
    for name, ordinal in controls.items():
        candidate = verify_candidate_exact(ordinal, float_grid)
        row = certify_coefficients(*_split(ordinal), grid=grid)
        assert UNRESOLVED not in row["criteria"].values(), name
        assert row["certified"] == candidate["passes"], name


def _split(ordinal):
    from sigma_theory_compiler.gpu_baryonic_interpolation_screen import decode_ordinal

    candidate = decode_ordinal(ordinal)
    return candidate["beta_index"], candidate["a"], candidate["b"]


# ---------------------------------------------------------------------------
# (d) Determinism, sealing, and tamper fail-closure
# ---------------------------------------------------------------------------


def test_certification_is_deterministic(result):
    assert canonical_json_bytes(certify_receipt(RECEIPT_PATH)) == canonical_json_bytes(result)


def test_validate_accepts_the_sealed_result_and_binds_source_bytes(result):
    validate_result(result)
    validate_result(result, source_receipt_path=RECEIPT_PATH)


def test_seal_tamper_fails_closed(result):
    tampered = json.loads(json.dumps(result))
    tampered["candidates"][0]["certified"] = False
    with pytest.raises(IntervalCertifierError):
        validate_result(tampered)


def test_resealed_tamper_fails_closed_via_replay(result):
    """Re-sealing a tampered row defeats the hash but not the exact replay."""

    tampered = json.loads(json.dumps(result))
    tampered["candidates"][0]["min_margin"] = "9.99999999999e-1"
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(IntervalCertifierError, match="replay"):
        validate_result(tampered)


def test_resealed_counts_tamper_fails_closed(result):
    tampered = json.loads(json.dumps(result))
    tampered["counts"] = {"candidates": 64, "certified": 63, "uncertified": 0, "unresolved": 1}
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(body)
    with pytest.raises(IntervalCertifierError, match="counts"):
        validate_result(tampered)


def test_tampered_source_receipt_is_refused(tmp_path):
    text = RECEIPT_PATH.read_text(encoding="utf-8")
    forged = tmp_path / "forged.json"
    forged.write_text(text.replace("NVIDIA GeForce RTX 5090", "TAMPERED"), encoding="utf-8")
    with pytest.raises(IntervalCertifierError, match="seal"):
        certify_receipt(forged)


def test_resealed_source_formula_tamper_is_refused(tmp_path):
    """Even a re-sealed source receipt cannot smuggle a wrong formula binding."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    receipt["pareto_front"][0]["formula"] = "nu(y) = [(1) / (1)]^1,  u = y^(-1/2)"
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)
    forged = tmp_path / "resealed.json"
    forged.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(IntervalCertifierError, match="formula binding"):
        certify_receipt(forged)


def test_wrong_source_binding_is_refused(result, tmp_path):
    other = tmp_path / "other.json"
    other.write_bytes(b"{}\n")
    with pytest.raises(IntervalCertifierError, match="source receipt bytes"):
        validate_result(result, source_receipt_path=other)


# ---------------------------------------------------------------------------
# (e) Claim boundary
# ---------------------------------------------------------------------------


def test_claims_boundary(result):
    assert result["claims"] == CLAIMS
    assert CLAIMS["fp64_luck_can_decide_a_verdict"] is False
    assert CLAIMS["interval_rounding_is_outward"] is True
    assert CLAIMS["corpus_absence_establishes_novelty"] is False
    assert CLAIMS["certification_is_not_physical_validation"] is True


# ---------------------------------------------------------------------------
# Grid rigor: the interval probes really contain the true Freeman values
# ---------------------------------------------------------------------------


def test_interval_grid_contains_higher_precision_truth(grid):
    probe = grid["probes"][grid["disks"][0]["points"][0]["probe"]]
    with mp.workdps(100):
        mass = mp.mpf(1) / 250
        truth = _disk_gbar(mass, mp.mpf(SCREEN_CONFIG["inner_radii"][0]), mp.mpf(1))
    lower, upper = probe["y"]._mpi_
    assert mpf_le(lower, truth._mpf_) and mpf_le(truth._mpf_, upper)


def test_monotone_probe_ordering_is_certain(grid):
    order = grid["monotone"]
    assert len(order) == 32
    for previous, current in itertools.pairwise(order):
        assert grid["probes"][previous]["y"].b < grid["probes"][current]["y"].a


# ---------------------------------------------------------------------------
# CLI round trip
# ---------------------------------------------------------------------------


def test_cli_certify_then_validate(tmp_path, monkeypatch, capsys):
    output = tmp_path / "certification.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--receipt", str(RECEIPT_PATH), "--output", str(output)],
    )
    assert main() == 0
    first = output.read_bytes()
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"]["certified"] == 64
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--receipt", str(RECEIPT_PATH), "--output", str(output), "--validate-checked"],
    )
    assert main() == 0
    assert output.read_bytes() == first
