"""Gates for the unsolved-dozen progress campaign.

The receipts here claim *progress on problems nobody has solved*, which is exactly
where a dishonest artifact would be most tempting.  The load-bearing tests are
therefore the boundary ones: claims are fixed schema booleans, tampered seals and
edited witnesses fail closed, receipts replay deterministically, and the shipped
run directory validates end to end against the sealed v2 queue.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.dozen_unsolved_progress_campaign import (
    CLAIMS,
    DOZEN_IDS,
    LANES,
    RECEIPT_SCHEMA,
    UnsolvedProgressError,
    build_receipt,
    build_summary,
    validate_receipt,
    validate_summary,
)
from sigma_theory_compiler.problem_queue import load_queue
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "configs" / "problem_queue_v2.json"
RUNS_DIR = ROOT / "runs" / "math" / "unsolved-dozen"

#: Small-but-real bounds for the three representative end-to-end problems, one per
#: registered machine-form kind.
SMALL_BOUNDS = {
    "twin_prime_infinitude": {"exponent_max": 7},
    "lychrel_196": {"max_iterations": 300},
    "brocard_problem": {"parameter_max": 300},
}


@pytest.fixture(scope="module")
def queue() -> dict[str, object]:
    return load_queue(QUEUE_PATH)


@pytest.fixture(scope="module")
def receipts(queue) -> dict[str, dict[str, object]]:
    return {
        problem_id: build_receipt(queue, problem_id, bounds=bounds, use_gpu=False)
        for problem_id, bounds in SMALL_BOUNDS.items()
    }


def _reseal(value: dict[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# End-to-end lanes, one per machine-form kind
# ---------------------------------------------------------------------------


def test_twin_prime_rows_match_oeis_and_survivors_are_recorded(queue, receipts):
    receipt = receipts["twin_prime_infinitude"]
    validate_receipt(receipt, queue)
    rows = receipt["results"]["sequence_rows"]["rows"]
    assert [r["pi2"] for r in rows] == [2, 8, 35, 205, 1224, 8169, 58980]
    lane = receipt["results"]["conjecture_generation"]
    kinds = {s["kind"] for s in lane["survivors"]}
    assert "monotonicity" in kinds and "sign" in kinds
    assert lane["receipt"]["claims"]["survival_on_holdout_establishes_truth"] is False


def test_lychrel_trajectory_reports_no_palindrome_and_digit_rows(queue, receipts):
    receipt = receipts["lychrel_196"]
    validate_receipt(receipt, queue)
    facts = receipt["results"]["integer_trajectory"]
    assert facts["palindrome_found"] is False
    assert facts["palindrome_at"] is None
    assert facts["iterations"] == 300
    assert facts["final_digit_count"] > 100  # 196 grows steadily
    lane = receipt["results"]["conjecture_generation"]
    assert lane["rows_supplied"] == 64
    # Digit lengths repeat, so strict monotonicity must not survive; sign must.
    kinds = {s["kind"] for s in lane["survivors"]}
    assert "sign" in kinds and "monotonicity" not in kinds


def test_brocard_sweep_recovers_exactly_the_three_known_solutions(queue, receipts):
    receipt = receipts["brocard_problem"]
    validate_receipt(receipt, queue)
    assert receipt["results"]["solutions"] == [
        {"n": 4, "m": 5},
        {"n": 5, "m": 11},
        {"n": 7, "m": 71},
    ]
    flags = receipt["results"]["flags"]
    assert flags["exceeds_literature_bound"] is False
    assert flags["mechanism_receipt_below_literature_bound"] is True


# ---------------------------------------------------------------------------
# Claims, determinism, tamper
# ---------------------------------------------------------------------------


def test_every_receipt_carries_the_exact_progress_claims(receipts):
    for receipt in receipts.values():
        assert receipt["claims"] == CLAIMS
        assert receipt["claims"]["problem_remains_open"] is True
        assert receipt["claims"]["progress_is_not_solution"] is True
        assert receipt["claims"]["corpus_absence_establishes_novelty"] is False
        assert receipt["schema_version"] == RECEIPT_SCHEMA


def test_rebuilding_a_receipt_is_byte_deterministic(queue, receipts):
    again = build_receipt(
        queue, "brocard_problem", bounds=SMALL_BOUNDS["brocard_problem"], use_gpu=False
    )
    assert again == receipts["brocard_problem"]
    assert canonical_json_bytes(again) == canonical_json_bytes(receipts["brocard_problem"])


def test_tampered_claims_seal_and_witnesses_fail_closed(queue, receipts):
    flipped = copy.deepcopy(receipts["twin_prime_infinitude"])
    flipped["claims"]["problem_remains_open"] = False
    with pytest.raises(UnsolvedProgressError):
        validate_receipt(_reseal(flipped), queue)
    corrupt = copy.deepcopy(receipts["lychrel_196"])
    seal = corrupt["content_sha256"]
    corrupt["content_sha256"] = ("0" if seal[0] != "0" else "1") + seal[1:]
    with pytest.raises(UnsolvedProgressError):
        validate_receipt(corrupt, queue)
    forged = copy.deepcopy(receipts["brocard_problem"])
    forged["results"]["solutions"][0]["m"] = 6  # 4! + 1 is 25, not 36
    with pytest.raises(UnsolvedProgressError):
        validate_receipt(_reseal(forged), queue)
    smuggled = copy.deepcopy(receipts["brocard_problem"])
    smuggled["solved"] = True
    with pytest.raises(UnsolvedProgressError):
        validate_receipt(_reseal(smuggled), queue)


def test_receipt_is_bound_to_the_sealed_queue(queue, receipts):
    receipt = copy.deepcopy(receipts["twin_prime_infinitude"])
    receipt["queue_content_sha256"] = "0" * 64
    with pytest.raises(UnsolvedProgressError):
        validate_receipt(_reseal(receipt), queue)


def test_unknown_problem_and_unknown_bound_fail_closed(queue):
    with pytest.raises(UnsolvedProgressError):
        build_receipt(queue, "collatz_stopping_time")  # in queue, not in the dozen
    with pytest.raises(UnsolvedProgressError):
        build_receipt(queue, "brocard_problem", bounds={"speed": 11})


# ---------------------------------------------------------------------------
# The shipped run directory
# ---------------------------------------------------------------------------


def _shipped(problem_id: str) -> dict[str, object]:
    return json.loads((RUNS_DIR / f"{problem_id}.json").read_text(encoding="utf-8"))


def test_shipped_receipts_exist_validate_and_cover_the_dozen(queue):
    receipts = []
    for problem_id in DOZEN_IDS:
        receipt = _shipped(problem_id)
        validate_receipt(receipt, queue)
        assert list(receipt["lanes_run"]) == list(LANES[problem_id])
        receipts.append(receipt)
    summary = json.loads((RUNS_DIR / "campaign.json").read_text(encoding="utf-8"))
    validate_summary(summary, receipts)
    assert summary["counts"]["problems"] == 12
    assert summary["counts"]["counterexamples_found"] == 0
    assert {b["problem_id"] for b in summary["blockers"]} == set(DOZEN_IDS)


def test_shipped_files_are_canonical_bytes():
    for name in [*DOZEN_IDS, "campaign"]:
        raw = (RUNS_DIR / f"{name}.json").read_bytes()
        value = json.loads(raw.decode("utf-8"))
        assert raw == canonical_json_bytes(value) + b"\n", name


def test_shipped_headline_facts_hold():
    """The campaign's most interesting exact facts, pinned."""

    recaman = _shipped("recaman_coverage")["results"]["integer_trajectory"]
    assert recaman["steps"] == 10000000
    # 1355 is the exact smallest unreached value at this budget (cross-checked with an
    # independent set-based run; semantics anchored by 19 first appearing at step
    # 99734, matching OEIS A057167).  852655 is the literature's smallest unreached
    # value only at budgets far beyond 10^15 steps.
    assert recaman["smallest_unreached"] == 1355
    lychrel = _shipped("lychrel_196")["results"]["integer_trajectory"]
    assert lychrel["iterations"] == 10000
    assert lychrel["palindrome_found"] is False
    untouchable = _shipped("odd_untouchable")["results"]
    assert untouchable["odd_untouchable"] == [5]
    singmaster = _shipped("singmaster_conjecture")["results"]["sequence_rows"]
    assert singmaster["max_multiplicity"] == 8
    assert singmaster["attainers_of_max"] == [3003]
    gilbreath = _shipped("gilbreath_conjecture")["results"]["sequence_rows"]
    assert gilbreath["all_leading_terms_one"] is True
    assert gilbreath["rows_computed"] == 500


def test_summary_tamper_fails_closed(queue):
    receipts = [_shipped(problem_id) for problem_id in DOZEN_IDS]
    summary = build_summary(receipts)
    validate_summary(summary, receipts)
    tampered = copy.deepcopy(summary)
    tampered["counts"]["counterexamples_found"] = 1
    with pytest.raises(UnsolvedProgressError):
        validate_summary(_reseal(tampered), receipts)
    with pytest.raises(UnsolvedProgressError):
        build_summary(receipts[:11])
