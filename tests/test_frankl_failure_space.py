from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.exclusion_ledger import evaluate_candidate
from sigma_theory_compiler.frankl_failure_space import (
    FranklFailureSpaceError,
    build_frankl_failure_ledger,
    classify_frankl_family,
    validate_frankl_failure_ledger,
)
from sigma_theory_compiler.mathoverflow_task2_evaluator import PROMISING_GENERATOR_IDS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "math" / "mathoverflow-task2"


def _read(name: str) -> dict:
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def _sources() -> tuple[dict, dict, dict, dict]:
    return (
        _read("public-submissions-v2.json"),
        _read("independent-evaluation-v2.json"),
        _read("blind-scoring-evidence-v2.json"),
        _read("adjudication-v2.json"),
    )


def test_real_task2_failures_compile_to_replayable_exclusion_ledger():
    sources = _sources()
    ledger = build_frankl_failure_ledger(*sources)

    validate_frankl_failure_ledger(ledger, *sources)
    assert ledger["counts"] == {
        "candidates": 36,
        "certificates": 40,
        "verified_certificates": 39,
        "heuristic_certificates": 1,
        "excluded_candidates": 36,
        "surviving_candidates": 0,
    }
    assert ledger["claims"]["historical_novelty_established"] is False
    assert ledger["claims"]["heuristic_failures_pruned_candidates"] is False


def test_exact_generator_failures_and_proved_families_are_distinguished():
    ledger = build_frankl_failure_ledger(*_sources())
    certificates = {row["certificate_id"]: row for row in ledger["certificates"]}

    exact_generator_ids = {
        row["witness"].get("submission_id")
        for row in certificates.values()
        if row["failure_mode"] == "exact_generator_closure_has_inequality_witness"
    }
    assert exact_generator_ids == PROMISING_GENERATOR_IDS
    assert certificates["frankl.family.universal-element"]["generalization_level"] == (
        "proved_parametric_family"
    )
    assert certificates["frankl.family.complete-nonempty-powerset"]["status"] == (
        "VERIFIED_EXCLUSION"
    )
    assert certificates["frankl.heuristic.high-symmetry-equality-trap"]["status"] == (
        "HEURISTIC_FAILURE"
    )


def test_all_historical_submissions_are_exactly_excluded_not_declared_novel():
    ledger = build_frankl_failure_ledger(*_sources())

    assert all(row["decision"] == "EXCLUDE" for row in ledger["candidate_outcomes"])
    assert ledger["survivor_frontier"]["candidate_ids"] == []
    assert any(
        row["kind"] == "epistemic_boundary"
        and "not novel" in row["constraint"]
        for row in ledger["survivor_frontier"]["constraints"]
    )


def test_changed_trial_evidence_fails_closed():
    public, evaluation, evidence, adjudication = _sources()
    tampered = copy.deepcopy(adjudication)
    tampered["decision"] = "PASS"

    with pytest.raises(FranklFailureSpaceError, match="seal"):
        build_frankl_failure_ledger(public, evaluation, evidence, tampered)


def test_proved_family_is_pruned_while_verified_reference_survives_frontier():
    ledger = build_frankl_failure_ledger(*_sources())
    dead_family = classify_frankl_family(
        [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5]],
        candidate_id="dead.n-minus-one",
    )
    reference = _read("accepted-reference-family-v2.json")
    live_family = classify_frankl_family(
        reference["family"], candidate_id="live.accepted-reference"
    )

    dead_outcome = evaluate_candidate(dead_family, ledger["certificates"])
    live_outcome = evaluate_candidate(live_family, ledger["certificates"])

    assert dead_outcome["decision"] == "EXCLUDE"
    assert "frankl.family.all-n-minus-one-plus-full" in (
        dead_outcome["verified_exclusion_certificate_ids"]
    )
    assert live_outcome["decision"] == "SURVIVE"
    assert live_family["verifier_receipt"]["exact_counterexample_valid"] is True
