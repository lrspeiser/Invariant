"""DG4/DG6 survey gates.

Three things can quietly hollow out a survey.  The instruments can drift, so the controls
are asserted first: four classical operators recovered exactly, and the one spectral
signal we know is published (Steinerberger's Ulam frequency) recovered to the lambda this
repository already sealed.  The receipts can drift from what the summary says about them,
so the summary is rebuilt from the sealed receipts on disk and must come back byte-equal.
And a headline can be asserted rather than measured, so the DG4 and DG6 discovery flags
are tested by doctoring receipts and watching the flags move: a survivor on a sequence
with no declared published signal turns DG4 on, an operator on a sequence with no declared
exact structure turns DG6 on, and the declared knowledge entries turn them back off.

The eligibility rows are gates too.  A problem that is not scanned must say which of the
three typed reasons kept it out, so "nobody ever pointed the instrument at it" stays a
recoverable fact rather than an absence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.holonomic_guesser import (
    validate_receipt as validate_holonomic_receipt,
)
from sigma_theory_compiler.problem_queue import load_queue
from sigma_theory_compiler.sequence_survey import (
    BUILTIN_KNOWLEDGE,
    CLAIMS,
    CONTROLS,
    DG4_SCREENS,
    SPECTRAL_MIN_TERMS,
    SUMMARY_SCHEMA,
    SURVEY_TERMS,
    SequenceSurveyError,
    build_summary,
    recheck_summary_against_receipts,
    run_controls,
    survey_rows,
    validate_summary,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes
from sigma_theory_compiler.spectral_signal_scan import SpectralScanError
from sigma_theory_compiler.spectral_signal_scan import (
    validate_receipt as validate_spectral_receipt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "runs" / "math" / "survey-summary-v1.json"

INELIGIBLE_REASONS = {
    "INELIGIBLE_INSUFFICIENT_TERMS",
    "INELIGIBLE_KIND",
    "INELIGIBLE_MISSING_GENERATOR",
}


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Controls: both instruments still work
# ---------------------------------------------------------------------------


def test_the_four_holonomic_controls_are_recovered(summary):
    controls = {item["control_id"]: item for item in summary["controls"]["holonomic"]}
    assert set(controls) == set(CONTROLS)
    for control_id, record in controls.items():
        assert record["decision"] == "OPERATOR_FOUND", control_id
        receipt = json.loads(
            (REPO_ROOT / record["receipt_path"]).read_text(encoding="utf-8")
        )
        validate_holonomic_receipt(receipt)
        assert receipt["content_sha256"] == record["content_sha256"]
    assert summary["controls"]["all_recovered"] is True
    assert (
        controls["catalan"]["operator_statement"]
        == "(n + 2)*a(n+1) - (4*n + 2)*a(n) = 0 for n >= 0"
    )
    assert (
        controls["factorial"]["operator_statement"] == "a(n+1) - (n + 1)*a(n) = 0 for n >= 0"
    )


@pytest.mark.empirical_validation
def test_the_spectral_control_recovers_the_published_ulam_frequency(summary):
    """Steinerberger 2017 reports lambda ~= 2.5714474995; survival is evidence, not proof."""

    control = summary["controls"]["spectral"]
    assert control["recovered"] is True
    assert control["published_lambda"] == "2.5714474995"
    assert control["measured_lambda"] == "2.571448360067"
    assert control["matches_committed_exemplar"] is True
    assert control["difference_from_published"] < control["tolerance"]


def test_running_the_controls_again_reproduces_the_sealed_receipts(tmp_path):
    """The controls are deterministic: rerunning them into a fresh root gives the same
    bytes as the committed receipts."""

    records = run_controls(tmp_path)
    for record in records:
        fresh = (tmp_path / record["receipt_path"]).read_bytes()
        committed = (REPO_ROOT / record["receipt_path"]).read_bytes()
        assert fresh == committed


# ---------------------------------------------------------------------------
# Coverage: every queue problem is scanned or typed ineligible
# ---------------------------------------------------------------------------


def test_every_queue_problem_appears_in_both_surveys(summary):
    queue = load_queue(REPO_ROOT / "configs" / "problem_queue_v3.json")
    declared = {entry["id"] for entry in queue["entries"]}
    assert {row["problem_id"] for row in summary["spectral"]} == declared
    assert {row["problem_id"] for row in summary["holonomic"]} == declared
    assert summary["counts"]["problems_in_queue"] == len(declared)


def test_every_unscanned_problem_carries_a_typed_reason(summary):
    for key, ran in (("spectral", "SCANNED"), ("holonomic", "GUESSED")):
        for row in summary[key]:
            if row["status"] == ran:
                assert row["receipt_path"] is not None
                continue
            assert row["status"] in INELIGIBLE_REASONS
            assert row["detail"] is None or row["detail"]
            assert row["receipt_path"] is None


def test_the_sealed_synthetic_holdout_is_the_only_missing_generator(summary):
    for key in ("spectral", "holonomic"):
        blocked = [
            row["problem_id"]
            for row in summary[key]
            if row["status"] == "INELIGIBLE_MISSING_GENERATOR"
        ]
        assert blocked == ["catalan_like_recurrence_holdout"]


def test_the_dg4_term_floor_is_enforced(summary):
    for row in summary["spectral"]:
        if row["status"] == "INELIGIBLE_INSUFFICIENT_TERMS":
            assert row["terms"] < SPECTRAL_MIN_TERMS
        elif row["status"] == "SCANNED":
            assert row["terms"] >= SPECTRAL_MIN_TERMS


def test_every_scanned_sequence_declared_its_builtin_knowledge_first(summary):
    for row in summary["spectral"]:
        if row["status"] != "SCANNED":
            continue
        assert row["problem_id"] in BUILTIN_KNOWLEDGE
        note = row["builtin_knowledge_note"]
        assert note
        receipt = json.loads(
            (REPO_ROOT / row["receipt_path"]).read_text(encoding="utf-8")
        )
        assert receipt["builtin_knowledge"] == note


def test_survey_rows_are_capped_by_the_declared_term_table():
    queue = load_queue(REPO_ROOT / "configs" / "problem_queue_v3.json")
    for entry in queue["entries"]:
        rows, provenance = survey_rows(entry)
        if provenance["reason"] is not None:
            assert not rows
            continue
        assert len(rows) <= SURVEY_TERMS[entry["id"]]


# ---------------------------------------------------------------------------
# Receipt integrity, determinism, tamper
# ---------------------------------------------------------------------------


def test_every_survey_receipt_validates(summary):
    for row in summary["spectral"]:
        if row["status"] != "SCANNED":
            continue
        receipt = json.loads((REPO_ROOT / row["receipt_path"]).read_text(encoding="utf-8"))
        validate_spectral_receipt(receipt)
        assert receipt["content_sha256"] == row["content_sha256"]
        assert receipt["decision"] == row["decision"]
    for row in summary["holonomic"]:
        if row["status"] != "GUESSED":
            continue
        receipt = json.loads((REPO_ROOT / row["receipt_path"]).read_text(encoding="utf-8"))
        validate_holonomic_receipt(receipt)
        assert receipt["content_sha256"] == row["content_sha256"]
        assert receipt["decision"] == row["decision"]


def test_the_committed_summary_validates_and_rechecks(summary):
    validate_summary(summary)
    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["claims"] == CLAIMS
    recheck_summary_against_receipts(REPO_ROOT, summary)


def test_the_summary_is_canonically_encoded():
    assert SUMMARY_PATH.read_bytes() == canonical_json_bytes(
        json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ) + b"\n"


def test_a_tampered_summary_seal_is_rejected(summary):
    tampered = {**summary, "counts": {**summary["counts"], "spectral_survivors": 99}}
    with pytest.raises(SequenceSurveyError):
        validate_summary(tampered)


def test_a_tampered_receipt_fails_the_recheck(tmp_path, summary):
    """Editing a sealed receipt under the summary must break the recheck, not slide by."""

    import shutil

    root = tmp_path / "repo"
    for relative in ("runs/math/spectral", "runs/math/holonomic"):
        shutil.copytree(REPO_ROOT / relative, root / relative)
    victim = root / "runs" / "math" / "spectral" / "survey" / "ulam_sequence_structure.json"
    receipt = json.loads(victim.read_text(encoding="utf-8"))
    receipt["peaks"][0]["prefix_negative_terms"] += 1
    victim.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises((SequenceSurveyError, SpectralScanError)):
        recheck_summary_against_receipts(root, summary)


# ---------------------------------------------------------------------------
# The discovery flags are computed from the receipts, not asserted
# ---------------------------------------------------------------------------


def _records(summary: dict, key: str) -> list[dict]:
    fields = ("problem_id", "generator", "terms", "status", "detail", "receipt_path")
    return [{field: row[field] for field in fields} for row in summary[key]]


def _controls(summary: dict) -> list[dict]:
    fields = ("control_id", "citation", "terms", "receipt_path")
    return [{field: row[field] for field in fields} for row in summary["controls"]["holonomic"]]


def test_the_discovery_flags_are_rederived_from_the_receipts(summary):
    rebuilt = build_summary(
        REPO_ROOT,
        _records(summary, "spectral"),
        _records(summary, "holonomic"),
        _controls(summary),
    )
    assert rebuilt == summary
    assert rebuilt["discovery_conditions"]["DG4"]["met"] == bool(
        rebuilt["discovery_conditions"]["DG4"]["survivors"]
    )
    assert rebuilt["discovery_conditions"]["DG6"]["met"] == bool(
        rebuilt["discovery_conditions"]["DG6"]["operators"]
    )


def test_dg4_turns_off_when_the_only_survivors_carry_a_published_signal(summary, monkeypatch):
    """Declare a published signal for every surviving sequence and the flag must fall."""

    survivors = summary["discovery_conditions"]["DG4"]["sequences"]
    assert survivors, "this test needs at least one surviving sequence to suppress"
    patched = {
        key: {**value, "published_spectral_signal": value["published_spectral_signal"]
              or "declared for this test only"}
        for key, value in BUILTIN_KNOWLEDGE.items()
    }
    monkeypatch.setattr("sigma_theory_compiler.sequence_survey.BUILTIN_KNOWLEDGE", patched)
    rebuilt = build_summary(
        REPO_ROOT,
        _records(summary, "spectral"),
        _records(summary, "holonomic"),
        _controls(summary),
    )
    assert rebuilt["discovery_conditions"]["DG4"]["met"] is False
    assert rebuilt["discovery_conditions"]["DG4"]["sequences"] == []


def test_dg6_turns_on_when_a_known_structure_is_withdrawn(summary, monkeypatch):
    """The operators the survey found are excluded only by the declared knowledge table;
    withdraw the declaration and DG6's flag must rise, which proves it is computed."""

    found = [
        row["problem_id"]
        for row in summary["holonomic"]
        if row.get("decision") == "OPERATOR_FOUND"
    ]
    assert found, "this test needs at least one recovered operator"
    assert summary["discovery_conditions"]["DG6"]["met"] is False
    patched = {
        key: {**value, "known_exact_structure": None}
        for key, value in BUILTIN_KNOWLEDGE.items()
    }
    monkeypatch.setattr("sigma_theory_compiler.sequence_survey.BUILTIN_KNOWLEDGE", patched)
    rebuilt = build_summary(
        REPO_ROOT,
        _records(summary, "spectral"),
        _records(summary, "holonomic"),
        _controls(summary),
    )
    assert rebuilt["discovery_conditions"]["DG6"]["met"] is True
    assert sorted(rebuilt["discovery_conditions"]["DG6"]["sequences"]) == sorted(found)


def test_every_dg4_survivor_carries_all_declared_screens(summary):
    screens = {screen["screen_id"] for screen in DG4_SCREENS}
    for survivor in summary["discovery_conditions"]["DG4"]["survivors"]:
        assert screens <= set(survivor)
        assert survivor["lambda"] and survivor["magnitude_at_lambda"]
        assert survivor["holdout_negative"]["terms"] > 0
    after = summary["discovery_conditions"]["DG4"]["sequences_after_declared_screens"]
    assert set(after) <= set(summary["discovery_conditions"]["DG4"]["sequences"])


def test_no_claim_in_the_summary_asserts_novelty(summary):
    assert summary["claims"]["builtin_knowledge_absence_establishes_novelty"] is False
    assert summary["claims"]["scalar_truth_or_probability_score"] is False
    assert summary["claims"]["survival_on_holdout_establishes_truth"] is False
    assert summary["claims"]["discovery_conditions_computed_from_receipts"] is True
    assert "not a novelty claim" in summary["discovery_conditions"]["DG4"]["note"] or (
        "candidate signal" in summary["discovery_conditions"]["DG4"]["note"]
    )
