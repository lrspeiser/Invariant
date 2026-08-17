"""A6 capability-gap-ledger gates.

The ledger is a build queue derived from real failures, so the load-bearing tests are the
ones a reader would use to decide whether to trust the ranking: the formula is declared
and hand-computable, the order actually follows it, provenance binds real receipt hashes,
and a gap only becomes ``discharged`` when evidence says so — proved twice, once for the
generic "same problem+stage now passes" rule and once for a frozen discharge rule.  The
live-corpus test asserts the ledger reports the repository's real current state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.capability_gap_ledger import (
    CLAIMS,
    DISCHARGE_RULES,
    EXTRACTORS,
    PIPELINE_DEPTH,
    SCAN_ROOTS,
    UNBLOCK_VALUE_FORMULA,
    CapabilityGapLedgerError,
    build_ledger,
    main,
    render_markdown,
    top_open_gaps,
    validate_ledger,
)
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


def _seal(body: dict) -> dict:
    return {**body, "content_sha256": canonical_sha256(body)}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _item(problem: str, stage: str, status: str, blockers: list[str]) -> dict:
    return _seal(
        {
            "blockers": [{"type": item, "detail": f"typed blocker {item}"} for item in blockers],
            "claims": {},
            "input_hash": "0" * 64,
            "payload": {},
            "problem_id": problem,
            "schema_version": "invariant-discovery-item-1.0",
            "scope": "test fixture",
            "stage": stage,
            "status": status,
        }
    )


def _epoch(epoch_id: int, outcomes: dict) -> dict:
    return _seal(
        {
            "claims": {},
            "decision": "COMPLETED",
            "epoch_id": epoch_id,
            "per_problem_outcomes": outcomes,
            "schema_version": "invariant-discovery-epoch-1.0",
            "scope": "test fixture",
        }
    )


# ---------------------------------------------------------------------------
# Extraction and provenance
# ---------------------------------------------------------------------------


def test_typed_blockers_are_extracted_with_provenance(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "discovery-engine" / "items"
    receipt = _item("collatz_stopping_time", "route_provers", "COMPLETED", ["missing_prover:sign"])
    _write(runs / "collatz" / "route_provers-a.json", receipt)

    ledger = build_ledger(tmp_path)
    validate_ledger(ledger)
    (gap,) = ledger["gaps"]
    assert gap["gap_id"] == "missing_prover:sign"
    assert gap["blocked_problems"] == ["collatz_stopping_time"]
    assert gap["stages"] == ["route_provers"]
    assert gap["extractors"] == ["discovery_item_blockers"]
    assert gap["status"] == "open"
    assert gap["lanes_that_emit_it"] == ["quantified_inequality_proofs"]
    (example,) = gap["example_receipts"]
    assert example["path"] == "runs/discovery-engine/items/collatz/route_provers-a.json"
    assert example["content_sha256"] == receipt["content_sha256"]
    assert gap["first_seen_receipt"] == example["path"]


def test_prose_is_not_mistaken_for_a_typed_blocker(tmp_path: Path) -> None:
    _write(
        tmp_path / "runs" / "x" / "prose.json",
        _seal(
            {
                "first_blocker": "the model could not be built because reasons apply here",
                "problem_id": "p",
                "schema_version": "invariant-test-1.0",
            }
        ),
    )
    _write(
        tmp_path / "runs" / "x" / "typed.json",
        _seal(
            {
                "first_blocker": "no_termination_lane",
                "problem_id": "lychrel_196",
                "schema_version": "invariant-test-1.0",
            }
        ),
    )
    ledger = build_ledger(tmp_path)
    assert [gap["gap_id"] for gap in ledger["gaps"]] == ["no_termination_lane"]
    assert ledger["corpus"]["files_parsed"] == 2


def test_unparseable_files_are_counted_not_swallowed(tmp_path: Path) -> None:
    broken = tmp_path / "runs" / "broken.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"{not json")
    ledger = build_ledger(tmp_path)
    assert ledger["corpus"]["files_seen"] == 1
    assert ledger["corpus"]["files_unparseable"] == 1
    assert ledger["corpus"]["files_parsed"] == 0


# ---------------------------------------------------------------------------
# The declared ranking formula
# ---------------------------------------------------------------------------


def test_ranking_follows_the_declared_formula_and_is_hand_computable(tmp_path: Path) -> None:
    items = tmp_path / "runs" / "discovery-engine" / "items"
    # `wide` blocks two problems at a shallow stage; `deep` blocks one at a deep stage.
    _write(items / "a.json", _item("problem_a", "generate_rows", "BLOCKED", ["missing_generator:x"]))
    _write(items / "b.json", _item("problem_b", "generate_rows", "BLOCKED", ["missing_generator:x"]))
    _write(items / "c.json", _item("problem_c", "sweep", "COMPLETED", ["missing_sweeper:y"]))

    ledger = build_ledger(tmp_path)
    validate_ledger(ledger)
    ranked = [(gap["gap_id"], gap["rank"]) for gap in ledger["gaps"]]
    assert ranked == [("missing_generator:x", 1), ("missing_sweeper:y", 2)]

    wide, deep = ledger["gaps"]
    assert wide["unblock_value"]["distinct_units_blocked"] == 2
    assert wide["unblock_value"]["pipeline_depth"] == PIPELINE_DEPTH["generate_rows"] == 1
    assert deep["unblock_value"]["distinct_units_blocked"] == 1
    assert deep["unblock_value"]["pipeline_depth"] == PIPELINE_DEPTH["sweep"] == 5
    assert wide["unblock_value"]["formula"] == UNBLOCK_VALUE_FORMULA


def test_depth_breaks_the_tie_upward(tmp_path: Path) -> None:
    items = tmp_path / "runs" / "e" / "items"
    _write(items / "shallow.json", _item("p", "generate_rows", "BLOCKED", ["shallow_gap"]))
    _write(items / "deep.json", _item("p", "sweep", "COMPLETED", ["deep_gap"]))
    ledger = build_ledger(tmp_path)
    assert [gap["gap_id"] for gap in ledger["gaps"]] == ["deep_gap", "shallow_gap"]


def test_ledger_declares_no_scalar_score(tmp_path: Path) -> None:
    _write(
        tmp_path / "runs" / "e" / "i.json",
        _item("p", "sweep", "COMPLETED", ["missing_sweeper:z"]),
    )
    ledger = build_ledger(tmp_path)
    assert ledger["claims"]["scalar_truth_or_probability_score"] is False
    assert ledger["claims"] == CLAIMS
    encoded = json.dumps(ledger)
    for forbidden in ("priority_score", "confidence", "credibility", "likelihood"):
        assert forbidden not in encoded
    # The only occurrence of "probability" is the sealed house-rule claim denying one.
    assert encoded.count("probability") == 1


# ---------------------------------------------------------------------------
# Discharge detection
# ---------------------------------------------------------------------------


def test_same_problem_stage_now_passing_discharges_the_gap(tmp_path: Path) -> None:
    """A later epoch that completes the same problem+stage without the blocker."""

    epochs = tmp_path / "runs" / "discovery-engine" / "epochs"
    _write(
        epochs / "epoch-1.json",
        _epoch(
            1,
            {
                "erdos_straus": {
                    "sweep": {"status": "BLOCKED", "blockers": ["missing_sweeper:diophantine"]}
                }
            },
        ),
    )
    _write(
        epochs / "epoch-2.json",
        _epoch(2, {"erdos_straus": {"sweep": {"status": "COMPLETED", "blockers": []}}}),
    )
    ledger = build_ledger(tmp_path)
    (gap,) = ledger["gaps"]
    assert gap["gap_id"] == "missing_sweeper:diophantine"
    assert gap["status"] == "discharged"
    assert gap["discharge_rule"] == "same_problem_stage_now_passing"
    assert "epoch 2" in gap["discharged_by"]
    assert ledger["counts"]["gaps_discharged"] == 1
    assert ledger["counts"]["gaps_open"] == 0


def test_a_later_epoch_that_still_carries_the_blocker_does_not_discharge(tmp_path: Path) -> None:
    epochs = tmp_path / "runs" / "e" / "epochs"
    _write(
        epochs / "e1.json",
        _epoch(1, {"p": {"sweep": {"status": "BLOCKED", "blockers": ["missing_sweeper:q"]}}}),
    )
    _write(
        epochs / "e2.json",
        _epoch(
            2, {"p": {"sweep": {"status": "COMPLETED", "blockers": ["missing_sweeper:q"]}}}
        ),
    )
    ledger = build_ledger(tmp_path)
    (gap,) = ledger["gaps"]
    assert gap["status"] == "open"
    assert gap["discharged_by"] is None


def test_an_earlier_pass_does_not_discharge_a_later_block(tmp_path: Path) -> None:
    """Ordering is by epoch id, so a pass that predates the block proves nothing."""

    epochs = tmp_path / "runs" / "e" / "epochs"
    _write(epochs / "e1.json", _epoch(1, {"p": {"sweep": {"status": "COMPLETED", "blockers": []}}}))
    _write(
        epochs / "e2.json",
        _epoch(2, {"p": {"sweep": {"status": "BLOCKED", "blockers": ["missing_sweeper:q"]}}}),
    )
    ledger = build_ledger(tmp_path)
    (gap,) = ledger["gaps"]
    assert gap["status"] == "open"


def test_declared_discharge_rule_needs_its_evidence_receipt(tmp_path: Path) -> None:
    """Without the sweeper receipt the gap stays open; with it, it is discharged."""

    _write(
        tmp_path / "runs" / "e" / "blocked.json",
        _item("erdos_straus", "sweep", "BLOCKED", ["missing_sweeper:diophantine_family"]),
    )
    ledger = build_ledger(tmp_path)
    (gap,) = ledger["gaps"]
    assert gap["status"] == "open"

    _write(
        tmp_path / "runs" / "math" / "exponent-diophantine" / "erdos_straus.json",
        _seal(
            {
                "decision": "NO_UNSOLVABLE_N_IN_RANGE",
                "problem_id": "erdos_straus_sweeper_target",
                "schema_version": "invariant-exponent-diophantine-sweep-1.0",
                "scope": (
                    "this module is the diophantine_family sweeper whose absence "
                    "discovery receipts record as the typed blocker "
                    "missing_sweeper:diophantine_family"
                ),
            }
        ),
    )
    ledger = build_ledger(tmp_path)
    (gap,) = ledger["gaps"]
    assert gap["status"] == "discharged"
    assert gap["discharge_rule"] == "declared_discharge_receipt"
    assert gap["discharged_by"] == "sigma_theory_compiler.exponent_diophantine_sweeper"


def test_a_discharge_rule_whose_receipt_never_names_the_gap_does_not_fire(tmp_path: Path) -> None:
    _write(
        tmp_path / "runs" / "e" / "blocked.json",
        _item("erdos_straus", "sweep", "BLOCKED", ["missing_sweeper:diophantine_family"]),
    )
    _write(
        tmp_path / "runs" / "math" / "exponent-diophantine" / "other.json",
        _seal(
            {
                "decision": "NO_UNSOLVABLE_N_IN_RANGE",
                "schema_version": "invariant-exponent-diophantine-sweep-1.0",
                "scope": "a sweep receipt that never names the blocker it would discharge",
            }
        ),
    )
    ledger = build_ledger(tmp_path)
    (gap,) = ledger["gaps"]
    assert gap["status"] == "open"


# ---------------------------------------------------------------------------
# Validation and render
# ---------------------------------------------------------------------------


def test_validation_fails_closed_on_tamper(tmp_path: Path) -> None:
    _write(tmp_path / "runs" / "e" / "i.json", _item("p", "sweep", "COMPLETED", ["gap_one"]))
    ledger = build_ledger(tmp_path)
    validate_ledger(ledger)

    tampered = {**ledger, "counts": {**ledger["counts"], "gaps_open": 99}}
    with pytest.raises(CapabilityGapLedgerError, match="seal changed"):
        validate_ledger(tampered)

    with pytest.raises(CapabilityGapLedgerError, match="schema changed"):
        validate_ledger({**ledger, "schema_version": "other"})

    broken = json.loads(json.dumps(ledger))
    broken["gaps"][0]["blocked_count"] = 42
    broken["content_sha256"] = canonical_sha256(
        {key: item for key, item in broken.items() if key != "content_sha256"}
    )
    with pytest.raises(CapabilityGapLedgerError, match="blocked_count"):
        validate_ledger(broken)


def test_markdown_render_is_deterministic_and_carries_the_formula(tmp_path: Path) -> None:
    _write(tmp_path / "runs" / "e" / "i.json", _item("p", "sweep", "COMPLETED", ["gap_one"]))
    ledger = build_ledger(tmp_path)
    first = render_markdown(ledger)
    assert first == render_markdown(ledger)
    assert UNBLOCK_VALUE_FORMULA in first
    assert "no scalar priority score" in first
    assert "`gap_one`" in first
    for extractor in EXTRACTORS:
        assert extractor["extractor_id"] in first


def test_top_open_gaps_skips_discharged_and_caps_the_list(tmp_path: Path) -> None:
    items = tmp_path / "runs" / "e"
    for index in range(8):
        _write(items / f"i{index}.json", _item(f"p{index}", "sweep", "COMPLETED", [f"gap_{index}"]))
    ledger = build_ledger(tmp_path)
    top = top_open_gaps(ledger, 5)
    assert len(top) == 5
    assert all(set(item) == {
        "blocked_count", "blocked_problems", "gap_id", "lanes_that_emit_it", "unblock_value"
    } for item in top)


def test_cli_build_then_validate_checked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(tmp_path / "runs" / "e" / "i.json", _item("p", "sweep", "COMPLETED", ["gap_one"]))
    assert main(["build", "--repo-root", str(tmp_path)]) == 0
    assert "BUILT gaps=1" in capsys.readouterr().out
    receipt = tmp_path / "runs" / "discovery-engine" / "capability-gap-ledger.json"
    assert receipt.exists()
    assert (tmp_path / "docs" / "CAPABILITY_GAPS.md").exists()

    assert main(["build", "--repo-root", str(tmp_path), "--validate-checked"]) == 0
    assert "VALID gaps=" in capsys.readouterr().out

    # A hand-edited receipt is caught.
    receipt.write_bytes(b"{}\n")
    assert main(["build", "--repo-root", str(tmp_path), "--validate-checked"]) == 1
    assert "INVALID" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The live corpus: the ledger must report the repository's real state
# ---------------------------------------------------------------------------


def test_live_corpus_reports_the_real_gaps_and_discharges() -> None:
    stored = REPO_ROOT / "runs" / "discovery-engine" / "capability-gap-ledger.json"
    if not stored.exists():
        pytest.skip("gap ledger has not been built in this checkout")
    ledger = json.loads(stored.read_text(encoding="utf-8"))
    validate_ledger(ledger)
    assert ledger["scan_roots"] == list(SCAN_ROOTS)
    by_id = {gap["gap_id"]: gap for gap in ledger["gaps"]}

    for gap_id in (
        "missing_prover:index_scaling_relation",
        "missing_prover:sign",
        "missing_adapter:cubic_g3_uniform_weak_field_cone",
        "missing_adapter:aqual_nu_to_kessence_inversion",
        "missing_adapter:uv_form_factor_operator",
        "missing_adapter:direct_scalar_matter_coupling",
        "infinitude_not_sweepable",
        "no_termination_lane",
        "bounded_multiplicity_not_expressible",
        "missing_generator:sealed_catalan_like_recurrence_v1",
    ):
        assert gap_id in by_id, gap_id
        assert by_id[gap_id]["status"] == "open", gap_id

    for gap_id, module in (
        ("missing_sweeper:diophantine_family", "exponent_diophantine_sweeper"),
        ("statement_kinds_too_weak", "spectral_signal_scan"),
        ("missing_adapter:nonlocal_fractional_operator", "nonlocal_fractional_adapter"),
    ):
        assert by_id[gap_id]["status"] == "discharged", gap_id
        assert module in by_id[gap_id]["discharged_by"]

    # The known real blast radius of the sign prover.  It grew when the six DG5 row
    # generators landed: problems that used to stop at `missing_generator` now reach the
    # prover lane and record what it cannot prove.  Unblocking a stage moves the gap
    # downstream, it does not remove it.
    assert by_id["missing_prover:sign"]["blocked_problems"] == [
        "aliquot_276",
        "continued_fraction_e_pattern",
        "lychrel_196",
        "prime_gap_polynomial",
        "quantified_inequality_families",
        "singmaster_conjecture",
        "twin_prime_infinitude",
        "ulam_sequence_structure",
    ]
    assert {rule["gap_id"] for rule in DISCHARGE_RULES} <= set(by_id)
