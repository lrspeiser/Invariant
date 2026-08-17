"""A1 scheduler fan-out gates.

Fan-out changes what an epoch *is*: instead of a fixed four-stage march per machine-form
kind, an epoch attacks each problem with every lane the A5 registry says applies.  The
load-bearing tests are therefore the three things that could quietly go wrong.  Lanes
must actually multiply into work items (a fan-out that silently collapses to the old list
is the failure this whole change exists to prevent).  The single GPU lease must still
serialize the two declared GPU lanes.  And the epoch must not end with a stale build
queue, so the summary carries the freshly rebuilt capability-gap ledger's top open gaps.

The legacy path is tested too, from the other side: with ``fanout=False`` the derived
stage set must still be exactly ``STAGES_BY_KIND``, because every receipt already sealed
in this repository was produced under it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sigma_theory_compiler.capability_gap_ledger import validate_ledger as validate_gap_ledger
from sigma_theory_compiler.discovery_scheduler import (
    STAGES_BY_KIND,
    SchedulerConfig,
    WorkLedger,
    _item_receipt,
    derive_work,
    fanout_skips,
    fanout_stages,
    run_discovery_epoch,
    run_scheduler,
    validate_epoch_receipt,
    validate_item_receipt,
)
from sigma_theory_compiler.lane_registry import (
    LANES,
    REGISTRY_CONTENT_SHA256,
    SKIP_REASONS,
    applicable_lanes,
    lanes_by_resource,
)
from sigma_theory_compiler.problem_queue import seal_queue
from sigma_theory_compiler.sigma_core import canonical_json_bytes

COLLATZ_FORM = {
    "kind": "sequence_rows",
    "generator": "collatz_total_stopping_time",
    "max_point": 64,
}


def _entry(entry_id: str, machine_form: dict) -> dict:
    return {
        "id": entry_id,
        "domain": "math",
        "statement": f"Test target {entry_id}.",
        "source_citation": "Internal scheduler test fixture citation.",
        "believed_open_because": "Not open mathematics: a scheduler test fixture.",
        "machine_form": machine_form,
        "progress_definition": "Receipts exist for every derived stage.",
        "control_rediscovery": False,
        "synthetic": False,
    }


def _write_queue(path: Path, entries: list[dict]) -> dict:
    queue = seal_queue(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(queue) + b"\n")
    return queue


def _config(tmp_path: Path, **overrides) -> SchedulerConfig:
    defaults: dict = {
        "queue_path": tmp_path / "queue.json",
        "ledger_path": tmp_path / "ledger.sqlite",
        "output_dir": tmp_path / "out",
        "repo_root": tmp_path / "repo",
        "epochs": 6,
        "sweep_hi_cpu": 4096,
        "caps": {"max_wall_seconds": 600},
    }
    defaults.update(overrides)
    defaults["repo_root"].mkdir(parents=True, exist_ok=True)
    return SchedulerConfig(**defaults)


def _stages_in_ledger(ledger_path: Path) -> set[str]:
    connection = sqlite3.connect(ledger_path)
    try:
        return {row[0] for row in connection.execute("SELECT stage FROM items")}
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Fan-out multiplies work items
# ---------------------------------------------------------------------------


def test_fanout_creates_one_work_item_per_applicable_lane(tmp_path: Path) -> None:
    """N applicable lanes must produce N derived stages, not the fixed four."""

    entry = _entry("collatz_stopping_time", COLLATZ_FORM)
    applicable = applicable_lanes(entry)
    assert len(applicable) == 10
    assert fanout_stages(entry) == tuple(spec.stage for spec in applicable)

    config = _config(tmp_path / "fan", epochs=8, fanout=True, gap_ledger=False)
    _write_queue(config.queue_path, [entry])
    result = run_scheduler(config)
    assert result["items_failed"] == 0

    stages = _stages_in_ledger(config.ledger_path)
    assert stages == set(fanout_stages(entry))
    assert len(stages) == len(applicable) == 10
    WorkLedger(config.ledger_path).validate_event_chain()


def test_legacy_mode_still_derives_exactly_the_historical_stage_list(tmp_path: Path) -> None:
    config = _config(tmp_path / "legacy", epochs=8)
    _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    result = run_scheduler(config)
    assert result["items_completed"] == 4
    assert _stages_in_ledger(config.ledger_path) == set(STAGES_BY_KIND["sequence_rows"])


def test_fanout_records_every_skip_with_a_typed_reason(tmp_path: Path) -> None:
    """An unattempted lane is a fact in the epoch receipt, never an absence."""

    entry = _entry("collatz_stopping_time", COLLATZ_FORM)
    config = _config(tmp_path, epochs=1, fanout=True, gap_ledger=False)
    queue = _write_queue(config.queue_path, [entry])
    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    summary = run_discovery_epoch(ledger, queue, config)
    validate_epoch_receipt(summary)

    fanout = summary["fanout"]
    assert fanout["enabled"] is True
    assert fanout["registry_content_sha256"] == REGISTRY_CONTENT_SHA256
    planned = fanout["lanes_planned_per_problem"]["collatz_stopping_time"]
    skips = fanout["lanes_skipped_per_problem"]["collatz_stopping_time"]
    assert len(planned) + len(skips) == len(LANES)
    assert fanout["totals"]["attempts_planned"] == len(planned)
    assert fanout["totals"]["skips_recorded"] == len(skips)

    reasons = {item["skip_reason"] for item in skips}
    assert reasons <= set(SKIP_REASONS)
    assert reasons == {
        "kind_mismatch",
        "needs_bounded_coloring_statement",
        "needs_target_constant",
        "not_in_declared_roster",
    }
    for skip in skips:
        assert skip["detail"], "a recorded skip must say why"
    assert fanout_skips(entry) == skips


def test_legacy_epoch_records_that_fanout_was_off(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1, gap_ledger=False)
    queue = _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    summary = run_discovery_epoch(ledger, queue, config)
    assert summary["fanout"]["enabled"] is False
    assert summary["fanout"]["stage_source"] == "STAGES_BY_KIND"


# ---------------------------------------------------------------------------
# Resource routing
# ---------------------------------------------------------------------------


def test_two_gpu_lanes_never_hold_the_lease_simultaneously(tmp_path: Path) -> None:
    gpu_lane_ids = {spec.lane_id for spec in lanes_by_resource("gpu")}
    assert {"gpu_counterexample_sweep", "spectral_signal_scan"} <= gpu_lane_ids

    config = _config(tmp_path, fanout=True, gap_ledger=False)
    ledger = WorkLedger(config.ledger_path)
    assert ledger.gpu_acquire("sweep-worker", 60) is True
    assert ledger.gpu_acquire("spectral-worker", 60) is False, (
        "the second GPU lane must be refused while the first holds the lease"
    )
    ledger.gpu_release("sweep-worker")
    assert ledger.gpu_acquire("spectral-worker", 60) is True
    ledger.gpu_release("spectral-worker")

    entry = _entry("collatz_stopping_time", COLLATZ_FORM)
    gpu_stages = {spec.stage for spec in applicable_lanes(entry) if spec.resource == "gpu"}
    assert gpu_stages == {"spectral_scan", "sweep"}


def test_cpu_lanes_are_never_tagged_for_the_gpu_lease(tmp_path: Path) -> None:
    config = _config(tmp_path, fanout=True, gap_ledger=False, use_gpu=False)
    queue = _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    derived = derive_work(ledger, queue, config)
    assert {item.stage for item in derived} == {"generate_rows"}
    assert all(item.gpu is False for item in derived)


# ---------------------------------------------------------------------------
# The lanes themselves
# ---------------------------------------------------------------------------


def test_fanout_lane_stages_execute_their_declared_modules(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=8, fanout=True, gap_ledger=False)
    _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    result = run_scheduler(config)
    assert result["items_failed"] == 0

    receipts = {}
    for path in (config.output_dir / "items" / "collatz_stopping_time").glob("*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_item_receipt(receipt)
        receipts[receipt["stage"]] = receipt

    expected_schemas = {
        "basis_synthesis": "invariant-basis-synthesis-result-1.0",
        "holonomic_guess": "invariant-holonomic-guess-result-1.0",
        "nonlinear_search": "invariant-nonlinear-coefficient-search-result-1.0",
        "spectral_scan": "invariant-spectral-signal-scan-result-1.0",
        "structural_repair": "invariant-structural-repair-result-1.0",
    }
    for stage, schema in expected_schemas.items():
        assert receipts[stage]["payload"]["lane_receipt"]["schema_version"] == schema
        assert receipts[stage]["payload"]["row_count"] == 64

    # The prover lanes split the routing route_provers used to do in one stage.
    assert receipts["lemma_decomposition"]["payload"]["routed_kinds"] == [
        "closed_form",
        "linear_recurrence",
    ]
    assert receipts["quantified_inequality"]["payload"]["routed_kinds"] == [
        "monotonicity",
        "sign",
    ]
    # Collatz's one survivor is index_scaling_relation, which neither prover owns, so
    # both lanes complete having examined nothing rather than inventing a route.
    assert receipts["lemma_decomposition"]["payload"]["survivors_examined"] == 0
    assert receipts["quantified_inequality"]["payload"]["survivors_examined"] == 0
    # M7 still sweeps it, exactly as it did before the fan-out.
    assert receipts["sweep"]["payload"]["sweeps"][0]["sweep_decision"] == (
        "NO_COUNTEREXAMPLE_IN_RANGE"
    )


def test_fanout_blocked_upstream_still_produces_typed_blockers(tmp_path: Path) -> None:
    """An unregistered generator blocks every row lane, each with its own typed record."""

    config = _config(tmp_path, epochs=8, fanout=True, gap_ledger=False)
    _write_queue(
        config.queue_path,
        [
            _entry(
                "sealed_world",
                {
                    "kind": "sequence_rows",
                    "generator": "sealed_catalan_like_recurrence_v1",
                    "max_point": 48,
                },
            )
        ],
    )
    run_scheduler(config)
    ledger = WorkLedger(config.ledger_path)
    blockers = set(ledger.counts()["blocker_types"])
    assert "missing_generator:sealed_catalan_like_recurrence_v1" in blockers
    assert "upstream_blocked:generate_rows" in blockers
    assert ledger.counts()["item_states"]["pending"] == 0


# ---------------------------------------------------------------------------
# The epoch must not end with a stale build queue
# ---------------------------------------------------------------------------


def test_epoch_summary_carries_the_top_open_capability_gaps(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1)
    queue = _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    prior = config.repo_root / "runs" / "discovery-engine" / "items" / "prior.json"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(
        canonical_json_bytes(
            _item_receipt(
                "prior_problem",
                "sweep",
                "0" * 64,
                "BLOCKED",
                {},
                [{"type": "missing_sweeper:prior_family", "detail": "unbuilt"}],
            )
        )
        + b"\n"
    )

    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    summary = run_discovery_epoch(ledger, queue, config)
    validate_epoch_receipt(summary)

    gaps = summary["capability_gaps"]
    assert gaps["built"] is True
    assert gaps["counts"]["gaps_total"] >= 1
    assert len(gaps["top_open_gaps"]) <= 5
    assert "missing_sweeper:prior_family" in {
        item["gap_id"] for item in gaps["top_open_gaps"]
    }

    stored_path = (
        config.repo_root / "runs" / "discovery-engine" / "capability-gap-ledger.json"
    )
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    validate_gap_ledger(stored)
    assert stored["content_sha256"] == gaps["ledger_content_sha256"]
    assert (config.repo_root / "docs" / "CAPABILITY_GAPS.md").exists()


def test_gap_ledger_can_be_switched_off_and_says_so(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1, gap_ledger=False)
    queue = _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    summary = run_discovery_epoch(ledger, queue, config)
    assert summary["capability_gaps"]["built"] is False
    assert "disabled" in summary["capability_gaps"]["reason"]
