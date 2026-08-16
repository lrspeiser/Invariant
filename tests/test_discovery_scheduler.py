"""A1 continuous-discovery-scheduler gates.

The scheduler's value is that unattended operation cannot silently lose or invent
work, so the load-bearing tests are the durability controls: a lease that outlives
its holder is reclaimed and completed after a hard kill, a completed key is never
rerun, and the hash-chained event log fails closed on tamper.  The honesty controls
are equally load-bearing: every stage ends in a sealed receipt or a typed blocker,
the collatz problem yields its known surviving relation end to end, a declared cap
trips into a recorded CAP_TRIPPED epoch with a consistent ledger, and the dashboard
regenerated after an epoch still validates against its sources.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sigma_theory_compiler.discovery_dashboard import validate_dashboard
from sigma_theory_compiler.discovery_scheduler import (
    GENERATOR_REGISTRY,
    DiscoverySchedulerError,
    SchedulerConfig,
    WorkLedger,
    _item_receipt,
    _sweep_statement,
    _write_receipt,
    execute_stage,
    main,
    run_discovery_epoch,
    run_scheduler,
    validate_epoch_receipt,
    validate_item_receipt,
    validate_soak_receipt,
)
from sigma_theory_compiler.problem_queue import load_queue, seal_queue
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _entry(entry_id: str, machine_form: dict, *, synthetic: bool = False) -> dict:
    return {
        "id": entry_id,
        "domain": "math",
        "statement": f"Test target {entry_id}.",
        "source_citation": "Internal scheduler test fixture citation.",
        "believed_open_because": "Not open mathematics: a scheduler test fixture.",
        "machine_form": machine_form,
        "progress_definition": "Receipts exist for every derived stage.",
        "control_rediscovery": False,
        "synthetic": synthetic,
    }


COLLATZ_FORM = {
    "kind": "sequence_rows",
    "generator": "collatz_total_stopping_time",
    "max_point": 64,
}


def _write_queue(path: Path, entries: list[dict]) -> dict:
    queue = seal_queue(entries)
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
    (defaults["repo_root"]).mkdir(parents=True, exist_ok=True)
    return SchedulerConfig(**defaults)


# ---------------------------------------------------------------------------
# Built-in generators
# ---------------------------------------------------------------------------


def test_generators_produce_known_exact_values() -> None:
    collatz, no_notes = GENERATOR_REGISTRY["collatz_total_stopping_time"](
        {"kind": "sequence_rows", "generator": "collatz_total_stopping_time", "max_point": 27}
    )
    sigma = {row["point"]: row["value"] for row in collatz}
    assert (sigma[1], sigma[2], sigma[3], sigma[6], sigma[27]) == (0, 1, 7, 8, 111)
    assert no_notes == []

    gaps, _ = GENERATOR_REGISTRY["prime_gap"](
        {"kind": "sequence_rows", "generator": "prime_gap", "max_point": 10}
    )
    assert [row["value"] for row in gaps] == [1, 2, 2, 4, 2, 4, 2, 4, 6, 2]

    contfrac, _ = GENERATOR_REGISTRY["contfrac_e_terms"](
        {"kind": "sequence_rows", "generator": "contfrac_e_terms", "max_point": 12}
    )
    assert [row["value"] for row in contfrac] == [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]

    aliquot, _ = GENERATOR_REGISTRY["aliquot_step_sum"](
        {"kind": "integer_trajectory", "map": "aliquot_step_sum", "seed": 276, "max_steps": 9}
    )
    assert [row["value"] for row in aliquot] == [
        396, 696, 1104, 1872, 3770, 3790, 3050, 2716, 2772
    ]


def test_aliquot_cap_truncation_is_typed_and_keeps_exact_prefix() -> None:
    """276's trajectory passes 1e12 at step 57: the prefix stays, the stop is typed."""

    rows, notes = GENERATOR_REGISTRY["aliquot_sum"](
        {"kind": "integer_trajectory", "map": "aliquot_sum", "seed": 276, "max_steps": 64}
    )
    assert len(rows) == 56
    assert notes[0]["type"] == "generator_cap_truncated:aliquot_sum"

    entry = _entry(
        "aliquot_276",
        {"kind": "integer_trajectory", "map": "aliquot_sum", "seed": 276, "max_steps": 64},
    )
    receipt = execute_stage("generate_rows", entry, {}, {"input_hash": "b" * 64})
    validate_item_receipt(receipt)
    assert receipt["status"] == "COMPLETED"
    assert receipt["payload"]["row_count"] == 56
    assert receipt["blockers"][0]["type"] == "generator_cap_truncated:aliquot_sum"


def test_generate_rows_unknown_generator_is_a_typed_blocker() -> None:
    entry = _entry(
        "sealed_world",
        {"kind": "sequence_rows", "generator": "sealed_catalan_like_recurrence_v1",
         "max_point": 48},
        synthetic=True,
    )
    receipt = execute_stage("generate_rows", entry, {}, {"input_hash": "0" * 64})
    validate_item_receipt(receipt)
    assert receipt["status"] == "BLOCKED"
    assert receipt["blockers"][0]["type"] == (
        "missing_generator:sealed_catalan_like_recurrence_v1"
    )


# ---------------------------------------------------------------------------
# Ledger durability
# ---------------------------------------------------------------------------


def test_ledger_lease_reclaim_and_idempotent_completion(tmp_path: Path) -> None:
    ledger = WorkLedger(tmp_path / "ledger.sqlite")
    input_hash = canonical_sha256({"probe": 1})
    item_id = ledger.ensure_item("problem_a", "generate_rows", input_hash)

    assert ledger.lease(item_id, "worker-1", lease_seconds=0)
    assert not ledger.lease(item_id, "worker-2", lease_seconds=60), "leased item releasable"
    time.sleep(0.01)
    assert ledger.reclaim_expired() == 1
    assert ledger.item_by_id(item_id)["state"] == "pending"

    assert ledger.lease(item_id, "worker-2", lease_seconds=300)
    receipt = _item_receipt("problem_a", "generate_rows", input_hash, "COMPLETED", {"rows": []}, [])
    path = _write_receipt(tmp_path / "items", "generate_rows", receipt)
    ledger.record_receipt(item_id, receipt, path, epoch_id=1)
    row = ledger.item("problem_a", "generate_rows", input_hash)
    assert row["state"] == "completed"
    assert row["receipt_sha256"] == receipt["content_sha256"]

    # A completed key is never rerun: it cannot be leased and derives to the same id.
    assert not ledger.lease(item_id, "worker-3", lease_seconds=60)
    assert ledger.ensure_item("problem_a", "generate_rows", input_hash) == item_id
    assert ledger.item_by_id(item_id)["state"] == "completed"
    ledger.validate_event_chain()


def test_event_chain_tamper_fails_closed(tmp_path: Path) -> None:
    ledger = WorkLedger(tmp_path / "ledger.sqlite")
    ledger.ensure_item("problem_a", "generate_rows", canonical_sha256({"n": 1}))
    ledger.ensure_item("problem_b", "generate_rows", canonical_sha256({"n": 2}))
    head = ledger.validate_event_chain()
    assert head == ledger.chain_head()

    connection = sqlite3.connect(tmp_path / "ledger.sqlite")
    connection.execute("UPDATE events SET payload_json='{\"forged\":1}' WHERE sequence=1")
    connection.commit()
    connection.close()
    with pytest.raises(DiscoverySchedulerError, match="event chain hash changed"):
        ledger.validate_event_chain()


def test_gpu_semaphore_serializes_to_one_holder(tmp_path: Path) -> None:
    ledger = WorkLedger(tmp_path / "ledger.sqlite")
    assert ledger.gpu_acquire("owner-1", lease_seconds=600)
    assert not ledger.gpu_acquire("owner-2", lease_seconds=600)
    ledger.gpu_release("owner-1")
    assert ledger.gpu_acquire("owner-2", lease_seconds=0)
    time.sleep(0.01)
    assert ledger.gpu_acquire("owner-3", lease_seconds=600), "expired GPU lease is reclaimable"


def test_stop_flag_roundtrip(tmp_path: Path) -> None:
    ledger = WorkLedger(tmp_path / "ledger.sqlite")
    assert not ledger.stop_requested()
    ledger.request_stop()
    assert ledger.stop_requested()
    ledger.clear_stop()
    assert not ledger.stop_requested()


def test_ledger_refuses_a_different_queue(tmp_path: Path) -> None:
    WorkLedger(tmp_path / "ledger.sqlite", queue_sha256="a" * 64)
    with pytest.raises(DiscoverySchedulerError, match="different sealed queue"):
        WorkLedger(tmp_path / "ledger.sqlite", queue_sha256="b" * 64)


# ---------------------------------------------------------------------------
# Stage receipts and typed blockers
# ---------------------------------------------------------------------------


def test_every_stage_yields_a_sealed_receipt_or_typed_blocker(tmp_path: Path) -> None:
    blocked_rows = _item_receipt(
        "sealed_world", "generate_rows", "0" * 64, "BLOCKED", {},
        [{"type": "missing_generator:x", "detail": "unregistered"}],
    )
    rows_path = _write_receipt(tmp_path, "rows", blocked_rows)

    conjecture = execute_stage(
        "conjecture",
        _entry("sealed_world", COLLATZ_FORM),
        {"rows_receipt_path": str(rows_path)},
        {"input_hash": "1" * 64},
    )
    validate_item_receipt(conjecture)
    assert conjecture["status"] == "BLOCKED"
    assert conjecture["blockers"][0]["type"] == "upstream_blocked:generate_rows"

    diophantine = execute_stage(
        "sweep",
        _entry(
            "erdos_straus",
            {"kind": "diophantine_family", "equation": "4/n = 1/x + 1/y + 1/z",
             "parameter": "n", "parameter_min": 2},
        ),
        {},
        {"input_hash": "2" * 64, "use_gpu": False, "sweep_hi": 4096},
    )
    validate_item_receipt(diophantine)
    assert diophantine["status"] == "BLOCKED"
    assert diophantine["blockers"][0]["type"] == "missing_sweeper:diophantine_family"

    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    noted = execute_stage(
        "note_gpu_campaign_receipts",
        _entry(
            "baryonic",
            {"kind": "dataset_law_fit", "dataset": "synthetic_analytic_disks",
             "target_relation": "flat_rotation_curves"},
        ),
        {},
        {"input_hash": "3" * 64, "repo_root": str(empty_repo)},
    )
    validate_item_receipt(noted)
    assert noted["status"] == "BLOCKED"
    assert noted["blockers"][0]["type"] == "missing_campaign_receipts:gpu-baryonic-screen"


def test_note_gpu_campaign_receipts_binds_sealed_sources(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    campaign_dir = repo / "runs" / "gpu-baryonic-screen"
    campaign_dir.mkdir(parents=True)
    body = {"schema_version": "invariant-test-campaign-1.0", "decision": "SURVIVORS_FOUND"}
    sealed = {**body, "content_sha256": canonical_sha256(body)}
    (campaign_dir / "campaign-v1.json").write_bytes(canonical_json_bytes(sealed) + b"\n")

    receipt = execute_stage(
        "note_gpu_campaign_receipts",
        _entry(
            "baryonic",
            {"kind": "dataset_law_fit", "dataset": "synthetic_analytic_disks",
             "target_relation": "flat_rotation_curves"},
        ),
        {},
        {"input_hash": "4" * 64, "repo_root": str(repo)},
    )
    validate_item_receipt(receipt)
    assert receipt["status"] == "COMPLETED"
    noted = receipt["payload"]["campaign_receipts"]
    assert len(noted) == 1
    assert noted[0]["seal_verified"] is True
    assert noted[0]["decision"] == "SURVIVORS_FOUND"
    assert noted[0]["path"] == "runs/gpu-baryonic-screen/campaign-v1.json"


def test_sweep_statement_parsing_round_trips() -> None:
    spec = {"sequence": "collatz_total_stopping_time", "sequence_params": {}, "step_cap": 10000}
    divisibility = _sweep_statement("divisibility", "7 divides a(n)", spec)
    assert (divisibility["kind"], divisibility["divisor"]) == ("divisibility", 7)

    congruence = _sweep_statement("congruence", "a(n) = 2 (mod 5)", spec)
    assert (congruence["residue"], congruence["modulus"]) == (2, 5)

    scaling = _sweep_statement(
        "index_scaling_relation", "a(2n) = (3/2)*a(n) + (-1/2)", spec
    )
    assert scaling["scale"] == 2
    assert scaling["alpha"] == {"numerator": 3, "denominator": 2}
    assert scaling["beta"] == {"numerator": -1, "denominator": 2}

    no_beta = _sweep_statement("index_scaling_relation", "a(3n) = (1)*a(n)", spec)
    assert no_beta["beta"] == {"numerator": 0, "denominator": 1}

    with pytest.raises(DiscoverySchedulerError, match="unparseable"):
        _sweep_statement("divisibility", "nonsense", spec)


# ---------------------------------------------------------------------------
# Prover routing
# ---------------------------------------------------------------------------


def _cubic_conjecture_receipt(tmp_path: Path) -> Path:
    rows = [{"point": n, "value": n**3 + n + 1} for n in range(16)]
    rows_receipt = _item_receipt(
        "cubic_fixture", "generate_rows", "5" * 64, "COMPLETED", {"rows": rows}, []
    )
    rows_path = _write_receipt(tmp_path, "rows", rows_receipt)
    entry = _entry("cubic_fixture", COLLATZ_FORM, synthetic=True)
    conjecture = execute_stage(
        "conjecture", entry, {"rows_receipt_path": str(rows_path)}, {"input_hash": "6" * 64}
    )
    assert conjecture["status"] == "COMPLETED"
    return _write_receipt(tmp_path, "conjecture", conjecture)


def test_route_provers_closed_form_through_lemma_decomposition(tmp_path: Path) -> None:
    conjecture_path = _cubic_conjecture_receipt(tmp_path)
    receipt = execute_stage(
        "route_provers",
        _entry("cubic_fixture", COLLATZ_FORM, synthetic=True),
        {"conjecture_receipt_path": str(conjecture_path)},
        {"input_hash": "7" * 64},
    )
    validate_item_receipt(receipt)
    assert receipt["status"] == "COMPLETED"
    assert receipt["payload"]["polynomial_closed_form"] == [1, 1, 0, 1]

    by_kind = {route["conjecture_kind"]: route for route in receipt["payload"]["routes"]}
    decomposition = by_kind["closed_form"]
    assert decomposition["prover"] == "lemma_decomposition"
    assert decomposition["result"]["decision"] == "DECOMPOSED"
    assert "theorem seqClosedForm" in decomposition["result"]["lean_source"]
    assert decomposition["result"]["kernel_verified"] is False

    monotonicity = by_kind["monotonicity"]
    assert monotonicity["prover"] == "quantified_inequality_proofs"
    assert monotonicity["result"]["decision"] == "PROVED_LOCALLY"
    assert monotonicity["routed_relation"] == "monotone_increasing"

    sign = by_kind["sign"]
    assert sign["result"]["decision"] == "PROVED_LOCALLY"
    assert sign["routed_relation"] == "nonnegative"

    blocker_types = {blocker["type"] for blocker in receipt["blockers"]}
    assert "missing_prover:partial_sum_closed_form" in blocker_types


def test_route_provers_records_missing_prover_for_unroutable_kinds(tmp_path: Path) -> None:
    entry = _entry("collatz_stopping_time", COLLATZ_FORM)
    rows = execute_stage("generate_rows", entry, {}, {"input_hash": "8" * 64})
    rows_path = _write_receipt(tmp_path, "rows", rows)
    conjecture = execute_stage(
        "conjecture", entry, {"rows_receipt_path": str(rows_path)}, {"input_hash": "9" * 64}
    )
    conjecture_path = _write_receipt(tmp_path, "conjecture", conjecture)
    receipt = execute_stage(
        "route_provers",
        entry,
        {"conjecture_receipt_path": str(conjecture_path)},
        {"input_hash": "a" * 64},
    )
    validate_item_receipt(receipt)
    assert receipt["status"] == "COMPLETED"
    assert receipt["payload"]["routes"] == []
    assert [blocker["type"] for blocker in receipt["blockers"]] == [
        "missing_prover:index_scaling_relation"
    ]


# ---------------------------------------------------------------------------
# End-to-end epochs
# ---------------------------------------------------------------------------


def test_collatz_end_to_end_micro_epochs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    result = run_scheduler(config)
    assert result["items_failed"] == 0
    assert result["items_completed"] == 4  # rows, conjecture, route_provers, sweep

    ledger = WorkLedger(config.ledger_path)
    counts = ledger.counts()
    assert counts["item_states"]["completed"] == 4
    assert counts["item_states"]["pending"] == 0
    assert "missing_prover:index_scaling_relation" in counts["blocker_types"]
    ledger.validate_event_chain()

    receipts = {
        json.loads(path.read_text(encoding="utf-8"))["stage"]: json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in (config.output_dir / "items" / "collatz_stopping_time").glob("*.json")
    }
    for receipt in receipts.values():
        validate_item_receipt(receipt)

    survived = [
        item
        for item in receipts["conjecture"]["payload"]["result"]["conjectures"]
        if item.get("status") == "SURVIVED"
    ]
    assert [item["kind"] for item in survived] == ["index_scaling_relation"]
    assert survived[0]["statement"] == "a(2n) = (1)*a(n) + (1)"

    sweeps = receipts["sweep"]["payload"]["sweeps"]
    assert len(sweeps) == 1
    assert sweeps[0]["sweep_decision"] == "NO_COUNTEREXAMPLE_IN_RANGE"
    assert sweeps[0]["sweep_receipt"]["counts"]["checked"] == 4095

    for path in (config.output_dir / "epochs").glob("*-summary-*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        validate_epoch_receipt(summary)
        assert summary["claims"]["unattended_operation"] is True
        assert summary["decision"] == "COMPLETED"


def test_dashboard_regenerated_after_epoch_validates(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1)
    queue = _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    # Give the dashboard a real declared source inside the temporary repo root.
    queue_copy = config.repo_root / "configs" / "problem_queue_v1.json"
    queue_copy.parent.mkdir(parents=True)
    queue_copy.write_bytes(canonical_json_bytes(queue) + b"\n")

    run_scheduler(config)
    stored = json.loads(
        (config.repo_root / "runs" / "discovery-dashboard" / "status-v1.json").read_text(
            encoding="utf-8"
        )
    )
    validate_dashboard(stored, root=config.repo_root)
    assert (config.repo_root / "runs" / "discovery-dashboard" / "status-v1.html").exists()

    ledger = WorkLedger(config.ledger_path)
    connection = sqlite3.connect(config.ledger_path)
    dashboard_sha = connection.execute(
        "SELECT dashboard_sha256 FROM epochs WHERE epoch_id=1"
    ).fetchone()[0]
    connection.close()
    assert dashboard_sha == stored["content_sha256"]
    ledger.validate_event_chain()


def test_watchdog_cap_trip_marks_epoch_and_ledger_stays_consistent(tmp_path: Path) -> None:
    ticks = {"ns": 0}

    def fake_monotonic() -> int:
        ticks["ns"] += 2_000_000_000
        return ticks["ns"]

    config = _config(
        tmp_path,
        caps={"max_wall_seconds": 1},
        probes={"monotonic_ns": fake_monotonic},
    )
    queue = _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_discovery_epoch(ledger, load_queue(config.queue_path), config)

    assert summary["decision"] == "CAP_TRIPPED:max_wall_seconds"
    assert summary["items"] == {"attempted": 1, "blocked": 0, "completed": 0, "failed": 0}
    counts = ledger.counts()
    assert counts["item_states"]["leased"] == 0, "tripped epoch must not strand leases"
    assert counts["item_states"]["pending"] == 1
    ledger.validate_event_chain()
    validate_epoch_receipt(summary)

    watchdog_paths = list((config.output_dir / "epochs").glob("*-watchdog-*.json"))
    assert len(watchdog_paths) == 1
    watchdog = json.loads(watchdog_paths[0].read_text(encoding="utf-8"))
    assert watchdog["decision"] == "CAP_TRIPPED:max_wall_seconds"
    assert summary["watchdog_receipt_sha256"] == watchdog["content_sha256"]


def test_epoch_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1)
    _write_queue(config.queue_path, [_entry("collatz_stopping_time", COLLATZ_FORM)])
    result = run_scheduler(config)
    summary = dict(result["summaries"][0])
    validate_epoch_receipt(summary)
    summary["items"] = {**summary["items"], "completed": 99}
    with pytest.raises(DiscoverySchedulerError, match="seal changed"):
        validate_epoch_receipt(summary)


def test_full_queue_kinds_derive_receipts_or_blockers(tmp_path: Path) -> None:
    """Every machine-form kind flows to receipts; blocked pipelines stay typed."""

    config = _config(tmp_path, epochs=8)
    campaign_dir = config.repo_root / "runs" / "gpu-baryonic-screen"
    campaign_dir.mkdir(parents=True)
    body = {"schema_version": "invariant-test-campaign-1.0", "decision": "SURVIVORS_FOUND"}
    sealed = {**body, "content_sha256": canonical_sha256(body)}
    (campaign_dir / "campaign-v1.json").write_bytes(canonical_json_bytes(sealed) + b"\n")

    entries = [
        _entry("collatz_stopping_time", COLLATZ_FORM),
        _entry(
            "sealed_world",
            {"kind": "sequence_rows", "generator": "sealed_catalan_like_recurrence_v1",
             "max_point": 48},
            synthetic=True,
        ),
        _entry(
            "erdos_straus",
            {"kind": "diophantine_family", "equation": "4/n = 1/x + 1/y + 1/z",
             "parameter": "n", "parameter_min": 2},
        ),
        _entry(
            "baryonic",
            {"kind": "dataset_law_fit", "dataset": "synthetic_analytic_disks",
             "target_relation": "flat_rotation_curves"},
        ),
        _entry(
            "inequality_families",
            {"kind": "module_target", "proof_module": "quantified_inequality_proofs",
             "decomposition_module": "lemma_decomposition"},
        ),
    ]
    _write_queue(config.queue_path, entries)
    result = run_scheduler(config)
    assert result["items_failed"] == 0

    ledger = WorkLedger(config.ledger_path)
    counts = ledger.counts()
    assert counts["item_states"]["pending"] == 0
    assert counts["item_states"]["leased"] == 0
    # collatz: 4 completed; sealed_world: rows + downstream all blocked (4);
    # erdos: 1 blocked; baryonic: 1 completed; module_target: 1 completed.
    assert counts["item_states"]["completed"] == 6
    assert counts["item_states"]["blocked"] == 5
    for blocker in (
        "missing_generator:sealed_catalan_like_recurrence_v1",
        "missing_sweeper:diophantine_family",
        "missing_prover:index_scaling_relation",
        "upstream_blocked:generate_rows",
    ):
        assert blocker in counts["blocker_types"]
    ledger.validate_event_chain()

    module_receipts = list((config.output_dir / "items" / "inequality_families").glob("*.json"))
    assert len(module_receipts) == 1
    module_receipt = json.loads(module_receipts[0].read_text(encoding="utf-8"))
    validate_item_receipt(module_receipt)
    assert module_receipt["status"] == "COMPLETED"
    assert module_receipt["payload"]["family_count"] == 0


# ---------------------------------------------------------------------------
# The soak harness: kill -9 mid-epoch, restart, recover, validate
# ---------------------------------------------------------------------------


def _scheduler_command(tmp_path: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sigma_theory_compiler.discovery_scheduler",
        "start",
        "--queue", str(tmp_path / "queue.json"),
        "--ledger", str(tmp_path / "ledger.sqlite"),
        "--output", str(tmp_path / "out"),
        "--repo-root", str(tmp_path / "repo"),
        "--sweep-hi-cpu", "4096",
        "--lease-seconds", "2",
        "--caps", json.dumps({"max_wall_seconds": 240}),
        *extra,
    ]


def test_micro_soak_survives_kill_and_resumes(tmp_path: Path) -> None:
    _write_queue(tmp_path / "queue.json", [_entry("collatz_stopping_time", COLLATZ_FORM)])
    (tmp_path / "repo").mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_SRC),
        "PYTHONUNBUFFERED": "1",
        "SIGMA_DISCOVERY_TEST_ITEM_DELAY_MS": "6000",
    }
    first = subprocess.Popen(
        _scheduler_command(tmp_path, "--soak-minutes", "10"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 60
        leased_line = None
        while time.monotonic() < deadline:
            line = first.stdout.readline()
            if line == "" and first.poll() is not None:
                break
            if "leased=1" in line:
                leased_line = line
                break
        assert leased_line is not None, "first soak never leased the seed item"
        time.sleep(1.0)  # kill mid-item: the worker is sleeping inside the lease
    finally:
        first.kill()
        first.wait(timeout=30)

    ledger = WorkLedger(tmp_path / "ledger.sqlite")
    assert ledger.counts()["item_states"]["leased"] == 1, "kill must strand a live lease"
    time.sleep(2.5)  # let the stranded lease expire

    env.pop("SIGMA_DISCOVERY_TEST_ITEM_DELAY_MS")
    second = subprocess.run(
        _scheduler_command(tmp_path, "--soak-minutes", "1"),
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
        check=False,
    )
    assert second.returncode == 0, second.stdout + second.stderr

    counts = ledger.counts()
    assert counts["item_states"] == {
        "pending": 0, "leased": 0, "completed": 4, "blocked": 0, "failed": 0,
    }
    assert counts["leases_reclaimed_total"] >= 1
    ledger.validate_event_chain()

    soak_paths = sorted(Path(tmp_path / "out").glob("soak-*.json"))
    assert soak_paths, "the second run must seal a soak receipt"
    soak = json.loads(soak_paths[-1].read_text(encoding="utf-8"))
    validate_soak_receipt(soak)
    assert soak["crash_recoveries"] >= 1
    assert soak["claims"]["zero_manual_steps"] is True
    assert soak["items_completed"] == 4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_status_and_stop(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger_path = tmp_path / "ledger.sqlite"
    WorkLedger(ledger_path).ensure_item("problem_a", "sweep", canonical_sha256({"n": 3}))
    assert main(["status", "--ledger", str(ledger_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["item_states"]["pending"] == 1
    assert report["event_chain_root"] != "0" * 64

    assert main(["stop", "--ledger", str(ledger_path)]) == 0
    assert WorkLedger(ledger_path).stop_requested()
