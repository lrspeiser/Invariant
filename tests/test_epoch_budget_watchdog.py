"""A4 epoch-budget-watchdog gates.

The watchdog's value is that its receipts cannot lie about budgets, so the
load-bearing tests are the controls: one epoch that completes under every declared
cap, and one epoch per cap kind that deliberately trips it and still yields a sealed
CAP_TRIPPED receipt.  The guard tests are just as load-bearing: a callable that
swallows or forges a trip must produce an error, never a receipt.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from sigma_theory_compiler.epoch_budget_watchdog import (
    CLAIMS,
    LEDGER_SCHEMA,
    RECEIPT_SCHEMA,
    EpochBudgetExceeded,
    EpochBudgetWatchdogError,
    main,
    run_epoch,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


class _FakeClock:
    """Injected monotonic source so time-driven paths are deterministic."""

    def __init__(self) -> None:
        self.ns = 0

    def advance_seconds(self, seconds: int) -> None:
        self.ns += seconds * 1_000_000_000

    def advance_ns(self, nanoseconds: int) -> None:
        self.ns += nanoseconds

    def __call__(self) -> int:
        return self.ns


def _write_ledger(path: Path, spent: int) -> None:
    path.write_text(
        json.dumps({"schema_version": LEDGER_SCHEMA, "spent_dollars_hundredths": spent}),
        encoding="utf-8",
    )


def _reseal(receipt: dict) -> dict:
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    return {**body, "content_sha256": canonical_sha256(body)}


@pytest.fixture(scope="module")
def wall_trip_receipt() -> dict:
    """One real slow-loop wall trip, shared by every test that inspects it."""

    def epoch(check):
        while True:
            time.sleep(0.02)
            check()

    return run_epoch(epoch, caps={"max_wall_seconds": 0}, probes=None)


# ---------------------------------------------------------------------------
# Control: completes under every declared cap
# ---------------------------------------------------------------------------


def test_control_epoch_completes_under_all_declared_caps(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, 250)

    def epoch(check):
        for index in range(20):
            (work_dir / f"artifact-{index}.bin").write_bytes(b"0123456789")
            check()

    receipt = run_epoch(
        epoch,
        caps={
            "max_wall_seconds": 30,
            "max_disk_write_bytes": 1_000_000,
            "max_llm_dollars_hundredths": 1000,
            "max_gpu_seconds": 10,
        },
        probes={
            "disk_directory": str(work_dir),
            "llm_ledger": str(ledger),
            "gpu_utilization_percent": lambda: 0,
        },
    )
    validate_receipt(receipt)
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["decision"] == "COMPLETED"
    assert receipt["claims"] == CLAIMS
    measured = receipt["measured"]
    assert measured["disk_write_bytes"] == 200
    assert measured["llm_dollars_hundredths"] == 0
    assert measured["gpu_seconds"] == 0
    assert 0 <= measured["wall_seconds"] <= 30
    assert receipt["probes"]["gpu_source"] == "probe_override"
    assert receipt["probes"]["overridden"] == ["gpu_utilization_percent"]
    assert set(receipt["caps"]) == {
        "max_wall_seconds",
        "max_disk_write_bytes",
        "max_llm_dollars_hundredths",
        "max_gpu_seconds",
    }


def test_none_callable_runs_an_empty_measured_epoch():
    receipt = run_epoch(None, caps={"max_wall_seconds": 30}, probes=None)
    validate_receipt(receipt)
    assert receipt["decision"] == "COMPLETED"
    assert receipt["measured"]["disk_write_bytes"] == 0
    assert receipt["measured"]["llm_dollars_hundredths"] == 0


# ---------------------------------------------------------------------------
# Controls: one deliberate trip per cap kind
# ---------------------------------------------------------------------------


def test_wall_cap_trips_via_slow_loop(wall_trip_receipt):
    validate_receipt(wall_trip_receipt)
    assert wall_trip_receipt["decision"] == "CAP_TRIPPED:max_wall_seconds"
    assert wall_trip_receipt["measured"]["wall_seconds"] >= 1
    assert wall_trip_receipt["caps"] == {"max_wall_seconds": 0}


def test_wall_seconds_round_down():
    clock = _FakeClock()

    def epoch(check):
        clock.advance_ns(1_900_000_000)  # 1.9 elapsed seconds
        check()

    receipt = run_epoch(
        epoch,
        caps={"max_wall_seconds": 30},
        probes={"monotonic_ns": clock, "gpu_utilization_percent": lambda: 0},
    )
    assert receipt["decision"] == "COMPLETED"
    assert receipt["measured"]["wall_seconds"] == 1


def test_disk_cap_trips_on_junk_writes_to_declared_directory(tmp_path):
    junk_dir = tmp_path / "declared"
    junk_dir.mkdir()

    def epoch(check):
        for index in range(64):
            (junk_dir / f"junk-{index}.bin").write_bytes(b"\0" * 4096)
            check()

    receipt = run_epoch(
        epoch,
        caps={"max_disk_write_bytes": 1000, "max_wall_seconds": 60},
        probes={"disk_directory": str(junk_dir), "gpu_utilization_percent": lambda: 0},
    )
    validate_receipt(receipt)
    assert receipt["decision"] == "CAP_TRIPPED:max_disk_write_bytes"
    assert receipt["measured"]["disk_write_bytes"] >= 4096
    assert receipt["probes"]["disk_directory"] == str(junk_dir)


def test_disk_measurement_is_peak_growth_so_deleting_junk_cannot_hide_it(tmp_path):
    work_dir = tmp_path / "declared"
    work_dir.mkdir()

    def epoch(check):
        target = work_dir / "burst.bin"
        target.write_bytes(b"\0" * 5000)
        check()
        target.unlink()
        check()

    receipt = run_epoch(
        epoch,
        caps={"max_disk_write_bytes": 10_000, "max_wall_seconds": 60},
        probes={"disk_directory": str(work_dir), "gpu_utilization_percent": lambda: 0},
    )
    assert receipt["decision"] == "COMPLETED"
    assert receipt["measured"]["disk_write_bytes"] == 5000


def test_llm_cap_trips_via_crafted_ledger_growth(tmp_path):
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, 100)

    def epoch(check):
        _write_ledger(ledger, 700)  # growth of 600 hundredths during the epoch
        while True:
            check()

    receipt = run_epoch(
        epoch,
        caps={"max_llm_dollars_hundredths": 500, "max_wall_seconds": 60},
        probes={"llm_ledger": str(ledger), "gpu_utilization_percent": lambda: 0},
    )
    validate_receipt(receipt)
    assert receipt["decision"] == "CAP_TRIPPED:max_llm_dollars_hundredths"
    assert receipt["measured"]["llm_dollars_hundredths"] == 600  # growth, not the raw total


def test_gpu_cap_trips_deterministically_via_injected_probes():
    clock = _FakeClock()
    calls: list[str] = []

    def epoch(check):
        calls.append("ran")
        clock.advance_seconds(1)
        while True:
            check()

    receipt = run_epoch(
        epoch,
        caps={"max_gpu_seconds": 0, "max_wall_seconds": 60},
        probes={"monotonic_ns": clock, "gpu_utilization_percent": lambda: 100},
    )
    validate_receipt(receipt)
    assert calls == ["ran"]
    assert receipt["decision"] == "CAP_TRIPPED:max_gpu_seconds"
    assert receipt["measured"]["gpu_seconds"] == 1
    assert receipt["probes"]["gpu_source"] == "probe_override"
    assert receipt["probes"]["overridden"] == ["gpu_utilization_percent", "monotonic_ns"]


# ---------------------------------------------------------------------------
# Declared-source refusals
# ---------------------------------------------------------------------------


def test_llm_cap_with_missing_ledger_refuses_to_run(tmp_path):
    ran: list[str] = []

    def epoch(check):
        ran.append("ran")

    with pytest.raises(EpochBudgetWatchdogError, match="ledger missing"):
        run_epoch(
            epoch,
            caps={"max_llm_dollars_hundredths": 100},
            probes={"llm_ledger": str(tmp_path / "absent.json")},
        )
    with pytest.raises(EpochBudgetWatchdogError, match="llm_ledger"):
        run_epoch(epoch, caps={"max_llm_dollars_hundredths": 100}, probes=None)
    assert ran == []


def test_llm_cap_with_invalid_ledger_refuses_to_run(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{\"schema_version\": \"wrong\", \"spent_dollars_hundredths\": 1}",
                      encoding="utf-8")
    with pytest.raises(EpochBudgetWatchdogError, match="wrong shape or schema"):
        run_epoch(None, caps={"max_llm_dollars_hundredths": 100},
                  probes={"llm_ledger": str(ledger)})


def test_disk_cap_without_declared_directory_refuses_to_run(tmp_path):
    with pytest.raises(EpochBudgetWatchdogError, match="disk_directory"):
        run_epoch(None, caps={"max_disk_write_bytes": 100}, probes=None)
    with pytest.raises(EpochBudgetWatchdogError, match="does not exist"):
        run_epoch(
            None,
            caps={"max_disk_write_bytes": 100},
            probes={"disk_directory": str(tmp_path / "absent")},
        )


def test_gpu_cap_with_unavailable_gpu_refuses_to_run():
    with pytest.raises(EpochBudgetWatchdogError, match="no GPU utilization source"):
        run_epoch(
            None,
            caps={"max_gpu_seconds": 5},
            probes={"gpu_utilization_percent": lambda: None},
        )


def test_unavailable_gpu_without_gpu_cap_is_declared_null():
    receipt = run_epoch(
        None,
        caps={"max_wall_seconds": 30},
        probes={"gpu_utilization_percent": lambda: None},
    )
    validate_receipt(receipt)
    assert receipt["probes"]["gpu_source"] == "unavailable"
    assert receipt["measured"]["gpu_seconds"] is None


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="nvidia-smi absent")
def test_real_nvidia_smi_polling_measures_integer_gpu_seconds():
    def epoch(check):
        for _ in range(3):
            check()

    receipt = run_epoch(
        epoch, caps={"max_gpu_seconds": 100_000, "max_wall_seconds": 60}, probes=None
    )
    validate_receipt(receipt)
    assert receipt["probes"]["gpu_source"] == "nvidia-smi"
    assert isinstance(receipt["measured"]["gpu_seconds"], int)
    assert receipt["probes"]["overridden"] == []


# ---------------------------------------------------------------------------
# No work after a trip; no forged trips
# ---------------------------------------------------------------------------


def test_callable_swallowing_a_trip_fails_closed_with_no_receipt():
    def epoch(check):
        clockout = time.monotonic() + 30
        while time.monotonic() < clockout:
            try:
                time.sleep(0.02)
                check()
            except EpochBudgetExceeded:
                return  # swallowed: pretends the epoch finished cleanly

    with pytest.raises(EpochBudgetWatchdogError, match="swallowed"):
        run_epoch(epoch, caps={"max_wall_seconds": 0}, probes=None)


def test_repeated_checks_after_a_trip_keep_raising():
    observed: list[str] = []

    def epoch(check):
        clockout = time.monotonic() + 30
        while time.monotonic() < clockout:
            try:
                time.sleep(0.02)
                check()
            except EpochBudgetExceeded as error:
                observed.append(error.cap_name)
                if len(observed) >= 3:
                    raise
    receipt = run_epoch(epoch, caps={"max_wall_seconds": 0}, probes=None)
    assert observed == ["max_wall_seconds"] * 3
    assert receipt["decision"] == "CAP_TRIPPED:max_wall_seconds"


def test_forged_trip_from_the_callable_fails_closed():
    def epoch(check):
        raise EpochBudgetExceeded("max_wall_seconds", 99, 1)

    with pytest.raises(EpochBudgetWatchdogError, match="not armed"):
        run_epoch(epoch, caps={"max_wall_seconds": 30}, probes=None)


def test_terminal_audit_catches_a_cap_outrun_between_checkpoints():
    clock = _FakeClock()

    def epoch(check):
        clock.advance_seconds(10)  # never calls check() again

    receipt = run_epoch(
        epoch,
        caps={"max_wall_seconds": 2},
        probes={"monotonic_ns": clock, "gpu_utilization_percent": lambda: 0},
    )
    validate_receipt(receipt)
    assert receipt["decision"] == "CAP_TRIPPED:max_wall_seconds"
    assert receipt["measured"]["wall_seconds"] == 10


# ---------------------------------------------------------------------------
# Receipt integrity: tamper, reseal, determinism
# ---------------------------------------------------------------------------


def test_tampered_receipt_fails_seal_replay(wall_trip_receipt):
    tampered = json.loads(json.dumps(wall_trip_receipt))
    tampered["decision"] = "COMPLETED"
    with pytest.raises(EpochBudgetWatchdogError, match="above its cap"):
        validate_receipt(tampered)
    tampered = json.loads(json.dumps(wall_trip_receipt))
    tampered["measured"]["wall_seconds"] += 1
    with pytest.raises(EpochBudgetWatchdogError, match="seal changed"):
        validate_receipt(tampered)


def test_resealed_decision_flip_fails_consistency_checks(wall_trip_receipt):
    flipped = json.loads(json.dumps(wall_trip_receipt))
    flipped["decision"] = "COMPLETED"
    with pytest.raises(EpochBudgetWatchdogError, match="above its cap"):
        validate_receipt(_reseal(flipped))

    undeclared = json.loads(json.dumps(wall_trip_receipt))
    undeclared["decision"] = "CAP_TRIPPED:max_disk_write_bytes"
    with pytest.raises(EpochBudgetWatchdogError, match="never declared"):
        validate_receipt(_reseal(undeclared))

    inside = json.loads(json.dumps(wall_trip_receipt))
    inside["measured"]["wall_seconds"] = 0
    with pytest.raises(EpochBudgetWatchdogError, match="inside its cap"):
        validate_receipt(_reseal(inside))

    wrong_claims = json.loads(json.dumps(wall_trip_receipt))
    wrong_claims["claims"] = {**CLAIMS, "caps_are_hard": False}
    with pytest.raises(EpochBudgetWatchdogError, match="claims"):
        validate_receipt(_reseal(wrong_claims))


def test_receipt_is_deterministic_modulo_the_measured_block(tmp_path):
    work_dir = tmp_path / "shared"
    work_dir.mkdir()
    ledger = tmp_path / "ledger.json"
    _write_ledger(ledger, 0)
    caps = {"max_wall_seconds": 30, "max_disk_write_bytes": 10_000,
            "max_llm_dollars_hundredths": 100}
    probes = {
        "disk_directory": str(work_dir),
        "llm_ledger": str(ledger),
        "gpu_utilization_percent": lambda: 0,
    }

    def epoch(check):
        for _ in range(5):
            check()

    first = run_epoch(epoch, caps=caps, probes=probes)
    second = run_epoch(epoch, caps=caps, probes=probes)
    validate_receipt(first)
    validate_receipt(second)

    def _without_measured(receipt: dict) -> dict:
        return {
            key: value
            for key, value in receipt.items()
            if key not in ("measured", "content_sha256")
        }

    assert _without_measured(first) == _without_measured(second)
    assert set(first["measured"]) == set(second["measured"])


def test_bad_caps_and_probes_are_rejected():
    with pytest.raises(EpochBudgetWatchdogError, match="unknown caps"):
        run_epoch(None, caps={"max_thought_seconds": 1}, probes=None)
    with pytest.raises(EpochBudgetWatchdogError, match="nonnegative integer"):
        run_epoch(None, caps={"max_wall_seconds": -1}, probes=None)
    with pytest.raises(EpochBudgetWatchdogError, match="nonnegative integer"):
        run_epoch(None, caps={"max_wall_seconds": True}, probes=None)
    with pytest.raises(EpochBudgetWatchdogError, match="nonnegative integer"):
        run_epoch(None, caps={"max_wall_seconds": 1.5}, probes=None)
    with pytest.raises(EpochBudgetWatchdogError, match="unknown probes"):
        run_epoch(None, caps={}, probes={"cpu_percent": lambda: 0})
    with pytest.raises(EpochBudgetWatchdogError, match="must be callable"):
        run_epoch(None, caps={}, probes={"monotonic_ns": 5})


# ---------------------------------------------------------------------------
# CLI demos
# ---------------------------------------------------------------------------


def test_cli_demo_complete_writes_a_sealed_receipt(tmp_path, capsys):
    output = tmp_path / "receipts" / "complete.json"
    assert main(["--demo-complete", "--output", str(output)]) == 0
    assert capsys.readouterr().out.startswith("COMPLETED ")
    receipt = json.loads(output.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    assert receipt["decision"] == "COMPLETED"


def test_cli_demo_trip_wall_writes_a_sealed_trip_receipt(tmp_path, capsys):
    output = tmp_path / "receipts" / "trip-wall.json"
    assert main(["--demo-trip", "wall", "--output", str(output)]) == 0
    assert capsys.readouterr().out.startswith("CAP_TRIPPED:max_wall_seconds ")
    receipt = json.loads(output.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    assert receipt["decision"] == "CAP_TRIPPED:max_wall_seconds"
    assert receipt["measured"]["wall_seconds"] >= 2

    # Immutable output: a conflicting receipt at the same path is refused.
    output.write_bytes(b"{}")
    with pytest.raises(EpochBudgetWatchdogError, match="refusing to overwrite"):
        main(["--demo-trip", "wall", "--output", str(output)])
