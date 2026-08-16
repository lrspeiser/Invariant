"""A4 — guarded epoch runner with hard, receipt-sealed resource caps.

Continuous operation is only safe when every epoch runs inside declared budgets and
leaves a sealed record of what it actually consumed.  This module runs one epoch
callable in the same process under cooperative enforcement: the callable receives a
``check()`` callback it must call periodically; every call samples the declared
resource probes and, the moment a cap is exceeded, raises :class:`EpochBudgetExceeded`
from inside the callable.  ``run_epoch`` catches that trip, marks the epoch
``CAP_TRIPPED:<cap>``, and still writes a sealed receipt — a tripped budget is a
recorded outcome, never a crash and never silence.

Three rules keep the watchdog honest.

**Caps are hard, including at the finish line.**  Enforcement is cooperative between
checkpoints, but the terminal audit is not: after the callable returns, the caps are
checked one final time against the measured totals, so an epoch that outran a budget
between checkpoints is still marked tripped rather than COMPLETED.

**No work after a trip.**  A trip raised by ``check()`` must propagate.  A callable
that swallows the exception and keeps running is detected when it returns, and the
run fails closed with an error instead of a receipt.  A callable that forges an
:class:`EpochBudgetExceeded` the watchdog never armed is likewise an error, never a
CAP_TRIPPED receipt.

**Measurements are declared, integer, and echoed.**  Disk growth is measured over a
declared directory; LLM spend is read from a declared integer-hundredths ledger file
and refused when the ledger is absent; GPU seconds come from polling ``nvidia-smi``
and, when that is unavailable, the GPU cap must be absent or the run refuses.  Any
probe overridden for testing is named in the receipt.

Claim boundary: the receipt records enforced budgets and measured consumption for one
epoch.  It claims nothing about what the epoch computed, and completion under caps
establishes no correctness, novelty, or significance of the work itself.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import SigmaCoreError, canonical_json_bytes, canonical_sha256

RECEIPT_SCHEMA = "invariant-epoch-budget-watchdog-1.0"
LEDGER_SCHEMA = "invariant-llm-spend-ledger-1.0"

#: Declared caps and the measured field each one bounds.
CAP_MEASURED_FIELD = {
    "max_disk_write_bytes": "disk_write_bytes",
    "max_gpu_seconds": "gpu_seconds",
    "max_llm_dollars_hundredths": "llm_dollars_hundredths",
    "max_wall_seconds": "wall_seconds",
}
CAP_NAMES = tuple(sorted(CAP_MEASURED_FIELD))

MEASURED_KEYS = {"disk_write_bytes", "gpu_seconds", "llm_dollars_hundredths", "wall_seconds"}
TOP_LEVEL_KEYS = {"schema_version", "caps", "measured", "decision", "claims", "probes", "scope",
                  "content_sha256"}
PROBE_DECL_KEYS = {"disk_directory", "llm_ledger"}
PROBE_OVERRIDE_KEYS = {"monotonic_ns", "disk_usage_bytes", "gpu_utilization_percent",
                       "llm_spent_hundredths"}
GPU_SOURCES = ("nvidia-smi", "probe_override", "unavailable")

_DECISION = re.compile(r"^(COMPLETED|CAP_TRIPPED:(" + "|".join(CAP_NAMES) + r"))$")

CLAIMS = {
    "caps_are_hard": True,
    "corpus_absence_establishes_novelty": False,
    "partial_work_after_trip": False,
}

SCOPE = (
    "One epoch executed in-process under declared hard resource caps with cooperative "
    "checkpoints and a terminal audit. Measured values are integers: wall seconds "
    "round down, disk bytes are the peak observed growth of the declared directory, "
    "LLM spend is the peak observed growth of the declared integer-hundredths ledger, "
    "and GPU seconds count checkpoint-sampled busy seconds (a lower bound between "
    "checkpoints). CAP_TRIPPED ends the epoch with no further work; COMPLETED means "
    "every declared cap held through the terminal audit. The receipt claims nothing "
    "about the correctness, novelty, or significance of the epoch's work."
)


class EpochBudgetWatchdogError(ValueError):
    """Raised when caps, probes, declarations, or receipt integrity are violated."""


class EpochBudgetExceeded(RuntimeError):
    """Raised by ``check()`` inside the epoch callable when a declared cap trips."""

    def __init__(self, cap_name: str, measured: int, cap: int) -> None:
        super().__init__(f"{cap_name} tripped: measured {measured} > cap {cap}")
        self.cap_name = cap_name
        self.measured = measured
        self.cap = cap


# ---------------------------------------------------------------------------
# Default probes
# ---------------------------------------------------------------------------


def _default_monotonic_ns() -> int:
    return time.monotonic_ns()


def _default_disk_usage_bytes(directory: str) -> int:
    total = 0
    stack = [Path(directory)]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError as error:
            raise EpochBudgetWatchdogError(f"declared disk directory unreadable: {error}") from error
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                total += entry.stat().st_size
    return total


def _default_llm_spent_hundredths(ledger_path: str) -> int:
    path = Path(ledger_path)
    if not path.is_file():
        raise EpochBudgetWatchdogError(f"declared LLM ledger missing: {ledger_path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EpochBudgetWatchdogError(f"declared LLM ledger unreadable: {error}") from error
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "spent_dollars_hundredths"}
        or value["schema_version"] != LEDGER_SCHEMA
    ):
        raise EpochBudgetWatchdogError("declared LLM ledger has the wrong shape or schema")
    spent = value["spent_dollars_hundredths"]
    if not isinstance(spent, int) or isinstance(spent, bool) or spent < 0:
        raise EpochBudgetWatchdogError("ledger spent_dollars_hundredths must be a nonnegative int")
    return spent


def _default_gpu_utilization_percent() -> int | None:
    """Max utilization percent across GPUs via nvidia-smi, or None when unavailable."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    readings: list[int] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                readings.append(int(line))
            except ValueError:
                return None
    return max(readings) if readings else None


# ---------------------------------------------------------------------------
# Validation of caps and probes
# ---------------------------------------------------------------------------


def _validate_caps(caps: Any) -> dict[str, int]:
    if not isinstance(caps, Mapping):
        raise EpochBudgetWatchdogError("caps must be an object")
    unknown = set(caps) - set(CAP_NAMES)
    if unknown:
        raise EpochBudgetWatchdogError(f"unknown caps: {sorted(unknown)}")
    result: dict[str, int] = {}
    for name in sorted(caps):
        value = caps[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EpochBudgetWatchdogError(f"cap {name} must be a nonnegative integer")
        result[name] = value
    return result


def _validate_probes(probes: Any) -> dict[str, Any]:
    if probes is None:
        return {}
    if not isinstance(probes, Mapping):
        raise EpochBudgetWatchdogError("probes must be an object or None")
    unknown = set(probes) - PROBE_DECL_KEYS - PROBE_OVERRIDE_KEYS
    if unknown:
        raise EpochBudgetWatchdogError(f"unknown probes: {sorted(unknown)}")
    result: dict[str, Any] = {}
    for name in PROBE_DECL_KEYS:
        if name in probes:
            value = probes[name]
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise EpochBudgetWatchdogError(f"probe {name} must be a nonempty path")
            result[name] = str(value)
    for name in PROBE_OVERRIDE_KEYS:
        if name in probes:
            if not callable(probes[name]):
                raise EpochBudgetWatchdogError(f"probe override {name} must be callable")
            result[name] = probes[name]
    return result


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class _Monitor:
    """Samples declared probes, enforces caps, and arms every trip it raises."""

    def __init__(self, caps: Mapping[str, int], probes: Mapping[str, Any]) -> None:
        self.caps = dict(caps)
        self.disk_directory: str | None = probes.get("disk_directory")
        self.llm_ledger: str | None = probes.get("llm_ledger")
        self.overridden = sorted(set(probes) & PROBE_OVERRIDE_KEYS)
        self._monotonic_ns = probes.get("monotonic_ns", _default_monotonic_ns)
        self._disk_usage = probes.get("disk_usage_bytes", _default_disk_usage_bytes)
        self._llm_spent = probes.get("llm_spent_hundredths", _default_llm_spent_hundredths)
        self._gpu_utilization = probes.get(
            "gpu_utilization_percent", _default_gpu_utilization_percent
        )

        if "max_disk_write_bytes" in self.caps and self.disk_directory is None:
            raise EpochBudgetWatchdogError(
                "max_disk_write_bytes declared without a probes disk_directory"
            )
        if self.disk_directory is not None and not Path(self.disk_directory).is_dir():
            raise EpochBudgetWatchdogError(
                f"declared disk directory does not exist: {self.disk_directory}"
            )
        if "max_llm_dollars_hundredths" in self.caps and self.llm_ledger is None:
            raise EpochBudgetWatchdogError(
                "max_llm_dollars_hundredths declared without a probes llm_ledger"
            )

        if "gpu_utilization_percent" in probes:
            probe_source = "probe_override"
        else:
            probe_source = "nvidia-smi"
        first_reading = self._gpu_reading()
        if first_reading is None:
            self.gpu_source = "unavailable"
            if "max_gpu_seconds" in self.caps:
                raise EpochBudgetWatchdogError(
                    "max_gpu_seconds declared but no GPU utilization source is available"
                )
        else:
            self.gpu_source = probe_source

        self._start_ns = 0
        self._disk_start = 0
        self._llm_start = 0
        self._disk_peak = 0
        self._llm_peak = 0
        self._gpu_seconds = 0
        self._gpu_last_sampled_second = 0
        self.tripped: EpochBudgetExceeded | None = None
        self._armed: list[EpochBudgetExceeded] = []

    # -- probe plumbing ----------------------------------------------------

    def _gpu_reading(self) -> int | None:
        reading = self._gpu_utilization()
        if reading is None:
            return None
        if not isinstance(reading, int) or isinstance(reading, bool) or reading < 0:
            raise EpochBudgetWatchdogError("GPU utilization probe must return None or an int >= 0")
        return reading

    def _now_ns(self) -> int:
        value = self._monotonic_ns()
        if not isinstance(value, int) or isinstance(value, bool):
            raise EpochBudgetWatchdogError("monotonic probe must return an integer")
        return value

    def start(self) -> None:
        self._start_ns = self._now_ns()
        if self.disk_directory is not None:
            self._disk_start = self._usage_reading()
        if self.llm_ledger is not None:
            self._llm_start = self._spent_reading()

    def _usage_reading(self) -> int:
        value = self._disk_usage(self.disk_directory)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EpochBudgetWatchdogError("disk usage probe must return a nonnegative integer")
        return value

    def _spent_reading(self) -> int:
        value = self._llm_spent(self.llm_ledger)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EpochBudgetWatchdogError("LLM spend probe must return a nonnegative integer")
        return value

    # -- sampling ----------------------------------------------------------

    def _sample(self) -> dict[str, int | None]:
        elapsed_ns = self._now_ns() - self._start_ns
        if elapsed_ns < 0:
            raise EpochBudgetWatchdogError("monotonic time went backwards")
        wall_seconds = elapsed_ns // 1_000_000_000
        if self.disk_directory is not None:
            growth = self._usage_reading() - self._disk_start
            self._disk_peak = max(self._disk_peak, growth)
        if self.llm_ledger is not None:
            growth = self._spent_reading() - self._llm_start
            if growth < 0:
                raise EpochBudgetWatchdogError("LLM ledger spend regressed during the epoch")
            self._llm_peak = max(self._llm_peak, growth)
        if self.gpu_source != "unavailable" and wall_seconds > self._gpu_last_sampled_second:
            reading = self._gpu_reading()
            if reading is None:
                raise EpochBudgetWatchdogError("GPU utilization source vanished during the epoch")
            if reading > 0:
                self._gpu_seconds += 1
            self._gpu_last_sampled_second = wall_seconds
        return {
            "disk_write_bytes": self._disk_peak,
            "gpu_seconds": self._gpu_seconds if self.gpu_source != "unavailable" else None,
            "llm_dollars_hundredths": self._llm_peak,
            "wall_seconds": wall_seconds,
        }

    # -- enforcement -------------------------------------------------------

    def owns(self, error: EpochBudgetExceeded) -> bool:
        return any(error is armed for armed in self._armed)

    def check(self) -> None:
        """Cooperative checkpoint for the epoch callable.  Raises on any tripped cap."""

        if self.tripped is not None:
            replay = EpochBudgetExceeded(
                self.tripped.cap_name, self.tripped.measured, self.tripped.cap
            )
            self._armed.append(replay)
            raise replay
        measured = self._sample()
        for cap_name in sorted(self.caps):
            observed = measured[CAP_MEASURED_FIELD[cap_name]]
            if observed is not None and observed > self.caps[cap_name]:
                trip = EpochBudgetExceeded(cap_name, observed, self.caps[cap_name])
                self.tripped = trip
                self._armed.append(trip)
                raise trip

    def finalize(self) -> dict[str, int | None]:
        """Terminal audit: one last sample and cap check, then the measured block."""

        measured = self._sample()
        if self.tripped is None:
            for cap_name in sorted(self.caps):
                observed = measured[CAP_MEASURED_FIELD[cap_name]]
                if observed is not None and observed > self.caps[cap_name]:
                    self.tripped = EpochBudgetExceeded(cap_name, observed, self.caps[cap_name])
                    break
        return measured


# ---------------------------------------------------------------------------
# The guarded runner
# ---------------------------------------------------------------------------


def run_epoch(
    epoch_callable: Callable[[Callable[[], None]], Any] | None,
    *,
    caps: Mapping[str, Any],
    probes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one epoch under hard caps and return its sealed receipt.

    ``epoch_callable`` receives the cooperative ``check()`` callback and must call it
    periodically; ``None`` runs an empty epoch (measurement and audit only).  A cap
    trip yields a ``CAP_TRIPPED:<cap>`` receipt; a callable that swallows a trip, or
    forges one, is an error and produces no receipt.
    """

    validated_caps = _validate_caps(caps)
    validated_probes = _validate_probes(probes)
    monitor = _Monitor(validated_caps, validated_probes)
    monitor.start()
    if epoch_callable is not None:
        try:
            epoch_callable(monitor.check)
        except EpochBudgetExceeded as error:
            if not monitor.owns(error):
                raise EpochBudgetWatchdogError(
                    "EpochBudgetExceeded raised by the callable was not armed by this watchdog"
                ) from error
        else:
            if monitor.tripped is not None:
                raise EpochBudgetWatchdogError(
                    "epoch callable swallowed a cap trip and continued running"
                )
    measured = monitor.finalize()
    decision = (
        "COMPLETED" if monitor.tripped is None else f"CAP_TRIPPED:{monitor.tripped.cap_name}"
    )
    body = {
        "caps": validated_caps,
        "claims": CLAIMS,
        "decision": decision,
        "measured": measured,
        "probes": {
            "disk_directory": monitor.disk_directory,
            "gpu_source": monitor.gpu_source,
            "llm_ledger": monitor.llm_ledger,
            "overridden": monitor.overridden,
        },
        "schema_version": RECEIPT_SCHEMA,
        "scope": SCOPE,
    }
    receipt = {**body, "content_sha256": canonical_sha256(body)}
    validate_receipt(receipt)
    return receipt


# ---------------------------------------------------------------------------
# Receipt validation (structure + seal replay; measured values are not re-run)
# ---------------------------------------------------------------------------


def _measured_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EpochBudgetWatchdogError(f"{label} must be a nonnegative integer")
    return value


def validate_receipt(value: Any) -> None:
    """Reject structural, decision-consistency, claims, or seal violations.

    Deliberately does not re-run the epoch: measured values are environment facts.
    The seal is replayed from the receipt body, and the decision must be consistent
    with the caps echo and the measured block.
    """

    if not isinstance(value, Mapping) or set(value) != TOP_LEVEL_KEYS:
        raise EpochBudgetWatchdogError("receipt top-level keys changed")
    if value["schema_version"] != RECEIPT_SCHEMA:
        raise EpochBudgetWatchdogError("receipt schema changed")
    if value["claims"] != CLAIMS:
        raise EpochBudgetWatchdogError("receipt claims changed")
    if not isinstance(value["scope"], str) or not value["scope"].strip():
        raise EpochBudgetWatchdogError("receipt scope must be a nonempty string")
    caps = _validate_caps(value["caps"])
    measured = value["measured"]
    if not isinstance(measured, Mapping) or set(measured) != MEASURED_KEYS:
        raise EpochBudgetWatchdogError("measured keys changed")
    for name in sorted(MEASURED_KEYS - {"gpu_seconds"}):
        _measured_int(measured[name], f"measured.{name}")
    if measured["gpu_seconds"] is not None:
        _measured_int(measured["gpu_seconds"], "measured.gpu_seconds")
    probes = value["probes"]
    if not isinstance(probes, Mapping) or set(probes) != {
        "disk_directory", "gpu_source", "llm_ledger", "overridden"
    }:
        raise EpochBudgetWatchdogError("probes keys changed")
    for name in ("disk_directory", "llm_ledger"):
        if probes[name] is not None and (
            not isinstance(probes[name], str) or not probes[name].strip()
        ):
            raise EpochBudgetWatchdogError(f"probes.{name} must be null or a nonempty string")
    if probes["gpu_source"] not in GPU_SOURCES:
        raise EpochBudgetWatchdogError(f"probes.gpu_source must be one of {GPU_SOURCES}")
    overridden = probes["overridden"]
    if (
        not isinstance(overridden, Sequence)
        or isinstance(overridden, (str, bytes))
        or list(overridden) != sorted(set(overridden))
        or not set(overridden) <= PROBE_OVERRIDE_KEYS
    ):
        raise EpochBudgetWatchdogError("probes.overridden must be sorted unique probe names")
    if probes["gpu_source"] == "unavailable":
        if measured["gpu_seconds"] is not None:
            raise EpochBudgetWatchdogError("gpu_seconds must be null when GPU is unavailable")
        if "max_gpu_seconds" in caps:
            raise EpochBudgetWatchdogError("max_gpu_seconds cap requires an available GPU source")
    elif measured["gpu_seconds"] is None:
        raise EpochBudgetWatchdogError("gpu_seconds must be an integer when a GPU source exists")
    if "max_disk_write_bytes" in caps and probes["disk_directory"] is None:
        raise EpochBudgetWatchdogError("disk cap requires a declared disk_directory")
    if "max_llm_dollars_hundredths" in caps and probes["llm_ledger"] is None:
        raise EpochBudgetWatchdogError("LLM cap requires a declared llm_ledger")

    decision = value["decision"]
    if not isinstance(decision, str) or _DECISION.fullmatch(decision) is None:
        raise EpochBudgetWatchdogError("decision must be COMPLETED or CAP_TRIPPED:<declared cap>")
    if decision == "COMPLETED":
        for cap_name, cap in caps.items():
            observed = measured[CAP_MEASURED_FIELD[cap_name]]
            if observed is not None and observed > cap:
                raise EpochBudgetWatchdogError(
                    f"COMPLETED receipt has measured {cap_name} above its cap"
                )
    else:
        cap_name = decision.split(":", 1)[1]
        if cap_name not in caps:
            raise EpochBudgetWatchdogError("tripped cap was never declared in the caps echo")
        observed = measured[CAP_MEASURED_FIELD[cap_name]]
        if observed is None or observed <= caps[cap_name]:
            raise EpochBudgetWatchdogError(
                "CAP_TRIPPED receipt has measured value inside its cap"
            )

    body = {key: item for key, item in value.items() if key != "content_sha256"}
    try:
        expected = canonical_sha256(body)
    except SigmaCoreError as error:
        raise EpochBudgetWatchdogError(
            f"receipt is not canonically encodable: {error}"
        ) from error
    if value["content_sha256"] != expected:
        raise EpochBudgetWatchdogError("receipt seal changed")


# ---------------------------------------------------------------------------
# Demo epochs and CLI
# ---------------------------------------------------------------------------


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise EpochBudgetWatchdogError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _demo_complete() -> dict[str, Any]:
    def epoch(check: Callable[[], None]) -> None:
        total = 0
        for index in range(1000):
            total += index * index
            check()

    return run_epoch(epoch, caps={"max_wall_seconds": 60}, probes=None)


def _demo_trip(cap: str, scratch: Path) -> dict[str, Any]:
    if cap == "wall":
        def epoch(check: Callable[[], None]) -> None:
            while True:
                time.sleep(0.02)
                check()

        return run_epoch(epoch, caps={"max_wall_seconds": 1}, probes=None)
    if cap == "disk":
        disk_dir = scratch / "demo-disk"
        disk_dir.mkdir(parents=True, exist_ok=True)

        def epoch(check: Callable[[], None]) -> None:
            for index in range(64):
                (disk_dir / f"junk-{index}.bin").write_bytes(b"\0" * 4096)
                check()

        return run_epoch(
            epoch,
            caps={"max_disk_write_bytes": 8192, "max_wall_seconds": 60},
            probes={"disk_directory": str(disk_dir)},
        )
    if cap == "llm":
        ledger = scratch / "demo-ledger.json"
        ledger.write_text(
            json.dumps({"schema_version": LEDGER_SCHEMA, "spent_dollars_hundredths": 0}),
            encoding="utf-8",
        )

        def epoch(check: Callable[[], None]) -> None:
            ledger.write_text(
                json.dumps({"schema_version": LEDGER_SCHEMA, "spent_dollars_hundredths": 500}),
                encoding="utf-8",
            )
            while True:
                check()

        return run_epoch(
            epoch,
            caps={"max_llm_dollars_hundredths": 100, "max_wall_seconds": 60},
            probes={"llm_ledger": str(ledger)},
        )
    if cap == "gpu":
        def epoch(check: Callable[[], None]) -> None:
            while True:
                time.sleep(0.02)
                check()

        return run_epoch(
            epoch,
            caps={"max_gpu_seconds": 0, "max_wall_seconds": 60},
            probes={"gpu_utilization_percent": lambda: 100},
        )
    raise EpochBudgetWatchdogError(f"unknown demo cap: {cap}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Epoch budget watchdog (A4).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo-complete", action="store_true", help="run a control epoch that stays under caps"
    )
    group.add_argument(
        "--demo-trip",
        choices=("wall", "disk", "llm", "gpu"),
        help="run a control epoch that deliberately trips the named cap",
    )
    parser.add_argument("--output", required=True, help="path for the sealed epoch receipt")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.demo_complete:
        receipt = _demo_complete()
    else:
        scratch = output.parent / f"{output.stem}.scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            receipt = _demo_trip(args.demo_trip, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    _write_immutable(output, receipt)
    print(f"{receipt['decision']} content_sha256={receipt['content_sha256']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
