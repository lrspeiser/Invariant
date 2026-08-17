"""A1 — continuous discovery scheduler over a durable SQLite work ledger.

Every discovery capability in this repository is a hand-cranked module: the A2 queue
declares targets, B3 proposes conjectures, B5/B6 prove restricted shapes, M7 hunts
counterexamples, A3 renders status, A4 caps an epoch.  Nothing keeps turning the crank.
This module is the crank: it derives work from the sealed problem queue, executes it in
parallel through the *landed* modules, and leaves receipts for every step — unattended,
resumable, and honest about what it could not do.

Three rules keep continuous operation trustworthy.

**The ledger is the only memory.**  Work items are keyed ``(problem_id, stage,
input_hash)`` in SQLite (WAL).  A completed key is never rerun; a lease that outlives
its holder is reclaimed after expiry and counted as a crash recovery; resuming is the
same command as starting.  Every state change appends to a hash-chained event log —
each event carries the SHA-256 of its predecessor — so history cannot be silently
edited, only extended.

**Every stage ends in a sealed receipt or a typed blocker.  Never silence.**  A stage
the system cannot perform (an unregistered row generator, a conjecture kind with no
prover, a sequence M7 cannot sweep) produces a typed blocker record such as
``missing_prover:index_scaling_relation``.  The blocker list *is* the build queue for
the next capability, extracted from real failures rather than guessed.

**Epochs run inside the A4 watchdog and end at the dashboard.**  Each epoch executes
under declared hard caps via :func:`epoch_budget_watchdog.run_epoch`; a tripped cap is
a recorded epoch outcome, never a crash.  After the guarded block the A3 dashboard is
rebuilt from the real sources and an epoch summary receipt binds the watchdog receipt,
the dashboard hash, and the event-chain root.

**Fan-out: every applicable lane, or a recorded reason it was skipped.**  With
``--fanout`` an epoch derives work from the A5 lane registry — the cross product of
(problem x applicable lane) — instead of the fixed :data:`STAGES_BY_KIND` list, honoring
each lane's declared resource (CPU pool or the single GPU lease).  The epoch summary
records the lanes planned *and* the typed reason every other lane was not attempted, so
"nobody ever tried X on Y" is a fact in the receipt rather than an absence.  The end of
every epoch regenerates the A6 capability gap ledger from the whole corpus and binds its
top open gaps into the summary, so an epoch that produced new blockers cannot leave the
build queue stale.  ``--fanout`` is opt-in: without it the historical stage graph, and
every receipt already sealed against it, is byte-for-byte unchanged.

Claim boundary: the scheduler asserts orchestration facts only — what ran, what it
produced, what blocked it, and under what budgets.  Completion of an epoch establishes
no novelty, correctness, or significance beyond what each embedded receipt itself
claims, and ``unattended_operation`` is true only when zero interactive prompts
occurred.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any

from .capability_gap_ledger import build_ledger as build_gap_ledger
from .capability_gap_ledger import render_markdown as render_gap_markdown
from .capability_gap_ledger import top_open_gaps
from .conjecture_generation import generate_conjectures
from .discovery_dashboard import build_dashboard, render_html
from .epoch_budget_watchdog import LEDGER_SCHEMA as LLM_LEDGER_SCHEMA
from .epoch_budget_watchdog import run_epoch
from .gpu_counterexample_sweep import sweep as run_counterexample_sweep
from .lane_registry import REGISTRY_CONTENT_SHA256, lane, lane_decisions, lane_for_stage
from .lemma_decomposition import decompose_closed_form_proof
from .problem_queue import load_queue
from .quantified_inequality_proofs import prove_quantified_inequality
from .sigma_core import canonical_json_bytes, canonical_sha256

LEDGER_SCHEMA_VERSION = "invariant-discovery-ledger-1.0"
ITEM_RECEIPT_SCHEMA = "invariant-discovery-item-1.0"
EPOCH_RECEIPT_SCHEMA = "invariant-discovery-epoch-1.0"
SOAK_RECEIPT_SCHEMA = "invariant-discovery-soak-1.0"

GENESIS_SHA = "0" * 64

#: Stages derived from each declared machine-form kind, in execution order.
STAGES_BY_KIND: dict[str, tuple[str, ...]] = {
    "sequence_rows": ("generate_rows", "conjecture", "route_provers", "sweep"),
    "integer_trajectory": ("generate_rows", "conjecture", "route_provers", "sweep"),
    "diophantine_family": ("sweep",),
    "dataset_law_fit": ("note_gpu_campaign_receipts",),
    "module_target": ("route_provers",),
}

#: Hard bounds for the built-in row generators.  Exceeding one is a typed blocker.
GENERATOR_CAPS = {
    "max_rows": 64,
    "max_seed": 10**9,
    "max_trajectory_value": 10**12,
    "collatz_step_cap": 10000,
}

#: Per-generator declared row caps.  ``GENERATOR_CAPS["max_rows"]`` is the *engine*
#: emission budget — how many rows one scheduler item may seal — and a generator with no
#: entry here still hard-blocks a larger request, exactly as before.  A generator listed
#: here declares how far its own exact arithmetic is willing to run; a request beyond
#: either bound is served with the exact prefix plus a typed ``generator_cap_truncated``
#: record, never a silent stop.
GENERATOR_ROW_CAPS: dict[str, int] = {
    "gilbreath_leading_terms": 500,
    "pascal_interior_multiplicity": 8192,
    "recaman": 10**6,
    "reverse_and_add_base10": 10000,
    "twin_prime_count_pi2_10_pow_k": 7,
    "ulam_u_1_2": 5000,
}

#: Fixed inputs the six DG5 generators sieve or scan over, declared once here so the
#: numbers in a receipt are readable without opening the campaign module.
GILBREATH_PRIME_LIMIT = 10**6
PASCAL_ENTRY_MAX = 10**6
TWIN_PRIME_MAX_EXPONENT = 7

#: Conjecture kinds M7 can sweep, and the M7 sequence spec per known generator.
SWEEPABLE_KINDS = ("divisibility", "congruence", "index_scaling_relation")
SWEEP_SEQUENCES: dict[str, dict[str, Any]] = {
    "collatz_total_stopping_time": {
        "sequence": "collatz_total_stopping_time",
        "sequence_params": {},
        "step_cap": GENERATOR_CAPS["collatz_step_cap"],
    },
}
SWEEP_LO = 1
SWEEP_HI_CPU_DEFAULT = 10**6
SWEEP_HI_GPU_DEFAULT = 10**8

MAX_ATTEMPTS = 3
DEFAULT_CAPS = {"max_wall_seconds": 1800}
GPU_LEASE_SECONDS = 900

ITEM_CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "scalar_truth_or_probability_score": False,
    "stage_completion_establishes_novelty_or_significance": False,
    "typed_blockers_replace_silent_skips": True,
}

ITEM_SCOPE = (
    "One scheduler work item: a single stage of a single declared problem, executed "
    "through the landed discovery modules. The receipt embeds or hash-binds every "
    "inner sealed receipt it produced and records a typed blocker for anything the "
    "system could not do. It asserts orchestration facts only and claims nothing "
    "beyond what each embedded receipt itself claims."
)

EPOCH_SCOPE = (
    "One scheduler epoch: lease currently-runnable items from the durable ledger, "
    "execute them under the A4 epoch budget watchdog, write per-item sealed receipts, "
    "rebuild the A3 discovery dashboard from the real sources, and bind the watchdog "
    "receipt, dashboard hash, and event-chain root. Epoch completion establishes no "
    "novelty, correctness, or significance of the work itself."
)

SOAK_SCOPE = (
    "One unattended soak: repeated scheduler epochs until a declared wall budget. "
    "Counts are ledger facts (epochs run, items completed and blocked, reclaimed "
    "leases). zero_manual_steps is true only when zero interactive prompts occurred; "
    "the receipt claims nothing about the mathematical content of the work."
)

#: Interactive prompts issued by this module.  The scheduler never prompts; the
#: counter exists so the unattended-operation claim is measured, not asserted.
_INTERACTIVE_PROMPTS = 0

_ITEM_DELAY_ENV = "SIGMA_DISCOVERY_TEST_ITEM_DELAY_MS"

_DIVISIBILITY_RE = re.compile(r"^(\d+) divides a\(n\)$")
_CONGRUENCE_RE = re.compile(r"^a\(n\) = (\d+) \(mod (\d+)\)$")
_INDEX_SCALING_RE = re.compile(
    r"^a\((\d+)n\) = \((-?\d+(?:/\d+)?)\)\*a\(n\)(?: \+ \((-?\d+(?:/\d+)?)\))?$"
)


class DiscoverySchedulerError(ValueError):
    """Raised on ledger corruption, chain tamper, config violation, or receipt tamper."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": canonical_sha256(body)}


def _write_receipt(directory: Path, stem: str, receipt: Mapping[str, Any]) -> Path:
    """Write a sealed receipt under a content-addressed name.  Never overwrites."""

    encoded = canonical_json_bytes(receipt) + b"\n"
    path = directory / f"{stem}-{receipt['content_sha256'][:12]}.json"
    if path.exists():
        if path.read_bytes() != encoded:
            raise DiscoverySchedulerError(f"refusing to overwrite receipt: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Built-in row generators (pure Python, exact, cap-bounded)
# ---------------------------------------------------------------------------


def _collatz_sigma(n: int) -> int | None:
    value = n
    for steps in range(GENERATOR_CAPS["collatz_step_cap"] + 1):
        if value == 1:
            return steps
        value = 3 * value + 1 if value & 1 else value >> 1
    return None


def _generate_collatz(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    rows = []
    for point in range(1, machine_form["max_point"] + 1):
        sigma = _collatz_sigma(point)
        if sigma is None:
            raise _GeneratorBlocked("generator_cap_exceeded:collatz_total_stopping_time")
        rows.append({"point": point, "value": sigma})
    return rows, []


def _primes_first(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _generate_prime_gap(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    max_point = machine_form["max_point"]
    primes = _primes_first(max_point + 1)
    rows = [
        {"point": index, "value": primes[index] - primes[index - 1]}
        for index in range(1, max_point + 1)
    ]
    return rows, []


def _contfrac_e_term(n: int) -> int:
    """Euler's proved continued fraction of e: 2 then blocks (1, 2k, 1)."""

    if n == 1:
        return 2
    block, offset = divmod(n - 2, 3)
    return 2 * (block + 1) if offset == 1 else 1


def _generate_contfrac_e(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    rows = [
        {"point": point, "value": _contfrac_e_term(point)}
        for point in range(1, machine_form["max_point"] + 1)
    ]
    return rows, []


def _aliquot_step(n: int) -> int:
    """s(n) = sigma(n) - n by trial-division factorization.  Exact, pure Python."""

    if n <= 1:
        return 0
    total = 1
    remaining = n
    factor = 2
    while factor * factor <= remaining:
        if remaining % factor == 0:
            power_sum, power = 1, 1
            while remaining % factor == 0:
                remaining //= factor
                power *= factor
                power_sum += power
            total *= power_sum
        factor += 1 if factor == 2 else 2
    if remaining > 1:
        total *= 1 + remaining
    return total - n


def _generate_aliquot(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    name = machine_form.get("map", "aliquot_step_sum")
    seed = machine_form["seed"]
    if not 1 <= seed <= GENERATOR_CAPS["max_seed"]:
        raise _GeneratorBlocked(f"generator_cap_exceeded:{name}")
    rows = []
    value = seed
    for step in range(1, machine_form["max_steps"] + 1):
        value = _aliquot_step(value)
        if value > GENERATOR_CAPS["max_trajectory_value"]:
            # The already-computed prefix is exact; the stop is declared, never silent.
            blocker = _blocker(
                f"generator_cap_truncated:{name}",
                f"trajectory value exceeded the declared cap "
                f"{GENERATOR_CAPS['max_trajectory_value']} at step {step}; "
                f"{len(rows)} exact rows were emitted",
            )
            return rows, [blocker]
        rows.append({"point": step, "value": value})
        if value == 0:
            break
    return rows, []


class _GeneratorBlocked(Exception):
    """Internal: a generator refused its input; carried as a typed blocker."""


# ---------------------------------------------------------------------------
# DG5 generators.  Eight queue problems declared a row generator nobody had
# built, so every downstream lane on them recorded `upstream_blocked:generate_rows`
# and the corpus could not say whether the instruments had anything to find.  The
# exact arithmetic already existed inside the unsolved-dozen campaign, verified there
# against published anchors; these are thin adapters onto the scheduler's row
# contract, so the sequence is computed in exactly one place in this repository.
# ---------------------------------------------------------------------------


def _rows_from_terms(terms: Sequence[int], start_point: int = 1) -> list[dict[str, int]]:
    """Consecutive-point rows, the shape every row-consuming lane parses."""

    return [
        {"point": start_point + offset, "value": int(value)}
        for offset, value in enumerate(terms)
    ]


def _refuse_below(count: int, floor: int, name: str) -> None:
    if count < floor:
        raise _GeneratorBlocked(f"generator_cap_exceeded:{name}")


def _generate_ulam(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    """Ulam numbers U(1,2) — OEIS A002858, anchor-checked inside the campaign lane."""

    from .dozen_unsolved_progress_campaign import _ulam_lane

    count = machine_form["max_point"]
    _refuse_below(count, 8, "ulam_u_1_2")
    return _rows_from_terms(_ulam_lane(count)["b3_values"]), []


def _generate_twin_prime_counts(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    """pi_2(10^k) by sieve for k = 1..max_point — OEIS A007508, anchor-checked."""

    from .dozen_unsolved_progress_campaign import _twin_prime_lane

    exponent_max = machine_form["max_point"]
    _refuse_below(exponent_max, 1, "twin_prime_count_pi2_10_pow_k")
    if exponent_max > TWIN_PRIME_MAX_EXPONENT:
        raise _GeneratorBlocked("generator_cap_exceeded:twin_prime_count_pi2_10_pow_k")
    return _rows_from_terms(_twin_prime_lane(exponent_max)["b3_values"]), []


def _generate_gilbreath(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    """Leading term of the k-th iterated absolute difference row of the primes."""

    from .dozen_unsolved_progress_campaign import _gilbreath_lane

    rows = machine_form["max_point"]
    _refuse_below(rows, 8, "gilbreath_leading_terms")
    lane = _gilbreath_lane(rows, GILBREATH_PRIME_LIMIT)
    return _rows_from_terms(lane["b3_values"]), []


def _generate_pascal_multiplicity(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    """N(t), the multiplicity of t in Pascal's triangle — OEIS A003016, offset 2.

    The interior table is built once over every entry at most
    :data:`PASCAL_ENTRY_MAX`, so each emitted N(t) counts *all* occurrences of t and
    not merely those inside the emitted window.

    Rows are indexed from 1 like every other generator here — row ``n`` carries
    ``N(n + 1)`` — because the downstream ladder caps the absolute point value and an
    offset-2 index would push a full row budget past it.  The offset lives in this
    docstring and in the survey's sequence label, never in a silent index shift.
    """

    from .dozen_unsolved_progress_campaign import (
        _pascal_multiplicities,
        _singmaster_multiplicity,
    )

    count = machine_form["max_point"]
    _refuse_below(count, 1, "pascal_interior_multiplicity")
    if count + 1 > PASCAL_ENTRY_MAX:
        raise _GeneratorBlocked("generator_cap_exceeded:pascal_interior_multiplicity")
    interior = _pascal_multiplicities(PASCAL_ENTRY_MAX)
    terms = [_singmaster_multiplicity(t, interior) for t in range(2, count + 2)]
    return _rows_from_terms(terms), []


def _recaman_terms(count: int) -> list[int]:
    """a(0) = 0; a(n) = a(n-1) - n when that is positive and unvisited, else a(n-1) + n."""

    terms = [0]
    seen = {0}
    current = 0
    for step in range(1, count):
        back = current - step
        current = back if back > 0 and back not in seen else current + step
        seen.add(current)
        terms.append(current)
    return terms


def _generate_recaman(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    """Recaman's sequence — OEIS A005132, head-anchored against the campaign table."""

    from .dozen_unsolved_progress_campaign import RECAMAN_HEAD_ANCHOR, _anchor_check

    count = machine_form["max_steps"]
    _refuse_below(count, 1, "recaman")
    terms = _recaman_terms(count)
    _anchor_check(terms, RECAMAN_HEAD_ANCHOR, "recaman")
    return _rows_from_terms(terms), []


def _generate_reverse_and_add(
    machine_form: Mapping[str, Any],
) -> tuple[list[dict[str, int]], list[dict[str, str]]]:
    """Digit lengths of the base-10 reverse-and-add trajectory, exact bigint."""

    from .dozen_unsolved_progress_campaign import _lychrel_trajectory

    steps = machine_form["max_steps"]
    seed = machine_form["seed"]
    _refuse_below(steps, 8, "reverse_and_add_base10")
    if not 1 <= seed <= GENERATOR_CAPS["max_seed"]:
        raise _GeneratorBlocked("generator_cap_exceeded:reverse_and_add_base10")
    trajectory = _lychrel_trajectory(seed, steps)
    rows = _rows_from_terms(trajectory["digit_lengths"])
    if trajectory["palindrome_found"]:
        # Reaching a palindrome ends the declared row sequence before the request does.
        # It is not a resource cap, but it is the same obligation: say so, in the lane's
        # declared vocabulary, rather than returning a short list in silence.
        return rows, [
            _blocker(
                "generator_cap_truncated:reverse_and_add_base10",
                f"seed {seed} reached a palindrome at iteration "
                f"{trajectory['palindrome_at']}, which terminates the declared row "
                f"sequence; {len(rows)} exact rows were emitted",
            )
        ]
    return rows, []


#: Registry keys cover both the roadmap names and the queue's machine-form names.
#: Each generator returns (exact rows, typed truncation records).
GENERATOR_REGISTRY: dict[
    str, Callable[[Mapping[str, Any]], tuple[list[dict[str, int]], list[dict[str, str]]]]
] = {
    "collatz_total_stopping_time": _generate_collatz,
    "prime_gap": _generate_prime_gap,
    "contfrac_e_terms": _generate_contfrac_e,
    "e_continued_fraction_terms": _generate_contfrac_e,
    "aliquot_step_sum": _generate_aliquot,
    "aliquot_sum": _generate_aliquot,
    "gilbreath_leading_terms": _generate_gilbreath,
    "pascal_interior_multiplicity": _generate_pascal_multiplicity,
    "recaman": _generate_recaman,
    "reverse_and_add_base10": _generate_reverse_and_add,
    "twin_prime_count_pi2_10_pow_k": _generate_twin_prime_counts,
    "ulam_u_1_2": _generate_ulam,
}


def _generator_name(machine_form: Mapping[str, Any]) -> str:
    if machine_form["kind"] == "sequence_rows":
        return machine_form["generator"]
    return machine_form["map"]


def _row_count_request(machine_form: Mapping[str, Any]) -> int:
    if machine_form["kind"] == "sequence_rows":
        return machine_form["max_point"]
    return machine_form["max_steps"]


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _item_receipt(
    problem_id: str,
    stage: str,
    input_hash: str,
    status: str,
    payload: Mapping[str, Any],
    blockers: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    body = {
        "blockers": [dict(item) for item in blockers],
        "claims": ITEM_CLAIMS,
        "input_hash": input_hash,
        "payload": dict(payload),
        "problem_id": problem_id,
        "schema_version": ITEM_RECEIPT_SCHEMA,
        "scope": ITEM_SCOPE,
        "stage": stage,
        "status": status,
    }
    return _seal(body)


def _blocker(blocker_type: str, detail: str) -> dict[str, str]:
    return {"type": blocker_type, "detail": detail}


def _stage_generate_rows(entry: Mapping[str, Any], input_hash: str) -> dict[str, Any]:
    machine_form = entry["machine_form"]
    name = _generator_name(machine_form)
    generator = GENERATOR_REGISTRY.get(name)
    if generator is None:
        blocker = _blocker(
            f"missing_generator:{name}",
            f"no built-in row generator is registered for {name!r}",
        )
        return _item_receipt(
            entry["id"], "generate_rows", input_hash, "BLOCKED",
            {"generator": name, "machine_form": dict(machine_form)}, [blocker],
        )
    request = _row_count_request(machine_form)
    engine_cap = GENERATOR_CAPS["max_rows"]
    declared_cap = GENERATOR_ROW_CAPS.get(name)
    requested_form: Mapping[str, Any] = machine_form
    truncation: list[dict[str, str]] = []
    if declared_cap is None:
        # Legacy contract, byte-for-byte: a generator that declares no cap of its own
        # cannot serve a request beyond the shared engine budget, so it blocks.
        if request > engine_cap:
            blocker = _blocker(
                f"generator_cap_exceeded:{name}",
                f"requested rows exceed the declared cap {engine_cap}",
            )
            return _item_receipt(
                entry["id"], "generate_rows", input_hash, "BLOCKED",
                {"generator": name, "machine_form": dict(machine_form)}, [blocker],
            )
    elif request > min(declared_cap, engine_cap):
        emit = min(declared_cap, engine_cap)
        bound = "generator" if declared_cap <= engine_cap else "engine emission budget"
        key = "max_point" if machine_form["kind"] == "sequence_rows" else "max_steps"
        requested_form = {**machine_form, key: emit}
        truncation = [
            _blocker(
                f"generator_cap_truncated:{name}",
                f"{request} rows were requested; the {bound} cap binds at {emit} "
                f"(declared generator cap {declared_cap}, engine emission budget "
                f"{engine_cap}), so {emit} exact rows were emitted",
            )
        ]
    try:
        rows, truncations = generator(requested_form)
        truncations = [*truncation, *truncations]
    except _GeneratorBlocked as refused:
        blocker = _blocker(str(refused), "generator refused its declared input caps")
        return _item_receipt(
            entry["id"], "generate_rows", input_hash, "BLOCKED",
            {"generator": name, "machine_form": dict(machine_form)}, [blocker],
        )
    if not rows:
        blocker = _blocker(
            f"generator_cap_exceeded:{name}", "no row was computable inside the declared caps"
        )
        return _item_receipt(
            entry["id"], "generate_rows", input_hash, "BLOCKED",
            {"generator": name, "machine_form": dict(machine_form)}, [blocker],
        )
    payload = {
        "generator": name,
        "generator_caps": GENERATOR_CAPS,
        "machine_form": dict(machine_form),
        "row_count": len(rows),
        "rows": rows,
    }
    return _item_receipt(
        entry["id"], "generate_rows", input_hash, "COMPLETED", payload, truncations
    )


def _upstream_blocked_receipt(
    entry_id: str, stage: str, input_hash: str, upstream_stage: str
) -> dict[str, Any]:
    blocker = _blocker(
        f"upstream_blocked:{upstream_stage}",
        f"stage {upstream_stage} did not complete for this problem",
    )
    return _item_receipt(
        entry_id, stage, input_hash, "BLOCKED", {"upstream_stage": upstream_stage}, [blocker]
    )


def _stage_conjecture(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    rows_receipt = _load_json(Path(upstream["rows_receipt_path"]))
    if rows_receipt["status"] != "COMPLETED":
        return _upstream_blocked_receipt(entry["id"], "conjecture", input_hash, "generate_rows")
    result = generate_conjectures(rows_receipt["payload"]["rows"])
    payload = {
        "result": result,
        "rows_receipt_sha256": rows_receipt["content_sha256"],
        "survived_kinds": sorted(
            item["kind"] for item in result["conjectures"] if item.get("status") == "SURVIVED"
        ),
    }
    return _item_receipt(entry["id"], "conjecture", input_hash, "COMPLETED", payload, [])


def _polynomial_closed_form(rows: Sequence[Mapping[str, Any]]) -> list[int] | None:
    """Ascending nonnegative-integer coefficients when B1 finds a polynomial basis."""

    from .basis_synthesis import synthesize_basis

    result = synthesize_basis([{"point": r["point"], "value": r["value"]} for r in rows])
    if result["decision"] != "PASS":
        return None
    family = result["result"]["family_id"]
    if family != "constant" and not family.startswith("polynomial_"):
        return None
    coefficients: list[int] = []
    for item in result["result"]["coefficients"]:
        if item["denominator"] != 1 or item["numerator"] < 0:
            return None
        coefficients.append(item["numerator"])
    return coefficients


def _forward_difference_coefficients(coefficients: Sequence[int]) -> list[int]:
    """Exact ascending coefficients of f(n+1) - f(n) via binomial expansion."""

    degree = len(coefficients) - 1
    result = [0] * max(degree, 1)
    for power, coefficient in enumerate(coefficients):
        for lower in range(power):
            result[lower] += coefficient * comb(power, lower)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _lean_namespace(problem_id: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", problem_id)
    return "Discovery" + "".join(part.capitalize() for part in parts if part)


def _route_one_conjecture(
    problem_id: str, conjecture: Mapping[str, Any], polynomial: list[int] | None
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Route one surviving conjecture to a landed prover, or produce a typed blocker."""

    kind = conjecture["kind"]
    statement = conjecture["statement"]
    if kind in ("closed_form", "linear_recurrence"):
        if polynomial is None:
            return None, _blocker(
                f"missing_prover:{kind}",
                f"{statement!r}: no Nat-domain polynomial closed form is available "
                "for lemma decomposition",
            )
        problem = {
            "base_value": polynomial[0],
            "closed_form": list(polynomial),
            "namespace": _lean_namespace(problem_id),
            "sequence_name": "seq",
            "step": _forward_difference_coefficients(polynomial),
        }
        result = decompose_closed_form_proof(problem)
        route = {
            "conjecture_kind": kind,
            "conjecture_statement": statement,
            "prover": "lemma_decomposition",
            "result": result,
        }
        return route, None
    if kind in ("monotonicity", "sign"):
        if polynomial is None:
            return None, _blocker(
                f"missing_prover:{kind}",
                f"{statement!r}: no Nat-domain polynomial closed form is available "
                "for a quantified inequality proof",
            )
        if kind == "monotonicity":
            if "<" not in statement:
                return None, _blocker(
                    "missing_prover:monotonicity",
                    f"{statement!r}: decreasing monotonicity has no declared relation",
                )
            relation = "monotone_increasing"
        else:
            if ">" not in statement:
                return None, _blocker(
                    "missing_prover:sign",
                    f"{statement!r}: negative sign has no Nat-domain relation",
                )
            relation = "nonnegative"
        problem = {
            "coefficients": list(polynomial),
            "name": f"seq{relation.title().replace('_', '')}",
            "namespace": _lean_namespace(problem_id),
            "relation": relation,
        }
        result = prove_quantified_inequality(problem)
        route = {
            "conjecture_kind": kind,
            "conjecture_statement": statement,
            "prover": "quantified_inequality_proofs",
            "routed_relation": relation,
            "result": result,
        }
        return route, None
    return None, _blocker(
        f"missing_prover:{kind}",
        f"{statement!r}: no landed prover accepts statement kind {kind!r}",
    )


def _survivors(conjecture_receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in conjecture_receipt["payload"]["result"]["conjectures"]
        if item.get("status") == "SURVIVED"
    ]


def _stage_route_provers(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    if entry["machine_form"]["kind"] == "module_target":
        return _stage_route_provers_module_target(entry, upstream, input_hash)
    conjecture_receipt = _load_json(Path(upstream["conjecture_receipt_path"]))
    if conjecture_receipt["status"] != "COMPLETED":
        return _upstream_blocked_receipt(entry["id"], "route_provers", input_hash, "conjecture")
    rows = conjecture_receipt["payload"]["result"]["public_rows"]
    polynomial = _polynomial_closed_form(rows)
    routes: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for conjecture in _survivors(conjecture_receipt):
        route, blocker = _route_one_conjecture(entry["id"], conjecture, polynomial)
        if route is not None:
            routes.append(route)
        if blocker is not None:
            blockers.append(blocker)
    payload = {
        "conjecture_receipt_sha256": conjecture_receipt["content_sha256"],
        "polynomial_closed_form": polynomial,
        "routes": routes,
        "survivors_examined": len(_survivors(conjecture_receipt)),
    }
    return _item_receipt(entry["id"], "route_provers", input_hash, "COMPLETED", payload, blockers)


def _stage_route_provers_module_target(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    """Prove the monotonicity/sign families that survived B3 across the whole ledger."""

    families: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for source in upstream["family_sources"]:
        if source["conjecture_receipt_path"] is None:
            blockers.append(
                _blocker(
                    "upstream_blocked:conjecture",
                    f"problem {source['problem_id']} has no completed conjecture receipt",
                )
            )
            continue
        conjecture_receipt = _load_json(Path(source["conjecture_receipt_path"]))
        rows = conjecture_receipt["payload"]["result"]["public_rows"]
        polynomial = _polynomial_closed_form(rows)
        for conjecture in _survivors(conjecture_receipt):
            if conjecture["kind"] not in ("monotonicity", "sign"):
                continue
            route, blocker = _route_one_conjecture(
                source["problem_id"], conjecture, polynomial
            )
            family = {
                "conjecture_statement": conjecture["statement"],
                "kind": conjecture["kind"],
                "problem_id": source["problem_id"],
                "route": route,
            }
            families.append(family)
            if blocker is not None:
                blockers.append(blocker)
    payload = {
        "families": families,
        "family_count": len(families),
        "proved_locally": sum(
            1
            for family in families
            if family["route"] is not None
            and family["route"]["result"]["decision"] == "PROVED_LOCALLY"
        ),
    }
    return _item_receipt(entry["id"], "route_provers", input_hash, "COMPLETED", payload, blockers)


def _sweep_statement(kind: str, statement: str, sequence_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild an M7 statement from B3's canonical statement rendering."""

    if kind == "divisibility":
        match = _DIVISIBILITY_RE.fullmatch(statement)
        if match is None:
            raise DiscoverySchedulerError(f"unparseable divisibility statement: {statement!r}")
        return {**sequence_spec, "kind": "divisibility", "divisor": int(match.group(1))}
    if kind == "congruence":
        match = _CONGRUENCE_RE.fullmatch(statement)
        if match is None:
            raise DiscoverySchedulerError(f"unparseable congruence statement: {statement!r}")
        return {
            **sequence_spec,
            "kind": "congruence",
            "residue": int(match.group(1)),
            "modulus": int(match.group(2)),
        }
    match = _INDEX_SCALING_RE.fullmatch(statement)
    if match is None:
        raise DiscoverySchedulerError(f"unparseable index-scaling statement: {statement!r}")
    alpha = Fraction(match.group(2))
    beta = Fraction(match.group(3)) if match.group(3) is not None else Fraction(0)
    return {
        **sequence_spec,
        "kind": "index_scaling_relation",
        "scale": int(match.group(1)),
        "alpha": {"numerator": alpha.numerator, "denominator": alpha.denominator},
        "beta": {"numerator": beta.numerator, "denominator": beta.denominator},
    }


def _stage_sweep(
    entry: Mapping[str, Any],
    upstream: Mapping[str, Any],
    input_hash: str,
    *,
    use_gpu: bool,
    sweep_hi: int,
) -> dict[str, Any]:
    machine_form = entry["machine_form"]
    if machine_form["kind"] == "diophantine_family":
        blocker = _blocker(
            "missing_sweeper:diophantine_family",
            f"M7 has no statement kind for the equation {machine_form['equation']!r}; "
            "a diophantine witness sweeper is unbuilt",
        )
        return _item_receipt(
            entry["id"], "sweep", input_hash, "BLOCKED",
            {"machine_form": dict(machine_form)}, [blocker],
        )
    conjecture_receipt = _load_json(Path(upstream["conjecture_receipt_path"]))
    if conjecture_receipt["status"] != "COMPLETED":
        return _upstream_blocked_receipt(entry["id"], "sweep", input_hash, "conjecture")
    generator = _generator_name(machine_form)
    sequence_spec = SWEEP_SEQUENCES.get(generator)
    sweeps: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for conjecture in _survivors(conjecture_receipt):
        if conjecture["kind"] not in SWEEPABLE_KINDS:
            continue
        if sequence_spec is None:
            blockers.append(
                _blocker(
                    f"missing_sweeper:{generator}",
                    f"{conjecture['statement']!r}: sequence {generator!r} is not in the "
                    "M7 sweep registry",
                )
            )
            continue
        statement = _sweep_statement(conjecture["kind"], conjecture["statement"], sequence_spec)
        receipt = run_counterexample_sweep(
            statement, SWEEP_LO, sweep_hi, use_gpu=use_gpu
        )
        sweeps.append(
            {
                "conjecture_kind": conjecture["kind"],
                "conjecture_statement": conjecture["statement"],
                "sweep_decision": receipt["decision"],
                "sweep_receipt": receipt,
            }
        )
    payload = {
        "conjecture_receipt_sha256": conjecture_receipt["content_sha256"],
        "range": {"lo": SWEEP_LO, "hi": sweep_hi},
        "sweeps": sweeps,
        "sweepable_survivors": sum(
            1 for item in _survivors(conjecture_receipt) if item["kind"] in SWEEPABLE_KINDS
        ),
        "use_gpu": use_gpu,
    }
    return _item_receipt(entry["id"], "sweep", input_hash, "COMPLETED", payload, blockers)


def _campaign_receipt_paths(repo_root: Path) -> list[Path]:
    directory = repo_root / "runs" / "gpu-baryonic-screen"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _stage_note_campaigns(
    entry: Mapping[str, Any], input_hash: str, *, repo_root: Path
) -> dict[str, Any]:
    import hashlib

    noted: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    for path in _campaign_receipt_paths(repo_root):
        data = path.read_bytes()
        record: dict[str, Any] = {
            "path": path.relative_to(repo_root).as_posix(),
            "file_sha256": hashlib.sha256(data).hexdigest(),
        }
        try:
            value = json.loads(data.decode("utf-8"))
            body = {key: item for key, item in value.items() if key != "content_sha256"}
            sealed = value.get("content_sha256") == canonical_sha256(body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            sealed = False
            value = {}
        record["seal_verified"] = bool(sealed)
        record["decision"] = value.get("decision") if sealed else None
        record["content_sha256"] = value.get("content_sha256") if sealed else None
        noted.append(record)
    if not noted:
        blockers.append(
            _blocker(
                "missing_campaign_receipts:gpu-baryonic-screen",
                "no GPU campaign receipts were found under runs/gpu-baryonic-screen",
            )
        )
    status = "COMPLETED" if noted else "BLOCKED"
    payload = {
        "campaign_receipts": noted,
        "dataset": entry["machine_form"]["dataset"],
        "target_relation": entry["machine_form"]["target_relation"],
    }
    return _item_receipt(
        entry["id"], "note_gpu_campaign_receipts", input_hash, status, payload, blockers
    )


# ---------------------------------------------------------------------------
# Fan-out lanes: one stage per declared lane in the A5 registry
# ---------------------------------------------------------------------------

#: Declared bounds for the fan-out lanes that take a finite box.  The scheduler never
#: invents a search range: every number here is a declared cap that lands in the lane's
#: own receipt, and the same boxes were used for the standalone campaigns already sealed
#: under ``runs/math/exponent-diophantine``.
FANOUT_BOXES: dict[str, dict[str, int]] = {
    "beal_conjecture": {"base_max": 2000, "exp_max": 9},
    "erdos_straus": {"n_max": 10**7},
    "erdos_straus_sweeper_target": {"n_max": 10**7},
    "fermat_catalan": {"base_max": 2000, "pq_max": 7, "r_max": 12},
}

#: Bounds for the unsolved-progress lane inside a fan-out epoch.  Every key already
#: exists in ``dozen_unsolved_progress_campaign.DEFAULT_BOUNDS``; these are strictly
#: smaller so an epoch stays inside its wall cap.  The lane's receipt records the bounds
#: it actually ran at, so a smaller box is stated, never hidden.
FANOUT_UNSOLVED_BOUNDS: dict[str, dict[str, int]] = {
    "brocard_problem": {"parameter_max": 500},
    "erdos_moser": {"m_max": 100000, "k_max": 10},
    "gilbreath_conjecture": {"rows": 200, "prime_limit": 200000},
    "giuga_conjecture": {"n_max": 200000, "direct_sum_max": 2000},
    "lehmer_totient": {"n_max": 200000},
    "lychrel_196": {"max_iterations": 1000},
    "odd_perfect_number": {"n_max": 1000000, "segment": 1000000},
    "odd_untouchable": {"limit": 50000},
    "recaman_coverage": {"steps": 200000},
    "singmaster_conjecture": {"value_max": 200000},
    # exponent_max 7 gives exactly the seven rows B3 needs for a prefix/holdout split;
    # 6 makes the campaign refuse the conjecture lane outright, which is the module
    # failing closed rather than silently conjecturing from too little data.
    "twin_prime_infinitude": {"exponent_max": 7},
    "ulam_sequence_structure": {"terms": 2000},
}

#: Spectral profile used by the fan-out.  ``engine`` is the same finite grid B3 uses.
FANOUT_SPECTRAL_PROFILE = "engine"

#: Stage -> the upstream stage whose COMPLETED receipt it consumes.
FANOUT_PREREQ: dict[str, str] = {
    "basis_synthesis": "generate_rows",
    "conjecture": "generate_rows",
    "holonomic_guess": "generate_rows",
    "lemma_decomposition": "conjecture",
    "nonlinear_search": "generate_rows",
    "quantified_inequality": "conjecture",
    "spectral_scan": "generate_rows",
    "structural_repair": "generate_rows",
    "sweep": "conjecture",
}


def _rows_from_upstream(upstream: Mapping[str, Any]) -> list[dict[str, int]] | None:
    receipt = _load_json(Path(upstream["rows_receipt_path"]))
    if receipt["status"] != "COMPLETED":
        return None
    return receipt["payload"]["rows"]


def _rows_lane_receipt(
    entry: Mapping[str, Any],
    upstream: Mapping[str, Any],
    input_hash: str,
    stage: str,
    runner: Callable[[list[dict[str, int]]], dict[str, Any]],
) -> dict[str, Any]:
    """Run one row-consuming lane, or record the typed upstream blocker."""

    rows = _rows_from_upstream(upstream)
    if rows is None:
        return _upstream_blocked_receipt(entry["id"], stage, input_hash, "generate_rows")
    result = runner(rows)
    payload = {
        "lane_decision": result.get("decision"),
        "lane_receipt": result,
        "row_count": len(rows),
        "rows_receipt_sha256": _load_json(Path(upstream["rows_receipt_path"]))["content_sha256"],
    }
    return _item_receipt(entry["id"], stage, input_hash, "COMPLETED", payload, [])


def _stage_basis_synthesis(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    from .basis_synthesis import synthesize_basis

    return _rows_lane_receipt(entry, upstream, input_hash, "basis_synthesis", synthesize_basis)


def _stage_nonlinear_search(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    from .nonlinear_coefficient_search import search_nonlinear

    return _rows_lane_receipt(entry, upstream, input_hash, "nonlinear_search", search_nonlinear)


def _stage_structural_repair(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    from .structural_repair import repair_structure

    return _rows_lane_receipt(entry, upstream, input_hash, "structural_repair", repair_structure)


def _stage_holonomic_guess(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str
) -> dict[str, Any]:
    from .holonomic_guesser import guess_receipt

    return _rows_lane_receipt(
        entry,
        upstream,
        input_hash,
        "holonomic_guess",
        lambda rows: guess_receipt(rows, entry["id"]),
    )


def _stage_spectral_scan(
    entry: Mapping[str, Any], upstream: Mapping[str, Any], input_hash: str, *, use_gpu: bool
) -> dict[str, Any]:
    from .spectral_signal_scan import scan_receipt

    return _rows_lane_receipt(
        entry,
        upstream,
        input_hash,
        "spectral_scan",
        lambda rows: scan_receipt(
            rows, entry["id"], profile_name=FANOUT_SPECTRAL_PROFILE, use_gpu=use_gpu
        ),
    )


def _stage_prover_lane(
    entry: Mapping[str, Any],
    upstream: Mapping[str, Any],
    input_hash: str,
    stage: str,
    kinds: tuple[str, ...],
) -> dict[str, Any]:
    """Route only the conjecture kinds this prover lane owns, blockers included."""

    if entry["machine_form"]["kind"] == "module_target":
        sources = upstream["family_sources"]
    else:
        sources = [{"problem_id": entry["id"], "conjecture_receipt_path": None}]
        terminal = _load_json(Path(upstream["conjecture_receipt_path"]))
        if terminal["status"] != "COMPLETED":
            return _upstream_blocked_receipt(entry["id"], stage, input_hash, "conjecture")
        sources[0]["conjecture_receipt_path"] = upstream["conjecture_receipt_path"]

    routes: list[dict[str, Any]] = []
    blockers: list[dict[str, str]] = []
    examined = 0
    for source in sources:
        if source["conjecture_receipt_path"] is None:
            blockers.append(
                _blocker(
                    "upstream_blocked:conjecture",
                    f"problem {source['problem_id']} has no completed conjecture receipt",
                )
            )
            continue
        conjecture_receipt = _load_json(Path(source["conjecture_receipt_path"]))
        polynomial = _polynomial_closed_form(
            conjecture_receipt["payload"]["result"]["public_rows"]
        )
        for conjecture in _survivors(conjecture_receipt):
            if conjecture["kind"] not in kinds:
                continue
            examined += 1
            route, blocker = _route_one_conjecture(
                source["problem_id"], conjecture, polynomial
            )
            if route is not None:
                routes.append({**route, "problem_id": source["problem_id"]})
            if blocker is not None:
                blockers.append(blocker)
    payload = {
        "routed_kinds": list(kinds),
        "routes": routes,
        "survivors_examined": examined,
    }
    return _item_receipt(entry["id"], stage, input_hash, "COMPLETED", payload, blockers)


def _stage_diophantine_sweep(
    entry: Mapping[str, Any], input_hash: str, *, use_gpu: bool, queue_path: str
) -> dict[str, Any]:
    from .exponent_diophantine_sweeper import (
        run_beal_sweep,
        run_erdos_straus_sweep,
        run_fermat_catalan_sweep,
    )

    problem_id = entry["id"]
    box = FANOUT_BOXES[problem_id]
    if problem_id == "beal_conjecture":
        receipt = run_beal_sweep(
            box["base_max"], box["exp_max"], use_gpu=use_gpu, queue_path=queue_path
        )
    elif problem_id == "fermat_catalan":
        receipt = run_fermat_catalan_sweep(
            box["base_max"], box["pq_max"], box["r_max"], use_gpu=use_gpu, queue_path=queue_path
        )
    else:
        receipt = run_erdos_straus_sweep(box["n_max"], use_gpu=use_gpu, queue_path=queue_path)
    payload = {
        "declared_box": box,
        "lane_decision": receipt["decision"],
        "lane_receipt_sha256": receipt["content_sha256"],
        "lane_receipt": receipt,
    }
    return _item_receipt(problem_id, "diophantine_sweep", input_hash, "COMPLETED", payload, [])


def _stage_unsolved_progress(
    entry: Mapping[str, Any], input_hash: str, *, queue: Mapping[str, Any], use_gpu: bool
) -> dict[str, Any]:
    from .dozen_unsolved_progress_campaign import build_receipt as build_progress_receipt

    problem_id = entry["id"]
    receipt = build_progress_receipt(
        queue, problem_id, FANOUT_UNSOLVED_BOUNDS.get(problem_id), use_gpu
    )
    blocker = _blocker(
        receipt["first_blocker"]["code"], receipt["first_blocker"]["detail"]
    )
    payload = {
        "declared_bounds": FANOUT_UNSOLVED_BOUNDS.get(problem_id),
        "lane_receipt_sha256": receipt["content_sha256"],
        "lanes_run": receipt["lanes_run"],
        "lane_receipt": receipt,
    }
    return _item_receipt(
        problem_id, "unsolved_progress", input_hash, "COMPLETED", payload, [blocker]
    )


def _stage_sat_certificate(entry: Mapping[str, Any], input_hash: str) -> dict[str, Any]:
    from .sat_certificate_lane import decide, statement_from_machine_form

    statement = statement_from_machine_form(entry["machine_form"])
    receipt = decide(statement)
    blockers = (
        [_blocker(receipt["decision"], "the SAT lane tripped a declared cap")]
        if str(receipt["decision"]).startswith("CAP_TRIPPED:")
        else []
    )
    payload = {"lane_decision": receipt["decision"], "lane_receipt": receipt}
    return _item_receipt(
        entry["id"], "sat_certificate", input_hash, "COMPLETED", payload, blockers
    )


def _stage_inverse_symbolic(entry: Mapping[str, Any], input_hash: str) -> dict[str, Any]:
    from .inverse_symbolic_engine import run_pslq_lane

    receipt = run_pslq_lane()
    payload = {"lane": "pslq", "lane_receipt_sha256": receipt["content_sha256"]}
    return _item_receipt(
        entry["id"], "inverse_symbolic", input_hash, "COMPLETED", payload, []
    )


def execute_stage(
    stage: str,
    entry: Mapping[str, Any],
    upstream: Mapping[str, Any],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one stage and return its sealed item receipt.  Raises only on defects."""

    input_hash = options["input_hash"]
    if stage == "generate_rows":
        return _stage_generate_rows(entry, input_hash)
    if stage == "conjecture":
        return _stage_conjecture(entry, upstream, input_hash)
    if stage == "route_provers":
        return _stage_route_provers(entry, upstream, input_hash)
    if stage == "sweep":
        return _stage_sweep(
            entry,
            upstream,
            input_hash,
            use_gpu=bool(options["use_gpu"]),
            sweep_hi=int(options["sweep_hi"]),
        )
    if stage == "note_gpu_campaign_receipts":
        return _stage_note_campaigns(entry, input_hash, repo_root=Path(options["repo_root"]))
    if stage == "basis_synthesis":
        return _stage_basis_synthesis(entry, upstream, input_hash)
    if stage == "nonlinear_search":
        return _stage_nonlinear_search(entry, upstream, input_hash)
    if stage == "structural_repair":
        return _stage_structural_repair(entry, upstream, input_hash)
    if stage == "holonomic_guess":
        return _stage_holonomic_guess(entry, upstream, input_hash)
    if stage == "spectral_scan":
        return _stage_spectral_scan(
            entry, upstream, input_hash, use_gpu=bool(options["use_gpu"])
        )
    if stage == "lemma_decomposition":
        return _stage_prover_lane(
            entry, upstream, input_hash, stage, ("closed_form", "linear_recurrence")
        )
    if stage == "quantified_inequality":
        return _stage_prover_lane(
            entry, upstream, input_hash, stage, ("monotonicity", "sign")
        )
    if stage == "diophantine_sweep":
        return _stage_diophantine_sweep(
            entry,
            input_hash,
            use_gpu=bool(options["use_gpu"]),
            queue_path=str(options["queue_path"]),
        )
    if stage == "unsolved_progress":
        return _stage_unsolved_progress(
            entry,
            input_hash,
            queue=load_queue(options["queue_path"]),
            use_gpu=bool(options["use_gpu"]),
        )
    if stage == "sat_certificate":
        return _stage_sat_certificate(entry, input_hash)
    if stage == "inverse_symbolic":
        return _stage_inverse_symbolic(entry, input_hash)
    raise DiscoverySchedulerError(f"unknown stage: {stage}")


def _stage_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker entry point; spawn-safe for the multiprocessing pool."""

    delay_ms = os.environ.get(_ITEM_DELAY_ENV)
    if delay_ms:
        time.sleep(int(delay_ms) / 1000.0)
    return execute_stage(
        payload["stage"], payload["entry"], payload["upstream"], payload["options"]
    )


# ---------------------------------------------------------------------------
# The durable ledger
# ---------------------------------------------------------------------------

_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
  item_id TEXT PRIMARY KEY,
  problem_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('pending','leased','completed','blocked','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT,
  lease_expires_ns INTEGER,
  receipt_path TEXT,
  receipt_sha256 TEXT,
  blocker TEXT,
  error TEXT,
  epoch_completed INTEGER,
  UNIQUE(problem_id, stage, input_hash)
);
CREATE TABLE IF NOT EXISTS blocker_records (
  item_id TEXT NOT NULL,
  blocker_type TEXT NOT NULL,
  detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS epochs (
  epoch_id INTEGER PRIMARY KEY,
  decision TEXT NOT NULL,
  watchdog_receipt_path TEXT NOT NULL,
  watchdog_receipt_sha256 TEXT NOT NULL,
  dashboard_sha256 TEXT NOT NULL,
  summary_receipt_path TEXT NOT NULL,
  summary_sha256 TEXT NOT NULL,
  event_chain_root TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gpu_semaphore (
  slot INTEGER PRIMARY KEY CHECK(slot = 1),
  holder TEXT,
  expires_ns INTEGER
);
CREATE TABLE IF NOT EXISTS control (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  sequence INTEGER PRIMARY KEY,
  created_utc TEXT NOT NULL,
  event_type TEXT NOT NULL,
  item_id TEXT,
  payload_json TEXT NOT NULL,
  prev_sha256 TEXT NOT NULL,
  event_sha256 TEXT NOT NULL
);
"""


class WorkLedger:
    """SQLite work ledger: items, leases, epochs, GPU semaphore, hash-chained events."""

    def __init__(self, path: Path | str, *, queue_sha256: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap = sqlite3.connect(self.path, timeout=30.0)
        try:
            bootstrap.execute("PRAGMA busy_timeout=30000")
            bootstrap.execute("PRAGMA journal_mode=WAL")
            bootstrap.executescript(_LEDGER_SQL)
        finally:
            bootstrap.close()
        with self._txn() as connection:
            stored = self._meta_get(connection, "schema_version")
            if stored is None:
                connection.execute(
                    "INSERT INTO meta VALUES ('schema_version', ?)", (LEDGER_SCHEMA_VERSION,)
                )
            elif stored != LEDGER_SCHEMA_VERSION:
                raise DiscoverySchedulerError("ledger schema_version changed")
            if queue_sha256 is not None:
                bound = self._meta_get(connection, "queue_content_sha256")
                if bound is None:
                    connection.execute(
                        "INSERT INTO meta VALUES ('queue_content_sha256', ?)", (queue_sha256,)
                    )
                elif bound != queue_sha256:
                    raise DiscoverySchedulerError(
                        "refusing to resume: ledger is bound to a different sealed queue"
                    )
            connection.execute("INSERT OR IGNORE INTO gpu_semaphore VALUES (1, NULL, NULL)")

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    @staticmethod
    def _meta_get(connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    # -- hash-chained event log -------------------------------------------

    def _append_event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        item_id: str | None,
        payload: Mapping[str, Any],
    ) -> str:
        row = connection.execute(
            "SELECT sequence, event_sha256 FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if row is None else row["sequence"] + 1
        prev_sha = GENESIS_SHA if row is None else row["event_sha256"]
        created = _utc_now()
        body = {
            "created_utc": created,
            "event_type": event_type,
            "item_id": item_id,
            "payload": dict(payload),
            "prev_sha256": prev_sha,
            "sequence": sequence,
        }
        digest = canonical_sha256(body)
        connection.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (
                sequence,
                created,
                event_type,
                item_id,
                json.dumps(dict(payload), sort_keys=True, separators=(",", ":")),
                prev_sha,
                digest,
            ),
        )
        return digest

    def chain_head(self) -> str:
        with self._txn() as connection:
            row = connection.execute(
                "SELECT event_sha256 FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            return GENESIS_SHA if row is None else row["event_sha256"]

    def validate_event_chain(self) -> str:
        """Recompute every event hash and its predecessor link.  Fail closed on tamper."""

        with self._txn() as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        prev_sha = GENESIS_SHA
        expected_sequence = 1
        for row in rows:
            if row["sequence"] != expected_sequence:
                raise DiscoverySchedulerError(
                    f"event chain gap at sequence {expected_sequence}"
                )
            if row["prev_sha256"] != prev_sha:
                raise DiscoverySchedulerError(
                    f"event chain predecessor changed at sequence {row['sequence']}"
                )
            body = {
                "created_utc": row["created_utc"],
                "event_type": row["event_type"],
                "item_id": row["item_id"],
                "payload": json.loads(row["payload_json"]),
                "prev_sha256": row["prev_sha256"],
                "sequence": row["sequence"],
            }
            if canonical_sha256(body) != row["event_sha256"]:
                raise DiscoverySchedulerError(
                    f"event chain hash changed at sequence {row['sequence']}"
                )
            prev_sha = row["event_sha256"]
            expected_sequence += 1
        return prev_sha

    # -- work items --------------------------------------------------------

    def ensure_item(self, problem_id: str, stage: str, input_hash: str) -> str:
        item_id = f"{problem_id}.{stage}.{input_hash[:12]}"
        with self._txn() as connection:
            existing = connection.execute(
                "SELECT item_id FROM items WHERE problem_id=? AND stage=? AND input_hash=?",
                (problem_id, stage, input_hash),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO items (item_id, problem_id, stage, input_hash, state) "
                    "VALUES (?,?,?,?, 'pending')",
                    (item_id, problem_id, stage, input_hash),
                )
                self._append_event(
                    connection,
                    "item_derived",
                    item_id,
                    {"problem_id": problem_id, "stage": stage, "input_hash": input_hash},
                )
        return item_id

    def item(self, problem_id: str, stage: str, input_hash: str) -> sqlite3.Row | None:
        with self._txn() as connection:
            return connection.execute(
                "SELECT * FROM items WHERE problem_id=? AND stage=? AND input_hash=?",
                (problem_id, stage, input_hash),
            ).fetchone()

    def item_by_id(self, item_id: str) -> sqlite3.Row | None:
        with self._txn() as connection:
            return connection.execute(
                "SELECT * FROM items WHERE item_id=?", (item_id,)
            ).fetchone()

    def terminal_item(self, problem_id: str, stage: str) -> sqlite3.Row | None:
        """The most recently derived terminal (completed/blocked) item for a stage."""

        with self._txn() as connection:
            return connection.execute(
                "SELECT * FROM items WHERE problem_id=? AND stage=? "
                "AND state IN ('completed','blocked') ORDER BY rowid DESC LIMIT 1",
                (problem_id, stage),
            ).fetchone()

    def lease(self, item_id: str, owner: str, lease_seconds: int) -> bool:
        now = time.time_ns()
        expires = now + lease_seconds * 1_000_000_000
        with self._txn() as connection:
            row = connection.execute(
                "SELECT state, attempts FROM items WHERE item_id=?", (item_id,)
            ).fetchone()
            if row is None or row["state"] != "pending":
                return False
            if row["attempts"] >= MAX_ATTEMPTS:
                connection.execute(
                    "UPDATE items SET state='failed', error='attempts_exhausted' "
                    "WHERE item_id=?",
                    (item_id,),
                )
                self._append_event(
                    connection, "item_failed", item_id, {"error": "attempts_exhausted"}
                )
                return False
            connection.execute(
                "UPDATE items SET state='leased', attempts=attempts+1, lease_owner=?, "
                "lease_expires_ns=? WHERE item_id=?",
                (owner, expires, item_id),
            )
            self._append_event(
                connection, "item_leased", item_id, {"owner": owner, "attempt": row["attempts"] + 1}
            )
        return True

    def reclaim_expired(self) -> int:
        now = time.time_ns()
        reclaimed = 0
        with self._txn() as connection:
            rows = connection.execute(
                "SELECT item_id, attempts FROM items WHERE state='leased' "
                "AND lease_expires_ns < ? ORDER BY item_id",
                (now,),
            ).fetchall()
            for row in rows:
                state = "failed" if row["attempts"] >= MAX_ATTEMPTS else "pending"
                connection.execute(
                    "UPDATE items SET state=?, lease_owner=NULL, lease_expires_ns=NULL, "
                    "error=CASE WHEN ?='failed' THEN 'attempts_exhausted' ELSE error END "
                    "WHERE item_id=?",
                    (state, state, row["item_id"]),
                )
                self._append_event(
                    connection, "lease_reclaimed", row["item_id"], {"next_state": state}
                )
                reclaimed += 1
        return reclaimed

    def release(self, item_id: str, reason: str) -> None:
        with self._txn() as connection:
            cursor = connection.execute(
                "UPDATE items SET state='pending', lease_owner=NULL, lease_expires_ns=NULL "
                "WHERE item_id=? AND state='leased'",
                (item_id,),
            )
            if cursor.rowcount:
                self._append_event(connection, "lease_released", item_id, {"reason": reason})

    def record_receipt(
        self, item_id: str, receipt: Mapping[str, Any], receipt_path: Path, epoch_id: int
    ) -> None:
        status = receipt["status"]
        state = "completed" if status == "COMPLETED" else "blocked"
        blockers = receipt["blockers"]
        with self._txn() as connection:
            connection.execute(
                "UPDATE items SET state=?, lease_owner=NULL, lease_expires_ns=NULL, "
                "receipt_path=?, receipt_sha256=?, blocker=?, epoch_completed=? "
                "WHERE item_id=?",
                (
                    state,
                    str(receipt_path),
                    receipt["content_sha256"],
                    blockers[0]["type"] if blockers else None,
                    epoch_id,
                    item_id,
                ),
            )
            for blocker in blockers:
                connection.execute(
                    "INSERT INTO blocker_records VALUES (?,?,?)",
                    (item_id, blocker["type"], blocker["detail"]),
                )
            self._append_event(
                connection,
                "item_completed" if state == "completed" else "item_blocked",
                item_id,
                {
                    "receipt_sha256": receipt["content_sha256"],
                    "blockers": [blocker["type"] for blocker in blockers],
                },
            )

    def record_failure(self, item_id: str, error: str) -> None:
        with self._txn() as connection:
            row = connection.execute(
                "SELECT attempts FROM items WHERE item_id=?", (item_id,)
            ).fetchone()
            state = "failed" if row is not None and row["attempts"] >= MAX_ATTEMPTS else "pending"
            connection.execute(
                "UPDATE items SET state=?, lease_owner=NULL, lease_expires_ns=NULL, error=? "
                "WHERE item_id=?",
                (state, error[:2000], item_id),
            )
            self._append_event(
                connection, "item_failed", item_id, {"error": error[:500], "next_state": state}
            )

    # -- GPU semaphore ------------------------------------------------------

    def gpu_acquire(self, owner: str, lease_seconds: int) -> bool:
        now = time.time_ns()
        with self._txn() as connection:
            row = connection.execute("SELECT * FROM gpu_semaphore WHERE slot=1").fetchone()
            if row["holder"] is not None and row["expires_ns"] > now:
                return False
            connection.execute(
                "UPDATE gpu_semaphore SET holder=?, expires_ns=? WHERE slot=1",
                (owner, now + lease_seconds * 1_000_000_000),
            )
            self._append_event(connection, "gpu_acquired", None, {"owner": owner})
        return True

    def gpu_release(self, owner: str) -> None:
        with self._txn() as connection:
            cursor = connection.execute(
                "UPDATE gpu_semaphore SET holder=NULL, expires_ns=NULL WHERE slot=1 AND holder=?",
                (owner,),
            )
            if cursor.rowcount:
                self._append_event(connection, "gpu_released", None, {"owner": owner})

    # -- control flags ------------------------------------------------------

    def request_stop(self) -> None:
        with self._txn() as connection:
            connection.execute(
                "INSERT INTO control VALUES ('stop_requested','1') "
                "ON CONFLICT(key) DO UPDATE SET value='1'"
            )
            self._append_event(connection, "stop_requested", None, {})

    def clear_stop(self) -> None:
        with self._txn() as connection:
            connection.execute("DELETE FROM control WHERE key='stop_requested'")

    def stop_requested(self) -> bool:
        with self._txn() as connection:
            row = connection.execute(
                "SELECT value FROM control WHERE key='stop_requested'"
            ).fetchone()
            return row is not None and row["value"] == "1"

    # -- epochs and status ---------------------------------------------------

    def next_epoch_id(self) -> int:
        with self._txn() as connection:
            row = connection.execute("SELECT MAX(epoch_id) AS top FROM epochs").fetchone()
            return 1 if row["top"] is None else row["top"] + 1

    def record_epoch(self, epoch: Mapping[str, Any]) -> None:
        with self._txn() as connection:
            connection.execute(
                "INSERT INTO epochs VALUES (?,?,?,?,?,?,?,?)",
                (
                    epoch["epoch_id"],
                    epoch["decision"],
                    epoch["watchdog_receipt_path"],
                    epoch["watchdog_receipt_sha256"],
                    epoch["dashboard_sha256"],
                    epoch["summary_receipt_path"],
                    epoch["summary_sha256"],
                    epoch["event_chain_root"],
                ),
            )
            self._append_event(
                connection,
                "epoch_sealed",
                None,
                {
                    "epoch_id": epoch["epoch_id"],
                    "decision": epoch["decision"],
                    "summary_sha256": epoch["summary_sha256"],
                },
            )

    def counts(self) -> dict[str, Any]:
        with self._txn() as connection:
            states = dict(
                connection.execute(
                    "SELECT state, COUNT(*) FROM items GROUP BY state ORDER BY state"
                ).fetchall()
            )
            epochs = connection.execute("SELECT COUNT(*) FROM epochs").fetchone()[0]
            blockers = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT blocker_type FROM blocker_records ORDER BY blocker_type"
                ).fetchall()
            ]
            reclaimed = connection.execute(
                "SELECT COUNT(*) FROM events WHERE event_type='lease_reclaimed'"
            ).fetchone()[0]
            events = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "blocker_types": blockers,
            "epochs": epochs,
            "events": events,
            "item_states": {key: states.get(key, 0) for key in
                            ("pending", "leased", "completed", "blocked", "failed")},
            "leases_reclaimed_total": reclaimed,
        }

    def event_count_of(self, event_type: str) -> int:
        with self._txn() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM events WHERE event_type=?", (event_type,)
            ).fetchone()[0]

    def record_soak(self, soak_sha256: str) -> None:
        with self._txn() as connection:
            self._append_event(connection, "soak_sealed", None, {"soak_sha256": soak_sha256})


# ---------------------------------------------------------------------------
# Work derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    problem_id: str
    stage: str
    input_hash: str
    entry: dict[str, Any]
    upstream: dict[str, Any]
    gpu: bool


@dataclass
class SchedulerConfig:
    queue_path: Path
    ledger_path: Path
    output_dir: Path
    repo_root: Path
    epochs: int = 1
    max_items: int = 64
    use_gpu: bool = False
    caps: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    workers: int = 0
    lease_seconds: int = 300
    soak_minutes: int | None = None
    sweep_hi_cpu: int = SWEEP_HI_CPU_DEFAULT
    sweep_hi_gpu: int = SWEEP_HI_GPU_DEFAULT
    probes: dict[str, Any] | None = None
    #: When true, an epoch fans out over (problem x applicable lane) from the A5
    #: registry instead of the fixed :data:`STAGES_BY_KIND` list.  Default false so the
    #: legacy stage graph, and every receipt already sealed against it, is unchanged.
    fanout: bool = False
    #: When true, the end of every epoch rebuilds the A6 capability gap ledger and binds
    #: its top open gaps into the epoch summary.
    gap_ledger: bool = True

    def resolved_workers(self) -> int:
        if self.workers > 0:
            return self.workers
        return max(1, min(8, (os.cpu_count() or 2) - 2))


def gpu_available() -> bool:
    try:
        import cupy

        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - any import/driver failure means no GPU
        return False


def _input_hash(stage: str, binding: Mapping[str, Any]) -> str:
    return canonical_sha256({"stage": stage, "binding": dict(binding)})


#: Fan-out stages that consume the ledger-wide conjecture corpus rather than one problem.
_MODULE_TARGET_STAGES = ("route_provers", "lemma_decomposition", "quantified_inequality")

#: Fan-out stages that take the queue entry alone and need no upstream artifact.
_STANDALONE_FANOUT_STAGES = (
    "diophantine_sweep",
    "inverse_symbolic",
    "sat_certificate",
    "unsolved_progress",
)


def _stage_prereq(kind: str, stage: str) -> str | None:
    if stage in _MODULE_TARGET_STAGES and kind == "module_target":
        return None
    if stage == "sweep" and kind == "diophantine_family":
        return None
    if stage == "route_provers":
        return "conjecture"
    return FANOUT_PREREQ.get(stage)


def derive_work(ledger: WorkLedger, queue: Mapping[str, Any], config: SchedulerConfig) -> list[WorkItem]:
    """Derive every currently-runnable work item, deterministically ordered."""

    effective_gpu = config.use_gpu and gpu_available()
    sweep_hi = config.sweep_hi_gpu if effective_gpu else config.sweep_hi_cpu
    entries = {entry["id"]: entry for entry in queue["entries"]}
    conjecture_problems = sorted(
        entry_id
        for entry_id, entry in entries.items()
        if "conjecture" in STAGES_BY_KIND[entry["machine_form"]["kind"]]
    )
    items: list[WorkItem] = []
    for entry_id in sorted(entries):
        entry = entries[entry_id]
        kind = entry["machine_form"]["kind"]
        stages = (
            fanout_stages(entry) if config.fanout else STAGES_BY_KIND[kind]
        )
        for stage in stages:
            derived = _derive_stage(
                ledger, entry, stage, conjecture_problems,
                effective_gpu=effective_gpu, sweep_hi=sweep_hi,
                repo_root=config.repo_root,
            )
            if derived is not None:
                items.append(derived)
    return items


def fanout_stages(entry: Mapping[str, Any]) -> tuple[str, ...]:
    """The stages of every A5 lane applicable to this entry, in declaration order."""

    return tuple(
        lane(decision.lane_id).stage
        for decision in lane_decisions(entry)
        if decision.applicable
    )


def fanout_skips(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The recorded skips for this entry: every lane not attempted, with a typed reason."""

    return [
        decision.as_record() for decision in lane_decisions(entry) if not decision.applicable
    ]


def _derive_stage(
    ledger: WorkLedger,
    entry: Mapping[str, Any],
    stage: str,
    conjecture_problems: Sequence[str],
    *,
    effective_gpu: bool,
    sweep_hi: int,
    repo_root: Path,
) -> WorkItem | None:
    kind = entry["machine_form"]["kind"]
    problem_id = entry["id"]
    upstream: dict[str, Any] = {}
    spec = lane_for_stage(stage)
    gpu = False

    if kind == "module_target" and stage in _MODULE_TARGET_STAGES:
        sources = []
        binding_sources = []
        for other_id in conjecture_problems:
            terminal = ledger.terminal_item(other_id, "conjecture")
            if terminal is None:
                return None  # not yet runnable: some conjecture stage is unfinished
            path = terminal["receipt_path"] if terminal["state"] == "completed" else None
            sources.append({"problem_id": other_id, "conjecture_receipt_path": path})
            binding_sources.append(
                {"problem_id": other_id, "receipt_sha256": terminal["receipt_sha256"]}
            )
        upstream["family_sources"] = sources
        input_hash = _input_hash(stage, {"sources": binding_sources})
    elif stage == "generate_rows":
        input_hash = _input_hash(stage, {"machine_form": entry["machine_form"]})
    elif stage == "note_gpu_campaign_receipts":
        import hashlib

        shas = [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in _campaign_receipt_paths(repo_root)
        ]
        input_hash = _input_hash(
            stage, {"machine_form": entry["machine_form"], "receipt_shas": shas}
        )
    elif stage == "sweep" and kind == "diophantine_family":
        input_hash = _input_hash(stage, {"machine_form": entry["machine_form"]})
    elif stage in _STANDALONE_FANOUT_STAGES:
        binding = {"machine_form": entry["machine_form"], "problem_id": problem_id}
        if stage == "diophantine_sweep":
            binding["box"] = FANOUT_BOXES[problem_id]
        if stage == "unsolved_progress":
            binding["bounds"] = FANOUT_UNSOLVED_BOUNDS.get(problem_id)
        if spec is not None and spec.resource == "gpu":
            binding["use_gpu"] = effective_gpu
            gpu = effective_gpu
        input_hash = _input_hash(stage, binding)
    else:
        prereq = _stage_prereq(kind, stage)
        if prereq is None:
            raise DiscoverySchedulerError(f"stage {stage!r} declares no prerequisite")
        terminal = ledger.terminal_item(problem_id, prereq)
        if terminal is None:
            return None
        key = "rows_receipt_path" if prereq == "generate_rows" else "conjecture_receipt_path"
        upstream[key] = terminal["receipt_path"]
        binding = {"upstream_receipt_sha256": terminal["receipt_sha256"]}
        if stage == "sweep":
            binding.update({"use_gpu": effective_gpu, "lo": SWEEP_LO, "hi": sweep_hi})
            gpu = effective_gpu
        elif spec is not None and spec.resource == "gpu":
            binding["use_gpu"] = effective_gpu
            gpu = effective_gpu
        input_hash = _input_hash(stage, binding)

    existing = ledger.item(problem_id, stage, input_hash)
    if existing is not None and existing["state"] in ("completed", "blocked", "failed", "leased"):
        return None
    item_id = ledger.ensure_item(problem_id, stage, input_hash)
    return WorkItem(
        item_id=item_id,
        problem_id=problem_id,
        stage=stage,
        input_hash=input_hash,
        entry=dict(entry),
        upstream=upstream,
        gpu=gpu,
    )


# ---------------------------------------------------------------------------
# Epoch execution
# ---------------------------------------------------------------------------


def _worker_payload(item: WorkItem, config: SchedulerConfig, sweep_hi: int) -> dict[str, Any]:
    return {
        "stage": item.stage,
        "entry": item.entry,
        "upstream": item.upstream,
        "options": {
            "input_hash": item.input_hash,
            "use_gpu": item.gpu,
            "sweep_hi": sweep_hi,
            "repo_root": str(config.repo_root),
            "queue_path": str(config.queue_path),
        },
    }


def _record_outcome(
    ledger: WorkLedger,
    config: SchedulerConfig,
    epoch_id: int,
    item: WorkItem,
    receipt: Mapping[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> None:
    directory = config.output_dir / "items" / item.problem_id
    path = _write_receipt(directory, f"{item.stage}-{item.input_hash[:12]}", receipt)
    ledger.record_receipt(item.item_id, receipt, path, epoch_id)
    outcomes.setdefault(item.problem_id, {})[item.stage] = {
        "status": receipt["status"],
        "receipt_sha256": receipt["content_sha256"],
        "blockers": [blocker["type"] for blocker in receipt["blockers"]],
    }


def _fanout_record(queue: Mapping[str, Any], config: SchedulerConfig) -> dict[str, Any]:
    """What the fan-out planned this epoch, skips included.  A skip is never silent."""

    if not config.fanout:
        return {
            "enabled": False,
            "registry_content_sha256": REGISTRY_CONTENT_SHA256,
            "stage_source": "STAGES_BY_KIND",
        }
    planned: dict[str, list[str]] = {}
    skipped: dict[str, list[dict[str, Any]]] = {}
    for entry in queue["entries"]:
        planned[entry["id"]] = list(fanout_stages(entry))
        skipped[entry["id"]] = fanout_skips(entry)
    return {
        "enabled": True,
        "lanes_planned_per_problem": planned,
        "lanes_skipped_per_problem": skipped,
        "registry_content_sha256": REGISTRY_CONTENT_SHA256,
        "stage_source": "lane_registry.applicable_lanes",
        "totals": {
            "attempts_planned": sum(len(item) for item in planned.values()),
            "problems": len(planned),
            "skips_recorded": sum(len(item) for item in skipped.values()),
        },
    }


def _rebuild_gap_ledger(config: SchedulerConfig) -> dict[str, Any]:
    """Regenerate the A6 gap ledger from the whole corpus and return its top open gaps.

    An epoch that produced new blockers must not leave the build queue stale, so the
    ledger is rebuilt from the real corpus at the end of every epoch and the summary
    binds its seal.  A failure here is reported in the summary, never raised into the
    epoch: the scheduler's own honesty machinery cannot be allowed to end a run.
    """

    if not config.gap_ledger:
        return {"built": False, "reason": "gap_ledger disabled by configuration"}
    try:
        gap_ledger = build_gap_ledger(config.repo_root)
        receipt_path = config.repo_root / "runs" / "discovery-engine" / "capability-gap-ledger.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(canonical_json_bytes(gap_ledger) + b"\n")
        markdown_path = config.repo_root / "docs" / "CAPABILITY_GAPS.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_bytes(render_gap_markdown(gap_ledger).encode("utf-8"))
    except Exception as error:  # noqa: BLE001 - reported, never fatal to the epoch
        return {"built": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "built": True,
        "counts": gap_ledger["counts"],
        "ledger_content_sha256": gap_ledger["content_sha256"],
        "top_open_gaps": top_open_gaps(gap_ledger, 5),
    }


def run_discovery_epoch(
    ledger: WorkLedger, queue: Mapping[str, Any], config: SchedulerConfig
) -> dict[str, Any]:
    """Run one epoch: reclaim, derive, lease, execute under the watchdog, seal."""

    epoch_id = ledger.next_epoch_id()
    reclaimed = ledger.reclaim_expired()
    derived = derive_work(ledger, queue, config)
    derived = derived[: config.max_items]
    leased: list[WorkItem] = []
    owner = f"scheduler-{os.getpid()}"
    for item in derived:
        if ledger.lease(item.item_id, owner, config.lease_seconds):
            leased.append(item)
    print(f"EPOCH {epoch_id} leased={len(leased)} reclaimed={reclaimed}", flush=True)

    effective_gpu = config.use_gpu and gpu_available()
    sweep_hi = config.sweep_hi_gpu if effective_gpu else config.sweep_hi_cpu
    gpu_items = [item for item in leased if item.gpu]
    cpu_items = [item for item in leased if not item.gpu]
    outcomes: dict[str, dict[str, Any]] = {}
    executed: set[str] = set()
    failed_items = 0
    stopped = False

    def epoch_callable(check: Callable[[], None]) -> None:
        nonlocal failed_items, stopped
        for item in gpu_items:
            check()
            if ledger.stop_requested():
                stopped = True
                return
            while not ledger.gpu_acquire(owner, GPU_LEASE_SECONDS):
                time.sleep(0.2)
                check()
            try:
                receipt = _stage_worker(_worker_payload(item, config, sweep_hi))
            except Exception as error:  # noqa: BLE001 - worker boundary, receipted as failure
                ledger.record_failure(item.item_id, f"{type(error).__name__}: {error}")
                failed_items += 1
            else:
                _record_outcome(ledger, config, epoch_id, item, receipt, outcomes)
            finally:
                ledger.gpu_release(owner)
            executed.add(item.item_id)
        if not cpu_items:
            return
        if len(cpu_items) == 1:
            item = cpu_items[0]
            check()
            if ledger.stop_requested():
                stopped = True
                return
            try:
                receipt = _stage_worker(_worker_payload(item, config, sweep_hi))
            except Exception as error:  # noqa: BLE001 - worker boundary, receipted as failure
                ledger.record_failure(item.item_id, f"{type(error).__name__}: {error}")
                failed_items += 1
            else:
                _record_outcome(ledger, config, epoch_id, item, receipt, outcomes)
            executed.add(item.item_id)
            return
        pool = ProcessPoolExecutor(max_workers=min(config.resolved_workers(), len(cpu_items)))
        graceful = False
        try:
            futures = {
                pool.submit(_stage_worker, _worker_payload(item, config, sweep_hi)): item
                for item in cpu_items
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    item = futures[future]
                    executed.add(item.item_id)
                    try:
                        receipt = future.result()
                    except Exception as error:  # noqa: BLE001 - worker boundary
                        ledger.record_failure(
                            item.item_id, f"{type(error).__name__}: {error}"
                        )
                        failed_items += 1
                    else:
                        _record_outcome(ledger, config, epoch_id, item, receipt, outcomes)
                check()
                if ledger.stop_requested():
                    stopped = True
                    graceful = True
                    return
            graceful = True
        finally:
            pool.shutdown(wait=graceful, cancel_futures=True)

    probes = dict(config.probes or {})
    if "max_disk_write_bytes" in config.caps and "disk_directory" not in probes:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        probes["disk_directory"] = str(config.output_dir)
    if "max_llm_dollars_hundredths" in config.caps and "llm_ledger" not in probes:
        llm_ledger = config.output_dir / "llm-spend-ledger.json"
        if not llm_ledger.exists():
            llm_ledger.parent.mkdir(parents=True, exist_ok=True)
            llm_ledger.write_text(
                json.dumps(
                    {"schema_version": LLM_LEDGER_SCHEMA, "spent_dollars_hundredths": 0}
                ),
                encoding="utf-8",
            )
        probes["llm_ledger"] = str(llm_ledger)
    watchdog_receipt = run_epoch(epoch_callable, caps=config.caps, probes=probes or None)

    # Anything leased but not executed (cap trip or stop) goes straight back to pending.
    for item in leased:
        if item.item_id not in executed:
            ledger.release(item.item_id, watchdog_receipt["decision"])

    epoch_dir = config.output_dir / "epochs"
    watchdog_path = _write_receipt(
        epoch_dir, f"epoch-{epoch_id:06d}-watchdog", watchdog_receipt
    )

    dashboard = build_dashboard(config.repo_root)
    dashboard_dir = config.repo_root / "runs" / "discovery-dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "status-v1.json").write_bytes(canonical_json_bytes(dashboard) + b"\n")
    (dashboard_dir / "status-v1.html").write_bytes(render_html(dashboard).encode("utf-8"))

    completed = sum(
        1 for problem in outcomes.values() for out in problem.values()
        if out["status"] == "COMPLETED"
    )
    blocked = sum(
        1 for problem in outcomes.values() for out in problem.values()
        if out["status"] == "BLOCKED"
    )
    capability_gaps = _rebuild_gap_ledger(config)
    summary_body = {
        "capability_gaps": capability_gaps,
        "claims": {
            "corpus_absence_establishes_novelty": False,
            "scalar_truth_or_probability_score": False,
            "unattended_operation": _INTERACTIVE_PROMPTS == 0,
        },
        "dashboard_sha256": dashboard["content_sha256"],
        "decision": watchdog_receipt["decision"],
        "epoch_id": epoch_id,
        "event_chain_root": ledger.chain_head(),
        "fanout": _fanout_record(queue, config),
        "items": {
            "attempted": len(leased),
            "blocked": blocked,
            "completed": completed,
            "failed": failed_items,
        },
        "leases_reclaimed_before_epoch": reclaimed,
        "per_problem_outcomes": outcomes,
        "schema_version": EPOCH_RECEIPT_SCHEMA,
        "scope": EPOCH_SCOPE,
        "stopped_by_request": stopped,
        "watchdog_receipt_sha256": watchdog_receipt["content_sha256"],
    }
    summary = _seal(summary_body)
    summary_path = _write_receipt(epoch_dir, f"epoch-{epoch_id:06d}-summary", summary)
    ledger.record_epoch(
        {
            "epoch_id": epoch_id,
            "decision": watchdog_receipt["decision"],
            "watchdog_receipt_path": str(watchdog_path),
            "watchdog_receipt_sha256": watchdog_receipt["content_sha256"],
            "dashboard_sha256": dashboard["content_sha256"],
            "summary_receipt_path": str(summary_path),
            "summary_sha256": summary["content_sha256"],
            "event_chain_root": summary["event_chain_root"],
        }
    )
    print(
        f"EPOCH {epoch_id} {watchdog_receipt['decision']} attempted={len(leased)} "
        f"completed={completed} blocked={blocked} failed={failed_items}",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# The scheduler loop and the soak harness
# ---------------------------------------------------------------------------


def run_scheduler(config: SchedulerConfig) -> dict[str, Any]:
    """Run epochs (or a wall-bounded soak) against the sealed queue.  Resumable."""

    queue = load_queue(config.queue_path)
    ledger = WorkLedger(config.ledger_path, queue_sha256=queue["content_sha256"])
    ledger.clear_stop()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    reclaimed_before = ledger.event_count_of("lease_reclaimed")
    summaries: list[dict[str, Any]] = []

    if config.soak_minutes is not None:
        deadline = time.monotonic() + config.soak_minutes * 60
        idle_streak = 0
        while time.monotonic() < deadline and not ledger.stop_requested():
            summary = run_discovery_epoch(ledger, queue, config)
            summaries.append(summary)
            if summary["items"]["attempted"] == 0:
                idle_streak += 1
                backoff = min(60, 2**idle_streak)
                if time.monotonic() + backoff >= deadline:
                    break
                time.sleep(backoff)
            else:
                idle_streak = 0
    else:
        for _ in range(config.epochs):
            if ledger.stop_requested():
                break
            summary = run_discovery_epoch(ledger, queue, config)
            summaries.append(summary)
            if summary["items"]["attempted"] == 0:
                break

    result: dict[str, Any] = {
        "epochs_run": len(summaries),
        "items_blocked": sum(item["items"]["blocked"] for item in summaries),
        "items_completed": sum(item["items"]["completed"] for item in summaries),
        "items_failed": sum(item["items"]["failed"] for item in summaries),
        "summaries": summaries,
    }
    if config.soak_minutes is not None:
        soak_body = {
            "claims": {
                "corpus_absence_establishes_novelty": False,
                "scalar_truth_or_probability_score": False,
                "zero_manual_steps": _INTERACTIVE_PROMPTS == 0,
            },
            "crash_recoveries": ledger.event_count_of("lease_reclaimed") - reclaimed_before,
            "declared_soak_minutes": config.soak_minutes,
            "epochs_run": len(summaries),
            "event_chain_root": ledger.chain_head(),
            "items_blocked": result["items_blocked"],
            "items_completed": result["items_completed"],
            "items_failed": result["items_failed"],
            "schema_version": SOAK_RECEIPT_SCHEMA,
            "scope": SOAK_SCOPE,
        }
        soak = _seal(soak_body)
        soak_index = 1 + ledger.event_count_of("soak_sealed")
        soak_path = _write_receipt(config.output_dir, f"soak-{soak_index:04d}", soak)
        ledger.record_soak(soak["content_sha256"])
        result["soak_receipt"] = soak
        result["soak_receipt_path"] = str(soak_path)
        print(
            f"SOAK sealed epochs={len(summaries)} completed={result['items_completed']} "
            f"crash_recoveries={soak['crash_recoveries']} -> {soak_path}",
            flush=True,
        )
    ledger.validate_event_chain()
    return result


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def _validate_seal(value: Any, schema: str, label: str) -> None:
    if not isinstance(value, Mapping):
        raise DiscoverySchedulerError(f"{label} must be a mapping")
    if value.get("schema_version") != schema:
        raise DiscoverySchedulerError(f"{label} schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise DiscoverySchedulerError(f"{label} seal changed")


def validate_item_receipt(value: Any) -> None:
    _validate_seal(value, ITEM_RECEIPT_SCHEMA, "item receipt")
    if value["status"] not in ("COMPLETED", "BLOCKED"):
        raise DiscoverySchedulerError("item receipt status changed")
    if value["claims"] != ITEM_CLAIMS:
        raise DiscoverySchedulerError("item receipt claims changed")
    if value["status"] == "BLOCKED" and not value["blockers"]:
        raise DiscoverySchedulerError("a BLOCKED item receipt must carry a typed blocker")
    for blocker in value["blockers"]:
        if set(blocker) != {"type", "detail"} or ":" not in blocker["type"]:
            raise DiscoverySchedulerError("blocker records must be typed as kind:subject")


def validate_epoch_receipt(value: Any) -> None:
    _validate_seal(value, EPOCH_RECEIPT_SCHEMA, "epoch receipt")
    items = value["items"]
    if set(items) != {"attempted", "blocked", "completed", "failed"}:
        raise DiscoverySchedulerError("epoch receipt items block changed")
    for key in items:
        if not isinstance(items[key], int) or isinstance(items[key], bool) or items[key] < 0:
            raise DiscoverySchedulerError(f"epoch receipt items.{key} must be a nonnegative int")
    claims = value["claims"]
    if set(claims) != {
        "corpus_absence_establishes_novelty",
        "scalar_truth_or_probability_score",
        "unattended_operation",
    }:
        raise DiscoverySchedulerError("epoch receipt claims changed")
    if claims["corpus_absence_establishes_novelty"] or claims["scalar_truth_or_probability_score"]:
        raise DiscoverySchedulerError("epoch receipt asserts a forbidden claim")


def validate_soak_receipt(value: Any) -> None:
    _validate_seal(value, SOAK_RECEIPT_SCHEMA, "soak receipt")
    for key in ("epochs_run", "items_completed", "crash_recoveries"):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 0:
            raise DiscoverySchedulerError(f"soak receipt {key} must be a nonnegative int")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous discovery scheduler (A1).")
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="run epochs (resume is implicit via the ledger)")
    start.add_argument("--queue", required=True, help="sealed A2 problem queue JSON")
    start.add_argument("--ledger", required=True, help="SQLite work ledger path")
    start.add_argument("--output", required=True, help="directory for sealed receipts")
    start.add_argument("--repo-root", default=".", help="repository root for dashboard sources")
    start.add_argument("--epochs", type=int, default=1)
    start.add_argument("--max-items", type=int, default=64)
    start.add_argument("--gpu", action="store_true", help="allow the GPU sweep lane")
    start.add_argument("--caps", default=None, help="watchdog caps as JSON")
    start.add_argument("--workers", type=int, default=0)
    start.add_argument("--lease-seconds", type=int, default=300)
    start.add_argument("--soak-minutes", type=int, default=None)
    start.add_argument("--sweep-hi-cpu", type=int, default=SWEEP_HI_CPU_DEFAULT)
    start.add_argument("--sweep-hi-gpu", type=int, default=SWEEP_HI_GPU_DEFAULT)
    start.add_argument(
        "--fanout",
        action="store_true",
        help="fan out over (problem x applicable lane) from the A5 registry instead of "
        "the fixed per-kind stage list",
    )
    start.add_argument(
        "--no-gap-ledger",
        action="store_true",
        help="skip the end-of-epoch capability gap ledger rebuild",
    )

    status = commands.add_parser("status", help="print ledger counts and chain state")
    status.add_argument("--ledger", required=True)

    stop = commands.add_parser("stop", help="request a graceful stop between items")
    stop.add_argument("--ledger", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "status":
        ledger = WorkLedger(args.ledger)
        report = ledger.counts()
        report["event_chain_root"] = ledger.validate_event_chain()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "stop":
        WorkLedger(args.ledger).request_stop()
        print("STOP requested")
        return 0
    caps = dict(DEFAULT_CAPS)
    if args.caps:
        declared = json.loads(args.caps)
        if not isinstance(declared, dict):
            raise DiscoverySchedulerError("--caps must be a JSON object")
        caps = declared
    config = SchedulerConfig(
        queue_path=Path(args.queue),
        ledger_path=Path(args.ledger),
        output_dir=Path(args.output),
        repo_root=Path(args.repo_root),
        epochs=args.epochs,
        max_items=args.max_items,
        use_gpu=args.gpu,
        caps=caps,
        workers=args.workers,
        lease_seconds=args.lease_seconds,
        soak_minutes=args.soak_minutes,
        sweep_hi_cpu=args.sweep_hi_cpu,
        sweep_hi_gpu=args.sweep_hi_gpu,
        fanout=args.fanout,
        gap_ledger=not args.no_gap_ledger,
    )
    result = run_scheduler(config)
    print(
        f"DONE epochs={result['epochs_run']} completed={result['items_completed']} "
        f"blocked={result['items_blocked']} failed={result['items_failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
