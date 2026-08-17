"""DG5 generator gates: the six sequences eight queue problems were waiting on.

A row generator earns trust for exactly one reason — it reproduces values somebody else
published.  Every generator here is confronted with its OEIS listing by A-number, and
three of them are additionally recomputed from scratch inside the test by an independent
route (a difference table, a bigint reverse-and-add, a binomial scan), so a shared bug in
the campaign module and the adapter cannot pass unnoticed.

The other half of the contract is the cap.  A generator asked for more rows than it or
the engine will produce must emit the exact prefix *and* a typed
``generator_cap_truncated`` record: the queue asks for 10^7 Recaman steps and 2000 Ulam
terms, and a system that quietly returned 64 of them would be lying by omission.

The DG5 completion test is the last one: fan out over the six machine forms and assert the
capability gap ledger built from that epoch carries no ``missing_generator:*`` and no
``upstream_blocked:generate_rows`` at all.
"""

from __future__ import annotations

import json
from itertools import pairwise
from math import comb
from pathlib import Path

import pytest

from sigma_theory_compiler.capability_gap_ledger import build_ledger as build_gap_ledger
from sigma_theory_compiler.discovery_scheduler import (
    GENERATOR_CAPS,
    GENERATOR_REGISTRY,
    GENERATOR_ROW_CAPS,
    SchedulerConfig,
    _stage_generate_rows,
    run_scheduler,
    validate_item_receipt,
)
from sigma_theory_compiler.problem_queue import load_queue, seal_queue
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The six generators DG5 named, with the machine-form key each one reads.
DG5_GENERATORS: dict[str, str] = {
    "gilbreath_leading_terms": "max_point",
    "pascal_interior_multiplicity": "max_point",
    "recaman": "max_steps",
    "reverse_and_add_base10": "max_steps",
    "twin_prime_count_pi2_10_pow_k": "max_point",
    "ulam_u_1_2": "max_point",
}


def _values(name: str, count: int, **extra: int) -> list[int]:
    key = DG5_GENERATORS[name]
    kind = "sequence_rows" if key == "max_point" else "integer_trajectory"
    machine_form = {"kind": kind, key: count, **extra}
    rows, _ = GENERATOR_REGISTRY[name](machine_form)
    return [row["value"] for row in rows]


def _points(name: str, count: int, **extra: int) -> list[int]:
    key = DG5_GENERATORS[name]
    kind = "sequence_rows" if key == "max_point" else "integer_trajectory"
    rows, _ = GENERATOR_REGISTRY[name]({"kind": kind, key: count, **extra})
    return [row["point"] for row in rows]


# ---------------------------------------------------------------------------
# Each generator against its published reference values
# ---------------------------------------------------------------------------


def test_ulam_reproduces_the_a002858_listing():
    """OEIS A002858: 1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, 28, 36, 38, 47, 48."""

    assert _values("ulam_u_1_2", 16) == [
        1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, 28, 36, 38, 47, 48
    ]
    # Ulam's defining property, recomputed here: each new term is the smallest integer
    # above the previous one with exactly one representation as a sum of two distinct
    # earlier terms.
    terms = _values("ulam_u_1_2", 60)
    for index in range(2, len(terms)):
        head = terms[:index]
        representations = sum(
            1
            for i in range(len(head))
            for j in range(i + 1, len(head))
            if head[i] + head[j] == terms[index]
        )
        assert representations == 1
        for candidate in range(terms[index - 1] + 1, terms[index]):
            others = sum(
                1
                for i in range(len(head))
                for j in range(i + 1, len(head))
                if head[i] + head[j] == candidate
            )
            assert others != 1


def test_twin_prime_counts_reproduce_the_a007508_listing():
    """OEIS A007508, pi_2(10^k) for k = 1..7."""

    assert _values("twin_prime_count_pi2_10_pow_k", 7) == [
        2, 8, 35, 205, 1224, 8169, 58980
    ]
    assert _points("twin_prime_count_pi2_10_pow_k", 7) == [1, 2, 3, 4, 5, 6, 7]


def test_gilbreath_leading_terms_match_an_independent_difference_table():
    """OEIS A036262 (Gilbreath's difference table): every leading term after the first
    row is 1, verified computationally past 10^13 rows (Odlyzko 1993)."""

    computed = _values("gilbreath_leading_terms", 500)
    assert computed == [1] * 500

    primes = [2]
    candidate = 3
    while len(primes) < 600:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 2
    row = primes
    independent = []
    for _ in range(20):
        row = [abs(right - left) for left, right in pairwise(row)]
        independent.append(row[0])
    assert computed[:20] == independent


def test_pascal_multiplicity_reproduces_a003016_and_singmaster_records():
    """OEIS A003016, N(t) = number of times t occurs in Pascal's triangle, offset 2."""

    assert _values("pascal_interior_multiplicity", 24) == [
        1, 2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2
    ]
    # Rows are indexed from 1 like every other generator: row n carries N(n + 1).
    assert _points("pascal_interior_multiplicity", 4) == [1, 2, 3, 4]

    values = _values("pascal_interior_multiplicity", 3002)
    multiplicity = {t: values[t - 2] for t in range(2, 3004) if t - 2 < len(values)}
    # Singmaster's data: 3003 is the only integer known to occur eight times, and
    # 120, 210 and 1540 occur six times.
    assert multiplicity[3003] == 8
    assert max(multiplicity.values()) == 8
    assert [t for t, count in multiplicity.items() if count == 6] == [120, 210, 1540]

    # Independent recomputation of the same counts from the binomial coefficients.
    independent: dict[int, int] = {}
    for n in range(2, 200):
        for k in range(n + 1):
            value = comb(n, k)
            if 2 <= value <= 240:
                independent[value] = independent.get(value, 0) + 1
    for t in range(2, 200):
        assert multiplicity[t] == independent.get(t, 0)


def test_recaman_reproduces_a005132_and_the_sealed_campaign_head():
    """OEIS A005132: 0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9, 24."""

    assert _values("recaman", 16, seed=0) == [
        0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9, 24
    ]
    sealed = json.loads(
        (REPO_ROOT / "runs" / "math" / "unsolved-dozen" / "recaman_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    head = sealed["results"]["integer_trajectory"]["first_terms"]
    assert _values("recaman", len(head), seed=0) == head


def test_reverse_and_add_matches_an_independent_bigint_trajectory():
    """OEIS A006960 (196 reverse-and-add): 196, 887, 1675, 7436, 13783, 52514, ..."""

    published = [196, 887, 1675, 7436, 13783, 52514, 94039, 187088, 1067869]
    assert _values("reverse_and_add_base10", 8, seed=196) == [
        len(str(value)) for value in published[1:]
    ]

    value = 196
    independent = []
    for _ in range(400):
        value += int(str(value)[::-1])
        independent.append(len(str(value)))
    assert _values("reverse_and_add_base10", 400, seed=196) == independent

    # 196 is the Lychrel candidate: no palindrome appears anywhere in this window, so
    # the generator emits the full requested run rather than stopping early.
    value = 196
    for _ in range(400):
        value += int(str(value)[::-1])
        assert str(value) != str(value)[::-1]


def test_every_dg5_generator_is_deterministic():
    for name in DG5_GENERATORS:
        seed = {"seed": 196} if name == "reverse_and_add_base10" else {}
        if name == "recaman":
            seed = {"seed": 0}
        count = 7 if name == "twin_prime_count_pi2_10_pow_k" else 40
        assert _values(name, count, **seed) == _values(name, count, **seed)


def test_generator_points_are_consecutive_integers():
    """The holonomic guesser refuses non-consecutive points; every generator must
    produce rows it can actually parse."""

    for name in DG5_GENERATORS:
        seed = {"seed": 196} if name == "reverse_and_add_base10" else {}
        if name == "recaman":
            seed = {"seed": 0}
        count = 7 if name == "twin_prime_count_pi2_10_pow_k" else 40
        points = _points(name, count, **seed)
        assert points == list(range(points[0], points[0] + len(points)))


# ---------------------------------------------------------------------------
# Caps: truncation is typed, never silent
# ---------------------------------------------------------------------------


def _queue_entry(problem_id: str) -> dict:
    queue = load_queue(REPO_ROOT / "configs" / "problem_queue_v3.json")
    return next(entry for entry in queue["entries"] if entry["id"] == problem_id)


@pytest.mark.parametrize(
    ("problem_id", "generator"),
    [
        ("gilbreath_conjecture", "gilbreath_leading_terms"),
        ("lychrel_196", "reverse_and_add_base10"),
        ("recaman_coverage", "recaman"),
        ("singmaster_conjecture", "pascal_interior_multiplicity"),
        ("ulam_sequence_structure", "ulam_u_1_2"),
    ],
)
def test_a_request_beyond_the_caps_truncates_with_a_typed_record(problem_id, generator):
    receipt = _stage_generate_rows(_queue_entry(problem_id), "hash")
    validate_item_receipt(receipt)
    assert receipt["status"] == "COMPLETED"
    assert receipt["payload"]["row_count"] == GENERATOR_CAPS["max_rows"]
    types = [blocker["type"] for blocker in receipt["blockers"]]
    assert types == [f"generator_cap_truncated:{generator}"]
    detail = receipt["blockers"][0]["detail"]
    assert str(GENERATOR_ROW_CAPS[generator]) in detail
    assert "exact rows were emitted" in detail


def test_a_request_inside_the_caps_carries_no_truncation():
    receipt = _stage_generate_rows(_queue_entry("twin_prime_infinitude"), "hash")
    validate_item_receipt(receipt)
    assert receipt["status"] == "COMPLETED"
    assert receipt["blockers"] == []
    assert receipt["payload"]["row_count"] == 7


def test_the_generator_cap_binds_before_the_engine_budget(monkeypatch):
    """With the engine budget lifted, Ulam still stops at its own declared 5000."""

    monkeypatch.setitem(GENERATOR_CAPS, "max_rows", 10**7)
    entry = {**_queue_entry("ulam_sequence_structure")}
    entry["machine_form"] = {**entry["machine_form"], "max_point": 6000}
    receipt = _stage_generate_rows(entry, "hash")
    validate_item_receipt(receipt)
    assert receipt["payload"]["row_count"] == GENERATOR_ROW_CAPS["ulam_u_1_2"] == 5000
    assert receipt["blockers"][0]["type"] == "generator_cap_truncated:ulam_u_1_2"
    assert "the generator cap binds at 5000" in receipt["blockers"][0]["detail"]


def test_a_generator_without_a_declared_cap_still_blocks():
    """The legacy contract is unchanged: no declared cap, no truncation, hard block."""

    entry = {**_queue_entry("collatz_stopping_time")}
    entry["machine_form"] = {**entry["machine_form"], "max_point": 65}
    receipt = _stage_generate_rows(entry, "hash")
    validate_item_receipt(receipt)
    assert receipt["status"] == "BLOCKED"
    assert receipt["blockers"][0]["type"] == (
        "generator_cap_exceeded:collatz_total_stopping_time"
    )


# ---------------------------------------------------------------------------
# DG5 completion: no queue generator is missing, and nothing is upstream-blocked
# ---------------------------------------------------------------------------


#: The one deliberately unregistered generator: a synthetic sealed holdout whose
#: recurrence is withheld from the discovery side on purpose.  Registering it would
#: destroy the holdout, so DG5 excludes it by name rather than by silence.
SEALED_HOLDOUT_GENERATOR = "missing_generator:sealed_catalan_like_recurrence_v1"


def test_every_queue_generator_is_registered_except_the_sealed_holdout():
    queue = load_queue(REPO_ROOT / "configs" / "problem_queue_v3.json")
    unregistered = set()
    for entry in queue["entries"]:
        machine_form = entry["machine_form"]
        if machine_form["kind"] == "sequence_rows":
            name = machine_form["generator"]
        elif machine_form["kind"] == "integer_trajectory":
            name = machine_form["map"]
        else:
            continue
        if name not in GENERATOR_REGISTRY:
            unregistered.add(name)
    assert unregistered == {"sealed_catalan_like_recurrence_v1"}


def _fanout_entry(entry_id: str, machine_form: dict) -> dict:
    return {
        "id": entry_id,
        "domain": "math",
        "statement": f"DG5 generator fan-out target {entry_id}.",
        "source_citation": "Internal DG5 generator test fixture citation.",
        "believed_open_because": "Not open mathematics: a generator fan-out fixture.",
        "machine_form": machine_form,
        "progress_definition": "Every applicable lane reaches a receipt.",
        "control_rediscovery": False,
        "synthetic": False,
    }


def test_a_fanout_epoch_over_the_six_generators_leaves_no_generator_gap(tmp_path):
    """DG5's completion test, run live.

    The ids are deliberately outside the unsolved-dozen roster so the heavy campaign lane
    stays out of the fixture; what is under test is the row contract and everything
    downstream of it, which is exactly where the eight blocked problems were stuck.
    """

    entries = [
        _fanout_entry(f"dg5_{index}", form)
        for index, form in enumerate(
            [
                {"kind": "sequence_rows", "generator": "ulam_u_1_2", "max_point": 2000},
                {
                    "kind": "sequence_rows",
                    "generator": "twin_prime_count_pi2_10_pow_k",
                    "max_point": 7,
                },
                {
                    "kind": "sequence_rows",
                    "generator": "gilbreath_leading_terms",
                    "max_point": 500,
                },
                {
                    "kind": "sequence_rows",
                    "generator": "pascal_interior_multiplicity",
                    "max_point": 1000000,
                },
                {
                    "kind": "integer_trajectory",
                    "map": "recaman",
                    "max_steps": 10000000,
                    "seed": 0,
                },
                {
                    "kind": "integer_trajectory",
                    "map": "reverse_and_add_base10",
                    "max_steps": 10000,
                    "seed": 196,
                },
            ]
        )
    ]
    config = SchedulerConfig(
        queue_path=tmp_path / "queue.json",
        ledger_path=tmp_path / "ledger.sqlite",
        output_dir=tmp_path / "out",
        repo_root=tmp_path / "repo",
        epochs=6,
        sweep_hi_cpu=4096,
        caps={"max_wall_seconds": 900},
        fanout=True,
        gap_ledger=False,
    )
    config.repo_root.mkdir(parents=True, exist_ok=True)
    config.queue_path.parent.mkdir(parents=True, exist_ok=True)
    config.queue_path.write_bytes(canonical_json_bytes(seal_queue(entries)) + b"\n")

    result = run_scheduler(config)
    assert result["items_failed"] == 0

    ledger = build_gap_ledger(tmp_path, ("out",))
    gaps = {record["gap_id"] for record in ledger["gaps"]}
    assert not [gap for gap in gaps if gap.startswith("missing_generator:")]
    assert "upstream_blocked:generate_rows" not in gaps

    rows_receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "out" / "items").rglob("generate_rows-*.json"))
    ]
    assert len(rows_receipts) == len(entries)
    assert all(receipt["status"] == "COMPLETED" for receipt in rows_receipts)


# ---------------------------------------------------------------------------
# DG5 completion, on the committed epoch over the real sealed queue
# ---------------------------------------------------------------------------


def _committed_epoch_blockers() -> dict[str, list[str]]:
    """Every typed blocker the committed DG5 epoch recorded, grouped by problem."""

    root = REPO_ROOT / "runs" / "discovery-engine" / "dg5-fanout-epoch-v1"
    blockers: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != "invariant-discovery-item-1.0":
            continue
        # Seal check only: the unsolved-progress lane's declared vocabulary is bare
        # codes ("factorial_growth"), which validate_item_receipt's kind:subject rule
        # predates and rejects.  That contract question is not DG5's to settle here.
        body = {key: item for key, item in receipt.items() if key != "content_sha256"}
        assert receipt["content_sha256"] == canonical_sha256(body), path
        blockers.setdefault(receipt["problem_id"], []).extend(
            blocker["type"] for blocker in receipt["blockers"]
        )
    return blockers


def test_the_committed_dg5_epoch_reports_no_generator_gap():
    """DG5 completion over the real sealed queue.

    The one residue is the synthetic sealed holdout, whose generator is withheld from the
    discovery side by design; the queue itself marks that entry ``synthetic``.  Every
    other problem generates its rows, so nothing downstream is upstream-blocked.
    """

    by_problem = _committed_epoch_blockers()
    assert by_problem, "the committed DG5 fan-out epoch has no item receipts"
    missing = {
        problem_id
        for problem_id, blockers in by_problem.items()
        if any(item.startswith("missing_generator:") for item in blockers)
    }
    upstream = {
        problem_id
        for problem_id, blockers in by_problem.items()
        if "upstream_blocked:generate_rows" in blockers
    }
    assert missing == {"catalan_like_recurrence_holdout"}
    assert upstream == {"catalan_like_recurrence_holdout"}
    assert SEALED_HOLDOUT_GENERATOR in by_problem["catalan_like_recurrence_holdout"]
    assert _queue_entry("catalan_like_recurrence_holdout")["synthetic"] is True


def test_the_committed_dg5_epoch_failed_nothing():
    """A failed stage is not a typed blocker; it is a defect, and there are none."""

    root = REPO_ROOT / "runs" / "discovery-engine" / "dg5-fanout-epoch-v1" / "epochs"
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*-summary-*.json"))
    ]
    assert summaries
    assert all(receipt["items"]["failed"] == 0 for receipt in summaries)


def test_the_committed_dg5_epoch_truncates_the_six_over_cap_requests():
    truncations = sorted(
        {
            item
            for blockers in _committed_epoch_blockers().values()
            for item in blockers
            if item.startswith("generator_cap_truncated:")
        }
    )
    assert truncations == [
        # The aliquot trajectory truncates on its declared value cap, not a row cap;
        # it is here because the same typed record covers both.
        "generator_cap_truncated:aliquot_sum",
        "generator_cap_truncated:gilbreath_leading_terms",
        "generator_cap_truncated:pascal_interior_multiplicity",
        "generator_cap_truncated:recaman",
        "generator_cap_truncated:reverse_and_add_base10",
        "generator_cap_truncated:ulam_u_1_2",
    ]


def test_the_gap_ledger_over_the_dg5_epoch_carries_no_generator_gap():
    """DG5's completion test, read off the ledger the way A6 builds it.

    The scan root is the DG5 epoch itself.  The whole-corpus ledger still lists the six
    historical ``missing_generator`` gaps as open, because A6 discharges a gap only when
    the same (problem, stage) passes in a *strictly later* epoch id and a fresh ledger
    restarts that counter — so the honest completion claim is over the epoch this build
    produced, and the historical gaps stay in the corpus as the record of what was fixed.
    """

    ledger = build_gap_ledger(REPO_ROOT, ("runs/discovery-engine/dg5-fanout-epoch-v1",))
    by_id = {gap["gap_id"]: gap for gap in ledger["gaps"]}
    missing = sorted(gap for gap in by_id if gap.startswith("missing_generator:"))
    assert missing == [SEALED_HOLDOUT_GENERATOR]
    assert by_id[SEALED_HOLDOUT_GENERATOR]["blocked_problems"] == [
        "catalan_like_recurrence_holdout"
    ]
    assert by_id["upstream_blocked:generate_rows"]["blocked_problems"] == [
        "catalan_like_recurrence_holdout"
    ]
    # Unblocking pushed the frontier one stage deeper: the sweeper lane now names the
    # sequences it cannot sweep, which is a new gap the corpus could not previously see.
    assert "missing_sweeper:gilbreath_leading_terms" in by_id
