"""A6 — the capability gap ledger: the machine's failures, aggregated into a build queue.

Every honest module in this repository ends a thing it cannot do with a typed blocker
rather than silence.  Those blockers are scattered across thousands of sealed receipts,
one per run, each naming one obstruction in one place.  Individually they are notes.
Aggregated they are the only build queue that was never guessed: the capabilities the
system actually reached for and did not have, ranked by how much they would unblock.

This module does the aggregation.  It walks the declared scan roots, extracts every typed
blocker with its provenance, groups them by the blocker string itself, and ranks them by a
formula written out in the receipt.

Four rules keep the ledger honest.

**The gap id is the blocker string, verbatim.**  No normalization, no clustering, no
renaming.  ``missing_prover:sign`` and ``missing_prover:monotonicity`` are two gaps
because two different provers are missing.  A gap the system stopped emitting does not
disappear; it becomes ``discharged`` with the receipt that discharged it.

**The ranking is a formula, not a score.**  ``unblock_value`` records the two integers it
is computed from — how many distinct problems (or declared physics family groups) the gap
blocks, and how deep in the pipeline the deepest block occurs — plus the exact ordering
rule as a string.  Anyone can recompute the whole ranking by hand from the receipt.  There
is no scalar priority beyond that, and ``scalar_truth_or_probability_score`` is sealed
``false`` per house rule.

**Discharge is evidence, never assumption.**  A gap is marked discharged only by one of
two declared rules: a later epoch shows the same ``(problem, stage)`` completing without
the blocker, or a frozen discharge rule names the module and receipt that resolved it and
that receipt is present and sealed.  The rule that fired is recorded on the gap.

**Extraction is declared.**  :data:`EXTRACTORS` names every place a blocker is read from,
by schema version and JSON location.  A receipt shape nobody declared contributes nothing,
and the count of files that parsed, failed, or were skipped is reported.

Claim boundary: the ledger asserts what the corpus recorded.  A gap's rank asserts how
many declared units it blocks, never how hard it is to build, how valuable the underlying
mathematics is, or how likely a fix is to succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .lane_registry import REGISTRY_CONTENT_SHA256, lanes_emitting
from .sigma_core import canonical_json_bytes, canonical_sha256

LEDGER_SCHEMA = "invariant-capability-gap-ledger-1.0"

#: Declared scan roots, repo-root relative.  Every ``*.json`` beneath them is a candidate
#: receipt; nothing outside them is read.
SCAN_ROOTS: tuple[str, ...] = ("runs",)

#: The ledger writes its own receipt inside a scan root, so it must not read itself: a
#: ledger that ingested its own output would change on every rebuild and could never be
#: byte-checked.  Excluded by exact repo-relative path, declared here and nowhere else.
SELF_EXCLUDED_PATHS: tuple[str, ...] = ("runs/discovery-engine/capability-gap-ledger.json",)

#: Files larger than this are recorded as skipped rather than parsed, so a single huge
#: artifact can never silently stall a ledger build.
MAX_RECEIPT_BYTES = 64 * 1024 * 1024

#: A typed blocker is an identifier, optionally ``kind:subject``.  Prose is not a blocker.
TYPED_BLOCKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(:[A-Za-z0-9_./*-]+)?$")
MAX_BLOCKER_CHARS = 200

#: How deep in the pipeline each stage sits.  A block that happens after more upstream
#: work has already succeeded costs more to reach, so it breaks ranking ties upward.
#: ``0`` is the declared depth of anything outside the scheduler's stage graph.
PIPELINE_DEPTH: dict[str, int] = {
    "generate_rows": 1,
    "note_gpu_campaign_receipts": 1,
    "basis_synthesis": 2,
    "conjecture": 2,
    "nonlinear_search": 2,
    "holonomic_guess": 3,
    "spectral_scan": 3,
    "structural_repair": 3,
    "unsolved_progress": 3,
    "inverse_symbolic": 3,
    "lemma_decomposition": 4,
    "quantified_inequality": 4,
    "route_provers": 4,
    "diophantine_sweep": 5,
    "sat_certificate": 5,
    "sweep": 5,
    "physics_ladder_rung": 6,
    "physics_materialization": 6,
}

UNBLOCK_VALUE_FORMULA = (
    "unblock_value.distinct_units_blocked = |blocked_problems| + |blocked_families|; "
    "unblock_value.pipeline_depth = max(PIPELINE_DEPTH[stage]) over the gap's evidence "
    "(0 for stages outside the declared graph). Rank ascending by the key "
    "(-distinct_units_blocked, -pipeline_depth, gap_id). Both integers are printed on "
    "every row, so the whole ranking is recomputable by hand from this receipt. "
    "families_declared_blocked_max reports the family count a physics receipt declared "
    "for itself and is deliberately NOT an input to the order."
)

FIRST_SEEN_RULE = (
    "first_seen_receipt is the lexicographically smallest repo-relative POSIX receipt "
    "path among the gap's evidence. Receipts carry no timestamps, so 'first' is defined "
    "by a total order on paths, never by wall-clock time."
)

#: Every declared place a typed blocker is read from.  ``matches`` is the receipt
#: predicate; ``location`` is the JSON path; ``stage`` is the pipeline stage attributed to
#: the block when the receipt does not name one.
EXTRACTORS: tuple[dict[str, str], ...] = (
    {
        "extractor_id": "discovery_item_blockers",
        "matches": "schema_version == invariant-discovery-item-1.0",
        "location": "blockers[].type",
        "stage": "(from the receipt's own stage field)",
        "unit": "problem",
    },
    {
        "extractor_id": "discovery_epoch_outcomes",
        "matches": "schema_version == invariant-discovery-epoch-1.0",
        "location": "per_problem_outcomes[problem_id][stage].blockers[]",
        "stage": "(the outcome's own stage key)",
        "unit": "problem",
    },
    {
        "extractor_id": "unsolved_progress_first_blocker",
        "matches": "schema_version == invariant-unsolved-progress-1.0",
        "location": "first_blocker.code",
        "stage": "unsolved_progress",
        "unit": "problem",
    },
    {
        "extractor_id": "unsolved_campaign_blockers",
        "matches": "schema_version == invariant-unsolved-progress-campaign-1.0",
        "location": "blockers[].code with blockers[].problem_id",
        "stage": "unsolved_progress",
        "unit": "problem",
    },
    {
        "extractor_id": "adapter_gap_report",
        "matches": "top-level adapter_gap_report.adapters_to_build",
        "location": "adapter_gap_report.adapters_to_build[].code",
        "stage": "physics_materialization",
        "unit": "family",
    },
    {
        "extractor_id": "residual_gap_report",
        "matches": "top-level residual_gap_report",
        "location": "residual_gap_report.residual_blockers[] and .still_blocked_at_a_rung{}",
        "stage": "physics_ladder_rung",
        "unit": "family",
    },
    {
        "extractor_id": "typed_first_blocker",
        "matches": "any other receipt carrying a top-level first_blocker",
        "location": "first_blocker (string) or first_blocker.code",
        "stage": "(unattributed)",
        "unit": "problem",
    },
)

#: Frozen discharge rules: gap_id -> the module and receipt shape that resolved it.
#: A rule fires only when a matching sealed receipt is actually present in the corpus.
DISCHARGE_RULES: tuple[dict[str, Any], ...] = (
    {
        "gap_id": "missing_sweeper:diophantine_family",
        "discharged_by_module": "sigma_theory_compiler.exponent_diophantine_sweeper",
        "evidence_schema_version": "invariant-exponent-diophantine-sweep-1.0",
        "evidence_decisions": (
            "COUNTEREXAMPLE_CANDIDATE",
            "KNOWN_SOLUTIONS_REDISCOVERED_ONLY",
            "NEW_TO_BUILTIN_TABLE_PRESENT",
            "NO_COUNTEREXAMPLE_IN_BOX",
            "NO_SOLUTION_IN_BOX",
            "NO_UNSOLVABLE_N_IN_RANGE",
            "UNSOLVABLE_CANDIDATE_IN_RANGE",
        ),
        "evidence_must_name_gap": True,
        "statement": (
            "the exponent-Diophantine sweeper is the diophantine_family witness sweeper "
            "whose absence the scheduler recorded; its scope names the blocker verbatim"
        ),
    },
    {
        "gap_id": "statement_kinds_too_weak",
        "discharged_by_module": "sigma_theory_compiler.spectral_signal_scan",
        "evidence_schema_version": "invariant-spectral-signal-scan-result-1.0",
        "evidence_decisions": ("SPECTRAL_BIAS_SURVIVED",),
        "evidence_must_name_gap": False,
        "evidence_field_contains": {"sequence_label": "ulam"},
        "statement": (
            "the Ulam progress receipt named statement_kinds_too_weak for a "
            "quasi-periodic signal no algebraic statement kind can express; the spectral "
            "bias lane expresses exactly that kind and its Ulam scan survived the holdout"
        ),
    },
    {
        "gap_id": "missing_adapter:nonlocal_fractional_operator",
        "discharged_by_module": "sigma_theory_compiler.nonlocal_fractional_adapter",
        "evidence_schema_version": "invariant-nonlocal-fractional-localization-result-1.0",
        "evidence_decisions": (),
        "evidence_must_name_gap": True,
        "statement": (
            "auxiliary-field localization materialized a nonlocal arm for every v3 family; "
            "the localization receipt's own decision states the discharge, at the level of "
            "the finite-pole approximant"
        ),
    },
)

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "gap_rank_measures_build_difficulty": False,
    "ranking_formula_is_declared_and_hand_computable": True,
    "scalar_truth_or_probability_score": False,
    "unattempted_lane_is_a_recorded_fact": True,
}

SCOPE = (
    "Aggregation of every typed blocker recorded in the sealed receipt corpus under the "
    "declared scan roots, grouped by the blocker string verbatim and ranked by the "
    "declared unblock_value formula. Each gap binds the receipts it was read from by "
    "content hash. A gap's rank asserts how many declared problems or physics family "
    "groups it blocks and how deep the block occurs; it asserts nothing about how hard "
    "the missing capability is to build, nor about the mathematics behind it."
)


class CapabilityGapLedgerError(ValueError):
    """Raised on a malformed ledger, a broken seal, or an undeclared extractor result."""


# ---------------------------------------------------------------------------
# Receipt scanning
# ---------------------------------------------------------------------------


def _is_typed_blocker(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_BLOCKER_CHARS
        and TYPED_BLOCKER_RE.fullmatch(value) is not None
    )


def _seal_of(value: Mapping[str, Any], data: bytes) -> str:
    """The receipt's own declared seal, or the SHA-256 of its bytes when it has none."""

    seal = value.get("content_sha256")
    if isinstance(seal, str) and re.fullmatch(r"[0-9a-f]{64}", seal):
        return seal
    return hashlib.sha256(data).hexdigest()


def iter_receipt_files(root: Path, scan_roots: Sequence[str] = SCAN_ROOTS) -> Iterator[Path]:
    """Every candidate receipt path beneath the declared scan roots, sorted."""

    for name in scan_roots:
        base = Path(root) / name
        if not base.is_dir():
            continue
        yield from sorted(base.rglob("*.json"))


class _Observation:
    """One typed blocker read out of one receipt."""

    __slots__ = (
        "extractor", "family_count", "gap_id", "path", "seal", "stage", "unit", "unit_kind",
    )

    def __init__(
        self,
        gap_id: str,
        unit: str | None,
        unit_kind: str,
        stage: str | None,
        path: str,
        seal: str,
        extractor: str,
        family_count: int = 0,
    ) -> None:
        self.gap_id = gap_id
        self.unit = unit
        self.unit_kind = unit_kind
        self.stage = stage
        self.path = path
        self.seal = seal
        self.extractor = extractor
        self.family_count = family_count


def _fallback_unit(path: str) -> str:
    """A stable unit id for a receipt that names neither a problem nor a campaign.

    The bare file stem is not unique across the corpus (``receipt.json`` appears in many
    directories), so the unit is the receipt's parent directory qualified by its stem.
    """

    parts = Path(path).parts
    parent = parts[-2] if len(parts) >= 2 else ""
    stem = Path(path).stem
    return f"{parent}/{stem}" if parent else stem


def _extract(value: Mapping[str, Any], path: str, seal: str) -> list[_Observation]:
    """Apply every declared extractor to one parsed receipt."""

    out: list[_Observation] = []
    schema = value.get("schema_version")

    if schema == "invariant-discovery-item-1.0":
        stage = value.get("stage")
        problem = value.get("problem_id")
        for blocker in value.get("blockers") or ():
            if isinstance(blocker, Mapping) and _is_typed_blocker(blocker.get("type")):
                out.append(
                    _Observation(
                        blocker["type"], problem, "problem", stage, path, seal,
                        "discovery_item_blockers",
                    )
                )
        return out

    if schema == "invariant-discovery-epoch-1.0":
        outcomes = value.get("per_problem_outcomes") or {}
        for problem, stages in outcomes.items():
            if not isinstance(stages, Mapping):
                continue
            for stage, outcome in stages.items():
                if not isinstance(outcome, Mapping):
                    continue
                for blocker in outcome.get("blockers") or ():
                    if _is_typed_blocker(blocker):
                        out.append(
                            _Observation(
                                blocker, problem, "problem", stage, path, seal,
                                "discovery_epoch_outcomes",
                            )
                        )
        return out

    if schema == "invariant-unsolved-progress-1.0":
        blocker = value.get("first_blocker")
        if isinstance(blocker, Mapping) and _is_typed_blocker(blocker.get("code")):
            out.append(
                _Observation(
                    blocker["code"], value.get("problem_id"), "problem", "unsolved_progress",
                    path, seal, "unsolved_progress_first_blocker",
                )
            )
        return out

    if schema == "invariant-unsolved-progress-campaign-1.0":
        for record in value.get("blockers") or ():
            if isinstance(record, Mapping) and _is_typed_blocker(record.get("code")):
                out.append(
                    _Observation(
                        record["code"], record.get("problem_id"), "problem",
                        "unsolved_progress", path, seal, "unsolved_campaign_blockers",
                    )
                )
        return out

    stem = Path(path).stem
    report = value.get("adapter_gap_report")
    if isinstance(report, Mapping):
        for adapter in report.get("adapters_to_build") or ():
            if not isinstance(adapter, Mapping) or not _is_typed_blocker(adapter.get("code")):
                continue
            mechanism = adapter.get("component_mechanism") or "unspecified_mechanism"
            declared = 0
            for key in ("families_blocked_at_materialization", "families_blocked_at_a_ladder_rung"):
                count = adapter.get(key)
                if isinstance(count, int) and not isinstance(count, bool):
                    declared = max(declared, count)
            out.append(
                _Observation(
                    adapter["code"], f"{stem}#{mechanism}", "family",
                    "physics_materialization", path, seal, "adapter_gap_report", declared,
                )
            )
    residual = value.get("residual_gap_report")
    if isinstance(residual, Mapping):
        for blocker in residual.get("residual_blockers") or ():
            if _is_typed_blocker(blocker):
                out.append(
                    _Observation(
                        blocker, f"{stem}#residual", "family", "physics_materialization",
                        path, seal, "residual_gap_report",
                    )
                )
        rung = residual.get("still_blocked_at_a_rung")
        if isinstance(rung, Mapping):
            for blocker, count in rung.items():
                if _is_typed_blocker(blocker):
                    declared = count if isinstance(count, int) and not isinstance(count, bool) else 0
                    out.append(
                        _Observation(
                            blocker, f"{stem}#rung", "family", "physics_ladder_rung",
                            path, seal, "residual_gap_report", declared,
                        )
                    )
    if out:
        return out

    blocker = value.get("first_blocker")
    code = blocker.get("code") if isinstance(blocker, Mapping) else blocker
    if _is_typed_blocker(code):
        unit = value.get("problem_id") or value.get("campaign_id") or _fallback_unit(path)
        out.append(
            _Observation(code, unit, "problem", None, path, seal, "typed_first_blocker")
        )
    return out


def scan_corpus(
    root: Path, scan_roots: Sequence[str] = SCAN_ROOTS
) -> tuple[list[_Observation], dict[str, Any], dict[str, Any]]:
    """Read every declared receipt once.  Returns observations, epoch facts, and counts."""

    observations: list[_Observation] = []
    epochs: list[dict[str, Any]] = []
    discharge_evidence: dict[str, list[dict[str, str]]] = {}
    counts = {"files_oversize": 0, "files_parsed": 0, "files_seen": 0, "files_unparseable": 0}
    base = Path(root)
    for file_path in iter_receipt_files(base, scan_roots):
        if file_path.relative_to(base).as_posix() in SELF_EXCLUDED_PATHS:
            # Declared exclusion, deliberately uncounted: a count that depends on whether
            # the previous build already ran would make the receipt unstable under
            # --validate-checked, which is exactly the property that check exists to test.
            continue
        counts["files_seen"] += 1
        try:
            size = file_path.stat().st_size
        except OSError:
            counts["files_unparseable"] += 1
            continue
        if size > MAX_RECEIPT_BYTES:
            counts["files_oversize"] += 1
            continue
        try:
            data = file_path.read_bytes()
            value = json.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            counts["files_unparseable"] += 1
            continue
        if not isinstance(value, Mapping):
            counts["files_parsed"] += 1
            continue
        counts["files_parsed"] += 1
        relative = file_path.relative_to(base).as_posix()
        seal = _seal_of(value, data)
        observations.extend(_extract(value, relative, seal))
        if value.get("schema_version") == "invariant-discovery-epoch-1.0":
            epochs.append(
                {
                    "epoch_id": value.get("epoch_id"),
                    "outcomes": value.get("per_problem_outcomes") or {},
                    "path": relative,
                }
            )
        for rule in DISCHARGE_RULES:
            if _rule_matches(rule, value, data):
                discharge_evidence.setdefault(rule["gap_id"], []).append(
                    {"path": relative, "content_sha256": seal}
                )
    observations.sort(key=lambda item: (item.gap_id, item.path, item.unit or "", item.stage or ""))
    epochs.sort(key=lambda item: (item["epoch_id"] if item["epoch_id"] is not None else 0))
    return observations, {"epochs": epochs, "discharge_evidence": discharge_evidence}, counts


def _rule_matches(rule: Mapping[str, Any], value: Mapping[str, Any], data: bytes) -> bool:
    if value.get("schema_version") != rule["evidence_schema_version"]:
        return False
    decisions = rule["evidence_decisions"]
    decision = value.get("decision")
    if decisions and not any(
        isinstance(decision, str) and decision.startswith(item) for item in decisions
    ):
        return False
    for field, needle in (rule.get("evidence_field_contains") or {}).items():
        item = value.get(field)
        if not isinstance(item, str) or needle not in item:
            return False
    return not (
        rule.get("evidence_must_name_gap") and rule["gap_id"].encode("utf-8") not in data
    )


# ---------------------------------------------------------------------------
# Discharge detection
# ---------------------------------------------------------------------------


def detect_discharges(
    observations: Sequence[_Observation], facts: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Which gaps are discharged, by which declared rule, on which evidence.

    Rule ``same_problem_stage_now_passing``: a ``(problem, stage)`` that carried the gap in
    one epoch and COMPLETED without it in a strictly later epoch.  Epoch ids order the
    corpus; no timestamp is consulted.

    Rule ``declared_discharge_receipt``: a frozen :data:`DISCHARGE_RULES` entry whose
    evidence receipt is present in the corpus.
    """

    discharged: dict[str, dict[str, Any]] = {}

    blocked_at: dict[tuple[str, str, str], int] = {}
    passed_at: dict[tuple[str, str], list[tuple[int, str, list[str]]]] = {}
    for epoch in facts["epochs"]:
        epoch_id = epoch["epoch_id"]
        if not isinstance(epoch_id, int):
            continue
        for problem, stages in epoch["outcomes"].items():
            if not isinstance(stages, Mapping):
                continue
            for stage, outcome in stages.items():
                if not isinstance(outcome, Mapping):
                    continue
                blockers = [item for item in (outcome.get("blockers") or []) if isinstance(item, str)]
                for gap_id in blockers:
                    key = (gap_id, problem, stage)
                    if key not in blocked_at or epoch_id < blocked_at[key]:
                        blocked_at[key] = epoch_id
                if outcome.get("status") == "COMPLETED":
                    passed_at.setdefault((problem, stage), []).append(
                        (epoch_id, epoch["path"], blockers)
                    )

    for (gap_id, problem, stage), first_epoch in sorted(blocked_at.items()):
        for epoch_id, path, blockers in sorted(passed_at.get((problem, stage), [])):
            if epoch_id > first_epoch and gap_id not in blockers:
                discharged.setdefault(
                    gap_id,
                    {
                        "discharge_rule": "same_problem_stage_now_passing",
                        "discharged_by": (
                            f"epoch {epoch_id}: {problem}/{stage} COMPLETED without the blocker"
                        ),
                        "evidence_receipts": [path],
                    },
                )
                break

    for rule in DISCHARGE_RULES:
        evidence = facts["discharge_evidence"].get(rule["gap_id"])
        if not evidence:
            continue
        discharged[rule["gap_id"]] = {
            "discharge_rule": "declared_discharge_receipt",
            "discharged_by": rule["discharged_by_module"],
            "evidence_receipts": [item["path"] for item in sorted(
                evidence, key=lambda record: record["path"]
            )][:5],
            "statement": rule["statement"],
        }
    return discharged


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_gap_records(
    observations: Sequence[_Observation], discharges: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Group observations into ranked :class:`GapRecord`-shaped dicts."""

    grouped: dict[str, list[_Observation]] = {}
    for item in observations:
        grouped.setdefault(item.gap_id, []).append(item)

    records: list[dict[str, Any]] = []
    for gap_id, items in grouped.items():
        problems = sorted({item.unit for item in items if item.unit_kind == "problem" and item.unit})
        families = sorted({item.unit for item in items if item.unit_kind == "family" and item.unit})
        stages = sorted({item.stage for item in items if item.stage})
        depth = max((PIPELINE_DEPTH.get(stage, 0) for stage in stages), default=0)
        deepest = sorted(
            (stage for stage in stages if PIPELINE_DEPTH.get(stage, 0) == depth), key=str
        )
        paths = sorted({item.path for item in items})
        seal_by_path = {item.path: item.seal for item in items}
        discharge = discharges.get(gap_id)
        records.append(
            {
                "blocked_count": len(problems) + len(families),
                "blocked_families": families,
                "blocked_problems": problems,
                "discharge_rule": discharge["discharge_rule"] if discharge else None,
                "discharged_by": discharge["discharged_by"] if discharge else None,
                "example_receipts": [
                    {"content_sha256": seal_by_path[path], "path": path} for path in paths[:5]
                ],
                "extractors": sorted({item.extractor for item in items}),
                "families_declared_blocked_max": max(
                    (item.family_count for item in items), default=0
                ),
                "first_seen_receipt": paths[0],
                "gap_id": gap_id,
                "lanes_that_emit_it": list(lanes_emitting(gap_id)),
                "observations": len(items),
                "receipts_total": len(paths),
                "stages": stages,
                "status": "discharged" if discharge else "open",
                "unblock_value": {
                    "deepest_stage": deepest[0] if deepest else None,
                    "distinct_units_blocked": len(problems) + len(families),
                    "formula": UNBLOCK_VALUE_FORMULA,
                    "pipeline_depth": depth,
                },
            }
        )
    records.sort(
        key=lambda record: (
            -record["unblock_value"]["distinct_units_blocked"],
            -record["unblock_value"]["pipeline_depth"],
            record["gap_id"],
        )
    )
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def build_ledger(root: Path, scan_roots: Sequence[str] = SCAN_ROOTS) -> dict[str, Any]:
    """Scan, aggregate, rank, and seal the capability gap ledger."""

    observations, facts, counts = scan_corpus(root, scan_roots)
    discharges = detect_discharges(observations, facts)
    records = build_gap_records(observations, discharges)
    open_records = [record for record in records if record["status"] == "open"]
    body = {
        "claims": CLAIMS,
        "corpus": {
            **counts,
            "epoch_receipts": len(facts["epochs"]),
            "observations": len(observations),
        },
        "counts": {
            "gaps_discharged": len(records) - len(open_records),
            "gaps_open": len(open_records),
            "gaps_total": len(records),
        },
        "discharge_rules": [
            {
                "discharged_by_module": rule["discharged_by_module"],
                "evidence_schema_version": rule["evidence_schema_version"],
                "gap_id": rule["gap_id"],
                "statement": rule["statement"],
            }
            for rule in DISCHARGE_RULES
        ],
        "extractors": [dict(item) for item in EXTRACTORS],
        "first_seen_rule": FIRST_SEEN_RULE,
        "gaps": records,
        "lane_registry_content_sha256": REGISTRY_CONTENT_SHA256,
        "pipeline_depth": dict(sorted(PIPELINE_DEPTH.items())),
        "ranking_formula": UNBLOCK_VALUE_FORMULA,
        "scan_roots": list(scan_roots),
        "self_excluded_paths": list(SELF_EXCLUDED_PATHS),
        "schema_version": LEDGER_SCHEMA,
        "scope": SCOPE,
        "top_open_gaps": [record["gap_id"] for record in open_records[:5]],
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def top_open_gaps(ledger: Mapping[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """The ``limit`` highest-ranked open gaps, for the epoch summary receipt."""

    out: list[dict[str, Any]] = []
    for record in ledger["gaps"]:
        if record["status"] != "open":
            continue
        out.append(
            {
                "blocked_count": record["blocked_count"],
                "blocked_problems": record["blocked_problems"][:8],
                "gap_id": record["gap_id"],
                "lanes_that_emit_it": record["lanes_that_emit_it"],
                "unblock_value": {
                    "distinct_units_blocked": record["unblock_value"]["distinct_units_blocked"],
                    "pipeline_depth": record["unblock_value"]["pipeline_depth"],
                },
            }
        )
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ledger(value: Any) -> None:
    """Fail closed on schema drift, a broken seal, or an undeclared claim."""

    if not isinstance(value, Mapping):
        raise CapabilityGapLedgerError("ledger must be a mapping")
    if value.get("schema_version") != LEDGER_SCHEMA:
        raise CapabilityGapLedgerError("ledger schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CapabilityGapLedgerError("ledger seal changed")
    if value["claims"] != CLAIMS:
        raise CapabilityGapLedgerError("ledger claims changed")
    if value["claims"]["scalar_truth_or_probability_score"]:
        raise CapabilityGapLedgerError("ledger asserts a forbidden scalar score")
    if value["ranking_formula"] != UNBLOCK_VALUE_FORMULA:
        raise CapabilityGapLedgerError("ledger ranking formula changed")
    previous: tuple[int, int, str] | None = None
    for record in value["gaps"]:
        if record["status"] not in ("open", "discharged"):
            raise CapabilityGapLedgerError(f"unknown gap status: {record['status']!r}")
        if record["status"] == "discharged" and not record["discharged_by"]:
            raise CapabilityGapLedgerError("a discharged gap must name what discharged it")
        if len(record["example_receipts"]) > 5:
            raise CapabilityGapLedgerError("example_receipts exceeds the declared cap of 5")
        expected = len(record["blocked_problems"]) + len(record["blocked_families"])
        if record["blocked_count"] != expected:
            raise CapabilityGapLedgerError(f"blocked_count is not the declared sum: {record['gap_id']}")
        if record["unblock_value"]["distinct_units_blocked"] != expected:
            raise CapabilityGapLedgerError(
                f"unblock_value disagrees with blocked_count: {record['gap_id']}"
            )
        key = (
            -record["unblock_value"]["distinct_units_blocked"],
            -record["unblock_value"]["pipeline_depth"],
            record["gap_id"],
        )
        if previous is not None and key < previous:
            raise CapabilityGapLedgerError(f"gaps are not in declared rank order at {record['gap_id']}")
        previous = key


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------


def render_markdown(ledger: Mapping[str, Any], limit: int = 40) -> str:
    """Deterministic human-readable render of the ranked ledger."""

    counts = ledger["counts"]
    lines = [
        "# Capability gaps",
        "",
        "Generated by `python -m sigma_theory_compiler.capability_gap_ledger build`.",
        "Do not edit by hand — regenerate.",
        "",
        f"- Ledger seal: `{ledger['content_sha256']}`",
        f"- Lane registry seal: `{ledger['lane_registry_content_sha256']}`",
        (
            f"- Scan roots: {', '.join('`' + item + '`' for item in ledger['scan_roots'])} "
            f"(excluding this ledger's own receipt)"
        ),
        (
            f"- Receipts parsed: {ledger['corpus']['files_parsed']} of "
            f"{ledger['corpus']['files_seen']} seen "
            f"({ledger['corpus']['files_unparseable']} unparseable, "
            f"{ledger['corpus']['files_oversize']} over the declared size cap)"
        ),
        f"- Typed blocker observations: {ledger['corpus']['observations']}",
        (
            f"- Gaps: {counts['gaps_total']} total, {counts['gaps_open']} open, "
            f"{counts['gaps_discharged']} discharged"
        ),
        "",
        "## How the ranking is computed",
        "",
        ledger["ranking_formula"],
        "",
        ledger["first_seen_rule"],
        "",
        (
            "There is no scalar priority score anywhere in this document; every row "
            "prints the two integers the order is computed from."
        ),
        "",
        f"## Ranked open gaps (top {limit})",
        "",
        "| # | gap_id | units | depth | blocked problems / families | lanes that emit it |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    shown = 0
    for record in ledger["gaps"]:
        if record["status"] != "open":
            continue
        shown += 1
        if shown > limit:
            break
        units = record["blocked_problems"] + record["blocked_families"]
        rendered = ", ".join(f"`{item}`" for item in units[:6])
        if len(units) > 6:
            rendered += f", +{len(units) - 6} more"
        lanes = ", ".join(f"`{item}`" for item in record["lanes_that_emit_it"]) or "_unregistered_"
        lines.append(
            f"| {shown} | `{record['gap_id']}` | "
            f"{record['unblock_value']['distinct_units_blocked']} | "
            f"{record['unblock_value']['pipeline_depth']} | {rendered} | {lanes} |"
        )

    lines += [
        "",
        "## Discharged gaps",
        "",
        "| gap_id | discharged by | rule | units it had blocked |",
        "| --- | --- | --- | --- |",
    ]
    for record in ledger["gaps"]:
        if record["status"] != "discharged":
            continue
        lines.append(
            f"| `{record['gap_id']}` | `{record['discharged_by']}` | "
            f"`{record['discharge_rule']}` | {record['blocked_count']} |"
        )
    lines += [
        "",
        "## Declared extractors",
        "",
        "| extractor | matches | location | unit |",
        "| --- | --- | --- | --- |",
    ]
    for extractor in ledger["extractors"]:
        lines.append(
            f"| `{extractor['extractor_id']}` | {extractor['matches']} | "
            f"`{extractor['location']}` | {extractor['unit']} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_RECEIPT = Path("runs/discovery-engine/capability-gap-ledger.json")
DEFAULT_MARKDOWN = Path("docs/CAPABILITY_GAPS.md")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capability gap ledger (A6).")
    parser.add_argument("command", choices=("build",), nargs="?", default="build")
    parser.add_argument("--repo-root", default=".", help="repository root to scan")
    parser.add_argument("--output", default=None, help="ledger receipt path")
    parser.add_argument("--markdown", default=None, help="markdown render path")
    parser.add_argument(
        "--validate-checked",
        action="store_true",
        help="rebuild and require the stored receipt and markdown to match byte for byte",
    )
    args = parser.parse_args(argv)

    root = Path(args.repo_root)
    receipt_path = Path(args.output) if args.output else root / DEFAULT_RECEIPT
    markdown_path = Path(args.markdown) if args.markdown else root / DEFAULT_MARKDOWN
    ledger = build_ledger(root)
    validate_ledger(ledger)
    encoded = canonical_json_bytes(ledger) + b"\n"
    markdown = render_markdown(ledger).encode("utf-8")

    if args.validate_checked:
        if not receipt_path.exists():
            print(f"INVALID missing ledger receipt: {receipt_path}")
            return 1
        if receipt_path.read_bytes() != encoded:
            print(f"INVALID stored ledger differs from the rebuild: {receipt_path}")
            return 1
        if markdown_path.exists() and markdown_path.read_bytes().replace(b"\r\n", b"\n") != markdown:
            print(f"INVALID stored markdown differs from the rebuild: {markdown_path}")
            return 1
        print(
            f"VALID gaps={ledger['counts']['gaps_total']} "
            f"open={ledger['counts']['gaps_open']} "
            f"discharged={ledger['counts']['gaps_discharged']} "
            f"content_sha256={ledger['content_sha256']}"
        )
        return 0

    _write_bytes(receipt_path, encoded)
    _write_bytes(markdown_path, markdown)
    print(
        f"BUILT gaps={ledger['counts']['gaps_total']} "
        f"open={ledger['counts']['gaps_open']} "
        f"discharged={ledger['counts']['gaps_discharged']} -> {receipt_path}"
    )
    for record in top_open_gaps(ledger, 10):
        print(
            f"  {record['gap_id']}  units={record['blocked_count']}  "
            f"depth={record['unblock_value']['pipeline_depth']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
