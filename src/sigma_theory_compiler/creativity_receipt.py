"""The behavioural creativity measure, wired into a campaign's own receipt.

``creativity_measure`` answers "how many genuinely different things did the search try".  It
worked and nothing called it, so every campaign that has ever run reported its program count --
a number that says nothing, because sixteen spellings of the identity map count as sixteen --
and its behavioural diversity had to be computed by hand afterwards, if at all.  A number
computed by hand afterwards is not a measurement of the run; it is an opinion about the run.

This module makes it a measurement.  ``creativity_block`` seals the measure over a run's own
sealed programs and ``run_problem`` puts the result in the receipt, so behavioural diversity
arrives with the same provenance as every score: sealed, hashed, and replayable.

The control that makes the number worth reading is ``replay_creativity``.  A sealed creativity
block is a claim about a specific population, and the population is sitting in the same receipt,
so the claim is checkable without re-running anything:

1. the block's own seal must still hold;
2. the declared measurement parameters must be the declared ones -- widening the tolerance
   merges every behaviour into one and would otherwise let a receipt be tuned into looking
   focused, or narrowed until float noise reads as invention;
3. recomputing the whole block from the receipt's own ``sealed_programs`` must reproduce it
   field for field.

Any of the three failing is a replay failure, and ``funsearch_loop.replay_from_receipt`` reports
it alongside the per-program score replay.

``compare_receipts`` is the A/B: two receipts in, one sealed report out, per run label, saying
which way each number moved and which direction is an improvement.  That is the question a
change to the proposer actually has to answer -- did the search try more different things, or
did it just find more ways to write down what it already knew.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .creativity_measure import (
    DEFAULT_TOLERANCE,
    NOVELTY_FLOOR,
    compare,
    measure_creativity,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-creativity-receipt-1.0"
COMPARISON_SCHEMA = "invariant-creativity-comparison-1.0"

#: Only programs the search produced are measured.  A seed is the operator's work and a planted
#: probe is a control; counting either as creativity would flatter or punish the proposer for
#: something it did not do.
MEASURED_ORIGIN = "proposed"

#: The measurement parameters a sealed block must declare, taken from the measure itself so the
#: two cannot drift apart.  ``replay_creativity`` refuses a block that declares anything else:
#: a receipt may not choose the tolerance that makes its own numbers look good.
DECLARED: dict[str, Any] = dict(
    measure_creativity(
        (), tolerance=DEFAULT_TOLERANCE, novelty_floor=NOVELTY_FLOOR, origin=MEASURED_ORIGIN
    )["declared"]
)


class CreativityReceiptError(ValueError):
    """A receipt that cannot carry a creativity claim, or carries one that does not hold."""


# ---------------------------------------------------------------------------
# 1. Sealing
# ---------------------------------------------------------------------------


def creativity_block(sealed_programs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure one run's sealed programs and seal the result.

    ``programs_without_a_behaviour`` is the gap between what was offered and what could be
    measured: a proposal that never executed has no output vector, so it is not a behaviour and
    must not be counted as one.  Reporting the gap rather than hiding it keeps the headline
    honest when a generation is mostly sandbox failures.
    """

    measure = measure_creativity(
        sealed_programs,
        tolerance=DEFAULT_TOLERANCE,
        novelty_floor=NOVELTY_FLOOR,
        origin=MEASURED_ORIGIN,
    )
    counted = sorted(
        str(program.get("program_sha256", ""))
        for program in sealed_programs
        if program.get("origin") == MEASURED_ORIGIN
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "measured_origin": MEASURED_ORIGIN,
        "programs_offered": len(counted),
        "programs_without_a_behaviour": len(counted) - int(measure["population"]["programs"]),
        "measured_population_sha256": canonical_sha256(counted),
        "measure": measure,
        "recomputation": (
            "every number here is a function of this block's own sealed_programs list and "
            "nothing else; replay_creativity recomputes it and a receipt that disagrees with "
            "its own programs fails replay"
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def headline_numbers(block: Mapping[str, Any]) -> dict[str, Any]:
    """The four numbers a reader wants, lifted out of a sealed block."""

    measure = block.get("measure") or {}
    population = measure.get("population") or {}
    return {
        "effective_novel_behaviours": measure.get("effective_novel_behaviours"),
        "effective_behaviours": measure.get("effective_behaviours"),
        "wasted_variation_ratio": measure.get("wasted_variation_ratio"),
        "known_collapse_fraction": measure.get("known_collapse_fraction"),
        "behavioural_span_median": measure.get("behavioural_span_median"),
        "distinct_sources": population.get("distinct_sources"),
        "distinct_behaviours": population.get("distinct_behaviours"),
        "programs_measured": population.get("programs"),
    }


# ---------------------------------------------------------------------------
# 2. The control: a block must agree with the programs it was computed from
# ---------------------------------------------------------------------------


def _differences(
    sealed: Mapping[str, Any], recomputed: Mapping[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    """Every leaf where two blocks disagree, named by its path."""

    faults: list[dict[str, Any]] = []
    for key in sorted(set(sealed) | set(recomputed)):
        path = f"{prefix}{key}"
        if key not in sealed:
            faults.append({"field": path, "sealed": None, "recomputed": recomputed[key]})
            continue
        if key not in recomputed:
            faults.append({"field": path, "sealed": sealed[key], "recomputed": None})
            continue
        left, right = sealed[key], recomputed[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            faults.extend(_differences(left, right, f"{path}."))
        elif left != right:
            faults.append({"field": path, "sealed": left, "recomputed": right})
    return faults


def replay_one_block(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Faults found in one problem block's creativity claim.  Empty means it holds."""

    label = str(block.get("run_label", "unlabelled"))
    sealed = block.get("creativity")
    if not isinstance(sealed, Mapping):
        return [
            {
                "run_label": label,
                "fault": "creativity_block_absent",
                "detail": "the run sealed programs but never measured its behavioural diversity",
            }
        ]

    faults: list[dict[str, Any]] = []
    body = {key: value for key, value in sealed.items() if key != "content_sha256"}
    if sealed.get("content_sha256") != canonical_sha256(body):
        faults.append(
            {
                "run_label": label,
                "fault": "block_seal_broken",
                "detail": "the creativity block does not hash to its own content_sha256",
            }
        )

    declared = (sealed.get("measure") or {}).get("declared")
    if declared != DECLARED:
        faults.append(
            {
                "run_label": label,
                "fault": "declared_measurement_parameters_changed",
                "sealed": declared,
                "declared": DECLARED,
            }
        )

    recomputed = creativity_block(block.get("sealed_programs") or ())
    recomputed_body = {
        key: value for key, value in recomputed.items() if key != "content_sha256"
    }
    for difference in _differences(body, recomputed_body):
        faults.append({"run_label": label, "fault": "recomputation_disagrees", **difference})
    return faults


def replay_creativity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every sealed creativity block from the receipt's own sealed programs."""

    blocks = receipt.get("problems")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise CreativityReceiptError("receipt carries no problems list")
    faults: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for block in blocks:
        faults.extend(replay_one_block(block))
        sealed = block.get("creativity")
        if isinstance(sealed, Mapping):
            rows.append(
                {"run_label": block.get("run_label", "unlabelled"), **headline_numbers(sealed)}
            )
    return {
        "schema_version": SCHEMA,
        "blocks_checked": len(blocks),
        "mismatches": faults,
        "identical": not faults,
        "rows": rows,
        "checks": [
            "the creativity block hashes to its own content_sha256",
            "the declared tolerance, novelty floor and origin filter are the declared ones",
            "recomputing the block from this receipt's sealed_programs reproduces every field",
        ],
    }


# ---------------------------------------------------------------------------
# 3. Reading a receipt
# ---------------------------------------------------------------------------


def receipt_creativity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Every run label's creativity block, sealed if present and recomputed if not.

    A receipt written before the measure was wired in has the programs but no block.  Rather
    than refuse it, the provenance of each measurement is stated: ``sealed_in_the_receipt`` is a
    claim the campaign made about itself, ``recomputed_from_sealed_programs`` is one this module
    made just now.  Both are checkable; only the first is a promise.
    """

    blocks = receipt.get("problems")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise CreativityReceiptError("receipt carries no problems list")
    labels: dict[str, Any] = {}
    for block in blocks:
        label = str(block.get("run_label", "unlabelled"))
        sealed = block.get("creativity")
        if isinstance(sealed, Mapping):
            provenance = "sealed_in_the_receipt"
        else:
            sealed = creativity_block(block.get("sealed_programs") or ())
            provenance = "recomputed_from_sealed_programs"
        labels[label] = {
            "provenance": provenance,
            "measure": sealed.get("measure") or {},
            "headline": headline_numbers(sealed),
        }
    return {
        "schema_version": SCHEMA,
        "content_sha256": receipt.get("content_sha256"),
        "run_labels": labels,
        "totals": {
            "run_labels": len(labels),
            # A plain sum over independent problems: behaviours on different problems are not
            # comparable, so this counts them, it does not pool them.
            "programs_measured": sum(
                int(item["headline"]["programs_measured"] or 0) for item in labels.values()
            ),
            "distinct_behaviours": sum(
                int(item["headline"]["distinct_behaviours"] or 0) for item in labels.values()
            ),
        },
    }


# ---------------------------------------------------------------------------
# 4. The A/B
# ---------------------------------------------------------------------------


def compare_receipts(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_label: str = "before",
    after_label: str = "after",
) -> dict[str, Any]:
    """A/B two campaigns' behavioural diversity, run label by run label."""

    left = receipt_creativity(before)
    right = receipt_creativity(after)
    shared = sorted(set(left["run_labels"]) & set(right["run_labels"]))
    rows: dict[str, Any] = {}
    tally = {"better": 0, "worse": 0, "unchanged": 0}
    for label in shared:
        report = compare(
            left["run_labels"][label]["measure"], right["run_labels"][label]["measure"]
        )
        tally[report["verdict"]] = tally.get(report["verdict"], 0) + 1
        rows[label] = {
            "before_provenance": left["run_labels"][label]["provenance"],
            "after_provenance": right["run_labels"][label]["provenance"],
            "before": left["run_labels"][label]["headline"],
            "after": right["run_labels"][label]["headline"],
            "comparison": report,
        }
    if not shared:
        verdict = "no_shared_run_labels"
    elif tally["better"] > tally["worse"]:
        verdict = "better"
    elif tally["worse"] > tally["better"]:
        verdict = "worse"
    else:
        verdict = "unchanged"
    payload: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "before": {
            "label": before_label,
            "content_sha256": before.get("content_sha256"),
            "proposer": (before.get("proposer") or {}).get("used"),
        },
        "after": {
            "label": after_label,
            "content_sha256": after.get("content_sha256"),
            "proposer": (after.get("proposer") or {}).get("used"),
        },
        "run_labels_compared": shared,
        "only_in_before": sorted(set(left["run_labels"]) - set(right["run_labels"])),
        "only_in_after": sorted(set(right["run_labels"]) - set(left["run_labels"])),
        "rows": rows,
        "tally_on_the_headline": tally,
        "verdict": verdict,
        "rule": (
            "the headline is effective_novel_behaviours and it must RISE for a change to be "
            "worth keeping: more genuinely different things tried that are not already-known "
            "answers. known_collapse_fraction is reported and never scored, because "
            "rediscovering a known law is a real success for a blind search"
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CreativityReceiptError(f"{path} is not a receipt object")
    return value


def _emit(payload: Mapping[str, Any], output: str | Path | None) -> None:
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sigma-creativity",
        description=(
            "Behavioural creativity of a sealed campaign receipt: read it, A/B two of them, "
            "or replay a receipt's creativity claim against its own sealed programs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="Report one receipt's behavioural diversity")
    show.add_argument("receipt", type=Path)
    show.add_argument("--output", type=Path)

    ab = sub.add_parser("compare", help="A/B two receipts, run label by run label")
    ab.add_argument("--before", type=Path, required=True)
    ab.add_argument("--after", type=Path, required=True)
    ab.add_argument("--before-label", default="before")
    ab.add_argument("--after-label", default="after")
    ab.add_argument("--output", type=Path)

    replay = sub.add_parser(
        "replay", help="Recompute a receipt's creativity blocks from its own sealed programs"
    )
    replay.add_argument("receipt", type=Path)
    replay.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            _emit(receipt_creativity(_load(args.receipt)), args.output)
            return 0
        if args.command == "compare":
            report = compare_receipts(
                _load(args.before),
                _load(args.after),
                before_label=args.before_label,
                after_label=args.after_label,
            )
            _emit(report, args.output)
            return 0 if report["verdict"] != "no_shared_run_labels" else 2
        report = replay_creativity(_load(args.receipt))
        _emit(report, args.output)
        return 0 if report["identical"] else 1
    except CreativityReceiptError as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


__all__ = [
    "COMPARISON_SCHEMA",
    "DECLARED",
    "MEASURED_ORIGIN",
    "SCHEMA",
    "CreativityReceiptError",
    "compare_receipts",
    "creativity_block",
    "headline_numbers",
    "main",
    "receipt_creativity",
    "replay_creativity",
    "replay_one_block",
]


if __name__ == "__main__":
    raise SystemExit(main())
