"""Exact verifier for finite counterexamples to the MathOverflow Frankl-family claim."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

SCHEMA = "invariant-frankl-counterexample-verifier-1.0"


class FranklVerifierError(ValueError):
    """The proposed finite family or its receipt is invalid."""


def canonical_family(raw_family: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw_family, Sequence) or isinstance(raw_family, (str, bytes)):
        raise FranklVerifierError("family must be a sequence")
    family: list[tuple[int, ...]] = []
    for raw_set in raw_family:
        if not isinstance(raw_set, Sequence) or isinstance(raw_set, (str, bytes)):
            raise FranklVerifierError("each member must be a sequence")
        if not raw_set or any(isinstance(x, bool) or not isinstance(x, int) for x in raw_set):
            raise FranklVerifierError("members must be nonempty integer sets")
        member = tuple(sorted(set(raw_set)))
        if len(member) != len(raw_set):
            raise FranklVerifierError("member contains duplicate elements")
        family.append(member)
    normalized = tuple(sorted(set(family), key=lambda row: (len(row), row)))
    if not normalized or len(normalized) != len(family):
        raise FranklVerifierError("family is empty or contains duplicate sets")
    return normalized


def verify_family(raw_family: Sequence[Sequence[int]]) -> dict[str, Any]:
    family = canonical_family(raw_family)
    members = {frozenset(row) for row in family}
    universe = sorted(set().union(*members))
    missing_unions = sorted(
        {
            tuple(sorted(left | right))
            for left in members
            for right in members
            if left | right not in members
        },
        key=lambda row: (len(row), row),
    )
    degrees = {str(x): sum(x in member for member in members) for x in universe}
    delta = max(degrees.values())
    residual_delta: dict[str, int] = {}
    residual_witness: dict[str, int | None] = {}
    for omitted in universe:
        residual = [member for member in members if omitted not in member]
        residual_degrees = {
            x: sum(x in member for member in residual) for x in universe if x != omitted
        }
        value = max(residual_degrees.values(), default=0)
        residual_delta[str(omitted)] = value
        residual_witness[str(omitted)] = min(
            (x for x, degree in residual_degrees.items() if degree == value),
            default=None,
        )
    union_closed = not missing_unions
    violates_for_every_element = all(2 * value > delta for value in residual_delta.values())
    body = {
        "schema_version": SCHEMA,
        "family": [list(row) for row in family],
        "canonical_family_sha256": canonical_sha256([list(row) for row in family]),
        "family_size": len(family),
        "universe": universe,
        "degrees": degrees,
        "delta": delta,
        "residual_delta": residual_delta,
        "residual_witness": residual_witness,
        "missing_unions": [list(row) for row in missing_unions],
        "union_closed": union_closed,
        "strict_violation_for_every_element": violates_for_every_element,
        "exact_counterexample_valid": union_closed and violates_for_every_element,
        "arithmetic_rule": "for every x, verify 2 * residual_delta[x] > delta",
        "claims": {"historical_novelty_established": False},
    }
    result = dict(body)
    result["content_sha256"] = canonical_sha256(body)
    return result


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != SCHEMA:
        raise FranklVerifierError("receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise FranklVerifierError("receipt seal changed")
    replay = verify_family(receipt["family"])
    if replay != receipt:
        raise FranklVerifierError("receipt replay changed")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise FranklVerifierError("input JSON must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = _read_json(args.input)
    receipt = verify_family(source["family"])
    validate_receipt(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "exact_counterexample_valid": receipt["exact_counterexample_valid"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
