"""A2 — hash-bound open-problem intake queue.

Continuous discovery needs a supply of targets, and the supply is where honesty is
easiest to lose: a system that manufactures its own "open problems" can afterwards
claim credit for solving them.  This module fixes the intake boundary.  The queue is
a sealed registry of declared targets; every entry cites a source, states in prose
why it is believed open, and defines in advance what counts as progress.

Three rules keep the registry honest.

**"Believed open" is a documented claim, never an inference.**  The
`believed_open_because` field must point at the cited literature — or, for internal
and synthetic targets, say plainly that the entry is *not* open mathematics.  The
system never promotes corpus absence, or its own failure to solve, into openness.

**Controls and synthetics are labeled in the schema, not in prose.**  A rediscovery
control (`control_rediscovery`) or a sealed synthetic world (`synthetic`) carries its
label as a validated boolean, so a calibration entry can never be reported as a
discovery by dropping a sentence.

**The queue is sealed.**  `content_sha256` binds the entries under canonical JSON;
editing a citation, a flag, or a progress definition without resealing is detected,
and the stored file must itself be canonically encoded byte for byte.  Floats are
forbidden everywhere, as in every Sigma receipt.

Claim boundary: queue membership asserts provenance, not importance, tractability,
or solvability.  A validated queue means the *claims about the problems* are
well-formed and sealed; it proves nothing about the problems themselves.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import SigmaCoreError, canonical_json_bytes, canonical_sha256

QUEUE_SCHEMA = "invariant-problem-queue-1.0"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_DOMAIN = re.compile(r"^(math|physics)(/[a-z][a-z0-9_]*)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

TOP_LEVEL_KEYS = {"schema_version", "entries", "content_sha256"}
ENTRY_KEYS = {
    "id",
    "domain",
    "statement",
    "source_citation",
    "believed_open_because",
    "machine_form",
    "progress_definition",
    "control_rediscovery",
    "synthetic",
}

#: Declared machine-form kinds.  Field values are strings or ints only — a machine
#: form is an address for a harness, never a place to smuggle measurements.
MACHINE_FORM_KINDS: dict[str, dict[str, type]] = {
    "sequence_rows": {"generator": str, "max_point": int},
    "integer_trajectory": {"map": str, "seed": int, "max_steps": int},
    "diophantine_family": {"equation": str, "parameter": str, "parameter_min": int},
    "dataset_law_fit": {"dataset": str, "target_relation": str},
    "module_target": {"proof_module": str, "decomposition_module": str},
}

#: Hard bounds.  Exceeding any of these is an error, never a silent truncation.
SYSTEM_CAPS = {
    "max_entries": 64,
    "max_text_chars": 1200,
    "max_machine_form_int": 10**9,
}

CLAIMS = {
    "believed_open_is_documented_with_citation": True,
    "believed_open_may_be_inferred_by_the_system": False,
    "control_and_synthetic_entries_are_schema_labeled": True,
    "entry_without_citation_admissible": False,
    "floats_admissible_anywhere": False,
    "queue_membership_establishes_solvability": False,
}


class ProblemQueueError(ValueError):
    """Raised when the queue file, schema, flags, or seal are violated."""


# ---------------------------------------------------------------------------
# Fail-closed field validation
# ---------------------------------------------------------------------------


def _reject_floats(value: Any, path: str = "$") -> None:
    """Reject floats anywhere.  Exactness is a schema property, not a convention."""

    if isinstance(value, float):
        raise ProblemQueueError(f"floating value forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_floats(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_floats(child, f"{path}[{index}]")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ProblemQueueError(f"{label} must be a nonempty stripped string")
    if len(value) > SYSTEM_CAPS["max_text_chars"]:
        raise ProblemQueueError(f"{label} exceeds text cap")
    return value


def _flag(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProblemQueueError(f"{label} must be a boolean")
    return value


def _validate_machine_form(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ProblemQueueError(f"{label} must be an object")
    kind = value.get("kind")
    if kind not in MACHINE_FORM_KINDS:
        raise ProblemQueueError(f"{label}.kind must be one of the declared machine-form kinds")
    fields = MACHINE_FORM_KINDS[kind]
    if set(value) != {"kind"} | set(fields):
        raise ProblemQueueError(f"{label} keys changed for kind {kind}")
    for name, expected in fields.items():
        item = value[name]
        if expected is str:
            _text(item, f"{label}.{name}")
        else:
            if not isinstance(item, int) or isinstance(item, bool):
                raise ProblemQueueError(f"{label}.{name} must be an integer")
            if not 0 <= item <= SYSTEM_CAPS["max_machine_form_int"]:
                raise ProblemQueueError(f"{label}.{name} exceeds integer cap")


def _validate_entry(value: Any, label: str) -> str:
    """Validate one entry and return its id."""

    if not isinstance(value, Mapping):
        raise ProblemQueueError(f"{label} must be an object")
    if set(value) != ENTRY_KEYS:
        raise ProblemQueueError(f"{label} keys changed")
    entry_id = value["id"]
    if not isinstance(entry_id, str) or _IDENTIFIER.fullmatch(entry_id) is None:
        raise ProblemQueueError(f"{label}.id must match {_IDENTIFIER.pattern}")
    domain = value["domain"]
    if not isinstance(domain, str) or _DOMAIN.fullmatch(domain) is None:
        raise ProblemQueueError(f"{label}.domain must match {_DOMAIN.pattern}")
    _text(value["statement"], f"{label}.statement")
    _text(value["source_citation"], f"{label}.source_citation")
    _text(value["believed_open_because"], f"{label}.believed_open_because")
    _text(value["progress_definition"], f"{label}.progress_definition")
    _validate_machine_form(value["machine_form"], f"{label}.machine_form")
    _flag(value["control_rediscovery"], f"{label}.control_rediscovery")
    _flag(value["synthetic"], f"{label}.synthetic")
    return entry_id


def validate_queue(value: Any) -> None:
    """Reject any structural, flag, citation, or seal violation.  Never repairs."""

    if not isinstance(value, Mapping):
        raise ProblemQueueError("queue must be an object")
    if set(value) != TOP_LEVEL_KEYS:
        raise ProblemQueueError("queue top-level keys changed")
    _reject_floats(value)
    if value["schema_version"] != QUEUE_SCHEMA:
        raise ProblemQueueError("queue schema changed")
    entries = value["entries"]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ProblemQueueError("entries must be a non-empty list")
    if len(entries) > SYSTEM_CAPS["max_entries"]:
        raise ProblemQueueError("entry count exceeds cap")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        entry_id = _validate_entry(entry, f"entries[{index}]")
        if entry_id in seen:
            raise ProblemQueueError(f"duplicate entry id: {entry_id}")
        seen.add(entry_id)
    seal = value["content_sha256"]
    if not isinstance(seal, str) or _SHA256.fullmatch(seal) is None:
        raise ProblemQueueError("content_sha256 must be a lowercase SHA-256 digest")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if seal != _canonical_sha256(body):
        raise ProblemQueueError("queue seal changed")


def _canonical_sha256(value: Any) -> str:
    try:
        return canonical_sha256(value)
    except SigmaCoreError as error:
        raise ProblemQueueError(f"queue is not canonically encodable: {error}") from error


def seal_queue(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build, seal, and validate a queue from entries.  The only sanctioned sealer."""

    body = {
        "entries": [dict(entry) for entry in entries],
        "schema_version": QUEUE_SCHEMA,
    }
    sealed = {**body, "content_sha256": _canonical_sha256(body)}
    validate_queue(sealed)
    return sealed


def load_queue(path: Path | str) -> dict[str, Any]:
    """Load, validate, and return the queue.  The stored bytes must be canonical."""

    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ProblemQueueError(f"unreadable queue file: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProblemQueueError(f"queue file is not valid JSON: {error}") from error
    validate_queue(value)
    if canonical_json_bytes(value) + b"\n" != raw:
        raise ProblemQueueError("queue file is not canonically encoded")
    return value


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def _domain_root(domain: str) -> str:
    return domain.split("/", 1)[0]


def _domain_targets(queue: Mapping[str, Any], root: str) -> tuple[dict[str, Any], ...]:
    validate_queue(queue)
    return tuple(
        dict(entry) for entry in queue["entries"] if _domain_root(entry["domain"]) == root
    )


def iter_math_targets(queue: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Every math-domain entry, in queue order.  Validates before selecting."""

    return _domain_targets(queue, "math")


def iter_physics_targets(queue: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Every physics-domain entry, in queue order.  Validates before selecting."""

    return _domain_targets(queue, "physics")


def summarize_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic summary of a validated queue for the CLI and dashboards."""

    validate_queue(queue)
    entries = queue["entries"]
    return {
        "claims": CLAIMS,
        "content_sha256": queue["content_sha256"],
        "counts": {
            "control_rediscovery": sum(1 for e in entries if e["control_rediscovery"]),
            "entries": len(entries),
            "math": sum(1 for e in entries if _domain_root(e["domain"]) == "math"),
            "physics": sum(1 for e in entries if _domain_root(e["domain"]) == "physics"),
            "synthetic": sum(1 for e in entries if e["synthetic"]),
        },
        "entry_ids": [entry["id"] for entry in entries],
        "schema_version": queue["schema_version"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ProblemQueueError("refusing to overwrite immutable queue file")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hash-bound open-problem intake queue (A2).")
    parser.add_argument("--queue", required=True, help="path to the sealed queue JSON")
    parser.add_argument(
        "--validate", action="store_true", help="validate only; exit 0 when the queue is valid"
    )
    args = parser.parse_args(argv)
    try:
        queue = load_queue(Path(args.queue))
    except ProblemQueueError as error:
        print(f"INVALID {args.queue}: {error}")
        return 1
    if args.validate:
        print(f"VALID entries={len(queue['entries'])} content_sha256={queue['content_sha256']}")
    else:
        print(json.dumps(summarize_queue(queue), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
