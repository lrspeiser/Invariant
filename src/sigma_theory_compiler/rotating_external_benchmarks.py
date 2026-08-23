"""Build rotating, anonymized benchmark packets from live external OEIS b-files.

The generator packet contains only anonymous indexed training terms, a representation family, and
a target commitment.  The coordinator packet binds the external response bytes and holdout packet.
The current OEIS HTTPS origin is not a cryptographic source signature, so these tasks are useful
creativity benchmarks but remain ineligible for a level-5 count until a distinct principal signs a
pack manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .claim_specific_prior_art import Transport, urllib_transport
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/rotating_external_benchmark_pack.json"
CONFIG_SCHEMA = "invariant-rotating-external-benchmark-config-1.0"
GENERATION_SCHEMA = "invariant-rotating-external-generation-packet-1.0"
TARGET_SCHEMA = "invariant-rotating-external-target-packet-1.0"
RECEIPT_SCHEMA = "invariant-rotating-external-coordinator-receipt-1.0"
_HEX = frozenset("0123456789abcdef")
_BFILE_LINE = re.compile(r"(-?[0-9]+) +(-?[0-9]+)")
_REPRESENTATIONS = {
    "finite_product",
    "finite_sum",
    "generating_function",
    "modular_object",
    "recurrence",
    "representation_bridge",
}


class RotationError(ValueError):
    """The external source, rotation, commitment, or blind packet failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RotationError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise RotationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _normalized_file_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _utc(value: str | None) -> str:
    value = value or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RotationError("rotation retrieval time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RotationError("rotation retrieval time lacks a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _strict(
        value,
        {
            "generator_principal_id",
            "pack_id",
            "rotation_epoch",
            "schema_version",
            "source_policy",
            "sources",
            "task_policy",
        },
        "rotation config",
    )
    if value["schema_version"] != CONFIG_SCHEMA or value["pack_id"] != "oeis-live-rotation":
        raise RotationError("rotation config identity changed")
    if value["generator_principal_id"] != "invariant.discovery-engine":
        raise RotationError("rotation generator principal changed")
    if re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{3}", value["rotation_epoch"]) is None:
        raise RotationError("rotation epoch is not explicit")
    task = value["task_policy"]
    _strict(
        task,
        {"holdout_terms", "maximum_window_start", "minimum_tasks", "training_terms"},
        "rotation task policy",
    )
    if (
        task["minimum_tasks"] < 24
        or task["training_terms"] < 12
        or task["holdout_terms"] < 6
        or task["maximum_window_start"] < 8
    ):
        raise RotationError("rotation task coverage is too weak")
    source_policy = value["source_policy"]
    _strict(
        source_policy,
        {
            "allowed_host",
            "cryptographic_signature_required_for_level5",
            "maximum_response_bytes",
            "request_timeout_seconds",
            "user_agent",
        },
        "rotation source policy",
    )
    if (
        source_policy["allowed_host"] != "oeis.org"
        or source_policy["cryptographic_signature_required_for_level5"] is not True
        or not 100_000 <= source_policy["maximum_response_bytes"] <= 5_000_000
        or not 1 <= source_policy["request_timeout_seconds"] <= 60
    ):
        raise RotationError("rotation source policy weakened")
    sources = value["sources"]
    if not isinstance(sources, list) or len(sources) < task["minimum_tasks"]:
        raise RotationError("rotation source registry is too small")
    source_ids = set()
    for source in sources:
        _strict(
            source,
            {"external_principal_id", "representation_family", "source_id", "source_uri"},
            "rotation source",
        )
        parsed = urllib.parse.urlparse(source["source_uri"])
        expected_tail = f"/{source['source_id'][5:]}/{source['source_id'][5:].lower().replace('a', 'b', 1)}.txt"
        if (
            re.fullmatch(r"OEIS-A[0-9]{6}", source["source_id"]) is None
            or source["source_id"] in source_ids
            or source["external_principal_id"] == value["generator_principal_id"]
            or not source["external_principal_id"].startswith("external.")
            or source["representation_family"] not in _REPRESENTATIONS
            or parsed.scheme != "https"
            or parsed.hostname != source_policy["allowed_host"]
            or parsed.path.lower() != expected_tail.lower()
        ):
            raise RotationError("external rotation source identity changed")
        source_ids.add(source["source_id"])
    if min(Counter(item["representation_family"] for item in sources).values()) < 4:
        raise RotationError("rotation representation families are not balanced")
    return value


def parse_bfile(body: bytes) -> tuple[tuple[int, int], ...]:
    if not body or body.startswith(b"\xef\xbb\xbf") or any(byte > 0x7F for byte in body):
        raise RotationError("OEIS b-file is empty, BOM-prefixed, or non-ASCII")
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as error:
        raise RotationError("OEIS b-file is not ASCII") from error
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _BFILE_LINE.fullmatch(line)
        if match is None:
            raise RotationError("OEIS b-file line violates index-value format")
        rows.append((int(match.group(1)), int(match.group(2))))
    if len(rows) < 26 or any(right[0] != left[0] + 1 for left, right in pairwise(rows)):
        raise RotationError("OEIS b-file is too short or has nonconsecutive indices")
    return tuple(rows)


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def build_pack(
    root: Path,
    *,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config = load_config(root)
    retrieved = _utc(retrieved_utc)
    policy = config["source_policy"]
    task_policy = config["task_policy"]
    headers = {
        "Accept": "text/plain",
        "Range": f"bytes=0-{policy['maximum_response_bytes'] - 1}",
        "User-Agent": policy["user_agent"],
    }
    tasks = []
    targets = []
    source_evidence = []
    for source in config["sources"]:
        response = transport(
            source["source_uri"],
            headers,
            policy["request_timeout_seconds"],
            policy["maximum_response_bytes"],
        )
        if response.status not in {200, 206}:
            raise RotationError(f"external benchmark source unavailable: {source['source_id']}")
        raw_sha = hashlib.sha256(response.body).hexdigest()
        parse_body = response.body
        if response.status == 206 and not parse_body.endswith(b"\n"):
            parse_body = parse_body.rsplit(b"\n", 1)[0] + b"\n"
        try:
            rows = parse_bfile(parse_body)
        except RotationError as error:
            raise RotationError(f"{source['source_id']}: {error}") from error
        required = task_policy["training_terms"] + task_policy["holdout_terms"]
        maximum_start = min(task_policy["maximum_window_start"], len(rows) - required)
        if maximum_start < 0:
            raise RotationError("external source lacks the requested train/holdout window")
        selector = hashlib.sha256(
            f"{config['rotation_epoch']}:{source['source_id']}:{raw_sha}".encode()
        ).digest()
        start = int.from_bytes(selector[:8], "big") % (maximum_start + 1)
        training = rows[start : start + task_policy["training_terms"]]
        holdout = rows[
            start + task_policy["training_terms"] : start + required
        ]
        target_body = {
            "external_principal_id": source["external_principal_id"],
            "holdout": [{"index": index, "value": str(value)} for index, value in holdout],
            "source_id": source["source_id"],
            "source_response_sha256": raw_sha,
            "source_uri": source["source_uri"],
        }
        commitment = canonical_sha256(target_body)
        task_id = f"blind.{canonical_sha256({'commitment': commitment, 'epoch': config['rotation_epoch']})[:24]}"
        tasks.append(
            {
                "external_source_disclosed_to_generator": False,
                "representation_family": source["representation_family"],
                "rotation_epoch": config["rotation_epoch"],
                "target_commitment": commitment,
                "task_id": task_id,
                "training": [{"index": index, "value": str(value)} for index, value in training],
            }
        )
        targets.append({"task_id": task_id, **target_body})
        source_evidence.append(
            {
                "content_type": response.headers.get("content-type", "")[:200],
                "external_principal_id": source["external_principal_id"],
                "http_status": response.status,
                "retrieved_utc": retrieved,
                "source_id": source["source_id"],
                "source_content_range": response.headers.get("content-range"),
                "source_response_bytes": len(response.body),
                "source_response_sha256": raw_sha,
                "source_uri": source["source_uri"],
                "transport": "https",
            }
        )
    generation = _sealed(
        {
            "schema_version": GENERATION_SCHEMA,
            "pack_id": config["pack_id"],
            "rotation_epoch": config["rotation_epoch"],
            "tasks": tasks,
            "claims": {
                "external_source_identity_visible_to_generator": False,
                "holdout_values_visible_to_generator": False,
            },
        }
    )
    target_packet = _sealed(
        {
            "schema_version": TARGET_SCHEMA,
            "pack_id": config["pack_id"],
            "rotation_epoch": config["rotation_epoch"],
            "targets": targets,
        }
    )
    receipt = _sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "pack_id": config["pack_id"],
            "rotation_epoch": config["rotation_epoch"],
            "retrieved_utc": retrieved,
            "source_bindings": {
                "config": {"path": CONFIG_PATH, "sha256": _normalized_file_sha256(root / CONFIG_PATH)},
                "generation_packet_content_sha256": generation["content_sha256"],
                "target_packet_content_sha256": target_packet["content_sha256"],
            },
            "source_evidence": source_evidence,
            "coverage": {
                "representation_counts": dict(
                    sorted(Counter(task["representation_family"] for task in tasks).items())
                ),
                "tasks": len(tasks),
                "unique_external_response_hashes": len(
                    {item["source_response_sha256"] for item in source_evidence}
                ),
            },
            "blind_protocol": {
                "generation_packet_excludes_holdouts": True,
                "generation_packet_excludes_source_identities": True,
                "target_commitments_verified": True,
            },
            "source_signature": {
                "cryptographic_signature_verified": False,
                "external_https_origin_hash_bound": True,
                "status": "PENDING_DISTINCT_PRINCIPAL_SIGNATURE",
            },
            "release_gate": {
                "level5_eligible": False,
                "status": "CREATIVITY_BENCHMARK_READY_LEVEL5_BLOCKED_UNSIGNED_SOURCE",
            },
            "claims": {
                "https_origin_is_cryptographic_source_signature": False,
                "pack_is_level5_success": False,
                "targets_are_literature_novel": False,
            },
        }
    )
    validate_pack(generation, target_packet, receipt, root)
    return generation, target_packet, receipt


def validate_pack(
    generation: Mapping[str, Any],
    targets: Mapping[str, Any],
    receipt: Mapping[str, Any],
    root: Path | None = None,
) -> None:
    for value, schema, label in (
        (generation, GENERATION_SCHEMA, "generation packet"),
        (targets, TARGET_SCHEMA, "target packet"),
        (receipt, RECEIPT_SCHEMA, "coordinator receipt"),
    ):
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("schema_version") != schema or value.get("content_sha256") != canonical_sha256(body):
            raise RotationError(f"{label} identity or content seal changed")
    tasks = generation.get("tasks", [])
    target_rows = targets.get("targets", [])
    if len(tasks) < 24 or len(tasks) != len(target_rows):
        raise RotationError("rotating pack task count changed")
    if len({item["task_id"] for item in tasks}) != len(tasks):
        raise RotationError("rotating generation tasks are not unique")
    target_by_id = {item["task_id"]: item for item in target_rows}
    if len(target_by_id) != len(target_rows):
        raise RotationError("rotating target tasks are not unique")
    for task in tasks:
        if (
            task["external_source_disclosed_to_generator"] is not False
            or "source_id" in task
            or "source_uri" in task
            or task["task_id"] not in target_by_id
        ):
            raise RotationError("generation packet leaked external target identity")
        target = target_by_id[task["task_id"]]
        target_body = {key: value for key, value in target.items() if key != "task_id"}
        if task["target_commitment"] != canonical_sha256(target_body):
            raise RotationError("rotating target commitment changed")
        training_indexes = {row["index"] for row in task["training"]}
        holdout_indexes = {row["index"] for row in target["holdout"]}
        if training_indexes & holdout_indexes:
            raise RotationError("rotating train/holdout split overlaps")
        if len(task["training"]) != 18 or len(target["holdout"]) != 8:
            raise RotationError("rotating train/holdout term counts changed")
    bindings = receipt.get("source_bindings", {})
    source_evidence = receipt.get("source_evidence", [])
    evidence_by_id = {item["source_id"]: item for item in source_evidence}
    if len(evidence_by_id) != len(target_rows) or len(source_evidence) != len(target_rows):
        raise RotationError("external source evidence is missing or duplicated")
    for target in target_rows:
        evidence = evidence_by_id.get(target["source_id"])
        if evidence is None or any(
            target[key] != evidence[key]
            for key in ("external_principal_id", "source_response_sha256", "source_uri")
        ):
            raise RotationError("target packet is not bound to its external source evidence")
    coverage = receipt.get("coverage", {})
    if (
        bindings.get("generation_packet_content_sha256") != generation["content_sha256"]
        or bindings.get("target_packet_content_sha256") != targets["content_sha256"]
        or receipt.get("blind_protocol", {}).get("target_commitments_verified") is not True
        or generation.get("claims")
        != {
            "external_source_identity_visible_to_generator": False,
            "holdout_values_visible_to_generator": False,
        }
        or coverage.get("tasks") != len(tasks)
        or coverage.get("representation_counts")
        != {family: 4 for family in sorted(_REPRESENTATIONS)}
        or coverage.get("unique_external_response_hashes")
        != len({item["source_response_sha256"] for item in source_evidence})
        or receipt.get("source_signature", {}).get("cryptographic_signature_verified") is not False
        or receipt.get("release_gate", {}).get("level5_eligible") is not False
        or any(receipt.get("claims", {}).values())
    ):
        raise RotationError("rotating pack release boundary changed")
    for source in source_evidence:
        _sha(source["source_response_sha256"], "external source response hash")
        parsed = urllib.parse.urlparse(source["source_uri"])
        if parsed.scheme != "https" or parsed.hostname != "oeis.org":
            raise RotationError("external source evidence escaped OEIS HTTPS")
    if root is not None:
        binding = bindings["config"]
        if binding["path"] != CONFIG_PATH or binding["sha256"] != _normalized_file_sha256(
            root / CONFIG_PATH
        ):
            raise RotationError("rotation config source binding changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--generation-output", type=Path, required=True)
    build.add_argument("--target-output", type=Path, required=True)
    build.add_argument("--receipt-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--generation", type=Path, required=True)
    validate.add_argument("--targets", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        generation, targets, receipt = build_pack(args.root)
        for path, value in (
            (args.generation_output, generation),
            (args.target_output, targets),
            (args.receipt_output, receipt),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        generation = json.loads(args.generation.read_text(encoding="utf-8"))
        targets = json.loads(args.targets.read_text(encoding="utf-8"))
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_pack(generation, targets, receipt, args.root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "level5_eligible": receipt["release_gate"]["level5_eligible"],
                "status": receipt["release_gate"]["status"],
                "tasks": receipt["coverage"]["tasks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
