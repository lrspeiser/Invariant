from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .knowledge import pareto_fronts

HEADER = struct.Struct("<8sHHQQQQ")
RECORD = struct.Struct("<QBBH6H")
MAGIC = b"SGSURV2\0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_blocks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [block for block in manifest["blocks"] if block.get("survivor_export")]


def iter_survivors(
    manifest_path: str | Path, survivor_directory: str | Path | None = None
) -> Iterator[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory = Path(survivor_directory or manifest["survivor_export_directory"])
    if not directory.is_absolute():
        directory = (manifest_path.parent / directory).resolve()
        if not directory.is_dir():
            directory = Path(survivor_directory or manifest["survivor_export_directory"]).resolve()
    for block in _export_blocks(manifest):
        export = block["survivor_export"]
        path = directory / export["file"]
        with path.open("rb") as handle:
            header = handle.read(HEADER.size)
            if len(header) != HEADER.size:
                raise ValueError(f"Truncated survivor header: {path}")
            magic, version, record_size, block_index, start, end, count = HEADER.unpack(header)
            if (magic, version, record_size) != (MAGIC, 1, RECORD.size):
                raise ValueError(f"Unsupported survivor format: {path}")
            if (block_index, start, end, count) != (
                block["block_index"],
                block["start_ordinal"],
                block["end_ordinal_exclusive"],
                export["record_count"],
            ):
                raise ValueError(f"Survivor block metadata mismatch: {path}")
            previous = start - 1 if start else -1
            for _ in range(count):
                payload = handle.read(RECORD.size)
                if len(payload) != RECORD.size:
                    raise ValueError(f"Truncated survivor record: {path}")
                ordinal, term_count, sign_mask, reserved, *term_ids = RECORD.unpack(payload)
                if reserved != 0 or not 1 <= term_count <= 6:
                    raise ValueError(f"Invalid survivor record: {path}")
                if ordinal <= previous or not start <= ordinal < end:
                    raise ValueError(f"Non-monotone survivor ordinal: {path}")
                previous = ordinal
                active_ids = term_ids[:term_count]
                if any(value == 0xFFFF for value in active_ids) or any(
                    value != 0xFFFF for value in term_ids[term_count:]
                ):
                    raise ValueError(f"Invalid survivor term padding: {path}")
                yield {
                    "ordinal": ordinal,
                    "term_ids": active_ids,
                    "sign_mask": sign_mask,
                }
            if handle.read(1):
                raise ValueError(f"Trailing bytes in survivor block: {path}")


def iter_dense_pass_survivors(
    dense_report_path: str | Path,
    survivor_directory: str | Path,
    status_directory: str | Path,
) -> Iterator[dict[str, Any]]:
    report = json.loads(Path(dense_report_path).read_text(encoding="utf-8"))
    survivor_directory = Path(survivor_directory)
    status_directory = Path(status_directory)
    dtype = np.dtype(
        [
            ("ordinal", "<u8"),
            ("term_count", "u1"),
            ("sign_mask", "u1"),
            ("reserved", "<u2"),
            ("term_ids", "<u2", (6,)),
        ]
    )
    for block in report["blocks"]:
        records = np.fromfile(
            survivor_directory / block["source_survivor_file"],
            dtype=dtype,
            offset=HEADER.size,
            count=block["records"],
        )
        statuses = np.fromfile(
            status_directory / block["status_file"],
            dtype=np.uint8,
            offset=60,
            count=block["records"],
        )
        if len(records) != block["records"] or len(statuses) != block["records"]:
            raise ValueError(f"Dense block length mismatch: {block['block_index']}")
        for record in records[statuses == 1]:
            term_count = int(record["term_count"])
            yield {
                "ordinal": int(record["ordinal"]),
                "term_ids": [int(value) for value in record["term_ids"][:term_count]],
                "sign_mask": int(record["sign_mask"]),
            }


def audit_survivor_export(
    manifest_path: str | Path,
    output: str | Path,
    survivor_directory: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory = Path(survivor_directory or manifest["survivor_export_directory"])
    if not directory.is_absolute() and not directory.is_dir():
        directory = (manifest_path.parent / directory).resolve()
    errors: list[str] = []
    total_records = 0
    total_bytes = 0
    for block in _export_blocks(manifest):
        export = block["survivor_export"]
        path = directory / export["file"]
        if not path.is_file():
            errors.append(f"missing:{export['file']}")
            continue
        total_records += export["record_count"]
        total_bytes += path.stat().st_size
        if path.stat().st_size != export["file_size_bytes"]:
            errors.append(f"size:{export['file']}")
        if _sha256(path) != export["file_sha256"]:
            errors.append(f"sha256:{export['file']}")
    try:
        iterated = sum(1 for _ in iter_survivors(manifest_path, directory))
    except ValueError as error:
        errors.append(str(error))
        iterated = -1
    if total_records != manifest["survivor_count"]:
        errors.append("manifest_survivor_count")
    if iterated != total_records:
        errors.append("iterated_record_count")
    report = {
        "schema_version": "sigma-survivor-audit-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "survivor_directory": str(directory),
        "block_count": len(_export_blocks(manifest)),
        "record_count": total_records,
        "iterated_record_count": iterated,
        "file_bytes": total_bytes,
        "all_checks_pass": not errors,
        "errors": errors,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _candidate_tags(terms: list[dict[str, Any]]) -> list[str]:
    tags = []
    if any(term["px"] for term in terms):
        tags.append("flux")
    if any(term["pq"] for term in terms):
        tags.append("gradient_state")
    if any(term["pz"] for term in terms):
        tags.append("measured_state")
    if any(term["transform"] != "Identity" for term in terms):
        tags.append("saturation")
    return tags


def _family_signature(terms: list[dict[str, Any]], sign_mask: int) -> tuple[Any, ...]:
    transforms = Counter(term["transform"] for term in terms)
    sectors = (
        sum(bool(term["px"]) for term in terms),
        sum(bool(term["pq"]) for term in terms),
        sum(bool(term["pz"]) for term in terms),
        sum(sum(bool(term[key]) for key in ("px", "pq", "pz")) > 1 for term in terms),
    )
    dangerous = sum(
        term["high_field_growth_numerator"] >= term["high_field_growth_denominator"]
        for term in terms
    )
    return (
        len(terms),
        *sectors,
        transforms["Identity"],
        transforms["Sqrt1pMinus1"],
        transforms["Saturate"],
        dangerous,
        sign_mask.bit_count(),
        max(term["px"] + term["pq"] + term["pz"] for term in terms),
    )


def _complexity(terms: list[dict[str, Any]]) -> int:
    transform_cost = {"Identity": 0, "Sqrt1pMinus1": 2, "Saturate": 3}
    return 100 * len(terms) + sum(
        10 * (term["px"] + term["pq"] + term["pz"]) + transform_cost[term["transform"]]
        for term in terms
    )


def _expression(terms: list[dict[str, Any]], sign_mask: int) -> str:
    return "".join(
        f"{'+' if sign_mask & (1 << position) else '-'}({term['expression']})"
        for position, term in enumerate(terms)
    )


def prioritize_generated_survivors(
    manifest_path: str | Path,
    basis_path: str | Path,
    database: str | Path,
    output: str | Path,
    survivor_directory: str | Path | None = None,
    max_fronts: int = 8,
    dense_report_path: str | Path | None = None,
    dense_status_directory: str | Path | None = None,
) -> dict[str, Any]:
    basis = json.loads(Path(basis_path).read_text(encoding="utf-8"))
    by_id = {term["id"]: term for term in basis}
    connection = sqlite3.connect(database)
    try:
        lessons = connection.execute(
            "SELECT inferred_outcome, tags_json FROM lessons WHERE admissible_for_priority=1"
        ).fetchall()
    finally:
        connection.close()
    history: dict[str, Counter[str]] = defaultdict(Counter)
    for outcome, tags_json in lessons:
        for tag in json.loads(tags_json):
            history[tag][outcome] += 1

    families: dict[tuple[Any, ...], dict[str, Any]] = {}
    survivor_count = 0
    if dense_report_path:
        if survivor_directory is None or dense_status_directory is None:
            raise ValueError("Dense prioritization requires survivor and status directories")
        record_iterator = iter_dense_pass_survivors(
            dense_report_path, survivor_directory, dense_status_directory
        )
    else:
        record_iterator = iter_survivors(manifest_path, survivor_directory)
    for record in record_iterator:
        survivor_count += 1
        terms = [by_id[term_id] for term_id in record["term_ids"]]
        signature = _family_signature(terms, record["sign_mask"])
        complexity = _complexity(terms)
        family = families.get(signature)
        if family is None:
            family = {
                "signature": signature,
                "count": 0,
                "representative": None,
                "representative_key": None,
            }
            families[signature] = family
        family["count"] += 1
        key = (complexity, record["ordinal"])
        if family["representative_key"] is None or key < family["representative_key"]:
            family["representative_key"] = key
            family["representative"] = {**record, "complexity": complexity}

    rows = []
    for signature, family in sorted(families.items()):
        representative = family["representative"]
        terms = [by_id[term_id] for term_id in representative["term_ids"]]
        tags = _candidate_tags(terms)
        tag_history = Counter()
        for tag in tags:
            tag_history.update(history[tag])
        dangerous = signature[8]
        sector_count = sum(any(term[key] for term in terms) for key in ("px", "pq", "pz"))
        pass_count = tag_history["pass"]
        reject_count = tag_history["reject"]
        family_id = "GF-" + hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
        rows.append(
            {
                "family_id": family_id,
                "family_survivor_count": family["count"],
                "ordinal": representative["ordinal"],
                "term_ids": representative["term_ids"],
                "sign_mask": representative["sign_mask"],
                "correction_expression": _expression(terms, representative["sign_mask"]),
                "mechanism_tags": tags,
                "complexity": representative["complexity"],
                "parsimony": 1.0 / representative["complexity"],
                "sector_coverage": sector_count / 3.0,
                "high_field_robustness": 1.0 / (1.0 + dangerous),
                "theory_history_signal": (pass_count + 1.0) / (pass_count + reject_count + 2.0),
                "theory_history_passes": pass_count,
                "theory_history_rejections": reject_count,
                "theory_history_coverage": pass_count + reject_count,
                "priority_semantics": "work ordering only; not probability of truth",
            }
        )

    controls = [row for row in rows if len(row["term_ids"]) == 1]
    discovery_rows = [row for row in rows if len(row["term_ids"]) > 1]
    axes = [
        "parsimony",
        "sector_coverage",
        "high_field_robustness",
        "theory_history_signal",
    ]
    fronts = pareto_fronts(discovery_rows, axes)
    queue = []
    for index, front in enumerate(fronts, start=1):
        for row in front:
            row["pareto_front"] = index
            if index <= max_fronts:
                queue.append(row)
    report = {
        "schema_version": "sigma-generated-priority-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "basis": str(basis_path),
        "knowledge_database": str(database),
        "survivor_count": survivor_count,
        "dense_static_report": str(dense_report_path) if dense_report_path else None,
        "family_count": len(rows),
        "discovery_family_count": len(discovery_rows),
        "control_family_count": len(controls),
        "pareto_front_count": len(fronts),
        "included_front_count": min(max_fronts, len(fronts)),
        "axes": axes,
        "work_queue": queue,
        "front_one": fronts[0] if fronts else [],
        "control_representatives": controls,
        "scientific_limits": [
            "Every record passed only the Generator v2 structural and sampled-static gates.",
            "Historical evidence orders work but cannot rescue any hard-gate failure.",
            "No dark-matter, redshift-distance, supernova-distance, or derived GR/NFW target is used.",
            "Covariant variation, constraints, degrees of freedom, characteristics, GR, Solar, and audited measurement gates remain required.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
