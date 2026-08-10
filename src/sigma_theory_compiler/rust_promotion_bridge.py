from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .high_throughput import build_basis, correction_expression, decode_ordinal
from .high_throughput import candidate_id as generator_candidate_id
from .promotion_orchestrator import (
    ELIGIBILITY,
    EVIDENCE_SCHEMA,
    PromotionOrchestrator,
)
from .rust_streaming_search import PROMOTION_HEADER, PROMOTION_MAGIC, PROMOTION_RECORD
from .rust_streaming_service import EXPORT_SCHEMA

BRIDGE_SCHEMA = "sigma-rust-promotion-bridge-1.0"
BLOCK_SCHEMA = "sigma-promotion-survivor-block-1.0"
SERVICE_ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "paid_llm_calls": False,
    "passed": True,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _portable_path(manifest_path: Path, value: Any) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError("promotion block path must be portable and relative")
    resolved = (manifest_path.parent / relative).resolve()
    try:
        resolved.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("promotion block path escapes export directory") from error
    return resolved


SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS bridge_state (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  service_id TEXT NOT NULL,
  generator_config_sha256 TEXT NOT NULL,
  generator_protocol_version TEXT NOT NULL,
  first_ordinal INTEGER NOT NULL,
  next_block_index INTEGER NOT NULL,
  next_record_index INTEGER NOT NULL,
  next_interval_start INTEGER NOT NULL,
  updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verified_blocks (
  block_index INTEGER PRIMARY KEY,
  start_ordinal INTEGER NOT NULL,
  end_ordinal_exclusive INTEGER NOT NULL,
  block_sha256 TEXT NOT NULL,
  block_lineage_sha256 TEXT NOT NULL,
  record_count INTEGER NOT NULL,
  verified_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consumed_records (
  block_index INTEGER NOT NULL,
  record_index INTEGER NOT NULL,
  ordinal INTEGER NOT NULL UNIQUE,
  sampled_status TEXT NOT NULL CHECK(sampled_status IN ('pass','ambiguous')),
  candidate_id TEXT,
  initial_lineage_sha256 TEXT,
  consumed_utc TEXT NOT NULL,
  PRIMARY KEY(block_index,record_index)
);
"""


class RustPromotionBridge:
    """Restart-safe, fail-closed importer for portable Rust promotion exports."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SQL)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _load_inputs(
        manifest_path: Path, generator_config_path: Path
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generator_raw = generator_config_path.read_bytes()
        generator = json.loads(generator_raw)
        if not isinstance(manifest, dict) or not isinstance(generator, dict):
            raise TypeError("promotion export and generator config must be JSON objects")
        if manifest.get("schema_version") != EXPORT_SCHEMA:
            raise ValueError("unsupported Rust promotion export schema")
        content_sha = manifest.get("content_sha256")
        unsigned = {key: value for key, value in manifest.items() if key != "content_sha256"}
        if not _is_sha256(content_sha) or content_sha != _sha(unsigned):
            raise ValueError("promotion export content hash mismatch")
        if manifest.get("data_eligibility") != SERVICE_ELIGIBILITY:
            raise ValueError("promotion export eligibility is not fail-closed")
        if generator.get("observational_data_opened") is not False:
            raise ValueError("generator config opens observational data")
        if not generator.get("protocol_version"):
            raise ValueError("generator protocol version is missing")
        if not 1 <= int(generator.get("max_action_terms", 0)) <= 6:
            raise ValueError("generator action width is incompatible with SGPROM1")
        return manifest, generator, hashlib.sha256(generator_raw).hexdigest()

    @staticmethod
    def _verify_manifest_shape(manifest: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = manifest.get("blocks")
        if not isinstance(blocks, list):
            raise TypeError("promotion export blocks must be a list")
        if (
            manifest.get("block_count") != len(blocks)
            or manifest.get("survivor_identity_count")
            != sum(int(block.get("record_count", -1)) for block in blocks)
            or manifest.get("pass_count")
            != sum(int(block.get("pass_count", -1)) for block in blocks)
            or manifest.get("ambiguous_count")
            != sum(int(block.get("ambiguous_count", -1)) for block in blocks)
            or manifest.get("blocks_root_sha256") != _sha(blocks)
        ):
            raise ValueError("promotion export aggregate accounting mismatch")
        artifact_bytes = int(manifest.get("artifact_bytes", -1))
        maximum_bytes = int(manifest.get("maximum_export_bytes", -1))
        if artifact_bytes < 0 or maximum_bytes <= 0 or artifact_bytes > maximum_bytes:
            raise ValueError("promotion export byte accounting is invalid")
        source = manifest.get("source")
        if (
            not isinstance(manifest.get("service_id"), str)
            or not manifest["service_id"]
            or not isinstance(source, dict)
            or not source.get("source_id")
        ):
            raise ValueError("promotion export source identity is missing")
        next_ordinal = int(source.get("next_ordinal", -1))
        stop_ordinal = int(source.get("stop_ordinal", -1))
        if next_ordinal < 0 or stop_ordinal < next_ordinal:
            raise ValueError("promotion export source cursor is invalid")
        previous_end: int | None = None
        for block in blocks:
            start = int(block.get("start_ordinal", -1))
            end = int(block.get("end_ordinal_exclusive", -1))
            if (
                start < 0
                or end <= start
                or end > next_ordinal
                or int(block.get("record_count", -1)) > end - start
            ):
                raise ValueError("invalid promotion block interval")
            if previous_end is not None and start != previous_end:
                raise ValueError("promotion block intervals contain a gap or replay")
            previous_end = end
            if block.get("source_id") != source["source_id"]:
                raise ValueError("promotion block source identity mismatch")
        return blocks

    @staticmethod
    def _verify_block(
        manifest_path: Path,
        block: dict[str, Any],
        generator: dict[str, Any],
        generator_sha: str,
    ) -> list[dict[str, Any]]:
        required_hashes = (
            "sha256",
            "source_block_sha256",
            "source_manifest_sha256",
            "generator_config_sha256",
            "result_status_root_sha256",
        )
        if block.get("schema_version") != BLOCK_SCHEMA or not all(
            _is_sha256(block.get(key)) for key in required_hashes
        ):
            raise ValueError("promotion block provenance is invalid")
        if block.get("record_format") != (
            "SGPROM1/1 fixed-25-byte status-plus-SGSURV2-identity"
        ):
            raise ValueError("unsupported promotion block record format")
        if block["generator_config_sha256"] != generator_sha:
            raise ValueError("promotion block generator hash mismatch")
        path = _portable_path(manifest_path, block.get("file"))
        if not path.is_file() or _file_sha(path) != block["sha256"]:
            raise ValueError("promotion block file hash mismatch")
        start = int(block["start_ordinal"])
        end = int(block["end_ordinal_exclusive"])
        records: list[dict[str, Any]] = []
        statuses: Counter[int] = Counter()
        previous = start - 1 if start else -1
        with path.open("rb") as handle:
            raw_header = handle.read(PROMOTION_HEADER.size)
            if len(raw_header) != PROMOTION_HEADER.size:
                raise ValueError("truncated promotion block header")
            magic, version, record_size, actual_start, actual_end, count = (
                PROMOTION_HEADER.unpack(raw_header)
            )
            if (magic, version, record_size) != (PROMOTION_MAGIC, 1, PROMOTION_RECORD.size):
                raise ValueError("invalid promotion block header")
            if (actual_start, actual_end, count) != (
                start,
                end,
                int(block["record_count"]),
            ):
                raise ValueError("promotion block header interval mismatch")
            for _ in range(count):
                raw = handle.read(PROMOTION_RECORD.size)
                if len(raw) != PROMOTION_RECORD.size:
                    raise ValueError("truncated promotion block record")
                status, ordinal, term_count, sign_mask, reserved, *term_ids = (
                    PROMOTION_RECORD.unpack(raw)
                )
                if (
                    status not in (1, 2)
                    or reserved != 0
                    or not start <= ordinal < end
                    or ordinal <= previous
                    or not 1 <= term_count <= int(generator["max_action_terms"])
                    or sign_mask >= 1 << term_count
                    or any(value == 0xFFFF for value in term_ids[:term_count])
                    or any(value != 0xFFFF for value in term_ids[term_count:])
                ):
                    raise ValueError("invalid promotion survivor identity")
                decoded = decode_ordinal(
                    int(generator["basis_count"]),
                    int(generator["max_action_terms"]),
                    int(ordinal),
                )
                decoded_mask = sum(
                    1 << position
                    for position, sign in enumerate(decoded["signs"])
                    if sign > 0
                )
                if decoded["term_ids"] != list(term_ids[:term_count]) or decoded_mask != sign_mask:
                    raise ValueError("promotion identity differs from ordinal decoder")
                records.append({"status": status, "ordinal": ordinal, "decoded": decoded})
                statuses[status] += 1
                previous = ordinal
            if handle.read(1):
                raise ValueError("promotion block has trailing bytes")
        if statuses[1] != int(block["pass_count"]) or statuses[2] != int(
            block["ambiguous_count"]
        ):
            raise ValueError("promotion block status accounting mismatch")
        return records

    def _initialize_or_validate_state(
        self,
        manifest: dict[str, Any],
        manifest_path: Path,
        generator: dict[str, Any],
        generator_sha: str,
        blocks: list[dict[str, Any]],
    ) -> sqlite3.Row:
        if not blocks:
            raise ValueError("cannot initialize a bridge from an empty export")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM bridge_state WHERE singleton=1").fetchone()
            if row is None:
                first = int(blocks[0]["start_ordinal"])
                connection.execute(
                    "INSERT INTO bridge_state VALUES (1,?,?,?,?,?,0,0,?,?)",
                    (
                        BRIDGE_SCHEMA,
                        str(manifest["service_id"]),
                        generator_sha,
                        str(generator["protocol_version"]),
                        first,
                        first,
                        _now(),
                    ),
                )
                row = connection.execute("SELECT * FROM bridge_state WHERE singleton=1").fetchone()
            expected = (
                BRIDGE_SCHEMA,
                str(manifest["service_id"]),
                generator_sha,
                str(generator["protocol_version"]),
                int(blocks[0]["start_ordinal"]),
            )
            actual = tuple(
                row[key]
                for key in (
                    "schema_version",
                    "service_id",
                    "generator_config_sha256",
                    "generator_protocol_version",
                    "first_ordinal",
                )
            )
            if actual != expected:
                raise ValueError("bridge source identity changed across restart")
            verified = connection.execute(
                "SELECT * FROM verified_blocks ORDER BY block_index"
            ).fetchall()
            if len(blocks) < len(verified):
                raise ValueError("promotion export regressed and omitted verified blocks")
            for stored in verified:
                block = blocks[int(stored["block_index"])]
                if (
                    int(block["start_ordinal"]) != stored["start_ordinal"]
                    or int(block["end_ordinal_exclusive"]) != stored["end_ordinal_exclusive"]
                    or block["sha256"] != stored["block_sha256"]
                    or _sha(block) != stored["block_lineage_sha256"]
                    or int(block["record_count"]) != stored["record_count"]
                ):
                    raise ValueError("verified promotion block was changed or replayed")
                path = _portable_path(manifest_path, block["file"])
                if not path.is_file() or _file_sha(path) != stored["block_sha256"]:
                    raise ValueError("verified promotion block file was changed")
            return row

    def import_incremental(
        self,
        export_path: str | Path,
        generator_config_path: str | Path,
        orchestrator: PromotionOrchestrator,
        *,
        maximum_records: int,
        maximum_blocks: int = 16,
    ) -> dict[str, Any]:
        """Audit the snapshot, then consume a bounded record prefix exactly once."""
        if not 1 <= maximum_records <= 1_000_000:
            raise ValueError("maximum records must be between one and one million")
        if not 1 <= maximum_blocks <= 10_000:
            raise ValueError("maximum blocks must be between one and ten thousand")
        export_path = Path(export_path).resolve()
        generator_path = Path(generator_config_path).resolve()
        manifest, generator, generator_sha = self._load_inputs(export_path, generator_path)
        blocks = self._verify_manifest_shape(manifest)
        artifact_bytes = 0
        for block in blocks:
            path = _portable_path(export_path, block.get("file"))
            if not path.is_file():
                raise ValueError("promotion export references a missing block")
            artifact_bytes += path.stat().st_size
        if artifact_bytes != int(manifest["artifact_bytes"]):
            raise ValueError("promotion export artifact byte count mismatch")
        if not blocks:
            with self.connect() as connection:
                initialized = connection.execute(
                    "SELECT 1 FROM bridge_state WHERE singleton=1"
                ).fetchone()
            if initialized is not None:
                raise ValueError("promotion export regressed and omitted verified blocks")
            return {
                "schema_version": BRIDGE_SCHEMA,
                "audited_blocks": 0,
                "consumed_records": 0,
                "registered_candidates": 0,
                "duplicate_candidates": 0,
                "ambiguous_not_promoted": 0,
                "completed_blocks": 0,
                "cursor": None,
                "snapshot_exhausted": True,
                "data_eligibility": dict(SERVICE_ELIGIBILITY),
            }
        state = self._initialize_or_validate_state(
            manifest, export_path, generator, generator_sha, blocks
        )
        block_index = int(state["next_block_index"])
        record_index = int(state["next_record_index"])
        next_interval = int(state["next_interval_start"])
        consumed = registered = duplicates = ambiguous = completed_blocks = 0
        basis = build_basis(int(generator["basis_count"]))
        audited_blocks = 0
        while (
            block_index < len(blocks)
            and consumed < maximum_records
            and completed_blocks < maximum_blocks
        ):
            block = blocks[block_index]
            records = self._verify_block(export_path, block, generator, generator_sha)
            audited_blocks += 1
            if int(block["start_ordinal"]) != next_interval:
                raise ValueError("bridge cursor does not match promotion block interval")
            block_lineage = _sha(block)
            with self.connect() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO verified_blocks VALUES (?,?,?,?,?,?,?)",
                    (
                        block_index,
                        int(block["start_ordinal"]),
                        int(block["end_ordinal_exclusive"]),
                        block["sha256"],
                        block_lineage,
                        int(block["record_count"]),
                        _now(),
                    ),
                )
            while record_index < len(records) and consumed < maximum_records:
                record = records[record_index]
                status = "pass" if record["status"] == 1 else "ambiguous"
                decoded = record["decoded"]
                identifier = generator_candidate_id(
                    str(generator["protocol_version"]), decoded
                )
                initial_lineage: str | None = None
                if status == "pass":
                    candidate = {
                        "candidate_id": identifier,
                        "ordinal": int(record["ordinal"]),
                        "term_ids": list(decoded["term_ids"]),
                        "signs": list(decoded["signs"]),
                        "correction_expression": correction_expression(decoded, basis),
                        "source_service_id": manifest["service_id"],
                        "source_promotion_block_sha256": block["sha256"],
                        "source_promotion_block_lineage_sha256": block_lineage,
                        "source_generator_config_sha256": generator_sha,
                        "source_generator_manifest_sha256": block[
                            "source_manifest_sha256"
                        ],
                        "data_eligibility": dict(ELIGIBILITY),
                    }
                    evidence = {
                        "schema_version": EVIDENCE_SCHEMA,
                        "candidate_id": identifier,
                        "ordinal": int(record["ordinal"]),
                        "status": "pass",
                        "sampled_static_status": "pass",
                        "source_result_sha256": block["result_status_root_sha256"],
                        "source_promotion_block_sha256": block["sha256"],
                        "source_promotion_block_lineage_sha256": block_lineage,
                        "source_generator_config_sha256": generator_sha,
                        "source_generator_block_sha256": block["source_block_sha256"],
                        "source_generator_manifest_sha256": block[
                            "source_manifest_sha256"
                        ],
                        "data_eligibility": dict(ELIGIBILITY),
                    }
                    with orchestrator.connect() as connection:
                        existed = connection.execute(
                            "SELECT 1 FROM candidates WHERE candidate_id=?", (identifier,)
                        ).fetchone()
                    initial_lineage = orchestrator.register_candidate(candidate, evidence)
                    if existed:
                        duplicates += 1
                    else:
                        registered += 1
                else:
                    ambiguous += 1
                with self.connect() as connection:
                    existing = connection.execute(
                        "SELECT sampled_status,candidate_id,initial_lineage_sha256 "
                        "FROM consumed_records WHERE block_index=? AND record_index=?",
                        (block_index, record_index),
                    ).fetchone()
                    expected = (status, identifier, initial_lineage)
                    if existing is not None and tuple(existing) != expected:
                        raise ValueError("promotion record replay has changed lineage")
                    if existing is None:
                        connection.execute(
                            "INSERT INTO consumed_records VALUES (?,?,?,?,?,?,?)",
                            (
                                block_index,
                                record_index,
                                int(record["ordinal"]),
                                status,
                                identifier,
                                initial_lineage,
                                _now(),
                            ),
                        )
                    connection.execute(
                        "UPDATE bridge_state SET next_record_index=?,updated_utc=? WHERE singleton=1",
                        (record_index + 1, _now()),
                    )
                record_index += 1
                consumed += 1
            if record_index == len(records):
                next_interval = int(block["end_ordinal_exclusive"])
                block_index += 1
                record_index = 0
                completed_blocks += 1
                with self.connect() as connection:
                    connection.execute(
                        "UPDATE bridge_state SET next_block_index=?,next_record_index=0,"
                        "next_interval_start=?,updated_utc=? WHERE singleton=1",
                        (block_index, next_interval, _now()),
                    )
        return {
            "schema_version": BRIDGE_SCHEMA,
            "audited_blocks": audited_blocks,
            "consumed_records": consumed,
            "registered_candidates": registered,
            "duplicate_candidates": duplicates,
            "ambiguous_not_promoted": ambiguous,
            "completed_blocks": completed_blocks,
            "cursor": {
                "next_block_index": block_index,
                "next_record_index": record_index,
                "next_interval_start": next_interval,
            },
            "snapshot_exhausted": block_index == len(blocks),
            "data_eligibility": dict(SERVICE_ELIGIBILITY),
        }

    def status(self) -> dict[str, Any]:
        with self.connect() as connection:
            state = connection.execute("SELECT * FROM bridge_state WHERE singleton=1").fetchone()
            counts = connection.execute(
                "SELECT sampled_status,COUNT(*) AS count FROM consumed_records "
                "GROUP BY sampled_status ORDER BY sampled_status"
            ).fetchall()
            verified = connection.execute("SELECT COUNT(*) FROM verified_blocks").fetchone()[0]
        return {
            "schema_version": BRIDGE_SCHEMA,
            "initialized": state is not None,
            "cursor": (
                {
                    "next_block_index": state["next_block_index"],
                    "next_record_index": state["next_record_index"],
                    "next_interval_start": state["next_interval_start"],
                }
                if state is not None
                else None
            ),
            "verified_blocks": verified,
            "consumed": {row["sampled_status"]: row["count"] for row in counts},
            "data_eligibility": dict(SERVICE_ELIGIBILITY),
        }
