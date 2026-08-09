from __future__ import annotations

import hashlib
import json

from sigma_theory_compiler.survivors import (
    HEADER,
    MAGIC,
    RECORD,
    audit_survivor_export,
    iter_survivors,
)


def test_compact_survivor_round_trip_and_audit(tmp_path):
    survivor_dir = tmp_path / "blocks"
    survivor_dir.mkdir()
    block_path = survivor_dir / "survivors-00000000-0-10.bin"
    payload = bytearray(HEADER.pack(MAGIC, 1, RECORD.size, 0, 0, 10, 1))
    payload.extend(RECORD.pack(7, 2, 1, 0, 3, 19, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF))
    block_path.write_bytes(payload)
    manifest = {
        "survivor_export_directory": str(survivor_dir),
        "survivor_count": 1,
        "blocks": [
            {
                "block_index": 0,
                "start_ordinal": 0,
                "end_ordinal_exclusive": 10,
                "survivor_export": {
                    "file": block_path.name,
                    "record_count": 1,
                    "file_size_bytes": len(payload),
                    "file_sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records = list(iter_survivors(manifest_path))
    assert records == [{"ordinal": 7, "term_ids": [3, 19], "sign_mask": 1}]
    report = audit_survivor_export(manifest_path, tmp_path / "audit.json")
    assert report["all_checks_pass"]
    assert report["record_count"] == 1


def test_audit_detects_hash_change(tmp_path):
    survivor_dir = tmp_path / "blocks"
    survivor_dir.mkdir()
    block_path = survivor_dir / "block.bin"
    payload = HEADER.pack(MAGIC, 1, RECORD.size, 0, 0, 1, 0)
    block_path.write_bytes(payload)
    manifest = {
        "survivor_export_directory": str(survivor_dir),
        "survivor_count": 0,
        "blocks": [
            {
                "block_index": 0,
                "start_ordinal": 0,
                "end_ordinal_exclusive": 1,
                "survivor_export": {
                    "file": block_path.name,
                    "record_count": 0,
                    "file_size_bytes": len(payload),
                    "file_sha256": "0" * 64,
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = audit_survivor_export(manifest_path, tmp_path / "audit.json")
    assert not report["all_checks_pass"]
    assert report["errors"] == ["sha256:block.bin"]
