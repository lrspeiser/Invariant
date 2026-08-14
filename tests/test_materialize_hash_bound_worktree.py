from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.materialize_hash_bound_worktree import materialize


def test_materializes_binding_reachable_only_from_campaign_config(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    unified = configs / "unified_engine_status.json"
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")

    relative = "runs/example/artifact.json"
    artifact = tmp_path / relative
    artifact.parent.mkdir(parents=True)
    lf = b'{\n  "value": 1\n}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    artifact.write_bytes(lf)
    (configs / "campaign.json").write_text(
        json.dumps(
            {
                "predecessors": {
                    "example": {
                        "path": relative,
                        "file_sha256": hashlib.sha256(crlf).hexdigest(),
                        "content_sha256": "0" * 64,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = materialize(tmp_path, unified)

    assert artifact.read_bytes() == crlf
    assert result["files_rewritten"] == 1
    assert result["missing_bound_paths"] == 0
    assert result["config_documents_scanned"] == 4
    assert result["materialization_passes"] == 2


def test_materializes_nested_binding_revealed_by_first_pass(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    unified = configs / "unified_engine_status.json"
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")

    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source_lf = b"value = 1\n"
    source_crlf = source_lf.replace(b"\n", b"\r\n")
    source.write_bytes(source_lf)

    artifact = tmp_path / "runs" / "nested.json"
    artifact.parent.mkdir()
    artifact_lf = (
        json.dumps(
            {
                "source_binding": {
                    "path": "src/example.py",
                    "file_sha256": hashlib.sha256(source_crlf).hexdigest(),
                }
            },
            indent=2,
        ).encode()
        + b"\n"
    )
    artifact_crlf = artifact_lf.replace(b"\n", b"\r\n")
    artifact.write_bytes(artifact_lf)
    (configs / "campaign.json").write_text(
        json.dumps(
            {
                "artifact": {
                    "path": "runs/nested.json",
                    "file_sha256": hashlib.sha256(artifact_crlf).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    result = materialize(tmp_path, unified)

    assert artifact.read_bytes() == artifact_crlf
    assert source.read_bytes() == source_crlf
    assert result["files_rewritten"] == 2
    assert result["materialization_passes"] == 2


def test_conflicting_line_endings_choose_first_reachable_registration(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    unified = configs / "unified_engine_status.json"
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")

    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source_lf = b"value = 1\n"
    source_crlf = source_lf.replace(b"\n", b"\r\n")
    source.write_bytes(source_lf)
    (configs / "a-authoritative.json").write_text(
        json.dumps(
            {
                "source_path": "src/example.py",
                "source_file_sha256": hashlib.sha256(source_crlf).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (configs / "z-historical.json").write_text(
        json.dumps(
            {
                "source": {
                    "path": "src/example.py",
                    "file_sha256": hashlib.sha256(source_lf).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    result = materialize(tmp_path, unified)

    assert source.read_bytes() == source_crlf
    assert result["files_rewritten"] == 1
    assert result["materialization_passes"] == 2
    assert result["superseded_bindings_skipped"] == 2
