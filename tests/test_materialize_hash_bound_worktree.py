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
    assert result["config_documents_scanned"] == 2
