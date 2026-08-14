from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.materialize_hash_bound_worktree import materialize

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_REL = Path(
    "configs/backgrounds/quartic_full_d2f_typed_partition_row_extension_gate_byte_authority.json"
)
DOWNSTREAM_REL = Path(
    "configs/backgrounds/quartic_registered_direction_cross_leaf_d2_replay_gate.json"
)


def _load(relative: Path) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_authority_is_exactly_the_checked_artifacts_direct_self_provenance() -> None:
    authority = _load(AUTHORITY_REL)
    artifact_binding = authority["checked_artifact"]
    artifact = json.loads((ROOT / artifact_binding["path"]).read_text(encoding="utf-8"))

    assert set(authority) == {
        "schema_version",
        "authority_id",
        "checked_artifact",
        "direct_self_bindings",
        "authority_contract",
    }
    assert artifact_binding == {
        "path": "runs/physics-language/"
        "quartic-full-d2f-typed-partition-row-extension-gate/campaign.json",
        "file_sha256": "9502843234509a4ddd21631acdfe412d0f17fe3552d7c9cac0daf7fb1475190a",
        "content_sha256": artifact["content_sha256"],
    }
    assert authority["direct_self_bindings"] == {
        label: artifact["source_bindings"][label] for label in ("config", "source", "test")
    }
    assert authority["authority_contract"] == {
        "scope": "line_ending_representation_only",
        "scientific_payload_changed": False,
        "checked_artifact_resealed": False,
        "downstream_artifact_resealed": False,
        "selection": "checked_artifact_direct_self_bindings_precede_copied_downstream_predecessor_bindings",
    }


def test_downstream_crlf_bytes_materialize_back_to_checked_lf_self_bytes(
    tmp_path: Path,
) -> None:
    authority = _load(AUTHORITY_REL)
    downstream = _load(DOWNSTREAM_REL)
    checked = authority["direct_self_bindings"]
    copied = downstream["predecessor"]

    assert AUTHORITY_REL.as_posix() < DOWNSTREAM_REL.as_posix()
    for label in ("config", "source", "test"):
        assert checked[label]["path"] == copied[label]["path"]
        assert checked[label]["file_sha256"] != copied[label]["file_sha256"]

    unified = tmp_path / "configs/unified_engine_status.json"
    unified.parent.mkdir(parents=True)
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")

    authority_path = tmp_path / AUTHORITY_REL
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_bytes((ROOT / AUTHORITY_REL).read_bytes())
    downstream_path = tmp_path / DOWNSTREAM_REL
    downstream_path.write_text(
        json.dumps({"predecessor": downstream["predecessor"]}), encoding="utf-8"
    )
    artifact_binding = authority["checked_artifact"]
    artifact_path = tmp_path / artifact_binding["path"]
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes((ROOT / artifact_binding["path"]).read_bytes())

    for binding in checked.values():
        source = ROOT / binding["path"]
        target = tmp_path / binding["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        lf = source.read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(lf.replace(b"\n", b"\r\n"))

    result = materialize(tmp_path, unified)

    assert result["files_rewritten"] >= 3
    for binding in checked.values():
        assert _sha((tmp_path / binding["path"]).read_bytes()) == binding["file_sha256"]
