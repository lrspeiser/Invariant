from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import receipt_dag as R
from sigma_theory_compiler.sigma_core import canonical_sha256


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def seal(value: dict) -> dict:
    return {**value, "content_sha256": canonical_sha256(value)}


def manifest() -> R.ReceiptDagManifest:
    return R.ReceiptDagManifest(
        "test-dag",
        ("receipts",),
        ("receipts/middle.json", "receipts/top.json"),
        (R.AliasRule("leaf_file_sha256", "inputs/leaf.txt", "portable_file_sha256"),),
    )


def fixture(root: Path) -> None:
    (root / "inputs").mkdir(parents=True)
    (root / "inputs/leaf.txt").write_bytes(b"new\r\nleaf\r\n")
    write_json(
        root / "receipts/middle.json",
        seal({"leaf_file_sha256": "0" * 64, "schema_version": "middle-1.0"}),
    )
    write_json(
        root / "receipts/top.json",
        seal(
            {
                "middle": {
                    "content_sha256": "1" * 64,
                    "file_sha256": "2" * 64,
                    "path": "receipts/middle.json",
                },
                "schema_version": "top-1.0",
            }
        ),
    )


def test_dag_discovers_alias_file_and_content_edges(tmp_path: Path) -> None:
    fixture(tmp_path)
    dag = R.discover_receipt_dag(tmp_path, manifest())
    assert dag.order == ("receipts/middle.json", "receipts/top.json")
    assert {(item.owner, item.target, item.hash_kind) for item in dag.bindings} == {
        ("receipts/middle.json", "inputs/leaf.txt", "portable_file_sha256"),
        ("receipts/top.json", "receipts/middle.json", "file_sha256"),
        ("receipts/top.json", "receipts/middle.json", "content_sha256"),
    }
    assert not R.audit_receipt_dag(dag)["valid"]


def test_reseal_updates_dependencies_before_owners_and_converges(tmp_path: Path) -> None:
    fixture(tmp_path)
    dag = R.discover_receipt_dag(tmp_path, manifest())
    result = R.reseal_receipt_dag(dag, write=True)
    assert result["audit"]["valid"]
    middle = json.loads((tmp_path / "receipts/middle.json").read_text())
    top = json.loads((tmp_path / "receipts/top.json").read_text())
    portable_leaf = hashlib.sha256(b"new\nleaf\n").hexdigest()
    assert middle["leaf_file_sha256"] == portable_leaf
    assert middle["content_sha256"] == canonical_sha256(
        {key: value for key, value in middle.items() if key != "content_sha256"}
    )
    assert top["middle"]["content_sha256"] == middle["content_sha256"]
    assert top["middle"]["file_sha256"] == hashlib.sha256(
        (tmp_path / "receipts/middle.json").read_bytes()
    ).hexdigest()


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    fixture(tmp_path)
    before = (tmp_path / "receipts/top.json").read_bytes()
    result = R.reseal_receipt_dag(R.discover_receipt_dag(tmp_path, manifest()), write=False)
    assert result["changes"]
    assert result["projected_audit"]["valid"]
    assert (tmp_path / "receipts/top.json").read_bytes() == before


def test_cycle_and_path_escape_fail_before_write(tmp_path: Path) -> None:
    write_json(
        tmp_path / "receipts/a.json",
        {"binding": {"path": "receipts/b.json", "file_sha256": "0" * 64}},
    )
    write_json(
        tmp_path / "receipts/b.json",
        {"binding": {"path": "receipts/a.json", "file_sha256": "0" * 64}},
    )
    cyclic = R.ReceiptDagManifest("cycle", ("receipts",), (), ())
    with pytest.raises(R.ReceiptDagError, match="cycle"):
        R.discover_receipt_dag(tmp_path, cyclic)
    escaping = R.ReceiptDagManifest(
        "escape",
        ("receipts",),
        (),
        (R.AliasRule("field_sha256", "../outside", "file_sha256"),),
    )
    write_json(tmp_path / "receipts/c.json", {"field_sha256": "0" * 64})
    with pytest.raises(R.ReceiptDagError, match="escapes"):
        R.discover_receipt_dag(tmp_path, escaping)


def test_v2_reverse_closure_auto_seals_and_uses_portable_hash_policy(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "inputs/portable.txt").write_bytes(b"portable\r\nbytes\r\n")
    write_json(
        tmp_path / "receipts/root.json",
        {
            "content_sha256": "0" * 64,
            "source": {
                "file_sha256": "1" * 64,
                "path": "inputs/portable.txt",
            },
        },
    )
    write_json(
        tmp_path / "receipts/dependent.json",
        {
            "content_sha256": "2" * 64,
            "root": {
                "content_sha256": "3" * 64,
                "file_sha256": "4" * 64,
                "path": "receipts/root.json",
            },
        },
    )
    write_json(
        tmp_path / "receipts/unrelated.json",
        {"content_sha256": "5" * 64, "schema_version": "unrelated-1.0"},
    )
    value = {
        "aliases": [],
        "auto_self_sealed": True,
        "builders": [],
        "manifest_id": "closure-v2",
        "portable_all_file_hashes": False,
        "portable_targets": ["inputs/portable.txt"],
        "root_nodes": ["receipts/root.json"],
        "scan_paths": ["receipts"],
        "schema_version": R.MANIFEST_SCHEMA,
        "self_sealed": [],
    }
    dag = R.discover_receipt_dag(tmp_path, R.ReceiptDagManifest.from_dict(value))
    assert dag.json_nodes == ("receipts/dependent.json", "receipts/root.json")
    assert dag.order == ("receipts/root.json", "receipts/dependent.json")
    assert dag.self_sealed == ("receipts/dependent.json", "receipts/root.json")
    source = next(item for item in dag.bindings if item.target == "inputs/portable.txt")
    assert source.hash_kind == "portable_file_sha256"
    result = R.reseal_receipt_dag(dag, write=True)
    assert result["audit"]["valid"]
    root = json.loads((tmp_path / "receipts/root.json").read_text(encoding="utf-8"))
    assert root["source"]["file_sha256"] == hashlib.sha256(
        b"portable\nbytes\n"
    ).hexdigest()
    refreshed = R.discover_receipt_dag(tmp_path, R.ReceiptDagManifest.from_dict(value))
    assert "receipts/unrelated.json" not in refreshed.json_nodes


def test_stale_builder_runs_dependency_first_and_is_verified_deterministic(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "inputs").mkdir()
    (tmp_path / "receipts").mkdir()
    (tmp_path / "inputs/value.txt").write_bytes(b"builder\r\ninput\r\n")
    (tmp_path / "src/builder_control.py").write_text(
        """import hashlib, json
from pathlib import Path
raw = Path('inputs/value.txt').read_bytes().replace(b'\\r\\n', b'\\n').replace(b'\\r', b'\\n')
body = {
    'binding': {
        'path': 'inputs/value.txt',
        'portable_file_sha256': hashlib.sha256(raw).hexdigest(),
    },
    'schema_version': 'builder-control-1.0',
}
seal = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
).hexdigest()
Path('receipts/generated.json').write_text(
    json.dumps({**body, 'content_sha256': seal}, indent=2) + '\\n',
    encoding='utf-8', newline='\\n'
)
""",
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        tmp_path / "receipts/generated.json",
        {
            "binding": {
                "path": "inputs/value.txt",
                "portable_file_sha256": "0" * 64,
            },
            "content_sha256": "1" * 64,
            "schema_version": "builder-control-1.0",
        },
    )
    manifest = R.ReceiptDagManifest(
        "builder-dag",
        ("receipts",),
        ("receipts/generated.json",),
        (),
        builders=(
            R.BuilderRule(
                "receipts/generated.json",
                "builder_control",
                ("--run",),
                30,
                True,
            ),
        ),
    )
    result = R.rebuild_receipt_dag(R.discover_receipt_dag(tmp_path, manifest))
    assert result["audit"]["valid"]
    assert result["builder_events"][0]["node"] == "receipts/generated.json"
    assert result["builder_events"][0]["verified_deterministic"]
    repeated = R.rebuild_receipt_dag(
        R.discover_receipt_dag(tmp_path, manifest), force_builders=True
    )
    assert repeated["audit"]["valid"]
    assert repeated["builder_events"][0]["changed"] is False
