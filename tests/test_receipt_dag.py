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
