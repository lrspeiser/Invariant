"""Discover, audit, and deterministically reseal explicit receipt dependencies.

The tool recognizes path/hash pairs rather than replacing arbitrary 64-character strings.
Edges point from a JSON owner to the file it binds.  Resealing visits dependencies before
owners, refreshes only recognized binding fields, then refreshes explicitly declared root
``content_sha256`` seals.  Cycles, escaping paths, ambiguous aliases, and missing targets fail
closed before any write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

MANIFEST_SCHEMA = "invariant-receipt-dag-manifest-1.0"
AUDIT_SCHEMA = "invariant-receipt-dag-audit-1.0"
HASH_KEYS = ("file_sha256", "portable_file_sha256", "content_sha256")


class ReceiptDagError(ValueError):
    """The receipt graph is malformed, cyclic, ambiguous, or stale."""


def _portable(path: Path) -> str:
    return path.as_posix()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ReceiptDagError("receipt DAG paths must be nonempty repository-relative POSIX paths")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ReceiptDagError(f"receipt DAG path escapes repository: {relative}") from error
    return candidate


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_sha(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptDagError(f"cannot read JSON node: {path}") from error
    if not isinstance(value, dict):
        raise ReceiptDagError(f"receipt DAG JSON node is not an object: {path}")
    return value


@dataclass(frozen=True, slots=True)
class AliasRule:
    hash_key: str
    target: str
    hash_kind: str

    def __post_init__(self) -> None:
        if not self.hash_key.endswith("sha256") or self.hash_kind not in HASH_KEYS:
            raise ReceiptDagError("receipt DAG alias rule has an unsupported hash field")


@dataclass(frozen=True, slots=True)
class ReceiptDagManifest:
    manifest_id: str
    scan_paths: tuple[str, ...]
    self_sealed: tuple[str, ...]
    aliases: tuple[AliasRule, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReceiptDagManifest:
        if set(value) != {"aliases", "manifest_id", "scan_paths", "schema_version", "self_sealed"}:
            raise ReceiptDagError("receipt DAG manifest keys changed")
        if value.get("schema_version") != MANIFEST_SCHEMA:
            raise ReceiptDagError("receipt DAG manifest schema changed")
        aliases = []
        for item in value["aliases"]:
            if not isinstance(item, Mapping) or set(item) != {"hash_key", "hash_kind", "target"}:
                raise ReceiptDagError("receipt DAG alias keys changed")
            aliases.append(AliasRule(item["hash_key"], item["target"], item["hash_kind"]))
        hash_keys = [item.hash_key for item in aliases]
        if len(hash_keys) != len(set(hash_keys)):
            raise ReceiptDagError("receipt DAG alias hash keys are ambiguous")
        result = cls(
            manifest_id=value["manifest_id"],
            scan_paths=tuple(value["scan_paths"]),
            self_sealed=tuple(value["self_sealed"]),
            aliases=tuple(aliases),
        )
        if not result.manifest_id or not result.scan_paths:
            raise ReceiptDagError("receipt DAG manifest identity or scan set is empty")
        return result


@dataclass(frozen=True, slots=True)
class Binding:
    owner: str
    target: str
    pointer: tuple[str | int, ...]
    hash_key: str
    hash_kind: str
    declared: str

    @property
    def pointer_text(self) -> str:
        return "/" + "/".join(str(item).replace("~", "~0").replace("/", "~1") for item in self.pointer)


@dataclass(frozen=True, slots=True)
class ReceiptDag:
    root: Path
    manifest: ReceiptDagManifest
    json_nodes: tuple[str, ...]
    bindings: tuple[Binding, ...]
    order: tuple[str, ...]


def load_manifest(root: Path, path: str | Path) -> ReceiptDagManifest:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _resolve(root, candidate.as_posix())
    return ReceiptDagManifest.from_dict(_load_json(candidate))


def _scan_nodes(root: Path, scan_paths: Sequence[str]) -> tuple[str, ...]:
    nodes: set[str] = set()
    for relative in scan_paths:
        candidate = _resolve(root, relative)
        if candidate.is_file():
            if candidate.suffix == ".json":
                nodes.add(_portable(candidate.relative_to(root.resolve())))
            continue
        if not candidate.is_dir():
            raise ReceiptDagError(f"receipt DAG scan path is missing: {relative}")
        nodes.update(
            _portable(path.relative_to(root.resolve())) for path in candidate.rglob("*.json")
        )
    return tuple(sorted(nodes))


def _walk(value: Any, pointer: tuple[str | int, ...] = ()) -> Iterable[tuple[tuple[str | int, ...], Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        yield pointer, value
        for key, child in value.items():
            yield from _walk(child, (*pointer, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*pointer, index))


def _binding(
    root: Path,
    owner: str,
    pointer: tuple[str | int, ...],
    mapping: Mapping[str, Any],
    path_key: str,
    hash_key: str,
    hash_kind: str,
) -> Binding | None:
    target = mapping.get(path_key)
    declared = mapping.get(hash_key)
    if not isinstance(target, str) or not isinstance(declared, str):
        return None
    target_path = _resolve(root, target)
    if not target_path.is_file():
        raise ReceiptDagError(f"bound target is missing: {target}")
    return Binding(owner, target, (*pointer, hash_key), hash_key, hash_kind, declared)


def _discover_bindings(
    root: Path, owner: str, value: Mapping[str, Any], aliases: Sequence[AliasRule]
) -> tuple[Binding, ...]:
    rows: dict[tuple[str | int, ...], Binding] = {}
    for pointer, mapping in _walk(value):
        if "path" in mapping:
            for hash_key in HASH_KEYS:
                item = _binding(root, owner, pointer, mapping, "path", hash_key, hash_key)
                if item is not None:
                    rows[item.pointer] = item
        for key in mapping:
            if not isinstance(key, str) or not key.endswith("_path"):
                continue
            stem = key[:-5]
            for suffix in HASH_KEYS:
                hash_key = f"{stem}_{suffix}"
                item = _binding(root, owner, pointer, mapping, key, hash_key, suffix)
                if item is not None:
                    rows[item.pointer] = item
        for alias in aliases:
            if alias.hash_key not in mapping:
                continue
            declared = mapping[alias.hash_key]
            if not isinstance(declared, str):
                raise ReceiptDagError(f"alias binding is not a string: {owner}:{alias.hash_key}")
            target_path = _resolve(root, alias.target)
            if not target_path.is_file():
                raise ReceiptDagError(f"alias target is missing: {alias.target}")
            item = Binding(
                owner,
                alias.target,
                (*pointer, alias.hash_key),
                alias.hash_key,
                alias.hash_kind,
                declared,
            )
            incumbent = rows.get(item.pointer)
            if incumbent is not None and (
                incumbent.target != item.target or incumbent.hash_kind != item.hash_kind
            ):
                raise ReceiptDagError(f"binding is ambiguous at {owner}:{item.pointer_text}")
            rows[item.pointer] = item
    return tuple(rows[key] for key in sorted(rows, key=repr))


def _topological(nodes: Sequence[str], bindings: Sequence[Binding]) -> tuple[str, ...]:
    node_set = set(nodes)
    dependencies = {
        node: {item.target for item in bindings if item.owner == node and item.target in node_set}
        for node in nodes
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def visit(node: str) -> None:
        if node in visiting:
            raise ReceiptDagError(f"receipt DAG contains a cycle through {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in sorted(dependencies[node]):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)
        order.append(node)

    for node in sorted(nodes):
        visit(node)
    return tuple(order)


def discover_receipt_dag(root: Path, manifest: ReceiptDagManifest) -> ReceiptDag:
    root = root.resolve()
    nodes = _scan_nodes(root, manifest.scan_paths)
    bindings = tuple(
        item
        for owner in nodes
        for item in _discover_bindings(root, owner, _load_json(_resolve(root, owner)), manifest.aliases)
    )
    self_sealed = set(manifest.self_sealed)
    if not self_sealed <= set(nodes):
        raise ReceiptDagError("self-sealed files must be JSON nodes inside the scan set")
    return ReceiptDag(root, manifest, nodes, bindings, _topological(nodes, bindings))


def expected_hash(root: Path, binding: Binding) -> str:
    target = _resolve(root, binding.target)
    if binding.hash_kind == "file_sha256":
        return _raw_sha(target)
    if binding.hash_kind == "portable_file_sha256":
        return _portable_sha(target)
    if binding.hash_kind == "content_sha256":
        content = _load_json(target).get("content_sha256")
        if not isinstance(content, str):
            raise ReceiptDagError(f"content binding target has no root seal: {binding.target}")
        return content
    raise ReceiptDagError(f"unsupported hash kind: {binding.hash_kind}")


def _root_seal(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    return canonical_sha256(body)


def audit_receipt_dag(dag: ReceiptDag) -> dict[str, Any]:
    stale = []
    for binding in dag.bindings:
        expected = expected_hash(dag.root, binding)
        if binding.declared != expected:
            stale.append(
                {
                    "declared": binding.declared,
                    "expected": expected,
                    "hash_kind": binding.hash_kind,
                    "owner": binding.owner,
                    "pointer": binding.pointer_text,
                    "target": binding.target,
                }
            )
    bad_seals = []
    for relative in dag.manifest.self_sealed:
        value = _load_json(_resolve(dag.root, relative))
        declared = value.get("content_sha256")
        expected = _root_seal(value)
        if declared != expected:
            bad_seals.append({"declared": declared, "expected": expected, "owner": relative})
    body = {
        "schema_version": AUDIT_SCHEMA,
        "manifest_id": dag.manifest.manifest_id,
        "counts": {
            "bindings": len(dag.bindings),
            "json_nodes": len(dag.json_nodes),
            "self_sealed": len(dag.manifest.self_sealed),
            "stale_bindings": len(stale),
            "stale_self_seals": len(bad_seals),
        },
        "stale_bindings": stale,
        "stale_self_seals": bad_seals,
        "topological_order": list(dag.order),
        "valid": not stale and not bad_seals,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _set_pointer(value: Any, pointer: Sequence[str | int], replacement: str) -> None:
    current = value
    for part in pointer[:-1]:
        current = current[part]
    current[pointer[-1]] = replacement


def reseal_receipt_dag(dag: ReceiptDag, *, write: bool) -> dict[str, Any]:
    """Return proposed updates, or write them dependency-first when explicitly requested."""

    changes: list[dict[str, Any]] = []
    bindings_by_owner = {
        owner: tuple(item for item in dag.bindings if item.owner == owner) for owner in dag.json_nodes
    }
    for owner in dag.order:
        path = _resolve(dag.root, owner)
        value = _load_json(path)
        owner_changes = []
        for binding in bindings_by_owner[owner]:
            expected = expected_hash(dag.root, binding)
            current = value
            for part in binding.pointer:
                current = current[part]
            if current != expected:
                _set_pointer(value, binding.pointer, expected)
                owner_changes.append(
                    {
                        "from": current,
                        "pointer": binding.pointer_text,
                        "target": binding.target,
                        "to": expected,
                    }
                )
        if owner in dag.manifest.self_sealed:
            expected = _root_seal(value)
            current = value.get("content_sha256")
            if current != expected:
                value["content_sha256"] = expected
                owner_changes.append(
                    {"from": current, "pointer": "/content_sha256", "target": owner, "to": expected}
                )
        if owner_changes:
            changes.append({"changes": owner_changes, "owner": owner})
            if write:
                path.write_text(
                    json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
    if write:
        # Rediscover because declared values and target file bytes changed during the walk.
        refreshed = discover_receipt_dag(dag.root, dag.manifest)
        audit = audit_receipt_dag(refreshed)
        if not audit["valid"]:
            raise ReceiptDagError("receipt DAG remained stale after a topological reseal")
    else:
        audit = audit_receipt_dag(dag)
    return {"audit": audit, "changes": changes, "write": write}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    manifest = load_manifest(root, args.manifest)
    dag = discover_receipt_dag(root, manifest)
    if args.write:
        result = reseal_receipt_dag(dag, write=True)
    else:
        result = {"audit": audit_receipt_dag(dag), "changes": [], "write": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["audit"]["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

