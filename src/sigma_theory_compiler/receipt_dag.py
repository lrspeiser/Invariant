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
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

MANIFEST_SCHEMA = "invariant-receipt-dag-manifest-2.0"
LEGACY_MANIFEST_SCHEMA = "invariant-receipt-dag-manifest-1.0"
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
class BuilderRule:
    node: str
    module: str
    arguments: tuple[str, ...]
    timeout_seconds: int
    verify_determinism: bool

    def __post_init__(self) -> None:
        if (
            not self.node.endswith(".json")
            or not self.module
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._"
                for character in self.module
            )
            or not self.arguments
            or any(not isinstance(item, str) or "\x00" in item for item in self.arguments)
            or isinstance(self.timeout_seconds, bool)
            or not 1 <= self.timeout_seconds <= 3600
            or not isinstance(self.verify_determinism, bool)
        ):
            raise ReceiptDagError("receipt DAG builder rule is malformed")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BuilderRule:
        if set(value) != {
            "arguments",
            "module",
            "node",
            "timeout_seconds",
            "verify_determinism",
        }:
            raise ReceiptDagError("receipt DAG builder keys changed")
        return cls(
            value["node"],
            value["module"],
            tuple(value["arguments"]),
            value["timeout_seconds"],
            value["verify_determinism"],
        )


@dataclass(frozen=True, slots=True)
class ReceiptDagManifest:
    manifest_id: str
    scan_paths: tuple[str, ...]
    self_sealed: tuple[str, ...]
    aliases: tuple[AliasRule, ...]
    root_nodes: tuple[str, ...] = ()
    auto_self_sealed: bool = False
    portable_targets: tuple[str, ...] = ()
    portable_all_file_hashes: bool = False
    builders: tuple[BuilderRule, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReceiptDagManifest:
        legacy_keys = {"aliases", "manifest_id", "scan_paths", "schema_version", "self_sealed"}
        current_keys = legacy_keys | {
            "auto_self_sealed",
            "builders",
            "portable_all_file_hashes",
            "portable_targets",
            "root_nodes",
        }
        keys = frozenset(value)
        if keys not in {frozenset(legacy_keys), frozenset(current_keys)}:
            raise ReceiptDagError("receipt DAG manifest keys changed")
        if value.get("schema_version") not in {MANIFEST_SCHEMA, LEGACY_MANIFEST_SCHEMA}:
            raise ReceiptDagError("receipt DAG manifest schema changed")
        if value.get("schema_version") == MANIFEST_SCHEMA and set(value) != current_keys:
            raise ReceiptDagError("receipt DAG v2 manifest omits operational fields")
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
            root_nodes=tuple(value.get("root_nodes", ())),
            auto_self_sealed=value.get("auto_self_sealed", False),
            portable_targets=tuple(value.get("portable_targets", ())),
            portable_all_file_hashes=value.get("portable_all_file_hashes", False),
            builders=tuple(BuilderRule.from_dict(item) for item in value.get("builders", ())),
        )
        if (
            not result.manifest_id
            or not result.scan_paths
            or not isinstance(result.auto_self_sealed, bool)
            or not isinstance(result.portable_all_file_hashes, bool)
            or len(set(result.root_nodes)) != len(result.root_nodes)
            or len(set(result.portable_targets)) != len(result.portable_targets)
            or len({item.node for item in result.builders}) != len(result.builders)
        ):
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
    self_sealed: tuple[str, ...]


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
    root: Path,
    owner: str,
    value: Mapping[str, Any],
    aliases: Sequence[AliasRule],
    portable_targets: frozenset[str] = frozenset(),
    portable_all_file_hashes: bool = False,
) -> tuple[Binding, ...]:
    rows: dict[tuple[str | int, ...], Binding] = {}
    for pointer, mapping in _walk(value):
        if "path" in mapping:
            for hash_key in HASH_KEYS:
                hash_kind = hash_key
                target_value = mapping.get("path")
                if (
                    hash_key == "file_sha256"
                    and (
                        portable_all_file_hashes
                        or (
                            isinstance(target_value, str)
                            and target_value in portable_targets
                        )
                    )
                ):
                    hash_kind = "portable_file_sha256"
                item = _binding(root, owner, pointer, mapping, "path", hash_key, hash_kind)
                if item is not None:
                    rows[item.pointer] = item
        for key in mapping:
            if not isinstance(key, str) or not key.endswith("_path"):
                continue
            stem = key[:-5]
            for suffix in HASH_KEYS:
                hash_key = f"{stem}_{suffix}"
                hash_kind = suffix
                target_value = mapping.get(key)
                if (
                    suffix == "file_sha256"
                    and (
                        portable_all_file_hashes
                        or (
                            isinstance(target_value, str)
                            and target_value in portable_targets
                        )
                    )
                ):
                    hash_kind = "portable_file_sha256"
                item = _binding(root, owner, pointer, mapping, key, hash_key, hash_kind)
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


def _reference_targets(
    root: Path, value: Mapping[str, Any], aliases: Sequence[AliasRule]
) -> frozenset[str]:
    """Index existing receipt targets without requiring unrelated scanned nodes to be valid."""

    targets: set[str] = set()
    for _, mapping in _walk(value):
        path = mapping.get("path")
        if isinstance(path, str) and any(isinstance(mapping.get(key), str) for key in HASH_KEYS):
            candidate = _resolve(root, path)
            if candidate.is_file():
                targets.add(_portable(candidate.relative_to(root.resolve())))
        for key in mapping:
            if not isinstance(key, str) or not key.endswith("_path"):
                continue
            stem = key[:-5]
            if not any(isinstance(mapping.get(f"{stem}_{suffix}"), str) for suffix in HASH_KEYS):
                continue
            path = mapping.get(key)
            if isinstance(path, str):
                candidate = _resolve(root, path)
                if candidate.is_file():
                    targets.add(_portable(candidate.relative_to(root.resolve())))
        for alias in aliases:
            if alias.hash_key in mapping:
                candidate = _resolve(root, alias.target)
                if candidate.is_file():
                    targets.add(_portable(candidate.relative_to(root.resolve())))
    return frozenset(targets)


def _reverse_closure(
    root: Path,
    candidates: Sequence[str],
    roots: Sequence[str],
    aliases: Sequence[AliasRule],
) -> tuple[str, ...]:
    candidate_set = set(candidates)
    selected = set(roots)
    if not selected or not selected <= candidate_set:
        raise ReceiptDagError("receipt DAG root nodes must exist inside the scan set")
    references = {
        owner: _reference_targets(root, _load_json(_resolve(root, owner)), aliases)
        for owner in candidates
    }
    changed = True
    while changed:
        changed = False
        for owner in candidates:
            if owner not in selected and references[owner] & selected:
                selected.add(owner)
                changed = True
    return tuple(sorted(selected))


def discover_receipt_dag(root: Path, manifest: ReceiptDagManifest) -> ReceiptDag:
    root = root.resolve()
    for relative in manifest.portable_targets:
        if not _resolve(root, relative).is_file():
            raise ReceiptDagError(f"portable receipt target is missing: {relative}")
    scanned = _scan_nodes(root, manifest.scan_paths)
    nodes = (
        _reverse_closure(root, scanned, manifest.root_nodes, manifest.aliases)
        if manifest.root_nodes
        else scanned
    )
    portable_targets = frozenset(manifest.portable_targets)
    bindings = tuple(
        item
        for owner in nodes
        for item in _discover_bindings(
            root,
            owner,
            _load_json(_resolve(root, owner)),
            manifest.aliases,
            portable_targets,
            manifest.portable_all_file_hashes,
        )
    )
    self_sealed = set(manifest.self_sealed)
    if manifest.auto_self_sealed:
        self_sealed.update(
            owner
            for owner in nodes
            if isinstance(_load_json(_resolve(root, owner)).get("content_sha256"), str)
        )
    if not self_sealed <= set(nodes):
        raise ReceiptDagError("self-sealed files must be JSON nodes inside the scan set")
    if not {item.node for item in manifest.builders} <= set(nodes):
        raise ReceiptDagError("receipt DAG builder outputs must be JSON nodes inside the scan set")
    return ReceiptDag(
        root,
        manifest,
        nodes,
        bindings,
        _topological(nodes, bindings),
        tuple(sorted(self_sealed)),
    )


def _json_bytes(value: Mapping[str, Any], original: bytes | None = None) -> bytes:
    compact = original is not None and b"\n" not in original.strip()
    if compact:
        rendered = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    else:
        rendered = json.dumps(value, indent=2, ensure_ascii=True)
    return (rendered + "\n").encode("utf-8")


def _generic_content_sha(value: Mapping[str, Any], *, ensure_ascii: bool) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _content_hash_candidates(value: Mapping[str, Any]) -> dict[str, str]:
    rows = {
        "generic_ascii": _generic_content_sha(value, ensure_ascii=True),
        "generic_utf8": _generic_content_sha(value, ensure_ascii=False),
    }
    try:
        rows["sigma_core"] = canonical_sha256(value)
    except (TypeError, ValueError):
        pass
    return rows


def _root_seal_mode(value: Mapping[str, Any]) -> str:
    body = dict(value)
    declared = body.pop("content_sha256", None)
    matches = [
        mode for mode, candidate in _content_hash_candidates(body).items() if candidate == declared
    ]
    if not matches:
        candidates = _content_hash_candidates(body)
        return "sigma_core" if "sigma_core" in candidates else "generic_ascii"
    return "sigma_core" if "sigma_core" in matches else min(matches)


def _seal_with_mode(value: Mapping[str, Any], mode: str) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    try:
        return _content_hash_candidates(body)[mode]
    except KeyError as error:
        raise ReceiptDagError(f"root content-seal mode became unavailable: {mode}") from error


def expected_hash(
    root: Path,
    binding: Binding,
    virtual_values: Mapping[str, Mapping[str, Any]] | None = None,
    virtual_bytes: Mapping[str, bytes] | None = None,
) -> str:
    target = _resolve(root, binding.target)
    virtual = virtual_values.get(binding.target) if virtual_values is not None else None
    data = virtual_bytes.get(binding.target) if virtual_bytes is not None else None
    if binding.hash_kind == "file_sha256":
        return hashlib.sha256(data).hexdigest() if data is not None else _raw_sha(target)
    if binding.hash_kind == "portable_file_sha256":
        return (
            hashlib.sha256(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
            if data is not None
            else _portable_sha(target)
        )
    if binding.hash_kind == "content_sha256":
        value = virtual if virtual is not None else _load_json(target)
        content = value.get("content_sha256")
        return (
            content
            if isinstance(content, str)
            else _generic_content_sha(value, ensure_ascii=True)
        )
    raise ReceiptDagError(f"unsupported hash kind: {binding.hash_kind}")


def _root_seal(value: Mapping[str, Any]) -> str:
    return _seal_with_mode(value, _root_seal_mode(value))


def audit_receipt_dag(
    dag: ReceiptDag,
    virtual_values: Mapping[str, Mapping[str, Any]] | None = None,
    virtual_bytes: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    stale = []
    for binding in dag.bindings:
        expected = expected_hash(dag.root, binding, virtual_values, virtual_bytes)
        declared = binding.declared
        if virtual_values is not None:
            current: Any = virtual_values[binding.owner]
            for part in binding.pointer:
                current = current[part]
            declared = current
        if declared != expected:
            stale.append(
                {
                    "declared": declared,
                    "expected": expected,
                    "hash_kind": binding.hash_kind,
                    "owner": binding.owner,
                    "pointer": binding.pointer_text,
                    "target": binding.target,
                }
            )
    bad_seals = []
    for relative in dag.self_sealed:
        value = (
            virtual_values[relative]
            if virtual_values is not None
            else _load_json(_resolve(dag.root, relative))
        )
        declared = value.get("content_sha256")
        body = dict(value)
        body.pop("content_sha256", None)
        candidates = _content_hash_candidates(body)
        preferred_mode = "sigma_core" if "sigma_core" in candidates else "generic_ascii"
        expected = candidates[preferred_mode]
        if declared not in set(candidates.values()):
            bad_seals.append({"declared": declared, "expected": expected, "owner": relative})
    body = {
        "schema_version": AUDIT_SCHEMA,
        "manifest_id": dag.manifest.manifest_id,
        "counts": {
            "bindings": len(dag.bindings),
            "json_nodes": len(dag.json_nodes),
            "self_sealed": len(dag.self_sealed),
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
    values = {
        owner: _load_json(_resolve(dag.root, owner)) for owner in dag.json_nodes
    }
    virtual_bytes = {
        owner: _resolve(dag.root, owner).read_bytes() for owner in dag.json_nodes
    }
    seal_modes = {owner: _root_seal_mode(values[owner]) for owner in dag.self_sealed}
    bindings_by_owner = {
        owner: tuple(item for item in dag.bindings if item.owner == owner) for owner in dag.json_nodes
    }
    for owner in dag.order:
        value = values[owner]
        owner_changes = []
        for binding in bindings_by_owner[owner]:
            expected = expected_hash(dag.root, binding, values, virtual_bytes)
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
        if owner in dag.self_sealed:
            expected = _seal_with_mode(value, seal_modes[owner])
            current = value.get("content_sha256")
            if current != expected:
                value["content_sha256"] = expected
                owner_changes.append(
                    {"from": current, "pointer": "/content_sha256", "target": owner, "to": expected}
                )
        if owner_changes:
            changes.append({"changes": owner_changes, "owner": owner})
            virtual_bytes[owner] = _json_bytes(value, virtual_bytes[owner])
    projected_audit = audit_receipt_dag(dag, values, virtual_bytes)
    if not projected_audit["valid"]:
        raise ReceiptDagError("receipt DAG projection remained stale after topological reseal")
    if write:
        changed_owners = {item["owner"] for item in changes}
        for owner in dag.order:
            if owner in changed_owners:
                _resolve(dag.root, owner).write_bytes(virtual_bytes[owner])
        refreshed = discover_receipt_dag(dag.root, dag.manifest)
        audit = audit_receipt_dag(refreshed)
        if not audit["valid"]:
            raise ReceiptDagError("receipt DAG remained stale after a topological reseal")
    else:
        audit = audit_receipt_dag(dag)
    return {
        "audit": audit,
        "changes": changes,
        "projected_audit": projected_audit,
        "write": write,
    }


def _owner_is_stale(audit: Mapping[str, Any], owner: str) -> bool:
    return any(item["owner"] == owner for item in audit["stale_bindings"]) or any(
        item["owner"] == owner for item in audit["stale_self_seals"]
    )


def _run_builder(root: Path, rule: BuilderRule) -> dict[str, Any]:
    output = _resolve(root, rule.node)
    before = hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None
    environment = os.environ.copy()
    source_path = str(root / "src")
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing_path else source_path + os.pathsep + existing_path
    )
    command = [sys.executable, "-m", rule.module, *rule.arguments]

    def execute() -> tuple[str, str]:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=rule.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ReceiptDagError(f"receipt builder timed out: {rule.node}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-2000:]
            raise ReceiptDagError(f"receipt builder failed for {rule.node}: {detail}")
        if not output.is_file():
            raise ReceiptDagError(f"receipt builder did not create its declared node: {rule.node}")
        return completed.stdout[-2000:], completed.stderr[-2000:]

    stdout, stderr = execute()
    first = output.read_bytes()
    if rule.verify_determinism:
        execute()
        second = output.read_bytes()
        if first != second:
            raise ReceiptDagError(f"receipt builder is nondeterministic: {rule.node}")
    after = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "after_file_sha256": after,
        "before_file_sha256": before,
        "changed": before != after,
        "module": rule.module,
        "node": rule.node,
        "stderr_tail": stderr,
        "stdout_tail": stdout,
        "verified_deterministic": rule.verify_determinism,
    }


def rebuild_receipt_dag(dag: ReceiptDag, *, force_builders: bool = False) -> dict[str, Any]:
    """Run stale generators dependency-first, then reseal non-generated receipt edges."""

    root = dag.root
    rules = {item.node: item for item in dag.manifest.builders}
    builder_events = []
    for owner in dag.order:
        rule = rules.get(owner)
        if rule is None:
            continue
        current = discover_receipt_dag(root, dag.manifest)
        current_audit = audit_receipt_dag(current)
        if not force_builders and not _owner_is_stale(current_audit, owner):
            continue
        builder_events.append(_run_builder(root, rule))
        refreshed = discover_receipt_dag(root, dag.manifest)
        refreshed_audit = audit_receipt_dag(refreshed)
        if _owner_is_stale(refreshed_audit, owner):
            raise ReceiptDagError(f"receipt builder left its output stale: {owner}")
    current = discover_receipt_dag(root, dag.manifest)
    reseal = reseal_receipt_dag(current, write=True)
    return {
        "audit": reseal["audit"],
        "builder_events": builder_events,
        "changes": reseal["changes"],
        "force_builders": force_builders,
        "projected_audit": reseal["projected_audit"],
        "write": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--rebuild-all", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    manifest = load_manifest(root, args.manifest)
    dag = discover_receipt_dag(root, manifest)
    if args.write:
        result = rebuild_receipt_dag(dag, force_builders=args.rebuild_all)
    else:
        if args.rebuild_all:
            raise ReceiptDagError("--rebuild-all requires --write")
        result = {"audit": audit_receipt_dag(dag), "changes": [], "write": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["audit"]["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
