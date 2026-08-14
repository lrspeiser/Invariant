from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

MAX_BOUND_FILE_BYTES = 64 * 1024 * 1024


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"bound path escapes project root: {relative}")
    return path


def _bindings(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        path = value.get("path")
        file_sha = value.get("file_sha256")
        if isinstance(path, str) and isinstance(file_sha, str) and len(file_sha) == 64:
            found.append((path, file_sha))
        # Some active runtime configs use flat named pairs rather than a nested
        # binding object, for example ``formal_controls_path`` together with
        # ``formal_controls_file_sha256``.  Treat those as first-class bindings
        # so live config contracts take precedence over later historical
        # artifact registrations for the same semantic file.
        for key, candidate_path in value.items():
            if not key.endswith("_path") or not isinstance(candidate_path, str):
                continue
            prefix = key.removesuffix("_path")
            for digest_key in (f"{prefix}_file_sha256", f"{prefix}_sha256"):
                digest = value.get(digest_key)
                if isinstance(digest, str) and len(digest) == 64:
                    found.append((candidate_path, digest))
                    break
        for child in value.values():
            found.extend(_bindings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_bindings(child))
    return found


def _registered_bytes(raw: bytes, expected_sha256: str) -> bytes | None:
    if _sha(raw) == expected_sha256:
        return raw
    if b"\x00" in raw:
        return None
    lf = raw.replace(b"\r\n", b"\n")
    candidates = (lf, lf.replace(b"\n", b"\r\n"))
    for candidate in candidates:
        if _sha(candidate) == expected_sha256:
            return candidate
    return None


def _materialize_pass(root: Path, config_path: Path) -> dict[str, int]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    queue = deque(_bindings(config.get("sources", [])))
    config_documents_scanned = 0
    # Some deterministic validators load campaign configs directly instead of
    # reaching them through an immutable source artifact.  Include those
    # registered bindings so a clean LF checkout can reproduce the exact CRLF
    # bytes sealed by the Windows-authored provenance records.
    for candidate in sorted((root / "configs").rglob("*.json")):
        if candidate.stat().st_size > MAX_BOUND_FILE_BYTES:
            raise ValueError(
                f"config exceeds bootstrap byte limit: {candidate.relative_to(root).as_posix()}"
            )
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        config_documents_scanned += 1
        queue.extend(_bindings(document))
    visited: set[tuple[str, str]] = set()
    selected_by_path: dict[str, str] = {}
    rewritten = 0
    matched = 0
    missing = 0
    superseded = 0
    while queue:
        relative, expected = queue.popleft()
        key = (relative, expected)
        if key in visited:
            continue
        visited.add(key)
        path = _inside(root, relative)
        if not path.is_file():
            missing += 1
            continue
        if path.stat().st_size > MAX_BOUND_FILE_BYTES:
            raise ValueError(f"bound file exceeds bootstrap byte limit: {relative}")
        raw = path.read_bytes()
        registered = _registered_bytes(raw, expected)
        if registered is None:
            # Reachable immutable artifacts may retain an older hash for a path whose
            # current authoritative source is registered elsewhere in the same graph.
            # Leave those bytes untouched; the semantic validators decide whether a
            # superseded binding is admissible for the artifact that contains it.
            superseded += 1
            continue
        selected = selected_by_path.get(relative)
        if selected is not None and selected != expected:
            # Multiple historical receipts can legitimately bind opposite line-ending
            # representations of the same semantic file.  Let the first reachable,
            # currently materializable registration claim the path for this pass.
            # Without this guard, later queue entries can toggle the file back and
            # make fixed-point materialization oscillate forever.
            superseded += 1
            continue
        selected_by_path[relative] = expected
        matched += 1
        if registered != raw:
            path.write_bytes(registered)
            rewritten += 1
        if path.suffix.lower() == ".json":
            try:
                document = json.loads(registered.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            queue.extend(_bindings(document))
    return {
        "bindings_visited": len(visited),
        "config_documents_scanned": config_documents_scanned,
        "files_matched": matched,
        "files_rewritten": rewritten,
        "missing_bound_paths": missing,
        "superseded_bindings_skipped": superseded,
    }


def materialize(root: Path, config_path: Path) -> dict[str, int]:
    """Materialize the transitive registered-byte closure to a fixed point.

    Some immutable JSON artifacts are themselves line-ending-normalized during
    the first pass. Their nested source bindings are only visible after those
    registered bytes have been restored, so repeat bounded full passes until a
    pass performs no writes. Every successful rewrite changes a finite checked
    file to one of its registered byte forms; the cap guards against cycles or
    conflicting live registrations.
    """

    aggregate = {
        "bindings_visited": 0,
        "config_documents_scanned": 0,
        "files_matched": 0,
        "files_rewritten": 0,
        "missing_bound_paths": 0,
        "superseded_bindings_skipped": 0,
        "materialization_passes": 0,
    }
    for pass_number in range(1, 9):
        current = _materialize_pass(root, config_path)
        aggregate["materialization_passes"] = pass_number
        for key, value in current.items():
            aggregate[key] += value
        if current["files_rewritten"] == 0:
            return aggregate
    raise ValueError("registered-byte materialization did not converge within 8 passes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/unified_engine_status.json")
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    result = materialize(root, _inside(root, arguments.config))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
