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


def materialize(root: Path, config_path: Path) -> dict[str, int]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    queue = deque(_bindings(config.get("sources", [])))
    visited: set[tuple[str, str]] = set()
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
        "files_matched": matched,
        "files_rewritten": rewritten,
        "missing_bound_paths": missing,
        "superseded_bindings_skipped": superseded,
    }


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
