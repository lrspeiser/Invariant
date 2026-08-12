"""Collect one pytest file deterministically and execute one contiguous shard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    return parser.parse_args()


def _collect_nodeids(test_file: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", test_file],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = f"{Path(test_file).as_posix()}::"
    nodeids = [
        line.strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip().replace("\\", "/").startswith(prefix)
    ]
    if not nodeids:
        raise RuntimeError(f"pytest collected no node ids from {test_file}")
    return nodeids


def _select_nodeids(nodeids: list[str], shard_index: int, shard_count: int) -> list[str]:
    """Return a balanced contiguous interval, keeping neighboring tamper cases together."""
    start = len(nodeids) * shard_index // shard_count
    stop = len(nodeids) * (shard_index + 1) // shard_count
    return nodeids[start:stop]


def main() -> int:
    args = _parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard count)")
    nodeids = _collect_nodeids(args.test_file)
    selected = _select_nodeids(nodeids, args.shard_index, args.shard_count)
    if not selected:
        raise RuntimeError(f"pytest shard {args.shard_index}/{args.shard_count} selected no tests")
    print(
        f"pytest shard {args.shard_index + 1}/{args.shard_count}: "
        f"{len(selected)} of {len(nodeids)} tests",
        flush=True,
    )
    for nodeid in selected:
        print(nodeid, flush=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *selected, "-vv", "--durations=20"],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
