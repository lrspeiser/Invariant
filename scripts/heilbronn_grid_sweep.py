"""Run a complete A248866 sweep in parallel, one process per ``(first, second)`` prefix.

This is the driver that settles the expensive half of ``a(n) = T``: that *no* ``n``-subset of
the ``n x n`` grid reaches ``T + 1``.  It owns no mathematics -- every branch is decided by
:func:`~sigma_theory_compiler.discrete_heilbronn_grid.branch_verdict`, and the branch list
comes from :func:`~sigma_theory_compiler.discrete_heilbronn_grid.enumerate_branches`, so what
counts as "complete" is defined in the library and merely obeyed here.

Two operational properties matter for a run measured in hours.  Every finished branch is
appended to a ledger, so a killed run resumes instead of restarting.  And the report states
how many branches were decided against how many exist: a sweep that does not close that gap
has not proved anything, and says so.

    python scripts/heilbronn_grid_sweep.py --n 14 --threshold 7 --workers 22
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigma_theory_compiler.discrete_heilbronn_grid import (
    branch_verdict,
    enumerate_branches,
)
from sigma_theory_compiler.sigma_core import canonical_sha256


def _decide(n: int, threshold: int, prefix: tuple[int, int]) -> dict:
    verdict = branch_verdict(n, threshold, prefix)
    return {
        "prefix": list(prefix),
        "feasible": verdict.feasible,
        "nodes": verdict.nodes,
        "witness": None if verdict.witness is None else [list(p) for p in verdict.witness],
    }


def _load_ledger(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    done: dict[tuple[int, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            done[tuple(record["prefix"])] = record
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    branches = enumerate_branches(args.n)
    ledger = args.ledger or Path(f"grid-sweep-n{args.n}-T{args.threshold}.jsonl")
    done = _load_ledger(ledger)
    todo = [b for b in branches if b not in done]
    print(
        f"n={args.n} threshold={args.threshold} branches={len(branches)} "
        f"resumed={len(done)} todo={len(todo)} workers={args.workers}",
        flush=True,
    )

    started = time.monotonic()
    nodes = sum(r["nodes"] for r in done.values())
    witness = next((r["witness"] for r in done.values() if r["feasible"]), None)
    with (
        ledger.open("a", encoding="utf-8") as handle,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {pool.submit(_decide, args.n, args.threshold, b): b for b in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            done[tuple(record["prefix"])] = record
            nodes += record["nodes"]
            if record["feasible"] and witness is None:
                witness = record["witness"]
            if index % 500 == 0:
                print(
                    f"  decided {len(done)}/{len(branches)} branches, "
                    f"nodes={nodes / 1e9:.3f}G, t={time.monotonic() - started:.0f}s",
                    flush=True,
                )

    complete = len(done) == len(branches)
    payload = {
        "problem": "discrete_heilbronn_grid_min_double_area",
        "n": args.n,
        "threshold": args.threshold,
        "branches_total": len(branches),
        "branches_decided": len(done),
        "sweep_complete": complete,
        "feasible": witness is not None,
        "search_nodes": nodes,
        "wall_seconds": f"{time.monotonic() - started:.1f}",
        "witness": witness,
        "verdict": (
            "no such configuration exists"
            if complete and witness is None
            else "configuration exhibited"
            if witness is not None
            else "INCOMPLETE - proves nothing"
        ),
        "absence_establishes_novelty": False,
    }
    payload["digest"] = canonical_sha256(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
