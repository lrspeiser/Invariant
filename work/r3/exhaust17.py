"""Parallel exhaustion of the A345731(17) search space, one L at a time.

For each L the declared space is partitioned by the pair (a_1, a_2); every cell
is dispatched to a worker and the traversal of each cell is complete.  A run
that finishes with declared_cells == traversed_cells and no witness proves
A345731(17) > L.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\henry\Documents\Codex\2026-08-06\for\wt-exh\src")

from sigma_theory_compiler.weak_sidon_exhaustion import (  # noqa: E402
    A345731_PUBLISHED,
    enumerate_prefix_cells,
    is_weak_sidon,
    search_within,
    witness_diameter,
)

N = 17


def wtab_for(lower_bound_17: int) -> list[int]:
    w = [0, 0]
    for m in range(2, 17):
        w.append(A345731_PUBLISHED[m])
    w.append(lower_bound_17)
    return w


def run_level(L: int, workers: int = 24, out=None) -> dict:
    w = wtab_for(L)
    cells = enumerate_prefix_cells(N, L, w)
    declared = len(cells)
    t0 = time.time()
    results: list[tuple[int, int, int, tuple]] = []

    def job(cell):
        found, nodes, wit = search_within(
            N, L, w,
            a1_range=(cell.a1, cell.a1),
            a2_range=(cell.a2, cell.a2),
            engine="numba",
        )
        return cell.a1, cell.a2, nodes, (wit if found else ())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(job, cells):
            results.append(r)

    total_nodes = sum(abs(r[2]) for r in results)
    aborted = [r for r in results if r[2] < 0]
    hits = [r for r in results if r[3]]
    rec = {
        "n": str(N),
        "lmax": str(L),
        "wtab": [str(v) for v in w],
        "declared_cells": str(declared),
        "traversed_cells": str(len(results)),
        "nodes": str(total_nodes),
        "exhaustive": not aborted,
        "found": bool(hits),
        "witness": [str(v) for v in (hits[0][3] if hits else ())],
        "seconds": str(int(time.time() - t0)),
    }
    if hits:
        wit = hits[0][3]
        assert is_weak_sidon(wit), wit
        rec["witness_diameter"] = str(witness_diameter(wit))
    print(json.dumps(rec, sort_keys=True))
    sys.stdout.flush()
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 149
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 170
    wk = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out = sys.argv[4] if len(sys.argv) > 4 else None
    for L in range(lo, hi + 1):
        rec = run_level(L, workers=wk, out=out)
        if rec["found"]:
            print("A345731(17) = %d" % L)
            break
        if not rec["exhaustive"]:
            print("level %d NOT exhaustive; stopping" % L)
            break
