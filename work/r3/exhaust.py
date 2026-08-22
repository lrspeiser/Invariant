"""Parallel, certificate-emitting exhaustion of the A345731 ladder.

For a given (n, L) the declared space is partitioned by the pair (a_1, a_2).
Every cell is dispatched; each cell traversal is complete.  A level that
finishes with declared_cells == traversed_cells, every cell exhaustive, and no
witness proves A345731(n) > L.

The chain is self-contained: wtab[m] used for pruning at stage n comes from
this run's own verified stages m < n, never from a published table.
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
    min_diameter_lower_bound,
    search_within,
    witness_diameter,
)


def run_level(n: int, L: int, wtab: list[int], workers: int, out) -> dict:
    w = list(wtab)
    w[n] = L                      # every smaller diameter already ruled out
    partitioned = n >= 6
    cells = enumerate_prefix_cells(n, L, w) if partitioned else [None]
    t0 = time.time()

    def job(cell):
        kw = {}
        if cell is not None:
            kw = {"a1_range": (cell.a1, cell.a1), "a2_range": (cell.a2, cell.a2)}
        found, nodes, wit = search_within(n, L, w, engine="numba", **kw)
        return nodes, (wit if found else ())

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(job, cells):
            results.append(r)

    aborted = [r for r in results if r[0] < 0]
    hits = [r for r in results if r[1]]
    rec = {
        "n": str(n),
        "lmax": str(L),
        "wtab": [str(v) for v in w[: n + 1]],
        "declared_cells": str(len(cells)),
        "partitioned": partitioned,
        "traversed_cells": str(len(results)),
        "nodes": str(sum(abs(r[0]) for r in results)),
        "exhaustive": not aborted,
        "found": bool(hits),
        "witness": [str(v) for v in (hits[0][1] if hits else ())],
        "seconds": str(int(time.time() - t0)),
    }
    if hits:
        wit = hits[0][1]
        assert is_weak_sidon(wit), wit
        assert witness_diameter(wit) == L, (wit, L)
    print(json.dumps(rec, sort_keys=True))
    sys.stdout.flush()
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def chain(nmax: int, workers: int, out, nmin: int = 2) -> list[int]:
    w = [0] * (nmax + 3)
    for n in range(2, nmax + 1):
        L = max(w[n - 1], min_diameter_lower_bound(n))
        while True:
            rec = run_level(n, L, w, workers, out)
            if not rec["exhaustive"]:
                print("stage n=%d L=%d not exhaustive; stopping" % (n, L))
                return w
            if rec["found"]:
                w[n] = L
                pub = A345731_PUBLISHED.get(n)
                print("VERIFIED A345731(%d) = %d  (published %s) %s"
                      % (n, L, pub, "MATCH" if pub == L else "*** DISAGREE ***"))
                sys.stdout.flush()
                break
            L += 1
    return w


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    out = sys.argv[3] if len(sys.argv) > 3 else None
    chain(nmax, workers, out)
