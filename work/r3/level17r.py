"""Resumable single-level exhaustion for A345731(n) at a fixed diameter bound L.

Each finished cell is appended to a per-cell log immediately, so a killed run
resumes instead of restarting.  The level counts as settled only when the log
holds one completed, non-aborted record for every declared cell.
"""
from __future__ import annotations
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, r"C:\Users\henry\Documents\Codex\2026-08-06\for\wt-exh\src")
from sigma_theory_compiler.weak_sidon_exhaustion import (
    A345731_PUBLISHED, enumerate_prefix_cells, is_weak_sidon, search_within, witness_diameter,
)

def main(n, L, workers, cell_log, out):
    w = [0, 0] + [A345731_PUBLISHED[m] for m in range(2, 17)] + [L]
    w = w[: n + 1] if n <= 16 else w
    cells = enumerate_prefix_cells(n, L, w, depth=3)
    done = {}
    if os.path.exists(cell_log):
        for line in open(cell_log, encoding="utf-8"):
            r = json.loads(line)
            done[(int(r["a1"]), int(r["a2"]), int(r["a3"]))] = int(r["nodes"])
    todo = [c for c in cells if (c.a1, c.a2, c.a3) not in done]
    print("n=%d L=%d declared=%d already_done=%d todo=%d"
          % (n, L, len(cells), len(done), len(todo))); sys.stdout.flush()
    t0 = time.time(); hits = []; lock_fh = open(cell_log, "a", encoding="utf-8")

    def job(c):
        f, nd, wit = search_within(n, L, w, a1_range=(c.a1, c.a1), a2_range=(c.a2, c.a2),
                                   a3_range=(c.a3, c.a3), engine="numba")
        return c, f, nd, wit

    k = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fu in as_completed([ex.submit(job, c) for c in todo]):
            c, f, nd, wit = fu.result(); k += 1
            done[(c.a1, c.a2, c.a3)] = nd
            lock_fh.write(json.dumps({"a1": str(c.a1), "a2": str(c.a2), "a3": str(c.a3),
                                      "nodes": str(nd), "found": bool(f)}) + "\n")
            lock_fh.flush(); os.fsync(lock_fh.fileno())
            if f:
                assert is_weak_sidon(wit) and witness_diameter(wit) <= L
                hits.append(list(map(int, wit)))
            if k % 50 == 0:
                print("  [%d/%d] %.0fs" % (len(done), len(cells), time.time() - t0)); sys.stdout.flush()
    lock_fh.close()
    aborted = [c for c, v in done.items() if v < 0]
    rec = {"n": str(n), "lmax": str(L), "wtab": [str(v) for v in w],
           "declared_cells": str(len(cells)), "traversed_cells": str(len(done)),
           "nodes": str(sum(abs(v) for v in done.values())),
           "exhaustive": not aborted and len(done) == len(cells),
           "found": bool(hits), "witness": [str(v) for v in (hits[0] if hits else [])],
           "seconds": str(int(time.time() - t0))}
    print(json.dumps(rec, sort_keys=True))
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
