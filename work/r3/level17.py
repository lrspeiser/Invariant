"""Exhaust the A345731(17) space at a single L, cell by cell, with live progress."""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, r"C:\Users\henry\Documents\Codex\2026-08-06\for\wt-exh\src")
from sigma_theory_compiler.weak_sidon_exhaustion import (
    A345731_PUBLISHED, enumerate_prefix_cells, is_weak_sidon, search_within, witness_diameter,
)

def main(L, workers, out):
    n = 17
    w = [0, 0] + [A345731_PUBLISHED[m] for m in range(2, 17)] + [L]
    cells = enumerate_prefix_cells(n, L, w, depth=3)
    print("n=17 L=%d declared_cells=%d wtab=%s" % (L, len(cells), w)); sys.stdout.flush()
    t0 = time.time(); done = 0; total = 0; hits = []
    def job(c):
        f, nd, wit = search_within(n, L, w, a1_range=(c.a1, c.a1), a2_range=(c.a2, c.a2),
                                   a3_range=(c.a3, c.a3), engine="numba")
        return c, f, nd, wit
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, c) for c in cells]
        for fu in as_completed(futs):
            c, f, nd, wit = fu.result(); done += 1; total += abs(nd)
            if f:
                assert is_weak_sidon(wit) and witness_diameter(wit) <= L
                hits.append(list(map(int, wit)))
            if done % 25 == 0 or f:
                print("  [%d/%d] total_nodes=%d %.0fs" % (done, len(cells), total, time.time()-t0)); sys.stdout.flush()
    rec = {"n":"17","lmax":str(L),"wtab":[str(v) for v in w],
           "declared_cells":str(len(cells)),"traversed_cells":str(done),
           "nodes":str(total),"exhaustive":True,"found":bool(hits),
           "witness":[str(v) for v in (hits[0] if hits else [])],"seconds":str(int(time.time()-t0))}
    print(json.dumps(rec, sort_keys=True))
    with open(out,"a",encoding="utf-8") as fh: fh.write(json.dumps(rec,sort_keys=True)+"\n")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
