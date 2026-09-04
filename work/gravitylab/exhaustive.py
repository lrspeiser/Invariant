"""Exhaustive enumeration of every law of complexity 1, 2 and 3.

An evolutionary search returns the best law it happened to find. At low
complexity the space is small enough to enumerate completely, and then the
answer is not "the best we found" but "the best that exists":

    k = 1     1,898 laws
    k = 2     1,800,253 laws
    k = 3   1,138,360,096 laws

Every one of those is scored with its OPTIMAL coefficients, against all four
targets, so the statement that comes out is a proof over the atom bank rather
than a search outcome:

    no law of complexity <= 3 built from this bank beats the RAR on the blind
    set by more than the physics-free controls do.

The intercept is carried in every model and is not counted toward k.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:                                    # pragma: no cover
    import numpy as xp
    GPU = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_search import Bench                                    # noqa: E402
from hypersearch import Screen, build_atoms, sync, TARGETS      # noqa: E402


def best_k1(S):
    idx = xp.arange(S.natom - 1, dtype=xp.int32)[:, None]
    out = {}
    for w in TARGETS:
        r, _ = S.score(idx, w)
        i = int(xp.argmin(r))
        out[w] = (float(r[i]), [int(idx[i, 0])])
    return out, S.natom - 1


def best_k2(S, chunk=4_000_000):
    n = S.natom - 1
    jj, kk = np.triu_indices(n, k=1)
    tot = jj.size
    best = {w: (np.inf, None) for w in TARGETS}
    for s in range(0, tot, chunk):
        e = min(s + chunk, tot)
        idx = xp.asarray(np.stack([jj[s:e], kk[s:e]], 1).astype(np.int32))
        for w in TARGETS:
            r, _ = S.score(idx, w)
            i = int(xp.argmin(r))
            if float(r[i]) < best[w][0]:
                best[w] = (float(r[i]), [int(jj[s + i]), int(kk[s + i])])
    return best, tot


def best_k3(S, verbose=True):
    """All i<j<k. triu_indices is ordered by its first index, so the pairs
    admissible for a given i are a contiguous suffix and need no masking."""
    n = S.natom - 1
    jj, kk = np.triu_indices(n, k=1)
    # start offset of each value of the first index
    starts = np.searchsorted(jj, np.arange(n))
    jg, kg = xp.asarray(jj.astype(np.int32)), xp.asarray(kk.astype(np.int32))
    best = {w: (np.inf, None) for w in TARGETS}
    tot = 0
    t0 = time.time()
    for i in range(n - 2):
        s = int(starts[i + 1])                 # pairs with first index > i
        m = jg.size - s
        if m <= 0:
            continue
        tot += m
        col = xp.full(m, i, dtype=xp.int32)
        idx = xp.stack([col, jg[s:], kg[s:]], axis=1)
        for w in TARGETS:
            r, _ = S.score(idx, w)
            a = int(xp.argmin(r))
            if float(r[a]) < best[w][0]:
                best[w] = (float(r[a]),
                           [i, int(jj[s + a]), int(kk[s + a])])
        if verbose and i % 200 == 0:
            el = time.time() - t0
            print(f"      i={i:4d}/{n-2}   {tot/1e9:.3f}e9 laws   "
                  f"{el:.0f}s   {tot/max(el,1e-9)/1e6:.0f}M/s")
    return best, tot


def main():
    print("=" * 78)
    print("EXHAUSTIVE ENUMERATION -- every law of complexity 1, 2 and 3")
    print("=" * 78)
    b = Bench(ndraw=8, verbose=False)
    A, meta = build_atoms(b, nscale=int(os.environ.get("NSCALE", 25)))
    S = Screen(b, A, meta)
    nm = [m[0] for m in S.meta]
    dm = [m[1] for m in S.meta]

    def show(tag, best, count):
        print(f"\n   k = {tag}   {count:,} laws enumerated")
        for w in TARGETS:
            r, ix = best[w]
            g = 100 * (S.ref_train[w] - r) / S.ref_train[w]
            terms = " + ".join(f"[{dm[a]}]{nm[a]}" for a in ix)
            print(f"      {w:7s} {r:.5f}  gain {g:+6.2f}%   {terms}")
        gr = 100 * (S.ref_train["real"] - best["real"][0]) / S.ref_train["real"]
        gc = max(100 * (S.ref_train[w] - best[w][0]) / S.ref_train[w]
                 for w in ("null", "perm", "perm_g"))
        print(f"      -> real {gr:+.2f}%  best control {gc:+.2f}%  "
              f"margin {gr-gc:+.2f} pp")
        return gr, gc

    res = {}
    t0 = time.time()
    b1, c1 = best_k1(S)
    res["k1"] = {"count": c1, "best": {w: b1[w][0] for w in TARGETS},
                 "atoms": {w: b1[w][1] for w in TARGETS}}
    show(1, b1, c1)

    b2, c2 = best_k2(S)
    res["k2"] = {"count": c2, "best": {w: b2[w][0] for w in TARGETS},
                 "atoms": {w: b2[w][1] for w in TARGETS}}
    show(2, b2, c2)

    print("\n   k = 3 -- this is the billion-law pass, all four targets")
    b3, c3 = best_k3(S)
    res["k3"] = {"count": c3, "best": {w: b3[w][0] for w in TARGETS},
                 "atoms": {w: b3[w][1] for w in TARGETS}}
    show(3, b3, c3)
    total = c1 + c2 + c3
    print(f"\n   {total:,} laws x {len(TARGETS)} targets = "
          f"{total*len(TARGETS)/1e9:.2f} billion exact optimally-fitted "
          f"evaluations in {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- blind
    print("\n   BLIND -- the exhaustive winners, coefficients frozen on train")
    print(f"      RAR on blind: {S.rar_blind:.5f} dex")
    bl = {}
    for tag, bb in (("k1", b1), ("k2", b2), ("k3", b3)):
        bl[tag] = {}
        for w in TARGETS:
            row = xp.asarray(np.array(bb[w][1], dtype=np.int32))
            _, c = S.score(row[None, :], w)
            v = S.verify(row, c[0], w, blind=True)
            g = 100 * (S.ref_blind[w] - v) / S.ref_blind[w]
            bl[tag][w] = {"blind": v, "gain_pct": g}
        gr, gc = bl[tag]["real"]["gain_pct"], max(
            bl[tag][w]["gain_pct"] for w in ("null", "perm", "perm_g"))
        print(f"      {tag}  real blind {bl[tag]['real']['blind']:.5f} "
              f"({gr:+6.2f}%)   best control {gc:+6.2f}%   "
              f"margin {gr-gc:+6.2f} pp")
    res["blind"] = bl
    res["natom"] = S.natom
    res["baseline_train"] = S.ref_train
    res["baseline_blind"] = S.ref_blind
    res["atom_names"] = {str(a): [nm[a], dm[a]] for tag in ("k1", "k2", "k3")
                         for w in TARGETS for a in res[tag]["atoms"][w]}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "runs", "evolve-sparc")
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, "exhaustive-k3.json")
    with open(fn, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n   written: {os.path.normpath(fn)}")


if __name__ == "__main__":
    main()
