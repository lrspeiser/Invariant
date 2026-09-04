"""Evolutionary structure search: propose, score, prune, mutate, repeat.

The loop is deliberately run THREE TIMES with identical settings, on three
targets that differ only in what physics they contain:

    real   log10(g_obs / g_bar) measured
    null   RAR + Gaussian noise matched to the real residual scatter
    perm   RAR + the real residuals permuted across galaxies

`null` and `perm` contain no gravitational information beyond the RAR that was
put into them by construction. Whatever gain the loop extracts from them is the
gain the SEARCH MACHINERY manufactures at this scale, on this bench, at this
complexity. The only claim the real run can make is the amount by which it beats
that, and this programme has already seen a case where the physics-free world
won (6.1% against 1.1%), so the control is not a formality.

Each generation:
    score      exact optimal coefficients from the precomputed atom Gram
    penalise   + lambda * (number of distinct atoms)
    prune      truncation selection, keep the top `keep` fraction
    breed      uniform crossover between two survivors
    mutate     global jumps, local walks along the parameter grid, and
               collapses that delete an atom by aliasing it to the intercept

Blind is touched once, at the very end, by the shortlist. KiDS and wide binaries
are sealed and are not in this module at all.
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
from gpu_search import Bench                            # noqa: E402
from hypersearch import Screen, build_atoms, sync       # noqa: E402


def _rng(seed):
    return xp.random.RandomState(seed) if GPU else np.random.RandomState(seed)


def n_distinct(idx, intercept):
    """Distinct non-intercept atoms per row: the complexity of the law."""
    s = xp.sort(idx, axis=1)
    d = xp.ones(s.shape, dtype=xp.float32)
    d[:, 1:] = (s[:, 1:] != s[:, :-1]).astype(xp.float32)
    d = d * (s != intercept).astype(xp.float32)
    return d.sum(axis=1)


def score_chunked(S, idx, which, blind=False, chunk=800_000):
    out = xp.empty(idx.shape[0], dtype=xp.float32)
    for i in range(0, idx.shape[0], chunk):
        r, _ = S.score(idx[i:i + chunk], which, blind=blind)
        out[i:i + chunk] = r
    return out


def evolve(S, which="real", K=6, pop=2_000_000, gens=120, keep=0.10,
           lam=0.002, seed=0, verbose=True, log=None):
    """One full evolutionary run against one target. Returns the elite pool."""
    natom = S.natom
    intercept = natom - 1
    rng = _rng(seed)
    idx = rng.randint(0, natom, size=(pop, K)).astype(xp.int32)
    nkeep = max(int(pop * keep), 1024)
    nhead = max(nkeep // 64, 64)
    nimm = max(int(pop * 0.05), 1024)
    hist = []
    hof = {}                       # key -> (rms, cpx, row) over ALL generations
    t0 = time.time()
    for g in range(gens):
        rms = score_chunked(S, idx, which)
        cpx = n_distinct(idx, intercept)
        fit = rms + lam * cpx
        order = xp.argsort(fit)[:nkeep]
        elite = idx[order]
        erms, ecpx = rms[order], cpx[order]
        bestk = {}
        for kk in range(1, K + 1):
            m = cpx == kk
            if bool(m.any()):
                bestk[kk] = float(rms[m].min())
        hist.append((g, float(erms[0]), float(ecpx[0]), float(fit[order[0]]),
                     bestk))
        if g % 5 == 0 or g == gens - 1:
            h = min(600, nkeep)
            rowsn = (elite[:h].get() if GPU else elite[:h])
            rn = (erms[:h].get() if GPU else erms[:h])
            cn = (ecpx[:h].get() if GPU else ecpx[:h])
            for t in range(h):
                key = tuple(sorted({int(v) for v in rowsn[t]} - {intercept}))
                if not key:
                    continue
                if key not in hof or rn[t] < hof[key][0]:
                    hof[key] = (float(rn[t]), int(cn[t]), rowsn[t].copy())
        if verbose and (g % 20 == 0 or g == gens - 1):
            print(f"      gen {g:3d}  best {float(erms[0]):.5f} dex "
                  f"(k={int(ecpx[0])})   elite median "
                  f"{float(xp.median(erms)):.5f}")
        if g == gens - 1:
            break
        # ---- breed: two parents drawn from the elite pool, uniform crossover
        pa = elite[rng.randint(0, nkeep, size=pop)]
        pb = elite[rng.randint(0, nkeep, size=pop)]
        take = rng.rand(pop, K) < 0.5
        child = xp.where(take, pa, pb)
        # ---- mutate
        u = rng.rand(pop, K)
        glob = u < 0.06                              # jump anywhere
        walk = (u >= 0.06) & (u < 0.20)              # step along the grid
        coll = (u >= 0.20) & (u < 0.23)              # delete an atom
        child = xp.where(glob, rng.randint(0, natom, size=(pop, K)), child)
        step = (rng.randint(-12, 13, size=(pop, K))).astype(xp.int32)
        child = xp.where(walk, xp.clip(child + step, 0, natom - 1), child)
        child = xp.where(coll, intercept, child)
        # ---- elitism on a SMALL head only. Copying the whole survivor pool
        #      back in made the population collapse onto one genome by gen 14,
        #      after which the remaining generations were wasted.
        child[:nhead] = elite[:nhead]
        # ---- random immigrants keep the search from sitting in one basin
        child[-nimm:] = rng.randint(0, natom, size=(nimm, K)).astype(xp.int32)
        idx = child.astype(xp.int32)
    if verbose:
        print(f"      {gens} generations x {pop:,} = "
              f"{gens*pop/1e9:.2f}e9 evaluations in {time.time()-t0:.1f}s")
    if log is not None:
        log[which] = hist
    if verbose:
        print(f"      hall of fame: {len(hof):,} distinct laws retained")
    return elite, erms, ecpx, hof


def dedupe(hof, ntop=40):
    """The best distinct laws in a run, taken from its hall of fame."""
    items = sorted(hof.values(), key=lambda t: t[0])[:ntop]
    return [(r, c, row) for r, c, row in items]


def describe(S, row):
    names = []
    r = row.get() if hasattr(row, 'get') else row
    for a in sorted(set(int(v) for v in r)):
        nm, dim = S.meta[a]
        if nm == "1":
            continue
        names.append(f"[{dim}]{nm}")
    return names


def dims_used(S, row):
    r = row.get() if hasattr(row, 'get') else row
    return sorted({S.meta[int(a)][1] for a in set(int(v) for v in r)
                   if S.meta[int(a)][0] != "1"})


TARGETS = ("real", "null", "perm", "perm_g")


def main():
    print("=" * 78)
    print("EVOLUTIONARY STRUCTURE SEARCH -- real SPARC, with physics-free twins")
    print("=" * 78)
    b = Bench(ndraw=8, verbose=False)
    A, meta = build_atoms(b, nscale=int(os.environ.get("NSCALE", 13)))
    S = Screen(b, A, meta)
    POP = int(os.environ.get("POP", 2_000_000))
    GENS = int(os.environ.get("GENS", 120))
    KMAX = int(os.environ.get("K", 6))
    SEED = int(os.environ.get("SEED", 0))
    print(f"\n   population {POP:,}   generations {GENS}   K_max {KMAX}")
    print(f"   evaluations: {POP*GENS/1e9:.2f}e9 per target, "
          f"{POP*GENS*len(TARGETS)/1e9:.2f}e9 total\n")
    print("   each target is measured against ITS OWN RAR baseline, because")
    print("   the twins are constructed with slightly different scatter:")
    for t in TARGETS:
        print(f"      {t:7s} baseline {S.ref_train[t]:.4f} dex")
    print("")

    log, res = {}, {}
    for which in TARGETS:
        print(f"   --- target: {which}")
        elite, erms, ecpx, hof = evolve(S, which, K=KMAX, pop=POP, gens=GENS,
                                        seed=SEED, log=log)
        top = dedupe(hof)
        res[which] = {"top": top, "hof": hof, "best": top[0][0],
                      "nuniq": len(hof)}
        print("")

    def gain(w):
        r = S.ref_train[w]
        return 100.0 * (r - res[w]["best"]) / r

    print("=" * 78)
    print("   RESULT -- training set, each against its own baseline")
    print("=" * 78)
    for w in TARGETS:
        print(f"      {w:7s} baseline {S.ref_train[w]:.5f}  best "
              f"{res[w]['best']:.5f}   gain {gain(w):+6.2f}%   "
              f"({res[w]['nuniq']:,} distinct laws seen)")
    gr = gain("real")
    gc = max(gain("null"), gain("perm"), gain("perm_g"))
    worst = max(("null", "perm", "perm_g"), key=gain)
    print("")
    print(f"      real {gr:+.2f}%   best control ({worst}) {gc:+.2f}%   "
          f"margin {gr-gc:+.2f} pp")
    if gr <= gc:
        print("      => the machinery manufactures at least as much gain on")
        print("         physics-free data. NOTHING HAS BEEN DISCOVERED.")
    else:
        print("      => real exceeds every control on train. Blind decides.")

    # ---- gain as a function of complexity, real against the worst control
    print("")
    print("   gain vs complexity (train), real minus best control:")
    hk = {}
    for w in TARGETS:
        d = {}
        for r, c, _ in res[w]["hof"].values():
            if c not in d or r < d[c]:
                d[c] = r
        hk[w] = d
    for k in range(1, KMAX + 1):
        if k not in hk["real"]:
            continue
        gk = 100 * (S.ref_train["real"] - hk["real"][k]) / S.ref_train["real"]
        cs = [100 * (S.ref_train[w] - hk[w][k]) / S.ref_train[w]
              for w in ("null", "perm", "perm_g") if k in hk[w]]
        if not cs:
            print(f"      k={k}  real {hk['real'][k]:.5f} ({gk:+5.2f}%)   "
                  f"no control reached this complexity")
            continue
        ck = max(cs)
        print(f"      k={k}  real {hk['real'][k]:.5f} ({gk:+5.2f}%)   "
              f"control best {ck:+5.2f}%   margin {gk-ck:+5.2f} pp")

    print("")
    print("   top ten real laws.  [g] uses only the RAR's own direction;")
    print("   [r] uses the second independent direction; [gal] is galaxy-level;")
    print("   [nl] is a nonlocal functional of the source.")
    ndim = {}
    for rms, cpx, row in res["real"]["top"][:10]:
        nm = describe(S, row)
        for d in dims_used(S, row):
            ndim[d] = ndim.get(d, 0) + 1
        print(f"      {rms:.5f}  k={cpx}  " + " + ".join(nm))
    print("      directions used across the top ten: " +
          "  ".join(f"{d}={n}" for d, n in sorted(ndim.items())))

    # ---------------------------------------------------------------- blind
    print("")
    print("   BLIND -- touched once, shortlist only")
    NS = min(20, min(len(res[w]["top"]) for w in TARGETS))
    print(f"      shortlist size {NS} (limited by the smallest distinct-law pool)")
    sl = xp.asarray(np.stack([r for _, _, r in res["real"]["top"][:NS]]))
    # coefficients are fitted on TRAIN and then frozen; the blind RMS uses those
    # frozen coefficients, so nothing about blind enters the fit.
    _, ctr = S.score(sl, "real")
    rbn = [S.verify(sl[j], ctr[j], "real", blind=True) for j in range(NS)]
    vtr = [S.verify(sl[j], ctr[j], "real", blind=False) for j in range(NS)]
    dmax = max(abs(vtr[j] - res["real"]["top"][j][0]) for j in range(NS))
    print(f"      Gram-path vs direct-from-data train RMS: max |diff| "
          f"{dmax:.2e} dex")
    print(f"      RAR on blind: {S.rar_blind:.5f} dex")
    bwin = 0
    for j, (rms, cpx, row) in enumerate(res["real"]["top"][:NS]):
        d = 100 * (S.rar_blind - rbn[j]) / S.rar_blind
        if d > 0:
            bwin += 1
        print(f"      #{j+1:2d}  train {rms:.5f} -> blind {rbn[j]:.5f}"
              f"   ({d:+6.2f}% vs RAR)   k={cpx}  "
              + ",".join(dims_used(S, row)))
    print(f"      {bwin}/{NS} of the shortlist beat the RAR out of sample.")

    # the same shortlist procedure applied to each control, so the blind
    # comparison is like-for-like rather than real-only
    print("")
    print("   the identical shortlist-then-blind procedure, on the controls:")
    cb = {}
    for w in ("null", "perm", "perm_g"):
        slw = xp.asarray(np.stack([r for _, _, r in res[w]["top"][:NS]]))
        _, cw = S.score(slw, w)
        bb = [S.verify(slw[j], cw[j], w, blind=True) for j in range(NS)]
        g = 100 * (S.ref_blind[w] - min(bb)) / S.ref_blind[w]
        cb[w] = {"best_blind": min(bb), "gain_pct": g,
                 "wins": sum(1 for v in bb if v < S.ref_blind[w])}
        print(f"      {w:7s} baseline {S.ref_blind[w]:.5f}  best blind "
              f"{min(bb):.5f}   gain {g:+6.2f}%   "
              f"{cb[w]['wins']}/{NS} beat their baseline")
    breal = 100 * (S.rar_blind - min(rbn)) / S.rar_blind
    bctl = max(cb[w]["gain_pct"] for w in cb)
    print("")
    print(f"      BLIND: real {breal:+.2f}%   best control {bctl:+.2f}%   "
          f"margin {breal-bctl:+.2f} pp")

    out = {
        "population": POP, "generations": GENS, "K_max": KMAX, "seed": SEED,
        "targets": list(TARGETS),
        "evaluations_total": POP * GENS * len(TARGETS),
        "natom": S.natom, "n_train": S.ntr, "n_blind": S.nbl,
        "baseline_train": S.ref_train, "baseline_blind": S.ref_blind,
        "best_train": {w: res[w]["best"] for w in TARGETS},
        "gain_pct_train": {w: gain(w) for w in TARGETS},
        "margin_pp": gr - gc,
        "gain_by_complexity": {
            str(k): {w: (100 * (S.ref_train[w] - hk[w][k]) / S.ref_train[w])
                     for w in TARGETS if k in hk[w]}
            for k in range(1, KMAX + 1) if k in hk["real"]},
        "blind_shortlist": [
            {"rank": j + 1, "train": rr, "train_direct": vtr[j],
             "blind": rbn[j], "k": c,
             "dims": dims_used(S, row),
             "atoms": describe(S, row)}
            for j, (rr, c, row) in enumerate(res["real"]["top"][:NS])],
        "blind_wins": bwin,
        "blind_controls": cb,
        "blind_gain_real_pct": breal,
        "blind_margin_pp": breal - bctl,
        "gram_vs_direct_max_abs_dex": dmax,
        "history": {w: [(h[0], h[1], h[2], h[3]) for h in log[w]]
                    for w in TARGETS},
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "runs", "evolve-sparc")
    os.makedirs(p, exist_ok=True)
    fn = os.path.join(p, f"evolve-seed{SEED}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   written: {os.path.normpath(fn)}")


if __name__ == "__main__":
    main()
