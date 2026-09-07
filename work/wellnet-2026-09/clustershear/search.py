"""search.py -- score ~1M candidate laws against the open half, and against their null twins.

WHAT THIS CAN AND CANNOT MEAN, stated before the numbers, because a top-25 list
is the single most misleading object this programme can produce.

`work/gravitylab/gpu_search.py` records two results that govern everything here:

  * THE RANK-2 THEOREM. The standard galaxy variable set has rank 2 -- a billion
    samples of f(a_N, r) is a billion samples of a two-dimensional space.
  * THE LABEL CONTROL. The identical search run on data containing only survey
    structure and no physics selected the same variables with a LARGER apparent
    gain: 6.1% against 1.1%.

Its conclusion -- "more search in the same space produces overfitting faster,
not discovery" -- applies with more force here, not less, because the admissible
variable set for these clusters is SMALLER than for galaxies. Without a baryon
model there is no g_bar, so the atoms are built from radius, redshift and X-ray
temperature only. A million models over three variables and ~2,800 points will
fit noise beautifully.

SO THE RANKING IS NOT THE OUTPUT. The output is the MARGIN: for every candidate,
the identical model form is refitted on two null twins, and what is reported is
how much better it does on the real clusters than on data that cannot contain a
gravity law. A model that wins on the real data and wins just as hard on the
nulls has found the survey, not the universe.

  NULL A  RANDOM POINTINGS -- the 120 cluster-free profiles from analyse.py,
          run through the identical estimator. Contains the catalogue's
          systematics and none of its clusters.
  NULL B  LABEL SCRAMBLE -- the real profiles with each cluster's (z, kT)
          permuted. Preserves every marginal distribution and destroys only the
          association between a cluster's properties and its lensing.

THE SEARCH. 70 atoms; every subset of size 1-4 enumerated exhaustively, which is
971,635 model forms, each fitted EXACTLY by weighted least squares rather than
sampled. Exhaustive beats sampled at this size: it removes the search-strategy
degree of freedom, and the Gram trick makes it cheap -- one shared 70x70 matrix,
then a batched 4x4 solve per subset.

NOTHING IS CONFIRMED HERE. The sealed 304 are not touched. This is a search on
the open half, and a search is where hypotheses come from, not where they are
tested.

    python search.py
"""
from __future__ import annotations

import io
import itertools
import json
import math
import os
import sys
import time

import numpy as np

import analyse
import controls as K

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "search.json")

KMAX = 4
TOP_N = 25
RNG = np.random.default_rng(20260907)


# --------------------------------------------------------------- the points
def points(recs, scramble=False, radial_scramble=False):
    """One row per (cluster, radial bin): features and inverse-variance weight.

    `scramble` permutes (z, kT) across clusters and is NULL B. Note what it can
    and cannot test: each profile keeps its own radii, so any purely radial term
    fits the scrambled data exactly as well as the real data. Null B therefore
    tests the PROPERTY dependence only, which is why `radial_scramble` (NULL C)
    exists -- it permutes each cluster's DeltaSigma values across its own radial
    bins, destroying the radial shape while preserving every marginal.
    """
    if scramble:
        # permute (z, kT) across clusters; keep each profile where it is
        props = [(r["z"], r.get("kt")) for r in recs]
        perm = RNG.permutation(len(props))
        recs = [dict(r, z=props[perm[i]][0], kt=props[perm[i]][1])
                for i, r in enumerate(recs)]
    if radial_scramble:
        out = []
        for r in recs:
            prof = [dict(b) for b in r["profile"]]
            good = [i for i, b in enumerate(prof) if b["gt"] is not None]
            if len(good) > 1:
                perm = RNG.permutation(len(good))
                vals = [(prof[good[j]]["gt"], prof[good[j]]["err"],
                         prof[good[j]]["beta"]) for j in range(len(good))]
                for j, i in enumerate(good):
                    prof[i]["gt"], prof[i]["err"], prof[i]["beta"] = vals[perm[j]]
            out.append(dict(r, profile=prof))
        recs = out
    R, Z, T, Y, W = [], [], [], [], []
    for r in recs:
        try:
            kt = float(r.get("kt"))
        except (TypeError, ValueError):
            kt = None
        if not kt or not np.isfinite(kt) or kt <= 0:
            continue
        Dl = float(K.C.d_ang(r["z"]))
        for b in r["profile"]:
            if b["gt"] is None or not (b["err"] or 0) > 0:
                continue
            beta = b.get("beta")
            if not beta or beta <= 0:
                continue
            inv_sc = K.SIGMA_CRIT_PREF * Dl * beta
            if inv_sc <= 0:
                continue
            R.append(b["R"])
            Z.append(r["z"])
            T.append(kt)
            Y.append(b["gt"] / inv_sc)                 # DeltaSigma, kg/m^2
            W.append((inv_sc / b["err"]) ** 2)         # inverse variance
    return (np.array(R), np.array(Z), np.array(T),
            np.array(Y), np.array(W))


# ---------------------------------------------------------------- the atoms
def atoms(R, Z, T):
    """70 atoms in (radius, redshift, temperature). Names kept for the report."""
    lr, lz, lt = np.log10(R), np.log10(1.0 + Z), np.log10(T)
    A, names = [], []

    def add(nm, v):
        v = np.asarray(v, dtype=float)
        if np.all(np.isfinite(v)):
            A.append(v)
            names.append(nm)

    add("1", np.ones_like(R))
    # radius: the axis round 2 said the surviving cluster excess is organised by
    for p in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        add("R^%+.1f" % p, R ** p)
    add("log R", lr)
    add("(log R)^2", lr ** 2)
    add("(log R)^3", lr ** 3)
    add("exp(-R)", np.exp(-R))
    add("exp(-R/2)", np.exp(-R / 2.0))
    add("exp(-R/4)", np.exp(-R / 4.0))
    add("1/(1+R)", 1.0 / (1.0 + R))
    add("1/(1+R^2)", 1.0 / (1.0 + R ** 2))
    add("R/(1+R)^2", R / (1.0 + R) ** 2)
    add("tanh R", np.tanh(R))
    add("log(1+R)", np.log1p(R))
    # temperature: the admissible mass proxy
    for p in (-1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        add("T^%+.1f" % p, T ** p)
    add("log T", lt)
    add("(log T)^2", lt ** 2)
    add("exp(-T/5)", np.exp(-T / 5.0))
    # redshift
    for p in (1.0, 2.0):
        add("(1+z)^%.0f" % p, (1.0 + Z) ** p)
    add("log(1+z)", lz)
    add("z", Z)
    add("z^2", Z ** 2)
    # products -- where any non-separable law would live
    for rp in (-1.0, -0.5, 0.5, 1.0):
        for tp in (0.5, 1.0, 1.5):
            add("R^%+.1f T^%+.1f" % (rp, tp), (R ** rp) * (T ** tp))
    for rp in (-1.0, 0.5, 1.0):
        add("R^%+.1f (1+z)" % rp, (R ** rp) * (1.0 + Z))
    add("T (1+z)", T * (1.0 + Z))
    add("T (1+z)^2", T * (1.0 + Z) ** 2)
    add("log R log T", lr * lt)
    add("log R log(1+z)", lr * lz)
    add("log T log(1+z)", lt * lz)
    add("T/R", T / R)
    add("T/R^2", T / R ** 2)
    add("T/(1+R)", T / (1.0 + R))
    add("sqrt(T)/R", np.sqrt(T) / R)
    add("T^1.5/R", T ** 1.5 / R)
    add("T exp(-R/2)", T * np.exp(-R / 2.0))
    add("T^1.5 exp(-R/2)", T ** 1.5 * np.exp(-R / 2.0))
    add("(1+z)^2/R", (1.0 + Z) ** 2 / R)
    add("T (1+z)/R", T * (1.0 + Z) / R)
    add("T (1+z)^2/R^2", T * (1.0 + Z) ** 2 / R ** 2)
    return np.vstack(A), names


# --------------------------------------------------------------- the search
def gram(A, y, w):
    """Shared normal equations. G = A W A^T, b = A W y, and the weighted TSS."""
    Aw = A * w
    return Aw @ A.T, Aw @ y, float(np.sum(w * y * y))


def exhaustive(A, y, w, kmax=KMAX, ridge=1e-12):
    """Every subset of size 1..kmax, fitted exactly. Returns (rms, subset) lists."""
    G, b, tss = gram(A, y, w)
    n = A.shape[0]
    sw = float(np.sum(w))
    results = []
    for k in range(1, kmax + 1):
        combos = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64)
        if combos.size == 0:
            continue
        # batched k x k solves -- this is the whole trick
        sub = G[combos[:, :, None], combos[:, None, :]]        # (m,k,k)
        sub = sub + ridge * np.eye(k)[None, :, :] * np.trace(G) / n
        rhs = b[combos]                                        # (m,k)
        try:
            coef = np.linalg.solve(sub, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            coef = np.stack([np.linalg.lstsq(sub[i], rhs[i], rcond=None)[0]
                             for i in range(sub.shape[0])])
        # weighted residual sum of squares = tss - 2 c.b + c G c
        cGc = np.einsum("mi,mij,mj->m", coef, sub, coef)
        rss = tss - 2.0 * np.einsum("mi,mi->m", coef, rhs) + cGc
        rss = np.maximum(rss, 0.0)
        rms = np.sqrt(rss / sw)
        results.append((rms, combos, coef))
    return results


def best_of(results, names, top=TOP_N):
    allr = np.concatenate([r[0] for r in results])
    order = np.argsort(allr)[:top]
    flat = []
    for rms, combos, coef in results:
        for i in range(len(rms)):
            flat.append((rms[i], combos[i], coef[i]))
    flat.sort(key=lambda t: t[0])
    return [dict(rms=float(f[0]),
                 terms=[names[j] for j in f[1]],
                 coef=[float(c) for c in f[2]]) for f in flat[:top]], allr


def rms_of_form(term_idx, A, y, w, ridge=1e-12):
    """Refit ONE model form on a different dataset. This is the null twin."""
    G, b, tss = gram(A, y, w)
    n = A.shape[0]
    idx = np.array(term_idx, dtype=np.int64)
    sub = G[np.ix_(idx, idx)] + ridge * np.eye(len(idx)) * np.trace(G) / n
    try:
        c = np.linalg.solve(sub, b[idx])
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(sub, b[idx], rcond=None)[0]
    rss = max(tss - 2 * c @ b[idx] + c @ sub @ c, 0.0)
    return float(math.sqrt(rss / float(np.sum(w))))


def main():
    sys.path.insert(0, os.path.dirname(HERE))
    from holdout import loader                                  # noqa: E402
    loader.verify()

    recs = analyse.load_profiles(analyse.PROF)
    rnd = analyse.load_profiles(analyse.RANDOM)
    import guard
    guard.assert_all_open([r["name"] for r in recs], "search input")

    R, Z, T, Y, W = points(recs)
    A, names = atoms(R, Z, T)
    n_models = sum(math.comb(len(names), k) for k in range(1, KMAX + 1))
    print("points=%d  atoms=%d  model forms=%s"
          % (len(Y), len(names), format(n_models, ",")), flush=True)

    t0 = time.time()
    res = exhaustive(A, Y, W)
    top, allr = best_of(res, names)
    dt = time.time() - t0
    print("searched %s forms in %.1f s (%s forms/s)"
          % (format(n_models, ","), dt, format(int(n_models / dt), ",")), flush=True)

    # ---- the null twins, same atoms, same forms
    Rr, Zr, Tr, Yr, Wr = points(rnd)
    Ar, _ = atoms(Rr, Zr, Tr)
    Rs, Zs, Ts, Ys, Ws = points(recs, scramble=True)
    As, _ = atoms(Rs, Zs, Ts)
    Rc, Zc, Tc, Yc, Wc = points(recs, radial_scramble=True)
    Ac, _ = atoms(Rc, Zc, Tc)
    print("null A (random pointings): %d pts | B (property scramble): %d | "
          "C (radial scramble): %d" % (len(Yr), len(Ys), len(Yc)), flush=True)

    name_idx = {nm: i for i, nm in enumerate(names)}
    baseline = float(np.sqrt(np.sum(W * (Y - np.average(Y, weights=W)) ** 2)
                             / np.sum(W)))
    br = float(np.sqrt(np.sum(Wr * (Yr - np.average(Yr, weights=Wr)) ** 2)
                       / np.sum(Wr)))
    bs = float(np.sqrt(np.sum(Ws * (Ys - np.average(Ys, weights=Ws)) ** 2)
                       / np.sum(Ws)))
    bc = float(np.sqrt(np.sum(Wc * (Yc - np.average(Yc, weights=Wc)) ** 2)
                       / np.sum(Wc)))

    ranked = []
    for m in top:
        idx = [name_idx[t] for t in m["terms"]]
        gain = 1.0 - m["rms"] / baseline
        gain_r = 1.0 - rms_of_form(idx, Ar, Yr, Wr) / br
        gain_s = 1.0 - rms_of_form(idx, As, Ys, Ws) / bs
        gain_c = 1.0 - rms_of_form(idx, Ac, Yc, Wc) / bc
        worst = max(gain_r, gain_s, gain_c)
        ranked.append(dict(
            terms=m["terms"], k=len(m["terms"]), rms=m["rms"],
            gain_real=gain, gain_null_random=gain_r,
            gain_null_property_scramble=gain_s, gain_null_radial_scramble=gain_c,
            margin=gain - worst,
            beats_all_nulls=bool(gain > worst)))

    nbeat = sum(1 for m in ranked if m["beats_all_nulls"])

    # --- how much of the top 25 is one model wearing 25 hats?
    cores = {}
    for m in ranked:
        key = tuple(sorted(t for t in m["terms"]
                           if t in ("log(1+z)", "log R log(1+z)", "z")))
        cores.setdefault(key, 0)
        cores[key] += 1
    distinct = len({frozenset(m["terms"]) for m in ranked})
    shared = max(cores.values()) if cores else 0

    # --- the complexity trade-off: best gain at each k, on real and on nulls
    per_k = []
    for k in range(1, KMAX + 1):
        cand = [(rms, combos, coef) for rms, combos, coef in res
                if combos.shape[1] == k]
        if not cand:
            continue
        rms, combos, coef = cand[0]
        j = int(np.argmin(rms))
        idx = list(combos[j])
        per_k.append(dict(
            k=k, terms=[names[t] for t in idx],
            gain_real=1.0 - float(rms[j]) / baseline,
            gain_null_random=1.0 - rms_of_form(idx, Ar, Yr, Wr) / br,
            gain_null_property=1.0 - rms_of_form(idx, As, Ys, Ws) / bs,
            gain_null_radial=1.0 - rms_of_form(idx, Ac, Yc, Wc) / bc))
    out = dict(lane="clustershear", half="OPEN",
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               n_points=len(Y), n_atoms=len(names), kmax=KMAX,
               n_model_forms=n_models, seconds=dt,
               forms_per_second=int(n_models / dt),
               baseline_rms=baseline, top=ranked,
               n_top_beating_all_nulls=nbeat,
               distinct_term_sets_in_top=distinct,
               top_sharing_one_redshift_core=shared,
               best_per_k=per_k,
               sealed_untouched=len(loader.sealed_names()),
               atoms=names,
               caveat=("All top forms sit at k=kmax and there is NO complexity "
                       "penalty in this ranking, so 'top 25' means 'the 25 with "
                       "most parameters that fit best'. Read best_per_k for the "
                       "complexity trade-off, and the null columns for whether "
                       "any of it is physics."))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")

    print("\ntop %d of %s model forms" % (TOP_N, format(n_models, ",")))
    print("  %-3s %-8s %-8s %-8s %-8s %-8s %s"
          % ("k", "gain", "nullA", "nullB", "nullC", "margin", "terms"))
    for m in ranked:
        print("  %-3d %-8.4f %-8.4f %-8.4f %-8.4f %+-8.4f %s"
              % (m["k"], m["gain_real"], m["gain_null_random"],
                 m["gain_null_property_scramble"], m["gain_null_radial_scramble"],
                 m["margin"], " + ".join(m["terms"])))
    print("\n%d of the top %d beat ALL THREE null twins." % (nbeat, TOP_N))
    print("distinct term sets in the top %d: %d; sharing one redshift core: %d"
          % (TOP_N, distinct, shared))
    print("\nbest model at each complexity (no penalty applied anywhere):")
    print("  %-3s %-8s %-8s %-8s %-8s %s"
          % ("k", "gain", "nullA", "nullB", "nullC", "terms"))
    for p in per_k:
        print("  %-3d %-8.4f %-8.4f %-8.4f %-8.4f %s"
              % (p["k"], p["gain_real"], p["gain_null_random"],
                 p["gain_null_property"], p["gain_null_radial"],
                 " + ".join(p["terms"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
