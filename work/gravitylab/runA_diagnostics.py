"""Run A, step 6: the required outputs the score table does not carry.

The program asks for more than a chi-squared. For every candidate it wants
residual correlations against the observables a model might be quietly using,
the BTFR slope and scatter computed from the model's PREDICTED flat velocity
rather than the observed one, and the inferred confinement thickness.

A candidate that fits only by pushing distances, inclinations or mass-to-light
ratios to the edges of their priors is not successful, so the nuisance pulls
are reported too.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import data as D
import runA as A

G = 6.674e-11
KPC = 3.0856775814913673e19
KMS = 1e3
MSUN = 1.98892e30
A0_CANON = 1.2e-10
BAR = "=" * 78


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 6:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


def main():
    gals = D.ingest(verbose=False)
    D.stratified_split(gals, verbose=False)
    gals = A.build_draws(gals)
    tr = [g for g in gals if g.split == "train"]
    ev = [g for g in gals if g.split in ("validation", "blind")]

    res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "outputs", "runA_results.json"),
                         encoding="utf-8"))
    by = {r["name"]: r for r in res}
    picks = ["D1 Newton", "D2 AQUAL simple", "K1xQ2 g", "K1xQ4 nonlocal"]

    head("Residual correlations on held-out galaxies (validation + blind)")
    print("   Spearman rho of mean log10(g_obs/g_pred) against each variable.")
    print("   A real law leaves no structure here.\n")
    print(f"   {'model':<20}{'R/R_d':>9}{'SB_eff':>9}{'M_b':>9}"
          f"{'f_gas':>9}{'incl':>9}{'dist':>9}{'Qual':>7}")
    print("   " + "-" * 81)
    store = {}
    for nm in picks:
        spec = next(s for s in A.SPECS if s["name"] == nm)
        p = dict(by[nm]["params"])
        p.setdefault("a0", A0_CANON)
        rr, sb, mb, fg, ic, dd, qq, resid = [], [], [], [], [], [], [], []
        for g in ev:
            d = g.draws
            gp = np.maximum(A.predict(spec, p, d), 1e-30)
            r = float(np.mean(np.log10(np.mean(d["gobs"], axis=0))
                              - np.log10(np.mean(gp, axis=0))))
            resid.append(r)
            rr.append(float(np.mean(d["R"]) / d["Rdisk"]))
            sb.append(g.SBeff); mb.append(math.log10(max(g.Mb, 1.0)))
            fg.append(g.fgas); ic.append(g.i0); dd.append(g.D0); qq.append(g.Qual)
        row = [spearman(x, resid) for x in (rr, sb, mb, fg, ic, dd, qq)]
        store[nm] = dict(resid=resid, corr=row)
        print(f"   {nm:<20}" + "".join(f"{v:>+9.3f}" for v in row[:6])
              + f"{row[6]:>+7.3f}")
    print("   " + "-" * 81)
    print("   |rho| > ~0.35 on 48 galaxies is nominally significant at p < 0.02.")

    head("BTFR from the model's PREDICTED flat velocity")
    print("   The program requires V_f,pred, not V_f,obs. V_f,pred is the model's")
    print("   circular velocity at the outermost measured radius.\n")
    print(f"   {'model':<20}{'slope s':>10}{'err':>8}{'scatter':>10}{'n':>5}")
    print("   " + "-" * 53)
    btf = {}
    for nm in picks:
        spec = next(s for s in A.SPECS if s["name"] == nm)
        p = dict(by[nm]["params"]); p.setdefault("a0", A0_CANON)
        lv, lm = [], []
        for g in gals:
            d = g.draws
            gp = np.maximum(A.predict(spec, p, d), 1e-30)
            Vf = float(np.mean(np.sqrt(gp[:, -1] * d["R"][:, -1] * KPC)) / KMS)
            Mb = float(np.mean(d["Mb"]))
            if Vf > 0 and Mb > 0:
                lv.append(math.log10(Vf / 100.0)); lm.append(math.log10(Mb))
        lv, lm = np.array(lv), np.array(lm)
        X = np.vstack([lv, np.ones_like(lv)]).T
        s, b = np.linalg.lstsq(X, lm, rcond=None)[0]
        r = lm - (s * lv + b)
        n = len(lm)
        se = float(np.sqrt(np.sum(r ** 2) / (n - 2)
                           / np.sum((lv - lv.mean()) ** 2)))
        btf[nm] = dict(slope=float(s), err=se, scatter=float(np.std(r, ddof=2)))
        print(f"   {nm:<20}{s:>+10.3f}{se:>8.3f}"
              f"{float(np.std(r, ddof=2)):>10.3f}{n:>5}")
    print("   " + "-" * 53)
    print("   MOND / RAR predicts s = 4.")

    head("Inferred confinement thickness")
    print("   h_eff = G M_b(<R) / (R g_R), weighted over the sub-a0 region.\n")
    print(f"   {'model':<20}{'slope s_h':>11}{'err':>8}{'scatter':>10}{'n':>5}")
    print("   " + "-" * 54)
    for nm in picks:
        spec = next(s for s in A.SPECS if s["name"] == nm)
        p = dict(by[nm]["params"]); p.setdefault("a0", A0_CANON)
        he, lm = [], []
        for g in gals:
            d = g.draws
            gp = np.maximum(A.predict(spec, p, d), 1e-30)
            gb = d["gbar"]; R = d["R"]
            sel = gb < A0_CANON
            if sel.sum() < 3 * gb.shape[0] * 0.1:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                h = np.where(sel, R * gb / gp, np.nan)
            hv = float(np.nanmean(h))
            if np.isfinite(hv) and hv > 0:
                he.append(math.log10(hv))
                lm.append(math.log10(float(np.mean(d["Mb"]))))
        if len(he) < 10:
            print(f"   {nm:<20}{'too few':>11}")
            continue
        he, lm = np.array(he), np.array(lm)
        X = np.vstack([lm, np.ones_like(lm)]).T
        s, b = np.linalg.lstsq(X, he, rcond=None)[0]
        r = he - (s * lm + b)
        se = float(np.sqrt(np.sum(r ** 2) / (len(he) - 2)
                           / np.sum((lm - lm.mean()) ** 2)))
        print(f"   {nm:<20}{s:>+11.4f}{se:>8.4f}"
              f"{float(np.std(r, ddof=2)):>10.4f}{len(he):>5}")
    print("   " + "-" * 54)
    print("   The cylindrical argument requires s_h = +0.5.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    json.dump(dict(correlations={k: v["corr"] for k, v in store.items()},
                   btfr=btf),
              open(os.path.join(out, "runA_diagnostics.json"), "w",
                   encoding="utf-8", newline="\n"), indent=1, default=float)
    print(f"\n   wrote outputs/runA_diagnostics.json")


if __name__ == "__main__":
    main()
