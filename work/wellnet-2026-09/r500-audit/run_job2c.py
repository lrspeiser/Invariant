"""
JOB 2C -- amplitude-matched discriminator.

Job 2B's injections did not reproduce the observed dispersion (simulated S2 was
0.157 against an observed 0.173), so their error bars were unrealistically tight
and the comparison was not apples to apples.  Here every candidate truth is first
TUNED so that the simulated pipeline output matches the real data on two declared
summaries -- S1 (the pooled Spearman) and S2 (the collapse rms) -- and only then
is the discriminator evaluated.

The discriminator is made scale-free:

    Drel = S2(r/R500) / S2(r_physical) - 1        negative = R500 tightens

so it cannot be moved by getting the overall amplitude of the excess wrong.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import nullsim as N

KPC = I.KPC
OUT = {}


def sim(TS, seed, n, s_scaled=0.0, s_abs=0.0, shape_amp=0.0):
    cfg = dict(N.DEFAULT_CFG, shape_amp=shape_amp)
    rng = np.random.default_rng(seed)
    S1, S2, D1, Dr, S3 = [], [], [], [], []
    for _ in range(n):
        res = N.one_realisation(TS, rng, cfg, s_scaled=s_scaled, s_abs=s_abs)
        if res is None:
            continue
        r, y, t = res["r"], res["y"], res["R500_obs"]
        a = N.spear(r / t, y)
        b = N.collapse_rms(r / t, y)
        c = N.collapse_rms(r / (1000 * KPC), y)
        S1.append(a); S2.append(b)
        D1.append(a - N.spear(r, y))
        Dr.append(b / c - 1.0)
        S3.append(N.slope_beyond(r / t, y))
    return {k: np.array(v, float) for k, v in
            dict(S1=S1, S2=S2, D1=D1, Drel=Dr, S3=S3).items()}


def tune(TS, obs_S1, obs_S2, kind, seed=1234, n=60):
    """1-D search on the injected slope to match S1, then on shape_amp to match S2."""
    lo, hi = 0.0, 2.5
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        kw = dict(s_scaled=-mid) if kind == "scaled" else dict(s_abs=-mid)
        v = sim(TS, seed, n, shape_amp=0.10, **kw)["S1"].mean()
        if v > obs_S1:          # not negative enough
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    kw = dict(s_scaled=-s) if kind == "scaled" else dict(s_abs=-s)
    lo2, hi2 = 0.0, 0.60
    for _ in range(14):
        mid = 0.5 * (lo2 + hi2)
        v = sim(TS, seed + 1, n, shape_amp=mid, **kw)["S2"].mean()
        if v < obs_S2:
            lo2 = mid
        else:
            hi2 = mid
    return s, 0.5 * (lo2 + hi2)


def summ(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                p05=float(np.percentile(a, 5)), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)))


def main():
    cl = I.load_all(verbose=False)
    TS = [N.Template(c) for c in cl]
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    y = I.rar_residual(gb, go)
    obs = dict(S1=N.spear(r / R5, y), S2=N.collapse_rms(r / R5, y),
               S3=N.slope_beyond(r / R5, y),
               D1=N.spear(r / R5, y) - N.spear(r, y),
               Drel=N.collapse_rms(r / R5, y) / N.collapse_rms(r / (1000 * KPC), y) - 1.0)
    print(f"OBSERVED  S1={obs['S1']:+.4f}  S2={obs['S2']:.4f}  S3={obs['S3']:+.4f}  "
          f"D1={obs['D1']:+.5f}  Drel={obs['Drel']:+.6f}")

    res = {}
    for kind in ("abs", "scaled"):
        s, amp = tune(TS, obs["S1"], obs["S2"], kind)
        kw = dict(s_scaled=-s) if kind == "scaled" else dict(s_abs=-s)
        v = sim(TS, 4321, 400, shape_amp=amp, **kw)
        res[kind] = dict(
            tuned_slope=float(-s), tuned_shape_amp_dex=float(amp),
            matched=dict(S1=summ(v["S1"]), S2=summ(v["S2"]), S3=summ(v["S3"])),
            D1=summ(v["D1"]), Drel=summ(v["Drel"]),
            pct_obs_D1=float(100 * np.mean(v["D1"] <= obs["D1"])),
            pct_obs_Drel=float(100 * np.mean(v["Drel"] <= obs["Drel"])),
            z_obs_Drel=float((obs["Drel"] - v["Drel"].mean()) / v["Drel"].std(ddof=1)))
        lab = "PHYSICAL radius" if kind == "abs" else "SCALED radius r/R500"
        print(f"\n  truth organised by {lab}")
        print(f"    tuned: slope {-s:+.4f} dex/dex, cluster shape scatter {amp:.4f} dex")
        print(f"    match: S1 {v['S1'].mean():+.4f}+-{v['S1'].std(ddof=1):.4f} "
              f"(obs {obs['S1']:+.4f}), S2 {v['S2'].mean():.4f}"
              f"+-{v['S2'].std(ddof=1):.4f} (obs {obs['S2']:.4f})")
        print(f"    Drel  {v['Drel'].mean():+.6f} +- {v['Drel'].std(ddof=1):.6f}   "
              f"observed {obs['Drel']:+.6f}  -> percentile "
              f"{100*np.mean(v['Drel']<=obs['Drel']):.1f}, z = "
              f"{(obs['Drel']-v['Drel'].mean())/v['Drel'].std(ddof=1):+.2f}")
        print(f"    D1    {v['D1'].mean():+.5f} +- {v['D1'].std(ddof=1):.5f}   "
              f"observed {obs['D1']:+.5f}  -> percentile "
              f"{100*np.mean(v['D1']<=obs['D1']):.1f}")

    # flat truth, matched on S2 only (it cannot match S1 by construction)
    _, amp0 = tune(TS, obs["S1"], obs["S2"], "abs")
    v0 = sim(TS, 8888, 300, shape_amp=amp0)
    res["flat"] = dict(tuned_shape_amp_dex=float(amp0),
                       matched=dict(S1=summ(v0["S1"]), S2=summ(v0["S2"]),
                                    S3=summ(v0["S3"])),
                       D1=summ(v0["D1"]), Drel=summ(v0["Drel"]),
                       pct_obs_Drel=float(100 * np.mean(v0["Drel"] <= obs["Drel"])))
    print(f"\n  flat truth (cannot match S1): S1 {v0['S1'].mean():+.4f}, "
          f"Drel {v0['Drel'].mean():+.6f} +- {v0['Drel'].std(ddof=1):.6f}")

    # the separation between the two structured hypotheses, in sd units
    a, b = res["abs"], res["scaled"]
    sep_Drel = (b["Drel"]["mean"] - a["Drel"]["mean"]) / math.sqrt(
        0.5 * (a["Drel"]["sd"] ** 2 + b["Drel"]["sd"] ** 2))
    sep_D1 = (b["D1"]["mean"] - a["D1"]["mean"]) / math.sqrt(
        0.5 * (a["D1"]["sd"] ** 2 + b["D1"]["sd"] ** 2))
    OUT["amplitude_matched"] = dict(observed=obs, by_truth=res,
                                    separation_scaled_vs_physical_Drel_sigma=float(sep_Drel),
                                    separation_scaled_vs_physical_D1_sigma=float(sep_D1))
    print(f"\nSEPARATION between 'organised by r/R500' and 'organised by r', at "
          f"matched S1 and S2:")
    print(f"   Drel: {sep_Drel:+.3f} sigma      D1: {sep_D1:+.3f} sigma")
    print(f"   -> the two hypotheses are {abs(sep_Drel):.2f} sigma apart. This IS "
          f"the power of\n      the whole Job-2 comparison, and it is the number "
          f"the verdict must be read against.")

    json.dump(OUT, open("job2c_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote job2c_results.json")


if __name__ == "__main__":
    main()
