"""
JOB 2B -- the properly calibrated discriminator.

Job 2's permutation ("scrambling") test turned out to be badly mis-sized: under a
FLAT truth with no radial structure at all it rejects at the nominal 5% level in
57-65% of realisations, because the inferred R500 is correlated with the
cluster's own excess.  That is the tautology, and it makes the permutation null
anti-conservative.  A statistic calibrated against it cannot be believed.

So we do two things properly:

  (1) DOUBLE CALIBRATION.  The statistic is the *percentile* of the observed
      value within its own permutation null; its null distribution comes from
      the forward flat-truth simulation, which contains the tautology.

  (2) A STATISTIC THAT ASKS THE RIGHT QUESTION.  "Is the excess organised by
      r/R500?" is not "is R500 correlated with the excess" -- it is "does
      normalising by R500 organise the data BETTER THAN NOT NORMALISING".

          D1 = S1(r/R500) - S1(r)              more negative = R500 helps
          D2 = S2(r/R500) - S2(r)              more negative = tighter collapse

      D1 and D2 are calibrated under three truths: flat, physical-radius
      organisation, and genuine scaled-radius organisation.
"""
from __future__ import annotations
import json

import numpy as np

import ingest as I
import nullsim as N
from run_job2 import scramble_null

KPC = I.KPC
OUT = {}


def D_of(res):
    r, y = res["r"], res["y"]
    t = res["R500_obs"]
    return (N.spear(r / t, y) - N.spear(r, y),
            N.collapse_rms(r / t, y) - N.collapse_rms(r / (1000 * KPC), y))


def run_truth(TS, kw, n=250, seed=909):
    rng = np.random.default_rng(seed)
    d1, d2, s1, s2 = [], [], [], []
    for _ in range(n):
        res = N.one_realisation(TS, rng, N.DEFAULT_CFG, **kw)
        if res is None:
            continue
        a, b = D_of(res)
        d1.append(a); d2.append(b)
        s1.append(N.spear(res["r"] / res["R500_obs"], res["y"]))
        s2.append(N.collapse_rms(res["r"] / res["R500_obs"], res["y"]))
    return (np.array(d1), np.array(d2), np.array(s1), np.array(s2))


def summ(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                median=float(np.median(a)),
                p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)))


def main():
    cl = I.load_all(verbose=False)
    TS = [N.Template(c) for c in cl]
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    names = np.array([p["name"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    y = I.rar_residual(gb, go)

    obs_D1 = N.spear(r / R5, y) - N.spear(r, y)
    obs_D2 = N.collapse_rms(r / R5, y) - N.collapse_rms(r / (1000 * KPC), y)
    print(f"OBSERVED  D1 = {obs_D1:+.5f}   D2 = {obs_D2:+.5f}")

    truths = [
        ("flat", dict(), "no radial structure of any kind in the excess"),
        ("physical_radius", dict(s_abs=-0.472),
         "excess organised by PHYSICAL radius with the observed slope"),
        ("scaled_radius_0p5", dict(s_scaled=-0.5), "genuine r/R500 organisation"),
        ("scaled_radius_1p0", dict(s_scaled=-1.0), "genuine r/R500 organisation"),
        ("scaled_radius_1p354", dict(s_scaled=-1.354),
         "genuine r/R500 organisation at the record's quoted slope"),
    ]
    res = {}
    print(f"\n{'truth':<22}{'D1 mean+-sd':>22}{'D2 mean+-sd':>24}"
          f"{'pct(obsD1)':>12}{'pct(obsD2)':>12}")
    for key, kw, desc in truths:
        d1, d2, s1, s2 = run_truth(TS, kw)
        res[key] = dict(description=desc, injection=kw, n_real=int(len(d1)),
                        D1=summ(d1), D2=summ(d2),
                        S1_hse=summ(s1), S2_hse=summ(s2),
                        percentile_of_observed_D1=float(100 * np.mean(d1 <= obs_D1)),
                        percentile_of_observed_D2=float(100 * np.mean(d2 <= obs_D2)))
        print(f"{key:<22}{f'{d1.mean():+.5f} +- {d1.std(ddof=1):.5f}':>22}"
              f"{f'{d2.mean():+.5f} +- {d2.std(ddof=1):.5f}':>24}"
              f"{100*np.mean(d1<=obs_D1):12.1f}{100*np.mean(d2<=obs_D2):12.1f}")
    OUT["discriminator"] = dict(
        observed_D1=float(obs_D1), observed_D2=float(obs_D2),
        definition="D1 = Spearman(r/R500,y) - Spearman(r,y);  "
                   "D2 = collapse_rms(r/R500) - collapse_rms(r).  Negative means "
                   "the R500 normalisation organises the data better than no "
                   "normalisation at all.",
        by_truth=res)

    # separation between the flat null and each alternative, in sd units
    f1 = np.array([res["flat"]["D1"]["mean"], res["flat"]["D1"]["sd"]])
    f2 = np.array([res["flat"]["D2"]["mean"], res["flat"]["D2"]["sd"]])
    sep = {}
    for key in res:
        if key == "flat":
            continue
        sep[key] = dict(
            D1_sigma=float((res[key]["D1"]["mean"] - f1[0]) / f1[1]),
            D2_sigma=float((res[key]["D2"]["mean"] - f2[0]) / f2[1]))
    OUT["discriminator_separation_sigma"] = sep
    print("\nseparation of each alternative from the flat null, in flat-null sd:")
    for k, v in sep.items():
        print(f"   {k:<22} D1 {v['D1_sigma']:+7.2f} sigma   D2 {v['D2_sigma']:+7.2f} sigma")

    # observed against the flat null, in sd units
    OUT["observed_vs_flat_null"] = dict(
        D1_sigma=float((obs_D1 - f1[0]) / f1[1]),
        D2_sigma=float((obs_D2 - f2[0]) / f2[1]))
    print(f"\nOBSERVED against the flat null: D1 {(obs_D1-f1[0])/f1[1]:+.2f} sigma, "
          f"D2 {(obs_D2-f2[0])/f2[1]:+.2f} sigma")

    # ---------------------------------------------- double-calibrated permutation test
    print("\nDOUBLE-CALIBRATED PERMUTATION TEST")
    s1p, s2p, _ = scramble_null(r, y, names,
                                {c["name"]: c["R500_hse"] for c in cl}, nperm=20000)
    obs_S1 = N.spear(r / R5, y)
    obs_S2 = N.collapse_rms(r / R5, y)
    P1 = float(100 * np.mean(s1p <= obs_S1))
    P2 = float(100 * np.mean(s2p <= obs_S2))
    print(f"   observed percentile within its own permutation null: "
          f"P(S1) = {P1:.2f}, P(S2) = {P2:.2f}")

    rng = np.random.default_rng(77)
    pc1, pc2 = [], []
    for _ in range(150):
        rr = N.one_realisation(TS, rng, N.DEFAULT_CFG)
        if rr is None:
            continue
        a, b, _ = scramble_null(rr["r"], rr["y"], rr["name"], rr["R500_obs_map"],
                                nperm=400, seed=int(rng.integers(1 << 30)))
        pc1.append(100 * np.mean(a <= N.spear(rr["r"] / rr["R500_obs"], rr["y"])))
        pc2.append(100 * np.mean(b <= N.collapse_rms(rr["r"] / rr["R500_obs"], rr["y"])))
    pc1, pc2 = np.array(pc1), np.array(pc2)
    OUT["double_calibrated_permutation"] = dict(
        observed_percentile_S1=P1, observed_percentile_S2=P2,
        flat_null_percentile_S1=summ(pc1), flat_null_percentile_S2=summ(pc2),
        false_positive_rate_at_nominal_5pct_S1=float(np.mean(pc1 <= 5)),
        false_positive_rate_at_nominal_5pct_S2=float(np.mean(pc2 <= 5)),
        p_value_S1=float(np.mean(pc1 <= P1)),
        p_value_S2=float(np.mean(pc2 <= P2)),
        note="the naive permutation p-value is anti-conservative because the "
             "inferred R500 is correlated with the cluster's own excess. The "
             "correctly calibrated p-value asks how often a FLAT truth produces a "
             "permutation percentile at least as extreme as the observed one.")
    print(f"   flat-truth null for that percentile: S1 {pc1.mean():.1f} +- "
          f"{pc1.std(ddof=1):.1f}, S2 {pc2.mean():.1f} +- {pc2.std(ddof=1):.1f}")
    print(f"   false-positive rate of the NAIVE test at nominal 5%: "
          f"S1 {np.mean(pc1<=5):.2f}, S2 {np.mean(pc2<=5):.2f}")
    print(f"   CALIBRATED p-value: S1 p = {np.mean(pc1<=P1):.3f}, "
          f"S2 p = {np.mean(pc2<=P2):.3f}")

    json.dump(OUT, open("job2b_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote job2b_results.json")


if __name__ == "__main__":
    main()
