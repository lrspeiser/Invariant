"""
JOB 2 -- the real relation under four radial definitions, each with its own null,
plus the POWER of the discriminating test.

Radial definitions
  r_phys    physical kpc, no normalisation at all
  r/R500_X  HYDROSTATIC R500 from the X-COP FITS header. THIS IS WHAT LANE 12
            USED.  It is not a weak-lensing radius.
  r/R500_WL Herbonnet+2020 deprojected-aperture weak-lensing R500. Independent
            mass, 4 of 12 clusters.
  r/R_b     gas only: R_b,gas (mean enclosed gas density = 500 rho_c f_b) and
            R_b,ne (n_e crosses 1e-4 cm^-3). No total mass anywhere.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import nullsim as N

KPC, G = I.KPC, I.G
OUT = {}


def stats(r, y, t):
    return dict(S1_spearman=N.spear(t, y), S1_pearson=N.pear(np.log10(t), y),
                S2_collapse_rms=N.collapse_rms(t, y),
                S3_slope=float(np.polyfit(np.log10(t), y, 1)[0]))


def scramble_null(r, y, names, radmap, nperm=20000, seed=7):
    """permute the per-cluster normalising radius across clusters."""
    rng = np.random.default_rng(seed)
    ks = sorted(radmap)
    idx = {c: np.where(names == c)[0] for c in ks}
    V = np.array([radmap[c] for c in ks])
    s1 = np.empty(nperm); s2 = np.empty(nperm); s3 = np.empty(nperm)
    t = np.empty(len(r))
    for i in range(nperm):
        p = rng.permutation(len(ks))
        for j, c in enumerate(ks):
            t[idx[c]] = V[p[j]]
        u = r / t
        s1[i] = N.spear(u, y)
        s2[i] = N.collapse_rms(u, y)
        s3[i] = float(np.polyfit(np.log10(u), y, 1)[0])
    return s1, s2, s3


def summ(a, obs):
    a = np.asarray(a, float)
    return dict(null_mean=float(a.mean()), null_sd=float(a.std(ddof=1)),
                null_median=float(np.median(a)),
                null_p05=float(np.percentile(a, 5)),
                null_p95=float(np.percentile(a, 95)),
                observed=float(obs),
                percentile_of_observed=float(100 * np.mean(a <= obs)))


def main():
    cl = I.load_all(verbose=False)
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    names = np.array([p["name"] for p in pts])
    y = I.rar_residual(gb, go)

    byname = {c["name"]: c for c in cl}
    R500_X = {c["name"]: c["R500_hse"] for c in cl}
    Rb_gas, Rb_ne = {}, {}
    for c in cl:
        a, b = I.baryonic_radii(c)
        Rb_gas[c["name"]] = a
        Rb_ne[c["name"]] = b
    herb = I.herbonnet_for_xcop(verbose=False)
    R500_WL = {k: v["R500_ap_Mpc"] * 1000 * KPC for k, v in herb.items()}

    # ---------------------------------------------------------- provenance table
    prov = []
    for c in sorted(R500_X):
        prov.append(dict(
            cluster=c, z=byname[c]["z"],
            R500_X_kpc=R500_X[c] / KPC,
            eR500_X_kpc=byname[c]["eR500_hse"] / KPC,
            M500_X_1e14=byname[c]["M500_hse"] / 1e14 / I.MSUN,
            R500_WL_kpc=(R500_WL[c] / KPC) if c in R500_WL else None,
            M500_WL_NFW_1e14=herb[c]["M500_nfw"] if c in herb else None,
            Rb_gas_kpc=Rb_gas[c] / KPC,
            Rb_ne_kpc=Rb_ne[c] / KPC,
            WL_over_X=(R500_WL[c] / R500_X[c]) if c in R500_WL else None,
            Rbgas_over_X=Rb_gas[c] / R500_X[c],
            Rbne_over_X=Rb_ne[c] / R500_X[c],
        ))
    OUT["provenance_table"] = prov
    wl = [p["WL_over_X"] for p in prov if p["WL_over_X"] is not None]
    OUT["provenance_summary"] = dict(
        n_clusters=len(prov), n_with_WL=len(wl),
        WL_over_X_median=float(np.median(wl)),
        WL_over_X_scatter_dex=float(np.std(np.log10(wl), ddof=1)),
        Rbgas_over_X_median=float(np.median([p["Rbgas_over_X"] for p in prov])),
        Rbgas_over_X_scatter_dex=float(np.std(np.log10([p["Rbgas_over_X"] for p in prov]), ddof=1)),
        Rbne_over_X_median=float(np.median([p["Rbne_over_X"] for p in prov])),
        Rbne_over_X_scatter_dex=float(np.std(np.log10([p["Rbne_over_X"] for p in prov]), ddof=1)),
        R500_X_range_kpc=[float(min(R500_X.values()) / KPC), float(max(R500_X.values()) / KPC)],
        lnR500_X_sd=float(np.std(np.log([v / KPC for v in R500_X.values()]), ddof=1)),
        ln_r_within_cluster_sd=float(np.mean([np.std(np.log(r[names == c]), ddof=1)
                                              for c in sorted(R500_X)])),
    )
    print("PROVENANCE")
    print(f"{'cluster':<9}{'R500_X':>9}{'R500_WL':>9}{'Rb,gas':>9}{'Rb,ne':>9}"
          f"{'WL/X':>7}{'gas/X':>7}{'ne/X':>7}")
    for p in prov:
        print(f"{p['cluster']:<9}{p['R500_X_kpc']:9.1f}"
              f"{(p['R500_WL_kpc'] or float('nan')):9.1f}{p['Rb_gas_kpc']:9.1f}"
              f"{p['Rb_ne_kpc']:9.1f}"
              f"{(p['WL_over_X'] or float('nan')):7.3f}{p['Rbgas_over_X']:7.3f}"
              f"{p['Rbne_over_X']:7.3f}")
    print(f"  ln R500_X sd across clusters   = {OUT['provenance_summary']['lnR500_X_sd']:.4f}")
    print(f"  ln r sd WITHIN a cluster (mean)= {OUT['provenance_summary']['ln_r_within_cluster_sd']:.4f}")

    # ---------------------------------------------------------- the four relations
    print("\nTHE RELATION UNDER EACH RADIAL DEFINITION (all 12 clusters where possible)")
    defs = {
        "r_physical": ({c: 1000 * KPC for c in R500_X}, "no normalisation (radius in Mpc)"),
        "r_over_R500_X": (R500_X, "HYDROSTATIC X-ray R500 -- what lane 12 used"),
        "r_over_Rb_gas": (Rb_gas, "gas-only overdensity radius, no total mass"),
        "r_over_Rb_ne": (Rb_ne, "n_e = 1e-4 cm^-3 radius, no total mass, no integral"),
    }
    rel = {}
    for k, (rm, desc) in defs.items():
        t = np.array([rm[n] for n in names])
        st = stats(r, y, r / t)
        s1, s2, s3 = scramble_null(r, y, names, rm)
        rel[k] = dict(description=desc, n_points=int(len(r)),
                      n_clusters=int(len(rm)), **st,
                      scramble_S1=summ(s1, st["S1_spearman"]),
                      scramble_S2=summ(s2, st["S2_collapse_rms"]),
                      scramble_S3=summ(s3, st["S3_slope"]))
        print(f"  {k:<16} S1={st['S1_spearman']:+.4f} (null {s1.mean():+.4f}+-{s1.std(ddof=1):.4f}, "
              f"pct {100*np.mean(s1<=st['S1_spearman']):5.1f})  "
              f"S2={st['S2_collapse_rms']:.4f} (null {s2.mean():.4f}, "
              f"pct {100*np.mean(s2<=st['S2_collapse_rms']):5.1f})  "
              f"S3={st['S3_slope']:+.4f}")
    OUT["relations_all12"] = rel

    # ------------------------------------------------- WL subset, matched comparison
    print("\nWEAK-LENSING SUBSET (the only place r/R500_WL exists)")
    sub = np.isin(names, sorted(R500_WL))
    rs, ys, ns = r[sub], y[sub], names[sub]
    relw = {}
    subdefs = {
        "r_physical": {c: 1000 * KPC for c in R500_WL},
        "r_over_R500_X": {c: R500_X[c] for c in R500_WL},
        "r_over_R500_WL": R500_WL,
        "r_over_Rb_gas": {c: Rb_gas[c] for c in R500_WL},
        "r_over_Rb_ne": {c: Rb_ne[c] for c in R500_WL},
    }
    for k, rm in subdefs.items():
        t = np.array([rm[n] for n in ns])
        st = stats(rs, ys, rs / t)
        s1, s2, s3 = scramble_null(rs, ys, ns, rm, nperm=5000)
        relw[k] = dict(n_points=int(sub.sum()), n_clusters=len(rm), **st,
                       scramble_S1=summ(s1, st["S1_spearman"]),
                       scramble_S2=summ(s2, st["S2_collapse_rms"]))
        print(f"  {k:<16} S1={st['S1_spearman']:+.4f}  S2={st['S2_collapse_rms']:.4f} "
              f"(null {s2.mean():.4f}+-{s2.std(ddof=1):.4f}, pct "
              f"{100*np.mean(s2<=st['S2_collapse_rms']):5.1f})  S3={st['S3_slope']:+.4f}")
    OUT["relations_WL_subset"] = dict(
        clusters=sorted(R500_WL), n_points=int(sub.sum()), by_definition=relw,
        note="only 4 of 12 X-COP clusters appear in Herbonnet+2020; with 4 objects "
             "the permutation null has 24 states, so this comparison has almost no "
             "resolving power and is reported for completeness")

    # ---------------------------------------------------------- POWER
    print("\nPOWER of the scrambling test: can it SEE genuine self-similarity?")
    TS = [N.Template(c) for c in cl]
    power = {}
    for label, kw in (("flat truth (null)", dict()),
                      ("true r/R500 organisation, s=-0.5", dict(s_scaled=-0.5)),
                      ("true r/R500 organisation, s=-1.0", dict(s_scaled=-1.0)),
                      ("true r/R500 organisation, s=-1.354", dict(s_scaled=-1.354)),
                      ("true PHYSICAL-radius organisation, s=-0.472", dict(s_abs=-0.472))):
        rng = np.random.default_rng(55)
        pc1, pc2 = [], []
        for _ in range(60):
            res = N.one_realisation(TS, rng, N.DEFAULT_CFG, **kw)
            if res is None:
                continue
            rm = res["R500_obs_map"]
            s1, s2, _ = scramble_null(res["r"], res["y"], res["name"], rm,
                                      nperm=400, seed=int(rng.integers(1 << 30)))
            o1 = N.spear(res["r"] / res["R500_obs"], res["y"])
            o2 = N.collapse_rms(res["r"] / res["R500_obs"], res["y"])
            pc1.append(100 * np.mean(s1 <= o1))
            pc2.append(100 * np.mean(s2 <= o2))
        pc1, pc2 = np.array(pc1), np.array(pc2)
        power[label] = dict(n_real=len(pc1),
                            S1_percentile_mean=float(pc1.mean()),
                            S1_percentile_sd=float(pc1.std(ddof=1)),
                            S1_power_at_5pct=float(np.mean(pc1 <= 5)),
                            S2_percentile_mean=float(pc2.mean()),
                            S2_percentile_sd=float(pc2.std(ddof=1)),
                            S2_power_at_5pct=float(np.mean(pc2 <= 5)))
        print(f"  {label:<40} S1 pct {pc1.mean():5.1f}+-{pc1.std(ddof=1):4.1f} "
              f"power {np.mean(pc1<=5):.2f} | S2 pct {pc2.mean():5.1f}"
              f"+-{pc2.std(ddof=1):4.1f} power {np.mean(pc2<=5):.2f}")
    OUT["scramble_test_power"] = dict(
        alpha=0.05, description="fraction of realisations in which the TRUE R500 "
        "assignment lands at or below the 5th percentile of its own permutation "
        "null. For a test with power, this should be near 1 when the truth really "
        "is organised by r/R500 and near 0.05 when it is not.",
        results=power)

    # ------------------------------------------- how much can R500 possibly move things
    lnR = np.log(np.array([R500_X[c] for c in sorted(R500_X)]))
    within = np.mean([np.std(np.log(r[names == c]), ddof=1) for c in sorted(R500_X)])
    OUT["leverage"] = dict(
        sd_ln_R500_across_clusters=float(np.std(lnR, ddof=1)),
        sd_ln_r_within_cluster=float(within),
        ratio=float(np.std(lnR, ddof=1) / within),
        note="the normalisation shifts each cluster's points bodily along ln r by "
             "its own ln R500. When that spread is small compared with the radial "
             "span inside a cluster, r/R500 and r are near-identical variables and "
             "no data set of this shape can tell them apart.")
    print(f"\nLEVERAGE: sd(ln R500) across clusters = {np.std(lnR, ddof=1):.4f}, "
          f"sd(ln r) within a cluster = {within:.4f}, ratio = "
          f"{np.std(lnR, ddof=1)/within:.3f}")

    json.dump(OUT, open("job2_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote job2_results.json")


if __name__ == "__main__":
    main()
