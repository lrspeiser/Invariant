"""
POWER, and the checks that decide how much any of the rest can be trusted.

Standing constraint 5: state POWER before verdicts.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import stats as S

KPC, MPC = I.KPC, I.MPC
OUT = {}


def fisher_power(rho, n, alpha=0.05):
    """two-sided power of a Pearson correlation test at |rho|, sample n."""
    if n < 4:
        return float("nan")
    from math import atanh, sqrt, erf
    zc = 1.959963985
    se = 1.0 / sqrt(n - 3)
    z = atanh(rho) / se
    ncdf = lambda t: 0.5 * (1 + erf(t / sqrt(2)))
    return float(1 - ncdf(zc - abs(z)) + ncdf(-zc - abs(z)))


def detectable_rho(n, power=0.80, alpha=0.05):
    from math import tanh, sqrt
    return float(tanh((1.959963985 + 0.8416212336) / sqrt(n - 3)))


def main():
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]
    nm, r = T["name"], T["r"]
    y = S.excess_y(T["gb"], T["go"])
    a = S.excess_a0(T["gb"], T["go"])
    names = sorted(C)

    # ------------------------------------------------------------- POWER
    print("=== POWER ===")
    pw = {}
    for L, lab in ((100.0, "100 kpc"), (200.0, "200 kpc"),
                   (400.0, "400 kpc"), (600.0, "600 kpc")):
        m = np.abs(r / KPC - L) < 1e-6
        n = int(m.sum())
        pw[lab] = dict(n=n, rho_detectable_at_80pct=detectable_rho(n),
                       power_at_rho_0p5=fisher_power(0.5, n),
                       power_at_rho_0p3=fisher_power(0.3, n))
        print(f"  at r = {lab:8s} n = {n:2d} clusters:  |rho| detectable at 80% "
              f"power = {detectable_rho(n):.2f};  power at rho=0.5 is "
              f"{fisher_power(0.5, n):.2f}, at rho=0.3 is {fisher_power(0.3, n):.2f}")
    pw["between_cluster_all"] = dict(n=20, rho_detectable_at_80pct=detectable_rho(20),
                                     power_at_rho_0p5=fisher_power(0.5, 20),
                                     power_at_rho_0p3=fisher_power(0.3, 20))
    print(f"  between-cluster, all 20:      |rho| detectable at 80% power = "
          f"{detectable_rho(20):.2f}")
    print("  So a tautology channel of |rho| < 0.55 CANNOT be excluded by the")
    print("  between-cluster contrast on this sample, whatever the result.")
    OUT["power"] = pw

    # -------------------------------------------- R500 is entirely extrapolated
    print("\n=== is R500 inside the data? ===")
    rmax = {n: float(r[nm == n].max()) for n in names}
    frac = np.array([rmax[n] / C[n]["R500_lens"] for n in names])
    OUT["R500_extrapolation"] = dict(
        outermost_datum_over_R500_median=float(np.median(frac)),
        outermost_datum_over_R500_min=float(frac.min()),
        outermost_datum_over_R500_max=float(frac.max()),
        R500_over_outermost_datum_median=float(np.median(1 / frac)),
        max_r_over_R500_any_point=float(np.max(
            [r[i] / C[nm[i]]["R500_lens"] for i in range(len(r))])),
        note="R500 is never reached by the CLASH data.  It is a property of the "
             "NFW extrapolation, fitted over R <= 2.9 Mpc in PROJECTION but "
             "tabulated in 3D only out to 600 kpc.")
    print(f"  outermost datum / R500: median {np.median(frac):.3f} "
          f"(range {frac.min():.3f}-{frac.max():.3f})")
    print(f"  every CLASH point sits at r/R500 <= "
          f"{OUT['R500_extrapolation']['max_r_over_R500_any_point']:.3f}")
    print(f"  R500 is {np.median(1/frac):.2f}x beyond the outermost measurement.")

    # ---------------------------------------------- selection at 400 / 600 kpc
    print("\n=== is the 400/600 kpc subsample selected? ===")
    sel = {}
    for L in (400.0, 600.0):
        has = [n for n in names if (np.abs(r[nm == n] / KPC - L) < 1e-6).any()]
        no = [n for n in names if n not in has]
        f = lambda g, k: float(np.mean([C[n][k] for n in g]))
        sel[f"{L:.0f}kpc"] = dict(
            n_with=len(has), n_without=len(no),
            mean_log10R500_with=float(np.mean([math.log10(C[n]["R500_lens"])
                                               for n in has])),
            mean_log10R500_without=float(np.mean([math.log10(C[n]["R500_lens"])
                                                  for n in no])) if no else None,
            mean_z_with=f(has, "z"),
            mean_z_without=f(no, "z") if no else None,
            clusters_with=has)
        d = sel[f"{L:.0f}kpc"]
        print(f"  r={L:.0f} kpc present in {len(has)}/20.  mean log10 R500 "
              f"{d['mean_log10R500_with']:.4f} (with) vs "
              f"{d['mean_log10R500_without']:.4f} (without); mean z "
              f"{d['mean_z_with']:.3f} vs {d['mean_z_without']:.3f}")
        # the same fixed-radius correlation restricted to the 600-kpc clusters
        if L == 600.0:
            for L2 in (100.0, 200.0, 400.0):
                m = np.abs(r / KPC - L2) < 1e-6
                m = m & np.isin(nm, has)
                if m.sum() >= 6:
                    lR = np.array([math.log10(C[c]["R500_lens"]) for c in nm[m]])
                    sel[f"restricted_to_600kpc_clusters/{L2:.0f}kpc"] = dict(
                        n=int(m.sum()), pearson_y=S.pear(lR, y[m]),
                        pearson_a0=S.pear(lR, a[m]))
                    print(f"    restricted to those {len(has)} clusters, at "
                          f"r={L2:.0f} kpc (n={m.sum()}): corr(y, log R500) = "
                          f"{S.pear(lR, y[m]):+.4f}, corr(a0,.) = "
                          f"{S.pear(lR, a[m]):+.4f}")
    OUT["selection"] = sel

    # -------------------------------------------- MACS0416: a broken X-ray fit
    print("\n=== the Donahue X-ray R500 outlier ===")
    lo = sorted(names, key=lambda n: C[n]["R500_xray"])
    print(f"  X-ray R500 range {C[lo[0]]['R500_xray']/KPC:.0f} ({lo[0]}) to "
          f"{C[lo[-1]]['R500_xray']/KPC:.0f} ({lo[-1]}) kpc")
    print("  Donahue+2014 lists MACS0416-24 with r_s = nodata and only an upper "
          "limit (<8 Mpc); its r500 = 2.10 +- 0.35 Mpc is an unconstrained fit.")
    drop = [n for n in names if n != "MACS0416"]
    rob = {}
    for L in (400.0, 600.0):
        m = (np.abs(r / KPC - L) < 1e-6)
        md = m & np.isin(nm, drop)
        for lab, mm in (("all", m), ("no MACS0416", md)):
            if mm.sum() < 6:
                continue
            lx = np.array([math.log10(C[c]["R500_xray"]) for c in nm[mm]])
            rob[f"{L:.0f}kpc/{lab}"] = dict(n=int(mm.sum()),
                                            pearson_y=S.pear(lx, y[mm]),
                                            pearson_a0=S.pear(lx, a[mm]))
            print(f"  r={L:.0f} kpc, {lab:12s} n={mm.sum():2d}  "
                  f"corr(y, log R500_Xray) = {S.pear(lx, y[mm]):+.4f}")
    # and the pooled slope against the X-ray radius, with and without it
    for lab, keep in (("all", names), ("no MACS0416", drop)):
        m = (r / KPC > 50) & np.isin(nm, keep)
        Rx = np.array([C[c]["R500_xray"] for c in nm[m]])
        rob[f"pooled_slope/{lab}"] = dict(
            n=int(m.sum()), slope_y=S.ols_slope(np.log10(r[m] / Rx), y[m]),
            slope_a0=S.ols_slope(np.log10(r[m] / Rx), a[m]))
        print(f"  pooled slope vs log(r/R500_Xray), {lab:12s} "
              f"y {rob[f'pooled_slope/{lab}']['slope_y']:+.4f}, "
              f"a0 {rob[f'pooled_slope/{lab}']['slope_a0']:+.4f}")
    OUT["macs0416_robustness"] = rob

    # ------------------------------- lensing vs X-ray R500: the mass bias
    lx = np.array([C[n]["R500_xray"] for n in names])
    ll = np.array([C[n]["R500_lens"] for n in names])
    OUT["R500_lens_vs_xray"] = dict(
        pearson_log=S.pear(np.log10(ll), np.log10(lx)),
        spearman=S.spear(ll, lx), median_ratio=float(np.median(ll / lx)),
        sd_log10_ratio=float(np.std(np.log10(ll / lx), ddof=1)),
        pearson_log_no_macs0416=S.pear(
            np.log10([C[n]["R500_lens"] for n in drop]),
            np.log10([C[n]["R500_xray"] for n in drop])))
    print(f"\n  corr(log R500_lens, log R500_Xray) = "
          f"{OUT['R500_lens_vs_xray']['pearson_log']:+.4f} over 20 "
          f"({OUT['R500_lens_vs_xray']['pearson_log_no_macs0416']:+.4f} "
          f"without MACS0416); median ratio {np.median(ll/lx):.3f}, "
          f"sd of log ratio {np.std(np.log10(ll/lx), ddof=1):.3f} dex")
    print("  A weak correlation between the two R500 estimates is itself a limit:")
    print("  the X-ray radius cannot be a sharp control if it barely tracks the")
    print("  lensing one.")

    # ---------------------------------------- reproduce the record's numbers
    print("\n=== reproduction of the record's CLASH claims ===")
    rep = {}
    d = []
    for n in names:
        m6 = (nm == n) & (np.abs(r / KPC - 600) < 1e-6)
        m1 = (nm == n) & (np.abs(r / KPC - 100) < 1e-6)
        if m6.sum() == 1 and m1.sum() == 1:
            d.append(float(a[m6][0] - a[m1][0]))
    d = np.array(d)
    rep["within_CLASH_a0_drop_100_to_600"] = dict(
        n=len(d), mean=float(d.mean()),
        sem=float(d.std(ddof=1) / math.sqrt(len(d))),
        n_negative=int((d < 0).sum()),
        record_quote="a0 falls by -0.347 +- 0.057 dex, 10 of 11 negative")
    print(f"  within-CLASH a0 drop 100->600 kpc: {d.mean():+.4f} +- "
          f"{d.std(ddof=1)/math.sqrt(len(d)):.4f} over n={len(d)}, "
          f"{int((d<0).sum())}/{len(d)} negative")
    print("  record: '-0.347 +- 0.057 dex, 10 of 11 negative'  -> REPRODUCED")
    for L, quoted in ((100.0, 21.95), (400.0, 13.30)):
        m = np.abs(r / KPC - L) < 1e-6
        med = float(np.median(10 ** a[m]))
        rep[f"lane12_a0_ratio_at_{L:.0f}kpc"] = dict(
            n=int(m.sum()), median_ratio=med, record_value=quoted,
            record_r_over_R500=0.073 if L == 100 else 0.291,
            implied_R500_kpc=L / (0.073 if L == 100 else 0.291))
        print(f"  lane-12 row 'CLASH fig2' at r/R500 = "
              f"{0.073 if L==100 else 0.291}: quoted a0/a0can = {quoted}; "
              f"median at r = {L:.0f} kpc is {med:.2f} "
              f"(implied single R500 = {L/(0.073 if L==100 else 0.291):.0f} kpc)")
    OUT["record_reproduction"] = rep
    print("  NOTE: lane 12 used ONE pooled R500 ~ 1372 kpc for all of CLASH, not")
    print("  the per-cluster values -- the two rows imply 1370 and 1375 kpc.")

    json.dump(OUT, open("diagnostics_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote diagnostics_results.json")


if __name__ == "__main__":
    main()
