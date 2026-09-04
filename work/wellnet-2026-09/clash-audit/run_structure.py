"""
JOBS 2.4, 2.5, 2.6 -- variance decomposition, the four radial definitions, and
the rank identity.

2.6 first, because it controls how 2.5 must be read.

Run AT proved for X-COP that log(r/R500_i) = log r - log R500_i and log R500_i is
constant within cluster i, hence already in the span of the cluster indicators;
so with per-cluster levels, r/R500 and r are the SAME regressor (rank 13 = 13).

CLASH's binned table DOES carry per-cluster levels -- Tian+2020 fig2.dat has an
AName column, 84 rows over 20 named clusters.  (The bench does not use it:
invariant_bench._clash() reads q[2],q[3],q[4] and discards q[1]=AName, which is
why the record says CLASH has no object identity.  The identity is in the file.)
So the identity applies.

BUT CLASH's radial grid is COMMON ACROSS CLUSTERS -- 100/200/400/600 kpc exactly,
plus one per-cluster BCG radius.  That makes the design crossed rather than
nested, and it separates the two contrasts perfectly:

  within-cluster, across levels : r varies, R500_i fixed
                                  -> r and r/R500 carry identical information
  between-cluster, at one level : r FIXED, R500_i varies
                                  -> r/R500 varies ONLY through R500_i, so any
                                     correlation of the excess with r/R500 at
                                     fixed r IS a correlation with -log R500_i

The second contrast is the tautology in its purest available form, and CLASH is
the only sample in this programme that isolates it.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import stats as S

KPC, MPC = I.KPC, I.MPC
OUT = {}
LEVELS = (100.0, 200.0, 400.0, 600.0)


def subsets(T):
    r = T["r"] / KPC
    return {"all_84": np.ones(len(r), bool), "cluster_scale_64": r > 50.0}


def main():
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]
    nm, r = T["name"], T["r"]
    stats = {"y": S.excess_y(T["gb"], T["go"]),
             "a0": S.excess_a0(T["gb"], T["go"])}
    x, norm, Rb, thr = S.radial_definitions(T, C)
    OUT["baryon_thresholds"] = thr
    OUT["baryon_radii_kpc"] = {c: {k: (v / KPC if k.endswith(("gas", "M", "g"))
                                      and np.isfinite(v) else v)
                                   for k, v in Rb[c].items()} for c in Rb}
    OUT["normalisers_kpc"] = {c: {k: (v / KPC if np.isfinite(v) else None)
                                  for k, v in norm[c].items()} for c in norm}

    # ---------------------------------------------------------------- 2.6
    print("=== 2.6  THE RANK IDENTITY ===")
    ident = {}
    for sub, mask in subsets(T).items():
        g = nm[mask]
        names = sorted(set(g.tolist()))
        Dm = np.column_stack([(g == c).astype(float) for c in names])
        lr = x["r_physical"][mask]
        lt = x["r_over_R500_lens"][mask]
        A = np.column_stack([Dm, lr]); B = np.column_stack([Dm, lt])
        Cc = np.column_stack([Dm, lr, lt])
        ra, rb, rc = (int(np.linalg.matrix_rank(M, tol=1e-9)) for M in (A, B, Cc))
        coef, *_ = np.linalg.lstsq(A, lt, rcond=None)
        res = lt - A @ coef
        # radius-LEVEL indicators (the crossed part of the design)
        lev = np.round(r[mask] / KPC, 1)
        levs = sorted(set(lev.tolist()))
        Lm = np.column_stack([(lev == L).astype(float) for L in levs])
        TW = np.column_stack([Dm, Lm])
        rtw = int(np.linalg.matrix_rank(TW, tol=1e-9))
        rtw2 = int(np.linalg.matrix_rank(np.column_stack([TW, lt]), tol=1e-9))
        ident[sub] = dict(
            n=int(mask.sum()), n_clusters=len(names), n_radius_levels=len(levs),
            rank_indicators_plus_logr=ra, rank_indicators_plus_log_r_over_R500=rb,
            rank_indicators_plus_BOTH=rc, columns_of_BOTH=int(Cc.shape[1]),
            extra_directions_from_R500=rc - ra,
            max_abs_residual_of_log_r_over_R500=float(np.abs(res).max()),
            rank_two_way_cluster_x_level=rtw,
            rank_two_way_plus_log_r_over_R500=rtw2,
            extra_directions_over_two_way=rtw2 - rtw)
        print(f"  [{sub}] n={mask.sum()}, {len(names)} clusters, {len(levs)} levels")
        print(f"    rank[ind|log r]={ra}  rank[ind|log(r/R500)]={rb}  "
              f"rank[ind|both]={rc} of {Cc.shape[1]} cols  -> R500 adds {rc-ra}")
        print(f"    residual of log(r/R500) on [ind|log r]: max |e| = "
              f"{np.abs(res).max():.2e}")
        print(f"    two-way [cluster x level] rank {rtw}; adding log(r/R500) "
              f"gives {rtw2} -> adds {rtw2-rtw}")
    OUT["rank_identity"] = ident

    # ---------------------------------------------------------------- 2.4
    print("\n=== 2.4  WITHIN- vs BETWEEN-CLUSTER VARIANCE ===")
    vd = {}
    for sub, mask in subsets(T).items():
        for sn, sv in stats.items():
            v = S.var_decomp(sv[mask], nm[mask])
            vd[f"{sub}/{sn}"] = v
            print(f"  [{sub}] {sn:3s}: total {v['total']:.5f}  "
                  f"between {v['between']:.5f} ({100*v['between_fraction']:.1f}%)  "
                  f"within {v['within']:.5f} ({100*v['within_fraction']:.1f}%)")
    vd["xcop_reference_within_fraction"] = 0.903
    OUT["variance_decomposition"] = vd
    print("  X-COP was 90.3% WITHIN.  The monotone-invariance protection covers")
    print("  only the within part; the between part is fully exposed.")

    # ---------------------------------------------------------------- 2.5
    print("\n=== 2.5  FOUR RADIAL DEFINITIONS -- SLOPES ===")
    slopes = {}
    order = ["r_physical", "r_over_R500_lens", "r_over_R500_xray",
             "r_over_R500_TX", "r_over_Rb_gas", "r_over_Rb_M", "r_over_Rb_g"]
    for sub, mask in subsets(T).items():
        for sn, sv in stats.items():
            print(f"\n  [{sub}] statistic = {sn}")
            print(f"    {'definition':<22}{'pooled':>10}{'within(FE)':>12}"
                  f"{'spearman':>11}{'n':>5}")
            for k in order:
                xx = x[k][mask]
                ok = np.isfinite(xx)
                if ok.sum() < 10:
                    continue
                yy = sv[mask][ok]; xx2 = xx[ok]; gg = nm[mask][ok]
                d = dict(pooled_slope=S.ols_slope(xx2, yy),
                         fe_slope=S.fe_slope(xx2, yy, gg),
                         spearman=S.spear(xx2, yy), n=int(ok.sum()),
                         n_clusters=len(set(gg.tolist())))
                slopes[f"{sub}/{sn}/{k}"] = d
                print(f"    {k:<22}{d['pooled_slope']:+10.4f}{d['fe_slope']:+12.4f}"
                      f"{d['spearman']:+11.4f}{d['n']:5d}")
    OUT["slopes"] = slopes
    # extrapolation bookkeeping for the baryon radii
    ex = {k: int(sum(Rb[c][k + "_extrap"] for c in Rb))
          for k in ("Rb_gas", "Rb_M", "Rb_g")}
    OUT["baryon_radius_extrapolated_clusters"] = ex
    print(f"\n  baryon radii requiring extrapolation beyond the measured profile: "
          f"{ex} of 20")

    # -------------------------------------------------- the pure tautology test
    print("\n=== 2.5b  THE PURE TAUTOLOGY CONTRAST: fixed r, R500 varying ===")
    pure = {}
    for sn, sv in stats.items():
        for L in LEVELS:
            m = np.abs(r / KPC - L) < 1e-6
            if m.sum() < 8:
                continue
            lR = np.array([math.log10(C[c]["R500_lens"]) for c in nm[m]])
            lRx = np.array([math.log10(C[c]["R500_xray"]) for c in nm[m]])
            pure[f"{sn}/{L:.0f}kpc"] = dict(
                n=int(m.sum()),
                pearson_excess_vs_log10R500_lens=S.pear(lR, sv[m]),
                spearman_excess_vs_log10R500_lens=S.spear(lR, sv[m]),
                slope_dexcess_dlog10R500_lens=S.ols_slope(lR, sv[m]),
                pearson_excess_vs_log10R500_xray=S.pear(lRx, sv[m]),
                slope_dexcess_dlog10R500_xray=S.ols_slope(lRx, sv[m]))
            d = pure[f"{sn}/{L:.0f}kpc"]
            print(f"  {sn:3s} r={L:5.0f} kpc  n={m.sum():2d}  "
                  f"corr(excess, log R500_lens) = "
                  f"{d['pearson_excess_vs_log10R500_lens']:+.4f}  "
                  f"slope {d['slope_dexcess_dlog10R500_lens']:+.3f}   |   "
                  f"X-ray R500: corr {d['pearson_excess_vs_log10R500_xray']:+.4f}")
    OUT["pure_tautology_contrast"] = pure
    print("  The tautology predicts a POSITIVE corr(excess, log R500) at fixed r")
    print("  (a mass over-estimate raises both), i.e. a NEGATIVE slope against")
    print("  log(r/R500).  An independent X-ray R500 has no such channel.")

    # ------------------------------- between-cluster regression, all levels
    print("\n=== 2.5c  BETWEEN-CLUSTER regression of the per-cluster mean ===")
    bet = {}
    for sub, mask in subsets(T).items():
        g = nm[mask]
        names = sorted(set(g.tolist()))
        for sn, sv in stats.items():
            mu = np.array([sv[mask][g == c].mean() for c in names])
            for key, lab in (("R500_lens", "lensing"), ("R500_xray", "X-ray"),
                             ("R500_TX", "T_X proxy")):
                v = np.array([norm[c][key] for c in names])
                ok = np.isfinite(v)
                bet[f"{sub}/{sn}/{key}"] = dict(
                    n=int(ok.sum()), pearson=S.pear(np.log10(v[ok]), mu[ok]),
                    spearman=S.spear(v[ok], mu[ok]),
                    slope=S.ols_slope(np.log10(v[ok]), mu[ok]))
                d = bet[f"{sub}/{sn}/{key}"]
                print(f"  [{sub}] {sn:3s} vs log10 {key:<11} ({lab:9s}) "
                      f"n={d['n']:2d}  r={d['pearson']:+.4f}  "
                      f"slope={d['slope']:+.3f}")
    OUT["between_cluster_regression"] = bet

    # ---------------- is each normaliser contaminated by the baryon amplitude?
    # The excess carries g_bar in its denominator (y subtracts log nu_RAR(g_bar/a0);
    # a0_eff = g_obs^2/g_bar), so a normaliser whose BETWEEN-cluster variation
    # tracks the baryon amplitude puts the same quantity on both axes.
    print("\n=== 2.5d  IS EACH NORMALISER CONTAMINATED BY g_bar? ===")
    m200 = np.abs(r / KPC - 200) < 1e-6
    n200 = [c for c in sorted(C) if (m200 & (nm == c)).sum() == 1]
    lgb200 = np.array([float(np.log10(T["gb"][m200 & (nm == c)][0])) for c in n200])
    con = {}
    for k in ("R500_lens", "R500_xray", "R500_TX", "Rb_gas", "Rb_M", "Rb_g"):
        v = np.array([norm[c][k] for c in n200])
        ok = np.isfinite(v)
        con[k] = dict(n=int(ok.sum()),
                      pearson_logRnorm_vs_log_gbar_at_200kpc=S.pear(
                          np.log10(v[ok]), lgb200[ok]))
        print(f"  {k:<12} corr(log R_norm, log g_bar at 200 kpc) = "
              f"{con[k]['pearson_logRnorm_vs_log_gbar_at_200kpc']:+.4f}")
    OUT["normaliser_gbar_contamination"] = con
    print("  A baryon-only normaliser is NOT automatically a clean control.")

    json.dump(OUT, open("structure_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote structure_results.json")


if __name__ == "__main__":
    main()
