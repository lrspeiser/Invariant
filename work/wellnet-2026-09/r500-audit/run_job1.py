"""JOB 1 driver -- the synthetic null, its sensitivity, and responsiveness."""
from __future__ import annotations
import json
import math
import time

import numpy as np

import ingest as I
import nullsim as N

KPC, G = I.KPC, I.G
OUT = {}


def observed():
    cl = I.load_all(verbose=False)
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    nm = np.array([p["name"] for p in pts])
    y = I.rar_residual(gb, go)
    return dict(cl=cl, r=r, gb=gb, go=go, R5=R5, name=nm, y=y)


def pct_of(val, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return float(100.0 * np.mean(null <= val))


def summarise(arr, obs=None):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    d = dict(n_real=int(len(a)), mean=float(a.mean()), sd=float(a.std(ddof=1)),
             median=float(np.median(a)),
             p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
             min=float(a.min()), max=float(a.max()))
    if obs is not None:
        d["observed"] = float(obs)
        d["percentile_of_observed"] = pct_of(obs, a)
        d["z_vs_null"] = float((obs - a.mean()) / a.std(ddof=1)) if a.std(ddof=1) > 0 else float("nan")
    return d


def main():
    t0 = time.time()
    O = observed()
    TS = [N.Template(c) for c in O["cl"]]
    r, y, R5, nm = O["r"], O["y"], O["R5"], O["name"]

    # ---------------------------------------------------- the observed statistics
    obs = dict(
        n_points=int(len(r)), n_clusters=int(len(set(nm))),
        S1_hse=N.spear(r / R5, y),
        S1_phys=N.spear(r, y),
        S1_hse_pearson=N.pear(r / R5, y),
        S2_hse=N.collapse_rms(r / R5, y),
        S2_phys=N.collapse_rms(r / (1000 * KPC), y),
        S3_hse=N.slope_beyond(r / R5, y),
        S3_phys_per_dex=float(np.polyfit(np.log10(r[r / R5 > 0.25] / (1000 * KPC)),
                                         y[r / R5 > 0.25], 1)[0]),
    )
    per = {}
    for c in sorted(set(nm)):
        m = nm == c
        per[c] = dict(n=int(m.sum()), spearman=N.spear(r[m], y[m]),
                      mean_y=float(y[m].mean()),
                      R500_kpc=float(R5[m][0] / KPC),
                      r_min_kpc=float(r[m].min() / KPC), r_max_kpc=float(r[m].max() / KPC))
    obs["per_cluster"] = per
    obs["within_cluster_spearman_mean"] = float(np.mean([v["spearman"] for v in per.values()]))
    # cross-cluster: does the per-cluster mean excess track ln R500?  that is the
    # ONLY channel by which the shared mass can act
    mu = np.array([per[c]["mean_y"] for c in sorted(per)])
    lR = np.log(np.array([per[c]["R500_kpc"] for c in sorted(per)]))
    obs["cross_cluster_corr_meanY_lnR500"] = N.pear(lR, mu)
    obs["cross_cluster_spearman_meanY_lnR500"] = N.spear(lR, mu)
    OUT["observed"] = obs
    print(f"OBSERVED  S1_hse={obs['S1_hse']:+.4f}  S1_phys={obs['S1_phys']:+.4f}  "
          f"S2_hse={obs['S2_hse']:.4f}  S3_hse={obs['S3_hse']:+.4f}")
    print(f"          within-cluster mean Spearman {obs['within_cluster_spearman_mean']:+.4f}")
    print(f"          cross-cluster corr(mean y, ln R500) = "
          f"{obs['cross_cluster_corr_meanY_lnR500']:+.4f}")

    # --------------------------------------------- the R500-scrambling null (real data)
    print("\n[A] R500-scrambling null on the REAL data")
    rng = np.random.default_rng(11)
    names = sorted(set(nm))
    R5map = {c: per[c]["R500_kpc"] * KPC for c in names}
    idx = {c: np.where(nm == c)[0] for c in names}
    NP = 20000
    s1 = np.empty(NP); s2 = np.empty(NP)
    Rv = np.array([R5map[c] for c in names])
    for i in range(NP):
        p = rng.permutation(len(names))
        t = np.empty(len(r))
        for j, c in enumerate(names):
            t[idx[c]] = Rv[p[j]]
        s1[i] = N.spear(r / t, y)
        s2[i] = N.collapse_rms(r / t, y)
    OUT["scramble_null"] = dict(
        description="permute the published R500 across the 12 clusters; the "
                    "within-cluster rank structure is untouched, so this isolates "
                    "exactly what R500 contributes",
        S1=summarise(s1, obs["S1_hse"]), S2=summarise(s2, obs["S2_hse"]),
        n_permutations=NP)
    print(f"    S1 null: mean {s1.mean():+.4f} sd {s1.std(ddof=1):.4f}; "
          f"observed {obs['S1_hse']:+.4f} at percentile "
          f"{pct_of(obs['S1_hse'], s1):.1f}")
    print(f"    S2 null: mean {s2.mean():.4f} sd {s2.std(ddof=1):.4f}; "
          f"observed {obs['S2_hse']:.4f} at percentile "
          f"{pct_of(obs['S2_hse'], s2):.1f}")

    # --------------------------------------------- forward synthetic null
    print("\n[B] forward synthetic null: NO true dependence of the excess on any radius")
    NR = 400
    res = N.run(TS, n_real=NR, seed=101)
    OUT["forward_null_primary"] = dict(
        description="y_true = per-cluster constant; full X-COP publication+analysis "
                    "chain with realistic n_e and T errors and R500 re-inferred from "
                    "the noisy hydrostatic profile",
        config={k: v for k, v in N.DEFAULT_CFG.items()},
        n_real=NR,
        S1_hse=summarise(res["S1_hse"], obs["S1_hse"]),
        S1_phys=summarise(res["S1_phys"], obs["S1_phys"]),
        S1_scr=summarise(res["S1_scr"]),
        S2_hse=summarise(res["S2_hse"], obs["S2_hse"]),
        S3_hse=summarise(res["S3_hse"], obs["S3_hse"]),
        R500_recovery=None,
    )
    a = res["S1_hse"]
    print(f"    S1_hse null: mean {a.mean():+.4f} sd {a.std(ddof=1):.4f} "
          f"[p05 {np.percentile(a,5):+.4f}, p95 {np.percentile(a,95):+.4f}]")
    print(f"    observed {obs['S1_hse']:+.4f}  ->  percentile {pct_of(obs['S1_hse'],a):.2f}, "
          f"z = {(obs['S1_hse']-a.mean())/a.std(ddof=1):+.2f}")

    # noiseless decomposition of the null: how much is pipeline bias vs noise?
    cfg0 = dict(N.DEFAULT_CFG, ne_scale=0.0, T_scale=0.0, T_calib=0.0, rho_corr=0.0)
    rr = N.one_realisation(TS, np.random.default_rng(0), cfg0)
    st = N.stats_of(rr)
    OUT["forward_null_noiseless"] = dict(
        description="same null, ALL measurement noise switched off -- what remains "
                    "is the deterministic bias of the analysis pipeline itself",
        S1_hse=st["S1_hse"], S1_phys=st["S1_phys"], S2_hse=st["S2_hse"],
        S3_hse=st["S3_hse"],
        y_minus_ytrue_median=float(np.median(rr["y"] - rr["y_true"])),
        y_minus_ytrue_rms=float(np.std(rr["y"] - rr["y_true"])))
    print(f"    noiseless (pure pipeline bias): S1_hse = {st['S1_hse']:+.4f}")

    # --------------------------------------------- sensitivity
    print("\n[C] null sensitivity")
    sens = {}
    grid = [
        ("white noise only", dict(rho_corr=0.0)),
        ("fully correlated bins", dict(rho_corr=0.95, ell=4.0)),
        ("2x quoted errors", dict(ne_scale=2.0, T_scale=2.0)),
        ("4x quoted errors", dict(ne_scale=4.0, T_scale=4.0)),
        ("R500 smeared 10% extra", dict(R500_extra_frac=0.10)),
        ("R500 smeared 25% extra", dict(R500_extra_frac=0.25)),
        ("R500 from raw pointwise crossing", dict(R500_mode="raw")),
        ("T calibration 10% per cluster", dict(T_calib=0.10)),
    ]
    for nm_, cfg in grid:
        rz = N.run(TS, n_real=150, seed=202, cfg=cfg)
        sens[nm_] = dict(config=cfg,
                         S1_hse=summarise(rz["S1_hse"], obs["S1_hse"]),
                         S2_hse=summarise(rz["S2_hse"], obs["S2_hse"]))
        b = rz["S1_hse"]
        print(f"    {nm_:<34} S1 null {b.mean():+.4f} +- {b.std(ddof=1):.4f}   "
              f"pct of observed {pct_of(obs['S1_hse'], b):.2f}")
    OUT["null_sensitivity"] = sens

    # --------------------------------------------- absolute-radius null
    print("\n[D] absolute-radius null (excess organised by r in kpc, not by r/R500)")
    s_abs = obs["S3_phys_per_dex"]
    rz = N.run(TS, n_real=200, seed=303, s_abs=s_abs)
    OUT["absolute_radius_null"] = dict(
        description="y_true depends on PHYSICAL radius with the observed pooled "
                    "slope; there is no scaled-radius structure in the truth",
        injected_slope_per_dex=float(s_abs),
        S1_hse=summarise(rz["S1_hse"], obs["S1_hse"]),
        S1_phys=summarise(rz["S1_phys"], obs["S1_phys"]),
        S2_hse=summarise(rz["S2_hse"], obs["S2_hse"]),
        S2_phys=summarise(rz["S2_phys"], obs["S2_phys"]),
        S1_hse_minus_S1_phys=summarise(rz["S1_hse"] - rz["S1_phys"],
                                       obs["S1_hse"] - obs["S1_phys"]),
    )
    print(f"    injected slope {s_abs:+.3f} dex/dex on ABSOLUTE radius")
    print(f"    S1_hse {rz['S1_hse'].mean():+.4f} +- {rz['S1_hse'].std(ddof=1):.4f}, "
          f"S1_phys {rz['S1_phys'].mean():+.4f} +- {rz['S1_phys'].std(ddof=1):.4f}")
    print(f"    S1_hse - S1_phys = {(rz['S1_hse']-rz['S1_phys']).mean():+.5f} "
          f"+- {(rz['S1_hse']-rz['S1_phys']).std(ddof=1):.5f}  "
          f"(observed {obs['S1_hse']-obs['S1_phys']:+.5f})")

    # --------------------------------------------- responsiveness
    print("\n[E] responsiveness: d(corr_measured)/d(corr_injected)")
    inj = []
    for s in (0.0, -0.25, -0.5, -0.75, -1.0, -1.354, -2.0):
        rz = N.run(TS, n_real=120, seed=404, s_scaled=s)
        inj.append(dict(s_scaled=s,
                        corr_injected=float(np.mean(rz["S1_truth_injected"])),
                        corr_injected_sd=float(np.std(rz["S1_truth_injected"], ddof=1)),
                        corr_measured=float(np.mean(rz["S1_hse"])),
                        corr_measured_sd=float(np.std(rz["S1_hse"], ddof=1)),
                        slope_measured=float(np.mean(rz["S3_hse"])),
                        slope_measured_sd=float(np.std(rz["S3_hse"], ddof=1))))
        print(f"    s={s:+.3f}  corr_inj {inj[-1]['corr_injected']:+.4f}  "
              f"corr_meas {inj[-1]['corr_measured']:+.4f} +- "
              f"{inj[-1]['corr_measured_sd']:.4f}  "
              f"slope_meas {inj[-1]['slope_measured']:+.4f}")
    ci = np.array([d["corr_injected"] for d in inj])
    cm = np.array([d["corr_measured"] for d in inj])
    si = np.array([d["s_scaled"] for d in inj])
    sm = np.array([d["slope_measured"] for d in inj])
    # local responsiveness near the null, and over the full range
    k = np.argsort(ci)
    resp_all = float(np.polyfit(ci, cm, 1)[0])
    near = np.abs(si) <= 0.75
    resp_near = float(np.polyfit(ci[near], cm[near], 1)[0])
    slope_resp = float(np.polyfit(si, sm, 1)[0])
    OUT["responsiveness"] = dict(
        points=inj,
        d_corr_measured_d_corr_injected_full=resp_all,
        d_corr_measured_d_corr_injected_near_null=resp_near,
        d_slope_measured_d_slope_injected=slope_resp,
        note="corr_injected is the Spearman of the NOISELESS truth against "
             "r/R500_true at the same points; corr_measured is the pipeline output "
             "against r/R500_inferred")
    print(f"    d(corr_meas)/d(corr_inj): {resp_all:.3f} over the full range, "
          f"{resp_near:.3f} near the null")
    print(f"    d(slope_meas)/d(slope_inj): {slope_resp:.3f}")

    OUT["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(OUT, open("job1_results.json", "w", encoding="utf-8"), indent=1)
    print(f"\nwrote job1_results.json  ({OUT['runtime_sec']} s)")


if __name__ == "__main__":
    main()
