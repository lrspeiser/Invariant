"""
JOB 2D -- (a) the well-posed self-similarity test, and (b) the clamped-temperature
bug in the pipeline that produced -0.788.

(a)  The identity in identity.py kills the comparison only when each cluster is
     allowed its own level.  The self-similarity CLAIM is exactly that the levels
     are not free.  So we build two GLOBAL-PARAMETER hypotheses -- no per-object
     parameter anywhere, as the standing constraints require -- that genuinely
     differ:

        H_scaled : y(r) = A + s*log10(r / R500_i)   the same excess at the same
                   r/R500 for every cluster; R500_i solved from the overdensity
                   condition given A
        H_phys   : y(r) = B + s*log10(r / 1 Mpc)    the same excess at the same
                   PHYSICAL radius; R500_i is then whatever the overdensity
                   condition gives

     A and B are single global constants tuned so the simulated R500 population
     matches the real one.  Under H_scaled the clusters collapse on r/R500;
     under H_phys they collapse on r.  The separation between them IS the power
     of the whole lane-12 argument.

(b)  15.8% of the 588 X-COP points sit beyond the outermost MEASURED temperature
     bin.  np.interp clamps there, so dln kT/dln r is forced to ~0 and the
     hydrostatic g is distorted -- at exactly the large radii that carry the
     claimed trend.  Quantified here.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import nullsim as N

KPC, G, MSUN, MU, MP = I.KPC, I.G, I.MSUN, I.MU, I.MP
OUT = {}


# ------------------------------------------------------------------ (a)
def truth_h(T, mode, level, s):
    """global-parameter truth; returns y_true, go, kT, R500_true."""
    r = T.r
    gb = T.gbar(T.ne)
    Mrar = gb * I.nu_rar(gb / N.A0) * r ** 2 / G
    A = (4 / 3) * np.pi * 500 * T.rhoc
    if mode == "scaled":
        # y(R500) = level for every cluster  ->  M(R500) = Mrar(R500)*10^level
        f = np.log(Mrar) + level * math.log(10) - np.log(A * r ** 3)
        R5 = N._cross(r, np.exp(np.log(Mrar) + level * math.log(10)), A)
        if not np.isfinite(R5):
            R5 = T.R500_pub
        y = level + s * np.log10(r / R5)
    else:
        y = level + s * np.log10(r / (1000 * KPC))
        R5 = N._cross(r, Mrar * 10 ** y, A)
        if not np.isfinite(R5):
            R5 = T.R500_pub
    M = Mrar * 10 ** y
    go = G * M / r ** 2
    integ = MU * MP * T.ne * go
    cum = np.concatenate([[0.], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
    Pb = np.interp(T.r_bnd, r, T.ne * T.kT_obs)
    cb = np.interp(T.r_bnd, r, cum)
    kT = (Pb + (cb - cum)) / T.ne
    kT = np.maximum(kT, 1e-4 * np.nanmax(kT))
    return dict(y_true=y, go=go, kT=kT, R500_true=R5)


def realise_h(TS, rng, mode, level, s, shape_amp=0.0):
    per = []
    for T in TS:
        tr = truth_h(T, mode, level, s)
        if shape_amp > 0:
            u = np.log10(T.r / tr["R500_true"])
            u = (u - u.mean()) / max(u.std(), 1e-9)
            q = u ** 2
            q = (q - q.mean()) / max(q.std(), 1e-9)
            c1, c2 = rng.standard_normal(2)
            bump = shape_amp * (c1 * u + c2 * q) / math.sqrt(2.0)
            tr["kT"] = tr["kT"] * 10 ** bump      # perturb the OBSERVABLE, not the truth
        ne, kT_c = N.observe(T, tr, rng, N.DEFAULT_CFG)
        R5 = N.infer_R500(T, ne, kT_c, mode="fit")
        if not np.isfinite(R5):
            R5 = tr["R500_true"]
        go, gb = N.analyse(T, ne, kT_c, R5)
        m = ((T.r > N.R_MIN) & (T.r < N.R_MAX) & (go > 0) & (gb > 0)
             & np.isfinite(go) & np.isfinite(gb))
        if m.sum() < 5:
            continue
        y = I.rar_residual(gb[m], go[m])
        g = np.isfinite(y)
        per.append((T.name, T.r[m][g], y[g], R5, tr["R500_true"]))
    if not per:
        return None
    r = np.concatenate([p[1] for p in per])
    y = np.concatenate([p[2] for p in per])
    t = np.concatenate([np.full(len(p[1]), p[3]) for p in per])
    R5t = np.array([p[4] for p in per])
    return dict(r=r, y=y, R500_obs=t, R500_true=R5t,
                names=np.concatenate([[p[0]] * len(p[1]) for p in per]))


def stats_h(res):
    r, y, t = res["r"], res["y"], res["R500_obs"]
    s2t = N.collapse_rms(r / t, y)
    s2p = N.collapse_rms(r / (1000 * KPC), y)
    return dict(S1=N.spear(r / t, y), S2=s2t, S2_phys=s2p, Drel=s2t / s2p - 1.0,
                S3=N.slope_beyond(r / t, y),
                medR500=float(np.median(res["R500_true"]) / KPC))


def tune_level(TS, mode, s, target_med_kpc, lo=-0.2, hi=1.2):
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        med = np.median([truth_h(T, mode, mid, s)["R500_true"] for T in TS]) / KPC
        if med < target_med_kpc:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tune_s(TS, mode, level_fn, target_S1, amp, seed=5):
    lo, hi = 0.0, 2.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        lv = level_fn(mid)
        rng = np.random.default_rng(seed)
        v = np.mean([stats_h(realise_h(TS, rng, mode, lv, -mid, amp))["S1"]
                     for _ in range(25)])
        if v > target_S1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    cl = I.load_all(verbose=False)
    TS = [N.Template(c) for c in cl]
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    nm = np.array([p["name"] for p in pts])
    y = I.rar_residual(gb, go)
    obs = dict(S1=N.spear(r / R5, y), S2=N.collapse_rms(r / R5, y),
               S2_phys=N.collapse_rms(r / (1000 * KPC), y), S3=N.slope_beyond(r / R5, y))
    obs["Drel"] = obs["S2"] / obs["S2_phys"] - 1.0
    med_target = float(np.median([c["R500_hse"] for c in cl]) / KPC)
    print(f"OBSERVED S1={obs['S1']:+.4f} S2={obs['S2']:.4f} Drel={obs['Drel']:+.5f} "
          f"median R500 = {med_target:.0f} kpc")

    print("\n(a) WELL-POSED SELF-SIMILARITY TEST -- two global-parameter hypotheses")
    AMP = 0.09
    hyp = {}
    for mode in ("scaled", "phys"):
        s = tune_s(TS, mode, lambda ss: tune_level(TS, mode, -ss, med_target),
                   obs["S1"], AMP)
        lv = tune_level(TS, mode, -s, med_target)
        rng = np.random.default_rng(2024)
        rows = [stats_h(realise_h(TS, rng, mode, lv, -s, AMP)) for _ in range(300)]
        agg = {k: np.array([q[k] for q in rows], float) for k in rows[0]}
        hyp[mode] = dict(global_level=float(lv), global_slope=float(-s),
                         shape_amp_dex=AMP,
                         **{k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                    p05=float(np.percentile(v, 5)),
                                    p95=float(np.percentile(v, 95)))
                            for k, v in agg.items()},
                         pct_obs_Drel=float(100 * np.mean(agg["Drel"] <= obs["Drel"])),
                         pct_obs_S2=float(100 * np.mean(agg["S2"] <= obs["S2"])))
        lab = "H_scaled (same excess at same r/R500)" if mode == "scaled" \
            else "H_phys  (same excess at same physical r)"
        print(f"  {lab}")
        print(f"     global level {lv:+.4f} dex, global slope {-s:+.4f} dex/dex, "
              f"median R500 {agg['medR500'].mean():.0f} kpc")
        print(f"     S1   {agg['S1'].mean():+.4f} +- {agg['S1'].std(ddof=1):.4f}")
        print(f"     S2   {agg['S2'].mean():.4f} +- {agg['S2'].std(ddof=1):.4f}")
        print(f"     Drel {agg['Drel'].mean():+.5f} +- {agg['Drel'].std(ddof=1):.5f}"
              f"   observed {obs['Drel']:+.5f} -> percentile "
              f"{100*np.mean(agg['Drel']<=obs['Drel']):.1f}")
    a, b = hyp["scaled"], hyp["phys"]
    sep = (b["Drel"]["mean"] - a["Drel"]["mean"]) / math.sqrt(
        0.5 * (a["Drel"]["sd"] ** 2 + b["Drel"]["sd"] ** 2))
    OUT["wellposed_selfsimilarity"] = dict(
        observed=obs, hypotheses=hyp, separation_sigma_Drel=float(sep),
        note="both hypotheses carry exactly two GLOBAL parameters and no "
             "per-object parameter. The separation is the power of the lane-12 "
             "self-similarity claim on this sample.")
    print(f"\n  SEPARATION H_phys - H_scaled on Drel: {sep:+.3f} sigma")
    print(f"  -> with 12 clusters spanning a factor "
          f"{max(c['R500_hse'] for c in cl)/min(c['R500_hse'] for c in cl):.2f} in "
          f"R500, this is the whole discriminating power available.")

    # ------------------------------------------------------------------ (b)
    print("\n(b) THE CLAMPED-TEMPERATURE BUG")
    keep = np.ones(len(r), bool)
    info = []
    k = 0
    for c in cl:
        p = I.build_profile(c)
        m = ((p["r"] > N.R_MIN) & (p["r"] < N.R_MAX) & (p["go"] > 0) & (p["gb"] > 0))
        rt = c["rw_x"] * c["R500_hse"]
        bad = (p["r"][m] > rt.max()) | (p["r"][m] < rt.min())
        keep[k:k + m.sum()] = ~bad
        info.append(dict(cluster=c["name"], n_points=int(m.sum()),
                         n_clamped=int(bad.sum()),
                         T_grid_max_kpc=float(rt.max() / KPC),
                         T_grid_max_over_R500=float(rt.max() / c["R500_hse"]),
                         data_max_kpc=float(p["r"][m].max() / KPC)))
        k += m.sum()
    n_bad = int((~keep).sum())
    S1_all = N.spear(r / R5, y)
    S1_keep = N.spear(r[keep] / R5[keep], y[keep])
    S3_all = N.slope_beyond(r / R5, y)
    S3_keep = N.slope_beyond(r[keep] / R5[keep], y[keep])
    print(f"   {n_bad}/{len(r)} = {100*n_bad/len(r):.1f}% of points lie beyond the "
          f"outermost MEASURED temperature bin")
    print(f"   S1: all points {S1_all:+.4f}   clamped points removed {S1_keep:+.4f}"
          f"   (change {S1_keep-S1_all:+.4f})")
    print(f"   S3: all points {S3_all:+.4f}   clamped points removed {S3_keep:+.4f}"
          f"   (change {S3_keep-S3_all:+.4f})")
    ycl = y[~keep]
    print(f"   mean residual: clamped points {ycl.mean():+.4f} dex, "
          f"clean points {y[keep].mean():+.4f} dex")
    # median outer edge of the T grid, in R500
    ed = np.array([d["T_grid_max_over_R500"] for d in info])
    print(f"   the measured temperature grid ends at r/R500 = {ed.min():.2f}-"
          f"{ed.max():.2f} (median {np.median(ed):.2f}), but the claim is quoted "
          f"out to r/R500 = {np.max(r/R5):.2f}")
    OUT["clamped_temperature_bug"] = dict(
        n_points_total=int(len(r)), n_clamped=n_bad,
        fraction_clamped=float(n_bad / len(r)),
        S1_all=float(S1_all), S1_clamped_removed=float(S1_keep),
        S3_all=float(S3_all), S3_clamped_removed=float(S3_keep),
        mean_residual_clamped=float(ycl.mean()),
        mean_residual_clean=float(y[keep].mean()),
        T_grid_outer_edge_over_R500=dict(min=float(ed.min()), max=float(ed.max()),
                                         median=float(np.median(ed))),
        max_r_over_R500_in_claim=float(np.max(r / R5)),
        per_cluster=info,
        description="invariant_bench._cluster_profile interpolates the published "
                    "T/T500 profile onto the finer n_e grid with np.interp, which "
                    "CLAMPS beyond the last measured temperature bin. There "
                    "dln kT/dln r is forced to zero and kT is held flat, so the "
                    "hydrostatic g is wrong at exactly the outer radii that carry "
                    "the claimed trend. No warning is emitted.")

    json.dump(OUT, open("job2d_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote job2d_results.json")


if __name__ == "__main__":
    main()
