"""
Which recorded VERDICTS move, not merely which digits.

Adds to results.json:
  * the derangement p-values c71 quotes, recomputed under each mode
  * c70's shuffled-temperature null fraction, recomputed under each mode
  * a verdict table: the recorded conclusion, the conclusion after dropping the
    clamped points, and whether they differ
  * what can honestly be said at R200 and about the a0 crossing
  * the Run AT null split (from at_null_split.py)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import tclamp as T
from tclamp import KPC, MODES, load_xcop, spearman
import run_audit as R

HERE = os.path.dirname(os.path.abspath(__file__))
KEV = 1.602176634e-16
C_LIGHT = 2.99792458e8
KAPPA = 1.36e5
BAR = "=" * 78


def head(x):
    print("\n" + BAR + "\n" + x + "\n" + BAR)


def c71_pvalues(CL, nperm=3000, seed=99):
    rng = np.random.default_rng(seed)
    tw, tm = R.c71_stats(CL)
    W, M = [], []
    n = len(CL)
    for _ in range(nperm):
        p = rng.permutation(n)
        while np.any(p == np.arange(n)):
            p = rng.permutation(n)
        w, m = R.c71_stats(CL, list(p))
        W.append(w); M.append(m)
    W, M = np.array(W), np.array(M)
    return dict(true_within=tw, true_median=tm,
                p_within=float(np.mean(W <= tw)),
                p_median=float(np.mean(M <= tm)),
                null_within_median=float(np.median(W)),
                null_median_median=float(np.median(M)))


def c70_nullfrac(CL, nperm=2000, seed=4242):
    rng = np.random.default_rng(seed)
    rr = []
    for c in CL:
        p = R.pred_pressure_curve(c["kT"])
        rr.append(spearman(c["exc"], p))
    obs = float(np.median(rr))
    null = []
    n = len(CL)
    for _ in range(nperm):
        perm = rng.permutation(n)
        vals = []
        for i, c in enumerate(CL):
            other = CL[perm[i]]
            kt = np.interp(np.linspace(0, 1, len(c["r"])),
                           np.linspace(0, 1, len(other["kT"])), other["kT"])
            vals.append(spearman(c["exc"], R.pred_pressure_curve(kt)))
        null.append(np.median(vals))
    null = np.array(null)
    return dict(observed=obs, null_frac=float(np.mean(null >= obs)),
                null_median=float(np.median(null)))


def reach_bounds(loads, res):
    """What can honestly be said at R200 and about the a0 crossing."""
    out = {}
    R200 = res["job3d"]["R200_over_R500"]
    for mode in MODES:
        xc, CL, tot = loads[mode]
        # per-cluster: excess at the LAST MEASURED temperature radius, and the
        # within-cluster slope; then the extrapolation to R200, with the
        # extrapolation distance stated in dex
        rows = []
        for c in CL:
            k = c["frac"] <= c["rw_last"]
            if k.sum() < 5:
                continue
            a, b = np.polyfit(np.log10(c["frac"][k]), np.log10(c["exc"][k]), 1)
            e_last = 10 ** (a * math.log10(c["rw_last"]) + b)
            # where the fitted relation reaches excess = 1 (no excess at all):
            # log10 excess = a*log10 t + b = 0  ->  t = 10^(-b/a)
            t_cross = float(10 ** (-b / a)) if a != 0 else float("nan")
            # where g_bar = a0, from the measured baryonic profile alone
            xg = c["gbar"] / 1.2e-10
            t_a0 = float("nan")
            if np.any(xg < 1) and np.any(xg > 1):
                j = int(np.argmax(xg < 1))
                lo, hi = np.log10(c["frac"][j - 1]), np.log10(c["frac"][j])
                f0, f1 = np.log10(xg[j - 1]), np.log10(xg[j])
                t_a0 = float(10 ** (lo + (hi - lo) * (0 - f0) / (f1 - f0)))
            rows.append(dict(name=c["name"], slope=float(a),
                             excess_at_last_measured_T=float(e_last),
                             last_T_over_R500=c["rw_last"],
                             dex_extrapolated_to_R200=float(
                                 math.log10(R200 / c["rw_last"])),
                             excess_at_R200_extrapolated=float(
                                 10 ** (a * math.log10(R200) + b)),
                             t_excess_unity=t_cross,
                             dex_to_excess_unity=float(
                                 math.log10(t_cross / c["rw_last"]))
                             if np.isfinite(t_cross) and t_cross > 0
                             else float("nan"),
                             t_gbar_equals_a0=t_a0,
                             gbar_a0_measured=bool(np.isfinite(t_a0))))
        sl = np.array([q["slope"] for q in rows])
        e200 = np.array([q["excess_at_R200_extrapolated"] for q in rows])
        elast = np.array([q["excess_at_last_measured_T"] for q in rows])
        tc = np.array([q["t_excess_unity"] for q in rows], float)
        ta = np.array([q["t_gbar_equals_a0"] for q in rows], float)
        out[mode] = dict(
            per_cluster=rows,
            slope_median=float(np.median(sl)),
            slope_sd=float(np.std(sl, ddof=1)),
            excess_at_last_measured_T_median=float(np.median(elast)),
            excess_at_R200_median=float(np.median(e200)),
            excess_at_R200_lo=float(np.percentile(e200, 16)),
            excess_at_R200_hi=float(np.percentile(e200, 84)),
            n_excess_at_R200_below_1=int((e200 < 1.0).sum()),
            n_clusters=len(rows),
            median_dex_extrapolated=float(np.median(
                [q["dex_extrapolated_to_R200"] for q in rows])),
            t_excess_unity_median=float(np.nanmedian(tc)),
            t_excess_unity_lo=float(np.nanpercentile(tc, 16)),
            t_excess_unity_hi=float(np.nanpercentile(tc, 84)),
            median_dex_to_excess_unity=float(np.nanmedian(
                [q["dex_to_excess_unity"] for q in rows])),
            n_gbar_a0_measured=int(np.isfinite(ta).sum()),
            t_gbar_a0_median=(float(np.nanmedian(ta))
                              if np.isfinite(ta).any() else float("nan")))
    return out


def main():
    res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
    loads = {m: load_xcop(m) for m in MODES}

    head("c71 derangement p-values, recomputed under each mode")
    c71 = {}
    for m in MODES:
        c71[m] = c71_pvalues(loads[m][1])
        print(f"   {m:<11} within {c71[m]['true_within']:.4f} "
              f"(p {c71[m]['p_within']:.4f})   "
              f"median-ratio {c71[m]['true_median']:.4f} "
              f"(p {c71[m]['p_median']:.4f})")

    head("c70 shuffled-temperature null, recomputed under each mode")
    c70 = {}
    for m in MODES:
        c70[m] = c70_nullfrac(loads[m][1])
        print(f"   {m:<11} within-rho median {c70[m]['observed']:+.4f}   "
              f"null fraction >= observed {c70[m]['null_frac']:.4f}")

    head("Honest reach: R200 and the a0 crossing")
    reach = reach_bounds(loads, res)
    for m in MODES:
        q = reach[m]
        print(f"   {m:<11} within-cluster slope (measured-T points only) "
              f"{q['slope_median']:+.4f} +- {q['slope_sd']:.4f}")
        print(f"   {'':<11} excess at the last MEASURED T radius "
              f"{q['excess_at_last_measured_T_median']:.3f}")
        print(f"   {'':<11} extrapolated to R200 = 1.52 R500 "
              f"{q['excess_at_R200_median']:.3f} "
              f"[{q['excess_at_R200_lo']:.3f}, {q['excess_at_R200_hi']:.3f}]"
              f"  ({q['median_dex_extrapolated']:.3f} dex beyond the data);"
              f" {q['n_excess_at_R200_below_1']}/{q['n_clusters']} below 1.0")
        print(f"   {'':<11} excess reaches 1.0 at r/R500 = "
              f"{q['t_excess_unity_median']:.2f} "
              f"[{q['t_excess_unity_lo']:.2f}, {q['t_excess_unity_hi']:.2f}]"
              f"  ({q['median_dex_to_excess_unity']:.2f} dex beyond measured T)")
        print(f"   {'':<11} g_bar = a0 directly measured in "
              f"{q['n_gbar_a0_measured']}/{q['n_clusters']} clusters")

    # ---------------- the verdict table -------------------------------------
    head("VERDICT TABLE -- does the conclusion change, not just the digit?")
    M = res["job3"]["modes"]
    V = []

    def add(name, recorded, clamp, drop, ll, verdict_recorded, verdict_drop,
            changes, note=""):
        V.append(dict(quantity=name, recorded=recorded, clamp=clamp, drop=drop,
                      loglinear=ll, verdict_recorded=verdict_recorded,
                      verdict_drop=verdict_drop, verdict_changes=changes,
                      note=note))

    rp = res["job3"]["recorded"]["paper"]
    rc70 = res["job3"]["recorded"]["c70"]
    rc71 = res["job3"]["recorded"]["c71"]

    add("rho_T = Spearman(kT, excess), n=12", rp["rho_T"],
        M["clamp"]["rho_T"], M["drop"]["rho_T"], M["loglinear"]["rho_T"],
        f"rho = +0.615, p = {rp['p_T']:.3f} -> H2 confirmed at alpha 0.05",
        f"rho = {M['drop']['rho_T']:+.3f}, p = {M['drop']['p_T']:.3f} "
        f"-> NOT significant",
        True,
        "the single largest verdict change in the audit")
    add("p_T (permutation)", rp["p_T"], M["clamp"]["p_T"], M["drop"]["p_T"],
        M["loglinear"]["p_T"], "p < 0.05", "p > 0.05", True,
        "test is correctly sized (FPR "
        f"{res['job3c']['realised_fpr']:.3f} vs nominal 0.05)")
    d_alg_c = abs(M["clamp"]["c70_slope"] - 1.0) / M["clamp"]["c70_slope_err"]
    d_prs_c = abs(M["clamp"]["c70_slope"] - 0.3918610832730946) / M["clamp"]["c70_slope_err"]
    d_alg_d = abs(M["drop"]["c70_slope"] - 1.0) / M["drop"]["c70_slope_err"]
    d_prs_d = abs(M["drop"]["c70_slope"] - 0.3918610832730946) / M["drop"]["c70_slope_err"]
    add("c70 d ln(excess)/d ln(kT)", rc70["slope"], M["clamp"]["c70_slope"],
        M["drop"]["c70_slope"], M["loglinear"]["c70_slope"],
        f"+0.517 +- 0.205: {d_alg_c:.1f}s from algebraic(+1), "
        f"{d_prs_c:.1f}s from pressure(+0.392)",
        f"{M['drop']['c70_slope']:+.3f} +- {M['drop']['c70_slope_err']:.3f}: "
        f"{d_alg_d:.1f}s from algebraic, {d_prs_d:.1f}s from pressure",
        False,
        "the number moves 43% but the conclusion -- not algebraic, "
        "consistent with pressure -- strengthens rather than flips")
    add("c70 within-cluster rho median (= power A)", rc70["within_rho_median"],
        M["clamp"]["c70_within_rho_median"], M["drop"]["c70_within_rho_median"],
        M["loglinear"]["c70_within_rho_median"],
        f"+0.643, null fraction {rc70['null_frac']:.4f} -> NOT significant",
        f"{c70['drop']['observed']:+.3f}, null fraction "
        f"{c70['drop']['null_frac']:.4f} -> NOT significant",
        False, "was never significant; still is not")
    add("c71 within-cluster scatter (= power B)", rc71["true_within"],
        M["clamp"]["c71_true_within"], M["drop"]["c71_true_within"],
        M["loglinear"]["c71_true_within"],
        f"0.1689, p = {rc71['p_within']:.3f} -> true pairing NOT tighter",
        f"{c71['drop']['true_within']:.4f}, p = {c71['drop']['p_within']:.3f} "
        f"-> true pairing NOT tighter",
        False, "")
    add("c71 median-ratio scatter (= power C)", rc71["true_median"],
        M["clamp"]["c71_true_median"], M["drop"]["c71_true_median"],
        M["loglinear"]["c71_true_median"],
        f"0.0683, p = {rc71['p_median']:.3f} -> borderline, called NOT tighter",
        f"{c71['drop']['true_median']:.4f}, p = {c71['drop']['p_median']:.3f}",
        bool((rc71["p_median"] <= 0.05) != (c71["drop"]["p_median"] <= 0.05)),
        f"drop does not flip it, but LOGLINEAR does: p = "
        f"{c71['loglinear']['p_median']:.4f} < 0.05, so this verdict is decided "
        f"by the extrapolation policy and by nothing else")
    add("kappa (pressure amplitude)", rp["kappa"], M["clamp"]["kappa"],
        M["drop"]["kappa"], M["loglinear"]["kappa"],
        f"1.563e5, 68% [{rp['kappa_lo']:.3e}, {rp['kappa_hi']:.3e}]",
        f"{M['drop']['kappa']:.3e}, 68% [{M['drop']['kappa_lo']:.3e}, "
        f"{M['drop']['kappa_hi']:.3e}]",
        not (rp["kappa_lo"] <= M["drop"]["kappa"] <= rp["kappa_hi"]),
        "the drop-mode value lies OUTSIDE the recorded 68% interval, so the "
        "recorded interval understates the systematic")
    add("within-cluster radial slope", -0.4802794465494304,
        M["clamp"]["slope_radius_within"], M["drop"]["slope_radius_within"],
        M["loglinear"]["slope_radius_within"],
        "-0.480: a cluster-only excess organised by radius",
        f"{M['drop']['slope_radius_within']:+.4f}: same statement, "
        f"{100*abs(M['drop']['slope_radius_within']-M['clamp']['slope_radius_within'])/abs(M['clamp']['slope_radius_within']):.1f}% smaller",
        False, "survives; the trend is not manufactured by the clamp")
    add("Spearman(r/R500, RAR residual)", -0.7884481495545205,
        M["clamp"]["spearman_frac_residual"], M["drop"]["spearman_frac_residual"],
        M["loglinear"]["spearman_frac_residual"],
        "-0.788", f"{M['drop']['spearman_frac_residual']:+.4f}", False,
        "correlations saturate; quote the slope instead (Run AT.6)")

    print(f"   {'quantity':<42}{'clamp':>12}{'drop':>12}{'verdict changes':>18}")
    print("   " + "-" * 84)
    for v in V:
        print(f"   {v['quantity']:<42}{v['clamp']:>12.4g}{v['drop']:>12.4g}"
              f"{('YES' if v['verdict_changes'] else 'no'):>18}")
    print("   " + "-" * 84)
    nch = sum(v["verdict_changes"] for v in V)
    print(f"   {nch} of {len(V)} recorded verdicts change when the clamped "
          f"15.8% is dropped")

    # the bench's own headline score, which every lane that calls b.score sees
    head("The bench's own baseline score, by extrapolation policy")
    import invariant_bench as IB
    IB.Bench._kids = lambda self: None          # SEALED, never loaded
    IB.Bench._widebin = lambda self: None       # SEALED, never loaded
    scores = {}
    for m in MODES:
        b = IB.Bench(verbose=False, temp_extrapolation=m,
                     warn_extrapolation=False)
        assert "kids" not in b.d and "widebin" not in b.d, "SEAL VIOLATION"
        scores[m] = {k: float(v) for k, v in b.score(
            lambda d: 1.0 / (1 - np.exp(-np.sqrt(d.x))), verbose=False).items()}
        scores[m]["n_xcop"] = int(len(b.d["xcop"]))
        print(f"   {m:<11} " + "  ".join(
            f"{k}={scores[m][k]:.4f}" for k in sorted(scores[m]) if k != "n_xcop")
            + f"   (n_xcop={scores[m]['n_xcop']})")
    res["bench_score_by_mode"] = scores

    res["verdicts"] = dict(table=V, n_changed=nch, n_total=len(V))
    res["c71_by_mode"] = c71
    res["c70_by_mode"] = c70
    res["reach"] = reach
    sp = os.path.join(HERE, "at_null_split.json")
    if os.path.exists(sp):
        res["at_null_split"] = json.load(open(sp, encoding="utf-8"))
    json.dump(res, open(os.path.join(HERE, "results.json"), "w",
                        encoding="utf-8"), indent=1, default=float)
    print("\n   results.json updated")


if __name__ == "__main__":
    main()
