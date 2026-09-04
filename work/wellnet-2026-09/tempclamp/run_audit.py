"""
TEMPCLAMP -- Jobs 1, 3 and 4.

  Job 1  consumer inventory, verified by import graph rather than asserted
  Job 3  impact audit: every recorded number that rests on clamped temperature
  Job 4  is clamping the right default?  bias of clamp / drop / loglinear
         against a synthetic truth in which temperature genuinely falls outward

Writes results.json.  REPORT.md is rendered from that file by report.py; no
number is typed by hand.

METHOD RULES OBSERVED
  * slopes, not correlations (Run AT: an injected slope of -0.25 already drives
    the correlation to -0.92, so correlations cannot size an effect)
  * every test's own false-positive rate is calibrated before it is read
  * row and column counts asserted on every ingest
  * KiDS and the wide binaries are never loaded
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

import tclamp as T
from tclamp import (A0, KPC, MSUN, G, MP, MU, ROOT, MODES, N_XCOP,
                    load_xcop, nu_rar, ols_slope, radial_slope,
                    rar_slope_vs_frac, spearman)

HERE = os.path.dirname(os.path.abspath(__file__))
KEV = 1.602176634e-16
C_LIGHT = 2.99792458e8
KAPPA_C70 = 1.36e5
BAR = "=" * 78
RES: dict = {}


def head(x):
    print("\n" + BAR + "\n" + x + "\n" + BAR)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ===========================================================================
#  JOB 1 -- the consumer inventory, derived not asserted
# ===========================================================================
def job1_inventory():
    head("JOB 1  --  every consumer of _cluster_profile")
    scan_roots = [ROOT + "work", ROOT + "src", ROOT + "tests",
                  ROOT + "scripts", ROOT + "configs", ROOT + "docs"]
    skip = ("work\\private", "work/private", "site-packages", "__pycache__",
            "node_modules")
    direct, indirect, reimpl, jsons = [], [], [], []
    for root in scan_roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            if any(s in dp for s in skip):
                dn[:] = []
                continue
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dp, f)
                try:
                    src = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                rel = os.path.relpath(p, ROOT).replace("\\", "/")
                if rel.endswith("work/gravity-wells-2026-09/invariant_bench.py"):
                    continue
                imports = ("from invariant_bench import" in src
                           or "import invariant_bench" in src)
                calls_prof = "_cluster_profile(" in src
                # a re-implementation: its own np.interp of the X-COP T table
                own_interp = ("RW_X" in src and "T_X" in src
                              and "np.interp" in src)
                if calls_prof:
                    direct.append(rel)
                if imports:
                    uses_xcop = ('d["xcop"]' in src or "_xcop(" in src)
                    indirect.append(dict(path=rel, uses_xcop=bool(uses_xcop),
                                         instantiates="Bench(" in src))
                if own_interp and not imports:
                    reimpl.append(rel)
                elif own_interp and imports:
                    reimpl.append(rel + "  (also imports the bench)")
    # recorded JSON produced by those lanes
    for rel, why in [
        ("work/gravity-wells-2026-09/paper_results.json", "p01_rigorous.py"),
        ("work/gravity-cluster-audit-2026-09/paper_results.json", "p01_rigorous.py (copy)"),
        ("work/gravity-wells-2026-09/xcop_identity.json", "m11_verify_identity.py"),
        ("work/gravity-cluster-audit-2026-09/c70_endogeneity.json", "c70_endogeneity.py"),
        ("work/gravity-cluster-audit-2026-09/c71.json", "c71_amplitude.py"),
        ("work/gravity-cluster-audit-2026-09/power/power_results.json", "power_analysis.py"),
        ("work/wellnet-2026-09/r500-audit/results.json", "r500-audit (Run AT)"),
        ("work/wellnet-2026-09/r500-audit/job1_results.json", "r500-audit (Run AT)"),
        ("work/wellnet-2026-09/r500-audit/job2_results.json", "r500-audit (Run AT)"),
        ("work/wellnet-2026-09/r500-audit/job2d_results.json", "r500-audit (Run AT)"),
        ("work/wellnet-2026-09/r500-audit/job3_results.json", "r500-audit (Run AT)"),
        ("work/wellnet-2026-09/r500-audit/identity_results.json", "r500-audit (Run AT)"),
    ]:
        p = ROOT + rel
        jsons.append(dict(path=rel, producer=why, exists=os.path.exists(p),
                          bytes=(os.path.getsize(p) if os.path.exists(p) else 0)))

    direct = sorted(set(direct))
    indirect = sorted(indirect, key=lambda q: q["path"])
    reimpl = sorted(set(reimpl))
    print(f"   direct callers of _cluster_profile : {len(direct)}")
    for p in direct:
        print(f"      {p}")
    print(f"\n   importers of invariant_bench.Bench : {len(indirect)}")
    for q in indirect:
        tag = "USES xcop" if q["uses_xcop"] else "loads xcop, does not read it"
        print(f"      {q['path']:<64}{tag}")
    print(f"\n   independent re-implementations of the same interp : {len(reimpl)}")
    for p in reimpl:
        print(f"      {p}")
    print(f"\n   recorded JSON downstream : "
          f"{sum(q['exists'] for q in jsons)} of {len(jsons)} present")
    RES["job1"] = dict(direct_callers=direct, importers=indirect,
                       reimplementations=reimpl, recorded_json=jsons,
                       n_direct=len(direct), n_importers=len(indirect),
                       n_reimpl=len(reimpl))
    return RES["job1"]


# ===========================================================================
#  the three loads
# ===========================================================================
def load_all():
    head("Loading X-COP under each extrapolation mode")
    out = {}
    for m in MODES:
        xc, CL, tot = load_xcop(m)
        out[m] = (xc, CL, tot)
        nx = int(sum(c["extrap"].sum() for c in CL))
        print(f"   mode={m:<10} n = {tot:>4}   extrapolated present = {nx:>3}")
    xcC, CLC, totC = out["clamp"]
    assert totC == T.N_POINTS_RECORDED, f"clamp gives {totC}, recorded 588"
    nx = int(sum(c["extrap"].sum() for c in CLC))
    RES["extrapolation"] = dict(
        n_points=totC, n_extrapolated=nx, frac=nx / totC,
        n_clusters=len(CLC),
        per_cluster=[dict(name=c["name"], n=c["n"],
                          n_extrap=int(c["extrap"].sum()),
                          frac=float(c["extrap"].mean()),
                          n_stencil=int(c["stencil"].sum()),
                          r_last_T_over_R500=c["rw_last"],
                          r_first_T_over_R500=c["rw_first"],
                          r_min_over_R500=float(c["frac"].min()),
                          r_max_over_R500=float(c["frac"].max()),
                          inner_extrap=int((c["frac"] < c["rw_first"]).sum()))
                     for c in sorted(CLC, key=lambda q: q["name"])])
    med_last = float(np.median([c["rw_last"] for c in CLC]))
    RES["extrapolation"].update(
        median_last_T_over_R500=med_last,
        min_last_T_over_R500=float(min(c["rw_last"] for c in CLC)),
        max_last_T_over_R500=float(max(c["rw_last"] for c in CLC)),
        max_r_over_R500=float(max(c["frac"].max() for c in CLC)),
        n_inner_extrap=int(sum((c["frac"] < c["rw_first"]).sum() for c in CLC)))
    print(f"\n   {nx}/{totC} = {100*nx/totC:.2f}% outside the measured T range")
    print(f"   last measured T: median r/R500 = {med_last:.3f}, "
          f"range {RES['extrapolation']['min_last_T_over_R500']:.3f} - "
          f"{RES['extrapolation']['max_last_T_over_R500']:.3f}")
    print(f"   points reach r/R500 = {RES['extrapolation']['max_r_over_R500']:.3f}")
    print(f"   INNER extrapolation: {RES['extrapolation']['n_inner_extrap']} points "
          f"(so the clamped set is the OUTERMOST points, exactly)")
    return out


# ===========================================================================
#  the recorded estimators, recomputed under each mode
# ===========================================================================
def excess_median(CL):
    return np.array([float(np.median(c["exc"])) for c in CL])


def kT_median(CL):
    return np.array([float(np.median(c["kT"])) for c in CL])


def pred_pressure_curve(kT, kap=KAPPA_C70):
    return np.sqrt(1.0 + kap * 3 * (np.asarray(kT) * KEV) / (MU * MP * C_LIGHT ** 2))


def c70_stats(CL):
    kT = kT_median(CL)
    ex = excess_median(CL)
    slope, se = ols_slope(np.log(kT), np.log(ex))
    rr, ra = [], []
    for c in CL:
        p = pred_pressure_curve(c["kT"])
        rr.append(spearman(c["exc"], p))
        ra.append(float(np.median(c["exc"] / p)))
    return dict(slope=slope, slope_err=se,
                within_rho_median=float(np.median(rr)),
                ratio_median=float(np.median(ra)),
                rho_kT_excess=spearman(kT, ex),
                n_pos=int(np.sum(np.array(rr) > 0)))


def c71_stats(CL, pairing=None):
    n = len(CL)
    pairing = list(range(n)) if pairing is None else pairing
    within, med = [], []
    for i, c in enumerate(CL):
        o = CL[pairing[i]]
        kt = np.interp(c["frac"], o["frac"], o["kT"])
        ratio = c["exc"] / pred_pressure_curve(kt)
        good = np.isfinite(ratio) & (ratio > 0)
        if good.sum() < 8:
            continue
        within.append(float(np.std(np.log10(ratio[good]))))
        med.append(float(np.median(ratio[good])))
    return float(np.median(within)), float(np.std(np.log10(med)))


def kappa_fit(CL, eex=None, ekT=None):
    """p01's one-parameter fit.  Errors are reproduced from the same recipe."""
    kT = kT_median(CL)
    ex = excess_median(CL)
    if eex is None:
        rng = np.random.default_rng(20260902)
        eex = np.array([float(np.std([np.median(rng.choice(c["exc"], len(c["exc"])))
                                      for _ in range(2000)])) for c in CL])
    if ekT is None:
        ekT = np.array([
            math.hypot(math.hypot(float(np.std(c["tsc"], ddof=1))
                                  / math.sqrt(max(1, len(c["tsc"]))) * c["T500"],
                                  float(np.median(c["etsc"])) * c["T500"]),
                       float(np.median(c["tsc"])) * c["eT500"]) for c in CL])

    def chi2(kap):
        pr = pred_pressure_curve(kT, kap)
        dpdT = (kap * 3 * KEV / (MU * MP * C_LIGHT ** 2)) / (2 * pr)
        sig = np.hypot(eex, dpdT * ekT)
        return float(np.sum(((ex - pr) / sig) ** 2))

    grid = 10 ** np.linspace(3, 7, 4001)
    c2 = np.array([chi2(k) for k in grid])
    i = int(np.argmin(c2))
    ok = c2 <= c2.min() + 1
    return dict(kappa=float(grid[i]), kappa_lo=float(grid[ok][0]),
                kappa_hi=float(grid[ok][-1]), chi2=float(c2.min()),
                eex=eex.tolist(), ekT=ekT.tolist())


def perm_p_spearman(a, b, n=200000, seed=20260902):
    rng = np.random.default_rng(seed)
    r0 = abs(spearman(a, b))
    a = np.asarray(a, float)
    cnt = 0
    for _ in range(n):
        if abs(spearman(rng.permutation(a), b)) >= r0 - 1e-15:
            cnt += 1
    return cnt / n


# ===========================================================================
#  JOB 3 -- the impact audit
# ===========================================================================
def job3_impact(loads):
    head("JOB 3  --  impact of the clamp on every recorded number")
    rec_paper = json.load(open(ROOT + "work/gravity-wells-2026-09/paper_results.json",
                               encoding="utf-8"))
    rec_c70 = json.load(open(ROOT + "work/gravity-cluster-audit-2026-09/"
                             "c70_endogeneity.json", encoding="utf-8"))
    rec_c71 = json.load(open(ROOT + "work/gravity-cluster-audit-2026-09/c71.json",
                             encoding="utf-8"))
    rec_pow = json.load(open(ROOT + "work/gravity-cluster-audit-2026-09/power/"
                             "power_results.json", encoding="utf-8"))

    tab = {}
    for m in MODES:
        xc, CL, tot = loads[m]
        CLs = sorted(CL, key=lambda q: q["name"])
        s_r = radial_slope(CL)
        s_f = rar_slope_vs_frac(CL)
        c70 = c70_stats(CL)
        w, md = c71_stats(CL)
        kap = kappa_fit(CL)
        kT, ex = kT_median(CL), excess_median(CL)
        # residual vs r/R500, pooled -- the AT.1 Spearman
        fr = np.concatenate([c["frac"] for c in CL])
        rs = np.concatenate([np.log10(c["exc"]) for c in CL])
        rphys = np.concatenate([c["r"] for c in CL])
        tab[m] = dict(
            n=tot,
            slope_radius_within=s_r[0], slope_radius_within_se=s_r[1],
            slope_frac_within=s_f,
            spearman_frac_residual=spearman(fr, rs),
            spearman_r_residual=spearman(rphys, rs),
            rho_T=c70["rho_kT_excess"],
            c70_slope=c70["slope"], c70_slope_err=c70["slope_err"],
            c70_within_rho_median=c70["within_rho_median"],
            c70_ratio_median=c70["ratio_median"],
            c70_n_pos=c70["n_pos"],
            c71_true_within=w, c71_true_median=md,
            kappa=kap["kappa"], kappa_lo=kap["kappa_lo"], kappa_hi=kap["kappa_hi"],
            chi2=kap["chi2"],
            per_cluster_excess={c["name"]: float(np.median(c["exc"])) for c in CLs},
            per_cluster_kT={c["name"]: float(np.median(c["kT"])) for c in CLs},
        )
    # permutation p for rho_T under each mode (its FPR is calibrated below)
    for m in MODES:
        xc, CL, tot = loads[m]
        tab[m]["p_T"] = perm_p_spearman(kT_median(CL), excess_median(CL))

    # ------- reproduction check: clamp must equal the recorded values --------
    rep = []

    def chk(name, got, want, tol, note=""):
        d = abs(got - want)
        rep.append(dict(quantity=name, recorded=want, clamp=got, abs_diff=d,
                        reproduced=bool(d <= tol), tol=tol, note=note))
        print(f"   {name:<42}{want:>14.6f}{got:>14.6f}{d:>12.2e}  "
              f"{'OK' if d <= tol else 'MISMATCH'}")

    print(f"   {'recorded quantity':<42}{'recorded':>14}{'clamp':>14}{'|diff|':>12}")
    print("   " + "-" * 84)
    byn = {c["name"]: c for c in rec_paper["clusters"]}
    worst = 0.0
    for nm, v in tab["clamp"]["per_cluster_excess"].items():
        worst = max(worst, abs(v - byn[nm]["exc"]))
    chk("per-cluster excess, worst of 12", worst, 0.0, 1e-12,
        "bit-identical reproduction of paper_results.json")
    chk("rho_T  Spearman(kT, excess)", tab["clamp"]["rho_T"], rec_paper["rho_T"], 1e-12)
    chk("p_T  permutation", tab["clamp"]["p_T"], rec_paper["p_T"], 3e-3,
        "Monte-Carlo, 200k draws, different seed stream")
    chk("kappa best fit", tab["clamp"]["kappa"], rec_paper["kappa"], 1.0,
        "grid identical; bootstrap error recipe re-run")
    chk("c70 d ln excess / d ln kT", tab["clamp"]["c70_slope"], rec_c70["slope"], 1e-9)
    chk("c70 slope error", tab["clamp"]["c70_slope_err"], rec_c70["slope_err"], 1e-9)
    chk("c70 within-cluster rho median", tab["clamp"]["c70_within_rho_median"],
        rec_c70["within_rho_median"], 1e-9)
    chk("c70 normalisation ratio median", tab["clamp"]["c70_ratio_median"],
        rec_c70["ratio_median"], 1e-9)
    chk("c71 within-cluster scatter", tab["clamp"]["c71_true_within"],
        rec_c71["true_within"], 1e-9)
    chk("c71 median-ratio scatter", tab["clamp"]["c71_true_median"],
        rec_c71["true_median"], 1e-9)
    chk("power A (= c70 within rho median)", tab["clamp"]["c70_within_rho_median"],
        rec_pow["reproduction"]["this_run"]["A"], 1e-9)
    chk("power B (= c71 within scatter)", tab["clamp"]["c71_true_within"],
        rec_pow["reproduction"]["this_run"]["B"], 1e-9)
    n_ok = sum(q["reproduced"] for q in rep)
    print("   " + "-" * 84)
    print(f"   {n_ok}/{len(rep)} recorded quantities reproduced under mode='clamp'")

    RES["job3"] = dict(modes=tab, reproduction=rep,
                       n_reproduced=n_ok, n_checked=len(rep),
                       recorded=dict(paper=rec_paper, c70=rec_c70, c71=rec_c71,
                                     power_repro=rec_pow["reproduction"]["this_run"],
                                     power_critical_p=rec_pow["false_positive"]["critical_p"]))
    return tab


# ===========================================================================
#  the placebo, and the FPR of the "did the number move" test
# ===========================================================================
def job3_placebo(loads, nperm=2000, seed=7):
    head("JOB 3b  --  placebo: is the change bigger than dropping ANY 15.8%?")
    rng = np.random.default_rng(seed)
    xc, CL, tot = loads["clamp"]
    s_full = radial_slope(CL)[0]
    s_drop = radial_slope(loads["drop"][1])[0]
    d_obs = s_drop - s_full
    counts = [int(c["extrap"].sum()) for c in CL]

    null = np.empty(nperm)
    for i in range(nperm):
        CLp = []
        for c, k in zip(CL, counts):
            idx = rng.choice(len(c["r"]), len(c["r"]) - k, replace=False)
            idx.sort()
            CLp.append({kk: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1]
                             == (len(c["r"]),) else v) for kk, v in c.items()})
        null[i] = radial_slope(CLp)[0] - s_full
    p = float(np.mean(np.abs(null) >= abs(d_obs)))
    print(f"   slope, all 588 points              {s_full:+.4f}")
    print(f"   slope, clamped points dropped      {s_drop:+.4f}")
    print(f"   change                             {d_obs:+.4f}")
    print(f"   RANDOM drop of the same per-cluster counts:")
    print(f"      null change  {null.mean():+.4f} +- {null.std():.4f}, "
          f"95% [{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]")
    print(f"      p (|change| at least as large)  {p:.4f}")
    # the degenerate placebo, stated because it MATTERS
    outer_same = all(int(c["extrap"].sum()) == 0
                     or bool(np.all(np.argsort(c["r"])[-int(c["extrap"].sum()):]
                                    == np.where(c["extrap"])[0]))
                     for c in CL)
    print(f"\n   Is 'drop the clamped points' the same set as 'drop the outermost "
          f"{sum(counts)} points'?  {outer_same}")
    print("   If True, the drop-vs-clamp difference CANNOT by itself attribute the")
    print("   change to clamping rather than to radial range. That attribution is")
    print("   what the synthetic truth in Job 4 exists to make.")
    RES["job3b"] = dict(slope_full=s_full, slope_drop=s_drop, change=d_obs,
                        null_mean=float(null.mean()), null_sd=float(null.std()),
                        null_lo=float(np.percentile(null, 2.5)),
                        null_hi=float(np.percentile(null, 97.5)),
                        p_vs_random_drop=p, nperm=nperm,
                        clamped_set_is_outermost=bool(outer_same),
                        n_dropped=int(sum(counts)))
    return RES["job3b"]


def job3_fpr(loads, ndraw=400, seed=11):
    """Calibrate the permutation test that produced p_T = 0.037.

    Run AT found an obvious permutation test running at FPR 0.53-0.70 against a
    nominal 0.05.  This one is different -- it permutes 12 independent cluster
    labels -- but it is not exempt from being checked.
    """
    head("JOB 3c  --  false-positive rate of the p_T permutation test")
    rng = np.random.default_rng(seed)
    xc, CL, tot = loads["clamp"]
    kT = kT_median(CL)
    ex = excess_median(CL)
    sd = float(np.std(np.log(ex)))
    hits = 0
    ps = []
    for _ in range(ndraw):
        # null: excess independent of kT, same marginal spread, same n = 12
        e = np.exp(rng.normal(np.mean(np.log(ex)), sd, len(ex)))
        p = perm_p_spearman(kT, e, n=4000, seed=int(rng.integers(1 << 30)))
        ps.append(p)
        hits += p <= 0.05
    fpr = hits / ndraw
    se = math.sqrt(fpr * (1 - fpr) / ndraw)
    print(f"   nominal alpha 0.05, realised FPR = {fpr:.3f} +- {se:.3f} "
          f"({ndraw} draws)")
    print(f"   -> the test is {'correctly sized' if abs(fpr-0.05) < 3*se else 'MIS-SIZED'}")
    print(f"   recorded p_T = 0.0370 is therefore "
          f"{'usable at face value' if abs(fpr-0.05) < 3*se else 'not usable at face value'}, "
          f"but n = 12 and the memory already records it as underpowered.")
    RES["job3c"] = dict(nominal=0.05, realised_fpr=fpr, se=se, ndraw=ndraw,
                        correctly_sized=bool(abs(fpr - 0.05) < 3 * se),
                        median_null_p=float(np.median(ps)))
    return RES["job3c"]


# ===========================================================================
#  JOB 3d -- the honest reach of the X-COP relation
# ===========================================================================
def job3_reach(loads):
    head("JOB 3d  --  how far out does X-COP actually reach?")
    xc, CL, tot = loads["clamp"]
    lasts = np.array([c["rw_last"] for c in CL])
    grid = np.linspace(0.5, 1.6, 111)
    cov = np.array([(lasts >= g).sum() for g in grid])
    npts = np.array([sum(int(((c["frac"] <= g) & (c["frac"] <= c["rw_last"])).sum())
                         for c in CL) for g in grid])
    r_all12 = float(grid[cov == N_XCOP].max())
    r_half = float(grid[cov >= 6].max())
    R200_OVER_R500 = 1.52          # NFW c500 ~ 3.2, the standard conversion
    n_at_R200 = int((lasts >= R200_OVER_R500).sum())
    n_at_cross = int((lasts >= 1.9).sum())
    frac_beyond = float(np.mean(np.concatenate([c["frac"] for c in CL])
                                > np.concatenate([np.full(c["n"], c["rw_last"])
                                                  for c in CL])))
    print(f"   clusters with MEASURED temperature out to r/R500 =")
    for g in (0.7, 0.8, 0.9, 1.0, 1.1, 1.52, 1.9, 2.5):
        print(f"      {g:>4.2f}   {int((lasts>=g).sum()):>2} of {N_XCOP}")
    print(f"\n   all 12 clusters measured out to      r/R500 = {r_all12:.3f}")
    print(f"   at least half measured out to        r/R500 = {r_half:.3f}")
    print(f"   R200 (= {R200_OVER_R500} R500): {n_at_R200} of {N_XCOP} clusters "
          f"have a measured temperature there")
    print(f"   a0 crossing 1.9-2.5 R500 : {n_at_cross} of {N_XCOP}")
    print(f"   fraction of the 588 quoted points beyond measured T: {100*frac_beyond:.2f}%")
    RES["job3d"] = dict(
        last_T_per_cluster={c["name"]: c["rw_last"] for c in CL},
        coverage_grid=[[float(g), int(cc), int(nn)]
                       for g, cc, nn in zip(grid, cov, npts)],
        r_all12=r_all12, r_half=r_half,
        R200_over_R500=R200_OVER_R500,
        n_clusters_measured_at_R200=n_at_R200,
        n_clusters_measured_at_crossing=n_at_cross,
        frac_points_beyond_measured_T=frac_beyond,
        max_r_over_R500=float(max(c["frac"].max() for c in CL)))
    return RES["job3d"]


# ===========================================================================
#  JOB 4 -- synthetic truth, and the bias of each policy
# ===========================================================================
def measured_outer_logslope(c, k=3):
    """This cluster's own measured d ln kT / d ln r over its outer k T bins."""
    rw = c["rw"]
    kw = np.interp(rw, c["r"] * KPC, c["kT"])
    k = max(2, min(k, len(rw)))
    return float(np.polyfit(np.log(rw[-k:]), np.log(kw[-k:]), 1)[0])


def synth_cluster(c, s_true, C0=2.0):
    """Build a TRUE temperature profile for this cluster's real n_e such that
    the RAR excess is exactly  C0 * (r/r_mid)^s_true  --  no noise at all.

    Hydrostatic:  d(n_e kT)/dr = -mu m_p n_e g_obs, so integrating INWARD from
    the outermost bin gives kT(r) = P(r)/n_e(r) for any target g_obs.

    BUG FOUND AND FIXED HERE (first version of this lane): anchoring the
    pressure at the MIDDLE bin let P(r_out) come out NEGATIVE for five of the
    twelve clusters, so kT_true went negative in the outskirts, log(kT) became
    nan, and `reconstruct` then silently dropped those points -- reproducing,
    inside the audit, exactly the class of silent mask this audit exists to
    catch.  Anchoring at the OUTER boundary makes positivity automatic: the
    integrand is positive, so P(r) = P_out + int_r^rout(...) > P_out > 0 for
    every r once P_out > 0.  Asserted below.
    """
    r = c["r"] * KPC
    ne = c["ne"]
    gb = c["gbar"]
    r_mid = float(np.exp(np.mean(np.log(r))))
    exc = C0 * (r / r_mid) ** s_true
    go = exc * nu_rar(gb / A0) * gb
    integ = MU * MP * ne * go
    dP = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
    P = dP[-1] - dP                       # >= 0, zero at the outer bin
    # OUTER boundary condition: continue this cluster's own measured outer
    # log-slope from the last measured T bin to the last density bin.  That is
    # a realistic, falling temperature -- which is the truth Job 4 is asked for.
    s_meas = measured_outer_logslope(c)
    kT_out = float(np.interp(c["rw"][-1], r, c["kT"])
                   * (r[-1] / c["rw"][-1]) ** s_meas)
    assert kT_out > 0, f"{c['name']}: non-positive outer boundary temperature"
    P = P + kT_out * KEV * ne[-1]
    kT_true = P / ne / KEV                # keV
    assert np.all(kT_true > 0), f"{c['name']}: synthetic kT went non-positive"
    assert np.all(np.isfinite(kT_true)), f"{c['name']}: synthetic kT not finite"
    lr = np.log(r)
    dlnT = np.gradient(np.log(kT_true), lr)
    return dict(r=r, ne=ne, gb=gb, go_true=go, kT_true=kT_true,
                exc_true=exc, s_true=s_true, ok=True,
                s_meas_outer=s_meas,
                dlnT_outer=float(np.median(dlnT[r > 0.5 * r.max()])))


def reconstruct(sy, c, mode, rw_grid=None, noise=0.0, rng=None):
    """Run the synthetic truth through the pipeline under one policy.

    Returns (log10 r, log10 excess, n_nonfinite).  Nothing is dropped silently:
    the caller asserts n_nonfinite == 0 except where dropping is the policy.
    """
    r = sy["r"]
    ne = sy["ne"]
    rw = c["rw"] if rw_grid is None else rw_grid          # metres
    kT_c = np.interp(rw, r, sy["kT_true"])
    if noise > 0:
        kT_c = kT_c * np.exp(rng.normal(0.0, noise, len(kT_c)))
    ext = (r < rw.min()) | (r > rw.max())
    kT = np.interp(r, rw, kT_c)
    if mode == "loglinear" and ext.any():
        k = max(2, min(3, len(rw)))
        so = float(np.polyfit(np.log(rw[-k:]), np.log(kT_c[-k:]), 1)[0])
        si = float(np.polyfit(np.log(rw[:k]), np.log(kT_c[:k]), 1)[0])
        kT = kT.copy()
        hi, lo = r > rw.max(), r < rw.min()
        kT[hi] = kT_c[-1] * (r[hi] / rw.max()) ** so
        kT[lo] = kT_c[0] * (r[lo] / rw.min()) ** si
    lr = np.log(r)
    go = -(kT * KEV / (MU * MP)) * (np.gradient(np.log(ne), lr)
                                    + np.gradient(np.log(kT), lr)) / r
    exc = (go / sy["gb"]) / nu_rar(sy["gb"] / A0)
    bad = int((~(np.isfinite(exc) & (exc > 0))).sum())
    keep = np.isfinite(exc) & (exc > 0)
    if mode == "drop":
        keep &= ~ext
    return np.log10(r[keep]), np.log10(exc[keep]), bad


def _attach_raw(CL):
    """n_e on the bench's own radial cut, plus the real coarse T grid."""
    from astropy.io import fits
    import glob as _g
    for c in CL:
        d = os.path.join(T.XR, c["name"])
        fd = _g.glob(os.path.join(d, "*density*.fits"))[0]
        ft = _g.glob(os.path.join(d, "*temperature*.fits"))[0]
        with fits.open(fd) as h:
            da = h[1].data
            R500m = float(h[1].header["R500"]) * KPC
            rr = 0.5 * (da["R_IN"].astype(np.float64)
                        + da["R_OUT"].astype(np.float64)) * KPC
            nee = da["NE"].astype(np.float64) * 1e6
            assert len(rr) == len(nee) == len(da), "density row-count mismatch"
        with fits.open(ft) as h:
            rw = np.asarray(h[1].data["RW_X"], float) * R500m
            assert len(rw) == len(h[1].data), "temperature row-count mismatch"
        m = (rr > 120 * KPC) & (rr < 1650 * KPC)
        sel = np.argsort(rr[m])
        c["ne"] = nee[m][sel]
        c["r_m"] = rr[m][sel]
        c["rw"] = rw
        c["R500_m"] = R500m
        assert len(c["ne"]) == c["n"], (
            f"{c['name']}: n_e length {len(c['ne'])} != bench count {c['n']}")
        assert np.allclose(c["r_m"] / KPC, c["r"], rtol=1e-12), "radius mismatch"
    return CL


def _pooled_S3(pairs, tmin=0.25):
    """Run AT's S3: pooled slope of y on log10(r/R500), points beyond 0.25."""
    t = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    m = t > tmin
    assert m.sum() >= 10, "too few points beyond 0.25 R500"
    return float(np.polyfit(np.log10(t[m]), y[m], 1)[0])


def _within(pairs):
    """within-cluster fixed-effects slope of y on log10 r."""
    x = np.concatenate([np.log10(p[0]) for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    g = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(pairs)])
    idx = np.unique(g)
    A = np.zeros((len(x), len(idx) + 1))
    A[:, 0] = x
    for j, gi in enumerate(idx):
        A[g == gi, j + 1] = 1.0
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


POLICIES = ("clamp", "drop", "loglinear", "full_coverage", "perfect")


def job4_bias(loads, s_grid=(-0.6, -0.4, -0.25, -0.1, 0.0), seed=5):
    head("JOB 4  --  bias of clamp / drop / loglinear against a synthetic truth")
    xc, CL, tot = loads["clamp"]
    CL = _attach_raw(CL)

    # ---- is the synthetic truth realistic? state it, do not assume it -------
    real = []
    for c in CL:
        sy = synth_cluster(c, 0.0)
        real.append(dict(name=c["name"], s_meas_outer=sy["s_meas_outer"],
                         synth_dlnT_outer=sy["dlnT_outer"],
                         kT_mid_synth=float(sy["kT_true"][len(sy["r"]) // 2]),
                         kT_mid_real=float(c["kT"][c["n"] // 2])))
    print(f"   {'cluster':<9}{'measured dlnT/dlnr':>20}{'synthetic dlnT/dlnr':>21}"
          f"{'kT mid synth':>14}{'kT mid real':>13}")
    for q in real:
        print(f"   {q['name']:<9}{q['s_meas_outer']:>20.3f}"
              f"{q['synth_dlnT_outer']:>21.3f}"
              f"{q['kT_mid_synth']:>14.2f}{q['kT_mid_real']:>13.2f}")
    n_fall = sum(q["synth_dlnT_outer"] < 0 for q in real)
    print(f"\n   synthetic temperature falls outward in {n_fall}/{len(real)} "
          f"clusters -- the truth Job 4 was asked for")

    # ---- the bias grid, on BOTH statistics ---------------------------------
    rows = []
    n_bad_total = 0
    for s_true in s_grid:
        row = dict(s_true=float(s_true))
        for stat in ("pooled_S3", "within"):
            vals = {}
            for mode in POLICIES:
                pr = []
                for c in CL:
                    sy = synth_cluster(c, s_true)
                    if mode == "perfect":
                        x, y, bad = reconstruct(sy, c, "clamp", rw_grid=sy["r"])
                    elif mode == "full_coverage":
                        rw = np.exp(np.linspace(np.log(sy["r"].min()),
                                                np.log(sy["r"].max()),
                                                len(c["rw"])))
                        x, y, bad = reconstruct(sy, c, "clamp", rw_grid=rw)
                    else:
                        x, y, bad = reconstruct(sy, c, mode)
                    n_bad_total += bad
                    assert bad == 0, (
                        f"{c['name']} / {mode} / s={s_true}: {bad} non-finite "
                        f"excess values -- the synthetic truth is broken, not "
                        f"the policy")
                    pr.append(((10 ** x) / c["R500_m"], y) if stat == "pooled_S3"
                              else (10 ** x, y))
                vals[mode] = (_pooled_S3(pr) if stat == "pooled_S3" else _within(pr))
            row[stat] = vals
        rows.append(row)
        print(f"\n   truth {s_true:+.3f}")
        for stat in ("pooled_S3", "within"):
            print(f"      {stat:<11}" + "  ".join(
                f"{k}={row[stat][k]:+.4f}" for k in POLICIES))
    assert n_bad_total == 0, "non-finite excess values appeared"

    # ---- bias at a flat truth, response, de-biased truth --------------------
    obs = {}
    for stat in ("pooled_S3", "within"):
        obs[stat] = {}
        for mode in ("clamp", "drop", "loglinear"):
            _, CLm, _ = load_xcop(mode)
            pr = [(((c["r"] * KPC) / (c["R500"] * KPC)), np.log10(c["exc"]))
                  if stat == "pooled_S3" else (c["r"], np.log10(c["exc"]))
                  for c in CLm]
            obs[stat][mode] = (_pooled_S3(pr) if stat == "pooled_S3"
                               else _within(pr))

    st = np.array([r["s_true"] for r in rows])
    flat = [r for r in rows if r["s_true"] == 0.0][0]
    summary = {}
    for stat in ("pooled_S3", "within"):
        summary[stat] = {}
        print(f"\n   {stat}")
        print(f"   {'policy':<16}{'bias at flat truth':>20}{'response':>11}"
              f"{'observed':>11}{'% of observed':>15}{'de-biased truth':>18}")
        print("   " + "-" * 91)
        for k in POLICIES:
            v = np.array([r[stat][k] for r in rows])
            a, b = np.polyfit(st, v, 1)
            o = obs[stat].get(k, obs[stat]["clamp"])
            deb = (o - b) / a if a != 0 else float("nan")
            summary[stat][k] = dict(bias_at_flat_truth=float(flat[stat][k]),
                                    response=float(a), intercept=float(b),
                                    observed=float(o), debiased_truth=float(deb),
                                    pct_of_observed=float(100 * flat[stat][k] / o))
            print(f"   {k:<16}{flat[stat][k]:>+20.4f}{a:>11.4f}{o:>+11.4f}"
                  f"{100*flat[stat][k]/o:>14.1f}%{deb:>+18.4f}")
        print("   " + "-" * 91)
        fb = summary[stat]["clamp"]["bias_at_flat_truth"]
        fc = summary[stat]["full_coverage"]["bias_at_flat_truth"]
        fp = summary[stat]["perfect"]["bias_at_flat_truth"]
        summary[stat]["decomposition"] = dict(
            total_flat_bias=fb, gradient_only=fp, coarse_full_coverage=fc,
            clamp_only=fb - fc,
            frac_gradient=fp / fb if fb else float("nan"),
            frac_coarse=(fc - fp) / fb if fb else float("nan"),
            frac_clamp=(fb - fc) / fb if fb else float("nan"))
        print(f"      flat-truth bias {fb:+.4f} decomposes as:")
        print(f"         np.gradient discretisation alone   {fp:+.4f}  "
              f"({100*fp/fb:>6.1f}%)")
        print(f"         coarse T grid, FULL coverage       {fc-fp:+.4f}  "
              f"({100*(fc-fp)/fb:>6.1f}%)")
        print(f"         the clamp (this bug)               {fb-fc:+.4f}  "
              f"({100*(fb-fc)/fb:>6.1f}%)")

    RES["job4"] = dict(rows=rows, summary=summary, observed=obs,
                       synthetic_realism=real, n_falling=n_fall,
                       s_grid=[float(s) for s in s_grid],
                       statistic_note=(
                           "pooled_S3 is Run AT's statistic (pooled slope beyond "
                           "0.25 R500 against r/R500, no per-cluster level); "
                           "within is the fixed-effects slope this programme "
                           "actually reads. They differ in sign of the clamp bias."))
    return RES["job4"]


# ===========================================================================
def main():
    t0 = time.time()
    RES["meta"] = dict(
        when=time.strftime("%Y-%m-%dT%H:%M:%S"),
        bench_path="work/gravity-wells-2026-09/invariant_bench.py",
        bench_sha256_after=sha(ROOT + "work/gravity-wells-2026-09/invariant_bench.py"),
        bench_sha256_before=sha(os.path.join(HERE, "invariant_bench.py.orig")),
        numpy=np.__version__, python=sys.version.split()[0],
        sealed_never_loaded=["kids", "widebin"])
    job1_inventory()
    loads = load_all()
    job3_impact(loads)
    job3_placebo(loads)
    job3_fpr(loads)
    job3_reach(loads)
    job4_bias(loads)
    RES["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=float)
    print(f"\n   wrote results.json  ({RES['meta']['runtime_s']} s)")


if __name__ == "__main__":
    main()
