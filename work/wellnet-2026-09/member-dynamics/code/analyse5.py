r"""
member-dynamics lane -- validation pass.

  1. INJECTION / MONOTONE-INVARIANCE CHECK.  The programme has been bitten by a
     rank statistic that was bit-identical across three decades of the parameter
     it was supposed to measure.  Here a known offset delta is injected into the
     cluster arm and the estimator is re-run: the recovered slope d(estimate)/
     d(delta) must be 1, and the intercept must reproduce the measurement.
  2. GRADIENT INJECTION.  A potential-depth law predicts an offset that grows
     with |Phi|.  A gradient of known size is injected and recovered, which
     turns the gradient test's null result into a stated sensitivity rather
     than an absence of evidence.
  3. PER-SYSTEM JACKKNIFE.  SAMI has only 8 clusters.  Drop each in turn.
  4. SAMI colour added to the contamination budget, so the SAMI and MaNGA
     budgets contain the same physics.
  5. Systematic totals restricted to terms that are individually significant.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

import analyse as A
import analyse2 as A2
import analyse3 as A3
import analyse4 as A4

LANE, ENV = A.LANE, A.ENV
RNG = np.random.default_rng(31415926)


def injection_constant(d, dcols, deltas=(-0.05, -0.02, 0.0, 0.02, 0.05)):
    out = []
    for dd in deltas:
        dy = (d["cl_Y"] + dd - d["fi_Y"]).to_numpy(float)
        D = np.nan_to_num(d[dcols].to_numpy(float), nan=0.0)
        m = np.isfinite(dy)
        e, _ = A.adjusted_offset(dy[m], D[m])
        out.append(dict(injected=float(dd), recovered=float(e)))
    x = np.array([o["injected"] for o in out])
    y = np.array([o["recovered"] for o in out])
    b = np.polyfit(x, y, 1)
    return dict(points=out, slope=float(b[0]), intercept=float(b[1]),
                spread=float(y.max() - y.min()),
                verdict="PASS" if abs(b[0] - 1) < 1e-6 else "FAIL")


def injection_gradient(d, dcols, ks=(0.0, 0.01, 0.02, 0.05)):
    """Inject Delta_i = k * (log Phi_i - median log Phi) and recover the slope."""
    phi = 2 * np.log10(d["depth_sigma_v"].to_numpy(float))
    phi = phi - np.nanmedian(phi)
    sysid = d["sysid"].to_numpy()
    sys_u = np.unique(sysid)
    idx_by_sys = {s: np.where(sysid == s)[0] for s in sys_u}
    D = np.nan_to_num(d[dcols].to_numpy(float), nan=0.0)
    out = []
    for k in ks:
        dy = (d["cl_Y"] + k * np.nan_to_num(phi) - d["fi_Y"]).to_numpy(float)
        m = np.isfinite(dy) & np.isfinite(phi)
        X = np.column_stack([np.ones(m.sum()), D[m]])
        r = dy[m] - X @ A.ols(X, dy[m])
        xx = phi[m]
        Aa = np.column_stack([np.ones(len(xx)), xx - xx.mean()])
        sl = float(A.ols(Aa, r)[1])
        boots = []
        for _ in range(1500):
            pick = RNG.integers(0, len(sys_u), len(sys_u))
            idx = np.concatenate([idx_by_sys[sys_u[p]] for p in pick])
            idx = idx[m[idx]]
            if len(idx) < 15:
                continue
            xb = phi[idx]
            rb = r[np.searchsorted(np.where(m)[0], idx)] if False else None
            # recompute residual directly on the resample to stay honest
            dyb = dy[idx]
            Xb = np.column_stack([np.ones(len(idx)), D[idx]])
            try:
                rb = dyb - Xb @ A.ols(Xb, dyb)
                Ab = np.column_stack([np.ones(len(xb)), xb - xb.mean()])
                boots.append(A.ols(Ab, rb)[1])
            except np.linalg.LinAlgError:
                pass
        out.append(dict(injected_slope=float(k), recovered_slope=sl,
                        slope_sd=float(np.std(boots, ddof=1)) if len(boots) > 20 else np.nan))
    sd = np.nanmedian([o["slope_sd"] for o in out])
    return dict(points=out, slope_sd=float(sd),
                min_detectable_gradient_3sigma=float(3 * sd),
                phi_range_dex=float(np.nanpercentile(phi, 90) - np.nanpercentile(phi, 10)),
                min_detectable_end_to_end_offset=float(
                    3 * sd * (np.nanpercentile(phi, 90) - np.nanpercentile(phi, 10))))


def jackknife_systems(d, dcols):
    dy = (d["cl_Y"] - d["fi_Y"]).to_numpy(float)
    D = np.nan_to_num(d[dcols].to_numpy(float), nan=0.0)
    m = np.isfinite(dy)
    dy, D, sysid = dy[m], D[m], d["sysid"].to_numpy()[m]
    full, _ = A.adjusted_offset(dy, D)
    out = []
    for s in np.unique(sysid):
        keep = sysid != s
        if keep.sum() < 12:
            continue
        e, _ = A.adjusted_offset(dy[keep], D[keep])
        out.append(dict(dropped=str(s), n_dropped=int((~keep).sum()),
                        offset_without=float(e), shift=float(e - full)))
    return dict(full=float(full), leave_one_out=out,
                max_abs_shift=float(max(abs(o["shift"]) for o in out)) if out else np.nan)


def sami_budget_with_colour(d):
    """SAMI budget including the g-i colour term, so it prices the same
    physics MaNGA's does."""
    yf = d["fi_Y"].to_numpy(float)
    base = ["fi_logMstar", "fi_logRd", "fi_incl_deg"]
    B = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in base])
    pairs = {"g_minus_i_colour": ("cl_g_i", "fi_g_i"),
             "mu_within_1re": ("cl_mu_within_1re", "fi_mu_within_1re"),
             "kinemetry_k5_k1": ("cl_k51", "fi_k51"),
             "gas_star_PA_misalignment_deg": ("cl_misalign_deg", "fi_misalign_deg"),
             "aperture_correction_flag": ("cl_aper_flag", "fi_aper_flag")}
    sysid = d["sysid"].to_numpy()
    sys_u = np.unique(sysid)
    idx_by_sys = {s: np.where(sysid == s)[0] for s in sys_u}
    out, tot2, tot2sig = {}, 0.0, 0.0
    for name, (cc, fc) in pairs.items():
        if fc not in d.columns or cc not in d.columns:
            out[name] = dict(status="column_absent")
            continue
        x = d[fc].to_numpy(float)
        xc = d[cc].to_numpy(float)
        mm = np.isfinite(x) & np.isfinite(yf) & np.isfinite(B).all(1)
        if mm.sum() < 25:
            out[name] = dict(status="too_few", n=int(mm.sum()))
            continue
        X = np.column_stack([B[mm], x[mm] - np.nanmean(x[mm])])
        slope = float(A.ols(X, yf[mm])[-1])
        dx = float(np.nanmean(xc) - np.nanmean(x))
        ind = slope * dx
        vals = []
        for _ in range(1500):
            pick = RNG.integers(0, len(sys_u), len(sys_u))
            idx = np.concatenate([idx_by_sys[sys_u[p]] for p in pick])
            m2 = np.isfinite(x[idx]) & np.isfinite(yf[idx]) & np.isfinite(B[idx]).all(1)
            if m2.sum() < 25:
                continue
            ii = idx[m2]
            X2 = np.column_stack([B[ii], x[ii] - np.mean(x[ii])])
            try:
                vals.append(A.ols(X2, yf[ii])[-1] * (np.nanmean(xc[idx]) - np.nanmean(x[idx])))
            except np.linalg.LinAlgError:
                pass
        sd = float(np.std(vals, ddof=1)) if len(vals) > 20 else np.nan
        sig = bool(np.isfinite(sd) and abs(ind) > 2 * sd)
        out[name] = dict(field_sensitivity_dY_dX=slope, mean_cluster_minus_field=dx,
                         induced_offset_logV=ind, induced_offset_logV_sd=sd,
                         significant=sig)
        tot2 += ind ** 2
        if sig:
            tot2sig += ind ** 2
    out["_quadrature_total_logV"] = float(np.sqrt(tot2))
    out["_quadrature_significant_only_logV"] = float(np.sqrt(tot2sig))
    return out


def main():
    dm, kinm = A2.build_manga2()
    ds, kins = A2.build_sami2()
    inv = pd.read_csv(os.path.join(ENV, "raw", "sami", "sami_dr3_master_galaxy_inventory.tsv"),
                      sep="\t", usecols=["CATID", "g_i", "mu_within_1re"])
    ds = ds.merge(inv.add_prefix("cl_"), left_on="cl_CATID", right_on="cl_CATID", how="left")
    ds = ds.merge(inv.add_prefix("fi_"), left_on="fi_CATID", right_on="fi_CATID", how="left")
    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], A.ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], A.ASYM_PERCENTILE))
    dm["cl_logReff"] = np.log10(dm["cl_reff_arcsec"])
    dm["fi_logReff"] = np.log10(dm["fi_reff_arcsec"])

    R = json.load(open(os.path.join(LANE, "member_dynamics.json")))
    R["validation"] = {}

    for survey, df, thr, dcols in (("manga", dm, a_manga, A.DCOLS_MANGA),
                                   ("sami", ds, a_sami, A.DCOLS_SAMI)):
        for tier in sorted(df["tier"].unique()):
            base = df[df["tier"] == tier].copy()
            d = base[A4.cutset(base, survey, thr, "B")]
            if len(d) < 20:
                continue
            key = f"{survey}:{tier}"
            v = dict(
                injection_constant=injection_constant(d, dcols),
                injection_gradient=injection_gradient(d, dcols),
                jackknife=jackknife_systems(d, dcols),
            )
            if survey == "sami":
                v["budget_with_colour"] = sami_budget_with_colour(d)
            else:
                cb = R["results_by_cutset"][key]["B"]["contamination_budget"]
                s2 = sum(x["induced_offset_logV"] ** 2 for n, x in cb.items()
                         if not n.startswith("_") and isinstance(x, dict)
                         and x.get("significant") and not x.get("circular_component_of_observable"))
                v["syst_significant_only_logV"] = float(np.sqrt(s2))
            R["validation"][key] = v

    # recompute the combined systematic with the improved SAMI budget
    sb = R["validation"]["sami:S2_diskbearing"]["budget_with_colour"]
    R["results_by_cutset"]["sami:S2_diskbearing"]["B"]["contamination_budget_with_colour"] = sb
    a = R["results_by_cutset"]["manga:B4_disk_wide"]["B"]
    b = R["results_by_cutset"]["sami:S2_diskbearing"]["B"]
    sa_, sb_ = a["sd_boot"], b["sd_boot"]
    wa, wb = 1 / sa_ ** 2, 1 / sb_ ** 2
    comb = (wa * a["offset_logV"] + wb * b["offset_logV"]) / (wa + wb)
    sd = 1 / np.sqrt(wa + wb)
    bud = float(np.hypot(a["contamination_budget"]["_quadrature_total_logV"] * wa / (wa + wb),
                         sb["_quadrature_total_logV"] * wb / (wa + wb)))
    tot = float(np.hypot(sd, bud))
    c = R["combined_corrected"]
    c["syst_with_sami_colour"] = bud
    c["total_with_sami_colour"] = tot
    c["sigma_from_H1_total_v2"] = float((comb - A.PRED_V) / tot)
    c["sigma_from_H1_total_plus_theory_v2"] = float((comb - A.PRED_V) / np.hypot(tot, A.PRED_V_SD))
    c["sigma_from_H2_total_v2"] = float(comb / tot)
    c["power_H1_total_v2"] = A2.power(A.PRED_V, tot)
    c["mdd_3sigma_logV_total_v2"] = float(3 * tot)
    c["mdd_3sigma_logg_total_v2"] = float(6 * tot)

    with open(os.path.join(LANE, "member_dynamics.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)

    for k in ["manga:B4_disk_wide", "sami:S2_diskbearing"]:
        v = R["validation"][k]
        ic, ig, jk = v["injection_constant"], v["injection_gradient"], v["jackknife"]
        print(f"{k}:")
        print(f"   injection slope {ic['slope']:.6f} ({ic['verdict']}), spread over the "
              f"injected range = {ic['spread']:.4f} dex")
        print(f"   gradient sensitivity: slope sd {ig['slope_sd']:.4f} per dex of log|Phi|, "
              f"3-sigma minimum detectable end-to-end offset "
              f"{ig['min_detectable_end_to_end_offset']:.4f} dex over a "
              f"{ig['phi_range_dex']:.2f} dex |Phi| range")
        print(f"   jackknife: full {jk['full']:+.4f}, max leave-one-system-out shift "
              f"{jk['max_abs_shift']:.4f}")
        if "budget_with_colour" in v:
            print(f"   SAMI budget incl. colour: {v['budget_with_colour']['_quadrature_total_logV']:.4f} "
                  f"(significant terms only {v['budget_with_colour']['_quadrature_significant_only_logV']:.4f})")
            for n, x in v["budget_with_colour"].items():
                if n.startswith("_") or "induced_offset_logV" not in x:
                    continue
                print(f"      {n:30s} {x['induced_offset_logV']:+.5f} +- {x['induced_offset_logV_sd']:.5f} sig={x['significant']}")
        else:
            print(f"   MaNGA syst, significant terms only: {v['syst_significant_only_logV']:.4f}")
    print(f"\nfinal combined: D = {comb:+.4f} +- {sd:.4f}(stat) +- {bud:.4f}(syst); "
          f"total {tot:.4f}; H1 {c['sigma_from_H1_total_v2']:+.2f} sigma "
          f"({c['sigma_from_H1_total_plus_theory_v2']:+.2f} incl theory), "
          f"H2 {c['sigma_from_H2_total_v2']:+.2f} sigma, power vs H1 {c['power_H1_total_v2']:.2f}")
    print("SAMI jackknife detail:")
    for o in R["validation"]["sami:S2_diskbearing"]["jackknife"]["leave_one_out"]:
        print(f"   drop {o['dropped']:14s} n={o['n_dropped']:3d} -> {o['offset_without']:+.4f} "
              f"(shift {o['shift']:+.4f})")
    return R


if __name__ == "__main__":
    main()
