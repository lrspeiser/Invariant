"""TEST 1 stage 2 -- field/cluster matched pairs from MaNGA DR17 + Tempel groups.

BLIND PROTECTION
  The matching uses ONLY structural, photometric and environmental quantities.
  No kinematic column enters the cost function, the tolerance box, the quality
  gate or the field/cluster split.  Enforced at run time by MATCH_FORBIDDEN.
  The only place a kinematic column is touched is the power calculation, and
  there it is restricted to the FIELD ARM ALONE, so the field-versus-cluster
  contrast this sample exists to measure is never evaluated here.

TIER GRID
  All tiers were declared before any kinematic residual was inspected.  Tier B1
  is the PRIMARY sample.  The relaxed tiers were added after seeing that the
  primary cluster arm contained only 48 galaxies -- a sample-size observation,
  not a residual observation -- and are reported separately, never merged.

KNOWN COLLINEARITY (see REPORT.md)
  log Sigma_b and log g_bar(2.2 R_d) correlate at r = 0.996 in the parent
  sample.  They are NOT independent matching variables.  The nominal "five
  matching variables" of the brief are, in practice, about three independent
  directions: M_star, R_d, and f_gas (itself correlated with M_star at -0.77).
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
CLEAN = os.path.join(LANE, "clean")

A0 = 1.2e-10                      # m/s^2, the RAR acceleration scale
MPC_M = 3.0856775814913673e22

MATCH_FORBIDDEN = ["vamp_", "STELLAR_SIGMA", "HA_GSIGMA", "STELLAR_VEL",
                   "HA_GVEL", "SFR_", "vrot", "lambda"]

# ------------------------------------------------------- DECLARED TOLERANCES
TOL = {
    "logMstar_nsa":   (0.10, "log10 M_star [dex]"),
    "logRd":          (0.10, "log10 R_d [dex]"),
    "logSigma_b":     (0.15, "log10 Sigma_b [dex]"),
    "log_gbar_2p2Rd": (0.10, "log10 g_bar(2.2 R_d) [dex]"),
    "f_gas_or_nan":   (0.10, "f_gas [absolute]"),
    "incl_deg":       (10.0, "inclination [deg]"),
    "pym_r_BT_SE":    (0.15, "B/T [absolute]"),
    "z":              (0.010, "redshift [absolute]"),
}
FIVE = ["logMstar_nsa", "logRd", "logSigma_b", "log_gbar_2p2Rd", "f_gas_or_nan"]
FOUR = ["logMstar_nsa", "logRd", "logSigma_b", "log_gbar_2p2Rd"]
CTRL = ["incl_deg", "pym_r_BT_SE", "z"]

# ------------------------------------------------------------ DECLARED TIERS
MORPH = {
    "latetype": dict(incl=(25.0, 75.0), ttype_min=0.0, pltg_min=0.5, bt_max=None,
                     desc="deep-learning late type: TType>0 and P_LTG>0.5, 25<i<75 deg"),
    "diskbearing": dict(incl=(20.0, 80.0), ttype_min=-3.0, pltg_min=None, bt_max=0.80,
                        desc="disk-bearing (S0 or later): TType>-3 and B/T<=0.80, 20<i<80 deg"),
}
ENV = {
    "strict": dict(sig=400.0, ngal=10, rrv=1.0,
                   desc="host sigma_v>=400 km/s, Ngal>=10, R_proj/R_vir<=1.0"),
    "wide": dict(sig=300.0, ngal=5, rrv=1.5,
                 desc="host sigma_v>=300 km/s, Ngal>=5, R_proj/R_vir<=1.5"),
    "xray": dict(sig=0.0, ngal=2, rrv=1.5, xray=True,
                 desc="galaxy lies within 2 Mpc projected of an MCXC X-ray peak "
                      "at |dz|<0.01 (L500 is a direct observable), and "
                      "R_proj/R_vir<=1.5 of its Tempel host"),
}
TIERS = [
    ("B1_primary",        "latetype",    "strict", FOUR),
    ("A1_gas_matched",    "latetype",    "strict", FIVE),
    ("B2_disk_strict",    "diskbearing", "strict", FOUR),
    ("B3_late_wide",      "latetype",    "wide",   FOUR),
    ("B4_disk_wide",      "diskbearing", "wide",   FOUR),
    ("C1_xray_late",      "latetype",    "xray",   FOUR),
    ("C2_xray_disk",      "diskbearing", "xray",   FOUR),
]
PRIMARY = "B1_primary"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def prepare(d):
    d = d.copy()
    d["logRd"] = np.log10(d["Rd_kpc"].where(d["Rd_kpc"] > 0))
    d["logSigma_b"] = np.log10(d["Sigma_b_Msun_pc2"].where(d["Sigma_b_Msun_pc2"] > 0))
    d["f_gas_or_nan"] = d["f_gas"]
    # external-field proxy in units of a0 (observable up to the virial-tracer
    # assumption; contains no mass and no dark-matter model)
    sv = d["t14_grp_sigma_v"].to_numpy() * 1e3                       # m/s
    R = d["t14_Rproj_kpc"].to_numpy() * MPC_M / 1e3                  # m
    with np.errstate(invalid="ignore", divide="ignore"):
        d["gext_over_a0"] = (sv ** 2 / R) / A0
    return d


def base_quality(d):
    return (d["logMstar_nsa"].notna() & d["logRd"].notna()
            & d["logSigma_b"].notna() & d["log_gbar_2p2Rd"].notna()
            & d["pym_r_BT_SE"].notna() & (d["pym_r_FLAG_FIT"] != 3)
            & (d["DAPQUAL"].fillna(-1) >= 0)).fillna(False)


def morph_gate(d, key):
    m = MORPH[key]
    g = d["incl_deg"].between(*m["incl"])
    g &= d["dl_TType"] > m["ttype_min"]
    if m["pltg_min"] is not None:
        g &= d["dl_P_LTG"] > m["pltg_min"]
    if m["bt_max"] is not None:
        g &= d["pym_r_BT_SE"] <= m["bt_max"]
    return g.fillna(False)


def cluster_gate(d, key):
    e = ENV[key]
    g = d["t14_sep_arcsec"].notna() & (d["R_over_Rvir_t14"] <= e["rrv"])
    if e.get("xray"):
        g &= d["xray_L500_1e44"].notna()
    else:
        g &= (d["t14_grp_sigma_v"] >= e["sig"]) & (d["t14_Ngal"] >= e["ngal"])
    return g.fillna(False)


def field_gate(d):
    """Tempel FoF singleton with no host group at all."""
    return (d["t14_sep_arcsec"].notna() & (d["t14_Ngal"] <= 1)
            & d["t14_grp_sigma_v"].isna()).fillna(False)


def match(cl, fi, varnames, label):
    for v in varnames:
        for bad in MATCH_FORBIDDEN:
            assert bad not in v, "BLIND VIOLATION: %s looks kinematic" % v
    tols = np.array([TOL[v][0] for v in varnames])
    C = cl[varnames].to_numpy(dtype=float)
    F = fi[varnames].to_numpy(dtype=float)
    ok_c, ok_f = np.isfinite(C).all(1), np.isfinite(F).all(1)
    cl2 = cl[ok_c].reset_index(drop=True)
    fi2 = fi[ok_f].reset_index(drop=True)
    C, F = C[ok_c], F[ok_f]
    if not len(C) or not len(F):
        return pd.DataFrame(), (len(C), len(F))
    D = np.abs(C[:, None, :] - F[None, :, :])
    feas = (D <= tols[None, None, :]).all(2)
    cost = np.sqrt(((D / tols[None, None, :]) ** 2).sum(2))
    BIG = 1e6
    cm = np.where(feas, cost, BIG)
    ri, ci = linear_sum_assignment(cm)
    k = cm[ri, ci] < BIG / 2
    ri, ci = ri[k], ci[k]

    carry_cl = ["plateifu", "mangaid", "objra", "objdec", "z", "t14_GroupID",
                "t14_grp_sigma_v", "t14_Ngal", "t14_Rproj_kpc", "R_over_Rvir_t14",
                "R_over_R200_t17", "t14_psi_norm_host_deg", "t14_theta_sky_deg",
                "incl_deg", "pa_disk_deg", "t14_PA_to_host_deg", "gext_over_a0",
                "t14_mcxc_name", "t14_mcxc_L500_1e44", "t14_grp_MNFW_rank_only",
                "xray_name", "xray_oname", "xray_L500_1e44", "xray_Rproj_Mpc",
                "xray_R_over_R500_rank_only",
                "t17_grp_M200_rank_only", "gema_grp_Q_group", "gema_lss_t1",
                "logMstar_nsa", "logRd", "logSigma_b", "log_gbar_2p2Rd",
                "f_gas_or_nan", "pym_r_BT_SE", "hi_detected", "logMHI_use",
                "logMHI_limit", "dl_TType", "dl_P_LTG", "struct_source",
                "vamp_ha_deproj_kms", "vamp_stel_deproj_kms", "STELLAR_SIGMA_1RE"]
    carry_fi = [c for c in carry_cl if not c.startswith(("t14_", "t17_", "gema_"))] + \
               ["gema_overdensity", "gema_local_density", "t14_Ngal"]
    rows = []
    for a, b in zip(ri, ci):
        r = {"tier": label, "match_cost": float(cost[a, b])}
        for c in carry_cl:
            if c in cl2.columns:
                r["cl_" + c] = cl2.loc[a, c]
        for c in carry_fi:
            if c in fi2.columns:
                r["fi_" + c] = fi2.loc[b, c]
        for v in varnames + CTRL:
            if v in cl2.columns and v in fi2.columns:
                r["d_" + v] = float(cl2.loc[a, v] - fi2.loc[b, v])
        rows.append(r)
    return pd.DataFrame(rows), (len(C), len(F))


def main():
    src = os.path.join(CLEAN, "manga_env_master.csv")
    d = prepare(pd.read_csv(src, low_memory=False))
    bq = base_quality(d)
    fg = field_gate(d)

    all_pairs, summary = [], {}
    for name, mkey, ekey, varset in TIERS:
        mg = morph_gate(d, mkey)
        cg = cluster_gate(d, ekey)
        varnames = varset + CTRL
        cl = d[cg & mg & bq].reset_index(drop=True)
        fi = d[fg & mg & bq].reset_index(drop=True)
        P, (nc, nf) = match(cl, fi, varnames, name)
        print("%-16s cluster arm %4d  field arm %5d  -> %4d pairs   [%s | %s | %d vars]"
              % (name, len(cl), len(fi), len(P), mkey, ekey, len(varnames)))
        s = {"n_pairs": int(len(P)),
             "cluster_arm_size": int(len(cl)), "field_arm_size": int(len(fi)),
             "cluster_arm_complete_data": int(nc), "field_arm_complete_data": int(nf),
             "morphology_gate": MORPH[mkey]["desc"],
             "environment_gate": ENV[ekey]["desc"],
             "matching_variables": varnames,
             "achieved": {}}
        for v in varnames:
            c = "d_" + v
            if len(P) and c in P.columns and P[c].notna().any():
                a = P[c].dropna().to_numpy()
                s["achieved"][v] = {"label": TOL[v][1], "declared_tolerance": TOL[v][0],
                                    "max_abs": float(np.abs(a).max()),
                                    "median_abs": float(np.median(np.abs(a))),
                                    "rms": float(np.sqrt((a ** 2).mean()))}
        if len(P):
            for c, k in (("cl_gext_over_a0", "gext_over_a0"),
                         ("cl_t14_grp_sigma_v", "host_sigma_v_kms"),
                         ("cl_R_over_Rvir_t14", "R_over_Rvir"),
                         ("cl_t14_psi_norm_host_deg", "psi_normal_to_host_deg")):
                if c in P.columns and P[c].notna().any():
                    q = P[c].dropna().to_numpy()
                    s[k] = {"min": float(q.min()), "median": float(np.median(q)),
                            "max": float(q.max())}
        summary[name] = s
        all_pairs.append(P)

    pairs = pd.concat([p for p in all_pairs if len(p)], ignore_index=True) \
        if any(len(p) for p in all_pairs) else pd.DataFrame()
    out = os.path.join(CLEAN, "matched_pairs.csv")
    pairs.to_csv(out, index=False)
    print("\nWROTE %s : %d rows, %d cols" % (out, len(pairs), pairs.shape[1]))

    # ---- collinearity of the matching variables (parent sample, both arms)
    base = d[bq]
    summary["_matching_variable_correlation"] = json.loads(
        base[FIVE].corr(method="pearson").to_json())
    summary["_collinearity_warning"] = (
        "log Sigma_b and log g_bar(2.2 R_d) correlate at r=%.4f: they are one "
        "matching direction, not two.  f_gas correlates with log M_star at "
        "r=%.4f.  The effective number of independent matching directions is "
        "about three, not five."
        % (base[["logSigma_b", "log_gbar_2p2Rd"]].corr().iloc[0, 1],
           base[["f_gas_or_nan", "logMstar_nsa"]].corr().iloc[0, 1]))

    # ---- power, measured on the FIELD ARM ONLY
    fq = d[fg & morph_gate(d, "latetype") & bq].copy()
    v = fq["vamp_ha_deproj_kms"]
    fq["logV"] = np.log10(v.where((v > 20) & (v < 500)))
    ok = fq["logV"].notna() & fq["logMstar_nsa"].notna()
    X = np.vstack([fq.loc[ok, "logMstar_nsa"].to_numpy(),
                   np.ones(int(ok.sum()))]).T
    y = fq.loc[ok, "logV"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sig_proxy = float(np.std(y - X @ coef, ddof=2))
    SIG_GOOD = 0.055     # dex; sTFR scatter achievable with resolved rotation curves
    pw = {"scatter_estimator": "log10(deprojected H-alpha velocity half-range from "
                               "the DAP summary) versus log10 M_star, FIELD ARM ONLY",
          "n_field_used": int(ok.sum()),
          "slope": float(coef[0]),
          "sigma_logV_dex_dap_proxy": sig_proxy,
          "sigma_logV_dex_assumed_with_resolved_rotation_curves": SIG_GOOD,
          "note": "The DAP velocity half-range is a crude rotation proxy; a proper "
                  "tilted-ring fit to the MAPS cubes should reach the literature "
                  "stellar-mass Tully-Fisher scatter of about 0.05-0.06 dex.  Both "
                  "cases are quoted.  The cluster arm was NOT used to estimate any "
                  "scatter, so the contrast under test is still blind."}
    for name, s in list(summary.items()):
        if name.startswith("_"):
            continue
        n = s["n_pairs"]
        if n:
            for tag, sg in (("dap_proxy", sig_proxy), ("resolved_rc", SIG_GOOD)):
                pw["%s__se_mean_offset_dex__%s" % (name, tag)] = \
                    float(sg * np.sqrt(2.0) / np.sqrt(n))
                pw["%s__min_detectable_3sigma_dex__%s" % (name, tag)] = \
                    float(3.0 * sg * np.sqrt(2.0) / np.sqrt(n))
    # how many pairs would be needed for a plausible effect size
    for eff in (0.02, 0.03, 0.05, 0.10):
        for tag, sg in (("dap_proxy", sig_proxy), ("resolved_rc", SIG_GOOD)):
            pw["N_pairs_needed_for_3sigma_on_%.2fdex__%s" % (eff, tag)] = \
                int(np.ceil(2.0 * (3.0 * sg / eff) ** 2))
    summary["_power"] = pw

    sp = os.path.join(CLEAN, "matched_pairs_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    man = {"file": "matched_pairs.csv",
           "produced_by": "env-data/code/build_matched_pairs.py",
           "produced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sha256": sha256(out), "bytes": os.path.getsize(out),
           "row_count": int(len(pairs)), "column_count": int(pairs.shape[1]),
           "columns": list(pairs.columns),
           "primary_tier": PRIMARY,
           "input": {"file": "manga_env_master.csv", "sha256": sha256(src)},
           "blind_protection": "no kinematic column in the matching space or the "
                               "environment split; power scatter from the field arm only"}
    with open(out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print("WROTE %s and manifest" % sp)
    print("\nPOWER: sigma(logV) field arm = %.3f dex (DAP proxy)" % sig_proxy)
    for name in [t[0] for t in TIERS]:
        n = summary[name]["n_pairs"]
        if n:
            print("   %-16s N=%3d  3-sigma min detectable: %.3f dex (proxy) / "
                  "%.3f dex (resolved RC)"
                  % (name, n, 3 * sig_proxy * np.sqrt(2) / np.sqrt(n),
                     3 * SIG_GOOD * np.sqrt(2) / np.sqrt(n)))


if __name__ == "__main__":
    main()
