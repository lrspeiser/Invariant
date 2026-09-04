"""TEST 1, second survey -- field/cluster matched pairs from SAMI DR3.

MaNGA was not a cluster survey; SAMI deliberately observed eight rich clusters,
so its cluster arm is 3-4x larger at the same morphological purity.  This build
is kept SEPARATE from the MaNGA one on purpose.  The two must not be merged into
one sample without a cross-calibration: the stellar masses, the effective radii
and the morphological classifications come from different pipelines, and a
zero-point offset between them would appear as a spurious field/cluster signal
if the two surveys contributed unequally to the two arms.

Structural caveat carried from the acquisition: SAMI DR3 releases NO Sersic
index for the cluster arm (the cluster fits are Owers et al. 2019, which
publishes no per-galaxy structural table).  Matching therefore uses the MGE
photometry, which IS homogeneous across both arms, and does NOT control bulge
fraction.  That is a genuine weakening relative to the MaNGA build, and it is
stated in the output rather than hidden.

BLIND PROTECTION: identical to the MaNGA build.  No kinematic column enters the
matching space, the quality gate or the environment split.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import linear_sum_assignment
from scipy.special import iv, kv

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")
CLEAN = os.path.join(LANE, "clean")
SRC = os.path.join(LANE, "raw", "sami", "sami_dr3_master_galaxy_inventory.tsv")

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
G = 4.30091e-6                      # kpc (km/s)^2 / Msun
A0 = 1.2e-10
MPC_M = 3.0856775814913673e22

MATCH_FORBIDDEN = ["SIGMA_RE", "LAMBDA", "VSIGMA", "V_GAS", "VDISP", "PA_STELKIN",
                   "PA_GASKIN", "HALPHA", "vamp"]

TOL = {"logMstar": (0.10, "log10 M_star [dex]"),
       "logRd": (0.10, "log10 R_d [dex]"),
       "logSigma_b": (0.15, "log10 Sigma_b [dex]"),
       "log_gbar_2p2Rd": (0.10, "log10 g_bar(2.2 R_d) [dex]"),
       "incl_deg": (10.0, "inclination [deg]"),
       "z_spec": (0.010, "redshift [absolute]")}
MATCH = ["logMstar", "logRd", "logSigma_b", "log_gbar_2p2Rd", "incl_deg", "z_spec"]

CLUSTER_R_ON_R200_MAX = 1.0
# SAMI DR3 morph_type ladder: -9 unknown, 0 E, 0.5 E/S0, 1 S0, 1.5 S0/Sa,
# 2 Sa (early spiral), 2.5 Sa/Sb, 3 late spiral, 5 unclassifiable-other.
LTG_MORPH_MIN = 2.0      # spirals only, the analogue of the MaNGA late-type gate
DISK_MORPH_MIN = 1.0     # S0 and later, the analogue of the MaNGA disk-bearing gate


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def freeman(M, Rd, R):
    y = R / (2.0 * Rd)
    Sig0 = M / (2.0 * np.pi * Rd ** 2)
    with np.errstate(over="ignore", invalid="ignore"):
        br = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
    return 4.0 * np.pi * G * Sig0 * Rd * y ** 2 * br / R


def main():
    d = pd.read_csv(SRC, sep="\t", low_memory=False)
    print("SAMI master inventory: %d rows x %d cols" % d.shape)
    assert len(d) == 3068, "expected 3068 SAMI galaxies, got %d" % len(d)

    for c in ["z_spec", "Mstar", "ReMGE", "epsMGE_Re", "R_on_rtwo", "host_sigma_200",
              "morph_type", "R_proj_Mpc_from_cat", "PA_STELKIN", "PA_GASKIN",
              "SIGMA_RE_MGE", "r_e", "ellip", "PA", "PAMGE"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    z = d.z_spec.to_numpy()
    dA = np.asarray(COSMO.angular_diameter_distance(np.clip(z, 1e-5, None)).to("kpc"))
    kpc_as = dA * (np.pi / 180.0 / 3600.0)
    d["kpc_per_arcsec"] = kpc_as
    d["Re_kpc"] = d.ReMGE * kpc_as
    d["Rd_kpc"] = d.Re_kpc / 1.678          # same exponential convention as the MaNGA build
    d["logRd"] = np.log10(d.Rd_kpc.where(d.Rd_kpc > 0))
    d["logMstar"] = d.Mstar                 # SAMI Mstar is already log10 Msun
    M = 10.0 ** d.logMstar.to_numpy()
    d["Sigma_b_Msun_pc2"] = M / (2.0 * np.pi * (d.Rd_kpc.to_numpy() * 1e3) ** 2)
    d["logSigma_b"] = np.log10(d.Sigma_b_Msun_pc2.where(d.Sigma_b_Msun_pc2 > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = freeman(M, d.Rd_kpc.to_numpy(), 2.2 * d.Rd_kpc.to_numpy())
        d["log_gbar_2p2Rd"] = np.log10(g * 1e6 / 3.0856775814913673e19)
    # inclination from the MGE ellipticity with the same q0 = 0.20
    q = np.clip(1.0 - d.epsMGE_Re.to_numpy(), 0.2001, 1.0)
    d["incl_deg"] = np.degrees(np.arccos(np.sqrt(
        np.clip((q ** 2 - 0.04) / 0.96, 0, 1))))
    # external field proxy, sigma-free radius in Mpc
    sv = d.host_sigma_200.to_numpy() * 1e3
    R = d.R_proj_Mpc_from_cat.to_numpy() * MPC_M
    with np.errstate(invalid="ignore", divide="ignore"):
        d["gext_over_a0"] = (sv ** 2 / R) / A0
    # gas-vs-stellar kinematic misalignment, folded to [0,180]  (Test 2e, free here)
    mis = np.abs(d.PA_STELKIN - d.PA_GASKIN) % 360.0
    d["gas_star_misalign_deg"] = np.where(mis > 180, 360 - mis, mis)

    bq = (d.logMstar.notna() & d.logRd.notna() & d.logSigma_b.notna()
          & d.log_gbar_2p2Rd.notna() & d.incl_deg.between(20, 80)
          & d.has_mge.astype(bool)).fillna(False)
    ltg = (d.morph_type >= LTG_MORPH_MIN).fillna(False)

    # is_mem is a 1.0/0.0 float in this table, not a boolean
    cl = ((d.is_mem == 1) & (d.R_on_rtwo <= CLUSTER_R_ON_R200_MAX)
          & d.host_sigma_200.notna()).fillna(False)
    fi = ((d.arm.astype(str) == "GAMA") & d.cluster.isna()).fillna(False)

    print("cluster arm (confirmed member, R/R200<=%.1f): %d   quality+LTG: %d"
          % (CLUSTER_R_ON_R200_MAX, int(cl.sum()), int((cl & bq & ltg).sum())))
    print("field arm   (GAMA, no cluster)             : %d   quality+LTG: %d"
          % (int(fi.sum()), int((fi & bq & ltg).sum())))

    out_rows, summary = [], {}
    for tier, mgate in (("S1_latetype", ltg),
                        ("S2_diskbearing", d.morph_type >= DISK_MORPH_MIN)):
        mg = mgate.fillna(False)
        C = d[cl & bq & mg].reset_index(drop=True)
        F = d[fi & bq & mg].reset_index(drop=True)
        for v in MATCH:
            for bad in MATCH_FORBIDDEN:
                assert bad not in v, "BLIND VIOLATION: %s" % v
        Cm, Fm = C[MATCH].to_numpy(float), F[MATCH].to_numpy(float)
        okc, okf = np.isfinite(Cm).all(1), np.isfinite(Fm).all(1)
        C, F, Cm, Fm = C[okc].reset_index(drop=True), F[okf].reset_index(drop=True), Cm[okc], Fm[okf]
        tols = np.array([TOL[v][0] for v in MATCH])
        D = np.abs(Cm[:, None, :] - Fm[None, :, :])
        feas = (D <= tols[None, None, :]).all(2)
        cost = np.sqrt(((D / tols[None, None, :]) ** 2).sum(2))
        cm = np.where(feas, cost, 1e6)
        ri, ci = linear_sum_assignment(cm)
        k = cm[ri, ci] < 5e5
        ri, ci = ri[k], ci[k]
        print("%-16s cluster %4d  field %4d  -> %4d pairs" % (tier, len(C), len(F), len(ri)))

        carry = ["CATID", "RA_OBJ", "DEC_OBJ", "z_spec", "logMstar", "logRd",
                 "logSigma_b", "log_gbar_2p2Rd", "incl_deg", "morph_type",
                 "Re_kpc", "SIGMA_RE_MGE", "LAMBDAR_RE", "VSIGMA_RE",
                 "gas_star_misalign_deg"]
        cc = carry + ["cluster", "R_on_rtwo", "R_proj_Mpc_from_cat", "v_pec_kms",
                      "host_sigma_200", "host_e_sigma_200", "host_N_mem_R200",
                      "gext_over_a0", "host_M_200_caustic", "host_M_200_virial",
                      "host_R_200"]
        rows = []
        for a, b in zip(ri, ci):
            r = {"tier": tier, "match_cost": float(cost[a, b])}
            for c in cc:
                if c in C.columns:
                    r["cl_" + c] = C.loc[a, c]
            for c in carry + ["SurfaceDensity"]:
                if c in F.columns:
                    r["fi_" + c] = F.loc[b, c]
            for v in MATCH:
                r["d_" + v] = float(C.loc[a, v] - F.loc[b, v])
            rows.append(r)
        P = pd.DataFrame(rows)
        out_rows.append(P)
        s = {"n_pairs": int(len(P)), "cluster_arm": int(len(C)), "field_arm": int(len(F)),
             "achieved": {}}
        for v in MATCH:
            if len(P):
                a = P["d_" + v].dropna().to_numpy()
                s["achieved"][v] = {"label": TOL[v][1], "declared_tolerance": TOL[v][0],
                                    "max_abs": float(np.abs(a).max()),
                                    "median_abs": float(np.median(np.abs(a))),
                                    "rms": float(np.sqrt((a ** 2).mean()))}
        if len(P):
            for c, k2 in (("cl_gext_over_a0", "gext_over_a0"),
                          ("cl_host_sigma_200", "host_sigma_v_kms"),
                          ("cl_R_on_rtwo", "R_over_R200")):
                q2 = P[c].dropna().to_numpy()
                if len(q2):
                    s[k2] = {"min": float(q2.min()), "median": float(np.median(q2)),
                             "max": float(q2.max())}
            s["hosts"] = sorted(set(P.cl_cluster.dropna().astype(str)))
        summary[tier] = s

    pairs = pd.concat(out_rows, ignore_index=True)
    out = os.path.join(CLEAN, "sami_matched_pairs.csv")
    pairs.to_csv(out, index=False)

    summary["_notes"] = {
        "not_mergeable_with_manga": "Stellar masses, effective radii and "
            "morphologies come from different pipelines than the MaNGA build. Do "
            "not concatenate the two pair tables into one sample without a "
            "cross-calibration; an offset between the pipelines would masquerade "
            "as a field/cluster signal.",
        "no_sersic_index_for_the_cluster_arm": "SAMI DR3 publishes no Sersic index "
            "for cluster galaxies, so bulge fraction is NOT controlled here, "
            "unlike the MaNGA build which matches on B/T to 0.15.",
        "no_gas_fraction": "SAMI has no HI counterpart in this lane, so f_gas is "
            "not matched and not bounded. The MaNGA build at least carries HI "
            "upper limits.",
        "shared_denominator_hazard": "R_on_rtwo = R_proj/R200 and R200 is "
            "proportional to sigma_200, so R/R200 and v_pec/sigma_200 both carry "
            "sigma_200. The sigma-free columns R_proj_Mpc_from_cat and v_pec_kms "
            "are carried and are what gext_over_a0 is built from.",
        "dark_matter_dependent": ["host_M_200_caustic", "host_M_200_virial",
                                  "host_R_200", "R_on_rtwo"],
    }
    with open(os.path.join(CLEAN, "sami_matched_pairs_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    man = {"file": "sami_matched_pairs.csv",
           "produced_by": "env-data/code/build_sami_pairs.py",
           "sha256": sha256(out), "bytes": os.path.getsize(out),
           "row_count": int(len(pairs)), "column_count": int(pairs.shape[1]),
           "columns": list(pairs.columns),
           "input": {"file": os.path.basename(SRC), "sha256": sha256(SRC),
                     "row_count": 3068},
           "blind_protection": "no kinematic column in the matching space or the "
                               "environment split"}
    with open(out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print("WROTE %s : %d rows" % (out, len(pairs)))

    m = d[(d.gas_star_misalign_deg.notna())]
    print("\nSAMI gas-vs-stellar misalignment (published kinematic PAs, n=%d):" % len(m))
    for lo, hi, nm in ((0, 30, "aligned"), (30, 60, "mild"), (60, 120, "ORTHOGONAL"),
                       (120, 150, "strong"), (150, 180, "COUNTER-ROT")):
        k2 = m.gas_star_misalign_deg.between(lo, hi)
        print("   %-12s %3d-%3d deg : %4d (%.1f%%)" % (nm, lo, hi, int(k2.sum()),
                                                       100 * k2.mean()))


if __name__ == "__main__":
    main()
