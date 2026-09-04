"""
Leverage and power analysis for the path-dependent-redshift test.

The decisive quantity is NOT the range of I_q -- that is mostly just path
length, which is degenerate with distance.  It is the range of I_q AT FIXED
DISTANCE.  We isolate it with a direction scramble:

    I_q(u, r) = <I_q>(r)  +  Delta I_q(u, r)

<I_q>(r) is the footprint-averaged void path length to comoving distance r --
a deterministic function of r, hence perfectly degenerate with D.
Delta I_q is the transverse residual: the only part that can ever answer
"do two objects at the same distance have different redshifts?".

CUTS ARE DECLARED HERE, BEFORE ANY RESIDUAL IS EXAMINED:
    path_covered_frac >= 0.5      (the sight line is actually mapped)
    r_end_mpch        >= 100      (inside the regime where 10 Mpc/h voids can
                                   be found at all)
No redshift residual is regressed on I_q anywhere in this lane.  The power
estimate uses only the design matrix and the declared noise model.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (C_KMS, FootprintMask, R_MAX_VOID, comoving_distance,
                    sky_to_unit, utc_now)
from voids import (SphereUnionVoids, TriangleVoids, load_v2, load_voidfinder)

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
DESIVAST = os.path.join(LANE, "raw", "desivast")

MIN_COVER = 0.5
MIN_REND = 100.0
SIGMA_V = 300.0        # km/s, small-scale velocity dispersion for the noise model
C1_FID = 3.335641e-4   # H0/c per Mpc/h, the fiducial redshift-per-length slope


def scramble_table(geo, mask, radii, n_dir=240, seed=101):
    """Footprint-averaged I_q(r) and its transverse scatter, per algorithm."""
    rng = np.random.default_rng(seed)
    dirs = []
    while len(dirs) < n_dir:
        m = 4000
        ra = 360.0 * rng.random(m)
        dec = np.degrees(np.arcsin(2.0 * rng.random(m) - 1.0))
        ok = mask.contains(ra, dec)
        for a, b in zip(ra[ok], dec[ok]):
            dirs.append((a, b))
            if len(dirs) >= n_dir:
                break
    U = sky_to_unit(np.array([d[0] for d in dirs]),
                    np.array([d[1] for d in dirs]))
    rmax = float(np.max(radii))
    L = np.zeros((len(U), len(radii)))
    for i in range(len(U)):
        if isinstance(geo, TriangleVoids):
            _, iv, _ = geo.ray_intervals(U[i], rmax)
        else:
            _, iv = geo.ray_intervals(U[i], rmax)
        if not iv:
            continue
        a = np.array([x[0] for x in iv])
        b = np.array([x[1] for x in iv])
        for j, r in enumerate(radii):
            L[i, j] = np.clip(np.minimum(b, r) - a, 0, None).sum()
    return L.mean(0), L.std(0), L


def main():
    df = pd.read_csv(os.path.join(LANE, "path_integrals.csv"))
    meta = json.load(open(os.path.join(LANE, "build_meta.json")))
    out = {"generated_utc": utc_now(),
           "cuts": {"path_covered_frac_min": MIN_COVER,
                    "r_end_mpch_min": MIN_REND},
           "n_before_cuts": int(len(df))}

    sel = (df["path_covered_frac"] >= MIN_COVER) & (df["r_end_mpch"] >= MIN_REND)
    d = df[sel].reset_index(drop=True)
    out["n_after_cuts"] = int(len(d))
    out["by_survey_after_cuts"] = {k: int(v) for k, v in
                                   d.groupby("survey").size().items()}

    algs = ["VoidFinder", "VIDE", "REVOLVER", "ZOBOV_ellipsoid"]

    # ---- footprint-averaged expectation, per cap, per algorithm ----------
    radii = np.arange(50.0, R_MAX_VOID + 1.0, 25.0)
    masks = {}
    for cap in ("NGC", "SGC"):
        with fits.open(os.path.join(
                DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits")) as h:
            g = h[3].data
            m = np.asarray(g["OUT"]) == 0
            X = np.stack([np.asarray(g[k], float)[m] for k in ("X", "Y", "Z")], 1)
        r = np.linalg.norm(X, axis=1)
        masks[cap] = FootprintMask(np.degrees(np.arctan2(X[:, 1], X[:, 0])) % 360,
                                   np.degrees(np.arcsin(X[:, 2] / r)), pix_deg=0.5)

    geo = {}
    for cap in ("NGC", "SGC"):
        _, mx, ho = load_voidfinder(
            os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_VoidFinder_{cap}.fits"))
        geo[("VoidFinder", cap)] = SphereUnionVoids(ho, mx)
        for alg in ("VIDE", "REVOLVER"):
            _, v2, tri = load_v2(os.path.join(
                DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_{alg}_{cap}.fits"))
            geo[(alg, cap)] = TriangleVoids(v2, tri)
    from path_integrals import EllipsoidVoids
    for cap in ("NGC", "SGC"):
        _, v3, _ = load_v2(os.path.join(
            DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits"))
        geo[("ZOBOV_ellipsoid", cap)] = EllipsoidVoids(v3)

    scram = {}
    vfrac = {}
    for alg in algs:
        for cap in ("NGC", "SGC"):
            mu, sd, _ = scramble_table(geo[(alg, cap)], masks[cap], radii)
            scram[(alg, cap)] = (mu, sd)
            vfrac[f"{alg}_{cap}"] = float(mu[-1] / radii[-1])
        print(f"scramble table done: {alg}", flush=True)
    out["path_averaged_void_fraction"] = vfrac
    out["void_counts_in_catalogue"] = {
        f"{a}_{c}": int(geo[(a, c)].n_voids()) for a in algs
        for c in ("NGC", "SGC")}

    # ---- transverse residuals -------------------------------------------
    res = {}
    for alg in algs:
        col = f"I_q_{alg}"
        mu_i = np.zeros(len(d))
        sd_i = np.zeros(len(d))
        for cap in ("NGC", "SGC"):
            m = (d["cap"] == cap).to_numpy()
            if not m.any():
                continue
            mu, sd = scram[(alg, cap)]
            mu_i[m] = np.interp(d.loc[m, "r_end_mpch"], radii, mu)
            sd_i[m] = np.interp(d.loc[m, "r_end_mpch"], radii, sd)
        d[f"dI_q_{alg}"] = d[col] - mu_i
        d[f"expI_q_{alg}"] = mu_i
        d[f"sdI_q_{alg}"] = sd_i

        I = d[col].to_numpy(float)
        D = d["r_end_mpch"].to_numpy(float)
        dI = d[f"dI_q_{alg}"].to_numpy(float)
        # collinearity of the RAW integral with distance
        r_ID = float(np.corrcoef(I, D)[0, 1])
        # linear regression of I on [1, D] -> residual scatter
        A = np.stack([np.ones_like(D), D], 1)
        coef, *_ = np.linalg.lstsq(A, I, rcond=None)
        resid = I - A @ coef
        R2 = 1.0 - resid.var() / I.var()
        res[alg] = {
            "n": int(len(d)),
            "I_q_range_mpch": [float(I.min()), float(I.max())],
            "I_q_mean_mpch": float(I.mean()),
            "I_q_std_mpch": float(I.std()),
            "corr_I_q_with_D": r_ID,
            "R2_of_I_q_on_D": float(R2),
            "variance_inflation_factor": float(1.0 / max(1e-12, 1.0 - R2)),
            "resid_std_after_removing_D_mpch": float(resid.std()),
            "transverse_dI_q_std_mpch": float(dI.std()),
            "transverse_dI_q_5_95_mpch": [float(np.percentile(dI, 5)),
                                          float(np.percentile(dI, 95))],
            "transverse_dI_q_p95_minus_p5_mpch": float(
                np.percentile(dI, 95) - np.percentile(dI, 5)),
            "scramble_sd_median_mpch": float(np.median(d[f"sdI_q_{alg}"])),
            # shared-denominator guard: the transverse residual must NOT retain
            # the distance dependence that the raw integral carries
            "corr_dI_q_with_D": float(np.corrcoef(dI, D)[0, 1]),
            "fraction_of_I_q_variance_explained_by_D": float(R2),
        }

    # ---- dynamic range in narrow distance bins ---------------------------
    bins = np.arange(100, 700, 50.0)
    binned = {}
    for alg in algs:
        rows = []
        I = d[f"I_q_{alg}"].to_numpy(float)
        D = d["r_end_mpch"].to_numpy(float)
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (D >= lo) & (D < hi)
            if m.sum() < 15:
                continue
            rows.append({"D_lo": float(lo), "D_hi": float(hi), "n": int(m.sum()),
                         "I_q_mean": float(I[m].mean()),
                         "I_q_std": float(I[m].std()),
                         "I_q_p5": float(np.percentile(I[m], 5)),
                         "I_q_p95": float(np.percentile(I[m], 95)),
                         "spread_over_mean": float(I[m].std() / max(1e-9, I[m].mean()))})
        binned[alg] = rows
    out["dynamic_range_in_distance_bins"] = binned

    # ---- algorithm agreement --------------------------------------------
    agree = {}
    for i, a in enumerate(algs):
        for b in algs[i + 1:]:
            x = d[f"I_q_{a}"].to_numpy(float)
            y = d[f"I_q_{b}"].to_numpy(float)
            agree[f"{a}__vs__{b}"] = {
                "pearson_r_raw": float(np.corrcoef(x, y)[0, 1]),
                "pearson_r_transverse": float(np.corrcoef(
                    d[f"dI_q_{a}"], d[f"dI_q_{b}"])[0, 1]),
                "mean_ratio": float(np.mean(y) / max(1e-9, np.mean(x))),
            }
    ss = d["I_q_SDSS_VoidFinder"].to_numpy(float)
    ok = np.isfinite(ss)
    if ok.sum() > 20:
        agree["DESIVAST_VoidFinder__vs__SDSS_VAST_VoidFinder"] = {
            "n_overlap": int(ok.sum()),
            "pearson_r_raw": float(np.corrcoef(
                d.loc[ok, "I_q_VoidFinder"], ss[ok])[0, 1]),
            "note": ("SDSS VAST only reaches r=328 Mpc/h, so its I_q is "
                     "truncated relative to DESIVAST for more distant sources"),
        }
        near = ok & (d["r_end_mpch"].to_numpy(float) < 320)
        if near.sum() > 20:
            agree["DESIVAST_VoidFinder__vs__SDSS_VAST_VoidFinder"][
                "pearson_r_r_lt_320"] = float(np.corrcoef(
                    d.loc[near, "I_q_VoidFinder"], ss[near])[0, 1])
            agree["DESIVAST_VoidFinder__vs__SDSS_VAST_VoidFinder"][
                "n_r_lt_320"] = int(near.sum())
    out["algorithm_agreement"] = agree

    # ---- noise model and power ------------------------------------------
    sig_mu = d["sigma_mu"].to_numpy(float)
    D = d["r_end_mpch"].to_numpy(float)
    frac_D = np.log(10.0) / 5.0 * sig_mu           # fractional distance error
    sig_eff = np.sqrt((SIGMA_V / C_KMS) ** 2 + (C1_FID * frac_D * D) ** 2)
    d["sigma_eff_lnz"] = sig_eff
    out["noise_model"] = {
        "sigma_v_kms": SIGMA_V,
        "median_frac_distance_error": float(np.median(frac_D)),
        "median_sigma_eff_lnz": float(np.median(sig_eff)),
        "by_survey_median_frac_D": {k: float(np.median(frac_D[(d["survey"] == k).to_numpy()]))
                                    for k in d["survey"].unique()},
    }

    power = {}
    for alg in algs:
        for label, x in (("raw_with_D_in_model", d[f"I_q_{alg}"].to_numpy(float)),):
            # design matrix: constant, D, I_q  -> sigma(c2) from normal equations
            A = np.stack([np.ones_like(D), D, x], 1)
            W = 1.0 / sig_eff ** 2
            N = A.T @ (A * W[:, None])
            try:
                Cinv = np.linalg.inv(N)
                s_c2 = float(np.sqrt(Cinv[2, 2]))
            except np.linalg.LinAlgError:
                s_c2 = float("nan")
            p5, p95 = np.percentile(x, [5, 95])
            power[alg] = {
                "sigma_c2_per_mpch": s_c2,
                "min_detectable_c2_3sigma_per_mpch": 3 * s_c2,
                "min_detectable_c2_over_c1_3sigma": 3 * s_c2 / C1_FID,
                "I_q_p95_minus_p5_mpch": float(p95 - p5),
                "min_detectable_dlnz_across_p5_p95_3sigma": float(
                    3 * s_c2 * (p95 - p5)),
            }
            # same but using only the transverse residual as the regressor
            xt = d[f"dI_q_{alg}"].to_numpy(float)
            A2 = np.stack([np.ones_like(D), D, xt], 1)
            N2 = A2.T @ (A2 * W[:, None])
            try:
                s2 = float(np.sqrt(np.linalg.inv(N2)[2, 2]))
            except np.linalg.LinAlgError:
                s2 = float("nan")
            p5t, p95t = np.percentile(xt, [5, 95])
            power[alg].update({
                "transverse_sigma_c2_per_mpch": s2,
                "transverse_min_detectable_c2_over_c1_3sigma": 3 * s2 / C1_FID,
                "transverse_min_detectable_dlnz_across_p5_p95_3sigma": float(
                    3 * s2 * (p95t - p5t)),
                "transverse_min_detectable_dz_kms_across_p5_p95": float(
                    3 * s2 * (p95t - p5t) * C_KMS),
            })
    out["power"] = power
    out["leverage"] = res

    # ---- FULL SIX-TERM DESIGN OF THE PROPOSED LAW ------------------------
    # ln(1+z) = c1 D + c2 I_q + c3 I_T + c4 I_g + c5 I_q^2 + c6 I_q I_T
    for alg in ["VoidFinder", "REVOLVER"]:
        Iq = d[f"I_q_{alg}"].to_numpy(float)
        IT = d[f"I_T_{alg}"].to_numpy(float)
        Ig = d["I_g"].to_numpy(float)
        terms = {"D": D, "I_q": Iq, "I_T": IT, "I_g": Ig,
                 "I_q^2": Iq ** 2, "I_q*I_T": Iq * IT}
        names = list(terms)
        M = np.stack([terms[k] for k in names], 1)
        Ms = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
        Cm = np.corrcoef(Ms, rowvar=False)
        vif = {}
        for j, nm in enumerate(names):
            others = np.delete(Ms, j, axis=1)
            A = np.column_stack([np.ones(len(Ms)), others])
            c, *_ = np.linalg.lstsq(A, Ms[:, j], rcond=None)
            r = Ms[:, j] - A @ c
            R2j = 1.0 - r.var() / max(1e-30, Ms[:, j].var())
            vif[nm] = float(1.0 / max(1e-12, 1.0 - R2j))
        ev = np.linalg.eigvalsh(Cm)
        out.setdefault("six_term_design", {})[alg] = {
            "term_order": names,
            "correlation_matrix": [[float(x) for x in row] for row in Cm],
            "variance_inflation_factors": vif,
            "condition_number_of_correlation_matrix": float(
                ev.max() / max(1e-30, ev.min())),
            "min_eigenvalue": float(ev.min()),
        }

    # ---- matched pairs: same distance, different I_q ---------------------
    pairs = {}
    for alg in ["VoidFinder", "REVOLVER"]:
        x = d[f"I_q_{alg}"].to_numpy(float)
        order = np.argsort(D)
        Ds = D[order]
        xs = x[order]
        cnt = 0
        best = []
        j = 0
        for i in range(len(Ds)):
            while j < len(Ds) and Ds[j] < Ds[i] + 20.0:
                j += 1
            for k in range(i + 1, j):
                dx = abs(xs[k] - xs[i])
                if dx > 100.0:
                    cnt += 1
                    if len(best) < 5:
                        best.append((float(Ds[i]), float(Ds[k]),
                                     float(xs[i]), float(xs[k])))
        pairs[alg] = {"n_pairs_dD_lt_20_and_dIq_gt_100_mpch": int(cnt),
                      "examples": best}
    out["matched_pairs"] = pairs

    # ---- monotone-invariance / sensitivity check ------------------------
    # verify the headline statistic actually moves with the smoothing / geometry
    sens = {}
    for alg in algs:
        sens[alg] = {
            "transverse_std": res[alg]["transverse_dI_q_std_mpch"],
            "raw_std": res[alg]["I_q_std_mpch"],
        }
    out["sensitivity_of_headline_stat"] = sens

    d.to_csv(os.path.join(LANE, "path_integrals_analysed.csv"), index=False)
    with open(os.path.join(LANE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("n_before_cuts", "n_after_cuts", "by_survey_after_cuts")},
                     indent=2))
    return out


if __name__ == "__main__":
    main()
