"""
Confound checks that must accompany any leverage claim.

C1. SOURCE-ENVIRONMENT CONFOUND.  A sight line with large I_q ends, more often
    than not, in a void.  If the distance indicator itself has an
    environment-dependent bias (Tully-Fisher and fundamental-plane calibrations
    do depend on galaxy type, and void galaxies are a different population),
    that bias would masquerade as a path effect.  We measure how strongly I_q
    predicts the density AT THE SOURCE, and how much of I_q survives after the
    source's own environment is projected out.

C2. SHARED-DENOMINATOR SIMULATION.  We simulate the null (no path effect at
    all: ln(1+z) = c1 D exactly, plus the declared noise) and check that the
    naive estimator of c2 has zero expectation, and that the transverse
    estimator does too.  This is the check that was skipped when rho_p = -0.304
    was retracted.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import C_KMS, FootprintMask, sky_to_unit, utc_now
from density_field import DensityField

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
DESIVAST = os.path.join(LANE, "raw", "desivast")
C1_FID = 3.335641e-4
SIGMA_V = 300.0


def main():
    d = pd.read_csv(os.path.join(LANE, "path_integrals_analysed.csv"))
    out = {"generated_utc": utc_now()}

    # ---- C1: density at the source ------------------------------------
    fields = {}
    for cap in ("NGC", "SGC"):
        with fits.open(os.path.join(
                DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits")) as h:
            g = h[3].data
            m = np.asarray(g["OUT"]) == 0
            X = np.stack([np.asarray(g[k], float)[m] for k in ("X", "Y", "Z")], 1)
        r = np.linalg.norm(X, axis=1)
        fm = FootprintMask(np.degrees(np.arctan2(X[:, 1], X[:, 0])) % 360,
                           np.degrees(np.arcsin(X[:, 2] / r)), pix_deg=0.5)
        fields[cap] = DensityField(X, fm, dx=4.0, smooth=5.0, name=cap)
        print(f"field {cap} rebuilt", flush=True)

    U = sky_to_unit(d["ra"].to_numpy(float), d["dec"].to_numpy(float))
    rend = d["r_end_mpch"].to_numpy(float)
    pts = U * rend[:, None]
    dsrc = np.zeros(len(d))
    for cap in ("NGC", "SGC"):
        m = (d["cap"] == cap).to_numpy()
        if not m.any():
            continue
        v, ins = fields[cap].sample(fields[cap].delta, pts[m])
        dsrc[m] = v
    d["delta_at_source"] = dsrc

    conf = {}
    for alg in ["VoidFinder", "REVOLVER", "VIDE", "ZOBOV_ellipsoid"]:
        I = d[f"I_q_{alg}"].to_numpy(float)
        dI = d[f"dI_q_{alg}"].to_numpy(float)
        conf[alg] = {
            "corr_I_q_with_delta_at_source": float(np.corrcoef(I, dsrc)[0, 1]),
            "corr_transverse_dI_q_with_delta_at_source":
                float(np.corrcoef(dI, dsrc)[0, 1]),
        }
        # how much transverse leverage survives projecting out delta_at_source
        A = np.stack([np.ones_like(dsrc), rend, dsrc], 1)
        c, *_ = np.linalg.lstsq(A, dI, rcond=None)
        resid = dI - A @ c
        conf[alg]["transverse_std_before_mpch"] = float(dI.std())
        conf[alg]["transverse_std_after_removing_source_density_mpch"] = float(
            resid.std())
        conf[alg]["fraction_of_leverage_retained"] = float(
            resid.std() / max(1e-9, dI.std()))
    out["source_environment_confound"] = conf
    out["delta_at_source_summary"] = {
        "median": float(np.median(dsrc)), "mean": float(dsrc.mean()),
        "frac_negative": float((dsrc < 0).mean())}

    # ---- C2: null simulation with the REAL error structure --------------
    # The decisive test regresses on the INDEPENDENT distance, which is the
    # noisy axis, while I_q is built from the ray truncated at D_C(z).  So I_q
    # knows the TRUE distance and the regressor does not.  Under the null that
    # asymmetry alone manufactures a non-zero c2.  This is the same shape of
    # artefact that retracted rho_p = -0.304, so it is simulated, not assumed.
    D_true = rend
    fr = np.log(10.0) / 5.0 * d["sigma_mu"].to_numpy(float)
    sig_pec = SIGMA_V / C_KMS
    rng = np.random.default_rng(9090)
    nullres = {}
    NSIM = 2000
    for alg in ["VoidFinder", "REVOLVER"]:
        for key, x in (("raw_I_q", d[f"I_q_{alg}"].to_numpy(float)),
                       ("transverse_dI_q", d[f"dI_q_{alg}"].to_numpy(float))):
            c2s = np.empty(NSIM)
            for s in range(NSIM):
                # truth: ln(1+z) = c1 * D_true, no path term at all
                y = C1_FID * D_true + rng.normal(0, sig_pec, len(D_true))
                # the independent distance is measured with error
                D_obs = D_true * (1.0 + rng.normal(0, fr, len(D_true)))
                A = np.stack([np.ones_like(D_obs), D_obs, x], 1)
                W = 1.0 / (sig_pec ** 2 + (C1_FID * fr * D_true) ** 2)
                N = A.T @ (A * W[:, None])
                b = A.T @ (W * y)
                c2s[s] = np.linalg.solve(N, b)[2]
            nullres[f"{alg}_{key}"] = {
                "null_mean_c2": float(c2s.mean()),
                "null_std_c2": float(c2s.std()),
                "null_mean_over_std": float(c2s.mean() / max(1e-30, c2s.std())),
                "null_mean_over_c1": float(c2s.mean() / C1_FID),
                "bias_free": bool(abs(c2s.mean()) < 0.25 * c2s.std()),
            }
            print(f"  null {alg} {key}: mean/sigma = "
                  f"{c2s.mean()/max(1e-30,c2s.std()):+.2f}", flush=True)
    out["null_simulation"] = nullres
    out["null_simulation_note"] = (
        "Truth is ln(1+z) = c1 D with NO path term. A non-zero null mean means "
        "the estimator invents a path signal from the distance error alone.")

    d.to_csv(os.path.join(LANE, "path_integrals_analysed.csv"), index=False)
    with open(os.path.join(LANE, "confounds.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2)[:2500])
    return out


if __name__ == "__main__":
    main()
