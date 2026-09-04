"""PART 8 -- sensitivity of the verdict to every declared modelling choice.

Each variant re-runs the whole hierarchy and the whole leave-one-survey-out
transfer, so what is compared is the VERDICT, not a single coefficient.
Writes sensitivity_results.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as K                                              # noqa: E402
import decl                                                     # noqa: E402
import fitlib as F                                              # noqa: E402
import pipeline as P                                            # noqa: E402

# The Abel quadrature is run at n_t = 200 here rather than 500.  The SIS test
# in test_transition.py shows Sigma is identical from n_t = 200 to 8000 at
# 1e-4, and this sweep needs ~2000 projections.
_ORIG = P.sigma_from_g
P.sigma_from_g = (lambda r, g, R, r_trunc_mpc=20.0, n_R=260, n_t=200:
                  _ORIG(r, g, R, r_trunc_mpc, n_R, n_t))

BETA = np.round(np.arange(-1.00, 0.41, 0.05), 4)
GAMMA = np.round(np.arange(-0.60, 0.61, 0.05), 4)
T0 = time.time()


def run_variant(name, note, bundle_kw, mass_key="lnM", prior_over=None):
    saved = {k: dict(v) for k, v in decl.OFFSET_PRIORS.items()}
    if prior_over:
        for k, v in prior_over.items():
            decl.OFFSET_PRIORS[k]["sd"] = v
    bd = K.Bundle(verbose=False, **bundle_kw)
    J0 = F.Joint(bd, mass_key=mass_key)
    sc = F.unpack("H_P", J0.fit("H_P"))
    fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
          sc["sigma_int_sl_cluster"])
    J = F.Joint(bd, mass_key=mass_key, fixed_scatter=fs, cache=J0._cache)
    N = len(bd.F.gt) + len(J.lo_lnS) + len(J.sl_lnS)
    kf = dict(H0=0, H_P=3, H_M=2, H_R=2, H_G=2, H_MR=3)
    rows = {}
    for m in ("H0", "H_P", "H_M", "H_R", "H_G", "H_MR"):
        grid = (BETA if m in ("H_R", "H_MR")
                else GAMMA if m == "H_G" else None)
        r = J.fit(m, shape_grid=grid) if grid is not None else J.fit(m)
        rows[m] = dict(m2lnL=r["m2lnL"],
                       bic=r["m2lnL"] + kf[m] * math.log(N),
                       pars=F.unpack(m, r))
    bmin = min(v["bic"] for v in rows.values())
    for v in rows.values():
        v["dbic"] = v["bic"] - bmin
    best = min(rows, key=lambda m: rows[m]["bic"])

    # leave-one-survey-out transfer, headline number only
    trans = {}
    for held in ("locuss", "sl"):
        train = tuple(s for s in K.SURVEYS if s != held)
        Jt = F.Joint(bd, surveys=train, mass_key=mass_key, fixed_scatter=fs,
                     cache=J._cache)
        rows_h = {}
        for m in ("H0", "H_M", "H_R", "H_G", "H_MR"):
            grid = (BETA if m in ("H_R", "H_MR")
                    else GAMMA if m == "H_G" else None)
            r = Jt.fit(m, shape_grid=grid) if grid is not None else Jt.fit(m)
            p = F.unpack(m, r)
            src = bd.lo_rows if held == "locuss" else bd.sl_rows
            x = np.array([[q[mass_key], q["lnx"], q["lng"]] for q in src])
            obs = np.array([q["lnS"] for q in src])
            e = np.sqrt(np.array([q["e_stat"] for q in src]) ** 2
                        + (fs[0] ** 2 if held == "locuss"
                           else fs[1] ** 2 + fs[2] ** 2))
            pred = (p.get("c", 0.0) + p.get("alpha", 0.0) * x[:, 0]
                    + p.get("beta", 0.0) * x[:, 1]
                    + p.get("gamma", 0.0) * x[:, 2])
            z = (obs - pred) / e
            rows_h[m] = dict(pred_S=float(math.exp(pred.mean())),
                             obs_S=float(math.exp(obs.mean())),
                             sigma=float(z.mean() * math.sqrt(len(z))))
        trans[held] = rows_h
    for k, v in saved.items():
        decl.OFFSET_PRIORS[k] = v
    print(f"\n   {name}   ({note})")
    print(f"      n = {len(bd.F.gt)} + {len(J.lo_lnS)} + {len(J.sl_lnS)};"
          f"  best on BIC = {best}")
    print(f"      {'model':6s} {'dBIC':>8s}  key parameter")
    for m in ("H0", "H_P", "H_M", "H_R", "H_G", "H_MR"):
        p = rows[m]["pars"]
        kp = " ".join(f"{k}={v:+.3f}" for k, v in p.items()
                      if k in ("alpha", "beta", "gamma"))
        print(f"      {m:6s} {rows[m]['dbic']:8.2f}  {kp}")
    for held, rh in trans.items():
        s = "  ".join(f"{m}:{v['sigma']:+.1f}" for m, v in rh.items())
        print(f"      held-out {held:7s} pull in sigma  {s}")
    return dict(note=note, n=[len(bd.F.gt), len(J.lo_lnS), len(J.sl_lnS)],
                best_bic=best, models=rows, transfer=trans)


def main():
    print("=" * 78)
    print("PART 8.  SENSITIVITY OF THE VERDICT")
    print("=" * 78)
    out = {}
    out["primary"] = run_variant(
        "PRIMARY", "R500 = external catalogue, M = M_gas(<R500), stars x1",
        dict(r500_mode="cat"))
    out["r500_dyn"] = run_variant(
        "R500_dyn", "declared alternative: R500 from the frozen law and the "
        "baryons alone; adds A370 to the SL sample; LoCuSS radius axis is "
        "CIRCULAR here and the result must be read with that in mind",
        dict(r500_mode="dyn"))
    out["mass_is_kT"] = run_variant(
        "M -> kT", "declared alternative mass axis: core-excised X-ray "
        "temperature, which breaks the shared path with the density fit",
        dict(r500_mode="cat"), mass_key="lnkT")
    out["stars_half"] = run_variant(
        "stars x0.5", "strong-lens stellar template halved",
        dict(r500_mode="cat", star_mult=0.5))
    out["stars_double"] = run_variant(
        "stars x2", "strong-lens stellar template doubled",
        dict(r500_mode="cat", star_mult=2.0))
    out["sl_theta_lt_100"] = run_variant(
        "SL theta < 100\"", "strong-lens image systems restricted to mean "
        "radius < 100 arcsec, roughly twice the largest cluster Einstein "
        "radius known, so that only cluster-scale critical-curve tracers are "
        "kept.  This removes the eFEDS/SL overlap in r/R500 almost entirely",
        dict(r500_mode="cat", theta_max=100.0))
    out["sl_theta_lt_60"] = run_variant(
        "SL theta < 60\"", "tighter still: only systems inside the largest "
        "observed cluster Einstein radii",
        dict(r500_mode="cat", theta_max=60.0))
    out["locuss_prior_wide"] = run_variant(
        "LoCuSS prior 0.10 dex", "widened because M_WL is an NFW-FITTED mass, "
        "not raw shear -- the profile-shape systematic is not in Okabe's "
        "quoted calibration budget",
        dict(r500_mode="cat"),
        prior_over=dict(locuss=0.10 * math.log(10.0)))
    out["sl_prior_wide"] = run_variant(
        "SL prior 0.30 dex", "doubled, to test whether the strong-lens anchor "
        "is doing the work through its prior rather than its data",
        dict(r500_mode="cat"),
        prior_over=dict(sl=0.30 * math.log(10.0)))
    out["seconds"] = time.time() - T0
    with open(os.path.join(HERE, "sensitivity_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n   wrote sensitivity_results.json in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
