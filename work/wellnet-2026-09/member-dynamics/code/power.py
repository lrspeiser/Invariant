r"""
PRE-STATEMENT OF POWER -- run and recorded BEFORE any field-minus-cluster
offset is evaluated.

The width of the blocked sign-flip null depends only on the MAGNITUDES |dY_i|
of the paired differences and on the host-system block structure.  It does not
depend on their signs, and therefore carries no information about the offset
being tested.  This script computes, for every tier:

  * the number of pairs surviving the declared quality cuts,
  * the number of independent host systems and the effective number,
  * the standard deviation of the estimator from the blocked sign-flip null,
  * the 3-sigma minimum detectable offset,
  * the power at alpha = 0.05 to detect the potential-depth prediction
    (+0.0155 dex in log V, i.e. +0.031 dex in log g),

and writes power_prestatement.json.  analyse.py is run afterwards.
"""
from __future__ import annotations

import json
import os
from math import erf, sqrt

import numpy as np

import analyse as A


def cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def power(mu, sd, alpha=0.05):
    zc = 1.959963984540054
    z = mu / sd
    return cdf(z - zc) + cdf(-z - zc)


def main():
    dm, kinm = A.build_manga()
    ds, kins = A.build_sami()
    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], A.ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], A.ASYM_PERCENTILE))
    A.CUTS["manga"]["A_kin"] = a_manga
    A.CUTS["sami"]["k51"] = a_sami

    out = {"prediction_logV": [A.PRED_V, A.PRED_V_SD],
           "prediction_logg": [A.PRED_G, A.PRED_G_SD],
           "asym_threshold_manga_A_kin": a_manga,
           "asym_threshold_sami_k51": a_sami,
           "note": "computed from |dY| and the block structure only; carries no "
                   "information about the sign of the offset",
           "tiers": {}}

    for survey, df, thr, dcols in (("manga", dm, a_manga, A.DCOLS_MANGA),
                                   ("sami", ds, a_sami, A.DCOLS_SAMI)):
        for tier in sorted(df["tier"].unique()):
            d = df[df["tier"] == tier].copy()
            n0 = len(d)
            d = d[A.apply_cuts(d, survey, thr)]
            if len(d) < 8:
                out["tiers"][f"{survey}:{tier}"] = dict(n_declared=n0, n_used=len(d),
                                                        status="too_few")
                continue
            dy = (d["cl_Y"] - d["fi_Y"]).to_numpy()
            D = np.nan_to_num(d[dcols].to_numpy(), nan=0.0)
            sysid = d["sysid"].to_numpy()
            perm = A.blocked_signflip(dy, D, sysid, nperm=20000, adjust=True)
            sd = float(np.std(perm, ddof=1))
            sysu, sysn = np.unique(sysid, return_counts=True)
            rec = dict(
                survey=survey, tier=tier, n_declared=n0, n_used=int(len(d)),
                n_systems=int(len(sysu)),
                n_eff_systems=float(sysn.sum() ** 2 / (sysn ** 2).sum()),
                largest_system_share=float(sysn.max() / sysn.sum()),
                sd_estimator_from_null=sd,
                rms_pair_difference=float(np.sqrt(np.mean(dy ** 2))),
                sd_pairwise_naive=float(np.std(dy, ddof=1) / np.sqrt(len(dy))),
                design_effect=float(sd ** 2 / (np.std(dy, ddof=1) ** 2 / len(dy))),
                mdd_3sigma_logV=3 * sd, mdd_3sigma_logg=6 * sd,
                power_at_prediction=power(A.PRED_V, sd),
                power_at_prediction_lo=power(max(A.PRED_V - A.PRED_V_SD, 1e-9), sd),
                power_at_prediction_hi=power(A.PRED_V + A.PRED_V_SD, sd),
                can_distinguish_at_3sigma=bool(3 * sd <= A.PRED_V),
            )
            out["tiers"][f"{survey}:{tier}"] = rec
            print(f"{survey:6s} {tier:16s} N={rec['n_used']:4d}/{n0:4d} "
                  f"sys={rec['n_systems']:3d} (eff {rec['n_eff_systems']:5.1f}) "
                  f"sd={sd:.4f} MDD3={3*sd:.4f} power={rec['power_at_prediction']:.3f}")

    p = os.path.join(A.LANE, "power_prestatement.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
