"""Write points.json: every object's place in the common (S, M, r/R500) space.

The eFEDS rows are the per-system chi2-optimal amplitude, which is a summary
of the shear fit rather than the fit itself -- the fit is done on the shear,
not on these numbers -- and is labelled as such.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as K
import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    bd = K.Bundle(verbose=False, r500_mode="cat")
    J = F.Joint(bd)
    S, dS = J.project("H0", None)
    p = bd.F.gplus(S, dS, 1.0)
    w = bd.F.w
    num = np.bincount(bd.F.sysi, weights=w * p * bd.F.gt, minlength=496)
    den = np.bincount(bd.F.sysi, weights=w * p * p, minlength=496)
    cstar = num / np.maximum(den, 1e-300)
    r500 = bd.ef_R500c
    ef = []
    for j, c in enumerate(bd.ef):
        ef.append(dict(id=c.id, z=c.z,
                       M_gas500=float(K.M_at(c, r500[j]) / K.MSUN),
                       R500_Mpc=float(r500[j] / K.MPC),
                       kT_keV=c.kT, S_chi2opt=float(cstar[j]),
                       n_bins=int(np.sum(bd.F.sysi == j))))
    out = dict(
        note="S_chi2opt for eFEDS is the per-system chi2-optimal linear scale "
             "of the RAR no-slip prediction; it is a SUMMARY, the fit itself "
             "is done on the 3365 shear points.",
        efeds=ef,
        locuss=[dict(id=r["cid"], S=r["S"],
                     M_gas500=float(math.exp(r["lnM"]) * 1e14),
                     r_Mpc=float(r["r"] / K.MPC),
                     x=float(math.exp(r["lnx"])), kT_keV=None,
                     e_stat=r["e_stat"]) for r in bd.lo_rows],
        sl=[dict(cluster=r["cid"], system=r["sid"], n_img=r["n_img"],
                 theta_as=r["theta_as"], z_s=r["z_s"], S=r["S"],
                 M_gas500=float(math.exp(r["lnM"]) * 1e14),
                 x=float(math.exp(r["lnx"])), kappa_bar=r["kappa_bar"],
                 e_stat=r["e_stat"]) for r in bd.sl_rows])
    # binned summary of the common space
    lnx = np.concatenate([bd.ef_x["lnx"],
                          np.array([r["lnx"] for r in bd.lo_rows]),
                          np.array([r["lnx"] for r in bd.sl_rows])])
    lab = (["efeds"] * len(bd.ef_x["lnx"]) + ["locuss"] * len(bd.lo_rows)
           + ["sl"] * len(bd.sl_rows))
    edges = np.array([-3.5, -2.5, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 2.5,
                      3.5])
    binned = []
    for i in range(len(edges) - 1):
        m = (lnx >= edges[i]) & (lnx < edges[i + 1])
        if m.sum() == 0:
            continue
        binned.append(dict(lnx_lo=float(edges[i]), lnx_hi=float(edges[i + 1]),
                           n=int(m.sum()),
                           by_survey={s: int(np.sum([lab[k] == s
                                                     for k in np.where(m)[0]]))
                                      for s in ("efeds", "locuss", "sl")}))
    out["radius_occupancy"] = binned
    with open(os.path.join(HERE, "points.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("wrote points.json:", len(ef), "eFEDS,", len(out["locuss"]),
          "LoCuSS,", len(out["sl"]), "SL")
    print("eFEDS per-system S: median %.3f, 16-84%% %.3f-%.3f"
          % (np.median(cstar), *np.percentile(cstar, [16, 84])))


if __name__ == "__main__":
    main()
