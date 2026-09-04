"""The H0' null, corrected.  Writes null_results.json.

TWO FIXES over the version inside transition.py, both forced by the numbers
coming back impossible:

 1. **A validity gate on each realisation.**  At error scale 1.0 the Bahar
    Vikhlinin parameters are redrawn at their full published MARGINAL errors,
    which are an upper bound on the independent variance because the true
    covariance is not published.  Some draws are pathological -- n0 near zero,
    beta near the 1/3 divergence -- and give a predicted shear near zero, so
    the linearised estimator's (g_t - p)/p explodes.  The first run returned
    E[c|H0] = -57 +- 798, which is not a bias, it is a broken realisation.
    Each realisation now has to pass the SAME M_gas,500 gate the ingest uses:
    the perturbed model must reproduce Bahar's own published M_gas,500 to
    within a factor of two in the median.  The rejection rate is reported.

 2. **Robust as well as moment summaries.**  Median and 1.4826 x MAD are
    reported beside the mean and sd, so a single surviving outlier cannot set
    the answer.

Accuracy note: the Abel quadrature is run at n_t = 200 rather than 500 here,
which the SIS test in test_transition.py shows is converged (Sigma is
identical from n_t = 200 to 8000 at 1e-4), and which makes 3 x 200
realisations affordable.
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
import nulls as NU                                              # noqa: E402
import pipeline as P                                            # noqa: E402

MPC, MSUN = P.MPC, P.MSUN
_ORIG = P.sigma_from_g


def _fast(r, g, R_out, r_trunc_mpc=20.0, n_R=260, n_t=200):
    return _ORIG(r, g, R_out, r_trunc_mpc, n_R, n_t)


KEYS = ("n0sq", "rs", "eps", "beta", "alpha")
EKEYS = ("e_n0sq", "e_rs", "e_eps", "e_beta", "e_alpha")


def main(n_mc=200, seed=20260904):
    rng = np.random.default_rng(seed)
    bd = K.Bundle(verbose=False, r500_mode="cat")
    J0 = F.Joint(bd)
    sc = F.unpack("H_P", J0.fit("H_P"))
    fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
          sc["sigma_int_sl_cluster"])
    o_lo, o_sl = sc["offset_locuss"], sc["offset_sl"]
    print("=" * 78)
    print("H0' NULL -- ln S = one constant per survey, no M/r/g dependence")
    print("=" * 78)
    print(f"   H0' amplitudes from H_P: LoCuSS S = {math.exp(o_lo):.3f}, "
          f"SL S = {math.exp(o_sl):.3f}")
    print(f"   frozen scatters {fs[0]:.4f} / {fs[1]:.4f} / {fs[2]:.4f}")

    # regressors and the beta lever arm, computed once on the real model
    S0, dS0 = J0.project("H0", None)
    p0 = bd.F.gplus(S0, dS0, 1.0)
    db = 0.05
    Sb, dSb = J0.project("H_R", np.array([db]))
    pb = bd.F.gplus(Sb, dSb, 1.0)
    Dbeta = (np.log(np.maximum(pb, 1e-300))
             - np.log(np.maximum(p0, 1e-300))) / db

    Pm = np.zeros((3, 6))
    for i in range(3):
        Pm[i, 3 + i] = 1.0
    Wp = np.array([1.0 / decl.OFFSET_PRIORS[k]["sd"] ** 2
                   for k in ("efeds", "locuss", "sl")])

    def linfit(gt, lo_lnS, sl_lnS, ef_lnM, lo_x, sl_x, lo_e, sl_e, pmod):
        cols, y, w = [], [], []
        yy = (gt - pmod) / np.where(np.abs(pmod) < 1e-12, 1e-12, pmod)
        ww = (pmod / bd.F.er) ** 2
        n = len(yy)
        cols.append(np.column_stack([np.ones(n), ef_lnM, Dbeta, np.ones(n),
                                     np.zeros(n), np.zeros(n)]))
        y.append(yy); w.append(ww)
        n = len(lo_lnS)
        v = lo_e ** 2 + fs[0] ** 2
        cols.append(np.column_stack([np.ones(n), lo_x[:, 0], lo_x[:, 1],
                                     np.zeros(n), np.ones(n), np.zeros(n)]))
        y.append(lo_lnS); w.append(1.0 / v)
        n = len(sl_lnS)
        v = sl_e ** 2 + fs[1] ** 2 + fs[2] ** 2
        cols.append(np.column_stack([np.ones(n), sl_x[:, 0], sl_x[:, 1],
                                     np.zeros(n), np.zeros(n), np.ones(n)]))
        y.append(sl_lnS); w.append(1.0 / v)
        X = np.vstack(cols)
        Y = np.concatenate(y)
        W = np.concatenate(w)
        A = X.T @ (W[:, None] * X) + Pm.T @ (Wp[:, None] * Pm)
        return np.linalg.solve(A, X.T @ (W * Y)), np.linalg.inv(A)

    lo_x0 = np.array([[r["lnM"], r["lnx"], r["lng"]] for r in bd.lo_rows])
    sl_x0 = np.array([[r["lnM"], r["lnx"], r["lng"]] for r in bd.sl_rows])
    lo_e0 = np.array([r["e_stat"] for r in bd.lo_rows])
    sl_e0 = np.array([r["e_stat"] for r in bd.sl_rows])
    th_obs, Cov = linfit(bd.F.gt,
                         np.array([r["lnS"] for r in bd.lo_rows]),
                         np.array([r["lnS"] for r in bd.sl_rows]),
                         bd.ef_x["lnM"], lo_x0, sl_x0, lo_e0, sl_e0, p0)
    print(f"\n   linearised estimator on the REAL data:"
          f"  c = {th_obs[0]:+.4f}  alpha = {th_obs[1]:+.4f}"
          f"  beta = {th_obs[2]:+.4f}")
    out = dict(real=dict(c=float(th_obs[0]), alpha=float(th_obs[1]),
                         beta=float(th_obs[2]),
                         fisher_sd=[float(math.sqrt(Cov[i, i]))
                                    for i in range(3)]),
               offsets=dict(locuss=o_lo, sl=o_sl), scatter=list(fs),
               n_mc=n_mc, scales={})

    recs = [dict(r) for r in bd.obs.sys]
    pub = np.array([r["Mgas500_pub"] for r in bd.obs.sys])
    lim = np.array([r["l_Mgas"] == "<" for r in bd.obs.sys])
    r500c = np.array([c.extra["R500_cat"] for c in bd.ef])
    P.sigma_from_g = _fast
    try:
        for scale in (0.25, 0.5, 1.0):
            t0 = time.time()
            keep, rej = [], 0
            while len(keep) < n_mc and rej < 20 * n_mc:
                rc2 = []
                for r in recs:
                    d = dict(r)
                    for k, ek in zip(KEYS, EKEYS):
                        e = r[ek]
                        if np.isfinite(e) and e > 0:
                            d[k] = r[k] + scale * e * rng.normal()
                    d["n0sq"] = max(d["n0sq"], 1e-6)
                    d["rs"] = max(d["rs"], 1e-4 * MPC)
                    d["beta"] = max(d["beta"], 0.34)
                    rc2.append(d)
                syss2 = [P.System(rc) for rc in rc2]
                M2 = np.array([float(np.interp(rr, s.r, s.M_gas))
                               for s, rr in zip(syss2, r500c)])
                ok = (~lim) & (pub > 0)
                ratio = np.median(M2[ok] / MSUN / 1e12 / pub[ok])
                if not (0.5 < ratio < 2.0):
                    rej += 1
                    continue
                S2, dS2 = K.project_slip(syss2, bd.obs, bd.ef_idx,
                                         lambda sm: 1.0)
                p_true = bd.F.gplus(S2, dS2, 1.0)
                gt2 = p_true + rng.normal(size=p_true.size) * bd.F.er
                lo2, loX, loE = NU.locuss_null(bd, rng, fs, o_lo, scale)
                sl2, slX, slE = NU.sl_null(bd, rng, fs, o_sl, scale)
                lnM2 = np.log(np.maximum(M2, 1e30) / K.M0)[bd.F.sysi]
                th, _ = linfit(gt2, lo2, sl2, lnM2, loX, slX, loE, slE,
                               p_true)
                if not np.all(np.isfinite(th)):
                    rej += 1
                    continue
                keep.append(th[:3])
            A = np.array(keep)
            row = dict(n_kept=len(keep), n_rejected=rej,
                       reject_frac=rej / max(rej + len(keep), 1),
                       seconds=time.time() - t0)
            for i, nm in enumerate(("c", "alpha", "beta")):
                v = A[:, i]
                med = float(np.median(v))
                mad = float(1.4826 * np.median(np.abs(v - med)))
                row[nm] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                               median=med, mad=mad,
                               p16=float(np.percentile(v, 16)),
                               p84=float(np.percentile(v, 84)))
            out["scales"][str(scale)] = row
            print(f"\n   scale {scale:4.2f}:  kept {len(keep)}, rejected"
                  f" {rej} ({100 * row['reject_frac']:.1f}%),"
                  f" {row['seconds']:.0f}s")
            for nm in ("c", "alpha", "beta"):
                v = row[nm]
                print(f"      E[{nm:5s}|H0] median {v['median']:+.5f}"
                      f"  MAD {v['mad']:.5f}"
                      f"   mean {v['mean']:+.5f}  sd {v['sd']:.5f}")
    finally:
        P.sigma_from_g = _ORIG

    print("\n   FISHER sigma against the NULL MAD -- never quote the first")
    print("   alone for a regressor built from someone else's fit:")
    fis = out["real"]["fisher_sd"]
    n1 = out["scales"]["1.0"]
    for i, nm in enumerate(("c", "alpha", "beta")):
        mad = n1[nm]["mad"]
        print(f"      {nm:6s} Fisher {fis[i]:.5f}   null MAD {mad:.5f}"
              f"   ratio {fis[i] / mad:6.3f}")
        out["real"].setdefault("fisher_over_null", {})[nm] = float(fis[i] / mad)
    print("\n   The measurement against its OWN null:")
    for i, nm in enumerate(("alpha", "beta")):
        est = out["real"][nm]
        lo_ = min(out["scales"][s][nm]["median"] for s in out["scales"])
        hi_ = max(out["scales"][s][nm]["median"] for s in out["scales"])
        mad = n1[nm]["mad"]
        print(f"      {nm:6s} = {est:+.4f};  null median brackets"
              f" [{lo_:+.4f}, {hi_:+.4f}] over the three error scalings;"
              f"\n             {(est - hi_) / mad:+.2f} to"
              f" {(est - lo_) / mad:+.2f} sigma from its own null")
        out["real"].setdefault("vs_null", {})[nm] = dict(
            est=float(est), null_bracket=[float(lo_), float(hi_)],
            mad=float(mad), sigma_lo=float((est - hi_) / mad),
            sigma_hi=float((est - lo_) / mad))
    with open(os.path.join(HERE, "null_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n   wrote null_results.json")


if __name__ == "__main__":
    main(n_mc=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
