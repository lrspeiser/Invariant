"""PART 6 (corrected) -- frozen transfer with the right uncertainty.

TWO BUGS IN THE FIRST VERSION, both found by asking what the number meant:

 1. The strong-lens pull was computed as mean(z) * sqrt(49), treating 49 image
    systems in 4 clusters as 49 independent measurements.  They are not: they
    share the gas model, the assumed centre and the monopole approximation.
    The fit itself carries a block covariance; this summary did not, and it
    overstated every strong-lensing significance by about sqrt(49/4) = 3.5.

 2. The held-out survey's own offset is set to its prior mean, as the freeze
    requires -- but the prediction interval must then INCLUDE the prior width,
    because that offset is genuinely unknown.  Leaving it out treats a
    calibration uncertainty of 0.15 dex as zero.

    sigma_pred^2 = prior_sd^2 + sigma_cluster^2/n_clusters
                              + (sigma_within^2 + <e_stat^2>)/n_points

Writes transfer_results.json, which render_report.py uses in place of the
part6 block.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as K                                              # noqa: E402
import decl                                                     # noqa: E402
import fitlib as F                                              # noqa: E402

BETA = np.round(np.arange(-1.00, 0.41, 0.05), 4)
GAMMA = np.round(np.arange(-0.60, 0.61, 0.05), 4)
MODELS = ("H0", "H_M", "H_R", "H_G", "H_MR")


def main():
    bd = K.Bundle(verbose=False, r500_mode="cat")
    J0 = F.Joint(bd)
    sc = F.unpack("H_P", J0.fit("H_P"))
    fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
          sc["sigma_int_sl_cluster"])
    out = {"fixed_scatter": list(fs), "held": {}}
    print("=" * 78)
    print("FROZEN TRANSFER, with the prediction interval done properly")
    print("=" * 78)
    print(f"   intrinsic scatters (frozen, from H_P): LoCuSS {fs[0]:.4f}, "
          f"SL within {fs[1]:.4f}, SL cluster-common {fs[2]:.4f}")

    for held in ("locuss", "sl", "efeds"):
        train = tuple(s for s in K.SURVEYS if s != held)
        Jt = F.Joint(bd, surveys=train, fixed_scatter=fs, cache=J0._cache)
        rows = {}
        for m in MODELS:
            grid = (BETA if m in ("H_R", "H_MR")
                    else GAMMA if m == "H_G" else None)
            r = Jt.fit(m, shape_grid=grid) if grid is not None else Jt.fit(m)
            p = F.unpack(m, r)
            c = p.get("c", 0.0)
            al, be, ga = (p.get("alpha", 0.0), p.get("beta", 0.0),
                          p.get("gamma", 0.0))
            if held == "efeds":
                S, dS = Jt.project(m, r.get("shape"))
                amp = np.exp(c + al * bd.ef_x["lnM"])
                ch = float(bd.F.chi2(S, dS, amp))
                rows[m] = dict(pars=p, chi2=ch, chi2_per_pt=ch / len(bd.F.gt))
                continue
            src = bd.lo_rows if held == "locuss" else bd.sl_rows
            x = np.array([[q["lnM"], q["lnx"], q["lng"]] for q in src])
            obs = np.array([q["lnS"] for q in src])
            est = np.array([q["e_stat"] for q in src])
            pred = c + al * x[:, 0] + be * x[:, 1] + ga * x[:, 2]
            resid = obs - pred
            prior = decl.OFFSET_PRIORS[held]["sd"]
            if held == "locuss":
                n_cl, s_w, s_c = len(obs), fs[0], 0.0
                mr = float(np.mean(resid))
            else:
                cid = np.array([q["cid"] for q in src])
                cl = sorted(set(cid))
                n_cl = len(cl)
                s_w, s_c = fs[1], fs[2]
                mr = float(np.mean([np.mean(resid[cid == c_]) for c_ in cl]))
            var = (prior ** 2 + s_c ** 2 / n_cl
                   + (s_w ** 2 + float(np.mean(est ** 2))) / len(obs))
            rows[m] = dict(pars=p, n=len(obs), n_clusters=n_cl,
                           pred_S=float(math.exp(np.mean(pred))),
                           obs_S=float(math.exp(np.mean(obs))),
                           mean_resid=mr, sigma_pred=math.sqrt(var),
                           sigma=mr / math.sqrt(var),
                           sigma_naive_wrong=float(
                               np.mean(resid / np.sqrt(est ** 2 + s_w ** 2
                                                       + s_c ** 2))
                               * math.sqrt(len(obs))))
        out["held"][held] = rows
        print(f"\n   HELD OUT = {held.upper()}   (trained on {train})")
        if held == "efeds":
            print(f"      {'model':6s} {'chi2':>10s} {'chi2/N':>8s}"
                  f"  key parameters")
            for m in MODELS:
                v = rows[m]
                kp = " ".join(f"{k}={vv:+.3f}" for k, vv in v["pars"].items()
                              if k in ("c", "alpha", "beta", "gamma"))
                print(f"      {m:6s} {v['chi2']:10.2f} "
                      f"{v['chi2_per_pt']:8.4f}  {kp}")
        else:
            print(f"      {'model':6s} {'pred S':>8s} {'obs S':>8s} "
                  f"{'ln resid':>9s} {'sigma_pred':>11s} {'sigma':>7s}"
                  f"  (naive, wrong)")
            for m in MODELS:
                v = rows[m]
                print(f"      {m:6s} {v['pred_S']:8.3f} {v['obs_S']:8.3f} "
                      f"{v['mean_resid']:+9.3f} {v['sigma_pred']:11.3f} "
                      f"{v['sigma']:+7.2f}  ({v['sigma_naive_wrong']:+.1f})")

    # the single cleanest statement available: eFEDS's OWN radial slope,
    # extrapolated, against the two cluster samples
    Jef = F.Joint(bd, surveys=("efeds",), fixed_scatter=fs, cache=J0._cache)
    r = Jef.fit("H_R", shape_grid=BETA)
    p = F.unpack("H_R", r)
    cur = np.array([cc[1] for cc in r["curve"]])
    xs = np.array([cc[0][0] for cc in r["curve"]])
    ok = xs[cur - cur.min() <= 1.0]
    lo_x, sl_x = 0.0, float(np.mean([q["lnx"] for q in bd.sl_rows]))
    print("\n" + "=" * 78)
    print("eFEDS ALONE, extrapolated: the cleanest statement in the lane")
    print("=" * 78)
    print(f"   eFEDS-only H_R:  beta = {p['beta']:+.4f} "
          f"[{ok.min():+.4f}, {ok.max():+.4f}],  c = {p['c']:+.4f}")
    ex = {}
    for nm, xv, obs in (("LoCuSS", lo_x,
                         float(np.mean([q["lnS"] for q in bd.lo_rows]))),
                        ("SL cores", sl_x,
                         float(np.mean([np.mean([q["lnS"] for q in bd.sl_rows
                                                 if q["cid"] == c_])
                                        for c_ in sorted(set(
                                            q["cid"] for q
                                            in bd.sl_rows))])))):
        pr = p["c"] + p["beta"] * xv
        band = [p["c"] + b * xv for b in (ok.min(), ok.max())]
        ex[nm] = dict(x=math.exp(xv), pred_S=math.exp(pr),
                      pred_band=[math.exp(min(band)), math.exp(max(band))],
                      obs_S=math.exp(obs), dln=obs - pr)
        print(f"   at r/R500 = {math.exp(xv):.3f}:  predicted S = "
              f"{math.exp(pr):.3f} [{math.exp(min(band)):.3f}, "
              f"{math.exp(max(band)):.3f}]   observed {math.exp(obs):.3f}"
              f"   ln difference {obs - pr:+.3f}")
    out["efeds_only_extrapolation"] = dict(beta=p["beta"],
                                           beta_ci=[float(ok.min()),
                                                    float(ok.max())],
                                           c=p["c"], targets=ex)
    with open(os.path.join(HERE, "transfer_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n   wrote transfer_results.json")


if __name__ == "__main__":
    main()
