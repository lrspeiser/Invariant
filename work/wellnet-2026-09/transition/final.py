"""PRIMARY analysis: the hierarchy and the frozen transfer, with the
strong-lens sample collapsed to one point per cluster.

WHY THE COLLAPSE IS THE PRIMARY, and it is a result rather than a convenience:

    S = 1/kappa_bar(theta)  and  ln(r/R500) = ln(theta D_l / R500)

share theta, with d ln x/d ln theta = 1 EXACTLY and d ln S/d ln theta ~ +1.
Any image system that is not actually on the cluster's tangential critical
curve is therefore pushed up and out together.  Measured, the strong-lens
sample's internal radial slope is +0.20 +- 0.03 -- POSITIVE, the opposite sign
to every other probe in this lane and to what the sample's own amplitude
requires.  Left in the fit, that artefact moves the joint beta by 65 in
-2 ln L while the 3365 eFEDS raw shear points move it by 12: 49 image systems
in 4 clusters were outvoting the entire weak-lensing dataset on the one
parameter the lane exists to measure.

Collapsing to the cluster mean keeps the Einstein-radius AMPLITUDE, which is
what the argument supports, and discards the within-cluster radial structure,
which it does not.  The uncollapsed fit is retained as the declared
alternative and both are reported.

Writes final_results.json.
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

BETA = np.round(np.arange(-1.00, 0.41, 0.05), 4)
GAMMA = np.round(np.arange(-0.80, 0.81, 0.05), 4)
KFREE = dict(H0=0, H_P=3, H_M=2, H_R=2, H_G=2, H_MR=3, H_T=3)
T0 = time.time()


def hierarchy(bd, J, N, tag=""):
    fits, rows = {}, []
    for m in ("H0", "H_P", "H_M", "H_R", "H_G", "H_MR"):
        grid = (BETA if m in ("H_R", "H_MR")
                else GAMMA if m == "H_G" else None)
        fits[m] = J.fit(m, shape_grid=grid) if grid is not None else J.fit(m)
    grid = [(A, lx) for A in np.arange(0.2, 4.01, 0.2)
            for lx in np.log(np.array([0.03, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0,
                                       1.8, 3.0, 6.0, 12.0]))]
    fits["H_T"] = J.fit("H_T", shape_grid=grid)
    for m, r in fits.items():
        k = KFREE[m]
        rows.append(dict(model=m, k=k, m2lnL=r["m2lnL"],
                         bic=r["m2lnL"] + k * math.log(N),
                         aic=r["m2lnL"] + 2 * k,
                         pars=F.unpack(m, r), shape=r.get("shape"),
                         desc=decl.HIERARCHY[m]["desc"]))
    bmin = min(r["bic"] for r in rows)
    for r in rows:
        r["dbic"] = r["bic"] - bmin
    rows.sort(key=lambda r: r["bic"])
    print(f"\n   HIERARCHY {tag}  (N = {N}, ln N = {math.log(N):.3f})")
    print(f"   {'model':6s} {'k':>2s} {'-2lnL':>10s} {'BIC':>10s} "
          f"{'dBIC':>8s}   parameters")
    for r in rows:
        s = " ".join(f"{k}={v:+.3f}" for k, v in r["pars"].items()
                     if k in ("c", "alpha", "beta", "gamma", "A", "lnxt"))
        print(f"   {r['model']:6s} {r['k']:2d} {r['m2lnL']:10.2f} "
              f"{r['bic']:10.2f} {r['dbic']:8.2f}   {s}")
    prof = {}
    for m in ("H_R", "H_MR", "H_G"):
        cur = fits[m]["curve"]
        x = np.array([c[0][0] for c in cur])
        y = np.array([c[1] for c in cur])
        y = y - y.min()
        ok = x[y <= 1.0]
        prof[m] = dict(x=x.tolist(), dm2lnL=y.tolist(),
                       ci68=[float(ok.min()), float(ok.max())])
    return rows, fits, prof


def transfer(bd, fs, tag=""):
    out = {}
    print(f"\n   FROZEN TRANSFER {tag}")
    for h in ("locuss", "sl"):
        train = tuple(s for s in K.SURVEYS if s != h)
        Jt = F.Joint(bd, surveys=train, fixed_scatter=fs)
        rows = {}
        for m in ("H0", "H_M", "H_R", "H_G", "H_MR"):
            grid = (BETA if m in ("H_R", "H_MR")
                    else GAMMA if m == "H_G" else None)
            r = Jt.fit(m, shape_grid=grid) if grid is not None else Jt.fit(m)
            p = F.unpack(m, r)
            src = bd.lo_rows if h == "locuss" else bd.sl_rows
            x = np.array([[q["lnM"], q["lnx"], q["lng"]] for q in src])
            obs = np.array([q["lnS"] for q in src])
            est = np.array([q["e_stat"] for q in src])
            pred = (p.get("c", 0.0) + p.get("alpha", 0.0) * x[:, 0]
                    + p.get("beta", 0.0) * x[:, 1]
                    + p.get("gamma", 0.0) * x[:, 2])
            resid = obs - pred
            prior = decl.OFFSET_PRIORS[h]["sd"]
            s_c = fs[0] if h == "locuss" else fs[2]
            var = prior ** 2 + (s_c ** 2 + float(np.mean(est ** 2))) / len(obs)
            rows[m] = dict(pars=p, n=len(obs),
                           pred_S=float(math.exp(pred.mean())),
                           obs_S=float(math.exp(obs.mean())),
                           mean_resid=float(resid.mean()),
                           sigma_pred=math.sqrt(var),
                           sigma=float(resid.mean() / math.sqrt(var)))
        out[h] = rows
        print(f"      held out {h.upper():7s}  {'model':6s} {'pred S':>8s} "
              f"{'obs S':>8s} {'ln resid':>9s} {'sigma':>7s}")
        for m, v in rows.items():
            print(f"                       {m:6s} {v['pred_S']:8.3f} "
                  f"{v['obs_S']:8.3f} {v['mean_resid']:+9.3f} "
                  f"{v['sigma']:+7.2f}")
    return out


def main():
    print("=" * 78)
    print("PRIMARY: strong lensing collapsed to one point per cluster")
    print("=" * 78)
    bd = K.Bundle(verbose=False, r500_mode="cat", sl_agg=True)
    print(f"   SL sample: {len(bd.sl_rows)} clusters "
          f"(from {len(bd.sl_rows_full)} image systems)")
    for r in bd.sl_rows:
        print(f"      {r['cid']:10s} {r['n_systems']:2d} systems  S = "
              f"{r['S']:.3f}  ln(r/R500) = {r['lnx']:+.3f}  "
              f"e = {r['e_stat']:.4f}  (within-cluster sd {r['lnS_sd']:.3f})")
    J0 = F.Joint(bd)
    sc = F.unpack("H_P", J0.fit("H_P"))
    fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
          sc["sigma_int_sl_cluster"])
    print(f"\n   scatters under H_P, frozen for every model: LoCuSS "
          f"{fs[0]:.4f}, SL within {fs[1]:.4f}, SL cluster {fs[2]:.4f}")
    J = F.Joint(bd, fixed_scatter=fs, cache=J0._cache)
    N = len(bd.F.gt) + len(bd.lo_rows) + len(bd.sl_rows)
    rows, fits, prof = hierarchy(bd, J, N, "(SL aggregated)")
    trans = transfer(bd, fs, "(SL aggregated)")

    # eFEDS alone, and its extrapolation
    Jef = F.Joint(bd, surveys=("efeds",), fixed_scatter=fs, cache=J._cache)
    r = Jef.fit("H_R", shape_grid=BETA)
    p = F.unpack("H_R", r)
    cur = np.array([c[1] for c in r["curve"]])
    xs = np.array([c[0][0] for c in r["curve"]])
    ok = xs[cur - cur.min() <= 1.0]
    ex = {}
    print(f"\n   eFEDS ALONE: beta = {p['beta']:+.4f} "
          f"[{ok.min():+.4f}, {ok.max():+.4f}], c = {p['c']:+.4f}")
    for nm, xv, obs, pri in (
            ("LoCuSS", 0.0, float(np.mean([q["lnS"] for q in bd.lo_rows])),
             decl.OFFSET_PRIORS["locuss"]["sd"]),
            ("SL cores", float(np.mean([q["lnx"] for q in bd.sl_rows])),
             float(np.mean([q["lnS"] for q in bd.sl_rows])),
             decl.OFFSET_PRIORS["sl"]["sd"])):
        pr = p["c"] + p["beta"] * xv
        band = sorted(p["c"] + b * xv for b in (ok.min(), ok.max()))
        n = len(bd.lo_rows) if nm == "LoCuSS" else len(bd.sl_rows)
        sc_ = fs[0] if nm == "LoCuSS" else fs[2]
        sig = math.sqrt(pri ** 2 + sc_ ** 2 / n
                        + (0.5 * (band[1] - band[0])) ** 2)
        ex[nm] = dict(x=math.exp(xv), pred_S=math.exp(pr),
                      pred_band=[math.exp(band[0]), math.exp(band[1])],
                      obs_S=math.exp(obs), dln=obs - pr, sigma_pred=sig,
                      sigma=(obs - pr) / sig)
        print(f"      at r/R500 = {math.exp(xv):.3f}: predicted S = "
              f"{math.exp(pr):.3f} [{math.exp(band[0]):.3f}, "
              f"{math.exp(band[1]):.3f}]   observed {math.exp(obs):.3f}"
              f"   {(obs - pr) / sig:+.2f} sigma")

    out = dict(sl_aggregated=True, N=N, fixed_scatter=list(fs),
               sl_rows=[{k: v for k, v in r.items()} for r in bd.sl_rows],
               hierarchy=rows, profiles=prof, transfer=trans,
               efeds_only=dict(beta=p["beta"],
                               beta_ci=[float(ok.min()), float(ok.max())],
                               c=p["c"], targets=ex),
               seconds=time.time() - T0)
    with open(os.path.join(HERE, "final_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n   wrote final_results.json in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
