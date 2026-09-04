"""ONE common observable space for the cluster lensing residual.

    S(M, r) = observed lensing response / RAR-with-no-slip predicted response

across eFEDS weak shear, LoCuSS massive clusters and Hubble Frontier Field
strong-lensing cores, in one forward framework, against a hierarchy declared in
decl.py before any residual was examined.

PART 2  design and leverage      -- what the data CAN separate, before fitting
PART 3  shared-quantity audit    -- construction expressions and the H0 null
PART 4  the hierarchy            -- fit, model comparison
PART 5  responsiveness           -- d(estimate)/d(injected) for every headline
PART 6  frozen transfer          -- fit two surveys, predict the third, once
PART 7  power                    -- what would be needed to decide
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build as B                                               # noqa: E402
import common as K                                              # noqa: E402
import decl                                                     # noqa: E402
import fitlib as F                                              # noqa: E402
import nulls as NU                                              # noqa: E402
import pipeline as P                                            # noqa: E402
import lead01 as L                                              # noqa: E402
import closure as C                                             # noqa: E402

MPC, MSUN = P.MPC, P.MSUN
RES: dict = {}
T0 = time.time()
RNG = np.random.default_rng(20260904)
BETA_GRID = np.round(np.arange(-1.00, 0.41, 0.05), 4)
GAMMA_GRID = np.round(np.arange(-0.60, 0.61, 0.05), 4)


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ==================================================================== PART 2
def part2_design(bd, J):
    hdr("PART 2.  DESIGN AND LEVERAGE -- measured BEFORE any residual is seen")
    out = {}
    rows = []
    for name, lnM, lnx, lng, n in (
        ("efeds", bd.ef_x["lnM"], bd.ef_x["lnx"], bd.ef_x["lng"],
         len(bd.ef_x["lnM"])),
        ("locuss", J.lo_x[:, 0], J.lo_x[:, 1], J.lo_x[:, 2], len(J.lo_x)),
        ("sl", J.sl_x[:, 0], J.sl_x[:, 1], J.sl_x[:, 2], len(J.sl_x)),
    ):
        rows.append(dict(survey=name, n=int(n),
                         lnM=[float(lnM.min()), float(np.median(lnM)),
                              float(lnM.max())],
                         lnx=[float(lnx.min()), float(np.median(lnx)),
                              float(lnx.max())],
                         lng=[float(lng.min()), float(np.median(lng)),
                              float(lng.max())]))
        print(f"   {name:7s} n={n:5d}  ln(M/M0) {lnM.min():+6.2f}"
              f" {np.median(lnM):+6.2f} {lnM.max():+6.2f}"
              f" | ln(r/R500) {lnx.min():+6.2f} {np.median(lnx):+6.2f}"
              f" {lnx.max():+6.2f}"
              f" | ln(g/a0) {lng.min():+7.2f} {np.median(lng):+6.2f}"
              f" {lng.max():+6.2f}")
    out["ranges"] = rows

    # pooled collinearity, weighted so each survey counts once per point
    X = np.vstack([
        np.column_stack([bd.ef_x["lnM"], bd.ef_x["lnx"], bd.ef_x["lng"]]),
        J.lo_x, J.sl_x])
    sur = np.array(["efeds"] * len(bd.ef_x["lnM"]) + ["locuss"] * len(J.lo_x)
                   + ["sl"] * len(J.sl_x))
    cc = np.corrcoef(X.T)
    print("\n   POOLED correlation of the three candidate axes:")
    for i, a in enumerate(("ln M", "ln r/R500", "ln g/a0")):
        print("      %-10s %s" % (a, "  ".join(f"{cc[i, j]:+.4f}"
                                               for j in range(3))))
    out["pooled_corr"] = cc.tolist()

    # is the acceleration axis a distinct story, or a combination of the other
    # two?  (Run AI.6 found 98.6% of potential depth was a function of (g, r).)
    A = np.column_stack([np.ones(len(X)), X[:, 0], X[:, 1]])
    b, *_ = np.linalg.lstsq(A, X[:, 2], rcond=None)
    r2 = 1.0 - np.var(X[:, 2] - A @ b) / np.var(X[:, 2])
    print(f"\n   R^2 of ln(g/a0) on [1, ln M, ln r/R500], pooled = {r2:.6f}"
          f"   residual sd {np.std(X[:, 2] - A @ b):.4f}")
    print("      -> the acceleration axis is NOT an independent story to the"
          " extent this is 1.")
    out["lng_on_M_and_r"] = dict(r2=float(r2),
                                 coef=[float(v) for v in b],
                                 resid_sd=float(np.std(X[:, 2] - A @ b)))

    # BETWEEN vs WITHIN survey leverage: the whole question
    print("\n   BETWEEN-survey vs WITHIN-survey spread of each axis"
          " (ln units, sd):")
    lev = {}
    for j, a in enumerate(("lnM", "lnx", "lng")):
        mus = np.array([X[sur == s, j].mean() for s in ("efeds", "locuss", "sl")])
        wit = np.array([X[sur == s, j].std() for s in ("efeds", "locuss", "sl")])
        lev[a] = dict(between_sd=float(mus.std()),
                      within_sd=[float(v) for v in wit],
                      survey_means=[float(v) for v in mus])
        print(f"      {a:5s} between {mus.std():6.3f} | within "
              f"{wit[0]:6.3f} {wit[1]:6.3f} {wit[2]:6.3f}"
              f" | survey means {mus[0]:+7.3f} {mus[1]:+7.3f} {mus[2]:+7.3f}")
    out["leverage"] = lev

    # the three survey means in the (M, r) plane -- are they collinear?
    Mm = np.array(lev["lnM"]["survey_means"])
    xm = np.array(lev["lnx"]["survey_means"])
    r_sv = float(np.corrcoef(Mm, xm)[0, 1])
    print(f"\n   The three SURVEY MEANS in (ln M, ln r/R500): "
          f"correlation = {r_sv:+.4f}")
    print("      With 3 points a correlation near -1 means the surveys lie on")
    print("      a LINE in the (mass, radius) plane, so between-survey")
    print("      information alone cannot separate mass from radius.  Only")
    print("      WITHIN-survey slopes can.")
    out["survey_mean_collinearity"] = r_sv

    # LoCuSS single-radius theorem, and the circularity of the alternatives
    print("\n   LoCuSS RADIUS COORDINATE -- three definitions, all disqualified"
          "\n   from carrying radial information:")
    lnx_cat = J.lo_x[:, 1]
    print(f"      R500 = r500(M_WL)  ->  ln(r/R500) is identically "
          f"{lnx_cat.min():.3f} .. {lnx_cat.max():.3f}: NO SPREAD, hence no"
          " leverage on beta.")
    lo_S = np.exp(J.lo_lnS)
    xdyn = np.array([p["r"] / c.R500 for c, p in zip(bd.lo, bd.lop)])
    cdyn = float(np.corrcoef(np.log(lo_S), np.log(xdyn))[0, 1])
    print(f"      R500 = R500_dyn    ->  corr(ln S, ln r/R500) = {cdyn:+.4f}."
          "  ALGEBRAIC: with")
    print("         M_dyn ~ r^m near the aperture, ln(r/R500_dyn) = ln S/(3-m)"
          " EXACTLY.")
    out["locuss_radius"] = dict(cat_spread=float(lnx_cat.std()),
                                corr_lnS_lnx_dyn=cdyn)
    RES["part2_design"] = out
    return out


# ==================================================================== PART 3
def part3_shared(bd, J, fs, offsets, n_mc=200):
    hdr("PART 3.  SHARED-QUANTITY AUDIT and the H0 NULL")
    out = {}
    print("""
   CONSTRUCTION EXPRESSIONS, written out before any correlation is believed.

   eFEDS   S enters as Sigma_s(r) on the 3-D mass; the datum is g_+ from
           DECADE galaxy shapes, weights and photo-z.  The MODEL side depends
           on the Bahar+2022 Vikhlinin parameters (n0, rs, eps, beta, alpha)
           and z.  M = M_gas(<R500_cat) and R500_cat depend on those SAME
           density parameters and on Bahar's R500.  -> the regressors share
           the density fit with the model, but NOT with the datum.  This is
           propagated fit noise, not an algebraic identity (the Run AI
           distinction).
   LoCuSS  S = M_WL / M_dyn(r500_WL).  M_WL appears in the numerator AND sets
           the aperture r500_WL, so it appears in the denominator too through
           M_dyn(r500_WL).  M = M_gas(<r500_WL) shares the ACCEPT density with
           the denominator.  -> heavily shared; simulated below.
   SL      S = 1/kappa_bar(theta).  theta is from HST image positions; the
           model side is ACCEPT gas + the declared stellar template.  M and
           r/R500 use ACCEPT and MCXC.  -> the datum (theta) shares nothing
           with the regressors; the model side shares ACCEPT gas with M.

   The null expectation is therefore NOT zero for alpha, and NOT zero for beta.
   It is measured.""")

    # ---- the linearised estimator, validated against the full fit
    S0, dS0 = J.project("H0", None)
    p0 = bd.F.gplus(S0, dS0, 1.0)
    db = 0.05
    Sb, dSb = J.project("H_R", np.array([db]))
    pb = bd.F.gplus(Sb, dSb, 1.0)
    Dbeta = (np.log(np.maximum(pb, 1e-300)) - np.log(np.maximum(p0, 1e-300))) / db
    print(f"\n   eFEDS beta regressor  D = d ln g_+/d beta at beta = 0:"
          f"  median {np.median(Dbeta):+.4f}, sd {Dbeta.std():.4f}"
          f"  (cf. ln(r/R500) median {np.median(bd.ef_x['lnx']):+.4f})")
    out["dbeta_regressor"] = dict(median=float(np.median(Dbeta)),
                                  sd=float(Dbeta.std()),
                                  corr_with_lnx=float(np.corrcoef(
                                      Dbeta, bd.ef_x["lnx"])[0, 1]))

    def linfit(gt, lo_lnS, sl_lnS, ef_lnM=None, lo_x=None, sl_x=None,
               lo_e=None, sl_e=None, p0=p0, Dbeta=Dbeta):
        """First-order joint WLS for (c, alpha, beta, o_ef, o_lo, o_sl).

        The REGRESSORS are arguments, not constants, because under the null
        they are rebuilt from the same redrawn inputs as the estimate.  That
        is the entire point of the exercise.
        """
        ef_lnM = bd.ef_x["lnM"] if ef_lnM is None else ef_lnM
        lo_x = J.lo_x if lo_x is None else lo_x
        sl_x = J.sl_x if sl_x is None else sl_x
        lo_e = J.lo_e if lo_e is None else lo_e
        sl_e = J.sl_e if sl_e is None else sl_e
        cols, y, w = [], [], []
        # eFEDS: g_+ ~ p0 (1 + c + alpha lnM + beta D)
        yy = (gt - p0) / np.where(np.abs(p0) < 1e-12, 1e-12, p0)
        ww = (p0 / bd.F.er) ** 2
        n = len(yy)
        cols.append(np.column_stack([np.ones(n), ef_lnM, Dbeta,
                                     np.ones(n), np.zeros(n), np.zeros(n)]))
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
        # priors on the three offsets
        Pm = np.zeros((3, 6))
        for i, k in enumerate(("efeds", "locuss", "sl")):
            Pm[i, 3 + i] = 1.0
        Wp = np.array([1.0 / decl.OFFSET_PRIORS[k]["sd"] ** 2
                       for k in ("efeds", "locuss", "sl")])
        A = X.T @ (W[:, None] * X) + Pm.T @ (Wp[:, None] * Pm)
        bvec = X.T @ (W * Y)
        return np.linalg.solve(A, bvec), np.linalg.inv(A)

    th_obs, Cov = linfit(bd.F.gt, J.lo_lnS, J.sl_lnS)
    print(f"\n   LINEARISED estimator on the real data:"
          f"  c={th_obs[0]:+.4f} alpha={th_obs[1]:+.4f} beta={th_obs[2]:+.4f}")
    out["linear_on_real"] = dict(c=float(th_obs[0]), alpha=float(th_obs[1]),
                                 beta=float(th_obs[2]),
                                 fisher_sd=[float(math.sqrt(Cov[i, i]))
                                            for i in range(3)])

    # ---- the null: truth is H0 (S = 1 everywhere), everything else redrawn
    print(f"""
   H0 NULL, {n_mc} realisations per scaling.  The TRUTH is S = 1 EXACTLY in
   all three surveys, and every REGRESSOR is rebuilt from the redrawn inputs
   rather than held fixed -- see nulls.py for what shares what.
      eFEDS   Bahar Vikhlinin parameters redrawn from their published
              marginal errors at three variance scalings (0.25/0.5/1.0),
              because Bahar publishes NO covariance and the parameters are
              strongly covariant; plus fresh DECADE shape noise.
      LoCuSS  M_WL redrawn about M_dyn(R500_dyn) -- its H0 value -- and the
              aperture r500_WL, ln M and ln S all recomputed from it, so the
              shared-numerator path is live.  ACCEPT n_e shells redrawn.
      SL      image radii redrawn about the frozen law's Einstein radius, so
              ln S and ln(r/R500) move together exactly as they do in the
              data.  ACCEPT n_e shells redrawn.""")
    o_lo, o_sl = offsets["locuss"], offsets["sl"]
    print(f"   H0' amplitudes taken from H_P: LoCuSS S = {math.exp(o_lo):.3f},"
          f"  SL S = {math.exp(o_sl):.3f}")
    recs = [dict(r) for r in bd.obs.sys]
    keys = ("n0sq", "rs", "eps", "beta", "alpha")
    ekeys = ("e_n0sq", "e_rs", "e_eps", "e_beta", "e_alpha")
    null = {}
    for scale in (0.25, 0.5, 1.0):
        al, be, cc = [], [], []
        for it in range(n_mc):
            rc2 = []
            for r in recs:
                d = dict(r)
                for k, ek in zip(keys, ekeys):
                    e = r[ek]
                    if np.isfinite(e) and e > 0:
                        d[k] = r[k] + scale * e * RNG.normal()
                d["n0sq"] = max(d["n0sq"], 1e-6)
                d["rs"] = max(d["rs"], 1e-4 * MPC)
                d["beta"] = max(d["beta"], 0.34)
                rc2.append(d)
            syss2 = [P.System(rc) for rc in rc2]
            r500_2 = np.array([c.extra["R500_cat"] for c in bd.ef])
            M2 = np.array([float(np.interp(rr, s.r, s.M_gas))
                           for s, rr in zip(syss2, r500_2)])
            lnM2 = np.log(np.maximum(M2, 1e30) / K.M0)[bd.F.sysi]
            S2, dS2 = K.project_slip(syss2, bd.obs, bd.ef_idx, lambda sm: 1.0)
            # H0 data: the truth is S = 1 for the PERTURBED baryon model
            p_true = bd.F.gplus(S2, dS2, 1.0)
            gt2 = p_true + RNG.normal(size=p_true.size) * bd.F.er
            lo2, loX, loE = NU.locuss_null(bd, RNG, fs, o_lo, scale)
            sl2, slX, slE = NU.sl_null(bd, RNG, fs, o_sl, scale)
            th, _ = linfit(gt2, lo2, sl2, ef_lnM=lnM2, lo_x=loX, sl_x=slX,
                           lo_e=loE, sl_e=slE)
            cc.append(th[0]); al.append(th[1]); be.append(th[2])
        al, be, cc = np.array(al), np.array(be), np.array(cc)
        null[str(scale)] = dict(
            alpha_mean=float(al.mean()), alpha_sd=float(al.std(ddof=1)),
            beta_mean=float(be.mean()), beta_sd=float(be.std(ddof=1)),
            c_mean=float(cc.mean()), c_sd=float(cc.std(ddof=1)))
        print(f"      scale {scale:4.2f}:  E[c|H0]={cc.mean():+.4f}"
              f" +-{cc.std(ddof=1):.4f}"
              f"   E[alpha|H0]={al.mean():+.5f} +-{al.std(ddof=1):.5f}"
              f"   E[beta|H0]={be.mean():+.5f} +-{be.std(ddof=1):.5f}")
    out["null"] = null
    n1 = null["1.0"]
    fis = out["linear_on_real"]["fisher_sd"]
    print("\n   FISHER sigma vs NULL sd -- never quote the first alone:")
    for i, nm in enumerate(("c", "alpha", "beta")):
        nsd = n1[f"{nm}_sd"]
        print(f"      {nm:6s} Fisher {fis[i]:.5f}   null sd {nsd:.5f}"
              f"   ratio {fis[i] / nsd:6.3f}")
    out["fisher_over_null"] = {nm: float(fis[i] / n1[f'{nm}_sd'])
                               for i, nm in enumerate(("c", "alpha", "beta"))}
    RES["part3_shared"] = out
    return out, (th_obs, Cov), linfit


# ==================================================================== PART 4
def part4_hierarchy(bd, J, fs):
    hdr("PART 4.  THE PRESPECIFIED HIERARCHY")
    N = len(bd.F.gt) + len(J.lo_lnS) + len(J.sl_lnS)
    res, out = {}, {}
    order = ["H0", "H_P", "H_M", "H_R", "G_placeholder", "H_MR"]
    fits = {}
    fits["H0"] = J.fit("H0")
    fits["H_P"] = J.fit("H_P")
    fits["H_M"] = J.fit("H_M")
    fits["H_R"] = J.fit("H_R", shape_grid=BETA_GRID)
    fits["H_G"] = J.fit("H_G", shape_grid=GAMMA_GRID)
    fits["H_MR"] = J.fit("H_MR", shape_grid=BETA_GRID)
    KFREE = dict(H0=0, H_P=3, H_M=2, H_R=2, H_G=2, H_MR=3, H_T=3)
    rows = []
    print(f"\n   N = {N} data points "
          f"({len(bd.F.gt)} shear + {len(J.lo_lnS)} LoCuSS "
          f"+ {len(J.sl_lnS)} SL).  ln N = {math.log(N):.3f}")
    print(f"\n   {'model':6s} {'k':>2s} {'-2lnL':>10s} {'AIC':>10s} "
          f"{'BIC':>10s} {'dBIC':>8s}   parameters")
    for m in ("H0", "H_P", "H_M", "H_R", "H_G", "H_MR"):
        r = fits[m]
        k = KFREE[m]
        aic = r["m2lnL"] + 2 * k
        bic = r["m2lnL"] + k * math.log(N)
        pars = F.unpack(m, r)
        rows.append(dict(model=m, k=k, m2lnL=r["m2lnL"], aic=aic, bic=bic,
                         pars=pars, desc=decl.HIERARCHY[m]["desc"],
                         shape=r.get("shape")))
    bmin = min(r["bic"] for r in rows)
    for r in rows:
        r["dbic"] = r["bic"] - bmin
        p = r["pars"]
        s = " ".join(f"{k}={v:+.4f}" for k, v in p.items()
                     if not k.startswith("sigma"))
        print(f"   {r['model']:6s} {r['k']:2d} {r['m2lnL']:10.2f} "
              f"{r['aic']:10.2f} {r['bic']:10.2f} {r['dbic']:8.2f}   {s}")
    out["models"] = rows
    out["N"] = N

    # the transition model, admitted ONLY if H_MR fails
    best_simple = min((r for r in rows if r["model"] != "H_P"),
                      key=lambda r: r["bic"])
    print(f"\n   Best non-pipeline model on BIC: {best_simple['model']}"
          f"  (dBIC {best_simple['dbic']:+.2f} from the overall best,"
          f" {min(r['model'] for r in rows if r['bic'] == bmin)})")
    hp = [r for r in rows if r["model"] == "H_P"][0]
    print(f"   H_P (three free amplitudes, NO physics) sits at dBIC "
          f"{hp['dbic']:+.2f}.")
    admit = hp["bic"] < best_simple["bic"] - 2.0
    print(f"\n   TRANSITION MODEL H_T: declared in advance, admitted only if"
          f" the\n   power-law hierarchy fails to describe the pooled data."
          f"  Admitted = {admit}")
    if admit:
        grid = [(A, lx) for A in np.arange(0.2, 4.01, 0.2)
                for lx in np.log(np.array([0.03, 0.05, 0.1, 0.2, 0.35, 0.6,
                                           1.0, 1.8, 3.0, 6.0, 12.0]))]
        rT = J.fit("H_T", shape_grid=grid)
        k = KFREE["H_T"]
        rowT = dict(model="H_T", k=k, m2lnL=rT["m2lnL"],
                    aic=rT["m2lnL"] + 2 * k,
                    bic=rT["m2lnL"] + k * math.log(N),
                    pars=F.unpack("H_T", rT), shape=rT["shape"],
                    desc=decl.HIERARCHY["H_T"]["desc"])
        rowT["dbic"] = rowT["bic"] - min(bmin, rowT["bic"])
        for r in rows:
            r["dbic"] = r["bic"] - min(bmin, rowT["bic"])
        rows.append(rowT)
        pp = " ".join(f"{k}={v:+.4f}" for k, v in rowT["pars"].items()
                      if not k.startswith("sigma"))
        print(f"   {'H_T':6s} {k:2d} {rowT['m2lnL']:10.2f} "
              f"{rowT['aic']:10.2f} {rowT['bic']:10.2f} "
              f"{rowT['dbic']:8.2f}   {pp}  "
              f"A={rowT['shape'][0]:.3f} x_t={math.exp(rowT['shape'][1]):.3f}")
    out["transition_admitted"] = bool(admit)

    # profile curves for beta and gamma
    for m in ("H_R", "H_MR", "H_G"):
        cur = fits[m]["curve"]
        v = np.array([c[0][0] for c in cur])
        y = np.array([c[1] for c in cur])
        y = y - y.min()
        lo = v[y <= 1.0]
        out[f"{m}_profile"] = dict(x=v.tolist(), dm2lnL=y.tolist(),
                                   ci68=[float(lo.min()), float(lo.max())]
                                   if lo.size else None)
    b = out["H_MR_profile"]["ci68"]
    g = out["H_G_profile"]["ci68"]
    print(f"\n   Profile 68% intervals (Delta(-2lnL) = 1):")
    print(f"      beta  (H_MR) = {fits['H_MR']['shape'][0]:+.3f} "
          f"[{b[0]:+.3f}, {b[1]:+.3f}]")
    print(f"      gamma (H_G)  = {fits['H_G']['shape'][0]:+.3f} "
          f"[{g[0]:+.3f}, {g[1]:+.3f}]")
    RES["part4_hierarchy"] = out
    return out, fits


# ==================================================================== PART 5
def part5_responsiveness(bd, J, fs, linfit, fits):
    hdr("PART 5.  RESPONSIVENESS  d(estimate)/d(injected)")
    print("""
   Injected with the FULL nonlinear fitter as well as the linearised one: the
   headline numbers come from the full fitter, so that is what must be shown
   to move.  Injection sizes are scaled to each axis's lever arm -- ln(M/M0)
   spans 9.3 in the pooled sample against 4.4 for ln(r/R500), so an equal
   injection in alpha is a much larger perturbation of the data than in beta.
   The injection replaces the model part and keeps the observed residual, so
   the noise realisation is unchanged.""")
    out = {}
    S0, dS0 = J.project("H0", None)
    p0 = bd.F.gplus(S0, dS0, 1.0)
    resid = bd.F.gt - p0
    gt_save = bd.F.gt.copy()
    lo_save = J.lo_lnS.copy()
    sl_save = J.sl_lnS.copy()
    for nm, key, inj in (("alpha", 0, [-0.10, -0.05, 0.0, 0.05, 0.10]),
                         ("beta", 1, [-0.30, -0.15, 0.0, 0.15, 0.30])):
        got, gotl = [], []
        for v in inj:
            if nm == "alpha":
                amp = np.exp(v * bd.ef_x["lnM"])
                gt2 = bd.F.gplus(S0, dS0, amp) + resid
            else:
                Sv, dSv = J.project("H_R", np.array([v]))
                gt2 = bd.F.gplus(Sv, dSv, 1.0) + resid
            lo2 = lo_save + v * J.lo_x[:, key]
            sl2 = sl_save + v * J.sl_x[:, key]
            th, _ = linfit(gt2, lo2, sl2)
            gotl.append(float(th[1 + key]))
            bd.F.gt = gt2
            J.lo_lnS = lo2
            J.sl_lnS = sl2
            if nm == "alpha":
                r = J.fit("H_M")
                got.append(float(F.unpack("H_M", r)["alpha"]))
            else:
                r = J.fit("H_MR", shape_grid=BETA_GRID)
                got.append(float(F.unpack("H_MR", r)["beta"]))
            bd.F.gt = gt_save.copy()
            J.lo_lnS = lo_save.copy()
            J.sl_lnS = sl_save.copy()
        got, gotl = np.array(got), np.array(gotl)
        slope = float(np.polyfit(inj, got - got[2], 1)[0])
        slopel = float(np.polyfit(inj, gotl - gotl[2], 1)[0])
        out[nm] = dict(injected=inj, recovered_full=got.tolist(),
                       recovered_linear=gotl.tolist(), slope=slope,
                       slope_linear=slopel,
                       spread=float(got.max() - got.min()))
        print("\n   %-6s injected  %s" % (nm, [round(v, 3) for v in inj]))
        print("          recovered %s   (full fitter)"
              % [round(v, 4) for v in got])
        print("          recovered %s   (linearised)"
              % [round(v, 4) for v in gotl])
        print("          d(est)/d(inj) = %.4f full, %.4f linearised;"
              "  spread %.4f over %.2f injected"
              % (slope, slopel, got.max() - got.min(), inj[-1] - inj[0]))
        if abs(slope) < 0.1:
            print("          *** CONSISTENT WITH ZERO: this statistic is BLIND"
                  " to its own parameter.  NO UPPER LIMIT IS SET.")
    bd.F.gt = gt_save
    J.lo_lnS = lo_save
    J.sl_lnS = sl_save
    RES["part5_responsiveness"] = out
    return out


# ==================================================================== PART 6
def part6_blind(bd, fs, fits):
    hdr("PART 6.  FROZEN TRANSFER -- fit two surveys, predict the third, ONCE")
    print(f"""
   Declared in decl.py before any fit: train = {decl.BLIND['train']},
   held out = {decl.BLIND['held']}.

   HONESTY NOTE, also declared in advance: the LoCuSS excess (E median 1.62)
   is already in the programme record and has been read by this lane's author.
   The freeze is PROCEDURAL -- the model space and the code were written and
   hashed first, and the held-out survey is touched once -- not epistemic.
   It is not a blind test in the strong sense and is not reported as one.

   Leave-one-survey-out is run for all three so the reader sees every
   extrapolation, not only the declared one.""")
    out = {}
    for held in ("locuss", "sl", "efeds"):
        train = tuple(s for s in K.SURVEYS if s != held)
        Jt = F.Joint(bd, surveys=train, fixed_scatter=fs)
        row = {}
        for m in ("H0", "H_M", "H_R", "H_MR", "H_G"):
            grid = (BETA_GRID if m in ("H_R", "H_MR")
                    else GAMMA_GRID if m == "H_G" else None)
            r = Jt.fit(m, shape_grid=grid) if grid is not None else Jt.fit(m)
            pars = F.unpack(m, r)
            # FREEZE, then predict the held-out survey.  The held-out survey's
            # own offset is NOT refitted: it is set to its prior mean (0).
            beta = pars.get("beta", 0.0)
            gamma = pars.get("gamma", 0.0)
            alpha = pars.get("alpha", 0.0)
            c = pars.get("c", 0.0)
            if held == "locuss":
                x = np.column_stack([[r_["lnM"] for r_ in bd.lo_rows],
                                     [r_["lnx"] for r_ in bd.lo_rows],
                                     [r_["lng"] for r_ in bd.lo_rows]])
                obs = np.array([r_["lnS"] for r_ in bd.lo_rows])
                e = np.sqrt(np.array([r_["e_stat"] for r_ in bd.lo_rows]) ** 2
                            + fs[0] ** 2)
            elif held == "sl":
                x = np.column_stack([[r_["lnM"] for r_ in bd.sl_rows],
                                     [r_["lnx"] for r_ in bd.sl_rows],
                                     [r_["lng"] for r_ in bd.sl_rows]])
                obs = np.array([r_["lnS"] for r_ in bd.sl_rows])
                e = np.sqrt(np.array([r_["e_stat"] for r_ in bd.sl_rows]) ** 2
                            + fs[1] ** 2 + fs[2] ** 2)
            else:
                x = np.column_stack([bd.ef_x["lnM"], bd.ef_x["lnx"],
                                     bd.ef_x["lng"]])
                obs = None
            pred = c + alpha * x[:, 0] + beta * x[:, 1] + gamma * x[:, 2]
            if held == "efeds":
                # for eFEDS the prediction is a shear chi2, not a mean lnS
                Jf = F.Joint(bd, surveys=("efeds",), fixed_scatter=fs)
                sh = r.get("shape")
                S, dS = Jf.project(m, sh)
                amp = np.exp(c + alpha * bd.ef_x["lnM"])
                ch = bd.F.chi2(S, dS, amp)
                row[m] = dict(pars=pars, chi2_held=float(ch),
                              chi2_per_pt=float(ch / len(bd.F.gt)),
                              mean_pred_lnS=float(np.mean(pred)))
            else:
                z = (obs - pred) / e
                row[m] = dict(pars=pars,
                              pred_mean_lnS=float(pred.mean()),
                              obs_mean_lnS=float(obs.mean()),
                              pred_mean_S=float(math.exp(pred.mean())),
                              obs_mean_S=float(math.exp(obs.mean())),
                              chi2_held=float(np.sum(z ** 2)),
                              n_held=int(len(obs)),
                              mean_pull=float(z.mean()),
                              sigma_of_mean=float(z.mean()
                                                  * math.sqrt(len(z))))
        out[held] = row
        print(f"\n   HELD OUT = {held.upper()}   (trained on {train})")
        if held == "efeds":
            print(f"      {'model':6s} {'chi2':>10s} {'chi2/N':>8s}")
            for m, v in row.items():
                print(f"      {m:6s} {v['chi2_held']:10.2f} "
                      f"{v['chi2_per_pt']:8.4f}")
        else:
            print(f"      {'model':6s} {'pred S':>8s} {'obs S':>8s} "
                  f"{'chi2':>9s} {'mean pull':>10s} {'sigma':>7s}")
            for m, v in row.items():
                print(f"      {m:6s} {v['pred_mean_S']:8.3f} "
                      f"{v['obs_mean_S']:8.3f} {v['chi2_held']:9.2f} "
                      f"{v['mean_pull']:+10.3f} {v['sigma_of_mean']:+7.2f}")
    RES["part6_blind"] = out
    return out


# ==================================================================== PART 7
def part7_power(bd, J, fs, null, linfit, fits):
    hdr("PART 7.  POWER")
    out = {}
    n1 = null["null"]["1.0"]
    sd_a, sd_b = n1["alpha_sd"], n1["beta_sd"]
    # what alpha/beta would each single-variable story need to carry the
    # observed between-survey offsets?
    lev = RES["part2_design"]["leverage"]
    Mm = lev["lnM"]["survey_means"]
    xm = lev["lnx"]["survey_means"]
    gm = lev["lng"]["survey_means"]
    hp = [r for r in RES["part4_hierarchy"]["models"]
          if r["model"] == "H_P"][0]["pars"]
    o = [hp["offset_efeds"], hp["offset_locuss"], hp["offset_sl"]]
    print("\n   Per-survey amplitudes from H_P (free, no physics):")
    for nm, v in zip(("efeds", "locuss", "sl"), o):
        print(f"      {nm:7s} ln S = {v:+.4f}   S = {math.exp(v):.3f}")
    req = {}
    for nm, m in (("alpha", Mm), ("beta", xm), ("gamma", gm)):
        # slope needed to join eFEDS to SL, and eFEDS to LoCuSS
        r1 = (o[2] - o[0]) / (m[2] - m[0]) if abs(m[2] - m[0]) > 1e-6 else None
        r2 = (o[1] - o[0]) / (m[1] - m[0]) if abs(m[1] - m[0]) > 1e-6 else None
        req[nm] = dict(efeds_to_sl=r1, efeds_to_locuss=r2)
        print(f"      {nm:6s} required eFEDS->SL {r1:+.4f}"
              f"   eFEDS->LoCuSS {r2:+.4f}")
    out["required_slopes"] = req

    # the within-survey slope actually measured, and its error
    lin = RES["part3_shared"]["linear_on_real"]
    print(f"\n   MEASURED joint slopes (linearised, null-referenced):")
    for i, nm in enumerate(("alpha", "beta")):
        est = lin[nm]
        bias = n1[f"{nm}_mean"]
        sd = n1[f"{nm}_sd"]
        print(f"      {nm:6s} = {est:+.4f}   null mean {bias:+.5f}"
              f"   null sd {sd:.5f}"
              f"   -> {(est - bias) / sd:+.2f} sigma from its own null")
        out[f"{nm}_vs_null"] = dict(est=est, null_mean=bias, null_sd=sd,
                                    sigma=float((est - bias) / sd))
    # THE DISCRIMINATOR: the slope eFEDS measures INTERNALLY against the
    # slope each story needs to join eFEDS to the other two surveys.
    print("\n   THE DISCRIMINATOR -- what eFEDS measures INTERNALLY (496"
          " systems, 3365\n   points, real spread on all three axes) against"
          " what each story NEEDS\n   in order to reach the other surveys:")
    Jef = F.Joint(bd, surveys=("efeds",), fixed_scatter=fs, cache=J._cache)
    intern = {}
    for nm, model, grid in (("alpha", "H_M", None),
                            ("beta", "H_R", BETA_GRID),
                            ("gamma", "H_G", GAMMA_GRID)):
        r = Jef.fit(model, shape_grid=grid) if grid is not None \
            else Jef.fit(model)
        pars = F.unpack(model, r)
        if grid is not None:
            xs = np.array([c[0][0] for c in r["curve"]])
            cur = np.array([c[1] for c in r["curve"]])
            v = float(pars[nm])
        else:
            v = float(pars["alpha"])
            xs = np.linspace(v - 0.25, v + 0.25, 51)
            base = np.asarray(r["x"], dtype=float)
            cur = []
            for a in xs:
                p2 = base.copy()
                p2[1] = a
                cur.append(Jef.m2lnL(model, None, p2))
            cur = np.array(cur)
        ok = xs[cur - cur.min() <= 1.0]
        ci = [float(ok.min()), float(ok.max())]
        need = req[nm]["efeds_to_sl"]
        sd = max((ci[1] - ci[0]) / 2.0, 1e-9)
        nsig = (need - v) / sd
        intern[nm] = dict(efeds_only=v, ci68=ci, required=float(need),
                          sigma_from_required=float(nsig))
        print("      %-6s eFEDS-only %+.4f [%+.4f,%+.4f]   needs %+.4f"
              "   -> %5.1f sigma away" % (nm, v, ci[0], ci[1], need,
                                          abs(nsig)))
    out["efeds_internal_vs_required"] = intern

    # detectable separation and required N
    print(f"\n   DETECTABLE SEPARATION at 5 sigma (null sd scaling as"
          f" 1/sqrt(N_survey-equivalent)):")
    for nm, sd, r in (("alpha", sd_a, req["alpha"]["efeds_to_sl"]),
                      ("beta", sd_b, req["beta"]["efeds_to_sl"])):
        print(f"      {nm:6s} 5 sigma = {5 * sd:+.4f};"
              f" the value required to explain the SL offset is {r:+.4f}"
              f"  ->  {abs(r / sd):.1f} sigma if real")
        out[f"{nm}_5sigma"] = float(5 * sd)
    RES["part7_power"] = out
    return out


# ======================================================================= main
def main(n_mc=200):
    hdr("TRANSITION LANE -- one common observable space for S(M, r)")
    print(f"\n   started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    dsha = sha(os.path.join(HERE, "decl.py"))
    print(f"   declaration decl.py sha256 = {dsha}")
    print(f"   frozen law: {decl.LAW_NAME}")
    print(f"   hierarchy: {', '.join(decl.HIERARCHY)}")
    print(f"   primary mass = {decl.PRIMARY_MASS}, "
          f"primary R500 = {decl.PRIMARY_R500}")
    RES["declaration"] = dict(
        sha256=dsha, version=decl.DECL_VERSION, law=decl.LAW_NAME,
        hierarchy={k: v["desc"] for k, v in decl.HIERARCHY.items()},
        offset_priors={k: dict(mean=v["mean"], sd=v["sd"], source=v["source"])
                       for k, v in decl.OFFSET_PRIORS.items()},
        blind=dict(train=list(decl.BLIND["train"]), held=decl.BLIND["held"],
                   note=decl.BLIND["note"]),
        question=decl.QUESTION,
        amendment=(
            "PRIMARY_R500 was declared as R500_dyn.  Before any residual was "
            "examined it was shown ALGEBRAICALLY that for a single-aperture "
            "dataset like LoCuSS, ln(r/R500_dyn) = ln S/(3-m) exactly, i.e. "
            "the radius axis is a deterministic function of the very ratio it "
            "is meant to explain (measured corr = +0.885).  The primary was "
            "therefore amended to the EXTERNAL CATALOGUE aperture R500_cat "
            "(Bahar+2022 / Okabe M_WL / MCXC), on a pre-data admissibility "
            "argument in the sense of Run AM.  R500_dyn is retained as the "
            "declared alternative and both are reported."))

    hdr("PART 1.  INGEST")
    bd = K.Bundle(verbose=True, r500_mode="cat")
    print()
    B.E.gate_mgas(bd.syss, bd.obs.sys)
    RES["part1_inputs"] = dict(
        manifest=B.input_manifest(),
        efeds=dict(n_systems=len(bd.ef), n_points=int(len(bd.F.gt)),
                   cuts=bd.ef_cuts, gate_mgas500=B.E.RES.get("gate_mgas500")),
        locuss=dict(n=len(bd.lo_rows), cuts=bd.lo_cuts,
                    ids=[r["cid"] for r in bd.lo_rows]),
        sl=dict(n_systems=len(bd.sl_rows), per_cluster=bd.sl_cuts,
                dropped=bd.sl_dropped, excluded=B.SL_EXCLUDED,
                clusters=sorted({r["cid"] for r in bd.sl_rows})))

    J0 = F.Joint(bd)
    rP = J0.fit("H_P")
    sc = F.unpack("H_P", rP)
    fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
          sc["sigma_int_sl_cluster"])
    print(f"\n   Intrinsic scatters, estimated ONCE under H_P (free per-survey"
          f" offsets,\n   so no between-survey structure can leak in) and"
          f" FROZEN for every model:")
    print(f"      sigma_int LoCuSS          {fs[0]:.4f}")
    print(f"      sigma_int SL within-cluster {fs[1]:.4f}")
    print(f"      sigma_int SL cluster-common {fs[2]:.4f}")
    RES["fixed_scatter"] = dict(locuss=fs[0], sl_within=fs[1],
                                sl_cluster=fs[2],
                                why="a free variance absorbs model misfit; "
                                    "with 4 SL clusters that degeneracy is "
                                    "severe (H0 drove it to 0.93)")
    J = F.Joint(bd, fixed_scatter=fs, cache=J0._cache)

    part2_design(bd, J)
    null, lin, linfit = part3_shared(
        bd, J, fs, dict(efeds=sc["offset_efeds"], locuss=sc["offset_locuss"],
                        sl=sc["offset_sl"]), n_mc=n_mc)
    out4, fits = part4_hierarchy(bd, J, fs)
    part5_responsiveness(bd, J, fs, linfit, fits)
    part6_blind(bd, fs, fits)
    part7_power(bd, J, fs, null, linfit, fits)

    RES["seconds"] = time.time() - T0
    RES["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(HERE, "transition_results.json"), "w") as f:
        json.dump(RES, f, indent=1, default=float)
    print(f"\n   wrote transition_results.json in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main(n_mc=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
