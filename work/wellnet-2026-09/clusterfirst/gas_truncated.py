"""gas_truncated.py -- let the X-ray data set the outer gas slope, instead of a beta-model.

THE PROBLEM THIS SETTLES. The residual (observed lensing / RAR-predicted) runs
2.02 +- 0.26 at 1.0-1.6 Mpc and 0.08 +- 0.44 beyond 2.5 Mpc. A single beta-model
cannot be trusted over that span: rho ~ r^(-3 beta) with beta ~ 0.6 gives
rho ~ r^-1.8, so the enclosed mass M(r) ~ r^1.2 DIVERGES. Real cluster gas
steepens beyond R500. An un-truncated model therefore over-predicts g_bar at
large radius, over-predicts the lensing, and pushes the residual down -- which is
the direction of the observed decline.

So the decline has two candidate causes and they have to be separated:
    (a) gravity departs from the RAR as a function of radius
    (b) the gas model has the wrong outer shape

WHY THIS IS DECIDABLE HERE. The Chandra surface brightness runs to 12 arcmin,
which for these redshifts is 0.94 to 4.57 Mpc -- past R500 for most of the
sample. The steepening is therefore IN THE DATA and does not have to be assumed.
The test is whether adding an outer break improves the surface-brightness fit,
and by how much it moves the gas mass.

THE MODEL. Vikhlinin's outer term on a beta-model core:

    n_e(r) = n0 (1 + (r/rc)^2)^(-3 beta / 2) (1 + (r/rs)^3)^(-eps / 6)

which is the beta-model for r << rs and steepens by eps/2 in the log-slope
beyond it. Surface brightness is the line-of-sight integral of n_e^2, done
numerically -- the closed form the beta-model enjoys does not survive the extra
term. eps is bounded at 5 (Vikhlinin et al. 2006); above that the model can
mimic a truncation sharper than any observed cluster.

Fitting is on the SAME exposure-corrected profiles as `regas.py`, with the same
error floor, the same ACCEPT anchoring and the same rejection bounds, so the
difference between gas_extended_expcorr.json and gas_truncated.json is
attributable to the outer term alone.

    python gas_truncated.py
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
sys.path.insert(0, HERE)

import cosmo as C                                               # noqa: E402
import regas as R                                               # noqa: E402

OUT = os.path.join(HERE, "gas_truncated.json")
EPS_MAX = 5.0                                                   # Vikhlinin bound

# line-of-sight grid, arcmin: fine near the plane, long tail to catch the wings
_U = np.concatenate([np.linspace(0.0, 6.0, 600),
                     np.geomspace(6.02, 150.0, 300)])


def shape(r, thc, beta, ths, eps):
    """n_e(r) / n0, r in the same units as thc and ths."""
    core = (1.0 + (r / thc) ** 2) ** (-1.5 * beta)
    if eps <= 0 or not np.isfinite(ths):
        return core
    return core * (1.0 + (r / ths) ** 3) ** (-eps / 6.0)


def sb_model(th, S0, thc, beta, ths, eps, bkg):
    """Surface brightness = S0 * integral of (n_e/n0)^2 along the line of sight."""
    rr = np.sqrt(th[:, None] ** 2 + _U[None, :] ** 2)
    f = shape(rr, thc, beta, ths, eps)
    return S0 * np.trapezoid(f ** 2, _U, axis=1) + bkg


def fit_core(th, sb, err):
    """Plain beta-model: S0, thc, beta, bkg. Uses the SAME numeric projection as
    the truncated fit, so the two chi-squares are comparable."""
    def resid(p):
        return (sb_model(th, math.exp(p[0]), math.exp(p[1]), p[2],
                         float("inf"), 0.0, math.exp(p[3])) - sb) / err

    lo = [-60.0, math.log(0.02), 0.30, -60.0]
    hi = [60.0, math.log(20.0), 2.00, 10.0]
    best, cost = None, np.inf
    for thc0 in (0.3, 0.8, 2.0):
        for b0 in (0.55, 0.7, 0.9):
            p0 = [math.log(max(sb[0], 1e-12)), math.log(thc0), b0,
                  math.log(max(np.median(sb[-3:]), 1e-15))]
            try:
                r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=6000)
            except Exception:                                   # noqa: BLE001
                continue
            if r.cost < cost:
                best, cost = r, r.cost
    if best is None:
        return None
    return dict(S0=math.exp(best.x[0]), thc=math.exp(best.x[1]),
                beta=float(best.x[2]), bkg=math.exp(best.x[3]),
                ths=float("inf"), eps=0.0,
                chi2=float(2.0 * best.cost), npar=4)


def fit_break(th, sb, err, thc, beta):
    """CORE PINNED, outer break free: S0, bkg, ths, eps.

    Fitting all six at once does not work. The core shape and the break trade
    against each other, and the six-parameter fit ran to degenerate corners --
    Rc of 4-5 kpc with beta pegged at its 0.30 floor -- on six of eleven
    clusters. The core is not the unknown here: it is set by the inner annuli
    and independently by ACCEPT's deprojected n_e, which the beta-model matches
    to 0.04-0.19 dex. So the core is pinned at the beta-model value and only the
    outer term is fitted, which is the question being asked anyway.
    """
    def resid(p):
        return (sb_model(th, math.exp(p[0]), thc, beta,
                         math.exp(p[2]), p[3], math.exp(p[1])) - sb) / err

    lo = [-60.0, -60.0, math.log(0.5), 0.0]
    hi = [60.0, 10.0, math.log(80.0), EPS_MAX]
    best, cost = None, np.inf
    for ths0 in (2.0, 5.0, 12.0):
        for eps0 in (0.5, 2.0, 4.0):
            p0 = [math.log(max(sb[0], 1e-12)),
                  math.log(max(np.median(sb[-3:]), 1e-15)),
                  math.log(ths0), eps0]
            try:
                r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=6000)
            except Exception:                                   # noqa: BLE001
                continue
            if r.cost < cost:
                best, cost = r, r.cost
    if best is None:
        return None
    return dict(S0=math.exp(best.x[0]), thc=thc, beta=beta,
                bkg=math.exp(best.x[1]), ths=math.exp(best.x[2]),
                eps=float(best.x[3]), chi2=float(2.0 * best.cost), npar=6)


def main():
    match = json.load(io.open(os.path.join(HERE, "accept_overlap.json"),
                              encoding="utf-8"))
    acc = json.load(io.open(os.path.join(HERE, "accept_profiles.json"),
                            encoding="utf-8"))
    centres = json.load(io.open(os.path.join(WELLNET, "clusterxray",
                                             "overlap_centres.json"),
                                encoding="utf-8"))
    key_of = {v[6]: (k, v[0], v[1], v[2]) for k, v in centres.items()}

    out = {}
    print("does the X-ray surface brightness REQUIRE an outer break?")
    print("")
    print("  %-24s %-9s %-9s %-8s %-8s %-6s %-7s %s"
          % ("cluster", "chi2/dof", "dchi2 raw", "rescaled", "p(break)", "rs Mpc",
             "eps", "Mgas kept"))

    for m in sorted(match, key=lambda m: -m["shear"]):
        got = key_of.get(m["erass"])
        if not got:
            continue
        key, ra0, de0, z = got
        imgs = sorted(glob.glob(os.path.join(R.EXP,
                                             "%s_*_0.5-2.0_thresh.img" % key)))
        if not imgs:
            continue

        tot_c = tot_e = tot_n = ctr = None
        nobs = 0
        for ip in imgs:
            ep = ip.replace("_0.5-2.0_thresh.img", "_0.5-2.0_thresh.expmap")
            if not os.path.exists(ep):
                continue
            g = R.radial(ip, ep, ra0, de0)
            if g is None:
                continue
            c, cc, ee, nn = g
            if ctr is None:
                ctr, tot_c, tot_e, tot_n = c, cc.copy(), ee.copy(), nn.copy()
            elif len(c) == len(ctr):
                tot_c += cc
                tot_e += ee
                tot_n += nn
            nobs += 1
        if ctr is None or nobs == 0:
            continue

        per_px = tot_e / np.maximum(tot_n, 1)
        ok = per_px > R.MIN_EXP_FRAC * np.nanmax(per_px)
        if ok.sum() < 8:                                        # 6 parameters now
            continue
        sb = tot_c[ok] / tot_e[ok]
        err = np.sqrt(np.maximum(tot_c[ok], 1.0)) / tot_e[ok]
        th = ctr[ok]
        err = np.sqrt(err ** 2 + (R.SYS_FRAC * sb) ** 2)

        f0 = fit_core(th, sb, err)
        if f0 is None:
            continue
        f1 = fit_break(th, sb, err, f0["thc"], f0["beta"])
        if f1 is None:
            continue

        # RESCALE BEFORE JUDGING. The beta-model returns chi2 of 120-160 on ~20
        # annuli: chi2/dof near 8, so the error bars are too tight by about a
        # factor of 3. That is not a broken fit, it is real cluster morphology --
        # substructure, cool cores, asymmetry -- that no smooth radial model
        # contains, and a 3% flat-fielding floor does not cover. A raw dchi2 of
        # 6 read against that is meaningless. Errors are rescaled so the
        # beta-model gives chi2/dof = 1, and the improvement is judged after.
        dof0 = max(len(th) - f0["npar"], 1)
        scale = f0["chi2"] / dof0
        dchi2_raw = f0["chi2"] - f1["chi2"]
        dchi2 = dchi2_raw / scale if scale > 0 else float("nan")
        try:
            from scipy.stats import chi2 as _c2
            p_break = float(_c2.sf(dchi2, 2)) if np.isfinite(dchi2) else float("nan")
        except Exception:                                       # noqa: BLE001
            p_break = float("nan")

        DA = float(C.d_ang(z)) / C.MPC
        a2m = math.radians(1.0 / 60.0) * DA                     # arcmin -> Mpc
        rc = f1["thc"] * a2m
        rs = f1["ths"] * a2m
        beta = f1["beta"]

        # anchor n0 on ACCEPT with the NEW shape
        ap = np.array(acc[m["accept"]])
        r_acc, ne_acc = ap[:, 0], ap[:, 1]
        lo = max(r_acc.min(), R.TH_MIN * a2m)
        hi = min(r_acc.max(), R.TH_MAX * a2m)
        sel = (r_acc >= lo) & (r_acc <= hi)
        if sel.sum() < 3:
            continue
        sh = shape(r_acc[sel], rc, beta, rs, f1["eps"])
        n0 = float(np.exp(np.median(np.log(ne_acc[sel] / sh))))
        dex = float(np.std(np.log10(ne_acc[sel] / (n0 * sh))))

        # how much gas mass does the break remove inside the shear range?
        rr = np.geomspace(0.01, 4.6, 400)
        rc0 = f0["thc"] * a2m
        m_b = np.trapezoid(rr ** 2 * shape(rr, rc0, f0["beta"], np.inf, 0.0), rr)
        m_t = np.trapezoid(rr ** 2 * shape(rr, rc, beta, rs, f1["eps"]), rr)
        ratio = float(m_t / m_b) if m_b > 0 else float("nan")

        bad = []
        if not (0.010 <= rc <= 1.0):
            bad.append("Rc %.0f kpc" % (rc * 1000))
        if not (0.35 <= beta <= 1.3):
            bad.append("beta %.2f" % beta)
        if not (1e-4 <= n0 <= 1.0):
            bad.append("n0 %.1e" % n0)
        if dex > 0.25:
            bad.append("anchor %.2f dex" % dex)
        if bad:
            print("  %-24s REJECTED: %s" % (m["erass"], "; ".join(bad)))
            continue

        out[m["erass"]] = dict(
            accept=m["accept"], z=z, beta=beta, rc_mpc=rc, n0=n0,
            rs_mpc=rs, eps=f1["eps"],
            reach_mpc=R.TH_MAX * a2m,
            anchor_bins=int(sel.sum()), anchor_scatter_dex=round(dex, 3),
            accept_r_max=float(r_acc.max()), n_obs=nobs,
            exposure_corrected=True, truncated=True,
            chi2_beta=f0["chi2"], chi2_trunc=f1["chi2"],
            chi2_per_dof_beta=scale, dchi2_raw=dchi2_raw, dchi2_rescaled=dchi2,
            p_break=p_break, n_annuli_fitted=int(len(th)),
            mgas_ratio_vs_beta=ratio,
            vignetting="CIAO exposure maps (fluximage), not an analytic curve")
        print("  %-24s %-9.1f %-9.1f %-8.1f %-8.4f %-6.2f %-7.2f %.3f"
              % (m["erass"], scale, dchi2_raw, dchi2, p_break, rs, f1["eps"],
                 ratio))

    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    if out:
        d = np.array([v["dchi2_rescaled"] for v in out.values()])
        e = np.array([v["eps"] for v in out.values()])
        q = np.array([v["mgas_ratio_vs_beta"] for v in out.values()])
        print("")
        pv = np.array([v["p_break"] for v in out.values()])
        print("  median RESCALED dchi2 (2 extra par) : %.1f" % np.median(d))
        print("  clusters where the break is needed  : %d of %d at p<0.05"
              % (int((pv < 0.05).sum()), len(pv)))
        print("  median eps (0 = no break needed)    : %.2f" % np.median(e))
        print("  median gas mass kept inside 4.6 Mpc : %.3f" % np.median(q))
    print("")
    print("wrote gas_truncated.json for %d clusters" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
