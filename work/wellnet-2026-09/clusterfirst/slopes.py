"""slopes.py -- give the RAR a free normalisation per cluster, then look for a radial residual.

WHY THIS TEST EXISTS. `rar_radius.py` finds that at fixed g_bar the boost the
lensing demands falls with radius, C = -9.63 +- 2.65, and every control run
against it failed to remove it -- except leave-one-out, where dropping
RX J1347.5-1145 leaves -1.9 sigma. Ten clusters is not enough. The sample is
capped at ten because the gas NORMALISATION came from ACCEPT, and only eleven
clusters have both ACCEPT and DECADE shear.

THE WAY PAST IT. The normalisation is not needed. Delta_Sigma_bar and g_bar are
both LINEAR in n0, so scaling n0 moves a cluster's residual level up or down
without changing the shape of its own radial trend -- and the per-cluster slope
significance, slope divided by its error, is exactly invariant under it. What is
needed is only the beta-model SHAPE, which comes straight from the Chandra
surface brightness. 41 further clusters already have Chandra events staged.

So each cluster gets its own FREE gas normalisation, fitted to make the RAR fit
that cluster as well as it possibly can. That is deliberately generous: it hands
the RAR one free parameter per object, which is exactly the freedom Milgrom
objected to in dark-matter halos and exactly what his law does not need. If a
radial residual survives even that, it is not something a normalisation can
absorb.

WHAT IS SPENT. Fitting n0 to the lensing spends the amplitude information: this
test can say nothing about the factor-of-two offset, only about the radial shape.
That is the intended trade.

THE STATISTIC. Per cluster, the weighted slope of (observed / RAR-predicted)
against log10 r, combined inverse-variance across clusters. Under the RAR with a
free normalisation the expectation is zero. The injection arm regenerates the
observations from the RAR at each cluster's own best-fit n0 and re-runs the whole
procedure, so whatever the fitting itself manufactures is measured rather than
assumed.

    python slopes.py
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
sys.path.insert(0, HERE)
sys.path.insert(0, WELLNET)

import cosmo as C                                               # noqa: E402
import firstprinciples as FP                                    # noqa: E402
import modifications as MOD                                     # noqa: E402
import regas as R                                               # noqa: E402

OUT = os.path.join(HERE, "slopes.json")
N_INJ = 200
SEED = 20260907
MIN_BINS = 4


def shape_from_sb(key, ra0, de0):
    """beta and rc (arcmin) from the exposure-corrected surface brightness."""
    imgs = sorted(glob.glob(os.path.join(R.EXP, "%s_*_0.5-2.0_thresh.img" % key)))
    if not imgs:
        return None
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
        return None
    per_px = tot_e / np.maximum(tot_n, 1)
    ok = per_px > R.MIN_EXP_FRAC * np.nanmax(per_px)
    if ok.sum() < 6:
        return None
    sb = tot_c[ok] / tot_e[ok]
    err = np.sqrt(np.maximum(tot_c[ok], 1.0)) / tot_e[ok]
    err = np.sqrt(err ** 2 + (R.SYS_FRAC * sb) ** 2)
    f = R.fit_beta(ctr[ok], sb, err)
    if f is None:
        return None
    if not (0.35 <= f["beta"] <= 1.3) or not (0.02 <= f["Rc_arcmin"] <= 20.0):
        return None
    return f, nobs


def wslope(x, v, e):
    w = 1.0 / e ** 2
    sw = w.sum()
    mx = (w * x).sum() / sw
    mv = (w * v).sum() / sw
    den = (w * (x - mx) ** 2).sum()
    if den <= 0:
        return float("nan"), float("nan")
    return float((w * (x - mx) * (v - mv)).sum() / den), float(math.sqrt(1.0 / den))


def best_n0(dsb1, gb1, dso, dse):
    """n0 minimising chi2 against the RAR. Both dsb and gb are linear in n0."""
    def chi2(ln):
        n = math.exp(ln)
        pred = n * dsb1 * MOD.boost_rar(n * gb1)
        return float(np.sum(((dso - pred) / dse) ** 2))
    r = minimize_scalar(chi2, bounds=(-8.0, 8.0), method="bounded")
    return math.exp(r.x)


def fit_gamma(dsb1, gb1, dso, dse, r_mpc):
    """Extra radial power the RAR needs: obs = n0 DS_bar F(g_bar) (r/Mpc)^gamma.

    The residual SLOPE in linear units is not scale-free. A cluster whose signal
    is near zero gets a tiny fitted n0, a large residual scale and therefore a
    huge slope -- RX J1347.5-1145 returned -86 per dex that way, on the same data
    that gives an ordinary answer here. gamma is dimensionless, comparable across
    clusters, and says directly what it means: the power of r by which the RAR
    prediction has to be multiplied.

    n0 and gamma are fitted jointly, so the normalisation cannot be blamed for
    gamma and vice versa.
    """
    x = np.log(r_mpc)

    def chi2(pars):
        n, gm = math.exp(pars[0]), pars[1]
        pred = n * dsb1 * MOD.boost_rar(n * gb1) * np.exp(gm * x)
        return float(np.sum(((dso - pred) / dse) ** 2))

    best, val = None, np.inf
    for ln0 in (-6.0, -4.0, -2.0, 0.0, 2.0):
        for g0 in (-2.0, 0.0, 2.0):
            r = minimize([ln0, g0], chi2)
            if r is not None and r[1] < val:
                best, val = r[0], r[1]
    if best is None:
        return None
    n0, gm = math.exp(best[0]), best[1]

    # error on gamma from the chi2 curvature, profiling n0 out at each step
    h = 0.05
    def prof(g):
        f = lambda ln: chi2([ln, g])
        rr = minimize_scalar(f, bounds=(-8.0, 8.0), method="bounded")
        return rr.fun
    c0, cp, cm = prof(gm), prof(gm + h), prof(gm - h)
    curv = (cp - 2.0 * c0 + cm) / h ** 2
    err = math.sqrt(2.0 / curv) if curv > 0 else float("nan")
    return n0, float(gm), float(err), float(c0)


def minimize(p0, f):
    """Nelder-Mead without importing scipy.optimize.minimize at module scope."""
    from scipy.optimize import minimize as _m
    try:
        r = _m(f, p0, method="Nelder-Mead",
               options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
    except Exception:                                           # noqa: BLE001
        return None
    return (list(r.x), float(r.fun)) if r.success or np.isfinite(r.fun) else None


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()

    centres = json.load(io.open(os.path.join(WELLNET, "clusterxray",
                                             "overlap_centres.json"),
                                encoding="utf-8"))
    prof = {}
    for line in io.open(os.path.join(WELLNET, "clustershear", "profiles.jsonl"),
                        encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            if d.get("profile"):
                prof[d["name"]] = d

    rng = np.random.default_rng(SEED)
    rows, used = [], []
    for key, v in sorted(centres.items()):
        ra0, de0, z, name = v[0], v[1], v[2], v[6]
        rec = prof.get(name)
        if rec is None:
            continue
        got = shape_from_sb(key, ra0, de0)
        if got is None:
            continue
        f, nobs = got
        DA = float(C.d_ang(z)) / C.MPC
        rc = math.radians(f["Rc_arcmin"] / 60.0) * DA * C.MPC
        beta = f["beta"]
        reach = math.radians(R.TH_MAX / 60.0) * DA * C.MPC

        def rho1(x):                                            # n0 = 1 cm^-3
            x = np.atleast_1d(np.asarray(x, dtype=float))
            return FP.MU_E * FP.M_P * (1.0 + (x / rc) ** 2) ** (-1.5 * beta) * 1e6

        obs = FP.observed(rec)
        Rv = np.array([o[0] for o in obs])
        ins = Rv <= reach
        if ins.sum() < MIN_BINS:
            continue
        Rv = Rv[ins]
        dso = np.array([o[1] for o in obs])[ins]
        dse = np.array([o[2] for o in obs])[ins]
        good = dse > 0
        Rv, dso, dse = Rv[good], dso[good], dse[good]
        if len(Rv) < MIN_BINS:
            continue

        rr = np.geomspace(1e-3 * C.MPC, reach * 1.02, 600)
        d1 = rho1(rr)
        m1 = np.concatenate(([0.0], np.cumsum(
            0.5 * (4 * math.pi * rr[:-1] ** 2 * d1[:-1]
                   + 4 * math.pi * rr[1:] ** 2 * d1[1:]) * np.diff(rr))))
        dsb1 = FP.delta_sigma_bar(rho1, Rv, reach)
        gb1 = FP.G * np.interp(Rv, rr, m1) / Rv ** 2
        if not np.all(dsb1 > 0):
            continue

        r_mpc = Rv / C.MPC
        got2 = fit_gamma(dsb1, gb1, dso, dse, r_mpc)
        if got2 is None:
            continue
        n0, s, se, _c = got2
        if not np.isfinite(s) or not np.isfinite(se) or se <= 0:
            continue

        # injection arm: same cluster, observations regenerated from the RAR at
        # this cluster's own best-fit normalisation, then the whole joint fit
        # re-run -- so whatever the fitting manufactures is measured, not assumed
        n0_rar = best_n0(dsb1, gb1, dso, dse)
        pred_rar = n0_rar * dsb1 * MOD.boost_rar(n0_rar * gb1)
        inj = []
        for _ in range(N_INJ):
            fake = pred_rar + rng.normal(0.0, dse)
            g2 = fit_gamma(dsb1, gb1, fake, dse, r_mpc)
            if g2 is not None and np.isfinite(g2[1]):
                inj.append(g2[1])
        if len(inj) < 20:
            continue
        inj_mean = float(np.mean(inj))
        inj_sd = float(np.std(inj))
        inj_neg = float(np.mean(np.array(inj) < 0.0))
        if not (inj_sd > 0):
            continue
        # p-value from the null's own tail, no Gaussian assumption
        inj_p = float(np.mean(np.array(inj) <= s))

        rows.append(dict(cluster=name, key=key, z=z, beta=beta,
                         rc_mpc=rc / C.MPC, n0_fitted=n0, n_bins=int(len(Rv)),
                         n_obs=nobs, gamma=s, gamma_err=se,
                         gamma_injected=inj_mean, gamma_injected_sd=inj_sd,
                         gamma_injected_frac_neg=inj_neg, p_one_sided=inj_p,
                         z_calibrated=(s - inj_mean) / inj_sd,
                         r_min=float(Rv.min() / C.MPC),
                         r_max=float(Rv.max() / C.MPC)))
        used.append(name)

    loader.assert_not_sealed(used, "free-normalisation radial slope test")

    print("extra radial power gamma the RAR needs:  obs = n0 DS_bar F(g_bar) r^gamma")
    print("with the gas normalisation fitted per cluster, jointly with gamma")
    print("")
    print("  %-24s %-5s %-6s %-11s %-9s %s"
          % ("cluster", "bins", "beta", "r range Mpc", "gamma", "sigma"))
    for d in sorted(rows, key=lambda d: d["gamma"]):
        print("  %-24s %-5d %-6.2f %-11s %+7.2f    %+.1f"
              % (d["cluster"], d["n_bins"], d["beta"],
                 "%.2f-%.2f" % (d["r_min"], d["r_max"]),
                 d["gamma"], d["gamma"] / d["gamma_err"]))

    s = np.array([d["gamma"] for d in rows])
    inj = np.array([d["gamma_injected"] for d in rows])
    isd = np.array([d["gamma_injected_sd"] for d in rows])
    ineg = np.array([d["gamma_injected_frac_neg"] for d in rows])

    # CALIBRATED against the null, per cluster. The curvature error is not the
    # spread: on RAR-generated data the joint (n0, gamma) fit returns a biased,
    # skewed distribution, so the injection supplies both the centre and the
    # width. z_i = (gamma_i - mu_i) / sigma_i, combined as sum(z)/sqrt(n).
    z = (s - inj) / isd
    zc = float(z.sum() / math.sqrt(len(z)))

    # inverse-variance combination using the INJECTION width as the error
    w = 1.0 / isd ** 2
    comb = float((w * s).sum() / w.sum())
    cerr = float(math.sqrt(1.0 / w.sum()))
    comb_inj = float((w * inj).sum() / w.sum())
    corrected = comb - comb_inj

    print("")
    n = len(s)
    n_neg = int((s < 0).sum())
    exp_neg = float(ineg.mean())                                # null's own rate
    print("  clusters                       %d" % n)
    print("  observed gamma (inj-weighted)  %+.2f" % comb)
    print("  RAR-generated data gives       %+.2f   <- the fit is BIASED, so this"
          % comb_inj)
    print("                                        is the null, not zero")
    print("  difference                     %+.2f +- %.2f" % (corrected, cerr))
    print("")
    print("  CALIBRATED per cluster against its own null distribution:")
    print("     combined z = %+.2f sigma" % zc)
    print("")
    print("  clusters with gamma below their own null median: %d of %d"
          % (int((z < 0).sum()), n))
    print("  clusters with negative gamma   %d of %d  (null expects %.1f)"
          % (n_neg, n, exp_neg * n))

    # sign test against the NULL's negative rate, not against 0.5
    from math import comb as _C
    p_sign = float(sum(_C(n, k) * exp_neg ** k * (1 - exp_neg) ** (n - k)
                       for k in range(n_neg, n + 1)))
    print("  sign test vs the null rate     p = %.4f" % p_sign)

    out = dict(lane="clusterfirst", stage="free-normalisation-gamma",
               n_clusters=len(rows), combined_gamma=comb, combined_err=cerr,
               manufactured=comb_inj, corrected=corrected,
               z_calibrated=zc, n_negative=n_neg, null_frac_negative=exp_neg,
               p_sign=p_sign, n_injections=N_INJ, clusters=rows,
               note=("each cluster gets a free gas normalisation fitted to "
                     "favour the RAR, so the amplitude information is spent and "
                     "only the radial shape is tested; ACCEPT is not used"),
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote slopes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
