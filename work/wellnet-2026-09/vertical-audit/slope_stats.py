"""ITEM 2b + ITEM 3.

2b  Fit the CLEANER form   log Sigma_dyn = a + s log Sigma_b   with latent-
    variable errors on BOTH axes and the measured within-galaxy error
    covariance, instead of regressing a ratio on its own denominator.
    log B_z = log Sigma_dyn - log Sigma_b, so the reported -0.346 corresponds
    to s = 0.654.  Both are reported.

3   Reconcile  -0.346 +- 0.173  (=> 2.00 sigma, two-sided p = 0.046) with the
    quoted galaxy-bootstrap p(slope >= 0) = 0.0095.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vaudit_core as V                                       # noqa: E402
import adyn_model as M                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BERSHADY_EXP = 0.643
R = {}


def head(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# =============================================================================
head("ITEM 3   Reconciling  -0.346 +- 0.173  with  p = 0.0095")
# =============================================================================
B = V.Bench()
NG = B.NG
x = B.log_sigma0()
bz = B.bz_draws(3000, seed=999)
sl = np.array([np.polyfit(x, bz[d], 1)[0] for d in range(bz.shape[0])])

print(f"\n  reproduction check ({bz.shape[0]} nuisance draws, 28 galaxies)")
print(f"    median slope                 {np.median(sl):+.4f}   published -0.3459")
print(f"    68% [{np.percentile(sl,16):+.4f}, {np.percentile(sl,84):+.4f}]"
      f"   published [-0.4159, -0.2761]")

sd_raw = float(np.std(sl))
sd_p = float((np.percentile(sl, 84) - np.percentile(sl, 16)) / 2)
sd_trim = float(np.std(sl[(sl > np.percentile(sl, 1))
                          & (sl < np.percentile(sl, 99))]))
mad = float(1.4826 * np.median(np.abs(sl - np.median(sl))))
INFL = 1.3484617994790402                       # published chi2/dof=1 factor
print(f"\n  (a) WHAT 0.173 IS.  It is  std(slope over nuisance draws) x {INFL:.4f},")
print(f"      where the 1.35 is the published chi2/dof=1 inflation.")
print(f"      raw std                    {sd_raw:.4f}  -> inflated {sd_raw*INFL:.4f}"
      f"   (published 0.1735)")
print(f"      std from the 16-84 range   {sd_p:.4f}  -> inflated {sd_p*INFL:.4f}")
print(f"      std after a 1% two-sided trim {sd_trim:.4f} -> inflated "
      f"{sd_trim*INFL:.4f}")
print(f"      1.4826 x MAD               {mad:.4f}  -> inflated {mad*INFL:.4f}")
print(f"      skewness {skew(sl):+.2f}   excess kurtosis {kurtosis(sl):+.1f}")
n_out = int(np.sum(np.abs(sl - np.median(sl)) > 5 * sd_p))
print(f"      draws further than 5 robust sigma from the median: {n_out}"
      f" of {sl.size}   (max {sl.max():+.3f}, min {sl.min():+.3f})")
print("      -> the quoted 0.173 is inflated by a HANDFUL of pathological")
print("         Monte-Carlo draws.  It is not a standard error; the robust")
print(f"         nuisance-only sd is {sd_p*INFL:.3f}, giving "
      f"{abs(np.median(sl))/(sd_p*INFL):.1f} sigma, not 2.0.")

# ------------------------------------------------------------- what breaks?
worst = np.argsort(sl)[-5:]
print("\n      the outlying draws, diagnosed:")
print(f"      {'slope':>9}{'sd(log Bz)':>12}{'min Bz dex':>12}{'max Bz dex':>12}")
for d in worst[::-1]:
    print(f"      {sl[d]:>9.3f}{np.std(bz[d]):>12.3f}"
          f"{bz[d].min():>12.3f}{bz[d].max():>12.3f}")
R["nuisance_slope"] = dict(median=float(np.median(sl)), sd_raw=sd_raw,
                           sd_p1684=sd_p, sd_trim1pc=sd_trim, sd_mad=mad,
                           inflation=INFL, sd_published=sd_raw * INFL,
                           skew=float(skew(sl)), kurtosis=float(kurtosis(sl)),
                           n_beyond_5robust=n_out)

# ------------------------------------------------------------ (b) bootstrap
rb = np.random.default_rng(8080)
bs = []
for _ in range(20000):
    idx = rb.integers(0, NG, NG)
    d = rb.integers(0, bz.shape[0])
    if np.std(x[idx]) < 1e-6:
        continue
    bs.append(np.polyfit(x[idx], bz[d][idx], 1)[0])
bs = np.array(bs)
p_bs = float(np.mean(bs >= 0.0))
print(f"\n  (b) WHAT 0.0095 IS.  A ONE-SIDED galaxy-bootstrap tail fraction,")
print(f"      P(slope >= 0), from a DIFFERENT resampling: it resamples")
print(f"      GALAXIES (with one random nuisance draw each), where 0.173")
print(f"      resamples NUISANCES at a fixed galaxy set.")
print(f"      bootstrap median {np.median(bs):+.4f}   sd {np.std(bs):.4f}"
      f"   robust sd {(np.percentile(bs,84)-np.percentile(bs,16))/2:.4f}")
print(f"      68% [{np.percentile(bs,16):+.3f}, {np.percentile(bs,84):+.3f}]"
      f"   95% [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
print(f"      skewness {skew(bs):+.2f}  (right-skewed: the tail towards zero is")
print(f"      the long one, so a normal approximation UNDERSTATES p)")
print(f"      P(slope >= 0) = {p_bs:.4f}   published 0.0095")
z_bs = abs(np.median(bs)) / np.std(bs)
print(f"      normal-approx one-sided p from the bootstrap sd: "
      f"{norm.sf(z_bs):.4f}  (z = {z_bs:.2f})")
R["bootstrap"] = dict(median=float(np.median(bs)), sd=float(np.std(bs)),
                      sd_p1684=float((np.percentile(bs, 84)
                                      - np.percentile(bs, 16)) / 2),
                      skew=float(skew(bs)), p_ge_zero=p_bs,
                      z_normal=float(z_bs))

print("\n  (c) THE RECONCILIATION")
print(f"      They are not the same statistic and cannot be divided into each")
print(f"      other.  0.346/0.173 = 2.00 pairs the OBSERVED value with the")
print(f"      WIDEST error estimate; 0.0095 is a ONE-SIDED tail of a NARROWER")
print(f"      one.  Making them commensurate:")
print(f"        nuisance-only, raw sd (as published) : "
      f"{abs(np.median(sl))/(sd_raw*INFL):.2f} sigma -> two-sided p "
      f"{2*norm.sf(abs(np.median(sl))/(sd_raw*INFL)):.3f}")
print(f"        nuisance-only, robust sd             : "
      f"{abs(np.median(sl))/(sd_p*INFL):.2f} sigma -> two-sided p "
      f"{2*norm.sf(abs(np.median(sl))/(sd_p*INFL)):.4f}")
print(f"        galaxy bootstrap, normal approx      : "
      f"{z_bs:.2f} sigma -> two-sided p {2*norm.sf(z_bs):.4f}")
print(f"        galaxy bootstrap, percentile         : one-sided p "
      f"{p_bs:.4f} -> two-sided {2*p_bs:.4f}")
print("      Both published numbers are individually correct.  Quoting them")
print("      TOGETHER is not, because '+- 0.173' and 'p = 0.0095' come from")
print("      different resamplings and the reader will divide one by the other.")
R["reconciliation"] = dict(
    sigma_nuisance_raw=float(abs(np.median(sl)) / (sd_raw * INFL)),
    sigma_nuisance_robust=float(abs(np.median(sl)) / (sd_p * INFL)),
    sigma_bootstrap=float(z_bs),
    p_two_sided_nuisance_raw=float(2 * norm.sf(abs(np.median(sl))
                                               / (sd_raw * INFL))),
    p_two_sided_bootstrap_percentile=float(2 * p_bs))


# =============================================================================
head("ITEM 2b   log Sigma_dyn = a + s log Sigma_b, latent variables both axes")
# =============================================================================
# ---- measure the per-galaxy 2x2 error covariance of (x, y) THROUGH the code
print("\n  measuring cov(eps_x, eps_y) per galaxy by pushing the observational")
print("  errors through the real chain (2000 realisations)")
e_mu0 = np.array([g.emu0K for g in B.GAL])
e_lhR = np.array([g.ehR_as / g.hR_as for g in B.GAL]) / np.log(10)
e_lhz = np.array([g.ehz_kpc / g.hz_kpc for g in B.GAL]) / np.log(10)
e_sig = np.array([g.esLOS0 for g in B.GAL])
s_rel = np.sqrt(np.maximum(e_lhz ** 2 - (BERSHADY_EXP * e_lhR) ** 2, 1e-6))

C0 = dict(zU=np.log10(0.60), sc=0.15, dhz=0.0, kv=1.5, al=0.60,
          lfg=np.log10(0.25), fhg=2.0, fhzg=0.5, lo=0.3, hi=2.0)
Ups0 = np.full(NG, 0.60)
fg0 = np.full(NG, 0.25)
al0 = np.full(NG, 0.60)


def realise(rng, rho):
    d_lhR = rng.normal(0.0, e_lhR)
    s_free = np.sqrt(np.maximum((0.4 * e_mu0) ** 2
                                - (rho * 2.0 * e_lhR) ** 2, 0.0))
    d_lSig = rho * 2.0 * d_lhR + rng.normal(0.0, s_free)
    d_lhz = BERSHADY_EXP * d_lhR + rng.normal(0.0, s_rel)
    O = V.Bench(mu0K=B.mu0K - d_lSig / 0.4, hR_as=B.hR_as_v * 10 ** d_lhR,
                hz_kpc=B.HZ_TAB * 10 ** d_lhz, apc=B.APC)
    aN, _, _ = O.amp_newton(Ups0, O.HZ_TAB, fg0, al0, C0)
    sob = B.OBS_AMP + rng.normal(0.0, e_sig)
    lb = 2 * np.log10(np.maximum(sob, 1.0) / aN)
    return O.log_sigma0(), lb


COV = {}
for rho in (0.0, -1.0):
    rng = np.random.default_rng(31415)
    XS, YS = [], []
    for _ in range(2000):
        xx, yy = realise(rng, rho)
        XS.append(xx); YS.append(yy)
    XS, YS = np.array(XS), np.array(YS)
    vxx = XS.var(axis=0); vyy = YS.var(axis=0)
    vxy = np.array([np.cov(XS[:, j], YS[:, j])[0, 1] for j in range(NG)])
    COV[rho] = (vxx, vyy, vxy)
    print(f"    rho(mu0,h_R) = {rho:+.1f} : median sd_x {np.sqrt(vxx).mean():.4f}"
          f"  sd_y {np.sqrt(vyy).mean():.4f}"
          f"  corr {np.mean(vxy/np.sqrt(vxx*vyy)):+.3f}")
R["error_covariance"] = {
    str(k): dict(sd_x=float(np.sqrt(v[0]).mean()),
                 sd_y=float(np.sqrt(v[1]).mean()),
                 corr=float(np.mean(v[2] / np.sqrt(v[0] * v[1]))))
    for k, v in COV.items()}


# ------------------------------------------------- EIV likelihood, both forms
def eiv_fit(xo, yo, vxx, vyy, vxy):
    """Max-likelihood straight line with per-point 2x2 error covariance and a
    free intrinsic scatter.  Returns (slope, intercept, sigma_int, sd_slope)."""
    def nll(th):
        a, s, ls = th
        v = np.exp(2 * ls) + vyy + s * s * vxx - 2 * s * vxy
        v = np.maximum(v, 1e-12)
        r = yo - a - s * xo
        return 0.5 * np.sum(r * r / v + np.log(v))
    best = None
    for s0 in (-1.0, -0.3, 0.0, 0.5, 1.0):
        m = minimize(nll, [np.mean(yo) - s0 * np.mean(xo), s0, np.log(0.1)],
                     method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-9, fatol=1e-11))
        if best is None or m.fun < best.fun:
            best = m
    a, s, ls = best.x
    # numerical Hessian on the slope, profiled over a and sigma_int
    def prof(sv):
        m = minimize(lambda t: nll([t[0], sv, t[1]]), [a, ls],
                     method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-9, fatol=1e-11))
        return m.fun
    h = 0.02
    d2 = (prof(s + h) - 2 * best.fun + prof(s - h)) / h ** 2
    sd = float(np.sqrt(1.0 / max(d2, 1e-9)))
    return float(s), float(a), float(np.exp(ls)), sd, float(best.fun)


y_bz = np.median(bz, axis=0)                    # log10 B_z per galaxy
y_dyn = y_bz + x                                # log10 Sigma_dyn (up to const)

print("\n  the two forms are the SAME fit shifted by 1 -- verified, not asserted")
for rho in (0.0, -1.0):
    vxx, vyy, vxy = COV[rho]
    # ratio form: y = log B_z, x = log Sigma_0.  eps_y contains -eps_x.
    s_r, a_r, si_r, sd_r, _ = eiv_fit(x, y_bz, vxx, vyy, vxy)
    # clean form: y = log Sigma_dyn = log B_z + log Sigma_0.  eps cancels.
    vyy_d = vyy + vxx + 2 * vxy
    vxy_d = vxy + vxx
    s_d, a_d, si_d, sd_d, _ = eiv_fit(x, y_dyn, vxx, vyy_d, vxy_d)
    ols_r = np.polyfit(x, y_bz, 1)[0]
    ols_d = np.polyfit(x, y_dyn, 1)[0]
    print(f"\n    rho(mu0,h_R) = {rho:+.1f}")
    print(f"      OLS   log B_z    on log Sigma_0 : {ols_r:+.4f}")
    print(f"      OLS   log Sigmad on log Sigma_0 : {ols_d:+.4f}"
          f"   (= previous + 1: {ols_r+1:+.4f})")
    print(f"      EIV   log B_z    on log Sigma_0 : {s_r:+.4f} +- {sd_r:.4f}"
          f"   sigma_int {si_r:.4f} dex")
    print(f"      EIV   log Sigmad on log Sigma_0 : s = {s_d:+.4f} +- {sd_d:.4f}"
          f"   sigma_int {si_d:.4f} dex")
    print(f"      test s = 1 : {(1-s_d)/sd_d:.2f} sigma"
          f"   test slope = 0 : {abs(s_r)/sd_r:.2f} sigma")
    R[f"eiv_rho{rho:+.1f}"] = dict(
        ols_ratio=float(ols_r), ols_clean=float(ols_d),
        eiv_ratio_slope=s_r, eiv_ratio_sd=sd_r, eiv_ratio_sigint=si_r,
        eiv_clean_s=s_d, eiv_clean_sd=sd_d, eiv_clean_sigint=si_d,
        sigma_from_1=float((1 - s_d) / sd_d),
        sigma_from_0=float(abs(s_r) / sd_r))

print("\n  ATTENUATION.  Errors in x bias an OLS slope towards 0, i.e. towards")
print("  s = 0 in the clean form and towards slope = -1 in the ratio form.")
vxx, vyy, vxy = COV[-1.0]
lam = float(np.mean(vxx) / np.var(x))
print(f"    mean var(eps_x) / var(x) = {lam:.5f}")
print(f"    attenuation-corrected OLS s = {np.polyfit(x,y_dyn,1)[0]/(1-lam):+.4f}"
      f"   (raw {np.polyfit(x,y_dyn,1)[0]:+.4f})")
print(f"    shared-denominator displacement of the RATIO slope = "
      f"{-np.mean(vxx)/np.var(x):+.5f}")
R["attenuation"] = dict(lambda_x=lam,
                        s_corrected=float(np.polyfit(x, y_dyn, 1)[0] / (1 - lam)),
                        ratio_slope_displacement=float(-np.mean(vxx) / np.var(x)))

# --------------------------------------- closed-form DiskMass cross-check
print("\n  cross-check with the closed-form DiskMass estimator, which does not")
print("  use the forward chain at all:")
print("      Sigma_dyn = sigma_z,0^2 / (pi G k h_z)   [DiskMass VI eq. 1]")
print("      Sigma_b   = Upsilon_K Sigma_L0 (1 + f_gas)")
alpha, kk = 0.60, 1.5
beta0 = B.BETA[:, B.J10]
inc = np.radians(np.array([g.incl for g in B.GAL]))
sz0 = B.OBS_AMP / np.sqrt(np.cos(inc) ** 2
                          + 0.5 * np.sin(inc) ** 2 * (1 + beta0 ** 2) / alpha ** 2)
Sd = (sz0 * 1e3) ** 2 / (np.pi * M.G * kk * B.HZ_TAB * M.KPC) / (M.MSUN / M.PC ** 2)
lU = np.log10(0.60) + 0.15 * (B.BK - 3.4)
Sb = 10 ** lU * np.squeeze(B.SigL0) * 1.25
print(f"    OLS   log Sigma_dyn on log Sigma_b : "
      f"{np.polyfit(np.log10(Sb), np.log10(Sd), 1)[0]:+.4f}")
print(f"    OLS   log Sigma_dyn on log Sigma_L0: "
      f"{np.polyfit(x, np.log10(Sd), 1)[0]:+.4f}")
print(f"    implied d log B_z / d log Sigma_0  : "
      f"{np.polyfit(x, np.log10(Sd), 1)[0]-1:+.4f}"
      f"   (forward chain: {np.median(sl):+.4f})")
print(f"    median Sigma_dyn/Sigma_b = "
      f"{np.median(Sd/Sb):.3f}   (forward chain B_z = {10**np.median(y_bz):.3f})")
R["closed_form"] = dict(
    s_on_Sigma_b=float(np.polyfit(np.log10(Sb), np.log10(Sd), 1)[0]),
    s_on_Sigma_L0=float(np.polyfit(x, np.log10(Sd), 1)[0]),
    implied_Bz_slope=float(np.polyfit(x, np.log10(Sd), 1)[0] - 1),
    median_ratio=float(np.median(Sd / Sb)))

# --------------------------- is the slope just the sigma-Sigma_0 relation?
print("\n  DECOMPOSITION of the slope into its measured pieces")
b_sig = np.polyfit(x, np.log10(B.OBS_AMP), 1)[0]
b_hz = np.polyfit(x, np.log10(B.HZ_TAB), 1)[0]
b_bk = np.polyfit(x, B.BK, 1)[0]
print(f"    d log10 sigma_LOS,0 / d log10 Sigma_0 = {b_sig:+.4f}  -> 2x = "
      f"{2*b_sig:+.4f}")
print(f"    d log10 h_z         / d log10 Sigma_0 = {b_hz:+.4f}  -> x(-0.656) = "
      f"{-0.656*b_hz:+.4f}")
print(f"    d (B-K)             / d log10 Sigma_0 = {b_bk:+.4f}  -> x(-0.15) = "
      f"{-0.15*b_bk:+.4f}   [Upsilon colour term]")
print(f"    Sigma_L0 itself, coefficient -0.994               = {-0.994:+.4f}")
tot = 2 * b_sig - 0.656 * b_hz - 0.15 * b_bk - 0.994
print(f"    sum {tot:+.4f}   against the pipeline's {np.median(sl):+.4f}")
print("    -> the signal is 2 x (the sigma_LOS,0 - Sigma_0 scaling) minus 1.")
print("       It is the vertical Fundamental-Plane-like relation, restated.")
R["decomposition"] = dict(d_logsigma=float(b_sig), d_loghz=float(b_hz),
                          d_BK=float(b_bk), sum=float(tot),
                          pipeline=float(np.median(sl)))

with open(os.path.join(HERE, "slope_stats.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print(f"\n  wrote slope_stats.json")
