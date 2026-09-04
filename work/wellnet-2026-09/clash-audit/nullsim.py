"""
JOB 2.3 -- the forward synthetic null.

Build clusters in which the excess has NO true dependence on radius (scaled or
physical), push them through the ACTUAL CLASH inference chain, and see what comes
out the other end.

THE CHAIN, AND WHERE THE TAUTOLOGY ENTERS
-----------------------------------------
truth       :  g_bar(r) from the published g_bar profile of the real cluster
               g_tot(r) = g_bar(r) nu_RAR(g_bar/a0) 10^(y_i + s log10(r/R500true))
                          s = 0 for the null; s != 0 for the responsiveness scan
               M(<r) = g_tot r^2 / G ,  rho(r) = M'(r)/(4 pi r^2)
projection  :  Sigma(R) = 2 int rho(sqrt(R^2+l^2)) dl        (Abel, exact)
measurement :  Sigma_obs = Sigma_true * (1 + noise)
inference   :  chi^2 fit of a SPHERICAL NFW Sigma(R|M200,c200) over R <= 2.9 Mpc
               -- Umetsu+2016's model and fit range
publication :  g_tot,pub(r) = G M_NFW(<r|M200,c200)/r^2      <- Tian's numerator
               M500c        = M_NFW(<R500)                   <- Umetsu Table 3
               R500         = [3 M500c/(4 pi 500 rho_c)]^(1/3)
analysis    :  excess vs log10(r / R500)

Two things can manufacture a radial trend here and NEITHER is physics:

  (i)  TEMPLATE BIAS.  The truth is not an NFW.  A 2-parameter NFW fitted to it
       misses in a radius-dependent way, so the published g_tot(r) carries a
       radial error even with no noise at all.  This is the CLASH analogue of
       Run AT's AT.5 finding that a flat truth returns slope -0.139 on X-COP.

  (ii) THE SHARED FIT.  R500 is a functional of the SAME (M200,c200).  An upward
       fluctuation raises g_tot at every r AND raises R500, lowering r/R500.
       Sign-definite negative.  Run AT's X-COP cancellation lemma removed this
       channel; CLASH has no such lemma (run_cancellation.py: the numerator moves
       2.02 dex per dex of R500, against 1.6e-13 for X-COP).

Declared before any null residual was looked at (2026-09-04):
  S1  pooled slope   d(excess)/d log10(r/R500)
  S2  Spearman(log10(r/R500), excess)
  S3  within-cluster fixed-effects slope
  S4  between-cluster Pearson(per-cluster mean excess, log10 R500)
"""
from __future__ import annotations
import math

import numpy as np

import ingest as I
import stats as S

KPC, MPC, MSUN, G, A0 = I.KPC, I.MPC, I.MSUN, I.G, I.A0
FIT_RMAX = 2.9 * MPC          # Umetsu+2016: R <= 2 Mpc/h = 2.9 Mpc h70^-1
FIT_RMIN = 0.03 * MPC

# BUG FOUND BY tests.py CHECK 10.  The first implementation truncated the truth
# density at 5 Mpc.  The Abel integral then loses 6.5% of Sigma at R = 1.5 Mpc
# and 21% at the outer fit radius of 2.9 Mpc, biasing the fitted NFW -- a
# manufactured radial effect in the null itself.  Converged to 2e-5 only beyond
# ~100 Mpc.  TRUNC is now 50 Mpc and the truth is continued outward as the
# PUBLISHED NFW (below), so the projection converges.
TRUNC = 50.0 * MPC

# SECOND DEFECT, found by comparing the null's R500 population with the real one:
# with the constant-excess law imposed all the way to 1.5 Mpc the null's R500 came
# out 1.2-1.7x too large and corr(excess, log R500) was +0.74 against +0.14 in the
# data.  A constant-excess cluster simply has far more mass at ~Mpc than the
# published NFW does -- but nothing in the CLASH table measures that region: the
# outermost datum is 600 kpc and R500 is ~2.3x further out.  The truth is
# therefore made to agree with the PUBLISHED NFW outside R_BREAK (continuous in
# M and in dlnM/dlnr through a logistic blend) and to carry the flat excess only
# where there are data.  The null then reproduces the real R500 population and
# isolates exactly the question asked: does a flat excess INSIDE the measured
# range come back out sloped?
R_BREAK = 600.0 * KPC         # the outermost measured radius; scanned 0.6/1.0/1.5 Mpc
BLEND = 0.15                  # width of the logistic blend in ln r


# --------------------------------------------------------------- NFW in 2D
def nfw_sigma(R, M200, c200, z):
    """Wright & Brainerd (1999) projected NFW surface density [kg/m^2]."""
    r200 = (3 * M200 * MSUN / (4 * math.pi * 200 * I.rhoc(z))) ** (1 / 3)
    rs = r200 / c200
    mu = math.log(1 + c200) - c200 / (1 + c200)
    rho_s = M200 * MSUN / (4 * math.pi * rs ** 3 * mu)
    x = np.asarray(R, float) / rs
    f = np.empty_like(x)
    lo = x < 1 - 1e-8
    hi = x > 1 + 1e-8
    eq = ~(lo | hi)
    xl = x[lo]
    f[lo] = (1 - 2 / np.sqrt(1 - xl ** 2) *
             np.arctanh(np.sqrt((1 - xl) / (1 + xl)))) / (xl ** 2 - 1)
    xh = x[hi]
    f[hi] = (1 - 2 / np.sqrt(xh ** 2 - 1) *
             np.arctan(np.sqrt((xh - 1) / (xh + 1)))) / (xh ** 2 - 1)
    f[eq] = 1.0 / 3.0
    return 2 * rho_s * rs * f


# ------------------------------------------------------ the truth generator
class Truth:
    """A cluster whose excess over the RAR has a PRESCRIBED radial dependence.

    g_bar(r) is taken from the real cluster: a quadratic in log-log through its
    published log g_bar points, continued as a power law outside the measured
    range (the continuation slope is the fitted slope at the end point).
    """

    def __init__(self, name, r_obs, lgb_obs, z, y0, s=0.0, R500_ref=None,
                 r_break=None, outer=None):
        self.name, self.z, self.y0, self.s = name, z, y0, s
        lr = np.log10(np.asarray(r_obs, float) / KPC)
        deg = 2 if len(lr) >= 4 else 1
        self.pc = np.polyfit(lr, np.asarray(lgb_obs, float), deg)
        self.lr_lo, self.lr_hi = lr.min(), lr.max()
        self.R500_ref = R500_ref
        self.outer = outer            # (M200_pub, c200_pub): the published NFW
        self.r_break = R_BREAK if r_break is None else r_break
        self._r = np.geomspace(1e-3 * MPC, TRUNC, 6000)
        self._rho = self._rho_of(self._r)
        # M(<r) by cumulative integration of the (possibly steepened) density
        # BUG FOUND BY tests.py CHECK 12.  The first implementation wrote the
        # inner-sphere mass into M[0] AFTER the cumulative integral instead of
        # adding it as an offset, so M[0] exceeded M[1] by ~180x and M(<r) was
        # not monotone.  Fixed by adding it to every element.
        w = 4 * math.pi * self._r ** 2 * self._rho
        inner = w[0] * self._r[0] / 3.0
        self._M = inner + np.concatenate([[0.0], np.cumsum(
            0.5 * (w[1:] + w[:-1]) * np.diff(self._r))])
        assert np.all(np.diff(self._M) > 0), f"{name}: M(<r) not monotone"
        assert np.all(self._rho >= 0), f"{name}: negative density"

    def gbar(self, r):
        lr = np.log10(np.asarray(r, float) / KPC)
        p = self.pc
        d = np.polyder(p)
        out = np.polyval(p, np.clip(lr, self.lr_lo, self.lr_hi))
        # power-law continuation outside the measured range
        below = lr < self.lr_lo
        above = lr > self.lr_hi
        if np.any(below):
            out = np.where(below,
                           np.polyval(p, self.lr_lo)
                           + np.polyval(d, self.lr_lo) * (lr - self.lr_lo), out)
        if np.any(above):
            out = np.where(above,
                           np.polyval(p, self.lr_hi)
                           + np.polyval(d, self.lr_hi) * (lr - self.lr_hi), out)
        return 10 ** out

    def excess_true(self, r):
        if self.s == 0.0:
            return np.full_like(np.asarray(r, float), self.y0)
        return self.y0 + self.s * np.log10(np.asarray(r, float) / self.R500_ref)

    def mass_law(self, r):
        """M(<r) implied by the prescribed excess -- valid inside r_break."""
        gb = self.gbar(r)
        gt = gb * I.nu_rar(gb / A0) * 10 ** self.excess_true(r)
        return gt * np.asarray(r, float) ** 2 / G

    def _rho_of(self, r):
        """the constant-excess law inside r_break, blended smoothly onto the
        PUBLISHED NFW outside it (rescaled to be continuous at the break)."""
        r = np.asarray(r, float)
        lr = np.log(r)
        lM = np.log(self.mass_law(r))
        if self.outer is not None:
            M200, c200 = self.outer
            lMn = np.log(np.asarray(I.nfw_mass(r, M200, c200, self.z), float))
            k = (float(np.interp(math.log(self.r_break), lr, lM))
                 - float(np.interp(math.log(self.r_break), lr, lMn)))
            w = 1.0 / (1.0 + np.exp(-(lr - math.log(self.r_break)) / BLEND))
            lM = (1 - w) * lM + w * (lMn + k)
        M = np.exp(lM)
        dlnM = np.gradient(lM, lr)
        return M * dlnM / (4 * math.pi * r ** 3)

    def rho(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        out = np.exp(np.interp(np.log(r), np.log(self._r),
                               np.log(np.maximum(self._rho, 1e-40))))
        return np.where(r > TRUNC, 0.0, out)

    def sigma(self, R, nu=200):
        """Abel projection, r = R cosh u -> no integrable singularity.
        Converged to 2e-5 against the analytic NFW Sigma (tests.py check 10)."""
        R = np.atleast_1d(np.asarray(R, float))
        umax = np.arccosh(np.maximum(TRUNC / R, 1.0 + 1e-12))
        u = np.linspace(0, 1, nu)[None, :] * umax[:, None]
        r = R[:, None] * np.cosh(u)
        integ = self.rho(r.ravel()).reshape(r.shape) * R[:, None] * np.cosh(u)
        return 2 * np.trapezoid(integ, u, axis=1)

    def M500_true(self):
        A = (4 / 3) * math.pi * 500 * I.rhoc(self.z)
        f = self._M - A * self._r ** 3
        i = int(np.argmax(f < 0))
        assert i > 0
        return float(np.exp(np.interp(0.0, [f[i], f[i - 1]],
                                      [math.log(self._r[i]),
                                       math.log(self._r[i - 1])])))


# ------------------------------------------------------------ the inference
R_FIT = np.geomspace(FIT_RMIN, FIT_RMAX, 16)


def fit_nfw(Sig, z, sig_err, guess=(1.5e15, 4.0)):
    """chi^2 fit of a spherical NFW Sigma to the (noisy) projected profile.
    Grid + local refinement -- robust, no optimiser dependency."""
    lM = np.linspace(math.log10(2e14), math.log10(8e15), 60)
    lc = np.linspace(math.log10(1.0), math.log10(12.0), 60)
    best = (None, np.inf)
    for _ in range(3):
        for a in lM:
            mods = np.array([nfw_sigma(R_FIT, 10 ** a, 10 ** b, z) for b in lc])
            chi = np.sum(((mods - Sig[None, :]) / sig_err[None, :]) ** 2, axis=1)
            j = int(np.argmin(chi))
            if chi[j] < best[1]:
                best = ((10 ** a, 10 ** lc[j]), float(chi[j]))
        M, c = best[0]
        lM = np.linspace(math.log10(M) - 0.08, math.log10(M) + 0.08, 40)
        lc = np.linspace(math.log10(c) - 0.10, math.log10(c) + 0.10, 40)
    return best[0]


def publish(M200, c200, z, r_points):
    """what the papers publish: g_tot at the tabulated radii, and R500."""
    gt = G * I.nfw_mass(r_points, M200, c200, z) * MSUN / np.asarray(r_points) ** 2
    R500 = I.r_delta(M200, c200, z, 500.0)
    return gt, R500


# --------------------------------------------------------------- one realisation
def realise(truths, r_by_cluster, frac_err, rng, noise=True):
    """returns (names, r, gbar, gtot_published, R500_published)."""
    nm, rr, gb, go, R5 = [], [], [], [], []
    for t in truths:
        r_pts = r_by_cluster[t.name]
        Sig = t.sigma(R_FIT)
        err = frac_err * Sig
        obs = Sig + (rng.normal(0, 1, len(Sig)) * err if noise else 0.0)
        M200, c200 = fit_nfw(obs, t.z, err)
        gt, R500 = publish(M200, c200, t.z, r_pts)
        nm += [t.name] * len(r_pts)
        rr += list(r_pts)
        gb += list(t.gbar(r_pts))
        go += list(gt)
        R5 += [R500] * len(r_pts)
    return (np.array(nm), np.array(rr), np.array(gb), np.array(go), np.array(R5))


def statistics(nm, r, gb, go, R5, stat="y"):
    e = S.excess_y(gb, go) if stat == "y" else S.excess_a0(gb, go)
    x = np.log10(r / R5)
    names = sorted(set(nm.tolist()))
    mu = np.array([e[nm == c].mean() for c in names])
    lR = np.array([math.log10(R5[nm == c][0]) for c in names])
    return dict(S1_pooled_slope=S.ols_slope(x, e),
                S2_spearman=S.spear(x, e),
                S3_fe_slope=S.fe_slope(x, e, nm),
                S4_between_pearson=S.pear(lR, mu),
                slope_vs_physical_r=S.ols_slope(np.log10(r / KPC), e))
