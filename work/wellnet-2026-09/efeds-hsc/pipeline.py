"""Baryons -> modified gravity -> REDUCED SHEAR forward model.

The point of this lane is that the observable must not contain the gas density
profile.  Run Z scored a hydrostatic g_obs, which is the density log-slope times
kT/r, against a potential built from the same density fit, and the two were
algebraically the same quantity (corr = +1.0000 after controlling g_bar and r).
Weak-lensing shear is measured from galaxy shapes and carries no dependence on
the X-ray fit at all, so the density profile enters the BARYONIC side only.

CHAIN
    n_e(r)          Bahar+2022 Vikhlinin fit, eFEDS, 542 systems
    M_b(<r)         4 pi Int rho_gas r^2 dr, optionally scaled for stars
    g_b(r)          G M_b(<r) / r^2
    DeltaPhi_b(r)   Int_r^{r_ref} g_b ds        PRESPECIFIED reference rule
    g_RAR           g_b / (1 - exp(-sqrt(g_b/a0)))
    g_pred(r)       g_RAR * 10^{beta * x_Phi},  x_Phi = log10(DeltaPhi_b/Phi_0)
    M_dyn(r)        g_pred r^2 / G
    rho_dyn(r)      (1/4 pi r^2) dM_dyn/dr           gate: >= 0 and M_dyn rising
    Sigma(R)        2 Int rho_dyn dl                 Abel, cosh substitution
    DeltaSigma(R)   Sigmabar(<R) - Sigma(R)
    kappa, gamma_t  DeltaSigma / Sigma_cr, Sigma / Sigma_cr
    g_+(R)          gamma_t/(1-kappa) * [1 + kappa(<b^2>/<b>^2 - 1)]

The last line is Chiu+2022 Eq. (23) (Seitz & Schneider 1997), and it is the
reason this file exists: weak lensing measures REDUCED shear, not mass.

RELATIVISTIC ASSUMPTION, DECLARED.  Mapping a modified non-relativistic g(r)
into a deflection assumes the relativistic completion gives light the same
potential as matter, i.e. no gravitational slip.  TeVeS-like constructions are
built to do this; it is an assumption, not a derivation, and it is stated here
because every number downstream depends on it.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEAD01 = os.path.join(os.path.dirname(HERE), "lead01")
if LEAD01 not in sys.path:
    sys.path.insert(0, LEAD01)
import lead01 as L                                              # noqa: E402

G = L.G
MSUN = L.MSUN
MPC = L.MPC
KPC = L.KPC
CLIGHT = L.CLIGHT
A0 = 1.2e-10                       # m/s^2, RAR acceleration scale
PHI0 = 1.0e12                      # m^2/s^2, programme's potential-depth scale
H_LITTLE = 0.7                     # Chiu+2022 and Bahar+2022 fiducial
ARCSEC = math.pi / (180.0 * 3600.0)
ARCMIN = 60.0 * ARCSEC


# ------------------------------------------------------------------ distances
_ZG = np.linspace(0.0, 3.0, 3001)
_DC = np.concatenate([[0.0], np.cumsum(
    0.5 * (1.0 / np.sqrt(L.OM * (1 + _ZG[1:]) ** 3 + L.OL)
           + 1.0 / np.sqrt(L.OM * (1 + _ZG[:-1]) ** 3 + L.OL))
    * np.diff(_ZG))]) * (CLIGHT / L.H0)


def d_com(z):
    return np.interp(z, _ZG, _DC)


def d_ang(z):
    return d_com(z) / (1.0 + z)


def d_ang12(z1, z2):
    """Flat-universe angular diameter distance between two redshifts."""
    return (d_com(z2) - d_com(z1)) / (1.0 + z2)


# ------------------------------------------------- HSC source population
# Calibrated on the source densities Chiu+2022 Sect. 4.3 quotes for their own
# P(z)-cut selection: ~15, 11, 9.6, 6, 2, 0.3 per sq. arcmin for clusters at
# z_cl = 0.05, 0.25, 0.35, 0.50, 0.78, 1.10.  Their cut keeps galaxies with
# 98% of P(z) above z_cl + 0.2, so those numbers are the cumulative counts of
# the parent n(z) above z_min = z_cl + 0.2.
NEFF_ANCHORS = [(0.05, 15.0), (0.25, 11.0), (0.35, 9.6),
                (0.50, 6.0), (0.78, 2.0), (1.10, 0.3)]


def _nz_shape(z, a, z0, b):
    return np.where(z > 0, z ** a * np.exp(-((np.maximum(z, 1e-9) / z0) ** b)),
                    0.0)


def fit_source_nz(verbose=True):
    """Fit a Smail-type parent n(z) to Chiu's quoted source densities."""
    zs = np.linspace(0.01, 2.5, 500)                 # z_MC < 2.5 cut applied
    best = None
    for a in np.linspace(0.4, 3.0, 27):
        for z0 in np.linspace(0.3, 1.6, 27):
            for b in np.linspace(0.8, 3.0, 23):
                sh = _nz_shape(zs, a, z0, b)
                tot = np.trapezoid(sh, zs)
                if tot <= 0:
                    continue
                pred = []
                for zc, _ in NEFF_ANCHORS:
                    m = zs > zc + 0.2
                    pred.append(np.trapezoid(sh[m], zs[m]) / tot)
                pred = np.array(pred)
                obs = np.array([n for _, n in NEFF_ANCHORS])
                # one free normalisation = the total source density
                norm = np.sum(pred * obs) / np.sum(pred * pred)
                chi = np.sum((norm * pred - obs) ** 2 / np.maximum(obs, .3))
                if best is None or chi < best[0]:
                    best = (chi, a, z0, b, norm)
    chi, a, z0, b, norm = best
    if verbose:
        sh = _nz_shape(zs, a, z0, b)
        tot = np.trapezoid(sh, zs)
        print(f"   HSC source n(z) ~ z^{a:.2f} exp[-(z/{z0:.2f})^{b:.2f}], "
              f"n_tot = {norm:.1f} /arcmin^2, chi2 = {chi:.3f}")
        for (zc, n) in NEFF_ANCHORS:
            m = zs > zc + 0.2
            p = norm * np.trapezoid(sh[m], zs[m]) / tot
            print(f"      z_cl {zc:4.2f}: Chiu {n:5.1f}  model {p:5.1f} "
                  f"/arcmin^2")
    return dict(a=a, z0=z0, b=b, ntot=norm)


class Sources:
    """Selected-source lensing efficiency and density versus lens redshift."""

    def __init__(self, par):
        self.par = par
        self.z = np.linspace(0.01, 2.5, 500)
        self.n = _nz_shape(self.z, par["a"], par["z0"], par["b"])
        self.n /= np.trapezoid(self.n, self.z)

    def _sel(self, zcl):
        return self.z > zcl + 0.2

    def neff(self, zcl):
        m = self._sel(zcl)
        return self.par["ntot"] * np.trapezoid(self.n[m], self.z[m])

    def beta_moments(self, zcl):
        """<beta> and <beta^2> with beta = D_ls/D_s for selected sources."""
        m = self._sel(zcl)
        z, n = self.z[m], self.n[m]
        if z.size < 3:
            return 0.0, 0.0
        b = d_ang12(zcl, z) / d_ang(z)
        w = n / np.trapezoid(n, z)
        return (float(np.trapezoid(b * w, z)),
                float(np.trapezoid(b * b * w, z)))

    def sigma_crit_eff(self, zcl):
        """Sigma_cr for the beta-weighted source population, SI (kg/m^2)."""
        bm, _ = self.beta_moments(zcl)
        if bm <= 0:
            return np.inf
        return CLIGHT ** 2 / (4.0 * math.pi * G * d_ang(zcl) * bm)


# --------------------------------------------------------- Abel deprojection
def sigma_from_g(r, g, R_out, r_trunc_mpc=20.0, n_R=260, n_t=500):
    """Sigma(R), DeltaSigma(R) for the spherical mass that produces g(r).

    M_dyn = g r^2 / G, rho_dyn = (1/4 pi r^2) dM_dyn/dr, then the Abel
    projection with r = R cosh t, which removes the 1/sqrt(r^2 - R^2)
    singularity exactly rather than smoothing over it.  A flat error curve
    versus n_t would mean a modelling mismatch, not a quadrature error -- this
    is checked in gates.py.
    """
    M_dyn = g * r ** 2 / G
    dM = np.diff(M_dyn)
    rho = np.zeros_like(r)
    rho[1:-1] = ((M_dyn[2:] - M_dyn[:-2]) / (r[2:] - r[:-2])
                 / (4.0 * math.pi * r[1:-1] ** 2))
    rho[0], rho[-1] = rho[1], rho[-2]
    rho = np.maximum(rho, 0.0)
    rt = r_trunc_mpc * MPC
    lr, lrho = np.log(r), np.log(np.maximum(rho, 1e-40))

    def rho_at(x):
        return np.where(x >= rt, 0.0,
                        np.exp(np.interp(np.log(np.clip(x, r[0], r[-1])),
                                         lr, lrho)))

    R_out = np.atleast_1d(R_out)
    Rg = np.geomspace(1e-3 * MPC, R_out.max() * 1.05, n_R)
    t = np.linspace(0.0, 1.0, n_t)
    # r = R cosh t  =>  r dr / sqrt(r^2 - R^2) = R cosh t dt, so the
    # integrable singularity at r = R is removed exactly, not smoothed.
    T = np.arccosh(np.maximum(rt / Rg, 1.0))[:, None]
    tt = t[None, :] * T
    ch = np.cosh(tt)
    Sig = 2.0 * Rg * np.trapezoid(rho_at(Rg[:, None] * ch) * ch, tt, axis=1)
    Sig[Rg >= rt] = 0.0
    inner = np.concatenate([[0.0], np.cumsum(
        0.5 * (Sig[1:] * Rg[1:] + Sig[:-1] * Rg[:-1]) * np.diff(Rg))])
    inner += 0.5 * Sig[0] * Rg[0] ** 2              # Sigma ~ const as R -> 0
    Sbar = 2.0 * inner / Rg ** 2
    S = np.interp(R_out, Rg, Sig)
    Sb = np.interp(R_out, Rg, Sbar)
    return S, Sb - S, dM


# --------------------------------------------------- baryonic profile per system
class System:
    """One eFEDS system: baryon profile, g_b, DeltaPhi_b, and lensing."""

    R_GRID = np.geomspace(5e-3, 30.0, 420) * MPC        # proper metres

    def __init__(self, rec, f_star=0.0):
        self.__dict__.update(rec)
        self.f_star = f_star
        u = self.R_GRID / self.rs
        ne = L.ne_of_u(u, self.n0sq, self.eps, self.beta, self.alpha) * 1e6
        rho = L.MU_E * L.M_P * ne
        r = self.R_GRID
        integ = 4.0 * math.pi * r ** 2 * rho
        Mg = np.concatenate([[0.0], np.cumsum(
            0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
        self.r = r
        self.rho_gas = rho
        self.M_gas = Mg
        self.M_b = Mg * (1.0 + f_star)
        self.g_b = G * self.M_b / r ** 2
        self.slope = L.dlnne_dlnr(u, self.eps, self.beta, self.alpha)

    # ---- boundary rules.  DECLARED IN ADVANCE, primary = 'fixed10Mpc'.
    def dphi(self, rule):
        r, g = self.r, self.g_b
        if rule.startswith("fixed"):
            r_ref = float(rule[5:].replace("Mpc", "")) * MPC
        elif rule == "2xR500":
            r_ref = 2.0 * self.R500
        elif rule == "10xrs":
            r_ref = 10.0 * self.rs
        else:
            raise ValueError(rule)
        r_ref = min(r_ref, r[-1])
        seg = 0.5 * (g[1:] + g[:-1]) * np.diff(r)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        cum_ref = float(np.interp(r_ref, r, cum))
        d = cum_ref - cum
        return np.maximum(d, 1e-30), r_ref

    # ---- the laws.  beta_pd is the hypothesis; tilt and a0_scale are the
    #      COMPETITORS, each with exactly one parameter, run through the
    #      identical pipeline so the model comparison is like for like.
    @staticmethod
    def g_rar(g_b, a0=A0):
        x = np.sqrt(np.maximum(g_b, 1e-30) / a0)
        return g_b / (1.0 - np.exp(-x))

    def set_xperp(self, coef):
        """Store x_Phi with the (log g_b, log r) quadratic projected out.

        96.6% of the variance of x_Phi across this sample is explained by a
        quadratic in log g_b and log r, so a fitted coefficient of x_Phi is
        mostly a coefficient of acceleration and radius.  This is the part
        that is not: the residual, which is the only piece of potential depth
        that could carry information a function of (g_bar, r) cannot.
        """
        lg = np.log10(np.maximum(self.g_b, 1e-30))
        lr = np.log10(self.r / MPC)
        d, _ = self.dphi("fixed10Mpc")
        x = np.log10(d / PHI0)
        X = np.column_stack([np.ones_like(lg), lg, lg ** 2, lr, lr ** 2,
                             lg * lr])
        self.xperp = x - X @ coef

    def g_pred(self, beta_pd=0.0, amp=0.0, rule="fixed10Mpc", law="rar",
               tilt=0.0, a0_scale=1.0, beta_perp=0.0):
        gb = self.g_b
        g = gb if law == "newton" else self.g_rar(gb, A0 * a0_scale)
        if beta_pd != 0.0:
            d, _ = self.dphi(rule)
            g = g * 10.0 ** (beta_pd * np.log10(d / PHI0))
        if beta_perp != 0.0:
            g = g * 10.0 ** (beta_perp * self.xperp)
        if tilt != 0.0:                       # a bare radius tilt, Run R's M2
            g = g * (self.r / MPC) ** tilt
        if amp != 0.0:
            g = g * 10.0 ** amp
        return g

    def trunc_for(self, rule, r_trunc_mpc):
        """Truncation radius, tied to the boundary rule.

        DeltaPhi_b -> 0 at r_ref by construction, so 10^{beta x_Phi} -> 0 and
        M_dyn turns over just inside r_ref -- measured at r = 8.3-9.0 Mpc for
        r_ref = 10 Mpc.  The projection is therefore cut at 0.8 r_ref, for
        EVERY model including beta = 0, so the comparison stays like for like.
        """
        _, r_ref = self.dphi(rule)
        return min(r_trunc_mpc, 0.8 * r_ref / MPC)

    # ---- lensing
    def sigma_profile(self, g, R_out, r_trunc_mpc=20.0):
        S, dS, dM = sigma_from_g(self.r, g, R_out, r_trunc_mpc)
        # monotonicity is only meaningful INSIDE the truncation radius: the
        # potential-difference factor drives M_dyn down as r -> r_ref, which
        # is why the projection is cut at 0.8 r_ref in the first place
        inside = self.r[1:] <= r_trunc_mpc * MPC
        self.last_dM = dM[inside] if np.any(inside) else dM
        return S, dS

    def reduced_shear(self, R_out, src, r_trunc_mpc=20.0, **kw):
        g = self.g_pred(**kw)
        rt = self.trunc_for(kw.get("rule", "fixed10Mpc"), r_trunc_mpc)
        S, dS = self.sigma_profile(g, R_out, rt)
        scr = src.sigma_crit_eff(self.z)
        bm, b2 = src.beta_moments(self.z)
        kap, gam = S / scr, dS / scr
        corr = 1.0 + kap * ((b2 / bm ** 2 - 1.0) if bm > 0 else 0.0)
        return gam / np.maximum(1.0 - kap, 1e-3) * corr, kap, gam


# ---------------------------------------------------- NFW, for the gates only
def nfw_delta_sigma(R, M200, c200, z):
    """Wright & Brainerd (2000) analytic DeltaSigma for an NFW halo, SI."""
    rho_c = (3.0 * (L.H0 * L.E(z)) ** 2) / (8.0 * math.pi * G)
    r200 = (M200 / (200.0 * rho_c * 4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
    rs = r200 / c200
    dc = (200.0 / 3.0) * c200 ** 3 / (math.log(1 + c200) - c200 / (1 + c200))
    x = R / rs
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        if abs(xi - 1.0) < 1e-6:
            f = 10.0 / 3.0 + 4.0 * math.log(0.5)
        elif xi < 1.0:
            s = math.sqrt(1 - xi ** 2)
            f = (8.0 * math.atanh(math.sqrt((1 - xi) / (1 + xi)))
                 / (xi ** 2 * s)
                 + 4.0 / xi ** 2 * math.log(xi / 2.0)
                 - 2.0 / (xi ** 2 - 1)
                 + 4.0 * math.atanh(math.sqrt((1 - xi) / (1 + xi)))
                 / ((xi ** 2 - 1) * s))
        else:
            s = math.sqrt(xi ** 2 - 1)
            f = (8.0 * math.atan(math.sqrt((xi - 1) / (1 + xi)))
                 / (xi ** 2 * s)
                 + 4.0 / xi ** 2 * math.log(xi / 2.0)
                 - 2.0 / (xi ** 2 - 1)
                 + 4.0 * math.atan(math.sqrt((xi - 1) / (1 + xi)))
                 / (xi ** 2 - 1) ** 1.5)
        out[i] = f
    return rs * dc * rho_c * out


def nfw_sigma(R, M200, c200, z):
    rho_c = (3.0 * (L.H0 * L.E(z)) ** 2) / (8.0 * math.pi * G)
    r200 = (M200 / (200.0 * rho_c * 4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
    rs = r200 / c200
    dc = (200.0 / 3.0) * c200 ** 3 / (math.log(1 + c200) - c200 / (1 + c200))
    x = np.atleast_1d(R / rs)
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        if abs(xi - 1.0) < 1e-6:
            f = 2.0 / 3.0
        elif xi < 1.0:
            s = math.sqrt(1 - xi ** 2)
            f = 2.0 / (xi ** 2 - 1) * (1 - 2.0 / s
                                       * math.atanh(math.sqrt((1 - xi)
                                                              / (1 + xi))))
        else:
            s = math.sqrt(xi ** 2 - 1)
            f = 2.0 / (xi ** 2 - 1) * (1 - 2.0 / s
                                       * math.atan(math.sqrt((xi - 1)
                                                             / (1 + xi))))
        out[i] = f
    return rs * dc * rho_c * out


def nfw_m200_to_m500(M200, c200, z):
    rho_c = (3.0 * (L.H0 * L.E(z)) ** 2) / (8.0 * math.pi * G)
    r200 = (M200 / (200.0 * rho_c * 4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
    rs = r200 / c200

    def m(r):
        return (M200 * (math.log(1 + r / rs) - r / (rs + r))
                / (math.log(1 + c200) - c200 / (1 + c200)))

    lo, hi = 1e-3 * r200, r200
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if m(mid) / (4.0 / 3.0 * math.pi * mid ** 3) > 500.0 * rho_c:
            lo = mid
        else:
            hi = mid
    return m(0.5 * (lo + hi))
