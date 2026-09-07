"""invariants.py -- invariant detector-level reductions and the P(G|B) residuals.

The discriminator (protocol item 3) must respect rotations, translations and
source permutations so it cannot learn catalogue ordering or coordinate
conventions.  Rather than a black box on pixels, this module reduces every
detector-level product to quantities that are invariant BY CONSTRUCTION and
complete for the azimuthal content the corpus can resolve:

  galaxies   per-ring harmonic decomposition of the PSF-convolved velocity
             field (m = 0..3, weighted least squares, with covariance): the
             rotation curve, the m=2 and m=3 powers, the m=3 complex
             amplitude RELATIVE to the disc axis; vertical dispersions;
             the photometric baryon model
  clusters   per-radial-bin harmonic fits (m = 0, 1, 2, 4) of BOTH the
             tangential and the cross ellipticity on a common source plane;
             the m=2 complex quadrupole per radial group with covariance;
             X-ray, SZ, member-dispersion and strong-lensing profiles; the
             member-locked well-strength correlation and its scrambled control

Phases enter only RELATIVE to observed axes (the baryon major axis, the
external axis), never as sky angles, so a global rotation changes nothing.
Sources and members are summed over, never indexed.

Protocol item 2 -- "match away the scalar monopole so total strength cannot
carry the answer" -- is built in: every gravitational quantity is expressed as
a RESIDUAL from the corpus's own cross-fitted universal scalar law nu-hat(g_bar),
fitted on half the galaxies and applied out of fold to the other half and to
every cluster.  A pure a0 shift, or any smooth change of the interpolating
function, is absorbed.  What remains is P(G | B): the freedom gravity has
once the baryons and the universal law are given.

Nothing here imports a projection, Jeans or hydrostatic routine from either
generator.  The analysis has its OWN Abel projection (density route, z-grid),
its own hydrostatic integral, its own Jeans solve (u = sqrt(r^2 - R^2)
substitution) and its own Einstein-radius finder.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.interpolate import BSpline

from universes.baryons import A0, G, disk_vc2, hernquist_M
from universes.physics import sigma_crit

KPC_M = 3.0856775814913673e19
RAD_AS = np.pi / 180.0 / 3600.0
KEV_PER_KMS2 = 1.0 / 1.5967e5
LAM_NET = 150.0
SHAPE_SD = 0.26

GAL_RING_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.75, 4.5, 5.2])   # x R_d
N_GRING = len(GAL_RING_EDGES) - 1
CLU_BIN_EDGES = np.geomspace(0.12, 2.3, 10)                                 # x R500
N_CBIN = len(CLU_BIN_EDGES) - 1
CLU_GROUPS = ((0, 3), (3, 6), (6, 9))                                       # bins per radial group
MEM_BINS = ((0.10, 0.40), (0.40, 0.90), (0.90, 1.80))
XR_GRID = np.geomspace(0.09, 1.25, 10)                                      # x R500, X-ray/SZ/HSE residual radii


# ======================================================================
# small numerical helpers (all analysis-side)
# ======================================================================
def wls(D, y, w):
    """Weighted least squares with a sandwich covariance."""
    A = D * w[:, None]
    XtX = D.T @ A
    try:
        Xi = np.linalg.inv(XtX + 1e-10 * np.eye(D.shape[1]))
    except np.linalg.LinAlgError:
        return None, None
    beta = Xi @ (A.T @ y)
    r = y - D @ beta
    B = D * (w * r)[:, None]
    cov = Xi @ (B.T @ B) @ Xi
    return beta, cov


def loglog_interp(x, xp, fp, slope_lo=None, slope_hi=None):
    """Interpolate log f against log x; power-law extrapolation outside."""
    x = np.atleast_1d(np.asarray(x, float))
    lx, lxp, lfp = np.log(x), np.log(xp), np.log(np.maximum(fp, 1e-300))
    out = np.interp(lx, lxp, lfp)
    if slope_lo is None:
        slope_lo = (lfp[1] - lfp[0]) / (lxp[1] - lxp[0])
    if slope_hi is None:
        slope_hi = (lfp[-1] - lfp[-3]) / (lxp[-1] - lxp[-3])
    lo, hi = lx < lxp[0], lx > lxp[-1]
    out[lo] = lfp[0] + slope_lo * (lx[lo] - lxp[0])
    out[hi] = lfp[-1] + slope_hi * (lx[hi] - lxp[-1])
    return np.exp(out)


def cumtrapz0(y, x):
    return np.concatenate(([0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))))


# ======================================================================
# the universal law: a cross-fitted P-spline in log g_bar
# ======================================================================
class UniversalLaw:
    """log10 nu-hat as a cubic B-spline in log10 g_bar [SI], ridge + roughness.

    Deliberately a different basis from BF's hat functions (7 knots) and
    from any injected family: the fitted law must be able to absorb any
    smooth scalar response, so that what the discriminator sees is not a
    poor interpolating function.
    """
    LO, HI = -12.8, -7.6
    K = 4

    def __init__(self, lg, ly, w=None, lam_ridge=1e-3, lam_rough=0.3):
        lg = np.asarray(lg, float)
        ly = np.asarray(ly, float)
        m = np.isfinite(lg) & np.isfinite(ly)
        lg, ly = lg[m], ly[m]
        w = np.ones_like(ly) if w is None else np.asarray(w, float)[m]
        self.t = np.concatenate(([self.LO] * self.K,
                                 np.linspace(self.LO, self.HI, 9)[1:-1],
                                 [self.HI] * self.K))
        n = len(self.t) - self.K
        B = self._basis(lg)
        A = B * w[:, None]
        Dm = np.diff(np.eye(n), n=2, axis=0)
        P = B.T @ A + lam_ridge * np.eye(n) + lam_rough * (Dm.T @ Dm)
        self.c = np.linalg.solve(P, A.T @ ly)
        self.support = (float(np.min(lg)), float(np.max(lg)))
        self.n_fit = int(len(lg))

    def _basis(self, lg):
        x = np.clip(np.asarray(lg, float), self.LO + 1e-9, self.HI - 1e-9)
        return BSpline.design_matrix(x, self.t, self.K - 1).toarray()

    def lognu(self, gbar_kms2_kpc):
        """log10 nu-hat at g_bar given in (km/s)^2/kpc, clipped to the fitted
        support (declared: C5 reads the fraction outside)."""
        lg = np.log10(np.maximum(np.asarray(gbar_kms2_kpc, float), 1e-30) * 1e6 / KPC_M)
        lgc = np.clip(lg, self.support[0], self.support[1])
        return self._basis(lgc) @ self.c

    def frac_outside(self, gbar_kms2_kpc):
        lg = np.log10(np.maximum(np.asarray(gbar_kms2_kpc, float), 1e-30) * 1e6 / KPC_M)
        return float(np.mean((lg < self.support[0]) | (lg > self.support[1])))


def _si(g):
    """(km/s)^2/kpc -> log10 m s^-2."""
    return np.log10(np.maximum(np.asarray(g, float), 1e-30) * 1e6 / KPC_M)


# ======================================================================
# galaxies
# ======================================================================
def reduce_galaxy(gd):
    """Everything invariant one can make from a galaxy's detector products.

    Returns None if the field is unusable (fewer than 4 good rings).
    """
    d = gd["dist_obs"] * 1e3
    ax = gd["ax_arcsec"] * d * RAD_AS
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    pa = np.deg2rad(gd["pa_obs"])
    xr = X * np.cos(pa) + Y * np.sin(pa)
    yr = -X * np.sin(pa) + Y * np.cos(pa)
    inc = np.deg2rad(gd["incl_obs"])
    sini = max(np.sin(inc), 0.2)
    yd = yr / max(np.cos(inc), 1e-3)
    R = np.hypot(xr, yd) + 1e-9
    th = np.arctan2(yd, xr)
    m0 = gd["mask"]
    Rd = gd["Rd_obs"]
    edges = GAL_RING_EDGES * Rd

    vrot = np.full(N_GRING, np.nan)
    verr = np.full(N_GRING, np.nan)
    p2 = np.full(N_GRING, np.nan)
    p3 = np.full(N_GRING, np.nan)
    Rc = np.sqrt(edges[1:] * edges[:-1])
    P3 = np.zeros((2, 2))
    v3 = np.zeros(2)
    for k in range(N_GRING):
        s = m0 & (R >= edges[k]) & (R < edges[k + 1])
        n = int(s.sum())
        if n < 10:
            continue
        t = th[s]
        y = gd["v_map"][s]
        w = 1.0 / np.maximum(gd["v_err"][s], 1.0) ** 2
        D = np.stack([np.ones(n), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t),
                      np.cos(3 * t), np.sin(3 * t)], 1)
        beta, cov = wls(D, y, w)
        if beta is None:
            continue
        a1 = float(beta[1])
        if a1 < 8.0:
            continue
        vrot[k] = a1 / sini
        verr[k] = float(np.sqrt(max(cov[1, 1], 0.0))) / sini
        # noise-debiased harmonic powers, normalised by the rotation amplitude
        p2[k] = (beta[3] ** 2 + beta[4] ** 2 - cov[3, 3] - cov[4, 4]) / a1 ** 2
        p3[k] = (beta[5] ** 2 + beta[6] ** 2 - cov[5, 5] - cov[6, 6]) / a1 ** 2
        Z = 4.0 * np.array([beta[5], beta[6]]) / a1
        C = 16.0 * cov[np.ix_([5, 6], [5, 6])] / a1 ** 2
        try:
            Pi = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            continue
        P3 += Pi
        v3 += Pi @ Z
    good = np.isfinite(vrot)
    if good.sum() < 4:
        return None
    if np.linalg.cond(P3) < 1e12:
        C3 = np.linalg.inv(P3)
        Z3 = C3 @ v3
    else:
        C3 = np.eye(2) * 1e6
        Z3 = np.zeros(2)

    # PSF-forward-modelled rotation curve.  The ring harmonics above are
    # biased at small radii by beam smearing (the PSF-weighted mean of a rising
    # curve), which the noise-free Newtonian control showed as a 0.1 dex
    # scatter in the radial boost.  The analyst knows the instrument PSF and
    # the distance, so the curve is fitted by convolving a model field with
    # the observed light, exactly as a real IFU analysis does.
    vfm = forward_curve(gd, X, Y, R, th, inc, sini)

    # baryon model from the OBSERVED photometry with the observed M/L gradient
    def gbar_of(Rk):
        grad = 10 ** (gd["ml_grad_obs"] * (Rk / Rd - 1.0))
        v2 = (disk_vc2(Rk, gd["Md_obs"], Rd) * grad
              + disk_vc2(Rk, gd["Mg_obs"], gd["Rg_obs"])
              + G * hernquist_M(Rk, gd["Mb_obs"], gd["ab_obs"]) / Rk)
        return np.maximum(v2, 1e-8) / Rk

    gbar = gbar_of(Rc)
    if vfm is not None:
        vcurve = np.maximum(vfm(Rc), 1.0)
        good = good & np.isfinite(vcurve)
    else:
        vcurve = vrot
    gobs = vcurve ** 2 / Rc
    ly = np.log10(np.maximum(gobs / gbar, 1e-4))
    lye = 2.0 * verr / np.maximum(vrot, 1.0) / np.log(10.0)

    # vertical: sigma_z^2 / h_z against 2 pi G Sigma, and the radial boost at the same radii
    Rv = np.asarray(gd["Rv"], float)
    Sig = (gd["Md_obs"] / (2 * np.pi * Rd ** 2) * np.exp(-Rv / Rd)
           + gd["Mg_obs"] / (2 * np.pi * gd["Rg_obs"] ** 2) * np.exp(-Rv / gd["Rg_obs"]))
    gz_bar = 2 * np.pi * G * np.maximum(Sig, 1e-12)
    gz_obs = np.asarray(gd["sz_obs"], float) ** 2 / max(gd["hz_obs"], 1e-3)
    if vfm is not None:
        gR_obs = np.maximum(vfm(Rv), 1.0) ** 2 / Rv
    else:
        lgR_obs = np.interp(np.log(Rv), np.log(Rc[good]), np.log(np.maximum(gobs[good], 1e-8)))
        gR_obs = np.exp(lgR_obs)
    gR_bar = gbar_of(Rv)
    vz = np.log10(np.maximum(gz_obs / gz_bar, 1e-4))
    vr = np.log10(np.maximum(gR_obs / gR_bar, 1e-4))
    return {
        "Rc": Rc, "Rd": Rd, "vrot": vrot, "verr": verr, "p2": p2, "p3": p3,
        "gbar": gbar, "gobs": gobs, "ly": ly, "lye": lye, "good": good,
        "m3_re": float(Z3[0]), "m3_im": float(Z3[1]),
        "m3_c11": float(C3[0, 0]), "m3_c12": float(C3[0, 1]), "m3_c22": float(C3[1, 1]),
        "psi_obs": float(gd["axis_ext_obs"] - gd["pa_obs"]),
        "Rv": Rv, "gz_bar": gz_bar, "gz_obs": gz_obs, "gR_bar": gR_bar, "gR_obs": gR_obs,
        "vz": vz, "vr": vr, "dz": vz - vr,
        "phot": dict(lMd=np.log10(gd["Md_obs"]), lMg=np.log10(max(gd["Mg_obs"], 1e5)),
                     lMb=np.log10(max(gd["Mb_obs"], 1e5)), lRd=np.log10(Rd),
                     lhz=np.log10(max(gd["hz_obs"], 1e-3) / Rd), incl=gd["incl_obs"],
                     ldist=np.log10(gd["dist_obs"]), lSext=np.log10(max(gd["S_ext_obs"], 1e-6)),
                     ltm=np.log10(max(gd["t_merge_proxy"], 1e-3)), void=gd["void_frac_obs"],
                     mlgrad=gd["ml_grad_obs"]),
        "name": gd["name"],
    }


def forward_curve(gd, X, Y, R, th, inc, sini):
    """Fit v_c(R) = V0 (1 - e^{-R/Rt}) (1 + a R/Rt) to the velocity field by
    forward modelling the PSF convolution with the photometric light model.
    Returns a callable v_c(R) or None."""
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import least_squares
    Rd = gd["Rd_obs"]
    I = (gd["Md_obs"] / (2 * np.pi * Rd ** 2) * np.exp(-R / Rd)
         + gd["Mg_obs"] / (2 * np.pi * gd["Rg_obs"] ** 2) * np.exp(-R / gd["Rg_obs"]))
    ax = gd["ax_arcsec"]
    pix_arcsec = float(ax[1] - ax[0])
    sig_pix = float(np.clip(1.5 / 2.355 / pix_arcsec, 0.3, 4.0))     # the instrument PSF
    Ic = gaussian_filter(I, sig_pix, mode="nearest")
    m = gd["mask"]
    y = gd["v_map"][m]
    w = 1.0 / np.maximum(gd["v_err"][m], 1.0)
    cth = np.cos(th) * sini
    # v_c(R) = sum_k c_k B_k(R) on hat functions: the convolved field is LINEAR
    # in c, so the fit is an exact weighted least squares with a smoothness
    # penalty, no iteration and no local minima.
    knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.75, 3.5, 4.5, 5.6]) * Rd

    def hats(Rq):
        Rq = np.asarray(Rq, float)
        out = np.zeros(Rq.shape + (len(knots),))
        for k in range(len(knots)):
            lo = knots[k - 1] if k > 0 else knots[0] - 1.0
            hi = knots[k + 1] if k < len(knots) - 1 else knots[-1] + 1.0
            out[..., k] = np.where(Rq < knots[k], (Rq - lo) / (knots[k] - lo),
                                   (hi - Rq) / (hi - knots[k]))
        return np.clip(out, 0.0, 1.0)

    Bmaps = hats(R)
    # v_c(0) = 0 is imposed by dropping the first hat BEFORE the solve
    cols = [gaussian_filter(I * Bmaps[..., k] * cth, sig_pix, mode="nearest")[m]
            / np.maximum(Ic[m], 1e-30) for k in range(1, len(knots))]
    D = np.stack(cols, 1) * w[:, None]
    yw = y * w
    n = len(knots) - 1
    Dm = np.diff(np.eye(n), n=2, axis=0)
    lam = 0.05 * np.sum(D ** 2) / n
    P = D.T @ D + lam * (Dm.T @ Dm) + 1e-6 * np.eye(n)
    try:
        c = np.linalg.solve(P, D.T @ yw)
    except np.linalg.LinAlgError:
        return None
    cc = np.concatenate(([0.0], c))
    return lambda Rq: hats(Rq) @ cc


def finish_galaxy(r, law):
    """Residual features of a reduced galaxy against a (frozen) universal law."""
    F = {}
    F.update(r["phot"])
    lnu = law.lognu(r["gbar"])
    res = r["ly"] - lnu
    g = r["good"]
    for k in range(N_GRING):
        F[f"lv_{k}"] = np.log10(r["vrot"][k]) if g[k] else np.nan
        F[f"ly_{k}"] = r["ly"][k] if g[k] else np.nan
        F[f"res_{k}"] = res[k] if g[k] else np.nan
        F[f"p2_{k}"] = r["p2"][k] if g[k] else np.nan
        F[f"p3_{k}"] = r["p3"][k] if g[k] else np.nan
        F[f"lgb_{k}"] = _si(r["gbar"][k]) if g[k] else np.nan
    rr = res[g]
    x = np.log10(r["Rc"][g] / r["Rd"])
    w = 1.0 / np.maximum(r["lye"][g], 0.01) ** 2
    F["res_mean"] = float(np.average(rr, weights=w))
    if len(rr) >= 3:
        A = np.stack([np.ones_like(x), x], 1)
        b = np.linalg.lstsq(A * w[:, None] ** 0.5, rr * w ** 0.5, rcond=None)[0]
        F["res_slope"] = float(b[1])
    else:
        F["res_slope"] = np.nan
    F["res_sd"] = float(np.std(rr)) if len(rr) >= 3 else np.nan
    F["res_out_in"] = float(rr[-1] - rr[0])
    F["lgb_out"] = _si(r["gbar"][g][-1])
    F["lgb_in"] = _si(r["gbar"][g][0])
    F["n_rings"] = float(g.sum())
    # vertical: raw, residual against the law, and the law-free isotropy contrast
    lnu_v = law.lognu(r["gR_bar"])
    for j in range(2):
        F[f"vz_{j}"] = float(r["vz"][j])
        F[f"vr_{j}"] = float(r["vr"][j])
        F[f"rz_{j}"] = float(r["vz"][j] - lnu_v[j])
        F[f"rr_{j}"] = float(r["vr"][j] - lnu_v[j])
        F[f"dz_{j}"] = float(r["dz"][j])
    # m=3 relative to the disc axis (stored raw; the phase features are made later)
    for k in ("m3_re", "m3_im", "m3_c11", "m3_c12", "m3_c22", "psi_obs"):
        F[k] = float(r[k])
    return F


# ======================================================================
# clusters
# ======================================================================
def project_sigma(rg, g, Rq):
    """Sigma(R), Sigmabar(<R) [Msun/kpc^2] for a spherical acceleration profile.

    Density route: M = g r^2 / G, rho = M' / (4 pi r^2), Sigma = 2 Int rho dz on a
    tanh-stretched z grid; Sigmabar by cumulative trapezoid in R.  Shares nothing
    with the generator's b cosh t quadrature of g.
    """
    rg = np.asarray(rg, float)
    M = g * rg ** 2 / G
    lnM = np.log(np.maximum(M, 1e-300))
    dlnM = np.gradient(lnM, np.log(rg))
    rho = np.maximum(M * dlnM / (4 * np.pi * rg ** 3), 0.0)
    # BUG FOUND BY THE LANE'S OWN CLOSED-FORM TEST: the first version used a
    # tanh-stretched z grid whose first step was ~40 r_max / 120, i.e. tens of
    # thousands of kpc, and returned Sigma too large by 6-60x with a 1/R
    # dependence.  A log-spaced z grid resolves every projection radius.
    Rq = np.atleast_1d(np.asarray(Rq, float))
    zmax = 40.0 * rg[-1]
    z = np.concatenate(([0.0], np.geomspace(1e-3 * min(rg[0], Rq.min()), zmax, 400)))
    r3 = np.sqrt(Rq[:, None] ** 2 + z[None, :] ** 2)
    rh = loglog_interp(r3.ravel(), rg, rho, slope_lo=0.0).reshape(r3.shape)
    Sig = 2.0 * np.trapezoid(rh, z, axis=1)
    # Sigmabar on the query radii: integrate Sigma R dR on a fine log grid
    Rf = np.geomspace(min(rg[0], Rq.min()) * 0.5, Rq.max(), 200)
    r3f = np.sqrt(Rf[:, None] ** 2 + z[None, :] ** 2)
    Sf = 2.0 * np.trapezoid(loglog_interp(r3f.ravel(), rg, rho, slope_lo=0.0).reshape(r3f.shape),
                            z, axis=1)
    cum = cumtrapz0(Sf * Rf, Rf) + 0.5 * Sf[0] * Rf[0] ** 2
    Sbar = 2.0 * np.interp(Rq, Rf, cum) / Rq ** 2
    return Sig, Sbar


def hydrostatic_kT(rg, g, rho):
    """kT(r) [keV] from P(r) = Int_r^inf rho g dr with a power-law tail."""
    w = rho * g
    cum = cumtrapz0(w, rg)
    m = -(np.log(w[-1]) - np.log(w[-4])) / (np.log(rg[-1]) - np.log(rg[-4]))
    m = float(np.clip(m, 1.5, 6.0))
    P = cum[-1] - cum + w[-1] * rg[-1] / (m - 1.0)
    return P / np.maximum(rho, 1e-300) * KEV_PER_KMS2


def jeans_sigma_los(rg, g, nu_r, Rq):
    """Isotropic spherical Jeans, projected, on the u = sqrt(r^2 - R^2) grid."""
    w = nu_r * g
    cum = cumtrapz0(w, rg)
    m = -(np.log(w[-1]) - np.log(w[-4])) / (np.log(rg[-1]) - np.log(rg[-4]))
    m = float(np.clip(m, 1.5, 8.0))
    nus2 = cum[-1] - cum + w[-1] * rg[-1] / (m - 1.0)      # nu sigma_r^2, with tail
    Rq = np.atleast_1d(np.asarray(Rq, float))
    umax = 30.0 * rg[-1]
    u = np.concatenate(([0.0], np.geomspace(1e-3 * Rq.min(), umax, 300)))
    r3 = np.sqrt(Rq[:, None] ** 2 + u[None, :] ** 2)
    num = np.trapezoid(loglog_interp(r3.ravel(), rg, np.maximum(nus2, 1e-300)).reshape(r3.shape), u, axis=1)
    den = np.trapezoid(loglog_interp(r3.ravel(), rg, np.maximum(nu_r, 1e-300)).reshape(r3.shape), u, axis=1)
    return np.sqrt(np.maximum(num / np.maximum(den, 1e-300), 1.0))


def fit_tracer_profile(Rm, R5):
    """N(R) ~ (1 + (R/a)^2)^-b from the projected member radii; grid search."""
    edges = np.geomspace(0.05, 2.5, 8) * R5
    h, _ = np.histogram(Rm, edges)
    area = np.pi * np.diff(edges ** 2)
    ctr = np.sqrt(edges[1:] * edges[:-1])
    dens = h / area
    ok = h > 0
    if ok.sum() < 3:
        return 0.3 * R5, 1.0
    best, bp = np.inf, (0.3 * R5, 1.0)
    for a in np.geomspace(0.08, 1.2, 12) * R5:
        for b in np.linspace(0.5, 2.5, 11):
            mod = (1 + (ctr / a) ** 2) ** (-b)
            amp = np.exp(np.mean(np.log(dens[ok] / mod[ok])))
            chi = np.sum(((dens[ok] - amp * mod[ok]) / np.sqrt(np.maximum(h[ok], 1)) * area[ok]) ** 2)
            if chi < best:
                best, bp = chi, (a, b)
    return bp


def reduce_cluster(cd):
    """Everything invariant one can make from a cluster's detector products."""
    R5 = float(cd["R500"])
    x, y = cd["src_x"], cd["src_y"]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    et = -(cd["e1"] * c2 + cd["e2"] * s2)
    ex = (cd["e1"] * s2 - cd["e2"] * c2)
    zs = np.maximum(cd["z_src_phot"], cd["z"] + 0.05)
    Scr = sigma_crit(cd["z"], zs)
    ok = np.isfinite(Scr) & (cd["z_src_phot"] > cd["z"] + 0.10) & (Scr > 0)
    if ok.sum() < 300:
        return None
    Sref = float(np.median(Scr[ok]))
    scale = Scr / Sref
    w_all = 1.0 / (SHAPE_SD ** 2 * np.maximum(scale, 1e-6) ** 2)

    edges = CLU_BIN_EDGES * R5
    mono = np.full(N_CBIN, np.nan)
    mono_e = np.full(N_CBIN, np.nan)
    xmono = np.full(N_CBIN, np.nan)
    q2 = np.full(N_CBIN, np.nan)
    q2e = np.full(N_CBIN, np.nan)
    q4 = np.full(N_CBIN, np.nan)
    q1 = np.full(N_CBIN, np.nan)
    Zb = np.zeros((N_CBIN, 2))
    Cb = np.zeros((N_CBIN, 2, 2))
    okb = np.zeros(N_CBIN, bool)
    resid_t = np.zeros_like(et)
    used = np.zeros_like(ok)
    for k in range(N_CBIN):
        s = ok & (r >= edges[k]) & (r < edges[k + 1])
        n = int(s.sum())
        if n < 25:
            continue
        p = phi[s]
        D = np.stack([np.ones(n), np.cos(2 * p), np.sin(2 * p), np.cos(4 * p), np.sin(4 * p),
                      np.cos(p), np.sin(p)], 1)
        wv = w_all[s] * scale[s] ** 2
        bt, ct = wls(D, et[s] * scale[s], wv)
        bx, cx = wls(D, ex[s] * scale[s], wv)
        if bt is None or bx is None:
            continue
        mono[k] = bt[0]
        mono_e[k] = np.sqrt(max(ct[0, 0], 1e-30))
        xmono[k] = bx[0]
        resid_t[s] = et[s] * scale[s] - D @ bt
        used |= s
        Zt = np.array([bt[1], bt[2]])
        Ct = ct[np.ix_([1, 2], [1, 2])]
        Zx = np.array([bx[2], -bx[1]])
        M = np.array([[0.0, 1.0], [-1.0, 0.0]])
        Cx = M @ cx[np.ix_([1, 2], [1, 2])] @ M.T
        try:
            Pt, Px = np.linalg.inv(Ct), np.linalg.inv(Cx)
            C = np.linalg.inv(Pt + Px)
            Z = C @ (Pt @ Zt + Px @ Zx)
        except np.linalg.LinAlgError:
            continue
        Zb[k], Cb[k], okb[k] = Z, C, True
        q2[k] = Z @ Z - np.trace(C)
        q2e[k] = np.sqrt(max(4.0 * Z @ C @ Z + 2.0 * np.trace(C @ C), 1e-30))
        q4[k] = (bt[3] ** 2 + bt[4] ** 2 - ct[3, 3] - ct[4, 4]
                 + bx[3] ** 2 + bx[4] ** 2 - cx[3, 3] - cx[4, 4]) / 2.0
        q1[k] = (bt[5] ** 2 + bt[6] ** 2 - ct[5, 5] - ct[6, 6]
                 + bx[5] ** 2 + bx[6] ** 2 - cx[5, 5] - cx[6, 6]) / 2.0
    if okb.sum() < 5:
        return None
    Rb = np.sqrt(edges[1:] * edges[:-1])

    # radial groups of the complex quadrupole
    Zg, Cg = [], []
    for (a, b) in CLU_GROUPS:
        P = np.zeros((2, 2))
        v = np.zeros(2)
        for k in range(a, b):
            if okb[k]:
                Pi = np.linalg.inv(Cb[k])
                P += Pi
                v += Pi @ Zb[k]
        if np.linalg.cond(P) < 1e12:
            C = np.linalg.inv(P)
            Zg.append(C @ v)
            Cg.append(C)
        else:
            Zg.append(np.zeros(2))
            Cg.append(np.eye(2) * 1e6)

    # ---- the observed baryons and the universal-law prediction machinery ----
    ra = np.asarray(cd["r_ann"], float)
    Mb_ann = np.maximum(np.asarray(cd["Mgas_obs"], float) + np.asarray(cd["Mstar_obs"], float), 1e8)
    Mg_ann = np.maximum(np.asarray(cd["Mgas_obs"], float), 1e7)
    rg = np.geomspace(0.01 * R5, 12.0 * R5, 260)
    s_lo = float(np.clip((np.log(Mb_ann[1]) - np.log(Mb_ann[0])) / (np.log(ra[1]) - np.log(ra[0])), 1.0, 3.0))
    s_hi = float(np.clip((np.log(Mb_ann[-1]) - np.log(Mb_ann[-3])) / (np.log(ra[-1]) - np.log(ra[-3])), 0.2, 1.6))
    Mb = loglog_interp(rg, ra, Mb_ann, s_lo, s_hi)
    gbar = G * Mb / rg ** 2
    Mg = loglog_interp(rg, ra, Mg_ann, s_lo, s_hi)
    rho_gas = np.maximum(Mg * np.gradient(np.log(Mg), np.log(rg)) / (4 * np.pi * rg ** 3), 1e-14)

    # members
    Rm = np.hypot(cd["mem_x"], cd["mem_y"])
    okm = cd["mem_p"] > 0.6
    sig_obs = np.full(3, np.nan)
    n_mem = np.zeros(3)
    for k, (a, b) in enumerate(MEM_BINS):
        s = okm & (Rm >= a * R5) & (Rm < b * R5)
        if s.sum() < 8:
            continue
        v = cd["mem_v"][s]
        v = v[np.abs(v - np.median(v)) < 3.2 * (np.std(v) + 1)]
        if len(v) < 6:
            continue
        sig_obs[k] = np.sqrt(max(np.var(v) - 30.0 ** 2, 1e2))
        n_mem[k] = len(v)
    a_tr, b_tr = fit_tracer_profile(Rm[okm], R5)
    nu_r = (1 + (rg / a_tr) ** 2) ** (-b_tr - 0.5)
    Rmem_c = np.array([np.sqrt(a * b) for a, b in MEM_BINS]) * R5

    # hydrostatic mass from the OBSERVED temperatures and gas density
    lnT = np.log(np.maximum(cd["kT_obs"], 1e-3))
    dlnT = np.gradient(lnT, np.log(ra))
    lnrho = np.log(loglog_interp(ra, rg, rho_gas))
    dlnn = np.gradient(lnrho, np.log(ra))
    Mhe = np.maximum(-(cd["kT_obs"] / KEV_PER_KMS2) * ra / G * (dlnT + dlnn), 1e10)

    # network: the m-fitted tangential residual against the member well-strength map
    net = _network(cd, r, used, resid_t, R5)

    # strong lensing: observed
    thE = float(cd["thetaE_kpc"])
    n_img = int(len(cd["sl_delays"]))
    delay = float(np.median(np.abs(cd["sl_delays"]))) if n_img else 0.0

    Rmem_all = Rm[okm]
    return {
        "R5": R5, "z": cd["z"], "Sref": Sref, "Rb": Rb,
        "mono": mono, "mono_e": mono_e, "xmono": xmono, "q2": q2, "q2e": q2e, "q4": q4, "q1": q1,
        "okb": okb, "Zg": np.array(Zg), "Cg": np.array(Cg),
        "pa_bar": float(cd["pa_bar_obs"]), "ax_ext": float(cd["axis_ext_obs"]),
        "ell": float(cd["ell_bar_obs"]),
        "rg": rg, "gbar": gbar, "rho_gas": rho_gas, "ra": ra,
        "kT_obs": np.asarray(cd["kT_obs"], float), "y_obs": np.asarray(cd["y_sz"], float),
        "Mhe": Mhe, "sig_obs": sig_obs, "n_mem": n_mem, "nu_r": nu_r, "Rmem_c": Rmem_c,
        "thE": thE, "n_img": n_img, "delay": delay, "z_sl": float(cd["z_sl"]),
        "net": net,
        "scene": dict(lMgas=np.log10(Mg_ann[np.argmin(np.abs(ra - R5))]),
                      lMstar=np.log10(max(np.asarray(cd["Mstar_obs"], float)[np.argmin(np.abs(ra - R5))], 1e8)),
                      lR500=np.log10(R5), zc=float(cd["z"]), ell=float(cd["ell_bar_obs"]),
                      loff=np.log10(max(cd["gas_gal_offset_obs"], 1.0)),
                      wcen=float(cd["t_merge_proxy"]), void=float(cd["void_frac_obs"]),
                      lnmem=np.log10(max(okm.sum(), 1)),
                      cmem=float(np.mean(Rmem_all < 0.5 * R5)) if len(Rmem_all) else np.nan,
                      lSref=np.log10(Sref)),
        "name": cd["name"],
    }


def _network(cd, r, used, resid_t, R5):
    sel = used & (r > 0.2 * R5) & (r < 2.2 * R5)
    if sel.sum() < 200:
        return dict(net_r=np.nan, net_ctrl=np.nan)
    mp = np.stack([cd["mem_x"], cd["mem_y"]], 1)
    mm = np.asarray(cd["mem_m_obs"], float)
    sx, sy = cd["src_x"][sel], cd["src_y"][sel]
    res = resid_t[sel]

    def slope_on(mpos):
        d2 = (sx[:, None] - mpos[None, :, 0]) ** 2 + (sy[:, None] - mpos[None, :, 1]) ** 2
        S = (G * mm[None, :] / (d2 + LAM_NET ** 2)).sum(1)
        # remove the radial trend of S in 8 quantile bins
        e = np.quantile(r[sel], np.linspace(0, 1, 9))
        St = S.copy()
        for a, b in zip(e[:-1], e[1:]):
            s = (r[sel] >= a) & (r[sel] <= b)
            if s.sum() > 2:
                St[s] = S[s] - S[s].mean()
        sd = np.std(St)
        if sd <= 0:
            return 0.0
        xs = St / sd
        return float(np.sum((xs - xs.mean()) * (res - res.mean())) / max(np.sum((xs - xs.mean()) ** 2), 1e-30))

    # the scrambled control must be ROTATION- and PERMUTATION-invariant like
    # everything else: angles are drawn in a deterministic member order (by
    # radius, then mass) and laid down RELATIVE to the observed baryon axis, so
    # rotating the catalogue with its axes rotates the control with it.
    rngc = np.random.default_rng(zlib.crc32(str(cd["name"]).encode()) & 0x7FFFFFFF)
    rmp = np.hypot(mp[:, 0], mp[:, 1])
    order = np.lexsort((mm, np.round(rmp, 3)))
    thp = np.empty(len(mm))
    thp[order] = rngc.uniform(0, 2 * np.pi, len(mm)) + np.deg2rad(float(cd["pa_bar_obs"]))
    mp2 = np.stack([rmp * np.cos(thp), rmp * np.sin(thp)], 1)
    return dict(net_r=slope_on(mp), net_ctrl=slope_on(mp2))


def finish_cluster(r, law):
    """Residual features of a reduced cluster against a (frozen) universal law."""
    F = {}
    F.update(r["scene"])
    rg, gbar = r["rg"], r["gbar"]
    lnu = law.lognu(gbar)
    g_pred = 10 ** lnu * gbar
    F["frac_out_support"] = law.frac_outside(gbar[(rg > 0.1 * r["R5"]) & (rg < 2.5 * r["R5"])])

    # --- weak lensing: predicted reduced tangential shear on the common source plane
    Sig, Sbar = project_sigma(rg, g_pred, r["Rb"])
    kap = Sig / r["Sref"]
    pred = (Sbar - Sig) / r["Sref"] / np.maximum(1.0 - kap, 0.25)
    okb = r["okb"]
    Rx = np.log10(r["Rb"] / r["R5"])
    wres = np.full(N_CBIN, np.nan)
    for k in range(N_CBIN):
        F[f"mono_{k}"] = r["mono"][k] if okb[k] else np.nan
        F[f"xmono_{k}"] = r["xmono"][k] if okb[k] else np.nan
        F[f"wpred_{k}"] = pred[k] if okb[k] else np.nan
        if okb[k] and pred[k] > 0:
            wres[k] = (r["mono"][k] - pred[k]) / pred[k]
        F[f"wres_{k}"] = float(np.clip(wres[k], -3.0, 30.0)) if np.isfinite(wres[k]) else np.nan
        F[f"q2_{k}"] = (r["q2"][k] / r["q2e"][k]) if okb[k] else np.nan
        F[f"q4_{k}"] = r["q4"][k] * 1e3 if okb[k] else np.nan
        F[f"q1_{k}"] = r["q1"][k] * 1e3 if okb[k] else np.nan
    g = okb & np.isfinite(r["mono"]) & (pred > 0)
    if g.sum() >= 3:
        wv = (pred[g] / np.maximum(r["mono_e"][g], 1e-6)) ** 2
        y = r["mono"][g] / pred[g]
        A = np.stack([np.ones(g.sum()), Rx[g]], 1)
        b = np.linalg.lstsq(A * wv[:, None] ** 0.5, y * wv ** 0.5, rcond=None)[0]
        F["wl_A"] = float(b[0])
        F["wl_slope"] = float(b[1])
        F["wl_lA"] = float(np.log10(max(np.average(y, weights=wv), 1e-3)))
        F["wl_chi2"] = float(np.sum(wv * (y - A @ b) ** 2) / max(g.sum() - 2, 1))
    else:
        F["wl_A"] = F["wl_slope"] = F["wl_lA"] = F["wl_chi2"] = np.nan
    # inner-vs-outer quadrupole power, studentised
    q2s = r["q2"] / r["q2e"]
    F["q2_in"] = float(np.nanmean(q2s[:3])) if np.any(okb[:3]) else np.nan
    F["q2_mid"] = float(np.nanmean(q2s[3:6])) if np.any(okb[3:6]) else np.nan
    F["q2_out"] = float(np.nanmean(q2s[6:])) if np.any(okb[6:]) else np.nan
    F["q2_grad"] = F["q2_out"] - F["q2_in"]
    for gi in range(3):
        F[f"z_re_{gi}"], F[f"z_im_{gi}"] = float(r["Zg"][gi][0]), float(r["Zg"][gi][1])
        F[f"z_c11_{gi}"], F[f"z_c12_{gi}"], F[f"z_c22_{gi}"] = (
            float(r["Cg"][gi][0, 0]), float(r["Cg"][gi][0, 1]), float(r["Cg"][gi][1, 1]))
    F["pa_bar"], F["ax_ext"] = r["pa_bar"], r["ax_ext"]

    # --- X-ray, SZ
    kT_pred = hydrostatic_kT(rg, g_pred, r["rho_gas"])
    kTp = loglog_interp(r["ra"], rg, np.maximum(kT_pred, 1e-3))
    tres = np.log10(np.maximum(r["kT_obs"], 1e-3) / kTp)
    rho_a = loglog_interp(r["ra"], rg, r["rho_gas"])
    yp = rho_a * kTp * r["R5"]
    yres = np.log10(np.maximum(r["y_obs"], 1e-30) / np.maximum(yp, 1e-30))
    Mp = g_pred * rg ** 2 / G
    hres = np.log10(r["Mhe"] / loglog_interp(r["ra"], rg, Mp))
    xa = np.log10(r["ra"] / r["R5"])
    # the per-radius residuals are REGRIDDED onto fixed radii in units of
    # R500, so the feature schema does not depend on how many annuli an
    # instrument happens to emit (generator 1: 13, generator 2: 11)
    xg = np.log10(XR_GRID)
    for tag, v, lim in (("t", tres, 2.0), ("y", yres, 2.0), ("h", hres, 3.0)):
        m = np.isfinite(v)
        vi = np.interp(xg, xa[m], v[m], left=np.nan, right=np.nan) if m.sum() >= 2 else np.full(len(xg), np.nan)
        for k in range(len(xg)):
            F[f"{tag}res_{k}"] = float(np.clip(vi[k], -lim, lim)) if np.isfinite(vi[k]) else np.nan
        A = np.stack([np.ones(m.sum()), xa[m]], 1)
        b = np.linalg.lstsq(A, v[m], rcond=None)[0]
        F[f"{tag}_lA"], F[f"{tag}_slope"] = float(b[0]), float(b[1])
        F[f"{tag}_sd"] = float(np.std(v[m] - A @ b))

    # --- member dynamics
    sp = jeans_sigma_los(rg, g_pred, r["nu_r"], r["Rmem_c"])
    dres = np.log10(np.maximum(r["sig_obs"], 1.0) / sp)
    for k in range(3):
        F[f"lsig_{k}"] = float(np.log10(r["sig_obs"][k])) if np.isfinite(r["sig_obs"][k]) else np.nan
        F[f"dres_{k}"] = float(np.clip(dres[k], -2, 2)) if np.isfinite(dres[k]) else np.nan
        F[f"nmem_{k}"] = float(r["n_mem"][k])
    m = np.isfinite(dres)
    F["d_lA"] = float(np.mean(dres[m])) if m.sum() else np.nan
    F["d_slope"] = float(dres[m][-1] - dres[m][0]) if m.sum() >= 2 else np.nan

    # --- strong lensing
    Scr_sl = float(sigma_crit(r["z"], r["z_sl"]))
    Rq = np.geomspace(0.02 * r["R5"], 0.6 * r["R5"], 40)
    _, Sb = project_sigma(rg, g_pred, Rq)
    kb = Sb / Scr_sl
    thE_p = 0.0
    if np.any(kb > 1.0) and np.any(kb <= 1.0):
        j = int(np.argmax(kb <= 1.0))
        j = max(j, 1)
        thE_p = float(np.interp(1.0, [kb[j], kb[j - 1]], [Rq[j], Rq[j - 1]]))
    F["thE_obs"] = r["thE"] / r["R5"]
    F["thE_pred"] = thE_p / r["R5"]
    F["sl_res"] = F["thE_obs"] - F["thE_pred"]
    F["sl_obs"] = float(r["thE"] > 0)
    F["sl_pred"] = float(thE_p > 0)
    F["n_img"] = float(r["n_img"])
    F["ldelay"] = float(np.log10(1.0 + r["delay"]))
    F["kb_in"] = float(np.log10(max(kb[0], 1e-4)))

    # --- network
    F["net_r"] = r["net"]["net_r"]
    F["net_ctrl"] = r["net"]["net_ctrl"]
    F["net_excess"] = r["net"]["net_r"] - r["net"]["net_ctrl"]

    # --- matter-photon covariance: lensing minus dynamics minus gas, same law
    F["ep_ld"] = F["wl_lA"] - F["d_lA"]
    F["ep_lt"] = F["wl_lA"] - F["t_lA"]
    F["ep_dt"] = F["d_lA"] - F["t_lA"]
    return F


# ======================================================================
# phase features (computed late, so alignment can be randomised)
# ======================================================================
def _proj(re, im, c11, c12, c22, ang_deg):
    a = np.deg2rad(ang_deg)
    u1, u2 = np.cos(2 * a), np.sin(2 * a)
    val = u1 * re + u2 * im
    var = u1 * u1 * c11 + 2 * u1 * u2 * c12 + u2 * u2 * c22
    return val / np.sqrt(np.maximum(var, 1e-30))


def cluster_phase_features(F, rng=None):
    """Studentised projections of the m=2 quadrupole on the observed baryon
    axis, the observed external axis and the 45-degree control, per radial
    group.  ``rng`` not None -> the observed axes are REPLACED by random ones
    (the 'randomise baryon-field alignment' ablation)."""
    n = len(F["pa_bar"])
    pb = np.asarray(F["pa_bar"], float)
    pe = np.asarray(F["ax_ext"], float)
    if rng is not None:
        pb = rng.uniform(0, 180, n)
        pe = rng.uniform(0, 180, n)
    out = {}
    for gi in range(3):
        re, im = F[f"z_re_{gi}"], F[f"z_im_{gi}"]
        c11, c12, c22 = F[f"z_c11_{gi}"], F[f"z_c12_{gi}"], F[f"z_c22_{gi}"]
        out[f"pb_{gi}"] = _proj(re, im, c11, c12, c22, pb)
        out[f"pe_{gi}"] = _proj(re, im, c11, c12, c22, pe)
        out[f"p45_{gi}"] = _proj(re, im, c11, c12, c22, pe + 45.0)
    out["pb_tot"] = out["pb_0"] + out["pb_1"] + out["pb_2"]
    out["pe_tot"] = out["pe_0"] + out["pe_1"] + out["pe_2"]
    out["p45_tot"] = out["p45_0"] + out["p45_1"] + out["p45_2"]
    out["pe_minus_pb"] = out["pe_tot"] - out["pb_tot"]
    return out


def galaxy_phase_features(F, rng=None):
    psi = np.asarray(F["psi_obs"], float)
    if rng is not None:
        psi = rng.uniform(0, 180, len(psi))
    out = {}
    re, im = F["m3_re"], F["m3_im"]
    c11, c12, c22 = F["m3_c11"], F["m3_c12"], F["m3_c22"]
    out["g3_ext"] = _proj(re, im, c11, c12, c22, psi)
    out["g3_45"] = _proj(re, im, c11, c12, c22, psi + 45.0)
    out["g3_disc"] = _proj(re, im, c11, c12, c22, np.zeros_like(psi))
    out["g3_amp"] = np.sqrt(np.maximum(re ** 2 + im ** 2 - c11 - c22, 0.0))
    return out


# ======================================================================
# corpus driver
# ======================================================================
def analyse_corpus(gal_dicts, clu_dicts, sn=None, split_seed=0):
    """Reduce a corpus; cross-fit the universal law on the galaxies (two folds
    by galaxy); return per-galaxy and per-cluster feature dicts and the
    corpus-level quantities."""
    rg = [reduce_galaxy(gd) for gd in gal_dicts]
    good_idx = [i for i, x in enumerate(rg) if x is not None]
    if len(good_idx) < 8:
        return None
    # the fold split is by galaxy index, so a galaxy sits in the same fold in
    # every arm of a paired set
    fold = np.random.default_rng(split_seed).permutation(len(rg)) % 2
    laws = []
    for f in (0, 1):
        sel = [i for i in good_idx if fold[i] == f]
        lg = np.concatenate([_si(rg[i]["gbar"][rg[i]["good"]]) for i in sel])
        ly = np.concatenate([rg[i]["ly"][rg[i]["good"]] for i in sel])
        w = np.concatenate([1.0 / np.maximum(rg[i]["lye"][rg[i]["good"]], 0.01) ** 2 for i in sel])
        laws.append(UniversalLaw(lg, ly, w))
    # a failed reduction is a NaN row, never a dropped one: object i of arm A
    # must stay paired with object i of arm B
    nan_g = {k: np.nan for k in galaxy_feature_names()}
    gal_F = []
    for i, x in enumerate(rg):
        gal_F.append(finish_galaxy(x, laws[1 - fold[i]]) if x is not None else dict(nan_g))
    law_c = _AvgLaw(laws)
    nan_c = {k: np.nan for k in cluster_feature_names()}
    clu_F = []
    for cd in clu_dicts:
        rc = reduce_cluster(cd)
        clu_F.append(finish_cluster(rc, law_c) if rc is not None else dict(nan_c))
    corpus = {"n_gal": len(good_idx), "n_clu": int(sum(np.isfinite(F["wl_lA"]) for F in clu_F)),
              "rar_scatter_oof": float(np.nanstd(np.concatenate(
                  [[F[f"res_{k}"] for k in range(N_GRING)] for F in gal_F]))),
              "law_at_m10p5": float(np.mean([l.lognu(10 ** -10.5 * KPC_M / 1e6) for l in laws])),
              "law_at_m11p5": float(np.mean([l.lognu(10 ** -11.5 * KPC_M / 1e6) for l in laws]))}
    if sn is not None:
        from universes.physics import comoving_Mpc
        z, mag, dur, vf = sn["z_obs"], sn["mag"], sn["duration"], sn["void_frac"]
        dlm = 5 * np.log10(np.maximum(comoving_Mpc(z) * (1 + z), 1e-3)) + 25.0
        hr = mag - dlm
        corpus["sn_void"] = _slope(vf, hr)
        corpus["sn_dur_void"] = _slope(vf, np.log10(np.maximum(dur, 1e-3)) - np.log10(1 + z))
        corpus["sn_scat"] = float(np.std(hr))
    return {"gal": gal_F, "clu": clu_F, "corpus": corpus}


class _AvgLaw:
    def __init__(self, laws):
        self.laws = laws
        self.support = (max(l.support[0] for l in laws), min(l.support[1] for l in laws))

    def lognu(self, g):
        return np.mean([l.lognu(g) for l in self.laws], axis=0)

    def frac_outside(self, g):
        return float(np.mean([l.frac_outside(g) for l in self.laws]))


def _slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return 0.0
    x, y = x[m], y[m]
    v = np.var(x)
    return float(np.mean((x - x.mean()) * (y - y.mean())) / v) if v > 0 else 0.0


# ======================================================================
# feature registry: names and channels
# ======================================================================
def galaxy_feature_names():
    names = ["lMd", "lMg", "lMb", "lRd", "lhz", "incl", "ldist", "lSext", "ltm", "void", "mlgrad"]
    for k in range(N_GRING):
        names += [f"lv_{k}", f"ly_{k}", f"res_{k}", f"p2_{k}", f"p3_{k}", f"lgb_{k}"]
    names += ["res_mean", "res_slope", "res_sd", "res_out_in", "lgb_out", "lgb_in", "n_rings"]
    for j in range(2):
        names += [f"vz_{j}", f"vr_{j}", f"rz_{j}", f"rr_{j}", f"dz_{j}"]
    names += ["m3_re", "m3_im", "m3_c11", "m3_c12", "m3_c22", "psi_obs"]
    return names


def cluster_feature_names():
    names = ["lMgas", "lMstar", "lR500", "zc", "ell", "loff", "wcen", "void", "lnmem", "cmem", "lSref",
             "frac_out_support"]
    for k in range(N_CBIN):
        names += [f"mono_{k}", f"xmono_{k}", f"wpred_{k}", f"wres_{k}", f"q2_{k}", f"q4_{k}", f"q1_{k}"]
    names += ["wl_A", "wl_slope", "wl_lA", "wl_chi2", "q2_in", "q2_mid", "q2_out", "q2_grad"]
    for gi in range(3):
        names += [f"z_re_{gi}", f"z_im_{gi}", f"z_c11_{gi}", f"z_c12_{gi}", f"z_c22_{gi}"]
    names += ["pa_bar", "ax_ext"]
    for k in range(len(XR_GRID)):
        names += [f"tres_{k}", f"yres_{k}", f"hres_{k}"]
    for tag in ("t", "y", "h"):
        names += [f"{tag}_lA", f"{tag}_slope", f"{tag}_sd"]
    for k in range(3):
        names += [f"lsig_{k}", f"dres_{k}", f"nmem_{k}"]
    names += ["d_lA", "d_slope", "thE_obs", "thE_pred", "sl_res", "sl_obs", "sl_pred", "n_img",
              "ldelay", "kb_in", "net_r", "net_ctrl", "net_excess", "ep_ld", "ep_lt", "ep_dt"]
    return names


GAL_PHASE = ["g3_ext", "g3_45", "g3_disc", "g3_amp"]
CLU_PHASE = [f"{p}_{g}" for g in range(3) for p in ("pb", "pe", "p45")] + \
            ["pb_tot", "pe_tot", "p45_tot", "pe_minus_pb"]


def galaxy_channel(name):
    """Channel of a galaxy feature, for the ablations."""
    if name in ("lMd", "lMg", "lMb", "lRd", "lhz", "incl", "ldist", "lSext", "ltm", "void", "mlgrad"):
        return "gal_baryons"
    if name.startswith(("lv_", "ly_", "lgb_", "n_rings")):
        return "gal_curve_raw"
    if name.startswith(("res_",)):
        return "gal_curve_res"
    if name.startswith(("p2_", "p3_")):
        return "gal_harmonic_power"
    if name.startswith(("vz_", "vr_", "rz_", "rr_", "dz_")):
        return "gal_vertical"
    if name.startswith(("m3_", "psi_", "g3_")):
        return "gal_m3_phase"
    return "other"


def cluster_channel(name):
    if name in ("lMgas", "lMstar", "lR500", "zc", "ell", "loff", "wcen", "void", "lnmem", "cmem", "lSref",
                "frac_out_support"):
        return "clu_baryons"
    if name.startswith(("mono_", "xmono_", "wpred_")):
        return "clu_shear_mono_raw"
    if name.startswith(("wres_", "wl_")):
        return "clu_shear_mono_res"
    if name.startswith(("q2_", "q4_", "q1_", "z_re", "z_im", "z_c")):
        return "clu_shear_quad_power"
    if name in ("pa_bar", "ax_ext") or name.startswith(("pb_", "pe_", "p45_")):
        return "clu_shear_quad_phase"
    if name.startswith(("tres_", "yres_", "hres_", "t_", "y_", "h_")):
        return "clu_gas"
    if name.startswith(("lsig_", "dres_", "nmem_", "d_")):
        return "clu_dynamics"
    if name.startswith(("thE", "sl_", "n_img", "ldelay", "kb_in")):
        return "clu_strong_lens"
    if name.startswith("net_"):
        return "clu_network"
    if name.startswith("ep_"):
        return "clu_matter_photon"
    return "other"
