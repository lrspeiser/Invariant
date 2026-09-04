"""forward.py -- an INDEPENDENTLY IMPLEMENTED weak-lensing forward model.

The inverse-crime rule: the inference must not share basis, discretisation,
solver or nuisance assumptions with the generator.  Run BF's generator builds a
cluster quadrupole by

    * evaluating a 3-D perturbation potential  chi(r) P2(u)  on a 64 x 64 x 31
      Cartesian grid,
    * collapsing it along the line of sight by a tanh-spaced sum,
    * taking second finite differences of the projected potential on that grid,
    * and interpolating bilinearly to the source positions.

This module shares none of that.  It works in two dimensions and in closed
form:

    * the monopole is the ANALYTIC NFW convergence and mean convergence
      (Wright & Brainerd 1999), not a numerical line-of-sight quadrature of an
      acceleration profile;
    * the quadrupole is specified as an m=2 CONVERGENCE profile f(R) and the
      lensing potential is recovered from the exact l=2 Green's function of
      the 2-D operator,

          psi(R) = -1/2 [ R^-2 Int_0^R f s^3 ds  +  R^2 Int_R^inf f s^-1 ds ],
          which solves  1/2 ( psi'' + psi'/R - 4 psi/R^2 ) = f;

    * the shear components follow analytically from psi,

          gamma_t = -1/2 [ psi'' - psi'/R + 4 psi/R^2 ] cos 2(phi-phi0)
          gamma_x =  2   [ psi'/R - psi/R^2 ]           sin 2(phi-phi0)

      (derived in the module test, which checks the ODE numerically);
    * sources are drawn from a different spatial law, and the nuisance model is
      written here rather than imported.

The point of the module is not to be a better simulator.  It is to let the same
statistic be scored against a quadrupole whose radial shape, axis distribution
and systematics were built by different code, and above all to let the HALO
AXIS DISTRIBUTION be varied -- which BF's generator fixes at
"baryon major axis plus 22 degrees of scatter, independent of the external
axis", a choice the separation result turns out to depend on entirely.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants
G = 4.300917270e-6                    # kpc (km/s)^2 / Msun
C_KMS = 299792.458
RHO_C = 136.0                         # Msun/kpc^3

# declared nuisance amplitudes (written here, not imported: the inference must
# not share the generator's nuisance model)
NOISE = dict(shape_sd=0.26, n_arcmin2=20.0, m_bias_sd=0.02,
             c_bias_sd=5.0e-4, psf_coherent=1.0e-3,
             photoz_sd=0.035, photoz_outlier=0.05,
             pa_bar_err_deg=7.0, axis_ext_err_deg=10.0, ell_bar_err=0.03)


def _E(z):
    return np.sqrt(0.3 * (1 + z) ** 3 + 0.7)


_ZT = np.linspace(0.0, 4.0, 8001)


def _build_dc():
    """Cumulative comoving distance by composite SIMPSON -- a different
    quadrature rule from the generator's cumulative trapezoid."""
    y = 1.0 / _E(_ZT)
    h = _ZT[1] - _ZT[0]
    out = np.zeros_like(_ZT)
    # Simpson over each consecutive PAIR of intervals, trapezoid for the odd
    # end point; accumulate.
    for i in range(2, len(_ZT), 2):
        out[i] = out[i - 2] + h / 3.0 * (y[i - 2] + 4 * y[i - 1] + y[i])
    for i in range(1, len(_ZT), 2):
        out[i] = out[i - 1] + 0.5 * h * (y[i - 1] + y[i])
    return out * (C_KMS / 70.0)


_DC = _build_dc()


def comoving(z):
    """Comoving distance [Mpc]."""
    return np.interp(np.asarray(z, float), _ZT, _DC)


def sigma_crit(zl, zs):
    zl = np.asarray(zl, float)
    zs = np.asarray(zs, float)
    Dl = comoving(zl) / (1 + zl) * 1e3
    Ds = comoving(zs) / (1 + zs) * 1e3
    Dls = (comoving(zs) - comoving(zl)) / (1 + zs) * 1e3
    Dls = np.where(Dls <= 0, np.nan, Dls)
    return C_KMS ** 2 * Ds / (4.0 * np.pi * G * Dl * Dls)


# ------------------------------------------------------------- analytic NFW
def nfw_sigma(R, M200, c, z=0.0):
    """Sigma(R) and Sigmabar(<R) for an NFW halo, closed form."""
    R = np.atleast_1d(np.asarray(R, float))
    r200 = (M200 / ((4.0 / 3.0) * np.pi * 200.0 * RHO_C)) ** (1.0 / 3.0)
    rs = r200 / c
    delta_c = (200.0 / 3.0) * c ** 3 / (np.log(1 + c) - c / (1 + c))
    k = 2.0 * rs * delta_c * RHO_C
    x = np.maximum(R / rs, 1e-8)
    S = np.empty_like(x)
    Sb = np.empty_like(x)
    lo, hi = x < 1 - 1e-6, x > 1 + 1e-6
    mid = ~(lo | hi)
    xl, xh = x[lo], x[hi]
    S[lo] = k / (xl ** 2 - 1) * (1 - 2.0 / np.sqrt(1 - xl ** 2)
                                 * np.arctanh(np.sqrt((1 - xl) / (1 + xl))))
    S[hi] = k / (xh ** 2 - 1) * (1 - 2.0 / np.sqrt(xh ** 2 - 1)
                                 * np.arctan(np.sqrt((xh - 1) / (xh + 1))))
    S[mid] = k / 3.0
    g = np.empty_like(x)
    g[lo] = (2.0 / np.sqrt(1 - xl ** 2)
             * np.arctanh(np.sqrt((1 - xl) / (1 + xl))) + np.log(xl / 2.0))
    g[hi] = (2.0 / np.sqrt(xh ** 2 - 1)
             * np.arctan(np.sqrt((xh - 1) / (xh + 1))) + np.log(xh / 2.0))
    g[mid] = 1.0 + np.log(0.5)
    Sb = 2.0 * k * g / x ** 2           # = (4/x^2) rs delta_c rho_c g(x)
    return S, Sb, rs, r200


# -------------------------------------------- the m=2 Green's function solve
def quad_potential(Rg, f):
    """psi(R) solving  1/2 (psi'' + psi'/R - 4 psi/R^2) = f(R)  on a log grid.

        psi = -1/2 [ R^-2 Int_0^R f s^3 ds + R^2 Int_R^inf f s^-1 ds ]
    """
    Rg = np.asarray(Rg, float)
    f = np.asarray(f, float)
    w = np.diff(Rg)
    a = f * Rg ** 3
    I1 = np.concatenate(([0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * w)))
    b = f / Rg
    I2c = np.concatenate(([0.0], np.cumsum(0.5 * (b[1:] + b[:-1]) * w)))
    I2 = I2c[-1] - I2c
    return -0.5 * (I1 / Rg ** 2 + Rg ** 2 * I2)


def quad_shear(Rg, f):
    """(T, X, kappa2) from the m=2 convergence profile f(R).

    gamma_t = T cos 2(phi-phi0),  gamma_x = X sin 2(phi-phi0),  kappa = f cos 2(.)
    """
    psi = quad_potential(Rg, f)
    d1 = np.gradient(psi, Rg)
    d2 = np.gradient(d1, Rg)
    T = -0.5 * (d2 - d1 / Rg + 4.0 * psi / Rg ** 2)
    X = 2.0 * (d1 / Rg - psi / Rg ** 2)
    return T, X, f


# ------------------------------------------------------ quadrupole profiles
def f_halo(Rg, kap0, e):
    """Elliptical-NFW convergence to first order in the ellipticity e.

    An isodensity contour with MAJOR axis along Delta = 0 and axis ratio
    q = 1 - e has elliptical radius R_e = sqrt(x^2 + y^2/q^2), so to first
    order in e

        R_e ~ R [ 1 - (e/2) cos 2Delta ],
        kappa(R, Delta) ~ kappa0(R) - (e/2) R kappa0'(R) cos 2Delta.

    A SOURCE-shape quadrupole: its amplitude is set by e and its radial run by
    the halo's own profile.

    BUG FOUND BY THIS LANE'S OWN TEST: the first version of this function
    returned +0.5 e R kappa0', i.e. it put the MINOR axis where the major axis
    belongs.  Because kappa0' < 0, that flipped the sign of every halo
    statistic -- S_bar came out at -2.4 in this model while Run BF's generator
    gives +10.6 for the same universe.  Two independent forward models
    disagreeing in SIGN is exactly what the cross-check is for; a single
    implementation would have reported a confident, wrong-signed result.
    """
    return -0.5 * e * Rg * np.gradient(kap0, Rg)


def f_tensor(Rg, kap0, A, r_t):
    """A LAW quadrupole: the response turns on outside r_t and is sourced by
    the field the visible matter makes.  Deliberately NOT the generator's l=2
    Green's-function solution of  div[(I + A f Q) grad Phi] -- a two-parameter
    turn-on the generator never used, so recovering it is an out-of-family test.
    """
    u = (Rg / r_t) ** 2
    return A * kap0 * u / (1.0 + u)


def f_ring(Rg, kap0, A, r0, s=0.35):
    """OUT-OF-GRAMMAR injection: a log-Gaussian ring in the quadrupole,
    belonging to neither the halo family nor the tensor family."""
    return A * kap0.max() * np.exp(-0.5 * (np.log(Rg / r0) / s) ** 2)


# ------------------------------------------------------------ the simulator
def draw_cluster_params(rng):
    M200 = 10 ** rng.uniform(14.2, 15.3)
    c = float(rng.uniform(3.5, 5.5))
    z = float(rng.uniform(0.15, 0.45))
    R500 = 1000.0 * (M200 / 6e14) ** (1 / 3) * 0.72
    ell_bar = float(np.clip(rng.beta(2.2, 5.0) * 1.3, 0.03, 0.6))
    pa_bar = float(rng.uniform(0, 180))
    ax_ext = float(rng.uniform(0, 180))
    return dict(M200=M200, c=c, z=z, R500=R500, ell_bar=ell_bar,
                pa_bar=pa_bar, ax_ext=ax_ext)


def halo_axis(p, rng, mis_deg=22.0, f_lss=0.0):
    """The collisionless halo's projected major axis.

    ``mis_deg``  scatter about the BARYON major axis (BF's generator: 22 deg).
    ``f_lss``    fraction of the alignment budget carried by the EXTERNAL axis
                 instead.  BF's generator sets f_lss = 0 -- its haloes know
                 nothing about the surrounding structure.  N-body haloes do
                 align with the filament they sit in, so this is the axis of
                 the answer, not a nuisance.
    """
    a_b = np.deg2rad(2 * p["pa_bar"])
    a_e = np.deg2rad(2 * p["ax_ext"])
    v = (1.0 - f_lss) * np.array([np.cos(a_b), np.sin(a_b)]) \
        + f_lss * np.array([np.cos(a_e), np.sin(a_e)])
    base = 0.5 * np.rad2deg(np.arctan2(v[1], v[0]))
    return float((base + rng.normal(0.0, mis_deg)) % 180.0)


def emit(p, kind, rng, e_halo=None, A_tensor=0.0, mis_deg=22.0, f_lss=0.0,
         ring=False, n_src=None, sys_scale=1.0):
    """One cluster's weak-lensing catalogue, in the SAME dict schema the
    estimator consumes.  ``kind`` in {'none', 'halo', 'tensor', 'both'}."""
    R500 = p["R500"]
    Rg = np.geomspace(0.02 * R500, 6.0 * R500, 900)
    S, Sb, rs, r200 = nfw_sigma(Rg, p["M200"], p["c"])
    zs_ref = p["z"] + 0.9
    Scr_ref = float(sigma_crit(p["z"], zs_ref))
    kap0 = S / Scr_ref
    gt0 = (Sb - S) / Scr_ref

    terms = []
    if kind in ("halo", "both"):
        e = e_halo if e_halo is not None else 0.7 * p["ell_bar"]
        e = float(np.clip(e, 0.0, 0.9))
        terms.append((f_halo(Rg, kap0, e), halo_axis(p, rng, mis_deg, f_lss)))
    if kind in ("tensor", "both"):
        fq = (f_ring(Rg, kap0, A_tensor, 0.8 * R500) if ring
              else f_tensor(Rg, kap0, A_tensor, 0.30 * R500))
        terms.append((fq, p["ax_ext"]))

    # sources: uniform in AREA over an annulus, a different law from the
    # generator's sqrt(uniform(r_min^2, r_max^2)) over a disc with a hole
    if n_src is None:
        n_src = int(np.clip(NOISE["n_arcmin2"] * 900.0 * (R500 / 1000.0) ** 2,
                            800, 9000))
    rmin, rmax = 0.10 * R500, 2.4 * R500
    u = rng.random(n_src)
    rr = np.sqrt(rmin ** 2 + u * (rmax ** 2 - rmin ** 2))
    pp = rng.uniform(0, 2 * np.pi, n_src)
    zs = np.clip(p["z"] + 0.25 + rng.gamma(2.6, 0.28, n_src), p["z"] + 0.08, 3.4)
    Scr = sigma_crit(p["z"], zs)
    w = Scr_ref / Scr                       # lensing efficiency vs the reference

    kap = np.interp(rr, Rg, kap0) * w
    gt = np.interp(rr, Rg, gt0) * w
    g1 = -gt * np.cos(2 * pp)
    g2 = -gt * np.sin(2 * pp)
    for (fq, phi0) in terms:
        T, X, k2 = quad_shear(Rg, fq)
        d = 2 * (pp - np.deg2rad(phi0))
        Tq = np.interp(rr, Rg, T) * w
        Xq = np.interp(rr, Rg, X) * w
        kq = np.interp(rr, Rg, k2) * w
        gt_q = Tq * np.cos(d)
        gx_q = Xq * np.sin(d)
        # rotate (gamma_t, gamma_x) back into (gamma_1, gamma_2)
        g1 += -(gt_q * np.cos(2 * pp) - gx_q * np.sin(2 * pp))
        g2 += -(gt_q * np.sin(2 * pp) + gx_q * np.cos(2 * pp))
        kap = kap + kq * np.cos(d)

    red1 = g1 / np.maximum(1.0 - kap, 0.25)
    red2 = g2 / np.maximum(1.0 - kap, 0.25)
    se = NOISE["shape_sd"]
    m = rng.normal(0, NOISE["m_bias_sd"] * sys_scale)
    c1 = rng.normal(0, NOISE["c_bias_sd"] * sys_scale)
    c2 = rng.normal(0, NOISE["c_bias_sd"] * sys_scale)
    # a coherent PSF residual with a DIFFERENT spatial form: a linear gradient
    # plus a quadratic term, not the generator's single Fourier mode
    sx, sy = rr * np.cos(pp), rr * np.sin(pp)
    gx_, gy_ = rng.normal(size=2)
    amp = NOISE["psf_coherent"] * sys_scale
    e1 = (1 + m) * red1 + c1 + amp * (gx_ * sx / rmax + (sx * sy) / rmax ** 2) \
        + rng.normal(0, se, n_src)
    e2 = (1 + m) * red2 + c2 + amp * (gy_ * sy / rmax + (sx ** 2 - sy ** 2) / rmax ** 2) \
        + rng.normal(0, se, n_src)
    zph = zs * (1 + rng.normal(0.0, NOISE["photoz_sd"], n_src))
    nout = int(NOISE["photoz_outlier"] * sys_scale * n_src)
    if nout > 0:
        oi = rng.choice(n_src, size=min(nout, n_src), replace=False)
        zph[oi] = rng.uniform(p["z"] + 0.05, 3.0, len(oi))
    return dict(
        name="F", z=p["z"], R500=R500, src_x=sx, src_y=sy, e1=e1, e2=e2,
        w=np.full(n_src, 1.0 / (se ** 2 + 0.09)), z_src_phot=zph,
        pa_bar_obs=float((p["pa_bar"]
                          + rng.normal(0, NOISE["pa_bar_err_deg"])) % 180.0),
        axis_ext_obs=float((p["ax_ext"]
                            + rng.normal(0, NOISE["axis_ext_err_deg"])) % 180.0),
        ell_bar_obs=float(max(p["ell_bar"] + rng.normal(0, NOISE["ell_bar_err"]),
                              0.01)))


def corpus(kind, rng, n_clu=12, **kw):
    return [emit(draw_cluster_params(rng), kind, rng, **kw) for _ in range(n_clu)]


# ======================================================================
# galaxy channel: an independent IFU forward model
# ======================================================================
def emit_galaxy(rng, kind, q_amp=0.0, mis_deg=25.0, f_lss=0.0, nspax=26):
    """A disc galaxy's PSF-convolved line-of-sight velocity field.

    kind:  'none'   -- axisymmetric
           'tensor' -- the m=2 modulation of v_c^2 is locked to the EXTERNAL axis
           'halo'   -- it is locked to a flattened halo's own in-plane axis,
                       which sits near the DISC axis with `mis_deg` of scatter
                       and takes a fraction `f_lss` of its alignment from the
                       external axis instead.

    BF's generator gives every CDM galaxy a spherical halo, so its galaxy m=3
    channel has nothing to fire on.  This is the missing arm.
    """
    Rd = float(10 ** rng.uniform(0.2, 0.9))
    incl = float(np.rad2deg(np.arccos(rng.uniform(np.cos(np.deg2rad(78)),
                                                  np.cos(np.deg2rad(28))))))
    pa = float(rng.uniform(0, 180))
    dist = float(10 ** rng.uniform(0.8, 1.9))
    ax_ext = float(rng.uniform(0, 180))
    vflat = float(rng.uniform(70, 260))

    if kind == "tensor":
        psi_true = np.deg2rad(ax_ext - pa)
    elif kind == "halo":
        a_d = 0.0                       # the disc's own axis, in the disc frame
        a_e = np.deg2rad(2 * (ax_ext - pa))
        v = (1 - f_lss) * np.array([np.cos(a_d), np.sin(a_d)]) \
            + f_lss * np.array([np.cos(a_e), np.sin(a_e)])
        psi_true = 0.5 * np.arctan2(v[1], v[0]) + np.deg2rad(rng.normal(0, mis_deg))
    else:
        psi_true = 0.0
        q_amp = 0.0

    ext = 5.4 * Rd
    ax = np.linspace(-ext, ext, nspax)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    par = np.deg2rad(pa)
    xr = X * np.cos(par) + Y * np.sin(par)
    yr = -X * np.sin(par) + Y * np.cos(par)
    inc = np.deg2rad(incl)
    yd = yr / max(np.cos(inc), 1e-3)
    R = np.hypot(xr, yd) + 1e-9
    th = np.arctan2(yd, xr)
    # a different rotation curve law from the generator's Freeman disc
    V = vflat * R / np.sqrt(R ** 2 + (0.8 * Rd) ** 2)
    q = q_amp * (R / (2.0 * Rd)) ** 2 / (1.0 + (R / (2.0 * Rd)) ** 2)
    V = V * np.sqrt(np.maximum(1.0 + q * np.cos(2 * (th - psi_true)), 0.05))
    v = V * np.cos(th) * np.sin(inc)
    I = np.exp(-R / Rd)
    from scipy.ndimage import gaussian_filter
    sig_pix = float(np.clip((1.5 / 2.355 * dist * 1e3 * np.pi / 180 / 3600)
                            / (ax[1] - ax[0]), 0.3, 4.0))
    Ic = gaussian_filter(I, sig_pix, mode="nearest")
    vc = gaussian_filter(I * v, sig_pix, mode="nearest") / np.maximum(Ic, 1e-30)
    sn = np.sqrt(np.maximum(Ic / Ic.max(), 1e-6))
    err = np.clip(8.0 / np.maximum(sn, 0.03), 3.0, 220.0)
    vobs = vc + rng.normal(0, err)
    return dict(
        ax_arcsec=ax / (dist * 1e3 * np.pi / 180 / 3600), v_map=vobs, v_err=err,
        mask=Ic > 1.5e-3 * Ic.max(),
        pa_obs=float(pa + rng.normal(0, 4.0)),
        incl_obs=float(np.clip(incl + rng.normal(0, 3.0), 12, 87)),
        dist_obs=float(dist * (1 + rng.normal(0, 0.10))),
        Rd_obs=float(Rd * (1 + rng.normal(0, 0.05))),
        axis_ext_obs=float((ax_ext + rng.normal(0, 12.0)) % 180.0))
