"""corpus.py -- the instrument forward model and the mock corpus.

The SAME instrument is applied to every universe.  What the corpus contains is
detector-facing, not summary statistics:

  galaxies   a PSF-convolved, aperture-integrated line-of-sight VELOCITY FIELD
             on a spaxel grid with per-spaxel errors; a surface-brightness map;
             a vertical stellar dispersion at two radii; photometric mass with
             its M/L uncertainty; an inclination and distance WITH their errors
  clusters   per-source weak-lensing ellipticities e1,e2 with weights and
             photometric-redshift posterior summaries; individual member sky
             positions and redshifts; X-ray annulus PHOTON COUNTS and measured
             temperatures; SZ y in annuli; multiple-image positions and time
             delays; the observed surrounding-structure catalogue
  cosmology  supernova redshifts, peak magnitudes and light-curve DURATIONS

Nothing in the corpus is a mass.  The analysis has to make its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from . import physics as ph
from .baryons import A0, C_KMS, G, hernquist_M
from .provenance import DECLARED_NOISE as DN
from .scenes import LAM_NET, Q_NET, S0_NET

KEV_PER_KMS2 = 1.0 / 1.5967e5           # kT[keV] = (P/rho)[(km/s)^2] * this
RHO_C = 136.0                            # Msun/kpc^3 at z=0, H0=70
MU_E = 1.14


# ---------------------------------------------------------------- CDM haloes
_M200G = np.geomspace(1e9, 5e15, 600)
_MSG = 2 * 0.0351 * _M200G / ((_M200G / 10 ** 11.59) ** -1.376
                              + (_M200G / 10 ** 11.59) ** 0.608)


def shmr_M200(Mstar):
    return 10 ** np.interp(np.log10(Mstar), np.log10(_MSG), np.log10(_M200G))


def r200_of(M200):
    return (M200 / ((4.0 / 3.0) * np.pi * 200.0 * RHO_C)) ** (1.0 / 3.0)


def dm_galaxy(gal, p, rng):
    M200 = shmr_M200(gal.Md + gal.Mb) * 10 ** rng.normal(0, p["shmr_scatter"])
    c = p["c_norm"] * 2.2 * (M200 / 1e12) ** -0.10 * 10 ** rng.normal(0, 0.11)
    r2 = r200_of(M200)
    rs = r2 / c
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    norm = (1.0 - 0.157) * M200 / mu(c)
    return {"Mdm": lambda r: norm * mu(np.asarray(r, float) / rs),
            "M200": M200, "c": c, "r200": r2}


def dm_cluster(clu, p, rng):
    M200 = clu.Mbar500 / p["fbar"] * 1.35 * 10 ** rng.normal(0, 0.05)
    c = p["c_norm"] * 10 ** rng.normal(0, 0.13)
    r2 = r200_of(M200)
    rs = r2 / c
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    norm = (1.0 - p["fbar"]) * M200 / mu(c)
    # collisionless: the halo follows the GALAXIES, offset from the gas
    e_h = float(np.clip(0.72 * clu.ell_bar + rng.normal(0, 0.10), 0.0, 0.55))
    pa_h = float((clu.pa_bar_deg + rng.normal(0, 22.0)) % 180.0)
    return {"Mdm": lambda r: norm * mu(np.asarray(r, float) / rs),
            "M200": M200, "c": c, "r200": r2, "ell": e_h, "pa": pa_h,
            "offset": clu.gas_gal_offset}


# ---------------------------------------------------------------- lens optics
def _extrap_g(rg, g, r):
    """log-log power-law extrapolation of g beyond the grid."""
    r = np.asarray(r, float)
    out = np.exp(np.interp(np.log(r), np.log(rg), np.log(np.maximum(g, 1e-300))))
    sl = (np.log(g[-1]) - np.log(g[-6])) / (np.log(rg[-1]) - np.log(rg[-6]))
    hi = r > rg[-1]
    out[hi] = g[-1] * (r[hi] / rg[-1]) ** sl
    return out


def projected_mass_2d(rg, g_l, b):
    """M2D(b) = b^2 int_b^inf g_l(r) dr / sqrt(r^2-b^2), via r = b cosh t."""
    t = np.linspace(0.0, 5.2, 90)
    r = b[:, None] * np.cosh(t)[None, :]
    gg = _extrap_g(rg, g_l, r.ravel()).reshape(r.shape)
    I = np.trapezoid(gg, t, axis=1)
    return b ** 2 * I / G


def lens_profiles(rg, g_l, bgrid, Sig_cr):
    """kappa(b), kappa_bar(b) for a source plane with Sigma_cr [Msun/kpc^2]."""
    M2 = projected_mass_2d(rg, g_l, bgrid)
    kbar = M2 / (np.pi * bgrid ** 2) / Sig_cr
    dM = np.gradient(M2, bgrid)
    Sig = dM / (2 * np.pi * bgrid)
    kap = Sig / Sig_cr
    return kap, kbar


def quad_potential_map(geom, chi=None, Bnet=0.0, ell_terms=()):
    """Sigma-like second derivatives of the PERTURBATION lensing potential.

    Returns (Sxx, Syy, Sxy) in Msun/kpc^2, so that
        kappa_pert = (Sxx+Syy)/Sigma_cr,
        gamma1     = (Sxx-Syy)/Sigma_cr,   gamma2 = 2 Sxy/Sigma_cr.
    """
    P = np.zeros(geom.r3.shape[:2])
    if chi is not None:
        P = P + ((np.interp(geom.r3.ravel(), geom.rg, chi).reshape(geom.r3.shape)
                  * ph.P2(geom.u3)) * geom.dz[None, None, :]).sum(-1)
    if Bnet != 0.0:
        P = P + (Bnet * geom.Ex3 * geom.dz[None, None, :]).sum(-1)
    for (amp, pa_deg, prof) in ell_terms:
        ang = np.deg2rad(pa_deg)
        # elliptical-potential quadrupole about an axis in the sky plane
        X = geom.r3 * 0.0
        nx_, ny_ = np.cos(ang), np.sin(ang)
        Xg, Yg = np.meshgrid(geom.xg, geom.xg, indexing="ij")
        u = (Xg[:, :, None] * nx_ + Yg[:, :, None] * ny_) / geom.r3
        P = P + (amp * np.interp(geom.r3.ravel(), geom.rg, prof).reshape(geom.r3.shape)
                 * ph.P2(u) * geom.dz[None, None, :]).sum(-1)
    if not np.any(P):
        z = np.zeros_like(P)
        return z, z, z
    Psi = (2.0 / C_KMS ** 2) * P                      # kpc
    h = geom.xg[1] - geom.xg[0]
    Pxx = np.gradient(np.gradient(Psi, h, axis=0), h, axis=0)
    Pyy = np.gradient(np.gradient(Psi, h, axis=1), h, axis=1)
    Pxy = np.gradient(np.gradient(Psi, h, axis=0), h, axis=1)
    k = C_KMS ** 2 / (8.0 * np.pi * G)
    return k * Pxx, k * Pyy, k * Pxy


def _bilinear(F, xg, x, y):
    h = xg[1] - xg[0]
    ix = (x - xg[0]) / h
    iy = (y - xg[0]) / h
    return map_coordinates(F, np.stack([ix, iy]), order=1, mode="nearest")


# ---------------------------------------------------------------- Jeans / gas
def sigma_los_profile(rg, g, rho_star, Rp, beta_ani=0.2):
    """Projected l.o.s. velocity dispersion of the member tracer population.

    Spherical Jeans with constant anisotropy beta:
        rho sigma_r^2 (r) = r^-2b int_r^inf s^2b rho(s) g(s) ds
        Sigma sigma_los^2 (R) = 2 int_0^inf (1 - b R^2/r^2) rho sigma_r^2 r dt
    with the substitution r = R cosh t, which removes the sqrt singularity.
    """
    w = rg ** (2 * beta_ani) * rho_star * g
    seg = 0.5 * (w[1:] + w[:-1]) * np.diff(rg)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    rs2 = (cum[-1] - cum) / rg ** (2 * beta_ani)          # = rho * sigma_r^2
    t = np.linspace(0.0, 4.2, 64)
    Rp = np.atleast_1d(np.asarray(Rp, float))
    r = Rp[:, None] * np.cosh(t)[None, :]
    rc = np.clip(r, rg[0], rg[-1])
    rho = np.interp(rc, rg, rho_star)
    ps = np.interp(rc, rg, rs2)
    f = 1.0 - beta_ani * (Rp[:, None] / r) ** 2
    num = np.trapezoid(f * ps * r, t, axis=1)
    den = np.trapezoid(rho * r, t, axis=1)
    return np.sqrt(np.maximum(num / np.maximum(den, 1e-300), 1.0))


def gas_temperature(rg, g, rho_gas):
    """kT(r) [keV] from hydrostatic equilibrium in the universe's OWN field."""
    w = rho_gas * g
    seg = 0.5 * (w[1:] + w[:-1]) * np.diff(rg)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    P = cum[-1] - cum + rho_gas[-1] * g[-1] * rg[-1]     # outer boundary term
    return (P / np.maximum(rho_gas, 1e-300)) * KEV_PER_KMS2


# ================================================================== corpus
@dataclass
class Corpus:
    gal: list = field(default_factory=list)
    clu: list = field(default_factory=list)
    sn: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _sys(rng, u, key, base=None):
    s = u.sys_scale
    return rng.normal(0.0, s * (DN[key] if base is None else base))


def emit_galaxy(u, gal, rng, dm_p=None, nspax=26):
    """Detector-level emission for one disk galaxy."""
    dm = dm_galaxy(gal, u.params, rng) if u.uid == "U02_cdm" else None
    Rmax = 5.2 * gal.Rd
    # ---- physical field
    Rr = np.geomspace(0.08 * gal.Rd, Rmax, 40)
    f = ph.galaxy_field(u, gal, Rr, dm=dm)
    vc = np.sqrt(np.maximum(f["g_R"] * Rr, 1.0))

    # ---- IFU velocity field (detector level)
    ext = Rmax * 1.05
    ax = np.linspace(-ext, ext, nspax)
    Xs, Ys = np.meshgrid(ax, ax, indexing="ij")
    pa = np.deg2rad(gal.pa_deg)
    xr = Xs * np.cos(pa) + Ys * np.sin(pa)
    yr = -Xs * np.sin(pa) + Ys * np.cos(pa)
    inc = np.deg2rad(gal.incl_deg)
    yd = yr / max(np.cos(inc), 1e-3)
    Rd_ = np.sqrt(xr ** 2 + yd ** 2) + 1e-6
    th = np.arctan2(yd, xr)
    vR = np.interp(Rd_, Rr, vc)
    if np.any(f["quad_amp"] > 0):
        qa = np.interp(Rd_, Rr, f["quad_amp"])
        dphi = th - np.deg2rad(f["quad_pa"] - gal.pa_deg)
        vR = vR * np.sqrt(np.maximum(1.0 + qa * np.cos(2 * dphi), 0.05))
    vlos = vR * np.cos(th) * np.sin(inc)
    I = (gal.Md / (2 * np.pi * gal.Rd ** 2) * np.exp(-Rd_ / gal.Rd)
         + gal.Mg / (2 * np.pi * gal.Rg ** 2) * np.exp(-Rd_ / gal.Rg))
    # PSF convolution of the flux-weighted field, then divide
    kpc_per_arcsec = gal.dist_Mpc * 1e3 * (np.pi / 180.0 / 3600.0)
    sig_pix = (DN["ifu_psf_fwhm_arcsec"] / 2.355 * kpc_per_arcsec) / (ax[1] - ax[0])
    sig_pix = float(np.clip(sig_pix, 0.3, 4.0))
    Ic = gaussian_filter(I, sig_pix, mode="nearest")
    IVc = gaussian_filter(I * vlos, sig_pix, mode="nearest")
    vobs = IVc / np.maximum(Ic, 1e-30)
    sn_pix = np.sqrt(np.maximum(Ic / Ic.max(), 1e-6))
    everr = (DN["ifu_velocity_error_kms_at_1Re"] * u.noise_scale
             * u.sys_scale ** 0.5 / np.maximum(sn_pix, 0.03))
    everr = np.clip(everr, 3.0, 220.0)
    vobs = vobs + rng.normal(0.0, everr)
    mask = Ic > 1.5e-3 * Ic.max()

    # ---- vertical stellar dispersion at 1 and 2 R_d
    Rv = np.array([1.0, 2.0]) * gal.Rd
    fz = ph.galaxy_field(u, gal, Rv, dm=dm)
    sz = np.sqrt(np.maximum(fz["g_z"] * gal.hz, 1.0))
    sz_obs = sz * (1.0 + rng.normal(0, 0.09 * u.sys_scale * u.noise_scale, 2))

    # ---- observational nuisances on the BARYON model
    ml_err = rng.normal(0, DN["ml_dex_scatter"] * u.sys_scale)
    ml_grad = rng.normal(0, 0.05 * u.sys_scale)     # radial M/L gradient [dex/Rd]
    inc_obs = gal.incl_deg + rng.normal(0, DN["inclination_error_deg"] * u.sys_scale)
    d_obs = gal.dist_Mpc * (1.0 + rng.normal(0, DN["distance_frac_error"] * u.sys_scale))
    fd = d_obs / gal.dist_Mpc      # distance error propagates: L ~ d^2, R ~ d
    rad_as = np.pi / 180.0 / 3600.0
    ax_arcsec = ax / (gal.dist_Mpc * 1e3 * rad_as)

    return {
        "name": gal.name, "ax_arcsec": ax_arcsec, "v_map": vobs, "v_err": everr,
        "I_map": Ic, "mask": mask,
        "pa_obs": gal.pa_deg + rng.normal(0, 4.0 * u.sys_scale),
        "incl_obs": float(np.clip(inc_obs, 12.0, 87.0)),
        "dist_obs": float(d_obs),
        "Md_obs": gal.Md * 10 ** ml_err * fd ** 2,
        "Mg_obs": gal.Mg * 10 ** rng.normal(0, 0.10) * fd ** 2,
        "Mb_obs": gal.Mb * 10 ** ml_err * fd ** 2,
        "Rd_obs": gal.Rd * (1 + rng.normal(0, 0.05)) * fd,
        "Rg_obs": gal.Rg * fd, "ab_obs": gal.ab * fd,
        "hz_obs": gal.hz * (1 + rng.normal(0, 0.15)) * fd,
        "ml_grad_obs": float(ml_grad),
        "sz_obs": sz_obs, "Rv": Rv,
        "S_ext_obs": gal.S_ext * 10 ** rng.normal(0, 0.15),
        "axis_ext_obs": (gal.axis_ext_deg + rng.normal(0, 12.0)) % 180.0,
        "t_merge_proxy": gal.t_merge * 10 ** rng.normal(0, 0.25),
        "void_frac_obs": float(np.clip(gal.void_frac + rng.normal(0, 0.06), 0, 1)),
        "Rd_true_grid": Rr,
    }


def emit_cluster(u, geom, rng, nsrc=None):
    clu = geom.clu
    if nsrc is None:
        # n_eff sources per arcmin^2 over the observed field
        Dl_ = ph.D_A(clu.z) * 1e3
        kpc_per_arcmin = Dl_ * (np.pi / 180.0 / 60.0)
        area = np.pi * (2.4 * clu.R500 / kpc_per_arcmin) ** 2
        nsrc = int(np.clip(DN["wl_source_density_arcmin2"] * area, 800, 9000))
    dm = dm_cluster(clu, u.params, rng) if u.uid == "U02_cdm" else None
    rg = geom.rg
    F = ph.cluster_field(u, clu, rg, dm=dm)
    g_m, g_l = F["g_m"], F["g_l"]

    Bnet = 0.0
    if u.uid == "U06_wellnet":
        Bnet = u.params["B"]
        dEx = np.gradient(geom.Ex_bar, rg)
        g_m = g_m + Bnet * dEx
        g_l = g_l + Bnet * dEx

    # ---------------- weak lensing catalogue -----------------------------
    zs = np.clip(clu.z + 0.25 + rng.gamma(2.6, 0.28, nsrc), clu.z + 0.08, 3.4)
    Scr = ph.sigma_crit(clu.z, zs)
    kap_p, kbar_p = lens_profiles(rg, g_l, geom.bgrid, 1.0)   # per unit Sigma_cr
    Dl = ph.D_A(clu.z) * 1e3
    rmax = 2.4 * clu.R500
    rr = np.sqrt(rng.uniform((0.09 * clu.R500) ** 2, rmax ** 2, nsrc))
    pp = rng.uniform(0, 2 * np.pi, nsrc)
    sx, sy = rr * np.cos(pp), rr * np.sin(pp)
    kap = np.interp(rr, geom.bgrid, kap_p) / Scr
    kbar = np.interp(rr, geom.bgrid, kbar_p) / Scr
    gt = kbar - kap
    g1 = -gt * np.cos(2 * pp)
    g2 = -gt * np.sin(2 * pp)

    # quadrupole perturbations: tensor, network lumps, halo/baryon ellipticity
    ell_terms = []
    if u.uid == "U02_cdm":
        # a TRIAXIAL collisionless halo with a random orientation is the
        # principal false-anisotropy generator this suite has to survive
        Phi_h = ph.cluster_potential_1d(rg, G * dm["Mdm"](rg) / rg ** 2)
        ell_terms.append((0.5 * dm["ell"], dm["pa"], Phi_h))
    if u.uid == "U10_systematics":
        Phi_b = ph.cluster_potential_1d(rg, F["gN"])
        ell_terms.append((0.5 * clu.ell_bar * 1.6, clu.pa_bar_deg, Phi_b))
    Sxx, Syy, Sxy = quad_potential_map(geom, chi=F["chi"], Bnet=Bnet,
                                       ell_terms=ell_terms)
    if np.any(Sxx) or np.any(Sxy):
        a = _bilinear(Sxx, geom.xg, sx, sy)
        b = _bilinear(Syy, geom.xg, sx, sy)
        c = _bilinear(Sxy, geom.xg, sx, sy)
        kap = kap + (a + b) / Scr
        g1 = g1 + (a - b) / Scr
        g2 = g2 + 2.0 * c / Scr

    gred1 = g1 / np.maximum(1.0 - kap, 0.25)
    gred2 = g2 / np.maximum(1.0 - kap, 0.25)
    se = DN["wl_shape_noise_per_component"] * u.noise_scale
    m_bias = rng.normal(0, DN["wl_multiplicative_bias_sigma"] * u.sys_scale)
    c1 = rng.normal(0, DN["wl_additive_bias_sigma"] * u.sys_scale)
    c2 = rng.normal(0, DN["wl_additive_bias_sigma"] * u.sys_scale)
    # spatially coherent PSF residual (a real additive systematic)
    kx, ky = rng.normal(size=2) * 2.0 / rmax
    ca = DN["wl_additive_bias_sigma"] * 2.0 * u.sys_scale
    e1 = (1 + m_bias) * gred1 + c1 + ca * np.cos(kx * sx + ky * sy) + rng.normal(0, se, nsrc)
    e2 = (1 + m_bias) * gred2 + c2 + ca * np.sin(kx * sx + ky * sy) + rng.normal(0, se, nsrc)
    zph = zs * (1 + rng.normal(DN["wl_photoz_mean_bias_sigma"] * u.sys_scale, 0.035, nsrc))
    nout = int(DN["wl_photoz_outlier_fraction"] * u.sys_scale * nsrc)
    if nout > 0:
        oi = rng.choice(nsrc, size=min(nout, nsrc), replace=False)
        zph[oi] = rng.uniform(clu.z + 0.05, 3.0, len(oi))
    wgt = 1.0 / (se ** 2 + 0.09)

    # ---------------- member galaxies ------------------------------------
    rmem = np.linalg.norm(clu.mem_xyz, axis=1)
    hh, ed = np.histogram(np.log(np.clip(rmem, rg[0], rg[-1])), bins=18,
                          range=(np.log(rg[0]), np.log(rg[-1])))
    ctr = np.exp(0.5 * (ed[1:] + ed[:-1]))
    dens = hh / (4 * np.pi * ctr ** 3 * np.diff(ed))
    rho_star = np.interp(np.log(rg), np.log(ctr), np.maximum(dens, 1e-12))
    beta_ani = float(np.clip(rng.normal(0.22, 0.14 * u.sys_scale), -0.3, 0.6))
    Rp_mem = np.sqrt((clu.mem_xyz[:, :2] ** 2).sum(1))
    Rp_mem = np.clip(Rp_mem, 0.05 * clu.R500, 2.2 * clu.R500)
    sig = sigma_los_profile(rg, g_m, rho_star, np.unique(np.round(Rp_mem, 1)),
                            beta_ani=beta_ani)
    sig_at = np.interp(Rp_mem, np.unique(np.round(Rp_mem, 1)), sig)
    vmem = rng.normal(0, sig_at) + rng.normal(0, DN["member_velocity_error_kms"] * u.noise_scale, len(sig_at))
    pmem = np.clip(rng.beta(9, 1.1, len(vmem)), 0, 1)

    # ---------------- X-ray -----------------------------------------------
    rann = np.geomspace(0.08, 1.4, 13) * clu.R500
    rho_gas = clu.Mgas_enc(rg)
    rho_gas = np.gradient(rho_gas, rg) / (4 * np.pi * rg ** 2)
    rho_gas = np.maximum(rho_gas, 1e-14)
    kT = gas_temperature(rg, g_m, rho_gas)
    kTa = np.interp(rann, rg, kT)
    # non-thermal pressure support: a real astrophysical systematic
    fnt = np.clip(0.06 * u.sys_scale * (rann / clu.R500) ** 0.8
                  + rng.normal(0, 0.02 * u.sys_scale), 0.0, 0.45)
    kT_obs = kTa * (1 - fnt) * (1 + rng.normal(0, DN["xray_kT_frac_error"] * u.noise_scale, len(rann)))
    ne = np.interp(rann, rg, rho_gas)                 # arbitrary normalisation
    rate = ne ** 2 * np.sqrt(np.maximum(kTa, 0.4)) * rann ** 3
    cnts = rng.poisson(np.maximum(rate / rate.max() * DN["xray_counts_per_annulus_ref"] / u.noise_scale ** 2, 1.0))
    y_sz = ne * kTa * clu.R500
    y_obs = y_sz * (1 + rng.normal(0, DN["sz_y_frac_error"] * u.sys_scale, len(rann)))

    # ---------------- strong lensing ---------------------------------------
    zsl = float(clu.z + 0.6 + rng.gamma(2.0, 0.4))
    Scr_sl = float(ph.sigma_crit(clu.z, zsl))
    kb = np.interp(geom.bgrid, geom.bgrid, kbar_p) / Scr_sl
    thE = 0.0
    if np.any(kb > 1):
        j = int(np.argmax(kb <= 1.0)) if np.any(kb <= 1.0) else len(kb) - 1
        j = max(j, 1)
        thE = float(np.interp(1.0, [kb[j], kb[j - 1]], [geom.bgrid[j], geom.bgrid[j - 1]]))
    fams, delays = [], []
    if thE > 0.06 * clu.R500:
        alpha_b = np.interp(geom.bgrid, geom.bgrid, kbar_p) / Scr_sl * geom.bgrid
        qamp = 0.0
        if F["chi"] is not None:
            qamp = float(np.interp(thE, rg, F["chi"]) /
                         max(abs(np.interp(thE, rg, ph.cluster_potential_1d(rg, g_l))), 1e-9))
        for k in range(4):
            phis = rng.uniform(0, 2 * np.pi)
            beta = rng.uniform(0.02, 0.35) * thE
            mod = 1.0 + qamp * np.cos(2 * (phis - np.deg2rad(clu.axis_ext_deg)))
            th = np.geomspace(0.25 * thE, 2.6 * thE, 400)
            al = np.interp(th, geom.bgrid, alpha_b) * mod
            fmin = th - al - beta
            s = np.sign(fmin)
            roots = th[:-1][s[:-1] != s[1:]]
            if len(roots) == 0:
                continue
            tp = float(roots[-1])
            tm = float(max(thE ** 2 / max(tp, 1e-6), 0.2 * thE))
            asec = 1.0 / (Dl * (np.pi / 180 / 3600))
            perr = DN["sl_image_position_error_arcsec"] * u.sys_scale
            fams.append([tp * asec + rng.normal(0, perr),
                         tm * asec + rng.normal(0, perr),
                         float(np.rad2deg(phis)), zsl])
            psi = np.concatenate(([0.0], np.cumsum(
                0.5 * (al[1:] + al[:-1]) * np.diff(th))))
            fer = lambda t: (0.5 * (t - beta) ** 2 - np.interp(t, th, psi))
            dt = (1 + clu.z) / (C_KMS) * (Dl * ph.D_A(zsl) / max(
                ph.D_A12(clu.z, zsl), 1e-6) / 1e3) * (fer(tm) - fer(tp)) / Dl ** 2
            dt = dt * 3.0857e16 / 86400.0
            delays.append(float(dt * (1 + rng.normal(0, DN["sl_time_delay_frac_error"]))))

    # ---------------- observed environment ---------------------------------
    ax_obs = (clu.axis_ext_deg + rng.normal(0, 10.0)) % 180.0
    pa_bar_obs = (clu.pa_bar_deg + rng.normal(0, 7.0)) % 180.0
    mis = rng.normal(0, 25.0 * u.sys_scale)          # miscentring [kpc]
    mem_xy_obs = clu.mem_xyz[:, :2] + rng.normal(0, 3.0, (len(clu.mem_m), 2)) + mis
    mem_m_obs = clu.mem_m * 10 ** rng.normal(0, 0.13 * u.sys_scale, len(clu.mem_m))

    return {
        "name": clu.name, "z": clu.z, "R500": clu.R500,
        "src_x": sx, "src_y": sy, "e1": e1, "e2": e2, "w": np.full(nsrc, wgt),
        "z_src_phot": zph,
        "mem_x": mem_xy_obs[:, 0], "mem_y": mem_xy_obs[:, 1],
        "mem_v": vmem, "mem_p": pmem, "mem_m_obs": mem_m_obs,
        "r_ann": rann, "xray_counts": cnts, "kT_obs": kT_obs,
        "kT_err": np.maximum(kT_obs * DN["xray_kT_frac_error"], 0.05),
        "y_sz": y_obs,
        "Mgas_obs": clu.Mgas_enc(rann) * 10 ** rng.normal(0, 0.04 * u.sys_scale),
        "Mstar_obs": (np.array([mem_m_obs[np.linalg.norm(clu.mem_xyz, axis=1) <= r].sum()
                                for r in rann])
                      + hernquist_M(rann, clu.M_bcg + clu.M_icl, clu.a_bcg + clu.a_icl))
        * 10 ** rng.normal(0, 0.11 * u.sys_scale),
        "sl_fams": np.array(fams) if fams else np.zeros((0, 4)),
        "sl_delays": np.array(delays), "thetaE_kpc": thE, "z_sl": zsl,
        "axis_ext_obs": ax_obs, "pa_bar_obs": pa_bar_obs, "ell_bar_obs": clu.ell_bar,
        "t_merge_proxy": clu.centroid_shift, "gas_gal_offset_obs":
            clu.gas_gal_offset * 10 ** rng.normal(0, 0.12),
        "void_frac_obs": float(np.clip(clu.void_frac + rng.normal(0, 0.06), 0, 1)),
        "Sig_cr_ref": float(ph.sigma_crit(clu.z, clu.z + 0.9)),
        "geom_idx": None,
    }


def emit_sn(u, rng, n=200):
    z_c = np.clip(rng.gamma(2.0, 0.18, n), 0.012, 1.4)
    void = np.clip(rng.beta(2.4, 2.0, n), 0.01, 0.99)
    onepz, stretch = ph.observed_redshift(u, z_c, void)
    z_obs = onepz - 1.0
    dl = ph.comoving_Mpc(z_c) * (1 + z_c)
    mu = 5 * np.log10(np.maximum(dl, 1e-3)) + 25.0
    mag = mu + rng.normal(0, DN["sn_peak_mag_scatter"] * u.sys_scale * u.noise_scale, n)
    dur = 20.0 * stretch * (1 + rng.normal(0, DN["sn_duration_frac_error"] * u.noise_scale, n))
    return {"z_obs": z_obs, "mag": mag, "duration": dur, "void_frac": void}


def draw_corpus(u, lib, rng, n_gal=34, n_clu=7, n_sn=200):
    gi = rng.choice(len(lib.galaxies), size=n_gal, replace=False)
    ci = rng.choice(len(lib.geoms), size=n_clu, replace=False)
    C = Corpus()
    for i in gi:
        C.gal.append(emit_galaxy(u, lib.galaxies[i], rng))
    for i in ci:
        d = emit_cluster(u, lib.geoms[i], rng)
        d["geom_idx"] = int(i)
        C.clu.append(d)
    C.sn = emit_sn(u, rng, n=n_sn)
    C.meta = {"uid": u.uid, "label": u.label, "sys_scale": u.sys_scale}
    return C
