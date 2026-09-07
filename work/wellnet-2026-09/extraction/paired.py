"""paired.py -- generator 1: Run BF's physics, emitted in PAIRED form.

Protocol item 1 of the Extraction Lane: "pair every universe on exactly the
same baryonic scene and detector noise".  Run BF's ``corpus.emit_*`` draw the
halo, the photon-path and every nuisance from ONE random stream, so a CDM
corpus consumes extra draws (its halo) and a supercritical cluster consumes
extra draws (its image families), and the noise realisations of two universes
on the same scene are never the same.  This module re-emits BF's instrument
with

  * one independent stream PER NOISE BLOCK, seeded by (set seed, object key,
    block id), so that a given scene under U3 and under U2 carries the SAME
    shape noise, the SAME photo-z outliers, the SAME velocity-field noise and
    the SAME photometric errors -- the only difference between the two
    catalogues is the law;
  * the dark-matter halo passed in EXPLICITLY (drawn by ``halo.py`` under a
    declared prior), so 'move the halo holding the baryons fixed' and 'move
    the baryons holding the halo fixed' are one-line counterfactuals;
  * the law itself unchanged: ``physics.galaxy_field`` / ``physics.cluster_field``
    for U1, U3-U10, and a CDM galaxy field written here only because BF's has
    no halo flattening, in-plane quadrupole or dark-disc degree of freedom.

Everything numerical about the instrument -- PSF, source density, noise
amplitudes, photo-z model, X-ray counts model, strong-lens image finder --
is BF's (imported or transcribed), so a corpus emitted here for a given
universe is drawn from the same distribution as BF's, only paired.
"""
from __future__ import annotations

import copy
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter

from universes import corpus as cp
from universes import physics as ph
from universes.baryons import C_KMS, G, hernquist_M
from universes.provenance import DECLARED_NOISE as DN

import halo as H

BLOCKS = {"ifu": 1, "sz": 2, "phot": 3, "src": 11, "shear": 12, "photoz": 13,
          "mem": 14, "xray": 15, "sl": 16, "env": 17, "sn": 21, "halo": 31,
          "universe": 41, "scene": 42}


def stream(seed, key, block):
    """One independent Generator per (set, object, noise block)."""
    return np.random.default_rng([int(seed) & 0x7FFFFFFF, int(key) & 0x7FFFFFFF,
                                  BLOCKS[block]])


def _key(name):
    return zlib.crc32(str(name).encode()) & 0x7FFFFFFF


# ======================================================================
# the CDM galaxy field, with the halo's own degrees of freedom
# ======================================================================
def galaxy_field_cdm(gal, R, dm):
    """Radial and vertical accelerations of a disc inside a collisionless halo.

    BF's version: g_R = g_N + G M_h/R^2, g_z = g_zN + G M_h/R^2 * h_z/R.
    Added here: oblateness q_h, dark-disc fraction f_dd (halo.halo_vertical_field)
    and an in-plane m=2 modulation of v_c^2 of amplitude q_amp locked to the
    halo's own axis (BK's galaxy-halo model, absent from BF's generator).
    """
    R = np.asarray(R, float)
    gN = gal.gN(R)
    Sig = gal.Sigma(R)
    gzN = 2.0 * np.pi * G * Sig
    gh = G * dm["Mdm"](R) / R ** 2
    gR = gN + gh
    gz = gzN + H.halo_vertical_field(R, gal.hz, dm)
    rt = 2.0 * gal.Rd
    f = (R / rt) ** 2 / (1.0 + (R / rt) ** 2)
    quad = dm.get("q_amp", 0.0) * f * (gh / np.maximum(gR, 1e-12))
    return {"gN": gN, "g_R": gR, "g_Rl": gR, "g_z": gz,
            "quad_amp": quad, "quad_pa": gal.pa_deg + dm.get("psi_deg", 0.0),
            "nu": gR / gN}


# ======================================================================
# galaxies
# ======================================================================
def emit_galaxy(u, gal, seed, dm=None, nuis=None, nspax=26):
    """Detector-level emission for one disc galaxy (transcribed from BF's
    corpus.emit_galaxy; noise blocks separated)."""
    nuis = nuis or {}
    key = _key(gal.name)
    S_ifu, S_sz, S_ph = (stream(seed, key, b) for b in ("ifu", "sz", "phot"))
    Rmax = 5.2 * gal.Rd
    Rr = np.geomspace(0.08 * gal.Rd, Rmax, 40)
    if u.uid == "U02_cdm":
        f = galaxy_field_cdm(gal, Rr, dm)
    else:
        f = ph.galaxy_field(u, gal, Rr, dm=None)
    vc = np.sqrt(np.maximum(f["g_R"] * Rr, 1.0))

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
    vobs = vobs + everr * S_ifu.standard_normal(vobs.shape)
    mask = Ic > 1.5e-3 * Ic.max()

    Rv = np.array([1.0, 2.0]) * gal.Rd
    if u.uid == "U02_cdm":
        fz = galaxy_field_cdm(gal, Rv, dm)
    else:
        fz = ph.galaxy_field(u, gal, Rv, dm=None)
    sz = np.sqrt(np.maximum(fz["g_z"] * gal.hz, 1.0))
    s_sz = 0.09 * u.sys_scale * u.noise_scale * nuis.get("sz_err", 1.0)
    sz_obs = sz * (1.0 + s_sz * S_sz.standard_normal(2))

    ml_err = S_ph.normal(0, DN["ml_dex_scatter"] * u.sys_scale * nuis.get("ml", 1.0))
    ml_grad = S_ph.normal(0, 0.05 * u.sys_scale * nuis.get("ml", 1.0))
    inc_obs = gal.incl_deg + S_ph.normal(0, DN["inclination_error_deg"] * u.sys_scale
                                         * nuis.get("incl", 1.0))
    d_obs = gal.dist_Mpc * max(1.0 + S_ph.normal(0, DN["distance_frac_error"] * u.sys_scale
                                                 * nuis.get("dist", 1.0)), 0.3)
    fd = d_obs / gal.dist_Mpc
    rad_as = np.pi / 180.0 / 3600.0
    ax_arcsec = ax / (gal.dist_Mpc * 1e3 * rad_as)
    return {
        "name": gal.name, "ax_arcsec": ax_arcsec, "v_map": vobs, "v_err": everr,
        "I_map": Ic, "mask": mask,
        "pa_obs": gal.pa_deg + S_ph.normal(0, 4.0 * u.sys_scale),
        "incl_obs": float(np.clip(inc_obs, 12.0, 87.0)),
        "dist_obs": float(d_obs),
        "Md_obs": gal.Md * 10 ** ml_err * fd ** 2,
        "Mg_obs": gal.Mg * 10 ** S_ph.normal(0, 0.10) * fd ** 2,
        "Mb_obs": gal.Mb * 10 ** ml_err * fd ** 2,
        "Rd_obs": gal.Rd * (1 + S_ph.normal(0, 0.05)) * fd,
        "Rg_obs": gal.Rg * fd, "ab_obs": gal.ab * fd,
        "hz_obs": gal.hz * (1 + S_ph.normal(0, 0.15 * nuis.get("hz", 1.0))) * fd,
        "ml_grad_obs": float(ml_grad),
        "sz_obs": sz_obs, "Rv": Rv,
        "S_ext_obs": gal.S_ext * 10 ** S_ph.normal(0, 0.15),
        "axis_ext_obs": (gal.axis_ext_deg + S_ph.normal(0, 12.0)) % 180.0,
        "t_merge_proxy": gal.t_merge * 10 ** S_ph.normal(0, 0.25),
        "void_frac_obs": float(np.clip(gal.void_frac + S_ph.normal(0, 0.06), 0, 1)),
        # truths, for the response analysis only (never seen by the analysis)
        "_truth": {"hz": gal.hz, "Rd": gal.Rd, "incl": gal.incl_deg,
                   "gz_true": sz ** 2 / gal.hz, "gR_true": np.interp(Rv, Rr, f["g_R"]),
                   "gzN": 2 * np.pi * G * gal.Sigma(Rv), "gN": gal.gN(Rv)},
    }


# ======================================================================
# clusters
# ======================================================================
def emit_cluster(u, geom, seed, dm=None, nsrc=None):
    """Transcribed from BF's corpus.emit_cluster with separated noise blocks."""
    clu = geom.clu
    key = _key(clu.name)
    S_src, S_sh, S_pz, S_mem, S_x, S_sl, S_env = (
        stream(seed, key, b) for b in ("src", "shear", "photoz", "mem", "xray", "sl", "env"))
    if nsrc is None:
        Dl_ = ph.D_A(clu.z) * 1e3
        kpc_per_arcmin = Dl_ * (np.pi / 180.0 / 60.0)
        area = np.pi * (2.4 * clu.R500 / kpc_per_arcmin) ** 2
        nsrc = int(np.clip(DN["wl_source_density_arcmin2"] * area, 800, 9000))
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
    zs = np.clip(clu.z + 0.25 + S_src.gamma(2.6, 0.28, nsrc), clu.z + 0.08, 3.4)
    Scr = ph.sigma_crit(clu.z, zs)
    kap_p, kbar_p = cp.lens_profiles(rg, g_l, geom.bgrid, 1.0)
    Dl = ph.D_A(clu.z) * 1e3
    rmax = 2.4 * clu.R500
    rr = np.sqrt(S_src.uniform((0.09 * clu.R500) ** 2, rmax ** 2, nsrc))
    pp = S_src.uniform(0, 2 * np.pi, nsrc)
    sx, sy = rr * np.cos(pp), rr * np.sin(pp)
    kap = np.interp(rr, geom.bgrid, kap_p) / Scr
    kbar = np.interp(rr, geom.bgrid, kbar_p) / Scr
    gt = kbar - kap
    g1 = -gt * np.cos(2 * pp)
    g2 = -gt * np.sin(2 * pp)

    ell_terms = []
    if u.uid == "U02_cdm":
        Phi_h = ph.cluster_potential_1d(rg, G * dm["Mdm"](rg) / rg ** 2)
        ell_terms.append((0.5 * dm["ell"], dm["pa"], Phi_h))
    if u.uid == "U10_systematics":
        Phi_b = ph.cluster_potential_1d(rg, F["gN"])
        ell_terms.append((0.5 * clu.ell_bar * 1.6, clu.pa_bar_deg, Phi_b))
    Sxx, Syy, Sxy = cp.quad_potential_map(geom, chi=F["chi"], Bnet=Bnet,
                                          ell_terms=ell_terms)
    if np.any(Sxx) or np.any(Sxy):
        a = cp._bilinear(Sxx, geom.xg, sx, sy)
        b = cp._bilinear(Syy, geom.xg, sx, sy)
        c = cp._bilinear(Sxy, geom.xg, sx, sy)
        kap = kap + (a + b) / Scr
        g1 = g1 + (a - b) / Scr
        g2 = g2 + 2.0 * c / Scr

    gred1 = g1 / np.maximum(1.0 - kap, 0.25)
    gred2 = g2 / np.maximum(1.0 - kap, 0.25)
    se = DN["wl_shape_noise_per_component"] * u.noise_scale
    m_bias = S_sh.normal(0, DN["wl_multiplicative_bias_sigma"] * u.sys_scale)
    c1 = S_sh.normal(0, DN["wl_additive_bias_sigma"] * u.sys_scale)
    c2 = S_sh.normal(0, DN["wl_additive_bias_sigma"] * u.sys_scale)
    kx, ky = S_sh.normal(size=2) * 2.0 / rmax
    ca = DN["wl_additive_bias_sigma"] * 2.0 * u.sys_scale
    n1 = S_sh.standard_normal(nsrc)
    n2 = S_sh.standard_normal(nsrc)
    e1 = (1 + m_bias) * gred1 + c1 + ca * np.cos(kx * sx + ky * sy) + se * n1
    e2 = (1 + m_bias) * gred2 + c2 + ca * np.sin(kx * sx + ky * sy) + se * n2
    zph = zs * (1 + S_pz.normal(DN["wl_photoz_mean_bias_sigma"] * u.sys_scale, 0.035, nsrc))
    nout = int(DN["wl_photoz_outlier_fraction"] * u.sys_scale * nsrc)
    if nout > 0:
        oi = S_pz.choice(nsrc, size=min(nout, nsrc), replace=False)
        zph[oi] = S_pz.uniform(clu.z + 0.05, 3.0, len(oi))
    wgt = 1.0 / (se ** 2 + 0.09)

    # ---------------- member galaxies ------------------------------------
    rmem = np.linalg.norm(clu.mem_xyz, axis=1)
    hh, ed = np.histogram(np.log(np.clip(rmem, rg[0], rg[-1])), bins=18,
                          range=(np.log(rg[0]), np.log(rg[-1])))
    ctr = np.exp(0.5 * (ed[1:] + ed[:-1]))
    dens = hh / (4 * np.pi * ctr ** 3 * np.diff(ed))
    rho_star = np.interp(np.log(rg), np.log(ctr), np.maximum(dens, 1e-12))
    beta_ani = float(np.clip(S_mem.normal(0.22, 0.14 * u.sys_scale), -0.3, 0.6))
    Rp_mem = np.sqrt((clu.mem_xyz[:, :2] ** 2).sum(1))
    Rp_mem = np.clip(Rp_mem, 0.05 * clu.R500, 2.2 * clu.R500)
    uq = np.unique(np.round(Rp_mem, 1))
    sig = cp.sigma_los_profile(rg, g_m, rho_star, uq, beta_ani=beta_ani)
    sig_at = np.interp(Rp_mem, uq, sig)
    nm = len(sig_at)
    vmem = sig_at * S_mem.standard_normal(nm) \
        + DN["member_velocity_error_kms"] * u.noise_scale * S_mem.standard_normal(nm)
    pmem = np.clip(S_mem.beta(9, 1.1, nm), 0, 1)

    # ---------------- X-ray -----------------------------------------------
    rann = np.geomspace(0.08, 1.4, 13) * clu.R500
    rho_gas = clu.Mgas_enc(rg)
    rho_gas = np.gradient(rho_gas, rg) / (4 * np.pi * rg ** 2)
    rho_gas = np.maximum(rho_gas, 1e-14)
    kT = cp.gas_temperature(rg, g_m, rho_gas)
    kTa = np.interp(rann, rg, kT)
    fnt = np.clip(0.06 * u.sys_scale * (rann / clu.R500) ** 0.8
                  + S_x.normal(0, 0.02 * u.sys_scale), 0.0, 0.45)
    kT_obs = kTa * (1 - fnt) * (1 + DN["xray_kT_frac_error"] * u.noise_scale
                                * S_x.standard_normal(len(rann)))
    ne = np.interp(rann, rg, rho_gas)
    rate = ne ** 2 * np.sqrt(np.maximum(kTa, 0.4)) * rann ** 3
    cnts = S_x.poisson(np.maximum(rate / rate.max() * DN["xray_counts_per_annulus_ref"]
                                  / u.noise_scale ** 2, 1.0))
    y_sz = ne * kTa * clu.R500
    y_obs = y_sz * (1 + DN["sz_y_frac_error"] * u.sys_scale * S_x.standard_normal(len(rann)))

    # ---------------- strong lensing ---------------------------------------
    zsl = float(clu.z + 0.6 + S_sl.gamma(2.0, 0.4))
    Scr_sl = float(ph.sigma_crit(clu.z, zsl))
    kb = kbar_p / Scr_sl
    thE = 0.0
    if np.any(kb > 1):
        j = int(np.argmax(kb <= 1.0)) if np.any(kb <= 1.0) else len(kb) - 1
        j = max(j, 1)
        thE = float(np.interp(1.0, [kb[j], kb[j - 1]], [geom.bgrid[j], geom.bgrid[j - 1]]))
    fams, delays = [], []
    # the image-family draws are taken from their own stream so that a
    # supercritical cluster consumes nothing another block would have used
    fam_draws = [(S_sl.uniform(0, 2 * np.pi), S_sl.uniform(0.02, 0.35),
                  S_sl.normal(0, 1.0), S_sl.normal(0, 1.0), S_sl.normal(0, 1.0))
                 for _ in range(4)]
    if thE > 0.06 * clu.R500:
        alpha_b = kbar_p / Scr_sl * geom.bgrid
        qamp = 0.0
        if F["chi"] is not None:
            qamp = float(np.interp(thE, rg, F["chi"]) /
                         max(abs(np.interp(thE, rg, ph.cluster_potential_1d(rg, g_l))), 1e-9))
        for (phis, bfac, n1_, n2_, n3_) in fam_draws:
            beta = bfac * thE
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
            fams.append([tp * asec + perr * n1_, tm * asec + perr * n2_,
                         float(np.rad2deg(phis)), zsl])
            psi = np.concatenate(([0.0], np.cumsum(
                0.5 * (al[1:] + al[:-1]) * np.diff(th))))
            fer = lambda t: (0.5 * (t - beta) ** 2 - np.interp(t, th, psi))
            dt = (1 + clu.z) / (C_KMS) * (Dl * ph.D_A(zsl) / max(
                ph.D_A12(clu.z, zsl), 1e-6) / 1e3) * (fer(tm) - fer(tp)) / Dl ** 2
            dt = dt * 3.0857e16 / 86400.0
            delays.append(float(dt * (1 + DN["sl_time_delay_frac_error"] * n3_)))

    # ---------------- observed environment ---------------------------------
    ax_obs = (clu.axis_ext_deg + S_env.normal(0, 10.0)) % 180.0
    pa_bar_obs = (clu.pa_bar_deg + S_env.normal(0, 7.0)) % 180.0
    mis = S_env.normal(0, 25.0 * u.sys_scale)
    mem_xy_obs = clu.mem_xyz[:, :2] + S_env.normal(0, 3.0, (len(clu.mem_m), 2)) + mis
    mem_m_obs = clu.mem_m * 10 ** S_env.normal(0, 0.13 * u.sys_scale, len(clu.mem_m))
    Mgas_obs = clu.Mgas_enc(rann) * 10 ** S_env.normal(0, 0.04 * u.sys_scale)
    Mstar_obs = (np.array([mem_m_obs[np.linalg.norm(clu.mem_xyz, axis=1) <= r].sum()
                           for r in rann])
                 + hernquist_M(rann, clu.M_bcg + clu.M_icl, clu.a_bcg + clu.a_icl)) \
        * 10 ** S_env.normal(0, 0.11 * u.sys_scale)
    return {
        "name": clu.name, "z": clu.z, "R500": clu.R500,
        "src_x": sx, "src_y": sy, "e1": e1, "e2": e2, "w": np.full(nsrc, wgt),
        "z_src_phot": zph,
        "mem_x": mem_xy_obs[:, 0], "mem_y": mem_xy_obs[:, 1],
        "mem_v": vmem, "mem_p": pmem, "mem_m_obs": mem_m_obs,
        "r_ann": rann, "xray_counts": cnts, "kT_obs": kT_obs,
        "kT_err": np.maximum(kT_obs * DN["xray_kT_frac_error"], 0.05),
        "y_sz": y_obs, "Mgas_obs": Mgas_obs, "Mstar_obs": Mstar_obs,
        "sl_fams": np.array(fams) if fams else np.zeros((0, 4)),
        "sl_delays": np.array(delays), "thetaE_kpc": thE, "z_sl": zsl,
        "axis_ext_obs": ax_obs, "pa_bar_obs": pa_bar_obs, "ell_bar_obs": clu.ell_bar,
        "t_merge_proxy": clu.centroid_shift,
        "gas_gal_offset_obs": clu.gas_gal_offset * 10 ** S_env.normal(0, 0.12),
        "void_frac_obs": float(np.clip(clu.void_frac + S_env.normal(0, 0.06), 0, 1)),
        "Sig_cr_ref": float(ph.sigma_crit(clu.z, clu.z + 0.9)),
        "_truth": {"g_m": g_m, "g_l": g_l, "rg": rg, "Mbar500": clu.Mbar500,
                   "halo": None if dm is None else {k: dm[k] for k in dm if k != "Mdm"}},
    }


def emit_sn(u, seed, n=200):
    S = stream(seed, 0, "sn")
    z_c = np.clip(S.gamma(2.0, 0.18, n), 0.012, 1.4)
    void = np.clip(S.beta(2.4, 2.0, n), 0.01, 0.99)
    onepz, stretch = ph.observed_redshift(u, z_c, void)
    z_obs = onepz - 1.0
    dl = ph.comoving_Mpc(z_c) * (1 + z_c)
    mu = 5 * np.log10(np.maximum(dl, 1e-3)) + 25.0
    mag = mu + DN["sn_peak_mag_scatter"] * u.sys_scale * u.noise_scale * S.standard_normal(n)
    dur = 20.0 * stretch * (1 + DN["sn_duration_frac_error"] * u.noise_scale * S.standard_normal(n))
    return {"z_obs": z_obs, "mag": mag, "duration": dur, "void_frac": void}


# ======================================================================
# counterfactual scene edits (baryons moved, halo held elsewhere)
# ======================================================================
def perturb_galaxy(gal, cf):
    g = copy.copy(gal)
    if cf.get("dlogM"):
        s = 10 ** cf["dlogM"]
        g.Md, g.Mg, g.Mb = gal.Md * s, gal.Mg * s, gal.Mb * s
    if cf.get("dlogR"):
        s = 10 ** cf["dlogR"]
        g.Rd, g.Rg, g.ab, g.hz = gal.Rd * s, gal.Rg * s, gal.ab * s, gal.hz * s
        g.R_bnd = 10.0 * g.Rd
    if cf.get("dlog_hz"):
        g.hz = gal.hz * 10 ** cf["dlog_hz"]
    if cf.get("axis_rot"):
        g.axis_ext_deg = (gal.axis_ext_deg + cf["axis_rot"]) % 180.0
    if cf.get("tmerge_mult"):
        g.t_merge = gal.t_merge * cf["tmerge_mult"]
    if cf.get("void_flip"):
        g.void_frac = 1.0 - gal.void_frac
    return g


def perturb_cluster_scene(clu, cf):
    c = copy.copy(clu)
    if cf.get("dlogM"):
        s = 10 ** cf["dlogM"]
        c.Mgas, c.M_bcg, c.M_icl = clu.Mgas * s, clu.M_bcg * s, clu.M_icl * s
        c.mem_m = clu.mem_m * s
        c.__post_init__()
    if cf.get("dlogR"):
        s = 10 ** cf["dlogR"]
        c.rc_gas, c.a_bcg, c.a_icl = clu.rc_gas * s, clu.a_bcg * s, clu.a_icl * s
        c.mem_xyz = clu.mem_xyz * s
        c.__post_init__()
    if cf.get("axis_rot"):
        c.axis_ext_deg = (clu.axis_ext_deg + cf["axis_rot"]) % 180.0
    if cf.get("tmerge_mult"):
        c.t_merge = clu.t_merge * cf["tmerge_mult"]
        c.gas_gal_offset = clu.gas_gal_offset * cf["tmerge_mult"] ** -0.5
        c.centroid_shift = clu.centroid_shift * cf["tmerge_mult"] ** -0.5
    if cf.get("void_flip"):
        c.void_frac = 1.0 - clu.void_frac
    return c


def geom_with_scene(geom, clu2):
    """A geometry object carrying an edited scene.  The precomputed
    well-network field is kept unless the members moved (a scramble), in which
    case the caller must supply a rebuilt geometry.  The direction cosines to
    the EXTERNAL axis (u3, used by the tensor universe's P2) are rebuilt when
    the axis moved -- the first version kept the stale grid, so the tensor's
    quadrupole stayed on the old axis while the observed axis rotated (caught
    by the counterfactual table: U5f responded to its own axis rotation)."""
    g = copy.copy(geom)
    g.clu = clu2
    if abs(((clu2.axis_ext_deg - geom.clu.axis_ext_deg) + 90.0) % 180.0 - 90.0) > 1e-9:
        X, Y, Z = np.meshgrid(geom.xg, geom.xg, geom.zg, indexing="ij")
        ang = np.deg2rad(clu2.axis_ext_deg)
        g.u3 = (X * np.cos(ang) + Y * np.sin(ang)) / geom.r3
    return g


# ======================================================================
# a PAIRED SET: one scene draw, one noise realisation, many universes
# ======================================================================
def make_universe(uid, knob, sys_scale, noise_scale, seed):
    """Universe constants drawn from the 'universe' stream of the set, so that
    every scalar universe in a paired set carries the same a0 draw."""
    rng = stream(seed, 0, "universe")
    if uid == "H0_scalar_null":
        u = ph.draw_scalar_null_universe(rng, sys_scale=sys_scale)
    else:
        u = ph.draw_universe(uid, rng, knob=knob, sys_scale=sys_scale)
    u.noise_scale = float(noise_scale)
    return u


def draw_scene_indices(seed, n_lib_gal, n_lib_clu, n_gal, n_clu):
    rng = stream(seed, 0, "scene")
    gi = rng.choice(n_lib_gal, size=n_gal, replace=False)
    ci = rng.choice(n_lib_clu, size=n_clu, replace=False)
    return gi, ci


def draw_halos(seed, gals, clus, p, k):
    """Halo draws for a set: seeded by (set, object, 'halo'); with
    fix_halo_per_object the set seed is dropped so a library object keeps ONE
    halo across every set (the 'halo held fixed across noise realisations'
    arm of the closure test)."""
    out_g, out_c = [], []
    for gal in gals:
        s = 0 if k.get("fix_halo_per_object") else seed
        out_g.append(H.draw_galaxy_halo(gal, p, stream(s, _key(gal.name), "halo"), k))
    for clu in clus:
        s = 0 if k.get("fix_halo_per_object") else seed
        out_c.append(H.draw_cluster_halo(clu, p, stream(s, _key(clu.name), "halo"), k))
    return out_g, out_c
