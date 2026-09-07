"""forward2.py -- GENERATOR 2: an independently implemented universe pair.

Protocol item 1: anything the extraction finds must be validated across two
generators that share nothing; what does not transfer is a simulator
fingerprint, not physics.  Run BF's generator (generator 1) builds Freeman
exponential discs + Hernquist bulges, NFW haloes drawn from a Behroozi-like
SHMR, beta-model gas, a 3-D Cartesian lensing grid, a cosh-substitution
projection, a constant-anisotropy Jeans solve and a tanh line-of-sight grid.
This module builds instead:

  discs      Miyamoto-Nagai stellar and gas discs (closed-form radial AND
             vertical fields), a Plummer bulge; the light the instrument sees
             is exponential, so the analyst's photometric model is mildly
             misspecified against the true potential, as it is in reality
  haloes     EINASTO (alpha = 0.17) with a quadratic SHMR, its own c(M),
             oblateness q_h, a dark-disc fraction f_dd, an in-plane m=2 of
             amplitude q_amp on the f_lss-mixed axis (halo.mixed_axis is the
             one shared line: it is the DEFINITION of the nuisance)
  the class  a universal local law g = nu(|g_N|/a0) g_N applied to the full
             vector, with nu drawn from a family that includes members
             outside BF's set (a delta-family)
  clusters   Vikhlinin-type gas density, Jaffe BCG and ICL, members from a
             cored power-law number density, triaxial with clumps
  lensing    shell-sum projection (exact per thin shell), BK's m=2 Green's
             function for the quadrupole (cdm-separation/forward.py), BK's
             source law, nuisance model and Simpson-rule cosmology
  X-ray      hydrostatic pressure by an outside-in Riemann sum, counts
             ~ n^2 T^0.3 r^3, a different non-thermal pressure law
  dynamics   Osipkov-Merritt anisotropic Jeans, projected on theta = acos(R/r)
  strong     bisection Einstein radius, bracketed image finder

Everything is emitted in the SAME dict schema generator 1 uses, so the very
same invariants.py runs on both; the only shared physical conventions are the
definitions of observables.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammainc

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.abspath(os.path.join(HERE, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)
_BK = os.path.abspath(os.path.join(HERE, "..", "cdm-separation"))
if _BK not in sys.path:
    sys.path.append(_BK)          # LAST: BK's worker.py/guard.py must not shadow this lane's

import forward as BK                                   # noqa: E402  (BK's module)
import halo as H                                       # noqa: E402
from paired import stream, _key                        # noqa: E402  (seeding helper only)

G = 4.300917270e-6
KPC_M = 3.0856775814913673e19
A0 = 1.2e-10 * KPC_M / 1e6            # (km/s)^2 / kpc
C_KMS = 299792.458
RHO_C = 136.0
KEV_PER_KMS2 = 1.0 / 1.55e5           # mu = 0.60 rather than BF's 0.62: a declared systematic

NOISE2 = dict(psf_arcsec=1.2, nspax=32, v_floor=4.0, v_ref=6.0, sz_err=0.08,
              ml=0.10, gas=0.08, dist=0.08, incl=4.0, pa=3.0, Rd=0.04, hz=0.12, mlgrad=0.04,
              shape=0.27, n_arcmin2=22.0, m_bias=0.015, c_bias=4e-4, psf_coh=1.2e-3,
              photoz=0.04, outlier=0.04, kT=0.05, counts_ref=2500.0, sz=0.08,
              mem_v=25.0, sl_pos=0.15, sl_dt=0.03, Mgas=0.035, Mstar=0.10,
              axis_ext=8.0, pa_bar=6.0, ell=0.03)


# ======================================================================
# response families for the universal-law class
# ======================================================================
def nu_simple(x):
    return 0.5 + np.sqrt(0.25 + 1.0 / np.maximum(x, 1e-14))


def nu_standard(x):
    x = np.maximum(x, 1e-14)
    return np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / x ** 2))


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-14))))


def nu_delta(x, d):
    x = np.maximum(x, 1e-14)
    return (1.0 - np.exp(-x ** (d / 2.0))) ** (-1.0 / d)


def draw_law(rng):
    fam = rng.choice(["simple", "standard", "rar", "delta"])
    return {"family": str(fam), "delta": float(rng.uniform(0.6, 2.0)),
            "a0": float(A0 * 10 ** rng.normal(0.0, 0.05))}


def nu_of(law, x):
    f = law["family"]
    if f == "simple":
        return nu_simple(x)
    if f == "standard":
        return nu_standard(x)
    if f == "rar":
        return nu_rar(x)
    return nu_delta(x, law["delta"])


# ======================================================================
# Einasto haloes
# ======================================================================
ALPHA_E = 0.17


def einasto(M200, c):
    r200 = (M200 / ((4.0 / 3.0) * np.pi * 200.0 * RHO_C)) ** (1.0 / 3.0)
    r2 = r200 / c
    frac = gammainc(3.0 / ALPHA_E, (2.0 / ALPHA_E) * c ** ALPHA_E)
    Mtot = M200 / frac
    return {"M": (lambda r: Mtot * gammainc(3.0 / ALPHA_E, (2.0 / ALPHA_E)
                                            * (np.asarray(r, float) / r2) ** ALPHA_E)),
            "M200": float(M200), "c": float(c), "r200": float(r200), "r2": float(r2)}


def shmr2(Mstar):
    x = np.log10(Mstar) - 10.5
    return 10 ** (12.0 + 1.6 * x + 0.35 * x * x)


# ======================================================================
# galaxies
# ======================================================================
def draw_galaxy2(rng, i):
    logMd = rng.uniform(8.8, 11.1)
    Md = 10 ** logMd
    Rd = 10 ** (0.28 * (logMd - 10.0) + 0.48 + rng.normal(0, 0.14))
    fgas = np.clip(10 ** (-0.5 * (logMd - 9.0) + 0.30 + rng.normal(0, 0.2)), 0.02, 5.0)
    fbul = float(np.clip(rng.beta(1.5, 6.0) * (logMd > 9.6), 0.0, 0.45))
    return dict(name=f"H{i:03d}", Md=Md, Rd=Rd, Mg=fgas * Md, Rg=1.9 * Rd, Mb=fbul * Md,
                ab=0.25 * Rd, hz=0.13 * Rd, incl_deg=float(rng.uniform(30, 76)),
                pa_deg=float(rng.uniform(0, 180)), dist_Mpc=float(10 ** rng.uniform(0.6, 2.0)),
                S_ext=float(10 ** rng.uniform(-2.5, 0.8)), axis_ext_deg=float(rng.uniform(0, 180)),
                t_merge=float(rng.exponential(4.5) + 0.2),
                void_frac=float(np.clip(rng.beta(2.4, 2.0), 0.02, 0.98)),
                k_z=float(rng.uniform(0.8, 1.25)))


def _mn(R, z, M, a, b):
    """Miyamoto-Nagai radial and vertical accelerations (positive inward/down)."""
    zeta = np.sqrt(z * z + b * b)
    den = (R * R + (a + zeta) ** 2) ** 1.5
    gR = G * M * R / den
    gz = G * M * z * (a + zeta) / (zeta * den)
    return gR, gz


def newton_field2(gal, R, z):
    R = np.asarray(R, float)
    z = np.asarray(z, float) + 0.0 * R
    gR1, gz1 = _mn(R, z, gal["Md"], 1.1 * gal["Rd"], gal["hz"])
    gR2, gz2 = _mn(R, z, gal["Mg"], 1.1 * gal["Rg"], 2.0 * gal["hz"])
    r = np.sqrt(R * R + z * z)
    gb = G * gal["Mb"] * r / (r * r + gal["ab"] ** 2) ** 1.5
    return gR1 + gR2 + gb * R / np.maximum(r, 1e-9), gz1 + gz2 + gb * z / np.maximum(r, 1e-9)


def galaxy_field2(gal, R, kind, law=None, dm=None, A_tensor=0.0):
    """Radial field in the plane, vertical field at z = h_z, in-plane m=2."""
    R = np.asarray(R, float)
    gRN, _ = newton_field2(gal, R, 0.0)
    gRh, gzh = newton_field2(gal, R, gal["hz"])
    quad = np.zeros_like(R)
    quad_pa = gal["pa_deg"]
    if kind == "class":
        x = gRN / law["a0"]
        gR = nu_of(law, x) * gRN
        xh = np.sqrt(gRh ** 2 + gzh ** 2) / law["a0"]
        gz = nu_of(law, xh) * gzh
        if A_tensor:
            f = (R / (3.0 * gal["Rd"])) ** 2 / (1.0 + (R / (3.0 * gal["Rd"])) ** 2)
            quad = 1.5 * A_tensor * f
            quad_pa = gal["axis_ext_deg"]
            cz = np.cos(np.deg2rad(gal["incl_deg"]))
            gz = gz * (1.0 + A_tensor * f * 0.5 * (3 * cz * cz - 1))
    elif kind == "cdm":
        Mh = dm["M"](R)
        gh = G * Mh / R ** 2
        gR = gRN + gh
        dR = 1e-3 * R
        dM = (dm["M"](R + dR) - dm["M"](R - dR)) / (2 * dR)
        gz_h = ((1 - dm["f_dd"]) * G * Mh * gal["hz"] / R ** 3 / dm["q_h"] ** 2
                + dm["f_dd"] * G * np.maximum(dM, 0.0) / R)
        gz = gzh + gz_h
        rt = 2.0 * gal["Rd"]
        f = (R / rt) ** 2 / (1.0 + (R / rt) ** 2)
        quad = dm["q_amp"] * f * gh / np.maximum(gR, 1e-12)
        quad_pa = gal["pa_deg"] + dm["psi_deg"]
    else:                                   # newton
        gR, gz = gRN, gzh
    return gR, gz, quad, quad_pa, gRN, gzh


def draw_galaxy_halo2(gal, rng, k):
    M200 = shmr2(gal["Md"] + gal["Mb"]) * 10 ** rng.normal(0, 0.17 * k["s_shmr"])
    c = 7.5 * (M200 / 1e12) ** -0.08 * 10 ** rng.normal(0, 0.12 * k["s_conc"])
    dm = einasto(M200, c)
    f_lss = float(k["f_lss"]) if k["f_lss"] is not None else float(rng.beta(*k["f_lss_prior"]))
    dm.update(q_amp=float(rng.uniform(0.0, k["q_max"])),
              psi_deg=H.mixed_axis(0.0, gal["axis_ext_deg"] - gal["pa_deg"], f_lss, rng, k["gal_mis_deg"]),
              q_h=float(rng.uniform(*k["q_h_range"])), f_dd=float(rng.uniform(*k["f_dd_range"])),
              f_lss=f_lss)
    return dm


def emit_galaxy2(gal, seed, kind, law=None, dm=None, A_tensor=0.0, sys_scale=1.0, noise_scale=1.0):
    N = NOISE2
    key = _key(gal["name"])
    S_ifu, S_sz, S_ph = (stream(seed, key, b) for b in ("ifu", "sz", "phot"))
    Rd = gal["Rd"]
    ext = 5.0 * Rd
    ax = np.linspace(-ext, ext, N["nspax"])
    Xs, Ys = np.meshgrid(ax, ax, indexing="ij")
    pa = np.deg2rad(gal["pa_deg"])
    xr = Xs * np.cos(pa) + Ys * np.sin(pa)
    yr = -Xs * np.sin(pa) + Ys * np.cos(pa)
    inc = np.deg2rad(gal["incl_deg"])
    yd = yr / max(np.cos(inc), 1e-3)
    Rp = np.sqrt(xr ** 2 + yd ** 2) + 1e-6
    th = np.arctan2(yd, xr)
    Rr = np.geomspace(0.05 * Rd, 5.5 * Rd, 60)
    gR, gz, quad, quad_pa, gRN, gzN = galaxy_field2(gal, Rr, kind, law, dm, A_tensor)
    vc = np.sqrt(np.maximum(gR * Rr, 1.0))
    vR = np.interp(Rp, Rr, vc)
    if np.any(quad > 0):
        qa = np.interp(Rp, Rr, quad)
        vR = vR * np.sqrt(np.maximum(1.0 + qa * np.cos(2 * (th - np.deg2rad(quad_pa - gal["pa_deg"]))), 0.05))
    vlos = vR * np.cos(th) * np.sin(inc)
    I = (gal["Md"] / (2 * np.pi * Rd ** 2) * np.exp(-Rp / Rd)
         + gal["Mg"] / (2 * np.pi * gal["Rg"] ** 2) * np.exp(-Rp / gal["Rg"]))
    kpc_per_arcsec = gal["dist_Mpc"] * 1e3 * (np.pi / 180.0 / 3600.0)
    sig_pix = float(np.clip((N["psf_arcsec"] / 2.355 * kpc_per_arcsec) / (ax[1] - ax[0]), 0.3, 4.0))
    Ic = gaussian_filter(I, sig_pix, mode="nearest")
    vobs = gaussian_filter(I * vlos, sig_pix, mode="nearest") / np.maximum(Ic, 1e-30)
    sn = np.sqrt(np.maximum(Ic / Ic.max(), 1e-6))
    everr = np.clip(N["v_floor"] + N["v_ref"] * noise_scale * np.sqrt(sys_scale) / np.maximum(sn, 0.03), 3.0, 150.0)
    vobs = vobs + everr * S_ifu.standard_normal(vobs.shape)
    mask = Ic > 2.0e-3 * Ic.max()

    Rv = np.array([1.5, 2.5]) * Rd
    gRv, gzv, _, _, gRNv, gzNv = galaxy_field2(gal, Rv, kind, law, dm, A_tensor)
    sz = np.sqrt(np.maximum(gal["k_z"] * gzv * gal["hz"], 1.0))
    sz_obs = sz * (1.0 + N["sz_err"] * sys_scale * noise_scale * S_sz.standard_normal(2))

    ml = S_ph.normal(0, N["ml"] * sys_scale)
    mlg = S_ph.normal(0, N["mlgrad"] * sys_scale)
    inc_obs = gal["incl_deg"] + S_ph.normal(0, N["incl"] * sys_scale)
    d_obs = gal["dist_Mpc"] * (1 + S_ph.normal(0, N["dist"] * sys_scale))
    fd = d_obs / gal["dist_Mpc"]
    rad_as = np.pi / 180.0 / 3600.0
    return {
        "name": gal["name"], "ax_arcsec": ax / (gal["dist_Mpc"] * 1e3 * rad_as),
        "v_map": vobs, "v_err": everr, "I_map": Ic, "mask": mask,
        "pa_obs": gal["pa_deg"] + S_ph.normal(0, N["pa"] * sys_scale),
        "incl_obs": float(np.clip(inc_obs, 12.0, 87.0)), "dist_obs": float(d_obs),
        "Md_obs": gal["Md"] * 10 ** ml * fd ** 2,
        "Mg_obs": gal["Mg"] * 10 ** S_ph.normal(0, N["gas"]) * fd ** 2,
        "Mb_obs": gal["Mb"] * 10 ** ml * fd ** 2,
        "Rd_obs": Rd * (1 + S_ph.normal(0, N["Rd"])) * fd,
        "Rg_obs": gal["Rg"] * fd, "ab_obs": gal["ab"] * fd,
        "hz_obs": gal["hz"] * (1 + S_ph.normal(0, N["hz"])) * fd,
        "ml_grad_obs": float(mlg), "sz_obs": sz_obs, "Rv": Rv,
        "S_ext_obs": gal["S_ext"] * 10 ** S_ph.normal(0, 0.15),
        "axis_ext_obs": (gal["axis_ext_deg"] + S_ph.normal(0, 12.0)) % 180.0,
        "t_merge_proxy": gal["t_merge"] * 10 ** S_ph.normal(0, 0.25),
        "void_frac_obs": float(np.clip(gal["void_frac"] + S_ph.normal(0, 0.06), 0, 1)),
        "_truth": {"hz": gal["hz"], "Rd": Rd, "incl": gal["incl_deg"],
                   "gz_true": sz ** 2 / gal["hz"], "gR_true": gRv, "gzN": gzNv, "gN": gRNv},
    }


# ======================================================================
# clusters
# ======================================================================
def draw_cluster2(rng, i):
    lMg = rng.uniform(12.9, 14.3)
    Mgas500 = 10 ** lMg
    M500_def = Mgas500 / 0.13
    R500 = (3.0 * M500_def / (4.0 * np.pi * 500.0 * RHO_C)) ** (1.0 / 3.0)
    c = dict(name=f"K{i:03d}", z=float(rng.uniform(0.12, 0.5)), R500=float(R500), Mgas500=Mgas500,
             alpha=float(rng.uniform(0.3, 1.0)), beta=float(rng.uniform(0.55, 0.8)),
             rc=float(rng.uniform(0.05, 0.15) * R500), rs=float(rng.uniform(0.6, 1.0) * R500),
             eps=float(rng.uniform(1.0, 3.0)))
    Mstar = 0.018 * M500_def * (M500_def / 1e14) ** -0.25
    c.update(M_bcg=0.25 * Mstar, a_bcg=float(rng.uniform(15, 35)), M_icl=0.30 * Mstar,
             a_icl=0.4 * R500)
    nmem = int(np.clip(rng.normal(40 + 110 * (M500_def / 6e14) ** 0.5, 15), 40, 300))
    m = 10 ** rng.uniform(9.2, 11.5, nmem)
    m = m / m.sum() * 0.45 * Mstar
    a_m = float(rng.uniform(0.2, 0.5) * R500)
    # number density ~ (1 + (r/a)^2)^-1.4 inside 2.5 R500, by inverse CDF on a grid
    rr = np.linspace(1e-3, 2.5 * R500, 3000)
    pdf = rr ** 2 * (1 + (rr / a_m) ** 2) ** -1.4
    cdf = np.cumsum(pdf) / np.sum(pdf)
    r3 = np.interp(rng.random(nmem), cdf, rr)
    u = rng.normal(size=(nmem, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    xyz = u * r3[:, None]
    axr = np.array([1.0, rng.uniform(0.55, 0.95), rng.uniform(0.5, 0.9)])
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    xyz = (xyz * axr) @ Q.T
    for _ in range(int(rng.integers(1, 3))):
        k = int(0.1 * nmem)
        idx = rng.choice(nmem, size=k, replace=False)
        ctr = rng.normal(size=3)
        ctr = ctr / np.linalg.norm(ctr) * rng.uniform(0.3, 1.1) * R500
        xyz[idx] = ctr + rng.normal(scale=0.12 * R500, size=(k, 3))
    P = xyz[:, :2]
    Cv = np.cov(P.T, aweights=m)
    ev, evec = np.linalg.eigh(Cv)
    c["ell_bar"] = float(1.0 - np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))
    c["pa_bar"] = float(np.rad2deg(np.arctan2(evec[1, 1], evec[0, 1])) % 180.0)
    nsur = int(rng.integers(3, 8))
    ang0 = rng.uniform(0, 2 * np.pi)
    sang = ang0 + rng.normal(scale=0.35, size=nsur)
    sr = rng.uniform(4.0, 10.0, nsur) * R500
    sm = 10 ** rng.uniform(13.0, 14.5, nsur)
    wx = np.sum(sm / sr ** 2 * np.cos(sang))
    wy = np.sum(sm / sr ** 2 * np.sin(sang))
    c["axis_ext"] = float(np.rad2deg(np.arctan2(wy, wx)) % 180.0)
    c["mem_m"], c["mem_xyz"] = m, xyz
    c["t_merge"] = float(rng.exponential(3.0) + 0.15)
    c["offset"] = float(150.0 * np.exp(-c["t_merge"] / 2.2) * 10 ** rng.normal(0, 0.2) + 6.0)
    c["wcen"] = float((0.01 + 0.05 * np.exp(-c["t_merge"] / 2.5)) * 10 ** rng.normal(0, 0.15))
    c["void"] = float(np.clip(rng.beta(2.4, 2.0), 0.02, 0.98))
    # radial grids and baryon profile
    rg = np.geomspace(3.0, 15.0 * R500, 400)
    n_gas = ((rg / c["rc"]) ** -c["alpha"] * (1 + (rg / c["rc"]) ** 2) ** (-1.5 * c["beta"] + 0.5 * c["alpha"])
             * (1 + (rg / c["rs"]) ** 3) ** (-c["eps"] / 3.0))
    Mg = _cummass(rg, n_gas)
    Mg = Mg / np.interp(R500, rg, Mg) * Mgas500
    rmem = np.linalg.norm(xyz, axis=1)
    Ms = np.array([m[rmem <= r].sum() for r in rg])
    Mb = Mg + Ms + _jaffe(rg, c["M_bcg"], c["a_bcg"]) + _jaffe(rg, c["M_icl"], c["a_icl"])
    c["rg"], c["Mgas"], c["Mbar"] = rg, Mg, Mb
    c["rho_gas"] = np.maximum(np.gradient(Mg, rg) / (4 * np.pi * rg ** 2), 1e-14)
    return c


def _jaffe(r, M, a):
    r = np.asarray(r, float)
    return M * r / (r + a)


def _cummass(rg, dens):
    integ = 4 * np.pi * rg ** 2 * dens
    return np.concatenate(([0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(rg))))


def draw_cluster_halo2(clu, rng, k):
    Mb500 = float(np.interp(clu["R500"], clu["rg"], clu["Mbar"]))
    fb = float(np.clip(rng.normal(0.15, 0.01), 0.1, 0.2))
    M200 = Mb500 / fb * 1.25 * 10 ** rng.normal(0, 0.06 * k["s_mclu"])
    c = 4.5 * (M200 / 1e15) ** -0.10 * 10 ** rng.normal(0, 0.12 * k["s_conc"])
    dm = einasto(M200, c)
    f_lss = float(k["f_lss"]) if k["f_lss"] is not None else float(rng.beta(*k["f_lss_prior"]))
    dm.update(ell=float(np.clip(0.5 * clu["ell_bar"] + rng.uniform(0.05, 0.35) * k["s_shape"]
                                + (0.1 if k["s_shape"] == 0 else 0.0), 0.0, 0.6)),
              pa=H.mixed_axis(clu["pa_bar"], clu["axis_ext"], f_lss, rng, 20.0), f_lss=f_lss,
              fb=fb)
    return dm


def shell_project(rg, M, Rq):
    """Sigma(R) and Sigmabar(<R) from an enclosed-mass profile by summing
    spherical shells with dM/dr uniform inside each shell [r_a, r_b]:

        Sigma(R)   = (1/2 pi) Int (dM/dr) dr / (r sqrt(r^2 - R^2))
                   = sum (dM/dr)/(2 pi R) [acos(R/r_b) - acos(R/max(r_a, R))]
        M_2D(<R)   = sum (dM/dr) Int (1 - sqrt(1 - R^2/r^2)) dr,
                     Int sqrt(1 - R^2/r^2) dr = sqrt(r^2 - R^2) - R acos(R/r)

    both exact per shell, so the inverse-square-root singularity at r -> R is
    integrated analytically rather than sampled.  (The first version summed
    thin shells at their mid-radius and was 15% off against the analytic NFW.)
    """
    ra, rb = rg[:-1], rg[1:]
    dMdr = np.diff(M) / (rb - ra)
    Rq = np.atleast_1d(np.asarray(Rq, float))[:, None]
    lo = np.maximum(ra[None, :], Rq)
    hi = rb[None, :]
    live = hi > lo
    x_lo = np.clip(Rq / np.maximum(lo, 1e-30), 0.0, 1.0)
    x_hi = np.clip(Rq / np.maximum(hi, 1e-30), 0.0, 1.0)
    Sig = np.sum(np.where(live, dMdr[None, :] / (2 * np.pi * Rq) * (np.arccos(x_hi) - np.arccos(x_lo)), 0.0), 1)
    # mass inside projected R: shells entirely inside R count fully; the part of
    # a shell beyond R contributes (r - sqrt(r^2-R^2) + R acos(R/r)) evaluated between lo and hi
    def F(r):
        return r - np.sqrt(np.maximum(r * r - Rq * Rq, 0.0)) + Rq * np.arccos(np.clip(Rq / np.maximum(r, 1e-30), 0.0, 1.0))
    inner = np.minimum(np.maximum(Rq - ra[None, :], 0.0), (rb - ra)[None, :])      # length of shell below R
    outer = np.where(live, F(hi) - F(lo), 0.0)
    M2 = np.sum(dMdr[None, :] * (inner + outer), 1) + M[0]
    return Sig, M2 / (np.pi * Rq[:, 0] ** 2)


def cluster_field2(clu, kind, law=None, dm=None):
    rg = clu["rg"]
    gN = G * clu["Mbar"] / rg ** 2
    if kind == "class":
        g = nu_of(law, gN / law["a0"]) * gN
        return g, g, gN
    if kind == "cdm":
        g = gN + G * dm["M"](rg) / rg ** 2
        return g, g, gN
    return gN, gN, gN


def hse_kT2(rg, g, rho):
    """Outside-in Riemann sum of rho g dr with a P ~ r^-3 tail."""
    w = rho * g
    P = np.zeros_like(w)
    P[-1] = w[-1] * rg[-1] / 2.0
    for i in range(len(rg) - 2, -1, -1):
        P[i] = P[i + 1] + 0.5 * (w[i] + w[i + 1]) * (rg[i + 1] - rg[i])
    return P / np.maximum(rho, 1e-300) * KEV_PER_KMS2


def om_jeans_los(rg, g, nu, Rq, ra):
    """Osipkov-Merritt Jeans, projected with theta = acos(R/r)."""
    f = nu * (1 + rg ** 2 / ra ** 2) * g
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(rg))))
    nus2 = (cum[-1] - cum + f[-1] * rg[-1] / 2.0) / (1 + rg ** 2 / ra ** 2)      # nu sigma_r^2
    th = np.linspace(0.0, np.pi / 2 - 1e-4, 300)
    out = []
    for R in np.atleast_1d(Rq):
        r = R / np.cos(th)
        rr = np.clip(r, rg[0], rg[-1])
        ns = np.interp(np.log(rr), np.log(rg), nus2)
        nn = np.interp(np.log(rr), np.log(rg), nu)
        bet = r ** 2 / (r ** 2 + ra ** 2)
        num = np.trapezoid((1 - bet * R ** 2 / r ** 2) * ns * r ** 2 / R, th)
        den = np.trapezoid(nn * r ** 2 / R, th)
        out.append(np.sqrt(max(num / max(den, 1e-300), 1.0)))
    return np.array(out)


def emit_cluster2(clu, seed, kind, law=None, dm=None, A_tensor=0.0, sys_scale=1.0, noise_scale=1.0):
    N = NOISE2
    key = _key(clu["name"])
    S_src, S_sh, S_pz, S_mem, S_x, S_sl, S_env = (
        stream(seed, key, b) for b in ("src", "shear", "photoz", "mem", "xray", "sl", "env"))
    R500, zl, rg = clu["R500"], clu["z"], clu["rg"]
    g_m, g_l, gN = cluster_field2(clu, kind, law, dm)
    M_l = g_l * rg ** 2 / G

    # ---------------- weak lensing (BK's source law and nuisance model) ------
    n_src = int(np.clip(N["n_arcmin2"] * 900.0 * (R500 / 1000.0) ** 2, 800, 9000))
    rmin, rmax = 0.10 * R500, 2.4 * R500
    rr = np.sqrt(rmin ** 2 + S_src.random(n_src) * (rmax ** 2 - rmin ** 2))
    pp = S_src.uniform(0, 2 * np.pi, n_src)
    zs = np.clip(zl + 0.20 + S_src.gamma(2.2, 0.35, n_src), zl + 0.08, 3.4)
    Scr = BK.sigma_crit(zl, zs)
    Rg = np.geomspace(0.02 * R500, 6.0 * R500, 700)
    Sig, Sbar = shell_project(rg, M_l, Rg)
    kap = np.interp(rr, Rg, Sig) / Scr
    gt = np.interp(rr, Rg, Sbar - Sig) / Scr
    g1 = -gt * np.cos(2 * pp)
    g2 = -gt * np.sin(2 * pp)
    zs_ref = zl + 0.9
    Scr_ref = float(BK.sigma_crit(zl, zs_ref))
    w = Scr_ref / Scr
    terms = []
    if kind == "cdm":
        Sh, _ = shell_project(rg, dm["M"](rg), Rg)
        terms.append((BK.f_halo(Rg, Sh / Scr_ref, dm["ell"]), dm["pa"]))
    if A_tensor:
        terms.append((BK.f_tensor(Rg, Sig / Scr_ref, A_tensor, 0.30 * R500), clu["axis_ext"]))
    for (fq, phi0) in terms:
        T, X, k2 = BK.quad_shear(Rg, fq)
        d = 2 * (pp - np.deg2rad(phi0))
        Tq, Xq, kq = np.interp(rr, Rg, T) * w, np.interp(rr, Rg, X) * w, np.interp(rr, Rg, k2) * w
        gt_q, gx_q = Tq * np.cos(d), Xq * np.sin(d)
        g1 += -(gt_q * np.cos(2 * pp) - gx_q * np.sin(2 * pp))
        g2 += -(gt_q * np.sin(2 * pp) + gx_q * np.cos(2 * pp))
        kap = kap + kq * np.cos(d)
    red1 = g1 / np.maximum(1.0 - kap, 0.25)
    red2 = g2 / np.maximum(1.0 - kap, 0.25)
    se = N["shape"] * noise_scale
    mb = S_sh.normal(0, N["m_bias"] * sys_scale)
    c1 = S_sh.normal(0, N["c_bias"] * sys_scale)
    c2 = S_sh.normal(0, N["c_bias"] * sys_scale)
    sx, sy = rr * np.cos(pp), rr * np.sin(pp)
    gx_, gy_ = S_sh.normal(size=2)
    amp = N["psf_coh"] * sys_scale
    e1 = (1 + mb) * red1 + c1 + amp * (gx_ * sx / rmax + (sx * sy) / rmax ** 2) + se * S_sh.standard_normal(n_src)
    e2 = (1 + mb) * red2 + c2 + amp * (gy_ * sy / rmax + (sx ** 2 - sy ** 2) / rmax ** 2) + se * S_sh.standard_normal(n_src)
    zph = zs * (1 + S_pz.normal(0.0, N["photoz"], n_src))
    nout = int(N["outlier"] * sys_scale * n_src)
    if nout > 0:
        oi = S_pz.choice(n_src, size=min(nout, n_src), replace=False)
        zph[oi] = S_pz.uniform(zl + 0.05, 3.0, len(oi))

    # ---------------- members ------------------------------------------------
    xyz, mm = clu["mem_xyz"], clu["mem_m"]
    Rp = np.clip(np.hypot(xyz[:, 0], xyz[:, 1]), 0.03 * R500, 2.4 * R500)
    a_m = 0.35 * R500
    nu_tr = (1 + (rg / a_m) ** 2) ** -1.4
    uq = np.unique(np.round(Rp, 0))
    sig = om_jeans_los(rg, g_m, nu_tr, uq, ra=float(S_mem.uniform(0.6, 1.5)) * R500)
    sig_at = np.interp(Rp, uq, sig)
    nm = len(mm)
    vmem = sig_at * S_mem.standard_normal(nm) + N["mem_v"] * noise_scale * S_mem.standard_normal(nm)
    pmem = np.clip(S_mem.beta(8, 1.3, nm), 0, 1)

    # ---------------- X-ray, SZ -------------------------------------------------
    rann = np.geomspace(0.07, 1.3, 11) * R500
    kT = hse_kT2(rg, g_m, clu["rho_gas"])
    kTa = np.interp(rann, rg, kT)
    fnt = np.clip(0.08 * sys_scale * (rann / R500) ** 0.6 + S_x.normal(0, 0.015 * sys_scale), 0.0, 0.4)
    kT_obs = kTa * (1 - fnt) * (1 + N["kT"] * noise_scale * S_x.standard_normal(len(rann)))
    ne = np.interp(rann, rg, clu["rho_gas"])
    rate = ne ** 2 * np.maximum(kTa, 0.3) ** 0.3 * rann ** 3
    cnts = S_x.poisson(np.maximum(rate / rate.max() * N["counts_ref"] / noise_scale ** 2, 1.0))
    y_obs = ne * kTa * R500 * (1 + N["sz"] * sys_scale * S_x.standard_normal(len(rann)))

    # ---------------- strong lensing --------------------------------------------
    zsl = float(zl + 0.5 + S_sl.gamma(2.0, 0.45))
    Scr_sl = float(BK.sigma_crit(zl, zsl))
    kb = Sbar / Scr_sl
    thE = 0.0
    if kb[0] > 1.0 and np.any(kb < 1.0):
        lo, hi = 0, int(np.argmax(kb < 1.0))
        for _ in range(40):
            mid = 0.5 * (Rg[lo] + Rg[hi])
            _, sb = shell_project(rg, M_l, np.array([mid]))
            if sb[0] / Scr_sl > 1.0:
                lo = int(np.searchsorted(Rg, mid))
            else:
                hi = int(np.searchsorted(Rg, mid))
            if hi - lo <= 1:
                break
        thE = float(0.5 * (Rg[lo] + Rg[hi]))
    fams, delays = [], []
    draws = [(S_sl.uniform(0, 2 * np.pi), S_sl.uniform(0.03, 0.30), S_sl.normal(0, 1), S_sl.normal(0, 1),
              S_sl.normal(0, 1)) for _ in range(3)]
    Dl = BK.comoving(zl) / (1 + zl) * 1e3
    if thE > 0.05 * R500:
        th = np.geomspace(0.2 * thE, 3.0 * thE, 600)
        al = th * np.interp(th, Rg, kb)
        for (phis, bfac, n1_, n2_, n3_) in draws:
            beta = bfac * thE
            fmin = th - al - beta
            s = np.sign(fmin)
            roots = th[:-1][s[:-1] != s[1:]]
            if len(roots) == 0:
                continue
            tp = float(roots[-1])
            tm = float(max(thE ** 2 / max(tp, 1e-6), 0.2 * thE))
            asec = 1.0 / (Dl * (np.pi / 180 / 3600))
            perr = N["sl_pos"] * sys_scale
            fams.append([tp * asec + perr * n1_, tm * asec + perr * n2_, float(np.rad2deg(phis)), zsl])
            psi = np.concatenate(([0.0], np.cumsum(0.5 * (al[1:] + al[:-1]) * np.diff(th))))
            fer = lambda t: 0.5 * (t - beta) ** 2 - np.interp(t, th, psi)
            Ds = BK.comoving(zsl) / (1 + zsl) * 1e3
            Dls = (BK.comoving(zsl) - BK.comoving(zl)) / (1 + zsl) * 1e3
            dt = (1 + zl) / C_KMS * (Dl * Ds / max(Dls, 1e-6)) * (fer(tm) - fer(tp)) / Dl ** 2
            dt = dt * 3.0857e16 / 86400.0
            delays.append(float(dt * (1 + N["sl_dt"] * n3_)))

    # ---------------- environment ----------------------------------------------
    ax_obs = (clu["axis_ext"] + S_env.normal(0, N["axis_ext"])) % 180.0
    pa_obs = (clu["pa_bar"] + S_env.normal(0, N["pa_bar"])) % 180.0
    mis = S_env.normal(0, 20.0 * sys_scale)
    mem_xy = xyz[:, :2] + S_env.normal(0, 2.5, (nm, 2)) + mis
    mem_m_obs = mm * 10 ** S_env.normal(0, 0.12 * sys_scale, nm)
    r3 = np.linalg.norm(xyz, axis=1)
    Mgas_obs = np.interp(rann, rg, clu["Mgas"]) * 10 ** S_env.normal(0, N["Mgas"] * sys_scale)
    Mstar_obs = (np.array([mem_m_obs[r3 <= r].sum() for r in rann])
                 + _jaffe(rann, clu["M_bcg"] + clu["M_icl"], 0.3 * R500)) * 10 ** S_env.normal(0, N["Mstar"] * sys_scale)
    return {
        "name": clu["name"], "z": zl, "R500": R500, "src_x": sx, "src_y": sy, "e1": e1, "e2": e2,
        "w": np.full(n_src, 1.0 / (se ** 2 + 0.09)), "z_src_phot": zph,
        "mem_x": mem_xy[:, 0], "mem_y": mem_xy[:, 1], "mem_v": vmem, "mem_p": pmem, "mem_m_obs": mem_m_obs,
        "r_ann": rann, "xray_counts": cnts, "kT_obs": kT_obs, "kT_err": np.maximum(kT_obs * N["kT"], 0.05),
        "y_sz": y_obs, "Mgas_obs": Mgas_obs, "Mstar_obs": Mstar_obs,
        "sl_fams": np.array(fams) if fams else np.zeros((0, 4)), "sl_delays": np.array(delays),
        "thetaE_kpc": thE, "z_sl": zsl, "axis_ext_obs": ax_obs, "pa_bar_obs": pa_obs,
        "ell_bar_obs": float(max(clu["ell_bar"] + S_env.normal(0, N["ell"]), 0.01)),
        "t_merge_proxy": clu["wcen"], "gas_gal_offset_obs": clu["offset"] * 10 ** S_env.normal(0, 0.12),
        "void_frac_obs": float(np.clip(clu["void"] + S_env.normal(0, 0.06), 0, 1)),
        "Sig_cr_ref": Scr_ref,
        "_truth": {"g_m": g_m, "g_l": g_l, "rg": rg, "Mbar500": float(np.interp(R500, rg, clu["Mbar"])),
                   "halo": None if dm is None else {k: dm[k] for k in dm if k != "M"}},
    }


def emit_sn2(seed, n=200):
    S = stream(seed, 0, "sn")
    z = np.clip(S.gamma(2.2, 0.16, n), 0.01, 1.3)
    void = np.clip(S.beta(2.4, 2.0, n), 0.01, 0.99)
    dl = BK.comoving(z) * (1 + z)
    mag = 5 * np.log10(np.maximum(dl, 1e-3)) + 25.0 + 0.12 * S.standard_normal(n)
    dur = 20.0 * (1 + z) * (1 + 0.04 * S.standard_normal(n))
    return {"z_obs": z, "mag": mag, "duration": dur, "void_frac": void}


# ======================================================================
# a paired set on generator 2
# ======================================================================
def draw_scene2(seed, n_gal, n_clu):
    rng = stream(seed, 0, "scene")
    gals = [draw_galaxy2(rng, i) for i in range(n_gal)]
    clus = [draw_cluster2(rng, i) for i in range(n_clu)]
    return gals, clus


def run_set2(job):
    """job = (seed, arms, opt); an arm here is dict(tag, kind in {'class',
    'cdm', 'newton'}, halo knobs, A_tensor, sys, noise)."""
    import invariants as IV
    seed, arms, opt = job
    n_gal, n_clu = opt.get("n_gal", 30), opt.get("n_clu", 12)
    gals, clus = draw_scene2(seed, n_gal, n_clu)
    law = draw_law(stream(seed, 0, "universe"))
    out = {}
    for arm in arms:
        kind = arm["kind"]
        A_t = arm.get("A_tensor", 0.0)
        ss, ns = arm.get("sys", 1.0), arm.get("noise", 1.0)
        dm_g, dm_c = [None] * n_gal, [None] * n_clu
        if kind == "cdm":
            k = H.knobs(**(arm.get("halo") or {}))
            dm_g = [draw_galaxy_halo2(g, stream(seed, _key(g["name"]), "halo"), k) for g in gals]
            dm_c = [draw_cluster_halo2(c, stream(seed, _key(c["name"]), "halo"), k) for c in clus]
        gal_d, tg = [], {"M200": [], "f_lss": [], "q_amp": [], "q_h": [], "f_dd": [],
                         "boost_z1": [], "boost_R1": [], "boost_z2": [], "boost_R2": []}
        for g, d in zip(gals, dm_g):
            gd = emit_galaxy2(g, seed, kind, law, d, A_t, ss, ns)
            gal_d.append(gd)
            T = gd["_truth"]
            for key in ("M200", "f_lss", "q_amp", "q_h", "f_dd"):
                tg[key].append(d[key] if d is not None else np.nan)
            tg["boost_z1"].append(np.log10(T["gz_true"][0] / T["gzN"][0]))
            tg["boost_R1"].append(np.log10(T["gR_true"][0] / T["gN"][0]))
            tg["boost_z2"].append(np.log10(T["gz_true"][1] / T["gzN"][1]))
            tg["boost_R2"].append(np.log10(T["gR_true"][1] / T["gN"][1]))
        clu_d, tc = [], {"M200": [], "f_lss": [], "ell": [], "pa": [], "boost_05": [], "boost_15": []}
        for c, d in zip(clus, dm_c):
            cd = emit_cluster2(c, seed, kind, law, d, A_t, ss, ns)
            clu_d.append(cd)
            T = cd["_truth"]
            for key in ("M200", "f_lss", "ell", "pa"):
                tc[key].append(d[key] if d is not None else np.nan)
            gN = G * c["Mbar"] / c["rg"] ** 2
            tc["boost_05"].append(np.log10(np.interp(0.5 * c["R500"], c["rg"], T["g_m"] / gN)))
            tc["boost_15"].append(np.log10(np.interp(1.5 * c["R500"], c["rg"], T["g_m"] / gN)))
        A = IV.analyse_corpus(gal_d, clu_d, emit_sn2(seed), split_seed=seed % 7919)
        if A is None:
            continue
        gal = {k: np.array([F.get(k, np.nan) for F in A["gal"]], float) for k in IV.galaxy_feature_names()}
        clu = {k: np.array([F.get(k, np.nan) for F in A["clu"]], float) for k in IV.cluster_feature_names()}
        for k, v in tg.items():
            gal["T_" + k] = np.asarray(v, float)
        for k, v in tc.items():
            clu["T_" + k] = np.asarray(v, float)
        A["corpus"]["scene_gal"] = list(range(n_gal))
        A["corpus"]["scene_clu"] = list(range(n_clu))
        A["corpus"]["a0"] = float(np.log10(law["a0"]))
        out[arm["tag"]] = {"gal": gal, "clu": clu, "corpus": A["corpus"]}
    return seed, out


def arms2():
    base = dict(f_lss=0.5, q_h_range=(0.85, 0.85), f_dd_range=(0.0, 0.0), q_max=0.05)
    arms = [dict(tag="C", kind="class"), dict(tag="D", kind="cdm", halo={}),
            dict(tag="N", kind="newton"), dict(tag="T", kind="class", A_tensor=0.3),
            dict(tag="D_nominal", kind="cdm", halo=dict(base))]
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        arms.append(dict(tag=f"D_flss_{f:g}", kind="cdm", halo=dict(base, f_lss=f)))
    for f in (0.1, 0.3, 0.5, 1.0):
        arms.append(dict(tag=f"D_fdd_{f:g}", kind="cdm", halo=dict(base, f_dd_range=(f, f))))
    arms.append(dict(tag="D_zeroscatter", kind="cdm",
                     halo=dict(base, s_shmr=0.0, s_conc=0.0, s_mclu=0.0, s_shape=0.0)))
    arms.append(dict(tag="D_shmr2", kind="cdm", halo=dict(base, s_shmr=2.0)))
    return arms


_POOL = None


def _init():
    import guard
    try:
        guard.start()
    except Exception:                                          # noqa: BLE001
        pass


def run_sets2(seeds, arms, opt, serial=False, chunk=2):
    global _POOL
    jobs = [(int(s), arms, opt) for s in seeds]
    if serial or len(jobs) < 4:
        return [run_set2(j) for j in jobs]
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context("spawn").Pool(max(1, min(20, (os.cpu_count() or 4) - 4)), initializer=_init)
    return _POOL.map(run_set2, jobs, chunksize=chunk)


def close_pool2():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None
