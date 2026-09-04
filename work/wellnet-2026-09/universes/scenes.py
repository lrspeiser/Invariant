"""scenes.py -- the resolved baryonic scene library, with the expensive
geometry precomputed once.

The scenes are drawn ONCE and shared by every universe.  That is deliberate:
if each universe drew its own scenes, a pairwise separation could come from the
scene prior rather than from the physics.  Sharing the scene library makes the
comparison a comparison of LAWS.

Per corpus draw we resample WHICH library objects are observed and every
observational nuisance, so corpora are independent realisations, but the
population of visible matter is common to all ten universes.

Precomputed per library cluster (the expensive part):
  rg          radial grid                                        (nr,)
  xg, yg      lens-plane grid [kpc]                              (nx,), (nx,)
  zg          line-of-sight grid [kpc], tanh-spaced              (nz,)
  r3, u3      3-D radius and cos(angle to the external axis)     (nx,nx,nz)
  Ex3         the well-network potential field (S_lam^1/2 * T)   (nx,nx,nz)
  Ex_bar      its angular mean profile on rg                     (nr,)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baryons import G, ClusterScene, DiskGalaxy, hernquist_M

LAM_NET = 150.0          # kpc -- the universal coherence scale of U06
# Q_NET is the exponent of the reciprocal pair weighting
#   Q_ab = 1 + B (S_a S_b / S0^2)^(Q_NET/2)
# and S0 is a declared universal well-strength scale built from a 1e10 Msun
# well at the coherence length.  Q_NET = 0.5 keeps the galaxy-scale and
# cluster-scale amplitudes within a factor of ~5 of each other, so a single
# global B is a meaningful universal constant rather than a scale switch.
Q_NET = 0.5
S0_NET = G * 1e12 / (LAM_NET ** 2)


# ------------------------------------------------------------------ galaxies
def draw_galaxy(rng, i):
    logMd = rng.uniform(8.6, 11.2)
    Md = 10 ** logMd
    Rd = 10 ** (0.32 * (logMd - 10.0) + 0.42 + rng.normal(0, 0.12))
    fgas = np.clip(10 ** (-0.45 * (logMd - 9.0) + 0.35 + rng.normal(0, 0.20)), 0.02, 6.0)
    Mg = fgas * Md
    fbul = np.clip(rng.beta(1.4, 5.0) * (logMd > 9.8), 0.0, 0.5)
    Mb = fbul * Md
    ab = 0.2 * Rd
    hz = 0.16 * Rd
    cosi = rng.uniform(np.cos(np.deg2rad(78.0)), np.cos(np.deg2rad(28.0)))
    incl = float(np.rad2deg(np.arccos(cosi)))
    pa = float(rng.uniform(0, 180))
    dist = float(10 ** rng.uniform(0.5, 2.1))
    # environment: a directionless external well strength and an axis
    S_ext = float(10 ** rng.uniform(-2.5, 0.8))
    axis = float(rng.uniform(0, 180))
    tidal = float(10 ** rng.uniform(-5.0, -2.5))
    t_merge = float(rng.exponential(4.5) + 0.2)
    void = float(np.clip(rng.beta(2.4, 2.0), 0.02, 0.98))
    return DiskGalaxy(name=f"G{i:03d}", Md=Md, Rd=Rd, Mg=Mg, Rg=2.2 * Rd,
                      Mb=Mb, ab=ab, hz=hz, incl_deg=incl, pa_deg=pa,
                      dist_Mpc=dist, S_ext=S_ext, axis_ext_deg=axis,
                      tidal=tidal, t_merge=t_merge, void_frac=void)


# ------------------------------------------------------------------ clusters
def _sample_nfw_radii(rng, n, c, rvir):
    """Inverse-CDF sample of an NFW number density inside rvir."""
    x = np.linspace(1e-4, 1.0, 4000)
    mu = np.log(1 + c * x) - c * x / (1 + c * x)
    cdf = mu / mu[-1]
    u = rng.random(n)
    return np.interp(u, cdf, x) * rvir


def draw_cluster(rng, i):
    logM500 = rng.uniform(13.7, 15.0)
    M500 = 10 ** logM500                      # total (incl. dark, if any)
    R500 = 1000.0 * (M500 / 6e14) ** (1 / 3)
    fgas = np.clip(0.115 + 0.02 * (logM500 - 14.4) + rng.normal(0, 0.012), 0.04, 0.19)
    Mgas = fgas * M500
    Mstar_tot = 0.021 * M500 * (M500 / 1e14) ** -0.28
    M_bcg = 0.22 * Mstar_tot
    M_icl = 0.32 * Mstar_tot
    Mmem_tot = Mstar_tot - M_bcg - M_icl

    nmem = int(np.clip(rng.normal(30 + 130 * (M500 / 6e14) ** 0.55, 15), 35, 320))
    # Schechter-like member stellar masses
    m = 10 ** (rng.uniform(9.3, 11.6, nmem))
    m = m / m.sum() * Mmem_tot

    c_num = float(rng.uniform(2.8, 5.0))
    r = _sample_nfw_radii(rng, nmem, c_num, 2.0 * R500)
    u = rng.normal(size=(nmem, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    xyz = u * r[:, None]
    # triaxial shape, random orientation
    ax = np.array([1.0, float(rng.uniform(0.62, 0.95)), float(rng.uniform(0.52, 0.90))])
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    xyz = (xyz * ax) @ Q.T
    # substructure: 1-3 clumps
    for _ in range(int(rng.integers(1, 4))):
        k = int(0.08 * nmem)
        idx = rng.choice(nmem, size=k, replace=False)
        ctr = rng.normal(size=3); ctr /= np.linalg.norm(ctr)
        ctr = ctr * rng.uniform(0.35, 1.2) * R500
        xyz[idx] = ctr + rng.normal(scale=0.13 * R500, size=(k, 3))

    # projected baryonic shape (x-y is the sky plane, z is the l.o.s.)
    P = xyz[:, :2]
    Cv = np.cov(P.T, aweights=m)
    ev, evec = np.linalg.eigh(Cv)
    ell_bar = float(1.0 - np.sqrt(max(ev[0], 1e-9) / max(ev[1], 1e-9)))
    pa_bar = float(np.rad2deg(np.arctan2(evec[1, 1], evec[0, 1])) % 180.0)

    # surrounding structure -> the INDEPENDENTLY OBSERVABLE external axis
    nsur = int(rng.integers(3, 9))
    dirn = rng.normal(size=2); dirn /= np.linalg.norm(dirn)
    ang = np.arctan2(dirn[1], dirn[0])
    sang = ang + rng.normal(scale=0.5, size=nsur)
    sr = rng.uniform(3.0, 11.0, nsur) * R500
    sxy = np.stack([sr * np.cos(sang), sr * np.sin(sang)], 1)
    sur_xyz = np.concatenate([sxy, rng.normal(scale=1.5 * R500, size=(nsur, 1))], 1)
    sur_m = 10 ** rng.uniform(13.2, 14.6, nsur)
    wsum = (sur_m[:, None] / np.linalg.norm(sur_xyz, axis=1, keepdims=True) ** 2 * sur_xyz[:, :2]).sum(0)
    axis_ext = float(np.rad2deg(np.arctan2(wsum[1], wsum[0])) % 180.0)

    t_merge = float(rng.exponential(3.2) + 0.15)
    off = float(180.0 * np.exp(-t_merge / 2.0) * 10 ** rng.normal(0, 0.18) + 8.0)
    wsh = float((0.012 + 0.055 * np.exp(-t_merge / 2.5)) * 10 ** rng.normal(0, 0.14))
    void = float(np.clip(rng.beta(2.4, 2.0), 0.02, 0.98))

    return ClusterScene(
        name=f"C{i:03d}", z=float(rng.uniform(0.15, 0.45)),
        M500_bar=0.0, R500=R500,
        rc_gas=float(R500 * rng.uniform(0.07, 0.16)),
        beta_gas=float(rng.uniform(0.58, 0.75)), Mgas=Mgas,
        M_bcg=M_bcg, a_bcg=float(rng.uniform(12, 30)),
        mem_m=m, mem_xyz=xyz, M_icl=M_icl, a_icl=0.45 * R500,
        axis_ext_deg=axis_ext, ell_bar=ell_bar, pa_bar_deg=pa_bar,
        t_merge=t_merge, gas_gal_offset=off, centroid_shift=wsh,
        surround_xyz=sur_xyz, surround_m=sur_m, void_frac=void)


# ------------------------------------------------ precomputed cluster geometry
@dataclass
class ClusterGeom:
    clu: ClusterScene
    rg: np.ndarray
    xg: np.ndarray
    zg: np.ndarray
    dz: np.ndarray
    r3: np.ndarray
    u3: np.ndarray
    Ex3: np.ndarray
    Ex_bar: np.ndarray
    bgrid: np.ndarray
    los_w: np.ndarray = field(default=None, repr=False)


def build_geom(clu: ClusterScene, nx=64, nz=31, half=2.5, zmax=20.0):
    R5 = clu.R500
    rg = np.geomspace(0.02 * R5, 6.0 * R5, 220)
    xg = np.linspace(-half * R5, half * R5, nx)
    t = np.linspace(-1.0, 1.0, nz)
    zg = zmax * R5 * np.tanh(2.6 * t) / np.tanh(2.6)
    dz = np.gradient(zg)

    X, Y, Z = np.meshgrid(xg, xg, zg, indexing="ij")
    r3 = np.sqrt(X * X + Y * Y + Z * Z) + 1e-6
    ang = np.deg2rad(clu.axis_ext_deg)
    nvec = np.array([np.cos(ang), np.sin(ang), 0.0])   # axis in the sky plane
    u3 = (X * nvec[0] + Y * nvec[1] + Z * nvec[2]) / r3

    # ---- well-network fields ------------------------------------------
    # every baryonic component is a well: members, BCG+ICL, and the gas
    # (represented by shells so the continuous ICM is not privileged out).
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    lam2 = LAM_NET ** 2
    S = np.zeros(P.shape[0])
    T = np.zeros(P.shape[0])
    ngs, ndir = 12, 12
    rgs = np.geomspace(0.05 * R5, 3.0 * R5, ngs + 1)
    mgs = np.diff(np.interp(rgs, clu._rg, clu._Mgas_grid)) / ndir
    rgc = np.sqrt(rgs[1:] * rgs[:-1])
    ug = np.random.default_rng(11).normal(size=(ndir, 3))
    ug /= np.linalg.norm(ug, axis=1, keepdims=True)
    gas_x = (rgc[:, None, None] * ug[None, :, :]).reshape(-1, 3)
    gas_m = np.repeat(mgs, ndir)
    mm = np.concatenate([clu.mem_m, gas_m])
    mx = np.concatenate([clu.mem_xyz, gas_x])
    # S at each member's own location (for the reciprocal pair weighting)
    d2mm = ((mx[:, None, :] - mx[None, :, :]) ** 2).sum(-1)
    S_mem = (G * mm[None, :] / (d2mm + lam2)).sum(1) \
        + G * (clu.M_bcg + clu.M_icl) / ((mx ** 2).sum(1) + lam2)
    wgt = mm * np.sqrt(np.maximum(S_mem / S0_NET, 1e-12))
    chunk = 40000
    for a in range(0, P.shape[0], chunk):
        b = min(a + chunk, P.shape[0])
        d2 = ((P[a:b, None, :] - mx[None, :, :]) ** 2).sum(-1)
        S[a:b] = (G * mm[None, :] / (d2 + lam2)).sum(1)
        T[a:b] = -(G * wgt[None, :] / np.sqrt(d2 + lam2)).sum(1)
        rr2 = (P[a:b] ** 2).sum(1)
        S[a:b] += G * (clu.M_bcg + clu.M_icl) / (rr2 + lam2)
        T[a:b] += -G * (clu.M_bcg + clu.M_icl) * max(
            S_mem.mean() / S0_NET, 1e-12) ** (0.5 * Q_NET) / np.sqrt(rr2 + lam2)
    Ex3 = (np.maximum(S / S0_NET, 1e-12) ** (0.5 * Q_NET) * T).reshape(r3.shape)

    # angular mean profile of Ex on rg (for the radial dynamics of U06)
    rb = np.concatenate([[0.0], np.sqrt(rg[1:] * rg[:-1]), [rg[-1] * 1.4]])
    idx = np.clip(np.digitize(r3.ravel(), rb) - 1, 0, len(rg) - 1)
    cnt = np.bincount(idx, minlength=len(rg))
    ssum = np.bincount(idx, weights=Ex3.ravel(), minlength=len(rg))
    Ex_bar = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0)
    good = cnt > 0
    if good.sum() > 4:
        Ex_bar = np.interp(np.log(rg), np.log(rg[good]), Ex_bar[good])

    bgrid = np.geomspace(0.04 * R5, 3.0 * R5, 42)
    return ClusterGeom(clu=clu, rg=rg, xg=xg, zg=zg, dz=dz, r3=r3, u3=u3,
                       Ex3=Ex3, Ex_bar=Ex_bar, bgrid=bgrid)


@dataclass
class SceneLibrary:
    galaxies: list
    geoms: list
    seed: int

    @property
    def clusters(self):
        return [g.clu for g in self.geoms]


def build_library(seed=20260904, n_gal=70, n_clu=14, nx=64, nz=31):
    rng = np.random.default_rng(seed)
    gals = [draw_galaxy(rng, i) for i in range(n_gal)]
    geoms = []
    for i in range(n_clu):
        clu = draw_cluster(rng, i)
        geoms.append(build_geom(clu, nx=nx, nz=nz))
    return SceneLibrary(galaxies=gals, geoms=geoms, seed=seed)


# ------------------------------------------------------- coarse-graining gate
def coarse_grain_check(clu: ClusterScene, lam=LAM_NET, nsub=10, seed=0):
    """Represent every member as 1, then as nsub subcomponents; compare S_lam.

    A network law that depends on how the cataloguer deblended the image is
    inadmissible (charter s.10).  The softened inverse-square well strength is
    linear in mass with a smooth kernel, so subdividing a source at scales
    << lam must leave S unchanged to O((d_sub/lam)^2).
    """
    rng = np.random.default_rng(seed)
    pts = clu.mem_xyz[:40] * 1.37 + 50.0
    S1 = clu.S_lambda(pts, lam)
    m2, x2 = [], []
    for mi, xi in zip(clu.mem_m, clu.mem_xyz):
        d = rng.normal(scale=6.0, size=(nsub, 3))
        d -= d.mean(0)
        x2.append(xi + d)
        m2.append(np.full(nsub, mi / nsub))
    import copy
    c2 = copy.copy(clu)
    c2.mem_m = np.concatenate(m2)
    c2.mem_xyz = np.concatenate(x2)
    S2 = c2.S_lambda(pts, lam)
    rel = np.abs(S2 - S1) / np.abs(S1)
    return {"lam_kpc": lam, "n_sub": nsub,
            "max_rel_change": float(rel.max()),
            "median_rel_change": float(np.median(rel))}


def reciprocity_check(clu: ClusterScene, lam=LAM_NET, B=0.15):
    """F(a<-b) + F(b<-a) = 0 for the network pair term, to machine precision.

    The extra pair energy is  U_ab = -(G/2) M_a M_b Q_ab / sqrt(d_ab^2+lam^2)
    with Q_ab = Q_ba, so the force is the gradient of a symmetric scalar of
    d_ab alone and is equal and opposite by construction.  We verify it
    numerically rather than asserting it.
    """
    x = clu.mem_xyz[:24]
    m = clu.mem_m[:24]
    S = clu.S_lambda(x, lam)
    Q = 1.0 + B * np.sqrt(np.outer(S, S)) / S0_NET
    d = x[:, None, :] - x[None, :, :]
    dd = np.sqrt((d ** 2).sum(-1) + lam * lam)
    coef = G * np.outer(m, m) * Q / dd ** 3
    F = (coef[:, :, None] * d)                 # F_ab on a from b
    np.einsum("iij->ij", F)[...] = 0.0
    asym = np.abs(F + np.transpose(F, (1, 0, 2)))
    scale = np.abs(F).max()
    return {"max_abs_asymmetry_over_maxF": float(asym.max() / max(scale, 1e-300))}
