"""baryons.py -- the visible source scenes.

Units: kpc, Msun, km/s, Gyr.  G = 4.300917270e-6 kpc (km/s)^2 / Msun.

Nothing here knows about gravity laws.  These objects describe the RESOLVED
baryonic scene that every alternate universe is then asked to act on:

  DiskGalaxy   stellar exponential disk + gas disk + Hernquist bulge, with a
               scale height, an inclination, a position angle and a declared
               environment (external well strength, external axis, tidal field).

  ClusterScene beta-model hot gas + Hernquist BCG + N individual member
               galaxies at sampled 3-D positions + an intracluster-light
               component + a surrounding-structure catalogue that defines an
               INDEPENDENTLY OBSERVABLE external axis.

The cluster keeps every member as an individual source down to the declared
completeness threshold, per the charter's root-data rule.  The smooth-profile
version is available (``smoothed``) only as a control.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import i0e, i1e, k0e, k1e

G = 4.300917270e-6            # kpc (km/s)^2 / Msun
KPC_PER_M = 1.0 / 3.0856775814913673e19
A0_SI = 1.2e-10               # m s^-2
A0 = A0_SI * 3.0856775814913673e13   # (km/s)^2 / kpc  = 3702.8
C_KMS = 299792.458


# ---------------------------------------------------------------- profiles
def disk_vc2(R, Md, Rd):
    """Newtonian v_c^2 for a razor-thin exponential disk (Freeman 1970)."""
    R = np.atleast_1d(np.asarray(R, float))
    y = np.clip(R / (2.0 * Rd), 1e-8, 300.0)
    # ive/kve are exponentially scaled; the scalings cancel in the products
    t = i0e(y) * k0e(y) - i1e(y) * k1e(y)
    sig0 = Md / (2.0 * np.pi * Rd * Rd)
    return 4.0 * np.pi * G * sig0 * Rd * y * y * t


def hernquist_M(r, M, a):
    r = np.asarray(r, float)
    return M * r * r / (r + a) ** 2


def hernquist_phi(r, M, a):
    return -G * M / (np.asarray(r, float) + a)


def beta_model_rho(r, rho0, rc, beta):
    return rho0 * (1.0 + (np.asarray(r, float) / rc) ** 2) ** (-1.5 * beta)


def _cum_mass_from_rho(rgrid, rho):
    """Enclosed mass by trapezoid on a supplied grid (rgrid ascending, >0)."""
    integ = 4.0 * np.pi * rgrid ** 2 * rho
    M = np.concatenate(([0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) *
                                         np.diff(rgrid))))
    return M


def nfw_M(r, M200, c, r200):
    r = np.asarray(r, float)
    rs = r200 / c
    mu = lambda x: np.log(1.0 + x) - x / (1.0 + x)
    return M200 * mu(r / rs) / mu(c)


# ---------------------------------------------------------------- galaxies
@dataclass
class DiskGalaxy:
    name: str
    Md: float                 # stellar disk mass [Msun]
    Rd: float                 # stellar scale length [kpc]
    Mg: float                 # gas disk mass [Msun]
    Rg: float                 # gas scale length [kpc]
    Mb: float                 # bulge mass [Msun]
    ab: float                 # bulge scale [kpc]
    hz: float                 # stellar scale height [kpc]
    incl_deg: float
    pa_deg: float
    dist_Mpc: float
    # environment (charter section 9): declared, observable with error
    S_ext: float              # directionless external well strength, (km/s)^2/kpc
    axis_ext_deg: float       # external axis position angle on the sky [deg]
    tidal: float              # external tidal magnitude, (km/s)^2/kpc^2
    t_merge: float            # time since last major merger [Gyr]
    void_frac: float          # fraction of the l.o.s. path in voids
    R_bnd: float = 0.0        # operational potential-boundary radius [kpc]

    def __post_init__(self):
        if self.R_bnd == 0.0:
            self.R_bnd = 10.0 * self.Rd

    @property
    def Mbar(self):
        return self.Md + self.Mg + self.Mb

    def vc2_newton(self, R):
        return (disk_vc2(R, self.Md, self.Rd)
                + disk_vc2(R, self.Mg, self.Rg)
                + G * hernquist_M(R, self.Mb, self.ab) / np.asarray(R, float))

    def gN(self, R):
        R = np.asarray(R, float)
        return self.vc2_newton(R) / R

    def Menc_eff(self, R):
        """g_N R^2 / G -- the spherical-equivalent enclosed mass of the disk."""
        R = np.asarray(R, float)
        return self.gN(R) * R * R / G

    def Sigma(self, R):
        R = np.asarray(R, float)
        return (self.Md / (2 * np.pi * self.Rd ** 2) * np.exp(-R / self.Rd)
                + self.Mg / (2 * np.pi * self.Rg ** 2) * np.exp(-R / self.Rg))

    def phi_depth(self, R):
        """Operational potential DIFFERENCE |Phi(R) - Phi(R_bnd)| from baryons.

        Declared primary boundary rule: R_bnd = 10 * R_d for a disk galaxy.
        Gauge-safe by construction (charter s.6: never use absolute Phi).
        """
        R = np.atleast_1d(np.asarray(R, float))
        rg = np.geomspace(min(R.min(), 0.05 * self.Rd), self.R_bnd, 256)
        g = self.gN(rg)
        # Phi(r) - Phi(R_bnd) = -int_r^{R_bnd} g dr'   (g > 0 outward)
        seg = 0.5 * (g[1:] + g[:-1]) * np.diff(rg)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        dphi = cum[-1] - cum                      # = int_r^{Rbnd} g dr' >= 0
        return np.interp(R, rg, dphi)


# ---------------------------------------------------------------- clusters
@dataclass
class ClusterScene:
    name: str
    z: float
    M500_bar: float           # total baryonic mass inside R500 [Msun]
    R500: float               # [kpc]
    rc_gas: float
    beta_gas: float
    Mgas: float
    M_bcg: float
    a_bcg: float
    mem_m: np.ndarray         # member stellar masses [Msun]
    mem_xyz: np.ndarray       # member 3-D positions [kpc], cluster frame
    M_icl: float
    a_icl: float
    axis_ext_deg: float       # external axis from surrounding structure [deg]
    ell_bar: float            # baryonic projected ellipticity
    pa_bar_deg: float         # baryonic major axis [deg]
    t_merge: float            # time since major merger [Gyr]
    gas_gal_offset: float     # observable gas-galaxy centroid offset [kpc]
    centroid_shift: float     # observable X-ray centroid shift w [dimensionless]
    surround_xyz: np.ndarray  # surrounding structure (observable) [kpc]
    surround_m: np.ndarray
    void_frac: float
    lam_grid: np.ndarray = field(default=None, repr=False)
    _rg: np.ndarray = field(default=None, repr=False)
    _Mb: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        rg = np.geomspace(5.0, 8.0 * self.R500, 320)
        rho = beta_model_rho(rg, 1.0, self.rc_gas, self.beta_gas)
        Mg = _cum_mass_from_rho(rg, rho)
        Mg = Mg / np.interp(self.R500, rg, Mg) * self.Mgas
        Mstar_mem = np.zeros_like(rg)
        rmem = np.linalg.norm(self.mem_xyz, axis=1)
        for i, r in enumerate(rg):
            Mstar_mem[i] = self.mem_m[rmem <= r].sum()
        Micl = hernquist_M(rg, self.M_icl, self.a_icl)
        Mbcg = hernquist_M(rg, self.M_bcg, self.a_bcg)
        self._rg = rg
        self._Mb = Mg + Mstar_mem + Micl + Mbcg
        self._Mgas_grid = Mg

    @property
    def Mbar500(self):
        return float(np.interp(self.R500, self._rg, self._Mb))

    def Mbar(self, r):
        return np.interp(np.asarray(r, float), self._rg, self._Mb)

    def Mgas_enc(self, r):
        return np.interp(np.asarray(r, float), self._rg, self._Mgas_grid)

    def gN(self, r):
        r = np.asarray(r, float)
        return G * self.Mbar(r) / r ** 2

    def phi_depth(self, r):
        """|Phi(r) - Phi(R_bnd)|, declared primary rule R_bnd = 3 * R500."""
        r = np.atleast_1d(np.asarray(r, float))
        rb = 3.0 * self.R500
        rg = np.geomspace(min(r.min(), 5.0), rb, 256)
        g = self.gN(rg)
        seg = 0.5 * (g[1:] + g[:-1]) * np.diff(rg)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        return np.interp(r, rg, cum[-1] - cum)

    # -- the well network, evaluated as a FIELD (coarse-graining stable) ----
    def S_lambda(self, xyz, lam):
        """Directionless inverse-square well strength, softened at lam.

        S(x) = sum_a G M_a / (d_a^2 + lam^2).  Linear in mass with a smooth
        kernel, so it converges when a source is subdivided at scales << lam.
        It derives from the symmetric pair energy sum_{a<b} G M_a M_b /
        (d_ab^2 + lam^2)^{1/2}-like construction, hence reciprocal.
        """
        xyz = np.atleast_2d(np.asarray(xyz, float))
        d2 = ((xyz[:, None, :] - self.mem_xyz[None, :, :]) ** 2).sum(-1)
        S = (G * self.mem_m[None, :] / (d2 + lam * lam)).sum(1)
        # BCG + ICL + gas treated as a single central smooth well
        r2 = (xyz ** 2).sum(1)
        S = S + G * (self.M_bcg + self.M_icl) / (r2 + lam * lam)
        return S

    def smoothed(self):
        """Control: same radial mass profile, members angularly averaged."""
        import copy
        c = copy.copy(self)
        r = np.linalg.norm(self.mem_xyz, axis=1)
        u = np.random.default_rng(0).normal(size=(len(r), 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        c.mem_xyz = u * r[:, None]
        return c
