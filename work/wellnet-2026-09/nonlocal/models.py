"""Spherical baryon models used by the nonlocal-kernel screen.

Each model is a sum of exponential spheres, rho = rho0 exp(-r/rs), for which
M(<r) = M [1 - e^-x (1 + x + x^2/2)], x = r/rs, and M = 8 pi rho0 rs^3.  An
exponential sphere is used rather than a disk because the headline result of
this lane is a statement about the boundedness of F, which is geometry
independent (it follows from qbar in [0,1)), and because the spherical
reduction removes the 1/|x-x'| singularity exactly.  Disk geometry changes the
normalisation of v_c by ~15 per cent, not its logarithmic slope.

THE JEANS SWINDLE, made explicit.  The kernel source is the density
PERTURBATION rho_b - rho_floor: a strictly uniform background exerts no net
force in an expanding universe and including it would make the outer force
grow linearly with r.  The q state, by contrast, is computed from the FULL
density including the floor, because q is a statement about how underdense a
place is relative to the cosmic mean, and a galaxy sitting in a mean-density
universe is nowhere underdense.  Those two facts pull in opposite directions
and their tension is the substance of the rotation-curve result.
"""
from __future__ import annotations

import math

import numpy as np

import nonlocal_kernel as NK


def exp_sphere_rho(r, M, rs):
    r = np.asarray(r, float)
    rho0 = M / (8.0 * math.pi * rs ** 3)
    return rho0 * np.exp(-r / rs)


def exp_sphere_M(r, M, rs):
    x = np.asarray(r, float) / rs
    return M * (1.0 - np.exp(-x) * (1.0 + x + 0.5 * x ** 2))


class Galaxy:
    """Stars + gas exponential spheres, optional uniform cosmic floor."""

    def __init__(self, name, Mstar, rd, Mgas=0.0, rg=None,
                 rho_floor=NK.RHO_BAR_B):
        self.name = name
        self.Mstar, self.rd = float(Mstar), float(rd)
        self.Mgas = float(Mgas)
        self.rg = float(rg) if rg is not None else 3.0 * float(rd)
        self.rho_floor = float(rho_floor)
        self.Mtot = self.Mstar + self.Mgas

    def rho_pert(self, r):
        return (exp_sphere_rho(r, self.Mstar, self.rd)
                + exp_sphere_rho(r, self.Mgas, self.rg))

    def rho_full(self, r):
        return self.rho_pert(r) + self.rho_floor

    def Menc(self, r):
        return (exp_sphere_M(r, self.Mstar, self.rd)
                + exp_sphere_M(r, self.Mgas, self.rg))

    def Sigma0_equiv(self):
        """Central surface density of the equivalent exponential disk,
        M/(2 pi rd^2), in Msun/pc^2 -- the SPARC-comparable quantity."""
        return self.Mstar / (2.0 * math.pi * self.rd ** 2) / 1.0e6


def build_field(gal: Galaxy, qdef="delta", rho_ref=NK.RHO_BAR_B, L_s=0.0,
                L_q=0.0, m=1.0, n=1.0, a0=NK.A0,
                r_lo=1e-3, r_hi=3.0e4, n_grid=2000):
    """Assemble a SphericalField with the requested void-state definition.

    qdef:
      'delta'   q = -delta/(1+delta) clipped to [0,1), delta from rho smoothed
                on scale L_s relative to rho_ref
      'smooth'  q = 1/(1 + (rho_s/rho_ref)^m)          (programme Q1)
      'screen'  (1 - L_q^2 lap) q = S(rho_s, g_N)      (programme Q3/Q4)
      'zero'    q = 0 everywhere (Newtonian control)
    """
    r = np.geomspace(r_lo, r_hi, n_grid)
    rho_p = gal.rho_pert(r)
    rho_f = gal.rho_full(r)
    Mfun = gal.Menc
    if L_s > 0:
        rho_s = NK.smooth_spherical(r, rho_f, L_s)
    else:
        rho_s = rho_f
    gN = NK.G * Mfun(r) / r ** 2
    if qdef == "zero":
        q = np.zeros_like(r)
    elif qdef == "delta":
        q = NK.q_from_delta(rho_s, rho_ref)
    elif qdef == "smooth":
        q = NK.q_from_smooth(rho_s, rho_ref, m=m)
    elif qdef == "screen":
        S = NK.q_source_Q3(rho_s, gN, rho_ref, m=m, a0=a0, n=n)
        q = NK.screen_spherical(r, S, L_q) if L_q > 0 else S
        q = np.clip(q, 0.0, 1.0 - 1e-12)
    else:
        raise KeyError(qdef)
    fld = NK.SphericalField(r=r, rho=rho_p, q=q,
                            rho_fun=gal.rho_pert, Menc_fun=Mfun,
                            label=f"{gal.name}|{qdef}")
    return fld


#: A ladder spanning the SPARC range in stellar mass, size and gas fraction.
GALAXY_LADDER = [
    Galaxy("dwarf_LSB", Mstar=3.0e8, rd=1.0, Mgas=1.2e9, rg=3.0),
    Galaxy("dwarf_HSB", Mstar=2.0e9, rd=1.2, Mgas=1.0e9, rg=3.6),
    Galaxy("LSB_large", Mstar=5.0e9, rd=6.0, Mgas=6.0e9, rg=18.0),
    Galaxy("spiral_mid", Mstar=1.5e10, rd=2.5, Mgas=5.0e9, rg=7.5),
    Galaxy("MW_like", Mstar=5.0e10, rd=3.0, Mgas=1.0e10, rg=9.0),
    Galaxy("massive", Mstar=2.0e11, rd=5.0, Mgas=1.0e10, rg=15.0),
]


class Cluster:
    """Beta-model ICM plus a stellar/galaxy component, spherical."""

    def __init__(self, name="cluster", Mgas=1.2e14, rc=200.0, beta=0.65,
                 rmax=3000.0, Mstar=1.5e13, rs_star=300.0,
                 rho_floor=NK.RHO_BAR_B):
        self.name = name
        self.rc, self.beta = float(rc), float(beta)
        self.rmax = float(rmax)
        self.Mgas = float(Mgas)
        self.Mstar, self.rs_star = float(Mstar), float(rs_star)
        self.rho_floor = float(rho_floor)
        # normalise the beta model so that M_gas(< rmax) = Mgas, then carry
        # the cumulative mass out to 30 Mpc so that M(<r) stays consistent
        # with rho(r) at every radius the potential integral touches.  A
        # cumulative mass truncated at rmax while rho keeps going is exactly
        # the kind of silent inconsistency that makes M_dyn/M_b come out
        # below one, which is impossible for alpha > 0.
        rr = np.geomspace(1.0, 3.0e4, 6000)
        shape = (1.0 + (rr / self.rc) ** 2) ** (-1.5 * self.beta)
        inr = rr <= self.rmax
        norm = np.trapezoid(4 * math.pi * rr[inr] ** 2 * shape[inr], rr[inr])
        self.rho0 = self.Mgas / norm
        self._rr = rr
        prof = 4 * math.pi * rr ** 2 * self.rho0 * shape
        self._Mgas_c = np.concatenate([[0.0], np.cumsum(
            0.5 * (prof[1:] + prof[:-1]) * np.diff(rr))])
        self.Mtot = self.Mgas + self.Mstar

    def rho_pert(self, r):
        r = np.asarray(r, float)
        gas = self.rho0 * (1.0 + (r / self.rc) ** 2) ** (-1.5 * self.beta)
        return gas + exp_sphere_rho(r, self.Mstar, self.rs_star)

    def rho_full(self, r):
        return self.rho_pert(r) + self.rho_floor

    def Menc(self, r):
        r = np.asarray(r, float)
        mg = np.interp(r, self._rr, self._Mgas_c)
        return mg + exp_sphere_M(r, self.Mstar, self.rs_star)


def sparc_equivalent_sphere(R, Vbar2, Mtot, r_tail, r_lo=1e-3, r_hi=3.0e4,
                            n=1400):
    """Spherical M(<r) that reproduces a measured baryonic curve exactly.

    Setting M(<R) = R V_bar^2 / G at the tabulated radii makes the NEWTONIAN
    circular speed of the model identical to the tabulated one by
    construction, which removes the baryon-geometry error from a forward
    comparison entirely.  Inside the first point the run is closed with a
    constant-density core, outside the last point M rises to the catalogue
    total on the scale r_tail.

    The price, stated rather than hidden: the equivalent spherical DENSITY is
    not the true three-dimensional density -- a disk's midplane density is
    several times higher at the same radius -- so the q field built from it is
    biased towards larger q.  It is nevertheless a far better proxy than an
    exponential sphere, whose Newtonian curve is wrong by 0.3 dex.
    """
    from scipy.interpolate import PchipInterpolator
    R = np.asarray(R, float)
    M = np.maximum.accumulate(R * np.maximum(Vbar2, 0.0) / NK.G)
    Mtot = max(float(Mtot), float(M[-1]))
    r = np.geomspace(r_lo, r_hi, n)
    knots_r = np.concatenate([[r_lo], R])
    knots_M = np.concatenate([[0.0], M])
    pch = PchipInterpolator(np.log(knots_r), knots_M, extrapolate=False)
    Mr = pch(np.log(np.clip(r, r_lo, R[-1])))
    inner = r < R[0]
    Mr[inner] = M[0] * (r[inner] / R[0]) ** 3
    outer = r > R[-1]
    Mr[outer] = Mtot - (Mtot - M[-1]) * np.exp(-(r[outer] - R[-1]) / r_tail)
    Mr = np.maximum.accumulate(np.nan_to_num(Mr))
    rho = np.gradient(Mr, r) / (4.0 * math.pi * r ** 2)
    rho = np.maximum(rho, 1e-30)
    Mfun = lambda x: np.interp(np.log(np.maximum(x, r_lo)), np.log(r), Mr)
    rfun = lambda x: np.exp(np.interp(np.log(np.maximum(x, r_lo)), np.log(r),
                                      np.log(rho)))
    return r, rho, Mr, rfun, Mfun, Mtot


def build_field_from_profile(r, rho, Mr, rfun, Mfun, qdef="screen",
                             rho_ref=NK.RHO_BAR_B, L_s=0.0, L_q=0.0, m=1.0,
                             n=1.0, a0=NK.A0, rho_floor=NK.RHO_BAR_B,
                             label=""):
    """SphericalField from an explicit profile, q built from rho + floor."""
    rho_f = rho + rho_floor
    rho_s = NK.smooth_spherical(r, rho_f, L_s) if L_s > 0 else rho_f
    gN = NK.G * Mr / r ** 2
    if qdef == "zero":
        q = np.zeros_like(r)
    elif qdef == "delta":
        q = NK.q_from_delta(rho_s, rho_ref)
    elif qdef == "smooth":
        q = NK.q_from_smooth(rho_s, rho_ref, m=m)
    elif qdef == "screen":
        S = NK.q_source_Q3(rho_s, gN, rho_ref, m=m, a0=a0, n=n)
        q = np.clip(NK.screen_spherical(r, S, L_q) if L_q > 0 else S,
                    0.0, 1.0 - 1e-12)
    else:
        raise KeyError(qdef)
    return NK.SphericalField(r=r, rho=rho, q=q, Menc=Mr.copy(),
                             rho_fun=rfun, Menc_fun=Mfun, label=label)
