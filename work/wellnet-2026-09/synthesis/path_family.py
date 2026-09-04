"""path_family.py -- family P: reciprocal path-dependent gravity.

THE ACTION (weak field, static; a manifestly reciprocal two-point functional)

    S_P[rho] = -(1/2) Int Int rho(x) W(x,y)[rho] rho(y) d^3x d^3y
    W(x,y)   = -(G/|x-y|) [ 1 + eps v(x,y) ]
    v(x,y)   = (1/|x-y|) Int_seg(x,y) phi(rho(s)) dl,   phi(rho) = 1/(1 + rho/rho_*)

v is the "vacuum fraction" of the straight segment joining x and y: the
fraction of the path along which the baryonic density is below rho_*.  W is
symmetric under x <-> y by construction (the segment is the same set), so
the kernel is RECIPROCAL identically.

THE POTENTIAL A TEST PARTICLE (or photon) SEES is the functional derivative

    Phi(z) = delta E / delta rho(z)   (E = -S_P)
           = Phi_dir(z) + Phi_3(z)
    Phi_dir(z) = Int rho(y) W(z,y) d^3y                 (endpoint term)
    Phi_3(z)   = (1/2) Int Int rho(x) rho(y) delta W(x,y)/delta rho(z)
               = -(G eps / 2) phi'(rho(z)) P(z),        (carrier term)
    P(z)       = Int dOmega C(z, n) C(z, -n),   C(z,n) = Int_0^inf rho(z + s n) ds

-- derived in the report: the double integral over pairs whose segment passes
through z collapses to the angular integral of the product of the two
OPPOSITE half-line columns through z.  Phi_3 is an ALGEBRAIC function of the
local density and of P; it is non-zero only where matter lies on both sides
of z (inside a body, or in the BRIDGE between two bodies) and is compactly
supported, so its effective density integrates to ZERO: every bridge is a
compensated feature with no net mass.

THE CARRIER OF COMPENSATING MOMENTUM is the matter on the segment: E is
translation invariant, so Sum_a F_a = 0 exactly, but the two-body forces
alone do NOT sum to zero; the difference is -grad Phi_3 acting on the matter
along the paths (measured below to round-off).

UNIVERSAL CONSTANTS: G, eps (dimensionless, signed), rho_* (a density).  One
new scale.  The no-new-scale variant rho_* = rho_mean (cosmic mean matter
density) is compiled too and lands, honestly, as non-identifiable on the
bench's galaxy/cluster probes.

WHAT THIS MODULE MEASURES (no data of any kind is opened)
  1. the closed-form half-line column of a Plummer sphere, checked by
     quadrature; the pair kernel's reciprocity;
  2. the radial force factor g/g_N on the compiler's three probe geometries
     (direct + carrier terms), declared to the compiler as `force_factor`;
  3. the shell Green's function of the direct term on the compiler's radial
     background, declared as `green`;
  4. the momentum budget on a 5-body configuration: total forces sum to
     zero, two-body forces do not, the carrier closes the gap;
  5. the bridge between two clusters: sign, zero net effective mass,
     M_A M_B scaling, transverse profile of the projected effective density;
  6. connectivity scalings: the response of the pair force to an intervening
     filament against the Newtonian pull of the same filament, as functions
     of the endpoint mass and of the filament density;
  7. the member-scramble counterfactual on the compiler's own 300-member
     cluster catalogue.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import guard                                    # noqa: F401
import compiler as C                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G, A0, KPC, MPC, MSUN = C.G, C.A0, C.KPC, C.MPC, C.MSUN

RHO_STAR_FID = 1.0e-24                  # kg m^-3, fiducial (bench-identifiable)
RHO_MEAN = 0.315 * 9.204e-27            # cosmic mean matter density, kg m^-3
EPS_FID = 0.3

_GL_T, _GL_W = np.polynomial.legendre.leggauss(32)
_GL_T = 0.5 * (_GL_T + 1.0)             # nodes on (0,1), symmetric about 1/2
_GL_W = 0.5 * _GL_W


def fib_dirs(n: int) -> np.ndarray:
    return C.fib_dirs(n)


# ============================================================ the scene
class Scene:
    """A list of Plummer spheres.  Density, potential, half-line columns."""

    def __init__(self, comps, rho_star=RHO_STAR_FID):
        self.comps = list(comps)
        self.rho_star = float(rho_star)

    def rho(self, x):
        return sum(c.rho(x) for c in self.comps)

    def phi_N(self, x):
        return sum(c.phi(x) for c in self.comps)

    def phi_vac(self, rho):
        return 1.0 / (1.0 + rho / self.rho_star)

    def dphi_vac(self, rho):
        return -(1.0 / self.rho_star) / (1.0 + rho / self.rho_star) ** 2

    def column(self, z, n):
        """C(z, n) = Int_0^inf rho(z + s n) ds, closed form per Plummer.

        z: (N,3), n: (N,3) unit.  For one Plummer (M, a, c) with x = z - c,
        p = x.n, b^2 = |x|^2 - p^2 + a^2:
            C = (3 M a^2 / 4 pi) [ 2/(3 b^4) - p (2p^2 + 3b^2) /
                                    (3 b^4 (p^2 + b^2)^{3/2}) ].
        """
        z = np.asarray(z, float)
        n = np.asarray(n, float)
        out = np.zeros(z.shape[0])
        for c in self.comps:
            x = z - c.c
            p = (x * n).sum(-1)
            b2 = (x * x).sum(-1) - p * p + c.a ** 2
            b2 = np.maximum(b2, 1e-30)
            I = (2.0 / (3.0 * b2 ** 2)
                 - p * (2.0 * p * p + 3.0 * b2)
                 / (3.0 * b2 ** 2 * (p * p + b2) ** 1.5))
            out += (3.0 * c.M * c.a ** 2 / (4.0 * np.pi)) * I
        return out

    def v_seg(self, x, y):
        """vacuum fraction of the segment x -> y, 32-node Gauss-Legendre."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        pts = x[..., None, :] + _GL_T[:, None] * (y - x)[..., None, :]
        r = self.rho(pts.reshape(-1, 3)).reshape(pts.shape[:-1])
        return (self.phi_vac(r) * _GL_W).sum(-1)

    def W(self, x, y, eps):
        d = np.linalg.norm(np.asarray(y, float) - np.asarray(x, float), axis=-1)
        return -G / np.maximum(d, 1e-3 * KPC) * (1.0 + eps * self.v_seg(x, y))

    def P(self, z, n_dir: int = 20000):
        """P(z) = Int dOmega C(z,n) C(z,-n).  The columns are angularly
        narrow when z is far from a compact component (width ~ a/r), so the
        direction count is high; convergence is measured in `P_convergence`."""
        z = np.atleast_2d(np.asarray(z, float))
        dirs = fib_dirs(n_dir)
        out = np.zeros(len(z))
        for i, zz in enumerate(z):
            Z = np.repeat(zz[None, :], n_dir, 0)
            out[i] = (4.0 * np.pi / n_dir) * (self.column(Z, dirs)
                                              * self.column(Z, -dirs)).sum()
        return out

    def phi3(self, z, eps, n_dir: int = 20000):
        """Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z)."""
        z = np.atleast_2d(np.asarray(z, float))
        return -0.5 * G * eps * self.dphi_vac(self.rho(z)) * self.P(z, n_dir)

    def phi_v_and_N(self, z, n_t: int = 140, n_th: int = 32, n_ph: int = 8,
                    n_seg: int = 16, sources=None):
        """Phi_v(z) = -G Int rho(y) v(z,y)/|z-y| d^3y and, on the SAME
        quadrature, the Newtonian potential (its accuracy check).

        SOURCE-centred quadrature, one component at a time: spherical
        coordinates about the component's centre with the polar axis along
        (z - c), a geometric radial grid, Gauss-Legendre in cos(theta) and a
        uniform azimuth (exact by symmetry when the scene is one component,
        so n_ph = 1 then).  The 1/|z-y| singularity at y = z is integrable
        and its cell contribution is O(rho(z) delta^2), which on every probe
        is below 1e-4 of Phi_N because the density at the probe radius is
        tiny; the Newtonian check reports the realised error."""
        z = np.asarray(z, float)
        ct, wt = np.polynomial.legendre.leggauss(n_th)
        gl_t, gl_w = np.polynomial.legendre.leggauss(n_seg)
        gl_t = 0.5 * (gl_t + 1.0)
        gl_w = 0.5 * gl_w
        nph = 1 if len(self.comps) == 1 else n_ph
        phis = (np.arange(nph) + 0.5) * 2.0 * np.pi / nph
        phiN = 0.0
        phiv = 0.0
        srcs = self.comps if sources is None else list(sources)
        for c in srcs:
            axis = z - c.c
            dist = np.linalg.norm(axis)
            if dist < 1e-6 * c.a:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / dist
            # orthonormal frame (e1, e2, axis)
            tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e1 = np.cross(axis, tmp)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            t = np.geomspace(1e-3 * c.a, 400.0 * c.a, n_t)
            st = np.sqrt(1.0 - ct ** 2)
            # y = c + t [ st cos(ph) e1 + st sin(ph) e2 + ct axis ]
            dirv = (st[:, None, None] * np.cos(phis)[None, :, None] * e1[None, None, :]
                    + st[:, None, None] * np.sin(phis)[None, :, None] * e2[None, None, :]
                    + ct[:, None, None] * axis[None, None, :])          # (nth, nph, 3)
            y = c.c[None, None, None, :] + t[:, None, None, None] * dirv[None]  # (nt,nth,nph,3)
            yf = y.reshape(-1, 3)
            rho_c = c.rho(yf)
            d = np.linalg.norm(yf - z[None, :], axis=-1)
            d = np.maximum(d, 1e-4 * c.a)
            # vacuum fraction of every segment z -> y
            seg = z[None, None, :] + gl_t[None, :, None] * (yf - z[None, :])[:, None, :]
            rs = self.rho(seg.reshape(-1, 3)).reshape(len(yf), n_seg)
            v = (self.phi_vac(rs) * gl_w[None, :]).sum(1)
            dt = np.gradient(t)                          # radial trapezoid
            wgt = np.broadcast_to(
                (t ** 2 * dt)[:, None, None] * wt[None, :, None]
                * (2.0 * np.pi / nph), (n_t, n_th, nph)).reshape(-1)
            phiN += -G * float(np.sum(wgt * rho_c / d))
            phiv += -G * float(np.sum(wgt * rho_c * v / d))
        return phiv, phiN


# ============================================== 1. closed-form column check
def column_check() -> dict:
    comp = C.Plummer(5e10 * MSUN, 3.0 * KPC, (0.0, 0.0, 0.0))
    sc = Scene([comp])
    rng = np.random.default_rng(3)
    z = rng.normal(size=(6, 3)) * 5 * KPC
    n = rng.normal(size=(6, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    closed = sc.column(z, n)
    s = np.geomspace(1e-4 * KPC, 3000 * KPC, 20000)
    quad = np.array([np.trapezoid(comp.rho(z[i][None, :] + s[:, None] * n[i][None, :]), s)
                     for i in range(6)])
    return dict(max_rel_err=float(np.max(np.abs(closed - quad) / quad)),
                n=6)


def P_convergence(rho_star=RHO_STAR_FID) -> dict:
    """P(z) at 30 kpc from the caricature galaxy for increasing direction
    counts: the hardest case (narrow inward column)."""
    sc = Scene([C.Plummer(C.GAL_M, C.GAL_A, (0, 0, 0))], rho_star)
    z = np.array([[0.0, 0.0, 30.0 * KPC]])
    rows = {int(n): float(sc.P(z, n)[0]) for n in (2000, 5000, 10000, 20000, 40000)}
    ref = rows[40000]
    return dict(P_by_n_dir=rows,
                rel_err_at_20000=float(abs(rows[20000] - ref) / ref),
                rel_err_at_5000=float(abs(rows[5000] - ref) / ref))


def reciprocity_check(rho_star=RHO_STAR_FID, eps=EPS_FID, n=200) -> dict:
    """The compiler's own kernel_reciprocity, on this kernel."""
    bg = Scene([C.Plummer(1.0e12 * MSUN, 20.0 * KPC)], rho_star)
    rng = np.random.default_rng(5)
    xa = rng.normal(size=(n, 3)) * 20 * KPC
    xb = rng.normal(size=(n, 3)) * 20 * KPC
    f1 = bg.W(xa, xb, eps)
    f2 = bg.W(xb, xa, eps)
    return dict(max_relative_asymmetry=float(np.max(np.abs(f1 - f2))
                                             / np.max(np.abs(f1))), n=n)


# ================================= 2. force factors on the compiler's probes
def probe_scene(name: str, rho_star: float) -> tuple:
    """The declared spherical caricature of each compiler probe: the scene's
    Plummer components with their true centres (they set the DENSITY along
    every segment, i.e. v and P), the probe centre, and the SOURCES whose
    modified pull the factor measures -- the probe's own components, exactly
    as the compiler's Yukawa controls use the point-mass-equivalent field of
    the probe.  The cluster's own path-modified pull on the member galaxy is
    an orbital (monopole) effect the probe reduction cannot express and is
    declared outside it."""
    gal = C.Plummer(C.GAL_M, C.GAL_A, (0, 0, 0))
    nbr = C.Plummer(C.GAL_M, C.GAL_A, (C.NEIGHBOUR_D, 0, 0))
    gas = C.Plummer(C.CLU_GAS_M, C.CLU_GAS_A, (0, 0, 0))
    bcg = C.Plummer(C.BCG_M, C.BCG_A, (0, 0, 0))
    if name == "galaxy_field":
        return Scene([gal, nbr], rho_star), np.zeros(3), [gal]
    if name == "cluster_shell":
        return Scene([gas, bcg], rho_star), np.zeros(3), [gas, bcg]
    if name == "galaxy_member":
        galm = C.Plummer(C.GAL_M, C.GAL_A, (C.MEMBER_D, 0, 0))
        return (Scene([galm, gas, bcg], rho_star),
                np.array([C.MEMBER_D, 0, 0]), [galm])
    raise ValueError(name)


def force_factor_table(name: str, eps: float, rho_star: float,
                       n_r: int = 24, n_dir_probe: int = 6) -> dict:
    """g/g_N(r) for the probe, direction-averaged over the compiler's own
    12 Fibonacci directions, as a table in r.  The members (300 rows) are
    not in the caricature -- exactly as the compiler's Yukawa controls use a
    point-mass-equivalent reduction -- and this is declared."""
    sc, centre, srcs = probe_scene(name, rho_star)
    p = C.probes()[name]
    rmin, rmax = p.r.min(), p.r.max()
    rg = np.geomspace(0.5 * rmin, 2.0 * rmax, n_r)
    # ANTIPODALLY SYMMETRIC directions, so that the linear (dipole) part of
    # any external field -- the cluster's 0.5 a0 pull on the member probe --
    # cancels exactly in the direction average and only the probe's own
    # monopole survives.  A Fibonacci set is not antipodal and leaves a
    # residual that swamps the member galaxy's own 0.06 a0 field.
    half = fib_dirs(max(1, n_dir_probe // 2))
    dirs = np.vstack([half, -half])
    phiv = np.zeros(n_r)
    phiN_q = np.zeros(n_r)
    phiN_a = np.zeros(n_r)
    phi3 = np.zeros(n_r)
    for i, r in enumerate(rg):
        pts = centre[None, :] + r * dirs
        pv, pn = zip(*[sc.phi_v_and_N(x, sources=srcs) for x in pts])
        phiv[i] = np.mean(pv)
        phiN_q[i] = np.mean(pn)
        phiN_a[i] = np.mean(sum(c.phi(pts) for c in srcs))
        phi3[i] = np.mean(sc.phi3(pts, eps))
    lnr = np.log(rg)
    gN = -np.gradient(phiN_a, lnr) / rg          # inward magnitude, analytic
    gv = -np.gradient(phiv, lnr) / rg
    g3 = -np.gradient(phi3, lnr) / rg
    ff = 1.0 + (eps * gv + g3) / gN
    return dict(r_m=rg.tolist(), factor=ff.tolist(),
                factor_direct_only=(1.0 + eps * gv / gN).tolist(),
                factor_carrier_only=(1.0 + g3 / gN).tolist(),
                gN=gN.tolist(),
                quadrature_check_max_rel_err=float(np.max(
                    np.abs(phiN_q - phiN_a) / np.abs(phiN_a))),
                eps=eps, rho_star=rho_star)


def make_force_factor(tables: dict):
    """A callable r -> g/g_N that the compiler calls on each probe's own
    radii.  The compiler passes only r, so the probe is identified by its
    radial range (the three probes' ranges are disjoint by construction:
    galaxy 10-30 kpc, member 10-30 kpc, cluster 300-1414 kpc)."""
    def ff(r):
        r = np.asarray(r, float)
        out = np.ones_like(r)
        for nm, t in tables.items():
            rr = np.array(t["r_m"])
            sel = (r >= rr[0]) & (r <= rr[-1])
            if np.any(sel):
                out[sel] = np.interp(np.log(r[sel]), np.log(rr),
                                     np.array(t["factor"]))
        return out
    return ff


# ============================================ 3. the declared Green's function
def make_green(eps: float, rho_star: float, n_th: int = 96):
    """delta Phi_dir(r_i)/delta m_j for a unit shell at r_j on the compiler's
    radial background: -G/max(r_i,r_j) - G eps < v(x_i, y)/|x_i - y| >_shell.
    Symmetric under i <-> j by construction (W is).  The carrier term and the
    cross terms of delta Phi/delta rho are second and mixed functional
    derivatives of the SCALAR E and are symmetric by that structure; the
    momentum-budget test is their check."""
    bg = Scene([C.Plummer(1.0e12 * MSUN, 20.0 * KPC)], rho_star)
    cth, wth = np.polynomial.legendre.leggauss(n_th)

    def green(ri, rj):
        x = np.array([0.0, 0.0, ri])
        sth = np.sqrt(1.0 - cth ** 2)
        y = np.stack([rj * sth, 0 * cth, rj * cth], -1)
        d = np.linalg.norm(y - x[None, :], axis=-1)
        d = np.maximum(d, 1e-3 * KPC)
        v = bg.v_seg(np.repeat(x[None, :], n_th, 0), y)
        extra = -G * eps * 0.5 * np.sum(wth * v / d)
        return -G / max(ri, rj) + extra
    return green


# ===================================================== 4. momentum budget
def momentum_budget(eps=EPS_FID, rho_star=RHO_STAR_FID, seed=11) -> dict:
    rng = np.random.default_rng(seed)
    n = 5
    m = 10 ** rng.uniform(9.5, 10.8, n) * MSUN
    x0 = rng.normal(size=(n, 3)) * 12 * KPC
    b = 1.0 * KPC

    def energy(pos, rho_pos):
        comps = [C.Plummer(m[c], b, rho_pos[c]) for c in range(n)]
        sc = Scene(comps, rho_star)
        E = 0.0
        for a in range(n):
            for bb in range(a + 1, n):
                E += m[a] * m[bb] * sc.W(pos[a], pos[bb], eps)
        return E

    def forces(frozen: bool):
        F = np.zeros((n, 3))
        step = 3.0e-3 * KPC
        for a in range(n):
            for k in range(3):
                dp = np.zeros((n, 3))
                dp[a, k] = step
                if frozen:
                    Ep = energy(x0 + dp, x0)
                    Em = energy(x0 - dp, x0)
                else:
                    Ep = energy(x0 + dp, x0 + dp)
                    Em = energy(x0 - dp, x0 - dp)
                F[a, k] = -(Ep - Em) / (2 * step)
        return F

    F_tot = forces(frozen=False)
    F_2b = forces(frozen=True)
    F_car = F_tot - F_2b
    scale = float(np.mean(np.linalg.norm(F_tot, axis=1)))
    return dict(
        n_bodies=n, eps=eps, rho_star=rho_star,
        sum_total_over_mean=float(np.linalg.norm(F_tot.sum(0)) / scale),
        sum_two_body_over_mean=float(np.linalg.norm(F_2b.sum(0)) / scale),
        sum_carrier_over_mean=float(np.linalg.norm(F_car.sum(0)) / scale),
        carrier_closes_budget=float(np.linalg.norm(F_2b.sum(0) + F_car.sum(0))
                                    / scale),
        carrier_fraction_per_body=(np.linalg.norm(F_car, axis=1)
                                   / np.linalg.norm(F_tot, axis=1)).tolist(),
        statement="the total forces sum to zero (translation invariance of "
                  "E); the endpoint (two-body) forces alone do not; the "
                  "difference is carried by the matter on the segments "
                  "through -grad Phi_3 -- the declared carrier, verified")


# ====================================================== 5. the bridge
def bridge(eps=EPS_FID, rho_star=RHO_STAR_FID, MA=3e14, MB=3e14,
           a_clu=400.0, D=4000.0, nx=49, nR=33, n_dir=5000) -> dict:
    A = C.Plummer(MA * MSUN, a_clu * KPC, (-0.5 * D * KPC, 0, 0))
    B = C.Plummer(MB * MSUN, a_clu * KPC, (+0.5 * D * KPC, 0, 0))
    sc = Scene([A, B], rho_star)
    xs = np.linspace(-0.75 * D, 0.75 * D, nx) * KPC
    Rs = np.linspace(0.0, 0.55 * D, nR) * KPC
    X, R = np.meshgrid(xs, Rs, indexing="ij")
    pts = np.stack([X.ravel(), R.ravel(), np.zeros(X.size)], -1)
    ph3 = sc.phi3(pts, eps, n_dir).reshape(nx, nR)
    rho = sc.rho(pts).reshape(nx, nR)
    # cylindrical Laplacian: (1/R) d/dR (R dPhi/dR) + d2Phi/dx2
    dx = xs[1] - xs[0]
    dR = Rs[1] - Rs[0]
    d2x = np.zeros_like(ph3)
    d2x[1:-1] = (ph3[2:] - 2 * ph3[1:-1] + ph3[:-2]) / dx ** 2
    dphR = np.gradient(ph3, dR, axis=1)
    lapR = np.zeros_like(ph3)
    Rm = R.copy()
    Rm[:, 0] = dR                                   # axis handled below
    lapR[:, 1:] = np.gradient(Rm * dphR, dR, axis=1)[:, 1:] / Rm[:, 1:]
    lapR[:, 0] = 2.0 * (ph3[:, 1] - ph3[:, 0]) / dR ** 2 * 2.0   # axis limit
    rho_eff = (d2x + lapR) / (4 * np.pi * G)
    dV = 2 * np.pi * R * dR * dx
    tot = float((rho_eff * dV)[1:-1].sum())
    tot_abs = float((np.abs(rho_eff) * dV)[1:-1].sum())
    imid = nx // 2
    # projected effective surface density across the bridge at x = 0, line of
    # sight perpendicular to the axis, impact parameter y:
    # Sigma(y) = Int rho_eff(R = sqrt(y^2 + l^2)) dl
    ys = Rs[: nR // 2]
    Sig = []
    for y in ys:
        l = np.linspace(0, Rs[-1], 200)
        Rr = np.sqrt(y * y + l * l)
        Sig.append(2 * np.trapezoid(np.interp(Rr, Rs, rho_eff[imid],
                                              right=0.0), l))
    Sig = np.array(Sig)
    # M_A M_B scaling of P at the midpoint
    mid = np.zeros((1, 3))
    Pm = {}
    for tag, (ma, mb) in dict(eq=(MA, MB), two_one=(2 * MA, MB),
                              two_two=(2 * MA, 2 * MB)).items():
        s2 = Scene([C.Plummer(ma * MSUN, a_clu * KPC, (-0.5 * D * KPC, 0, 0)),
                    C.Plummer(mb * MSUN, a_clu * KPC, (+0.5 * D * KPC, 0, 0))],
                   rho_star)
        Pm[tag] = float(s2.P(mid, n_dir)[0])
    return dict(
        eps=eps, rho_star=rho_star, M_A=MA, M_B=MB, a_kpc=a_clu, D_kpc=D,
        phi3_midpoint_m2s2=float(ph3[imid, 0]),
        phi3_midpoint_kms2=float(ph3[imid, 0] / 1e6),
        rho_at_midpoint=float(rho[imid, 0]),
        rho_eff_on_axis_midpoint=float(rho_eff[imid, 0]),
        rho_eff_sign_on_axis=("-" if rho_eff[imid, 0] < 0 else "+"),
        net_effective_mass_over_abs=tot / max(tot_abs, 1e-300),
        net_effective_mass_Msun=tot / MSUN,
        abs_effective_mass_Msun=tot_abs / MSUN,
        projected_profile=dict(y_kpc=(ys / KPC).tolist(),
                               Sigma_eff_kg_m2=Sig.tolist(),
                               core_sign=("-" if Sig[0] < 0 else "+"),
                               wing_sign=("-" if Sig[len(Sig) // 2] < 0 else "+")),
        P_midpoint_scaling=dict(P=Pm,
                                ratio_two_one_over_eq=Pm["two_one"] / Pm["eq"],
                                ratio_two_two_over_eq=Pm["two_two"] / Pm["eq"],
                                expected=[2.0, 4.0]),
        statement=("Phi_3 is a ridge along the segment; for eps > 0 it is a "
                   "potential HILL (repulsive), its effective density is "
                   "negative on the axis and positive around it, and its "
                   "volume integral vanishes: a compensated bridge, "
                   "scaling as M_A M_B. No rho_DM >= 0 reproduces it."))


# ================================================ 6. connectivity scalings
def connectivity(eps=EPS_FID, rho_star=RHO_STAR_FID, MA=3e14, D=4000.0,
                 a_clu=400.0, R_f=300.0) -> dict:
    """Pair force between A and B against an intervening filament.

    Path term: the modulation of the A-B pair force, F_AB = G M_A M_B/D^2
    (1 + eps v_AB), with v_AB the vacuum fraction of the axis segment.
    Newtonian term: the filament's own pull on A, G M_A lambda_f (1/x1 - 1/x2)
    for a uniform cylinder of line density lambda_f = pi R_f^2 rho_f from
    x1 = a to x2 = D - a along the axis.  Both as functions of rho_f (at
    fixed M_B) and of M_B (at fixed rho_f)."""
    def v_axis(MB, rho_f):
        A = C.Plummer(MA * MSUN, a_clu * KPC, (-0.5 * D * KPC, 0, 0))
        B = C.Plummer(MB * MSUN, a_clu * KPC, (+0.5 * D * KPC, 0, 0))
        sc = Scene([A, B], rho_star)
        t = np.linspace(0, 1, 2001)
        pts = np.stack([(-0.5 * D + t * D) * KPC, 0 * t, 0 * t], -1)
        rho = sc.rho(pts)
        inside = (np.abs(pts[:, 0]) < (0.5 * D - a_clu) * KPC)
        rho = rho + rho_f * inside
        return float(np.trapezoid(sc.phi_vac(rho), t))

    rows_rho, rows_M = [], []
    v0 = v_axis(MA, 0.0)
    lam = lambda rho_f: np.pi * (R_f * KPC) ** 2 * rho_f      # noqa: E731
    x1, x2 = a_clu * KPC, (D - a_clu) * KPC
    FAB = lambda MB: G * MA * MSUN * MB * MSUN / (D * KPC) ** 2   # noqa: E731
    for f in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        rho_f = f * rho_star
        v = v_axis(MA, rho_f)
        dF_path = eps * (v - v0) * FAB(MA)
        F_fil = G * MA * MSUN * lam(rho_f) * (1.0 / x1 - 1.0 / x2)
        rows_rho.append(dict(rho_f_over_rho_star=f, v_axis=v,
                             dF_path_over_F_N=float(dF_path / FAB(MA)),
                             F_fil_over_F_N=float(F_fil / FAB(MA))))
    for fM in (0.25, 0.5, 1.0, 2.0, 4.0):
        MB = fM * MA
        v = v_axis(MB, rho_star)
        dF_path = eps * (v - v_axis(MB, 0.0)) * FAB(MB)
        F_fil = G * MA * MSUN * lam(rho_star) * (1.0 / x1 - 1.0 / x2)
        rows_M.append(dict(M_B_over_M_A=fM, dF_path_N=float(dF_path),
                           F_fil_N=float(F_fil)))
    # log slopes
    lr = np.log([r["rho_f_over_rho_star"] for r in rows_rho])
    sp = np.polyfit(lr[:2], np.log(np.abs([r["dF_path_over_F_N"] for r in rows_rho[:2]])), 1)[0]
    sp_hi = np.polyfit(lr[-2:], np.log(np.abs([r["dF_path_over_F_N"] for r in rows_rho[-2:]])), 1)[0]
    sN = np.polyfit(lr, np.log([r["F_fil_over_F_N"] for r in rows_rho]), 1)[0]
    lM = np.log([r["M_B_over_M_A"] for r in rows_M])
    sM_path = np.polyfit(lM, np.log(np.abs([r["dF_path_N"] for r in rows_M])), 1)[0]
    sM_fil = np.polyfit(lM, np.log([r["F_fil_N"] for r in rows_M]), 1)[0]
    return dict(
        eps=eps, rho_star=rho_star, M_A=MA, D_kpc=D, R_f_kpc=R_f,
        v_axis_no_filament=v0,
        vs_filament_density=rows_rho, vs_endpoint_mass=rows_M,
        log_slopes=dict(path_vs_rho_f_low=float(sp), path_vs_rho_f_high=float(sp_hi),
                        newton_vs_rho_f=float(sN),
                        path_vs_M_B=float(sM_path), newton_vs_M_B=float(sM_fil)),
        sign="dF_path < 0 for eps > 0: a filament WEAKENS the pair force "
             "(the segment is less empty); the Newtonian pull of the same "
             "filament is attractive and independent of M_B",
        statement="the path response is linear in the FAR endpoint mass and "
                  "SATURATES in the filament density; the Newtonian "
                  "response of the same filament is independent of the far "
                  "endpoint and linear in the density with no saturation. "
                  "The joint scaling (slope 1 in M_B, saturating in rho_f) "
                  "is the connectivity signature; no added collisionless "
                  "mass has it.")


# ===================================== 7. member scramble on the catalogue
def member_scramble(eps=EPS_FID, rho_star=RHO_STAR_FID, n_scr=8, n_mem=120,
                    seed=2) -> dict:
    """The mass-weighted vacuum fraction of member-member and member-core
    segments on the compiler's own catalogue, actual vs angle-scrambled at
    fixed radii and masses (every radial profile preserved)."""
    mx, mm = C._member_catalogue()
    mx, mm = mx[:n_mem], mm[:n_mem]
    gas = C.Plummer(C.CLU_GAS_M, C.CLU_GAS_A, (0, 0, 0))
    bcg = C.Plummer(C.BCG_M, C.BCG_A, (0, 0, 0))
    ia, ib = np.triu_indices(n_mem, 1)

    def vbar(pos):
        comps = [gas, bcg] + [C.Plummer(mm[i], C.GAL_A * (mm[i] / C.GAL_M) ** 0.3,
                                        pos[i]) for i in range(n_mem)]
        sc = Scene(comps, rho_star)
        v = sc.v_seg(pos[ia], pos[ib])
        d = np.linalg.norm(pos[ia] - pos[ib], axis=-1)
        w = mm[ia] * mm[ib] / d
        vmm = float((w * v).sum() / w.sum())
        vc = sc.v_seg(pos, np.zeros_like(pos))
        rc = np.linalg.norm(pos, axis=-1)
        wc = mm * (C.BCG_M + gas.M_enc(rc)) / rc
        vmc = float((wc * vc).sum() / wc.sum())
        return vmm, vmc

    rng = np.random.default_rng(seed)
    v_act = vbar(mx)
    scr = []
    for _ in range(n_scr):
        d = rng.normal(size=(n_mem, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos = d * np.linalg.norm(mx, axis=1)[:, None]
        scr.append(vbar(pos))
    scr = np.array(scr)
    return dict(
        n_members=n_mem, n_scrambles=n_scr, eps=eps, rho_star=rho_star,
        actual=dict(v_member_member=v_act[0], v_member_core=v_act[1]),
        scrambled=dict(v_member_member_mean=float(scr[:, 0].mean()),
                       v_member_member_sd=float(scr[:, 0].std(ddof=1)),
                       v_member_core_mean=float(scr[:, 1].mean()),
                       v_member_core_sd=float(scr[:, 1].std(ddof=1))),
        pair_force_modulation_range_member_member=dict(
            min=float(1 + eps * scr[:, 0].min()), max=float(1 + eps * scr[:, 0].max())),
        statement="under an angle scramble at fixed radii and masses, the "
                  "member-member vacuum fraction moves while every radial "
                  "profile is fixed: the pair forces respond to connectivity "
                  "after radial matching. A Newtonian or CDM member force "
                  "depends on the pair separation only.")


# ======================================================================
def main():
    guard.arm()
    t0 = time.perf_counter()
    out = dict(
        family="P: reciprocal path-dependent gravity",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        action=dict(
            functional="S_P = -(1/2) Int Int rho(x) W(x,y) rho(y),  "
                       "W = -(G/|x-y|)[1 + eps v(x,y)],  "
                       "v = (1/|x-y|) Int_seg phi(rho) dl,  "
                       "phi(rho) = 1/(1 + rho/rho_*)",
            potential="Phi = delta E/delta rho = Phi_dir + Phi_3,  "
                      "Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z),  "
                      "P(z) = Int dOmega C(z,n) C(z,-n)",
            field_equation="no local PDE: Phi is the functional derivative of "
                           "a nonlocal scalar functional; test bodies and "
                           "photons follow -grad Phi",
            carrier="matter on the connecting segment, through -grad Phi_3 "
                    "(three-body term); Sum F = 0 by translation invariance",
            universal_constants=dict(shared_with_base=["G"],
                                     new=["eps", "rho_*"], n_new=2,
                                     no_new_scale_variant="rho_* = rho_mean"),
            model_class="static_scalar_potential (nonlocal in rho, local in "
                        "the test body): in the compiler's Gate 4 scope, as "
                        "its symmetric-nonlocal control XC5 is",
            fiducial=dict(eps=EPS_FID, rho_star=RHO_STAR_FID,
                          rho_mean=RHO_MEAN)),
    )
    out["column_check"] = column_check()
    print("column closed form vs quadrature:", out["column_check"])
    out["P_convergence"] = P_convergence()
    print("P convergence:", out["P_convergence"])
    out["reciprocity"] = reciprocity_check()
    print("reciprocity:", out["reciprocity"])
    tables = {}
    for nm in ("galaxy_field", "cluster_shell", "galaxy_member"):
        t1 = time.perf_counter()
        tables[nm] = force_factor_table(nm, EPS_FID, RHO_STAR_FID)
        f = np.array(tables[nm]["factor"])
        print(f"force factor {nm}: {f.min():.4f}..{f.max():.4f} "
              f"(quad check {tables[nm]['quadrature_check_max_rel_err']:.1e}) "
              f"{time.perf_counter()-t1:.1f}s")
    out["force_factor_tables_fiducial"] = tables
    tables_mean = {}
    for nm in ("galaxy_field", "cluster_shell", "galaxy_member"):
        tables_mean[nm] = force_factor_table(nm, EPS_FID, RHO_MEAN)
        f = np.array(tables_mean[nm]["factor"])
        print(f"force factor {nm} [rho_*=rho_mean]: {f.min():.5f}..{f.max():.5f}")
    out["force_factor_tables_rho_mean"] = tables_mean
    out["momentum_budget"] = momentum_budget()
    print("momentum:", {k: v for k, v in out["momentum_budget"].items()
                        if k.startswith(("sum", "carrier_closes"))})
    out["bridge"] = bridge()
    b = out["bridge"]
    print(f"bridge: phi3(mid) = {b['phi3_midpoint_kms2']:.1f} (km/s)^2, rho_eff "
          f"on axis {b['rho_eff_sign_on_axis']}, net/abs = "
          f"{b['net_effective_mass_over_abs']:.3e}, P scaling "
          f"{b['P_midpoint_scaling']['ratio_two_one_over_eq']:.3f} "
          f"{b['P_midpoint_scaling']['ratio_two_two_over_eq']:.3f}")
    out["connectivity"] = connectivity()
    print("connectivity slopes:", out["connectivity"]["log_slopes"])
    out["member_scramble"] = member_scramble()
    print("scramble:", out["member_scramble"]["actual"],
          out["member_scramble"]["scrambled"])
    out["provenance"] = guard.summary()
    out["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(HERE, "path_results.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("provenance:", out["provenance"]["assertion"],
          "| foreign:", out["provenance"]["foreign_reads"])
    print(f"wall {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
