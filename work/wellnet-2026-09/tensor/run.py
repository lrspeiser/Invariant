"""Driver: assemble a K field on the cluster grid and solve the field equation.

Everything the gates and the mechanism map need to do a full 3-D run lives
here so it is written once.
"""
from __future__ import annotations

import math

import numpy as np

import channels as CH
import cluster as CL
import field as F
import wellnet as W
from wellnet import G, A0, KPC, MSUN


def to_xp(a, xp):
    return a if xp is np else xp.asarray(a)


def newton_potential(c, xp=np, tol=1e-11):
    """Phi_N and |g_N| of the baryons on the grid (K = I, mu = 1)."""
    n, h = c["n"], c["dx"]
    rho = to_xp(c["rho"], xp)
    one = xp.ones(rho.shape)
    zer = xp.zeros(rho.shape)
    A = (one, one, one, zer, zer, zer)
    R = to_xp(c["R"], xp)
    Mtot = float(rho.sum() * h ** 3)
    bc = -G * Mtot / xp.maximum(R, 0.5 * h)
    Phi, it, rel = F.solve_linear(rho, A, h, bc, tol=tol, maxiter=6000, xp=xp)
    gmag, g = F.gradient_mag(Phi, h, xp)
    return Phi, gmag, g, dict(it=it, rel=rel)


def newton_analytic(points, c, soft=1.0 * KPC, xp=np, chunk=1 << 15):
    """Phi_N and |g_N| of (smooth gas) + (members as point masses).

    Used for the environmental gates instead of the grid Poisson solve,
    because the gates are evaluated 10 kpc from a galaxy centre as well as
    1 Mpc from a cluster centre, and a 94 kpc grid cell cannot represent the
    first.  The smooth part is exact for the spherical gas profile:

        Phi_gas(r) = -G M_gas(<r)/r - G Int_r^inf dM/r' .
    """
    rg = xp.asarray(c["r_gas"])
    Mg = xp.asarray(c["M_gas"])
    dM = xp.asarray(c["dM_gas"])
    tailrev = xp.cumsum((dM / rg)[::-1])[::-1]
    Phi_gas = -G * (Mg / rg + tailrev)
    g_gas = G * Mg / rg ** 2
    R = xp.sqrt(xp.sum(points ** 2, axis=1))
    Phi = xp.interp(xp.clip(R, rg[0], rg[-1]), rg, Phi_gas)
    gr = xp.interp(xp.clip(R, rg[0], rg[-1]), rg, g_gas)
    gv = -gr[:, None] * points / xp.maximum(R, 1e-30)[:, None]
    wx, wm = xp.asarray(c["pos"]), xp.asarray(c["Mg"])
    P = points.shape[0]
    for i0 in range(0, P, chunk):
        i1 = min(P, i0 + chunk)
        d = points[i0:i1, None, :] - wx[None, :, :]
        dist = xp.maximum(xp.sqrt(xp.sum(d * d, axis=-1)), soft)
        Phi[i0:i1] -= xp.sum(G * wm[None, :] / dist, axis=1)
        gv[i0:i1] -= xp.sum(G * wm[None, :, None] * d / dist[:, :, None] ** 3,
                            axis=1)
    return Phi, xp.sqrt(xp.sum(gv * gv, axis=1))


def points_of(c, xp=np):
    X, Y, Z = (to_xp(c[k], xp) for k in ("X", "Y", "Z"))
    return xp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def _fill_empty(v, cnt, xp):
    """Nearest-value fill for shells that contain no cell.

    Empty shells are unavoidable at small radius (bin width << cell size) and
    an unfilled zero there propagates straight into the boundary condition as
    k = 0 -> 1/sqrt(k) = inf -> a NaN field.  That was a real bug here, not a
    hypothetical one.
    """
    idx = xp.arange(v.shape[0]).astype(xp.float64)
    good = cnt > 0
    if bool(good.all()):
        return v
    return xp.interp(idx, idx[good], v[good])


def k_radial_profile(Kf, c, edges, xp=np):
    """<rhat^T K rhat> in spherical shells, plus <tr K/3> for reference."""
    X, Y, Z, R = (to_xp(c[k], xp) for k in ("X", "Y", "Z", "R"))
    Rs = xp.maximum(R, 1e-30)
    rhat = xp.stack([X / Rs, Y / Rs, Z / Rs], axis=-1)
    Km = xp.moveaxis(Kf, 0, -1)
    krr = W.sym3_quad(Km, rhat, xp)
    kav, cnt = F.shell_average(krr, R, edges, xp)
    kiso, _ = F.shell_average((Kf[0] + Kf[1] + Kf[2]) / 3.0, R, edges, xp)
    return _fill_empty(kav, cnt, xp), _fill_empty(kiso, cnt, xp), krr


def solve_with_K(c, Kf, mu, xp=np, x0=None, outer=60, tol_outer=3e-7,
                 tol_inner=1e-9, verbose=False):
    """Full nonlinear solve with the tensor field Kf on the cluster grid."""
    h = c["dx"]
    rho = to_xp(c["rho"], xp)
    R = to_xp(c["R"], xp)
    nb = max(48, 2 * c["n"])
    edges = xp.linspace(0.0, float(R.max()), nb + 1)
    kav, kiso, krr = k_radial_profile(Kf, c, edges, xp)
    rc = 0.5 * (edges[1:] + edges[:-1])
    r_prof = to_xp(c["r_prof"], xp)
    M_prof = to_xp(c["M_prof"], xp)
    kp = xp.interp(r_prof, rc, kav)
    X, Y, Z = (to_xp(c[k], xp) for k in ("X", "Y", "Z"))
    bc, r_ext, gp = F.dirichlet_shell(X, Y, Z, r_prof, M_prof, kp, mu, xp=xp)
    Psi, info = F.solve_field(rho, Kf, h, bc, mu, xp=xp, x0=x0, outer=outer,
                              tol_outer=tol_outer, tol_inner=tol_inner,
                              verbose=verbose)
    info["k_shell"] = kav
    info["k_iso"] = kiso
    info["edges"] = edges
    info["bc"] = bc
    return Psi, info


def consistent_pair(Psi, Kf, rho, h, bc, mu, xp=np, tol=1e-13, maxiter=12000):
    """(Psi, A) that satisfy the LINEAR system to CG tolerance.

    Flux conservation is a property of the discretisation, so it has to be
    measured on a Psi and an A that solve div(A grad Psi) = 4 pi G rho, not on
    a Psi solved with the previous outer iterate's A.  A is frozen from the
    converged Picard field, then one tight linear solve is done with it.
    """
    gx = F._centred(Psi, 0, h, xp)
    gy = F._centred(Psi, 1, h, xp)
    gz = F._centred(Psi, 2, h, xp)
    gvec = xp.stack([gx, gy, gz], axis=-1)
    Km = xp.moveaxis(Kf, 0, -1)
    X = xp.sqrt(xp.maximum(W.sym3_quad(Km, gvec, xp), 0.0)) / mu.a0
    A = tuple(mu(X, xp) * Kf[i] for i in range(6))
    Psi2, it, rel = F.solve_linear(rho, A, h, bc, tol=tol, maxiter=maxiter,
                                   xp=xp, x0=Psi)
    return Psi2, A, it, rel


def identity_K(shape, xp=np):
    one = xp.ones(shape)
    zer = xp.zeros(shape)
    return xp.stack([one, one, one, zer, zer, zer], axis=0)


def shell_boost(Psi_a, Psi_b, c, radii_kpc, xp=np, width=1.0):
    """<|g_a|>/<|g_b|> in shells at the requested radii."""
    h = c["dx"]
    R = to_xp(c["R"], xp)
    ga, _ = F.gradient_mag(Psi_a, h, xp)
    gb, _ = F.gradient_mag(Psi_b, h, xp)
    out = []
    for rk in radii_kpc:
        sel = xp.abs(R - rk * KPC) < width * h
        if int(sel.sum()) < 20:
            out.append(float("nan"))
            continue
        out.append(float(ga[sel].mean() / gb[sel].mean()))
    return out


def radial_boost_1d(r, Menc, k, mu, xp=np):
    """Boost |g(k)|/|g(k=1)| from the exact 1-D reduction."""
    g1 = mu.invert(G * Menc / r ** 2, xp.ones_like(r), xp)
    gk = mu.invert(G * Menc / r ** 2, k, xp)
    return gk / g1
