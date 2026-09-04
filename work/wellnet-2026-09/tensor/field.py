"""Nonlinear field solver for  div[ mu(X) K grad Phi ] = 4 pi G rho_b,
X = sqrt( grad Phi . K . grad Phi ) / a0,  g = -grad Phi.

The discretisation is EXACTLY the one in work/gravitylab/solver.py -- same
face-flux stencil, same one-sided-at-the-edge centred derivatives, same open
Dirichlet shell.  That file is not modified; it is imported, and gate
`operator_match` checks this module's backend-generic operator against it to
round-off on random inputs.  What is added here:

  * a CuPy backend, because a 128^3 Picard solve is ~10^5 operator applies;
  * Jacobi-preconditioned CG (the diagonal of the FV operator is exactly
    -(sum of the six face conductivities)/h^2, since the cross terms use
    centred derivatives which carry no self-coefficient at interior cells);
  * the mu(X) outer iteration (lagged diffusivity / Picard with damping);
  * an exact 1-D radial reduction of the same equation, used both for the
    outer Dirichlet values and as the fast surrogate for the parameter map.

THE BOUNDARY CONDITION.  For an isolated source the far field is not zero and
not Newtonian: at 1.4 Mpc an A2029-like cluster sits at 0.08 a0, deep in the
MOND regime, where Phi grows logarithmically.  A sealed box is
self-contradictory (it cost the original solver three gates).  Here the shell
values come from the spherical reduction of the SAME equation,

    mu( sqrt(k) |Phi'| / a0 ) k |Phi'| = G M(<r) / r^2 ,

with k(r) = < rhat^T K rhat > taken from the model's own tensor field.  For a
statistically spherical configuration this reduction is exact, because a
spherically symmetric well distribution forces K = a(r) I + b(r)(rhat rhat^T
- I/3), whose radial eigenvalue is exactly the k above.  Only an additive
constant in Phi is arbitrary, and constants are annihilated by the operator.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from wellnet import (G, A0, KPC, MSUN, get_xp, asnumpy, sym3_quad,
                     sym3_eigvals)

GRAVITYLAB = Path(__file__).resolve().parents[2] / "gravitylab"
if str(GRAVITYLAB) not in sys.path:
    sys.path.insert(0, str(GRAVITYLAB))

C_LIGHT = 2.99792458e8


# ------------------------------------------------------------------- grid
def grids(n, L, xp=np):
    h = L / n
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X, Y, Z = xp.meshgrid(ax, ax, ax, indexing="ij")
    return h, ax, X, Y, Z


def _centred(f, ax, h, xp):
    return xp.gradient(f, h, axis=ax)


def apply_operator(Psi, A, h, xp=np):
    """div(A grad Psi); identical stencil to gravitylab.solver.apply_operator."""
    Axx, Ayy, Azz, Axy, Axz, Ayz = A
    gx = _centred(Psi, 0, h, xp)
    gy = _centred(Psi, 1, h, xp)
    gz = _centred(Psi, 2, h, xp)

    dfx = (Psi[1:, :, :] - Psi[:-1, :, :]) / h
    Fx = (0.5 * (Axx[1:] + Axx[:-1]) * dfx
          + 0.5 * (Axy[1:] + Axy[:-1]) * 0.5 * (gy[1:] + gy[:-1])
          + 0.5 * (Axz[1:] + Axz[:-1]) * 0.5 * (gz[1:] + gz[:-1]))
    dfy = (Psi[:, 1:, :] - Psi[:, :-1, :]) / h
    Fy = (0.5 * (Ayy[:, 1:] + Ayy[:, :-1]) * dfy
          + 0.5 * (Axy[:, 1:] + Axy[:, :-1]) * 0.5 * (gx[:, 1:] + gx[:, :-1])
          + 0.5 * (Ayz[:, 1:] + Ayz[:, :-1]) * 0.5 * (gz[:, 1:] + gz[:, :-1]))
    dfz = (Psi[:, :, 1:] - Psi[:, :, :-1]) / h
    Fz = (0.5 * (Azz[:, :, 1:] + Azz[:, :, :-1]) * dfz
          + 0.5 * (Axz[:, :, 1:] + Axz[:, :, :-1]) * 0.5 * (gx[:, :, 1:] + gx[:, :, :-1])
          + 0.5 * (Ayz[:, :, 1:] + Ayz[:, :, :-1]) * 0.5 * (gy[:, :, 1:] + gy[:, :, :-1]))

    out = xp.zeros_like(Psi)
    out[1:-1, 1:-1, 1:-1] = (
        (Fx[1:, 1:-1, 1:-1] - Fx[:-1, 1:-1, 1:-1]) / h
        + (Fy[1:-1, 1:, 1:-1] - Fy[1:-1, :-1, 1:-1]) / h
        + (Fz[1:-1, 1:-1, 1:] - Fz[1:-1, 1:-1, :-1]) / h)
    return out


def flux_faces(Psi, A, h, xp=np):
    """The three face-flux arrays the FV operator conserves exactly."""
    Axx, Ayy, Azz, Axy, Axz, Ayz = A
    gx = _centred(Psi, 0, h, xp)
    gy = _centred(Psi, 1, h, xp)
    gz = _centred(Psi, 2, h, xp)
    Fx = (0.5 * (Axx[1:] + Axx[:-1]) * (Psi[1:] - Psi[:-1]) / h
          + 0.5 * (Axy[1:] + Axy[:-1]) * 0.5 * (gy[1:] + gy[:-1])
          + 0.5 * (Axz[1:] + Axz[:-1]) * 0.5 * (gz[1:] + gz[:-1]))
    Fy = (0.5 * (Ayy[:, 1:] + Ayy[:, :-1]) * (Psi[:, 1:] - Psi[:, :-1]) / h
          + 0.5 * (Axy[:, 1:] + Axy[:, :-1]) * 0.5 * (gx[:, 1:] + gx[:, :-1])
          + 0.5 * (Ayz[:, 1:] + Ayz[:, :-1]) * 0.5 * (gz[:, 1:] + gz[:, :-1]))
    Fz = (0.5 * (Azz[:, :, 1:] + Azz[:, :, :-1]) * (Psi[:, :, 1:] - Psi[:, :, :-1]) / h
          + 0.5 * (Axz[:, :, 1:] + Axz[:, :, :-1]) * 0.5 * (gx[:, :, 1:] + gx[:, :, :-1])
          + 0.5 * (Ayz[:, :, 1:] + Ayz[:, :, :-1]) * 0.5 * (gy[:, :, 1:] + gy[:, :, :-1]))
    return Fx, Fy, Fz


def _diagonal(A, h, xp):
    """Diagonal of the FV operator: -(sum of the six face conductivities)/h^2."""
    Axx, Ayy, Azz = A[0], A[1], A[2]
    d = xp.zeros_like(Axx)
    fx = 0.5 * (Axx[1:] + Axx[:-1])
    fy = 0.5 * (Ayy[:, 1:] + Ayy[:, :-1])
    fz = 0.5 * (Azz[:, :, 1:] + Azz[:, :, :-1])
    d[1:-1, 1:-1, 1:-1] = -(
        fx[1:, 1:-1, 1:-1] + fx[:-1, 1:-1, 1:-1]
        + fy[1:-1, 1:, 1:-1] + fy[1:-1, :-1, 1:-1]
        + fz[1:-1, 1:-1, 1:] + fz[1:-1, 1:-1, :-1]) / h ** 2
    return d


def solve_linear(rho, A, h, Psi_bc, tol=1e-10, maxiter=4000, xp=np, x0=None,
                 precond=True):
    """Jacobi-PCG for div(A grad Psi) = 4 pi G rho, Dirichlet outer shell."""
    mask = xp.zeros(rho.shape, bool)
    mask[1:-1, 1:-1, 1:-1] = True
    x = xp.where(mask, 0.0 if x0 is None else x0, Psi_bc)
    b = 4.0 * math.pi * G * rho
    if precond:
        d = _diagonal(A, h, xp)
        Minv = xp.where(mask, 1.0 / xp.where(d != 0, d, 1.0), 0.0)
    else:
        Minv = mask.astype(xp.float64)
    r = (b - apply_operator(x, A, h, xp)) * mask
    z = Minv * r
    p = z.copy()
    rz = float(xp.sum(r * z))
    bn = float(xp.sqrt(xp.sum((b * mask) ** 2))) or 1.0
    rel = float(xp.sqrt(xp.sum(r * r))) / bn
    it = 0
    for it in range(1, maxiter + 1):
        if rel < tol:
            break
        Ap = apply_operator(p * mask, A, h, xp) * mask
        den = float(xp.sum(p * Ap))
        if den == 0.0:
            break
        alpha = rz / den
        x = x + alpha * p * mask
        r = r - alpha * Ap
        rel = float(xp.sqrt(xp.sum(r * r))) / bn
        z = Minv * r
        rz_new = float(xp.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it, rel


# --------------------------------------------------------------- mu laws
class Mu:
    """mu(X) and the pointwise inverse used by the spherical reduction.

    `floor` keeps the discrete operator non-singular.  mu(0) = 0 exactly, and
    grad Phi vanishes at the centre of a symmetric source by symmetry, so
    without a floor the operator has an all-zero row there and CG stalls.  The
    floor is active only where |g| < floor*a0/(1-floor) ~ 1e-4 a0, i.e. inside
    one cell of the centre; the gates measure that it changes nothing.
    """

    def __init__(self, kind="simple", a0=A0, floor=1e-4):
        self.kind, self.a0, self.floor = kind, a0, floor

    def __call__(self, X, xp=np):
        if self.kind == "one":
            return xp.ones_like(X)
        if self.kind == "simple":
            m = X / (1.0 + X)
        elif self.kind == "standard":
            m = X / xp.sqrt(1.0 + X * X)
        else:
            raise ValueError(self.kind)
        return xp.maximum(m, self.floor)

    def invert(self, F, k, xp=np):
        """|Phi'| solving mu(sqrt(k)|Phi'|/a0) k |Phi'| = F."""
        a0 = self.a0
        if self.kind == "one":
            return F / k
        beta = F / (xp.sqrt(k) * a0)
        if self.kind == "simple":            # X^2 - b X - b = 0
            X = 0.5 * (beta + xp.sqrt(beta * beta + 4.0 * beta))
        elif self.kind == "standard":        # u^2 - b^2 u - b^2 = 0, u = X^2
            b2 = beta * beta
            u = 0.5 * (b2 + xp.sqrt(b2 * b2 + 4.0 * b2))
            X = xp.sqrt(u)
        else:
            raise ValueError(self.kind)
        return a0 * X / xp.sqrt(k)


# ------------------------------------------------- 1-D spherical reduction
def radial_solution(r, Menc, k, mu, r_ref=None, xp=np):
    """|Phi'|(r) and Phi(r) for the spherical reduction of the field equation.

    r ascending (m), Menc(<r) (kg), k(r) the radial eigenvalue of K.
    Phi is defined up to a constant; it is set to zero at r_ref (default r[0]).
    """
    F = G * Menc / r ** 2
    gp = mu.invert(F, k, xp)
    Phi = xp.concatenate([xp.zeros(1), xp.cumsum(
        0.5 * (gp[1:] + gp[:-1]) * (r[1:] - r[:-1]))])
    if r_ref is not None:
        Phi = Phi - xp.interp(xp.asarray(r_ref), r, Phi)
    return gp, Phi


def dirichlet_shell(X, Y, Z, r_prof, M_prof, k_prof, mu, xp=np, pad=60.0):
    """Dirichlet values on the whole grid from the spherical reduction.

    The profile is continued outside r_prof[-1] with constant enclosed mass
    and constant k, out to pad * r_prof[-1], so the shell radii (which reach
    sqrt(3)/2 of the box) are always covered.
    """
    tail = xp.asarray(np.geomspace(1.02, pad, 160))
    r_ext = xp.concatenate([r_prof, r_prof[-1] * tail])
    M_ext = xp.concatenate([M_prof, xp.full(160, M_prof[-1])])
    k_ext = xp.concatenate([k_prof, xp.full(160, k_prof[-1])])
    gp, Phi = radial_solution(r_ext, M_ext, k_ext, mu, xp=xp)
    R = xp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    return xp.interp(xp.maximum(R, r_ext[0]), r_ext, Phi), r_ext, gp


# --------------------------------------------------- the nonlinear solver
def solve_field(rho, Kf, h, Psi_bc, mu, xp=np, x0=None, outer=40, omega=0.75,
                tol_outer=1e-6, tol_inner=1e-9, maxiter=4000, verbose=False):
    """Picard/lagged-diffusivity iteration on A = mu(X) K.

    Kf is (6, nx, ny, nz).  Returns Psi, diagnostics.
    """
    Psi = Psi_bc.copy() if x0 is None else x0.copy()
    mask = xp.zeros(rho.shape, bool)
    mask[1:-1, 1:-1, 1:-1] = True
    Psi = xp.where(mask, Psi, Psi_bc)
    hist = []
    for k in range(outer):
        gx = _centred(Psi, 0, h, xp)
        gy = _centred(Psi, 1, h, xp)
        gz = _centred(Psi, 2, h, xp)
        gvec = xp.stack([gx, gy, gz], axis=-1)
        Km = xp.moveaxis(Kf, 0, -1)
        X = xp.sqrt(xp.maximum(sym3_quad(Km, gvec, xp), 0.0)) / mu.a0
        m = mu(X, xp)
        A = tuple(m * Kf[i] for i in range(6))
        Psi_new, it, rel = solve_linear(rho, A, h, Psi_bc, tol=tol_inner,
                                        maxiter=maxiter, xp=xp, x0=Psi)
        dn = float(xp.max(xp.abs(Psi_new - Psi)[mask]))
        sc = float(xp.max(xp.abs(Psi_new)[mask])) or 1.0
        Psi = Psi + omega * (Psi_new - Psi)
        hist.append(dn / sc)
        if verbose:
            print(f"      outer {k:3d}  dPhi/Phi = {dn/sc:.3e}  "
                  f"CG {it:5d} resid {rel:.2e}")
        if dn / sc < tol_outer:
            break
    gx = _centred(Psi, 0, h, xp)
    gy = _centred(Psi, 1, h, xp)
    gz = _centred(Psi, 2, h, xp)
    gvec = xp.stack([gx, gy, gz], axis=-1)
    Km = xp.moveaxis(Kf, 0, -1)
    X = xp.sqrt(xp.maximum(sym3_quad(Km, gvec, xp), 0.0)) / mu.a0
    A = tuple(mu(X, xp) * Kf[i] for i in range(6))
    return Psi, dict(outer=k + 1, hist=hist, A=A, mu=mu(X, xp))


# ------------------------------------------------------------ observables
def gradient_mag(Psi, h, xp=np):
    gx = _centred(Psi, 0, h, xp)
    gy = _centred(Psi, 1, h, xp)
    gz = _centred(Psi, 2, h, xp)
    return xp.sqrt(gx * gx + gy * gy + gz * gz), (gx, gy, gz)


def shell_average(F, R, edges, xp=np):
    """Mean of F in spherical shells."""
    idx = xp.clip(xp.searchsorted(edges, R.ravel()) - 1, 0, len(edges) - 2)
    n = len(edges) - 1
    tot = xp.bincount(idx, weights=F.ravel(), minlength=n)
    cnt = xp.bincount(idx, minlength=n).astype(xp.float64)
    return tot / xp.maximum(cnt, 1.0), cnt


def projected_deflection(gx, gy, h, zmask, xp=np):
    """|integral of g_perp dz| on the (x,y) plane, up to the 2/c^2 factor."""
    ax = gx[:, :, zmask].sum(axis=2) * h
    ay = gy[:, :, zmask].sum(axis=2) * h
    return xp.sqrt(ax * ax + ay * ay)
