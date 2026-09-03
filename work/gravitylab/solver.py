"""Finite-volume solver for the tensor field equation.

    div( A grad Psi ) = 4 pi G rho,      A = mu(X) K

A is symmetric positive definite, so the discrete operator is symmetric and
conjugate gradients apply. The program writes the theory as a conserved flux
equation, so a finite-volume discretisation is the right choice: flux leaving
a face is exactly the flux entering its neighbour, which makes Gauss's law
hold by construction rather than by accident.

BOUNDARY CONDITIONS, and why they are not optional. A first version used
zero-flux (Neumann) edges. That is self-contradictory for an isolated source:
Gauss's law over any surface enclosing all the mass demands a flux of
4 pi G M, but a sealed box forces the total to zero, so the solver silently
manufactures a compensating uniform background. It cost three of the six
mandatory gates -- the analytic comparison stalled at 1.8e-2 with convergence
order -0.13, eps_flux reached 0.32, and the box-size test moved 5.6%.

The fix is open boundaries: Dirichlet values on the outer shell taken from the
known far field. For constant K the exact monopole is

    Psi(r) = -G M / ( sqrt(det K) sqrt(r^T K^-1 r) )

which is what `far_field` returns. Interior cells are then solved with the
boundary held fixed, so flux may leave the domain and Gauss's law is honest.
"""
from __future__ import annotations

import numpy as np

G = 6.674e-11


# --------------------------------------------------------------- geometry
def grids(n, L):
    h = L / n
    ax = (np.arange(n) - n / 2 + 0.5) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    return h, ax, X, Y, Z


def far_field(X, Y, Z, Mtot, Kmat):
    """Exact constant-K monopole potential."""
    Kinv = np.linalg.inv(Kmat)
    q = (Kinv[0, 0] * X ** 2 + Kinv[1, 1] * Y ** 2 + Kinv[2, 2] * Z ** 2
         + 2 * Kinv[0, 1] * X * Y + 2 * Kinv[0, 2] * X * Z
         + 2 * Kinv[1, 2] * Y * Z)
    q = np.maximum(q, 1e-30)
    return -G * Mtot / (np.sqrt(np.linalg.det(Kmat)) * np.sqrt(q))


# --------------------------------------------------------------- operator
def _centred(f, ax, h):
    """Centred derivative, one-sided at the array edges (no wraparound)."""
    return np.gradient(f, h, axis=ax)


def apply_operator(Psi, A, h):
    """div(A grad Psi) on the interior; boundary shell returns 0.

    Face fluxes are computed with explicit forward differences, never np.roll,
    so nothing wraps around the box.
    """
    Axx, Ayy, Azz, Axy, Axz, Ayz = A
    gx = _centred(Psi, 0, h)
    gy = _centred(Psi, 1, h)
    gz = _centred(Psi, 2, h)

    # x-faces, between i and i+1
    dfx = (Psi[1:, :, :] - Psi[:-1, :, :]) / h
    axx = 0.5 * (Axx[1:, :, :] + Axx[:-1, :, :])
    axy = 0.5 * (Axy[1:, :, :] + Axy[:-1, :, :])
    axz = 0.5 * (Axz[1:, :, :] + Axz[:-1, :, :])
    Fx = axx * dfx + axy * 0.5 * (gy[1:, :, :] + gy[:-1, :, :]) \
        + axz * 0.5 * (gz[1:, :, :] + gz[:-1, :, :])

    dfy = (Psi[:, 1:, :] - Psi[:, :-1, :]) / h
    ayy = 0.5 * (Ayy[:, 1:, :] + Ayy[:, :-1, :])
    axy2 = 0.5 * (Axy[:, 1:, :] + Axy[:, :-1, :])
    ayz = 0.5 * (Ayz[:, 1:, :] + Ayz[:, :-1, :])
    Fy = ayy * dfy + axy2 * 0.5 * (gx[:, 1:, :] + gx[:, :-1, :]) \
        + ayz * 0.5 * (gz[:, 1:, :] + gz[:, :-1, :])

    dfz = (Psi[:, :, 1:] - Psi[:, :, :-1]) / h
    azz = 0.5 * (Azz[:, :, 1:] + Azz[:, :, :-1])
    axz2 = 0.5 * (Axz[:, :, 1:] + Axz[:, :, :-1])
    ayz2 = 0.5 * (Ayz[:, :, 1:] + Ayz[:, :, :-1])
    Fz = azz * dfz + axz2 * 0.5 * (gx[:, :, 1:] + gx[:, :, :-1]) \
        + ayz2 * 0.5 * (gy[:, :, 1:] + gy[:, :, :-1])

    out = np.zeros_like(Psi)
    out[1:-1, 1:-1, 1:-1] = (
        (Fx[1:, 1:-1, 1:-1] - Fx[:-1, 1:-1, 1:-1]) / h
        + (Fy[1:-1, 1:, 1:-1] - Fy[1:-1, :-1, 1:-1]) / h
        + (Fz[1:-1, 1:-1, 1:] - Fz[1:-1, 1:-1, :-1]) / h
    )
    return out


def flux_faces(Psi, A, h):
    """The three face-flux arrays the finite-volume operator actually uses.

    Fx has shape (n-1, n, n) and lives on the faces between cells i and i+1,
    and so on. These are the quantities the discretisation conserves exactly;
    re-deriving a flux from a centre-differenced gradient is a DIFFERENT
    quantity that agrees only to O(h^2), which is why the first version of the
    flux gate reported eps ~ 1e-2 on a solver that conserves to round-off.
    """
    Axx, Ayy, Azz, Axy, Axz, Ayz = A
    gx = _centred(Psi, 0, h)
    gy = _centred(Psi, 1, h)
    gz = _centred(Psi, 2, h)
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


# ----------------------------------------------------------------- solve
def solve(rho, A, h, Psi_bc, tol=1e-11, maxiter=6000):
    """CG for div(A grad Psi) = 4 pi G rho with Dirichlet outer shell.

    Psi_bc supplies the boundary values; only interior cells are unknowns, so
    the operator has no null space and flux may leave the domain.
    """
    mask = np.zeros(rho.shape, bool)
    mask[1:-1, 1:-1, 1:-1] = True
    x = np.where(mask, 0.0, Psi_bc)
    b = 4.0 * np.pi * G * rho
    r = (b - apply_operator(x, A, h)) * mask
    p = r.copy()
    rs = float(np.sum(r * r))
    bn = float(np.sqrt(np.sum((b * mask) ** 2))) or 1.0
    rel = float(np.sqrt(rs)) / bn
    for it in range(maxiter):
        if rel < tol:
            return x, it, rel
        Ap = apply_operator(p * mask, A, h) * mask
        den = float(np.sum(p * Ap))
        if den == 0.0:
            break
        alpha = rs / den
        x = x + alpha * p * mask
        r = r - alpha * Ap
        rs_new = float(np.sum(r * r))
        rel = float(np.sqrt(rs_new)) / bn
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, maxiter, rel


# ---------------------------------------------------------------- tensors
def isotropic(shape, value=1.0):
    o = np.full(shape, float(value))
    z = np.zeros(shape)
    return o, o.copy(), o.copy(), z, z.copy(), z.copy()


def axis_tensor(shape, e_hat, k_par, k_perp):
    """A = k_par e e^T + k_perp (I - e e^T) for a fixed unit axis e."""
    e = np.asarray(e_hat, float)
    e = e / np.linalg.norm(e)
    M = k_perp * np.eye(3) + (k_par - k_perp) * np.outer(e, e)
    comps = [np.full(shape, M[i, j]) for i, j in
             ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
    return tuple(comps), M
