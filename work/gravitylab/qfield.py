"""Void-state fields Q1-Q4 for Run A, step 5.

    Q1  q_rho   = [1 + (rho_L/rho_c)^m]^-1
    Q2  q_g     = [1 + (|g_N|/a0)^n]^-1
    Q3  q_rho_g = [1 + (rho_L/rho_c)^m + (|g_N|/a0)^n]^-1
    Q4  (1 - L_q^2 grad^2) q = q_rho_g

Q4 is solved, not approximated. In spherical symmetry the screened-Poisson
operator is

    q - L_q^2 (1/r^2) d/dr ( r^2 dq/dr ) = q_src

which is tridiagonal on a radial grid. It is solved on a uniform log-r grid
that extends well beyond the data, with dq/dr = 0 at the centre and q = q_env
at the outer boundary, then interpolated back to the observed radii.

A NOTE ON rho_L IN ONE DIMENSION. The program defines rho_L as the baryonic
density smoothed with a Gaussian of scale L_rho, which needs a 3-D map. Run A
has only a rotation curve, so the density proxy used here is the MEAN ENCLOSED
baryonic density,

    rho_bar(<R) = 3 M_b(<R) / (4 pi R^3),

which is a genuine density with the right dimensions and the right monotonic
behaviour, but it is not the program's rho_L and carries no L_rho. Q1 and Q3
are therefore screened here in a degraded form; the smoothing scale that the
rank-2 argument says carries the real information cannot be exercised until
the PDE stage. This is stated rather than hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30


def rho_enclosed(gN, R_kpc):
    """Mean enclosed baryonic density from the Newtonian acceleration.

    M(<R) = g_N R^2 / G, so rho_bar = 3 g_N / (4 pi G R).
    """
    Rm = np.asarray(R_kpc, float) * KPC
    return 3.0 * gN / (4.0 * np.pi * G * Rm)


def q_rho(gN, R_kpc, rho_c=1e-24, m=1.0, **kw):
    return 1.0 / (1.0 + (rho_enclosed(gN, R_kpc) / rho_c) ** m)


def q_g(gN, a0=1.2e-10, n=1.0, **kw):
    return 1.0 / (1.0 + (gN / a0) ** n)


def q_rho_g(gN, R_kpc, rho_c=1e-24, m=1.0, a0=1.2e-10, n=1.0, **kw):
    return 1.0 / (1.0 + (rho_enclosed(gN, R_kpc) / rho_c) ** m
                  + (gN / a0) ** n)


def q_nonlocal(gN, R_kpc, L_q_kpc=10.0, q_env=1.0, rho_c=1e-24, m=1.0,
               a0=1.2e-10, n=1.0, ngrid=256, pad=8.0, **kw):
    """Solve (1 - L_q^2 grad^2) q = q_src on a radial grid.

    Returns q at the observed radii. The tridiagonal system is assembled on a
    uniform grid in ln r spanning [R_min/pad, R_max*pad].
    """
    R = np.asarray(R_kpc, float)
    lo, hi = np.log(R.min() / pad), np.log(R.max() * pad)
    x = np.linspace(lo, hi, ngrid)
    r = np.exp(x)
    h = x[1] - x[0]
    L2 = (L_q_kpc / 1.0) ** 2                      # kpc^2, r is in kpc

    # source term evaluated on the grid: extend g_N as Keplerian outside the
    # data and as solid-body inside it, which is what the profile does.
    gi = np.interp(r, R, gN)
    out = r > R.max()
    if out.any():
        gi[out] = gN[-1] * (R.max() / r[out]) ** 2
    inn = r < R.min()
    if inn.any():
        gi[inn] = gN[0] * (r[inn] / R.min())
    src = 1.0 / (1.0 + (rho_enclosed(gi, r) / rho_c) ** m + (gi / a0) ** n)

    # In log coordinates: (1/r^2) d/dr (r^2 dq/dr) = (1/r^2)(q_xx + q_x),
    # so the operator is q - (L2/r^2)(q_xx + q_x). Tridiagonal, solved banded
    # in O(ngrid) so it can sit inside an optimiser loop.
    c = L2 / r ** 2
    lower = np.zeros(ngrid)
    diag = np.zeros(ngrid)
    upper = np.zeros(ngrid)
    b = src.copy()
    diag[1:-1] = 1.0 + c[1:-1] * (2.0 / h ** 2)
    lower[1:-1] = -c[1:-1] * (1.0 / h ** 2 - 0.5 / h)
    upper[1:-1] = -c[1:-1] * (1.0 / h ** 2 + 0.5 / h)
    diag[0], upper[0], b[0] = 1.0, -1.0, 0.0      # dq/dr = 0 at the centre
    diag[-1], lower[-1], b[-1] = 1.0, 0.0, q_env  # q = q_env outside
    ab = np.zeros((3, ngrid))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    qg = solve_banded((1, 1), ab, b)
    return np.interp(R, r, qg)


QREGISTRY = {
    "Q1_rho": dict(fn=q_rho, params=("rho_c", "m")),
    "Q2_g": dict(fn=q_g, params=("a0", "n")),
    "Q3_rho_g": dict(fn=q_rho_g, params=("rho_c", "m", "a0", "n")),
    "Q4_nonlocal": dict(fn=q_nonlocal,
                        params=("rho_c", "m", "a0", "n", "L_q_kpc", "q_env")),
}
