"""Axisymmetric finite-volume solver and disk baryon models, for Run B.

    (1/R) d/dR [ R (A_RR dPsi/dR + A_Rz dPsi/dz) ]
        + d/dz [ A_zR dPsi/dR + A_zz dPsi/dz ]  =  4 pi G rho

Cell-centred on a uniform (R, z) grid, with the cylindrical face areas carried
explicitly so Gauss's law holds by construction:

    R-face area  =  2 pi R_(i+1/2) dz
    z-face area  =  pi ( R_(i+1/2)^2 - R_(i-1/2)^2 )
    cell volume  =  pi ( R_(i+1/2)^2 - R_(i-1/2)^2 ) dz

Boundaries: reflecting at R = 0 and at the midplane by symmetry; Dirichlet on
the outer shell from the monopole, which is why the box has to extend well past
the disk. The 3-D Cartesian solver in solver.py passes all seven gates of the
program's section 6; this one is validated against it and against the exact
Freeman disk, since a 2-D solve is the only way a 24-model tournament finishes.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded  # noqa: F401  (kept for parity)
from scipy.special import i0, i1, k0, k1

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30


# --------------------------------------------------------------- baryon model
def exponential_disk(R_kpc, z_kpc, Sigma0, Rd_kpc, hz_kpc):
    """Exponential radially, sech^2 vertically. Returns rho in kg/m^3."""
    R = np.asarray(R_kpc, float)[:, None]
    z = np.asarray(z_kpc, float)[None, :]
    sig = Sigma0 * np.exp(-R / Rd_kpc)                    # kg/m^2
    return sig / (2 * hz_kpc * KPC) / np.cosh(z / hz_kpc) ** 2


def freeman_vc(R_kpc, Sigma0, Rd_kpc):
    """Exact circular speed of a razor-thin exponential disk (Freeman 1970).

    V^2 = 4 pi G Sigma0 Rd y^2 [ I0(y)K0(y) - I1(y)K1(y) ],  y = R/(2Rd)
    Used only to validate the solver; the production model has finite thickness.
    """
    y = np.asarray(R_kpc, float) / (2.0 * Rd_kpc)
    y = np.maximum(y, 1e-8)
    br = i0(y) * k0(y) - i1(y) * k1(y)
    return np.sqrt(np.maximum(4 * np.pi * G * Sigma0 * (Rd_kpc * KPC) * y ** 2 * br, 0))


# ------------------------------------------------------------------- geometry
class Grid:
    def __init__(self, nR, nz, Rmax_kpc, zmax_kpc):
        self.nR, self.nz = nR, nz
        self.dR = Rmax_kpc / nR * KPC
        self.dz = zmax_kpc / nz * KPC
        self.Rc = (np.arange(nR) + 0.5) * self.dR          # cell centres
        self.Rf = (np.arange(nR + 1)) * self.dR            # faces, R_(i-1/2)
        self.zc = (np.arange(nz) + 0.5) * self.dz          # z >= 0, midplane sym
        self.AR = 2 * np.pi * self.Rf[:, None] * self.dz   # R-face areas
        self.Az = (np.pi * (self.Rf[1:] ** 2 - self.Rf[:-1] ** 2))[:, None]
        self.V = self.Az * self.dz                          # cell volumes


def apply_axi(Psi, A, g: Grid):
    """div(A grad Psi) integrated over each cell, divided by cell volume."""
    ARR, Azz, ARz = A
    dPdR = np.zeros((g.nR + 1, g.nz))
    dPdR[1:-1] = (Psi[1:] - Psi[:-1]) / g.dR               # interior R-faces
    dPdz = np.zeros((g.nR, g.nz + 1))
    dPdz[:, 1:-1] = (Psi[:, 1:] - Psi[:, :-1]) / g.dz
    dPdz[:, 0] = 0.0                                       # midplane symmetry

    # transverse derivatives averaged onto the faces
    gz_c = np.zeros_like(Psi)
    gz_c[:, 1:-1] = (Psi[:, 2:] - Psi[:, :-2]) / (2 * g.dz)
    gR_c = np.zeros_like(Psi)
    gR_c[1:-1] = (Psi[2:] - Psi[:-2]) / (2 * g.dR)

    aRR = np.zeros((g.nR + 1, g.nz))
    aRR[1:-1] = 0.5 * (ARR[1:] + ARR[:-1])
    aRz_R = np.zeros((g.nR + 1, g.nz))
    aRz_R[1:-1] = 0.5 * (ARz[1:] + ARz[:-1])
    gz_R = np.zeros((g.nR + 1, g.nz))
    gz_R[1:-1] = 0.5 * (gz_c[1:] + gz_c[:-1])

    azz = np.zeros((g.nR, g.nz + 1))
    azz[:, 1:-1] = 0.5 * (Azz[:, 1:] + Azz[:, :-1])
    aRz_z = np.zeros((g.nR, g.nz + 1))
    aRz_z[:, 1:-1] = 0.5 * (ARz[:, 1:] + ARz[:, :-1])
    gR_z = np.zeros((g.nR, g.nz + 1))
    gR_z[:, 1:-1] = 0.5 * (gR_c[:, 1:] + gR_c[:, :-1])

    FR = (aRR * dPdR + aRz_R * gz_R) * g.AR
    Fz = (azz * dPdz + aRz_z * gR_z) * g.Az
    div = (FR[1:] - FR[:-1] + Fz[:, 1:] - Fz[:, :-1]) / g.V
    return div


def solve_axi(rho, A, g: Grid, Psi_bc, tol=1e-11, maxiter=4000):
    """CG with the outer shell held at Psi_bc."""
    mask = np.ones(rho.shape, bool)
    mask[-1, :] = False
    mask[:, -1] = False
    x = np.where(mask, 0.0, Psi_bc)
    b = 4 * np.pi * G * rho
    r = (b - apply_axi(x, A, g)) * mask
    p = r.copy()
    rs = float(np.sum(r * r * g.V))
    bn = float(np.sqrt(np.sum((b * mask) ** 2 * g.V))) or 1.0
    rel = float(np.sqrt(rs)) / bn
    for it in range(maxiter):
        if rel < tol:
            return x, it, rel
        Ap = apply_axi(p * mask, A, g) * mask
        den = float(np.sum(p * Ap * g.V))
        if den == 0.0:
            break
        alpha = rs / den
        x = x + alpha * p * mask
        r = r - alpha * Ap
        rs_new = float(np.sum(r * r * g.V))
        rel = float(np.sqrt(rs_new)) / bn
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, maxiter, rel


def monopole_bc(g: Grid, Mtot):
    R = g.Rc[:, None] * np.ones((1, g.nz))
    z = np.ones((g.nR, 1)) * g.zc[None, :]
    return -G * Mtot / np.sqrt(R ** 2 + z ** 2 + (0.05 * g.dR) ** 2)


def isotropic_A(shape):
    o = np.ones(shape)
    return o, o.copy(), np.zeros(shape)


def disk_axis_A(shape, k_par, k_perp):
    """K2 of the program: the disk normal is z, so K = diag(k_perp, k_par).

    k_par multiplies the z direction (along the disk normal) and k_perp the
    radial direction. Off-diagonal is zero because the axes align with the grid.
    """
    return (np.full(shape, k_perp), np.full(shape, k_par), np.zeros(shape))


def midplane_vc(Psi, g: Grid):
    """Circular speed in the midplane, V^2 = R dPsi/dR at z = 0."""
    dPdR = np.zeros(g.nR)
    dPdR[1:-1] = (Psi[2:, 0] - Psi[:-2, 0]) / (2 * g.dR)
    dPdR[0] = (Psi[1, 0] - Psi[0, 0]) / g.dR
    dPdR[-1] = (Psi[-1, 0] - Psi[-2, 0]) / g.dR
    return np.sqrt(np.maximum(g.Rc * dPdR, 0.0))


def vertical_Kz(Psi, g: Grid, iz=1):
    """K_z = dPsi/dz, evaluated a little above the midplane."""
    return (Psi[:, iz + 1] - Psi[:, iz - 1]) / (2 * g.dz)
