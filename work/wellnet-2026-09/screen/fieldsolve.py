"""Solver wrappers for the screen.

WHAT THIS ADDS TO ``work/gravitylab/solver.py`` (which is imported, never
modified):

  1. ``solve_K``            -- driver for a general position-dependent K field,
                               with the Dirichlet shell taken from the exact
                               constant-K monopole evaluated with the SHELL-MEAN
                               K, plus a reported measure of how much K actually
                               varies over that shell (so the boundary error is
                               quantified rather than assumed away).
  2. ``solve_aqual``        -- damped Picard iteration for the nonlinear AQUAL
                               operator, with a Dirichlet shell from the exact
                               spherical AQUAL solution.
  3. ``solve_qumond``       -- two linear solves; the effective source is built
                               with ``solver.apply_operator`` itself so the
                               divergence is the one the discretisation
                               conserves, not a re-derived centre-difference.
  4. ``net_force`` / ``force_at`` / ``vcirc_axis`` -- the diagnostics the
                               reciprocity, midpoint and rotation-curve screens
                               need.
  5. ``spherical_g``        -- the EXACT spherically symmetric reduction, used
                               for the asymptotic screens, where a box large
                               enough to see r -> infinity does not exist.

Nothing here changes solver.py's discretisation; the operator, the flux faces
and the CG loop are all the originals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_GL = Path(__file__).resolve().parents[2] / "gravitylab"
if str(_GL) not in sys.path:
    sys.path.insert(0, str(_GL))

import solver as S                                    # noqa: E402
import families as F                                  # noqa: E402

G = S.G
KPC = F.KPC
MSUN = F.MSUN


# ----------------------------------------------------------------- boundaries
def _shell_mask(shape):
    m = np.zeros(shape, bool)
    m[0, :, :] = m[-1, :, :] = True
    m[:, 0, :] = m[:, -1, :] = True
    m[:, :, 0] = m[:, :, -1] = True
    return m


def shell_K(K, shape):
    """Mean K over the outer shell, and the relative spread over that shell."""
    msk = _shell_mask(shape).ravel()
    Ks = K[msk]
    Km = Ks.mean(0)
    Km = 0.5 * (Km + Km.T)
    spread = float(np.abs(Ks - Km[None]).max() / max(np.abs(Km).max(), 1e-300))
    return Km, spread


def radial_far_field(K, box, Mtot, eps=1e-3):
    """Dirichlet shell for a K field that becomes RADIALLY ALIGNED at infinity.

    Families C and E do not tend to a constant K far from the source: every
    n_a tends to the same -x_hat, so S tends to (x_hat x_hat - I/3) and K keeps
    a fixed anisotropy locked to the radius direction.  For such a medium the
    exact spherical exterior solution is

        Psi(r) = -G M / ( k_r r ),    k_r(x) = x_hat . K(x) . x_hat,

    because only the radial eigenvalue enters the radial flux.  Using the
    constant-K monopole instead leaves an O(s_T) error on the whole shell.
    """
    r = np.maximum(box.r, eps * box.h)
    nx = box.X / r
    ny = box.Y / r
    nz = box.Z / r
    n = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], 1)
    kr = np.einsum("pi,pij,pj->p", n, K, n).reshape(box.shape)
    return -G * Mtot / (kr * r), kr


# --------------------------------------------------------------- linear K solve
def solve_K(rho, K, box, tol=1e-11, maxiter=8000, Mtot=None, bc_mode="radial"):
    """div[K grad Psi] = 4 pi G rho with an open (Dirichlet) shell."""
    A = F.pack_A(K)
    Km, spread = shell_K(K, box.shape)
    M = float(rho.sum() * box.vol) if Mtot is None else Mtot
    if bc_mode == "radial":
        bc, kr = radial_far_field(K, box, M)
    else:
        bc = S.far_field(box.X, box.Y, box.Z, M, Km)
    Psi, it, rel = S.solve(rho, A, box.h, bc, tol=tol, maxiter=maxiter)
    return dict(Psi=Psi, iters=it, resid=rel, shell_spread=spread, K_shell=Km)


def solve_newton(rho, box, tol=1e-11, maxiter=8000, Mtot=None):
    A = F.iso_A(box.shape, 1.0)
    M = float(rho.sum() * box.vol) if Mtot is None else Mtot
    bc = S.far_field(box.X, box.Y, box.Z, M, np.eye(3))
    Psi, it, rel = S.solve(rho, A, box.h, bc, tol=tol, maxiter=maxiter)
    return dict(Psi=Psi, iters=it, resid=rel, shell_spread=0.0,
                K_shell=np.eye(3))


# ------------------------------------------------------------------- MOND
def _spherical_mond_bc(box, Mtot, a0, form="simple", nrad=4000, far=60.0):
    """Dirichlet shell from the exact spherical MOND solution.

    In spherical symmetry AQUAL integrates exactly to mu(g/a0) g = GM/r^2, and
    QUMOND to g = nu(g_N/a0) g_N; the two agree in spherical symmetry.  Psi is
    obtained by integrating g inward from far * L with Psi(far * L) = 0, so the
    shell carries the log divergence correctly and only an irrelevant additive
    constant is left free.
    """
    Rmax = far * box.L
    r = np.geomspace(box.h * 0.25, Rmax, nrad)
    gN = G * Mtot / r ** 2
    if form == "rar":
        g = gN / (1.0 - np.exp(-np.sqrt(gN / a0)))
    else:
        g = 0.5 * (gN + np.sqrt(gN * gN + 4.0 * gN * a0))
    # Psi(r) = -int_r^Rmax g dr'
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(r))])
    Psi_r = -(integ[-1] - integ)
    rr = np.maximum(box.r, r[0])
    return np.interp(rr, r, Psi_r)


def solve_aqual(rho, box, a0=F.A0, form="simple", Mtot=None, iters=40,
                damp=0.7, tol=1e-11, picard_tol=1e-8):
    """Damped Picard for div[mu(|grad Psi|/a0) grad Psi] = 4 pi G rho.

    The inner linear solve is given a TOLERANCE SCHEDULE rather than the final
    tolerance from the start: while the coefficient mu(|grad Psi|) is still
    moving there is no point converging the linear problem to 1e-11, so the
    inner tolerance is tightened geometrically and only the last few Picard
    steps pay full price.  This is the standard inexact-Newton argument and it
    changes nothing about the answer -- the loop still exits on the OUTER
    change falling below picard_tol with the inner solve at its final
    tolerance.
    """
    M = float(rho.sum() * box.vol) if Mtot is None else Mtot
    bc = _spherical_mond_bc(box, M, a0, form)
    mu = F.mu_simple if form == "simple" else F.mu_standard
    Psi = bc.copy()
    hist, inner = [], []
    zero = np.zeros(box.shape)
    for k in range(iters):
        gx, gy, gz = np.gradient(Psi, box.h, edge_order=2)
        gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        X = np.maximum(gmag, 1e-30) / a0
        m = np.clip(mu(X), 1e-8, None)
        A = (m, m.copy(), m.copy(), zero, zero.copy(), zero.copy())
        itol = max(tol, 1e-4 * 0.35 ** k)
        new, it, rel = S.solve(rho, A, box.h, bc, tol=itol, maxiter=4000)
        d = float(np.abs(new - Psi).max() / max(np.abs(new).max(), 1e-300))
        hist.append(d)
        inner.append(rel)
        Psi = (1 - damp) * Psi + damp * new
        if d < picard_tol and itol <= tol:
            break
    return dict(Psi=Psi, iters=k + 1, resid=hist[-1], picard=hist,
                inner_resid=inner, shell_spread=0.0, K_shell=np.eye(3))


def solve_qumond(rho, box, a0=F.A0, form="simple", Mtot=None, A0_field=None,
                 tol=1e-11):
    """lap Psi = div[nu(|grad Phi_N|/A0) grad Phi_N].

    ``A0_field`` lets family B pass a position-dependent acceleration scale.
    """
    M = float(rho.sum() * box.vol) if Mtot is None else Mtot
    N = solve_newton(rho, box, Mtot=M, tol=tol)
    PhiN = N["Psi"]
    gx, gy, gz = np.gradient(PhiN, box.h, edge_order=2)
    gN = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    scale = a0 if A0_field is None else A0_field
    y = np.maximum(gN, 1e-30) / scale
    nu = F.nu_rar(y) if form == "rar" else F.nu_simple(y)
    nu = np.clip(nu, 1.0, 1e8)
    A = (nu, nu.copy(), nu.copy(), np.zeros(box.shape), np.zeros(box.shape),
         np.zeros(box.shape))
    src = S.apply_operator(PhiN, A, box.h)             # div[nu grad Phi_N]
    rho_eff = src / (4 * np.pi * G)
    bc = _spherical_mond_bc(box, M, a0, form)
    AI = F.iso_A(box.shape, 1.0)
    Psi, it, rel = S.solve(rho_eff, AI, box.h, bc, tol=tol, maxiter=8000)
    return dict(Psi=Psi, PhiN=PhiN, iters=it, resid=rel, nu=nu,
                shell_spread=0.0, K_shell=np.eye(3))


# ------------------------------------------------------------- diagnostics
def grad(Psi, h):
    gx, gy, gz = np.gradient(Psi, h, edge_order=2)
    return gx, gy, gz


def net_force(rho, Psi, box):
    """Total force on the whole configuration, integral of rho * (-grad Psi).

    For a reciprocal law this is exactly zero: internal forces cancel in pairs.
    Any nonzero value that survives the Newtonian null is a violation of
    Newton's third law by the LAW, not by the grid.
    """
    gx, gy, gz = grad(Psi, box.h)
    w = rho * box.vol
    return np.array([-(w * gx).sum(), -(w * gy).sum(), -(w * gz).sum()])


def force_at(Psi, box, point):
    """-grad Psi at an arbitrary point, by trilinear interpolation."""
    gx, gy, gz = grad(Psi, box.h)
    ax = box.ax
    out = []
    for gcomp in (gx, gy, gz):
        out.append(_trilerp(gcomp, ax, point))
    return -np.array(out)


def _trilerp(f, ax, p):
    n = len(ax)
    h = ax[1] - ax[0]
    t = (np.asarray(p, float) - ax[0]) / h
    i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
    w = t - i0
    c = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wt = ((1 - w[0]) if dx == 0 else w[0]) \
                    * ((1 - w[1]) if dy == 0 else w[1]) \
                    * ((1 - w[2]) if dz == 0 else w[2])
                c = c + wt * f[i0[0] + dx, i0[1] + dy, i0[2] + dz]
    return float(c)


def vcirc_axis(Psi, box, R):
    """Circular speed on the +x axis in the z = 0 plane: v^2 = R dPsi/dR."""
    n = box.n
    j = k = n // 2
    xs = box.ax
    prof = 0.25 * (Psi[:, j, k] + Psi[:, j - 1, k] + Psi[:, j, k - 1]
                   + Psi[:, j - 1, k - 1])
    d = np.gradient(prof, box.h, edge_order=2)
    dp = np.interp(R, xs, d)
    return np.sqrt(np.maximum(R * dp, 0.0))


def enclosed_mass_profile(rho, box, radii):
    r = box.r
    return np.array([float(rho[r <= R].sum() * box.vol) for R in radii])


# -------------------------------------------------------- spherical reduction
def spherical_g(cand, radii, Mtot, kr=None, PhiN=None):
    """Exact g(r) in spherical symmetry for each candidate kind.

    ``kr`` is the radial eigenvalue of K along the ray (families C, D, E);
    ``PhiN`` the Newtonian potential (family B).
    """
    gN = G * Mtot / radii ** 2
    kind, prm = cand.kind, cand.prm
    if kind == "scalar_mu":
        a0 = prm["a0"]
        return 0.5 * (gN + np.sqrt(gN * gN + 4.0 * gN * a0))
    if kind == "qumond":
        a0 = prm["a0"]
        y = gN / a0
        nu = F.nu_rar(y) if prm.get("form") == "rar" else F.nu_simple(y)
        return nu * gN
    if kind == "depth":
        A0 = F.A0_depth(gN, PhiN, prm)
        return F.nu_simple(gN / A0) * gN
    if kind in ("wells", "pairs", "tidal"):
        return gN / kr
    raise ValueError(kind)
