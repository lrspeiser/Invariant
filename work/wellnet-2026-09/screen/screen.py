"""screen.py -- STAGE 1 of the well-network funnel: mathematical elimination.

A candidate law is rejected here without fitting anything to anything.  The
screens, in the order they run:

    S1   dimensional consistency            (executed, not asserted)
    S2   rotation covariance
    S3   translation covariance
    S4   potential-gauge / external-offset invariance
    S5   positive-definiteness of K, and the critical point of parameterisations
         that break it
    S6   Newtonian high-acceleration limit, g -> g_N as g_N/a0 -> infinity
    S7   asymptotics: finite at large r, no worse than Newtonian at r -> 0
    S8   asymptotic-gain bound  (a bounded K cannot make a flat rotation curve)
    S9   source-label permutation invariance
    S10  reciprocity / momentum conservation
    S11  coarse-graining under uniform partition refinement          (Stage 1b)
    S12  coarse-graining under SELECTIVE refinement of one object    (Stage 1b)
    S13  coherence-scale classification: physical length or catalogue rows

Usage
-----
    import screen, families
    res = screen.run_screen(families.CANDIDATES["C1_wells_pow_p1"])
    print(res["verdict"], res["failed"])

Everything is callable on a candidate spec; nothing here reads data of any
kind.  KiDS and the wide binaries are not touched -- no observational file is
opened anywhere in this lane.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import dimx
import families as F
import fieldsolve as FS
from dimx import Q, DIMLESS, MASS, LENGTH, ACCEL, POTENTIAL, TIDAL, DimensionError

G = F.G
KPC = F.KPC
MSUN = F.MSUN
A0 = F.A0

# ------------------------------------------------------------------ tolerances
# Conjugate gradients on a system of condition number kappa cannot reach a
# relative residual below about kappa * eps_machine.  With the 1e-11 target of
# solver.py that forecloses anything above ~5e4, but the screen only refuses at
# 1e9, where even a 1e-7 residual is out of reach; between the two the solve is
# attempted and its residual reported.
COND_MAX = 1e9

TOL = dict(
    covariance=1e-10,        # algebraic covariance is exact arithmetic
    grid_covariance=1e-6,    # 90-degree lattice rotation of a solved field
    permutation=1e-12,
    pos_def=1e-8,
    newton_limit=1e-3,       # |g/g_N - 1| at g_N/a0 = 1e4
    newton_limit_y=1e4,
    asym_slope=0.05,         # |d ln g / d ln r + 2| for a bounded-K law
    reciprocity=1e-6,        # |F_net| / F_pair, above the Newtonian null
    coarse=1e-3,             # |dPhi|/|Phi| between N=1 and N=1e4
    coarse_tail=0.10,        # last-decade drift as a fraction of the total
    gauge=1e-3,              # |dv_c|/v_c under an external potential offset
)


# ================================================================== scenes
@dataclass
class Scene:
    """A fixed smooth mass distribution plus a well list describing it."""
    name: str
    wx: np.ndarray
    wm: np.ndarray
    Mtot: float
    scale: float                       # characteristic length
    rho_grid: Optional[np.ndarray] = None
    box: Optional[object] = None


def galaxy_cloud(Nq=65536, M=5e10 * MSUN, a=3 * KPC, centre=(0, 0, 0)):
    return F.equal_mass_cloud("plummer", Nq, M, a=a, umax=0.98, centre=centre)


def probe_points(nr=12, ndir=24, rmin=0.5 * KPC, rmax=40 * KPC, seed=7):
    """Probe points on a deterministic set of rays, log-spaced in radius."""
    rs = np.geomspace(rmin, rmax, nr)
    dirs = F._fib_dirs(ndir)
    return (dirs[None, :, :] * rs[:, None, None]).reshape(-1, 3)


# ================================================================ K at points
def _xp(use_gpu):
    use_gpu = F.HAVE_GPU if use_gpu is None else (use_gpu and F.HAVE_GPU)
    return (F._cp if use_gpu else np), use_gpu


@F.gpu_guard
def gN_analytic(points, wx, wm, soft=0.05 * KPC, block=1024, pblock=8192,
                use_gpu=None):
    xp, gpu = _xp(use_gpu)
    points = np.asarray(points, float)
    WX, WM = xp.asarray(wx), xp.asarray(wm)
    out = np.empty(len(points))
    for s in range(0, len(points), pblock):
        P = xp.asarray(points[s:s + pblock])
        g = xp.zeros((P.shape[0], 3))
        for i in range(0, WX.shape[0], block):
            d = WX[None, i:i + block, :] - P[:, None, :]
            r = xp.maximum(xp.sqrt((d * d).sum(-1)), soft)
            g += (WM[None, i:i + block, None] * d / r[..., None] ** 3).sum(1)
        g = G * xp.sqrt((g * g).sum(-1))
        out[s:s + pblock] = F.asnumpy(g) if gpu else g
    return out


def phiN_analytic(points, wx, wm, soft=0.05 * KPC):
    d = wx[None, :, :] - points[:, None, :]
    r = np.maximum(np.sqrt((d * d).sum(-1)), soft)
    return -(G * wm[None, :] / r).sum(1)


@F.gpu_guard
def tidal_analytic(points, wx, wm, soft=0.05 * KPC, block=512, pblock=8192,
                   use_gpu=None):
    """T_ij = sum_a G M_a (delta_ij - 3 n_i n_j) / r_a^3, exact for point wells."""
    xp, gpu = _xp(use_gpu)
    points = np.asarray(points, float)
    WX, WM = xp.asarray(wx), xp.asarray(wm)
    eye = xp.eye(3)
    out = np.empty((len(points), 3, 3))
    for s in range(0, len(points), pblock):
        P = xp.asarray(points[s:s + pblock])
        acc = xp.zeros((P.shape[0], 3, 3))
        for i in range(0, WX.shape[0], block):
            d = P[:, None, :] - WX[None, i:i + block, :]
            r = xp.maximum(xp.sqrt((d * d).sum(-1)), soft)
            n = d / r[..., None]
            c = G * WM[None, i:i + block] / r ** 3
            acc += (c[..., None, None]
                    * (eye[None, None]
                       - 3 * n[..., :, None] * n[..., None, :])).sum(1)
        out[s:s + pblock] = F.asnumpy(acc) if gpu else acc
    return out


@F.gpu_guard
def _smooth_rho(points, wx, wm, L, block=1024, pblock=8192, use_gpu=None):
    """rho_L(x) = sum_a M_a (2 pi L^2)^{-3/2} exp(-|x - x_a|^2 / 2 L^2)."""
    xp, gpu = _xp(use_gpu)
    points = np.asarray(points, float)
    WX, WM = xp.asarray(wx), xp.asarray(wm)
    norm = (2 * np.pi * L ** 2) ** -1.5
    out = np.empty(len(points))
    for s in range(0, len(points), pblock):
        P = xp.asarray(points[s:s + pblock])
        acc = xp.zeros(P.shape[0])
        for i in range(0, WX.shape[0], block):
            d2 = ((WX[None, i:i + block] - P[:, None]) ** 2).sum(-1)
            acc += (WM[None, i:i + block] * xp.exp(-d2 / (2 * L ** 2))).sum(1)
        acc = acc * norm
        out[s:s + pblock] = F.asnumpy(acc) if gpu else acc
    return out


def That_from_T(T, prm):
    tr = np.trace(T, axis1=-2, axis2=-1)
    T0 = T - tr[..., None, None] * np.eye(3) / 3.0
    nrm = np.sqrt(prm["eps_T"] ** 2 + (T0 * T0).sum((-1, -2)))
    return T0 / nrm[..., None, None]


def K_at(cand, points, wx, wm, use_gpu=None):
    """The response tensor of ``cand`` at ``points`` given the well list."""
    prm = cand.prm
    if cand.kind == "wells":
        gN = gN_analytic(points, wx, wm) if prm.get("shape") == "pow_g" else None
        S = F.S_wells(points, wx, wm, prm, gN=gN, use_gpu=use_gpu)
        return F.K_from_S(S, prm)
    if cand.kind == "wells_linear":                      # broken parameterisation
        S = F.S_wells(points, wx, wm, prm, use_gpu=use_gpu)
        return np.eye(3)[None] + prm["sT"] * S
    if cand.kind == "pairs":
        C = F.C_pairs(points, wx, wm, prm, use_gpu=use_gpu)
        return F.K_from_C(C, prm)
    if cand.kind == "pairs_linear":
        C = F.C_pairs(points, wx, wm, prm, use_gpu=use_gpu)
        return np.eye(3)[None] - prm["alpha"] * C
    if cand.kind == "tidal":
        T = tidal_analytic(points, wx, wm)
        return F.K_from_T(That_from_T(T, prm), prm)
    if cand.kind == "smoothrho":
        # rho_L(x) = sum_a M_a (2 pi L^2)^{-3/2} exp(-|x - x_a|^2 / 2 L^2)
        # A law with a REAL coherence length: the sum is a quadrature of a
        # convolution, so once the partition is finer than L it stops moving.
        rho = _smooth_rho(points, wx, wm, prm["L"], use_gpu=use_gpu)
        k = np.exp(prm["s"] * rho / prm["rho0"])
        return k[:, None, None] * np.eye(3)[None]
    if cand.kind == "count":
        cnt = np.zeros(len(points))
        for s in range(0, len(points), 8192):
            P = points[s:s + 8192]
            c = np.zeros(len(P))
            for i in range(0, len(wx), 1024):
                d2 = ((wx[None, i:i + 1024] - P[:, None]) ** 2).sum(-1)
                c += (d2 < prm["L"] ** 2).sum(1)
            cnt[s:s + 8192] = c
        k = np.exp(np.minimum(prm["sc"] * cnt / prm["N0"], 40.0))
        return k[:, None, None] * np.eye(3)[None]
    if cand.kind == "newton":
        return np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy()
    raise ValueError(f"K_at not defined for kind {cand.kind}")


TENSOR_KINDS = {"wells", "wells_linear", "pairs", "pairs_linear", "tidal",
                "count", "smoothrho", "newton"}


def solve_candidate(cand, rho, box, wx, wm, Mtot, tol=1e-11):
    """One dispatcher so every screen and every Stage-2 geometry solves the
    same candidate the same way."""
    if cand.kind == "newton":
        return FS.solve_newton(rho, box, Mtot=Mtot, tol=tol)
    if cand.kind == "scalar_mu":
        return FS.solve_aqual(rho, box, a0=cand.prm["a0"], Mtot=Mtot, tol=tol)
    if cand.kind == "qumond":
        return FS.solve_qumond(rho, box, a0=cand.prm["a0"],
                               form=cand.prm.get("form", "simple"),
                               Mtot=Mtot, tol=tol)
    if cand.kind == "depth":
        N = FS.solve_newton(rho, box, Mtot=Mtot, tol=tol)
        A0f = F.A0_depth(np.zeros_like(N["Psi"]), N["Psi"], cand.prm)
        return FS.solve_qumond(rho, box, a0=cand.prm["a0"], Mtot=Mtot,
                               A0_field=A0f, tol=tol)
    K = K_at(cand, box.pts, wx, wm)
    ev = np.linalg.eigvalsh(K)
    lo, hi = float(ev.min()), float(ev.max())
    cond = float((ev[:, 2] / np.maximum(ev[:, 0], 1e-300)).max())
    # A degenerate K is a property of the candidate, not of the solver, and it
    # has to be reported as one.  Handing a system with condition number 1e12
    # to conjugate gradients produces eight thousand useless iterations and a
    # residual that looks like a convergence failure; refusing it up front says
    # what actually happened.  Family D at p < 1 reaches this because
    # ||C|| grows like N^(2-2p), so exp[-alpha C] collapses as the catalogue is
    # refined.
    if lo <= 0.0 or cond > COND_MAX or not np.isfinite(cond):
        raise F.SingularK(
            f"response tensor degenerate: lambda_min = {lo:.3e}, "
            f"lambda_max = {hi:.3e}, condition number = {cond:.3e}. The field "
            f"equation div[K grad Psi] = 4 pi G rho is not solvable to the "
            f"required tolerance with this K: conjugate gradients on a system "
            f"of condition number kappa cannot reach a relative residual below "
            f"about kappa * 2e-16 in float64, so kappa > {COND_MAX:.0e} "
            f"forecloses the 1e-11 target before a single iteration runs")
    r = FS.solve_K(rho, K, box, Mtot=Mtot, tol=tol)
    r["K_eig_min"] = lo
    r["K_eig_max"] = hi
    r["K_cond"] = cond
    return r


# ================================================================== controls
CONTROLS: Dict[str, F.Candidate] = {}


def _ctrl(c):
    CONTROLS[c.name] = c
    return c


_ctrl(F.Candidate("X0_newton", "control", "newton", dict(), dict(),
                  "K = I. Negative control: must pass every screen.",
                  reciprocal_by_construction=True))
_ctrl(F.Candidate("X1_wells_linear", "control", "wells_linear",
                  F._pC(sT=2.0), F._DIMS_C,
                  "K = I + sT S instead of exp[...]: the parameterisation the "
                  "positive-definiteness screen is supposed to catch."))
_ctrl(F.Candidate("X2_count_wells", "control", "count",
                  dict(L=8.0 * KPC, N0=50.0, sc=0.02),
                  dict(L=LENGTH, N0=DIMLESS, sc=DIMLESS),
                  "K = exp[s_c n_wells(<L)/N_0] I. Pure catalogue counting -- "
                  "the number of rows, with no mass weighting at all. Positive "
                  "control: the coarse-graining screen MUST fire on this."))
_ctrl(F.Candidate("X3_pairs_linear", "control", "pairs_linear",
                  F._pD(alpha=3.0), F._DIMS_D,
                  "K = I - alpha C instead of exp[-alpha C]."))
_ctrl(F.Candidate("X4_smooth_density", "control", "smoothrho",
                  dict(L=8.0 * KPC, rho0=4.0e-22, s=0.5),
                  dict(L=LENGTH, rho0=dimx.DENSITY, s=DIMLESS),
                  "K = exp[s rho_L/rho_0] I with rho_L the mass density "
                  "smoothed on a fixed length L. Positive control for the "
                  "OTHER branch of the coherence test: this law has a genuine "
                  "physical scale, so its drift must stop once the partition "
                  "is finer than L and stay stopped."))

ALL = dict(F.CANDIDATES)
ALL.update(CONTROLS)


# ============================================================== S1 dimensions
def s1_dimensions(cand):
    """Run the candidate's kernel with dimensional arithmetic."""
    prm = cand.prm
    dims = cand.prm_dims
    detail, ok = [], True

    def qprm():
        out = {}
        for k, v in prm.items():
            if isinstance(v, str):
                out[k] = v
            else:
                out[k] = Q(v, dims.get(k, DIMLESS), tag=k)
        return out

    try:
        P = qprm()
        if cand.kind in ("wells", "wells_linear"):
            w = F.weight_C(Q(1e10 * MSUN, MASS, "M_a"),
                           Q(5 * KPC, LENGTH, "r_a"),
                           Q(1e-10, ACCEL, "g_N"), P, xp=dimx)
            detail.append(f"w_a = {w}")
            ok &= w.is_dimensionless()
            detail.append(f"S is w-weighted and normalised by sum|w| -> "
                          f"dimensionless by construction")
        elif cand.kind in ("pairs", "pairs_linear"):
            w = F.weight_D(Q(1e10 * MSUN, MASS, "M_a"),
                           Q(2e10 * MSUN, MASS, "M_b"),
                           Q(3 * KPC, LENGTH, "d_ab"), P, xp=dimx)
            detail.append(f"w_ab = {w}")
            ok &= w.is_dimensionless()
            tube = dimx.exp(-(Q(1 * KPC, LENGTH, "d_perp") ** 2)
                            / (2 * P["sigma_perp"] ** 2))
            detail.append(f"tube weight = {tube}")
            ok &= tube.is_dimensionless()
        elif cand.kind == "tidal":
            T = Q(1e-30, TIDAL, "T_ij")
            nrm = (P["eps_T"] ** 2 + T ** 2) ** 0.5
            That = T / nrm
            detail.append(f"That = {That}")
            ok &= That.is_dimensionless()
        elif cand.kind == "depth":
            a0 = F.A0_depth(Q(1e-10, ACCEL, "g_N"), Q(-3e10, POTENTIAL, "Phi"),
                            P, xp=dimx)
            detail.append(f"A_0 = {a0}")
            ok &= (a0.d == ACCEL)
            arg = Q(1e-10, ACCEL, "g_N") / a0
            detail.append(f"g_N/A_0 = {arg}")
            ok &= arg.is_dimensionless()
        elif cand.kind in ("scalar_mu", "qumond"):
            X = Q(1e-10, ACCEL, "|grad Phi|") / P["a0"]
            detail.append(f"mu argument = {X}")
            ok &= X.is_dimensionless()
        elif cand.kind == "count":
            arg = Q(37.0, DIMLESS, "n_wells") / P["N0"] * P["sc"]
            detail.append(f"exp argument = {arg}")
            ok &= arg.is_dimensionless()
        elif cand.kind == "newton":
            detail.append("K = I, trivially dimensionless")
        # every law: div[K grad Phi] must match 4 pi G rho
        lhs = Q(1.0, POTENTIAL) / Q(1.0, LENGTH) ** 2
        rhs = Q(G, dimx.G_DIM) * Q(1.0, dimx.DENSITY)
        ok &= (lhs.d == rhs.d)
        detail.append(f"field equation balance: div[K grad Phi] "
                      f"[{dimx._fmt(lhs.d)}] vs 4 pi G rho "
                      f"[{dimx._fmt(rhs.d)}]")
    except DimensionError as e:
        ok = False
        detail.append(f"DimensionError: {e}")

    # negative control: corrupt one parameter and require the screen to fire
    caught = None
    if cand.kind in ("wells", "pairs", "depth", "scalar_mu", "qumond", "tidal"):
        bad = dict(cand.prm_dims)
        key = {"wells": "L", "pairs": "L", "depth": "Phi0",
               "scalar_mu": "a0", "qumond": "a0", "tidal": "eps_T"}[cand.kind]
        bad[key] = DIMLESS if bad.get(key) != DIMLESS else LENGTH
        c2 = F.Candidate(cand.name, cand.family, cand.kind, cand.prm, bad)
        caught = _dim_probe(c2)
    return dict(**_pf(ok), value=float(ok), tol=None,
                detail="; ".join(detail),
                negative_control_fires=caught)


def _dim_probe(cand):
    """Return True if running the kernel with these (deliberately wrong)
    parameter dimensions raises DimensionError -- i.e. the screen has teeth."""
    P = {k: (v if isinstance(v, str) else Q(v, cand.prm_dims.get(k, DIMLESS), k))
         for k, v in cand.prm.items()}
    try:
        if cand.kind in ("wells", "wells_linear"):
            w = F.weight_C(Q(1e10 * MSUN, MASS), Q(5 * KPC, LENGTH),
                           Q(1e-10, ACCEL), P, xp=dimx)
            return not w.is_dimensionless()
        if cand.kind in ("pairs", "pairs_linear"):
            w = F.weight_D(Q(1e10 * MSUN, MASS), Q(2e10 * MSUN, MASS),
                           Q(3 * KPC, LENGTH), P, xp=dimx)
            return not w.is_dimensionless()
        if cand.kind == "depth":
            a0 = F.A0_depth(Q(1e-10, ACCEL), Q(-3e10, POTENTIAL), P, xp=dimx)
            return a0.d != ACCEL
        if cand.kind in ("scalar_mu", "qumond"):
            return not (Q(1e-10, ACCEL) / P["a0"]).is_dimensionless()
        if cand.kind == "tidal":
            T = Q(1e-30, TIDAL)
            return not (T / (P["eps_T"] ** 2 + T ** 2) ** 0.5).is_dimensionless()
    except DimensionError:
        return True
    return False


def _pf(ok):
    return dict(passed=bool(ok))


# =========================================================== S2/S3 covariance
def _rand_rot(seed=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Qm, R = np.linalg.qr(A)
    if np.linalg.det(Qm) < 0:
        Qm[:, 0] *= -1
    return Qm


def s2_rotation(cand, wx, wm, pts):
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, value=0.0, tol=TOL["covariance"],
                    detail="scalar law: mu / nu depend only on |grad Phi|, "
                           "a rotational invariant; covariance is structural")
    R = _rand_rot()
    K0 = K_at(cand, pts, wx, wm)
    K1 = K_at(cand, pts @ R.T, wx @ R.T, wm)
    tgt = np.einsum("ij,pjk,lk->pil", R, K0, R)
    err = float(np.abs(K1 - tgt).max() / max(np.abs(K0).max(), 1e-300))
    return dict(passed=err < TOL["covariance"], value=err,
                tol=TOL["covariance"],
                detail="K(Rx; Rw) vs R K(x; w) R^T, random SO(3)")


def s2b_grid_rotation(cand, n=36, Lbox=100.0):
    """Rotation covariance of the SOLVED field, not just of the construction.

    A 90-degree turn about z is an exact symmetry of the cubic lattice, so this
    tests the discretisation as well as the law: Psi_rotated must equal
    np.rot90(Psi, 1, axes=(0,1)) to solver tolerance.  An arbitrary angle would
    mix in interpolation error and could not separate the two.
    """
    box = F.Box(n, Lbox)
    M1, M2 = 6e10 * MSUN, 2e10 * MSUN
    p1 = np.array([-14.0, 6.0, 0.0]) * KPC
    p2 = np.array([11.0, -9.0, 3.0]) * KPC
    sig = 4.0 * KPC

    def rot(v):
        return np.array([-v[1], v[0], v[2]])

    def build(a, b):
        rho = (F.gauss_rho(box.pts, M1, sig, a)
               + F.gauss_rho(box.pts, M2, sig, b)).reshape(box.shape)
        return F.normalise_mass(rho, box.vol, M1 + M2)

    wa = np.stack([p1, p2])
    wb = np.stack([rot(p1), rot(p2)])
    wm = np.array([M1, M2])
    ra = solve_candidate(cand, build(p1, p2), box, wa, wm, M1 + M2)
    rb = solve_candidate(cand, build(rot(p1), rot(p2)), box, wb, wm, M1 + M2)
    tgt = np.rot90(ra["Psi"], 1, axes=(0, 1))
    err = float(np.abs(rb["Psi"] - tgt).max() / np.abs(tgt).max())
    return dict(passed=err < TOL["grid_covariance"], value=err,
                tol=TOL["grid_covariance"],
                cg_resid=float(max(ra["resid"], rb["resid"])),
                detail="90-degree lattice rotation of the whole problem, "
                       "solved field compared cell by cell")


def s3_translation(cand, wx, wm, pts):
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, value=0.0, tol=TOL["covariance"],
                    detail="scalar law: depends on grad Phi only")
    t = np.array([3.1, -7.7, 11.3]) * KPC
    K0 = K_at(cand, pts, wx, wm)
    K1 = K_at(cand, pts + t, wx + t, wm)
    err = float(np.abs(K1 - K0).max() / max(np.abs(K0).max(), 1e-300))
    return dict(passed=err < TOL["covariance"], value=err,
                tol=TOL["covariance"], detail="K(x+t; w+t) vs K(x; w)")


def s4_gauge(cand, wx, wm, pts):
    """Is the prediction changed by an external, nearly uniform potential?

    A galaxy sitting in a cluster acquires a large |Phi| offset with almost no
    local tidal field.  A law that reads |Phi| rather than its derivatives
    responds to that offset, which means the answer depends on a quantity no
    local experiment can measure.
    """
    if cand.kind != "depth":
        return dict(passed=True, value=0.0, tol=TOL["gauge"],
                    detail="law uses only derivatives of Phi; |Phi| never enters")
    M_ext, R_ext = 1e14 * MSUN, 1.0e3 * KPC
    dPhi = -G * M_ext / R_ext
    g_ext = G * M_ext / R_ext ** 2
    r = np.geomspace(2 * KPC, 30 * KPC, 24)
    Mtot = wm.sum()
    gN = G * Mtot / r ** 2
    PhiN = -G * Mtot / r
    A_a = F.A0_depth(gN, PhiN, cand.prm)
    A_b = F.A0_depth(gN, PhiN + dPhi, cand.prm)
    g_a = F.nu_simple(gN / A_a) * gN
    g_b = F.nu_simple(gN / A_b) * gN
    dv = float(np.abs(np.sqrt(g_b / g_a) - 1).max())
    return dict(passed=dv < TOL["gauge"], value=dv, tol=TOL["gauge"],
                detail=(f"external M={M_ext/MSUN:.0e} Msun at {R_ext/KPC:.0f} kpc: "
                        f"dPhi={dPhi:.3e} m2/s2 (|dPhi|/Phi0="
                        f"{abs(dPhi)/cand.prm.get('Phi0',1):.3g}), local tide "
                        f"g_ext={g_ext:.2e} m/s2; max |dv_c|/v_c = {dv:.4g}"))


# ======================================================== S5 positive definite
def s5_positive_definite(cand, wx, wm, pts):
    if cand.kind not in TENSOR_KINDS:
        # for the scalar families the response tensor is mu(X) I (AQUAL) or
        # nu(y) I (QUMOND / family B), isotropic by construction; the question
        # is whether the scalar stays strictly positive over its whole range.
        y = np.geomspace(1e-8, 1e8, 200)
        if cand.kind == "scalar_mu":
            s = F.mu_simple(y)
            nm = "mu(X)"
        else:
            s = F.nu_rar(y) if cand.prm.get("form") == "rar" else F.nu_simple(y)
            nm = "nu(y)"
        lmin = float(np.nanmin(s))
        # Positive-definiteness and UNIFORM ellipticity are different claims.
        # mu(X) -> X as X -> 0, so AQUAL's operator is strictly positive at
        # every finite acceleration but its ellipticity constant goes to zero
        # in the deep-MOND limit.  That is a conditioning statement, not a
        # well-posedness failure, so it is reported separately rather than
        # being charged as a positive-definiteness failure.
        return dict(passed=bool(lmin > 0.0 and np.all(np.isfinite(s))),
                    value=lmin, tol=0.0,
                    lambda_max=float(np.nanmax(s)), cond=float(np.nanmax(s) / lmin),
                    asymmetry=0.0, critical=None,
                    uniformly_elliptic=bool(lmin > TOL["pos_def"]),
                    detail=(f"isotropic response {nm} I; min over "
                            f"g_N/a0 in [1e-8, 1e8] is {lmin:.3e}, strictly "
                            f"positive. Not uniformly elliptic for AQUAL: "
                            f"mu -> X as X -> 0, so the ellipticity constant "
                            f"vanishes with the acceleration"))
    K = K_at(cand, pts, wx, wm)
    asym = float(np.abs(K - np.swapaxes(K, -1, -2)).max()
                 / max(np.abs(K).max(), 1e-300))
    ev = np.linalg.eigvalsh(0.5 * (K + np.swapaxes(K, -1, -2)))
    lmin, lmax = float(ev.min()), float(ev.max())
    ok = (lmin > TOL["pos_def"]) and (asym < 1e-12)
    crit = None
    if cand.kind in ("wells", "wells_linear"):
        crit = _critical_sT(cand, wx, wm, pts)
    if cand.kind in ("pairs", "pairs_linear"):
        crit = _critical_alpha(cand, wx, wm, pts)
    return dict(passed=ok, value=lmin, tol=TOL["pos_def"],
                lambda_max=lmax, cond=lmax / max(lmin, 1e-300),
                asymmetry=asym, critical=crit,
                detail=("eigenvalues of K over the probe set; 'critical' is the "
                        "parameter value at which the LINEARISED form loses "
                        "positive-definiteness (exp[.] never does)"))


def _critical_sT(cand, wx, wm, pts):
    gN = (gN_analytic(pts, wx, wm)
          if cand.prm.get("shape") == "pow_g" else None)
    S = F.S_wells(pts, wx, wm, cand.prm, gN=gN)
    ev = np.linalg.eigvalsh(S)
    lo, hi = float(ev.min()), float(ev.max())
    out = dict(S_eig_min=lo, S_eig_max=hi)
    out["sT_crit_linear_pos"] = float(-1.0 / lo) if lo < 0 else float("inf")
    out["sT_crit_linear_neg"] = float(-1.0 / hi) if hi > 0 else float("-inf")
    out["exp_form_pos_def_for_all_sT"] = True
    return out


def _critical_alpha(cand, wx, wm, pts):
    C = F.C_pairs(pts, wx, wm, cand.prm)
    ev = np.linalg.eigvalsh(C)
    hi = float(ev.max())
    return dict(C_eig_max=hi, C_eig_min=float(ev.min()),
                alpha_crit_linear=float(1.0 / hi) if hi > 0 else float("inf"),
                exp_form_pos_def_for_all_alpha=True)


# ======================================================== S6 Newtonian limit
def s6_newtonian_limit(cand, wx, wm):
    y = np.array([1e1, 1e2, 1e3, 1e4, 1e6])
    if cand.kind in ("scalar_mu", "qumond", "depth"):
        a0 = cand.prm["a0"]
        gN = y * a0
        if cand.kind == "scalar_mu":
            g = 0.5 * (gN + np.sqrt(gN * gN + 4 * gN * a0))
        elif cand.kind == "qumond":
            nu = F.nu_rar(y) if cand.prm.get("form") == "rar" else F.nu_simple(y)
            g = nu * gN
        else:
            Phi_deep = -(200e3) ** 2                      # a 200 km/s galaxy
            A = F.A0_depth(gN, Phi_deep, cand.prm)
            g = F.nu_simple(gN / A) * gN
        dev = np.abs(g / gN - 1.0)
        v = float(dev[y == TOL["newton_limit_y"]][0])
        return dict(passed=v < TOL["newton_limit"], value=v,
                    tol=TOL["newton_limit"],
                    curve={f"{yy:.0e}": float(dd) for yy, dd in zip(y, dev)},
                    detail="|g/g_N - 1| against g_N/a0"
                           + ("; |Phi| set to (200 km/s)^2, the depth a real "
                              "galaxy has" if cand.kind == "depth" else ""))
    # tensor families: the requirement is K -> I where g_N/a0 is large
    pts, gN = _high_accel_points(wx, wm)
    K = K_at(cand, pts, wx, wm)
    dK = np.abs(K - np.eye(3)[None]).reshape(len(pts), -1).max(1)
    ev = np.linalg.eigvalsh(K)
    aniso = ev[:, 2] / ev[:, 0] - 1.0
    y = gN / A0
    sel = y > TOL["newton_limit_y"]
    if not sel.any():
        sel = gN > np.quantile(gN, 0.9)
    # the deviation anywhere in the galaxy, for comparison: these families have
    # no g_N dependence at all, so the two numbers differ only through geometry
    gen = probe_points(10, 20)
    Kg = K_at(cand, gen, wx, wm)
    dKg = float(np.abs(Kg - np.eye(3)[None]).max())
    v = float(dK[sel].max())
    return dict(passed=v < TOL["newton_limit"], value=v,
                tol=TOL["newton_limit"],
                anisotropy=float(aniso[sel].max()),
                max_gN_over_a0=float(y.max()),
                min_gN_over_a0_in_selection=float(y[sel].min()),
                max_deviation_anywhere=dKg,
                depends_on_gN=bool(cand.prm.get("shape") == "pow_g"),
                detail=("max |K - I| at the highest-acceleration points "
                        "sampled. K -> I is necessary for g -> g_N. For "
                        "families C, D and E the tensor carries no explicit "
                        "g_N dependence at all, so the anisotropy at "
                        "g_N/a0 = 1e7 is the same object as the anisotropy in "
                        "the outskirts; 'max_deviation_anywhere' is that "
                        "galaxy-scale value"))


def _high_accel_points(wx, wm, n=400):
    """Points very close to individual wells, where g_N/a0 reaches
    solar-system values (1e6 - 1e8) rather than merely galactic ones."""
    rng = np.random.default_rng(11)
    idx = rng.integers(0, len(wx), n)
    d = F._fib_dirs(n) * (np.geomspace(2e-4, 2e-2, n)[:, None] * KPC)
    pts = wx[idx] + d
    return pts, gN_analytic(pts, wx, wm, soft=1e-5 * KPC)


# ============================================================ S7 asymptotics
def s7_asymptotics(cand, wx, wm):
    """Large-r and small-r behaviour, from the exact spherical reduction."""
    Mtot = float(wm.sum())
    r = np.geomspace(50 * KPC, 5e5 * KPC, 40)
    dirs = F._fib_dirs(16)
    pts = (dirs[None] * r[:, None, None]).reshape(-1, 3)
    if cand.kind in TENSOR_KINDS:
        K = K_at(cand, pts, wx, wm)
        n = pts / np.linalg.norm(pts, axis=1)[:, None]
        kr = np.einsum("pi,pij,pj->p", n, K, n).reshape(len(r), -1).mean(1)
        g = G * Mtot / (kr * r ** 2)
    else:
        PhiN = -G * Mtot / r
        g = FS.spherical_g(cand, r, Mtot, PhiN=PhiN)
    sl = np.gradient(np.log(g), np.log(r))
    slope_out = float(sl[-5:].mean())
    finite = bool(np.all(np.isfinite(g)) and np.all(g > 0))
    # r -> 0 (near a single well)
    r0 = np.geomspace(1e-4, 1.0, 30) * KPC
    p0 = np.stack([r0, np.zeros_like(r0), np.zeros_like(r0)], 1) + wx[0]
    if cand.kind in TENSOR_KINDS:
        K0 = K_at(cand, p0, wx, wm)
        kr0 = K0[:, 0, 0]
        g0 = G * wm[0] / (kr0 * r0 ** 2)
    else:
        g0 = FS.spherical_g(cand, r0, float(wm[0]),
                            PhiN=-G * wm[0] / r0)
    sl0 = float(np.gradient(np.log(g0), np.log(r0))[:5].mean())
    # Direction discontinuity of K at a catalogue point.  n_a n_a^T is EVEN in
    # n_a, so approaching along +e and -e gives the same limit; the genuine
    # discontinuity is between DIFFERENT lines of approach, and it survives
    # eps -> 0.  The softening r_soft has to be pushed well below eps or it is
    # the regulator, not the law, that is being measured.
    disc = _well_discontinuity(cand, wx, wm)
    ok = finite and (slope_out < -0.5) and (sl0 > -3.5)
    return dict(passed=ok, value=slope_out, tol=None,
                slope_large_r=slope_out, slope_small_r=sl0,
                finite=finite, K_discontinuity_at_well=disc,
                detail=("d ln g / d ln r at r ~ 1e5 r_s (Newton -2, MOND -1) "
                        "and at r -> 0 near one well (Newton -2). "
                        "'K_discontinuity_at_well' is |K(+eps) - K(-eps)| "
                        "across a catalogue point: the unit vector n_a is "
                        "undefined there, so K is direction-discontinuous"))


# ====================================================== S8 asymptotic gain
def _well_discontinuity(cand, wx, wm, eps_frac=1e-3):
    """max_|e|=1 spread of K(x_a + eps e) over the direction e, as eps -> 0.

    For family C the limit is  |exp[s_T(2/3) f] - exp[-s_T(1/3) f]|  with
    f = w_a / sum_b w_b the fractional weight of that one row: the response
    tensor has no value AT a catalogue point, only a direction-dependent limit.
    The jump does not shrink with eps, so it is not a resolution effect.
    """
    if cand.kind not in TENSOR_KINDS:
        return None
    d = np.linalg.norm(wx - wx[0], axis=1)
    d[0] = np.inf
    nn = float(d.min()) if len(wx) > 1 else 10 * KPC
    eps = eps_frac * nn
    prm = dict(cand.prm)
    for k in ("r_soft", "d_soft"):
        if k in prm:
            prm[k] = min(prm[k], eps * 1e-3)
    c2 = cand.copy_with(prm=prm)
    dirs = F._fib_dirs(24)
    out = {}
    for tag, W, M in (("N_partition", wx, wm),
                      ("single_well", wx[:1], wm[:1])):
        K = K_at(c2, W[0] + eps * dirs, W, M)
        spread = float((K.max(0) - K.min(0)).max() / max(np.abs(K).max(), 1e-300))
        out[tag] = spread
    out["eps_kpc"] = eps / KPC
    out["nearest_neighbour_kpc"] = nn / KPC
    out["softening_used_kpc"] = min(
        [prm[k] for k in ("r_soft", "d_soft") if k in prm] or [0.0]) / KPC
    return out


def s8_gain_bound(cand, wx, wm):
    """Can the law supply an UNBOUNDED boost g/g_N at large radius?

    Pure mathematics, no data: a flat rotation curve v_c = const needs
    g/g_N = v_c^2 r / (G M) which grows without bound in r.  If K tends to a
    constant tensor then Psi -> -GM/(sqrt(det K) sqrt(r^T K^-1 r)) exactly
    (this is the far field solver.py already uses) and the boost saturates at
    1/lambda_min(K).  A bounded K therefore cannot make a flat rotation curve
    -- it can only rescale G.
    """
    Mtot = float(wm.sum())
    r = np.geomspace(20 * KPC, 2e4 * KPC, 30)
    dirs = F._fib_dirs(16)
    pts = (dirs[None] * r[:, None, None]).reshape(-1, 3)
    if cand.kind in TENSOR_KINDS:
        K = K_at(cand, pts, wx, wm)
        ev = np.linalg.eigvalsh(K)
        boost = 1.0 / ev[:, 0]
        bmax = float(boost.max())
        n = pts / np.linalg.norm(pts, axis=1)[:, None]
        kr = np.einsum("pi,pij,pj->p", n, K, n).reshape(len(r), -1).mean(1)
        boost_r = 1.0 / kr
        growing = bool(boost_r[-1] > 3 * boost_r[0])
        unbounded = growing
    else:
        gN = G * Mtot / r ** 2
        g = FS.spherical_g(cand, r, Mtot, PhiN=-G * Mtot / r)
        boost_r = g / gN
        bmax = float(boost_r.max())
        unbounded = bool(boost_r[-1] > 3 * boost_r[0])
    need = (200e3) ** 2 * r / (G * Mtot)                  # boost for v_flat=200
    return dict(passed=None, value=bmax, tol=None,
                unbounded_boost=bool(unbounded),
                boost_at_20kpc=float(boost_r[0]),
                boost_at_2e4kpc=float(boost_r[-1]),
                boost_needed_at_2e4kpc=float(need[-1]),
                shortfall=float(need[-1] / max(boost_r[-1], 1e-300)),
                detail=("INFORMATIONAL, not a pass/fail gate: max g/g_N the "
                        "law can supply. A law whose K is bounded saturates, "
                        "so it can only rescale G and can never make a flat "
                        "rotation curve; 'needed' is the kinematic requirement "
                        "for v_c = 200 km/s, which grows linearly with r. "
                        "Newton itself is 'bounded' here, which is the point "
                        "of the programme"))


# ==================================================== S9 permutation invariance
def s9_permutation(cand, wx, wm, pts, ntrial=6):
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, value=0.0, tol=TOL["permutation"],
                    detail="no well list enters the law")
    K0 = K_at(cand, pts, wx, wm)
    worst = 0.0
    rng = np.random.default_rng(101)
    orders = [rng.permutation(len(wm)) for _ in range(ntrial - 2)]
    orders.append(np.arange(len(wm))[::-1])
    orders.append(np.argsort(wm))
    for o in orders:
        K1 = K_at(cand, pts, wx[o], wm[o])
        worst = max(worst, float(np.abs(K1 - K0).max()
                                 / max(np.abs(K0).max(), 1e-300)))
    return dict(passed=worst < TOL["permutation"], value=worst,
                tol=TOL["permutation"],
                detail=("random permutations, reversal and sort-by-mass of the "
                        "catalogue rows; only floating-point summation order "
                        "should change"))


# =========================================================== S10 reciprocity
def s10_reciprocity(cand, n=40, Lbox=120.0):
    """Net force on the whole configuration, which must vanish.

    Two UNEQUAL masses, so no symmetry hides the violation.  The Newtonian
    K = I run on the identical grid gives the numerical null; anything the law
    produces above that null is a real violation of Newton's third law.

    The identity checked alongside it is exact (see ``_identity_force``):

        F_net,i = - oint T_ij n_j dS
                  - (1/8 pi G) integral (d_i K_jk) d_j Psi d_k Psi

    with T the generalised Maxwell stress.  The surface piece is what Newton
    has; the volume piece exists only because K depends on position, and it IS
    the third-law violation.  Both are computed here, so a nonzero net force is
    attributed to a named term rather than to the solver.
    """
    box = F.Box(n, Lbox)
    M1, M2 = 8e10 * MSUN, 2e10 * MSUN
    d = 25.0 * KPC
    c1 = np.array([-d * M2 / (M1 + M2), 0.0, 0.0])
    c2 = np.array([+d * M1 / (M1 + M2), 0.0, 0.0])
    sig = 3.0 * KPC
    rho = (F.gauss_rho(box.pts, M1, sig, c1)
           + F.gauss_rho(box.pts, M2, sig, c2)).reshape(box.shape)
    Mtot = M1 + M2
    rho = F.normalise_mass(rho, box.vol, Mtot)
    wx = np.stack([c1, c2])
    wm = np.array([M1, M2])
    Fref = G * M1 * M2 / d ** 2

    null = FS.solve_newton(rho, box, Mtot=Mtot)
    Fn = FS.net_force(rho, null["Psi"], box)
    null_rel = float(np.linalg.norm(Fn) / Fref)

    if cand.kind == "newton":
        law_rel, ident_rel, extra = null_rel, 0.0, {}
    elif cand.kind == "scalar_mu":
        r = FS.solve_aqual(rho, box, a0=cand.prm["a0"], Mtot=Mtot)
        Fl = FS.net_force(rho, r["Psi"], box)
        law_rel, ident_rel, extra = float(np.linalg.norm(Fl) / Fref), None, {}
    elif cand.kind == "qumond":
        r = FS.solve_qumond(rho, box, a0=cand.prm["a0"],
                            form=cand.prm.get("form", "simple"), Mtot=Mtot)
        Fl = FS.net_force(rho, r["Psi"], box)
        law_rel, ident_rel, extra = float(np.linalg.norm(Fl) / Fref), None, {}
    elif cand.kind == "depth":
        N = FS.solve_newton(rho, box, Mtot=Mtot)
        A0f = F.A0_depth(np.zeros_like(N["Psi"]), N["Psi"], cand.prm)
        r = FS.solve_qumond(rho, box, a0=cand.prm["a0"], Mtot=Mtot,
                            A0_field=A0f)
        Fl = FS.net_force(rho, r["Psi"], box)
        law_rel, ident_rel, extra = float(np.linalg.norm(Fl) / Fref), None, {}
    else:
        K = K_at(cand, box.pts, wx, wm)
        ev = np.linalg.eigvalsh(K)
        cnd = float((ev[:, 2] / np.maximum(ev[:, 0], 1e-300)).max())
        if ev.min() <= 0 or cnd > COND_MAX or not np.isfinite(cnd):
            raise F.SingularK(
                f"response tensor degenerate (lambda_min = {ev.min():.3e}, "
                f"condition number = {cnd:.3e}); no bounded solution to "
                f"compare forces with")
        r = FS.solve_K(rho, K, box, Mtot=Mtot)
        Fl = FS.net_force(rho, r["Psi"], box)
        law_rel = float(np.linalg.norm(Fl) / Fref)
        Fid, Fsurf, Fvol = _identity_force(K, r["Psi"], box)
        ident_rel = float(np.linalg.norm(Fid) / Fref)
        mass_out = float(rho[5:-5, 5:-5, 5:-5].sum() * box.vol / Mtot)
        extra = dict(F_direct=[float(x) for x in Fl],
                     F_identity=[float(x) for x in Fid],
                     F_identity_surface=[float(x) for x in Fsurf],
                     F_identity_gradK_volume=[float(x) for x in Fvol],
                     gradK_term_rel=float(np.linalg.norm(Fvol) / Fref),
                     surface_term_rel=float(np.linalg.norm(Fsurf) / Fref),
                     mass_inside_identity_box=mass_out,
                     identity_agreement=float(
                         np.linalg.norm(Fl - Fid) / max(np.linalg.norm(Fl), 1e-300)),
                     solver_resid=float(r["resid"]))
    excess = max(law_rel - null_rel, 0.0)
    declared = cand.reciprocal_by_construction or bool(cand.momentum_carrier)
    ok = (excess < TOL["reciprocity"]) or declared
    return dict(passed=bool(ok), value=excess, tol=TOL["reciprocity"],
                law_net_force_rel=law_rel, newton_null_rel=null_rel,
                identity_net_force_rel=ident_rel,
                reciprocal_by_construction=cand.reciprocal_by_construction,
                momentum_carrier=cand.momentum_carrier or None,
                detail=("|F_net| / (G M1 M2 / d^2) for M1 = 4 M2 at 25 kpc, "
                        "minus the Newtonian null on the same grid"),
                **extra)


def _identity_force(K, Psi, box, pad=5):
    """The exact momentum identity for div[K(x) grad Psi] = 4 pi G rho.

    Integrating rho (-grad Psi) by parts over a region V gives

        F_i = - oint_dV T_ij n_j dS  -  (1/8 pi G) int_V (d_i K_jk) d_j Psi d_k Psi

    with the generalised Maxwell stress

        T_ij = (1/4 pi G) [ K_jk d_k Psi d_i Psi
                            - (1/2) delta_ij K_kl d_k Psi d_l Psi ].

    For constant K the volume term vanishes identically and the surface term is
    the ordinary Newtonian stress, so momentum is conserved.  For K = K(x) the
    volume term is generically nonzero and IS the third-law violation.  Both
    pieces are evaluated here on an inner sub-box, well away from the Dirichlet
    shell, so the identity can be checked against the direct force rather than
    asserted.
    """
    n = box.n
    lo, hi = pad, n - 1 - pad
    sl = slice(lo, hi + 1)
    Kg = K.reshape(box.shape + (3, 3))
    gv = np.stack(np.gradient(Psi, box.h, edge_order=2), -1)

    vol = np.zeros(3)
    for i in range(3):
        acc = 0.0
        for j in range(3):
            for k in range(3):
                dK = np.gradient(Kg[..., j, k], box.h, axis=i, edge_order=2)
                acc += float((dK * gv[..., j] * gv[..., k])[sl, sl, sl].sum())
        vol[i] = acc
    vol *= -box.vol / (8 * np.pi * G)

    Kgg = np.einsum("...jk,...j,...k->...", Kg, gv, gv)          # K_kl g_k g_l
    Kg_g = np.einsum("...jk,...k->...j", Kg, gv)                 # (K grad Psi)_j
    surf = np.zeros(3)
    h2 = box.h ** 2
    for ax in range(3):
        for side, idx in ((-1.0, lo), (+1.0, hi)):
            take = [sl, sl, sl]
            take[ax] = idx
            t = tuple(take)
            nj = np.zeros(3)
            nj[ax] = side
            for i in range(3):
                T = (Kg_g[t][..., ax] * gv[t][..., i]
                     - 0.5 * (1.0 if i == ax else 0.0) * Kgg[t]) / (4 * np.pi * G)
                surf[i] += side * float(T.sum()) * h2
    return -surf + vol, -surf, vol


# ================================================ S11/S12/S13 coarse graining
def _partition_scale(wx, wm):
    """Mean nearest-neighbour distance between wells: the partition scale."""
    if len(wx) < 2:
        return float("inf")
    d = np.sqrt(((wx[:, None, :] - wx[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def _steps(Ks):
    """Reference-free convergence test.

    step(N) = |K_N - K_prev| / |K_prev| over consecutive members of the
    refinement series.  A convergent (Cauchy) series has step -> 0; a law with
    no continuum limit -- one whose answer is a property of the row count --
    has step that stays finite or grows, and no choice of reference can hide
    that.  This is the primary criterion, because the 'continuum' reference is
    itself a finite cloud and would flatter a law that merely converges to the
    reference's own row count.
    """
    have = sorted(Ks)
    out = {}
    for a, b in zip(have[:-1], have[1:]):
        out[b] = float(np.abs(Ks[b] - Ks[a]).max()
                       / max(np.abs(Ks[a]).max(), 1e-300))
    return out


def _fit_beta(Ns, dr, nlast=5):
    up = [N for N in Ns if dr.get(N, 0) > 0]
    if len(up) < 3:
        return None
    up = up[-nlast:]
    x = np.log(np.array(up, float))
    y = np.log(np.array([dr[N] for N in up]))
    if np.ptp(x) <= 0:
        return None
    return float(-np.polyfit(x, y, 1)[0])


def _classify(drift, step, scales, L_kpc, beta_step, tol=None):
    """Four-way verdict on what a law's discreteness actually is.

    partition-independent  : the row list never matters at all
    coherence-limited      : the field stops moving once the partition scale
                             drops below the law's own length L, and STAYS
                             stopped -- a real physical scale
    convergent-quadrature  : successive refinements shrink as a power of N with
                             no plateau; the law has a continuum limit, but at
                             any finite catalogue resolution the answer is set
                             by how finely the mass happens to be tabulated
    catalogue-artefactual  : successive refinements do not shrink, or grow
    """
    tol = TOL["coarse"] if tol is None else tol
    have = sorted(drift)
    if max(drift.values()) < tol and max(step.values() or [0]) < tol:
        return "partition-independent", None
    sN = sorted(step)
    if not sN:
        return "catalogue-artefactual", None
    growing = step[sN[-1]] > step[sN[len(sN) // 2]] * 1.5
    # A fitted exponent is not enough on its own: a series that drifts by
    # 0.44 +- 0.05 across four decades of N can still return beta ~ 0.07 and be
    # called "convergent".  Require the successive-refinement STEP to have
    # actually fallen by at least a factor of three, which is what "converging"
    # has to mean if the word is to do any work.  The step series is used
    # rather than the drift because when the cloud reference is infeasible the
    # finest partition becomes the reference and its drift is zero by
    # construction, which would make any series look infinitely convergent.
    fell = step[sN[0]] / max(step[sN[-1]], 1e-300)
    if (growing or beta_step is None or beta_step < 0.05 or fell < 3.0):
        return "catalogue-artefactual", None
    # "the partition scale is below L" is not enough: a one-point quadrature of
    # a kernel of width L has relative error O((l/L)^2), so the drift only
    # becomes negligible a decade below L, not just below it.  The criterion is
    # therefore l_N < L/10.
    inside = [N for N in have if scales.get(N, np.inf) < L_kpc / 10.0]
    if inside:
        n0 = inside[0]
        after = max([drift[N] for N in have if N >= n0]
                    + [step[N] for N in sN if N >= n0])
        if after < tol:
            return "coherence-limited", n0
    n_safe = next((N for N in have if all(drift[M] < tol
                                          for M in have if M >= N)), None)
    return "convergent-quadrature", n_safe


def _Kseries(cand, parts, pts, Ns):
    out, scales, fails = {}, {}, {}
    for N in Ns:
        wx, wm = parts[N]
        scales[N] = _partition_scale(wx, wm)
        try:
            out[N] = K_at(cand, pts, wx, wm)
        except (MemoryError, F.SingularK) as e:
            fails[N] = f"{type(e).__name__}: {e}"
    return out, scales, fails


def s11_coarse_uniform(cand, cloud, pts, Ns=(1, 2, 4, 10, 40, 100, 400, 1000,
                                             4000, 10000)):
    """Hold the smooth density FIXED; vary only how it is cut into rows.

    The reference is the FULL cloud used as its own well list -- the mass
    distribution itself, not any member of the refinement series.  That matters:
    using the finest partition as the reference would put the same quantity on
    both sides of the comparison, which is the shared-denominator failure this
    programme has already been bitten by once.
    """
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, value=0.0, tol=TOL["coarse"],
                    detail=("the law is a local functional of rho and of "
                            "grad Phi; it never sees the row list, so partition "
                            "refinement is exact by construction"))
    qx, qm = cloud
    parts = F.nested_partitions(qx, qm, Ns)
    for N in Ns:
        F.check_partition(*parts[N], float(qm.sum()))
    Ks, scales, fails = _Kseries(cand, parts, pts, Ns)
    have = sorted(Ks)
    ref_kind = "full cloud (independent)"
    try:
        ref = K_at(cand, pts, qx, qm)
        n_ref = len(qm)
    except (MemoryError, F.SingularK):
        ref = Ks[have[-1]]
        n_ref = have[-1]
        ref_kind = "finest feasible partition (cloud reference infeasible)"
    nref = max(np.abs(ref).max(), 1e-300)
    drift = {N: float(np.abs(Ks[N] - ref).max() / nref) for N in have}
    step = _steps(Ks)
    d0 = drift[have[0]]
    tail = float(abs(drift[have[-2]] - drift[have[-1]])) if len(have) > 1 else 0.0
    beta = _fit_beta(have, drift)
    beta_step = _fit_beta(sorted(step), step)
    cls, n_safe = _classify(drift, step, {k: scales[k] / KPC for k in have},
                            cand.prm.get("L", 10 * KPC) / KPC, beta_step)
    ok = cls in ("partition-independent", "coherence-limited",
                 "convergent-quadrature")
    return dict(passed=bool(ok), value=d0, tol=TOL["coarse"],
                change_1_to_max=d0, change_at_max=drift[have[-1]],
                drift={str(k): v for k, v in drift.items()},
                step={str(k): v for k, v in step.items()},
                partition_scale_kpc={str(k): scales[k] / KPC for k in have},
                tail_fraction=tail / max(d0, 1e-300), rate_beta=beta,
                rate_beta_step=beta_step,
                classification=cls, N_safe=n_safe,
                reference=ref_kind, N_reference=n_ref,
                infeasible={str(k): v for k, v in fails.items()},
                detail=("|K_N - K_ref| / |K_ref| with rho held FIXED and only "
                        "the row list refined; 'step' is the reference-free "
                        "successive difference |K_N - K_prev|/|K_prev|, and "
                        "beta is the fitted exponent in ~ N^-beta"))


def s11b_coarse_potential(cand, n=48, Lbox=90.0, Ns=(1, 10, 100, 10000),
                          M=5e10 * MSUN, a=3 * KPC, Nq=32768):
    """The deliverable numbers: fractional change in Phi and in v_c between
    N = 1 and N = 10^4, from a full 3-D solve.

    The SOURCE density on the right-hand side is identical in every run.  Only
    the well list that feeds K changes.  Any difference is therefore caused by
    the partition and nothing else.
    """
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, value=0.0, tol=TOL["coarse"],
                    detail="local functional of rho: no row list enters")
    box = F.Box(n, Lbox)
    rho = F.normalise_mass(
        F.plummer_rho(box.pts, M, a).reshape(box.shape), box.vol, M)
    qx, qm = F.equal_mass_cloud("plummer", Nq, M, a=a, umax=0.98)
    parts = F.nested_partitions(qx, qm, Ns)
    R = np.geomspace(3 * KPC, 30 * KPC, 12)
    msk = (box.r > 2 * KPC) & (box.r < 0.35 * box.L)
    Phi, Vc, fails = {}, {}, {}
    for N in Ns:
        wx, wm = parts[N]
        F.check_partition(wx, wm, M)
        try:
            r = solve_candidate(cand, rho, box, wx, wm, M)
            assert r["resid"] < 1e-8, f"CG stalled at {r['resid']:.2e}"
            Phi[N] = r["Psi"]
            Vc[N] = FS.vcirc_axis(r["Psi"], box, R)
        except (MemoryError, AssertionError, F.SingularK) as e:
            fails[N] = f"{type(e).__name__}: {e}"
    have = sorted(Phi)
    if len(have) < 2:
        return dict(passed=False, value=None,
                    infeasible={str(k): v for k, v in fails.items()},
                    detail="fewer than two feasible partitions")
    lo, hi = have[0], have[-1]
    dphi = float(np.abs(Phi[lo][msk] - Phi[hi][msk]).max()
                 / np.abs(Phi[hi][msk]).max())
    dphi_rms = float(np.sqrt(np.mean((Phi[lo][msk] - Phi[hi][msk]) ** 2))
                     / np.sqrt(np.mean(Phi[hi][msk] ** 2)))
    dv = float(np.abs(Vc[lo] / np.maximum(Vc[hi], 1e-30) - 1).max())
    series = {str(N): dict(
        dPhi_vs_max=float(np.abs(Phi[N][msk] - Phi[hi][msk]).max()
                          / np.abs(Phi[hi][msk]).max()),
        dvc_vs_max=float(np.abs(Vc[N] / np.maximum(Vc[hi], 1e-30) - 1).max()),
        vc_kms=[float(x / 1e3) for x in Vc[N]]) for N in have}
    return dict(passed=bool(dphi < TOL["coarse"]), value=dphi,
                tol=TOL["coarse"],
                dPhi_1_to_max=dphi, dPhi_rms_1_to_max=dphi_rms,
                dvc_1_to_max=dv, N_lo=lo, N_hi=hi,
                radii_kpc=[float(x / KPC) for x in R], series=series,
                infeasible={str(k): v for k, v in fails.items()},
                detail=("full 3-D solve, same rho every time, only the row "
                        "list changes; dPhi is max|Phi_1 - Phi_Nmax|/max|Phi| "
                        "over 2 kpc < r < 0.35 L"))


def s12_coarse_selective(cand, pts, Ns=(1, 4, 16, 64, 256, 1024, 4096)):
    """Refine ONE object and leave its neighbour alone.

    This is the test with teeth.  Under uniform refinement a mass-weight
    exponent p != 1 partly cancels between numerator and denominator; under
    SELECTIVE refinement it does not, because the relative weight of the
    refined object against the unrefined one moves as N^(1-p).
    """
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, value=0.0, tol=TOL["coarse"],
                    detail="no row list enters the law")
    M1, M2 = 6e10 * MSUN, 6e10 * MSUN
    sep = 40 * KPC
    c1 = np.array([-sep / 2, 0, 0.])
    c2 = np.array([+sep / 2, 0, 0.])
    q1 = F.equal_mass_cloud("plummer", 16384, M1, a=3 * KPC, umax=0.95,
                            centre=c1)
    parts = F.nested_partitions(q1[0], q1[1], Ns)
    probe = c2 + F._fib_dirs(48) * (8 * KPC)
    probe = np.vstack([probe, pts[:100]])
    res, fails = {}, {}
    for N in Ns:
        wx = np.vstack([parts[N][0], c2[None]])
        wm = np.concatenate([parts[N][1], [M2]])
        F.check_partition(wx, wm, M1 + M2)
        try:
            res[N] = K_at(cand, probe, wx, wm)
        except (MemoryError, F.SingularK) as e:
            fails[N] = f"{type(e).__name__}: {e}"
    have = sorted(res)
    ref_kind = "full cloud (independent)"
    try:
        ref = K_at(cand, probe, np.vstack([q1[0], c2[None]]),
                   np.concatenate([q1[1], [M2]]))
    except (MemoryError, F.SingularK):
        ref = res[have[-1]]
        ref_kind = "finest feasible partition"
    nref = max(np.abs(ref).max(), 1e-300)
    drift = {N: float(np.abs(res[N] - ref).max() / nref) for N in have}
    step = _steps(res)
    d0 = drift[have[0]]
    tail = abs(drift[have[-2]] - drift[have[-1]]) if len(have) > 1 else 0.0
    beta = _fit_beta(have, drift)
    beta_step = _fit_beta(sorted(step), step)
    sN = sorted(step)
    growing = bool(sN and step[sN[-1]] > step[sN[len(sN) // 2]] * 1.5)
    # the analytic prediction: relative weight of the refined object against
    # the unrefined one moves as N^(1-p).  Measured directly, not asserted.
    wratio, wslope = {}, None
    if cand.kind in ("wells", "wells_linear"):
        x0 = c2 + np.array([0.0, 8 * KPC, 0.0])
        # the g_N factor of weight form 3 depends only on the FIELD point, so
        # it is common to both objects and cancels in this ratio exactly; the
        # ratio is therefore computed with the shape factor alone.
        prm_r = dict(cand.prm)
        if prm_r.get("shape") == "pow_g":
            prm_r["shape"] = "pow"
        for N in have:
            wx1 = parts[N][0]
            wm1 = parts[N][1]
            r1 = np.maximum(np.linalg.norm(wx1 - x0, axis=1),
                            cand.prm["r_soft"])
            w1 = F.weight_C(wm1, r1, None, prm_r).sum()
            r2 = np.maximum(np.linalg.norm(c2 - x0), cand.prm["r_soft"])
            w2 = F.weight_C(np.array([M2]), np.array([r2]), None, prm_r)[0]
            wratio[N] = float(w1 / w2)
        if len(have) >= 3:
            wslope = float(np.polyfit(np.log(np.array(have, float)),
                                      np.log([wratio[N] for N in have]), 1)[0])
    ok = (d0 < TOL["coarse"]) or (
        not growing and beta_step is not None and beta_step > 0.05
        and (wslope is None or abs(wslope) < 0.05))
    return dict(passed=bool(ok), value=d0, tol=TOL["coarse"],
                drift={str(k): v for k, v in drift.items()},
                step={str(k): v for k, v in step.items()},
                tail_fraction=float(tail / max(d0, 1e-300)), rate_beta=beta,
                rate_beta_step=beta_step,
                growing=growing, reference=ref_kind,
                weight_ratio_obj1_obj2={str(k): v for k, v in wratio.items()},
                weight_ratio_slope=wslope,
                weight_ratio_slope_predicted=(1.0 - cand.prm["p"])
                if cand.kind in ("wells", "wells_linear") else None,
                infeasible={str(k): v for k, v in fails.items()},
                detail=("two equal 6e10 Msun objects 40 kpc apart; object 1 is "
                        "split into N rows, object 2 is never split. K is read "
                        "8 kpc from object 2, where nothing physical changed. "
                        "A mass exponent p != 1 moves the relative weight of "
                        "the two objects as N^(1-p); uniform refinement partly "
                        "hides that and this test does not"))


def s13_coherence(cand, s11, s12=None):
    """Is the discreteness a physical coherence length, or catalogue rows?"""
    if cand.kind not in TENSOR_KINDS:
        return dict(passed=True, classification="not applicable",
                    detail="no discrete sum in the law")
    dr = {int(k): v for k, v in s11.get("drift", {}).items()}
    sc = {int(k): v for k, v in s11.get("partition_scale_kpc", {}).items()}
    if not dr:
        return dict(passed=False, classification="undetermined",
                    detail="no drift series")
    L = cand.prm.get("L", cand.prm.get("sigma_perp", 10 * KPC)) / KPC
    have = sorted(dr)
    inside = [N for N in have if sc.get(N, 1e9) < L]
    if inside:
        post = max(dr[N] for N in have if N >= inside[0])
        rel = post / max(dr[have[0]], 1e-300)
        n_in = inside[0]
    else:
        post, rel, n_in = float("nan"), float("nan"), None
    cls = s11.get("classification", "undetermined")
    sel = (s12 or {}).get("classification_hint")
    if s12 is not None and s12.get("growing"):
        cls = "catalogue-artefactual"
    verdict = dict(
        **{"partition-independent": dict(physical=True,
                                         reading="the row list never enters"),
           "coherence-limited": dict(
               physical=True,
               reading="a real coherence length generated by the law: the "
                       "drift stops once the partition is finer than L"),
           "convergent-quadrature": dict(
               physical=False,
               reading="no plateau at L. The discrete sum is a one-point "
                       "quadrature of a continuum integral; it has a limit, "
                       "but at any finite catalogue resolution the answer is "
                       "set by how finely the mass happens to be tabulated"),
           "catalogue-artefactual": dict(
               physical=False,
               reading="the field keeps moving as rows are added; the "
                       "prediction is a property of the catalogue")}
        .get(cls, dict(physical=False, reading="undetermined")))
    return dict(passed=cls in ("partition-independent", "coherence-limited"),
                classification=cls, coherence_L_kpc=float(L),
                rate_beta=s11.get("rate_beta"),
                N_first_inside_L=n_in,
                drift_after_partition_finer_than_L=float(post),
                fraction_of_total_drift_after_L=float(rel),
                N_safe_for_1e_3=s11.get("N_safe"),
                selective_hint=sel,
                partition_scale_kpc={str(k): v for k, v in sc.items()},
                detail=("a law with a real coherence length L converges once "
                        "the partition scale drops below L and then stops "
                        "moving; a law that merely has a continuum limit keeps "
                        "improving as N^-beta with no plateau; a law that "
                        "counts rows keeps drifting or grows"),
                **verdict)


# =============================================================== the driver
def run_screen(cand, Nq=32768, nprobe=(10, 20), verbose=False, skip=()):
    """Run every Stage-1 screen on one candidate spec."""
    t0 = time.time()
    qx, qm = galaxy_cloud(Nq=Nq)
    parts0 = F.nested_partitions(qx, qm, [100])
    wx, wm = parts0[100]
    pts = probe_points(*nprobe)
    out = dict(candidate=cand.name, family=cand.family, kind=cand.kind,
               note=cand.note,
               params={k: (v if isinstance(v, str) else float(v))
                       for k, v in cand.prm.items()},
               screens={})

    def run(key, fn):
        if key in skip:
            out["screens"][key] = dict(passed=None, detail="skipped")
            return out["screens"][key]
        try:
            r = fn()
        except Exception as e:                       # noqa: BLE001
            r = dict(passed=False, value=None, error=f"{type(e).__name__}: {e}",
                     trace=traceback.format_exc(limit=3))
        out["screens"][key] = r
        if verbose:
            print(f"   {key:26s} {'PASS' if r.get('passed') else 'FAIL'} "
                  f"{r.get('value')}")
        return r

    run("S1_dimensions", lambda: s1_dimensions(cand))
    run("S2_rotation", lambda: s2_rotation(cand, wx, wm, pts))
    run("S2b_grid_rotation", lambda: s2b_grid_rotation(cand))
    run("S3_translation", lambda: s3_translation(cand, wx, wm, pts))
    run("S4_gauge_offset", lambda: s4_gauge(cand, wx, wm, pts))
    run("S5_positive_definite", lambda: s5_positive_definite(cand, wx, wm, pts))
    run("S6_newtonian_limit", lambda: s6_newtonian_limit(cand, wx, wm))
    run("S7_asymptotics", lambda: s7_asymptotics(cand, wx, wm))
    run("S8_gain_bound", lambda: s8_gain_bound(cand, wx, wm))
    run("S9_permutation", lambda: s9_permutation(cand, wx, wm, pts))
    run("S10_reciprocity", lambda: s10_reciprocity(cand))
    Ns = (1, 2, 4, 10, 40, 100, 400, 1000, 4000, 10000)
    if cand.kind in ("pairs", "pairs_linear"):
        Ns = (1, 2, 4, 10, 40, 100, 400, 1000)
    s11 = run("S11_coarse_uniform",
              lambda: s11_coarse_uniform(cand, (qx, qm), pts, Ns=Ns))
    NsP = (1, 10, 100, 10000)
    if cand.kind in ("pairs", "pairs_linear"):
        # family D costs O(P N^2) on a 48^3 grid; 256 rows is already
        # 32,640 pair evaluations per cell and is the practical ceiling
        NsP = (1, 10, 100, 256)
    run("S11b_coarse_potential",
        lambda: s11b_coarse_potential(cand, Ns=NsP))
    Ns2 = (1, 4, 16, 64, 256, 1024, 4096)
    if cand.kind in ("pairs", "pairs_linear"):
        Ns2 = (1, 4, 16, 64, 256)
    s12 = run("S12_coarse_selective",
              lambda: s12_coarse_selective(cand, pts, Ns=Ns2))
    run("S13_coherence", lambda: s13_coherence(
        cand, out["screens"].get("S11_coarse_uniform", {}),
        out["screens"].get("S12_coarse_selective", {})))

    failed = [k for k, v in out["screens"].items() if v.get("passed") is False]
    out["failed"] = failed
    out["verdict"] = "PASS" if not failed else "FAIL"
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ------------------------------------------------------ sensitivity harness
def sensitivity(cand, key, values, statistic, Nq=16384):
    """Verify numerically that a headline statistic actually MOVES with the
    parameter it is supposed to measure.

    The programme has been bitten by a rank statistic that was bit-identical
    across three decades of its own parameter; every headline number here gets
    this check and the spread is printed.
    """
    qx, qm = galaxy_cloud(Nq=Nq)
    pts = probe_points(8, 16)
    parts = F.nested_partitions(qx, qm, [100])
    wx, wm = parts[100]
    vals = []
    for v in values:
        c2 = cand.copy_with(prm={key: v})
        vals.append(float(statistic(c2, wx, wm, pts, (qx, qm))))
    vals = np.array(vals)
    spread = float(np.nanmax(vals) - np.nanmin(vals))
    rel = spread / max(abs(np.nanmedian(vals)), 1e-300)
    return dict(param=key, values=[float(x) for x in values],
                statistic=[float(x) for x in vals],
                spread=spread, relative_spread=rel,
                responds=bool(spread > 0))


# ------------------------------------------------------- targeted analyses
def analysis_C4_gN_cancellation(Nq=16384):
    """Does the acceleration suppression in family C's third weight form do
    anything at all?

    w_a = (M_a/M_0)^p / {[1 + (g_N/a0)^m][1 + (r_a/L)^q]^s}

    g_N carries no index a, so it is the LOCAL Newtonian acceleration at the
    field point.  It is therefore a factor common to every term, and S is a
    ratio of two sums over a -- so it cancels exactly, up to the regulator eps.
    The suppression the form was written to provide is not there.  Both
    readings are measured below.
    """
    qx, qm = galaxy_cloud(Nq=Nq)
    wx, wm = F.nested_partitions(qx, qm, [200])[200]
    pts = probe_points(10, 20)
    c_g = F.CANDIDATES["C4_wells_gsupp_p1"]
    c_0 = c_g.copy_with(prm={"shape": "pow"})
    Sg = F.S_wells(pts, wx, wm, c_g.prm, gN=gN_analytic(pts, wx, wm))
    S0 = F.S_wells(pts, wx, wm, c_0.prm)
    cancel = float(np.abs(Sg - S0).max() / max(np.abs(S0).max(), 1e-300))
    # alternative reading: g_N evaluated AT each well
    prm = dict(c_g.prm)
    gw = gN_analytic(wx + 0.3 * KPC, wx, wm)          # offset avoids self-pole
    d = wx[None, :, :] - pts[:, None, :]
    r = np.maximum(np.sqrt((d * d).sum(-1)), prm["r_soft"])
    n = d / r[..., None]
    u = (wm[None, :] / prm["M0"]) ** prm["p"]
    w = u / ((1 + (gw[None, :] / prm["a0"]) ** prm["m"])
             * (1 + (r / prm["L"]) ** prm["q"]) ** prm["s"])
    num = np.einsum("pb,pbi,pbj->pij", w, n, n)
    Sa = (num - w.sum(1)[:, None, None] * np.eye(3)[None] / 3.0) \
        / (prm["eps"] + np.abs(w).sum(1))[:, None, None]
    per_well = float(np.abs(Sa - S0).max() / max(np.abs(S0).max(), 1e-300))
    # how much does eps have to be raised before the g factor bites?
    sweep = {}
    for e in (1e-12, 1e-6, 1e-3, 1e-1, 1.0):
        p2 = dict(c_g.prm)
        p2["eps"] = e
        Sg2 = F.S_wells(pts, wx, wm, p2, gN=gN_analytic(pts, wx, wm))
        p3 = dict(c_0.prm)
        p3["eps"] = e
        S02 = F.S_wells(pts, wx, wm, p3)
        sweep[str(e)] = float(np.abs(Sg2 - S02).max()
                              / max(np.abs(S02).max(), 1e-300))
    return dict(local_gN_reading_effect=cancel,
                per_well_gN_reading_effect=per_well,
                effect_vs_eps=sweep,
                verdict=("the local-g_N reading is exactly degenerate with a "
                         "rescaling of eps and changes S by "
                         f"{cancel:.2e}; only the per-well reading has any "
                         f"effect, and that changes S by {per_well:.3g}"))


def analysis_D_scaling(Ns=(4, 16, 64, 256, 1024, 4096),
                       names=("D1_pairs_p1_q1", "D2_pairs_p05_q1",
                              "D3_pairs_p1_q3"), Nq=32768):
    """Measure how family D's pair tensor scales with the number of rows.

    Analytic expectation for N equal-mass rows of one object: the double sum
    contributes (M/N)^{2p} * (N^2/2) * <(d/L)^-q>, and the pair-separation
    distribution goes as d^2 dd at small d, so <d^-q> converges for q < 3.

        ||C|| ~ N^(2 - 2p)                 for q < 3
        ||C|| ~ N^(2 - 2p) log N           for q = 3
        ||C|| ~ N^(2 - 2p + (q-3)/3)       for q > 3

    p = 1 with q < 3 is the only combination that has a finite limit.
    """
    qx, qm = galaxy_cloud(Nq=Nq)
    parts = F.nested_partitions(qx, qm, Ns)
    x0 = np.array([[4.0, 1.0, 0.5]]) * KPC
    out = {}
    for nm in names:
        c = F.CANDIDATES[nm]
        nrm, npair = {}, {}
        for N in Ns:
            wx, wm = parts[N]
            C = F.C_pairs(x0, wx, wm, c.prm)
            nrm[N] = float(np.sqrt((C * C).sum()))
            npair[N] = N * (N - 1) // 2
        x = np.log(np.array(Ns, float))
        y = np.log(np.array([max(nrm[N], 1e-300) for N in Ns]))
        local = list(np.diff(y) / np.diff(x))
        slope = float(local[-1])
        p, q = c.prm["p"], c.prm["q"]
        pred = 2 - 2 * p + (max(q - 3.0, 0.0)) / 3.0
        # for q = 3 the pair integral is log divergent: ||C|| ~ A + B ln N,
        # whose local log-slope is B / (A + B ln N) -> 0 only as 1/ln N
        logfit = None
        if abs(q - 3.0) < 1e-9:
            A, B = np.polyfit(np.log(np.array(Ns[-4:], float)),
                              [nrm[N] for N in Ns[-4:]], 1)[::-1]
            pr = A / (A + 0.0)
            logfit = dict(A=float(B), B=float(A),
                          predicted_local_slope_at_Nmax=float(
                              A / (B + A * np.log(Ns[-1]))),
                          note="fit ||C|| = A + B ln N over the last four N")
        out[nm] = dict(p=p, q=q, norms={str(k): v for k, v in nrm.items()},
                       pairs={str(k): v for k, v in npair.items()},
                       local_log_slopes=[float(v) for v in local],
                       measured_slope_at_Nmax=slope, predicted_slope=pred,
                       converges=bool(abs(slope) < 0.05),
                       log_divergence_fit=logfit,
                       pairs_at_1e6_rows=int(1e6 * (1e6 - 1) // 2))
    return out


# ================================================== the funnel front proper
# Screening fifteen hand-written specs is a demonstration, not a funnel.  The
# sweeps below run the DECISIVE Stage-1 conditions over the full exponent and
# coupling grids of families C and D.  Most of those conditions turn out to be
# closed form once the eigenvalues of S (or C) are known at three probe sets,
# and the eigenvalues do not depend on the couplings s_0, s_T (or alpha) at
# all.  That factorisation is what makes the front cheap:
#
#     K = exp[s_0 I + s_T S]   =>   lambda_i(K) = exp(s_0 + s_T lambda_i(S))
#
# so |K - I|_2 = max_i |exp(s_0 + s_T lambda_i) - 1| is monotone in lambda and
# is attained at lambda_min or lambda_max.  Six numbers per weight-parameter
# setting therefore decide the whole (s_0, s_T) plane exactly.

def _lam_extremes(Sarr):
    ev = np.linalg.eigvalsh(Sarr)
    return float(ev.min()), float(ev.max())


def _probe_sets(wx, wm):
    hi, gN = _high_accel_points(wx, wm, n=96)
    far = F._fib_dirs(96) * (3.0e3 * KPC)
    mid = probe_points(8, 16)
    return hi, gN, far, mid


def sweep_C(ps=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            qs=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
            ss=(0.5, 1.0, 1.5, 2.0, 3.0),
            Ls=(1.0, 3.0, 10.0, 30.0, 100.0),
            shapes=("pow", "exp", "pow_g"),
            n_s0=201, n_sT=201, s0_rng=(-2.0, 2.0), sT_rng=(-2.0, 2.0),
            Nq=8192, nwells=200):
    """Exact Stage-1 verdicts over the whole family-C grid."""
    t0 = time.time()
    qx, qm = galaxy_cloud(Nq=Nq)
    wx, wm = F.nested_partitions(qx, qm, [nwells])[nwells]
    hi, gN_hi, far, mid = _probe_sets(wx, wm)
    base = F.CANDIDATES["C1_wells_pow_p1"].prm
    rows = []
    for shape in shapes:
        for p in ps:
            for q in qs:
                for s in ss:
                    for L in Ls:
                        prm = dict(base, shape=shape, p=p, q=q, s=s,
                                   L=L * KPC)
                        g = gN_hi if shape == "pow_g" else None
                        a, b = _lam_extremes(F.S_wells(hi, wx, wm, prm, gN=g))
                        gf = (gN_analytic(far, wx, wm)
                              if shape == "pow_g" else None)
                        c, d = _lam_extremes(F.S_wells(far, wx, wm, prm, gN=gf))
                        gm = (gN_analytic(mid, wx, wm)
                              if shape == "pow_g" else None)
                        e, f = _lam_extremes(F.S_wells(mid, wx, wm, prm, gN=gm))
                        rows.append((shape, p, q, s, L, a, b, c, d, e, f))
    t_eig = time.time() - t0
    s0 = np.linspace(*s0_rng, n_s0)[:, None]
    sT = np.linspace(*sT_rng, n_sT)[None, :]
    G = n_s0 * n_sT
    counts = dict(total=0, S6_newton=0, S10_reciprocity=0, S12_selective=0,
                  S8_unbounded_boost=0, all_of_them=0)
    lam_abs_max = max(max(abs(r[5]), abs(r[6]), abs(r[7]), abs(r[8]),
                          abs(r[9]), abs(r[10])) for r in rows)
    for (shape, p, q, s, L, a, b, c, d, e, f) in rows:
        # S6: ||K - I||_2 at high acceleration, exact via the extreme eigenvalues
        dev = np.maximum(np.abs(np.exp(s0 + sT * a) - 1.0),
                         np.abs(np.exp(s0 + sT * b) - 1.0))
        ok6 = dev < TOL["newton_limit"]
        # S10: K must be homogeneous or momentum is not conserved
        spread = np.abs(np.exp(s0 + sT * f) - np.exp(s0 + sT * e))
        ok10 = spread / np.exp(s0) < TOL["reciprocity"]
        # S12: the selective-refinement weight slope is exactly 1 - p
        ok12 = np.full_like(dev, abs(1.0 - p) < 0.05, dtype=bool)
        # S8: the boost at large r saturates at exp(-(s0 + sT lam_min_far))
        ok8 = np.zeros_like(ok6)          # bounded K: never unbounded
        counts["total"] += G
        counts["S6_newton"] += int(ok6.sum())
        counts["S10_reciprocity"] += int(ok10.sum())
        counts["S12_selective"] += int(ok12.sum())
        counts["S8_unbounded_boost"] += int(ok8.sum())
        counts["all_of_them"] += int((ok6 & ok10 & ok12).sum())
    dt = time.time() - t0
    return dict(family="C", settings=counts["total"],
                weight_parameter_settings=len(rows),
                coupling_grid=G, counts=counts,
                survivors=counts["all_of_them"],
                survivor_fraction=counts["all_of_them"] / max(counts["total"], 1),
                seconds=round(dt, 2), seconds_eigen=round(t_eig, 2),
                rate_per_second=round(counts["total"] / max(dt, 1e-9)),
                max_abs_eigenvalue_of_S=float(lam_abs_max),
                bound_on_S="|S|_2 < 2/3 = 0.6667 for every weight, by "
                           "construction: |n n^T - I/3|_2 = 2/3 and S is a "
                           "normalised weighted average of such matrices",
                note=("survivors are settings that pass S6, S10 and S12 "
                      "simultaneously; the only ones are the trivial "
                      "s_T = 0, s_0 = 0 line, which is Newton"))


def sweep_D(ps=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            qs=(0.5, 1.0, 2.0, 3.0, 4.0),
            ss=(1.0, 2.0, 3.0),
            Ls=(3.0, 10.0, 30.0),
            sperp=(1.0, 2.0, 5.0), spar=(2.0, 5.0, 15.0),
            n_alpha=4001, alpha_rng=(1e-4, 1e2),
            Nq=8192, nwells=120):
    """Exact Stage-1 verdicts over the whole family-D grid."""
    t0 = time.time()
    qx, qm = galaxy_cloud(Nq=Nq)
    wx, wm = F.nested_partitions(qx, qm, [nwells])[nwells]
    hi, _, far, mid = _probe_sets(wx, wm)
    base = F.CANDIDATES["D1_pairs_p1_q1"].prm
    rows = []
    for p in ps:
        for q in qs:
            for s in ss:
                for L in Ls:
                    for sp in sperp:
                        for sl in spar:
                            prm = dict(base, p=p, q=q, s=s, L=L * KPC,
                                       sigma_perp=sp * KPC,
                                       sigma_par=sl * KPC)
                            a, b = _lam_extremes(F.C_pairs(hi, wx, wm, prm))
                            e, f = _lam_extremes(F.C_pairs(mid, wx, wm, prm))
                            rows.append((p, q, s, L, sp, sl, a, b, e, f))
    t_eig = time.time() - t0
    al = np.geomspace(*alpha_rng, n_alpha)
    counts = dict(total=0, S6_newton=0, S10_reciprocity=0,
                  S11_pair_convergence=0, all_of_them=0)
    for (p, q, s, L, sp, sl, a, b, e, f) in rows:
        dev = np.maximum(np.abs(np.exp(-al * a) - 1.0),
                         np.abs(np.exp(-al * b) - 1.0))
        ok6 = dev < TOL["newton_limit"]
        spread = np.abs(np.exp(-al * f) - np.exp(-al * e))
        ok10 = spread < TOL["reciprocity"]
        # ||C|| ~ N^(2-2p) for q < 3, log divergent at q = 3, power divergent
        # above: only p = 1 with q < 3 has a partition-independent limit
        ok11 = np.full_like(dev, (abs(2 - 2 * p) < 0.05) and q < 3.0,
                            dtype=bool)
        counts["total"] += n_alpha
        counts["S6_newton"] += int(ok6.sum())
        counts["S10_reciprocity"] += int(ok10.sum())
        counts["S11_pair_convergence"] += int(ok11.sum())
        counts["all_of_them"] += int((ok6 & ok10 & ok11).sum())
    dt = time.time() - t0
    return dict(family="D", settings=counts["total"],
                weight_parameter_settings=len(rows),
                coupling_grid=n_alpha, counts=counts,
                survivors=counts["all_of_them"],
                survivor_fraction=counts["all_of_them"] / max(counts["total"], 1),
                seconds=round(dt, 2), seconds_eigen=round(t_eig, 2),
                rate_per_second=round(counts["total"] / max(dt, 1e-9)))


def analysis_coherence_scaling(targets=(("C1_wells_pow_p1", "L"),
                                        ("C3_wells_exp_p1", "L"),
                                        ("C5_wells_pow_p2", "L"),
                                        ("D1_pairs_p1_q1", "sigma_perp"),
                                        ("E1_tidal", None),
                                        ("X4_smooth_density", "L"),
                                        ("X2_count_wells", "L")),
                               Ls_kpc=(2.0, 4.0, 8.0, 16.0, 32.0),
                               Nfix=(100, 1000), Nq=16384):
    """The decisive test of whether a law's discreteness is physical.

    Both a genuine coherence-length law and a row-counting law can converge
    under refinement, so "does it converge?" does not separate them.  What
    separates them is WHAT SETS the catalogue resolution you need:

      * a law with a real coherence length L is a quadrature of a kernel of
        width L, so its error is a function of l_N / L and the drift at fixed N
        falls steeply as L is increased.  d ln(drift)/d ln L is around -2.
      * a law whose discreteness comes from the row list has structure on the
        scale of the distance to the nearest row, which no parameter controls,
        so the drift at fixed N barely moves with L.  The slope is near zero.

    Family C's directional factor (n_a n_a^T - I/3) turns over when the field
    point moves by of order its distance to well a.  That distance is set by
    the partition, not by L, which is why L does not buy accuracy.
    """
    qx, qm = galaxy_cloud(Nq=Nq)
    pts = probe_points(8, 16)
    N_ref_fallback = 4000
    parts = F.nested_partitions(qx, qm, sorted(set(Nfix) | {N_ref_fallback}))
    out = {}
    for nm, param in targets:
        c0 = ALL[nm]
        rows = {}
        ref_kind = "full cloud"
        grid = Ls_kpc if param else (0.0,)
        for L in grid:
            c = c0 if param is None else c0.copy_with(prm={param: L * KPC})
            try:
                try:
                    ref = K_at(c, pts, qx, qm)
                except MemoryError:
                    # family D cannot afford the full cloud as a reference
                    # (16,384 rows is 1.3e8 pairs); the finest affordable
                    # partition stands in and the substitution is recorded
                    ref = K_at(c, pts, *parts[N_ref_fallback])
                    ref_kind = f"partition with {N_ref_fallback} rows"
                nref = max(np.abs(ref).max(), 1e-300)
                rows[L] = {str(N): float(np.abs(K_at(c, pts, *parts[N]) - ref).max()
                                         / nref) for N in Nfix}
            except (MemoryError, F.SingularK) as e:
                rows[L] = {"error": f"{type(e).__name__}: {e}"}
        slopes = {}
        for N in Nfix:
            xs, ys = [], []
            for L in grid:
                v = rows[L].get(str(N))
                if param and v and v > 0:
                    xs.append(np.log(L))
                    ys.append(np.log(v))
            slopes[str(N)] = (float(np.polyfit(xs, ys, 1)[0])
                              if len(xs) >= 3 else None)
        sl = [v for v in slopes.values() if v is not None]
        mean_slope = float(np.mean(sl)) if sl else None
        if mean_slope is None and param is None:
            verdict = ("no free spatial scale to sweep: the law has no length "
                       "parameter of its own, it reads the field of the rows")
        elif mean_slope is None:
            verdict = "sweep failed; see drift_by_scale for the reason"
        elif mean_slope <= -2.0:
            verdict = ("drift is set by the law's own length: a wider kernel "
                       "buys accuracy at fixed catalogue resolution")
        elif mean_slope <= -1.0:
            verdict = ("drift is only partly set by the law's length; the "
                       "directional factor still has no scale of its own")
        else:
            verdict = "drift is set by the row list, not by the law's length"
        out[nm] = dict(
            parameter=param, scale_kpc=list(grid), drift_by_scale=rows,
            reference=ref_kind,
            slope_by_N=slopes, mean_slope=mean_slope,
            partition_scale_kpc={str(N): _partition_scale(*parts[N]) / KPC
                                 for N in Nfix},
            verdict=verdict)
    return out


def analysis_D_collapse(Ns=(10, 25, 50, 100, 200, 400, 800),
                        names=("D1_pairs_p1_q1", "D2_pairs_p05_q1",
                               "D3_pairs_p1_q3"), Nq=32768):
    """What the pair tensor's growth does to K itself.

    K = exp[-alpha C], so ||C|| ~ N^(2-2p) means ln lambda_min(K) falls LINEARLY
    in N^(2-2p).  At p = 1/2 that is exponential collapse of the response
    tensor with catalogue resolution: the same galaxy described by more rows
    has a smaller and smaller K, hence a larger and larger effective G, without
    limit.  This is not a law that converges slowly; it is a law with no
    continuum limit.
    """
    qx, qm = galaxy_cloud(Nq=Nq)
    parts = F.nested_partitions(qx, qm, Ns)
    pts = probe_points(6, 12)
    out = {}
    for nm in names:
        c = F.CANDIDATES[nm]
        lmin, lmax = {}, {}
        for N in Ns:
            wx, wm = parts[N]
            try:
                C = F.C_pairs(pts, wx, wm, c.prm)
                ev = np.linalg.eigvalsh(F.K_from_C(C, c.prm))
                lmin[N], lmax[N] = float(ev.min()), float(ev.max())
            except (MemoryError, F.SingularK) as e:
                lmin[N] = None
                lmax[N] = f"{type(e).__name__}: {e}"
        have = [N for N in Ns if lmin.get(N)]
        slope = None
        if len(have) >= 3:
            y = np.log(np.array([lmin[N] for N in have]))
            slope = float(np.polyfit(np.array(have, float), y, 1)[0])
        out[nm] = dict(p=c.prm["p"], q=c.prm["q"],
                       lambda_min={str(k): v for k, v in lmin.items()},
                       lambda_max={str(k): v for k, v in lmax.items()},
                       d_ln_lambda_min_dN=slope,
                       collapses=bool(slope is not None and slope < -1e-3))
    return out


def analysis_momentum_identity(name="C1_wells_pow_p1", ns=(32, 40, 52, 64),
                               Lbox=120.0):
    """Show that the third-law violation is the continuum identity, not the grid.

    If the measured net force were a discretisation artefact it would shrink
    with h.  It does not: it converges to a fixed fraction of the pair force,
    while the gap between the DIRECT force and the grad-K identity shrinks with
    h, which is exactly what an O(h^2) by-parts error does.
    """
    c = ALL[name]
    rows = []
    for n in ns:
        r = s10_reciprocity(c, n=n, Lbox=Lbox)
        rows.append(dict(n=n, h_kpc=Lbox / n,
                         direct=r["law_net_force_rel"],
                         identity=r["identity_net_force_rel"],
                         gradK_term=r["gradK_term_rel"],
                         surface_term=r["surface_term_rel"],
                         newton_null=r["newton_null_rel"],
                         agreement=r["identity_agreement"]))
    a = np.array([x["agreement"] for x in rows])
    h = np.array([x["h_kpc"] for x in rows])
    order = float(-np.polyfit(np.log(h[1:]), np.log(a[1:]), 1)[0])
    return dict(candidate=name, rows=rows,
                agreement_convergence_order=order,
                verdict=("the direct net force converges to a fixed fraction "
                         "of G M1 M2 / d^2 as h -> 0 while the identity gap "
                         f"falls as h^{order:.2f}: the violation is the law's, "
                         "not the solver's"))


def run_sweeps():
    c = sweep_C()
    d = sweep_D()
    tot = c["settings"] + d["settings"]
    sec = c["seconds"] + d["seconds"]
    return dict(C=c, D=d, total_settings=tot, total_seconds=round(sec, 1),
                overall_rate_per_second=round(tot / max(sec, 1e-9)),
                survivors=c["survivors"] + d["survivors"])


def run_sensitivities():
    """Every headline statistic gets dS/dtheta != 0 verified numerically."""
    C = F.CANDIDATES["C1_wells_pow_p1"]
    D = F.CANDIDATES["D1_pairs_p1_q1"]
    E = F.CANDIDATES["E1_tidal"]

    def stat_aniso(c, wx, wm, pts, cloud):
        K = K_at(c, pts, wx, wm)
        return np.abs(K - np.eye(3)[None]).max()

    def stat_wslope(c, wx, wm, pts, cloud):
        return s12_coarse_selective(c, pts, Ns=(1, 8, 64, 512))[
            "weight_ratio_slope"]

    def stat_coarse(c, wx, wm, pts, cloud):
        return s11_coarse_uniform(c, cloud, pts,
                                  Ns=(1, 10, 100, 1000))["value"]

    def stat_recip(c, wx, wm, pts, cloud):
        return s10_reciprocity(c, n=32, Lbox=120.0)["value"]

    out = {}
    out["C_anisotropy_vs_sT"] = sensitivity(
        C, "sT", [0.02, 0.05, 0.2, 0.5, 1.0, 2.0], stat_aniso)
    out["C_selective_weight_slope_vs_p"] = sensitivity(
        C, "p", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], stat_wslope)
    out["C_coarse_drift_vs_L"] = sensitivity(
        C, "L", [2 * KPC, 5 * KPC, 10 * KPC, 30 * KPC, 100 * KPC], stat_coarse)
    out["C_reciprocity_vs_sT"] = sensitivity(
        C, "sT", [0.0, 0.1, 0.3, 0.6, 1.0], stat_recip)
    out["D_anisotropy_vs_alpha"] = sensitivity(
        D, "alpha", [0.03, 0.1, 0.3, 1.0, 3.0], stat_aniso)
    # family D's default alpha makes K - I only a few parts in a thousand, so
    # its reciprocity "pass" is a statement about the coupling strength, not
    # about the structure.  Sweeping alpha shows the violation is there and
    # scales with it.
    out["D_reciprocity_vs_alpha"] = sensitivity(
        D, "alpha", [0.3, 3.0, 30.0, 100.0, 300.0], stat_recip)
    out["E_anisotropy_vs_fT"] = sensitivity(
        E, "fT", [0.05, 0.2, 0.5, 1.0, 2.0], stat_aniso)
    for k, v in out.items():
        v["monotone_invariance_guard"] = (
            "PASS: statistic moves with its parameter"
            if v["responds"] else
            "FAIL: statistic is bit-identical across the swept range")
    return out


FAILURE_MODE_CHECKS = {
    "sealed_holdouts": (
        "No observational file of any kind is opened by this lane. KiDS and "
        "the wide binaries were not loaded, listed, or referenced. Every "
        "number here comes from synthetic geometries and closed-form "
        "constructions."),
    "shared_denominator": (
        "Checked. The coarse-graining reference is the FULL equal-mass cloud "
        "used as its own well list, not a member of the refinement series, so "
        "no quantity appears on both sides. The reference-free successive-step "
        "series is reported alongside and is the primary criterion. Family C's "
        "own construction does put a shared denominator (eps + sum|w_a|) into "
        "every component of S; that is a property of the candidate, and it is "
        "why the g_N factor in weight form 3 cancels exactly (see "
        "analysis_C4_gN_cancellation)."),
    "monotone_invariant_statistics": (
        "Checked. run_sensitivities() sweeps each headline statistic against "
        "the parameter it is supposed to measure and reports the spread; see "
        "the 'sensitivities' block."),
    "refitting_on_holdout": (
        "Not applicable: nothing is fitted anywhere in Stage 1 or Stage 2. "
        "All parameters are the declared global constants of each candidate."),
    "silent_extraction_failures": (
        "Checked. families.check_partition asserts row count, array shape, "
        "positivity and mass conservation after every partition (worst "
        "relative mass error observed: < 1e-15). Every solve asserts its CG "
        "residual. Infeasible configurations raise and are reported in the "
        "'infeasible' field rather than being silently skipped."),
    "test_bugs_that_look_like_solver_bugs": (
        "Checked. Open Dirichlet boundaries are used everywhere (never "
        "zero-flux). For the anisotropic families the shell value is the exact "
        "radially-aligned exterior solution -GM/(k_r r), not the constant-K "
        "monopole, because K does not become constant at infinity for these "
        "laws. Box-size convergence at FIXED grid spacing is 0.39% from "
        "L = 60 to 120 kpc; the 21% seen at fixed n was resolution, not the "
        "boundary. Every Stage-2 geometry is run with K = I on the identical "
        "grid and the null is reported next to the result."),
}


def source_hashes():
    """SHA-256 of every file that produced these numbers, including the two
    gravitylab modules this lane imports but never modifies."""
    import hashlib
    here = Path(__file__).resolve().parent
    files = [here / n for n in ("screen.py", "stage2.py", "families.py",
                                "fieldsolve.py", "dimx.py", "summarise.py")]
    files += [Path(FS._GL) / n for n in ("solver.py", "axisym.py")]
    out = {}
    for f in files:
        if f.exists():
            b = f.read_bytes()
            out[str(f)] = dict(sha256=hashlib.sha256(b).hexdigest(),
                               bytes=len(b))
    return out


def stamp_sources(path="screen_results.json"):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    doc["source_hashes"] = source_hashes()
    doc["gpu_fallbacks"] = list(F.GPU_FALLBACKS)
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc["source_hashes"]


def repair(names, path="screen_results.json", Nq=32768):
    """Re-run named candidates and merge them back into an existing results
    file, leaving everything else byte-identical.  Used when a bug is found in
    one screen path; re-running eighteen candidates to fix one is wasteful and
    would also change unrelated timings in the record."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for nm in names:
        r = run_screen(ALL[nm], Nq=Nq)
        doc["candidates"][nm] = r
        print(f"repaired {nm:24s} {r['verdict']:5s} failed={r['failed']}",
              flush=True)
    doc.setdefault("repairs", []).append(dict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        candidates=list(names)))
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def main(path="screen_results.json", Nq=32768, include_controls=True,
         verbose=True):
    names = list(F.CANDIDATES)
    if include_controls:
        names += list(CONTROLS)
    res = {}
    for nm in names:
        t = time.time()
        r = run_screen(ALL[nm], Nq=Nq)
        res[nm] = r
        if verbose:
            print(f"{nm:24s} {r['verdict']:5s} {r['seconds']:6.1f}s  "
                  f"failed={r['failed']}", flush=True)
    doc = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="work/wellnet-2026-09/screen",
        stage="1 (mathematical elimination) + 1b (coarse graining)",
        tolerances=TOL,
        gpu=F.HAVE_GPU,
        gpu_fallbacks=list(F.GPU_FALLBACKS),
        source_hashes=source_hashes(),
        failure_mode_checks=FAILURE_MODE_CHECKS,
        candidates=res)
    if verbose:
        print("\nanalyses ...", flush=True)
    doc["analyses"] = dict(
        C4_gN_cancellation=analysis_C4_gN_cancellation(),
        D_pair_scaling=analysis_D_scaling(),
        D_response_collapse=analysis_D_collapse(),
        momentum_identity=analysis_momentum_identity())
    if verbose:
        print("sensitivities ...", flush=True)
    doc["sensitivities"] = run_sensitivities()
    if verbose:
        print("exponent sweeps ...", flush=True)
    doc["sweeps"] = run_sweeps()
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    if verbose:
        print(f"\nwrote {path}")
    return doc


if __name__ == "__main__":
    main()
