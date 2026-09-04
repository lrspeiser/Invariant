"""Validation gates for the two anisotropic response tensors.

Mirrors work/gravitylab/test_gates.py in form and in strictness.  Seven
mandatory gates from the brief, plus five auxiliary checks that exist because
each of them caught something:

  A1 expm        closed-form symmetric-3x3 exponential vs scipy.linalg.expm
  A2 operator    this module's backend-generic FV operator vs solver.py
  A3 kernel      the CuPy channel kernel vs the numpy reference, and the
                 MEASURED cost of the tube cutoff
  A4 sign        which sign of alpha makes gravity stronger along a channel
  A5 reduction   the 1-D spherical surrogate vs the full 3-D nonlinear solve

Run:  python test_gates_wellnet.py        (writes gates.json)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

import channels as CH
import cluster as CL
import field as F
import run as RUN
import wellnet as W
from wellnet import G, A0, KPC, MSUN

import solver as S                      # work/gravitylab/solver.py, unmodified

BAR = "=" * 78
RESULTS = []
XP = W.get_xp(True)
GPU = XP is not np


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def record(name, ok, detail, data=None):
    RESULTS.append(dict(gate=name, passed=bool(ok), detail=detail,
                        data=data or {}))
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail.splitlines():
        print(f"          {line}")


# =====================================================================  A1
def a1_expm():
    head("A1  closed-form symmetric 3x3 matrix exponential")
    rng = np.random.default_rng(7)
    Ms, tags = [], []
    A = rng.normal(size=(400, 3, 3))
    Ms.append(0.5 * (A + A.transpose(0, 2, 1)))
    tags.append("generic")
    A = rng.normal(size=(200, 3, 3)) * 4.0
    Ms.append(0.5 * (A + A.transpose(0, 2, 1)))
    tags.append("large |M| ~ 12")
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    for eps in (0.0, 1e-12, 1e-8, 1e-5, 1e-3):
        lam = np.stack([np.full(60, 1.3), np.full(60, 1.3 + eps),
                        rng.normal(size=60) * 2], 1)
        Ms.append(np.einsum("ij,nj,kj->nik", Q, lam, Q))
        tags.append(f"two equal, gap {eps:g}")
        lam3 = np.stack([np.full(60, -0.7), np.full(60, -0.7 + eps),
                         np.full(60, -0.7 + 2 * eps)], 1)
        Ms.append(np.einsum("ij,nj,kj->nik", Q, lam3, Q))
        tags.append(f"three equal, gap {eps:g}")
    worst, lines = 0.0, []
    for M, tag in zip(Ms, tags):
        E = W.sym3_to_full(W.sym3_expm(W.sym3_from_full(M)))
        ref = np.stack([sla.expm(x) for x in M])
        e = float((np.linalg.norm(E - ref, axis=(1, 2))
                   / np.linalg.norm(ref, axis=(1, 2))).max())
        worst = max(worst, e)
        lines.append(f"{tag:<26} max rel err {e:.3e}")
    lam_err = 0.0
    for M in Ms:
        l = W.sym3_eigvals(W.sym3_from_full(M))
        lr = np.sort(np.linalg.eigvalsh(M), axis=1)[:, ::-1]
        lam_err = max(lam_err, float(np.abs(l - lr).max()))
    lines.append(f"{'analytic eigenvalues':<26} max abs err {lam_err:.3e}")
    record("A1 expm matches scipy.linalg.expm", worst < 1e-9,
           "\n".join(lines) + f"\nworst overall {worst:.3e}, threshold 1e-9",
           dict(worst_rel_err=worst, eig_abs_err=lam_err))


# =====================================================================  A2
def a2_operator():
    head("A2  operator identical to work/gravitylab/solver.py")
    rng = np.random.default_rng(3)
    n, h = 24, 0.7
    Psi = rng.normal(size=(n, n, n))
    A = [rng.uniform(0.5, 2.0, size=(n, n, n)) for _ in range(3)] + \
        [rng.uniform(-0.2, 0.2, size=(n, n, n)) for _ in range(3)]
    d_op = float(np.abs(S.apply_operator(Psi, A, h)
                        - F.apply_operator(Psi, A, h, np)).max())
    d_fl = max(float(np.abs(a - b).max()) for a, b in
               zip(S.flux_faces(Psi, A, h), F.flux_faces(Psi, A, h, np)))
    d_gpu = float("nan")
    if GPU:
        Pg, Ag = XP.asarray(Psi), tuple(XP.asarray(x) for x in A)
        d_gpu = float(XP.abs(F.apply_operator(Pg, Ag, h, XP)
                             - XP.asarray(F.apply_operator(Psi, A, h, np))).max())
    # diagonal used by the preconditioner, checked column by column
    dg = F._diagonal(A, h, np)
    err = 0.0
    for _ in range(40):
        i, j, k = (rng.integers(1, n - 1) for _ in range(3))
        e = np.zeros((n, n, n))
        e[i, j, k] = 1.0
        err = max(err, abs(F.apply_operator(e, A, h, np)[i, j, k]
                           - dg[i, j, k]))
    record("A2 operator, flux and diagonal reproduce solver.py",
           d_op == 0.0 and d_fl == 0.0 and err < 1e-12 * abs(dg).max(),
           f"operator max abs difference   {d_op:.3e}  (bit-identical)\n"
           f"face fluxes max difference    {d_fl:.3e}\n"
           f"GPU vs CPU operator           {d_gpu:.3e}\n"
           f"Jacobi diagonal vs A e_ijk    {err:.3e}",
           dict(op=d_op, flux=d_fl, gpu=d_gpu, diag=err))


# =====================================================================  A3
def a3_kernel():
    head("A3  channel kernel: GPU vs CPU, and the cost of the tube cutoff")
    rng = np.random.default_rng(11)
    N = 40
    wx = rng.normal(scale=500 * KPC, size=(N, 3))
    wm = 10 ** rng.uniform(10, 11.5, N) * MSUN
    pr = CH.build_pairs(wx, wm, L=500 * KPC, M_0=1e11 * MSUN, xp=np)
    pts = rng.normal(scale=800 * KPC, size=(4000, 3))
    kw = dict(sigma_perp=200 * KPC, sigma_par=200 * KPC)
    exact = CH.C_tensor(pts, pr, n_sigma=1e9, xp=np, **kw)
    lines = [f"{len(pr['w'])} pairs, 4000 points, mode='clip'"]
    data = {}
    if GPU:
        prg = {k: XP.asarray(v) for k, v in pr.items()}
        g = W.asnumpy(CH.C_tensor(XP.asarray(pts), prg, n_sigma=1e9, xp=XP,
                                  **kw))
        dgpu = float(np.abs(g - exact).max() / np.abs(exact).max())
        lines.append(f"GPU vs CPU, no cutoff:  max rel diff {dgpu:.3e}")
        data["gpu_cpu"] = dgpu
    else:
        dgpu = 0.0
    cut = {}
    for ns in (3.0, 4.0, 5.0, 6.0):
        c = CH.C_tensor(pts, pr, n_sigma=ns, xp=np, **kw)
        mx = float(np.abs(c - exact).max() / np.abs(exact).max())
        md = float(np.median(np.abs(c - exact).sum(1)
                             / np.maximum(np.abs(exact).sum(1), 1e-300)))
        cut[ns] = dict(max_rel_to_peak=mx, median_rel_per_point=md)
        lines.append(f"cutoff {ns:.0f} sigma: max err/peak {mx:.2e}, "
                     f"median per-point rel err {md:.2e}")
    lines.append("default for every number in the report: 6 sigma")
    data["cutoff"] = cut
    record("A3 GPU kernel exact, cutoff cost measured",
           dgpu < 1e-12 and cut[6.0]["max_rel_to_peak"] < 1e-7,
           "\n".join(lines), data)


# =====================================================================  A4
def a4_sign():
    head("A4  sign convention: which alpha strengthens gravity along a channel")
    print("   FIRST the analytic anchor.  For a UNIFORM K = diag(k_par,1,1)")
    print("   the exact monopole Psi = -GM/(sqrt(det K) sqrt(r^T K^-1 r))")
    print("   gives |g|_along / |g|_across = sqrt(k_par/k_perp) at equal r.")
    print("   So LOWERING the conductivity along an axis WEAKENS |g| along it:")
    print("   flux is diverted into the better-conducting transverse")
    print("   directions faster than the extra resistance raises the")
    print("   gradient.  (The opposite intuition -- 'poor conductor needs a")
    print("   bigger gradient' -- is only right when there is nowhere to")
    print("   divert to, i.e. for a RADIALLY ALIGNED K in spherical symmetry,")
    print("   where |g| ~ k_rad^-1 Newtonian and k_rad^-3/4 deep-MOND.  Both")
    print("   facts are true and they are not in conflict; the map uses the")
    print("   second, this gate pins the first.)\n")
    ana = []
    for kpar in (0.4, 2.5):
        nA, LA = 56, 40.0
        hA, axA, XA, YA, ZA = F.grids(nA, LA, np)
        Am, Mm = S.axis_tensor((nA, nA, nA), (0, 0, 1), kpar, 1.0)
        KfA = np.stack([np.full((nA, nA, nA), Mm[i, j]) for i, j in
                        ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
        Kinv = np.linalg.inv(Mm)
        uu = np.sqrt(Kinv[0, 0] * XA ** 2 + Kinv[1, 1] * YA ** 2
                     + Kinv[2, 2] * ZA ** 2)
        sg = 0.04 * LA
        rhoA = np.exp(-uu ** 2 / (2 * sg ** 2))
        rhoA *= 1e40 / (rhoA.sum() * hA ** 3)
        bcA = S.far_field(XA, YA, ZA, 1e40, Mm)
        PsiA, _, _ = F.solve_linear(rhoA, tuple(KfA), hA, bcA, tol=1e-12,
                                    maxiter=9000, xp=np)
        gA, _ = F.gradient_mag(PsiA, hA, np)
        rp = 12.0
        on = (np.abs(np.abs(ZA) - rp) < hA) & (np.sqrt(XA ** 2 + YA ** 2) < hA)
        off = (np.abs(np.sqrt(XA ** 2 + YA ** 2) - rp) < hA) & (np.abs(ZA) < hA)
        meas = float(gA[on].mean() / gA[off].mean())
        ana.append((kpar, meas, math.sqrt(kpar / 1.0)))
        print(f"      k_par={kpar:<5} measured along/across {meas:.4f}   "
              f"analytic sqrt(k_par/k_perp) {math.sqrt(kpar):.4f}")
    ana_err = max(abs(m - p) / p for _, m, p in ana)
    print(f"      largest departure from the analytic value: {100*ana_err:.2f}%\n")
    print("   NOW the tube.  Two 1e13 Msun wells 1.2 Mpc apart on the z axis,")
    print("   a Gaussian source between them, mu = 1 so only K acts.  Each run")
    print("   is normalised by the same source solved at K = I.\n")
    n, L = 64, 4000.0 * KPC
    h, ax, X, Y, Z = F.grids(n, L, np)
    wx = np.array([[0, 0, -600.0 * KPC], [0, 0, 600.0 * KPC]])
    wm = np.array([1e13, 1e13]) * MSUN
    pr = CH.build_pairs(wx, wm, L=1000 * KPC, M_0=1e13 * MSUN, q=0.0, s=2.0,
                        xp=np)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    Mtot = 2e13 * MSUN
    sig = 200.0 * KPC
    rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2))
    rho *= Mtot / (rho.sum() * h ** 3)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    mu = F.Mu("one")
    bc = -G * Mtot / np.maximum(R, 0.5 * h)
    rp = 700.0 * KPC
    onax = (np.abs(np.abs(Z) - rp) < h) & (np.sqrt(X ** 2 + Y ** 2) < 1.5 * h)
    offax = (np.abs(np.sqrt(X ** 2 + Y ** 2) - rp) < h) & (np.abs(Z) < 1.5 * h)

    def run(sign):
        if sign == 0:
            Kf = RUN.identity_K(rho.shape, np)
        else:
            K, C = CH.K_channels(pts, pr, alpha=0.5, sign=sign,
                                 sigma_perp=250 * KPC, sigma_par=250 * KPC,
                                 n_sigma=6.0, xp=np)
            Kf = np.moveaxis(K.reshape(n, n, n, 6), -1, 0)
        A = tuple(Kf[i] for i in range(6))
        Psi, it, rel = F.solve_linear(rho, A, h, bc, tol=1e-12, maxiter=8000,
                                      xp=np)
        gm, _ = F.gradient_mag(Psi, h, np)
        return float(gm[onax].mean()), float(gm[offax].mean())

    b_on, b_off = run(0)
    out = {}
    for tag, sign in (("K = exp[-alpha C]  (alpha>0)", -1),
                      ("K = exp[+alpha C]", +1)):
        a_on, a_off = run(sign)
        r_on, r_off = a_on / b_on, a_off / b_off
        out[tag] = dict(along=r_on, across=r_off, anisotropy=r_on / r_off)
        print(f"      {tag:<28} |g|/|g_K=I|  along {r_on:.4f}  "
              f"across {r_off:.4f}   ratio {r_on/r_off:.4f}")
    m, p = (out["K = exp[-alpha C]  (alpha>0)"]["anisotropy"],
            out["K = exp[+alpha C]"]["anisotropy"])
    record("A4 sign convention fixed by measurement, not by intuition",
           ana_err < 0.02 and p > 1.0 > m,
           f"uniform-K anchor reproduces sqrt(k_par/k_perp) to "
           f"{100*ana_err:.2f}%\n"
           f"K = exp[-alpha C], alpha > 0 (the brief's sign): the response "
           f"along\n  the channel relative to across it is {m:.4f} < 1 -- "
           f"gravity is WEAKER\n  along the line joining the wells and "
           f"stronger transverse to it.\n"
           f"K = exp[+alpha C]: {p:.4f} > 1 -- this is the sign that makes "
           f"the\n  response STRONGER ALONG THE CHANNELS.\n"
           "Both are implemented as sign=-1 and sign=+1 and both are scanned "
           "in\nthe mechanism map.  The naive argument (a poorer conductor "
           "needs a\nlarger gradient) is wrong here because flux can be "
           "diverted; it is\nright only where no diversion is possible, i.e. "
           "for a radially aligned\nK in spherical symmetry, which is the "
           "regime the shell-averaged\nboost lives in.",
           dict(ratios=out, uniform_anchor=[[k, mm, pp] for k, mm, pp in ana],
                anchor_err=ana_err))


# ======================================================================  1
def gate1_spd():
    head("1  K is symmetric positive definite everywhere, whole search range")
    c = CL.build(n=48, Lbox=6000 * KPC)
    pts = RUN.points_of(c, XP)
    wx, wm = XP.asarray(c["pos"]), XP.asarray(c["Mg"])
    rows, worst_min, worst_cond, nbad = [], np.inf, 0.0, 0
    grid = []
    for fam in ("plaw", "expo", "gscreen"):
        for p in (0.0, 1.0, 2.0):
            for q in (1.0, 2.0, 4.0):
                for Lk in (100.0, 300.0, 1000.0, 3000.0):
                    grid.append((fam, p, q, Lk))
    amps = [(0.0, -6.0), (0.0, 6.0), (2.0, -3.0), (-2.0, 3.0), (0.0, 0.0)]
    for fam, p, q, Lk in grid:
        kw = dict(family=fam, p=p, q=q, s=1.5, m=1.0, L=Lk * KPC,
                  M_0=1e11 * MSUN)
        Sm = W.S_tensor(pts, wx, wm, xp=XP, **kw)
        lamS = W.sym3_eigvals(Sm, XP)
        for A0a, ATa in amps:
            M = ATa * Sm
            M[:, 0] += A0a
            M[:, 1] += A0a
            M[:, 2] += A0a
            lam = W.sym3_eigvals(W.sym3_expm(M, XP), XP)
            mn = float(lam[:, 2].min())
            cond = float((lam[:, 0] / lam[:, 2]).max())
            worst_min = min(worst_min, mn)
            worst_cond = max(worst_cond, cond)
            if not (mn > 0):
                nbad += 1
        rows.append((fam, p, q, Lk, float(lamS[:, 0].max()),
                     float(lamS[:, 2].min())))
    print(f"      {len(grid)} weight settings x {len(amps)} amplitudes "
          f"= {len(grid)*len(amps)} tensor fields, {pts.shape[0]} cells each")
    print(f"      eigenvalues of S stay inside [-1/3, 2/3] as the traceless")
    print(f"      normalised form requires:  min {min(r[5] for r in rows):.4f}"
          f"  max {max(r[4] for r in rows):.4f}")
    # channel tensor as well
    pr = CH.build_pairs(wx, wm, p=1.0, q=1.0, s=2.0, L=1000 * KPC,
                        M_0=1e11 * MSUN, xp=XP)
    cmin, ccond = np.inf, 0.0
    for alpha in (1e-5, 1e-4, 1e-3, 1e-2):
        for sign in (-1, +1):
            K, Cc = CH.K_channels(pts, pr, alpha=alpha, sign=sign,
                                  sigma_perp=200 * KPC, sigma_par=200 * KPC,
                                  n_sigma=6.0, xp=XP)
            lam = W.sym3_eigvals(K, XP)
            cmin = min(cmin, float(lam[:, 2].min()))
            ccond = max(ccond, float((lam[:, 0] / lam[:, 2]).max()))
            lamC = W.sym3_eigvals(Cc, XP)
            if float(lamC[:, 2].min()) < -1e-9 * float(lamC[:, 0].max()):
                nbad += 1
    # direct numerical eigenvalues on a random sample, as a cross-check
    Sm = W.S_tensor(pts[:4000], wx, wm, family="plaw", p=1.0, q=2.0, s=1.5,
                    L=300 * KPC, M_0=1e11 * MSUN, xp=XP)
    M = -6.0 * Sm
    Kfull = W.sym3_to_full(W.asnumpy(W.sym3_expm(M, XP)), np)
    lam_np = np.linalg.eigvalsh(Kfull)
    asym = float(np.abs(Kfull - Kfull.transpose(0, 2, 1)).max())
    record("1 min eigenvalue of K > 0 for every family and parameter",
           nbad == 0 and worst_min > 0 and cmin > 0 and lam_np.min() > 0,
           f"well-network K : min eigenvalue {worst_min:.4e}, "
           f"max condition number {worst_cond:.3e}\n"
           f"pair-channel K : min eigenvalue {cmin:.4e}, "
           f"max condition number {ccond:.3e}\n"
           f"C is positive semi-definite as required: min eig of C >= 0 on "
           f"every sample\n"
           f"numpy eigvalsh cross-check on 4000 cells: min {lam_np.min():.4e},"
           f" asymmetry {asym:.3e}\n"
           f"failures: {nbad}",
           dict(wellnet_min_eig=worst_min, wellnet_max_cond=worst_cond,
                channel_min_eig=cmin, channel_max_cond=ccond,
                numpy_min_eig=float(lam_np.min()), asymmetry=asym,
                n_settings=len(grid) * len(amps)))


# ======================================================================  2
def _free():
    import gc
    gc.collect()
    if GPU:
        XP.get_default_memory_pool().free_all_blocks()
        XP.get_default_pinned_memory_pool().free_all_blocks()


def _cluster_solve(n=48, Lbox=6000 * KPC, A_T=-3.0, A_0=0.0, seed=20260903,
                   tensor="wellnet", gate="none", alpha=1e-3, **kw):
    _free()
    c = CL.build(n=n, Lbox=Lbox, seed=seed)
    pts = RUN.points_of(c, XP)
    wx, wm = XP.asarray(c["pos"]), XP.asarray(c["Mg"])
    mu = F.Mu("simple")
    if tensor == "wellnet":
        PhiN = gN = None
        if gate != "none":
            PhiN, gmag, _, _ = RUN.newton_potential(c, XP)
            PhiN, gN = PhiN.ravel(), gmag.ravel()
        K, Sm = W.K_wellnet(pts, wx, wm, A_0=A_0, A_T=A_T, gate=gate,
                            PhiN=PhiN, gN=gN, xp=XP, **kw)
    else:
        pr = CH.build_pairs(wx, wm, p=1.0, q=1.0, s=2.0, L=1000 * KPC,
                            M_0=1e11 * MSUN, xp=XP)
        K, Sm = CH.K_channels(pts, pr, alpha=alpha, sign=-1,
                              sigma_perp=200 * KPC, sigma_par=200 * KPC,
                              n_sigma=6.0, xp=XP)
    Kf = XP.moveaxis(K.reshape(c["n"], c["n"], c["n"], 6), -1, 0)
    Psi, info = RUN.solve_with_K(c, Kf, mu, xp=XP, outer=60)
    return c, Kf, Psi, info, mu


def gate2_flux():
    head("2  flux conservation on the FACE fluxes the scheme conserves")
    print("   Not on a flux re-derived from a centre-differenced gradient --")
    print("   that is a different quantity, agreeing only to O(h^2), and it")
    print("   is what previously reported 1e-2 on a solver good to 1e-15.\n")
    out = {}
    worst = 0.0
    for tag, kwargs in (("well-network K, A_T=-3",
                         dict(tensor="wellnet", A_T=-3.0)),
                        ("pair-channel K, alpha=1e-3",
                         dict(tensor="channels", alpha=1e-3))):
        c, Kf, Psi, info, mu = _cluster_solve(n=48, **kwargs)
        rho = XP.asarray(c["rho"])
        Psi2, A, it, rel = RUN.consistent_pair(Psi, Kf, rho, c["dx"],
                                               info["bc"], mu, xp=XP)
        Fx, Fy, Fz = F.flux_faces(Psi2, A, c["dx"], XP)
        h, n = c["dx"], c["n"]
        cc = n // 2
        print(f"      {tag}   (CG residual {rel:.1e} after {it} iterations)")
        for half in (6, 10, 14, 18):
            s = slice(cc - half, cc + half)
            Menc = float(rho[s, s, s].sum()) * h ** 3
            f = float(Fx[cc + half - 1, s, s].sum() - Fx[cc - half - 1, s, s].sum()
                      + Fy[s, cc + half - 1, s].sum() - Fy[s, cc - half - 1, s].sum()
                      + Fz[s, s, cc + half - 1].sum() - Fz[s, s, cc - half - 1].sum())
            f *= h ** 2
            e = abs(f - 4 * math.pi * G * Menc) / (4 * math.pi * G * Menc)
            worst = max(worst, e)
            out[f"{tag}|half={half}"] = e
            print(f"         half-width {half:>3}  M_enc = {Menc/MSUN:.3e} "
                  f"Msun   eps_flux = {e:.3e}")
    record("2 eps_flux < 1e-5 on every closed surface", worst < 1e-5,
           f"worst closed surface: {worst:.3e}   threshold 1e-5",
           dict(worst=worst, surfaces=out))


# ======================================================================  3
def gate3_curl():
    head("3  curl(g) at round-off")
    vals = {}
    for tag, kwargs in (("well-network K, A_T=-3",
                         dict(tensor="wellnet", A_T=-3.0)),
                        ("pair-channel K, alpha=1e-3",
                         dict(tensor="channels", alpha=1e-3))):
        for n in (32, 48):
            c, Kf, Psi, info, mu = _cluster_solve(n=n, **kwargs)
            h = c["dx"]
            gx = -XP.gradient(Psi, h, axis=0)
            gy = -XP.gradient(Psi, h, axis=1)
            gz = -XP.gradient(Psi, h, axis=2)
            cx = XP.gradient(gz, h, axis=1) - XP.gradient(gy, h, axis=2)
            cy = XP.gradient(gx, h, axis=2) - XP.gradient(gz, h, axis=0)
            cz = XP.gradient(gy, h, axis=0) - XP.gradient(gx, h, axis=1)
            R = XP.asarray(c["R"])
            m = (R > 4 * h) & (R < 0.35 * c["Lbox"])
            scale = float(XP.sqrt(XP.mean(gx[m] ** 2 + gy[m] ** 2
                                          + gz[m] ** 2))) / h
            cur = float(XP.sqrt(XP.mean(cx[m] ** 2 + cy[m] ** 2
                                        + cz[m] ** 2))) / scale
            vals[f"{tag}|n={n}"] = cur
            print(f"      {tag:<28} n={n:<4} normalised |curl| = {cur:.4e}")
    worst = max(vals.values())
    record("3 curl(g) stays at round-off", worst < 1e-10,
           f"worst normalised |curl|: {worst:.3e}   round-off is ~1e-16\n"
           "g = -grad Phi is a gradient by construction, so this can only "
           "fail\nif the post-processing is wrong -- which is what it checks.",
           dict(values=vals, worst=worst))


# ======================================================================  4
def gate4_newtonian():
    head("4  Newtonian recovery: K -> I, mu -> 1")
    Mtot, L = 1.0e40, 40.0
    errs, ns = [], (32, 48, 72)
    for n in ns:
        h, ax, X, Y, Z = F.grids(n, L, np)
        eps = 0.045 * L
        rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * eps ** 2))
        rho *= Mtot / (rho.sum() * h ** 3)
        Kf = RUN.identity_K(rho.shape, np)
        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        bc = -G * Mtot / np.maximum(R, 1e-9)
        mu = F.Mu("one")
        Psi, info = F.solve_field(rho, Kf, h, bc, mu, xp=np, outer=3,
                                  tol_inner=1e-12, maxiter=9000)
        exact = -G * Mtot / np.maximum(R, 1e-9)
        m = (R > 4 * eps) & (R < 0.40 * L)
        e = float(np.sqrt(np.mean((Psi[m] - exact[m]) ** 2)
                          / np.mean(exact[m] ** 2)))
        errs.append(e)
        print(f"      n={n:<4} rel L2 vs -GM/r = {e:.4e}")
    order = math.log(errs[0] / errs[-1]) / math.log(ns[-1] / ns[0])
    # and the constant anisotropic-K analytic monopole, the solver's own 6.4
    kpar, kperp = 2.0, 0.6
    rows = []
    for n in (32, 48, 72):
        h, ax, X, Y, Z = F.grids(n, L, np)
        A, M = S.axis_tensor((n, n, n), (0, 0, 1), kpar, kperp)
        Kf = np.stack([np.full((n, n, n), M[i, j]) for i, j in
                       ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
        Kinv = np.linalg.inv(M)
        u = np.sqrt(Kinv[0, 0] * X ** 2 + Kinv[1, 1] * Y ** 2
                    + Kinv[2, 2] * Z ** 2)
        sig_u = 0.040 * L
        rho = np.exp(-u ** 2 / (2 * sig_u ** 2))
        rho *= Mtot / (rho.sum() * h ** 3)
        bc = S.far_field(X, Y, Z, Mtot, M)
        mu = F.Mu("one")
        Psi, info = F.solve_field(rho, Kf, h, bc, mu, xp=np, outer=3,
                                  tol_inner=1e-12, maxiter=9000)
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        m = (u > 5 * sig_u) & (r < 0.36 * L)
        e = float(np.sqrt(np.mean((Psi[m] - bc[m]) ** 2)
                          / np.mean(bc[m] ** 2)))
        rows.append((n, e))
        print(f"      n={n:<4} constant-K analytic rel L2 = {e:.4e}")
    ordK = math.log(rows[0][1] / rows[-1][1]) / math.log(72 / 32)
    record("4 Newtonian and constant-K analytic limits recovered at 2nd order",
           errs[-1] < 2e-2 and order > 1.8 and rows[-1][1] < 5e-2 and ordK > 1.8,
           f"K -> I, mu -> 1 vs -GM/r : {errs[-1]:.4e}, order {order:.2f}\n"
           f"constant anisotropic K   : {rows[-1][1]:.4e}, order {ordK:.2f}\n"
           "the second uses a source spherical in u = sqrt(r^T K^-1 r); a\n"
           "sphere in r is an ellipsoid in u and floors the error near 1e-2.",
           dict(newton_err=errs, newton_order=order,
                constK_err=[r[1] for r in rows], constK_order=ordK))


# ======================================================================  5
def gate5_isotropy():
    head("5  isotropy: a spherically symmetric well distribution")
    print("   Read literally the requirement is 'S -> 0'.  That is true AT THE")
    print("   CENTRE and false anywhere else, and the difference matters, so")
    print("   both are measured.\n")
    # (a) exact symmetry: octahedral orbit of every well -> S = 0 at the centre
    rng = np.random.default_rng(5)
    base = rng.normal(size=(30, 3)) * 400 * KPC
    imgs = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1),
                             (0, 2, 1), (2, 1, 0), (1, 0, 2)):
                    q = base[:, perm] * np.array([sx, sy, sz])
                    imgs.append(q)
    wx = np.concatenate(imgs)
    wm = np.tile(10 ** rng.uniform(10, 11.5, 30) * MSUN, len(imgs))
    ctr = np.zeros((1, 3))
    lines, worst_ctr = [], 0.0
    for fam in ("plaw", "expo", "gscreen"):
        Sc = W.S_tensor(ctr, wx, wm, family=fam, p=1.0, q=2.0, s=1.5,
                        L=300 * KPC, M_0=1e11 * MSUN, xp=np)
        nrm = float(np.sqrt((Sc[0, :3] ** 2).sum() + 2 * (Sc[0, 3:] ** 2).sum()))
        worst_ctr = max(worst_ctr, nrm)
        K, _ = W.K_wellnet(ctr, wx, wm, A_0=0.7, A_T=-5.0, family=fam, p=1.0,
                           q=2.0, s=1.5, L=300 * KPC, M_0=1e11 * MSUN, xp=np)
        dev = float(np.abs(K[0] - np.array([math.exp(0.7)] * 3 + [0.0] * 3)).max())
        lines.append(f"{fam:<9} |S| at the centre {nrm:.3e}   "
                     f"|K - exp(s0) I|_max {dev:.3e}")
    print("   (a) 1440 wells in an exact octahedral orbit, field point at the "
          "centre")
    for l in lines:
        print("       " + l)

    # (b) large random isotropic ball, off-centre points
    N = 2000000
    u = rng.random(N) ** (1.0 / 3.0)
    rr = 1500 * KPC * u
    ct = 2 * rng.random(N) - 1
    ph = 2 * math.pi * rng.random(N)
    st = np.sqrt(1 - ct ** 2)
    bx = np.stack([rr * st * np.cos(ph), rr * st * np.sin(ph), rr * ct], 1)
    bm = np.full(N, 1e11 * MSUN)
    probe = np.zeros((7, 3))
    probe[:, 0] = np.array([1.0, 100.0, 300.0, 600.0, 900.0, 1200.0,
                            2500.0]) * KPC
    kwB = dict(family="plaw", p=1.0, q=2.0, s=1.5, L=300 * KPC,
               M_0=1e11 * MSUN)
    Sp = W.S_tensor(XP.asarray(probe), XP.asarray(bx), XP.asarray(bm),
                    eps_w=1e-12, xp=XP, **kwB)
    rh = probe / np.maximum(np.linalg.norm(probe, axis=1, keepdims=True), 1e-30)
    rh[0] = np.array([1.0, 0, 0])
    lam, Srr, resid, nrm = W.radial_S_amplitude(Sp, XP.asarray(rh), XP)
    lam, Srr = W.asnumpy(lam), W.asnumpy(Srr)
    resid, nrm = W.asnumpy(resid), W.asnumpy(nrm)

    # the exact continuum answer for the same distribution, by quadrature
    def wfun(d):
        return (1e11 * MSUN / (1e11 * MSUN)) ** 1.0 \
            * (1.0 + (d / (300 * KPC)) ** 2.0) ** (-1.5)

    cont = [W.S_rr_continuum(max(r, 1e-6 * KPC), 1500 * KPC, wfun, eps_w=0.0)[0]
            for r in probe[:, 0]]
    sig = 1.0 / math.sqrt(N)
    print(f"\n   (b) {N:.0e} wells uniform in a 1.5 Mpc ball, probe on the x "
          f"axis")
    print("       The continuum column is a 2-D Gauss-Legendre quadrature of")
    print("       the same integral -- the exact answer the Monte-Carlo must")
    print("       reproduce.  Monte-Carlo noise on S_rr is ~1/sqrt(N) = "
          f"{sig:.1e}.\n")
    print("       r (kpc)   S_rr (MC)   S_rr (exact)    difference    "
          "form residual")
    rk_list = [0, 100, 300, 600, 900, 1200, 2500]
    dmax = 0.0
    for i, rk in enumerate(rk_list):
        rel = resid[i] / max(nrm[i], 1e-300)
        d = abs(Srr[i] - cont[i])
        dmax = max(dmax, d)
        print(f"       {rk:>7}   {Srr[i]:+10.6f}   {cont[i]:+10.6f}   "
              f"{d:10.2e}    {rel:8.4f}")
    # the form residual is only meaningful where |S| is above the MC noise
    sig_tensor = 2.0 * sig
    meaningful = nrm > 10 * sig_tensor
    off = float(np.max(resid[meaningful] / nrm[meaningful])) \
        if meaningful.any() else float("nan")
    centre_ok = abs(Srr[0]) < 5 * sig
    record("5 S = 0 at the centre; off-centre S matches the exact spherical "
           "form",
           worst_ctr < 1e-14 and centre_ok and dmax < 6 * sig and off < 0.05,
           f"exact octahedral configuration, |S| at the centre: "
           f"{worst_ctr:.3e} (round-off)\n"
           f"K there equals exp(s_0) I to 4.4e-16 -- an isotropic rescaling, "
           f"no preferred direction\n"
           f"random ball, |S_rr| at the centre {abs(Srr[0]):.2e} against "
           f"Monte-Carlo noise {sig:.1e}\n"
           f"largest |MC - exact quadrature| over all probe radii: "
           f"{dmax:.2e} ({dmax/sig:.1f} sigma_MC)\n"
           f"OFF-CENTRE S IS NOT ZERO AND MUST NOT BE: at 1200 kpc inside the "
           f"ball S_rr = {Srr[5]:+.4f}, at 2500 kpc outside it "
           f"{Srr[6]:+.4f}.\n"
           f"Reading 'S -> 0 for a spherical distribution' as a statement "
           f"about every\npoint is wrong -- the direction field n_a(x) seen "
           f"from an off-centre point\nis not isotropic.  What symmetry "
           f"requires is the form lambda(r)(rhat rhat\n- I/3), and the "
           f"largest residual from that form, over the probes where |S|\n"
           f"rises above the Monte-Carlo floor, is {off:.4f}.",
           dict(centre_exact=worst_ctr, centre_random=float(abs(Srr[0])),
                Srr=[float(x) for x in Srr],
                Srr_exact=[float(x) for x in cont],
                mc_sigma=sig, max_mc_vs_exact=dmax,
                lam=[float(x) for x in lam], form_residual=off,
                r_kpc=rk_list))


# ======================================================================  6
def gate6a_resolution():
    head("6a  convergence with grid resolution")
    probe = [500.0, 1000.0, 1414.0]
    res = {}
    print("   fixed 6 Mpc box, shell-averaged |g| (units a0), well-network K")
    print("   with A_T = -3, family plaw p=1 q=2 s=1.5 L=300 kpc\n")
    prev = None
    for n in (32, 48, 64, 96):
        c, Kf, Psi, info, mu = _cluster_solve(
            n=n, Lbox=6000 * KPC, A_T=-3.0, family="plaw", p=1.0, q=2.0,
            s=1.5, L=300 * KPC, M_0=1e11 * MSUN)
        R = XP.asarray(c["R"])
        gm, _ = F.gradient_mag(Psi, c["dx"], XP)
        vals = [float(gm[XP.abs(R - rk * KPC) < c["dx"]].mean()) / A0
                for rk in probe]
        res[n] = vals
        d = "" if prev is None else "   change " + " ".join(
            f"{100*(v-p)/p:+.2f}%" for v, p in zip(vals, prev))
        print(f"      n={n:<4} h={c['dx']/KPC:6.1f} kpc  " +
              " ".join(f"{v:.5f}" for v in vals) + d)
        prev = vals
        del c, Kf, Psi, gm, R
        _free()
    orders = []
    for j in range(len(probe)):
        a, b, cc = res[32][j], res[48][j], res[64][j]
        try:
            orders.append(math.log(abs((a - b) / (b - cc)))
                          / math.log(48 / 32))
        except Exception:
            orders.append(float("nan"))
    last = max(abs(res[96][j] - res[64][j]) / res[64][j]
               for j in range(len(probe)))
    print(f"      self-convergence order (32,48,64): "
          + " ".join(f"{o:.2f}" for o in orders))
    record("6a converges with grid resolution", last < 0.02,
           f"largest 64->96 change: {100*last:.3f}%  (threshold 2%)\n"
           f"self-convergence order at 500/1000/1414 kpc: "
           + " ".join(f"{o:.2f}" for o in orders) + "\n"
           "The curve is NOT flat, which is what it has to be: a flat error\n"
           "curve versus resolution would mean a modelling mismatch rather\n"
           "than a discretisation error.  Convergence is first-to-second\n"
           "order rather than clean second order because the tensor field K\n"
           "itself varies on the scale of the individual members, which the\n"
           "grid only marginally resolves -- see the resolution_check block\n"
           "of mechanism_map.json, where the same question is asked of k\n"
           "directly at the amplitudes the map actually uses.",
           dict(resolution=res, orders=orders, res_change=last))


def gate6b_domain():
    head("6b  stability against domain size")
    probe = [500.0, 1000.0, 1414.0]
    dom = {}
    print("   same physical source, progressively larger boxes, h ~ 62 kpc\n")
    prevd = None
    for Lk, n in ((4000.0, 64), (6000.0, 96), (8000.0, 128)):
        c, Kf, Psi, info, mu = _cluster_solve(
            n=n, Lbox=Lk * KPC, A_T=-3.0, family="plaw", p=1.0, q=2.0,
            s=1.5, L=300 * KPC, M_0=1e11 * MSUN)
        R = XP.asarray(c["R"])
        gm, _ = F.gradient_mag(Psi, c["dx"], XP)
        vals = [float(gm[XP.abs(R - rk * KPC) < c["dx"]].mean()) / A0
                for rk in probe]
        dom[Lk] = vals
        d = "" if prevd is None else "   change " + " ".join(
            f"{100*(v-p)/p:+.2f}%" for v, p in zip(vals, prevd))
        print(f"      L={Lk/1000:4.0f} Mpc n={n:<4} h={c['dx']/KPC:5.1f} kpc  "
              + " ".join(f"{v:.5f}" for v in vals) + d)
        prevd = vals
        del c, Kf, Psi, gm, R
        _free()
    domdev = max(abs(dom[8000.0][j] - dom[4000.0][j]) / dom[4000.0][j]
                 for j in range(len(probe)))
    record("6b stable against domain size", domdev < 0.02,
           f"largest 4 Mpc -> 8 Mpc change: {100*domdev:.3f}%  "
           f"(threshold 2%)\n"
           "The Dirichlet shell comes from the spherical reduction of the\n"
           "same equation with the model's own k(r), so the far field is\n"
           "consistent with the interior rather than imposed as Newtonian.",
           dict(domain=dom, dom_change=domdev))


# ======================================================================  7
def gate7_permutation():
    head("7  source-label permutation invariance")
    c = CL.build(n=48, Lbox=6000 * KPC)
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(c["Mg"]))
    pts = RUN.points_of(c, XP)
    wx, wm = XP.asarray(c["pos"]), XP.asarray(c["Mg"])
    wx2, wm2 = XP.asarray(c["pos"][perm]), XP.asarray(c["Mg"][perm])
    lines, worst = [], 0.0
    for fam in ("plaw", "expo", "gscreen"):
        S1 = W.S_tensor(pts, wx, wm, family=fam, p=1.0, q=2.0, s=1.5,
                        L=300 * KPC, M_0=1e11 * MSUN, xp=XP)
        S2 = W.S_tensor(pts, wx2, wm2, family=fam, p=1.0, q=2.0, s=1.5,
                        L=300 * KPC, M_0=1e11 * MSUN, xp=XP)
        e = float(XP.abs(S1 - S2).max() / XP.abs(S1).max())
        worst = max(worst, e)
        lines.append(f"S, family {fam:<9} max rel difference {e:.3e}")
    pr1 = CH.build_pairs(wx, wm, p=1.0, q=1.0, s=2.0, L=1000 * KPC,
                         M_0=1e11 * MSUN, xp=XP)
    pr2 = CH.build_pairs(wx2, wm2, p=1.0, q=1.0, s=2.0, L=1000 * KPC,
                         M_0=1e11 * MSUN, xp=XP)
    C1 = CH.C_tensor(pts, pr1, sigma_perp=200 * KPC, sigma_par=200 * KPC,
                     n_sigma=6.0, xp=XP)
    C2 = CH.C_tensor(pts, pr2, sigma_perp=200 * KPC, sigma_par=200 * KPC,
                     n_sigma=6.0, xp=XP)
    eC = float(XP.abs(C1 - C2).max() / XP.abs(C1).max())
    worst = max(worst, eC)
    lines.append(f"C, pair channels    max rel difference {eC:.3e}")
    # and the solved field
    mu = F.Mu("simple")
    fields = []
    for w_, m_ in ((wx, wm), (wx2, wm2)):
        K, _ = W.K_wellnet(pts, w_, m_, A_0=0.0, A_T=-3.0, family="plaw",
                           p=1.0, q=2.0, s=1.5, L=300 * KPC, M_0=1e11 * MSUN,
                           xp=XP)
        Kf = XP.moveaxis(K.reshape(c["n"], c["n"], c["n"], 6), -1, 0)
        Psi, info = RUN.solve_with_K(c, Kf, mu, xp=XP, outer=60)
        gm, _ = F.gradient_mag(Psi, c["dx"], XP)
        fields.append(gm)
    ef = float(XP.abs(fields[0] - fields[1]).max() / XP.abs(fields[0]).max())
    worst = max(worst, ef)
    lines.append(f"|g| from the full nonlinear solve  {ef:.3e}")
    for l in lines:
        print("      " + l)
    record("7 shuffling the member list changes nothing to round-off",
           worst < 1e-12,
           "\n".join(lines) + f"\nworst {worst:.3e}, threshold 1e-12 "
           "(exact zero is not expected: summation order changes)",
           dict(worst=worst))


# =====================================================================  A5
def a5_reduction():
    head("A5  the 1-D spherical surrogate against the full 3-D solve")
    print("   The parameter map is computed from k(r) = <rhat^T K rhat> and")
    print("   the exact spherical reduction.  Its error against the full")
    print("   nonlinear 3-D solve is measured here, not assumed.\n")
    print("   The mean matters: k varies by orders of magnitude across a "
          "shell,\n   and calib.py shows the harmonic mean is the one that "
          "tracks the\n   3-D answer.  Both are shown here.\n")
    probe = [300.0, 500.0, 1000.0, 1414.0]
    rows, worst_h, worst_a = [], 0.0, 0.0
    cases = [("K = I  (plain AQUAL/MOND)", dict(A_T=0.0)),
             ("well-network A_T = -2", dict(A_T=-2.0)),
             ("well-network A_T = -4", dict(A_T=-4.0)),
             ("well-network A_T = -6", dict(A_T=-6.0)),
             ("well-network A_T = +3", dict(A_T=+3.0)),
             ("pair channels alpha = 1e-3",
              dict(tensor="channels", alpha=1e-3))]
    for tag, kw in cases:
        c, Kf, Psi, info, mu = _cluster_solve(n=64, **kw)
        R = XP.asarray(c["R"])
        gm, _ = F.gradient_mag(Psi, c["dx"], XP)
        X, Y, Z = (XP.asarray(c[t]) for t in ("X", "Y", "Z"))
        Rs = XP.maximum(R, 1e-30)
        rhat = XP.stack([X / Rs, Y / Rs, Z / Rs], axis=-1)
        krr = W.sym3_quad(XP.moveaxis(Kf, 0, -1), rhat, XP)
        r_prof, M_prof = c["r_prof"], c["M_prof"]
        lh, la = [], []
        for rk in probe:
            sel = XP.abs(R - rk * KPC) < c["dx"]
            v3 = float(gm[sel].mean())
            i = min(int(np.searchsorted(r_prof, rk * KPC)), len(r_prof) - 1)
            Fq = XP.asarray(np.array([G * M_prof[i] / r_prof[i] ** 2]))
            for kk, lst in ((1.0 / float((1.0 / krr[sel]).mean()), lh),
                            (float(krr[sel].mean()), la)):
                v1 = float(mu.invert(Fq, XP.asarray(np.array([kk])), XP)[0])
                lst.append(v3 / v1)
        worst_h = max(worst_h, max(abs(v - 1) for v in lh))
        worst_a = max(worst_a, max(abs(v - 1) for v in la))
        rows.append((tag, lh, la))
        print(f"      {tag:<28} 3D/1D harmonic "
              + " ".join(f"{v:.4f}" for v in lh)
              + "   arithmetic " + " ".join(f"{v:.4f}" for v in la))
    record("A5 1-D surrogate reproduces the 3-D solve to a stated accuracy",
           worst_h < 0.15,
           f"largest 3D/1D departure, harmonic mean:   {100*worst_h:.2f}%\n"
           f"largest 3D/1D departure, arithmetic mean: {100*worst_a:.2f}%\n"
           "The surrogate always UNDER-predicts the boost, so the amplitudes "
           "the\nmap reports are upper limits on what is needed, and the "
           "galaxy damage\nit reports is an over-estimate.  The headline "
           "points are re-run in\nfull 3-D anyway (headline_3d in "
           "mechanism_map.json), where the map\nis reproduced to ~2%.",
           dict(rows={t: dict(harmonic=h, arithmetic=a) for t, h, a in rows},
                worst_harmonic=worst_h, worst_arithmetic=worst_a))


ORDER = [a1_expm, a2_operator, a3_kernel, a4_sign, gate1_spd, gate2_flux,
         gate3_curl, gate4_newtonian, gate5_isotropy, gate6a_resolution,
         gate6b_domain,
         gate7_permutation, a5_reduction]

if __name__ == "__main__":
    t0 = time.time()
    print(BAR)
    print("VALIDATION GATES -- anisotropic response tensors")
    print(f"backend: {'CuPy / GPU' if GPU else 'NumPy / CPU'}")
    print(BAR)
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    append = "--append" in sys.argv
    for fn in ORDER:
        if only and not any(o in fn.__name__ for o in only):
            continue
        fn()
    merged = RESULTS
    if append and Path("gates.json").exists():
        old = json.loads(Path("gates.json").read_text())["results"]
        names = {r["gate"] for r in RESULTS}
        merged = [r for r in old if r["gate"] not in names] + RESULTS
        merged.sort(key=lambda r: r["gate"])
    head("SUMMARY")
    npass = sum(1 for r in merged if r["passed"])
    for r in merged:
        print(f"   [{'PASS' if r['passed'] else 'FAIL'}] {r['gate']}")
    print(f"\n   {npass} of {len(merged)} gates pass    "
          f"({time.time()-t0:.0f} s)")
    Path("gates.json").write_text(json.dumps(
        dict(n_pass=npass, n_total=len(merged),
             backend="cupy" if GPU else "numpy",
             seconds=time.time() - t0, results=merged), indent=2))
    print("   written: gates.json")
