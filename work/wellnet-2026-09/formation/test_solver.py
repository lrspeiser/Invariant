"""Gates for the periodic solver.  Nothing downstream is trusted until these
pass, because a test bug that looks like a solver bug has already cost this
programme four numbers.

G1  constant-K plane wave, analytic, with the convergence ORDER measured
G2  anisotropic constant K, analytic (this is the discretisation the
    anisotropy result rests on, so it is checked at a tensor, not just at I)
G3  the operator is SYMMETRIC to round-off (otherwise CG is not applicable)
G4  discrete flux conservation: the divergence of the face fluxes sums to zero
G5  exact 1-D AQUAL: mu(|psi'|/a0) psi' = S(z) solved algebraically, compared
    with the full 3-D nonlinear solve
G6  net force vanishes for a variational law on a random field (the momentum
    null), so the non-zero values later are physics and not the solver
"""
from __future__ import annotations

import json
import time

import numpy as np

import linear_response as LR
from linear_response import G, MPC, PeriodicBox, mu_simple

OUT = {}


def _iso(shape, v=1.0):
    o = np.full(shape, float(v))
    z = np.zeros(shape)
    return (o, o.copy(), o.copy(), z, z.copy(), z.copy())


def g1_constant_K():
    rows = []
    L = 100.0 * MPC
    rho0, dlt = 1.0e-27, 0.1
    for n in (16, 24, 32, 48, 64):
        b = PeriodicBox(n, L)
        k = 2 * np.pi / L
        rho = rho0 * (1.0 + dlt * np.cos(k * b.Z))
        A = _iso(rho.shape)
        psi, it, rel = b.solve_linear(rho, A)
        exact = -4 * np.pi * G * rho0 * dlt / k ** 2 * np.cos(k * b.Z)
        err = float(np.max(np.abs(psi - exact)) / np.max(np.abs(exact)))
        rows.append(dict(n=n, err=err, cg_iters=it, cg_rel=rel))
    ns = np.array([r["n"] for r in rows], float)
    es = np.array([r["err"] for r in rows], float)
    order = float(-np.polyfit(np.log(ns), np.log(es), 1)[0])
    return dict(rows=rows, order=order, order_target=2.0,
                pass_=bool(order > 1.85 and es[-1] < 3e-3))


def g2_anisotropic_K():
    L = 100.0 * MPC
    rho0, dlt = 1.0e-27, 0.1
    n = 48
    b = PeriodicBox(n, L)
    K = np.array([[1.7, 0.3, -0.2], [0.3, 0.8, 0.15], [-0.2, 0.15, 1.25]])
    A = tuple(np.full(b.X.shape, K[i, j]) for i, j in
              ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))
    out = []
    for kvec in ([1, 0, 0], [0, 1, 1], [1, 2, -1]):
        kv = 2 * np.pi / L * np.array(kvec, float)
        ph = kv[0] * b.X + kv[1] * b.Y + kv[2] * b.Z
        rho = rho0 * (1.0 + dlt * np.cos(ph))
        psi, it, rel = b.solve_linear(rho, A)
        kKk = float(kv @ K @ kv)
        exact = -4 * np.pi * G * rho0 * dlt / kKk * np.cos(ph)
        out.append(dict(k=kvec,
                        err=float(np.max(np.abs(psi - exact))
                                  / np.max(np.abs(exact))),
                        cg_iters=it))
    return dict(rows=out, pass_=bool(max(r["err"] for r in out) < 6e-3))


def g3_symmetry(seed=1):
    rng = np.random.default_rng(seed)
    b = PeriodicBox(12, 100.0 * MPC)
    sh = b.X.shape
    M = rng.normal(size=(3, 3))
    M = M @ M.T + 3 * np.eye(3)
    A = tuple(np.full(sh, M[i, j]) * (1.0 + 0.3 * np.cos(b.Z * 6e-25))
              for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))
    u = rng.normal(size=sh)
    v = rng.normal(size=sh)
    u -= u.mean()
    v -= v.mean()
    a = float(np.sum(v * b.apply(u, A)))
    c = float(np.sum(u * b.apply(v, A)))
    return dict(vAu=a, uAv=c,
                rel_asym=float(abs(a - c) / max(abs(a), abs(c), 1e-300)),
                pass_=bool(abs(a - c) / max(abs(a), abs(c)) < 1e-12))


def g4_flux_conservation(seed=2):
    rng = np.random.default_rng(seed)
    b = PeriodicBox(24, 100.0 * MPC)
    sh = b.X.shape
    A = _iso(sh, 1.0)
    A = tuple(a * (1.0 + 0.5 * rng.random(sh)) for a in A[:3]) + A[3:]
    psi = rng.normal(size=sh)
    d = b.apply(psi, A)
    return dict(sum_div_rel=float(abs(d.sum())
                                  / max(np.abs(d).sum(), 1e-300)),
                pass_=bool(abs(d.sum()) / np.abs(d).sum() < 1e-12))


def g5_aqual_1d():
    """3-D nonlinear AQUAL solve vs the EXACT 1-D algebraic solution."""
    L = 60.0 * MPC
    rho0, dlt = 4.2e-28, 0.4
    a0 = LR.CANDIDATES["aqual"]["a0"]
    rows = []
    for n in (32, 48, 64):
        b = PeriodicBox(n, L)
        k = 2 * np.pi / L
        rho = rho0 * (1.0 + dlt * np.cos(k * b.Z))

        def Afun(psi):
            return LR.aqual_tensor_field(b, psi, a0)

        psi, A, hist = b.solve_nonlinear(rho, Afun)
        # exact: mu(|psi'|/a0) psi' = S(z), S = 4 pi G Int (rho - rhobar) dz.
        # Compared on the GRADIENT at the i+1/2 faces, which is the quantity
        # the discretisation actually carries -- re-integrating psi would
        # compare a rectangle rule with a solve and mislabel the difference.
        zf = (np.arange(n) + 1.0) * b.h
        S = 4 * np.pi * G * rho0 * dlt * np.sin(k * zf) / k
        u_ex = 0.5 * (S + np.sign(S) * np.sqrt(S ** 2 + 4 * np.abs(S) * a0))
        line = psi[0, 0, :]
        u_num = (np.roll(line, -1) - line) / b.h
        sc = np.max(np.abs(u_ex))
        # THE DEEP-MOND FIELD IS NOT SMOOTH.  u = nu(|S|/a0) S ~ sqrt(|S| a0)
        # wherever S passes through zero, i.e. at every node of the source, so
        # |grad psi| has a SQUARE-ROOT CUSP there and no discretisation can be
        # second order in the max norm.  Both norms are therefore reported: the
        # max norm away from the cusps (expect order 2) and the L2 norm over
        # the whole line (expect ~1.5, cusp-limited).
        away = np.abs(S) > 0.3 * np.max(np.abs(S))
        rows.append(dict(n=n,
                         err=float(np.max(np.abs(u_num - u_ex)) / sc),
                         err_away=float(np.max(np.abs((u_num - u_ex)[away]))
                                        / sc),
                         err_l2=float(np.sqrt(np.mean((u_num - u_ex) ** 2))
                                      / sc),
                         picard=len(hist), dpsi=hist[-1]["dpsi"]))
    ns = np.array([r["n"] for r in rows], float)
    ordr = {k: float(-np.polyfit(np.log(ns),
                                 np.log([r[k] for r in rows]), 1)[0])
            for k in ("err", "err_away", "err_l2")}
    return dict(rows=rows, order=ordr,
                deep_mond_ratio=float(4 * np.pi * G * rho0 * dlt
                                      / (2 * np.pi / L) / a0),
                note="order['err'] ~ 0.5 is the square-root cusp at the source "
                     "nodes, a property of the deep-MOND field equation, not of "
                     "the discretisation",
                pass_=bool(rows[-1]["err_away"] < 1e-2
                           and ordr["err_away"] > 1.5))


def g6_momentum_null(seed=3):
    """A variational law in a periodic box has zero net force.  Measured."""
    rng = np.random.default_rng(seed)
    L = 60.0 * MPC
    n = 40
    b = PeriodicBox(n, L)
    rho0 = 4.2e-28
    d = np.zeros(b.X.shape)
    for _ in range(6):
        kv = 2 * np.pi / L * rng.integers(-2, 3, 3).astype(float)
        if not np.any(kv):
            continue
        d += 0.25 * np.cos(kv[0] * b.X + kv[1] * b.Y + kv[2] * b.Z
                           + rng.random() * 2 * np.pi)
    rho = rho0 * (1.0 + d)
    a0 = LR.CANDIDATES["aqual"]["a0"]
    out = {}
    for tag, a0f in (("newton", None), ("aqual", a0)):
        if a0f is None:
            A = _iso(rho.shape)
            psi, _, _ = b.solve_linear(rho, A)
        else:
            psi, A, _ = b.solve_nonlinear(
                rho, lambda p: LR.aqual_tensor_field(b, p, a0f))
        F, acc = b.net_force(rho, psi)
        gx, gy, gz = b.grad(psi)
        scale = float(np.sqrt(np.mean(gx ** 2 + gy ** 2 + gz ** 2)))
        out[tag] = dict(net_accel=acc.tolist(),
                        rel=float(np.linalg.norm(acc) / max(scale, 1e-300)))
    return dict(rows=out,
                pass_=bool(out["aqual"]["rel"] < 1e-3
                           and out["newton"]["rel"] < 1e-10))


if __name__ == "__main__":
    t0 = time.time()
    for fn in (g1_constant_K, g2_anisotropic_K, g3_symmetry,
               g4_flux_conservation, g5_aqual_1d, g6_momentum_null):
        t = time.time()
        r = fn()
        r["seconds"] = round(time.time() - t, 2)
        OUT[fn.__name__] = r
        print(f"{fn.__name__:24s} pass={r['pass_']}  {r['seconds']}s")
        print("   ", json.dumps({k: v for k, v in r.items()
                                 if k not in ("pass_", "seconds")})[:400])
    OUT["all_pass"] = all(v["pass_"] for k, v in OUT.items() if k != "all_pass")
    OUT["seconds"] = round(time.time() - t0, 2)
    json.dump(OUT, open("solver_gates.json", "w"), indent=1)
    print("ALL PASS:", OUT["all_pass"], OUT["seconds"], "s")
