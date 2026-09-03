"""The mandatory numerical gates of section 6 of the test program.

"These are mandatory. A model that fails them should never reach the fitting
stage." Each gate below is the program's own criterion, run against the
finite-volume solver with open (Dirichlet far-field) boundaries.

Three findings from building these are worth keeping, because in each case the
first version of the TEST was wrong rather than the solver:

  * Zero-flux boundaries are self-contradictory for an isolated source. Gauss's
    law over a surface enclosing all the mass demands 4 pi G M of flux, but a
    sealed box forces zero, so the solver manufactures a compensating uniform
    background. It cost three gates at once.
  * Flux conservation must be measured on the FACE fluxes the discretisation
    actually conserves. Re-deriving a flux from a centre-differenced gradient
    is a different quantity, agreeing only to O(h^2); it reported eps ~ 1e-2 on
    a solver that conserves to 1e-15.
  * The analytic comparison needs a source spherical in u = sqrt(r^T K^-1 r),
    not in r. A sphere in r is an ellipsoid in u and carries a u-space
    quadrupole falling only as (sigma/u)^2, flooring the error near 1e-2 at
    EVERY resolution -- the signature of a modelling mismatch, not a
    discretisation error.
"""
from __future__ import annotations

import math

import numpy as np

import solver as S

G = 6.674e-11
BAR = "=" * 78
results = []


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail.splitlines():
        print(f"          {line}")


def sphere_source(n, L, Mtot, soft_frac=0.045):
    h, ax, X, Y, Z = S.grids(n, L)
    eps = soft_frac * L
    rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * eps ** 2))
    rho *= Mtot / (rho.sum() * h ** 3)
    return h, X, Y, Z, rho, eps


def setup(n, L, Mtot, kpar, kperp, e=(0, 0, 1)):
    h, X, Y, Z, rho, eps = sphere_source(n, L, Mtot)
    A, M = S.axis_tensor(rho.shape, e, kpar, kperp)
    bc = S.far_field(X, Y, Z, Mtot, M)
    return h, X, Y, Z, rho, eps, A, M, bc


def gate_dimensions():
    head("6.1  Dimensional consistency")
    import qfield as Q
    R = np.geomspace(0.5, 30, 20)
    gN = 1e-10 * (R / 10) ** -2
    checks = [("rho_L/rho_c", Q.q_rho(gN, R, rho_c=1e-24, m=1.0)),
              ("|g_N|/a0", Q.q_g(gN, a0=1.2e-10, n=1.0)),
              ("L_q^2 grad^2", Q.q_nonlocal(gN, R, L_q_kpc=5.0))]
    ok = True
    for nm, q in checks:
        good = bool(np.all(np.isfinite(q)) and np.all((q > 0) & (q <= 1)))
        ok &= good
        print(f"      {nm:<24} q in ({q.min():.4f}, {q.max():.4f}]  "
              f"{'ok' if good else 'FAILED'}")
    record("all interpolation arguments dimensionless, q bounded in (0,1]",
           ok, "every void field stays inside the program's stated range")


def gate_positive_definite():
    head("6.2  Positive-definite tensor")
    worst, ok = 1e9, True
    for kpar, kperp in ((1.0, 1.0), (2.0, 0.5), (0.3, 1.7), (1e-3, 1.0)):
        _, M = S.axis_tensor((2, 2, 2), (0, 0, 1), kpar, kperp)
        lam = np.linalg.eigvalsh(M)
        worst = min(worst, float(lam.min()))
        ok &= bool(lam.min() > 1e-6)
        print(f"      k_par={kpar:<6} k_perp={kperp:<5} eigenvalues "
              f"{lam[0]:.4f} {lam[1]:.4f} {lam[2]:.4f}")
    record("eigenvalues of K exceed 1e-6", ok,
           f"smallest eigenvalue seen: {worst:.3e}   threshold 1e-6")


def gate_constant_tensor():
    head("6.4  Constant-K analytic test")
    print("   Psi = -GM / ( sqrt(det K) sqrt(r^T K^-1 r) )")
    print("   Source spherical in u, so no u-space quadrupole and the")
    print("   exterior potential is exactly the monopole.")
    print("")
    Mtot, L, kpar, kperp = 1.0e40, 40.0, 2.0, 0.6
    rows = []
    for n in (32, 48, 72):
        h, ax, X, Y, Z = S.grids(n, L)
        A, M = S.axis_tensor((n, n, n), (0, 0, 1), kpar, kperp)
        Kinv = np.linalg.inv(M)
        u = np.sqrt(Kinv[0, 0] * X ** 2 + Kinv[1, 1] * Y ** 2
                    + Kinv[2, 2] * Z ** 2)
        sig_u = 0.040 * L
        rho = np.exp(-u ** 2 / (2 * sig_u ** 2))
        rho *= Mtot / (rho.sum() * h ** 3)
        bc = S.far_field(X, Y, Z, Mtot, M)
        Psi, it, rel = S.solve(rho, A, h, bc, tol=1e-12, maxiter=9000)
        exact = S.far_field(X, Y, Z, Mtot, M)
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        m = (u > 5 * sig_u) & (r < 0.36 * L)
        err = float(np.sqrt(np.mean((Psi[m] - exact[m]) ** 2))
                    / np.sqrt(np.mean(exact[m] ** 2)))
        rows.append((n, err))
        print(f"      n={n:<4} h={h:5.2f}  rel L2 = {err:.4e}  "
              f"CG iters={it}  resid={rel:.1e}")
    order = (math.log(rows[0][1] / rows[-1][1])
             / math.log(rows[-1][0] / rows[0][0]))
    record("matches the analytic solution, 2nd order or better",
           bool(order > 1.8 and rows[-1][1] < 5e-2),
           f"convergence order: {order:.2f}  (required > 1.8)\n"
           f"finest-grid relative error: {rows[-1][1]:.4e}")


def gate_flux():
    head("6.3  Flux conservation")
    print("   closed-surface flux of A grad Psi vs 4 pi G M enclosed")
    print("")
    n, L, Mtot = 64, 40.0, 1.0e40
    h, X, Y, Z, rho, eps, A, M, bc = setup(n, L, Mtot, 1.7, 0.55)
    Psi, it, rel = S.solve(rho, A, h, bc, tol=1e-13, maxiter=12000)
    Fx, Fy, Fz = S.flux_faces(Psi, A, h)
    worst = 0.0
    c = n // 2
    for half in (10, 14, 18, 22):
        s = slice(c - half, c + half)
        Menc = rho[s, s, s].sum() * h ** 3
        f = (Fx[c + half - 1, s, s].sum() - Fx[c - half - 1, s, s].sum()
             + Fy[s, c + half - 1, s].sum() - Fy[s, c - half - 1, s].sum()
             + Fz[s, s, c + half - 1].sum() - Fz[s, s, c - half - 1].sum())
        f *= h ** 2
        e = abs(f - 4 * math.pi * G * Menc) / (4 * math.pi * G * Menc)
        worst = max(worst, e)
        print(f"      half-width {half:>3}   M_enc/M = {Menc/Mtot:6.4f}   "
              f"eps_flux = {e:.3e}")
    record("eps_flux < 1e-5 on every closed surface", bool(worst < 1e-5),
           f"worst surface: {worst:.3e}   threshold 1e-5")


def gate_curl():
    head("6.5  Curl test")
    vals = []
    for n in (32, 48, 72):
        h, X, Y, Z, rho, eps, A, M, bc = setup(n, 40.0, 1.0e40, 2.2, 0.5,
                                               e=(0.0, 0.6, 0.8))
        Psi, _, _ = S.solve(rho, A, h, bc, tol=1e-11, maxiter=8000)
        gx, gy, gz = (-np.gradient(Psi, h, axis=i) for i in range(3))
        cx = np.gradient(gz, h, axis=1) - np.gradient(gy, h, axis=2)
        cy = np.gradient(gx, h, axis=2) - np.gradient(gz, h, axis=0)
        cz = np.gradient(gy, h, axis=0) - np.gradient(gx, h, axis=1)
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        m = (r > 4 * eps) & (r < 0.35 * 40.0)
        scale = np.sqrt(np.mean(gx[m] ** 2 + gy[m] ** 2 + gz[m] ** 2)) / h
        cur = np.sqrt(np.mean(cx[m] ** 2 + cy[m] ** 2 + cz[m] ** 2)) / scale
        vals.append(cur)
        print(f"      n={n:<4} normalised |curl| = {cur:.4e}")
    record("curl stays at round-off and does not grow",
           bool(vals[-1] < 1e-10),
           f"finest grid: {vals[-1]:.3e}  (round-off is ~1e-16)")


def gate_newtonian():
    head("6.6  Newtonian recovery")
    errs = []
    for n in (32, 48, 72):
        h, X, Y, Z, rho, eps, A, M, bc = setup(n, 40.0, 1.0e40, 1.0, 1.0)
        Psi, _, _ = S.solve(rho, A, h, bc, tol=1e-12, maxiter=9000)
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        exact = -G * 1.0e40 / np.maximum(r, 1e-9)
        m = (r > 4 * eps) & (r < 0.40 * 40.0)
        err = float(np.sqrt(np.mean((Psi[m] - exact[m]) ** 2))
                    / np.sqrt(np.mean(exact[m] ** 2)))
        errs.append(err)
        print(f"      n={n:<4} rel L2 vs -GM/r = {err:.4e}")
    order = math.log(errs[0] / errs[-1]) / math.log(72 / 32)
    record("K -> I recovers the Newtonian potential", bool(errs[-1] < 2e-2),
           f"finest grid: {errs[-1]:.4e}, convergence order {order:.2f}")


def gate_domain():
    head("6.7  Domain and boundary convergence")
    print("   same physical source, progressively larger boxes, h fixed")
    print("")
    Mtot, probe_r = 1.0e40, 5.0
    vals = []
    for L, n in ((20.0, 40), (40.0, 80), (80.0, 160)):
        h, ax, X, Y, Z = S.grids(n, L)
        A, M = S.axis_tensor((n, n, n), (0, 0, 1), 1.6, 0.6)
        rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * 1.10 ** 2))
        rho *= Mtot / (rho.sum() * h ** 3)
        bc = S.far_field(X, Y, Z, Mtot, M)
        Psi, _, _ = S.solve(rho, A, h, bc, tol=1e-11, maxiter=9000)
        gx, gy, gz = (-np.gradient(Psi, h, axis=i) for i in range(3))
        gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        sel = np.abs(r - probe_r) < max(1.2 * h, 0.35)
        v = float(np.sqrt(np.mean(gm[sel]) * probe_r))
        vals.append(v)
        print(f"      L={L:6.1f}  n={n:<4} h={h:5.3f}   "
              f"V(r={probe_r}) = {v:.6e}")
    dev = max(abs(v - vals[0]) / vals[0] for v in vals)
    record("velocity at fixed radius stable to 0.5% across box size",
           bool(dev < 0.005),
           f"largest fractional change: {100*dev:.3f}%   threshold 0.5%")


if __name__ == "__main__":
    print(BAR)
    print("MANDATORY NUMERICAL GATES -- section 6 of the test program")
    print(BAR)
    gate_dimensions()
    gate_positive_definite()
    gate_constant_tensor()
    gate_flux()
    gate_curl()
    gate_newtonian()
    gate_domain()
    head("SUMMARY")
    npass = sum(1 for _, ok, _ in results if ok)
    for nm, ok, _ in results:
        print(f"   [{'PASS' if ok else 'FAIL'}] {nm}")
    print(f"\n   {npass} of {len(results)} gates pass.")
    print("   Solver cleared for Run B." if npass == len(results)
          else "   The solver is NOT cleared for the fitting stage.")
