"""Where anisotropic response tensors can and cannot be tested.

Before spending compute on well-alignment and pair-channel tensors, there is a
structural question that decides the whole experimental design: which
observations can see them at all?

THEOREM (spherical blindness).

Let the field equation be

    div_i [ mu(X) K^ij grad_j Phi ] = 4 pi G rho,
    X = sqrt( grad_i Phi K^ij grad_j Phi ) / a0,

with rho = rho(r) spherically symmetric. The most general spherically symmetric
response tensor is

    K(x) = kappa_r(r) rhat rhat^T + kappa_t(r) ( I - rhat rhat^T ).

Then Phi(r) is determined by kappa_r alone, and kappa_t does not appear anywhere
in the solution.

    Proof.  Spherical symmetry gives grad Phi = Phi'(r) rhat, so
      K grad Phi = kappa_r Phi' rhat        (the transverse block annihilates it)
      X          = |Phi'| sqrt(kappa_r) / a0.
    The divergence of a radial field F(r) rhat is (1/r^2) d/dr (r^2 F), so the
    equation integrates exactly over a sphere:
      r^2 mu(X) kappa_r(r) Phi'(r) = G M(<r).
    kappa_t never enters.  QED

COROLLARY 1 -- what the well-alignment tensor reduces to.
For K = exp[s_0 I + s_T S] with S traceless, spherical symmetry forces
S = s(r) ( rhat rhat^T - I/3 ), the only traceless spherically symmetric form.
Then kappa_r = exp(s_0 + (2/3) s_T s) and kappa_t = exp(s_0 - (1/3) s_T s), so
ONLY the combination s_0 + (2/3) s_T s is observable. The anisotropy parameter
s_T is not separately measurable; it is degenerate with the isotropic part.

COROLLARY 2 -- a constant isotropic part is not observable at all.
If s_0 is constant then K = e^{s_0} I and the equation is identical to Newtonian
gravity with G -> G e^{-s_0}. Only the VARIATION of s_0 is physical.

COROLLARY 3 -- lensing does not rescue it.
With no gravitational slip, lensing deflection is built from the same Phi, which
Corollary 1 already shows depends only on the radial combination. A spherically
averaged lensing profile therefore cannot separate kappa_t either.

CONSEQUENCE FOR THE PROGRAMME.
Any test of a well-network or pair-channel tensor performed on spherically
averaged cluster data is VACUOUS with respect to the anisotropy it claims to
test: it measures an effective radial rescaling, which is degenerate with a
modified mu and with a modified G. The resolved, non-spherical configuration is
therefore not merely a better test. It is the ONLY thing that can see the
mechanism, which is exactly why the resolved-versus-scrambled comparison is the
right primary statistic.

This module states the theorem, then verifies it numerically on the validated
3-D solver rather than leaving it as algebra.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "gravitylab")))

import solver as S                                            # noqa: E402

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30


def radial_tensor(X, Y, Z, kappa_r, kappa_t, soft):
    """K = k_r rhat rhat^T + k_t (I - rhat rhat^T), as the six components.

    kappa_r and kappa_t are callables of r (in metres).
    """
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2 + soft ** 2)
    kr, kt = kappa_r(r), kappa_t(r)
    nx, ny, nz = X / r, Y / r, Z / r
    d = kr - kt
    return (kt + d * nx * nx, kt + d * ny * ny, kt + d * nz * nz,
            d * nx * ny, d * nx * nz, d * ny * nz)


def run(n=64, L_kpc=600.0, Mtot=1e14, rs_kpc=300.0, kt_scale=1.0, kr_scale=1.0):
    """Solve for a spherical source with a chosen (kappa_r, kappa_t)."""
    L = L_kpc * KPC
    h, ax, X, Y, Z = S.grids(n, L)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rs = rs_kpc * KPC
    # a smooth spherical source, compactly supported well inside the box
    rho = np.exp(-(r / rs) ** 2)
    rho *= Mtot * MSUN / (rho.sum() * h ** 3)

    def kr(rr):
        return kr_scale * (1.0 + 0.6 * np.exp(-(rr / rs) ** 2))

    def kt(rr):
        return kt_scale * (1.0 + 0.9 * np.tanh(rr / rs))

    A = radial_tensor(X, Y, Z, kr, kt, 0.5 * h)
    # far field uses the isotropic monopole with the asymptotic radial value,
    # which is what the operator sees at the boundary in spherical symmetry
    Kfar = np.eye(3) * kr(np.array([L]))[0]
    bc = S.far_field(X, Y, Z, rho.sum() * h ** 3, Kfar)
    Psi, it, rel = S.solve(rho, A, h, bc, tol=1e-10, maxiter=4000)
    return Psi, r, h, rel, it


#: fixed physical shell edges in kpc -- independent of the grid, so every
#: resolution is compared at the SAME radii. Inner edge clears the softening at
#: the coarsest grid; outer edge clears the Dirichlet shell.
FIXED_EDGES_KPC = np.linspace(80.0, 240.0, 17)


def profile(Psi, r, h, edges_kpc=None):
    """Shell-averaged |grad Psi| on FIXED physical shells."""
    gx = np.gradient(Psi, h, axis=0)
    gy = np.gradient(Psi, h, axis=1)
    gz = np.gradient(Psi, h, axis=2)
    g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    e = (FIXED_EDGES_KPC if edges_kpc is None else np.asarray(edges_kpc)) * KPC
    out = np.full(len(e) - 1, np.nan)
    for i in range(len(e) - 1):
        sel = (r >= e[i]) & (r < e[i + 1])
        if sel.sum() > 8:
            out[i] = g[sel].mean()
    return 0.5 * (e[1:] + e[:-1]), out


def resolution_study(ns=(32, 48, 64, 80)):
    """Is the residual transverse dependence discretisation, or is it real?

    The algebra is exact, so any transverse sensitivity on a Cartesian grid must
    be a discretisation artefact and must CONVERGE AWAY. A sensitivity that is
    flat in resolution would mean a modelling mismatch, not a numerical error --
    that distinction has already cost this programme a false solver bug once.
    """
    print("")
    print("   RESOLUTION STUDY -- transverse sensitivity vs grid spacing")
    print("      n     h [kpc]   transv max    transv med    "
          "radial max   ratio")
    rows = []
    for n in ns:
        pa, r, h, _, _ = run(n=n, kt_scale=0.5)
        pb, _, _, _, _ = run(n=n, kt_scale=1.5)
        pc, _, _, _, _ = run(n=n, kr_scale=1.5)
        _, ga = profile(pa, r, h)
        _, gb = profile(pb, r, h)
        _, gc = profile(pc, r, h)
        dt = float(np.nanmax(np.abs(gb - ga) / np.abs(ga)))
        dtm = float(np.nanmedian(np.abs(gb - ga) / np.abs(ga)))
        dr = float(np.nanmax(np.abs(gc - ga) / np.abs(ga)))
        rows.append((n, h / KPC, dt, dr, dtm))
        print(f"      {n:3d}   {h/KPC:7.2f}   {dt:.4e}  {dtm:.4e}   "
              f"{dr:.4e}   {dr/max(dt,1e-300):7.1f}")
    if len(rows) >= 2:
        hs = np.array([x[1] for x in rows])
        ds = np.array([x[2] for x in rows])
        ok = ds > 0
        if ok.sum() >= 2:
            p_ord = np.polyfit(np.log(hs[ok]), np.log(ds[ok]), 1)[0]
            print(f"      convergence order of the transverse sensitivity: "
                  f"{p_ord:.2f}")
            drop = ds[ok][-1] / ds[ok][0]
            hrat = hs[ok][0] / hs[ok][-1]
            print(f"      grid refined {hrat:.2f}x, sensitivity fell to "
                  f"{drop:.3f} of its coarsest value")
            print(f"      a genuine order-{p_ord:.2f} artefact would have fallen "
                  f"to {hrat**-p_ord:.3f}")
            print("      (a value that flattens toward a non-zero floor is a")
            print("       modelling mismatch, NOT a discretisation error)")
            return rows, float(p_ord)
    return rows, float("nan")


if __name__ == "__main__":
    print("=" * 78)
    print("SPHERICAL BLINDNESS -- can an anisotropic tensor be seen at all?")
    print("=" * 78)
    res = {}

    print("\n   TEST 1: vary the TRANSVERSE eigenvalue by a factor of 3,")
    print("           hold the radial one fixed. The theorem says nothing moves.")
    base = None
    rows = []
    for kt in (0.5, 1.0, 1.5):
        Psi, r, h, rel, it = run(kt_scale=kt)
        rr, g = profile(Psi, r, h)
        if base is None:
            base = g
        d = np.nanmax(np.abs(g - base) / np.abs(base))
        rows.append((kt, float(d), float(rel), int(it)))
        print(f"      kappa_t scale {kt:4.2f}   max |dg/g| vs baseline "
              f"{d:.3e}   (CG residual {rel:.1e}, {it} its)")
    res["transverse"] = rows

    print("\n   TEST 2: vary the RADIAL eigenvalue by the same factor.")
    print("           The theorem says this MUST move the solution.")
    base = None
    rows = []
    for kr in (0.5, 1.0, 1.5):
        Psi, r, h, rel, it = run(kr_scale=kr)
        rr, g = profile(Psi, r, h)
        if base is None:
            base = g
        d = np.nanmax(np.abs(g - base) / np.abs(base))
        rows.append((kr, float(d), float(rel), int(it)))
        print(f"      kappa_r scale {kr:4.2f}   max |dg/g| vs baseline "
              f"{d:.3e}   (CG residual {rel:.1e}, {it} its)")
    res["radial"] = rows

    t = max(r[1] for r in res["transverse"])
    q = max(r[1] for r in res["radial"])
    print("\n" + "=" * 78)
    print(f"   transverse sensitivity  max |dg/g| = {t:.3e}")
    print(f"   radial sensitivity      max |dg/g| = {q:.3e}")
    print(f"   ratio                                {q/max(t,1e-300):.3e}")
    print("")
    if t < 1e-3 * q:
        print("   CONFIRMED. The transverse eigenvalue is invisible in spherical")
        print("   symmetry, to solver accuracy. Any anisotropy test run on")
        print("   spherically averaged data measures a radial rescaling only, and")
        print("   is degenerate with a modified mu and with a modified G.")
        print("   The resolved non-spherical configuration is the ONLY probe.")
    else:
        print("   NOT confirmed at this resolution -- investigate before relying")
        print("   on the theorem. A non-zero transverse sensitivity here would")
        print("   most likely be a boundary-condition or softening artefact,")
        print("   since the algebra is exact; check both before believing it.")
    rows, order = resolution_study()
    res["resolution_study"] = rows
    res["transverse_convergence_order"] = order
    print("")
    tail = [r for r in rows if r[0] >= 48]
    tail_ratio = tail[-1][2] / tail[0][2] if len(tail) >= 2 else 1.0
    htail = tail[0][1] / tail[-1][1] if len(tail) >= 2 else 1.0
    print(f"   tail check: refining {htail:.2f}x over the finest grids changed"
          f" the transverse sensitivity by a factor {tail_ratio:.3f}")
    res["tail_ratio"] = tail_ratio
    if order > 0.7 and tail_ratio < 0.75:
        print(f"   CONFIRMED. The transverse sensitivity converges away at order")
        print(f"   {order:.2f}, so it is discretisation error on a Cartesian grid,")
        print("   not physics. The theorem holds: in spherical symmetry the")
        print("   transverse eigenvalue is unobservable, any anisotropy test on")
        print("   spherically averaged data is degenerate with a modified mu and")
        print("   a modified G, and only a RESOLVED non-spherical configuration")
        print("   can see the mechanism.")
    else:
        print(f"   The transverse sensitivity does NOT converge away: global "
              f"slope {order:.2f}, but")
        print(f"   the finest grids only move it by a factor {tail_ratio:.3f}. "
              "It is asymptoting to a")
        print("   NON-ZERO FLOOR, which is the signature of a modelling "
              "mismatch, not")
        print("   a numerical error. The algebra is exact, so something in the "
              "SETUP")
        print("   is not spherically symmetric. Do not rely on the theorem "
              "until this")
        print("   is explained.")
        print("   That is the signature of a modelling mismatch, not a numerical")
        print("   error. Do not rely on the theorem until this is explained.")
    res["verdict"] = {"transverse_max": t, "radial_max": q,
                      "convergence_order": order,
                      "tail_ratio": tail_ratio,
                      "confirmed": bool(order > 0.7 and tail_ratio < 0.75)}
    with open(os.path.join(HERE, "spherical_blindness.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n   written: spherical_blindness.json")
