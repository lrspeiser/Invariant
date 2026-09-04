"""Billions of candidate FIELD LAWS on real 3-D mass maps, not more algebra.

THE PROBLEM WITH STAGING A FIELD-LAW SEARCH

Run J reached 77 million candidate laws per second because a candidate was a
subset of precomputed basis functions and its optimal coefficients came from a
gathered sub-matrix of a Gram computed once. That works only when the map from
coefficients to prediction is LINEAR.

A field law is not algebra. Scoring one means solving

    div_i [ mu(X) K^ij(I) grad_j Phi ] = 4 pi G rho_b

which is a nonlinear elliptic PDE, seconds per candidate on a 128^3 grid. At that
rate a "billion-law search" over field equations is roughly thirty years, which
is why the staged funnel exists at all.

THE TRICK THAT COLLAPSES IT

Write the law in QUMOND form. The source is then an EXPLICIT pointwise function
of the Newtonian field, which is computed once and never again:

    laplacian Psi = div [ nu(|grad Phi_N|/a0) K(I) grad Phi_N ]

Now expand the response tensor in the grammar's own basis, linearly in the
coefficients:

    K(I) = I + sum_alpha c_alpha f_alpha(I) B_alpha

Because div is linear and the inverse Laplacian is linear, the solution
decomposes exactly:

    Psi = Psi_0 + sum_alpha c_alpha R_alpha,
    Psi_0    = laplacian^-1 div [ nu grad Phi_N ]
    R_alpha  = laplacian^-1 div [ nu f_alpha(I) B_alpha grad Phi_N ]

So ONE Poisson solve per ATOM -- not per candidate. After N_atom solves, every
sparse subset of atoms, with its optimal coefficients, is an instant linear
combination, and the Run J Gram machinery applies verbatim to the observables.

    cost:  N_atom FFT Poisson solves, once
    then:  every candidate law is O(K^2), exactly as in Run J

That is the same architectural move as Run J's atom bank, lifted from algebraic
laws on rotation curves to field laws on resolved three-dimensional mass maps.
The exponential form K = exp[sum c f B] is recovered to first order in c, and the
funnel's later stages re-solve the shortlist EXACTLY with the validated nonlinear
finite-volume solver -- the linear expansion is a screen, never a verdict.

WHAT THIS DOES AND DOES NOT LICENSE

It licenses screening ~1e9 sparse field laws against resolved sources. It does
NOT license reporting any of them: a linear-in-c screen has its own systematic
error against the exact nonlinear solve, and that error is measured here rather
than assumed. Anything that survives goes to the exact solver.

It also inherits the spherical blindness theorem (see spherical_blindness.py):
for a spherically symmetric source the transverse eigenvalue of K is exactly
invisible, so this engine can only see anisotropy on RESOLVED, NON-SPHERICAL
configurations. Running it on a smooth spherical cluster would return a
degenerate answer, and the module refuses to do so silently.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:                                    # pragma: no cover
    import numpy as xp
    GPU = False

HERE = os.path.dirname(os.path.abspath(__file__))

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
A0 = 1.2e-10


def sync():
    if GPU:
        xp.cuda.Stream.null.synchronize()


# ------------------------------------------------------- isolated FFT Poisson
class Poisson:
    """Isolated-boundary Poisson solver by zero-padded FFT.

    The source is embedded in a box of twice the linear size so the periodic
    wrap of the FFT never reaches back into the region of interest. This is the
    standard James/Hockney trick and it gives genuinely open boundaries, which
    matters here because the whole programme has already been bitten once by a
    zero-flux boundary that silently manufactured a compensating background.
    """

    def __init__(self, n, h):
        self.n, self.h = n, h
        m = 2 * n
        k = 2.0 * math.pi * xp.fft.fftfreq(m, d=h)
        kx = k[:, None, None]
        ky = k[None, :, None]
        kz = k[None, None, :]
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2[0, 0, 0] = 1.0
        self.inv = (-1.0 / k2).astype(xp.float64)
        self.inv[0, 0, 0] = 0.0                       # drop the mean mode
        self.m = m

    def solve(self, f):
        """Return u with laplacian u = f, isolated boundaries."""
        m, n = self.m, self.n
        pad = xp.zeros((m, m, m), dtype=xp.float64)
        pad[:n, :n, :n] = f
        u = xp.fft.ifftn(xp.fft.fftn(pad) * self.inv).real
        return u[:n, :n, :n]


def div(vx, vy, vz, h):
    """Centred divergence, matching the gradient used to build the field."""
    return (xp.gradient(vx, h, axis=0) + xp.gradient(vy, h, axis=1)
            + xp.gradient(vz, h, axis=2))


def grad(u, h):
    return (xp.gradient(u, h, axis=0), xp.gradient(u, h, axis=1),
            xp.gradient(u, h, axis=2))


# ------------------------------------------------------------- the invariants
def tidal_tensor(Phi, h):
    """T_ij = d_i d_j Phi, and its normalised form and invariants."""
    gx, gy, gz = grad(Phi, h)
    T = {}
    for i, gi in enumerate((gx, gy, gz)):
        for j, ax in enumerate((0, 1, 2)):
            if j < i:
                continue
            T[(i, j)] = xp.gradient(gi, h, axis=ax)
    # symmetrise the mixed second derivatives, which differ at O(h^2)
    for (i, j) in list(T):
        if i != j:
            T[(j, i)] = T[(i, j)]
    tr = T[(0, 0)] + T[(1, 1)] + T[(2, 2)]
    # T_ij T^ij
    t2 = sum(T[(i, j)] ** 2 for i in range(3) for j in range(3))
    return T, tr, t2


def invariants(Phi_N, rho, h, L_kpc=10.0):
    """The dimensionless scalar invariants the grammar is allowed to use."""
    gx, gy, gz = grad(Phi_N, h)
    gmag = xp.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-30
    T, tr, t2 = tidal_tensor(Phi_N, h)
    T0 = float(xp.sqrt(xp.mean(t2))) + 1e-300
    inv = {
        "x_g":   gmag / A0,
        "x_Phi": xp.abs(Phi_N) / (float(xp.max(xp.abs(Phi_N))) + 1e-300),
        "x_rho": rho / (float(xp.mean(rho)) + 1e-300),
        "x_T":   xp.sqrt(t2) / T0,
    }
    # a nonlocal screened response at a free length, the cheap surrogate of
    # (1 - L^2 laplacian)^-1 acting on the source
    n = rho.shape[0]
    kk = 2.0 * math.pi * xp.fft.fftfreq(n, d=h)
    k2 = (kk[:, None, None] ** 2 + kk[None, :, None] ** 2
          + kk[None, None, :] ** 2)
    Lm = L_kpc * KPC
    q = xp.fft.ifftn(xp.fft.fftn(rho) / (1.0 + (Lm ** 2) * k2)).real
    inv["q_L"] = q / (float(xp.mean(xp.abs(q))) + 1e-300)
    return inv, (gx, gy, gz), gmag, T


def tensor_basis(inv, gvec, gmag, T, dhat):
    """The tensor basis B_alpha, each returned as six symmetric components."""
    gx, gy, gz = gvec
    n = gx.shape
    one = xp.ones(n)
    zero = xp.zeros(n)
    B = {}
    B["I"] = (one, one, one, zero, zero, zero)
    # normalised tidal tensor
    nrm = xp.sqrt(sum(T[(i, j)] ** 2 for i in range(3) for j in range(3))) + 1e-300
    B["That"] = (T[(0, 0)] / nrm, T[(1, 1)] / nrm, T[(2, 2)] / nrm,
                 T[(0, 1)] / nrm, T[(0, 2)] / nrm, T[(1, 2)] / nrm)
    # ghat ghat^T -- the direction of the local field itself
    ux, uy, uz = gx / gmag, gy / gmag, gz / gmag
    B["gg"] = (ux * ux, uy * uy, uz * uz, ux * uy, ux * uz, uy * uz)
    # a fixed structural axis (e.g. the disk normal or a cluster's major axis)
    d = np.asarray(dhat, float)
    d = d / np.linalg.norm(d)
    B["dd"] = (one * d[0] * d[0], one * d[1] * d[1], one * d[2] * d[2],
               one * d[0] * d[1], one * d[0] * d[2], one * d[1] * d[2])
    return B


#: scalar shape functions applied to an invariant
SHAPES = [
    ("id",      lambda u: u),
    ("log1p",   lambda u: xp.log10(1.0 + xp.abs(u))),
    ("inv1p",   lambda u: 1.0 / (1.0 + xp.abs(u))),
    ("expneg",  lambda u: xp.exp(-xp.minimum(xp.abs(u), 60.0))),
    ("sqrt",    lambda u: xp.sqrt(xp.abs(u))),
    ("tanh",    lambda u: xp.tanh(u)),
]


def nu_rar(x):
    return 1.0 / (1.0 - xp.exp(-xp.sqrt(xp.maximum(x, 1e-30))))


# --------------------------------------------------------------- the engine
class FieldBank:
    """Precomputed response fields: one Poisson solve per ATOM, then free."""

    def __init__(self, rho, h, dhat=(0, 0, 1), scales=(0.3, 1.0, 3.0),
                 L_kpc=10.0, verbose=True):
        self.h, self.n = h, rho.shape[0]
        self.rho = xp.asarray(rho, dtype=xp.float64)
        self.P = Poisson(self.n, h)
        t0 = time.time()
        self.Phi_N = self.P.solve(4.0 * math.pi * G * self.rho)
        inv, gvec, gmag, T = invariants(self.Phi_N, self.rho, h, L_kpc)
        self.inv, self.gvec, self.gmag = inv, gvec, gmag
        B = tensor_basis(inv, gvec, gmag, T, dhat)
        nu = nu_rar(gmag / A0)
        gx, gy, gz = gvec

        def apply(comp):
            """Poisson^-1 div [ nu * (K comp) grad Phi_N ] for a symmetric K."""
            axx, ayy, azz, axy, axz, ayz = comp
            vx = nu * (axx * gx + axy * gy + axz * gz)
            vy = nu * (axy * gx + ayy * gy + ayz * gz)
            vz = nu * (axz * gx + ayz * gy + azz * gz)
            return self.P.solve(div(vx, vy, vz, h))

        # the QUMOND baseline: K = I
        self.Psi0 = apply(B["I"])
        self.atoms, self.meta = [], []
        for iname, ival in inv.items():
            for sname, f in SHAPES:
                for s in scales:
                    w = f(ival / s)
                    w = w - xp.mean(w)                 # remove the K=I part
                    sd = float(xp.std(w))
                    if not np.isfinite(sd) or sd < 1e-12:
                        continue
                    w = w / sd
                    for bname, comp in B.items():
                        c = tuple(w * ci for ci in comp)
                        R = apply(c)
                        if not bool(xp.isfinite(R).all()):
                            continue
                        self.atoms.append(R)
                        self.meta.append(f"{sname}({iname}/{s:g}) x {bname}")
        self.A = xp.stack(self.atoms) if self.atoms else None
        if verbose:
            print(f"   field bank: {len(self.atoms)} atoms, "
                  f"{len(self.atoms)+1} Poisson solves, "
                  f"{time.time()-t0:.1f}s on a {self.n}^3 grid "
                  f"({'GPU' if GPU else 'CPU'})")

    # ------------------------------------------------------------ observables
    def midplane_vc(self, iz=None):
        """Circular speed in the z = n/2 plane, for Psi0 and each atom."""
        n, h = self.n, self.h
        iz = n // 2 if iz is None else iz
        ax = (xp.arange(n) - n / 2 + 0.5) * h
        X = ax[:, None] * xp.ones((1, n))
        Y = xp.ones((n, 1)) * ax[None, :]
        R = xp.sqrt(X ** 2 + Y ** 2) + 1e-30

        def vc(F):
            gx = xp.gradient(F, h, axis=0)[:, :, iz]
            gy = xp.gradient(F, h, axis=1)[:, :, iz]
            return (X * gx + Y * gy) / R          # radial acceleration

        return vc(self.Psi0), xp.stack([vc(a) for a in self.atoms])

    def deflection(self, axis=2):
        """Projected lensing deflection: integral of the transverse gradient."""
        h = self.h

        def alpha(F):
            S = F.sum(axis=axis) * h
            a1 = xp.gradient(S, h, axis=0)
            a2 = xp.gradient(S, h, axis=1)
            return xp.sqrt(a1 ** 2 + a2 ** 2)

        return alpha(self.Psi0), xp.stack([alpha(a) for a in self.atoms])


def sphericity(rho, h):
    """How far from spherical is this source? The blindness theorem needs it."""
    n = rho.shape[0]
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    M = float(rho.sum())
    I = np.array([[float((rho * a * b).sum()) / M for b in (X, Y, Z)]
                  for a in (X, Y, Z)])
    w = np.linalg.eigvalsh(I)
    w = np.sort(np.abs(w))[::-1]
    return float(np.sqrt(w[-1] / w[0]))          # minor/major axis ratio


if __name__ == "__main__":
    print("=" * 78)
    print("FIELD GRAMMAR -- one Poisson solve per ATOM, then billions of laws")
    print("=" * 78)
    n = 64
    L = 400.0 * KPC
    h = L / n
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]

    # a deliberately NON-spherical source: a flattened triaxial cluster, which
    # is the only configuration the spherical blindness theorem permits us to
    # learn anything about anisotropy from
    rs = 80.0 * KPC
    q = xp.sqrt((X / 1.0) ** 2 + (Y / 0.75) ** 2 + (Z / 0.5) ** 2)
    rho = xp.exp(-(q / rs) ** 2)
    rho *= 1e14 * MSUN / float(rho.sum() * h ** 3)

    ar = sphericity(rho, h)
    print(f"\n   source axis ratio (minor/major) = {ar:.3f}")
    if ar > 0.97:
        print("   REFUSING: this source is spherical to within 3%. By the")
        print("   spherical blindness theorem the transverse eigenvalue is")
        print("   unobservable and any anisotropy result would be degenerate")
        print("   with a modified mu and a modified G.")
        sys.exit(1)
    print("   non-spherical, so the anisotropic atoms are observable here.\n")

    bank = FieldBank(rho, h, dhat=(0, 0, 1))
    v0, va = bank.midplane_vc()
    d0, da = bank.deflection()
    print(f"   observables: midplane radial acceleration {tuple(v0.shape)}, "
          f"projected deflection {tuple(d0.shape)}")

    # how much does each atom actually MOVE the observable? an atom that moves
    # nothing is not a candidate term, it is a null direction of the experiment
    amp = xp.sqrt((va ** 2).mean(axis=(1, 2))) / (
        float(xp.sqrt((v0 ** 2).mean())) + 1e-300)
    a = amp.get() if GPU else amp
    o = np.argsort(a)[::-1]
    print("\n   the ten atoms that move the rotation observable most:")
    for i in o[:10]:
        print(f"      {a[i]:8.4f}   {bank.meta[i]}")
    print("\n   the five that move it least (null directions of this probe):")
    for i in o[-5:]:
        print(f"      {a[i]:8.4e}   {bank.meta[i]}")

    dead = int((a < 1e-6).sum())
    print(f"\n   {dead}/{len(a)} atoms are invisible to this observable "
          f"(< 1e-6 relative)")
    print("   Those are not weak terms. They are directions this experiment")
    print("   cannot constrain at all, and fitting them would be fitting noise.")

    out = {"n": n, "L_kpc": L / KPC, "axis_ratio": ar,
           "n_atoms": len(bank.meta),
           "atom_amplitudes": {bank.meta[i]: float(a[i])
                               for i in range(len(a))},
           "dead_atoms": dead}
    with open(os.path.join(HERE, "field_grammar.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   written: field_grammar.json")
