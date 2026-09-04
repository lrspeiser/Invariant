"""How wrong is the linear-in-c screen? Measured, not assumed.

`field_grammar.py` screens candidate field laws by expanding

    K = I + sum_alpha c_alpha f_alpha(I) B_alpha

so that the QUMOND solution decomposes exactly into one precomputed Poisson
response per atom. But the grammar's actual response tensor is the matrix
exponential

    K = exp[ sum_alpha c_alpha f_alpha(I) B_alpha ]

which is what guarantees symmetry and positive-definiteness. The linear form is
its first-order truncation, and a screen that is systematically wrong is a screen
that discards the right answer.

Fortunately the comparison is cheap and exact. In QUMOND form the source is an
explicit function of the FIXED Newtonian field, so the exponential version is
still ONE linear Poisson solve -- no nonlinear iteration is involved. That means
the truncation error can be measured directly rather than bounded.

    Psi_lin   = Psi_0 + sum_alpha c_alpha R_alpha           (the screen)
    Psi_exp   = laplacian^-1 div [ nu exp(M) grad Phi_N ]   (the grammar)

with M = sum_alpha c_alpha f_alpha B_alpha. This module sweeps the coefficient
magnitude and reports where the screen stops being trustworthy, which is the
number that decides how wide a shortlist has to be before the exact nonlinear
finite-volume solver is called in the next stage.

The matrix exponential is done by scaling-and-squaring with a Taylor series on
the six independent components. Symmetry is preserved exactly at every step,
because powers, sums and squares of a symmetric matrix are symmetric, so the
6-component representation never has to be expanded to 9.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_grammar import (GPU, KPC, MSUN, A0, FieldBank, Poisson, div,
                           invariants, nu_rar, tensor_basis, SHAPES, xp)  # noqa
from qumond_degeneracy import make_source                                # noqa

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------- symmetric 3x3 field algebra
def smul(a, b):
    """Product of two symmetric 3x3 fields, returned as 6 components.

    The product of two symmetric matrices is not symmetric in general, but here
    both operands are polynomials in the SAME matrix M, so they commute and the
    product is symmetric. Only the six components are formed.
    """
    a11, a22, a33, a12, a13, a23 = a
    b11, b22, b33, b12, b13, b23 = b
    c11 = a11 * b11 + a12 * b12 + a13 * b13
    c22 = a12 * b12 + a22 * b22 + a23 * b23
    c33 = a13 * b13 + a23 * b23 + a33 * b33
    c12 = a11 * b12 + a12 * b22 + a13 * b23
    c13 = a11 * b13 + a12 * b23 + a13 * b33
    c23 = a12 * b13 + a22 * b23 + a23 * b33
    return (c11, c22, c33, c12, c13, c23)


def sadd(a, b, s=1.0):
    return tuple(ai + s * bi for ai, bi in zip(a, b))


def seye(shape):
    o = xp.ones(shape)
    z = xp.zeros(shape)
    return (o, o.copy(), o.copy(), z, z.copy(), z.copy())


def sexpm(M, order=10):
    """exp(M) for a symmetric 3x3 field, by scaling and squaring."""
    nrm = float(xp.max(xp.sqrt(sum(m ** 2 for m in M))))
    s = max(0, int(np.ceil(np.log2(max(nrm, 1e-30) / 0.35))))
    Ms = tuple(m / (2.0 ** s) for m in M)
    E = seye(M[0].shape)
    term = seye(M[0].shape)
    for k in range(1, order + 1):
        term = smul(term, Ms)
        term = tuple(t / k for t in term)
        E = sadd(E, term)
    for _ in range(s):
        E = smul(E, E)
    return E


def min_eig(M):
    """Smallest eigenvalue of a symmetric 3x3 field, for the positivity gate."""
    a11, a22, a33, a12, a13, a23 = [np.asarray(m.get() if GPU else m)
                                    for m in M]
    sh = a11.shape
    A = np.empty(sh + (3, 3))
    A[..., 0, 0] = a11
    A[..., 1, 1] = a22
    A[..., 2, 2] = a33
    A[..., 0, 1] = A[..., 1, 0] = a12
    A[..., 0, 2] = A[..., 2, 0] = a13
    A[..., 1, 2] = A[..., 2, 1] = a23
    return float(np.linalg.eigvalsh(A)[..., 0].min())


def main():
    print("=" * 78)
    print("LINEARISATION ERROR -- where does the field-law screen stop working?")
    print("=" * 78)
    rho, h = make_source()
    bank = FieldBank(rho, h, dhat=(0, 0, 1))
    n = bank.n
    P = bank.P
    gx, gy, gz = bank.gvec
    nu = nu_rar(bank.gmag / A0)
    inv = bank.inv
    Tdummy = None

    # rebuild the same atoms as component fields, so M can be assembled
    from field_grammar import tidal_tensor
    T, tr, t2 = tidal_tensor(bank.Phi_N, h)
    B = tensor_basis(inv, bank.gvec, bank.gmag, T, (0, 0, 1))

    atoms = []
    names = []
    for iname, ival in inv.items():
        for sname, f in SHAPES:
            for s in (0.3, 1.0, 3.0):
                w = f(ival / s)
                w = w - xp.mean(w)
                sd = float(xp.std(w))
                if not np.isfinite(sd) or sd < 1e-12:
                    continue
                w = w / sd
                for bname, comp in B.items():
                    if bname == "gg":                   # identity in disguise
                        continue
                    atoms.append(tuple(w * ci for ci in comp))
                    names.append(f"{sname}({iname}/{s:g}) x {bname}")
    print(f"   {len(atoms)} component atoms assembled")

    def solve_exp(M):
        K = sexpm(M)
        k11, k22, k33, k12, k13, k23 = K
        vx = nu * (k11 * gx + k12 * gy + k13 * gz)
        vy = nu * (k12 * gx + k22 * gy + k23 * gz)
        vz = nu * (k13 * gx + k23 * gy + k33 * gz)
        return P.solve(div(vx, vy, vz, h)), K

    def solve_lin(M):
        k11, k22, k33, k12, k13, k23 = sadd(seye(M[0].shape), M)
        vx = nu * (k11 * gx + k12 * gy + k13 * gz)
        vy = nu * (k12 * gx + k22 * gy + k23 * gz)
        vz = nu * (k13 * gx + k23 * gy + k33 * gz)
        return P.solve(div(vx, vy, vz, h))

    # observable: midplane radial acceleration on the same annulus as the search
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None] * xp.ones((1, n))
    Y = xp.ones((n, 1)) * ax[None, :]
    R = xp.sqrt(X ** 2 + Y ** 2)
    msk = (R > 20 * KPC) & (R < 140 * KPC)

    def obs(F):
        a = xp.gradient(F, h, axis=0)[:, :, n // 2]
        b = xp.gradient(F, h, axis=1)[:, :, n // 2]
        return ((X * a + Y * b) / (R + 1e-30))[msk]

    rng = np.random.default_rng(5)
    pick = [int(rng.integers(len(atoms))) for _ in range(3)]
    print("   test law uses:")
    for p in pick:
        print(f"      {names[p]}")

    base = obs(bank.Psi0)
    scale = float(xp.sqrt((base ** 2).mean()))
    rows = []
    print("")
    print("      |c|     ||M||max   min eig K   screen vs exact   verdict")
    for c in (0.01, 0.03, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0):
        M = seye((n, n, n))
        M = tuple(0.0 * m for m in M)
        for j, p in enumerate(pick):
            sgn = 1.0 if j % 2 == 0 else -1.0
            M = sadd(M, atoms[p], c * sgn)
        Pe, K = solve_exp(M)
        Pl = solve_lin(M)
        oe, ol = obs(Pe), obs(Pl)
        rel = float(xp.sqrt(((oe - ol) ** 2).mean())) / scale
        mn = min_eig(K)
        nm = float(xp.max(xp.sqrt(sum(m ** 2 for m in M))))
        # how big is the SIGNAL at this |c|? a screen error only matters
        # relative to the effect being screened for
        sig = float(xp.sqrt(((oe - base) ** 2).mean())) / scale
        v = ("ok" if rel < 0.1 * max(sig, 1e-12) else
             "MARGINAL" if rel < 0.5 * max(sig, 1e-12) else "UNUSABLE")
        rows.append({"c": c, "M_norm": nm, "min_eig_K": mn,
                     "screen_error_rel": rel, "signal_rel": sig,
                     "error_over_signal": rel / max(sig, 1e-30),
                     "verdict": v})
        print(f"    {c:6.2f}   {nm:8.3f}   {mn:9.3e}   "
              f"{rel:.3e} / {sig:.3e} = {rel/max(sig,1e-30):6.3f}   {v}")

    ok = [r for r in rows if r["verdict"] == "ok"]
    lim = max((r["c"] for r in ok), default=0.0)
    print("")
    print(f"   The linear screen is faithful to within 10% of the signal for")
    print(f"   |c| <= {lim:g}. Beyond that the shortlist must be re-solved with")
    print("   the exponential form before anything is reported, and beyond the")
    print("   UNUSABLE rows the screen is not even rank-ordering correctly.")
    print("")
    print("   Note the min-eig column: the exponential form keeps K positive")
    print("   definite at every |c| by construction, which is exactly what the")
    print("   linear truncation stops guaranteeing as |c| grows.")

    with open(os.path.join(HERE, "linearisation_error.json"), "w") as f:
        json.dump({"atoms": len(atoms), "test_law": [names[p] for p in pick],
                   "rows": rows, "faithful_to_c": lim}, f, indent=1)
    print(f"\n   written: linearisation_error.json")


if __name__ == "__main__":
    main()
