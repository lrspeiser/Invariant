"""A second blindness theorem, found by the field-grammar engine itself.

When `field_grammar.py` first ran, the tensor basis elements I and ghat ghat^T
produced bit-comparable amplitudes on every scalar shape. That is not a numerical
coincidence and it is not a bug.

THEOREM (QUMOND tensor degeneracy).

In QUMOND form the field equation is

    laplacian Psi = div [ nu(|grad Phi_N|/a0) K grad Phi_N ]

and the response tensor K appears ONLY through the vector field K grad Phi_N.
Therefore any two tensors K_1, K_2 satisfying

    K_1 grad Phi_N = K_2 grad Phi_N   everywhere

give literally identical solutions. K is not observable; only its action on the
Newtonian field is.

COROLLARY. Let ghat = grad Phi_N / |grad Phi_N|. Then

    (ghat ghat^T) grad Phi_N = ghat (ghat . grad Phi_N)
                             = ghat |grad Phi_N|
                             = grad Phi_N
                             = I grad Phi_N

so the field-direction projector is EXACTLY degenerate with the identity. Adding
it to a grammar adds a duplicate column, not a new mechanism. More generally, for
any scalar field w, the atoms w*I and w*ghat ghat^T are the same atom.

WHAT SURVIVES. A tensor basis element is independent only if its action on
grad Phi_N differs from a scalar multiple of grad Phi_N -- that is, only if it
ROTATES the flux away from the Newtonian direction. The tidal tensor That and a
fixed structural axis dd do rotate it; the field-direction projector cannot, by
construction.

WHY THIS MATTERS BEYOND HOUSEKEEPING. Together with the spherical blindness
theorem it narrows what an anisotropic law can possibly be tested on:

    spherical source        transverse eigenvalue invisible  (any form)
    QUMOND form             only K grad Phi_N is observable
    => a mechanism is visible only if it TURNS the flux, on a NON-SPHERICAL
       source, and is measured with an observable sensitive to direction.

This module proves the degeneracy numerically rather than leaving it as algebra,
and measures how many atoms of the grammar it removes.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_grammar import (GPU, KPC, MSUN, FieldBank, sphericity, xp)  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))


def make_source(n=64, L_kpc=400.0, axis=(1.0, 0.75, 0.5), rs_kpc=80.0):
    L = L_kpc * KPC
    h = L / n
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    q = xp.sqrt((X / axis[0]) ** 2 + (Y / axis[1]) ** 2 + (Z / axis[2]) ** 2)
    rho = xp.exp(-(q / (rs_kpc * KPC)) ** 2)
    rho *= 1e14 * MSUN / float(rho.sum() * h ** 3)
    return rho, h


def independent_atoms(bank, tol=1e-8):
    """Group atoms into exact-duplicate classes by their response fields."""
    A = bank.A.reshape(len(bank.meta), -1)
    nrm = xp.sqrt((A ** 2).sum(axis=1)) + 1e-300
    An = A / nrm[:, None]
    Cm = An @ An.T
    C = Cm.get() if GPU else Cm
    n = C.shape[0]
    seen = -np.ones(n, dtype=int)
    groups = []
    for i in range(n):
        if seen[i] >= 0:
            continue
        members = [j for j in range(i, n)
                   if seen[j] < 0 and abs(abs(C[i, j]) - 1.0) < tol]
        for j in members:
            seen[j] = len(groups)
        groups.append(members)
    return groups, C


if __name__ == "__main__":
    print("=" * 78)
    print("QUMOND TENSOR DEGENERACY -- K is not observable, K grad Phi_N is")
    print("=" * 78)

    rho, h = make_source()
    print(f"\n   source axis ratio {sphericity(rho, h):.3f} (non-spherical)")
    bank = FieldBank(rho, h, dhat=(0, 0, 1), verbose=True)

    # --- the direct algebraic check, before any grouping
    gx, gy, gz = bank.gvec
    gm = bank.gmag
    ux, uy, uz = gx / gm, gy / gm, gz / gm
    # (ghat ghat^T) grad Phi_N, component by component
    vx = ux * (ux * gx + uy * gy + uz * gz)
    vy = uy * (ux * gx + uy * gy + uz * gz)
    vz = uz * (ux * gx + uy * gy + uz * gz)
    err = float(xp.max(xp.abs(vx - gx) + xp.abs(vy - gy) + xp.abs(vz - gz))
                / (float(xp.max(gm)) + 1e-300))
    print(f"\n   COROLLARY, checked directly:")
    print(f"      max | (ghat ghat^T - I) grad Phi_N | / max|grad Phi_N| "
          f"= {err:.3e}")
    print("      (this is round-off; the projector cannot turn the flux)")

    # --- how many atoms does the grammar actually have?
    groups, C = independent_atoms(bank)
    ndup = len(bank.meta) - len(groups)
    print(f"\n   atom bank: {len(bank.meta)} generated, "
          f"{len(groups)} INDEPENDENT, {ndup} exact duplicates")
    print("\n   the largest duplicate classes:")
    for g in sorted(groups, key=len, reverse=True)[:6]:
        if len(g) < 2:
            continue
        print(f"      {len(g)} identical atoms:")
        for j in g[:4]:
            print(f"         {bank.meta[j]}")
        if len(g) > 4:
            print(f"         ... and {len(g)-4} more")

    # --- which tensor basis elements survive?
    def basis_of(m):
        return m.split(" x ")[-1]

    surv = {}
    for g in groups:
        for j in g:
            surv.setdefault(basis_of(bank.meta[j]), set()).add(len(groups))
    counts = {}
    for gi, g in enumerate(groups):
        for j in g:
            counts.setdefault(basis_of(bank.meta[j]), set()).add(gi)
    print("\n   independent classes each tensor basis element appears in:")
    for b, s in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        print(f"      {b:6s}  {len(s)}")

    print("\n" + "=" * 78)
    print("   The grammar must drop ghat ghat^T. It is the identity in disguise")
    print("   whenever the tensor contracts only with grad Phi_N, which is what")
    print("   QUMOND form does. Keeping it would have inflated the apparent")
    print("   search space and put duplicate columns into every Gram.")

    out = {"n_generated": len(bank.meta), "n_independent": len(groups),
           "n_duplicates": ndup, "projector_residual": err,
           "classes_per_basis": {k: len(v) for k, v in counts.items()}}
    with open(os.path.join(HERE, "qumond_degeneracy.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n   written: qumond_degeneracy.json")
