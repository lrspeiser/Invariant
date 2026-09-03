"""Candidate acceleration laws for Run A, step 5.

Every model here takes the SAME inputs -- the Newtonian baryonic acceleration
on a radial grid, plus the galaxy's scalar properties -- and returns a
predicted acceleration. Global parameters are shared by every galaxy; nothing
in this file may take a per-galaxy gravity parameter.

Where the program's model is a PDE, the spherical/algebraic reduction used here
is stated in the docstring. Those reductions are exact in spherical symmetry
and are the reason Run A can screen without a solver; they are NOT a substitute
for Run B, and any model that survives here must be re-run on the PDE.
"""
from __future__ import annotations

import numpy as np

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30


# ----------------------------------------------------------------- mu families
def mu_simple(X):
    return X / (1.0 + X)


def mu_standard(X):
    return X / np.sqrt(1.0 + X * X)


# --------------------------------------------------------- diagnostic models
def D1_newton(gN, **kw):
    """Ordinary baryons. No free parameters."""
    return gN


def D2_aqual(gN, a0=1.2e-10, form="simple", **kw):
    """Scalar AQUAL. In spherical symmetry the field equation integrates to
    mu(g/a0) g = g_N exactly, which inverts in closed form.

      simple:    g^2 - g_N g - g_N a0 = 0
      standard:  u^2 - g_N^2 u - g_N^2 a0^2 = 0   with u = g^2
    """
    if form == "simple":
        return 0.5 * (gN + np.sqrt(gN * gN + 4.0 * gN * a0))
    u = 0.5 * (gN ** 2 + np.sqrt(gN ** 4 + 4.0 * gN ** 2 * a0 ** 2))
    return np.sqrt(u)


def D3_piecewise(gN, a0=1.2e-10, **kw):
    """Cylindrical-confinement target law. Not a field theory: it asserts the
    1/R asymptote and the Tully-Fisher normalisation and asks whether they are
    sufficient."""
    return np.where(gN >= a0, gN, np.sqrt(a0 * gN))


def D4_flatlog(gN, R_kpc=None, Mb_msun=None, a0=1.2e-10, c_rc=0.5, **kw):
    """Flattened logarithmic potential, midplane.

      Psi = (v_f^2/2) ln(r_c^2 + R^2 + z^2/q^2),   v_f^4 = G M_b a0
      V^2 = R dPsi/dR = v_f^2 R^2/(r_c^2 + R^2)
      g   = V^2/R

    r_c is set as c_rc * R_disk with c_rc GLOBAL, so no galaxy gets its own
    core radius.
    """
    vf4 = G * (Mb_msun * MSUN) * a0
    vf2 = np.sqrt(vf4)
    Rm = R_kpc * KPC
    rc = c_rc * kw.get("Rdisk_kpc", 1.0) * KPC
    V2 = vf2 * Rm ** 2 / (rc ** 2 + Rm ** 2)
    return V2 / Rm


def rar_reference(gN, a0=1.2e-10, **kw):
    """McGaugh, Lelli & Schombert 2016. Reference only -- not a candidate."""
    return gN / (1.0 - np.exp(-np.sqrt(gN / a0)))


# ------------------------------------------------------- void x scalar tensor
def scalar_void(gN, q, alpha=1.0, **kw):
    """K1 = exp(-alpha q) I. Spherical symmetry integrates exactly:

        exp(-alpha q) g r^2 = G M   =>   g = g_N exp(alpha q)

    Note this family is provably incapable of a flat rotation curve: q is
    bounded in (0,1] and tends to 1 far from the source, so g -> exp(alpha) g_N,
    an inverse-square law with a rescaled G. It is run anyway because the
    program's step 5 asks for it, and because the blind set should confirm the
    analytic result rather than take it on faith.
    """
    return gN * np.exp(alpha * q)


REGISTRY = {
    "D1_newton": dict(fn=D1_newton, params=()),
    "D2_aqual_simple": dict(fn=lambda gN, **k: D2_aqual(gN, form="simple", **k),
                            params=("a0",)),
    "D2_aqual_standard": dict(fn=lambda gN, **k: D2_aqual(gN, form="standard", **k),
                              params=("a0",)),
    "D3_piecewise": dict(fn=D3_piecewise, params=("a0",)),
    "D4_flatlog": dict(fn=D4_flatlog, params=("a0", "c_rc")),
}
