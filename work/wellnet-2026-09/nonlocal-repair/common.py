"""Shared setup for the nonlocal-repair lane.

Import-order note (a real trap, recorded because it bit once): both
`work/gravitylab` and `work/wellnet-2026-09/nonlocal` contain a module called
`models.py`.  The nonlocal one must win, so it is imported FIRST and only then
is the gravitylab path appended for `data`.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = ("C:/Users/henry/Documents/Codex/2026-08-21/"
        "Invariant-main-integration/")
NL = ROOT + "work/wellnet-2026-09/nonlocal"
GLAB = ROOT + "work/gravitylab"
TENSOR = ROOT + "work/wellnet-2026-09/tensor"
SCREEN = ROOT + "work/wellnet-2026-09/screen"
HERE = os.path.dirname(os.path.abspath(__file__))

for p in (HERE, NL):
    if p not in sys.path:
        sys.path.insert(0, p)
import nonlocal_kernel as NK        # noqa: E402
import models as MO                 # noqa: E402  (the NONLOCAL models.py)
if GLAB not in sys.path:
    sys.path.append(GLAB)
import data as SP                   # noqa: E402

import dcore as DC                  # noqa: E402

G = NK.G
A0 = NK.A0                          # 1.2e-10 m/s^2 in (km/s)^2/kpc
AU_KPC = NK.AU_KPC


# --------------------------------------------------------------------------
#  SPARC: frozen split, declared cuts, declared nuisance treatment
# --------------------------------------------------------------------------
#  Nuisance treatment, identical for EVERY model compared in this lane:
#      V_bar^2 = |V_gas| V_gas + 0.5 V_disk^2 + 0.7 V_bul^2
#  i.e. Upsilon*_disk = 0.5, Upsilon*_bulge = 0.7 at 3.6 micron, no per-galaxy
#  freedom, no distance or inclination rescaling.  Both are held fixed across
#  Newton / RAR / AQUAL / kernel so the comparison is like for like.

UPS_DISK, UPS_BULGE = 0.5, 0.7


def vbar2(g):
    return (np.abs(g.Vgas) * g.Vgas + UPS_DISK * g.Vdisk ** 2
            + UPS_BULGE * g.Vbul ** 2)


def sparc(split="train", min_beyond=3, rcut=2.0):
    """Galaxies in one split with >= `min_beyond` points beyond rcut*R_disk."""
    gals = SP.ingest(verbose=False)
    SP.stratified_split(gals, verbose=False)
    sel = [g for g in gals if g.split == split]
    return [g for g in sel
            if g.Mb > 0 and g.Rdisk > 0
            and np.sum(g.R0 >= rcut * g.Rdisk) >= min_beyond]


DEFAULT_PROFILE = dict(n=1400, r_lo=1e-3, r_hi=3.0e4, tail_mult=1.0)


def build_profile(g, n=1400, r_lo=1e-3, r_hi=3.0e4, tail_mult=1.0):
    """Equivalent spherical baryon profile for one SPARC galaxy."""
    return MO.sparc_equivalent_sphere(
        g.R0, vbar2(g), g.Mb, tail_mult * max(g.Rdisk, 0.2),
        r_lo=r_lo, r_hi=r_hi, n=n)


DEFAULT_Q = dict(qdef="screen", rho_ref=1e6, L_q=2.0, L_s=0.0, m=1.0, n=1.0,
                 rho_floor=NK.RHO_BAR_B)

#: The two parameter sets the previous lane selected ON THE TRAIN SPLIT.
#: Frozen here; nothing in this lane re-selects them.
KERNEL_BEST = dict(tag="best-overall", qdef="screen", rho_ref=1e6, L_q=2.0,
                   Fname="F3_pade", alpha=10.0, beta=1.0, p=1.0)
KERNEL_LOCAL = dict(tag="best-passing-local", qdef="screen", rho_ref=1e6,
                    L_q=2.0, Fname="F1_poly", alpha=10.0, beta=0.0, p=2.0)

DEFAULT_QUAD = dict(n_D=32, n_s=12, n_gl=8, dlnr_max=0.35)
FINE_QUAD = dict(n_D=96, n_s=32, n_gl=14, dlnr_max=0.08)


def build_field(prof, qdef="screen", rho_ref=1e6, L_q=2.0, L_s=0.0, m=1.0,
                n=1.0, rho_floor=NK.RHO_BAR_B, label="", clip=None):
    """SphericalField from a profile tuple, with the requested q."""
    r, rho, Mr, rfun, Mfun, Mtot = prof
    if clip is None:
        return MO.build_field_from_profile(
            r, rho, Mr, rfun, Mfun, qdef=qdef, rho_ref=rho_ref, L_s=L_s,
            L_q=L_q, m=m, n=n, rho_floor=rho_floor, label=label)
    # smooth-clip variants live in clipmod.py
    import clipmod
    return clipmod.build_field_clipped(prof, rho_ref=rho_ref, L_s=L_s,
                                       rho_floor=rho_floor, label=label,
                                       **clip)


# --------------------------------------------------------------------------
#  What the data demand
# --------------------------------------------------------------------------

def required(g, Mtot, rcut=2.0):
    """(R, F_req, D_req, g_obs) at the measured radii beyond rcut*R_disk.

    F_req = -R Phi_obs/(G Mtot) with Phi_obs = -Int_R^inf v^2 dln r and the
    tail beyond the last point taken Keplerian.  D_req = R v_obs^2/(G Mtot) is
    the FORCE-space quantity and involves no integration at all -- it is the
    honest observable, and F_req is its integral.
    """
    R = g.R0
    v2 = g.Vobs0 ** 2
    lnr = np.log(R)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (v2[1:] + v2[:-1])
                                           * np.diff(lnr))])
    Phi = -((cum[-1] - cum) + v2[-1])
    F = -R * Phi / (G * Mtot)
    D = R * v2 / (G * Mtot)
    m = R >= rcut * g.Rdisk
    return R[m], F[m], D[m], (v2 / R)[m]


# --------------------------------------------------------------------------
#  Point-local competitors, in the SAME currency
# --------------------------------------------------------------------------

def g_newton(gbar):
    return np.asarray(gbar, float)


def g_rar(gbar, a0=A0):
    """McGaugh, Lelli & Schombert 2016 RAR: g = gN / (1 - exp(-sqrt(gN/a0))).

    The small-x branch is the ASYMPTOTIC one, g -> sqrt(gN a0) + gN/2: an
    earlier version returned a0 there, which is finite where the correct
    limit is zero, and it made the RAR look artificially good at points with
    a vanishing baryonic contribution.  Those points are excluded from the
    statistics anyway (see `gbar_positive`), but the branch is fixed so the
    function is right on its own terms.
    """
    gbar = np.maximum(np.asarray(gbar, float), 0.0)
    x = np.sqrt(gbar / a0)
    small = x < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, np.sqrt(gbar * a0) + 0.5 * gbar,
                    gbar / (1.0 - np.exp(-xs)))


def g_aqual_simple(gbar, a0=A0):
    """AQUAL/QUMOND with mu(x) = x/(1+x); in spherical symmetry the field
    equation is exactly algebraic, so this IS the AQUAL solution for the
    equivalent spherical model.  g^2/(g+a0) = gN."""
    gbar = np.maximum(np.asarray(gbar, float), 0.0)
    return 0.5 * (gbar + np.sqrt(gbar ** 2 + 4.0 * gbar * a0))


def g_aqual_standard(gbar, a0=A0):
    """mu(x) = x/sqrt(1+x^2):  g^2/sqrt(g^2+a0^2) = gN."""
    gbar = np.maximum(np.asarray(gbar, float), 0.0)
    y = gbar ** 2
    return np.sqrt(0.5 * (y + np.sqrt(y ** 2 + 4.0 * y * a0 ** 2)))


COMPETITORS = {
    "newton": g_newton,
    "RAR": g_rar,
    "AQUAL_simple": g_aqual_simple,
    "AQUAL_standard": g_aqual_standard,
}


def phi_from_g(R, gvec, v2_last=None):
    """Phi(R) = -Int_R^inf g dr with the tail beyond R[-1] taken Keplerian,
    i.e. g = g(R[-1]) (R[-1]/r)^2, so Int_{R[-1]}^inf g dr = g[-1] R[-1].

    Used to put every competitor into the SAME potential-space currency as
    the kernel's F_eff, so the two statistics can both be quoted.
    """
    R = np.asarray(R, float)
    gv = np.asarray(gvec, float)
    tail = gv[-1] * R[-1]
    seg = np.concatenate([[0.0], np.cumsum(0.5 * (gv[1:] + gv[:-1])
                                           * np.diff(R))])
    return -((seg[-1] - seg) + tail)
