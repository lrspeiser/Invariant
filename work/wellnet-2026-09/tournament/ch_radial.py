"""CHANNEL 1 -- radial rotation.  SPARC, frozen split e5f74522, TRAIN ONLY.

Metric: RMS of log10(g_pred/g_obs) over every retained radial point of every
TRAIN galaxy, with g_obs = V_obs^2/R and g_bar from the tabulated components at
DECLARED mass-to-light ratios (disk 0.5, bulge 0.7, gas 1.0 with the 1.33
helium factor already in V_gas).  No per-galaxy freedom of any kind: distance
and inclination are held at their catalogue values, which is what makes this
comparable across candidates and is why the number is larger than the 0.11 dex
quoted for the RAR with per-galaxy nuisances marginalised.  The RAR's value
UNDER THIS PROTOCOL is measured here and is the bar.

The blind and validation galaxies are never loaded into any scoring array.

OPERATIONAL RULE FOR |Phi_N|, declared before any residual is examined.
    "inf"      PRIMARY.  Phi -> 0 at infinity.  |Phi_N|(r) = int_r^Rlast g_bar
               dr' + G M_b(<Rlast)/Rlast, i.e. the baryon distribution is
               continued outside the last measured point as a point mass.
               This is a GLOBAL prescription: it names no object-specific
               reference radius, so it satisfies the programme's rule that a
               law gets universal constants only.
    "flat"     Global alternative.  The outer continuation is a flat rotation
               curve V_flat truncated at a UNIVERSAL R_trunc = 1 Mpc:
               tail = V_flat^2 ln(R_trunc/Rlast) + G M_b/R_trunc.
    "last"     SENSITIVITY ONLY.  Reference at the object's own last radius.
    "half"     SENSITIVITY ONLY.  |Phi| = G M_b/max(r, R_eff).
The last two are object-relative and therefore violate the global-parameter
rule; they are carried to show how far the answer moves, never as primaries.
"""
from __future__ import annotations

import os
import sys

import numpy as np

GRAVLAB = ("C:/Users/henry/Documents/Codex/2026-08-21/"
           "Invariant-main-integration/work/gravitylab")
if GRAVLAB not in sys.path:
    sys.path.insert(0, GRAVLAB)

import data as SP                                              # noqa: E402
from tw_core import G, KPC, MSUN, A0, g_response, W_cand        # noqa: E402

UPS_DISK = 0.5
UPS_BULGE = 0.7
R_TRUNC = 1000.0 * KPC
PHI_RULES = ("inf", "flat", "last", "half")


def _phi_profiles(r_m, gbar, Vflat_ms, Mb_tot, Reff_m):
    """|Phi_N|(r) under each declared boundary rule, SI."""
    # inward-cumulative integral of g_bar from r to R_last
    dr = np.diff(r_m)
    mid = 0.5 * (gbar[1:] + gbar[:-1])
    seg = mid * dr
    inner = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    Rl = r_m[-1]
    Ml = gbar[-1] * Rl ** 2 / G
    out = {}
    out["last"] = inner
    out["inf"] = inner + G * Ml / Rl
    vf2 = Vflat_ms ** 2 if Vflat_ms > 0 else gbar[-1] * Rl
    tail = vf2 * np.log(max(R_TRUNC / Rl, 1.0)) + G * Ml / R_TRUNC
    out["flat"] = inner + tail
    out["half"] = G * Mb_tot / np.maximum(r_m, Reff_m)
    return out


def build(verbose=True):
    """Ingest, cut, split, and precompute every invariant on the TRAIN set."""
    gals = SP.ingest(verbose=verbose)
    SP.stratified_split(gals, verbose=verbose)
    rows = dict(name=[], split=[], r=[], gobs=[], gbar=[], egobs=[],
                Mb_enc=[], Vflat=[], gal=[])
    phi = {k: [] for k in PHI_RULES}
    for gi, g in enumerate(gals):
        r = g.R0 * KPC
        v2 = (g.Vgas * np.abs(g.Vgas) + UPS_DISK * g.Vdisk ** 2
              + UPS_BULGE * g.Vbul ** 2) * 1e6
        # DECLARED CUT, before any residual: a point whose net baryonic v^2 is
        # negative (gas term dominating and signed negative) has no meaningful
        # g_bar to feed a law.  Identical to the cut adyn_run.sparc_arrays uses,
        # so the two lanes' RMS numbers are directly comparable.
        keep = (v2 > 0) & (g.Vobs0 > 0)
        if keep.sum() < 5:
            continue
        r, v2 = r[keep], v2[keep]
        gbar = v2 / r
        gobs = (g.Vobs0[keep] * 1e3) ** 2 / r
        egobs = 2.0 * (g.Vobs0[keep] * 1e3) * (g.eV[keep] * 1e3) / r
        Menc = np.maximum.accumulate(v2 * r / G)      # spherical-equivalent
        Mb_tot = max(0.5 * g.L36 * 1e9 + 1.33 * g.MHI * 1e9, 1.0) * MSUN
        Reff = max(g.Reff, 0.3) * KPC
        P = _phi_profiles(r, gbar, g.Vflat * 1e3, Mb_tot, Reff)
        n = len(r)
        rows["name"] += [g.name] * n
        rows["split"] += [g.split] * n
        rows["gal"] += [gi] * n
        for k, v in (("r", r), ("gobs", gobs), ("gbar", gbar),
                     ("egobs", egobs), ("Mb_enc", Menc)):
            rows[k].append(v)
        rows["Vflat"].append(np.full(n, g.Vflat * 1e3))
        for k in PHI_RULES:
            phi[k].append(P[k])
    out = {k: (np.concatenate(v) if k in ("r", "gobs", "gbar", "egobs",
                                          "Mb_enc", "Vflat")
               else np.asarray(v))
           for k, v in rows.items()}
    for k in PHI_RULES:
        out["phi_" + k] = np.concatenate(phi[k])
    out["is_train"] = out["split"] == "train"
    out["gals"] = gals
    return out


#: globals of the nonlocal invariant, declared once and used in EVERY channel
L_NL = 300.0 * KPC
M_NL = 1.0e12 * MSUN


def invariants(D, phi_rule="inf"):
    """Raw invariant fields on the SPARC points, SI, before division by I_0.

    Every invariant here is a genuine FIELD -- a function of position that
    needs no object-specific centre, radius or catalogue row -- so that the
    same definition can be evaluated in a disk, in a cluster shell and inside
    a cluster member galaxy without a per-object convention creeping in.
        rhobar  = div g_N / (4 pi G), i.e. the Poisson source itself
        tidal   = |traceless Hessian of Phi_N|_F, in s^-2
        qbar    = M_b(within L_NL of x)/(M_b + M_NL), bounded in [0,1)
    """
    r, gb = D["r"], D["gbar"]
    rho = np.empty_like(r)
    tid = np.empty_like(r)
    qb = np.empty_like(r)
    for gi in np.unique(D["gal"]):
        m = D["gal"] == gi
        rr, gg = r[m], gb[m]
        dg = np.gradient(gg, rr)
        rho[m] = (dg + 2.0 * gg / rr) / (4.0 * np.pi * G)
        t_rr, t_tt = -dg, -gg / rr
        tr = (t_rr + 2.0 * t_tt) / 3.0
        tid[m] = np.sqrt((t_rr - tr) ** 2 + 2.0 * (t_tt - tr) ** 2)
        # every SPARC point lies inside L_NL of the whole galaxy (r_last is at
        # most a few tens of kpc against L_NL = 300 kpc), so M_within = M_b,tot
        Mtot = float(D["Mb_enc"][m][-1])
        qb[m] = Mtot / (Mtot + M_NL)
    return dict(
        one=np.ones_like(r),
        gn=gb / A0,
        phi=D["phi_" + phi_rule],
        rhobar=np.maximum(rho, 1e-40),
        tidal=np.maximum(tid, 1e-45),
        qbar=qb,
    )


def score(cand, D, INV, mask=None):
    """RMS dex, and the per-galaxy residual means for bootstrapping."""
    m = D["is_train"] if mask is None else mask
    W = W_cand(cand, {k: v[m] for k, v in INV.items()})
    gp = g_response(cand, D["gbar"][m], W)
    res = np.log10(np.maximum(gp, 1e-300) / np.maximum(D["gobs"][m], 1e-300))
    return float(np.sqrt(np.mean(res ** 2))), res


def fit_a0(cand, D, INV, lo=0.3e-10, hi=4.0e-10, n=41, refine=3):
    """Golden-free bracketed scan for a0 on the TRAIN split only."""
    best = (np.inf, cand.a0)
    for _ in range(refine):
        grid = np.linspace(lo, hi, n)
        vals = []
        for a in grid:
            cand.a0 = float(a)
            vals.append(score(cand, D, INV)[0])
        vals = np.asarray(vals)
        j = int(np.argmin(vals))
        if vals[j] < best[0]:
            best = (float(vals[j]), float(grid[j]))
        lo = grid[max(j - 1, 0)]
        hi = grid[min(j + 1, n - 1)]
        if hi <= lo:
            break
    cand.a0 = best[1]
    return best[1], best[0]
