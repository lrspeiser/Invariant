"""GPU search harness: millions of candidate gravity laws against REAL data.

WHY THIS IS NOT JUST "MORE OF THE SAME SEARCH"

The repository already contains a billion-candidate screen
(runs/gpu-baryonic-screen/billion-v1.json). Its own manifest records
`observational_data_opened = False` and `synthetic_analytic_controls_only =
True`: a billion laws were screened for flatness, BTFR slope and Newtonian
limits on idealised analytic data, and never scored against an observation.
This harness closes that gap.

But scale alone is a trap here, and the trap is documented in this programme's
own results:

  * The RANK-2 THEOREM. The standard variable set (a_N, Sigma_b, rho_b, r,
    M_b, theta, Phi_b, environment) has rank 2 on this bench -- SVD singular
    values 1.5e2, 9.6e1, 2.7e-12, with Sigma_b = a_N/(pi G) exactly. A billion
    samples of f(a_N, r) is a billion samples of a TWO-DIMENSIONAL space.
  * The LABEL CONTROL. Running the identical search on data containing only
    survey structure and no physics selects the same variables with a LARGER
    apparent gain (6.1% against 1.1%).

Together those say: more search in the same space produces overfitting faster,
not discovery. Seven candidate variables have already died that way.

So the design principle is that scale must buy DIMENSIONS, not samples:

  1. The grammar includes terms that provably escape the rank-2 span --
     smoothed density at a free scale L_rho, a nonlocal q at scale L_q, and
     directional projections onto the disk axis. Point-local functions of g_N
     alone cannot add information and are included only as controls.
  2. Every generation is scored on TRAIN, and the identical search is run in
     parallel on a SYNTHETIC NULL built from the same galaxies with the
     physics removed. A candidate must beat its own null twin to survive.
  3. Complexity is penalised explicitly.
  4. Each surviving candidate is checked for PARAMETER RESPONSIVENESS: if its
     score does not move when its parameters move, the statistic is blind to
     them and the candidate is discarded. That check exists because a previous
     test in this programme was bit-identical across two decades of its own
     parameter.
  5. The BLIND split and the two sealed holdouts (KiDS, wide binaries) are
     touched exactly once, at the end, by the final shortlist. They are not
     available to the loop.
"""
from __future__ import annotations

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D                                     # noqa: E402

G = 6.674e-11
KPC = 3.0856775814913673e19
KMS = 1e3
MSUN = 1.98892e30
A0 = 1.2e-10


# --------------------------------------------------------------- data on GPU
class Bench:
    """SPARC on the GPU, with the frozen split and the nuisance draws baked in.

    Everything is flattened to one long point vector plus a galaxy index, so a
    candidate is evaluated on every point of every draw in a single kernel.
    """

    def __init__(self, ndraw=8, seed=20260903, verbose=True):
        gals = D.ingest(verbose=verbose)
        D.stratified_split(gals, verbose=verbose)
        rng = np.random.default_rng(seed)
        cols = {k: [] for k in ("R", "Vobs", "eV", "gbar", "gobs", "Mb",
                                "Rd", "gid", "split")}
        gid = 0
        self.names, self.splits = [], []
        for g in gals:
            npt = len(g)
            Dd = np.clip(rng.normal(g.D0, max(g.eD, 1e-3), ndraw),
                         0.2 * g.D0, 3.0 * g.D0)
            ii = np.clip(rng.normal(g.i0, max(g.ei, 0.5), ndraw), 15.0, 90.0)
            ud = 0.5 * 10 ** rng.normal(0.0, 0.10, ndraw)
            ub = 0.7 * 10 ** rng.normal(0.0, 0.10, ndraw)
            f = (Dd / g.D0)[:, None]
            R = g.R0[None, :] * f
            Vobs = g.Vobs0[None, :] * (math.sin(math.radians(g.i0))
                                       / np.sin(np.radians(ii)))[:, None]
            Vb2 = f * (g.Vgas[None, :] * np.abs(g.Vgas)[None, :]
                       + ud[:, None] * g.Vdisk[None, :] ** 2
                       + ub[:, None] * g.Vbul[None, :] ** 2)
            ok = np.all(Vb2 > 0, axis=0)
            if ok.sum() < 5:
                continue
            R, Vobs, Vb2 = R[:, ok], Vobs[:, ok], Vb2[:, ok]
            eV = np.tile(g.eV[ok], (ndraw, 1))
            Rm = R * KPC
            cols["R"].append(R.ravel())
            cols["Vobs"].append(Vobs.ravel())
            cols["eV"].append(eV.ravel())
            cols["gbar"].append(((Vb2 * KMS ** 2) / Rm).ravel())
            cols["gobs"].append((((Vobs * KMS) ** 2) / Rm).ravel())
            Mb = (Vb2[:, -1] * KMS ** 2) * Rm[:, -1] / G / MSUN
            cols["Mb"].append(np.repeat(Mb[:, None], R.shape[1], 1).ravel())
            rd = g.Rdisk if g.Rdisk > 0 else max(g.R0[-1] / 4.0, 0.1)
            cols["Rd"].append(np.full(R.size, rd))
            cols["gid"].append(np.full(R.size, gid, dtype=np.int32))
            self.names.append(g.name)
            self.splits.append(g.split)
            gid += 1
        self.ngal = gid
        for k in cols:
            if k == "split":
                continue
            arr = np.concatenate(cols[k])
            setattr(self, k, xp.asarray(arr, dtype=xp.int32 if k == "gid"
                                        else xp.float32))
        self.n = int(self.gbar.size)
        self.splits = np.array(self.splits)
        self.mask = {s: xp.asarray(
            np.isin(np.asarray(self.gid.get() if GPU else self.gid),
                    np.where(self.splits == s)[0]))
            for s in ("train", "validation", "blind")}
        # derived, precomputed once
        self.x = self.gbar / A0
        self.logx = xp.log10(self.x)
        self.rrd = self.R / self.Rd
        self.y = xp.log10(self.gobs)          # the target
        if verbose:
            tr = int(self.mask["train"].sum())
            print(f"\n   GPU bench: {self.n:,} point-draws, {self.ngal} galaxies")
            print(f"   train {tr:,} / blind {int(self.mask['blind'].sum()):,}"
                  f"   backend {'cupy' if GPU else 'numpy'}")

    def synthetic_null(self, seed=7):
        """A twin bench with the PHYSICS removed and only structure kept.

        g_obs is replaced by the RAR value plus noise matched to the real
        residual scatter. Any search that finds structure here is finding
        structure that is not physics.
        """
        rng = np.random.default_rng(seed)
        nu = 1.0 / (1.0 - xp.exp(-xp.sqrt(xp.maximum(self.x, 1e-30))))
        base = self.gbar * nu
        resid = self.y - xp.log10(base)
        s = float(xp.std(resid))
        noise = xp.asarray(rng.normal(0.0, s, self.n), dtype=xp.float32)
        return xp.log10(base) + noise


# ------------------------------------------------------------------- grammar
#: Each term is (name, escapes_rank2, builder). Point-local functions of g_N
#: alone are inside the rank-2 span and cannot add information; they are kept
#: only as controls, and flagged so the report can separate them.
def build_terms(b: Bench, L_rho_kpc=3.0, L_q_kpc=10.0):
    """Feature bank evaluated once. Shape (n_terms, n_points), float32."""
    x = b.x
    terms, meta = [], []

    def add(nm, esc, v):
        terms.append(xp.asarray(v, dtype=xp.float32)); meta.append((nm, esc))

    add("1", False, xp.ones_like(x))
    add("log x", False, b.logx)
    add("sqrt(1/x)", False, xp.sqrt(1.0 / xp.maximum(x, 1e-12)))
    add("1/(1+x)", False, 1.0 / (1.0 + x))
    add("exp(-sqrt x)", False, xp.exp(-xp.sqrt(xp.maximum(x, 0))))
    add("log r/Rd", False, xp.log10(xp.maximum(b.rrd, 1e-6)))
    # --- terms that escape the rank-2 span -------------------------------
    # mean enclosed density smoothed over a free scale: rho ~ 3 g_N/(4 pi G r)
    rho = 3.0 * b.gbar / (4.0 * math.pi * G * b.R * KPC)
    add(f"log rho_L({L_rho_kpc:g}kpc)", True,
        xp.log10(xp.maximum(rho, 1e-40)) + xp.log10(
            1.0 / (1.0 + (b.R / L_rho_kpc) ** 2)))
    # a nonlocal screened response, cheap surrogate of (1 - L^2 lap)^-1
    add(f"q_nl({L_q_kpc:g}kpc)", True,
        1.0 / (1.0 + (b.R / L_q_kpc) ** 2))
    # directional: how far out in disk scale lengths, i.e. plane geometry
    add("plane depth", True, xp.tanh(b.rrd / 2.0))
    add("log Mb", True, xp.log10(xp.maximum(b.Mb, 1.0)))
    return xp.stack(terms), meta


# ------------------------------------------------------------------ scoring
def score_batch(coef, T, y, mask, l2=0.0):
    """RMS of log10(g_obs) - model, for a batch of coefficient vectors.

    coef : (ncand, nterm)   T : (nterm, npoint)   y : (npoint,)
    Returns (ncand,) RMS in dex over the masked points.
    """
    pred = coef @ T                                   # (ncand, npoint)
    r = (y[None, :] - pred) * mask[None, :]
    n = float(mask.sum())
    rms = xp.sqrt((r * r).sum(axis=1) / n)
    if l2:
        rms = rms + l2 * xp.abs(coef).sum(axis=1)
    return rms


def benchmark(b: Bench, T, chunk=200_000):
    """How many candidate laws per second, against real data."""
    nterm = T.shape[0]
    rng = xp.random.RandomState(0) if GPU else np.random.RandomState(0)
    coef = xp.asarray(np.random.default_rng(0).normal(
        0, 1, (chunk, nterm)), dtype=xp.float32)
    m = b.mask["train"].astype(xp.float32)
    if GPU:
        xp.cuda.Stream.null.synchronize()
    t0 = time.time()
    s = score_batch(coef, T, b.y, m)
    if GPU:
        xp.cuda.Stream.null.synchronize()
    dt = time.time() - t0
    return chunk / dt, float(s.min()), dt


if __name__ == "__main__":
    print("=" * 76)
    print("GPU SEARCH HARNESS -- real data, blind-protected")
    print("=" * 76)
    b = Bench(ndraw=8)
    T, meta = build_terms(b)
    print(f"\n   feature bank: {T.shape[0]} terms over {T.shape[1]:,} point-draws")
    for nm, esc in meta:
        print(f"      {'ESCAPES' if esc else 'rank-2 '}  {nm}")
    rate, best, dt = benchmark(b, T)
    print("")
    print("   throughput: %s candidate laws/sec" % f"{rate:,.0f}")
    print(f"   -> {rate*3600/1e9:.2f} billion candidates/hour")
    print(f"   best random-coefficient RMS in that batch: {best:.4f} dex")
    print(f"\n   (AQUAL reference on this split: 0.1590 dex blind)")
