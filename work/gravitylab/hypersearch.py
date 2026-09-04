"""Billion-scale structure search over gravity laws, against REAL data.

THE ARCHITECTURE, AND WHY IT IS THE ONE THAT MAKES BILLIONS MEANINGFUL

A "candidate law" here is a functional FORM, not a coefficient vector. Searching
random coefficients is nearly worthless: for any fixed set of basis functions the
optimal coefficients solve a linear system, so sampling them at random
rediscovers, slowly and badly, a thing linear algebra gives exactly. The
expensive and interesting axis is WHICH basis functions appear.

So the search is organised as an ATOM BANK plus SUBSET SELECTION:

  1. An atom is one parameterised basis function, e.g. 1/(1 + x/s) at one
     particular s, evaluated on every point-draw of the bench. A thousand atoms
     tile the continuous parameter space of ten functional families over eight
     physical variables.
  2. The full atom Gram G = A A^T and the projections v = A y are computed ONCE,
     on the training points only, in float64.
  3. A candidate law is a SUBSET of K atoms. Its optimal coefficients and its
     exact training RMS come from a K x K sub-matrix GATHERED out of G -- the
     data is never touched again. Scoring one candidate costs O(K^2) memory
     traffic and a K x K Cholesky, not O(K * n_points).

That is the whole trick: it turns "score a candidate law with its optimal
coefficients" from ~1e5 flops into ~1e2, which is what makes billions per second
real rather than rhetorical. With ~1000 atoms and K = 6 the space is C(1000, 6)
~ 1e15 subsets, so the loop is a genuine search and not an enumeration.

WHAT THE RANK-2 THEOREM ACTUALLY LICENSES

The theorem found on this bench is that the eight-variable set (a_N, Sigma_b,
rho_b, r, M_b, theta, Phi_b, environment) has numerical rank 2 -- singular
values 1.5e2, 9.6e1, 2.7e-12, with Sigma_b = a_N/(pi G) exactly. It does NOT
say that only functions of a_N are available. It says there are exactly TWO
independent directions, and a_N and r span them. So:

    * an atom built from g_bar or x alone lives in the RAR's OWN direction and
      can only re-fit the interpolating function;
    * an atom carrying r or r/R_d uses the SECOND direction, which is the only
      genuinely new point-local information on this bench;
    * M_b is galaxy-level, constant along a rotation curve, and so is a third
      and different kind of term;
    * q_nl is a nonlocal functional of the source and is not a point function of
      either direction.

Every atom is tagged with which of these it uses, so the report can state what
a winning law actually needed rather than asserting it.

WHAT KEEPS IT HONEST

Each candidate is scored against four targets that differ only in v and y.y, so
all four cost the same gather:

    real     measured log10(g_obs / g_bar)
    null     RAR + Gaussian noise matched to the real residual scatter
    perm     RAR + the real residuals permuted across ALL points
    perm_g   RAR + the real residuals permuted WITHIN each galaxy, which keeps
             every galaxy's mean offset (distance, inclination and M/L errors
             are real and are not new physics) and destroys only the radial
             dependence -- the sharper control of the two

Each target is scored against its OWN baseline, because the twins are built with
slightly different scatter and comparing a control to the real target's baseline
is a baseline mismatch, not a control. This programme has already seen a case
where the physics-free world won (6.1% against 1.1%), so none of this is a
formality.

KiDS and wide binaries are sealed holdouts and are not present in this module.
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
from gpu_search import Bench, A0, G as GNEWT, KPC       # noqa: E402


def sync():
    if GPU:
        xp.cuda.Stream.null.synchronize()


# --------------------------------------------------------------- the atom bank
FAMILIES = [
    ("log1p",   lambda u: xp.log10(1.0 + u)),
    ("inv1p",   lambda u: 1.0 / (1.0 + u)),
    ("expneg",  lambda u: xp.exp(-xp.minimum(u, 60.0))),
    ("invsqrt", lambda u: 1.0 / xp.sqrt(u + 1e-12)),
    ("sqrt",    lambda u: xp.sqrt(u)),
    ("tanh",    lambda u: xp.tanh(u)),
    # the RAR / MOND interpolation functions, as families rather than fixtures
    ("nu_rar",  lambda u: xp.log10(1.0 / (1.0 - xp.exp(-xp.sqrt(u + 1e-12))))),
    ("nu_simp", lambda u: xp.log10(0.5 + xp.sqrt(0.25 + 1.0 / (u + 1e-12)))),
    ("nu_std",  lambda u: xp.log10(xp.sqrt(0.5 + xp.sqrt(0.25 + 1.0
                                                         / (u ** 2 + 1e-12))))),
    ("logistic", lambda u: 1.0 / (1.0 + xp.exp(-xp.minimum(
        xp.maximum(xp.log(u + 1e-12), -60.0), 60.0)))),
]

#: which of the two independent directions (or neither) an atom uses
DIMS = ("g", "r", "g+r", "gal", "nl")


def build_atoms(b: Bench, nscale=13, verbose=True):
    """Evaluate every (family, variable, scale) atom on every point-draw.

    Returns A (natom, npoint) float32 and a list of (name, dim) metadata.
    """
    R = b.R                                                     # kpc
    rho = 3.0 * b.gbar / (4.0 * math.pi * GNEWT * b.R * KPC)    # kg/m^3

    VARS = [
        # (name, dimension used, positive values, scale grid)
        ("x",    "g",   b.gbar / A0,      np.logspace(-2.0, 2.0, nscale)),
        ("gbar", "g",   b.gbar / 1e-10,   np.logspace(-2.0, 2.0, nscale)),
        ("rho",  "g+r", rho / 1e-22,      np.logspace(-3.0, 3.0, nscale)),
        ("r",    "r",   b.R,              np.logspace(-0.7, 1.8, nscale)),
        ("r_Rd", "r",   b.rrd,            np.logspace(-1.0, 1.3, nscale)),
        ("Mb",   "gal", b.Mb / 1e9,       np.logspace(-2.0, 3.0, nscale)),
        ("q_nl", "nl",  None,             np.logspace(-0.3, 1.8, nscale)),
        ("Sig",  "g+r", b.Mb / (2.0 * math.pi * xp.maximum(b.R, 1e-3) ** 2)
                        / 1e8,            np.logspace(-2.0, 3.0, nscale)),
    ]
    cols, meta = [], []
    for vname, dim, val, grid in VARS:
        for s in grid:
            if vname == "q_nl":
                base = xp.asarray(1.0 / (1.0 + (R / s) ** 2), dtype=xp.float32)
                for fname, f in FAMILIES[:6]:
                    cols.append(xp.asarray(f(xp.maximum(base, 1e-12)),
                                           dtype=xp.float32))
                    meta.append((f"{fname}(q_nl/{s:.2f})", dim))
                continue
            u = xp.maximum(xp.asarray(val, dtype=xp.float32) / s, 1e-12)
            for fname, f in FAMILIES:
                cols.append(xp.asarray(f(u), dtype=xp.float32))
                meta.append((f"{fname}({vname}/{s:.3g})", dim))
    cols.append(xp.ones(b.n, dtype=xp.float32))
    meta.append(("1", "g"))
    A = xp.stack(cols)
    fin = xp.isfinite(A).all(axis=1)
    keep = fin & (A.std(axis=1) > 1e-8)
    keep[-1] = True
    A = A[keep]
    kb = keep.get() if GPU else keep
    meta = [m for m, k in zip(meta, kb) if k]
    mu = A.mean(axis=1, keepdims=True)
    sg = xp.maximum(A.std(axis=1, keepdims=True), 1e-8)
    A = (A - mu) / sg
    A[-1] = 1.0
    if verbose:
        cnt = {d: sum(1 for _, dd in meta if dd == d) for d in DIMS}
        print(f"   atom bank: {A.shape[0]:,} atoms x {A.shape[1]:,} point-draws")
        print("      by direction used: " +
              "  ".join(f"{d}={cnt[d]}" for d in DIMS))
    return A, meta


# ------------------------------------------------------------------- targets
def targets(b: Bench, seed=11):
    """The real target and its three physics-free twins, all in dex."""
    y = b.y - xp.log10(b.gbar)                    # log10(g_obs / g_bar)
    nu = 1.0 / (1.0 - xp.exp(-xp.sqrt(xp.maximum(b.x, 1e-30))))
    rar = xp.log10(nu)
    resid = y - rar
    s = float(xp.std(resid))
    rng = np.random.default_rng(seed)
    null = rar + xp.asarray(rng.normal(0, s, b.n), dtype=xp.float32)
    perm = rar + resid[xp.asarray(rng.permutation(b.n))]
    gid = (b.gid.get() if GPU else b.gid)
    og = np.arange(b.n)
    for gg in range(b.ngal):
        w = np.where(gid == gg)[0]
        og[w] = w[rng.permutation(w.size)]
    permg = rar + resid[xp.asarray(og)]
    return ({"real": y.astype(xp.float32), "null": null.astype(xp.float32),
             "perm": perm.astype(xp.float32),
             "perm_g": permg.astype(xp.float32)}, rar, s)


TARGETS = ("real", "null", "perm", "perm_g")


# ------------------------------------------------ precomputed Gram and vectors
class Screen:
    """Everything a candidate needs, with the data already collapsed away."""

    def __init__(self, b: Bench, A, meta, ridge=1e-3, verbose=True):
        self.meta = meta
        self.dim = [d for _, d in meta]
        self.natom = A.shape[0]
        self.ridge = ridge
        self.A = A                     # kept for direct verification only
        ys, self.rar, self.sigma = targets(b)
        self.ys = ys
        self.tr, self.bl = b.mask["train"], b.mask["blind"]
        self.ntr, self.nbl = int(self.tr.sum()), int(self.bl.sum())
        t0 = time.time()
        # float64 throughout: a float32 Gram of near-collinear atoms produced
        # non-finite Cholesky factors and RMS values worse than predicting zero.
        At = A[:, self.tr].astype(xp.float64)
        Ab = A[:, self.bl].astype(xp.float64)
        self.G = At @ At.T
        self.Gb = Ab @ Ab.T
        self.v = {k: At @ ys[k][self.tr].astype(xp.float64) for k in ys}
        self.vb = {k: Ab @ ys[k][self.bl].astype(xp.float64) for k in ys}
        self.yy = {k: float(ys[k][self.tr] @ ys[k][self.tr]) for k in ys}
        self.yyb = {k: float(ys[k][self.bl] @ ys[k][self.bl]) for k in ys}
        del At, Ab
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()
        sync()
        self.ref_train = {k: float(xp.sqrt(xp.mean(
            (ys[k][self.tr] - self.rar[self.tr]) ** 2))) for k in ys}
        self.ref_blind = {k: float(xp.sqrt(xp.mean(
            (ys[k][self.bl] - self.rar[self.bl]) ** 2))) for k in ys}
        self.rar_train = self.ref_train["real"]
        self.rar_blind = self.ref_blind["real"]
        if verbose:
            print(f"   Gram {self.natom}x{self.natom} (float64) in "
                  f"{time.time()-t0:.1f}s   train {self.ntr:,} / "
                  f"blind {self.nbl:,}")
            print(f"   RAR reference: train {self.rar_train:.4f} dex   "
                  f"blind {self.rar_blind:.4f} dex   "
                  f"(residual scatter {self.sigma:.4f})")

    def _aug(self, idx):
        """Append the mandatory intercept column to a batch of genomes."""
        ones = xp.full((idx.shape[0], 1), self.natom - 1, dtype=idx.dtype)
        return xp.concatenate([idx, ones], axis=1)

    # -------------------------------------------------------------- scoring
    def score(self, idx, which="real", blind=False):
        """Exact optimal-coefficient RMS for a batch of atom subsets.

        idx : (B, K) int32.  Returns ((B,) RMS in dex, (B,K) coefficients).
        A subset whose Gram is too ill-conditioned to solve is returned at RMS
        1e3 so that selection discards it, rather than at a spuriously small
        value produced by a broken factorisation.
        """
        Gsrc = self.Gb if blind else self.G
        vsrc = self.vb[which] if blind else self.v[which]
        yy = self.yyb[which] if blind else self.yy[which]
        n = self.nbl if blind else self.ntr
        idx = self._aug(idx)
        Gs = Gsrc[idx[:, :, None], idx[:, None, :]]     # (B,K+1,K+1)
        vs = vsrc[idx]                                  # (B,K+1)
        c, ok = _chol_solve(Gs, vs, self.ridge * n)
        sse = yy - 2.0 * (c * vs).sum(1) + ((c[:, None, :] @ Gs)[:, 0] * c).sum(1)
        rms = xp.sqrt(xp.maximum(sse, 0.0) / n)
        bad = (~ok) | (~xp.isfinite(rms)) | (sse > yy * 1.0000001)
        rms = xp.where(bad, xp.float64(1e3), rms)
        return rms.astype(xp.float32), c

    def verify(self, idx_row, coef, which="real", blind=False):
        """RMS recomputed straight from the data, bypassing the Gram entirely.

        The Gram path is an algebraic shortcut; a shortlist that is going to be
        reported has to agree with the direct computation, or the shortcut is
        what is being reported.
        """
        m = self.bl if blind else self.tr
        r = (self._aug(idx_row[None, :])[0].get() if GPU
             else self._aug(idx_row[None, :])[0])
        Asub = self.A[xp.asarray(r), :][:, m].astype(xp.float64)
        pred = xp.asarray(coef, dtype=xp.float64) @ Asub
        y = self.ys[which][m].astype(xp.float64)
        return float(xp.sqrt(xp.mean((y - pred) ** 2)))


def _chol_solve(G, v, ridge):
    """Batched Cholesky solve of (G + ridge I) c = v for small K, in float64."""
    B, K, _ = G.shape
    L = xp.zeros_like(G)
    ok = xp.ones(B, dtype=bool)
    for i in range(K):
        s = G[:, i, i] + ridge
        for k in range(i):
            s = s - L[:, i, k] ** 2
        ok = ok & (s > 1e-8)
        d = xp.sqrt(xp.maximum(s, 1e-8))
        L[:, i, i] = d
        for j in range(i + 1, K):
            t = G[:, j, i]
            for k in range(i):
                t = t - L[:, j, k] * L[:, i, k]
            L[:, j, i] = t / d
    z = xp.zeros_like(v)
    for i in range(K):
        t = v[:, i]
        for k in range(i):
            t = t - L[:, i, k] * z[:, k]
        z[:, i] = t / L[:, i, i]
    c = xp.zeros_like(v)
    for i in range(K - 1, -1, -1):
        t = z[:, i]
        for k in range(i + 1, K):
            t = t - L[:, k, i] * c[:, k]
        c[:, i] = t / L[:, i, i]
    return c, ok & xp.isfinite(c).all(axis=1)


# ------------------------------------------------------------------ benchmark
def benchmark(S: Screen, K=6, B=1_000_000, reps=3):
    idx = xp.asarray(np.random.default_rng(1).integers(
        0, S.natom, (B, K)), dtype=xp.int32)
    S.score(idx[:1000])
    sync()
    t0 = time.time()
    for _ in range(reps):
        r, _ = S.score(idx)
    sync()
    dt = (time.time() - t0) / reps
    return B / dt, float(r.min()), dt


if __name__ == "__main__":
    print("=" * 78)
    print("HYPERSEARCH -- exact-coefficient structure search on real SPARC")
    print("=" * 78)
    b = Bench(ndraw=8)
    A, meta = build_atoms(b)
    S = Screen(b, A, meta)
    rate, best, dt = benchmark(S)
    print("")
    print("   throughput: %s optimally-fitted laws/sec" % f"{rate:,.0f}")
    print("   -> %s per hour" % f"{rate*3600:,.0f}")
    print(f"   best of a random 1e6-law batch: {best:.4f} dex "
          f"(RAR {S.rar_train:.4f})")
