"""Sparse search over field laws, and the injection-recovery test that licenses it.

The engine in `field_grammar.py` precomputes one Poisson solve per atom, so any
sparse subset of atoms with its optimal coefficients is an instant linear
combination. That makes billions of candidate FIELD laws affordable. It does not,
by itself, make them trustworthy.

Two things have to be demonstrated before any result from this engine counts,
and they are the two halves of control 5 in the programme's control harness:

  RECOVERY   generate an observation from a KNOWN law inside the grammar, and
             require the search to find that law.
  ABSTINENCE generate an observation from a purely SCALAR law -- no anisotropy
             anywhere -- and require the search NOT to select tensor atoms. The
             rate at which it does anyway is the false-positive rate for
             "anisotropy detected", and it is the credibility of every future
             tensor claim this engine could make.

Run J's lesson applies unchanged: a search of this size manufactures fit. The
number that matters is never the fit, it is the margin over a control that
contains no physics.

The grammar is deduplicated first, because ghat ghat^T is exactly the identity
when the tensor contracts only with grad Phi_N (see qumond_degeneracy.py) and
linear shape functions collapse across their scale after standardisation.
Duplicate columns in a Gram are not merely wasteful; they make the normal
equations singular and the "selected" atom arbitrary among its clones.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_grammar import GPU, KPC, MSUN, FieldBank, sphericity, xp   # noqa
from qumond_degeneracy import independent_atoms, make_source          # noqa

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ design
def observables(bank, rmin_kpc=20.0, rmax_kpc=140.0):
    """Concatenate the two observables into one vector per atom.

    Restricted to an annulus that excludes the softened centre and the padded
    boundary, so nothing in the fit is driven by either.
    """
    n, h = bank.n, bank.h
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None] * xp.ones((1, n))
    Y = xp.ones((n, 1)) * ax[None, :]
    R = xp.sqrt(X ** 2 + Y ** 2)
    m = (R > rmin_kpc * KPC) & (R < rmax_kpc * KPC)
    v0, va = bank.midplane_vc()
    d0, da = bank.deflection()
    y0 = xp.concatenate([v0[m], d0[m]])
    A = xp.concatenate([va[:, m], da[:, m]], axis=1)
    # put the two observables on a common scale so neither dominates by units
    s1 = float(xp.std(v0[m])) + 1e-300
    s2 = float(xp.std(d0[m])) + 1e-300
    w = xp.concatenate([xp.full(int(m.sum()), 1.0 / s1),
                        xp.full(int(m.sum()), 1.0 / s2)])
    return y0 * w, A * w[None, :], int(m.sum())


def chol_solve(G, v, ridge):
    B, K, _ = G.shape
    L = xp.zeros_like(G)
    ok = xp.ones(B, dtype=bool)
    for i in range(K):
        s = G[:, i, i] + ridge
        for k in range(i):
            s = s - L[:, i, k] ** 2
        ok = ok & (s > 1e-10)
        d = xp.sqrt(xp.maximum(s, 1e-10))
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


class Search:
    """Gram over the atom RESPONSES, exactly as in Run J but on field laws."""

    def __init__(self, A, ridge=1e-6):
        self.A = A.astype(xp.float64)
        self.natom = A.shape[0]
        self.npt = A.shape[1]
        self.G = self.A @ self.A.T
        self.ridge = ridge

    def set_target(self, y):
        self.y = y.astype(xp.float64)
        self.v = self.A @ self.y
        self.yy = float(self.y @ self.y)

    def score(self, idx):
        Gs = self.G[idx[:, :, None], idx[:, None, :]]
        vs = self.v[idx]
        c, ok = chol_solve(Gs, vs, self.ridge * self.npt)
        sse = (self.yy - 2.0 * (c * vs).sum(1)
               + ((c[:, None, :] @ Gs)[:, 0] * c).sum(1))
        rms = xp.sqrt(xp.maximum(sse, 0.0) / self.npt)
        bad = (~ok) | (~xp.isfinite(rms)) | (sse > self.yy * 1.0000001)
        return xp.where(bad, xp.float64(1e9), rms), c

    def exhaustive_k(self, k):
        """Every subset of size k. Returns (best rms, best index tuple)."""
        n = self.natom
        if k == 1:
            idx = xp.arange(n, dtype=xp.int32)[:, None]
            r, _ = self.score(idx)
            i = int(xp.argmin(r))
            return float(r[i]), [i]
        if k == 2:
            jj, kk = np.triu_indices(n, k=1)
            idx = xp.asarray(np.stack([jj, kk], 1).astype(np.int32))
            r, _ = self.score(idx)
            i = int(xp.argmin(r))
            return float(r[i]), [int(jj[i]), int(kk[i])]
        if k == 3:
            jj, kk = np.triu_indices(n, k=1)
            starts = np.searchsorted(jj, np.arange(n))
            jg = xp.asarray(jj.astype(np.int32))
            kg = xp.asarray(kk.astype(np.int32))
            best, bidx = np.inf, None
            for i in range(n - 2):
                s = int(starts[i + 1])
                m = jg.size - s
                if m <= 0:
                    continue
                col = xp.full(m, i, dtype=xp.int32)
                idx = xp.stack([col, jg[s:], kg[s:]], axis=1)
                r, _ = self.score(idx)
                a = int(xp.argmin(r))
                if float(r[a]) < best:
                    best = float(r[a])
                    bidx = [i, int(jj[s + a]), int(kk[s + a])]
            return best, bidx
        raise ValueError(k)


def is_tensor(name):
    """Does this atom carry anisotropy, or is it a scalar rescaling?"""
    return name.split(" x ")[-1] not in ("I", "gg")


def main():
    print("=" * 78)
    print("FIELD-LAW SPARSE SEARCH -- with injection recovery and abstinence")
    print("=" * 78)
    rho, h = make_source()
    print(f"\n   source axis ratio {sphericity(rho, h):.3f}")
    bank = FieldBank(rho, h, dhat=(0, 0, 1))

    groups, _ = independent_atoms(bank)
    keep = [g[0] for g in groups]
    keep = [i for i in keep if bank.meta[i].split(" x ")[-1] != "gg"]
    meta = [bank.meta[i] for i in keep]
    print(f"   deduplicated grammar: {len(bank.meta)} -> {len(meta)} atoms "
          f"({sum(is_tensor(m) for m in meta)} anisotropic, "
          f"{sum(not is_tensor(m) for m in meta)} scalar)")

    y0, Afull, npt = observables(bank)
    A = Afull[xp.asarray(np.array(keep, dtype=np.int64))]
    print(f"   observable: {npt:,} annulus points x 2 probes = "
          f"{2*npt:,} numbers")

    S = Search(A)
    rng = np.random.default_rng(12)
    tens = [i for i, m in enumerate(meta) if is_tensor(m)]
    scal = [i for i, m in enumerate(meta) if not is_tensor(m)]
    res = {"n_atoms": len(meta), "n_tensor": len(tens), "n_scalar": len(scal)}

    def trial(truth_idx, coefs, noise, tag):
        y = y0 + sum(c * A[i] for i, c in zip(truth_idx, coefs))
        y = y + xp.asarray(rng.normal(0, noise * float(xp.std(y)), y.size))
        S.set_target(y - y0)
        rows = []
        for k in (1, 2, 3):
            r, idx = S.exhaustive_k(k)
            rows.append((k, r, idx))
        return rows

    # ---------------------------------------------------- 1. RECOVERY
    print("\n   1. RECOVERY -- inject a known anisotropic law, can we find it?")
    inj = [tens[3], tens[40]]
    amp = float(xp.std(y0)) * 0.25
    coefs = [amp / float(xp.std(A[inj[0]])), -0.6 * amp / float(xp.std(A[inj[1]]))]
    print(f"      injected: {meta[inj[0]]}")
    print(f"                {meta[inj[1]]}")
    rec = {}
    for noise in (0.0, 0.02, 0.10):
        rows = trial(inj, coefs, noise, "rec")
        k2 = [r for r in rows if r[0] == 2][0]
        found = set(k2[2])
        hit = len(found & set(inj))
        print(f"      noise {noise:4.0%}   best k=2 residual {k2[1]:.3e}   "
              f"recovered {hit}/2   -> " +
              " + ".join(meta[i] for i in k2[2]))
        rec[str(noise)] = {"residual": k2[1], "recovered": hit,
                           "selected": [meta[i] for i in k2[2]]}
    res["recovery"] = rec

    # ------------------------------------------------- 2. ABSTINENCE
    print("\n   2. ABSTINENCE -- inject a purely SCALAR law. The search must")
    print("      NOT select anisotropic atoms. Every time it does is a false")
    print("      'anisotropy detected'.")
    fp = 0
    trials = 24
    detail = []
    for t in range(trials):
        pick = [scal[rng.integers(len(scal))] for _ in range(2)]
        cs = [float(xp.std(y0)) * 0.25 / float(xp.std(A[p])) *
              (1.0 if i == 0 else -0.6) for i, p in enumerate(pick)]
        y = y0 + sum(c * A[i] for i, c in zip(pick, cs))
        y = y + xp.asarray(rng.normal(0, 0.05 * float(xp.std(y)), y.size))
        S.set_target(y - y0)
        r, idx = S.exhaustive_k(2)
        anis = [i for i in idx if is_tensor(meta[i])]
        if anis:
            fp += 1
            detail.append([meta[i] for i in idx])
    print(f"      {fp}/{trials} scalar-only injections selected at least one")
    print(f"      anisotropic atom  =>  false-positive rate "
          f"{fp/trials:.1%}")
    if detail[:3]:
        print("      examples of the false selections:")
        for d in detail[:3]:
            print("         " + " + ".join(d))
    res["abstinence"] = {"trials": trials, "false_positives": fp,
                         "rate": fp / trials, "examples": detail[:5]}

    # ------------------------------------------- 3. NULL: no law at all
    print("\n   3. PURE NOISE -- no law injected. Whatever residual reduction")
    print("      the search achieves here is what it manufactures from nothing.")
    gains = []
    for t in range(12):
        y = y0 + xp.asarray(rng.normal(0, 0.05 * float(xp.std(y0)), y0.size))
        S.set_target(y - y0)
        base = float(xp.sqrt(S.yy / S.npt))
        r1, _ = S.exhaustive_k(1)
        r3, _ = S.exhaustive_k(3)
        gains.append((base, r1, r3))
    g = np.array(gains)
    print(f"      median residual with no law:   {np.median(g[:,0]):.4e}")
    print(f"      after the best single atom:    {np.median(g[:,1]):.4e}  "
          f"({100*(1-np.median(g[:,1])/np.median(g[:,0])):+.1f}%)")
    print(f"      after the best three atoms:    {np.median(g[:,2]):.4e}  "
          f"({100*(1-np.median(g[:,2])/np.median(g[:,0])):+.1f}%)")
    print("      That percentage is the floor any real claim must clear.")
    res["noise_floor"] = {
        "base": float(np.median(g[:, 0])), "k1": float(np.median(g[:, 1])),
        "k3": float(np.median(g[:, 2])),
        "gain_k3_pct": float(100 * (1 - np.median(g[:, 2]) / np.median(g[:, 0])))}

    with open(os.path.join(HERE, "field_search.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n   written: field_search.json")


if __name__ == "__main__":
    main()
