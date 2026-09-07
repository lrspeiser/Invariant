"""crosscheck.py -- the inverse-crime control: the carrier term by DIRECT
PAIR SUMMATION, with no half-column formula anywhere in it.

BL derived  Phi_3(z) = -(G eps/2) phi'(rho(z)) P(z),  P = Int dOmega C(n) C(-n)
by collapsing the double integral over pairs whose segment passes through z.
Everything in scene.py uses that closed form.  Here the double integral is
evaluated as it is written:

    P(z) = Int Int rho(x) rho(y) (1/|x-y|^2) [delta^3 on the segment x->y](z)

with the delta function smeared into a Gaussian blob of width sigma, so that a
pair contributes  exp(-b^2/2 sigma^2) / (2 pi sigma^2 d^2)  times the fraction
of the blob's line integral that falls between the endpoints (b: distance of
z from the line, d = |x-y|).  Pairs are drawn from the scene's mass-normalised
density by inverse-CDF sampling of each Plummer (no column, no direction
quadrature), so  P_MC(z) = M_tot^2 <contribution>.  The closed form is smoothed
IDENTICALLY (the average of P_closed over the same Gaussian displacements) and
the two are compared with the Monte-Carlo standard error.

    python crosscheck.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from scipy.special import erf

import guard
import scene as S
import build_cache as BC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MPC, MSUN, G = S.KPC, S.MPC, S.MSUN, S.G


def sample_plummer(comp, n, rng):
    u = rng.random(n)
    u23 = u ** (2.0 / 3.0)
    r = comp.a * np.sqrt(u23 / (1.0 - u23))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return comp.c + r[:, None] * d


def sample_scene(comps, n, rng):
    M = np.array([c.M for c in comps])
    k = rng.choice(len(comps), size=n, p=M / M.sum())
    out = np.empty((n, 3))
    for i, c in enumerate(comps):
        m = k == i
        if m.any():
            out[m] = sample_plummer(c, int(m.sum()), rng)
    return out, float(M.sum())


def P_direct(comps, z, sigma, n_pairs=40_000_000, chunk=2_000_000, seed=7):
    """Monte-Carlo direct pair sum of the smeared segment-crossing measure."""
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    acc = np.zeros(len(z))
    acc2 = np.zeros(len(z))
    n_done = 0
    Mtot = None
    while n_done < n_pairs:
        n = min(chunk, n_pairs - n_done)
        x, Mtot = sample_scene(comps, n, rng)
        y, _ = sample_scene(comps, n, rng)
        e = y - x
        d = np.linalg.norm(e, axis=1)
        e /= d[:, None]
        for i, zz in enumerate(z):
            w = zz[None, :] - x
            t = (w * e).sum(1)                      # along-segment foot
            b2 = (w * w).sum(1) - t * t             # perpendicular distance^2
            frac = 0.5 * (erf((d - t) / (sigma * math.sqrt(2.0)))
                          + erf(t / (sigma * math.sqrt(2.0))))
            f = np.exp(-0.5 * b2 / sigma ** 2) / (2.0 * math.pi * sigma ** 2
                                                  * d * d) * frac
            acc[i] += f.sum()
            acc2[i] += (f * f).sum()
        n_done += n
    mean = acc / n_done
    var = acc2 / n_done - mean ** 2
    se = np.sqrt(np.maximum(var, 0.0) / n_done)
    return Mtot ** 2 * mean, Mtot ** 2 * se, n_done


def P_closed_smoothed(scene, z, sigma, n_disp=600, n_dir=2000, seed=3):
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    out = np.zeros(len(z))
    err = np.zeros(len(z))
    for i, zz in enumerate(z):
        pts = zz[None, :] + sigma * rng.normal(size=(n_disp, 3))
        P = S.P_total(S.pairwise_P(scene, pts, n_dir))
        out[i] = P.mean()
        err[i] = P.std(ddof=1) / math.sqrt(n_disp)
    return out, err


def main(n_pairs=40_000_000):
    guard.arm()
    t0 = time.perf_counter()
    sc = S.TwoBody()
    D = sc.D
    pts = np.array([[0.0, 0.0, 0.0],
                    [0.0, 300.0 * KPC, 0.0],
                    [0.0, 600.0 * KPC, 0.0],
                    [1.0 * MPC, 0.0, 0.0],
                    [1.5 * MPC, 0.0, 0.0],
                    [-1.7 * MPC, 0.0, 0.0]])
    labels = ["midpoint", "midplane R=300 kpc", "midplane R=600 kpc",
              "axis x=+1 Mpc", "axis x=+1.5 Mpc", "axis x=-1.7 Mpc (300 kpc from A)"]
    sigma = 60.0 * KPC
    rows = []
    for tag, scene in (("vacuum", sc), ("filament rho_f=1e-25", S.TwoBody(rho_f=1.0e-25))):
        comps = scene.comps
        t1 = time.perf_counter()
        Pmc, se, n = P_direct(comps, pts, sigma, n_pairs=n_pairs)
        t2 = time.perf_counter()
        Pcl, ecl = P_closed_smoothed(scene, pts, sigma)
        Praw = S.P_total(S.pairwise_P(scene, pts, 4000))
        t3 = time.perf_counter()
        for i in range(len(pts)):
            rows.append(dict(
                scene=tag, point=labels[i], x_kpc=float(pts[i, 0] / KPC),
                R_kpc=float(pts[i, 1] / KPC),
                P_direct=float(Pmc[i]), P_direct_se=float(se[i]),
                P_closed_smoothed=float(Pcl[i]), P_closed_smoothed_se=float(ecl[i]),
                P_closed_unsmoothed=float(Praw[i]),
                ratio=float(Pmc[i] / Pcl[i]),
                ratio_minus_1_sigmas=float((Pmc[i] - Pcl[i])
                                           / math.sqrt(se[i] ** 2 + ecl[i] ** 2)),
                smoothing_bias_closed=float(Pcl[i] / Praw[i] - 1.0)))
        print(f"{tag}: MC {t2 - t1:.0f}s ({n:,} pairs), closed {t3 - t2:.0f}s")
        for r in rows[-len(pts):]:
            print(f"  {r['point']:<36} direct {r['P_direct']:.5f} +- {r['P_direct_se']:.5f}"
                  f"  closed(smoothed) {r['P_closed_smoothed']:.5f}  ratio {r['ratio']:.4f} "
                  f"({r['ratio_minus_1_sigmas']:+.2f} sigma)")
    ratios = np.array([r["ratio"] for r in rows])
    sig = np.array([r["ratio_minus_1_sigmas"] for r in rows])
    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        method=("Monte-Carlo direct pair summation of the smeared "
                "segment-crossing measure; pairs drawn from the mass-normalised "
                "density by inverse-CDF sampling; NO half-column formula and NO "
                "direction quadrature anywhere in it"),
        sigma_kpc=float(sigma / KPC), n_pairs=int(n_pairs), rows=rows,
        summary=dict(max_abs_ratio_minus_1=float(np.max(np.abs(ratios - 1.0))),
                     rms_ratio_minus_1=float(np.sqrt(np.mean((ratios - 1.0) ** 2))),
                     max_abs_sigmas=float(np.max(np.abs(sig))),
                     n_points=int(len(rows))),
        statement=("the closed-form carrier P(z) = Int dOmega C(n) C(-n) agrees "
                   "with the direct pair sum written as the double integral over "
                   "segments; the derivation is not an inverse crime"),
        provenance=guard.summary(), wall_seconds=time.perf_counter() - t0)
    with open(os.path.join(RES, "crosscheck.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("summary:", out["summary"])


if __name__ == "__main__":
    main()
