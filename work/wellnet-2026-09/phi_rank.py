"""Does baryonic potential depth add a DIRECTION on SPARC, or is it already spanned?

The potential-depth hypothesis is that the acceleration scale itself depends on
how deep the well is:

    g_obs = nu( g_bar / A_0(Phi_b) ) g_bar,
    A_0(Phi_b) = a0 [ 1 + alpha (|Phi_b|/Phi_c)^p / (1 + (|Phi_b|/Phi_c)^p) ]

Before assembling a galaxies-to-clusters ladder to test it, there is a cheap and
decisive question: on the SPARC bench alone, is |Phi_b| a NEW direction, or is it
a function of the two the rank-2 theorem already found?

If |Phi_b| lies in the span of (log g_N, log r) then SPARC cannot test this
hypothesis AT ALL -- any apparent potential-depth effect fitted on SPARC would be
a relabelled function of variables the exhaustive k<=3 enumeration has already
searched to the ground. The whole burden would then fall on the group and cluster
ladder, and the experiment's leverage is entirely in how far that ladder decouples
|Phi_b| from g_bar.

Phi_b is computed with an explicitly stated outer boundary condition:

    Phi_b(r) = - [ Int_r^Rmax g_bar dr'  +  g_bar(Rmax) * Rmax ]

i.e. Newtonian point-mass falloff beyond the last measured point, for which
Int_Rmax^inf (G M / r'^2) dr' = G M / Rmax = g_bar(Rmax) * Rmax exactly. The
sensitivity of every conclusion to that choice is reported, because the outer
term is not small.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "gravitylab")))

KPC = 3.0856775814913673e19
A0 = 1.2e-10


def phi_of(R_kpc, gbar, tail="point"):
    """Baryonic potential depth at each radius, in (m/s)^2, negative."""
    r = np.asarray(R_kpc, float) * KPC
    g = np.asarray(gbar, float)
    o = np.argsort(r)
    r, g = r[o], g[o]
    # inward-cumulative trapezoid of g dr from Rmax down to each r
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(r)
    inner = np.concatenate([[0.0], np.cumsum(seg)])
    inner = inner[-1] - inner                      # Int_r^Rmax
    if tail == "point":
        outer = g[-1] * r[-1]                      # GM/Rmax
    elif tail == "flat":
        # v_flat^2 ln(r/Rmax) diverges; use a 10x Rmax cutoff and say so
        outer = g[-1] * r[-1] * np.log(10.0)
    elif tail == "none":
        outer = 0.0
    else:
        raise ValueError(tail)
    out = np.empty_like(r)
    out[:] = -(inner + outer)
    inv = np.empty_like(out)
    inv[o] = out
    return inv


def build(tail="point"):
    import data as D
    gals = D.ingest(verbose=False)
    D.stratified_split(gals, verbose=False)
    rows = []
    for g in gals:
        Vb2 = (g.Vgas * np.abs(g.Vgas) + 0.5 * g.Vdisk ** 2 + 0.7 * g.Vbul ** 2)
        ok = Vb2 > 0
        if ok.sum() < 5:
            continue
        R = g.R0[ok]
        gbar = (Vb2[ok] * 1e6) / (R * KPC)
        gobs = ((g.Vobs0[ok] * 1e3) ** 2) / (R * KPC)
        ph = phi_of(R, gbar, tail)
        for i in range(len(R)):
            rows.append((g.name, R[i], gbar[i], gobs[i], ph[i], g.split))
    return rows


def report(tail, rows):
    R = np.array([r[1] for r in rows])
    gb = np.array([r[2] for r in rows])
    go = np.array([r[3] for r in rows])
    ph = np.abs(np.array([r[4] for r in rows]))
    lg, lr, lp = np.log10(gb), np.log10(R), np.log10(ph)

    print(f"\n=== outer boundary condition: {tail} ===")
    print(f"   {len(rows):,} points, {len(set(r[0] for r in rows))} galaxies")
    print(f"   log|Phi_b| range {lp.min():.3f} .. {lp.max():.3f}  "
          f"({lp.max()-lp.min():.2f} dex)")

    # --- is log|Phi_b| a function of (log g_N, log r)?
    A = np.column_stack([np.ones_like(lg), lg, lr])
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    pred = A @ coef
    res = lp - pred
    r2 = 1.0 - np.var(res) / np.var(lp)
    print(f"   linear fit  log|Phi_b| = {coef[0]:+.4f} {coef[1]:+.4f} log g_N "
          f"{coef[2]:+.4f} log r")
    print(f"   R^2 = {r2:.6f}   residual scatter {np.std(res):.4f} dex")

    # --- and with quadratic terms, i.e. any smooth function of the two
    A2 = np.column_stack([A, lg ** 2, lr ** 2, lg * lr])
    c2, *_ = np.linalg.lstsq(A2, lp, rcond=None)
    res2 = lp - A2 @ c2
    r2q = 1.0 - np.var(res2) / np.var(lp)
    print(f"   quadratic in (log g_N, log r):  R^2 = {r2q:.6f}   "
          f"residual {np.std(res2):.4f} dex")

    # --- the SVD statement, matching the rank-2 theorem's construction
    M = np.column_stack([lg, lr, lp])
    M = (M - M.mean(0)) / M.std(0)
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"   SVD of standardised (log g_N, log r, log|Phi_b|): "
          + "  ".join(f"{s:.4g}" for s in sv))
    print(f"   condition number {sv[0]/sv[-1]:.4g}")

    # --- how much does Phi_b actually vary at FIXED g_bar? the whole leverage
    print("   spread of log|Phi_b| within narrow log g_N bins "
          "(this is the leverage SPARC alone can offer):")
    edges = np.percentile(lg, np.linspace(0, 100, 9))
    tot = []
    for i in range(len(edges) - 1):
        m = (lg >= edges[i]) & (lg < edges[i + 1])
        if m.sum() < 20:
            continue
        tot.append(np.std(lp[m]))
        print(f"      log g_N in [{edges[i]:+.2f},{edges[i+1]:+.2f})  "
              f"n={m.sum():5d}   sd(log|Phi_b|) = {np.std(lp[m]):.3f} dex")
    print(f"   median within-bin spread: {np.median(tot):.3f} dex")

    # --- residual of the RAR against Phi_b, the direct test
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(gb / A0, 1e-30))))
    rar_res = np.log10(go) - np.log10(gb * nu)
    rp = np.corrcoef(rar_res, lp)[0, 1]
    # partial out log g_N and log r, since those are the known directions
    B = np.column_stack([np.ones_like(lg), lg, lr])
    ra = rar_res - B @ np.linalg.lstsq(B, rar_res, rcond=None)[0]
    pa = lp - B @ np.linalg.lstsq(B, lp, rcond=None)[0]
    rpp = np.corrcoef(ra, pa)[0, 1] if np.std(pa) > 1e-12 else float("nan")
    print(f"   corr(RAR residual, log|Phi_b|)            = {rp:+.4f}")
    print(f"   partial, controlling for log g_N & log r  = {rpp:+.4f}"
          f"   [sd of the partialled Phi_b: {np.std(pa):.4f} dex]")
    names = np.array([r[0] for r in rows])
    uniq = np.unique(names)
    rng = np.random.default_rng(4)
    boot = []
    for _ in range(2000):
        pick = rng.choice(uniq, uniq.size, replace=True)
        m = np.concatenate([np.where(names == u)[0] for u in pick])
        Bb = np.column_stack([np.ones(m.size), lg[m], lr[m]])
        try:
            rr = rar_res[m] - Bb @ np.linalg.lstsq(Bb, rar_res[m], rcond=None)[0]
            pp = lp[m] - Bb @ np.linalg.lstsq(Bb, lp[m], rcond=None)[0]
            if np.std(pp) > 1e-12 and np.std(rr) > 1e-12:
                boot.append(np.corrcoef(rr, pp)[0, 1])
        except Exception:
            pass
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"   galaxy-level bootstrap of the partial: {rpp:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  ({uniq.size} galaxies resampled)")
    print(f"   => SPARC can exclude a partial correlation larger than "
          f"|{max(abs(lo),abs(hi)):.3f}| and no more.")
    return {"tail": tail, "n": len(rows), "ngal": int(uniq.size),
            "partial_ci": [float(lo), float(hi)],
            "r2_linear": float(r2),
            "r2_quadratic": float(r2q), "sv": [float(s) for s in sv],
            "cond": float(sv[0] / sv[-1]),
            "median_within_bin_spread_dex": float(np.median(tot)),
            "corr_resid_phi": float(rp), "partial_corr": float(rpp),
            "phi_residual_sd_dex": float(np.std(pa))}


if __name__ == "__main__":
    print("=" * 78)
    print("IS BARYONIC POTENTIAL DEPTH A NEW DIRECTION ON SPARC?")
    print("=" * 78)
    out = []
    for tail in ("point", "flat"):
        out.append(report(tail, build(tail)))
    with open(os.path.join(HERE, "phi_rank.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n" + "=" * 78)
    print("   Read the quadratic R^2 and the partialled residual sd together.")
    print("   If R^2 is ~1 and the partialled sd is small, |Phi_b| carries no")
    print("   information on SPARC beyond (g_N, r) and the potential-depth")
    print("   hypothesis is UNTESTABLE on this bench -- the ladder is required.")
