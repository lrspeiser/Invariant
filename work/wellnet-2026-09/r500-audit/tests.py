"""
Self-tests for the R500 tautology audit.

Every lane in this programme found a real bug in its own first implementation.
The bugs this file actually caught are recorded in REPORT.md; the tests are
kept so the numbers cannot drift.
"""
from __future__ import annotations
import math
import sys

import numpy as np

import ingest as I
import nullsim as N

KPC, G, MU, MP = I.KPC, I.G, I.MU, I.MP
FAIL = []


def check(name, cond, msg=""):
    tag = "PASS" if cond else "FAIL"
    print(f"   [{tag}] {name}" + (f"  -- {msg}" if msg else ""))
    if not cond:
        FAIL.append(name)


def main():
    cl = I.load_all(verbose=False)
    TS = [N.Template(c) for c in cl]
    print("T1  rank/correlation primitives")
    a = np.array([1., 1., 2., 3., 3., 3.])
    check("ties share the average rank",
          np.allclose(N.rank(a), [0.5, 0.5, 2.0, 4.0, 4.0, 4.0]),
          str(N.rank(a)))
    x = np.array([1., 2., 3., 4.])
    check("spearman of a monotone map is +1", abs(N.spear(x, np.exp(x)) - 1) < 1e-12)
    check("spearman is invariant to positive rescaling of either axis",
          abs(N.spear(x, -x) - N.spear(7.3 * x, -0.2 * x)) < 1e-12)

    print("\nT2  THE CANCELLATION LEMMA -- R500 drops out of the numerator")
    T = TS[0]
    truth = N.make_truth(T)
    rng = np.random.default_rng(1)
    ne, kT_c = N.observe(T, truth, rng, N.DEFAULT_CFG)
    g1, b1 = N.analyse(T, ne, kT_c, T.R500_pub)
    g2, b2 = N.analyse(T, ne, kT_c, 0.55 * T.R500_pub)
    g3, b3 = N.analyse(T, ne, kT_c, 2.30 * T.R500_pub)
    d = max(np.nanmax(np.abs(g2 / g1 - 1)), np.nanmax(np.abs(g3 / g1 - 1)))
    check("g_obs identical for R500 scaled by 0.55x and 2.30x, when the profile "
          "is republished in those same units", d < 1e-12, f"max rel diff {d:.2e}")
    check("g_bar does not depend on R500 at all",
          np.allclose(b1, b2) and np.allclose(b1, b3))

    print("\nT3  monotone-invariance: the R500 tautology cannot act WITHIN a cluster")
    c = cl[0]
    p = I.build_profile(c)
    m = (p["r"] > N.R_MIN) & (p["r"] < N.R_MAX) & (p["go"] > 0)
    y = I.rar_residual(p["gb"][m], p["go"][m])
    vals = [N.spear(p["r"][m] / (f * c["R500_hse"]), y) for f in (0.5, 1.0, 2.0, 5.0)]
    check("within-cluster Spearman is bit-identical across a 10x range of R500",
          max(vals) - min(vals) == 0.0, f"{vals}")
    print("      (this is the point: any per-cluster normalisation is a monotone "
          "map, so\n       it CANNOT change a within-cluster rank statistic. "
          "R500 can only act\n       across clusters.)")

    print("\nT4  make_truth is self-consistent")
    bad = []
    for T in TS:
        tr = N.make_truth(T)
        A = (4 / 3) * np.pi * 500 * T.rhoc
        M = tr["go"] * T.r ** 2 / G
        Mat = np.interp(tr["R500_true"], T.r, M)
        bad.append(abs(Mat / (A * tr["R500_true"] ** 3) - 1))
        # the anchor: R500_true must equal the published R500
        bad.append(abs(tr["R500_true"] / T.R500_pub - 1))
    check("R500_true satisfies the overdensity definition and equals R500_pub",
          max(bad) < 2e-3, f"max rel error {max(bad):.2e}")

    print("\nT5  the HSE integration inverts the bench's gradient formula")
    errs = []
    for T in TS:
        tr = N.make_truth(T)
        g_back = N.hse_g(T, T.ne, tr["kT"])
        w = (T.r > 200 * KPC) & (T.r < T.r_bnd * 0.9)
        errs.append(np.nanmax(np.abs(g_back[w] / tr["go"][w] - 1)))
    check("g recovered from the constructed T matches the injected g",
          max(errs) < 0.05, f"max rel error {max(errs):.3f} over 200 kpc - 0.9 r_bnd")

    print("\nT6  noiseless pipeline recovery (this is where the pipeline bias lives)")
    cfg0 = dict(N.DEFAULT_CFG, ne_scale=0.0, T_scale=0.0, T_calib=0.0, rho_corr=0.0)
    rng = np.random.default_rng(0)
    res = N.one_realisation(TS, rng, cfg0, s_scaled=0.0, s_abs=0.0)
    dy = res["y"] - res["y_true"]
    S = N.stats_of(res)
    print(f"      noiseless y - y_true : median {np.median(dy):+.4f} dex, "
          f"rms {np.std(dy):.4f}, range {dy.min():+.3f} to {dy.max():+.3f}")
    print(f"      noiseless S1_hse = {S['S1_hse']:+.4f}   S1_phys = {S['S1_phys']:+.4f}")
    check("a flat truth does not come back with a strong radial trend",
          abs(S["S1_hse"]) < 0.45,
          f"S1_hse = {S['S1_hse']:+.4f} with ZERO noise and a FLAT truth "
          f"-- this is pure pipeline bias")

    print("\nT7  responsiveness is non-zero (the monotone-invariant-statistic trap)")
    vals = []
    for s in (0.0, -0.5, -1.0, -1.5):
        rng = np.random.default_rng(3)
        r = N.one_realisation(TS, rng, cfg0, s_scaled=s)
        vals.append(N.stats_of(r)["S1_hse"])
    print(f"      S1_hse vs injected slope 0,-0.5,-1,-1.5 : "
          + ", ".join(f"{v:+.4f}" for v in vals))
    check("dS1/d(injected slope) != 0", (max(vals) - min(vals)) > 0.05,
          f"spread {max(vals)-min(vals):.4f}")

    print("\nT8  ingest guards actually fire")
    ok = False
    try:
        I.load_xcop_cluster("NOT_A_CLUSTER")
    except Exception:
        ok = True
    check("missing cluster returns None or raises",
          ok or I.load_xcop_cluster("NOT_A_CLUSTER") is None)
    H = I.load_herbonnet()
    check("Herbonnet table is 100 rows (two table* envs joined)", len(H) == 100)

    print("\nT9  baryon-only radii use no total mass")
    src = open("ingest.py", encoding="utf-8").read()
    fn = src[src.index("def baryonic_radii"):src.index("def load_all")]
    check("baryonic_radii references no M500/R500 header quantity",
          "M500" not in fn and "R500_hse" not in fn,
          "gas density and n_e only")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILING: {FAIL}")
        return 1
    print("all tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
