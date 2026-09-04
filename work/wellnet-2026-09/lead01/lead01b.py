"""Lead 01, part 2: the within-class leverage, and the test it enables.

Part 1 validated the chain by reproducing the paper's own M_gas,500 to
median 1.0079 with 0.0476 dex scatter. This part answers the question the
ladder could not:

    Within ONE class -- one instrument, one pipeline, no class label anywhere --
    how much does |Phi_b| vary at fixed g_bar?

Benchmarks to beat, all from Run R:
    SPARC alone                          0.309 dex
    full six-rung ladder                 0.766 dex  (86% of it the class label)
    ladder minus the class label         0.286 dex
    the X-ray GROUP rungs specifically   0.10 - 0.17 dex   <-- the cap eFEDS removes

Radial range: 0.15 R500 to R500, the core-excised aperture in which the
temperature is actually measured. Nothing is evaluated where the profile is
extrapolated, though the enclosed mass is integrated from the centre because the
gas inside 0.15 R500 is really there.

Baryons are GAS ONLY, declared. Stars add roughly 10-20% of M_b at group scales
and would raise g_bar and |Phi_b| together, moving both axes along the
degeneracy; the leverage measurement is a spread at fixed g_bar and is therefore
insensitive to it at first order. The boost measurement is not, and carries it as
a stated systematic.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import lead01 as L1

HERE = os.path.dirname(os.path.abspath(__file__))
MSUN, KPC = L1.MSUN, L1.KPC
A0 = L1.A0


def build():
    h1, d1 = L1.read_tsv(os.path.join(HERE, "efeds_bahar2022_table1_density.tsv"))
    h2, d2 = L1.read_tsv(os.path.join(HERE, "efeds_bahar2022_table2.tsv"))
    t2 = {r["ID"]: r for r in d2}
    out = []
    for r1 in d1:
        r2 = t2.get(r1["ID"])
        if r2 is None:
            continue
        n0sq, rs_as = L1.num(r1, "n0"), L1.num(r1, "rs")
        eps, beta, alpha = (L1.num(r1, "epsilon"), L1.num(r1, "beta"),
                            L1.num(r1, "alpha"))
        z, R500_am = L1.num(r2, "z"), L1.num(r2, "R500")
        T = L1.num(r2, "Tcex500")
        eT = max(L1.num(r2, "e_Tcex500"), L1.num(r2, "E_Tcex500"))
        if not np.isfinite(T) or T <= 0:
            T = L1.num(r2, "T500")
            eT = max(L1.num(r2, "e_T500"), L1.num(r2, "E_T500"))
        if not all(np.isfinite([n0sq, rs_as, eps, beta, alpha, z, R500_am,
                                T, eT])):
            continue
        if n0sq <= 0 or rs_as <= 0 or z <= 0 or R500_am <= 0 or T <= 0:
            continue
        if eT / T > 0.5 or not (beta > 1.0 / 3.0):
            continue
        DA = L1.d_angular(z)
        asec = math.pi / (180.0 * 3600.0)
        rs, R500 = rs_as * asec * DA, R500_am * 60.0 * asec * DA
        # shape-parameter precision, for the robustness split
        eb = max(L1.num(r1, "e_beta"), L1.num(r1, "E_beta")) / max(beta, 1e-9)
        ee = max(L1.num(r1, "e_epsilon"), L1.num(r1, "E_epsilon")) / max(eps, 1e-9)
        ers = max(L1.num(r1, "e_rs"), L1.num(r1, "E_rs")) / max(rs_as, 1e-9)
        out.append(dict(id=r1["ID"], z=z, T=T, eT=eT, rs=rs, R500=R500,
                        n0sq=n0sq, eps=eps, beta=beta, alpha=alpha,
                        e_beta=eb, e_eps=ee, e_rs=ers))
    return out


def radial(rec, nr=400, lo=0.15, hi=1.0):
    R500 = rec["R500"]
    r, ne, rho, M, slope = L1.profiles(rec["rs"], rec["n0sq"], rec["eps"],
                                       rec["beta"], rec["alpha"], R500, nr=nr)
    g = L1.G * M / np.maximum(r, 1e-12) ** 2
    phi = L1.phi_from_g(r, g)
    kT = rec["T"] * L1.KEV
    gobs = -(kT / (L1.MU * L1.M_P * r)) * slope
    m = (r >= lo * R500) & (r <= hi * R500) & (g > 0) & (gobs > 0) & (phi > 0)
    return dict(r=r[m], g=g[m], phi=phi[m], gobs=gobs[m], M=M[m],
                R500=R500, frac=r[m] / R500)


def leverage(lg, lp, nbin=8, label=""):
    """Median spread of log|Phi_b| inside equal-count log g_bar bins."""
    if lg.size < nbin * 8:
        return float("nan"), []
    edges = np.percentile(lg, np.linspace(0, 100, nbin + 1))
    sds, rows = [], []
    for i in range(nbin):
        m = (lg >= edges[i]) & (lg < edges[i + 1] if i < nbin - 1
                                else lg <= edges[i + 1])
        if m.sum() < 12:
            continue
        sds.append(np.std(lp[m]))
        rows.append((float(edges[i]), float(edges[i + 1]), int(m.sum()),
                     float(np.std(lp[m])), float(lp[m].max() - lp[m].min())))
    return (float(np.median(sds)) if sds else float("nan")), rows


def r2_quadratic(lg, lr, lp):
    A = np.column_stack([np.ones_like(lg), lg, lr, lg ** 2, lr ** 2, lg * lr])
    c, *_ = np.linalg.lstsq(A, lp, rcond=None)
    res = lp - A @ c
    return 1.0 - np.var(res) / np.var(lp), float(np.std(res))


def main():
    recs = build()
    print("=" * 78)
    print("LEAD 01b -- within-class leverage from resolved eFEDS profiles")
    print("=" * 78)
    print(f"\n   {len(recs)} systems pass the part-1 cuts")

    rows = []
    for rec in recs:
        p = radial(rec)
        if p["r"].size < 20:
            continue
        for i in range(0, p["r"].size, 8):        # thin to ~50 points/system
            rows.append((rec["id"], p["r"][i] / KPC, p["g"][i], p["phi"][i],
                         p["gobs"][i], p["frac"][i], rec["e_beta"],
                         rec["e_eps"], rec["T"], rec["z"]))
    ids = np.array([r[0] for r in rows])
    rk = np.array([r[1] for r in rows])
    gb = np.array([r[2] for r in rows])
    ph = np.array([r[3] for r in rows])
    go = np.array([r[4] for r in rows])
    fr = np.array([r[5] for r in rows])
    ebeta = np.array([r[6] for r in rows])
    nsys = len(set(ids))
    print(f"   {len(rows):,} radial points over {nsys} systems, "
          f"0.15-1.0 R500")

    lg, lr, lp = np.log10(gb), np.log10(rk), np.log10(ph)
    print(f"   log g_bar range {lg.min():.2f} .. {lg.max():.2f}   "
          f"g_bar/a0 = {gb.min()/A0:.3f} .. {gb.max()/A0:.3f}")
    print(f"   log|Phi_b| range {lp.min():.2f} .. {lp.max():.2f}  "
          f"({lp.max()-lp.min():.2f} dex)")

    lev, brows = leverage(lg, lp)
    print("\n   WITHIN-CLASS LEVERAGE  (no class label exists in this sample)")
    print("      log g_bar bin           n     sd(log|Phi_b|)   range")
    for a, b, n, sd, rg in brows:
        print(f"      [{a:+.2f},{b:+.2f})   {n:5d}      {sd:.3f}        {rg:.2f}")
    print(f"      median within-bin sd = {lev:.3f} dex")
    print("\n      benchmarks from Run R:")
    print("         SPARC alone                       0.309 dex")
    print("         full ladder (86% class label)     0.766")
    print("         ladder MINUS the class label      0.286")
    print("         X-ray group rungs                 0.10 - 0.17   <- the cap")
    verdict = lev > 0.30
    print(f"      => eFEDS gives {lev:.3f} dex within ONE class, "
          f"{'ABOVE' if verdict else 'below'} SPARC's 0.309 and "
          f"{lev/0.17:.1f}x the two-radius group cap")

    r2, resid = r2_quadratic(lg, lr, lp)
    print(f"\n   collinearity: R^2 of log|Phi_b| on a quadratic in "
          f"(log g_bar, log r) = {r2:.4f}")
    print(f"      residual scatter {resid:.3f} dex   "
          f"(SPARC 0.9322 / 0.218, ladder 0.9147 / 0.247)")

    # robustness: does the leverage survive keeping only well-constrained shapes?
    tight = ebeta < 0.5
    if tight.sum() > 200:
        lev2, _ = leverage(lg[tight], lp[tight])
        print(f"\n   robustness: restricting to beta measured to <50% "
              f"({tight.sum():,} points, {len(set(ids[tight]))} systems)")
        print(f"      median within-bin sd = {lev2:.3f} dex")
    else:
        lev2 = float("nan")
        print(f"\n   robustness: only {tight.sum()} points have beta to <50%; "
              "not split")

    out = {"n_systems": nsys, "n_points": len(rows),
           "leverage_dex": lev, "leverage_tight_beta_dex": lev2,
           "bins": brows, "r2_quadratic": float(r2),
           "residual_dex": resid,
           "benchmarks": {"sparc": 0.309, "ladder": 0.766,
                          "ladder_minus_class": 0.286,
                          "group_rungs_max": 0.17},
           "beats_sparc": bool(verdict)}
    with open(os.path.join(HERE, "lead01b_leverage.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(HERE, "lead01_points.npz"), ids=ids, r_kpc=rk,
             gbar=gb, phi=ph, gobs=go, frac=fr, ebeta=ebeta)
    print(f"\n   written: lead01b_leverage.json, lead01_points.npz")


if __name__ == "__main__":
    main()
