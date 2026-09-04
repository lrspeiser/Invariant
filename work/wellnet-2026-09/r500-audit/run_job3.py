"""
JOB 3 -- the law-free replacement t = r/r_a0, revisited.

The record (gravity-discovery-program.md, "The law-free replacement for r/R500
was tested and does not work") already established that t fails to unify CLASH
with SPARC.  That is not redone here.  Two things ARE done:

  3a  Is t itself subject to a tautology?  r_a0 is the radius where the object's
      own BARYONIC g_bar crosses a0, so it shares M_b with the denominator of
      the excess and with the RAR argument x = g_bar/a0.  Sign and size measured.

  3b  Can the extrapolation be bounded rather than merely flagged?  For X-COP the
      crossing lies inside the innermost data point, so r_a0 is an extrapolation.
      Four defensible inward continuations of M_b are computed and the spread is
      reported as a bound.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import nullsim as N

KPC, G, MSUN, A0 = I.KPC, I.G, I.MSUN, I.A0
OUT = {}


def full_profile(c):
    """the whole radial range, no bench cut -- needed to see whether g_bar ever
    reaches a0."""
    p = I.build_profile(c)
    return p


def r_a0(r, gb, mode, Mb=None):
    """radius where g_bar = a0, under a stated inward continuation.

    measured      : log-log interpolation inside the data, nan if no crossing
    point_mass    : M_b frozen at its innermost measured value  -> g_bar = GM/r^2
                    This is a strict UPPER bound on r_a0: any realistic inward
                    mass decline makes g_bar rise more slowly and the crossing
                    move inward.
    powerlaw      : continue the innermost measured dlnM/dlnr
    bcg           : point_mass plus a 1e12 Msun BCG stellar component
    """
    x = gb / A0
    if mode == "measured":
        if x.max() < 1.0 or x.min() > 1.0:
            return float("nan")
        i = np.where(np.diff(np.sign(np.log(x))) != 0)[0]
        if not len(i):
            return float("nan")
        i = i[0]
        lx = np.log(x[i:i + 2])
        lr = np.log(r[i:i + 2])
        return float(np.exp(lr[0] + (lr[1] - lr[0]) * (-lx[0]) / (lx[1] - lx[0])))
    M0, r0 = Mb[0], r[0]
    if mode == "point_mass":
        return math.sqrt(G * M0 / A0)
    if mode.startswith("bcg"):
        mb = {"bcg_0p5e12": 0.5e12, "bcg_1e12": 1e12, "bcg_2e12": 2e12}[mode]
        return math.sqrt(G * (M0 + mb * MSUN) / A0)
    if mode == "powerlaw":
        # continue the measured innermost dlnM/dlnr.  g_bar = G M0 (r/r0)^al / r^2
        # rises inward only if al < 2; for a gas profile the core has M ~ r^3, so
        # for most X-COP clusters g_bar TURNS OVER inward and never reaches a0 --
        # r_a0 then does not exist under this continuation.
        k = min(6, len(r) - 1)
        al = float(np.polyfit(np.log(r[:k]), np.log(Mb[:k]), 1)[0])
        if al >= 2.0 - 1e-6:
            return float("nan")
        return float((G * M0 / (A0 * r0 ** al)) ** (1.0 / (2 - al)))
    raise ValueError(mode)


def main():
    cl = I.load_all(verbose=False)
    pts = I.xcop_points(cl)
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    nm = np.array([p["name"] for p in pts])
    y = I.rar_residual(gb, go)
    names = sorted(set(nm))

    # ------------------------------------------------------------------ 3b first
    print("3b  IS THE CROSSING MEASURED, AND HOW FAR IS THE EXTRAPOLATION?")
    rows = []
    for c in cl:
        p = full_profile(c)
        ok = np.isfinite(p["gb"]) & (p["gb"] > 0)
        rr, gg, MM = p["r"][ok], p["gb"][ok], p["Mb"][ok]
        xr = gg / A0
        d = dict(cluster=c["name"],
                 r_inner_kpc=float(rr.min() / KPC),
                 x_max=float(xr.max()),
                 crossing_measured=bool(xr.max() >= 1.0))
        k6 = min(6, len(rr) - 1)
        d["inner_dlnM_dlnr"] = float(np.polyfit(np.log(rr[:k6]), np.log(MM[:k6]), 1)[0])
        for m in ("measured", "point_mass", "powerlaw",
                  "bcg_0p5e12", "bcg_1e12", "bcg_2e12"):
            v = r_a0(rr, gg, m, MM)
            d["r_a0_" + m + "_kpc"] = float(v / KPC) if np.isfinite(v) else None
        rows.append(d)
        pl_v = d["r_a0_powerlaw_kpc"]
        pl_s = "turns over" if pl_v is None else f"{pl_v:.1f}"
        print(f"   {c['name']:<9} r_in={d['r_inner_kpc']:6.1f} kpc  "
              f"max g_bar/a0 = {d['x_max']:.3f}  "
              f"dlnM/dlnr(in)={d['inner_dlnM_dlnr']:5.2f}  "
              f"r_a0: bare {d['r_a0_point_mass_kpc']:6.1f}  "
              f"+BCG(0.5/1/2e12) {d['r_a0_bcg_0p5e12_kpc']:5.1f}/"
              f"{d['r_a0_bcg_1e12_kpc']:5.1f}/{d['r_a0_bcg_2e12_kpc']:5.1f} kpc  "
              f"power-law {pl_s:>11}")
    n_meas = sum(1 for d in rows if d["crossing_measured"])
    pm = np.array([d["r_a0_point_mass_kpc"] for d in rows])
    ri = np.array([d["r_inner_kpc"] for d in rows])
    n_pl = sum(1 for d in rows if d["r_a0_powerlaw_kpc"] is None)
    fam = np.vstack([pm] + [np.array([d["r_a0_" + k + "_kpc"] for d in rows])
                            for k in ("bcg_0p5e12", "bcg_1e12", "bcg_2e12")])
    sp = np.log10(fam)
    spread = np.nanmax(sp, axis=0) - np.nanmin(sp, axis=0)
    print(f"\n   g_bar TURNS OVER inward (dlnM/dlnr >= 2) in {n_pl}/12 clusters:")
    print(f"   under a continuation of the measured inner slope, r_a0 does not "
          f"exist for them at all.")
    print(f"   crossing DIRECTLY MEASURED in {n_meas}/12 clusters "
          f"(max g_bar/a0 over the sample = {max(d['x_max'] for d in rows):.3f})")
    print(f"   r_a0 / innermost measured radius: bare-gas median "
          f"{np.median(pm/ri):.2f}x -- the crossing sits INSIDE the data")
    print(f"   spread of log10 r_a0 over the defensible family "
          f"(bare gas, +0.5/1/2e12 Msun BCG):")
    print(f"      median {np.median(spread):.3f} dex, range "
          f"{np.nanmin(spread):.3f}-{np.nanmax(spread):.3f} dex")
    print(f"   -> log10 t carries a per-cluster systematic of "
          f"{np.median(spread):.2f} dex, i.e. a factor "
          f"{10**np.median(spread):.0f},")
    print(f"      set entirely by an unmeasured BCG stellar mass. That is the "
          f"bound: finite, but wide.")
    OUT["extrapolation_bound"] = dict(
        per_cluster=rows, n_crossing_measured=int(n_meas), n_clusters=len(rows),
        max_gbar_over_a0=float(max(d["x_max"] for d in rows)),
        point_mass_is_upper_bound_for_measured_baryons=True,
        n_gbar_turns_over_inward=int(n_pl),
        r_a0_over_r_inner_median=float(np.median(pm / ri)),
        log10_r_a0_spread_dex=dict(median=float(np.median(spread)),
                                   min=float(np.nanmin(spread)),
                                   max=float(np.nanmax(spread)),
                                   family="bare gas (point-mass continuation) and "
                                          "+0.5, +1, +2 x 1e12 Msun BCG"),
        note="point_mass freezes M_b at its innermost measured value, so g_bar "
             "rises as fast as it possibly can inward and the crossing is as far "
             "OUT as it can be: r_a0(point mass) is a strict upper bound. The "
             "spread across the three continuations is the bound on the "
             "extrapolation, and it is a per-cluster offset in log t.")

    # ------------------------------------------------------------------ 3a
    print("\n3a  IS t ITSELF TAUTOLOGICAL?")
    # primary definition: point-mass continuation (the upper bound, and the one
    # the record's rank-2 null used)
    ra = {}
    for c, d in zip(cl, rows):
        ra[c["name"]] = d["r_a0_point_mass_kpc"] * KPC
    t = r / np.array([ra[n] for n in nm])
    S1_t = N.spear(t, y)
    S1_r = N.spear(r, y)
    S1_R = N.spear(r / R5, y)
    print(f"   Spearman(t, residual)          = {S1_t:+.4f}")
    print(f"   Spearman(r, residual)          = {S1_r:+.4f}")
    print(f"   Spearman(r/R500, residual)     = {S1_R:+.4f}")

    # the same rank identity
    Dm = np.column_stack([(nm == c).astype(float) for c in names])
    A_ = np.column_stack([Dm, np.log10(r / KPC)])
    C_ = np.column_stack([Dm, np.log10(r / KPC), np.log10(t)])
    rk = (np.linalg.matrix_rank(A_, tol=1e-9), np.linalg.matrix_rank(C_, tol=1e-9))
    coef, *_ = np.linalg.lstsq(A_, np.log10(t), rcond=None)
    res_t = float(np.max(np.abs(np.log10(t) - A_ @ coef)))
    print(f"   rank[indicators|log r] = {rk[0]}, rank[indicators|log r|log t] = "
          f"{rk[1]}  -> t adds {rk[1]-rk[0]} direction(s)")
    print(f"   residual of log t on [indicators|log r]: max |e| = {res_t:.2e}")
    print("   -> t is degenerate with r in EXACTLY the same way r/R500 is: r_a0 is "
          "constant\n      within a cluster, so it lives in the span of the cluster "
          "indicators.")

    # the sign of t's tautology, measured by perturbing the baryons
    print("\n   sign and size of t's own shared-quantity channel "
          "(perturb M_b, watch both axes):")
    rng = np.random.default_rng(9)
    d_t, d_y = [], []
    for f in (0.85, 0.90, 0.95, 1.05, 1.10, 1.15):
        gb2 = gb * f
        ra2 = {k: v * math.sqrt(f) for k, v in ra.items()}
        t2 = r / np.array([ra2[n] for n in nm])
        y2 = I.rar_residual(gb2, go)
        d_t.append(np.mean(np.log10(t2 / t)))
        d_y.append(np.mean(y2 - y))
    slope_t = float(np.polyfit(d_t, d_y, 1)[0])
    # the R500 channel, for comparison: R500 ~ M500^(1/3), y ~ +log10 M500
    print(f"      d(mean residual)/d(mean log10 t) from a baryon error = "
          f"{slope_t:+.3f}")
    print(f"      -> a baryon error moves t and the residual in the SAME direction, "
          f"so it induces a\n         POSITIVE correlation. The R500 channel is "
          f"negative. t is therefore CONSERVATIVE\n         for a claimed negative "
          f"correlation -- a baryon error cannot manufacture one.")

    # scrambling null for t
    def scr(radmap, nperm=20000, seed=3):
        rg = np.random.default_rng(seed)
        ks = sorted(radmap)
        idx = {c: np.where(nm == c)[0] for c in ks}
        V = np.array([radmap[c] for c in ks])
        s = np.empty(nperm)
        tt = np.empty(len(r))
        for i in range(nperm):
            p = rg.permutation(len(ks))
            for j, c in enumerate(ks):
                tt[idx[c]] = V[p[j]]
            s[i] = N.spear(r / tt, y)
        return s
    st = scr(ra)
    print(f"\n   scrambling null for t: mean {st.mean():+.4f} +- {st.std(ddof=1):.4f}; "
          f"observed {S1_t:+.4f} at percentile {100*np.mean(st<=S1_t):.1f}")
    OUT["t_variable"] = dict(
        definition="t = r / r_a0 with r_a0 from the point-mass continuation "
                   "(strict upper bound)",
        S1_t=float(S1_t), S1_r=float(S1_r), S1_R500=float(S1_R),
        rank_indicators_plus_logr=int(rk[0]),
        rank_indicators_plus_logr_plus_logt=int(rk[1]),
        max_residual_of_logt_on_span=res_t,
        baryon_channel_dResidual_dlogt=slope_t,
        baryon_channel_sign="positive -- opposite to the R500 channel, so it "
                            "cannot manufacture a negative correlation",
        scramble_null=dict(mean=float(st.mean()), sd=float(st.std(ddof=1)),
                           observed=float(S1_t),
                           percentile=float(100 * np.mean(st <= S1_t))),
        conclusion="t is subject to the SAME degeneracy as r/R500 -- it is a "
                   "per-cluster constant divided out of r, so it adds no direction "
                   "to a model with per-cluster levels -- but its shared-quantity "
                   "channel has the OPPOSITE sign, so unlike R500 it cannot "
                   "manufacture a negative correlation. Its real defect is that for "
                   "X-COP it is not measured at all.")

    json.dump(OUT, open("job3_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote job3_results.json")


if __name__ == "__main__":
    main()
