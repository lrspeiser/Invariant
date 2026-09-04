"""THE OVERLAP TEST -- the one place mass and radius are actually separable.

The three surveys barely share the r/R500 axis: strong lensing lives at
0.03-0.78, LoCuSS sits at exactly 1, and eFEDS runs from 0.22 to 33 with most
of its weight above 1.  There is exactly ONE band where two surveys overlap in
radius while differing by more than an order of magnitude in mass:

    0.22 < r/R500 < 0.78

If S depends on RADIUS, the eFEDS groups and the strong-lens clusters must
agree there.  If S depends on MASS, they must not.  This is the only
model-free discriminator the present data contain, and it does not rely on any
extrapolation.

Writes overlap_results.json.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as K
import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [(0.20, 0.80), (0.30, 0.80), (0.20, 1.30), (0.80, 1.30)]


def efeds_amp(bd, J, mask):
    """chi2-optimal amplitude of the RAR no-slip prediction on a subset.

    Profiled on a grid so the nonlinearity of g_+ in the amplitude is kept,
    with a parabolic Delta chi2 = 1 interval.
    """
    S, dS = J.project("H0", None)
    grid = np.linspace(-1.0, 1.5, 501)
    ch = []
    for a in grid:
        r = (bd.F.gplus(S, dS, 10.0 ** a) - bd.F.gt) / bd.F.er
        ch.append(float(np.sum(r[mask] ** 2)))
    ch = np.array(ch)
    i = int(np.argmin(ch))
    ok = grid[ch - ch[i] <= 1.0]
    return (10.0 ** grid[i], 10.0 ** ok.min(), 10.0 ** ok.max(),
            int(mask.sum()), float(ch[i]))


def main(theta_max=None):
    bd = K.Bundle(verbose=False, r500_mode="cat", theta_max=theta_max)
    J = F.Joint(bd)
    x_ef = np.exp(bd.ef_x["lnx"])
    M_ef = np.exp(bd.ef_x["lnM"]) * 1e14
    sl_x = np.array([r["lnx"] for r in bd.sl_rows])
    sl_S = np.array([r["lnS"] for r in bd.sl_rows])
    sl_M = np.exp(np.array([r["lnM"] for r in bd.sl_rows])) * 1e14
    sl_cid = np.array([r["cid"] for r in bd.sl_rows])
    J0 = F.Joint(bd)
    sc = F.unpack("H_P", J0.fit("H_P"))
    fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
          sc["sigma_int_sl_cluster"])

    out = {"bands": [], "sigma_int_sl": list(fs)}
    print("=" * 78)
    print("THE OVERLAP TEST -- eFEDS groups vs strong-lens clusters at the")
    print("SAME r/R500, differing by more than a decade in mass")
    print("=" * 78)
    for lo, hi in BANDS:
        m_ef = (x_ef >= lo) & (x_ef < hi)
        m_sl = (np.exp(sl_x) >= lo) & (np.exp(sl_x) < hi)
        if m_ef.sum() < 20 or m_sl.sum() < 2:
            print(f"\n   band {lo}-{hi}: eFEDS {m_ef.sum()}, SL {m_sl.sum()}"
                  " -- too few, skipped")
            continue
        a, alo, ahi, n_ef, chi = efeds_amp(bd, J, m_ef)
        # strong lensing: cluster-level mean, so the cluster-common systematic
        # is not divided away by counting image systems
        cl = sorted(set(sl_cid[m_sl]))
        per = [float(np.mean(sl_S[m_sl & (sl_cid == c)])) for c in cl]
        s_sl = float(np.mean(per))
        n_cl = len(cl)
        e_sl = (math.sqrt(fs[2] ** 2 / n_cl
                          + fs[1] ** 2 / max(int(m_sl.sum()), 1)))
        e_ef = 0.5 * (math.log(ahi) - math.log(alo))
        d = s_sl - math.log(a)
        sig = d / math.hypot(e_ef, e_sl)
        row = dict(band=[lo, hi], n_efeds=n_ef, n_sl=int(m_sl.sum()),
                   sl_clusters=cl,
                   efeds_S=a, efeds_S_lo=alo, efeds_S_hi=ahi,
                   efeds_medianM=float(np.median(M_ef[m_ef])),
                   sl_S=math.exp(s_sl), sl_medianM=float(np.median(sl_M[m_sl])),
                   mass_ratio=float(np.median(sl_M[m_sl])
                                    / np.median(M_ef[m_ef])),
                   dlnS=d, sigma=sig,
                   alpha_implied=float(d / math.log(np.median(sl_M[m_sl])
                                                    / np.median(M_ef[m_ef]))))
        out["bands"].append(row)
        print(f"\n   r/R500 in [{lo}, {hi})")
        print(f"      eFEDS   n = {n_ef:4d} points, median M_gas500 = "
              f"{row['efeds_medianM']:.2e} Msun")
        print(f"              S = {a:.3f}  [{alo:.3f}, {ahi:.3f}]")
        print(f"      SL      n = {int(m_sl.sum()):3d} image systems in "
              f"{n_cl} clusters {cl}, median M_gas500 = "
              f"{row['sl_medianM']:.2e} Msun")
        print(f"              S = {math.exp(s_sl):.3f}  +- "
              f"{e_sl:.3f} in ln")
        print(f"      mass ratio SL/eFEDS = {row['mass_ratio']:.1f}x"
              f"   at the SAME r/R500")
        print(f"      ln S difference = {d:+.3f}  ->  {sig:+.1f} sigma")
        print(f"      implied alpha if the difference is MASS: "
              f"{row['alpha_implied']:+.4f}")
    # ---- eFEDS's own radial run of S, which is its internal beta
    print("\n" + "=" * 78)
    print("eFEDS ALONE: S in radial bands.  This is the only internal radial")
    print("information any survey here has.")
    print("=" * 78)
    edges = [0.2, 0.5, 0.8, 1.3, 2.0, 3.2, 5.0, 8.0, 35.0]
    rad = []
    for i in range(len(edges) - 1):
        m = (x_ef >= edges[i]) & (x_ef < edges[i + 1])
        if m.sum() < 20:
            continue
        a, alo, ahi, n, chi = efeds_amp(bd, J, m)
        rad.append(dict(lo=edges[i], hi=edges[i + 1], n=n,
                        x_median=float(np.median(x_ef[m])), S=a,
                        S_lo=alo, S_hi=ahi,
                        M_median=float(np.median(M_ef[m]))))
        print(f"   r/R500 {edges[i]:5.2f}-{edges[i+1]:5.2f}  n={n:5d}"
              f"  median x={np.median(x_ef[m]):6.2f}"
              f"  S = {a:6.3f} [{alo:6.3f}, {ahi:6.3f}]"
              f"  median M_gas500 = {np.median(M_ef[m]):.2e}")
    out["efeds_radial"] = rad
    if len(rad) >= 3:
        lx = np.log([r["x_median"] for r in rad])
        ly = np.log([r["S"] for r in rad])
        le = np.array([0.5 * (math.log(r["S_hi"]) - math.log(r["S_lo"]))
                       for r in rad])
        Wm = 1.0 / le ** 2
        X = np.column_stack([np.ones(len(lx)), lx - lx.mean()])
        C = np.linalg.inv(X.T @ (Wm[:, None] * X))
        bco = C @ (X.T @ (Wm * ly))
        out["efeds_radial_slope"] = dict(beta=float(bco[1]),
                                         sd=float(math.sqrt(C[1, 1])))
        print(f"\n   eFEDS internal radial slope  beta = {bco[1]:+.4f}"
              f" +- {math.sqrt(C[1, 1]):.4f}"
              f"   (weighted fit to the {len(rad)} bands above)")
        need = out["bands"][0]["dlnS"] / (math.log(0.5) - math.log(2.5))
        out["beta_needed_efeds_to_sl"] = float(need)
        print(f"   For comparison, a PURE radius story joining eFEDS at"
              f" r/R500 ~ 2.5 to the\n   strong-lens cores at ~0.1 needs"
              f" beta = {need:+.4f}.")

    # ---- is that internal slope real?  three checks it has to survive
    print("\n" + "=" * 78)
    print("IS THE eFEDS INTERNAL RADIAL SLOPE REAL?  three checks")
    print("=" * 78)
    checks = {}

    def slope_on(mask_extra=None, edges=None):
        e = edges or [0.2, 0.5, 0.8, 1.3, 2.0, 3.2, 5.0, 8.0, 35.0]
        xs, ys, es = [], [], []
        for i in range(len(e) - 1):
            m = (x_ef >= e[i]) & (x_ef < e[i + 1])
            if mask_extra is not None:
                m = m & mask_extra
            if m.sum() < 20:
                continue
            a, alo, ahi, n, _ = efeds_amp(bd, J, m)
            if not (alo > 0 and ahi > alo):
                continue
            xs.append(math.log(float(np.median(x_ef[m]))))
            ys.append(math.log(a))
            es.append(0.5 * (math.log(ahi) - math.log(alo)))
        if len(xs) < 3:
            return None, None, len(xs)
        xs, ys, es = np.array(xs), np.array(ys), np.array(es)
        Wm = 1.0 / es ** 2
        X = np.column_stack([np.ones(len(xs)), xs - xs.mean()])
        C = np.linalg.inv(X.T @ (Wm[:, None] * X))
        bb = C @ (X.T @ (Wm * ys))
        return float(bb[1]), float(math.sqrt(C[1, 1])), len(xs)

    # 1. the closure lane's declared train/held split, by sorted system id
    order = np.argsort([c.id for c in bd.ef])
    tr = np.zeros(len(bd.ef), bool)
    tr[np.sort(order[0::2])] = True
    m_tr = tr[bd.F.sysi]
    b1, e1, n1_ = slope_on(m_tr)
    b2, e2, n2_ = slope_on(~m_tr)
    checks["split"] = dict(train=[b1, e1], held=[b2, e2])
    print(f"   1. declared train/held split (the closure lane's, by sorted id)")
    print(f"      TRAIN beta = {b1:+.4f} +- {e1:.4f} ({n1_} bands)")
    print(f"      HELD  beta = {b2:+.4f} +- {e2:.4f} ({n2_} bands)")
    print(f"      difference {b1 - b2:+.4f} +- {math.hypot(e1, e2):.4f}"
          f"  ->  {(b1 - b2) / math.hypot(e1, e2):+.1f} sigma")

    # 2. restrict to where the X-ray density fit is NOT wildly extrapolated
    b3, e3, n3_ = slope_on(edges=[0.2, 0.5, 0.8, 1.3, 2.0, 3.2])
    checks["inside_3R500"] = [b3, e3]
    print(f"\n   2. r/R500 < 3.2 only, where the Vikhlinin fit is not far"
          f" extrapolated")
    print(f"      beta = {b3:+.4f} +- {e3:.4f} ({n3_} bands)")

    # 3. B-mode: the same statistic on the cross component, which must be zero
    gt_save = bd.F.gt.copy()
    bd.F.gt = bd.F.gx.copy()
    b4, e4, n4_ = slope_on()
    bd.F.gt = gt_save
    checks["bmode"] = [b4, e4]
    print(f"\n   3. the SAME statistic on the B-mode (cross) component, which"
          f" must be flat")
    print(f"      beta = {b4:+.4f} +- {e4:.4f} ({n4_} bands)")
    out["internal_slope_checks"] = checks

    # ---- the SECOND matched-radius comparison: eFEDS vs LoCuSS at r/R500 ~ 1
    print("\n" + "=" * 78)
    print("MATCHED-RADIUS COMPARISON 2: eFEDS vs LoCuSS at r/R500 ~ 1")
    print("=" * 78)
    m = (x_ef >= 0.8) & (x_ef < 1.3)
    a, alo, ahi, n, _ = efeds_amp(bd, J, m)
    lo_S = np.array([r["lnS"] for r in bd.lo_rows])
    lo_M = np.exp(np.array([r["lnM"] for r in bd.lo_rows])) * 1e14
    lo_e = np.array([r["e_stat"] for r in bd.lo_rows])
    e_lo = math.sqrt(np.sum(1.0 / (lo_e ** 2 + fs[0] ** 2))) ** -1
    mu_lo = float(np.sum(lo_S / (lo_e ** 2 + fs[0] ** 2))
                  / np.sum(1.0 / (lo_e ** 2 + fs[0] ** 2)))
    e_ef = 0.5 * (math.log(ahi) - math.log(alo))
    mr = float(np.median(lo_M) / np.median(M_ef[m]))
    d = mu_lo - math.log(a)
    al2 = d / math.log(mr)
    out["matched_r1"] = dict(efeds_S=a, efeds_ci=[alo, ahi], efeds_n=n,
                             efeds_M=float(np.median(M_ef[m])),
                             locuss_S=math.exp(mu_lo), locuss_n=len(lo_S),
                             locuss_M=float(np.median(lo_M)),
                             mass_ratio=mr, dlnS=d,
                             sigma=d / math.hypot(e_ef, e_lo),
                             alpha_implied=al2)
    print(f"   eFEDS  r/R500 0.8-1.3, n = {n}, median M_gas500 = "
          f"{np.median(M_ef[m]):.2e}:  S = {a:.3f} [{alo:.3f}, {ahi:.3f}]")
    print(f"   LoCuSS r/R500 = 1 exactly, n = {len(lo_S)}, median M_gas500 = "
          f"{np.median(lo_M):.2e}:  S = {math.exp(mu_lo):.3f} +- {e_lo:.3f}"
          f" in ln")
    print(f"   mass ratio {mr:.1f}x at the SAME radius"
          f"  ->  ln S difference {d:+.3f}"
          f" ({d / math.hypot(e_ef, e_lo):+.1f} sigma)")
    print(f"   implied alpha = {al2:+.4f}")
    b0 = out["bands"][0]
    print(f"\n   *** The two matched-radius comparisons DISAGREE: alpha ="
          f" {b0['alpha_implied']:+.3f}")
    print(f"   from eFEDS-vs-SL at r/R500 ~ 0.5, but {al2:+.3f} from"
          f" eFEDS-vs-LoCuSS at")
    print(f"   r/R500 ~ 1, over comparable mass ratios ({b0['mass_ratio']:.0f}x"
          f" and {mr:.0f}x).")
    print(f"   A single mass power law cannot produce both.")
    out["alpha_disagreement"] = dict(
        at_half_R500=b0["alpha_implied"], at_R500=al2,
        ratio=float(b0["alpha_implied"] / al2) if al2 else None)

    # ---- INTERNAL radial slope of the strong-lens clusters themselves
    print("\n" + "=" * 78)
    print("STRONG-LENS CLUSTERS: their OWN internal radial slope, at fixed")
    print("mass, from many image systems in one cluster")
    print("=" * 78)
    per_cl = {}
    for c in sorted(set(sl_cid)):
        m2 = sl_cid == c
        if m2.sum() < 5:
            continue
        lx = sl_x[m2]
        if lx.max() - lx.min() < 0.7:
            print(f"   {c:10s} n={int(m2.sum()):2d}  ln r range only "
                  f"{lx.max() - lx.min():.2f} -- too narrow, skipped")
            continue
        ly = sl_S[m2]
        co = np.polyfit(lx, ly, 1)
        res = ly - np.polyval(co, lx)
        sd = float(np.std(res, ddof=2))
        se = sd / math.sqrt(np.sum((lx - lx.mean()) ** 2))
        per_cl[str(c)] = dict(n=int(m2.sum()), beta=float(co[0]), sd=se,
                              lnr_range=float(lx.max() - lx.min()))
        print(f"   {c:10s} n={int(m2.sum()):2d}  ln r span "
              f"{lx.max() - lx.min():.2f}  beta = {co[0]:+.4f} +- {se:.4f}")
    out["sl_internal_beta"] = per_cl
    if per_cl:
        wts = np.array([1.0 / v["sd"] ** 2 for v in per_cl.values()])
        bs = np.array([v["beta"] for v in per_cl.values()])
        bb = float(np.sum(wts * bs) / np.sum(wts))
        se = float(1.0 / math.sqrt(np.sum(wts)))
        out["sl_internal_beta_combined"] = dict(beta=bb, sd=se)
        print(f"   combined  beta = {bb:+.4f} +- {se:.4f}"
              f"   (eFEDS internal: {out['efeds_radial_slope']['beta']:+.4f}"
              f" +- {out['efeds_radial_slope']['sd']:.4f})")

    out["theta_max"] = theta_max
    return out


if __name__ == "__main__":
    all_out = {}
    for tm in (None, 100.0):
        key = "no_theta_cut" if tm is None else f"theta_lt_{int(tm)}"
        print("\n\n" + "#" * 78)
        print(f"#  STRONG-LENS SELECTION: {key}")
        print("#" * 78)
        all_out[key] = main(tm)
    with open(os.path.join(HERE, "overlap_results.json"), "w") as fh:
        json.dump(all_out, fh, indent=1, default=float)
    print("\n   wrote overlap_results.json")
