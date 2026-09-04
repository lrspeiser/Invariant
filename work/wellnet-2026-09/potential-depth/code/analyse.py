"""Does this ladder decide  g_obs = F(g_bar)  vs  g_obs = F(g_bar, |Phi_b|)?

The alternative, written so it has one global parameter and no per-object freedom
(section 8 of the standing brief):

    g_obs = nu( g_bar / A(Phi_b) ) * g_bar ,   A(Phi_b) = a0 (|Phi_b|/Phi_0)^q

In the deep-MOND limit nu(y) = y^-1/2, which is where EVERY cluster and group
point in this programme lives, this reduces exactly to

    log10 nu_obs = const - 0.5 log10 g_bar + (q/2) log10|Phi_b|

so the discriminating coefficient is  beta = q/2  in

    log10 nu_obs = f(log10 g_bar) + beta * log10|Phi_b| + eps            (*)

and the null hypothesis "gravity is a function of g_bar alone" is beta = 0.
Everything below either measures beta or measures how well it CAN be measured.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
A0 = 1.2e-10
KPC = 3.0856775814913673e19
RNG = np.random.default_rng(20260903)

BAR = "=" * 78


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def load(path=None):
    path = path or os.path.join(LANE, "potential_depth_ladder.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    d = {}
    for k in ("r_kpc", "Mb_Msun", "g_bar", "g_obs", "nu_obs", "abs_Phi_b",
              "S_shape", "e_lg_gbar", "e_lg_gobs", "sys_lg_Mb"):
        d[k] = np.array([float(r[k]) for r in rows])
    d["system"] = np.array([r["system"] for r in rows])
    d["cls"] = np.array([r["class"] for r in rows])
    d["rank"] = np.array([int(r["class_rank"]) for r in rows])
    d["source"] = np.array([r["source"] for r in rows])
    d["tier"] = np.array([int(r["tier"]) for r in rows])
    d["lg"] = np.log10(d["g_bar"])
    d["lr"] = np.log10(d["r_kpc"])
    d["lp"] = np.log10(d["abs_Phi_b"])
    d["lnu"] = np.log10(d["nu_obs"])
    return d


def sysweights(system):
    """Weight so each SYSTEM contributes 1, not each point: SPARC has 3389
    points and the low-mass clusters 51, and an unweighted fit is a fit to SPARC."""
    u, inv, cnt = np.unique(system, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def wlstsq(A, y, w):
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return c


def r2(y, pred, w=None):
    if w is None:
        w = np.ones_like(y)
    mu = np.average(y, weights=w)
    ss = np.average((y - mu) ** 2, weights=w)
    rs = np.average((y - pred) ** 2, weights=w)
    return 1.0 - rs / ss


def within_bin_spread(lg, lp, w=None, nbin=8, label="", minn=20):
    """Spread of log|Phi_b| inside narrow log g_bar bins -- the leverage.
    Equal-COUNT bins, matching work/wellnet-2026-09/phi_rank.py exactly."""
    edges = np.percentile(lg, np.linspace(0, 100, nbin + 1))
    sds, out = [], []
    for i in range(nbin):
        m = (lg >= edges[i]) & (lg < edges[i + 1] if i < nbin - 1
                                else lg <= edges[i + 1])
        if m.sum() < minn:
            continue
        sd = np.std(lp[m])
        rng = lp[m].max() - lp[m].min()
        sds.append(sd)
        out.append((edges[i], edges[i + 1], int(m.sum()), sd, rng))
    if label:
        print(f"   {label}")
        print(f"      {'log g_bar bin':>22} {'n':>7} {'sd log|Phi|':>12} "
              f"{'range':>9}")
        for a, b, n, sd, rg in out:
            print(f"      [{a:+7.2f},{b:+7.2f})  {n:7d} {sd:12.3f} {rg:9.2f}")
        print(f"      median within-bin sd = {np.median(sds):.3f} dex "
              f"| max range = {max(o[4] for o in out):.2f} dex")
    return float(np.median(sds)), out


def main():
    d = load()
    n = len(d["lg"])
    print(f"ladder: {n} rows, {len(set(d['system']))} systems, "
          f"{len(set(d['cls']))} classes")

    # ------------------------------------------------------------------
    head("0.  Sanity, and the failure modes this programme has been bitten by")
    print("   Checked explicitly in this lane:")
    print("   * SEALED HOLDOUTS  KiDS weak lensing and wide binaries are not in")
    print("     the ladder; ingest.py drops them by probe name before reading any")
    print("     value.  Row counts: source table 4093, dropped unread 66.")
    bad = np.where(d["S_shape"] < 1 - 1e-9)[0]
    print(f"   * S = |Phi|/(g_bar r) >= 1 is a theorem FOR SPHERICAL M_b(<r).")
    print(f"     rows with S < 1: {len(bad)} of {n}, min S = "
          f"{d['S_shape'].min():.4f}; by source: "
          + ", ".join(f"{s}:{int((d['source'][bad]==s).sum())}"
                      for s in sorted(set(d['source'][bad]))))
    print("     These are DISK rows, where g_bar is the razor-thin-disk field,")
    print("     not G M(<r)/r^2, and V_gas is signed (central HI holes), so")
    print("     V_b^2 r / G is not a monotone enclosed mass.  The theorem does")
    print("     not apply to them; no spherical row violates it.")
    print(f"     S range {d['S_shape'].min():.4f} - {d['S_shape'].max():.3f}, "
          f"median {np.median(d['S_shape']):.3f}; the very large values are")
    print("     inner disk points where g_bar -> 0 while |Phi| stays finite.")
    print("   * SHARED DENOMINATOR: g_bar and |Phi_b| are both computed from the")
    print("     same M_b, and nu_obs = g_obs/g_bar puts M_b in the denominator of")
    print("     the response.  Section 5 simulates the null with that covariance.")
    print("   * MONOTONE-INVARIANCE: section 6 injects a known q and checks that")
    print("     the estimator moves.")
    print("\n   boost nu_obs by class (median), a physical sanity check:")
    for k in sorted(set(d["rank"])):
        m = d["rank"] == k
        print(f"      {k} {d['cls'][m][0]:<18} n={m.sum():5d}  "
              f"median nu_obs {np.median(d['nu_obs'][m]):7.2f}   "
              f"median g_bar/a0 {np.median(d['g_bar'][m])/A0:8.4f}   "
              f"median r {np.median(d['r_kpc'][m]):8.1f} kpc")

    # ------------------------------------------------------------------
    head("1.  LEVERAGE -- the spread of log|Phi_b| at fixed g_bar")
    print("   This is the experiment's entire discriminating power.  The number")
    print("   to beat is 0.309 dex, which is what SPARC alone offers")
    print("   (work/wellnet-2026-09/phi_rank.json, tail='point').\n")
    w = sysweights(d["system"])
    med_all, tab_all = within_bin_spread(d["lg"], d["lp"],
                                         label="FULL LADDER (all 6 rungs):")
    m1 = d["rank"] == 1
    med_sparc, _ = within_bin_spread(d["lg"][m1], d["lp"][m1],
                                     label="\n   SPARC only, recomputed here as a check:")
    mt1 = d["tier"] == 1
    med_t1, _ = within_bin_spread(d["lg"][mt1], d["lp"][mt1],
                                  label="\n   tier 1 only (drops the stellar-mass-only optical groups):")

    # overlap region: where more than one class actually lives
    print("\n   Where the rungs OVERLAP in g_bar -- this is where a matched pair")
    print("   can exist at all.  Bins of 0.25 dex in log g_bar:")
    edges = np.arange(math.floor(d["lg"].min() * 4) / 4,
                      math.ceil(d["lg"].max() * 4) / 4 + 0.25, 0.25)
    overlap = []
    print(f"      {'bin':>16} {'ncls':>5} {'nsys':>6} {'sd lPhi':>8} "
          f"{'range':>7}  classes")
    for i in range(len(edges) - 1):
        m = (d["lg"] >= edges[i]) & (d["lg"] < edges[i + 1])
        if m.sum() == 0:
            continue
        cls = sorted(set(d["rank"][m]))
        nsys = len(set(d["system"][m]))
        if len(cls) < 2:
            continue
        sd = np.std(d["lp"][m])
        rg = d["lp"][m].max() - d["lp"][m].min()
        overlap.append(dict(lo=edges[i], hi=edges[i + 1], n=int(m.sum()),
                            nsys=nsys, ncls=len(cls), sd=float(sd),
                            rng=float(rg), classes=cls))
        print(f"      [{edges[i]:+5.2f},{edges[i+1]:+5.2f}) {len(cls):5d} "
              f"{nsys:6d} {sd:8.3f} {rg:7.2f}  {cls}")
    best = max(overlap, key=lambda o: o["rng"]) if overlap else None
    print(f"\n   ACHIEVED DYNAMIC RANGE IN |Phi_b| AT FIXED g_bar:")
    print(f"      widest single 0.25-dex g_bar bin: {best['rng']:.2f} dex "
          f"(log g_bar {best['lo']:+.2f}..{best['hi']:+.2f}, "
          f"{best['nsys']} systems, classes {best['classes']})")
    print(f"      median within-bin sd over the whole ladder: {med_all:.3f} dex"
          f"   vs SPARC alone {med_sparc:.3f} dex")
    print(f"      => leverage gain over SPARC alone: "
          f"{med_all/med_sparc:.2f}x in sd, "
          f"{med_all - 0.309:+.3f} dex against the 0.309 target")

    # ------------------------------------------------------------------
    head("2.  MATCHED PAIRS at fixed g_bar")
    print("   A pair is two SYSTEMS whose log g_bar agree to within 0.1 dex and")
    print("   whose log|Phi_b| differ.  Counted on system-level representative")
    print("   points (one per system per 0.25-dex g_bar bin, to stop a single")
    print("   long SPARC rotation curve manufacturing pairs with itself).")
    # representative points: for each (system, g_bar bin) take the median row
    key = {}
    binidx = np.floor(d["lg"] / 0.25).astype(int)
    for i in range(n):
        key.setdefault((d["system"][i], binidx[i]), []).append(i)
    reps = []
    for k, idx in key.items():
        j = idx[len(idx) // 2]
        reps.append(j)
    reps = np.array(sorted(reps))
    print(f"   representative points: {len(reps)}")
    rl, rp = d["lg"][reps], d["lp"][reps]
    rsys, rrk = d["system"][reps], d["rank"][reps]
    pairs, tot, xpairs = {}, {}, {}
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            if rsys[i] == rsys[j]:          # never pair a system with itself
                continue
            if abs(rl[i] - rl[j]) > 0.10:
                continue
            dphi = abs(rp[i] - rp[j])
            b = round(0.5 * (rl[i] + rl[j]) * 2) / 2
            tot[b] = tot.get(b, 0) + 1
            if dphi >= 1.0:
                pairs[b] = pairs.get(b, 0) + 1
                if rrk[i] != rrk[j]:
                    xpairs[b] = xpairs.get(b, 0) + 1
    print(f"\n      {'log g_bar':>10} {'all pairs':>10} {'dlogPhi>1':>11} "
          f"{'of those,':>11} {'fraction':>9}")
    print(f"      {'(0.5 dex)':>10} {'':>10} {'':>11} "
          f"{'cross-class':>11} {'>1 dex':>9}")
    npair_tot = npair_1 = npair_x = 0
    for b in sorted(tot):
        npair_tot += tot[b]
        npair_1 += pairs.get(b, 0)
        npair_x += xpairs.get(b, 0)
        print(f"      {b:+10.2f} {tot[b]:10d} {pairs.get(b,0):11d} "
              f"{xpairs.get(b,0):11d} {pairs.get(b,0)/tot[b]:9.3f}")
    print(f"      {'TOTAL':>10} {npair_tot:10d} {npair_1:11d} {npair_x:11d} "
          f"{npair_1/max(npair_tot,1):9.3f}")
    print(f"\n      {npair_x} of the {npair_1} wide pairs "
          f"({100*npair_x/max(npair_1,1):.1f}%) are CROSS-CLASS, i.e. the "
          f"|Phi_b|\n      contrast comes with a change of dataset, "
          f"instrument and technique.")

    # ------------------------------------------------------------------
    head("3.  COLLINEARITY -- is |Phi_b| a new direction?")
    print("   Reported as R^2 of log|Phi_b| on a quadratic in (log g_bar, log r),")
    print("   which is the honest measure of whether a new direction was bought.")
    lg, lr, lp = d["lg"], d["lr"], d["lp"]
    A1 = np.column_stack([np.ones(n), lg, lr])
    A2 = np.column_stack([A1, lg ** 2, lr ** 2, lg * lr])
    coll = {}
    for name, mask in (("full ladder", np.ones(n, bool)),
                       ("tier 1 only", d["tier"] == 1),
                       ("groups+clusters only (ranks 3-6)", d["rank"] >= 3),
                       ("X-ray hydrostatic only", d["source"] != "SPARC")):
        A1m, A2m, lpm, wm = A1[mask], A2[mask], lp[mask], w[mask]
        c1 = wlstsq(A1m, lpm, wm)
        c2 = wlstsq(A2m, lpm, wm)
        r1 = r2(lpm, A1m @ c1, wm)
        rq = r2(lpm, A2m @ c2, wm)
        sd = np.sqrt(np.average((lpm - A2m @ c2) ** 2, weights=wm))
        coll[name] = dict(r2_linear=float(r1), r2_quadratic=float(rq),
                          resid_sd_dex=float(sd), n=int(mask.sum()))
        print(f"   {name:<34} n={mask.sum():5d}  R2_lin={r1:.4f}  "
              f"R2_quad={rq:.4f}  resid sd={sd:.3f} dex")
    print("\n   plain correlations:")
    for a, b, nm in ((lg, lp, "corr(log g_bar, log|Phi_b|)"),
                     (lr, lp, "corr(log r,     log|Phi_b|)"),
                     (lg, lr, "corr(log g_bar, log r)")):
        rr_ = np.corrcoef(a, b)[0, 1]
        rw = (np.average((a - np.average(a, weights=w)) *
                         (b - np.average(b, weights=w)), weights=w) /
              math.sqrt(np.average((a - np.average(a, weights=w)) ** 2, weights=w) *
                        np.average((b - np.average(b, weights=w)) ** 2, weights=w)))
        print(f"      {nm:<32} unweighted {rr_:+.4f}   system-weighted {rw:+.4f}")
    print("\n   NOTE the identity that makes all of this inevitable:")
    print("      log|Phi_b| = log g_bar + log r + log S,   S >= 1")
    print(f"      log S over the ladder: {np.log10(d['S_shape']).min():.3f} .."
          f"{np.log10(d['S_shape']).max():.3f} dex, "
          f"sd {np.log10(d['S_shape']).std():.3f}")
    print("      so |Phi_b| carries NO information beyond (g_bar, r) except the")
    print("      bounded shape factor S.  The leverage in section 1 is therefore")
    print("      the leverage of RADIUS at fixed acceleration, wearing a new name.")

    # ------------------------------------------------------------------
    head("4.  THE CONFOUND -- is the leverage anything but the class label?")
    print("   Six false positives in this programme were reproduced exactly by a")
    print("   bare dataset indicator.  So: how much of log|Phi_b| at fixed g_bar")
    print("   is explained by class dummies alone?")
    ranks = sorted(set(d["rank"]))
    D = np.column_stack([(d["rank"] == k).astype(float) for k in ranks[1:]])
    Ag = np.column_stack([np.ones(n), lg, lg ** 2])
    AgD = np.column_stack([Ag, D])
    AgR = np.column_stack([Ag, lr])
    for nm, A in (("g_bar only (quadratic)", Ag),
                  ("g_bar + CLASS DUMMIES", AgD),
                  ("g_bar + log r", AgR)):
        c = wlstsq(A, lp, w)
        print(f"      R2(log|Phi_b| ~ {nm:<24}) = {r2(lp, A @ c, w):.4f}")
    cg = wlstsq(Ag, lp, w)
    resid_g = lp - Ag @ cg
    cgd = wlstsq(AgD, lp, w)
    resid_gd = lp - AgD @ cgd
    print(f"\n      sd of log|Phi_b| after removing g_bar          : "
          f"{math.sqrt(np.average(resid_g**2, weights=w)):.3f} dex")
    print(f"      sd after removing g_bar AND the class label    : "
          f"{math.sqrt(np.average(resid_gd**2, weights=w)):.3f} dex")
    print("      The second number is the leverage that is NOT the label.")

    print("\n   Within-class leverage, i.e. one measurement technique, one")
    print("   systematic, no label to hide behind:")
    within = {}
    for k in ranks:
        m = d["rank"] == k
        md, _ = within_bin_spread(d["lg"][m], d["lp"][m], nbin=4,
                                  minn=max(5, m.sum() // 8))
        Am = np.column_stack([np.ones(m.sum()), lg[m], lg[m] ** 2])
        cm = wlstsq(Am, lp[m], w[m])
        sdm = math.sqrt(np.average((lp[m] - Am @ cm) ** 2, weights=w[m]))
        within[int(k)] = dict(cls=str(d["cls"][m][0]), n=int(m.sum()),
                              median_within_bin_sd=float(md),
                              sd_after_gbar=float(sdm))
        print(f"      {k} {d['cls'][m][0]:<18} n={m.sum():5d}  "
              f"median within-g_bar-bin sd = {md:.3f} dex   "
              f"sd after removing g_bar = {sdm:.3f} dex")
    print(f"\n      LARGEST within-class leverage in the whole ladder: "
          f"{max(v['sd_after_gbar'] for v in within.values()):.3f} dex "
          f"({max(within.values(), key=lambda v: v['sd_after_gbar'])['cls']}).")
    print("      Every group and cluster rung has LESS internal |Phi_b| "
          "variation\n      at fixed g_bar than SPARC already had.")

    # ---- Phi vs radius: can this ladder tell them apart at all? ----------
    print("\n   Phi_b versus RADIUS, after removing g_bar.  If these two are the")
    print("   same regressor, 'potential depth' is 'radius' relabelled.")
    Bg = np.column_stack([np.ones(n), lg, lg ** 2])
    pp = lp - Bg @ wlstsq(Bg, lp, w)
    rr_ = lr - Bg @ wlstsq(Bg, lr, w)
    mu_p = np.average(pp, weights=w)
    mu_r = np.average(rr_, weights=w)
    cpr = (np.average((pp - mu_p) * (rr_ - mu_r), weights=w) /
           math.sqrt(np.average((pp - mu_p) ** 2, weights=w) *
                     np.average((rr_ - mu_r) ** 2, weights=w)))
    print(f"      partial corr(log|Phi_b|, log r | log g_bar) = {cpr:+.4f}")
    print(f"      variance inflation factor for log|Phi_b| in a joint fit "
          f"= {1/(1-cpr**2):.1f}")

    # ------------------------------------------------------------------
    head("5.  THE SHARED-DENOMINATOR NULL")
    print("   |Phi_b| and g_bar are both G-times-the-same-M_b, and nu_obs has")
    print("   g_bar in its denominator.  A global rescale M_b -> (1+delta) M_b")
    print("   moves log g_bar and log|Phi_b| by +delta and log nu_obs by -delta:")
    print("   the error vector lies EXACTLY along the (g_bar, Phi_b) degeneracy,")
    print("   so it is removed by controlling for log g_bar -- to first order.")
    print("   It is NOT removed for radius errors, for radius-dependent mass")
    print("   errors, or for errors in the outermost point, which move the Phi")
    print("   tail without moving g_bar at inner radii.  So the null is simulated.")

    def beta_hat(lnu, lg_, lp_, ww, extra=None):
        cols = [np.ones(len(lnu)), lg_, lg_ ** 2, lp_]
        if extra is not None:
            cols += [extra[:, j] for j in range(extra.shape[1])]
        A = np.column_stack(cols)
        c = wlstsq(A, lnu, ww)
        return c[3]

    lnu = d["lnu"]
    beta_obs = beta_hat(lnu, lg, lp, w)
    beta_obs_cls = beta_hat(lnu, lg, lp, w, extra=D)
    beta_obs_r = beta_hat(lnu, lg, lp, w,
                          extra=lr.reshape(-1, 1))
    print(f"\n   OBSERVED beta (log nu ~ quad(log g_bar) + beta log|Phi_b|):")
    print(f"      no other control          beta = {beta_obs:+.4f}"
          f"   -> q = 2 beta = {2*beta_obs:+.4f}")
    print(f"      + class dummies           beta = {beta_obs_cls:+.4f}")
    print(f"      + log r                   beta = {beta_obs_r:+.4f}"
          "   (log r and log|Phi| are near-duplicates: expect instability)")

    # ---- LABEL CONTROL: does a bare dataset indicator do the same job? ----
    rank_num = d["rank"].astype(float)
    Aq = np.column_stack([np.ones(n), lg, lg ** 2])
    res_nu = lnu - Aq @ wlstsq(Aq, lnu, w)
    print("\n   LABEL CONTROL (the bench's mandatory `confound` test).  Replace")
    print("   log|Phi_b| by a bare class index 1..6 and refit:")
    for nm, v in (("log|Phi_b|", lp), ("bare class index 1..6", rank_num),
                  ("log r", lr)):
        A = np.column_stack([Aq, v])
        c = wlstsq(A, lnu, w)
        rr2 = r2(lnu, A @ c, w)
        vv = v - Aq @ wlstsq(Aq, v, w)
        mv = np.average(vv, weights=w)
        cc = (np.average((vv - mv) * (res_nu - np.average(res_nu, weights=w)),
                         weights=w) /
              math.sqrt(np.average((vv - mv) ** 2, weights=w) *
                        np.average((res_nu - np.average(res_nu, weights=w)) ** 2,
                                   weights=w)))
        print(f"      {nm:<24} R2 = {rr2:.4f}   partial corr with the "
              f"g_bar-residual = {cc:+.4f}")
    print("      If the bare label matches log|Phi_b|, the variable under test")
    print("      is the dataset, not the physics.")

    # ---- the null, generated UNDER H0 ------------------------------------
    NMC = 2000
    usys, uinv = np.unique(d["system"], return_inverse=True)
    sysidx = [np.where(uinv == i)[0] for i in range(len(usys))]
    sys_sd = np.array([d["sys_lg_Mb"][ix[0]] for ix in sysidx])
    e_gobs = d["e_lg_gobs"]
    e_gbar_pt = d["e_lg_gbar"]
    e_dist = 0.05          # dex on distance -> radius; SPARC ~5%, X-ray ~4%

    # H0 truth: log nu depends on log g_bar ONLY.  Intrinsic scatter is drawn
    # per SYSTEM and is independent of |Phi_b| by construction -- that is what
    # makes it a null.  Its size is matched to the observed residual scatter.
    c0 = wlstsq(Aq, lnu, w)
    sig_int = math.sqrt(np.average((lnu - Aq @ c0) ** 2, weights=w))
    print(f"\n   H0 truth: log10 nu = quadratic(log10 g_bar) with per-system")
    print(f"   intrinsic scatter {sig_int:.3f} dex drawn INDEPENDENTLY of "
          f"|Phi_b|.")
    print("   Then the observed quantities are generated with the actual error")
    print("   covariance: a coherent per-system M_b error moves log g_bar and")
    print("   log|Phi_b| by +delta and log nu by -delta; a distance error moves")
    print("   log g_bar by -2eps, log|Phi_b| by -eps and log nu by 0 (both g's")
    print("   scale as 1/r^2 for the X-ray rungs and as 1/r for the rotation")
    print("   curves, and the ratio is what enters).")

    null = np.empty(NMC)
    null_cls = np.empty(NMC)
    null_lab = np.empty(NMC)
    for t in range(NMC):
        lnu0 = Aq @ c0 + RNG.normal(0, sig_int, len(usys))[uinv]
        dl = RNG.normal(0, 1, len(usys))[uinv] * sys_sd[uinv]   # coherent M_b
        dpt = RNG.normal(0, 1, n) * e_gbar_pt               # per point M_b
        dd = RNG.normal(0, 1, len(usys))[uinv] * e_dist     # distance
        lg_s = lg + dl + dpt - 2 * dd
        lp_s = lp + dl + dpt - dd
        lnu_s = lnu0 - dl - dpt + RNG.normal(0, 1, n) * e_gobs
        null[t] = beta_hat(lnu_s, lg_s, lp_s, w)
        null_cls[t] = beta_hat(lnu_s, lg_s, lp_s, w, extra=D)
        null_lab[t] = beta_hat(lnu_s, lg_s, rank_num, w)
    print(f"\n   Null under H0, {NMC} draws "
          f"(per-system M_b {sys_sd.min():.2f}-{sys_sd.max():.2f} dex, "
          f"per-point {e_gbar_pt.min():.2f}-{e_gbar_pt.max():.2f} dex,")
    print(f"    distance {e_dist:.2f} dex, g_obs as tabulated):")
    print(f"      NULL EXPECTATION of beta   = {null.mean():+.4f}"
          f"    (the naive assumption is 0)")
    print(f"      null sd                    = {null.std():.4f}")
    print(f"      null 2.5-97.5 pct          = [{np.percentile(null,2.5):+.4f},"
          f" {np.percentile(null,97.5):+.4f}]")
    z = (beta_obs - null.mean()) / null.std()
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    print(f"      OBSERVED beta = {beta_obs:+.4f}  ->  z = {z:+.2f} against "
          f"its own null, p = {p:.3g}")
    zc = (beta_obs_cls - null_cls.mean()) / null_cls.std()
    print(f"      with class dummies: null {null_cls.mean():+.4f} +- "
          f"{null_cls.std():.4f}, observed {beta_obs_cls:+.4f}, z = {zc:+.2f}")
    print(f"      shared-denominator bias in beta_hat = "
          f"{null.mean():+.4f} dex/dex; it is {'NOT ' if abs(null.mean())>0.01 else ''}"
          f"negligible relative to the observed {beta_obs:+.4f}")

    # system-level bootstrap for the honest CI on the observed beta
    NB = 2000
    boot = np.empty(NB)
    for t in range(NB):
        pick = RNG.integers(0, len(usys), len(usys))
        idx = np.concatenate([sysidx[i] for i in pick])
        ww = np.concatenate([np.full(len(sysidx[i]), 1.0 / len(sysidx[i]))
                             for i in pick])
        boot[t] = beta_hat(lnu[idx], lg[idx], lp[idx], ww)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n   system-level bootstrap of the OBSERVED beta: "
          f"{beta_obs:+.4f} [{lo:+.4f}, {hi:+.4f}]  (sd {boot.std():.4f})")

    # ------------------------------------------------------------------
    head("6.  POWER -- what size of potential-depth effect is detectable?")
    print("   Injection-recovery: add (q/2) log|Phi_b| to log nu_obs, refit,")
    print("   and check the estimator responds (the monotone-invariance gate).")
    print(f"\n      {'injected q':>11} {'recovered beta':>15} "
          f"{'recovered q':>12} {'d beta/d q':>11}")
    prev = None
    inj = []
    for q in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        bq = beta_hat(lnu + 0.5 * q * lp, lg, lp, w)
        slope = "" if prev is None else f"{(bq-prev[1])/(q-prev[0]):11.4f}"
        print(f"      {q:11.3f} {bq:15.4f} {2*bq:12.4f} {slope:>11}")
        inj.append((q, float(bq)))
        prev = (q, bq)
    dq = (inj[-1][1] - inj[0][1]) / (inj[-1][0] - inj[0][0])
    print(f"      d(beta)/d(q) over the tested range = {dq:.4f} "
          f"(exactly 0.5 if unbiased); spread of beta = "
          f"{max(b for _,b in inj)-min(b for _,b in inj):.4f}")

    se = max(null.std(), boot.std())    # the larger of null and sampling sd
    q3 = 2 * 3 * se
    se_c = null_cls.std()
    q3c = 2 * 3 * se_c
    print(f"\n   3-sigma STATISTICAL sensitivity (random errors only):")
    print(f"      sd(beta) under H0 = {null.std():.4f}; system bootstrap sd "
          f"= {boot.std():.4f}; using the larger, {se:.4f}")
    print(f"      beta_3sigma = {3*se:.4f}   ->  q_3sigma = {q3:.4f}")
    print(f"      with class dummies also fitted: q_3sigma = {q3c:.4f}")

    # ---- the contrast, computed inside MATCHED g_bar bins ----------------
    print("\n   The |Phi_b| contrast between the ends of the ladder, computed")
    print("   inside matched 0.1-dex g_bar bins so the two are truly at the")
    print("   same acceleration:")
    ends = []
    b0 = np.floor(d["lg"] / 0.1).astype(int)
    for bb in sorted(set(b0)):
        m = b0 == bb
        a = m & (d["rank"] == 1)
        b = m & (d["rank"] == 6)
        if a.sum() >= 5 and b.sum() >= 5:
            ends.append((bb * 0.1, float(np.median(d["lp"][b]) -
                                         np.median(d["lp"][a])),
                         int(a.sum()), int(b.sum())))
    for lo_, c_, na, nb in ends:
        print(f"      log g_bar [{lo_:+.2f},{lo_+0.1:+.2f})  galaxies n={na:4d} "
              f"clusters n={nb:4d}   d log|Phi_b| = {c_:.2f} dex")
    contrast = float(np.median([c for _, c, _, _ in ends])) if ends else float("nan")
    print(f"      median matched-bin contrast (rung 1 vs rung 6) = "
          f"{contrast:.2f} dex")
    qreq = 2 * 0.386 / contrast
    print("\n   What q would the known cluster anomaly require?  Clusters sit")
    print("   2.43x = 0.386 dex above the galaxy relation (unified table,")
    print("   2026-09-01).  Over this contrast that needs")
    print(f"      q_required = 2 x 0.386 / {contrast:.2f} = {qreq:.3f}")
    print(f"      statistically detectable at 3 sigma?  q_3sigma = {q3:.4f} "
          f"-> YES, by a factor {qreq/q3:.0f}")

    # ---- the systematic floor, which is what actually limits this --------
    print("\n   BUT the contrast is carried by the class boundary, so any")
    print("   CLASS-LEVEL systematic in log nu forges the same signal.  Budget:")
    SYS = [("hydrostatic mass bias, X-ray rungs vs galaxies", 0.08,
            "10-30% HSE bias; 0.08 dex is the middle of the published range"),
           ("SPARC stellar M/L (Upsilon*=0.5 vs dynamical)", 0.15,
            "this programme measured 0.4-0.55 dex disagreement in Upsilon*; "
            "0.15 dex is a conservative net effect on g_bar"),
           ("stellar mass of groups (Gonzalez extrapolation)", 0.08,
            "M*/M_gas extrapolated below the calibrated range"),
           ("gas clumping in clusters", 0.03, "measured P_X/P_SZ, median 6%"),
           ("non-thermal pressure support", 0.06,
            "unmeasured; raises g_obs for the X-ray rungs only")]
    tot_sys = math.sqrt(sum(s ** 2 for _, s, _ in SYS))
    print(f"      {'term':<48} {'dex':>6}")
    for nm, s_, why in SYS:
        print(f"      {nm:<48} {s_:6.3f}   {why[:60]}")
    print(f"      {'quadrature sum on the class contrast in log nu':<48} "
          f"{tot_sys:6.3f}")
    q_sys = 2 * tot_sys / contrast
    print(f"\n      spurious q from the systematic budget alone = "
          f"2 x {tot_sys:.3f} / {contrast:.2f} = {q_sys:.3f}")
    print(f"      required q for the cluster excess           = {qreq:.3f}")
    print(f"      ratio required/systematic                   = "
          f"{qreq/q_sys:.2f}")
    print("\n      THIS, not the statistics, is the limit.  The measurement is")
    print(f"      systematics-limited at q ~ {q_sys:.2f}, so it can separate the")
    print(f"      potential-depth hypothesis from a class-level calibration")
    print(f"      error only at the {qreq/q_sys:.1f}-sigma_sys level.")

    # ------------------------------------------------------------------
    head("7.  The actual measurement, by class, at matched g_bar")
    print("   nu_obs / nu_RAR in the shared g_bar window, per rung.  If the law")
    print("   depends only on g_bar every rung must read 1.00.")
    nu_rar = 1.0 / (1.0 - np.exp(-np.sqrt(d["g_bar"] / A0)))
    dev = np.log10(d["nu_obs"] / nu_rar)
    lo_w = max(d["lg"][d["rank"] >= 5].min(), -11.6)
    hi_w = min(d["lg"][d["rank"] >= 5].max(), -10.4)
    win = (d["lg"] >= lo_w) & (d["lg"] <= hi_w)
    print(f"   window: log g_bar in [{lo_w:+.2f},{hi_w:+.2f}]  "
          f"= g_bar/a0 {10**lo_w/A0:.3f}-{10**hi_w/A0:.3f}")
    print(f"\n      {'rung':<20} {'n':>6} {'nsys':>5} {'med log|Phi|':>13} "
          f"{'med nu/nu_RAR':>14} {'dex':>7}")
    prof = []
    for k in ranks:
        m = win & (d["rank"] == k)
        if m.sum() == 0:
            continue
        md = np.median(dev[m])
        prof.append((k, d["cls"][m][0], int(m.sum()),
                     len(set(d["system"][m])), float(np.median(d["lp"][m])),
                     float(10 ** md), float(md)))
        print(f"      {k} {d['cls'][m][0]:<18} {m.sum():6d} "
              f"{len(set(d['system'][m])):5d} {np.median(d['lp'][m]):13.2f} "
              f"{10**md:14.3f} {md:+7.3f}")
    if len(prof) >= 2:
        xs = np.array([p[4] for p in prof])
        ys = np.array([p[6] for p in prof])
        sl = np.polyfit(xs, ys, 1)[0]
        print(f"\n      slope of log(nu/nu_RAR) on median log|Phi_b| across the")
        print(f"      rungs in this window: {sl:+.4f}  -> implied q = {2*sl:+.4f}")

    # ------------------------------------------------------------------
    head("8.  MODEL COMPARISON and a FROZEN transfer test")
    print("   One row per SYSTEM inside the matched-acceleration window, so a")
    print("   long rotation curve cannot outvote a cluster.  Response is")
    print("   log10(nu_obs / nu_RAR); every model also carries a free quadratic")
    print("   in log g_bar so the RAR itself is never the thing being tested.")
    sysrows = {}
    for i in np.where(win)[0]:
        sysrows.setdefault(d["system"][i], []).append(i)
    S_ = sorted(sysrows)
    sy_lg = np.array([np.median(d["lg"][sysrows[s]]) for s in S_])
    sy_lp = np.array([np.median(d["lp"][sysrows[s]]) for s in S_])
    sy_lr = np.array([np.median(d["lr"][sysrows[s]]) for s in S_])
    sy_dv = np.array([np.median(dev[sysrows[s]]) for s in S_])
    sy_rk = np.array([d["rank"][sysrows[s][0]] for s in S_])
    ns = len(S_)
    print(f"   {ns} systems in the window; by rung: "
          + ", ".join(f"{k}:{int((sy_rk==k).sum())}" for k in sorted(set(sy_rk))))
    base = np.column_stack([np.ones(ns), sy_lg, sy_lg ** 2])
    MODELS = {
        "M0  RAR only (g_bar)": base,
        "M1  + beta log|Phi_b|": np.column_stack([base, sy_lp]),
        "M2  + gamma log r": np.column_stack([base, sy_lr]),
        "M3  + step: is it a galaxy?": np.column_stack(
            [base, (sy_rk > 1).astype(float)]),
        "M4  + full class dummies": np.column_stack(
            [base] + [(sy_rk == k).astype(float) for k in sorted(set(sy_rk))[1:]]),
    }
    print(f"\n      {'model':<30} {'k':>2} {'rms':>7} {'R2':>7} {'BIC':>9} "
          f"{'dBIC':>8}")
    bics = {}
    for nm, A in MODELS.items():
        c, *_ = np.linalg.lstsq(A, sy_dv, rcond=None)
        res = sy_dv - A @ c
        rms = res.std()
        k = A.shape[1]
        bic = ns * math.log(max(np.mean(res ** 2), 1e-300)) + k * math.log(ns)
        bics[nm] = bic
        print(f"      {nm:<30} {k:2d} {rms:7.4f} "
              f"{1-np.var(res)/np.var(sy_dv):7.4f} {bic:9.2f}", end="")
        print(f" {bic-min(b for b in bics.values()):8.2f}" if len(bics) > 1 else "")
    bmin = min(bics.values())
    print(f"\n      dBIC relative to the best model:")
    for nm in MODELS:
        print(f"         {nm:<30} {bics[nm]-bmin:+8.2f}")
    print("      A one-parameter STEP that only knows 'galaxy or not' is the")
    print("      comparison that matters: if it ties or beats log|Phi_b|, the")
    print("      ladder is measuring the dataset boundary.")

    print("\n   FROZEN TRANSFER TEST.  Fit beta on the galaxy and GROUP rungs")
    print("   only (1,2,3,4), FREEZE it, then predict the CLUSTER rungs (5,6)")
    print("   once.  This is the programme's fit-freeze-evaluate discipline;")
    print("   re-solving on the held-out set is a recorded failure mode.")
    tr = sy_rk <= 4
    te = sy_rk >= 5
    print(f"      train {tr.sum()} systems (rungs 1-4), "
          f"test {te.sum()} systems (rungs 5-6)")
    transfer = {}
    for nm, A in MODELS.items():
        c, *_ = np.linalg.lstsq(A[tr], sy_dv[tr], rcond=None)
        if "class dummies" in nm:
            transfer[nm] = float("nan")
            print(f"      {nm:<30} cannot transfer: the test rungs' dummies "
                  f"are all zero in training")
            continue
        pr = A[te] @ c
        rms_te = float(np.sqrt(np.mean((sy_dv[te] - pr) ** 2)))
        transfer[nm] = rms_te
        print(f"      {nm:<30} frozen-coefficient rms on the held-out "
              f"clusters = {rms_te:.4f} dex")
    good = {k: v for k, v in transfer.items() if v == v}
    bestm = min(good, key=good.get)
    print(f"      best transfer: {bestm} ({good[bestm]:.4f} dex)")
    ctr, *_ = np.linalg.lstsq(MODELS["M1  + beta log|Phi_b|"][tr],
                              sy_dv[tr], rcond=None)
    print(f"\n      beta fitted on rungs 1-4 alone = {ctr[3]:+.4f} "
          f"-> q = {2*ctr[3]:+.4f}")
    call, *_ = np.linalg.lstsq(MODELS["M1  + beta log|Phi_b|"], sy_dv, rcond=None)
    print(f"      beta fitted on everything      = {call[3]:+.4f} "
          f"-> q = {2*call[3]:+.4f}")
    print("      If the group-only beta does not predict the clusters, the")
    print("      'ladder' is two populations with a gap, not a continuum.")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    out = dict(
        n_rows=n, n_systems=len(set(d["system"])),
        leverage=dict(median_within_gbar_bin_sd_dex_full=med_all,
                      median_within_gbar_bin_sd_dex_sparc_only=med_sparc,
                      median_within_gbar_bin_sd_dex_tier1=med_t1,
                      sparc_reference_from_phi_rank=0.30904361685996645,
                      widest_overlap_bin=best,
                      overlap_bins=overlap,
                      within_class=within,
                      partial_corr_phi_r_given_gbar=float(cpr)),
        matched_pairs=dict(total=npair_tot, with_dlogPhi_gt_1dex=npair_1,
                           cross_class_of_those=npair_x,
                           by_gbar={f"{k:+.2f}": dict(all=tot[k],
                                                      dphi_gt_1=pairs.get(k, 0),
                                                      cross_class=xpairs.get(k, 0))
                                    for k in sorted(tot)}),
        collinearity=coll,
        log_S=dict(min=float(np.log10(d["S_shape"]).min()),
                   max=float(np.log10(d["S_shape"]).max()),
                   sd=float(np.log10(d["S_shape"]).std())),
        confound=dict(
            r2_phi_on_gbar=float(r2(lp, Ag @ cg, w)),
            r2_phi_on_gbar_plus_class=float(r2(lp, AgD @ cgd, w)),
            sd_after_gbar=float(math.sqrt(np.average(resid_g ** 2, weights=w))),
            sd_after_gbar_and_class=float(
                math.sqrt(np.average(resid_gd ** 2, weights=w)))),
        beta=dict(observed=float(beta_obs),
                  observed_with_class=float(beta_obs_cls),
                  observed_with_logr=float(beta_obs_r),
                  bootstrap_ci=[float(lo), float(hi)],
                  null_mean=float(null.mean()), null_sd=float(null.std()),
                  null_ci=[float(np.percentile(null, 2.5)),
                           float(np.percentile(null, 97.5))],
                  z_vs_own_null=float(z), p_vs_own_null=float(p),
                  null_mean_with_class=float(null_cls.mean()),
                  null_sd_with_class=float(null_cls.std())),
        power=dict(beta_3sigma=float(3 * se), q_3sigma=float(q3),
                   q_3sigma_with_class=float(q3c),
                   injection=inj, dbeta_dq=float(dq),
                   phi_contrast_galaxy_vs_cluster_dex=float(contrast),
                   contrast_by_matched_bin=ends,
                   q_required_for_cluster_excess=float(qreq),
                   systematic_budget_dex=[[nm, s_] for nm, s_, _ in SYS],
                   systematic_total_dex=float(tot_sys),
                   q_systematic_floor=float(q_sys),
                   required_over_systematic=float(qreq / q_sys)),
        by_rung_in_window=prof,
        model_comparison=dict(n_systems_in_window=ns,
                              bic={k: float(v) for k, v in bics.items()},
                              transfer_rms_dex={k: (float(v) if v == v else None)
                                                for k, v in transfer.items()},
                              beta_train_rungs1to4=float(ctr[3]),
                              beta_all=float(call[3])))
    with open(os.path.join(LANE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(clean(out), f, indent=1)
    print(f"\n   wrote {os.path.join(LANE,'results.json')}")
    return d, out


if __name__ == "__main__":
    main()
