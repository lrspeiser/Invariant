"""Three gates on the two-dimensional measurement, run on the REAL data.

1. ESTIMATOR AGREEMENT.  The monopole this lane measures with a per-bin
   weighted least squares in (1, cos 2phi, sin 2phi) must agree with the
   tangential profile the eFEDS lane already produced with a plain weighted
   mean (`../efeds-hsc/decade_efeds_shear_profiles.tsv`).  They use the same
   catalogue and the same cuts but different estimators and different binning,
   so agreement is a genuine check on the harmonic decomposition and on the
   metacalibration response being applied the same way.

2. MONOTONIC M_dyn.  The predicted monopole comes from an Abel projection of
   rho_dyn = (1/4 pi r^2) dM_dyn/dr.  A non-monotonic M_dyn or a clipped outer
   slope is a known way for this chain to produce a silently wrong Sigma, so
   dM/dr is required to be non-negative inside the truncation radius for every
   system used.

3. END-TO-END INJECTION, which is the real-data power curve.  A synthetic
   phase-misaligned quadrupole a2s -> a2s + A sin2Delta g_pred is added to the
   MEASURED harmonics and the whole fit is rerun.  Two things are read off:
   whether alpha is recovered without bias (dS/dtheta != 0, and equal to 1),
   and the amplitude at which the axis-randomisation test would have rejected
   at 95%.  That last number is the honest answer to "could this test have seen
   it", because it is measured on the actual noise of the actual sample rather
   than on a simulation of it.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
for p in (os.path.join(LANE, "efeds-hsc"), os.path.join(LANE, "lead01")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline as P                                            # noqa: E402
import efeds_hsc as E                                           # noqa: E402
import shear2d as S                                             # noqa: E402


def gate_estimator(rows, tsv):
    """Compare a0 against the eFEDS lane's independently binned g_t."""
    ref = {}
    with open(tsv, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith("#") or ln.startswith("id\t"):
                continue
            p = ln.rstrip("\n").split("\t")
            try:
                gid, R, gt, err = p[0], float(p[3]), float(p[6]), float(p[8])
            except ValueError:
                continue
            if not np.isfinite(gt) or not np.isfinite(err) or err <= 0:
                continue
            ref.setdefault(gid, []).append((R, gt, err))
    pairs = []
    for r in rows:
        if r["id"] not in ref:
            continue
        a = np.array([(b["R"], b["a0"], b["e0"]) for b in r["bins"]
                      if b is not None and np.isfinite(b["a0"])
                      and b["e0"] > 0])
        c = np.array(ref[r["id"]])
        if a.size == 0 or c.size == 0:
            continue
        # inverse-variance mean of each, over the common radial range
        lo = max(a[:, 0].min(), c[:, 0].min())
        hi = min(a[:, 0].max(), c[:, 0].max())
        ma = (a[:, 0] >= lo) & (a[:, 0] <= hi)
        mc = (c[:, 0] >= lo) & (c[:, 0] <= hi)
        if ma.sum() < 2 or mc.sum() < 2:
            continue

        def ivm(x):
            w = 1.0 / x[:, 2] ** 2
            return float(np.sum(w * x[:, 1]) / np.sum(w)), \
                float(math.sqrt(1.0 / np.sum(w)))
        va, ea = ivm(a[ma])
        vc, ec = ivm(c[mc])
        pairs.append((va, ea, vc, ec))
    pairs = np.array(pairs)
    d = pairs[:, 0] - pairs[:, 2]
    e = np.hypot(pairs[:, 1], pairs[:, 3])
    chi = float(np.sum((d / e) ** 2) / max(len(d), 1))
    w = 1.0 / e ** 2
    off = float(np.sum(w * d) / np.sum(w))
    offe = float(math.sqrt(1.0 / np.sum(w)))
    return dict(n=len(d), mean_difference=off, err=offe,
                chi2_per_system=chi,
                mine_mean=float(np.mean(pairs[:, 0])),
                theirs_mean=float(np.mean(pairs[:, 2])))


def gate_monotone(ids, by_id):
    bad = []
    for i in ids:
        sysm = P.System(by_id[i], f_star=S.F_STAR)
        g = sysm.g_pred(law="rar")
        sysm.sigma_profile(g, np.array([1.0 * P.MPC]), r_trunc_mpc=10.0)
        dM = sysm.last_dM
        if np.any(dM < -1e-6 * np.max(np.abs(dM))):
            bad.append(i)
    return dict(n=len(ids), n_non_monotone=len(bad), offenders=bad[:10])


def injection_curve(rows, amps=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8,
                                1.0, 1.2, 1.6, 2.4),
                    n_rot=2000, seed=7, use_pred=True, fixed_effects=False):
    """Add a known misaligned quadrupole to the real data and re-measure."""
    rng = np.random.default_rng(seed)
    base = fit_null_scale(rows, rng, n_rot, use_pred=use_pred,
                          fixed_effects=fixed_effects)
    out = []
    for A in amps:
        inj = []
        for r in rows:
            rr = dict(r)
            bins = []
            for b in r["bins"]:
                if b is None:
                    bins.append(None)
                    continue
                bb = dict(b)
                bb["a2s"] = b["a2s"] + A * r["sin2delta"] * b["g_pred"]
                bins.append(bb)
            rr["bins"] = bins
            inj.append(rr)
        f = S.fit_alpha(inj, key="a2s", use_pred=use_pred,
                        fixed_effects=fixed_effects)
        det = float(np.mean(np.abs(base["null_pred"]) >= abs(f["alpha"])))
        out.append(dict(injected=A, alpha=f["alpha"], e_alpha=f["e_alpha"],
                        p_two_sided=det, detected=bool(det < 0.05)))
    sig = base["sigma_pred"]
    # the two standard numbers.  The 95% EXCLUSION limit is where the observed
    # alpha would clear the axis-randomisation threshold; the 95% POWER
    # amplitude is where a true signal would clear it with probability 0.95,
    # which is 1.96 + 1.645 = 3.60 sigma, not 1.96.
    q95 = float(np.percentile(np.abs(base["null_pred"]), 95))
    return dict(null_sigma_pred=sig, null_abs_q95=q95, curve=out,
                exclusion_2sided_95=q95,
                amplitude_for_95pc_power=float(q95 + 1.645 * sig))


def fit_null_scale(rows, rng, n_rot, use_pred=True, fixed_effects=False):
    na = []
    for _ in range(n_rot):
        g = S.fit_alpha(rows, key="a2s", use_pred=use_pred,
                        fixed_effects=fixed_effects,
                        sin2d=S.randomised_axes(rows, rng))
        if g:
            na.append(g["alpha"])
    na = np.array(na)
    return dict(null_pred=na, sigma_pred=float(np.std(na)))


def rebuild_rows(sel, tag, by_id, edges):
    rows = []
    for s in sel[tag]:
        npz = os.path.join(S.CACHE, f"shear_{s['id']}.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        meas = {k: d[k] for k in d.files}
        gp = S.predicted_monopole(
            by_id[s["id"]],
            [0.5 * (edges[i] + edges[i + 1]) for i in range(S.NBIN)],
            float(meas["beta"]), float(meas["beta2"]))
        phi_rel = meas["phi"] - math.radians(s["pa_mem_deg"])
        hb = S.harmonics(meas["et"], meas["w"], meas["R"], phi_rel, edges,
                         list(meas["resp"]))
        hx = S.harmonics(meas["ex"], meas["w"], meas["R"], phi_rel, edges,
                         list(meas["resp"]))
        for i, b in enumerate(hb):
            if b is not None:
                b["g_pred"] = float(gp[i])
                b["x2s"] = hx[i]["a2s"] if hx[i] else np.nan
                b["x2c"] = hx[i]["a2c"] if hx[i] else np.nan
                b["x0"] = hx[i]["a0"] if hx[i] else np.nan
        rows.append(dict(id=s["id"], z=s["z"],
                         misalign_deg=s["misalign_deg"],
                         sin2delta=s["sin2delta"],
                         pa_mem_deg=s["pa_mem_deg"], bins=hb))
    return rows


def main():
    print("=" * 78)
    print("CROSSCHECK -- estimator agreement, monotone M_dyn, and the")
    print("              END-TO-END injection power curve on the real data")
    print("=" * 78)
    sel = json.load(open(os.path.join(HERE, "selection.json"),
                         encoding="utf-8"))
    recs, _ = E.load_efeds()
    by_id = {r["id"]: r for r in recs}
    edges = np.geomspace(S.RMIN_H / P.H_LITTLE, S.RMAX_H / P.H_LITTLE,
                         S.NBIN + 1)
    res = {}
    for tag in ("dev", "ctrl"):
        rows = rebuild_rows(sel, tag, by_id, edges)
        print(f"\n   {tag.upper()}: {len(rows)} systems rebuilt from cache")
        if not rows:
            continue
        g1 = gate_estimator(rows, os.path.join(
            LANE, "efeds-hsc", "decade_efeds_shear_profiles.tsv"))
        print(f"      GATE 1 estimator agreement vs the eFEDS lane's own "
              f"profiles, {g1['n']} systems")
        print(f"         this lane's mean g_t {g1['mine_mean']:+.5f}, "
              f"theirs {g1['theirs_mean']:+.5f}")
        print(f"         weighted difference {g1['mean_difference']:+.5f} "
              f"+- {g1['err']:.5f}   chi2 per system "
              f"{g1['chi2_per_system']:.2f}")
        g2 = gate_monotone([r["id"] for r in rows], by_id)
        print(f"      GATE 2 M_dyn monotone inside the truncation radius: "
              f"{g2['n'] - g2['n_non_monotone']}/{g2['n']} pass")
        res[tag] = dict(estimator_gate=g1, monotone_gate=g2)
        if tag == "dev":
            res["injection"] = {}
            for up, fe, lab, key in (
                    (True, False, "predicted monopole (PRIMARY)", "pred"),
                    (False, False, "measured monopole", "meas"),
                    (True, True, "predicted monopole, per-cluster fixed "
                                 "effects", "fe")):
                inj = injection_curve(rows, use_pred=up, fixed_effects=fe)
                print(f"      GATE 3 end-to-end injection, {lab}: "
                      f"axis-randomisation sigma "
                      f"{inj['null_sigma_pred']:.4f}")
                print("         injected   recovered alpha        p        "
                      "detected")
                for c in inj["curve"]:
                    print(f"         {c['injected']:8.3f}   "
                          f"{c['alpha']:+8.4f} +- {c['e_alpha']:.4f}   "
                          f"{c['p_two_sided']:6.4f}   "
                          f"{'YES' if c['detected'] else 'no'}")
                sl = np.polyfit([c["injected"] for c in inj["curve"]],
                                [c["alpha"] for c in inj["curve"]], 1)[0]
                print(f"         d(recovered)/d(injected) = {sl:.4f}   "
                      f"(1.0 is unbiased; 0 would mean the statistic does "
                      f"not respond to what it claims to measure)")
                inj["response_slope"] = float(sl)
                det = [c["injected"] for c in inj["curve"] if c["detected"]]
                inj["min_detectable_alpha"] = float(min(det)) if det else None
                print(f"         95% exclusion limit  |alpha| < "
                      f"{inj['exclusion_2sided_95']:.3f}")
                print(f"         amplitude this sample would detect with 95% "
                      f"probability: {inj['amplitude_for_95pc_power']:.3f}")
                sl = inj["response_slope"]
                if abs(sl) > 1e-6:
                    inj["effective_sigma"] = (inj["null_sigma_pred"]
                                              / abs(sl))
                    inj["effective_exclusion_95"] = (
                        inj["exclusion_2sided_95"] / abs(sl))
                    print(f"         DILUTION-CORRECTED sensitivity: "
                          f"sigma/slope = {inj['effective_sigma']:.3f}, "
                          f"95% exclusion on the TRUE amplitude "
                          f"{inj['effective_exclusion_95']:.3f}")
                res["injection"][key] = inj
    with open(os.path.join(HERE, "crosscheck.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\n   written: crosscheck.json")


if __name__ == "__main__":
    main()
