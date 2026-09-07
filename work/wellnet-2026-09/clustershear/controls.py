"""controls.py -- kill the two trends that look like findings and are probably geometry.

`analyse.py` reports the raw amplitude of g_t against three observables. Two of
them beat the label-scramble null. Neither should be believed as it stands, and
this module is the reason why.

    kT        slope +0.0049, z = +5.1
    redshift  slope +0.0085, z = +11.7

THE PROBLEM. `g_t` is REDUCED SHEAR, not a mass. It carries the lensing geometry
inside it:

    g_t ~ DeltaSigma * Sigma_crit^-1 = DeltaSigma * (4 pi G / c^2) * D_l * <D_ls/D_s>

so g_t rises with lens redshift for a fixed cluster simply because the geometric
prefactor rises -- no gravity required. On top of that, eRASS1 is flux-limited,
so the high-redshift end of the sample contains only the most massive clusters.
Both effects push the same way, and kT correlates with redshift through the same
selection, so the kT trend can be the redshift trend wearing a different label.
This is the shared-denominator failure mode that has now caught this programme
eight times.

THE CONTROLS.
  C1  DIVIDE THE GEOMETRY OUT. Form DeltaSigma = g_t / [(4 pi G/c^2) D_l <D_ls/D_s>]
      per bin, using the per-bin <D_ls/D_s> that the extractor already stored, and
      re-run the redshift trend on that. If the trend was geometry, it collapses.
  C2  CONTROL kT FOR REDSHIFT. Fit amplitude ~ a*log10(kT) + b*z jointly and
      report the partial slope on kT. If the kT trend was the redshift trend in
      disguise, `a` goes to zero.
  C3  FIXED-REDSHIFT SLICE. Repeat the kT trend inside a narrow redshift band,
      where the geometric prefactor is nearly constant by construction. This is
      the assumption-free version of C2.

    python controls.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np

import analyse
import cosmo as C
import extract

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "controls.json")

SIGMA_CRIT_PREF = 4.0 * math.pi * C.G / C.CLIGHT ** 2      # m/kg


def delta_sigma_amplitude(r):
    """Inverse-variance mean of DeltaSigma over a cluster's bins, kg/m^2.

    DeltaSigma = g_t / Sigma_crit^-1,  Sigma_crit^-1 = (4piG/c^2) D_l <D_ls/D_s>.
    The per-bin <D_ls/D_s> is the `beta` the extractor stored, so no new
    assumption enters here -- only the geometry that was already implicit.
    """
    Dl = float(C.d_ang(r["z"]))
    num = den = 0.0
    for b in r["profile"]:
        if b["gt"] is None or not (b["err"] or 0) > 0:
            continue
        beta = b.get("beta")
        if not beta or beta <= 0:
            continue
        inv_sc = SIGMA_CRIT_PREF * Dl * beta               # m^2/kg
        if inv_sc <= 0:
            continue
        ds = b["gt"] / inv_sc                              # kg/m^2
        eds = b["err"] / inv_sc
        iv = 1.0 / eds ** 2
        num += ds * iv
        den += iv
    if den == 0:
        return None, None
    return num / den, math.sqrt(1.0 / den)


def joint_slope(x1, x2, y, w):
    """Weighted least squares y ~ c + a*x1 + b*x2.  Returns (a, b)."""
    X = np.column_stack([np.ones(len(y)), np.asarray(x1), np.asarray(x2)])
    W = np.asarray(w)
    A = X.T @ (W[:, None] * X)
    v = X.T @ (W * np.asarray(y))
    coef = np.linalg.solve(A, v)
    return float(coef[1]), float(coef[2])


def joint_null(x1, x2, y, w, n=analyse.N_SCRAMBLE, seed=analyse.RNG_SEED):
    """Permute y against BOTH regressors, preserving their mutual correlation."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    w = np.asarray(w)
    out = np.empty(n)
    idx = np.arange(len(y))
    for k in range(n):
        p = rng.permutation(idx)
        out[k] = joint_slope(x1, x2, y[p], w[p])[0]
    return out


def collect(recs, amp_fn, key):
    xs, ys, ws, zs = [], [], [], []
    for r in recs:
        v = r.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        a, ea = amp_fn(r)
        if a is None or not ea > 0:
            continue
        xs.append(math.log10(v))
        ys.append(a)
        ws.append(1.0 / ea ** 2)
        zs.append(r["z"])
    return xs, ys, ws, zs


def main():
    recs = analyse.load_profiles(analyse.PROF)
    print("clusters with profiles: %d\n" % len(recs))
    out = dict(lane="clustershear", half="OPEN", n_clusters=len(recs), controls={})

    # ---- C1: does the redshift trend survive removing the lensing geometry?
    xs, ys, ws, _ = collect(recs, analyse.amplitude, "z")
    raw = analyse.weighted_slope(xs, ys, ws)
    raw_null = analyse.scramble_null(xs, ys, ws)
    raw_z = (raw - raw_null.mean()) / raw_null.std()

    xs2, ys2, ws2, _ = collect(recs, delta_sigma_amplitude, "z")
    ds = analyse.weighted_slope(xs2, ys2, ws2)
    ds_null = analyse.scramble_null(xs2, ys2, ws2)
    ds_z = (ds - ds_null.mean()) / ds_null.std()
    ds_p = float((np.abs(ds_null - ds_null.mean()) >= abs(ds - ds_null.mean())).mean())

    out["controls"]["C1_geometry_removed"] = dict(
        question="does the redshift trend survive dividing out Sigma_crit?",
        n=len(xs2),
        raw_gt_slope=raw, raw_gt_z=raw_z,
        delta_sigma_slope=ds, delta_sigma_z=ds_z, delta_sigma_p=ds_p,
        verdict=("SURVIVES -- not geometry" if ds_p < 0.05
                 else "COLLAPSES -- the raw trend was lensing geometry"))
    print("C1  redshift trend, geometry removed")
    print("    raw g_t         : slope %+.5f  z=%+.2f" % (raw, raw_z))
    print("    DeltaSigma      : slope %+.4g  z=%+.2f  p=%.4f" % (ds, ds_z, ds_p))
    print("    -> %s\n" % out["controls"]["C1_geometry_removed"]["verdict"])

    # ---- C2: does the kT trend survive controlling for redshift?
    xk, yk, wk, zk = collect(recs, analyse.amplitude, "kt")
    a_only = analyse.weighted_slope(xk, yk, wk)
    a_joint, b_joint = joint_slope(xk, zk, yk, wk)
    nulls = joint_null(xk, zk, yk, wk)
    z_joint = (a_joint - nulls.mean()) / nulls.std()
    p_joint = float((np.abs(nulls - nulls.mean()) >= abs(a_joint - nulls.mean())).mean())
    out["controls"]["C2_kT_controlled_for_z"] = dict(
        question="does the kT trend survive a joint fit with redshift?",
        n=len(xk), kT_slope_alone=a_only,
        kT_slope_given_z=a_joint, z_slope_given_kT=b_joint,
        z_score=z_joint, p_permutation=p_joint,
        verdict=("SURVIVES -- kT carries information beyond redshift"
                 if p_joint < 0.05 else
                 "COLLAPSES -- the kT trend was the redshift trend relabelled"))
    print("C2  kT trend controlled for redshift  (n=%d)" % len(xk))
    print("    kT alone        : slope %+.5f" % a_only)
    print("    kT | z          : slope %+.5f  z=%+.2f  p=%.4f" % (a_joint, z_joint, p_joint))
    print("    z  | kT         : slope %+.5f" % b_joint)
    print("    -> %s\n" % out["controls"]["C2_kT_controlled_for_z"]["verdict"])

    # ---- C3: kT inside a narrow redshift slice
    zs = sorted(zk)
    lo, hi = zs[len(zs) // 4], zs[3 * len(zs) // 4]
    sub = [(x, y, w) for x, y, w, z in zip(xk, yk, wk, zk) if lo <= z <= hi]
    if len(sub) >= 30:
        sx, sy, sw = map(list, zip(*sub))
        s = analyse.weighted_slope(sx, sy, sw)
        n3 = analyse.scramble_null(sx, sy, sw)
        z3 = (s - n3.mean()) / n3.std()
        p3 = float((np.abs(n3 - n3.mean()) >= abs(s - n3.mean())).mean())
        v3 = ("SURVIVES at fixed redshift" if p3 < 0.05
              else "COLLAPSES at fixed redshift")
    else:
        s = z3 = p3 = None
        v3 = "TOO FEW in the slice"
    out["controls"]["C3_fixed_redshift_slice"] = dict(
        question="does the kT trend survive inside a narrow redshift band?",
        z_range=[lo, hi], n=len(sub), slope=s, z_score=z3, p_permutation=p3,
        verdict=v3)
    print("C3  kT inside z in [%.3f, %.3f]  (n=%d)" % (lo, hi, len(sub)))
    if s is not None:
        print("    slope %+.5f  z=%+.2f  p=%.4f" % (s, z3, p3))
    print("    -> %s" % v3)

    # ---- C4: is the surviving redshift trend just flux-limited SELECTION?
    # C1 removed the GEOMETRY, not the selection function. eRASS1 is
    # flux-limited, so its high-redshift end holds only the most massive
    # clusters, and mass causes lensing. kT is the mass proxy in hand, so if
    # the redshift slope on DeltaSigma collapses once kT is controlled for,
    # the trend is the survey's selection and not a property of gravity.
    xz, yz, wz, zz = [], [], [], []
    for r in recs:
        try:
            kt = float(r.get("kt"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(kt) or kt <= 0:
            continue
        a, ea = delta_sigma_amplitude(r)
        if a is None or not ea > 0:
            continue
        xz.append(r["z"])
        zz.append(math.log10(kt))
        yz.append(a)
        wz.append(1.0 / ea ** 2)
    z_alone = analyse.weighted_slope(xz, yz, wz)
    z_given_kt, kt_given_z = joint_slope(xz, zz, yz, wz)
    n4 = joint_null(xz, zz, yz, wz)
    z4 = (z_given_kt - n4.mean()) / n4.std()
    p4 = float((np.abs(n4 - n4.mean()) >= abs(z_given_kt - n4.mean())).mean())
    out["controls"]["C4_redshift_controlled_for_mass_proxy"] = dict(
        question="is the DeltaSigma redshift trend just flux-limited selection?",
        n=len(xz), z_slope_alone=z_alone, z_slope_given_kT=z_given_kt,
        kT_slope_given_z=kt_given_z, z_score=z4, p_permutation=p4,
        verdict=("SURVIVES -- not explained by the kT mass proxy alone"
                 if p4 < 0.05 else
                 "COLLAPSES -- consistent with flux-limited selection"))
    print("\nC4  DeltaSigma redshift trend controlled for kT  (n=%d)" % len(xz))
    print("    z alone         : slope %+.4g" % z_alone)
    print("    z | kT          : slope %+.4g  z=%+.2f  p=%.4f"
          % (z_given_kt, z4, p4))
    print("    -> %s" % out["controls"]["C4_redshift_controlled_for_mass_proxy"]["verdict"])

    out["interpretation"] = (
        "NONE OF THIS IS A GRAVITY RESULT, and none of it was meant to be. "
        "Every surviving trend has a conventional explanation that this lane "
        "cannot exclude, and naming them is more useful than a generic caveat. "
        "(a) kT surviving at fixed redshift (C2, C3) is EXPECTED: kT tracks "
        "mass and mass lenses, so it is the pipeline passing a sanity check. "
        "(b) The X-ray-counts null is EXPECTED: counts depend on exposure and "
        "distance as much as on mass. (c) DeltaSigma rising with redshift at "
        "fixed kT (C4) is ALSO expected, for two reasons that have nothing to "
        "do with gravity -- DeltaSigma is a surface density, and self-similar "
        "clusters are more compact at higher redshift (R500 shrinks as "
        "E(z)^(-2/3)), so DeltaSigma rises with z at fixed mass; and the "
        "background geometry here rests on dnf_z photometric redshifts, whose "
        "bias grows with lens redshift in the same direction. Separating any "
        "of that from new physics needs a baryon model and a mass calibration "
        "that does not come from weak lensing -- which eRASS1's M500 does, so "
        "it is inadmissible for the job. "
        "What this lane DOES establish is a working, null-validated 301-cluster "
        "lensing dataset: cross component consistent with zero, random "
        "pointings consistent with zero, a 31.8-sigma stacked profile, and a "
        "monotonic decline from 0.0136 at 0.34 Mpc to 0.0013 at 4.4 Mpc. "
        "Testing a gravity law on it is a separate, pre-registered step, and "
        "confirming one belongs on the sealed half.")

    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("\nwrote controls.json")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    sys.exit(main())
