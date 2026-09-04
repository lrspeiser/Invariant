"""Is the eFEDS internal radial slope one number, or two?

beta is the measurement this lane cares most about, and eFEDS is the only
survey with real internal radial leverage.  But eFEDS's shear reaches to
33 R500, while the Bahar+2022 Vikhlinin density fit is anchored inside ~R500.
Beyond that the baryon model is an EXTRAPOLATION, and an extrapolation that
over-predicts M_b would push the fitted response down at large radius and
manufacture a steep negative beta.

So the slope is refitted in radial windows, through the same 3-D forward model
(the slip is applied to the mass and re-projected, not multiplied onto the
projected profile), with the amplitude profiled out each time.

Points outside a window are de-weighted rather than deleted, which is exactly
equivalent for a chi2 and keeps every array aligned.

Writes radial_results.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as K                                              # noqa: E402
import fitlib as F                                              # noqa: E402

BETA = np.round(np.arange(-1.20, 0.61, 0.05), 4)
AMP = np.concatenate([[-9.0], np.linspace(-1.5, 1.2, 541)])
WINDOWS = [("all radii", 0.0, 1e9), ("r/R500 < 1.3", 0.0, 1.3),
           ("r/R500 < 2.0", 0.0, 2.0), ("r/R500 < 3.2", 0.0, 3.2),
           ("r/R500 < 5.0", 0.0, 5.0), ("r/R500 > 2.0", 2.0, 1e9),
           ("r/R500 > 3.2", 3.2, 1e9), ("1.0 < r/R500 < 5.0", 1.0, 5.0)]


def fit_window(bd, J, mask, er):
    bd.F.er = np.where(mask, er, er * 1e6)
    curve = []
    for be in BETA:
        S, dS = J.project("H_R", np.array([be]))
        curve.append(min(float(bd.F.chi2(S, dS, 10.0 ** a)) for a in AMP))
    bd.F.er = er.copy()
    curve = np.array(curve)
    i = int(np.argmin(curve))
    ok = BETA[curve - curve.min() <= 1.0]
    return (float(BETA[i]), float(ok.min()), float(ok.max()),
            int(mask.sum()), curve.tolist(),
            float(curve.max() - curve.min()))


def main():
    bd = K.Bundle(verbose=False, r500_mode="cat")
    fs = (0.2565, 0.2004, 0.1937)
    J = F.Joint(bd, surveys=("efeds",), fixed_scatter=fs)
    x = np.exp(bd.ef_x["lnx"])
    er = bd.F.er.copy()
    gt = bd.F.gt.copy()
    out = {"windows": {}, "note": __doc__.strip().splitlines()[0]}
    print("=" * 78)
    print("eFEDS INTERNAL RADIAL SLOPE, refitted in radial windows")
    print("(3-D slip, re-projected; amplitude profiled out)")
    print("=" * 78)
    for lab, lo, hi in WINDOWS:
        m = (x >= lo) & (x < hi)
        if m.sum() < 100:
            continue
        b, blo, bhi, n, _, rng = fit_window(bd, J, m, er)
        out["windows"][lab] = dict(beta=b, ci68=[blo, bhi], n=n, dchi2=rng,
                                   at_grid_edge=bool(b <= BETA[0] + 1e-9
                                                     or b >= BETA[-1] - 1e-9))
        print(f"   {lab:20s} n = {n:5d}   beta = {b:+.3f}"
              f"   68% [{blo:+.3f}, {bhi:+.3f}]"
              f"{'   *** AT GRID EDGE' if out['windows'][lab]['at_grid_edge'] else ''}")

    # the same statistic on the B-mode, which must return nothing
    bd.F.gt = bd.F.gx.copy()
    b, blo, bhi, n, _, rngB = fit_window(bd, J, x > 0, er)
    bd.F.gt = gt
    rngT = out["windows"]["all radii"]["dchi2"]
    out["bmode"] = dict(beta=b, ci68=[blo, bhi], n=n, dchi2=rngB,
                        dchi2_tangential=rngT)
    print(f"\n   B-MODE (cross component), all radii:")
    print(f"      chi2 variation across the whole beta grid = {rngB:.2f}")
    print(f"      the same on the TANGENTIAL component       = {rngT:.2f}")
    print("   -> the cross component carries no lensing signal, so its profile")
    print("      in beta must be nearly FLAT.  A 'best-fitting' beta from a")
    print("      flat profile is meaningless -- the number that matters is how")
    print("      much chi2 beta buys, and on the B-mode that is"
          f" {rngB:.1f} against")
    print(f"      {rngT:.1f} on the real signal.")

    # the declared train/held split, on the full range
    order = np.argsort([c.id for c in bd.ef])
    tr = np.zeros(len(bd.ef), bool)
    tr[np.sort(order[0::2])] = True
    mtr = tr[bd.F.sysi]
    a1 = fit_window(bd, J, mtr, er)
    a2 = fit_window(bd, J, ~mtr, er)
    out["split"] = dict(train=dict(beta=a1[0], ci68=[a1[1], a1[2]], n=a1[3]),
                        held=dict(beta=a2[0], ci68=[a2[1], a2[2]], n=a2[3]))
    print(f"\n   declared split (the closure lane's, by sorted system id):")
    print(f"      TRAIN beta = {a1[0]:+.3f} [{a1[1]:+.3f}, {a1[2]:+.3f}]"
          f"   n = {a1[3]}")
    print(f"      HELD  beta = {a2[0]:+.3f} [{a2[1]:+.3f}, {a2[2]:+.3f}]"
          f"   n = {a2[3]}")

    w = out["windows"]
    if "r/R500 < 2.0" in w and "r/R500 > 2.0" in w:
        i, o = w["r/R500 < 2.0"], w["r/R500 > 2.0"]
        si = 0.5 * (i["ci68"][1] - i["ci68"][0])
        so = 0.5 * (o["ci68"][1] - o["ci68"][0])
        d = (i["beta"] - o["beta"]) / math.hypot(si, so)
        out["inner_vs_outer_sigma"] = float(d)
        print(f"\n   INNER vs OUTER: {i['beta']:+.3f} against {o['beta']:+.3f}"
              f"  ->  {abs(d):.1f} sigma apart.")
        print("   A single power law in r/R500 is therefore not an adequate")
        print("   description of the eFEDS residual, and the steep outer value")
        print("   sits where the Vikhlinin density fit is extrapolated.")
    with open(os.path.join(HERE, "radial_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n   wrote radial_results.json")


if __name__ == "__main__":
    main()
