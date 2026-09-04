"""Apply the Stage 4 certificate to the transition lane's headline claim.

The claim: a radius law `beta` measured on eFEDS shear alone, frozen, extrapolated
7x inward in radius and 25x upward in mass, predicts the strong-lens cores at
+0.90 sigma -- and therefore the cluster excess is organised by RADIUS.

This is the first result produced after the certificate existed, so it is the
first to be gated rather than reported.  Every input below is the lane's own
number, read from its JSON.

    python certify_transition.py
"""
import io
import json
import os

import numpy as np

import certificate as C

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.abspath(os.path.join(HERE, "..", "transition"))


def load(name):
    return json.load(io.open(os.path.join(LANE, name), encoding="utf-8"))


def main():
    beta = load("beta_radial_range.json")
    pts = load("points.json")

    # ---- C5: where does eFEDS actually have points, and where is beta read?
    occ = pts["radius_occupancy"]
    efeds_bins = [b for b in occ if b["by_survey"].get("efeds", 0) > 0]
    sl_bins = [b for b in occ if b["by_survey"].get("sl", 0) > 0]
    efeds_lo = min(b["lnx_lo"] for b in efeds_bins)
    efeds_hi = max(b["lnx_hi"] for b in efeds_bins)
    read_lo = min(b["lnx_lo"] for b in sl_bins)      # the claim reaches the cores
    read_hi = efeds_hi

    print("=" * 74)
    print("SUPPORT: where the surveys actually live, in ln(r/R500)")
    print("=" * 74)
    for b in occ:
        s = b["by_survey"]
        print(f"  [{b['lnx_lo']:+.2f}, {b['lnx_hi']:+.2f})  n={b['n']:>4}   "
              f"efeds {s.get('efeds',0):>4}  locuss {s.get('locuss',0):>3}  "
              f"sl {s.get('sl',0):>3}")
    print()
    print(f"  eFEDS support        ln x in [{efeds_lo:+.2f}, {efeds_hi:+.2f}]"
          f"   -> r/R500 {np.exp(efeds_lo):.3f} to {np.exp(efeds_hi):.2f}")
    print(f"  the claim is read to ln x = {read_lo:+.2f}"
          f"   -> r/R500 {np.exp(read_lo):.3f}")
    print(f"  eFEDS points below the eFEDS floor: "
          f"{sum(b['by_survey'].get('efeds',0) for b in occ if b['lnx_hi'] <= efeds_lo)}")

    # ---- C1/C2: is beta stable, or is it a property of the window chosen?
    print()
    print("=" * 74)
    print("STABILITY: beta by radial window (the lane's own table)")
    print("=" * 74)
    for k, v in beta.items():
        print(f"  {k:<16} beta = {v[0]:+.3f}  [{v[1]:+.3f}, {v[2]:+.3f}]  n={v[3]}")
    inner = beta["r/R500 < 2.0"]
    outer = beta["r/R500 > 2.0"]
    disjoint = inner[1] > outer[2] or outer[1] > inner[2]
    print()
    print(f"  inner and outer 68% intervals "
          f"{'DO NOT OVERLAP' if disjoint else 'overlap'}")

    res = {}
    res["C5_support"] = C.c5_support(read_range=(read_lo, read_hi),
                                     measured_range=(efeds_lo, efeds_hi))
    # C1: beta responds -- but the thing it responds to is the window, not only
    # the physics.  Report it as measured.
    windows = [inner[0], beta["r/R500 < 3.2"][0], beta["all radii"][0], outer[0]]
    res["C1_responsive"] = C.c1_responsive(lambda i: windows[int(i)],
                                           np.arange(len(windows)))
    # C2: does the choice of radial window move beta more than the claimed
    # effect itself?  The claimed effect is the difference from beta = 0.
    res["C2_not_a_restatement"] = C.c2_not_a_restatement(
        stat_fn=lambda e: beta["all radii"][0] * e, effects=np.linspace(0, 1, 5),
        nuisance_fn=lambda i: windows[int(i)], nuis_range=np.arange(len(windows)))
    # C4: statistics are not limiting -- beta to +-0.05 against a beta of -0.35
    res["C4_powered"] = C.c4_powered(responsiveness=1.0, predicted_effect=0.350,
                                     noise_sd=0.05)

    print()
    ok = C.certify("transition: 'the excess is organised by RADIUS'", res)

    doc = dict(claim="cluster excess organised by radius; eFEDS beta extrapolates "
                     "7x inward to the strong-lens cores",
               efeds_support_lnx=[efeds_lo, efeds_hi],
               read_to_lnx=read_lo,
               efeds_points_below_floor=0,
               beta_windows=beta, checks=res, certificate_issued=ok)
    p = os.path.join(HERE, "certify_transition.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(doc, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
