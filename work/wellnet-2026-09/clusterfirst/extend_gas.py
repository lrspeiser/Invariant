"""extend_gas.py -- gas density out to the lensing radii, from the Chandra events.

Run BT could only compare one to three lensing bins per cluster, because ACCEPT
stops between 0.34 and 1.18 Mpc while the shear runs to 4.4. The Chandra field
already on disk reaches 2.3x to 8.2x further -- 0.94 to 4.6 Mpc. This turns the
events into a density profile over that whole range.

METHOD, and what each step assumes.

  1. Surface brightness SB(theta) in the 0.5-2 keV band, from the events, in
     logarithmic annuli. This band is where the ICM dominates and the effective
     area is best known.

  2. Background from the outermost annuli of the field, fitted as a constant and
     subtracted. Cluster emission is still present there at some level, so this
     OVERSUBTRACTS slightly and biases the outer density low.

  3. Vignetting. ACIS loses effective area off-axis, energy-dependently, and
     without CIAO exposure maps it must be modelled. An analytic approximation
     is applied, V(theta) = (1 + (theta/theta_v)^2)^-alpha with theta_v = 11
     arcmin and alpha = 0.8, which reproduces the ACIS-I 1 keV falloff to about
     10%. THIS IS THE DOMINANT SYSTEMATIC in the outer bins and it is stated
     rather than hidden: it biases the outer SB, hence beta, hence the outer
     density.

  4. A beta model is fitted to the corrected profile. For that model the surface
     brightness has a closed form,
         SB(R) = S0 (1 + (R/Rc)^2)^(0.5 - 3 beta) + bkg
     and the density follows analytically,
         n_e(r) = n0 (1 + (r/Rc)^2)^(-3 beta / 2)
     so no numerical deprojection is needed and no new geometric assumption
     enters beyond the sphericity ACCEPT already assumed.

  5. n0 is NOT fitted from the X-ray normalisation, which would need the
     effective area, the emissivity and the exposure. It is anchored to ACCEPT
     in the radial range where the two overlap. So the SHAPE is measured from
     the Chandra events and the ABSOLUTE SCALE is inherited from a calibrated
     published product.

The result is a density profile whose inner region reproduces ACCEPT by
construction and whose outer region is a measured extrapolation rather than a
fitted power law -- which is the difference between this and the version Run BT
refused.

    python extend_gas.py
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clusterxray"))
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))

import maps as XM                                              # noqa: E402
import cosmo as C                                              # noqa: E402

RAW = os.path.join(WELLNET, "clusterxray", "raw")
OUT = os.path.join(HERE, "gas_extended.json")

SOFT = (500.0, 2000.0)          # the band used for the profile
THETA_V, ALPHA = 11.0, 0.8      # ACIS vignetting approximation, arcmin
NANN = 18
TH_MIN, TH_MAX = 0.15, 12.0     # arcmin


def vignette(theta):
    return (1.0 + (theta / THETA_V) ** 2) ** (-ALPHA)


def sb_profile(files, ra0, de0):
    """Background-subtracted, vignetting-corrected SB(theta), counts/arcmin^2."""
    cosd = math.cos(math.radians(de0))
    th = []
    for f in files:
        got = XM.events_radec(f)
        if got is None:
            continue
        ra, dec, e = got
        m = (e >= SOFT[0]) & (e < SOFT[1])
        dx = ((ra[m] - ra0 + 180.0) % 360.0 - 180.0) * cosd
        dy = dec[m] - de0
        th.append(np.hypot(dx, dy) * 60.0)                     # arcmin
    if not th:
        return None
    th = np.concatenate(th)
    edges = np.geomspace(TH_MIN, TH_MAX, NANN + 1)
    n, _ = np.histogram(th, bins=edges)
    area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    ctr = np.sqrt(edges[1:] * edges[:-1])
    sb = n / area
    err = np.sqrt(np.maximum(n, 1.0)) / area
    keep = n > 0
    return ctr[keep], sb[keep], err[keep], n[keep]


def fit_beta(ctr, sb, err):
    """SB(R) = S0 (1+(R/Rc)^2)^(0.5-3beta) + bkg, on vignetting-corrected data."""
    sbc = sb / vignette(ctr)
    errc = err / vignette(ctr)

    def resid(p):
        S0, Rc, beta, bkg = np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3])
        model = S0 * (1.0 + (ctr / Rc) ** 2) ** (0.5 - 3.0 * beta) + bkg
        return (model - sbc) / errc

    best, bestcost = None, np.inf
    for Rc0 in (0.3, 0.8, 2.0):
        for b0 in (0.55, 0.7, 0.9):
            p0 = [math.log(max(sbc[0], 1e-6)), math.log(Rc0), b0,
                  math.log(max(np.median(sbc[-3:]), 1e-9))]
            try:
                r = least_squares(resid, p0, method="lm", max_nfev=6000)
            except Exception:                                  # noqa: BLE001
                continue
            if r.cost < bestcost:
                best, bestcost = r, r.cost
    if best is None:
        return None
    S0, Rc, beta, bkg = (math.exp(best.x[0]), math.exp(best.x[1]),
                         float(best.x[2]), math.exp(best.x[3]))
    return dict(S0=S0, Rc_arcmin=Rc, beta=beta, bkg=bkg,
                cost=float(best.cost), nbin=int(len(ctr)))


def main():
    match = json.load(io.open(os.path.join(HERE, "accept_overlap.json"),
                              encoding="utf-8"))
    acc = json.load(io.open(os.path.join(HERE, "accept_profiles.json"),
                            encoding="utf-8"))
    key_of = {}
    centres = json.load(io.open(os.path.join(WELLNET, "clusterxray",
                                             "overlap_centres.json"), encoding="utf-8"))
    for k, v in centres.items():
        key_of[v[6]] = (k, v[0], v[1], v[2])

    out = {}
    print("  %-24s %-7s %-7s %-9s %-9s %s"
          % ("cluster", "beta", "Rc kpc", "n0 cm^-3", "reach Mpc", "anchor bins"))
    for m in sorted(match, key=lambda m: -m["shear"]):
        got = key_of.get(m["erass"])
        if not got:
            continue
        key, ra0, de0, z = got
        files = sorted(glob.glob(os.path.join(RAW, "%s_*evt2*" % key)))
        if not files:
            continue
        prof = sb_profile(files, ra0, de0)
        if prof is None:
            continue
        ctr, sb, err, cnt = prof
        fit = fit_beta(ctr, sb, err)
        if fit is None:
            continue

        DA = float(C.d_ang(z)) / C.MPC                          # Mpc
        rc_mpc = math.radians(fit["Rc_arcmin"] / 60.0) * DA
        beta = fit["beta"]

        # anchor n0 to ACCEPT over the radii both cover
        ap = np.array(acc[m["accept"]])
        r_acc, ne_acc = ap[:, 0], ap[:, 1]
        lo = max(r_acc.min(), math.radians(TH_MIN / 60.0) * DA)
        hi = min(r_acc.max(), math.radians(TH_MAX / 60.0) * DA)
        sel = (r_acc >= lo) & (r_acc <= hi)
        if sel.sum() < 3:
            continue
        shape = (1.0 + (r_acc[sel] / rc_mpc) ** 2) ** (-1.5 * beta)
        n0 = float(np.exp(np.median(np.log(ne_acc[sel] / shape))))
        reach = math.radians(TH_MAX / 60.0) * DA

        # REJECT degenerate fits rather than carry them. The first run produced
        # one cluster with Rc = 0 kpc and n0 = 1.4e5 cm^-3 -- five orders of
        # magnitude above any cluster core -- because the optimiser found a
        # corner where a vanishing core radius and a huge normalisation trade
        # off exactly. Such a fit reproduces the SB profile and is physically
        # impossible, so it must fail loudly here rather than propagate into a
        # gravity test as a very confident wrong baryon mass.
        bad = []
        if not (0.010 <= rc_mpc <= 1.0):
            bad.append("Rc = %.0f kpc outside 10-1000" % (rc_mpc * 1000))
        if not (0.35 <= beta <= 1.3):
            bad.append("beta = %.2f outside 0.35-1.3" % beta)
        if not (1e-4 <= n0 <= 1.0):
            bad.append("n0 = %.2e cm^-3 outside 1e-4 to 1" % n0)
        # the anchor must actually agree with ACCEPT, not just be scaled to it
        pred = n0 * shape
        dex = float(np.std(np.log10(ne_acc[sel] / pred)))
        if dex > 0.25:
            bad.append("anchor scatter %.2f dex > 0.25" % dex)
        if bad:
            print("  %-24s REJECTED: %s" % (m["erass"], "; ".join(bad)), flush=True)
            continue

        out[m["erass"]] = dict(
            accept=m["accept"], z=z, beta=beta, rc_mpc=rc_mpc, n0=n0,
            reach_mpc=reach, anchor_bins=int(sel.sum()),
            anchor_scatter_dex=round(dex, 3),
            accept_r_max=float(r_acc.max()),
            vignetting=dict(theta_v_arcmin=THETA_V, alpha=ALPHA,
                            note="analytic ACIS approximation, dominant outer systematic"),
            sb_fit=fit)
        print("  %-24s %-7.3f %-7.0f %-9.2e %-9.2f %-4d %.2f dex"
              % (m["erass"], beta, rc_mpc * 1000, n0, reach, sel.sum(), dex),
              flush=True)

    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("\nwrote gas_extended.json for %d clusters" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
