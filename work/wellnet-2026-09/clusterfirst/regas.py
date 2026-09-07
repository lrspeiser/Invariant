"""regas.py -- rebuild the gas profiles on real exposure maps.

Replaces the analytic ACIS vignetting stand-in in `extend_gas.py`:

    V(theta) = (1 + (theta/11')^2)^-0.8

with the exposure maps CIAO computes per observation from the aspect solution,
the instrument map and the CALDB effective area. Those carry the actual dither,
chip gaps, bad pixels, the time-dependent contamination layer and the
energy-weighted vignetting for the band in use -- none of which a one-parameter
curve can represent.

The surface brightness becomes

    SB(theta) = sum(counts in annulus) / sum(exposure in annulus)

which is the exposure-weighted estimator, rather than raw counts divided by a
guessed curve. Annuli with too little exposure -- chip gaps, off-field corners --
drop out on their own, because their exposure sum is small rather than because a
threshold was chosen.

Everything downstream is unchanged: the same beta-model form, the same anchoring
to ACCEPT, the same rejection bounds. So the difference between this and
`gas_extended.json` is attributable to the exposure correction alone, which is
the point -- Run BW put a conservative 15% systematic on that correction and it
alone moved the universality test from p = 0.009 to p = 0.086.

    python regas.py
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
import cosmo as C                                              # noqa: E402

EXP = os.path.join(HERE, "expmaps")
OUT = os.path.join(HERE, "gas_extended_expcorr.json")

NANN = 20
TH_MIN, TH_MAX = 0.15, 12.0        # arcmin
MIN_EXP_FRAC = 0.15                # annulus needs this share of the peak per-pixel exposure
SYS_FRAC = 0.03                    # flat-field/background systematic, added in quadrature


def radial(img_path, exp_path, ra0, de0):
    """Counts and exposure summed in annuli about (ra0, de0)."""
    with fits.open(img_path) as h:
        img = np.asarray(h[0].data, dtype=float)
        w = WCS(h[0].header)
    with fits.open(exp_path) as h:
        exp = np.asarray(h[0].data, dtype=float)
    if img.shape != exp.shape:
        return None

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra, dec = w.all_pix2world(xx, yy, 0)
    dx = ((ra - ra0 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(de0))
    th = np.hypot(dx, dec - de0) * 60.0                        # arcmin

    edges = np.geomspace(TH_MIN, TH_MAX, NANN + 1)
    ctr, cts, ex, npx = [], [], [], []
    for i in range(NANN):
        m = (th >= edges[i]) & (th < edges[i + 1]) & np.isfinite(exp)
        if not m.any():
            continue
        ctr.append(math.sqrt(edges[i] * edges[i + 1]))
        cts.append(float(np.nansum(img[m])))
        ex.append(float(np.nansum(exp[m])))
        npx.append(int(m.sum()))
    return np.array(ctr), np.array(cts), np.array(ex), np.array(npx)


def fit_beta(ctr, sb, err):
    def resid(p):
        S0, Rc, beta, bkg = np.exp(p[0]), np.exp(p[1]), p[2], np.exp(p[3])
        model = S0 * (1.0 + (ctr / Rc) ** 2) ** (0.5 - 3.0 * beta) + bkg
        return (model - sb) / err

    best, cost = None, np.inf
    for Rc0 in (0.3, 0.8, 2.0):
        for b0 in (0.55, 0.7, 0.9):
            p0 = [math.log(max(sb[0], 1e-12)), math.log(Rc0), b0,
                  math.log(max(np.median(sb[-3:]), 1e-15))]
            try:
                r = least_squares(resid, p0, method="lm", max_nfev=8000)
            except Exception:                                  # noqa: BLE001
                continue
            if r.cost < cost:
                best, cost = r, r.cost
    if best is None:
        return None
    return dict(S0=math.exp(best.x[0]), Rc_arcmin=math.exp(best.x[1]),
                beta=float(best.x[2]), bkg=math.exp(best.x[3]),
                cost=float(best.cost), nbin=int(len(ctr)))


def main():
    match = json.load(io.open(os.path.join(HERE, "accept_overlap.json"),
                              encoding="utf-8"))
    acc = json.load(io.open(os.path.join(HERE, "accept_profiles.json"),
                            encoding="utf-8"))
    centres = json.load(io.open(os.path.join(WELLNET, "clusterxray",
                                             "overlap_centres.json"), encoding="utf-8"))
    key_of = {v[6]: (k, v[0], v[1], v[2]) for k, v in centres.items()}
    old = json.load(io.open(os.path.join(HERE, "gas_extended.json"), encoding="utf-8"))

    out = {}
    print("  %-24s %-7s %-7s %-9s %-6s %-8s %s"
          % ("cluster", "beta", "Rc kpc", "n0", "obs", "anchor", "beta was"))
    for m in sorted(match, key=lambda m: -m["shear"]):
        got = key_of.get(m["erass"])
        if not got:
            continue
        key, ra0, de0, z = got
        imgs = sorted(glob.glob(os.path.join(EXP, "%s_*_0.5-2.0_thresh.img" % key)))
        if not imgs:
            continue

        tot_c, tot_e, tot_n, ctr = None, None, None, None
        nobs = 0
        for ip in imgs:
            ep = ip.replace("_0.5-2.0_thresh.img", "_0.5-2.0_thresh.expmap")
            if not os.path.exists(ep):
                continue
            got2 = radial(ip, ep, ra0, de0)
            if got2 is None:
                continue
            c, cc, ee, nn = got2
            if ctr is None:
                ctr, tot_c, tot_e, tot_n = c, cc.copy(), ee.copy(), nn.copy()
            elif len(c) == len(ctr):
                tot_c += cc
                tot_e += ee
                tot_n += nn
            nobs += 1
        if ctr is None or nobs == 0:
            continue

        # exposure per pixel, NOT the sum: the sum grows with annulus area, so a
        # threshold on it removes the innermost bins -- the brightest ones -- and
        # leaves only the background plateau. That is what made the first attempt
        # report 'only 0-3 annuli above 2x background' on every cluster.
        per_px = tot_e / np.maximum(tot_n, 1)
        ok = per_px > MIN_EXP_FRAC * np.nanmax(per_px)
        if ok.sum() < 6:
            print("  %-24s too few annuli with exposure (%d)" % (m["erass"], ok.sum()))
            continue
        sb = tot_c[ok] / tot_e[ok]
        err = np.sqrt(np.maximum(tot_c[ok], 1.0)) / tot_e[ok]
        th = ctr[ok]

        # NO CUT ON THE SURFACE BRIGHTNESS. An earlier attempt restricted the
        # fit to annuli above 2x an estimated background, on the theory that the
        # sub-1% outer errors were making the fit abandon the cluster to match
        # the plateau. Fitting each cluster three ways settled it (the run is in
        # `sb_cut_test` below):
        #
        #   cluster J123625.2+163246   all annuli  beta 0.58  Rc 137 kpc  0.08 dex
        #                              >2x bkg cut beta 4789  Rc 13939 kpc 0.56 dex
        #
        # The cut was not repairing the runaway, it was CAUSING it -- and on ten
        # of eleven clusters the cut and no-cut fits agree. The real runaway had
        # one cause, the exposure threshold above, and it is fixed there.
        #
        # Cutting on sb is also the mechanism of this programme's artefact #10:
        # a threshold on the noisy axis keeps upward fluctuations at the
        # boundary and discards downward ones. It comes out.
        #
        # What stays is a systematic floor on the errors, which is a statement
        # about the measurement rather than a selection on it: flat-fielding and
        # background-model error that the Poisson term does not contain. The
        # background itself remains a free parameter of the model, which is
        # where a background belongs.
        err = np.sqrt(err ** 2 + (SYS_FRAC * sb) ** 2)

        fit = fit_beta(th, sb, err)
        if fit is None:
            continue
        fit["n_annuli_fitted"] = int(len(th))

        DA = float(C.d_ang(z)) / C.MPC
        rc_mpc = math.radians(fit["Rc_arcmin"] / 60.0) * DA
        beta = fit["beta"]

        ap = np.array(acc[m["accept"]])
        r_acc, ne_acc = ap[:, 0], ap[:, 1]
        lo = max(r_acc.min(), math.radians(TH_MIN / 60.0) * DA)
        hi = min(r_acc.max(), math.radians(TH_MAX / 60.0) * DA)
        sel = (r_acc >= lo) & (r_acc <= hi)
        if sel.sum() < 3:
            continue
        shape = (1.0 + (r_acc[sel] / rc_mpc) ** 2) ** (-1.5 * beta)
        n0 = float(np.exp(np.median(np.log(ne_acc[sel] / shape))))
        dex = float(np.std(np.log10(ne_acc[sel] / (n0 * shape))))

        # A core smaller than the innermost annulus is not measured, it is
        # extrapolated, and beta is degenerate with it there. Say so explicitly
        # rather than letting a bound reject it as if the fit had misbehaved.
        rc_resolved = math.radians(TH_MIN / 60.0) * DA
        bad = []
        if rc_mpc < rc_resolved:
            bad.append("core unresolved: Rc %.0f kpc < innermost annulus %.0f kpc"
                       % (rc_mpc * 1000, rc_resolved * 1000))
        if not (0.010 <= rc_mpc <= 1.0):
            bad.append("Rc %.0f kpc" % (rc_mpc * 1000))
        if not (0.35 <= beta <= 1.3):
            bad.append("beta %.2f" % beta)
        if not (1e-4 <= n0 <= 1.0):
            bad.append("n0 %.1e" % n0)
        if dex > 0.25:
            bad.append("anchor %.2f dex" % dex)
        if bad:
            print("  %-24s REJECTED: %s" % (m["erass"], "; ".join(bad)))
            continue

        out[m["erass"]] = dict(
            accept=m["accept"], z=z, beta=beta, rc_mpc=rc_mpc, n0=n0,
            reach_mpc=math.radians(TH_MAX / 60.0) * DA,
            anchor_bins=int(sel.sum()), anchor_scatter_dex=round(dex, 3),
            rc_resolved_mpc=rc_resolved, sb_cut_test="no cut on sb; see comment in regas.py",
            accept_r_max=float(r_acc.max()), n_obs=nobs,
            exposure_corrected=True,
            vignetting="CIAO exposure maps (fluximage), not an analytic curve",
            sb_fit=fit)
        was = old.get(m["erass"], {}).get("beta")
        print("  %-24s %-7.3f %-7.0f %-9.2e %-6d %-8.2f %s"
              % (m["erass"], beta, rc_mpc * 1000, n0, nobs, dex,
                 ("%.3f" % was) if was else "-"))

    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("\nwrote gas_extended_expcorr.json for %d clusters" % len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
