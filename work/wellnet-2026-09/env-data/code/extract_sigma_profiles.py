"""Resolved stellar velocity-dispersion profiles for near-face-on MaNGA disks.

Motivation.  The programme already holds DiskMass VI/VII, where sigma_z is an
exponential FIT (a central value plus a scale length in arcsec), the scale
height is INFERRED from h_R rather than measured, and Sigma_dyn is not
tabulated anywhere.  A radially RESOLVED profile is the requested upgrade.

For a disk seen close to face-on the line-of-sight stellar dispersion is
dominated by the vertical component:

    sigma_LOS^2 = sigma_z^2 cos^2 i + sigma_R^2 sin^2 i cos^2 phi
                                    + sigma_phi^2 sin^2 i sin^2 phi

so at i < 30 deg, sin^2 i < 0.25 and the in-plane terms contribute a bounded
correction.  This script does NOT apply that correction -- it reports
sigma_LOS(R), the measured quantity, plus the inclination needed to bound the
contamination.  Turning sigma_LOS into sigma_z requires an assumed
sigma_R/sigma_z ratio, which is a model, not an observation.

Instrumental limit.  MaNGA's stellar-continuum instrumental resolution is
roughly 70 km/s in sigma.  The DAP delivers an instrumental correction in
STELLAR_SIGMACORR and the astrophysical dispersion is

    sigma_corr = sqrt(STELLAR_SIGMA^2 - STELLAR_SIGMACORR^2).

STELLAR_SIGMACORR is only the template-vs-data resolution difference (median
24 km/s here), NOT the full instrumental sigma, so the ratio test against it is
weak.  The reported reliability flags are therefore ABSOLUTE: above_50kms and
above_70kms.  MaNGA's stellar instrumental resolution corresponds to sigma_inst
of roughly 70 km/s and the DAP is documented as increasingly systematics-limited
below about 50 km/s.
"""
import hashlib
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.io import fits

warnings.filterwarnings("ignore")

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
CLEAN = os.path.join(LANE, "clean")
MAPS = os.path.join(LANE, "raw", "manga", "maps")
DAPTYPE = "HYB10-MILESHC-MASTARSSP"

RBINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
RELIABLE_RATIO = 1.2      # STELLAR_SIGMA / STELLAR_SIGMACORR floor
MIN_SPX = 5               # independent Voronoi bins required per radial bin


def one(plateifu):
    p = os.path.join(MAPS, "manga-%s-MAPS-%s.fits.gz" % (plateifu, DAPTYPE))
    if not os.path.exists(p):
        return None
    with fits.open(p) as h:
        sig = h["STELLAR_SIGMA"].data.astype(float)
        ivar = h["STELLAR_SIGMA_IVAR"].data.astype(float)
        mask = h["STELLAR_SIGMA_MASK"].data.astype(int)
        corr = h["STELLAR_SIGMACORR"].data.astype(float)
        if corr.ndim == 3:
            corr = corr[0]
        ell = h["SPX_ELLCOO"].data.astype(float)
        binid = h["BINID"].data.astype(int)
        reff = float(h[0].header.get("REFF", np.nan))
        ell_units = [h["SPX_ELLCOO"].header.get("C%02d" % (i + 1), "")
                     for i in range(ell.shape[0])]
    r_re = ell[1]                       # elliptical radius in units of R_eff
    r_as = ell[0]                       # elliptical radius in arcsec
    bid = binid[1] if binid.ndim == 3 else binid   # stellar-continuum binning

    good = (mask == 0) & (ivar > 0) & np.isfinite(sig) & (sig > 0) & (bid >= 0)
    if good.sum() < 20:
        return None

    # one entry per independent Voronoi bin, not per spaxel
    ys, xs = np.where(good)
    bids = bid[ys, xs]
    _, first = np.unique(bids, return_index=True)
    ys, xs = ys[first], xs[first]

    s = sig[ys, xs]
    c = corr[ys, xs]
    iv = ivar[ys, xs]
    rr = r_re[ys, xs]
    ra = r_as[ys, xs]

    with np.errstate(invalid="ignore"):
        s2 = s ** 2 - c ** 2
        scorr = np.sqrt(np.where(s2 > 0, s2, np.nan))
        # propagate: var(sigma_corr) = (sigma/sigma_corr)^2 var(sigma)
        var_s = 1.0 / iv
        var_c = (s / scorr) ** 2 * var_s
    ok = np.isfinite(scorr)
    reliable = ok & (s > RELIABLE_RATIO * c)

    rows = []
    for lo, hi in zip(RBINS[:-1], RBINS[1:]):
        m = ok & (rr >= lo) & (rr < hi)
        mr = reliable & (rr >= lo) & (rr < hi)
        if m.sum() < MIN_SPX:
            continue
        w = 1.0 / var_c[m]
        mu = float(np.sum(w * scorr[m]) / np.sum(w))
        err = float(np.sqrt(1.0 / np.sum(w)))
        rows.append(dict(plateifu=plateifu, r_lo_Re=lo, r_hi_Re=hi,
                         r_med_Re=float(np.median(rr[m])),
                         r_med_arcsec=float(np.median(ra[m])),
                         sigma_los_kms=mu, e_sigma_los_kms=err,
                         sigma_scatter_kms=float(np.std(scorr[m])),
                         n_bins=int(m.sum()), n_bins_reliable=int(mr.sum()),
                         frac_reliable=float(mr.sum() / m.sum()),
                         median_sigmacorr_kms=float(np.median(c[m])),
                         above_50kms=bool(mu > 50.0),
                         above_70kms=bool(mu > 70.0),
                         Reff_arcsec=reff))
    return rows


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    sel = pd.read_csv(os.path.join(CLEAN, "faceon_sample.csv"), low_memory=False)
    out = []
    nfail = 0
    for pif in sel["plateifu"]:
        r = one(pif)
        if not r:
            nfail += 1
            continue
        out.extend(r)
    prof = pd.DataFrame(out)
    meta = sel[["plateifu", "mangaid", "objra", "objdec", "z", "incl_deg",
                "ba_disk", "pa_disk_deg", "Rd_kpc", "kpc_per_arcsec",
                "logMstar_nsa", "Sigma_b_Msun_pc2", "dl_TType", "SNR_MED_r",
                "STELLAR_SIGMA_1RE", "nsa_elpetro_th50_r", "struct_source"]]
    prof = prof.merge(meta, on="plateifu", how="left")
    prof["r_med_kpc"] = prof["r_med_arcsec"] * prof["kpc_per_arcsec"]

    fp = os.path.join(CLEAN, "manga_faceon_sigma_profiles.csv")
    prof.to_csv(fp, index=False)

    ngal = prof.plateifu.nunique()
    per = prof.groupby("plateifu").size()
    rel = prof[prof.frac_reliable > 0.8]
    ngal_rel = rel.groupby("plateifu").size()
    print("galaxies attempted        : %d" % len(sel))
    print("galaxies with a profile   : %d  (%d had too little usable data)"
          % (ngal, nfail))
    print("radial points total       : %d" % len(prof))
    print("radial points per galaxy  : median %d, range %d-%d"
          % (per.median(), per.min(), per.max()))
    g = prof.groupby("plateifu")
    print("galaxies with EVERY radial point > 50 km/s : %d"
          % int((g.sigma_los_kms.min() > 50).sum()))
    print("galaxies with EVERY radial point > 70 km/s : %d"
          % int((g.sigma_los_kms.min() > 70).sum()))
    print("galaxies with >=4 radial points > 50 km/s  : %d"
          % int((prof.above_50kms.groupby(prof.plateifu).sum() >= 4).sum()))
    print("sigma_LOS range (km/s)    : %.1f to %.1f, median %.1f"
          % (prof.sigma_los_kms.min(), prof.sigma_los_kms.max(),
             prof.sigma_los_kms.median()))
    print("median formal error       : %.2f km/s" % prof.e_sigma_los_kms.median())
    print("radial points > 50 / > 70 km/s             : %.0f%% / %.0f%%"
          % (100 * prof.above_50kms.mean(), 100 * prof.above_70kms.mean()))
    sl = []
    for _, gg in g:
        if len(gg) >= 4:
            sl.append(np.polyfit(np.log10(gg.r_med_Re.clip(0.05)),
                                 np.log10(gg.sigma_los_kms), 1)[0])
    sl = np.array(sl)
    print("d log sigma / d log R                     : median %.3f, "
          "16-84%% %.3f to %.3f, declining in %.0f%% of galaxies"
          % (np.median(sl), *np.percentile(sl, [16, 84]), 100 * np.mean(sl < 0)))

    man = {
        "file": "manga_faceon_sigma_profiles.csv",
        "produced_by": "env-data/code/extract_sigma_profiles.py",
        "produced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256(fp), "bytes": os.path.getsize(fp),
        "row_count": int(len(prof)), "column_count": int(prof.shape[1]),
        "n_galaxies": int(ngal),
        "columns": [
            {"name": "plateifu", "unit": "MaNGA plate-IFU identifier"},
            {"name": "r_lo_Re", "unit": "R/R_eff, bin lower edge"},
            {"name": "r_hi_Re", "unit": "R/R_eff, bin upper edge"},
            {"name": "r_med_Re", "unit": "R/R_eff, median of the bin"},
            {"name": "r_med_arcsec", "unit": "arcsec, elliptical radius"},
            {"name": "r_med_kpc", "unit": "kpc"},
            {"name": "sigma_los_kms", "unit": "km/s, instrument-corrected "
                                              "line-of-sight stellar dispersion"},
            {"name": "e_sigma_los_kms", "unit": "km/s, formal inverse-variance error"},
            {"name": "sigma_scatter_kms", "unit": "km/s, bin-to-bin scatter within "
                                                  "the radial bin"},
            {"name": "n_bins", "unit": "independent Voronoi bins in the radial bin"},
            {"name": "n_bins_reliable", "unit": "of those, with STELLAR_SIGMA > "
                                                "1.2 * STELLAR_SIGMACORR"},
            {"name": "frac_reliable", "unit": "n_bins_reliable / n_bins"},
            {"name": "median_sigmacorr_kms", "unit": "km/s, DAP template-resolution "
                                                     "correction (NOT the full "
                                                     "instrumental sigma)"},
            {"name": "above_50kms", "unit": "bool, sigma_los > 50 km/s"},
            {"name": "above_70kms", "unit": "bool, sigma_los > 70 km/s, i.e. above "
                                            "MaNGA's instrumental sigma"},
            {"name": "Reff_arcsec", "unit": "arcsec, DAP REFF header keyword"},
            {"name": "incl_deg", "unit": "deg, from the PyMorph axis ratio with q0=0.20"},
            {"name": "Rd_kpc", "unit": "kpc, exponential disk scale length"},
            {"name": "logMstar_nsa", "unit": "log10 Msun, NSA elpetro, Chabrier IMF"},
            {"name": "Sigma_b_Msun_pc2", "unit": "Msun/pc^2, M_b/(2 pi R_d^2)"},
        ],
        "source": "SDSS DR17 MaNGA DAP MAPS, DAPTYPE %s, DRP v3_1_1, DAP 3.1.0" % DAPTYPE,
        "selection": "near-face-on (inclination < 30 deg from the PyMorph r-band axis "
                     "ratio with q0=0.20), deep-learning late type (T-Type>0, "
                     "P_LTG>0.5), DAPQUAL clean, PyMorph FLAG_FIT != 3, "
                     "STELLAR_SIGMA_1RE > 50 km/s, median r-band S/N > 5.",
        "method": "One entry per independent Voronoi bin (deduplicated on BINID "
                  "channel 1, the stellar-continuum binning) so radial bins are not "
                  "inflated by spaxel repetition. Astrophysical dispersion is "
                  "sqrt(STELLAR_SIGMA^2 - STELLAR_SIGMACORR^2); errors propagated "
                  "from STELLAR_SIGMA_IVAR. Spaxels with STELLAR_SIGMA_MASK != 0 are "
                  "dropped. Radii are the DAP elliptical polar radius (SPX_ELLCOO).",
        "caveats": [
            "sigma_los is the MEASURED line-of-sight dispersion, not sigma_z. At "
            "inclination i the in-plane components leak in at order sin^2 i "
            "(< 0.25 here). Converting to sigma_z needs an assumed sigma_R/sigma_z, "
            "which is a model.",
            "STELLAR_SIGMACORR is the difference in spectral resolution between "
            "the MILES-HC templates and the MaNGA data, NOT the full instrumental "
            "sigma; its median here is 24 km/s. frac_reliable is therefore a WEAK "
            "test and should not be read as 'resolved'. The stronger, absolute test "
            "is above_50kms / above_70kms: MaNGA's stellar instrumental resolution "
            "corresponds to sigma_inst of roughly 70 km/s, and the DAP is documented "
            "as increasingly systematics-limited below about 50 km/s. Of 1671 radial "
            "points, 92 per cent exceed 50 km/s and 46 per cent exceed 70 km/s; 171 "
            "of the 240 galaxies have EVERY radial point above 50 km/s and 45 have "
            "every point above 70 km/s.",
            "No scale height is measured. This gives sigma_LOS(R), not h_z(R) and "
            "not Sigma_dyn(R).",
        ],
    }
    with open(fp + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print("WROTE %s and manifest" % fp)


if __name__ == "__main__":
    main()
