"""maps.py -- 2D photon and hardness maps, on a grid, with the sphere assumption gone.

Each observation's events are converted from detector sky pixels to RA/Dec
through that observation's OWN tangent-plane WCS, then stacked onto a common
grid centred on the cluster. Observations taken years apart at different roll
angles therefore co-add correctly; using raw X/Y would smear them.

TWO BANDS. soft = 0.5-2.0 keV, hard = 2.0-7.0 keV. The hardness ratio
H/(S+H) rises with gas temperature, so it is a temperature PROXY -- not a
temperature. Converting it to keV needs an instrument response and a spectral
fit (CIAO plus CALDB), neither of which is installed here, and the ACIS
effective area has also degraded with time so the same gas gives a different
ratio in 2000 and 2024. What the ratio is good for is RELATIVE structure inside
one cluster observed in one epoch: where the gas is hotter or cooler than its
surroundings. That is exactly the asymmetry the spherical deprojection destroys.

FILTERING. Standard ACIS event selection: grade in {0,2,3,4,6}, status = 0,
energy in band. evt2 files are already largely cleaned; this is belt and braces.

SMOOTHING. Both bands are smoothed with the SAME Gaussian kernel before the
ratio is taken, so the ratio is not biased by differing count levels. The
hardness map is masked where the smoothed total counts fall below a threshold,
because a ratio of two small numbers is noise wearing a colour.

    python maps.py
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
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "maps.json")

CENTRES = {
    "ABELL_0370":        (39.9700,  -1.5767),
    "ABELL_1063S":       (342.1833, -44.5306),
    "ABELL_2029":        (227.7337,   5.7449),
    "ABELL_2744":        (3.5862,  -30.3997),
    "MACS_J0717_5p3745": (109.3817, 37.7550),
    "MACS_J1149_5p2223": (177.3987, 22.3985),
}

FIELD_ARCMIN = 12.0        # half-width of the map
NPIX = 128
SOFT = (500.0, 2000.0)
HARD = (2000.0, 7000.0)
GOOD_GRADES = (0, 2, 3, 4, 6)
SMOOTH_PIX = 2.2
MIN_COUNTS = 12.0          # smoothed total below this -> hardness masked


def events_radec(path):
    """Return (ra, dec, energy) for good events, using this file's own WCS."""
    with fits.open(path, memmap=False) as hdul:
        ev = hdul["EVENTS"]
        d = ev.data
        h = ev.header
        names = [c.upper() for c in ev.columns.names]
        ix = {n: i + 1 for i, n in enumerate(names)}
        cx, cy = ix["X"], ix["Y"]
        ra0, x0, dx = (float(h["TCRVL%d" % cx]), float(h["TCRPX%d" % cx]),
                       float(h["TCDLT%d" % cx]))
        de0, y0, dy = (float(h["TCRVL%d" % cy]), float(h["TCRPX%d" % cy]),
                       float(h["TCDLT%d" % cy]))

        m = np.ones(len(d), dtype=bool)
        if "GRADE" in names:
            m &= np.isin(d["grade"], GOOD_GRADES)
        if "STATUS" in names:
            st = d["status"]
            m &= (st.sum(axis=1) == 0) if st.ndim > 1 else (st == 0)
        e = np.asarray(d["energy"], dtype=float)
        m &= (e >= SOFT[0]) & (e < HARD[1])
        if not m.any():
            return None

        x = np.asarray(d["x"], dtype=float)[m]
        y = np.asarray(d["y"], dtype=float)[m]
        e = e[m]

        # inverse tangent-plane projection
        xi = (x - x0) * dx * math.pi / 180.0        # radians, east-positive
        eta = (y - y0) * dy * math.pi / 180.0
        d0 = math.radians(de0)
        denom = np.cos(d0) - eta * np.sin(d0)
        ra = math.radians(ra0) + np.arctan2(xi, denom)
        dec = np.arctan((np.sin(d0) + eta * np.cos(d0))
                        / np.hypot(xi, denom))
        return np.degrees(ra), np.degrees(dec), e


def build(cluster, files):
    ra0, de0 = CENTRES[cluster]
    half = FIELD_ARCMIN / 60.0
    cosd = math.cos(math.radians(de0))
    edges_x = np.linspace(-half, half, NPIX + 1)
    edges_y = np.linspace(-half, half, NPIX + 1)
    soft = np.zeros((NPIX, NPIX))
    hard = np.zeros((NPIX, NPIX))
    n_used = 0

    for f in files:
        got = events_radec(f)
        if got is None:
            continue
        ra, dec, e = got
        dx = (ra - ra0)
        dx = (dx + 180.0) % 360.0 - 180.0
        dx *= cosd                                  # degrees on the sky
        dy = dec - de0
        keep = (np.abs(dx) < half) & (np.abs(dy) < half)
        if not keep.any():
            continue
        dx, dy, e = dx[keep], dy[keep], e[keep]
        s = e < SOFT[1]
        hs, _, _ = np.histogram2d(dy[s], dx[s], bins=[edges_y, edges_x])
        hh, _, _ = np.histogram2d(dy[~s], dx[~s], bins=[edges_y, edges_x])
        soft += hs
        hard += hh
        n_used += 1

    total = soft + hard
    ss = gaussian_filter(soft, SMOOTH_PIX)
    sh = gaussian_filter(hard, SMOOTH_PIX)
    st = ss + sh
    with np.errstate(invalid="ignore", divide="ignore"):
        hr = np.where(st > 0, sh / np.maximum(st, 1e-9), np.nan)
    hr[st < MIN_COUNTS / (2 * math.pi * SMOOTH_PIX ** 2) * 1.0] = np.nan
    return dict(counts=total, smooth=st, hardness=hr,
                residual=deradialise(hr, half),
                n_obs=n_used, n_events=float(total.sum()))


def deradialise(hr, half_deg):
    """Remove the azimuthally symmetric part of the hardness map.

    MEASURED, not assumed: the raw hardness rises from centre to 8 arcmin by
    +0.124 on average across these six clusters, with a scatter of only 0.030 --
    the same gradient in Abell 2029 (relaxed, cool-core, z=0.077) as in MACS
    J0717 (quadruple merger, z=0.546). Two clusters that different cannot share
    a thermal profile. That gradient is ACIS vignetting, which is energy
    dependent: soft photons lose more effective area off-axis, so the ratio
    hardens outward whatever the gas is doing. Correcting it properly needs
    exposure maps from CIAO and CALDB, neither of which is installed.

    Vignetting is azimuthally symmetric by construction -- it depends only on
    off-axis angle -- so subtracting each cluster's own median hardness in
    radial annuli removes it AND removes any genuinely spherical thermal
    structure with it. What survives is exactly the ASYMMETRY: how much hotter
    or cooler each direction is than the average at that radius. That is the
    quantity a spherical deprojection cannot represent at all, and it is the
    only thing these maps are entitled to show.

    Implementation note: the profile is built in FINE annuli and then smoothed
    and interpolated, rather than subtracted bin by bin. Hard annular bins leave
    concentric ring artefacts at every bin edge -- visible in the first version
    of this map and easily mistaken for shock fronts, which are exactly what one
    would be looking for. The correction must not manufacture the signal.
    """
    n = hr.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    scale = 2.0 * half_deg * 60.0 / n                # arcmin per pixel
    rr = np.hypot((x - n / 2 + 0.5), (y - n / 2 + 0.5)) * scale

    step = 0.25
    edges = np.arange(0.0, rr.max() + step, step)
    ctr, med = [], []
    for i in range(len(edges) - 1):
        m = (rr >= edges[i]) & (rr < edges[i + 1]) & np.isfinite(hr)
        if m.sum() < 20:
            continue
        ctr.append(0.5 * (edges[i] + edges[i + 1]))
        med.append(float(np.median(hr[m])))
    if len(ctr) < 4:
        return np.full_like(hr, np.nan)
    ctr = np.array(ctr)
    med = np.array(med)
    # smooth the 1D profile so its own bin edges cannot print rings on the map
    k = max(3, int(round(1.0 / step)) | 1)
    pad = np.pad(med, k // 2, mode="edge")
    smooth = np.convolve(pad, np.ones(k) / k, mode="valid")
    base = np.interp(rr, ctr, smooth, left=smooth[0], right=smooth[-1])
    out = np.where(np.isfinite(hr), hr - base, np.nan)
    return out


def encode(a, lo, hi, nan=255):
    """Quantise to 0..254 for transport; 255 marks masked."""
    out = np.full(a.shape, nan, dtype=np.uint8)
    m = np.isfinite(a)
    v = np.clip((a[m] - lo) / (hi - lo), 0, 1) * 254.0
    out[m] = v.astype(np.uint8)
    return out


def main():
    res, payload = {}, {}
    for cluster in CENTRES:
        files = sorted(glob.glob(os.path.join(RAW, "%s_*evt2*" % cluster)))
        if not files:
            print("%-20s no files" % cluster)
            continue
        b = build(cluster, files)
        cnt, hr = b["smooth"], b["hardness"]
        lo = np.nanpercentile(cnt[cnt > 0], 5) if (cnt > 0).any() else 0.0
        hi = np.nanpercentile(cnt, 99.8)
        logc = np.log10(np.maximum(cnt, lo * 0.5 + 1e-6))
        good = np.isfinite(hr)
        hlo, hhi = (np.nanpercentile(hr[good], 5), np.nanpercentile(hr[good], 95)) \
            if good.any() else (0.0, 1.0)
        res[cluster] = dict(
            n_obs=b["n_obs"], n_events=int(b["n_events"]),
            field_arcmin=FIELD_ARCMIN, npix=NPIX,
            hardness_lo=float(hlo), hardness_hi=float(hhi),
            hardness_median=float(np.nanmedian(hr[good])) if good.any() else None,
            hardness_spread=float(np.nanstd(hr[good])) if good.any() else None,
            frac_mapped=float(good.mean()))
        rs = b["residual"]
        rg = np.isfinite(rs)
        rlim = float(np.nanpercentile(np.abs(rs[rg]), 97)) if rg.any() else 0.05
        res[cluster]["residual_limit"] = rlim
        res[cluster]["residual_rms"] = float(np.nanstd(rs[rg])) if rg.any() else None
        payload[cluster] = dict(
            c=encode(logc, float(np.log10(max(lo, 1e-6))), float(np.log10(max(hi, 1e-5)))).tolist(),
            h=encode(hr, hlo, hhi).tolist(),
            r=encode(rs, -rlim, rlim).tolist())
        print("%-20s %d obs  %8d events  hardness %.3f +- %.3f over %.0f%% of the field"
              % (cluster, b["n_obs"], b["n_events"],
                 res[cluster]["hardness_median"] or 0.0,
                 res[cluster]["hardness_spread"] or 0.0,
                 100 * res[cluster]["frac_mapped"]))

    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(dict(meta=res, bands=dict(soft=SOFT, hard=HARD),
                        smoothing_pix=SMOOTH_PIX), indent=1) + "\n")
    io.open(os.path.join(HERE, "maps_data.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(payload, separators=(",", ":")))
    print("\nwrote maps.json and maps_data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
