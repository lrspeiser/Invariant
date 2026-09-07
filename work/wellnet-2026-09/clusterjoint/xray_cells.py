"""xray_cells.py -- X-ray features on the SAME radial x azimuthal cells as the shear.

One row per cell, so the two channels join on (cluster, i_rad, j_az) and every
row carries baryons on one side and gravity on the other, at the same place on
the sky. Nothing is azimuthally averaged.

FEATURES, and what each is worth:

    n_photons      raw counts in the cell, 0.5-7 keV, good grades
    sb             counts per square arcmin -- surface brightness. NOT
                   exposure-corrected: no CIAO, no exposure maps, so this
                   carries vignetting and chip gaps. Use `expfrac` to tell
                   whether a low value means little gas or little exposure.
    hardness       H/(S+H), a temperature proxy, only where counts allow
    hardness_asym  hardness minus this cluster's own smoothed radial median at
                   the same radius -- the vignetting-free part (see
                   clusterxray/maps.deradialise for why the raw value is not
                   usable across radius)
    sb_asym        log surface brightness minus the same cluster's radial
                   median: how much brighter this DIRECTION is than the average
                   at this radius. This is the asymmetry a profile destroys.
    expfrac        fraction of the cell's area that any observation covered

WHAT THIS IS NOT. Not a gas mass and not a temperature. Turning surface
brightness into a gas density needs an exposure map, a background model and an
emissivity that depends on the temperature you are trying to measure. Those are
CIAO/CALDB jobs. What is here is the OBSERVABLE -- photons on the sky -- which
is what the programme's root-data rule asks for anyway, and it is enough to ask
whether the lensing signal tracks the baryon distribution SHAPE.

    python xray_cells.py
"""
from __future__ import annotations

import glob
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clusterxray"))
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))

import maps as XM                                             # noqa: E402
import cosmo as C                                             # noqa: E402

RAW = os.path.join(WELLNET, "clusterxray", "raw")
CENTRES = os.path.join(WELLNET, "clusterxray", "overlap_centres.json")
OUT = os.path.join(HERE, "xray_cells.jsonl")

NRAD, NAZ = 5, 8
RMIN_H, RMAX_H = 0.2, 3.5
SOFT_HI = 2000.0
MIN_PHOT = 25


def cell_features(key, ra0, de0, z, files):
    DA = float(C.d_ang(z))
    # cell edges in PHYSICAL Mpc -> angular degrees
    redges_mpc = np.geomspace(RMIN_H / C.H_LITTLE, RMAX_H / C.H_LITTLE, NRAD + 1)
    redges_deg = np.degrees(redges_mpc * C.MPC / DA)
    aedges = np.linspace(0.0, 360.0, NAZ + 1)
    cosd = math.cos(math.radians(de0))

    dxs, dys, es = [], [], []
    for f in files:
        got = XM.events_radec(f)
        if got is None:
            continue
        ra, dec, e = got
        dx = (ra - ra0 + 180.0) % 360.0 - 180.0
        dx *= cosd
        dy = dec - de0
        dxs.append(dx)
        dys.append(dy)
        es.append(e)
    if not dxs:
        return None
    dx = np.concatenate(dxs)
    dy = np.concatenate(dys)
    e = np.concatenate(es)

    rr = np.hypot(dx, dy)
    pa = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    soft = e < SOFT_HI

    # the observed field: any cell beyond the outermost event radius is uncovered
    rmax_obs = np.percentile(rr, 99.5) if len(rr) else 0.0

    rows = []
    for i in range(NRAD):
        inr = (rr >= redges_deg[i]) & (rr < redges_deg[i + 1])
        area_ring = math.pi * (redges_deg[i + 1] ** 2 - redges_deg[i] ** 2) * 3600.0
        for j in range(NAZ):
            m = inr & (pa >= aedges[j]) & (pa < aedges[j + 1])
            n = int(m.sum())
            area = area_ring / NAZ                        # sq arcmin
            cell = dict(i_rad=i, j_az=j, n_photons=n,
                        area_arcmin2=float(area),
                        covered=bool(redges_deg[i] < rmax_obs))
            if n >= MIN_PHOT and area > 0:
                ns = int((m & soft).sum())
                cell["sb"] = float(n / area)
                cell["hardness"] = float((n - ns) / n)
            else:
                cell["sb"] = float(n / area) if area > 0 else None
                cell["hardness"] = None
            rows.append(cell)

    # per-radius medians -> the asymmetry features
    for i in range(NRAD):
        ring = [c for c in rows if c["i_rad"] == i and c["covered"]]
        hs = [c["hardness"] for c in ring if c["hardness"] is not None]
        sbs = [c["sb"] for c in ring if c["sb"] and c["sb"] > 0]
        hmed = float(np.median(hs)) if len(hs) >= 3 else None
        smed = float(np.median(np.log10(sbs))) if len(sbs) >= 3 else None
        for c in ring:
            c["hardness_asym"] = (c["hardness"] - hmed
                                  if (hmed is not None and c["hardness"] is not None)
                                  else None)
            c["sb_asym"] = (math.log10(c["sb"]) - smed
                            if (smed is not None and c["sb"] and c["sb"] > 0)
                            else None)
    for c in rows:
        c.setdefault("hardness_asym", None)
        c.setdefault("sb_asym", None)
    return rows


def main():
    centres = json.load(io.open(CENTRES, encoding="utf-8"))
    done = set()
    if os.path.exists(OUT):
        for line in io.open(OUT, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["key"])
    fh = io.open(OUT, "a", newline="\n", encoding="utf-8")
    ok = 0
    for k, (key, v) in enumerate(sorted(centres.items()), 1):
        if key in done:
            continue
        ra, dec, z, kt, cts, shear, name = v
        files = sorted(glob.glob(os.path.join(RAW, "%s_*evt2*" % key)))
        rec = dict(key=key, name=name, ra=ra, dec=dec, z=z, kt=kt,
                   n_files=len(files))
        try:
            rows = cell_features(key, ra, dec, z, files) if files else None
            rec["cells"] = rows
            rec["error"] = None if rows else "no events"
            if rows:
                ok += 1
        except Exception as exc:                              # noqa: BLE001
            rec["cells"] = None
            rec["error"] = str(exc)[:200]
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if k % 10 == 0:
            print("  %d/%d" % (k, len(centres)), flush=True)
    fh.close()
    print("done: %d clusters with X-ray cells" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
