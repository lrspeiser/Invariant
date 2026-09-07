"""build_table.py -- join the two channels into one machine-learning table.

One row per (cluster, radial bin, azimuthal sector). Baryons on one side from
Chandra photons, gravity on the other from DECADE shapes, at the same place on
the sky, with nothing azimuthally averaged.

This is the first dataset in the programme where a candidate law that depends on
DIRECTION could be distinguished from one that does not. Every previous cluster
input was a radial profile or a stacked mean, and both are azimuthally symmetric
by construction, so the question could not even be posed.

    python build_table.py

Emits table.csv (for anything that reads CSV) and table.json (with the column
documentation attached, because a column called `sb` is useless six months from
now without the note saying it is not exposure-corrected).
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHEAR = os.path.join(HERE, "shear_cells.jsonl")
XRAY = os.path.join(HERE, "xray_cells.jsonl")

COLUMNS = [
    ("cluster", "eRASS1 designation"),
    ("i_rad", "radial bin index, 0 innermost of 5 over 0.2-3.5 h^-1 Mpc"),
    ("j_az", "azimuthal sector index, 0-7, position angle east of north"),
    ("z", "cluster redshift"),
    ("kT", "eRASS1 X-ray temperature, keV, blank where unmeasured"),
    ("R_mpc", "weighted mean physical radius of the lensing sources in the cell"),
    ("pa_deg", "weighted mean position angle of those sources"),
    ("n_src", "background sources in the cell"),
    ("g_t", "tangential reduced shear, metacalibration response applied"),
    ("g_x", "cross component -- must be consistent with zero; a null, not a feature"),
    ("g_err", "shape-noise error on g_t"),
    ("n_phot", "Chandra photons in the cell, 0.5-7 keV, good grades"),
    ("sb", "photons per square arcmin. NOT exposure corrected"),
    ("hardness", "H/(S+H) temperature proxy, blank below 25 photons"),
    ("hardness_asym", "hardness minus this cluster's radial median -- vignetting-free"),
    ("sb_asym", "log10 sb minus this cluster's radial median -- the asymmetry"),
    ("xray_covered", "1 if Chandra observed this cell's radius at all"),
]


def load(path, key="key"):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("cells"):
            out[r[key]] = r
    return out


def main():
    sh = load(SHEAR)
    xr = load(XRAY)
    common = sorted(set(sh) & set(xr))
    print("clusters with both channels: %d" % len(common))

    rows = []
    for k in common:
        s, x = sh[k], xr[k]
        xmap = {(c["i_rad"], c["j_az"]): c for c in x["cells"]}
        for c in s["cells"]:
            xc = xmap.get((c["i_rad"], c["j_az"]))
            if xc is None:
                continue
            rows.append(dict(
                cluster=s["name"], i_rad=c["i_rad"], j_az=c["j_az"],
                z=s["z"], kT=s.get("kt") or "",
                R_mpc=c.get("R"), pa_deg=c.get("pa"), n_src=c.get("n"),
                g_t=c.get("gt"), g_x=c.get("gx"), g_err=c.get("err"),
                n_phot=xc.get("n_photons"), sb=xc.get("sb"),
                hardness=xc.get("hardness"),
                hardness_asym=xc.get("hardness_asym"),
                sb_asym=xc.get("sb_asym"),
                xray_covered=int(bool(xc.get("covered")))))

    names = [c for c, _ in COLUMNS]
    with io.open(os.path.join(HERE, "table.csv"), "w", newline="",
                 encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=names)
        wr.writeheader()
        for r in rows:
            wr.writerow({n: ("" if r.get(n) is None else r.get(n)) for n in names})

    usable = [r for r in rows
              if r["g_t"] is not None and r["g_err"] and r["xray_covered"]
              and r["sb_asym"] is not None]
    with_h = [r for r in usable if r["hardness_asym"] is not None]

    doc = dict(
        rows=len(rows), usable=len(usable), usable_with_hardness=len(with_h),
        clusters=len(common), nrad=5, naz=8,
        columns=[dict(name=n, meaning=m) for n, m in COLUMNS],
        target="g_t, weighted by 1/g_err**2",
        nulls=["g_x must be consistent with zero on any subset used",
               "permute cluster labels to destroy the baryon-gravity association",
               "permute j_az within a cluster to destroy the ASYMMETRY while "
               "leaving every radial quantity intact -- the null that matters "
               "here, because a purely radial model survives it"],
        caveats=["sb is not exposure corrected: vignetting and chip gaps are in it. "
                 "Use sb_asym, which has the radial part removed, for anything "
                 "that must not depend on the instrument.",
                 "hardness is a proxy, not a temperature, and is not comparable "
                 "between clusters at different redshift or epoch.",
                 "outer cells often fall outside the Chandra field; xray_covered "
                 "flags that and unflagged rows must be dropped, not imputed."])
    io.open(os.path.join(HERE, "table.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(doc, indent=1) + "\n")

    print("rows written        : %d" % len(rows))
    print("usable (shear + X-ray coverage + asymmetry): %d" % len(usable))
    print("  of those, with a hardness measurement    : %d" % len(with_h))
    if usable:
        gt = np.array([r["g_t"] for r in usable])
        er = np.array([r["g_err"] for r in usable])
        print("  weighted mean g_t %.5f, median S/N per cell %.2f"
              % (float(np.sum(gt / er ** 2) / np.sum(1 / er ** 2)),
                 float(np.median(np.abs(gt) / er))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
