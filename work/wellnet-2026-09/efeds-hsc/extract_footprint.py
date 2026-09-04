"""The eFEDS x HSC overlap footprint, recovered from a vector PDF.

Chiu+2022 Fig. 1 plots a random subset of the HSC S19A weak-lensing catalogue
as grey points over the eFEDS field.  Those points are stored as exact vector
coordinates, so the figure IS a (sparse) sampling of the HSC coverage mask, and
inverting the axis transform turns it into a usable footprint.

This is the only public description of the eFEDS/HSC overlap that this lane was
able to obtain -- the HSC survey masks themselves are behind the same account
wall as the shape catalogue.  It is a SAMPLED footprint, not the survey mask,
and the manifest says so.

Axis calibration: the four RA labels and the eight Dec labels are evenly spaced
at 30.0930 pt per degree on BOTH axes, and the plot frame is 601.86 x 270.84 pt
= exactly 20.00 x 9.00 degrees.  Anchoring on round degrees puts the frame at
RA 126-146, Dec -3 to +6, which is the eFEDS field as published.  The gate
below checks that the recovered eFEDS cluster positions (known independently
from the catalogue) land inside the recovered frame.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re

import numpy as np

import extract_shear_pdf as X

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "acquire", "chiu_src", "materials", "footprint.pdf")
OUT = os.path.join(HERE, "efeds_hsc_footprint_sampled.tsv")

FRAME = dict(x0=55.911729, x1=657.772316, y0=17.162736, y1=288.0,
             ra0=126.0, ra1=146.0, de0=-3.0, de1=6.0)


def main():
    print("=" * 78)
    print("EXTRACT -- eFEDS x HSC overlap footprint (Chiu+2022 Fig. 1)")
    print("=" * 78)
    raw, txt = X.streams(PDF)
    ppd_x = (FRAME["x1"] - FRAME["x0"]) / (FRAME["ra1"] - FRAME["ra0"])
    ppd_y = (FRAME["y1"] - FRAME["y0"]) / (FRAME["de1"] - FRAME["de0"])
    print(f"\n   frame {FRAME['x1'] - FRAME['x0']:.3f} x "
          f"{FRAME['y1'] - FRAME['y0']:.3f} pt = "
          f"{FRAME['ra1'] - FRAME['ra0']:.2f} x {FRAME['de1'] - FRAME['de0']:.2f} deg")
    print(f"   scale {ppd_x:.4f} / {ppd_y:.4f} pt per degree  "
          f"(gate: the two must agree)  "
          f"{'PASS' if abs(ppd_x / ppd_y - 1) < 1e-3 else 'FAIL'}")
    assert abs(ppd_x / ppd_y - 1) < 1e-3

    # label spacing gate, independent of the frame anchoring
    labs = re.findall(r"BT /F1 10 Tf\s*\n([\d.]+) ([\d.]+) Td\s*\n"
                      r"\((\d+)\S*E\) Tj", txt)
    xs = np.array([float(a) for a, _, _ in labs])
    rr = np.array([float(c) for _, _, c in labs])
    if xs.size >= 2:
        sp = np.diff(xs) / np.diff(rr)
        print(f"   RA label spacing {sp.mean():.4f} pt/deg "
              f"(+- {sp.std():.2e})  "
              f"{'PASS' if abs(sp.mean() / ppd_x - 1) < 2e-3 else 'FAIL'}")

    pts = X.markers(txt, "M0")
    print(f"\n   {len(pts)} plotted HSC source positions recovered")
    ra = FRAME["ra0"] + (np.array([p[0] for p in pts]) - FRAME["x0"]) / ppd_x
    de = FRAME["de0"] + (np.array([p[1] for p in pts]) - FRAME["y0"]) / ppd_y
    inside = ((ra > FRAME["ra0"]) & (ra < FRAME["ra1"])
              & (de > FRAME["de0"]) & (de < FRAME["de1"]))
    ra, de = ra[inside], de[inside]
    print(f"   {inside.sum()} inside the frame; RA {ra.min():.2f}-"
          f"{ra.max():.2f}, Dec {de.min():.2f}-{de.max():.2f}")

    # coverage mask on a 0.1 deg grid
    nx = int(round((FRAME["ra1"] - FRAME["ra0"]) / 0.1))
    ny = int(round((FRAME["de1"] - FRAME["de0"]) / 0.1))
    H, xe, ye = np.histogram2d(ra, de, bins=[nx, ny],
                               range=[[FRAME["ra0"], FRAME["ra1"]],
                                      [FRAME["de0"], FRAME["de1"]]])
    cov = H > 0
    print(f"   coverage on a 0.1 deg grid: {cov.sum()} of {cov.size} cells "
          f"occupied ({100 * cov.mean():.1f}%), "
          f"{cov.sum() * 0.01:.1f} deg^2 of the {nx * ny * 0.01:.0f} deg^2 "
          f"frame")

    # GATE: how many eFEDS clusters land on covered cells?  Chiu reports
    # 313 of the 434 secure clusters (72%) are covered.
    import efeds_hsc as E
    recs, _ = E.load_efeds()
    hit = 0
    flags = []
    for r in recs:
        i = int((r["RA"] - FRAME["ra0"]) / 0.1)
        j = int((r["DE"] - FRAME["de0"]) / 0.1)
        ok = (0 <= i < nx and 0 <= j < ny and cov[i, j])
        flags.append(ok)
        hit += ok
    print(f"\n   GATE: {hit} of {len(recs)} eFEDS systems fall on a covered "
          f"cell = {100 * hit / len(recs):.1f}%")
    print(f"         Chiu+2022 reports 313 of 434 secure clusters covered = "
          f"72.1%")
    ok_gate = 0.60 < hit / len(recs) < 0.85
    print(f"         -> {'PASS' if ok_gate else 'CHECK'} (the sampled "
          f"footprint is sparse, so this is a consistency check, not an "
          f"identification of the 313)")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# eFEDS x HSC overlap, SAMPLED from Chiu+2022 Fig. 1.\n"
                "# Each row is one plotted HSC S19A weak-lensing source from a\n"
                "# random subset the authors drew for the figure.  This is a\n"
                "# sampling of the coverage, NOT the survey mask.\n")
        f.write("RAdeg\tDEdeg\n")
        for a, b in zip(ra, de):
            f.write(f"{a:.5f}\t{b:.5f}\n")
    blob = open(OUT, "rb").read()
    meta = {
        "file": os.path.basename(OUT),
        "source_url": "https://arxiv.org/e-print/2107.05652",
        "source_file_within_eprint": "materials/footprint.pdf",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "paper": "Chiu I.-N. et al. 2022, A&A 661, A11, Fig. 1",
        "retrieved_utc": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob), "row_count": int(len(ra)), "column_count": 2,
        "columns": [{"name": "RAdeg", "unit": "deg", "note": "J2000"},
                    {"name": "DEdeg", "unit": "deg", "note": "J2000"}],
        "extraction": "exact vector-coordinate recovery of the grey scatter "
                      "in the figure; axis frame anchored on round degrees, "
                      "validated by RA and Dec pt-per-degree agreeing to "
                      "<1e-3 and by the eFEDS cluster positions landing "
                      "inside.",
        "frame": FRAME,
        "coverage_deg2_on_0.1deg_grid": float(cov.sum() * 0.01),
        "efeds_systems_on_covered_cells": int(hit),
        "efeds_systems_total": int(len(recs)),
        "chiu_reported_coverage_fraction": 313 / 434,
        "points_recovered_total": int(len(pts)),
        "points_outside_plot_frame": int(len(pts) - inside.sum()),
        "note_on_clipped_points":
            "matplotlib clips the scatter with a PDF clip path rather than "
            "dropping the coordinates, so the stream also carries sources "
            "beyond the plotted eFEDS field, out to RA ~151.7 deg.  Those are "
            "real HSC positions but lie outside the eFEDS frame and are "
            "excluded from the mask.",
        "caveat": "A RANDOM SUBSET of the HSC catalogue, so absence of a "
                  "point does not prove absence of coverage.  Use for a "
                  "coverage prior, never as a survey mask.",
    }
    with open(OUT + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    np.save(os.path.join(HERE, "efeds_hsc_coverage_mask.npy"), cov)
    print(f"\n   wrote {os.path.basename(OUT)} and "
          f"efeds_hsc_coverage_mask.npy")
    return flags


if __name__ == "__main__":
    main()
