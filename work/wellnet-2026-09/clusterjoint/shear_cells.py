"""shear_cells.py -- lensing in radial x azimuthal cells, not radial profiles.

The programme's every previous lensing input was azimuthally averaged, which
makes any law that depends on direction indistinguishable from one that does not.
This module keeps the angle.

For each cluster in the OPEN half that also has Chandra coverage, DECADE sources
are binned into NRAD radial bins x NAZ azimuthal sectors, and the tangential
shear is measured in each cell with its own metacalibration response. A cell is
the unit of the machine-learning table: one row, X-ray features on one side,
lensing on the other.

WHY TANGENTIAL SHEAR PER SECTOR AND NOT A MASS MAP. A Kaiser-Squires
reconstruction would give kappa directly but needs the whole field, propagates
noise non-locally, and has a free additive constant (the mass-sheet
degeneracy). Tangential shear in a cell is local, its error is analytic, and it
is what the survey actually measures. The asymmetry lives in the variation
BETWEEN sectors at fixed radius, which is exactly what a profile averages away.

The estimator, response handling and MEASURED sign convention are inherited from
`clustershear/extract.py` unchanged, so cells and profiles are comparable.

    python shear_cells.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
for _p in (WELLNET, os.path.join(WELLNET, "clustershear")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import extract as EX                                          # noqa: E402
import cosmo as C                                             # noqa: E402

OUT = os.path.join(HERE, "shear_cells.jsonl")
OVERLAP = os.path.join(WELLNET, "clusterxray", "overlap_centres.json")

NRAD = 5
NAZ = 8
RMIN_H, RMAX_H = 0.2, 3.5          # h^-1 Mpc, same as the profiles
MIN_PER_CELL = 15


def cells_for(rec, raw):
    """Bin one cluster's sources into NRAD x NAZ cells."""
    head, a = EX.parse_csv(raw)
    if a.shape[0] < 200:
        return None, a.shape[0]
    col = {n: i for i, n in enumerate(head)}
    ra_s, de_s = a[:, col["ra"]], a[:, col["dec"]]
    e1, e2 = a[:, col["mcal_g_1_noshear"]], a[:, col["mcal_g_2_noshear"]]
    w = a[:, col["mcal_w_noshear"]]
    zs = a[:, col["dnf_z"]]
    e1p, e1m = a[:, col["mcal_g_1_1p"]], a[:, col["mcal_g_1_1m"]]
    e2p, e2m = a[:, col["mcal_g_2_2p"]], a[:, col["mcal_g_2_2m"]]

    z_l = rec["z"]
    bg = zs > z_l + EX.DZ_BG
    et, ex, th = EX.tangential(ra_s, de_s, rec["ra"], rec["dec"], e1, e2)

    # position angle east of north, for the azimuthal binning
    d2r = math.pi / 180.0
    dra = (ra_s - rec["ra"]) * math.cos(rec["dec"] * d2r)
    dde = de_s - rec["dec"]
    phi = (np.degrees(np.arctan2(dra, dde)) + 360.0) % 360.0

    DA = float(C.d_ang(z_l))
    Rphys = th * DA / C.MPC
    redges = np.geomspace(RMIN_H / C.H_LITTLE, RMAX_H / C.H_LITTLE, NRAD + 1)
    aedges = np.linspace(0.0, 360.0, NAZ + 1)

    rows = []
    for i in range(NRAD):
        inr = bg & (Rphys >= redges[i]) & (Rphys < redges[i + 1])
        for j in range(NAZ):
            m = inr & (phi >= aedges[j]) & (phi < aedges[j + 1])
            n = int(m.sum())
            cell = dict(i_rad=i, j_az=j, n=n,
                        R_lo=float(redges[i]), R_hi=float(redges[i + 1]),
                        pa_lo=float(aedges[j]), pa_hi=float(aedges[j + 1]))
            if n < MIN_PER_CELL:
                cell.update(R=None, pa=None, gt=None, gx=None, err=None, R_resp=None)
                rows.append(cell)
                continue
            ww = w[m]
            sw = float(ww.sum())
            R11 = float(np.sum(ww * (e1p[m] - e1m[m])) / sw / (2 * EX.DGAMMA))
            R22 = float(np.sum(ww * (e2p[m] - e2m[m])) / sw / (2 * EX.DGAMMA))
            Rbar = 0.5 * (R11 + R22)
            if not np.isfinite(Rbar) or abs(Rbar) < 1e-3:
                cell.update(R=None, pa=None, gt=None, gx=None, err=None, R_resp=Rbar)
                rows.append(cell)
                continue
            cell.update(
                R=float(np.sum(ww * Rphys[m]) / sw),
                pa=float(np.sum(ww * phi[m]) / sw),
                gt=float(np.sum(ww * et[m]) / sw / Rbar),
                gx=float(np.sum(ww * ex[m]) / sw / Rbar),
                err=float(math.sqrt(np.sum(ww ** 2 * et[m] ** 2)) / sw / abs(Rbar)),
                R_resp=Rbar)
            rows.append(cell)
    return rows, a.shape[0]


def main():
    from holdout import loader                                # noqa: E402
    loader.verify()
    centres = json.load(io.open(OVERLAP, encoding="utf-8"))
    names = [v[6] for v in centres.values()]
    loader.assert_not_sealed(names, "shear_cells targets")
    print("clusters with both channels: %d (sealed %d untouched)"
          % (len(centres), len(loader.sealed_names())), flush=True)

    done = set()
    if os.path.exists(OUT):
        for line in io.open(OUT, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["key"])

    fh = io.open(OUT, "a", newline="\n", encoding="utf-8")
    t0 = time.time()
    n_ok = 0
    for k, (key, v) in enumerate(sorted(centres.items()), 1):
        if key in done:
            continue
        ra, dec, z, kt, cts, shear, name = v
        rec = dict(name=name, ra=ra, dec=dec, z=z)
        rad = EX.cone_radius_deg(z)
        q = ("SELECT %s FROM delve_dr3.decade_shear WHERE "
             "'t' = q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) AND %s"
             % (EX.COLS, ra, dec, rad, EX.SEL))
        out = dict(key=key, name=name, ra=ra, dec=dec, z=z, kt=kt,
                   cts500=cts, nrad=NRAD, naz=NAZ, cone_deg=round(rad, 4))
        try:
            rows, nsrc = cells_for(rec, EX.tap(q))
            out["n_sources"] = nsrc
            out["cells"] = rows
            out["error"] = None
            if rows:
                n_ok += 1
        except Exception as exc:                              # noqa: BLE001
            out["n_sources"] = None
            out["cells"] = None
            out["error"] = str(exc)[:200]
        fh.write(json.dumps(out) + "\n")
        fh.flush()
        if k % 10 == 0:
            print("  %d/%d  %.0f min" % (k, len(centres), (time.time() - t0) / 60),
                  flush=True)
    fh.close()
    print("done: %d clusters with cells in %.0f min" % (n_ok, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
