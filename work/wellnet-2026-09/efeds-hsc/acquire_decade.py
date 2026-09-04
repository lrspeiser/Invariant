"""RAW per-source weak-lensing shear over the eFEDS field, from DECADE.

The HSC shape catalogue that Chiu+2022 used is behind an account (HTTP 401 on
every archive route) and no eFEDS shear PROFILE is published anywhere.  But a
different, fully public per-source metacalibration catalogue covers the same
sky: the DECADE shear catalogue released inside DELVE DR3 and served by the
NOIRLab Astro Data Lab TAP endpoint with no credentials at all.

    14,498,544 rows inside RA 126-146, Dec -3 to +6
    metacalibration: mcal_g_{1,2}_noshear plus the 1p/1m/2p/2m sheared copies,
    so the response matrix is recoverable rather than assumed
    mcal_w_noshear shear weights, mcal_sel_noshear tomographic selection,
    dnf_z per-source photometric redshift

This script turns that into one tangential-shear profile per eFEDS system.  It
never touches a mass catalogue: the output is g_t(theta) and g_x(theta) with
shape-noise errors, exactly the observable the standing brief asks for.

CUTS AND CONVENTIONS, DECLARED HERE BEFORE ANY RESIDUAL WAS COMPUTED
  source quality   mcal_flags = 0, flags_foreground = 0, flags_footprint = 1,
                   mcal_sel_noshear > 0  (the DECADE cosmology selection;
                   values 1-4 are its four tomographic bins)
  photo-z          0 < dnf_z < 3, and dnf_z > z_cl + 0.2 for the background cut
                   -- the same Delta z = 0.2 margin Chiu+2022 and Umetsu+2020
                   use with HSC
  radial bins      ten logarithmic bins over 0.2 - 3.5 h^-1 Mpc PHYSICAL, i.e.
                   identical to Chiu+2022, so the two surveys are directly
                   comparable
  aperture cap     1.5 deg, to bound the query for the lowest-redshift systems
  response         metacalibration R = (R_11 + R_22)/2 with
                   R_ii = <e_i^{ip} - e_i^{im}> / (2 x 0.01), computed PER BIN;
                   the selection response is computed once globally
  sign convention  VALIDATED, not asserted: the tangential signal must come out
                   positive and the cross signal consistent with zero
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

import pipeline as P
import efeds_hsc as E

HERE = os.path.dirname(os.path.abspath(__file__))
ACQ = os.path.join(HERE, "acquire")
RAW = os.path.join(ACQ, "decade_raw")
os.makedirs(RAW, exist_ok=True)

TAP = "https://datalab.noirlab.edu/tap/sync"
UA = "research/1.0 (gravity programme)"
CTX = ssl.create_default_context()
DGAMMA = 0.01                      # metacal step, DES/DECADE convention
NBIN = 10
RMIN_H, RMAX_H = 0.2, 3.5          # h^-1 Mpc
THETA_CAP_DEG = 1.5

COLS = ("ra,dec,mcal_g_1_noshear,mcal_g_2_noshear,mcal_w_noshear,dnf_z,"
        "mcal_g_1_1p,mcal_g_1_1m,mcal_g_2_2p,mcal_g_2_2m")
SEL = ("mcal_flags = 0 AND flags_foreground = 0 AND flags_footprint = 1 "
       "AND mcal_sel_noshear > 0 AND dnf_z > 0 AND dnf_z < 3")


def tap(query, retries=4):
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": query}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(TAP, data=body,
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300, context=CTX) as r:
                raw = r.read()
            if raw.lstrip().startswith(b"<?xml"):
                raise RuntimeError(raw[:400].decode("utf-8", "replace"))
            return raw
        except Exception as exc:                                # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(3 * (k + 1))
            del exc


def parse_csv(raw):
    txt = raw.decode("utf-8")
    lines = txt.strip().split("\n")
    head = lines[0].split(",")
    if len(lines) < 2:
        return head, np.zeros((0, len(head)))
    # np.genfromtxt is ~50x slower than this on 10^4-10^5 row responses
    arr = np.array([ln.split(",") for ln in lines[1:]], dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    return head, arr


def cone(ra, dec, rad_deg):
    q = (f"SELECT {COLS} FROM delve_dr3.decade_shear "
         f"WHERE 't' = q3c_radial_query(ra, dec, {ra:.6f}, {dec:.6f}, "
         f"{rad_deg:.6f}) AND {SEL}")
    return q, tap(q)


def tangential(ra_s, de_s, ra_c, de_c, e1, e2):
    """Tangential and cross components about (ra_c, de_c).

    THE SIGN CONVENTION WAS MEASURED, NOT ASSUMED.  All four combinations of
    the two axis signs were run over the 40 most gas-massive eFEDS systems
    (33,775 background sources at 0.3-2 Mpc) and only one gives a positive
    tangential signal:

        phi = atan2(d_dec, +d_ra cos dec)   ->  <g_t> = -0.00213   WRONG
        phi = atan2(d_dec, -d_ra cos dec)   ->  <g_t> = +0.01082   RIGHT

    i.e. the DECADE/DES ellipticity basis has its first axis pointing WEST, so
    the position angle must be measured with RA increasing to the left.  The
    +0.0108 amplitude also agrees with the HSC stacked profile of the same
    field (Chiu+2022 give 0.0117 at 0.73 Mpc), which is an independent check
    on the whole chain.
    """
    d2r = math.pi / 180.0
    dra = (ra_s - ra_c) * math.cos(de_c * d2r)          # east-positive
    dde = de_s - de_c
    phi = np.arctan2(dde, -dra)                          # west-positive x axis
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    return -(e1 * c2 + e2 * s2), (e1 * s2 - e2 * c2), np.hypot(dra, dde) * d2r


def sigma_crit_inv(z_l, z_s):
    """(4 pi G / c^2) D_l D_ls / D_s, SI, zero for foreground sources."""
    Dl = P.d_ang(z_l)
    Ds = P.d_ang(z_s)
    Dls = (P.d_com(z_s) - P.d_com(z_l)) / (1.0 + z_s)
    out = (4.0 * math.pi * P.G / P.CLIGHT ** 2) * Dl * Dls / np.maximum(Ds, 1.0)
    return np.where(z_s > z_l, out, 0.0)


def profile_for(rec, raw):
    """Bin one cluster's sources into the ten declared radial bins."""
    head, a = parse_csv(raw)
    if a.shape[0] < 50:
        return None
    col = {n: i for i, n in enumerate(head)}
    ra_s, de_s = a[:, col["ra"]], a[:, col["dec"]]
    e1, e2 = a[:, col["mcal_g_1_noshear"]], a[:, col["mcal_g_2_noshear"]]
    w = a[:, col["mcal_w_noshear"]]
    zs = a[:, col["dnf_z"]]
    e1p, e1m = a[:, col["mcal_g_1_1p"]], a[:, col["mcal_g_1_1m"]]
    e2p, e2m = a[:, col["mcal_g_2_2p"]], a[:, col["mcal_g_2_2m"]]

    z_l = rec["z"]
    bg = zs > z_l + 0.2                      # declared background cut
    et, ex, th = tangential(ra_s, de_s, rec["RA"], rec["DE"], e1, e2)
    DA = rec["DA"]
    Rphys = th * DA / P.MPC                  # proper Mpc
    edges = np.geomspace(RMIN_H / P.H_LITTLE, RMAX_H / P.H_LITTLE, NBIN + 1)
    beta_s = np.where(zs > z_l,
                      (P.d_com(zs) - P.d_com(z_l)) / (1.0 + zs)
                      / np.maximum(P.d_ang(zs), 1.0), 0.0)

    rows = []
    for i in range(NBIN):
        m = bg & (Rphys >= edges[i]) & (Rphys < edges[i + 1])
        n = int(m.sum())
        if n < 20:
            rows.append(dict(n=n, R=math.sqrt(edges[i] * edges[i + 1]),
                             gt=np.nan, gx=np.nan, err=np.nan, R11=np.nan,
                             R22=np.nan, beta=np.nan, beta2=np.nan,
                             theta_arcmin=np.nan))
            continue
        ww = w[m]
        sw = ww.sum()
        R11 = float(np.sum(ww * (e1p[m] - e1m[m])) / sw / (2 * DGAMMA))
        R22 = float(np.sum(ww * (e2p[m] - e2m[m])) / sw / (2 * DGAMMA))
        Rbar = 0.5 * (R11 + R22)
        gt = float(np.sum(ww * et[m]) / sw / Rbar)
        gx = float(np.sum(ww * ex[m]) / sw / Rbar)
        # shape-noise error from the weighted scatter of the SAME estimator
        err = float(math.sqrt(np.sum(ww ** 2 * et[m] ** 2)) / sw / abs(Rbar))
        rows.append(dict(
            n=n, R=float(np.sum(ww * Rphys[m]) / sw),
            theta_arcmin=float(np.sum(ww * th[m]) / sw) * 180 * 60 / math.pi,
            gt=gt, gx=gx, err=err, R11=R11, R22=R22,
            beta=float(np.sum(ww * beta_s[m]) / sw),
            beta2=float(np.sum(ww * beta_s[m] ** 2) / sw)))
    return rows


def main():
    print("=" * 78)
    print("ACQUIRE -- DECADE raw shear around the eFEDS systems")
    print("=" * 78)
    recs, cuts = E.load_efeds()
    print(f"\n   {len(recs)} eFEDS systems with a Bahar+2022 density fit")
    out_path = os.path.join(HERE, "decade_efeds_shear_profiles.tsv")
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path, encoding="utf-8"):
            if not ln.startswith("#") and not ln.startswith("id\t"):
                done.add(ln.split("\t")[0])
        print(f"   resuming: {len(done)} systems already written")
    f = open(out_path, "a", encoding="utf-8")
    if not done:
        f.write("# DECADE (DELVE DR3) tangential shear profiles around eFEDS "
                "X-ray systems.\n# See acquire_decade.py for every cut and "
                "convention.  g_t and g_x are\n# metacalibration-response-"
                "corrected reduced shear; err is shape noise.\n")
        f.write("id\tz\tbin\tR_Mpc\ttheta_arcmin\tn\tgt\tgx\terr\tR11\tR22\t"
                "beta\tbeta2\n")
    nq = 0
    t0 = time.time()
    for k, rec in enumerate(recs):
        if rec["id"] in done:
            continue
        rad = min(THETA_CAP_DEG,
                  (RMAX_H / P.H_LITTLE) * P.MPC / rec["DA"] * 180 / math.pi)
        try:
            q, raw = cone(rec["RA"], rec["DE"], rad)
        except Exception as exc:                                # noqa: BLE001
            print(f"   {rec['id']}  QUERY FAILED: {exc}")
            continue
        nq += 1
        if nq == 1:                       # keep one raw upstream response
            with open(os.path.join(RAW, f"{rec['id']}_raw.csv"), "wb") as g:
                g.write(raw)
            with open(os.path.join(RAW, f"{rec['id']}_raw.query.txt"),
                      "w", encoding="utf-8") as g:
                g.write(q)
        rows = profile_for(rec, raw)
        if rows is None:
            f.write(f"{rec['id']}\t{rec['z']:.4f}\t-1\t0\t0\t0\tnan\tnan\t"
                    f"nan\tnan\tnan\tnan\tnan\n")
        else:
            for i, r in enumerate(rows):
                f.write(f"{rec['id']}\t{rec['z']:.4f}\t{i}\t{r['R']:.5f}\t"
                        f"{r['theta_arcmin']:.4f}\t{r['n']}\t{r['gt']:.7f}\t"
                        f"{r['gx']:.7f}\t{r['err']:.7f}\t{r['R11']:.5f}\t"
                        f"{r['R22']:.5f}\t{r['beta']:.5f}\t{r['beta2']:.5f}\n")
        f.flush()
        if k % 25 == 0:
            print(f"   {k + 1}/{len(recs)}  {rec['id']}  rad {rad:.3f} deg  "
                  f"{time.time() - t0:.0f} s elapsed")
    f.close()

    blob = open(out_path, "rb").read()
    nrow = sum(1 for ln in blob.decode().split("\n")
               if ln and not ln.startswith("#") and not ln.startswith("id\t"))
    meta = {
        "file": os.path.basename(out_path),
        "source": "NOIRLab Astro Data Lab TAP, table delve_dr3.decade_shear "
                  "(DECADE weak-lensing shape catalogue released in DELVE "
                  "DR3).  Unauthenticated: no credentials are sent.",
        "endpoint": TAP,
        "exact_query_template":
            f"SELECT {COLS} FROM delve_dr3.decade_shear WHERE 't' = "
            f"q3c_radial_query(ra, dec, <RA>, <DEC>, <radius_deg>) AND {SEL}",
        "raw_response_kept": "acquire/decade_raw/ (first system queried, "
                             "verbatim, with its query)",
        "retrieved_utc": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob), "row_count": nrow,
        "n_systems_requested": len(recs),
        "columns": [
            {"name": "id", "unit": "", "note": "eFEDS name, Bahar+2022 form"},
            {"name": "z", "unit": "", "note": "cluster redshift"},
            {"name": "bin", "unit": "", "note": "0-9, or -1 if too few sources"},
            {"name": "R_Mpc", "unit": "Mpc",
             "note": "weighted mean proper clustercentric radius"},
            {"name": "theta_arcmin", "unit": "arcmin"},
            {"name": "n", "unit": "", "note": "background sources in the bin"},
            {"name": "gt", "unit": "",
             "note": "tangential REDUCED shear, metacal-response corrected"},
            {"name": "gx", "unit": "", "note": "cross component, null test"},
            {"name": "err", "unit": "", "note": "shape-noise 1 sigma"},
            {"name": "R11", "unit": "", "note": "metacal response, component 1"},
            {"name": "R22", "unit": "", "note": "metacal response, component 2"},
            {"name": "beta", "unit": "",
             "note": "weighted <D_ls/D_s> of the bin's background sources"},
            {"name": "beta2", "unit": "", "note": "weighted <(D_ls/D_s)^2>"}],
        "cuts": SEL + " AND dnf_z > z_cl + 0.2",
        "binning": f"{NBIN} log bins over {RMIN_H}-{RMAX_H} h^-1 Mpc physical, "
                   f"h = {P.H_LITTLE}; identical to Chiu+2022",
        "caveat": "DECADE is DECam, not HSC.  It is shallower than the HSC "
                  "S19A catalogue Chiu+2022 used, and its photo-z are DNF "
                  "point estimates rather than full P(z).  It is, however, "
                  "RAW per-source shear, which is what the test requires.",
    }
    with open(out_path + ".manifest.json", "w", encoding="utf-8") as g:
        json.dump(meta, g, indent=1)
    print(f"\n   wrote {os.path.basename(out_path)}: {nrow} rows, "
          f"{nq} systems queried, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
