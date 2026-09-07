"""extract.py -- tangential shear profiles for the OPEN half of the gold-cluster pool.

This lane DOES open shear -- that is its job, and it is why it is a separate lane
from `goldcluster/`, whose guard forbids exactly this. What it must never do is
open shear for a cluster in the SEALED half. That is enforced twice:

    1. the target list comes from `holdout.loader.open_pool()`, which cannot
       return a sealed cluster;
    2. every cone query passes through `guard.check_target()`, which re-checks
       the identity against the sealed set immediately before the archive call.

Belt and braces, because this is the one mistake that cannot be undone: a single
query against a sealed cluster spends the programme's only confirmation set.

THE ESTIMATOR is Run BM's, from `efeds-hsc/acquire_decade.py`, reused rather
than rewritten so the two lanes' numbers are comparable. In particular the SIGN
CONVENTION IS MEASURED, NOT ASSUMED: that lane ran all four axis-sign
combinations over 33,775 background sources and found the DECADE/DES ellipticity
basis has its first axis pointing WEST, giving <g_t> = +0.0108 where the naive
convention gives -0.0021. The amplitude agreed with Chiu+2022's independent HSC
profile of the same field. `test_lane.py` re-checks the convention on the stack
built here, so a silent basis flip upstream would fail the run rather than
quietly invert every profile.

    python extract.py            # resumable; one JSON line per cluster

Output is the PROFILE, not the sources: ten log-spaced bins of g_t, g_x, their
shape-noise errors, the per-bin metacalibration response and the lensing
efficiency. Per-source shapes are never written to disk.
"""
from __future__ import annotations

import io
import json
import math
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

import cosmo as C
import guard

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "profiles.jsonl")

TAP = "https://datalab.noirlab.edu/tap/sync"
UA = "gravity-clustershear/1.0 (research)"
CTX = ssl.create_default_context()

DGAMMA = 0.01                       # metacal step, DES/DECADE convention
NBIN = 10
RMIN_H, RMAX_H = 0.2, 3.5           # h^-1 Mpc, as Chiu+2022 and the eFEDS lane
THETA_CAP_DEG = 1.5
DZ_BG = 0.20                        # background margin
MIN_SOURCES = 200                   # below this a profile is not worth writing
MIN_PER_BIN = 20

COLS = ("ra,dec,mcal_g_1_noshear,mcal_g_2_noshear,mcal_w_noshear,dnf_z,"
        "mcal_g_1_1p,mcal_g_1_1m,mcal_g_2_2p,mcal_g_2_2m")
SEL = ("mcal_flags = 0 AND flags_foreground = 0 AND flags_footprint = 1 "
       "AND mcal_sel_noshear > 0 AND dnf_z > 0 AND dnf_z < 3")


def tap(query, timeout=600, retries=3):
    body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                   "FORMAT": "csv", "QUERY": query}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(TAP, data=body,
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                raw = r.read()
            if raw.lstrip().startswith(b"<?xml"):
                raise RuntimeError("TAP ERROR under HTTP 200: "
                                   + raw[:300].decode("utf-8", "replace"))
            return raw
        except Exception:                                   # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


def parse_csv(raw):
    txt = raw.decode("utf-8")
    lines = txt.strip().split("\n")
    head = lines[0].split(",")
    if len(lines) < 2:
        return head, np.zeros((0, len(head)))
    arr = np.array([ln.split(",") for ln in lines[1:]], dtype=float)
    return head, (arr[None, :] if arr.ndim == 1 else arr)


def tangential(ra_s, de_s, ra_c, de_c, e1, e2):
    """Tangential and cross components about (ra_c, de_c).

    The x axis points WEST -- measured by Run BM, see the module docstring.
    """
    d2r = math.pi / 180.0
    dra = (ra_s - ra_c) * math.cos(de_c * d2r)          # east-positive
    dde = de_s - de_c
    phi = np.arctan2(dde, -dra)                          # west-positive x axis
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    return -(e1 * c2 + e2 * s2), (e1 * s2 - e2 * c2), np.hypot(dra, dde) * d2r


def cone_radius_deg(z):
    """Angular radius that contains RMAX_H, capped."""
    r_max_m = (RMAX_H / C.H_LITTLE) * C.MPC
    theta = r_max_m / C.d_ang(z)                          # radians
    return min(math.degrees(theta) * 1.05, THETA_CAP_DEG)


def profile_for(rec, raw):
    """Bin one cluster's sources into the ten declared radial bins."""
    head, a = parse_csv(raw)
    if a.shape[0] < MIN_SOURCES:
        return None, a.shape[0]
    col = {n: i for i, n in enumerate(head)}
    ra_s, de_s = a[:, col["ra"]], a[:, col["dec"]]
    e1, e2 = a[:, col["mcal_g_1_noshear"]], a[:, col["mcal_g_2_noshear"]]
    w = a[:, col["mcal_w_noshear"]]
    zs = a[:, col["dnf_z"]]
    e1p, e1m = a[:, col["mcal_g_1_1p"]], a[:, col["mcal_g_1_1m"]]
    e2p, e2m = a[:, col["mcal_g_2_2p"]], a[:, col["mcal_g_2_2m"]]

    z_l = rec["z"]
    bg = zs > z_l + DZ_BG
    et, ex, th = tangential(ra_s, de_s, rec["ra"], rec["dec"], e1, e2)
    DA = float(C.d_ang(z_l))
    Rphys = th * DA / C.MPC                               # proper Mpc
    edges = np.geomspace(RMIN_H / C.H_LITTLE, RMAX_H / C.H_LITTLE, NBIN + 1)
    dcl = float(C.d_com(z_l))
    beta_s = np.where(zs > z_l,
                      (C.d_com(zs) - dcl) / (1.0 + zs)
                      / np.maximum(C.d_ang(zs), 1.0), 0.0)

    rows = []
    for i in range(NBIN):
        m = bg & (Rphys >= edges[i]) & (Rphys < edges[i + 1])
        n = int(m.sum())
        if n < MIN_PER_BIN:
            rows.append(dict(n=n, R=math.sqrt(edges[i] * edges[i + 1]),
                             gt=None, gx=None, err=None, R11=None, R22=None,
                             beta=None, theta_arcmin=None))
            continue
        ww = w[m]
        sw = float(ww.sum())
        R11 = float(np.sum(ww * (e1p[m] - e1m[m])) / sw / (2 * DGAMMA))
        R22 = float(np.sum(ww * (e2p[m] - e2m[m])) / sw / (2 * DGAMMA))
        Rbar = 0.5 * (R11 + R22)
        if not np.isfinite(Rbar) or abs(Rbar) < 1e-3:
            rows.append(dict(n=n, R=math.sqrt(edges[i] * edges[i + 1]),
                             gt=None, gx=None, err=None, R11=R11, R22=R22,
                             beta=None, theta_arcmin=None))
            continue
        rows.append(dict(
            n=n,
            R=float(np.sum(ww * Rphys[m]) / sw),
            theta_arcmin=float(np.sum(ww * th[m]) / sw) * 180 * 60 / math.pi,
            gt=float(np.sum(ww * et[m]) / sw / Rbar),
            gx=float(np.sum(ww * ex[m]) / sw / Rbar),
            err=float(math.sqrt(np.sum(ww ** 2 * et[m] ** 2)) / sw / abs(Rbar)),
            R11=R11, R22=R22,
            beta=float(np.sum(ww * beta_s[m]) / sw)))
    return rows, a.shape[0]


def main():
    from holdout import loader                      # noqa: E402

    loader.verify()
    targets = loader.open_pool()
    guard.assert_all_open([c["name"] for c in targets], "extract targets")
    print("open-half targets: %d (sealed half untouched: %d)"
          % (len(targets), len(loader.sealed_names())), flush=True)

    done = set()
    if os.path.exists(OUT):
        for line in io.open(OUT, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["name"])
    todo = [c for c in targets if c["name"] not in done]
    print("already extracted=%d  to do=%d" % (len(done), len(todo)), flush=True)

    fh = io.open(OUT, "a", newline="\n", encoding="utf-8")
    t0 = time.time()
    nok = 0
    for i, c in enumerate(todo, 1):
        guard.check_target(c["name"])          # immediately before the query
        rad = cone_radius_deg(c["z"])
        q = ("SELECT %s FROM delve_dr3.decade_shear WHERE "
             "'t' = q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) AND %s"
             % (COLS, c["ra"], c["dec"], rad, SEL))
        rec = dict(name=c["name"], ra=c["ra"], dec=c["dec"], z=c["z"],
                   zType=c["zType"], cts500=c["cts500"], kt=c["kt"],
                   cone_radius_deg=round(rad, 4))
        try:
            raw = tap(q)
            rows, nsrc = profile_for(rec, raw)
            rec["n_sources"] = nsrc
            rec["profile"] = rows
            rec["error"] = None
            if rows:
                nok += 1
        except Exception as exc:                            # noqa: BLE001
            rec["n_sources"] = None
            rec["profile"] = None
            rec["error"] = str(exc)[:200]
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if i % 20 == 0:
            print("  %d/%d  %d with profiles  %.0f min"
                  % (i, len(todo), nok, (time.time() - t0) / 60), flush=True)
    fh.close()
    print("done: %d profiles from %d clusters in %.0f min"
          % (nok, len(todo), (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    sys.exit(main())
