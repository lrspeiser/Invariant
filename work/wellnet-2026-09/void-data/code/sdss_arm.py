"""
Second, independent arm: I_q from the SDSS DR7 VAST VoidFinder catalogue.

This is NOT a DESI product and shares no pipeline with DESIVAST.  It reaches
only z = 0.1125 (r = 328.3 Mpc/h) but covers ~7000 deg^2, so it contains four
times as many independently-distanced sources as the DESI DR1 BGS footprint.

Only I_q is computed here.  I_T and I_g need a density field, and this lane has
no SDSS galaxy catalogue -- saying so is better than substituting a proxy.

Writes ../path_integrals_sdss.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (FootprintMask, comoving_distance, sky_to_unit, utc_now)
from ingest_sources import all_sources
from voids import SphereUnionVoids, load_vast_sdss_holes

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(LANE, "..", "..", ".."))
VASTSDSS = os.path.join(REPO, "work", "private", "open-gravity-void-source-v2")

MIN_REND = 100.0


def main():
    t0 = time.time()
    mx, ho = load_vast_sdss_holes(
        os.path.join(VASTSDSS, "VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes.txt"),
        os.path.join(VASTSDSS, "VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal.txt"))
    assert len(ho["X"]) == 39735, f"SDSS holes {len(ho['X'])} != 39735"
    assert len(mx["X"]) == 1163, f"SDSS maximals {len(mx['X'])} != 1163"
    geo = SphereUnionVoids(ho, mx, "SDSS_VAST_VoidFinder")

    hx, hy, hz = ho["X"], ho["Y"], ho["Z"]
    hr = np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
    mask = FootprintMask(np.degrees(np.arctan2(hy, hx)) % 360.0,
                         np.degrees(np.arcsin(hz / hr)), pix_deg=2.0)
    r_max = float(hr.max())
    z_max = float(np.interp(r_max, comoving_distance(np.linspace(0, .3, 30001)),
                            np.linspace(0, .3, 30001)))
    print(f"SDSS VAST: {len(ho['X'])} holes, {geo.n_voids()} voids, "
          f"mask {mask.area_deg2:.0f} deg^2, r_max {r_max:.1f} Mpc/h "
          f"(z_max {z_max:.4f})", flush=True)

    src, _, _ = all_sources()
    ins = mask.contains(src["ra"].to_numpy(float), src["dec"].to_numpy(float))
    z = src["z_cmb"].to_numpy(float)
    keep = ins & (z > 0) & (z < z_max)
    d = src[keep].reset_index(drop=True)
    r_end = comoving_distance(d["z_cmb"].to_numpy(float))
    U = sky_to_unit(d["ra"].to_numpy(float), d["dec"].to_numpy(float))
    print(f"sources in SDSS footprint: {len(d)}", flush=True)

    Iq = np.zeros(len(d))
    for i in range(len(d)):
        L, _ = geo.ray_intervals(U[i], float(r_end[i]))
        Iq[i] = L
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(d)} ({time.time()-t0:.0f}s)", flush=True)
    d["r_end_mpch"] = r_end
    d["I_q_SDSS_VoidFinder"] = Iq

    # footprint-averaged expectation and transverse residual
    radii = np.arange(50.0, r_max + 1.0, 25.0)
    rng = np.random.default_rng(202)
    dirs = []
    while len(dirs) < 300:
        ra = 360.0 * rng.random(6000)
        dec = np.degrees(np.arcsin(2.0 * rng.random(6000) - 1.0))
        ok = mask.contains(ra, dec)
        for a, b in zip(ra[ok], dec[ok]):
            dirs.append((a, b))
            if len(dirs) >= 300:
                break
    Ud = sky_to_unit(np.array([x[0] for x in dirs]),
                     np.array([x[1] for x in dirs]))
    L = np.zeros((len(Ud), len(radii)))
    for i in range(len(Ud)):
        _, iv = geo.ray_intervals(Ud[i], r_max)
        if not iv:
            continue
        a = np.array([x[0] for x in iv])
        b = np.array([x[1] for x in iv])
        for j, r in enumerate(radii):
            L[i, j] = np.clip(np.minimum(b, r) - a, 0, None).sum()
    mu = L.mean(0)
    sd = L.std(0)
    d["expI_q_SDSS"] = np.interp(r_end, radii, mu)
    d["sdI_q_SDSS"] = np.interp(r_end, radii, sd)
    d["dI_q_SDSS"] = d["I_q_SDSS_VoidFinder"] - d["expI_q_SDSS"]

    sel = d["r_end_mpch"] >= MIN_REND
    dd = d[sel]
    res = {
        "generated_utc": utc_now(),
        "catalogue": "SDSS DR7 VAST VoidFinder (Zenodo 11043278, Planck2018)",
        "n_holes": int(len(ho["X"])), "n_voids": int(geo.n_voids()),
        "mask_area_deg2": float(mask.area_deg2),
        "r_max_mpch": r_max, "z_max": z_max,
        "n_sources_in_footprint": int(len(d)),
        "n_after_r_cut": int(len(dd)),
        "by_survey": {k: int(v) for k, v in dd.groupby("survey").size().items()},
        "I_q_mean_mpch": float(dd["I_q_SDSS_VoidFinder"].mean()),
        "I_q_std_mpch": float(dd["I_q_SDSS_VoidFinder"].std()),
        "corr_I_q_with_D": float(np.corrcoef(dd["I_q_SDSS_VoidFinder"],
                                             dd["r_end_mpch"])[0, 1]),
        "transverse_dI_q_std_mpch": float(dd["dI_q_SDSS"].std()),
        "transverse_dI_q_p5_p95_mpch": [float(np.percentile(dd["dI_q_SDSS"], 5)),
                                        float(np.percentile(dd["dI_q_SDSS"], 95))],
        "void_volume_fraction_along_rays": float(
            (mu[-1] / radii[-1])),
    }
    d.to_csv(os.path.join(LANE, "path_integrals_sdss.csv"), index=False)
    with open(os.path.join(LANE, "results_sdss.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
