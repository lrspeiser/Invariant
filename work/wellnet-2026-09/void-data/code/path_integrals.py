"""
Compute the line-of-sight path integrals I_q, I_T, I_g for every
independently-distanced source whose sight line lies inside a void catalogue.

Outputs  ../path_integrals.csv  and  ../results.json.

Definitions (all in the fiducial comoving frame of common.py)
-------------------------------------------------------------
For a source at unit direction u and comoving endpoint r_end, with
x(l) = l u for l in [0, r_end]:

  I_q^ALG   = Integral_0^r_end  1_void^ALG(x(l)) dl                  [Mpc/h]
  I_q^den   = Integral_0^r_end  max(0, -delta(x(l))) dl              [Mpc/h]
  I_T^ALG   = (1/c^2) Integral 1_void^ALG * T_ij u^i u^j dl          [1/(Mpc/h)]
  I_g       = (1/c^2) Integral |grad phi| dl                         [dimensionless]

with T_ij = d_i d_j phi and phi the comoving peculiar potential in (km/s)^2.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (C_KMS, FootprintMask, R_MAX_VOID, Z_MAX_VOID,
                    comoving_distance, sky_to_unit, utc_now)
from density_field import DensityField
from ingest_sources import all_sources
from voids import (SphereUnionVoids, TriangleVoids, combine_caps_spheres,
                   load_v2, load_voidfinder, load_vast_sdss_holes,
                   union_length)

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(LANE, "..", "..", ".."))
DESIVAST = os.path.join(LANE, "raw", "desivast")
VASTSDSS = os.path.join(REPO, "work", "private", "open-gravity-void-source-v2")

DL_STEP = 2.0          # ray sampling step, Mpc/h
DX = 4.0               # density grid cell, Mpc/h
SMOOTH = 5.0           # Gaussian smoothing, Mpc/h
CAPS = ("NGC", "SGC")


# --------------------------------------------------------------------------
class EllipsoidVoids:
    """V2 voids approximated by their published inertia ellipsoids."""

    def __init__(self, voids, name="ZOBOV-ellipsoid"):
        self.name = name
        self.c = np.stack([voids["X"], voids["Y"], voids["Z"]], axis=1)
        ax = []
        for i in (1, 2, 3):
            ax.append(np.stack([voids[f"X{i}"], voids[f"Y{i}"], voids[f"Z{i}"]],
                               axis=1))
        self.axes = np.stack(ax, axis=1)          # (N, 3, 3) rows = axis vectors
        self.lens = np.linalg.norm(self.axes, axis=2)
        self.bs_r = self.lens.max(axis=1)
        self.c2 = np.einsum("ij,ij->i", self.c, self.c)

    def n_voids(self):
        return len(self.c)

    def ray_intervals(self, u, t_max):
        b = self.c @ u
        d2 = self.c2 - b * b
        cand = np.where((d2 < self.bs_r ** 2) & (b + self.bs_r > 0))[0]
        iv = []
        for k in cand:
            L = self.lens[k]
            if np.any(L <= 0):
                continue
            E = self.axes[k] / L[:, None]          # unit axes, rows
            M = (E.T / L ** 2) @ E                 # metric of the ellipsoid
            oc = -self.c[k]
            A = u @ M @ u
            B = 2.0 * (oc @ M @ u)
            Cc = oc @ M @ oc - 1.0
            disc = B * B - 4 * A * Cc
            if disc <= 0 or A <= 0:
                continue
            s = np.sqrt(disc)
            t1 = (-B - s) / (2 * A)
            t2 = (-B + s) / (2 * A)
            if t2 > 0:
                iv.append([t1, t2])
        return union_length(iv, 0.0, t_max)


# --------------------------------------------------------------------------
def load_cap_galaxies(cap):
    with fits.open(os.path.join(DESIVAST,
                                f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits")) as h:
        g = h[3].data
        m = np.asarray(g["OUT"]) == 0
        X = np.stack([np.asarray(g[k], float)[m] for k in ("X", "Y", "Z")], axis=1)
    return X


def build_all():
    t0 = time.time()
    meta = {"fiducial": {"Omega_m": 0.315, "h": 1.0, "frame": "comoving Mpc/h",
                         "matches_DESIVAST_header": True},
            "dl_step_mpch": DL_STEP, "grid_dx_mpch": DX,
            "gaussian_smoothing_mpch": SMOOTH,
            "built_utc": utc_now()}

    caps = {}
    for cap in CAPS:
        X = load_cap_galaxies(cap)
        r = np.linalg.norm(X, axis=1)
        ra = np.degrees(np.arctan2(X[:, 1], X[:, 0])) % 360.0
        dec = np.degrees(np.arcsin(X[:, 2] / r))
        fm = FootprintMask(ra, dec, pix_deg=0.5)
        print(f"[{cap}] galaxies {len(X)}  footprint {fm.area_deg2:.1f} deg^2",
              flush=True)
        df = DensityField(X, fm, dx=DX, smooth=SMOOTH, name=cap)
        print(f"[{cap}] density grid {df.shape} built ({time.time()-t0:.0f}s)",
              flush=True)
        caps[cap] = {"gal": X, "mask": fm, "field": df}

    # ---- void geometries -------------------------------------------------
    geo = {}
    for cap in CAPS:
        hdr, mx, ho = load_voidfinder(
            os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_VoidFinder_{cap}.fits"))
        geo[("VoidFinder", cap)] = SphereUnionVoids(ho, mx, f"VoidFinder_{cap}")
        for alg in ("VIDE", "REVOLVER"):
            h2, v2, tri = load_v2(
                os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_{alg}_{cap}.fits"))
            geo[(alg, cap)] = TriangleVoids(v2, tri, f"{alg}_{cap}")
        h3, v3, _ = load_v2(
            os.path.join(DESIVAST, f"DESIVAST_BGS_VOLLIM_V2_ZOBOV_{cap}.fits"))
        geo[("ZOBOV_ellipsoid", cap)] = EllipsoidVoids(v3, f"ZOBOV_{cap}")
    print(f"void geometries loaded ({time.time()-t0:.0f}s)", flush=True)

    # ---- independent SDSS VAST VoidFinder --------------------------------
    smx, sho = load_vast_sdss_holes(
        os.path.join(VASTSDSS, "VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes.txt"),
        os.path.join(VASTSDSS, "VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal.txt"))
    sdss = SphereUnionVoids(sho, smx, "SDSS_VAST_VoidFinder")
    sdss_mask = FootprintMask(smx["RA"], smx["DEC"], pix_deg=3.0)
    sdss_rmax = float(np.linalg.norm(
        np.stack([sho["X"], sho["Y"], sho["Z"]], 1), axis=1).max())
    meta["sdss_vast"] = {"n_holes": int(len(sho["X"])),
                         "n_maximal": int(len(smx["X"])),
                         "r_max_mpch": sdss_rmax,
                         "mask_area_deg2": float(sdss_mask.area_deg2)}
    print(f"SDSS VAST: {len(sho['X'])} holes, {len(smx['X'])} voids, "
          f"r_max {sdss_rmax:.0f} Mpc/h", flush=True)

    # ---- sources ---------------------------------------------------------
    src, cov, asym = all_sources()
    ra = src["ra"].to_numpy(float)
    dec = src["dec"].to_numpy(float)
    cap_of = np.array([""] * len(src), dtype=object)
    for cap in CAPS:
        inm = caps[cap]["mask"].contains(ra, dec)
        cap_of[inm & (cap_of == "")] = cap
    z = src["z_cmb"].to_numpy(float)
    keep = (cap_of != "") & (z > 0) & (z < Z_MAX_VOID)
    src = src[keep].reset_index(drop=True)
    cap_of = cap_of[keep]
    print(f"sources kept: {len(src)}  ({time.time()-t0:.0f}s)", flush=True)
    print(src.groupby("survey").size(), flush=True)

    U = sky_to_unit(src["ra"].to_numpy(float), src["dec"].to_numpy(float))
    r_end = comoving_distance(src["z_cmb"].to_numpy(float))

    algs = ["VoidFinder", "VIDE", "REVOLVER", "ZOBOV_ellipsoid"]
    n = len(src)
    res = {f"I_q_{a}": np.zeros(n) for a in algs}
    res["I_q_SDSS_VoidFinder"] = np.full(n, np.nan)
    for a in algs:
        res[f"I_T_{a}"] = np.zeros(n)
    res["I_q_den"] = np.zeros(n)
    res["I_T_den"] = np.zeros(n)
    res["I_g"] = np.zeros(n)
    res["path_covered_frac"] = np.zeros(n)
    res["mean_delta_los"] = np.zeros(n)
    res["odd_parity_voids"] = np.zeros(n, int)

    for i in range(n):
        cap = cap_of[i]
        u = U[i]
        te = float(r_end[i])
        fld = caps[cap]["field"]

        ivs = {}
        for a in algs:
            g = geo[(a, cap)]
            if isinstance(g, TriangleVoids):
                L, iv, nodd = g.ray_intervals(u, te)
                res["odd_parity_voids"][i] += nodd
            else:
                L, iv = g.ray_intervals(u, te)
            res[f"I_q_{a}"][i] = L
            ivs[a] = iv

        # SDSS VAST comparison where the sight line is in the SDSS footprint
        if sdss_mask.contains(np.array([src["ra"][i]]),
                              np.array([src["dec"][i]]))[0]:
            Ls, _ = sdss.ray_intervals(u, min(te, sdss_rmax))
            res["I_q_SDSS_VoidFinder"][i] = Ls

        # sampled quantities
        ell = np.arange(0.5 * DL_STEP, te, DL_STEP)
        if len(ell) == 0:
            continue
        pts = ell[:, None] * u[None, :]
        dl = DL_STEP
        dvals, ins1 = fld.sample(fld.delta, pts)
        gm, ins2 = fld.sample(fld.gradmag, pts)
        tll, ins3 = fld.tidal_ll(pts, u)
        insurv = fld.in_survey(pts)
        good = insurv & ins1
        res["path_covered_frac"][i] = good.sum() * dl / te
        if good.any():
            res["mean_delta_los"][i] = dvals[good].mean()
        res["I_q_den"][i] = np.sum(np.clip(-dvals[good], 0, None)) * dl
        res["I_g"][i] = np.sum(gm[good]) * dl / C_KMS ** 2
        res["I_T_den"][i] = np.sum(
            np.clip(-dvals[good], 0, None) * tll[good]) * dl / C_KMS ** 2
        # void-indicator weighted tidal term, per algorithm
        for a in algs:
            iv = ivs[a]
            if not iv:
                continue
            insidem = np.zeros(len(ell), bool)
            for (aa, bb) in iv:
                insidem |= (ell >= aa) & (ell <= bb)
            m = insidem & good
            if m.any():
                res[f"I_T_{a}"][i] = np.sum(tll[m]) * dl / C_KMS ** 2

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    out = src.copy()
    out["cap"] = cap_of
    out["r_end_mpch"] = r_end
    out["D_comov_indep_mpc"] = src["dl_mpc"].to_numpy(float) / (
        1.0 + src["z_helio"].to_numpy(float))
    for k, v in res.items():
        out[k] = v
    out.to_csv(os.path.join(LANE, "path_integrals.csv"), index=False)

    meta["n_sources"] = int(n)
    meta["by_survey"] = {k: int(v) for k, v in out.groupby("survey").size().items()}
    meta["pantheon_cov_asymmetry_max"] = float(asym)
    meta["void_counts"] = {}
    for a in algs:
        tot = sum(geo[(a, c)].n_voids() for c in CAPS)
        meta["void_counts"][a] = int(tot)
    with open(os.path.join(LANE, "build_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote path_integrals.csv", flush=True)
    return out, meta, caps, geo


if __name__ == "__main__":
    build_all()
