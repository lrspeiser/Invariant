"""Job 1 -- the void path-length map, and every template the analysis will use.

NOTHING IN THIS FILE READS A CMB TEMPERATURE.  It builds the regressor and the
nuisance templates on the Planck HEALPix grid so that the Stage 4 certificate can
be issued against the real geometry before any cross-correlation value exists.

GEOMETRY, declared here and not changed afterwards
--------------------------------------------------
  grid        HEALPix nside 64 (0.839 deg^2 pixels), NESTED, GALACTIC -- the
              native Planck grid, so no CMB pixel is ever interpolated
  catalogue   SDSS DR7 / NSA v1.0.1 VAST VoidFinder, Planck2018 comoving
              (Zenodo 11043278; Douglass, Veyrat & BenZvi 2023, ApJS 265 7)
              39,735 holes, 1,163 voids, union-of-spheres geometry, EXACT
              ray-sphere intersection (no gridding, no proxy)
  range       chi in [0, 332.386] Mpc/h -- the catalogue's OWN declared distance
              limit, read from NSA_main_mask.pickle.  z <= 0.1125.
  footprint   NSA main survey mask (1 deg RA/Dec cells), ERODED by theta_e so
              that no ray sits within theta_e of the survey boundary; baseline
              theta_e = 5 deg (28.9 Mpc/h transverse at chi_max, ~1.7 median
              void radii).  2 and 8 deg are carried as robustness variants.

TEMPLATES
---------
  I_q     union-of-spheres path length                       [Mpc/h]  REGRESSOR
  I_void  sum over voids of that void's own union length     [Mpc/h]  (check)
  N_v     number of distinct voids the ray enters            [-]
  I_R     sum_v L_v * (Reff_v / <Reff>)                      [Mpc/h]  out-of-grammar
  I_sat   sum_v min(L_v, 10 Mpc/h)                           [Mpc/h]  out-of-grammar
  I_phi_in  LOS integral of the void interior potential          [Mpc^3/h^3]
  I_phi_k3  same, plus the exterior shell out to 3 Reff                ISW template
  I_q_near / I_q_far   I_q split at chi_max/2                    [Mpc/h]  tomography
  b_gal   galactic latitude                                  [deg]    foreground
  csc_b   1/sin|b|                                           [-]      dust proxy
  edge    angular distance to the footprint boundary         [deg]    edge control

Why I_phi separates from I_q: a top-hat void of radius R contributes a chord
~2R to I_q but ~R^3 to the potential integral, so the two templates weight the
void radius function differently by two powers.  That difference, not an
assumed amplitude, is what makes the ISW nuisance separable.

    python pathmap.py [--nside 64] [--erode 5.0]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(LANE, "..", ".."))
VAST = os.path.join(REPO, "work", "private", "open-gravity-void-source-v2")

# --- fiducial normalisation, identical to the redshift lane (Run AK 2.1) ------
C_KMS = 299792.458
H0_H = 100.0                       # h = 1, lengths in Mpc/h
C1_FIDUCIAL = H0_H / C_KMS         # 3.33564e-4 per Mpc/h
T_CMB_K = 2.7255

# dT/T = -c2 * dI_q  =>  c2 = -(dT/d dI_q)/T_CMB ; ratio to fiducial c1
# beta in uK per (Mpc/h)  ->  c2/c1
UK_PER_MPCH_TO_C2C1 = -1e-6 / (T_CMB_K * C1_FIDUCIAL)      # = -1.09989e-3


def load_mask():
    """NSA main angular mask: mask[floor(ra), floor(dec+90)], 1 deg cells."""
    with open(os.path.join(VAST, "NSA_main_mask.pickle"), "rb") as fh:
        mask, res, lim = pickle.load(fh)
    assert mask.shape == (360 * res, 180 * res), mask.shape
    return np.asarray(mask, bool), int(res), np.asarray(lim, float)


def load_holes():
    ho = np.loadtxt(os.path.join(
        VAST, "VoidFinder-nsa_v1_0_1_Planck2018_comoving_holes.txt"), skiprows=1)
    mx = np.loadtxt(os.path.join(
        VAST, "VoidFinder-nsa_v1_0_1_Planck2018_comoving_maximal.txt"), skiprows=1)
    assert ho.shape[0] == 39735, ho.shape
    assert mx.shape[0] == 1163, mx.shape
    holes = dict(c=ho[:, :3], R=ho[:, 3], vid=ho[:, 4].astype(np.int64))
    maxi = dict(c=mx[:, :3], R=mx[:, 3], vid=mx[:, 4].astype(np.int64),
                edge=mx[:, 5].astype(np.int64), r=mx[:, 6], ra=mx[:, 7],
                dec=mx[:, 8], reff=mx[:, 9])
    return holes, maxi


def galactic_pixel_dirs(nside):
    """Pixel centres of the Planck grid, as galactic (l,b) and EQUATORIAL unit vecs."""
    from astropy.coordinates import SkyCoord
    from astropy_healpix import HEALPix
    import astropy.units as u
    hp = HEALPix(nside=nside, order="nested")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    gal = SkyCoord(l=lon, b=lat, frame="galactic")
    eq = gal.icrs
    ra = eq.ra.deg
    dec = eq.dec.deg
    cd = np.cos(np.radians(dec))
    U = np.stack([cd * np.cos(np.radians(ra)), cd * np.sin(np.radians(ra)),
                  np.sin(np.radians(dec))], axis=1)
    return dict(nside=nside, npix=hp.npix, l=lon.deg, b=lat.deg, ra=ra, dec=dec, U=U)


def erode_mask(mask, res, theta_deg):
    """Keep a 1-deg cell only if every cell within theta_deg on the sky is set.

    Done on the sphere (great-circle), not in flat RA/Dec, so the RA convergence
    near the pole cannot silently under-erode.
    """
    nra, ndec = mask.shape
    ii, jj = np.meshgrid(np.arange(nra), np.arange(ndec), indexing="ij")
    ra_c = (ii + 0.5) / res
    dec_c = (jj + 0.5) / res - 90.0
    cd = np.cos(np.radians(dec_c))
    V = np.stack([cd * np.cos(np.radians(ra_c)), cd * np.sin(np.radians(ra_c)),
                  np.sin(np.radians(dec_c))], axis=-1).reshape(-1, 3)
    inside = mask.reshape(-1)
    out_idx = np.where(~inside)[0]
    if len(out_idx) == 0:
        return mask.copy()
    cos_t = np.cos(np.radians(theta_deg))
    keep = inside.copy()
    Vout = V[out_idx]
    # chunk over the inside cells
    idx_in = np.where(inside)[0]
    for s in range(0, len(idx_in), 2000):
        sl = idx_in[s:s + 2000]
        d = V[sl] @ Vout.T                      # cos of separation
        keep[sl] = d.max(axis=1) < cos_t        # nearest outside cell farther than theta
    return keep.reshape(mask.shape)


def mask_lookup(mask, res, ra, dec):
    i = np.clip(np.floor((ra % 360.0) * res).astype(np.int64), 0, mask.shape[0] - 1)
    j = np.clip(np.floor((dec + 90.0) * res).astype(np.int64), 0, mask.shape[1] - 1)
    return mask[i, j]


def edge_distance_deg(mask, res, ra, dec):
    """Great-circle angular distance from each direction to the nearest OUTSIDE cell."""
    nra, ndec = mask.shape
    ii, jj = np.meshgrid(np.arange(nra), np.arange(ndec), indexing="ij")
    ra_c = (ii + 0.5) / res
    dec_c = (jj + 0.5) / res - 90.0
    out = ~mask.reshape(-1)
    cd = np.cos(np.radians(dec_c))
    V = np.stack([cd * np.cos(np.radians(ra_c)), cd * np.sin(np.radians(ra_c)),
                  np.sin(np.radians(dec_c))], axis=-1).reshape(-1, 3)[out]
    cdq = np.cos(np.radians(dec))
    Q = np.stack([cdq * np.cos(np.radians(ra)), cdq * np.sin(np.radians(ra)),
                  np.sin(np.radians(dec))], axis=-1)
    d = np.empty(len(Q))
    for s in range(0, len(Q), 2000):
        d[s:s + 2000] = np.degrees(np.arccos(np.clip((Q[s:s + 2000] @ V.T).max(axis=1), -1, 1)))
    return d


def _union_len(t1, t2, lo, hi):
    """Length of the union of intervals [t1,t2] clipped to [lo,hi]."""
    a = np.clip(t1, lo, hi)
    b = np.clip(t2, lo, hi)
    m = b > a
    if not np.any(m):
        return 0.0
    a = a[m]
    b = b[m]
    o = np.argsort(a)
    a = a[o]
    b = b[o]
    total = 0.0
    ca, cb = a[0], b[0]
    for k in range(1, len(a)):
        if a[k] <= cb:
            cb = max(cb, b[k])
        else:
            total += cb - ca
            ca, cb = a[k], b[k]
    return total + (cb - ca)


def trace(holes, maxi, U, chi_max, verbose_every=2000):
    """Exact ray tracing. Returns the template dictionary for the given directions."""
    c = holes["c"]
    R = holes["R"]
    vid = holes["vid"]
    c2 = np.einsum("ij,ij->i", c, c)
    R2 = R * R
    reff_by_void = np.zeros(int(maxi["vid"].max()) + 1)
    reff_by_void[maxi["vid"]] = maxi["reff"]
    reff_bar = float(maxi["reff"].mean())
    edge_by_void = np.zeros(int(maxi["vid"].max()) + 1, dtype=np.int64)
    edge_by_void[maxi["vid"]] = maxi["edge"]

    n = len(U)
    out = {k: np.zeros(n) for k in
           ("I_q", "I_void", "N_v", "I_R", "I_sat", "I_q_nonedge")}
    t0 = time.time()
    for i in range(n):
        u = U[i]
        b = c @ u
        disc = R2 - (c2 - b * b)
        hit = (disc > 0) & (b + R > 0)
        if not np.any(hit):
            continue
        s = np.sqrt(disc[hit])
        bh = b[hit]
        t1 = bh - s
        t2 = bh + s
        vh = vid[hit]
        out["I_q"][i] = _union_len(t1, t2, 0.0, chi_max)
        ne = edge_by_void[vh] == 0
        if np.any(ne):
            out["I_q_nonedge"][i] = _union_len(t1[ne], t2[ne], 0.0, chi_max)
        # per-void unions
        order = np.argsort(vh, kind="stable")
        vs = vh[order]
        a1 = t1[order]
        a2 = t2[order]
        bounds = np.flatnonzero(np.diff(vs)) + 1
        starts = np.concatenate([[0], bounds])
        stops = np.concatenate([bounds, [len(vs)]])
        tot = 0.0
        nv = 0
        ir = 0.0
        isat = 0.0
        for p, q in zip(starts, stops):
            L = _union_len(a1[p:q], a2[p:q], 0.0, chi_max)
            if L <= 0.0:
                continue
            nv += 1
            tot += L
            ir += L * (reff_by_void[vs[p]] / reff_bar)
            isat += min(L, 10.0)
        out["I_void"][i] = tot
        out["N_v"][i] = nv
        out["I_R"][i] = ir
        out["I_sat"][i] = isat
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    ray {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def potential_integral(maxi, U, chi_max, k_out=3.0):
    """LOS integral of a LOCAL void potential -- the ISW nuisance template.

    Shape (the amplitude (1/4) H0^2 Omega_m |delta| is FITTED, never assumed):

        phi(r) = 3R^2 - r^2   for r < R          (top-hat interior)
               = 2R^3 / r     for R < r < k R    (exterior, continuous at r=R)
               = 0            beyond k R

    THE EXTERIOR MUST BE TRUNCATED.  A first version integrated 2R^3/r over the
    whole ray; the resulting template correlated at -0.83 with the distance to
    the footprint edge and only -0.14 with I_q, i.e. it was a survey-geometry
    template wearing a physics label.  It is the unbounded far field -- which
    this truncated survey cannot measure anyway -- that produced that, so it is
    removed rather than fitted.  k_out = 3 keeps the term local.

    Returns (J_in, J_k): interior only, and interior + exterior to k_out * R.
    """
    c = maxi["c"]
    Rv = maxi["reff"]
    c2 = np.einsum("ij,ij->i", c, c)
    n = len(U)
    J_in = np.zeros(n)
    J_k = np.zeros(n)
    for i in range(n):
        u = U[i]
        t0 = c @ u                                # closest approach along the ray
        b2 = np.maximum(c2 - t0 * t0, 0.0)
        bb = np.maximum(np.sqrt(b2), 1e-6)

        # --- interior: integral of (3R^2 - r^2) dl, r^2 = b^2 + l^2
        ins = b2 < Rv * Rv
        L = np.zeros_like(Rv)
        L[ins] = np.sqrt(Rv[ins] ** 2 - b2[ins])
        d1 = np.clip(np.minimum(t0 + L, chi_max) - t0, 0.0, None)
        d0 = np.clip(t0 - np.maximum(t0 - L, 0.0), 0.0, None)
        val_in = np.where(ins, (3 * Rv ** 2 - b2) * (d0 + d1)
                          - (d1 ** 3 + d0 ** 3) / 3.0, 0.0)
        J_in[i] = float(val_in.sum())

        # --- exterior shell R < r < k R: integral of 2R^3/sqrt(b^2+l^2) dl
        Rk = k_out * Rv
        ins_k = b2 < Rk * Rk
        Lk = np.zeros_like(Rv)
        Lk[ins_k] = np.sqrt(Rk[ins_k] ** 2 - b2[ins_k])
        e1 = np.clip(np.minimum(t0 + Lk, chi_max) - t0, 0.0, None)
        e0 = np.clip(t0 - np.maximum(t0 - Lk, 0.0), 0.0, None)
        outer = np.where(ins_k, np.arcsinh(e1 / bb) + np.arcsinh(e0 / bb), 0.0)
        inner = np.where(ins, np.arcsinh(d1 / bb) + np.arcsinh(d0 / bb), 0.0)
        val_out = 2.0 * Rv ** 3 * np.maximum(outer - inner, 0.0)
        J_k[i] = J_in[i] + float(val_out.sum())
    return J_in, J_k


def split_path(holes, U, chi_max, chi_split):
    """I_q restricted to the near and far halves of the ray (tomographic check)."""
    c = holes["c"]
    R = holes["R"]
    c2 = np.einsum("ij,ij->i", c, c)
    R2 = R * R
    near = np.zeros(len(U))
    far = np.zeros(len(U))
    for i in range(len(U)):
        u = U[i]
        b = c @ u
        disc = R2 - (c2 - b * b)
        hit = (disc > 0) & (b + R > 0)
        if not np.any(hit):
            continue
        s_ = np.sqrt(disc[hit])
        bh = b[hit]
        near[i] = _union_len(bh - s_, bh + s_, 0.0, chi_split)
        far[i] = _union_len(bh - s_, bh + s_, chi_split, chi_max)
    return near, far


def build(nside=64, erode_deg=5.0, tag=""):
    t0 = time.time()
    mask, res, lim = load_mask()
    chi_max = float(lim[1])
    holes, maxi = load_holes()
    print(f"NSA mask {mask.sum()} cells (res {res}); chi_max {chi_max:.3f} Mpc/h")
    print(f"holes {len(holes['R'])}, voids {len(maxi['R'])}, "
          f"Reff {maxi['reff'].min():.1f}-{maxi['reff'].max():.1f} Mpc/h")

    grid = galactic_pixel_dirs(nside)
    er = erode_mask(mask, res, erode_deg) if erode_deg > 0 else mask
    ins = mask_lookup(er, res, grid["ra"], grid["dec"])
    idx = np.where(ins)[0]
    print(f"nside {nside}: {len(idx)} footprint pixels "
          f"({len(idx) * (4 * np.pi / grid['npix']) * (180/np.pi)**2:.0f} deg^2) "
          f"after {erode_deg:g} deg erosion", flush=True)

    U = grid["U"][idx]
    tem = trace(holes, maxi, U, chi_max)
    tem["I_phi_in"], tem["I_phi_k3"] = potential_integral(maxi, U, chi_max, 3.0)
    chi_split = 0.5 * chi_max
    tem["I_q_near"], tem["I_q_far"] = split_path(holes, U, chi_max, chi_split)
    tem["b_gal"] = grid["b"][idx]
    tem["csc_b"] = 1.0 / np.maximum(np.abs(np.sin(np.radians(grid["b"][idx]))), 1e-3)
    tem["edge_deg"] = edge_distance_deg(mask, res, grid["ra"][idx], grid["dec"][idx])
    tem["ra"] = grid["ra"][idx]
    tem["dec"] = grid["dec"][idx]
    tem["l_gal"] = grid["l"][idx]

    meta = dict(
        nside=nside, order="NESTED", coordsys="GALACTIC", erode_deg=erode_deg,
        npix_full=int(grid["npix"]), n_footprint=int(len(idx)),
        area_deg2=float(len(idx) * (4 * np.pi / grid["npix"]) * (180 / np.pi) ** 2),
        chi_max_mpch=chi_max, z_max=0.1125,
        catalogue="SDSS DR7/NSA v1.0.1 VAST VoidFinder, Planck2018 comoving",
        n_holes=int(len(holes["R"])), n_voids=int(len(maxi["R"])),
        reff_median=float(np.median(maxi["reff"])),
        c1_fiducial_per_mpch=C1_FIDUCIAL, T_CMB_K=T_CMB_K,
        uK_per_mpch_to_c2_over_c1=UK_PER_MPCH_TO_C2C1,
        I_q_mean=float(tem["I_q"].mean()), I_q_sd=float(tem["I_q"].std()),
        I_q_min=float(tem["I_q"].min()), I_q_max=float(tem["I_q"].max()),
        void_volume_fraction=float(tem["I_q"].mean() / chi_max),
        union_vs_sum_mean_excess=float(np.mean(tem["I_void"] - tem["I_q"])),
        chi_split_mpch=float(0.5 * chi_max),
        seconds=time.time() - t0)
    name = f"pathmap_ns{nside}_er{erode_deg:g}{tag}"
    np.savez_compressed(os.path.join(HERE, name + ".npz"), pix=idx, **tem)
    io.open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8",
            newline="\n").write(json.dumps(meta, indent=1, default=float))
    print(json.dumps(meta, indent=1, default=float))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--erode", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    build(a.nside, a.erode, a.tag)
