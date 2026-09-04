"""
Shared geometry / cosmology / provenance utilities for the void-data lane.

FIDUCIAL GEOMETRY (declared once, used everywhere)
--------------------------------------------------
Flat LCDM, Omega_m = 0.315, Omega_L = 0.685, h = 1 (all comoving lengths in
Mpc/h).  This is *exactly* the cosmology DESIVAST used to build its void
catalogues (FITS PRIMARY header: OMEGAM = 0.315, HP = 1.0, METRIC = 'comoving',
DLIMU = 677.4038194061428 Mpc/h at ZLIMU = 0.24).  Reproducing DLIMU to 3e-8
Mpc/h is the check that we are in the same frame as the void catalogue.

Comoving Cartesian convention (verified against the DESIVAST MAXIMALS table to
1e-14 deg on RA/Dec and 0.0 on radius):

    x = r cos(dec) cos(ra)
    y = r cos(dec) sin(ra)
    z = r sin(dec)
    r = D_C(z_cosmological)

Redshifts used to place *catalogue* objects are observed redshifts in the CMB
frame; see REPORT.md section on circularity for what that assumption costs.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

C_KMS = 299792.458
OMEGA_M = 0.315
OMEGA_L = 1.0 - OMEGA_M
H0_H = 100.0  # km/s/Mpc with h=1, i.e. lengths are Mpc/h
DH = C_KMS / H0_H  # Hubble distance, 2997.92458 Mpc/h

# DESIVAST survey limits
Z_MAX_VOID = 0.24
R_MAX_VOID = 677.4038194061428  # Mpc/h, from the DESIVAST FITS header


# --------------------------------------------------------------------------
# cosmology
# --------------------------------------------------------------------------
def _e_inv(z):
    return 1.0 / np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


_ZGRID = np.linspace(0.0, 3.0, 300001)
_mid = 0.5 * (_ZGRID[1:] + _ZGRID[:-1])
_DCGRID = np.concatenate([[0.0], np.cumsum(np.diff(_ZGRID) * _e_inv(_mid))]) * DH


def comoving_distance(z):
    """Comoving distance in Mpc/h for the fiducial cosmology."""
    return np.interp(np.asarray(z, float), _ZGRID, _DCGRID)


def z_of_comoving(r):
    """Inverse of comoving_distance."""
    return np.interp(np.asarray(r, float), _DCGRID, _ZGRID)


def luminosity_to_comoving(dl_mpc, z):
    """Convert a luminosity distance (Mpc, h=1 convention) to comoving Mpc/h."""
    return np.asarray(dl_mpc, float) / (1.0 + np.asarray(z, float))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def sky_to_unit(ra_deg, dec_deg):
    """Unit vectors from RA/Dec in degrees. Shape (N, 3)."""
    ra = np.radians(np.asarray(ra_deg, float))
    dec = np.radians(np.asarray(dec_deg, float))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def sky_to_cartesian(ra_deg, dec_deg, r):
    return sky_to_unit(ra_deg, dec_deg) * np.asarray(r, float)[..., None]


def cartesian_to_sky(xyz):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    r = np.linalg.norm(xyz, axis=1)
    ra = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.where(r > 0, r, 1), -1, 1)))
    return ra, dec, r


# --------------------------------------------------------------------------
# angular footprint mask (no healpy on this machine -> equal-area lon/sin(lat))
# --------------------------------------------------------------------------
class FootprintMask:
    """
    Equal-area (RA, sin Dec) pixelisation built from a galaxy catalogue.

    A pixel is inside the footprint if it contains at least `min_count`
    catalogue galaxies.  `pix_deg` sets the nominal pixel scale at the equator.
    """

    def __init__(self, ra, dec, pix_deg=0.5, min_count=1):
        self.pix_deg = float(pix_deg)
        self.min_count = int(min_count)
        self.n_ra = int(round(360.0 / pix_deg))
        self.n_sd = int(round(180.0 / pix_deg))  # bins in sin(dec) over [-1, 1]
        idx = self._index(np.asarray(ra, float), np.asarray(dec, float))
        counts = np.bincount(idx, minlength=self.n_ra * self.n_sd)
        self.counts = counts
        self.mask = counts >= self.min_count
        # solid angle of one pixel, steradians: (2*pi/n_ra) * (2/n_sd)
        self.pix_sr = (2.0 * np.pi / self.n_ra) * (2.0 / self.n_sd)

    def _index(self, ra, dec):
        i = np.floor((ra % 360.0) / 360.0 * self.n_ra).astype(np.int64)
        s = np.sin(np.radians(dec))
        j = np.floor((s + 1.0) * 0.5 * self.n_sd).astype(np.int64)
        i = np.clip(i, 0, self.n_ra - 1)
        j = np.clip(j, 0, self.n_sd - 1)
        return j * self.n_ra + i

    def contains(self, ra, dec):
        return self.mask[self._index(np.asarray(ra, float), np.asarray(dec, float))]

    @property
    def area_sr(self):
        return self.mask.sum() * self.pix_sr

    @property
    def area_deg2(self):
        return self.area_sr * (180.0 / np.pi) ** 2


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(path, source_url, query, rows, columns, notes=None,
                   retrieved=None, extra=None):
    """Write <path>.manifest.json next to a downloaded file."""
    man = {
        "file": os.path.basename(path),
        "source_url": source_url,
        "query_issued": query,
        "retrieved_utc": retrieved or utc_now(),
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "row_count": rows,
        "columns": columns,
        "notes": notes or "",
    }
    if extra:
        man.update(extra)
    with open(path + ".manifest.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    return man
