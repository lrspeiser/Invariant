"""Planck map I/O: read, mask, degrade -- and a BLIND GUARD on the true footprint.

All Planck products used here are NESTED-ordered HEALPix in GALACTIC coordinates,
so degrading from nside 2048 to nside 64 is an exact block mean over contiguous
index ranges: superpixel = pix >> 2*(order_hi - order_lo).  No interpolation, no
regridding, and the CMB is never resampled onto the void map's grid -- the void
map was built ON the Planck grid instead (pathmap.py).

THE BLIND GUARD
---------------
`BlindGuard` refuses to return the statistic for a pixel set that overlaps the
true survey footprint by more than `max_overlap`.  The Stage 4 certificate runs
with the guard ARMED, so it can read real Planck temperatures at rotated sky
positions (which is what sizing the test requires) while being unable to see the
measurement.  `disarm(reason)` is a one-shot, logged action.
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

MAPS = {
    "smica_nosz_pr3": ("COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits", "I_STOKES", 2048),
    "smica_pr2": ("COM_CMB_IQU-smica_1024_R2.02_full.fits", "I_STOKES", 1024),
}
MASK = ("COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits", "TMASK", 2048)
FOREGROUNDS = {
    "dust": ("COM_CompMap_dust-commander_0256_R2.00.fits", "I_ML", 256),
    "sync": ("COM_CompMap_Synchrotron-commander_0256_R2.00.fits", "I_ML", 256),
    "freefree": ("COM_CompMap_freefree-commander_0256_R2.00.fits", "EM_ML", 256),
    "ame": ("COM_CompMap_AME-commander_0256_R2.00.fits", "I_ML", 256),
    "co": ("COM_CompMap_CO-commander_0256_R2.00.fits", "I_ML", 256),
    "dust_temp": ("COM_CompMap_dust-commander_0256_R2.00.fits", "TEMP_ML", 256),
}


def _read_col(fname, col, nside_expected):
    p = os.path.join(RAW, fname)
    with fits.open(p, memmap=True) as h:
        hd = h[1].header
        assert int(hd["NSIDE"]) == nside_expected, (fname, hd["NSIDE"])
        assert str(hd["ORDERING"]).strip().upper() == "NESTED", fname
        assert str(hd["COORDSYS"]).strip().upper().startswith("G"), fname
        unit = None
        for i, c in enumerate(h[1].columns):
            if c.name == col:
                unit = c.unit
        v = np.asarray(h[1].data[col], dtype=np.float64).ravel()
    assert v.size == 12 * nside_expected ** 2, (fname, v.size)
    return v, unit


def degrade_nested(v, nside_in, nside_out, weights=None):
    """Exact NESTED block mean from nside_in to nside_out."""
    assert nside_in >= nside_out
    k = int(round(np.log2(nside_in / nside_out)))
    assert 2 ** k * nside_out == nside_in
    n = 4 ** k
    v = v.reshape(-1, n)
    if weights is None:
        return v.mean(axis=1), np.ones(v.shape[0])
    w = np.asarray(weights, float).reshape(-1, n)
    sw = w.sum(axis=1)
    out = np.where(sw > 0, (v * w).sum(axis=1) / np.where(sw > 0, sw, 1), np.nan)
    return out, sw / n


def upgrade_nested(v, nside_in, nside_out):
    """Exact NESTED replication from a coarser map onto a finer grid."""
    assert nside_out >= nside_in
    k = int(round(np.log2(nside_out / nside_in)))
    return np.repeat(v, 4 ** k)


def load_mask(nside_out=64):
    m, _ = _read_col(*MASK)
    frac, _ = degrade_nested(m, MASK[2], nside_out)
    return m, frac


def load_map(key, nside_out=64, mask_hi=None):
    fname, col, nside = MAPS[key]
    v, unit = _read_col(fname, col, nside)
    bad = ~np.isfinite(v) | (np.abs(v) > 1.0)          # Planck flags with -1.6375e30
    v = np.where(bad, 0.0, v)
    w = (~bad).astype(np.float64)
    if mask_hi is not None:
        mh = mask_hi if len(mask_hi) == len(v) else (
            degrade_nested(mask_hi, MASK[2], nside)[0] if len(mask_hi) > len(v)
            else upgrade_nested(mask_hi, MASK[2], nside))
        w = w * (mh > 0.5)
    lo, frac = degrade_nested(v, nside, nside_out, weights=w)
    return dict(key=key, nside=nside_out, unit=unit, T=lo * 1e6, frac=frac,
                native_nside=nside, file=fname, column=col)   # T in microkelvin


def load_foreground(key, nside_out=64):
    fname, col, nside = FOREGROUNDS[key]
    v, unit = _read_col(fname, col, nside)
    v = np.where(np.isfinite(v) & (np.abs(v) < 1e20), v, np.nan)
    if nside >= nside_out:
        lo, _ = degrade_nested(np.nan_to_num(v), nside, nside_out,
                               weights=np.isfinite(v).astype(float))
    else:
        lo = upgrade_nested(v, nside, nside_out)
    return lo, unit


# ---------------------------------------------------------------- blind guard
class BlindGuard:
    """Refuses any evaluation whose pixel set overlaps the true footprint."""

    def __init__(self, true_pixels, max_overlap=0.05, log=None):
        self.true = np.asarray(true_pixels, np.int64)
        self.true_set = set(self.true.tolist())
        self.max_overlap = float(max_overlap)
        self.armed = True
        self.log = log or os.path.join(HERE, "blind_guard.log")
        self.n_checked = 0
        self.n_refused = 0

    def check(self, pixels, what=""):
        self.n_checked += 1
        if not self.armed:
            return True
        pix = np.asarray(pixels, np.int64)
        ov = len(self.true_set.intersection(pix.tolist())) / max(len(pix), 1)
        if ov > self.max_overlap:
            self.n_refused += 1
            raise PermissionError(
                f"BlindGuard: refused '{what}' -- pixel set overlaps the true "
                f"footprint at {ov:.1%} > {self.max_overlap:.0%}. The certificate "
                f"may not see the measurement.")
        return True

    def disarm(self, reason):
        self.armed = False
        rec = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   reason=reason, checked=self.n_checked, refused=self.n_refused)
        with io.open(self.log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec
