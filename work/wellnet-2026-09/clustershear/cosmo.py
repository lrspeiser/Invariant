"""cosmo.py -- flat LCDM distances, self-contained.

Deliberately not imported from efeds-hsc/pipeline.py: that module reads files
inside its own lane, which would be a foreign read here, and a shared mutable
cosmology is exactly the kind of coupling that makes two lanes' numbers
silently incomparable. The constants are the same ones the eFEDS lane used
(H0 = 70, Om = 0.3, h = 0.7), stated here so a reader can check them in one
place.
"""
from __future__ import annotations

import numpy as np

H0_KM_S_MPC = 70.0
OM = 0.3
OL = 0.7
H_LITTLE = 0.7

CLIGHT = 2.99792458e8            # m/s
G = 6.67430e-11                  # m^3 kg^-1 s^-2
MPC = 3.085677581491367e22       # m

_C_KM_S = CLIGHT / 1000.0
_HUBBLE_DIST_MPC = _C_KM_S / H0_KM_S_MPC

#: comoving distance on a fixed grid, interpolated -- the extractor calls this
#: once per source, so a 2000-step quadrature per call would dominate the run.
_ZGRID = np.linspace(0.0, 4.0, 4001)


def _E_inv(z):
    return 1.0 / np.sqrt(OM * (1.0 + z) ** 3 + OL)


_DC = np.concatenate(([0.0], np.cumsum(
    0.5 * (_E_inv(_ZGRID[:-1]) + _E_inv(_ZGRID[1:])) * np.diff(_ZGRID))))
_DC *= _HUBBLE_DIST_MPC * MPC          # metres


def d_com(z):
    """Comoving distance, metres."""
    return np.interp(z, _ZGRID, _DC)


def d_ang(z):
    """Angular diameter distance, metres."""
    return d_com(z) / (1.0 + np.asarray(z, dtype=float))
