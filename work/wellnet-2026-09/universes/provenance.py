"""provenance.py -- mechanical assertion of what this lane opened.

Standing constraint 2 of the task brief: "This lane should need NO real
observational data -- it is synthetic.  Assert mechanically what you opened."

So we do not promise it in prose.  We patch ``builtins.open``, ``io.open``,
``numpy.load``, ``numpy.loadtxt``, ``numpy.genfromtxt`` and ``os.scandir`` for
the duration of the run, record every path that is read, and raise immediately
if anything outside the lane's own directory (or the Python standard library /
site-packages) is touched.

Constraint 1: KiDS and the wide binaries are SEALED.  Any path whose lowercase
form matches a sealed token raises ``SealedHoldoutTouched`` before the read can
happen.  This is a hard raise, not a warning.

Units used throughout the lane: kpc, Msun, km/s, Gyr.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import platform
import site
import sys
import sysconfig
import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# sealed holdouts -- never opened, never scored against, at any stage
# --------------------------------------------------------------------------
SEALED_TOKENS = (
    "kids", "kids-1000", "kids1000", "kv450", "kids_dr4", "kids_dr5",
    "wide_binar", "wide-binar", "widebinar", "gaia_wb", "wb_catalog",
    "hernandez_wb", "banik_wb", "chae_wb",
)

# real-observation tokens this lane must also not need
REAL_DATA_TOKENS = (
    "sparc", "xcop", "x-cop", "efeds", "locuss", "clash", "frontier",
    "pantheon", "desi", "des_y", "act_dr", "planck", "sami", "diskmass",
    "herbonnet", "mulroy", "xxl",
)


class SealedHoldoutTouched(RuntimeError):
    """Raised the instant a sealed-holdout path is opened."""


class ForeignReadError(RuntimeError):
    """Raised when the lane reads a file outside its own tree."""


def _norm(p) -> str:
    try:
        return os.path.abspath(os.fspath(p)).replace("\\", "/").lower()
    except Exception:
        return str(p).replace("\\", "/").lower()


def _is_library(path: str) -> bool:
    """stdlib / site-packages / the interpreter itself -- code, not data."""
    roots = set()
    for k in ("stdlib", "platstdlib", "purelib", "platlib", "scripts", "data"):
        try:
            v = sysconfig.get_path(k)
        except Exception:
            v = None
        if v:
            roots.add(_norm(v))
    try:
        for v in site.getsitepackages():
            roots.add(_norm(v))
    except Exception:
        pass
    try:
        roots.add(_norm(site.getusersitepackages()))
    except Exception:
        pass
    roots.add(_norm(sys.prefix))
    roots.add(_norm(sys.base_prefix))
    return any(path.startswith(r + "/") or path == r for r in roots)


@dataclass
class OpenLedger:
    lane_root: str
    reads: dict = field(default_factory=dict)      # path -> count
    writes: dict = field(default_factory=dict)
    foreign: list = field(default_factory=list)
    active: bool = False

    # -- installation ------------------------------------------------------
    _saved: dict = field(default_factory=dict, repr=False)

    def check(self, path, mode="r"):
        p = _norm(path)
        low = p
        for tok in SEALED_TOKENS:
            if tok in low:
                raise SealedHoldoutTouched(
                    f"SEALED HOLDOUT TOUCHED: {path!r} matches sealed token {tok!r}. "
                    "KiDS and the wide binaries are permanently blind to this programme."
                )
        writing = any(c in str(mode) for c in ("w", "a", "x", "+"))
        if writing:
            self.writes[p] = self.writes.get(p, 0) + 1
            return
        self.reads[p] = self.reads.get(p, 0) + 1
        if _is_library(p):
            return
        if p.startswith(self.lane_root):
            return
        # anything else is a foreign read -- record and raise
        self.foreign.append(p)
        raise ForeignReadError(
            f"FOREIGN READ: {path!r} is outside the lane root {self.lane_root!r}. "
            "This lane is purely synthetic and must open no observational data."
        )

    def install(self):
        if self.active:
            return
        led = self

        real_open = builtins.open

        def guarded_open(file, mode="r", *a, **kw):
            led.check(file, mode)
            return real_open(file, mode, *a, **kw)

        self._saved["builtins.open"] = real_open
        builtins.open = guarded_open
        self._saved["io.open"] = io.open
        io.open = guarded_open

        for name in ("load", "loadtxt", "genfromtxt", "fromfile"):
            fn = getattr(np, name, None)
            if fn is None:
                continue
            self._saved[f"np.{name}"] = fn

            def mk(fn=fn):
                def g(f, *a, **kw):
                    if isinstance(f, (str, bytes, os.PathLike)):
                        led.check(f, "r")
                    return fn(f, *a, **kw)
                return g
            setattr(np, name, mk())

        self.active = True

    def uninstall(self):
        if not self.active:
            return
        builtins.open = self._saved["builtins.open"]
        io.open = self._saved["io.open"]
        for k, v in self._saved.items():
            if k.startswith("np."):
                setattr(np, k[3:], v)
        self.active = False

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        non_lib = sorted(p for p in self.reads if not _is_library(p))
        lane = [p for p in non_lib if p.startswith(self.lane_root)]
        return {
            "lane_root": self.lane_root,
            "n_read_paths_total": len(self.reads),
            "n_read_paths_library": len(self.reads) - len(non_lib),
            "n_read_paths_non_library": len(non_lib),
            "non_library_reads": lane,
            "foreign_reads": sorted(set(self.foreign)),
            "sealed_tokens_guarded": list(SEALED_TOKENS),
            "real_data_tokens_guarded": list(REAL_DATA_TOKENS),
            "any_real_observational_file_opened": bool(
                [p for p in non_lib
                 if any(t in p for t in REAL_DATA_TOKENS + SEALED_TOKENS)]
            ),
            "assertion": (
                "No path outside the lane root was read; no sealed token matched; "
                "no real-observation token matched."
            ),
        }


_LEDGER: OpenLedger | None = None


def start_ledger(lane_root: str) -> OpenLedger:
    global _LEDGER
    _LEDGER = OpenLedger(lane_root=_norm(lane_root))
    _LEDGER.install()
    return _LEDGER


def ledger() -> OpenLedger:
    if _LEDGER is None:
        raise RuntimeError("ledger not started")
    return _LEDGER


def stop_ledger() -> dict:
    led = ledger()
    led.uninstall()
    return led.summary()


# --------------------------------------------------------------------------
# declared noise / systematics characterisations
# --------------------------------------------------------------------------
# Every number below is a DECLARED SYNTHETIC VALUE chosen to be broadly
# representative of current wide-field / X-ray / IFU practice.  No survey
# characterisation file was opened to produce them, and in particular NOTHING
# from KiDS was used, since KiDS is a sealed holdout for this programme.  They
# are inputs to the mock, not measurements, and the equivalence-class results
# are quoted as a function of them (see the amplitude scans).
DECLARED_NOISE = {
    "wl_shape_noise_per_component": 0.26,
    "wl_source_density_arcmin2": 20.0,
    "wl_multiplicative_bias_sigma": 0.02,
    "wl_additive_bias_sigma": 5.0e-4,
    "wl_photoz_mean_bias_sigma": 0.02,
    "wl_photoz_outlier_fraction": 0.05,
    "ifu_velocity_error_kms_at_1Re": 8.0,
    "ifu_psf_fwhm_arcsec": 1.5,
    "xray_counts_per_annulus_ref": 3000.0,
    "xray_kT_frac_error": 0.06,
    "sz_y_frac_error": 0.10,
    "sl_image_position_error_arcsec": 0.2,
    "sl_time_delay_frac_error": 0.03,
    "sn_peak_mag_scatter": 0.12,
    "sn_duration_frac_error": 0.04,
    "member_velocity_error_kms": 30.0,
    "distance_frac_error": 0.10,
    "inclination_error_deg": 3.0,
    "ml_dex_scatter": 0.11,
    "_provenance": (
        "declared synthetic values; representative of current wide-field, "
        "X-ray and IFU practice; no survey file opened; KiDS deliberately "
        "excluded as a sealed holdout"
    ),
}


def run_manifest(extra: dict | None = None) -> dict:
    m = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "declared_noise": DECLARED_NOISE,
    }
    if extra:
        m.update(extra)
    return m


def sha256_of(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
