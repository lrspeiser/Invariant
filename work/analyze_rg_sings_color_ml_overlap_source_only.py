"""Benchmark published SINGS two-band stellar-mass fallbacks against S4G P5."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


ROOT = Path(__file__).resolve().parents[1]
IRAC1_ROOT = ROOT / "work/private/open-gravity-rg-sings-s4g-overlap-source-only-v1"
IRAC2_ROOT = ROOT / "work/private/open-gravity-rg-sings-irac2-source-only-v1"
S4G_ROOT = ROOT / "work/private/open-gravity-rg-12gal-source-only-v1"
OUTPUT = ROOT / "work/rg-sings-color-ml-overlap-source-comparison-v1.json"
OBJECTS = ("NGC2976", "NGC3198", "NGC3521")
S4G_RMS_MJY_SR = 0.0072
SOURCE_THRESHOLD_SIGMA = 5.0
LUMINOSITY_PER_MJY_SR = 704.04
FIXED_ML = 0.6
IRAC1_ZERO_JY = 280.9
IRAC2_ZERO_JY = 179.7


def _read(path: Path) -> tuple[np.ndarray, fits.Header]:
    return np.asarray(fits.getdata(path, memmap=False), dtype=np.float64), fits.getheader(path)


def _wcs(header: fits.Header) -> WCS:
    value = WCS(header, relax=True)
    value.sip = None
    return value


def _reproject(data: np.ndarray, header: fits.Header, target: WCS, shape: tuple[int, int], *, order: str) -> tuple[np.ndarray, np.ndarray]:
    return reproject_interp((data, _wcs(header)), target, shape_out=shape, order=order)


def _effective_ml(color_mag: float) -> float:
    return 10.0 ** (-0.339 * color_mag - 0.336)


def _relative(candidate: float, reference: float) -> float:
    return candidate / reference - 1.0


def analyze(object_id: str) -> dict[str, object]:
    f36, h36 = _read(IRAC1_ROOT / f"{object_id}__SINGS_IRAC1__STELLAR_IRAC1_FLUX.fits")
    w36, hw36 = _read(IRAC1_ROOT / f"{object_id}__SINGS_IRAC1__STELLAR_IRAC1_WEIGHT.fits")
    f45, h45 = _read(IRAC2_ROOT / f"{object_id}__SINGS_IRAC2__STELLAR_IRAC2_FLUX.fits")
    w45, hw45 = _read(IRAC2_ROOT / f"{object_id}__SINGS_IRAC2__STELLAR_IRAC2_WEIGHT.fits")
    stellar, hs = _read(S4G_ROOT / f"{object_id}__S4G_P5__STELLAR_MASS_MAP.fits")
    color, hc = _read(S4G_ROOT / f"{object_id}__S4G_P5__STELLAR_COLOR_MAP.fits")
    mask, hm = _read(S4G_ROOT / f"{object_id}__S4G_P5__STELLAR_ICA_MASK.fits")
    target = _wcs(hs)
    shape = stellar.shape
    f36, p36 = _reproject(f36, h36, target, shape, order="bilinear")
    w36, pw36 = _reproject(w36, hw36, target, shape, order="nearest-neighbor")
    f45, p45 = _reproject(f45, h45, target, shape, order="bilinear")
    w45, pw45 = _reproject(w45, hw45, target, shape, order="nearest-neighbor")
    mask_check, pm = _reproject(mask, hm, target, shape, order="nearest-neighbor")
    color_check, pc = _reproject(color, hc, target, shape, order="bilinear")
    valid = (
        np.isfinite(stellar)
        & np.isfinite(f36)
        & np.isfinite(f45)
        & np.isfinite(w36)
        & np.isfinite(w45)
        & np.isfinite(color_check)
        & (p36 > 0.999)
        & (pw36 > 0.999)
        & (p45 > 0.999)
        & (pw45 > 0.999)
        & (pm > 0.999)
        & (pc > 0.999)
        & (w36 > 0.0)
        & (w45 > 0.0)
        & (mask_check == 0.0)
        & (stellar > SOURCE_THRESHOLD_SIGMA * S4G_RMS_MJY_SR)
        & (f36 > 0.0)
        & (f45 > 0.0)
    )
    if np.count_nonzero(valid) < 1000:
        raise RuntimeError(f"too few benchmark pixels for {object_id}")
    sum36 = float(np.sum(f36[valid]))
    sum45 = float(np.sum(f45[valid]))
    observed_color = -2.5 * math.log10((sum36 / sum45) * (IRAC2_ZERO_JY / IRAC1_ZERO_JY))
    global_ml = _effective_ml(observed_color)
    reference_fixed = float(np.sum(stellar[valid]) * LUMINOSITY_PER_MJY_SR * FIXED_ML)
    reference_ml = np.full_like(stellar, FIXED_ML)
    clean_color_valid = valid & (color_check >= -0.15) & (color_check <= -0.02)
    reference_ml[clean_color_valid] = 10.0 ** (
        -0.339 * color_check[clean_color_valid] - 0.336
    )
    reference_color = float(
        np.sum(stellar[valid] * reference_ml[valid]) * LUMINOSITY_PER_MJY_SR
    )
    candidate_fixed = sum36 * LUMINOSITY_PER_MJY_SR * FIXED_ML
    candidate_global_color = sum36 * LUMINOSITY_PER_MJY_SR * global_ml
    return {
        "object_id": object_id,
        "comparison_pixels": int(np.count_nonzero(valid)),
        "observed_global_color_3p6_minus_4p5_mag": observed_color,
        "published_effective_ml": global_ml,
        "reference_clean_fixed0p6_mass_proxy": reference_fixed,
        "reference_clean_color_mass_proxy": reference_color,
        "candidate_raw_fixed0p6_mass_proxy": candidate_fixed,
        "candidate_raw_global_color_mass_proxy": candidate_global_color,
        "raw_fixed0p6_vs_clean_fixed0p6_fractional_error": _relative(
            candidate_fixed, reference_fixed
        ),
        "raw_global_color_vs_clean_fixed0p6_fractional_error": _relative(
            candidate_global_color, reference_fixed
        ),
        "raw_global_color_vs_clean_color_fractional_error": _relative(
            candidate_global_color, reference_color
        ),
        "clean_color_ml_pixel_fraction": float(np.mean(clean_color_valid[valid])),
        "sings_backgrounds_mjy_sr": {
            "irac1": float(h36["BACKGRND"]),
            "irac2": float(h45["BACKGRND"]),
        },
    }


def _content_sha256(payload: dict[str, object]) -> str:
    clean = dict(payload)
    clean.pop("content_sha256", None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    records = [analyze(object_id) for object_id in OBJECTS]
    payload: dict[str, object] = {
        "schema": "invariant-work-rg-sings-color-ml-overlap-source-comparison-1.0",
        "purpose": "Source-only validation of published fixed and observed-global-color SINGS stellar-mass fallback transformations.",
        "paper_bindings": {
            "sings_dr5": "https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/doc/sings_fifth_delivery_v2.pdf",
            "s4g_p5_and_effective_ml": "https://arxiv.org/abs/1410.0009",
            "old_stellar_ml": "https://arxiv.org/abs/1402.5210",
        },
        "formulae": {
            "color_mag": "-2.5*log10((sum_F36/sum_F45)*(F0_45/F0_36))",
            "effective_ml": "10**(-0.339*observed_global_color-0.336)",
            "fixed_ml": FIXED_ML,
            "luminosity_lsun_pc2_per_mjy_sr": LUMINOSITY_PER_MJY_SR,
            "irac_zero_flux_jy": {"3p6": IRAC1_ZERO_JY, "4p5": IRAC2_ZERO_JY},
        },
        "scope": "Observed-global-color M/L is tested as a total-mass fallback. It does not remove local dust morphology and is not promoted to an ICA-equivalent spatial mass map.",
        "records": records,
        "response_files_opened": 0,
        "response_rows_opened": 0,
        "scores_computed": 0,
    }
    payload["max_abs_global_color_vs_clean_color_fractional_error"] = max(
        abs(float(row["raw_global_color_vs_clean_color_fractional_error"])) for row in records
    )
    payload["content_sha256"] = _content_sha256(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing benchmark changed")
    if not OUTPUT.exists():
        OUTPUT.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
