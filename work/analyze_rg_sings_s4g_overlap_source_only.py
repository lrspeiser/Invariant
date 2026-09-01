"""Source-only SINGS-to-S4G stellar-light comparison; no rotation responses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SINGS_ROOT = ROOT / "work/private/open-gravity-rg-sings-s4g-overlap-source-only-v1"
SOURCE_ROOT = ROOT / "work/private/open-gravity-rg-12gal-source-only-v1"
OUTPUT = ROOT / "work/rg-sings-s4g-overlap-source-comparison-v3.json"
OBJECTS = ("NGC2976", "NGC3198", "NGC3521")
S4G_RMS_MJY_SR = 0.0072
SOURCE_THRESHOLD_SIGMA = 5.0


def _content_sha256(payload: dict[str, object]) -> str:
    import hashlib

    clean = dict(payload)
    clean.pop("content_sha256", None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> tuple[np.ndarray, fits.Header]:
    return np.asarray(fits.getdata(path, memmap=False), dtype=np.float64), fits.getheader(path)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"q{int(q * 100):02d}": float(np.quantile(values, q))
        for q in (0.05, 0.16, 0.50, 0.84, 0.95)
    }


def _metrics(reference: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    positive = (reference > 0.0) & (observed > 0.0)
    x = reference[positive]
    y = observed[positive]
    ratio = y / x
    residual = y - x
    return {
        "comparison_pixels": int(x.size),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_r": float(spearmanr(x, y).statistic),
        "through_origin_observed_per_reference_slope": float(np.dot(x, y) / np.dot(x, x)),
        "integrated_observed_per_reference_flux_ratio": float(np.sum(y) / np.sum(x)),
        "pixel_observed_per_reference_ratio_quantiles": _quantiles(ratio),
        "residual_mjy_sr_quantiles": _quantiles(residual),
        "observed_ge_reference_fraction": float(np.mean(y >= x)),
    }


def _wcs(header: fits.Header, *, use_sip: bool) -> WCS:
    value = WCS(header, relax=True)
    if not use_sip:
        value.sip = None
    return value


def analyze_object(object_id: str, *, use_sip: bool) -> dict[str, object]:
    sings, sings_header = _read(
        SINGS_ROOT / f"{object_id}__SINGS_IRAC1__STELLAR_IRAC1_FLUX.fits"
    )
    weight, weight_header = _read(
        SINGS_ROOT / f"{object_id}__SINGS_IRAC1__STELLAR_IRAC1_WEIGHT.fits"
    )
    stellar, stellar_header = _read(
        SOURCE_ROOT / f"{object_id}__S4G_P5__STELLAR_MASS_MAP.fits"
    )
    ica_mask, mask_header = _read(
        SOURCE_ROOT / f"{object_id}__S4G_P5__STELLAR_ICA_MASK.fits"
    )
    nonstellar, nonstellar_header = _read(
        SINGS_ROOT / f"{object_id}__S4G_P5__NONSTELLAR_MAP.fits"
    )
    if nonstellar.shape != stellar.shape:
        raise RuntimeError(f"S4G P5 component shape mismatch for {object_id}")
    target = _wcs(stellar_header, use_sip=use_sip)
    if _wcs(nonstellar_header, use_sip=use_sip).wcs.compare(target.wcs) == 0:
        raise RuntimeError(f"S4G P5 component WCS mismatch for {object_id}")
    target_wcs = target
    shape = stellar.shape
    sings_on_s4g, footprint = reproject_interp(
        (sings, _wcs(sings_header, use_sip=use_sip)),
        target_wcs,
        shape_out=shape,
        order="bilinear",
    )
    weight_on_s4g, weight_footprint = reproject_interp(
        (weight, _wcs(weight_header, use_sip=use_sip)),
        target_wcs,
        shape_out=shape,
        order="nearest-neighbor",
    )
    mask_on_s4g, mask_footprint = reproject_interp(
        (ica_mask, _wcs(mask_header, use_sip=use_sip)),
        target_wcs,
        shape_out=shape,
        order="nearest-neighbor",
    )
    valid = (
        np.isfinite(stellar)
        & np.isfinite(sings_on_s4g)
        & np.isfinite(weight_on_s4g)
        & (footprint > 0.999)
        & (weight_footprint > 0.999)
        & (mask_footprint > 0.999)
        & (weight_on_s4g > 0.0)
        & (mask_on_s4g == 0.0)
        & (stellar > SOURCE_THRESHOLD_SIGMA * S4G_RMS_MJY_SR)
    )
    stellar_values = stellar[valid]
    reconstructed_values = (stellar + nonstellar)[valid]
    sings_values = sings_on_s4g[valid]
    if np.count_nonzero((stellar_values > 0.0) & (sings_values > 0.0)) < 1000:
        raise RuntimeError(f"too few comparison pixels for {object_id}")
    return {
        "object_id": object_id,
        "wcs_mode": "HEADER_SIP_SENSITIVITY" if use_sip else "CORE_TAN_PRIMARY",
        "source_threshold_mjy_sr": SOURCE_THRESHOLD_SIGMA * S4G_RMS_MJY_SR,
        "sings_vs_s4g_stellar": _metrics(stellar_values, sings_values),
        "sings_vs_s4g_stellar_plus_nonstellar": _metrics(
            reconstructed_values, sings_values
        ),
        "sings_header_background_subtracted": bool(sings_header["BACK_SUB"]),
        "sings_header_background_mjy_sr": float(sings_header["BACKGRND"]),
        "sings_unit": str(sings_header["BUNIT"]).strip(),
        "s4g_unit": str(stellar_header["BUNIT"]).strip(),
    }


def main() -> None:
    records = [
        analyze_object(object_id, use_sip=use_sip)
        for object_id in OBJECTS
        for use_sip in (False, True)
    ]
    payload: dict[str, object] = {
        "schema": "invariant-work-rg-sings-s4g-overlap-source-comparison-3.0",
        "purpose": "Source-only comparison of raw SINGS IRAC1 intensity with independent S4G P5 ICA-cleaned old-stellar intensity.",
        "interpretation": "SINGS includes nonstellar 3.6 micron emission that S4G P5 removes; equality is not expected and differences are not gravity evidence.",
        "reconstruction_identity_under_test": "S4G P5 stellar + nonstellar should reconstruct the S4G input 3.6 micron field; agreement with independently reduced SINGS is a cross-pipeline source benchmark, not exact identity.",
        "paper_bindings": {
            "sings_dr5": "https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/doc/sings_fifth_delivery_v2.pdf",
            "s4g_p5": "https://arxiv.org/abs/1410.0009",
            "fixed_ml_3p6": "https://arxiv.org/abs/1402.5210",
        },
        "conversion": {
            "luminosity_lsun_pc2_per_mjy_sr": 704.04,
            "fixed_ml_msun_per_lsun": 0.6,
            "mass_surface_density_msun_pc2_per_mjy_sr": 422.424,
        },
        "benchmark_contract": {
            "source_threshold_sigma": SOURCE_THRESHOLD_SIGMA,
            "s4g_rms_mjy_sr": S4G_RMS_MJY_SR,
            "reprojection": "bilinear SINGS flux and nearest-neighbor weight/mask onto exact S4G P5 WCS",
            "wcs_modes": {
                "primary": "core TAN with inherited SIP coefficients disabled because the delivered drizzled mosaics omit the -SIP CTYPE declaration",
                "sensitivity": "apply inherited SIP coefficients despite the missing -SIP CTYPE declaration",
            },
            "validity": "positive finite overlap with SINGS weight>0, S4G ICAmask==0, and S4G stellar intensity above 5 sigma",
        },
        "records": records,
        "response_files_opened": 0,
        "response_rows_opened": 0,
        "scores_computed": 0,
    }
    payload["content_sha256"] = _content_sha256(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing comparison changed")
    if not OUTPUT.exists():
        OUTPUT.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
