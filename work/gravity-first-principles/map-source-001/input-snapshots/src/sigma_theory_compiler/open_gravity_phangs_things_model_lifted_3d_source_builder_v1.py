"""Build response-independent model-lifted 2.5-D/3-D galaxy source profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy import log
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.signal import fftconvolve
from scipy.special import iv, kv

CONFIG_PATH = Path("configs/open_gravity_phangs_things_model_lifted_3d_source_builder_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_phangs_things_model_lifted_3d_source_builder_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_phangs_things_model_lifted_3d_source_builder_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-phangs-things-model-lifted-3d-source-builder-v1/receipt.json"
)

_CONFIG_RAW_SHA256 = "9360749f561fbe783d7a21d1b179ab947fa2f29214bb4425495dddd3f7465ede"
_CONFIG_CONTENT_SHA256 = "846e77923a5fde4d4fa95d6707071265b95cdc88bb3f4e9b3c8b3d6bb7d15638"
_MODULE_SEMANTIC_SHA256 = "639b3f982a1c973b83d2108c60ebf7ed90df5aa08782d40346dbff2b05b36087"
_TEST_RAW_SHA256 = "a90f1c45575320c85a751e41ebe26381ca2caa25953a4bf4d156de3f1b15afbb"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f* ]{64}("\r?\n)')
_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-source-builder-1.0"
_PROFILE_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-source-profiles-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-source-builder-receipt-1.0"


class SourceBuilderError(RuntimeError):
    """Raised when the frozen source-building contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceBuilderError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    candidate = (root / relative).resolve()
    _require(candidate == root or root in candidate.parents, "path escaped repository")
    return candidate


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, count = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBuilderError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_config(config: dict[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "MODEL_LIFTED_2P5D_SOURCE_PROFILES_DEVELOPMENT_ONLY",
        "status changed",
    )
    _require(
        [row["object_id"] for row in config["objects"]] == ["NGC2903", "NGC3351", "NGC3627"],
        "objects changed",
    )
    cells = config["cell_contract"]
    _require(cells["primary_cartesian_cells_per_object"] == 72, "cell count changed")
    _require(cells["total_cells_per_object"] == 75, "control count changed")
    _require(cells["total_cells"] == 225, "total cell count changed")
    _require(cells["response_based_cell_selection"] is False, "response selection enabled")
    _require(cells["retain_every_failure"] is True, "failure retention lost")
    anchors = config["published_anchor_contract"]
    _require(len(anchors["datasets_and_methods"]) == 6, "published anchors changed")
    _require(
        all(anchors["mandatory_synthetic_and_published_benchmarks"].values()), "benchmark removed"
    )
    boundary = config["scientific_boundary"]
    _require(boundary["source_files_opened"] == 21, "source count hidden")
    _require(boundary["source_bytes_opened"] == 74_030_400, "source bytes hidden")
    _require(boundary["response_rows_opened"] == 0, "response rows changed")
    _require(boundary["scores_computed"] == 0 and boundary["models_fit"] == 0, "science executed")
    _require(boundary["network_calls"] == 0, "network enabled")
    _require(boundary["development_only"] is True, "development boundary lost")
    claims = config["claims"]
    _require(claims["model_lifted_source_profiles_derived"] is True, "source claim lost")
    _require(claims["source_systematics_enumerated"] is True, "systematics claim lost")
    _require(
        not any(
            value
            for key, value in claims.items()
            if key not in {"model_lifted_source_profiles_derived", "source_systematics_enumerated"}
        ),
        "claim ceiling exceeded",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _validate_package_files() -> None:
    _require(
        module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256, "module changed"
    )
    _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def _load_acquisition(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["acquisition_binding"]
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(binding[f"{role}_path"])
        _require(file_sha256(path) == binding[f"{role}_raw_sha256"], f"acquisition {role} changed")
    acquisition_config = _read_json(_repo_path(binding["config_path"]), "acquisition config")
    receipt = _read_json(_repo_path(binding["receipt_path"]), "acquisition receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"],
        "acquisition receipt content changed",
    )
    _require(
        receipt["inventory_summary"] == acquisition_config["inventory_contract"],
        "acquisition inventory mismatch",
    )
    return acquisition_config, receipt


def _fits_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdus:
        _require(len(hdus) == 1 and hdus[0].data is not None, "FITS image inventory changed")
        data = np.asarray(hdus[0].data, dtype=np.float64).squeeze()
        _require(data.ndim == 2, "FITS image is not two-dimensional after singleton removal")
        return data, hdus[0].header.copy()


def _wcs(header: fits.Header, *, use_sip: bool) -> WCS:
    previous_level = log.level
    try:
        log.setLevel("ERROR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            world = WCS(header).celestial
    finally:
        log.setLevel(previous_level)
    if not use_sip:
        world.sip = None
    return world


def _disk_grid(
    metadata: dict[str, Any], n: int, box_kpc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    dx_pc = box_kpc * 1000.0 / n
    axis_pc = (np.arange(n, dtype=np.float64) + 0.5 - n / 2.0) * dx_pc
    x_pc, y_pc = np.meshgrid(axis_pc, axis_pc)
    pa = math.radians(float(metadata["position_angle_deg"]))
    inclination = math.radians(float(metadata["inclination_deg"]))
    projected_minor_pc = y_pc * math.cos(inclination)
    east_pc = x_pc * math.sin(pa) + projected_minor_pc * math.cos(pa)
    north_pc = x_pc * math.cos(pa) - projected_minor_pc * math.sin(pa)
    distance_pc = float(metadata["distance_mpc"]) * 1_000_000.0
    center = SkyCoord(float(metadata["ra_deg"]) * u.deg, float(metadata["dec_deg"]) * u.deg)
    sky = center.spherical_offsets_by(
        (east_pc / distance_pc) * u.rad, (north_pc / distance_pc) * u.rad
    )
    return x_pc, y_pc, np.asarray(sky.ra.deg), np.asarray(sky.dec.deg), dx_pc


def _nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    _require(values.shape == valid.shape and bool(valid.any()), "source mask has no valid pixels")
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return np.asarray(values[tuple(indices)], dtype=np.float64)


def _sample_image(
    data: np.ndarray,
    header: fits.Header,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    use_sip: bool,
    order: int,
) -> np.ndarray:
    world = _wcs(header, use_sip=use_sip)
    px, py = world.all_world2pix(ra_deg, dec_deg, 0)
    sampled = map_coordinates(
        data,
        [py, px],
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return np.asarray(sampled, dtype=np.float64)


def _additional_beam_sigma_pixels(
    target_fwhm_pc: float, native_fwhm_arcsec: float, distance_mpc: float, dx_pc: float
) -> float:
    native_pc = distance_mpc * 1_000_000.0 * math.radians(native_fwhm_arcsec / 3600.0)
    additional_pc = math.sqrt(max(target_fwhm_pc * target_fwhm_pc - native_pc * native_pc, 0.0))
    return additional_pc / (2.0 * math.sqrt(2.0 * math.log(2.0)) * dx_pc)


def _smooth_to_beam(values: np.ndarray, sigma_pixels: float) -> np.ndarray:
    safe = np.where(np.isfinite(values), values, 0.0)
    if sigma_pixels <= 1.0e-12:
        return safe
    return gaussian_filter(safe, sigma=sigma_pixels, mode="constant", cval=0.0, truncate=5.0)


def _things_beam(header: fits.Header) -> tuple[float, float, float]:
    pattern = re.compile(
        r"CLEAN BMAJ=\s*([0-9.+\-Ee]+) BMIN=\s*([0-9.+\-Ee]+) BPA=\s*([0-9.+\-Ee]+)"
    )
    history = header.get("HISTORY", [])
    if isinstance(history, str):
        history = [history]
    matches = [match for row in history if (match := pattern.search(str(row)))]
    _require(len(matches) == 1, "THINGS beam history changed")
    return tuple(float(value) for value in matches[0].groups())


def _source_paths(acquisition_config: dict[str, Any]) -> dict[tuple[str, str], Path]:
    root = _repo_path(acquisition_config["private_source_root"])
    rows: dict[tuple[str, str], Path] = {}
    for source in _read_json(
        _repo_path(acquisition_config["predecessor"]["config_path"]), "source preflight"
    )["source_files"]:
        name = f"{source['object_id']}__{source['survey']}__{source['role']}.fits"
        path = (root / name).resolve()
        _require(path.parent == root and path.is_file(), "private source file missing")
        rows[(source["object_id"], source["role"])] = path
    _require(len(rows) == 21, "source path count changed")
    return rows


def _load_object_images(
    object_id: str, paths: dict[tuple[str, str], Path]
) -> dict[str, tuple[np.ndarray, fits.Header]]:
    roles = (
        "STELLAR_FLUX",
        "STELLAR_ICA_MASK",
        "STELLAR_COLOR",
        "HI_MOM0_NATURAL_SENSITIVITY",
        "HI_MOM0_ROBUST_PRIMARY",
        "CO21_BROAD_MOM0",
        "CO21_BROAD_EMOM0",
    )
    return {role: _fits_image(paths[(object_id, role)]) for role in roles}


def _surface_maps(
    config: dict[str, Any],
    metadata: dict[str, Any],
    images: dict[str, tuple[np.ndarray, fits.Header]],
    *,
    n: int,
    box_kpc: float,
    beam: str,
    use_sip: bool,
) -> dict[str, Any]:
    x_pc, y_pc, ra_deg, dec_deg, dx_pc = _disk_grid(metadata, n, box_kpc)
    inclination = math.radians(float(metadata["inclination_deg"]))
    cos_i = math.cos(inclination)
    transform = config["map_transform"]
    conversion = config["mass_conversion"]

    stellar_raw, stellar_header = images["STELLAR_FLUX"]
    mask_raw, _ = images["STELLAR_ICA_MASK"]
    color_raw, color_header = images["STELLAR_COLOR"]
    valid_stellar = (mask_raw == 0.0) & np.isfinite(stellar_raw)
    stellar_filled = _nearest_fill(stellar_raw, valid_stellar)
    color_valid = valid_stellar & np.isfinite(color_raw)
    color_filled = _nearest_fill(color_raw, color_valid)
    stellar_intensity = _sample_image(
        stellar_filled, stellar_header, ra_deg, dec_deg, use_sip=use_sip, order=1
    )
    color = _sample_image(color_filled, color_header, ra_deg, dec_deg, use_sip=use_sip, order=1)
    stellar_intensity = np.maximum(
        np.where(np.isfinite(stellar_intensity), stellar_intensity, 0.0), 0.0
    )

    hi_role = (
        "HI_MOM0_ROBUST_PRIMARY" if beam == "ROBUST_PRIMARY" else "HI_MOM0_NATURAL_SENSITIVITY"
    )
    hi_raw, hi_header = images[hi_role]
    hi_map = _sample_image(hi_raw, hi_header, ra_deg, dec_deg, use_sip=False, order=1)
    beam_major_deg, beam_minor_deg, _ = _things_beam(hi_header)
    target_fwhm_pc = float(metadata["distance_mpc"]) * 1_000_000.0 * math.radians(beam_major_deg)

    co_raw, co_header = images["CO21_BROAD_MOM0"]
    eco_raw, eco_header = images["CO21_BROAD_EMOM0"]
    co_map = _sample_image(co_raw, co_header, ra_deg, dec_deg, use_sip=False, order=1)
    eco_map = _sample_image(eco_raw, eco_header, ra_deg, dec_deg, use_sip=False, order=1)
    co_map = np.where(
        np.isfinite(co_map) & np.isfinite(eco_map) & (co_map > 0.0) & (co_map >= 3.0 * eco_map),
        co_map,
        0.0,
    )

    stellar_sigma_pixels = _additional_beam_sigma_pixels(
        target_fwhm_pc,
        float(transform["s4g_native_fwhm_arcsec"]),
        float(metadata["distance_mpc"]),
        dx_pc,
    )
    co_native_arcsec = float(co_header["BMAJ"]) * 3600.0
    co_sigma_pixels = _additional_beam_sigma_pixels(
        target_fwhm_pc,
        co_native_arcsec,
        float(metadata["distance_mpc"]),
        dx_pc,
    )
    stellar_intensity = _smooth_to_beam(stellar_intensity, stellar_sigma_pixels)
    color = _smooth_to_beam(color, stellar_sigma_pixels)
    co_map = _smooth_to_beam(co_map, co_sigma_pixels)

    stellar_factor = float(conversion["stellar_lsun_pc2_per_mjy_sr"]) * cos_i
    stellar_fixed = stellar_intensity * stellar_factor * float(conversion["stellar_ml_fixed"])
    ml_color = np.full_like(color, float(conversion["stellar_color_fallback_ml"]))
    color_ok = (color >= float(conversion["stellar_color_valid_min_mag"])) & (
        color <= float(conversion["stellar_color_valid_max_mag"])
    )
    ml_color[color_ok] = 10.0 ** (-0.339 * color[color_ok] - 0.336)
    stellar_color = stellar_intensity * stellar_factor * ml_color

    nu_ghz = float(hi_header["RESTFREQ"]) / 1.0e9
    brightness_per_jy = float(conversion["hi_brightness_temperature_constant"]) / (
        nu_ghz * nu_ghz * beam_major_deg * 3600.0 * beam_minor_deg * 3600.0
    )
    hi_k_kms = (
        np.maximum(np.where(np.isfinite(hi_map), hi_map, 0.0), 0.0)
        * float(conversion["hi_jybeam_mps_to_jybeam_kms"])
        * brightness_per_jy
    )
    hi_sigma = (
        hi_k_kms
        * float(conversion["hi_column_per_k_kms_cm2"])
        / float(conversion["hi_column_per_msun_pc2_cm2"])
        * float(conversion["helium_factor"])
        * cos_i
    )
    co_sigma = (
        co_map
        * float(conversion["co_alpha_co10_with_helium"])
        / float(conversion["co_r21"])
        * cos_i
    )
    return {
        "x_pc": x_pc,
        "y_pc": y_pc,
        "dx_pc": dx_pc,
        "target_fwhm_pc": target_fwhm_pc,
        "stellar_fixed": stellar_fixed,
        "stellar_color": stellar_color,
        "hi": hi_sigma,
        "co": co_sigma,
        "ml_color_fraction": float(np.count_nonzero(color_ok) / color_ok.size),
    }


def _half_mass_radius_pc(
    sigma: np.ndarray, x_pc: np.ndarray, y_pc: np.ndarray, dx_pc: float
) -> float:
    radius = np.hypot(x_pc, y_pc).ravel()
    mass = (np.maximum(sigma, 0.0) * dx_pc * dx_pc).ravel()
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(mass[order])
    _require(cumulative[-1] > 0.0, "stellar mass vanished")
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(radius[order[index]])


def vertical_kernel(
    n: int, dx_pc: float, height_pc: float, *, nodes: int, g_constant: float
) -> np.ndarray:
    _require(n >= 8 and dx_pc > 0.0 and height_pc >= 0.0 and nodes >= 8, "invalid kernel inputs")
    center = n // 2 - 1 if n % 2 == 0 else n // 2
    axis = (np.arange(n, dtype=np.float64) - center) * dx_pc
    xx, yy = np.meshgrid(axis, axis)
    radius2 = xx * xx + yy * yy
    epsilon2 = (dx_pc / math.sqrt(math.pi)) ** 2
    roots, weights = leggauss(nodes)
    uu = 0.5 * (roots + 1.0)
    ww = 0.5 * weights
    inverse_distance = np.zeros_like(radius2)
    for value, weight in zip(uu, ww, strict=True):
        z_pc = height_pc * np.arctanh(value)
        inverse_distance += weight / np.sqrt(radius2 + z_pc * z_pc + epsilon2)
    return -g_constant * inverse_distance


def potential_from_surface_density(
    config: dict[str, Any], sigma: np.ndarray, dx_pc: float, height_pc: float
) -> np.ndarray:
    gravity = config["vertical_and_gravity_model"]
    kernel = vertical_kernel(
        sigma.shape[0],
        dx_pc,
        height_pc,
        nodes=int(gravity["vertical_gauss_legendre_nodes"]),
        g_constant=float(gravity["newton_g_pc_kms2_msun"]),
    )
    mass_pixels = np.maximum(np.asarray(sigma, dtype=np.float64), 0.0) * dx_pc * dx_pc
    return fftconvolve(mass_pixels, kernel, mode="same")


def _sample_ring(
    values: np.ndarray, radius_pc: float, dx_pc: float, angles: np.ndarray
) -> np.ndarray:
    n = values.shape[0]
    x = radius_pc * np.cos(angles) / dx_pc + n / 2.0 - 0.5
    y = radius_pc * np.sin(angles) / dx_pc + n / 2.0 - 0.5
    return map_coordinates(values, [y, x], order=1, mode="constant", cval=np.nan, prefilter=False)


def radial_profile(
    config: dict[str, Any],
    phi: np.ndarray,
    sigma: np.ndarray,
    dx_pc: float,
    hstar_pc: float,
    hgas_pc: float,
    stellar_sigma: np.ndarray,
    gas_sigma: np.ndarray,
) -> list[dict[str, float]]:
    gravity = config["vertical_and_gravity_model"]
    pc_m = float(gravity["pc_m"])
    c_km_s = float(gravity["c_km_s"])
    dphi_dy, dphi_dx = np.gradient(phi, dx_pc, dx_pc, edge_order=2)
    d2phi_dyy, d2phi_dyx = np.gradient(dphi_dy, dx_pc, dx_pc, edge_order=2)
    d2phi_dxy, d2phi_dxx = np.gradient(dphi_dx, dx_pc, dx_pc, edge_order=2)
    tidal = (
        np.sqrt(d2phi_dxx**2 + d2phi_dyy**2 + d2phi_dxy**2 + d2phi_dyx**2) * 1.0e6 / (pc_m * pc_m)
    )
    rho = stellar_sigma / (2.0 * hstar_pc) + gas_sigma / (2.0 * hgas_pc)
    angles = np.linspace(0.0, 2.0 * math.pi, int(gravity["azimuth_sample_count"]), endpoint=False)
    radii_kpc = np.linspace(
        float(gravity["radial_min_kpc"]),
        float(gravity["radial_max_kpc"]),
        int(gravity["radial_sample_count"]),
    )
    rows: list[dict[str, float]] = []
    for radius_kpc in radii_kpc:
        radius_pc = radius_kpc * 1000.0
        gx = _sample_ring(dphi_dx, radius_pc, dx_pc, angles)
        gy = _sample_ring(dphi_dy, radius_pc, dx_pc, angles)
        radial_gradient = gx * np.cos(angles) + gy * np.sin(angles)
        finite = np.isfinite(radial_gradient)
        _require(int(finite.sum()) >= len(angles) * 3 // 4, "radial ring escaped source box")
        radial_gradient = radial_gradient[finite]
        selected_angles = angles[finite]
        mean_gradient = float(np.mean(radial_gradient))
        g_m_s2 = max(mean_gradient, 0.0) * 1.0e6 / pc_m
        normalization = max(abs(mean_gradient), 1.0e-30)
        fourier = 0.0
        for mode in gravity["nonaxisymmetric_modes"]:
            coefficient = np.mean(radial_gradient * np.exp(-1j * int(mode) * selected_angles))
            fourier += float(abs(coefficient) ** 2)
        rows.append(
            {
                "radius_kpc": float(radius_kpc),
                "g_b_m_s2": g_m_s2,
                "potential_depth_c2": float(
                    np.nanmean(np.abs(_sample_ring(phi, radius_pc, dx_pc, angles)))
                    / (c_km_s * c_km_s)
                ),
                "tidal_frobenius_s2": float(
                    np.nanmean(_sample_ring(tidal, radius_pc, dx_pc, angles))
                ),
                "rho_midplane_msun_pc3": float(
                    np.nanmean(_sample_ring(rho, radius_pc, dx_pc, angles))
                ),
                "sigma_b_msun_pc2": float(
                    np.nanmean(_sample_ring(sigma, radius_pc, dx_pc, angles))
                ),
                "radial_force_rms_asymmetry": float(np.std(radial_gradient) / normalization),
                "radial_force_fourier_m1_m4": float(math.sqrt(fourier) / normalization),
            }
        )
    return rows


def _cell_summary(
    config: dict[str, Any],
    profile: list[dict[str, float]],
    *,
    cell_id: str,
    metadata: dict[str, Any],
    maps: dict[str, Any],
    stellar_sigma: np.ndarray,
    hi_sigma: np.ndarray,
    co_sigma: np.ndarray,
    hstar_pc: float,
    hgas_pc: float,
) -> dict[str, Any]:
    target = float(config["vertical_and_gravity_model"]["matched_acceleration_m_s2"])
    matched = min(profile, key=lambda row: abs(math.log10(max(row["g_b_m_s2"], 1.0e-30) / target)))
    dx_pc = float(maps["dx_pc"])
    return {
        "cell_id": cell_id,
        "object_id": metadata["object_id"],
        "grid_pixels": int(stellar_sigma.shape[0]),
        "dx_pc": dx_pc,
        "target_fwhm_pc": float(maps["target_fwhm_pc"]),
        "hstar_pc": hstar_pc,
        "hgas_pc": hgas_pc,
        "stellar_mass_msun": float(np.sum(stellar_sigma) * dx_pc * dx_pc),
        "hi_helium_mass_msun": float(np.sum(hi_sigma) * dx_pc * dx_pc),
        "co_helium_mass_msun": float(np.sum(co_sigma) * dx_pc * dx_pc),
        "matched_acceleration": matched,
        "profile_sha256": content_sha256(profile),
    }


def _build_cell(
    config: dict[str, Any],
    metadata: dict[str, Any],
    maps: dict[str, Any],
    *,
    cell_id: str,
    stellar_ml: str,
    co_source: str,
    hstar_pc: float,
    hgas_pc: float,
    cache: dict[tuple[str, float], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    stellar_sigma = maps["stellar_fixed"] if stellar_ml == "FIXED_0P6" else maps["stellar_color"]
    hi_sigma = maps["hi"]
    co_sigma = maps["co"] if co_source == "WITH_CO" else np.zeros_like(maps["co"])
    dx_pc = float(maps["dx_pc"])

    def component(label: str, sigma: np.ndarray, height: float) -> np.ndarray:
        key = (label, float(height))
        if key not in cache:
            cache[key] = potential_from_surface_density(config, sigma, dx_pc, height)
        return cache[key]

    phi = component(f"star:{stellar_ml}", stellar_sigma, hstar_pc) + component(
        "hi", hi_sigma, hgas_pc
    )
    if co_source == "WITH_CO":
        phi = phi + component("co", co_sigma, hgas_pc)
    gas_sigma = hi_sigma + co_sigma
    total_sigma = stellar_sigma + gas_sigma
    profile = radial_profile(
        config, phi, total_sigma, dx_pc, hstar_pc, hgas_pc, stellar_sigma, gas_sigma
    )
    summary = _cell_summary(
        config,
        profile,
        cell_id=cell_id,
        metadata=metadata,
        maps=maps,
        stellar_sigma=stellar_sigma,
        hi_sigma=hi_sigma,
        co_sigma=co_sigma,
        hstar_pc=hstar_pc,
        hgas_pc=hgas_pc,
    )
    return summary, profile


def _benchmark_report(config: dict[str, Any]) -> dict[str, Any]:
    gravity = config["vertical_and_gravity_model"]
    g_constant = float(gravity["newton_g_pc_kms2_msun"])
    nodes = int(gravity["vertical_gauss_legendre_nodes"])
    _roots, weights = leggauss(nodes)
    vertical_normalization = float(np.sum(weights) / 2.0)

    n = 257
    dx_pc = 100.0
    axis = (np.arange(n, dtype=np.float64) - n // 2) * dx_pc
    x_pc, y_pc = np.meshgrid(axis, axis)
    radius = np.hypot(x_pc, y_pc)
    rd_pc = 2000.0
    sigma0 = 100.0
    exponential = sigma0 * np.exp(-radius / rd_pc)
    thin_phi = potential_from_surface_density(config, exponential, dx_pc, 1.0)
    thick_phi = potential_from_surface_density(config, exponential, dx_pc, 400.0)
    thin_gradient = np.gradient(thin_phi[n // 2], dx_pc, edge_order=2)
    thick_gradient = np.gradient(thick_phi[n // 2], dx_pc, edge_order=2)
    # The finite numerical source box is benchmarked over 0.5--3 R_d; farther
    # points would deliberately measure the source truncation control instead.
    test_radii = np.arange(1000.0, 6000.1, 500.0)
    pixel = np.rint(test_radii / dx_pc).astype(int) + n // 2
    numerical = thin_gradient[pixel]
    y = test_radii / (2.0 * rd_pc)
    v2 = (
        4.0
        * math.pi
        * g_constant
        * sigma0
        * rd_pc
        * y
        * y
        * (iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y))
    )
    analytic = v2 / test_radii
    relative = np.abs(numerical - analytic) / np.maximum(np.abs(analytic), 1.0e-30)
    freeman_max_relative = float(np.max(relative))
    peak_pixel = n // 2 + round(2.2 * rd_pc / dx_pc)
    thickness_ratio = float(thick_gradient[peak_pixel] / thin_gradient[peak_pixel])

    point = np.zeros((n, n), dtype=np.float64)
    point[n // 2, n // 2] = 1.0e10 / (dx_pc * dx_pc)
    point_phi = potential_from_surface_density(config, point, dx_pc, 1.0)
    far_r_pc = 8000.0
    far_pixel = n // 2 + round(far_r_pc / dx_pc)
    point_expected = -g_constant * 1.0e10 / far_r_pc
    point_relative = float(abs(point_phi[n // 2, far_pixel] - point_expected) / abs(point_expected))
    passed = {
        "sech2_vertical_kernel_normalization": abs(vertical_normalization - 1.0) < 1.0e-14,
        "point_mass_far_field_limit": point_relative < 0.03,
        "freeman_exponential_disk_thin_limit": freeman_max_relative < 0.08,
        "finite_thickness_monotone_suppression": 0.0 < thickness_ratio < 1.0,
    }
    _require(all(passed.values()), "published gravity benchmark failed")
    return {
        "passed": passed,
        "vertical_normalization": vertical_normalization,
        "point_mass_far_field_relative_error": point_relative,
        "freeman_max_relative_error": freeman_max_relative,
        "finite_thickness_force_ratio_at_2p2rd": thickness_ratio,
    }


def build_profiles(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    acquisition_config, acquisition_receipt = _load_acquisition(config)
    paths = _source_paths(acquisition_config)
    profiles: dict[str, Any] = {
        "schema": _PROFILE_SCHEMA,
        "package_id": config["package_id"],
        "acquisition_receipt_content_sha256": acquisition_receipt["content_sha256"],
        "benchmarks": _benchmark_report(config),
        "objects": [],
        "scientific_boundary": config["scientific_boundary"],
    }
    gravity = config["vertical_and_gravity_model"]
    middle_ratio = float(gravity["stellar_height_over_exponential_scale_cells"][1])
    for metadata in config["objects"]:
        images = _load_object_images(metadata["object_id"], paths)
        primary_maps = _surface_maps(
            config,
            metadata,
            images,
            n=int(config["map_transform"]["primary_grid_pixels"]),
            box_kpc=float(config["map_transform"]["primary_box_kpc"]),
            beam="ROBUST_PRIMARY",
            use_sip=False,
        )
        rhalf_pc = _half_mass_radius_pc(
            primary_maps["stellar_fixed"],
            primary_maps["x_pc"],
            primary_maps["y_pc"],
            float(primary_maps["dx_pc"]),
        )
        rd_pc = rhalf_pc / 1.678
        _require(
            500.0 < rhalf_pc < 15_000.0, "observed stellar half-mass radius outside sanity range"
        )
        object_cells: list[dict[str, Any]] = []
        object_profiles: list[dict[str, Any]] = []
        for beam in config["cell_contract"]["primary_cartesian_axes"]["beam"]:
            maps = (
                primary_maps
                if beam == "ROBUST_PRIMARY"
                else _surface_maps(
                    config,
                    metadata,
                    images,
                    n=int(config["map_transform"]["primary_grid_pixels"]),
                    box_kpc=float(config["map_transform"]["primary_box_kpc"]),
                    beam=beam,
                    use_sip=False,
                )
            )
            cache: dict[tuple[str, float], np.ndarray] = {}
            for stellar_ml in config["cell_contract"]["primary_cartesian_axes"]["stellar_ml"]:
                for co_source in config["cell_contract"]["primary_cartesian_axes"]["co_source"]:
                    for ratio in gravity["stellar_height_over_exponential_scale_cells"]:
                        hstar_pc = rd_pc * float(ratio)
                        for hgas_pc in gravity["gas_height_pc_cells"]:
                            cell_id = f"{beam}:{stellar_ml}:{co_source}:HS{float(ratio):.15g}:HG{float(hgas_pc):.15g}"
                            summary, profile = _build_cell(
                                config,
                                metadata,
                                maps,
                                cell_id=cell_id,
                                stellar_ml=stellar_ml,
                                co_source=co_source,
                                hstar_pc=hstar_pc,
                                hgas_pc=float(hgas_pc),
                                cache=cache,
                            )
                            object_cells.append(summary)
                            object_profiles.append({"cell_id": cell_id, "radial_profile": profile})
        controls = (
            (
                "COARSE_128_PRIMARY_PHYSICS",
                int(config["map_transform"]["coarse_grid_pixels"]),
                float(config["map_transform"]["primary_box_kpc"]),
                False,
            ),
            (
                "PADDED_512_PRIMARY_PHYSICS",
                int(config["map_transform"]["padded_grid_pixels"]),
                float(config["map_transform"]["padded_box_kpc"]),
                False,
            ),
            (
                "S4G_SIP_HEADER_SENSITIVITY_PRIMARY_PHYSICS",
                int(config["map_transform"]["primary_grid_pixels"]),
                float(config["map_transform"]["primary_box_kpc"]),
                True,
            ),
        )
        for cell_id, n, box_kpc, use_sip in controls:
            maps = _surface_maps(
                config,
                metadata,
                images,
                n=n,
                box_kpc=box_kpc,
                beam="ROBUST_PRIMARY",
                use_sip=use_sip,
            )
            summary, profile = _build_cell(
                config,
                metadata,
                maps,
                cell_id=cell_id,
                stellar_ml="FIXED_0P6",
                co_source="WITH_CO",
                hstar_pc=rd_pc * middle_ratio,
                hgas_pc=200.0,
                cache={},
            )
            object_cells.append(summary)
            object_profiles.append({"cell_id": cell_id, "radial_profile": profile})
        _require(
            len(object_cells) == int(config["cell_contract"]["total_cells_per_object"]),
            "object cell count changed",
        )
        primary_id = f"ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS{middle_ratio:.15g}:HG200"
        primary = next(row for row in object_cells if row["cell_id"] == primary_id)
        coarse = next(row for row in object_cells if row["cell_id"] == "COARSE_128_PRIMARY_PHYSICS")
        padded = next(row for row in object_cells if row["cell_id"] == "PADDED_512_PRIMARY_PHYSICS")
        potential_primary = float(primary["matched_acceleration"]["potential_depth_c2"])
        convergence = {
            "coarse_g_relative": abs(
                float(coarse["matched_acceleration"]["g_b_m_s2"])
                - float(primary["matched_acceleration"]["g_b_m_s2"])
            )
            / max(float(primary["matched_acceleration"]["g_b_m_s2"]), 1.0e-30),
            "padded_potential_relative": abs(
                float(padded["matched_acceleration"]["potential_depth_c2"]) - potential_primary
            )
            / max(potential_primary, 1.0e-30),
        }
        convergence["passed"] = bool(
            convergence["coarse_g_relative"] < 0.30
            and convergence["padded_potential_relative"] < 0.30
        )
        _require(convergence["passed"], f"source convergence failed for {metadata['object_id']}")
        _require(
            1.0e8 < primary["stellar_mass_msun"] < 3.0e11,
            "stellar mass outside published-source sanity range",
        )
        _require(
            1.0e7 < primary["hi_helium_mass_msun"] < 1.0e11,
            "HI mass outside published-source sanity range",
        )
        _require(
            50.0 < primary["target_fwhm_pc"] < 2500.0,
            "physical resolution outside survey sanity range",
        )
        profiles["objects"].append(
            {
                "object_id": metadata["object_id"],
                "rhalf_pc": rhalf_pc,
                "rd_pc": rd_pc,
                "primary_cell_id": primary_id,
                "primary_summary": primary,
                "convergence": convergence,
                "cell_summaries": object_cells,
                "cell_profiles": object_profiles,
                "cell_summary_root_sha256": content_sha256(object_cells),
                "cell_profile_root_sha256": content_sha256(object_profiles),
            }
        )
    _require(
        sum(len(row["cell_summaries"]) for row in profiles["objects"]) == 225,
        "global cell count changed",
    )
    profiles["cell_summary_root_sha256"] = content_sha256(
        [row["cell_summaries"] for row in profiles["objects"]]
    )
    profiles["cell_profile_root_sha256"] = content_sha256(
        [row["cell_profiles"] for row in profiles["objects"]]
    )
    profiles["content_sha256"] = content_sha256(profiles)
    return profiles


def _public_object_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": row["object_id"],
        "rhalf_pc": row["rhalf_pc"],
        "rd_pc": row["rd_pc"],
        "primary_cell_id": row["primary_cell_id"],
        "primary_summary": row["primary_summary"],
        "convergence": row["convergence"],
        "cell_summary_root_sha256": row["cell_summary_root_sha256"],
        "cell_profile_root_sha256": row["cell_profile_root_sha256"],
    }


def build_packet(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = build_profiles(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "SOURCE_PROFILES_READY_FOR_FIXED_THEORY_PREDICTIONS_DEVELOPMENT_ONLY",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "acquisition_binding": config["acquisition_binding"],
        "published_anchor_contract": config["published_anchor_contract"],
        "benchmarks": profiles["benchmarks"],
        "object_summaries": [_public_object_summary(row) for row in profiles["objects"]],
        "cell_count": 225,
        "cell_summary_root_sha256": profiles["cell_summary_root_sha256"],
        "cell_profile_root_sha256": profiles["cell_profile_root_sha256"],
        "private_profile_path": config["private_profile_output_path"],
        "private_profile_raw_sha256": hashlib.sha256(canonical_bytes(profiles)).hexdigest(),
        "private_profile_content_sha256": profiles["content_sha256"],
        "scientific_boundary": config["scientific_boundary"],
        "claims": config["claims"],
        "access_state": {
            "source_files_opened": 21,
            "source_bytes_opened": 74_030_400,
            "source_cells_derived": 225,
            "response_rows_opened": 0,
            "scores_computed": 0,
            "models_fit": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return profiles, receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent output differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_packet() -> str:
    config = load_config()
    profiles, receipt = build_packet(config)
    profile_status = _atomic_no_clobber(
        _repo_path(config["private_profile_output_path"]), canonical_bytes(profiles)
    )
    receipt_status = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return "CREATED" if "CREATED" in {profile_status, receipt_status} else "EXISTING_IDENTICAL"


def check_packet() -> str:
    config = load_config()
    profile_path = _repo_path(config["private_profile_output_path"])
    receipt_path = _repo_path(OUTPUT_PATH)
    _require(profile_path.is_file() and receipt_path.is_file(), "packet output missing")
    profiles, receipt = build_packet(config)
    _require(
        profile_path.read_bytes() == canonical_bytes(profiles), "private profiles do not rebuild"
    )
    _require(receipt_path.read_bytes() == canonical_bytes(receipt), "receipt does not rebuild")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_packet())
    elif args.command == "check":
        print(check_packet())
    else:
        config = load_config()
        if _repo_path(OUTPUT_PATH).exists():
            receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "decision": receipt["decision"],
                        "cells": receipt["cell_count"],
                        "responses": receipt["access_state"]["response_rows_opened"],
                    },
                    sort_keys=True,
                )
            )
        else:
            print(
                json.dumps({"status": config["status"], "decision": "NOT_WRITTEN"}, sort_keys=True)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
