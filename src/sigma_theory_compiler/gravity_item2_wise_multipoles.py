"""Acquire target-blind unWISE NEO11 W1 cutouts and extract 2D multipoles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.stats import spearmanr

from . import gravity_item2_shape_anisotropy as item2_v1
from .gravity_g1_pilot import _file_sha256, _load_json, _metric
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_SCHEMA = "invariant-gravity-roadmap-item2-wise-multipoles-config-1.0"
MANIFEST_SCHEMA = "invariant-gravity-item2-unwise-neo11-w1-manifest-1.0"
CONFIG_PATH = "configs/gravity_item2_wise_multipoles.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_item2_wise_multipoles.py"

FEATURE_COLUMNS = (
    "name",
    "ra_deg",
    "dec_deg",
    "distance_mpc",
    "inclination_deg",
    "effective_radius_kpc",
    "aperture_arcsec",
    "concentration_c20",
    "centroid_shift",
    "quadrupole_amplitude",
    "m3_aperture_amplitude",
    "m4_aperture_amplitude",
    "multipole_energy",
    "radial_quadrupole_coherence",
    "projected_outer_axis_ratio",
    "background",
    "background_mad_sigma",
    "positive_aperture_flux",
    "signed_aperture_flux",
    "central_flux_snr",
    "aperture_flux_snr",
    "brightest_pixel_flux_fraction",
    "aperture_pixels",
    "center_refinement_fraction",
    "measurement_valid",
    "image_quality_pass",
    "quality_failure_reason",
    "position_angle_deg",
    "image_sha256",
    "image_version",
    "s4g_family",
    "s4g_bar_ellipticity",
    "s4g_bar_quality",
)


class GravityItem2WiseMultipolesError(ValueError):
    """The WISE multipole extraction contract or its evidence changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_envelope(value: Mapping[str, Any], *, label: str) -> None:
    body = dict(value)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem2WiseMultipolesError(f"{label} content seal changed")


def load_extraction_config(root: Path) -> Mapping[str, Any]:
    """Validate the frozen target-blind extraction contract."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityItem2WiseMultipolesError("WISE multipole config schema changed")
    if config.get("status") != "exploratory_real_data_model_development":
        raise GravityItem2WiseMultipolesError("WISE multipole status changed")
    roadmap = config.get("roadmap_binding", {})
    if (
        roadmap.get("item_number") != 2
        or roadmap.get("item_title") != "Shape and anisotropy"
        or _file_sha256(root / str(roadmap.get("path"))) != roadmap.get("file_sha256")
    ):
        raise GravityItem2WiseMultipolesError("WISE multipole roadmap binding changed")

    predecessors = config.get("predecessor_bindings", {})
    for binding in predecessors.values():
        path = root / str(binding.get("path"))
        if _file_sha256(path) != binding.get("file_sha256"):
            raise GravityItem2WiseMultipolesError("WISE multipole predecessor file changed")
        receipt = _load_json(path)
        _verify_envelope(receipt, label="predecessor")
        if receipt.get("content_sha256") != binding.get("content_sha256") or receipt.get(
            "decision"
        ) != binding.get("required_decision"):
            raise GravityItem2WiseMultipolesError("WISE multipole predecessor content changed")

    sources = config.get("sources", {})
    bindings = [sources.get("sparc_exploration_membership", {})]
    bindings.append(sources.get("sparc_global_properties", {}))
    bindings.append(sources.get("clash_xray_morphology", {}))
    bindings.extend(sources.get("s4g_external_validation", {}).get("files", ()))
    for binding in bindings:
        if _file_sha256(root / str(binding.get("path"))) != binding.get("file_sha256"):
            raise GravityItem2WiseMultipolesError("WISE multipole source changed")
    membership = _load_json(root / str(sources["sparc_exploration_membership"]["path"]))
    _verify_envelope(membership, label="target-blind SPARC exploration membership")
    if (
        membership.get("claim") != sources["sparc_exploration_membership"].get("required_claim")
        or membership.get("counts", {}).get("exploration_galaxies") != 139
        or membership.get("counts", {}).get("confirmation_galaxies") != 0
    ):
        raise GravityItem2WiseMultipolesError("SPARC exploration membership changed")
    unwise = sources.get("unwise_w1", {})
    if (
        unwise.get("dataset_doi") != "10.26131/IRSA524"
        or unwise.get("band") != "W1"
        or unwise.get("layer") != "unwise-neo11"
        or float(unwise.get("pixel_scale_arcsec", 0.0)) != 2.75
        or unwise.get("cutout_service") != "https://www.legacysurvey.org/viewer/fits-cutout"
    ):
        raise GravityItem2WiseMultipolesError("unWISE source contract changed")
    authorization = config.get("authorization", {})
    if (
        authorization.get("network_acquisition_allowed") is not True
        or authorization.get("paid_model_calls_allowed") is not False
        or authorization.get("sparc_confirmation_evaluator_accesses_allowed") != 0
        or authorization.get("direct_lensing_likelihood_evaluations_allowed") != 0
        or authorization.get("sequential_G6_G7_G8_advanced") is not False
    ):
        raise GravityItem2WiseMultipolesError("WISE multipole authorization changed")
    extraction = config.get("image_extraction", {})
    if (
        float(extraction.get("aperture_effective_radius_multiple", 0.0)) != 2.5
        or float(extraction.get("positive_flux_winsor_quantile", 0.0)) != 0.995
        or extraction.get("target_fields_available_to_feature_computation") is not False
    ):
        raise GravityItem2WiseMultipolesError("WISE multipole extraction grammar changed")
    quality = extraction.get("quality_gate", {})
    if quality != {
        "maximum_center_refinement_fraction": 0.05,
        "minimum_central_flux_snr": 5,
        "minimum_aperture_flux_snr": 20,
        "minimum_aperture_pixels": 100,
        "minimum_concentration_c20": 0.03,
        "maximum_brightest_pixel_flux_fraction": 0.1,
    }:
        raise GravityItem2WiseMultipolesError("WISE multipole quality gate changed")
    population = config.get("population", {})
    if (
        int(population.get("expected_wise_cutouts", 0)) != 83
        or int(population.get("minimum_quality_wise_galaxies", 0)) != 40
        or int(population.get("expected_clash_clusters", 0)) != 20
    ):
        raise GravityItem2WiseMultipolesError("WISE multipole population changed")
    expected_models = [
        "constant",
        "linear_concentration",
        "linear_centroid_shift",
        "linear_quadrupole",
        "linear_m3",
        "linear_m4",
        "linear_multipole_energy",
        "quadratic_multipole_energy",
        "log_multipole_energy",
        "concentration_plus_energy",
        "energy_concentration_interaction",
        "all_multipoles",
        "linear_support_dimension_proxy",
        "support_plus_all_multipoles",
    ]
    models = config.get("models", ())
    if [str(row.get("id")) for row in models] != expected_models:
        raise GravityItem2WiseMultipolesError("WISE multipole model grammar changed")
    if any(
        row.get("qualifying") is not False
        for row in models
        if row.get("id")
        in {"constant", "linear_support_dimension_proxy", "support_plus_all_multipoles"}
    ):
        raise GravityItem2WiseMultipolesError("population proxy entered WISE admission")
    if config.get("claim_boundaries", {}).get("alternative_to_gr_established") is not False:
        raise GravityItem2WiseMultipolesError("WISE multipole contract overstates its claim")
    return config


def normalize_galaxy_name(name: str) -> str:
    """Normalize common NGC/UGC zero padding without using coordinates or targets."""

    compact = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    match = re.fullmatch(r"([A-Z]+)0*([0-9]+)([A-Z]*)", compact)
    if match is None:
        return compact
    return f"{match.group(1)}{int(match.group(2))}{match.group(3)}"


def _eligible_galaxies(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    sparc_path = root / str(config["sources"]["sparc_global_properties"]["path"])
    allowed_columns = tuple(config["sources"]["sparc_global_properties"]["allowed_columns"])
    if set(allowed_columns) != {"Name", "Dist", "i", "Reff", "_RA", "_DE"}:
        raise GravityItem2WiseMultipolesError("SPARC extraction columns changed")
    raw_rows = item2_v1._vizier_rows(sparc_path)
    rows = {
        str(raw["Name"]): {column: raw[column] for column in allowed_columns} for raw in raw_rows
    }
    membership = _load_json(root / str(config["sources"]["sparc_exploration_membership"]["path"]))
    exploration_names = {str(row["galaxy"]) for row in membership["galaxies"]}
    if len(exploration_names) != 139:
        raise GravityItem2WiseMultipolesError("SPARC exploration boundary changed")
    maximum = float(config["population"]["galaxy_inclination_maximum_deg"])
    eligible = []
    for name in sorted(exploration_names):
        raw = rows[name]
        allowed = {
            "dec_deg": float(raw["_DE"]),
            "distance_mpc": float(raw["Dist"]),
            "effective_radius_kpc": float(raw["Reff"]),
            "inclination_deg": float(raw["i"]),
            "name": name,
            "ra_deg": float(raw["_RA"]),
        }
        if allowed["inclination_deg"] <= maximum:
            eligible.append(allowed)
    if len(eligible) != int(config["population"]["expected_wise_cutouts"]):
        raise GravityItem2WiseMultipolesError("eligible WISE galaxy count changed")
    return eligible


def _parse_s4g_validation(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    files = config["sources"]["s4g_external_validation"]["files"]
    cvrhs_rows = item2_v1._vizier_rows(root / str(files[0]["path"]))
    families = {}
    for row in cvrhs_rows:
        if str(row.get("<F>", "")).strip():
            families[normalize_galaxy_name(row["Name"])] = float(row["<F>"])
    feature_rows = item2_v1._vizier_rows(root / str(files[2]["path"]))
    bars: dict[str, dict[str, Any]] = {}
    for row in feature_rows:
        if str(row.get("Type", "")).strip() != "bar":
            continue
        ellipticity = str(row.get("dEll", "")).strip() or str(row.get("Ell", "")).strip()
        bars[normalize_galaxy_name(row["Name"])] = {
            "ellipticity": float(ellipticity) if ellipticity else None,
            "quality": int(row["Qual"]),
        }
    return families, bars


def _winsorized_positive_flux(
    signal: np.ndarray,
    radius: np.ndarray,
    aperture: float,
    *,
    bins: int,
    quantile: float,
) -> np.ndarray:
    positive = np.maximum(signal, 0.0)
    clipped = positive.copy()
    edges = np.linspace(0.0, 1.35 * aperture, bins + 1)
    for lower, upper in pairwise(edges):
        mask = np.isfinite(positive) & (radius >= lower) & (radius < upper) & (positive > 0.0)
        values = positive[mask]
        if values.size < 20:
            continue
        cap = float(np.quantile(values, quantile))
        clipped[mask] = np.minimum(clipped[mask], cap)
    return clipped


def measure_w1_multipoles(
    data: np.ndarray,
    wcs: WCS,
    *,
    ra_deg: float,
    dec_deg: float,
    aperture_arcsec: float,
    inclination_deg: float,
    extraction: Mapping[str, Any],
) -> dict[str, float | int]:
    """Measure deprojected, target-blind W1 aperture multipoles."""

    image = np.asarray(data, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 20:
        raise GravityItem2WiseMultipolesError("invalid unWISE cutout shape")
    celestial = wcs.celestial
    center = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
    x_center, y_center = celestial.world_to_pixel(center)
    scales = np.abs(proj_plane_pixel_scales(celestial)) * 3600.0
    pixel_scale = float(np.sqrt(scales[0] * scales[1]))
    if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise GravityItem2WiseMultipolesError("invalid unWISE pixel scale")
    yy, xx = np.indices(image.shape, dtype=np.float64)
    dx = (xx - float(x_center)) * pixel_scale
    dy = (yy - float(y_center)) * pixel_scale
    sky_radius = np.hypot(dx, dy)
    inner_bg, outer_bg = [
        float(value) * aperture_arcsec
        for value in extraction["background_annulus_aperture_fraction"]
    ]
    background_mask = np.isfinite(image) & (sky_radius >= inner_bg) & (sky_radius <= outer_bg)
    background_values = image[background_mask]
    if background_values.size < 100:
        raise GravityItem2WiseMultipolesError("insufficient unWISE background pixels")
    background = float(np.median(background_values))
    background_mad = float(1.4826 * np.median(np.abs(background_values - background)))
    if not np.isfinite(background_mad) or background_mad <= 0.0:
        raise GravityItem2WiseMultipolesError("invalid unWISE background noise")
    signal = np.where(np.isfinite(image), image - background, 0.0)
    weights = _winsorized_positive_flux(
        signal,
        sky_radius,
        aperture_arcsec,
        bins=int(extraction["radial_winsor_bins"]),
        quantile=float(extraction["positive_flux_winsor_quantile"]),
    )

    refinement_mask = sky_radius <= 0.2 * aperture_arcsec
    refinement_flux = float(np.sum(weights[refinement_mask]))
    if refinement_flux <= 0.0:
        raise GravityItem2WiseMultipolesError("zero central unWISE flux")
    shift_x = float(np.sum(weights[refinement_mask] * dx[refinement_mask]) / refinement_flux)
    shift_y = float(np.sum(weights[refinement_mask] * dy[refinement_mask]) / refinement_flux)
    shift = float(np.hypot(shift_x, shift_y))
    maximum_shift = 0.08 * aperture_arcsec
    if shift > maximum_shift:
        scale = maximum_shift / shift
        shift_x *= scale
        shift_y *= scale
        shift = maximum_shift
    dx = dx - shift_x
    dy = dy - shift_y
    sky_radius = np.hypot(dx, dy)

    pa_lower, pa_upper = [
        float(value) * aperture_arcsec
        for value in extraction["disk_position_angle_annulus_aperture_fraction"]
    ]
    pa_mask = (sky_radius >= pa_lower) & (sky_radius <= pa_upper) & (weights > 0.0)
    pa_flux = float(np.sum(weights[pa_mask]))
    if pa_flux <= 0.0 or int(np.sum(pa_mask)) < 50:
        raise GravityItem2WiseMultipolesError("insufficient outer-disk unWISE flux")
    covariance = (
        np.asarray(
            [
                [
                    np.sum(weights[pa_mask] * dx[pa_mask] ** 2),
                    np.sum(weights[pa_mask] * dx[pa_mask] * dy[pa_mask]),
                ],
                [
                    np.sum(weights[pa_mask] * dx[pa_mask] * dy[pa_mask]),
                    np.sum(weights[pa_mask] * dy[pa_mask] ** 2),
                ],
            ],
            dtype=np.float64,
        )
        / pa_flux
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(~np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
        raise GravityItem2WiseMultipolesError("invalid unWISE outer-disk moments")
    major = eigenvectors[:, -1]
    angle = float(np.arctan2(major[1], major[0]))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    x_prime = dx * cosine + dy * sine
    y_prime = -dx * sine + dy * cosine
    axis_ratio = float(np.cos(np.deg2rad(inclination_deg)))
    if not 0.4 <= axis_ratio <= 1.0:
        raise GravityItem2WiseMultipolesError("unWISE deprojection axis ratio changed")
    y_deprojected = y_prime / axis_ratio
    radius = np.hypot(x_prime, y_deprojected)
    theta = np.arctan2(y_deprojected, x_prime)
    aperture_geometry = (radius <= aperture_arcsec) & np.isfinite(image)
    aperture_mask = aperture_geometry & (weights > 0.0)
    aperture_flux = float(np.sum(weights[aperture_mask]))
    aperture_pixels = int(np.sum(aperture_geometry))
    if aperture_flux <= 0.0 or aperture_pixels < 100:
        raise GravityItem2WiseMultipolesError("insufficient deprojected unWISE aperture")
    inner_mask = aperture_mask & (radius <= 0.2 * aperture_arcsec)
    concentration = float(np.sum(weights[inner_mask]) / aperture_flux)

    minimum_radius = float(extraction["feature_radial_minimum_aperture_fraction"]) * aperture_arcsec
    feature_mask = aperture_mask & (radius >= minimum_radius)
    feature_flux = float(np.sum(weights[feature_mask]))
    radial_second = float(np.sum(weights[feature_mask] * radius[feature_mask] ** 2))
    q2_complex = np.sum(
        weights[feature_mask] * radius[feature_mask] ** 2 * np.exp(2.0j * theta[feature_mask])
    )
    quadrupole = float(abs(q2_complex) / radial_second)

    higher = {}
    for order in (3, 4):
        moment = np.sum(
            weights[feature_mask]
            * (radius[feature_mask] / aperture_arcsec) ** order
            * np.exp(1.0j * order * theta[feature_mask])
        )
        higher[order] = float(abs(moment) / feature_flux)

    centroid_points = []
    for fraction in extraction["centroid_aperture_fractions"]:
        mask = aperture_mask & (radius <= float(fraction) * aperture_arcsec)
        flux = float(np.sum(weights[mask]))
        if flux <= 0.0:
            raise GravityItem2WiseMultipolesError("invalid unWISE centroid aperture")
        centroid_points.append(
            (
                float(np.sum(weights[mask] * x_prime[mask]) / flux),
                float(np.sum(weights[mask] * y_deprojected[mask]) / flux),
            )
        )
    centroid_array = np.asarray(centroid_points)
    centroid_mean = np.mean(centroid_array, axis=0)
    centroid_shift = float(
        np.sqrt(np.mean(np.sum((centroid_array - centroid_mean) ** 2, axis=1))) / aperture_arcsec
    )

    radial_q2 = []
    for lower, upper in pairwise(np.linspace(0.1, 1.0, 7)):
        mask = (
            feature_mask & (radius >= lower * aperture_arcsec) & (radius < upper * aperture_arcsec)
        )
        denominator = float(np.sum(weights[mask] * radius[mask] ** 2))
        if denominator <= 0.0:
            continue
        radial_q2.append(
            np.sum(weights[mask] * radius[mask] ** 2 * np.exp(2.0j * theta[mask])) / denominator
        )
    if len(radial_q2) < 4:
        raise GravityItem2WiseMultipolesError("insufficient radial quadrupole bins")
    radial_q2_array = np.asarray(radial_q2)
    coherence = float(abs(np.sum(radial_q2_array)) / max(np.sum(np.abs(radial_q2_array)), 1.0e-15))
    outer_axis_ratio = float(np.sqrt(max(eigenvalues[0], 0.0) / eigenvalues[1]))
    energy = float(np.sqrt(quadrupole**2 + higher[3] ** 2 + higher[4] ** 2))
    signed_aperture_flux = float(np.sum(signal[aperture_geometry]))
    central_geometry = aperture_geometry & (radius <= 0.2 * aperture_arcsec)
    signed_central_flux = float(np.sum(signal[central_geometry]))
    central_pixels = int(np.sum(central_geometry))
    aperture_snr = signed_aperture_flux / (background_mad * math.sqrt(aperture_pixels))
    central_snr = signed_central_flux / (background_mad * math.sqrt(max(central_pixels, 1)))
    brightest_fraction = float(np.max(weights[aperture_mask]) / aperture_flux)
    quality = extraction["quality_gate"]
    failures = []
    if shift / aperture_arcsec > float(quality["maximum_center_refinement_fraction"]):
        failures.append("center_refinement")
    if central_snr < float(quality["minimum_central_flux_snr"]):
        failures.append("central_flux_snr")
    if aperture_snr < float(quality["minimum_aperture_flux_snr"]):
        failures.append("aperture_flux_snr")
    if aperture_pixels < int(quality["minimum_aperture_pixels"]):
        failures.append("aperture_pixels")
    if concentration < float(quality["minimum_concentration_c20"]):
        failures.append("concentration_c20")
    if brightest_fraction > float(quality["maximum_brightest_pixel_flux_fraction"]):
        failures.append("brightest_pixel_flux_fraction")
    result = {
        "aperture_flux_snr": aperture_snr,
        "aperture_pixels": aperture_pixels,
        "background": background,
        "background_mad_sigma": background_mad,
        "brightest_pixel_flux_fraction": brightest_fraction,
        "center_refinement_fraction": shift / aperture_arcsec,
        "central_flux_snr": central_snr,
        "centroid_shift": centroid_shift,
        "concentration_c20": concentration,
        "image_quality_pass": not failures,
        "m3_aperture_amplitude": higher[3],
        "m4_aperture_amplitude": higher[4],
        "measurement_valid": True,
        "multipole_energy": energy,
        "position_angle_deg": float(np.rad2deg(angle) % 180.0),
        "positive_aperture_flux": aperture_flux,
        "projected_outer_axis_ratio": outer_axis_ratio,
        "quality_failure_reason": ",".join(failures),
        "quadrupole_amplitude": quadrupole,
        "radial_quadrupole_coherence": coherence,
        "signed_aperture_flux": signed_aperture_flux,
    }
    numeric = [
        value
        for value in result.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if any(not np.isfinite(float(value)) for value in numeric):
        raise GravityItem2WiseMultipolesError("non-finite unWISE multipole")
    if not 0.0 < concentration < 1.0 or not 0.0 <= quadrupole <= 1.0:
        raise GravityItem2WiseMultipolesError("nonphysical unWISE morphology")
    return result


def _feature_payload(row: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for column in FEATURE_COLUMNS:
        value = row.get(column, "")
        if value is None:
            result[column] = ""
        elif isinstance(value, float):
            result[column] = format(value, ".15e")
        else:
            result[column] = str(value)
    return result


def _render_features(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FEATURE_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_feature_payload(row))
    return buffer.getvalue().encode("utf-8")


def _failed_measurement(reason: str) -> dict[str, Any]:
    """Represent a target-blind image-measurement failure without dropping its object."""

    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", reason).strip("_")[:160]
    return {
        "aperture_flux_snr": 0.0,
        "aperture_pixels": 0,
        "background": 0.0,
        "background_mad_sigma": 0.0,
        "brightest_pixel_flux_fraction": 1.0,
        "center_refinement_fraction": 1.0,
        "central_flux_snr": 0.0,
        "centroid_shift": 0.0,
        "concentration_c20": 0.0,
        "image_quality_pass": False,
        "m3_aperture_amplitude": 0.0,
        "m4_aperture_amplitude": 0.0,
        "measurement_valid": False,
        "multipole_energy": 0.0,
        "position_angle_deg": 0.0,
        "positive_aperture_flux": 0.0,
        "projected_outer_axis_ratio": 0.0,
        "quality_failure_reason": f"measurement_error:{cleaned}",
        "quadrupole_amplitude": 0.0,
        "radial_quadrupole_coherence": 0.0,
        "signed_aperture_flux": 0.0,
    }


def _manifest_feature(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    return _metric(float(value))


def _sealable_contract(value: Any) -> Any:
    """Convert a JSON-like preregistration fragment to canonical receipt scalars."""

    if isinstance(value, Mapping):
        return {str(key): _sealable_contract(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sealable_contract(child) for child in value]
    if isinstance(value, float):
        return _metric(value)
    return value


def acquire_unwise_features(
    root: Path,
    *,
    cache_dir: Path,
    manifest_path: Path | None = None,
    feature_path: Path | None = None,
) -> dict[str, Any]:
    """Download hash-addressed cutouts and write an immutable feature source receipt."""

    root = root.resolve()
    config = load_extraction_config(root)
    unwise = config["sources"]["unwise_w1"]
    manifest_path = manifest_path or (root / str(unwise["image_manifest_path"]))
    feature_path = feature_path or (root / str(unwise["derived_feature_path"]))
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    galaxies = _eligible_galaxies(root, config)
    families, bars = _parse_s4g_validation(root, config)
    extraction = config["image_extraction"]
    rows = []
    manifest_records = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Invariant-target-blind-morphology-audit/1.0"})
    for ordinal, galaxy in enumerate(galaxies, start=1):
        aperture_arcsec = (
            206.265
            * float(extraction["aperture_effective_radius_multiple"])
            * float(galaxy["effective_radius_kpc"])
            / float(galaxy["distance_mpc"])
        )
        cutout_size = max(
            float(extraction["minimum_cutout_diameter_arcsec"]),
            float(extraction["cutout_diameter_aperture_multiple"]) * aperture_arcsec,
        )
        pixel_scale = float(unwise["pixel_scale_arcsec"])
        size_pixels = max(64, math.ceil(cutout_size / pixel_scale))
        query = urlencode(
            {
                "bands": "1",
                "dec": format(float(galaxy["dec_deg"]), ".10f"),
                "layer": str(unwise["layer"]),
                "pixscale": format(pixel_scale, ".8g"),
                "ra": format(float(galaxy["ra_deg"]), ".10f"),
                "size": str(size_pixels),
            }
        )
        url = f"{unwise['cutout_service']}?{query}"
        cache_path = cache_dir / f"{galaxy['name']}-unwise-neo11-w1.fits"
        if not cache_path.exists():
            response = session.get(url, timeout=180)
            response.raise_for_status()
            if not response.content.startswith(b"SIMPLE"):
                raise GravityItem2WiseMultipolesError("unWISE cutout was not FITS")
            temporary = cache_path.with_suffix(".fits.tmp")
            temporary.write_bytes(response.content)
            temporary.replace(cache_path)
        with fits.open(cache_path, memmap=False) as handle:
            header = handle[0].header
            image = np.squeeze(np.asarray(handle[0].data, dtype=np.float64))
            image_version = str(header.get("VERSION", "")).strip()
            image_bands = str(header.get("BANDS", "")).strip()
            if image_version != str(unwise["layer"]) or image_bands != "1":
                raise GravityItem2WiseMultipolesError("unWISE cutout provenance changed")
            wcs = WCS(header)
            try:
                features = measure_w1_multipoles(
                    image,
                    wcs,
                    ra_deg=float(galaxy["ra_deg"]),
                    dec_deg=float(galaxy["dec_deg"]),
                    aperture_arcsec=aperture_arcsec,
                    inclination_deg=float(galaxy["inclination_deg"]),
                    extraction=extraction,
                )
            except GravityItem2WiseMultipolesError as error:
                features = _failed_measurement(str(error))
            image_shape = [int(value) for value in image.shape]
        normalized = normalize_galaxy_name(str(galaxy["name"]))
        family = families.get(normalized)
        bar = bars.get(normalized, {})
        image_sha = _sha256(cache_path)
        row = {
            **galaxy,
            **features,
            "aperture_arcsec": aperture_arcsec,
            "image_version": image_version,
            "image_sha256": image_sha,
            "s4g_bar_ellipticity": bar.get("ellipticity"),
            "s4g_bar_quality": bar.get("quality"),
            "s4g_family": family,
        }
        rows.append(row)
        manifest_records.append(
            {
                "aperture_arcsec": _metric(aperture_arcsec),
                "bytes": cache_path.stat().st_size,
                "cutout_size_arcsec": _metric(cutout_size),
                "features": {
                    key: _manifest_feature(value) for key, value in sorted(features.items())
                },
                "image_sha256": image_sha,
                "image_shape": image_shape,
                "image_version": image_version,
                "name": galaxy["name"],
                "size_pixels": size_pixels,
                "url": url,
            }
        )
        print(
            f"[{ordinal:02d}/{len(galaxies)}] {galaxy['name']} "
            f"q2={features['quadrupole_amplitude']:.5f} "
            f"m3={features['m3_aperture_amplitude']:.5f} "
            f"quality={features['image_quality_pass']}",
            flush=True,
        )

    feature_payload = _render_features(rows)
    quality_rows = [row for row in rows if row["image_quality_pass"]]
    matched_family = [row for row in quality_rows if row["s4g_family"] is not None]
    matched_ellipticity = [row for row in quality_rows if row["s4g_bar_ellipticity"] is not None]

    def correlation(rows_to_compare: Sequence[Mapping[str, Any]], field: str) -> str | None:
        if len(rows_to_compare) < 3:
            return None
        statistic = float(
            spearmanr(
                [float(row[field]) for row in rows_to_compare],
                [float(row["quadrupole_amplitude"]) for row in rows_to_compare],
            ).statistic
        )
        return _metric(statistic) if np.isfinite(statistic) else None

    failure_counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["quality_failure_reason"])
        if not reason:
            continue
        for piece in reason.split(","):
            failure_counts[piece] = failure_counts.get(piece, 0) + 1
    feature_sha = hashlib.sha256(feature_payload).hexdigest()
    body: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "goal": "TARGET_BLIND_UNWISE_NEO11_W1_MULTIPOLE_EXTRACTION",
        "unwise": {
            "band": unwise["band"],
            "cutout_service": unwise["cutout_service"],
            "dataset_doi": unwise["dataset_doi"],
            "layer": unwise["layer"],
            "pixel_scale_arcsec": _metric(float(unwise["pixel_scale_arcsec"])),
        },
        "counts": {
            "eligible_galaxies": len(galaxies),
            "images": len(manifest_records),
            "measurement_valid": sum(bool(row["measurement_valid"]) for row in rows),
            "quality_failure_reasons": dict(sorted(failure_counts.items())),
            "quality_pass_galaxies": len(quality_rows),
            "s4g_bar_ellipticity_matches": len(matched_ellipticity),
            "s4g_family_matches": len(matched_family),
            "target_fields_used_by_feature_computation": 0,
        },
        "external_validation": {
            "quadrupole_vs_s4g_bar_ellipticity_spearman": correlation(
                matched_ellipticity, "s4g_bar_ellipticity"
            ),
            "quadrupole_vs_s4g_family_spearman": correlation(matched_family, "s4g_family"),
            "role": "target_blind_image_feature_validation_only",
        },
        "extraction_contract_sha256": canonical_sha256(
            _sealable_contract(
                {
                    "image_extraction": config["image_extraction"],
                    "population_filter": {
                        "galaxy_inclination_maximum_deg": config["population"][
                            "galaxy_inclination_maximum_deg"
                        ],
                        "source_boundary": config["population"]["source_boundary"],
                    },
                }
            )
        ),
        "feature_file": {
            "path": str(feature_path.relative_to(root)).replace("\\", "/"),
            "sha256": feature_sha,
        },
        "records": manifest_records,
        "source_bindings": {
            "extractor": {
                "path": SOURCE_PATH,
                "sha256": _file_sha256(root / SOURCE_PATH),
            },
            "sparc": {
                "path": config["sources"]["sparc_global_properties"]["path"],
                "sha256": config["sources"]["sparc_global_properties"]["file_sha256"],
            },
            "sparc_exploration_membership": {
                "path": config["sources"]["sparc_exploration_membership"]["path"],
                "sha256": config["sources"]["sparc_exploration_membership"]["file_sha256"],
            },
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    manifest_payload = canonical_json_bytes(body) + b"\n"
    for path, payload in ((feature_path, feature_payload), (manifest_path, manifest_payload)):
        if path.exists() and path.read_bytes() != payload:
            raise GravityItem2WiseMultipolesError(f"refusing to overwrite immutable {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
    return body


def validate_extraction(root: Path, *, cache_dir: Path | None = None) -> Mapping[str, Any]:
    """Validate the derived table, manifest, and optionally every cached raw cutout."""

    root = root.resolve()
    config = load_extraction_config(root)
    unwise = config["sources"]["unwise_w1"]
    manifest_path = root / str(unwise["image_manifest_path"])
    feature_path = root / str(unwise["derived_feature_path"])
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise GravityItem2WiseMultipolesError("unWISE manifest schema changed")
    _verify_envelope(manifest, label="unWISE manifest")
    if _sha256(manifest_path) != unwise.get("image_manifest_sha256"):
        raise GravityItem2WiseMultipolesError("unWISE manifest file changed")
    if _sha256(feature_path) != unwise.get("derived_feature_sha256"):
        raise GravityItem2WiseMultipolesError("unWISE feature file changed")
    if manifest.get("feature_file", {}).get("sha256") != _sha256(feature_path):
        raise GravityItem2WiseMultipolesError("unWISE feature binding changed")
    rows = list(csv.DictReader(feature_path.open(encoding="utf-8"), delimiter="\t"))
    if len(rows) != 83 or len({row["name"] for row in rows}) != 83:
        raise GravityItem2WiseMultipolesError("unWISE feature population changed")
    if set(rows[0]) != set(FEATURE_COLUMNS):
        raise GravityItem2WiseMultipolesError("unWISE feature schema changed")
    quality_rows = [row for row in rows if row["image_quality_pass"] == "True"]
    if len(quality_rows) != int(manifest.get("counts", {}).get("quality_pass_galaxies", -1)):
        raise GravityItem2WiseMultipolesError("unWISE quality count changed")
    if manifest.get("counts", {}).get("target_fields_used_by_feature_computation") != 0:
        raise GravityItem2WiseMultipolesError("target leakage entered unWISE extraction")
    if manifest.get("counts", {}).get("images") != 83:
        raise GravityItem2WiseMultipolesError("unWISE image count changed")
    extractor = manifest.get("source_bindings", {}).get("extractor", {})
    if extractor != {"path": SOURCE_PATH, "sha256": _file_sha256(root / SOURCE_PATH)}:
        raise GravityItem2WiseMultipolesError("unWISE extractor binding changed")
    if cache_dir is not None:
        cache_dir = cache_dir.resolve()
        for record in manifest["records"]:
            path = cache_dir / f"{record['name']}-unwise-neo11-w1.fits"
            if _sha256(path) != record["image_sha256"]:
                raise GravityItem2WiseMultipolesError("cached unWISE image changed")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "acquire":
        if args.cache_dir is None:
            raise GravityItem2WiseMultipolesError("--cache-dir is required for acquisition")
        manifest = acquire_unwise_features(root, cache_dir=args.cache_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    validate_extraction(root, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "GravityItem2WiseMultipolesError",
    "acquire_unwise_features",
    "load_extraction_config",
    "main",
    "measure_w1_multipoles",
    "normalize_galaxy_name",
    "validate_extraction",
]
