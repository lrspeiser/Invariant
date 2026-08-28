"""Frozen GZ3D resonance/coherence search for gravity roadmap Item 14."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_item11_neargalcat_external_field import (
    _content_hashed,
    _metric,
    _minimum_separation_arcsec,
    _ridge_fit,
    _ridge_predict,
    _sha256_file,
    _source_rows,
    _validate_content_hash,
    canonical_json_bytes,
)

CONFIG_PATH = Path("configs/gravity_item14_gz3d_resonance_coherence_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "095c7ddfc4ca16c8ae1e4aeaafbfbd180e902197"
SAMPLE_FREEZE_COMMIT = "PENDING_ITEM14_SAMPLE_FREEZE_COMMIT"


class GravityItem14CoherenceError(RuntimeError):
    """Raised when an Item 14 scientific or response boundary drifts."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    return str(value).strip()


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GravityItem14CoherenceError(f"missing or invalid {label}") from exc
    if not math.isfinite(result) or result <= -900:
        raise GravityItem14CoherenceError(f"missing or invalid {label}")
    return result


def _nonnegative_integer_count(value: Any, label: str) -> int:
    number = _number(value, label)
    count = int(number)
    if number != count or count < 0:
        raise GravityItem14CoherenceError(f"missing or invalid {label}")
    return count


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/Item14-GZ3D"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise GravityItem14CoherenceError("empty remote payload")
    return payload


def _serialize(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in row.items():
        if isinstance(value, (np.bool_, bool)):
            output[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            output[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            output[key] = _metric(float(value))
        else:
            output[key] = value
    return output


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem14CoherenceError("stable gravity roadmap changed")
    predecessor_binding = config["predecessor"]
    predecessor_path = root / predecessor_binding["path"]
    if _sha256_file(predecessor_path) != predecessor_binding["file_sha256"]:
        raise GravityItem14CoherenceError("Item 13 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predecessor, "Item 13 synthesis")
    if predecessor.get("content_sha256") != predecessor_binding["content_sha256"]:
        raise GravityItem14CoherenceError("Item 13 synthesis content binding changed")
    if predecessor.get("decision") != predecessor_binding["required_decision"]:
        raise GravityItem14CoherenceError("Item 13 synthesis decision changed")
    predictor_binding = config["sources"]["item13_predictors"]
    predictor_path = root / predictor_binding["path"]
    if _sha256_file(predictor_path) != predictor_binding["file_sha256"]:
        raise GravityItem14CoherenceError("Item 13 predictor source changed")
    predictors = json.loads(predictor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predictors, "Item 13 predictor source")
    if predictors.get("content_sha256") != predictor_binding["content_sha256"]:
        raise GravityItem14CoherenceError("Item 13 predictor content binding changed")
    if len(predictors.get("records", [])) != int(predictor_binding["records"]):
        raise GravityItem14CoherenceError("Item 13 predictor row count changed")
    for section in ("identity_exclusions", "coordinate_exclusions"):
        for entry in config["independence"][section]:
            if _sha256_file(root / entry["path"]) != entry["file_sha256"]:
                raise GravityItem14CoherenceError(
                    f"predecessor exclusion source changed: {entry['path']}"
                )
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem14CoherenceError("Item 14 config contains an overclaim")
    if int(config["candidate_generator"]["candidate_cells"]) != 262144:
        raise GravityItem14CoherenceError("Item 14 candidate count changed")
    if int(config["sample"]["maximum_total_objects"]) != 320:
        raise GravityItem14CoherenceError("Item 14 sample size changed")
    sample = config["sample"]
    if int(sample["exploration_objects"]) + int(sample["confirmation_objects"]) != int(
        sample["maximum_total_objects"]
    ):
        raise GravityItem14CoherenceError("Item 14 exploration/confirmation split changed")
    if 4 * int(sample["objects_per_cell"]) != int(sample["maximum_total_objects"]):
        raise GravityItem14CoherenceError("Item 14 target-blind cell balance changed")
    if int(sample["exploration_objects"]) % (4 * int(config["evaluation"]["outer_folds"])):
        raise GravityItem14CoherenceError("Item 14 fold balance is not integral")
    if not bool(config["sources"]["response"]["confirmation_query_forbidden"]):
        raise GravityItem14CoherenceError("Item 14 confirmation MAPS boundary changed")
    authorization = config["authorization"]
    if bool(authorization["paid_model_calls_allowed"]):
        raise GravityItem14CoherenceError("Item 14 paid model boundary changed")
    if bool(authorization["response_query_allowed_before_sample_freeze"]):
        raise GravityItem14CoherenceError("Item 14 pre-freeze MAPS boundary changed")
    if not bool(authorization["exploration_response_query_allowed_after_sample_freeze"]):
        raise GravityItem14CoherenceError("Item 14 exploration MAPS boundary changed")
    if bool(authorization["confirmation_response_query_allowed"]):
        raise GravityItem14CoherenceError("Item 14 confirmation authorization changed")
    if bool(authorization["post_response_candidate_generation_allowed"]):
        raise GravityItem14CoherenceError("Item 14 post-response authorization changed")
    if bool(authorization["kinematic_response_as_mask_predictor_allowed"]):
        raise GravityItem14CoherenceError("Item 14 response-as-predictor boundary changed")
    if bool(authorization["object_identity_as_numeric_feature_allowed"]):
        raise GravityItem14CoherenceError("Item 14 identity-feature boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem14CoherenceError("Item 14 post-response candidate boundary changed")
    quality = config["quality"]
    inner = [float(value) for value in quality["inner_annulus_re"]]
    outer = [float(value) for value in quality["outer_annulus_re"]]
    quantiles = [float(value) for value in quality["velocity_span_quantiles"]]
    if not (
        len(inner) == len(outer) == len(quantiles) == 2
        and 0 <= inner[0] < inner[1] == outer[0] < outer[1]
        and 0 <= quantiles[0] < quantiles[1] <= 1
    ):
        raise GravityItem14CoherenceError("Item 14 annular response definition changed")
    if config["sources"]["prefreeze_access"]["metadata_row_values_read"] != 0:
        raise GravityItem14CoherenceError("GZ3D metadata row boundary changed")
    if config["sources"]["prefreeze_access"]["mask_pixel_values_read"] != 0:
        raise GravityItem14CoherenceError("GZ3D mask boundary changed")
    if config["sources"]["prefreeze_access"]["maps_payload_downloads"] != 0:
        raise GravityItem14CoherenceError("MaNGA MAPS payload boundary changed")
    if config["sources"]["prefreeze_access"]["maps_pixel_values_read"] != 0:
        raise GravityItem14CoherenceError("MaNGA MAPS pixel boundary changed")
    if config["sources"]["prefreeze_access"]["resolved_kinematic_response_objects_read"] != 0:
        raise GravityItem14CoherenceError("resolved kinematic response boundary changed")
    return config


def _require_scientific_freeze() -> None:
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem14CoherenceError("Item 14 scientific freeze is not bound")


def _parse_sha1_manifest(payload: bytes) -> dict[str, str]:
    mapping = {}
    for line in payload.decode("utf-8", errors="strict").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 40:
            raise GravityItem14CoherenceError("invalid GZ3D SHA1 manifest row")
        digest, name = parts
        name = name.lstrip("*./")
        if name in mapping or any(character not in "0123456789abcdef" for character in digest):
            raise GravityItem14CoherenceError("duplicate or invalid GZ3D SHA1 entry")
        mapping[name] = digest
    return mapping


def _excluded_identities(root: Path, config: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    plateifus: set[str] = set()
    mangaids: set[str] = set()
    for entry in config["independence"]["identity_exclusions"]:
        source = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        for row in source[entry["objects_key"]]:
            plateifus.add(str(row[entry["plateifu_key"]]).upper())
            mangaids.add(str(row[entry["mangaid_key"]]).upper())
    return plateifus, mangaids


def _coordinates(root: Path, config: Mapping[str, Any]) -> np.ndarray:
    rows = []
    for entry in config["independence"]["coordinate_exclusions"]:
        for row in _source_rows(root / entry["path"], entry["format"]):
            try:
                ra = float(row[entry["ra_key"]])
                dec = float(row[entry["dec_key"]])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(ra) and math.isfinite(dec):
                rows.append((ra, dec))
    if not rows:
        raise GravityItem14CoherenceError("empty predecessor coordinate registry")
    return np.asarray(rows, dtype=np.float64)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0:
        raise GravityItem14CoherenceError("empty weighted quantile")
    threshold = float(quantile) * float(cumulative[-1])
    return float(sorted_values[min(int(np.searchsorted(cumulative, threshold)), len(values) - 1)])


def derive_mask_features(
    compressed_payload: bytes, predictor: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, float | int]:
    try:
        from astropy.io import fits

        decompressed = gzip.decompress(compressed_payload)
        with fits.open(io.BytesIO(decompressed), memmap=False) as hdus:
            spiral = np.asarray(
                hdus[int(config["sources"]["gz3d_masks"]["spiral_hdu"])].data,
                dtype=np.float64,
            )
            bar = np.asarray(
                hdus[int(config["sources"]["gz3d_masks"]["bar_hdu"])].data,
                dtype=np.float64,
            )
    except (OSError, ValueError, TypeError, IndexError) as exc:
        raise GravityItem14CoherenceError("invalid GZ3D mask FITS") from exc
    expected_shape = tuple(
        int(value) for value in config["mask_feature_extraction"]["expected_shape_pixels"]
    )
    if spiral.ndim != 2 or spiral.shape != bar.shape or spiral.shape != expected_shape:
        raise GravityItem14CoherenceError("GZ3D mask shape changed")
    if np.any(~np.isfinite(spiral)) or np.any(~np.isfinite(bar)):
        raise GravityItem14CoherenceError("non-finite GZ3D mask")
    if np.any(spiral < 0) or np.any(bar < 0):
        raise GravityItem14CoherenceError("negative GZ3D mask weight")
    height, width = spiral.shape
    yy, xx = np.indices(spiral.shape, dtype=np.float64)
    xx -= (width - 1.0) / 2.0
    yy -= (height - 1.0) / 2.0
    radius_pixels = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    half_light_arcsec = 10.0 ** float(predictor["log_half_light_radius"])
    pixel_scale = float(config["sources"]["gz3d_masks"]["fixed_pixel_scale_arcsec"])
    half_light_pixels = half_light_arcsec / pixel_scale
    if not math.isfinite(half_light_pixels) or half_light_pixels <= 0:
        raise GravityItem14CoherenceError("invalid mask half-light scale")
    radius_re = radius_pixels / half_light_pixels
    radial_min, radial_max = (
        float(value) for value in config["mask_feature_extraction"]["radial_range_re"]
    )
    valid = (spiral > 0) & (radius_re >= radial_min) & (radius_re <= radial_max)
    minimum_pixels = int(config["mask_feature_extraction"]["minimum_nonzero_spiral_pixels"])
    if int(np.sum(valid)) < minimum_pixels:
        raise GravityItem14CoherenceError("insufficient nonzero spiral-mask pixels")
    weights = spiral[valid]
    radii = radius_re[valid]
    angles = theta[valid]
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise GravityItem14CoherenceError("empty spiral-mask weight")
    amplitudes = {
        mode: float(abs(np.sum(weights * np.exp(1j * mode * angles))) / total_weight)
        for mode in config["mask_feature_extraction"]["harmonic_modes"]
    }
    amplitude_values = np.asarray([amplitudes[mode] for mode in (1, 2, 3, 4)])
    amplitude_total = float(np.sum(amplitude_values))
    if amplitude_total <= 1e-15:
        mode_entropy = 1.0
    else:
        probabilities = amplitude_values / amplitude_total
        nonzero = probabilities > 0
        mode_entropy = float(
            -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])) / np.log(4.0)
        )
    quantiles = [
        _weighted_quantile(radii, weights, float(value))
        for value in config["mask_feature_extraction"]["weighted_radial_quantiles"]
    ]
    radial_edges = np.asarray(
        config["mask_feature_extraction"]["radial_phase_bins_re"], dtype=np.float64
    )
    phase_radii = []
    phase_values = []
    phase_weights = []
    for lower, upper in pairwise(radial_edges):
        mask = (radii >= lower) & (radii < upper)
        if int(np.sum(mask)) < 3 or float(np.sum(weights[mask])) <= 0:
            continue
        moment = np.sum(weights[mask] * np.exp(2j * angles[mask]))
        amplitude = float(abs(moment) / np.sum(weights[mask]))
        if amplitude <= 1e-8:
            continue
        phase_radii.append(math.sqrt(float(lower) * float(upper)))
        phase_values.append(float(np.angle(moment)))
        phase_weights.append(amplitude)
    minimum_bins = int(config["mask_feature_extraction"]["minimum_phase_bins"])
    if len(phase_values) < minimum_bins:
        raise GravityItem14CoherenceError("insufficient spiral radial phase bins")
    phase_x = np.log(np.asarray(phase_radii, dtype=np.float64))
    phase_y = np.unwrap(np.asarray(phase_values, dtype=np.float64))
    phase_w = np.asarray(phase_weights, dtype=np.float64)
    design = np.column_stack((np.ones(len(phase_x)), phase_x))
    weighted_design = design * np.sqrt(phase_w)[:, None]
    weighted_target = phase_y * np.sqrt(phase_w)
    intercept, slope = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
    residual = phase_y - (intercept + slope * phase_x)
    phase_linearity = float(abs(np.sum(phase_w * np.exp(1j * residual))) / np.sum(phase_w))
    pitch_abs = float(abs(slope) / 2.0)
    phase_twist = float(abs(phase_y[-1] - phase_y[0]) / 2.0)
    inner_spiral_angle = float(phase_y[0] / 2.0)
    bar_valid = bar > 0
    if int(np.sum(bar_valid)) >= 3 and float(np.sum(bar[bar_valid])) > 0:
        bar_weights = bar[bar_valid]
        bar_x = xx[bar_valid]
        bar_y = yy[bar_valid]
        bar_weight_total = float(np.sum(bar_weights))
        mean_x = float(np.sum(bar_weights * bar_x) / bar_weight_total)
        mean_y = float(np.sum(bar_weights * bar_y) / bar_weight_total)
        centered = np.column_stack((bar_x - mean_x, bar_y - mean_y))
        covariance = (centered * bar_weights[:, None]).T @ centered / bar_weight_total
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = np.abs(centered @ major) / half_light_pixels
        bar_extent = _weighted_quantile(projection, bar_weights, 0.9)
        maximum = max(float(np.max(eigenvalues)), 1e-15)
        minimum = max(float(np.min(eigenvalues)), 0.0)
        bar_ellipticity = float(1.0 - math.sqrt(minimum / maximum))
        bar_angle = float(math.atan2(major[1], major[0]))
        bar_spiral_lock = float(math.cos(2.0 * (bar_angle - inner_spiral_angle)))
        bar_spiral_ratio = float(math.log(max(bar_extent, 1e-6) / max(quantiles[0], 1e-6)))
    else:
        bar_extent = 0.0
        bar_ellipticity = 0.0
        bar_spiral_lock = 0.0
        bar_spiral_ratio = 0.0
    result: dict[str, float | int] = {
        "spiral_nonzero_pixels": int(np.sum(valid)),
        "mode_m1": amplitudes[1],
        "mode_m2": amplitudes[2],
        "mode_m3": amplitudes[3],
        "mode_m4": amplitudes[4],
        "mode_entropy": mode_entropy,
        "m2_phase_linearity": phase_linearity,
        "m2_pitch_abs": pitch_abs,
        "m2_phase_twist": phase_twist,
        "bar_ellipticity": bar_ellipticity,
        "bar_extent_re": bar_extent,
        "bar_spiral_phase_lock": bar_spiral_lock,
        "bar_spiral_log_radius_ratio": bar_spiral_ratio,
        "spiral_radial_coverage_re": float(quantiles[2] - quantiles[0]),
        "spiral_radial_median_re": float(quantiles[1]),
    }
    if any(not math.isfinite(float(value)) for value in result.values()):
        raise GravityItem14CoherenceError("non-finite GZ3D derived feature")
    return result


def _metadata_rows(payload: bytes, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    from astropy.io import fits

    with fits.open(io.BytesIO(payload), memmap=False) as hdus:
        hdu = hdus[int(config["sources"]["gz3d_metadata"]["hdu"])]
        observed_names = list(hdu.columns.names)
        expected_names = config["sources"]["gz3d_metadata"]["observed_columns"]
        if observed_names != expected_names:
            raise GravityItem14CoherenceError("GZ3D metadata schema changed")
        data = hdu.data
        rows = [{name: data[index][name] for name in observed_names} for index in range(len(data))]
    if len(rows) != int(config["sources"]["gz3d_metadata"]["observed_rows"]):
        raise GravityItem14CoherenceError("GZ3D metadata row count changed")
    return rows


def _eligible_metadata_records(
    root: Path,
    config: Mapping[str, Any],
    metadata_payload: bytes,
    sha1_by_name: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    predictor_binding = config["sources"]["item13_predictors"]
    predictor_source = json.loads((root / predictor_binding["path"]).read_text(encoding="utf-8"))
    predictors_by_mangaid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictor_source["records"]:
        predictors_by_mangaid[str(row["mangaid"]).upper()].append(row)
    excluded_plates, excluded_mangaids = _excluded_identities(root, config)
    predecessor_coordinates = _coordinates(root, config)
    separation_limit = float(config["independence"]["coordinate_exclusion_arcseconds"])
    sample_config = config["sample"]
    failures = Counter()
    records_by_mangaid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metadata in _metadata_rows(metadata_payload, config):
        mangaid = _decode(metadata["MANGAID"]).upper()
        if mangaid in excluded_mangaids:
            failures["excluded_mangaid"] += 1
            continue
        candidates = predictors_by_mangaid.get(mangaid, [])
        candidates = [
            row for row in candidates if str(row["plateifu"]).upper() not in excluded_plates
        ]
        if not candidates:
            failures["no_fresh_predictor"] += 1
            continue
        candidates.sort(
            key=lambda row: _split_hash(
                f"{mangaid}|{row['plateifu']}", str(sample_config["split_salt"])
            )
        )
        predictor = candidates[0]
        if bool(sample_config["require_observed"]) and not bool(metadata["observed"]):
            failures["not_observed"] += 1
            continue
        try:
            total = _nonnegative_integer_count(
                metadata["GZ_total_classifications"], "GZ classifications"
            )
            spiral_votes = _nonnegative_integer_count(
                metadata["GZ_spiral_votes"], "GZ spiral votes"
            )
            bar_votes = _nonnegative_integer_count(metadata["GZ_bar_votes"], "GZ bar votes")
        except GravityItem14CoherenceError:
            failures["invalid_vote_counts"] += 1
            continue
        if spiral_votes > total or bar_votes > total:
            failures["invalid_vote_counts"] += 1
            continue
        if total < int(sample_config["minimum_total_classifications"]):
            failures["too_few_classifications"] += 1
            continue
        spiral_fraction = spiral_votes / total
        bar_fraction = bar_votes / total
        if spiral_fraction < float(sample_config["minimum_spiral_vote_fraction"]):
            failures["low_spiral_vote_fraction"] += 1
            continue
        axis_ratio = float(predictor["axis_ratio"])
        if not (
            float(sample_config["minimum_axis_ratio"])
            <= axis_ratio
            <= float(sample_config["maximum_axis_ratio"])
        ):
            failures["axis_ratio"] += 1
            continue
        separation = _minimum_separation_arcsec(
            float(predictor["ra"]), float(predictor["dec"]), predecessor_coordinates
        )
        if separation <= separation_limit:
            failures["predecessor_coordinate"] += 1
            continue
        file_name = _decode(metadata["file_name"])
        if (
            file_name != Path(file_name).name
            or file_name not in sha1_by_name
            or not file_name.endswith(".fits.gz")
        ):
            failures["missing_official_mask_hash"] += 1
            continue
        mass_state = (
            "higher_mass"
            if float(predictor["log_stellar_mass"])
            > float(sample_config["stellar_mass_threshold_log10"])
            else "lower_mass"
        )
        bar_state = (
            "bar_high" if bar_fraction >= float(sample_config["bar_vote_threshold"]) else "bar_low"
        )
        records_by_mangaid[mangaid].append(
            {
                **predictor,
                "mangaid": mangaid,
                "gz3d_file_name": file_name,
                "gz3d_official_sha1": sha1_by_name[file_name],
                "gz_total_classifications": total,
                "gz_spiral_votes": spiral_votes,
                "gz_bar_votes": bar_votes,
                "spiral_vote_fraction": spiral_fraction,
                "bar_vote_fraction": bar_fraction,
                "bar_vote_state": bar_state,
                "stellar_mass_state": mass_state,
                "sample_cell": f"{bar_state}|{mass_state}",
            }
        )
    deduplicated = []
    for mangaid, rows in records_by_mangaid.items():
        rows.sort(
            key=lambda row: _split_hash(
                f"{mangaid}|{row['plateifu']}|{row['gz3d_file_name']}",
                str(sample_config["split_salt"]),
            )
        )
        deduplicated.append(rows[0])
        failures["duplicate_metadata_or_observation"] += len(rows) - 1
    return deduplicated, dict(sorted(failures.items()))


def _mask_payload(
    root: Path, record: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[bytes, str]:
    cache = root / config["sources"]["gz3d_masks"]["raw_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / str(record["gz3d_file_name"])
    expected_sha1 = str(record["gz3d_official_sha1"])
    if path.exists():
        payload = path.read_bytes()
        if hashlib.sha1(payload).hexdigest() != expected_sha1:
            raise GravityItem14CoherenceError("cached GZ3D mask SHA1 changed")
    else:
        payload = _download(
            str(config["sources"]["gz3d_masks"]["base_url"]) + str(record["gz3d_file_name"])
        )
        if hashlib.sha1(payload).hexdigest() != expected_sha1:
            raise GravityItem14CoherenceError("GZ3D mask official SHA1 mismatch")
        path.write_bytes(payload)
    return payload, hashlib.sha256(payload).hexdigest()


def _select_with_mask_features(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    loader: Callable[
        [Path, Mapping[str, Any], Mapping[str, Any]], tuple[bytes, str]
    ] = _mask_payload,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["sample_cell"])].append(row)
    expected_cells = {
        f"{bar}|{mass}" for bar in ("bar_low", "bar_high") for mass in ("lower_mass", "higher_mass")
    }
    if set(grouped) != expected_cells:
        raise GravityItem14CoherenceError("Item 14 sample cells unavailable")
    selected = []
    feature_records = []
    failures = []
    per_cell = int(config["sample"]["objects_per_cell"])
    exploration_per_cell = int(config["sample"]["exploration_objects"]) // 4
    salt = str(config["sample"]["split_salt"])
    for cell in sorted(expected_cells):
        candidates = sorted(
            grouped[cell],
            key=lambda row: _split_hash(
                f"{row['mangaid']}|{row['plateifu']}|{row['gz3d_file_name']}", salt
            ),
        )
        successes = []
        for candidate in candidates:
            if len(successes) >= per_cell:
                break
            try:
                payload, mask_sha256 = loader(root, candidate, config)
                features = derive_mask_features(payload, candidate, config)
            except (GravityItem14CoherenceError, OSError) as exc:
                failures.append(
                    {
                        "mangaid": candidate["mangaid"],
                        "plateifu": candidate["plateifu"],
                        "file_name": candidate["gz3d_file_name"],
                        "reason": str(exc),
                    }
                )
                continue
            successes.append(
                {
                    **candidate,
                    **features,
                    "gz3d_mask_sha256": mask_sha256,
                }
            )
        if len(successes) != per_cell:
            raise GravityItem14CoherenceError(
                f"Item 14 sample cell {cell} has {len(successes)} valid masks, needs {per_cell}"
            )
        for ordinal, row in enumerate(successes):
            role = "exploration" if ordinal < exploration_per_cell else "reserved_confirmation"
            outer_fold = ordinal % int(config["evaluation"]["outer_folds"])
            sample_row = {
                "mangaid": row["mangaid"],
                "plateifu": row["plateifu"],
                "ra": row["ra"],
                "dec": row["dec"],
                "sample_cell": cell,
                "bar_vote_state": row["bar_vote_state"],
                "stellar_mass_state": row["stellar_mass_state"],
                "gz3d_file_name": row["gz3d_file_name"],
                "gz3d_official_sha1": row["gz3d_official_sha1"],
                "gz3d_mask_sha256": row["gz3d_mask_sha256"],
                "role": role,
                "outer_fold": outer_fold,
                "response_read": False,
            }
            selected.append(sample_row)
            feature_records.append(_serialize(row))
    return selected, feature_records, failures


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    scale_min, scale_max = (float(value) for value in generator["scale_log_uniform"])
    power_min, power_max = (float(value) for value in generator["power_log_uniform"])
    return {
        "family": random.integers(0, len(generator["families"]), count, dtype=np.int16),
        "threshold": random.uniform(*generator["threshold_uniform"], count),
        "scale": np.exp(random.uniform(math.log(scale_min), math.log(scale_max), count)),
        "power": np.exp(random.uniform(math.log(power_min), math.log(power_max), count)),
        "phase": random.uniform(*generator["phase_uniform"], count),
        "modulation": random.integers(0, len(generator["modulations"]), count, dtype=np.int8),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("family", "threshold", "scale", "power", "phase", "modulation"):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_prepared_sources(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    _require_scientific_freeze()
    config = load_config(root)
    metadata_binding = config["sources"]["gz3d_metadata"]
    metadata_payload = _download(str(metadata_binding["url"]))
    if len(metadata_payload) != int(metadata_binding["file_bytes"]):
        raise GravityItem14CoherenceError("GZ3D metadata byte count changed")
    if hashlib.sha256(metadata_payload).hexdigest() != metadata_binding["file_sha256"]:
        raise GravityItem14CoherenceError("GZ3D metadata SHA256 changed")
    if hashlib.sha1(metadata_payload).hexdigest() != metadata_binding["official_sha1"]:
        raise GravityItem14CoherenceError("GZ3D metadata SHA1 changed")
    manifest_binding = config["sources"]["gz3d_sha1_manifest"]
    manifest_payload = _download(str(manifest_binding["url"]))
    if len(manifest_payload) != int(manifest_binding["file_bytes"]):
        raise GravityItem14CoherenceError("GZ3D SHA1 manifest byte count changed")
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_binding["file_sha256"]:
        raise GravityItem14CoherenceError("GZ3D SHA1 manifest changed")
    sha1_by_name = _parse_sha1_manifest(manifest_payload)
    if len(sha1_by_name) != int(manifest_binding["observed_lines"]):
        raise GravityItem14CoherenceError("GZ3D SHA1 manifest line count changed")
    if sha1_by_name.get("gz3d_metadata.fits") != metadata_binding["official_sha1"]:
        raise GravityItem14CoherenceError("GZ3D metadata official hash binding changed")
    metadata_path = root / config["outputs"]["metadata_raw"]
    manifest_path = root / config["outputs"]["sha1_manifest_raw"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_bytes(metadata_payload)
    manifest_path.write_bytes(manifest_payload)
    eligible, exclusion_counts = _eligible_metadata_records(
        root, config, metadata_payload, sha1_by_name
    )
    objects, features, mask_failures = _select_with_mask_features(root, eligible, config)
    cell_counts = Counter(row["sample_cell"] for row in objects)
    role_counts = Counter(row["role"] for row in objects)
    fold_counts = Counter(row["outer_fold"] for row in objects if row["role"] == "exploration")
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item14-gz3d-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "objects": sorted(objects, key=lambda row: (row["role"], row["plateifu"])),
            "cell_counts": dict(sorted(cell_counts.items())),
            "fold_counts_exploration": {
                str(key): value for key, value in sorted(fold_counts.items())
            },
            "exclusion_counts": exclusion_counts,
            "mask_failures_before_sample_freeze": mask_failures,
            "counts": {
                "eligible_before_mask_quality": len(eligible),
                "selected": len(objects),
                "exploration": role_counts["exploration"],
                "reserved_confirmation": role_counts["reserved_confirmation"],
                "response_rows_read": 0,
                "predecessor_selected": 0,
                "mask_failures_before_sample_freeze": len(mask_failures),
            },
            "claims": {"confirmation_opened": False},
        }
    )
    feature_source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item14-gz3d-mask-features-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "metadata_file_sha256": metadata_binding["file_sha256"],
            "sha1_manifest_file_sha256": manifest_binding["file_sha256"],
            "records": sorted(features, key=lambda row: str(row["plateifu"])),
            "counts": {
                "records": len(features),
                "mask_files_verified": len(features),
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "temporal_pattern_speed_measured": False,
                "orbital_resonance_established": False,
            },
        }
    )
    arrays = generate_candidates(config)
    families = config["candidate_generator"]["families"]
    family_counts = Counter(families[int(value)]["id"] for value in arrays["family"])
    origin_counts = Counter(families[int(value)]["origin_status"] for value in arrays["family"])
    candidates = _content_hashed(
        {
            "schema_version": "invariant-gravity-item14-gz3d-coherence-candidates-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "family_counts": dict(sorted(family_counts.items())),
            "origin_status_counts": dict(sorted(origin_counts.items())),
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "counts": {
                "candidate_cells": len(arrays["family"]),
                "response_rows_read": 0,
                "post_response_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"historical_novelty_established": False},
        }
    )
    sample_path = root / config["outputs"]["sample_manifest"]
    feature_path = root / config["outputs"]["mask_feature_source"]
    candidate_path = root / config["outputs"]["candidate_manifest"]
    for path, artifact in (
        (sample_path, sample),
        (feature_path, feature_source),
        (candidate_path, candidates),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    return sample_path, feature_path, candidate_path


def validate_prepared_sources(
    sample: Mapping[str, Any],
    features: Mapping[str, Any],
    candidates: Mapping[str, Any],
    root: Path,
) -> None:
    config = load_config(root)
    for value, label in (
        (sample, "Item 14 sample manifest"),
        (features, "Item 14 mask feature source"),
        (candidates, "Item 14 candidate manifest"),
    ):
        _validate_content_hash(value, label)
        if value["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
            raise GravityItem14CoherenceError(f"{label} scientific binding changed")
    if (
        _sha256_file(root / config["outputs"]["metadata_raw"])
        != config["sources"]["gz3d_metadata"]["file_sha256"]
    ):
        raise GravityItem14CoherenceError("stored GZ3D metadata changed")
    if (
        _sha256_file(root / config["outputs"]["sha1_manifest_raw"])
        != config["sources"]["gz3d_sha1_manifest"]["file_sha256"]
    ):
        raise GravityItem14CoherenceError("stored GZ3D SHA1 manifest changed")
    expected_total = int(config["sample"]["maximum_total_objects"])
    if sample["counts"]["selected"] != expected_total:
        raise GravityItem14CoherenceError("Item 14 selected sample count changed")
    if sample["counts"]["exploration"] != int(config["sample"]["exploration_objects"]):
        raise GravityItem14CoherenceError("Item 14 exploration count changed")
    if sample["counts"]["reserved_confirmation"] != int(config["sample"]["confirmation_objects"]):
        raise GravityItem14CoherenceError("Item 14 confirmation count changed")
    if sample["counts"]["response_rows_read"] != 0:
        raise GravityItem14CoherenceError("response entered Item 14 sample freeze")
    if sample["counts"]["predecessor_selected"] != 0:
        raise GravityItem14CoherenceError("predecessor entered Item 14 sample")
    expected_per_cell = int(config["sample"]["objects_per_cell"])
    if set(sample["cell_counts"].values()) != {expected_per_cell}:
        raise GravityItem14CoherenceError("Item 14 cell balance changed")
    expected_per_fold = int(config["sample"]["exploration_objects"]) // int(
        config["evaluation"]["outer_folds"]
    )
    if set(sample["fold_counts_exploration"].values()) != {expected_per_fold}:
        raise GravityItem14CoherenceError("Item 14 fold balance changed")
    sample_ids = {row["plateifu"] for row in sample["objects"]}
    feature_ids = {row["plateifu"] for row in features["records"]}
    if (
        sample_ids != feature_ids
        or len(sample_ids) != expected_total
        or len(sample["objects"]) != expected_total
        or len(features["records"]) != expected_total
    ):
        raise GravityItem14CoherenceError("Item 14 feature/sample identities changed")
    if features["counts"]["response_rows_read"] != 0:
        raise GravityItem14CoherenceError("response entered Item 14 mask features")
    if features["counts"]["mask_files_verified"] != expected_total:
        raise GravityItem14CoherenceError("Item 14 verified mask count changed")
    arrays = generate_candidates(config)
    if candidates["candidate_digest_sha256"] != _candidate_digest(arrays):
        raise GravityItem14CoherenceError("Item 14 candidate digest changed")
    if candidates["counts"]["candidate_cells"] != 262144:
        raise GravityItem14CoherenceError("Item 14 candidate count changed")
    if candidates["counts"]["post_response_cells"] != 0:
        raise GravityItem14CoherenceError("post-response formula entered Item 14")
    families = config["candidate_generator"]["families"]
    expected_family_counts = Counter(families[int(value)]["id"] for value in arrays["family"])
    expected_origin_counts = Counter(
        families[int(value)]["origin_status"] for value in arrays["family"]
    )
    if candidates["family_counts"] != dict(sorted(expected_family_counts.items())):
        raise GravityItem14CoherenceError("Item 14 candidate family counts changed")
    if candidates["origin_status_counts"] != dict(sorted(expected_origin_counts.items())):
        raise GravityItem14CoherenceError("Item 14 candidate provenance counts changed")
    if candidates["counts"]["paid_model_calls"] != 0:
        raise GravityItem14CoherenceError("paid model call entered Item 14 candidates")


def _load_prepared(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(root)
    sample = json.loads((root / config["outputs"]["sample_manifest"]).read_text(encoding="utf-8"))
    features = json.loads(
        (root / config["outputs"]["mask_feature_source"]).read_text(encoding="utf-8")
    )
    candidates = json.loads(
        (root / config["outputs"]["candidate_manifest"]).read_text(encoding="utf-8")
    )
    validate_prepared_sources(sample, features, candidates, root)
    return sample, features, candidates


def _channel_index(hdu: Any, label: str) -> int:
    channels = int(hdu.header.get("NAXIS3", 1))
    for ordinal in range(1, channels + 1):
        if str(hdu.header.get(f"C{ordinal}", "")).strip() == label:
            return ordinal - 1
    raise GravityItem14CoherenceError(f"required MAPS channel {label!r} missing from {hdu.name}")


def _maps_location(config: Mapping[str, Any], plateifu: str) -> tuple[str, str]:
    parts = str(plateifu).split("-")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise GravityItem14CoherenceError("invalid MaNGA plateifu for MAPS response")
    plate, ifu = parts
    source = config["sources"]["response"]
    filename = str(source["filename_template"]).format(plateifu=plateifu, daptype=source["daptype"])
    url = "/".join(
        (
            str(source["base_url"]).rstrip("/"),
            str(source["daptype"]),
            plate,
            ifu,
            filename,
        )
    )
    return filename, url


def _maps_payload(
    root: Path, record: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[bytes, str, str]:
    filename, url = _maps_location(config, str(record["plateifu"]))
    cache = root / config["sources"]["response"]["raw_cache"]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / filename
    if path.exists():
        payload = path.read_bytes()
    else:
        attempts = int(config["sources"]["response"]["download_attempts"])
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                payload = _download(url)
                break
            except (OSError, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(float(attempt + 1))
        else:
            raise GravityItem14CoherenceError(
                f"failed to acquire MaNGA MAPS response after {attempts} attempts"
            ) from last_error
        path.write_bytes(payload)
    if not payload:
        raise GravityItem14CoherenceError("empty MaNGA MAPS response")
    return payload, filename, url


def _annular_span(
    velocities: np.ndarray,
    radii: np.ndarray,
    azimuths: np.ndarray,
    bounds: Sequence[float],
    minimum_count: int,
    minimum_quadrants: int,
    quantiles: Sequence[float],
    label: str,
) -> dict[str, float | int]:
    lower, upper = (float(value) for value in bounds)
    selected = (
        np.isfinite(velocities)
        & np.isfinite(radii)
        & np.isfinite(azimuths)
        & (radii >= lower)
        & (radii < upper)
    )
    count = int(np.sum(selected))
    if count < minimum_count:
        raise GravityItem14CoherenceError(f"insufficient {label} measurements")
    quadrants = np.floor(np.mod(azimuths[selected], 360.0) / 90.0).astype(int)
    quadrant_count = len({int(value) for value in quadrants})
    if quadrant_count < minimum_quadrants:
        raise GravityItem14CoherenceError(f"insufficient {label} azimuth coverage")
    q_low, q_high = (float(value) for value in quantiles)
    low, high = np.quantile(velocities[selected], [q_low, q_high])
    span = float(high - low)
    if not math.isfinite(span) or span <= 0:
        raise GravityItem14CoherenceError(f"invalid {label} velocity span")
    return {
        "measurements": count,
        "azimuth_quadrants": quadrant_count,
        "velocity_span_km_s": span,
    }


def _unique_stellar_measurements(
    bin_ids: np.ndarray,
    valid: np.ndarray,
    inverse_variance: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    azimuths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_ids = bin_ids.ravel()
    flat_valid = valid.ravel()
    flat_ivar = inverse_variance.ravel()
    indices = []
    for bin_id in np.unique(flat_ids[flat_valid]):
        members = np.flatnonzero(flat_valid & (flat_ids == bin_id))
        indices.append(int(members[int(np.argmax(flat_ivar[members]))]))
    chosen = np.asarray(indices, dtype=int)
    return velocities.ravel()[chosen], radii.ravel()[chosen], azimuths.ravel()[chosen]


def derive_radial_response(
    compressed_payload: bytes,
    sample_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from astropy.io import fits

        payload = gzip.decompress(compressed_payload)
        with fits.open(io.BytesIO(payload), memmap=False) as hdus:
            source = config["sources"]["response"]
            primary = hdus[0].header
            for key, expected in source["required_primary_headers"].items():
                if str(primary.get(key, "")).strip() != str(expected):
                    raise GravityItem14CoherenceError(f"MaNGA MAPS header {key} changed")
            if str(primary.get("PLATEIFU", "")).strip() != str(sample_row["plateifu"]):
                raise GravityItem14CoherenceError("MaNGA MAPS plateifu changed")
            if (
                str(primary.get("MANGAID", "")).strip().upper()
                != str(sample_row["mangaid"]).upper()
            ):
                raise GravityItem14CoherenceError("MaNGA MAPS mangaid changed")
            required = [hdus[name] for name in source["required_extensions"]]
            if bool(source["fits_checksum_required"]):
                for hdu in [hdus[0], *required]:
                    if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                        raise GravityItem14CoherenceError("MaNGA MAPS FITS checksum failed")

            channels = source["channels"]
            radius_label = str(channels["radius_re"])
            bin_radius_hdu = hdus["BIN_LWELLCOO"]
            spaxel_radius_hdu = hdus["SPX_ELLCOO"]
            stellar_bin_hdu = hdus["BINID"]
            stellar_fom_hdu = hdus["STELLAR_FOM"]
            halpha_label = str(channels["halpha"])
            halpha_velocity_hdu = hdus["EMLINE_GVEL"]

            stellar_radius = np.asarray(
                bin_radius_hdu.data[_channel_index(bin_radius_hdu, radius_label)],
                dtype=np.float64,
            )
            stellar_azimuth = np.asarray(
                bin_radius_hdu.data[_channel_index(bin_radius_hdu, str(channels["bin_azimuth"]))],
                dtype=np.float64,
            )
            stellar_bin = np.asarray(
                stellar_bin_hdu.data[
                    _channel_index(stellar_bin_hdu, str(channels["stellar_bin_id"]))
                ],
                dtype=np.int64,
            )
            stellar_velocity = np.asarray(hdus["STELLAR_VEL"].data, dtype=np.float64)
            stellar_ivar = np.asarray(hdus["STELLAR_VEL_IVAR"].data, dtype=np.float64)
            stellar_mask = np.asarray(hdus["STELLAR_VEL_MASK"].data, dtype=np.int64)
            stellar_snr = np.asarray(hdus["BIN_SNR"].data, dtype=np.float64)
            stellar_rchi2 = np.asarray(
                stellar_fom_hdu.data[
                    _channel_index(stellar_fom_hdu, str(channels["stellar_rchi2"]))
                ],
                dtype=np.float64,
            )

            halpha_channel = _channel_index(halpha_velocity_hdu, halpha_label)
            spaxel_radius = np.asarray(
                spaxel_radius_hdu.data[_channel_index(spaxel_radius_hdu, radius_label)],
                dtype=np.float64,
            )
            spaxel_azimuth = np.asarray(
                spaxel_radius_hdu.data[
                    _channel_index(spaxel_radius_hdu, str(channels["spaxel_azimuth"]))
                ],
                dtype=np.float64,
            )
            halpha_velocity = np.asarray(halpha_velocity_hdu.data[halpha_channel], dtype=np.float64)
            halpha_ivar = np.asarray(
                hdus["EMLINE_GVEL_IVAR"].data[halpha_channel], dtype=np.float64
            )
            halpha_mask = np.asarray(hdus["EMLINE_GVEL_MASK"].data[halpha_channel], dtype=np.int64)
            halpha_anr = np.asarray(hdus["EMLINE_GANR"].data[halpha_channel], dtype=np.float64)
            halpha_ew = np.asarray(hdus["EMLINE_GEW"].data[halpha_channel], dtype=np.float64)
            halpha_ew_mask = np.asarray(
                hdus["EMLINE_GEW_MASK"].data[halpha_channel], dtype=np.int64
            )
            halpha_rchi2 = np.asarray(hdus["EMLINE_LFOM"].data[halpha_channel], dtype=np.float64)
            drp3qual = int(primary.get("DRP3QUAL", -1))
            dapqual = int(primary.get("DAPQUAL", -1))
    except GravityItem14CoherenceError:
        raise
    except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
        raise GravityItem14CoherenceError("invalid MaNGA MAPS response FITS") from exc

    shape = stellar_velocity.shape
    arrays = (
        stellar_radius,
        stellar_azimuth,
        stellar_bin,
        stellar_ivar,
        stellar_mask,
        stellar_snr,
        stellar_rchi2,
        spaxel_radius,
        spaxel_azimuth,
        halpha_velocity,
        halpha_ivar,
        halpha_mask,
        halpha_anr,
        halpha_ew,
        halpha_ew_mask,
        halpha_rchi2,
    )
    if any(array.shape != shape for array in arrays):
        raise GravityItem14CoherenceError("MaNGA MAPS response array shape changed")
    quality = config["quality"]
    stellar_valid = (
        (stellar_bin >= 0)
        & (stellar_mask == 0)
        & (stellar_ivar > 0)
        & (stellar_snr >= float(quality["minimum_stellar_bin_snr"]))
        & (stellar_rchi2 >= 0)
        & (stellar_rchi2 <= float(quality["maximum_stellar_rchi2"]))
        & np.isfinite(stellar_velocity)
        & np.isfinite(stellar_radius)
        & np.isfinite(stellar_azimuth)
        & np.isfinite(stellar_ivar)
        & np.isfinite(stellar_snr)
        & np.isfinite(stellar_rchi2)
    )
    stellar_values, stellar_radii, stellar_azimuths = _unique_stellar_measurements(
        stellar_bin,
        stellar_valid,
        stellar_ivar,
        stellar_velocity,
        stellar_radius,
        stellar_azimuth,
    )
    halpha_valid = (
        (halpha_mask == 0)
        & (halpha_ew_mask == 0)
        & (halpha_ivar > 0)
        & (halpha_anr >= float(quality["minimum_halpha_amplitude_to_noise"]))
        & (halpha_ew >= float(quality["minimum_halpha_equivalent_width_angstrom"]))
        & (halpha_rchi2 >= 0)
        & (halpha_rchi2 <= float(quality["maximum_halpha_local_rchi2"]))
        & np.isfinite(halpha_velocity)
        & np.isfinite(spaxel_radius)
        & np.isfinite(spaxel_azimuth)
        & np.isfinite(halpha_ivar)
        & np.isfinite(halpha_anr)
        & np.isfinite(halpha_ew)
        & np.isfinite(halpha_rchi2)
    )
    minimum_quadrants = int(quality["minimum_azimuth_quadrants_per_annulus"])
    quantiles = quality["velocity_span_quantiles"]
    stellar_summaries = {}
    halpha_summaries = {}
    for annulus in ("inner", "outer"):
        bounds = quality[f"{annulus}_annulus_re"]
        stellar_summaries[annulus] = _annular_span(
            stellar_values,
            stellar_radii,
            stellar_azimuths,
            bounds,
            int(quality["minimum_unique_stellar_bins_per_annulus"]),
            minimum_quadrants,
            quantiles,
            f"stellar {annulus}-annulus",
        )
        halpha_summaries[annulus] = _annular_span(
            halpha_velocity[halpha_valid],
            spaxel_radius[halpha_valid],
            spaxel_azimuth[halpha_valid],
            bounds,
            int(quality["minimum_halpha_spaxels_per_annulus"]),
            minimum_quadrants,
            quantiles,
            f"H-alpha {annulus}-annulus",
        )
    stellar_ratio = float(
        stellar_summaries["outer"]["velocity_span_km_s"]
        / stellar_summaries["inner"]["velocity_span_km_s"]
    )
    halpha_ratio = float(
        halpha_summaries["outer"]["velocity_span_km_s"]
        / halpha_summaries["inner"]["velocity_span_km_s"]
    )
    return _serialize(
        {
            "plateifu": sample_row["plateifu"],
            "mangaid": sample_row["mangaid"],
            "drp3qual": drp3qual,
            "dapqual": dapqual,
            "stellar_inner_measurements": stellar_summaries["inner"]["measurements"],
            "stellar_outer_measurements": stellar_summaries["outer"]["measurements"],
            "stellar_inner_azimuth_quadrants": stellar_summaries["inner"]["azimuth_quadrants"],
            "stellar_outer_azimuth_quadrants": stellar_summaries["outer"]["azimuth_quadrants"],
            "stellar_inner_velocity_span_km_s": stellar_summaries["inner"]["velocity_span_km_s"],
            "stellar_outer_velocity_span_km_s": stellar_summaries["outer"]["velocity_span_km_s"],
            "stellar_outer_to_inner_span_ratio": stellar_ratio,
            "halpha_inner_measurements": halpha_summaries["inner"]["measurements"],
            "halpha_outer_measurements": halpha_summaries["outer"]["measurements"],
            "halpha_inner_azimuth_quadrants": halpha_summaries["inner"]["azimuth_quadrants"],
            "halpha_outer_azimuth_quadrants": halpha_summaries["outer"]["azimuth_quadrants"],
            "halpha_inner_velocity_span_km_s": halpha_summaries["inner"]["velocity_span_km_s"],
            "halpha_outer_velocity_span_km_s": halpha_summaries["outer"]["velocity_span_km_s"],
            "halpha_outer_to_inner_span_ratio": halpha_ratio,
        }
    )


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem14CoherenceError("Item 14 sample freeze is not bound")
    config = load_config(root)
    sample, _, _ = _load_prepared(root)
    exploration = sorted(
        (row for row in sample["objects"] if row["role"] == "exploration"),
        key=lambda row: str(row["plateifu"]),
    )
    confirmation = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    records = []
    failures = []
    files = []
    for sample_row in exploration:
        plateifu = str(sample_row["plateifu"])
        if plateifu in confirmation:
            raise GravityItem14CoherenceError("Item 14 confirmation entered response query")
        payload, filename, url = _maps_payload(root, sample_row, config)
        file_record = {
            "plateifu": plateifu,
            "file_name": filename,
            "url": url,
            "file_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
        try:
            response = derive_radial_response(payload, sample_row, config)
        except GravityItem14CoherenceError as exc:
            failures.append({**file_record, "reason": str(exc)})
            continue
        records.append({**response, **file_record, "fits_checksum_verified": True})
        files.append(
            {
                **file_record,
                "fits_checksum_verified": True,
            }
        )
    if confirmation & {str(row["plateifu"]) for row in [*records, *failures]}:
        raise GravityItem14CoherenceError("Item 14 confirmation response entered source")
    source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item14-manga-maps-response-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "data_model": config["sources"]["response"]["data_model"],
            "files": files,
            "records": sorted(records, key=lambda row: str(row["plateifu"])),
            "failures": sorted(failures, key=lambda row: str(row["plateifu"])),
            "counts": {
                "exploration_response_objects_attempted": len(exploration),
                "exploration_response_objects_parsed": len(records),
                "exploration_response_failures": len(failures),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"confirmation_opened": False},
        }
    )
    path = root / config["outputs"]["response_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(source) + b"\n")
    return path


def validate_response_source(source: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(source, "Item 14 response source")
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem14CoherenceError("Item 14 scientific response binding changed")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem14CoherenceError("Item 14 sample response binding changed")
    if source["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem14CoherenceError("Item 14 confirmation opened")
    if source["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem14CoherenceError("post-response formula entered Item 14")
    if source["counts"]["paid_model_calls"] != 0:
        raise GravityItem14CoherenceError("paid model call entered Item 14 response")
    sample, _, _ = _load_prepared(root)
    exploration = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    }
    confirmation = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    record_ids = [str(row["plateifu"]) for row in source["records"]]
    failure_ids = [str(row["plateifu"]) for row in source["failures"]]
    observed = record_ids + failure_ids
    if len(observed) != len(set(observed)) or set(observed) != exploration:
        raise GravityItem14CoherenceError("Item 14 MAPS response identity set changed")
    if confirmation & set(observed):
        raise GravityItem14CoherenceError("Item 14 confirmation MAPS response opened")
    if source["counts"]["exploration_response_objects_attempted"] != len(exploration):
        raise GravityItem14CoherenceError("Item 14 MAPS attempt count changed")
    if source["counts"]["exploration_response_objects_parsed"] != len(record_ids):
        raise GravityItem14CoherenceError("Item 14 MAPS parsed count changed")
    if source["counts"]["exploration_response_failures"] != len(failure_ids):
        raise GravityItem14CoherenceError("Item 14 MAPS failure count changed")
    file_ids = [str(row["plateifu"]) for row in source["files"]]
    if set(file_ids) != set(record_ids) or len(file_ids) != len(record_ids):
        raise GravityItem14CoherenceError("Item 14 MAPS file receipt set changed")
    if any(not bool(row["fits_checksum_verified"]) for row in source["files"]):
        raise GravityItem14CoherenceError("Item 14 MAPS checksum receipt changed")
    if any(bool(value) for value in source["claims"].values()):
        raise GravityItem14CoherenceError("Item 14 response contains an overclaim")


def extract_rows(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    sample, features, _ = _load_prepared(root)
    response = json.loads((root / config["outputs"]["response_source"]).read_text(encoding="utf-8"))
    validate_response_source(response, root)
    feature_by_plate = {str(row["plateifu"]): row for row in features["records"]}
    sample_by_plate = {
        str(row["plateifu"]): row for row in sample["objects"] if row["role"] == "exploration"
    }
    records = []
    failures = [
        {"plateifu": str(row["plateifu"]), "reasons": [str(row["reason"])]}
        for row in response["failures"]
    ]
    quality = config["quality"]
    for raw in response["records"]:
        plateifu = str(raw["plateifu"])
        feature = feature_by_plate[plateifu]
        sample_row = sample_by_plate[plateifu]
        reasons = []
        if int(raw["drp3qual"]) != int(quality["required_drp3qual"]):
            reasons.append("drp3qual")
        if int(raw["dapqual"]) != int(quality["required_dapqual"]):
            reasons.append("dapqual")
        minimum_span = float(quality["minimum_annular_velocity_span_km_s"])
        for tracer, maximum in (
            ("stellar", float(quality["maximum_stellar_annular_velocity_span_km_s"])),
            ("halpha", float(quality["maximum_halpha_annular_velocity_span_km_s"])),
        ):
            for annulus in ("inner", "outer"):
                span = float(raw[f"{tracer}_{annulus}_velocity_span_km_s"])
                if not (minimum_span <= span <= maximum):
                    reasons.append(f"{tracer}_{annulus}_span")
            ratio = float(raw[f"{tracer}_outer_to_inner_span_ratio"])
            if not (
                math.isfinite(ratio)
                and float(quality["minimum_outer_to_inner_span_ratio"])
                <= ratio
                <= float(quality["maximum_outer_to_inner_span_ratio"])
            ):
                reasons.append(f"{tracer}_outer_inner_ratio")
        output = {
            **feature,
            **raw,
            "outer_fold": sample_row["outer_fold"],
            "quality_pass": not reasons,
            "quality_failure_reasons": reasons,
        }
        records.append(_serialize(output))
        if reasons:
            failures.append({"plateifu": plateifu, "reasons": reasons})
    passing = sum(bool(row["quality_pass"]) for row in records)
    expected = int(config["sample"]["exploration_objects"])
    retention = passing / expected
    passing_rows = [row for row in records if row["quality_pass"]]
    fold_counts = Counter(int(row["outer_fold"]) for row in passing_rows)
    stratum_counts = {
        "bar_low": sum(row["bar_vote_state"] == "bar_low" for row in passing_rows),
        "bar_high": sum(row["bar_vote_state"] == "bar_high" for row in passing_rows),
        "lower_mass": sum(row["stellar_mass_state"] == "lower_mass" for row in passing_rows),
        "higher_mass": sum(row["stellar_mass_state"] == "higher_mass" for row in passing_rows),
        "prior_age_low": sum(
            float(row["prior_age_lead"]) <= float(config["sample"]["prior_age_threshold"])
            for row in passing_rows
        ),
        "prior_age_high": sum(
            float(row["prior_age_lead"]) > float(config["sample"]["prior_age_threshold"])
            for row in passing_rows
        ),
    }
    fold_quality = set(fold_counts) == set(range(int(config["evaluation"]["outer_folds"])))
    fold_quality = fold_quality and min(fold_counts.values(), default=0) >= int(
        quality["minimum_quality_passing_per_outer_fold"]
    )
    stratum_quality = min(stratum_counts.values(), default=0) >= int(
        quality["minimum_quality_passing_per_gate_stratum"]
    )
    quality_pass = passing >= int(
        quality["minimum_quality_passing_exploration_galaxies"]
    ) and retention >= float(quality["minimum_quality_retention_fraction"])
    quality_pass = quality_pass and fold_quality and stratum_quality
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item14-gz3d-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": (
                "PASS_ITEM14_GZ3D_QUALITY" if quality_pass else "INCONCLUSIVE_ITEM14_GZ3D_QUALITY"
            ),
            "records": sorted(records, key=lambda row: str(row["plateifu"])),
            "failures": failures,
            "quality_fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
            "quality_gate_stratum_counts": dict(sorted(stratum_counts.items())),
            "counts": {
                "exploration_response_objects_attempted": response["counts"][
                    "exploration_response_objects_attempted"
                ],
                "exploration_response_objects_parsed": len(response["records"]),
                "response_parse_failures": len(response["failures"]),
                "quality_passing_galaxies": passing,
                "quality_failed_galaxies": expected - passing,
                "quality_retention_fraction": _metric(retention),
                "predecessor_selected": 0,
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": config["claim_boundaries"],
        }
    )
    path = root / config["outputs"]["extraction_summary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return path


def _fixed_array(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], field: str
) -> np.ndarray:
    center, scale = (
        float(value) for value in config["evaluation"]["fixed_feature_normalization"][field]
    )
    if scale <= 0:
        raise GravityItem14CoherenceError("fixed mask feature normalization changed")
    return (np.asarray([float(row[field]) for row in rows]) - center) / scale


def _load_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = json.loads(
        (root / config["outputs"]["extraction_summary"]).read_text(encoding="utf-8")
    )
    _validate_content_hash(summary, "Item 14 extraction summary")
    if summary["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem14CoherenceError("Item 14 extraction scientific binding changed")
    if summary["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem14CoherenceError("Item 14 extraction sample binding changed")
    if summary["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem14CoherenceError("Item 14 extraction opened confirmation")
    if summary["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem14CoherenceError("post-response formula entered Item 14 extraction")
    if summary["counts"]["paid_model_calls"] != 0:
        raise GravityItem14CoherenceError("paid model call entered Item 14 extraction")
    if any(bool(value) for value in summary["claims"].values()):
        raise GravityItem14CoherenceError("Item 14 extraction contains an overclaim")
    _, _, candidates = _load_prepared(root)
    rows = [row for row in summary["records"] if row["quality_pass"]]
    if not rows:
        raise GravityItem14CoherenceError("no Item 14 quality rows")
    structural = np.column_stack(
        [
            np.asarray([float(row[field]) for row in rows])
            for field in config["evaluation"]["structural_features"]
        ]
    )
    modulation_config = config["evaluation"]["fixed_modulation_normalization"]

    def modulation(field: str, values: np.ndarray) -> np.ndarray:
        center, scale = (float(value) for value in modulation_config[field])
        if scale <= 0:
            raise GravityItem14CoherenceError("fixed modulation normalization changed")
        return (values - center) / scale

    stellar_ratio = np.asarray([float(row["stellar_outer_to_inner_span_ratio"]) for row in rows])
    halpha_ratio = np.asarray([float(row["halpha_outer_to_inner_span_ratio"]) for row in rows])
    return {
        "summary": summary,
        "candidate_manifest": candidates,
        "rows": rows,
        "folds": np.asarray([int(row["outer_fold"]) for row in rows]),
        "y": np.log10(stellar_ratio),
        "y_halpha": np.log10(halpha_ratio),
        "design_control": structural,
        "design_secondary": structural,
        "mode_m1": _fixed_array(rows, config, "mode_m1"),
        "mode_m2": _fixed_array(rows, config, "mode_m2"),
        "mode_m3": _fixed_array(rows, config, "mode_m3"),
        "mode_m4": _fixed_array(rows, config, "mode_m4"),
        "mode_entropy": _fixed_array(rows, config, "mode_entropy"),
        "phase_linearity": _fixed_array(rows, config, "m2_phase_linearity"),
        "pitch_abs": _fixed_array(rows, config, "m2_pitch_abs"),
        "phase_twist": _fixed_array(rows, config, "m2_phase_twist"),
        "bar_phase_lock": _fixed_array(rows, config, "bar_spiral_phase_lock"),
        "bar_radius_ratio": _fixed_array(rows, config, "bar_spiral_log_radius_ratio"),
        "coverage": _fixed_array(rows, config, "spiral_radial_coverage_re"),
        "bar_vote_modulation": modulation(
            "bar_vote_fraction",
            np.asarray([float(row["bar_vote_fraction"]) for row in rows]),
        ),
        "surface_modulation": modulation(
            "stellar_surface_density",
            np.asarray([float(row["log_surface_density"]) for row in rows]),
        ),
        "mass_modulation": modulation(
            "stellar_mass",
            np.asarray([float(row["log_stellar_mass"]) for row in rows]),
        ),
        "age_modulation": modulation(
            "prior_age_lead",
            np.asarray([float(row["prior_age_lead"]) for row in rows]),
        ),
        "coverage_modulation": modulation(
            "spiral_radial_coverage",
            np.asarray([float(row["spiral_radial_coverage_re"]) for row in rows]),
        ),
        "bar_state": np.asarray(
            [1.0 if row["bar_vote_state"] == "bar_high" else -1.0 for row in rows]
        ),
        "mass": np.asarray([float(row["log_stellar_mass"]) for row in rows]),
        "prior_age": np.asarray([float(row["prior_age_lead"]) for row in rows]),
    }


def _candidate_components(
    arrays: Mapping[str, np.ndarray],
    data: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    family = xp.asarray(arrays["family"][begin:end], dtype=xp.int32)[:, None]
    threshold = xp.asarray(arrays["threshold"][begin:end], dtype=xp.float64)[:, None]
    scale = xp.asarray(arrays["scale"][begin:end], dtype=xp.float64)[:, None]
    power = xp.asarray(arrays["power"][begin:end], dtype=xp.float64)[:, None]
    phase = xp.asarray(arrays["phase"][begin:end], dtype=xp.float64)[:, None]
    modulation_index = xp.asarray(arrays["modulation"][begin:end], dtype=xp.int32)[:, None]

    def value(key: str) -> Any:
        return xp.asarray(data[key], dtype=xp.float64)[None, :]

    m1 = value("mode_m1")
    m2 = value("mode_m2")
    m3 = value("mode_m3")
    m4 = value("mode_m4")
    entropy = value("mode_entropy")
    linearity = value("phase_linearity")
    pitch = value("pitch_abs")
    twist = value("phase_twist")
    bar_lock = value("bar_phase_lock")
    radius_ratio = value("bar_radius_ratio")
    coverage = value("coverage")

    def signed_power(raw: Any) -> Any:
        z = (raw - threshold) / scale
        magnitude = xp.abs(z) ** power
        return xp.sign(z) * magnitude / (1.0 + magnitude)

    two_arm = signed_power(m2)
    three_arm = signed_power(m3)
    four_arm = signed_power(m4)
    hierarchy = signed_power(m2 - 0.5 * (m1 + m3))
    radial_lock = signed_power(linearity)
    pitch_coherence = signed_power(linearity / (1.0 + xp.abs(pitch - twist)))
    bar_spiral_lock = signed_power(bar_lock)
    resonance_ratio = xp.exp(-0.5 * ((radius_ratio - threshold) / scale) ** 2) ** power
    persistence = signed_power(linearity * coverage)
    coupling = signed_power(m2 * m4)
    entropy_suppression = xp.exp(-xp.abs((entropy - threshold) / scale)) ** power
    log_periodic = linearity * xp.cos(phase + power * xp.log1p(xp.abs(pitch) / scale))
    components = xp.where(family == 0, two_arm, three_arm)
    components = xp.where(family == 2, four_arm, components)
    components = xp.where(family == 3, hierarchy, components)
    components = xp.where(family == 4, radial_lock, components)
    components = xp.where(family == 5, pitch_coherence, components)
    components = xp.where(family == 6, bar_spiral_lock, components)
    components = xp.where(family == 7, resonance_ratio, components)
    components = xp.where(family == 8, persistence, components)
    components = xp.where(family == 9, coupling, components)
    components = xp.where(family == 10, entropy_suppression, components)
    components = xp.where(family == 11, log_periodic, components)
    modulations = xp.stack(
        (
            xp.ones_like(value("bar_vote_modulation")),
            xp.tanh(value("bar_vote_modulation")),
            xp.tanh(value("surface_modulation")),
            xp.tanh(value("mass_modulation")),
            xp.tanh(value("age_modulation")),
            xp.tanh(value("coverage_modulation")),
        ),
        axis=0,
    )[:, 0, :]
    selected_modulation = xp.take_along_axis(
        modulations[None, :, :], modulation_index[:, :, None], axis=1
    )[:, 0, :]
    return components * selected_modulation


def _fit_component(
    component: np.ndarray, residual: np.ndarray, ridge: float
) -> tuple[float, float, float]:
    mean = float(np.mean(component))
    scale = max(float(np.std(component)), 1e-12)
    standardized = (component - mean) / scale
    coefficient = float(np.sum(standardized * residual) / (np.sum(standardized**2) + ridge))
    return mean, scale, coefficient


def _nested_select(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    try:
        import cupy as xp

        if int(xp.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("no CUDA device")
        backend = "gpu_cupy"
        device = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except (ImportError, RuntimeError):
        xp = np
        backend = "cpu_numpy"
        device = None
    arrays = generate_candidates(config)
    candidate_count = len(arrays["family"])
    folds = np.asarray(data["folds"])
    y = np.asarray(data["y"])
    y_halpha = np.asarray(data["y_halpha"])
    control = np.asarray(data["design_control"])
    secondary = np.asarray(data["design_secondary"])
    predictions = {
        key: np.full(len(y), np.nan)
        for key in ("control", "full", "secondary_control", "secondary_full")
    }
    selections = []
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["coherence_coefficient_ridge"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    component_crosscheck = 0.0
    for outer in range(outer_folds):
        inner_records = []
        for inner in [value for value in range(outer_folds) if value != outer]:
            train = (folds != outer) & (folds != inner)
            validation = folds == inner
            model = _ridge_fit(control[train], y[train], alpha)
            inner_records.append(
                {
                    "train": train,
                    "validation": validation,
                    "train_residual": y[train] - _ridge_predict(model, control[train]),
                    "validation_residual": y[validation]
                    - _ridge_predict(model, control[validation]),
                }
            )
        scores = np.full(candidate_count, np.inf)
        for begin in range(0, candidate_count, batch_size):
            end = min(begin + batch_size, candidate_count)
            components = _candidate_components(arrays, data, begin, end, xp)
            loss = xp.zeros(end - begin, dtype=xp.float64)
            for inner in inner_records:
                train_component = components[:, inner["train"]]
                validation_component = components[:, inner["validation"]]
                mean = xp.mean(train_component, axis=1)
                scale = xp.maximum(xp.std(train_component, axis=1), 1e-12)
                standardized = (train_component - mean[:, None]) / scale[:, None]
                coefficient = xp.sum(
                    standardized * xp.asarray(inner["train_residual"])[None, :], axis=1
                ) / (xp.sum(standardized**2, axis=1) + coefficient_ridge)
                residual = (
                    xp.asarray(inner["validation_residual"])[None, :]
                    - coefficient[:, None] * (validation_component - mean[:, None]) / scale[:, None]
                )
                loss += xp.mean(residual**2, axis=1)
            batch_scores = loss / len(inner_records)
            scores[begin:end] = xp.asnumpy(batch_scores) if backend == "gpu_cupy" else batch_scores
            if begin == 0:
                check_count = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), end)
                cpu = _candidate_components(arrays, data, 0, check_count, np)
                observed = (
                    xp.asnumpy(components[:check_count])
                    if backend == "gpu_cupy"
                    else components[:check_count]
                )
                component_crosscheck = max(
                    component_crosscheck, float(np.max(np.abs(cpu - observed)))
                )
        selected = int(np.argmin(scores))
        train = folds != outer
        test = folds == outer
        control_model = _ridge_fit(control[train], y[train], alpha)
        control_train = _ridge_predict(control_model, control[train])
        predictions["control"][test] = _ridge_predict(control_model, control[test])
        selected_component = _candidate_components(arrays, data, selected, selected + 1, np)[0]
        mean, scale, coefficient = _fit_component(
            selected_component[train], y[train] - control_train, coefficient_ridge
        )
        predictions["full"][test] = (
            predictions["control"][test] + coefficient * (selected_component[test] - mean) / scale
        )
        secondary_model = _ridge_fit(secondary[train], y_halpha[train], alpha)
        secondary_train = _ridge_predict(secondary_model, secondary[train])
        predictions["secondary_control"][test] = _ridge_predict(secondary_model, secondary[test])
        secondary_mean, secondary_scale, secondary_coefficient = _fit_component(
            selected_component[train],
            y_halpha[train] - secondary_train,
            coefficient_ridge,
        )
        predictions["secondary_full"][test] = (
            predictions["secondary_control"][test]
            + secondary_coefficient * (selected_component[test] - secondary_mean) / secondary_scale
        )
        family = config["candidate_generator"]["families"][int(arrays["family"][selected])]
        selections.append(
            {
                "outer_fold": outer,
                "selected_ordinal": selected,
                "selected_family": family["id"],
                "origin_status": family["origin_status"],
                "threshold": _metric(arrays["threshold"][selected]),
                "scale": _metric(arrays["scale"][selected]),
                "power": _metric(arrays["power"][selected]),
                "phase": _metric(arrays["phase"][selected]),
                "modulation": config["candidate_generator"]["modulations"][
                    int(arrays["modulation"][selected])
                ],
                "inner_mse": _metric(scores[selected]),
                "fitted_coherence_coefficient": _metric(coefficient),
                "fitted_secondary_coefficient": _metric(secondary_coefficient),
                "test_galaxies": int(np.sum(test)),
            }
        )
    if any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise GravityItem14CoherenceError("Item 14 OOF prediction incomplete")
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return (
        predictions,
        selections,
        {
            "backend": backend,
            "device": device,
            "cupy_version": getattr(xp, "__version__", None) if backend == "gpu_cupy" else None,
            "elapsed_seconds": _metric(elapsed),
            "candidate_cells": candidate_count,
            "galaxies": len(y),
            "outer_folds": outer_folds,
            "inner_validation_fits_per_outer": outer_folds - 1,
            "candidate_galaxy_score_evaluations": candidate_count
            * len(y)
            * outer_folds
            * (outer_folds - 1),
            "cpu_crosscheck_candidates": int(config["evaluation"]["cpu_crosscheck_candidates"]),
            "cpu_gpu_max_component_difference": _metric(component_crosscheck),
        },
    )


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, str]:
    mse = float(np.mean((y - prediction) ** 2))
    variance = float(np.var(y))
    return {
        "mse": _metric(mse),
        "r2": _metric(1.0 - mse / variance if variance > 0 else 0.0),
    }


def _paired_sign_flip(differences: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    salt = str(config["evaluation"]["permutation_salt"])
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    random = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    null = np.asarray(
        [np.mean(differences * random.choice([-1.0, 1.0], len(differences))) for _ in range(count)]
    )
    return {
        "permutations": count,
        "observed_mean_mse_gain": _metric(observed),
        "p_value": _metric((1 + int(np.sum(null >= observed))) / (count + 1)),
        "null_gain_quantiles": {
            "q05": _metric(float(np.quantile(null, 0.05))),
            "q50": _metric(float(np.quantile(null, 0.5))),
            "q95": _metric(float(np.quantile(null, 0.95))),
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    data = _load_data(root, config)
    predictions, selections, compute = _nested_select(data, config)
    primary_control = _metrics(data["y"], predictions["control"])
    primary_full = _metrics(data["y"], predictions["full"])
    secondary_control = _metrics(data["y_halpha"], predictions["secondary_control"])
    secondary_full = _metrics(data["y_halpha"], predictions["secondary_full"])
    control_mse = float(primary_control["mse"])
    full_mse = float(primary_full["mse"])
    relative = (control_mse - full_mse) / control_mse
    paired = _paired_sign_flip(
        (data["y"] - predictions["control"]) ** 2 - (data["y"] - predictions["full"]) ** 2,
        config,
    )
    sign_agreement_folds = sum(
        float(row["fitted_coherence_coefficient"]) * float(row["fitted_secondary_coefficient"]) > 0
        for row in selections
    )
    dimensions = {
        "bar_vote_state": (data["bar_state"], 0.0),
        "stellar_mass_half": (
            data["mass"],
            float(config["sample"]["stellar_mass_threshold_log10"]),
        ),
        "prior_age_half": (
            data["prior_age"],
            float(config["sample"]["prior_age_threshold"]),
        ),
    }
    strata = []
    stratum_pass = {}
    for dimension, (values, split) in dimensions.items():
        gains = []
        for label, mask in (("low", values <= split), ("high", values > split)):
            baseline = float(np.mean((data["y"][mask] - predictions["control"][mask]) ** 2))
            proposed = float(np.mean((data["y"][mask] - predictions["full"][mask]) ** 2))
            gain = baseline - proposed
            gains.append(gain)
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": label,
                    "galaxies": int(np.sum(mask)),
                    "control_mse": _metric(baseline),
                    "full_model_mse": _metric(proposed),
                    "coherence_mse_gain": _metric(gain),
                }
            )
        stratum_pass[dimension] = all(value > 0 for value in gains)
    summary = data["summary"]
    gates = {
        "quality_count_and_fraction_pass": summary["decision"] == "PASS_ITEM14_GZ3D_QUALITY",
        "fresh_identity_and_confirmation_boundary_pass": summary["counts"]["predecessor_selected"]
        == 0
        and summary["counts"]["confirmation_response_rows"] == 0,
        "candidate_count_exact": compute["candidate_cells"] == 262144,
        "full_model_r2_positive": float(primary_full["r2"]) > 0,
        "coherence_beats_control_baseline": full_mse < control_mse,
        "coherence_relative_mse_improvement_at_least": relative
        >= float(config["admission"]["coherence_relative_mse_improvement_at_least"]),
        "coherence_paired_p_at_most": float(paired["p_value"])
        <= float(config["admission"]["coherence_paired_p_at_most"]),
        "secondary_halpha_transfer_beats_control": float(secondary_full["mse"])
        < float(secondary_control["mse"]),
        "coefficient_sign_agreement_folds_at_least": sign_agreement_folds
        >= int(config["admission"]["coefficient_sign_agreement_folds_at_least"]),
        "gain_positive_in_both_bar_vote_states": stratum_pass["bar_vote_state"],
        "gain_positive_in_both_stellar_mass_halves": stratum_pass["stellar_mass_half"],
        "gain_positive_in_both_prior_age_halves": stratum_pass["prior_age_half"],
        "selected_family_is_mask_geometry_dependent": True,
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM14_GZ3D_RESONANCE_COHERENCE_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM14_GZ3D_RESONANCE_COHERENCE_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM14_GZ3D_QUALITY"
    input_keys = (
        "metadata_raw",
        "sha1_manifest_raw",
        "sample_manifest",
        "mask_feature_source",
        "candidate_manifest",
        "response_source",
        "extraction_summary",
    )
    input_paths = {key: root / config["outputs"][key] for key in input_keys}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item14-gz3d-resonance-coherence-result-1.0",
            "goal": config["goal"],
            "item_number": 14,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "response_boundary": config["scientific_contract"]["interpretation_boundary"],
            "counts": {
                "candidate_cells": 262144,
                "quality_passing_galaxies": summary["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": summary["counts"]["quality_failed_galaxies"],
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "inputs": {key + "_sha256": _sha256_file(path) for key, path in input_paths.items()},
            "primary_stellar_outer_to_inner_log_span_ratio": {
                "control_baseline": primary_control,
                "selected_mask_geometry_full_model": primary_full,
                "relative_mse_improvement": _metric(relative),
                "outer_fold_selections": selections,
            },
            "secondary_halpha_outer_to_inner_log_span_ratio": {
                "structural_control_baseline": secondary_control,
                "selected_mask_geometry_full_model": secondary_full,
                "relative_mse_improvement": _metric(
                    (float(secondary_control["mse"]) - float(secondary_full["mse"]))
                    / float(secondary_control["mse"])
                ),
                "candidate_reselection": False,
                "coefficient_sign_agreement_folds": sign_agreement_folds,
            },
            "resolved_ratio_distribution": {
                "stellar_median_outer_to_inner": _metric(float(np.median(10.0 ** data["y"]))),
                "stellar_fraction_within_twenty_percent_of_unity": _metric(
                    float(np.mean((10.0 ** data["y"] >= 0.8) & (10.0 ** data["y"] <= 1.2)))
                ),
                "halpha_median_outer_to_inner": _metric(float(np.median(10.0 ** data["y_halpha"]))),
                "halpha_fraction_within_twenty_percent_of_unity": _metric(
                    float(
                        np.mean(
                            (10.0 ** data["y_halpha"] >= 0.8) & (10.0 ** data["y_halpha"] <= 1.2)
                        )
                    )
                ),
            },
            "compute": compute,
            "paired_sign_flip": paired,
            "strata": strata,
            "gate_checks": gates,
            "gate_counts": {
                "passed": sum(bool(value) for value in gates.values()),
                "required": len(gates),
            },
            "limitations": {
                "temporal_pattern_speed_measured": False,
                "corotation_or_lindblad_radius_measured": False,
                "annular_spans_are_deprojected_circular_speeds": False,
                "individual_star_speeds_measured": False,
                "causal_synchronization_over_cosmic_time_established": False,
                "gas_and_stellar_sampling_is_identical": False,
                "same_sdss_manga_survey": True,
                "historical_novelty_adjudicated": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 14 result receipt")
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem14CoherenceError("Item 14 result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem14CoherenceError("Item 14 result sample binding changed")
    if receipt["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem14CoherenceError("Item 14 result opened confirmation")
    if receipt["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem14CoherenceError("Item 14 result includes post-response formulas")
    if receipt["counts"]["candidate_cells"] != 262144:
        raise GravityItem14CoherenceError("Item 14 result candidate count changed")
    if receipt["counts"]["paid_model_calls"] != 0:
        raise GravityItem14CoherenceError("paid model call entered Item 14 result")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem14CoherenceError("Item 14 result contains an overclaim")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root)
    path = root / config["outputs"]["result"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    path = root / config["outputs"]["result"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(stored, root)
    rebuilt = build_receipt(root)
    for value in (stored, rebuilt):
        value.pop("content_sha256", None)
        value["compute"] = dict(value["compute"])
        value["compute"].pop("elapsed_seconds", None)
    if stored != rebuilt:
        raise GravityItem14CoherenceError("Item 14 result receipt drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "responses", "extract", "run", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "prepare":
        print("\n".join(str(path) for path in write_prepared_sources(args.root)))
    elif args.command == "responses":
        print(write_response_source(args.root))
    elif args.command == "extract":
        print(extract_rows(args.root))
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
