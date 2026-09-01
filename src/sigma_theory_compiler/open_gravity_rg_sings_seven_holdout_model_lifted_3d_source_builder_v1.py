"""Build seven response-blind SINGS/THINGS/HERACLES model-lifted sources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy import log as astropy_log
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy import ndimage

from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as base,
)

CONFIG_PATH = Path(
    "configs/open_gravity_rg_sings_seven_holdout_model_lifted_3d_source_builder_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_sings_seven_holdout_model_lifted_3d_source_builder_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_rg_sings_seven_holdout_model_lifted_3d_source_builder_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-sings-seven-holdout-model-lifted-3d-source-builder-v1/receipt.json"
)

_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-model-lifted-3d-source-builder-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-sings-seven-holdout-model-lifted-3d-source-builder-receipt-1.0"
)
_OBJECTS = ("UGC04305", "NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331")
_CELLS = ("IRAC1_FIXED_ML0P6", "IRAC1_GLOBAL_COLOR_ML", "IRAC1_IRAC2_FASTICA36")
_IRAC1_ZERO_JY = 280.9
_IRAC2_ZERO_JY = 179.7
_CONFIG_RAW_SHA256 = "fb2f16d3708ee52abd53315ae0562e443fa3f5520a86afc9f2e6fffb98f6ee4b"
_CONFIG_CONTENT_SHA256 = "2e40ad4647b1242c63b2abe5bb0b0c20862e97ad74f54dad2cb5ced58805046c"
_MODULE_SEMANTIC_SHA256 = "e6872d902c0230838f2409ffb18e23e7eba57229cd9cdcd3f00504b95192c321"
_TEST_RAW_SHA256 = "5608dfaf0d31530fd9ee85af0cdd27912158a4b9966adc497992beaad4cdd912"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")

astropy_log.setLevel("ERROR")


class SourceBuildError(RuntimeError):
    """Raised when a source, paper, conversion, or numerical gate fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceBuildError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    path = (root / Path(relative)).resolve()
    _require(path == root or root in path.parents, "path escaped repository")
    return path


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        b'_MODULE_SEMANTIC_SHA256 = "' + b"0" * 64 + b'"', path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _verify_binding(binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = _repo_path(str(binding["path"]))
    _require(file_sha256(path) == binding["raw_sha256"], f"{label} raw hash changed")
    value = _read_json(path, label)
    if "content_sha256" in binding:
        _require(
            value.get("content_sha256") == binding["content_sha256"], f"{label} content changed"
        )
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(tuple(config["object_order"]) == _OBJECTS, "object order changed")
    _require(tuple(row["cell_id"] for row in config["conversion_cells"]) == _CELLS, "cells changed")
    admission = config["admission_rule"]
    _require(admission["real_public_source_data_required"] is True, "source gate removed")
    _require(admission["primary_measurement_papers_required"] is True, "paper gate removed")
    _require(
        admission["independent_known_answer_benchmark_required"] is True, "benchmark gate removed"
    )
    _require(admission["standard_gravity_controls_required_downstream"] is True, "controls removed")
    _require(
        admission["missing_source_disposition"] == "SOURCE_BLOCKED", "source fail-close changed"
    )
    _require(
        admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY", "paper disposition changed"
    )
    _require(
        admission["projected_source_plus_assumed_vertical_disposition"] == "MODEL_LIFTED_2P5D",
        "dimension claim changed",
    )
    _require(
        admission["spherical_or_one_dimensional_data_proves_general_3d"] is False, "3D overclaim"
    )
    fastica = config["fastica_source_contract"]
    _require(fastica["minimum_high_signal_training_pixels"] == 32, "FastICA minimum changed")
    _require(fastica["maximum_training_pixels"] == 250000, "FastICA ceiling changed")
    _require(
        fastica["threshold_selected_from_source_only_data"] is True,
        "source-only threshold rule changed",
    )
    _require(
        fastica["low_training_support_must_be_reported"] is True,
        "low-support disclosure removed",
    )
    _require(fastica["response_values_used"] is False, "response entered FastICA threshold")
    objects = config["objects"]
    _require(
        len(objects) == 7 and {row["object_id"] for row in objects} == set(_OBJECTS),
        "objects changed",
    )
    holmberg = next(row for row in objects if row["object_id"] == "UGC04305")
    _require(
        holmberg["inclination_cells_deg"] == [27.0, 38.0, 49.0], "Holmberg II uncertainty removed"
    )
    _require(holmberg["primary_inclination_deg"] == 38.0, "Holmberg II photometric primary changed")
    for row in objects:
        _require(0.0 < float(row["distance_mpc"]) < 20.0, "distance invalid")
        _require(0.0 < float(row["primary_inclination_deg"]) < 90.0, "inclination invalid")
        _require(len(row["inclination_cells_deg"]) >= 1, "geometry cells missing")
    response = config["response_boundary"]
    _require(all(value == 0 for value in response.values()), "response boundary changed")
    claims = config["claim_boundary"]
    for key in (
        "source_maps_built",
        "full_3d_source_observed",
        "general_3d_gravity_validated",
        "scientific_response_scored",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim promoted: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _validate_package() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    value = _read_json(path, "config")
    validate_config(value)
    if verify_package:
        _validate_package()
    return value


def _load_contracts(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings = config["bindings"]
    _verify_binding(bindings["source_acquisition_config"], "source acquisition config")
    acquisition = _verify_binding(
        bindings["source_acquisition_receipt"], "source acquisition receipt"
    )
    _verify_binding(bindings["stellar_conversion_config"], "stellar conversion config")
    stellar = _verify_binding(bindings["stellar_conversion_receipt"], "stellar conversion receipt")
    irac2 = _verify_binding(bindings["irac2_inventory"], "IRAC2 inventory")
    _verify_binding(bindings["operator_benchmark_config"], "operator benchmark config")
    benchmark = _verify_binding(
        bindings["operator_benchmark_receipt"], "operator benchmark receipt"
    )
    paper = bindings["primary_geometry_pdf"]
    paper_path = _repo_path(paper["local_evidence_path"])
    _require(file_sha256(paper_path) == paper["raw_sha256"], "geometry paper bytes changed")
    _require(
        acquisition["decision"]
        == "PASS_EXACT_SOURCE_BYTES_AND_FITS_SCHEMAS_READY_FOR_RESPONSE_BLIND_BUILDERS",
        "source acquisition not ready",
    )
    _require(
        stellar["decision"] == "PASS_RETAIN_ALL_THREE_STELLAR_CONVERSION_CELLS_BEFORE_NEW_RESPONSE",
        "stellar contract not ready",
    )
    _require(
        irac2["content_sha256"] == bindings["irac2_inventory"]["content_sha256"],
        "IRAC2 content changed",
    )
    _require(benchmark["status"].startswith("PASS"), "operator benchmark failed")
    return acquisition, irac2


def _source_paths(
    acquisition: Mapping[str, Any], irac2: Mapping[str, Any]
) -> dict[tuple[str, str], Path]:
    records = [row for row in acquisition["inventory"]["records"] if row["object_id"] in _OBJECTS]
    records.extend(row for row in irac2["records"] if row["object_id"] in _OBJECTS)
    _require(len(records) == 56, "seven-object source record count changed")
    paths: dict[tuple[str, str], Path] = {}
    for row in records:
        key = (str(row["object_id"]), str(row["role"]))
        _require(key not in paths, "duplicate source role")
        path = _repo_path(str(row["relative_path"]))
        _require(path.is_file(), "source file missing")
        _require(path.stat().st_size == int(row["bytes"]), "source bytes changed")
        _require(file_sha256(path) == row["sha256"], "source hash changed")
        paths[key] = path
    expected_roles = {
        "STELLAR_IRAC1_FLUX",
        "STELLAR_IRAC1_WEIGHT",
        "STELLAR_IRAC2_FLUX",
        "STELLAR_IRAC2_WEIGHT",
        "HI_MOM0_NATURAL",
        "HI_MOM0_ROBUST",
        "CO21_MOM0",
        "CO21_EMOM0",
    }
    for object_id in _OBJECTS:
        _require(
            {role for name, role in paths if name == object_id} == expected_roles,
            f"source roles changed for {object_id}",
        )
    return paths


def _wcs(header: fits.Header) -> WCS:
    value = WCS(header, relax=True)
    value.sip = None
    return value


def _robust_sigma(data: np.ndarray, finite: np.ndarray) -> float:
    values = data[finite]
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    _require(math.isfinite(sigma) and sigma > 0.0, "invalid IRAC noise")
    return sigma


def _largest_center_source(
    data: np.ndarray, finite: np.ndarray, sigma: float, *, center_yx: tuple[int, int]
) -> np.ndarray:
    smooth = ndimage.gaussian_filter(np.where(finite, data, 0.0), sigma=2.0)
    detected = ndimage.binary_closing(finite & (smooth > 3.0 * sigma), iterations=3)
    labels, count = ndimage.label(detected)
    _require(count > 0, "no central IRAC source")
    cy, cx = center_yx
    central = int(labels[cy, cx])
    if central == 0:
        centers = ndimage.center_of_mass(detected, labels, range(1, count + 1))
        central = min(
            range(1, count + 1),
            key=lambda index: (centers[index - 1][0] - cy) ** 2 + (centers[index - 1][1] - cx) ** 2,
        )
    return ndimage.binary_dilation(labels == central, iterations=8)


def _color_from_ratio(f36_over_f45: float) -> float:
    return -2.5 * math.log10(f36_over_f45 * _IRAC2_ZERO_JY / _IRAC1_ZERO_JY)


def _ratio45_over36(color: float) -> float:
    return (_IRAC2_ZERO_JY / _IRAC1_ZERO_JY) * 10.0 ** (0.4 * color)


def _symmetric_decorrelation(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix @ matrix.T)
    _require(float(np.min(values)) > 0.0, "singular FastICA decorrelation")
    return (vectors @ np.diag(values**-0.5) @ vectors.T) @ matrix


def _fastica_two_source(
    samples: np.ndarray,
    *,
    stellar_seed_color: float,
    dust_seed_color: float,
    tolerance: float = 1.0e-10,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, int, float]:
    _require(samples.shape[0] == 2, "FastICA requires two bands")
    centered = samples - np.mean(samples, axis=1, keepdims=True)
    covariance = centered @ centered.T / centered.shape[1]
    values, vectors = np.linalg.eigh(covariance)
    _require(float(np.min(values)) > 1.0e-18, "degenerate FastICA covariance")
    whitening = np.diag(values**-0.5) @ vectors.T
    white = whitening @ centered
    mixing_seed = np.array(
        [[1.0, 1.0], [_ratio45_over36(stellar_seed_color), _ratio45_over36(dust_seed_color)]],
        dtype=np.float64,
    )
    unmixing = _symmetric_decorrelation(np.linalg.inv(whitening @ mixing_seed))
    convergence = math.inf
    for iteration in range(1, max_iterations + 1):
        projected = unmixing @ white
        exponential = np.exp(-0.5 * projected**2)
        update = (projected * exponential) @ white.T / white.shape[1]
        update -= np.mean((1.0 - projected**2) * exponential, axis=1)[:, None] * unmixing
        update = _symmetric_decorrelation(update)
        convergence = float(np.max(np.abs(np.abs(np.diag(update @ unmixing.T)) - 1.0)))
        unmixing = update
        if convergence <= tolerance:
            break
    else:
        raise SourceBuildError("FastICA failed to converge")
    return np.linalg.inv(unmixing @ whitening), iteration, convergence


def _component_colors(mixing: np.ndarray) -> list[float]:
    result: list[float] = []
    for index in range(2):
        column = mixing[:, index]
        if column[0] == 0.0 or column[1] / column[0] <= 0.0:
            result.append(math.nan)
        else:
            result.append(_color_from_ratio(float(column[0] / column[1])))
    return result


def _estimate_colors(f36: np.ndarray, f45: np.ndarray, training: np.ndarray) -> dict[str, Any]:
    samples = np.vstack([f36[training], f45[training]])
    if samples.shape[1] > 250_000:
        indices = np.linspace(0, samples.shape[1] - 1, 250_000, dtype=np.int64)
        samples = samples[:, indices]
    solutions: list[dict[str, float | int]] = []
    stellar_seeds = np.arange(-0.20, 0.0001, 0.04)
    dust_seeds = np.arange(0.0, 1.5001, 0.30)
    for stellar_seed in stellar_seeds:
        for dust_seed in dust_seeds:
            if dust_seed <= stellar_seed:
                continue
            try:
                mixing, iterations, residual = _fastica_two_source(
                    samples,
                    stellar_seed_color=float(stellar_seed),
                    dust_seed_color=float(dust_seed),
                )
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            colors = _component_colors(mixing)
            if not all(math.isfinite(value) for value in colors):
                continue
            stellar_index = int(np.argmin(colors))
            dust_index = 1 - stellar_index
            if colors[dust_index] <= colors[stellar_index]:
                continue
            solutions.append(
                {
                    "stellar_color": float(colors[stellar_index]),
                    "dust_color": float(colors[dust_index]),
                    "iterations": iterations,
                    "convergence": residual,
                }
            )
    _require(bool(solutions), "no physical FastICA solution")
    stellar = np.asarray([row["stellar_color"] for row in solutions], dtype=np.float64)
    dust = np.asarray([row["dust_color"] for row in solutions], dtype=np.float64)
    return {
        "documented_seed_grid_count": int(stellar_seeds.size * dust_seeds.size),
        "converged_solution_count": len(solutions),
        "stellar_color_mean": float(np.mean(stellar)),
        "stellar_color_std": float(np.std(stellar)),
        "dust_color_mean": float(np.mean(dust)),
        "dust_color_std": float(np.std(dust)),
        "max_iterations": max(int(row["iterations"]) for row in solutions),
        "max_convergence_residual": max(float(row["convergence"]) for row in solutions),
    }


def _decompose(
    f36: np.ndarray, f45: np.ndarray, star_color: float, dust_color: float
) -> tuple[np.ndarray, np.ndarray]:
    mixing = np.array(
        [[1.0, 1.0], [_ratio45_over36(star_color), _ratio45_over36(dust_color)]],
        dtype=np.float64,
    )
    sources = np.linalg.solve(mixing, np.vstack([f36.ravel(), f45.ravel()]))
    return sources[0].reshape(f36.shape), sources[1].reshape(f36.shape)


def _native_stellar_cells(
    config: Mapping[str, Any], paths: Mapping[tuple[str, str], Path], object_id: str
) -> tuple[dict[str, tuple[np.ndarray, fits.Header, np.ndarray]], dict[str, Any]]:
    f36, h36 = base._fits_image(paths[(object_id, "STELLAR_IRAC1_FLUX")])
    w36, _ = base._fits_image(paths[(object_id, "STELLAR_IRAC1_WEIGHT")])
    f45_raw, h45 = base._fits_image(paths[(object_id, "STELLAR_IRAC2_FLUX")])
    w45_raw, hw45 = base._fits_image(paths[(object_id, "STELLAR_IRAC2_WEIGHT")])
    f45, footprint = reproject_interp(
        (f45_raw, _wcs(h45)), _wcs(h36), shape_out=f36.shape, order="bilinear"
    )
    w45, weight_footprint = reproject_interp(
        (w45_raw, _wcs(hw45)), _wcs(h36), shape_out=f36.shape, order="nearest-neighbor"
    )
    finite = (
        np.isfinite(f36)
        & np.isfinite(f45)
        & np.isfinite(w36)
        & np.isfinite(w45)
        & (w36 > 0.0)
        & (w45 > 0.0)
        & (footprint > 0.999)
        & (weight_footprint > 0.999)
    )
    sigma36 = _robust_sigma(f36, finite)
    sigma45 = _robust_sigma(f45, finite)
    galaxy = _largest_center_source(
        f36,
        finite,
        sigma36,
        center_yx=(round(float(h36["CRPIX2"]) - 1.0), round(float(h36["CRPIX1"]) - 1.0)),
    )
    positive = galaxy & finite & (f36 > 0.0) & (f45 > 0.0)
    _require(int(np.count_nonzero(positive)) >= 1000, f"too few stellar pixels for {object_id}")
    sum36 = float(np.sum(f36[positive]))
    sum45 = float(np.sum(f45[positive]))
    global_color = _color_from_ratio(sum36 / sum45)
    global_ml = 10.0 ** (-0.339 * global_color - 0.336)
    _require(0.05 <= global_ml <= 2.0, f"global M/L invalid for {object_id}")
    color = np.full(f36.shape, np.nan, dtype=np.float64)
    color[positive] = -2.5 * np.log10(
        (f36[positive] / f45[positive]) * (_IRAC2_ZERO_JY / _IRAC1_ZERO_JY)
    )
    training = (
        positive & (f36 > 10.0 * sigma36) & (f45 > 10.0 * sigma45) & (color > -0.3) & (color < 1.5)
    )
    minimum_training = int(config["fastica_source_contract"]["minimum_high_signal_training_pixels"])
    _require(
        int(np.count_nonzero(training)) >= minimum_training,
        f"too few FastICA pixels for {object_id}: {int(np.count_nonzero(training))}",
    )
    first = _estimate_colors(f36, f45, training)
    _first_star, first_dust = _decompose(
        f36, f45, float(first["stellar_color_mean"]), float(first["dust_color_mean"])
    )
    red = ndimage.binary_dilation(
        positive & (color > float(first["dust_color_mean"])), iterations=1
    )
    dust_training = first_dust[training]
    high_dust = (
        training
        & (first_dust > float(np.mean(dust_training) + 5.0 * np.std(dust_training)))
        & (color > min(float(first["dust_color_mean"]), 0.1))
    )
    second_training = training & ~red & ~high_dust
    second = (
        _estimate_colors(f36, f45, second_training)
        if int(np.count_nonzero(second_training)) >= minimum_training
        else None
    )
    use_second = bool(
        second is not None
        and -0.2 <= float(second["stellar_color_mean"]) <= 0.0
        and 0.0 <= float(second["dust_color_mean"]) <= 1.5
        and float(second["dust_color_mean"]) < float(first["dust_color_mean"])
    )
    solution = second if use_second else first
    assert solution is not None
    star, dust = _decompose(
        f36, f45, float(solution["stellar_color_mean"]), float(solution["dust_color_mean"])
    )
    reconstruction = float(np.nanmax(np.abs(star + dust - f36)))
    _require(
        reconstruction <= 1.0e-9 * max(float(np.nanmax(np.abs(f36))), 1.0),
        "FastICA reconstruction failed",
    )
    mask = ~galaxy | ~finite
    fixed = np.where(mask, 0.0, np.maximum(f36, 0.0))
    global_scaled = fixed * (global_ml / 0.6)
    fastica = np.where(mask, 0.0, np.maximum(star, 0.0))
    diagnostics = {
        "object_id": object_id,
        "sigma_mjy_sr": {"irac1": sigma36, "irac2": sigma45},
        "galaxy_pixels": int(np.count_nonzero(galaxy)),
        "positive_two_band_pixels": int(np.count_nonzero(positive)),
        "training_pixels": int(np.count_nonzero(training)),
        "minimum_training_pixels": minimum_training,
        "low_training_support": int(np.count_nonzero(training)) < 1000,
        "second_training_pixels": int(np.count_nonzero(second_training)),
        "global_color_mag": global_color,
        "global_ml": global_ml,
        "fastica_solution": solution,
        "fastica_selected_iteration": 2 if use_second else 1,
        "fastica_physical_color_gate_pass": bool(
            -0.2 <= float(solution["stellar_color_mean"]) <= 0.0
            and 0.0 <= float(solution["dust_color_mean"]) <= 1.5
        ),
        "fastica_reconstruction_max_abs_mjy_sr": reconstruction,
        "fastica_stellar_nonnegative_fraction": float(
            np.count_nonzero(star[positive] >= 0.0) / np.count_nonzero(positive)
        ),
    }
    cells = {
        "IRAC1_FIXED_ML0P6": (fixed, h36, mask),
        "IRAC1_GLOBAL_COLOR_ML": (global_scaled, h36, mask),
        "IRAC1_IRAC2_FASTICA36": (fastica, h36, mask),
    }
    return cells, diagnostics


def _images_for_cell(
    paths: Mapping[tuple[str, str], Path],
    object_id: str,
    cell: tuple[np.ndarray, fits.Header, np.ndarray],
) -> dict[str, tuple[np.ndarray, fits.Header]]:
    stellar, header, invalid = cell
    aliases = {
        "HI_MOM0_NATURAL_SENSITIVITY": "HI_MOM0_NATURAL",
        "HI_MOM0_ROBUST_PRIMARY": "HI_MOM0_ROBUST",
        "CO21_BROAD_MOM0": "CO21_MOM0",
        "CO21_BROAD_EMOM0": "CO21_EMOM0",
    }
    images = {
        target: base._fits_image(paths[(object_id, source)]) for target, source in aliases.items()
    }
    images["STELLAR_FLUX"] = (stellar, header)
    images["STELLAR_ICA_MASK"] = (invalid.astype(np.float64), header)
    images["STELLAR_COLOR"] = (np.zeros_like(stellar), header)
    return images


def _metadata(row: Mapping[str, Any], inclination: float) -> dict[str, Any]:
    return {
        "object_id": row["object_id"],
        "ra_deg": float(row["ra_deg"]),
        "dec_deg": float(row["dec_deg"]),
        "distance_mpc": float(row["distance_mpc"]),
        "position_angle_deg": float(row["position_angle_deg"]),
        "inclination_deg": float(inclination),
        "geometry_variant_id": f"I{str(float(inclination)).replace('.', 'P')}",
    }


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def _source_file_payloads(
    config: Mapping[str, Any], acquisition: Mapping[str, Any], irac2: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    paths = _source_paths(acquisition, irac2)
    public_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    stellar_diagnostics: list[dict[str, Any]] = []
    object_map = {row["object_id"]: row for row in config["objects"]}
    for object_id in _OBJECTS:
        native_cells, diagnostics = _native_stellar_cells(config, paths, object_id)
        stellar_diagnostics.append(diagnostics)
        object_row = object_map[object_id]
        for inclination in object_row["inclination_cells_deg"]:
            metadata = _metadata(object_row, float(inclination))
            for cell_id in _CELLS:
                if (
                    cell_id == "IRAC1_IRAC2_FASTICA36"
                    and not diagnostics["fastica_physical_color_gate_pass"]
                ):
                    public_rows.append(
                        {
                            "object_id": object_id,
                            "conversion_cell_id": cell_id,
                            "geometry": metadata,
                            "model_lift_label": "MODEL_LIFTED_2P5D",
                            "disposition": "SOURCE_CONVERSION_FAILED_UNPHYSICAL_FASTICA_COLOR_RETAINED",
                            "failure_evidence": {
                                "training_pixels": diagnostics["training_pixels"],
                                "low_training_support": diagnostics["low_training_support"],
                                "stellar_color_mean": diagnostics["fastica_solution"][
                                    "stellar_color_mean"
                                ],
                                "dust_color_mean": diagnostics["fastica_solution"][
                                    "dust_color_mean"
                                ],
                            },
                            "array_files": [],
                        }
                    )
                    continue
                images = _images_for_cell(paths, object_id, native_cells[cell_id])
                maps = base._surface_maps(
                    dict(config),
                    metadata,
                    images,
                    n=int(config["map_transform"]["primary_grid_pixels"]),
                    box_kpc=float(config["map_transform"]["primary_box_kpc"]),
                    beam="ROBUST_PRIMARY",
                    use_sip=False,
                )
                half_mass = base._half_mass_radius_pc(
                    maps["stellar_fixed"], maps["x_pc"], maps["y_pc"], float(maps["dx_pc"])
                )
                _require(50.0 < half_mass < 25_000.0, f"stellar scale invalid for {object_id}")
                summary, profile = base._build_cell(
                    dict(config),
                    metadata,
                    maps,
                    cell_id=f"{cell_id}:{metadata['geometry_variant_id']}",
                    stellar_ml="FIXED_0P6",
                    co_source="WITH_CO",
                    hstar_pc=half_mass
                    / 1.678
                    * float(
                        config["vertical_and_gravity_model"][
                            "primary_stellar_height_over_exponential_scale"
                        ]
                    ),
                    hgas_pc=float(config["vertical_and_gravity_model"]["primary_gas_height_pc"]),
                    cache={},
                )
                prefix = f"{object_id}__{cell_id}__{metadata['geometry_variant_id']}"
                arrays = {
                    "stellar_surface_msun_pc2": np.asarray(maps["stellar_fixed"], dtype=np.float32),
                    "hi_surface_msun_pc2": np.asarray(maps["hi"], dtype=np.float32),
                    "co_surface_msun_pc2": np.asarray(maps["co"], dtype=np.float32),
                    "x_pc": np.asarray(maps["x_pc"], dtype=np.float32),
                    "y_pc": np.asarray(maps["y_pc"], dtype=np.float32),
                }
                files: list[dict[str, Any]] = []
                for role, array in arrays.items():
                    relative = f"{prefix}__{role}.npy"
                    body = _npy_bytes(array)
                    payloads[relative] = body
                    files.append(
                        {
                            "role": role,
                            "relative_path": relative,
                            "bytes": len(body),
                            "sha256": hashlib.sha256(body).hexdigest(),
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                        }
                    )
                public_rows.append(
                    {
                        "object_id": object_id,
                        "conversion_cell_id": cell_id,
                        "geometry": metadata,
                        "model_lift_label": "MODEL_LIFTED_2P5D",
                        "disposition": "SOURCE_MAP_BUILT_RESPONSE_BLIND",
                        "dx_pc": float(maps["dx_pc"]),
                        "stellar_half_mass_radius_pc": half_mass,
                        "summary": summary,
                        "profile_sha256": content_sha256(profile),
                        "array_files": files,
                    }
                )
    return public_rows, payloads, stellar_diagnostics


def _without_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "content_sha256"}


def build_packet() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = load_config()
    acquisition, irac2 = _load_contracts(config)
    benchmark = base._benchmark_report(dict(config))
    _require(all(benchmark["passed"].values()), "independent benchmark failed")
    rows, payloads, diagnostics = _source_file_payloads(config, acquisition, irac2)
    expected_cells = sum(len(row["inclination_cells_deg"]) for row in config["objects"]) * len(
        _CELLS
    )
    _require(len(rows) == expected_cells == 27, "source cell count changed")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RESPONSE_BLIND_SOURCE_BUILD_WITH_RETAINED_FASTICA_FAILURES",
        "decision": "PASS_24_THEORY_NEUTRAL_SOURCE_MAPS_READY_THREE_FASTICA_FAILURES_RETAINED",
        "admission_rule": config["admission_rule"],
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "bindings": config["bindings"],
        "object_count": len(_OBJECTS),
        "conversion_cell_count": len(_CELLS),
        "geometry_cell_count": sum(len(row["inclination_cells_deg"]) for row in config["objects"]),
        "source_cell_count": len(rows),
        "built_source_map_count": sum(
            row["disposition"] == "SOURCE_MAP_BUILT_RESPONSE_BLIND" for row in rows
        ),
        "failed_source_conversion_count": sum(
            row["disposition"] == "SOURCE_CONVERSION_FAILED_UNPHYSICAL_FASTICA_COLOR_RETAINED"
            for row in rows
        ),
        "private_array_file_count": len(payloads),
        "private_array_bytes": sum(len(value) for value in payloads.values()),
        "private_array_root_sha256": content_sha256(
            [
                {
                    "relative_path": path,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                }
                for path, body in sorted(payloads.items())
            ]
        ),
        "benchmarks": benchmark,
        "stellar_conversion_diagnostics": diagnostics,
        "source_cells": rows,
        "response_boundary": config["response_boundary"],
        "claim_boundary": {
            **config["claim_boundary"],
            "source_maps_built": True,
        },
        "next_action": "Apply each frozen 3D gravity operator to these same theory-neutral source arrays before opening THINGS velocity responses.",
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt, payloads


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    rebuilt, _payloads = build_packet()
    _require(dict(receipt) == rebuilt, "receipt differs from deterministic rebuild")
    _require(
        receipt["content_sha256"] == content_sha256(_without_hash(receipt)),
        "receipt self-hash changed",
    )


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            return "CREATED"
        except FileExistsError:
            _require(path.read_bytes() == payload, f"existing artifact differs: {path.name}")
            return "EXISTING_IDENTICAL"
    finally:
        if temporary.exists():
            temporary.unlink()


def write_packet() -> str:
    config = load_config()
    receipt, payloads = build_packet()
    private = _repo_path(config["private_output_directory"])
    statuses = [
        _atomic_no_clobber(private / relative, body) for relative, body in sorted(payloads.items())
    ]
    status = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return (
        status
        if all(value == status for value in statuses)
        else "MIXED_CREATED_AND_EXISTING_IDENTICAL"
    )


def check_packet() -> str:
    receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    validate_receipt(receipt)
    config = load_config()
    private = _repo_path(config["private_output_directory"])
    for row in receipt["source_cells"]:
        for artifact in row["array_files"]:
            path = private / artifact["relative_path"]
            _require(path.is_file(), "private source array missing")
            _require(path.stat().st_size == artifact["bytes"], "private array bytes changed")
            _require(file_sha256(path) == artifact["sha256"], "private array hash changed")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_packet())
    elif args.command == "check":
        print(check_packet())
    else:
        receipt, _ = build_packet()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "objects": receipt["object_count"],
                    "source_cells": receipt["source_cell_count"],
                    "velocity_values_opened": receipt["response_boundary"][
                        "velocity_values_opened"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
