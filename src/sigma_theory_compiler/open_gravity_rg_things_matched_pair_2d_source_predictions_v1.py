from __future__ import annotations

import argparse
import gc
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
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as baseline
from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge
from sigma_theory_compiler import open_gravity_refracted_gravity_3d_primary_benchmark_v1 as rg
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as mechanics,
)
from sigma_theory_compiler import open_gravity_rg_things_2d_projection_benchmark_v1 as projection
from sigma_theory_compiler import (
    open_gravity_rg_things_heracles_s4g_model_lifted_3d_source_builder_v1 as sources,
)
from sigma_theory_compiler import (
    open_gravity_rg_things_heracles_s4g_scoring_resolution_v1 as source_resolution,
)

CONFIG_PATH = Path("configs/open_gravity_rg_things_matched_pair_2d_source_predictions_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_matched_pair_2d_source_predictions_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_matched_pair_2d_source_predictions_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-matched-pair-2d-source-predictions-v1/receipt.json"
)
PRIVATE_DIRECTORY = Path(
    "work/private/open-gravity-rg-things-matched-pair-2d-source-predictions-v1"
)
PRIVATE_MANIFEST_PATH = PRIVATE_DIRECTORY / "manifest.json"

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-source-predictions-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-source-predictions-receipt-1.0"
_MANIFEST_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-source-predictions-private-1.0"
_OBJECTS = ("NGC2976", "NGC4214")
_MODELS = ("NEWTON_3D_DST", "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG")
_CONFIG_RAW_SHA256 = "ce675f417980870dcf74cff18d976d1247f160a3f0fa3f1d1773dd146650eee5"
_CONFIG_CONTENT_SHA256 = "60c41a541a0949ac4af0244880464764a2bc64cca7c0dde2113e2ad95f4abc22"
_MODULE_SEMANTIC_SHA256 = "6f2e16c34c89cbb7e430cc14a2c7dff955219b942ceadada989935397a42196d"
_TEST_RAW_SHA256 = "09c1177f5cdfa70afcc02404c891ee074193655ee725814de8c1ffaf6eaa8e2b"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class MatchedPairPredictionError(RuntimeError):
    """Raised when a source, paper, solver, or prediction gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedPairPredictionError(message)


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


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        b'_MODULE_SEMANTIC_SHA256 = "' + b"0" * 64 + b'"', path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedPairPredictionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_RESPONSE_BLIND_MATCHED_PAIR_MODEL_LIFTED_2P5D_PREDICTIONS",
        "status changed",
    )
    admission = config["admission_rule"]
    for key in (
        "real_public_source_data_required",
        "primary_measurement_and_data_release_papers_required",
        "independent_operator_benchmarks_required",
        "known_newtonian_control_required",
        "spherical_or_1d_data_cannot_validate_general_3d",
        "model_lifted_vertical_structure_is_not_observed_3d",
    ):
        _require(admission[key] is True, f"admission gate changed: {key}")
    _require(admission["missing_data_disposition"] == "SOURCE_BLOCKED", "source rule changed")
    _require(
        admission["failed_operator_disposition"] == "BUILDER_BLOCKED_RETAIN_FAILURE",
        "operator failure rule changed",
    )
    _require([row["object_id"] for row in config["objects"]] == list(_OBJECTS), "objects changed")
    _require(
        config["objects"][1]["kinematic_disposition"]
        == "RETAINED_LOW_INCLINATION_COUNTEREXAMPLE_NOT_STANDARD_ROTATION_CURVE",
        "NGC4214 caveat changed",
    )
    _require(config["source_cell"]["selection_from_response"] is False, "source leakage")
    grid = config["field_grid"]
    _require(grid["half_box_kpc"] == 30.0, "solver box changed")
    _require(grid["source_map_box_kpc"] == 40.0, "source map box changed")
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["fine_spacing_kpc"] == 0.25, "fine spacing changed")
    _require(grid["convergence_spacing_kpc"] == 0.3125, "coarse spacing changed")
    _require(
        grid["maximum_local_radial_acceleration_relative_difference"] == 0.1,
        "convergence gate changed",
    )
    operator = config["operator"]
    _require(operator["models"] == list(_MODELS), "models changed")
    _require(operator["epsilon_0"] == 0.661, "epsilon0 changed")
    _require(operator["Q"] == 1.79, "Q changed")
    _require(operator["log10_rho_c_g_cm3"] == -24.54, "rho_c changed")
    _require(operator["response_parameter_fitting"] is False, "response fit enabled")
    _require(operator["boundary_parameter_fitting"] is False, "boundary fit enabled")
    projection_contract = config["projection"]
    _require(
        projection_contract["response_values_used_to_build_predictions"] is False,
        "response leakage enabled",
    )
    private = config["private_output"]
    _require(private["directory"] == PRIVATE_DIRECTORY.as_posix(), "private directory changed")
    _require(private["manifest"] == PRIVATE_MANIFEST_PATH.as_posix(), "manifest path changed")
    _require(
        private["arrays_per_object"]
        == [
            "source_intensity.npy",
            "source_eligibility.npy",
            "radius_kpc.npy",
            "newton_vlos_plus_m_s.npy",
            "rg_vlos_plus_m_s.npy",
            "newton_tangential_ratio.npy",
            "rg_tangential_ratio.npy",
            "newton_convergence_relative.npy",
            "rg_convergence_relative.npy",
        ],
        "private array contract changed",
    )
    boundary = config["scientific_boundary"]
    _require(boundary["real_source_sets_opened"] == 2, "source-set accounting changed")
    _require(boundary["velocity_headers_opened"] == 2, "header accounting changed")
    _require(
        boundary["response_files_hashed_as_opaque_bytes"] == 2,
        "opaque response-file accounting changed",
    )
    _require(boundary["response_bytes_hashed"] == 8_487_360, "opaque bytes changed")
    for key in (
        "velocity_pixel_values_decoded",
        "dispersion_pixel_values_decoded",
        "scientific_scores_computed",
        "response_selected_parameters",
        "network_calls",
        "model_calls",
        "paid_calls",
    ):
        _require(boundary[key] == 0, f"forbidden access enabled: {key}")
    _require(boundary["observed_full_3d_geometry"] is False, "3D overclaim")
    _require(boundary["model_lifted_2p5d_only"] is True, "2.5D caveat removed")
    claims = config["claim_boundary"]
    for key in (
        "scientific_fit_tested",
        "ngc4214_standard_rotation_curve_valid",
        "gas_dynamics_solved",
        "general_3d_validated",
        "refracted_gravity_supported",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overreach: {key}")
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
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["bindings"]:
        receipt_path: Path | None = None
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "bound artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "bound artifact changed")
            if artifact["path"].endswith("/receipt.json"):
                receipt_path = path
        _require(receipt_path is not None, "bound receipt missing")
        receipt = _read_json(receipt_path, "bound receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "bound receipt content changed",
        )
        receipts[binding["role"]] = receipt
    _require(
        receipts["FIVE_OBJECT_REAL_SOURCE_BUILDER"]["cell_count"] == 393,
        "source coverage changed",
    )
    _require(
        receipts["FIVE_OBJECT_RG_SCORING_NUMERICS"]["all_object_gates_pass"] is True,
        "field numerics failed",
    )
    _require(
        receipts["TARGET_FREE_2D_PROJECTION_BENCHMARK"]["benchmarks"]["all_pass"] is True,
        "2D projection benchmark failed",
    )
    _require(
        receipts["EXACT_THINGS_MATCHED_PAIR_RESPONSE_SOURCE"]["access_accounting"][
            "velocity_pixel_values_decoded"
        ]
        == 0,
        "response source already decoded",
    )
    return receipts


def _response_source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = next(
        row
        for row in config["bindings"]
        if row["role"] == "EXACT_THINGS_MATCHED_PAIR_RESPONSE_SOURCE"
    )
    config_artifact = next(
        row for row in binding["artifacts"] if row["path"].startswith("configs/")
    )
    return _read_json(_repo_path(config_artifact["path"]), "response source config")


def _response_header_path(config: Mapping[str, Any], object_id: str) -> tuple[Path, dict[str, Any]]:
    source_config = _response_source_config(config)
    row = next(
        item
        for item in source_config["files"]
        if item["object_id"] == object_id and item["role"] == "HI_MOM1_NATURAL_VELOCITY_FIELD"
    )
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "response source missing")
    _require(path.stat().st_size == row["bytes"], "response source size changed")
    _require(file_sha256(path) == row["sha256"], "response source bytes changed")
    return path, row


def _response_header_only(path: Path) -> fits.Header:
    """Open only the FITS header; response pixels remain undecoded."""
    return fits.getheader(path)


def _field_config(config: Mapping[str, Any]) -> dict[str, Any]:
    grid = config["field_grid"]
    return {
        "grid_contract": {"solver_half_box_kpc": float(grid["half_box_kpc"])},
        "source_cell": config["source_cell"],
    }


def _solve_grid(
    config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
    maps: Mapping[str, Any],
    expected_source: Mapping[str, Any],
    *,
    exponential_scale_pc: float,
    nodes: int,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    field_config = _field_config(config)
    built = mechanics._build_density(
        field_config,
        bridge_config,
        maps,
        exponential_scale_pc=exponential_scale_pc,
        nodes=nodes,
    )
    rhs = 4.0 * math.pi * built["density_dimensionless"]
    newton, newton_residual = mechanics.solve_poisson_dst(
        rhs, built["newton_boundary"], built["grid"].spacing
    )
    operator = config["operator"]
    epsilon = rg.published_permittivity(
        built["density_g_cm3"],
        epsilon_0=float(operator["epsilon_0"]),
        rho_c=10.0 ** float(operator["log10_rho_c_g_cm3"]),
        q_slope=float(operator["Q"]),
    )
    refracted, rg_metrics = mechanics.solve_variable_pcg(
        rhs,
        built["newton_boundary"] / float(operator["epsilon_0"]),
        epsilon,
        built["grid"].spacing,
        relative_tolerance=float(operator["pcg_relative_tolerance"]),
        absolute_tolerance=float(operator["pcg_absolute_tolerance"]),
        max_iterations=int(operator["pcg_max_iterations"]),
        initial_potential=newton / float(operator["epsilon_0"]),
    )
    newton_acceleration = baseline.acceleration(newton, built["grid"].spacing)
    rg_acceleration = baseline.acceleration(refracted, built["grid"].spacing)
    middle = nodes // 2
    fields = {
        _MODELS[0]: (
            np.asarray(newton_acceleration[0][:, :, middle]).copy(),
            np.asarray(newton_acceleration[1][:, :, middle]).copy(),
        ),
        _MODELS[1]: (
            np.asarray(rg_acceleration[0][:, :, middle]).copy(),
            np.asarray(rg_acceleration[1][:, :, middle]).copy(),
        ),
    }
    source_mass_error = mechanics.source_systematics._mass_error(built["masses"], expected_source)
    metrics = {
        "nodes_per_axis": nodes,
        "spacing_kpc": built["grid"].spacing * float(config["field_grid"]["half_box_kpc"]),
        "source_mass_relative_error": source_mass_error,
        "dimensionless_mass_relative_error": built["dimensionless_mass_relative_error"],
        "newton_relative_residual": newton_residual,
        "refracted_gravity_solver": rg_metrics,
        "density_sha256": mechanics.array_sha256(built["density_dimensionless"]),
        "epsilon_sha256": mechanics.array_sha256(epsilon),
        "newton_potential_sha256": mechanics.array_sha256(newton),
        "refracted_gravity_potential_sha256": mechanics.array_sha256(refracted),
    }
    del built, rhs, newton, refracted, epsilon, newton_acceleration, rg_acceleration
    gc.collect()
    return metrics, fields


def _world_grid(header: fits.Header) -> tuple[np.ndarray, np.ndarray]:
    _require(int(header["NAXIS1"]) == 1024 and int(header["NAXIS2"]) == 1024, "shape changed")
    y, x = np.indices((1024, 1024), dtype=np.float64)
    ra, dec = WCS(header).celestial.all_pix2world(x, y, 0)
    _require(bool(np.all(np.isfinite(ra) & np.isfinite(dec))), "WCS transform failed")
    return np.asarray(ra), np.asarray(dec)


def _disk_sky_coordinates(
    metadata: Mapping[str, Any], ra_deg: np.ndarray, dec_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_kpc = float(metadata["distance_mpc"]) * 1000.0
    delta_ra = (ra_deg - float(metadata["ra_deg"]) + 180.0) % 360.0 - 180.0
    east_kpc = (
        np.deg2rad(delta_ra) * math.cos(math.radians(float(metadata["dec_deg"]))) * distance_kpc
    )
    north_kpc = np.deg2rad(dec_deg - float(metadata["dec_deg"])) * distance_kpc
    return projection.disk_coordinates(
        east_kpc,
        north_kpc,
        position_angle_deg=float(metadata["position_angle_deg"]),
        inclination_deg=float(metadata["inclination_deg"]),
    )


def _sample_force(
    field: tuple[np.ndarray, np.ndarray],
    major_kpc: np.ndarray,
    disk_y_kpc: np.ndarray,
    radius_kpc: np.ndarray,
    *,
    half_box_kpc: float,
    a0_m_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    nodes = field[0].shape[0]
    spacing = 2.0 / (nodes - 1)
    i = (major_kpc / half_box_kpc + 1.0) / spacing
    j = (disk_y_kpc / half_box_kpc + 1.0) / spacing
    gx = map_coordinates(field[0], [i, j], order=1, mode="constant", cval=np.nan, prefilter=False)
    gy = map_coordinates(field[1], [i, j], order=1, mode="constant", cval=np.nan, prefilter=False)
    cosine = np.divide(major_kpc, radius_kpc, out=np.zeros_like(radius_kpc), where=radius_kpc > 0)
    sine = np.divide(disk_y_kpc, radius_kpc, out=np.zeros_like(radius_kpc), where=radius_kpc > 0)
    radial = -(gx * cosine + gy * sine) * a0_m_s2
    tangential = (-gx * sine + gy * cosine) * a0_m_s2
    return radial, tangential


def _beam_covariance(major_pixels: float, minor_pixels: float, pa_deg: float) -> np.ndarray:
    factor = 1.0 / math.sqrt(8.0 * math.log(2.0))
    major_sigma = major_pixels * factor
    minor_sigma = minor_pixels * factor
    angle = math.radians(pa_deg)
    major = np.asarray([math.sin(angle), math.cos(angle)])
    minor = np.asarray([-math.cos(angle), math.sin(angle)])
    return major_sigma**2 * np.outer(major, major) + minor_sigma**2 * np.outer(minor, minor)


def additional_beam(
    source_beam_deg: tuple[float, float, float],
    target_beam_deg: tuple[float, float, float],
    pixel_scale_deg: float,
) -> dict[str, float | int | np.ndarray]:
    source = _beam_covariance(
        source_beam_deg[0] / pixel_scale_deg,
        source_beam_deg[1] / pixel_scale_deg,
        source_beam_deg[2],
    )
    target = _beam_covariance(
        target_beam_deg[0] / pixel_scale_deg,
        target_beam_deg[1] / pixel_scale_deg,
        target_beam_deg[2],
    )
    covariance = 0.5 * ((target - source) + (target - source).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    _require(float(np.min(eigenvalues)) > 0.0, "beam covariance difference is not positive")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    factor = math.sqrt(8.0 * math.log(2.0))
    major_pixels = factor * math.sqrt(float(eigenvalues[0]))
    minor_pixels = factor * math.sqrt(float(eigenvalues[1]))
    vector = eigenvectors[:, 0]
    pa_deg = math.degrees(math.atan2(float(vector[0]), float(vector[1])))
    size = 2 * math.ceil(5.0 * math.sqrt(float(eigenvalues[0]))) + 1
    size = max(int(size), 3)
    if size % 2 == 0:
        size += 1
    kernel = projection.elliptical_gaussian_kernel(
        size,
        beam_major_pixels=major_pixels,
        beam_minor_pixels=minor_pixels,
        beam_position_angle_deg=pa_deg,
    )
    return {
        "kernel": kernel,
        "kernel_size": size,
        "major_pixels": major_pixels,
        "minor_pixels": minor_pixels,
        "position_angle_deg": pa_deg,
        "covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def _native_robust_intensity(
    images: Mapping[str, tuple[np.ndarray, fits.Header]], ra: np.ndarray, dec: np.ndarray
) -> tuple[np.ndarray, fits.Header]:
    raw, header = images["HI_MOM0_ROBUST_PRIMARY"]
    sampled = sources.base._sample_image(raw, header, ra, dec, use_sip=False, order=1)
    intensity = np.maximum(np.where(np.isfinite(sampled), sampled, 0.0), 0.0)
    _require(float(np.max(intensity)) > 0.0, "source intensity vanished")
    return intensity, header


def _predict_object(
    config: Mapping[str, Any],
    object_id: str,
    source_config: Mapping[str, Any],
    geometry: Mapping[str, Any],
    paths: Mapping[tuple[str, str], Path],
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    bridge_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = sources.geometry_variants(source_config, dict(geometry[object_id]))[0]
    images = sources._load_images(object_id, dict(paths))
    maps = sources._maps(
        source_config,
        metadata,
        images,
        n=int(config["field_grid"]["source_map_pixels"]),
        box_kpc=float(config["field_grid"]["source_map_box_kpc"]),
    )
    _rhalf_pc, exponential_scale_pc = sources._scale_length(maps)
    source_id = config["source_cell"]["id"]
    fine_metrics, fine_fields = _solve_grid(
        config,
        bridge_config,
        maps,
        expected[(object_id, source_id)],
        exponential_scale_pc=exponential_scale_pc,
        nodes=int(config["field_grid"]["fine_nodes_per_axis"]),
    )
    coarse_metrics, coarse_fields = _solve_grid(
        config,
        bridge_config,
        maps,
        expected[(object_id, source_id)],
        exponential_scale_pc=exponential_scale_pc,
        nodes=int(config["field_grid"]["convergence_nodes_per_axis"]),
    )
    response_path, response_row = _response_header_path(config, object_id)
    response_header = _response_header_only(response_path)
    ra, dec = _world_grid(response_header)
    major, disk_y, radius, cosine = _disk_sky_coordinates(metadata, ra, dec)
    intensity_robust, robust_header = _native_robust_intensity(images, ra, dec)
    robust_beam = sources.base._things_beam(robust_header)
    target_beam = (
        float(response_row["beam_major_deg"]),
        float(response_row["beam_minor_deg"]),
        float(response_row["beam_position_angle_deg"]),
    )
    pixel_scale_deg = abs(float(response_header["CDELT2"]))
    beam = additional_beam(robust_beam, target_beam, pixel_scale_deg)
    half_box = float(config["field_grid"]["half_box_kpc"])
    a0 = float(bridge_config["normalization_contract"]["a0_m_s2"])
    arrays: dict[str, np.ndarray] = {
        "radius_kpc": radius.astype(np.float32),
    }
    model_rows: dict[str, Any] = {}
    global_eligible = (
        np.isfinite(radius)
        & (radius >= float(config["field_grid"]["minimum_radius_kpc"]))
        & (radius <= float(config["field_grid"]["maximum_radius_kpc"]))
        & (intensity_robust > 0.0)
    )
    for model_id, prefix in ((_MODELS[0], "newton"), (_MODELS[1], "rg")):
        fine_radial, fine_tangential = _sample_force(
            fine_fields[model_id],
            major,
            disk_y,
            radius,
            half_box_kpc=half_box,
            a0_m_s2=a0,
        )
        coarse_radial, _coarse_tangential = _sample_force(
            coarse_fields[model_id],
            major,
            disk_y,
            radius,
            half_box_kpc=half_box,
            a0_m_s2=a0,
        )
        relative = np.abs(fine_radial - coarse_radial) / np.maximum(
            np.maximum(np.abs(fine_radial), np.abs(coarse_radial)), a0 * 1.0e-4
        )
        model_eligible = (
            np.isfinite(fine_radial)
            & np.isfinite(coarse_radial)
            & (fine_radial > 0.0)
            & (
                relative
                <= float(
                    config["field_grid"]["maximum_local_radial_acceleration_relative_difference"]
                )
            )
        )
        global_eligible &= model_eligible
        radius_m = radius * 1000.0 * float(bridge_config["normalization_contract"]["pc_m"])
        speed = np.sqrt(np.maximum(radius_m * fine_radial, 0.0))
        raw_vlos = speed * math.sin(math.radians(float(metadata["inclination_deg"]))) * cosine
        convolved, denominator = projection.intensity_weighted_beam(
            np.where(np.isfinite(raw_vlos), raw_vlos, 0.0),
            intensity_robust,
            np.asarray(beam["kernel"]),
        )
        tangential_ratio = np.abs(fine_tangential) / np.maximum(np.abs(fine_radial), a0 * 1.0e-4)
        arrays[f"{prefix}_vlos_plus_m_s"] = convolved.astype(np.float64)
        arrays[f"{prefix}_tangential_ratio"] = tangential_ratio.astype(np.float32)
        arrays[f"{prefix}_convergence_relative"] = relative.astype(np.float32)
        model_rows[model_id] = {
            "fine_positive_pixels": int(
                np.count_nonzero(np.isfinite(fine_radial) & (fine_radial > 0))
            ),
            "fine_coarse_converged_pixels": int(np.count_nonzero(model_eligible)),
            "maximum_eligible_tangential_ratio": None,
            "beam_finite_pixels": int(np.count_nonzero(np.isfinite(convolved))),
            "fine_field_sha256": {
                "gx": array_sha256(fine_fields[model_id][0]),
                "gy": array_sha256(fine_fields[model_id][1]),
            },
            "coarse_field_sha256": {
                "gx": array_sha256(coarse_fields[model_id][0]),
                "gy": array_sha256(coarse_fields[model_id][1]),
            },
        }
    denominator = projection.intensity_weighted_beam(
        np.ones_like(intensity_robust), intensity_robust, np.asarray(beam["kernel"])
    )[1]
    global_eligible &= np.isfinite(denominator) & (
        denominator > 1.0e-10 * float(np.max(denominator))
    )
    arrays["source_intensity"] = denominator.astype(np.float32)
    arrays["source_eligibility"] = global_eligible.astype(np.uint8)
    eligible_count = int(np.count_nonzero(global_eligible))
    _require(eligible_count > 0, f"no eligible pixels for {object_id}")
    for model_id, prefix in ((_MODELS[0], "newton"), (_MODELS[1], "rg")):
        values = arrays[f"{prefix}_tangential_ratio"]
        model_rows[model_id]["maximum_eligible_tangential_ratio"] = float(
            np.max(values[global_eligible])
        )
    operator = config["operator"]
    all_solver_gates = (
        fine_metrics["newton_relative_residual"]
        <= float(operator["maximum_solver_relative_residual"])
        and fine_metrics["refracted_gravity_solver"]["relative_residual"]
        <= float(operator["maximum_solver_relative_residual"])
        and coarse_metrics["newton_relative_residual"]
        <= float(operator["maximum_solver_relative_residual"])
        and coarse_metrics["refracted_gravity_solver"]["relative_residual"]
        <= float(operator["maximum_solver_relative_residual"])
        and fine_metrics["source_mass_relative_error"]
        <= float(operator["maximum_source_mass_relative_error"])
        and coarse_metrics["source_mass_relative_error"]
        <= float(operator["maximum_source_mass_relative_error"])
    )
    object_row = {
        "object_id": object_id,
        "geometry": metadata,
        "kinematic_disposition": next(
            row["kinematic_disposition"]
            for row in config["objects"]
            if row["object_id"] == object_id
        ),
        "response_header": {
            "path": response_row["relative_path"],
            "raw_sha256": response_row["sha256"],
            "shape": [int(response_header["NAXIS2"]), int(response_header["NAXIS1"])],
            "wcs_ctype": [response_header["CTYPE1"], response_header["CTYPE2"]],
            "data_values_decoded": 0,
        },
        "beam_transition": {
            "source_robust_deg": list(robust_beam),
            "target_natural_deg": list(target_beam),
            "kernel_size": beam["kernel_size"],
            "additional_major_pixels": beam["major_pixels"],
            "additional_minor_pixels": beam["minor_pixels"],
            "additional_position_angle_deg": beam["position_angle_deg"],
            "covariance_minimum_eigenvalue": beam["covariance_minimum_eigenvalue"],
        },
        "fine_solver": fine_metrics,
        "convergence_solver": coarse_metrics,
        "models": model_rows,
        "source_intensity_positive_pixels": int(np.count_nonzero(intensity_robust > 0.0)),
        "eligible_prediction_pixels": eligible_count,
        "all_solver_gates_pass": bool(all_solver_gates),
    }
    _require(all_solver_gates, f"solver gate failed for {object_id}")
    del fine_fields, coarse_fields, maps, images, ra, dec, major, disk_y, radius, cosine
    gc.collect()
    return object_row, arrays


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _array_manifest(object_id: str, arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        relative_path = PRIVATE_DIRECTORY / object_id / f"{name}.npy"
        finite = (
            np.isfinite(array)
            if np.issubdtype(array.dtype, np.floating)
            else np.ones_like(array, bool)
        )
        rows.append(
            {
                "object_id": object_id,
                "array_id": name,
                "relative_path": relative_path.as_posix(),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "array_sha256": array_sha256(array),
                "file_sha256": hashlib.sha256(_npy_bytes(array)).hexdigest(),
                "finite_count": int(np.count_nonzero(finite)),
            }
        )
    return rows


def build_packet(
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any], dict[str, Any]]:
    validate_config(config)
    receipts = _validate_bindings(config)
    source_config, _acquisition, geometry, paths, expected = source_resolution._source_evidence(
        source_resolution.load_config()
    )
    bridge_config = bridge.load_config()
    objects: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for object_id in _OBJECTS:
        object_row, object_arrays = _predict_object(
            config,
            object_id,
            source_config,
            geometry,
            paths,
            expected,
            bridge_config,
        )
        objects.append(object_row)
        arrays[object_id] = object_arrays
    array_rows = [
        row for object_id in _OBJECTS for row in _array_manifest(object_id, arrays[object_id])
    ]
    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_FIXED_RESPONSE_BLIND_MODEL_LIFTED_2P5D_PREDICTIONS",
        "objects": objects,
        "arrays": array_rows,
        "array_count": len(array_rows),
        "array_root_sha256": content_sha256(array_rows),
        "scientific_boundary": config["scientific_boundary"],
        "content_sha256": "",
    }
    manifest["content_sha256"] = content_sha256({**manifest, "content_sha256": ""})
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_FIXED_RESPONSE_BLIND_MODEL_LIFTED_2P5D_PREDICTIONS",
        "decision": "READY_FOR_PREREGISTERED_MATCHED_PAIR_THINGS_PIXEL_SCORE",
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "binding_content_sha256": {
            role: value["content_sha256"] for role, value in sorted(receipts.items())
        },
        "primary_sources": config["primary_sources"],
        "source_cell": config["source_cell"],
        "field_grid": config["field_grid"],
        "operator": config["operator"],
        "projection": config["projection"],
        "objects": objects,
        "private_manifest_path": PRIVATE_MANIFEST_PATH.as_posix(),
        "private_manifest_content_sha256": manifest["content_sha256"],
        "private_array_count": len(array_rows),
        "private_array_root_sha256": manifest["array_root_sha256"],
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return arrays, manifest, receipt


def validate_receipt_payload(
    config: Mapping[str, Any], payload: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> None:
    validate_config(config)
    _require(dict(payload) == dict(rebuilt), "receipt differs from rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing nonidentical overwrite")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent nonidentical output")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_packet() -> str:
    config = load_config()
    arrays, manifest, receipt = build_packet(config)
    statuses: list[str] = []
    for object_id in _OBJECTS:
        for name, value in arrays[object_id].items():
            statuses.append(
                _atomic_no_clobber(
                    _repo_path(PRIVATE_DIRECTORY / object_id / f"{name}.npy"), _npy_bytes(value)
                )
            )
    statuses.append(
        _atomic_no_clobber(_repo_path(PRIVATE_MANIFEST_PATH), canonical_bytes(manifest) + b"\n")
    )
    statuses.append(_atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n"))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def check_packet() -> None:
    config = load_config()
    arrays, manifest, receipt = build_packet(config)
    _require(
        _repo_path(PRIVATE_MANIFEST_PATH).read_bytes() == canonical_bytes(manifest) + b"\n",
        "private manifest differs",
    )
    for object_id in _OBJECTS:
        for name, value in arrays[object_id].items():
            _require(
                _repo_path(PRIVATE_DIRECTORY / object_id / f"{name}.npy").read_bytes()
                == _npy_bytes(value),
                "private array differs",
            )
    path = _repo_path(OUTPUT_PATH)
    _require(path.read_bytes() == canonical_bytes(receipt) + b"\n", "receipt differs")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"), nargs="?", default="check")
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_packet())
    elif args.command == "check":
        check_packet()
        print("VALID")
    else:
        output = _repo_path(OUTPUT_PATH)
        if output.is_file():
            receipt = _read_json(output, "receipt")
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "decision": receipt["decision"],
                        "objects": len(receipt["objects"]),
                        "eligible_pixels": sum(
                            row["eligible_prediction_pixels"] for row in receipt["objects"]
                        ),
                        "velocity_pixel_values_decoded": 0,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps({"status": "UNBUILT_RESPONSE_BLIND", "output_exists": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
