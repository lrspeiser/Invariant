"""Build sealed response-blind Holmberg II two-dimensional velocity predictions."""

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
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as mechanics,
)
from sigma_theory_compiler import open_gravity_rg_things_2d_projection_benchmark_v1 as projection

CONFIG_PATH = Path(
    "configs/open_gravity_rg_holmberg_ii_things_2d_response_blind_predictions_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/"
    "open_gravity_rg_holmberg_ii_things_2d_response_blind_predictions_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_rg_holmberg_ii_things_2d_response_blind_predictions_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-holmberg-ii-things-2d-response-blind-predictions-v1/receipt.json"
)
PRIVATE_DIRECTORY = Path(
    "work/private/open-gravity-rg-holmberg-ii-things-2d-response-blind-predictions-v1"
)
PRIVATE_MANIFEST_PATH = PRIVATE_DIRECTORY / "manifest.json"

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-response-blind-predictions-1.0"
_CELL_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-response-blind-prediction-cell-1.0"
_MANIFEST_SCHEMA = (
    "invariant-open-gravity-rg-holmberg-ii-things-2d-response-blind-private-manifest-1.0"
)
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-holmberg-ii-things-2d-response-blind-predictions-receipt-1.0"
)
_CANDIDATES = (
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
)
_RESOLUTIONS = ("ROBUST", "NATURAL")
_CONFIG_RAW_SHA256 = "d5de3b5601d95c5a413789dd432527375e404279bbe8e030d6109c4deb43789a"
_CONFIG_CONTENT_SHA256 = "e6d4bd8fc85ec5225cdaa22e0ef0fc5cb8829c12bd03b859a5a0b2eb018fc323"
_MODULE_SEMANTIC_SHA256 = "bd6155408a66e0110d5ae7a715ac7452ba96e8c2e4a03744a6f16d0f19a4a869"
_TEST_RAW_SHA256 = "48c0bb9a742bd09abcd8873fc3449542e98c18aefdbaa0f9cbf1b66cb0703ff2"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class HolmbergPredictionError(RuntimeError):
    """Raised when a source, numerical, response-blind, or seal gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HolmbergPredictionError(message)


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
        rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes()
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
        raise HolmbergPredictionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_RESPONSE_BLIND_HOLMBERG_II_2D_FOUR_LAW_PREDICTION_BUILD",
        "status changed",
    )
    expected_roles = [
        "HOLMBERG_II_2D_REPLICATION_PREFLIGHT",
        "SEVEN_HOLDOUT_SOURCE_BUILDER",
        "AUDITED_3D_DST_PCG_MECHANICS",
        "PUBLISHED_CONTROL_FORMULAS",
        "AUDITED_2D_WCS_BEAM_PROJECTION",
    ]
    _require(
        [row["role"] for row in config["predecessor_bindings"]] == expected_roles,
        "predecessor inventory changed",
    )
    candidates = config["candidate_contract"]
    _require(candidates["candidate_ids"] == list(_CANDIDATES), "candidates changed")
    _require(candidates["a0_m_s2"] == 1.2e-10, "a0 changed")
    _require(
        candidates["refracted_gravity_parameters"]
        == {
            "published_parameter_id": "DISKMASS_UNIVERSAL_MEDIAN",
            "epsilon_0": 0.661,
            "Q": 1.79,
            "log10_rho_c_g_cm3": -24.54,
        },
        "RG parameters changed",
    )
    for key in ("response_parameter_fitting", "response_parameter_tuning", "best_cell_selection"):
        _require(candidates[key] is False, f"forbidden selection enabled: {key}")
    source = config["source_contract"]
    _require(source["object_id"] == "UGC04305", "object changed")
    _require(
        source["stellar_conversion_cells"]
        == ["IRAC1_FIXED_ML0P6", "IRAC1_GLOBAL_COLOR_ML", "IRAC1_IRAC2_FASTICA36"],
        "source conversions changed",
    )
    _require(source["inclination_cells_deg"] == [27.0, 38.0, 49.0], "inclinations changed")
    _require(source["source_cells"] == 9, "source cell count changed")
    _require(source["model_lift_label"] == "MODEL_LIFTED_2P5D", "3D overclaim")
    _require(source["response_values_used"] is False, "response entered source")
    header = config["response_header_contract"]
    _require(header["shape"] == [1, 1, 1024, 1024], "response shape changed")
    _require(header["pixel_values_decoded"] == 0, "response values entered header contract")
    grid = config["grid_contract"]
    _require(grid["solver_half_box_kpc"] == 30.0, "box changed")
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["minimum_radius_kpc"] == 0.5, "minimum radius changed")
    _require(grid["maximum_radius_kpc"] == 15.0, "maximum radius changed")
    operator = config["operator_contract"]
    _require(operator["pcg_relative_tolerance"] == 1e-10, "PCG tolerance changed")
    _require(operator["pcg_absolute_tolerance"] == 0.0, "PCG absolute tolerance changed")
    _require(operator["pcg_max_iterations"] == 100, "PCG iteration ceiling changed")
    _require(operator["maximum_solver_relative_residual"] == 1e-8, "residual gate changed")
    _require(operator["maximum_source_mass_relative_error"] == 2e-9, "mass gate changed")
    _require(operator["maximum_local_relative_difference"] == 0.05, "local gate changed")
    projection_contract = config["projection_contract"]
    _require(
        projection_contract["minimum_natural_eligible_intensity_fraction"] == 0.99,
        "coverage gate changed",
    )
    _require(projection_contract["response_values_used"] is False, "response entered projection")
    execution = config["execution_contract"]
    _require(execution["source_cells"] == 9, "execution cells changed")
    _require(execution["field_solver_runs"] == 36, "solver accounting changed")
    _require(execution["candidate_resolution_predictions"] == 72, "prediction count changed")
    _require(execution["private_arrays_per_cell"] == 13, "array count changed")
    _require(execution["private_array_files"] == 117, "array total changed")
    for key in ("response_pixels_opened", "network_calls", "model_calls", "paid_calls"):
        _require(execution[key] == 0, f"forbidden execution enabled: {key}")
    private = config["private_output"]
    _require(private["directory"] == PRIVATE_DIRECTORY.as_posix(), "private directory changed")
    _require(private["manifest"] == PRIVATE_MANIFEST_PATH.as_posix(), "manifest changed")
    _require(len(private["array_roles"]) == 13, "private roles changed")
    _require(all(value == 0 for value in config["response_boundary"].values()), "response leak")
    claims = config["claim_boundary"]
    _require(claims["response_assets_sealed"] is True, "response seal removed")
    _require(claims["response_headers_validated"] is True, "header validation removed")
    for key in (
        "response_pixels_opened",
        "response_blind_predictions_built",
        "two_dimensional_score_completed",
        "holmberg_ii_signal_replicated",
        "inclination_resolved",
        "refracted_gravity_generalizes",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim promoted before build: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


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


def _load_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        receipt: dict[str, Any] | None = None
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor artifact changed")
            if artifact["path"].endswith("receipt.json"):
                receipt = _read_json(path, "predecessor receipt")
        _require(receipt is not None, "predecessor receipt missing")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        receipts[binding["role"]] = receipt
    preflight_binding = config["predecessor_bindings"][0]
    preflight_config = _read_json(
        _repo_path(preflight_binding["artifacts"][0]["path"]), "preflight config"
    )
    _require(
        preflight_config["status"] == preflight_binding["required_status"],
        "preflight status changed",
    )
    receipts["HOLMBERG_II_2D_REPLICATION_PREFLIGHT"]["_config"] = preflight_config
    source_binding = config["predecessor_bindings"][1]
    receipts["SEVEN_HOLDOUT_SOURCE_BUILDER"]["_config"] = _read_json(
        _repo_path(source_binding["artifacts"][0]["path"]), "source config"
    )
    return receipts


def _validate_header(header: fits.Header, contract: Mapping[str, Any]) -> None:
    shape = [int(header[f"NAXIS{index}"]) for index in (4, 3, 2, 1)]
    _require(shape == contract["shape"], "FITS shape changed")
    _require(str(header["BUNIT"]).strip() == contract["bunit"], "FITS unit changed")
    _require([header["CTYPE1"], header["CTYPE2"]] == contract["ctype"], "WCS type changed")
    observed = {
        "crval_deg": [float(header["CRVAL1"]), float(header["CRVAL2"])],
        "crpix": [float(header["CRPIX1"]), float(header["CRPIX2"])],
        "cdelt_deg": [float(header["CDELT1"]), float(header["CDELT2"])],
    }
    for key, values in observed.items():
        _require(
            all(
                math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12)
                for value, expected in zip(values, contract[key], strict=True)
            ),
            f"WCS {key} changed",
        )


def _load_response_headers(
    config: Mapping[str, Any], preflight_config: Mapping[str, Any]
) -> dict[tuple[str, str], fits.Header]:
    headers: dict[tuple[str, str], fits.Header] = {}
    for row in preflight_config["response_assets"]:
        path = _repo_path(row["relative_path"])
        _require(path.is_file(), "response asset missing")
        _require(path.stat().st_size == row["bytes"], "response asset size changed")
        _require(file_sha256(path) == row["sha256"], "response asset bytes changed")
        header = fits.getheader(path)
        _validate_header(header, config["response_header_contract"])
        headers[(row["resolution"], row["observable"])] = header
    _require(
        set(headers)
        == {("NATURAL", "MOM1"), ("NATURAL", "MOM2"), ("ROBUST", "MOM1"), ("ROBUST", "MOM2")},
        "response header inventory changed",
    )
    return headers


def cell_run_id(source_cell: Mapping[str, Any]) -> str:
    return (
        f"{source_cell['object_id']}__{source_cell['conversion_cell_id']}__"
        f"{source_cell['geometry']['geometry_variant_id']}"
    )


def _source_cells(
    config: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    cells = [
        row
        for row in source_receipt["source_cells"]
        if row["object_id"] == config["source_contract"]["object_id"]
        and row["disposition"] == "SOURCE_MAP_BUILT_RESPONSE_BLIND"
    ]
    conversion_order = {
        value: index
        for index, value in enumerate(config["source_contract"]["stellar_conversion_cells"])
    }
    inclination_order = {
        float(value): index
        for index, value in enumerate(config["source_contract"]["inclination_cells_deg"])
    }
    cells.sort(
        key=lambda row: (
            conversion_order[row["conversion_cell_id"]],
            inclination_order[float(row["geometry"]["inclination_deg"])],
        )
    )
    _require(len(cells) == 9, "Holmberg II source cell count changed")
    _require(len({cell_run_id(row) for row in cells}) == 9, "source cell IDs duplicated")
    for row in cells:
        _require(row["model_lift_label"] == "MODEL_LIFTED_2P5D", "source dimension changed")
        _require(float(row["geometry"]["position_angle_deg"]) == 175.0, "position angle changed")
    return cells


def _load_source_arrays(
    source_receipt: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_cell: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    private = _repo_path(source_config["private_output_directory"])
    _require(private.is_dir(), "source private directory missing")
    expected_roles = {
        "stellar_surface_msun_pc2",
        "hi_surface_msun_pc2",
        "co_surface_msun_pc2",
        "x_pc",
        "y_pc",
    }
    _require(
        {row["role"] for row in source_cell["array_files"]} == expected_roles,
        "source array roles changed",
    )
    _require(source_receipt["private_array_file_count"] == 120, "source array count changed")
    arrays: dict[str, np.ndarray] = {}
    for row in source_cell["array_files"]:
        path = (private / row["relative_path"]).resolve()
        _require(private in path.parents, "source array path escaped")
        _require(path.is_file(), "source array missing")
        _require(file_sha256(path) == row["sha256"], "source array changed")
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise HolmbergPredictionError("invalid source array") from exc
        _require(list(value.shape) == row["shape"], "source array shape changed")
        _require(str(value.dtype) == row["dtype"], "source array dtype changed")
        _require(bool(np.all(np.isfinite(value))), "source array nonfinite")
        arrays[row["role"]] = np.asarray(value, dtype=np.float64)
    return arrays


def _field_config(config: Mapping[str, Any], source_cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "grid_contract": {
            "solver_half_box_kpc": float(config["grid_contract"]["solver_half_box_kpc"])
        },
        "source_cell": {
            "stellar_height_over_exponential_scale": 4.0 / 29.2,
            "gas_height_pc": float(source_cell["summary"]["hgas_pc"]),
        },
    }


def _solve_grid(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    source_cell: Mapping[str, Any],
    *,
    nodes: int,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    maps = {
        "stellar_fixed": arrays["stellar_surface_msun_pc2"],
        "hi": arrays["hi_surface_msun_pc2"],
        "co": arrays["co_surface_msun_pc2"],
        "x_pc": arrays["x_pc"],
        "y_pc": arrays["y_pc"],
        "dx_pc": float(source_cell["dx_pc"]),
    }
    bridge_config = mechanics.bridge.load_config()
    source_ratio = 4.0 / 29.2
    exponential_scale_pc = float(source_cell["summary"]["hstar_pc"]) / source_ratio
    built = mechanics._build_density(
        _field_config(config, source_cell),
        bridge_config,
        maps,
        exponential_scale_pc=exponential_scale_pc,
        nodes=nodes,
    )
    rhs = 4.0 * math.pi * built["density_dimensionless"]
    newton, newton_residual = mechanics.solve_poisson_dst(
        rhs, built["newton_boundary"], built["grid"].spacing
    )
    parameters = config["candidate_contract"]["refracted_gravity_parameters"]
    epsilon = mechanics.rg.published_permittivity(
        built["density_g_cm3"],
        epsilon_0=float(parameters["epsilon_0"]),
        rho_c=10.0 ** float(parameters["log10_rho_c_g_cm3"]),
        q_slope=float(parameters["Q"]),
    )
    operator = config["operator_contract"]
    refracted, rg_metrics = mechanics.solve_variable_pcg(
        rhs,
        built["newton_boundary"] / float(parameters["epsilon_0"]),
        epsilon,
        built["grid"].spacing,
        relative_tolerance=float(operator["pcg_relative_tolerance"]),
        absolute_tolerance=float(operator["pcg_absolute_tolerance"]),
        max_iterations=int(operator["pcg_max_iterations"]),
        initial_potential=newton / float(parameters["epsilon_0"]),
    )
    newton_acceleration = baseline.acceleration(newton, built["grid"].spacing)
    rg_acceleration = baseline.acceleration(refracted, built["grid"].spacing)
    middle = nodes // 2
    fields = {
        _CANDIDATES[0]: (
            np.asarray(newton_acceleration[0][:, :, middle]).copy(),
            np.asarray(newton_acceleration[1][:, :, middle]).copy(),
        ),
        _CANDIDATES[3]: (
            np.asarray(rg_acceleration[0][:, :, middle]).copy(),
            np.asarray(rg_acceleration[1][:, :, middle]).copy(),
        ),
    }
    metrics = {
        "nodes_per_axis": nodes,
        "spacing_kpc": built["grid"].spacing
        * float(config["grid_contract"]["solver_half_box_kpc"]),
        "dimensionless_mass_relative_error": float(built["dimensionless_mass_relative_error"]),
        "source_masses_msun": built["masses"],
        "total_mass_msun": float(built["total_mass_msun"]),
        "newton_relative_residual": float(newton_residual),
        "refracted_gravity_solver": rg_metrics,
        "density_sha256": mechanics.array_sha256(built["density_dimensionless"]),
        "epsilon_sha256": mechanics.array_sha256(epsilon),
        "newton_potential_sha256": mechanics.array_sha256(newton),
        "refracted_gravity_potential_sha256": mechanics.array_sha256(refracted),
        "field_sha256": {
            candidate: {"gx": array_sha256(field[0]), "gy": array_sha256(field[1])}
            for candidate, field in fields.items()
        },
    }
    del built, rhs, newton, refracted, epsilon, newton_acceleration, rg_acceleration
    gc.collect()
    return metrics, fields


def _world_grid(header: fits.Header) -> tuple[np.ndarray, np.ndarray]:
    size_x = int(header["NAXIS1"])
    size_y = int(header["NAXIS2"])
    _require((size_y, size_x) == (1024, 1024), "response image shape changed")
    y, x = np.indices((size_y, size_x), dtype=np.float64)
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


def _sample_source_intensity(
    arrays: Mapping[str, np.ndarray], major_kpc: np.ndarray, disk_y_kpc: np.ndarray
) -> np.ndarray:
    x_axis = arrays["x_pc"][0, :] / 1000.0
    y_axis = arrays["y_pc"][:, 0] / 1000.0
    _require(bool(np.all(np.diff(x_axis) > 0.0)), "source x axis changed")
    _require(bool(np.all(np.diff(y_axis) > 0.0)), "source y axis changed")
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))
    _require(
        float(np.max(np.abs(np.diff(x_axis) - dx))) < 1e-5 * dx,
        "source x grid nonuniform",
    )
    _require(
        float(np.max(np.abs(np.diff(y_axis) - dy))) < 1e-5 * dy,
        "source y grid nonuniform",
    )
    column = (major_kpc - float(x_axis[0])) / dx
    row = (disk_y_kpc - float(y_axis[0])) / dy
    intensity = map_coordinates(
        arrays["hi_surface_msun_pc2"],
        [row, column],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    intensity = np.maximum(np.where(np.isfinite(intensity), intensity, 0.0), 0.0)
    _require(float(np.max(intensity)) > 0.0, "source intensity vanished")
    return intensity


def _candidate_accelerations(
    newton: np.ndarray, refracted: np.ndarray, a0: float
) -> dict[str, np.ndarray]:
    newton = np.asarray(newton, dtype=np.float64)
    refracted = np.asarray(refracted, dtype=np.float64)
    safe = np.maximum(newton, 0.0)
    denominator = -np.expm1(-np.sqrt(safe / a0))
    rar = np.divide(safe, denominator, out=np.zeros_like(safe), where=denominator > 0.0)
    mond = np.sqrt(0.5 * (safe**2 + np.sqrt(safe**4 + 4.0 * safe**2 * a0**2)))
    return {
        _CANDIDATES[0]: newton,
        _CANDIDATES[1]: rar,
        _CANDIDATES[2]: mond,
        _CANDIDATES[3]: refracted,
    }


def _beam_covariance(major_pixels: float, minor_pixels: float, pa_deg: float) -> np.ndarray:
    factor = 1.0 / math.sqrt(8.0 * math.log(2.0))
    major_sigma = major_pixels * factor
    minor_sigma = minor_pixels * factor
    angle = math.radians(pa_deg)
    major = np.asarray([math.sin(angle), math.cos(angle)])
    minor = np.asarray([-math.cos(angle), math.sin(angle)])
    return major_sigma**2 * np.outer(major, major) + minor_sigma**2 * np.outer(minor, minor)


def additional_beam(
    source_beam_deg: Sequence[float], target_beam_deg: Sequence[float], pixel_scale_deg: float
) -> dict[str, Any]:
    source = _beam_covariance(
        float(source_beam_deg[0]) / pixel_scale_deg,
        float(source_beam_deg[1]) / pixel_scale_deg,
        float(source_beam_deg[2]),
    )
    target = _beam_covariance(
        float(target_beam_deg[0]) / pixel_scale_deg,
        float(target_beam_deg[1]) / pixel_scale_deg,
        float(target_beam_deg[2]),
    )
    covariance = 0.5 * ((target - source) + (target - source).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    _require(float(np.min(eigenvalues)) > 0.0, "beam difference is not positive")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    factor = math.sqrt(8.0 * math.log(2.0))
    major_pixels = factor * math.sqrt(float(eigenvalues[0]))
    minor_pixels = factor * math.sqrt(float(eigenvalues[1]))
    vector = eigenvectors[:, 0]
    pa_deg = math.degrees(math.atan2(float(vector[0]), float(vector[1])))
    size = max(2 * math.ceil(5.0 * math.sqrt(float(eigenvalues[0]))) + 1, 3)
    if size % 2 == 0:
        size += 1
    kernel = projection.elliptical_gaussian_kernel(
        int(size),
        beam_major_pixels=major_pixels,
        beam_minor_pixels=minor_pixels,
        beam_position_angle_deg=pa_deg,
    )
    return {
        "kernel": kernel,
        "kernel_size": int(size),
        "major_pixels": major_pixels,
        "minor_pixels": minor_pixels,
        "position_angle_deg": pa_deg,
        "covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def _solver_gates(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, bool]:
    gate = config["operator_contract"]
    return {
        "newton_residual": float(metrics["newton_relative_residual"])
        <= float(gate["maximum_solver_relative_residual"]),
        "rg_residual": float(metrics["refracted_gravity_solver"]["relative_residual"])
        <= float(gate["maximum_solver_relative_residual"]),
        "source_mass": float(metrics["dimensionless_mass_relative_error"])
        <= float(gate["maximum_source_mass_relative_error"]),
    }


def _build_cell_arrays(
    config: Mapping[str, Any],
    source_cell: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    source_config: Mapping[str, Any],
    robust_header: fits.Header,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = _load_source_arrays(source_receipt, source_config, source_cell)
    fine_metrics, fine_fields = _solve_grid(
        config, arrays, source_cell, nodes=int(config["grid_contract"]["fine_nodes_per_axis"])
    )
    coarse_metrics, coarse_fields = _solve_grid(
        config,
        arrays,
        source_cell,
        nodes=int(config["grid_contract"]["convergence_nodes_per_axis"]),
    )
    ra, dec = _world_grid(robust_header)
    metadata = source_cell["geometry"]
    major, disk_y, radius, cosine = _disk_sky_coordinates(metadata, ra, dec)
    intensity = _sample_source_intensity(arrays, major, disk_y)
    a0 = float(config["candidate_contract"]["a0_m_s2"])
    half_box = float(config["grid_contract"]["solver_half_box_kpc"])
    fine_newton, fine_newton_tangential = _sample_force(
        fine_fields[_CANDIDATES[0]], major, disk_y, radius, half_box_kpc=half_box, a0_m_s2=a0
    )
    coarse_newton, _ = _sample_force(
        coarse_fields[_CANDIDATES[0]], major, disk_y, radius, half_box_kpc=half_box, a0_m_s2=a0
    )
    fine_rg, fine_rg_tangential = _sample_force(
        fine_fields[_CANDIDATES[3]], major, disk_y, radius, half_box_kpc=half_box, a0_m_s2=a0
    )
    coarse_rg, _ = _sample_force(
        coarse_fields[_CANDIDATES[3]], major, disk_y, radius, half_box_kpc=half_box, a0_m_s2=a0
    )
    fine = _candidate_accelerations(fine_newton, fine_rg, a0)
    coarse = _candidate_accelerations(coarse_newton, coarse_rg, a0)
    solver_checks = {
        "fine": _solver_gates(config, fine_metrics),
        "convergence": _solver_gates(config, coarse_metrics),
    }
    all_solver_gates_pass = all(
        value for group in solver_checks.values() for value in group.values()
    )
    robust_eligible = (
        np.isfinite(radius)
        & (radius >= float(config["grid_contract"]["minimum_radius_kpc"]))
        & (radius <= float(config["grid_contract"]["maximum_radius_kpc"]))
        & np.isfinite(intensity)
        & (intensity > 0.0)
    )
    candidate_metrics: dict[str, Any] = {}
    relative_maps: dict[str, np.ndarray] = {}
    for candidate in _CANDIDATES:
        relative = np.abs(fine[candidate] - coarse[candidate]) / np.maximum(
            np.maximum(np.abs(fine[candidate]), np.abs(coarse[candidate])), a0 * 1e-4
        )
        valid = (
            np.isfinite(fine[candidate])
            & np.isfinite(coarse[candidate])
            & (fine[candidate] > 0.0)
            & (coarse[candidate] > 0.0)
            & np.isfinite(relative)
            & (relative <= float(config["operator_contract"]["maximum_local_relative_difference"]))
        )
        robust_eligible &= valid
        relative_maps[candidate] = relative
        candidate_metrics[candidate] = {
            "fine_positive_pixels": int(
                np.count_nonzero(np.isfinite(fine[candidate]) & (fine[candidate] > 0.0))
            ),
            "fine_coarse_converged_pixels": int(np.count_nonzero(valid)),
        }
    if not all_solver_gates_pass:
        robust_eligible[:] = False
    pixel_scale = abs(float(robust_header["CDELT2"]))
    response_header = config["response_header_contract"]
    beam = additional_beam(
        response_header["robust_beam_deg"], response_header["natural_beam_deg"], pixel_scale
    )
    full_natural_denominator = projection.intensity_weighted_beam(
        np.ones_like(intensity), intensity, np.asarray(beam["kernel"])
    )[1]
    eligible_intensity = intensity * robust_eligible.astype(np.float64)
    eligible_natural_denominator = projection.intensity_weighted_beam(
        np.ones_like(intensity), eligible_intensity, np.asarray(beam["kernel"])
    )[1]
    coverage = np.divide(
        eligible_natural_denominator,
        full_natural_denominator,
        out=np.zeros_like(full_natural_denominator),
        where=full_natural_denominator > 0.0,
    )
    natural_eligible = (
        np.isfinite(radius)
        & (radius >= float(config["grid_contract"]["minimum_radius_kpc"]))
        & (radius <= float(config["grid_contract"]["maximum_radius_kpc"]))
        & np.isfinite(coverage)
        & (
            coverage
            >= float(config["projection_contract"]["minimum_natural_eligible_intensity_fraction"])
        )
        & (full_natural_denominator > 1e-15 * max(float(np.max(full_natural_denominator)), 1.0))
    )
    bridge_config = mechanics.bridge.load_config()
    radius_m = radius * 1000.0 * float(bridge_config["normalization_contract"]["pc_m"])
    sine_i = math.sin(math.radians(float(metadata["inclination_deg"])))
    output: dict[str, np.ndarray] = {
        "radius_kpc": radius.astype(np.float32),
        "source_intensity_robust": intensity.astype(np.float32),
        "source_intensity_natural": full_natural_denominator.astype(np.float32),
        "robust_eligibility": robust_eligible.astype(np.uint8),
        "natural_eligibility": natural_eligible.astype(np.uint8),
    }
    for candidate in _CANDIDATES:
        speed = np.sqrt(np.maximum(radius_m * fine[candidate], 0.0))
        raw = speed * sine_i * cosine
        robust = np.where(robust_eligible & np.isfinite(raw), raw, 0.0)
        natural, _ = projection.intensity_weighted_beam(
            np.where(robust_eligible & np.isfinite(raw), raw, 0.0),
            eligible_intensity,
            np.asarray(beam["kernel"]),
        )
        natural = np.where(natural_eligible & np.isfinite(natural), natural, 0.0)
        output[f"{candidate}__ROBUST"] = robust.astype(np.float64)
        output[f"{candidate}__NATURAL"] = natural.astype(np.float64)
        eligible_relative = relative_maps[candidate][robust_eligible]
        candidate_metrics[candidate]["maximum_eligible_relative_difference"] = (
            float(np.max(eligible_relative)) if eligible_relative.size else None
        )
        candidate_metrics[candidate]["robust_prediction_pixels"] = int(
            np.count_nonzero(robust_eligible)
        )
        candidate_metrics[candidate]["natural_prediction_pixels"] = int(
            np.count_nonzero(natural_eligible)
        )
    newton_tangential = np.abs(fine_newton_tangential) / np.maximum(np.abs(fine_newton), a0 * 1e-4)
    rg_tangential = np.abs(fine_rg_tangential) / np.maximum(np.abs(fine_rg), a0 * 1e-4)
    diagnostics = {
        "fine_solver": fine_metrics,
        "convergence_solver": coarse_metrics,
        "solver_checks": solver_checks,
        "all_solver_gates_pass": bool(all_solver_gates_pass),
        "candidate_metrics": candidate_metrics,
        "robust_eligible_pixels": int(np.count_nonzero(robust_eligible)),
        "natural_eligible_pixels": int(np.count_nonzero(natural_eligible)),
        "source_positive_pixels": int(np.count_nonzero(intensity > 0.0)),
        "natural_coverage_minimum_eligible": (
            float(np.min(coverage[natural_eligible])) if np.any(natural_eligible) else None
        ),
        "maximum_eligible_tangential_ratio": {
            _CANDIDATES[0]: float(np.max(newton_tangential[robust_eligible]))
            if np.any(robust_eligible)
            else None,
            _CANDIDATES[3]: float(np.max(rg_tangential[robust_eligible]))
            if np.any(robust_eligible)
            else None,
        },
        "beam_transition": {key: value for key, value in beam.items() if key != "kernel"},
    }
    del arrays, fine_fields, coarse_fields, ra, dec, major, disk_y, cosine
    gc.collect()
    return output, diagnostics


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
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
            _require(path.read_bytes() == payload, "concurrent output differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _private_path(relative: str) -> Path:
    private = _repo_path(PRIVATE_DIRECTORY)
    path = (private / relative).resolve()
    _require(path == private or private in path.parents, "private path escaped")
    return path


def _cell_payload_path(source_cell: Mapping[str, Any]) -> Path:
    return _private_path(f"{cell_run_id(source_cell)}__cell.json")


def _array_relative_path(source_cell: Mapping[str, Any], role: str) -> str:
    _require(re.fullmatch(r"[A-Z0-9_]+(?:__[A-Z]+)?|[a-z_]+", role) is not None, "bad role")
    return f"{cell_run_id(source_cell)}__{role}.npy"


def _validate_array_rows(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    expected_roles = config["private_output"]["array_roles"]
    _require([row["role"] for row in rows] == expected_roles, "array roles changed")
    for row in rows:
        path = _private_path(row["relative_path"])
        _require(path.is_file(), "prediction array missing")
        _require(path.stat().st_size == row["bytes"], "prediction array size changed")
        _require(file_sha256(path) == row["file_sha256"], "prediction array bytes changed")
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise HolmbergPredictionError("invalid prediction array") from exc
        _require(list(array.shape) == row["shape"], "prediction shape changed")
        _require(str(array.dtype) == row["dtype"], "prediction dtype changed")
        _require(array_sha256(array) == row["array_sha256"], "prediction values changed")


def _build_cell_payload(
    config: Mapping[str, Any],
    source_cell: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    serialized: dict[str, bytes] = {}
    for role in config["private_output"]["array_roles"]:
        _require(role in arrays, "prediction array role missing")
        payload = _npy_bytes(arrays[role])
        relative = _array_relative_path(source_cell, role)
        serialized[relative] = payload
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "bytes": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "array_sha256": array_sha256(arrays[role]),
                "shape": list(arrays[role].shape),
                "dtype": str(arrays[role].dtype),
            }
        )
    cell: dict[str, Any] = {
        "schema": _CELL_SCHEMA,
        "package_id": config["package_id"],
        "package_bindings": _package_bindings(),
        "cell_run_id": cell_run_id(source_cell),
        "object_id": source_cell["object_id"],
        "conversion_cell_id": source_cell["conversion_cell_id"],
        "geometry": source_cell["geometry"],
        "source_profile_sha256": source_cell["profile_sha256"],
        "model_lift_label": source_cell["model_lift_label"],
        "arrays": rows,
        "diagnostics": diagnostics,
        "response_boundary": config["response_boundary"],
    }
    cell["content_sha256"] = content_sha256(cell)
    return cell, serialized


def validate_cell_payload(
    config: Mapping[str, Any], source_cell: Mapping[str, Any], cell: Mapping[str, Any]
) -> None:
    _require(cell["schema"] == _CELL_SCHEMA, "cell schema changed")
    _require(cell["package_id"] == config["package_id"], "cell package changed")
    _require(cell["package_bindings"] == _package_bindings(), "cell package seal changed")
    _require(cell["cell_run_id"] == cell_run_id(source_cell), "cell ID changed")
    _require(cell["object_id"] == source_cell["object_id"], "cell object changed")
    _require(cell["conversion_cell_id"] == source_cell["conversion_cell_id"], "conversion changed")
    _require(cell["geometry"] == source_cell["geometry"], "geometry changed")
    _require(cell["source_profile_sha256"] == source_cell["profile_sha256"], "source changed")
    _require(cell["model_lift_label"] == "MODEL_LIFTED_2P5D", "dimension overclaim")
    _require(cell["response_boundary"] == config["response_boundary"], "response leak")
    copy = dict(cell)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "cell content hash changed")
    _validate_array_rows(config, cell["arrays"])


def write_cell(cell_id: str) -> str:
    config = load_config()
    predecessors = _load_predecessors(config)
    source_receipt = predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    source_config = source_receipt["_config"]
    source_cells = _source_cells(config, source_receipt)
    source_cell = next((row for row in source_cells if cell_run_id(row) == cell_id), None)
    _require(source_cell is not None, "unknown source cell")
    cell_path = _cell_payload_path(source_cell)
    if cell_path.is_file():
        cell = _read_json(cell_path, "prediction cell")
        validate_cell_payload(config, source_cell, cell)
        return "EXISTING_VALID"
    preflight_config = predecessors["HOLMBERG_II_2D_REPLICATION_PREFLIGHT"]["_config"]
    headers = _load_response_headers(config, preflight_config)
    arrays, diagnostics = _build_cell_arrays(
        config, source_cell, source_receipt, source_config, headers[("ROBUST", "MOM1")]
    )
    cell, serialized = _build_cell_payload(config, source_cell, arrays, diagnostics)
    statuses = [
        _atomic_no_clobber(_private_path(relative), payload)
        for relative, payload in serialized.items()
    ]
    statuses.append(_atomic_no_clobber(cell_path, canonical_bytes(cell) + b"\n"))
    del arrays
    gc.collect()
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def _load_completed_cells(
    config: Mapping[str, Any], source_cells: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_cell in source_cells:
        path = _cell_payload_path(source_cell)
        _require(path.is_file(), f"prediction cell missing: {cell_run_id(source_cell)}")
        cell = _read_json(path, "prediction cell")
        validate_cell_payload(config, source_cell, cell)
        output.append(cell)
    return output


def build_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessors = _load_predecessors(config)
    source_cells = _source_cells(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    cells = _load_completed_cells(config, source_cells)
    arrays = [
        dict(row) | {"cell_run_id": cell["cell_run_id"]} for cell in cells for row in cell["arrays"]
    ]
    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "package_id": config["package_id"],
        "package_bindings": _package_bindings(),
        "cell_count": len(cells),
        "array_file_count": len(arrays),
        "candidate_resolution_prediction_count": len(cells) * len(_CANDIDATES) * len(_RESOLUTIONS),
        "cell_content_sha256": [cell["content_sha256"] for cell in cells],
        "arrays": arrays,
        "response_boundary": config["response_boundary"],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def validate_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    _require(manifest["schema"] == _MANIFEST_SCHEMA, "manifest schema changed")
    _require(manifest["package_id"] == config["package_id"], "manifest package changed")
    _require(manifest["package_bindings"] == _package_bindings(), "manifest seal changed")
    _require(manifest["cell_count"] == 9, "manifest cell count changed")
    _require(manifest["array_file_count"] == 117, "manifest array count changed")
    _require(
        manifest["candidate_resolution_prediction_count"] == 72, "manifest predictions changed"
    )
    _require(manifest["response_boundary"] == config["response_boundary"], "manifest response leak")
    copy = dict(manifest)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "manifest content hash changed")


def write_manifest() -> str:
    config = load_config()
    manifest = build_manifest(config)
    validate_manifest(config, manifest)
    return _atomic_no_clobber(_repo_path(PRIVATE_MANIFEST_PATH), canonical_bytes(manifest) + b"\n")


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _repo_path(PRIVATE_MANIFEST_PATH)
    _require(manifest_path.is_file(), "private manifest missing")
    manifest = _read_json(manifest_path, "private manifest")
    validate_manifest(config, manifest)
    predecessors = _load_predecessors(config)
    preflight_config = predecessors["HOLMBERG_II_2D_REPLICATION_PREFLIGHT"]["_config"]
    _load_response_headers(config, preflight_config)
    source_cells = _source_cells(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    cells = _load_completed_cells(config, source_cells)
    claims = dict(config["claim_boundary"])
    claims["response_blind_predictions_built"] = True
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RESPONSE_BLIND_72_HOLMBERG_II_PREDICTIONS_SEALED",
        "decision": "READY_TO_OPEN_SEALED_THINGS_RESPONSE_FOR_FIXED_2D_SCORE",
        "package_bindings": _package_bindings(),
        "predecessor_receipt_content_sha256": {
            binding["role"]: binding["receipt_content_sha256"]
            for binding in config["predecessor_bindings"]
        },
        "source_cells": 9,
        "candidate_ids": list(_CANDIDATES),
        "response_resolutions": list(_RESOLUTIONS),
        "candidate_resolution_predictions": 72,
        "private_manifest": {
            "path": PRIVATE_MANIFEST_PATH.as_posix(),
            "raw_sha256": file_sha256(manifest_path),
            "content_sha256": manifest["content_sha256"],
            "array_file_count": manifest["array_file_count"],
        },
        "cell_summaries": [
            {
                "cell_run_id": cell["cell_run_id"],
                "content_sha256": cell["content_sha256"],
                "conversion_cell_id": cell["conversion_cell_id"],
                "geometry_variant_id": cell["geometry"]["geometry_variant_id"],
                "inclination_deg": cell["geometry"]["inclination_deg"],
                "all_solver_gates_pass": cell["diagnostics"]["all_solver_gates_pass"],
                "robust_eligible_pixels": cell["diagnostics"]["robust_eligible_pixels"],
                "natural_eligible_pixels": cell["diagnostics"]["natural_eligible_pixels"],
            }
            for cell in cells
        ],
        "all_solver_gates_pass": all(
            cell["diagnostics"]["all_solver_gates_pass"] for cell in cells
        ),
        "response_boundary": config["response_boundary"],
        "execution_accounting": config["execution_contract"],
        "claim_boundary": claims,
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected = build_receipt(config)
    _require(dict(receipt) == expected, "receipt differs from deterministic rebuild")


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    validate_receipt(config, receipt)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n")


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    receipt = _read_json(path, "receipt")
    validate_receipt(config, receipt)
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    predecessors = _load_predecessors(config)
    cells = _source_cells(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    completed = sum(_cell_payload_path(cell).is_file() for cell in cells)
    return {
        "package_id": config["package_id"],
        "completed_cells": completed,
        "required_cells": len(cells),
        "manifest_exists": _repo_path(PRIVATE_MANIFEST_PATH).is_file(),
        "receipt_exists": _repo_path(OUTPUT_PATH).is_file(),
        "response_pixels_opened": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    cell = sub.add_parser("write-cell")
    cell.add_argument("--cell-id", required=True)
    sub.add_parser("write-all")
    sub.add_parser("write-manifest")
    sub.add_parser("write-receipt")
    sub.add_parser("check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "status":
        print(json.dumps(status(), sort_keys=True))
    elif arguments.command == "write-cell":
        print(write_cell(arguments.cell_id))
    elif arguments.command == "write-all":
        config = load_config()
        predecessors = _load_predecessors(config)
        cells = _source_cells(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
        for cell in cells:
            print(f"{cell_run_id(cell)} {write_cell(cell_run_id(cell))}", flush=True)
    elif arguments.command == "write-manifest":
        print(write_manifest())
    elif arguments.command == "write-receipt":
        print(write_receipt())
    else:
        print(check_receipt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
