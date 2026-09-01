from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as controls,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as mechanics,
)

CONFIG_PATH = Path("configs/open_gravity_rg_sings_seven_holdout_response_blind_predictions_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_sings_seven_holdout_response_blind_predictions_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_sings_seven_holdout_response_blind_predictions_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-sings-seven-holdout-response-blind-predictions-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-response-blind-predictions-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-sings-seven-holdout-response-blind-predictions-receipt-1.0"
)
_CELL_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-response-blind-prediction-cell-1.0"
_CONFIG_RAW_SHA256 = "e8c4acae5ff09a81b21f96e8c4bf70397497955b33f5dec38a05c994c9b7f5cd"
_CONFIG_CONTENT_SHA256 = "57523fa2781bbe2f9f506b253e013187ea7cf2802ec3a15bf296f7bad7033d31"
_MODULE_SEMANTIC_SHA256 = "bd69345f9512132ba87ec19cc3f7e67fbc02afa37e74a21bead142cd92da651c"
_TEST_RAW_SHA256 = "f2f4e493f5ea32de3f12a76d54fc4cd46b1ed156249252fe329260e8164529dc"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class PredictionBuildError(RuntimeError):
    """Raised when the response-blind prediction contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictionBuildError(message)


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


def _repo_path(relative: Path | str) -> Path:
    candidate = (_ROOT / relative).resolve()
    _require(candidate == _ROOT or _ROOT in candidate.parents, "path escaped repository")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBuildError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_RESPONSE_BLIND_FOUR_LAW_FORWARD_PREDICTION_BUILD",
        "status changed",
    )
    candidates = config["candidate_contract"]
    _require(
        candidates["candidate_ids"]
        == [
            "NEWTON_3D_DST",
            "RAR_2016_ON_NEWTON_3D",
            "MOND_STANDARD_MU_ON_NEWTON_3D",
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
        ],
        "candidate inventory changed",
    )
    _require(candidates["a0_m_s2"] == 1.2e-10, "a0 changed")
    _require(candidates["per_object_fitted_parameters"] == 0, "object fitting enabled")
    _require(candidates["global_fitted_parameters"] == 0, "global fitting enabled")
    _require(candidates["response_parameter_fitting"] is False, "response fitting enabled")
    rg_parameters = candidates["refracted_gravity_parameters"]
    _require(
        rg_parameters
        == {
            "published_parameter_id": "DISKMASS_UNIVERSAL_MEDIAN",
            "epsilon_0": 0.661,
            "Q": 1.79,
            "log10_rho_c_g_cm3": -24.54,
        },
        "published RG parameters changed",
    )
    grid = config["grid_contract"]
    _require(grid["solver_half_box_kpc"] == 30.0, "box changed")
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["radial_points"] == 291, "radial grid changed")
    execution = config["execution_contract"]
    _require(execution["declared_source_cells"] == 27, "source count changed")
    _require(execution["expected_built_source_cells"] == 24, "built count changed")
    _require(execution["expected_retained_source_failures"] == 3, "failure count changed")
    _require(execution["expected_total_field_solver_runs"] == 96, "solver count changed")
    _require(execution["response_based_cell_or_radius_selection"] is False, "selection leak")
    _require(config["numerical_gate"]["response_values_used"] is False, "gate leak")
    _require(
        all(value == 0 for value in config["response_boundary"].values()),
        "response access enabled",
    )
    _require(
        all(
            config["claim_boundary"][key] is False
            for key in (
                "real_source_predictions_generated",
                "response_blind_numerical_radius_masks_generated",
                "scientific_response_scored",
                "refracted_gravity_preferred",
                "general_3d_gravity_validated",
                "unique_theory_established",
                "publication_ready",
            )
        ),
        "claim promoted before build",
    )
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


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def _load_predecessors(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_roles = [
        "SEVEN_HOLDOUT_SOURCE_BUILDER",
        "AUDITED_3D_DST_PCG_MECHANICS",
        "PUBLISHED_CONTROL_FORMULAS",
    ]
    _require(
        [row["role"] for row in config["predecessor_bindings"]] == expected_roles,
        "predecessor roles changed",
    )
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        receipt_path: Path | None = None
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor artifact changed")
            if artifact["path"].endswith("receipt.json"):
                receipt_path = path
        _require(receipt_path is not None, "predecessor receipt missing")
        receipt = _read_json(receipt_path, "predecessor receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        receipts[binding["role"]] = receipt
    source = receipts["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    _require(source["source_cell_count"] == 27, "source ledger count changed")
    _require(source["built_source_map_count"] == 24, "source built count changed")
    _require(source["failed_source_conversion_count"] == 3, "source failures changed")
    _require(
        all(value == 0 for value in source["response_boundary"].values()),
        "source predecessor exposed response",
    )
    mechanics_receipt = receipts["AUDITED_3D_DST_PCG_MECHANICS"]
    _require(mechanics_receipt["all_object_gates_pass"] is True, "mechanics gate failed")
    return receipts


def _built_source_cells(source_receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cells = [
        row
        for row in source_receipt["source_cells"]
        if row["disposition"] == "SOURCE_MAP_BUILT_RESPONSE_BLIND"
    ]
    _require(len(cells) == 24, "built source-cell inventory changed")
    return cells


def cell_run_id(source_cell: Mapping[str, Any]) -> str:
    return "__".join(
        (
            str(source_cell["object_id"]),
            str(source_cell["conversion_cell_id"]),
            str(source_cell["geometry"]["geometry_variant_id"]),
        )
    )


def _prediction_directory(config: Mapping[str, Any]) -> Path:
    return _repo_path(config["execution_contract"]["private_prediction_directory"])


def _cell_output_path(config: Mapping[str, Any], source_cell: Mapping[str, Any]) -> Path:
    return (_prediction_directory(config) / f"{cell_run_id(source_cell)}.json").resolve()


def _source_private_directory(source_receipt: Mapping[str, Any]) -> Path:
    source_config = _read_json(
        _repo_path(
            "configs/open_gravity_rg_sings_seven_holdout_model_lifted_3d_source_builder_v1.json"
        ),
        "source config",
    )
    private = _repo_path(source_config["private_output_directory"])
    _require(private.is_dir(), "source private directory missing")
    _require(source_receipt["private_array_file_count"] == 120, "source array count changed")
    return private


def _load_source_arrays(
    source_receipt: Mapping[str, Any], source_cell: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    private = _source_private_directory(source_receipt)
    arrays: dict[str, np.ndarray] = {}
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
    for row in source_cell["array_files"]:
        path = (private / row["relative_path"]).resolve()
        _require(private in path.parents, "source array path escaped")
        _require(path.is_file(), "source array missing")
        _require(file_sha256(path) == row["sha256"], "source array changed")
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise PredictionBuildError("invalid source array") from exc
        _require(list(value.shape) == row["shape"], "source array shape changed")
        _require(str(value.dtype) == row["dtype"], "source array dtype changed")
        _require(np.all(np.isfinite(value)), "source array nonfinite")
        arrays[row["role"]] = np.asarray(value, dtype=np.float64)
    return arrays


def _radii(config: Mapping[str, Any]) -> list[float]:
    grid = config["grid_contract"]
    values = np.linspace(
        float(grid["radial_min_kpc"]),
        float(grid["radial_max_kpc"]),
        int(grid["radial_points"]),
        dtype=np.float64,
    )
    _require(
        np.max(np.abs(np.diff(values) - float(grid["radial_step_kpc"]))) < 1.0e-12,
        "radial step changed",
    )
    return [float(value) for value in values]


def _candidate_profiles(
    config: Mapping[str, Any],
    newton_profile: Sequence[Mapping[str, Any]],
    rg_profile: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, float]]]:
    a0 = float(config["candidate_contract"]["a0_m_s2"])
    _require(len(newton_profile) == len(rg_profile), "profile lengths differ")
    output: dict[str, list[dict[str, float]]] = {
        key: [] for key in config["candidate_contract"]["candidate_ids"]
    }
    for newton_row, rg_row in zip(newton_profile, rg_profile, strict=True):
        radius = float(newton_row["radius_kpc"])
        _require(radius == float(rg_row["radius_kpc"]), "profile radii differ")
        newton = float(newton_row["radial_acceleration_m_s2"])
        refracted = float(rg_row["radial_acceleration_m_s2"])
        _require(math.isfinite(newton) and newton >= 0.0, "invalid Newton prediction")
        _require(math.isfinite(refracted) and refracted >= 0.0, "invalid RG prediction")
        values = {
            "NEWTON_3D_DST": newton,
            "RAR_2016_ON_NEWTON_3D": controls.rar_2016(newton, a0),
            "MOND_STANDARD_MU_ON_NEWTON_3D": controls.mond_standard(newton, a0),
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG": refracted,
        }
        for candidate_id, acceleration in values.items():
            _require(math.isfinite(acceleration) and acceleration >= 0.0, "invalid prediction")
            output[candidate_id].append(
                {
                    "radius_kpc": radius,
                    "radial_acceleration_m_s2": float(acceleration),
                }
            )
    return output


def _solve_grid(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    source_cell: Mapping[str, Any],
    *,
    nodes: int,
) -> dict[str, Any]:
    maps = {
        "stellar_fixed": arrays["stellar_surface_msun_pc2"],
        "hi": arrays["hi_surface_msun_pc2"],
        "co": arrays["co_surface_msun_pc2"],
        "x_pc": arrays["x_pc"],
        "y_pc": arrays["y_pc"],
        "dx_pc": float(source_cell["dx_pc"]),
    }
    bridge_config = mechanics.bridge.load_config()
    hstar_pc = float(source_cell["summary"]["hstar_pc"])
    source_ratio = 4.0 / 29.2
    exponential_scale_pc = hstar_pc / source_ratio
    solver_config = {
        "grid_contract": config["grid_contract"],
        "source_cell": {
            "stellar_height_over_exponential_scale": source_ratio,
            "gas_height_pc": float(source_cell["summary"]["hgas_pc"]),
        },
    }
    built = mechanics._build_density(
        solver_config,
        bridge_config,
        maps,
        exponential_scale_pc=exponential_scale_pc,
        nodes=nodes,
    )
    grid = built["grid"]
    rhs = 4.0 * math.pi * built["density_dimensionless"]
    newton, newton_residual = mechanics.solve_poisson_dst(
        rhs, built["newton_boundary"], grid.spacing
    )
    rg_parameters = config["candidate_contract"]["refracted_gravity_parameters"]
    epsilon = mechanics.rg.published_permittivity(
        built["density_g_cm3"],
        epsilon_0=float(rg_parameters["epsilon_0"]),
        rho_c=10.0 ** float(rg_parameters["log10_rho_c_g_cm3"]),
        q_slope=float(rg_parameters["Q"]),
    )
    rg_boundary = built["newton_boundary"] / float(rg_parameters["epsilon_0"])
    initial = newton / float(rg_parameters["epsilon_0"])
    operator = config["operator_contract"]
    refracted, rg_metrics = mechanics.solve_variable_pcg(
        rhs,
        rg_boundary,
        epsilon,
        grid.spacing,
        relative_tolerance=float(operator["pcg_relative_tolerance"]),
        absolute_tolerance=float(operator["pcg_absolute_tolerance"]),
        max_iterations=int(operator["pcg_max_iterations"]),
        initial_potential=initial,
    )
    profile_options = {
        "half_box_kpc": float(config["grid_contract"]["solver_half_box_kpc"]),
        "radii_kpc": _radii(config),
        "azimuth_samples": int(config["grid_contract"]["azimuth_samples"]),
        "a0_m_s2": float(config["candidate_contract"]["a0_m_s2"]),
    }
    newton_profile = mechanics.midplane_radial_profile(newton, grid, **profile_options)
    rg_profile = mechanics.midplane_radial_profile(refracted, grid, **profile_options)
    profiles = _candidate_profiles(config, newton_profile, rg_profile)
    output = {
        "nodes_per_axis": nodes,
        "spacing_kpc": grid.spacing * float(config["grid_contract"]["solver_half_box_kpc"]),
        "source_masses_msun": built["masses"],
        "total_mass_msun": built["total_mass_msun"],
        "dimensionless_mass_relative_error": built["dimensionless_mass_relative_error"],
        "solver_metrics": {
            "newton_relative_residual": newton_residual,
            "refracted_gravity": rg_metrics,
        },
        "profiles": profiles,
        "field_hashes": {
            "density_dimensionless": mechanics.array_sha256(built["density_dimensionless"]),
            "epsilon": mechanics.array_sha256(epsilon),
            "newton_potential": mechanics.array_sha256(newton),
            "refracted_gravity_potential": mechanics.array_sha256(refracted),
        },
    }
    del rhs, newton, epsilon, refracted, built, grid, initial, rg_boundary
    gc.collect()
    return output


def _numerical_mask(
    config: Mapping[str, Any], fine: Mapping[str, Any], convergence: Mapping[str, Any]
) -> dict[str, Any]:
    gate = config["numerical_gate"]
    candidate_ids = config["candidate_contract"]["candidate_ids"]
    threshold = float(gate["fine_vs_convergence_relative_difference_max"])
    rows: list[dict[str, Any]] = []
    for index, radius in enumerate(_radii(config)):
        relative: dict[str, float] = {}
        for candidate_id in candidate_ids:
            first = float(fine["profiles"][candidate_id][index]["radial_acceleration_m_s2"])
            second = float(convergence["profiles"][candidate_id][index]["radial_acceleration_m_s2"])
            relative[candidate_id] = abs(first - second) / max(abs(first), abs(second), 1.0e-30)
        enough_cells = radius / float(config["grid_contract"]["fine_spacing_kpc"]) >= float(
            config["grid_contract"]["minimum_fine_cells_per_radius"]
        )
        rows.append(
            {
                "radius_kpc": radius,
                "relative_differences": relative,
                "eligible": enough_cells and all(value <= threshold for value in relative.values()),
            }
        )
    global_gates = {
        "fine_newton_residual": float(fine["solver_metrics"]["newton_relative_residual"])
        <= float(gate["newton_relative_residual_max"]),
        "convergence_newton_residual": float(
            convergence["solver_metrics"]["newton_relative_residual"]
        )
        <= float(gate["newton_relative_residual_max"]),
        "fine_rg_residual": float(fine["solver_metrics"]["refracted_gravity"]["relative_residual"])
        <= float(gate["rg_relative_residual_max"]),
        "convergence_rg_residual": float(
            convergence["solver_metrics"]["refracted_gravity"]["relative_residual"]
        )
        <= float(gate["rg_relative_residual_max"]),
        "fine_source_mass": float(fine["dimensionless_mass_relative_error"])
        <= float(gate["source_mass_relative_error_max"]),
        "convergence_source_mass": float(convergence["dimensionless_mass_relative_error"])
        <= float(gate["source_mass_relative_error_max"]),
    }
    if not all(global_gates.values()):
        for row in rows:
            row["eligible"] = False
    return {
        "global_gates": global_gates,
        "all_global_gates_pass": all(global_gates.values()),
        "eligible_radius_count": sum(row["eligible"] for row in rows),
        "failed_radius_count": sum(not row["eligible"] for row in rows),
        "rows": rows,
    }


def build_cell(config: Mapping[str, Any], source_cell: Mapping[str, Any]) -> dict[str, Any]:
    predecessors = _load_predecessors(config)
    arrays = _load_source_arrays(predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"], source_cell)
    fine = _solve_grid(
        config,
        arrays,
        source_cell,
        nodes=int(config["grid_contract"]["fine_nodes_per_axis"]),
    )
    convergence = _solve_grid(
        config,
        arrays,
        source_cell,
        nodes=int(config["grid_contract"]["convergence_nodes_per_axis"]),
    )
    payload: dict[str, Any] = {
        "schema": _CELL_SCHEMA,
        "package_id": config["package_id"],
        "package_bindings": _package_bindings(),
        "cell_run_id": cell_run_id(source_cell),
        "object_id": source_cell["object_id"],
        "conversion_cell_id": source_cell["conversion_cell_id"],
        "geometry": source_cell["geometry"],
        "source_profile_sha256": source_cell["profile_sha256"],
        "fine": fine,
        "convergence": convergence,
        "numerical_mask": _numerical_mask(config, fine, convergence),
        "response_boundary": config["response_boundary"],
    }
    payload["content_sha256"] = content_sha256(payload)
    validate_cell(config, source_cell, payload)
    return payload


def validate_cell(
    config: Mapping[str, Any], source_cell: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    _require(payload["schema"] == _CELL_SCHEMA, "cell schema changed")
    _require(payload["package_id"] == config["package_id"], "cell package changed")
    _require(payload["package_bindings"] == _package_bindings(), "cell package seal changed")
    _require(payload["cell_run_id"] == cell_run_id(source_cell), "cell ID changed")
    _require(payload["object_id"] == source_cell["object_id"], "cell object changed")
    _require(
        payload["conversion_cell_id"] == source_cell["conversion_cell_id"],
        "cell conversion changed",
    )
    _require(payload["geometry"] == source_cell["geometry"], "cell geometry changed")
    _require(
        payload["source_profile_sha256"] == source_cell["profile_sha256"],
        "cell source changed",
    )
    _require(payload["response_boundary"] == config["response_boundary"], "cell response leak")
    _require(
        len(payload["fine"]["profiles"]) == len(config["candidate_contract"]["candidate_ids"]),
        "cell candidate count changed",
    )
    for grid_key in ("fine", "convergence"):
        _require(
            set(payload[grid_key]["profiles"])
            == set(config["candidate_contract"]["candidate_ids"]),
            "cell candidate inventory changed",
        )
        for rows in payload[grid_key]["profiles"].values():
            _require(len(rows) == int(config["grid_contract"]["radial_points"]), "profile changed")
    copy = dict(payload)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "cell content hash changed")


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


def write_cell(cell_id: str) -> str:
    config = load_config()
    source_receipt = _load_predecessors(config)["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    source_cell = next(
        (row for row in _built_source_cells(source_receipt) if cell_run_id(row) == cell_id),
        None,
    )
    _require(source_cell is not None, "unknown built source cell")
    payload = build_cell(config, source_cell)
    return _atomic_no_clobber(_cell_output_path(config, source_cell), canonical_bytes(payload))


def _load_completed_cells(
    config: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for source_cell in _built_source_cells(source_receipt):
        path = _cell_output_path(config, source_cell)
        if not path.is_file():
            continue
        payload = _read_json(path, "prediction cell")
        validate_cell(config, source_cell, payload)
        completed.append(
            {
                "cell_run_id": cell_run_id(source_cell),
                "relative_path": path.relative_to(_ROOT).as_posix(),
                "file_sha256": file_sha256(path),
                "content_sha256": payload["content_sha256"],
                "all_global_gates_pass": payload["numerical_mask"]["all_global_gates_pass"],
                "eligible_radius_count": payload["numerical_mask"]["eligible_radius_count"],
                "failed_radius_count": payload["numerical_mask"]["failed_radius_count"],
            }
        )
    return completed


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessors = _load_predecessors(config)
    source = predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    completed = _load_completed_cells(config, source)
    completed_ids = {row["cell_run_id"] for row in completed}
    expected_ids = {cell_run_id(row) for row in _built_source_cells(source)}
    missing = sorted(expected_ids - completed_ids)
    status = (
        "PASS_RESPONSE_BLIND_ALL_CELL_PREDICTIONS_BUILT"
        if not missing
        else "IN_PROGRESS_RESPONSE_BLIND_CELL_PREDICTIONS"
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": status,
        "package_bindings": {
            **_package_bindings(),
        },
        "candidate_ids": config["candidate_contract"]["candidate_ids"],
        "declared_source_cells": source["source_cell_count"],
        "built_source_cells": source["built_source_map_count"],
        "retained_source_failures": source["failed_source_conversion_count"],
        "completed_prediction_cells": len(completed),
        "missing_prediction_cells": missing,
        "cell_artifacts": completed,
        "field_solver_runs_completed": len(completed) * 4,
        "response_boundary": config["response_boundary"],
        "claim_boundary": {
            **config["claim_boundary"],
            "real_source_predictions_generated": not missing,
            "response_blind_numerical_radius_masks_generated": not missing,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    validate_receipt(config, receipt)
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(receipt["schema"] == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["package_id"] == config["package_id"], "receipt package changed")
    _require(receipt["package_bindings"] == _package_bindings(), "receipt package seal changed")
    _require(receipt["response_boundary"] == config["response_boundary"], "receipt response leak")
    _require(receipt["declared_source_cells"] == 27, "receipt source count changed")
    _require(receipt["built_source_cells"] == 24, "receipt built count changed")
    _require(receipt["retained_source_failures"] == 3, "receipt failures changed")
    _require(
        receipt["field_solver_runs_completed"] == receipt["completed_prediction_cells"] * 4,
        "receipt solver accounting changed",
    )
    copy = dict(receipt)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "receipt content hash changed")


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    _require(not receipt["missing_prediction_cells"], "prediction cells remain incomplete")
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    stored = _read_json(path, "receipt")
    validate_receipt(config, stored)
    _require(stored == build_receipt(config), "receipt rebuild differs")
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    receipt = build_receipt(config)
    return {
        "status": receipt["status"],
        "completed_prediction_cells": receipt["completed_prediction_cells"],
        "missing_prediction_cells": len(receipt["missing_prediction_cells"]),
        "response_boundary": receipt["response_boundary"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("build-cell")
    cell.add_argument("cell_id")
    subparsers.add_parser("write")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-cell":
        print(write_cell(args.cell_id))
    elif args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
