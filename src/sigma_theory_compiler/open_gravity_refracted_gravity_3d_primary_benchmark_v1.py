"""Primary-paper-bound target-free 3-D Refracted Gravity benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.sparse.linalg import spsolve

from sigma_theory_compiler import open_gravity_3d_halo_modified_gravity_comparators_v1 as prior
from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base

CONFIG_PATH = Path("configs/open_gravity_refracted_gravity_3d_primary_benchmark_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_refracted_gravity_3d_primary_benchmark_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_refracted_gravity_3d_primary_benchmark_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-3d-primary-benchmark-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-refracted-gravity-3d-primary-benchmark-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-refracted-gravity-3d-primary-benchmark-receipt-1.0"
_CONFIG_RAW_SHA256 = "9f5f127b8078ef95ff47ce073cb0e2c9254ca1fd9c8113880ac8e42ea52a8308"
_CONFIG_CONTENT_SHA256 = "06dba57b358a8b9d2527870b1884b8da20817b44838289bf0392229ea7c1e39d"
_MODULE_SEMANTIC_SHA256 = "d80095bf56c5e185560a0f7beebc28ce5eb318a05cc61123b6e4c804408534da"
_TEST_RAW_SHA256 = "fe49a5a4de7bd5fc341dfdd4aab2283498dfc746ca066e6cb8b0caa16b932971"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_REQUIRED_GATES = (
    "PUBLISHED_FORMULA_IDENTITY",
    "PUBLISHED_PHYSICAL_PARAMETER_ANCHOR",
    "PERMITTIVITY_BOUNDS_MONOTONICITY_AND_DERIVATIVE",
    "LOW_HIGH_DENSITY_AND_TRANSITION_LIMITS",
    "NEWTON_EPSILON_ONE_LIMIT",
    "CONSTANT_EPSILON_FIELD_SCALE",
    "VARIABLE_COEFFICIENT_MANUFACTURED_SECOND_ORDER",
    "SPHERICAL_GAUSS_LAW_SECOND_ORDER",
    "NONSPHERICAL_DENSITY_GRADIENT_TERM_ACTIVE",
    "ROTATION_COVARIANCE",
    "POSITIVE_ELLIPTICITY_PUBLISHED_SENSITIVITY_GRID",
    "PUBLISHED_COUNTEREVIDENCE_RETAINED",
    "DESIGNED_INVALID_PARAMETER_FAILURES",
    "ZERO_RESPONSE_ACCESS",
)


class RefractedGravityBenchmarkError(RuntimeError):
    """Raised when the primary-source or numerical benchmark fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefractedGravityBenchmarkError(message)


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
        raise RefractedGravityBenchmarkError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _git_show(commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RefractedGravityBenchmarkError("committed predecessor unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_PRIMARY_PAPER_BOUND_TARGET_FREE_BENCHMARK",
        "status changed",
    )
    _require(len(config["primary_sources"]) == 3, "primary-source inventory changed")
    _require(
        [row["id"] for row in config["primary_sources"]]
        == ["RG_ORIGINAL_2016", "RG_DISKMASS_2021_V2", "RG_REVIEW_2024"],
        "primary-source identities changed",
    )
    _require(tuple(config["required_gates"]) == _REQUIRED_GATES, "gate contract changed")
    parameters = config["published_parameters"]
    median = parameters["universal_diskmass_median"]
    _require(median["epsilon_0"] == 0.661, "published epsilon_0 changed")
    _require(median["Q"] == 1.79, "published Q changed")
    _require(median["log10_rho_c_g_cm3"] == -24.54, "published rho_c changed")
    _require(
        parameters["sensitivity_cells"]["count"] == 9,
        "published sensitivity count changed",
    )
    benchmark = config["benchmark_contract"]
    _require(benchmark["manufactured_grids"] == [9, 13, 17], "manufactured grids changed")
    _require(benchmark["spherical_grids"] == [13, 17, 21], "spherical grids changed")
    _require(benchmark["response_tuning"] is False, "response tuning enabled")
    _require(benchmark["retain_all_failures"] is True, "failure retention removed")
    access = config["access_contract"]
    _require(access["paper_metadata_records_frozen"] == 3, "paper accounting changed")
    for key, value in access.items():
        if key != "paper_metadata_records_frozen":
            _require(value == 0, f"nonzero forbidden access: {key}")
    claims = config["claim_boundary"]
    _require(claims["published_non_covariant_operator_implemented"] is True, "operator lost")
    _require(claims["target_free_3d_benchmark_attempted"] is True, "benchmark lost")
    for key, value in claims.items():
        if key not in {
            "published_non_covariant_operator_implemented",
            "target_free_3d_benchmark_attempted",
        }:
            _require(value is False, f"claim ceiling exceeded: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _validate_package_files() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module semantics changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "benchmark config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessor(config: Mapping[str, Any]) -> None:
    binding = config["predecessor_binding"]
    commit = binding["commit"]
    _require(type(commit) is str and len(commit) == 40, "predecessor commit changed")
    for artifact in binding["artifacts"]:
        relative = artifact["path"]
        expected = artifact["sha256"]
        current = _repo_path(relative)
        _require(current.is_file(), "predecessor artifact missing")
        _require(file_sha256(current) == expected, "predecessor worktree artifact changed")
        _require(
            hashlib.sha256(_git_show(commit, relative)).hexdigest() == expected,
            "predecessor commit artifact changed",
        )
    receipt_artifact = next(
        row for row in binding["artifacts"] if row["path"].endswith("receipt.json")
    )
    receipt = _read_json(_repo_path(receipt_artifact["path"]), "predecessor receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"],
        "predecessor receipt content changed",
    )


def published_permittivity(
    density: np.ndarray | float,
    *,
    epsilon_0: float,
    rho_c: float,
    q_slope: float,
) -> np.ndarray:
    _require(type(epsilon_0) is float and 0.0 < epsilon_0 <= 1.0, "invalid epsilon_0")
    _require(type(rho_c) is float and rho_c > 0.0 and math.isfinite(rho_c), "invalid rho_c")
    _require(
        type(q_slope) is float and q_slope > 0.0 and math.isfinite(q_slope),
        "invalid Q",
    )
    values = np.asarray(density, dtype=np.float64)
    _require(np.all(np.isfinite(values)) and np.all(values >= 0.0), "invalid density")
    safe = np.maximum(values, np.finfo(np.float64).tiny)
    return epsilon_0 + (1.0 - epsilon_0) * (1.0 + np.tanh(q_slope * np.log(safe / rho_c))) / 2.0


def permittivity_derivative_log_density(
    log_density_ratio: np.ndarray | float, *, epsilon_0: float, q_slope: float
) -> np.ndarray:
    _require(type(epsilon_0) is float and 0.0 < epsilon_0 <= 1.0, "invalid epsilon_0")
    _require(
        type(q_slope) is float and q_slope > 0.0 and math.isfinite(q_slope),
        "invalid Q",
    )
    value = np.asarray(log_density_ratio, dtype=np.float64)
    _require(np.all(np.isfinite(value)), "invalid log density")
    tanh = np.tanh(q_slope * value)
    return 0.5 * (1.0 - epsilon_0) * q_slope * (1.0 - tanh * tanh)


def published_parameter_cells(config: Mapping[str, Any]) -> list[dict[str, float | str]]:
    parameters = config["published_parameters"]
    median = parameters["universal_diskmass_median"]
    cells: list[dict[str, float | str]] = [
        {
            "id": "DISKMASS_UNIVERSAL_MEDIAN",
            "epsilon_0": float(median["epsilon_0"]),
            "Q": float(median["Q"]),
            "log10_rho_c_g_cm3": float(median["log10_rho_c_g_cm3"]),
        }
    ]
    bounds = parameters["diskmass_flat_prior_bounds"]
    for epsilon_0, q_slope, log_rho in itertools.product(
        bounds["epsilon_0"], bounds["Q"], bounds["log10_rho_c_g_cm3"]
    ):
        cells.append(
            {
                "id": f"PRIOR_CORNER_E{epsilon_0:g}_Q{q_slope:g}_R{log_rho:g}",
                "epsilon_0": float(epsilon_0),
                "Q": float(q_slope),
                "log10_rho_c_g_cm3": float(log_rho),
            }
        )
    _require(len(cells) == 9, "published parameter-cell count changed")
    return cells


def _solve_variable(
    rhs: np.ndarray, boundary: np.ndarray, coefficient: np.ndarray, spacing: float
) -> tuple[np.ndarray, float]:
    _require(rhs.shape == boundary.shape == coefficient.shape, "solver shape mismatch")
    _require(rhs.ndim == 3 and min(rhs.shape) >= 5, "invalid solver grid")
    _require(spacing > 0.0 and math.isfinite(spacing), "invalid solver spacing")
    _require(np.all(np.isfinite(rhs)), "invalid solver rhs")
    _require(np.all(np.isfinite(boundary)), "invalid solver boundary")
    _require(np.all(np.isfinite(coefficient)) and np.all(coefficient > 0.0), "invalid coefficient")
    matrix, vector = base._variable_matrix_and_rhs(rhs, boundary, coefficient, spacing)
    interior = np.asarray(spsolve(matrix, vector), dtype=np.float64)
    potential = np.asarray(boundary, dtype=np.float64).copy()
    potential[1:-1, 1:-1, 1:-1] = interior.reshape(
        potential.shape[0] - 2,
        potential.shape[1] - 2,
        potential.shape[2] - 2,
    )
    residual = base._variable_divergence(potential, coefficient, spacing) - rhs
    scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return potential, relative


def solve_published_refracted_gravity(
    density: np.ndarray,
    boundary: np.ndarray,
    spacing: float,
    *,
    epsilon_0: float,
    rho_c: float,
    q_slope: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    coefficient = published_permittivity(
        density,
        epsilon_0=epsilon_0,
        rho_c=rho_c,
        q_slope=q_slope,
    )
    potential, residual = _solve_variable(
        4.0 * math.pi * np.asarray(density, dtype=np.float64),
        boundary,
        coefficient,
        spacing,
    )
    return potential, coefficient, residual


def _manufactured_case(nodes: int, *, epsilon_0: float, q_slope: float) -> dict[str, float]:
    grid = base.make_grid(nodes, 1.0)
    x, y, z = grid.x, grid.y, grid.z
    a, b, c = 0.4, 0.2, -0.3
    log_ratio = a * x + b * y + c * z
    coefficient = epsilon_0 + (1.0 - epsilon_0) * (1.0 + np.tanh(q_slope * log_ratio)) / 2.0
    derivative = permittivity_derivative_log_density(
        log_ratio,
        epsilon_0=epsilon_0,
        q_slope=q_slope,
    )
    phi = (1.0 - x * x) * (1.0 - y * y) * (1.0 - z * z)
    grad_x = -2.0 * x * (1.0 - y * y) * (1.0 - z * z)
    grad_y = -2.0 * y * (1.0 - x * x) * (1.0 - z * z)
    grad_z = -2.0 * z * (1.0 - x * x) * (1.0 - y * y)
    laplacian = -2.0 * (
        (1.0 - y * y) * (1.0 - z * z)
        + (1.0 - x * x) * (1.0 - z * z)
        + (1.0 - x * x) * (1.0 - y * y)
    )
    rhs = coefficient * laplacian + derivative * (a * grad_x + b * grad_y + c * grad_z)
    boundary = phi.copy()
    boundary[1:-1, 1:-1, 1:-1] = 0.0
    solved, residual = _solve_variable(rhs, boundary, coefficient, grid.spacing)
    return {
        "nodes": float(nodes),
        "spacing": float(grid.spacing),
        "maximum_solution_error": float(np.max(np.abs(solved - phi))),
        "relative_discrete_residual": residual,
    }


def _spherical_case(nodes: int, *, epsilon_0: float, q_slope: float) -> dict[str, float]:
    grid = base.make_grid(nodes, 1.0)
    radius = np.sqrt(grid.x * grid.x + grid.y * grid.y + grid.z * grid.z)
    sigma = 0.42
    rho_c = 0.15
    density = np.exp(-((radius / sigma) ** 2))
    coefficient = published_permittivity(
        density,
        epsilon_0=epsilon_0,
        rho_c=rho_c,
        q_slope=q_slope,
    )
    radial_grid = np.linspace(0.0, math.sqrt(3.0) + 0.01, 60001)
    radial_density = np.exp(-((radial_grid / sigma) ** 2))
    enclosed_mass = (
        4.0
        * math.pi
        * cumulative_trapezoid(
            radial_density * radial_grid * radial_grid,
            radial_grid,
            initial=0.0,
        )
    )
    radial_epsilon = published_permittivity(
        radial_density,
        epsilon_0=epsilon_0,
        rho_c=rho_c,
        q_slope=q_slope,
    )
    radial_field = np.zeros_like(radial_grid)
    radial_field[1:] = enclosed_mass[1:] / (radial_epsilon[1:] * radial_grid[1:] * radial_grid[1:])
    integral = cumulative_trapezoid(radial_field, radial_grid, initial=0.0)
    radial_potential = integral - integral[-1]
    exact = np.interp(radius.reshape(-1), radial_grid, radial_potential).reshape(radius.shape)
    boundary = exact.copy()
    boundary[1:-1, 1:-1, 1:-1] = 0.0
    solved, residual = _solve_variable(
        4.0 * math.pi * density,
        boundary,
        coefficient,
        grid.spacing,
    )
    scale = max(float(np.max(np.abs(exact))), 1.0e-12)
    return {
        "nodes": float(nodes),
        "spacing": float(grid.spacing),
        "relative_potential_error": float(np.max(np.abs(solved - exact)) / scale),
        "relative_discrete_residual": residual,
    }


def _convergence_order(coarse: Mapping[str, float], fine: Mapping[str, float], key: str) -> float:
    return math.log(coarse[key] / fine[key]) / math.log(coarse["spacing"] / fine["spacing"])


def _anisotropic_density(grid: base.Grid3D) -> np.ndarray:
    density = np.exp(-0.5 * ((grid.x / 0.47) ** 2 + (grid.y / 0.24) ** 2 + (grid.z / 0.13) ** 2))
    return density / (float(np.sum(density)) * grid.spacing**3)


def _max_rotation_error(first: np.ndarray, rotated: np.ndarray) -> float:
    expected = np.rot90(first, axes=(0, 1))
    scale = max(float(np.max(np.abs(expected))), 1.0e-12)
    return float(np.max(np.abs(expected - rotated)) / scale)


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    validate_predecessor(config)
    median = config["published_parameters"]["universal_diskmass_median"]
    epsilon_0 = float(median["epsilon_0"])
    q_slope = float(median["Q"])
    physical_rho_c = 10.0 ** float(median["log10_rho_c_g_cm3"])
    gates: dict[str, dict[str, Any]] = {}

    comparison_density = np.geomspace(1.0e-12, 1.0e12, 257)
    ours = published_permittivity(
        comparison_density,
        epsilon_0=epsilon_0,
        rho_c=1.0,
        q_slope=q_slope,
    )
    predecessor = prior.permittivity(
        comparison_density,
        epsilon_0=epsilon_0,
        rho_c=1.0,
        q_slope=q_slope,
    )
    gates["PUBLISHED_FORMULA_IDENTITY"] = {
        "passed": bool(np.array_equal(ours, predecessor)),
        "metrics": {
            "maximum_absolute_difference_from_bound_predecessor": float(
                np.max(np.abs(ours - predecessor))
            ),
            "equation_source": "RG_DISKMASS_2021_V2_EQ_5",
        },
    }

    grams_per_cm3_to_kg_per_m3 = 1000.0
    kiloparsec_m = 3.085677581491367e19
    solar_mass_kg = 1.98847e30
    rho_c_kg_m3 = physical_rho_c * grams_per_cm3_to_kg_per_m3
    rho_c_msun_kpc3 = rho_c_kg_m3 * kiloparsec_m**3 / solar_mass_kg
    bounds = config["published_parameters"]["diskmass_flat_prior_bounds"]
    anchor_passed = (
        bounds["epsilon_0"][0] <= epsilon_0 <= bounds["epsilon_0"][1]
        and bounds["Q"][0] <= q_slope <= bounds["Q"][1]
        and bounds["log10_rho_c_g_cm3"][0]
        <= median["log10_rho_c_g_cm3"]
        <= bounds["log10_rho_c_g_cm3"][1]
    )
    gates["PUBLISHED_PHYSICAL_PARAMETER_ANCHOR"] = {
        "passed": anchor_passed,
        "metrics": {
            "epsilon_0": epsilon_0,
            "Q": q_slope,
            "rho_c_g_cm3": physical_rho_c,
            "rho_c_kg_m3": rho_c_kg_m3,
            "rho_c_msun_kpc3": rho_c_msun_kpc3,
            "sensitivity_cells": len(published_parameter_cells(config)),
        },
    }

    log_ratios = np.linspace(-6.0, 6.0, 1201)
    density_ratios = np.exp(log_ratios)
    epsilon = published_permittivity(
        density_ratios,
        epsilon_0=epsilon_0,
        rho_c=1.0,
        q_slope=q_slope,
    )
    derivative = permittivity_derivative_log_density(
        log_ratios,
        epsilon_0=epsilon_0,
        q_slope=q_slope,
    )
    probe = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    step = 1.0e-6
    plus = published_permittivity(
        np.exp(probe + step), epsilon_0=epsilon_0, rho_c=1.0, q_slope=q_slope
    )
    minus = published_permittivity(
        np.exp(probe - step), epsilon_0=epsilon_0, rho_c=1.0, q_slope=q_slope
    )
    finite_difference = (plus - minus) / (2.0 * step)
    exact_derivative = permittivity_derivative_log_density(
        probe, epsilon_0=epsilon_0, q_slope=q_slope
    )
    derivative_error = float(
        np.max(np.abs(finite_difference - exact_derivative))
        / max(float(np.max(np.abs(exact_derivative))), 1.0e-15)
    )
    gates["PERMITTIVITY_BOUNDS_MONOTONICITY_AND_DERIVATIVE"] = {
        "passed": bool(
            np.all(epsilon >= epsilon_0)
            and np.all(epsilon <= 1.0)
            and np.all(np.diff(epsilon) >= 0.0)
            and np.all(derivative >= 0.0)
            and derivative_error < 1.0e-8
        ),
        "metrics": {
            "minimum_epsilon": float(np.min(epsilon)),
            "maximum_epsilon": float(np.max(epsilon)),
            "minimum_log_density_derivative": float(np.min(derivative)),
            "finite_difference_relative_error": derivative_error,
        },
    }

    limits = published_permittivity(
        np.array([1.0e-12, 1.0, 1.0e12]),
        epsilon_0=epsilon_0,
        rho_c=1.0,
        q_slope=q_slope,
    )
    transition_expected = 0.5 * (1.0 + epsilon_0)
    gates["LOW_HIGH_DENSITY_AND_TRANSITION_LIMITS"] = {
        "passed": bool(
            abs(float(limits[0]) - epsilon_0) < 1.0e-14
            and abs(float(limits[1]) - transition_expected) < 1.0e-15
            and abs(float(limits[2]) - 1.0) < 1.0e-14
        ),
        "metrics": {
            "low_density_epsilon": float(limits[0]),
            "transition_epsilon": float(limits[1]),
            "high_density_epsilon": float(limits[2]),
        },
    }

    grid = base.make_grid(11, 1.0)
    density = _anisotropic_density(grid)
    zero = np.zeros(grid.shape, dtype=np.float64)
    newton = base.solve_poisson(4.0 * math.pi * density, zero, grid.spacing)
    rg_newton, coefficient_one, residual_one = solve_published_refracted_gravity(
        density,
        zero,
        grid.spacing,
        epsilon_0=1.0,
        rho_c=0.2,
        q_slope=q_slope,
    )
    newton_error = float(
        np.max(np.abs(rg_newton - newton.potential))
        / max(float(np.max(np.abs(newton.potential))), 1.0e-12)
    )
    gates["NEWTON_EPSILON_ONE_LIMIT"] = {
        "passed": bool(
            np.array_equal(coefficient_one, np.ones_like(coefficient_one))
            and newton_error < 1.0e-13
            and residual_one < 1.0e-11
        ),
        "metrics": {
            "relative_potential_error": newton_error,
            "relative_discrete_residual": residual_one,
        },
    }

    constant_epsilon = 0.4
    constant_solution, constant_residual = _solve_variable(
        4.0 * math.pi * density,
        zero,
        np.full(grid.shape, constant_epsilon),
        grid.spacing,
    )
    constant_expected = newton.potential / constant_epsilon
    constant_error = float(
        np.max(np.abs(constant_solution - constant_expected))
        / max(float(np.max(np.abs(constant_expected))), 1.0e-12)
    )
    gates["CONSTANT_EPSILON_FIELD_SCALE"] = {
        "passed": constant_error < 1.0e-12 and constant_residual < 1.0e-11,
        "metrics": {
            "epsilon": constant_epsilon,
            "field_enhancement": 1.0 / constant_epsilon,
            "relative_potential_error": constant_error,
            "relative_discrete_residual": constant_residual,
        },
    }

    manufactured = [
        _manufactured_case(nodes, epsilon_0=epsilon_0, q_slope=q_slope)
        for nodes in config["benchmark_contract"]["manufactured_grids"]
    ]
    manufactured_order = _convergence_order(
        manufactured[0], manufactured[-1], "maximum_solution_error"
    )
    gates["VARIABLE_COEFFICIENT_MANUFACTURED_SECOND_ORDER"] = {
        "passed": bool(
            all(
                manufactured[index]["maximum_solution_error"]
                > manufactured[index + 1]["maximum_solution_error"]
                for index in range(len(manufactured) - 1)
            )
            and manufactured_order >= config["benchmark_contract"]["required_order"]
            and manufactured[-1]["maximum_solution_error"]
            <= config["benchmark_contract"]["maximum_finest_manufactured_error"]
            and max(row["relative_discrete_residual"] for row in manufactured) < 1.0e-11
        ),
        "metrics": {"grids": manufactured, "observed_order": manufactured_order},
    }

    spherical = [
        _spherical_case(nodes, epsilon_0=epsilon_0, q_slope=q_slope)
        for nodes in config["benchmark_contract"]["spherical_grids"]
    ]
    spherical_order = _convergence_order(spherical[0], spherical[-1], "relative_potential_error")
    gates["SPHERICAL_GAUSS_LAW_SECOND_ORDER"] = {
        "passed": bool(
            all(
                spherical[index]["relative_potential_error"]
                > spherical[index + 1]["relative_potential_error"]
                for index in range(len(spherical) - 1)
            )
            and spherical_order >= config["benchmark_contract"]["required_order"]
            and spherical[-1]["relative_potential_error"]
            <= config["benchmark_contract"]["maximum_finest_spherical_potential_error"]
            and max(row["relative_discrete_residual"] for row in spherical) < 1.0e-10
        ),
        "metrics": {"grids": spherical, "observed_order": spherical_order},
    }

    rg, rg_epsilon, rg_residual = solve_published_refracted_gravity(
        density,
        zero,
        grid.spacing,
        epsilon_0=epsilon_0,
        rho_c=0.2,
        q_slope=q_slope,
    )
    epsilon_gradient = np.gradient(rg_epsilon, grid.spacing, edge_order=2)
    potential_gradient = np.gradient(rg, grid.spacing, edge_order=2)
    gradient_term = sum(epsilon_gradient[index] * potential_gradient[index] for index in range(3))
    local_rescale = newton.potential / rg_epsilon
    local_difference = float(
        np.max(np.abs(rg - local_rescale)) / max(float(np.max(np.abs(rg))), 1.0e-12)
    )
    gradient_term_max = float(np.max(np.abs(gradient_term[1:-1, 1:-1, 1:-1])))
    gates["NONSPHERICAL_DENSITY_GRADIENT_TERM_ACTIVE"] = {
        "passed": bool(
            rg_residual < 1.0e-11 and gradient_term_max > 1.0e-4 and local_difference > 1.0e-3
        ),
        "metrics": {
            "maximum_abs_grad_epsilon_dot_grad_phi": gradient_term_max,
            "relative_difference_from_pointwise_newton_rescale": local_difference,
            "relative_discrete_residual": rg_residual,
        },
    }

    rotated_density = np.rot90(density, axes=(0, 1)).copy()
    rotated_rg, _, rotated_residual = solve_published_refracted_gravity(
        rotated_density,
        zero,
        grid.spacing,
        epsilon_0=epsilon_0,
        rho_c=0.2,
        q_slope=q_slope,
    )
    rotation_error = _max_rotation_error(rg, rotated_rg)
    gates["ROTATION_COVARIANCE"] = {
        "passed": rotation_error < 1.0e-12 and rotated_residual < 1.0e-11,
        "metrics": {
            "relative_rotation_error": rotation_error,
            "rotated_relative_discrete_residual": rotated_residual,
        },
    }

    physical_density_grid = np.geomspace(1.0e-30, 1.0e-20, 401)
    cell_metrics: list[dict[str, Any]] = []
    all_positive = True
    for cell in published_parameter_cells(config):
        cell_epsilon = published_permittivity(
            physical_density_grid,
            epsilon_0=float(cell["epsilon_0"]),
            rho_c=10.0 ** float(cell["log10_rho_c_g_cm3"]),
            q_slope=float(cell["Q"]),
        )
        positive = bool(
            np.all(cell_epsilon > 0.0)
            and np.all(cell_epsilon <= 1.0)
            and np.all(np.diff(cell_epsilon) >= -1.0e-15)
        )
        all_positive = all_positive and positive
        cell_metrics.append(
            {
                "id": cell["id"],
                "minimum_epsilon": float(np.min(cell_epsilon)),
                "maximum_epsilon": float(np.max(cell_epsilon)),
                "positive_elliptic": positive,
            }
        )
    gates["POSITIVE_ELLIPTICITY_PUBLISHED_SENSITIVITY_GRID"] = {
        "passed": all_positive and len(cell_metrics) == 9,
        "metrics": {"cells": cell_metrics, "count": len(cell_metrics)},
    }

    counterevidence_ids = [row["id"] for row in config["published_counterevidence"]]
    gates["PUBLISHED_COUNTEREVIDENCE_RETAINED"] = {
        "passed": counterevidence_ids
        == [
            "UNIVERSAL_PARAMETERS_WORSEN_SOME_ROTATION_CURVES",
            "LOW_ACCELERATION_RAR_UNDERESTIMATE",
            "RAR_RESIDUAL_CORRELATIONS",
        ],
        "metrics": {
            "count": len(counterevidence_ids),
            "ids": counterevidence_ids,
            "pruning_from_published_failure": False,
        },
    }

    invalid_calls = (
        lambda: published_permittivity(np.ones(1), epsilon_0=0.0, rho_c=1.0, q_slope=1.0),
        lambda: published_permittivity(np.ones(1), epsilon_0=1.1, rho_c=1.0, q_slope=1.0),
        lambda: published_permittivity(np.ones(1), epsilon_0=0.5, rho_c=0.0, q_slope=1.0),
        lambda: published_permittivity(np.ones(1), epsilon_0=0.5, rho_c=1.0, q_slope=0.0),
        lambda: published_permittivity(np.array([-1.0]), epsilon_0=0.5, rho_c=1.0, q_slope=1.0),
        lambda: published_permittivity(np.array([math.nan]), epsilon_0=0.5, rho_c=1.0, q_slope=1.0),
        lambda: _solve_variable(np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), np.ones((3, 3, 3)), 1.0),
        lambda: _solve_variable(np.zeros((5, 5, 5)), np.zeros((5, 5, 5)), np.zeros((5, 5, 5)), 1.0),
    )
    rejected = 0
    for call in invalid_calls:
        try:
            call()
        except RefractedGravityBenchmarkError:
            rejected += 1
    gates["DESIGNED_INVALID_PARAMETER_FAILURES"] = {
        "passed": rejected == len(invalid_calls),
        "metrics": {"attempted": len(invalid_calls), "rejected": rejected},
    }

    access = config["access_contract"]
    zero_keys = [key for key in access if key != "paper_metadata_records_frozen"]
    gates["ZERO_RESPONSE_ACCESS"] = {
        "passed": all(access[key] == 0 for key in zero_keys),
        "metrics": access,
    }

    _require(tuple(gates) == _REQUIRED_GATES, "implemented gate order changed")
    failed = [gate_id for gate_id, row in gates.items() if row["passed"] is not True]
    _require(not failed, f"benchmark gates failed: {failed}")
    return {"gates": gates, "passed": len(gates), "failed": 0}


def build_receipt() -> dict[str, Any]:
    config = load_config()
    suite = run_suite(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_PRIMARY_PAPER_BOUND_TARGET_FREE_3D_BENCHMARK_ZERO_RESPONSE_ACCESS",
        "bindings": {
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "sha256": _CONFIG_RAW_SHA256,
                "content_sha256": _CONFIG_CONTENT_SHA256,
            },
            "module": {
                "path": MODULE_PATH.as_posix(),
                "sha256": file_sha256(_repo_path(MODULE_PATH)),
                "semantic_sha256": _MODULE_SEMANTIC_SHA256,
            },
            "test": {"path": TEST_PATH.as_posix(), "sha256": _TEST_RAW_SHA256},
            "predecessor": config["predecessor_binding"],
        },
        "primary_sources": config["primary_sources"],
        "published_parameter_cells": published_parameter_cells(config),
        "benchmark_suite": suite,
        "published_counterevidence": config["published_counterevidence"],
        "access_accounting": config["access_contract"],
        "admission_rule": config["admission_rule"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt must be an object")
    _require(payload == build_receipt(), "receipt is not deterministic")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    path = _repo_path(OUTPUT_PATH)
    _require(path == (_ROOT / OUTPUT_PATH).resolve(), "output path changed")
    return path


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "CREATED"


def validate_receipt() -> None:
    validate_receipt_payload(_read_json(_output_path(), "benchmark receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "gates_passed": receipt["benchmark_suite"]["passed"],
                    "parameter_cells": len(receipt["published_parameter_cells"]),
                    "scientific_response_files_opened": 0,
                    "scores_computed": 0,
                    "observational_authority": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
