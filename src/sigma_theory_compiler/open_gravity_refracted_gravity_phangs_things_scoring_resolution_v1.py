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
from scipy.fft import dstn, idstn
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import LinearOperator, cg

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge
from sigma_theory_compiler import (
    open_gravity_phangs_things_full3d_source_systematics_v1 as source_systematics,
)
from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as source_builder,
)
from sigma_theory_compiler import open_gravity_refracted_gravity_3d_primary_benchmark_v1 as rg

CONFIG_PATH = Path(
    "configs/open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/"
    "open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-phangs-things-scoring-resolution-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-refracted-gravity-phangs-things-scoring-resolution-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-phangs-things-scoring-resolution-receipt-1.0"
)
_CONFIG_RAW_SHA256 = "963d7b7b5c7845917febcfbcaca1001434ef6bbd78b097af7b227607c6d9c2f4"
_CONFIG_CONTENT_SHA256 = "8637dd088e23a206cb4658239b20c269b50531cf0cb9c88036e18ba17497df04"
_MODULE_SEMANTIC_SHA256 = "34195911e209c5bd9eae979db2829c2fe72d3a8eb067fa59ab525f44e2756fe8"
_TEST_RAW_SHA256 = "c803d3e8e947fd44f61b4f31582040a6eb28c5167d4e914c266a298900f3cc7a"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class ScoringResolutionError(RuntimeError):
    """Raised when the scoring-resolution contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoringResolutionError(message)


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
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


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
        raise ScoringResolutionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["status"] == "SOURCE_ONLY_SCORING_RESOLUTION_GATE", "status changed")
    gate = config["admission_rule"]
    _require(gate["real_input_required"] is True, "real-source gate removed")
    _require(
        gate["primary_paper_or_exact_analytic_benchmark_required"] is True,
        "benchmark gate removed",
    )
    _require(gate["response_may_select_grid_boundary_parameter_or_radius"] is False, "leakage")
    _require(config["objects"] == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    source = config["source_cell"]
    _require(
        source["id"] == "ROBUST_PRIMARY:FIXED_0P6:WITH_CO:HS0.136986301369863:HG200",
        "source cell changed",
    )
    _require(source["selection_from_response"] is False, "response source selection enabled")
    grid = config["grid_contract"]
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["fine_spacing_kpc"] == 0.25, "fine spacing changed")
    _require(grid["convergence_spacing_kpc"] == 0.3125, "convergence spacing changed")
    _require(grid["radial_points"] == 291, "radial grid count changed")
    operator = config["operator_contract"]
    _require(
        operator["operators"] == ["NEWTON_3D_DST", "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"],
        "operators changed",
    )
    _require(operator["response_parameter_fitting"] is False, "response fitting enabled")
    _require(operator["epsilon_0"] == 0.661, "published epsilon0 changed")
    _require(operator["Q"] == 1.79, "published Q changed")
    _require(operator["log10_rho_c_g_cm3"] == -24.54, "published rho_c changed")
    benchmarks = config["benchmark_contract"]
    _require(benchmarks["radius_gate_is_response_blind"] is True, "radius leakage enabled")
    boundary = config["scientific_boundary"]
    for key in (
        "response_files_opened",
        "response_rows_opened",
        "response_values_opened",
        "scores_computed",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(boundary[key] == 0, f"forbidden access enabled: {key}")
    claims = config["claim_boundary"]
    for key in (
        "observational_fit_tested",
        "refracted_gravity_preferred",
        "all_225_cells_high_resolution",
        "lensing_closure_established",
        "relativistic_completion_established",
        "novelty_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim ceiling exceeded: {key}")
    _require(config["output_contract"]["receipt"] == OUTPUT_PATH.as_posix(), "output changed")


def _validate_package_files() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected_roles = [
        "REAL_SOURCE_BUILDER",
        "FULL3D_SOLVER_BRIDGE",
        "REFRACTED_GRAVITY_PRIMARY_BENCHMARK",
        "REFRACTED_GRAVITY_225_BY_9_SOURCE_SCREEN",
    ]
    _require(
        [row["role"] for row in config["predecessor_bindings"]] == expected_roles,
        "predecessor roles changed",
    )
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        _require(binding["commit"] is None, "unverified commit introduced")
        _require(binding["promotion_authority"] is False, "uncommitted authority overclaimed")
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor artifact changed")
        receipt_artifact = next(
            row for row in binding["artifacts"] if row["path"].endswith("receipt.json")
        )
        receipt = _read_json(_repo_path(receipt_artifact["path"]), "predecessor receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        receipts[binding["role"]] = receipt
    _require(
        all(receipts["REAL_SOURCE_BUILDER"]["benchmarks"]["passed"].values()),
        "source benchmark failed",
    )
    _require(
        receipts["FULL3D_SOLVER_BRIDGE"]["all_object_gates_pass"] is True,
        "3D bridge failed",
    )
    _require(
        receipts["REFRACTED_GRAVITY_PRIMARY_BENCHMARK"]["benchmark_suite"]["failed"] == 0,
        "RG primary benchmark failed",
    )
    _require(
        receipts["REFRACTED_GRAVITY_225_BY_9_SOURCE_SCREEN"]["registered_source_parameter_pairs"]
        == 2025,
        "RG source screen changed",
    )
    return receipts


def _boundary_only(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[[0, -1], :, :] = values[[0, -1], :, :]
    result[:, [0, -1], :] = values[:, [0, -1], :]
    result[:, :, [0, -1]] = values[:, :, [0, -1]]
    return result


def solve_poisson_dst(
    rhs: np.ndarray, boundary: np.ndarray, spacing: float
) -> tuple[np.ndarray, float]:
    rhs = np.asarray(rhs, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    _require(rhs.shape == boundary.shape, "Poisson shape mismatch")
    _require(rhs.ndim == 3 and len(set(rhs.shape)) == 1 and rhs.shape[0] >= 5, "bad grid")
    _require(spacing > 0.0 and math.isfinite(spacing), "bad spacing")
    _require(np.all(np.isfinite(rhs)) and np.all(np.isfinite(boundary)), "nonfinite Poisson")
    interior_nodes = rhs.shape[0] - 2
    effective = (rhs - base._constant_laplacian(boundary, spacing))[1:-1, 1:-1, 1:-1]
    modes = np.arange(1, interior_nodes + 1, dtype=np.float64)
    one_dimensional = (
        2.0 * (np.cos(math.pi * modes / (interior_nodes + 1)) - 1.0) / (spacing * spacing)
    )
    eigenvalues = (
        one_dimensional[:, None, None]
        + one_dimensional[None, :, None]
        + one_dimensional[None, None, :]
    )
    interior = idstn(
        dstn(effective, type=1, norm="ortho") / eigenvalues,
        type=1,
        norm="ortho",
    )
    potential = boundary.copy()
    potential[1:-1, 1:-1, 1:-1] = interior
    residual = base._constant_laplacian(potential, spacing) - rhs
    scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return potential, relative


def _pcg_face_coefficients(coefficient: np.ndarray) -> tuple[np.ndarray, ...]:
    centre = coefficient[1:-1, 1:-1, 1:-1]
    return (
        0.5 * (centre + coefficient[2:, 1:-1, 1:-1]),
        0.5 * (centre + coefficient[:-2, 1:-1, 1:-1]),
        0.5 * (centre + coefficient[1:-1, 2:, 1:-1]),
        0.5 * (centre + coefficient[1:-1, :-2, 1:-1]),
        0.5 * (centre + coefficient[1:-1, 1:-1, 2:]),
        0.5 * (centre + coefficient[1:-1, 1:-1, :-2]),
    )


def solve_variable_pcg(
    rhs: np.ndarray,
    boundary: np.ndarray,
    coefficient: np.ndarray,
    spacing: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    max_iterations: int,
    initial_potential: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    rhs = np.asarray(rhs, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    _require(rhs.shape == boundary.shape == coefficient.shape, "variable shape mismatch")
    _require(rhs.ndim == 3 and len(set(rhs.shape)) == 1 and rhs.shape[0] >= 5, "bad grid")
    _require(spacing > 0.0 and math.isfinite(spacing), "bad spacing")
    _require(relative_tolerance > 0.0 and absolute_tolerance >= 0.0, "bad tolerance")
    _require(max_iterations >= 1, "bad iteration ceiling")
    _require(np.all(np.isfinite(rhs)) and np.all(np.isfinite(boundary)), "nonfinite input")
    _require(
        np.all(np.isfinite(coefficient)) and np.all(coefficient > 0.0),
        "invalid coefficient",
    )
    shape = rhs.shape
    interior_shape = tuple(value - 2 for value in shape)
    unknowns = int(np.prod(interior_shape))
    faces = _pcg_face_coefficients(coefficient)
    h2 = spacing * spacing

    def matvec(vector: np.ndarray) -> np.ndarray:
        padded = np.zeros(shape, dtype=np.float64)
        padded[1:-1, 1:-1, 1:-1] = vector.reshape(interior_shape)
        centre = padded[1:-1, 1:-1, 1:-1]
        divergence = (
            faces[0] * (padded[2:, 1:-1, 1:-1] - centre)
            - faces[1] * (centre - padded[:-2, 1:-1, 1:-1])
            + faces[2] * (padded[1:-1, 2:, 1:-1] - centre)
            - faces[3] * (centre - padded[1:-1, :-2, 1:-1])
            + faces[4] * (padded[1:-1, 1:-1, 2:] - centre)
            - faces[5] * (centre - padded[1:-1, 1:-1, :-2])
        ) / h2
        return -divergence.reshape(-1)

    interior_nodes = interior_shape[0]
    modes = np.arange(1, interior_nodes + 1, dtype=np.float64)
    positive_one_dimensional = -2.0 * (np.cos(math.pi * modes / (interior_nodes + 1)) - 1.0) / h2
    preconditioner_eigenvalues = (
        positive_one_dimensional[:, None, None]
        + positive_one_dimensional[None, :, None]
        + positive_one_dimensional[None, None, :]
    )
    coefficient_scale = float(np.mean(coefficient))

    def precondition(vector: np.ndarray) -> np.ndarray:
        transformed = dstn(vector.reshape(interior_shape), type=1, norm="ortho")
        solved = idstn(
            transformed / preconditioner_eigenvalues,
            type=1,
            norm="ortho",
        )
        return (solved / coefficient_scale).reshape(-1)

    operator = LinearOperator((unknowns, unknowns), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator((unknowns, unknowns), matvec=precondition, dtype=np.float64)
    boundary_divergence = base._variable_divergence(boundary, coefficient, spacing)
    vector = (boundary_divergence[1:-1, 1:-1, 1:-1] - rhs[1:-1, 1:-1, 1:-1]).reshape(-1)
    x0 = None
    if initial_potential is not None:
        initial = np.asarray(initial_potential, dtype=np.float64)
        _require(initial.shape == shape and np.all(np.isfinite(initial)), "bad initial field")
        x0 = initial[1:-1, 1:-1, 1:-1].reshape(-1)
    iterations = [0]

    def callback(_vector: np.ndarray) -> None:
        iterations[0] += 1

    interior, info = cg(
        operator,
        vector,
        x0=x0,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        maxiter=max_iterations,
        M=preconditioner,
        callback=callback,
    )
    potential = boundary.copy()
    potential[1:-1, 1:-1, 1:-1] = interior.reshape(interior_shape)
    residual = base._variable_divergence(potential, coefficient, spacing) - rhs
    scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return potential, {
        "converged": info == 0,
        "iterations": iterations[0],
        "cg_info": int(info),
        "relative_residual": relative,
        "coefficient_minimum": float(np.min(coefficient)),
        "coefficient_maximum": float(np.max(coefficient)),
    }


def _radial_grid(config: Mapping[str, Any]) -> list[float]:
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


def midplane_radial_profile(
    potential: np.ndarray,
    grid: base.Grid3D,
    *,
    half_box_kpc: float,
    radii_kpc: Sequence[float],
    azimuth_samples: int,
    a0_m_s2: float,
) -> list[dict[str, float]]:
    _require(potential.shape == grid.shape, "profile grid mismatch")
    _require(azimuth_samples >= 16, "azimuth undersampled")
    middle = (len(grid.coordinates) - 1) // 2
    plane = potential[:, :, middle]
    grad_x, grad_y = np.gradient(plane, grid.spacing, edge_order=2)
    accel_x = -grad_x
    accel_y = -grad_y
    angles = np.linspace(0.0, 2.0 * math.pi, azimuth_samples, endpoint=False)
    rows: list[dict[str, float]] = []
    for radius_kpc in radii_kpc:
        radius_dimensionless = float(radius_kpc) / half_box_kpc
        x = radius_dimensionless * np.cos(angles)
        y = radius_dimensionless * np.sin(angles)
        i = (x - grid.coordinates[0]) / grid.spacing
        j = (y - grid.coordinates[0]) / grid.spacing
        ax = map_coordinates(accel_x, [i, j], order=1, mode="nearest", prefilter=False)
        ay = map_coordinates(accel_y, [i, j], order=1, mode="nearest", prefilter=False)
        inward = -(ax * np.cos(angles) + ay * np.sin(angles))
        rows.append(
            {
                "radius_kpc": float(radius_kpc),
                "radial_acceleration_over_a0": float(np.mean(inward)),
                "radial_acceleration_m_s2": float(np.mean(inward)) * a0_m_s2,
                "azimuthal_rms_over_a0": float(np.std(inward)),
            }
        )
    return rows


def _source_evidence() -> tuple[
    dict[str, Any], dict[str, Any], dict[tuple[str, str], Mapping[str, Any]], dict[str, Any]
]:
    source_config, acquisition, private = source_systematics._load_source_builder_evidence()
    expected = source_systematics._source_summary_index(private)
    return source_config, acquisition, expected, bridge.load_config()


def _primary_maps(
    source_config: Mapping[str, Any], acquisition: Mapping[str, Any], object_id: str
) -> tuple[dict[str, Any], float]:
    source_paths = source_builder._source_paths(acquisition)
    metadata = next(row for row in source_config["objects"] if row["object_id"] == object_id)
    images = source_builder._load_object_images(object_id, source_paths)
    maps = source_builder._surface_maps(
        source_config,
        metadata,
        images,
        n=256,
        box_kpc=40.0,
        beam="ROBUST_PRIMARY",
        use_sip=False,
    )
    half_mass_pc = source_builder._half_mass_radius_pc(
        maps["stellar_fixed"], maps["x_pc"], maps["y_pc"], float(maps["dx_pc"])
    )
    return maps, half_mass_pc / 1.678


def _build_density(
    config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    exponential_scale_pc: float,
    nodes: int,
) -> dict[str, Any]:
    half_box_kpc = float(config["grid_contract"]["solver_half_box_kpc"])
    half_box_pc = half_box_kpc * 1000.0
    grid = base.make_grid(nodes)
    coordinates_pc = grid.coordinates * half_box_pc
    spacing_pc = grid.spacing * half_box_pc
    source = config["source_cell"]
    components = (
        (
            "stellar",
            np.asarray(maps["stellar_fixed"], dtype=np.float64),
            exponential_scale_pc * float(source["stellar_height_over_exponential_scale"]),
        ),
        ("hi", np.asarray(maps["hi"], dtype=np.float64), float(source["gas_height_pc"])),
        ("co", np.asarray(maps["co"], dtype=np.float64), float(source["gas_height_pc"])),
    )
    density_msun_pc3 = np.zeros(grid.shape, dtype=np.float64)
    masses: dict[str, float] = {}
    for label, surface, height_pc in components:
        component, mass = bridge.deposit_surface_component(
            surface,
            np.asarray(maps["x_pc"], dtype=np.float64),
            np.asarray(maps["y_pc"], dtype=np.float64),
            float(maps["dx_pc"]),
            coordinates_pc,
            spacing_pc,
            height_pc,
        )
        density_msun_pc3 += component
        masses[f"{label}_mass_msun"] = mass
    normal = bridge_config["normalization_contract"]
    a0_pc = float(normal["a0_m_s2"]) * float(normal["pc_m"]) / 1.0e6
    gravity_pc = float(normal["G_pc_km2_s2_msun"])
    density_dimensionless = density_msun_pc3 * gravity_pc * half_box_pc / a0_pc
    density_g_cm3 = density_msun_pc3 * 6.768109983980884e-23
    total_mass = float(sum(masses.values()))
    expected_mass = gravity_pc * total_mass / (a0_pc * half_box_pc**2)
    deposited_mass = float(density_dimensionless.sum() * grid.spacing**3)
    newton_boundary = bridge.spherical_boundary(
        grid,
        expected_mass,
        mond=False,
        integration_samples=100000,
    )
    return {
        "grid": grid,
        "density_dimensionless": density_dimensionless,
        "density_g_cm3": density_g_cm3,
        "masses": masses,
        "total_mass_msun": total_mass,
        "dimensionless_mass_relative_error": abs(deposited_mass - expected_mass) / expected_mass,
        "newton_boundary": newton_boundary,
    }


def _solve_source_grid(
    config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    exponential_scale_pc: float,
    expected_source: Mapping[str, Any],
    nodes: int,
) -> dict[str, Any]:
    built = _build_density(
        config,
        bridge_config,
        maps,
        exponential_scale_pc=exponential_scale_pc,
        nodes=nodes,
    )
    grid = built["grid"]
    rhs = 4.0 * math.pi * built["density_dimensionless"]
    newton, newton_residual = solve_poisson_dst(rhs, built["newton_boundary"], grid.spacing)
    operator = config["operator_contract"]
    epsilon = rg.published_permittivity(
        built["density_g_cm3"],
        epsilon_0=float(operator["epsilon_0"]),
        rho_c=10.0 ** float(operator["log10_rho_c_g_cm3"]),
        q_slope=float(operator["Q"]),
    )
    rg_boundary = built["newton_boundary"] / float(operator["epsilon_0"])
    initial = newton / float(operator["epsilon_0"])
    refracted, rg_metrics = solve_variable_pcg(
        rhs,
        rg_boundary,
        epsilon,
        grid.spacing,
        relative_tolerance=float(operator["pcg_relative_tolerance"]),
        absolute_tolerance=float(operator["pcg_absolute_tolerance"]),
        max_iterations=int(operator["pcg_max_iterations"]),
        initial_potential=initial,
    )
    radii = _radial_grid(config)
    profile_options = {
        "half_box_kpc": float(config["grid_contract"]["solver_half_box_kpc"]),
        "radii_kpc": radii,
        "azimuth_samples": int(config["grid_contract"]["azimuth_samples"]),
        "a0_m_s2": float(bridge_config["normalization_contract"]["a0_m_s2"]),
    }
    newton_profile = midplane_radial_profile(newton, grid, **profile_options)
    rg_profile = midplane_radial_profile(refracted, grid, **profile_options)
    source_mass_error = source_systematics._mass_error(built["masses"], expected_source)
    result = {
        "nodes_per_axis": nodes,
        "spacing_kpc": grid.spacing * float(config["grid_contract"]["solver_half_box_kpc"]),
        "source_masses_msun": built["masses"],
        "total_mass_msun": built["total_mass_msun"],
        "source_builder_mass_relative_error": source_mass_error,
        "dimensionless_mass_relative_error": built["dimensionless_mass_relative_error"],
        "solver_metrics": {
            "newton_relative_residual": newton_residual,
            "refracted_gravity": rg_metrics,
        },
        "profiles": {
            "NEWTON_3D_DST": newton_profile,
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG": rg_profile,
        },
        "field_hashes": {
            "density_dimensionless": array_sha256(built["density_dimensionless"]),
            "epsilon": array_sha256(epsilon),
            "newton_potential": array_sha256(newton),
            "refracted_gravity_potential": array_sha256(refracted),
        },
    }
    del rhs, newton, epsilon, refracted, built, grid, initial, rg_boundary
    gc.collect()
    return result


def _profile_difference(
    fine: Sequence[Mapping[str, Any]], coarse: Sequence[Mapping[str, Any]]
) -> list[float]:
    _require(len(fine) == len(coarse), "profile lengths changed")
    result: list[float] = []
    for first, second in zip(fine, coarse, strict=True):
        _require(first["radius_kpc"] == second["radius_kpc"], "profile radii changed")
        a = float(first["radial_acceleration_over_a0"])
        b = float(second["radial_acceleration_over_a0"])
        result.append(abs(a - b) / max(abs(a), abs(b), 1.0e-12))
    return result


def _adjudicate_object(
    config: Mapping[str, Any],
    object_id: str,
    fine: Mapping[str, Any],
    coarse: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = float(
        config["benchmark_contract"]["fine_vs_convergence_radial_relative_difference_max"]
    )
    operators = config["operator_contract"]["operators"]
    differences = {
        operator: _profile_difference(fine["profiles"][operator], coarse["profiles"][operator])
        for operator in operators
    }
    radii = _radial_grid(config)
    minimum_cells = float(config["grid_contract"]["minimum_fine_cells_per_scored_radius"])
    fine_spacing = float(fine["spacing_kpc"])
    rows: list[dict[str, Any]] = []
    for index, radius in enumerate(radii):
        positive = all(
            math.isfinite(float(fine["profiles"][operator][index]["radial_acceleration_over_a0"]))
            and float(fine["profiles"][operator][index]["radial_acceleration_over_a0"]) > 0.0
            for operator in operators
        )
        enough_cells = radius / fine_spacing >= minimum_cells
        converged = all(differences[operator][index] <= threshold for operator in operators)
        eligible = positive and enough_cells and converged
        rows.append(
            {
                "radius_kpc": radius,
                "fine_cells_per_radius": radius / fine_spacing,
                "positive_finite": positive,
                "fine_vs_convergence_relative_difference": {
                    operator: differences[operator][index] for operator in operators
                },
                "response_scoring_eligible": eligible,
                "disposition": (
                    "SOURCE_NUMERICALLY_ELIGIBLE_PENDING_RESPONSE_CONTRACT"
                    if eligible
                    else "NUMERICALLY_UNRESOLVED_RADIUS_RETAINED_NOT_SCORED"
                ),
            }
        )
    eligible = [row for row in rows if row["response_scoring_eligible"]]
    gates = config["benchmark_contract"]
    fine_metrics = fine["solver_metrics"]
    coarse_metrics = coarse["solver_metrics"]
    object_gates = {
        "source_mass": max(
            float(fine["source_builder_mass_relative_error"]),
            float(coarse["source_builder_mass_relative_error"]),
        )
        <= float(gates["source_mass_relative_error_max"]),
        "newton_residual": max(
            float(fine_metrics["newton_relative_residual"]),
            float(coarse_metrics["newton_relative_residual"]),
        )
        <= float(gates["pcg_relative_residual_max"]),
        "rg_pcg_converged": fine_metrics["refracted_gravity"]["converged"] is True
        and coarse_metrics["refracted_gravity"]["converged"] is True,
        "rg_residual": max(
            float(fine_metrics["refracted_gravity"]["relative_residual"]),
            float(coarse_metrics["refracted_gravity"]["relative_residual"]),
        )
        <= float(gates["pcg_relative_residual_max"]),
        "at_least_one_eligible_radius": len(eligible) > 0,
    }
    return {
        "object_id": object_id,
        "object_gates": object_gates,
        "all_object_gates_pass": all(object_gates.values()),
        "eligible_radius_count": len(eligible),
        "ineligible_radius_count": len(rows) - len(eligible),
        "eligible_radius_min_kpc": min((row["radius_kpc"] for row in eligible), default=None),
        "eligible_radius_max_kpc": max((row["radius_kpc"] for row in eligible), default=None),
        "maximum_relative_difference": {
            operator: max(values) for operator, values in differences.items()
        },
        "fine": fine,
        "convergence": coarse,
        "radius_adjudication": rows,
    }


def run_target_free_benchmarks(config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["benchmark_contract"]
    grid = base.make_grid(33)
    phi = (
        np.cos(math.pi * grid.x / 2.0)
        * np.cos(math.pi * grid.y / 2.0)
        * np.cos(math.pi * grid.z / 2.0)
        + 0.1 * grid.x
    )
    boundary = _boundary_only(phi)
    poisson_rhs = base._constant_laplacian(phi, grid.spacing)
    poisson, poisson_residual = solve_poisson_dst(poisson_rhs, boundary, grid.spacing)
    epsilon = (
        0.661 + 0.339 * (1.0 + np.tanh(1.79 * (0.4 * grid.x + 0.2 * grid.y - 0.3 * grid.z))) / 2.0
    )
    variable_rhs = base._variable_divergence(phi, epsilon, grid.spacing)
    pcg, pcg_metrics = solve_variable_pcg(
        variable_rhs,
        boundary,
        epsilon,
        grid.spacing,
        relative_tolerance=1.0e-11,
        absolute_tolerance=0.0,
        max_iterations=100,
    )
    small = base.make_grid(17)
    small_phi = (
        np.cos(math.pi * small.x / 2.0)
        * np.cos(math.pi * small.y / 2.0)
        * np.cos(math.pi * small.z / 2.0)
        + 0.05 * small.y
    )
    small_boundary = _boundary_only(small_phi)
    small_epsilon = (
        0.7 + 0.3 * (1.0 + np.tanh(1.2 * (small.x - 0.2 * small.y + 0.1 * small.z))) / 2.0
    )
    small_rhs = base._variable_divergence(small_phi, small_epsilon, small.spacing)
    pcg_small, _ = solve_variable_pcg(
        small_rhs,
        small_boundary,
        small_epsilon,
        small.spacing,
        relative_tolerance=1.0e-12,
        absolute_tolerance=0.0,
        max_iterations=100,
    )
    direct_small, _ = rg._solve_variable(small_rhs, small_boundary, small_epsilon, small.spacing)
    density = np.exp(-((grid.x / 0.3) ** 2 + (grid.y / 0.45) ** 2 + (grid.z / 0.2) ** 2))
    density_rhs = 4.0 * math.pi * density
    zero = np.zeros_like(density)
    newton, _ = solve_poisson_dst(density_rhs, zero, grid.spacing)
    constant_epsilon = 0.7
    scaled, scaled_metrics = solve_variable_pcg(
        density_rhs,
        zero,
        np.full_like(density, constant_epsilon),
        grid.spacing,
        relative_tolerance=1.0e-11,
        absolute_tolerance=0.0,
        max_iterations=100,
        initial_potential=newton / constant_epsilon,
    )
    anisotropic_epsilon = rg.published_permittivity(
        density,
        epsilon_0=0.661,
        rho_c=0.2,
        q_slope=1.79,
    )
    anisotropic, _ = solve_variable_pcg(
        density_rhs,
        zero,
        anisotropic_epsilon,
        grid.spacing,
        relative_tolerance=1.0e-11,
        absolute_tolerance=0.0,
        max_iterations=100,
    )
    rotated_density = np.rot90(density, axes=(0, 1)).copy()
    rotated_epsilon = np.rot90(anisotropic_epsilon, axes=(0, 1)).copy()
    rotated, _ = solve_variable_pcg(
        4.0 * math.pi * rotated_density,
        zero,
        rotated_epsilon,
        grid.spacing,
        relative_tolerance=1.0e-11,
        absolute_tolerance=0.0,
        max_iterations=100,
    )
    invalid_rejected = False
    try:
        solve_variable_pcg(
            np.zeros((5, 5, 5)),
            np.zeros((5, 5, 5)),
            np.zeros((5, 5, 5)),
            1.0,
            relative_tolerance=1.0e-10,
            absolute_tolerance=0.0,
            max_iterations=10,
        )
    except ScoringResolutionError:
        invalid_rejected = True
    metrics = {
        "dst_discrete_manufactured_max_error": float(np.max(np.abs(poisson - phi))),
        "dst_relative_residual": poisson_residual,
        "pcg_discrete_manufactured_max_error": float(np.max(np.abs(pcg - phi))),
        "pcg_discrete_relative_residual": float(pcg_metrics["relative_residual"]),
        "pcg_vs_direct_small_grid_max_error": float(np.max(np.abs(pcg_small - direct_small))),
        "constant_epsilon_scaling_max_error": float(
            np.max(np.abs(scaled - newton / constant_epsilon))
        ),
        "constant_epsilon_relative_residual": float(scaled_metrics["relative_residual"]),
        "rotation_relative_error": float(
            np.max(np.abs(rotated - np.rot90(anisotropic, axes=(0, 1))))
            / max(float(np.max(np.abs(anisotropic))), 1.0e-30)
        ),
        "invalid_coefficient_rejected": invalid_rejected,
    }
    checks = {
        "dst_discrete_manufactured": metrics["dst_discrete_manufactured_max_error"]
        <= float(gates["dst_discrete_manufactured_solution_max_error"]),
        "pcg_discrete_manufactured": metrics["pcg_discrete_manufactured_max_error"]
        <= float(gates["pcg_discrete_manufactured_solution_max_error"]),
        "pcg_residual": metrics["pcg_discrete_relative_residual"]
        <= float(gates["pcg_relative_residual_max"]),
        "pcg_matches_direct": metrics["pcg_vs_direct_small_grid_max_error"]
        <= float(gates["pcg_vs_direct_small_grid_max_error"]),
        "constant_epsilon_scaling": metrics["constant_epsilon_scaling_max_error"]
        <= float(gates["constant_epsilon_scaling_max_error"]),
        "rotation": metrics["rotation_relative_error"]
        <= float(gates["rotation_relative_error_max"]),
        "invalid_coefficient_control": invalid_rejected,
    }
    return {"metrics": metrics, "checks": checks, "all_pass": all(checks.values())}


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predecessors = validate_predecessors(config)
    benchmarks = run_target_free_benchmarks(config)
    _require(benchmarks["all_pass"] is True, "target-free scoring solver benchmark failed")
    source_config, acquisition, expected, bridge_config = _source_evidence()
    object_rows: list[dict[str, Any]] = []
    source_id = config["source_cell"]["id"]
    for object_id in config["objects"]:
        maps, exponential_scale_pc = _primary_maps(source_config, acquisition, object_id)
        convergence = _solve_source_grid(
            config,
            bridge_config,
            maps,
            exponential_scale_pc=exponential_scale_pc,
            expected_source=expected[(object_id, source_id)],
            nodes=int(config["grid_contract"]["convergence_nodes_per_axis"]),
        )
        fine = _solve_source_grid(
            config,
            bridge_config,
            maps,
            exponential_scale_pc=exponential_scale_pc,
            expected_source=expected[(object_id, source_id)],
            nodes=int(config["grid_contract"]["fine_nodes_per_axis"]),
        )
        object_rows.append(_adjudicate_object(config, object_id, fine, convergence))
        del maps, fine, convergence
        gc.collect()
    all_object_gates_pass = all(row["all_object_gates_pass"] for row in object_rows)
    eligible_points = sum(row["eligible_radius_count"] for row in object_rows)
    ineligible_points = sum(row["ineligible_radius_count"] for row in object_rows)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "PASS_SOURCE_ONLY_SCORING_RESOLUTION_WITH_RESPONSE_BLIND_RADIUS_MASK"
            if all_object_gates_pass
            else "BLOCK_SOURCE_OR_NUMERICAL_GATE_FAILURE_RETAINED"
        ),
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "predecessor_receipt_content_sha256": {
            role: row["content_sha256"] for role, row in predecessors.items()
        },
        "real_source_and_paper_anchors": config["real_source_and_paper_anchors"],
        "grid_contract": config["grid_contract"],
        "operator_contract": config["operator_contract"],
        "target_free_benchmarks": benchmarks,
        "objects": object_rows,
        "all_object_gates_pass": all_object_gates_pass,
        "response_blind_radius_summary": {
            "registered_points": len(config["objects"])
            * int(config["grid_contract"]["radial_points"]),
            "eligible_points": eligible_points,
            "ineligible_points": ineligible_points,
            "selection_used_velocity_values": False,
        },
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    expected = build_receipt(config)
    _require(dict(payload) == expected, "receipt does not match deterministic rebuild")


def _output_path() -> Path:
    path = _repo_path(OUTPUT_PATH)
    _require(path == (_ROOT / OUTPUT_PATH).resolve(), "output path changed")
    return path


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing to overwrite nonidentical receipt")
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
            _require(path.read_bytes() == payload, "concurrent nonidentical receipt")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    return _atomic_no_clobber(_output_path(), canonical_bytes(receipt) + b"\n")


def validate_receipt() -> None:
    config = load_config()
    path = _output_path()
    _require(path.is_file(), "receipt missing")
    payload = _read_json(path, "receipt")
    validate_receipt_payload(config, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"), nargs="?", default="check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        config = load_config()
        receipt = build_receipt(config)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "eligible_points": receipt["response_blind_radius_summary"]["eligible_points"],
                    "ineligible_points": receipt["response_blind_radius_summary"][
                        "ineligible_points"
                    ],
                    "response_files_opened": receipt["scientific_boundary"][
                        "response_files_opened"
                    ],
                    "scores_computed": receipt["scientific_boundary"]["scores_computed"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
