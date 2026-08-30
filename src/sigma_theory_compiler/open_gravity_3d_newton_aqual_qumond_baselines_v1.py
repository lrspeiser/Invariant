"""Target-free full-3D Newton, AQUAL, and QUMOND solver baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

CONFIG_PATH = Path("configs/open_gravity_3d_newton_aqual_qumond_baselines_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_3d_newton_aqual_qumond_baselines_v1.py")
TEST_PATH = Path("tests/test_open_gravity_3d_newton_aqual_qumond_baselines_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-3d-newton-aqual-qumond-baselines-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE_SHA256 = "cf2e19a8e19d73f855e3efcbff514098ffed07a578187e94f9b7a6b66a8a51d0"
_CONFIG_CONTENT_SHA256 = "3725f938173d9b527704248cb4ee21c44b2741774c2fa4e777dd2a78d966a1b0"
_SCHEMA = "invariant-open-gravity-3d-newton-aqual-qumond-baselines-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-3d-newton-aqual-qumond-receipt-1.0"
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_3d_newton_aqual_qumond_baselines_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_newton_aqual_qumond_baselines_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_3d_newton_aqual_qumond_baselines_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-3d-newton-aqual-qumond-baselines-v1/receipt.json"
)


class BaselineSolverError(RuntimeError):
    """Raised when a baseline solver or package gate fails closed."""


@dataclass(frozen=True)
class Grid3D:
    coordinates: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    spacing: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.x.shape


@dataclass(frozen=True)
class SolverResult:
    potential: np.ndarray
    relative_residual: float
    iterations: int
    converged: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineSolverError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineSolverError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    resolved = (_ROOT / expected).resolve()
    _require(resolved.is_relative_to(_ROOT), f"{label} escaped repository")
    return resolved


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselineSolverError("foundation commit binding failed") from exc


def make_grid(nodes: int, half_width: float = 1.0) -> Grid3D:
    _require(type(nodes) is int and nodes >= 7 and nodes % 2 == 1, "grid must be odd >=7")
    coordinates = np.linspace(-half_width, half_width, nodes, dtype=np.float64)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return Grid3D(coordinates, x, y, z, float(coordinates[1] - coordinates[0]))


def mu_simple(x: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    safe = np.maximum(np.asarray(x, dtype=np.float64), floor)
    return safe / (1.0 + safe)


def nu_simple(y: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    safe = np.maximum(np.asarray(y, dtype=np.float64), floor)
    return 0.5 + np.sqrt(0.25 + 1.0 / safe)


def _interior_indices(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    nx, ny, nz = shape
    mapping = -np.ones(shape, dtype=np.int64)
    count = (nx - 2) * (ny - 2) * (nz - 2)
    mapping[1:-1, 1:-1, 1:-1] = np.arange(count, dtype=np.int64).reshape(nx - 2, ny - 2, nz - 2)
    return mapping, count


def _constant_laplacian(phi: np.ndarray, spacing: float) -> np.ndarray:
    result = np.zeros_like(phi)
    h2 = spacing * spacing
    result[1:-1, 1:-1, 1:-1] = (
        phi[2:, 1:-1, 1:-1]
        + phi[:-2, 1:-1, 1:-1]
        + phi[1:-1, 2:, 1:-1]
        + phi[1:-1, :-2, 1:-1]
        + phi[1:-1, 1:-1, 2:]
        + phi[1:-1, 1:-1, :-2]
        - 6.0 * phi[1:-1, 1:-1, 1:-1]
    ) / h2
    return result


def _linear_matrix_and_rhs(
    rhs: np.ndarray, boundary: np.ndarray, spacing: float
) -> tuple[sparse.csr_matrix, np.ndarray]:
    _require(rhs.shape == boundary.shape, "linear shapes differ")
    mapping, count = _interior_indices(rhs.shape)
    h2 = spacing * spacing
    matrix = sparse.lil_matrix((count, count), dtype=np.float64)
    vector = rhs[1:-1, 1:-1, 1:-1].reshape(-1).astype(np.float64).copy()
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for i in range(1, rhs.shape[0] - 1):
        for j in range(1, rhs.shape[1] - 1):
            for k in range(1, rhs.shape[2] - 1):
                row = int(mapping[i, j, k])
                matrix[row, row] = -6.0 / h2
                for di, dj, dk in directions:
                    ni, nj, nk = i + di, j + dj, k + dk
                    column = int(mapping[ni, nj, nk])
                    if column >= 0:
                        matrix[row, column] = 1.0 / h2
                    else:
                        vector[row] -= boundary[ni, nj, nk] / h2
    return matrix.tocsr(), vector


def solve_poisson(rhs: np.ndarray, boundary: np.ndarray, spacing: float) -> SolverResult:
    matrix, vector = _linear_matrix_and_rhs(rhs, boundary, spacing)
    interior = spsolve(matrix, vector)
    potential = np.asarray(boundary, dtype=np.float64).copy()
    potential[1:-1, 1:-1, 1:-1] = interior.reshape(
        potential.shape[0] - 2, potential.shape[1] - 2, potential.shape[2] - 2
    )
    residual = _constant_laplacian(potential, spacing) - rhs
    scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return SolverResult(potential, relative, 1, math.isfinite(relative))


def _gradient(phi: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.gradient(phi, spacing, edge_order=2)
    return values[0], values[1], values[2]


def acceleration(phi: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gradient = _gradient(phi, spacing)
    return tuple(-component for component in gradient)


def _variable_divergence(phi: np.ndarray, coefficient: np.ndarray, spacing: float) -> np.ndarray:
    result = np.zeros_like(phi)
    h2 = spacing * spacing
    centre = phi[1:-1, 1:-1, 1:-1]
    cxp = 0.5 * (coefficient[1:-1, 1:-1, 1:-1] + coefficient[2:, 1:-1, 1:-1])
    cxm = 0.5 * (coefficient[1:-1, 1:-1, 1:-1] + coefficient[:-2, 1:-1, 1:-1])
    cyp = 0.5 * (coefficient[1:-1, 1:-1, 1:-1] + coefficient[1:-1, 2:, 1:-1])
    cym = 0.5 * (coefficient[1:-1, 1:-1, 1:-1] + coefficient[1:-1, :-2, 1:-1])
    czp = 0.5 * (coefficient[1:-1, 1:-1, 1:-1] + coefficient[1:-1, 1:-1, 2:])
    czm = 0.5 * (coefficient[1:-1, 1:-1, 1:-1] + coefficient[1:-1, 1:-1, :-2])
    result[1:-1, 1:-1, 1:-1] = (
        cxp * (phi[2:, 1:-1, 1:-1] - centre)
        - cxm * (centre - phi[:-2, 1:-1, 1:-1])
        + cyp * (phi[1:-1, 2:, 1:-1] - centre)
        - cym * (centre - phi[1:-1, :-2, 1:-1])
        + czp * (phi[1:-1, 1:-1, 2:] - centre)
        - czm * (centre - phi[1:-1, 1:-1, :-2])
    ) / h2
    return result


def _variable_matrix_and_rhs(
    rhs: np.ndarray, boundary: np.ndarray, coefficient: np.ndarray, spacing: float
) -> tuple[sparse.csr_matrix, np.ndarray]:
    mapping, count = _interior_indices(rhs.shape)
    h2 = spacing * spacing
    matrix = sparse.lil_matrix((count, count), dtype=np.float64)
    vector = rhs[1:-1, 1:-1, 1:-1].reshape(-1).astype(np.float64).copy()
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for i in range(1, rhs.shape[0] - 1):
        for j in range(1, rhs.shape[1] - 1):
            for k in range(1, rhs.shape[2] - 1):
                row = int(mapping[i, j, k])
                diagonal = 0.0
                for di, dj, dk in directions:
                    ni, nj, nk = i + di, j + dj, k + dk
                    face = 0.5 * (coefficient[i, j, k] + coefficient[ni, nj, nk]) / h2
                    diagonal -= face
                    column = int(mapping[ni, nj, nk])
                    if column >= 0:
                        matrix[row, column] = face
                    else:
                        vector[row] -= face * boundary[ni, nj, nk]
                matrix[row, row] = diagonal
    return matrix.tocsr(), vector


def solve_aqual(
    rhs: np.ndarray,
    boundary: np.ndarray,
    spacing: float,
    *,
    a0: float = 1.0,
    mu_floor: float = 1e-6,
    damping: float = 0.7,
    max_iterations: int = 300,
    delta_tolerance: float = 1e-9,
    residual_tolerance: float = 2e-7,
) -> SolverResult:
    _require(a0 > 0.0 and mu_floor > 0.0, "AQUAL scales must be positive")
    initial = solve_poisson(rhs, boundary, spacing).potential
    phi = initial
    relative = math.inf
    for iteration in range(1, max_iterations + 1):
        gradient = _gradient(phi, spacing)
        magnitude = np.sqrt(sum(component * component for component in gradient))
        coefficient = mu_simple(magnitude / a0, mu_floor)
        matrix, vector = _variable_matrix_and_rhs(rhs, boundary, coefficient, spacing)
        solved = np.asarray(spsolve(matrix, vector)).reshape(
            phi.shape[0] - 2, phi.shape[1] - 2, phi.shape[2] - 2
        )
        candidate = phi.copy()
        candidate[1:-1, 1:-1, 1:-1] = solved
        updated = phi.copy()
        updated[1:-1, 1:-1, 1:-1] = (
            damping * candidate[1:-1, 1:-1, 1:-1] + (1.0 - damping) * phi[1:-1, 1:-1, 1:-1]
        )
        delta_scale = max(float(np.max(np.abs(updated[1:-1, 1:-1, 1:-1]))), 1.0)
        delta = float(
            np.max(np.abs(updated[1:-1, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1])) / delta_scale
        )
        phi = updated
        final_gradient = _gradient(phi, spacing)
        final_magnitude = np.sqrt(sum(component * component for component in final_gradient))
        final_mu = mu_simple(final_magnitude / a0, mu_floor)
        residual = _variable_divergence(phi, final_mu, spacing) - rhs
        rhs_scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
        relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / rhs_scale)
        if delta <= delta_tolerance and relative <= residual_tolerance:
            return SolverResult(phi, relative, iteration, True)
    return SolverResult(phi, relative, max_iterations, False)


def _qumond_flux_divergence(
    phi: np.ndarray, spacing: float, a0: float, nu_floor: float
) -> np.ndarray:
    gx, gy, gz = _gradient(phi, spacing)
    normal_x = (phi[1:, :, :] - phi[:-1, :, :]) / spacing
    cross_xy = 0.5 * (gy[1:, :, :] + gy[:-1, :, :])
    cross_xz = 0.5 * (gz[1:, :, :] + gz[:-1, :, :])
    magnitude_x = np.sqrt(normal_x * normal_x + cross_xy * cross_xy + cross_xz * cross_xz)
    flux_x = nu_simple(magnitude_x / a0, nu_floor) * normal_x

    normal_y = (phi[:, 1:, :] - phi[:, :-1, :]) / spacing
    cross_yx = 0.5 * (gx[:, 1:, :] + gx[:, :-1, :])
    cross_yz = 0.5 * (gz[:, 1:, :] + gz[:, :-1, :])
    magnitude_y = np.sqrt(normal_y * normal_y + cross_yx * cross_yx + cross_yz * cross_yz)
    flux_y = nu_simple(magnitude_y / a0, nu_floor) * normal_y

    normal_z = (phi[:, :, 1:] - phi[:, :, :-1]) / spacing
    cross_zx = 0.5 * (gx[:, :, 1:] + gx[:, :, :-1])
    cross_zy = 0.5 * (gy[:, :, 1:] + gy[:, :, :-1])
    magnitude_z = np.sqrt(normal_z * normal_z + cross_zx * cross_zx + cross_zy * cross_zy)
    flux_z = nu_simple(magnitude_z / a0, nu_floor) * normal_z

    result = np.zeros_like(phi)
    result[1:-1, 1:-1, 1:-1] = (
        (flux_x[1:, 1:-1, 1:-1] - flux_x[:-1, 1:-1, 1:-1])
        + (flux_y[1:-1, 1:, 1:-1] - flux_y[1:-1, :-1, 1:-1])
        + (flux_z[1:-1, 1:-1, 1:] - flux_z[1:-1, 1:-1, :-1])
    ) / spacing
    return result


def solve_qumond(
    rhs: np.ndarray,
    newton_boundary: np.ndarray,
    mond_boundary: np.ndarray,
    spacing: float,
    *,
    a0: float = 1.0,
    nu_floor: float = 1e-6,
) -> tuple[SolverResult, SolverResult, np.ndarray]:
    newton = solve_poisson(rhs, newton_boundary, spacing)
    qumond_rhs = _qumond_flux_divergence(newton.potential, spacing, a0, nu_floor)
    qumond = solve_poisson(qumond_rhs, mond_boundary, spacing)
    return newton, qumond, qumond_rhs


def _relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(expected.ravel())), 1e-30)
    return float(np.linalg.norm((observed - expected).ravel()) / scale)


def _internal_potential(
    phi: np.ndarray, grid: Grid3D, external: tuple[float, float, float]
) -> np.ndarray:
    ext = -(external[0] * grid.x + external[1] * grid.y + external[2] * grid.z)
    return phi - ext


def _gaussian(
    grid: Grid3D,
    centres: Sequence[tuple[float, float, float]],
    widths: Sequence[tuple[float, float, float]],
    weights: Sequence[float],
) -> np.ndarray:
    density = np.zeros(grid.shape, dtype=np.float64)
    for centre, width, weight in zip(centres, widths, weights, strict=True):
        density += weight * np.exp(
            -(
                ((grid.x - centre[0]) / width[0]) ** 2
                + ((grid.y - centre[1]) / width[1]) ** 2
                + ((grid.z - centre[2]) / width[2]) ** 2
            )
        )
    return density


def _zero_boundary(grid: Grid3D) -> np.ndarray:
    return np.zeros(grid.shape, dtype=np.float64)


def _linear_boundary(grid: Grid3D, field: tuple[float, float, float]) -> np.ndarray:
    boundary = -(field[0] * grid.x + field[1] * grid.y + field[2] * grid.z)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    result = np.zeros(grid.shape, dtype=np.float64)
    result[mask] = boundary[mask]
    return result


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    nodes = config["grid_contract"]["primary_nodes_per_axis"]
    grid = make_grid(nodes, config["normalized_units"]["box_half_width"])
    h = grid.spacing
    gates = config["gate_contract"]
    solver = config["grid_contract"]
    zero = _zero_boundary(grid)

    manufactured = (
        np.cos(math.pi * grid.x / 2.0)
        * np.cos(math.pi * grid.y / 2.0)
        * np.cos(math.pi * grid.z / 2.0)
    )
    newton_rhs_discrete = _constant_laplacian(manufactured, h)
    newton_manufactured = solve_poisson(newton_rhs_discrete, manufactured, h)
    newton_error = _relative_error(newton_manufactured.potential, manufactured)

    manufactured_gradient = _gradient(manufactured, h)
    manufactured_mu = mu_simple(
        np.sqrt(sum(component * component for component in manufactured_gradient)),
        config["equation_contract"]["aqual_numerical_floor"],
    )
    aqual_rhs = _variable_divergence(manufactured, manufactured_mu, h)
    aqual_manufactured = solve_aqual(
        aqual_rhs,
        manufactured,
        h,
        mu_floor=config["equation_contract"]["aqual_numerical_floor"],
        damping=solver["aqual_damping"],
        max_iterations=solver["aqual_max_iterations"],
        delta_tolerance=solver["aqual_delta_tolerance"],
        residual_tolerance=solver["aqual_residual_tolerance"],
    )
    aqual_error = _relative_error(aqual_manufactured.potential, manufactured)

    sphere_density = (grid.x * grid.x + grid.y * grid.y + grid.z * grid.z <= 0.35**2).astype(
        np.float64
    )
    sphere_rhs = 4.0 * math.pi * sphere_density
    sphere_aqual = solve_aqual(sphere_rhs, zero, h, **_aqual_options(config))
    _, sphere_qumond, _ = solve_qumond(sphere_rhs, zero, zero, h)
    spherical_difference = _relative_error(sphere_aqual.potential, sphere_qumond.potential)

    radius = np.sqrt(grid.x * grid.x + grid.y * grid.y + grid.z * grid.z)
    shell_density = ((radius >= 0.35) & (radius <= 0.55)).astype(np.float64)
    shell = solve_poisson(4.0 * math.pi * shell_density, zero, h)
    shell_acceleration = acceleration(shell.potential, h)
    centre = nodes // 2
    shell_centre = float(
        math.sqrt(sum(component[centre, centre, centre] ** 2 for component in shell_acceleration))
    )

    disk_density = _gaussian(grid, [(0.0, 0.0, 0.0)], [(0.45, 0.45, 0.08)], [1.0])
    disk_rhs = 4.0 * math.pi * disk_density
    disk_aqual = solve_aqual(disk_rhs, zero, h, **_aqual_options(config))
    _, disk_qumond, _ = solve_qumond(disk_rhs, zero, zero, h)
    disk_difference = _relative_error(disk_aqual.potential, disk_qumond.potential)

    bar_density = _gaussian(grid, [(0.0, 0.0, 0.0)], [(0.55, 0.20, 0.12)], [1.0])
    bar = solve_poisson(4.0 * math.pi * bar_density, zero, h)
    rotated_density = np.rot90(bar_density, axes=(0, 1))
    rotated = solve_poisson(4.0 * math.pi * rotated_density, zero, h)
    rotation_error = _relative_error(rotated.potential, np.rot90(bar.potential, axes=(0, 1)))

    pair_density = _gaussian(
        grid,
        [(-0.35, 0.0, 0.0), (0.35, 0.0, 0.0)],
        [(0.12, 0.12, 0.12), (0.12, 0.12, 0.12)],
        [1.0, 1.0],
    )
    pair = solve_poisson(4.0 * math.pi * pair_density, zero, h)
    pair_acceleration = acceleration(pair.potential, h)
    pair_centre = float(
        math.sqrt(sum(component[centre, centre, centre] ** 2 for component in pair_acceleration))
    )

    compact_density = _gaussian(grid, [(0.0, 0.0, 0.0)], [(0.18, 0.18, 0.18)], [1.0])
    compact_rhs = 4.0 * math.pi * compact_density
    isolated = solve_aqual(compact_rhs, zero, h, **_aqual_options(config))
    external_field = (0.35, 0.0, 0.0)
    external_boundary = _linear_boundary(grid, external_field)
    external = solve_aqual(compact_rhs, external_boundary, h, **_aqual_options(config))
    internal_external = _internal_potential(external.potential, grid, external_field)
    external_change = _relative_error(internal_external, isolated.potential)
    _, qumond_isolated, _ = solve_qumond(compact_rhs, zero, zero, h)
    external_magnitude = external_field[0]
    newton_external_magnitude = external_magnitude * float(
        mu_simple(np.array([external_magnitude]))[0]
    )
    newton_external_field = (newton_external_magnitude, 0.0, 0.0)
    newton_external_boundary = _linear_boundary(grid, newton_external_field)
    _, qumond_external, _ = solve_qumond(
        compact_rhs,
        newton_external_boundary,
        external_boundary,
        h,
    )
    qumond_internal_external = _internal_potential(qumond_external.potential, grid, external_field)
    qumond_external_change = _relative_error(qumond_internal_external, qumond_isolated.potential)

    high_density = _gaussian(
        grid,
        [(0.04, 0.03, -0.02)],
        [(0.18, 0.17, 0.16)],
        [1.0],
    )
    high_rhs = 1.0e5 * 4.0 * math.pi * high_density
    high_newton = solve_poisson(high_rhs, zero, h)
    high_aqual = solve_aqual(high_rhs, zero, h, **_aqual_options(config))
    _, high_qumond, _ = solve_qumond(high_rhs, zero, zero, h)
    high_aqual_difference = _relative_error(high_aqual.potential, high_newton.potential)
    high_qumond_difference = _relative_error(high_qumond.potential, high_newton.potential)
    qumond_linear_residual_max = max(
        sphere_qumond.relative_residual,
        disk_qumond.relative_residual,
        qumond_isolated.relative_residual,
        qumond_external.relative_residual,
        high_qumond.relative_residual,
    )

    resolution_errors: dict[str, float] = {}
    for resolution in config["grid_contract"]["convergence_nodes_per_axis"]:
        current = make_grid(resolution)
        exact = (
            np.cos(math.pi * current.x / 2.0)
            * np.cos(math.pi * current.y / 2.0)
            * np.cos(math.pi * current.z / 2.0)
        )
        continuous_rhs = -3.0 * (math.pi / 2.0) ** 2 * exact
        solved = solve_poisson(continuous_rhs, exact, current.spacing)
        resolution_errors[str(resolution)] = _relative_error(solved.potential, exact)

    checks = {
        "B3D01_NEWTON_MANUFACTURED": newton_manufactured.relative_residual
        <= gates["linear_relative_residual_max"]
        and newton_error <= gates["manufactured_relative_error_max"],
        "B3D02_AQUAL_MANUFACTURED": aqual_manufactured.converged
        and aqual_manufactured.relative_residual <= gates["aqual_relative_residual_max"]
        and aqual_error <= gates["manufactured_relative_error_max"],
        "B3D03_SPHERE": sphere_aqual.converged
        and sphere_qumond.relative_residual <= gates["linear_relative_residual_max"]
        and spherical_difference <= gates["spherical_aqual_qumond_relative_difference_max"],
        "B3D04_SHELL": shell_centre <= gates["shell_centre_acceleration_max"],
        "B3D05_DISK": disk_aqual.converged
        and disk_qumond.relative_residual <= gates["linear_relative_residual_max"]
        and disk_difference >= gates["nonspherical_aqual_qumond_minimum_difference"],
        "B3D06_BAR_TRIAXIAL": rotation_error <= gates["rotation_relative_error_max"],
        "B3D07_PAIR_SADDLE": pair_centre <= gates["pair_centre_acceleration_max"],
        "B3D08_EXTERNAL_FIELD": external.converged
        and qumond_isolated.relative_residual <= gates["linear_relative_residual_max"]
        and qumond_external.relative_residual <= gates["linear_relative_residual_max"]
        and external_change >= gates["external_field_minimum_internal_change"]
        and qumond_external_change >= gates["external_field_minimum_internal_change"],
        "B3D09_HIGH_ACCELERATION": high_aqual.converged
        and high_newton.relative_residual <= gates["linear_relative_residual_max"]
        and high_qumond.relative_residual <= gates["linear_relative_residual_max"]
        and high_aqual_difference <= gates["high_acceleration_newton_relative_difference_max"]
        and high_qumond_difference <= gates["high_acceleration_newton_relative_difference_max"],
        "B3D10_RESOLUTION": all(
            later < earlier
            for earlier, later in zip(
                list(resolution_errors.values())[:-1],
                list(resolution_errors.values())[1:],
                strict=True,
            )
        ),
    }
    metrics = {
        "newton_manufactured_relative_error": newton_error,
        "newton_linear_relative_residual": newton_manufactured.relative_residual,
        "aqual_manufactured_relative_error": aqual_error,
        "aqual_relative_residual": aqual_manufactured.relative_residual,
        "aqual_iterations": aqual_manufactured.iterations,
        "spherical_aqual_qumond_relative_difference": spherical_difference,
        "shell_centre_acceleration": shell_centre,
        "nonspherical_disk_aqual_qumond_relative_difference": disk_difference,
        "bar_rotation_relative_error": rotation_error,
        "pair_centre_acceleration": pair_centre,
        "external_field_internal_relative_change": external_change,
        "qumond_external_field_internal_relative_change": qumond_external_change,
        "high_acceleration_aqual_newton_relative_difference": high_aqual_difference,
        "high_acceleration_qumond_newton_relative_difference": high_qumond_difference,
        "qumond_linear_relative_residual_max": qumond_linear_residual_max,
        "resolution_relative_errors": resolution_errors,
    }
    return {
        "checks": checks,
        "metrics": metrics,
        "all_pass": all(checks.values()),
        "passed": sum(checks.values()),
        "total": len(checks),
    }


def _aqual_options(config: Mapping[str, Any]) -> dict[str, Any]:
    solver = config["grid_contract"]
    return {
        "mu_floor": config["equation_contract"]["aqual_numerical_floor"],
        "damping": solver["aqual_damping"],
        "max_iterations": solver["aqual_max_iterations"],
        "delta_tolerance": solver["aqual_delta_tolerance"],
        "residual_tolerance": solver["aqual_residual_tolerance"],
    }


def validate_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "foundation_binding",
        "primary_sources",
        "equation_contract",
        "normalized_units",
        "grid_contract",
        "boundary_contract",
        "fixture_contract",
        "gate_contract",
        "independence_contract",
        "access_contract",
        "claim_boundary",
        "output_path",
        "section_sha256",
    }
    _require(type(config) is dict and set(config) == expected_keys, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-3d-newton-aqual-qumond-baselines-v1",
        "package ID changed",
    )
    _require(config["status"] == "TARGET_FREE_SYNTHETIC_SOLVER_PREFLIGHT", "status changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    seals = config["section_sha256"]
    sealed_sections = expected_keys - {
        "schema",
        "package_id",
        "status",
        "purpose",
        "output_path",
        "section_sha256",
    }
    _require(type(seals) is dict and set(seals) == sealed_sections, "section seal set changed")
    for key in sealed_sections:
        _require(seals[key] == content_sha256(config[key]), f"section {key} changed")
    _require(
        config["access_contract"]
        == {
            "scientific_source_files": 0,
            "scientific_response_files": 0,
            "scientific_rows": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "access contract changed",
    )
    _require(len(config["fixture_contract"]) == 10, "fixture count changed")
    _require(config["boundary_contract"]["periodic_forbidden"] is True, "boundary changed")
    _require(
        config["boundary_contract"]["response_fitted_boundary_forbidden"] is True,
        "response boundary changed",
    )


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_FILE_SHA256, "config bytes changed")
    config = _read_json(path, "baseline config")
    validate_config(config)
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config content changed")
    binding = config["foundation_binding"]
    commit = binding["commit"]
    for artifact in binding["artifacts"]:
        expected = artifact["sha256"]
        _require(
            hashlib.sha256(_git_show(commit, artifact["path"])).hexdigest() == expected,
            "committed foundation changed",
        )
        _require(file_sha256(_ROOT / artifact["path"]) == expected, "foundation worktree changed")
    return config


def build_receipt() -> dict[str, Any]:
    config = load_config()
    suite = run_suite(config)
    _require(suite["all_pass"], "one or more synthetic solver gates failed")
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_TARGET_FREE_NEWTON_AQUAL_QUMOND_3D_BASELINES",
        "access_accounting": config["access_contract"],
        "bindings": {
            "config": {
                "path": _CANONICAL_CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / _CANONICAL_CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": file_sha256(module_path),
            },
            "test": {"path": _CANONICAL_TEST_PATH.as_posix(), "sha256": file_sha256(test_path)},
            "foundation": config["foundation_binding"],
        },
        "equations": config["equation_contract"],
        "primary_sources": config["primary_sources"],
        "synthetic_suite": suite,
        "independence_contract": config["independence_contract"],
        "claim_boundary": config["claim_boundary"],
        "section_sha256": config["section_sha256"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "refusing to overwrite nonidentical receipt")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return "CREATED"


def validate_receipt_payload(stored: Mapping[str, Any]) -> None:
    observed = stored.get("content_sha256")
    body = {key: value for key, value in stored.items() if key != "content_sha256"}
    _require(observed == content_sha256(body), "receipt self-hash changed")
    _require(dict(stored) == build_receipt(), "receipt is not reproducible")


def validate_receipt() -> None:
    validate_receipt_payload(_read_json(_output_path(), "baseline receipt"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "build", "validate"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        print(json.dumps(run_suite(load_config()), sort_keys=True, indent=2))
    elif args.command == "build":
        print(write_receipt())
    else:
        validate_receipt()
        print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
