"""Executable disk-polar load and directional-carrier gravity formulas.

The package registers a source-derived disk normal, trace-preserving anisotropic
diffusion, density-gradient drift, baryonic absorption, and two observables from
one load-potential state.  Its distinct kinetic parent evolves a directional
carrier distribution and derives its scalar, flux, and pressure moments.  A
standard quadrupole-radiation control prevents static fields from being confused
with radiation.  It intentionally opens no scientific response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_disk_polar_escape_load_v1.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_disk_polar_escape_load_v1.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_disk_polar_escape_load_v1.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-disk-polar-escape-load-v4/receipt.json"

_CONFIG_RAW_SHA256 = "3deeb79c4cc51eaf7f03c768cb736cfdd7563666a3485e2745e4e7c919bd4b24"
_CONFIG_CONTENT_SHA256 = "67b4b749162dd2484886059a3e5646c8abeb1f81bc0cbee40431bab2871ecb4e"
_MODULE_SEMANTIC_SHA256 = "00dcb6a886c677e28f347accfc368f06dbf5f1396702ffd44ef329f8dd83461f"
_TEST_RAW_SHA256 = "dfa3a4d5658436e988ca068cc3c07107b769e1c16c7a2bf2531053e22b7d3687"


class DiskPolarEscapeError(RuntimeError):
    """Raised when the registered disk-polar formula fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiskPolarEscapeError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    ):
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        stop = text.index('"', start)
        text = text[:start] + "0" * 64 + text[stop:]
    return hashlib.sha256(text.encode()).hexdigest()


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw drift")


def load_config() -> dict[str, Any]:
    validate_code_pins()
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config content drift")
    _require(
        value["status"] == "FROZEN_TARGET_FREE_EXECUTABLE_SOURCE_READY_RESPONSE_UNOPENED",
        "status drift",
    )
    _require(value["output_path"] == OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(), "output drift")
    accounting = value["access_accounting"]
    _require(all(entry == 0 for entry in accounting.values()), "nonzero access accounting")
    _require(
        [row["id"] for row in value["formula_catalog"]]
        == [
            "DPEL01_DISK_POLAR_ESCAPE_LOAD",
            "DGKT01_DIRECTIONAL_GRAVITY_KINETIC_TRANSPORT",
            "GRRAD00_STATIC_FIELD_VS_RADIATION_CONTROL",
        ],
        "formula catalog drift",
    )
    _require(
        value["second_proposal_origin"]["attachment_content_sha256"]
        == "12974c1d8d06972145ce04d142e85e32a23669f4ded8af4c85ca3c029e12deb8",
        "second proposal binding drift",
    )
    return value


def _finite_array(value: Any, shape: tuple[int, ...] | None, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None:
        _require(array.shape == shape, f"invalid {label} shape")
    _require(bool(np.all(np.isfinite(array))), f"nonfinite {label}")
    return array


def _canonical_unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    _require(math.isfinite(norm) and norm > 0.0, "invalid direction norm")
    result = vector / norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result = -result
    return result


def disk_shape_from_points(points: Any, weights: Any) -> dict[str, Any]:
    """Return the source barycenter, covariance, disk normal, and bounded activation."""

    xyz = _finite_array(points, None, "points")
    mass = _finite_array(weights, None, "weights")
    _require(xyz.ndim == 2 and xyz.shape[1] == 3, "points must be N by 3")
    _require(mass.shape == (xyz.shape[0],), "weight length mismatch")
    _require(bool(np.all(mass >= 0.0)) and float(np.sum(mass)) > 0.0, "invalid weights")
    total = float(np.sum(mass))
    center = np.sum(xyz * mass[:, None], axis=0) / total
    centered = xyz - center
    covariance = np.einsum("ni,nj,n->ij", centered, centered, mass) / total
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    _require(bool(np.all(eigenvalues >= -1e-13)), "negative covariance eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues))
    _require(trace > 0.0, "degenerate source extent")
    normal = _canonical_unit_vector(eigenvectors[:, 0])
    projector = np.outer(normal, normal)
    activation = float(np.clip(1.0 - 3.0 * eigenvalues[0] / trace, 0.0, 1.0))
    return {
        "center": center,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "normal": normal,
        "projector": projector,
        "activation": activation,
    }


def disk_shape_from_density(density: Any, spacing: float) -> dict[str, Any]:
    rho = _finite_array(density, None, "density")
    _require(rho.ndim == 3 and min(rho.shape) >= 3, "density must be a 3D grid")
    _require(bool(np.all(rho >= 0.0)), "negative density")
    _require(math.isfinite(spacing) and spacing > 0.0, "invalid spacing")
    axes = [(np.arange(size, dtype=float) - 0.5 * (size - 1)) * spacing for size in rho.shape]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return disk_shape_from_points(points, rho.reshape(-1))


def diffusion_tensor(
    normal: Any,
    activation: float,
    d_parallel: float,
    d_perpendicular: float,
) -> np.ndarray:
    n = _canonical_unit_vector(_finite_array(normal, (3,), "normal"))
    values = (activation, d_parallel, d_perpendicular)
    _require(all(math.isfinite(item) for item in values), "nonfinite diffusion parameter")
    _require(0.0 <= activation <= 1.0, "activation outside [0,1]")
    _require(d_parallel > 0.0 and d_perpendicular > 0.0, "diffusion must be positive")
    projector = np.outer(n, n)
    identity = np.eye(3)
    d_iso = (2.0 * d_parallel + d_perpendicular) / 3.0
    directional = d_parallel * (identity - projector) + d_perpendicular * projector
    tensor = (1.0 - activation) * d_iso * identity + activation * directional
    tensor = 0.5 * (tensor + tensor.T)
    eigenvalues = np.linalg.eigvalsh(tensor)
    _require(bool(np.all(eigenvalues > 0.0)), "diffusion tensor is not positive definite")
    _require(abs(float(np.trace(tensor)) - 3.0 * d_iso) <= 1e-12 * max(1.0, d_iso), "trace drift")
    return tensor


def periodic_gradient(field: Any, spacing: float) -> np.ndarray:
    scalar = _finite_array(field, None, "field")
    _require(scalar.ndim == 3, "field must be 3D")
    _require(math.isfinite(spacing) and spacing > 0.0, "invalid spacing")
    return np.stack(
        [
            (np.roll(scalar, -1, axis=axis) - np.roll(scalar, 1, axis=axis)) / (2.0 * spacing)
            for axis in range(3)
        ]
    )


def periodic_divergence(vector: Any, spacing: float) -> np.ndarray:
    field = _finite_array(vector, None, "vector field")
    _require(field.ndim == 4 and field.shape[0] == 3, "vector field must be 3 by Nx by Ny by Nz")
    _require(math.isfinite(spacing) and spacing > 0.0, "invalid spacing")
    result = np.zeros(field.shape[1:], dtype=float)
    for axis in range(3):
        result += (np.roll(field[axis], -1, axis=axis) - np.roll(field[axis], 1, axis=axis)) / (
            2.0 * spacing
        )
    return result


def density_gradient_drift(density_ratio: Any, spacing: float, chi: float) -> np.ndarray:
    _require(math.isfinite(chi) and chi >= 0.0, "invalid drift coefficient")
    return -chi * periodic_gradient(density_ratio, spacing)


def load_rhs(
    load: Any,
    density_ratio: Any,
    spacing: float,
    tensor: Any,
    chi: float,
    source_gain: float,
    gamma0: float,
    beta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    u = _finite_array(load, None, "load")
    source = _finite_array(density_ratio, None, "density ratio")
    _require(u.shape == source.shape and u.ndim == 3, "load/source shape mismatch")
    _require(bool(np.all(source >= 0.0)), "negative density ratio")
    d_tensor = _finite_array(tensor, (3, 3), "diffusion tensor")
    _require(
        bool(np.all(np.linalg.eigvalsh(0.5 * (d_tensor + d_tensor.T)) > 0.0)), "invalid tensor"
    )
    parameters = (chi, source_gain, gamma0, beta)
    _require(
        all(math.isfinite(item) and item >= 0.0 for item in parameters), "invalid PDE parameter"
    )

    drift = density_gradient_drift(source, spacing, chi)
    advective_flux = u[None, ...] * drift
    gradient = periodic_gradient(u, spacing)
    diffusive_flux = np.einsum("ij,j...->i...", d_tensor, gradient)
    source_term = source_gain * source
    sink_term = (gamma0 + beta * source) * u
    rhs = (
        -periodic_divergence(advective_flux, spacing)
        + periodic_divergence(diffusive_flux, spacing)
        + source_term
        - sink_term
    )
    expected_sum = float(np.sum(source_term - sink_term))
    observed_sum = float(np.sum(rhs))
    budget_scale = max(1.0, abs(expected_sum), abs(observed_sum))
    budget_residual = abs(observed_sum - expected_sum) / budget_scale
    _require(bool(np.all(np.isfinite(rhs))), "nonfinite PDE right-hand side")
    return rhs, {
        "source_sum": float(np.sum(source_term)),
        "sink_sum": float(np.sum(sink_term)),
        "rhs_sum": observed_sum,
        "periodic_budget_relative_residual": budget_residual,
    }


def integrate_load(
    density_ratio: Any,
    spacing: float,
    tensor: Any,
    *,
    chi: float,
    source_gain: float,
    gamma0: float,
    beta: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, dict[str, float]]:
    source = _finite_array(density_ratio, None, "density ratio")
    _require(bool(np.all(source >= 0.0)), "negative source")
    _require(
        math.isfinite(dt) and dt > 0.0 and isinstance(steps, int) and steps >= 0,
        "invalid integrator",
    )
    d_tensor = _finite_array(tensor, (3, 3), "diffusion tensor")
    maximum_diffusion = float(np.max(np.linalg.eigvalsh(d_tensor)))
    _require(
        dt * maximum_diffusion / (spacing * spacing) <= 0.12, "explicit diffusion CFL exceeded"
    )
    load = np.zeros_like(source)
    maximum_budget = 0.0
    minimum_load = 0.0
    for _ in range(steps):
        rhs, budget = load_rhs(
            load,
            source,
            spacing,
            d_tensor,
            chi,
            source_gain,
            gamma0,
            beta,
        )
        load = load + dt * rhs
        minimum_load = min(minimum_load, float(np.min(load)))
        _require(minimum_load >= -1e-11, "load positivity failure")
        maximum_budget = max(maximum_budget, budget["periodic_budget_relative_residual"])
    load = np.maximum(load, 0.0)
    return load, {
        "steps": float(steps),
        "time": float(steps * dt),
        "minimum_pre_roundoff_load": minimum_load,
        "maximum_budget_relative_residual": maximum_budget,
        "total_load": float(np.sum(load) * spacing**3),
        "maximum_load": float(np.max(load)),
    }


def anisotropic_green(
    point: Any,
    tensor: Any,
    gamma0: float,
    source_strength: float = 1.0,
) -> float:
    x = _finite_array(point, (3,), "Green point")
    d_tensor = _finite_array(tensor, (3, 3), "Green tensor")
    _require(math.isfinite(gamma0) and gamma0 >= 0.0, "invalid screening")
    _require(math.isfinite(source_strength) and source_strength >= 0.0, "invalid Green source")
    determinant = float(np.linalg.det(d_tensor))
    _require(determinant > 0.0, "singular Green tensor")
    inverse = np.linalg.inv(d_tensor)
    radius = math.sqrt(float(x @ inverse @ x))
    _require(radius > 0.0, "Green function singular at source")
    return (
        source_strength
        * math.exp(-math.sqrt(gamma0) * radius)
        / (4.0 * math.pi * math.sqrt(determinant) * radius)
    )


def anisotropic_green_residual(point: Any, tensor: Any, gamma0: float, h: float = 2e-4) -> float:
    x = _finite_array(point, (3,), "residual point")
    d_tensor = _finite_array(tensor, (3, 3), "residual tensor")
    _require(math.isfinite(h) and h > 0.0, "invalid difference step")
    center = anisotropic_green(x, d_tensor, gamma0)
    hessian = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for i in range(3):
        plus = anisotropic_green(x + h * identity[i], d_tensor, gamma0)
        minus = anisotropic_green(x - h * identity[i], d_tensor, gamma0)
        hessian[i, i] = (plus - 2.0 * center + minus) / h**2
        for j in range(i + 1, 3):
            pp = anisotropic_green(x + h * identity[i] + h * identity[j], d_tensor, gamma0)
            pm = anisotropic_green(x + h * identity[i] - h * identity[j], d_tensor, gamma0)
            mp = anisotropic_green(x - h * identity[i] + h * identity[j], d_tensor, gamma0)
            mm = anisotropic_green(x - h * identity[i] - h * identity[j], d_tensor, gamma0)
            hessian[i, j] = hessian[j, i] = (pp - pm - mp + mm) / (4.0 * h**2)
    residual = float(np.sum(d_tensor * hessian) - gamma0 * center)
    scale = max(abs(gamma0 * center), abs(float(np.sum(d_tensor * hessian))), abs(center), 1e-30)
    return abs(residual) / scale


def matter_acceleration_from_load(
    load: Any, spacing: float, c_squared_scale: float = 1.0
) -> np.ndarray:
    _require(math.isfinite(c_squared_scale) and c_squared_scale > 0.0, "invalid potential scale")
    return c_squared_scale * periodic_gradient(load, spacing)


def photon_path_log_redshift(
    load_over_c_squared: Any, path_spacing: float, eta_over_c: float
) -> float:
    values = _finite_array(load_over_c_squared, None, "path load")
    _require(values.ndim == 1 and values.size >= 2, "path requires at least two samples")
    _require(math.isfinite(path_spacing) and path_spacing > 0.0, "invalid path spacing")
    _require(math.isfinite(eta_over_c), "invalid photon coupling")
    return float(eta_over_c * np.trapezoid(values, dx=path_spacing))


def disk_aligned_angular_quadrature(
    normal: Any = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return six inversion-symmetric rays aligned to a source normal."""

    n = _canonical_unit_vector(_finite_array(normal, (3,), "quadrature normal"))
    seed = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = _canonical_unit_vector(seed - float(seed @ n) * n)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    directions = np.array([e1, -e1, e2, -e2, n, -n], dtype=float)
    return directions, np.full(6, 1.0 / 6.0)


def cartesian_angular_quadrature() -> tuple[np.ndarray, np.ndarray]:
    """Return the registered z-normal quadrature."""

    return disk_aligned_angular_quadrature()


def disk_aligned_collision_rates(
    activation: float,
    d_parallel: float,
    d_perpendicular: float,
    carrier_speed: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map DPEL diffusion eigenvalues to conservative odd-mode rates."""

    values = (activation, d_parallel, d_perpendicular, carrier_speed)
    _require(all(math.isfinite(value) for value in values), "nonfinite collision parameter")
    _require(0.0 <= activation <= 1.0, "collision activation outside [0,1]")
    _require(
        d_parallel > 0.0 and d_perpendicular > 0.0 and carrier_speed > 0.0,
        "bad collision parameter",
    )
    d_iso = (2.0 * d_parallel + d_perpendicular) / 3.0
    effective_parallel = (1.0 - activation) * d_iso + activation * d_parallel
    effective_perpendicular = (1.0 - activation) * d_iso + activation * d_perpendicular
    kappa_parallel = carrier_speed**2 / (3.0 * effective_parallel)
    kappa_perpendicular = carrier_speed**2 / (3.0 * effective_perpendicular)
    return np.array(
        [
            kappa_parallel,
            kappa_parallel,
            kappa_parallel,
            kappa_parallel,
            kappa_perpendicular,
            kappa_perpendicular,
        ]
    ), {
        "effective_parallel_diffusion": effective_parallel,
        "effective_perpendicular_diffusion": effective_perpendicular,
        "kappa_parallel": kappa_parallel,
        "kappa_perpendicular": kappa_perpendicular,
    }


def _validate_angular_quadrature(directions: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
    omega = _finite_array(directions, None, "carrier directions")
    angular_weights = _finite_array(weights, None, "angular weights")
    _require(omega.ndim == 2 and omega.shape[1] == 3, "directions must be M by 3")
    _require(angular_weights.shape == (omega.shape[0],), "angular-weight mismatch")
    _require(bool(np.all(angular_weights > 0.0)), "angular weights must be positive")
    _require(abs(float(np.sum(angular_weights)) - 1.0) < 2e-15, "angular weights must sum to one")
    norms = np.linalg.norm(omega, axis=1)
    _require(bool(np.all(np.abs(norms - 1.0) < 2e-15)), "carrier directions must be unit")
    return omega, angular_weights


def angular_moments(
    distribution: Any,
    directions: Any,
    weights: Any,
    carrier_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive Q, F, and P from one grey directional distribution."""

    population = _finite_array(distribution, None, "carrier distribution")
    omega, angular_weights = _validate_angular_quadrature(directions, weights)
    _require(
        population.ndim >= 1 and population.shape[0] == omega.shape[0], "population ray mismatch"
    )
    _require(bool(np.all(population >= 0.0)), "negative carrier population")
    _require(math.isfinite(carrier_speed) and carrier_speed > 0.0, "invalid carrier speed")
    scalar = np.einsum("a,a...->...", angular_weights, population)
    flux = carrier_speed * np.einsum("a,ai,a...->i...", angular_weights, omega, population)
    pressure = carrier_speed**2 * np.einsum(
        "a,ai,aj,a...->ij...", angular_weights, omega, omega, population
    )
    return scalar, flux, pressure


def kinetic_transport_rhs(
    distribution: Any,
    density_ratio: Any,
    spacing: float,
    directions: Any,
    weights: Any,
    *,
    carrier_speed: float,
    source_gain: float,
    absorption0: float,
    absorption_baryonic: float,
    collision_rates: Any,
    drift_velocity: Any | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the registered periodic grey transport equation.

    The centered streaming operator is used only for the target-free identity
    tests.  A data run must replace it with a frozen positive finite-volume
    transport solver and an independently converged open boundary.
    """

    population = _finite_array(distribution, None, "carrier distribution")
    source = _finite_array(density_ratio, None, "kinetic density ratio")
    omega, angular_weights = _validate_angular_quadrature(directions, weights)
    _require(
        population.ndim == 4 and population.shape[0] == omega.shape[0], "kinetic grid mismatch"
    )
    _require(population.shape[1:] == source.shape and source.ndim == 3, "kinetic source mismatch")
    _require(
        bool(np.all(population >= 0.0)) and bool(np.all(source >= 0.0)), "negative kinetic state"
    )
    parameters = (
        carrier_speed,
        source_gain,
        absorption0,
        absorption_baryonic,
    )
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in parameters) and carrier_speed > 0.0,
        "invalid kinetic parameter",
    )
    rates = _finite_array(collision_rates, None, "collision rates")
    if rates.ndim == 0:
        rates = np.full(omega.shape[0], float(rates))
    _require(rates.shape == (omega.shape[0],), "collision-rate mismatch")
    _require(bool(np.all(rates >= 0.0)), "negative collision rate")
    if drift_velocity is None:
        drift = np.zeros((3, *source.shape), dtype=float)
    else:
        drift = _finite_array(drift_velocity, (3, *source.shape), "kinetic drift")
    scalar, flux, _ = angular_moments(population, omega, angular_weights, carrier_speed)
    absorption = absorption0 + absorption_baryonic * source
    _require(omega.shape[0] == 6, "registered Markov collision requires six rays")
    _require(
        all(abs(float(rates[index] - rates[index ^ 1])) < 2e-15 for index in range(6)),
        "opposite-ray collision-rate mismatch",
    )
    base_rate = float(np.min(rates))
    collision = base_rate * (scalar[None, ...] - population)
    for ray in range(6):
        reversal_rate = 0.5 * (float(rates[ray]) - base_rate)
        collision[ray] += reversal_rate * (population[ray ^ 1] - population[ray])
    rhs = np.empty_like(population)
    for ray, direction in enumerate(omega):
        velocity = carrier_speed * direction[:, None, None, None] + drift
        streaming = periodic_divergence(velocity * population[ray], spacing)
        rhs[ray] = -streaming + source_gain * source - absorption * population[ray] + collision[ray]
    scalar_rhs = np.einsum("a,a...->...", angular_weights, rhs)
    total_flux = flux + scalar[None, ...] * drift
    moment_residual = (
        scalar_rhs
        + periodic_divergence(total_flux, spacing)
        - source_gain * source
        + absorption * scalar
    )
    scale = max(
        1.0,
        float(np.max(np.abs(scalar_rhs))),
        float(np.max(np.abs(periodic_divergence(total_flux, spacing)))),
        float(np.max(np.abs(source_gain * source))),
    )
    return rhs, {
        "zeroth_moment_max_relative_residual": float(np.max(np.abs(moment_residual)) / scale),
        "weighted_collision_source_sum": float(
            np.sum(np.einsum("a,a...->...", angular_weights, collision))
        ),
        "minimum_collision_rate": base_rate,
        "minimum_off_diagonal_transition_rate": min(
            base_rate / 6.0,
            *[0.5 * (float(rate) - base_rate) for rate in rates],
        ),
    }


def quadrupole_radiation_measure(quadrupole_history: Any, time_step: float) -> np.ndarray:
    """Return the interior finite-difference norm of d2Q/dt2 (a GR control)."""

    history = _finite_array(quadrupole_history, None, "quadrupole history")
    _require(
        history.ndim == 3 and history.shape[1:] == (3, 3) and history.shape[0] >= 3,
        "bad quadrupole history",
    )
    _require(math.isfinite(time_step) and time_step > 0.0, "invalid quadrupole time step")
    second = (history[2:] - 2.0 * history[1:-1] + history[:-2]) / time_step**2
    return np.linalg.norm(second, axis=(1, 2))


def carrier_quantum_energy(angular_frequency: float, hbar: float = 1.0) -> float:
    """Return E=hbar*omega without imposing a fixed minimum carrier energy."""

    _require(math.isfinite(angular_frequency) and angular_frequency > 0.0, "invalid frequency")
    _require(math.isfinite(hbar) and hbar > 0.0, "invalid hbar")
    return hbar * angular_frequency


def _fixture_density(
    size: int = 17, spacing: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = (np.arange(size, dtype=float) - 0.5 * (size - 1)) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    disk = np.exp(-0.5 * ((x / 0.75) ** 2 + (y / 0.75) ** 2 + (z / 0.16) ** 2))
    sphere = np.exp(-0.5 * ((x / 0.62) ** 2 + (y / 0.62) ** 2 + (z / 0.62) ** 2))
    bulged = 0.55 * disk + 0.45 * sphere
    return disk, sphere, bulged


def _rotation_matrix() -> np.ndarray:
    axis = np.array([1.0, -2.0, 0.7], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.731
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def target_free_suite() -> dict[str, Any]:
    disk, sphere, bulged = _fixture_density()
    spacing = 0.25
    disk_shape = disk_shape_from_density(disk, spacing)
    sphere_shape = disk_shape_from_density(sphere, spacing)
    bulged_shape = disk_shape_from_density(bulged, spacing)

    d_parallel = 0.5
    d_perpendicular = 1.5
    disk_tensor = diffusion_tensor(
        disk_shape["normal"], disk_shape["activation"], d_parallel, d_perpendicular
    )
    sphere_tensor = diffusion_tensor(
        sphere_shape["normal"], sphere_shape["activation"], d_parallel, d_perpendicular
    )
    polar = anisotropic_green(disk_shape["normal"], disk_tensor, 0.2)
    equatorial_direction = np.array([1.0, 0.0, 0.0])
    if abs(float(equatorial_direction @ disk_shape["normal"])) > 0.5:
        equatorial_direction = np.array([0.0, 1.0, 0.0])
    equatorial_direction -= (
        float(equatorial_direction @ disk_shape["normal"]) * disk_shape["normal"]
    )
    equatorial_direction /= np.linalg.norm(equatorial_direction)
    equatorial = anisotropic_green(equatorial_direction, disk_tensor, 0.2)
    reverse_tensor = diffusion_tensor(disk_shape["normal"], disk_shape["activation"], 1.5, 0.5)
    reverse_polar = anisotropic_green(disk_shape["normal"], reverse_tensor, 0.2)
    reverse_equatorial = anisotropic_green(equatorial_direction, reverse_tensor, 0.2)

    # An independently rotated weighted point cloud must rotate the full tensor.
    indices = np.indices(disk.shape, dtype=float)
    points = np.stack(
        [(indices[axis] - 0.5 * (disk.shape[axis] - 1)) * spacing for axis in range(3)], axis=-1
    ).reshape(-1, 3)
    weights = disk.reshape(-1)
    rotation = _rotation_matrix()
    rotated_shape = disk_shape_from_points(points @ rotation.T, weights)
    rotated_tensor = diffusion_tensor(
        rotated_shape["normal"], rotated_shape["activation"], d_parallel, d_perpendicular
    )
    rotation_error = float(np.linalg.norm(rotated_tensor - rotation @ disk_tensor @ rotation.T))

    load, integration = integrate_load(
        disk,
        spacing,
        disk_tensor,
        chi=0.08,
        source_gain=1.0,
        gamma0=0.2,
        beta=0.35,
        dt=0.0015,
        steps=320,
    )
    _absorbed, absorbed_integration = integrate_load(
        disk,
        spacing,
        disk_tensor,
        chi=0.08,
        source_gain=1.0,
        gamma0=0.2,
        beta=1.2,
        dt=0.0015,
        steps=320,
    )
    zero, zero_integration = integrate_load(
        np.zeros_like(disk),
        spacing,
        disk_tensor,
        chi=0.08,
        source_gain=1.0,
        gamma0=0.2,
        beta=0.35,
        dt=0.0015,
        steps=16,
    )
    middle = disk.shape[0] // 2
    polar_line = load[middle, middle, :]
    equatorial_line = load[:, middle, middle]
    polar_path = photon_path_log_redshift(polar_line, spacing, 1.0)
    equatorial_path = photon_path_log_redshift(equatorial_line, spacing, 1.0)
    acceleration = matter_acceleration_from_load(load, spacing)

    directions, angular_weights = disk_aligned_angular_quadrature(disk_shape["normal"])
    collision_rates, collision_summary = disk_aligned_collision_rates(
        disk_shape["activation"], d_parallel, d_perpendicular, 1.0
    )
    kinetic_axis = np.linspace(-1.0, 1.0, 7)
    kx, ky, kz = np.meshgrid(kinetic_axis, kinetic_axis, kinetic_axis, indexing="ij")
    kinetic_source = 0.15 + np.exp(-2.0 * (kx**2 + ky**2 + 2.0 * kz**2))
    isotropic_population = np.broadcast_to(0.25 + kinetic_source, (6, *kinetic_source.shape)).copy()
    isotropic_q, isotropic_f, isotropic_p = angular_moments(
        isotropic_population, directions, angular_weights, 1.0
    )
    pencil_population = np.zeros_like(isotropic_population)
    pencil_population[0] = 6.0 * (0.25 + kinetic_source)
    pencil_q, pencil_f, pencil_p = angular_moments(
        pencil_population, directions, angular_weights, 1.0
    )
    structured_population = np.stack(
        [
            (0.3 + kinetic_source) * (1.0 + 0.08 * ray) + 0.01 * (ray + 1) * (kx + 1.1)
            for ray in range(6)
        ]
    )
    kinetic_rhs, kinetic_budget = kinetic_transport_rhs(
        structured_population,
        kinetic_source,
        kinetic_axis[1] - kinetic_axis[0],
        directions,
        angular_weights,
        carrier_speed=1.0,
        source_gain=0.7,
        absorption0=0.2,
        absorption_baryonic=0.4,
        collision_rates=collision_rates,
        drift_velocity=density_gradient_drift(
            kinetic_source, kinetic_axis[1] - kinetic_axis[0], 0.08
        ),
    )
    _require(bool(np.all(np.isfinite(kinetic_rhs))), "nonfinite kinetic fixture")
    scattering_population = np.zeros_like(isotropic_population)
    scattering_population[0] = 6.0 * np.ones_like(kinetic_source)
    _, scattering_flux, _ = angular_moments(scattering_population, directions, angular_weights, 1.0)
    scattering_rhs, scattering_budget = kinetic_transport_rhs(
        scattering_population,
        np.zeros_like(kinetic_source),
        kinetic_axis[1] - kinetic_axis[0],
        directions,
        angular_weights,
        carrier_speed=1.0,
        source_gain=0.0,
        absorption0=0.0,
        absorption_baryonic=0.0,
        collision_rates=0.7,
    )
    scattering_q_rhs = np.einsum("a,a...->...", angular_weights, scattering_rhs)
    scattering_f_rhs = np.einsum("a,ai,a...->i...", angular_weights, directions, scattering_rhs)
    scattering_flux_error = float(np.max(np.abs(scattering_f_rhs + 0.7 * scattering_flux)))
    boundary_population = np.zeros_like(isotropic_population)
    boundary_population[4] = 6.0
    boundary_rhs, boundary_budget = kinetic_transport_rhs(
        boundary_population,
        np.zeros_like(kinetic_source),
        kinetic_axis[1] - kinetic_axis[0],
        directions,
        angular_weights,
        carrier_speed=1.0,
        source_gain=0.0,
        absorption0=0.0,
        absorption_baryonic=0.0,
        collision_rates=collision_rates,
    )
    empty_ray_minimum_derivative = float(np.min(boundary_rhs[[0, 1, 2, 3, 5]]))
    absorption_population = np.ones_like(isotropic_population)
    absorption_rhs, absorption_budget = kinetic_transport_rhs(
        absorption_population,
        np.zeros_like(kinetic_source),
        kinetic_axis[1] - kinetic_axis[0],
        directions,
        angular_weights,
        carrier_speed=1.0,
        source_gain=0.0,
        absorption0=0.35,
        absorption_baryonic=0.0,
        collision_rates=0.0,
    )
    absorption_q, _, _ = angular_moments(absorption_population, directions, angular_weights, 1.0)
    absorption_q_rhs = np.einsum("a,a...->...", angular_weights, absorption_rhs)
    absorption_error = float(np.max(np.abs(absorption_q_rhs + 0.35 * absorption_q)))
    closure_error = max(
        abs(
            1.0 / (3.0 * collision_summary["kappa_parallel"])
            - collision_summary["effective_parallel_diffusion"]
        ),
        abs(
            1.0 / (3.0 * collision_summary["kappa_perpendicular"])
            - collision_summary["effective_perpendicular_diffusion"]
        ),
    )
    fourth_moment = np.einsum(
        "a,ai,aj,ak,al->ijkl",
        angular_weights,
        directions,
        directions,
        directions,
        directions,
    )
    rotated_directions, rotated_weights = disk_aligned_angular_quadrature(
        rotation @ disk_shape["normal"]
    )
    rotated_fourth_moment = np.einsum(
        "a,ai,aj,ak,al->ijkl",
        rotated_weights,
        rotated_directions,
        rotated_directions,
        rotated_directions,
        rotated_directions,
    )
    expected_rotated_fourth_moment = np.einsum(
        "ia,jb,kc,ld,abcd->ijkl",
        rotation,
        rotation,
        rotation,
        rotation,
        fourth_moment,
    )
    fourth_moment_rotation_error = float(
        np.linalg.norm(rotated_fourth_moment - expected_rotated_fourth_moment)
    )
    radii = np.array([0.7, 1.0, 2.5, 9.0])
    luminosity = 2.3
    radial_flux = luminosity / (4.0 * math.pi * radii**2)
    enclosed_flux = 4.0 * math.pi * radii**2 * radial_flux
    quadrupole_static = np.broadcast_to(np.diag([2.0, -1.0, -1.0]), (33, 3, 3)).copy()
    time = np.arange(33, dtype=float) * 0.1
    oscillating = np.cos(1.7 * time)[:, None, None] * np.diag([2.0, -1.0, -1.0])
    static_radiation = quadrupole_radiation_measure(quadrupole_static, 0.1)
    dynamic_radiation = quadrupole_radiation_measure(oscillating, 0.1)
    low_energy = carrier_quantum_energy(1e-9)
    high_energy = carrier_quantum_energy(1e9)

    checks = [
        {
            "check_id": "SPHERE_DISABLES_ORIENTATION",
            "passed": bool(
                sphere_shape["activation"] < 1e-12
                and np.linalg.norm(sphere_tensor - np.trace(sphere_tensor) * np.eye(3) / 3.0)
                < 1e-12
            ),
            "diagnostic": float(sphere_shape["activation"]),
        },
        {
            "check_id": "THIN_DISK_AND_BULGE_ORDER",
            "passed": bool(
                disk_shape["activation"] > 0.75
                and 0.0 < bulged_shape["activation"] < disk_shape["activation"]
            ),
            "diagnostic": float(disk_shape["activation"] - bulged_shape["activation"]),
        },
        {
            "check_id": "TRACE_POSITIVITY",
            "passed": bool(np.all(np.linalg.eigvalsh(disk_tensor) > 0.0))
            and abs(float(np.trace(disk_tensor)) - 2.5) < 1e-12,
            "diagnostic": float(np.min(np.linalg.eigvalsh(disk_tensor))),
        },
        {
            "check_id": "DIFFUSION_CLOSURE_SO3_COVARIANCE",
            "passed": bool(rotation_error < 2e-12),
            "diagnostic": rotation_error,
        },
        {
            "check_id": "POLAR_ESCAPE_AND_REVERSAL",
            "passed": bool(polar > equatorial and reverse_polar < reverse_equatorial),
            "diagnostic": float((polar / equatorial) - (reverse_polar / reverse_equatorial)),
        },
        {
            "check_id": "ANISOTROPIC_GREEN_EQUATION",
            "passed": bool(anisotropic_green_residual([0.8, -0.5, 0.7], disk_tensor, 0.2) < 2e-6),
            "diagnostic": anisotropic_green_residual([0.8, -0.5, 0.7], disk_tensor, 0.2),
        },
        {
            "check_id": "FINITE_VOLUME_BUDGET_AND_POSITIVITY",
            "passed": bool(
                integration["maximum_budget_relative_residual"] < 2e-13
                and integration["minimum_pre_roundoff_load"] >= -1e-11
            ),
            "diagnostic": integration["maximum_budget_relative_residual"],
        },
        {
            "check_id": "ABSORPTION_LOWERS_LOAD",
            "passed": bool(absorbed_integration["total_load"] < integration["total_load"]),
            "diagnostic": float(absorbed_integration["total_load"] / integration["total_load"]),
        },
        {
            "check_id": "ZERO_SOURCE_ZERO_STATE",
            "passed": bool(
                float(np.max(np.abs(zero))) == 0.0 and zero_integration["total_load"] == 0.0
            ),
            "diagnostic": float(np.max(np.abs(zero))),
        },
        {
            "check_id": "SAME_STATE_OBSERVABLES",
            "passed": bool(
                math.isfinite(polar_path)
                and math.isfinite(equatorial_path)
                and bool(np.all(np.isfinite(acceleration)))
                and float(np.max(np.abs(acceleration))) > 0.0
            ),
            "diagnostic": float(polar_path / max(equatorial_path, 1e-30)),
        },
        {
            "check_id": "KINETIC_ISOTROPIC_AND_PENCIL_MOMENTS",
            "passed": bool(
                np.max(np.abs(isotropic_f)) < 2e-15
                and np.max(
                    np.abs(isotropic_p - np.eye(3).reshape(3, 3, 1, 1, 1) * isotropic_q / 3.0)
                )
                < 2e-15
                and np.max(np.abs(pencil_f[0] - pencil_q)) < 2e-15
                and np.max(np.abs(pencil_f[1:])) < 2e-15
                and np.max(np.abs(pencil_p[0, 0] - pencil_q)) < 2e-15
            ),
            "diagnostic": float(np.max(np.abs(isotropic_f))),
        },
        {
            "check_id": "KINETIC_ZEROTH_MOMENT_IDENTITY",
            "passed": bool(kinetic_budget["zeroth_moment_max_relative_residual"] < 2e-15),
            "diagnostic": kinetic_budget["zeroth_moment_max_relative_residual"],
        },
        {
            "check_id": "SCATTERING_CONSERVES_Q_AND_DAMPS_FLUX",
            "passed": bool(
                np.max(np.abs(scattering_q_rhs)) < 2e-15
                and scattering_flux_error < 2e-15
                and abs(scattering_budget["weighted_collision_source_sum"]) < 2e-13
            ),
            "diagnostic": scattering_flux_error,
        },
        {
            "check_id": "KINETIC_ABSORPTION_LOWERS_Q",
            "passed": bool(
                absorption_error < 2e-15
                and absorption_budget["zeroth_moment_max_relative_residual"] < 2e-15
            ),
            "diagnostic": absorption_error,
        },
        {
            "check_id": "MARKOV_COLLISION_BOUNDARY_POSITIVITY",
            "passed": bool(
                empty_ray_minimum_derivative >= 0.0
                and boundary_budget["minimum_off_diagonal_transition_rate"] >= 0.0
            ),
            "diagnostic": empty_ray_minimum_derivative,
        },
        {
            "check_id": "DISK_ALIGNED_COLLISION_TO_DIFFUSION_LIMIT",
            "passed": bool(
                closure_error < 2e-15
                and collision_summary["effective_perpendicular_diffusion"]
                > collision_summary["effective_parallel_diffusion"]
            ),
            "diagnostic": closure_error,
        },
        {
            "check_id": "FINITE_ANGULAR_DISCRETIZATION_LIMIT_RETAINED",
            "passed": bool(fourth_moment_rotation_error > 1e-6),
            "diagnostic": fourth_moment_rotation_error,
            "disposition": "RETAINED_BLOCK_DGKT01_FINITE_TRANSPORT_RESPONSE",
        },
        {
            "check_id": "INVERSE_SQUARE_GAUSS_AREA_GEOMETRY_CONTROL",
            "passed": bool(np.max(np.abs(enclosed_flux - luminosity)) < 2e-15),
            "diagnostic": float(np.max(np.abs(enclosed_flux - luminosity))),
        },
        {
            "check_id": "STATIC_FIELD_NOT_GR_RADIATION",
            "passed": bool(
                np.max(np.abs(static_radiation)) == 0.0 and np.max(dynamic_radiation) > 1.0
            ),
            "diagnostic": float(np.max(dynamic_radiation)),
        },
        {
            "check_id": "CARRIER_ENERGY_IS_FREQUENCY_DEPENDENT",
            "passed": bool(low_energy > 0.0 and high_energy / low_energy == 1e18),
            "diagnostic": float(high_energy / low_energy),
        },
    ]
    _require(all(row["passed"] for row in checks), "target-free suite failed")
    return {
        "checks": checks,
        "shape": {
            "disk_activation": float(disk_shape["activation"]),
            "bulged_activation": float(bulged_shape["activation"]),
            "sphere_activation": float(sphere_shape["activation"]),
            "disk_normal": [float(item) for item in disk_shape["normal"]],
            "diffusion_eigenvalues": [float(item) for item in np.linalg.eigvalsh(disk_tensor)],
        },
        "constant_coefficient_signature": {
            "polar_over_equatorial": float(polar / equatorial),
            "reversed_polar_over_equatorial": float(reverse_polar / reverse_equatorial),
            "green_relative_residual": anisotropic_green_residual(
                [0.8, -0.5, 0.7], disk_tensor, 0.2
            ),
        },
        "dynamic_signature": {
            "integration": integration,
            "strong_absorption_total_load": absorbed_integration["total_load"],
            "polar_path_log_redshift_proxy": polar_path,
            "equatorial_path_log_redshift_proxy": equatorial_path,
            "polar_over_equatorial_path_proxy": float(polar_path / max(equatorial_path, 1e-30)),
            "maximum_load_acceleration_proxy": float(np.max(np.linalg.norm(acceleration, axis=0))),
        },
        "kinetic_signature": {
            "quadrature_rays": 6,
            "zeroth_moment_max_relative_residual": kinetic_budget[
                "zeroth_moment_max_relative_residual"
            ],
            "scattering_flux_damping_error": scattering_flux_error,
            "kinetic_absorption_error": absorption_error,
            "empty_ray_minimum_collision_derivative": empty_ray_minimum_derivative,
            "minimum_off_diagonal_transition_rate": boundary_budget[
                "minimum_off_diagonal_transition_rate"
            ],
            "collision_to_diffusion_closure_error": closure_error,
            "finite_six_ray_fourth_moment_rotation_error": fourth_moment_rotation_error,
            "effective_parallel_diffusion": collision_summary["effective_parallel_diffusion"],
            "effective_perpendicular_diffusion": collision_summary[
                "effective_perpendicular_diffusion"
            ],
            "static_quadrupole_radiation_measure": float(np.max(static_radiation)),
            "dynamic_quadrupole_radiation_measure": float(np.max(dynamic_radiation)),
            "enclosed_free_streaming_flux_spread": float(np.ptp(enclosed_flux)),
            "carrier_energy_ratio_for_registered_frequency_pair": float(high_energy / low_energy),
        },
    }


def _bind_source_predecessor(config: Mapping[str, Any]) -> dict[str, str]:
    preflight = config["source_and_response_preflight"]
    result: dict[str, str] = {}
    for key in ("source_predecessor_config", "source_predecessor_receipt"):
        binding = preflight[key]
        path = REPO_ROOT / binding["path"]
        observed = file_sha256(path)
        _require(observed == binding["raw_sha256"], f"{key} drift")
        result[key] = observed
    return result


def build_receipt() -> dict[str, Any]:
    config = load_config()
    source_bindings = _bind_source_predecessor(config)
    suite = target_free_suite()
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-disk-polar-escape-load-receipt-4.0",
        "package_id": config["package_id"],
        "status": "PASS_DPEL_TARGET_FREE_SOURCE_READY__DGKT_MOMENT_CLOSURE_PASS_FINITE_TRANSPORT_BLOCKED__RESPONSES_UNOPENED",
        "decision": "ADVANCE_DPEL_ONLY_TO_REAL_SHAPED_SYNTHETIC_FIRST_PASS__KEEP_DGKT_FINITE_TRANSPORT_BLOCKED_ON_ANGULAR_CONVERGENCE",
        "formula_ids": [row["id"] for row in config["formula_catalog"]],
        "formula_catalog": config["formula_catalog"],
        "law": config["law"],
        "kinetic_parent_law": config["kinetic_parent_law"],
        "dimension_ledger": config["dimension_ledger"],
        "registered_parameterization": config["registered_parameterization"],
        "target_free_suite": suite,
        "source_and_response_preflight": config["source_and_response_preflight"],
        "source_binding_raw_sha256": source_bindings,
        "claim_boundary": config["claim_boundary"],
        "access_accounting": config["access_accounting"],
        "bindings": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode() + b"\n"
    return _atomic_no_clobber(OUTPUT_PATH, payload)


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(observed == expected, "receipt rebuild drift")
    _require(observed["content_sha256"] == _self_hash(observed), "receipt self-hash drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        print(write_receipt())
    else:
        receipt = check_receipt()
        print("VALID" if arguments.command == "check" else receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
