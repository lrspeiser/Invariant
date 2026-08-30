"""Target-free full-3D GP01 elliptic and bounded telegraph mechanics."""

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

from sigma_theory_compiler import gravity_gain_persistence_gp01_foundation as gp
from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base

CONFIG_PATH = Path("configs/open_gravity_gp01_full3d_dynamics_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_gp01_full3d_dynamics_v1.py")
TEST_PATH = Path("tests/test_open_gravity_gp01_full3d_dynamics_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-gp01-full3d-dynamics-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_gp01_full3d_dynamics_v1.json")
_CANONICAL_MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_gp01_full3d_dynamics_v1.py")
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_gp01_full3d_dynamics_v1.py")
_CANONICAL_OUTPUT_PATH = Path("runs/gravity/open-gravity-gp01-full3d-dynamics-v1/receipt.json")
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE_SHA256 = "153ddc0e5643dce4d0ad0b76738b34c1f9e061450fa67ad2b95460768aff3f4a"
_CONFIG_CONTENT_SHA256 = "aa06ef3c90bd7ac590ca40eacf49d37fc70156829a7d38f1748d08308571756e"
_SCHEMA = "invariant-open-gravity-gp01-full3d-dynamics-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-gp01-full3d-dynamics-receipt-1.0"


class GP01Full3DError(RuntimeError):
    """Raised whenever a GP01 full-3D gate fails closed."""


@dataclass(frozen=True)
class GainResult:
    gamma: np.ndarray
    relative_residual: float


@dataclass(frozen=True)
class CoupledResult:
    gamma: np.ndarray
    potential: np.ndarray
    gain_residual: float
    potential_residual: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GP01Full3DError(message)


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


def _path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    path = (_ROOT / expected).resolve()
    _require(path.is_relative_to(_ROOT), f"{label} escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GP01Full3DError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GP01Full3DError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "equations",
        "numerical_contract",
        "branch_dispositions",
        "required_gates",
        "remaining_blockers",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-gp01-full3d-dynamics-v1", "ID changed")
    _require(
        config["status"] == "FROZEN_TARGET_FREE_FULL3D_ELLIPTIC_AND_PARTIAL_TELEGRAPH",
        "status changed",
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(len(config["branch_dispositions"]) == 7, "branch count changed")
    _require(len(config["required_gates"]) == 15, "gate count changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_FILE_SHA256, "config bytes changed")
    config = _read_json(path, "GP01 full-3D config")
    validate_config(config)
    for binding in config["bindings"]:
        for artifact in binding["artifacts"]:
            expected = artifact["sha256"]
            _require(
                hashlib.sha256(_git_show(binding["commit"], artifact["path"])).hexdigest()
                == expected,
                f"committed {binding['role']} changed",
            )
            _require(
                file_sha256(_ROOT / artifact["path"]) == expected,
                f"working {binding['role']} changed",
            )
    return config


def solve_quasi_static_gain(
    target: np.ndarray, boundary: np.ndarray, spacing: float, *, length: float
) -> GainResult:
    source = np.asarray(target, dtype=np.float64)
    outer = np.asarray(boundary, dtype=np.float64)
    _require(source.shape == outer.shape and source.ndim == 3, "gain shapes changed")
    _require(np.all(np.isfinite(source) & (source >= 0.0)), "gain target invalid")
    _require(
        type(length) is float and length >= 0.0 and math.isfinite(length), "gain length invalid"
    )
    if length == 0.0:
        gamma = source.copy()
        gamma[0, :, :] = outer[0, :, :]
        gamma[-1, :, :] = outer[-1, :, :]
        gamma[:, 0, :] = outer[:, 0, :]
        gamma[:, -1, :] = outer[:, -1, :]
        gamma[:, :, 0] = outer[:, :, 0]
        gamma[:, :, -1] = outer[:, :, -1]
        return GainResult(gamma, 0.0)
    laplacian, boundary_vector = base._linear_matrix_and_rhs(np.zeros_like(source), outer, spacing)
    matrix = sparse.eye(laplacian.shape[0], format="csr") - length**2 * laplacian
    vector = source[1:-1, 1:-1, 1:-1].reshape(-1) - length**2 * boundary_vector
    interior = spsolve(matrix, vector)
    gamma = outer.copy()
    gamma[1:-1, 1:-1, 1:-1] = interior.reshape(
        gamma.shape[0] - 2, gamma.shape[1] - 2, gamma.shape[2] - 2
    )
    residual = gamma - length**2 * base._constant_laplacian(gamma, spacing) - source
    scale = max(float(np.max(np.abs(source[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return GainResult(gamma, relative)


def solve_coupled_potential(
    density: np.ndarray,
    gamma: np.ndarray,
    boundary: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, float]:
    _require(density.shape == gamma.shape == boundary.shape, "coupled shapes changed")
    coefficient = np.exp(-gamma)
    rhs = 4.0 * math.pi * np.asarray(density, dtype=np.float64)
    matrix, vector = base._variable_matrix_and_rhs(rhs, boundary, coefficient, spacing)
    interior = spsolve(matrix, vector)
    potential = np.asarray(boundary, dtype=np.float64).copy()
    potential[1:-1, 1:-1, 1:-1] = interior.reshape(
        potential.shape[0] - 2, potential.shape[1] - 2, potential.shape[2] - 2
    )
    residual = base._variable_divergence(potential, coefficient, spacing) - rhs
    scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return potential, relative


def baryonic_target(
    density: np.ndarray,
    grid: base.Grid3D,
    boundary: np.ndarray,
    *,
    n: int,
    gamma_max: float,
) -> tuple[np.ndarray, np.ndarray, base.SolverResult]:
    newton = base.solve_poisson(4.0 * math.pi * density, boundary, grid.spacing)
    acceleration = base.acceleration(newton.potential, grid.spacing)
    magnitude = np.sqrt(sum(component * component for component in acceleration))
    target = np.asarray(
        gp.bounded_gamma_target(
            magnitude, w=np.ones_like(magnitude), a_star=1.0, n=n, gamma_max=gamma_max
        )
    )
    return target, magnitude, newton


def solve_gp01_elliptic(
    density: np.ndarray,
    grid: base.Grid3D,
    *,
    n: int,
    gamma_max: float,
    length: float,
    baryonic_boundary: np.ndarray | None = None,
) -> CoupledResult:
    zero = np.zeros(grid.shape, dtype=np.float64)
    source_boundary = zero if baryonic_boundary is None else baryonic_boundary
    target, _magnitude, _newton = baryonic_target(
        density, grid, source_boundary, n=n, gamma_max=gamma_max
    )
    gain = solve_quasi_static_gain(target, zero, grid.spacing, length=length)
    potential, potential_residual = solve_coupled_potential(density, gain.gamma, zero, grid.spacing)
    return CoupledResult(gain.gamma, potential, gain.relative_residual, potential_residual)


def _gaussian(grid: base.Grid3D, sx: float, sy: float, sz: float, x0: float = 0.0) -> np.ndarray:
    density = np.exp(-0.5 * (((grid.x - x0) / sx) ** 2 + (grid.y / sy) ** 2 + (grid.z / sz) ** 2))
    return density / (float(np.sum(density)) * grid.spacing**3)


def _rotation_error(first: np.ndarray, rotated: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(first))), 1.0e-12)
    return float(np.max(np.abs(np.rot90(first, axes=(0, 1)) - rotated)) / scale)


def _curl_norm(potential: np.ndarray, spacing: float) -> float:
    gx, gy, gz = base.acceleration(potential, spacing)
    dgz_dy = np.gradient(gz, spacing, axis=1, edge_order=2)
    dgy_dz = np.gradient(gy, spacing, axis=2, edge_order=2)
    dgx_dz = np.gradient(gx, spacing, axis=2, edge_order=2)
    dgz_dx = np.gradient(gz, spacing, axis=0, edge_order=2)
    dgy_dx = np.gradient(gy, spacing, axis=0, edge_order=2)
    dgx_dy = np.gradient(gx, spacing, axis=1, edge_order=2)
    curl = np.sqrt((dgz_dy - dgy_dz) ** 2 + (dgx_dz - dgz_dx) ** 2 + (dgy_dx - dgx_dy) ** 2)
    return float(np.max(curl[2:-2, 2:-2, 2:-2]))


def telegraph_energy(
    gamma: np.ndarray, velocity: np.ndarray, spacing: float, *, length: float, tau: float
) -> float:
    gradients = np.gradient(gamma, spacing, edge_order=2)
    gradient_squared = sum(component * component for component in gradients)
    density = tau**2 * velocity**2 + length**2 * gradient_squared + gamma**2
    return float(0.5 * np.sum(density) * spacing**3)


def evolve_telegraph(
    gamma: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    spacing: float,
    *,
    length: float,
    tau: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    _require(length > 0.0 and tau > 0.0 and dt > 0.0, "telegraph scales invalid")
    _require(type(steps) is int and steps > 0, "telegraph steps invalid")
    state = np.asarray(gamma, dtype=np.float64).copy()
    speed = np.asarray(velocity, dtype=np.float64).copy()
    forcing = np.asarray(target, dtype=np.float64)
    _require(state.shape == speed.shape == forcing.shape, "telegraph shapes changed")
    energies = [telegraph_energy(state, speed, spacing, length=length, tau=tau)]
    for _ in range(steps):
        acceleration = (
            length**2 * base._constant_laplacian(state, spacing) - state + forcing - tau * speed
        ) / tau**2
        speed += dt * acceleration
        state += dt * speed
        for array in (state, speed):
            array[0, :, :] = 0.0
            array[-1, :, :] = 0.0
            array[:, 0, :] = 0.0
            array[:, -1, :] = 0.0
            array[:, :, 0] = 0.0
            array[:, :, -1] = 0.0
        energies.append(telegraph_energy(state, speed, spacing, length=length, tau=tau))
    _require(np.all(np.isfinite(state)) and np.all(np.isfinite(speed)), "telegraph diverged")
    return state, speed, energies


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    numerical = config["numerical_contract"]
    grid = base.make_grid(numerical["grid_nodes"], numerical["half_width"])
    zero = np.zeros(grid.shape, dtype=np.float64)
    h = grid.spacing
    gamma_max = float(numerical["Gamma_max"])
    gates: dict[str, dict[str, Any]] = {}

    length = 0.2
    exact = 0.3 * (1.0 - grid.x**2) * (1.0 - grid.y**2) * (1.0 - grid.z**2)
    manufactured_target = exact - length**2 * base._constant_laplacian(exact, h)
    manufactured = solve_quasi_static_gain(manufactured_target, zero, h, length=length)
    manufactured_error = float(np.max(np.abs(manufactured.gamma - exact)))
    gates["ELLIPTIC_MANUFACTURED"] = {
        "passed": manufactured.relative_residual < 1.0e-12 and manufactured_error < 1.0e-12,
        "metrics": {
            "relative_residual": manufactured.relative_residual,
            "maximum_solution_error": manufactured_error,
        },
    }

    source = _gaussian(grid, 0.45, 0.22, 0.15)
    target, _magnitude, _ = baryonic_target(source, grid, zero, n=2, gamma_max=gamma_max)
    local = solve_quasi_static_gain(target, zero, h, length=0.0)
    local_expected = target.copy()
    local_expected[[0, -1], :, :] = 0.0
    local_expected[:, [0, -1], :] = 0.0
    local_expected[:, :, [0, -1]] = 0.0
    local_error = float(np.max(np.abs(local.gamma - local_expected)))
    gates["LOCAL_L_ZERO_LIMIT"] = {
        "passed": local_error == 0.0,
        "metrics": {"maximum_error": local_error},
    }

    gain = solve_quasi_static_gain(target, zero, h, length=0.35)
    minimum = float(np.min(gain.gamma))
    maximum = float(np.max(gain.gamma))
    gates["ELLIPTIC_MAXIMUM_PRINCIPLE"] = {
        "passed": minimum >= -1.0e-14 and maximum <= gamma_max + 1.0e-14,
        "metrics": {"minimum": minimum, "maximum": maximum},
    }
    coefficient_minimum = float(np.min(np.exp(-gain.gamma)))
    gates["COEFFICIENT_ELLIPTICITY"] = {
        "passed": coefficient_minimum >= math.exp(-gamma_max) - 1.0e-14,
        "metrics": {
            "coefficient_minimum": coefficient_minimum,
            "declared_lower_bound": math.exp(-gamma_max),
        },
    }

    solution = solve_gp01_elliptic(source, grid, n=2, gamma_max=gamma_max, length=0.35)
    rotated_source = np.rot90(source, axes=(0, 1)).copy()
    rotated = solve_gp01_elliptic(rotated_source, grid, n=2, gamma_max=gamma_max, length=0.35)
    gamma_rotation = _rotation_error(solution.gamma, rotated.gamma)
    potential_rotation = _rotation_error(solution.potential, rotated.potential)
    gates["ROTATION_COVARIANCE"] = {
        "passed": max(gamma_rotation, potential_rotation) < 1.0e-9,
        "metrics": {
            "gamma_relative_error": gamma_rotation,
            "potential_relative_error": potential_rotation,
        },
    }

    curl = _curl_norm(solution.potential, h)
    gates["NONSYMMETRIC_CONSERVATIVE_CURL"] = {
        "passed": curl < 1.0e-11,
        "metrics": {"maximum_interior_curl": curl},
    }

    pair = _gaussian(grid, 0.18, 0.18, 0.18, -0.38) + _gaussian(grid, 0.18, 0.18, 0.18, 0.38)
    pair_target, pair_magnitude, _ = baryonic_target(pair, grid, zero, n=2, gamma_max=gamma_max)
    centre = tuple(size // 2 for size in grid.shape)
    gates["SADDLE_EXACT_NULL_TARGET"] = {
        "passed": pair_magnitude[centre] < 1.0e-12
        and abs(pair_target[centre] - gamma_max) < 1.0e-8,
        "metrics": {
            "central_field_magnitude": float(pair_magnitude[centre]),
            "central_target": float(pair_target[centre]),
        },
    }

    external_boundary = -0.25 * grid.x
    external_target, _, _ = baryonic_target(
        source, grid, external_boundary, n=2, gamma_max=gamma_max
    )
    external_difference = float(
        np.max(np.abs(external_target[2:-2, 2:-2, 2:-2] - target[2:-2, 2:-2, 2:-2]))
    )
    gates["EXTERNAL_FIELD_SENSITIVITY"] = {
        "passed": external_difference > 1.0e-3,
        "metrics": {"maximum_target_difference": external_difference},
    }

    pde_residual = gain.gamma - 0.35**2 * base._constant_laplacian(gain.gamma, h) - target
    pde_residual_max = float(np.max(np.abs(pde_residual[1:-1, 1:-1, 1:-1])))
    gates["SPATIAL_KERNEL_PDE_IDENTITY"] = {
        "passed": pde_residual_max < 1.0e-11,
        "metrics": {
            "maximum_operator_residual": pde_residual_max,
            "finite_box_convolution_claimed": False,
        },
    }

    tele_length = float(numerical["telegraph_L_g"])
    tau = float(numerical["telegraph_tau_g"])
    dt = float(numerical["telegraph_dt"])
    steps = int(numerical["telegraph_steps"])
    initial = 0.2 * np.exp(-8.0 * (grid.x**2 + grid.y**2 + grid.z**2))
    initial[[0, -1], :, :] = 0.0
    initial[:, [0, -1], :] = 0.0
    initial[:, :, [0, -1]] = 0.0
    _final, final_velocity, energies = evolve_telegraph(
        initial,
        zero,
        zero,
        h,
        length=tele_length,
        tau=tau,
        dt=dt,
        steps=steps,
    )
    gates["TELEGRAPH_ENERGY_DECAY"] = {
        "passed": energies[-1] < energies[0] and max(np.diff(energies)) < 2.0e-5,
        "metrics": {
            "initial_energy": energies[0],
            "final_energy": energies[-1],
            "largest_step_increase": float(max(np.diff(energies))),
            "final_velocity_norm": float(np.linalg.norm(final_velocity)),
        },
    }

    pulse_target = 0.3 * np.exp(-8.0 * (grid.x**2 + grid.y**2 + grid.z**2))
    pulse_state, pulse_velocity, _ = evolve_telegraph(
        zero,
        zero,
        pulse_target,
        h,
        length=tele_length,
        tau=tau,
        dt=dt,
        steps=100,
    )
    after, _, _ = evolve_telegraph(
        pulse_state,
        pulse_velocity,
        zero,
        h,
        length=tele_length,
        tau=tau,
        dt=dt,
        steps=1,
    )
    persistence = float(np.max(np.abs(after)))
    gates["TELEGRAPH_TEMPORAL_PERSISTENCE"] = {
        "passed": persistence > 1.0e-6,
        "metrics": {
            "post_source_state_maximum": persistence,
            "instantaneous_elliptic_zero_target_maximum": 0.0,
        },
    }

    characteristic_speed = tele_length / tau
    gates["TELEGRAPH_NECESSARY_SPEED"] = {
        "passed": characteristic_speed <= 1.0,
        "metrics": {
            "c_gamma_over_c": characteristic_speed,
            "common_cone_proved": False,
        },
    }

    dispositions = {row["id"]: row["status"] for row in config["branch_dispositions"]}
    gates["TRANSPORT_OBSTRUCTIONS_RETAINED"] = {
        "passed": dispositions["GP01-T1"].startswith("BLOCKED_")
        and dispositions["GP01-T2"].startswith("BLOCKED_"),
        "metrics": {"blocked_branches": 2, "promoted_branches": 0},
    }

    singularities = {str(n): gp.action_regularity_class(n) for n in numerical["n_grid"]}
    gates["ACTION_SINGULARITIES_RETAINED"] = {
        "passed": len(singularities) == 3
        and dispositions["GP01-ACTION-PLACEHOLDER"] == "INCOMPLETE_QUARANTINE",
        "metrics": singularities,
    }
    gates["ZERO_RESPONSE_ACCESS"] = {
        "passed": all(value == 0 for value in config["access_contract"].values()),
        "metrics": config["access_contract"],
    }

    _require(list(gates) == config["required_gates"], "gate order changed")
    for row in gates.values():
        row["passed"] = bool(row["passed"])
    _require(all(row["passed"] is True for row in gates.values()), "target-free gate failed")
    return {
        "gates": gates,
        "passed": len(gates),
        "failed": 0,
        "grid_nodes": grid.shape[0],
        "parameter_cells": len(numerical["n_grid"]) * len(numerical["L_g_grid"]),
        "instantaneous_baryonic_source_remains": True,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    suite = run_suite(config)
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_FULL3D_ELLIPTIC_PARTIAL_TELEGRAPH_TARGET_FREE_ONLY",
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
            "predecessors": config["bindings"],
        },
        "branch_dispositions": config["branch_dispositions"],
        "target_free_suite": suite,
        "remaining_blockers": config["remaining_blockers"],
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt is not an object")
    _require(payload == build_receipt(), "receipt is not reproducible")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


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
    validate_receipt_payload(_read_json(_output_path(), "GP01 full-3D receipt"))


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
                    "gates_passed": receipt["target_free_suite"]["passed"],
                    "full3d_elliptic": True,
                    "telegraph": "PARTIAL",
                    "observational_authority": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
