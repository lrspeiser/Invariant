"""Target-free 3-D halo, weak-field MOG, and refracted-gravity comparators."""

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
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base

CONFIG_PATH = Path("configs/open_gravity_3d_halo_modified_gravity_comparators_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_halo_modified_gravity_comparators_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_3d_halo_modified_gravity_comparators_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-3d-halo-modified-gravity-comparators-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_3d_halo_modified_gravity_comparators_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_halo_modified_gravity_comparators_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_3d_halo_modified_gravity_comparators_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-3d-halo-modified-gravity-comparators-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE_SHA256 = "f590210669b67bd228b2cef9c843f37b0b9dea7c1564d988f492167ab2fd6bab"
_CONFIG_CONTENT_SHA256 = "c9763dc35861a42e2e96fe9b313ec5e409b3fcc341ef1f5a3df46e2dbf9523e2"
_SCHEMA = "invariant-open-gravity-3d-halo-modified-gravity-comparators-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-3d-halo-modified-gravity-comparators-receipt-1.0"
_COMPARATOR_IDS = (
    "HALO_NFW",
    "HALO_BURKERT",
    "HALO_EINASTO",
    "MOG_WEAK_FIELD",
    "REFRACTED_GRAVITY",
)


class ComparatorError(RuntimeError):
    """Raised when a comparator or package gate fails closed."""


@dataclass(frozen=True)
class EllipticResult:
    potential: np.ndarray
    relative_residual: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparatorError(message)


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
        raise ComparatorError(f"cannot read {label}") from exc
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
        raise ComparatorError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "primary_sources",
        "comparators",
        "numerical_contract",
        "required_target_free_gates",
        "blocked_comparators",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-3d-halo-modified-gravity-comparators-v1", "ID changed"
    )
    _require(config["status"] == "FROZEN_TARGET_FREE_3D_COMPARATOR_MECHANICS", "status changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(
        tuple(row["id"] for row in config["comparators"]) == _COMPARATOR_IDS,
        "comparators changed",
    )
    _require(len(config["required_target_free_gates"]) == 12, "gate count changed")
    _require(len(config["blocked_comparators"]) == 4, "blocked controls changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_FILE_SHA256, "config bytes changed")
    config = _read_json(path, "comparator config")
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


def nfw_enclosed_mass(radius: float, rho_s: float = 1.0, r_s: float = 0.35) -> float:
    _require(radius >= 0.0 and rho_s > 0.0 and r_s > 0.0, "invalid NFW parameters")
    x = radius / r_s
    return 4.0 * math.pi * rho_s * r_s**3 * (math.log1p(x) - x / (1.0 + x))


def _profile_density(
    profile: str,
    radius: np.ndarray | float,
    *,
    density_scale: float = 1.0,
    radius_scale: float = 0.35,
    alpha: float = 0.25,
) -> np.ndarray:
    _require(density_scale > 0.0 and radius_scale > 0.0, "invalid halo scale")
    _require(alpha > 0.0, "invalid Einasto alpha")
    radii = np.asarray(radius, dtype=np.float64)
    x = radii / radius_scale
    if profile == "HALO_NFW":
        safe = np.maximum(x, np.finfo(np.float64).tiny)
        return density_scale / (safe * (1.0 + safe) ** 2)
    if profile == "HALO_BURKERT":
        return density_scale / ((1.0 + x) * (1.0 + x * x))
    if profile == "HALO_EINASTO":
        return density_scale * np.exp(-(2.0 / alpha) * (x**alpha - 1.0))
    raise ComparatorError("unknown halo profile")


def halo_density_on_grid(
    profile: str,
    grid: base.Grid3D,
    *,
    density_scale: float = 1.0,
    radius_scale: float = 0.35,
    alpha: float = 0.25,
) -> np.ndarray:
    radius = np.sqrt(grid.x * grid.x + grid.y * grid.y + grid.z * grid.z)
    density = _profile_density(
        profile,
        radius,
        density_scale=density_scale,
        radius_scale=radius_scale,
        alpha=alpha,
    )
    centre = tuple(size // 2 for size in grid.shape)
    if profile == "HALO_NFW":
        cell_radius = grid.spacing / 2.0
        volume = 4.0 * math.pi * cell_radius**3 / 3.0
        density[centre] = nfw_enclosed_mass(cell_radius, density_scale, radius_scale) / volume
    return density


def solve_helmholtz(
    rhs: np.ndarray, boundary: np.ndarray, spacing: float, *, mass: float
) -> EllipticResult:
    _require(type(mass) is float and mass > 0.0 and math.isfinite(mass), "invalid Helmholtz mass")
    matrix, vector = base._linear_matrix_and_rhs(rhs, boundary, spacing)
    matrix = matrix - sparse.eye(matrix.shape[0], format="csr") * mass**2
    interior = spsolve(matrix, vector)
    potential = np.asarray(boundary, dtype=np.float64).copy()
    potential[1:-1, 1:-1, 1:-1] = interior.reshape(
        potential.shape[0] - 2, potential.shape[1] - 2, potential.shape[2] - 2
    )
    residual = base._constant_laplacian(potential, spacing) - mass**2 * potential - rhs
    scale = max(float(np.max(np.abs(rhs[1:-1, 1:-1, 1:-1]))), 1.0)
    relative = float(np.max(np.abs(residual[1:-1, 1:-1, 1:-1])) / scale)
    return EllipticResult(potential, relative)


def solve_mog_weak_field(
    density: np.ndarray,
    boundary: np.ndarray,
    spacing: float,
    *,
    alpha: float,
    mu: float,
) -> tuple[EllipticResult, base.SolverResult, EllipticResult]:
    _require(type(alpha) is float and alpha >= 0.0 and math.isfinite(alpha), "invalid MOG alpha")
    _require(type(mu) is float and mu > 0.0 and math.isfinite(mu), "invalid MOG mu")
    rhs = 4.0 * math.pi * np.asarray(density, dtype=np.float64)
    newton = base.solve_poisson(rhs, boundary, spacing)
    yukawa = solve_helmholtz(rhs, boundary, spacing, mass=mu)
    potential = (1.0 + alpha) * newton.potential - alpha * yukawa.potential
    return (
        EllipticResult(potential, max(newton.relative_residual, yukawa.relative_residual)),
        newton,
        yukawa,
    )


def mog_point_acceleration_ratio(radius: float, *, alpha: float, mu: float) -> float:
    _require(radius > 0.0 and alpha >= 0.0 and mu > 0.0, "invalid MOG point parameters")
    return 1.0 + alpha - alpha * (1.0 + mu * radius) * math.exp(-mu * radius)


def permittivity(
    density: np.ndarray, *, epsilon_0: float, rho_c: float, q_slope: float
) -> np.ndarray:
    _require(0.0 < epsilon_0 <= 1.0, "invalid epsilon_0")
    _require(rho_c > 0.0 and q_slope > 0.0, "invalid permittivity transition")
    safe = np.maximum(np.asarray(density, dtype=np.float64), np.finfo(np.float64).tiny)
    return epsilon_0 + (1.0 - epsilon_0) * (1.0 + np.tanh(q_slope * np.log(safe / rho_c))) / 2.0


def solve_refracted(
    density: np.ndarray,
    boundary: np.ndarray,
    spacing: float,
    *,
    epsilon_0: float,
    rho_c: float,
    q_slope: float,
) -> tuple[EllipticResult, np.ndarray]:
    coefficient = permittivity(density, epsilon_0=epsilon_0, rho_c=rho_c, q_slope=q_slope)
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
    return EllipticResult(potential, relative), coefficient


def _gaussian(grid: base.Grid3D, sx: float, sy: float, sz: float) -> np.ndarray:
    density = np.exp(-0.5 * ((grid.x / sx) ** 2 + (grid.y / sy) ** 2 + (grid.z / sz) ** 2))
    return density / (float(np.sum(density)) * grid.spacing**3)


def _max_rotation_error(first: np.ndarray, rotated: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(first))), 1.0e-12)
    return float(np.max(np.abs(np.rot90(first, axes=(0, 1)) - rotated)) / scale)


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    numerical = config["numerical_contract"]
    grid = base.make_grid(numerical["grid_nodes"], numerical["half_width"])
    zero = np.zeros(grid.shape, dtype=np.float64)
    h = grid.spacing
    gates: dict[str, dict[str, Any]] = {}

    radii = np.geomspace(1.0e-4, 3.0, 500)
    profile_metrics: dict[str, dict[str, float]] = {}
    for profile in ("HALO_NFW", "HALO_BURKERT", "HALO_EINASTO"):
        values = _profile_density(profile, radii)
        integral = quad(
            lambda radius, selected=profile: (
                4.0 * math.pi * radius**2 * float(_profile_density(selected, radius))
            ),
            0.0,
            1.0,
            epsabs=1.0e-11,
        )[0]
        _require(np.all(values > 0.0), f"{profile} density nonpositive")
        _require(np.all(np.diff(values) < 0.0), f"{profile} density not monotonic")
        _require(integral > 0.0 and math.isfinite(integral), f"{profile} mass invalid")
        profile_metrics[profile] = {
            "density_at_minimum_radius": float(values[0]),
            "density_at_maximum_radius": float(values[-1]),
            "mass_within_unit_radius": float(integral),
        }
    gates["HALO_POSITIVITY_MONOTONICITY_AND_MASS"] = {
        "passed": True,
        "metrics": profile_metrics,
    }

    nfw_numeric = quad(
        lambda radius: 4.0 * math.pi * radius**2 * float(_profile_density("HALO_NFW", radius)),
        0.0,
        1.0,
        epsabs=1.0e-12,
    )[0]
    nfw_exact = nfw_enclosed_mass(1.0)
    nfw_error = abs(nfw_numeric - nfw_exact) / nfw_exact
    gates["NFW_ANALYTIC_ENCLOSED_MASS"] = {
        "passed": nfw_error < 1.0e-10,
        "metrics": {"relative_error": nfw_error},
    }

    centre_density = {
        profile: float(
            halo_density_on_grid(profile, grid)[grid.shape[0] // 2][grid.shape[1] // 2][
                grid.shape[2] // 2
            ]
        )
        for profile in ("HALO_NFW", "HALO_BURKERT", "HALO_EINASTO")
    }
    gates["HALO_CUSP_CORE_DISTINCTION"] = {
        "passed": centre_density["HALO_NFW"] > centre_density["HALO_BURKERT"]
        and centre_density["HALO_EINASTO"] > centre_density["HALO_BURKERT"],
        "metrics": centre_density,
    }

    nfw_grid = halo_density_on_grid("HALO_NFW", grid)
    halo_poisson = base.solve_poisson(4.0 * math.pi * nfw_grid, zero, h)
    gates["POISSON_HALO_RESIDUAL"] = {
        "passed": halo_poisson.relative_residual < 1.0e-11,
        "metrics": {"relative_residual": halo_poisson.relative_residual},
    }

    manufactured = (1.0 - grid.x**2) * (1.0 - grid.y**2) * (1.0 - grid.z**2)
    mu = float(numerical["mog_parameters"]["mu"])
    manufactured_rhs = base._constant_laplacian(manufactured, h) - mu**2 * manufactured
    helmholtz = solve_helmholtz(manufactured_rhs, zero, h, mass=mu)
    helmholtz_error = float(np.max(np.abs(helmholtz.potential - manufactured)))
    gates["HELMHOLTZ_MANUFACTURED"] = {
        "passed": helmholtz.relative_residual < 1.0e-11 and helmholtz_error < 1.0e-11,
        "metrics": {
            "relative_residual": helmholtz.relative_residual,
            "maximum_solution_error": helmholtz_error,
        },
    }

    mog_alpha = float(numerical["mog_parameters"]["alpha"])
    near_ratio = mog_point_acceleration_ratio(1.0e-6, alpha=mog_alpha, mu=mu)
    far_ratio = mog_point_acceleration_ratio(50.0 / mu, alpha=mog_alpha, mu=mu)
    gates["MOG_POINT_KERNEL_LIMITS"] = {
        "passed": abs(near_ratio - 1.0) < 1.0e-10 and abs(far_ratio - (1.0 + mog_alpha)) < 1.0e-10,
        "metrics": {"near_ratio": near_ratio, "far_ratio": far_ratio},
    }

    bar = _gaussian(grid, 0.45, 0.2, 0.3)
    rotated_bar = np.rot90(bar, axes=(0, 1)).copy()
    mog_bar, _, _ = solve_mog_weak_field(bar, zero, h, alpha=mog_alpha, mu=mu)
    mog_rotated, _, _ = solve_mog_weak_field(rotated_bar, zero, h, alpha=mog_alpha, mu=mu)
    mog_rotation_error = _max_rotation_error(mog_bar.potential, mog_rotated.potential)
    gates["MOG_ROTATION_COVARIANCE"] = {
        "passed": mog_rotation_error < 1.0e-12,
        "metrics": {"relative_rotation_error": mog_rotation_error},
    }

    source = _gaussian(grid, 0.3, 0.3, 0.3)
    rhs = 4.0 * math.pi * source
    newton = base.solve_poisson(rhs, zero, h)
    epsilon_constant = 0.4
    matrix, vector = base._variable_matrix_and_rhs(
        rhs, zero, np.full(grid.shape, epsilon_constant), h
    )
    interior = spsolve(matrix, vector)
    constant_solution = zero.copy()
    constant_solution[1:-1, 1:-1, 1:-1] = interior.reshape(
        grid.shape[0] - 2, grid.shape[1] - 2, grid.shape[2] - 2
    )
    epsilon_limit_error = float(
        np.max(np.abs(constant_solution - newton.potential / epsilon_constant))
        / max(np.max(np.abs(newton.potential / epsilon_constant)), 1.0e-12)
    )
    gates["REFRACTED_CONSTANT_EPSILON_LIMIT"] = {
        "passed": epsilon_limit_error < 1.0e-12,
        "metrics": {"relative_error": epsilon_limit_error},
    }

    rg_parameters = numerical["refracted_parameters"]
    disk = _gaussian(grid, 0.5, 0.5, 0.12)
    sphere = _gaussian(grid, 0.3, 0.3, 0.3)
    disk_rg, _ = solve_refracted(
        disk,
        zero,
        h,
        epsilon_0=float(rg_parameters["epsilon_0"]),
        rho_c=float(rg_parameters["rho_c"]),
        q_slope=float(rg_parameters["Q"]),
    )
    sphere_rg, _ = solve_refracted(
        sphere,
        zero,
        h,
        epsilon_0=float(rg_parameters["epsilon_0"]),
        rho_c=float(rg_parameters["rho_c"]),
        q_slope=float(rg_parameters["Q"]),
    )
    disk_newton = base.solve_poisson(4.0 * math.pi * disk, zero, h)
    sphere_newton = base.solve_poisson(4.0 * math.pi * sphere, zero, h)
    centre = grid.shape[0] // 2
    sample = (centre + 2, centre, centre)
    disk_gain = abs(disk_rg.potential[sample] / disk_newton.potential[sample])
    sphere_gain = abs(sphere_rg.potential[sample] / sphere_newton.potential[sample])
    geometry_difference = float(abs(disk_gain - sphere_gain))
    gates["REFRACTED_DISK_SPHERE_GEOMETRY_DISCRIMINATOR"] = {
        "passed": geometry_difference > 1.0e-3,
        "metrics": {
            "disk_potential_gain": float(disk_gain),
            "sphere_potential_gain": float(sphere_gain),
            "absolute_gain_difference": geometry_difference,
        },
    }

    rotated_disk = np.rot90(disk, axes=(0, 1)).copy()
    rotated_rg, _ = solve_refracted(
        rotated_disk,
        zero,
        h,
        epsilon_0=float(rg_parameters["epsilon_0"]),
        rho_c=float(rg_parameters["rho_c"]),
        q_slope=float(rg_parameters["Q"]),
    )
    rg_rotation_error = _max_rotation_error(disk_rg.potential, rotated_rg.potential)
    gates["REFRACTED_ROTATION_COVARIANCE"] = {
        "passed": rg_rotation_error < 1.0e-12,
        "metrics": {"relative_rotation_error": rg_rotation_error},
    }

    failures = 0
    invalid_calls = (
        lambda: _profile_density("HALO_NFW", 1.0, density_scale=-1.0),
        lambda: _profile_density("HALO_BURKERT", 1.0, radius_scale=0.0),
        lambda: _profile_density("HALO_EINASTO", 1.0, alpha=0.0),
        lambda: solve_helmholtz(rhs, zero, h, mass=0),
        lambda: solve_mog_weak_field(source, zero, h, alpha=-1.0, mu=mu),
        lambda: permittivity(source, epsilon_0=0.0, rho_c=0.1, q_slope=1.0),
        lambda: permittivity(source, epsilon_0=0.2, rho_c=-0.1, q_slope=1.0),
        lambda: permittivity(source, epsilon_0=0.2, rho_c=0.1, q_slope=0.0),
    )
    for call in invalid_calls:
        try:
            call()
        except ComparatorError:
            failures += 1
    gates["DESIGNED_NEGATIVE_PARAMETER_FAILURES"] = {
        "passed": failures == len(invalid_calls),
        "metrics": {"rejected": failures, "attempted": len(invalid_calls)},
    }
    gates["ZERO_RESPONSE_ACCESS"] = {
        "passed": all(value == 0 for value in config["access_contract"].values()),
        "metrics": config["access_contract"],
    }
    _require(set(gates) == set(config["required_target_free_gates"]), "gate coverage changed")
    _require(all(row["passed"] is True for row in gates.values()), "target-free gate failed")
    return {"gates": gates, "passed": len(gates), "failed": 0}


def build_receipt() -> dict[str, Any]:
    config = load_config()
    suite = run_suite(config)
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_TARGET_FREE_3D_COMPARATOR_MECHANICS_ZERO_OBSERVATIONAL_AUTHORITY",
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
        "comparators": {
            "count": len(config["comparators"]),
            "ids": list(_COMPARATOR_IDS),
            "stream_sha256": content_sha256(config["comparators"]),
            "blocked_count": len(config["blocked_comparators"]),
        },
        "target_free_suite": suite,
        "primary_sources": config["primary_sources"],
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
    payload = _read_json(_output_path(), "comparator receipt")
    validate_receipt_payload(payload)


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
                    "comparators": receipt["comparators"]["count"],
                    "target_free_gates_passed": receipt["target_free_suite"]["passed"],
                    "scientific_rows": 0,
                    "observational_authority": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
