"""Shared target-free 3-D, history, ray, clock, wave, and adversarial fixtures."""

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

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base

CONFIG_PATH = Path("configs/open_gravity_common_3d_synthetic_universe_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_common_3d_synthetic_universe_v1.py")
TEST_PATH = Path("tests/test_open_gravity_common_3d_synthetic_universe_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-common-3d-synthetic-universe-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_common_3d_synthetic_universe_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_common_3d_synthetic_universe_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_common_3d_synthetic_universe_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-common-3d-synthetic-universe-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "d74af8797baaf8671844ef4a662682be4dc9af8df6edd3e15b894ddc5b98e57b"
_CONFIG_CONTENT_SHA256 = "4c405acd97c1b86f32e43c1d9040a82d7cb23bfad0fa81defe2a6e51652d251f"
_SCHEMA = "invariant-open-gravity-common-3d-synthetic-universe-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-common-3d-synthetic-universe-receipt-1.0"
_FIXTURE_IDS = tuple(
    f"F{index:02d}_{name}"
    for index, name in enumerate(
        (
            "POINT_MASS",
            "CONSTANT_DENSITY_SPHERE",
            "THIN_SHELL",
            "EXPONENTIAL_DISK",
            "THICK_DISK",
            "BULGE_DISK_GAS",
            "BAR_SPIRAL",
            "TRIAXIAL_GALAXY",
            "GALAXY_PAIR",
            "SATELLITE_EXTERNAL_FIELD",
            "FILAMENT",
            "WALL",
            "SPHERICAL_CLUSTER",
            "MERGING_CLUSTER",
            "COMPACT_BINARY",
            "SADDLE_EXACT_NULL",
            "VOID_CONTRAST",
            "HOMOGENEOUS_EXPANSION",
            "EVOLVING_SOURCE",
            "FLYBY",
            "MERGER",
            "SWITCH_ON_OFF",
            "LENSING_RAY_BUNDLE",
            "CLOCK_REDSHIFT_PATHS",
            "GW_PACKET_POLARIZATIONS",
            "ADVERSARIAL_TRANSFORMS",
        ),
        start=1,
    )
)


class SyntheticUniverseError(RuntimeError):
    """Raised when a fixture or packet fails closed."""


@dataclass(frozen=True)
class FixtureSet:
    grid: base.Grid3D
    sources: dict[str, np.ndarray]
    metadata: dict[str, dict[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticUniverseError(message)


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
        raise SyntheticUniverseError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=_ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SyntheticUniverseError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "numerical_contract",
        "fixtures",
        "required_gates",
        "anti_leakage",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-common-3d-synthetic-universe-v1", "ID changed")
    _require(config["status"] == "FROZEN_TARGET_FREE_COMMON_SYNTHETIC_UNIVERSE", "status changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(tuple(row["id"] for row in config["fixtures"]) == _FIXTURE_IDS, "fixtures changed")
    _require(len(config["required_gates"]) == 15, "gate inventory changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "synthetic universe config")
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


def _normalize(values: np.ndarray, spacing: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 3 and np.all(np.isfinite(array)), "source array invalid")
    _require(np.all(array >= 0.0), "positive source became negative")
    mass = float(np.sum(array) * spacing**3)
    _require(mass > 0.0 and math.isfinite(mass), "source mass invalid")
    return array / mass


def _gaussian(grid: base.Grid3D, sx: float, sy: float, sz: float, x0: float = 0.0) -> np.ndarray:
    value = np.exp(-0.5 * (((grid.x - x0) / sx) ** 2 + (grid.y / sy) ** 2 + (grid.z / sz) ** 2))
    return _normalize(value, grid.spacing)


def _moments(density: np.ndarray, grid: base.Grid3D) -> np.ndarray:
    mass_weights = density * grid.spacing**3
    mass = float(np.sum(mass_weights))
    centre = np.array(
        [np.sum(mass_weights * coordinate) / mass for coordinate in (grid.x, grid.y, grid.z)]
    )
    covariance = np.empty((3, 3), dtype=np.float64)
    coordinates = (grid.x - centre[0], grid.y - centre[1], grid.z - centre[2])
    for i in range(3):
        for j in range(3):
            covariance[i, j] = np.sum(mass_weights * coordinates[i] * coordinates[j]) / mass
    return np.linalg.eigvalsh(covariance)


def build_fixtures(config: Mapping[str, Any]) -> FixtureSet:
    numerical = config["numerical_contract"]
    grid = base.make_grid(numerical["grid_nodes"], numerical["half_width"])
    h = grid.spacing
    radius = np.sqrt(grid.x**2 + grid.y**2 + grid.z**2)
    cylindrical = np.sqrt(grid.x**2 + grid.y**2)
    azimuth = np.arctan2(grid.y, grid.x)
    sources: dict[str, np.ndarray] = {}
    sources["F01_POINT_MASS"] = _gaussian(grid, 0.07, 0.07, 0.07)
    sources["F02_CONSTANT_DENSITY_SPHERE"] = _normalize((radius <= 0.45).astype(float), h)
    sources["F03_THIN_SHELL"] = _normalize((np.abs(radius - 0.55) <= h / 2.0).astype(float), h)
    sources["F04_EXPONENTIAL_DISK"] = _normalize(
        np.exp(-cylindrical / 0.32) * np.exp(-np.abs(grid.z) / 0.045), h
    )
    sources["F05_THICK_DISK"] = _normalize(
        np.exp(-cylindrical / 0.32) * np.exp(-np.abs(grid.z) / 0.18), h
    )
    bulge = _gaussian(grid, 0.13, 0.13, 0.13)
    disk = sources["F04_EXPONENTIAL_DISK"]
    gas = _normalize(np.exp(-(((cylindrical - 0.55) / 0.18) ** 2) - (grid.z / 0.08) ** 2), h)
    sources["F06_BULGE_DISK_GAS"] = 0.4 * bulge + 0.45 * disk + 0.15 * gas
    modulation = (
        1.0
        + 0.22 * np.cos(2.0 * azimuth)
        + 0.10 * np.cos(2.0 * azimuth + 5.0 * np.log(cylindrical + 0.12))
    )
    sources["F07_BAR_SPIRAL"] = _normalize(disk * modulation, h)
    sources["F08_TRIAXIAL_GALAXY"] = _gaussian(grid, 0.48, 0.27, 0.13)
    sources["F09_GALAXY_PAIR"] = 0.5 * _gaussian(grid, 0.16, 0.13, 0.11, -0.38) + 0.5 * _gaussian(
        grid, 0.16, 0.13, 0.11, 0.38
    )
    sources["F10_SATELLITE_EXTERNAL_FIELD"] = _gaussian(grid, 0.15, 0.13, 0.11)
    sources["F11_FILAMENT"] = _normalize(
        np.exp(-0.5 * ((grid.y / 0.12) ** 2 + (grid.z / 0.12) ** 2)), h
    )
    sources["F12_WALL"] = _normalize(np.exp(-0.5 * (grid.z / 0.10) ** 2), h)
    sources["F13_SPHERICAL_CLUSTER"] = _normalize((1.0 + (radius / 0.35) ** 2) ** -1.5, h)
    sources["F14_MERGING_CLUSTER"] = 0.55 * _gaussian(
        grid, 0.30, 0.25, 0.22, -0.35
    ) + 0.45 * _gaussian(grid, 0.24, 0.21, 0.19, 0.43)
    sources["F15_COMPACT_BINARY"] = 0.5 * _gaussian(
        grid, 0.055, 0.055, 0.055, -0.25
    ) + 0.5 * _gaussian(grid, 0.055, 0.055, 0.055, 0.25)
    sources["F16_SADDLE_EXACT_NULL"] = 0.5 * _gaussian(
        grid, 0.13, 0.13, 0.13, -0.42
    ) + 0.5 * _gaussian(grid, 0.13, 0.13, 0.13, 0.42)
    void_profile = np.exp(-0.5 * (radius / 0.28) ** 2)
    sources["F17_VOID_CONTRAST"] = -void_profile + float(np.mean(void_profile))
    metadata = {
        "F10_SATELLITE_EXTERNAL_FIELD": {"external_acceleration": [0.25, 0.0, 0.0]},
        "F15_COMPACT_BINARY": {"component_masses": [0.5, 0.5], "separation": 0.5},
        "F17_VOID_CONTRAST": {"is_density_contrast": True},
    }
    return FixtureSet(grid, sources, metadata)


def _gate(passed: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "metrics": dict(metrics)}


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    fixtures = build_fixtures(config)
    grid = fixtures.grid
    h = grid.spacing
    gates: dict[str, dict[str, Any]] = {}
    configured = {row["id"]: row for row in config["fixtures"]}
    gates["FIXTURE_INVENTORY_AND_ORACLES"] = _gate(
        tuple(configured) == _FIXTURE_IDS and all(row["oracle"] for row in configured.values()),
        {
            "fixtures": len(configured),
            "fixtures_with_oracles": sum(bool(x["oracle"]) for x in configured.values()),
        },
    )

    positive_ids = tuple(
        f"F{index:02d}_{name}"
        for index, name in enumerate(
            (
                "POINT_MASS",
                "CONSTANT_DENSITY_SPHERE",
                "THIN_SHELL",
                "EXPONENTIAL_DISK",
                "THICK_DISK",
                "BULGE_DISK_GAS",
                "BAR_SPIRAL",
                "TRIAXIAL_GALAXY",
                "GALAXY_PAIR",
                "SATELLITE_EXTERNAL_FIELD",
                "FILAMENT",
                "WALL",
                "SPHERICAL_CLUSTER",
                "MERGING_CLUSTER",
                "COMPACT_BINARY",
                "SADDLE_EXACT_NULL",
            ),
            start=1,
        )
    )
    mass_errors = {
        fixture_id: abs(float(np.sum(fixtures.sources[fixture_id]) * h**3) - 1.0)
        for fixture_id in positive_ids
    }
    gates["SOURCE_MASS_NORMALIZATION"] = _gate(
        max(mass_errors.values()) < 2.0e-15,
        {"positive_sources": len(mass_errors), "maximum_mass_error": max(mass_errors.values())},
    )

    sphere_formula_error = max(
        abs((radius / 0.45) ** 3 - (radius**3 / 0.45**3)) for radius in (0.1, 0.2, 0.4)
    )
    zero = np.zeros(grid.shape)
    shell_solution = base.solve_poisson(4.0 * math.pi * fixtures.sources["F03_THIN_SHELL"], zero, h)
    shell_acceleration = base.acceleration(shell_solution.potential, h)
    centre = tuple(size // 2 for size in grid.shape)
    shell_centre = math.sqrt(sum(float(component[centre]) ** 2 for component in shell_acceleration))
    gates["SPHERE_SHELL_ANALYTIC_IDENTITIES"] = _gate(
        sphere_formula_error < 1.0e-14 and shell_centre < 1.0e-12,
        {
            "sphere_enclosed_mass_identity_error": sphere_formula_error,
            "shell_central_field": shell_centre,
        },
    )

    thin_moments = _moments(fixtures.sources["F04_EXPONENTIAL_DISK"], grid)
    thick_moments = _moments(fixtures.sources["F05_THICK_DISK"], grid)
    triaxial_moments = _moments(fixtures.sources["F08_TRIAXIAL_GALAXY"], grid)
    gates["DISK_THICKNESS_AND_TRIAXIAL_GEOMETRY"] = _gate(
        thick_moments[0] > thin_moments[0] and np.min(np.diff(triaxial_moments)) > 1.0e-3,
        {
            "thin_vertical_moment": float(thin_moments[0]),
            "thick_vertical_moment": float(thick_moments[0]),
            "triaxial_principal_moments": [float(value) for value in triaxial_moments],
        },
    )

    binary = fixtures.sources["F15_COMPACT_BINARY"]
    pair_com = float(np.sum(binary * grid.x) * h**3)
    saddle_solution = base.solve_poisson(
        4.0 * math.pi * fixtures.sources["F16_SADDLE_EXACT_NULL"], zero, h
    )
    saddle_acceleration = base.acceleration(saddle_solution.potential, h)
    saddle_centre = math.sqrt(
        sum(float(component[centre]) ** 2 for component in saddle_acceleration)
    )
    pair_reflection = float(
        np.max(
            np.abs(fixtures.sources["F09_GALAXY_PAIR"] - fixtures.sources["F09_GALAXY_PAIR"][::-1])
        )
    )
    gates["PAIR_BINARY_AND_SADDLE_SYMMETRIES"] = _gate(
        abs(pair_com) < 1.0e-14 and saddle_centre < 1.0e-12 and pair_reflection < 1.0e-14,
        {
            "binary_center_of_mass": pair_com,
            "saddle_central_field": saddle_centre,
            "pair_reflection_error": pair_reflection,
        },
    )

    filament_moments = _moments(fixtures.sources["F11_FILAMENT"], grid)
    wall_moments = _moments(fixtures.sources["F12_WALL"], grid)
    cluster_moments = _moments(fixtures.sources["F13_SPHERICAL_CLUSTER"], grid)
    merge_line = fixtures.sources["F14_MERGING_CLUSTER"][:, centre[1], centre[2]]
    maxima = sum(
        merge_line[index] > merge_line[index - 1] and merge_line[index] > merge_line[index + 1]
        for index in range(1, len(merge_line) - 1)
    )
    gates["FILAMENT_WALL_CLUSTER_TOPOLOGY"] = _gate(
        filament_moments[-1] > 4.0 * filament_moments[0]
        and wall_moments[-2] > 4.0 * wall_moments[0]
        and np.ptp(cluster_moments) < 1.0e-12
        and maxima == 2,
        {
            "filament_axis_ratio": float(filament_moments[-1] / filament_moments[0]),
            "wall_second_axis_ratio": float(wall_moments[-2] / wall_moments[0]),
            "cluster_moment_spread": float(np.ptp(cluster_moments)),
            "merger_axis_maxima": int(maxima),
        },
    )

    void = fixtures.sources["F17_VOID_CONTRAST"]
    gates["VOID_ZERO_MEAN_CONTRAST"] = _gate(
        abs(float(np.sum(void))) < 1.0e-12 and float(void[centre]) < 0.0,
        {"box_sum": float(np.sum(void)), "central_contrast": float(void[centre])},
    )

    times = np.linspace(
        0.0,
        config["numerical_contract"]["history_duration"],
        config["numerical_contract"]["history_steps"],
    )
    hubble = 0.17
    scale_factor = np.exp(hubble * times)
    recovered_hubble = np.gradient(np.log(scale_factor), times)
    gates["HOMOGENEOUS_BACKGROUND_IDENTITY"] = _gate(
        float(np.max(np.abs(recovered_hubble - hubble))) < 1.0e-13,
        {"H": hubble, "maximum_recovery_error": float(np.max(np.abs(recovered_hubble - hubble)))},
    )

    normalized_time = times / times[-1]
    width = 0.12 + 0.16 * normalized_time
    evolving_mass = np.ones_like(times)
    flyby_x = 0.8 * (2.0 * normalized_time - 1.0)
    flyby_symmetry = float(np.max(np.abs(flyby_x + flyby_x[::-1])))
    merger_mass = np.ones_like(times)
    envelope = np.where(
        (normalized_time > 0.2) & (normalized_time < 0.6),
        np.sin(math.pi * (normalized_time - 0.2) / 0.4) ** 2,
        0.0,
    )
    gates["SOURCE_HISTORY_CONSERVATION_AND_ORDER"] = _gate(
        np.all(evolving_mass == 1.0)
        and np.all(merger_mass == 1.0)
        and flyby_symmetry < 1.0e-14
        and envelope[0] == envelope[-1] == 0.0
        and width[-1] > width[0],
        {
            "evolving_mass_error": float(np.max(np.abs(evolving_mass - 1.0))),
            "flyby_time_reversal_error": flyby_symmetry,
            "merger_mass_error": float(np.max(np.abs(merger_mass - 1.0))),
            "switch_support_fraction": float(np.mean(envelope > 0.0)),
        },
    )

    impact = np.linspace(0.25, 1.25, config["numerical_contract"]["ray_count"])
    deflection = 4.0 / impact
    ray_invariant = deflection * impact
    gates["LENSING_RAY_ORACLE"] = _gate(
        float(np.max(np.abs(ray_invariant - 4.0))) < 1.0e-14,
        {
            "rays": len(impact),
            "maximum_inverse_impact_error": float(np.max(np.abs(ray_invariant - 4.0))),
        },
    )

    emitter_potential = -0.08
    observer_potential = -0.01
    endpoint_ratio = 1.0 + emitter_potential - observer_potential
    path_ratios = np.full(3, endpoint_ratio)
    gates["CLOCK_ENDPOINT_NOT_PATH_RULE"] = _gate(
        float(np.ptp(path_ratios)) == 0.0,
        {
            "frequency_ratio": endpoint_ratio,
            "tested_paths": 3,
            "path_spread": float(np.ptp(path_ratios)),
        },
    )

    plus = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    cross = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    wave_time = np.linspace(-4.0, 4.0, config["numerical_contract"]["gw_samples"])
    wave_packet = np.exp(-(wave_time**2)) * np.cos(8.0 * wave_time)
    tt_error = max(
        abs(float(np.trace(plus))),
        abs(float(np.trace(cross))),
        float(np.max(np.abs(plus[2]))),
        float(np.max(np.abs(cross[2]))),
    )
    gates["GW_TRANSVERSE_TRACELESS_POLARIZATIONS"] = _gate(
        tt_error == 0.0 and float(np.max(np.abs(wave_packet))) > 0.9,
        {"transverse_traceless_error": tt_error, "samples": len(wave_packet)},
    )

    exemplar = fixtures.sources["F08_TRIAXIAL_GALAXY"]
    rotated = np.rot90(exemplar, axes=(0, 1))
    rotation_mass_error = abs(float((np.sum(rotated) - np.sum(exemplar)) * h**3))
    eigen_rotation_error = float(np.max(np.abs(_moments(exemplar, grid) - _moments(rotated, grid))))
    labels = list(_FIXTURE_IDS)
    shuffled_labels = labels[1::2] + labels[::2]
    gates["ADVERSARIAL_ROTATION_AND_SHUFFLE_INVARIANCE"] = _gate(
        rotation_mass_error < 1.0e-15
        and eigen_rotation_error < 1.0e-14
        and set(shuffled_labels) == set(labels),
        {
            "rotation_mass_error": rotation_mass_error,
            "moment_eigenvalue_error": eigen_rotation_error,
            "label_set_preserved": True,
        },
    )

    sign_corrupt = -exemplar
    unit_corrupt_mass = float(np.sum(exemplar * 1000.0) * h**3)
    reversed_times = times[::-1]
    detected = {
        "sign": bool(np.min(sign_corrupt) < 0.0),
        "unit": bool(abs(unit_corrupt_mass - 1.0) > 1.0),
        "chronology": bool(np.any(np.diff(reversed_times) <= 0.0)),
    }
    gates["ADVERSARIAL_SIGN_UNIT_TIME_CORRUPTION_DETECTED"] = _gate(
        all(detected.values()), detected
    )
    gates["ZERO_RESPONSE_ACCESS"] = _gate(
        all(value == 0 for value in config["access_contract"].values()), config["access_contract"]
    )

    _require(list(gates) == config["required_gates"], "gate order changed")
    _require(all(row["passed"] is True for row in gates.values()), "synthetic gate failed")
    return {
        "fixtures": len(configured),
        "source_arrays": len(fixtures.sources),
        "gates": gates,
        "passed": len(gates),
        "failed": 0,
        "real_response_scoring_eligible": False,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_COMMON_SYNTHETIC_UNIVERSE_TARGET_FREE_ONLY",
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
        "suite": run_suite(config),
        "anti_leakage": config["anti_leakage"],
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
    validate_receipt_payload(_read_json(_output_path(), "synthetic universe receipt"))


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
                    "fixtures": receipt["suite"]["fixtures"],
                    "gates_passed": receipt["suite"]["passed"],
                    "observational_authority": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
