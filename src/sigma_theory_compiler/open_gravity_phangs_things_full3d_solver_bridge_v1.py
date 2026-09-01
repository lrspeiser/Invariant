"""Response-blind full-3D Newton/AQUAL/QUMOND fields for frozen source maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import map_coordinates

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as baseline
from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as source_builder,
)

CONFIG_PATH = Path("configs/open_gravity_phangs_things_full3d_solver_bridge_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_phangs_things_full3d_solver_bridge_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_phangs_things_full3d_solver_bridge_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-phangs-things-full3d-solver-bridge-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-phangs-things-full3d-solver-bridge-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-phangs-things-full3d-solver-bridge-receipt-1.0"
_PRIVATE_SCHEMA = "invariant-open-gravity-phangs-things-full3d-field-cubes-1.0"
_CONFIG_RAW_SHA256 = "ec4260bb669e6b750c6bde511254c33643208ac10075755810e34cfdbe2d0e15"
_CONFIG_CONTENT_SHA256 = "7a76e1cf295c3676580373a1fa396efc17649107035cef177a60e278b7e50bd1"
_MODULE_SEMANTIC_SHA256 = "f283712fbdfeb83520c6882ab49340357c425519960138df7ca5afaf14fb4092"
_TEST_RAW_SHA256 = "4280edd312bdb8413f217418818ec039c79a20b2a8bc781ce2dd63231181938a"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class Full3DBridgeError(RuntimeError):
    """Raised when a source, paper, numerical, or artifact gate fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Full3DBridgeError(message)


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


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    header = canonical_bytes({"shape": list(array.shape), "dtype": "<f8"})
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    candidate = (_ROOT / relative).resolve()
    _require(candidate == _ROOT or _ROOT in candidate.parents, "path escaped repository")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Full3DBridgeError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
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
        raise Full3DBridgeError("solver predecessor commit binding failed") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "RESPONSE_BLIND_REAL_SOURCE_FULL3D_SOLVER_PREFLIGHT_DEVELOPMENT_ONLY",
        "status changed",
    )
    _require(config["objects"] == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    _require(len(config["real_source_anchors"]) == 4, "real source anchors changed")
    _require(len(config["published_solver_anchors"]) == 5, "solver anchors changed")
    _require(
        all(row["url"].startswith("https://") for row in config["real_source_anchors"]),
        "real source anchor URL changed",
    )
    _require(
        all(row["url"].startswith("https://") for row in config["published_solver_anchors"]),
        "solver anchor URL changed",
    )
    volume = config["volume_contract"]
    _require(volume["primary_nodes_per_axis"] == 25, "primary grid changed")
    _require(volume["convergence_nodes_per_axis"] == 21, "convergence grid changed")
    _require(volume["solver_half_box_kpc"] == 30.0, "solver box changed")
    _require(volume["direct_volumetric_observation_claim"] is False, "3-D source overclaimed")
    solver = config["solver_contract"]
    _require(
        solver["operators"] == ["NEWTON", "AQUAL_SIMPLE_MU", "QUMOND_SIMPLE_NU"],
        "operators changed",
    )
    _require(solver["radial_profile_radii_kpc"] == [5.0, 10.0, 15.0], "radii changed")
    _require(solver["response_tuning_forbidden"] is True, "response tuning enabled")
    gates = config["gate_contract"]
    _require(gates["source_builder_benchmarks_must_all_pass"] is True, "source gate removed")
    _require(gates["solver_baseline_benchmarks_must_all_pass"] is True, "solver gate removed")
    _require(
        gates["primary_vs_convergence_radial_relative_difference_max"] == 0.08,
        "convergence gate changed",
    )
    boundary = config["scientific_boundary"]
    _require(boundary["source_files_opened_per_build"] == 21, "source access hidden")
    _require(boundary["source_bytes_opened_per_build"] == 74_030_400, "source bytes hidden")
    _require(boundary["response_files_opened"] == 0, "response files enabled")
    _require(boundary["response_rows_opened"] == 0, "response rows enabled")
    _require(boundary["scores_computed"] == 0, "scoring enabled")
    _require(boundary["network_calls"] == 0, "network enabled")
    claims = config["claim_boundary"]
    _require(claims["real_source_maps_compiled_to_full3d_density"] is True, "source claim lost")
    _require(
        claims["published_newton_aqual_qumond_operators_applied_on_same_grid"] is True,
        "solver claim lost",
    )
    for key in (
        "direct_3d_density_observed",
        "kinematic_fit_tested",
        "observational_preference_established",
        "full_source_systematic_grid_solved",
        "external_environment_included",
        "lensing_closure_established",
        "relativistic_completion_established",
        "novelty_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim ceiling exceeded: {key}")
    _require(
        config["output_contract"]["public_receipt"] == OUTPUT_PATH.as_posix(),
        "output path changed",
    )


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


def _validate_predecessors(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    bindings = config["predecessor_bindings"]
    for label in ("source_builder", "solver_baseline"):
        binding = bindings[label]
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), f"{label} artifact missing")
            _require(file_sha256(path) == artifact["sha256"], f"{label} artifact changed")
            if binding["commit"] is not None:
                _require(
                    hashlib.sha256(_git_show(binding["commit"], artifact["path"])).hexdigest()
                    == artifact["sha256"],
                    f"{label} commit artifact mismatch",
                )
        receipt_artifact = next(
            row for row in binding["artifacts"] if row["path"].endswith("receipt.json")
        )
        receipt = _read_json(_repo_path(receipt_artifact["path"]), f"{label} receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            f"{label} receipt content changed",
        )
        receipts[label] = receipt
    _require(
        all(receipts["source_builder"]["benchmarks"]["passed"].values()),
        "source builder published benchmark failed",
    )
    _require(
        receipts["solver_baseline"]["synthetic_suite"]["all_pass"] is True,
        "solver baseline benchmark failed",
    )
    return receipts["source_builder"], receipts["solver_baseline"]


def vertical_slab_fractions(
    coordinates_pc: np.ndarray, spacing_pc: float, height_pc: float
) -> np.ndarray:
    _require(height_pc > 0.0 and spacing_pc > 0.0, "invalid vertical deposition scales")
    lower = coordinates_pc - 0.5 * spacing_pc
    upper = coordinates_pc + 0.5 * spacing_pc
    fractions = 0.5 * (np.tanh(upper / height_pc) - np.tanh(lower / height_pc))
    fractions[[0, -1]] = 0.0
    total = float(fractions.sum())
    _require(total > 0.0 and np.isfinite(fractions).all(), "vertical deposition failed")
    return fractions / total


def deposit_surface_component(
    surface_density_msun_pc2: np.ndarray,
    x_pc: np.ndarray,
    y_pc: np.ndarray,
    source_spacing_pc: float,
    solver_coordinates_pc: np.ndarray,
    solver_spacing_pc: float,
    height_pc: float,
) -> tuple[np.ndarray, float]:
    _require(
        surface_density_msun_pc2.shape == x_pc.shape == y_pc.shape,
        "surface component shapes differ",
    )
    nodes = len(solver_coordinates_pc)
    plane_mass = np.zeros((nodes, nodes), dtype=np.float64)
    ix = np.rint((x_pc.ravel() - solver_coordinates_pc[0]) / solver_spacing_pc).astype(np.int64)
    iy = np.rint((y_pc.ravel() - solver_coordinates_pc[0]) / solver_spacing_pc).astype(np.int64)
    pixel_mass = (
        np.maximum(np.asarray(surface_density_msun_pc2, dtype=np.float64).ravel(), 0.0)
        * source_spacing_pc**2
    )
    valid = (ix > 0) & (ix < nodes - 1) & (iy > 0) & (iy < nodes - 1)
    _require(bool(np.all(valid)), "source pixel escaped the solver interior")
    np.add.at(plane_mass, (ix, iy), pixel_mass)
    fractions = vertical_slab_fractions(solver_coordinates_pc, solver_spacing_pc, height_pc)
    cell_volume_pc3 = solver_spacing_pc**3
    density = plane_mass[:, :, None] * fractions[None, None, :] / cell_volume_pc3
    return density, float(pixel_mass.sum())


def spherical_boundary(
    grid: baseline.Grid3D,
    mass_dimensionless: float,
    *,
    mond: bool,
    integration_samples: int,
) -> np.ndarray:
    _require(mass_dimensionless > 0.0, "boundary mass must be positive")
    _require(integration_samples >= 10_000, "boundary integral undersampled")
    radius = np.sqrt(grid.x * grid.x + grid.y * grid.y + grid.z * grid.z)
    radius_max = float(radius.max())
    samples = np.linspace(1.0e-5, radius_max, integration_samples, dtype=np.float64)
    g_newton = mass_dimensionless / (samples * samples)
    gravity = baseline.nu_simple(g_newton) * g_newton if mond else g_newton
    integral = cumulative_trapezoid(gravity, samples, initial=0.0)
    potential = np.interp(radius, samples, integral - integral[-1])
    boundary = np.zeros_like(radius)
    mask = np.zeros_like(radius, dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    boundary[mask] = potential[mask]
    return boundary


def radial_acceleration_profile(
    potential: np.ndarray,
    grid: baseline.Grid3D,
    *,
    half_box_kpc: float,
    radii_kpc: Sequence[float],
    azimuth_samples: int,
    a0_m_s2: float,
) -> list[dict[str, float]]:
    gx, gy, gz = baseline.acceleration(potential, grid.spacing)
    angles = np.linspace(0.0, 2.0 * math.pi, azimuth_samples, endpoint=False)
    rows: list[dict[str, float]] = []
    for radius_kpc in radii_kpc:
        radius_dimensionless = radius_kpc / half_box_kpc
        x = radius_dimensionless * np.cos(angles)
        y = radius_dimensionless * np.sin(angles)
        i = (x + 1.0) / grid.spacing
        j = (y + 1.0) / grid.spacing
        k = np.full_like(i, (len(grid.coordinates) - 1) / 2.0)
        ax = map_coordinates(gx, [i, j, k], order=1, mode="nearest", prefilter=False)
        ay = map_coordinates(gy, [i, j, k], order=1, mode="nearest", prefilter=False)
        az = map_coordinates(gz, [i, j, k], order=1, mode="nearest", prefilter=False)
        radial = -(ax * np.cos(angles) + ay * np.sin(angles))
        mean = float(np.mean(radial))
        rows.append(
            {
                "radius_kpc": float(radius_kpc),
                "radial_acceleration_over_a0": mean,
                "radial_acceleration_m_s2": mean * a0_m_s2,
                "azimuthal_rms_over_a0": float(np.std(radial)),
                "vertical_rms_over_a0": float(np.sqrt(np.mean(az * az))),
            }
        )
    return rows


def _solve_grid(
    config: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    rhalf_pc: float,
    nodes: int,
) -> dict[str, Any]:
    volume = config["volume_contract"]
    normal = config["normalization_contract"]
    source_cell = config["source_cell"]
    solver = config["solver_contract"]
    half_box_kpc = float(volume["solver_half_box_kpc"])
    half_box_pc = half_box_kpc * 1000.0
    grid = baseline.make_grid(nodes)
    coordinates_pc = grid.coordinates * half_box_pc
    spacing_pc = grid.spacing * half_box_pc
    hstar_pc = rhalf_pc / 1.678 * float(source_cell["stellar_height_over_exponential_scale"])
    hgas_pc = float(source_cell["gas_height_pc"])
    components: list[tuple[str, np.ndarray, float]] = [
        ("stellar", np.asarray(maps["stellar_fixed"], dtype=np.float64), hstar_pc),
        ("hi", np.asarray(maps["hi"], dtype=np.float64), hgas_pc),
        ("co", np.asarray(maps["co"], dtype=np.float64), hgas_pc),
    ]
    density_physical = np.zeros(grid.shape, dtype=np.float64)
    masses: dict[str, float] = {}
    for label, surface, height in components:
        component_density, component_mass = deposit_surface_component(
            surface,
            np.asarray(maps["x_pc"]),
            np.asarray(maps["y_pc"]),
            float(maps["dx_pc"]),
            coordinates_pc,
            spacing_pc,
            height,
        )
        density_physical += component_density
        masses[f"{label}_mass_msun"] = component_mass
    a0_m_s2 = float(normal["a0_m_s2"])
    a0_km2_s2_per_pc = a0_m_s2 * float(normal["pc_m"]) / 1.0e6
    g_pc = float(normal["G_pc_km2_s2_msun"])
    density_dimensionless = density_physical * g_pc * half_box_pc / a0_km2_s2_per_pc
    total_mass = float(sum(masses.values()))
    expected_dimensionless_mass = g_pc * total_mass / (a0_km2_s2_per_pc * half_box_pc**2)
    deposited_dimensionless_mass = float(density_dimensionless.sum() * grid.spacing**3)
    rhs = 4.0 * math.pi * density_dimensionless
    integration_samples = int(config["boundary_contract"]["radial_integral_samples"])
    newton_boundary = spherical_boundary(
        grid,
        expected_dimensionless_mass,
        mond=False,
        integration_samples=integration_samples,
    )
    mond_boundary = spherical_boundary(
        grid,
        expected_dimensionless_mass,
        mond=True,
        integration_samples=integration_samples,
    )
    newton = baseline.solve_poisson(rhs, newton_boundary, grid.spacing)
    _, qumond, _ = baseline.solve_qumond(
        rhs,
        newton_boundary,
        mond_boundary,
        grid.spacing,
        a0=1.0,
        nu_floor=float(solver["qumond_nu_floor"]),
    )
    aqual = baseline.solve_aqual(
        rhs,
        mond_boundary,
        grid.spacing,
        a0=1.0,
        mu_floor=float(solver["aqual_mu_floor"]),
        damping=float(solver["aqual_damping"]),
        max_iterations=int(solver["aqual_max_iterations"]),
        delta_tolerance=float(solver["aqual_delta_tolerance"]),
        residual_tolerance=float(solver["aqual_residual_tolerance"]),
    )
    potentials = {
        "NEWTON": newton.potential,
        "AQUAL_SIMPLE_MU": aqual.potential,
        "QUMOND_SIMPLE_NU": qumond.potential,
    }
    profiles = {
        label: radial_acceleration_profile(
            potential,
            grid,
            half_box_kpc=half_box_kpc,
            radii_kpc=solver["radial_profile_radii_kpc"],
            azimuth_samples=int(solver["azimuth_samples"]),
            a0_m_s2=a0_m_s2,
        )
        for label, potential in potentials.items()
    }
    return {
        "nodes": nodes,
        "grid_spacing_kpc": grid.spacing * half_box_kpc,
        "coordinates_dimensionless": grid.coordinates,
        "density_dimensionless": density_dimensionless,
        "potentials": potentials,
        "masses": masses,
        "total_mass_msun": total_mass,
        "expected_dimensionless_mass": expected_dimensionless_mass,
        "deposited_dimensionless_mass": deposited_dimensionless_mass,
        "dimensionless_mass_relative_error": abs(
            deposited_dimensionless_mass - expected_dimensionless_mass
        )
        / expected_dimensionless_mass,
        "solver_metrics": {
            "newton_relative_residual": newton.relative_residual,
            "qumond_relative_residual": qumond.relative_residual,
            "aqual_relative_residual": aqual.relative_residual,
            "aqual_converged": aqual.converged,
            "aqual_iterations": aqual.iterations,
        },
        "profiles": profiles,
        "field_hashes": {
            "density_dimensionless": array_sha256(density_dimensionless),
            **{
                f"{label.lower()}_potential": array_sha256(value)
                for label, value in potentials.items()
            },
        },
    }


def _relative_profile_difference(
    primary: Sequence[Mapping[str, float]], convergence: Sequence[Mapping[str, float]]
) -> float:
    _require(len(primary) == len(convergence), "profile grids differ")
    differences = []
    for first, second in zip(primary, convergence, strict=True):
        _require(first["radius_kpc"] == second["radius_kpc"], "profile radii differ")
        scale = max(abs(float(first["radial_acceleration_over_a0"])), 1.0e-12)
        differences.append(
            abs(
                float(first["radial_acceleration_over_a0"])
                - float(second["radial_acceleration_over_a0"])
            )
            / scale
        )
    return float(max(differences))


def _public_grid_summary(grid: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "nodes": grid["nodes"],
        "grid_spacing_kpc": grid["grid_spacing_kpc"],
        "masses": grid["masses"],
        "total_mass_msun": grid["total_mass_msun"],
        "expected_dimensionless_mass": grid["expected_dimensionless_mass"],
        "deposited_dimensionless_mass": grid["deposited_dimensionless_mass"],
        "dimensionless_mass_relative_error": grid["dimensionless_mass_relative_error"],
        "solver_metrics": grid["solver_metrics"],
        "profiles": grid["profiles"],
        "field_hashes": grid["field_hashes"],
    }


def _expected_source_masses(builder_receipt: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    expected: dict[str, dict[str, float]] = {}
    for row in builder_receipt["object_summaries"]:
        primary = row["primary_summary"]
        expected[row["object_id"]] = {
            "stellar_mass_msun": float(primary["stellar_mass_msun"]),
            "hi_mass_msun": float(primary["hi_helium_mass_msun"]),
            "co_mass_msun": float(primary["co_helium_mass_msun"]),
        }
    return expected


def _source_mass_relative_error(
    actual: Mapping[str, float], expected: Mapping[str, float]
) -> float:
    mapping = {
        "stellar_mass_msun": "stellar_mass_msun",
        "hi_mass_msun": "hi_mass_msun",
        "co_mass_msun": "co_mass_msun",
    }
    return float(
        max(
            abs(float(actual[key]) - float(expected[target]))
            / max(abs(float(expected[target])), 1.0e-30)
            for key, target in mapping.items()
        )
    )


def _gate_object(config: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, bool]:
    gates = config["gate_contract"]
    primary = row["primary"]
    convergence = row["convergence"]
    profiles = primary["profiles"]
    finite_positive = all(
        math.isfinite(float(point["radial_acceleration_over_a0"]))
        and float(point["radial_acceleration_over_a0"]) > 0.0
        for values in profiles.values()
        for point in values
    )
    return {
        "dimensionless_mass": primary["dimensionless_mass_relative_error"]
        <= gates["dimensionless_mass_relative_error_max"],
        "source_mass": row["source_mass_relative_error"] <= gates["source_mass_relative_error_max"],
        "newton_residual": primary["solver_metrics"]["newton_relative_residual"]
        <= gates["linear_relative_residual_max"],
        "qumond_residual": primary["solver_metrics"]["qumond_relative_residual"]
        <= gates["linear_relative_residual_max"],
        "aqual_residual": primary["solver_metrics"]["aqual_relative_residual"]
        <= gates["aqual_relative_residual_max"],
        "aqual_converged": primary["solver_metrics"]["aqual_converged"] is True,
        "resolution": max(row["profile_convergence_relative"].values())
        <= gates["primary_vs_convergence_radial_relative_difference_max"],
        "radial_acceleration": finite_positive,
        "convergence_grid_complete": convergence["nodes"]
        == config["volume_contract"]["convergence_nodes_per_axis"],
    }


def build_packet(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_config(config)
    builder_receipt, baseline_receipt = _validate_predecessors(config)
    acquisition_config, _ = source_builder._load_acquisition(dict(source_builder.load_config()))
    source_paths = source_builder._source_paths(acquisition_config)
    expected_masses = _expected_source_masses(builder_receipt)
    private: dict[str, Any] = {
        "schema": _PRIVATE_SCHEMA,
        "package_id": config["package_id"],
        "source_builder_receipt_content_sha256": builder_receipt["content_sha256"],
        "solver_baseline_receipt_content_sha256": baseline_receipt["content_sha256"],
        "objects": [],
        "scientific_boundary": config["scientific_boundary"],
    }
    public_objects: list[dict[str, Any]] = []
    source_config = source_builder.load_config()
    metadata_by_id = {row["object_id"]: row for row in source_config["objects"]}
    for object_id in config["objects"]:
        metadata = metadata_by_id[object_id]
        images = source_builder._load_object_images(object_id, source_paths)
        maps = source_builder._surface_maps(
            source_config,
            metadata,
            images,
            n=int(source_config["map_transform"]["primary_grid_pixels"]),
            box_kpc=float(source_config["map_transform"]["primary_box_kpc"]),
            beam=config["source_cell"]["beam"],
            use_sip=False,
        )
        rhalf_pc = source_builder._half_mass_radius_pc(
            maps["stellar_fixed"], maps["x_pc"], maps["y_pc"], float(maps["dx_pc"])
        )
        convergence_grid = _solve_grid(
            config,
            maps,
            rhalf_pc=rhalf_pc,
            nodes=int(config["volume_contract"]["convergence_nodes_per_axis"]),
        )
        primary_grid = _solve_grid(
            config,
            maps,
            rhalf_pc=rhalf_pc,
            nodes=int(config["volume_contract"]["primary_nodes_per_axis"]),
        )
        profile_convergence = {
            label: _relative_profile_difference(
                primary_grid["profiles"][label], convergence_grid["profiles"][label]
            )
            for label in config["solver_contract"]["operators"]
        }
        source_mass_error = _source_mass_relative_error(
            primary_grid["masses"], expected_masses[object_id]
        )
        public_row: dict[str, Any] = {
            "object_id": object_id,
            "rhalf_pc": rhalf_pc,
            "source_mass_relative_error": source_mass_error,
            "profile_convergence_relative": profile_convergence,
            "primary": _public_grid_summary(primary_grid),
            "convergence": _public_grid_summary(convergence_grid),
        }
        public_row["gates"] = _gate_object(config, public_row)
        public_row["all_gates_pass"] = all(public_row["gates"].values())
        _require(public_row["all_gates_pass"], f"three-dimensional source gate failed: {object_id}")
        public_objects.append(public_row)
        private["objects"].append(
            {
                "object_id": object_id,
                "nodes": primary_grid["nodes"],
                "coordinates_dimensionless": primary_grid["coordinates_dimensionless"].tolist(),
                "density_dimensionless": primary_grid["density_dimensionless"].tolist(),
                "newton_potential_dimensionless": primary_grid["potentials"]["NEWTON"].tolist(),
                "aqual_potential_dimensionless": primary_grid["potentials"][
                    "AQUAL_SIMPLE_MU"
                ].tolist(),
                "qumond_potential_dimensionless": primary_grid["potentials"][
                    "QUMOND_SIMPLE_NU"
                ].tolist(),
                "field_hashes": primary_grid["field_hashes"],
            }
        )
    private["object_field_root_sha256"] = content_sha256(private["objects"])
    private["content_sha256"] = content_sha256(private)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RESPONSE_BLIND_REAL_SOURCE_FULL3D_SOLVER_PREFLIGHT_DEVELOPMENT_ONLY",
        "decision": "SOURCE_ONLY_FIELDS_READY_FOR_FROZEN_SYSTEMATIC_EXPANSION_NOT_RESPONSE_SCORING",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "predecessor_bindings": config["predecessor_bindings"],
        "admission_rule": config["admission_rule"],
        "real_source_anchors": config["real_source_anchors"],
        "published_solver_anchors": config["published_solver_anchors"],
        "source_cell": config["source_cell"],
        "volume_contract": config["volume_contract"],
        "normalization_contract": config["normalization_contract"],
        "boundary_contract": config["boundary_contract"],
        "solver_contract": config["solver_contract"],
        "gate_contract": config["gate_contract"],
        "inherited_benchmarks": {
            "source_builder_all_pass": all(builder_receipt["benchmarks"]["passed"].values()),
            "solver_baseline_all_pass": baseline_receipt["synthetic_suite"]["all_pass"],
        },
        "objects": public_objects,
        "object_count": len(public_objects),
        "field_solution_count": len(public_objects) * 2 * 3,
        "all_object_gates_pass": all(row["all_gates_pass"] for row in public_objects),
        "private_field_path": config["output_contract"]["private_field_cubes"],
        "private_field_raw_sha256": hashlib.sha256(canonical_bytes(private)).hexdigest(),
        "private_field_content_sha256": private["content_sha256"],
        "private_object_field_root_sha256": private["object_field_root_sha256"],
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return private, receipt


def validate_packet(
    config: Mapping[str, Any], private: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    expected_private, expected_receipt = build_packet(config)
    _require(dict(private) == expected_private, "private field packet differs from rebuild")
    _require(dict(receipt) == expected_receipt, "receipt differs from rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
        temporary.unlink(missing_ok=True)


def write_packet() -> str:
    config = load_config()
    private, receipt = build_packet(config)
    private_status = _atomic_no_clobber(
        _repo_path(config["output_contract"]["private_field_cubes"]), canonical_bytes(private)
    )
    receipt_status = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return "CREATED" if "CREATED" in {private_status, receipt_status} else "EXISTING_IDENTICAL"


def check_packet() -> str:
    config = load_config()
    private_path = _repo_path(config["output_contract"]["private_field_cubes"])
    receipt_path = _repo_path(OUTPUT_PATH)
    _require(private_path.is_file() and receipt_path.is_file(), "packet output missing")
    private = _read_json(private_path, "private field packet")
    receipt = _read_json(receipt_path, "receipt")
    validate_packet(config, private, receipt)
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        print(write_packet())
    elif arguments.command == "check":
        print(check_packet())
    else:
        load_config()
        path = _repo_path(OUTPUT_PATH)
        if not path.is_file():
            print("UNBUILT_RESPONSE_BLIND_SOURCE_ONLY")
        else:
            receipt = _read_json(path, "receipt")
            print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
