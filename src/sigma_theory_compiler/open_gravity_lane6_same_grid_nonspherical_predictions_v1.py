from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from sigma_theory_compiler import open_gravity_3d_halo_modified_gravity_comparators_v1 as halo
from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import open_gravity_gp01_full3d_dynamics_v1 as gp01
from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge
from sigma_theory_compiler import (
    open_gravity_phangs_things_full3d_source_systematics_v1 as source_screen,
)
from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as source_builder,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_3d_primary_benchmark_v1 as refracted,
)

_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/open_gravity_lane6_same_grid_nonspherical_predictions_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_lane6_same_grid_nonspherical_predictions_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_lane6_same_grid_nonspherical_predictions_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-lane6-same-grid-nonspherical-predictions-v1/receipt.json"
)
_CONFIG_RAW_SHA256 = "1e11ff6ac54b50fd0a70c52184a3b96f7dd5315efc37ed5d91129795c93be22c"
_CONFIG_CONTENT_SHA256 = "aa8e75c67e0c26d09ef40a579b23869d4a5866d19fb3d80b8824b00217112394"
_MODULE_SEMANTIC_SHA256 = "506a8405d2681675dafd0c78e2effdf5309c8c8fa93662a40225a0253582115e"
_TEST_RAW_SHA256 = "6d59bba68c3aa0d6e49e9241cfa8ae9aa380110832ab804512ae158982528bf2"
_RECEIPT_SCHEMA = "invariant-open-gravity-lane6-same-grid-nonspherical-predictions-receipt-1.0"
_MSUN_PC3_TO_G_CM3 = 1.98847e33 / 3.085677581491367e18**3


class Lane6NonsphericalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Lane6NonsphericalError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    for label in (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    ):
        text = __import__("re").sub(rf'({label}\s*=\s*)"[0-9a-f]{{64}}"', rf'\1"{"0" * 64}"', text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_path(relative: str | Path) -> Path:
    path = (_ROOT / Path(relative)).resolve()
    _require(path.is_relative_to(_ROOT.resolve()), "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lane6NonsphericalError(f"cannot read {label}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema")
        == "invariant-open-gravity-lane6-same-grid-nonspherical-predictions-1.0",
        "config schema changed",
    )
    _require(
        config.get("status") == "FROZEN_RESPONSE_BLIND_MODEL_LIFTED_2P5D_SAME_GRID_PREDICTIONS",
        "config status changed",
    )
    source = config.get("source_contract", {})
    _require(source.get("objects") == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    _require(
        source.get("geometry_label") == "MODEL_LIFTED_2P5D_SOURCE_IN_FULL_3D_FIELD_SOLVER",
        "geometry label changed",
    )
    _require(source.get("measured_3d_objects") == 0, "false measured-3D claim")
    _require(source.get("source_cells") == 225, "source cell count changed")
    _require(source.get("response_opening_forbidden") is True, "response boundary changed")
    ids = [row["id"] for row in config.get("mechanisms", [])]
    _require(
        ids
        == [
            "NEWTON",
            "NFW_SOURCE_MATCHED_CONTROL",
            "AQUAL_SIMPLE_MU",
            "QUMOND_SIMPLE_NU",
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN",
            "GP01_ELLIPTIC_N2_L035",
            "MASHHOON_RAHVAR_NLG_Q0",
            "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE",
        ],
        "mechanism inventory changed",
    )
    _require(config["projection_contract"]["radii_kpc"] == [5.0, 10.0, 15.0], "radii changed")
    _require(config["projection_contract"]["azimuth_samples"] == 48, "azimuth grid changed")
    _require(
        config["projection_contract"]["kinematic_response_used"] is False, "response use changed"
    )
    access = config.get("access_contract", {})
    for key in (
        "scientific_response_files",
        "scientific_response_rows",
        "scores",
        "parameters_fit",
        "network_calls_by_builder",
        "model_calls",
        "paid_calls",
    ):
        _require(access.get(key) == 0, f"access contract changed: {key}")
    _require(config.get("output_path") == OUTPUT_PATH.as_posix(), "output path changed")


def _validate_package_files() -> None:
    _require(file_sha256(_repo_path(CONFIG_PATH)) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(_repo_path(CONFIG_PATH), "config")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config content changed")
    _require(
        module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
        "module semantics changed",
    )
    _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "test bytes changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    config = _read_json(_repo_path(CONFIG_PATH), "config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    for binding in config["bindings"]:
        for artifact in binding["artifacts"]:
            _require(
                file_sha256(_repo_path(artifact["path"])) == artifact["sha256"],
                f"bound artifact changed: {artifact['path']}",
            )
    source_receipt = _read_json(
        _repo_path(
            "runs/gravity/open-gravity-phangs-things-full3d-source-systematics-v1/receipt.json"
        ),
        "source-systematics receipt",
    )
    _require(source_receipt["cell_count"] == 225, "source-systematics cell count changed")
    _require(
        source_receipt["numerical_counterexample_count"] == 1, "source counterexamples changed"
    )
    _require(
        source_receipt["scientific_boundary"]["response_rows_opened"] == 0,
        "predecessor response boundary changed",
    )
    return source_receipt


def _source_cell_id(
    beam_name: str, stellar: str, co_source: str, ratio: float, gas_height: float
) -> str:
    return f"{beam_name}:{stellar}:{co_source}:HS{ratio:.15g}:HG{gas_height:.15g}"


def _density_variant(
    bridge_config: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    stellar_surface: np.ndarray,
    co_surface: np.ndarray,
    hstar_pc: float,
    hgas_pc: float,
    half_box_kpc: float,
    nodes: int,
) -> dict[str, Any]:
    grid = base.make_grid(nodes)
    half_box_pc = half_box_kpc * 1000.0
    coordinates_pc = grid.coordinates * half_box_pc
    spacing_pc = grid.spacing * half_box_pc
    density_physical = np.zeros(grid.shape, dtype=np.float64)
    masses: dict[str, float] = {}
    for label, surface, height in (
        ("stellar", np.asarray(stellar_surface, dtype=np.float64), hstar_pc),
        ("hi", np.asarray(maps["hi"], dtype=np.float64), hgas_pc),
        ("co", np.asarray(co_surface, dtype=np.float64), hgas_pc),
    ):
        component, mass = bridge.deposit_surface_component(
            surface,
            np.asarray(maps["x_pc"]),
            np.asarray(maps["y_pc"]),
            float(maps["dx_pc"]),
            coordinates_pc,
            spacing_pc,
            height,
        )
        density_physical += component
        masses[f"{label}_mass_msun"] = mass
    normal = bridge_config["normalization_contract"]
    a0_pc = float(normal["a0_m_s2"]) * float(normal["pc_m"]) / 1.0e6
    density = density_physical * float(normal["G_pc_km2_s2_msun"]) * half_box_pc / a0_pc
    total_mass_msun = float(sum(masses.values()))
    expected_mass = float(normal["G_pc_km2_s2_msun"]) * total_mass_msun / (a0_pc * half_box_pc**2)
    return {
        "grid": grid,
        "density": density,
        "physical_density_g_cm3": density_physical * _MSUN_PC3_TO_G_CM3,
        "expected_mass": expected_mass,
        "total_mass_msun": total_mass_msun,
        "masses": masses,
        "half_box_kpc": half_box_kpc,
    }


def _iter_source_densities(
    config: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    source_config, acquisition, source_private = source_screen._load_source_builder_evidence()
    bridge_config = bridge.load_config()
    source_paths = source_builder._source_paths(acquisition)
    metadata_by_id = {row["object_id"]: row for row in source_config["objects"]}
    private_by_id = {row["object_id"]: row for row in source_private["objects"]}
    sealed = {(row["object_id"], row["cell_id"]): row for row in source_receipt["cells"]}
    axes = source_receipt["cell_contract"]["primary_axes"]
    for object_id in config["source_contract"]["objects"]:
        metadata = metadata_by_id[object_id]
        images = source_builder._load_object_images(object_id, source_paths)
        robust_maps = source_builder._surface_maps(
            source_config,
            metadata,
            images,
            n=256,
            box_kpc=40.0,
            beam="ROBUST_PRIMARY",
            use_sip=False,
        )
        rhalf_pc = source_builder._half_mass_radius_pc(
            robust_maps["stellar_fixed"],
            robust_maps["x_pc"],
            robust_maps["y_pc"],
            float(robust_maps["dx_pc"]),
        )
        rd_pc = rhalf_pc / 1.678
        maps_by_beam = {"ROBUST_PRIMARY": robust_maps}
        maps_by_beam["NATURAL_SENSITIVITY"] = source_builder._surface_maps(
            source_config,
            metadata,
            images,
            n=256,
            box_kpc=40.0,
            beam="NATURAL_SENSITIVITY",
            use_sip=False,
        )
        for beam_name, stellar, co_source, ratio, gas_height in itertools.product(
            axes["beam"],
            axes["stellar_mass_to_light"],
            axes["co_source"],
            axes["stellar_height_over_exponential_scale"],
            axes["gas_height_pc"],
        ):
            maps = maps_by_beam[beam_name]
            cell_id = _source_cell_id(
                beam_name, stellar, co_source, float(ratio), float(gas_height)
            )
            result = _density_variant(
                bridge_config,
                maps,
                stellar_surface=(
                    maps["stellar_fixed"] if stellar == "FIXED_0P6" else maps["stellar_color"]
                ),
                co_surface=(maps["co"] if co_source == "WITH_CO" else np.zeros_like(maps["co"])),
                hstar_pc=rd_pc * float(ratio),
                hgas_pc=float(gas_height),
                half_box_kpc=30.0,
                nodes=17,
            )
            yield {
                "object_id": object_id,
                "cell_id": cell_id,
                "cell_kind": "PRIMARY_CARTESIAN",
                "primary_cell": cell_id == private_by_id[object_id]["primary_cell_id"],
                "sealed": sealed[(object_id, cell_id)],
                **result,
            }
        control_physics = source_receipt["cell_contract"]["control_source_physics"]
        for control in source_receipt["cell_contract"]["controls_per_object"]:
            maps = source_builder._surface_maps(
                source_config,
                metadata,
                images,
                n=int(control["source_pixels"]),
                box_kpc=float(control["source_box_kpc"]),
                beam=control_physics["beam"],
                use_sip=bool(control["sip"]),
            )
            result = _density_variant(
                bridge_config,
                maps,
                stellar_surface=maps["stellar_fixed"],
                co_surface=maps["co"],
                hstar_pc=rd_pc * float(control_physics["stellar_height_over_exponential_scale"]),
                hgas_pc=float(control_physics["gas_height_pc"]),
                half_box_kpc=float(control["solver_half_box_kpc"]),
                nodes=17,
            )
            yield {
                "object_id": object_id,
                "cell_id": control["id"],
                "cell_kind": "NUMERICAL_SOURCE_CONTROL",
                "primary_cell": False,
                "sealed": sealed[(object_id, control["id"])],
                **result,
            }


def _sample_acceleration(
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: base.Grid3D,
    *,
    radius_dimensionless: float,
    azimuth_samples: int,
    z_dimensionless: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, azimuth_samples, endpoint=False)
    x = radius_dimensionless * np.cos(angles)
    y = radius_dimensionless * np.sin(angles)
    i = (x + 1.0) / grid.spacing
    j = (y + 1.0) / grid.spacing
    k = np.full_like(i, (z_dimensionless + 1.0) / grid.spacing)
    sampled = tuple(
        map_coordinates(field, [i, j, k], order=1, mode="nearest", prefilter=False)
        for field in components
    )
    return angles, sampled[0], sampled[1], sampled[2]


def _profile_from_acceleration(
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: base.Grid3D,
    *,
    half_box_kpc: float,
    radii_kpc: Sequence[float],
    azimuth_samples: int,
    a0_m_s2: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for radius_kpc in radii_kpc:
        radius_dimensionless = float(radius_kpc) / half_box_kpc
        angles, ax, ay, az = _sample_acceleration(
            components,
            grid,
            radius_dimensionless=radius_dimensionless,
            azimuth_samples=azimuth_samples,
            z_dimensionless=0.0,
        )
        radial = -(ax * np.cos(angles) + ay * np.sin(angles))
        mean = float(np.mean(radial))
        _, _, _, az_off = _sample_acceleration(
            components,
            grid,
            radius_dimensionless=radius_dimensionless,
            azimuth_samples=azimuth_samples,
            z_dimensionless=grid.spacing,
        )
        scale = max(abs(mean), 1.0e-15)
        harmonics = {
            f"m{mode}_over_mean": float(
                2.0 * abs(np.mean(radial * np.exp(-1j * mode * angles))) / scale
            )
            for mode in range(1, 5)
        }
        rows.append(
            {
                "radius_kpc": float(radius_kpc),
                "radial_acceleration_over_a0": mean,
                "radial_acceleration_m_s2": mean * a0_m_s2,
                "azimuthal_rms_over_a0": float(np.std(radial)),
                "vertical_midplane_rms_over_a0": float(np.sqrt(np.mean(az * az))),
                "vertical_one_cell_rms_over_a0": float(np.sqrt(np.mean(az_off * az_off))),
                **harmonics,
            }
        )
    return rows


def _profile_potential(
    potential: np.ndarray,
    grid: base.Grid3D,
    *,
    half_box_kpc: float,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return _profile_from_acceleration(
        base.acceleration(potential, grid.spacing),
        grid,
        half_box_kpc=half_box_kpc,
        radii_kpc=config["projection_contract"]["radii_kpc"],
        azimuth_samples=int(config["projection_contract"]["azimuth_samples"]),
        a0_m_s2=1.2e-10,
    )


def _rms_geometry(density: np.ndarray, grid: base.Grid3D) -> tuple[float, float, list[float]]:
    weights = np.asarray(density, dtype=np.float64)
    total = float(weights.sum())
    _require(total > 0.0, "empty source")
    coordinates = np.stack((grid.x, grid.y, grid.z), axis=-1)
    centre = np.sum(weights[..., None] * coordinates, axis=(0, 1, 2)) / total
    shifted = coordinates - centre
    flat_shifted = shifted.reshape(-1, 3)
    covariance = np.einsum("n,ni,nj->ij", weights.ravel(), flat_shifted, flat_shifted) / total
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = float(eigenvalues.sum())
    _require(trace > 0.0, "zero source size")
    numerator = (
        (eigenvalues[0] - eigenvalues[1]) ** 2
        + (eigenvalues[1] - eigenvalues[2]) ** 2
        + (eigenvalues[2] - eigenvalues[0]) ** 2
    )
    anisotropy = float(math.sqrt(float(numerator) / (2.0 * trace * trace)))
    return math.sqrt(trace), min(max(anisotropy, 0.0), 1.0), eigenvalues.tolist()


def _newton_boundary(grid: base.Grid3D, mass: float) -> np.ndarray:
    return bridge.spherical_boundary(grid, mass, mond=False, integration_samples=100_000)


def _solve_nfw_control(
    density: np.ndarray,
    grid: base.Grid3D,
    *,
    baryonic_mass: float,
    rms_radius: float,
) -> tuple[np.ndarray, dict[str, float]]:
    radius_scale = max(float(rms_radius), grid.spacing)
    raw = halo.halo_density_on_grid("HALO_NFW", grid, density_scale=1.0, radius_scale=radius_scale)
    radius = np.sqrt(grid.x * grid.x + grid.y * grid.y + grid.z * grid.z)
    raw_inside = float(raw[radius <= radius_scale].sum() * grid.spacing**3)
    _require(raw_inside > 0.0, "NFW normalization failed")
    halo_density = raw * (baryonic_mass / raw_inside)
    total_density = density + halo_density
    total_mass = float(total_density.sum() * grid.spacing**3)
    result = base.solve_poisson(
        4.0 * math.pi * total_density,
        _newton_boundary(grid, total_mass),
        grid.spacing,
    )
    return result.potential, {
        "relative_residual": result.relative_residual,
        "radius_scale_over_half_box": radius_scale,
        "halo_mass_inside_radius_scale_over_baryonic_mass": float(
            halo_density[radius <= radius_scale].sum() * grid.spacing**3 / baryonic_mass
        ),
        "halo_mass_in_box_over_baryonic_mass": float(
            halo_density.sum() * grid.spacing**3 / baryonic_mass
        ),
    }


def _gp01_target(potential: np.ndarray, grid: base.Grid3D) -> np.ndarray:
    acceleration = base.acceleration(potential, grid.spacing)
    magnitude = np.sqrt(sum(component * component for component in acceleration))
    target = np.full_like(magnitude, 1.5)
    positive = magnitude > 0.0
    driver = np.logaddexp(0.0, -2.0 * np.log(magnitude[positive]))
    target[positive] = 1.5 * np.tanh(driver / (2.0 * 2.0 * 1.5))
    return target


def _solve_gp01(
    density: np.ndarray,
    newton_potential: np.ndarray,
    newton_boundary: np.ndarray,
    grid: base.Grid3D,
) -> tuple[np.ndarray, dict[str, float]]:
    target = _gp01_target(newton_potential, grid)
    gain = gp01.solve_quasi_static_gain(target, np.zeros_like(target), grid.spacing, length=0.35)
    potential, residual = gp01.solve_coupled_potential(
        density, gain.gamma, newton_boundary, grid.spacing
    )
    return potential, {
        "gain_relative_residual": gain.relative_residual,
        "potential_relative_residual": residual,
        "gamma_minimum": float(gain.gamma.min()),
        "gamma_maximum": float(gain.gamma.max()),
    }


def _solve_refracted(
    density: np.ndarray,
    physical_density_g_cm3: np.ndarray,
    newton_boundary: np.ndarray,
    grid: base.Grid3D,
) -> tuple[np.ndarray, dict[str, float]]:
    epsilon = refracted.published_permittivity(
        physical_density_g_cm3,
        epsilon_0=0.661,
        rho_c=10.0**-24.54,
        q_slope=1.79,
    )
    potential, residual = refracted._solve_variable(
        4.0 * math.pi * density,
        newton_boundary / 0.661,
        epsilon,
        grid.spacing,
    )
    return potential, {
        "relative_residual": residual,
        "minimum_epsilon": float(epsilon.min()),
        "maximum_epsilon": float(epsilon.max()),
    }


def _solve_gqns(
    density: np.ndarray, grid: base.Grid3D
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rms_radius, anisotropy, eigenvalues = _rms_geometry(density, grid)
    length = max(rms_radius, grid.spacing)
    inverse_length = 1.0 / length
    dark = halo.solve_helmholtz(
        -(inverse_length**2) * anisotropy * density,
        np.zeros_like(density),
        grid.spacing,
        mass=float(inverse_length),
    )
    minimum = float(dark.potential.min())
    _require(minimum >= -1.0e-10, "GQNS effective density became materially negative")
    rho_dark = np.maximum(dark.potential, 0.0)
    effective = density + rho_dark
    effective_mass = float(effective.sum() * grid.spacing**3)
    field = base.solve_poisson(
        4.0 * math.pi * effective,
        _newton_boundary(grid, effective_mass),
        grid.spacing,
    )
    baryonic_mass = float(density.sum() * grid.spacing**3)
    return (
        field.potential,
        rho_dark,
        {
            "anisotropy_A_Q": anisotropy,
            "kernel_length_over_half_box": length,
            "second_moment_eigenvalues": eigenvalues,
            "effective_nonlocal_mass_over_baryonic_mass_in_box": float(
                rho_dark.sum() * grid.spacing**3 / baryonic_mass
            ),
            "helmholtz_relative_residual": dark.relative_residual,
            "poisson_relative_residual": field.relative_residual,
            "minimum_unclipped_effective_density": minimum,
        },
    )


def _nlg_delta_factor(radius_kpc: np.ndarray) -> np.ndarray:
    values = np.asarray(radius_kpc, dtype=np.float64)
    return 10.94 * (1.0 - np.exp(-0.059 * values) * (1.0 + 0.5 * 0.059 * values))


def _nlg_correction(
    density: np.ndarray,
    grid: base.Grid3D,
    targets: np.ndarray,
    *,
    half_box_kpc: float,
) -> np.ndarray:
    positions = np.stack((grid.x.ravel(), grid.y.ravel(), grid.z.ravel()), axis=1)
    masses = density.ravel() * grid.spacing**3
    keep = masses > 0.0
    positions = positions[keep]
    masses = masses[keep]
    output = np.zeros_like(targets, dtype=np.float64)
    for index, target in enumerate(targets):
        displacement = target - positions
        radius = np.linalg.norm(displacement, axis=1)
        active = radius > 1.0e-14
        factor = _nlg_delta_factor(radius[active] * half_box_kpc)
        output[index] = -np.sum(
            (masses[active] * factor / radius[active] ** 3)[:, None] * displacement[active],
            axis=0,
        )
    return output


def _nlg_profile(
    density: np.ndarray,
    newton_components: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: base.Grid3D,
    *,
    half_box_kpc: float,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = int(config["projection_contract"]["azimuth_samples"])
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    for radius_kpc in config["projection_contract"]["radii_kpc"]:
        radius = float(radius_kpc) / half_box_kpc
        for_z = []
        for z in (0.0, grid.spacing):
            targets = np.column_stack(
                (
                    radius * np.cos(angles),
                    radius * np.sin(angles),
                    np.full(samples, z),
                )
            )
            _, nx, ny, nz = _sample_acceleration(
                newton_components,
                grid,
                radius_dimensionless=radius,
                azimuth_samples=samples,
                z_dimensionless=z,
            )
            correction = _nlg_correction(density, grid, targets, half_box_kpc=half_box_kpc)
            for_z.append(np.column_stack((nx, ny, nz)) + correction)
        midplane, offplane = for_z
        radial = -(midplane[:, 0] * np.cos(angles) + midplane[:, 1] * np.sin(angles))
        mean = float(radial.mean())
        scale = max(abs(mean), 1.0e-15)
        rows.append(
            {
                "radius_kpc": float(radius_kpc),
                "radial_acceleration_over_a0": mean,
                "radial_acceleration_m_s2": mean * 1.2e-10,
                "azimuthal_rms_over_a0": float(radial.std()),
                "vertical_midplane_rms_over_a0": float(np.sqrt(np.mean(midplane[:, 2] ** 2))),
                "vertical_one_cell_rms_over_a0": float(np.sqrt(np.mean(offplane[:, 2] ** 2))),
                **{
                    f"m{mode}_over_mean": float(
                        2.0 * abs(np.mean(radial * np.exp(-1j * mode * angles))) / scale
                    )
                    for mode in range(1, 5)
                },
            }
        )
    return rows


def _profile_replay_error(
    actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> float:
    _require(len(actual) == len(expected), "profile length changed")
    return float(
        max(
            abs(
                float(row["radial_acceleration_over_a0"])
                - float(target["radial_acceleration_over_a0"])
            )
            / max(abs(float(target["radial_acceleration_over_a0"])), 1.0e-15)
            for row, target in zip(actual, expected, strict=True)
        )
    )


def _finite_positive(profiles: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    return all(
        math.isfinite(float(point["radial_acceleration_over_a0"]))
        and float(point["radial_acceleration_over_a0"]) > 0.0
        for rows in profiles.values()
        for point in rows
    )


def _solve_real_cell(config: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    grid = item["grid"]
    density = item["density"]
    mass = float(item["expected_mass"])
    boundary = _newton_boundary(grid, mass)
    mond_boundary = bridge.spherical_boundary(grid, mass, mond=True, integration_samples=100_000)
    newton = base.solve_poisson(4.0 * math.pi * density, boundary, grid.spacing)
    _, qumond, _ = base.solve_qumond(
        4.0 * math.pi * density,
        boundary,
        mond_boundary,
        grid.spacing,
        a0=1.0,
        nu_floor=1.0e-6,
    )
    aqual = base.solve_aqual(
        4.0 * math.pi * density,
        mond_boundary,
        grid.spacing,
        a0=1.0,
        mu_floor=1.0e-6,
        damping=0.5,
        max_iterations=500,
        delta_tolerance=1.0e-8,
        residual_tolerance=2.0e-7,
    )
    rms_radius, anisotropy, _ = _rms_geometry(density, grid)
    nfw_potential, nfw_metrics = _solve_nfw_control(
        density, grid, baryonic_mass=mass, rms_radius=rms_radius
    )
    rg_potential, rg_metrics = _solve_refracted(
        density, item["physical_density_g_cm3"], boundary, grid
    )
    gp_potential, gp_metrics = _solve_gp01(density, newton.potential, boundary, grid)
    gq_potential, gq_density, gq_metrics = _solve_gqns(density, grid)
    profiles: dict[str, list[dict[str, Any]]] = {
        "NEWTON": _profile_potential(
            newton.potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
        "NFW_SOURCE_MATCHED_CONTROL": _profile_potential(
            nfw_potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
        "AQUAL_SIMPLE_MU": _profile_potential(
            aqual.potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
        "QUMOND_SIMPLE_NU": _profile_potential(
            qumond.potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
        "REFRACTED_GRAVITY_DISKMASS_MEDIAN": _profile_potential(
            rg_potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
        "GP01_ELLIPTIC_N2_L035": _profile_potential(
            gp_potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
        "MASHHOON_RAHVAR_NLG_Q0": _nlg_profile(
            density,
            base.acceleration(newton.potential, grid.spacing),
            grid,
            half_box_kpc=item["half_box_kpc"],
            config=config,
        ),
        "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE": _profile_potential(
            gq_potential, grid, half_box_kpc=item["half_box_kpc"], config=config
        ),
    }
    density_hash = bridge.array_sha256(density)
    replay_error = _profile_replay_error(profiles["NEWTON"], item["sealed"]["profiles"]["NEWTON"])
    gates = {
        "density_hash_replay": density_hash == item["sealed"]["field_hashes"]["density"],
        "newton_profile_replay": replay_error
        <= config["gate_contract"]["newton_profile_replay_relative_max"],
        "newton_residual": newton.relative_residual
        <= config["gate_contract"]["linear_relative_residual_max"],
        "aqual_converged": aqual.converged,
        "aqual_residual": aqual.relative_residual
        <= config["gate_contract"]["variable_relative_residual_max"],
        "qumond_residual": qumond.relative_residual
        <= config["gate_contract"]["linear_relative_residual_max"],
        "nfw_residual": nfw_metrics["relative_residual"]
        <= config["gate_contract"]["linear_relative_residual_max"],
        "refracted_residual": rg_metrics["relative_residual"]
        <= config["gate_contract"]["variable_relative_residual_max"],
        "gp01_gain_residual": gp_metrics["gain_relative_residual"]
        <= config["gate_contract"]["linear_relative_residual_max"],
        "gp01_potential_residual": gp_metrics["potential_relative_residual"]
        <= config["gate_contract"]["linear_relative_residual_max"],
        "gqns_helmholtz_residual": gq_metrics["helmholtz_relative_residual"]
        <= config["gate_contract"]["linear_relative_residual_max"],
        "gqns_poisson_residual": gq_metrics["poisson_relative_residual"]
        <= config["gate_contract"]["linear_relative_residual_max"],
        "finite_positive_projection": _finite_positive(profiles),
        "predecessor_source_gate": item["sealed"]["all_numerical_gates_pass"] is True,
    }
    return {
        "object_id": item["object_id"],
        "cell_id": item["cell_id"],
        "cell_kind": item["cell_kind"],
        "primary_cell": item["primary_cell"],
        "geometry_label": "MODEL_LIFTED_2P5D_SOURCE_IN_FULL_3D_FIELD_SOLVER",
        "grid_nodes": len(grid.coordinates),
        "half_box_kpc": item["half_box_kpc"],
        "density_hash": density_hash,
        "total_mass_msun": item["total_mass_msun"],
        "source_geometry": {
            "rms_radius_over_half_box": rms_radius,
            "anisotropy_A_Q": anisotropy,
        },
        "profiles": profiles,
        "solver_metrics": {
            "NEWTON": {"relative_residual": newton.relative_residual},
            "AQUAL_SIMPLE_MU": {
                "relative_residual": aqual.relative_residual,
                "converged": aqual.converged,
                "iterations": aqual.iterations,
            },
            "QUMOND_SIMPLE_NU": {"relative_residual": qumond.relative_residual},
            "NFW_SOURCE_MATCHED_CONTROL": nfw_metrics,
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN": rg_metrics,
            "GP01_ELLIPTIC_N2_L035": gp_metrics,
            "MASHHOON_RAHVAR_NLG_Q0": {
                "published_alpha_0": 10.94,
                "published_mu_0_kpc_inverse": 0.059,
                "direct_extended_source_correction": True,
            },
            "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE": gq_metrics,
        },
        "field_hashes": {
            "newton": bridge.array_sha256(newton.potential),
            "aqual": bridge.array_sha256(aqual.potential),
            "qumond": bridge.array_sha256(qumond.potential),
            "nfw_control": bridge.array_sha256(nfw_potential),
            "refracted": bridge.array_sha256(rg_potential),
            "gp01": bridge.array_sha256(gp_potential),
            "gqns_potential": bridge.array_sha256(gq_potential),
            "gqns_effective_density": bridge.array_sha256(gq_density),
        },
        "newton_profile_replay_relative": replay_error,
        "gates": gates,
        "all_numerical_gates_pass": all(gates.values()),
        "failed_gates": sorted(key for key, passed in gates.items() if not passed),
    }


def _gaussian(grid: base.Grid3D, sx: float, sy: float, sz: float, *, x0: float = 0.0) -> np.ndarray:
    value = np.exp(-0.5 * (((grid.x - x0) / sx) ** 2 + (grid.y / sy) ** 2 + (grid.z / sz) ** 2))
    return value / (float(value.sum()) * grid.spacing**3)


def _spiral(grid: base.Grid3D) -> np.ndarray:
    radius = np.sqrt(grid.x * grid.x + grid.y * grid.y)
    angle = np.arctan2(grid.y, grid.x)
    phase = 2.0 * angle - 2.0 * np.log(np.maximum(radius, 0.08) / 0.25)
    value = (
        np.exp(-radius / 0.35) * (1.0 + 0.45 * np.cos(phase)) * np.exp(-0.5 * (grid.z / 0.10) ** 2)
    )
    return value / (float(value.sum()) * grid.spacing**3)


def _relative_field_difference(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(first))), 1.0e-15)
    return float(np.max(np.abs(first - second)) / scale)


def run_fixture_suite() -> dict[str, Any]:
    grid = base.make_grid(13)
    zero = np.zeros(grid.shape)
    gates: dict[str, dict[str, Any]] = {}

    radius_kpc = np.asarray([1.0, 5.0, 15.0])
    direct = 1.0 + _nlg_delta_factor(radius_kpc)
    analytic = 1.0 + 10.94 * (1.0 - (1.0 + 0.5 * 0.059 * radius_kpc) * np.exp(-0.059 * radius_kpc))
    nlg_error = float(np.max(np.abs(direct - analytic)))
    gates["PUBLISHED_Q0_POINT_FORCE_IDENTITY"] = {
        "passed": nlg_error <= 1.0e-14,
        "metrics": {"maximum_absolute_error": nlg_error},
    }

    k = math.pi / 2.0
    manufactured_source = (
        np.sin(k * (grid.x + 1.0)) * np.sin(k * (grid.y + 1.0)) * np.sin(k * (grid.z + 1.0))
    )
    length = 0.25
    amplitude = 0.4
    inverse = 1.0 / length
    solved = halo.solve_helmholtz(
        -(inverse**2) * amplitude * manufactured_source,
        zero,
        grid.spacing,
        mass=float(inverse),
    )
    expected = amplitude * inverse**2 / (inverse**2 + 3.0 * k**2) * manufactured_source
    manufactured_error = float(
        np.max(np.abs(solved.potential - expected)) / np.max(np.abs(expected))
    )
    gates["GQNS_HELMHOLTZ_MANUFACTURED"] = {
        "passed": manufactured_error <= 0.03 and solved.relative_residual <= 1.0e-12,
        "metrics": {
            "relative_solution_error": manufactured_error,
            "relative_residual": solved.relative_residual,
        },
    }

    sphere = _gaussian(grid, 0.22, 0.22, 0.22)
    _, sphere_a, _ = _rms_geometry(sphere, grid)
    gates["GQNS_EXACT_SPHERICAL_SHUTOFF"] = {
        "passed": sphere_a <= 1.0e-14,
        "metrics": {"anisotropy_A_Q": sphere_a},
    }

    bar_density = _gaussian(grid, 0.42, 0.16, 0.09)
    spiral_density = _spiral(grid)
    bar_phi, _, bar_metrics = _solve_gqns(bar_density, grid)
    spiral_phi, _, spiral_metrics = _solve_gqns(spiral_density, grid)
    bar_newton = base.solve_poisson(4.0 * math.pi * bar_density, zero, grid.spacing)
    spiral_newton = base.solve_poisson(4.0 * math.pi * spiral_density, zero, grid.spacing)
    bar_difference = _relative_field_difference(bar_newton.potential, bar_phi)
    spiral_difference = _relative_field_difference(spiral_newton.potential, spiral_phi)
    gates["BAR_BRANCH_ACTIVE"] = {
        "passed": bar_metrics["anisotropy_A_Q"] > 0.3 and bar_difference > 0.01,
        "metrics": {
            "anisotropy_A_Q": bar_metrics["anisotropy_A_Q"],
            "relative_potential_difference": bar_difference,
        },
    }
    gates["SPIRAL_BRANCH_ACTIVE"] = {
        "passed": spiral_metrics["anisotropy_A_Q"] > 0.1 and spiral_difference > 0.01,
        "metrics": {
            "anisotropy_A_Q": spiral_metrics["anisotropy_A_Q"],
            "relative_potential_difference": spiral_difference,
        },
    }

    thin = _gaussian(grid, 0.35, 0.35, 0.07)
    thick = _gaussian(grid, 0.35, 0.35, 0.25)
    _, thin_a, _ = _rms_geometry(thin, grid)
    _, thick_a, _ = _rms_geometry(thick, grid)
    gates["VERTICAL_THICKNESS_ORDERING"] = {
        "passed": thin_a > thick_a > 0.0,
        "metrics": {"thin_A_Q": thin_a, "thick_A_Q": thick_a},
    }

    saddle = _gaussian(grid, 0.13, 0.13, 0.13, x0=-0.36) + _gaussian(
        grid, 0.13, 0.13, 0.13, x0=0.36
    )
    saddle /= float(saddle.sum()) * grid.spacing**3
    saddle_phi, _, _ = _solve_gqns(saddle, grid)
    saddle_acceleration = base.acceleration(saddle_phi, grid.spacing)
    centre = tuple(size // 2 for size in grid.shape)
    central = float(
        math.sqrt(sum(float(component[centre]) ** 2 for component in saddle_acceleration))
    )
    gates["SADDLE_EXACT_NULL"] = {
        "passed": central <= 1.0e-12,
        "metrics": {"central_field_magnitude": central},
    }

    external = 0.17 * grid.x - 0.09 * grid.y
    shifted = base.acceleration(bar_phi + external, grid.spacing)
    original = base.acceleration(bar_phi, grid.spacing)
    shift_error = float(
        max(
            np.max(np.abs((shifted[0] - original[0]) + 0.17)),
            np.max(np.abs((shifted[1] - original[1]) - 0.09)),
            np.max(np.abs(shifted[2] - original[2])),
        )
    )
    gates["UNIFORM_EXTERNAL_FIELD_SUPERPOSITION"] = {
        "passed": shift_error <= 1.0e-13,
        "metrics": {"maximum_shift_error": shift_error},
    }

    rotated_density = np.rot90(bar_density, axes=(0, 1))
    rotated_phi, _, rotated_metrics = _solve_gqns(rotated_density, grid)
    rotation_error = _relative_field_difference(np.rot90(bar_phi, axes=(0, 1)), rotated_phi)
    gates["ROTATION_COVARIANCE"] = {
        "passed": rotation_error <= 1.0e-11
        and abs(bar_metrics["anisotropy_A_Q"] - rotated_metrics["anisotropy_A_Q"]) <= 1.0e-14,
        "metrics": {"relative_potential_error": rotation_error},
    }
    failed = sorted(name for name, row in gates.items() if not row["passed"])
    return {
        "passed": len(gates) - len(failed),
        "failed": len(failed),
        "failed_gates": failed,
        "gates": gates,
    }


def _envelopes(
    cells: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mechanism_ids = [row["id"] for row in config["mechanisms"]]
    for object_id in config["source_contract"]["objects"]:
        eligible = [
            row
            for row in cells
            if row["object_id"] == object_id
            and row["cell_kind"] == "PRIMARY_CARTESIAN"
            and row["all_numerical_gates_pass"]
        ]
        for mechanism in mechanism_ids:
            for index, radius in enumerate(config["projection_contract"]["radii_kpc"]):
                values = [
                    (
                        float(cell["profiles"][mechanism][index]["radial_acceleration_over_a0"]),
                        cell["cell_id"],
                    )
                    for cell in eligible
                ]
                if not values:
                    rows.append(
                        {
                            "object_id": object_id,
                            "mechanism": mechanism,
                            "radius_kpc": radius,
                            "status": "NO_VALID_COMMON_CELL",
                        }
                    )
                    continue
                minimum = min(values)
                maximum = max(values)
                rows.append(
                    {
                        "object_id": object_id,
                        "mechanism": mechanism,
                        "radius_kpc": radius,
                        "status": "COMMON_SOURCE_SYSTEMATIC_ENVELOPE",
                        "minimum_over_a0": minimum[0],
                        "minimum_cell_id": minimum[1],
                        "maximum_over_a0": maximum[0],
                        "maximum_cell_id": maximum[1],
                        "maximum_to_minimum_ratio": maximum[0] / minimum[0],
                        "valid_common_cell_count": len(values),
                    }
                )
    return rows


def _equivalences(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for cell in cells:
        for mechanism, profiles in cell["profiles"].items():
            key = hashlib.sha256(canonical_bytes(profiles)).hexdigest()
            groups[f"{mechanism}:{key}"].append(f"{cell['object_id']}::{cell['cell_id']}")
    return [
        {"equivalence_key": key, "multiplicity": len(members), "members": sorted(members)}
        for key, members in sorted(groups.items())
        if len(members) > 1
    ]


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    source_receipt = validate_bindings(config)
    fixture_suite = run_fixture_suite()
    _require(fixture_suite["failed"] == 0, "target-free fixture suite failed")
    cells = [
        _solve_real_cell(config, item) for item in _iter_source_densities(config, source_receipt)
    ]
    _require(len(cells) == 225, "real-source cell count changed")
    _require(len({(row["object_id"], row["cell_id"]) for row in cells}) == 225, "duplicate cell")
    counterexamples = [
        {
            "object_id": row["object_id"],
            "cell_id": row["cell_id"],
            "failed_gates": row["failed_gates"],
        }
        for row in cells
        if not row["all_numerical_gates_pass"]
    ]
    envelopes = _envelopes(cells, config)
    equivalences = _equivalences(cells)
    primary = [row for row in cells if row["primary_cell"]]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "PASS_RESPONSE_BLIND_225_CELL_EIGHT_MECHANISM_SCREEN_WITH_RETAINED_COUNTEREXAMPLES"
            if counterexamples
            else "PASS_RESPONSE_BLIND_225_CELL_EIGHT_MECHANISM_SCREEN"
        ),
        "decision": "SOURCE_ONLY_NONSYMMETRIC_PREDICTIONS_READY_FOR_SEPARATELY_AUTHORIZED_DEVELOPMENT_RESPONSE_SCORE",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "source_contract": config["source_contract"],
        "mechanisms": config["mechanisms"],
        "published_anchors": config["published_anchors"],
        "projection_contract": config["projection_contract"],
        "benchmark_contract": config["benchmark_contract"],
        "fixture_suite": fixture_suite,
        "cell_count": len(cells),
        "primary_cell_count": len(primary),
        "numerical_pass_cell_count": len(cells) - len(counterexamples),
        "retained_counterexample_count": len(counterexamples),
        "retained_counterexamples": counterexamples,
        "cells": cells,
        "cell_ledger_root_sha256": hashlib.sha256(canonical_bytes(cells)).hexdigest(),
        "source_systematic_envelopes": envelopes,
        "source_systematic_envelope_root_sha256": hashlib.sha256(
            canonical_bytes(envelopes)
        ).hexdigest(),
        "equivalence_groups": equivalences,
        "equivalence_group_count": len(equivalences),
        "primary_predictions": primary,
        "nearest_neighbor_boundary": [
            {
                "neighbor": "MASHHOON_RAHVAR_NLG_Q0",
                "shared": "rho_D is a spatial convolution of the baryonic source with a positive isotropic kernel",
                "difference": "published q0 has universal fitted alpha0 and mu0; GQNS derives both activation amplitude and range from each source geometry, vanishes exactly for a sphere, and is not represented as published NLG",
            },
            {
                "neighbor": "GP01_ELLIPTIC",
                "shared": "a screened elliptic auxiliary field carries spatial nonlocality",
                "difference": "GP01 smooths a local-acceleration gain target and changes gravitational permittivity; GQNS convolves baryonic density and activates only through the global 3D quadrupole invariant",
            },
            {
                "neighbor": "AQUAL_QUMOND_EXTERNAL_FIELD_EFFECT",
                "shared": "nonspherical geometry can distinguish the law from its spherical algebraic limit",
                "difference": "GQNS is source-geometry coupled and exactly superposes a uniform external field; it is not an acceleration-dependent EFE or an anisotropic directional kernel",
            },
            {
                "neighbor": "REFRACTED_GRAVITY",
                "shared": "baryonic morphology changes a three-dimensional field prediction",
                "difference": "RG uses local density-dependent permittivity; GQNS uses a global quadrupole scalar multiplying a nonlocal reciprocal source kernel",
            },
        ],
        "source_readiness": {
            "NEWTON": "SOURCE_READY_MODEL_LIFTED_2P5D_225_CELLS",
            "NFW_SOURCE_MATCHED_CONTROL": "SOURCE_READY_GEOMETRY_CONTROL_NOT_OBSERVATIONAL_HALO_FIT",
            "AQUAL_SIMPLE_MU": "SOURCE_READY_MODEL_LIFTED_2P5D_225_CELLS",
            "QUMOND_SIMPLE_NU": "SOURCE_READY_MODEL_LIFTED_2P5D_225_CELLS",
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN": "SOURCE_READY_NONCOVARIANT_MODEL_LIFTED_2P5D_225_CELLS",
            "GP01_ELLIPTIC_N2_L035": "SOURCE_READY_PHENOMENOLOGICAL_MODEL_LIFTED_2P5D_225_CELLS",
            "MASHHOON_RAHVAR_NLG_Q0": "SOURCE_READY_PUBLISHED_STATIC_LINEAR_Q0_MODEL_LIFTED_2P5D_225_CELLS",
            "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE": "SOURCE_READY_EXPLORATORY_ZERO_FIT_MODEL_LIFTED_2P5D_225_CELLS",
            "MEASURED_3D": "SOURCE_BLOCKED_ZERO_OBJECTS",
            "OBSERVED_EXTERNAL_TIDAL_FIELD": "SOURCE_BLOCKED_NOT_IN_CURRENT_SOURCE_PACKET",
        },
        "unique_geometry_discriminators": [
            "exact GQNS spherical shutoff versus disk/bar/spiral activation",
            "m=1 through m=4 force harmonics at fixed radii",
            "thin-versus-thick vertical-force ordering",
            "AQUAL/GP01 external-field response versus GQNS uniform-field superposition",
            "RG local-density sensitivity versus GQNS global-quadrupole sensitivity",
            "published NLG universal q0 range versus GQNS source-derived RMS range",
        ],
        "next_real_data_falsifier": {
            "primary": "At matched baryonic Newtonian acceleration and baryonic mass, compare a round unbarred disk with a strongly barred or two-arm disk using resolved two-dimensional velocity fields; GQNS predicts a source-anisotropy-linked change in the force harmonics while published q0 NLG does not switch off with roundness.",
            "currently_available_development_objects": ["NGC2903", "NGC3351", "NGC3627"],
            "confirmation_requirement": "Acquire an unopened matched round/barred pair with the same S4G+H I+CO source products and an independent 2D kinematic response; do not call model-lifted source geometry measured 3D.",
        },
        "access_contract": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    expected = build_receipt(config)
    _require(dict(payload) == expected, "receipt differs from deterministic rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))


def validate_receipt() -> None:
    config = load_config()
    receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
    validate_receipt_payload(config, receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        config = load_config()
        if _repo_path(OUTPUT_PATH).exists():
            receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
            print(receipt["status"])
        else:
            print(config["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
