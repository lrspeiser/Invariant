"""Real-source, zero-response Refracted Gravity 225-by-9 3-D screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge
from sigma_theory_compiler import (
    open_gravity_phangs_things_full3d_source_systematics_v1 as source_systematics,
)
from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as source_builder,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_3d_primary_benchmark_v1 as rg_benchmark,
)

CONFIG_PATH = Path(
    "configs/open_gravity_refracted_gravity_phangs_things_full3d_source_screen_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_refracted_gravity_phangs_things_full3d_source_screen_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_refracted_gravity_phangs_things_full3d_source_screen_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-phangs-things-full3d-source-screen-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-refracted-gravity-phangs-things-full3d-source-screen-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-phangs-things-full3d-source-screen-receipt-1.0"
)
_CONFIG_RAW_SHA256 = "f40f511a29d633f57ff7ce1217d1b8dba23b9daf3f4626584969bf1b442ee307"
_CONFIG_CONTENT_SHA256 = "43fddeade4cd281e64629b98b61512750351d801c599d3735bbd9c7068642fbf"
_MODULE_SEMANTIC_SHA256 = "fe841fd7857087773dffc2a178b37774b96278d6074ffcdfb7d9c42acc3c256a"
_TEST_RAW_SHA256 = "c9e79b05ee8fad0a11acc1ad407a2810c098716b24c1b75a9a4b5c82ea33d8c4"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class RefractedGravitySourceScreenError(RuntimeError):
    """Raised when the real-source operator screen fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefractedGravitySourceScreenError(message)


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
        raise RefractedGravitySourceScreenError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_REAL_SOURCE_ZERO_RESPONSE_FULL_225_BY_9_SCREEN",
        "status changed",
    )
    _require(config["objects"] == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    sources = config["real_source_contract"]
    _require(sources["source_cells"] == 225, "source-cell count changed")
    _require(sources["primary_cartesian_cells"] == 216, "primary-cell count changed")
    _require(sources["numerical_source_controls"] == 9, "control count changed")
    _require(sources["source_selection_from_response"] is False, "response selection enabled")
    units = config["unit_contract"]
    expected_conversion = units["solar_mass_kg"] / units["parsec_m"] ** 3 / 1000.0
    _require(
        units["msun_pc3_to_g_cm3"] == expected_conversion,
        "density conversion changed",
    )
    operator = config["operator_contract"]
    _require(operator["parameter_cells"] == 9, "parameter-cell count changed")
    _require(operator["source_parameter_pairs"] == 2025, "pair count changed")
    _require(operator["maximum_unique_linear_solves"] == 1350, "solve ceiling changed")
    _require(operator["nodes_per_axis"] == 17, "screen grid changed")
    _require(operator["response_parameter_fitting"] is False, "response fitting enabled")
    gates = config["gate_contract"]
    _require(gates["one_failure_cannot_prune_theory_family"] is True, "family pruning enabled")
    _require(gates["benchmark_failure_blocks_scoring"] is True, "benchmark gate weakened")
    access = config["access_contract"]
    _require(access["scientific_source_files_opened_per_build"] == 21, "source access hidden")
    _require(access["scientific_source_bytes_opened_per_build"] == 74030400, "bytes changed")
    for key in (
        "scientific_response_files_opened",
        "scientific_response_rows_opened",
        "scientific_response_values_opened",
        "scores_computed",
        "parameters_fit",
        "network_calls",
        "model_calls",
        "paid_calls",
    ):
        _require(access[key] == 0, f"forbidden access enabled: {key}")
    _require(access["development_only"] is True, "development boundary changed")
    claims = config["claim_boundary"]
    for key in (
        "real_kinematic_response_tested",
        "observational_preference_established",
        "published_parameter_universality_established",
        "covariant_refracted_gravity_tested",
        "lensing_closure_established",
        "source_uncertainty_is_statistical_posterior",
        "novelty_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim ceiling exceeded: {key}")
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
    config = _read_json(path, "source-screen config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    _require(
        [row["role"] for row in config["predecessor_bindings"]]
        == [
            "REAL_SOURCE_225_CELL_SYSTEMATICS",
            "REFRACTED_GRAVITY_PRIMARY_PAPER_BENCHMARK",
        ],
        "predecessor roles changed",
    )
    for binding in config["predecessor_bindings"]:
        _require(binding["commit"] is None, "unexpected unverified commit binding")
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
    source_receipt = receipts["REAL_SOURCE_225_CELL_SYSTEMATICS"]
    _require(source_receipt["cell_count"] == 225, "source predecessor cell count changed")
    _require(
        source_receipt["scientific_boundary"]["response_files_opened"] == 0,
        "source response opened",
    )
    benchmark_receipt = receipts["REFRACTED_GRAVITY_PRIMARY_PAPER_BENCHMARK"]
    _require(benchmark_receipt["benchmark_suite"]["failed"] == 0, "RG benchmark failed")
    _require(benchmark_receipt["benchmark_suite"]["passed"] == 14, "RG benchmark coverage changed")
    return receipts


def _build_density(
    config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    stellar_surface: np.ndarray,
    co_surface: np.ndarray,
    hstar_pc: float,
    hgas_pc: float,
    half_box_kpc: float,
) -> dict[str, Any]:
    operator = config["operator_contract"]
    normal = bridge_config["normalization_contract"]
    grid = base.make_grid(int(operator["nodes_per_axis"]))
    half_box_pc = half_box_kpc * 1000.0
    coordinates_pc = grid.coordinates * half_box_pc
    spacing_pc = grid.spacing * half_box_pc
    surfaces = (
        ("stellar", np.asarray(stellar_surface, dtype=np.float64), hstar_pc),
        ("hi", np.asarray(maps["hi"], dtype=np.float64), hgas_pc),
        ("co", np.asarray(co_surface, dtype=np.float64), hgas_pc),
    )
    density_msun_pc3 = np.zeros(grid.shape, dtype=np.float64)
    masses: dict[str, float] = {}
    for label, surface, height in surfaces:
        component_density, mass = bridge.deposit_surface_component(
            surface,
            np.asarray(maps["x_pc"]),
            np.asarray(maps["y_pc"]),
            float(maps["dx_pc"]),
            coordinates_pc,
            spacing_pc,
            height,
        )
        density_msun_pc3 += component_density
        masses[f"{label}_mass_msun"] = mass
    a0_pc = float(normal["a0_m_s2"]) * float(normal["pc_m"]) / 1.0e6
    gravity_pc = float(normal["G_pc_km2_s2_msun"])
    density_dimensionless = density_msun_pc3 * gravity_pc * half_box_pc / a0_pc
    density_g_cm3 = density_msun_pc3 * float(config["unit_contract"]["msun_pc3_to_g_cm3"])
    total_mass = float(sum(masses.values()))
    expected_mass = gravity_pc * total_mass / (a0_pc * half_box_pc**2)
    deposited_mass = float(density_dimensionless.sum() * grid.spacing**3)
    newton_boundary = bridge.spherical_boundary(
        grid,
        expected_mass,
        mond=False,
        integration_samples=100000,
    )
    rhs = 4.0 * math.pi * density_dimensionless
    newton = base.solve_poisson(rhs, newton_boundary, grid.spacing)
    return {
        "grid": grid,
        "half_box_kpc": half_box_kpc,
        "density_msun_pc3": density_msun_pc3,
        "density_g_cm3": density_g_cm3,
        "density_dimensionless": density_dimensionless,
        "rhs": rhs,
        "newton_boundary": newton_boundary,
        "newton_potential": newton.potential,
        "newton_residual": newton.relative_residual,
        "masses": masses,
        "total_mass_msun": total_mass,
        "dimensionless_mass_relative_error": abs(deposited_mass - expected_mass) / expected_mass,
    }


def _iter_source_cells(
    config: Mapping[str, Any],
) -> Iterator[tuple[dict[str, Any], Mapping[str, Any]]]:
    source_config, acquisition, private = source_systematics._load_source_builder_evidence()
    systematics_config = source_systematics.load_config()
    bridge_config = bridge.load_config()
    source_paths = source_builder._source_paths(acquisition)
    expected = source_systematics._source_summary_index(private)
    metadata_by_id = {row["object_id"]: row for row in source_config["objects"]}
    axes = systematics_config["cell_contract"]["primary_axes"]
    for object_id in config["objects"]:
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
        half_mass_pc = source_builder._half_mass_radius_pc(
            robust_maps["stellar_fixed"],
            robust_maps["x_pc"],
            robust_maps["y_pc"],
            float(robust_maps["dx_pc"]),
        )
        exponential_scale_pc = half_mass_pc / 1.678
        maps_by_beam = {
            "ROBUST_PRIMARY": robust_maps,
            "NATURAL_SENSITIVITY": source_builder._surface_maps(
                source_config,
                metadata,
                images,
                n=256,
                box_kpc=40.0,
                beam="NATURAL_SENSITIVITY",
                use_sip=False,
            ),
        }
        for beam, stellar, co_source, ratio, gas_height in itertools.product(
            axes["beam"],
            axes["stellar_mass_to_light"],
            axes["co_source"],
            axes["stellar_height_over_exponential_scale"],
            axes["gas_height_pc"],
        ):
            maps = maps_by_beam[beam]
            cell_id = source_systematics._cell_id(
                beam,
                stellar,
                co_source,
                float(ratio),
                float(gas_height),
            )
            built = _build_density(
                config,
                bridge_config,
                maps,
                stellar_surface=(
                    maps["stellar_fixed"] if stellar == "FIXED_0P6" else maps["stellar_color"]
                ),
                co_surface=(maps["co"] if co_source == "WITH_CO" else np.zeros_like(maps["co"])),
                hstar_pc=exponential_scale_pc * float(ratio),
                hgas_pc=float(gas_height),
                half_box_kpc=30.0,
            )
            yield (
                {
                    "object_id": object_id,
                    "cell_id": cell_id,
                    "cell_kind": "PRIMARY_CARTESIAN",
                    "source_builder_mass_relative_error": source_systematics._mass_error(
                        built["masses"], expected[(object_id, cell_id)]
                    ),
                },
                built,
            )
        control_physics = systematics_config["cell_contract"]["control_source_physics"]
        for control in systematics_config["cell_contract"]["controls_per_object"]:
            maps = source_builder._surface_maps(
                source_config,
                metadata,
                images,
                n=int(control["source_pixels"]),
                box_kpc=float(control["source_box_kpc"]),
                beam=control_physics["beam"],
                use_sip=bool(control["sip"]),
            )
            built = _build_density(
                config,
                bridge_config,
                maps,
                stellar_surface=maps["stellar_fixed"],
                co_surface=maps["co"],
                hstar_pc=exponential_scale_pc
                * float(control_physics["stellar_height_over_exponential_scale"]),
                hgas_pc=float(control_physics["gas_height_pc"]),
                half_box_kpc=float(control["solver_half_box_kpc"]),
            )
            yield (
                {
                    "object_id": object_id,
                    "cell_id": control["id"],
                    "cell_kind": "NUMERICAL_SOURCE_CONTROL",
                    "source_builder_mass_relative_error": source_systematics._mass_error(
                        built["masses"], expected[(object_id, control["id"])]
                    ),
                },
                built,
            )


def _profile(
    potential: np.ndarray, built: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, float]]:
    return bridge.radial_acceleration_profile(
        potential,
        built["grid"],
        half_box_kpc=float(built["half_box_kpc"]),
        radii_kpc=config["operator_contract"]["radial_profile_radii_kpc"],
        azimuth_samples=int(config["operator_contract"]["azimuth_samples"]),
        a0_m_s2=1.2e-10,
    )


def _solve_parameter_cell(
    config: Mapping[str, Any],
    built: Mapping[str, Any],
    parameter: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, str]:
    epsilon_0 = float(parameter["epsilon_0"])
    q_slope = float(parameter["Q"])
    rho_c = 10.0 ** float(parameter["log10_rho_c_g_cm3"])
    coefficient = rg_benchmark.published_permittivity(
        built["density_g_cm3"],
        epsilon_0=epsilon_0,
        rho_c=rho_c,
        q_slope=q_slope,
    )
    coefficient_hash = bridge.array_sha256(coefficient)
    boundary = np.asarray(built["newton_boundary"]) / epsilon_0
    potential, residual = rg_benchmark._solve_variable(
        built["rhs"], boundary, coefficient, built["grid"].spacing
    )
    profile = _profile(potential, built, config)
    newton_error: float | None = None
    if epsilon_0 == 1.0:
        newton_error = float(
            np.max(np.abs(potential - built["newton_potential"]))
            / max(float(np.max(np.abs(built["newton_potential"]))), 1.0e-12)
        )
    gates = config["gate_contract"]
    finite_positive_profile = all(
        math.isfinite(float(row["radial_acceleration_over_a0"]))
        and float(row["radial_acceleration_over_a0"]) > 0.0
        for row in profile
    )
    gate_results = {
        "positive_ellipticity": bool(np.all(coefficient > 0.0)),
        "permittivity_upper_bound": bool(np.all(coefficient <= gates["permittivity_maximum"])),
        "linear_residual": residual <= gates["linear_relative_residual_max"],
        "radial_acceleration": finite_positive_profile,
        "epsilon_one_newton_control": (
            newton_error is None or newton_error <= gates["epsilon_one_relative_newton_error_max"]
        ),
    }
    return (
        {
            "coefficient_hash": coefficient_hash,
            "minimum_epsilon": float(np.min(coefficient)),
            "maximum_epsilon": float(np.max(coefficient)),
            "potential_hash": bridge.array_sha256(potential),
            "relative_discrete_residual": residual,
            "epsilon_one_relative_newton_error": newton_error,
            "profiles": profile,
            "operator_gates": gate_results,
            "operator_numerical_pass": all(gate_results.values()),
        },
        potential,
        coefficient_hash,
    )


def _source_envelopes(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, all_parameters: bool
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    radii = config["operator_contract"]["radial_profile_radii_kpc"]
    for object_id in config["objects"]:
        for radius_index, radius in enumerate(radii):
            values: list[tuple[float, str, str]] = []
            for source_row in rows:
                if (
                    source_row["object_id"] != object_id
                    or source_row["cell_kind"] != "PRIMARY_CARTESIAN"
                ):
                    continue
                for parameter_row in source_row["parameter_results"]:
                    if (
                        not all_parameters
                        and parameter_row["parameter_id"] != "DISKMASS_UNIVERSAL_MEDIAN"
                    ):
                        continue
                    if parameter_row["future_response_eligible"] is not True:
                        continue
                    values.append(
                        (
                            float(
                                parameter_row["profiles"][radius_index][
                                    "radial_acceleration_over_a0"
                                ]
                            ),
                            source_row["cell_id"],
                            parameter_row["parameter_id"],
                        )
                    )
            _require(values, "source envelope has no valid values")
            minimum = min(values)
            maximum = max(values)
            result.append(
                {
                    "object_id": object_id,
                    "radius_kpc": float(radius),
                    "scope": (
                        "ALL_PUBLISHED_PARAMETERS_AND_PRIMARY_SOURCES"
                        if all_parameters
                        else "FIXED_DISKMASS_MEDIAN_ACROSS_PRIMARY_SOURCES"
                    ),
                    "minimum_over_a0": minimum[0],
                    "minimum_source_cell_id": minimum[1],
                    "minimum_parameter_id": minimum[2],
                    "maximum_over_a0": maximum[0],
                    "maximum_source_cell_id": maximum[1],
                    "maximum_parameter_id": maximum[2],
                    "maximum_to_minimum_ratio": maximum[0] / minimum[0],
                    "value_count": len(values),
                }
            )
    _require(len(result) == 9, "source-envelope row count changed")
    return result


def _theory_comparisons(
    rows: Sequence[Mapping[str, Any]],
    parent_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    radii = config["operator_contract"]["radial_profile_radii_kpc"]
    for object_id in config["objects"]:
        object_rows = [
            row
            for row in rows
            if row["object_id"] == object_id and row["cell_kind"] == "PRIMARY_CARTESIAN"
        ]
        for index, radius in enumerate(radii):
            ratios: dict[str, list[float]] = {
                "RG_OVER_NEWTON": [],
                "ABS_RG_MINUS_AQUAL_OVER_AQUAL": [],
                "ABS_RG_MINUS_QUMOND_OVER_QUMOND": [],
            }
            for row in object_rows:
                median = next(
                    item
                    for item in row["parameter_results"]
                    if item["parameter_id"] == "DISKMASS_UNIVERSAL_MEDIAN"
                )
                if median["future_response_eligible"] is not True:
                    continue
                rg_value = float(median["profiles"][index]["radial_acceleration_over_a0"])
                parent = parent_rows[(object_id, row["cell_id"])]
                newton = float(parent["profiles"]["NEWTON"][index]["radial_acceleration_over_a0"])
                aqual = float(
                    parent["profiles"]["AQUAL_SIMPLE_MU"][index]["radial_acceleration_over_a0"]
                )
                qumond = float(
                    parent["profiles"]["QUMOND_SIMPLE_NU"][index]["radial_acceleration_over_a0"]
                )
                ratios["RG_OVER_NEWTON"].append(rg_value / newton)
                ratios["ABS_RG_MINUS_AQUAL_OVER_AQUAL"].append(abs(rg_value - aqual) / aqual)
                ratios["ABS_RG_MINUS_QUMOND_OVER_QUMOND"].append(abs(rg_value - qumond) / qumond)
            _require(all(ratios.values()), "theory comparison has no valid values")
            output.append(
                {
                    "object_id": object_id,
                    "radius_kpc": float(radius),
                    "fixed_parameter_id": "DISKMASS_UNIVERSAL_MEDIAN",
                    "valid_primary_source_cells": len(ratios["RG_OVER_NEWTON"]),
                    "metrics": {
                        key: {
                            "minimum": min(values),
                            "median": statistics.median(values),
                            "maximum": max(values),
                        }
                        for key, values in ratios.items()
                    },
                }
            )
    _require(len(output) == 9, "theory-comparison row count changed")
    return output


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predecessor_receipts = validate_predecessors(config)
    source_receipt = predecessor_receipts["REAL_SOURCE_225_CELL_SYSTEMATICS"]
    parent_rows = {(row["object_id"], row["cell_id"]): row for row in source_receipt["cells"]}
    _require(len(parent_rows) == 225, "parent source ledger changed")
    benchmark_config = rg_benchmark.load_config()
    parameters = rg_benchmark.published_parameter_cells(benchmark_config)
    _require(len(parameters) == 9, "published parameter inventory changed")
    result_rows: list[dict[str, Any]] = []
    unique_solves = 0
    for descriptor, built in _iter_source_cells(config):
        parent = parent_rows[(descriptor["object_id"], descriptor["cell_id"])]
        source_gates = {
            "parent_source_screen": parent["all_numerical_gates_pass"] is True,
            "source_builder_mass": descriptor["source_builder_mass_relative_error"]
            <= config["gate_contract"]["source_mass_relative_error_max"],
            "dimensionless_mass": built["dimensionless_mass_relative_error"] <= 1.0e-12,
            "source_free_boundary": bool(
                np.all(built["density_msun_pc3"][[0, -1], :, :] == 0.0)
                and np.all(built["density_msun_pc3"][:, [0, -1], :] == 0.0)
                and np.all(built["density_msun_pc3"][:, :, [0, -1]] == 0.0)
            ),
            "newton_residual": built["newton_residual"]
            <= config["gate_contract"]["linear_relative_residual_max"],
        }
        cache: dict[str, dict[str, Any]] = {}
        parameter_results: list[dict[str, Any]] = []
        for parameter in parameters:
            epsilon = rg_benchmark.published_permittivity(
                built["density_g_cm3"],
                epsilon_0=float(parameter["epsilon_0"]),
                rho_c=10.0 ** float(parameter["log10_rho_c_g_cm3"]),
                q_slope=float(parameter["Q"]),
            )
            coefficient_hash = bridge.array_sha256(epsilon)
            reused = coefficient_hash in cache
            if reused:
                solved = copy.deepcopy(cache[coefficient_hash])
            else:
                solved, _potential, observed_hash = _solve_parameter_cell(config, built, parameter)
                _require(observed_hash == coefficient_hash, "coefficient hash changed")
                cache[coefficient_hash] = copy.deepcopy(solved)
                unique_solves += 1
            all_pass = all(source_gates.values()) and solved["operator_numerical_pass"] is True
            parameter_results.append(
                {
                    "parameter_id": parameter["id"],
                    "epsilon_0": parameter["epsilon_0"],
                    "Q": parameter["Q"],
                    "log10_rho_c_g_cm3": parameter["log10_rho_c_g_cm3"],
                    "equivalence_reused": reused,
                    **solved,
                    "future_response_eligible": all_pass,
                    "future_response_disposition": (
                        config["gate_contract"]["passed_disposition"]
                        if all_pass
                        else config["gate_contract"]["failed_disposition"]
                    ),
                }
            )
        _require(len(parameter_results) == 9, "source parameter results changed")
        _require(len(cache) == 6, "per-source exact-equivalence count changed")
        result_rows.append(
            {
                **descriptor,
                "half_box_kpc": built["half_box_kpc"],
                "grid_nodes": len(built["grid"].coordinates),
                "source_density_hash": bridge.array_sha256(built["density_msun_pc3"]),
                "physical_density_g_cm3_hash": bridge.array_sha256(built["density_g_cm3"]),
                "dimensionless_density_hash": bridge.array_sha256(built["density_dimensionless"]),
                "minimum_positive_density_g_cm3": float(
                    np.min(built["density_g_cm3"][built["density_g_cm3"] > 0.0])
                ),
                "maximum_density_g_cm3": float(np.max(built["density_g_cm3"])),
                "dimensionless_mass_relative_error": built["dimensionless_mass_relative_error"],
                "source_gates": source_gates,
                "source_numerical_pass": all(source_gates.values()),
                "unique_coefficient_fields": len(cache),
                "parameter_results": parameter_results,
            }
        )
    _require(len(result_rows) == 225, "source result count changed")
    _require(
        len({(row["object_id"], row["cell_id"]) for row in result_rows}) == 225,
        "duplicate source result",
    )
    registered_pairs = sum(len(row["parameter_results"]) for row in result_rows)
    _require(registered_pairs == 2025, "registered source-parameter count changed")
    _require(unique_solves == 1350, "actual unique solve count changed")
    counterexamples = [
        {
            "object_id": row["object_id"],
            "source_cell_id": row["cell_id"],
            "parameter_id": parameter["parameter_id"],
            "failed_source_gates": sorted(
                key for key, passed in row["source_gates"].items() if passed is not True
            ),
            "failed_operator_gates": sorted(
                key for key, passed in parameter["operator_gates"].items() if passed is not True
            ),
        }
        for row in result_rows
        for parameter in row["parameter_results"]
        if parameter["future_response_eligible"] is not True
    ]
    eligible_pairs = registered_pairs - len(counterexamples)
    median_envelopes = _source_envelopes(result_rows, config, all_parameters=False)
    full_envelopes = _source_envelopes(result_rows, config, all_parameters=True)
    comparisons = _theory_comparisons(result_rows, parent_rows, config)
    equivalence_groups: dict[str, list[str]] = defaultdict(list)
    for row in result_rows:
        for parameter in row["parameter_results"]:
            identity = content_sha256(
                {
                    "source_density_hash": row["source_density_hash"],
                    "coefficient_hash": parameter["coefficient_hash"],
                    "potential_hash": parameter["potential_hash"],
                }
            )
            equivalence_groups[identity].append(
                f"{row['object_id']}::{row['cell_id']}::{parameter['parameter_id']}"
            )
    equivalence = [
        {
            "equivalence_sha256": key,
            "multiplicity": len(members),
            "members": sorted(members),
        }
        for key, members in sorted(equivalence_groups.items())
    ]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "PASS_FULL_225_BY_9_REAL_SOURCE_OPERATOR_SCREEN_ZERO_COUNTEREXAMPLES"
            if not counterexamples
            else "PASS_FULL_225_BY_9_REAL_SOURCE_OPERATOR_SCREEN_WITH_RETAINED_COUNTEREXAMPLES"
        ),
        "decision": "SOURCE_OPERATOR_PREDICTIONS_READY_RESPONSE_SCORING_STILL_UNRUN",
        "bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_raw_sha256": file_sha256(_repo_path(MODULE_PATH)),
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
            "predecessors": config["predecessor_bindings"],
        },
        "objects": config["objects"],
        "operator_contract": config["operator_contract"],
        "unit_contract": config["unit_contract"],
        "published_parameter_cells": parameters,
        "source_rows": result_rows,
        "source_cell_count": len(result_rows),
        "registered_source_parameter_pairs": registered_pairs,
        "unique_linear_solves": unique_solves,
        "eligible_pair_count": eligible_pairs,
        "retained_counterexample_count": len(counterexamples),
        "retained_counterexamples": counterexamples,
        "equivalence_groups": equivalence,
        "equivalence_group_count": len(equivalence),
        "fixed_median_source_envelopes": median_envelopes,
        "all_parameter_and_source_envelopes": full_envelopes,
        "same_source_theory_comparisons": comparisons,
        "roots": {
            "source_parameter_ledger_sha256": content_sha256(result_rows),
            "counterexample_ledger_sha256": content_sha256(counterexamples),
            "equivalence_ledger_sha256": content_sha256(equivalence),
            "fixed_median_source_envelope_sha256": content_sha256(median_envelopes),
            "all_parameter_source_envelope_sha256": content_sha256(full_envelopes),
            "same_source_theory_comparison_sha256": content_sha256(comparisons),
        },
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt must be an object")
    _require(payload == build_receipt(config), "receipt is not deterministic")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    path = _repo_path(OUTPUT_PATH)
    _require(path == (_ROOT / OUTPUT_PATH).resolve(), "output path changed")
    return path


def write_receipt() -> str:
    config = load_config()
    payload = json.dumps(build_receipt(config), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path = _output_path()
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
    config = load_config()
    validate_receipt_payload(config, _read_json(_output_path(), "source-screen receipt"))


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
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "source_cells": receipt["source_cell_count"],
                    "registered_pairs": receipt["registered_source_parameter_pairs"],
                    "eligible_pairs": receipt["eligible_pair_count"],
                    "retained_counterexamples": receipt["retained_counterexample_count"],
                    "scientific_response_files_opened": 0,
                    "scores_computed": 0,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
