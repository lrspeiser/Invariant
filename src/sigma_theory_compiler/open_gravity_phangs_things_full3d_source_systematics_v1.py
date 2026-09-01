"""Screen every frozen real-source systematic through three published 3-D solvers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as baseline
from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge
from sigma_theory_compiler import (
    open_gravity_phangs_things_model_lifted_3d_source_builder_v1 as source_builder,
)

CONFIG_PATH = Path("configs/open_gravity_phangs_things_full3d_source_systematics_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_phangs_things_full3d_source_systematics_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_phangs_things_full3d_source_systematics_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-phangs-things-full3d-source-systematics-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-phangs-things-full3d-source-systematics-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-phangs-things-full3d-source-systematics-receipt-1.0"
_CONFIG_RAW_SHA256 = "8a8267e4a29a13910de6e56d5a9f27904557425409a6dda42d489831a6e601f8"
_CONFIG_CONTENT_SHA256 = "1501a31975c916bdae5d5eeaa90040dbd018ec17ea518c67a4e8b7d5be01df58"
_MODULE_SEMANTIC_SHA256 = "3f8a4ae558f20f48a68b3a934674c71768dd2014396972f49e3d029b4abac300"
_TEST_RAW_SHA256 = "0a9408dc686ebf4ae8ea5a12c877537eac621710ce7e3b365f446e9fd7ba1606"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class SourceSystematicsError(RuntimeError):
    """Raised when a source-systematic contract or package invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSystematicsError(message)


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
        raise SourceSystematicsError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "RESPONSE_BLIND_FULL_225_SOURCE_SYSTEMATIC_SCREEN_DEVELOPMENT_ONLY",
        "status changed",
    )
    _require(config["objects"] == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    cells = config["cell_contract"]
    _require(cells["primary_cells_per_object"] == 72, "primary cell count changed")
    _require(len(cells["controls_per_object"]) == 3, "control count changed")
    _require(cells["total_cells_per_object"] == 75, "object cell count changed")
    _require(cells["total_cells"] == 225, "global cell count changed")
    _require(cells["response_based_selection"] is False, "response selection enabled")
    _require(cells["retain_all_failures"] is True, "failure retention removed")
    screen = config["screen_contract"]
    _require(screen["nodes_per_axis"] == 17, "screen grid changed")
    _require(
        screen["operators"] == ["NEWTON", "AQUAL_SIMPLE_MU", "QUMOND_SIMPLE_NU"],
        "operators changed",
    )
    _require(screen["radial_profile_radii_kpc"] == [5.0, 10.0, 15.0], "radii changed")
    gates = config["gate_contract"]
    _require(gates["no_family_pruning_from_one_failure"] is True, "failure pruning enabled")
    _require(
        gates["primary_17_vs_bound_25_radial_relative_difference_max"] == 0.2,
        "bridge comparison gate changed",
    )
    boundary = config["scientific_boundary"]
    _require(boundary["source_files_opened_per_build"] == 21, "source access hidden")
    _require(boundary["response_files_opened"] == 0, "response access enabled")
    _require(boundary["response_rows_opened"] == 0, "response rows enabled")
    _require(boundary["scores_computed"] == 0, "scoring enabled")
    _require(boundary["network_calls"] == 0, "network enabled")
    claims = config["claim_boundary"]
    _require(claims["all_source_systematics_propagated"] is True, "systematic claim lost")
    _require(claims["all_numerical_failures_retained"] is True, "failure claim lost")
    for key in (
        "response_fit_tested",
        "observational_preference_established",
        "high_resolution_all_cell_convergence_established",
        "source_uncertainty_is_statistical_posterior",
        "lensing_closure_established",
        "novelty_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim ceiling exceeded: {key}")
    _require(config["output_contract"]["receipt"] == OUTPUT_PATH.as_posix(), "output path changed")


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


def _validate_predecessor(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["predecessor_binding"]
    for artifact in binding["artifacts"]:
        path = _repo_path(artifact["path"])
        _require(path.is_file(), "bridge artifact missing")
        _require(file_sha256(path) == artifact["sha256"], "bridge artifact changed")
    receipt_path = next(
        row["path"] for row in binding["artifacts"] if row["path"].endswith("receipt.json")
    )
    receipt = _read_json(_repo_path(receipt_path), "bridge receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"],
        "bridge receipt content changed",
    )
    _require(receipt["all_object_gates_pass"] is True, "bridge gate no longer passes")
    private_path = _repo_path(receipt["private_field_path"])
    _require(
        file_sha256(private_path) == binding["private_field_raw_sha256"],
        "bridge private fields changed",
    )
    return receipt


def _load_source_builder_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = source_builder.load_config()
    acquisition, _ = source_builder._load_acquisition(config)
    receipt = _read_json(_repo_path(source_builder.OUTPUT_PATH), "source builder receipt")
    private_path = _repo_path(receipt["private_profile_path"])
    _require(
        file_sha256(private_path) == receipt["private_profile_raw_sha256"],
        "source profile bytes changed",
    )
    private = _read_json(private_path, "source profiles")
    _require(
        private["content_sha256"] == receipt["private_profile_content_sha256"],
        "source profile content changed",
    )
    return config, acquisition, private


def _source_summary_index(private: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for object_row in private["objects"]:
        for cell in object_row["cell_summaries"]:
            key = (object_row["object_id"], cell["cell_id"])
            _require(key not in result, "duplicate source builder cell")
            result[key] = cell
    _require(len(result) == 225, "source builder cell inventory changed")
    return result


def _solve_variant(
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
    screen = config["screen_contract"]
    normal = bridge_config["normalization_contract"]
    nodes = int(screen["nodes_per_axis"])
    grid = baseline.make_grid(nodes)
    half_box_pc = half_box_kpc * 1000.0
    coordinates_pc = grid.coordinates * half_box_pc
    spacing_pc = grid.spacing * half_box_pc
    surfaces = (
        ("stellar", np.asarray(stellar_surface, dtype=np.float64), hstar_pc),
        ("hi", np.asarray(maps["hi"], dtype=np.float64), hgas_pc),
        ("co", np.asarray(co_surface, dtype=np.float64), hgas_pc),
    )
    density_physical = np.zeros(grid.shape, dtype=np.float64)
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
        density_physical += component_density
        masses[f"{label}_mass_msun"] = mass
    a0_m_s2 = float(normal["a0_m_s2"])
    a0_pc = a0_m_s2 * float(normal["pc_m"]) / 1.0e6
    g_pc = float(normal["G_pc_km2_s2_msun"])
    density = density_physical * g_pc * half_box_pc / a0_pc
    total_mass = float(sum(masses.values()))
    expected_mass = g_pc * total_mass / (a0_pc * half_box_pc**2)
    deposited_mass = float(density.sum() * grid.spacing**3)
    rhs = 4.0 * math.pi * density
    boundary_samples = int(screen["boundary_integral_samples"])
    newton_boundary = bridge.spherical_boundary(
        grid, expected_mass, mond=False, integration_samples=boundary_samples
    )
    mond_boundary = bridge.spherical_boundary(
        grid, expected_mass, mond=True, integration_samples=boundary_samples
    )
    newton = baseline.solve_poisson(rhs, newton_boundary, grid.spacing)
    _, qumond, _ = baseline.solve_qumond(
        rhs,
        newton_boundary,
        mond_boundary,
        grid.spacing,
        a0=1.0,
        nu_floor=float(screen["qumond_nu_floor"]),
    )
    aqual = baseline.solve_aqual(
        rhs,
        mond_boundary,
        grid.spacing,
        a0=1.0,
        mu_floor=float(screen["aqual_mu_floor"]),
        damping=float(screen["aqual_damping"]),
        max_iterations=int(screen["aqual_max_iterations"]),
        delta_tolerance=float(screen["aqual_delta_tolerance"]),
        residual_tolerance=float(screen["aqual_residual_tolerance"]),
    )
    potentials = {
        "NEWTON": newton.potential,
        "AQUAL_SIMPLE_MU": aqual.potential,
        "QUMOND_SIMPLE_NU": qumond.potential,
    }
    profiles = {
        label: bridge.radial_acceleration_profile(
            potential,
            grid,
            half_box_kpc=half_box_kpc,
            radii_kpc=screen["radial_profile_radii_kpc"],
            azimuth_samples=int(screen["azimuth_samples"]),
            a0_m_s2=a0_m_s2,
        )
        for label, potential in potentials.items()
    }
    return {
        "nodes": nodes,
        "half_box_kpc": half_box_kpc,
        "grid_spacing_kpc": grid.spacing * half_box_kpc,
        "masses": masses,
        "total_mass_msun": total_mass,
        "dimensionless_mass_relative_error": abs(deposited_mass - expected_mass) / expected_mass,
        "solver_metrics": {
            "newton_relative_residual": newton.relative_residual,
            "qumond_relative_residual": qumond.relative_residual,
            "aqual_relative_residual": aqual.relative_residual,
            "aqual_converged": aqual.converged,
            "aqual_iterations": aqual.iterations,
        },
        "profiles": profiles,
        "field_hashes": {
            "density": bridge.array_sha256(density),
            **{
                f"{label.lower()}_potential": bridge.array_sha256(value)
                for label, value in potentials.items()
            },
        },
    }


def _mass_error(actual: Mapping[str, float], expected: Mapping[str, Any]) -> float:
    pairs = (
        ("stellar_mass_msun", "stellar_mass_msun"),
        ("hi_mass_msun", "hi_helium_mass_msun"),
        ("co_mass_msun", "co_helium_mass_msun"),
    )
    return float(
        max(
            abs(float(actual[key]) - float(expected[target]))
            / max(abs(float(expected[target])), 1.0e-30)
            for key, target in pairs
        )
    )


def _primary_bridge_difference(
    profiles: Mapping[str, Sequence[Mapping[str, float]]],
    bridge_object: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    expected_profiles = bridge_object["primary"]["profiles"]
    for operator, rows in profiles.items():
        expected = expected_profiles[operator]
        _require(len(rows) == len(expected), "bridge profile length changed")
        result[operator] = float(
            max(
                abs(
                    float(row["radial_acceleration_over_a0"])
                    - float(target["radial_acceleration_over_a0"])
                )
                / max(abs(float(target["radial_acceleration_over_a0"])), 1.0e-12)
                for row, target in zip(rows, expected, strict=True)
            )
        )
    return result


def _cell_gates(
    config: Mapping[str, Any],
    solved: Mapping[str, Any],
    *,
    source_mass_error: float,
    primary_bridge_difference: Mapping[str, float] | None,
) -> dict[str, bool]:
    gates = config["gate_contract"]
    metrics = solved["solver_metrics"]
    positive = all(
        math.isfinite(float(point["radial_acceleration_over_a0"]))
        and float(point["radial_acceleration_over_a0"]) > 0.0
        for rows in solved["profiles"].values()
        for point in rows
    )
    result = {
        "dimensionless_mass": solved["dimensionless_mass_relative_error"]
        <= gates["dimensionless_mass_relative_error_max"],
        "source_builder_mass": source_mass_error <= gates["source_builder_mass_relative_error_max"],
        "newton_residual": metrics["newton_relative_residual"]
        <= gates["linear_relative_residual_max"],
        "qumond_residual": metrics["qumond_relative_residual"]
        <= gates["linear_relative_residual_max"],
        "aqual_residual": metrics["aqual_relative_residual"]
        <= gates["aqual_relative_residual_max"],
        "aqual_converged": metrics["aqual_converged"] is True,
        "radial_acceleration": positive,
    }
    if primary_bridge_difference is not None:
        result["primary_vs_bound_bridge"] = (
            max(primary_bridge_difference.values())
            <= gates["primary_17_vs_bound_25_radial_relative_difference_max"]
        )
    return result


def _cell_id(
    beam: str,
    stellar: str,
    co_source: str,
    stellar_ratio: float,
    gas_height: float,
) -> str:
    return f"{beam}:{stellar}:{co_source}:HS{stellar_ratio:.15g}:HG{gas_height:.15g}"


def _equivalence_ledger(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for cell in cells:
        key = content_sha256(cell["field_hashes"])
        groups[key].append(f"{cell['object_id']}::{cell['cell_id']}")
    return [
        {"equivalence_sha256": key, "multiplicity": len(members), "members": sorted(members)}
        for key, members in sorted(groups.items())
    ]


def _envelopes(
    cells: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    radii = config["screen_contract"]["radial_profile_radii_kpc"]
    for object_id in config["objects"]:
        object_cells = [
            cell
            for cell in cells
            if cell["object_id"] == object_id and cell["cell_kind"] == "PRIMARY_CARTESIAN"
        ]
        for operator in config["screen_contract"]["operators"]:
            for index, radius in enumerate(radii):
                values = [
                    (
                        float(cell["profiles"][operator][index]["radial_acceleration_over_a0"]),
                        cell["cell_id"],
                    )
                    for cell in object_cells
                    if cell["all_numerical_gates_pass"]
                ]
                if not values:
                    rows.append(
                        {
                            "object_id": object_id,
                            "operator": operator,
                            "radius_kpc": float(radius),
                            "status": "NO_NUMERICALLY_VALID_CELL",
                        }
                    )
                    continue
                minimum = min(values)
                maximum = max(values)
                rows.append(
                    {
                        "object_id": object_id,
                        "operator": operator,
                        "radius_kpc": float(radius),
                        "status": "SOURCE_SYSTEMATIC_ENVELOPE",
                        "minimum_over_a0": minimum[0],
                        "minimum_cell_id": minimum[1],
                        "maximum_over_a0": maximum[0],
                        "maximum_cell_id": maximum[1],
                        "maximum_to_minimum_ratio": maximum[0] / minimum[0],
                        "valid_cell_count": len(values),
                    }
                )
    return rows


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    bridge_receipt = _validate_predecessor(config)
    bridge_config = bridge.load_config()
    source_config, acquisition, source_private = _load_source_builder_evidence()
    source_paths = source_builder._source_paths(acquisition)
    expected = _source_summary_index(source_private)
    bridge_objects = {row["object_id"]: row for row in bridge_receipt["objects"]}
    metadata_by_id = {row["object_id"]: row for row in source_config["objects"]}
    axes = config["cell_contract"]["primary_axes"]
    all_cells: list[dict[str, Any]] = []
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
        for beam, stellar, co_source, ratio, gas_height in itertools.product(
            axes["beam"],
            axes["stellar_mass_to_light"],
            axes["co_source"],
            axes["stellar_height_over_exponential_scale"],
            axes["gas_height_pc"],
        ):
            maps = maps_by_beam[beam]
            cell_id = _cell_id(beam, stellar, co_source, float(ratio), float(gas_height))
            stellar_surface = (
                maps["stellar_fixed"] if stellar == "FIXED_0P6" else maps["stellar_color"]
            )
            co_surface = maps["co"] if co_source == "WITH_CO" else np.zeros_like(maps["co"])
            solved = _solve_variant(
                config,
                bridge_config,
                maps,
                stellar_surface=stellar_surface,
                co_surface=co_surface,
                hstar_pc=rd_pc * float(ratio),
                hgas_pc=float(gas_height),
                half_box_kpc=float(config["screen_contract"]["solver_half_box_kpc_primary"]),
            )
            mass_error = _mass_error(solved["masses"], expected[(object_id, cell_id)])
            is_primary = (
                cell_id
                == source_private["objects"][config["objects"].index(object_id)]["primary_cell_id"]
            )
            bridge_difference = (
                _primary_bridge_difference(solved["profiles"], bridge_objects[object_id])
                if is_primary
                else None
            )
            gates = _cell_gates(
                config,
                solved,
                source_mass_error=mass_error,
                primary_bridge_difference=bridge_difference,
            )
            all_cells.append(
                {
                    "object_id": object_id,
                    "cell_id": cell_id,
                    "cell_kind": "PRIMARY_CARTESIAN",
                    **solved,
                    "source_builder_mass_relative_error": mass_error,
                    "primary_vs_bound_bridge_relative": bridge_difference,
                    "gates": gates,
                    "all_numerical_gates_pass": all(gates.values()),
                    "future_response_disposition": (
                        config["gate_contract"]["passed_cell_disposition"]
                        if all(gates.values())
                        else config["gate_contract"]["failed_cell_disposition"]
                    ),
                }
            )
        control_physics = config["cell_contract"]["control_source_physics"]
        for control in config["cell_contract"]["controls_per_object"]:
            maps = source_builder._surface_maps(
                source_config,
                metadata,
                images,
                n=int(control["source_pixels"]),
                box_kpc=float(control["source_box_kpc"]),
                beam=control_physics["beam"],
                use_sip=bool(control["sip"]),
            )
            solved = _solve_variant(
                config,
                bridge_config,
                maps,
                stellar_surface=maps["stellar_fixed"],
                co_surface=maps["co"],
                hstar_pc=rd_pc * float(control_physics["stellar_height_over_exponential_scale"]),
                hgas_pc=float(control_physics["gas_height_pc"]),
                half_box_kpc=float(control["solver_half_box_kpc"]),
            )
            mass_error = _mass_error(solved["masses"], expected[(object_id, control["id"])])
            gates = _cell_gates(
                config,
                solved,
                source_mass_error=mass_error,
                primary_bridge_difference=None,
            )
            all_cells.append(
                {
                    "object_id": object_id,
                    "cell_id": control["id"],
                    "cell_kind": "NUMERICAL_SOURCE_CONTROL",
                    **solved,
                    "source_builder_mass_relative_error": mass_error,
                    "primary_vs_bound_bridge_relative": None,
                    "gates": gates,
                    "all_numerical_gates_pass": all(gates.values()),
                    "future_response_disposition": (
                        config["gate_contract"]["passed_cell_disposition"]
                        if all(gates.values())
                        else config["gate_contract"]["failed_cell_disposition"]
                    ),
                }
            )
    _require(len(all_cells) == 225, "compiled cell count changed")
    _require(
        len({(row["object_id"], row["cell_id"]) for row in all_cells}) == 225, "duplicate cell"
    )
    equivalence = _equivalence_ledger(all_cells)
    envelopes = _envelopes(all_cells, config)
    counterexamples = [
        {
            "object_id": row["object_id"],
            "cell_id": row["cell_id"],
            "failed_gates": sorted(key for key, passed in row["gates"].items() if not passed),
        }
        for row in all_cells
        if not row["all_numerical_gates_pass"]
    ]
    passed = len(all_cells) - len(counterexamples)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "PASS_FULL_225_SOURCE_SCREEN_ZERO_NUMERICAL_COUNTEREXAMPLES"
            if not counterexamples
            else "PASS_FULL_225_SOURCE_SCREEN_WITH_RETAINED_NUMERICAL_COUNTEREXAMPLES"
        ),
        "decision": "SOURCE_SYSTEMATIC_ENVELOPES_READY_HIGH_RESOLUTION_AND_RESPONSE_STILL_UNRUN",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "predecessor_binding": config["predecessor_binding"],
        "admission_rule": config["admission_rule"],
        "cell_contract": config["cell_contract"],
        "screen_contract": config["screen_contract"],
        "gate_contract": config["gate_contract"],
        "cells": all_cells,
        "cell_count": len(all_cells),
        "numerical_pass_cell_count": passed,
        "numerical_counterexample_count": len(counterexamples),
        "counterexamples": counterexamples,
        "cell_ledger_root_sha256": content_sha256(all_cells),
        "equivalence_group_count": len(equivalence),
        "equivalence_ledger": equivalence,
        "equivalence_ledger_root_sha256": content_sha256(equivalence),
        "source_systematic_envelopes": envelopes,
        "source_systematic_envelope_root_sha256": content_sha256(envelopes),
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected = build_receipt(config)
    _require(dict(receipt) == expected, "receipt differs from exact source-only rebuild")


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


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    validate_receipt(config, _read_json(path, "receipt"))
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        print(write_receipt())
    elif arguments.command == "check":
        print(check_receipt())
    else:
        load_config()
        path = _repo_path(OUTPUT_PATH)
        if path.is_file():
            print(_read_json(path, "receipt")["status"])
        else:
            print("UNBUILT_RESPONSE_BLIND_SOURCE_SYSTEMATICS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
