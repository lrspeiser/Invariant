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

from sigma_theory_compiler import open_gravity_refracted_gravity_3d_primary_benchmark_v1 as rg
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_sparc_development_score_v1 as fixed_score,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as resolution,
)

CONFIG_PATH = Path(
    "configs/open_gravity_refracted_gravity_published_prior_development_scan_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/"
    "open_gravity_refracted_gravity_published_prior_development_scan_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_refracted_gravity_published_prior_development_scan_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-published-prior-development-scan-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-refracted-gravity-published-prior-development-scan-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-published-prior-development-scan-receipt-1.0"
)
_CONFIG_RAW_SHA256 = "56adb5a46903648aeaee0057535e42ab65b0e89dc040c76652e30a11ecbb8a9c"
_CONFIG_CONTENT_SHA256 = "d24e2b71d39acb9dd18a67a418149c7ecd756bef11111587b0dfa4e96dc90710"
_MODULE_SEMANTIC_SHA256 = "6a3462405b81afe85aa47b8c58c704f098e7f8a58312b45b086bdda8d109b853"
_TEST_RAW_SHA256 = "b58cfa5a92cd6f424b7d9a04381d279e0b7f020937677aed58d8e3b02294d1a6"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_CONTROL_IDS = [
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
]


class PriorScanError(RuntimeError):
    """Raised when the preregistered published-prior scan fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PriorScanError(message)


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
        raise PriorScanError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_NINE_CELL_PUBLISHED_PRIOR_DEVELOPMENT_SCAN",
        "status changed",
    )
    gate = config["admission_rule"]
    _require(gate["parameter_cells_frozen_before_this_response_scan"] is True, "post-hoc cells")
    _require(gate["new_parameter_generation_or_repair"] is False, "parameter repair enabled")
    _require(gate["all_source_and_solver_gates_complete_before_response_open"] is True, "order")
    _require(gate["best_cell_is_development_selection_not_confirmation"] is True, "overclaim")
    _require(config["objects"] == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    grid = config["grid_contract"]
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["fine_vs_convergence_maximum_relative_difference"] == 0.05, "gate changed")
    parameters = config["parameter_contract"]
    _require(parameters["registered_refracted_gravity_cells"] == 9, "cell count changed")
    _require(parameters["unique_refracted_gravity_epsilon_fields_per_source_grid"] == 6, "unique")
    _require(parameters["registered_cell_multiplicity_charged"] == 9, "multiplicity lost")
    _require(parameters["response_fitted_continuous_parameters"] == 0, "continuous fitting")
    _require(parameters["parameter_cells_added_after_fixed_median_result"] == 0, "post-hoc cells")
    _require(
        config["scoring_contract"]["response_based_source_or_row_threshold_selection"] is False,
        "response selection enabled",
    )
    _require(config["adjudication"]["multiplicity_adjusted_global_discovery_claim"] is False, "p")
    access = config["access_scope"]
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(access[key] == 0, f"forbidden access enabled: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


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


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, Any]:
    roles = [
        "FIXED_MEDIAN_DEVELOPMENT_RESULT",
        "SCORING_RESOLUTION_SOLVER_AND_SOURCES",
        "PUBLISHED_PARAMETER_AND_OPERATOR_BENCHMARK",
    ]
    _require(
        [row["role"] for row in config["predecessor_bindings"]] == roles,
        "predecessor roles changed",
    )
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        _require(binding["commit"] is None, "unexpected commit claim")
        _require(binding["promotion_authority"] is False, "authority overclaimed")
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor changed")
        receipt_artifact = next(
            row for row in binding["artifacts"] if row["path"].endswith("receipt.json")
        )
        receipt = _read_json(_repo_path(receipt_artifact["path"]), "predecessor receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        receipts[binding["role"]] = receipt
    fixed = receipts["FIXED_MEDIAN_DEVELOPMENT_RESULT"]
    _require(fixed["status"] == "NO_DEVELOPMENT_SIGNAL_FOR_FIXED_PUBLISHED_RG_CONTROL", "fixed")
    _require(fixed["access_accounting"]["tuning_calls"] == 0, "fixed tuning")
    scoring = receipts["SCORING_RESOLUTION_SOLVER_AND_SOURCES"]
    _require(scoring["all_object_gates_pass"] is True, "scoring solver failed")
    benchmark = receipts["PUBLISHED_PARAMETER_AND_OPERATOR_BENCHMARK"]
    _require(benchmark["benchmark_suite"]["failed"] == 0, "paper benchmark failed")
    return receipts


def parameter_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    benchmark_config = rg.load_config()
    cells = [dict(row) for row in rg.published_parameter_cells(benchmark_config)]
    _require(
        len(cells) == config["parameter_contract"]["registered_refracted_gravity_cells"], "cells"
    )
    _require(
        sum(
            row["id"] == config["parameter_contract"]["published_universal_median_id"]
            for row in cells
        )
        == 1,
        "median changed",
    )
    observed_bounds = {
        "epsilon_0": sorted(
            {row["epsilon_0"] for row in cells if row["id"] != "DISKMASS_UNIVERSAL_MEDIAN"}
        ),
        "Q": sorted({row["Q"] for row in cells if row["id"] != "DISKMASS_UNIVERSAL_MEDIAN"}),
        "log10_rho_c_g_cm3": sorted(
            {row["log10_rho_c_g_cm3"] for row in cells if row["id"] != "DISKMASS_UNIVERSAL_MEDIAN"}
        ),
    }
    _require(observed_bounds == config["parameter_contract"]["flat_prior_bounds"], "prior bounds")
    return cells


def _profile(
    potential: np.ndarray,
    built: Mapping[str, Any],
    config: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
) -> list[dict[str, float]]:
    return resolution.midplane_radial_profile(
        potential,
        built["grid"],
        half_box_kpc=float(config["grid_contract"]["solver_half_box_kpc"]),
        radii_kpc=resolution._radial_grid(resolution.load_config()),
        azimuth_samples=int(config["grid_contract"]["azimuth_samples"]),
        a0_m_s2=float(bridge_config["normalization_contract"]["a0_m_s2"]),
    )


def _solve_all_cells_on_grid(
    config: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    exponential_scale_pc: float,
    expected_source: Mapping[str, Any],
    bridge_config: Mapping[str, Any],
    nodes: int,
) -> dict[str, Any]:
    resolution_config = resolution.load_config()
    built = resolution._build_density(
        resolution_config,
        bridge_config,
        maps,
        exponential_scale_pc=exponential_scale_pc,
        nodes=nodes,
    )
    rhs = 4.0 * math.pi * built["density_dimensionless"]
    newton, newton_residual = resolution.solve_poisson_dst(
        rhs, built["newton_boundary"], built["grid"].spacing
    )
    profiles: dict[str, list[dict[str, float]]] = {
        "NEWTON_3D_DST": _profile(newton, built, config, bridge_config)
    }
    field_hashes: dict[str, str] = {"NEWTON_3D_DST": resolution.array_sha256(newton)}
    metrics: dict[str, dict[str, Any]] = {
        "NEWTON_3D_DST": {"converged": True, "relative_residual": newton_residual, "iterations": 1}
    }
    equivalence: dict[str, list[str]] = {}
    cache: dict[str, tuple[list[dict[str, float]], dict[str, Any], str]] = {}
    operator = resolution_config["operator_contract"]
    for cell in parameter_cells(config):
        epsilon = rg.published_permittivity(
            built["density_g_cm3"],
            epsilon_0=float(cell["epsilon_0"]),
            rho_c=10.0 ** float(cell["log10_rho_c_g_cm3"]),
            q_slope=float(cell["Q"]),
        )
        epsilon_hash = resolution.array_sha256(epsilon)
        if epsilon_hash not in cache:
            boundary = built["newton_boundary"] / float(cell["epsilon_0"])
            initial = newton / float(cell["epsilon_0"])
            potential, solve_metrics = resolution.solve_variable_pcg(
                rhs,
                boundary,
                epsilon,
                built["grid"].spacing,
                relative_tolerance=float(operator["pcg_relative_tolerance"]),
                absolute_tolerance=float(operator["pcg_absolute_tolerance"]),
                max_iterations=int(operator["pcg_max_iterations"]),
                initial_potential=initial,
            )
            cache[epsilon_hash] = (
                _profile(potential, built, config, bridge_config),
                solve_metrics,
                resolution.array_sha256(potential),
            )
            del potential, boundary, initial
            gc.collect()
        profile, solve_metrics, field_hash = cache[epsilon_hash]
        profiles[str(cell["id"])] = profile
        metrics[str(cell["id"])] = solve_metrics
        field_hashes[str(cell["id"])] = field_hash
        equivalence.setdefault(epsilon_hash, []).append(str(cell["id"]))
        del epsilon
    _require(
        len(cache)
        == config["parameter_contract"]["unique_refracted_gravity_epsilon_fields_per_source_grid"],
        "unique solve count changed",
    )
    source_mass_error = resolution.source_systematics._mass_error(built["masses"], expected_source)
    gates = resolution_config["benchmark_contract"]
    solver_pass = all(
        row["converged"] is True
        and float(row["relative_residual"]) <= float(gates["pcg_relative_residual_max"])
        for candidate_id, row in metrics.items()
        if candidate_id != "NEWTON_3D_DST"
    )
    result = {
        "nodes_per_axis": nodes,
        "spacing_kpc": built["grid"].spacing
        * float(config["grid_contract"]["solver_half_box_kpc"]),
        "source_builder_mass_relative_error": source_mass_error,
        "dimensionless_mass_relative_error": built["dimensionless_mass_relative_error"],
        "density_hash": resolution.array_sha256(built["density_dimensionless"]),
        "profiles": profiles,
        "solver_metrics": metrics,
        "field_hashes": field_hashes,
        "equivalence_groups": [
            {"epsilon_field_sha256": key, "members": sorted(value), "multiplicity": len(value)}
            for key, value in sorted(equivalence.items())
        ],
        "unique_rg_solves": len(cache),
        "source_mass_gate": source_mass_error
        <= float(resolution_config["benchmark_contract"]["source_mass_relative_error_max"]),
        "solver_gate": solver_pass,
    }
    del built, rhs, newton, cache
    gc.collect()
    return result


def build_source_fields(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_config, acquisition, expected, bridge_config = resolution._source_evidence()
    source_id = resolution.load_config()["source_cell"]["id"]
    rows: list[dict[str, Any]] = []
    for object_id in config["objects"]:
        maps, exponential_scale_pc = resolution._primary_maps(source_config, acquisition, object_id)
        convergence = _solve_all_cells_on_grid(
            config,
            maps,
            exponential_scale_pc=exponential_scale_pc,
            expected_source=expected[(object_id, source_id)],
            bridge_config=bridge_config,
            nodes=int(config["grid_contract"]["convergence_nodes_per_axis"]),
        )
        fine = _solve_all_cells_on_grid(
            config,
            maps,
            exponential_scale_pc=exponential_scale_pc,
            expected_source=expected[(object_id, source_id)],
            bridge_config=bridge_config,
            nodes=int(config["grid_contract"]["fine_nodes_per_axis"]),
        )
        rows.append(
            {
                "object_id": object_id,
                "convergence": convergence,
                "fine": fine,
                "source_and_solver_gates_pass": convergence["source_mass_gate"]
                and convergence["solver_gate"]
                and fine["source_mass_gate"]
                and fine["solver_gate"],
            }
        )
        del maps, convergence, fine
        gc.collect()
    return rows


def _candidate_profiles(
    config: Mapping[str, Any], object_row: Mapping[str, Any], grid_key: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    profiles = object_row[grid_key]["profiles"]
    newton_rows = profiles["NEWTON_3D_DST"]
    radius = np.asarray([float(row["radius_kpc"]) for row in newton_rows])
    newton = np.asarray([float(row["radial_acceleration_m_s2"]) for row in newton_rows])
    result = {
        candidate_id: np.asarray(
            [float(row["radial_acceleration_m_s2"]) for row in profiles[candidate_id]]
        )
        for candidate_id in [str(row["id"]) for row in parameter_cells(config)]
    }
    a0 = 1.2e-10
    result["NEWTON_3D_DST"] = newton
    result["RAR_2016_ON_NEWTON_3D"] = fixed_score._candidate_acceleration(
        "RAR_2016_ON_NEWTON_3D", newton, newton, a0_m_s2=a0
    )
    result["MOND_STANDARD_MU_ON_NEWTON_3D"] = fixed_score._candidate_acceleration(
        "MOND_STANDARD_MU_ON_NEWTON_3D", newton, newton, a0_m_s2=a0
    )
    return radius, result


def _score_object(
    config: Mapping[str, Any],
    source_row: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    asymmetric: bool,
    perform_scoring: bool = True,
) -> dict[str, Any]:
    fine_radius, fine = _candidate_profiles(config, source_row, "fine")
    coarse_radius, coarse = _candidate_profiles(config, source_row, "convergence")
    _require(np.array_equal(fine_radius, coarse_radius), "radial grids changed")
    candidate_ids = _CONTROL_IDS + [str(row["id"]) for row in parameter_cells(config)]
    threshold = float(config["grid_contract"]["fine_vs_convergence_maximum_relative_difference"])
    used: list[Mapping[str, Any]] = []
    prediction_rows: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    excluded: list[dict[str, Any]] = []
    for response_row in response_rows:
        radius = float(response_row["radius_kpc"])
        in_range = (
            float(config["grid_contract"]["radial_min_kpc"])
            <= radius
            <= float(config["grid_contract"]["radial_max_kpc"])
        )
        enough_cells = radius / float(config["grid_contract"]["fine_spacing_kpc"]) >= float(
            config["grid_contract"]["minimum_fine_cells_per_radius"]
        )
        values: dict[str, float] = {}
        differences: dict[str, float] = {}
        if in_range:
            for candidate_id in candidate_ids:
                fine_value = float(np.interp(radius, fine_radius, fine[candidate_id]))
                coarse_value = float(np.interp(radius, coarse_radius, coarse[candidate_id]))
                values[candidate_id] = fine_value
                differences[candidate_id] = abs(fine_value - coarse_value) / max(
                    abs(fine_value), abs(coarse_value), 1.0e-30
                )
        eligible = (
            in_range
            and enough_cells
            and all(value <= threshold for value in differences.values())
            and all(math.isfinite(value) and value > 0.0 for value in values.values())
        )
        if eligible:
            used.append(response_row)
            for candidate_id in candidate_ids:
                prediction_rows[candidate_id].append(values[candidate_id])
        else:
            excluded.append(
                {
                    "radius_kpc": radius,
                    "in_range": in_range,
                    "enough_fine_cells": enough_cells,
                    "maximum_relative_difference": max(differences.values(), default=None),
                }
            )
    minimum_rows = int(config["scoring_contract"]["minimum_rows_per_object"])
    public = {
        "object_id": source_row["object_id"],
        "rows_available": len(response_rows),
        "rows_eligible_common": len(used),
        "rows_excluded": len(excluded),
        "excluded_rows": excluded,
        "minimum_rows_required": minimum_rows,
        "common_convergence_gate_pass": len(used) >= minimum_rows,
        "eligibility_used_velocity_values": False,
        "scores": {},
    }
    if not perform_scoring:
        return public
    _require(public["common_convergence_gate_pass"], "rows")
    radius = np.asarray([float(row["radius_kpc"]) for row in used])
    observed = np.asarray([float(row["velocity_km_s"]) for row in used])
    scores: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        predicted = (
            np.sqrt(np.asarray(prediction_rows[candidate_id]) * radius * 3.085677581491367e19)
            / 1000.0
        )
        if asymmetric:
            upper = np.asarray([float(row["upper_error_km_s"]) for row in used])
            lower = np.asarray([float(row["lower_error_km_s"]) for row in used])
            sigma = np.where(predicted >= observed, upper, lower)
        else:
            sigma = np.asarray([float(row["error_km_s"]) for row in used])
        residual = (predicted - observed) / sigma
        squared = residual * residual
        worst = int(np.argmax(squared))
        scores[candidate_id] = {
            "loss": float(np.mean(squared)),
            "rows_scored": int(squared.size),
            "worst_radius_kpc": float(radius[worst]),
            "worst_standardized_residual": float(residual[worst]),
        }
    return {**public, "rows_scored_common": len(used), "scores": scores}


def _aggregate(objects: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    rows = [{"object_id": row["object_id"], **row["scores"][candidate_id]} for row in objects]
    worst = max(rows, key=lambda row: (float(row["loss"]), str(row["object_id"])))
    return {
        "loss": float(np.mean([float(row["loss"]) for row in rows])),
        "object_count": len(rows),
        "rows_scored": sum(int(row["rows_scored"]) for row in rows),
        "worst_object": worst["object_id"],
        "objects": rows,
    }


def _improvement(candidate: float, comparator: float) -> float:
    _require(candidate >= 0.0 and comparator > 0.0, "invalid loss")
    return (comparator - candidate) / comparator


def adjudicate(
    config: Mapping[str, Any],
    phangs: Sequence[Mapping[str, Any]],
    sparc: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rg_ids = [str(row["id"]) for row in parameter_cells(config)]
    candidate_ids = _CONTROL_IDS + rg_ids
    phangs_aggregates = {
        candidate_id: _aggregate(phangs, candidate_id) for candidate_id in candidate_ids
    }
    sparc_aggregates = {
        candidate_id: _aggregate(sparc, candidate_id) for candidate_id in candidate_ids
    }
    ranking = sorted(rg_ids, key=lambda row: (phangs_aggregates[row]["loss"], row))
    best_rg = ranking[0]
    best_control = min(_CONTROL_IDS, key=lambda row: phangs_aggregates[row]["loss"])
    best_sparc_control = min(_CONTROL_IDS, key=lambda row: sparc_aggregates[row]["loss"])
    phangs_improvement = _improvement(
        float(phangs_aggregates[best_rg]["loss"]), float(phangs_aggregates[best_control]["loss"])
    )
    sparc_improvement = _improvement(
        float(sparc_aggregates[best_rg]["loss"]),
        float(sparc_aggregates[best_sparc_control]["loss"]),
    )
    support_rows = []
    for object_row in phangs:
        best_object_control = min(_CONTROL_IDS, key=lambda row: object_row["scores"][row]["loss"])
        rg_loss = float(object_row["scores"][best_rg]["loss"])
        control_loss = float(object_row["scores"][best_object_control]["loss"])
        support_rows.append(
            {
                "object_id": object_row["object_id"],
                "best_comparator_id": best_object_control,
                "fractional_improvement": _improvement(rg_loss, control_loss),
                "supports_best_rg": rg_loss < control_loss,
            }
        )
    support = sum(row["supports_best_rg"] for row in support_rows)
    threshold = float(config["adjudication"]["minimum_meaningful_fractional_improvement"])
    checks = {
        "phangs_improvement_above_threshold": phangs_improvement > threshold,
        "phangs_object_support": support
        >= int(config["adjudication"]["phangs_minimum_object_support"]),
        "sparc_same_direction": sparc_improvement > 0.0,
    }
    parameters_by_id = {str(row["id"]): row for row in parameter_cells(config)}
    best_parameters = parameters_by_id[best_rg]
    bounds = config["parameter_contract"]["flat_prior_bounds"]
    boundary_hit = best_rg != config["parameter_contract"]["published_universal_median_id"] and any(
        float(best_parameters[key]) in [float(value) for value in bounds[key]]
        for key in ("epsilon_0", "Q", "log10_rho_c_g_cm3")
    )
    return {
        "registered_rg_cells": len(rg_ids),
        "multiplicity_charge": len(rg_ids),
        "rg_ranking_by_phangs_loss": [
            {
                "rank": index + 1,
                "parameter_id": candidate_id,
                "loss": phangs_aggregates[candidate_id]["loss"],
            }
            for index, candidate_id in enumerate(ranking)
        ],
        "best_rg_parameter_id": best_rg,
        "best_rg_parameters": best_parameters,
        "best_cell_at_published_prior_boundary": boundary_hit,
        "best_phangs_comparator_id": best_control,
        "best_sparc_comparator_id": best_sparc_control,
        "phangs_fractional_improvement": phangs_improvement,
        "sparc_fractional_improvement": sparc_improvement,
        "phangs_object_support_count": support,
        "phangs_object_comparisons": support_rows,
        "phangs_candidate_aggregates": phangs_aggregates,
        "sparc_candidate_aggregates": sparc_aggregates,
        "checks": checks,
        "development_signal": all(checks.values()),
        "multiplicity_adjusted_global_discovery_claimed": False,
        "source_systematic_score_robustness_established": False,
        "best_cell_is_development_selection_not_confirmation": True,
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predecessors = validate_predecessors(config)
    fields = build_source_fields(config)
    _require(
        all(row["source_and_solver_gates_pass"] for row in fields), "source/solver gate failed"
    )
    # This order is material: no response loader is called until every field cell has passed.
    fixed_config = fixed_score.load_config()
    fixed_evidence = fixed_score.validate_predecessors(fixed_config)
    response_config = fixed_evidence["response_config"]
    phangs_response, phangs_access = fixed_score.responses._load_phangs_responses(response_config)
    sparc_response, sparc_access = fixed_score.responses._load_sparc_responses(response_config)
    by_object = {row["object_id"]: row for row in fields}
    phangs_eligibility = [
        _score_object(
            config,
            by_object[object_id],
            phangs_response[object_id],
            asymmetric=True,
            perform_scoring=False,
        )
        for object_id in config["objects"]
    ]
    sparc_eligibility = [
        _score_object(
            config,
            by_object["NGC2903"],
            fixed_score.responses._sparc_rows(sparc_response["NGC2903"]),
            asymmetric=False,
            perform_scoring=False,
        )
    ]
    response_radius_gate_failures = [
        {
            "tracer": tracer,
            "object_id": row["object_id"],
            "rows_eligible_common": row["rows_eligible_common"],
            "minimum_rows_required": row["minimum_rows_required"],
        }
        for tracer, rows in (("PHANGS", phangs_eligibility), ("SPARC", sparc_eligibility))
        for row in rows
        if not row["common_convergence_gate_pass"]
    ]
    if response_radius_gate_failures:
        phangs_scores: list[dict[str, Any]] = []
        sparc_scores: list[dict[str, Any]] = []
        result = {
            "performed": False,
            "reason": "INSUFFICIENT_PREDECLARED_COMMON_CONVERGED_RADII",
            "registered_rg_cells": 9,
            "multiplicity_charge": 9,
            "rg_ranking_by_phangs_loss": [],
            "best_rg_parameter_id": None,
            "best_cell_at_published_prior_boundary": None,
            "phangs_fractional_improvement": None,
            "sparc_fractional_improvement": None,
            "development_signal": False,
            "multiplicity_adjusted_global_discovery_claimed": False,
            "source_systematic_score_robustness_established": False,
            "best_cell_is_development_selection_not_confirmation": False,
        }
    else:
        phangs_scores = [
            _score_object(
                config,
                by_object[object_id],
                phangs_response[object_id],
                asymmetric=True,
            )
            for object_id in config["objects"]
        ]
        sparc_scores = [
            _score_object(
                config,
                by_object["NGC2903"],
                fixed_score.responses._sparc_rows(sparc_response["NGC2903"]),
                asymmetric=False,
            )
        ]
        result = adjudicate(config, phangs_scores, sparc_scores)
    numerical_failures = [
        {
            "object_id": row["object_id"],
            "source_and_solver_gates_pass": row["source_and_solver_gates_pass"],
        }
        for row in fields
        if not row["source_and_solver_gates_pass"]
    ]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "BLOCKED_INSUFFICIENT_COMMON_CONVERGED_RADII_NO_RANKING"
            if response_radius_gate_failures
            else (
                "DEVELOPMENT_PRIOR_GRID_SIGNAL_RETAINED_WITH_MULTIPLICITY"
                if result["development_signal"]
                else "NO_DEVELOPMENT_SIGNAL_ACROSS_FROZEN_PUBLISHED_PRIOR_GRID"
            )
        ),
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "predecessor_receipt_content_sha256": {
            role: receipt["content_sha256"] for role, receipt in predecessors.items()
        },
        "primary_paper": config["primary_paper"],
        "parameter_cells": parameter_cells(config),
        "source_field_rows": fields,
        "numerical_failures": numerical_failures,
        "phangs_object_eligibility": phangs_eligibility,
        "sparc_object_eligibility": sparc_eligibility,
        "response_radius_gate_failures": response_radius_gate_failures,
        "phangs_object_scores": phangs_scores,
        "sparc_object_scores": sparc_scores,
        "adjudication": result,
        "access_accounting": {
            "source_files_opened_before_response": 21,
            "source_bytes_opened_before_response": 74030400,
            "source_grids_solved": 6,
            "registered_rg_field_rows": 3 * 2 * 9,
            "unique_rg_linear_solves": 3 * 2 * 6,
            "newton_linear_solves": 3 * 2,
            "responses_opened_after_all_source_gates": True,
            "phangs": phangs_access,
            "sparc": sparc_access,
            "response_velocity_values_used_for_scoring": 0
            if response_radius_gate_failures
            else sum(row["rows_scored_common"] for row in phangs_scores + sparc_scores),
            "object_candidate_scores_computed": 0
            if response_radius_gate_failures
            else (3 + 1) * (9 + 3),
            "registered_rg_multiplicity": 9,
            "best_cell_selection_events": 0 if response_radius_gate_failures else 1,
            "continuous_parameter_fits": 0,
            "confirmation_rows_opened": 0,
            "independent_rows_opened": 0,
            "group_rows_opened": 0,
            "lensing_rows_opened": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
        },
        "claim_boundary": {
            "all_nine_preregistered_cells_scored": not response_radius_gate_failures,
            "best_cell_is_development_selection": not response_radius_gate_failures,
            "global_significance_established": False,
            "source_systematic_score_robustness_established": False,
            "independent_confirmation": False,
            "cluster_fit_tested": False,
            "lensing_closure_established": False,
            "relativistic_completion_established": False,
            "novelty_established": False,
            "publication_ready": False,
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(config), "receipt does not match deterministic rebuild")


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
    receipt = build_receipt(load_config())
    return _atomic_no_clobber(_output_path(), canonical_bytes(receipt) + b"\n")


def validate_receipt() -> None:
    config = load_config()
    path = _output_path()
    _require(path.is_file(), "receipt missing")
    validate_receipt_payload(config, _read_json(path, "receipt"))


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
        receipt = build_receipt(load_config())
        adjudication = receipt["adjudication"]
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "best_rg_parameter_id": adjudication.get("best_rg_parameter_id"),
                    "best_at_prior_boundary": adjudication.get(
                        "best_cell_at_published_prior_boundary"
                    ),
                    "phangs_improvement": adjudication.get("phangs_fractional_improvement"),
                    "sparc_improvement": adjudication.get("sparc_fractional_improvement"),
                    "development_signal": adjudication["development_signal"],
                    "response_radius_gate_failures": receipt["response_radius_gate_failures"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
