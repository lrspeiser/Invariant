"""Score frozen environmental leads on the disjoint 139-galaxy SPARC development ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_post_response_environmental_placement_expansion_v1 as expansion,
)
from sigma_theory_compiler import sparc_full_sample
from sigma_theory_compiler.real_data_gravity_confrontation import baryonic_v_squared

CONFIG_PATH = Path("configs/open_gravity_sparc_139_environmental_generalization_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_sparc_139_environmental_generalization_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_sparc_139_environmental_generalization_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-sparc-139-environmental-generalization-v1/receipt.json"
)
PRIVATE_LEDGER_PATH = Path(
    "work/private/open-gravity-sparc-139-environmental-generalization-v1/score-ledger.json"
)

_CONFIG_RAW_SHA256 = "f5cfedeb21e2fff46b815d5d83605b3d975956d4637c73d1a623c4aacdf55a16"
_CONFIG_CONTENT_SHA256 = "c357dd0113a4624c3d4823af62bbb43243d5c69ec748c9b8b9e827e9f1bec8ec"
_MODULE_SEMANTIC_SHA256 = "1c1d1f2fc11344baaedd2c5e7aa353b84a23ae2545826a84e112432dc4fc7892"
_TEST_RAW_SHA256 = "2f2b16723c71a746208e33c2ed375d9b68bebf61c2d5aece8964c7437c81fab6"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_CONFIG_SCHEMA = "invariant-open-gravity-sparc-139-environmental-generalization-1.0"
_LEDGER_SCHEMA = "invariant-open-gravity-sparc-139-environmental-generalization-ledger-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-sparc-139-environmental-generalization-receipt-1.0"
_KPC_M = 3.085677581491367e19
_C_M_S = 299_792_458.0


class SparcGeneralizationError(RuntimeError):
    """Raised when a frozen source, formula, score, or output changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SparcGeneralizationError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    candidate = (root / relative).resolve()
    _require(candidate == root or root in candidate.parents, "path escaped repository")
    return candidate


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SparcGeneralizationError(f"invalid {label}") from error
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _CONFIG_SCHEMA, "config schema changed")
    _require(
        config["status"] == "FROZEN_POST_RESPONSE_DISJOINT_SPARC_DEVELOPMENT_GENERALIZATION",
        "status changed",
    )
    chronology = config["chronology_and_scope"]
    _require(
        chronology["candidate_ids_selected_after_PHANGS_XCOP_and_NGC2903_response_access"] is True,
        "chronology hidden",
    )
    _require(chronology["target_blind_claim"] is False, "target-blind overclaim enabled")
    _require(
        chronology["historical_139_exploration_ledger_is_disjoint_from_NGC2903"] is True,
        "disjointness lost",
    )
    _require(
        chronology["formula_or_grid_repair_after_this_freeze"] is False, "post-score repair enabled"
    )
    sparc = config["sparc_binding"]
    _require(sparc["scored_exploration_galaxies"] == 139, "galaxy count changed")
    _require(sparc["scored_exploration_rows"] == 2720, "row count changed")
    _require(sparc["historical_confirmation_galaxies_scored"] == 0, "confirmation scoring enabled")
    gate = config["published_and_analytic_admission_gate"]
    _require(len(gate["primary_sources"]) == 3, "primary source inventory changed")
    _require(
        gate["missing_source_disposition"] == "SOURCE_BLOCKED_RETAINED_NOT_SCORED",
        "source block changed",
    )
    source = config["source_reconstruction"]
    _require(source["density_driver_available"] is False, "density proxy enabled")
    _require(source["geometry_driver_available"] is False, "geometry proxy enabled")
    _require(
        source["source_values_may_not_use_Vobs_or_eVobs"] is True, "response leaked into sources"
    )
    _require(len(source["tail_systematic_cells"]) == 3, "tail cells changed")
    _require(
        source["response_based_source_cell_selection"] is False, "source-cell selection enabled"
    )
    candidates = config["candidate_program"]
    _require(len(candidates["published_control_ids"]) == 4, "control count changed")
    _require(len(candidates["frozen_post_response_signal_ids"]) == 14, "lead count changed")
    _require(candidates["formula_or_parameter_changes"] == 0, "formula changes enabled")
    _require(candidates["per_galaxy_parameters"] == 0, "per-galaxy parameters enabled")
    scoring = config["scoring_and_adjudication"]
    _require(scoring["minimum_meaningful_fractional_improvement"] == 0.02, "threshold changed")
    _require(scoring["minimum_object_support"] == 70, "support changed")
    _require(scoring["response_based_source_cell_selection"] is False, "response selection enabled")
    _require(scoring["theory_health_separate"] is True, "theory health used as veto")
    access = config["access_scope"]
    _require(access["development_only"] is True, "development boundary changed")
    _require(
        not any(value for key, value in access.items() if key != "development_only"),
        "forbidden access enabled",
    )
    _require(
        config["private_ledger_path"] == PRIVATE_LEDGER_PATH.as_posix(), "private path changed"
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")
    return config


def _verify_file(binding: Mapping[str, Any], path_key: str, hash_key: str) -> Path:
    path = _repo_path(str(binding[path_key]))
    _require(file_sha256(path) == binding[hash_key], f"binding changed: {path_key}")
    return path


def _load_predecessor(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["predecessor_binding"]
    for role in ("config", "module", "test", "receipt", "ledger"):
        _verify_file(binding, f"{role}_path", f"{role}_raw_sha256")
    receipt = _read_json(_repo_path(binding["receipt_path"]), "predecessor receipt")
    ledger = _read_json(_repo_path(binding["ledger_path"]), "predecessor ledger")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"],
        "predecessor receipt changed",
    )
    _require(
        ledger["content_sha256"] == binding["ledger_content_sha256"], "predecessor ledger changed"
    )
    _require(
        ledger["candidate_registry_sha256"] == binding["candidate_registry_sha256"],
        "predecessor registry changed",
    )
    _require(
        receipt["signal_ids"] == config["candidate_program"]["frozen_post_response_signal_ids"],
        "signal IDs changed",
    )
    return receipt, ledger


def _load_population(config: Mapping[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    binding = config["sparc_binding"]
    for path_key, hash_key in (
        ("dataset_path", "dataset_raw_sha256"),
        ("loader_path", "loader_raw_sha256"),
        ("loader_test_path", "loader_test_raw_sha256"),
        ("population_receipt_path", "population_receipt_raw_sha256"),
        ("g0_config_path", "g0_config_raw_sha256"),
        ("g0_module_path", "g0_module_raw_sha256"),
        ("g0_test_path", "g0_test_raw_sha256"),
        ("g0_receipt_path", "g0_receipt_raw_sha256"),
    ):
        _verify_file(binding, path_key, hash_key)
    g0 = _read_json(_repo_path(binding["g0_receipt_path"]), "G0 receipt")
    _require(g0["content_sha256"] == binding["g0_receipt_content_sha256"], "G0 receipt changed")
    population = sparc_full_sample.assemble(_root())
    exploration = tuple(population.exploration)
    _require(len(exploration) == 139, "SPARC exploration count changed")
    _require(sum(row.count for row in exploration) == 2720, "SPARC exploration rows changed")
    _require("NGC2903" not in {row.name for row in exploration}, "SPARC disjointness failed")
    return exploration, {
        "container_galaxies_parsed": 175,
        "container_rows_parsed": 3391,
        "scored_exploration_galaxies": len(exploration),
        "scored_exploration_rows": sum(row.count for row in exploration),
        "historical_confirmation_galaxies_scored": 0,
        "NGC2903_scored": False,
    }


def _radial_sources(
    radius_m: np.ndarray,
    g_b: np.ndarray,
    *,
    tail_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require(radius_m.ndim == g_b.ndim == 1 and radius_m.size == g_b.size, "invalid radial arrays")
    _require(radius_m.size >= 3 and bool(np.all(np.diff(radius_m) > 0.0)), "invalid radii")
    _require(bool(np.all(np.isfinite(g_b) & (g_b > 0.0))), "invalid baryonic acceleration")
    potential = np.empty_like(g_b)
    potential[-1] = tail_multiplier * g_b[-1] * radius_m[-1]
    for index in range(radius_m.size - 2, -1, -1):
        potential[index] = potential[index + 1] + 0.5 * (g_b[index] + g_b[index + 1]) * (
            radius_m[index + 1] - radius_m[index]
        )
    derivative = np.gradient(g_b, radius_m, edge_order=2)
    tidal = np.sqrt(derivative * derivative + 2.0 * (g_b / radius_m) ** 2)
    _require(bool(np.all(np.isfinite(potential) & (potential >= 0.0))), "invalid potential")
    _require(bool(np.all(np.isfinite(tidal) & (tidal > 0.0))), "invalid tidal proxy")
    return potential / (_C_M_S * _C_M_S), tidal


def _operator_benchmarks(config: Mapping[str, Any]) -> dict[str, Any]:
    gravity = 6.67430e-11
    mass = 1.0e41
    radius = np.geomspace(1.0e19, 1.0e21, 4097)
    g_b = gravity * mass / radius**2
    potential, tidal = _radial_sources(radius, g_b, tail_multiplier=1.0)
    exact_potential = gravity * mass / radius / (_C_M_S * _C_M_S)
    exact_tidal = math.sqrt(6.0) * gravity * mass / radius**3
    potential_error = float(np.max(np.abs(potential / exact_potential - 1.0)))
    tidal_error = float(np.max(np.abs(tidal[1:-1] / exact_tidal[1:-1] - 1.0)))
    _require(potential_error < 1.0e-6, "point-mass potential benchmark failed")
    _require(tidal_error < 1.0e-5, "point-mass tidal benchmark failed")
    source_config = expansion.source_compiler.load_config()
    placement = expansion._benchmark_placements(
        config={
            "candidate_program": {
                "placements": [
                    {"placement_id": "RAR_TRANSITION_SCALE"},
                    {"placement_id": "RAR_EXCESS_AMPLITUDE"},
                    {"placement_id": "RAR_LOW_FIELD_TOTAL_GAIN"},
                ]
            },
            "benchmarks_and_admission": {
                "high_acceleration_solar_recovery_max_fractional_deviation": 1.0e-6
            },
            "published_method_anchor": {
                "id": "RAR_2016",
                "url": "https://arxiv.org/abs/1609.05917",
            },
        },
        source_config=source_config,
    )
    return {
        "point_mass_potential_max_fractional_error": potential_error,
        "point_mass_tidal_interior_max_fractional_error": tidal_error,
        "RAR_at_F_equal_1_exact": placement["RAR_at_F_equal_1_exact"],
        "maximum_solar_fractional_deviation": placement["maximum_solar_fractional_deviation"],
        "all_operator_gates_pass": True,
        "primary_sources": config["published_and_analytic_admission_gate"]["primary_sources"],
    }


def _candidate_registry(
    config: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> list[dict[str, Any]]:
    wanted = (
        config["candidate_program"]["published_control_ids"]
        + config["candidate_program"]["frozen_post_response_signal_ids"]
    )
    by_id = {row["candidate_id"]: row for row in predecessor["candidate_registry"]}
    _require(set(wanted) <= set(by_id), "candidate missing from predecessor")
    rows = [copy.deepcopy(by_id[candidate_id]) for candidate_id in wanted]
    _require(
        len(rows) == 18 and len({row["candidate_id"] for row in rows}) == 18,
        "candidate inventory changed",
    )
    for row in rows:
        definition = row.get("base_driver_definition", {})
        drivers = (
            [definition["driver"]] if "driver" in definition else definition.get("drivers", [])
        )
        unavailable = sorted(set(drivers) & {"DENSITY", "GEOMETRY"})
        row["SPARC_radial_source_disposition"] = (
            "SOURCE_BLOCKED_RETAINED_NOT_SCORED" if unavailable else "SOURCE_READY_RADIAL_PROXY"
        )
        row["unavailable_drivers"] = unavailable
    return rows


def _galaxy_source(galaxy: Any, cell: Mapping[str, Any]) -> dict[str, Any]:
    radius_kpc = np.asarray([float(value) for value in galaxy.radius], dtype=float)
    radius_m = radius_kpc * _KPC_M
    vbar2_km2_s2 = np.asarray(
        [float(value) for value in baryonic_v_squared(galaxy, Fraction(1, 2), Fraction(7, 10))],
        dtype=float,
    )
    g_b = vbar2_km2_s2 * 1.0e6 / radius_m
    potential, tidal = _radial_sources(
        radius_m, g_b, tail_multiplier=float(cell["tail_multiplier"])
    )
    return {
        "object_id": galaxy.name,
        "cell_id": str(cell["cell_id"]),
        "radius_kpc": radius_kpc,
        "g_b_m_s2": g_b,
        "potential_depth_c2": potential,
        "tidal_s2": tidal,
        "v_obs_km_s": np.asarray([float(value) for value in galaxy.v_obs]),
        "e_v_obs_km_s": np.asarray([float(value) for value in galaxy.e_v_obs]),
    }


def _predict_candidate(
    source_config: dict[str, Any],
    candidate: Mapping[str, Any],
    source: Mapping[str, Any],
    references: dict[str, float],
) -> np.ndarray:
    values = []
    for index in range(len(source["radius_kpc"])):
        row = {
            "domain": "GALAXY_RADIAL_PROXY",
            "radius_kpc": float(source["radius_kpc"][index]),
            "g_b_m_s2": float(source["g_b_m_s2"][index]),
            "potential_depth_c2": float(source["potential_depth_c2"][index]),
            "density_kg_m3": None,
            "tidal_s2": float(source["tidal_s2"][index]),
            "geometry_asymmetry": None,
        }
        disposition, prediction, _factor = expansion.predict(
            source_config, candidate, row, references
        )
        _require(disposition == "COMPILED" and prediction is not None, "prediction failed")
        values.append(float(prediction))
    result = np.asarray(values)
    _require(bool(np.all(np.isfinite(result) & (result > 0.0))), "invalid predictions")
    return result


def _object_loss(source: Mapping[str, Any], acceleration: np.ndarray) -> dict[str, Any]:
    radius_m = np.asarray(source["radius_kpc"], dtype=float) * _KPC_M
    predicted = np.sqrt(acceleration * radius_m) / 1000.0
    observed = np.asarray(source["v_obs_km_s"], dtype=float)
    error = np.asarray(source["e_v_obs_km_s"], dtype=float)
    _require(bool(np.all(np.isfinite(error) & (error > 0.0))), "invalid SPARC errors")
    residual = (predicted - observed) / error
    square = residual * residual
    worst = int(np.argmax(square))
    return {
        "loss": float(np.mean(square)),
        "rows_scored": int(square.size),
        "worst_radius_kpc": float(source["radius_kpc"][worst]),
        "worst_standardized_residual": float(residual[worst]),
        "worst_standardized_square": float(square[worst]),
    }


def _fractional_improvement(candidate: float, control: float) -> float:
    _require(math.isfinite(candidate) and candidate >= 0.0, "invalid candidate loss")
    _require(math.isfinite(control) and control > 0.0, "invalid control loss")
    return (control - candidate) / control


def _score(
    config: Mapping[str, Any],
    population: Sequence[Any],
    predecessor: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = _candidate_registry(config, predecessor)
    source_config = expansion.source_compiler.load_config()
    source_prediction_ledger = expansion.sealed_scorer._load_prediction_ledger(
        expansion.sealed_scorer.load_config()
    )
    references = dict(source_prediction_ledger["reference_values"])
    cells = config["source_reconstruction"]["tail_systematic_cells"]
    sources = {
        (galaxy.name, str(cell["cell_id"])): _galaxy_source(galaxy, cell)
        for galaxy in population
        for cell in cells
    }
    score_rows = []
    blocked_rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate["SPARC_radial_source_disposition"].startswith("SOURCE_BLOCKED"):
            blocked_rows.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "REQUIRED_LOCAL_3D_DENSITY_OR_GEOMETRY_SOURCE_UNAVAILABLE",
                    "unavailable_drivers": candidate["unavailable_drivers"],
                    "retained": True,
                }
            )
            continue
        for cell in cells:
            cell_id = str(cell["cell_id"])
            objects = []
            for galaxy in population:
                source = sources[(galaxy.name, cell_id)]
                acceleration = _predict_candidate(source_config, candidate, source, references)
                objects.append(
                    {
                        "object_id": galaxy.name,
                        **_object_loss(source, acceleration),
                    }
                )
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "cell_id": cell_id,
                    "loss": float(np.mean([row["loss"] for row in objects])),
                    "object_count": len(objects),
                    "rows_scored": sum(row["rows_scored"] for row in objects),
                    "objects": objects,
                }
            )
    controls = config["candidate_program"]["published_control_ids"]
    by_key = {(row["candidate_id"], row["cell_id"]): row for row in score_rows}
    control_population = {
        str(cell["cell_id"]): min(
            float(by_key[(control, str(cell["cell_id"]))]["loss"]) for control in controls
        )
        for cell in cells
    }
    control_object = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        for galaxy in population:
            control_object[(cell_id, galaxy.name)] = min(
                float(
                    next(
                        row["loss"]
                        for row in by_key[(control, cell_id)]["objects"]
                        if row["object_id"] == galaxy.name
                    )
                )
                for control in controls
            )
    primary = str(config["scoring_and_adjudication"]["primary_source_cell"])
    threshold = float(
        config["scoring_and_adjudication"]["minimum_meaningful_fractional_improvement"]
    )
    minimum_support = int(config["scoring_and_adjudication"]["minimum_object_support"])
    summaries = []
    registry = {row["candidate_id"]: row for row in candidates}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate["SPARC_radial_source_disposition"].startswith("SOURCE_BLOCKED"):
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "kind": candidate["kind"],
                    "disposition": "SOURCE_BLOCKED_RETAINED_NOT_SCORED",
                    "unavailable_drivers": candidate["unavailable_drivers"],
                    "generalization_signal": False,
                    "retained": True,
                }
            )
            continue
        rows = [by_key[(candidate_id, str(cell["cell_id"]))] for cell in cells]
        improvements = [
            _fractional_improvement(float(row["loss"]), control_population[str(cell["cell_id"])])
            for row, cell in zip(rows, cells, strict=True)
        ]
        primary_row = by_key[(candidate_id, primary)]
        primary_objects = []
        for row in primary_row["objects"]:
            object_id = str(row["object_id"])
            primary_objects.append(
                {
                    **row,
                    "strongest_control_loss": control_object[(primary, object_id)],
                    "fractional_improvement": _fractional_improvement(
                        float(row["loss"]), control_object[(primary, object_id)]
                    ),
                    "improves_strongest_control": float(row["loss"])
                    < control_object[(primary, object_id)],
                }
            )
        support = sum(row["improves_strongest_control"] for row in primary_objects)
        primary_improvement = _fractional_improvement(
            float(primary_row["loss"]), control_population[primary]
        )
        is_new = str(candidate["kind"]).startswith("NEW_")
        signal = (
            is_new
            and primary_improvement > threshold
            and support >= minimum_support
            and float(np.median(improvements)) > 0.0
            and min(improvements) > 0.0
        )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "kind": registry[candidate_id]["kind"],
                "disposition": "SCORED",
                "primary_loss": float(primary_row["loss"]),
                "primary_strongest_control_loss": control_population[primary],
                "primary_fractional_improvement": primary_improvement,
                "primary_object_support": support,
                "primary_objects": primary_objects,
                "median_tail_loss": float(np.median([row["loss"] for row in rows])),
                "worst_tail_loss": max(float(row["loss"]) for row in rows),
                "median_paired_fractional_improvement": float(np.median(improvements)),
                "worst_paired_fractional_improvement": min(improvements),
                "best_source_cell_diagnostic_only": rows[int(np.argmax(improvements))]["cell_id"],
                "generalization_signal": signal,
                "retained": True,
            }
        )
    summaries.sort(
        key=lambda row: (
            -(float(row.get("primary_fractional_improvement", -math.inf))),
            row["candidate_id"],
        )
    )
    return {
        "candidate_count": len(candidates),
        "scored_candidate_count": sum(row["disposition"] == "SCORED" for row in summaries),
        "blocked_candidate_count": len(blocked_rows),
        "source_cell_count": len(cells),
        "score_row_count": len(score_rows),
        "formula_row_evaluations": sum(row["rows_scored"] for row in score_rows),
        "score_rows": score_rows,
        "blocked_rows": blocked_rows,
        "candidate_summaries": summaries,
        "generalization_signal_ids": [
            row["candidate_id"] for row in summaries if row["generalization_signal"]
        ],
        "all_candidates_retained": len(summaries),
        "theory_health_separate": True,
    }, candidates


def build_score_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    _predecessor_receipt, predecessor_ledger = _load_predecessor(config)
    population, access = _load_population(config)
    benchmarks = _operator_benchmarks(config)
    scores, candidates = _score(config, population, predecessor_ledger)
    ledger: dict[str, Any] = {
        "schema": _LEDGER_SCHEMA,
        "package_id": config["package_id"],
        "status": "POST_RESPONSE_DISJOINT_SPARC_DEVELOPMENT_GENERALIZATION_SCORED",
        "chronology_and_scope": config["chronology_and_scope"],
        "predecessor_binding": config["predecessor_binding"],
        "sparc_binding": config["sparc_binding"],
        "published_and_analytic_admission_gate": config["published_and_analytic_admission_gate"],
        "operator_benchmarks": benchmarks,
        "candidate_registry": candidates,
        "candidate_registry_sha256": content_sha256(candidates),
        "access": {**access, **config["access_scope"]},
        "scores": scores,
        "claim_boundary": {
            "post_response_disjoint_development_generalization_signal_only": True,
            "target_blind": False,
            "independent_confirmation": False,
            "full_3D_source_validation": False,
            "theory_health_established": False,
            "novelty_established": False,
            "publication_ready": False,
            "dark_matter_eliminated": False,
        },
    }
    ledger["score_rows_sha256"] = content_sha256(
        {"scored": scores["score_rows"], "blocked": scores["blocked_rows"]}
    )
    ledger["content_sha256"] = content_sha256(ledger)
    return ledger


def build_packet(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = build_score_ledger(config)
    signals = ledger["scores"]["generalization_signal_ids"]
    public_summaries = []
    for row in ledger["scores"]["candidate_summaries"]:
        public_summaries.append(
            {key: value for key, value in row.items() if key != "primary_objects"}
        )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": ledger["status"],
        "decision": (
            "POST_RESPONSE_DISJOINT_SPARC_DEVELOPMENT_GENERALIZATION_SIGNALS_FOUND"
            if signals
            else "NO_DISJOINT_SPARC_GENERALIZATION_SIGNAL_ALL_CANDIDATES_RETAINED"
        ),
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "chronology_and_scope": config["chronology_and_scope"],
        "predecessor_binding": config["predecessor_binding"],
        "sparc_binding": config["sparc_binding"],
        "operator_benchmarks": ledger["operator_benchmarks"],
        "access": ledger["access"],
        "candidate_count": ledger["scores"]["candidate_count"],
        "scored_candidate_count": ledger["scores"]["scored_candidate_count"],
        "blocked_candidate_count": ledger["scores"]["blocked_candidate_count"],
        "source_cell_count": ledger["scores"]["source_cell_count"],
        "score_row_count": ledger["scores"]["score_row_count"],
        "formula_row_evaluations": ledger["scores"]["formula_row_evaluations"],
        "generalization_signal_ids": signals,
        "candidate_summaries": public_summaries,
        "private_score_ledger_path": config["private_ledger_path"],
        "private_score_ledger_raw_sha256": hashlib.sha256(canonical_bytes(ledger)).hexdigest(),
        "private_score_ledger_content_sha256": ledger["content_sha256"],
        "score_rows_sha256": ledger["score_rows_sha256"],
        "claim_boundary": ledger["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return ledger, receipt


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
    ledger, receipt = build_packet(config)
    private = _atomic_no_clobber(_repo_path(PRIVATE_LEDGER_PATH), canonical_bytes(ledger))
    public = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return "CREATED" if "CREATED" in {private, public} else "EXISTING_IDENTICAL"


def check_packet() -> str:
    config = load_config()
    ledger, receipt = build_packet(config)
    _require(
        _repo_path(PRIVATE_LEDGER_PATH).read_bytes() == canonical_bytes(ledger),
        "private ledger does not rebuild",
    )
    _require(
        _repo_path(OUTPUT_PATH).read_bytes() == canonical_bytes(receipt), "receipt does not rebuild"
    )
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_packet())
    elif args.command == "check":
        print(check_packet())
    elif _repo_path(OUTPUT_PATH).exists():
        receipt = _read_json(_repo_path(OUTPUT_PATH), "receipt")
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "signals": receipt["generalization_signal_ids"],
                },
                sort_keys=True,
            )
        )
    else:
        print("UNWRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
