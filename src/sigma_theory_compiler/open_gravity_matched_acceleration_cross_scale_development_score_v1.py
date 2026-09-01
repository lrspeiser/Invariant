"""Score fixed matched-acceleration candidates on admitted development responses."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from sigma_theory_compiler import gravity_cluster_comparator_suite as cluster_suite
from sigma_theory_compiler import gravity_extended_source_clock_xcop_development as clock
from sigma_theory_compiler import gravity_gain_persistence_gp01_xcop_source_preflight as xcop_source
from sigma_theory_compiler import gravity_item7_baryonic_composition as item7
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59
from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as prediction_compiler,
)
from sigma_theory_compiler import sparc_full_sample

CONFIG_PATH = Path(
    "configs/open_gravity_matched_acceleration_cross_scale_development_score_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_matched_acceleration_cross_scale_development_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_matched_acceleration_cross_scale_development_score_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-matched-acceleration-cross-scale-development-score-v1/receipt.json"
)
PRIVATE_LEDGER_PATH = Path(
    "work/private/open-gravity-matched-acceleration-cross-scale-development-score-v1/score-ledger.json"
)

_CONFIG_RAW_SHA256 = "9d98a875024cb5503cd73525e5207c4075ee4cb52e49c116609b17a5da56dc11"
_CONFIG_CONTENT_SHA256 = "6e53b197919b42e998a8446016c5cfb22b1d023a4ef3dfcc1f951e42f542744f"
_MODULE_SEMANTIC_SHA256 = "43a92f8cbec48d1bfa88d640d35b6526616b2bb9c8448fceabdb1ddb75eb0e8d"
_TEST_RAW_SHA256 = "4c7f48d242eb5ead34ed741010b57de92f982d78a9b84d85a1f0f5755185074b"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_CONFIG_SCHEMA = "invariant-open-gravity-matched-acceleration-cross-scale-development-score-1.0"
_LEDGER_SCHEMA = (
    "invariant-open-gravity-matched-acceleration-cross-scale-development-score-ledger-1.0"
)
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-matched-acceleration-cross-scale-development-score-receipt-1.0"
)


class DevelopmentScoreError(RuntimeError):
    """Raised when a frozen scoring input or rule changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentScoreError(message)


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
        raise DevelopmentScoreError(f"invalid {label}") from error
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _CONFIG_SCHEMA, "config schema changed")
    _require(config["status"] == "FROZEN_DEVELOPMENT_RESPONSE_SCORER", "status changed")
    galaxy = config["galaxy_scoring"]
    _require(galaxy["source_cells"] == 75, "galaxy source-cell count changed")
    _require(galaxy["response_based_source_cell_selection"] is False, "response selection enabled")
    _require(galaxy["best_source_cell_is_diagnostic_only"] is True, "best cell promoted")
    xcop = config["xcop_scoring"]
    _require(len(xcop["nuisance_scenarios"]) == 3, "X-COP scenarios changed")
    _require(xcop["minimum_fractional_error"] == 0.05, "X-COP error floor changed")
    _require(xcop["outer_pressure_boundary_scored"] is False, "boundary score enabled")
    _require(xcop["density_used_as_response"] is False, "density target enabled")
    _require(xcop["inferred_total_mass_used"] is False, "total mass enabled")
    adjudication = config["adjudication"]
    _require(len(adjudication["published_control_ids"]) == 4, "control inventory changed")
    _require(adjudication["theory_health_can_veto_data_fit_signal"] is False, "theory veto enabled")
    _require(adjudication["theory_health_is_separate_followup"] is True, "theory follow-up lost")
    _require(
        adjudication["candidate_repair_or_tuning_after_response_access"] is False, "tuning enabled"
    )
    _require(adjudication["global_discovery_p_value_claimed"] is False, "global p-value enabled")
    access = config["access_scope"]
    _require(access["development_only"] is True, "development boundary changed")
    _require(
        not any(value for key, value in access.items() if key != "development_only"),
        "forbidden access enabled",
    )
    _require(
        config["private_score_ledger_path"] == PRIVATE_LEDGER_PATH.as_posix(),
        "private path changed",
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


def _load_prediction_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["prediction_binding"]
    for role in ("config", "module", "test", "receipt"):
        _verify_file(binding, f"{role}_path", f"{role}_raw_sha256")
    ledger_path = _verify_file(binding, "ledger_path", "ledger_raw_sha256")
    receipt = _read_json(_repo_path(binding["receipt_path"]), "prediction receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "prediction receipt changed"
    )
    ledger = _read_json(ledger_path, "prediction ledger")
    _require(
        ledger["content_sha256"] == binding["ledger_content_sha256"],
        "prediction ledger content changed",
    )
    _require(
        ledger["prediction_row_root_sha256"] == binding["prediction_row_root_sha256"],
        "prediction rows changed",
    )
    _require(len(ledger["candidate_registry"]) == 32, "candidate count changed")
    _require(
        ledger["scientific_boundary"]["scores_computed"] == 0, "source packet was already scored"
    )
    return ledger


def _load_phangs_responses(
    config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, Any]]:
    binding = config["response_bindings"]["phangs_co"]
    for role in ("config", "module", "test", "manifest"):
        _verify_file(binding, f"{role}_path", f"{role}_raw_sha256")
    item7_config = item7.load_config(_root())
    manifest = _read_json(_repo_path(binding["manifest_path"]), "PHANGS response manifest")
    item7.validate_source_manifest(manifest, item7_config)
    selected_ids = set(binding["scored_objects"])
    selected = {
        str(row["galaxy"]): [dict(point) for point in row["rotation_curve"]]
        for row in manifest["records"]
        if str(row["galaxy"]) in selected_ids
    }
    _require(set(selected) == selected_ids, "PHANGS selected objects changed")
    _require(
        len(manifest["records"]) == binding["container_galaxies_opened"], "PHANGS container changed"
    )
    _require(
        manifest["boundary"]["exploration_rotation_curve_rows"]
        == binding["container_response_rows_opened"],
        "PHANGS row count changed",
    )
    return selected, {
        "container_objects_opened": len(manifest["records"]),
        "container_response_rows_opened": manifest["boundary"]["exploration_rotation_curve_rows"],
        "scored_objects": sorted(selected),
        "selected_rows_available": sum(len(rows) for rows in selected.values()),
    }


def _load_sparc_responses(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["response_bindings"]["sparc"]
    dataset_path = _verify_file(binding, "dataset_path", "dataset_raw_sha256")
    _verify_file(binding, "module_path", "module_raw_sha256")
    _verify_file(binding, "test_path", "test_raw_sha256")
    galaxies, provenance = sparc_full_sample.load_full_sample(dataset_path)
    _require(len(galaxies) == binding["container_galaxies_opened"], "SPARC galaxy count changed")
    _require(
        sum(row.count for row in galaxies) == binding["container_response_rows_opened"],
        "SPARC row count changed",
    )
    wanted = set(binding["scored_objects"])
    selected = {row.name: row for row in galaxies if row.name in wanted}
    _require(set(selected) == wanted, "SPARC selected object changed")
    return selected, {
        "container_objects_opened": len(galaxies),
        "container_response_rows_opened": sum(row.count for row in galaxies),
        "scored_objects": sorted(selected),
        "selected_rows_available": sum(row.count for row in selected.values()),
        "dataset_sha256": provenance["dataset_sha256"],
    }


def _load_xcop_responses(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    binding = config["response_bindings"]["xcop"]
    for stem in ("clock", "item59", "cluster_suite"):
        for role in ("config", "module", "test"):
            _verify_file(binding, f"{stem}_{role}_path", f"{stem}_{role}_raw_sha256")
    clock_config = clock.load_config(_root())
    _require(
        content_sha256(clock_config["input_contract"]) == binding["input_contract_sha256"],
        "X-COP input contract changed",
    )
    payloads, files = clock._load_allowed_payloads(_root(), clock_config)
    _require(len(files) == binding["input_files_opened"], "X-COP file count changed")
    _require(
        sum(row["bytes"] for row in files) == binding["input_bytes_opened"], "X-COP bytes changed"
    )
    item59_config = item59.load_config(_root())
    packets = []
    for cluster in binding["scored_objects"]:
        packet = clock._parse_packet(cluster, payloads, item59_config)
        clock._add_rows(packet, item59_config)
        packets.append(packet)
    response_rows_opened = sum(
        int(packet["source_rows"]["pressure"]) + int(packet["source_rows"]["temperature"])
        for packet in packets
    )
    _require(
        response_rows_opened == binding["response_rows_opened"],
        "X-COP opened response rows changed",
    )
    _require(
        sum(len(packet["rows"]) for packet in packets) == binding["response_rows_scored"],
        "X-COP scored rows changed",
    )
    return (
        packets,
        item59_config,
        {
            "input_files_opened": len(files),
            "input_bytes_opened": sum(row["bytes"] for row in files),
            "response_rows_opened": response_rows_opened,
            "response_rows_scored": sum(len(packet["rows"]) for packet in packets),
            "scored_objects": [packet["cluster"] for packet in packets],
        },
    )


def _prediction_groups(ledger: Mapping[str, Any]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger["prediction_rows"]:
        source = row["source"]
        if source["domain"] == "GALAXY":
            groups[(str(source["object_id"]), str(source["source_cell_id"]))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: float(row["source"]["radius_kpc"]))
        _require(len(rows) == 60, "galaxy source radial grid changed")
    _require(len(groups) == 225, "galaxy source group count changed")
    return groups


def _velocity_score(
    source_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    response_rows: Sequence[Mapping[str, Any]],
    *,
    asymmetric: bool,
) -> dict[str, Any]:
    radius = np.asarray([float(row["source"]["radius_kpc"]) for row in source_rows])
    gravity = np.asarray(
        [float(row["predictions"][candidate_id]["g_prediction_m_s2"]) for row in source_rows]
    )
    velocity = np.sqrt(gravity * radius * 3.085677581491367e19) / 1000.0
    used = [
        row
        for row in response_rows
        if float(radius[0]) <= float(row["radius_kpc"]) <= float(radius[-1])
    ]
    _require(len(used) >= 5, "insufficient overlapping galaxy response rows")
    predicted = np.interp([float(row["radius_kpc"]) for row in used], radius, velocity)
    observed = np.asarray([float(row["velocity_km_s"]) for row in used])
    if asymmetric:
        upper = np.asarray([float(row["upper_error_km_s"]) for row in used])
        lower = np.asarray([float(row["lower_error_km_s"]) for row in used])
        sigma = np.where(predicted >= observed, upper, lower)
    else:
        sigma = np.asarray([float(row["error_km_s"]) for row in used])
    _require(bool(np.all(np.isfinite(sigma) & (sigma > 0.0))), "invalid galaxy errors")
    residual = (predicted - observed) / sigma
    square = residual * residual
    worst = int(np.argmax(square))
    return {
        "loss": float(np.mean(square)),
        "rows_scored": int(square.size),
        "rows_outside_source_grid": len(response_rows) - int(square.size),
        "worst_radius_kpc": float(used[worst]["radius_kpc"]),
        "worst_standardized_residual": float(residual[worst]),
        "worst_standardized_square": float(square[worst]),
    }


def _sparc_rows(galaxy: Any) -> list[dict[str, float]]:
    return [
        {
            "radius_kpc": float(radius),
            "velocity_km_s": float(velocity),
            "error_km_s": float(error),
        }
        for radius, velocity, error in zip(galaxy.radius, galaxy.v_obs, galaxy.e_v_obs, strict=True)
    ]


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "empty aggregation")
    worst = max(rows, key=lambda row: (float(row["loss"]), str(row["object_id"])))
    return {
        "loss": float(np.mean([float(row["loss"]) for row in rows])),
        "object_count": len(rows),
        "rows_scored": sum(int(row["rows_scored"]) for row in rows),
        "worst_object": str(worst["object_id"]),
        "worst_object_loss": float(worst["loss"]),
        "objects": list(rows),
    }


def _fractional_improvement(candidate: float, control: float) -> float:
    _require(math.isfinite(candidate) and candidate >= 0.0, "invalid candidate loss")
    _require(math.isfinite(control) and control > 0.0, "invalid control loss")
    return (control - candidate) / control


def _score_galaxies(
    config: Mapping[str, Any],
    ledger: Mapping[str, Any],
    phangs: Mapping[str, Sequence[Mapping[str, Any]]],
    sparc: Mapping[str, Any],
) -> dict[str, Any]:
    groups = _prediction_groups(ledger)
    candidates = ledger["candidate_registry"]
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    cell_ids = sorted({cell for _object, cell in groups})
    _require(len(cell_ids) == config["galaxy_scoring"]["source_cells"], "source cells changed")
    score_rows = []
    for candidate_id in candidate_ids:
        for cell_id in cell_ids:
            phangs_objects = []
            for object_id, response in sorted(phangs.items()):
                score = _velocity_score(
                    groups[(object_id, cell_id)],
                    candidate_id,
                    response,
                    asymmetric=True,
                )
                phangs_objects.append({"object_id": object_id, **score})
            sparc_objects = []
            for object_id, galaxy in sorted(sparc.items()):
                score = _velocity_score(
                    groups[(object_id, cell_id)],
                    candidate_id,
                    _sparc_rows(galaxy),
                    asymmetric=False,
                )
                sparc_objects.append({"object_id": object_id, **score})
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_cell_id": cell_id,
                    "PHANGS_CO": _aggregate(phangs_objects),
                    "SPARC": _aggregate(sparc_objects),
                }
            )
    controls = sorted(config["adjudication"]["published_control_ids"])
    by_key = {(row["candidate_id"], row["source_cell_id"]): row for row in score_rows}
    control_domain: dict[tuple[str, str], float] = {}
    control_object: dict[tuple[str, str, str], float] = {}
    for cell_id in cell_ids:
        for tracer in ("PHANGS_CO", "SPARC"):
            control_domain[(cell_id, tracer)] = min(
                float(by_key[(control, cell_id)][tracer]["loss"]) for control in controls
            )
            for object_row in by_key[(controls[0], cell_id)][tracer]["objects"]:
                object_id = str(object_row["object_id"])
                control_object[(cell_id, tracer, object_id)] = min(
                    float(
                        next(
                            row["loss"]
                            for row in by_key[(control, cell_id)][tracer]["objects"]
                            if row["object_id"] == object_id
                        )
                    )
                    for control in controls
                )
    primary = str(config["galaxy_scoring"]["primary_source_cell"])
    summaries = []
    candidate_kind = {str(row["candidate_id"]): str(row["kind"]) for row in candidates}
    for candidate_id in candidate_ids:
        summary: dict[str, Any] = {
            "candidate_id": candidate_id,
            "kind": candidate_kind[candidate_id],
        }
        for tracer in ("PHANGS_CO", "SPARC"):
            rows = [by_key[(candidate_id, cell_id)][tracer] for cell_id in cell_ids]
            primary_row = by_key[(candidate_id, primary)][tracer]
            improvements = [
                _fractional_improvement(float(row["loss"]), control_domain[(cell_id, tracer)])
                for cell_id, row in zip(cell_ids, rows, strict=True)
            ]
            support = sum(
                float(row["loss"]) < control_object[(primary, tracer, str(row["object_id"]))]
                for row in primary_row["objects"]
            )
            best_index = min(range(len(rows)), key=lambda index: float(rows[index]["loss"]))
            summary[tracer] = {
                "primary_loss": float(primary_row["loss"]),
                "primary_strongest_control_loss": control_domain[(primary, tracer)],
                "primary_fractional_improvement": _fractional_improvement(
                    float(primary_row["loss"]), control_domain[(primary, tracer)]
                ),
                "primary_object_support": support,
                "primary_objects": primary_row["objects"],
                "median_source_cell_loss": float(np.median([row["loss"] for row in rows])),
                "worst_source_cell_loss": max(float(row["loss"]) for row in rows),
                "median_paired_fractional_improvement": float(np.median(improvements)),
                "worst_paired_fractional_improvement": min(improvements),
                "best_source_cell_diagnostic_only": {
                    "source_cell_id": cell_ids[best_index],
                    "loss": float(rows[best_index]["loss"]),
                },
            }
        summaries.append(summary)
    return {
        "source_cell_count": len(cell_ids),
        "score_row_count": len(score_rows),
        "score_rows": score_rows,
        "candidate_summaries": summaries,
    }


def _loss_rows(
    predictions: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_fractional_error: float,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    detailed = []
    for row in rows:
        row_id = str(row["row_id"])
        observed = float(row["observed"])
        error = float(row["error"])
        predicted = float(predictions[row_id])
        _require(observed > 0.0 and error >= 0.0, "invalid X-COP response row")
        _require(math.isfinite(predicted) and predicted > 0.0, "invalid X-COP prediction")
        fractional = max(error / observed, minimum_fractional_error)
        residual = math.log(predicted / observed) / fractional
        square = residual * residual
        groups[str(row["observable"])].append(square)
        detailed.append(
            {
                "row_id": row_id,
                "radius_kpc": float(row["radius_kpc"]),
                "observable": str(row["observable"]),
                "standardized_residual": residual,
                "standardized_square": square,
            }
        )
    _require(set(groups) == {"pressure", "temperature"}, "X-COP observable groups changed")
    by_observable = {key: float(np.mean(values)) for key, values in groups.items()}
    worst = max(detailed, key=lambda row: (float(row["standardized_square"]), row["row_id"]))
    return {
        "loss": float(np.mean(list(by_observable.values()))),
        "by_observable": by_observable,
        "rows_scored": len(detailed),
        "worst_row_id": worst["row_id"],
        "worst_radius_kpc": worst["radius_kpc"],
        "worst_standardized_residual": worst["standardized_residual"],
        "worst_standardized_square": worst["standardized_square"],
    }


def _xcop_source_rows(
    prediction_config: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    item59_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    constants = item59_config["constants"]
    radius_m = np.asarray(state["radius_m"], dtype=float)
    mass = np.asarray(state["gas_mass"], dtype=float) + np.asarray(
        state["member_mass"], dtype=float
    )
    gravity = float(constants["gravity_si"])
    g_b = gravity * mass / radius_m**2
    density = xcop_source._effective_density(radius_m, mass)
    tidal = math.sqrt(2.0 / 3.0) * np.abs(4.0 * math.pi * gravity * density - 3.0 * g_b / radius_m)
    potential = prediction_compiler._spherical_potential(
        gravity,
        radius_m,
        mass,
        float(prediction_config["constants"]["c_m_s"]),
    )
    return [
        {
            "domain": "CLUSTER",
            "radius_kpc": float(state["calc_radius"][index]),
            "g_b_m_s2": float(g_b[index]),
            "potential_depth_c2": float(potential[index]),
            "density_kg_m3": float(density[index]),
            "tidal_s2": float(tidal[index]),
            "geometry_asymmetry": None,
        }
        for index in range(radius_m.size)
    ]


def _score_xcop(
    config: Mapping[str, Any],
    prediction_ledger: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    item59_config: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_config = prediction_compiler.load_config()
    candidates = prediction_ledger["candidate_registry"]
    references = prediction_ledger["reference_values"]
    score_rows = []
    blocked = []
    for scenario in config["xcop_scoring"]["nuisance_scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        nuisance = {key: float(value) for key, value in scenario.items() if key != "scenario_id"}
        for packet in packets:
            scaled = copy.deepcopy(dict(packet))
            scaled["ne_cm3"] = np.asarray(packet["ne_cm3"], dtype=float) * nuisance["density_scale"]
            state = cluster_suite._state(scaled, nuisance, item59_config)
            source_rows = _xcop_source_rows(prediction_config, state, item59_config)
            for candidate in candidates:
                candidate_id = str(candidate["candidate_id"])
                outputs = [
                    prediction_compiler.predict(prediction_config, candidate, row, references)
                    for row in source_rows
                ]
                dispositions = {row[0] for row in outputs}
                if dispositions != {"COMPILED"}:
                    _require(
                        dispositions == {"SOURCE_BLOCKED_DRIVER_UNAVAILABLE"},
                        "unexpected X-COP prediction disposition",
                    )
                    blocked.append(
                        {
                            "candidate_id": candidate_id,
                            "scenario_id": scenario_id,
                            "object_id": str(packet["cluster"]),
                            "reason": "NONSPHERICAL_GEOMETRY_DRIVER_UNAVAILABLE",
                        }
                    )
                    continue
                acceleration = np.asarray([float(row[1]) for row in outputs])
                predictions = cluster_suite._predictions_from_acceleration(
                    scaled, state, acceleration, nuisance, item59_config
                )
                score = _loss_rows(
                    predictions,
                    scaled["rows"],
                    minimum_fractional_error=float(
                        config["xcop_scoring"]["minimum_fractional_error"]
                    ),
                )
                score_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario_id": scenario_id,
                        "object_id": str(packet["cluster"]),
                        **score,
                    }
                )
    controls = sorted(config["adjudication"]["published_control_ids"])
    by_candidate_scenario: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_candidate_scenario[(row["candidate_id"], row["scenario_id"])].append(row)
    control_object: dict[tuple[str, str], float] = {}
    control_domain: dict[str, float] = {}
    scenarios = [str(row["scenario_id"]) for row in config["xcop_scoring"]["nuisance_scenarios"]]
    for scenario_id in scenarios:
        control_domain[scenario_id] = min(
            float(np.mean([row["loss"] for row in by_candidate_scenario[(control, scenario_id)]]))
            for control in controls
        )
        for packet in packets:
            object_id = str(packet["cluster"])
            control_object[(scenario_id, object_id)] = min(
                float(
                    next(
                        row["loss"]
                        for row in by_candidate_scenario[(control, scenario_id)]
                        if row["object_id"] == object_id
                    )
                )
                for control in controls
            )
    nominal_id = "XCOP-SOURCE-NOMINAL"
    summaries = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if (candidate_id, nominal_id) not in by_candidate_scenario:
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "kind": candidate["kind"],
                    "disposition": "SOURCE_BLOCKED_NONSPHERICAL_GEOMETRY",
                }
            )
            continue
        scenario_results = []
        for scenario_id in scenarios:
            object_rows = sorted(
                by_candidate_scenario[(candidate_id, scenario_id)],
                key=lambda row: row["object_id"],
            )
            aggregate = _aggregate(object_rows)
            aggregate["scenario_id"] = scenario_id
            aggregate["strongest_control_loss"] = control_domain[scenario_id]
            aggregate["fractional_improvement"] = _fractional_improvement(
                aggregate["loss"], control_domain[scenario_id]
            )
            aggregate["object_support"] = sum(
                float(row["loss"]) < control_object[(scenario_id, str(row["object_id"]))]
                for row in object_rows
            )
            scenario_results.append(aggregate)
        nominal = next(row for row in scenario_results if row["scenario_id"] == nominal_id)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "kind": candidate["kind"],
                "disposition": "SCORED",
                "nominal_loss": nominal["loss"],
                "nominal_strongest_control_loss": nominal["strongest_control_loss"],
                "nominal_fractional_improvement": nominal["fractional_improvement"],
                "nominal_object_support": nominal["object_support"],
                "median_nuisance_loss": float(np.median([row["loss"] for row in scenario_results])),
                "worst_nuisance_loss": max(row["loss"] for row in scenario_results),
                "worst_paired_fractional_improvement": min(
                    row["fractional_improvement"] for row in scenario_results
                ),
                "scenario_results": scenario_results,
            }
        )
    return {
        "score_row_count": len(score_rows),
        "blocked_row_count": len(blocked),
        "score_rows": score_rows,
        "blocked_rows": blocked,
        "candidate_summaries": summaries,
    }


def _cross_scale_adjudication(
    config: Mapping[str, Any],
    galaxy: Mapping[str, Any],
    xcop: Mapping[str, Any],
) -> dict[str, Any]:
    galaxy_map = {row["candidate_id"]: row for row in galaxy["candidate_summaries"]}
    xcop_map = {row["candidate_id"]: row for row in xcop["candidate_summaries"]}
    threshold = float(config["adjudication"]["minimum_meaningful_fractional_improvement"])
    rows = []
    for candidate_id, galaxy_row in galaxy_map.items():
        cluster_row = xcop_map[candidate_id]
        is_new = str(galaxy_row["kind"]).startswith("NEW_")
        phangs_improvement = float(galaxy_row["PHANGS_CO"]["primary_fractional_improvement"])
        phangs_support = int(galaxy_row["PHANGS_CO"]["primary_object_support"])
        sparc_improvement = float(galaxy_row["SPARC"]["primary_fractional_improvement"])
        if cluster_row["disposition"] == "SCORED":
            xcop_improvement = float(cluster_row["nominal_fractional_improvement"])
            xcop_support = int(cluster_row["nominal_object_support"])
            combined = 0.5 * (phangs_improvement + xcop_improvement)
            data_fit_signal = (
                is_new
                and phangs_improvement > threshold
                and xcop_improvement > threshold
                and phangs_support >= int(config["adjudication"]["phangs_minimum_object_support"])
                and xcop_support >= int(config["adjudication"]["xcop_minimum_object_support"])
            )
        else:
            xcop_improvement = None
            xcop_support = 0
            combined = None
            data_fit_signal = False
        domain_specific = is_new and (
            phangs_improvement > threshold
            or (xcop_improvement is not None and xcop_improvement > threshold)
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "kind": galaxy_row["kind"],
                "PHANGS_primary_fractional_improvement": phangs_improvement,
                "PHANGS_primary_object_support": phangs_support,
                "SPARC_primary_fractional_improvement": sparc_improvement,
                "XCOP_nominal_fractional_improvement": xcop_improvement,
                "XCOP_nominal_object_support": xcop_support,
                "combined_cross_scale_improvement": combined,
                "data_fit_signal": data_fit_signal,
                "domain_specific_signal": domain_specific,
                "theory_health_adjudicated": False,
                "retained": True,
            }
        )
    rows.sort(
        key=lambda row: (
            -(
                float(row["combined_cross_scale_improvement"])
                if row["combined_cross_scale_improvement"] is not None
                else -math.inf
            ),
            row["candidate_id"],
        )
    )
    return {
        "candidate_rows": rows,
        "data_fit_signal_ids": [row["candidate_id"] for row in rows if row["data_fit_signal"]],
        "domain_specific_signal_ids": [
            row["candidate_id"] for row in rows if row["domain_specific_signal"]
        ],
        "all_candidates_retained": len(rows),
        "theory_health_separate": True,
        "maximum_claim": config["adjudication"]["maximum_claim"],
    }


def build_score_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predictions = _load_prediction_ledger(config)
    phangs, phangs_access = _load_phangs_responses(config)
    sparc, sparc_access = _load_sparc_responses(config)
    packets, item59_config, xcop_access = _load_xcop_responses(config)
    galaxy_scores = _score_galaxies(config, predictions, phangs, sparc)
    xcop_scores = _score_xcop(config, predictions, packets, item59_config)
    adjudication = _cross_scale_adjudication(config, galaxy_scores, xcop_scores)
    ledger: dict[str, Any] = {
        "schema": _LEDGER_SCHEMA,
        "package_id": config["package_id"],
        "status": "DEVELOPMENT_RESPONSES_SCORED_NO_TUNING",
        "prediction_binding": config["prediction_binding"],
        "response_bindings": config["response_bindings"],
        "primary_data_and_method_anchors": config["primary_data_and_method_anchors"],
        "access": {
            "PHANGS_CO": phangs_access,
            "SPARC": sparc_access,
            "XCOP": xcop_access,
            **config["access_scope"],
        },
        "galaxy_scores": galaxy_scores,
        "xcop_scores": xcop_scores,
        "adjudication": adjudication,
        "claim_boundary": {
            "development_data_fit_signal_only": True,
            "independent_confirmation": False,
            "theory_health_established": False,
            "novelty_established": False,
            "publication_ready": False,
            "alternative_to_gr": False,
            "dark_matter_eliminated": False,
        },
    }
    ledger["score_rows_sha256"] = content_sha256(
        {
            "galaxy": galaxy_scores["score_rows"],
            "xcop": xcop_scores["score_rows"],
            "blocked": xcop_scores["blocked_rows"],
        }
    )
    ledger["content_sha256"] = content_sha256(ledger)
    return ledger


def build_packet(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = build_score_ledger(config)
    ranked = ledger["adjudication"]["candidate_rows"]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": ledger["status"],
        "decision": (
            "DEVELOPMENT_DATA_FIT_LEADS_FOUND_THEORY_AND_CONFIRMATION_PENDING"
            if ledger["adjudication"]["data_fit_signal_ids"]
            else "NO_CROSS_SCALE_DEVELOPMENT_DATA_FIT_SIGNAL_ALL_CANDIDATES_RETAINED"
        ),
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "prediction_binding": config["prediction_binding"],
        "response_bindings": config["response_bindings"],
        "primary_data_and_method_anchors": config["primary_data_and_method_anchors"],
        "access": ledger["access"],
        "candidate_count": len(ranked),
        "data_fit_signal_ids": ledger["adjudication"]["data_fit_signal_ids"],
        "domain_specific_signal_ids": ledger["adjudication"]["domain_specific_signal_ids"],
        "top_ranked_candidates": ranked[:10],
        "galaxy_source_cells": ledger["galaxy_scores"]["source_cell_count"],
        "galaxy_score_rows": ledger["galaxy_scores"]["score_row_count"],
        "xcop_score_rows": ledger["xcop_scores"]["score_row_count"],
        "xcop_blocked_rows": ledger["xcop_scores"]["blocked_row_count"],
        "private_score_ledger_path": config["private_score_ledger_path"],
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
    private_status = _atomic_no_clobber(
        _repo_path(config["private_score_ledger_path"]), canonical_bytes(ledger)
    )
    public_status = _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))
    return "CREATED" if "CREATED" in {private_status, public_status} else "EXISTING_IDENTICAL"


def check_packet() -> str:
    config = load_config()
    ledger, receipt = build_packet(config)
    _require(
        _repo_path(config["private_score_ledger_path"]).read_bytes() == canonical_bytes(ledger),
        "private score ledger does not rebuild",
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
                    "data_fit_signals": len(receipt["data_fit_signal_ids"]),
                    "domain_specific_signals": len(receipt["domain_specific_signal_ids"]),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps({"status": load_config()["status"], "decision": "NOT_WRITTEN"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
