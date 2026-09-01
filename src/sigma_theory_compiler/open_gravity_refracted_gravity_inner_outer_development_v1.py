"""Retrospective development decomposition of nine published RG parameter cells."""

from __future__ import annotations

import argparse
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

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_sparc_development_score_v1 as fixed_score,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_phangs_things_scoring_resolution_v1 as resolution,
)
from sigma_theory_compiler import (
    open_gravity_refracted_gravity_published_prior_development_scan_v1 as prior,
)

CONFIG_PATH = Path("configs/open_gravity_refracted_gravity_inner_outer_development_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_refracted_gravity_inner_outer_development_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_refracted_gravity_inner_outer_development_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-inner-outer-development-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-refracted-gravity-inner-outer-development-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-refracted-gravity-inner-outer-development-receipt-1.0"
_CONFIG_RAW_SHA256 = "65d22084b80d121071d9831a35bdf7e4aeeea57a163c84d8094de7d063fa17e6"
_CONFIG_CONTENT_SHA256 = "f8a39077cff74d931093de5c683e84d0646201c1097d39a991c7a82d1636f9eb"
_MODULE_SEMANTIC_SHA256 = "6de1473a453e0f99d257b455a12fbfad573d3021cff502384e20422c0a15c21f"
_TEST_RAW_SHA256 = "b3416a367994f9b7c524208741a8b0ec8ea9742fd40b31f3dc069b2644a9a262"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_CONTROLS = [
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
]


class InnerOuterDevelopmentError(RuntimeError):
    """Fail-closed package error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InnerOuterDevelopmentError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
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
        raise InnerOuterDevelopmentError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"]
        == "FROZEN_RETROSPECTIVE_DEVELOPMENT_DECOMPOSITION_NINE_REGISTERED_EIGHT_SCORED_ONE_NUMERICALLY_BLOCKED",
        "status changed",
    )
    admission = config["builder_solver_admission"]
    policy_path = _repo_path(admission["policy_path"])
    _require(policy_path.is_file(), "builder/solver admission policy missing")
    _require(
        file_sha256(policy_path) == admission["policy_sha256"],
        "builder/solver admission policy changed",
    )
    _require(admission["real_public_source_maps_required"] is True, "source gate weakened")
    _require(admission["primary_measurement_papers_required"] is True, "paper gate weakened")
    _require(
        admission["independent_analytic_and_manufactured_benchmarks_required"] is True,
        "benchmark gate weakened",
    )
    _require(admission["one_failed_cell_cannot_prune_theory_family"] is True, "family pruned")
    parameters = config["parameter_contract"]
    _require(parameters["registered_cells"] == 9, "registered multiplicity changed")
    _require(parameters["multiplicity_charge"] == 9, "multiplicity charge changed")
    _require(len(parameters["source_admitted_cells"]) == 8, "admitted cells changed")
    _require(
        parameters["numerically_blocked_cells"] == ["PRIOR_CORNER_E0.1_Q2_R-23"],
        "blocked cell changed",
    )
    _require(parameters["new_cells_generated"] == 0, "new cells generated")
    _require(parameters["continuous_parameter_fits"] == 0, "continuous fit enabled")
    gate = config["source_admission_gate"]
    _require(gate["per_radius_relative_difference_max"] == 0.05, "point gate changed")
    _require(
        gate["maximum_fraction_of_full_radial_grid_exceeding_per_radius_gate"] == 0.05,
        "cell gate changed",
    )
    scoring = config["scoring_contract"]
    _require(scoring["comparators"] == _CONTROLS, "comparators changed")
    _require(scoring["no_threshold_or_parameter_tuning"] is True, "tuning enabled")
    _require(
        scoring["inner_outer_split_is_post_response_descriptive_not_candidate_selection"] is True,
        "post-response diagnostic promoted",
    )
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
    claims = config["claim_boundary"]
    for key in (
        "published_refracted_gravity_universality_established",
        "global_cross_tracer_signal_established",
        "source_map_truncation_explains_outer_failure",
        "tracer_systematics_resolved",
        "independent_confirmation",
        "lensing_closure_established",
        "relativistic_completion_established",
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
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "inner/outer config")
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected_roles = [
        "PRIMARY_3D_OPERATOR_BENCHMARK",
        "FINE_AND_CONVERGENCE_3D_SOURCES",
        "FIXED_MEDIAN_NEGATIVE_DEVELOPMENT_CONTROL",
        "BLOCKED_NINE_CELL_SCAN_SOURCE_FIELD_CACHE_AND_COUNTEREVIDENCE",
    ]
    _require(
        [row["role"] for row in config["predecessor_bindings"]] == expected_roles,
        "predecessor roles changed",
    )
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor artifact changed")
        receipt_artifact = next(
            row for row in binding["artifacts"] if row["path"].endswith("receipt.json")
        )
        receipt = _read_json(_repo_path(receipt_artifact["path"]), "predecessor receipt")
        _require(receipt["content_sha256"] == binding["receipt_content_sha256"], "receipt changed")
        receipts[binding["role"]] = receipt
    _require(
        receipts["FINE_AND_CONVERGENCE_3D_SOURCES"]["all_object_gates_pass"] is True,
        "source-resolution predecessor failed",
    )
    blocked = receipts["BLOCKED_NINE_CELL_SCAN_SOURCE_FIELD_CACHE_AND_COUNTEREVIDENCE"]
    _require(
        blocked["status"] == "BLOCKED_INSUFFICIENT_COMMON_CONVERGED_RADII_NO_RANKING",
        "blocked counterevidence status changed",
    )
    _require(blocked["phangs_object_scores"] == [], "blocked PHANGS scores unexpectedly present")
    _require(blocked["sparc_object_scores"] == [], "blocked SPARC scores unexpectedly present")
    _require(blocked["adjudication"]["performed"] is False, "blocked scan was ranked")
    return receipts


def _relative_profile_difference(
    fine_rows: Sequence[Mapping[str, Any]], coarse_rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    _require(len(fine_rows) == len(coarse_rows), "profile length changed")
    fine_radius = np.asarray([float(row["radius_kpc"]) for row in fine_rows])
    coarse_radius = np.asarray([float(row["radius_kpc"]) for row in coarse_rows])
    _require(np.array_equal(fine_radius, coarse_radius), "profile radii changed")
    fine = np.asarray([float(row["radial_acceleration_m_s2"]) for row in fine_rows])
    coarse = np.asarray([float(row["radial_acceleration_m_s2"]) for row in coarse_rows])
    return np.abs(fine - coarse) / np.maximum.reduce(
        [np.abs(fine), np.abs(coarse), np.full_like(fine, 1.0e-30)]
    )


def source_admission(
    config: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _require(len(source_rows) == 3, "source object count changed")
    _require(
        all(row["source_and_solver_gates_pass"] is True for row in source_rows),
        "source/solver gate failed",
    )
    cell_ids = [str(row["id"]) for row in prior.parameter_cells(prior.load_config())]
    threshold = float(config["source_admission_gate"]["per_radius_relative_difference_max"])
    fraction_max = float(
        config["source_admission_gate"][
            "maximum_fraction_of_full_radial_grid_exceeding_per_radius_gate"
        ]
    )
    evidence: list[dict[str, Any]] = []
    admitted: list[str] = []
    blocked: list[str] = []
    for cell_id in cell_ids:
        objects: list[dict[str, Any]] = []
        for source_row in source_rows:
            difference = _relative_profile_difference(
                source_row["fine"]["profiles"][cell_id],
                source_row["convergence"]["profiles"][cell_id],
            )
            _require(
                difference.size
                >= int(config["source_admission_gate"]["minimum_radial_grid_points"]),
                "radial grid shortened",
            )
            objects.append(
                {
                    "object_id": source_row["object_id"],
                    "profile_points": int(difference.size),
                    "points_over_gate": int(np.sum(difference > threshold)),
                    "fraction_over_gate": float(np.mean(difference > threshold)),
                    "median_relative_difference": float(np.median(difference)),
                    "maximum_relative_difference": float(np.max(difference)),
                }
            )
        maximum_fraction = max(float(row["fraction_over_gate"]) for row in objects)
        disposition = (
            "SOURCE_ADMITTED_FOR_SCORING"
            if maximum_fraction <= fraction_max
            else "NUMERICALLY_BLOCKED_RETAINED"
        )
        (admitted if disposition == "SOURCE_ADMITTED_FOR_SCORING" else blocked).append(cell_id)
        evidence.append(
            {
                "parameter_id": cell_id,
                "maximum_fraction_over_gate": maximum_fraction,
                "disposition": disposition,
                "objects": objects,
            }
        )
    _require(
        admitted == config["parameter_contract"]["source_admitted_cells"], "admitted set changed"
    )
    _require(
        blocked == config["parameter_contract"]["numerically_blocked_cells"], "blocked set changed"
    )
    return {"cells": evidence, "admitted": admitted, "blocked": blocked}


def _candidate_profiles(
    config: Mapping[str, Any], source_row: Mapping[str, Any], grid_key: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return prior._candidate_profiles(prior.load_config(), source_row, grid_key)


def score_object(
    config: Mapping[str, Any],
    source_row: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    *,
    asymmetric: bool,
    admitted: Sequence[str],
) -> dict[str, Any]:
    fine_radius, fine = _candidate_profiles(config, source_row, "fine")
    coarse_radius, coarse = _candidate_profiles(config, source_row, "convergence")
    _require(np.array_equal(fine_radius, coarse_radius), "radial grids changed")
    candidate_ids = _CONTROLS + list(admitted)
    threshold = float(config["source_admission_gate"]["per_radius_relative_difference_max"])
    used: list[Mapping[str, Any]] = []
    predictions: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    excluded = 0
    for response_row in response_rows:
        radius = float(response_row["radius_kpc"])
        in_range = 0.5 <= radius <= 15.0
        enough_cells = radius / 0.25 >= 2.0
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
                predictions[candidate_id].append(values[candidate_id])
        else:
            excluded += 1
    minimum = int(config["scoring_contract"]["minimum_rows_per_object"])
    _require(len(used) >= minimum, f"insufficient common rows: {source_row['object_id']}")
    radius = np.asarray([float(row["radius_kpc"]) for row in used])
    observed = np.asarray([float(row["velocity_km_s"]) for row in used])
    scores: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        predicted = (
            np.sqrt(np.asarray(predictions[candidate_id]) * radius * 3.085677581491367e19) / 1000.0
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
            "worst_radius_kpc": float(radius[worst]),
            "worst_standardized_residual": float(residual[worst]),
        }
    return {
        "object_id": source_row["object_id"],
        "rows_available": len(response_rows),
        "rows_scored_common": len(used),
        "rows_excluded": excluded,
        "scores": scores,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], candidate_id: str) -> float:
    return float(np.mean([float(row["scores"][candidate_id]["loss"]) for row in rows]))


def _fractional_improvement(candidate: float, comparator: float) -> float:
    _require(candidate >= 0.0 and comparator > 0.0, "invalid loss")
    return (comparator - candidate) / comparator


def _source_coverage() -> dict[str, Any]:
    source_config, acquisition, _, _ = resolution._source_evidence()
    maps, _ = resolution._primary_maps(source_config, acquisition, "NGC2903")
    radius = np.hypot(maps["x_pc"], maps["y_pc"]) / 1000.0
    components = {
        "stars": maps["stellar_fixed"],
        "hi": maps["hi"],
        "co": maps["co"],
        "total": maps["stellar_fixed"] + maps["hi"] + maps["co"],
    }
    result: dict[str, Any] = {}
    for name, surface_density in components.items():
        mass = surface_density * float(maps["dx_pc"]) ** 2
        total = float(np.sum(mass))
        _require(total > 0.0, "empty source component")
        result[name] = {
            "total_msun": total,
            "inside_5p675_fraction": float(np.sum(mass[radius <= 5.675]) / total),
            "outside_5p675_fraction": float(np.sum(mass[radius > 5.675]) / total),
            "inside_10_fraction": float(np.sum(mass[radius <= 10.0]) / total),
            "inside_15_fraction": float(np.sum(mass[radius <= 15.0]) / total),
            "maximum_nonzero_radius_kpc": float(np.max(radius[surface_density > 0.0])),
        }
    _require(result["total"]["outside_5p675_fraction"] > 0.3, "outer source mass vanished")
    _require(result["total"]["inside_15_fraction"] > 0.9, "15 kpc source coverage weakened")
    return result


def _primary_adjudication(
    config: Mapping[str, Any],
    phangs: Sequence[Mapping[str, Any]],
    sparc: Sequence[Mapping[str, Any]],
    admitted: Sequence[str],
) -> dict[str, Any]:
    candidate_ids = _CONTROLS + list(admitted)
    phangs_loss = {candidate: _aggregate(phangs, candidate) for candidate in candidate_ids}
    sparc_loss = {candidate: _aggregate(sparc, candidate) for candidate in candidate_ids}
    ranking = sorted(admitted, key=lambda candidate: (phangs_loss[candidate], candidate))
    best = ranking[0]
    best_phangs_control = min(_CONTROLS, key=phangs_loss.__getitem__)
    best_sparc_control = min(_CONTROLS, key=sparc_loss.__getitem__)
    phangs_improvement = _fractional_improvement(
        phangs_loss[best], phangs_loss[best_phangs_control]
    )
    sparc_improvement = _fractional_improvement(sparc_loss[best], sparc_loss[best_sparc_control])
    support = []
    for row in phangs:
        control = min(_CONTROLS, key=lambda candidate: row["scores"][candidate]["loss"])
        improvement = _fractional_improvement(
            float(row["scores"][best]["loss"]), float(row["scores"][control]["loss"])
        )
        support.append(
            {
                "object_id": row["object_id"],
                "best_comparator": control,
                "fractional_improvement": improvement,
                "supports": improvement > 0.0,
            }
        )
    checks = {
        "phangs_improvement_over_2_percent": phangs_improvement > 0.02,
        "at_least_two_phangs_objects_support": sum(row["supports"] for row in support) >= 2,
        "sparc_same_direction": sparc_improvement > 0.0,
    }
    return {
        "registered_multiplicity": 9,
        "scored_cells": len(admitted),
        "best_candidate": best,
        "best_phangs_control": best_phangs_control,
        "best_sparc_control": best_sparc_control,
        "phangs_fractional_improvement": phangs_improvement,
        "sparc_fractional_improvement": sparc_improvement,
        "phangs_support": support,
        "checks": checks,
        "global_cross_tracer_development_signal": all(checks.values()),
        "phangs_losses": phangs_loss,
        "sparc_losses": sparc_loss,
        "ranking": [
            {
                "rank": index + 1,
                "candidate": candidate,
                "phangs_loss": phangs_loss[candidate],
                "sparc_loss": sparc_loss[candidate],
                "phangs_improvement": _fractional_improvement(
                    phangs_loss[candidate], phangs_loss[best_phangs_control]
                ),
                "sparc_improvement": _fractional_improvement(
                    sparc_loss[candidate], sparc_loss[best_sparc_control]
                ),
            }
            for index, candidate in enumerate(ranking)
        ],
    }


def _inner_outer_diagnostic(
    config: Mapping[str, Any],
    source_row: Mapping[str, Any],
    phangs_rows: Sequence[Mapping[str, Any]],
    sparc_rows: Sequence[Mapping[str, Any]],
    admitted: Sequence[str],
    best_candidate: str,
) -> dict[str, Any]:
    overlap_min = max(
        min(float(row["radius_kpc"]) for row in phangs_rows),
        min(float(row["radius_kpc"]) for row in sparc_rows),
    )
    overlap_max = min(
        max(float(row["radius_kpc"]) for row in phangs_rows),
        max(float(row["radius_kpc"]) for row in sparc_rows),
    )
    phangs_inner = [
        row for row in phangs_rows if overlap_min <= float(row["radius_kpc"]) <= overlap_max
    ]
    sparc_inner = [
        row for row in sparc_rows if overlap_min <= float(row["radius_kpc"]) <= overlap_max
    ]
    sparc_outer = [row for row in sparc_rows if float(row["radius_kpc"]) > overlap_max]
    phangs_score = score_object(
        config, source_row, phangs_inner, asymmetric=True, admitted=admitted
    )
    sparc_inner_score = score_object(
        config, source_row, sparc_inner, asymmetric=False, admitted=admitted
    )
    sparc_outer_score = score_object(
        config, source_row, sparc_outer, asymmetric=False, admitted=admitted
    )
    phangs_radius = np.asarray([float(row["radius_kpc"]) for row in phangs_inner])
    phangs_velocity = np.asarray([float(row["velocity_km_s"]) for row in phangs_inner])
    phangs_sigma = np.asarray(
        [
            0.5 * (float(row["upper_error_km_s"]) + float(row["lower_error_km_s"]))
            for row in phangs_inner
        ]
    )
    sparc_radius = np.asarray([float(row["radius_kpc"]) for row in sparc_inner])
    sparc_velocity = np.asarray([float(row["velocity_km_s"]) for row in sparc_inner])
    sparc_sigma = np.asarray([float(row["error_km_s"]) for row in sparc_inner])
    phangs_interp = np.interp(sparc_radius, phangs_radius, phangs_velocity)
    phangs_sigma_interp = np.interp(sparc_radius, phangs_radius, phangs_sigma)
    fractional = np.abs(phangs_interp - sparc_velocity) / np.maximum(
        0.5 * (np.abs(phangs_interp) + np.abs(sparc_velocity)), 1.0e-12
    )
    standardized = (phangs_interp - sparc_velocity) / np.sqrt(
        phangs_sigma_interp**2 + sparc_sigma**2
    )

    def improvement(score: Mapping[str, Any], candidate: str) -> float:
        control_loss = min(float(score["scores"][control]["loss"]) for control in _CONTROLS)
        return _fractional_improvement(float(score["scores"][candidate]["loss"]), control_loss)

    candidates = [
        {
            "candidate": candidate,
            "phangs_inner_improvement": improvement(phangs_score, candidate),
            "sparc_inner_improvement": improvement(sparc_inner_score, candidate),
            "sparc_outer_improvement": improvement(sparc_outer_score, candidate),
        }
        for candidate in admitted
    ]
    best_row = next(row for row in candidates if row["candidate"] == best_candidate)
    _require(best_row["phangs_inner_improvement"] > 0.0, "inner PHANGS signal vanished")
    _require(best_row["sparc_inner_improvement"] > 0.0, "inner SPARC signal vanished")
    _require(best_row["sparc_outer_improvement"] < 0.0, "outer counterexample vanished")
    return {
        "boundary_is_post_response_descriptive": True,
        "radius_min_kpc": overlap_min,
        "radius_max_kpc": overlap_max,
        "phangs_rows_available": len(phangs_inner),
        "sparc_inner_rows_available": len(sparc_inner),
        "sparc_outer_rows_available": len(sparc_outer),
        "phangs_rows_scored": phangs_score["rows_scored_common"],
        "sparc_inner_rows_scored": sparc_inner_score["rows_scored_common"],
        "sparc_outer_rows_scored": sparc_outer_score["rows_scored_common"],
        "median_absolute_fractional_tracer_difference": float(np.median(fractional)),
        "maximum_absolute_fractional_tracer_difference": float(np.max(fractional)),
        "rms_standardized_tracer_difference": float(np.sqrt(np.mean(standardized**2))),
        "candidate_improvements": candidates,
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    predecessors = validate_predecessors(config)
    source_cache = predecessors["BLOCKED_NINE_CELL_SCAN_SOURCE_FIELD_CACHE_AND_COUNTEREVIDENCE"]
    source_rows = source_cache["source_field_rows"]
    source_gate = source_admission(config, source_rows)
    source_coverage = _source_coverage()
    # Material ordering: source/paper/numerical admission completes before response loaders run.
    fixed_config = fixed_score.load_config()
    fixed_evidence = fixed_score.validate_predecessors(fixed_config)
    response_config = fixed_evidence["response_config"]
    phangs_response, phangs_access = fixed_score.responses._load_phangs_responses(response_config)
    sparc_response, sparc_access = fixed_score.responses._load_sparc_responses(response_config)
    by_object = {row["object_id"]: row for row in source_rows}
    admitted = source_gate["admitted"]
    phangs_scores = [
        score_object(
            config,
            by_object[object_id],
            phangs_response[object_id],
            asymmetric=True,
            admitted=admitted,
        )
        for object_id in config["objects"]
    ]
    sparc_rows = fixed_score.responses._sparc_rows(sparc_response["NGC2903"])
    sparc_scores = [
        score_object(
            config,
            by_object["NGC2903"],
            sparc_rows,
            asymmetric=False,
            admitted=admitted,
        )
    ]
    primary = _primary_adjudication(config, phangs_scores, sparc_scores, admitted)
    diagnostic = _inner_outer_diagnostic(
        config,
        by_object["NGC2903"],
        phangs_response["NGC2903"],
        sparc_rows,
        admitted,
        primary["best_candidate"],
    )
    _require(primary["global_cross_tracer_development_signal"] is False, "global signal changed")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "NO_GLOBAL_CROSS_TRACER_DEVELOPMENT_SIGNAL_INNER_SHARED_RANGE_SIGNAL_OUTER_FAILURE_RETAINED",
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "predecessor_receipt_content_sha256": {
            role: row["content_sha256"] for role, row in predecessors.items()
        },
        "real_source_and_paper_anchors": config["real_source_and_paper_anchors"],
        "source_admission": source_gate,
        "ngc2903_source_coverage": source_coverage,
        "phangs_object_scores": phangs_scores,
        "sparc_object_scores": sparc_scores,
        "primary_adjudication": primary,
        "inner_outer_diagnostic": diagnostic,
        "publication_assessment": {
            "disposition": config["expected_outcome_contract"]["publication_disposition"],
            "potentially_interesting": True,
            "reason": "Two published low-permittivity cells improve both tracers on the shared inner range, but every admitted cell fails the full cross-tracer requirement and the PHANGS winner fails strongly in the outer SPARC disk.",
            "next_falsifier": "Repeat the frozen inner/outer decomposition on additional galaxies with resolved stellar, HI, and molecular source maps plus rotation curves extending beyond the molecular-disk range.",
        },
        "access_accounting": {
            "source_files_opened_before_response": config["access_scope"][
                "source_files_opened_per_build"
            ],
            "responses_opened_after_all_source_gates": True,
            "phangs": phangs_access,
            "sparc": sparc_access,
            "registered_multiplicity": 9,
            "source_admitted_cells_scored": 8,
            "numerically_blocked_cells_scored": 0,
            "object_candidate_scores_computed": 4 * (8 + 3),
            "post_response_diagnostic_score_sets": 3,
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
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(config), "receipt does not match deterministic rebuild")


def _output_path() -> Path:
    return _repo_path(OUTPUT_PATH)


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent receipt differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(_output_path(), canonical_bytes(build_receipt(config)))


def validate_receipt() -> None:
    config = load_config()
    path = _output_path()
    _require(path.is_file(), "receipt missing")
    validate_receipt_payload(config, _read_json(path, "receipt"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "check", "status"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command = _parser().parse_args(argv).command
    if command == "build":
        print(write_receipt())
    elif command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "best_candidate": receipt["primary_adjudication"]["best_candidate"],
                    "phangs_improvement": receipt["primary_adjudication"][
                        "phangs_fractional_improvement"
                    ],
                    "sparc_improvement": receipt["primary_adjudication"][
                        "sparc_fractional_improvement"
                    ],
                    "inner_outer": next(
                        row
                        for row in receipt["inner_outer_diagnostic"]["candidate_improvements"]
                        if row["candidate"] == receipt["primary_adjudication"]["best_candidate"]
                    ),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
