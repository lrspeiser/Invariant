"""Score frozen post-response RAR environmental placements on development data."""

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

from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_development_score_v1 as sealed_scorer,
)
from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as source_compiler,
)

CONFIG_PATH = Path("configs/open_gravity_post_response_environmental_placement_expansion_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_post_response_environmental_placement_expansion_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_post_response_environmental_placement_expansion_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-post-response-environmental-placement-expansion-v1/receipt.json"
)
PRIVATE_LEDGER_PATH = Path(
    "work/private/open-gravity-post-response-environmental-placement-expansion-v1/score-ledger.json"
)

_CONFIG_RAW_SHA256 = "76de0071247f77ad1e0c0bc08e6d4bfebe39fbe72a546a88058f21a36fcb1204"
_CONFIG_CONTENT_SHA256 = "c5494c9739f76e3da17e5b6e717a33194bf8be141202eb414ffd22bf84b316bf"
_MODULE_SEMANTIC_SHA256 = "43bdfebd7b8d19c3db00abeee97a2b3d4edf875cecb61e869390fab558927fb5"
_TEST_RAW_SHA256 = "f0ce40471bfd89abc29d6c9aeaff146924ed27e435b6936d5af0be5796e4901d"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

_CONFIG_SCHEMA = "invariant-open-gravity-post-response-environmental-placement-expansion-1.0"
_LEDGER_SCHEMA = "invariant-open-gravity-post-response-environmental-placement-expansion-ledger-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-post-response-environmental-placement-expansion-receipt-1.0"
)


class PlacementExpansionError(RuntimeError):
    """Raised when a frozen placement, input, score, or output changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlacementExpansionError(message)


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
        raise PlacementExpansionError(f"invalid {label}") from error
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _CONFIG_SCHEMA, "config schema changed")
    _require(
        config["status"] == "FROZEN_POST_RESPONSE_DEVELOPMENT_REPAIR_EXPLORATION",
        "status changed",
    )
    chronology = config["chronology_and_honesty"]
    _require(chronology["designed_after_development_response_access"] is True, "chronology hidden")
    _require(chronology["target_blind_claim"] is False, "target-blind overclaim enabled")
    _require(chronology["independent_confirmation_required"] is True, "confirmation dropped")
    _require(chronology["post_score_formula_repair_allowed"] is False, "post-score repair enabled")
    program = config["candidate_program"]
    _require(
        program["driver_cells_inherited_exactly_from_source_prediction_packet"] == 28,
        "driver count changed",
    )
    _require(len(program["placements"]) == 3, "placement count changed")
    _require(program["new_candidate_count"] == 84, "new-candidate count changed")
    _require(program["total_candidate_count_with_controls"] == 88, "total candidate count changed")
    _require(program["same_constants_across_every_object"] is True, "object constants enabled")
    _require(program["per_object_parameters"] == 0, "per-object parameters enabled")
    _require(program["candidate_generation_after_this_freeze"] == 0, "candidate generation enabled")
    gate = config["benchmarks_and_admission"]
    _require(gate["real_source_data_required"] is True, "real-source gate removed")
    _require(gate["primary_paper_or_exact_analytic_limit_required"] is True, "paper gate removed")
    _require(gate["RAR_at_F_equal_1"] is True, "RAR limit removed")
    _require(
        gate["failed_benchmark_disposition"] == "RETAINED_NOT_SCORED", "failure policy changed"
    )
    scoring = config["scoring_and_adjudication"]
    _require(
        scoring["reuse_exact_response_loaders_losses_source_cells_and_nuisances_from_bound_scorer"]
        is True,
        "scoring reuse lost",
    )
    _require(scoring["all_candidates_and_failures_retained"] is True, "failure retention lost")
    _require(scoring["theory_health_separate"] is True, "theory health used as score veto")
    access = config["access_scope"]
    _require(access["development_responses_reused"] is True, "development response scope changed")
    _require(
        not any(value for key, value in access.items() if key != "development_responses_reused"),
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


def _load_predecessors(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = config["source_prediction_binding"]
    for role in ("config", "module", "test", "ledger"):
        _verify_file(source, f"{role}_path", f"{role}_raw_sha256")
    scorer_binding = config["scorer_binding"]
    for role in ("config", "module", "test", "receipt"):
        _verify_file(scorer_binding, f"{role}_path", f"{role}_raw_sha256")
    trigger = _read_json(_repo_path(scorer_binding["receipt_path"]), "trigger receipt")
    _require(
        trigger["content_sha256"] == scorer_binding["receipt_content_sha256"],
        "trigger receipt content changed",
    )
    _require(trigger["data_fit_signal_ids"] == [], "trigger result unexpectedly changed")
    scorer_config = sealed_scorer.load_config()
    source_config = source_compiler.load_config()
    source_ledger = sealed_scorer._load_prediction_ledger(scorer_config)
    _require(
        source_ledger["content_sha256"] == source["ledger_content_sha256"],
        "source ledger content changed",
    )
    _require(len(source_ledger["candidate_registry"]) == 32, "source candidate inventory changed")
    return source_config, scorer_config, source_ledger


def candidate_registry(
    config: Mapping[str, Any], source_ledger: Mapping[str, Any]
) -> list[dict[str, Any]]:
    controls = set(config["candidate_program"]["published_controls"])
    source_rows = list(source_ledger["candidate_registry"])
    control_rows = [copy.deepcopy(row) for row in source_rows if row["candidate_id"] in controls]
    _require(len(control_rows) == 4, "published control inventory changed")
    drivers = [row for row in source_rows if row["candidate_id"] not in controls]
    _require(len(drivers) == 28, "driver inventory changed")
    rows = control_rows
    for placement in config["candidate_program"]["placements"]:
        for driver in drivers:
            rows.append(
                {
                    "candidate_id": f"{placement['placement_id']}__{driver['candidate_id']}",
                    "kind": "NEW_POST_RESPONSE_REPAIR_HYPOTHESIS",
                    "placement_id": placement["placement_id"],
                    "placement_equation": placement["equation"],
                    "base_driver_id": driver["candidate_id"],
                    "base_driver_definition": copy.deepcopy(driver),
                    "chronology": "DESIGNED_AFTER_DEVELOPMENT_RESPONSE_ACCESS",
                    "target_blind": False,
                    "independent_confirmation_required": True,
                }
            )
    _require(len(rows) == 88, "candidate registry count changed")
    _require(len({row["candidate_id"] for row in rows}) == 88, "candidate IDs collided")
    return rows


def _factor(
    source_config: dict[str, Any],
    candidate: Mapping[str, Any],
    source: dict[str, Any],
    references: dict[str, float],
) -> float | None:
    base = candidate["base_driver_definition"]
    if "driver" in base:
        return source_compiler._driver_factor(
            source_config, source, references, str(base["driver"]), float(base["exponent"])
        )
    values = [
        source_compiler._driver_factor(
            source_config, source, references, str(driver), float(exponent)
        )
        for driver, exponent in zip(base["drivers"], base["exponents"], strict=True)
    ]
    if any(value is None for value in values):
        return None
    return min(max(math.prod(float(value) for value in values), 1.0 / 256.0), 256.0)


def placed_rar(g_b: float, a0: float, factor: float, placement_id: str) -> float:
    _require(math.isfinite(g_b) and g_b >= 0.0, "invalid baryonic acceleration")
    _require(math.isfinite(a0) and a0 > 0.0, "invalid acceleration scale")
    _require(math.isfinite(factor) and factor > 0.0, "invalid environmental factor")
    base = source_compiler.rar_2016(g_b, a0)
    if placement_id == "RAR_TRANSITION_SCALE":
        value = source_compiler.rar_2016(g_b, a0 * factor)
    elif placement_id == "RAR_EXCESS_AMPLITUDE":
        value = g_b + factor * (base - g_b)
    elif placement_id == "RAR_LOW_FIELD_TOTAL_GAIN":
        switch = 1.0 / (1.0 + (g_b / a0) ** 2)
        value = base * (1.0 + switch * (factor - 1.0))
    else:
        raise PlacementExpansionError("unknown placement")
    _require(math.isfinite(value) and value >= 0.0, "invalid placed prediction")
    return value


def predict(
    source_config: dict[str, Any],
    candidate: Mapping[str, Any],
    source: dict[str, Any],
    references: dict[str, float],
) -> tuple[str, float | None, float | None]:
    if candidate["kind"].startswith("PUBLISHED"):
        return source_compiler.predict(source_config, dict(candidate), source, references)
    factor = _factor(source_config, candidate, source, references)
    if factor is None:
        return "SOURCE_BLOCKED_DRIVER_UNAVAILABLE", None, None
    value = placed_rar(
        float(source["g_b_m_s2"]),
        float(source_config["constants"]["a0_m_s2"]),
        factor,
        str(candidate["placement_id"]),
    )
    return "COMPILED", value, factor


def _galaxy_prediction_ledger(
    source_config: dict[str, Any],
    source_ledger: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    references = dict(source_ledger["reference_values"])
    rows = []
    for original in source_ledger["prediction_rows"]:
        source = dict(original["source"])
        if source["domain"] != "GALAXY":
            continue
        predictions = {}
        for candidate in candidates:
            disposition, value, factor = predict(source_config, candidate, source, references)
            _require(disposition == "COMPILED", "galaxy prediction unexpectedly blocked")
            predictions[str(candidate["candidate_id"])] = {
                "disposition": disposition,
                "g_prediction_m_s2": value,
                "environmental_factor": factor,
            }
        rows.append({"source": source, "predictions": predictions})
    _require(len(rows) == 13_500, "galaxy source-row count changed")
    return {
        "candidate_registry": list(candidates),
        "reference_values": references,
        "prediction_rows": rows,
    }


def _score_xcop(
    scorer_config: Mapping[str, Any],
    source_config: dict[str, Any],
    references: dict[str, float],
    candidates: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
    item59_config: Mapping[str, Any],
) -> dict[str, Any]:
    score_rows = []
    blocked = []
    for scenario in scorer_config["xcop_scoring"]["nuisance_scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        nuisance = {key: float(value) for key, value in scenario.items() if key != "scenario_id"}
        for packet in packets:
            scaled = copy.deepcopy(dict(packet))
            scaled["ne_cm3"] = np.asarray(packet["ne_cm3"], dtype=float) * nuisance["density_scale"]
            state = sealed_scorer.cluster_suite._state(scaled, nuisance, item59_config)
            source_rows = sealed_scorer._xcop_source_rows(source_config, state, item59_config)
            for candidate in candidates:
                outputs = [
                    predict(source_config, candidate, row, references) for row in source_rows
                ]
                dispositions = {row[0] for row in outputs}
                candidate_id = str(candidate["candidate_id"])
                if dispositions != {"COMPILED"}:
                    _require(
                        dispositions == {"SOURCE_BLOCKED_DRIVER_UNAVAILABLE"},
                        "unexpected X-COP disposition",
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
                predictions = sealed_scorer.cluster_suite._predictions_from_acceleration(
                    scaled, state, acceleration, nuisance, item59_config
                )
                score = sealed_scorer._loss_rows(
                    predictions,
                    scaled["rows"],
                    minimum_fractional_error=float(
                        scorer_config["xcop_scoring"]["minimum_fractional_error"]
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
    controls = sorted(scorer_config["adjudication"]["published_control_ids"])
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_key[(str(row["candidate_id"]), str(row["scenario_id"]))].append(row)
    scenario_ids = [
        str(row["scenario_id"]) for row in scorer_config["xcop_scoring"]["nuisance_scenarios"]
    ]
    control_domain = {}
    control_object = {}
    for scenario_id in scenario_ids:
        control_domain[scenario_id] = min(
            float(np.mean([row["loss"] for row in by_key[(control, scenario_id)]]))
            for control in controls
        )
        for packet in packets:
            object_id = str(packet["cluster"])
            control_object[(scenario_id, object_id)] = min(
                float(
                    next(
                        row["loss"]
                        for row in by_key[(control, scenario_id)]
                        if row["object_id"] == object_id
                    )
                )
                for control in controls
            )
    nominal_id = "XCOP-SOURCE-NOMINAL"
    summaries = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if (candidate_id, nominal_id) not in by_key:
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "kind": candidate["kind"],
                    "disposition": "SOURCE_BLOCKED_NONSPHERICAL_GEOMETRY",
                }
            )
            continue
        scenarios = []
        for scenario_id in scenario_ids:
            object_rows = sorted(
                by_key[(candidate_id, scenario_id)], key=lambda row: row["object_id"]
            )
            aggregate = sealed_scorer._aggregate(object_rows)
            aggregate["scenario_id"] = scenario_id
            aggregate["strongest_control_loss"] = control_domain[scenario_id]
            aggregate["fractional_improvement"] = sealed_scorer._fractional_improvement(
                aggregate["loss"], control_domain[scenario_id]
            )
            aggregate["object_support"] = sum(
                float(row["loss"]) < control_object[(scenario_id, str(row["object_id"]))]
                for row in object_rows
            )
            scenarios.append(aggregate)
        nominal = next(row for row in scenarios if row["scenario_id"] == nominal_id)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "kind": candidate["kind"],
                "disposition": "SCORED",
                "nominal_loss": nominal["loss"],
                "nominal_strongest_control_loss": nominal["strongest_control_loss"],
                "nominal_fractional_improvement": nominal["fractional_improvement"],
                "nominal_object_support": nominal["object_support"],
                "median_nuisance_loss": float(np.median([row["loss"] for row in scenarios])),
                "worst_nuisance_loss": max(row["loss"] for row in scenarios),
                "worst_paired_fractional_improvement": min(
                    row["fractional_improvement"] for row in scenarios
                ),
                "scenario_results": scenarios,
            }
        )
    return {
        "score_row_count": len(score_rows),
        "blocked_row_count": len(blocked),
        "score_rows": score_rows,
        "blocked_rows": blocked,
        "candidate_summaries": summaries,
    }


def _adjudicate(
    config: Mapping[str, Any], galaxy: Mapping[str, Any], xcop: Mapping[str, Any]
) -> dict[str, Any]:
    galaxies = {row["candidate_id"]: row for row in galaxy["candidate_summaries"]}
    clusters = {row["candidate_id"]: row for row in xcop["candidate_summaries"]}
    gate = config["scoring_and_adjudication"]
    threshold = float(gate["minimum_meaningful_fractional_improvement"])
    rows = []
    for candidate_id, galaxy_row in galaxies.items():
        cluster_row = clusters[candidate_id]
        is_new = str(galaxy_row["kind"]).startswith("NEW_")
        phangs = galaxy_row["PHANGS_CO"]
        phangs_improvement = float(phangs["primary_fractional_improvement"])
        phangs_support = int(phangs["primary_object_support"])
        if cluster_row["disposition"] == "SCORED":
            cluster_improvement = float(cluster_row["nominal_fractional_improvement"])
            cluster_support = int(cluster_row["nominal_object_support"])
            combined = 0.5 * (phangs_improvement + cluster_improvement)
            signal = (
                is_new
                and phangs_improvement > threshold
                and cluster_improvement > threshold
                and phangs_support >= int(gate["phangs_minimum_object_support"])
                and cluster_support >= int(gate["xcop_minimum_object_support"])
            )
        else:
            cluster_improvement = None
            cluster_support = 0
            combined = None
            signal = False
        rows.append(
            {
                "candidate_id": candidate_id,
                "kind": galaxy_row["kind"],
                "PHANGS_primary_loss": float(phangs["primary_loss"]),
                "PHANGS_primary_fractional_improvement": phangs_improvement,
                "PHANGS_primary_object_support": phangs_support,
                "SPARC_primary_loss": float(galaxy_row["SPARC"]["primary_loss"]),
                "SPARC_primary_fractional_improvement": float(
                    galaxy_row["SPARC"]["primary_fractional_improvement"]
                ),
                "XCOP_nominal_loss": cluster_row.get("nominal_loss"),
                "XCOP_nominal_fractional_improvement": cluster_improvement,
                "XCOP_nominal_object_support": cluster_support,
                "combined_cross_scale_improvement": combined,
                "post_response_development_repair_signal": signal,
                "target_blind": False if is_new else None,
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
        "signal_ids": [
            row["candidate_id"] for row in rows if row["post_response_development_repair_signal"]
        ],
        "all_candidates_retained": len(rows),
        "charged_post_response_candidates": sum(
            str(row["kind"]).startswith("NEW_") for row in rows
        ),
        "target_blind": False,
        "independent_confirmation_required": True,
        "theory_health_separate": True,
        "maximum_claim": gate["maximum_claim"],
    }


def _benchmark_placements(
    config: Mapping[str, Any], source_config: Mapping[str, Any]
) -> dict[str, Any]:
    a0 = float(source_config["constants"]["a0_m_s2"])
    placements = [str(row["placement_id"]) for row in config["candidate_program"]["placements"]]
    test_accelerations = (a0 / 100.0, a0, a0 * 100.0)
    exact = all(
        placed_rar(g_b, a0, 1.0, placement) == source_compiler.rar_2016(g_b, a0)
        for placement in placements
        for g_b in test_accelerations
    )
    solar_g = float(source_config["constants"]["solar_benchmark_g_m_s2"])
    deviations = [
        abs(placed_rar(solar_g, a0, factor, placement) / solar_g - 1.0)
        for placement in placements
        for factor in (1.0 / 256.0, 1.0 / 16.0, 1.0, 16.0, 256.0)
    ]
    maximum = max(deviations)
    _require(exact, "F=1 did not recover the published RAR")
    _require(
        maximum
        < float(
            config["benchmarks_and_admission"][
                "high_acceleration_solar_recovery_max_fractional_deviation"
            ]
        ),
        "high-acceleration solar recovery failed",
    )
    return {
        "RAR_at_F_equal_1_exact": exact,
        "test_acceleration_count": len(test_accelerations),
        "factor_probe_count": 5,
        "maximum_solar_fractional_deviation": maximum,
        "paper_anchor": config["published_method_anchor"],
    }


def build_score_ledger(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    source_config, scorer_config, source_ledger = _load_predecessors(config)
    candidates = candidate_registry(config, source_ledger)
    benchmarks = _benchmark_placements(config, source_config)
    galaxy_predictions = _galaxy_prediction_ledger(source_config, source_ledger, candidates)
    phangs, phangs_access = sealed_scorer._load_phangs_responses(scorer_config)
    sparc, sparc_access = sealed_scorer._load_sparc_responses(scorer_config)
    packets, item59_config, xcop_access = sealed_scorer._load_xcop_responses(scorer_config)
    galaxy_scores = sealed_scorer._score_galaxies(scorer_config, galaxy_predictions, phangs, sparc)
    xcop_scores = _score_xcop(
        scorer_config,
        source_config,
        galaxy_predictions["reference_values"],
        candidates,
        packets,
        item59_config,
    )
    adjudication = _adjudicate(config, galaxy_scores, xcop_scores)
    ledger: dict[str, Any] = {
        "schema": _LEDGER_SCHEMA,
        "package_id": config["package_id"],
        "status": "POST_RESPONSE_DEVELOPMENT_REPAIR_EXPLORATION_SCORED_NO_TUNING",
        "chronology_and_honesty": config["chronology_and_honesty"],
        "source_prediction_binding": config["source_prediction_binding"],
        "scorer_binding": config["scorer_binding"],
        "published_method_anchor": config["published_method_anchor"],
        "candidate_registry": candidates,
        "candidate_registry_sha256": content_sha256(candidates),
        "benchmarks": benchmarks,
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
            "post_response_development_repair_signal_only": True,
            "target_blind": False,
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
    signals = ledger["adjudication"]["signal_ids"]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": ledger["status"],
        "decision": (
            "POST_RESPONSE_DEVELOPMENT_REPAIR_SIGNALS_FOUND_CONFIRMATION_REQUIRED"
            if signals
            else "NO_POST_RESPONSE_CROSS_SCALE_REPAIR_SIGNAL_ALL_CANDIDATES_RETAINED"
        ),
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "chronology_and_honesty": ledger["chronology_and_honesty"],
        "source_prediction_binding": config["source_prediction_binding"],
        "scorer_binding": config["scorer_binding"],
        "published_method_anchor": config["published_method_anchor"],
        "candidate_count": len(ranked),
        "new_candidate_count": ledger["adjudication"]["charged_post_response_candidates"],
        "candidate_registry_sha256": ledger["candidate_registry_sha256"],
        "benchmarks": ledger["benchmarks"],
        "access": ledger["access"],
        "signal_ids": signals,
        "top_ranked_candidates": ranked[:15],
        "galaxy_source_cells": ledger["galaxy_scores"]["source_cell_count"],
        "galaxy_score_rows": ledger["galaxy_scores"]["score_row_count"],
        "xcop_score_rows": ledger["xcop_scores"]["score_row_count"],
        "xcop_blocked_rows": ledger["xcop_scores"]["blocked_row_count"],
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
                    "signals": len(receipt["signal_ids"]),
                    "top": receipt["top_ranked_candidates"][:5],
                },
                sort_keys=True,
            )
        )
    else:
        print("UNWRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
