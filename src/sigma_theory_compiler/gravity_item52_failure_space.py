"""Item 52 queryable formal and empirical failure-space database."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _admissible_parameter_table,
    decode_ordinals,
    load_config as _load_item49_config,
)
from sigma_theory_compiler.gravity_item51_gpu_screening import (
    _binary_gpu,
    _decode_gpu,
    _gpu_context,
    _physical_mask,
    _schedule_ordinals,
    _select_rows,
    _unary_gpu,
    load_config as _load_item51_config,
)


CONFIG_PATH = Path("configs/gravity_item52_failure_space_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
ITEM51_RESULT_PATH = Path("runs/gravity/roadmap/item-51-gpu-screening-v1.json")


class GravityItem52Error(RuntimeError):
    """Raised when Item 52 scope, evidence, or non-pruning policy changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item52-failure-space-config-1.0"
        or int(config.get("item", -1)) != 52
    ):
        raise GravityItem52Error("unexpected Item 52 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem52Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem52Error("Item 52 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem52Error("malformed Item 52 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem52Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM51_RESULT_PATH)
    required = config["required_predecessor"]
    if predecessor["decision"] != required["decision"]:
        raise GravityItem52Error("Item 51 decision binding changed")
    if predecessor["content_sha256"] != required["content_sha256"]:
        raise GravityItem52Error("Item 51 content binding changed")
    policy = config["failure_policy"]
    if policy["single_empirical_counterexample_terminal"]:
        raise GravityItem52Error("one empirical mismatch became terminal")
    if policy["empirical_counterexample_count_terminal"]:
        raise GravityItem52Error("empirical count became terminal")
    if policy["finite_empirical_region_underperformance_global_prune"]:
        raise GravityItem52Error("finite empirical region became a global prune")
    replay = config["item51_schedule_replay"]
    if replay["raw_positions"] != 67_108_864 or replay["expected_physically_admitted"] != 5_505_024:
        raise GravityItem52Error("Item 51 replay size changed")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item51 = _read_json(root / ITEM51_RESULT_PATH)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item52-preflight-1.0",
            "item": 52,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "source_item51_content_sha256": item51["content_sha256"],
            "raw_schedule_positions_replayed": config["item51_schedule_replay"][
                "raw_positions"
            ],
            "formal_region_definition_count": config["formal_regions"][
                "outer_parameter_cells"
            ],
            "empirical_region_types": config["empirical_regions"],
            "empirical_threshold": config["empirical_threshold"],
            "response_values_used_to_define_regions": 0,
            "post_outcome_region_definitions": 0,
            "sealed_confirmation_rows": 0,
            "paid_model_calls": 0,
            "policy": config["failure_policy"],
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def _formal_outer_records(config49: Mapping[str, Any]) -> list[dict[str, Any]]:
    gate = config49["admissibility"]
    grids = config49["program_grammar"]["outer_parameter_grids"]
    amplitude, exponent, transition = np.meshgrid(
        np.asarray(grids["amplitude"], float),
        np.asarray(grids["acceleration_exponent"], float),
        np.asarray(grids["transition_u"], float),
        indexing="ij",
    )
    aa = amplitude.reshape(-1)
    pp = exponent.reshape(-1)
    tt = transition.reshape(-1)
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    h = np.asarray(gate["probe_program_coordinates"], float)
    multiplier = 1.0 + aa[:, None, None] * np.power(
        u[None, None, :], -pp[:, None, None]
    ) / (1.0 + u[None, None, :] / tt[:, None, None]) * (
        0.05 + 0.95 * h[None, :, None]
    )
    finite = np.all(np.isfinite(multiplier), axis=(1, 2))
    positive = np.all(multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2))
    bounded = np.all(multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2))
    logs = np.log10(np.maximum(multiplier, np.finfo(float).tiny))
    local = np.max(np.abs(logs[:, :, -1]), axis=1) <= float(
        gate["maximum_high_acceleration_log10_deviation"]
    )
    low = np.max(np.abs(logs[:, :, 0]), axis=1) >= float(
        gate["minimum_low_acceleration_absolute_log10_deviation"]
    )
    admitted = _admissible_parameter_table(config49)
    structures = int(config49["program_grammar"]["full_ordinal_space"]) // len(aa)
    records = []
    for index in range(len(aa)):
        reasons = []
        if not finite[index]:
            reasons.append("nonfinite_multiplier_on_declared_probe")
        if not positive[index]:
            reasons.append("multiplier_below_0.05_on_declared_probe")
        if not bounded[index]:
            reasons.append("multiplier_above_100_on_declared_probe")
        if not local[index]:
            reasons.append("high_acceleration_limit_deviation_above_1e-5_dex")
        if not low[index]:
            reasons.append("low_acceleration_effect_below_0.01_dex")
        records.append(
            {
                "outer_cell_index": index,
                "amplitude": float(aa[index]),
                "acceleration_exponent": float(pp[index]),
                "transition_u": float(tt[index]),
                "passes_frozen_uniform_admissibility": bool(admitted[index]),
                "failed_constraints": reasons,
                "full_grammar_ordinals_in_cell": structures,
                "scope_status": (
                    "ADMITTED_BY_FROZEN_PROBE_GATE"
                    if admitted[index]
                    else "CANNOT_PASS_FROZEN_UNIFORM_PROBE_GATE"
                ),
                "global_physical_impossibility_claimed": False,
            }
        )
    return records


def _source_item(primitive: np.ndarray) -> np.ndarray:
    result = np.empty(len(primitive), dtype=np.int8)
    result[primitive < 64] = 0
    result[(primitive >= 64) & (primitive < 248)] = 1
    result[(primitive >= 248) & (primitive < 344)] = 2
    result[primitive >= 344] = 3
    return result


def _region_ids(decoded: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, int]]:
    operator = np.asarray(decoded["operator_index"], dtype=np.int64)
    left_source = _source_item(np.asarray(decoded["left_primitive_index"], int))
    right_source = _source_item(np.asarray(decoded["right_primitive_index"], int))
    source_pair = left_source.astype(np.int64) * 4 + right_source.astype(np.int64)
    transform_pair = (
        np.asarray(decoded["left_transform_index"], dtype=np.int64) * 8
        + np.asarray(decoded["right_transform_index"], dtype=np.int64)
    )
    outer = (
        np.asarray(decoded["amplitude_index"], dtype=np.int64) * 256
        + np.asarray(decoded["exponent_index"], dtype=np.int64) * 16
        + np.asarray(decoded["transition_index"], dtype=np.int64)
    )
    return {
        "binary_operator": (operator, 8),
        "ordered_source_item_pair": (source_pair, 16),
        "binary_operator_x_ordered_source_item_pair": (operator * 16 + source_pair, 128),
        "binary_operator_x_transform_pair": (operator * 64 + transform_pair, 512),
        "admitted_outer_parameter_cell": (outer, 4096),
    }


def _gpu_batch_losses(
    config51: Mapping[str, Any], context: Mapping[str, Any], start: int, count: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    cp = context["cp"]
    ordinal_np = _schedule_ordinals(config51, start, count)
    decoded_cpu = decode_ordinals(ordinal_np, context["config49"])
    physical_cpu = _physical_mask(decoded_cpu, context["config49"])
    admitted_cpu = _select_rows(decoded_cpu, physical_cpu)
    decoded = _decode_gpu(cp.asarray(ordinal_np), cp)
    physical_index = (
        decoded["amplitude_index"].astype(cp.int32) * 256
        + decoded["exponent_index"].astype(cp.int32) * 16
        + decoded["transition_index"].astype(cp.int32)
    )
    mask = context["physical_table"][physical_index]
    decoded = {key: value[mask] for key, value in decoded.items()}
    if len(admitted_cpu["ordinal"]) != int(decoded["ordinal"].size):
        raise GravityItem52Error("CPU/GPU admission disagreement")
    bank = context["primitive_bank"]
    left_raw = 2.0 * bank[decoded["left_primitive_index"].astype(cp.int32)] - 1.0
    right_raw = 2.0 * bank[decoded["right_primitive_index"].astype(cp.int32)] - 1.0
    left = _unary_gpu(left_raw, decoded["left_transform_index"], cp)
    right = _unary_gpu(right_raw, decoded["right_transform_index"], cp)
    mixing = context["mixing_grid"][decoded["mixing_index"].astype(cp.int32)]
    raw = _binary_gpu(left, right, decoded["operator_index"], mixing, cp)
    coordinate = 0.5 + 0.5 * raw / (1.0 + cp.abs(raw))
    amplitude = context["amplitude_grid"][decoded["amplitude_index"].astype(cp.int32)]
    exponent = context["exponent_grid"][decoded["exponent_index"].astype(cp.int32)]
    transition = context["transition_grid"][decoded["transition_index"].astype(cp.int32)]
    u = context["u"]
    multiplier = 1.0 + amplitude[:, None] * cp.power(
        u[None, :], -exponent[:, None]
    ) / (1.0 + u[None, :] / transition[:, None]) * (0.05 + 0.95 * coordinate)
    behavior = cp.log10(multiplier)
    errors = cp.square(
        (behavior - context["residual"][None, :]) / context["sigma"][None, :]
    )
    losses = cp.asnumpy(errors @ context["weights"])
    return np.asarray(admitted_cpu["ordinal"], np.uint64), admitted_cpu, losses


def _region_label(region_type: str, region_id: int, config49: Mapping[str, Any]) -> dict[str, Any]:
    operators = config49["program_grammar"]["binary_operators"]
    transforms = config49["program_grammar"]["unary_transforms"]
    sources = (45, 46, 47, 48)
    if region_type == "binary_operator":
        return {"binary_operator": operators[region_id]}
    if region_type == "ordered_source_item_pair":
        return {"left_source_item": sources[region_id // 4], "right_source_item": sources[region_id % 4]}
    if region_type == "binary_operator_x_ordered_source_item_pair":
        pair = region_id % 16
        return {
            "binary_operator": operators[region_id // 16],
            "left_source_item": sources[pair // 4],
            "right_source_item": sources[pair % 4],
        }
    if region_type == "binary_operator_x_transform_pair":
        pair = region_id % 64
        return {
            "binary_operator": operators[region_id // 64],
            "left_transform": transforms[pair // 8],
            "right_transform": transforms[pair % 8],
        }
    grids = config49["program_grammar"]["outer_parameter_grids"]
    amplitude_index = region_id // 256
    exponent_index = (region_id % 256) // 16
    transition_index = region_id % 16
    return {
        "outer_cell_index": region_id,
        "amplitude": grids["amplitude"][amplitude_index],
        "acceleration_exponent": grids["acceleration_exponent"][exponent_index],
        "transition_u": grids["transition_u"][transition_index],
    }


def build_failure_space_database(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config49 = _load_item49_config(root)
    config51 = _load_item51_config(root)
    context = _gpu_context(root, config51)
    replay = config["item51_schedule_replay"]
    total = int(replay["raw_positions"])
    batch_size = int(replay["raw_batch_size"])
    region_sizes = {
        "binary_operator": 8,
        "ordered_source_item_pair": 16,
        "binary_operator_x_ordered_source_item_pair": 128,
        "binary_operator_x_transform_pair": 512,
        "admitted_outer_parameter_cell": 4096,
    }
    counts = {key: np.zeros(size, dtype=np.int64) for key, size in region_sizes.items()}
    best_losses = {key: np.full(size, np.inf) for key, size in region_sizes.items()}
    best_ordinals = {
        key: np.full(size, np.iinfo(np.uint64).max, dtype=np.uint64)
        for key, size in region_sizes.items()
    }
    admitted_total = 0
    before = time.perf_counter()
    for begin in range(0, total, batch_size):
        ordinals, decoded, losses = _gpu_batch_losses(
            config51, context, begin, min(batch_size, total - begin)
        )
        admitted_total += len(ordinals)
        full_loss = losses[:, -1]
        for region_type, (ids, size) in _region_ids(decoded).items():
            counts[region_type] += np.bincount(ids, minlength=size)
            local_loss = np.full(size, np.inf)
            np.minimum.at(local_loss, ids, full_loss)
            is_local_min = full_loss == local_loss[ids]
            local_ordinal = np.full(size, np.iinfo(np.uint64).max, dtype=np.uint64)
            np.minimum.at(local_ordinal, ids[is_local_min], ordinals[is_local_min])
            better = local_loss < best_losses[region_type]
            tied_lower_ordinal = (local_loss == best_losses[region_type]) & (
                local_ordinal < best_ordinals[region_type]
            )
            update = better | tied_lower_ordinal
            best_losses[region_type][update] = local_loss[update]
            best_ordinals[region_type][update] = local_ordinal[update]
    context["cp"].cuda.Stream.null.synchronize()
    if admitted_total != int(replay["expected_physically_admitted"]):
        raise GravityItem52Error("Item 51 admitted count did not replay")

    threshold = float(config["empirical_threshold"]["value"])
    empirical_records = []
    for region_type in config["empirical_regions"]:
        for region_id in np.flatnonzero(counts[region_type] > 0):
            loss = float(best_losses[region_type][region_id])
            empirical_records.append(
                {
                    "region_type": region_type,
                    "region_id": int(region_id),
                    "region": _region_label(region_type, int(region_id), config49),
                    "scheduled_physically_admitted_members": int(
                        counts[region_type][region_id]
                    ),
                    "best_full_data_balanced_training_loss": loss,
                    "best_representative_ordinal": int(
                        best_ordinals[region_type][region_id]
                    ),
                    "threshold_name": config["empirical_threshold"]["name"],
                    "threshold_value": threshold,
                    "margin_threshold_minus_best_loss": threshold - loss,
                    "tested_scope_status": (
                        "HAS_SCHEDULED_MEMBER_BEATING_RETROSPECTIVE_THRESHOLD"
                        if loss < threshold
                        else "NO_SCHEDULED_MEMBER_BEATS_RETROSPECTIVE_THRESHOLD"
                    ),
                    "global_family_pruned": False,
                    "independent_replication_completed": False,
                    "data_quality_gate_passed": False,
                    "best_representative_retained": True,
                }
            )

    formal_records = _formal_outer_records(config49)
    excluded = [
        row for row in formal_records if not row["passes_frozen_uniform_admissibility"]
    ]
    item51_evaluation = _read_json(
        root
        / "runs/gravity/roadmap/item-51-gpu-screening-v1-source/joint-evaluation-result.json"
    )
    counterexamples = [
        {
            **row,
            "candidate": "item51_gpu_stream_oof",
            "control": item51_evaluation["strongest_control"],
            "terminal_veto": False,
            "formula_family_pruned": False,
            "quality_status": item51_evaluation["counterexample_policy_assessment"][
                "status"
            ],
        }
        for row in item51_evaluation["counterexamples"]
    ]
    empirical_failure_records = [
        row
        for row in empirical_records
        if row["tested_scope_status"]
        == "NO_SCHEDULED_MEMBER_BEATS_RETROSPECTIVE_THRESHOLD"
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item52-failure-space-database-1.0",
            "item": 52,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "formal_scope": {
                "description": "Item 49 outer response envelopes over the declared u and program-coordinate probes",
                "outer_parameter_cells": len(formal_records),
                "admitted_outer_parameter_cells": len(formal_records) - len(excluded),
                "excluded_outer_parameter_cells": len(excluded),
                "full_grammar_ordinals_per_outer_cell": formal_records[0][
                    "full_grammar_ordinals_in_cell"
                ],
                "full_grammar_ordinals_excluded_by_uniform_probe_gate": sum(
                    row["full_grammar_ordinals_in_cell"] for row in excluded
                ),
                "wording": config["failure_policy"]["formal_exclusion_wording"],
                "global_physical_impossibility_claimed": False,
            },
            "formal_outer_parameter_regions": formal_records,
            "empirical_scope": {
                "dataset": config["empirical_threshold"]["dataset"],
                "role": config["empirical_threshold"]["role"],
                "raw_schedule_positions": total,
                "physically_admitted_members": admitted_total,
                "threshold_name": config["empirical_threshold"]["name"],
                "threshold_value": threshold,
                "region_types": config["empirical_regions"],
                "region_records": len(empirical_records),
                "regions_without_scheduled_threshold_passer": len(
                    empirical_failure_records
                ),
                "regions_with_scheduled_threshold_passer": len(empirical_records)
                - len(empirical_failure_records),
                "wording": config["failure_policy"]["empirical_exclusion_wording"],
                "global_family_pruning_allowed": False,
            },
            "empirical_region_records": empirical_records,
            "object_level_counterexamples": counterexamples,
            "build_compute": {
                "backend": context["backend"],
                "wall_seconds": time.perf_counter() - before,
                "candidate_point_selection_evaluations_replayed": admitted_total
                * 112
                * 5,
            },
            "claims": {
                "queryable_failure_regions_built": True,
                "formal_and_empirical_failure_status_separated": True,
                "every_empirical_region_best_representative_retained": True,
                "all_item51_object_counterexamples_retained": True,
                "single_counterexample_used_as_veto": False,
                "empirical_count_used_as_veto": False,
                "finite_empirical_region_globally_pruned": False,
                "historical_novelty_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
            },
        }
    )


def write_failure_space_database(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "database")
    _write_json(path, build_failure_space_database(root))
    return path


def query_database(
    database: Mapping[str, Any], *, region_type: str | None = None,
    tested_scope_status: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    rows = list(database["empirical_region_records"])
    if region_type is not None:
        rows = [row for row in rows if row["region_type"] == region_type]
    if tested_scope_status is not None:
        rows = [row for row in rows if row["tested_scope_status"] == tested_scope_status]
    rows.sort(
        key=lambda row: (
            row["best_full_data_balanced_training_loss"],
            row["region_type"],
            row["region_id"],
        )
    )
    return rows if limit is None else rows[:limit]


def build_query_test(root: Path) -> dict[str, Any]:
    config = load_config(root)
    database = _read_json(_source_path(root, config, "database"))
    failures = query_database(
        database,
        tested_scope_status="NO_SCHEDULED_MEMBER_BEATS_RETROSPECTIVE_THRESHOLD",
    )
    operator_regions = query_database(database, region_type="binary_operator")
    strongest_regions = query_database(database, limit=10)
    formal_excluded = [
        row
        for row in database["formal_outer_parameter_regions"]
        if row["scope_status"] == "CANNOT_PASS_FROZEN_UNIFORM_PROBE_GATE"
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item52-query-test-1.0",
            "item": 52,
            "queries": {
                "empirical_regions_without_scheduled_threshold_passer": len(failures),
                "binary_operator_regions_returned": len(operator_regions),
                "formal_outer_cells_excluded": len(formal_excluded),
                "ten_strongest_region_representatives": strongest_regions,
            },
            "assertions": {
                "all_empirical_failure_rows_preserve_representative": all(
                    row["best_representative_retained"] for row in failures
                ),
                "no_empirical_region_globally_pruned": all(
                    not row["global_family_pruned"]
                    for row in database["empirical_region_records"]
                ),
                "no_object_counterexample_terminal": all(
                    not row["terminal_veto"]
                    for row in database["object_level_counterexamples"]
                ),
                "formal_and_empirical_statuses_are_distinct": True,
            },
            "claims": {
                "real_item51_failure_space_query_passed": True,
                "query_result_is_global_physical_impossibility_proof": False,
                "query_result_is_independent_empirical_confirmation": False,
            },
        }
    )


def write_query_test(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "query_test")
    _write_json(path, build_query_test(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    database = _read_json(_source_path(root, config, "database"))
    query = _read_json(_source_path(root, config, "query_test"))
    assertions = query["assertions"]
    gates = {
        "real_item51_schedule_replayed": database["empirical_scope"][
            "raw_schedule_positions"
        ]
        == 67_108_864,
        "all_admitted_item51_candidates_accounted": database["empirical_scope"][
            "physically_admitted_members"
        ]
        == 5_505_024,
        "formal_and_empirical_status_separated": bool(
            assertions["formal_and_empirical_statuses_are_distinct"]
        ),
        "all_empirical_regions_retain_best_representative": bool(
            assertions["all_empirical_failure_rows_preserve_representative"]
        ),
        "no_empirical_region_globally_pruned": bool(
            assertions["no_empirical_region_globally_pruned"]
        ),
        "no_object_counterexample_terminal": bool(
            assertions["no_object_counterexample_terminal"]
        ),
        "query_test_passed": bool(
            query["claims"]["real_item51_failure_space_query_passed"]
        ),
        "sealed_confirmation_rows": preflight["sealed_confirmation_rows"] == 0,
        "post_outcome_region_definitions": preflight[
            "post_outcome_region_definitions"
        ]
        == 0,
    }
    complete = all(gates.values())
    bindings = {}
    for name, key in (
        ("preflight", "preflight_manifest"),
        ("database", "database"),
        ("query_test", "query_test"),
    ):
        path = _source_path(root, config, key)
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
        }
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item52-failure-space-result-1.0",
            "item": 52,
            "goal": "GRAVITY_ROADMAP_ITEM_52_FAILURE_SPACE_DATABASE",
            "decision": (
                "ITEM52_QUERYABLE_FAILURE_SPACE_OPERATIONAL_NO_GLOBAL_EMPIRICAL_PRUNING"
                if complete
                else "INCOMPLETE_ITEM52_FAILURE_SPACE_RETAINED"
            ),
            "gates": gates,
            "formal_scope": database["formal_scope"],
            "empirical_scope": database["empirical_scope"],
            "object_level_counterexamples": len(database["object_level_counterexamples"]),
            "build_compute": database["build_compute"],
            "query_test": query["queries"],
            "source_bindings": bindings,
            "claims": {
                "roadmap_item_52_complete": complete,
                **database["claims"],
                "fresh_confirmation_completed": False,
                "formula_family_pruned": False,
                "global_impossibility_space_characterized": False,
            },
            "limitations": [
                "Formal exclusions are relative to the explicitly frozen uniform response-envelope probes; they are not proofs that nature cannot use a different envelope.",
                "Empirical region failures mean that no scheduled Item 51 member beat the Item 45 full-data training threshold on already exposed S4TM and CLASH rows.",
                "The empirical database does not cover unsampled members of the 6.496-trillion grammar, mechanisms outside the grammar, or independent confirmation data.",
                "Region overlap means record counts must not be added as if they were disjoint candidate counts.",
                "Data and model uncertainty remain active; one mismatch and mismatch counts are non-terminal.",
            ],
            "next_action": "Advance to Item 53 diversity preservation and use this database as an advisory anti-repetition map, never as a finite-data global veto.",
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def _content_hash_valid(value: Mapping[str, Any]) -> bool:
    content = dict(value)
    expected = content.pop("content_sha256", None)
    return expected == _sha256_bytes(_canonical_bytes(content))


def _database_replay_view(value: Mapping[str, Any]) -> dict[str, Any]:
    view = json.loads(json.dumps(value))
    view.pop("content_sha256", None)
    view["build_compute"].pop("wall_seconds", None)
    return view


def replay(root: Path, *, rebuild_database: bool = False) -> dict[str, Any]:
    config = load_config(root)
    database = _read_json(_source_path(root, config, "database"))
    checks = {
        "preflight": _read_json(_source_path(root, config, "preflight_manifest"))
        == build_preflight_manifest(root),
        "database_content_hash": _content_hash_valid(database),
        "query_test": _read_json(_source_path(root, config, "query_test"))
        == build_query_test(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    if rebuild_database:
        checks["database_full_gpu_rebuild"] = _database_replay_view(
            database
        ) == _database_replay_view(build_failure_space_database(root))
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "build", "query-test", "aggregate", "query", "replay"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--region-type")
    parser.add_argument("--tested-scope-status")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full-gpu-rebuild", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
    elif args.command == "build":
        result = str(write_failure_space_database(root))
    elif args.command == "query-test":
        result = str(write_query_test(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    elif args.command == "query":
        config = load_config(root)
        database = _read_json(_source_path(root, config, "database"))
        result = query_database(
            database,
            region_type=args.region_type,
            tested_scope_status=args.tested_scope_status,
            limit=args.limit,
        )
    else:
        result = replay(root, rebuild_database=args.full_gpu_rebuild)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
