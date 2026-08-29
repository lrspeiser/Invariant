"""Item 53 mechanism-niche archive versus score-only selection."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item45_universal_interactions import (
    _object_weights,
    _paired_p,
    _score,
)
from sigma_theory_compiler.gravity_item48_action_generator import (
    _evaluation_arrays as _item48_evaluation_arrays,
    load_config as _load_item48_config,
)
from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    _primitive_bank_from_arrays,
    _primitive_sources,
    _program_description,
    decode_ordinals,
    load_config as _load_item49_config,
    primitive_labels,
    program_log_multiplier,
)


CONFIG_PATH = Path("configs/gravity_item53_diversity_preservation_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM52_RESULT_PATH = Path("runs/gravity/roadmap/item-52-failure-space-v1.json")
ITEM52_DATABASE_PATH = Path(
    "runs/gravity/roadmap/item-52-failure-space-v1-source/failure-space-database.json"
)


class GravityItem53Error(RuntimeError):
    """Raised when the frozen archive or non-pruning contract changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item53-diversity-preservation-config-1.0"
        or int(config.get("item", -1)) != 53
    ):
        raise GravityItem53Error("unexpected Item 53 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem53Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem53Error("Item 53 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem53Error("malformed Item 53 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem53Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM52_RESULT_PATH)
    required = config["required_predecessor"]
    if predecessor["decision"] != required["decision"]:
        raise GravityItem53Error("Item 52 decision binding changed")
    if predecessor["content_sha256"] != required["content_sha256"]:
        raise GravityItem53Error("Item 52 content binding changed")
    if config["archives"]["slots"] != 64:
        raise GravityItem53Error("archive size changed")
    if not config["archives"]["selection_never_deletes_source_database_records"]:
        raise GravityItem53Error("archive selection became destructive")
    policy = config["policy"]
    if policy["single_counterexample_terminal"]:
        raise GravityItem53Error("one mismatch became terminal")
    if policy["finite_empirical_failure_prunes_niche"]:
        raise GravityItem53Error("finite empirical failure prunes a niche")
    if policy["low_score_alone_deletes_candidate"]:
        raise GravityItem53Error("score-only deletion entered diversity preservation")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _source_item(primitive: np.ndarray) -> np.ndarray:
    result = np.empty(len(primitive), dtype=np.int8)
    result[primitive < 64] = 0
    result[(primitive >= 64) & (primitive < 248)] = 1
    result[(primitive >= 248) & (primitive < 344)] = 2
    result[primitive >= 344] = 3
    return result


def _pool(root: Path) -> list[dict[str, Any]]:
    config = load_config(root)
    database = _read_json(root / ITEM52_DATABASE_PATH)
    records = database["empirical_region_records"]
    if len(records) != config["candidate_pool"]["expected_region_records"]:
        raise GravityItem53Error("Item 52 region count changed")
    by_ordinal: dict[int, dict[str, Any]] = {}
    memberships: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        ordinal = int(row["best_representative_ordinal"])
        memberships.setdefault(ordinal, []).append(
            {"region_type": row["region_type"], "region_id": row["region_id"]}
        )
        candidate = {
            "ordinal": ordinal,
            "item52_full_data_training_loss": float(
                row["best_full_data_balanced_training_loss"]
            ),
        }
        if ordinal not in by_ordinal or (
            candidate["item52_full_data_training_loss"], ordinal
        ) < (
            by_ordinal[ordinal]["item52_full_data_training_loss"], ordinal
        ):
            by_ordinal[ordinal] = candidate
    if len(by_ordinal) != config["candidate_pool"]["expected_unique_ordinals"]:
        raise GravityItem53Error("unique Item 52 representative count changed")
    config49 = _load_item49_config(root)
    ordinals = np.asarray(sorted(by_ordinal), dtype=np.uint64)
    decoded = decode_ordinals(ordinals, config49)
    left_source = _source_item(np.asarray(decoded["left_primitive_index"], int))
    right_source = _source_item(np.asarray(decoded["right_primitive_index"], int))
    pool = []
    for index, ordinal_value in enumerate(ordinals):
        ordinal = int(ordinal_value)
        row = dict(by_ordinal[ordinal])
        row.update(
            {
                "operator_index": int(decoded["operator_index"][index]),
                "left_source_index": int(left_source[index]),
                "right_source_index": int(right_source[index]),
                "source_pair_index": int(left_source[index] * 4 + right_source[index]),
                "operator_source_niche": int(
                    decoded["operator_index"][index] * 16
                    + left_source[index] * 4
                    + right_source[index]
                ),
                "left_transform_index": int(decoded["left_transform_index"][index]),
                "right_transform_index": int(decoded["right_transform_index"][index]),
                "transform_pair_index": int(
                    decoded["left_transform_index"][index] * 8
                    + decoded["right_transform_index"][index]
                ),
                "outer_cell_index": int(
                    decoded["amplitude_index"][index] * 256
                    + decoded["exponent_index"][index] * 16
                    + decoded["transition_index"][index]
                ),
                "left_primitive_index": int(decoded["left_primitive_index"][index]),
                "right_primitive_index": int(decoded["right_primitive_index"][index]),
                "item52_region_memberships": sorted(
                    memberships[ordinal], key=lambda value: (value["region_type"], value["region_id"])
                ),
            }
        )
        pool.append(row)
    return pool


def _ranked(pool: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in sorted(
            pool,
            key=lambda value: (
                value["item52_full_data_training_loss"],
                value["ordinal"],
            ),
        )
    ]


def _score_only_archive(pool: Sequence[Mapping[str, Any]], slots: int) -> list[dict[str, Any]]:
    return _ranked(pool)[:slots]


def _diversity_archive(pool: Sequence[Mapping[str, Any]], slots: int) -> list[dict[str, Any]]:
    ranked = _ranked(pool)
    selected: list[dict[str, Any]] = []
    selected_ordinals: set[int] = set()

    def add_best(field: str, values: Sequence[int]) -> None:
        for value in values:
            for row in ranked:
                if row[field] == value and row["ordinal"] not in selected_ordinals:
                    selected.append(row)
                    selected_ordinals.add(row["ordinal"])
                    break

    add_best("operator_index", range(8))
    add_best("source_pair_index", range(16))
    niche_champions = []
    for niche in range(128):
        champion = next(
            (row for row in ranked if row["operator_source_niche"] == niche), None
        )
        if champion is not None:
            niche_champions.append(champion)
    for row in _ranked(niche_champions):
        if len(selected) >= slots:
            break
        if row["ordinal"] not in selected_ordinals:
            selected.append(row)
            selected_ordinals.add(row["ordinal"])
    for row in ranked:
        if len(selected) >= slots:
            break
        if row["ordinal"] not in selected_ordinals:
            selected.append(row)
            selected_ordinals.add(row["ordinal"])
    if len(selected) != slots:
        raise GravityItem53Error("diversity archive did not fill its frozen slots")
    return selected


def _archive_diversity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "slots": len(rows),
        "distinct_ordinals": len({row["ordinal"] for row in rows}),
        "distinct_binary_operators": len({row["operator_index"] for row in rows}),
        "distinct_ordered_source_pairs": len({row["source_pair_index"] for row in rows}),
        "distinct_operator_source_niches": len(
            {row["operator_source_niche"] for row in rows}
        ),
        "distinct_transform_pairs": len({row["transform_pair_index"] for row in rows}),
        "distinct_outer_parameter_cells": len({row["outer_cell_index"] for row in rows}),
        "distinct_primitives": len(
            {row["left_primitive_index"] for row in rows}
            | {row["right_primitive_index"] for row in rows}
        ),
    }


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item53-preflight-1.0",
            "item": 53,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "candidate_pool_source": config["candidate_pool"]["source"],
            "candidate_pool_expected_unique_ordinals": config["candidate_pool"][
                "expected_unique_ordinals"
            ],
            "archive_slots_per_lane": config["archives"]["slots"],
            "archive_algorithms": config["archives"],
            "response_values_used_to_define_pool": config["candidate_pool"][
                "response_fields_used_to_define_pool"
            ],
            "post_outcome_archive_changes": 0,
            "sealed_confirmation_rows": 0,
            "paid_model_calls": 0,
            "policy": config["policy"],
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def build_archive_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    pool = _pool(root)
    slots = int(config["archives"]["slots"])
    archives = {
        "score_only": _score_only_archive(pool, slots),
        "diversity_preserving": _diversity_archive(pool, slots),
    }
    source = _read_json(root / ITEM52_DATABASE_PATH)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item53-archive-manifest-1.0",
            "item": 53,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "source_database_content_sha256": source["content_sha256"],
            "source_region_records": len(source["empirical_region_records"]),
            "source_unique_representative_ordinals": len(pool),
            "source_database_records_deleted": 0,
            "archives": archives,
            "diversity": {
                name: _archive_diversity(rows) for name, rows in archives.items()
            },
            "cross_archive_ordinal_overlap": len(
                {row["ordinal"] for row in archives["score_only"]}
                & {row["ordinal"] for row in archives["diversity_preserving"]}
            ),
            "response_values_used_to_change_frozen_archive_after_selection": 0,
            "claims": {
                "all_source_database_records_preserved": True,
                "archive_members_are_only_views_not_deletions": True,
                "historical_novelty_established": False,
                "formula_family_pruned": False,
            },
        }
    )


def write_archive_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "archive_manifest")
    _write_json(path, build_archive_manifest(root))
    return path


def _archive_behavior(
    root: Path, arrays: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    config49 = _load_item49_config(root)
    programs = decode_ordinals(
        np.asarray([row["ordinal"] for row in rows], dtype=np.uint64), config49
    )
    behavior = program_log_multiplier(
        programs,
        _primitive_bank_from_arrays(arrays),
        np.asarray(arrays["u"], float),
        config49,
    )
    return programs, behavior


def _best_row(
    behavior: np.ndarray, arrays: Mapping[str, Any], selected: np.ndarray
) -> tuple[int, float]:
    weights = _object_weights(arrays, selected)
    residual = arrays["target"] - arrays["base"]
    losses = np.sum(
        np.square((behavior - residual[None, :]) / arrays["sigma"][None, :])
        * weights[None, :],
        axis=1,
    )
    row = int(np.argmin(losses))
    return row, float(losses[row])


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    manifest = _read_json(_source_path(root, config, "archive_manifest"))
    arrays = _item48_evaluation_arrays(root, _load_item48_config(root))
    config49 = _load_item49_config(root)
    labels = primitive_labels(_primitive_sources(root))
    lane_names = ("score_only", "diversity_preserving")
    programs = {}
    behaviors = {}
    for lane in lane_names:
        programs[lane], behaviors[lane] = _archive_behavior(
            root, arrays, manifest["archives"][lane]
        )
    predictions = {
        lane: np.empty(len(arrays["target"]), dtype=float) for lane in lane_names
    }
    fold_ledger = []
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        selections = {}
        for lane in lane_names:
            row, loss = _best_row(behaviors[lane], arrays, train)
            predictions[lane][test] = arrays["base"][test] + behaviors[lane][row, test]
            selections[lane] = {
                "program": _program_description(
                    programs[lane], row, labels, config49
                ),
                "training_balanced_loss": loss,
            }
        fold_ledger.append(
            {
                "fold": fold,
                "score_only_selection": selections["score_only"],
                "diversity_preserving_selection": selections["diversity_preserving"],
                "heldout_s4tm_objects": sorted(
                    set(arrays["object"][test & (arrays["population"] == "S4TM")].tolist())
                ),
                "heldout_clash_objects": sorted(
                    set(arrays["object"][test & (arrays["population"] == "CLASH")].tolist())
                ),
            }
        )
    full = np.ones(len(arrays["target"]), dtype=bool)
    full_selections = {}
    for lane in lane_names:
        row, loss = _best_row(behaviors[lane], arrays, full)
        full_selections[lane] = {
            "program": _program_description(programs[lane], row, labels, config49),
            "full_data_balanced_training_loss": loss,
        }
    scores = {lane: _score(arrays, predictions[lane]) for lane in lane_names}
    score_objects = scores["score_only"]["object_losses"]
    diverse_objects = scores["diversity_preserving"]["object_losses"]
    keys = sorted(score_objects)
    diff = np.asarray([score_objects[key] - diverse_objects[key] for key in keys])
    counterexample = diff < 0.0
    leave_one = [float(np.mean(np.delete(diff, index))) for index in range(len(diff))]
    trim_count = max(1, int(0.1 * len(diff)))
    trimmed = np.sort(diff)[trim_count:-trim_count]
    improvement = 100.0 * (
        scores["score_only"]["balanced_loss"]
        - scores["diversity_preserving"]["balanced_loss"]
    ) / scores["score_only"]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(keys),
        "raw_counterexample_count": int(np.sum(counterexample)),
        "quality_verified_counterexample_count": int(np.sum(counterexample)),
        "uncertainty_resolved_counterexample_count": 0,
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool(
            (min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "trim_changes_sign": bool(
            (float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item53-archive-evaluation-1.0",
            "item": 53,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "fold_ledger": fold_ledger,
            "full_data_selections": full_selections,
            "scores": scores,
            "diversity": manifest["diversity"],
            "cross_archive_ordinal_overlap": manifest[
                "cross_archive_ordinal_overlap"
            ],
            "diversity_preserving_improvement_percent": improvement,
            "diversity_to_score_only_loss_ratio": scores[
                "diversity_preserving"
            ]["balanced_loss"]
            / scores["score_only"]["balanced_loss"],
            "paired_sign_flip_p": _paired_p(diff, config),
            "robustness": {
                "leave_one_min_mean_score_only_minus_diverse_loss": min(leave_one),
                "leave_one_max_mean_score_only_minus_diverse_loss": max(leave_one),
                "trimmed_mean_score_only_minus_diverse_loss": float(np.mean(trimmed)),
            },
            "counterexamples": [
                {
                    "object": key,
                    "diversity_archive_worse_than_score_only": bool(counterexample[index]),
                    "terminal_veto": False,
                }
                for index, key in enumerate(keys)
            ],
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": assessment,
            "counts": {
                "source_database_records": manifest["source_region_records"],
                "unique_candidate_pool": manifest[
                    "source_unique_representative_ordinals"
                ],
                "archive_slots_per_lane": config["archives"]["slots"],
                "source_database_records_deleted": 0,
                "post_evaluation_archive_changes": 0,
                "sealed_confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "archive_comparison_completed": True,
                "all_source_database_records_preserved": True,
                "low_score_candidates_deleted": False,
                "formula_family_pruned": False,
                "fresh_confirmation_completed": False,
                "historical_novelty_established": False,
            },
            "limitations": [
                "The candidate pool consists of Item 52 region champions selected using already exposed full-data losses, so this is retrospective archive engineering rather than fresh confirmation.",
                "A 64-slot archive is a view over all 1,000 source records; it is not a deletion policy.",
                "Categorical niche coverage does not prove physical-mechanism independence or historical novelty.",
                "The same S4TM and CLASH data limitations and baryonic-mass uncertainties remain active.",
                "Neither one worse object nor the number of worse objects prunes a candidate or niche.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    manifest = _read_json(_source_path(root, config, "archive_manifest"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    diversity = evaluation["diversity"]["diversity_preserving"]
    gates_config = config["evaluation"]["promotion_gates"]
    gates = {
        "all_8_operators_preserved": diversity["distinct_binary_operators"] == 8,
        "all_16_ordered_source_pairs_preserved": diversity[
            "distinct_ordered_source_pairs"
        ]
        == 16,
        "operator_source_niches_at_least": diversity[
            "distinct_operator_source_niches"
        ]
        >= gates_config["operator_source_niches_at_least"],
        "diversity_archive_oof_loss_at_most_score_only_multiplier": evaluation[
            "diversity_to_score_only_loss_ratio"
        ]
        <= gates_config[
            "diversity_archive_oof_loss_at_most_score_only_multiplier"
        ],
        "all_source_database_records_preserved": evaluation["counts"][
            "source_database_records_deleted"
        ]
        == 0,
        "post_evaluation_archive_changes": evaluation["counts"][
            "post_evaluation_archive_changes"
        ]
        == 0,
        "sealed_confirmation_rows": evaluation["counts"][
            "sealed_confirmation_rows"
        ]
        == 0,
    }
    complete = all(gates.values())
    bindings = {}
    for name, key in (
        ("preflight", "preflight_manifest"),
        ("archive_manifest", "archive_manifest"),
        ("evaluation", "evaluation_result"),
    ):
        path = _source_path(root, config, key)
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
        }
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item53-diversity-preservation-result-1.0",
            "item": 53,
            "goal": "GRAVITY_ROADMAP_ITEM_53_DIVERSITY_PRESERVATION",
            "decision": (
                "ITEM53_DIVERSITY_ARCHIVE_PRESERVES_NICHES_WITHIN_PREDICTIVE_TOLERANCE"
                if complete
                else "ITEM53_DIVERSITY_ARCHIVE_CONFIGURATION_NOT_PROMOTED"
            ),
            "gates": gates,
            "scores": evaluation["scores"],
            "diversity": evaluation["diversity"],
            "diversity_preserving_improvement_percent": evaluation[
                "diversity_preserving_improvement_percent"
            ],
            "diversity_to_score_only_loss_ratio": evaluation[
                "diversity_to_score_only_loss_ratio"
            ],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
            "cross_archive_ordinal_overlap": evaluation[
                "cross_archive_ordinal_overlap"
            ],
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "counts": evaluation["counts"],
            "source_bindings": bindings,
            "claims": {
                "roadmap_item_53_complete": complete,
                "diversity_preservation_operational": complete,
                "all_source_database_records_preserved": True,
                "low_score_candidates_deleted": False,
                "formula_family_pruned": False,
                "fresh_confirmation_completed": False,
                "historical_novelty_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Advance to Item 54 equivalence detection; use the diversity archive as a protected view while merging only demonstrated rewrites or behaviorally identical programs.",
            "preflight": preflight,
            "archive_claims": manifest["claims"],
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "preflight": _read_json(_source_path(root, config, "preflight_manifest"))
        == build_preflight_manifest(root),
        "archive_manifest": _read_json(_source_path(root, config, "archive_manifest"))
        == build_archive_manifest(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "archive", "evaluate", "aggregate", "replay")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
    elif args.command == "archive":
        result = str(write_archive_manifest(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
