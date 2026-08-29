"""Item 55 observational causal-variable diagnostics for the Item 45 lead."""

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
    AXES,
    _best_candidate,
    _evaluation_arrays,
    _object_weights,
    _paired_p,
    _predict,
    _score,
    admissible_candidates,
    interaction_bank,
    load_config as _load_item45_config,
)


CONFIG_PATH = Path("configs/gravity_item55_causal_variable_tests_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM54_RESULT_PATH = Path("runs/gravity/roadmap/item-54-equivalence-detection-v1.json")
ITEM45_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-45-universal-interactions-v1-source/joint-evaluation-result.json"
)


class GravityItem55Error(RuntimeError):
    """Raised when the causal diagnostic or conservative claim boundary changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item55-causal-variable-tests-config-1.0"
        or int(config.get("item", -1)) != 55
    ):
        raise GravityItem55Error("unexpected Item 55 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem55Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem55Error("Item 55 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem55Error("malformed Item 55 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem55Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM54_RESULT_PATH)
    required = config["required_predecessor"]
    if predecessor["decision"] != required["decision"]:
        raise GravityItem55Error("Item 54 decision binding changed")
    if predecessor["content_sha256"] != required["content_sha256"]:
        raise GravityItem55Error("Item 54 content binding changed")
    policy = config["claim_policy"]
    if policy["observational_ablation_establishes_causality"]:
        raise GravityItem55Error("observational ablation was mislabeled causal proof")
    if policy["matched_observational_subset_establishes_intervention"]:
        raise GravityItem55Error("observational matching was mislabeled intervention")
    if policy["single_counterexample_terminal"] or policy["formula_family_pruned"]:
        raise GravityItem55Error("causal diagnostic became a terminal prune")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item45 = _read_json(root / ITEM45_EVALUATION_PATH)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item55-preflight-1.0",
            "item": 55,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "target_candidate": config["target_candidate"],
            "item45_selected_candidate": item45["selected_candidate"],
            "tests": config["tests"],
            "causal_lead_gates": config["evaluation"]["causal_lead_gates"],
            "post_evaluation_tests": 0,
            "sealed_confirmation_rows": 0,
            "paid_model_calls": 0,
            "claim_policy": config["claim_policy"],
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def _fold_ids(root: Path) -> dict[int, int]:
    evaluation = _read_json(root / ITEM45_EVALUATION_PATH)
    return {
        int(row["fold"]): int(row["selected_interaction"]["candidate_id"])
        for row in evaluation["fold_ledger"]
    }


def _fixed_oof(
    arrays: Mapping[str, Any], config45: Mapping[str, Any], fold_ids: Mapping[int, int]
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _predict(
            candidate_id, arrays, config45, bank_key="interaction_bank"
        )[test]
    return prediction


def _population_label_control(arrays: Mapping[str, Any], folds: int) -> np.ndarray:
    residual = arrays["target"] - arrays["base"]
    prediction = np.empty(len(residual), dtype=float)
    for fold in range(folds):
        train = arrays["fold"] != fold
        test = ~train
        weights = _object_weights(arrays, train)
        for population in ("S4TM", "CLASH"):
            train_pop = train & (arrays["population"] == population)
            test_pop = test & (arrays["population"] == population)
            mean = float(
                np.sum(residual[train_pop] * weights[train_pop])
                / np.sum(weights[train_pop])
            )
            prediction[test_pop] = arrays["base"][test_pop] + mean
    return prediction


def _ablation_oof(
    arrays: Mapping[str, Any], config45: Mapping[str, Any],
    fold_ids: Mapping[int, int], axis_index: int
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        train = arrays["fold"] != fold
        test = ~train
        ablated = dict(arrays)
        primitives = np.asarray(arrays["primitives"], float).copy()
        for population in ("S4TM", "CLASH"):
            train_pop = train & (arrays["population"] == population)
            primitives[arrays["population"] == population, axis_index] = np.median(
                primitives[train_pop, axis_index]
            )
        ablated["primitives"] = primitives
        ablated["interaction_bank"] = interaction_bank(primitives, config45)[0].T
        prediction[test] = _predict(
            candidate_id, ablated, config45, bank_key="interaction_bank"
        )[test]
    return prediction


def _object_primitives(arrays: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = sorted(
        {(str(pop), str(obj)) for pop, obj in zip(arrays["population"], arrays["object"], strict=True)}
    )
    values = []
    populations = []
    names = []
    for population, name in keys:
        mask = (arrays["population"] == population) & (arrays["object"] == name)
        values.append(np.mean(arrays["primitives"][mask], axis=0))
        populations.append(population)
        names.append(name)
    return np.asarray(values), np.asarray(populations), np.asarray(names)


def _overlap_coefficients(values: np.ndarray, populations: np.ndarray) -> dict[str, float]:
    result = {}
    for index, axis in enumerate(AXES):
        left = values[populations == "S4TM", index]
        right = values[populations == "CLASH", index]
        overlap = max(0.0, min(float(np.max(left)), float(np.max(right))) - max(float(np.min(left)), float(np.min(right))))
        denominator = min(float(np.ptp(left)), float(np.ptp(right)))
        result[axis] = 0.0 if denominator <= 0.0 else overlap / denominator
    return result


def _classification_accuracy(values: np.ndarray, populations: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    predictions = []
    for index in range(len(values)):
        train = np.arange(len(values)) != index
        mean = np.mean(values[train], axis=0)
        scale = np.std(values[train], axis=0)
        scale[scale < 1e-12] = 1.0
        point = (values[index] - mean) / scale
        distances = {}
        for population in ("S4TM", "CLASH"):
            centroid = np.mean((values[train & (populations == population)] - mean) / scale, axis=0)
            distances[population] = float(np.linalg.norm(point - centroid))
        predicted = min(distances, key=distances.get)
        predictions.append(
            {
                "index": index,
                "observed_population": str(populations[index]),
                "predicted_population": predicted,
                "correct": predicted == populations[index],
                "distances": distances,
            }
        )
    return float(np.mean([row["correct"] for row in predictions])), predictions


def _common_support(
    values: np.ndarray, populations: np.ndarray, names: np.ndarray, caliper: float
) -> dict[str, Any]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (values - mean) / scale
    galaxy = np.flatnonzero(populations == "S4TM")
    cluster = np.flatnonzero(populations == "CLASH")
    pairs = []
    for left in galaxy:
        distances = np.linalg.norm(standardized[cluster] - standardized[left], axis=1)
        local = int(np.argmin(distances))
        right = int(cluster[local])
        pairs.append(
            {
                "s4tm_object": str(names[left]),
                "nearest_clash_object": str(names[right]),
                "standardized_distance": float(distances[local]),
                "within_caliper": bool(distances[local] <= caliper),
            }
        )
    return {
        "caliper": caliper,
        "s4tm_objects": len(galaxy),
        "clash_objects": len(cluster),
        "pairs_within_caliper": sum(row["within_caliper"] for row in pairs),
        "nearest_pairs": pairs,
    }


def _population_loss(
    arrays: Mapping[str, Any], prediction: np.ndarray, population: str
) -> float:
    error = np.square((prediction - arrays["target"]) / arrays["sigma"])
    mask = arrays["population"] == population
    names = sorted(set(arrays["object"][mask].tolist()))
    return float(
        np.mean([np.mean(error[mask & (arrays["object"] == name)]) for name in names])
    )


def build_diagnostic_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    config45 = _load_item45_config(root)
    arrays = _evaluation_arrays(root, config45)
    fold_ids = _fold_ids(root)
    baseline_prediction = _fixed_oof(arrays, config45, fold_ids)
    label_prediction = _population_label_control(
        arrays, int(config["evaluation"]["outer_folds"])
    )
    baseline_score = _score(arrays, baseline_prediction)
    label_score = _score(arrays, label_prediction)

    ablations = {}
    for axis_index, axis in enumerate(AXES):
        prediction = _ablation_oof(arrays, config45, fold_ids, axis_index)
        score = _score(arrays, prediction)
        ablations[axis] = {
            "score": score,
            "relative_balanced_loss_increase": (
                score["balanced_loss"] - baseline_score["balanced_loss"]
            )
            / baseline_score["balanced_loss"],
            "interpretation": "predictive reliance under within-population median replacement; not an intervention",
        }

    object_values, object_populations, object_names = _object_primitives(arrays)
    overlap = _overlap_coefficients(object_values, object_populations)
    accuracy, classification_records = _classification_accuracy(
        object_values, object_populations
    )
    support = _common_support(
        object_values,
        object_populations,
        object_names,
        float(config["tests"]["common_support_caliper"]),
    )

    admitted, admission_audit = admissible_candidates(config45)
    ood = {}
    backends = set()
    for train_population, test_population in (("S4TM", "CLASH"), ("CLASH", "S4TM")):
        train = arrays["population"] == train_population
        candidate_id, training_loss, backend, evaluations = _best_candidate(
            admitted, arrays, train, config45, bank_key="interaction_bank"
        )
        prediction = _predict(
            candidate_id, arrays, config45, bank_key="interaction_bank"
        )
        candidate_loss = _population_loss(arrays, prediction, test_population)
        newton_loss = _population_loss(arrays, arrays["base"], test_population)
        backends.add(backend)
        ood[f"train_{train_population}_test_{test_population}"] = {
            "selected_candidate_id": candidate_id,
            "training_balanced_loss_on_single_population_weight_scale": training_loss,
            "test_population_loss": candidate_loss,
            "baryonic_newton_test_population_loss": newton_loss,
            "relative_improvement_over_baryonic_newton": (
                newton_loss - candidate_loss
            )
            / newton_loss,
            "candidate_point_evaluations": evaluations,
            "backend": backend,
        }

    keys = sorted(baseline_score["object_losses"])
    diff = np.asarray(
        [
            label_score["object_losses"][key]
            - baseline_score["object_losses"][key]
            for key in keys
        ]
    )
    raw_counterexample = diff < 0.0
    improvement = 100.0 * (
        label_score["balanced_loss"] - baseline_score["balanced_loss"]
    ) / label_score["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": 0,
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": False,
        "trim_changes_sign": False,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item55-causal-diagnostic-1.0",
            "item": 55,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "scores": {
                "item45_universal_interaction": baseline_score,
                "population_label_only": label_score,
                "baryonic_newton": _score(arrays, arrays["base"]),
            },
            "item45_improvement_over_population_label_percent": improvement,
            "paired_sign_flip_p_item45_vs_population_label": _paired_p(diff, config),
            "within_population_axis_ablations": ablations,
            "object_level_population_overlap": overlap,
            "population_label_predictability": {
                "leave_one_object_out_nearest_centroid_accuracy": accuracy,
                "records": classification_records,
            },
            "common_support": support,
            "cross_population_ood": ood,
            "counterexamples_item45_vs_population_label": [
                {
                    "object": key,
                    "population_label_control_better": bool(raw_counterexample[index]),
                    "terminal_veto": False,
                }
                for index, key in enumerate(keys)
            ],
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": assessment,
            "compute": {
                "backends": sorted(backends),
                "admissible_item45_candidates": len(admitted["candidate_id"]),
                "admission_audit": admission_audit,
                "ood_candidate_point_evaluations": sum(
                    row["candidate_point_evaluations"] for row in ood.values()
                ),
            },
            "counts": {
                "primitive_axes": len(AXES),
                "object_level_rows": len(object_values),
                "s4tm_objects": int(np.sum(object_populations == "S4TM")),
                "clash_objects": int(np.sum(object_populations == "CLASH")),
                "sealed_confirmation_rows": 0,
                "post_evaluation_tests": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "observational_causal_diagnostics_completed": True,
                "causality_established": False,
                "intervention_completed": False,
                "fresh_confirmation_completed": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
            },
            "limitations": [
                "All diagnostics use already exposed S4TM and CLASH development data and cannot establish causality without interventions, natural experiments, or independent prospective tests.",
                "The primitive definitions include constructed proxies such as horizon occupancy, analytic galaxy gradients, and model-dependent cluster profiles.",
                "Median replacement is an ablation and can create unrealistic combinations; it measures reliance rather than causal effect.",
                "Nearest-centroid classification and a fixed distance caliper are diagnostics, not exhaustive tests for dataset leakage.",
                "Cross-population transfer selects within the existing Item 45 grammar and does not prove universal physics.",
                "One empirical mismatch and mismatch counts remain non-terminal.",
            ],
        }
    )


def write_diagnostic_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "diagnostic_result")
    _write_json(path, build_diagnostic_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    result = _read_json(_source_path(root, config, "diagnostic_result"))
    thresholds = config["evaluation"]["causal_lead_gates"]
    overlap = result["object_level_population_overlap"]
    ablations = result["within_population_axis_ablations"]
    gates = {
        "item45_beats_population_label_control": result["scores"][
            "item45_universal_interaction"
        ]["balanced_loss"]
        < result["scores"]["population_label_only"]["balanced_loss"],
        "geometry_or_density_ablation_relative_loss_increase_at_least": max(
            ablations[axis]["relative_balanced_loss_increase"]
            for axis in ("geometry", "density")
        )
        >= thresholds[
            "geometry_or_density_ablation_relative_loss_increase_at_least"
        ],
        "geometry_and_density_cross_population_overlap_each_at_least": all(
            overlap[axis]
            >= thresholds[
                "geometry_and_density_cross_population_overlap_each_at_least"
            ]
            for axis in ("geometry", "density")
        ),
        "nearest_cross_population_object_pairs_within_caliper_at_least": result[
            "common_support"
        ]["pairs_within_caliper"]
        >= thresholds["nearest_cross_population_object_pairs_within_caliper_at_least"],
        "each_cross_population_ood_prediction_beats_baryonic_newton": all(
            row["test_population_loss"] < row["baryonic_newton_test_population_loss"]
            for row in result["cross_population_ood"].values()
        ),
        "population_classification_accuracy_at_most": result[
            "population_label_predictability"
        ]["leave_one_object_out_nearest_centroid_accuracy"]
        <= thresholds["population_classification_accuracy_at_most"],
        "sealed_confirmation_rows": result["counts"]["sealed_confirmation_rows"] == 0,
        "post_evaluation_tests": result["counts"]["post_evaluation_tests"] == 0,
    }
    complete = bool(result["claims"]["observational_causal_diagnostics_completed"])
    causal_lead = complete and all(gates.values())
    bindings = {}
    for name, key in (
        ("preflight", "preflight_manifest"),
        ("diagnostic", "diagnostic_result"),
    ):
        path = _source_path(root, config, key)
        bindings[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256_file(path),
        }
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item55-causal-variable-result-1.0",
            "item": 55,
            "goal": "GRAVITY_ROADMAP_ITEM_55_CAUSAL_VARIABLE_TESTS",
            "decision": (
                "ITEM55_OBSERVATIONAL_CAUSAL_LEAD_REQUIRES_INTERVENTION_AND_FRESH_TEST"
                if causal_lead
                else "ITEM55_CAUSALITY_NOT_ESTABLISHED_PROXY_RISK_RETAINED"
            ),
            "gates": gates,
            "scores": result["scores"],
            "item45_improvement_over_population_label_percent": result[
                "item45_improvement_over_population_label_percent"
            ],
            "paired_sign_flip_p_item45_vs_population_label": result[
                "paired_sign_flip_p_item45_vs_population_label"
            ],
            "axis_ablation_relative_loss_increase": {
                axis: row["relative_balanced_loss_increase"]
                for axis, row in result["within_population_axis_ablations"].items()
            },
            "object_level_population_overlap": overlap,
            "population_label_predictability": result[
                "population_label_predictability"
            ]["leave_one_object_out_nearest_centroid_accuracy"],
            "common_support_pairs_within_caliper": result["common_support"][
                "pairs_within_caliper"
            ],
            "cross_population_ood": result["cross_population_ood"],
            "counterexample_policy_assessment": result[
                "counterexample_policy_assessment"
            ],
            "counts": result["counts"],
            "compute": result["compute"],
            "source_bindings": bindings,
            "claims": {
                "roadmap_item_55_complete": complete,
                "causal_variable_lead_passed_all_observational_gates": causal_lead,
                "causality_established": False,
                "intervention_completed": False,
                "fresh_confirmation_completed": False,
                "population_proxy_risk_eliminated": causal_lead,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
                "historical_novelty_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
            },
            "limitations": result["limitations"],
            "next_action": "Advance to Item 56 disk-galaxy gate. Treat geometry-density as a predictive observational lead only if its proxy risks are made explicit; do not call it causal without intervention or prospective transfer.",
            "preflight": preflight,
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
        "diagnostic_result": _read_json(_source_path(root, config, "diagnostic_result"))
        == build_diagnostic_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "diagnose", "aggregate", "replay")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
    elif args.command == "diagnose":
        result = str(write_diagnostic_result(root))
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
