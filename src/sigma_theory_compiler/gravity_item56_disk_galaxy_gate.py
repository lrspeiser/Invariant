"""Item 56 fixed-candidate disk-galaxy rotation-curve gate."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_g0_experiment import (
    _empirical_rar,
    _galaxy_arrays,
    _newtonian,
    _nfw_out_of_fold,
    radial_folds,
    score_predictions,
)
from sigma_theory_compiler.gravity_item2_shape_anisotropy import parse_sparc_properties
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.sparc_full_sample import assemble

CONFIG_PATH = Path("configs/gravity_item56_disk_galaxy_gate_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM55_RESULT_PATH = Path("runs/gravity/roadmap/item-55-causal-variable-tests-v1.json")
ITEM45_RESULT_PATH = Path("runs/gravity/roadmap/item-45-universal-interactions-v1.json")
SPARC_PROPERTIES_PATH = Path(
    "runs/gravity/roadmap/item-02-shape-anisotropy-v1-source/sparc_table1.tsv"
)


class GravityItem56Error(RuntimeError):
    """Raised when the disk-galaxy protocol, boundary, or result changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(root: Path, config: Mapping[str, Any], *, require_bound: bool = True) -> None:
    if (
        config.get("schema_version") != "invariant-gravity-item56-disk-galaxy-gate-config-1.0"
        or int(config.get("item", -1)) != 56
    ):
        raise GravityItem56Error("unexpected Item 56 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem56Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem56Error("Item 56 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem56Error("malformed Item 56 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem56Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / ITEM55_RESULT_PATH)
    required = config["required_predecessor"]
    if predecessor["decision"] != required["decision"]:
        raise GravityItem56Error("Item 55 decision binding changed")
    if predecessor["content_sha256"] != required["content_sha256"]:
        raise GravityItem56Error("Item 55 content binding changed")
    item45 = _read_json(root / ITEM45_RESULT_PATH)
    target = config["target_candidate"]
    selected = item45["selected_candidate"]
    if (
        int(selected["candidate_id"]) != int(target["candidate_id"])
        or int(selected["recipe_id"]) != int(target["recipe_id"])
        or selected["parameters"] != target["parameters"]
        or selected["interaction_expression"] != target["interaction_expression"]
    ):
        raise GravityItem56Error("Item 45 target candidate changed")
    boundary = config["data_boundary"]
    if (
        int(boundary["confirmation_response_rows_allowed"]) != 0
        or boundary["fresh_confirmation_claim_allowed"]
        or boundary["candidate_lineage_used_sparc_responses"]
    ):
        raise GravityItem56Error("Item 56 confirmation or lineage boundary changed")
    policy = config["counterexample_policy"]
    if (
        policy["single_counterexample_terminal"]
        or policy["counterexample_count_terminal"]
        or policy["finite_sparc_exploration_sample_may_prune_formula_family"]
    ):
        raise GravityItem56Error("Item 56 permits empirical over-pruning")
    predictor = config["predictor_contract"]
    if (
        predictor["galaxy_identifier_allowed_as_predictor"]
        or predictor["observed_velocity_allowed_as_predictor"]
        or predictor["uncertainty_allowed_as_predictor"]
        or predictor["per_galaxy_gravitational_coefficient_allowed"]
    ):
        raise GravityItem56Error("Item 56 predictor contract leaks or retunes")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def candidate_velocity(
    radius: np.ndarray,
    vbar2: np.ndarray,
    effective_radius: float,
    candidate: Mapping[str, Any],
    acceleration_scale: float,
) -> np.ndarray:
    """Apply the unchanged Item 45 response to one disk galaxy."""

    if effective_radius <= 0.0 or acceleration_scale <= 0.0:
        raise GravityItem56Error("non-positive physical scale")
    if np.any(radius <= 0.0) or np.any(vbar2 <= 0.0):
        raise GravityItem56Error("non-positive radius or baryonic velocity squared")
    u = vbar2 / radius / acceleration_scale
    log_geometry = np.log10(radius / effective_radius)
    log_density = np.log10(u)
    geometry = log_geometry / (1.0 + np.abs(log_geometry))
    density = log_density / (3.0 + np.abs(log_density))
    interaction = geometry * np.tanh(2.0 * density)
    gate = 0.5 + 0.5 * np.tanh(2.0 * interaction)
    parameters = candidate["parameters"]
    amplitude = float(parameters["amplitude"])
    exponent = float(parameters["acceleration_exponent"])
    transition = float(parameters["transition_u"])
    nu = 1.0 + amplitude * np.power(u, -exponent) / (1.0 + u / transition) * (0.05 + 0.95 * gate)
    prediction = np.sqrt(vbar2 * nu)
    if np.any(~np.isfinite(prediction)) or np.any(prediction < 0.0):
        raise GravityItem56Error("candidate produced an invalid velocity")
    return prediction


def _numeric_score(prediction: np.ndarray, observed: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(np.square((prediction - observed) / sigma)))


def _score_bundle(
    predictions: Mapping[str, np.ndarray], observed: np.ndarray, sigma: np.ndarray
) -> dict[str, Any]:
    return {
        model: {
            "conditional_metrics": score_predictions(prediction, observed, sigma),
            "mean_squared_standardized_residual": _numeric_score(prediction, observed, sigma),
        }
        for model, prediction in predictions.items()
    }


def _paired_p(difference: np.ndarray, config: Mapping[str, Any]) -> float:
    observed = float(np.mean(difference))
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    rng = np.random.default_rng(int(config["evaluation"]["permutation_seed"]))
    extreme = 1
    for _ in range(count):
        signs = rng.choice((-1.0, 1.0), size=len(difference))
        if float(np.mean(difference * signs)) >= observed:
            extreme += 1
    return extreme / (count + 1)


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item45 = _read_json(root / ITEM45_RESULT_PATH)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item56-preflight-1.0",
            "item": 56,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "target_candidate": config["target_candidate"],
            "item45_selected_candidate": item45["selected_candidate"],
            "predictor_contract": config["predictor_contract"],
            "evaluation": config["evaluation"],
            "confirmation_response_rows": 0,
            "post_evaluation_formula_cells": 0,
            "paid_model_calls": 0,
        }
    )


def write_preflight_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "preflight_manifest")
    _write_json(path, build_preflight_manifest(root))
    return path


def _systematic_predictions(
    arrays: Mapping[str, np.ndarray],
    effective_radius: float,
    properties: Mapping[str, Any],
    config: Mapping[str, Any],
    variant: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    radius = arrays["radius"]
    vbar2 = arrays["vbar2"].copy()
    observed = arrays["vobs"].copy()
    sigma = arrays["sigma"].copy()
    if variant.startswith("baryonic_mass_"):
        shift = float(variant.rsplit("_", 1)[1])
        vbar2 *= 10.0**shift
    elif variant.startswith("inclination_"):
        sign = -1.0 if variant.endswith("minus") else 1.0
        inclination = float(properties["inclination_deg"])
        uncertainty = float(properties["inclination_uncertainty_deg"])
        low, high = config["evaluation"]["systematic_variants"]["inclination_clip_degrees"]
        endpoint = float(np.clip(inclination + sign * uncertainty, low, high))
        scale = math.sin(math.radians(inclination)) / math.sin(math.radians(endpoint))
        observed *= scale
        sigma *= scale
    else:
        raise GravityItem56Error(f"unknown systematic variant: {variant}")
    a0 = float(config["predictor_contract"]["acceleration_scale_km2_s2_kpc"])
    predictions = {
        "item45_geometry_density": candidate_velocity(
            radius, vbar2, effective_radius, config["target_candidate"], a0
        ),
        "empirical_rar": _empirical_rar(radius, vbar2, a0),
        "newtonian_baryons": _newtonian(vbar2),
    }
    return predictions, observed, sigma


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    population = assemble(root)
    properties = parse_sparc_properties(root / SPARC_PROPERTIES_PATH)
    expected_galaxies = 139
    expected_points = 2720
    if len(population.exploration) != expected_galaxies:
        raise GravityItem56Error("SPARC exploration galaxy count changed")
    if sum(galaxy.count for galaxy in population.exploration) != expected_points:
        raise GravityItem56Error("SPARC exploration row count changed")

    model_ids = (
        "item45_geometry_density",
        "empirical_rar",
        "newtonian_baryons",
        "nfw_halo_ceiling",
    )
    pooled_prediction: dict[str, list[np.ndarray]] = {model: [] for model in model_ids}
    pooled_observed: list[np.ndarray] = []
    pooled_sigma: list[np.ndarray] = []
    galaxy_losses: dict[str, dict[str, float]] = {model: {} for model in model_ids}
    per_galaxy = []
    fold_rows: dict[int, dict[str, Any]] = {
        fold: {
            "prediction": {model: [] for model in model_ids},
            "observed": [],
            "sigma": [],
            "object_losses": {model: [] for model in model_ids},
        }
        for fold in range(5)
    }
    variant_ids = [
        *(
            f"baryonic_mass_{float(shift):+.2f}"
            for shift in config["evaluation"]["systematic_variants"]["baryonic_mass_log10_shifts"]
        ),
        "inclination_minus",
        "inclination_plus",
    ]
    variant_losses: dict[str, dict[str, dict[str, float]]] = {
        variant: {model: {} for model in model_ids[:3]} for variant in variant_ids
    }
    a0 = float(config["predictor_contract"]["acceleration_scale_km2_s2_kpc"])

    for galaxy in population.exploration:
        arrays = _galaxy_arrays(galaxy)
        property_row = properties[galaxy.name]
        effective_radius = float(property_row["effective_radius_kpc"])
        folds = radial_folds(galaxy.count, maximum_folds=5, minimum_training_rows=3)
        nfw_prediction, nfw_fits = _nfw_out_of_fold(arrays, folds, 64)
        predictions = {
            "item45_geometry_density": candidate_velocity(
                arrays["radius"],
                arrays["vbar2"],
                effective_radius,
                config["target_candidate"],
                a0,
            ),
            "empirical_rar": _empirical_rar(arrays["radius"], arrays["vbar2"], a0),
            "newtonian_baryons": _newtonian(arrays["vbar2"]),
            "nfw_halo_ceiling": nfw_prediction,
        }
        scores = _score_bundle(predictions, arrays["vobs"], arrays["sigma"])
        for model in model_ids:
            pooled_prediction[model].append(predictions[model])
            galaxy_losses[model][galaxy.name] = scores[model]["mean_squared_standardized_residual"]
        pooled_observed.append(arrays["vobs"])
        pooled_sigma.append(arrays["sigma"])
        for fold in folds:
            held = np.asarray(fold.holdout, dtype=np.int64)
            block = fold_rows[fold.fold_id]
            block["observed"].append(arrays["vobs"][held])
            block["sigma"].append(arrays["sigma"][held])
            for model in model_ids:
                block["prediction"][model].append(predictions[model][held])
                block["object_losses"][model].append(
                    _numeric_score(
                        predictions[model][held],
                        arrays["vobs"][held],
                        arrays["sigma"][held],
                    )
                )
        per_galaxy.append(
            {
                "galaxy": galaxy.name,
                "point_count": galaxy.count,
                "effective_radius_kpc": effective_radius,
                "fold_count": len(folds),
                "scores": scores,
                "nfw_training_fits": nfw_fits,
                "candidate_beats_newton": (
                    galaxy_losses["item45_geometry_density"][galaxy.name]
                    < galaxy_losses["newtonian_baryons"][galaxy.name]
                ),
                "candidate_not_worse_than_rar": (
                    galaxy_losses["item45_geometry_density"][galaxy.name]
                    <= galaxy_losses["empirical_rar"][galaxy.name]
                ),
                "terminal_veto": False,
            }
        )
        for variant in variant_ids:
            varied, observed, sigma = _systematic_predictions(
                arrays, effective_radius, property_row, config, variant
            )
            for model, prediction in varied.items():
                variant_losses[variant][model][galaxy.name] = _numeric_score(
                    prediction, observed, sigma
                )

    observed = np.concatenate(pooled_observed)
    sigma = np.concatenate(pooled_sigma)
    aggregate = {}
    for model in model_ids:
        prediction = np.concatenate(pooled_prediction[model])
        losses = np.asarray(list(galaxy_losses[model].values()))
        aggregate[model] = {
            "equal_galaxy_mean_squared_standardized_residual": float(np.mean(losses)),
            "median_galaxy_mean_squared_standardized_residual": float(np.median(losses)),
            "pooled": score_predictions(prediction, observed, sigma),
        }
    radial_blocks = {}
    for fold, block in fold_rows.items():
        block_observed = np.concatenate(block["observed"])
        block_sigma = np.concatenate(block["sigma"])
        radial_blocks[str(fold)] = {
            "radial_role": ("inner" if fold == 0 else "outer" if fold == 4 else "intermediate"),
            "row_count": len(block_observed),
            "scores": {
                model: {
                    "equal_galaxy_mean_squared_standardized_residual": float(
                        np.mean(block["object_losses"][model])
                    ),
                    "pooled": score_predictions(
                        np.concatenate(block["prediction"][model]),
                        block_observed,
                        block_sigma,
                    ),
                }
                for model in model_ids
            },
        }
    systematics = {}
    for variant, models in variant_losses.items():
        systematics[variant] = {
            "scores": {
                model: {
                    "equal_galaxy_mean_squared_standardized_residual": float(
                        np.mean(list(losses.values()))
                    )
                }
                for model, losses in models.items()
            },
            "candidate_beats_newton": (
                np.mean(list(models["item45_geometry_density"].values()))
                < np.mean(list(models["newtonian_baryons"].values()))
            ),
            "candidate_galaxy_wins_vs_rar": sum(
                models["item45_geometry_density"][name] <= models["empirical_rar"][name]
                for name in models["item45_geometry_density"]
            ),
        }

    names = sorted(galaxy_losses["item45_geometry_density"])
    difference_vs_newton = np.asarray(
        [
            galaxy_losses["newtonian_baryons"][name]
            - galaxy_losses["item45_geometry_density"][name]
            for name in names
        ]
    )
    raw_counterexamples = [
        name
        for name in names
        if galaxy_losses["item45_geometry_density"][name] > galaxy_losses["empirical_rar"][name]
    ]
    stable_counterexamples = [
        name
        for name in raw_counterexamples
        if all(
            variant_losses[variant]["item45_geometry_density"][name]
            > variant_losses[variant]["empirical_rar"][name]
            for variant in variant_ids
        )
    ]
    improvement_vs_newton = (
        100.0
        * (
            aggregate["newtonian_baryons"]["equal_galaxy_mean_squared_standardized_residual"]
            - aggregate["item45_geometry_density"][
                "equal_galaxy_mean_squared_standardized_residual"
            ]
        )
        / aggregate["newtonian_baryons"]["equal_galaxy_mean_squared_standardized_residual"]
    )
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(names),
        "raw_counterexample_count": len(raw_counterexamples),
        "quality_verified_counterexample_count": len(stable_counterexamples),
        "uncertainty_resolved_counterexample_count": 0,
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement_vs_newton,
        "quality_gate_passed": False,
        "strongest_baseline_failed": (
            aggregate["item45_geometry_density"]["equal_galaxy_mean_squared_standardized_residual"]
            > aggregate["empirical_rar"]["equal_galaxy_mean_squared_standardized_residual"]
        ),
        "leave_one_changes_sign": bool(
            len(difference_vs_newton) > 1
            and any(
                float(np.mean(np.delete(difference_vs_newton, index))) <= 0.0
                for index in range(len(difference_vs_newton))
            )
        ),
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
            "schema_version": "invariant-gravity-item56-disk-galaxy-evaluation-1.0",
            "item": 56,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "target_candidate": config["target_candidate"],
            "aggregate_scores": aggregate,
            "radial_blocks": radial_blocks,
            "systematic_variants": systematics,
            "per_galaxy": per_galaxy,
            "candidate_improvement_over_newton_equal_galaxy_percent": improvement_vs_newton,
            "candidate_galaxy_wins_vs_newton": int(np.sum(difference_vs_newton > 0.0)),
            "candidate_galaxy_wins_vs_rar": len(names) - len(raw_counterexamples),
            "paired_sign_flip_p_candidate_vs_newton": _paired_p(difference_vs_newton, config),
            "raw_counterexamples_vs_rar": raw_counterexamples,
            "counterexamples_stable_across_all_systematic_variants": stable_counterexamples,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": assessment,
            "counts": {
                "exploration_galaxies": len(names),
                "exploration_rows": len(observed),
                "radial_folds": sum(row["fold_count"] for row in per_galaxy),
                "fixed_candidate_cells": 1,
                "post_evaluation_formula_cells": 0,
                "confirmation_response_rows": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "fixed_disk_galaxy_evaluation_completed": True,
                "fresh_confirmation_completed": False,
                "causality_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": [
                "SPARC exploration responses were previously exposed in this repository, so this is a cross-dataset transfer test in the candidate lineage but not fresh confirmation.",
                "Published e_Vobs omits full inclination and distance covariance; the score is conditional on random errors.",
                "Published stellar effective radius is a light-geometry proxy and may not equal the relevant baryonic or gravitational boundary scale.",
                "The NFW comparator is only a flexible performance ceiling and is not used as a target, feature, or claim of truth.",
                "Mass and inclination endpoint audits are narrow sensitivity checks, not a complete nuisance likelihood.",
                "No finite set of galaxy mismatches prunes the formula family, and no single mismatch is terminal.",
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
    result = _read_json(_source_path(root, config, "evaluation_result"))
    scores = result["aggregate_scores"]
    candidate = scores["item45_geometry_density"]
    newton = scores["newtonian_baryons"]
    rar = scores["empirical_rar"]
    threshold = config["evaluation"]["promotion_gates"]
    radial = result["radial_blocks"]
    gates = {
        "beats_newton_equal_galaxy": candidate["equal_galaxy_mean_squared_standardized_residual"]
        < newton["equal_galaxy_mean_squared_standardized_residual"],
        "beats_newton_pooled": float(candidate["pooled"]["chi_square"])
        < float(newton["pooled"]["chi_square"]),
        "relative_equal_galaxy_loss_vs_rar_at_most": candidate[
            "equal_galaxy_mean_squared_standardized_residual"
        ]
        / rar["equal_galaxy_mean_squared_standardized_residual"]
        <= float(threshold["relative_equal_galaxy_loss_vs_rar_maximum"]),
        "radial_blocks_beating_newton_minimum": sum(
            block["scores"]["item45_geometry_density"][
                "equal_galaxy_mean_squared_standardized_residual"
            ]
            < block["scores"]["newtonian_baryons"][
                "equal_galaxy_mean_squared_standardized_residual"
            ]
            for block in radial.values()
        )
        >= int(threshold["radial_blocks_beating_newton_minimum"]),
        "radial_blocks_not_worse_than_rar_minimum": sum(
            block["scores"]["item45_geometry_density"][
                "equal_galaxy_mean_squared_standardized_residual"
            ]
            <= block["scores"]["empirical_rar"]["equal_galaxy_mean_squared_standardized_residual"]
            for block in radial.values()
        )
        >= int(threshold["radial_blocks_not_worse_than_rar_minimum"]),
        "systematic_variants_beating_newton_minimum": sum(
            row["candidate_beats_newton"] for row in result["systematic_variants"].values()
        )
        >= int(threshold["systematic_variants_beating_newton_minimum"]),
        "paired_p_vs_newton_at_most": result["paired_sign_flip_p_candidate_vs_newton"]
        <= float(threshold["paired_p_vs_newton_maximum"]),
        "confirmation_response_rows_zero": result["counts"]["confirmation_response_rows"] == 0,
        "post_evaluation_formula_cells_zero": result["counts"]["post_evaluation_formula_cells"]
        == 0,
    }
    passed = all(gates.values())
    bindings = {}
    for name, key in (("preflight", "preflight_manifest"), ("evaluation", "evaluation_result")):
        path = _source_path(root, config, key)
        bindings[name] = {"path": str(path.relative_to(root)), "sha256": _sha256_file(path)}
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item56-disk-galaxy-result-1.0",
            "item": 56,
            "goal": "GRAVITY_ROADMAP_ITEM_56_DISK_GALAXY_GATE",
            "decision": (
                "PASS_ITEM56_DISK_GALAXY_PREDICTIVE_GATE_NOT_CONFIRMATION"
                if passed
                else "ITEM56_DISK_GALAXY_GATE_NOT_PASSED_LEAD_AND_FAILURES_RETAINED"
            ),
            "gates": gates,
            "target_candidate": result["target_candidate"],
            "aggregate_scores": scores,
            "radial_blocks": result["radial_blocks"],
            "systematic_variants": result["systematic_variants"],
            "candidate_improvement_over_newton_equal_galaxy_percent": result[
                "candidate_improvement_over_newton_equal_galaxy_percent"
            ],
            "candidate_galaxy_wins_vs_newton": result["candidate_galaxy_wins_vs_newton"],
            "candidate_galaxy_wins_vs_rar": result["candidate_galaxy_wins_vs_rar"],
            "paired_sign_flip_p_candidate_vs_newton": result[
                "paired_sign_flip_p_candidate_vs_newton"
            ],
            "counterexample_policy_assessment": result["counterexample_policy_assessment"],
            "counts": result["counts"],
            "source_bindings": bindings,
            "claims": {
                "roadmap_item_56_complete": True,
                "disk_galaxy_predictive_gate_passed": passed,
                "fresh_confirmation_completed": False,
                "causality_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": result["limitations"],
            "next_action": (
                "Preserve the fixed Item 45 result and every SPARC mismatch. Advance to Item 57 independent-galaxy transfer without opening the sealed SPARC confirmation set; an Item 56 miss is diagnostic and does not prune the larger geometry-density family."
            ),
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
        "evaluation": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "evaluate", "aggregate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        result: Any = str(write_preflight_manifest(root))
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
