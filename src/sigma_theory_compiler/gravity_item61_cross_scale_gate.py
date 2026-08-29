"""Item 61: unchanged-parameter cross-scale gate."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_g0_experiment import (
    _empirical_rar,
    _galaxy_arrays,
    _newtonian,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item57_independent_galaxy_gate import (
    ITEM5_SOURCE_PATH,
    _parse_existing_target,
    _parse_predictor_surface_density,
    _predictions,
)
from sigma_theory_compiler.gravity_item57_independent_galaxy_gate import (
    load_config as load_item57_config,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    _law_acceleration,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    load_config as load_item59_config,
)
from sigma_theory_compiler.sparc_full_sample import assemble

CONFIG_PATH = Path("configs/gravity_item61_cross_scale_gate_v1.json")
ITEM59_RESULT = Path("runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json")
ITEM57_SOURCE = Path("runs/gravity/roadmap/item-57-independent-galaxy-gate-v1-source")


class GravityItem61Error(RuntimeError):
    """Raised when the frozen Item 61 boundary or replay changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version") != "invariant-gravity-item61-cross-scale-config-1.0"
        or config.get("item") != 61
        or config.get("status") != "scientific_freeze_before_cross_scale_evaluation"
    ):
        raise GravityItem61Error("unsupported Item 61 config")
    freeze = str(config.get("scientific_freeze_commit", ""))
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem61Error("Item 61 scientific freeze is not bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem61Error("invalid Item 61 freeze marker")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem61Error(f"scientific dependency changed: {relative}")
    candidate = config["candidate"]
    item59 = _read_json(root / ITEM59_RESULT)
    selected = item59["selection"]["selected_qualifying"]["variant"]
    if (
        selected["variant_id"] != candidate["variant_id"]
        or selected["family_id"] != candidate["family_id"]
        or selected["parameters"] != candidate["parameters"]
    ):
        raise GravityItem61Error("Item 59 candidate binding changed")
    if candidate["formula_refit_allowed"] or candidate["scale_specific_parameter_allowed"]:
        raise GravityItem61Error("scale-specific fitting is forbidden")
    populations = config["populations"]
    if (
        populations["sealed_sparc_confirmation_rows_allowed"] != 0
        or populations["new_target_queries_allowed"] != 0
    ):
        raise GravityItem61Error("sealed or new target access is forbidden")
    policy = config["counterexample_policy"]
    if (
        policy["single_counterexample_terminal"]
        or policy["counterexample_count_alone_terminal"]
        or policy["finite_sample_may_prune_formula_family"]
        or policy["global_family_pruning_allowed"]
    ):
        raise GravityItem61Error("empirical over-pruning is forbidden")


def _candidate_velocity(
    radius_kpc: np.ndarray,
    vbar2_km2_s2: np.ndarray,
    config59: Mapping[str, Any],
    beta: float,
) -> np.ndarray:
    kpc_m = float(config59["constants"]["kiloparsec_m"])
    gbar_si = vbar2_km2_s2 / radius_kpc * 1.0e6 / kpc_m
    acceleration = _law_acceleration(
        "cross_scale_boundary",
        {"beta": beta},
        radius_kpc,
        float(np.max(radius_kpc)),
        gbar_si,
        config59,
    )
    return np.sqrt(acceleration * radius_kpc * kpc_m / 1.0e6)


def _loss(prediction: np.ndarray, observed: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(np.square((prediction - observed) / sigma)))


def _aggregate(losses: Mapping[str, list[float]]) -> dict[str, Any]:
    return {
        model: {
            "equal_object_mean_squared_standardized_residual": float(np.mean(values)),
            "median_object_mean_squared_standardized_residual": float(np.median(values)),
        }
        for model, values in losses.items()
    }


def _sparc_evaluation(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    population = assemble(root)
    config59 = load_item59_config(root)
    beta = float(config["candidate"]["parameters"]["beta"])
    a0 = 3702.81458
    losses: dict[str, list[float]] = {name: [] for name in ("candidate", "empirical_rar", "newtonian_baryons")}
    per_object = []
    rows = 0
    for galaxy in population.exploration:
        arrays = _galaxy_arrays(galaxy)
        predictions = {
            "candidate": _candidate_velocity(arrays["radius"], arrays["vbar2"], config59, beta),
            "empirical_rar": _empirical_rar(arrays["radius"], arrays["vbar2"], a0),
            "newtonian_baryons": _newtonian(arrays["vbar2"]),
        }
        scores = {
            name: _loss(prediction, arrays["vobs"], arrays["sigma"])
            for name, prediction in predictions.items()
        }
        for name, score in scores.items():
            losses[name].append(score)
        rows += galaxy.count
        per_object.append(
            {
                "galaxy": galaxy.name,
                "rows": galaxy.count,
                "losses": scores,
                "candidate_beats_rar": scores["candidate"] < scores["empirical_rar"],
                "candidate_beats_newton": scores["candidate"] < scores["newtonian_baryons"],
                "terminal_veto": False,
            }
        )
    expected = config["populations"]["sparc_exploration"]
    if len(per_object) != expected["galaxies"] or rows != expected["rows"]:
        raise GravityItem61Error("SPARC population changed")
    aggregate = _aggregate(losses)
    return {
        "galaxies": len(per_object),
        "rows": rows,
        "aggregate": aggregate,
        "candidate_wins_vs_rar": sum(row["candidate_beats_rar"] for row in per_object),
        "candidate_wins_vs_newton": sum(row["candidate_beats_newton"] for row in per_object),
        "per_object": per_object,
    }


def _little_things_evaluation(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    config57 = load_item57_config(root)
    config59 = load_item59_config(root)
    beta = float(config["candidate"]["parameters"]["beta"])
    photometry_manifest = _read_json(
        root / ITEM57_SOURCE / "photometry-source-manifest.json"
    )
    photometry = {
        str(record["slug"]): record["parsed"] for record in photometry_manifest["records"]
    }
    source = _read_json(root / ITEM5_SOURCE_PATH)
    source_by_slug = {str(record["galaxy"]): record for record in source["records"]}
    losses: dict[str, list[float]] = {name: [] for name in ("candidate", "empirical_rar", "newtonian_baryons")}
    per_object = []
    rows = 0
    for object_row in config57["little_things"]["exploration_objects"]:
        slug = str(object_row["slug"])
        name = str(object_row["vizier_name"])
        record = source_by_slug[slug]
        density_radius, surface_density = _parse_predictor_surface_density(
            root / str(record["predictor"]["path"])
        )
        target = _parse_existing_target(root / str(record["target"]["path"]), expected_name=name)
        valid = (
            (target["radius"] > 0.0)
            & (target["radius"] <= density_radius[-1])
            & (target["observed"] >= 0.0)
            & (target["sigma"] > 0.0)
        )
        radius = target["radius"][valid]
        observed = target["observed"][valid]
        sigma = target["sigma"][valid]
        baseline, _ = _predictions(
            radius,
            density_radius,
            surface_density,
            photometry[slug],
            config57,
            "nominal",
        )
        predictions = {
            "candidate": _candidate_velocity(
                radius, np.square(baseline["newtonian_baryons"]), config59, beta
            ),
            "empirical_rar": baseline["empirical_rar"],
            "newtonian_baryons": baseline["newtonian_baryons"],
        }
        scores = {key: _loss(value, observed, sigma) for key, value in predictions.items()}
        for model, score in scores.items():
            losses[model].append(score)
        rows += len(radius)
        per_object.append(
            {
                "galaxy": slug,
                "rows": len(radius),
                "losses": scores,
                "candidate_beats_rar": scores["candidate"] < scores["empirical_rar"],
                "candidate_beats_newton": scores["candidate"] < scores["newtonian_baryons"],
                "terminal_veto": False,
            }
        )
    expected = config["populations"]["little_things_exploration"]
    if len(per_object) != expected["galaxies"] or rows != expected["evaluated_rows"]:
        raise GravityItem61Error("LITTLE THINGS population changed")
    return {
        "galaxies": len(per_object),
        "rows": rows,
        "aggregate": _aggregate(losses),
        "candidate_wins_vs_rar": sum(row["candidate_beats_rar"] for row in per_object),
        "candidate_wins_vs_newton": sum(row["candidate_beats_newton"] for row in per_object),
        "per_object": per_object,
    }


def build_evaluation(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item59 = _read_json(root / ITEM59_RESULT)
    sparc = _sparc_evaluation(root, config)
    little = _little_things_evaluation(root, config)
    cluster = {
        "clusters": item59["counts"]["clusters"],
        "confirmation_clusters": item59["counts"]["confirmation_clusters"],
        "confirmation_rows": item59["counts"]["confirmation_rows"],
        "gate_passed": item59["gate_passed"],
        "confirmation_minimum_improvement_over_every_baseline": item59[
            "splits"
        ]["confirmation"]["improvements"]["minimum"],
    }
    sparc_gate = all(
        sparc["aggregate"]["candidate"]["equal_object_mean_squared_standardized_residual"]
        < sparc["aggregate"][baseline]["equal_object_mean_squared_standardized_residual"]
        for baseline in ("empirical_rar", "newtonian_baryons")
    )
    little_gate = all(
        little["aggregate"]["candidate"]["equal_object_mean_squared_standardized_residual"]
        < little["aggregate"][baseline]["equal_object_mean_squared_standardized_residual"]
        for baseline in ("empirical_rar", "newtonian_baryons")
    )
    gates = {
        "candidate_beats_every_comparator_on_sparc": sparc_gate,
        "candidate_beats_every_comparator_on_little_things": little_gate,
        "item59_xcop_gate_passed": bool(cluster["gate_passed"]),
        "groups_and_transition_objects_evaluable": False,
        "one_formula_and_parameter_set_unchanged": True,
    }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item61-cross-scale-evaluation-1.0",
            "item": 61,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "candidate": config["candidate"],
            "sparc": sparc,
            "little_things": little,
            "xcop": cluster,
            "groups_and_transition_objects": {"evaluable_objects": 0, "gate_passed": False},
            "gates": gates,
            "gate_passed": all(gates.values()),
            "decision": "ITEM61_CROSS_SCALE_GATE_NOT_PASSED_EXACT_PARAMETERIZATION_RETAINED",
            "counterexample_interpretation": {
                "single_counterexample_terminal": False,
                "aggregate_multi_dataset_pattern_withholds_universal_promotion": True,
                "formula_family_pruned": False,
                "item59_cluster_result_rejected": False,
            },
            "next_action": (
                "Keep the boundary/nonlocal family, but require a measurable transition variable "
                "that suppresses its large beta response in disks while preserving the cluster "
                "response; acquire a direct group/transition sample before universal promotion."
            ),
        }
    )


def build_aggregate(root: Path) -> dict[str, Any]:
    config = load_config(root)
    evaluation = build_evaluation(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item61-cross-scale-gate-1.0",
            "goal": "GRAVITY_ROADMAP_ITEM_61_CROSS_SCALE_GATE",
            "item": 61,
            "hypothesis": config["hypothesis"],
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "evaluation": evaluation,
            "decision": evaluation["decision"],
            "gate_passed": evaluation["gate_passed"],
            "result_class": "INCONCLUSIVE",
            "counts": {
                "sparc_galaxies": evaluation["sparc"]["galaxies"],
                "sparc_rows": evaluation["sparc"]["rows"],
                "little_things_galaxies": evaluation["little_things"]["galaxies"],
                "little_things_rows": evaluation["little_things"]["rows"],
                "xcop_clusters": evaluation["xcop"]["clusters"],
                "group_transition_objects": 0,
                "formula_parameter_sets": 1,
                "sealed_sparc_confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "roadmap_item_61_attempt_complete": True,
                "universal_cross_scale_gate_passed": False,
                "item59_cluster_result_rejected": False,
                "formula_or_feature_family_pruned": False,
                "single_counterexample_used_as_veto": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
            },
            "compute": {"backend": "numpy_cpu", "gpu_used": False, "paid_api_cost_usd": 0.0},
            "limitations": [
                "SPARC and LITTLE THINGS response rows were exposed before Item 61, so this is a fixed-candidate transfer diagnostic rather than fresh confirmation.",
                "No authorized direct group/transition sample was available, which independently prevents a full cross-scale pass.",
                "The cluster score retains Item 59 hydrostatic, boundary-pressure, spherical, and nuisance limitations.",
            ],
            "next_action": evaluation["next_action"],
        }
    )


def write_results(root: Path) -> tuple[Path, Path]:
    config = load_config(root)
    evaluation_path = root / str(config["paths"]["evaluation"])
    aggregate_path = root / str(config["paths"]["aggregate"])
    _write_json(evaluation_path, build_evaluation(root))
    _write_json(aggregate_path, build_aggregate(root))
    return evaluation_path, aggregate_path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    evaluation_path = root / str(config["paths"]["evaluation"])
    aggregate_path = root / str(config["paths"]["aggregate"])
    checks = {
        "evaluation": evaluation_path.is_file()
        and _read_json(evaluation_path) == build_evaluation(root),
        "aggregate": aggregate_path.is_file()
        and _read_json(aggregate_path) == build_aggregate(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "evaluate":
        paths = write_results(root)
        print(json.dumps({"paths": [str(path) for path in paths]}, sort_keys=True))
        return 0
    result = replay(root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
