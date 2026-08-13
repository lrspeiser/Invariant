"""Eight-seed CPU replay robustness and leave-one-generator-out ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .alpha_operational_rehearsal import (
    run_operational_rehearsal,
    validate_operational_receipt,
)
from .candidate_evaluation_ladder import validate_evaluation_replay
from .candidate_generator_portfolio import (
    build_generator_portfolio,
    validate_generator_portfolio,
)
from .candidate_pareto_explanations import (
    MetricReceipt,
    ParetoLimits,
    build_pareto_explanations,
    validate_pareto_replay,
)
from .generated_candidate_formula_gpu_stress_campaign import (
    validate_campaign as validate_gpu_campaign,
)
from .prospective_blind_cross_generator_tournament import (
    FAMILIES,
    _TournamentPack,
)
from .prospective_blind_cross_generator_tournament import (
    validate_campaign as validate_tournament,
)
from .sigma_core import (
    CandidateArtifact,
    DomainPackDescriptor,
    GateOutcome,
    canonical_sha256,
)

CONFIG_SCHEMA = "sigma-prospective-tournament-robustness-ablation-config-1.0"
RESULT_SCHEMA = "sigma-prospective-tournament-robustness-ablation-result-1.0"
CAMPAIGN_ID = "prospective-tournament-robustness-ablation-001"
CONFIG_PATH = "configs/prospective_tournament_robustness_ablation.json"
SOURCE_PATH = "src/sigma_theory_compiler/prospective_tournament_robustness_ablation.py"
TEST_PATH = "tests/test_prospective_tournament_robustness_ablation.py"
OUTPUT_PATH = "runs/math/prospective-tournament-robustness-ablation/campaign.json"
TOURNAMENT_PATH = "runs/math/prospective-blind-cross-generator-tournament/campaign.json"
GPU_RECEIPT_PATH = "runs/engine/generated-candidate-formula-gpu-stress-campaign.json"
GPU_CONFIG_PATH = "configs/generated_candidate_formula_gpu_stress_campaign.json"
OPERATIONAL_SOURCE_PATH = "src/sigma_theory_compiler/alpha_operational_rehearsal.py"
SEEDS = tuple(f"robustness-order-{index:02d}-20260813" for index in range(1, 9))
CLAIMS = {
    "eight_seed_cpu_exact_replay_completed": True,
    "candidate_overlap_measured": True,
    "gate_outcome_stability_measured": True,
    "pareto_front_stability_measured": True,
    "all_seven_leave_one_generator_out_ablations_completed": True,
    "historical_gpu_control_validated": True,
    "tournament_gpu_replay_performed": False,
    "operational_rehearsal_executed": False,
    "stability_establishes_truth": False,
    "stability_establishes_novelty": False,
    "ablation_authorizes_promotion": False,
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("path is not a portable relative path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("path escapes project root") from error
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("robustness input must be an object")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def _expected_config() -> dict[str, Any]:
    return {
        "budgets": {
            "cpu_exact_replays": 8,
            "leave_one_family_out_ablations": 7,
            "maximum_candidates_per_replay": 21,
            "maximum_evaluation_replays": 168,
            "maximum_pareto_recomputations": 32,
        },
        "campaign_id": CAMPAIGN_ID,
        "inputs": {
            "gpu_stress_config": GPU_CONFIG_PATH,
            "gpu_stress_receipt": GPU_RECEIPT_PATH,
            "operational_rehearsal_source": OPERATIONAL_SOURCE_PATH,
            "tournament_receipt": TOURNAMENT_PATH,
        },
        "policies": {
            "gpu_runtime_access": "forbidden_historical_receipt_only",
            "live_sqlite_access": "forbidden",
            "network_access": "forbidden",
            "operational_rehearsal_execution": "forbidden_interface_binding_only",
            "process_control": "forbidden",
            "stability_establishes_truth": False,
        },
        "schema_version": CONFIG_SCHEMA,
        "seeds": list(SEEDS),
    }


def _load_config(root: Path, config_path: Path | None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ValueError("robustness preregistration path changed")
    value = _load_json(path)
    if value != _expected_config():
        raise ValueError("robustness preregistration changed")
    return value


def _candidate_rows(
    tournament: Mapping[str, Any],
) -> list[tuple[str, str, CandidateArtifact]]:
    rows = []
    for world in tournament["world_results"]:
        candidates = {
            row["artifact_id"]: CandidateArtifact.from_dict(row) for row in world["candidates"]
        }
        for binding in world["family_bindings"]:
            rows.append(
                (
                    world["world_id"],
                    binding["family"],
                    candidates[binding["candidate"]["artifact_id"]],
                )
            )
    if len(rows) != 21 or {(world, family) for world, family, _ in rows} != {
        (world["world_id"], family) for world in tournament["world_results"] for family in FAMILIES
    }:
        raise ValueError("sealed tournament candidate inventory changed")
    return rows


def _permutation_key(seed: str, world_id: str, candidate: CandidateArtifact) -> str:
    return hashlib.sha256(f"{seed}|{world_id}|{candidate.artifact_id}".encode()).hexdigest()


def _front_assignment(pareto: Mapping[str, Any] | None) -> dict[str, int]:
    if pareto is None:
        return {}
    return {
        row["artifact_id"]: index
        for index, front in enumerate(pareto["pareto_fronts"])
        for row in front
    }


def _world_components(
    world: Mapping[str, Any],
) -> tuple[
    _TournamentPack,
    dict[str, CandidateArtifact],
    dict[str, dict[str, Any]],
    list[MetricReceipt],
]:
    descriptor = DomainPackDescriptor.from_dict(world["domain_pack"])
    pack = _TournamentPack(descriptor, world["unsealed_target"]["hypothesis"])
    candidates_by_id = {
        row["artifact_id"]: CandidateArtifact.from_dict(row) for row in world["candidates"]
    }
    candidates = {
        binding["family"]: candidates_by_id[binding["candidate"]["artifact_id"]]
        for binding in world["family_bindings"]
    }
    evaluations = dict(world["evaluations"])
    metrics = [MetricReceipt.from_dict(row) for row in world["metric_receipts"]]
    return pack, candidates, evaluations, metrics


def _eligible_gates(
    evaluations: Mapping[str, Mapping[str, Any]], eligible_families: Sequence[str]
) -> list[GateOutcome]:
    return [
        GateOutcome.from_dict(row)
        for family in eligible_families
        for row in evaluations[family]["gate_outcomes"]
    ]


def _recompute_pareto(
    candidates: Sequence[CandidateArtifact],
    gates: Sequence[GateOutcome],
    metrics: Sequence[MetricReceipt],
) -> dict[str, Any]:
    directions = {"lineage_inputs": "maximize", "representation_bytes": "minimize"}
    limits = ParetoLimits(7, 2, 2, 512)
    result = build_pareto_explanations(
        candidates,
        gates,
        metrics,
        required_gate_ids=("hard_exact", "hard_holdout"),
        metric_directions=directions,
        limits=limits,
    )
    validate_pareto_replay(
        result,
        candidates,
        gates,
        metrics,
        required_gate_ids=("hard_exact", "hard_holdout"),
        metric_directions=directions,
        limits=limits,
    )
    return result


def _baseline_gate_map(tournament: Mapping[str, Any]) -> dict[str, str]:
    return {
        f"{world['world_id']}|{family}|{gate['gate_id']}": gate["status"]
        for world in tournament["world_results"]
        for family, evaluation in world["evaluations"].items()
        for gate in evaluation["gate_outcomes"]
    }


def _baseline_front_map(tournament: Mapping[str, Any]) -> dict[str, int]:
    return {
        f"{world['world_id']}|{artifact_id}": front
        for world in tournament["world_results"]
        for artifact_id, front in _front_assignment(world["pareto"]).items()
    }


def _seed_replay(
    tournament: Mapping[str, Any], seed: str, baseline_candidates: set[str]
) -> dict[str, Any]:
    ordered = sorted(
        _candidate_rows(tournament), key=lambda row: _permutation_key(seed, row[0], row[2])
    )
    replayed_gate_map: dict[str, str] = {}
    replayed_front_map: dict[str, int] = {}
    pareto_recomputations = 0
    for world in tournament["world_results"]:
        pack, candidates, evaluations, metrics = _world_components(world)
        for world_id, family, candidate in ordered:
            if world_id != world["world_id"]:
                continue
            evaluation = evaluations[family]
            validate_evaluation_replay(evaluation, pack, candidate)
            replayed_gate_map.update(
                {
                    f"{world_id}|{family}|{gate['gate_id']}": gate["status"]
                    for gate in evaluation["gate_outcomes"]
                }
            )
        eligible = [
            family
            for world_id, family, _ in ordered
            if world_id == world["world_id"] and family in world["pareto_eligible_families"]
        ]
        if eligible:
            eligible_candidates = [candidates[family] for family in eligible]
            gates = _eligible_gates(evaluations, eligible)
            eligible_ids = {candidate.artifact_id for candidate in eligible_candidates}
            eligible_metrics = [row for row in metrics if row.candidate.artifact_id in eligible_ids]
            recomputed = _recompute_pareto(eligible_candidates, gates, eligible_metrics)
            pareto_recomputations += 1
            if recomputed != world["pareto"]:
                raise ValueError(f"Pareto replay mismatch at seed {seed}")
            replayed_front_map.update(
                {
                    f"{world['world_id']}|{artifact_id}": front
                    for artifact_id, front in _front_assignment(recomputed).items()
                }
            )
        elif world["pareto"] is not None:
            raise ValueError("sealed world has Pareto output without eligible candidates")
    candidate_ids = {candidate.artifact_id for _, _, candidate in ordered}
    if candidate_ids != baseline_candidates:
        raise ValueError(f"candidate set replay mismatch at seed {seed}")
    baseline_gates = _baseline_gate_map(tournament)
    baseline_fronts = _baseline_front_map(tournament)
    if replayed_gate_map != baseline_gates:
        raise ValueError(f"gate replay mismatch at seed {seed}")
    if replayed_front_map != baseline_fronts:
        raise ValueError(f"front replay mismatch at seed {seed}")
    return {
        "seed": seed,
        "candidate_order": [
            {"world_id": world_id, "family": family, "candidate": candidate.ref.to_dict()}
            for world_id, family, candidate in ordered
        ],
        "candidate_overlap": {
            "intersection": len(candidate_ids & baseline_candidates),
            "union": len(candidate_ids | baseline_candidates),
            "jaccard": {"numerator": 1, "denominator": 1},
        },
        "evaluation_replays": len(ordered),
        "gate_outcomes_compared": len(replayed_gate_map),
        "gate_status_sha256": canonical_sha256(replayed_gate_map),
        "gate_outcomes_stable": True,
        "pareto_recomputations": pareto_recomputations,
        "front_assignment_sha256": canonical_sha256(replayed_front_map),
        "fronts_stable": True,
    }


def _ablation(tournament: Mapping[str, Any], removed_family: str) -> dict[str, Any]:
    world_effects = []
    total_remaining_eligible = 0
    total_front_changes = 0
    pass_to_reject = 0
    for world in tournament["world_results"]:
        _, candidates, evaluations, metrics = _world_components(world)
        baseline_eligible = list(world["pareto_eligible_families"])
        remaining_eligible = [family for family in baseline_eligible if family != removed_family]
        total_remaining_eligible += len(remaining_eligible)
        recomputed = None
        if remaining_eligible:
            remaining_candidates = [candidates[family] for family in remaining_eligible]
            gates = _eligible_gates(evaluations, remaining_eligible)
            remaining_ids = {candidate.artifact_id for candidate in remaining_candidates}
            remaining_metrics = [
                row for row in metrics if row.candidate.artifact_id in remaining_ids
            ]
            recomputed = _recompute_pareto(remaining_candidates, gates, remaining_metrics)
            if removed_family not in baseline_eligible and recomputed != world["pareto"]:
                raise ValueError("non-eligible family ablation changed a Pareto result")
        before_fronts = _front_assignment(world["pareto"])
        after_fronts = _front_assignment(recomputed)
        changed_ids = sorted(
            artifact_id
            for artifact_id in set(before_fronts) | set(after_fronts)
            if before_fronts.get(artifact_id) != after_fronts.get(artifact_id)
        )
        total_front_changes += len(changed_ids)
        before_decision = "pass" if baseline_eligible else "reject"
        after_decision = "pass" if remaining_eligible else "reject"
        pass_to_reject += before_decision == "pass" and after_decision == "reject"
        world_effects.append(
            {
                "world_id": world["world_id"],
                "decision_before": before_decision,
                "decision_after": after_decision,
                "eligible_families_before": baseline_eligible,
                "eligible_families_after": remaining_eligible,
                "front_assignment_changes": changed_ids,
                "pareto_content_sha256_after": None
                if recomputed is None
                else recomputed["content_sha256"],
            }
        )
    return {
        "removed_family": removed_family,
        "remaining_candidate_count": 18,
        "candidate_overlap": {
            "intersection": 18,
            "union": 21,
            "jaccard": {"numerator": 6, "denominator": 7},
        },
        "remaining_pareto_eligible_candidates": total_remaining_eligible,
        "pareto_recomputations": sum(bool(row["eligible_families_after"]) for row in world_effects),
        "world_pass_to_reject_count": pass_to_reject,
        "front_assignment_change_count": total_front_changes,
        "world_effects": world_effects,
    }


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = _load_config(root, config_path)
    tournament_path = _resolve(root, TOURNAMENT_PATH)
    gpu_receipt_path = _resolve(root, GPU_RECEIPT_PATH)
    tournament = _load_json(tournament_path)
    gpu = _load_json(gpu_receipt_path)
    validate_tournament(tournament, root)
    validate_gpu_campaign(gpu, _resolve(root, GPU_CONFIG_PATH))
    portfolio = build_generator_portfolio()
    validate_generator_portfolio(portfolio)
    if not callable(run_operational_rehearsal) or not callable(validate_operational_receipt):
        raise TypeError("operational rehearsal interface changed")
    baseline_candidates = {candidate.artifact_id for _, _, candidate in _candidate_rows(tournament)}
    seed_replays = [_seed_replay(tournament, seed, baseline_candidates) for seed in config["seeds"]]
    ablations = [_ablation(tournament, family) for family in FAMILIES]
    gpu_summary = {
        "candidate_count": gpu["counts"]["candidate_count"],
        "comparison_count": gpu["gpu_cpu_comparison"]["comparison_count"],
        "exact_rational_checks": gpu["counts"]["cpu_exact_rational_crosschecks"],
        "violations": gpu["gpu_cpu_comparison"]["violating_point_count"],
        "within_bounds": gpu["gpu_cpu_comparison"]["within_bounds"],
        "role": "historical_synthetic_backend_control_not_tournament_gpu_replay",
    }
    if gpu_summary != {
        "candidate_count": 163,
        "comparison_count": 5_341_184,
        "exact_rational_checks": 5_216,
        "violations": 0,
        "within_bounds": True,
        "role": "historical_synthetic_backend_control_not_tournament_gpu_replay",
    }:
        raise ValueError("historical GPU control changed")
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_eight_seed_exact_replay_and_seven_family_ablation",
        "preregistration": config,
        "tournament_binding": {
            "path": TOURNAMENT_PATH,
            "file_sha256": _file_sha(tournament_path),
            "content_sha256": tournament["content_sha256"],
        },
        "portfolio": portfolio,
        "cpu_exact_replays": seed_replays,
        "ablations": ablations,
        "gpu_backend_binding": {
            "path": GPU_RECEIPT_PATH,
            "file_sha256": _file_sha(gpu_receipt_path),
            "content_sha256": gpu["content_sha256"],
            "summary": gpu_summary,
            "tournament_gpu_replay_performed": False,
        },
        "operational_interface_binding": {
            "path": OPERATIONAL_SOURCE_PATH,
            "file_sha256": _file_sha(_resolve(root, OPERATIONAL_SOURCE_PATH)),
            "run_callable": (
                f"{run_operational_rehearsal.__module__}:{run_operational_rehearsal.__name__}"
            ),
            "validator_callable": (
                f"{validate_operational_receipt.__module__}:{validate_operational_receipt.__name__}"
            ),
            "executed": False,
        },
        "counts": {
            "registered_seeds": 8,
            "cpu_replay_passes": 8,
            "candidate_evaluation_replays": sum(row["evaluation_replays"] for row in seed_replays),
            "gate_outcome_comparisons": sum(row["gate_outcomes_compared"] for row in seed_replays),
            "pareto_recomputations": sum(row["pareto_recomputations"] for row in seed_replays)
            + sum(row["pareto_recomputations"] for row in ablations),
            "leave_one_family_out_ablations": 7,
            "world_pass_to_reject_ablation_changes": sum(
                row["world_pass_to_reject_count"] for row in ablations
            ),
            "tournament_gpu_replays": 0,
            "operational_rehearsal_executions": 0,
        },
        "claims": dict(CLAIMS),
        "scope": (
            "order-perturbed CPU replay robustness of one sealed prospective tournament plus "
            "leave-one-generator-out sensitivity; historical GPU evidence is separate, and "
            "stability or ablation behavior establishes no truth, novelty, or promotion"
        ),
        "next_gate": (
            "repeat_the_preregistered_tournament_on_independently_authored_external_worlds_"
            "and_only_then_compare_cross_seed_discovery_rates"
        ),
        "source_bindings": {
            label: {"path": path, "file_sha256": _file_sha(_resolve(root, path))}
            for label, path in (
                ("config", CONFIG_PATH),
                ("source", SOURCE_PATH),
                ("test", TEST_PATH),
                ("tournament", TOURNAMENT_PATH),
                ("gpu_config", GPU_CONFIG_PATH),
                ("gpu_receipt", GPU_RECEIPT_PATH),
                ("operational_source", OPERATIONAL_SOURCE_PATH),
            )
        },
    }
    return _seal(body)


def validate_campaign(
    value: Mapping[str, Any], root: Path, config_path: Path | None = None
) -> None:
    if value.get("schema_version") != RESULT_SCHEMA or value.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("robustness campaign identity changed")
    if value.get("content_sha256") != canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    ):
        raise ValueError("robustness campaign self-seal changed")
    if value.get("claims") != CLAIMS:
        raise ValueError("robustness campaign claim boundary changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("robustness campaign exact replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable robustness artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    result = run(root, _resolve(root, arguments.config))
    _write_immutable(_resolve(root, arguments.output), result)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
