"""Paired, blinded evaluation of whether a system yields more useful creative branches."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

PROTOCOL_PATH = "configs/creativity_ablation_protocol.json"
PROTOCOL_SCHEMA = "invariant-creativity-ablation-protocol-1.0"
OBSERVATION_SCHEMA = "invariant-creativity-ablation-observations-1.0"
RESULT_SCHEMA = "invariant-creativity-ablation-result-1.0"


class CreativityAblationError(ValueError):
    """The paired creativity experiment is unblinded, unmatched, or malformed."""


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def load_protocol(root: Path) -> dict[str, Any]:
    value = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    expected = {
        "arms",
        "baseline_commit",
        "claim_boundary",
        "decision_rule",
        "experiment_id",
        "human_review",
        "primary_metric",
        "resource_matching",
        "schema_version",
        "secondary_metrics",
        "task_protocol",
    }
    if set(value) != expected or value["schema_version"] != PROTOCOL_SCHEMA:
        raise CreativityAblationError("creativity protocol identity changed")
    if value["arms"][:2] != ["baseline", "full_creativity_first"]:
        raise CreativityAblationError("paired primary arms changed")
    if value["task_protocol"]["minimum_paired_tasks"] < 20:
        raise CreativityAblationError("creativity experiment is underpowered by policy")
    if not all(value["resource_matching"].values()):
        raise CreativityAblationError("creativity resources are not fully matched")
    boundary = value["claim_boundary"]
    if any(boundary.get(key) is not False for key in boundary):
        raise CreativityAblationError("creativity claim boundary changed")
    return value


def _useful(idea: Mapping[str, Any], protocol: Mapping[str, Any]) -> bool:
    review_policy = protocol["human_review"]
    reviews = idea.get("human_reviews")
    if not isinstance(reviews, list) or len(reviews) < review_policy["minimum_independent_reviewers"]:
        return False
    reviewer_ids = {review.get("reviewer_id") for review in reviews if isinstance(review, Mapping)}
    if len(reviewer_ids) != len(reviews) or None in reviewer_ids:
        return False
    threshold = review_policy["useful_threshold_each_axis"]
    axes = review_policy["axes"]
    return all(
        isinstance(review, Mapping)
        and all(
            isinstance(review.get(axis), int)
            and not isinstance(review.get(axis), bool)
            and threshold <= review[axis] <= max(review_policy["rating_scale"])
            for axis in axes
        )
        for review in reviews
    )


def _validate_record(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    expected = {
        "arm",
        "blinded_output_id",
        "ideas",
        "resource_budget",
        "task_id",
        "tokens_used",
        "typed_usable_ideas",
    }
    if set(record) != expected or record["arm"] not in protocol["arms"]:
        raise CreativityAblationError("creativity observation record changed")
    if not isinstance(record["ideas"], list) or not isinstance(record["tokens_used"], int):
        raise CreativityAblationError("creativity observation types changed")
    if record["tokens_used"] <= 0 or not 0 <= record["typed_usable_ideas"] <= len(record["ideas"]):
        raise CreativityAblationError("creativity observation counts are invalid")
    budget = record["resource_budget"]
    if not isinstance(budget, Mapping) or set(budget) != {
        "calls",
        "grammar_depth",
        "tokens",
        "verifier_invocations",
        "wall_clock_milliseconds",
    }:
        raise CreativityAblationError("creativity resource budget changed")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in budget.values()
    ):
        raise CreativityAblationError("creativity resource budget is invalid")
    for idea in record["ideas"]:
        if not isinstance(idea, Mapping) or set(idea) != {
            "behavior_sha256",
            "human_reviews",
            "initial_check_status",
            "later_used_as_parent",
            "llm_origin_assessment",
            "prior_art_classification",
            "proof_mechanism_sha256",
            "representation",
            "source_domains",
        }:
            raise CreativityAblationError("creativity idea observation changed")


def _arm_task_metrics(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    ideas = record["ideas"]
    useful = [idea for idea in ideas if _useful(idea, protocol)]
    useful_behaviors = {idea["behavior_sha256"] for idea in useful}
    proof_mechanisms = {idea["proof_mechanism_sha256"] for idea in useful}
    recovered = [
        idea
        for idea in ideas
        if idea["initial_check_status"] in {"blocked", "failed"}
        and idea["later_used_as_parent"] is True
    ]
    return {
        "task_id": record["task_id"],
        "tokens_used": record["tokens_used"],
        "useful_distinct_behavior_branches": len(useful_behaviors),
        "useful_behavior_branches_per_10000_tokens": _fraction_text(
            Fraction(10_000 * len(useful_behaviors), record["tokens_used"])
        ),
        "distinct_proof_mechanisms": len(proof_mechanisms),
        "cross_domain_useful_branches": sum(len(set(idea["source_domains"])) >= 2 for idea in useful),
        "representations": sorted({idea["representation"] for idea in useful}),
        "recovered_initially_failed_or_blocked_branches": len(recovered),
        "typed_usability_rate": _fraction_text(
            Fraction(record["typed_usable_ideas"], max(1, len(ideas)))
        ),
    }


def _one_sided_sign_pvalue(differences: Sequence[Fraction]) -> Fraction:
    nonzero = [item for item in differences if item != 0]
    wins = sum(item > 0 for item in nonzero)
    if not nonzero:
        return Fraction(1)
    return Fraction(
        sum(math.comb(len(nonzero), count) for count in range(wins, len(nonzero) + 1)),
        2 ** len(nonzero),
    )


def score_experiment(
    observations: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if observations.get("schema_version") != OBSERVATION_SCHEMA:
        raise CreativityAblationError("creativity observation schema changed")
    if observations.get("experiment_id") != protocol["experiment_id"]:
        raise CreativityAblationError("creativity observation experiment changed")
    if observations.get("baseline_commit") != protocol["baseline_commit"]:
        raise CreativityAblationError("creativity baseline commit changed")
    treatment_commit = observations.get("treatment_commit")
    if (
        not isinstance(treatment_commit, str)
        or len(treatment_commit) != 40
        or any(character not in "0123456789abcdef" for character in treatment_commit)
        or treatment_commit == protocol["baseline_commit"]
    ):
        raise CreativityAblationError("creativity treatment commit is not independently frozen")
    records = observations.get("records")
    if not isinstance(records, list):
        raise CreativityAblationError("creativity observations are not an array")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CreativityAblationError("creativity record is not an object")
        _validate_record(record, protocol)
        key = (record["task_id"], record["arm"])
        if key in indexed:
            raise CreativityAblationError("duplicate task-arm observation")
        indexed[key] = record
    paired_tasks = sorted(
        task_id
        for task_id, arm in indexed
        if arm == "baseline" and (task_id, "full_creativity_first") in indexed
    )
    if len(paired_tasks) < protocol["task_protocol"]["minimum_paired_tasks"]:
        raise CreativityAblationError("not enough paired tasks for the preregistered decision")
    baseline_metrics = []
    treatment_metrics = []
    differences = []
    for task_id in paired_tasks:
        baseline_record = indexed[(task_id, "baseline")]
        treatment_record = indexed[(task_id, "full_creativity_first")]
        if baseline_record["resource_budget"] != treatment_record["resource_budget"]:
            raise CreativityAblationError("paired task resource budgets differ")
        baseline = _arm_task_metrics(baseline_record, protocol)
        treatment = _arm_task_metrics(treatment_record, protocol)
        baseline_metrics.append(baseline)
        treatment_metrics.append(treatment)
        differences.append(
            Fraction(treatment["useful_behavior_branches_per_10000_tokens"])
            - Fraction(baseline["useful_behavior_branches_per_10000_tokens"])
        )
    baseline_total = sum(
        Fraction(item["useful_behavior_branches_per_10000_tokens"])
        for item in baseline_metrics
    )
    treatment_total = sum(
        Fraction(item["useful_behavior_branches_per_10000_tokens"])
        for item in treatment_metrics
    )
    mean_baseline = baseline_total / len(paired_tasks)
    mean_treatment = treatment_total / len(paired_tasks)
    rule = protocol["decision_rule"]
    required_ratio = Fraction(
        rule["minimum_relative_primary_improvement_numerator"],
        rule["minimum_relative_primary_improvement_denominator"],
    )
    alpha = Fraction(
        rule["one_sided_sign_test_alpha_numerator"],
        rule["one_sided_sign_test_alpha_denominator"],
    )
    relative = (
        Fraction(0)
        if mean_baseline == 0 and mean_treatment == 0
        else Fraction(10**9)
        if mean_baseline == 0
        else (mean_treatment - mean_baseline) / mean_baseline
    )
    pvalue = _one_sided_sign_pvalue(differences)
    baseline_usability = sum(
        Fraction(item["typed_usability_rate"]) for item in baseline_metrics
    ) / len(paired_tasks)
    treatment_usability = sum(
        Fraction(item["typed_usability_rate"]) for item in treatment_metrics
    ) / len(paired_tasks)
    usability_margin = Fraction(
        rule["typed_usability_noninferiority_margin_numerator"],
        rule["typed_usability_noninferiority_margin_denominator"],
    )
    usability_noninferior = treatment_usability + usability_margin >= baseline_usability
    verdict = (
        "MORE_CREATIVE_ON_PREREGISTERED_BOUNDED_PROTOCOL"
        if relative >= required_ratio and pvalue <= alpha and usability_noninferior
        else "NOT_ESTABLISHED_MORE_CREATIVE"
    )
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "paired_tasks": len(paired_tasks),
        "primary_metric": protocol["primary_metric"],
        "baseline_mean": _fraction_text(mean_baseline),
        "treatment_mean": _fraction_text(mean_treatment),
        "relative_improvement": _fraction_text(relative),
        "one_sided_sign_test_pvalue": _fraction_text(pvalue),
        "typed_usability_baseline_mean": _fraction_text(baseline_usability),
        "typed_usability_noninferior": usability_noninferior,
        "typed_usability_treatment_mean": _fraction_text(treatment_usability),
        "verdict": verdict,
        "claims": {
            "literature_novelty_established": False,
            "single_run_used_for_decision": False,
            "unbounded_general_creativity_established": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.root)
    observations = json.loads(args.observations.read_text(encoding="utf-8"))
    result = score_experiment(observations, protocol)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"paired_tasks": result["paired_tasks"], "verdict": result["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
