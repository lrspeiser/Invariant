"""End-to-end fail-closed evaluation and Pareto-ranking boundary for Sigma candidates.

The pipeline deliberately does not generate evidence. It sequences a registered evaluation
ladder, opens soft metrics only for candidates that pass every hard gate, and delegates exact
Pareto explanations to the typed ranking engine. A blocked, rejected, or errored candidate is
retained in the batch ledger with a null front and can never be rescued by a metric.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .candidate_evaluation_ladder import (
    EvaluationLadder,
    evaluate_candidate,
    validate_evaluation_replay,
    validate_evaluation_result,
)
from .candidate_pareto_explanations import (
    MetricReceipt,
    ParetoLimits,
    build_pareto_explanations,
    validate_pareto_result,
)
from .sigma_core import (
    CandidateArtifact,
    DomainPack,
    GateOutcome,
    OutcomeStatus,
    canonical_sha256,
)

RESULT_SCHEMA = "sigma-candidate-discovery-pipeline-1.0"
_CLAIMS = {
    "soft_metrics_opened_before_all_hard_gates_passed": False,
    "failed_hard_gate_compensated_by_soft_metric": False,
    "unranked_candidate_omitted": False,
    "truth_established": False,
    "novelty_established": False,
    "promotion_authorized": False,
}


class CandidateDiscoveryPipelineError(ValueError):
    """The batch crossed its exact evaluation, metric, or ranking boundary."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CandidateDiscoveryPipelineError(f"{label} keys changed")


@dataclass(frozen=True, slots=True)
class DiscoveryPipelineLimits:
    maximum_candidates: int = 128

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_candidates, bool)
            or not isinstance(self.maximum_candidates, int)
            or self.maximum_candidates <= 0
        ):
            raise CandidateDiscoveryPipelineError("maximum_candidates must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"maximum_candidates": self.maximum_candidates}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DiscoveryPipelineLimits:
        _exact_keys(value, {"maximum_candidates"}, "pipeline limits")
        return cls(value["maximum_candidates"])


def _candidates(
    values: Sequence[CandidateArtifact], limits: DiscoveryPipelineLimits
) -> tuple[CandidateArtifact, ...]:
    raw = tuple(values)
    if not raw or len(raw) > limits.maximum_candidates:
        raise CandidateDiscoveryPipelineError("candidate batch is empty or exceeds its cap")
    by_id: dict[str, CandidateArtifact] = {}
    for candidate in raw:
        if not isinstance(candidate, CandidateArtifact):
            raise CandidateDiscoveryPipelineError("batch members must be CandidateArtifacts")
        candidate.validate()
        if candidate.artifact_id in by_id:
            raise CandidateDiscoveryPipelineError("candidate batch contains duplicate artifacts")
        by_id[candidate.artifact_id] = candidate
    return tuple(sorted(by_id.values(), key=lambda item: item.artifact_id))


def _metric_registry(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise CandidateDiscoveryPipelineError("metric registry must be a nonempty mapping")
    rows = tuple(sorted((str(metric_id), str(direction)) for metric_id, direction in value.items()))
    if len(rows) != len(value) or any(
        direction not in {"minimize", "maximize"} for _, direction in rows
    ):
        raise CandidateDiscoveryPipelineError(
            "metric registry is duplicate or has an invalid direction"
        )
    return rows


def _receipts(
    values: Sequence[MetricReceipt],
    eligible: Sequence[CandidateArtifact],
    registry: Sequence[tuple[str, str]],
) -> tuple[MetricReceipt, ...]:
    eligible_refs = {item.artifact_id: item.ref for item in eligible}
    directions = dict(registry)
    rows: dict[tuple[str, str], MetricReceipt] = {}
    for receipt in values:
        if not isinstance(receipt, MetricReceipt):
            raise CandidateDiscoveryPipelineError("soft evidence must use MetricReceipt")
        expected = eligible_refs.get(receipt.candidate.artifact_id)
        if expected is None:
            raise CandidateDiscoveryPipelineError(
                "soft metric supplied for a candidate that did not pass every hard gate"
            )
        if receipt.candidate != expected or directions.get(receipt.metric_id) != receipt.direction:
            raise CandidateDiscoveryPipelineError("metric receipt binding or direction changed")
        key = receipt.candidate.artifact_id, receipt.metric_id
        if key in rows:
            raise CandidateDiscoveryPipelineError("duplicate metric receipt")
        rows[key] = receipt
    expected_keys = {
        (candidate.artifact_id, metric_id) for candidate in eligible for metric_id, _ in registry
    }
    if set(rows) != expected_keys:
        raise CandidateDiscoveryPipelineError("eligible candidates lack complete metric receipts")
    return tuple(rows[key] for key in sorted(rows))


def _body(
    *,
    limits: DiscoveryPipelineLimits,
    pareto_limits: ParetoLimits,
    ladder: EvaluationLadder,
    candidates: Sequence[CandidateArtifact],
    evaluations: Sequence[Mapping[str, Any]],
    receipts: Sequence[MetricReceipt],
    registry: Sequence[tuple[str, str]],
    pareto: Mapping[str, Any] | None,
) -> dict[str, Any]:
    evaluation_by_id = {row["artifact"]["artifact_id"]: row for row in evaluations}
    explanation_by_id = (
        {row["candidate"]["artifact_id"]: row for row in pareto["explanations"]}
        if pareto is not None
        else {}
    )
    rows = []
    statuses = Counter()
    observations_opened = 0
    for candidate in candidates:
        evaluation = evaluation_by_id[candidate.artifact_id]
        status = evaluation["status"]
        statuses[status] += 1
        observations_opened += int(evaluation["observational_phase_opened"])
        explanation = explanation_by_id.get(candidate.artifact_id)
        eligible = evaluation["all_required_gates_passed"] is True
        rows.append(
            {
                "candidate": candidate.ref.to_dict(),
                "evaluation_content_sha256": evaluation["content_sha256"],
                "status": status,
                "all_required_gates_passed": eligible,
                "observational_phase_opened": evaluation["observational_phase_opened"],
                "ranking_eligible": eligible,
                "pareto_front": explanation["pareto_front"] if explanation else None,
                "explanation_sha256": explanation["explanation_sha256"] if explanation else None,
                "exclusion_reason": None if eligible else "did_not_pass_all_required_hard_gates",
                "promotion_authorized": False,
            }
        )
    ranked = sum(row["ranking_eligible"] for row in rows)
    decision = (
        "completed_with_hard_gate_admitted_pareto_candidates"
        if ranked
        else "completed_with_no_candidate_admitted_to_soft_ranking"
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "limits": limits.to_dict(),
        "pareto_limits": pareto_limits.to_dict(),
        "ladder": ladder.to_dict(),
        "metric_registry": [
            {"metric_id": metric_id, "direction": direction} for metric_id, direction in registry
        ],
        "candidates": [item.to_dict() for item in candidates],
        "evaluations": [dict(item) for item in evaluations],
        "metric_receipts": [item.to_dict() for item in receipts],
        "pareto_result": dict(pareto) if pareto is not None else None,
        "candidate_rows": rows,
        "counts": {
            "candidates": len(candidates),
            "evaluation_statuses": {
                status.value: statuses[status.value] for status in OutcomeStatus
            },
            "all_required_gates_passed": ranked,
            "ranked_candidates": ranked,
            "unranked_candidates": len(candidates) - ranked,
            "observational_phase_opened": observations_opened,
            "metric_receipts": len(receipts),
            "pareto_fronts": len(pareto["pareto_fronts"]) if pareto is not None else 0,
        },
        "decision": decision,
        "claims": dict(_CLAIMS),
    }


def run_candidate_discovery_pipeline(
    pack: DomainPack,
    candidates: Sequence[CandidateArtifact],
    ladder: EvaluationLadder,
    metric_receipts: Sequence[MetricReceipt],
    *,
    metric_directions: Mapping[str, str],
    limits: DiscoveryPipelineLimits | None = None,
    pareto_limits: ParetoLimits | None = None,
) -> dict[str, Any]:
    """Evaluate a batch, open metrics only after all gates, then rank exact survivors."""

    limits = DiscoveryPipelineLimits() if limits is None else limits
    pareto_limits = ParetoLimits() if pareto_limits is None else pareto_limits
    if not isinstance(limits, DiscoveryPipelineLimits) or not isinstance(
        pareto_limits, ParetoLimits
    ):
        raise CandidateDiscoveryPipelineError("pipeline and Pareto limits must be typed")
    ordered = _candidates(candidates, limits)
    evaluations = tuple(evaluate_candidate(pack, candidate, ladder) for candidate in ordered)
    eligible = tuple(
        candidate
        for candidate, evaluation in zip(ordered, evaluations, strict=True)
        if evaluation["all_required_gates_passed"] is True
    )
    registry = _metric_registry(metric_directions)
    receipts = _receipts(metric_receipts, eligible, registry)
    pareto = None
    if eligible:
        eligible_ids = {item.artifact_id for item in eligible}
        gates = tuple(
            GateOutcome.from_dict(row)
            for evaluation in evaluations
            if evaluation["artifact"]["artifact_id"] in eligible_ids
            for row in evaluation["gate_outcomes"]
        )
        pareto = build_pareto_explanations(
            eligible,
            gates,
            receipts,
            required_gate_ids=tuple(step.gate_id for step in ladder.steps),
            metric_directions=dict(registry),
            limits=pareto_limits,
        )
    body = _body(
        limits=limits,
        pareto_limits=pareto_limits,
        ladder=ladder,
        candidates=ordered,
        evaluations=evaluations,
        receipts=receipts,
        registry=registry,
        pareto=pareto,
    )
    result = {**body, "content_sha256": canonical_sha256(body)}
    validate_candidate_discovery_result(result)
    return result


def validate_candidate_discovery_result(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "limits",
        "pareto_limits",
        "ladder",
        "metric_registry",
        "candidates",
        "evaluations",
        "metric_receipts",
        "pareto_result",
        "candidate_rows",
        "counts",
        "decision",
        "claims",
        "content_sha256",
    }
    _exact_keys(value, expected, "candidate discovery result")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["schema_version"] != RESULT_SCHEMA or value["content_sha256"] != canonical_sha256(
        unsigned
    ):
        raise CandidateDiscoveryPipelineError("candidate discovery result identity changed")
    for key in (
        "metric_registry",
        "candidates",
        "evaluations",
        "metric_receipts",
        "candidate_rows",
    ):
        if not isinstance(value[key], list):
            raise CandidateDiscoveryPipelineError(f"candidate discovery {key} must be an array")
    limits = DiscoveryPipelineLimits.from_dict(value["limits"])
    pareto_limits = ParetoLimits.from_dict(value["pareto_limits"])
    ladder = EvaluationLadder.from_dict(value["ladder"])
    candidates = _candidates(
        tuple(CandidateArtifact.from_dict(item) for item in value["candidates"]), limits
    )
    evaluations = tuple(value["evaluations"])
    if len(evaluations) != len(candidates):
        raise CandidateDiscoveryPipelineError("evaluation coverage is incomplete")
    for candidate, evaluation in zip(candidates, evaluations, strict=True):
        validate_evaluation_result(evaluation)
        if evaluation["artifact"] != candidate.ref.to_dict():
            raise CandidateDiscoveryPipelineError("evaluation candidate binding changed")
    eligible = tuple(
        candidate
        for candidate, evaluation in zip(candidates, evaluations, strict=True)
        if evaluation["all_required_gates_passed"] is True
    )
    registry_rows = tuple(
        (row["metric_id"], row["direction"])
        for row in value["metric_registry"]
        if isinstance(row, Mapping) and set(row) == {"metric_id", "direction"}
    )
    if len(registry_rows) != len(value["metric_registry"]):
        raise CandidateDiscoveryPipelineError("metric registry row schema changed")
    registry = _metric_registry(dict(registry_rows))
    receipts = _receipts(
        tuple(MetricReceipt.from_dict(item) for item in value["metric_receipts"]),
        eligible,
        registry,
    )
    pareto = value["pareto_result"]
    if eligible:
        if not isinstance(pareto, Mapping):
            raise CandidateDiscoveryPipelineError("eligible candidates lack a Pareto result")
        validate_pareto_result(pareto)
    elif pareto is not None:
        raise CandidateDiscoveryPipelineError("Pareto result exists without eligible candidates")
    expected_body = _body(
        limits=limits,
        pareto_limits=pareto_limits,
        ladder=ladder,
        candidates=candidates,
        evaluations=evaluations,
        receipts=receipts,
        registry=registry,
        pareto=pareto,
    )
    if unsigned != expected_body:
        raise CandidateDiscoveryPipelineError("candidate discovery result boundary changed")


def validate_candidate_discovery_replay(
    value: Mapping[str, Any],
    pack: DomainPack,
    candidates: Sequence[CandidateArtifact],
    metric_receipts: Sequence[MetricReceipt],
) -> None:
    """Replay every hard evaluation and the exact ranking from supplied typed inputs."""

    validate_candidate_discovery_result(value)
    ladder = EvaluationLadder.from_dict(value["ladder"])
    ordered = _candidates(candidates, DiscoveryPipelineLimits.from_dict(value["limits"]))
    for evaluation, candidate in zip(value["evaluations"], ordered, strict=True):
        validate_evaluation_replay(evaluation, pack, candidate)
    rebuilt = run_candidate_discovery_pipeline(
        pack,
        ordered,
        ladder,
        metric_receipts,
        metric_directions={row["metric_id"]: row["direction"] for row in value["metric_registry"]},
        limits=DiscoveryPipelineLimits.from_dict(value["limits"]),
        pareto_limits=ParetoLimits.from_dict(value["pareto_limits"]),
    )
    if dict(value) != rebuilt:
        raise CandidateDiscoveryPipelineError("candidate discovery batch is not replayable")


__all__ = [
    "RESULT_SCHEMA",
    "CandidateDiscoveryPipelineError",
    "DiscoveryPipelineLimits",
    "run_candidate_discovery_pipeline",
    "validate_candidate_discovery_replay",
    "validate_candidate_discovery_result",
]
