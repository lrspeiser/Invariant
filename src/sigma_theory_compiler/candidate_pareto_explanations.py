"""Fail-closed hard-gate partitioning and exact Pareto explanations for Sigma candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .sigma_core import (
    ArtifactRef,
    CandidateArtifact,
    GateOutcome,
    OutcomeStatus,
    canonical_sha256,
)

RESULT_SCHEMA = "sigma-candidate-pareto-explanations-1.0"
METRIC_RECEIPT_SCHEMA = "sigma-exact-metric-receipt-1.0"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIRECTIONS = {"maximize", "minimize"}
_STATUS_RANK = {
    OutcomeStatus.PASS: 0,
    OutcomeStatus.BLOCK: 1,
    OutcomeStatus.REJECT: 2,
    OutcomeStatus.ERROR: 3,
}
_CLAIM_BOUNDARY = {
    "truth_established": False,
    "equivalence_established": False,
    "novelty_established": False,
    "absence_established": False,
    "promotion_authorized": False,
}


class ParetoExplanationError(ValueError):
    """An input or result crossed the registered Pareto explanation boundary."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ParetoExplanationError(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ParetoExplanationError(f"{label} is not a registered identifier")
    return value


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ParetoExplanationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_fraction(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ParetoExplanationError("metric value must be an exact integer or Fraction")
    return Fraction(value)


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction_from_data(value: Mapping[str, Any]) -> Fraction:
    _exact_keys(value, {"numerator", "denominator"}, "metric value")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ParetoExplanationError("metric value must be a normalized exact rational")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise ParetoExplanationError("metric value is not canonically normalized")
    return result


@dataclass(frozen=True, slots=True)
class MetricReceipt:
    candidate: ArtifactRef
    metric_id: str
    direction: str
    value: Fraction
    evidence_sha256: str
    receipt_sha256: str
    schema_version: str = METRIC_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ArtifactRef):
            raise ParetoExplanationError("metric candidate must be a Sigma Core ArtifactRef")
        _identifier(self.metric_id, "metric_id")
        if self.direction not in _DIRECTIONS:
            raise ParetoExplanationError("metric direction must be maximize or minimize")
        exact = _exact_fraction(self.value)
        object.__setattr__(self, "value", exact)
        _digest(self.evidence_sha256, "metric evidence_sha256")
        _digest(self.receipt_sha256, "metric receipt_sha256")
        if self.schema_version != METRIC_RECEIPT_SCHEMA:
            raise ParetoExplanationError("metric receipt schema_version changed")
        if self.receipt_sha256 != canonical_sha256(self._body()):
            raise ParetoExplanationError("metric receipt canonical identity changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_dict(),
            "metric_id": self.metric_id,
            "direction": self.direction,
            "value": _fraction_data(self.value),
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def create(
        cls,
        candidate: ArtifactRef,
        metric_id: str,
        direction: str,
        value: int | Fraction,
        evidence_sha256: str,
    ) -> MetricReceipt:
        exact = _exact_fraction(value)
        body = {
            "schema_version": METRIC_RECEIPT_SCHEMA,
            "candidate": candidate.to_dict(),
            "metric_id": metric_id,
            "direction": direction,
            "value": _fraction_data(exact),
            "evidence_sha256": evidence_sha256,
        }
        return cls(
            candidate,
            metric_id,
            direction,
            exact,
            evidence_sha256,
            canonical_sha256(body),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetricReceipt:
        _exact_keys(
            value,
            {
                "schema_version",
                "candidate",
                "metric_id",
                "direction",
                "value",
                "evidence_sha256",
                "receipt_sha256",
            },
            "metric receipt",
        )
        return cls(
            ArtifactRef.from_dict(value["candidate"]),
            value["metric_id"],
            value["direction"],
            _fraction_from_data(value["value"]),
            value["evidence_sha256"],
            value["receipt_sha256"],
            value["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ParetoLimits:
    maximum_candidates: int = 128
    maximum_hard_gates: int = 16
    maximum_metrics: int = 16
    maximum_work_units: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "maximum_candidates",
            "maximum_hard_gates",
            "maximum_metrics",
            "maximum_work_units",
        ):
            item = getattr(self, name)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ParetoExplanationError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_candidates": self.maximum_candidates,
            "maximum_hard_gates": self.maximum_hard_gates,
            "maximum_metrics": self.maximum_metrics,
            "maximum_work_units": self.maximum_work_units,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParetoLimits:
        _exact_keys(
            value,
            {
                "maximum_candidates",
                "maximum_hard_gates",
                "maximum_metrics",
                "maximum_work_units",
            },
            "Pareto limits",
        )
        return cls(**value)


def _normalize_identifiers(values: Sequence[str], label: str, maximum: int) -> tuple[str, ...]:
    raw = tuple(values)
    if not raw:
        raise ParetoExplanationError(f"{label} must be nonempty")
    normalized = tuple(sorted({_identifier(item, label) for item in raw}))
    if len(normalized) > maximum:
        raise ParetoExplanationError(f"{label} exceed their registered maximum")
    return normalized


def _normalize_directions(
    value: Mapping[str, str], limits: ParetoLimits
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ParetoExplanationError("metric_directions must be a nonempty mapping")
    rows = []
    for metric_id, direction in value.items():
        _identifier(metric_id, "metric_id")
        if direction not in _DIRECTIONS:
            raise ParetoExplanationError("metric direction must be maximize or minimize")
        rows.append((metric_id, direction))
    result = tuple(sorted(rows))
    if len(result) > limits.maximum_metrics:
        raise ParetoExplanationError("metrics exceed maximum_metrics")
    return result


def _normalize_candidates(
    candidates: Sequence[CandidateArtifact], limits: ParetoLimits
) -> tuple[CandidateArtifact, ...]:
    raw = tuple(candidates)
    if not raw:
        raise ParetoExplanationError("at least one candidate is required")
    if len(raw) > limits.maximum_candidates:
        raise ParetoExplanationError("candidate inputs exceed maximum_candidates")
    unique: dict[str, CandidateArtifact] = {}
    for candidate in raw:
        if not isinstance(candidate, CandidateArtifact):
            raise ParetoExplanationError("candidates must be Sigma Core CandidateArtifacts")
        candidate.validate()
        previous = unique.get(candidate.content_sha256)
        if previous is not None and previous.to_dict() != candidate.to_dict():
            raise ParetoExplanationError("candidate digest collision changed canonical body")
        unique[candidate.content_sha256] = candidate
    return tuple(sorted(unique.values(), key=lambda item: item.artifact_id))


def _normalize_gate_outcomes(
    outcomes: Sequence[GateOutcome],
    candidates: Sequence[CandidateArtifact],
    required_gate_ids: Sequence[str],
) -> dict[tuple[str, str], GateOutcome]:
    candidate_refs = {item.artifact_id: item.ref for item in candidates}
    required = set(required_gate_ids)
    result: dict[tuple[str, str], GateOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, GateOutcome):
            raise ParetoExplanationError("gate outcomes must be Sigma Core GateOutcomes")
        if outcome.gate_id not in required:
            raise ParetoExplanationError("gate outcome used an unregistered required gate")
        expected_ref = candidate_refs.get(outcome.artifact.artifact_id)
        if expected_ref is None or outcome.artifact != expected_ref:
            raise ParetoExplanationError("gate outcome is not bound to an input candidate")
        key = outcome.artifact.artifact_id, outcome.gate_id
        previous = result.get(key)
        if previous is not None and previous.to_dict() != outcome.to_dict():
            raise ParetoExplanationError("conflicting gate outcomes bind one candidate and gate")
        result[key] = outcome
    expected = {
        (candidate.artifact_id, gate_id)
        for candidate in candidates
        for gate_id in required_gate_ids
    }
    if set(result) != expected:
        raise ParetoExplanationError("required hard gate outcome coverage is incomplete")
    return result


def _normalize_metric_receipts(
    receipts: Sequence[MetricReceipt],
    candidates: Sequence[CandidateArtifact],
    directions: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], MetricReceipt]:
    candidate_refs = {item.artifact_id: item.ref for item in candidates}
    registered = dict(directions)
    result: dict[tuple[str, str], MetricReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, MetricReceipt):
            raise ParetoExplanationError("metrics must be typed MetricReceipts")
        expected_ref = candidate_refs.get(receipt.candidate.artifact_id)
        if expected_ref is None or receipt.candidate != expected_ref:
            raise ParetoExplanationError("metric receipt is not bound to an input candidate")
        if registered.get(receipt.metric_id) != receipt.direction:
            raise ParetoExplanationError("metric receipt direction or registration changed")
        key = receipt.candidate.artifact_id, receipt.metric_id
        previous = result.get(key)
        if previous is not None and previous.to_dict() != receipt.to_dict():
            raise ParetoExplanationError(
                "conflicting metric receipts bind one candidate and metric"
            )
        result[key] = receipt
    expected = {
        (candidate.artifact_id, metric_id)
        for candidate in candidates
        for metric_id, _ in directions
    }
    if set(result) != expected:
        raise ParetoExplanationError("exact metric receipt coverage is incomplete")
    return result


def _relation(left: Fraction, right: Fraction, direction: str) -> str:
    if left == right:
        return "equal"
    better = left > right if direction == "maximize" else left < right
    return "better" if better else "worse"


def _pairwise_dominance(
    eligible: Sequence[CandidateArtifact],
    directions: Sequence[tuple[str, str]],
    metrics: Mapping[tuple[str, str], MetricReceipt],
) -> tuple[
    dict[str, set[str]],
    dict[tuple[str, str], list[dict[str, Any]]],
    int,
]:
    dominates = {candidate.artifact_id: set() for candidate in eligible}
    comparisons: dict[tuple[str, str], list[dict[str, Any]]] = {}
    work_units = 0
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            left_rows = []
            right_rows = []
            left_relations = []
            right_relations = []
            for metric_id, direction in directions:
                left_receipt = metrics[left.artifact_id, metric_id]
                right_receipt = metrics[right.artifact_id, metric_id]
                left_relation = _relation(left_receipt.value, right_receipt.value, direction)
                right_relation = _relation(right_receipt.value, left_receipt.value, direction)
                left_relations.append(left_relation)
                right_relations.append(right_relation)
                left_rows.append(
                    {
                        "metric_id": metric_id,
                        "direction": direction,
                        "self_value": _fraction_data(left_receipt.value),
                        "other_value": _fraction_data(right_receipt.value),
                        "self_relation": left_relation,
                        "self_receipt_sha256": left_receipt.receipt_sha256,
                        "other_receipt_sha256": right_receipt.receipt_sha256,
                    }
                )
                right_rows.append(
                    {
                        "metric_id": metric_id,
                        "direction": direction,
                        "self_value": _fraction_data(right_receipt.value),
                        "other_value": _fraction_data(left_receipt.value),
                        "self_relation": right_relation,
                        "self_receipt_sha256": right_receipt.receipt_sha256,
                        "other_receipt_sha256": left_receipt.receipt_sha256,
                    }
                )
                work_units += 1
            comparisons[left.artifact_id, right.artifact_id] = left_rows
            comparisons[right.artifact_id, left.artifact_id] = right_rows
            if "worse" not in left_relations and "better" in left_relations:
                dominates[left.artifact_id].add(right.artifact_id)
            if "worse" not in right_relations and "better" in right_relations:
                dominates[right.artifact_id].add(left.artifact_id)
    return dominates, comparisons, work_units


def _fronts(
    eligible: Sequence[CandidateArtifact], dominates: Mapping[str, set[str]]
) -> tuple[list[list[str]], dict[str, int]]:
    remaining = {candidate.artifact_id for candidate in eligible}
    rows: list[list[str]] = []
    front_by_candidate: dict[str, int] = {}
    while remaining:
        front = sorted(
            candidate_id
            for candidate_id in remaining
            if not any(candidate_id in dominates[other] for other in remaining)
        )
        if not front:
            raise ParetoExplanationError("Pareto dominance graph unexpectedly contains a cycle")
        rows.append(front)
        for candidate_id in front:
            front_by_candidate[candidate_id] = len(rows)
        remaining.difference_update(front)
    return rows, front_by_candidate


def _build_result(
    candidates_input: Sequence[CandidateArtifact],
    gate_outcomes_input: Sequence[GateOutcome],
    metric_receipts_input: Sequence[MetricReceipt],
    *,
    required_gate_ids: Sequence[str],
    metric_directions: Mapping[str, str],
    limits: ParetoLimits,
    validate_output: bool,
) -> dict[str, Any]:
    candidates = _normalize_candidates(candidates_input, limits)
    gates = _normalize_identifiers(
        required_gate_ids, "required_gate_ids", limits.maximum_hard_gates
    )
    directions = _normalize_directions(metric_directions, limits)
    outcomes = _normalize_gate_outcomes(gate_outcomes_input, candidates, gates)
    metrics = _normalize_metric_receipts(metric_receipts_input, candidates, directions)

    eligible = tuple(
        candidate
        for candidate in candidates
        if all(
            outcomes[candidate.artifact_id, gate_id].status is OutcomeStatus.PASS
            for gate_id in gates
        )
    )
    required_work = len(eligible) * (len(eligible) - 1) // 2 * len(directions)
    if required_work > limits.maximum_work_units:
        raise ParetoExplanationError("exact Pareto comparisons exceed maximum_work_units")
    dominates, comparisons, work_units = _pairwise_dominance(eligible, directions, metrics)
    fronts, front_by_candidate = _fronts(eligible, dominates)
    by_id = {candidate.artifact_id: candidate for candidate in candidates}

    explanations = []
    for candidate in candidates:
        hard_rows = []
        hard_key = []
        for gate_id in gates:
            outcome = outcomes[candidate.artifact_id, gate_id]
            hard_key.append(_STATUS_RANK[outcome.status])
            hard_rows.append(
                {
                    "gate_id": gate_id,
                    "status": outcome.status.value,
                    "outcome_sha256": outcome.outcome_sha256,
                    "check_details_sha256s": [check.details_sha256 for check in outcome.checks],
                    "source_evidence_sha256s": [item.file_sha256 for item in outcome.evidence],
                    "reason_codes": list(outcome.reason_codes),
                }
            )
        metric_rows = [
            {
                "metric_id": metric_id,
                "direction": direction,
                "value": _fraction_data(metrics[candidate.artifact_id, metric_id].value),
                "evidence_sha256": metrics[candidate.artifact_id, metric_id].evidence_sha256,
                "receipt_sha256": metrics[candidate.artifact_id, metric_id].receipt_sha256,
            }
            for metric_id, direction in directions
        ]
        eligible_for_soft_metrics = candidate.artifact_id in front_by_candidate
        dominators = (
            sorted(
                other.artifact_id
                for other in eligible
                if candidate.artifact_id in dominates[other.artifact_id]
            )
            if eligible_for_soft_metrics
            else []
        )
        dominance_evidence = [
            {
                "dominating_candidate": by_id[other_id].ref.to_dict(),
                "metric_comparisons": comparisons[candidate.artifact_id, other_id],
            }
            for other_id in dominators
        ]
        evidence_binding = {
            "hard_gate_outcome_sha256s": [row["outcome_sha256"] for row in hard_rows],
            "metric_receipt_sha256s": [row["receipt_sha256"] for row in metric_rows],
        }
        body = {
            "candidate": candidate.ref.to_dict(),
            "hard_gate_key": hard_key,
            "hard_gate_outcomes": hard_rows,
            "soft_metric_eligible": eligible_for_soft_metrics,
            "pareto_front": front_by_candidate.get(candidate.artifact_id),
            "metric_values": metric_rows,
            "dominated_by": [by_id[item].ref.to_dict() for item in dominators],
            "dominance_evidence": dominance_evidence,
            "evidence_binding_sha256": canonical_sha256(evidence_binding),
            "promotion_authorized": False,
        }
        explanations.append({**body, "explanation_sha256": canonical_sha256(body)})

    explanation_by_id = {row["candidate"]["artifact_id"]: row for row in explanations}
    display_order = sorted(
        (candidate.artifact_id for candidate in candidates),
        key=lambda candidate_id: (
            0 if explanation_by_id[candidate_id]["soft_metric_eligible"] else 1,
            explanation_by_id[candidate_id]["pareto_front"] or 0,
            tuple(explanation_by_id[candidate_id]["hard_gate_key"]),
            candidate_id,
        ),
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "limits": limits.to_dict(),
        "required_gate_ids": list(gates),
        "metric_registry": [
            {"metric_id": metric_id, "direction": direction} for metric_id, direction in directions
        ],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "gate_outcomes": [
            outcomes[key].to_dict() for key in sorted(outcomes, key=lambda item: (item[0], item[1]))
        ],
        "metric_receipts": [
            metrics[key].to_dict() for key in sorted(metrics, key=lambda item: (item[0], item[1]))
        ],
        "pareto_fronts": [
            [by_id[candidate_id].ref.to_dict() for candidate_id in front] for front in fronts
        ],
        "explanations": explanations,
        "display_order": [by_id[candidate_id].ref.to_dict() for candidate_id in display_order],
        "counts": {
            "candidates": len(candidates),
            "hard_gate_eligible": len(eligible),
            "hard_gate_ineligible": len(candidates) - len(eligible),
            "required_gates": len(gates),
            "metrics": len(directions),
            "pareto_fronts": len(fronts),
            "work_units_consumed": work_units,
        },
        "decision": "completed_hard_gates_then_exact_pareto",
        "claims": dict(_CLAIM_BOUNDARY),
    }
    result["content_sha256"] = canonical_sha256(result)
    if validate_output:
        validate_pareto_result(result)
    return result


def build_pareto_explanations(
    candidates: Sequence[CandidateArtifact],
    gate_outcomes: Sequence[GateOutcome],
    metric_receipts: Sequence[MetricReceipt],
    *,
    required_gate_ids: Sequence[str],
    metric_directions: Mapping[str, str],
    limits: ParetoLimits | None = None,
) -> dict[str, Any]:
    """Apply hard gates before exact soft-metric Pareto dominance."""

    limits = ParetoLimits() if limits is None else limits
    if not isinstance(limits, ParetoLimits):
        raise ParetoExplanationError("limits must be ParetoLimits")
    return _build_result(
        candidates,
        gate_outcomes,
        metric_receipts,
        required_gate_ids=required_gate_ids,
        metric_directions=metric_directions,
        limits=limits,
        validate_output=True,
    )


def validate_pareto_result(value: Mapping[str, Any]) -> None:
    """Rebuild all fronts and explanations from embedded canonical typed evidence."""

    expected_keys = {
        "schema_version",
        "limits",
        "required_gate_ids",
        "metric_registry",
        "candidates",
        "gate_outcomes",
        "metric_receipts",
        "pareto_fronts",
        "explanations",
        "display_order",
        "counts",
        "decision",
        "claims",
        "content_sha256",
    }
    _exact_keys(value, expected_keys, "Pareto result")
    unsigned = {key: child for key, child in value.items() if key != "content_sha256"}
    if value["schema_version"] != RESULT_SCHEMA or value["content_sha256"] != canonical_sha256(
        unsigned
    ):
        raise ParetoExplanationError("Pareto result canonical identity changed")
    for key in (
        "required_gate_ids",
        "metric_registry",
        "candidates",
        "gate_outcomes",
        "metric_receipts",
        "pareto_fronts",
        "explanations",
        "display_order",
    ):
        if not isinstance(value[key], list):
            raise ParetoExplanationError(f"Pareto result {key} must be an array")
    directions = {}
    for row in value["metric_registry"]:
        _exact_keys(row, {"metric_id", "direction"}, "metric registry row")
        if row["metric_id"] in directions:
            raise ParetoExplanationError("metric registry contains duplicates")
        directions[row["metric_id"]] = row["direction"]
    rebuilt = _build_result(
        tuple(CandidateArtifact.from_dict(item) for item in value["candidates"]),
        tuple(GateOutcome.from_dict(item) for item in value["gate_outcomes"]),
        tuple(MetricReceipt.from_dict(item) for item in value["metric_receipts"]),
        required_gate_ids=tuple(value["required_gate_ids"]),
        metric_directions=directions,
        limits=ParetoLimits.from_dict(value["limits"]),
        validate_output=False,
    )
    if rebuilt != value:
        raise ParetoExplanationError("Pareto result relationships or explanations changed")


def validate_pareto_replay(
    value: Mapping[str, Any],
    candidates: Sequence[CandidateArtifact],
    gate_outcomes: Sequence[GateOutcome],
    metric_receipts: Sequence[MetricReceipt],
    *,
    required_gate_ids: Sequence[str],
    metric_directions: Mapping[str, str],
    limits: ParetoLimits | None = None,
) -> None:
    """Require an exact replay from caller-supplied candidates and evidence receipts."""

    validate_pareto_result(value)
    replayed = build_pareto_explanations(
        candidates,
        gate_outcomes,
        metric_receipts,
        required_gate_ids=required_gate_ids,
        metric_directions=metric_directions,
        limits=limits,
    )
    if replayed != value:
        raise ParetoExplanationError("Pareto result is not replayable from supplied evidence")
